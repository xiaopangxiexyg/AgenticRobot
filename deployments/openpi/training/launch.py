"""Launch one additive trained model-RTC run inside an OpenPI checkout.

Example (from the OpenPI repository root):

    PYTHONPATH=/path/to/openpi:/path/to/DexTeleop \
      python -m deployments.openpi.training.launch \
        --base-config 0712-wrist-cube \
        --base-checkpoint-dir checkpoints/0712-wrist-cube/EXP/92599 \
        --artifact-template-json /path/to/trained.policy_artifact.json \
        --sampling-spec-json /path/to/sampling.json \
        --source-dataset-root /path/to/dataset \
        --camera-key left_wrist --camera-key right_wrist \
        --output-config-name 0712-wrist-cube-model-rtc-v1 \
        --exp-name tiny-preflight --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from deployments.openpi.cpu_resources import (
    add_cpu_resource_arguments,
    bootstrap_training_cpu_resources,
    require_cpu_resource_cli,
)

if __name__ == "__main__":
    bootstrap_training_cpu_resources("deployments.openpi.training.launch")

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    require_trained_model_rtc_v1_contract,
)
from dataset_contracts.policy_artifact import (
    PolicyArtifactContract,
    policy_artifact_from_metadata,
)
from dataset_interface.model_continuation import (
    ModelContinuationTrainingSamplingSpec,
)
from dataset_interface.model_continuation_training import (
    model_continuation_training_source_binding,
)
from dataset_interface.camera_semantics import (
    DatasetCameraSemanticResolver,
    load_camera_role_binding,
)
from deployments.policy_artifact import load_policy_artifact
from deployments.openpi.camera_semantics import (
    LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1,
    camera_input_routing_hash,
    policy_artifact_base_hash,
    validate_legacy_swapped_wrist_route,
)

from .adapter import (
    OpenPIModelRtcDataConfigFactory,
    OpenPIModelRtcTrainingTransform,
    OpenPICameraSemanticTransform,
    OpenPIOverlayCheckpointWeightLoader,
    build_openpi_model_rtc_training_manifest,
)


MODEL_CONTINUATION_TRAINING_RUN_METADATA_KEY_V1 = (
    "model_continuation_training_run"
)
MODEL_CONTINUATION_TRAINING_RUN_HASH_METADATA_KEY_V1 = (
    "model_continuation_training_run_hash"
)
CAMERA_SEMANTIC_TRAINING_AUDIT_METADATA_KEY_V1 = (
    "camera_semantic_training_audit"
)
CAMERA_SEMANTIC_TRAINING_AUDIT_HASH_METADATA_KEY_V1 = (
    "camera_semantic_training_audit_hash"
)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CAMERA_ROLE_REGISTRY = (
    _PROJECT_ROOT / "robot_motion/configs/rm65_camera_registry.json"
)


def _json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {
            field.name: _stable_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(
                value.items(),
                key=lambda pair: str(pair[0]),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    representation = repr(value)
    if "0x" in representation:
        raise ValueError(
            f"Cannot produce a stable training identity for {type(value).__name__}."
        )
    return {
        "python_type": (
            f"{type(value).__module__}.{type(value).__qualname__}"
        ),
        "representation": representation,
    }


def _train_config_hash(
    *,
    base_config: Any,
    output_config_name: str,
    exp_name: str,
    base_checkpoint_artifact: PolicyArtifactContract,
    artifact_template: PolicyArtifactContract,
    sampling_spec: ModelContinuationTrainingSamplingSpec,
    source_binding_hash: str,
    camera_semantic_audit_hash: str | None,
    overrides: Mapping[str, Any],
) -> str:
    payload = {
        "schema": "dexteleop_openpi_model_rtc_train_config_v1",
        "base_config_name": str(base_config.name),
        "output_config_name": output_config_name,
        "exp_name": exp_name,
        "model": _stable_value(base_config.model),
        "data_factory_type": (
            f"{type(base_config.data).__module__}."
            f"{type(base_config.data).__qualname__}"
        ),
        "data_factory": _stable_value(base_config.data),
        "lr_schedule": _stable_value(base_config.lr_schedule),
        "optimizer": _stable_value(base_config.optimizer),
        "freeze_filter": _stable_value(base_config.freeze_filter),
        "seed": int(base_config.seed),
        "base_checkpoint_model_id": base_checkpoint_artifact.model_id,
        "base_checkpoint_step": base_checkpoint_artifact.checkpoint_step,
        "artifact_template_model_id": artifact_template.model_id,
        "sampling_spec_hash": sampling_spec.semantic_content_hash,
        "source_binding_hash": source_binding_hash,
        "camera_semantic_audit_hash": camera_semantic_audit_hash,
        "overrides": _stable_value(dict(overrides)),
    }
    return "sha256:" + canonical_content_hash(
        payload,
        domain="dexteleop.openpi_model_rtc_train_config.v1",
    )


def _openpi_source_dataset_root(data_factory: Any) -> Path:
    candidates: list[tuple[str, Path]] = []
    for name in (
        "dataset_root",
        "dexteleop_v22_dataset_root",
        "dexteleop_v23_dataset_root",
    ):
        value = getattr(data_factory, name, None)
        if value:
            candidates.append(
                (name, Path(str(value)).expanduser().resolve())
            )
    if len(candidates) != 1:
        raise ValueError(
            "The first trained model-RTC launcher requires exactly one "
            "explicit DexTeleop dataset root in the OpenPI data factory."
        )
    return candidates[0][1]


def _openpi_train_main() -> tuple[Any, Path]:
    """Load the OpenPI checkout's train script independent of caller cwd."""

    import openpi

    checkout_root = Path(openpi.__file__).resolve().parents[2]
    train_path = checkout_root / "scripts" / "train.py"
    if not train_path.is_file():
        raise FileNotFoundError(
            "OpenPI training checkout cannot be resolved from "
            f"{openpi.__file__}: missing {train_path}"
        )
    spec = importlib.util.spec_from_file_location(
        "_dexteleop_openpi_train",
        train_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OpenPI train module: {train_path}")
    module = importlib.util.module_from_spec(spec)
    module_name = spec.name
    had_previous = module_name in sys.modules
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if had_previous:
            sys.modules[module_name] = previous
        else:
            sys.modules.pop(module_name, None)
        raise
    main = getattr(module, "main", None)
    if not callable(main):
        raise TypeError("OpenPI scripts/train.py does not expose main(config).")
    return main, checkout_root


def _run_openpi_train(config: Any) -> None:
    """Run OpenPI with its checkout as cwd, restoring the caller afterwards."""

    train_main, checkout_root = _openpi_train_main()
    caller_cwd = Path.cwd()
    try:
        os.chdir(checkout_root)
        train_main(config)
    finally:
        os.chdir(caller_cwd)


def _validate_checkpoint_camera_compatibility(
    base_artifact: PolicyArtifactContract,
    artifact_template: PolicyArtifactContract,
) -> None:
    base_camera = base_artifact.camera_input
    target_camera = artifact_template.camera_input
    if base_camera is None or target_camera is None:
        if base_camera is not target_camera:
            raise ValueError(
                "Base checkpoint and trained artifact camera contracts differ."
            )
        return
    if base_camera.to_dict() == target_camera.to_dict():
        return
    camera_semantics = str(
        artifact_template.metadata.get("camera_semantics") or ""
    ).strip()
    route_hash = str(
        artifact_template.metadata.get("camera_input_routing_hash") or ""
    ).strip()
    base_hash = str(
        artifact_template.metadata.get("base_policy_artifact_hash") or ""
    ).strip()
    if (
        camera_semantics != LEGACY_SWAPPED_WRIST_CAMERA_SEMANTICS_V1
        or route_hash != camera_input_routing_hash(target_camera)
        or base_hash != policy_artifact_base_hash(base_artifact)
    ):
        raise ValueError(
            "Base checkpoint and trained artifact camera contracts differ "
            "without an explicit checkpoint-compatible semantic route."
        )
    validate_legacy_swapped_wrist_route(base_camera, target_camera)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build or launch one default-off OpenPI trained model-RTC "
            "configuration without registering it in OpenPI's default config list."
        )
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--base-checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--artifact-template-json",
        type=Path,
        required=True,
    )
    parser.add_argument("--sampling-spec-json", type=Path, required=True)
    parser.add_argument("--source-dataset-root", type=Path, required=True)
    parser.add_argument(
        "--camera-key",
        action="append",
        dest="camera_keys",
        required=True,
    )
    parser.add_argument("--source-id", default=None)
    parser.add_argument(
        "--camera-role-registry",
        type=Path,
        default=_DEFAULT_CAMERA_ROLE_REGISTRY,
        help=(
            "Shared camera registry whose wrist serials define canonical "
            "physical roles. Read only; defaults to the DexTeleop RM65 registry."
        ),
    )
    parser.add_argument(
        "--disable-camera-semantic-resolution",
        action="store_true",
        help=(
            "Explicit rollback: do not resolve stored wrist keys by serial. "
            "Unsafe for mixed historical/corrected wrist datasets."
        ),
    )
    parser.add_argument("--output-config-name", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--num-train-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--keep-period", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-base-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--adapter-debug", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and print contracts without loading data/model weights.",
    )
    add_cpu_resource_arguments(parser, check_only=True)
    return parser


def build_train_config(args: argparse.Namespace) -> tuple[Any, dict[str, Any]]:
    from openpi.models import pi0
    from openpi.policies import policy as openpi_policy
    from openpi.training import config as openpi_config

    base = openpi_config.resolve_policy_metadata(
        openpi_config.get_config(str(args.base_config))
    )
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    dataset_sources = tuple(
        getattr(base.data, "dataset_sources", ()) or ()
    )
    if dataset_sources:
        raise ValueError(
            "The first OpenPI model-RTC adapter accepts one dataset source; "
            "mixed-source identity must be explicit before training."
        )
    action_pose_mode = str(
        getattr(base.data, "action_pose_mode", "absolute")
    ).strip().lower()
    if action_pose_mode != "absolute":
        raise ValueError(
            "The first trained model-RTC profile requires absolute actions."
        )

    artifact_template = load_policy_artifact(
        args.artifact_template_json.expanduser()
    )
    normalized = require_trained_model_rtc_v1_contract(
        artifact_template.model_continuation
    )
    rtc_model = dataclasses.replace(
        base.model,
        rtc_prefix_mask_conditioning=True,
    )
    descriptor = pi0.rtc_continuation_implementation_descriptor(
        mask_aware=True,
    )
    descriptor_parameters = {
        "schema": openpi_policy.RTC_IMPLEMENTATION_PARAMETERS_SCHEMA_V1,
        "model_descriptor": dict(descriptor["parameters"]),
        "effective_num_steps": 10,
        "internal_padding_rule": (
            "zero_horizon_then_existing_action_transforms"
        ),
        "final_external_projection": (
            "exact_selected_rows_after_output_transforms"
        ),
    }
    encoded_parameters = json.dumps(
        descriptor_parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    expected_parameters_hash = (
        "sha256:" + hashlib.sha256(encoded_parameters).hexdigest()
    )
    if (
        normalized.runtime.continuation_implementation_id
        != descriptor["implementation_id"]
        or normalized.runtime.continuation_implementation_parameters_hash
        != expected_parameters_hash
    ):
        raise ValueError(
            "Trained artifact template does not bind the live mask-aware "
            "OpenPI RTC implementation. Regenerate the template before training."
        )
    if artifact_template.model_id != str(args.output_config_name):
        raise ValueError(
            "artifact template model_id must equal --output-config-name."
        )
    if artifact_template.checkpoint_step is not None:
        raise ValueError(
            "artifact template checkpoint_step must be null; OpenPI binds the "
            "actual saved step when it writes each checkpoint sidecar."
        )
    if artifact_template.pose_action_chunk is None:
        raise ValueError(
            "The trained artifact template must bind a pose action chunk."
        )
    if artifact_template.pose_action_chunk.action_pose_mode != "absolute":
        raise ValueError(
            "The trained artifact template must use absolute actions."
        )
    if (
        normalized.runtime.prediction_horizon_steps
        != int(base.model.action_horizon)
    ):
        raise ValueError(
            "Base OpenPI model horizon differs from the trained artifact."
        )

    base_checkpoint_dir = args.base_checkpoint_dir.expanduser()
    base_artifact_path = base_checkpoint_dir / "policy_artifact.json"
    base_artifact = load_policy_artifact(base_artifact_path)
    if base_artifact.layout_hash != artifact_template.layout_hash:
        raise ValueError(
            "Base checkpoint and trained artifact layouts differ."
        )
    _validate_checkpoint_camera_compatibility(
        base_artifact,
        artifact_template,
    )
    sampling_spec = ModelContinuationTrainingSamplingSpec.from_dict(
        _json_object(args.sampling_spec_json)
    )
    source_binding = model_continuation_training_source_binding(
        args.source_dataset_root,
        camera_keys=tuple(args.camera_keys),
        source_id=args.source_id,
    )
    requested_source_root = args.source_dataset_root.expanduser().resolve()
    configured_source_root = _openpi_source_dataset_root(base.data)
    if configured_source_root != requested_source_root:
        raise ValueError(
            "OpenPI base data factory dataset root differs from "
            "--source-dataset-root."
        )
    artifact_camera_keys = (
        ()
        if artifact_template.camera_input is None
        else tuple(
            item.source_key
            for item in artifact_template.camera_input.inputs
        )
    )
    if (
        artifact_template.camera_input is not None
        and tuple(source_binding.camera_keys) != artifact_camera_keys
    ):
        raise ValueError(
            "--camera-key order must exactly match the artifact logical "
            "camera input order."
        )
    source_paths = tuple(artifact_template.source_datasets)
    if source_paths and str(requested_source_root) not in source_paths:
        raise ValueError(
            "Training dataset root is not declared by the artifact template."
        )

    camera_transform = None
    camera_semantic_audit = None
    camera_semantic_audit_hash = None
    if (
        not bool(args.disable_camera_semantic_resolution)
        and artifact_template.camera_input is not None
    ):
        binding = load_camera_role_binding(args.camera_role_registry)
        expected_binding_hash = str(
            artifact_template.metadata.get("camera_role_binding_hash") or ""
        ).strip()
        if expected_binding_hash and expected_binding_hash != binding.contract_hash:
            raise ValueError(
                "Artifact camera_role_binding_hash differs from the selected "
                "camera role registry."
            )
        if set(artifact_template.camera_input.source_keys).intersection(
            binding.canonical_roles
        ):
            semantic_resolver = DatasetCameraSemanticResolver(
                requested_source_root,
                binding,
            )
            camera_transform = OpenPICameraSemanticTransform(
                artifact=artifact_template,
                resolver=semantic_resolver,
            )
            camera_semantic_audit = semantic_resolver.audit_summary()
            camera_semantic_audit_hash = canonical_content_hash(
                camera_semantic_audit,
                domain="dexteleop.openpi.camera_semantic_training_audit.v1",
            )

    overrides = {
        "num_train_steps": (
            int(base.num_train_steps)
            if args.num_train_steps is None
            else int(args.num_train_steps)
        ),
        "save_interval": (
            int(base.save_interval)
            if args.save_interval is None
            else int(args.save_interval)
        ),
        "keep_period": (
            base.keep_period
            if args.keep_period is None
            else int(args.keep_period)
        ),
        "batch_size": (
            int(base.batch_size)
            if args.batch_size is None
            else int(args.batch_size)
        ),
        "num_workers": (
            int(base.num_workers)
            if args.num_workers is None
            else int(args.num_workers)
        ),
        "save_optimizer_state": bool(base.save_optimizer_state),
        "checkpoint_base_dir": (
            str(base.checkpoint_base_dir)
            if args.checkpoint_base_dir is None
            else str(args.checkpoint_base_dir.expanduser())
        ),
        "resume": bool(args.resume),
        "overwrite": bool(args.overwrite),
        "rtc_prefix_mask_conditioning": True,
    }
    train_config_hash = _train_config_hash(
        base_config=base,
        output_config_name=str(args.output_config_name),
        exp_name=str(args.exp_name),
        base_checkpoint_artifact=base_artifact,
        artifact_template=artifact_template,
        sampling_spec=sampling_spec,
        source_binding_hash=source_binding.semantic_content_hash,
        camera_semantic_audit_hash=camera_semantic_audit_hash,
        overrides=overrides,
    )
    manifest = build_openpi_model_rtc_training_manifest(
        output_model_id=str(args.output_config_name),
        artifact_template=artifact_template,
        base_checkpoint_artifact=base_artifact,
        sampling_spec=sampling_spec,
        source_binding=source_binding,
        train_config_hash=train_config_hash,
    )
    transform = OpenPIModelRtcTrainingTransform(
        artifact=artifact_template,
        sampling_spec=sampling_spec,
        source_binding=source_binding,
        debug=bool(args.adapter_debug),
    )
    policy_metadata = dict(base.policy_metadata or {})
    policy_metadata["policy_artifact"] = artifact_template.to_dict()
    policy_metadata[MODEL_CONTINUATION_TRAINING_RUN_METADATA_KEY_V1] = (
        manifest.to_dict()
    )
    policy_metadata[
        MODEL_CONTINUATION_TRAINING_RUN_HASH_METADATA_KEY_V1
    ] = manifest.semantic_content_hash
    if camera_semantic_audit is not None:
        policy_metadata[CAMERA_SEMANTIC_TRAINING_AUDIT_METADATA_KEY_V1] = (
            camera_semantic_audit
        )
        policy_metadata[
            CAMERA_SEMANTIC_TRAINING_AUDIT_HASH_METADATA_KEY_V1
        ] = camera_semantic_audit_hash
    params_path = base_checkpoint_dir / "params"
    config = dataclasses.replace(
        base,
        name=str(args.output_config_name),
        exp_name=str(args.exp_name),
        model=rtc_model,
        data=OpenPIModelRtcDataConfigFactory(
            base_factory=base.data,
            transform=transform,
            camera_transform=camera_transform,
        ),
        weight_loader=OpenPIOverlayCheckpointWeightLoader(
            base_loader=base.weight_loader,
            params_path=str(params_path),
        ),
        policy_metadata=policy_metadata,
        **overrides,
    )
    summary = {
        "schema": "dexteleop_openpi_model_rtc_launch_preflight_v1",
        "dry_run": bool(args.dry_run),
        "base_config": str(base.name),
        "base_checkpoint_dir": str(base_checkpoint_dir),
        "output_config_name": str(config.name),
        "exp_name": str(config.exp_name),
        "checkpoint_dir": str(config.checkpoint_dir),
        "artifact_template_hash": manifest.policy_artifact_template_hash,
        "base_checkpoint_identity_hash": (
            manifest.base_checkpoint_identity_hash
        ),
        "sampling_spec_hash": manifest.sampling_spec_hash,
        "source_binding_hash": source_binding.semantic_content_hash,
        "training_run_manifest_hash": manifest.semantic_content_hash,
        "train_config_hash": train_config_hash,
        "camera_semantic_resolution_enabled": camera_transform is not None,
        "camera_semantic_audit_hash": camera_semantic_audit_hash,
        "camera_semantic_audit": camera_semantic_audit,
        "rtc_prefix_mask_conditioning": bool(
            config.model.rtc_prefix_mask_conditioning
        ),
        "save_optimizer_state": bool(config.save_optimizer_state),
        "resume_optimizer_semantics": (
            "restored_optimizer_state_v1"
            if config.save_optimizer_state
            else "reset_moments_restore_counts_v1"
        ),
        "action_pose_mode": "absolute",
        "action_horizon": int(base.model.action_horizon),
        "ordinary_openpi_config_registry_modified": False,
    }
    return config, summary


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    require_cpu_resource_cli(args, parser)
    config, summary = build_train_config(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.dry_run:
        return 0

    _run_openpi_train(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

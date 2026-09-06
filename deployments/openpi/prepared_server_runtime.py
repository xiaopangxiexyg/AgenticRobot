"""Serve one prepared OpenPI checkpoint with its exact derived config.

OpenPI keeps training configs in an in-process registry.  This runtime rebuilds
the immutable inference-relevant config from the checked-in training profile,
validates it against the checkpoint-bound deployment artifact, registers it
only for this process, and then delegates to OpenPI's existing server script.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from deployments.openpi.prepared_deployment import (
    PreparedDeploymentResolution,
    validate_prepared_deployment_artifact,
)
from deployments.openpi.image_wire import (
    OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1,
    OpenPIImageWireServerConfig,
    build_openpi_image_wire_capability,
    install_openpi_image_wire_server_adapter,
)
from deployments.openpi.state_only import (
    OpenPIStateOnlyInputBridge,
    install_openpi_state_only_server_adapter,
)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import OpenPI server script: {path}")
    module = importlib.util.module_from_spec(spec)
    had_previous = name in sys.modules
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if had_previous:
            sys.modules[name] = previous
        else:
            sys.modules.pop(name, None)
        raise
    return module


def _mixing_sources(payload: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    mixing = payload.get("dataset_mixing")
    if not isinstance(mixing, Mapping):
        raise ValueError("Prepared deployment artifact lacks dataset_mixing.")
    values = mixing.get("sources")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("Prepared deployment dataset_mixing.sources is invalid.")
    if not values or any(not isinstance(value, Mapping) for value in values):
        raise ValueError("Prepared deployment requires dataset source mappings.")
    return tuple(values)


def _occlusion_schedule(value: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("front_occlusion_schedule must be a sequence.")
    schedule: list[tuple[float, float]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("front_occlusion_schedule entries must be mappings.")
        schedule.append(
            (float(item["ratio"]), float(item["probability"]))
        )
    if not schedule:
        raise ValueError("front_occlusion_schedule must not be empty.")
    return tuple(schedule)


def _prepared_weight_loader(
    profile: Mapping[str, Any],
    weight_loader: Any,
    checkout: Path,
) -> Any:
    """Restore the profile-pinned base loader used to train the checkpoint."""

    configured = str(
        profile.get("openpi_weight_loader_params_path") or ""
    ).strip()
    if not configured:
        return weight_loader
    params_path = Path(configured).expanduser()
    if not params_path.is_absolute():
        params_path = checkout / params_path
    params_path = params_path.resolve()
    if not params_path.is_dir():
        raise FileNotFoundError(
            f"Pinned OpenPI base parameter directory is missing: {params_path}."
        )
    required = (
        "_CHECKPOINT_METADATA",
        "_METADATA",
        "_sharding",
        "commit_success.txt",
        "manifest.ocdbt",
    )
    missing = [name for name in required if not (params_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Pinned OpenPI base parameter directory is incomplete: "
            f"path={params_path} missing={missing}."
        )
    if not dataclasses.is_dataclass(weight_loader) or not hasattr(
        weight_loader, "params_path"
    ):
        raise TypeError(
            "openpi_weight_loader_params_path requires a dataclass weight "
            "loader with params_path."
        )
    return dataclasses.replace(weight_loader, params_path=str(params_path))


def build_prepared_openpi_config(
    resolution: PreparedDeploymentResolution,
    *,
    training_config: Any,
    dataset_policy: Any,
) -> Any:
    """Rebuild the exact training-derived config needed for policy creation."""

    profile = resolution.profile
    artifact = resolution.deployment_artifact
    base = training_config.get_config(str(profile["openpi_base_config"]))
    augmentation_source_name = str(
        profile.get("openpi_state_augmentation_from_config") or ""
    )
    state_augmentation = base.data.state_augmentation
    if augmentation_source_name:
        augmentation_source = training_config.get_config(
            augmentation_source_name
        )
        state_augmentation = getattr(
            augmentation_source.data,
            "state_augmentation",
            None,
        )
        if state_augmentation is None:
            raise ValueError(
                "Prepared augmentation source has no state_augmentation."
            )
    state_augmentation_mode = str(
        profile.get("openpi_state_augmentation_mode", "configured_v1")
    ).strip()
    if state_augmentation_mode == "disabled_v1":
        state_augmentation = dataclasses.replace(
            state_augmentation,
            enabled=False,
            noise_std=0.0,
            dropout_probability=0.0,
            structured_enabled=False,
            arm_translation_std_m=0.0,
            arm_rotation_std_deg=0.0,
            gripper_noise_std=0.0,
            side_dropout_probability=0.0,
            arm_dropout_probability=0.0,
            end_effector_dropout_probability=0.0,
        )
    elif state_augmentation_mode != "configured_v1":
        raise ValueError(
            "openpi_state_augmentation_mode must be configured_v1 or disabled_v1."
        )
    raw_state_augmentation = resolution.deployment_payload.get(
        "state_augmentation"
    )
    if not isinstance(raw_state_augmentation, Mapping):
        raise ValueError(
            "Prepared deployment artifact lacks state_augmentation metadata."
        )
    to_dict = getattr(state_augmentation, "to_dict", None)
    if not callable(to_dict) or to_dict() != dict(raw_state_augmentation):
        raise ValueError(
            "OpenPI state-augmentation config differs from the checkpoint "
            "artifact."
        )
    if str(getattr(state_augmentation, "state_input_mode", "")) != str(
        artifact.state_input_mode
    ):
        raise ValueError(
            "OpenPI state input mode differs from the checkpoint artifact."
        )

    expected_slots = tuple(
        str(value) for value in profile["camera_slot_to_dataset_key"].keys()
    )
    if (
        int(base.model.action_horizon) != int(profile["action_horizon"])
        or tuple(base.model.image_keys) != expected_slots
        or str(base.data.action_pose_mode) != str(profile["action_pose_mode"])
        or str(base.data.last_action_pose_mode)
        != str(profile["action_pose_mode"])
        or str(base.data.tail_policy) != str(profile["tail_policy"])
    ):
        raise ValueError(
            "OpenPI base config differs from the prepared model contract."
        )

    if resolution.model_rtc_profile is None:
        source_configs = tuple(
            dataset_policy.DexTeleopV23DatasetSourceConfig(
                name=str(source["name"]),
                dataset_root=str(source["dataset_root"]),
                sample_weight=int(source.get("sample_weight", 1)),
                front_occlusion_schedule=_occlusion_schedule(
                    source["front_occlusion_schedule"]
                ),
                explicit_mask_keys=tuple(
                    str(value)
                    for value in source.get("explicit_mask_keys", ())
                ),
                front_force_black=bool(
                    source.get("front_force_black", False)
                ),
                episode_manifest=str(source["episode_manifest"]),
            )
            for source in _mixing_sources(resolution.deployment_payload)
        )
    else:
        explicit_mask_keys = tuple(
            str(value)
            for value in getattr(base.data, "explicit_mask_keys", ())
        )
        source_configs = tuple(
            dataset_policy.DexTeleopV23DatasetSourceConfig(
                name=str(source["source_dataset_id"]),
                dataset_root=str(source["source_dataset_root"]),
                sample_weight=1,
                front_occlusion_schedule=((0.0, 1.0),),
                explicit_mask_keys=explicit_mask_keys,
                front_force_black=False,
                episode_manifest=str(source["episode_manifest"]),
            )
            for source in resolution.prepared_source_manifests
        )
    if not source_configs:
        raise ValueError("Prepared deployment has no source configs.")
    first_root = Path(str(source_configs[0].dataset_root)).expanduser().resolve()
    layout_path = first_root / "meta" / "info.json"
    if not layout_path.is_file():
        raise FileNotFoundError(
            f"Prepared deployment layout metadata is missing: {layout_path}"
        )
    layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
    if not isinstance(layout_payload, Mapping):
        raise ValueError(
            f"Prepared deployment layout metadata must be an object: {layout_path}"
        )
    actual_layout_hash = str(
        layout_payload.get("policy_io_layout_hash") or ""
    )
    if actual_layout_hash != artifact.layout_hash:
        raise ValueError(
            "Prepared deployment source policy-I/O layout differs from the "
            "checkpoint artifact."
        )
    assets = dataclasses.replace(
        base.data.assets,
        assets_dir=str(resolution.checkpoint_dir / "assets"),
        asset_id=str(profile["openpi_asset_id"]),
    )
    data = dataclasses.replace(
        base.data,
        assets=assets,
        dataset_root=str(first_root),
        layout_path=str(layout_path),
        layout_hash=artifact.layout_hash,
        camera_slots=expected_slots,
        action_pose_mode=str(profile["action_pose_mode"]),
        last_action_pose_mode=str(profile["action_pose_mode"]),
        tail_policy=str(profile["tail_policy"]),
        strict_media_integrity=True,
        state_augmentation=state_augmentation,
        dataset_sources=source_configs,
        default_prompt=str(
            profile.get("default_prompt", base.data.default_prompt)
        ),
    )
    config_overrides: dict[str, Any] = {
        "name": resolution.openpi_config_name,
        "data": data,
        "exp_name": str(profile.get("default_exp_name") or base.exp_name),
        "resume": False,
        "overwrite": False,
        "num_workers": int(
            profile.get("openpi_num_workers", base.num_workers)
        ),
        "weight_loader": _prepared_weight_loader(
            profile,
            base.weight_loader,
            Path(str(profile["openpi_checkout"])).expanduser().resolve(),
        ),
    }
    if resolution.model_rtc_profile is not None:
        if not hasattr(base, "policy_metadata"):
            raise ValueError(
                "Prepared model-RTC base config has no policy_metadata field."
            )
        mask_aware = bool(
            resolution.model_rtc_profile.get(
                "rtc_prefix_mask_conditioning",
                False,
            )
        )
        config_overrides["model"] = dataclasses.replace(
            base.model,
            rtc_prefix_mask_conditioning=mask_aware,
        )
        policy_metadata = dict(getattr(base, "policy_metadata", None) or {})
        policy_metadata["policy_artifact"] = artifact.to_dict()
        config_overrides["policy_metadata"] = policy_metadata
    config = dataclasses.replace(base, **config_overrides)
    if config.name != artifact.model_id:
        raise ValueError(
            "Rebuilt OpenPI config name differs from the deployment artifact."
        )
    if str(config.data.assets.asset_id) != str(profile["openpi_asset_id"]):
        raise ValueError("Rebuilt OpenPI config asset id is incorrect.")
    if Path(str(config.data.assets.assets_dir)).resolve() != (
        resolution.checkpoint_dir / "assets"
    ).resolve():
        raise ValueError(
            "Rebuilt OpenPI config must load normalization assets from the "
            "validated checkpoint."
        )
    return config


def prepared_config_summary(
    resolution: PreparedDeploymentResolution,
    config: Any,
    *,
    image_wire_config: OpenPIImageWireServerConfig | None = None,
) -> dict[str, Any]:
    """Return an auditable, JSON-safe summary of the restored config."""

    summary = {
        "ok": True,
        "config_name": str(config.name),
        "base_config_name": str(
            resolution.profile["openpi_base_config"]
        ),
        "asset_id": str(config.data.assets.asset_id),
        "action_horizon": int(config.model.action_horizon),
        "image_keys": [str(value) for value in config.model.image_keys],
        "camera_slots": [str(value) for value in config.data.camera_slots],
        "action_pose_mode": str(config.data.action_pose_mode),
        "last_action_pose_mode": str(config.data.last_action_pose_mode),
        "tail_policy": str(config.data.tail_policy),
        "default_prompt": str(config.data.default_prompt),
        "layout_hash": str(config.data.layout_hash),
        "layout_path": str(config.data.layout_path),
        "dataset_sources": [
            {
                "name": str(source.name),
                "dataset_root": str(source.dataset_root),
                "sample_weight": int(source.sample_weight),
                "episode_manifest": str(source.episode_manifest),
            }
            for source in config.data.dataset_sources
        ],
        "profile_hash": str(resolution.binding["profile_hash"]),
        "norm_stats_hash": str(resolution.binding["norm_stats_hash"]),
        "binding_hash": str(resolution.binding["binding_hash"]),
        "model_rtc_profile_id": (
            None
            if resolution.model_rtc_profile is None
            else str(resolution.model_rtc_profile["profile_id"])
        ),
        "rtc_prefix_mask_conditioning": bool(
            getattr(config.model, "rtc_prefix_mask_conditioning", False)
        ),
    }
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        resolution.profile.get("openpi_observation_mode"),
        image_keys=tuple(str(value) for value in config.model.image_keys),
    )
    summary["observation_mode"] = (
        "state_only" if state_only_bridge is not None else "configured_cameras"
    )
    summary["external_camera_slots"] = (
        []
        if state_only_bridge is not None
        else [str(value) for value in config.data.camera_slots]
    )
    if state_only_bridge is not None:
        summary["state_only_input_contract"] = state_only_bridge.contract
    if image_wire_config is not None and image_wire_config.enabled:
        summary["image_wire_capability"] = build_openpi_image_wire_capability(
            image_wire_config,
            image_keys=tuple(str(value) for value in config.model.image_keys),
        )
    return summary


def _serve_until_stopped(server_module: Any, server_args: Any) -> int:
    """Run the delegated server and keep an operator stop traceback-free."""

    try:
        server_module.main(server_args)
    except KeyboardInterrupt:
        print("OpenPI policy server stopped by operator.", file=sys.stderr)
        return 130
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve a checkpoint-bound prepared OpenPI config."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--policy-artifact-json", type=Path, required=True)
    parser.add_argument("--prepared-training-profile", type=Path, required=True)
    parser.add_argument("--model-rtc-profile", type=Path, default=None)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8078)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--enable-rtc-continuation", action="store_true")
    parser.add_argument(
        "--image-wire-profile",
        choices=("off", "resize-raw", "jpeg"),
        default="off",
        help=(
            "Additive prepared-server image capability. off keeps the exact "
            "historical server class and metadata."
        ),
    )
    parser.add_argument("--image-wire-height", type=int, default=224)
    parser.add_argument("--image-wire-width", type=int, default=224)
    parser.add_argument("--image-wire-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Rebuild and temporarily register the prepared OpenPI config, "
            "print its bound summary, and exit without loading a model or "
            "starting a server."
        ),
    )
    args = parser.parse_args(argv)
    image_wire_config = OpenPIImageWireServerConfig(
        profile=args.image_wire_profile,
        target_height=args.image_wire_height,
        target_width=args.image_wire_width,
        jpeg_quality=args.image_wire_jpeg_quality,
    )

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    openpi_root = args.openpi_root.expanduser().resolve()
    resolution = validate_prepared_deployment_artifact(
        checkpoint_dir,
        args.prepared_training_profile,
        args.policy_artifact_json,
        model_rtc_profile_path=args.model_rtc_profile,
    )
    profile_default_prompt = str(
        resolution.profile.get("default_prompt") or ""
    ).strip()
    if (
        args.default_prompt is not None
        and profile_default_prompt
        and args.default_prompt != profile_default_prompt
    ):
        parser.error(
            "--default-prompt differs from the prepared training profile."
        )
    resolved_default_prompt = (
        args.default_prompt
        if args.default_prompt is not None
        else (profile_default_prompt or None)
    )
    if resolution.deployment_artifact.model_id != str(args.model):
        parser.error("--model differs from the prepared deployment artifact.")
    profile_openpi_root = Path(
        str(resolution.profile["openpi_checkout"])
    ).expanduser().resolve()
    if profile_openpi_root != openpi_root:
        parser.error(
            "--openpi-root differs from the prepared training profile."
        )

    import openpi.training.config as training_config
    from openpi.policies import dexteleop_v23_policy

    config = build_prepared_openpi_config(
        resolution,
        training_config=training_config,
        dataset_policy=dexteleop_v23_policy,
    )
    state_only_bridge = OpenPIStateOnlyInputBridge.from_profile_value(
        resolution.profile.get("openpi_observation_mode"),
        image_keys=tuple(str(value) for value in config.model.image_keys),
    )
    if state_only_bridge is not None and image_wire_config.enabled:
        raise ValueError(
            "State-only prepared deployment cannot enable image-wire transport."
        )
    if image_wire_config.enabled:
        from openpi.models import model as openpi_model

        model_resolution = tuple(int(value) for value in openpi_model.IMAGE_RESOLUTION)
        requested_resolution = (
            image_wire_config.target_height,
            image_wire_config.target_width,
        )
        if requested_resolution != model_resolution:
            raise ValueError(
                "Prepared image-wire target differs from the loaded OpenPI "
                f"model resolution: {requested_resolution!r} != "
                f"{model_resolution!r}."
            )
        advertised_capability = build_openpi_image_wire_capability(
            image_wire_config,
            image_keys=tuple(str(value) for value in config.model.image_keys),
        )
        training_alignment = resolution.checkpoint_payload.get(
            OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1
        )
        if training_alignment is not None:
            if not isinstance(training_alignment, Mapping) or (
                training_alignment.get("server_capability")
                != advertised_capability
            ):
                raise ValueError(
                    "Prepared image-wire server capability differs from the "
                    "checkpoint training alignment contract."
                )
    registry = training_config._CONFIGS_DICT
    previous = registry.get(config.name)
    if previous is not None and previous != config:
        raise ValueError(
            "OpenPI config registry already contains a different config with "
            f"name {config.name!r}."
        )
    registry[config.name] = config
    try:
        if args.validate_only:
            print(
                json.dumps(
                    prepared_config_summary(
                        resolution,
                        config,
                        image_wire_config=image_wire_config,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        server_module = _load_module(
            openpi_root / "scripts" / "serve_dexteleop_policy.py",
            "_dexteleop_prepared_openpi_server",
        )
        server_args = server_module.Args(
            model=str(args.model),
            registry=args.registry.expanduser().resolve(),
            host=str(args.host),
            port=int(args.port),
            checkpoint_dir=None,
            default_prompt=resolved_default_prompt,
            record=bool(args.record),
            enable_rtc_continuation=bool(args.enable_rtc_continuation),
        )
        with install_openpi_image_wire_server_adapter(
            server_module.websocket_policy_server,
            image_wire_config,
        ), install_openpi_state_only_server_adapter(
            server_module.websocket_policy_server,
            state_only_bridge,
        ):
            return _serve_until_stopped(server_module, server_args)
    finally:
        if previous is None:
            registry.pop(config.name, None)
        else:
            registry[config.name] = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_prepared_openpi_config",
    "main",
    "prepared_config_summary",
]

"""Launch the pinned or explicitly selected OpenPI server from a checkpoint.

The OpenPI server currently selects its training config through a registry.
This wrapper derives the registry entry from the checkpoint's authoritative
policy artifact and writes only a disposable file under the system temp
directory.  It does not modify the OpenPI checkout or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

from dataset_contracts.normalization import (
    NormalizationContract,
    NormalizationContractError,
)
from deployments.openpi.camera_semantics import (
    validate_declared_camera_semantics,
    validate_legacy_swapped_wrist_artifact,
)
from deployments.openpi.cpu_resources import (
    add_cpu_resource_arguments,
)
from deployments.openpi.cpu_resources import (
    cpu_exclusion_plan as _server_cpu_exclusion_plan,
)
from deployments.openpi.cpu_resources import (
    exclude_cpus_before_exec as _exclude_server_cpus_before_exec,
)
from deployments.openpi.cpu_resources import (
    native_thread_environment as _native_thread_environment,
)
from deployments.openpi.image_wire import OpenPIImageWireServerConfig
from deployments.openpi.prepared_deployment import (
    prepare_checkpoint_deployment_artifact,
    validate_prepared_deployment_artifact,
)
from deployments.policy_artifact import load_policy_artifact


def _openpi_root_from_checkpoint(checkpoint_dir: Path) -> Path | None:
    for parent in (checkpoint_dir, *checkpoint_dir.parents):
        if parent.name == "checkpoints":
            return parent.parent
    return None


def _bundled_openpi_root() -> Path:
    """Return AgenticRobot's pinned OpenPI submodule checkout location."""

    return Path(__file__).resolve().parents[2] / "model" / "openpi"


def _is_openpi_checkout(root: Path) -> bool:
    return all(
        (
            (root / "src" / "openpi").is_dir(),
            (root / "scripts" / "serve_dexteleop_policy.py").is_file(),
            (root / "configs" / "dexteleop_policy_registry.json").is_file(),
        )
    )


def _resolve_openpi_root(
    checkpoint_dir: Path,
    explicit_root: Path | None,
) -> tuple[Path | None, str | None]:
    """Resolve OpenPI without changing existing explicit/inferred precedence."""

    if explicit_root is not None:
        return explicit_root.expanduser().resolve(), "explicit"
    inferred_root = _openpi_root_from_checkpoint(checkpoint_dir)
    if inferred_root is not None and _is_openpi_checkout(inferred_root):
        return inferred_root, "checkpoint"
    bundled_root = _bundled_openpi_root()
    if _is_openpi_checkout(bundled_root):
        return bundled_root.resolve(), "bundled_submodule"
    return None, None


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "policy"


def _pythonpath_with_project(
    project_root: Path,
    current: str | None,
    *,
    openpi_root: Path | None = None,
) -> str:
    """Build an exec-time path with the pinned OpenPI sources first.

    The OpenPI virtual environment may expose a different checkout through
    site configuration (for example ``/home/user/zc/openpi/src``).  A
    prepared profile, however, binds the server to one exact checkout.  Put
    that checkout's source roots before the inherited path so the registry
    and dataclass definitions used by the exec'd server are the same ones
    used to prepare the checkpoint.  The project root remains first because
    the prepared server runtime belongs to AgenticRobot.
    """

    entries = [str(project_root.resolve())]
    if openpi_root is not None:
        resolved_openpi_root = openpi_root.expanduser().resolve()
        for source_root in (
            resolved_openpi_root / "src",
            resolved_openpi_root / "packages" / "openpi-client" / "src",
        ):
            if source_root.is_dir():
                entries.append(str(source_root))
    entries.extend(
        value
        for value in str(current or "").split(os.pathsep)
        if value and value not in entries
    )
    return os.pathsep.join(entries)


def build_runtime_registry_entry(
    checkpoint_dir: Path,
    *,
    policy_artifact_json: Path | None = None,
    prepared_training_profile: Path | None = None,
    model_rtc_profile: Path | None = None,
    config_name: str | None = None,
    default_prompt: str | None = None,
    action_horizon: int = 50,
) -> tuple[str, dict[str, object]]:
    if model_rtc_profile is not None and prepared_training_profile is None:
        raise ValueError(
            "model_rtc_profile requires prepared_training_profile."
        )
    checkpoint_artifact = load_policy_artifact(
        checkpoint_dir / "policy_artifact.json"
    )
    artifact = checkpoint_artifact
    if policy_artifact_json is not None:
        artifact = load_policy_artifact(policy_artifact_json)
        if prepared_training_profile is not None:
            validate_prepared_deployment_artifact(
                checkpoint_dir,
                prepared_training_profile,
                policy_artifact_json,
                model_rtc_profile_path=model_rtc_profile,
            )
        else:
            validate_legacy_swapped_wrist_artifact(
                checkpoint_artifact,
                artifact,
            )
    elif prepared_training_profile is not None:
        raise ValueError(
            "prepared_training_profile requires a generated policy artifact."
        )
    else:
        validate_declared_camera_semantics(artifact)
    model_id = artifact.model_id
    if model_id is None:
        raise ValueError("policy artifact must declare model_id for server launch")
    pose_chunk = artifact.pose_action_chunk
    resolved_action_horizon = (
        int(action_horizon)
        if pose_chunk is None
        else int(pose_chunk.action_horizon)
    )
    camera_slots = (
        []
        if artifact.camera_input is None
        else list(artifact.camera_input.server_camera_slots)
    )
    entry: dict[str, object] = {
        "config": config_name or model_id,
        "checkpoint_dir": str(checkpoint_dir),
        "deploy_schema_version": artifact.deploy_schema_version,
        "action_key": artifact.action_key,
        "action_discrete_key": artifact.action_discrete_key,
        "action_dim": artifact.policy_io_layout.action_dim,
        "action_discrete_dim": artifact.policy_io_layout.action_discrete_dim,
        "state_dim": artifact.state_input_dim,
        "state_discrete_dim": artifact.policy_io_layout.state_discrete_dim,
        "action_horizon": resolved_action_horizon,
        "active_sides": ["left", "right"],
        "camera_slots": camera_slots,
    }
    raw_normalization = artifact.metadata.get("normalization_contract")
    if raw_normalization is not None:
        try:
            normalization = NormalizationContract.from_optional_mapping(
                raw_normalization,
                allow_legacy=True,
            )
        except NormalizationContractError as exc:
            raise ValueError(
                "policy artifact metadata.normalization_contract is invalid: " + str(exc)
            ) from exc
        entry["normalization_contract_status"] = (
            "explicit" if normalization is not None else "legacy_opaque"
        )
        if normalization is not None:
            entry["normalization_contract"] = normalization.to_dict()
            entry["normalization_hash"] = normalization.contract_hash
    else:
        # Absence is not identity: older OpenPI artifacts simply did not
        # publish the server-side stats owner/hash.  Keep that distinction in
        # the disposable registry so callers cannot mistake an omitted field
        # for a verified no-op contract.
        entry["normalization_contract_status"] = "missing"
    if default_prompt is not None:
        entry["default_prompt"] = default_prompt
    return model_id, entry


def write_runtime_registry(
    *,
    base_registry: Path,
    checkpoint_dir: Path,
    policy_artifact_json: Path | None = None,
    prepared_training_profile: Path | None = None,
    model_rtc_profile: Path | None = None,
    config_name: str | None = None,
    default_prompt: str | None = None,
    action_horizon: int = 50,
) -> tuple[str, Path]:
    payload = json.loads(base_registry.read_text(encoding="utf-8"))
    models = payload.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"registry 'models' must be an object: {base_registry}")
    model_id, entry = build_runtime_registry_entry(
        checkpoint_dir,
        policy_artifact_json=policy_artifact_json,
        prepared_training_profile=prepared_training_profile,
        model_rtc_profile=model_rtc_profile,
        config_name=config_name,
        default_prompt=default_prompt,
        action_horizon=action_horizon,
    )
    models[model_id] = entry
    step = checkpoint_dir.name
    target = Path(tempfile.gettempdir()) / (
        f"dexteleop_openpi_{_safe_filename(model_id)}_{_safe_filename(step)}.json"
    )
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return model_id, target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch scripts/serve_dexteleop_policy.py from a checkpoint directory."
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--policy-artifact-json",
        type=Path,
        default=None,
        help=(
            "Optional versioned compatibility artifact bound to the immutable "
            "checkpoint artifact. The checkpoint params remain unchanged."
        ),
    )
    parser.add_argument(
        "--prepared-training-profile",
        type=Path,
        default=None,
        help=(
            "Checked-in prepared training profile used to reconstruct the exact "
            "OpenPI config and bind checkpoint norm-stats. This is mutually "
            "exclusive with --policy-artifact-json and --config."
        ),
    )
    parser.add_argument(
        "--model-rtc-profile",
        type=Path,
        default=None,
        help=(
            "Optional trained model-RTC profile bound together with "
            "--prepared-training-profile and the checkpoint receipt."
        ),
    )
    parser.add_argument(
        "--openpi-root",
        type=Path,
        default=None,
        help=(
            "Training-profile OpenPI checkout. Otherwise infer it from a "
            "checkpoints/ ancestor, then fall back to the initialized "
            "model/openpi submodule."
        ),
    )
    parser.add_argument(
        "--rtc-runtime-openpi-root",
        type=Path,
        default=None,
        help=(
            "Explicit deployment runtime checkout for the 8-step trained "
            "model-RTC derivation. It must resolve to this AgenticRobot "
            "snapshot's pinned model/openpi submodule; the training checkout "
            "and prepared profile remain unchanged."
        ),
    )
    parser.add_argument("--config", default=None, help="Training config override.")
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8078)
    add_cpu_resource_arguments(parser)
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--enable-rtc-continuation",
        action="store_true",
        help=(
            "Explicitly enable the upstream OpenPI conditioned inference "
            "operation. Disabled by default."
        ),
    )
    parser.add_argument(
        "--rtc-inference-steps",
        type=int,
        choices=(8, 10),
        default=10,
        help=(
            "Versioned conditioned-RTC sampler budget. The trained/default "
            "contract is 10; 8 is an explicit latency deployment derivation."
        ),
    )
    parser.add_argument(
        "--image-wire-profile",
        choices=("off", "resize-raw", "jpeg"),
        default="off",
        help=(
            "Optional prepared-server image codec capability. off preserves "
            "the historical server command and metadata."
        ),
    )
    parser.add_argument("--image-wire-height", type=int, default=224)
    parser.add_argument("--image-wire-width", type=int, default=224)
    parser.add_argument("--image-wire-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write the temporary registry and print the server command without starting it.",
    )
    parser.add_argument(
        "--prepared-config-check-only",
        action="store_true",
        help=(
            "Rebuild and temporarily register a prepared OpenPI config, then "
            "exit without loading model weights or starting the server."
        ),
    )
    args = parser.parse_args(argv)
    try:
        cpu_plan = _server_cpu_exclusion_plan(args.exclude_cpu, args.exclude_llc_of_cpu)
    except RuntimeError as exc:
        parser.error(str(exc))
    try:
        image_wire_config = OpenPIImageWireServerConfig(
            profile=args.image_wire_profile,
            target_height=args.image_wire_height,
            target_width=args.image_wire_width,
            jpeg_quality=args.image_wire_jpeg_quality,
        )
    except ValueError as exc:
        parser.error(str(exc))

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    artifact_path = checkpoint_dir / "policy_artifact.json"
    if not checkpoint_dir.is_dir():
        parser.error(f"checkpoint directory does not exist: {checkpoint_dir}")
    if not artifact_path.is_file():
        parser.error(f"checkpoint has no policy_artifact.json: {checkpoint_dir}")
    policy_artifact_json = (
        None
        if args.policy_artifact_json is None
        else args.policy_artifact_json.expanduser().resolve()
    )
    if policy_artifact_json is not None and not policy_artifact_json.is_file():
        parser.error(
            "compatibility policy artifact does not exist: "
            f"{policy_artifact_json}"
        )
    prepared_training_profile = (
        None
        if args.prepared_training_profile is None
        else args.prepared_training_profile.expanduser().resolve()
    )
    model_rtc_profile = (
        None
        if args.model_rtc_profile is None
        else args.model_rtc_profile.expanduser().resolve()
    )
    if model_rtc_profile is not None and prepared_training_profile is None:
        parser.error(
            "--model-rtc-profile requires --prepared-training-profile."
        )
    if model_rtc_profile is not None and not model_rtc_profile.is_file():
        parser.error(
            f"model-RTC profile does not exist: {model_rtc_profile}"
        )
    if (
        prepared_training_profile is not None
        and policy_artifact_json is not None
    ):
        parser.error(
            "--prepared-training-profile and --policy-artifact-json are "
            "mutually exclusive."
        )
    if prepared_training_profile is not None and args.config is not None:
        parser.error(
            "--prepared-training-profile owns the exact config; omit --config."
        )
    if args.prepared_config_check_only and prepared_training_profile is None:
        parser.error(
            "--prepared-config-check-only requires "
            "--prepared-training-profile."
        )
    if args.prepared_config_check_only and args.dry_run:
        parser.error(
            "--prepared-config-check-only and --dry-run are mutually exclusive."
        )
    if not args.enable_rtc_continuation and args.rtc_inference_steps != 10:
        parser.error(
            "--rtc-inference-steps 8 requires --enable-rtc-continuation."
        )
    if args.rtc_inference_steps == 10 and args.rtc_runtime_openpi_root is not None:
        parser.error(
            "--rtc-runtime-openpi-root is only valid with "
            "--rtc-inference-steps 8."
        )
    prepared_resolution = None
    if prepared_training_profile is not None:
        if not prepared_training_profile.is_file():
            parser.error(
                "prepared training profile does not exist: "
                f"{prepared_training_profile}"
            )
        prepared_resolution = prepare_checkpoint_deployment_artifact(
            checkpoint_dir,
            prepared_training_profile,
            model_rtc_profile_path=model_rtc_profile,
        )
        policy_artifact_json = prepared_resolution.deployment_artifact_path
    if image_wire_config.enabled and prepared_resolution is None:
        parser.error(
            "Optimized --image-wire-profile requires "
            "--prepared-training-profile so the adapter remains process-local."
        )

    openpi_root, openpi_root_source = _resolve_openpi_root(
        checkpoint_dir,
        args.openpi_root,
    )
    if openpi_root is None:
        parser.error(
            "cannot infer OpenPI root and the bundled OpenPI submodule is not "
            "initialized; run 'git submodule update --init model/openpi' or "
            "pass --openpi-root"
        )
    if prepared_resolution is not None:
        profile_openpi_root = Path(
            str(prepared_resolution.profile["openpi_checkout"])
        ).expanduser().resolve()
        if profile_openpi_root != openpi_root:
            parser.error(
                "--openpi-root differs from the prepared training profile: "
                f"{openpi_root} != {profile_openpi_root}"
            )
    if (
        args.rtc_inference_steps == 8
        and model_rtc_profile is not None
        and args.rtc_runtime_openpi_root is None
    ):
        parser.error(
            "8-step prepared model-RTC requires --rtc-runtime-openpi-root."
        )
    runtime_openpi_root = openpi_root
    if args.rtc_runtime_openpi_root is not None:
        if prepared_resolution is None or model_rtc_profile is None:
            parser.error(
                "--rtc-runtime-openpi-root requires a prepared trained "
                "model-RTC checkpoint."
            )
        runtime_openpi_root = args.rtc_runtime_openpi_root.expanduser().resolve()
        bundled_openpi_root = _bundled_openpi_root().resolve()
        if runtime_openpi_root != bundled_openpi_root:
            parser.error(
                "--rtc-runtime-openpi-root must resolve to the current "
                "AgenticRobot snapshot's pinned model/openpi submodule: "
                f"{runtime_openpi_root} != {bundled_openpi_root}"
            )
    server_script = runtime_openpi_root / "scripts/serve_dexteleop_policy.py"
    base_registry = runtime_openpi_root / "configs/dexteleop_policy_registry.json"
    if not server_script.is_file():
        parser.error(f"OpenPI server script does not exist: {server_script}")
    if not base_registry.is_file():
        parser.error(f"OpenPI registry does not exist: {base_registry}")

    model_id, runtime_registry = write_runtime_registry(
        base_registry=base_registry,
        checkpoint_dir=checkpoint_dir,
        policy_artifact_json=policy_artifact_json,
        prepared_training_profile=prepared_training_profile,
        model_rtc_profile=model_rtc_profile,
        config_name=(
            prepared_resolution.openpi_config_name
            if prepared_resolution is not None
            else args.config
        ),
        default_prompt=args.default_prompt,
        action_horizon=args.action_horizon,
    )
    if prepared_resolution is None:
        command = [
            sys.executable,
            str(server_script),
            "--model",
            model_id,
            "--registry",
            str(runtime_registry),
            "--host",
            str(args.host),
            "--port",
            str(args.port),
        ]
    else:
        command = [
            sys.executable,
            "-m",
            "deployments.openpi.prepared_server_runtime",
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--policy-artifact-json",
            str(prepared_resolution.deployment_artifact_path),
            "--prepared-training-profile",
            str(prepared_training_profile),
            "--openpi-root",
            str(runtime_openpi_root),
            "--training-openpi-root",
            str(openpi_root),
            "--model",
            model_id,
            "--registry",
            str(runtime_registry),
            "--host",
            str(args.host),
            "--port",
            str(args.port),
        ]
        if model_rtc_profile is not None:
            command.extend(("--model-rtc-profile", str(model_rtc_profile)))
    if args.default_prompt is not None:
        command.extend(("--default-prompt", args.default_prompt))
    if args.record:
        command.append("--record")
    if args.enable_rtc_continuation:
        command.append("--enable-rtc-continuation")
    if args.rtc_inference_steps != 10:
        command.extend(
            ("--rtc-inference-steps", str(args.rtc_inference_steps))
        )
    if image_wire_config.enabled:
        command.extend(
            (
                "--image-wire-profile",
                image_wire_config.profile,
                "--image-wire-height",
                str(image_wire_config.target_height),
                "--image-wire-width",
                str(image_wire_config.target_width),
                "--image-wire-jpeg-quality",
                str(image_wire_config.jpeg_quality),
            )
        )
    if args.prepared_config_check_only:
        command.append("--validate-only")
    project_root = Path(__file__).resolve().parents[2]
    runtime_pythonpath = _pythonpath_with_project(
        project_root,
        os.environ.get("PYTHONPATH"),
        openpi_root=runtime_openpi_root,
    )
    native_thread_environment = _native_thread_environment(
        args.native_thread_count
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_id": model_id,
                    "openpi_root": str(openpi_root),
                    "openpi_root_source": openpi_root_source,
                    "runtime_openpi_root": str(runtime_openpi_root),
                    "command": command,
                    "environment": {
                        "PYTHONPATH": runtime_pythonpath,
                        **native_thread_environment,
                    },
                    "excluded_cpu_ids": (
                        [] if cpu_plan is None else cpu_plan["excluded_cpu_ids"]
                    ),
                    "cpu_resource_plan": cpu_plan,
                    "policy_artifact_json": (
                        str(artifact_path)
                        if policy_artifact_json is None
                        else str(policy_artifact_json)
                    ),
                    "prepared_deployment_binding": (
                        None
                        if prepared_resolution is None
                        else dict(prepared_resolution.binding)
                    ),
                    "norm_stats_path": (
                        None
                        if prepared_resolution is None
                        else str(prepared_resolution.norm_stats_path)
                    ),
                    "model_rtc_profile": (
                        None
                        if model_rtc_profile is None
                        else str(model_rtc_profile)
                    ),
                    **(
                        {
                            "image_wire": {
                                "profile": image_wire_config.profile,
                                "target_shape": [
                                    image_wire_config.target_height,
                                    image_wire_config.target_width,
                                    3,
                                ],
                                "jpeg_quality": image_wire_config.jpeg_quality,
                            }
                        }
                        if image_wire_config.enabled
                        else {}
                    ),
                },
                indent=2,
            )
        )
        return 0

    if cpu_plan is not None:
        # Reject cpuset changes during model/config preparation; never execute
        # with a mask different from the one shown in the resolved plan.
        if sorted(os.sched_getaffinity(0)) != cpu_plan["allowed_cpu_ids_before"]:
            raise RuntimeError("launcher CPU affinity changed after resource planning")
        excluded_cpu_ids = tuple(cpu_plan["excluded_cpu_ids"])
        reservation = _exclude_server_cpus_before_exec(excluded_cpu_ids)
        print(
            "[OpenPICPU] READY: server process excludes control CPUs "
            f"{list(excluded_cpu_ids)}; migrated_native_threads="
            f"{reservation['native_threads_migrated']} inherited_after_exec=true; "
            f"resource_plan={json.dumps(cpu_plan, sort_keys=True)}",
            flush=True,
        )
    if native_thread_environment:
        os.environ.update(native_thread_environment)
        print(
            "[OpenPICPU] READY: native math thread pools capped at "
            f"{args.native_thread_count}; inherited_after_exec=true.",
            flush=True,
        )
    os.environ["PYTHONPATH"] = runtime_pythonpath
    os.chdir(runtime_openpi_root)
    os.execv(sys.executable, command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deployments.client import load_policy_artifact
from deployments.openpi.client.preflight import preflight_policy_artifact
from robot_motion.deploy.async_producer import (
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEPLOYMENT_PROFILES = ("rm65_bimanual_dmtac",)


def _rm65_bimanual_dmtac_args(*, rtc_real_shadow: bool = False) -> list[str]:
    """Return the audited lab defaults without granting dispatch authority."""

    args = [
        "--active-sides",
        "both",
        "--end-effector-sides",
        "both",
        "--arm-execution-profile",
        "impedance_40hz",
        "--left-arm-ip",
        "192.168.101.21",
        "--right-arm-ip",
        "192.168.1.20",
        "--left-tool-profile",
        str(
            _PROJECT_ROOT
            / "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"
        ),
        "--right-tool-profile",
        str(
            _PROJECT_ROOT
            / "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json"
        ),
        "--enable-hands",
        "--hand-backend",
        "dmtac-gripper",
        "--dmtac-config",
        "latest",
        "--require-dmtac-config",
        "--camera-mode",
        "multi",
        "--camera-config",
        str(_PROJECT_ROOT / "teleop_runtime/configs/devices/realsense.json"),
        "--require-camera-frame",
        "--require-ready-observation",
        "--action-queue-policy",
        "replace",
        "--min-action-queue",
        "0",
        "--init-source",
        "current",
    ]
    if bool(rtc_real_shadow):
        args.extend(
            [
                "--auto-start",
                "--no-wait-for-ready-confirm",
                "--dmtac-no-tactile",
                "--camera-startup-hardware-reset",
                "--camera-start-stagger-sec",
                "0.5",
            ]
        )
    else:
        args.extend(
            [
                "--wait-for-ready-confirm",
                "--policy-qpos-init-dataset",
                str(_PROJECT_ROOT / "datasets/sorted_v22/bimanual-cube-new-success"),
                "--policy-qpos-init-episode",
                "0",
                "--policy-qpos-init-frame",
                "0",
                "--policy-qpos-init-source",
                "auto",
                "--policy-qpos-init-collision-check",
            ]
        )
    return args


def _artifact_path_from_checkpoint(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.expanduser() / "policy_artifact.json"


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _image_key_map_json(value: str) -> dict[str, str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError("--image-key-map-json must be a JSON object.") from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError("--image-key-map-json must be a JSON object.")
    normalized: dict[str, str] = {}
    for source_key, model_input_key in payload.items():
        source = str(source_key).strip()
        destination = str(model_input_key).strip()
        if not source or not destination:
            raise argparse.ArgumentTypeError(
                "--image-key-map-json keys and values must be non-empty strings."
            )
        normalized[source] = destination
    return normalized


def _normalization_contract_json(value: str) -> dict[str, Any]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            "--normalization-contract-json must be a JSON object."
        ) from exc
    if not isinstance(payload, dict):
        raise argparse.ArgumentTypeError(
            "--normalization-contract-json must be a JSON object."
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Launch the OpenPI client through the real teleop_runtime backend.",
        add_help=True,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--model", "--expected-model-id", dest="expected_model_id", default=None)
    parser.add_argument("--checkpoint-step", "--expected-checkpoint-step", dest="expected_checkpoint_step", default=None)
    parser.add_argument("--action-dim", type=int, default=26)
    parser.add_argument("--action-key", default="actions")
    artifact_group = parser.add_mutually_exclusive_group()
    artifact_group.add_argument(
        "--policy-artifact-json",
        type=Path,
        default=None,
        help=(
            "Checkpoint dexteleop_policy_artifact_v1 sidecar. Its model I/O, "
            "pose-training provenance, state/action, and camera bindings are "
            "passed to both the OpenPI client and the shared deployment runtime."
        ),
    )
    artifact_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help=(
            "Checkpoint directory containing policy_artifact.json. This is a "
            "short alias for --policy-artifact-json CHECKPOINT/policy_artifact.json."
        ),
    )
    parser.add_argument(
        "--deployment-profile",
        choices=_DEPLOYMENT_PROFILES,
        default=None,
        help=(
            "Inject the checked-in real-runtime arguments for the selected lab "
            "profile. Explicit trailing runtime flags override profile values. "
            "The profile never enables --send-actions."
        ),
    )
    parser.add_argument(
        "--contract-check-only",
        action="store_true",
        help=(
            "Validate the policy artifact, relative/absolute pose binding, and "
            "OpenPI tensor contract, print JSON, and exit without connecting to "
            "the server, cameras, hands, or arms. Requires --checkpoint-dir or "
            "--policy-artifact-json."
        ),
    )
    parser.add_argument("--required-camera-slot", action="append", dest="required_camera_slots", default=None)
    parser.add_argument("--required-image-key", action="append", dest="required_image_keys", default=None)
    parser.add_argument(
        "--image-key-map-json",
        type=_image_key_map_json,
        default=None,
        help=(
            "Explicit DexTeleop logical-camera to OpenPI input-key map, for example "
            "'{\"left_wrist\":\"observation.images.base_0_rgb\"}'. "
            "Normally supplied by --policy-artifact-json."
        ),
    )
    parser.add_argument(
        "--normalization-contract-json",
        type=_normalization_contract_json,
        default=None,
        help=(
            "Explicit shared numeric-normalization contract as JSON. The OpenPI "
            "client passes raw values through and binds this contract/hash; it "
            "does not execute a client-owned transform."
        ),
    )
    parser.add_argument("--hand-layout", default="q6_slot0_gripper")
    parser.add_argument(
        "--rtc-profile",
        choices=(
            "off",
            RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
            RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        ),
        default=None,
        help=(
            "Optional default-off shared deployment RTC profile. The OpenPI raw-RTC "
            "profile keeps the existing final-chunk model/server interface and only "
            "moves inference into the client worker/timed plan."
        ),
    )
    parser.add_argument(
        "--rtc-real-shadow",
        action="store_true",
        help=(
            "Explicit live observation/transport no-action shadow for the raw RTC "
            "profile. The checked-in real profile omits qpos-init authority in this mode."
        ),
    )
    parser.add_argument(
        "--rtc-normalization-hash",
        default=None,
        help=(
            "Explicit normalization/statistics identity for raw-RTC. It is required "
            "when the raw profile is enabled and is never inferred from a checkpoint name."
        ),
    )
    parser.add_argument(
        "--rtc-frame-contract-hash",
        default=None,
        help=(
            "Optional explicit raw-RTC frame identity. The resolved runtime pose-frame "
            "contract is authoritative when available; use this only for an explicitly "
            "bound legacy/identity artifact."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write deploy action/pose debug JSONL through teleop_runtime.deploy.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="Directory for deployment debug JSONL.",
    )
    parser.add_argument(
        "--rtc-debug-level",
        choices=("off", "summary", "trace"),
        default=None,
        help="RTC debug detail; requires --debug. Default for RTC is summary.",
    )
    parser.add_argument(
        "--rtc-debug-queue-capacity",
        type=int,
        default=None,
        help="Bounded non-blocking RTC debug event queue capacity.",
    )
    parser.add_argument(
        "--runtime-help",
        action="store_true",
        help="Show the delegated real-runtime mode, rate, safety, and hardware options.",
    )
    args, teleop_args = parser.parse_known_args(argv)

    if args.runtime_help:
        from teleop_runtime.deploy.run_policy import main as run_real_policy

        return run_real_policy(["--help"])

    if args.rtc_real_shadow and args.rtc_profile != RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        parser.error(
            "--rtc-real-shadow requires "
            f"--rtc-profile {RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1}."
        )

    if (
        args.rtc_profile == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1
        and not _optional_text(args.rtc_normalization_hash)
    ):
        parser.error(
            f"{RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1} requires --rtc-normalization-hash."
        )

    artifact_path = args.policy_artifact_json
    if args.checkpoint_dir is not None:
        artifact_path = _artifact_path_from_checkpoint(args.checkpoint_dir)
    if artifact_path is not None and not artifact_path.is_file():
        parser.error(f"policy artifact does not exist or is not a file: {artifact_path}")
    policy_artifact = None if artifact_path is None else load_policy_artifact(artifact_path)
    if args.contract_check_only:
        if policy_artifact is None:
            parser.error(
                "--contract-check-only requires --checkpoint-dir or "
                "--policy-artifact-json."
            )
        print(
            json.dumps(
                preflight_policy_artifact(policy_artifact).to_dict(),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    policy_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": int(args.port),
        "api_key": _optional_text(args.api_key),
        "prompt": _optional_text(args.prompt),
        "expected_model_id": _optional_text(args.expected_model_id),
        "expected_checkpoint_step": _optional_text(args.expected_checkpoint_step),
        "action_dim": int(args.action_dim),
        "action_key": str(args.action_key),
        "required_camera_slots": (
            None
            if args.required_camera_slots is None
            else tuple(args.required_camera_slots)
        ),
        "required_image_keys": (
            None if args.required_image_keys is None else tuple(args.required_image_keys)
        ),
        "image_key_map": args.image_key_map_json,
        "hand_layout": _optional_text(args.hand_layout),
        "policy_artifact": (
            None if policy_artifact is None else policy_artifact.to_dict()
        ),
        "normalization_contract": args.normalization_contract_json,
    }
    policy_kwargs = {key: value for key, value in policy_kwargs.items() if value is not None}

    from teleop_runtime.deploy.run_policy import main as run_real_policy

    runtime_args: list[str] = []
    if args.deployment_profile == "rm65_bimanual_dmtac":
        runtime_args.extend(
            _rm65_bimanual_dmtac_args(rtc_real_shadow=bool(args.rtc_real_shadow))
        )
    runtime_args.extend(teleop_args)
    if artifact_path is not None:
        runtime_args.extend(("--policy-artifact-json", str(artifact_path)))
    if args.debug:
        runtime_args.append("--debug")
    if args.debug_dir is not None:
        runtime_args.extend(("--debug-dir", str(args.debug_dir)))
    if args.rtc_debug_level is not None:
        runtime_args.extend(("--rtc-debug-level", str(args.rtc_debug_level)))
    if args.rtc_debug_queue_capacity is not None:
        runtime_args.extend(
            (
                "--rtc-debug-queue-capacity",
                str(args.rtc_debug_queue_capacity),
            )
        )

    if args.rtc_profile is not None:
        runtime_args.extend(("--rtc-profile", str(args.rtc_profile)))
    if args.rtc_real_shadow:
        runtime_args.append("--rtc-real-shadow")

    rtc_model_adapter_factory = None
    if args.rtc_profile == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1:
        normalization_hash = _optional_text(args.rtc_normalization_hash)
        explicit_frame_hash = _optional_text(args.rtc_frame_contract_hash)

        def _create_openpi_raw_rtc_adapter(**factory_kwargs: Any):
            policy = factory_kwargs["policy"]
            session_contract = factory_kwargs["deploy_session_contract"]
            runtime_args = factory_kwargs["args"]
            artifact = factory_kwargs.get("policy_artifact")
            frame_contract = getattr(session_contract, "runtime_pose_frame_contract", None)
            frame_hash = (
                explicit_frame_hash
                or getattr(frame_contract, "contract_hash", None)
                or (
                    None
                    if artifact is None or not artifact.pose_frame_contract_hashes
                    else artifact.pose_frame_contract_hashes[0]
                )
            )
            if frame_hash is None:
                raise ValueError(
                    "OpenPI raw-RTC could not resolve a frame identity; pass "
                    "--rtc-frame-contract-hash for an explicitly bound identity artifact."
                )
            creator = getattr(policy, "create_raw_rtc_adapter", None)
            if not callable(creator):
                raise TypeError(
                    "The selected OpenPI policy does not expose the raw-RTC adapter seam."
                )
            action_pose_mode = str(
                getattr(runtime_args, "policy_action_pose_mode", "absolute")
            ).strip().lower()
            result_action_representation = (
                "typed_canonical_v1"
                if action_pose_mode == "relative"
                else "legacy_normalized_v1"
            )
            return creator(
                frame_contract_hash=str(frame_hash),
                normalization_hash=str(normalization_hash),
                active_sides=tuple(factory_kwargs["active_sides"]),
                action_pose_mode=action_pose_mode,
                result_action_representation=result_action_representation,
            )

        rtc_model_adapter_factory = _create_openpi_raw_rtc_adapter

    delegated_args = [
        "--policy",
        "deployments.openpi.client.policy:create_policy",
        "--policy-kwargs-json",
        json.dumps(policy_kwargs, separators=(",", ":")),
        *runtime_args,
    ]
    if rtc_model_adapter_factory is None:
        return run_real_policy(delegated_args)
    return run_real_policy(
        delegated_args,
        rtc_model_adapter_factory=rtc_model_adapter_factory,
    )


if __name__ == "__main__":
    raise SystemExit(main())

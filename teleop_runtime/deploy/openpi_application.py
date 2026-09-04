"""Bind OpenPI model integrations to the existing real deployment runtime.

This module owns application arguments and RM65 runtime-profile selection. It
delegates model transport and artifact semantics to ``deployments.openpi`` and
delegates authority, safety, and dispatch to the canonical real deploy loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from deployments.policy_artifact import load_policy_artifact
from deployments.openpi.client.preflight import preflight_policy_artifact
from deployments.openpi.client.observation_adapter import (
    resolve_prepared_prompt_selection,
)
from deployments.openpi.camera_semantics import validate_declared_camera_semantics
from deployments.openpi.image_wire import (
    OPENPI_IMAGE_WIRE_JPEG_V1,
    OPENPI_IMAGE_WIRE_LEGACY_RAW_V1,
    OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,
)
from dataset_interface.camera_semantics import load_camera_role_binding
from dataset_contracts.model_continuation import (
    MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1,
    normalize_model_continuation_contract,
)
from robot_motion.deploy.async_producer import (
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
)

from .openpi_deployment_profile import (
    RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID,
    available_openpi_real_deployment_profiles,
    openpi_real_deployment_profile_args,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEPLOYMENT_PROFILES = available_openpi_real_deployment_profiles()
_IMAGE_WIRE_CLI_TO_CODEC = {
    "legacy-raw": OPENPI_IMAGE_WIRE_LEGACY_RAW_V1,
    "resize-raw": OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,
    "jpeg": OPENPI_IMAGE_WIRE_JPEG_V1,
}


def _rm65_bimanual_dmtac_args(*, rtc_real_shadow: bool = False) -> list[str]:
    """Compatibility wrapper for the reviewed versioned lab profile."""

    return openpi_real_deployment_profile_args(
        RM65_BIMANUAL_DMTAC_OPENPI_PROFILE_ID,
        rtc_real_shadow=rtc_real_shadow,
        project_root=_PROJECT_ROOT,
    )


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
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Explicit prompt text. Prompt-conditioned prepared artifacts accept "
            "only their checkpoint-bound canonical or training-variant prompts."
        ),
    )
    parser.add_argument(
        "--prompt-stage",
        default=None,
        metavar="STAGE_ID",
        help=(
            "Resolve a stage such as stage1, stage2, stage3, or stage4 to its "
            "canonical prompt from the prepared checkpoint artifact."
        ),
    )
    parser.add_argument(
        "--image-wire-codec",
        choices=tuple(_IMAGE_WIRE_CLI_TO_CODEC),
        default="legacy-raw",
        help=(
            "Optional negotiated image transport. legacy-raw preserves the "
            "existing ndarray protocol; resize-raw and jpeg require a prepared "
            "server that advertises the matching v1 capability."
        ),
    )
    parser.add_argument(
        "--image-wire-strict",
        action="store_true",
        help=(
            "Fail before sending if no negotiated optimized image codec is "
            "available. JPEG may still roll back to resize-raw; without this "
            "flag, the final rollback is the historical raw ndarray path."
        ),
    )
    parser.add_argument(
        "--model",
        "--expected-model-id",
        dest="expected_model_id",
        default=None,
        help="Optional model identity hint; enforced only with --strict-model-identity.",
    )
    parser.add_argument(
        "--checkpoint-step",
        "--expected-checkpoint-step",
        dest="expected_checkpoint_step",
        default=None,
        help="Optional checkpoint identity hint; enforced only with --strict-model-identity.",
    )
    parser.add_argument(
        "--strict-model-identity",
        action="store_true",
        help=(
            "Require the server model id and checkpoint step to match the "
            "provided values or policy artifact. By default deployment checks "
            "protocol compatibility only."
        ),
    )
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
    parser.add_argument(
        "--camera-role-registry",
        type=Path,
        default=(
            _PROJECT_ROOT / "robot_motion/configs/rm65_camera_registry.json"
        ),
        help=(
            "Registry used to validate an artifact camera_role_binding_hash "
            "before any real runtime resources are created."
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
        "--prepared-training-profile",
        type=Path,
        default=None,
        help=(
            "Checked-in prepared training profile used with --checkpoint-dir "
            "to derive and validate the same normalization-bound deployment "
            "artifact as the OpenPI server."
        ),
    )
    parser.add_argument(
        "--model-rtc-profile",
        type=Path,
        default=None,
        help=(
            "Optional trained model-RTC profile used with the prepared "
            "training profile to verify checkpoint receipt and deployment binding."
        ),
    )
    parser.add_argument(
        "--deployment-profile",
        choices=_DEPLOYMENT_PROFILES,
        default=None,
        help=(
            "Inject the checked-in real-runtime arguments for the selected lab "
            "profile. Explicit trailing runtime flags override profile values. "
            "The profile never enables --send-actions; callers may explicitly add "
            "it after the profile's qpos-init and safety defaults."
        ),
    )
    parser.add_argument(
        "--disconnected-gripper-sides",
        choices=("none", "left", "right", "both"),
        default="none",
        help=(
            "Keep selected DMTAC gripper side(s) intentionally offline while "
            "preserving the prepared model I/O layout. No serial/tactile "
            "connection or gripper command is sent for those sides."
        ),
    )
    parser.add_argument(
        "--disconnected-gripper-state",
        type=float,
        default=0.0,
        metavar="SCALAR",
        help=(
            "Fixed assumed [0,1] state for selected disconnected grippers "
            "(0=open, 1=closed); it is marked as assumed, not measured."
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
            MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
        ),
        default=None,
        help=(
            "Optional default-off shared deployment RTC profile. The OpenPI raw-RTC "
            "profile keeps the existing final-chunk model/server interface; the "
            "model-RTC profile additionally uses the artifact-bound conditioned "
            "provider without changing dispatch authority."
        ),
    )
    parser.add_argument(
        "--rtc-real-shadow",
        action="store_true",
        help=(
            "Explicit live model RTC transport and observation path. Without "
            "--send-actions it is observation-only; with explicit --send-actions the "
            "checked-in profile retains its dataset-frame qpos-init trajectory and "
            "safety gates before dispatch."
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
        "--debug-payload-mode",
        choices=("compact", "full"),
        default="compact",
        help=(
            "Deployment debug detail. compact is bounded for live control; "
            "full restores historical unbounded payloads for offline diagnosis."
        ),
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

    model_transport_profiles = {
        RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1,
        MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1,
    }
    if args.rtc_real_shadow and args.rtc_profile not in model_transport_profiles:
        parser.error(
            "--rtc-real-shadow requires a raw or conditioned model RTC profile."
        )
    if args.image_wire_strict and args.image_wire_codec == "legacy-raw":
        parser.error("--image-wire-strict requires an optimized image-wire codec.")

    if (
        args.rtc_profile == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1
        and not _optional_text(args.rtc_normalization_hash)
    ):
        parser.error(
            f"{RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1} requires --rtc-normalization-hash."
        )

    artifact_path = args.policy_artifact_json
    prepared_resolution = None
    if args.checkpoint_dir is not None:
        artifact_path = _artifact_path_from_checkpoint(args.checkpoint_dir)
    if args.prepared_training_profile is not None:
        if args.checkpoint_dir is None:
            parser.error(
                "--prepared-training-profile requires --checkpoint-dir."
            )
        if args.policy_artifact_json is not None:
            parser.error(
                "--prepared-training-profile cannot be combined with "
                "--policy-artifact-json."
            )
        from deployments.openpi.prepared_deployment import (
            prepare_checkpoint_deployment_artifact,
        )

        prepared_resolution = prepare_checkpoint_deployment_artifact(
            args.checkpoint_dir,
            args.prepared_training_profile,
            model_rtc_profile_path=args.model_rtc_profile,
        )
        artifact_path = prepared_resolution.deployment_artifact_path
    elif args.model_rtc_profile is not None:
        parser.error(
            "--model-rtc-profile requires --prepared-training-profile."
        )
    if artifact_path is not None and not artifact_path.is_file():
        parser.error(f"policy artifact does not exist or is not a file: {artifact_path}")
    policy_artifact = None if artifact_path is None else load_policy_artifact(artifact_path)
    if policy_artifact is not None:
        validate_declared_camera_semantics(policy_artifact)
        expected_camera_binding_hash = str(
            policy_artifact.metadata.get("camera_role_binding_hash") or ""
        ).strip()
        if expected_camera_binding_hash:
            actual_camera_binding_hash = load_camera_role_binding(
                args.camera_role_registry
            ).contract_hash
            if expected_camera_binding_hash != actual_camera_binding_hash:
                parser.error(
                    "policy artifact camera_role_binding_hash differs from "
                    "--camera-role-registry."
                )
    try:
        resolved_prompt, resolved_prompt_stage = resolve_prepared_prompt_selection(
            None if policy_artifact is None else policy_artifact.metadata,
            prompt=_optional_text(args.prompt),
            prompt_stage=_optional_text(args.prompt_stage),
        )
    except ValueError as exc:
        parser.error(str(exc))
    if (
        args.rtc_profile
        == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
        and policy_artifact is not None
        and policy_artifact.model_continuation is not None
        and normalize_model_continuation_contract(
            policy_artifact.model_continuation
        ).conditioning_origin
        == MODEL_CONTINUATION_ORIGIN_TRAINED_PREFIX_V1
    ):
        from deployments.openpi.training import (
            verify_openpi_model_rtc_checkpoint,
        )

        evidence = verify_openpi_model_rtc_checkpoint(
            artifact_path.parent
            if prepared_resolution is None
            else prepared_resolution.checkpoint_dir
        )
        expected_checkpoint_artifact = (
            policy_artifact
            if prepared_resolution is None
            else prepared_resolution.checkpoint_artifact
        )
        if evidence.artifact.to_dict() != expected_checkpoint_artifact.to_dict():
            parser.error(
                "Verified trained model-RTC artifact differs from the "
                "requested policy artifact."
            )
    if args.contract_check_only:
        if policy_artifact is None:
            parser.error(
                "--contract-check-only requires --checkpoint-dir or "
                "--policy-artifact-json."
            )
        report = preflight_policy_artifact(policy_artifact).to_dict()
        if resolved_prompt is not None:
            report["prompt_selection"] = {
                "source": (
                    "prepared_stage"
                    if resolved_prompt_stage is not None
                    else "explicit_prompt"
                ),
                "stage_id": resolved_prompt_stage,
                "prompt": resolved_prompt,
            }
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    strict_model_id = _optional_text(args.expected_model_id)
    strict_checkpoint_step = _optional_text(args.expected_checkpoint_step)
    if args.strict_model_identity and policy_artifact is not None:
        strict_model_id = strict_model_id or policy_artifact.model_id
        if strict_checkpoint_step is None and policy_artifact.checkpoint_step is not None:
            strict_checkpoint_step = str(policy_artifact.checkpoint_step)

    policy_kwargs: dict[str, Any] = {
        "host": args.host,
        "port": int(args.port),
        "api_key": _optional_text(args.api_key),
        "prompt": resolved_prompt,
        # Model identity is intentionally not part of the default downstream
        # protocol gate. Keep the old exact check available only when a caller
        # explicitly opts in.
        "expected_model_id": (
            strict_model_id
            if args.strict_model_identity
            else None
        ),
        "expected_checkpoint_step": (
            strict_checkpoint_step
            if args.strict_model_identity
            else None
        ),
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
    if args.image_wire_codec != "legacy-raw":
        policy_kwargs["image_wire_config"] = {
            "preferred_codec": _IMAGE_WIRE_CLI_TO_CODEC[args.image_wire_codec],
            "allow_legacy_fallback": not bool(args.image_wire_strict),
        }
    policy_kwargs = {key: value for key, value in policy_kwargs.items() if value is not None}

    if resolved_prompt_stage is not None:
        print(
            "[OpenPI] prepared prompt stage selected: "
            f"stage={resolved_prompt_stage} prompt={resolved_prompt!r}",
            flush=True,
        )

    from teleop_runtime.deploy.run_policy import main as run_real_policy

    runtime_args: list[str] = []
    if args.deployment_profile == "rm65_bimanual_dmtac":
        runtime_args.extend(
            _rm65_bimanual_dmtac_args(rtc_real_shadow=bool(args.rtc_real_shadow))
        )
    runtime_args.extend(teleop_args)
    if args.disconnected_gripper_sides != "none":
        runtime_args.extend(
            (
                "--disconnected-gripper-sides",
                str(args.disconnected_gripper_sides),
                "--disconnected-gripper-state",
                str(float(args.disconnected_gripper_state)),
            )
        )
    if artifact_path is not None:
        runtime_args.extend(("--policy-artifact-json", str(artifact_path)))
    if args.debug:
        runtime_args.append("--debug")
    if args.debug_dir is not None:
        runtime_args.extend(("--debug-dir", str(args.debug_dir)))
    if args.debug:
        runtime_args.extend(("--debug-payload-mode", str(args.debug_payload_mode)))
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
    if args.rtc_profile in model_transport_profiles:
        normalization_hash = _optional_text(args.rtc_normalization_hash)
        if (
            args.rtc_profile
            == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
        ):
            if (
                policy_artifact is None
                or policy_artifact.model_continuation is None
            ):
                parser.error(
                    "model-RTC requires an artifact-bound continuation contract."
                )
            contract_normalization_hash = (
                normalize_model_continuation_contract(
                    policy_artifact.model_continuation
                ).runtime.normalization_hash
            )
            if (
                normalization_hash is not None
                and normalization_hash != contract_normalization_hash
            ):
                parser.error(
                    "--rtc-normalization-hash disagrees with the model-RTC "
                    "artifact continuation contract."
                )
            normalization_hash = contract_normalization_hash
        explicit_frame_hash = _optional_text(args.rtc_frame_contract_hash)

        def _create_openpi_rtc_adapter(**factory_kwargs: Any):
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
            creator_name = (
                "create_model_rtc_adapter"
                if args.rtc_profile
                == MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_ID_V1
                else "create_raw_rtc_adapter"
            )
            creator = getattr(policy, creator_name, None)
            if not callable(creator):
                raise TypeError(
                    "The selected OpenPI policy does not expose the requested "
                    f"RTC adapter seam {creator_name!r}."
                )
            action_pose_mode = str(
                getattr(runtime_args, "policy_action_pose_mode", "absolute")
            ).strip().lower()
            creator_kwargs = {
                "frame_contract_hash": str(frame_hash),
                "normalization_hash": str(normalization_hash),
                "active_sides": tuple(factory_kwargs["active_sides"]),
                "action_pose_mode": action_pose_mode,
            }
            if (
                args.rtc_profile
                == RAW_RTC_FINAL_CHUNK_PROFILE_ID_V1
            ):
                creator_kwargs["result_action_representation"] = (
                    "typed_canonical_v1"
                    if action_pose_mode == "relative"
                    else "legacy_normalized_v1"
                )
            return creator(
                **creator_kwargs,
            )

        rtc_model_adapter_factory = _create_openpi_rtc_adapter

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

"""Dedicated default-off launcher for independent real RM65 Skills."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_group_parent,
    validate_agent_gateway_unix_server_preflight,
)
from deployments.perception import (
    GraspNetPointGraspModelSpec,
    Sam2GraspNetPointGraspPlanner,
    Sam2GroundingDinoLocalizer,
    Sam2GroundingDinoModelSpec,
    Sam2VgnPointGraspPlanner,
    VgnPointGraspModelSpec,
)
from deployments.perception.dmtac_grasp_overlay import (
    DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD,
)
from robot_motion.rm65 import RM65_BIMANUAL_POSITION_V1_NAME
from teleop_runtime.skill_deploy import (
    ExclusiveRM65SkillDeploymentProfile,
    assemble_exclusive_rm65_skill_lower,
    resolve_exclusive_rm65_skill_deployment_profile,
)
from teleop_runtime.skill_deploy.motion_trace import (
    DEFAULT_SKILL_MOTION_TRACE_MAX_DURATION_SEC,
    SkillMotionTraceConfig,
    SkillMotionTraceRecorder,
)

from .exclusive_application import ExclusiveRealSkillApplicationState
from .rm65_composition import (
    ExclusiveRM65AgentPrimitiveComposition,
    assemble_exclusive_rm65_agent_primitive_composition,
)
from .session_authorization import (
    REAL_AGENT_PRIMITIVE_ACTIONS,
    REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RM65_CAMERA_MOUNT_URDF = (
    _REPO_ROOT / "assets/rm65_inspire_gripper_v4/"
    "rm65_dual_arm_wrist_cameras_v4_head_camera_4b2_cad_with_legacy_housing.urdf"
)
DEFAULT_RM65_CAMERA_REGISTRY_V4 = (
    _REPO_ROOT / "robot_motion/configs/rm65_camera_registry_v4.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway real-rm65",
        description=(
            "Serve one independently owned real RM65 Skill Gateway. The "
            "process is default-off and starts hardware only after every "
            "trusted profile resolves and the exact confirmation is present."
        ),
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        required=True,
        help=(
            "absolute Unix socket path in an existing private real directory; "
            "the encoded path must fit Linux's 107-byte limit"
        ),
    )
    parser.add_argument("--arm-ip-assignment", type=Path, required=True)
    parser.add_argument("--left-tool-profile", type=Path, required=True)
    parser.add_argument("--right-tool-profile", type=Path, required=True)
    parser.add_argument("--envelope-profile", type=Path, required=True)
    parser.add_argument(
        "--camera-config",
        type=Path,
        help="absolute RealSense config; requires --camera-registry and --artifact-root",
    )
    parser.add_argument(
        "--camera-registry",
        type=Path,
        help="absolute trusted camera registry used for serials and depth geometry",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="absolute existing non-world-writable directory for the v2 Artifact CAS",
    )
    parser.add_argument(
        "--enable-point-grasp",
        action="store_true",
        help="enable main-view proposal generation and three-view candidate rendering",
    )
    parser.add_argument(
        "--debug-render-object-coordinates",
        action="store_true",
        help=(
            "debug only: overlay the depth-localized target point and canonical "
            "Station XYZ axes on point-grasp renderings"
        ),
    )
    parser.add_argument("--object-localization-asset-bundle", type=Path)
    parser.add_argument("--object-localization-device", default="cuda")
    parser.add_argument(
        "--point-grasp-backend", choices=("vgn", "graspnet"), default="vgn"
    )
    parser.add_argument("--point-grasp-asset-bundle", type=Path)
    parser.add_argument("--point-grasp-device", default="cuda")
    parser.add_argument(
        "--camera-mount-urdf",
        type=Path,
        help=(
            "reviewed URDF containing the Station, head camera, and both wrist "
            "optical-frame chains; defaults to the detailed RM65 v4 asset"
        ),
    )
    parser.add_argument(
        "--position-profile",
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help="trusted lower RM65 position-control profile name",
    )
    parser.add_argument(
        "--collision-scene-profile",
        default="dual_arm_hand_proxy_open",
        help="trusted lower coupled-IK/collision scene profile name",
    )
    parser.add_argument(
        "--motion-trace-root",
        type=Path,
        help=(
            "optional absolute existing non-world-writable directory; enables "
            "preallocated 30/60/120 Hz RM65 Skill motion tracing and writes "
            "NPZ/manifest/summary only after lower shutdown"
        ),
    )
    parser.add_argument(
        "--motion-trace-max-duration-sec",
        type=float,
        default=DEFAULT_SKILL_MOTION_TRACE_MAX_DURATION_SEC,
        help="fixed preallocated trace duration (default: 120s)",
    )
    parser.add_argument(
        "--debug-motion-skill",
        action="store_true",
        help=(
            "debug only: add bounded IK phase profiling to the existing "
            "preallocated motion trace; requires --motion-trace-root"
        ),
    )
    parser.add_argument("--authorized-side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--authorized-action",
        choices=REAL_AGENT_PRIMITIVE_ACTIONS,
        help="one reviewed primitive action allowed for this process session",
    )
    parser.add_argument(
        "--authorized-actions",
        choices=REAL_AGENT_PRIMITIVE_ACTIONS,
        nargs="+",
        help=(
            "one or more reviewed primitive actions allowed for this persistent "
            "Gateway process session"
        ),
    )
    parser.add_argument(
        "--confirm",
        required=True,
        help=("must exactly equal " + REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN),
    )
    parser.add_argument("--client-id", default="rpent.dexteleop")
    parser.add_argument("--client-uid", type=int, default=os.getuid())
    parser.add_argument(
        "--ssh-read-only-client",
        action="append",
        nargs=2,
        default=[],
        metavar=("CLIENT_ID", "UID"),
        help=(
            "additional distinct client identity and Unix UID restricted by the "
            "Gateway to the SSH read-only method/Route surface; may be repeated"
        ),
    )
    parser.add_argument("--runtime-session-id", default=None)
    parser.add_argument(
        "--control-domain-id",
        default="exclusive-rm65-agent-skill",
    )
    parser.add_argument("--embodiment-instance-id", default=None)
    parser.add_argument("--max-retained-operations", type=int, default=16)
    parser.add_argument("--peer-idle-timeout-sec", type=float, default=10.0)
    parser.add_argument("--grant-lifetime-sec", type=float, default=5.0)
    return parser


def _absolute_existing_file(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error(f"{flag} must be an absolute path")
    resolved = expanded.resolve()
    if not resolved.is_file():
        parser.error(f"{flag} is not a file: {resolved}")
    return resolved


def _effective_camera_registry(args: argparse.Namespace) -> Path | None:
    if args.camera_registry is not None:
        return args.camera_registry
    if args.enable_point_grasp:
        return DEFAULT_RM65_CAMERA_REGISTRY_V4
    return None


def _effective_camera_mount_urdf(args: argparse.Namespace) -> Path | None:
    if args.camera_mount_urdf is not None:
        return args.camera_mount_urdf
    if args.enable_point_grasp:
        return DEFAULT_RM65_CAMERA_MOUNT_URDF
    return None


def _absolute_socket_path(
    parser: argparse.ArgumentParser,
    value: Path,
) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error("--socket-path must be an absolute path")
    return Path(os.path.abspath(expanded))


def _absolute_managed_directory(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error(f"{flag} must be an absolute path")
    path = Path(os.path.abspath(expanded))
    try:
        info = os.lstat(path)
    except OSError:
        parser.error(f"{flag} is not an existing directory: {path}")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        parser.error(f"{flag} must be a real directory, not a symlink: {path}")
    if info.st_mode & stat.S_IWOTH:
        parser.error(f"{flag} cannot be world-writable: {path}")
    return path


def _runtime_session_id(explicit: str | None) -> str:
    if explicit is not None:
        return explicit
    return "dexteleop-real-rm65-skill:" + uuid.uuid4().hex


def _ready_payload(
    composition: ExclusiveRM65AgentPrimitiveComposition,
) -> dict[str, Any]:
    manifest = composition.deployment.gateway_session.contract_bundle.route_manifest
    session = composition.deployment.gateway_session
    capabilities = getattr(
        getattr(getattr(session, "service", None), "protocol_runtime", None),
        "session_capabilities",
        None,
    )
    perception_capture = getattr(composition, "perception_capture", None)
    authorized_actions = tuple(
        getattr(
            composition,
            "authorized_actions",
            (composition.authorized_action,),
        )
    )
    return {
        "event": "gateway_ready",
        "socket_path": str(composition.socket_path),
        "client_id": composition.client_id,
        "read_only_client_ids": list(getattr(composition, "read_only_client_ids", ())),
        "socket_mode": format(getattr(composition, "socket_mode", 0o600), "04o"),
        "target_kind": manifest.target_kind,
        "current_mode": manifest.current_mode,
        "runtime_session_id": manifest.runtime_session_id,
        "control_domain_id": manifest.control_domain_id,
        "target_binding_id": manifest.target_binding_id,
        "route_manifest_hash": manifest.semantic_content_hash,
        "artifact_transfer_enabled": bool(
            capabilities is not None and capabilities.artifact_transfer_enabled
        ),
        "artifact_access_mechanisms": (
            []
            if capabilities is None
            else list(capabilities.artifact_access_mechanisms)
        ),
        "domain_emergency_stop_available": bool(
            session.emergency_stop_port is not None
            and "own_domain_emergency_stop_v1"
            in manifest.supported_transport_features
        ),
        "station_manifest_hash": manifest.station_manifest_hash,
        "execution_manifest_hash": manifest.execution_manifest_hash,
        "deployment_profile_hash": composition.lower.profile.semantic_content_hash,
        "authorized_side": composition.authorized_side,
        "authorized_action": composition.authorized_action,
        "authorized_actions": list(authorized_actions),
        "authorization_scope": (
            "persistent_session_matching_action"
            if len(authorized_actions) == 1
            else "persistent_session_listed_primitive_actions"
        ),
        "perception": (
            None
            if perception_capture is None
            else {
                "rgb_roles": list(perception_capture.profile.rgb_roles),
                "depth_roles": list(perception_capture.profile.depth_roles),
                "control_effect_class": "read_only",
                "point_grasp_enabled": (
                    composition.station_point_grasp_registration is not None
                ),
                "point_grasp_render_view_roles": (
                    []
                    if composition.station_point_grasp_render_registration is None
                    else list(
                        composition.station_point_grasp_render_registration.profile.source_by_role
                    )
                ),
                "point_grasp_debug_object_coordinates": (
                    False
                    if composition.station_point_grasp_render_registration is None
                    else composition.station_point_grasp_render_registration.profile.debug_target_coordinate_overlay
                ),
            }
        ),
        "routes": [
            {
                "skill_id": entry.route.skill_id,
                "skill_version": entry.route.skill_version,
                "route_id": entry.route.route_id,
                "route_version": entry.route.route_version,
                "route_manifest_entry_hash": entry.semantic_content_hash,
            }
            for entry in manifest.skill_routes
        ],
    }


def _stopped_payload(
    composition: ExclusiveRM65AgentPrimitiveComposition,
    *,
    motion_trace: SkillMotionTraceRecorder | None = None,
    motion_trace_close_error: BaseException | None = None,
) -> dict[str, Any]:
    snapshot = composition.deployment.control_runtime.motion_port.snapshot()
    hold = composition.deployment.control_runtime.last_hold_receipt
    emergency_stop = composition.lower.ports.last_emergency_stop_receipt
    gateway_emergency_stop_receipt = (
        composition.deployment.gateway_session.service.emergency_stop_receipt
    )
    control_fault = composition.application.control_fault
    lower_hold_proved = bool(hold is not None and hold.succeeded)
    lower_emergency_stop_proved = bool(
        emergency_stop is not None
        and emergency_stop.succeeded
        and set(emergency_stop.status_by_side.values()) == {"confirmed"}
        and gateway_emergency_stop_receipt is not None
        and gateway_emergency_stop_receipt.succeeded
        and gateway_emergency_stop_receipt.proves_motion_or_hold
    )
    passive_read_failures = []
    for (
        skill_id,
        executor,
    ) in composition.deployment.gateway_session.passive_read_executors_by_skill_id.items():
        diagnostics = getattr(executor, "failure_diagnostics", ())
        passive_read_failures.extend(
            {"skill_id": skill_id, **dict(diagnostic)} for diagnostic in diagnostics
        )
    return {
        "event": "gateway_stopped",
        "application_state": composition.application.state.value,
        "motion_command_dispatched": snapshot.dispatch_count > 0,
        "dispatch_count": snapshot.dispatch_count,
        "control_fault": (
            None
            if control_fault is None
            else f"{type(control_fault).__name__}: {control_fault}"[:512]
        ),
        "measurement_sampler": (
            composition.lower.ports.measurement_sampler_snapshot()
        ),
        "control_cycle": (
            composition.deployment.control_runtime.control_cycle_snapshot()
        ),
        "motion_trace": (
            {
                "enabled": False,
                "closed": False,
                "output_dir": None,
                "close_error": None,
            }
            if motion_trace is None
            else {
                "enabled": True,
                **motion_trace.snapshot(),
                "close_error": (
                    None
                    if motion_trace_close_error is None
                    else (
                        f"{type(motion_trace_close_error).__name__}: "
                        f"{motion_trace_close_error}"
                    )[:512]
                ),
            }
        ),
        "passive_read_failures": passive_read_failures,
        "lower_hold_proved": lower_hold_proved,
        "lower_hold_evidence_refs": ([] if hold is None else list(hold.evidence_refs)),
        "emergency_stop_latched": (
            composition.deployment.gateway_session.service.emergency_stop_latched
        ),
        "emergency_stop_confirmed": lower_emergency_stop_proved,
        "emergency_stop_status_by_side": (
            {}
            if emergency_stop is None
            else dict(emergency_stop.status_by_side)
        ),
        "emergency_stop_evidence_refs": (
            []
            if emergency_stop is None
            else list(emergency_stop.evidence_refs)
        ),
        "safe_shutdown": lower_hold_proved or lower_emergency_stop_proved,
        "safe_shutdown_basis": (
            "lower_emergency_stop_receipt"
            if lower_emergency_stop_proved
            else "lower_hold_receipt"
            if lower_hold_proved
            else "missing_lower_safety_stop_proof"
        ),
    }


def _attach_motion_trace(
    composition: ExclusiveRM65AgentPrimitiveComposition,
    recorder: SkillMotionTraceRecorder,
) -> None:
    """Attach all fixed-field observers before any lower worker starts."""

    control_runtime = composition.deployment.control_runtime
    target_builder = composition.lower.target_builder
    backends = composition.lower.arm_backends
    ik_setter = getattr(target_builder, "set_ik_shadow_trace_sink", None)
    if not callable(ik_setter):
        raise TypeError("RM65 target builder lacks the fixed-field IK trace hook")
    detach: list[tuple[object, str]] = []
    try:
        control_runtime.set_motion_trace_sink(recorder.record_control_cycle)
        detach.append((control_runtime, "set_motion_trace_sink"))
        ik_setter(recorder.record_ik_shadow)
        detach.append((target_builder, "set_ik_shadow_trace_sink"))
        for side in ("left", "right"):
            backend = backends[side]
            setter = getattr(backend, "set_actuator_command_trace_sink", None)
            if not callable(setter):
                raise TypeError(f"{side} RM65 backend lacks the actuator trace hook")
            setter(recorder.record_actuator_command)
            detach.append((backend, "set_actuator_command_trace_sink"))
    except Exception as attach_error:
        rollback_errors: list[str] = []
        for owner, method in reversed(detach):
            try:
                getattr(owner, method)(None)
            except Exception as rollback_error:  # noqa: BLE001 - bounded cleanup.
                rollback_errors.append(
                    f"{method}:{type(rollback_error).__name__}:{rollback_error}"[:512]
                )
        if rollback_errors:
            raise RuntimeError(
                "motion trace attach failed and rollback was incomplete: "
                + "; ".join(rollback_errors)
            ) from attach_error
        raise


def _detach_motion_trace(
    composition: ExclusiveRM65AgentPrimitiveComposition,
) -> list[str]:
    """Best-effort detach after lower shutdown; return bounded diagnostics."""

    errors: list[str] = []
    for label, owner, method in (
        (
            "control",
            composition.deployment.control_runtime,
            "set_motion_trace_sink",
        ),
        ("ik", composition.lower.target_builder, "set_ik_shadow_trace_sink"),
        (
            "left_actuator",
            composition.lower.arm_backends["left"],
            "set_actuator_command_trace_sink",
        ),
        (
            "right_actuator",
            composition.lower.arm_backends["right"],
            "set_actuator_command_trace_sink",
        ),
    ):
        try:
            getattr(owner, method)(None)
        except Exception as exc:  # noqa: BLE001 - preserve all detach failures.
            errors.append(f"{label}:{type(exc).__name__}:{exc}"[:512])
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.confirm != REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN:
        parser.error(
            "--confirm must exactly equal " + REAL_AGENT_PRIMITIVE_SESSION_CONFIRM_TOKEN
        )
    if (args.authorized_action is None) == (args.authorized_actions is None):
        parser.error(
            "provide exactly one of --authorized-action or --authorized-actions"
        )
    if args.client_uid < 0:
        parser.error("--client-uid must be non-negative")
    try:
        client_bindings = compile_agent_gateway_unix_client_bindings(
            primary_client_id=args.client_id,
            primary_client_uid=args.client_uid,
            ssh_read_only_clients=args.ssh_read_only_client,
        )
    except (TypeError, ValueError) as error:
        parser.error(f"invalid Agent Gateway client bindings: {error}")
    if args.max_retained_operations <= 0:
        parser.error("--max-retained-operations must be positive")
    if not 0.0 < args.peer_idle_timeout_sec:
        parser.error("--peer-idle-timeout-sec must be positive")
    if not 0.0 < args.grant_lifetime_sec <= 30.0:
        parser.error("--grant-lifetime-sec must be in (0, 30]")
    if (
        not np.isfinite(float(args.motion_trace_max_duration_sec))
        or args.motion_trace_max_duration_sec <= 0.0
    ):
        parser.error("--motion-trace-max-duration-sec must be positive and finite")
    if args.debug_motion_skill and args.motion_trace_root is None:
        parser.error("--debug-motion-skill requires --motion-trace-root")

    socket_path = _absolute_socket_path(parser, args.socket_path)
    camera_registry_arg = _effective_camera_registry(args)
    perception_supplied = tuple(
        value is not None
        for value in (args.camera_config, camera_registry_arg, args.artifact_root)
    )
    if any(perception_supplied) and not all(perception_supplied):
        parser.error(
            "--camera-config, --camera-registry, and --artifact-root must be "
            "provided together"
        )
    if args.enable_point_grasp and not all(perception_supplied):
        parser.error("--enable-point-grasp requires the composed camera options")
    if args.debug_render_object_coordinates and not args.enable_point_grasp:
        parser.error("--debug-render-object-coordinates requires --enable-point-grasp")
    camera_mount_urdf_arg = _effective_camera_mount_urdf(args)
    point_grasp_values = (
        args.object_localization_asset_bundle,
        args.point_grasp_asset_bundle,
        camera_mount_urdf_arg,
    )
    if not args.enable_point_grasp and any(
        value is not None
        for value in (
            args.object_localization_asset_bundle,
            args.point_grasp_asset_bundle,
            args.camera_mount_urdf,
        )
    ):
        parser.error("point-grasp model bundle options require --enable-point-grasp")
    if args.enable_point_grasp != all(
        value is not None for value in point_grasp_values
    ):
        parser.error(
            "--enable-point-grasp requires --object-localization-asset-bundle, "
            "--point-grasp-asset-bundle, and --camera-mount-urdf; those options "
            "are invalid without --enable-point-grasp"
        )
    try:
        validate_agent_gateway_unix_server_preflight(
            socket_path=socket_path,
            peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        )
    except (FileExistsError, TypeError, ValueError) as error:
        parser.error(f"--socket-path preflight failed: {error}")
    if client_bindings.read_only_client_ids:
        try:
            validate_agent_gateway_unix_group_parent(socket_path=socket_path)
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"invalid read-only socket parent: {error}")
    files = {
        "arm_ip_assignment": _absolute_existing_file(
            parser,
            args.arm_ip_assignment,
            flag="--arm-ip-assignment",
        ),
        "left_tool_profile": _absolute_existing_file(
            parser,
            args.left_tool_profile,
            flag="--left-tool-profile",
        ),
        "right_tool_profile": _absolute_existing_file(
            parser,
            args.right_tool_profile,
            flag="--right-tool-profile",
        ),
        "envelope_profile": _absolute_existing_file(
            parser,
            args.envelope_profile,
            flag="--envelope-profile",
        ),
    }
    resolved = resolve_exclusive_rm65_skill_deployment_profile(
        ExclusiveRM65SkillDeploymentProfile(
            arm_ip_assignment_path=files["arm_ip_assignment"],
            tool_profile_path_by_side={
                "left": files["left_tool_profile"],
                "right": files["right_tool_profile"],
            },
            envelope_profile_path=files["envelope_profile"],
            position_profile_name=args.position_profile,
            collision_scene_profile=args.collision_scene_profile,
        )
    )
    if args.enable_point_grasp:
        selected_tool_tcp = resolved.tool_tcp_pose_flange_m_rad_by_side[
            args.authorized_side
        ]
        if not np.allclose(
            selected_tool_tcp,
            DMTAC_M_REVIEWED_TOOL_TCP_POSE_FLANGE_M_RAD,
            atol=1e-9,
            rtol=0.0,
        ):
            parser.error(
                "--enable-point-grasp requires the reviewed DMTAC_M actual "
                "tool_tcp pose [0, 0, 0.14, 0, 0, 0] for --authorized-side; "
                "candidate geometry cannot be reinterpreted for another tool"
            )
    lower_kwargs: dict[str, object] = {"profile": resolved}
    if args.debug_motion_skill:
        lower_kwargs["debug_motion_skill"] = True
    lower = assemble_exclusive_rm65_skill_lower(**lower_kwargs)
    camera_config = (
        None
        if args.camera_config is None
        else _absolute_existing_file(parser, args.camera_config, flag="--camera-config")
    )
    camera_registry = (
        None
        if camera_registry_arg is None
        else _absolute_existing_file(
            parser,
            camera_registry_arg,
            flag="--camera-registry",
        )
    )
    artifact_root = (
        None
        if args.artifact_root is None
        else _absolute_managed_directory(
            parser,
            args.artifact_root,
            flag="--artifact-root",
        )
    )
    motion_trace_root = (
        None
        if args.motion_trace_root is None
        else _absolute_managed_directory(
            parser,
            args.motion_trace_root,
            flag="--motion-trace-root",
        )
    )
    object_localizer = None
    point_grasp_planner = None
    camera_mount_urdf = None
    if args.enable_point_grasp:
        object_bundle = _absolute_managed_directory(
            parser,
            args.object_localization_asset_bundle,
            flag="--object-localization-asset-bundle",
        )
        point_grasp_bundle = _absolute_managed_directory(
            parser,
            args.point_grasp_asset_bundle,
            flag="--point-grasp-asset-bundle",
        )
        camera_mount_urdf = _absolute_existing_file(
            parser,
            camera_mount_urdf_arg,
            flag="--camera-mount-urdf",
        )
        try:
            localization_spec = Sam2GroundingDinoModelSpec.from_asset_bundle(
                object_bundle
            )
            object_localizer = Sam2GroundingDinoLocalizer(
                model_spec=localization_spec,
                device=args.object_localization_device,
            )
            if args.point_grasp_backend == "vgn":
                point_grasp_spec = VgnPointGraspModelSpec.from_asset_bundle(
                    point_grasp_bundle
                )
                point_grasp_planner = Sam2VgnPointGraspPlanner(
                    model_spec=point_grasp_spec,
                    point_segmenter=object_localizer,
                    device=args.point_grasp_device,
                )
            else:
                point_grasp_spec = GraspNetPointGraspModelSpec.from_asset_bundle(
                    point_grasp_bundle
                )
                point_grasp_planner = Sam2GraspNetPointGraspPlanner(
                    model_spec=point_grasp_spec,
                    point_segmenter=object_localizer,
                    device=args.point_grasp_device,
                )
        except (OSError, TypeError, ValueError) as error:
            parser.error(f"invalid point-grasp model bundle: {error}")
    embodiment_instance_id = (
        args.embodiment_instance_id
        if args.embodiment_instance_id is not None
        else "dual-rm65:" + resolved.semantic_content_hash[:16]
    )
    runtime_session_id = _runtime_session_id(args.runtime_session_id)
    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=socket_path,
        authorized_side=args.authorized_side,
        authorized_action=args.authorized_action,
        authorized_actions=(
            None if args.authorized_actions is None else tuple(args.authorized_actions)
        ),
        confirmation_token=args.confirm,
        client_id=args.client_id,
        client_uid=args.client_uid,
        ssh_read_only_client_id_by_uid={
            uid: client_id
            for uid, client_id in client_bindings.client_id_by_uid.items()
            if client_id in client_bindings.read_only_client_ids
        },
        runtime_session_id=runtime_session_id,
        control_domain_id=args.control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        max_retained_operations=args.max_retained_operations,
        peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        grant_lifetime_sec=args.grant_lifetime_sec,
        camera_config_path=camera_config,
        camera_registry_path=camera_registry,
        artifact_root=artifact_root,
        camera_mount_urdf_path=camera_mount_urdf,
        object_localizer=object_localizer,
        point_grasp_planner=point_grasp_planner,
        debug_render_object_coordinates=args.debug_render_object_coordinates,
    )

    motion_trace: SkillMotionTraceRecorder | None = None
    if motion_trace_root is not None:
        trace_session_dir = motion_trace_root / (
            time.strftime("%Y%m%d_%H%M%S")
            + "_exclusive_rm65_skill_"
            + uuid.uuid4().hex[:8]
        )
        motion_trace = SkillMotionTraceRecorder(
            SkillMotionTraceConfig(
                output_dir=trace_session_dir,
                reference_hz=resolved.position_profile.rates.reference_hz,
                command_hz=resolved.position_profile.rates.command_hz,
                measurement_hz=composition.lower.ports.profile.state_refresh_hz,
                max_duration_sec=args.motion_trace_max_duration_sec,
                metadata={
                    "entrypoint": "python -m agent_gateway real-rm65",
                    "runtime_session_id": runtime_session_id,
                    "control_domain_id": args.control_domain_id,
                    "authorized_side": args.authorized_side,
                    "authorized_action": args.authorized_action,
                    "authorized_actions": args.authorized_actions,
                    "position_profile": args.position_profile,
                    "position_profile_hash": (
                        resolved.position_profile.to_metadata()["sha256"]
                    ),
                    "collision_scene_profile": args.collision_scene_profile,
                    "lower_profile_hash": resolved.semantic_content_hash,
                    "debug_motion_skill": args.debug_motion_skill,
                },
            )
        )
        try:
            _attach_motion_trace(composition, motion_trace)
        except Exception:
            motion_trace.close()
            raise

    ready = threading.Event()
    stop_requested = threading.Event()
    application_error: list[BaseException] = []

    def serve() -> None:
        try:
            composition.application.serve_forever(ready_callback=ready.set)
        except BaseException as error:  # noqa: BLE001 - surfaced to main.
            application_error.append(error)
        finally:
            stop_requested.set()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    application_thread = threading.Thread(
        target=serve,
        name="dexteleop-agent-gateway-real-rm65",
        daemon=False,
    )
    application_thread.start()
    stopped_payload: dict[str, Any] | None = None
    motion_trace_close_error: BaseException | None = None
    try:
        while not ready.wait(0.1):
            if application_error or not application_thread.is_alive():
                break
        if application_error:
            raise RuntimeError("real RM65 Skill application failed to start") from (
                application_error[0]
            )
        if not ready.is_set():
            raise RuntimeError("real RM65 Skill application stopped before ready")
        if (
            composition.application.state != ExclusiveRealSkillApplicationState.RUNNING
            or not application_thread.is_alive()
        ):
            raise RuntimeError("real RM65 Skill application lost readiness")
        print(json.dumps(_ready_payload(composition), sort_keys=True), flush=True)
        if stop_requested.is_set():
            composition.application.request_shutdown()
        while application_thread.is_alive():
            if stop_requested.wait(0.25):
                composition.application.request_shutdown()
                break
        application_thread.join(timeout=3.0)
        if application_thread.is_alive():
            raise RuntimeError("real RM65 Skill application did not stop")
        if application_error:
            raise RuntimeError("real RM65 Skill application failed") from (
                application_error[0]
            )
    finally:
        if application_thread.is_alive():
            composition.application.request_shutdown()
            application_thread.join(timeout=3.0)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        if motion_trace is not None and application_thread.is_alive():
            motion_trace.stop_accepting()
            motion_trace_close_error = RuntimeError(
                "lower application thread did not stop; motion trace was not "
                "detached or written"
            )
        elif motion_trace is not None:
            detach_errors = _detach_motion_trace(composition)
            if detach_errors:
                motion_trace_close_error = RuntimeError(
                    "motion trace detach failed: " + "; ".join(detach_errors)
                )
            try:
                motion_trace.close()
            except BaseException as error:  # noqa: BLE001 - report after HOLD proof.
                if motion_trace_close_error is None:
                    motion_trace_close_error = error
        stopped_payload = _stopped_payload(
            composition,
            motion_trace=motion_trace,
            motion_trace_close_error=motion_trace_close_error,
        )
        print(json.dumps(stopped_payload, sort_keys=True), flush=True)
    if stopped_payload is None or not stopped_payload["safe_shutdown"]:
        raise RuntimeError(
            "real RM65 Skill Gateway stopped without lower safety-stop proof"
        )
    if motion_trace_close_error is not None:
        raise RuntimeError("requested RM65 Skill motion trace was not completed") from (
            motion_trace_close_error
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_RM65_CAMERA_MOUNT_URDF",
    "DEFAULT_RM65_CAMERA_REGISTRY_V4",
    "build_parser",
    "main",
]

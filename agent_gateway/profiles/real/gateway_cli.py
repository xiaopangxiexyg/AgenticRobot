"""Dedicated default-off launcher for independent real RM65 Skills."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import threading
import uuid
from pathlib import Path
from typing import Any

from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_server_preflight,
)
from robot_motion.rm65 import RM65_BIMANUAL_POSITION_V1_NAME
from teleop_runtime.skill_deploy import (
    ExclusiveRM65SkillDeploymentProfile,
    assemble_exclusive_rm65_skill_lower,
    resolve_exclusive_rm65_skill_deployment_profile,
)

from .exclusive_application import ExclusiveRealSkillApplicationState
from .one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_ACTIONS,
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
)
from .rm65_composition import (
    ExclusiveRM65AgentPrimitiveComposition,
    assemble_exclusive_rm65_agent_primitive_composition,
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
        "--position-profile",
        default=RM65_BIMANUAL_POSITION_V1_NAME,
        help="trusted lower RM65 position-control profile name",
    )
    parser.add_argument(
        "--collision-scene-profile",
        default="dual_arm_hand_proxy_open",
        help="trusted lower coupled-IK/collision scene profile name",
    )
    parser.add_argument("--authorized-side", choices=("left", "right"), required=True)
    parser.add_argument(
        "--authorized-action",
        choices=REAL_AGENT_PRIMITIVE_ACTIONS,
        help="one reviewed primitive action (compatibility form)",
    )
    parser.add_argument(
        "--authorized-actions",
        choices=REAL_AGENT_PRIMITIVE_ACTIONS,
        nargs="+",
        help=(
            "one or more reviewed primitive actions allowed for this temporary "
            "test profile"
        ),
    )
    parser.add_argument(
        "--confirm",
        required=True,
        help=(
            "must exactly equal " + REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN
        ),
    )
    parser.add_argument("--client-id", default="rpent.dexteleop")
    parser.add_argument("--client-uid", type=int, default=os.getuid())
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
        "station_manifest_hash": manifest.station_manifest_hash,
        "execution_manifest_hash": manifest.execution_manifest_hash,
        "deployment_profile_hash": composition.lower.profile.semantic_content_hash,
        "authorized_side": composition.authorized_side,
        "authorized_action": composition.authorized_action,
        "authorized_actions": list(authorized_actions),
        "authorization_scope": (
            "one_matching_skill_request"
            if len(authorized_actions) == 1
            else "one_request_from_listed_primitive_actions"
        ),
        "perception": (
            None
            if perception_capture is None
            else {
                "rgb_roles": list(perception_capture.profile.rgb_roles),
                "depth_roles": list(perception_capture.profile.depth_roles),
                "control_effect_class": "read_only",
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
) -> dict[str, Any]:
    snapshot = composition.deployment.control_runtime.motion_port.snapshot()
    hold = composition.deployment.control_runtime.last_hold_receipt
    lower_hold_proved = bool(hold is not None and hold.succeeded)
    return {
        "event": "gateway_stopped",
        "application_state": composition.application.state.value,
        "motion_command_dispatched": snapshot.dispatch_count > 0,
        "dispatch_count": snapshot.dispatch_count,
        "lower_hold_proved": lower_hold_proved,
        "lower_hold_evidence_refs": (
            [] if hold is None else list(hold.evidence_refs)
        ),
        "safe_shutdown": lower_hold_proved,
        "safe_shutdown_basis": (
            "lower_hold_receipt"
            if lower_hold_proved
            else "missing_lower_hold_proof"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.confirm != REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN:
        parser.error(
            "--confirm must exactly equal "
            + REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN
        )
    if (args.authorized_action is None) == (args.authorized_actions is None):
        parser.error(
            "provide exactly one of --authorized-action or --authorized-actions"
        )
    if args.client_uid < 0:
        parser.error("--client-uid must be non-negative")
    if args.max_retained_operations <= 0:
        parser.error("--max-retained-operations must be positive")
    if not 0.0 < args.peer_idle_timeout_sec:
        parser.error("--peer-idle-timeout-sec must be positive")
    if not 0.0 < args.grant_lifetime_sec <= 30.0:
        parser.error("--grant-lifetime-sec must be in (0, 30]")

    socket_path = _absolute_socket_path(parser, args.socket_path)
    perception_supplied = tuple(
        value is not None
        for value in (args.camera_config, args.camera_registry, args.artifact_root)
    )
    if any(perception_supplied) and not all(perception_supplied):
        parser.error(
            "--camera-config, --camera-registry, and --artifact-root must be "
            "provided together"
        )
    try:
        validate_agent_gateway_unix_server_preflight(
            socket_path=socket_path,
            peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        )
    except (FileExistsError, TypeError, ValueError) as error:
        parser.error(f"--socket-path preflight failed: {error}")
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
    lower = assemble_exclusive_rm65_skill_lower(profile=resolved)
    camera_config = (
        None
        if args.camera_config is None
        else _absolute_existing_file(parser, args.camera_config, flag="--camera-config")
    )
    camera_registry = (
        None
        if args.camera_registry is None
        else _absolute_existing_file(
            parser,
            args.camera_registry,
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
    embodiment_instance_id = (
        args.embodiment_instance_id
        if args.embodiment_instance_id is not None
        else "dual-rm65:" + resolved.semantic_content_hash[:16]
    )
    composition = assemble_exclusive_rm65_agent_primitive_composition(
        lower=lower,
        socket_path=socket_path,
        authorized_side=args.authorized_side,
        authorized_action=args.authorized_action,
        authorized_actions=(
            None
            if args.authorized_actions is None
            else tuple(args.authorized_actions)
        ),
        confirmation_token=args.confirm,
        client_id=args.client_id,
        client_uid=args.client_uid,
        runtime_session_id=_runtime_session_id(args.runtime_session_id),
        control_domain_id=args.control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        max_retained_operations=args.max_retained_operations,
        peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        grant_lifetime_sec=args.grant_lifetime_sec,
        camera_config_path=camera_config,
        camera_registry_path=camera_registry,
        artifact_root=artifact_root,
    )

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
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
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
            composition.application.state
            != ExclusiveRealSkillApplicationState.RUNNING
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
        stopped_payload = _stopped_payload(composition)
        print(json.dumps(stopped_payload, sort_keys=True), flush=True)
    if stopped_payload is None or not stopped_payload["safe_shutdown"]:
        raise RuntimeError("real RM65 Skill Gateway stopped without lower HOLD proof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

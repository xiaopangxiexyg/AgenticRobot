"""Camera-only real RGB/depth Skill host; no RM65 backend is constructed."""

from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import threading
import uuid
from pathlib import Path

from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_group_parent,
    validate_agent_gateway_unix_server_preflight,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.rm65 import (
    build_rm65_dual_arm_scene_profile,
    load_world_from_arm_base_transforms,
)

from .perception_application import (
    CameraOnlyRealPerceptionApplicationState,
    assemble_camera_only_real_perception_composition,
)
from .realsense_perception import build_real_perception_source_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway real-perception",
        description=(
            "Serve passive real RGB/depth Skills from an exclusive RealSense "
            "inventory. This profile never constructs or connects an RM65 backend."
        ),
    )
    parser.add_argument("--socket-path", type=Path, required=True)
    parser.add_argument("--camera-config", type=Path, required=True)
    parser.add_argument("--camera-registry", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--collision-scene-profile",
        default="dual_arm_hand_proxy_open",
        help="trusted fixed arm-base geometry used only for static camera poses",
    )
    parser.add_argument("--client-id", default="rpent.dexteleop")
    parser.add_argument("--client-uid", type=int, default=os.getuid())
    parser.add_argument(
        "--ssh-read-only-client",
        action="append",
        nargs=2,
        default=[],
        metavar=("CLIENT_ID", "UID"),
    )
    parser.add_argument("--runtime-session-id", default=None)
    parser.add_argument(
        "--control-domain-id",
        default="real-perception-only",
    )
    parser.add_argument("--max-retained-operations", type=int, default=16)
    parser.add_argument("--peer-idle-timeout-sec", type=float, default=10.0)
    return parser


def _absolute_existing_file(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        parser.error(f"{flag} must be an absolute path")
    resolved = path.resolve()
    if not resolved.is_file():
        parser.error(f"{flag} is not a file: {resolved}")
    return resolved


def _absolute_directory(
    parser: argparse.ArgumentParser,
    value: Path,
    *,
    flag: str,
) -> Path:
    path = Path(os.path.abspath(value.expanduser()))
    if not value.expanduser().is_absolute():
        parser.error(f"{flag} must be an absolute path")
    try:
        info = os.lstat(path)
    except OSError:
        parser.error(f"{flag} is not an existing directory: {path}")
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        parser.error(f"{flag} must be a real directory, not a symlink: {path}")
    if info.st_mode & stat.S_IWOTH:
        parser.error(f"{flag} cannot be world-writable: {path}")
    return path


def _runtime_session_id(value: str | None) -> str:
    return value or "dexteleop-real-perception:" + uuid.uuid4().hex


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    socket_path = Path(os.path.abspath(args.socket_path.expanduser()))
    if not args.socket_path.expanduser().is_absolute():
        parser.error("--socket-path must be an absolute path")
    try:
        endpoint_value, peer_timeout = validate_agent_gateway_unix_server_preflight(
            socket_path=socket_path,
            peer_idle_timeout_sec=args.peer_idle_timeout_sec,
        )
    except (OSError, TypeError, ValueError) as error:
        parser.error(f"invalid Unix endpoint: {error}")
    camera_config = _absolute_existing_file(
        parser,
        args.camera_config,
        flag="--camera-config",
    )
    camera_registry = _absolute_existing_file(
        parser,
        args.camera_registry,
        flag="--camera-registry",
    )
    artifact_root = _absolute_directory(
        parser,
        args.artifact_root,
        flag="--artifact-root",
    )
    read_only_clients = tuple(
        (str(client_id), int(uid_text))
        for client_id, uid_text in args.ssh_read_only_client
    )
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id=args.client_id,
        primary_client_uid=args.client_uid,
        ssh_read_only_clients=read_only_clients,
    )
    if client_bindings.socket_mode == 0o660:
        validate_agent_gateway_unix_group_parent(socket_path=endpoint_value)
    scene_profile = build_rm65_dual_arm_scene_profile(args.collision_scene_profile)
    world_from_base = load_world_from_arm_base_transforms(scene_profile)
    scene_geometry_hash = canonical_content_hash(
        {
            "collision_scene_profile": args.collision_scene_profile,
            "world_from_arm_base_by_side": {
                side: world_from_base[side].tolist() for side in ("left", "right")
            },
        },
        domain="dexteleop.real_perception.scene_geometry.v1",
    )
    source_binding = build_real_perception_source_binding(
        camera_config_path=camera_config,
        camera_registry_path=camera_registry,
        scene_world_from_arm_base_by_side=world_from_base,
    )
    composition = assemble_camera_only_real_perception_composition(
        source_binding=source_binding,
        socket_path=endpoint_value,
        artifact_root=artifact_root,
        client_bindings=client_bindings,
        runtime_session_id=_runtime_session_id(args.runtime_session_id),
        control_domain_id=args.control_domain_id,
        scene_geometry_hash=scene_geometry_hash,
        max_retained_operations=args.max_retained_operations,
        peer_idle_timeout_sec=peer_timeout,
    )
    ready = threading.Event()
    stop_requested = threading.Event()
    errors: list[BaseException] = []

    def ready_callback() -> None:
        manifest = composition.contract_bundle.route_manifest
        print(
            json.dumps(
                {
                    "event": "gateway_ready",
                    "profile": "real-perception",
                    "robot_backend": "absent",
                    "socket_path": str(composition.socket_path),
                    "runtime_session_id": manifest.runtime_session_id,
                    "current_mode": manifest.current_mode,
                    "route_manifest_hash": manifest.semantic_content_hash,
                    "deployment_profile_hash": composition.deployment_profile_hash,
                    "rgb_roles": list(composition.registration.profile.rgb_roles),
                    "depth_roles": list(composition.registration.profile.depth_roles),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        ready.set()

    def serve() -> None:
        try:
            composition.application.serve_forever(ready_callback=ready_callback)
        except BaseException as error:  # noqa: BLE001 - surfaced to the main thread.
            errors.append(error)
        finally:
            stop_requested.set()

    def request_stop(_signum: int, _frame: object) -> None:
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    thread = threading.Thread(
        target=serve,
        name="dexteleop-agent-gateway-real-perception",
        daemon=False,
    )
    thread.start()
    try:
        while not ready.wait(0.1):
            if errors or not thread.is_alive():
                break
        if errors:
            raise RuntimeError("real perception Gateway failed to start") from errors[0]
        if not ready.is_set():
            raise RuntimeError("real perception Gateway stopped before ready")
        while thread.is_alive():
            if stop_requested.wait(0.25):
                composition.application.request_shutdown()
                break
        thread.join(timeout=3.0)
        if thread.is_alive():
            raise RuntimeError("real perception Gateway did not stop")
        if errors:
            raise RuntimeError("real perception Gateway failed") from errors[0]
    finally:
        if thread.is_alive():
            composition.application.request_shutdown()
            thread.join(timeout=3.0)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        print(
            json.dumps(
                {
                    "event": "gateway_stopped",
                    "profile": "real-perception",
                    "state": composition.application.state.value,
                    "robot_backend": "absent",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if composition.application.state != CameraOnlyRealPerceptionApplicationState.CLOSED:
        raise RuntimeError("real perception Gateway did not close cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

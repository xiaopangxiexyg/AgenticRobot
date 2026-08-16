"""DexTeleop-owned MuJoCo launcher for the generic Agent TCP-pose Gateway.

This application leaf owns simulator preparation and the local Gateway server.
It does not embed RPent, an Agent planner, or an upper-level ProfileRegistry.
The same Gateway protocol remains the attachment boundary for later real
deployments; only this lower-owned launcher and its published manifest change.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
from pathlib import Path
from typing import Any

from project_paths import repo_root
from sim.mujoco.deploy import MujocoSimHILTargetConfig

from .motion_factory import (
    MujocoAgentTcpPoseHistoryStorageConfig,
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m agent_gateway mujoco",
        description=(
            "Serve the generic right-arm TCP-pose Agent Gateway on MuJoCo. "
            "This starts no Agent planner and connects no real device."
        ),
    )
    parser.add_argument(
        "--socket-path",
        type=Path,
        required=True,
        help=(
            "absolute Unix socket path; its real, non-world-writable parent "
            "must already exist and an existing endpoint is never replaced"
        ),
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        required=True,
        help="absolute MuJoCo task-config JSON path",
    )
    parser.add_argument(
        "--client-id",
        default="rpent.dexteleop",
        help="exact authenticated client identity advertised by this deployment",
    )
    parser.add_argument(
        "--client-uid",
        type=int,
        default=os.getuid(),
        help="local Unix peer UID authorized for --client-id (default: current UID)",
    )
    parser.add_argument(
        "--runtime-session-id",
        default="dexteleop-agent-mujoco-tcp-pose",
        help="lower-owned runtime session identity published in the manifest",
    )
    parser.add_argument(
        "--control-domain-id",
        default="mujoco-agent-tcp-pose",
        help="lower-owned control-domain identity published in the manifest",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        help=(
            "optional absolute disk-primary planning-history directory; "
            "omitting it removes the history Route"
        ),
    )
    parser.add_argument(
        "--history-filename",
        default="planning_history.sqlite3",
        help="SQLite history basename used only with --history-root",
    )
    parser.add_argument(
        "--max-retained-operations",
        type=int,
        default=16,
        help="bounded lower operation-retention capacity",
    )
    parser.add_argument(
        "--peer-idle-timeout-sec",
        type=float,
        default=10.0,
        help="authenticated Unix connection idle timeout",
    )
    parser.add_argument(
        "--no-realtime-worker",
        action="store_true",
        help="run MuJoCo stepping without realtime pacing (simulation only)",
    )
    return parser


def _absolute_existing_file(parser: argparse.ArgumentParser, value: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error("--task-config must be an absolute path")
    resolved = expanded.resolve()
    if not resolved.is_file():
        parser.error(f"--task-config is not a file: {resolved}")
    return resolved


def _absolute_path(parser: argparse.ArgumentParser, value: Path, *, flag: str) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        parser.error(f"{flag} must be an absolute path")
    # Normalize ``..`` without resolving symlinks. The transport must inspect
    # the caller-selected endpoint parent itself instead of receiving a path
    # whose symlink identity was erased by this CLI.
    return Path(os.path.abspath(expanded))


def _ready_payload(*, session: Any, socket_path: Path, client_id: str) -> dict[str, Any]:
    manifest = session.contract_bundle.route_manifest
    return {
        "event": "gateway_ready",
        "socket_path": str(socket_path),
        "client_id": client_id,
        "target_kind": manifest.target_kind,
        "current_mode": manifest.current_mode,
        "runtime_session_id": manifest.runtime_session_id,
        "control_domain_id": manifest.control_domain_id,
        "target_binding_id": manifest.target_binding_id,
        "route_manifest_hash": manifest.semantic_content_hash,
        "station_manifest_hash": manifest.station_manifest_hash,
        "execution_manifest_hash": manifest.execution_manifest_hash,
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


def _stopped_payload(*, server: Any, session: Any) -> dict[str, Any]:
    outcome = server.gateway_close_outcome
    receipt = None if outcome is None else outcome.port_receipt
    lower_hold_proved = bool(
        receipt is not None
        and getattr(receipt, "succeeded", False)
        and getattr(receipt, "proves_motion_or_hold", False)
    )
    motion_ever_activated = bool(session.application.started)
    safe_shutdown = not motion_ever_activated or lower_hold_proved
    return {
        "event": "gateway_stopped",
        "gateway_close_error": (
            None
            if server.gateway_close_error is None
            else type(server.gateway_close_error).__name__
        ),
        "close_accepted": None if outcome is None else outcome.accepted,
        "close_reason_code": None if outcome is None else outcome.reason_code,
        "motion_ever_activated": motion_ever_activated,
        "lower_hold_proved": lower_hold_proved,
        "safe_shutdown": safe_shutdown,
        "safe_shutdown_basis": (
            "lower_hold_proved"
            if lower_hold_proved
            else "motion_never_activated"
            if not motion_ever_activated
            else "missing_lower_hold_proof"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task_config = _absolute_existing_file(parser, args.task_config)
    socket_path = _absolute_path(parser, args.socket_path, flag="--socket-path")
    if args.client_uid < 0:
        parser.error("--client-uid must be non-negative")
    if args.max_retained_operations <= 0:
        parser.error("--max-retained-operations must be positive")
    history_storage = None
    if args.history_root is not None:
        history_storage = MujocoAgentTcpPoseHistoryStorageConfig(
            root=_absolute_path(parser, args.history_root, flag="--history-root"),
            filename=args.history_filename,
        )

    # Task configs may contain repository-relative calibration/profile paths.
    # This dedicated process owns its working directory, so normalize it once
    # instead of making an RPent caller know DexTeleop's repository layout.
    os.chdir(repo_root())

    target_config = MujocoSimHILTargetConfig(
        task_config_path=task_config,
        realtime_worker=not args.no_realtime_worker,
        runtime_session_id=args.runtime_session_id,
        control_domain_id=args.control_domain_id,
        execution_observation_plan=(
            mujoco_agent_tcp_pose_execution_observation_plan_v1()
        ),
        exposed_pose_sides=("right",),
        allow_scripted_task_reset=False,
    )
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=target_config,
        allowed_client_ids=(args.client_id,),
        max_retained_operations=args.max_retained_operations,
        history_storage=history_storage,
    )
    server = session.build_unix_json_server(
        socket_path=str(socket_path),
        client_id_by_uid={args.client_uid: args.client_id},
        peer_idle_timeout_sec=args.peer_idle_timeout_sec,
    )
    stop_requested = threading.Event()
    server_error: list[BaseException] = []
    stopped_payload: dict[str, Any] | None = None

    def serve() -> None:
        try:
            server.serve_forever(poll_interval=0.1)
        except BaseException as error:  # noqa: BLE001 - surfaced to main.
            server_error.append(error)
            stop_requested.set()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, request_stop)
    server_thread = threading.Thread(
        target=serve,
        name="dexteleop-agent-gateway-mujoco",
        daemon=True,
    )
    server_thread_started = False
    try:
        server_thread.start()
        server_thread_started = True
        if session.history_recorder is not None:
            session.history_recorder.capture_once()
            session.history_recorder.start()
        print(
            json.dumps(
                _ready_payload(
                    session=session,
                    socket_path=socket_path,
                    client_id=args.client_id,
                ),
                sort_keys=True,
            ),
            flush=True,
        )
        while server_thread.is_alive() and not stop_requested.wait(0.25):
            if (
                session.history_recorder is not None
                and session.history_recorder.last_error is not None
            ):
                raise RuntimeError("planning history recorder failed") from (
                    session.history_recorder.last_error
                )
        if server_error:
            raise RuntimeError("Agent Gateway server failed") from server_error[0]
    finally:
        if server_thread_started:
            server.shutdown()
        server.server_close()
        if server_thread_started:
            server_thread.join(timeout=2.0)
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        stopped_payload = _stopped_payload(server=server, session=session)
        print(json.dumps(stopped_payload, sort_keys=True), flush=True)
    if server.gateway_close_error is not None:
        raise RuntimeError("Agent Gateway close failed") from server.gateway_close_error
    if stopped_payload is None or not stopped_payload["safe_shutdown"]:
        raise RuntimeError("Agent Gateway stopped without a safe lower boundary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]

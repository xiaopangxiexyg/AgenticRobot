"""Read-only RM65 SDK/FK host. No motion backend, cameras, HOLD or STOP."""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import uuid
from pathlib import Path

from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_server_preflight,
)
from teleop_runtime.skill_deploy.rm65_profile import (
    ExclusiveRM65SkillDeploymentProfile,
    resolve_exclusive_rm65_skill_deployment_profile,
)
from teleop_runtime.skill_deploy.rm65_readonly import (
    RM65ReadonlyTcpSampler,
)

from .observation_gateway import assemble_real_tcp_pose_observe_gateway_session
from .readonly_tcp import ReadonlyRM65TcpSource, build_readonly_rm65_tcp_contract


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("socket-path", "arm-ip-assignment", "left-tool-profile", "right-tool-profile", "envelope-profile"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--collision-scene-profile", default="dual_arm_hand_proxy_dmtac_strict")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-uid", type=int, default=os.getuid())
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path) and (not value.is_absolute() or value == Path("/")):
            parser.error(f"{name} requires an absolute path")
    validate_agent_gateway_unix_server_preflight(socket_path=str(args.socket_path))
    profile = resolve_exclusive_rm65_skill_deployment_profile(ExclusiveRM65SkillDeploymentProfile(
        arm_ip_assignment_path=args.arm_ip_assignment,
        tool_profile_path_by_side={"left": args.left_tool_profile, "right": args.right_tool_profile},
        envelope_profile_path=args.envelope_profile,
        collision_scene_profile=args.collision_scene_profile,
    ))
    # Reuse the reviewed profile for tool/URDF identity, but never apply its
    # motion envelope to observations or construct its lower control assembly.
    bundle = build_readonly_rm65_tcp_contract(profile, runtime_session_id="rm65-readonly:" + uuid.uuid4().hex)
    sampler = RM65ReadonlyTcpSampler(profile)
    source = ReadonlyRM65TcpSource(sampler, bundle)
    session = assemble_real_tcp_pose_observe_gateway_session(
        contract_bundle=bundle, source=source, allowed_client_ids=(args.client_id,)
    )
    stop = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stop.set()) for sig in (signal.SIGINT, signal.SIGTERM)}
    server = None
    thread = None
    errors = []
    cleanup_errors = []
    try:
        sampler.start()
        session.observation_producer.capture_and_publish(session.service)
        server = session.build_unix_json_server(
            socket_path=str(args.socket_path), client_id_by_uid={args.client_uid: args.client_id}
        )

        def serve():
            try:
                server.serve_forever(poll_interval=.1)
            except BaseException as error:  # noqa: BLE001 - re-raised on the lifecycle thread.
                errors.append(error)
            finally:
                stop.set()

        thread = threading.Thread(target=serve, name="rm65-readonly-gateway", daemon=False)
        thread.start()
        print(json.dumps({"event": "gateway_ready", "profile": "real-tcp-observe",
            "current_mode": "real_observe", "robot_backend": "readonly_sdk_sampler",
            "runtime_session_id": bundle.attachment.runtime_session_id,
            "component_ids": list(bundle.attachment.component_ids),
            "reference_frame_id": bundle.attachment.reference_frame_id,
            "command_owner_count": 0, "motion_enabled": False, "gripper_enabled": False}), flush=True)
        while not stop.wait(.1):
            _ = sampler.latest  # Surface a stopped/failed sampler even without clients.
        if errors:
            raise RuntimeError("Read-only Gateway server failed") from errors[0]
        return 0
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        def close_server():
            if server is not None:
                if thread is not None and thread.ident is not None:
                    server.shutdown()
                    thread.join(timeout=3)
                    if thread.is_alive():
                        raise RuntimeError("Read-only Gateway server did not stop")
                server.server_close()

        for close in (close_server, session.close, sampler.close):
            try:
                close()
            except Exception as error:  # noqa: BLE001 - collect failures without skipping other owned resources.
                cleanup_errors.append(f"{type(error).__name__}: {error}")
        print(json.dumps({"event": "gateway_stopped", "profile": "real-tcp-observe",
            "command_owner_count": 0, "motion_enabled": False, "gripper_enabled": False,
            "cleanup_complete": not cleanup_errors, "cleanup_errors": cleanup_errors}), flush=True)
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))


if __name__ == "__main__":
    raise SystemExit(main())

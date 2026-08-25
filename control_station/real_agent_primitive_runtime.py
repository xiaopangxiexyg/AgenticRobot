"""Default-off Skill-only application binding for real TCP primitives."""

from __future__ import annotations

import argparse
import os
import threading
import time
from pathlib import Path

from agent_gateway.profiles.real.motion_factory import (
    RealAgentPrimitiveRealTeleopAttachment,
    attach_real_agent_primitive_gateway,
)
from agent_gateway.profiles.real.motion_primitive_profiles import (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1,
    REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
)
from agent_gateway.profiles.real.observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
)
from agent_gateway.protocol.skill_catalog import SkillDeploymentSafetyEnvelopeRef
from agent_gateway.profiles.real.one_shot_authorization import (
    REAL_AGENT_PRIMITIVE_ACTIONS,
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
    RealAgentPrimitiveOneShotAuthorization,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph, CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from teleop_runtime.real_teleop.agent_motion_port import (
    RealAgentMotionPhase,
    RealAgentMotionPortConfig,
)
from teleop_runtime.real_teleop.components import RealTeleopComponents

_ALL_REAL_PRIMITIVE_ROUTE_IDS = (
    REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    *tuple(REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1.values()),
    *tuple(REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1.values()),
)


def add_real_agent_primitive_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    group = parser.add_argument_group(
        "default-off real Agent TCP primitive test",
    )
    group.add_argument(
        "--agent-primitive-skill-only",
        action="store_true",
        help="claim the RealTeleop command source exclusively for one Agent primitive",
    )
    group.add_argument(
        "--agent-primitive-socket-path",
        default="",
        help="absolute private Unix socket path for the reviewed one-shot client",
    )
    group.add_argument(
        "--agent-primitive-client-id",
        default="dexteleop-local-primitive-operator",
    )
    group.add_argument(
        "--agent-primitive-client-uid",
        type=int,
        default=os.getuid(),
    )
    group.add_argument(
        "--agent-primitive-authorized-side",
        choices=("left", "right"),
    )
    group.add_argument(
        "--agent-primitive-authorized-action",
        choices=REAL_AGENT_PRIMITIVE_ACTIONS,
    )
    group.add_argument(
        "--agent-primitive-confirm",
        default="",
        help="exact one-run operator confirmation token",
    )
    return parser


def validate_real_agent_primitive_arguments(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> None:
    if not bool(getattr(args, "agent_primitive_skill_only", False)):
        return
    if getattr(args, "agent_primitive_confirm", "") != (
        REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN
    ):
        parser.error(
            "--agent-primitive-skill-only requires --agent-primitive-confirm "
            + REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN
        )
    socket_path = str(getattr(args, "agent_primitive_socket_path", "") or "")
    if not socket_path or not Path(socket_path).is_absolute():
        parser.error(
            "--agent-primitive-socket-path must be an absolute path in an "
            "existing private directory"
        )
    if getattr(args, "agent_primitive_authorized_side", None) is None:
        parser.error("--agent-primitive-authorized-side is required")
    if getattr(args, "agent_primitive_authorized_action", None) is None:
        parser.error("--agent-primitive-authorized-action is required")
    if set(getattr(args, "active_sides", ())) != {"left", "right"}:
        parser.error(
            "Skill-only primitive requires --active-sides both: the existing "
            "coupled collision solver and final pre-IK TCP safety owner must "
            "observe both arms even though only the authorized side may move"
        )
    if not bool(getattr(args, "tcp_pose_envelope_active", False)):
        parser.error(
            "real Agent primitives require --tcp-pose-envelope-active"
        )
    if not str(getattr(args, "tcp_pose_envelope_profile", "") or "").strip():
        parser.error(
            "real Agent primitives require --tcp-pose-envelope-profile"
        )
    if bool(getattr(args, "disable_dual_arm_collision_avoidance", False)):
        parser.error("real Agent primitives refuse --no-collision")
    if str(getattr(args, "arm_control_mode", "")) != "position":
        parser.error(
            "real Agent primitives currently require the existing position "
            "control path"
        )
    if not bool(getattr(args, "position_force_guard", False)):
        parser.error("real Agent primitives require --position-force-guard")
    if any(
        bool(getattr(args, name, False))
        for name in (
            "position_admittance_active",
            "position_admittance_rotation_active",
            "position_admittance_combined_active",
        )
    ):
        parser.error(
            "the first real Agent primitive test refuses Active admittance overlays"
        )
    if len(os.fsencode(socket_path)) > 107:
        parser.error(
            "--agent-primitive-socket-path exceeds the Linux Unix-socket "
            "107-byte limit"
        )


def _actual_binding_hashes(
    *,
    args: argparse.Namespace,
    components: RealTeleopComponents,
    side: str,
    envelope_profile: object,
    frame_graph: CoordinateFrameGraph,
) -> dict[str, str]:
    calibration = getattr(components.session, "calibration", None)
    calibration_payload = (
        calibration.to_dict() if callable(getattr(calibration, "to_dict", None)) else {}
    )
    calibration_hash = canonical_content_hash(
        calibration_payload,
        domain="dexteleop.real_agent_motion.runtime_calibration.v1",
    )
    tool_hashes = dict(envelope_profile.tool_frame_binding_hash_by_side)
    tool_config_hash = canonical_content_hash(
        {"side": side, "tool_frame_binding_hash": tool_hashes[side]},
        domain="dexteleop.real_agent_motion.tool_config.v1",
    )
    execution_manifest_hash = canonical_content_hash(
        {
            "application": "control_station.real_teleop",
            "mode": "skill_only_real_tcp_primitive",
            "side": side,
            "arm_execution_profile": str(
                getattr(args, "arm_execution_profile", "") or ""
            ),
            "position_control_profile": str(
                getattr(args, "position_control_profile", "") or ""
            ),
            "collision_scene_profile": str(
                getattr(args, "collision_scene_profile", "") or ""
            ),
            "collision_avoidance_mode": str(
                getattr(args, "collision_avoidance_mode", "") or ""
            ),
            "envelope_profile_hash": envelope_profile.semantic_content_hash,
            "frame_graph_hash": frame_graph.semantic_content_hash,
        },
        domain="dexteleop.real_agent_motion.execution_manifest.v1",
    )
    command_backend_by_side = {
        "left": components.left_command_backend,
        "right": components.right_command_backend,
    }
    backend_inventory_hash = canonical_content_hash(
        {
            "authorized_side": side,
            "runtime_active_sides": list(components.active_sides),
            "command_backend_type_by_side": {
                backend_side: (
                    None
                    if backend is None
                    else f"{type(backend).__module__}.{type(backend).__qualname__}"
                )
                for backend_side, backend in command_backend_by_side.items()
            },
            "inactive_side_policy": "exact_q_hold_then_coupled_collision_recheck",
            "target_builder_type": (
                f"{type(components.dual_arm_retargeter).__module__}."
                f"{type(components.dual_arm_retargeter).__qualname__}"
            ),
            "dispatch_owner": "TeleopSession",
        },
        domain="dexteleop.real_agent_motion.backend_inventory.v1",
    )
    return {
        "calibration_hash": calibration_hash,
        "tool_config_hash": tool_config_hash,
        "execution_manifest_hash": execution_manifest_hash,
        "backend_inventory_hash": backend_inventory_hash,
    }


class RealAgentPrimitiveSkillOnlyExtension:
    """Application-leaf owner for one explicitly authorized primitive."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.side = str(args.agent_primitive_authorized_side)
        self.action = str(args.agent_primitive_authorized_action)
        self.client_id = str(args.agent_primitive_client_id)
        self.client_uid = int(args.agent_primitive_client_uid)
        self.socket_path = str(args.agent_primitive_socket_path)
        self._attachment: RealAgentPrimitiveRealTeleopAttachment | None = None
        self._server: object | None = None
        self._server_thread: threading.Thread | None = None
        self._authorization: RealAgentPrimitiveOneShotAuthorization | None = None
        self._started = False

    def attach_components(
        self,
        components: RealTeleopComponents,
    ) -> RealTeleopComponents:
        if set(components.active_sides) != {"left", "right"}:
            raise RuntimeError(
                "Skill-only primitive requires RealTeleop --active-sides both "
                "for the coupled collision and final pre-IK safety owner."
            )
        if not components.collision_guard_requested or (
            components.dual_arm_retargeter is None
        ):
            raise RuntimeError(
                "Skill-only primitive requires the existing coupled collision guard."
            )
        target_filter = getattr(
            components.dual_arm_retargeter,
            "pre_ik_target_safety_filter",
            None,
        )
        if target_filter is None or not bool(
            getattr(target_filter, "apply_to_target", False)
        ):
            raise RuntimeError(
                "Skill-only primitive requires an Active final pre-IK TCP envelope."
            )
        envelope_profile = getattr(target_filter, "profile", None)
        if envelope_profile is None or self.side not in set(
            getattr(envelope_profile, "enabled_sides", ())
        ):
            raise RuntimeError(
                "Active TCP envelope does not enable the authorized arm side."
            )
        execution_frame = CoordinateFrameSpec(
            frame_id="dual_arm_solver_world",
            semantic_parameters={
                "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
            },
        )
        frame_graph = CoordinateFrameGraph(
            graph_id="realteleop-agent-primitive-solver-world-v1",
            graph_version="1",
            frames=(execution_frame,),
            transforms=(),
        )
        hashes = _actual_binding_hashes(
            args=self.args,
            components=components,
            side=self.side,
            envelope_profile=envelope_profile,
            frame_graph=frame_graph,
        )
        component_id = f"{self.side}_arm"
        runtime_nonce = f"{os.getpid()}:{time.time_ns()}"
        runtime_session_id = "real-agent-primitive:" + canonical_content_hash(
            {"nonce": runtime_nonce, "side": self.side},
            domain="dexteleop.real_agent_motion.runtime_session.v1",
        )
        target_binding_id = "realteleop-tool-tcp:" + self.side
        attachment = RealTcpPoseObserveAttachmentSpec(
            runtime_session_id=runtime_session_id,
            control_domain_id="realteleop-agent-primitive:" + self.side,
            embodiment_instance_id="dual-rm65-realteleop:" + self.side,
            target_binding_id=target_binding_id,
            component_ids=(component_id,),
            station_manifest_hash=canonical_content_hash(
                {
                    "application": "control_station.real_teleop",
                    "skill_only": True,
                },
                domain="dexteleop.real_agent_motion.station_manifest.v1",
            ),
            execution_manifest_hash=hashes["execution_manifest_hash"],
            backend_inventory_hash=hashes["backend_inventory_hash"],
            source_binding_id="realteleop-hardware-qpos-fk:" + self.side,
            source_binding_hash=canonical_content_hash(
                {"source": "existing_realteleop_hardware_qpos_fk", "side": self.side},
                domain="dexteleop.real_agent_motion.source_binding.v1",
            ),
            observation_binding_id="realteleop-agent-measured-tcp:" + self.side,
            observation_binding_hash=canonical_content_hash(
                {"component_id": component_id, "frame": execution_frame.frame_id},
                domain="dexteleop.real_agent_motion.observation_binding.v1",
            ),
            observation_plan_hash=canonical_content_hash(
                {"mode": "existing_control_loop_measured_qpos_fk", "side": self.side},
                domain="dexteleop.real_agent_motion.observation_plan.v1",
            ),
            observation_capability_hash=canonical_content_hash(
                {"schema": "tool_tcp_pose_xyz_quat_xyzw", "side": self.side},
                domain="dexteleop.real_agent_motion.observation_capability.v1",
            ),
            pose_schema_hash=canonical_content_hash(
                {"component_id": component_id, "pose": "xyz_quat_xyzw"},
                domain="dexteleop.real_agent_motion.pose_schema.v1",
            ),
            reference_frame_id=execution_frame.frame_id,
            reference_frame_spec_hash=execution_frame.semantic_content_hash,
            frame_graph_hash=frame_graph.semantic_content_hash,
            tool_config_hash=hashes["tool_config_hash"],
            calibration_hash=hashes["calibration_hash"],
            clock_domain_id="python-time-time-wall-clock",
            producer_epoch=1,
        )
        side_config = envelope_profile.side_config_by_side[self.side]
        constraint_set_hash = canonical_content_hash(
            {
                "profile_hash": envelope_profile.semantic_content_hash,
                "side": self.side,
                "side_config": side_config.semantic_payload(),
            },
            domain="dexteleop.real_agent_motion.safety_constraint_set.v1",
        )
        envelope_ref = SkillDeploymentSafetyEnvelopeRef(
            object_id="tcp-pose-envelope:" + envelope_profile.profile_id,
            schema="dexteleop.deployment_safety_envelope.v1",
            content_hash=envelope_profile.semantic_content_hash,
            constraint_set_hash=constraint_set_hash,
            allowed_route_ids=_ALL_REAL_PRIMITIVE_ROUTE_IDS,
        )
        port_config = RealAgentMotionPortConfig(
            active_sides=(self.side,),
            component_id_by_side={self.side: component_id},
            execution_frame_id=execution_frame.frame_id,
            execution_frame_spec_hash=execution_frame.semantic_content_hash,
            exclusive_command_claim=True,
        )
        self._attachment = attach_real_agent_primitive_gateway(
            components,
            port_config=port_config,
            observation_attachment=attachment,
            frame_graph=frame_graph,
            primitive_reference_frame_id=execution_frame.frame_id,
            safety_envelope_refs=(envelope_ref,),
            allowed_client_ids=(self.client_id,),
        )
        return self._attachment.components

    def start(self, *, components: RealTeleopComponents) -> None:
        attachment = self._require_attachment()
        if components is not attachment.components:
            raise RuntimeError("RealTeleop did not retain the attached components.")
        authorization = RealAgentPrimitiveOneShotAuthorization(
            gateway_session=attachment.gateway_session,
            side=self.side,
            action=self.action,
            client_id=self.client_id,
            confirmation_token=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        )
        server = attachment.gateway_session.build_unix_json_server(
            socket_path=self.socket_path,
            client_id_by_uid={self.client_uid: self.client_id},
            peer_idle_timeout_sec=30.0,
            grant_resolver=authorization.grant_resolver,
        )
        thread = threading.Thread(
            target=server.serve_forever,
            kwargs={"poll_interval": 0.05},
            name="real-agent-primitive-gateway",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._server_thread = thread
        self._authorization = authorization
        self._started = True
        print(
            "[AgentPrimitive] Skill-only READY boundary installed; "
            f"authorized={self.side}:{self.action} socket={self.socket_path}. "
            "Vive command fallback is disabled."
        )

    def close_before_session(
        self,
        *,
        components: RealTeleopComponents,
        terminal_error: BaseException | None,
    ) -> None:
        del terminal_error
        if not self._started:
            return
        attachment = self._require_attachment()
        port = attachment.gateway_session.port
        snapshot = port.snapshot()
        now = time.time()
        if snapshot.phase not in {
            RealAgentMotionPhase.HOLD_REQUESTED,
            RealAgentMotionPhase.HOLD_APPLYING,
            RealAgentMotionPhase.HELD,
        }:
            if snapshot.operation_id is None:
                port.request_domain_hold(
                    reason="real_agent_primitive_runtime_shutdown",
                    requested_at=now,
                )
            else:
                port.request_hold(
                    operation_id=snapshot.operation_id,
                    reason="real_agent_primitive_runtime_shutdown",
                    requested_at=now,
                )
        snapshot = port.apply_owner_requests(session=components.session, now=now)
        if snapshot.phase != RealAgentMotionPhase.HELD or not snapshot.hold_succeeded:
            raise RuntimeError("Skill-only runtime shutdown did not confirm lower HOLD.")
        server = self._server
        thread = self._server_thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("Agent primitive Gateway thread did not stop.")
        self._started = False

    def _require_attachment(self) -> RealAgentPrimitiveRealTeleopAttachment:
        if self._attachment is None:
            raise RuntimeError("Real Agent primitive extension is not attached.")
        return self._attachment


def build_real_agent_primitive_extension(
    args: argparse.Namespace,
) -> RealAgentPrimitiveSkillOnlyExtension | None:
    if not bool(getattr(args, "agent_primitive_skill_only", False)):
        return None
    return RealAgentPrimitiveSkillOnlyExtension(args)


__all__ = [
    "REAL_AGENT_PRIMITIVE_ACTIONS",
    "REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN",
    "RealAgentPrimitiveSkillOnlyExtension",
    "add_real_agent_primitive_arguments",
    "build_real_agent_primitive_extension",
    "validate_real_agent_primitive_arguments",
]

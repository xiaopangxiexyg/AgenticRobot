from __future__ import annotations

import argparse
import time
from argparse import Namespace
from dataclasses import MISSING, fields
from types import SimpleNamespace

import numpy as np
import pytest

from control_station.real_agent_primitive_runtime import (
    REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
    RealAgentPrimitiveSkillOnlyExtension,
    validate_real_agent_primitive_arguments,
)
from robot_motion.runtime import EffectorCommand, TeleopCommand
from robot_motion.target_safety.tcp_pose_envelope import (
    TCPPoseEnvelopeProfile,
    TCPPoseEnvelopeSideConfig,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.components import RealTeleopComponents
from teleop_runtime.session import TeleopCommandOverride


class _Calibration:
    def to_dict(self):
        return {"fixture": "no-hardware"}


class _Session:
    def __init__(self) -> None:
        self.command_override_provider = None
        self.command_dispatch_admission = None
        self.calibration = _Calibration()
        self.state_machine = SimpleNamespace(state=TeleopRunState.ARMED)
        self.holds: list[tuple[str, bool]] = []

    def hold_hardware(self, reason, *, require_home_recalibration=False):
        self.holds.append((reason, require_home_recalibration))
        self.state_machine.state = TeleopRunState.DEBUG_ONLY
        return True


class _TargetBuilder:
    pre_ik_target_safety_filter = None

    def current_tool_tcp_world_targets(self):
        return {"left": np.eye(4), "right": np.eye(4)}

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        del side
        return np.asarray(tool_tcp_world_target, dtype=float).copy()

    def build_command_for_target_pair(self, target):
        command = TeleopCommand(timestamp=target.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.zeros(6),
            )
        )
        return command

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        del left_qpos, right_qpos
        return self.current_tool_tcp_world_targets()


def _components() -> RealTeleopComponents:
    session = _Session()
    target_builder = _TargetBuilder()
    profile = TCPPoseEnvelopeProfile(
        profile_id="agent-primitive-left-relative-tight-v1",
        target_frame_binding_hash="target-binding:test",
        tool_frame_binding_hash_by_side={"left": "tool-binding:test"},
        reference_frame_by_side={"left": "dual_arm_solver_world"},
        side_config_by_side={
            "left": TCPPoseEnvelopeSideConfig(
                relative_min_xyz_m=(-0.1, -0.1, -0.1),
                relative_max_xyz_m=(0.1, 0.1, 0.1),
                max_rotation_rad=np.deg2rad(45.0),
            )
        },
    )
    values = {}
    for item in fields(RealTeleopComponents):
        if item.default is MISSING and item.default_factory is MISSING:
            values[item.name] = None
    target_builder.pre_ik_target_safety_filter = SimpleNamespace(
        apply_to_target=True,
        profile=profile,
    )
    values.update(
        session=session,
        active_sides=("left", "right"),
        dual_arm_retargeter=target_builder,
        collision_guard_requested=True,
    )
    return RealTeleopComponents(**values)


def _args(socket_path: str) -> Namespace:
    return Namespace(
        agent_primitive_authorized_side="left",
        agent_primitive_authorized_action="forward",
        agent_primitive_client_id="dexteleop-local-primitive-operator",
        agent_primitive_client_uid=0,
        agent_primitive_socket_path=socket_path,
        arm_execution_profile="position_15_60_120hz",
        position_control_profile="rm65_bimanual_position_v1",
        collision_scene_profile="default",
        collision_avoidance_mode="active",
    )


def _validation_args(socket_path: str) -> Namespace:
    return Namespace(
        agent_primitive_skill_only=True,
        agent_primitive_confirm=REAL_AGENT_PRIMITIVE_CONFIRM_TOKEN,
        agent_primitive_socket_path=socket_path,
        agent_primitive_authorized_side="left",
        agent_primitive_authorized_action="up",
        active_sides=("left", "right"),
        tcp_pose_envelope_active=True,
        tcp_pose_envelope_profile="left-relative-tight.json",
        disable_dual_arm_collision_avoidance=False,
        arm_control_mode="position",
        position_force_guard=True,
        position_admittance_active=False,
        position_admittance_rotation_active=False,
        position_admittance_combined_active=False,
    )


def test_skill_only_argument_contract_accepts_reviewed_configuration(
    tmp_path,
) -> None:
    args = _validation_args(str(tmp_path / "primitive.sock"))

    validate_real_agent_primitive_arguments(
        args,
        argparse.ArgumentParser(),
    )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("active_sides", ("left",)),
        ("disable_dual_arm_collision_avoidance", True),
        ("position_force_guard", False),
        ("position_admittance_active", True),
    ),
)
def test_skill_only_argument_contract_rejects_unsafe_composition(
    tmp_path,
    field_name: str,
    invalid_value: object,
) -> None:
    args = _validation_args(str(tmp_path / "primitive.sock"))
    setattr(args, field_name, invalid_value)

    with pytest.raises(SystemExit):
        validate_real_agent_primitive_arguments(
            args,
            argparse.ArgumentParser(),
        )


def test_skill_only_extension_attaches_one_exclusive_lower_owner() -> None:
    original = _components()
    extension = RealAgentPrimitiveSkillOnlyExtension(_args("/tmp/agp-test.sock"))

    attached = extension.attach_components(original)
    port = attached.agent_motion_port

    assert original.agent_motion_port is None
    assert port is not None
    assert port.config.exclusive_command_claim is True
    idle = attached.session.command_override_provider(None, 1.0)
    assert isinstance(idle, TeleopCommandOverride)
    assert idle.command is None

    now = time.time() - 0.01
    port.publish_measured_pose(
        pose_world_by_component={"left_arm": np.eye(4)},
        measured_at=now,
        available_at=now + 0.001,
        source_evidence_refs=("fixture-measurement",),
    )
    first = port.publish_owner_state(session=attached.session, captured_at=now)
    second = port.publish_owner_state(
        session=attached.session,
        captured_at=now + 0.001,
    )
    assert first.lifecycle_state == "running"
    assert second.owner_state_revision == first.owner_state_revision + 1

    gateway = extension._require_attachment().gateway_session
    outcome = gateway.observation_producer.capture_and_publish(gateway.service)
    assert outcome.accepted is True
    assert outcome.snapshot is not None
    assert outcome.snapshot.execution_gate == "ready"


def test_skill_only_extension_shutdown_confirms_hold_before_session_close(
    short_unix_socket_path_factory,
) -> None:
    components = _components()
    socket_path = short_unix_socket_path_factory("primitive")
    extension = RealAgentPrimitiveSkillOnlyExtension(_args(str(socket_path)))
    attached = extension.attach_components(components)

    extension.start(components=attached)
    extension.close_before_session(components=attached, terminal_error=None)

    assert attached.agent_motion_port.snapshot().phase.value == "held"
    assert attached.session.holds == [
        (
            (
                "real Agent motion operation real-agent-domain-hold:2: "
                "real_agent_primitive_runtime_shutdown"
            ),
            False,
        )
    ]
    assert not socket_path.exists()

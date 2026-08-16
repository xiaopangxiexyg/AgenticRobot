from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.agent_motion_port import (
    RealAgentMotionActivation,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from teleop_runtime.session import TeleopCommandOverride


def _pose(*, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = (x, y, z)
    return pose


def _activation(*, expires_at: float = 2.0) -> RealAgentMotionActivation:
    return RealAgentMotionActivation(
        operation_id="operation-1",
        request_id="request-1",
        activation_grant_id="operator-grant-1",
        activation_grant_hash="grant-hash-1",
        component_id="right-tool-tcp",
        side="right",
        primitive_id="motion.step_tcp_cardinal.forward@1",
        primitive_hash="primitive-hash-1",
        target_pose_world=_pose(x=0.45, y=-0.10, z=0.35),
        execution_frame_id="dual-arm-solver-world",
        execution_frame_spec_hash="solver-world-spec-hash-1",
        measured_state_revision=7,
        measured_state_hash="measured-cut-hash-7",
        submitted_at=1.0,
        expires_at=expires_at,
    )


class _TargetBuilder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.targets = []

    def current_tool_tcp_world_targets(self):
        return {
            "left": _pose(x=0.25, y=0.20, z=0.30),
            "right": _pose(x=0.25, y=-0.20, z=0.30),
        }

    def build_command_for_target_pair(self, target):
        self.targets.append(target)
        if self.fail:
            raise RuntimeError("coupled IK rejected target")
        command = TeleopCommand(timestamp=target.timestamp)
        command.add_command(
            EffectorCommand(
                effector_name="right_arm",
                side="right",
                command_type="joint_vector",
                value=np.zeros(6, dtype=float),
            )
        )
        return command


class _Session:
    def __init__(self, state: TeleopRunState) -> None:
        self.state_machine = SimpleNamespace(state=state)
        self.holds = []

    def hold_hardware(self, reason, *, require_home_recalibration=False):
        self.holds.append((reason, require_home_recalibration))
        self.state_machine.state = TeleopRunState.DEBUG_ONLY
        return True


def _port(builder: _TargetBuilder | None = None) -> RealAgentMotionPort:
    return RealAgentMotionPort(
        config=RealAgentMotionPortConfig(
            active_sides=("right",),
            component_id_by_side={"right": "right-tool-tcp"},
            execution_frame_id="dual-arm-solver-world",
            execution_frame_spec_hash="solver-world-spec-hash-1",
        ),
        target_builder=builder or _TargetBuilder(),
    )


def test_owner_activation_builds_through_coupled_target_solver() -> None:
    builder = _TargetBuilder()
    port = _port(builder)
    pending = port.submit_activation(_activation())
    assert pending.phase == RealAgentMotionPhase.PENDING_ACTIVATION
    assert isinstance(
        port.command_for_observation(TeleopObservation(timestamp=1.1), 1.1),
        TeleopCommandOverride,
    )

    active = port.apply_owner_requests(
        session=_Session(TeleopRunState.ARMED),
        now=1.1,
    )
    assert active.phase == RealAgentMotionPhase.ACTIVE
    command = port.command_for_observation(
        TeleopObservation(timestamp=1.2),
        1.2,
    )

    assert isinstance(command, TeleopCommand)
    assert len(builder.targets) == 1
    target = builder.targets[0]
    assert target.active_sides == ("right",)
    np.testing.assert_allclose(
        target.target_pose_by_side["left"],
        _pose(x=0.25, y=0.20, z=0.30),
    )
    np.testing.assert_allclose(
        target.target_pose_by_side["right"],
        _activation().target_pose_world,
    )
    assert port.admit_command(
        command,
        TeleopObservation(timestamp=1.2),
        1.2,
    )
    observed = port.observe_session_step(
        command=command,
        command_dispatched=True,
    )
    assert observed.dispatch_count == 1
    assert observed.last_command_at == 1.2


def test_hold_request_wins_admission_race_and_is_applied_by_owner() -> None:
    port = _port()
    session = _Session(TeleopRunState.ARMED)
    port.submit_activation(_activation())
    port.apply_owner_requests(session=session, now=1.1)
    command = port.command_for_observation(
        TeleopObservation(timestamp=1.2),
        1.2,
    )
    assert isinstance(command, TeleopCommand)

    requested = port.request_hold(
        operation_id="operation-1",
        reason="terminal target reached",
        requested_at=1.21,
    )
    assert requested.phase == RealAgentMotionPhase.HOLD_REQUESTED
    assert not port.admit_command(
        command,
        TeleopObservation(timestamp=1.2),
        1.2,
    )
    inhibited = port.command_for_observation(
        TeleopObservation(timestamp=1.22),
        1.22,
    )
    assert isinstance(inhibited, TeleopCommandOverride)
    assert inhibited.command is None

    held = port.apply_owner_requests(session=session, now=1.23)
    assert held.phase == RealAgentMotionPhase.HELD
    assert held.operation_id == "operation-1"
    assert held.hold_succeeded is True
    assert session.holds == [
        (
            "real Agent motion operation operation-1: terminal target reached",
            False,
        )
    ]
    assert port.command_for_observation(
        TeleopObservation(timestamp=1.24),
        1.24,
    ) is None


def test_pending_activation_expires_without_motion_or_hold() -> None:
    port = _port()
    session = _Session(TeleopRunState.DEBUG_ONLY)
    port.submit_activation(_activation(expires_at=1.1))

    snapshot = port.apply_owner_requests(session=session, now=1.2)

    assert snapshot.phase == RealAgentMotionPhase.FAULTED
    assert snapshot.operation_id == "operation-1"
    assert snapshot.fault == "activation expired before the owner became motion-ready"
    assert session.holds == []


def test_coupled_target_failure_is_latched_for_session_fail_hold() -> None:
    port = _port(_TargetBuilder(fail=True))
    port.submit_activation(_activation())
    port.apply_owner_requests(
        session=_Session(TeleopRunState.ARMED),
        now=1.1,
    )

    with pytest.raises(RuntimeError, match="coupled IK rejected target"):
        port.command_for_observation(
            TeleopObservation(timestamp=1.2),
            1.2,
        )

    snapshot = port.snapshot()
    assert snapshot.phase == RealAgentMotionPhase.FAULTED
    assert snapshot.operation_id == "operation-1"
    assert snapshot.fault is not None
    assert "coupled IK rejected target" in snapshot.fault

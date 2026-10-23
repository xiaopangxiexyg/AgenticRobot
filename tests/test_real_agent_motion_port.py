from __future__ import annotations

from dataclasses import MISSING, fields, replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.runtime import EffectorCommand, TeleopCommand, TeleopObservation
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.real_teleop.agent_motion_port import (
    RealAgentJointTrajectory,
    RealAgentMotionActivation,
    RealAgentMotionMeasuredPoseCut,
    RealAgentMotionOwnerIdentity,
    RealAgentMotionPhase,
    RealAgentMotionPort,
    RealAgentMotionPortConfig,
)
from teleop_runtime.real_teleop.agent_motion_runtime import (
    attach_prepared_real_agent_motion_port,
    attach_real_agent_motion_port,
    build_real_agent_motion_port,
)
from teleop_runtime.real_teleop.components import RealTeleopComponents
from teleop_runtime.real_teleop.control_loop import (
    _publish_real_agent_motion_measured_cut,
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
    def __init__(self, *, fail: bool = False, tcp_offset_m: float = 0.0) -> None:
        self.fail = fail
        self.tcp_offset_m = float(tcp_offset_m)
        self.targets = []

    def current_tool_tcp_world_targets(self):
        return {
            "left": _pose(x=0.25, y=0.20, z=0.30),
            "right": _pose(x=0.25, y=-0.20, z=0.30),
        }

    def solver_world_target_from_tool_tcp_world_target(
        self,
        *,
        side,
        tool_tcp_world_target,
    ):
        del side
        tool_tcp = np.asarray(tool_tcp_world_target, dtype=float)
        tool_tcp_from_solver = np.eye(4, dtype=float)
        tool_tcp_from_solver[2, 3] = self.tcp_offset_m
        return tool_tcp @ np.linalg.inv(tool_tcp_from_solver)

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

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        left = _pose(x=float(np.asarray(left_qpos)[0]), y=0.20, z=0.30)
        right = _pose(x=float(np.asarray(right_qpos)[0]), y=-0.20, z=0.30)
        return {"left": left, "right": right}


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


@pytest.mark.parametrize("side", ["left", "right"])
def test_dual_authorized_port_keeps_unrequested_arm_out_of_ik(side):
    class Builder(_TargetBuilder):
        def build_command_for_target_pair(self, target):
            self.targets.append(target)
            command = TeleopCommand(timestamp=target.timestamp)
            selected = target.active_sides[0]
            command.add_command(EffectorCommand(effector_name=selected+"_arm", side=selected,
                command_type="joint_vector", value=np.zeros(6)))
            return command
    builder = Builder()
    port = RealAgentMotionPort(config=RealAgentMotionPortConfig(
        active_sides=("left", "right"),
        component_id_by_side={"left": "left-tool-tcp", "right": "right-tool-tcp"},
        execution_frame_id="dual-arm-solver-world", execution_frame_spec_hash="solver-world-spec-hash-1"),
        target_builder=builder)
    activation = replace(_activation(), side=side, component_id=side+"-tool-tcp")
    forbidden = RealAgentJointTrajectory(time_from_start_sec=np.array([0., .1]),
        qpos_by_side={arm: np.array([[0.] * 6, [.1] * 6]) for arm in ("left", "right")},
        planner_evidence_refs=("test:both-arms-moving",))
    with pytest.raises(ValueError, match="unauthorized inactive arm"):
        port.submit_activation(replace(activation, joint_trajectory=forbidden))
    port.submit_activation(activation)
    other = "left" if side == "right" else "right"
    with pytest.raises(RuntimeError, match="busy"):
        port.submit_activation(replace(activation, operation_id="second-operation",
            side=other, component_id=other+"-tool-tcp"))
    port.apply_owner_requests(session=_Session(TeleopRunState.ARMED), now=1.1)
    port.command_for_observation(TeleopObservation(timestamp=1.2), 1.2)
    assert builder.targets[-1].active_sides == (side,)
    assert builder.targets[-1].hard_hold_inactive_sides is True
    np.testing.assert_array_equal(builder.targets[-1].target_pose_by_side[other],
                                  builder.current_tool_tcp_world_targets()[other])


def _components_for_attachment(
    *,
    session: object,
    target_builder: object,
    active_sides: tuple[str, ...] = ("right",),
) -> RealTeleopComponents:
    values = {}
    for item in fields(RealTeleopComponents):
        if item.default is MISSING and item.default_factory is MISSING:
            values[item.name] = None
    values.update(
        session=session,
        active_sides=active_sides,
        dual_arm_retargeter=target_builder,
    )
    return RealTeleopComponents(**values)


def test_port_rejects_target_builder_without_tcp_solver_conversion() -> None:
    with pytest.raises(TypeError, match="tool-TCP-to-solver-world"):
        _port(object())


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
    assert target.hard_hold_inactive_sides is True
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


def test_preplanned_joint_path_uses_control_clock_and_freezes_inactive_arm() -> None:
    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.1, 0.2]),
        qpos_by_side={
            "left": np.array([[1.0] * 6, [1.0] * 6, [1.0] * 6]),
            "right": np.array([[0.0] * 6, [0.1] * 6, [0.2] * 6]),
        },
        planner_evidence_refs=("urdf-fcl-path:test",),
    )
    # The certified grasp path must be consumed directly in Active control.
    # A dynamic target solve here can monopolize the process long enough to
    # starve the independent RM65 measurement/force-watchdog threads.
    builder = _TargetBuilder(fail=True)
    port = _port(builder)
    port.publish_measured_qpos_pair(
        qpos_by_side={
            "left": trajectory.qpos_by_side["left"][0],
            "right": trajectory.qpos_by_side["right"][0],
        },
        measured_at=1.0,
        available_at=1.0,
        source_evidence_refs=("hardware-qpos-cut:preplanned-start",),
    )
    port.submit_activation(replace(_activation(), joint_trajectory=trajectory))
    port.apply_owner_requests(session=_Session(TeleopRunState.ARMED), now=1.0)

    # Commit a regular control frame first.  A single 150 ms gap exercises the
    # anti-catch-up cap, not interpolation under the normal control clock.
    first_command = port.command_for_observation(
        TeleopObservation(timestamp=1.05), 1.05,
    )
    port.observe_session_step(command=first_command, command_dispatched=True)
    command = port.command_for_observation(
        TeleopObservation(timestamp=1.15),
        1.15,
    )

    assert isinstance(command, TeleopCommand)
    np.testing.assert_allclose(command.commands["left_arm"].value, 1.0)
    np.testing.assert_allclose(command.commands["right_arm"].value, 0.15)
    operation = command.metadata["real_agent_motion_operation"]
    assert operation["trajectory_sample_index"] == 1
    assert operation["trajectory_next_sample_index"] == 2
    assert operation["trajectory_interpolation_alpha"] == pytest.approx(0.5)
    assert operation["joint_trajectory_hash"] == trajectory.semantic_content_hash
    assert builder.targets == []


@pytest.mark.parametrize("skip_delayed_frame", [False, True])
def test_preplanned_joint_path_rebases_after_a_gap_without_catching_up(
    skip_delayed_frame,
) -> None:
    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 2.0]),
        qpos_by_side={
            "left": np.ones((2, 6)),
            "right": np.array([[0.0] * 6, [2.0] * 6]),
        },
        planner_evidence_refs=("urdf-fcl-path:clock-gap-test",),
    )
    port = _port(_TargetBuilder(fail=True))
    port.publish_measured_qpos_pair(
        qpos_by_side={side: trajectory.qpos_by_side[side][0] for side in ("left", "right")},
        measured_at=1.0, available_at=1.0,
        source_evidence_refs=("hardware-qpos-cut:clock-gap-test",),
    )
    port.submit_activation(replace(_activation(expires_at=5.0), joint_trajectory=trajectory))
    port.apply_owner_requests(session=_Session(TeleopRunState.ARMED), now=1.0)
    first = port.command_for_observation(TeleopObservation(timestamp=1.05), 1.05)
    port.observe_session_step(command=first, command_dispatched=True)

    delayed = port.command_for_observation(TeleopObservation(timestamp=1.5), 1.5)
    assert delayed.metadata["real_agent_motion_operation"]["trajectory_clock_advance_capped"]
    np.testing.assert_allclose(delayed.commands["right_arm"].value, 0.15)
    port.observe_session_step(command=delayed, command_dispatched=not skip_delayed_frame)

    resumed = port.command_for_observation(TeleopObservation(timestamp=1.51), 1.51)
    # Resume from the last dispatched sample, not from skipped wall-clock time.
    np.testing.assert_allclose(
        resumed.commands["right_arm"].value, 0.06 if skip_delayed_frame else 0.16,
    )
    np.testing.assert_allclose(resumed.commands["left_arm"].value, 1.0)


def test_preplanned_joint_path_rechecks_start_at_owner_activation() -> None:
    initial = {
        "left": np.ones(6, dtype=float),
        "right": np.zeros(6, dtype=float),
    }
    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.1]),
        qpos_by_side={
            side: np.stack([qpos, qpos]) for side, qpos in initial.items()
        },
        planner_evidence_refs=("urdf-fcl-path:activation-race",),
    )
    port = _port()
    port.publish_measured_qpos_pair(
        qpos_by_side=initial,
        measured_at=1.0,
        available_at=1.0,
        source_evidence_refs=("hardware-qpos-cut:planned-start",),
    )
    port.submit_activation(replace(_activation(), joint_trajectory=trajectory))
    moved = {side: qpos.copy() for side, qpos in initial.items()}
    moved["right"][0] += 0.02
    port.publish_measured_qpos_pair(
        qpos_by_side=moved,
        measured_at=1.01,
        available_at=1.01,
        source_evidence_refs=("hardware-qpos-cut:changed-before-activation",),
    )
    session = _Session(TeleopRunState.ARMED)

    snapshot = port.apply_owner_requests(session=session, now=1.02)

    assert snapshot.phase == RealAgentMotionPhase.HELD
    assert snapshot.dispatch_count == 0
    assert snapshot.hold_succeeded is True
    assert snapshot.hold_reason == (
        "preplanned trajectory start changed before activation"
    )
    assert len(session.holds) == 1


def test_preplanned_joint_path_cannot_move_unauthorized_inactive_arm() -> None:
    trajectory = RealAgentJointTrajectory(
        time_from_start_sec=np.array([0.0, 0.1]),
        qpos_by_side={
            "left": np.array([[0.0] * 6, [0.01] * 6]),
            "right": np.array([[0.0] * 6, [0.01] * 6]),
        },
        planner_evidence_refs=("urdf-fcl-path:invalid-inactive-motion",),
    )

    with pytest.raises(ValueError, match="unauthorized inactive arm"):
        _port().submit_activation(
            replace(_activation(), joint_trajectory=trajectory)
        )


def test_owner_activation_converts_tcp_target_before_solver_boundary() -> None:
    builder = _TargetBuilder(tcp_offset_m=0.14)
    port = _port(builder)
    pending = port.submit_activation(_activation())
    assert pending.phase == RealAgentMotionPhase.PENDING_ACTIVATION
    port.apply_owner_requests(
        session=_Session(TeleopRunState.ARMED),
        now=1.1,
    )

    command = port.command_for_observation(
        TeleopObservation(timestamp=1.2),
        1.2,
    )
    assert isinstance(command, TeleopCommand)
    assert len(builder.targets) == 1
    target = builder.targets[0]

    expected_solver_target = _activation().target_pose_world.copy()
    expected_solver_target[2, 3] -= 0.14
    expected_inactive_solver_target = _pose(x=0.25, y=0.20, z=0.30)
    expected_inactive_solver_target[2, 3] -= 0.14
    np.testing.assert_allclose(
        target.target_pose_by_side["left"],
        expected_inactive_solver_target,
    )
    np.testing.assert_allclose(
        target.target_pose_by_side["right"],
        expected_solver_target,
    )
    np.testing.assert_allclose(
        target.source_by_side["right"].value,
        _activation().target_pose_world,
    )
    assert target.source_by_side["right"].metadata["controlled_frame_id"] == (
        "tool_tcp"
    )
    assert target.source_by_side["right"].metadata[
        "solver_target_pose_semantics"
    ] == "solver_world_endpoint"


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


def test_attachment_composes_existing_admission_and_remains_explicit() -> None:
    admission_calls = []

    def existing_admission(command, observation, now):
        del command, observation, now
        admission_calls.append("existing")
        return True

    session = SimpleNamespace(
        command_override_provider=None,
        command_dispatch_admission=existing_admission,
    )
    target_builder = _TargetBuilder()
    original = _components_for_attachment(
        session=session,
        target_builder=target_builder,
    )
    assert original.agent_motion_port is None

    attached = attach_real_agent_motion_port(
        original,
        config=RealAgentMotionPortConfig(
            active_sides=("right",),
            component_id_by_side={"right": "right-tool-tcp"},
            execution_frame_id="dual-arm-solver-world",
            execution_frame_spec_hash="solver-world-spec-hash-1",
        ),
    )

    assert original.agent_motion_port is None
    assert attached.agent_motion_port is not None
    assert session.command_override_provider == (
        attached.agent_motion_port.command_for_observation
    )
    human_command = TeleopCommand()
    assert session.command_dispatch_admission(
        human_command,
        TeleopObservation(timestamp=1.0),
        1.0,
    )
    assert admission_calls == ["existing"]

    with pytest.raises(RuntimeError, match="already has an Agent motion port"):
        attach_real_agent_motion_port(
            attached,
            config=attached.agent_motion_port.config,
        )


def test_exclusive_single_side_port_can_own_coupled_dual_arm_runtime() -> None:
    session = SimpleNamespace(
        command_override_provider=None,
        command_dispatch_admission=None,
    )
    target_builder = _TargetBuilder()
    components = _components_for_attachment(
        session=session,
        target_builder=target_builder,
        active_sides=("left", "right"),
    )
    config = RealAgentMotionPortConfig(
        active_sides=("right",),
        component_id_by_side={"right": "right-tool-tcp"},
        execution_frame_id="dual-arm-solver-world",
        execution_frame_spec_hash="solver-world-spec-hash-1",
        exclusive_command_claim=True,
    )

    port = build_real_agent_motion_port(
        components,
        config=config,
        owner_identity=RealAgentMotionOwnerIdentity(
            runtime_session_id="real-session:test",
            control_domain_id="real-domain:test",
            embodiment_instance_id="rm65:test",
            target_binding_id="tool-tcp:test",
            execution_manifest_hash="execution-manifest:test",
            backend_inventory_hash="backend-inventory:test",
            clock_domain_id="wall-clock:test",
        ),
    )
    attached = attach_prepared_real_agent_motion_port(components, port=port)

    assert attached.agent_motion_port is not None
    assert attached.agent_motion_port.config.active_sides == ("right",)
    assert isinstance(
        session.command_override_provider(TeleopObservation(timestamp=1.0), 1.0),
        TeleopCommandOverride,
    )


def test_nonexclusive_single_side_port_cannot_claim_dual_arm_runtime() -> None:
    components = _components_for_attachment(
        session=SimpleNamespace(
            command_override_provider=None,
            command_dispatch_admission=None,
        ),
        target_builder=_TargetBuilder(),
        active_sides=("left", "right"),
    )

    with pytest.raises(ValueError, match="exclusive coupled dual-arm"):
        attach_real_agent_motion_port(
            components,
            config=RealAgentMotionPortConfig(
                active_sides=("right",),
                component_id_by_side={"right": "right-tool-tcp"},
                execution_frame_id="dual-arm-solver-world",
                execution_frame_spec_hash="solver-world-spec-hash-1",
                exclusive_command_claim=False,
            ),
        )


def test_control_thread_publishes_immutable_measured_pose_cut() -> None:
    port = _port()
    cut = port.publish_measured_pose(
        pose_world_by_component={
            "right-tool-tcp": _pose(x=0.30, y=-0.20, z=0.31),
        },
        measured_at=1.0,
        available_at=1.01,
        source_evidence_refs=("rm65-state-sample:right:42",),
    )

    assert isinstance(cut, RealAgentMotionMeasuredPoseCut)
    assert cut.revision == 0
    assert cut.pose_world_by_component["right-tool-tcp"].flags.writeable is False
    assert cut.evidence_refs[0] == "rm65-state-sample:right:42"
    assert cut.evidence_refs[1].startswith("real-agent-measured-pose:")
    assert port.latest_measured_pose() is cut

    with pytest.raises(ValueError, match="timestamp regressed"):
        port.publish_measured_pose(
            pose_world_by_component={
                "right-tool-tcp": _pose(x=0.31, y=-0.20, z=0.31),
            },
            measured_at=0.9,
            available_at=1.02,
            source_evidence_refs=("rm65-state-sample:right:43",),
        )


def test_idle_domain_hold_is_applied_without_an_agent_operation() -> None:
    port = _port()
    session = _Session(TeleopRunState.ARMED)
    requested = port.request_domain_hold(
        reason="gateway close",
        requested_at=1.0,
    )
    assert requested.phase == RealAgentMotionPhase.HOLD_REQUESTED
    assert requested.operation_id is None

    held = port.apply_owner_requests(session=session, now=1.01)

    assert held.phase == RealAgentMotionPhase.HELD
    assert held.operation_id is None
    assert held.hold_succeeded is True
    assert held.evidence_ref.startswith("real-agent-motion-port:")
    assert session.holds == [
        ("real Agent motion operation real-agent-domain-hold:1: gateway close", False)
    ]


def test_control_frame_publishes_only_existing_hardware_measurements() -> None:
    port = _port()
    components = SimpleNamespace(
        agent_motion_port=port,
        active_sides=("right",),
    )
    cut = _publish_real_agent_motion_measured_cut(
        components,
        hardware_state_by_side={
            "left": {},
            "right": {
                "measured_qpos": np.array([0.42, 0, 0, 0, 0, 0], dtype=float),
                "hardware_measurement_valid": True,
                "hardware_measurement_timestamp": 2.0,
                "hardware_measurement_sequence": 12,
            },
        },
        preview_qpos_by_side={
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        },
        available_at=2.01,
    )

    assert isinstance(cut, RealAgentMotionMeasuredPoseCut)
    np.testing.assert_allclose(
        cut.pose_world_by_component["right-tool-tcp"][:3, 3],
        np.array([0.42, -0.20, 0.30]),
    )
    assert cut.source_evidence_refs == (
        "realteleop-hardware-qpos:right:12:2.000000000",
    )

    missing = _publish_real_agent_motion_measured_cut(
        components,
        hardware_state_by_side={"left": {}, "right": {}},
        preview_qpos_by_side={
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        },
        available_at=2.02,
    )
    assert missing is None
    assert port.latest_measured_pose() is cut


def test_exclusive_port_blocks_human_fallback_and_publishes_fresh_owner_cuts() -> None:
    identity = RealAgentMotionOwnerIdentity(
        runtime_session_id="real-session:test",
        control_domain_id="real-domain:test",
        embodiment_instance_id="rm65:test",
        target_binding_id="tool-tcp:test",
        execution_manifest_hash="execution-manifest:test",
        backend_inventory_hash="backend-inventory:test",
        clock_domain_id="wall-clock:test",
    )
    port = RealAgentMotionPort(
        config=RealAgentMotionPortConfig(
            active_sides=("right",),
            component_id_by_side={"right": "right-tool-tcp"},
            execution_frame_id="dual-arm-solver-world",
            execution_frame_spec_hash="solver-world-spec-hash-1",
            exclusive_command_claim=True,
        ),
        target_builder=_TargetBuilder(),
        owner_identity=identity,
    )
    session = _Session(TeleopRunState.ARMED)

    idle = port.command_for_observation(TeleopObservation(timestamp=1.0), 1.0)
    assert isinstance(idle, TeleopCommandOverride)
    assert idle.command is None
    with pytest.raises(RuntimeError, match="not exclusively attached"):
        port.publish_owner_state(session=session, captured_at=1.0)

    evidence = port.confirm_exclusive_attachment()
    first = port.publish_owner_state(session=session, captured_at=1.0)
    second = port.publish_owner_state(session=session, captured_at=1.1)

    assert evidence in first.evidence_refs
    assert first.lifecycle_state == "running"
    assert first.execution_gate == "ready"
    assert second.owner_state_revision == first.owner_state_revision + 1
    assert second.captured_at == pytest.approx(1.1)
    with pytest.raises(ValueError, match="timestamp regressed"):
        port.publish_owner_state(session=session, captured_at=1.05)

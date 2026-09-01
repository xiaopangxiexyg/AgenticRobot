from __future__ import annotations

import numpy as np
import pytest

from robot_motion.runtime import TeleopObservation
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
)
from teleop_runtime.skill_deploy.control_runtime import (
    RealSkillMotionDispatchReceipt,
    RealSkillMotionHoldReceipt,
    RealSkillMotionObservationCut,
)
from teleop_runtime.skill_deploy.dmtac_shadow_ports import (
    DMTacRM65RegisteredTrajectoryPorts,
    DMTacShadowReplayPortProfile,
)
from teleop_runtime.skill_deploy.registered_trajectory import (
    RegisteredTrajectoryExecutionAuthority,
    RegisteredTrajectoryPointDispatchReceipt,
    RegisteredTrajectoryRealReplayStatus,
    capture_authoritative_shadow_state_from_backends,
    execute_registered_shadow_trajectory_with_ports,
    read_measured_gripper_state_cut,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _GripperBackend:
    def __init__(self, payload):
        self.payload = payload
        self.read_count = 0

    def read_gripper_state_now(self):
        self.read_count += 1
        return self.payload


def _identity(*, gripper_model_hash: str = "gripper:test") -> ShadowExecutionIdentity:
    return ShadowExecutionIdentity(
        station_revision="station:test",
        reference_frame_id="desk",
        reference_frame_spec_hash="frame-spec:test",
        frame_graph_hash="frame-graph:test",
        tool_profile_hash_by_side={"left": "tool:left", "right": "tool:right"},
        robot_model_hash="robot:test",
        collision_model_hash="collision:test",
        joint_layout_hash="joints:test",
        gripper_model_hash=gripper_model_hash,
    )


def _backend(side: str, close: float, timestamp: float):
    return _GripperBackend(
        {
            "joint_pos": close,
            "timestamp": timestamp,
            "gripper_state_source": "runtime_backend",
            "gripper_state_assumed": False,
            "side": side,
        }
    )


def test_fresh_dual_gripper_read_builds_authoritative_real_state() -> None:
    backends = {
        "left": _backend("left", 0.2, 9.91),
        "right": _backend("right", 0.7, 9.92),
    }
    arm_cut = RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=9.93),
        qpos_by_side={"left": np.zeros(6), "right": np.ones(6) * 0.1},
        measured_at=9.90,
        available_at=9.93,
        evidence_refs=("arm-cut:test",),
    )

    snapshot = capture_authoritative_shadow_state_from_backends(
        arm_observation_cut=arm_cut,
        gripper_backends_by_side=backends,
        identity=_identity(),
        clock=lambda: 9.94,
    )

    assert snapshot.gripper_close_by_side == {"left": 0.2, "right": 0.7}
    assert snapshot.gripper_measured_at_by_side == {"left": 9.91, "right": 9.92}
    assert snapshot.available_at == 9.94
    assert len(snapshot.gripper_evidence_refs) == 3
    assert all(backend.read_count == 1 for backend in backends.values())


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        (None, "fresh-state read is unavailable"),
        ({"joint_pos": 0.2}, "no measurement timestamp"),
        (
            {
                "joint_pos": 0.2,
                "timestamp": 9.9,
                "gripper_state_assumed": True,
            },
            "assumed, not measured",
        ),
        (
            {
                "joint_pos": 1.2,
                "timestamp": 9.9,
                "gripper_state_source": "runtime_backend",
            },
            "must be in \\[0, 1\\]",
        ),
    ),
)
def test_gripper_cut_rejects_missing_assumed_or_invalid_reads(payload, message) -> None:
    backends = {
        "left": _GripperBackend(payload),
        "right": _backend("right", 0.3, 9.9),
    }
    with pytest.raises((RuntimeError, ValueError), match=message):
        read_measured_gripper_state_cut(
            backends_by_side=backends,
            clock=lambda: 10.0,
        )


def test_gripper_cut_rejects_future_or_excessively_skewed_measurements() -> None:
    with pytest.raises(RuntimeError, match="in the future"):
        read_measured_gripper_state_cut(
            backends_by_side={
                "left": _backend("left", 0.2, 10.01),
                "right": _backend("right", 0.3, 9.99),
            },
            clock=lambda: 10.0,
        )
    with pytest.raises(RuntimeError, match="allowed time skew"):
        read_measured_gripper_state_cut(
            backends_by_side={
                "left": _backend("left", 0.2, 9.8),
                "right": _backend("right", 0.3, 9.9),
            },
            clock=lambda: 10.0,
            maximum_inter_side_skew_sec=0.05,
        )


def test_authoritative_state_rejects_arm_gripper_cross_source_skew() -> None:
    arm_cut = RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=9.5),
        qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
        measured_at=9.4,
        available_at=9.5,
        evidence_refs=("arm-cut:test",),
    )
    with pytest.raises(RuntimeError, match="arm and gripper measurements"):
        capture_authoritative_shadow_state_from_backends(
            arm_observation_cut=arm_cut,
            gripper_backends_by_side={
                "left": _backend("left", 0.2, 9.9),
                "right": _backend("right", 0.3, 9.91),
            },
            identity=_identity(),
            clock=lambda: 9.92,
            maximum_arm_gripper_skew_sec=0.1,
        )


def _snapshot(*, qpos_by_side=None, gripper_by_side=None, now=9.95):
    qpos = qpos_by_side or {"left": np.zeros(6), "right": np.zeros(6)}
    grippers = gripper_by_side or {"left": 0.2, "right": 0.3}
    return AuthoritativeRobotStateSnapshot(
        qpos_by_side=qpos,
        gripper_close_by_side=grippers,
        arm_measured_at=now - 0.02,
        gripper_measured_at_by_side={"left": now - 0.015, "right": now - 0.01},
        available_at=now,
        arm_evidence_refs=(f"arm:{now}",),
        gripper_evidence_refs=(f"gripper:{now}",),
        identity=_identity(),
    )


def _trajectory():
    initial = _snapshot()
    final = {side: value.copy() for side, value in initial.qpos_by_side.items()}
    final["right"][1] += 0.05
    return RegisteredShadowJointTrajectory(
        trajectory_id="trajectory:real-port-test",
        created_at=9.96,
        initial_snapshot=initial,
        scene_hash="scene:test",
        target_hash="target:test",
        planner_config_hash="planner:test",
        moving_sides=("right",),
        points=(
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.0,
                qpos_by_side=initial.qpos_by_side,
            ),
            RegisteredJointTrajectoryPoint(
                time_from_start_sec=0.04,
                qpos_by_side=final,
            ),
        ),
    )


def _simulation_receipt(trajectory, *, completed=True):
    return ShadowSimulationReceipt(
        trajectory_hash=trajectory.semantic_content_hash,
        initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
        identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=(
            ShadowSimulationStatus.COMPLETED
            if completed
            else ShadowSimulationStatus.PATH_COLLISION_BLOCKED
        ),
        executed_point_count=len(trajectory.points) if completed else 0,
        terminal_hold_confirmed=completed,
        exact_planned_qpos_execution_confirmed=completed,
        model_identity_confirmed=completed,
        evidence_refs=("mujoco:test",),
    )


class _ReplayPorts:
    def __init__(
        self,
        trajectory,
        *,
        dispatch_failure_index=None,
        dispatch_raise_index=None,
        update_qpos=True,
        gripper_drift_index=None,
        hold_succeeds=True,
        hold_raises=False,
        snapshot_raise_call=None,
    ):
        self.trajectory = trajectory
        self.wall = 10.0
        self.monotonic = 0.0
        self.qpos = {
            side: value.copy()
            for side, value in trajectory.initial_snapshot.qpos_by_side.items()
        }
        self.grippers = dict(trajectory.initial_snapshot.gripper_close_by_side)
        self.dispatch_failure_index = dispatch_failure_index
        self.dispatch_raise_index = dispatch_raise_index
        self.update_qpos = update_qpos
        self.gripper_drift_index = gripper_drift_index
        self.hold_succeeds = hold_succeeds
        self.hold_raises = hold_raises
        self.snapshot_raise_call = snapshot_raise_call
        self.snapshot_count = 0
        self.dispatched: list[str] = []
        self.hold_reasons: list[str] = []
        self.authority_consumptions = 0

    def wall_clock(self):
        return self.wall

    def monotonic_clock(self):
        return self.monotonic

    def sleep(self, duration):
        self.monotonic += float(duration)
        self.wall += float(duration)

    def snapshot(self):
        call_index = self.snapshot_count
        self.snapshot_count += 1
        if call_index == self.snapshot_raise_call:
            raise RuntimeError(f"snapshot failure at call {call_index}")
        return _snapshot(
            qpos_by_side=self.qpos,
            gripper_by_side=self.grippers,
            now=self.wall,
        )

    def dispatch(self, point, trajectory_hash):
        assert trajectory_hash == self.trajectory.semantic_content_hash
        index = len(self.dispatched)
        if index == self.dispatch_raise_index:
            raise RuntimeError(f"dispatch failure at point {index}")
        succeeded = index != self.dispatch_failure_index
        self.dispatched.append(point.semantic_content_hash)
        if succeeded and self.update_qpos:
            self.qpos = {
                side: value.copy() for side, value in point.qpos_by_side.items()
            }
        if succeeded and index == self.gripper_drift_index:
            self.grippers["right"] += 0.1
        return RegisteredTrajectoryPointDispatchReceipt(
            point_hash=point.semantic_content_hash,
            succeeded=succeeded,
            evidence_refs=(f"dispatch:{index}",) if succeeded else (),
        )

    def hold(self, reason):
        self.hold_reasons.append(reason)
        if self.hold_raises:
            raise RuntimeError("hold transport failed")
        return RealSkillMotionHoldReceipt(
            succeeded=self.hold_succeeds,
            evidence_refs=("real-hold:test",) if self.hold_succeeds else (),
        )

    def consume(self, authority):
        self.authority_consumptions += 1
        return (f"authority-consumed:{authority.authority_id}",)


def _authority(trajectory, simulation_receipt, *, trajectory_hash=None):
    return RegisteredTrajectoryExecutionAuthority(
        authority_id="authority:test",
        operator_confirmation_ref="operator-confirmation:test",
        trajectory_hash=trajectory_hash or trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation_receipt.semantic_content_hash,
        issued_at=9.99,
        expires_at=10.5,
    )


def _execute(ports, trajectory, simulation_receipt, **overrides):
    values = {
        "trajectory": trajectory,
        "simulation_receipt": simulation_receipt,
        "current_scene_hash": "scene:test",
        "current_snapshot_provider": ports.snapshot,
        "point_dispatcher": ports.dispatch,
        "hold_controller": ports.hold,
        "authority": _authority(trajectory, simulation_receipt),
        "consume_execution_authority": ports.consume,
        "execution_enabled": True,
        "wall_clock": ports.wall_clock,
        "monotonic_clock": ports.monotonic_clock,
        "sleep": ports.sleep,
        "final_tracking_timeout_sec": 0.06,
        "feedback_poll_sec": 0.01,
    }
    values.update(overrides)
    return execute_registered_shadow_trajectory_with_ports(**values)


def test_mock_real_ports_replay_exact_simulated_points_then_hold() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory)

    receipt = _execute(ports, trajectory, simulation)

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.COMPLETED
    assert ports.dispatched == [point.semantic_content_hash for point in trajectory.points]
    assert ports.authority_consumptions == 1
    assert receipt.executed_point_count == len(trajectory.points)
    assert receipt.terminal_hold_confirmed is True
    assert receipt.real_motion_executed is True


def test_mock_real_replay_is_default_disabled_even_with_valid_simulation() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory)

    receipt = _execute(
        ports,
        trajectory,
        simulation,
        authority=None,
        consume_execution_authority=None,
        execution_enabled=False,
    )

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.EXECUTION_DISABLED
    assert ports.dispatched == []
    assert ports.authority_consumptions == 0
    assert receipt.terminal_hold_confirmed is True


def test_mock_real_replay_rejects_failed_sim_or_wrong_authority() -> None:
    trajectory = _trajectory()
    failed_simulation = _simulation_receipt(trajectory, completed=False)
    ports = _ReplayPorts(trajectory)
    failed_sim = _execute(ports, trajectory, failed_simulation)
    assert failed_sim.status is RegisteredTrajectoryRealReplayStatus.PREFLIGHT_REJECTED
    assert ports.dispatched == []

    simulation = _simulation_receipt(trajectory)
    wrong_authority = _authority(
        trajectory,
        simulation,
        trajectory_hash="different-trajectory:test",
    )
    wrong = _execute(ports, trajectory, simulation, authority=wrong_authority)
    assert wrong.status is RegisteredTrajectoryRealReplayStatus.AUTHORITY_REJECTED
    assert ports.dispatched == []
    assert ports.authority_consumptions == 0


def test_mock_real_replay_dispatch_failure_and_tracking_timeout_both_hold() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    failed_ports = _ReplayPorts(trajectory, dispatch_failure_index=1)
    failed = _execute(failed_ports, trajectory, simulation)
    assert failed.status is RegisteredTrajectoryRealReplayStatus.DISPATCH_FAILED
    assert failed.executed_point_count == 1
    assert failed.terminal_hold_confirmed is True

    frozen_ports = _ReplayPorts(trajectory, update_qpos=False)
    timed_out = _execute(frozen_ports, trajectory, simulation)
    assert timed_out.status is RegisteredTrajectoryRealReplayStatus.FINAL_TRACKING_TIMEOUT
    assert timed_out.executed_point_count == len(trajectory.points)
    assert timed_out.terminal_hold_confirmed is True


def test_mock_real_replay_gripper_drift_and_hold_failure_are_terminal() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    drift_ports = _ReplayPorts(trajectory, gripper_drift_index=0)
    drift = _execute(drift_ports, trajectory, simulation)
    assert drift.status is RegisteredTrajectoryRealReplayStatus.GRIPPER_HOLD_DRIFT
    assert drift.executed_point_count == 1
    assert drift.terminal_hold_confirmed is True

    hold_failure_ports = _ReplayPorts(trajectory, hold_succeeds=False)
    hold_failure = _execute(hold_failure_ports, trajectory, simulation)
    assert (
        hold_failure.status
        is RegisteredTrajectoryRealReplayStatus.TERMINAL_HOLD_FAILED
    )
    assert hold_failure.executed_point_count == len(trajectory.points)
    assert hold_failure.terminal_hold_confirmed is False


def test_mock_real_replay_lower_dispatch_or_feedback_error_always_holds() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)

    dispatch_error_ports = _ReplayPorts(trajectory, dispatch_raise_index=1)
    dispatch_error = _execute(dispatch_error_ports, trajectory, simulation)
    assert dispatch_error.status is RegisteredTrajectoryRealReplayStatus.LOWER_PORT_ERROR
    assert dispatch_error.executed_point_count == 1
    assert dispatch_error.real_motion_executed is True
    assert dispatch_error.terminal_hold_confirmed is True
    assert dispatch_error_ports.hold_reasons == [
        "registered shadow replay lower port error"
    ]

    feedback_error_ports = _ReplayPorts(trajectory, snapshot_raise_call=2)
    feedback_error = _execute(feedback_error_ports, trajectory, simulation)
    assert feedback_error.status is RegisteredTrajectoryRealReplayStatus.LOWER_PORT_ERROR
    assert feedback_error.executed_point_count == 1
    assert feedback_error.real_motion_executed is True
    assert feedback_error.terminal_hold_confirmed is True


class _StartedRM65Ports:
    def __init__(self, trajectory, *, clock, admitted=True, hold_succeeds=True):
        self.trajectory = trajectory
        self.clock = clock
        self.admitted = admitted
        self.hold_succeeds = hold_succeeds
        self.qpos = {
            side: value.copy()
            for side, value in trajectory.initial_snapshot.qpos_by_side.items()
        }
        self.admitted_commands = []
        self.dispatched_commands = []
        self.hold_reasons = []

    def observe(self):
        now = self.clock()
        return RealSkillMotionObservationCut(
            observation=TeleopObservation(timestamp=now),
            qpos_by_side=self.qpos,
            measured_at=now - 0.01,
            available_at=now,
            evidence_refs=(f"rm65-cut:{now}",),
        )

    def admit_command(self, command, observation, now):
        assert observation.timestamp == pytest.approx(now)
        self.admitted_commands.append(command)
        return self.admitted

    def dispatch(self, command):
        self.dispatched_commands.append(command)
        self.qpos = {
            side: np.asarray(command.commands[f"{side}_arm"].value).copy()
            for side in ("left", "right")
        }
        return RealSkillMotionDispatchReceipt(
            succeeded=True,
            evidence_refs=("rm65-dispatch:test",),
        )

    def hold(self, reason):
        self.hold_reasons.append(reason)
        return RealSkillMotionHoldReceipt(
            succeeded=self.hold_succeeds,
            evidence_refs=("rm65-hold:test",),
        )


def _dmtac_replay_ports(*, admitted=True, right_close=0.3, identity=None):
    trajectory = _trajectory()
    now = 10.0
    rm65 = _StartedRM65Ports(
        trajectory,
        clock=lambda: now,
        admitted=admitted,
    )
    dmtac = {
        "left": _backend("left", 0.2, now - 0.009),
        "right": _backend("right", right_close, now - 0.008),
    }
    ports = DMTacRM65RegisteredTrajectoryPorts(
        trajectory=trajectory,
        identity=identity or trajectory.initial_snapshot.identity,
        rm65_ports=rm65,
        dmtac_backends_by_side=dmtac,
        profile=DMTacShadowReplayPortProfile(
            maximum_inter_gripper_skew_sec=0.05,
            maximum_arm_gripper_skew_sec=0.05,
            maximum_gripper_hold_drift=0.02,
        ),
        clock=lambda: now,
    )
    return trajectory, ports, rm65, dmtac


def test_dmtac_rm65_ports_join_measured_state_and_dispatch_only_arm_qpos() -> None:
    trajectory, ports, rm65, dmtac = _dmtac_replay_ports()

    snapshot = ports.current_snapshot()
    receipt = ports.dispatch_point(
        trajectory.points[-1],
        trajectory.semantic_content_hash,
    )

    assert snapshot.gripper_close_by_side == {"left": 0.2, "right": 0.3}
    assert receipt.succeeded is True
    assert len(rm65.admitted_commands) == 1
    assert len(rm65.dispatched_commands) == 1
    assert set(rm65.dispatched_commands[0].commands) == {
        "left_arm",
        "right_arm",
    }
    assert all(backend.read_count == 1 for backend in dmtac.values())


def test_dmtac_rm65_ports_never_dispatch_when_lower_admission_rejects() -> None:
    trajectory, ports, rm65, _ = _dmtac_replay_ports(admitted=False)
    ports.current_snapshot()

    receipt = ports.dispatch_point(
        trajectory.points[0],
        trajectory.semantic_content_hash,
    )

    assert receipt.succeeded is False
    assert len(rm65.admitted_commands) == 1
    assert rm65.dispatched_commands == []


def test_dmtac_rm65_current_identity_is_independent_of_registered_trajectory() -> None:
    trajectory, ports, rm65, _ = _dmtac_replay_ports(
        identity=_identity(gripper_model_hash="dmtac-model:changed"),
    )

    snapshot = ports.current_snapshot()
    dispatch = ports.dispatch_point(
        trajectory.points[0],
        trajectory.semantic_content_hash,
    )

    assert (
        snapshot.identity.semantic_content_hash
        != trajectory.initial_snapshot.identity.semantic_content_hash
    )
    assert dispatch.succeeded is False
    assert rm65.admitted_commands == []


def test_dmtac_rm65_hold_is_arm_hold_plus_passive_measured_opening_fence() -> None:
    _, ports, rm65, dmtac = _dmtac_replay_ports()

    receipt = ports.hold("test terminal hold")

    assert receipt.succeeded is True
    assert rm65.hold_reasons == ["test terminal hold"]
    assert all(backend.read_count == 1 for backend in dmtac.values())
    assert any(ref.startswith("dmtac-passive-fence-hold:") for ref in receipt.evidence_refs)


def test_dmtac_rm65_hold_reports_measured_gripper_drift_without_commanding_it() -> None:
    _, ports, rm65, dmtac = _dmtac_replay_ports(right_close=0.36)

    receipt = ports.hold("drift hold")

    assert receipt.succeeded is False
    assert rm65.hold_reasons == ["drift hold"]
    assert all(backend.read_count == 1 for backend in dmtac.values())


def test_dmtac_rm65_adapter_completes_injected_exact_replay_without_gripper_command() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    clock = {"wall": 10.0, "monotonic": 0.0}

    def sleep(duration):
        clock["wall"] += float(duration)
        clock["monotonic"] += float(duration)

    rm65 = _StartedRM65Ports(
        trajectory,
        clock=lambda: clock["wall"],
    )
    dmtac = {
        "left": _backend("left", 0.2, 9.995),
        "right": _backend("right", 0.3, 9.996),
    }
    ports = DMTacRM65RegisteredTrajectoryPorts(
        trajectory=trajectory,
        identity=trajectory.initial_snapshot.identity,
        rm65_ports=rm65,
        dmtac_backends_by_side=dmtac,
        profile=DMTacShadowReplayPortProfile(
            maximum_inter_gripper_skew_sec=0.05,
            maximum_arm_gripper_skew_sec=0.1,
            maximum_gripper_hold_drift=0.02,
        ),
        clock=lambda: clock["wall"],
    )

    receipt = execute_registered_shadow_trajectory_with_ports(
        trajectory=trajectory,
        simulation_receipt=simulation,
        current_scene_hash=trajectory.scene_hash,
        current_snapshot_provider=ports.current_snapshot,
        point_dispatcher=ports.dispatch_point,
        hold_controller=ports.hold,
        authority=_authority(trajectory, simulation),
        consume_execution_authority=lambda authority: (
            f"authority-consumed:{authority.authority_id}",
        ),
        execution_enabled=True,
        wall_clock=lambda: clock["wall"],
        monotonic_clock=lambda: clock["monotonic"],
        sleep=sleep,
        final_tracking_timeout_sec=0.06,
        feedback_poll_sec=0.01,
    )

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.COMPLETED
    assert receipt.executed_point_count == len(trajectory.points)
    assert all(
        set(command.commands) == {"left_arm", "right_arm"}
        for command in rm65.dispatched_commands
    )
    assert all(backend.read_count >= 4 for backend in dmtac.values())


def test_mock_real_replay_initial_read_and_hold_transport_errors_fail_closed() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)

    initial_read_error_ports = _ReplayPorts(trajectory, snapshot_raise_call=0)
    with pytest.raises(RuntimeError, match="snapshot failure at call 0"):
        _execute(initial_read_error_ports, trajectory, simulation)
    assert initial_read_error_ports.dispatched == []
    assert initial_read_error_ports.hold_reasons == [
        "registered shadow replay initial state read failed"
    ]

    hold_error_ports = _ReplayPorts(trajectory, hold_raises=True)
    hold_error = _execute(hold_error_ports, trajectory, simulation)
    assert (
        hold_error.status
        is RegisteredTrajectoryRealReplayStatus.TERMINAL_HOLD_FAILED
    )
    assert hold_error.terminal_hold_confirmed is False
    assert any(ref.startswith("hold-port-error:") for ref in hold_error.evidence_refs)

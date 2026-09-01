from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

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
    RegisteredTrajectoryAuthorityConsumptionReceipt,
    RegisteredTrajectoryAuthorityConsumptionStatus,
    RegisteredTrajectoryExecutionAuthority,
    RegisteredTrajectoryPointDispatchReceipt,
    RegisteredTrajectoryRealReplayStatus,
    SingleUseRegisteredTrajectoryAuthorityConsumer,
    TerminalToolTcpVerificationReceipt,
    TerminalToolTcpVerificationStatus,
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


def _snapshot(*, qpos_by_side=None, gripper_by_side=None, identity=None, now=9.95):
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
        identity=identity or _identity(),
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


class _ToolTcpWorldFk:
    def __init__(self):
        self.calls: list[dict[str, np.ndarray]] = []

    def tool_tcp_world_targets_for_qpos_pair(self, *, left_qpos, right_qpos):
        qpos_by_side = {
            "left": np.asarray(left_qpos, dtype=float).copy(),
            "right": np.asarray(right_qpos, dtype=float).copy(),
        }
        self.calls.append(qpos_by_side)
        result = {}
        for side, qpos in qpos_by_side.items():
            pose = np.eye(4)
            pose[0, 3] = qpos[1]
            result[side] = pose
        return result


def _terminal_target(trajectory):
    fk = _ToolTcpWorldFk()
    observed = fk.tool_tcp_world_targets_for_qpos_pair(
        left_qpos=trajectory.points[-1].qpos_by_side["left"],
        right_qpos=trajectory.points[-1].qpos_by_side["right"],
    )
    return fk, {
        side: observed[side] for side in trajectory.moving_sides
    }


def _tcp_receipt(snapshot, *, verified=True):
    return TerminalToolTcpVerificationReceipt(
        status=(
            TerminalToolTcpVerificationStatus.VERIFIED
            if verified
            else TerminalToolTcpVerificationStatus.TARGET_MISMATCH
        ),
        frame_id="dual_arm_solver_world",
        observed_snapshot_hash=snapshot.semantic_content_hash,
        position_error_m_by_side={"right": 0.0 if verified else 0.01},
        orientation_error_deg_by_side={"right": 0.0},
        position_tolerance_m=0.002,
        orientation_tolerance_deg=0.5,
        evidence_refs=snapshot.arm_evidence_refs,
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
        identity_change_call=None,
        arm_drift_call=None,
        stale_snapshot_call=None,
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
        self.identity_change_call = identity_change_call
        self.arm_drift_call = arm_drift_call
        self.stale_snapshot_call = stale_snapshot_call
        self.snapshot_count = 0
        self.dispatched: list[str] = []
        self.hold_reasons: list[str] = []
        self.authority_consumptions = 0
        self.session_active = False
        self.session_end_reasons: list[str] = []

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
        observed_qpos = {
            side: value.copy() for side, value in self.qpos.items()
        }
        if call_index == self.arm_drift_call:
            observed_qpos["right"][0] += 0.2
        return _snapshot(
            qpos_by_side=observed_qpos,
            gripper_by_side=self.grippers,
            identity=(
                _identity(gripper_model_hash="gripper:changed")
                if call_index == self.identity_change_call
                else None
            ),
            now=(
                self.wall - 1.0
                if call_index == self.stale_snapshot_call
                else self.wall
            ),
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
        return RegisteredTrajectoryAuthorityConsumptionReceipt(
            status=RegisteredTrajectoryAuthorityConsumptionStatus.CONSUMED,
            authority_hash=authority.semantic_content_hash,
            authority_id=authority.authority_id,
            trajectory_hash=authority.trajectory_hash,
            simulation_receipt_hash=authority.simulation_receipt_hash,
            run_id=authority.run_id,
            owner_epoch=authority.owner_epoch,
            consumed_at=self.wall,
            evidence_refs=(f"authority-consumed:{authority.authority_id}",),
        )

    def begin_replay_session(self, consumption):
        assert consumption.consumed
        assert self.session_active is False
        self.session_active = True
        return (f"session-begin:{consumption.semantic_content_hash}",)

    def end_replay_session(self, reason):
        assert self.session_active is True
        self.session_active = False
        self.session_end_reasons.append(reason)
        return (f"session-end:{reason}",)


def _authority(trajectory, simulation_receipt, *, trajectory_hash=None):
    return RegisteredTrajectoryExecutionAuthority(
        authority_id="authority:test",
        operator_confirmation_ref="operator-confirmation:test",
        trajectory_hash=trajectory_hash or trajectory.semantic_content_hash,
        simulation_receipt_hash=simulation_receipt.semantic_content_hash,
        run_id="run:test",
        owner_epoch=1,
        issued_at=9.99,
        expires_at=10.5,
    )


def _execute(ports, trajectory, simulation_receipt, **overrides):
    values = {
        "trajectory": trajectory,
        "simulation_receipt": simulation_receipt,
        "current_scene_hash": "scene:test",
        "current_scene_hash_provider": lambda: "scene:test",
        "current_snapshot_provider": ports.snapshot,
        "point_dispatcher": ports.dispatch,
        "hold_controller": ports.hold,
        "authority": _authority(trajectory, simulation_receipt),
        "consume_execution_authority": ports.consume,
        "begin_replay_session": ports.begin_replay_session,
        "end_replay_session": ports.end_replay_session,
        "execution_enabled": True,
        "wall_clock": ports.wall_clock,
        "monotonic_clock": ports.monotonic_clock,
        "sleep": ports.sleep,
        "final_tracking_timeout_sec": 0.2,
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
    assert ports.monotonic >= 0.12


def test_required_terminal_tcp_verifier_is_checked_before_authority_or_dispatch():
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory)

    receipt = _execute(
        ports,
        trajectory,
        simulation,
        terminal_tool_tcp_verification_required=True,
    )

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED
    assert ports.authority_consumptions == 0
    assert ports.dispatched == []
    assert ports.hold_reasons == [
        "registered shadow replay terminal TCP verifier missing"
    ]


def test_terminal_tcp_must_share_the_tracking_dwell_and_failure_holds():
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory)

    receipt = _execute(
        ports,
        trajectory,
        simulation,
        terminal_tool_tcp_verifier=lambda snapshot: _tcp_receipt(
            snapshot,
            verified=False,
        ),
        terminal_tool_tcp_verification_required=True,
    )

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.FINAL_TCP_TARGET_MISMATCH
    assert receipt.terminal_tool_tcp_verification is not None
    assert receipt.terminal_tool_tcp_verification.verified is False
    assert receipt.executed_point_count == len(trajectory.points)
    assert receipt.terminal_hold_confirmed is True


def test_terminal_tcp_verifier_exception_is_lower_port_error_and_holds():
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory)

    receipt = _execute(
        ports,
        trajectory,
        simulation,
        terminal_tool_tcp_verifier=lambda _snapshot: (_ for _ in ()).throw(
            RuntimeError("FK unavailable")
        ),
        terminal_tool_tcp_verification_required=True,
    )

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.LOWER_PORT_ERROR
    assert receipt.terminal_hold_confirmed is True


def test_authority_consumer_is_atomic_single_use_for_sequential_and_concurrent_calls():
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    authority = _authority(trajectory, simulation)

    sequential = SingleUseRegisteredTrajectoryAuthorityConsumer(
        authority=authority,
        clock=lambda: 10.0,
    )
    first = sequential.consume(authority)
    second = sequential.consume(authority)
    assert first.status is RegisteredTrajectoryAuthorityConsumptionStatus.CONSUMED
    assert (
        second.status
        is RegisteredTrajectoryAuthorityConsumptionStatus.ALREADY_CONSUMED
    )

    concurrent = SingleUseRegisteredTrajectoryAuthorityConsumer(
        authority=authority,
        clock=lambda: 10.0,
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = tuple(pool.map(concurrent.consume, (authority, authority)))
    assert sum(receipt.consumed for receipt in receipts) == 1
    assert {
        receipt.status for receipt in receipts
    } == {
        RegisteredTrajectoryAuthorityConsumptionStatus.CONSUMED,
        RegisteredTrajectoryAuthorityConsumptionStatus.ALREADY_CONSUMED,
    }


def test_reusing_consumed_authority_never_dispatches_second_replay() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    authority = _authority(trajectory, simulation)
    consumer = SingleUseRegisteredTrajectoryAuthorityConsumer(
        authority=authority,
        clock=lambda: 10.0,
    )

    first_ports = _ReplayPorts(trajectory)
    first = _execute(
        first_ports,
        trajectory,
        simulation,
        authority=authority,
        consume_execution_authority=consumer.consume,
    )
    second_ports = _ReplayPorts(trajectory)
    second = _execute(
        second_ports,
        trajectory,
        simulation,
        authority=authority,
        consume_execution_authority=consumer.consume,
    )

    assert first.status is RegisteredTrajectoryRealReplayStatus.COMPLETED
    assert second.status is RegisteredTrajectoryRealReplayStatus.AUTHORITY_REJECTED
    assert second_ports.dispatched == []
    assert second_ports.hold_reasons == [
        "registered shadow replay authority consumption failed"
    ]


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
    assert failed.real_motion_executed is True
    assert failed.real_motion_may_have_executed is True
    assert failed.terminal_hold_confirmed is True

    frozen_ports = _ReplayPorts(trajectory, update_qpos=False)
    timed_out = _execute(frozen_ports, trajectory, simulation)
    assert timed_out.status is RegisteredTrajectoryRealReplayStatus.FINAL_TRACKING_TIMEOUT
    assert timed_out.executed_point_count == len(trajectory.points)
    assert timed_out.terminal_hold_confirmed is True


def test_first_dual_arm_dispatch_failure_is_reported_as_possible_motion() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory, dispatch_failure_index=0)

    receipt = _execute(ports, trajectory, simulation)

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.DISPATCH_FAILED
    assert receipt.executed_point_count == 0
    assert receipt.real_motion_executed is False
    assert receipt.real_motion_may_have_executed is True
    assert receipt.terminal_hold_confirmed is True
    assert ports.hold_reasons == ["registered shadow replay point dispatch failed"]


def test_identity_or_scene_change_before_first_point_causes_zero_dispatch() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)

    identity_ports = _ReplayPorts(trajectory, identity_change_call=1)
    identity_changed = _execute(identity_ports, trajectory, simulation)
    assert (
        identity_changed.status
        is RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED
    )
    assert identity_ports.dispatched == []
    assert identity_changed.real_motion_executed is False
    assert identity_changed.real_motion_may_have_executed is False

    scene_ports = _ReplayPorts(trajectory)
    scene_changed = _execute(
        scene_ports,
        trajectory,
        simulation,
        current_scene_hash_provider=lambda: "scene:changed",
    )
    assert (
        scene_changed.status
        is RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED
    )
    assert scene_ports.dispatched == []


@pytest.mark.parametrize(
    "ports",
    (
        pytest.param(
            lambda trajectory: _ReplayPorts(trajectory, arm_drift_call=1),
            id="arm_tracking_drift",
        ),
        pytest.param(
            lambda trajectory: _ReplayPorts(trajectory, stale_snapshot_call=1),
            id="snapshot_stale",
        ),
    ),
)
def test_stale_or_drifted_pre_dispatch_state_causes_zero_dispatch(ports) -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    replay_ports = ports(trajectory)

    receipt = _execute(replay_ports, trajectory, simulation)

    assert (
        receipt.status
        is RegisteredTrajectoryRealReplayStatus.PRE_DISPATCH_STATE_REJECTED
    )
    assert receipt.executed_point_count == 0
    assert replay_ports.dispatched == []


def test_timeline_lateness_does_not_burst_dispatch_overdue_points() -> None:
    trajectory = _trajectory()
    simulation = _simulation_receipt(trajectory)
    ports = _ReplayPorts(trajectory)
    monotonic_reads = iter((0.0, 0.2))

    receipt = _execute(
        ports,
        trajectory,
        simulation,
        monotonic_clock=lambda: next(monotonic_reads),
        maximum_point_lateness_sec=0.01,
    )

    assert (
        receipt.status
        is RegisteredTrajectoryRealReplayStatus.TIMELINE_DEADLINE_MISSED
    )
    assert receipt.executed_point_count == 0
    assert receipt.real_motion_executed is False
    assert receipt.real_motion_may_have_executed is False
    assert ports.dispatched == []
    assert ports.hold_reasons == [
        "registered shadow replay timeline deadline missed"
    ]


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

    dispatch_and_hold_failure_ports = _ReplayPorts(
        trajectory,
        dispatch_failure_index=0,
        hold_succeeds=False,
    )
    dispatch_and_hold_failure = _execute(
        dispatch_and_hold_failure_ports,
        trajectory,
        simulation,
    )
    assert (
        dispatch_and_hold_failure.status
        is RegisteredTrajectoryRealReplayStatus.DISPATCH_FAILED
    )
    assert dispatch_and_hold_failure.terminal_hold_confirmed is False
    assert dispatch_and_hold_failure.real_motion_may_have_executed is True
    assert any(
        ref.startswith("terminal-hold-unconfirmed:")
        for ref in dispatch_and_hold_failure.evidence_refs
    )


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
    tool_tcp_fk, targets = _terminal_target(trajectory)
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
        tool_tcp_world_fk=tool_tcp_fk,
        target_tool_tcp_world_by_side=targets,
        target_reference_frame_id="dual_arm_solver_world",
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
    simulation = _simulation_receipt(trajectory)
    authority = _authority(trajectory, simulation)
    ports.begin_replay_session(_ReplayPorts(trajectory).consume(authority))

    snapshot = ports.current_snapshot()
    receipt = ports.dispatch_point(
        trajectory.points[0],
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
    simulation = _simulation_receipt(trajectory)
    authority = _authority(trajectory, simulation)
    ports.begin_replay_session(_ReplayPorts(trajectory).consume(authority))
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
    simulation = _simulation_receipt(trajectory)
    authority = _authority(trajectory, simulation)
    ports.begin_replay_session(_ReplayPorts(trajectory).consume(authority))

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


def test_dmtac_terminal_tcp_uses_measured_qpos_and_only_the_moving_side() -> None:
    trajectory, ports, rm65, _ = _dmtac_replay_ports()

    before = ports.verify_terminal_tool_tcp(ports.current_snapshot())
    assert before.status is TerminalToolTcpVerificationStatus.TARGET_MISMATCH
    assert set(before.position_error_m_by_side) == {"right"}

    rm65.qpos = {
        "left": np.ones(6) * 0.4,
        "right": trajectory.points[-1].qpos_by_side["right"].copy(),
    }
    terminal_snapshot = ports.current_snapshot()
    verified = ports.verify_terminal_tool_tcp(terminal_snapshot)

    assert verified.status is TerminalToolTcpVerificationStatus.VERIFIED
    assert verified.observed_snapshot_hash == terminal_snapshot.semantic_content_hash
    np.testing.assert_allclose(
        ports.tool_tcp_world_fk.calls[-1]["left"],
        rm65.qpos["left"],
    )
    np.testing.assert_allclose(
        ports.tool_tcp_world_fk.calls[-1]["right"],
        rm65.qpos["right"],
    )


def test_dmtac_rm65_adapter_completes_injected_exact_replay_without_gripper_command() -> None:
    trajectory = _trajectory()
    tool_tcp_fk, targets = _terminal_target(trajectory)
    simulation = _simulation_receipt(trajectory)
    clock = {"wall": 10.0, "monotonic": 0.0}

    def sleep(duration):
        clock["wall"] += float(duration)
        clock["monotonic"] += float(duration)
        dmtac["left"].payload["timestamp"] = clock["wall"] - 0.009
        dmtac["right"].payload["timestamp"] = clock["wall"] - 0.008

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
        tool_tcp_world_fk=tool_tcp_fk,
        target_tool_tcp_world_by_side=targets,
        target_reference_frame_id="dual_arm_solver_world",
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
        current_scene_hash_provider=lambda: trajectory.scene_hash,
        current_snapshot_provider=ports.current_snapshot,
        point_dispatcher=ports.dispatch_point,
        hold_controller=ports.hold,
        terminal_tool_tcp_verifier=ports.verify_terminal_tool_tcp,
        terminal_tool_tcp_verification_required=True,
        authority=_authority(trajectory, simulation),
        consume_execution_authority=_ReplayPorts(
            trajectory
        ).consume,
        begin_replay_session=ports.begin_replay_session,
        end_replay_session=ports.end_replay_session,
        execution_enabled=True,
        wall_clock=lambda: clock["wall"],
        monotonic_clock=lambda: clock["monotonic"],
        sleep=sleep,
        final_tracking_timeout_sec=0.2,
        feedback_poll_sec=0.01,
    )

    assert receipt.status is RegisteredTrajectoryRealReplayStatus.COMPLETED
    assert receipt.terminal_tool_tcp_verification is not None
    assert receipt.terminal_tool_tcp_verification.verified is True
    assert receipt.executed_point_count == len(trajectory.points)
    assert all(
        set(command.commands) == {"left_arm", "right_arm"}
        for command in rm65.dispatched_commands
    )
    assert all(backend.read_count >= 4 for backend in dmtac.values())


def test_dmtac_rm65_direct_or_out_of_order_dispatch_is_rejected_without_send() -> None:
    trajectory, ports, rm65, _ = _dmtac_replay_ports()

    without_session = ports.dispatch_point(
        trajectory.points[0],
        trajectory.semantic_content_hash,
    )
    assert without_session.succeeded is False
    assert rm65.dispatched_commands == []

    simulation = _simulation_receipt(trajectory)
    authority = _authority(trajectory, simulation)
    ports.begin_replay_session(_ReplayPorts(trajectory).consume(authority))
    out_of_order = ports.dispatch_point(
        trajectory.points[1],
        trajectory.semantic_content_hash,
    )
    assert out_of_order.succeeded is False
    assert rm65.dispatched_commands == []

    first = ports.dispatch_point(
        trajectory.points[0],
        trajectory.semantic_content_hash,
    )
    repeated = ports.dispatch_point(
        trajectory.points[0],
        trajectory.semantic_content_hash,
    )
    assert first.succeeded is True
    assert repeated.succeeded is False
    assert len(rm65.dispatched_commands) == 1


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

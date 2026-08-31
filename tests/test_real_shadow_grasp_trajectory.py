from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.runtime import TeleopObservation
from robot_motion.shadow_execution import (
    AuthoritativeRobotStateSnapshot,
    RegisteredJointTrajectoryPoint,
    RegisteredShadowJointTrajectory,
    ShadowExecutionIdentity,
    ShadowReplayAdmissionStatus,
    ShadowSimulationReceipt,
    ShadowSimulationStatus,
    evaluate_shadow_replay_admission,
)
from teleop_runtime.skill_deploy.control_runtime import (
    RealSkillMotionObservationCut,
)
from teleop_runtime.skill_deploy.shadow_trajectory import (
    capture_authoritative_shadow_state,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _identity(*, collision_model_hash: str = "collision:test") -> ShadowExecutionIdentity:
    return ShadowExecutionIdentity(
        station_revision="canonical-station:test",
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame-spec:test",
        frame_graph_hash="frame-graph:test",
        tool_profile_hash_by_side={
            "left": "tool:left:test",
            "right": "tool:right:test",
        },
        robot_model_hash="robot-model:test",
        collision_model_hash=collision_model_hash,
        joint_layout_hash="joint-layout:test",
        gripper_model_hash="gripper-model:test",
    )


def _snapshot(
    *,
    qpos_offset: float = 0.0,
    gripper_offset: float = 0.0,
    arm_measured_at: float = 9.9,
    identity: ShadowExecutionIdentity | None = None,
) -> AuthoritativeRobotStateSnapshot:
    return AuthoritativeRobotStateSnapshot(
        qpos_by_side={
            "left": np.arange(6, dtype=float) * 0.01 + qpos_offset,
            "right": -np.arange(6, dtype=float) * 0.01 + qpos_offset,
        },
        gripper_close_by_side={
            "left": 0.2 + gripper_offset,
            "right": 0.3 + gripper_offset,
        },
        arm_measured_at=arm_measured_at,
        gripper_measured_at_by_side={"left": 9.91, "right": 9.92},
        available_at=9.95,
        arm_evidence_refs=("arm-cut:test",),
        gripper_evidence_refs=("gripper-cut:left", "gripper-cut:right"),
        identity=identity or _identity(),
    )


def _trajectory(
    snapshot: AuthoritativeRobotStateSnapshot | None = None,
) -> RegisteredShadowJointTrajectory:
    initial = snapshot or _snapshot()
    final = {
        side: initial.qpos_by_side[side].copy() for side in ("left", "right")
    }
    final["right"][1] += 0.05
    return RegisteredShadowJointTrajectory(
        trajectory_id="shadow-trajectory:test",
        created_at=10.0,
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
                time_from_start_sec=0.5,
                qpos_by_side=final,
            ),
        ),
    )


def _completed_receipt(
    trajectory: RegisteredShadowJointTrajectory,
) -> ShadowSimulationReceipt:
    return ShadowSimulationReceipt(
        trajectory_hash=trajectory.semantic_content_hash,
        initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
        identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=ShadowSimulationStatus.COMPLETED,
        executed_point_count=len(trajectory.points),
        terminal_hold_confirmed=True,
        exact_planned_qpos_execution_confirmed=True,
        model_identity_confirmed=True,
        minimum_clearance_m=0.02,
        evidence_refs=("mujoco-run:test", "mujoco-terminal-hold:test"),
    )


def _admit(
    current: AuthoritativeRobotStateSnapshot,
    trajectory: RegisteredShadowJointTrajectory,
    receipt: ShadowSimulationReceipt,
):
    return evaluate_shadow_replay_admission(
        current_snapshot=current,
        trajectory=trajectory,
        simulation_receipt=receipt,
        current_scene_hash="scene:test",
        evaluated_at=10.0,
        maximum_snapshot_age_sec=0.2,
        maximum_arm_start_drift_rad=0.01,
        maximum_gripper_start_drift=0.02,
    )


def test_existing_lower_arm_cut_and_measured_grippers_form_complete_snapshot() -> None:
    arm_cut = RealSkillMotionObservationCut(
        observation=TeleopObservation(timestamp=9.95),
        qpos_by_side={"left": np.zeros(6), "right": np.ones(6) * 0.1},
        measured_at=9.9,
        available_at=9.94,
        evidence_refs=("rm65-hardware-qpos-cut:test",),
    )

    snapshot = capture_authoritative_shadow_state(
        arm_observation_cut=arm_cut,
        gripper_close_by_side={"left": 0.25, "right": 0.75},
        gripper_measured_at_by_side={"left": 9.91, "right": 9.92},
        gripper_available_at=9.96,
        gripper_evidence_refs=("dmtac:left:42", "dmtac:right:51"),
        identity=_identity(),
    )

    assert snapshot.available_at == 9.96
    assert snapshot.oldest_measurement_at == 9.9
    assert snapshot.gripper_close_by_side == {"left": 0.25, "right": 0.75}
    assert snapshot.arm_evidence_refs == ("rm65-hardware-qpos-cut:test",)
    assert snapshot.semantic_content_hash
    with pytest.raises(ValueError, match="assumed gripper state"):
        capture_authoritative_shadow_state(
            arm_observation_cut=arm_cut,
            gripper_close_by_side={"left": 0.0, "right": 0.0},
            gripper_measured_at_by_side={"left": 9.91, "right": 9.92},
            gripper_available_at=9.96,
            gripper_evidence_refs=("fallback:left", "fallback:right"),
            identity=_identity(),
            gripper_state_assumed=True,
        )
    with pytest.raises(ValueError, match="cannot precede a gripper measurement"):
        capture_authoritative_shadow_state(
            arm_observation_cut=arm_cut,
            gripper_close_by_side={"left": 0.2, "right": 0.3},
            gripper_measured_at_by_side={"left": 9.97, "right": 9.92},
            gripper_available_at=9.96,
            gripper_evidence_refs=("dmtac:left:42", "dmtac:right:51"),
            identity=_identity(),
        )


def test_registered_trajectory_binds_exact_start_and_holds_inactive_arm() -> None:
    trajectory = _trajectory()

    assert trajectory.non_authoritative is True
    assert trajectory.gripper_policy == "hold_measured"
    assert trajectory.points[0].time_from_start_sec == 0.0
    with pytest.raises(ValueError, match="inactive left arm"):
        moved_left = {
            side: trajectory.initial_snapshot.qpos_by_side[side].copy()
            for side in ("left", "right")
        }
        moved_left["left"][0] += 0.01
        replace(
            trajectory,
            points=(
                trajectory.points[0],
                RegisteredJointTrajectoryPoint(
                    time_from_start_sec=0.5,
                    qpos_by_side=moved_left,
                ),
            ),
        )


def test_successful_replay_admission_remains_preflight_only() -> None:
    initial = _snapshot()
    trajectory = _trajectory(initial)
    receipt = _completed_receipt(trajectory)

    admission = _admit(initial, trajectory, receipt)

    assert admission.status is ShadowReplayAdmissionStatus.ADMISSIBLE_PREFLIGHT_ONLY
    assert admission.admissible is True
    assert admission.preflight_only is True
    assert admission.real_motion_authorized is False
    assert admission.reason_codes == ()


@pytest.mark.parametrize(
    ("current", "expected"),
    (
        (_snapshot(arm_measured_at=9.0), ShadowReplayAdmissionStatus.SNAPSHOT_STALE),
        (_snapshot(qpos_offset=0.02), ShadowReplayAdmissionStatus.ARM_START_STATE_DRIFT),
        (
            _snapshot(gripper_offset=0.03),
            ShadowReplayAdmissionStatus.GRIPPER_START_STATE_DRIFT,
        ),
        (
            _snapshot(identity=_identity(collision_model_hash="collision:changed")),
            ShadowReplayAdmissionStatus.IDENTITY_MISMATCH,
        ),
    ),
)
def test_replay_admission_rejects_stale_drifted_or_identity_changed_state(
    current: AuthoritativeRobotStateSnapshot,
    expected: ShadowReplayAdmissionStatus,
) -> None:
    trajectory = _trajectory()
    admission = _admit(current, trajectory, _completed_receipt(trajectory))

    assert admission.admissible is False
    assert admission.status is expected
    assert expected.value in admission.reason_codes
    assert admission.real_motion_authorized is False


def test_failed_or_mismatched_simulation_cannot_be_replayed() -> None:
    initial = _snapshot()
    trajectory = _trajectory(initial)
    failed = ShadowSimulationReceipt(
        trajectory_hash="another-trajectory:test",
        initial_snapshot_hash=initial.semantic_content_hash,
        identity_hash=initial.identity.semantic_content_hash,
        scene_hash=trajectory.scene_hash,
        status=ShadowSimulationStatus.PATH_COLLISION_BLOCKED,
        executed_point_count=0,
        terminal_hold_confirmed=False,
        exact_planned_qpos_execution_confirmed=False,
        model_identity_confirmed=False,
        evidence_refs=("mujoco-path-collision:test",),
    )

    admission = _admit(initial, trajectory, failed)

    assert admission.status is ShadowReplayAdmissionStatus.TRAJECTORY_BINDING_MISMATCH
    assert admission.reason_codes == (
        "trajectory_binding_mismatch",
        "sim_execution_unconfirmed",
        "sim_terminal_hold_unconfirmed",
    )
    assert admission.real_motion_authorized is False


def test_completed_simulation_requires_exact_qpos_identity_and_terminal_hold() -> None:
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="exact qpos execution, verified model identity"):
        ShadowSimulationReceipt(
            trajectory_hash=trajectory.semantic_content_hash,
            initial_snapshot_hash=trajectory.initial_snapshot.semantic_content_hash,
            identity_hash=trajectory.initial_snapshot.identity.semantic_content_hash,
            scene_hash=trajectory.scene_hash,
            status=ShadowSimulationStatus.COMPLETED,
            executed_point_count=len(trajectory.points),
            terminal_hold_confirmed=True,
            exact_planned_qpos_execution_confirmed=False,
            model_identity_confirmed=True,
            evidence_refs=("mujoco-run:test",),
        )

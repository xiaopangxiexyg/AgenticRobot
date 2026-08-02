from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control import (
    JointHorizonLimits,
    SampleAlignedJerkLimitedJointTrajectoryPlanner,
)


def _planner(**overrides) -> SampleAlignedJerkLimitedJointTrajectoryPlanner:
    planner = SampleAlignedJerkLimitedJointTrajectoryPlanner(
        JointHorizonLimits(
            min_position_rad=np.full(6, -2.0),
            max_position_rad=np.full(6, 2.0),
            max_velocity_rad_s=np.full(6, 1.4),
            max_acceleration_rad_s2=np.full(6, 7.0),
            max_jerk_rad_s3=np.full(6, 7.0 / 0.075),
        ),
        source_hz=15.0,
        command_hz=120.0,
        horizon_source_steps=4,
        **overrides,
    )
    planner.reset(np.zeros(6))
    return planner


def _commit_cut(
    planner: SampleAlignedJerkLimitedJointTrajectoryPlanner,
    *,
    count: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_rows = [np.zeros(6)]
    qd_rows = [np.zeros(6)]
    qdd_rows = [np.zeros(6)]
    for index in range(count):
        proposal = planner.propose(monotonic_time=index / 120.0)
        q_rows.append(np.asarray(proposal.q_command, dtype=float))
        qd_rows.append(np.asarray(proposal.qd_command_rad_s, dtype=float))
        qdd_rows.append(np.asarray(proposal.qdd_command_rad_s2, dtype=float))
        planner.commit(proposal)
    return np.asarray(q_rows), np.asarray(qd_rows), np.asarray(qdd_rows)


def test_large_source_action_is_projected_and_action_endpoint_is_executed() -> None:
    planner = _planner()
    admission = planner.prepare_target(
        np.ones(6),
        source_timestamp=10.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )

    assert admission.new_source_bucket is True
    assert admission.metadata["admission_limited"] is True
    assert np.max(admission.admitted_qpos) < 0.02
    planner.activate_target(admission, sequence=1)
    q, qd, qdd = _commit_cut(planner)

    np.testing.assert_allclose(q[-1], admission.admitted_qpos, atol=1e-9)
    assert np.max(np.abs(qd[1:])) <= 1.4 + 1e-6
    assert np.max(np.abs(qdd[1:])) <= 7.0 + 1e-6
    jerk = np.diff(qdd, axis=0) * 120.0
    assert np.max(np.abs(jerk)) <= 7.0 / 0.075 + 1e-6


def test_intermediate_sixty_hz_targets_do_not_preempt_fifteen_hz_cut() -> None:
    planner = _planner()
    first = planner.prepare_target(
        np.full(6, 0.2),
        source_timestamp=5.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    planner.activate_target(first, sequence=1)

    for offset in (1.0 / 60.0, 2.0 / 60.0, 3.0 / 60.0):
        held = planner.prepare_target(
            np.full(6, 1.5),
            source_timestamp=5.0 + offset,
            speed_scale=1.0,
            mode="tracking",
            reason="normal",
        )
        assert held.new_source_bucket is False
        np.testing.assert_allclose(held.admitted_qpos, first.admitted_qpos)

    next_cut = planner.prepare_target(
        np.full(6, 0.4),
        source_timestamp=5.0 + 4.0 / 60.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    assert next_cut.new_source_bucket is True


def test_degraded_speed_scale_reduces_admitted_endpoint() -> None:
    normal = _planner().prepare_target(
        np.full(6, 0.02),
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    degraded = _planner().prepare_target(
        np.full(6, 0.02),
        source_timestamp=1.0,
        speed_scale=0.5,
        mode="degraded",
        reason="low_control_rate",
    )

    assert np.all(degraded.admitted_qpos <= normal.admitted_qpos + 1e-10)
    assert degraded.speed_scale == pytest.approx(0.5)


def test_v2_degraded_policy_preserves_absolute_target_for_hard_projection() -> None:
    normal = _planner(scale_degraded_target=False).prepare_target(
        np.full(6, 0.02),
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    degraded = _planner(scale_degraded_target=False).prepare_target(
        np.full(6, 0.02),
        source_timestamp=1.0,
        speed_scale=0.5,
        mode="degraded",
        reason="execution_step_limited",
    )

    np.testing.assert_allclose(degraded.admitted_qpos, normal.admitted_qpos)
    assert degraded.speed_scale == pytest.approx(0.5)
    assert degraded.metadata["policy_mode"] == "degraded"
    assert degraded.metadata["policy_reason"] == "execution_step_limited"
    assert degraded.metadata["degraded_target_scaled"] is False


def test_v2_qp_regularization_reduces_jerk_reversals_without_weakening_limits() -> None:
    baseline = _planner()
    smooth = _planner(
        scale_degraded_target=False,
        jerk_regularization=1e-6,
        jerk_delta_regularization=1e-5,
        terminal_velocity_regularization=0.0,
        terminal_acceleration_regularization=0.0,
    )
    target = np.full(6, 0.08)
    baseline_admission = baseline.prepare_target(
        target,
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    smooth_admission = smooth.prepare_target(
        target,
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    baseline_plan = baseline_admission.plan
    smooth_plan = smooth_admission.plan
    assert baseline_plan is not None
    assert smooth_plan is not None

    baseline_jerk = np.diff(baseline_plan.qdd_path_rad_s2, axis=0) * 120.0
    smooth_jerk = np.diff(smooth_plan.qdd_path_rad_s2, axis=0) * 120.0
    baseline_variation = np.sum(np.abs(np.diff(baseline_jerk, axis=0)))
    smooth_variation = np.sum(np.abs(np.diff(smooth_jerk, axis=0)))
    assert smooth_variation < baseline_variation
    assert np.max(np.abs(smooth_plan.qd_path_rad_s[1:])) <= 1.4 + 1e-6
    assert np.max(np.abs(smooth_plan.qdd_path_rad_s2[1:])) <= 7.0 + 1e-6
    assert np.max(np.abs(smooth_jerk)) <= 7.0 / 0.075 + 1e-6
    # The soft costs may trade a small amount of one-cut endpoint progress for
    # smoothness, but must not reintroduce the 50% semantic target shrink.
    assert np.max(smooth_admission.admitted_qpos) >= (
        0.8 * np.max(baseline_admission.admitted_qpos)
    )


def test_quick_stop_holds_without_advancing_cached_cut() -> None:
    planner = _planner()
    admission = planner.prepare_target(
        np.full(6, 0.3),
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    planner.activate_target(admission, sequence=1)
    moving = planner.propose(monotonic_time=0.0)
    planner.commit(moving)

    planner.quick_stop(sequence=2, reason="operator_hold")
    stopped = planner.propose(monotonic_time=1.0 / 120.0)
    np.testing.assert_allclose(stopped.q_command, moving.q_command)
    np.testing.assert_allclose(stopped.qd_command_rad_s, 0.0)
    np.testing.assert_allclose(stopped.qdd_command_rad_s2, 0.0)

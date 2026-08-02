from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control import (
    JointHorizonBatchedWorkspace,
    JointHorizonLimits,
    JointHorizonState,
    plan_jerk_limited_joint_horizon,
    plan_jerk_limited_joint_horizon_batched,
)


def _limits(dimension: int = 2) -> JointHorizonLimits:
    return JointHorizonLimits(
        min_position_rad=np.full(dimension, -2.0),
        max_position_rad=np.full(dimension, 2.0),
        max_velocity_rad_s=np.full(dimension, 1.0),
        max_acceleration_rad_s2=np.full(dimension, 2.0),
        max_jerk_rad_s3=np.full(dimension, 10.0),
    )


def test_horizon_governor_respects_inherited_third_order_state() -> None:
    dt = 0.01
    state = JointHorizonState(
        q_rad=np.zeros(2),
        qd_rad_s=np.array([0.2, -0.1]),
        qdd_rad_s2=np.array([0.3, -0.2]),
    )
    reference = np.repeat(np.array([[1.0, -1.0]]), 21, axis=0)
    reference[0] = state.q_rad

    plan = plan_jerk_limited_joint_horizon(
        reference,
        initial_state=state,
        limits=_limits(),
        dt_sec=dt,
        commit_count=10,
    )

    velocity = plan.qd_path_rad_s
    acceleration = plan.qdd_path_rad_s2
    jerk = np.diff(acceleration, axis=0) / dt
    assert np.max(np.abs(velocity[1:])) <= 1.0 + 1e-6
    assert np.max(np.abs(acceleration[1:])) <= 2.0 + 1e-6
    assert np.max(np.abs(jerk)) <= 10.0 + 1e-5
    np.testing.assert_allclose(velocity[0], state.qd_rad_s)
    np.testing.assert_allclose(acceleration[0], state.qdd_rad_s2)
    assert plan.q_control_cut_rad.shape == (10, 2)
    assert (
        plan.metadata["planner"]
        == "jerk_limited_joint_horizon_qp_v2_jerk_only"
    )


def test_horizon_governor_is_a_candidate_and_does_not_mutate_inputs() -> None:
    state_q = np.zeros(1)
    reference = np.linspace(0.0, 0.2, 9).reshape(-1, 1)
    plan = plan_jerk_limited_joint_horizon(
        reference,
        initial_state=JointHorizonState(
            q_rad=state_q,
            qd_rad_s=np.zeros(1),
            qdd_rad_s2=np.zeros(1),
        ),
        limits=_limits(1),
        dt_sec=0.01,
        commit_count=4,
    )

    np.testing.assert_array_equal(state_q, np.zeros(1))
    np.testing.assert_allclose(reference[:, 0], np.linspace(0.0, 0.2, 9))
    assert not plan.q_path_rad.flags.writeable
    with pytest.raises(ValueError):
        plan.q_path_rad[0, 0] = 1.0


def test_batched_horizon_governor_matches_independent_joint_solver() -> None:
    dt = 1.0 / 120.0
    state = JointHorizonState(
        q_rad=np.array([0.1, -0.2, 0.05]),
        qd_rad_s=np.array([0.08, -0.04, 0.02]),
        qdd_rad_s2=np.array([0.15, -0.1, 0.05]),
    )
    reference = np.column_stack(
        (
            np.linspace(0.1, 0.35, 17),
            np.linspace(-0.2, -0.05, 17),
            np.linspace(0.05, -0.1, 17),
        )
    )
    limits = _limits(3)

    independent = plan_jerk_limited_joint_horizon(
        reference,
        initial_state=state,
        limits=limits,
        dt_sec=dt,
        commit_count=8,
        solver_tolerance=1e-7,
    )
    batched = plan_jerk_limited_joint_horizon_batched(
        reference,
        initial_state=state,
        limits=limits,
        dt_sec=dt,
        commit_count=8,
        solver_tolerance=1e-7,
    )

    np.testing.assert_allclose(
        batched.q_path_rad,
        independent.q_path_rad,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        batched.qd_path_rad_s,
        independent.qd_path_rad_s,
        rtol=0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        batched.qdd_path_rad_s2,
        independent.qdd_path_rad_s2,
        rtol=0.0,
        atol=2e-4,
    )
    assert batched.metadata["planner"] == "jerk_limited_joint_horizon_qp_v3_batched"
    assert all(
        receipt["hard_constraint_certificate_passed"]
        for receipt in batched.metadata["solver_receipts"]
    )

    workspace = JointHorizonBatchedWorkspace(
        limits=limits,
        dt_sec=dt,
        horizon_transitions=16,
        solver_tolerance=1e-7,
    )
    reused = workspace.solve(
        reference,
        initial_state=state,
        commit_count=8,
    )
    np.testing.assert_allclose(
        reused.q_path_rad,
        batched.q_path_rad,
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        reused.qd_path_rad_s,
        batched.qd_path_rad_s,
        rtol=0.0,
        atol=2e-5,
    )
    np.testing.assert_allclose(
        reused.qdd_path_rad_s2,
        batched.qdd_path_rad_s2,
        rtol=0.0,
        atol=2e-4,
    )
    assert reused.metadata["solver_workspace_reused"] is True


def test_horizon_governor_rejects_invalid_contracts() -> None:
    state = JointHorizonState(
        q_rad=np.zeros(2),
        qd_rad_s=np.zeros(2),
        qdd_rad_s2=np.zeros(2),
    )
    with pytest.raises(ValueError, match="reference_qpos"):
        plan_jerk_limited_joint_horizon(
            np.zeros((1, 2)),
            initial_state=state,
            limits=_limits(),
            dt_sec=0.01,
            commit_count=1,
        )
    with pytest.raises(ValueError, match="commit_count"):
        plan_jerk_limited_joint_horizon(
            np.zeros((3, 2)),
            initial_state=state,
            limits=_limits(),
            dt_sec=0.01,
            commit_count=3,
        )


def test_horizon_governor_clamps_only_numeric_position_limit_noise() -> None:
    limits = _limits(1)
    reference = np.full((5, 1), 2.0)
    plan = plan_jerk_limited_joint_horizon(
        reference,
        initial_state=JointHorizonState(
            q_rad=np.array([2.0 + 1e-8]),
            qd_rad_s=np.zeros(1),
            qdd_rad_s2=np.zeros(1),
        ),
        limits=limits,
        dt_sec=0.01,
        commit_count=2,
    )
    assert plan.q_path_rad[0, 0] == pytest.approx(2.0)
    assert np.all(plan.q_path_rad <= limits.max_position_rad)
    assert np.all(plan.q_path_rad >= limits.min_position_rad)

    with pytest.raises(ValueError, match="outside"):
        plan_jerk_limited_joint_horizon(
            reference,
            initial_state=JointHorizonState(
                q_rad=np.array([2.0 + 1e-4]),
                qd_rad_s=np.zeros(1),
                qdd_rad_s2=np.zeros(1),
            ),
            limits=limits,
            dt_sec=0.01,
            commit_count=2,
        )


def test_horizon_governor_rejects_state_outside_jerk_viability_kernel() -> None:
    """The ep34 state is inside box limits but cannot brake before v-limit."""

    dt = 1.0 / 120.0
    max_velocity = np.array([5.6])
    max_acceleration = np.array([56.0])
    limits = JointHorizonLimits(
        min_position_rad=np.radians([-178.0]),
        max_position_rad=np.radians([178.0]),
        max_velocity_rad_s=max_velocity,
        max_acceleration_rad_s2=max_acceleration,
        max_jerk_rad_s3=max_acceleration / 0.1,
    )
    state = JointHorizonState(
        q_rad=np.radians([-59.35845658]),
        qd_rad_s=np.radians([-192.29615]),
        qdd_rad_s2=np.radians([-3125.5985]),
    )
    reference = np.linspace(
        state.q_rad[0],
        state.q_rad[0] - np.radians(20.0),
        17,
    ).reshape(-1, 1)

    with pytest.raises(RuntimeError, match="primal infeasible"):
        plan_jerk_limited_joint_horizon(
            reference,
            initial_state=state,
            limits=limits,
            dt_sec=dt,
            commit_count=8,
        )


def test_horizon_governor_accepts_high_recoverable_state() -> None:
    dt = 1.0 / 120.0
    max_velocity = np.array([5.6])
    max_acceleration = np.array([56.0])
    limits = JointHorizonLimits(
        min_position_rad=np.radians([-178.0]),
        max_position_rad=np.radians([178.0]),
        max_velocity_rad_s=max_velocity,
        max_acceleration_rad_s2=max_acceleration,
        max_jerk_rad_s3=max_acceleration / 0.1,
    )
    state = JointHorizonState(
        q_rad=np.radians([-59.0]),
        qd_rad_s=np.radians([-192.0]),
        qdd_rad_s2=np.radians([-2400.0]),
    )
    reference = np.linspace(
        state.q_rad[0],
        state.q_rad[0] - np.radians(20.0),
        17,
    ).reshape(-1, 1)

    plan = plan_jerk_limited_joint_horizon(
        reference,
        initial_state=state,
        limits=limits,
        dt_sec=dt,
        commit_count=8,
    )

    assert np.max(np.abs(plan.qd_path_rad_s[1:])) <= max_velocity[0] + 1e-8
    assert (
        np.max(np.abs(plan.qdd_path_rad_s2[1:]))
        <= max_acceleration[0] + 1e-8
    )

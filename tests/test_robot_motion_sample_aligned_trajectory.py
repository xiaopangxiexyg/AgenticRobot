from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control import (
    JointHorizonLimits,
    JointHorizonPlan,
    SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
    SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
    SampleAlignedJerkLimitedJointTrajectoryPlanner,
)


def _planner(**overrides) -> SampleAlignedJerkLimitedJointTrajectoryPlanner:
    source_hz = float(overrides.pop("source_hz", 15.0))
    command_hz = float(overrides.pop("command_hz", 120.0))
    horizon_source_steps = int(overrides.pop("horizon_source_steps", 4))
    planner = SampleAlignedJerkLimitedJointTrajectoryPlanner(
        JointHorizonLimits(
            min_position_rad=np.full(6, -2.0),
            max_position_rad=np.full(6, 2.0),
            max_velocity_rad_s=np.full(6, 1.4),
            max_acceleration_rad_s2=np.full(6, 7.0),
            max_jerk_rad_s3=np.full(6, 7.0 / 0.075),
        ),
        source_hz=source_hz,
        command_hz=command_hz,
        horizon_source_steps=horizon_source_steps,
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


def test_split_horizon_uses_one_long_safety_qp_and_fails_closed() -> None:
    planner = _planner(safety_horizon_source_steps=6)
    admission = planner.prepare_target(
        np.full(6, 0.02),
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="split_horizon_test",
    )
    assert admission.plan is not None
    assert admission.plan.q_path_rad.shape[0] == 6 * 8 + 1
    assert admission.plan.metadata["horizon_transitions"] == 6 * 8
    assert admission.plan.metadata["tracking_horizon_transitions"] == 4 * 8
    assert admission.metadata[
        "sample_aligned_safety_horizon_source_steps"
    ] == 6
    assert admission.metadata[
        "sample_aligned_tracking_horizon_source_steps"
    ] == 4

    original_solve = planner._workspace.solve

    def fail_safety_horizon(*_args, **_kwargs):
        raise RuntimeError("synthetic safety-horizon infeasible")

    planner._workspace.solve = fail_safety_horizon
    with pytest.raises(RuntimeError, match="synthetic safety-horizon infeasible"):
        planner.prepare_target(
            np.full(6, 0.03),
            source_timestamp=2.0,
            speed_scale=1.0,
            mode="tracking",
            reason="split_horizon_test",
        )
    planner._workspace.solve = original_solve

def test_split_horizon_remains_certified_when_next_source_is_late() -> None:
    planner = _planner(safety_horizon_source_steps=6)
    admission = planner.prepare_target(
        np.full(6, 0.2),
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="split_horizon_test",
    )
    planner.activate_target(admission, sequence=1)
    _, _, qdd = _commit_cut(planner)

    extension = planner.propose(monotonic_time=8.0 / 120.0)
    assert extension.metadata["sample_aligned_plan_row"] == 9
    np.testing.assert_allclose(extension.q_target, admission.admitted_qpos)
    boundary_jerk = (
        np.asarray(extension.qdd_command_rad_s2) - qdd[-1]
    ) * 120.0
    assert np.max(np.abs(boundary_jerk)) <= 7.0 / 0.075 + 1e-6
    planner.commit(extension)
    snapshot = planner.snapshot()
    assert snapshot["safety_horizon_source_steps"] == 6


def test_split_horizon_contract_is_default_off_and_linear_hold_only() -> None:
    planner = _planner()
    assert planner.snapshot()["safety_horizon_source_steps"] is None

    with pytest.raises(ValueError, match="greater than horizon_source_steps"):
        _planner(safety_horizon_source_steps=4)
    with pytest.raises(ValueError, match="requires reference_mode='linear_hold'"):
        _planner(
            safety_horizon_source_steps=6,
            reference_mode="causal_hermite",
        )


def test_split_horizon_static_target_converges_without_a_tail_limit_cycle() -> None:
    """A stationary right-arm target must not keep reversing the qcmd tail.

    The formal 60 Hz profile commits two 120 Hz commands per source cut while
    certifying a 20-source-step safety horizon.  Before the tail objective was
    strengthened, a constant J3 target settled into a roughly 0.2 rad limit
    cycle because only the first four transitions carried tracking cost.
    """

    limits = JointHorizonLimits(
        min_position_rad=np.array([-2.0]),
        max_position_rad=np.array([2.0]),
        max_velocity_rad_s=np.array([7.2]),
        max_acceleration_rad_s2=np.array([72.0]),
        max_jerk_rad_s3=np.array([1440.0]),
    )
    planner = SampleAlignedJerkLimitedJointTrajectoryPlanner(
        limits,
        source_hz=60.0,
        command_hz=120.0,
        source_latch_mode=SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
        horizon_source_steps=2,
        safety_horizon_source_steps=20,
        scale_degraded_target=False,
        jerk_regularization=1e-6,
        jerk_delta_regularization=1e-4,
    )
    planner.reset(np.zeros(1))
    target = np.array([0.35])
    q_rows: list[float] = []
    qd_rows: list[float] = []
    last_admission = None
    for source_index in range(120):
        admission = planner.prepare_target(
            target,
            source_timestamp=source_index / 60.0,
            speed_scale=1.0,
            mode="tracking",
            reason="stationary_target_regression",
        )
        last_admission = admission
        if admission.new_source_bucket:
            planner.activate_target(admission, sequence=source_index + 1)
        for command_index in range(planner.commands_per_source):
            proposal = planner.propose(
                monotonic_time=(
                    source_index * planner.commands_per_source + command_index
                )
                / planner.command_hz
            )
            q_rows.append(float(proposal.q_command[0]))
            qd_rows.append(float(proposal.qd_command_rad_s[0]))
            planner.commit(proposal)

    assert last_admission is not None and last_admission.plan is not None
    assert last_admission.plan.metadata["tail_tracking_weight"] == pytest.approx(
        0.03
    )
    # The final half-second is the stationary portion, not the initial move.
    assert np.max(np.abs(np.asarray(q_rows[-60:]) - target[0])) < 1e-6
    assert np.max(np.abs(np.asarray(qd_rows[-60:]))) < 1e-4


def test_certified_qp_position_roundoff_is_projected_inside_transport_limits() -> None:
    planner = _planner()
    horizon_rows = planner.horizon_transitions + 1
    q_path = np.zeros((horizon_rows, 6), dtype=float)
    q_path[:, 2] = 2.0 - 1.0e-7
    q_path[1:, 2] = 2.0 + 1.0e-6
    dt_sec = 1.0 / planner.command_hz
    qd_path = np.zeros_like(q_path)
    qdd_path = np.zeros_like(q_path)
    qd_path[1:] = np.diff(q_path, axis=0) / dt_sec
    qdd_path[1:] = np.diff(qd_path, axis=0) / dt_sec
    planner.reset(q_path[0])
    plan = JointHorizonPlan(
        q_path_rad=q_path,
        qd_path_rad_s=qd_path,
        qdd_path_rad_s2=qdd_path,
        commit_count=planner.commands_per_source,
        dt_sec=dt_sec,
        metadata={"planner": "fake_certified_qp"},
    )
    planner._workspace.solve = lambda *_args, **_kwargs: plan

    admission = planner.prepare_target(
        np.full(6, 2.0),
        source_timestamp=10.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )

    assert admission.plan is not None
    assert np.max(admission.plan.q_path_rad[1:, 2]) == pytest.approx(
        2.0 - 1.0e-7
    )
    assert admission.admitted_qpos[2] < 2.0
    assert admission.plan.metadata["position_output_roundoff_projected"] is True
    assert admission.plan.metadata["position_output_dynamics_reconstructed"] is True
    assert admission.plan.metadata["position_output_max_projection_rad"] == pytest.approx(
        1.1e-6
    )
    np.testing.assert_allclose(
        admission.plan.qd_path_rad_s[1:],
        np.diff(admission.plan.q_path_rad, axis=0) / dt_sec,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        admission.plan.qdd_path_rad_s2[1:],
        np.diff(admission.plan.qd_path_rad_s, axis=0) / dt_sec,
        rtol=0.0,
        atol=1e-12,
    )
    assert admission.metadata["governor_position_output_roundoff_projected"] is True
    assert admission.metadata[
        "governor_position_output_max_projection_rad"
    ] == pytest.approx(1.1e-6)


def test_qp_position_violation_beyond_numeric_budget_remains_fail_closed() -> None:
    planner = _planner()
    horizon_rows = planner.horizon_transitions + 1
    q_path = np.zeros((horizon_rows, 6), dtype=float)
    q_path[:, 2] = 2.0 - 1.0e-7
    q_path[1:, 2] = 2.0 + 3.0e-6
    planner.reset(q_path[0])
    plan = JointHorizonPlan(
        q_path_rad=q_path,
        qd_path_rad_s=np.zeros_like(q_path),
        qdd_path_rad_s2=np.zeros_like(q_path),
        commit_count=planner.commands_per_source,
        dt_sec=1.0 / planner.command_hz,
        metadata={"planner": "invalid_qp"},
    )
    planner._workspace.solve = lambda *_args, **_kwargs: plan

    with pytest.raises(RuntimeError, match="exceeds numeric projection budget"):
        planner.prepare_target(
            np.full(6, 2.0),
            source_timestamp=10.0,
            speed_scale=1.0,
            mode="tracking",
            reason="normal",
        )


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


def test_strict_timestamp_latch_accepts_each_jittered_sixty_hz_source_once() -> None:
    timestamps = (100.0, 100.012, 100.033, 100.046, 100.067)
    strict = _planner(
        source_hz=60.0,
        horizon_source_steps=2,
        source_latch_mode=SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
    )
    strict_buckets = []
    for index, timestamp in enumerate(timestamps, start=1):
        admission = strict.prepare_target(
            np.full(6, 0.002 * index),
            source_timestamp=timestamp,
            speed_scale=1.0,
            mode="tracking",
            reason="jittered_60hz_source",
        )
        assert admission.new_source_bucket is True
        assert admission.metadata["source_latch_mode"] == (
            SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
        )
        strict_buckets.append(admission.source_bucket)
        strict.activate_target(admission, sequence=index)
        _commit_cut(strict, count=strict.commands_per_source)
    assert strict_buckets == [0, 1, 2, 3, 4]

    duplicate = strict.prepare_target(
        np.full(6, 0.5),
        source_timestamp=timestamps[-1],
        speed_scale=1.0,
        mode="tracking",
        reason="duplicate_source",
    )
    assert duplicate.new_source_bucket is False
    assert duplicate.metadata["source_hold_reason"] == (
        "source_timestamp_not_increasing"
    )
    older = strict.prepare_target(
        np.full(6, -0.5),
        source_timestamp=timestamps[-1] - 0.001,
        speed_scale=1.0,
        mode="tracking",
        reason="out_of_order_source",
    )
    assert older.new_source_bucket is False

    bucketed = _planner(
        source_hz=60.0,
        horizon_source_steps=2,
        source_latch_mode=SAMPLE_ALIGNED_SOURCE_LATCH_TIME_BUCKET,
    )
    bucketed_new = []
    for index, timestamp in enumerate(timestamps, start=1):
        admission = bucketed.prepare_target(
            np.full(6, 0.002 * index),
            source_timestamp=timestamp,
            speed_scale=1.0,
            mode="tracking",
            reason="legacy_time_bucket",
        )
        bucketed_new.append(admission.new_source_bucket)
        if admission.new_source_bucket:
            bucketed.activate_target(admission, sequence=index)
            _commit_cut(bucketed, count=bucketed.commands_per_source)
    assert bucketed_new == [True, False, True, True, True]


def test_causal_observer_uses_intermediate_targets_without_preempting_cut() -> None:
    planner = _planner(
        scale_degraded_target=False,
        jerk_regularization=1e-6,
        jerk_delta_regularization=1e-5,
        initial_jerk_delta_regularization=1e-5,
        reference_mode="causal_hermite",
        observer_window=4,
        terminal_velocity_displacement_scale=0.75,
    )
    first = planner.prepare_target(
        np.zeros(6),
        source_timestamp=5.0,
        speed_scale=1.0,
        mode="tracking",
        reason="normal",
    )
    planner.activate_target(first, sequence=1)
    _commit_cut(planner)

    for index in range(1, 5):
        admission = planner.prepare_target(
            np.full(6, 0.005 * index),
            source_timestamp=5.0 + index / 60.0,
            speed_scale=1.0,
            mode="tracking",
            reason="normal",
        )
        if index < 4:
            assert admission.new_source_bucket is False
        else:
            assert admission.new_source_bucket is True
            assert (
                admission.metadata["sample_aligned_observer_fallback_reason"]
                == "source_velocity_incoherent"
            )
            planner.activate_target(admission, sequence=2)
    _commit_cut(planner)

    for index in range(5, 9):
        admission = planner.prepare_target(
            np.full(6, 0.005 * index),
            source_timestamp=5.0 + index / 60.0,
            speed_scale=1.0,
            mode="tracking",
            reason="normal",
        )
    assert admission.new_source_bucket is True
    assert admission.metadata["sample_aligned_observer_fallback_reason"] is None
    terminal_velocity = np.asarray(
        admission.metadata["sample_aligned_terminal_velocity_rad_s"],
        dtype=float,
    )
    assert np.all(terminal_velocity > 0.0)
    assert np.max(terminal_velocity) <= 0.5 * 1.4 + 1e-9
    assert admission.plan is not None
    jerk = np.diff(admission.plan.qdd_path_rad_s2, axis=0) * 120.0
    assert np.max(np.abs(jerk)) <= 7.0 / 0.075 + 1e-6


def test_external_causal_velocity_is_explicit_and_fail_closed() -> None:
    planner = _planner(
        scale_degraded_target=False,
        jerk_regularization=1e-6,
        jerk_delta_regularization=1e-5,
        reference_mode="causal_hermite_selective",
        observer_velocity_ema_alpha=0.5,
        observer_velocity_cap_scale=0.5,
        observer_coherence_min=0.4,
        observer_speed_guard_scale=0.5,
        terminal_velocity_displacement_scale=0.75,
    )
    velocity = np.full(6, 0.10, dtype=float)
    first = planner.prepare_target(
        np.full(6, 0.01),
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="external_velocity_test",
        requested_velocity_rad_s=velocity,
    )
    assert first.metadata["sample_aligned_observer_source"] == (
        "external_causal_velocity"
    )
    assert first.metadata["sample_aligned_observer_fallback_reason"] == (
        "source_velocity_incoherent"
    )
    planner.activate_target(first, sequence=1)
    _commit_cut(planner)

    second = planner.prepare_target(
        np.full(6, 0.02),
        source_timestamp=1.0 + 1.0 / 15.0,
        speed_scale=1.0,
        mode="tracking",
        reason="external_velocity_test",
        requested_velocity_rad_s=velocity,
    )
    assert second.metadata["sample_aligned_observer_source"] == (
        "external_causal_velocity"
    )
    assert second.metadata["sample_aligned_observer_fallback_reason"] is None
    assert np.all(
        np.asarray(
            second.metadata["sample_aligned_terminal_velocity_rad_s"],
            dtype=float,
        )
        > 0.0
    )

    with pytest.raises(ValueError, match="requested_velocity_rad_s"):
        planner.prepare_target(
            np.full(6, 0.03),
            source_timestamp=1.0 + 2.0 / 15.0,
            speed_scale=1.0,
            mode="tracking",
            reason="external_velocity_test",
            requested_velocity_rad_s=np.full(6, np.nan),
        )


def test_linear_hold_is_numerically_unchanged_by_unused_external_velocity() -> None:
    baseline = _planner(reference_mode="linear_hold")
    with_velocity = _planner(reference_mode="linear_hold")
    target = np.asarray([0.04, -0.03, 0.02, -0.01, 0.03, -0.02])
    baseline_admission = baseline.prepare_target(
        target,
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="baseline",
    )
    velocity_admission = with_velocity.prepare_target(
        target,
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="baseline",
        requested_velocity_rad_s=np.full(6, 0.2),
    )
    assert baseline_admission.plan is not None
    assert velocity_admission.plan is not None
    np.testing.assert_array_equal(
        velocity_admission.plan.q_path_rad,
        baseline_admission.plan.q_path_rad,
    )
    np.testing.assert_array_equal(
        velocity_admission.plan.qd_path_rad_s,
        baseline_admission.plan.qd_path_rad_s,
    )
    np.testing.assert_array_equal(
        velocity_admission.plan.qdd_path_rad_s2,
        baseline_admission.plan.qdd_path_rad_s2,
    )


def test_invalid_external_velocity_falls_back_to_exact_linear_hold() -> None:
    baseline = _planner(reference_mode="linear_hold_nullspace_smooth")
    assisted = _planner(reference_mode="causal_hermite_nullspace_smooth")
    target = np.asarray([0.04, -0.03, 0.02, -0.01, 0.03, -0.02])
    baseline_admission = baseline.prepare_target(
        target,
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="baseline",
    )
    assisted_admission = assisted.prepare_target(
        target,
        source_timestamp=1.0,
        speed_scale=1.0,
        mode="tracking",
        reason="invalid_external_velocity",
        requested_velocity_rad_s=np.full(6, 0.2),
        requested_velocity_valid=False,
    )
    assert baseline_admission.plan is not None
    assert assisted_admission.plan is not None
    np.testing.assert_array_equal(
        assisted_admission.plan.q_path_rad,
        baseline_admission.plan.q_path_rad,
    )
    np.testing.assert_array_equal(
        assisted_admission.plan.qd_path_rad_s,
        baseline_admission.plan.qd_path_rad_s,
    )
    np.testing.assert_array_equal(
        assisted_admission.plan.qdd_path_rad_s2,
        baseline_admission.plan.qdd_path_rad_s2,
    )
    assert assisted_admission.metadata[
        "sample_aligned_observer_fallback_reason"
    ] == "external_velocity_invalid"
    assert assisted_admission.metadata["sample_aligned_observer_used"] is False


def test_external_velocity_validity_requires_external_velocity() -> None:
    planner = _planner(reference_mode="causal_hermite_nullspace_smooth")
    with pytest.raises(ValueError, match="requested_velocity_valid"):
        planner.prepare_target(
            np.zeros(6),
            source_timestamp=1.0,
            speed_scale=1.0,
            mode="tracking",
            reason="invalid_contract",
            requested_velocity_valid=False,
        )


def test_nullspace_smoothing_preserves_terminal_dynamic_anchor_and_limits() -> None:
    common = {
        "scale_degraded_target": False,
        "jerk_regularization": 1e-6,
        "jerk_delta_regularization": 1e-5,
    }
    baseline = _planner(reference_mode="linear_hold", **common)
    smooth = _planner(
        reference_mode="linear_hold_nullspace_smooth",
        **common,
    )
    target = np.asarray([0.08, -0.06, 0.04, -0.03, 0.02, -0.01])
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
    commit = baseline_plan.commit_count
    np.testing.assert_array_equal(
        smooth_plan.q_path_rad[commit],
        baseline_plan.q_path_rad[commit],
    )
    np.testing.assert_array_equal(
        smooth_plan.qd_path_rad_s[commit],
        baseline_plan.qd_path_rad_s[commit],
    )
    np.testing.assert_array_equal(
        smooth_plan.qdd_path_rad_s2[commit],
        baseline_plan.qdd_path_rad_s2[commit],
    )
    np.testing.assert_array_equal(
        smooth_admission.admitted_qpos,
        baseline_admission.admitted_qpos,
    )
    assert smooth_admission.metadata[
        "sample_aligned_nullspace_smoothing_terminal_anchor"
    ] == "exact_q_qd_qdd"
    smooth_jerk = (
        np.diff(smooth_plan.qdd_path_rad_s2[: commit + 1], axis=0) * 120.0
    )
    assert np.max(np.abs(smooth_jerk)) <= 7.0 / 0.075 + 1e-6
    assert np.max(np.abs(smooth_plan.qd_path_rad_s[: commit + 1])) <= 1.4 + 1e-6
    assert np.max(np.abs(smooth_plan.qdd_path_rad_s2[: commit + 1])) <= 7.0 + 1e-6


def test_nullspace_smoothing_selects_only_verified_nonreversing_improvement() -> None:
    common = {
        "scale_degraded_target": False,
        "jerk_regularization": 1e-6,
        "jerk_delta_regularization": 1e-5,
    }
    baseline = _planner(reference_mode="linear_hold", **common)
    smooth = _planner(
        reference_mode="linear_hold_nullspace_smooth",
        **common,
    )
    rng = np.random.default_rng(7)
    timestamp = 1.0
    selected_admission = None
    baseline_admission = None

    for index in range(40):
        target = np.clip(
            (index + 1)
            * np.asarray([0.004, -0.003, 0.0025, -0.002, 0.0015, -0.001])
            + rng.normal(0.0, 0.003, 6),
            -0.3,
            0.3,
        )
        baseline_admission = baseline.prepare_target(
            target,
            source_timestamp=timestamp,
            speed_scale=1.0,
            mode="tracking",
            reason="normal",
        )
        smooth_admission = smooth.prepare_target(
            target,
            source_timestamp=timestamp,
            speed_scale=1.0,
            mode="tracking",
            reason="normal",
        )
        if smooth_admission.metadata[
            "sample_aligned_nullspace_smoothing_selected"
        ]:
            selected_admission = smooth_admission
            break
        baseline.activate_target(baseline_admission, sequence=index + 1)
        smooth.activate_target(smooth_admission, sequence=index + 1)
        _commit_cut(baseline)
        _commit_cut(smooth)
        timestamp += 1.0 / 15.0

    assert selected_admission is not None
    assert baseline_admission is not None
    baseline_plan = baseline_admission.plan
    smooth_plan = selected_admission.plan
    assert baseline_plan is not None
    assert smooth_plan is not None
    commit = baseline_plan.commit_count
    np.testing.assert_array_equal(
        smooth_plan.q_path_rad[commit],
        baseline_plan.q_path_rad[commit],
    )
    np.testing.assert_array_equal(
        smooth_plan.qd_path_rad_s[commit],
        baseline_plan.qd_path_rad_s[commit],
    )
    np.testing.assert_array_equal(
        smooth_plan.qdd_path_rad_s2[commit],
        baseline_plan.qdd_path_rad_s2[commit],
    )

    baseline_scores = np.asarray(
        selected_admission.metadata[
            "sample_aligned_nullspace_smoothing_baseline_score_by_joint"
        ],
        dtype=float,
    )
    selected_scores = np.asarray(
        selected_admission.metadata[
            "sample_aligned_nullspace_smoothing_selected_score_by_joint"
        ],
        dtype=float,
    )
    selected_joints = np.asarray(
        selected_admission.metadata[
            "sample_aligned_nullspace_smoothing_alpha_by_joint"
        ],
        dtype=float,
    ) > 0.0
    assert np.any(selected_joints)
    assert np.all(
        selected_scores[selected_joints]
        <= 0.98 * baseline_scores[selected_joints]
    )

    previous_step = baseline_plan.qd_path_rad_s[0] / 120.0
    baseline_steps = np.vstack(
        (previous_step, np.diff(baseline_plan.q_path_rad[: commit + 1], axis=0))
    )
    smooth_steps = np.vstack(
        (previous_step, np.diff(smooth_plan.q_path_rad[: commit + 1], axis=0))
    )
    active = np.abs(baseline_steps) >= np.deg2rad(0.002)
    np.testing.assert_array_equal(
        np.abs(smooth_steps) >= np.deg2rad(0.002),
        active,
    )
    np.testing.assert_array_equal(
        np.sign(smooth_steps[active]),
        np.sign(baseline_steps[active]),
    )


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

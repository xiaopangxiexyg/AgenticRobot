from __future__ import annotations

import numpy as np
import pytest

from robot_motion.position_control import (
    AccelerationLimitedJointTrajectoryPlanner,
    JointTrajectoryLimits,
    JointTrajectoryProposal,
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    normalize_position_command_planner,
)


def _vector_proposal(value: np.ndarray, **overrides: object) -> JointTrajectoryProposal:
    kwargs = dict(
        token=0, generation=0, target_sequence=0, sample_monotonic_time=0.0,
        integration_dt_sec=1 / 120, q_command=value, q_target=value,
        qd_command_rad_s=value, qd_reference_rad_s=value, qd_desired_rad_s=value,
        qdd_command_rad_s2=value, speed_scale=1.0, mode="tracking", reason="normal",
        quick_stop=False, acceleration_limited=True, velocity_limited=True,
        command_step_limited=True, safety_envelope_limited=False, metadata={},
    )
    return JointTrajectoryProposal(**{**kwargs, **overrides})


@pytest.mark.parametrize("dimension", [2, 6, 7, 24])
def test_command_proposal_keeps_exact_detached_independent_readonly_vectors(dimension: int) -> None:
    source = np.linspace(-0.3, 0.4, dimension * 2)[::2]
    source[0] = -0.0
    source[1] = np.nextafter(0.0, 1.0)
    expected = source.copy()
    proposal = _vector_proposal(source)
    fields = ("q_command", "q_target", "qd_command_rad_s", "qd_reference_rad_s",
              "qd_desired_rad_s", "qdd_command_rad_s2")
    source[:] = 7.0
    for index, name in enumerate(fields):
        result = getattr(proposal, name)
        np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))
        assert result.dtype == np.float64
        assert not result.flags.writeable
        assert not np.shares_memory(result, source)
        assert all(not np.shares_memory(result, getattr(proposal, other)) for other in fields[index+1:])
        with pytest.raises(ValueError, match="read-only"):
            result[0] = 1.0


@pytest.mark.parametrize("field", ["q_command", "q_target", "qd_command_rad_s", "qd_reference_rad_s",
                                  "qd_desired_rad_s", "qdd_command_rad_s2"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_six_joint_command_proposal_rejects_nonfinite_in_every_vector(field: str, bad: float) -> None:
    invalid = np.zeros(6)
    invalid[-1] = bad
    with pytest.raises(ValueError, match=f"{field} must be a non-empty finite vector"):
        _vector_proposal(np.zeros(6), **{field: invalid})


def _planner(
    *,
    max_velocity: float = 1.0,
    max_acceleration: float = 10.0,
    max_step: float | None = None,
    target_velocity_filter_tau_sec: float = 0.0,
) -> AccelerationLimitedJointTrajectoryPlanner:
    limits = JointTrajectoryLimits(
        max_velocity_rad_s=np.full(2, max_velocity),
        max_acceleration_rad_s2=np.full(2, max_acceleration),
        max_command_step_rad=(
            None if max_step is None else np.full(2, max_step)
        ),
    )
    planner = AccelerationLimitedJointTrajectoryPlanner(
        limits,
        velocity_ramp_sec=0.1,
        target_velocity_filter_tau_sec=target_velocity_filter_tau_sec,
    )
    planner.reset(np.zeros(2), monotonic_time=0.0)
    return planner


def _accept(
    planner: AccelerationLimitedJointTrajectoryPlanner,
    target: tuple[float, float],
    *,
    sequence: int,
    now: float,
    speed_scale: float = 1.0,
    mode: str = "tracking",
) -> None:
    planner.accept_target(
        target,
        sequence=sequence,
        published_monotonic_time=now,
        monotonic_time=now,
        speed_scale=speed_scale,
        mode=mode,
        reason=mode,
    )


def test_position_command_planner_names_are_normalized() -> None:
    assert normalize_position_command_planner("linear") == POSITION_COMMAND_PLANNER_LINEAR
    assert (
        normalize_position_command_planner("velocity")
        == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    )
    assert (
        normalize_position_command_planner("sample_aligned")
        == POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
    )
    with pytest.raises(ValueError, match="Unsupported position command planner"):
        normalize_position_command_planner("cubic_canfd")


def test_acceleration_limited_planner_respects_velocity_and_acceleration() -> None:
    planner = _planner(max_velocity=1.0, max_acceleration=2.0)
    _accept(planner, (1.0, -1.0), sequence=1, now=0.0)

    previous_velocity = np.zeros(2)
    now = 0.0
    for _ in range(30):
        now += 0.01
        proposal = planner.propose(monotonic_time=now)
        assert np.all(np.abs(proposal.qd_command_rad_s) <= 1.0 + 1e-12)
        acceleration = (proposal.qd_command_rad_s - previous_velocity) / 0.01
        assert np.all(np.abs(acceleration) <= 2.0 + 1e-9)
        planner.commit(proposal, accepted_monotonic_time=now)
        previous_velocity = proposal.qd_command_rad_s


def test_reference_pause_decelerates_and_resumes_without_reanchoring_endpoint():
    planner = _planner(max_velocity=.1, max_acceleration=.2)
    _accept(planner, (.5, -.5), sequence=1, now=0.)
    for i in range(1, 21):
        proposal = planner.propose(monotonic_time=i*.01)
        planner.commit(proposal, accepted_monotonic_time=i*.01)
    before = planner.snapshot()
    planner.pause_reference(monotonic_time=.2, reason="observed_lead_reserve")
    assert planner.snapshot()["q_command"] == before["q_command"]
    assert planner.snapshot()["qd_command_rad_s"] == before["qd_command_rad_s"]
    previous = np.asarray(before["qd_command_rad_s"])
    previous_q = np.asarray(before["q_command"])
    for i in range(21, 81):
        proposal = planner.propose(monotonic_time=i*.01)
        np.testing.assert_allclose(proposal.q_target, (.5, -.5))
        assert np.max(np.abs(proposal.qd_command_rad_s-previous)) <= .002 + 1e-12
        np.testing.assert_allclose(proposal.q_command, previous_q + .01*proposal.qd_command_rad_s)
        planner.commit(proposal, accepted_monotonic_time=i*.01)
        previous, previous_q = proposal.qd_command_rad_s, proposal.q_command
    np.testing.assert_array_equal(previous, (0., 0.))
    stopped_q = previous_q.copy()
    _accept(planner, (.5, -.5), sequence=2, now=.8)
    proposal = planner.propose(monotonic_time=.81)
    assert np.max(np.abs(proposal.qd_command_rad_s)) <= .002 + 1e-12
    np.testing.assert_allclose(proposal.q_command, stopped_q + .01*proposal.qd_command_rad_s)


def test_reference_pause_requires_acknowledged_tracking_and_keeps_quick_stop_terminal():
    planner = _planner()
    _accept(planner, (1., -1.), sequence=1, now=0.)
    pending = planner.propose(monotonic_time=.01)
    with pytest.raises(RuntimeError, match="acknowledged"):
        planner.pause_reference(monotonic_time=.01, reason="pause")
    planner.commit(pending, accepted_monotonic_time=.01)
    with pytest.raises(ValueError, match="precede"):
        planner.pause_reference(monotonic_time=0., reason="pause")
    with pytest.raises(ValueError, match="non-empty"):
        planner.pause_reference(monotonic_time=.01, reason=" ")
    planner.quick_stop(sequence=2, monotonic_time=.01, reason="operator_stop")
    with pytest.raises(RuntimeError, match="acknowledged"):
        planner.pause_reference(monotonic_time=.02, reason="pause")


def test_target_preemption_inherits_velocity_instead_of_resetting_it() -> None:
    planner = _planner(max_velocity=1.0, max_acceleration=10.0)
    _accept(planner, (0.2, 0.0), sequence=1, now=0.0)
    first = planner.propose(monotonic_time=0.01)
    planner.commit(first, accepted_monotonic_time=0.01)
    assert first.qd_command_rad_s[0] == pytest.approx(0.1)

    _accept(planner, (0.3, 0.0), sequence=2, now=0.02)
    second = planner.propose(monotonic_time=0.02)
    assert second.qd_command_rad_s[0] == pytest.approx(0.2)
    assert second.q_command[0] > first.q_command[0]


def test_target_velocity_low_pass_uses_reference_dt_and_reports_both_values() -> None:
    planner = _planner(target_velocity_filter_tau_sec=0.03)
    _accept(planner, (0.0, 0.0), sequence=1, now=0.0)
    _accept(planner, (0.01, 0.0), sequence=2, now=0.01)

    proposal = planner.propose(monotonic_time=0.01)
    expected_alpha = 1.0 - np.exp(-0.01 / 0.03)
    assert proposal.qd_reference_rad_s[0] == pytest.approx(expected_alpha)
    assert proposal.metadata["trajectory_target_velocity_unfiltered_rad_s"][
        0
    ] == pytest.approx(1.0)
    assert proposal.metadata["trajectory_target_velocity_filtered_rad_s"][
        0
    ] == pytest.approx(expected_alpha)
    assert proposal.metadata[
        "trajectory_target_velocity_filter_alpha"
    ] == pytest.approx(expected_alpha)
    assert proposal.metadata[
        "trajectory_target_velocity_filter_dt_sec"
    ] == pytest.approx(0.01)
    assert proposal.metadata["trajectory_target_velocity_filter_applied"] is True


def test_zero_target_velocity_filter_tau_is_an_exact_bypass() -> None:
    planner = _planner(target_velocity_filter_tau_sec=0.0)
    _accept(planner, (0.0, 0.0), sequence=1, now=0.0)
    _accept(planner, (0.01, 0.0), sequence=2, now=0.01)

    proposal = planner.propose(monotonic_time=0.01)
    assert proposal.qd_reference_rad_s[0] == pytest.approx(1.0)
    assert proposal.metadata["trajectory_target_velocity_filter_applied"] is False


@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_target_velocity_filter_tau_rejects_invalid_values(value: float) -> None:
    limits = JointTrajectoryLimits(
        max_velocity_rad_s=np.ones(2),
        max_acceleration_rad_s2=np.ones(2),
    )
    with pytest.raises(ValueError, match="target_velocity_filter_tau_sec"):
        AccelerationLimitedJointTrajectoryPlanner(
            limits,
            target_velocity_filter_tau_sec=value,
        )


def test_reversal_brakes_before_reversing_and_stays_in_safe_endpoint_envelope() -> None:
    planner = _planner(max_velocity=1.0, max_acceleration=5.0)
    _accept(planner, (0.5, 0.0), sequence=1, now=0.0)
    now = 0.0
    for _ in range(10):
        now += 0.01
        proposal = planner.propose(monotonic_time=now)
        planner.commit(proposal, accepted_monotonic_time=now)
    preemption_q = proposal.q_command.copy()
    assert proposal.qd_command_rad_s[0] > 0.0

    _accept(planner, (-0.2, 0.0), sequence=2, now=0.11)
    braking = planner.propose(monotonic_time=0.11)
    assert braking.qd_command_rad_s[0] > 0.0
    assert -0.2 <= braking.q_command[0] <= 0.5
    assert braking.q_command[0] >= preemption_q[0]


def test_repeated_75hz_reversal_does_not_force_a_150hz_acceleration_spike() -> None:
    limits = JointTrajectoryLimits(
        max_velocity_rad_s=np.ones(2),
        max_acceleration_rad_s2=np.full(2, 10.0),
    )
    planner = AccelerationLimitedJointTrajectoryPlanner(
        limits,
        velocity_ramp_sec=0.1,
        reference_period_sec=1.0 / 75.0,
    )
    planner.reset(np.zeros(2), monotonic_time=0.0)

    # A 0.5 rad/s reference reverses abruptly at 0.8 s.  New safe targets
    # arrive every second transport tick, including while inherited positive
    # velocity is still braking against already reversing references.
    sequence = 0
    max_acceleration = 0.0
    safety_infeasible_count = 0
    for tick in range(1, 301):
        now = tick / 150.0
        if tick % 2 == 0:
            sequence += 1
            phase = now if now <= 0.8 else 1.6 - now
            planner.accept_target(
                (0.5 * phase, 0.0),
                sequence=sequence,
                published_monotonic_time=now,
                monotonic_time=now,
            )
        if sequence == 0:
            continue
        proposal = planner.propose(monotonic_time=now)
        planner.commit(proposal, accepted_monotonic_time=now)
        max_acceleration = max(
            max_acceleration,
            float(np.max(np.abs(proposal.qdd_command_rad_s2))),
        )
        safety_infeasible_count += int(
            proposal.metadata["trajectory_safety_velocity_infeasible"]
        )

    assert max_acceleration <= 10.0 + 1e-9
    assert safety_infeasible_count == 0
    assert proposal.qd_command_rad_s[0] < 0.0


def test_degraded_speed_transition_uses_acceleration_limit_then_converges_to_ten_percent() -> None:
    planner = _planner(max_velocity=1.0, max_acceleration=10.0)
    _accept(planner, (2.0, 0.0), sequence=1, now=0.0)
    now = 0.0
    for _ in range(10):
        now += 0.01
        proposal = planner.propose(monotonic_time=now)
        planner.commit(proposal, accepted_monotonic_time=now)
    assert proposal.qd_command_rad_s[0] == pytest.approx(1.0)

    _accept(
        planner,
        (2.1, 0.0),
        sequence=2,
        now=0.11,
        speed_scale=0.1,
        mode="degraded",
    )
    first_degraded = planner.propose(monotonic_time=0.11)
    assert first_degraded.qd_command_rad_s[0] == pytest.approx(0.9)
    assert first_degraded.qd_command_rad_s[0] > 0.1
    planner.commit(first_degraded, accepted_monotonic_time=0.11)

    for index in range(1, 10):
        tick_time = 0.11 + index * 0.01
        proposal = planner.propose(monotonic_time=tick_time)
        planner.commit(proposal, accepted_monotonic_time=tick_time)
    assert abs(proposal.qd_command_rad_s[0]) <= 0.1 + 1e-9

    _accept(planner, (2.2, 0.0), sequence=3, now=0.21)
    recovering = planner.propose(monotonic_time=0.21)
    assert recovering.qd_command_rad_s[0] == pytest.approx(0.2)
    assert recovering.qd_command_rad_s[0] < 1.0


def test_static_reference_keeps_position_feedback_until_target_is_reached() -> None:
    planner = _planner(max_velocity=1.0, max_acceleration=10.0)
    _accept(planner, (0.5, 0.0), sequence=1, now=0.0)
    now = 0.0
    sequence = 1
    saw_zero_reference_with_position_drive = False
    for tick in range(1, 301):
        now = tick / 150.0
        if tick % 2 == 0:
            sequence += 1
            _accept(planner, (0.5, 0.0), sequence=sequence, now=now)
        proposal = planner.propose(monotonic_time=now)
        if (
            abs(proposal.qd_reference_rad_s[0]) <= 1e-12
            and abs(proposal.q_target[0] - proposal.q_command[0]) > 1e-3
            and proposal.qd_desired_rad_s[0] > 0.0
        ):
            saw_zero_reference_with_position_drive = True
        planner.commit(proposal, accepted_monotonic_time=now)

    assert saw_zero_reference_with_position_drive is True
    assert proposal.q_command[0] == pytest.approx(0.5, abs=1e-4)
    assert abs(proposal.qd_command_rad_s[0]) <= 1e-3


def test_quick_stop_holds_last_committed_qpos_and_zeros_velocity_immediately() -> None:
    planner = _planner()
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)
    moving = planner.propose(monotonic_time=0.01)
    planner.commit(moving, accepted_monotonic_time=0.01)
    assert moving.qd_command_rad_s[0] > 0.0

    planner.quick_stop(sequence=2, monotonic_time=0.02, reason="collision_hold")
    stopped = planner.propose(monotonic_time=0.03)
    assert stopped.quick_stop is True
    assert stopped.reason == "collision_hold"
    np.testing.assert_allclose(stopped.q_command, moving.q_command)
    np.testing.assert_allclose(stopped.qd_command_rad_s, 0.0)
    np.testing.assert_allclose(stopped.qdd_command_rad_s2, 0.0)


def test_post_hold_reference_uses_upstream_publication_timeline() -> None:
    planner = _planner()
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)

    # The local simulation/planner clock may advance faster than the receiver
    # clock used to publish references in a non-realtime test or replay.
    planner.quick_stop(sequence=2, monotonic_time=1.0, reason="operator_hold")
    planner.accept_target(
        (0.5, 0.0),
        sequence=3,
        published_monotonic_time=0.5,
        monotonic_time=1.1,
    )

    snapshot = planner.snapshot()
    assert snapshot["target_sequence"] == 3
    np.testing.assert_allclose(snapshot["target_qd_raw_rad_s"], 0.0)


def test_rejected_proposal_does_not_advance_position_or_velocity_state() -> None:
    planner = _planner()
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)
    rejected = planner.propose(monotonic_time=0.01)
    assert planner.reject(rejected) is True
    snapshot = planner.snapshot()
    np.testing.assert_allclose(snapshot["q_command"], 0.0)
    np.testing.assert_allclose(snapshot["qd_command_rad_s"], 0.0)

    retried = planner.propose(monotonic_time=0.01)
    np.testing.assert_allclose(retried.q_command, rejected.q_command)
    np.testing.assert_allclose(
        retried.qd_command_rad_s,
        rejected.qd_command_rad_s,
    )


def test_bounded_transport_adjustment_commits_actual_qpos_and_velocity() -> None:
    planner = _planner(max_velocity=2.0, max_acceleration=100.0)
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)
    proposal = planner.propose(monotonic_time=0.01)
    accepted = 0.1 * proposal.q_command

    planner.commit(
        proposal,
        accepted_monotonic_time=0.01,
        accepted_q_command=accepted,
        allow_bounded_accepted_adjustment=True,
    )

    snapshot = planner.snapshot()
    np.testing.assert_allclose(snapshot["q_command"], accepted)
    np.testing.assert_allclose(
        snapshot["qd_command_rad_s"],
        accepted / 0.01,
    )


def test_transport_adjustment_rejects_values_beyond_transactional_segment() -> None:
    planner = _planner(max_velocity=2.0, max_acceleration=100.0)
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)
    proposal = planner.propose(monotonic_time=0.01)

    with pytest.raises(ValueError, match="must remain between"):
        planner.commit(
            proposal,
            accepted_monotonic_time=0.01,
            accepted_q_command=proposal.q_command * 1.1,
            allow_bounded_accepted_adjustment=True,
        )


def test_command_step_limit_is_a_hard_bound_even_after_a_scheduler_gap() -> None:
    planner = _planner(max_velocity=2.0, max_acceleration=100.0, max_step=0.02)
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)
    proposal = planner.propose(monotonic_time=0.1)
    assert proposal.command_step_limited is True
    assert proposal.q_command[0] == pytest.approx(0.02)


def test_anchor_resync_is_explicit_and_resets_untrusted_velocity() -> None:
    planner = _planner()
    _accept(planner, (1.0, 0.0), sequence=1, now=0.0)
    proposal = planner.propose(monotonic_time=0.01)
    planner.commit(proposal, accepted_monotonic_time=0.01)

    planner.accept_target(
        (0.6, 0.0),
        sequence=2,
        published_monotonic_time=0.02,
        monotonic_time=0.02,
        anchor_q_command=(0.5, 0.0),
    )
    snapshot = planner.snapshot()
    np.testing.assert_allclose(snapshot["q_command"], (0.5, 0.0))
    np.testing.assert_allclose(snapshot["qd_command_rad_s"], 0.0)
    assert snapshot["anchor_resync_count"] == 1

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from robot_motion.rebot_b601_rs.joint_motion import (
    B601JointMotionConfig,
    B601JointMotionTracker,
    B601WindowStopEvidence,
    B601WindowVelocityEstimator,
    slew_driver_damping,
    window_stationarity_gate_reason,
)

pytestmark = pytest.mark.unit


def _update(tracker, *, now=1.02, q=0.0, raw_v=0.0, control_v=0.0, seq=1, brake=False):
    return tracker.update(
        now=now,
        q=q,
        raw_v=raw_v,
        control_v=control_v,
        control_sequence=seq,
        authority=1.0,
        kp=6.0,
        kd=1.0,
        stop_error_rad=0.002,
        stop_velocity_rad_s=0.005,
        stop_dwell_sec=0.1,
        request_brake=brake,
    )


def _tracker(**kwargs):
    tracker = B601JointMotionTracker(B601JointMotionConfig(**kwargs))
    tracker.start(now=1.0, q=0.0, target=0.05)
    return tracker


def test_window_velocity_uses_actual_irregular_feedback_clock():
    estimator = B601WindowVelocityEstimator(0.08)
    for t in (10000.0, 10000.01, 10000.032, 10000.071):
        v = estimator.update(
            timestamp_sec=t, qpos_rad=0.2 + 0.02 * (t - 10000), fallback_rad_s=0.0
        )
    assert v == pytest.approx(0.02)
    assert estimator.span_sec == pytest.approx(0.071)
    with pytest.raises(ValueError, match="advance"):
        estimator.update(timestamp_sec=t, qpos_rad=0.2, fallback_rad_s=0.0)
    assert (
        estimator.update(timestamp_sec=t + 1, qpos_rad=5, fallback_rad_s=-0.1) == -0.1
    )


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize(
    "responsive,span,expected",
    [
        (False, 0.001534, "track"),
        (True, 0.001534, "brake"),
        (True, 0.000383496, "track"),
    ],
)
def test_resolved_tracking_brake_uses_reference_speed_plus_resolution_allowance(
    direction, responsive, span, expected
):
    config = B601JointMotionConfig(
        windowed_soft_velocity=True,
        position_resolution_rad=0.0004,
        responsive_tracking_brake=responsive,
        max_reference_velocity_rad_s=0.0075,
        max_control_gap_sec=0.04,
    )
    tracker = B601JointMotionTracker(config)
    tracker.start(now=1.0, q=0.1, target=0.1 + direction * 0.008)
    raw = (0.0, 0.0, direction * 0.038, 0.0, 0.0, 0.0)
    velocity = (0.0, 0.0, direction * 0.0144844, 0.0, 0.0, 0.0)
    evidence = B601WindowStopEvidence(
        1.01,
        (0.0, 0.0, 0.1, 0.0, 0.0, 0.0),
        raw,
        velocity,
        (0.0, 0.0, span, 0.0, 0.0, 0.0),
        0.08,
        0.05,
    )
    frame = tracker.update(
        now=1.01,
        q=0.1,
        raw_v=raw[2],
        control_v=velocity[2],
        control_sequence=1,
        authority=1,
        kp=12,
        kd=1,
        stop_error_rad=0.003,
        stop_velocity_rad_s=0.005,
        stop_dwell_sec=0.15,
        stop_evidence=evidence,
    )
    assert frame.phase == expected
    if responsive:
        assert frame.tracking_speed_brake_limit_rad_s == pytest.approx(0.0125)
    if expected == "brake":
        assert frame.reason == "resolved_tracking_speed_braking"
        assert frame.position_torque_nm == 0
        assert frame.requested_torque_nm * direction < 0


@pytest.mark.parametrize("invalid", ["missing", "cold", "stale", "raw_fast"])
def test_responsive_tracking_cannot_advance_with_unusable_motion_evidence(invalid):
    tracker = _tracker(
        windowed_soft_velocity=True,
        position_resolution_rad=0.0004,
        responsive_tracking_brake=True,
    )
    now = 1.01
    raw = 0.051 if invalid == "raw_fast" else 0.0
    evidence = (
        None
        if invalid == "missing"
        else B601WindowStopEvidence(
            1.0 if invalid == "stale" else now,
            (0.0,) * 6,
            (0.0, 0.0, raw, 0.0, 0.0, 0.0),
            (0.0,) * 6,
            (0.0,) * 6,
            0.02 if invalid == "cold" else 0.08,
            0.05,
        )
    )
    frame = tracker.update(
        now=now,
        q=0.0,
        raw_v=raw,
        control_v=0.0,
        control_sequence=1,
        authority=1,
        kp=12,
        kd=1,
        stop_error_rad=0.003,
        stop_velocity_rad_s=0.005,
        stop_dwell_sec=0.15,
        stop_evidence=evidence,
    )
    assert frame.motion_evidence_block_reason == {
        "missing": "feedback_evidence_missing",
        "cold": "velocity_window_not_ready",
        "stale": "feedback_evidence_timestamp_mismatch",
        "raw_fast": "raw_velocity_hard_guard",
    }[invalid]
    assert frame.phase == "authority_exhausted"
    assert frame.requested_torque_nm == 0


def _governor_tick(tracker, now, *, q=.006, velocity=0., raw=0., span=0.,
                   permit=True, zero=True, brake=False, support=0., evidence=True, stop_error=.003,
                   support_span=0., raw_limit=.05):
    positions = (0., 0., q, 0., 0., 0.)
    raw_values = (0., 0., raw, 0., 0., 0.)
    velocities = (support, 0., velocity, 0., 0., 0.)
    stop = B601WindowStopEvidence(now, positions, raw_values, velocities,
        (support_span, 0., span, 0., 0., 0.), .08, raw_limit) if evidence else None
    frame = tracker.update(now=now, q=q, raw_v=raw, control_v=velocity,
        control_sequence=round(now*1000), authority=1., kp=12., kd=1.,
        stop_error_rad=stop_error, stop_velocity_rad_s=.005, stop_dwell_sec=.15,
        request_brake=brake, support_max_velocity_rad_s=support,
        stop_evidence=stop, speed_governor_permitted=permit,
        zero_dispatch_confirmed=zero)
    tracker.record_dispatch(round(now*1000))
    return frame


def _governed_tracker(**kwargs):
    return _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004,
        responsive_tracking_brake=True, reference_pacing=True, speed_governor=True, **kwargs)


@pytest.mark.parametrize("axis", range(6))
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("hard_limit,raw_speed,window_speed,expected", [
    (.07, .07674896623177167, .014247198706110768, "authority_exhausted"),
    (.10, .07674896623177167, .014247198706110768, "track"),
    (.10, .10, .014247198706110768, "track"),
    (.10, .100001, .014247198706110768, "authority_exhausted"),
    (.10, .07674896623177167, .020, "brake"),
])
def test_explicit_acquisition_raw_speed_headroom_preserves_soft_braking_and_hard_stop(
    axis, direction, hard_limit, raw_speed, window_speed, expected
):
    # The recorded two-count receive velocity is a software observation, not a
    # physical speed guarantee. Only the explicit input limit changes here.
    tracker = _governed_tracker(max_reference_velocity_rad_s=.0075, velocity_headroom_ratio=.30)
    tracker.start(now=1., q=.1, target=.1+direction*.02)
    q = (0., 0., .1, 0., 0., 0.)
    raw = tuple(direction*raw_speed if i == axis else 0. for i in range(6))
    window = (0., 0., direction*window_speed, 0., 0., 0.)
    evidence = B601WindowStopEvidence(1.01, q, raw, window,
        (0., 0., .001534, 0., 0., 0.), .089, hard_limit)
    frame = tracker.update(now=1.01, q=q[2], raw_v=raw[2], control_v=window[2],
        control_sequence=1, authority=1., kp=12., kd=1., stop_error_rad=.005,
        stop_velocity_rad_s=.005, stop_dwell_sec=.15, stop_evidence=evidence,
        support_raw_max_velocity_rad_s=0. if axis == 2 else raw_speed)
    assert frame.phase == expected
    assert frame.tracking_speed_brake_limit_rad_s == pytest.approx(.01625)
    if expected == "authority_exhausted":
        assert frame.motion_evidence_block_reason == "raw_velocity_hard_guard"
        assert frame.requested_torque_nm == 0.
    elif expected == "brake":
        assert frame.reason == "resolved_tracking_speed_braking"


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_stabilizes_current_goal_and_waits_for_post_zero_stillness(direction):
    tracker = _governed_tracker(arrival_stabilization=True)
    tracker.start(now=1., q=direction*.0181, target=direction*.02)
    frame = _governor_tick(tracker, 1.01, q=direction*.0181, zero=False)
    assert frame.arrival_stabilization_active and not frame.arrival_unloading
    assert frame.q_reference_rad == direction*.02
    assert frame.position_torque_nm*direction > 0
    # Reproduce the recorded braking retreat. Quiet at the displaced point
    # must correct the accepted goal, not capture a new stop point there.
    now = 1.01
    for n in range(1, 40):
        now += .01
        frame = _governor_tick(tracker, now, q=direction*(.0181-n*.0001), zero=False)
        assert frame.phase == "brake" and not frame.arrival_unloading
    assert frame.stop_capture_block_reason == "arrival_position_outside_tolerance"
    for n in range(1, 49):
        now += .01
        _governor_tick(tracker, now, q=direction*(.0142+n*.0001), zero=False)
    for _ in range(30):
        now += .01
        frame = _governor_tick(tracker, now, q=direction*.019, zero=False)
        assert frame.phase == "brake"
    assert frame.arrival_unloading and frame.requested_torque_nm == 0.
    assert frame.hold_position_rad is None
    # A quiet dwell collected under torque is not a zero-assist dwell.
    for _ in range(11):
        now += .01
        frame = _governor_tick(tracker, now, q=direction*.019, zero=True)
        assert frame.phase == "brake"
    frame = _governor_tick(tracker, now+.06, q=direction*.019, zero=True)
    assert frame.phase == "soft_hold"
    assert frame.hold_position_rad == direction*.019


@pytest.mark.parametrize("cause", ["permit", "stop", "support", "raw", "missing"])
def test_arrival_stabilization_cannot_survive_withdrawal_or_fault(cause):
    tracker = _governed_tracker(arrival_stabilization=True)
    tracker.start(now=1., q=.049, target=.05)
    _governor_tick(tracker, 1.01, q=.049, zero=False)
    args = {"permit": {"permit": False}, "stop": {"brake": True},
        "support": {"support": .031},
        "raw": {"raw": .071}, "missing": {"evidence": False}}[cause]
    frame = _governor_tick(tracker, 1.02, q=.049, **args)
    assert not frame.arrival_stabilization_active
    assert frame.position_torque_nm == 0.
    for n in range(1, 22):
        frame = _governor_tick(tracker, 1.02+n*.01, q=.045)
        assert not frame.arrival_stabilization_active
        assert frame.position_torque_nm == 0.


def test_arrival_soft_speed_discards_goal_spring_until_zero_still_and_limited_recovery():
    tracker = _governed_tracker(arrival_stabilization=True)
    tracker.start(now=1., q=.049, target=.05)
    _governor_tick(tracker, 1.01, q=.049, zero=False)
    frame = _governor_tick(tracker, 1.02, q=.049, velocity=.031, span=.003, zero=False)
    assert frame.event_severity == "warning" and frame.speed_governor_active
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.02+n*.01, q=.045, zero=False)
        assert not frame.arrival_stabilization_active
        assert frame.position_torque_nm == 0.
        assert frame.phase == "brake"
    frame = _governor_tick(tracker, 1.26, q=.045)
    assert frame.phase == "track" and frame.speed_governor_resume_count == 1
    assert frame.q_reference_rad == .045 and frame.target_position_rad == .05
    assert frame.position_torque_nm == frame.requested_torque_nm == 0.


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_unloading_hysteresis_preserves_recorded_one_count_toggle(direction):
    target = direction*.09932825714349747
    near = direction*.09741077572107315
    far = direction*.09702728688716888
    tracker = _governed_tracker(arrival_stabilization=True, arrival_margin_rad=.0008)
    tracker.start(now=1., q=near, target=target)
    for n in range(1, 23):
        frame = _governor_tick(tracker, 1.+n*.01, q=near, zero=False)
    assert frame.arrival_unloading
    for n in range(1, 61):
        frame = _governor_tick(tracker, 1.22+n*.01,
            q=near if n%2 else far, span=abs(near-far), zero=False)
        assert frame.phase == "brake" and frame.arrival_unloading
        assert frame.requested_torque_nm == 0.
        assert frame.arrival_unload_exit_error_rad == .003
        assert frame.arrival_brake_error_rad == pytest.approx(.0022)
    # Capture must still wait for the full post-zero six-axis dwell, not just
    # a tolerance check while torque remains on the motor.
    for n in range(1, 19):
        frame = _governor_tick(tracker, 1.82+n*.01,
            q=near if n%2 else far, span=abs(near-far), zero=True)
        if n < 15:
            assert frame.phase == "brake"
    assert frame.phase == "soft_hold"
    assert abs(frame.target_error_rad) <= .003


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_hysteresis_does_not_allow_early_entry_or_outside_tolerance_exit(direction):
    tracker = _governed_tracker(arrival_stabilization=True, arrival_margin_rad=.0008)
    tracker.start(now=1., q=direction*.0181, target=direction*.02)
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.+n*.01, q=direction*.0177, zero=False)
        assert not frame.arrival_unloading
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.23+n*.01, q=direction*.0181, zero=False)
    assert frame.arrival_unloading
    # The exact final tolerance is retained, not enlarged by one resolution.
    frame = _governor_tick(tracker, 1.47, q=direction*.016999999, zero=False)
    assert not frame.arrival_unloading
    assert frame.stop_capture_block_reason == "arrival_position_outside_tolerance"
    assert frame.position_torque_nm*direction > 0.


def test_explicit_five_milliradian_arrival_keeps_margin_hysteresis_and_exit_bound():
    tracker = _governed_tracker(arrival_stabilization=True, arrival_margin_rad=.0008)
    tracker.start(now=1., q=.0159, target=.02)
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.+n*.01, q=.0159, zero=False, stop_error=.005)
    assert frame.arrival_unloading
    assert frame.arrival_brake_error_rad == pytest.approx(.0042)
    assert frame.arrival_unload_exit_error_rad == .005
    frame = _governor_tick(tracker, 1.24, q=.0155, zero=False, stop_error=.005)
    assert frame.arrival_unloading and frame.requested_torque_nm == 0.
    frame = _governor_tick(tracker, 1.25, q=.014999999, zero=False, stop_error=.005)
    assert not frame.arrival_unloading
    assert frame.stop_capture_block_reason == "arrival_position_outside_tolerance"


def test_arrival_recoil_after_unloading_restarts_goal_check_without_renewing_timeout():
    tracker = _governed_tracker(arrival_stabilization=True)
    tracker.start(now=1., q=.049, target=.05)
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.+n*.01, q=.049, zero=False)
    assert frame.arrival_unloading
    for n in range(1, 49):
        frame = _governor_tick(tracker, 1.23+n*.01, q=.049-n*.0001, zero=True)
    assert frame.arrival_stabilization_active and not frame.arrival_unloading
    assert frame.position_torque_nm > 0 and frame.phase == "brake"
    for n in range(1, 340):
        frame = _governor_tick(tracker, 1.71+n*.01, q=.0442)
    assert frame.phase == "authority_exhausted"
    assert frame.reason == "braking_timeout_assist_withdrawn"
    assert frame.requested_torque_nm == 0.


@pytest.mark.parametrize(
    "stop_error, expected_phase",
    [(.01, "not_captured"), (.0104, "soft_hold"), (.01352, "soft_hold")],
)
def test_arrival_capture_survives_one_encoder_count_toggling_at_tolerance_boundary(
    stop_error, expected_phase,
):
    target = .2679353111214396
    near_error = .009835546535944595
    far_error = .010219042820429458
    tracker = _governed_tracker(
        arrival_stabilization=True,
        arrival_margin_rad=.0008,
        braking_timeout_sec=.6,
    )
    tracker.start(now=1., q=target-.0091, target=target)
    frame = _governor_tick(
        tracker, 1.01, q=target-.0091, zero=True, stop_error=stop_error,
    )
    assert frame.phase == "brake" and frame.reason == "target_reached_braking"
    for n in range(1, 40):
        error = near_error if n % 2 else far_error
        frame = _governor_tick(
            tracker, 1.01+n*.02, q=target-error,
            span=.0003834962844848633, zero=True, stop_error=stop_error,
        )
        if frame.phase in {"soft_hold", "authority_exhausted"}:
            break
    if expected_phase == "not_captured":
        assert frame.phase in {"track", "brake"}
        assert frame.hold_position_rad is None
        assert frame.speed_governor_resume_count > 0
        return
    assert frame.phase == expected_phase
    if stop_error >= .0104:
        assert frame.reason == "measured_stop_point_captured"
        assert frame.hold_position_rad is not None
        assert abs(frame.target_error_rad) <= stop_error


@pytest.mark.parametrize("value", [0., 1, None, "true"])
def test_arrival_stabilization_requires_boolean(value):
    with pytest.raises(TypeError, match="arrival stabilization"):
        B601JointMotionConfig(arrival_stabilization=value)


def test_arrival_stabilization_requires_governor_and_preserves_reference_lag_gate():
    with pytest.raises(ValueError, match="governed tracking"):
        B601JointMotionConfig(arrival_stabilization=True)
    tracker = _governed_tracker(arrival_stabilization=True)
    tracker.start(now=1., q=.049, target=.05)
    _governor_tick(tracker, 1.01, q=.049)
    frame = _governor_tick(tracker, 1.02, q=.035)
    assert frame.reason == "reference_lag_braking"
    assert not frame.arrival_stabilization_active
    assert frame.position_torque_nm == 0.


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("governor", [False, True])
def test_recent_resolved_progress_advances_soft_brake_before_lagging_long_window(direction, governor):
    # Preserved pre-guard excerpt, translated/reflected around zero. The lost
    # raw trace cannot be reconstructed: the quiet warmup here is synthetic.
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004,
        responsive_tracking_brake=True, reference_pacing=True, speed_governor=governor)
    tracker.start(now=1., q=0., target=direction*.02)
    for n in range(1, 11):
        _governor_tick(tracker, 1.+n*.01, q=0.)
    counts = (0, 0, 0, 1, 1, 2, 2, 3, 4, 5)
    long_speeds = (.00565160, .00634972, .00634992, .00825525,
        .00889041, .01079548, .01079486, .01460378, .01904750, .02349159)
    first_brake = None
    previous = 0
    for n, (count, speed) in enumerate(zip(counts, long_speeds)):
        frame = _governor_tick(tracker, 1.11+n*.01,
            q=direction*count*.000383496,
            raw=direction*(count-previous)*.000383496/.01,
            velocity=direction*speed, span=max(.000384, count*.000383496))
        previous = count
        if frame.phase == "brake" and first_brake is None:
            first_brake = n
            assert frame.reason == "resolved_tracking_speed_braking"
            assert frame.requested_torque_nm*direction <= 0.
            if governor:
                assert speed < frame.tracking_speed_brake_limit_rad_s
                assert abs(frame.tracking_recent_velocity_rad_s) >= frame.tracking_speed_brake_limit_rad_s
    assert first_brake is not None
    assert first_brake < 9 if governor else first_brake == 9


@pytest.mark.parametrize("direction", [-1, 1])
def test_recent_window_does_not_treat_single_count_toggle_as_resolved_motion(direction):
    tracker = _governed_tracker()
    for n in range(1, 61):
        q = direction*(n % 2)*.000383496
        frame = _governor_tick(tracker, 1.+n*.01, q=q,
            raw=direction*(1 if n % 2 else -1)*.0383496)
        assert frame.phase == "track"
    assert frame.tracking_recent_position_span_rad < .0004


@pytest.mark.parametrize("direction", [-1, 1])
def test_explicit_transfer_handoff_keeps_same_tick_short_window_but_new_start_drops_it(direction):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05)
    for n in range(6):
        tracker.observe_feedback(now=1.+n*.01, q=direction*n*.0003, raw_v=direction*.03)
    timestamp, q = tracker._recent_velocity.latest_observation
    tracker.start(now=timestamp, q=q, target=direction*.35, preserve_velocity_history=True)
    frame = _governor_tick(tracker, timestamp, q=q,
        raw=direction*.03, velocity=direction*.005, span=.0015)
    assert frame.phase == "track"
    assert frame.tracking_recent_velocity_rad_s == pytest.approx(direction*.03)
    assert frame.tracking_recent_position_span_rad > .0004
    assert frame.continuous_speed_scale < 1.
    # Same physical feedback sampled again must neither advance nor duplicate
    # the observer; conflicting feedback at the same clock is rejected.
    before = tracker._recent_velocity.span_sec
    tracker.observe_feedback(now=timestamp, q=q, raw_v=direction*.03)
    assert tracker._recent_velocity.span_sec == before
    with pytest.raises(ValueError, match="timestamps must advance"):
        tracker.observe_feedback(now=1.05, q=direction*.002, raw_v=direction*.03)
    tracker.start(now=2., q=0., target=direction*.35)
    assert not tracker._recent_velocity.ready and tracker._recent_speed == 0.


def test_transfer_cannot_import_stale_or_other_position_velocity_history():
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05)
    tracker.observe_feedback(now=1., q=0., raw_v=0.)
    for now, q in [(1.01, 0.), (1., .001)]:
        with pytest.raises(ValueError, match="same-tick"):
            tracker.start(now=now, q=q, target=.3, preserve_velocity_history=True)
    ordinary = _governed_tracker()
    ordinary.observe_feedback(now=1., q=0., raw_v=0.)
    with pytest.raises(ValueError, match="explicit continuous"):
        ordinary.start(now=1., q=0., target=.3, preserve_velocity_history=True)


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_transfer_regulates_resolved_speed_without_spending_resume_budget(direction):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05)
    tracker.start(now=1., q=0., target=direction*.35)
    for n in range(1, 25):
        frame = _governor_tick(tracker, 1.+n*.01, q=direction*n*.00025,
            raw=direction*.025, velocity=direction*.025, span=.002)
        assert frame.phase == "track"
        assert frame.speed_governor_resume_count == 0
        assert 0 < frame.continuous_speed_scale < 1
        if direction * (frame.q_reference_rad-direction*n*.00025) < 0:
            # The reference must catch up without jumping when measured motion
            # is already ahead. Output torque is still independently regulated.
            assert frame.reference_speed_scale == pytest.approx(1.)
        else:
            assert frame.reference_speed_scale <= frame.continuous_speed_scale
        assert frame.target_position_rad == direction*.35
    assert frame.tracking_speed_brake_limit_rad_s == .05
    assert frame.continuous_speed_start_rad_s < .025


@pytest.mark.parametrize("cause,reason", [
    ("ceiling", "soft_velocity_braking"),
    ("support", "support_velocity_braking"),
    ("operator", "operator_stop_and_hold"),
    ("raw", "tracking_motion_evidence_unavailable_or_unsafe"),
])
@pytest.mark.parametrize("ceiling", [.03, .05])
def test_continuous_transfer_preserves_stop_and_safety_gates(cause, reason, ceiling):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=ceiling)
    kwargs = {"ceiling": {"velocity": .052}, "support": {"support": .031},
              "operator": {"brake": True}, "raw": {"raw": .051}}[cause]
    frame = _governor_tick(tracker, 1.01, q=.001, span=.002, **kwargs)
    assert frame.phase in {"brake", "authority_exhausted"}
    assert frame.reason == reason
    assert frame.requested_torque_nm <= 0


def test_continuous_transfer_does_not_charge_single_count_jitter_as_speed():
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05)
    for n in range(1, 25):
        frame = _governor_tick(tracker, 1.+n*.01, q=n%2*.000383496,
            raw=.038, velocity=.004, span=.000383496)
        assert frame.continuous_speed_scale == 1
        assert frame.phase == "track"


@pytest.mark.parametrize("direction", [-1, 1])
def test_continuous_transfer_reverse_motion_keeps_corrective_budget_and_absolute_stop(direction):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05)
    tracker.start(now=1., q=0., target=direction*.35)
    frame = _governor_tick(tracker, 1.01, q=-direction*.001,
        velocity=-direction*.025, raw=-direction*.038, span=.002)
    assert frame.phase == "track"
    assert frame.continuous_speed_scale == 1.
    assert frame.continuous_speed_observed_rad_s == 0.
    assert direction*frame.requested_torque_nm > 0.
    frame = _governor_tick(tracker, 1.02, q=-direction*.0013,
        velocity=-direction*.052, raw=-direction*.038, span=.003)
    assert frame.phase == "brake" and frame.reason == "soft_velocity_braking"


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("terminal", [None, "operator", "permit", "hard"])
def test_transfer_predictive_headroom_brake_resumes_only_after_zero_still_and_fresh_permit(direction, terminal):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05, max_control_gap_sec=.04)
    tracker.start(now=1., q=0., target=direction*.35)
    for n in range(1, 501):
        frame = _governor_tick(tracker, 1.+n*.01, q=0.)
    assert frame.phase == "track"
    frame = _governor_tick(tracker, 6.01, q=-direction*.0002,
        velocity=-direction*.025, raw=-direction*.02, span=.002, zero=False)
    assert frame.reason == "reference_pacing_headroom_braking"
    assert frame.speed_governor_active
    assert abs(frame.brake_trigger_lag_rad) < .013
    if terminal is not None:
        kwargs = {"operator": {"brake": True}, "permit": {"permit": False},
                  "hard": {"raw": .051}}[terminal]
        _governor_tick(tracker, 6.02, q=-direction*.0002, zero=False, **kwargs)
    for n in range(3, 25):
        frame = _governor_tick(tracker, 6.+n*.01, q=-direction*.0002, zero=False)
        assert frame.phase != "track"
        assert frame.speed_governor_resume_count == 0
    frame = _governor_tick(tracker, 6.25, q=-direction*.0002, zero=True)
    if terminal is None:
        assert frame.reason == "speed_governor_resumed_original_target"
        assert frame.speed_governor_resume_count == 1
        assert frame.q_reference_rad == -direction*.0002
        assert frame.target_position_rad == direction*.35
    else:
        assert frame.phase != "track" and frame.speed_governor_resume_count == 0


@pytest.mark.parametrize("q", [-.014, .014])
def test_transfer_actual_reference_lag_remains_terminal_not_predictive_recovery(q):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05)
    frame = _governor_tick(tracker, 1.01, q=q)
    assert frame.reason == "reference_lag_braking" and not frame.speed_governor_active
    for n in range(2, 35):
        frame = _governor_tick(tracker, 1.+n*.01, q=q)
        assert frame.phase != "track" and frame.speed_governor_resume_count == 0


def test_continuous_transfer_arrival_still_unloads_and_requires_zero_receipt():
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.05,
        arrival_stabilization=True, arrival_margin_rad=.0008)
    tracker.start(now=1., q=.346, target=.35)
    for n in range(1, 25):
        frame = _governor_tick(tracker, 1.+n*.01, q=.346, zero=False, stop_error=.005)
    assert frame.phase == "brake" and frame.arrival_unloading
    assert frame.continuous_speed_scale is None
    assert frame.hold_position_rad is None
    for n in range(25, 50):
        frame = _governor_tick(tracker, 1.+n*.01, q=.346, zero=True, stop_error=.005)
    assert frame.phase == "soft_hold"
    assert frame.requested_torque_nm == 0.
    assert frame.speed_governor_resume_count == 0


@pytest.mark.parametrize("value", [0., -.1, .02, True, float("nan"), float("inf")])
def test_continuous_transfer_ceiling_requires_valid_explicit_configuration(value):
    with pytest.raises(ValueError):
        _governed_tracker(continuous_speed_limit_rad_s=value)
    with pytest.raises(ValueError):
        B601JointMotionConfig(continuous_speed_limit_rad_s=.05)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("reference_speed", [.0075, .015])
def test_continuous_acquisition_regulates_between_early_gate_and_unchanged_soft_stop(
    direction, reference_speed,
):
    tracker = _governed_tracker(
        max_reference_velocity_rad_s=reference_speed,
        velocity_headroom_ratio=.30, continuous_speed_limit_rad_s=.03,
    )
    tracker.start(now=1., q=.25, target=.25+direction*.02)
    speed = (tracker.config.resolved_soft_velocity_rad_s + .03) / 2
    for n in range(1, 9):
        frame = _governor_tick(tracker, 1.+n*.01,
            q=.25+direction*speed*n*.01,
            velocity=direction*speed, raw=direction*speed, span=.0012)
        assert frame.phase == "track"
        assert 0. < frame.continuous_speed_scale < 1.
        assert frame.speed_governor_resume_count == 0
        assert frame.tracking_speed_brake_limit_rad_s == .03
    frame = _governor_tick(tracker, 1.09,
        q=.25+direction*(speed*.08+.00031),
        velocity=direction*.031, raw=direction*.031, span=.0012)
    assert frame.phase == "brake"
    assert frame.reason == "soft_velocity_braking"


def test_continuous_acquisition_rejects_a_collapsed_regulation_band():
    # If resolution/headroom consume the full stop ceiling, no continuous
    # band exists. Do not silently widen the stop to admit the configuration.
    with pytest.raises(ValueError, match="above the resolved gate"):
        _governed_tracker(continuous_speed_limit_rad_s=.03,
            velocity_headroom_ratio=1.)


def test_recent_window_requires_fresh_history_and_resets_on_new_measured_start():
    tracker = _governed_tracker()
    _governor_tick(tracker, 1.01, q=0.)
    frame = _governor_tick(tracker, 1.06, q=.001, raw=.02)
    assert frame.tracking_recent_velocity_rad_s is None
    assert frame.tracking_recent_position_span_rad == 0.
    assert frame.phase == "track"
    # Same-tick projection must not append a duplicate estimator sample.
    repeated = _governor_tick(tracker, 1.06, q=.001, raw=.02)
    assert repeated.tracking_recent_velocity_rad_s is None
    tracker.start(now=2., q=.001, target=.02)
    frame = _governor_tick(tracker, 2.01, q=.001)
    assert frame.tracking_recent_velocity_rad_s is None
    assert frame.tracking_recent_position_span_rad == 0.


def test_speed_governor_requires_still_window_dwell_and_complete_zero_then_keeps_original_goal():
    tracker = _governed_tracker()
    frame = _governor_tick(tracker, 1.01, velocity=.023, span=.002)
    assert frame.phase == "brake" and frame.speed_governor_active
    for n in range(1, 22):
        frame = _governor_tick(tracker, 1.01+n*.01, zero=False)
        assert frame.phase == "brake" and frame.speed_governor_resume_count == 0
    assert frame.reason == "speed_governor_waiting_for_zero_dispatch"
    assert frame.requested_torque_nm == 0.
    assert frame.speed_governor_resume_block_reason == "native_zero_dispatch_not_confirmed"
    frame = _governor_tick(tracker, 1.23)
    assert frame.phase == "track" and frame.reason == "speed_governor_resumed_original_target"
    assert frame.speed_governor_resume_count == 1
    assert frame.target_position_rad == .05
    assert frame.q_reference_rad == .006 and frame.qvel_reference_rad_s == 0.
    frame = _governor_tick(tracker, 1.24)
    assert frame.phase == "track"
    assert 0 < frame.q_reference_rad-.006 <= .03*.01**2+1e-12


@pytest.mark.parametrize("cause", ["permit", "stop", "hard_speed", "missing", "support", "support_and_target"])
def test_speed_governor_does_not_resume_after_revocation_or_fault(cause):
    tracker = _governed_tracker()
    _governor_tick(tracker, 1.01, velocity=.023, span=.002)
    options = {
        "permit": {"permit":False}, "stop": {"brake":True},
        "hard_speed": {"raw":.051}, "missing": {"evidence":False},
        "support": {"support":.031}, "support_and_target": {"support":.031,"velocity":.031,"span":.003},
    }[cause]
    _governor_tick(tracker, 1.02, **options)
    for n in range(1, 35):
        frame = _governor_tick(tracker, 1.02+n*.01)
        assert frame.phase != "track" and frame.speed_governor_resume_count == 0


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("early_brake", [False, True])
def test_target_soft_overspeed_is_a_warning_with_same_bounded_zero_and_still_recovery(direction, early_brake):
    tracker = _governed_tracker(velocity_headroom_ratio=.30)
    tracker.start(now=1., q=0., target=direction*.02)
    _governor_tick(tracker, 1.01, q=direction*.001,
        velocity=direction*(.028 if early_brake else .033), span=.002)
    frame = _governor_tick(tracker, 1.02, q=direction*.0014,
        velocity=direction*.033, span=.003)
    assert frame.reason == "soft_velocity_braking"
    assert frame.speed_governor_active
    assert frame.event_severity == "warning"
    assert frame.event_action == "brake_zero_still_then_resume"
    assert frame.position_torque_nm == 0.
    assert frame.velocity_torque_nm*direction < 0.
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.02+n*.01, q=direction*.004, zero=False)
    assert frame.phase == "brake" and frame.speed_governor_resume_count == 0
    assert frame.speed_governor_resume_block_reason == "native_zero_dispatch_not_confirmed"
    frame = _governor_tick(tracker, 1.26, q=direction*.004)
    assert frame.phase == "track" and frame.speed_governor_resume_count == 1
    assert frame.reason == "speed_governor_resumed_original_target"
    assert frame.target_position_rad == direction*.02
    assert frame.q_reference_rad == direction*.004 and frame.qvel_reference_rad_s == 0.
    assert frame.requested_torque_nm == 0.


def test_soft_overspeed_remains_terminal_without_explicit_governor():
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004,
        responsive_tracking_brake=True, reference_pacing=True)
    frame = _governor_tick(tracker, 1.01, velocity=.033, span=.003)
    assert not frame.speed_governor_active
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.01+n*.01)
    assert frame.phase == "soft_hold" and frame.speed_governor_resume_count == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_positioning_support_soft_brake_recovers_only_with_explicit_mode_zero_and_still(enabled):
    tracker = _governed_tracker(support_soft_speed_recovery=enabled)
    tracker.start(now=1., q=0., target=math.radians(20))
    _governor_tick(tracker, 1.01, velocity=.023, span=.002)
    _governor_tick(tracker, 1.02, velocity=.0362, span=.003)
    frame = _governor_tick(tracker, 1.03, q=.00767, support=.0305)
    assert frame.reason == "support_velocity_braking"
    assert frame.speed_governor_active is enabled
    for n in range(1, 25):
        frame = _governor_tick(tracker, 1.03+n*.01, q=.00767, zero=False)
        assert frame.phase != "track"
    if enabled:
        assert frame.reason == "speed_governor_waiting_for_zero_dispatch"
        assert frame.speed_governor_resume_count == 0
        frame = _governor_tick(tracker, 1.28, q=.00767)
        assert frame.phase == "track" and frame.speed_governor_resume_count == 1
        assert frame.target_position_rad == math.radians(20)
        assert frame.q_reference_rad == .00767 and frame.qvel_reference_rad_s == 0.
    else:
        assert frame.phase == "soft_hold" and frame.speed_governor_resume_count == 0


@pytest.mark.parametrize("cause", ["permit", "stop", "hard_speed", "missing", "persistent_support"])
def test_opted_in_support_recovery_never_resumes_after_fault_or_ongoing_motion(cause):
    tracker = _governed_tracker(support_soft_speed_recovery=True)
    _governor_tick(tracker, 1.01, support=.0305)
    options = {"permit": {"permit": False}, "stop": {"brake": True},
               "hard_speed": {"raw": .051}, "missing": {"evidence": False},
               "persistent_support": {"support": .0305, "support_span": .0025}}[cause]
    _governor_tick(tracker, 1.02, **options)
    for n in range(1, 430):
        frame = _governor_tick(tracker, 1.02+n*.01,
                               **(options if cause == "persistent_support" else {}))
        assert frame.phase != "track" and frame.speed_governor_resume_count == 0
    if cause == "persistent_support":
        assert frame.phase == "authority_exhausted"


def test_support_recovery_continues_after_repeated_soft_stops():
    tracker = _governed_tracker(support_soft_speed_recovery=True)
    now = 1.
    for cycle in range(3):
        now += .01
        _governor_tick(tracker, now, support=.0305)
        for _ in range(26):
            now += .01
            frame = _governor_tick(tracker, now)
        assert frame.speed_governor_resume_count == cycle+1
        assert frame.phase == "track"


@pytest.mark.parametrize("value", [None, 0, 1, -1.0, float("nan"), float("inf"), "yes"])
def test_support_recovery_requires_explicit_boolean_and_governor(value):
    with pytest.raises(TypeError, match="must be boolean"):
        _governed_tracker(support_soft_speed_recovery=value)
    with pytest.raises(ValueError, match="requires governed tracking"):
        _tracker(support_soft_speed_recovery=True)


@pytest.mark.parametrize("speed", [.023, .033])
def test_speed_governor_resumes_after_repeated_soft_stops_without_a_retry_cap(speed):
    tracker = _governed_tracker()
    now = 1.
    for cycle in range(8):
        now += .01
        frame = _governor_tick(tracker, now, velocity=speed, span=.002)
        for n in range(1, 24):
            frame = _governor_tick(tracker, now+n*.01)
            if frame.phase in {"track", "authority_exhausted"}:
                break
        now += n*.01
        assert frame.speed_governor_resume_count == cycle+1
        assert frame.phase == "track"


def test_speed_governor_reached_goal_uses_original_final_stop_path():
    tracker = _governed_tracker()
    _governor_tick(tracker, 1.01, velocity=.023, span=.002)
    for n in range(1, 24):
        frame = _governor_tick(tracker, 1.01+n*.01, q=.048)
    assert frame.phase == "soft_hold" and frame.speed_governor_resume_count == 0


@pytest.mark.parametrize("changes", [
    {"speed_governor":1}, {"speed_governor":True},
])
def test_speed_governor_configuration_rejects_implicit_or_invalid_authority(changes):
    with pytest.raises((ValueError, TypeError)):
        B601JointMotionConfig(**changes)


def test_responsive_brake_is_explicit_and_requires_resolved_window_configuration():
    assert B601JointMotionConfig().responsive_tracking_brake is False
    with pytest.raises(ValueError, match="resolved"):
        B601JointMotionConfig(responsive_tracking_brake=True)
    for value in (0, 1, None, "yes"):
        with pytest.raises(TypeError, match="boolean"):
            B601JointMotionConfig(responsive_tracking_brake=value)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("margin,expected", [(0.0, "brake"), (0.0008, "track")])
def test_arrival_margin_reserves_unloading_room_on_recorded_boundary(direction, margin, expected):
    config = B601JointMotionConfig(
        windowed_soft_velocity=True, position_resolution_rad=.0004,
        responsive_tracking_brake=True, max_reference_velocity_rad_s=.0075,
        arrival_margin_rad=margin,
    )
    tracker = B601JointMotionTracker(config)
    anchor, error = .1127505898475647, .002631066918373115
    target = anchor + direction * .008
    tracker.start(now=1., q=anchor, target=target)

    def tick(now, q, sequence, raw=0., window=0.):
        evidence = B601WindowStopEvidence(
            now, (0., 0., q, 0., 0., 0.), (0., 0., raw, 0., 0., 0.),
            (0., 0., window, 0., 0., 0.), (0., 0., .00038349628448, 0., 0., 0.),
            .08, .05,
        )
        return tracker.update(
            now=now, q=q, raw_v=raw, control_v=window, control_sequence=sequence,
            authority=1., kp=12., kd=1., stop_error_rad=.003,
            stop_velocity_rad_s=.005, stop_dwell_sec=.15, stop_evidence=evidence,
        )

    frame = tick(1.01, target - direction * error, 1, direction * .03093496, direction * .00248158)
    assert frame.phase == expected
    assert frame.arrival_brake_error_rad == pytest.approx(.003 - margin)
    if margin:
        tracker.record_dispatch(1)
        frame = tick(1.02, target - direction * .002, 2)
    assert frame.reason == "target_reached_braking"
    if margin:
        assert abs(frame.target_error_rad) + .00038349628448 < .003


@pytest.mark.parametrize("margin", [-.001, float("nan"), float("inf"), True])
def test_arrival_margin_rejects_invalid_config(margin):
    with pytest.raises(ValueError, match="arrival margin"):
        B601JointMotionConfig(arrival_margin_rad=margin)


def test_arrival_margin_requires_resolution_and_positive_inner_tolerance():
    with pytest.raises(ValueError, match="resolved"):
        B601JointMotionConfig(arrival_margin_rad=.0008)
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004, arrival_margin_rad=.002)
    with pytest.raises(ValueError, match="below final"):
        _update(tracker)


@pytest.mark.parametrize("resolution,expected", [(None, "authority_exhausted"), (.0004, "soft_hold")])
def test_same_six_axis_one_count_trace_stop_dwell_legacy_and_acquisition(resolution, expected):
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=resolution, braking_timeout_sec=.4)
    estimators = tuple(B601WindowVelocityEstimator(.08) for _ in range(6))
    previous = (0.0,) * 6
    captured = None
    for i in range(60):
        now = 1.01 + i * 0.01
        q = tuple(0.000383496 * ((i + j) // 2 % 2) for j in range(6))
        raw = tuple((a - b) / 0.01 for a, b in zip(q, previous))
        velocity = tuple(
            e.update(timestamp_sec=now, qpos_rad=p, fallback_rad_s=v)
            for e, p, v in zip(estimators, q, raw)
        )
        evidence = B601WindowStopEvidence(
            now,
            q,
            raw,
            velocity,
            tuple(e.position_span_rad for e in estimators),
            min(e.span_sec for e in estimators),
            0.05,
        )
        frame = tracker.update(
            now=now,
            q=q[2],
            raw_v=raw[2],
            control_v=velocity[2],
            control_sequence=i,
            authority=1,
            kp=6,
            kd=1,
            stop_error_rad=0.002,
            stop_velocity_rad_s=0.005,
            stop_dwell_sec=0.15,
            request_brake=i == 0,
            support_max_velocity_rad_s=max(
                abs(v) for j, v in enumerate(velocity) if j != 2
            ),
            support_raw_max_velocity_rad_s=max(
                abs(v) for j, v in enumerate(raw) if j != 2
            ),
            stop_evidence=evidence,
        )
        previous = q
        if frame.phase in {"soft_hold", "authority_exhausted"}:
            captured = frame
            break
    assert captured is not None
    assert captured.phase == expected
    if resolution:
        assert captured.stop_capture_dwell_elapsed_sec >= 0.15
        assert captured.stop_position_resolution_rad == 0.0004
        assert captured.hold_position_rad == q[2]


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("missing", "same_tick_stop_evidence_unavailable"),
        ("stale", "same_tick_stop_evidence_unavailable"),
        ("cold", "velocity_window_not_ready"),
        ("raw_fast", "raw_velocity_hard_guard"),
        ("support_progress", "six_axis_window_velocity"),
        ("support_range", "six_axis_window_displacement"),
    ],
)
def test_resolution_stop_evidence_never_captures_invalid_or_resolved_support_motion(
    fault, reason
):
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=0.0004)
    for i in range(30):
        now = 1.01 + i * 0.01
        raw = (0.051, 0, 0, 0, 0, 0) if fault == "raw_fast" else (0.0,) * 6
        velocity = (0, 0, 0, 0.006, 0, 0) if fault == "support_progress" else (0.0,) * 6
        spans = (
            (0, 0, 0, 0.0008, 0, 0)
            if fault in {"support_progress", "support_range"}
            else (0.0,) * 6
        )
        evidence = (
            None
            if fault == "missing"
            else B601WindowStopEvidence(
                now - 0.01 if fault == "stale" else now,
                (0.0,) * 6,
                raw,
                velocity,
                spans,
                0.02 if fault == "cold" else 0.08,
                0.05,
            )
        )
        frame = tracker.update(
            now=now,
            q=0,
            raw_v=0,
            control_v=0,
            control_sequence=i,
            authority=1,
            kp=0,
            kd=0,
            stop_error_rad=0.002,
            stop_velocity_rad_s=0.005,
            stop_dwell_sec=0.15,
            request_brake=i == 0,
            stop_evidence=evidence,
        )
        assert frame.phase == "brake"
        assert frame.stop_capture_block_reason == reason


def test_stop_dwell_bounds_accumulated_six_axis_range_not_only_each_window():
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=0.0004)
    for i in range(22):
        now = 1.01 + i * 0.01
        # A support axis progresses slowly through adjacent bins: each short
        # window is unresolved, but the whole dwell must not hide a staircase.
        q = (0, 0, 0, (i // 4) * 0.000383496, 0, 0)
        evidence = B601WindowStopEvidence(
            now, q, (0.0,) * 6, (0.0,) * 6, (0.000383496,) * 6, 0.08, 0.05
        )
        frame = tracker.update(
            now=now,
            q=0,
            raw_v=0,
            control_v=0,
            control_sequence=i,
            authority=1,
            kp=0,
            kd=0,
            stop_error_rad=0.002,
            stop_velocity_rad_s=0.005,
            stop_dwell_sec=0.15,
            request_brake=i == 0,
            stop_evidence=evidence,
        )
        assert frame.phase == "brake"
    assert frame.stop_capture_dwell_elapsed_sec < 0.15


def test_captured_stop_cannot_restore_gains_during_unresolved_window_staircase():
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004)
    frames = []
    for i in range(36):
        now = 1.01 + i * .01
        q = (0., 0., 0., max(0, (i - 20) // 4) * .000383496, 0., 0.)
        evidence = B601WindowStopEvidence(now, q, (0.,) * 6, (0.,) * 6, (.000383496,) * 6, .08, .05)
        frames.append(tracker.update(
            now=now, q=0., raw_v=0., control_v=0., control_sequence=i, authority=1., kp=0., kd=0.,
            stop_error_rad=.002, stop_velocity_rad_s=.005, stop_dwell_sec=.15,
            request_brake=i == 0, stop_evidence=evidence,
        ))
    assert frames[16].phase == "soft_hold"
    assert frames[28].phase == "brake"
    assert frames[-1].phase == "brake"


@pytest.mark.parametrize("updates", [
    {"qpos_rad": (0, 0, .001, 0, 0, 0)},
    {"raw_qvel_rad_s": (0, 0, .051, 0, 0, 0)},
    {"window_qvel_rad_s": (0, 0, .01, 0, 0, 0)},
])
def test_stop_evidence_cannot_capture_a_different_selected_feedback(updates):
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004)
    evidence = B601WindowStopEvidence(1.01, (0.,) * 6, (0.,) * 6, (0.,) * 6, (0.,) * 6, .08, .05)
    evidence = replace(evidence, **updates)
    with pytest.raises(ValueError, match="match selected feedback"):
        tracker.update(now=1.01, q=0., raw_v=0., control_v=0., control_sequence=1,
            authority=1., kp=0., kd=0., stop_error_rad=.002, stop_velocity_rad_s=.005,
            stop_dwell_sec=.15, request_brake=True, stop_evidence=evidence)


@pytest.mark.parametrize(
    "updates",
    [
        {"position_span_rad": (-0.01,) * 6},
        {"qpos_rad": (0.0,) * 5},
        {"window_qvel_rad_s": (float("nan"),) * 6},
        {"raw_velocity_limit_rad_s": 0.0},
        {"window_span_sec": float("inf")},
        {"timestamp_sec": True},
    ],
)
def test_stop_evidence_rejects_invalid_boundary_data(updates):
    values = {
        "timestamp_sec": 1.0,
        "qpos_rad": (0.0,) * 6,
        "raw_qvel_rad_s": (0.0,) * 6,
        "window_qvel_rad_s": (0.0,) * 6,
        "position_span_rad": (0.0,) * 6,
        "window_span_sec": 0.08,
        "raw_velocity_limit_rad_s": 0.05,
    }
    values.update(updates)
    with pytest.raises(ValueError):
        B601WindowStopEvidence(**values)


def test_window_velocity_at_high_rate_does_not_treat_one_count_as_onset():
    estimator = B601WindowVelocityEstimator(0.05)
    for i in range(8):
        v = estimator.update(
            timestamp_sec=1 + i * 0.01,
            qpos_rad=(i % 2) * 0.000384,
            fallback_rad_s=0.0384,
        )
    assert abs(v) < 0.005


@pytest.mark.parametrize("position", [.004109217373936344, -.004109217373936344, .29185, -3.12, 0.])
@pytest.mark.parametrize("origin", [1., 10000., 2000000.])
def test_exactly_constant_position_has_exact_zero_window_slope(position, origin):
    estimator = B601WindowVelocityEstimator(.08)
    for i in range(100):
        velocity = estimator.update(timestamp_sec=origin + i*.01, qpos_rad=position, fallback_rad_s=.1)
        if i > 0:
            assert velocity == 0.0


@pytest.mark.parametrize("direction", [-1, 1])
def test_origin_centered_estimator_does_not_hide_small_real_movement(direction):
    estimator = B601WindowVelocityEstimator(.08)
    for i in range(12):
        v = estimator.update(timestamp_sec=1. + i*.01, qpos_rad=.29185 + direction*i*1e-12, fallback_rad_s=0.)
    assert v * direction > 0
    assert v == pytest.approx(direction * 1e-10, rel=1e-4, abs=1e-14)


def test_velocity_window_readiness_range_and_reset_are_derived_from_samples():
    estimator = B601WindowVelocityEstimator(.08)
    assert not estimator.ready
    assert estimator.position_span_rad == 0
    for i in range(9):
        estimator.update(timestamp_sec=1+i*.01,qpos_rad=(i%2)*.000384,fallback_rad_s=0.)
    assert estimator.ready
    assert estimator.position_span_rad == pytest.approx(.000384)
    estimator.reset()
    assert not estimator.ready
    assert estimator.span_sec == 0
    assert estimator.position_span_rad == 0


@pytest.mark.parametrize("condition,reason", [
    ("quiet", "ready"), ("cold", "velocity_window_not_ready"),
    ("raw_fast", "raw_velocity_hard_guard"), ("window_fast", "not_stationary"),
    ("large_range", "velocity_window_displacement_exceeded"),
])
def test_shared_window_preparation_gate_preserves_speed_and_displacement_conditions(condition, reason):
    estimators = tuple(B601WindowVelocityEstimator(.08) for _ in range(6))
    for i in range(2 if condition == "cold" else 10):
        q = i*.0003 if condition == "window_fast" else (i%2)*.0022 if condition == "large_range" and i < 9 else 0.
        velocity = tuple(e.update(timestamp_sec=1+i*.01, qpos_rad=q, fallback_rad_s=0.) for e in estimators)
    raw = (.06,)*6 if condition == "raw_fast" else (0.,)*6
    assert window_stationarity_gate_reason(estimators, raw, velocity, soft_limit_rad_s=.02, hard_limit_rad_s=.05) == reason


def test_shared_window_gate_selected_joint_keeps_full_window_readiness():
    estimators = tuple(B601WindowVelocityEstimator(.08) for _ in range(6))
    for i in range(9):
        velocity = tuple(e.update(timestamp_sec=1+i*.01, qpos_rad=0., fallback_rad_s=0.) for e in estimators)
    raw = (.06, 0., 0., 0., 0., 0.)
    assert window_stationarity_gate_reason(estimators, raw, velocity, soft_limit_rad_s=.02, hard_limit_rad_s=.05) == "raw_velocity_hard_guard"
    assert window_stationarity_gate_reason(estimators, raw, velocity, soft_limit_rad_s=.02, hard_limit_rad_s=.05, joint_index=2) == "ready"
    estimators[5].reset()
    assert window_stationarity_gate_reason(estimators, raw, velocity, soft_limit_rad_s=.02, hard_limit_rad_s=.05, joint_index=2) == "velocity_window_not_ready"


@pytest.mark.parametrize("invalid", ["short", "nan", "boolean_velocity", "bad_index", "boolean_index", "bad_limit"])
def test_shared_window_gate_rejects_invalid_boundary_inputs(invalid):
    estimators = tuple(B601WindowVelocityEstimator(.08) for _ in range(6))
    raw = (0.,)*5 if invalid == "short" else (float("nan"),)+(0.,)*5 if invalid == "nan" else (True,)+(0.,)*5 if invalid == "boolean_velocity" else (0.,)*6
    with pytest.raises(ValueError):
        window_stationarity_gate_reason(
            estimators, raw, (0.,)*6, soft_limit_rad_s=0. if invalid == "bad_limit" else .02,
            hard_limit_rad_s=.05, joint_index=6 if invalid == "bad_index" else True if invalid == "boolean_index" else None,
        )


@pytest.mark.parametrize("field", tuple(name for name in vars(B601JointMotionConfig()) if name not in {"windowed_soft_velocity", "reference_pacing", "responsive_tracking_brake", "arrival_margin_rad", "speed_governor", "support_soft_speed_recovery", "arrival_stabilization", "velocity_headroom_ratio"}))
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), True])
def test_motion_settings_reject_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(B601JointMotionConfig(), **{field: value})


@pytest.mark.parametrize("speed,expected", [(.0075, .01625), (.015, .026)])
def test_velocity_headroom_is_shared_bounded_and_preserves_legacy(speed, expected):
    config = B601JointMotionConfig(windowed_soft_velocity=True,
        position_resolution_rad=.0004, responsive_tracking_brake=True,
        max_reference_velocity_rad_s=speed)
    assert config.resolved_soft_velocity_rad_s == pytest.approx(speed + .005)
    candidate = replace(config, velocity_headroom_ratio=.30)
    assert candidate.resolved_soft_velocity_rad_s == pytest.approx(expected)
    assert replace(candidate, velocity_headroom_ratio=100).resolved_soft_velocity_rad_s == .03
    assert candidate.max_reference_velocity_rad_s == config.max_reference_velocity_rad_s
    assert candidate.max_control_gap_sec == config.max_control_gap_sec


@pytest.mark.parametrize("value", [-.01, True, float("nan"), float("inf")])
def test_velocity_headroom_requires_valid_explicit_resolved_control(value):
    with pytest.raises(ValueError):
        B601JointMotionConfig(velocity_headroom_ratio=value)
    with pytest.raises(ValueError, match="explicit responsive"):
        B601JointMotionConfig(velocity_headroom_ratio=.30)


@pytest.mark.parametrize("margin,velocity,expected", [
    (0., .014, "brake"), (.30, .014, "track"), (.30, .017, "brake"),
])
def test_tracking_headroom_admits_normal_band_but_still_brakes_above_it(margin, velocity, expected):
    tracker = _tracker(windowed_soft_velocity=True, position_resolution_rad=.0004,
        responsive_tracking_brake=True, reference_pacing=True, speed_governor=True,
        max_reference_velocity_rad_s=.0075, velocity_headroom_ratio=margin)
    frame = _governor_tick(tracker, 1.01, q=0., velocity=velocity, span=.001)
    assert frame.phase == expected
    assert frame.tracking_speed_brake_limit_rad_s == pytest.approx(.0125*(1+margin))


@pytest.mark.parametrize("timeout,expected", [(3., "authority_exhausted"), (3.9, "brake")])
def test_braking_wait_headroom_keeps_a_finite_original_deadline(timeout, expected):
    tracker = _governed_tracker(braking_timeout_sec=timeout)
    _governor_tick(tracker, 1.01, velocity=.023, span=.002, zero=False)
    for n in range(1, 311):
        frame = _governor_tick(tracker, 1.01+n*.01, zero=False)
    assert frame.phase == expected
    assert frame.speed_governor_resume_count == 0
    for n in range(311, 402):
        frame = _governor_tick(tracker, 1.01+n*.01, zero=False)
    assert frame.phase == "authority_exhausted"
    assert frame.reason == "braking_timeout_assist_withdrawn"
    assert frame.requested_torque_nm == 0.


def test_reference_requires_matching_dispatch_and_missing_receipt_withdraws():
    tracker = _tracker()
    frame = _update(tracker)
    assert frame.phase == "track"
    tracker.record_dispatch(999)
    missing = _update(tracker, now=1.04, seq=2)
    assert missing.reason == "reference_dispatch_not_confirmed"
    assert missing.requested_torque_nm == 0.0


def test_existing_reference_planner_enforces_velocity_and_acceleration():
    tracker = _tracker()
    q, previous_v = 0.0, 0.0
    for i in range(1, 100):
        frame = _update(tracker, now=1 + i * 0.02, q=q, seq=i)
        assert frame.phase == "track"
        assert abs(frame.qvel_reference_rad_s) <= 0.015 + 1e-9
        assert abs(frame.qvel_reference_rad_s - previous_v) <= 0.03 * 0.02 + 1e-9
        assert abs(frame.q_reference_rad - q) <= 0.015 * 0.02 + 1e-9
        tracker.record_dispatch(i)
        q, previous_v = frame.q_reference_rad, frame.qvel_reference_rad_s


def test_reference_pacing_is_opt_in_and_rejects_non_boolean_values():
    assert B601JointMotionConfig().reference_pacing is False
    for value in (0, 1, "true", None):
        with pytest.raises(TypeError, match="boolean"):
            B601JointMotionConfig(reference_pacing=value)


def test_pacing_same_tick_start_commits_original_reference_before_renewal():
    tracker = _tracker(reference_pacing=True)
    first = _update(tracker, now=1., seq=1)
    assert first.phase == "track"
    assert first.q_reference_rad == 0
    assert first.qvel_reference_rad_s == 0
    tracker.record_dispatch(1)
    second = _update(tracker, now=1.01, seq=2)
    assert second.phase == "track"
    assert second.reference_speed_scale > 0
    tracker.record_dispatch(2)
    duplicate = _update(tracker, now=1.01, seq=3)
    assert duplicate.q_reference_rad == second.q_reference_rad
    assert duplicate.qvel_reference_rad_s == second.qvel_reference_rad_s


@pytest.mark.parametrize("direction", [-1, 1])
def test_pacing_reduces_reference_lead_without_resetting_or_claiming_stuck_motion(direction):
    baseline = _tracker(max_control_gap_sec=.04)
    paced = _tracker(max_control_gap_sec=.04, reference_pacing=True)
    for tracker in (baseline, paced):
        tracker.start(now=1., q=0., target=direction*.05)
    previous_q, previous_v, baseline_braked = 0., 0., False
    for i in range(1, 501):
        now = 1 + i*.01
        old = _update(baseline, now=now, seq=i)
        baseline.record_dispatch(i)
        baseline_braked |= old.reason == "reference_lag_braking"
        frame = _update(paced, now=now, seq=i)
        assert frame.phase == "track"
        assert 0 < frame.reference_speed_scale <= 1
        assert frame.reference_headroom_rad > 0
        assert abs(frame.q_reference_rad) < paced.config.max_reference_lag_rad
        assert abs(frame.qvel_reference_rad_s - previous_v) <= .03*.01 + 1e-9
        assert frame.q_reference_rad - previous_q == pytest.approx(frame.qvel_reference_rad_s*.01)
        assert frame.target_position_rad == direction*.05
        assert frame.target_error_rad == direction*.05  # The plant never moved.
        paced.record_dispatch(i)
        previous_q, previous_v = frame.q_reference_rad, frame.qvel_reference_rad_s
    assert baseline_braked
    assert frame.reference_speed_scale < .01
    # Slowing is not new authority: missing ACK still withdraws the next frame.
    _update(paced, now=6.01, seq=501)
    assert _update(paced, now=6.02, seq=502).reason == "reference_dispatch_not_confirmed"


@pytest.mark.parametrize("kwargs,reason", [
    ({"brake": True}, "operator_stop_and_hold"),
    ({"raw_v": .031}, "soft_velocity_braking"),
    ({"q": -.014}, "reference_lag_braking"),
    ({"q": .05}, "target_reached_braking"),
])
def test_reference_pacing_does_not_bypass_existing_brakes(kwargs, reason):
    tracker = _tracker(reference_pacing=True)
    if reason == "target_reached_braking":
        tracker.start(now=1., q=.048, target=.05)
    frame = _update(tracker, **kwargs)
    assert frame.phase == "brake"
    assert frame.reason == reason
    assert frame.position_torque_nm == 0
    assert frame.qvel_reference_rad_s == 0


def test_pacing_reserves_adverse_measured_motion_and_returns_to_existing_brake():
    frame = _update(_tracker(reference_pacing=True), q=-.012, control_v=-.02)
    assert frame.phase == "brake"
    assert frame.reason == "reference_pacing_headroom_braking"
    assert frame.reference_headroom_rad < 0
    assert frame.position_torque_nm == 0
    assert frame.velocity_torque_nm > 0


def test_reference_pacing_does_not_reanchor_when_feedback_catches_up():
    tracker = _tracker(reference_pacing=True, max_control_gap_sec=.04)
    q, previous_q, previous_v = 0., 0., 0.
    scales = []
    for i in range(1, 401):
        frame = _update(tracker, now=1+i*.01, q=q, seq=i)
        assert frame.phase == "track"
        assert abs(frame.qvel_reference_rad_s - previous_v) <= .03*.01 + 1e-9
        assert frame.q_reference_rad - previous_q == pytest.approx(frame.qvel_reference_rad_s*.01)
        scales.append(frame.reference_speed_scale)
        tracker.record_dispatch(i)
        previous_q, previous_v = frame.q_reference_rad, frame.qvel_reference_rad_s
        if i > 200:
            q = frame.q_reference_rad - .001  # Catch up without resetting history.
    assert min(scales[100:200]) < .1
    assert max(scales[250:]) == 1


def test_pacing_far_from_lead_boundary_keeps_original_planner_endpoint_approach():
    baseline, paced = _tracker(), _tracker(reference_pacing=True)
    for tracker in (baseline, paced):
        tracker.start(now=1., q=0., target=.004)
    q = 0.
    for i in range(1, 101):
        old = _update(baseline, now=1+i*.01, q=q, seq=i)
        frame = _update(paced, now=1+i*.01, q=q, seq=i)
        assert frame.q_reference_rad == old.q_reference_rad
        assert frame.qvel_reference_rad_s == old.qvel_reference_rad_s
        assert frame.requested_torque_nm == old.requested_torque_nm
        assert frame.phase == old.phase
        baseline.record_dispatch(i)
        paced.record_dispatch(i)
        q = frame.q_reference_rad


@pytest.mark.parametrize("direction", [-1, 1])
def test_quantized_pacing_reserves_one_count_and_pauses_without_losing_target(direction):
    config = B601JointMotionConfig(reference_pacing=True, windowed_soft_velocity=True,
        position_resolution_rad=.0004, responsive_tracking_brake=True, max_control_gap_sec=.04)
    tracker = B601JointMotionTracker(config)
    tracker.start(now=1., q=0., target=direction*.05)
    estimator = B601WindowVelocityEstimator(.08)
    for i in range(10):
        estimator.update(timestamp_sec=.91+i*.01, qpos_rad=0., fallback_rad_s=0.)
    previous_q, previous_v, measured = 0., 0., 0.
    paused = False
    for i in range(1, 501):
        now=1.+i*.01
        # Once the reference is almost at its lead envelope, alternate one
        # physically unresolved count. Do not fabricate a zero window speed.
        q = -direction*.000384 if i>180 and i%20<10 else 0.
        raw=(q-measured)/.01
        control=estimator.update(timestamp_sec=now,qpos_rad=q,fallback_rad_s=raw)
        measured=q
        evidence=B601WindowStopEvidence(now,(0.,0.,q,0.,0.,0.),(0.,0.,raw,0.,0.,0.),
            (0.,0.,control,0.,0.,0.),(0.,0.,estimator.position_span_rad,0.,0.,0.),
            estimator.span_sec,.05)
        frame=tracker.update(now=now,q=q,raw_v=raw,control_v=control,control_sequence=i,
            authority=1.,kp=12.,kd=1.,stop_error_rad=.003,stop_velocity_rad_s=.005,
            stop_dwell_sec=.15,stop_evidence=evidence)
        assert frame.phase == "track", frame
        assert abs(frame.q_reference_rad-q) <= config.max_reference_lag_rad
        assert abs(frame.qvel_reference_rad_s-previous_v) <= .03*.01 + 1e-9
        assert frame.q_reference_rad-previous_q == pytest.approx(frame.qvel_reference_rad_s*.01)
        assert frame.target_position_rad == direction*.05
        assert frame.reference_position_reserve_rad == .0004
        paused |= frame.reference_speed_scale == 0.
        tracker.record_dispatch(i)
        previous_q,previous_v=frame.q_reference_rad,frame.qvel_reference_rad_s
    assert paused
    # Measured catch-up resumes the same endpoint through the same planner.
    moving_q = measured
    for i in range(501, 551):
        now=1.+i*.01
        q=moving_q+direction*.00006
        raw=(q-moving_q)/.01
        control=estimator.update(timestamp_sec=now,qpos_rad=q,fallback_rad_s=raw)
        moving_q=q
        evidence=B601WindowStopEvidence(now,(0.,0.,q,0.,0.,0.),(0.,0.,raw,0.,0.,0.),
            (0.,0.,control,0.,0.,0.),(0.,0.,estimator.position_span_rad,0.,0.,0.),
            estimator.span_sec,.05)
        frame=tracker.update(now=now,q=q,raw_v=raw,control_v=control,control_sequence=i,
            authority=1.,kp=12.,kd=1.,stop_error_rad=.003,stop_velocity_rad_s=.005,
            stop_dwell_sec=.15,stop_evidence=evidence)
        assert frame.phase == "track"
        assert frame.target_position_rad == direction*.05
        assert abs(frame.qvel_reference_rad_s-previous_v) <= .03*.01 + 1e-9
        assert frame.q_reference_rad-previous_q == pytest.approx(frame.qvel_reference_rad_s*.01)
        tracker.record_dispatch(i)
        previous_q,previous_v=frame.q_reference_rad,frame.qvel_reference_rad_s
    assert frame.reference_speed_scale > 0
    # The independent lead guard is still an actual brake, not a larger range.
    q=-direction*.002
    evidence=B601WindowStopEvidence(6.51,(0.,0.,q,0.,0.,0.),(0.,)*6,(0.,)*6,
        (0.,0.,.002,0.,0.,0.),.08,.05)
    frame=tracker.update(now=6.51,q=q,raw_v=0.,control_v=0.,control_sequence=551,
        authority=1.,kp=12.,kd=1.,stop_error_rad=.003,stop_velocity_rad_s=.005,
        stop_dwell_sec=.15,stop_evidence=evidence)
    assert frame.phase == "brake" and frame.reason == "reference_lag_braking"


def test_quantized_pacing_still_forecasts_resolved_reverse_travel():
    tracker=_tracker(reference_pacing=True,windowed_soft_velocity=True,
        position_resolution_rad=.0004,responsive_tracking_brake=True,max_control_gap_sec=.04)
    now,q,raw,window=1.01,-.0125,-.015,-.015
    evidence=B601WindowStopEvidence(now,(0.,0.,q,0.,0.,0.),(0.,0.,raw,0.,0.,0.),
        (0.,0.,window,0.,0.,0.),(0.,0.,.001,0.,0.,0.),.08,.05)
    frame=tracker.update(now=now,q=q,raw_v=raw,control_v=window,control_sequence=1,
        authority=1.,kp=12.,kd=1.,stop_error_rad=.003,stop_velocity_rad_s=.005,
        stop_dwell_sec=.15,stop_evidence=evidence)
    assert frame.phase == "brake" and frame.reason == "reference_pacing_headroom_braking"
    assert frame.reference_safety_headroom_rad == pytest.approx(.0005-.015*.04)


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"brake": True}, "operator_stop_and_hold"),
        ({"raw_v": 0.031}, "soft_velocity_braking"),
        ({"q": -0.014}, "reference_lag_braking"),
        ({"q": 0.05}, "target_reached_braking"),
    ],
)
def test_brake_has_no_old_target_position_spring(kwargs, reason):
    tracker = _tracker()
    if reason == "target_reached_braking":
        tracker.start(now=1.0, q=0.048, target=0.05)
    frame = _update(tracker, **kwargs)
    assert frame.phase == "brake"
    assert frame.reason == reason
    assert frame.position_torque_nm == 0.0
    assert frame.qvel_reference_rad_s == 0.0
    assert frame.velocity_torque_nm * kwargs.get("raw_v", 0.0) <= 0.0


def test_braking_raw_direction_wins_over_delayed_opposite_window_velocity():
    frame = _update(_tracker(), raw_v=-0.04, control_v=0.06)
    assert frame.phase == "brake"
    assert frame.requested_torque_nm > 0.0


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("raw,window,expected", [
    (.058, -.023, 0.),       # Actual reversal in cr02hmu0: old window would propel recoil.
    (.096, .052, -.104),
    (.013, .052, -.026),     # Both estimates bound magnitude, not the raw count alone.
    (0., .052, 0.),
    (.0384, 0., 0.),
])
def test_continuous_brake_never_injects_torque_against_latest_direction_evidence(
    direction, raw, window, expected,
):
    tracker = _governed_tracker(continuous_speed_limit_rad_s=.039)
    frame = _governor_tick(tracker, 1.02, q=.001, velocity=direction*window,
                           raw=direction*raw, span=.002, brake=True, raw_limit=.13)
    assert frame.phase == "brake"
    assert frame.position_torque_nm == 0.
    assert frame.velocity_torque_nm == pytest.approx(direction*expected)
    assert frame.velocity_torque_nm * direction*raw <= 0.
    assert abs(frame.velocity_torque_nm) <= 2*abs(window)+1e-12
    assert frame.hold_position_rad is None


def test_brake_retains_trigger_reference_and_reports_unreached_original_target():
    tracker = _tracker()
    braking = _update(tracker, q=-0.014)
    assert braking.reason == "reference_lag_braking"
    assert braking.q_reference_rad == -0.014
    assert braking.brake_trigger_reference_rad > braking.q_reference_rad
    assert braking.brake_trigger_lag_rad > 0.013
    assert braking.target_position_rad == 0.05
    assert braking.target_error_rad == pytest.approx(0.064)
    stopped = _update(tracker, now=1.16, q=-0.014)
    assert stopped.phase == "soft_hold"
    assert stopped.hold_position_rad == -0.014
    assert stopped.target_error_rad == pytest.approx(0.064)
    assert stopped.brake_trigger_reference_rad == braking.brake_trigger_reference_rad
    tracker.start(now=2, q=0.02, target=0.04)
    next_target = _update(tracker, now=2.02, q=0.02)
    assert next_target.brake_trigger_reference_rad is None
    assert next_target.target_position_rad == .04


def test_capture_actual_stop_not_initial_pose_or_target_and_hold_is_bounded():
    tracker = _tracker()
    first = _update(tracker, q=0.018, raw_v=0.02, control_v=0.02, brake=True)
    assert first.phase == "brake"
    _update(tracker, now=1.04, q=0.021)
    stopped = _update(tracker, now=1.16, q=0.021)
    assert stopped.phase == "soft_hold"
    assert stopped.hold_position_rad == 0.021
    held = _update(tracker, now=1.18, q=0.025)
    assert held.position_torque_nm < 0.0
    assert held.requested_torque_nm == pytest.approx(-0.008)
    saturated = _update(tracker, now=1.20, q=0.07)
    assert abs(saturated.requested_torque_nm) == 0.05


def test_brake_needs_continuous_stationarity_and_has_timeout():
    tracker = _tracker(braking_timeout_sec=0.3)
    _update(tracker, brake=True)
    _update(tracker, now=1.1, raw_v=0.01)
    assert _update(tracker, now=1.15).phase == "brake"
    assert _update(tracker, now=1.21).phase == "brake"
    exhausted = _update(tracker, now=1.34, raw_v=0.01)
    assert exhausted.reason == "braking_timeout_assist_withdrawn"
    assert exhausted.requested_torque_nm == 0.0


def test_drive_damping_is_explicit_six_axis_and_zero_by_default():
    assert B601JointMotionConfig().drive_driver_kd_nm_s_per_rad == (0,) * 6
    config = B601JointMotionConfig(drive_driver_kd_nm_s_per_rad=[.1] * 6)
    assert config.drive_driver_kd_nm_s_per_rad == (.1,) * 6
    for vector in ((0,) * 5, (-.1,) * 6, (float("nan"),) * 6, (True,) * 6, (5.1,) * 6):
        with pytest.raises(ValueError):
            B601JointMotionConfig(drive_driver_kd_nm_s_per_rad=vector)


def test_shared_motor_damping_slew_has_symmetric_rate_bound_and_exact_targets():
    assert slew_driver_damping((0.,)*6, (.6,)*6, rate_per_sec=2., dt_sec=.01) == (.02,)*6
    assert slew_driver_damping((.6,)*6, (0.,)*6, rate_per_sec=2., dt_sec=.01) == pytest.approx((.58,)*6)
    assert slew_driver_damping((.01,)*6, (0.,)*6, rate_per_sec=2., dt_sec=.01) == (0.,)*6
    for bad in ((0.,)*5, (-.1,)*6, (True,)*6, (float("nan"),)*6, (5.1,)*6):
        with pytest.raises(ValueError):
            slew_driver_damping(bad, (0.,)*6, rate_per_sec=2., dt_sec=.01)
    for rate, dt in ((0., .01), (float("inf"), .01), (2., -.01), (2., float("nan"))):
        with pytest.raises(ValueError):
            slew_driver_damping((0.,)*6, (.6,)*6, rate_per_sec=rate, dt_sec=dt)


def test_stop_capture_requires_support_axes_to_settle():
    tracker = _tracker()
    def tick(t, support):
        return tracker.update(
            now=t, q=.01, raw_v=0, control_v=0, control_sequence=1,
            authority=1, kp=6, kd=1, stop_error_rad=.002,
            stop_velocity_rad_s=.005, stop_dwell_sec=.1,
            support_max_velocity_rad_s=support,
        )
    first = tick(1.02, .031)
    assert first.phase == "brake"
    assert first.reason == "support_velocity_braking"
    assert first.position_torque_nm == 0
    assert tick(1.15, .01).phase == "brake"
    assert tick(1.20, 0).phase == "brake"
    assert tick(1.32, 0).phase == "soft_hold"

    # Support motion also prevents a previously captured point staying held.
    assert tick(1.34, .04).phase == "brake"


def test_windowed_soft_velocity_is_explicit_and_keeps_raw_stop_dwell():
    for value in (0, 1, "true", None):
        with pytest.raises(TypeError, match="boolean"):
            B601JointMotionConfig(windowed_soft_velocity=value)
    tracker = _tracker(windowed_soft_velocity=True)
    frame = _update(tracker, raw_v=.0384, control_v=.015)
    assert frame.phase == "track"
    tracker.record_dispatch(1)
    frame = _update(tracker, now=1.04, raw_v=.0384, control_v=.031, seq=2)
    assert frame.phase == "brake"
    assert frame.velocity_torque_nm == pytest.approx(-.062)
    # Low window speed must not turn one raw zero sample into a stop capture.
    assert _update(tracker, now=1.06).phase == "brake"
    assert _update(tracker, now=1.08, raw_v=.0384, control_v=.003).phase == "brake"
    assert _update(tracker, now=1.14).phase == "brake"
    assert _update(tracker, now=1.26).phase == "soft_hold"


def test_windowed_stop_capture_also_bounds_position_change():
    tracker = _tracker(windowed_soft_velocity=True)
    _update(tracker, brake=True)
    assert _update(tracker, now=1.14, q=.0006).phase == "brake"
    assert _update(tracker, now=1.26, q=.0006).phase == "soft_hold"


@pytest.mark.parametrize("raw_v,control_v,support_window,support_raw,reason", [
    (.0384, .001, 0., 0., "selected_raw_velocity"),
    (0., .006, 0., 0., "selected_window_velocity"),
    (0., 0., .006, 0., "support_window_velocity"),
    (0., 0., 0., .006, "support_raw_velocity"),
])
def test_stop_capture_reports_existing_velocity_gate_without_relaxing_it(
    raw_v, control_v, support_window, support_raw, reason,
):
    tracker = _tracker(windowed_soft_velocity=True)
    frame = tracker.update(
        now=1.02, q=.001, raw_v=raw_v, control_v=control_v, control_sequence=1,
        authority=1., kp=6., kd=1., stop_error_rad=.002,
        stop_velocity_rad_s=.005, stop_dwell_sec=.15, request_brake=True,
        support_max_velocity_rad_s=support_window, support_raw_max_velocity_rad_s=support_raw,
    )
    assert frame.phase == "brake" and frame.hold_position_rad is None
    assert frame.stop_capture_block_reason == reason
    assert frame.stop_capture_dwell_elapsed_sec == 0.
    assert frame.stop_capture_velocity_limit_rad_s == .005
    assert frame.stop_capture_required_dwell_sec == .15
    assert frame.position_torque_nm == 0.
    assert frame.requested_torque_nm == pytest.approx(-2 * control_v)


def test_stop_capture_reports_continuous_dwell_and_position_reset():
    tracker = _tracker(windowed_soft_velocity=True)
    first = _update(tracker, brake=True)
    assert first.stop_capture_block_reason == "stationary_dwell_incomplete"
    middle = _update(tracker, now=1.07)
    assert middle.stop_capture_dwell_elapsed_sec == pytest.approx(.05)
    reset = _update(tracker, now=1.10, q=.0006)
    assert reset.stop_capture_block_reason == "measured_position_changed"
    assert reset.stop_capture_dwell_elapsed_sec == 0.
    captured = _update(tracker, now=1.22, q=.0006)
    assert captured.phase == "soft_hold"
    assert captured.stop_capture_block_reason is None
    assert captured.hold_position_rad == .0006

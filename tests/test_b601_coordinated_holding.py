"""Coordinated software contracts; no devices or measured force evidence."""

import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_b601_automatic_calibration import AcquisitionReplay
from test_b601_calibration_session import settings as shared_settings
from test_b601_workspace_calibration import LIMITS, production_controller

from control_station.b601_torque_console.workspace_calibration import (
    COORDINATED_CALIBRATION_OVERRIDES,
    B601WorkspaceCalibrationController,
    WorkspaceCalibrationConfig,
)
from robot_motion.rebot_b601_rs.coordinated_holding import (
    ZERO,
    CoordinatedHoldingConfig,
    CoordinatedPoseHolding,
)
from robot_motion.rebot_b601_rs.holding_calibration import HoldingSearchConfig

pytestmark = [pytest.mark.unit, pytest.mark.contract]


@pytest.fixture
def settings(tmp_path):
    return shared_settings.__wrapped__(tmp_path)


def runtime(settings):
    from test_b601_holding_calibration import holding_replay
    resolved = holding_replay(settings).controller.settings
    return CoordinatedPoseHolding(resolved, CoordinatedHoldingConfig(total_assist_cap_nm=.3, bias_cap_nm=.2))


def warm(core):
    for n in range(12):
        dt = core.observe(1+n*.01, ZERO, ZERO)
        core.step(sequence=n, dt=dt, permitted=False)
        core.record_dispatch(n, ZERO)
    core.set_goal((0, .01, .01, 0, 0, 0))


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("conditioned_tail", [False, True])
@pytest.mark.parametrize("software_kd", [0., 6.])
def test_supported_soft_stop_opposes_slip_and_resumes_only_after_actual_support_receipt(
    settings, direction, conditioned_tail, software_kd,
):
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(total_assist_cap_nm=.3, bias_cap_nm=.2,
        supported_soft_recovery=True, recovery_kd_nm_s_per_rad=software_kd)
    warm(core)
    core.adopt_hold(1, .1)
    core.adopt_hold(2, -direction*.1)
    before_goal = core.goal
    saw_support = resumed = False
    for n in range(1, 100):
        q = (0., 0., direction*min(n, 3)*.0008, 0., 0., 0.)
        raw = tuple((a-b)/.025 for a, b in zip(q, core.q))
        dt = core.observe(1.11+n*.025, q, raw)
        previous = core.last_delta
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True, point_search=False)
        assert sum(abs(v) for v in command) <= .3+1e-9
        if audit["supported_soft_stop"]:
            saw_support = True
            if abs(core.velocity[2]) > .02:
                assert command[2]*core.velocity[2] < 0
                assert command[1] > 0  # Do not drop a quiet peer's support.
        if any(f["reason"] == "speed_governor_resumed_supported_target" for f in audit["axes"]):
            assert core.quiet() and core._support_receipt
            assert max(abs(v) for v in previous) > 0
            assert not conditioned_tail or n > 30
            resumed = True
        actual = list(command)
        if conditioned_tail and n <= 30:
            actual[1] = .001 if command[1] != .001 else .002
        core.record_dispatch(n+11, actual)
        assert core.goal == before_goal
        if resumed:
            break
    assert saw_support and resumed
    dt = core.observe(core.last_time+.025, core.q, ZERO)
    command, _ = core.step(sequence=1000, dt=dt, permitted=False)
    core.record_dispatch(1000, command)
    assert command == ZERO and core.bias == ZERO
    assert core._support_anchor is None and not core._support_receipt


def test_supported_recovery_does_not_bypass_reference_lag_or_raw_velocity(settings, monkeypatch):
    from robot_motion.rebot_b601_rs.joint_motion import (
        B601JointMotionFrame,
    )

    core = runtime(settings)
    core.config = replace(core.config, supported_soft_recovery=True)
    warm(core)
    failed = B601JointMotionFrame(phase="soft_hold", reason="measured_stop_point_captured",
        q_reference_rad=.02, qvel_reference_rad_s=0, control_qvel_rad_s=0,
        position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0,
        brake_trigger_lag_rad=core.motion.max_reference_lag_rad*1.01)
    monkeypatch.setattr(core.trackers[0], "update", lambda **kwargs: failed)
    with pytest.raises(ValueError, match="coordinated_reference_lag_stop"):
        core.step(sequence=12, dt=.01, permitted=True)
    core.raw = (0, 0, core.settings.limits.max_joint_velocity_rad_s*1.01, 0, 0, 0)
    with pytest.raises(ValueError, match="raw joint velocity"):
        core.step(sequence=12, dt=.01, permitted=True)


@pytest.mark.parametrize("software_kd", [0., 6.])
def test_supported_stop_retains_one_witnessed_seed_without_recursive_spring_debit(settings, monkeypatch, software_kd):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=software_kd)
    core.settings = replace(core.settings, limits=replace(core.settings.limits,
        max_abs_assist_nm=1., max_joint_velocity_rad_s=.195))
    warm(core)
    core.bias = core.last_delta = (0., .2, .3, 0., 0., 0.)
    for tracker in core.trackers:
        frame = B601JointMotionFrame(phase="brake", reason="soft_velocity_braking",
            q_reference_rad=0, qvel_reference_rad_s=0, control_qvel_rad_s=0,
            position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0,
            speed_governor_active=True)
        monkeypatch.setattr(tracker, "update", lambda frame=frame, **kwargs: frame)
    for n in range(30):
        q = (0., 0., min(n, 5)*.001, 0., 0., 0.)
        raw = tuple((a-b)/.01 for a, b in zip(q, core.q))
        dt = core.observe(1.12+n*.01, q, raw)
        command, audit = core.step(sequence=12+n, dt=dt, permitted=True)
        core.record_dispatch(12+n, command)
        assert core._brake_seed == (0., .2, .3, 0., 0., 0.)
        if software_kd == 0:
            assert audit["support_damping_owner"] == "motor_local_profile"
            assert audit["support_damping_velocity_rad_s"][2] == 0
    # Allow the original finite torque slew to reach the stable plateau.
    assert command[2] == pytest.approx(.3-core.motion.tracking_kp_nm_per_rad*.005)
    assert command[1] == .2


@pytest.mark.parametrize("axis,value", [(1, 0), (2, 0), (2, 5.01)])
def test_motor_local_recovery_requires_configured_bounded_driver_damping(axis, value):
    gains = [.4, 3., 3., 1.2, .2, .1]
    gains[axis] = value
    with pytest.raises(ValueError, match="motor-local recovery"):
        WorkspaceCalibrationConfig(coordinated_holding=CoordinatedHoldingConfig(
            supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0),
            holding_search=HoldingSearchConfig(), joint2_drive_kd=gains, joint3_drive_kd=gains)


@pytest.mark.parametrize("value", ["0", True, float("nan"), -.1])
def test_recovery_software_damping_rejects_invalid_coefficients(value):
    with pytest.raises(ValueError, match="recovery"):
        CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=value)


@pytest.mark.parametrize("direction", [-1, 1])
def test_recorded_slip_speed_requests_fast_bounded_support_not_zero(settings, monkeypatch, direction):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    # x5ipi26g at 946231.407: |window v3|=0.1309, brake overlay=0.2618.
    # Compare commands at that same state, not future physical trajectories.
    outputs = []
    for supported in (False, True):
        core = runtime(settings)
        core.config = CoordinatedHoldingConfig(supported_soft_recovery=supported)
        core.settings = replace(core.settings, limits=replace(core.settings.limits,
            max_abs_assist_nm=1., max_joint_velocity_rad_s=.195))
        warm(core)
        core.velocity = core.raw = (0., 0., direction*.1309, 0., 0., 0.)
        core.last_delta = (0., .1, -direction*.2618, 0., 0., 0.)
        core.bias = (0., .1, 0., 0., 0., 0.)
        for i, tracker in zip((1, 2), core.trackers):
            frame = B601JointMotionFrame(phase="brake", reason="soft_velocity_braking",
                q_reference_rad=core.q[i], qvel_reference_rad_s=0,
                control_qvel_rad_s=core.velocity[i], position_torque_nm=0,
                velocity_torque_nm=-2*core.velocity[i], requested_torque_nm=-2*core.velocity[i],
                speed_governor_active=True)
            monkeypatch.setattr(tracker, "update", lambda frame=frame, **kwargs: frame)
        command, audit = core.step(sequence=12, dt=.025, permitted=True)
        assert sum(abs(v) for v in command) <= 1+1e-9
        assert command[2]*core.velocity[2] < 0
        assert audit["supported_soft_stop"] == supported
        core.record_dispatch(12, command)
        outputs.append(command)
    assert outputs[0][2] == pytest.approx(-direction*.2618)
    assert abs(outputs[1][2]) > abs(outputs[0][2])
    assert outputs[1][1] == pytest.approx(.1)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("recent_ratio", [None, 2., .5, 0., -1.])
def test_supported_damping_releases_stale_window_tail_without_amplifying_or_erasing_support(
    settings, monkeypatch, direction, recent_ratio,
):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True)
    core.settings = replace(core.settings, limits=replace(core.settings.limits,
        max_abs_assist_nm=1., max_joint_velocity_rad_s=.195))
    warm(core)
    # Same-state command comparison of 9b78u5th's late brake window.
    velocity = direction*.06047901995951213
    core.velocity = (0., 0., velocity, 0., 0., 0.)
    core.raw = ZERO
    core.bias = (0., .1, 0., 0., 0., 0.)
    core.last_delta = (0., .1, -6*velocity, 0., 0., 0.)
    for i, tracker in zip((1, 2), core.trackers):
        recent = None if recent_ratio is None else core.velocity[i]*recent_ratio
        frame = B601JointMotionFrame(phase="brake", reason="resolved_tracking_speed_braking",
            q_reference_rad=core.q[i], qvel_reference_rad_s=0,
            control_qvel_rad_s=core.velocity[i], position_torque_nm=0,
            velocity_torque_nm=-2*core.velocity[i], requested_torque_nm=-2*core.velocity[i],
            speed_governor_active=True, tracking_recent_velocity_rad_s=recent)
        monkeypatch.setattr(tracker, "update", lambda frame=frame, **kwargs: frame)
    command, audit = core.step(sequence=12, dt=.025, permitted=True)
    fraction = 1. if recent_ratio is None else max(0., min(1., recent_ratio))
    assert audit["support_damping_velocity_rad_s"][2] == pytest.approx(velocity*fraction)
    expected = -6*velocity*fraction
    previous = core.last_delta[2]
    bound = core.motion.braking_slew_nm_s*.025
    assert command[2] == pytest.approx(previous+max(-bound, min(bound, expected-previous)))
    assert command[1] == .1
    assert sum(abs(v) for v in command) <= 1+1e-9
    core.record_dispatch(12, command)


@pytest.mark.parametrize("period", [.01, .025])
@pytest.mark.parametrize("joint", [1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
def test_resolved_working_tail_uses_brake_slew_only_toward_zero(settings, period, joint, direction):
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig()
    core.last_delta = tuple(direction*.86248278 if i == joint else 0. for i in range(6))
    core.velocity = core.raw = tuple(direction*.01381 if i == joint else 0. for i in range(6))
    command, reason, rate = core._slew_axis(joint, direction*.4312406856, period, True)
    assert command == pytest.approx(direction*max(.4312406856, .86248278-core.motion.braking_slew_nm_s*period))
    assert reason == "resolved_working_tail_unload"
    assert rate == core.motion.braking_slew_nm_s
    # A reversed target drains the old drive to zero, never fast reverse drive.
    core.last_delta = tuple(direction*.01 if i == joint else 0. for i in range(6))
    command, _, _ = core._slew_axis(joint, -direction*.2, period, True)
    assert command == 0.
    core.last_delta = ZERO
    command, _, rate = core._slew_axis(joint, -direction*.2, period, True)
    assert command == pytest.approx(-direction*core.config.rise_slew_nm_s*period)
    assert rate == core.config.rise_slew_nm_s


@pytest.mark.parametrize("resolved,raw", [(False, .01), (True, 0.), (True, -.01)])
def test_fast_tail_unload_requires_resolved_and_agreeing_motion(settings, resolved, raw):
    core = runtime(settings)
    core.last_delta = (0, 0, .2, 0, 0, 0)
    core.velocity, core.raw = (0, 0, .01, 0, 0, 0), (0, 0, raw, 0, 0, 0)
    command, _, rate = core._slew_axis(2, .1, .025, resolved)
    assert command == pytest.approx(.2-core.config.fall_slew_nm_s*.025)
    assert rate == core.config.fall_slew_nm_s


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("scale", [0., .4, 1.])
@pytest.mark.parametrize("driving", [False, True])
def test_speed_governor_limits_composed_bias_but_not_fixed_probes(settings, monkeypatch, sign, probe, scale, driving):
    core = runtime(settings)
    warm(core)
    core.config = CoordinatedHoldingConfig()
    bias = sign*.8*(1 if driving else -1)
    core.bias = (0., 0., bias, 0., 0., 0.)
    core.velocity = core.raw = (0., 0., sign*.05, 0., 0., 0.)
    # Reuse complete native frames; isolate only the already-authoritative
    # speed factor so this tests composition, not a second velocity estimator.
    template = core.trackers[1].update(now=core.last_time, q=core.q[2], raw_v=0., control_v=0.,
        control_sequence=12, authority=1., kp=2., kd=2.,
        stop_error_rad=.003, stop_velocity_rad_s=.005, stop_dwell_sec=.15)
    first = replace(template, phase="track", reason="bounded_reference_tracking",
        requested_torque_nm=0., continuous_speed_scale=1.)
    second = replace(template, phase="track",
        reason="calibration_fixed_command_observation" if probe else "bounded_reference_tracking",
        requested_torque_nm=sign*.2 if probe else 0., continuous_speed_scale=scale)
    monkeypatch.setattr(core.trackers[0], "update", lambda **kwargs:first)
    monkeypatch.setattr(core.trackers[1], "update", lambda **kwargs:second)
    monkeypatch.setattr(core, "_slew_axis", lambda i, target, dt, resolved_motion, **kwargs:(target, "isolated_composition", 1.))
    command, audit = core.step(sequence=12, dt=.01, permitted=True,
        probe_joint=2 if probe else None, probe_nm=sign*.2 if probe else None,
        point_search=False)
    expected_scale = scale if driving and not probe else 1.
    assert audit["composed_speed_scales"][2] == expected_scale
    assert command[2] == pytest.approx(sign*.2 if probe else bias*expected_scale)
    assert core.bias[2] == bias  # No persistent state commit before receipt.
    assert sum(abs(v) for v in command) <= 1.
    if not probe and driving and scale == 0:
        assert command[2] == 0.


@pytest.mark.parametrize("joint", [1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("phase", ["track", "arrival", "supported_stop"])
@pytest.mark.parametrize("source", ["window", "recent"])
def test_composite_governor_covers_arrival_away_motion_and_supported_stop(
    settings, monkeypatch, joint, sign, phase, source,
):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0.)
    warm(core)
    # The nominal goal is in the opposite direction to this retained drive.
    goal = list(core.goal)
    goal[joint] = -sign*.01
    core.goal = tuple(goal)
    core.bias = core.last_delta = tuple(sign*.6 if i == joint else 0. for i in range(6))
    limit = core.motion.continuous_speed_limit_rad_s
    start = core.motion.resolved_soft_velocity_rad_s
    speed = (limit+start)/2
    core.observe(core.last_time+.01,
                 tuple(sign*.002 if i == joint and source == "window" else 0. for i in range(6)), ZERO)
    core.velocity = tuple(sign*speed if i == joint and source == "window" else 0. for i in range(6))
    # Isolate full-demand speed composition from position-aware bias release,
    # which has its own receipt/reversal regressions below.
    monkeypatch.setattr(core, "_approach_bias",
        lambda i, frame, dt, moving, anchor, **kwargs: (core.bias[i], moving, anchor, "isolated_bias"))
    for i in (1, 2):
        frame = B601JointMotionFrame(
            phase="track" if phase == "track" else "brake",
            reason={"track": "bounded_reference_tracking", "arrival": "target_reached_braking",
                    "supported_stop": "soft_velocity_braking"}[phase],
            q_reference_rad=core.q[i], qvel_reference_rad_s=0., control_qvel_rad_s=core.velocity[i],
            position_torque_nm=0., velocity_torque_nm=0., requested_torque_nm=0.,
            speed_governor_active=phase == "supported_stop",
            tracking_recent_velocity_rad_s=sign*speed if i == joint and source == "recent" else 0.,
            tracking_recent_position_span_rad=.002 if i == joint and source == "recent" else 0.,
            # A goal-directed governor sees no toward-goal movement here.
            continuous_speed_scale=1. if phase == "track" else None)
        monkeypatch.setattr(core.trackers[i-1], "update", lambda frame=frame, **kwargs: frame)
    monkeypatch.setattr(core, "_slew_axis", lambda i, target, dt, resolved_motion, **kwargs:(target, "isolated_composition", 1.))
    command, audit = core.step(sequence=12, dt=.01, permitted=True, point_search=False)
    assert audit["composed_speed_scales"][joint] == pytest.approx(.5)
    assert command[joint] == pytest.approx(sign*.3)
    assert core.bias[joint] == sign*.6  # Receipt still owns all persistent state.
    assert sum(abs(v) for v in command) <= 1.
    core.record_dispatch(12, command)
    assert core.last_delta == command


@pytest.mark.parametrize("sign", [-1, 1])
def test_composite_speed_envelope_preserves_static_support_jitter_and_opposing_effort(settings, sign):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    warm(core)
    frame = B601JointMotionFrame(phase="brake", reason="target_reached_braking",
        q_reference_rad=0., qvel_reference_rad_s=0., control_qvel_rad_s=0.,
        position_torque_nm=0., velocity_torque_nm=0., requested_torque_nm=0.)
    core.raw = (0., 0., sign*.19, 0., 0., 0.)
    core.velocity = (0., 0., sign*.1, 0., 0., 0.)
    # Unresolved windows, regardless of a raw spike, cannot unload support.
    assert core._composed_speed_scale(2, frame, sign*.6) == 1.
    core.observe(core.last_time+.01, (0., 0., sign*.002, 0., 0., 0.), ZERO)
    core.velocity = (0., 0., sign*core.motion.continuous_speed_limit_rad_s, 0., 0., 0.)
    assert core._composed_speed_scale(2, frame, sign*.6) == 0.
    assert core._composed_speed_scale(2, frame, -sign*.6) == 1.
    core.velocity = ZERO
    assert core._composed_speed_scale(2, frame, sign*.6) == 1.


@pytest.mark.parametrize("joint", [1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("mismatched_receipt", [False, True])
def test_arrival_releases_starting_increment_only_through_actual_receipt(
    settings, monkeypatch, joint, sign, mismatched_receipt,
):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0.)
    warm(core)
    core.adopt_hold(joint, sign*.216)
    core.adopt_hold(3-joint, -.1)
    core.set_goal(tuple(sign*.01 if i == joint else core.q[i] for i in range(6)))
    core.bias = tuple(sign*.84 if i == joint else value for i, value in enumerate(core.bias))
    core.last_delta = core.bias
    core.observe(core.last_time+.01,
        tuple(sign*.009 if i == joint else 0. for i in range(6)), ZERO)
    core.velocity = core.raw = tuple(sign*.01 if i == joint else 0. for i in range(6))
    for i, tracker in zip((1, 2), core.trackers):
        frame = B601JointMotionFrame(phase="brake", reason="target_reached_braking",
            q_reference_rad=core.goal[i], qvel_reference_rad_s=0., control_qvel_rad_s=core.velocity[i],
            position_torque_nm=0., velocity_torque_nm=0., requested_torque_nm=0.)
        monkeypatch.setattr(tracker, "update", lambda frame=frame, **kwargs: frame)
    old_bias = core.bias
    command, audit = core.step(sequence=12, dt=.01, permitted=True, point_search=False)
    assert audit["approach_reasons"][joint] == "arrival_releases_departure_drive"
    assert audit["proposed_bias_nm"][joint] == pytest.approx(sign*.216)
    assert command[joint] == pytest.approx(sign*(.84-core.motion.braking_slew_nm_s*.01))
    assert command[3-joint] == -.1
    assert core.bias == old_bias
    actual = tuple(value/2 if i == joint else value for i, value in enumerate(command)) if mismatched_receipt else command
    core.record_dispatch(12, actual)
    assert core.bias[joint] == pytest.approx(sign*(.84 if mismatched_receipt else .216))
    assert core.last_delta == actual
    assert sum(abs(v) for v in actual) <= 1.


@pytest.mark.parametrize("sign", [-1, 1])
def test_released_arrival_drive_does_not_return_when_velocity_drops(settings, monkeypatch, sign):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0.)
    warm(core)
    core.adopt_hold(2, sign*.2)
    core.set_goal((0., 0., sign*.01, 0., 0., 0.))
    core.bias = core.last_delta = (0., 0., sign*.8, 0., 0., 0.)
    for n in range(1, 45):
        dt = core.observe(1.11+n*.01, (0., 0., sign*.009, 0., 0., 0.), ZERO)
        if n == 1:
            core.velocity = core.raw = (0., 0., sign*.01, 0., 0., 0.)
        for i, tracker in zip((1, 2), core.trackers):
            frame = B601JointMotionFrame(phase="brake", reason="target_reached_braking",
                q_reference_rad=core.goal[i], qvel_reference_rad_s=0., control_qvel_rad_s=core.velocity[i],
                position_torque_nm=0., velocity_torque_nm=0., requested_torque_nm=0.)
            monkeypatch.setattr(tracker, "update", lambda frame=frame, **kwargs: frame)
        command, audit = core.step(sequence=11+n, dt=dt, permitted=True, point_search=False)
        core.record_dispatch(11+n, command)
        assert core.bias[2] == pytest.approx(sign*.2)
        assert abs(command[2]) <= .8+1e-9
    assert command[2] == pytest.approx(sign*.2)
    assert audit["approach_reasons"][2] == "arrival_retains_support_without_extra_integration"
    core.adopt_hold(2, sign*.35)
    assert core._departure_bias[2] == sign*.35  # A later verified hold replaces the prior.
    dt = core.observe(core.last_time+.01, core.q, ZERO)
    command, _ = core.step(sequence=100, dt=dt, permitted=False)
    core.record_dispatch(100, command)
    assert core.bias == ZERO and core._departure_bias == ZERO


@pytest.mark.parametrize("sign", [-1, 1])
def test_moving_error_unwinds_obsolete_bias_without_integrating_reverse_drive(settings, sign):
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0.)
    core.goal = (0., 0., 0., 0., 0., 0.)
    core.q = (0., 0., sign*.006, 0., 0., 0.)
    core.velocity = (0., 0., sign*.01, 0., 0., 0.)
    core.estimators = tuple(SimpleNamespace(position_span_rad=.002) for _ in range(6))
    core.bias = (0., 0., sign*.84, 0., 0., 0.)
    frame = SimpleNamespace(reason="bounded_reference_tracking", q_reference_rad=0., requested_torque_nm=-sign*.1)
    bias, _, _, reason = core._approach_bias(2, frame, .01, True, 0., point_search=False)
    assert reason == "moving_target_error_unwinds_bias"
    assert bias == pytest.approx(sign*(.84-core.config.bias_gain_nm_per_rad_sec*.006*.01))
    for _ in range(800):
        core.bias = (0., 0., bias, 0., 0., 0.)
        bias, _, _, _ = core._approach_bias(2, frame, .01, True, 0., point_search=False)
        assert bias*sign >= 0
    assert bias == pytest.approx(0.)


def test_recorded_uirvzgyc_arrival_releases_increment_not_departure_support(settings):
    # Same-state branch comparison, not a prediction of later real motion.
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(bias_gain_nm_per_rad_sec=40.,
        supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0.)
    core.goal = (0., 0., .04235987755982989, 0., 0., 0.)
    core.q = (0., 0., math.radians(2.4610234527745525), 0., 0., 0.)
    core.velocity = core.raw = (0., 0., -.010164443993499674, 0., 0., 0.)
    core.bias = (0., 0., -.8423252060973547, 0., 0., 0.)
    core._departure_bias = (0., 0., -.2161, 0., 0., 0.)
    core.last_delta = (0., 0., -.8392777780224778, 0., 0., 0.)
    core.estimators = tuple(SimpleNamespace(position_span_rad=.002) for _ in range(6))
    frame = SimpleNamespace(reason="target_reached_braking", q_reference_rad=core.goal[2],
                            requested_torque_nm=.003047428074876908)
    bias, moving, _, reason = core._approach_bias(2, frame, .01, True, .04525, point_search=False)
    assert reason == "arrival_releases_departure_drive" and moving
    assert bias == -.2161
    assert frame.requested_torque_nm+bias == pytest.approx(-.2130525719251231)


@pytest.mark.parametrize("sign", [-1, 1])
def test_arrival_tail_fast_release_cannot_turn_into_fast_reverse_drive(settings, sign):
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig()
    core.last_delta = (0., 0., sign*.1, 0., 0., 0.)
    command, reason, rate = core._slew_axis(2, -sign*.5, .025, False, arrival_unload=True)
    assert reason == "arrival_drive_tail_unload"
    assert rate == core.motion.braking_slew_nm_s
    assert command*sign >= 0
    core.last_delta = ZERO
    command, _, rate = core._slew_axis(2, -sign*.5, .025, False, arrival_unload=True)
    assert command == pytest.approx(-sign*core.config.rise_slew_nm_s*.025)
    assert rate == core.config.rise_slew_nm_s


@pytest.mark.parametrize("sign", [-1, 1])
def test_anti_windup_keeps_reductions_but_blocks_growth_and_reverse_integration(settings, sign):
    core = runtime(settings)
    core.bias = (0., 0., sign*.6, 0., 0., 0.)
    assert core._unwind_only(2, sign*.2) == sign*.2
    assert core._unwind_only(2, sign*.8) == sign*.6
    assert core._unwind_only(2, -sign*.1) == 0.


@pytest.mark.parametrize("limit", ["speed", "combined_envelope"])
def test_limited_arrival_cannot_restore_the_unloaded_increment(settings, monkeypatch, limit):
    from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame

    core = runtime(settings)
    core.config = CoordinatedHoldingConfig(supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0.)
    warm(core)
    core.adopt_hold(1, .1)
    core.adopt_hold(2, .2)
    core.set_goal((0., 0., .01, 0., 0., 0.))
    core.bias = core.last_delta = (0., .1, .8, 0., 0., 0.)
    core.observe(core.last_time+.01, (0., 0., .009, 0., 0., 0.), ZERO)
    core.velocity = core.raw = (0., 0., .01, 0., 0., 0.)
    for i, tracker in zip((1, 2), core.trackers):
        frame = B601JointMotionFrame(phase="brake", reason="target_reached_braking",
            q_reference_rad=core.goal[i], qvel_reference_rad_s=0., control_qvel_rad_s=core.velocity[i],
            position_torque_nm=0., velocity_torque_nm=0.,
            requested_torque_nm=.8 if limit == "combined_envelope" and i == 1 else 0.,
            continuous_speed_scale=.4 if limit == "speed" and i == 2 else None)
        monkeypatch.setattr(tracker, "update", lambda frame=frame, **kwargs: frame)
    command, audit = core.step(sequence=12, dt=.01, permitted=True, point_search=False)
    assert audit["composed_speed_scales"][2] < 1 if limit == "speed" else audit["saturated"]
    assert audit["proposed_bias_nm"][2] == .2
    assert core.bias[2] == .8
    core.record_dispatch(12, command)
    assert core.bias[2] == .2 and sum(abs(v) for v in command) <= 1.


def test_both_axes_share_one_envelope_and_commit_only_after_actual_receipt(settings):
    core = runtime(settings)
    warm(core)
    seen = False
    for n in range(12, 150):
        dt = core.observe(1+n*.01, ZERO, ZERO)
        before = core.bias
        command, audit = core.step(sequence=n, dt=dt, permitted=True)
        assert core.bias == before
        assert sum(abs(v) for v in command) <= .3+1e-12
        assert all(command[i] == 0 for i in (0, 3, 4, 5))
        seen |= command[1] > 0 and command[2] > 0
        assert audit["target_qpos_rad"][1:3] == (.01, .01)
        core.record_dispatch(n, command)
    assert seen


def test_missing_duplicate_and_mismatched_receipts_never_commit(settings):
    core = runtime(settings)
    warm(core)
    dt = core.observe(1.12, ZERO, ZERO)
    command, _ = core.step(sequence=12, dt=dt, permitted=True)
    with pytest.raises(ValueError, match="previous dispatch"):
        core.observe(1.13, ZERO, ZERO)
    with pytest.raises(ValueError, match="pending frame"):
        core.record_dispatch(13, command)
    core.record_dispatch(12, command)
    with pytest.raises(ValueError, match="pending frame"):
        core.record_dispatch(12, command)


@pytest.mark.parametrize("period", [.01, .025])
@pytest.mark.parametrize("joint", [1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("point_search", [False, True])
def test_resolved_assisting_motion_proposes_half_actual_command_once(settings, period, joint, direction, point_search):
    core = runtime(settings)
    warm(core)
    goal = list(ZERO)
    goal[joint] = direction*.04
    core.set_goal(goal)
    for n in range(1, 61):
        dt = core.observe(1.11+n*period, ZERO, ZERO)
        command, _ = core.step(sequence=n+11, dt=dt, permitted=True)
        core.record_dispatch(n+11, command)
    events = 0
    for n in range(61, 101):
        q = list(ZERO)
        q[joint] = direction*(n-60)*period*.008
        raw = tuple((a-b)/period for a,b in zip(q, core.q))
        dt = core.observe(1.11+n*period, q, raw)
        before = core.bias
        previous = core.last_delta[joint]
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True, point_search=point_search)
        assert audit["point_search_enabled"] is point_search
        assert core.bias == before
        assert not audit["braking"]
        if audit["approach_reasons"][joint] == "resolved_motion_half_command_proposal":
            events += 1
            native = audit["axes"][joint-1]["requested_torque_nm"]
            assert audit["proposed_bias_nm"][joint]+native == pytest.approx(previous/2)
            assert audit["torque_transition_reasons"][joint] == "resolved_working_tail_unload"
            assert audit["torque_transition_rates_nm_s"][joint] == core.motion.braking_slew_nm_s
            assert abs(command[joint]) == pytest.approx(max(abs(previous/2),
                abs(previous)-core.motion.braking_slew_nm_s*dt))
        core.record_dispatch(n+11, command)
        assert sum(abs(v) for v in command) <= core.config.total_assist_cap_nm+1e-9
    assert events == int(point_search)


def test_nuhnenxy_return_motion_retains_support_instead_of_proposing_midpoint(settings):
    # Recorded return at 20362.171582; this tests the causal branch, not
    # a prediction of what the real arm will do after the changed command.
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig()
    core.q = (0, math.radians(2.41708), 0, 0, 0, 0)
    core.velocity = (0, -.012335, 0, 0, 0, 0)
    core.last_delta = (0, -.379231, 0, 0, 0, 0)
    core.bias = (0, -.162208, 0, 0, 0, 0)
    core.estimators = (None, SimpleNamespace(position_span_rad=.000767), None)
    frame = SimpleNamespace(requested_torque_nm=-.210129, q_reference_rad=math.radians(1.38124))
    anchor = core.q[1]+.002
    bias, moving, _, reason = core._approach_bias(1, frame, .025, False, anchor, point_search=False)
    assert bias == core.bias[1]
    assert moving and reason == "resolved_motion_continuous_tracking"
    search_bias, _, _, reason = core._approach_bias(1, frame, .025, False, anchor)
    assert reason == "resolved_motion_half_command_proposal"
    assert frame.requested_torque_nm+search_bias == pytest.approx(core.last_delta[1]/2)


@pytest.mark.parametrize("flag", [None, 0, 1, "false"])
def test_point_search_flag_requires_explicit_boolean(settings, flag):
    core = runtime(settings)
    warm(core)
    with pytest.raises(TypeError, match="point search must be boolean"):
        core.step(sequence=12, dt=.01, permitted=True, point_search=flag)


@pytest.mark.parametrize("joint", [1, 2])
@pytest.mark.parametrize("lag_fault", [False, True])
def test_continuous_transit_keeps_native_terminal_guards_and_failed_axis(settings, monkeypatch, joint, lag_fault):
    from robot_motion.rebot_b601_rs.joint_motion import (
        B601JointMotionFrame,
    )

    core = runtime(settings)
    warm(core)
    failed = B601JointMotionFrame(
        phase="soft_hold", reason="measured_stop", q_reference_rad=.02,
        qvel_reference_rad_s=0, control_qvel_rad_s=0, position_torque_nm=0,
        velocity_torque_nm=0, requested_torque_nm=0,
        brake_trigger_lag_rad=core.motion.max_reference_lag_rad*1.01 if lag_fault else None)
    monkeypatch.setattr(core.trackers[joint-1], "update", lambda **kwargs: failed)
    reason = "coordinated_reference_lag_stop" if lag_fault else "coordinated_target_stopped_before_goal"
    with pytest.raises(ValueError, match=reason):
        core.step(sequence=12, dt=.01, permitted=True, point_search=False)
    assert core.fault_frame["joint_name"] == f"joint{joint+1}"
    assert core.fault_frame["brake_trigger_lag_rad"] == failed.brake_trigger_lag_rad
    assert core.pending is None


@pytest.mark.parametrize("supported,software_kd", [(False, 6.), (True, 6.), (True, 0.)])
def test_coordinated_fault_retains_actual_failed_axis_and_new_brake_clock(settings, supported, software_kd):
    core = runtime(settings)
    core.config = replace(core.config, supported_soft_recovery=supported, recovery_kd_nm_s_per_rad=software_kd)
    warm(core)
    core.set_goal(ZERO)
    for tracker in core.trackers:
        tracker.config = replace(tracker.config, braking_timeout_sec=.2)
    for n in range(1, 41):
        dt = core.observe(1.11+n*.025, ZERO, ZERO)
        command, _ = core.step(sequence=n+11, dt=dt, permitted=True)
        core.record_dispatch(n+11, command)
    for n in range(1, 21):
        # Equal peer motion triggers both trackers together; J2 is evaluated
        # first, regardless of the workspace's displayed active task joint.
        q = (n*.001, 0, 0, 0, 0, 0)
        dt = core.observe(2.11+n*.025, q, (.04, 0, 0, 0, 0, 0))
        try:
            command, _ = core.step(sequence=n+51, dt=dt, permitted=True)
        except ValueError as exc:
            assert str(exc) == "braking_timeout_assist_withdrawn"
            break
        core.record_dispatch(n+51, command)
    else:
        pytest.fail("continued resolved motion must still exhaust the real brake deadline")
    assert core.fault_frame["joint_name"] == "joint2"
    assert core.fault_frame["brake_started_at_sec"] > 2.11
    assert .2 <= core.fault_frame["brake_elapsed_sec"] <= .225+1e-9
    assert core.fault_frame["requested_torque_nm"] == 0.


def test_one_count_noise_and_opposing_load_do_not_halve_support(settings):
    for opposing in (False, True):
        core = runtime(settings)
        warm(core)
        core.adopt_hold(2, -.1 if opposing else .1)
        for n in range(1, 100):
            q = (0, 0, n*.0002 if opposing else (n%2)*.000383496, 0, 0, 0)
            raw = tuple((a-b)/.025 for a,b in zip(q, core.q))
            dt = core.observe(1.11+n*.025, q, raw)
            command, audit = core.step(sequence=n+11, dt=dt, permitted=True)
            assert "resolved_motion_half_command_proposal" not in audit["approach_reasons"]
            core.record_dispatch(n+11, command)


def test_fqpl1w8y_same_motion_sample_reduces_instead_of_accumulating_bias(settings):
    # Recorded cw4:sweep-3 at 12585.719749760. This is a branch replay,
    # not a counterfactual simulation of subsequent physical motion.
    core = runtime(settings)
    core.config = CoordinatedHoldingConfig()
    core.q = (0, 0, .052923835813999176, 0, 0, 0)
    core.velocity = (0, 0, -.007647546304050753, 0, 0, 0)
    core.last_delta = (0, 0, -.4194492729307253, 0, 0, 0)
    core.bias = (0, 0, -.2972894895922154, 0, 0, 0)
    core.estimators = (None, None, SimpleNamespace(position_span_rad=.0003834962844848633))
    frame = SimpleNamespace(requested_torque_nm=-.11911995274598071,
                            q_reference_rad=.04235987755982989)
    before = core.bias
    bias, moving, _, reason = core._approach_bias(2, frame, .025, False, .05522480607032776)
    old_bias = before[2]+20*(frame.q_reference_rad-core.q[2])*.025
    assert abs(old_bias) > abs(before[2])
    assert frame.requested_torque_nm+bias == pytest.approx(core.last_delta[2]/2)
    assert moving and reason == "resolved_motion_half_command_proposal"
    assert core.bias == before


@pytest.mark.parametrize("direction", [-1, 1])
def test_motion_midpoint_cannot_cancel_stronger_native_deceleration(settings, direction):
    core = runtime(settings)
    core.q = (0, 0, direction*.002, 0, 0, 0)
    core.velocity = (0, 0, direction*.01, 0, 0, 0)
    core.last_delta = (0, 0, direction*.2, 0, 0, 0)
    core.bias = (0, 0, direction*.1, 0, 0, 0)
    core.estimators = (None, None, SimpleNamespace(position_span_rad=.002))
    frame = SimpleNamespace(requested_torque_nm=-direction*.25, q_reference_rad=0)
    bias, _, _, reason = core._approach_bias(2, frame, .025, False, 0)
    assert reason == "motion_unload_keeps_stronger_native_braking"
    assert bias == core.bias[2]
    assert (frame.requested_torque_nm+bias)*core.velocity[2] < 0


def test_unmatched_receipt_cannot_commit_motion_half_or_recovery_state(settings):
    core = runtime(settings)
    warm(core)
    core.adopt_hold(2, .1)
    for n in range(1, 60):
        q = (0, 0, n*.0002, 0, 0, 0)
        dt = core.observe(1.11+n*.025, q, (0, 0, .008, 0, 0, 0))
        before = (core.bias, core._approach_anchor, core._approach_moving, core._brake_seed)
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True)
        if "resolved_motion_half_command_proposal" in audit["approach_reasons"]:
            core.record_dispatch(n+11, ZERO)
            assert (core.bias, core._approach_anchor, core._approach_moving, core._brake_seed) == before
            return
        core.record_dispatch(n+11, command)
    pytest.fail("no resolved assisting-motion proposal")


@pytest.mark.parametrize("joint", [1, 2])
def test_brake_recovery_proposal_waits_for_zero_receipt_and_cannot_survive_revocation(settings, joint):
    core = runtime(settings)
    warm(core)
    goal = list(ZERO)
    goal[joint] = .04
    core.set_goal(goal)
    saw_brake = saw_proposal = False
    for n in range(1, 150):
        q = list(ZERO)
        if n > 30:
            q[joint] = min(n-30, 4)*.0008
        raw = tuple((a-b)/.025 for a,b in zip(q, core.q))
        dt = core.observe(1.11+n*.025, q, raw)
        previous = core.last_delta
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True)
        if audit["braking"]:
            saw_brake = True
            assert audit["proposed_bias_nm"] == ZERO
        if "post_brake_reduced_command_proposal" in audit["approach_reasons"]:
            assert n > 65 and max(abs(v) for v in previous) == 0 and core.quiet()
            saw_proposal = True
        actual = list(command)
        if 30 < n <= 65:
            actual[3-joint] = .01  # Final conditioner tail blocks native resume.
        core.record_dispatch(n+11, actual)
        if saw_proposal:
            break
    assert saw_brake and saw_proposal
    # A withdrawal is not a soft-brake handoff: no stale recovery survives it.
    core._brake_seed = (0, .1, .1, 0, 0, 0)
    dt = core.observe(core.last_time+.025, core.q, ZERO)
    command, _ = core.step(sequence=1000, dt=dt, permitted=False)
    core.record_dispatch(1000, command)
    assert command == core.bias == ZERO and core._brake_seed is None
    core._brake_seed = (0, .1, .1, 0, 0, 0)
    core.set_goal(core.goal)
    assert core._brake_seed is None and core._approach_anchor == core.q
    core._brake_seed = (0, .1, .1, 0, 0, 0)
    dt = core.observe(core.last_time+.025, core.q, ZERO)
    core.step(sequence=1001, dt=dt, permitted=False)
    core.record_dispatch(1001, (0, .01, .01, 0, 0, 0))
    assert core._brake_seed is None


def test_saturated_native_brake_cannot_restore_a_prebrake_integral(settings):
    core = runtime(settings)
    warm(core)
    core.config = CoordinatedHoldingConfig(total_assist_cap_nm=.05, bias_cap_nm=.05)
    core.set_goal((0, .04, .04, 0, 0, 0))
    core.adopt_hold(1, .02)
    core.adopt_hold(2, .02)
    for n in range(1, 20):
        q = (0, min(n, 4)*.0008, min(n, 4)*.0008, 0, 0, 0)
        raw = tuple((a-b)/.025 for a,b in zip(q, core.q))
        dt = core.observe(1.11+n*.025, q, raw)
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True)
        core.record_dispatch(n+11, command)
        if audit["braking"] and audit["saturated"]:
            assert audit["proposed_bias_nm"] == core.bias == ZERO
            return
    pytest.fail("native braking did not exercise the common final limiter")


@pytest.mark.parametrize("value", [0, -.001, .1, float("nan"), True])
def test_stage_arrival_alignment_cannot_relax_the_native_arrival_envelope(settings, value):
    core = runtime(settings)
    with pytest.raises(ValueError):
        CoordinatedPoseHolding(core.settings, core.config, arrival_tolerance_rad=value)


@pytest.mark.parametrize("period", [.01, .025])
@pytest.mark.parametrize("direction", [-1, 1])
def test_coordinated_quantized_stall_preserves_goal_through_predictive_stop(settings, period, direction):
    core = runtime(settings)
    warm(core)
    goal = (0, 0, direction*.04, 0, 0, 0)
    core.set_goal(goal)
    saw_predictive_stop = saw_resume = False
    for n in range(1, 1001):
        # A static joint with one-count changes; no identified plant claimed.
        q = (0, 0, direction*(n//17 % 2)*.000383496, 0, 0, 0)
        raw = tuple((a-b)/period for a,b in zip(q, core.q))
        dt = core.observe(1.11+n*period, q, raw)
        command, audit = core.step(sequence=11+n, dt=dt, permitted=True)
        selected = audit["axes"][1]
        assert selected["reason"] != "reference_lag_braking"
        assert selected["target_position_rad"] == goal[2]
        saw_predictive_stop |= selected["reason"] == "reference_pacing_headroom_braking"
        saw_resume |= selected["reason"] == "speed_governor_resumed_original_target"
        core.record_dispatch(11+n, command)
    assert saw_predictive_stop and saw_resume


def test_coordinated_soft_stop_near_goal_does_not_reanchor_to_wrong_point(settings):
    core = runtime(settings)
    warm(core)
    goal = (0, 0, .004, 0, 0, 0)
    core.set_goal(goal)
    # Support-axis motion requests the native stop; the selected axis remains
    # short of its goal. Subsequent feedback is quiet, not a new target.
    for n in range(1, 150):
        q = (0, 0, .0008, min(n, 4)*.0009, 0, 0)
        raw = tuple((a-b)/.025 for a,b in zip(q, core.q))
        dt = core.observe(1.11+n*.025, q, raw)
        command, audit = core.step(sequence=11+n, dt=dt, permitted=True)
        core.record_dispatch(11+n, command)
    assert audit["axes"][1]["q_reference_rad"] == goal[2]
    assert core.bias[2] > 0


@pytest.mark.parametrize("moving_axis", [1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("conditioned_tail", [False, True])
def test_50txgbn2_peer_arrival_effort_cannot_deadlock_the_full_zero_receipt(
    settings, moving_axis, direction, conditioned_tail,
):
    core = runtime(settings)
    warm(core)
    peer = 3-moving_axis
    goal = list(ZERO)
    goal[moving_axis], goal[peer] = direction*.04, -direction*.0017
    core.set_goal(goal)
    saw_conflict = saw_wait = resumed = False
    for n in range(1, 100):
        q = list(ZERO)
        # Two early-speed steps trigger one tracker, but do not cross the
        # peer's support soft-speed ceiling. Remaining feedback is quiet.
        q[moving_axis] = direction*min(n, 2)*.0008
        raw = tuple((a-b)/.025 for a, b in zip(q, core.q))
        dt = core.observe(1.11+n*.025, q, raw)
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True)
        moving, support = audit["axes"][moving_axis-1], audit["axes"][peer-1]
        saw_conflict |= (audit["braking"] and support["reason"] == "target_reached_braking"
                         and abs(support["requested_torque_nm"]) > .001)
        saw_wait |= moving["reason"] == "speed_governor_waiting_for_zero_dispatch"
        if moving["reason"] == "speed_governor_resumed_original_target":
            assert not conditioned_tail or n > 30
            resumed = True
        actual = list(command)
        if conditioned_tail and n <= 30:
            actual[peer] = direction*.01
        core.record_dispatch(n+11, actual)
        assert core.goal == tuple(goal)
        assert sum(abs(v) for v in command) <= core.config.total_assist_cap_nm+1e-9
        if resumed:
            break
    assert saw_conflict and resumed
    if conditioned_tail:
        assert saw_wait


def test_normal_arrival_without_a_coordinated_stop_keeps_position_effort(settings):
    core = runtime(settings)
    warm(core)
    core.set_goal((0, .04, -.0017, 0, 0, 0))
    for n in range(1, 30):
        dt = core.observe(1.11+n*.025, ZERO, ZERO)
        command, audit = core.step(sequence=n+11, dt=dt, permitted=True)
        core.record_dispatch(n+11, command)
        assert not audit["braking"]
    assert audit["axes"][1]["reason"] == "target_reached_braking"
    assert command[2] < -.01


@pytest.mark.parametrize("state", ["idle", "position", "observing", "failed"])
def test_real_annotation_windows_exclude_coordinated_holds_without_report_error(state):
    from test_b601_command_calibration import trace_fixture

    from control_station.b601_torque_console.calibration_report import (
        summarize_calibration,
    )
    events, rows = trace_fixture()
    for row in rows:
        if row.get("kind") == "active_dispatch":
            row["torque_console"].update(
                holding_calibration={"enabled": True},
                automatic_calibration={"workspace": {"coordinated_holding_enabled": True}},
                calibration_trial={"batch_id": None if state == "idle" else "batch",
                    "trial_id": None, "role": "holding_not_for_running_fit", "state": state})
    report = summarize_calibration(events, rows, run_id="nonfit-coordinated")
    assert report["rejected"]["holding_search_not_running_resistance"] > 0
    assert not report["motion_windows"]


def test_non_holding_running_trials_still_reject_unknown_markers():
    from test_b601_command_calibration import trace_fixture

    from control_station.b601_torque_console.calibration_report import (
        summarize_calibration,
    )
    events, rows = trace_fixture()
    for row in rows:
        if row.get("kind") == "active_dispatch":
            row["torque_console"]["calibration_trial"] = {
                "batch_id": None, "trial_id": None, "role": "unknown", "state": "moving"}
    with pytest.raises(ValueError, match="invalid automatic calibration trial marker"):
        summarize_calibration(events, rows, run_id="invalid-running")


def test_reference_speed_is_bounded_and_changes_only_at_quiet_consumed_handoff(settings):
    core = runtime(settings)
    warm(core)
    bias, delta = core.bias, core.last_delta
    speed = core.motion.max_reference_velocity_rad_s / 2
    core.set_goal(core.goal, max_reference_velocity_rad_s=speed)
    assert all(t.config.max_reference_velocity_rad_s == speed for t in core.trackers)
    assert (core.bias, core.last_delta) == (bias, delta)
    with pytest.raises(ValueError, match="resolved ceiling"):
        core.set_goal(core.goal, max_reference_velocity_rad_s=core.motion.max_reference_velocity_rad_s*2)
    dt = core.observe(1.12, ZERO, ZERO)
    core.step(sequence=12, dt=dt, permitted=True)
    with pytest.raises(ValueError, match="consumed"):
        core.set_goal(core.goal)
    core.record_dispatch(12, ZERO)
    core.raw = (0, 1, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="stillness"):
        core.set_goal(core.goal)


@pytest.mark.parametrize("joint", [1, 2])
def test_probe_replaces_only_measured_axis_and_withdrawal_clears_both(settings, joint):
    core = runtime(settings)
    warm(core)
    for n in range(12, 70):
        dt = core.observe(1+n*.01, ZERO, ZERO)
        cmd, _ = core.step(sequence=n, dt=dt, permitted=True, probe_joint=joint, probe_nm=-.08)
        core.record_dispatch(n, cmd)
    assert cmd[joint] < 0
    assert cmd[3-joint] > 0
    dt = core.observe(1.7, ZERO, ZERO)
    cmd, _ = core.step(sequence=70, dt=dt, permitted=False)
    assert cmd == ZERO
    core.record_dispatch(70, cmd)
    assert core.bias == ZERO


def test_combined_cap_does_not_speed_up_normal_support_release(settings):
    core = runtime(settings)
    warm(core)
    for n in range(12, 200):
        dt = core.observe(1+n*.01, ZERO, ZERO)
        command, _ = core.step(sequence=n, dt=dt, permitted=True)
        core.record_dispatch(n, command)
    assert sum(abs(v) for v in command) == pytest.approx(.3)
    previous = command
    dt = core.observe(3.0, ZERO, ZERO)
    command, _ = core.step(sequence=200, dt=dt, permitted=True, probe_joint=1, probe_nm=.3)
    assert abs(command[2]-previous[2]) <= core.config.fall_slew_nm_s*dt+1e-12
    assert abs(command[1]-previous[1]) <= core.config.rise_slew_nm_s*dt+1e-12
    assert sum(abs(v) for v in command) <= .3+1e-12


def test_cancellation_waits_for_actual_zero_tail_base_damping_and_fresh_stillness(tmp_path):
    r = replay(tmp_path)
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(80):
        r.tick()
    before = r.controller.result_record()["workspace"]["command_exposure_nm_sec"]
    goal = r.controller._coordinated.pose.goal
    r.controller.submit("withdraw", now_sec=r.now)
    r.tick(final_delta=(0, .1, .05, 0, 0, 0))
    assert r.delta == ZERO
    assert r.controller.result_record()["state"] == "stopping"
    assert any(r.last_kd)
    r.tick()
    assert r.controller.result_record()["state"] == "stopping"
    assert r.controller.result_record()["workspace"]["command_exposure_nm_sec"] > before
    for _ in range(100):
        r.tick()
        if r.controller.result_record()["state"] == "cancelled":
            break
    assert r.controller.result_record()["state"] == "cancelled"
    assert r.last_kd == ZERO
    assert r.controller._coordinated.pose.goal == goal
    assert not r.controller.cycle_disable_confirmation_ready
    with pytest.raises(ValueError, match="idle owner"):
        r.controller.submit("arm", now_sec=r.now)


def replay(tmp_path, *, initial_q=ZERO, mode_wrapper=False, **changes):
    whole = production_controller.__wrapped__(tmp_path)
    legacy = whole._workspace
    cfg = replace(legacy.workspace, **{
        "joint2_centres_deg": (3,), "joint3_centres_deg": (3,),
        "gravity_observation_sec": .15, "holding_excursion_rad": .01,
        "holding_search": HoldingSearchConfig(observation_sec=.15, observation_timeout_sec=1),
        "coordinated_holding": CoordinatedHoldingConfig(), **changes})
    controller = B601WorkspaceCalibrationController(legacy.settings, legacy.config, cfg)
    if mode_wrapper:
        whole._workspace = controller
        whole.submit("select_mode", now_sec=1.0, operation_mode="workspace")
        controller = whole
    return AcquisitionReplay(legacy.settings, controller=controller, position_limits=LIMITS, initial_q=initial_q)


def test_gui_mode_idle_reentry_requires_new_stillness_but_does_not_relax_active_watchdog(tmp_path):
    r = replay(tmp_path, mode_wrapper=True)
    r.controller.submit("select_mode", now_sec=r.now, operation_mode="fixed_position")
    for _ in range(25):
        r.tick()
    r.controller.submit("select_mode", now_sec=r.now, operation_mode="workspace")
    r.tick()
    assert r.delta == ZERO
    assert not r.controller._workspace._coordinated.pose.quiet()
    for _ in range(12):
        r.tick()
    r.controller.submit("arm", now_sec=r.now, operation_mode="workspace")
    assert r.controller.status()["mode_selection"]["locked"]
    r.tick()
    with pytest.raises(ValueError, match="watchdog gap"):
        r.tick(dt_sec=.05)
    assert r.controller.result_record()["state"] == "failed"


def test_coordinated_profile_is_explicit_and_does_not_change_legacy(tmp_path):
    r = replay(tmp_path)
    assert r.controller.status()["automatic_calibration"]["workspace"]["coordinated_holding_enabled"]
    assert WorkspaceCalibrationConfig().coordinated_holding is None
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(80):
        r.tick()
    assert any(a["overlay_torque_nm"][2] > 0 for a in r.audits)
    assert r.audit["fault"] is None
    view = r.controller.status()
    j = int(view["joint_name"][-1])-1
    assert view["last_dispatch"]["receipt_sequence"] == r.receipt["sequence"]
    assert view["effective_assist_nm"] == pytest.approx(r.delta[j])
    assert view["last_dispatch"]["joint_commands"]["overlay_nm"] == r.delta
    assert view["motion_control"] is not None


@pytest.mark.parametrize("kind", ["human", "withdraw", "lease", "hard_speed", "receipt", "cap"])
@pytest.mark.parametrize("supported,software_kd", [(False, 6.), (True, 6.), (True, 0.)])
def test_all_axis_authority_and_final_envelope_remain_enforced(tmp_path, kind, supported, software_kd):
    r = replay(tmp_path, coordinated_holding=CoordinatedHoldingConfig(
        supported_soft_recovery=supported, recovery_kd_nm_s_per_rad=software_kd))
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(80):
        r.tick()
    if kind == "human":
        r.controller.set_human_intervention(True)
    elif kind == "withdraw":
        r.controller.submit("withdraw", now_sec=r.now)
    elif kind == "lease":
        r.controller._coordinated._permit = r.now-r.controller.limits.command_timeout_sec-1
    q = list(r.q)
    if kind == "hard_speed":
        q[1] += r.controller.limits.max_joint_velocity_rad_s*.011
    if kind in {"receipt", "cap"}:
        with pytest.raises(ValueError):
            r.tick(q, receipt_updates={"sent_joint_names": ["joint2"]} if kind == "receipt" else None,
                   final_delta=(0, .7, .7, 0, 0, 0) if kind == "cap" else None)
    else:
        r.tick(q, permit=kind != "lease")
        assert r.delta == ZERO
    assert r.controller.result_record()["state"] in {"stopping", "failed"}


def test_passive_joint_hard_velocity_reports_actual_fault_axis_not_task_axis(tmp_path):
    r = replay(tmp_path, coordinated_holding=CoordinatedHoldingConfig(supported_soft_recovery=True))
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(80):
        r.tick()
    q = list(r.q)
    q[3] += .20159287702047662*.01
    r.tick(q)
    assert r.audit["fault"] == "joint_velocity_exceeded"
    fault = r.audit["motion_fault"]
    assert fault["joint_name"] == "joint4"
    assert abs(fault["raw_qvel_rad_s"]) > fault["limit_rad_s"]
    assert r.audit["joint_name"] != fault["joint_name"]
    assert r.delta == ZERO
    assert r.controller.result_record()["state"] == "failed"


def test_coverage_local_damping_ramps_through_complete_receipts_and_restores_on_withdrawal(tmp_path):
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coverage_support_v1.json"
    r = replay(tmp_path, **json.loads(profile.read_text())["workspace"])
    r.controller.submit("arm", now_sec=r.now)
    workspace = r.controller._coordinated
    target = workspace.workspace.joint3_drive_kd
    previous = r.last_kd
    for _ in range(200):
        r.tick()
        assert max(abs(a-b) for a, b in zip(previous, r.last_kd)) <= .02+1e-9
        if workspace._phase == "preparing":
            assert r.delta == ZERO and workspace.pose.goal is None
        else:
            assert previous == target
        previous = r.last_kd
    assert r.last_kd == target == (.4, 3., 3., 1.2, .2, .1)
    r.controller.submit("withdraw", now_sec=r.now)
    for _ in range(200):
        r.tick()
        if r.controller.result_record()["state"] == "cancelled":
            break
    assert r.last_kd == r.delta == ZERO
    assert r.controller.result_record()["state"] == "cancelled"


def test_missing_local_damping_receipt_aborts_preparation_before_active_force(tmp_path):
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coverage_support_v1.json"
    r = replay(tmp_path, **json.loads(profile.read_text())["workspace"])
    r.controller.submit("arm", now_sec=r.now)
    with pytest.raises(ValueError, match="coordinated_damping_receipt_mismatch"):
        r.tick(kd_override=ZERO)
    assert r.delta == ZERO
    assert r.controller._coordinated.pose.goal is None
    assert r.controller.result_record()["state"] == "failed"


def coupled_tick(r, *, period=.025, passive_offset=0):
    """Declared overdamped toy, not identified B601 physics or a force sensor."""
    q = list(r.q)
    # Pose-dependent residual load, a Coulomb dead zone and off-diagonal
    # mobility: moving one axis can move the other without its command.
    load = (.04 + .25*q[2], -.05 - .2*q[1])
    net = [r.delta[i]-b for i, b in zip((1, 2), load)]
    drive = [(max(0, abs(v)-.07) * (1 if v > 0 else -1)) for v in net]
    velocity = ((8*drive[0]-2*drive[1])/60, (8*drive[1]-2*drive[0])/60)
    for i, v in zip((1, 2), velocity):
        q[i] += v*period
    q[3] += max(-.00015, min(.00015, passive_offset-q[3]))
    r.tick(q, dt_sec=period)


def scripted_coordinated_tick(r, *, passive_target=.006519419, period=.025):
    """Script feedback, not physics; consume real six-axis controller receipts."""
    q = list(r.q)
    goal = r.controller._coordinated.pose.goal
    if goal is not None and r.controller._coordinated._probe is None:
        for i in (1, 2):
            q[i] += max(-.00015, min(.00015, goal[i]-q[i]))
    q[3] += max(-.00015, min(.00015, passive_target-q[3]))
    r.tick(q, dt_sec=period)


def advance_to_coordinated_probe(r):
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(2500):
        scripted_coordinated_tick(r)
        if r.controller._coordinated._probe is not None:
            return
        assert r.controller.result_record()["state"] not in {"failed", "cancelled"}
    pytest.fail("stationary passive offset prevented holding-probe entry")


def test_position_arrival_accepts_settled_passive_offset_without_retargeting_it(tmp_path):
    r = replay(tmp_path)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    assert workspace._index == 3
    assert r.q[3] == pytest.approx(.006519419)
    assert workspace.pose.goal[3] == workspace._origin[3]
    assert all(abs(r.q[i]-workspace.pose.goal[i]) <= .0015 for i in (1, 2))
    assert workspace.pose.quiet()
    assert workspace._sample_pose == r.q
    assert workspace._sample_pose != workspace.pose.goal
    assert all(workspace.pose.goal[i] == workspace._sample_pose[i] for i in (1, 2))
    assert workspace._holding_status()["nominal_target_rad"] == workspace._nominal_target
    assert workspace._holding_status()["holding_reference_rad"] == workspace.pose.goal
    assert workspace.pose.trackers[0].config.max_reference_velocity_rad_s == workspace._steps[0]["max_reference_velocity_rad_s"]
    assert r.controller.status()["automatic_calibration"]["workspace"]["arrival"]["reason"] == "ready"


@pytest.mark.parametrize("response", ["creep", "settle", "one_count_jitter"])
def test_fixed_hold_needs_a_continuous_quiet_dwell_not_just_a_tolerated_displacement(tmp_path, response):
    r = replay(tmp_path, holding_search=HoldingSearchConfig(
        observation_sec=.5, observation_timeout_sec=1.5, movement_rad=.003))
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    for _ in range(50):
        r.tick(dt_sec=.025)
        if workspace._observation and workspace._observation["started_at_sec"] is not None:
            break
    else:
        pytest.fail("no causal constant-command plateau")
    assert workspace._search.status()["interior_recheck_enabled"]
    anchor = list(r.q)
    j = int(workspace._tasks[workspace._index]["joint"][-1])-1
    resolution = workspace.pose.motion.position_resolution_rad
    command = workspace._probe
    for n in range(1, 75):
        q = list(anchor)
        progress = n % 2 if response == "one_count_jitter" else min(n, 22) // 5 if response == "settle" else n // 5
        q[j] += resolution*progress
        r.tick(q, dt_sec=.025)
        if response != "one_count_jitter" and n <= 22:
            assert workspace._search.observations == []
        if workspace._search.observations:
            break
    observations = workspace._search.observations
    assert observations
    assert observations[0]["command_nm"] == command
    assert observations[0]["outcome"] == ("positive_motion" if response == "creep" else "stationary")
    if response != "creep":
        assert observations[0]["stationary_dwell_sec"] >= .5
    assert workspace._sample_pose == tuple(anchor)


@pytest.mark.parametrize("axis", [0, 1, 3, 4, 5])
@pytest.mark.parametrize("period", [.01, .025])
def test_fixed_probe_keeps_fine_position_dwell_through_bounded_count_jitter(tmp_path, axis, period):
    r = replay(tmp_path, holding_search=HoldingSearchConfig(
        observation_sec=.5, observation_timeout_sec=1.5, movement_rad=.003))
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    for _ in range(100):
        r.tick(dt_sec=period)
        if workspace._observation and workspace._observation["started_at_sec"] is not None:
            break
    else:
        pytest.fail("no actually dispatched constant-command plateau")
    anchor, search, command = r.q, workspace._search, workspace._probe
    not_quiet = False
    for n in range(int(1.3/period)):
        q = list(anchor)
        q[axis] += (1 if (n // round(.2/period)) % 2 else -1) * .000383496
        r.tick(q, dt_sec=period)
        not_quiet |= not workspace.pose.quiet()
        if search.observations:
            break
    assert not_quiet
    assert search.observations
    obs = search.observations[0]
    assert obs["outcome"] == "stationary"
    assert obs["command_nm"] == command
    assert obs["stationary_dwell_sec"] >= .5
    assert obs["completion_criterion"] == "fixed_command_fine_position_dwell"
    assert obs["handoff_quiet_confirmed"]
    assert obs["allowed_stationary_span_rad"] == 2*workspace.pose.motion.position_resolution_rad
    assert obs["stationary_maximum_rad"]-obs["stationary_minimum_rad"] <= obs["allowed_stationary_span_rad"]+1e-9
    assert max(b-a for a,b in zip(obs["minimum_rad"], obs["maximum_rad"])) <= .003
    assert workspace._sample_pose == anchor


@pytest.mark.parametrize("period", [.01, .025])
def test_fixed_probe_rejects_varying_support_effort_even_with_bounded_encoder_jitter(tmp_path, period):
    r = replay(tmp_path, holding_search=HoldingSearchConfig(
        observation_sec=.5, observation_timeout_sec=1.5, movement_rad=.003))
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    for _ in range(100):
        r.tick(dt_sec=period)
        if workspace._observation and workspace._observation["started_at_sec"] is not None:
            break
    anchor, search = r.q, workspace._search
    for n in range(int(1.8/period)):
        q = list(anchor)
        # J3 is the actively held support axis during this J2 probe. Its PD
        # response varies by more than the admitted command-context bound.
        q[2] += (1 if (n // round(.2/period)) % 2 else -1) * .000383496
        r.tick(q, dt_sec=period)
        rejected = [v for v in r.records if v["kind"] == "coordinated_holding_observation"
                    and v["outcome"] == "censored_context_not_stationary"]
        if rejected:
            break
    assert not search.observations
    assert rejected
    obs = rejected[0]
    assert max(b-a for a,b in zip(obs["minimum_rad"], obs["maximum_rad"])) < .003
    assert obs["maximum_delta_nm"][2]-obs["minimum_delta_nm"][2] > workspace.workspace.holding_search.torque_resolution_nm


@pytest.mark.parametrize("supported", [False, True])
def test_native_speed_brake_retains_meaningful_probe_progress_as_a_censored_transition(tmp_path, supported):
    r = replay(tmp_path, holding_search=HoldingSearchConfig(
        observation_sec=1.5, observation_timeout_sec=4.5, movement_rad=.0045),
        coordinated_holding=CoordinatedHoldingConfig(supported_soft_recovery=supported))
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    for _ in range(60):
        r.tick(dt_sec=.025)
        if workspace._observation and workspace._observation["started_at_sec"] is not None:
            break
    else:
        pytest.fail("no causal probe plateau")
    search, anchor = workspace._search, workspace._sample_pose
    j = int(workspace._tasks[workspace._index]["joint"][-1])-1
    requested = workspace._probe
    direction = 1 if requested >= 0 else -1
    for _ in range(8):
        q = list(r.q)
        q[j] += direction*.0016
        previous = r.delta[j]
        r.tick(q, dt_sec=.025)
        if search.transitions:
            break
    assert len(search.transitions) == 1
    transition = search.transitions[0]
    assert transition["outcome"] == "censored_native_brake"
    assert transition["command_nm"] == pytest.approx(previous)
    assert transition["requested_command_nm"] == requested
    assert search.candidate_nm == pytest.approx(previous/2)
    assert search.observations == []
    assert search.status()["positive_motion_above_nm"] is search.status()["negative_motion_below_nm"] is None
    assert workspace._sample_pose == anchor
    assert workspace._probe is None


@pytest.mark.parametrize("supported", [False, True])
def test_support_only_brake_does_not_invent_a_selected_axis_holding_response(tmp_path, supported):
    r = replay(tmp_path, coordinated_holding=CoordinatedHoldingConfig(supported_soft_recovery=supported))
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    search, candidate = workspace._search, workspace._probe
    for _ in range(8):
        q = list(r.q)
        q[2] += .002
        r.tick(q, dt_sec=.025)
        if workspace.pose.braking:
            break
    assert workspace.pose.braking
    assert search.transitions == search.observations == []
    assert search.candidate_nm == candidate
    assert workspace._probe is None


def test_workspace_leg_overrides_require_the_explicit_coordinated_mode():
    for name in COORDINATED_CALIBRATION_OVERRIDES:
        with pytest.raises(ValueError, match="explicit coordinated holding"):
            WorkspaceCalibrationConfig(**{name: 24})


@pytest.mark.parametrize("gate", ["time", "exposure"])
def test_explicit_workspace_leg_margin_replaces_only_the_old_gate_and_remains_finite(tmp_path, gate):
    r = replay(tmp_path, max_sweep_leg_sec=58.5, max_sweep_leg_command_exposure_nm_sec=24)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    assert r.controller.config.max_sweep_leg_sec == workspace.config.max_sweep_leg_sec == 58.5
    assert r.controller.config.max_sweep_leg_command_exposure_nm_sec == 24
    if gate == "time":
        workspace._leg_started = r.now-39.1
    else:
        workspace._leg_exposure = 16.1
    r.tick(dt_sec=.025)
    assert workspace._phase == "sweep"
    if gate == "time":
        workspace._leg_started = r.now-58.6
    else:
        workspace._leg_exposure = 24
    r.tick(dt_sec=.025)
    assert workspace._phase == "stopping"
    assert workspace._reason == ("leg_timeout" if gate == "time" else "leg_command_exposure_budget")
    view = r.controller.status()["automatic_calibration"]["workspace"]
    assert view["max_sweep_leg_sec"] == 58.5
    assert view["max_sweep_leg_command_exposure_nm_sec"] == 24
    assert view["leg_command_exposure_nm_sec"] == workspace._leg_exposure


@pytest.mark.parametrize("values", [
    {"max_sweep_leg_sec": 0}, {"max_sweep_leg_sec": float("nan")},
    {"max_sweep_leg_sec": True}, {"max_sweep_leg_sec": 10000},
    {"max_sweep_leg_command_exposure_nm_sec": -1},
    {"max_sweep_leg_command_exposure_nm_sec": float("inf")},
    {"max_sweep_leg_command_exposure_nm_sec": 10000},
])
def test_explicit_leg_overrides_are_validated_before_any_acquisition(tmp_path, values):
    with pytest.raises((ValueError, TypeError)):
        replay(tmp_path, **values)


def test_four_centres_profile_scales_only_whole_workspace_and_reuses_controller(tmp_path):
    from control_station.b601_torque_console.automatic_calibration import (
        load_automatic_profile,
    )

    profiles = Path(__file__).resolve().parents[1]/"docs/hardware/profiles"
    baseline = load_automatic_profile(profiles/"b601_j2_j3_coordinated_holding_v1.json")
    expanded = load_automatic_profile(profiles/"b601_j2_j3_four_centres_holding_v1.json")
    before = baseline.pop("workspace")
    after = expanded.pop("workspace")
    assert expanded == baseline
    assert {k for k in before if before[k] != after[k]} == {
        "joint2_centres_deg", "joint3_centres_deg",
        "joint2_acquisition_range_deg", "joint3_acquisition_range_deg",
        "max_workspace_sec", "max_command_exposure_nm_sec",
    }
    assert after == {**before, "joint2_centres_deg": [3, 8], "joint3_centres_deg": [3, 8],
                     "joint2_acquisition_range_deg": [0, 12], "joint3_acquisition_range_deg": [0, 12],
                     "max_workspace_sec": before["max_workspace_sec"]*4,
                     "max_command_exposure_nm_sec": before["max_command_exposure_nm_sec"]*4}
    whole = production_controller.__wrapped__(tmp_path)
    old = whole._workspace
    controller = B601WorkspaceCalibrationController(old.settings, old.config, WorkspaceCalibrationConfig(**after))
    controller.validate_position_limits(LIMITS)
    centres = tuple(tuple(math.degrees(v) for v in q) for q in controller.workspace.pose_centres())
    assert len(centres) == 4
    for actual, expected in zip(centres, ((3, 3), (3, 8), (8, 8), (8, 3))):
        assert actual == pytest.approx(expected)
    tasks = controller.workspace.tasks(ZERO)
    sweeps = [t for t in tasks if t["kind"] == "sweep"]
    assert [(t["pose"], t["joint"]) for t in sweeps] == [
        (pose, joint) for pose in range(4) for joint in ("joint2", "joint3")]
    assert sum(t["kind"] == "observe" for t in tasks) == 4
    assert [t["joint"] for t in tasks if t["kind"] == "return"] == ["joint2", "joint3"]
    assert tasks[-1]["kind"] == "return_observe"
    assert controller.config.max_sweep_leg_sec == before["max_sweep_leg_sec"]
    assert controller.config.max_sweep_leg_command_exposure_nm_sec == before["max_sweep_leg_command_exposure_nm_sec"]
    assert controller.workspace.coordinated_holding == WorkspaceCalibrationConfig(**before).coordinated_holding
    assert controller.settings.staged_assist.motion_control.max_control_gap_sec == .04
    with pytest.raises(ValueError, match="insufficient room"):
        B601WorkspaceCalibrationController(old.settings, old.config, replace(controller.workspace,
            joint2_acquisition_range_deg=(0, 10), joint3_acquisition_range_deg=(0, 10)))


def test_current_margin_profile_resolves_workspace_only_and_keeps_hard_envelopes(tmp_path):
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coordinated_holding_v1.json"
    values = json.loads(profile.read_text())["workspace"]
    whole = production_controller.__wrapped__(tmp_path)
    old = whole._workspace
    updated = B601WorkspaceCalibrationController(old.settings, old.config, WorkspaceCalibrationConfig(**values))
    for name, before in {"prepare_timeout_sec": 13,
                         "motion_braking_timeout_sec": 6.591,
                         "motion_soft_velocity_rad_s": .0585}.items():
        assert getattr(updated.config, name) == pytest.approx(before*1.3)
        assert getattr(old.config, name) == before
    assert updated.config.max_batch_sec == 1200
    assert updated.config.max_sweep_command_exposure_nm_sec == 480
    assert updated.config.max_sweep_leg_sec == 180
    assert updated.config.max_sweep_leg_command_exposure_nm_sec == 120
    assert old.config.max_batch_sec == 390
    assert old.config.max_sweep_command_exposure_nm_sec == 80
    assert updated._coordinated.pose.arrival_tolerance_rad == .003375
    motion = updated.settings.staged_assist.motion_control
    assert motion is updated._coordinated.pose.motion
    assert motion.soft_velocity_rad_s == motion.continuous_speed_limit_rad_s == .07605
    assert motion.braking_timeout_sec == 8.5683
    assert motion.braking_slew_nm_s == 20
    assert old.settings.staged_assist.motion_control.braking_slew_nm_s == 10
    assert old.settings.staged_assist.motion_control.soft_velocity_rad_s == .0585
    assert whole._fixed.config is old.config
    assert whole._supported.config is old.config
    assert updated.limits.max_joint_velocity_rad_s == old.limits.max_joint_velocity_rad_s
    assert updated.workspace.coordinated_holding.total_assist_cap_nm == 1
    assert motion.max_control_gap_sec == .04
    assert motion.drive_driver_kd_nm_s_per_rad == old.settings.staged_assist.motion_control.drive_driver_kd_nm_s_per_rad
    assert updated.workspace.holding_search.movement_rad == .0045
    assert updated.workspace.holding_search.observation_sec == 1.0
    assert updated.workspace.coordinated_holding.rise_slew_nm_s == 1.5
    assert updated.workspace.coordinated_holding.fall_slew_nm_s == 1.0
    assert updated.workspace.holding_search.observation_timeout_sec == pytest.approx(4.5*1.3)


def test_coverage_profile_reuses_paths_with_four_legs_and_keeps_hold_verification(tmp_path):
    profiles = Path(__file__).resolve().parents[1]/"docs/hardware/profiles"
    baseline = json.loads((profiles/"b601_j2_j3_four_centres_holding_v1.json").read_text())
    candidate = json.loads((profiles/"b601_j2_j3_coverage_support_v1.json").read_text())
    assert candidate["cycle"] == baseline["cycle"]
    assert candidate["workspace"]["holding_search"] == baseline["workspace"]["holding_search"]
    r = replay(tmp_path, **candidate["workspace"])
    controller = r.controller._coordinated
    workspace = controller.workspace
    total = 0
    for joint in ("joint2", "joint3"):
        plan = workspace.sweep_plan(condition=controller._condition, joint_name=joint,
            anchor_qpos_rad=(0, .05236, .05236, 0, 0, 0),
            position_limits_rad=controller._position_limits, calibration=controller.config)
        assert len(plan["steps"]) == 4
        assert [s["role"] for s in plan["steps"]] == ["training", "training", "validation", "validation"]
        assert {s["max_reference_velocity_rad_s"] for s in plan["steps"]} == {.015}
        assert len({tuple(s["target_qpos_rad"]) for s in plan["steps"]}) == 2
        total += len(plan["steps"])*len(workspace.pose_centres())
    assert total == 32
    assert workspace.holding_search.verification_repeats == 2
    assert workspace.coordinated_holding.total_assist_cap_nm == 1
    assert workspace.coordinated_holding.bias_gain_nm_per_rad_sec == 40
    assert workspace.coordinated_holding.supported_soft_recovery
    assert workspace.joint2_drive_kd == workspace.joint3_drive_kd == (.4, 3., 3., 1.2, .2, .1)
    assert workspace.coordinated_holding.recovery_kd_nm_s_per_rad == 0
    assert workspace.coordinated_holding.motor_local_recovery
    assert controller.config.max_sweep_leg_sec == 180
    assert controller.pose.motion.max_control_gap_sec == .04
    assert not WorkspaceCalibrationConfig().coordinated_holding


@pytest.mark.parametrize("value", [0, -.001, float("nan"), True, .0046])
def test_explicit_holding_entry_tolerance_must_fit_the_measured_point(tmp_path, value):
    with pytest.raises(ValueError):
        replay(tmp_path, holding_arrival_tolerance_rad=value,
               holding_search=HoldingSearchConfig(movement_rad=.0045))
    with pytest.raises(ValueError, match="explicit coordinated holding"):
        WorkspaceCalibrationConfig(holding_arrival_tolerance_rad=.001)


@pytest.mark.parametrize("axis", [1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("expanded", [False, True])
def test_nominal_entry_margin_freezes_measured_pose_without_relaxing_fine_hold(tmp_path, axis, direction, expanded):
    r = replay(tmp_path, holding_search=HoldingSearchConfig(
        movement_rad=.0045, observation_sec=.5, observation_timeout_sec=2),
        holding_arrival_tolerance_rad=.003375 if expanded else None)
    r.controller.submit("arm", now_sec=r.now)
    workspace = r.controller._coordinated
    for _ in range(2500):
        if workspace._phase == "sweep":
            break
        scripted_coordinated_tick(r)
    else:
        pytest.fail("no first sweep")
    nominal = workspace.pose.goal
    target = list(r.q)
    target[1:3] = nominal[1:3]
    target[axis] += direction*.00251055863172895
    for _ in range(200):
        q = list(r.q)
        for i in (1, 2):
            q[i] += max(-.00015, min(.00015, target[i]-q[i]))
        r.tick(q, dt_sec=.025)
        if workspace._probe is not None:
            break
    if not expanded:
        assert workspace._probe is None
        assert workspace._sample_pose is None
        return
    assert workspace._sample_pose == tuple(target)
    assert workspace._nominal_target == nominal
    assert workspace.pose.goal[1:3] == tuple(target[1:3])
    assert workspace._arrival_status()["tolerance_rad"] == .003375
    search = workspace._search
    for _ in range(120):
        r.tick(dt_sec=.025)
        if search.observations:
            break
    assert search.observations[0]["outcome"] == "stationary"
    assert search.observations[0]["stationary_dwell_sec"] >= .5
    assert search.observations[0]["allowed_stationary_span_rad"] == 2*workspace.pose.motion.position_resolution_rad


@pytest.mark.parametrize("gate,limit,reason", [
    ("_leg_exposure", 120, "leg_command_exposure_budget"),
    ("_sweep_exposure", 480, "active_command_exposure_budget"),
    ("_exposure", 1040, "workspace_command_exposure_budget"),
])
def test_expanded_profile_budgets_count_support_and_still_stop_without_renewal(tmp_path, gate, limit, reason):
    r = replay(tmp_path, max_sweep_leg_sec=180, max_sweep_leg_command_exposure_nm_sec=120,
        max_sweep_command_exposure_nm_sec=480, max_batch_sec=1200,
        max_command_exposure_nm_sec=1040)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    setattr(workspace, gate, limit*.9)
    r.tick(dt_sec=.025)
    assert workspace._phase == "sweep"
    assert getattr(workspace, gate) >= limit*.9
    # Pending support work and the last worst-case dispatch reserve count too.
    workspace._last_delta = (0, .3, -.2, 0, 0, 0)
    setattr(workspace, gate, limit-.02)
    r.tick(dt_sec=.025)
    assert workspace._phase == "stopping"
    assert workspace._reason == reason
    assert getattr(workspace, gate) >= limit-.02


@pytest.mark.parametrize("expanded", [False, True])
def test_native_probe_soft_brake_uses_the_scoped_speed_margin(tmp_path, expanded):
    changes = {"holding_search": HoldingSearchConfig(
        observation_sec=1.5, observation_timeout_sec=4.5, movement_rad=.0045)}
    if expanded:
        changes["motion_soft_velocity_rad_s"] = .07605
    r = replay(tmp_path, **changes)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    for _ in range(60):
        r.tick(dt_sec=.025)
        if workspace._observation and workspace._observation["started_at_sec"] is not None:
            break
    else:
        pytest.fail("no causal probe plateau")
    j = int(workspace._tasks[workspace._index]["joint"][-1])-1
    direction = 1 if workspace._probe >= 0 else -1
    for _ in range(2):
        q = list(r.q)
        q[j] += direction*.0016  # 0.064 rad/s, between the old/new soft ceilings.
        r.tick(q, dt_sec=.025)
    assert workspace.pose.braking is (not expanded)
    assert not r.audit["fault"]


@pytest.mark.parametrize("gate,field,old_limit,new_limit,reason", [
    ("leg_time", "_leg_started", 114.075, 180, "leg_timeout"),
    ("leg_exposure", "_leg_exposure", 46.8, 120, "leg_command_exposure_budget"),
    ("sweep_time", "_sweep_started", 760.5, 1200, "batch_timeout"),
    ("sweep_exposure", "_sweep_exposure", 156, 480, "active_command_exposure_budget"),
    ("workspace_time", "_started", 2400, 3120, "workspace_timeout"),
    ("workspace_exposure", "_exposure", 800, 1040, "workspace_command_exposure_budget"),
])
def test_current_margin_budgets_admit_the_old_edge_but_stop_at_new_finite_edge(
    tmp_path, gate, field, old_limit, new_limit, reason,
):
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coordinated_holding_v1.json"
    r = replay(tmp_path, **json.loads(profile.read_text())["workspace"])
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    for value in (old_limit+.1, new_limit+.1):
        setattr(workspace, field, r.now-value if gate.endswith("_time") else value)
        r.tick(dt_sec=.025)
        if value < new_limit:
            assert workspace._phase == "sweep"
        else:
            assert workspace._phase == "stopping"
            assert workspace._reason == reason


@pytest.mark.parametrize("values", [
    {"motion_soft_velocity_rad_s": .2}, {"motion_soft_velocity_rad_s": .02},
    {"motion_soft_velocity_rad_s": float("nan")}, {"motion_braking_timeout_sec": -1},
    {"prepare_timeout_sec": True}, {"max_batch_sec": float("inf")},
    {"max_sweep_command_exposure_nm_sec": 1},
])
def test_scoped_margin_overrides_cannot_bypass_native_config_validation(tmp_path, values):
    with pytest.raises((ValueError, TypeError)):
        replay(tmp_path, **values)


@pytest.mark.parametrize("enabled", [False, True])
def test_positioning_support_brake_uses_explicit_stage_recovery_only(tmp_path, enabled):
    r = replay(tmp_path, positioning_support_soft_speed_recovery=enabled)
    r.controller.submit("arm", now_sec=r.now)
    r.tick(dt_sec=.025)
    workspace = r.controller._coordinated
    assert all(t.config.support_soft_speed_recovery is enabled for t in workspace.pose.trackers)
    assert workspace.pose.motion.support_soft_speed_recovery is False
    # A selected-axis soft stop, then a passive-axis soft stop, as observed
    # in mc44tra7. Both remain below the original raw hard-speed ceiling.
    for axis in [2, 2, 2, 3, 3, 3]:
        q = list(r.q)
        q[axis] += .002
        r.tick(q, dt_sec=.025)
    resumed = False
    for _ in range(100):
        r.tick(dt_sec=.025)
        frames = r.audit.get("coordinated_holding", {}).get("axes", [])
        resumed |= any(f["reason"] == "speed_governor_resumed_original_target" for f in frames)
        if r.audit["fault"] or resumed:
            break
    if enabled:
        assert resumed
        assert workspace._phase == "position"
        assert workspace._index == 0
        assert workspace.pose.goal[2] == pytest.approx(.05235987755982989)
    else:
        assert not resumed
        assert r.audit["fault"] == "coordinated_target_stopped_before_goal"


def test_stage_recovery_switch_requires_consumed_quiet_handoff(settings):
    core = runtime(settings)
    warm(core)
    core.set_goal(core.goal, support_soft_speed_recovery=True)
    assert all(t.config.support_soft_speed_recovery for t in core.trackers)
    for value in [0, 1, "true"]:
        with pytest.raises(TypeError, match="boolean"):
            core.set_goal(core.goal, support_soft_speed_recovery=value)
    dt = core.observe(1.12, ZERO, ZERO)
    command, _ = core.step(sequence=12, dt=dt, permitted=True)
    with pytest.raises(ValueError, match="consumed"):
        core.set_goal(core.goal)
    core.record_dispatch(12, command)
    core.raw = (0, .03, 0, 0, 0, 0)
    core.velocity = core.raw
    with pytest.raises(ValueError, match="stillness"):
        core.set_goal(core.goal)


def test_positioning_support_resume_waits_for_complete_zero_tail_receipt(tmp_path):
    r = replay(tmp_path)
    r.controller.submit("arm", now_sec=r.now)
    r.tick(dt_sec=.025)
    for axis in [2, 2, 2, 3, 3, 3]:
        q = list(r.q)
        q[axis] += .002
        r.tick(q, dt_sec=.025)
    saw_wait = False
    for _ in range(30):
        r.tick(dt_sec=.025, final_delta=(0, .01, 0, 0, 0, 0))
        frames = r.audit.get("coordinated_holding", {}).get("axes", [])
        assert not r.audit["fault"]
        assert all(f["reason"] != "speed_governor_resumed_original_target" for f in frames)
        saw_wait |= any(f["speed_governor_resume_block_reason"] == "native_zero_dispatch_not_confirmed" for f in frames)
    assert saw_wait
    for _ in range(30):
        r.tick(dt_sec=.025)
        if any(f["reason"] == "speed_governor_resumed_original_target" for f in r.audit["coordinated_holding"]["axes"]):
            break
    else:
        pytest.fail("admitted zero tail and fresh stillness did not resume the original goal")


@pytest.mark.parametrize("passive_axis", [0, 3, 4, 5])
def test_passive_context_change_censors_probe_and_never_reanchors_same_point(tmp_path, passive_axis):
    r = replay(tmp_path)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    anchor = workspace._sample_pose
    search = workspace._search
    for _ in range(200):
        if search.observations and workspace._probe is not None:
            break
        r.tick(dt_sec=.025)
    before = len(search.observations)
    assert before > 0
    retained = list(search.observations)
    for _ in range(40):
        q = list(r.q)
        q[passive_axis] += .00015
        r.tick(q, dt_sec=.025)
        if workspace._phase == "stopping":
            break
    assert workspace._phase == "stopping"
    assert workspace._reason == "holding_search_incomplete"
    assert workspace._sample_pose == anchor
    assert len(search.observations) == before
    assert search.observations == retained
    assert any(row.get("outcome") == "censored_support_pose_changed" for row in r.records)
    for _ in range(100):
        r.tick(dt_sec=.025)
        if workspace._phase == "failed":
            break
    assert workspace._phase == "failed"
    assert workspace._search is search
    assert r.delta == r.last_kd == ZERO
    assert not r.controller.cycle_disable_confirmation_ready


def test_controlled_support_reacquires_same_frozen_sample_without_global_withdrawal(tmp_path):
    r = replay(tmp_path)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    anchor = workspace._sample_pose
    search = workspace._search
    measured = int(workspace._tasks[workspace._index]["joint"][-1])-1
    support = 3-measured
    for _ in range(20):
        q = list(r.q)
        q[support] += .00015
        r.tick(q, dt_sec=.025)
        if workspace._probe is None:
            break
    assert workspace._phase == "sweep"
    assert workspace._probe is None
    assert workspace._sample_pose == anchor
    for _ in range(200):
        scripted_coordinated_tick(r)
        if workspace._probe is not None:
            break
    assert workspace._probe is not None
    assert workspace._search is search
    assert workspace._sample_pose == anchor
    assert workspace._phase == "sweep"


@pytest.mark.parametrize("supported,software_kd", [(False, 6.), (True, 6.), (True, 0.)])
def test_position_arrival_still_requires_six_axis_stationarity(tmp_path, supported, software_kd):
    r = replay(tmp_path, coordinated_holding=CoordinatedHoldingConfig(
        supported_soft_recovery=supported, recovery_kd_nm_s_per_rad=software_kd))
    for _ in range(12):
        q = list(r.q)
        q[3] += .0002
        r.tick(q, dt_sec=.025)
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(25):
        q = list(r.q)
        q[3] += .0002
        r.tick(q, dt_sec=.025)
    assert r.controller.result_record()["state"] == "preparing"
    assert not r.controller._coordinated.pose.quiet()
    assert r.delta == ZERO
    assert r.controller._coordinated.pose.goal is None
    started = r.controller._coordinated._started
    for _ in range(100):
        r.tick(dt_sec=.025)
        if r.controller.result_record()["state"] == "position":
            break
    assert r.controller.result_record()["state"] == "position"
    assert r.controller._coordinated._started == started


def test_static_pose_dwell_does_not_hide_slow_passive_drift(tmp_path):
    r = replay(tmp_path, gravity_observation_sec=2)
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(2500):
        scripted_coordinated_tick(r)
        if r.controller.result_record()["state"] == "observe":
            break
    workspace = r.controller._coordinated
    assert workspace._phase == "observe"
    started = workspace._leg_started
    # Below the quiet-window velocity gate, but beyond the full dwell's
    # original position band. It must not become a static pose observation.
    for _ in range(60):
        q = list(r.q)
        q[3] += .00005
        r.tick(q, dt_sec=.025)
        if any(row["kind"] == "workspace_pose_observation_rejected" for row in r.records):
            break
    assert workspace.pose.quiet()
    assert not workspace._pose_observations
    assert any(row["kind"] == "workspace_pose_observation_rejected" for row in r.records)
    assert workspace._leg_started == started
    assert workspace._phase == "observe"


def advance_to_pose_observation(r):
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(2500):
        scripted_coordinated_tick(r)
        if r.controller.result_record()["state"] == "observe":
            return
    pytest.fail("scripted arrival did not reach pose observation")


@pytest.mark.parametrize("axis", range(6))
@pytest.mark.parametrize("phase", ["observe", "return_observe"])
def test_pose_observation_accepts_bounded_two_count_jitter_without_short_gate_reset(tmp_path, axis, phase):
    r = replay(tmp_path, gravity_observation_sec=1)
    advance_to_pose_observation(r)
    workspace = r.controller._coordinated
    workspace._phase = phase
    anchor, started = r.q, workspace._leg_started
    saw_not_quiet = False
    for n in range(160):
        q = list(anchor)
        q[axis] += (1 if (n // 20) % 2 else -1) * .000383496
        r.tick(q, dt_sec=.01)
        saw_not_quiet |= not workspace.pose.quiet()
        if workspace._pose_observations:
            break
        assert workspace._phase == phase
    assert saw_not_quiet
    assert len(workspace._pose_observations) == 1
    obs = workspace._pose_observations[0]
    assert obs["completed_at_sec"]-obs["started_at_sec"] >= 1
    assert obs["completed_at_sec"]-started < 1.6
    assert obs["completion_criterion"] == "bounded_six_axis_position_window"
    assert obs["handoff_quiet_confirmed"]
    assert obs["allowed_position_span_rad"] == workspace.workspace.holding_search.movement_rad
    assert max(b-a for a,b in zip(obs["minimum_rad"], obs["maximum_rad"])) == pytest.approx(.000766992)
    assert all(abs(obs["qpos_rad"][i]-obs["nominal_target_rad"][i]) <= .0015 for i in (1, 2))


@pytest.mark.parametrize("kind", ["human", "withdraw", "lease", "hard_speed", "watchdog", "receipt"])
def test_bounded_pose_observation_keeps_authority_clock_speed_and_receipt_guards(tmp_path, kind):
    r = replay(tmp_path, gravity_observation_sec=1)
    advance_to_pose_observation(r)
    workspace = r.controller._coordinated
    if kind == "human":
        r.controller.set_human_intervention(True)
    elif kind == "withdraw":
        r.controller.submit("withdraw", now_sec=r.now)
    elif kind == "lease":
        workspace._permit = r.now-r.controller.limits.command_timeout_sec-1
    q = list(r.q)
    if kind == "hard_speed":
        q[1] += r.controller.limits.max_joint_velocity_rad_s*.011
    if kind in {"watchdog", "receipt"}:
        with pytest.raises(ValueError):
            r.tick(q, dt_sec=.05 if kind == "watchdog" else .01,
                receipt_updates={"sent_joint_names": ["joint2"]} if kind == "receipt" else None)
    else:
        r.tick(q, permit=kind != "lease")
        assert r.delta == ZERO
    assert not workspace._pose_observations
    assert workspace._phase in {"stopping", "failed"}


def test_pose_observation_outside_target_band_does_not_complete(tmp_path):
    r = replay(tmp_path, gravity_observation_sec=.2)
    advance_to_pose_observation(r)
    workspace = r.controller._coordinated
    q = list(r.q)
    for _ in range(15):
        q[1] += .0001
        r.tick(q)
    for _ in range(35):
        r.tick(q)
    assert not workspace._pose_observations
    assert workspace._quiet_since is None


def test_pose_observation_unacknowledged_dwell_does_not_complete(tmp_path):
    r = replay(tmp_path, gravity_observation_sec=.2)
    advance_to_pose_observation(r)
    workspace = r.controller._coordinated
    for _ in range(18):
        r.tick()
    assert workspace._quiet_since is not None
    # Crossing the dwell boundary in computation alone cannot commit a sample.
    r.tick(dt_sec=.025, ack=False)
    assert not workspace._pose_observations
    assert workspace._pending is not None


def test_pose_observation_retains_finished_window_while_waiting_for_quiet_handoff(tmp_path):
    r = replay(tmp_path, gravity_observation_sec=.3)
    advance_to_pose_observation(r)
    workspace = r.controller._coordinated
    anchor = r.q
    for n in range(50):
        q = list(anchor)
        q[5] += (1 if n % 2 else -1) * .000383496
        r.tick(q)
    status = workspace._static_observation_status()
    assert status["reason"] == "awaiting_handoff_stillness"
    assert status["elapsed_sec"] >= .3
    assert not workspace._pose_observations
    since = workspace._quiet_since
    for _ in range(20):
        r.tick()
        if workspace._pose_observations:
            break
    assert len(workspace._pose_observations) == 1
    assert workspace._pose_observations[0]["started_at_sec"] == since
    assert workspace._pose_observations[0]["handoff_quiet_confirmed"]


def test_passive_offset_still_obeys_original_support_excursion_guard(tmp_path):
    r = replay(tmp_path)
    advance_to_coordinated_probe(r)
    workspace = r.controller._coordinated
    q = list(r.q)
    q[3] = workspace._origin[3]+r.controller.limits.max_support_drift_rad+.001
    r.tick(q, dt_sec=.025)
    assert r.audit["fault"] == "workspace_support_excursion"
    assert r.delta == ZERO


def test_coupled_partial_batch_holds_other_axis_handoffs_and_final_withdrawal(tmp_path, monkeypatch):
    original = WorkspaceCalibrationConfig.sweep_plan
    def one_leg(self, **kwargs):
        plan = original(self, **kwargs)
        plan["steps"] = plan["steps"][:1]
        return plan
    monkeypatch.setattr(WorkspaceCalibrationConfig, "sweep_plan", one_leg)
    r = replay(tmp_path, initial_q=(0, .01, .01, 0, 0, 0))
    r.controller.submit("arm", now_sec=r.now)
    nonzero_support = False
    for _ in range(20000):
        state = r.controller.result_record()["state"]
        if state in {"complete", "failed", "cancelled"}:
            break
        coupled_tick(r)
        nonzero_support |= abs(r.delta[1]) > .01 and abs(r.delta[2]) > .01
    result = r.controller.result_record()
    assert result["state"] == "complete", (result["reason"], r.controller._coordinated._search.status(detailed=True) if r.controller._coordinated._search else None, r.q)
    assert nonzero_support
    assert len(result["workspace"]["acquisitions"]) == 2
    assert len(result["motion"]["completed_legs"]) == 2
    assert all(not leg["final_zero_and_still_confirmed"] and leg["handoff_full_dispatch_confirmed"]
               for leg in result["motion"]["completed_legs"])
    assert r.delta == r.last_kd == ZERO
    assert r.controller.cycle_disable_confirmation_ready


@pytest.mark.parametrize("period,passive_offset,coverage", [
    (.01, 0, False), (.01, .006519419, False),
    (.025, 0, False), (.025, .006519419, False), (.01, 0, True),
])
def test_full_coupled_batch_and_report_keep_handoffs_distinct_from_withdrawal(tmp_path, period, passive_offset, coverage, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from control_station.b601_torque_console import transparency
    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )
    changes = {}
    if coverage:
        changes.update(holding_sampling_plan="coverage", coordinated_holding=CoordinatedHoldingConfig(
            bias_gain_nm_per_rad_sec=40, supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0),
            joint2_drive_kd=(.4, 3., 3., 1.2, .2, .1), joint3_drive_kd=(.4, 3., 3., 1.2, .2, .1))
    if period == .025 and passive_offset:
        # Exercise the actual explicit profile at the observed cadence too,
        # rather than validating the new margins only with shortened dwells.
        profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coordinated_holding_v1.json"
        values = json.loads(profile.read_text())["workspace"]
        changes["holding_search"] = HoldingSearchConfig(**values["holding_search"])
        changes["holding_arrival_tolerance_rad"] = values["holding_arrival_tolerance_rad"]
        changes.update({name: values[name] for name in COORDINATED_CALIBRATION_OVERRIDES})
    r = replay(tmp_path, initial_q=(0, .01, .01, 0, 0, 0), **changes)
    movement = r.controller._coordinated.workspace.holding_search.movement_rad
    monkeypatch.setattr(transparency, "time", SimpleNamespace(monotonic=lambda: r.now+.002))
    journal = transparency.TransparencyAssessment(tmp_path / "b601_owner.transparency.jsonl", automatic=True)
    def snapshot():
        return {**r.controller.status(), "feedback_age_sec": 0.0}
    journal.record({"action": "context", "human_intervention": False}, snapshot())
    assert snapshot()["calibration_trial"] is None
    r.controller.submit("arm", now_sec=r.now)
    handoffs = []
    for _ in range(50000):
        state = r.controller.status()["automatic_calibration"]["state"]
        if state in {"complete", "failed", "cancelled"}:
            break
        old = len(r.controller._coordinated._acquisitions)
        coupled_tick(r, period=period, passive_offset=passive_offset)
        controller = r.controller._coordinated
        flags = [tracker.config.support_soft_speed_recovery for tracker in controller.pose.trackers]
        if controller._phase in {"position", "return"}:
            assert flags == [coverage or controller.workspace.positioning_support_soft_speed_recovery]*2
        elif controller._phase == "sweep":
            assert flags == [coverage or controller.pose.motion.support_soft_speed_recovery]*2
        if r.audit.get("coordinated_holding") is not None:
            # Audit reports the phase that produced this frame, before a
            # consumed receipt may advance the outer workspace lifecycle.
            audit = r.audit["coordinated_holding"]
            assert audit["point_search_enabled"] == (not coverage and r.audit["automatic_calibration"]["workspace"]["phase"] == "sweep")
        journal.observe(snapshot())
        assert sum(abs(v) for v in r.delta) <= 1+1e-9
        if old != len(r.controller._coordinated._acquisitions):
            assert tuple(r.delta) == tuple(r.audit["coordinated_holding"]["combined_command_nm"])
            handoffs.append(tuple(r.delta))
    result = r.controller.result_record()
    assert result["state"] == "complete", (result["reason"], r.q, r.controller._coordinated._holding_status())
    legs = 8 if coverage else 24
    assert len(result["motion"]["completed_legs"]) == legs
    assert {leg["max_reference_velocity_rad_s"] for leg in result["motion"]["completed_legs"]} == (
        {max(r.controller.config.sweep_speeds_rad_s)} if coverage else set(r.controller.config.sweep_speeds_rad_s))
    # A learned small hold is valid too. Handoff must preserve the actual
    # dispatched composite, not manufacture a minimum support magnitude.
    assert len(handoffs) == 2 and all(any(abs(v) > 1e-9 for v in h) for h in handoffs)
    assert r.delta == r.last_kd == ZERO
    assert r.controller.cycle_disable_confirmation_ready
    assert all(abs(r.q[i]-.01) < .003 for i in (1, 2))
    journal.close()
    (tmp_path / "b601_owner.debug.jsonl").write_text("\n".join(json.dumps(x) for x in [*r.records, result])+"\n")
    report = build_automatic_report(tmp_path)
    assert report["holding_calibration"]["verified_points"] == legs
    assert all(not p["withdrawal_confirmed"] for p in report["holding_calibration"]["points"])
    assert report["workspace"]["gravity_observations"] == []
    assert len(report["workspace"]["coordinated_pose_observations"]) == 2
    for observation in report["workspace"]["coordinated_pose_observations"]:
        assert max(b-a for a,b in zip(observation["minimum_rad"], observation["maximum_rad"])) <= movement
        assert observation["qpos_rad"][3] == pytest.approx(passive_offset)
    assert report["cycle"]["zero_assist_base_damping_confirmed"]
    assert (tmp_path / "b601_owner.transparency.jsonl").stat().st_size > 0
    for point in report["holding_calibration"]["points"]:
        assert point["sample_pose_rad"][3] == pytest.approx(passive_offset)
        for obs in point["observations"]:
            if obs["outcome"] != "stationary":
                continue
            assert obs["started_at_sec"] > obs["first_causal_receipt_at_sec"]
            assert max(b-a for a,b in zip(obs["minimum_rad"], obs["maximum_rad"])) <= movement
            assert sum(abs(v) for v in obs["actual_delta_nm"]) <= 1+1e-9


def test_authority_revoked_between_computation_and_dispatch_vetoes_old_nonzero_frame(tmp_path):
    r = replay(tmp_path)
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(80):
        r.tick()
    r.tick(ack=False)
    assert any(r.delta)
    r.controller.set_human_intervention(True)
    with pytest.raises(ValueError, match="authority_changed"):
        r.controller.validate_dispatch_start(feedback_sequence=r.sequence,
            feedback_timestamp_sec=r.now, now_sec=r.now+.0003)
    assert r.controller.result_record()["pending_dispatch"]["receipt_confirmed"] is False


@pytest.mark.parametrize("values", [
    {"total_assist_cap_nm": float("nan")}, {"bias_cap_nm": 2},
    {"rise_slew_nm_s": True}, {"bias_gain_nm_per_rad_sec": 0},
])
def test_coordinated_parameters_are_finite_and_explicit(values):
    with pytest.raises(ValueError):
        CoordinatedHoldingConfig(**values)

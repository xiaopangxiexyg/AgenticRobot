"""Explicit acquisition velocity feedback; no devices or hardware evidence."""

import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from control_station.b601_torque_console.automatic_calibration import (
    load_automatic_profile,
)
from control_station.b601_torque_console.workspace_calibration import (
    WorkspaceCalibrationConfig,
)
from robot_motion.rebot_b601_rs.coordinated_holding import (
    AXES,
    ZERO,
    CoordinatedHoldingConfig,
)
from robot_motion.rebot_b601_rs.holding_calibration import HoldingSearchConfig
from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionFrame
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

pytestmark = [pytest.mark.unit, pytest.mark.contract]
PROFILES = Path(__file__).resolve().parents[1]/"docs/hardware/profiles"
BASE = PROFILES/"b601_j2_j3_continuous_identification_v1.json"
CANDIDATE = PROFILES/"b601_j2_j3_cruise_identification_v1.json"
FOUR_CENTRES = PROFILES/"b601_j2_j3_four_centres_cruise_identification_v1.json"
RESOLUTION = .0003834962844848633


def controller_from_profile(tmp_path, *, candidate=True, profile=None, extra=()):
    args = owner._parser().parse_args([
        "--allow-motion", "--rate-hz", "100", "--preflight-only",
        "--torque-console-auto-calibration-profile", str(profile or (CANDIDATE if candidate else BASE)),
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE", "--torque-console-joint", "joint3",
        "--torque-console-max-assist-nm", "1", "--torque-console-max-joint-velocity-rad-s", ".195",
        "--torque-console-max-excursion-rad", ".15", "--torque-console-staged-target-offset-limit-rad", ".09",
        "--torque-console-staged-breakaway-torque-nm", "1", "--torque-console-staged-breakaway-rise-slew-nm-s", "1",
        "--torque-console-staged-breakaway-duration-sec", "3", "--torque-console-staged-breakaway-impulse-nm-sec", "1.5",
        "--torque-console-staged-tracking-overlay-cap-nm", "1", *extra])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_cli(parsed)
    return probe._automatic_calibration_controller_from_args(parsed)


def test_profile_changes_only_explicit_velocity_feedback_and_keeps_existing_envelopes(tmp_path):
    old, new = load_automatic_profile(BASE), load_automatic_profile(CANDIDATE)
    gains = new["workspace"]["coordinated_holding"]
    assert CoordinatedHoldingConfig().cruise_enabled is False
    assert not old["workspace"]["coordinated_holding"].get("cruise_velocity_ki_nm_per_rad")
    for key in ("cruise_velocity_kp_nm_s_per_rad", "cruise_velocity_ki_nm_per_rad", "cruise_velocity_window_sec"):
        gains.pop(key)
    assert new["workspace"].pop("identification_position_bin_width_rad") == .0045
    assert new == old
    core = controller_from_profile(tmp_path)._workspace._coordinated.pose
    assert core.config.cruise_enabled
    assert core.config.total_assist_cap_nm == 1
    assert core.motion.max_control_gap_sec == .04
    assert core.settings.limits.max_joint_velocity_rad_s == .195
    assert controller_from_profile(tmp_path, candidate=False)._workspace._coordinated.pose.cruise_estimators == ()


def test_four_centre_profile_changes_only_centres_and_explicit_arrival_feedback():
    old, new = load_automatic_profile(CANDIDATE), load_automatic_profile(FOUR_CENTRES)
    for name in ("joint2_centres_deg", "joint3_centres_deg"):
        assert new["workspace"][name] == [3, 6]
        new["workspace"][name] = old["workspace"][name]
    assert new["workspace"]["coordinated_holding"].pop("cruise_arrival_feedback") is True
    assert new == old


@pytest.mark.parametrize("changes", [
    {"cruise_velocity_kp_nm_s_per_rad": -1}, {"cruise_velocity_ki_nm_per_rad": float("nan")},
    {"cruise_velocity_window_sec": 0}, {"cruise_velocity_kp_nm_s_per_rad": True},
])
def test_invalid_cruise_gains_fail_before_backend_construction(changes):
    with pytest.raises(ValueError):
        CoordinatedHoldingConfig(**changes)


def test_cruise_is_bound_to_motor_local_continuous_identification():
    with pytest.raises(ValueError, match="motor-local"):
        CoordinatedHoldingConfig(cruise_velocity_kp_nm_s_per_rad=6)
    with pytest.raises(ValueError, match="continuous identification"):
        WorkspaceCalibrationConfig(coordinated_holding=CoordinatedHoldingConfig(
            supported_soft_recovery=True, recovery_kd_nm_s_per_rad=0, cruise_velocity_kp_nm_s_per_rad=6),
            holding_search=HoldingSearchConfig())


def warm_core(core, *, q=ZERO, target=.02):
    for n in range(30):
        dt = core.observe(1+n*.01, q, ZERO)
        command, _ = core.step(sequence=n, dt=dt, permitted=False)
        core.record_dispatch(n, command)
    goal = list(q)
    goal[2] += target
    core.set_goal(goal)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("matching_receipt", [False, True])
def test_resolved_overspeed_reduces_bias_only_after_matching_receipt(tmp_path, monkeypatch, direction, matching_receipt):
    core = controller_from_profile(tmp_path)._workspace._coordinated.pose
    warm_core(core, target=direction*.02)
    core.adopt_hold(1, .2)
    core.adopt_hold(2, direction*.3)
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, now, i=i, **kwargs):
            speed = direction*.0075 if i == 2 else 0
            return B601JointMotionFrame(phase="track", reason="bounded_reference_tracking",
                q_reference_rad=speed*(now-1.29), qvel_reference_rad_s=speed,
                control_qvel_rad_s=0, position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0)
        monkeypatch.setattr(tracker, "update", frame)
    saw_active = False
    for n in range(1, 31):
        q = (0, 0, direction*n*.0002, 0, 0, 0)
        dt = core.observe(1.29+n*.01, q, (0, 0, direction*.02, 0, 0, 0))
        before = core.bias
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        status = audit["cruise_velocity_control"]
        if status["active"]:
            saw_active = True
            assert status["error_rad_s"]*direction < 0
            assert status["error_rad_s"] == pytest.approx(-direction*.0125)
            assert status["proportional_request_nm"]*direction < 0
            assert abs(audit["proposed_bias_nm"][2]) < abs(before[2])
            assert core.bias == before
        actual = command if matching_receipt or not status["active"] else tuple(v*.99 for v in command)
        core.record_dispatch(30+n, actual)
        assert core.bias[1] == .2
        if not matching_receipt:
            assert core.bias == before
        assert sum(abs(v) for v in command) <= 1+1e-12
    assert saw_active
    command, audit = core.step(sequence=1000, dt=.01, permitted=False, point_search=False, cruise_joint=2)
    core.record_dispatch(1000, command)
    assert core.bias == command == ZERO
    assert not audit["cruise_velocity_control"]["active"]


def test_cruise_integral_does_not_accumulate_unrealized_cap_or_slew_growth(tmp_path, monkeypatch):
    core = controller_from_profile(tmp_path)._workspace._coordinated.pose
    warm_core(core)
    core.adopt_hold(1, .3)
    core.adopt_hold(2, .7)
    core.last_delta = core.bias
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, now, i=i, **kwargs):
            speed = .015 if i == 2 else 0
            return B601JointMotionFrame(phase="track", reason="bounded_reference_tracking",
                q_reference_rad=speed*(now-1.29), qvel_reference_rad_s=speed,
                control_qvel_rad_s=0, position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0)
        monkeypatch.setattr(tracker, "update", frame)
    saw_active = False
    for n in range(1, 40):
        before = core.bias
        dt = core.observe(1.29+n*.01, (0, 0, n*.00005, 0, 0, 0), (0, 0, .005, 0, 0, 0))
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        if audit["cruise_velocity_control"]["active"]:
            saw_active = True
            assert audit["cruise_velocity_control"]["requested_bias_increment_nm"] > 0
            assert audit["proposed_bias_nm"][2] <= before[2]+1e-12
        core.record_dispatch(30+n, command)
        assert sum(abs(v) for v in command) <= 1+1e-12
    assert saw_active


def test_matched_accelerating_reference_never_accumulates_filtered_velocity_lag(tmp_path, monkeypatch):
    core = controller_from_profile(tmp_path)._workspace._coordinated.pose
    warm_core(core)
    core.adopt_hold(2, .1)
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, now, q, i=i, **kwargs):
            speed = .005+.03*(now-1.29) if i == 2 else 0
            return B601JointMotionFrame(phase="track", reason="bounded_reference_tracking",
                q_reference_rad=q, qvel_reference_rad_s=speed, control_qvel_rad_s=0,
                position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0)
        monkeypatch.setattr(tracker, "update", frame)
    active = False
    for n in range(1, 41):
        t = n*.01
        q = (0, 0, .005*t+.015*t*t, 0, 0, 0)
        dt = core.observe(1.29+t, q, (0, 0, .005+.03*t, 0, 0, 0))
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        status = audit["cruise_velocity_control"]
        if status["active"]:
            active = True
            assert status["error_rad_s"] == 0
            assert status["proportional_request_nm"] == 0
            assert status["requested_bias_increment_nm"] == 0
        core.record_dispatch(30+n, command)
        assert core.bias[2] == .1
    assert active


@pytest.mark.parametrize("arrival", [False, True])
def test_native_brake_resets_cruise_and_retains_raw_speed_and_watchdog_guards(tmp_path, monkeypatch, arrival):
    core = controller_from_profile(tmp_path)._workspace._coordinated.pose
    core.config = replace(core.config, cruise_arrival_feedback=arrival)
    warm_core(core)
    core.adopt_hold(2, .2)
    braking = False
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, now, i=i, **kwargs):
            return B601JointMotionFrame(
                phase="brake" if braking and i == 2 else "track",
                reason="soft_velocity_braking" if braking and i == 2 else "bounded_reference_tracking",
                q_reference_rad=.0075*(now-1.29) if i == 2 else 0,
                qvel_reference_rad_s=.0075 if i == 2 else 0,
                control_qvel_rad_s=0, position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0,
                speed_governor_active=braking and i == 2)
        monkeypatch.setattr(tracker, "update", frame)
    active = False
    for n in range(1, 40):
        dt = core.observe(1.29+n*.01, (0, 0, n*.0001, 0, 0, 0), (0, 0, .01, 0, 0, 0))
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        active |= audit["cruise_velocity_control"]["active"]
        core.record_dispatch(30+n, command)
    assert active
    braking = True
    dt = core.observe(1.69, core.q, ZERO)
    command, audit = core.step(sequence=100, dt=dt, permitted=True, point_search=False, cruise_joint=2)
    assert audit["braking"]
    assert not audit["cruise_velocity_control"]["active"]
    assert audit["cruise_velocity_control"]["proportional_request_nm"] == 0
    core.record_dispatch(100, command)
    assert not any(e.ready for e in core.cruise_estimators)
    with pytest.raises(ValueError, match="watchdog gap"):
        core.observe(1.74, core.q, ZERO)
    dt = core.observe(1.70, core.q, (0, 0, .196, 0, 0, 0))
    with pytest.raises(ValueError, match="raw joint velocity exceeded"):
        core.step(sequence=101, dt=dt, permitted=True, point_search=False, cruise_joint=2)


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_velocity_feedback_preserves_support_and_ignores_endpoint_reference_jump(tmp_path, monkeypatch, direction):
    core = controller_from_profile(tmp_path)._workspace._coordinated.pose
    core.config = replace(core.config, cruise_arrival_feedback=True)
    # Stay inside the accepted position band to isolate velocity damping from
    # the separate, bounded stopped-short restoration term.
    start = .02-direction*.002
    warm_core(core, q=(0, 0, start, 0, 0, 0), target=direction*.002)
    core.adopt_hold(2, .3)
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, i=i, **kwargs):
            return B601JointMotionFrame(phase="brake", reason="target_reached_braking",
                q_reference_rad=.02 if i == 2 else 0, qvel_reference_rad_s=0,
                control_qvel_rad_s=0, position_torque_nm=0, velocity_torque_nm=0, requested_torque_nm=0)
        monkeypatch.setattr(tracker, "update", frame)
    active = False
    for n in range(1, 41):
        dt = core.observe(1.29+n*.01, (0, 0, start+direction*n*.0001, 0, 0, 0), (0, 0, direction*.01, 0, 0, 0))
        before = core.bias
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        status = audit["cruise_velocity_control"]
        if status["active"]:
            active = True
            assert status["arrival"] and status["error_rad_s"] == pytest.approx(-direction*.01)
            assert status["aligned_reference_velocity_rad_s"] == 0
            assert status["proportional_request_nm"] == pytest.approx(-direction*.06)
            assert (audit["proposed_bias_nm"][2]-before[2])*direction <= 0
        assert core.bias == before
        core.record_dispatch(30+n, command)
        assert sum(abs(v) for v in command) <= 1+1e-12
    assert active and not core.braking


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("one_count_jitter", [False, True])
def test_soy9sdfx_arrival_keeps_one_feedback_owner_across_quantized_plateaus(
    tmp_path, monkeypatch, direction, one_count_jitter,
):
    core = controller_from_profile(tmp_path, profile=FOUR_CENTRES)._workspace._coordinated.pose
    # Recorded sweep-12: the endpoint had already been reached, then native
    # arrival lasted 172 seconds. Low-count windows repeatedly selected the
    # stationary integrator instead of arrival velocity feedback.
    goal = .0873598775598299
    position = goal + direction*.01043439444572812
    warm_core(core, q=(0, .05235987755982989, position, 0, 0, 0), target=goal-position)
    core.adopt_hold(1, .06)
    core.adopt_hold(2, direction*.24518015077561686)
    core.last_delta = core.bias
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, q, i=i, **kwargs):
            target = core.goal[i]
            effort = core.motion.tracking_kp_nm_per_rad*(target-q)
            return B601JointMotionFrame(phase="brake", reason="target_reached_braking",
                q_reference_rad=target, qvel_reference_rad_s=0,
                control_qvel_rad_s=0, position_torque_nm=effort,
                velocity_torque_nm=0, requested_torque_nm=effort)
        monkeypatch.setattr(tracker, "update", frame)
    active = 0
    for n in range(1, 81):
        q = list(core.q)
        q[2] = position + (RESOLUTION if one_count_jitter and n % 2 else 0)
        raw = tuple((a-b)/.01 for a, b in zip(q, core.q))
        dt = core.observe(core.last_time+.01, q, raw)
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        status = audit["cruise_velocity_control"]
        if n >= 25:
            active += status["active"]
            assert status["active"] and status["arrival"]
            assert audit["approach_reasons"][2] == "arrival_position_velocity_bias_refinement"
            assert status["arrival_position_bias_rate_nm_s"]*direction < 0
            assert abs(status["arrival_position_bias_rate_nm_s"]) <= (
                core.config.cruise_velocity_ki_nm_per_rad*core.trackers[1].config.max_reference_velocity_rad_s)
        elif n <= 20:
            assert audit["proposed_bias_nm"] == core.bias
        core.record_dispatch(30+n, command)
        assert core.bias[1] == .06
        assert sum(abs(v) for v in command) <= 1+1e-12
    assert active == 56
    # Rejected conditioning must not commit the restoration proposal or feed
    # a velocity observer. The following warm-up cannot revive the old
    # stationary integrator. Revocation still clears all active support.
    before = core.bias
    dt = core.observe(core.last_time+.01, core.q, ZERO)
    command, _ = core.step(sequence=111, dt=dt, permitted=True, point_search=False, cruise_joint=2)
    core.record_dispatch(111, tuple(v*.99 for v in command))
    assert core.bias == before
    dt = core.observe(core.last_time+.01, core.q, ZERO)
    command, audit = core.step(sequence=112, dt=dt, permitted=True, point_search=False, cruise_joint=2)
    assert not audit["cruise_velocity_control"]["active"]
    assert audit["proposed_bias_nm"] == before
    core.record_dispatch(112, command)
    command, _ = core.step(sequence=113, dt=.01, permitted=False, point_search=False, cruise_joint=2)
    core.record_dispatch(113, command)
    assert core.bias == command == ZERO


@pytest.mark.parametrize("direction", [-1, 1])
def test_arrival_still_window_inside_tolerance_retains_unknown_support(tmp_path, monkeypatch, direction):
    core = controller_from_profile(tmp_path, profile=FOUR_CENTRES)._workspace._coordinated.pose
    q = (0, .05, .08, 0, 0, 0)
    warm_core(core, q=q, target=direction*core.arrival_tolerance_rad/2)
    core.adopt_hold(2, direction*.3)
    core.last_delta = core.bias
    for tracker, i in zip(core.trackers, AXES, strict=True):
        def frame(*, i=i, **kwargs):
            return B601JointMotionFrame(phase="brake", reason="target_reached_braking",
                q_reference_rad=core.goal[i], qvel_reference_rad_s=0,
                control_qvel_rad_s=0, position_torque_nm=0,
                velocity_torque_nm=0, requested_torque_nm=0)
        monkeypatch.setattr(tracker, "update", frame)
    for n in range(1, 51):
        dt = core.observe(core.last_time+.01, q, ZERO)
        command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False, cruise_joint=2)
        if n >= 25:
            status = audit["cruise_velocity_control"]
            assert status["active"] and status["position_span_rad"] == 0
            assert status["arrival_position_bias_rate_nm_s"] == 0
            assert status["requested_bias_increment_nm"] == 0
        core.record_dispatch(30+n, command)
        assert core.bias[2] == pytest.approx(direction*.3)


@pytest.mark.parametrize("width", [True, .002, float("nan"), float("inf"), -.1])
def test_invalid_position_bin_option_rejected_before_start(width):
    with pytest.raises(ValueError):
        WorkspaceCalibrationConfig(holding_sampling_plan="identification",
            coordinated_holding=CoordinatedHoldingConfig(), holding_search=HoldingSearchConfig(),
            identification_position_bin_width_rad=width)


@pytest.mark.parametrize("period,profile", [(.01, CANDIDATE), (.025, CANDIDATE), (.01, FOUR_CENTRES)])
def test_complete_identification_wires_cruise_only_to_training_validation_axis(tmp_path, period, profile):
    from test_b601_automatic_calibration import AcquisitionReplay
    from test_b601_workspace_calibration import LIMITS

    controller = controller_from_profile(tmp_path, profile=profile)
    r = AcquisitionReplay(controller._workspace.settings, controller=controller, position_limits=LIMITS)
    controller.submit("select_mode", now_sec=r.now, operation_mode="workspace")
    for _ in range(20):
        r.tick()
    controller.submit("arm", now_sec=r.now, operation_mode="workspace", return_and_disable=True)
    active_roles = set()
    centres = len(controller._workspace.workspace.pose_centres())
    for _ in range(math.ceil(450*centres/period)):
        # Prescribed encoder motion exercises the real GUI-mode owner/receipt
        # adapter. Plant response is assessed separately below.
        q = list(r.q)
        goal = controller._workspace._coordinated.pose.goal
        if goal is not None:
            for i in AXES:
                q[i] += max(-.008*period, min(.008*period, goal[i]-q[i]))
        r.tick(q, dt_sec=period)
        audit = r.audit.get("coordinated_holding") or {}
        cruise = audit.get("cruise_velocity_control") or {}
        marker = r.audit.get("calibration_trial") or {}
        if cruise.get("active"):
            assert marker["role"] in {"training", "validation"}
            assert cruise["joint_name"] == r.audit["joint_name"]
            active_roles.add(marker["role"])
        if (r.audit.get("automatic_calibration") or {}).get("state") in {"complete", "failed", "cancelled"}:
            break
    result = controller.result_record()
    assert result["state"] == "complete", result["reason"]
    assert len(result["motion"]["completed_legs"]) == 28*centres
    assert active_roles == {"training", "validation"}
    assert controller.cycle_disable_confirmation_ready


def simulate_motion(tmp_path, *, candidate, speed, direction, inertia=.2, period=.01, arrival=False,
                    static_scale=1., load_offset=0., field_mode="off", field_source=None):
    """Declared stick/slip plant, not identified B601 mechanics.

    Known motor damping is instantaneous in this toy. Gravity residual and
    friction vary with position; sensor positions are quantized. The exact
    same initial conditions and drive limits are used for both controllers.
    """
    core = controller_from_profile(tmp_path, candidate=candidate)._workspace._coordinated.pose
    if field_source is not None:
        from robot_motion.rebot_b601_rs.coordinated_holding import (
            CoordinatedPoseHolding,
        )

        core = CoordinatedPoseHolding(core.settings, replace(core.config, command_field_mode=field_mode),
            arrival_tolerance_rad=core.arrival_tolerance_rad, command_field_source=field_source)
    if arrival:
        core.config = replace(core.config, cruise_arrival_feedback=True)
    centre = math.radians(5)
    q = [0, centre, centre-direction*.035, 0, 0, 0]
    measured = tuple(round(v/RESOLUTION)*RESOLUTION for v in q)
    for n in range(30):
        dt = core.observe(1+n*.01, measured, ZERO)
        command, _ = core.step(sequence=n, dt=dt, permitted=False)
        core.record_dispatch(n, command)
    goal = list(measured)
    goal[2] += direction*.07
    core.set_goal(goal, max_reference_velocity_rad_s=speed)
    core.adopt_hold(2, direction*.35)
    q = list(measured)
    velocity = [0.0]*6
    command = ZERO
    now = core.last_time
    samples, active, peak_speed, field_predictions, field_increments = [], 0, 0.0, 0, 0
    outcome = "timeout"
    quiet_since = None
    for n in range(1, int(40/period)+1):
        for _ in range(round(period/.001)):
            for i in AXES:
                load = .08+.2*(q[3-i]-centre)+load_offset
                static = static_scale*(.24+.04*math.sin(45*q[i]))
                kinetic = .10+.02*math.cos(45*q[i])
                net = command[i]-load
                v = velocity[i]
                if abs(v) < 1e-8 and abs(net) <= static:
                    velocity[i] = 0.0
                    continue
                friction = kinetic+(static-kinetic)*math.exp(-(v/.003)**2)
                friction *= math.copysign(1, v if abs(v) > 1e-8 else net)
                next_v = v+(net-friction-4.5*v)/inertia*.001
                if next_v*v < 0 and abs(net) <= static:
                    next_v = 0.0
                velocity[i] = next_v
                q[i] += next_v*.001
        now += period
        current = tuple(round(v/RESOLUTION)*RESOLUTION for v in q)
        raw = tuple((a-b)/period for a, b in zip(current, measured, strict=True))
        measured = current
        dt = core.observe(now, measured, raw)
        try:
            command, audit = core.step(sequence=30+n, dt=dt, permitted=True, point_search=False,
                                      cruise_joint=2 if candidate else None,
                                      field_pose_index=0 if field_source is not None else None,
                                      field_driver_kd=(.4, 4.5, 4.5, 1.2, .2, .1) if field_source is not None else None)
        except ValueError as error:
            outcome = str(error)
            break
        core.record_dispatch(30+n, command)
        assert sum(abs(v) for v in command) <= 1+1e-12
        peak_speed = max(peak_speed, abs(velocity[2]))
        progress = direction*(q[2]-(goal[2]-direction*.07))/.07
        if .2 <= progress <= .8:
            samples.append(velocity[2]-direction*speed)
        active += audit["cruise_velocity_control"]["active"]
        field_predictions += (audit.get("command_field") or {}).get("predicted", False)
        field_increments += abs((audit.get("command_field") or {}).get("requested_increment_nm", 0)) > 1e-12
        # The real continuous-acquisition handoff requires a continuous
        # settle dwell. A single quiet, in-band sample can precede rebound.
        if max(abs(q[i]-goal[i]) for i in AXES) <= .003375 and core.quiet():
            if quiet_since is None:
                quiet_since = now
            if now-quiet_since >= core.settings.staged_assist.settle_dwell_sec:
                outcome = "arrived"
                break
        else:
            quiet_since = None
    return {"outcome": outcome, "samples": len(samples), "active_cycles": active,
        "rms_speed_error_rad_s": math.sqrt(sum(v*v for v in samples)/len(samples)) if samples else None,
        "peak_speed_rad_s": peak_speed, "duration_sec": now-1.29,
        "final_error_rad": q[2]-goal[2], "inertia": inertia, "period_sec": period,
        "reference_speed_rad_s": speed, "direction": direction,
        "field_predicted_cycles": field_predictions, "field_increment_cycles": field_increments}


@pytest.mark.parametrize("speed,direction,inertia,period", [
    (.0075, 1, .08, .01), (.0075, -1, .2, .01),
    (.015, 1, .2, .01), (.015, -1, .6, .01),
])
def test_quantized_stick_slip_cruise_improves_actual_speed_without_relaxing_limits(
    tmp_path, speed, direction, inertia, period,
):
    baseline = simulate_motion(tmp_path, candidate=False, speed=speed, direction=direction, inertia=inertia, period=period)
    candidate = simulate_motion(tmp_path, candidate=True, speed=speed, direction=direction, inertia=inertia, period=period)
    print(json.dumps({"baseline": baseline, "candidate": candidate}))
    assert candidate["outcome"] == "arrived", candidate
    assert candidate["active_cycles"] > 0 and candidate["samples"] > 10
    assert candidate["peak_speed_rad_s"] < .195
    assert candidate["rms_speed_error_rad_s"] < baseline["rms_speed_error_rad_s"]


@pytest.mark.parametrize("direction,inertia,static_scale,load", [
    (1, .2, 2., .15), (1, .6, 2., .15), (-1, .2, 2., -.15), (1, .2, 1., 0),
])
def test_arrival_feedback_keeps_cruise_and_reaches_existing_tolerance_in_declared_plant(
    tmp_path, direction, inertia, static_scale, load,
):
    results = [simulate_motion(tmp_path, candidate=True, speed=.0075, direction=direction,
        inertia=inertia, static_scale=static_scale, load_offset=load, arrival=enabled)
        for enabled in (False, True)]
    print(json.dumps({"declared_plant_not_identified_hardware": True, "comparisons": results}))
    for result in results:
        assert result["outcome"] == "arrived", result
        assert abs(result["final_error_rad"]) <= .003375
        assert result["peak_speed_rad_s"] < .195
    assert results[1]["rms_speed_error_rad_s"] == pytest.approx(results[0]["rms_speed_error_rad_s"], abs=1e-12)

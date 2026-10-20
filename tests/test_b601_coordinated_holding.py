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


def test_coordinated_fault_retains_actual_failed_axis_and_new_brake_clock(settings):
    core = runtime(settings)
    for tracker in core.trackers:
        tracker.config = replace(tracker.config, braking_timeout_sec=.2)
    warm(core)
    core.set_goal(ZERO)
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
def test_all_axis_authority_and_final_envelope_remain_enforced(tmp_path, kind):
    r = replay(tmp_path)
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


def test_native_speed_brake_retains_meaningful_probe_progress_as_a_censored_transition(tmp_path):
    r = replay(tmp_path, holding_search=HoldingSearchConfig(
        observation_sec=1.5, observation_timeout_sec=4.5, movement_rad=.0045))
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


def test_support_only_brake_does_not_invent_a_selected_axis_holding_response(tmp_path):
    r = replay(tmp_path)
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


def test_current_margin_profile_resolves_workspace_only_and_keeps_hard_envelopes(tmp_path):
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coordinated_holding_v1.json"
    values = json.loads(profile.read_text())["workspace"]
    whole = production_controller.__wrapped__(tmp_path)
    old = whole._workspace
    updated = B601WorkspaceCalibrationController(old.settings, old.config, WorkspaceCalibrationConfig(**values))
    for name, before in {"prepare_timeout_sec": 13, "max_batch_sec": 390,
                         "max_sweep_command_exposure_nm_sec": 80,
                         "motion_braking_timeout_sec": 6.591,
                         "motion_soft_velocity_rad_s": .0585}.items():
        factor = 1.3 * (1.5 if name in {"max_batch_sec", "max_sweep_command_exposure_nm_sec"} else 1)
        assert getattr(updated.config, name) == pytest.approx(before*factor)
        assert getattr(old.config, name) == before
    assert updated.config.max_sweep_leg_sec == pytest.approx(76.05*1.5)
    assert updated.config.max_sweep_leg_command_exposure_nm_sec == pytest.approx(31.2*1.5)
    assert updated._coordinated.pose.arrival_tolerance_rad == .00225
    motion = updated.settings.staged_assist.motion_control
    assert motion is updated._coordinated.pose.motion
    assert motion.soft_velocity_rad_s == motion.continuous_speed_limit_rad_s == .07605
    assert motion.braking_timeout_sec == 8.5683
    assert old.settings.staged_assist.motion_control.soft_velocity_rad_s == .0585
    assert whole._fixed.config is old.config
    assert whole._supported.config is old.config
    assert updated.limits.max_joint_velocity_rad_s == old.limits.max_joint_velocity_rad_s
    assert updated.workspace.coordinated_holding.total_assist_cap_nm == 1
    assert motion.max_control_gap_sec == .04
    assert motion.drive_driver_kd_nm_s_per_rad == old.settings.staged_assist.motion_control.drive_driver_kd_nm_s_per_rad
    assert updated.workspace.holding_search.movement_rad == .0045
    assert updated.workspace.holding_search.observation_sec == 1.5
    assert updated.workspace.holding_search.observation_timeout_sec == pytest.approx(4.5*1.3)


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
    ("leg_time", "_leg_started", 76.05, 114.075, "leg_timeout"),
    ("leg_exposure", "_leg_exposure", 31.2, 46.8, "leg_command_exposure_budget"),
    ("sweep_time", "_sweep_started", 507, 760.5, "batch_timeout"),
    ("sweep_exposure", "_sweep_exposure", 104, 156, "active_command_exposure_budget"),
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


def test_position_arrival_still_requires_six_axis_stationarity(tmp_path):
    r = replay(tmp_path)
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
    from control_station.b601_torque_console import coordinated_workspace as module
    original = module.plan_local_sweeps
    def one_leg(**kwargs):
        plan = original(**kwargs)
        plan["steps"] = plan["steps"][:1]
        return plan
    monkeypatch.setattr(module, "plan_local_sweeps", one_leg)
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


@pytest.mark.parametrize("period", [.01, .025])
@pytest.mark.parametrize("passive_offset", [0, .006519419])
def test_full_coupled_batch_and_report_keep_handoffs_distinct_from_withdrawal(tmp_path, period, passive_offset, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from control_station.b601_torque_console import transparency
    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )
    changes = {}
    if period == .025 and passive_offset:
        # Exercise the actual explicit profile at the observed cadence too,
        # rather than validating the new margins only with shortened dwells.
        profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_coordinated_holding_v1.json"
        values = json.loads(profile.read_text())["workspace"]
        changes["holding_search"] = HoldingSearchConfig(**values["holding_search"])
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
            assert flags == [controller.workspace.positioning_support_soft_speed_recovery]*2
        elif controller._phase == "sweep":
            assert flags == [controller.pose.motion.support_soft_speed_recovery]*2
        if r.audit.get("coordinated_holding") is not None:
            # Audit reports the phase that produced this frame, before a
            # consumed receipt may advance the outer workspace lifecycle.
            audit = r.audit["coordinated_holding"]
            assert audit["point_search_enabled"] == (r.audit["automatic_calibration"]["workspace"]["phase"] == "sweep")
        journal.observe(snapshot())
        assert sum(abs(v) for v in r.delta) <= 1+1e-9
        if old != len(r.controller._coordinated._acquisitions):
            assert tuple(r.delta) == tuple(r.audit["coordinated_holding"]["combined_command_nm"])
            handoffs.append(tuple(r.delta))
    result = r.controller.result_record()
    assert result["state"] == "complete", (result["reason"], r.q, r.controller._coordinated._holding_status())
    assert len(result["motion"]["completed_legs"]) == 24
    assert {leg["max_reference_velocity_rad_s"] for leg in result["motion"]["completed_legs"]} == set(r.controller.config.sweep_speeds_rad_s)
    # A learned small hold is valid too. Handoff must preserve the actual
    # dispatched composite, not manufacture a minimum support magnitude.
    assert len(handoffs) == 2 and all(any(abs(v) > 1e-9 for v in h) for h in handoffs)
    assert r.delta == r.last_kd == ZERO
    assert r.controller.cycle_disable_confirmation_ready
    assert all(abs(r.q[i]-.01) < .003 for i in (1, 2))
    journal.close()
    (tmp_path / "b601_owner.debug.jsonl").write_text("\n".join(json.dumps(x) for x in [*r.records, result])+"\n")
    report = build_automatic_report(tmp_path)
    assert report["holding_calibration"]["verified_points"] == 24
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

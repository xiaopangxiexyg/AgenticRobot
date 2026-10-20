"""Coordinated software contracts; no devices or measured force evidence."""

import json
from dataclasses import replace

import pytest
from test_b601_automatic_calibration import AcquisitionReplay
from test_b601_calibration_session import settings as shared_settings
from test_b601_workspace_calibration import LIMITS, production_controller

from control_station.b601_torque_console.workspace_calibration import (
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
    assert r.controller.status()["automatic_calibration"]["workspace"]["arrival"]["reason"] == "ready"


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
    from types import SimpleNamespace

    from control_station.b601_torque_console import transparency
    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )
    r = replay(tmp_path, initial_q=(0, .01, .01, 0, 0, 0))
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
        journal.observe(snapshot())
        assert sum(abs(v) for v in r.delta) <= 1+1e-9
        if old != len(r.controller._coordinated._acquisitions):
            handoffs.append(tuple(r.delta))
    result = r.controller.result_record()
    assert result["state"] == "complete", (result["reason"], r.q, r.controller._coordinated._holding_status())
    assert len(result["motion"]["completed_legs"]) == 24
    assert {leg["max_reference_velocity_rad_s"] for leg in result["motion"]["completed_legs"]} == set(r.controller.config.sweep_speeds_rad_s)
    assert len(handoffs) == 2 and all(any(abs(v) > .01 for v in h) for h in handoffs)
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
        assert max(b-a for a,b in zip(observation["minimum_rad"], observation["maximum_rad"])) <= .003
        assert observation["qpos_rad"][3] == pytest.approx(passive_offset)
    assert report["cycle"]["zero_assist_base_damping_confirmed"]
    assert (tmp_path / "b601_owner.transparency.jsonl").stat().st_size > 0
    for point in report["holding_calibration"]["points"]:
        assert point["sample_pose_rad"][3] == pytest.approx(passive_offset)
        for obs in point["observations"]:
            if obs["outcome"] != "stationary":
                continue
            assert obs["started_at_sec"] > obs["first_causal_receipt_at_sec"]
            assert max(b-a for a,b in zip(obs["minimum_rad"], obs["maximum_rad"])) <= .003
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

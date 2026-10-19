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
    cfg = replace(legacy.workspace, joint2_centres_deg=(3,), joint3_centres_deg=(3,),
        gravity_observation_sec=.15, holding_excursion_rad=.01,
        holding_search=HoldingSearchConfig(observation_sec=.15, observation_timeout_sec=1),
        coordinated_holding=CoordinatedHoldingConfig(), **changes)
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


def coupled_tick(r, *, period=.025):
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
    r.tick(q, dt_sec=period)


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
def test_full_coupled_batch_and_report_keep_handoffs_distinct_from_withdrawal(tmp_path, period):
    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )
    r = replay(tmp_path, initial_q=(0, .01, .01, 0, 0, 0))
    r.controller.submit("arm", now_sec=r.now)
    handoffs = []
    for _ in range(50000):
        state = r.controller.status()["automatic_calibration"]["state"]
        if state in {"complete", "failed", "cancelled"}:
            break
        old = len(r.controller._coordinated._acquisitions)
        coupled_tick(r, period=period)
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
    (tmp_path / "b601_owner.debug.jsonl").write_text("\n".join(json.dumps(x) for x in [*r.records, result])+"\n")
    (tmp_path / "b601_owner.transparency.jsonl").write_text("")
    report = build_automatic_report(tmp_path)
    assert report["holding_calibration"]["verified_points"] == 24
    assert all(not p["withdrawal_confirmed"] for p in report["holding_calibration"]["points"])
    assert report["workspace"]["gravity_observations"] == []
    assert len(report["workspace"]["coordinated_pose_observations"]) == 2
    assert report["cycle"]["zero_assist_base_damping_confirmed"]
    for point in report["holding_calibration"]["points"]:
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

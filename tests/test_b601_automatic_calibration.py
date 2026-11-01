"""Automatic acquisition with native controllers and scripted encoder feedback."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_b601_calibration_session import settings as shared_settings
from test_b601_command_calibration import condition

from control_station.b601_torque_console.automatic_calibration import (
    AutomaticCalibrationConfig,
    B601AutomaticCalibrationController,
)
from control_station.b601_torque_console.calibration_session import (
    complete_calibration_receipt,
)
from hardware.rebot_b601_rs.backend import B601CommandReceipt
from robot_motion.rebot_b601_rs.calibration import summarize_automatic_acquisition
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

ZERO = (0.0,) * 6


@pytest.fixture
def settings(tmp_path):
    return shared_settings.__wrapped__(tmp_path)


class AcquisitionReplay:
    def __init__(self, settings, *, config=None, controller=None, position_limits=None, initial_q=ZERO):
        self.settings = settings
        self.controller = controller or B601AutomaticCalibrationController(
            settings, config or AutomaticCalibrationConfig()
        )
        self.condition = condition(run_id="automatic-contract-replay")
        self.controller.configure_context(self.condition, position_limits or ((-1.0, 1.0),) * 6)
        self.controller.owner_ready()
        self.controller.set_human_intervention(False)
        self.now, self.sequence = 1.0, 0
        self.q = initial_q
        self.delta = ZERO
        self.audit = {}
        self.records = []
        self.audits = []
        for _ in range(12):
            self.tick()

    def tick(
        self,
        q=None,
        *,
        permit=True,
        receipt_updates=None,
        final_delta=None,
        ack=True,
        kd_override=None,
        dt_sec=.01,
        dispatch_start_delay_sec=.0002,
        completion_delay_sec=.0008,
        heartbeat_active=None,
    ):
        self.now = round(self.now + dt_sec, 9)
        self.sequence += 1
        position = self.q if q is None else tuple(q)
        velocity = tuple((a - b) / dt_sec for a, b in zip(position, self.q))
        self.q = position
        # Reuse only a caller's snapshot from before any controller mutation.
        active = (
            self.controller.status()["automatic_calibration"]["active"]
            if heartbeat_active is None else heartbeat_active
        )
        if active and permit:
            self.controller.submit("heartbeat", now_sec=self.now)
        self.delta, self.audit = self.controller.step(
            now_sec=self.now,
            control_now_sec=self.now + 0.0001,
            qpos_rad=position,
            qvel_rad_s=velocity,
            feedback_sequence=self.sequence,
        )
        self.audits.append(self.audit)
        if self.audit["phase"] == "fault":
            return
        final = self.delta if final_delta is None else final_delta
        self.controller.validate_final_command(gravity_nm=ZERO, total_nm=final)
        # Use the owner adapter too: direct delegation alone misses wiring
        # failures when a GUI/cycle controller wraps the acquisition controller.
        kd = probe._motion_console_driver_kd(
            self.controller, ZERO, final_command_delta_nm=final,
        )
        self.last_kd = kd if kd_override is None else kd_override
        self.controller.validate_dispatch_start(
            feedback_sequence=self.sequence, feedback_timestamp_sec=self.now,
            now_sec=self.now+dispatch_start_delay_sec,
        )
        # Exercise the live in-process serializer, not a JSON-shaped stand-in.
        # Command and feedback sequences are intentionally separate counters.
        receipt = probe._command_receipt_payload(
            B601CommandReceipt(
                accepted=True,
                mode="mit",
                frame_id="b601_rs/joint_mit_reference",
                sequence=self.sequence + 500,
                source_sequence=self.sequence,
                requested_joint_names=B601_RS_ARM_JOINT_NAMES,
                sent_joint_names=B601_RS_ARM_JOINT_NAMES,
                errors=(),
                failed_joint_names=(),
                requested_at_sec=self.now + dispatch_start_delay_sec,
                completed_at_sec=self.now + completion_delay_sec,
            )
        )
        receipt.update(receipt_updates or {})
        self.receipt = receipt
        if not ack:
            return
        self.controller.record_dispatch(
            native_receipt=receipt,
            joint_name=self.audit["joint_name"],
            receipt_sequence=receipt["sequence"],
            feedback_timestamp_sec=self.now,
            completed_at_sec=self.now + completion_delay_sec,
            overlay_after_profile_nm=self.delta[B601_RS_ARM_JOINT_NAMES.index(self.audit["joint_name"])],
            gravity_nm=0.0,
            requested_total_nm=self.delta[B601_RS_ARM_JOINT_NAMES.index(self.audit["joint_name"])],
            conditioned_total_nm=final[B601_RS_ARM_JOINT_NAMES.index(self.audit["joint_name"])],
            joint_commands={
                "gravity_nm": ZERO,
                "overlay_nm": self.delta,
                "total_nm": final,
                "driver_kd_nm_s_per_rad": self.last_kd,
            },
            torque_rate_nm_s=(28.8,) * 6,
        )
        self.records.extend(self.controller.drain_records())

    def begin(self):
        self.controller.submit("arm", now_sec=self.now + 0.001)

    def run(self, *, onset_move=True, motion_move=True, ticks=20000, until=None):
        for _ in range(ticks):
            view = self.controller.status()
            state = view["automatic_calibration"]
            if until is not None and until(state):
                return
            if state["state"] in {"complete", "failed", "cancelled"}:
                return
            q = list(self.q)
            if (
                onset_move
                and state["state"] == "onset_scan"
                and abs(self.delta[2]) >= 0.18
            ):
                q[2] += state["direction"] * 0.0001
            if (
                motion_move
                and state["stage"] == "motion"
                and state["state"] == "motion"
            ):
                # A callback may mutate the controller even when it returns false.
                offset = (
                    view if until is None else self.controller.status()
                ).get("target_offset_rad")
                audit = self.audit
                motion = audit.get("motion_control")
                if motion and motion["phase"] == "track":
                    q[2] = motion["q_reference_rad"]
                elif (
                    audit.get("phase") == "breakaway"
                    and offset
                    or (
                        audit.get("phase") == "releasing"
                        and not audit.get("release_terminal")
                        and abs(q[2] - audit["anchor_rad"]) < 0.001
                        and offset
                    )
                ):
                    q[2] += 0.0001 if offset > 0 else -0.0001
            self.tick(q, heartbeat_active=state["active"] if until is None else None)
        pytest.fail(str(self.controller.status()["automatic_calibration"]))


def _pending_onset_with_two_counts(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_arm", onset_move=False, motion_move=False)
    q = list(replay.q)
    count = .0003834962844848633
    q[2] += count
    replay.tick(q)
    assert replay.controller.status()["automatic_calibration"]["state"] == "onset_scan"
    assert replay.audit["onset_capture"]["trial"] is None
    deadline = replay.controller.status()["automatic_calibration"]["onset_baseline"]["prepare_deadline_sec"]
    q[2] += count
    replay.tick(q)
    assert replay.audit["phase"] == "armed" and replay.delta == ZERO
    assert replay.audit["onset_capture"]["trial"] is None
    assert replay.audit["onset_start_wait_reason"] == "onset_capture_position_not_settled"
    return replay, deadline


def test_onset_baseline_race_holds_zero_and_resumes_same_trial_after_continuous_stillness(settings):
    replay, deadline = _pending_onset_with_two_counts(settings)
    quiet_at, q = replay.now, replay.q
    started = replay.controller._scan_started
    for _ in range(10):
        replay.tick()
        assert replay.delta == ZERO
        assert replay.audit["onset_capture"]["trial"] is None
    replay.run(until=lambda s: replay.audit["onset_capture"]["trial"] is not None,
        onset_move=False, motion_move=False)
    trial = replay.audit["onset_capture"]["trial"]
    assert trial["baseline_qpos_rad"] == q
    assert trial["started_at_sec"] - quiet_at >= settings.staged_assist.settle_dwell_sec
    assert replay.controller._scan_started == started
    assert replay.controller.status()["automatic_calibration"]["onset_baseline"]["prepare_deadline_sec"] == deadline
    assert trial["trial_sequence"] == 1
    assert replay.delta[2] < 0
    assert replay.controller.status()["automatic_calibration"]["completed_onset_trials"] == 0


def test_onset_first_force_includes_current_feedback_in_whole_baseline_range(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["onset_baseline"]["dwell_sec"] >= .05,
        onset_move=False, motion_move=False)
    q = list(replay.q)
    count = .0003834962844848633
    q[2] += count
    replay.tick(q)
    first_move_at = replay.now
    replay.run(until=lambda s: s["state"] == "onset_scan", onset_move=False, motion_move=False)
    assert replay.audit["onset_capture"]["trial"] is None
    deadline = replay.controller.status()["automatic_calibration"]["onset_baseline"]["prepare_deadline_sec"]
    q[2] += count
    replay.tick(q)
    # The first count has aged out of the 80ms estimator, but is still in the
    # whole 150ms admission dwell. Current raw/window gates alone are NOT proof.
    assert replay.now - first_move_at > .08
    assert replay.audit["velocity_window"]["position_span_rad"][2] <= .000384
    assert abs(replay.audit["qvel_fd_rad_s"][2]) < .05
    assert replay.delta == ZERO
    assert replay.audit["onset_capture"]["trial"] is None
    quiet_at = replay.now
    replay.run(until=lambda s: replay.audit["onset_capture"]["trial"] is not None,
        onset_move=False, motion_move=False)
    assert replay.audit["onset_capture"]["trial"]["started_at_sec"] - quiet_at >= settings.staged_assist.settle_dwell_sec
    assert replay.controller.status()["automatic_calibration"]["onset_baseline"]["prepare_deadline_sec"] == deadline


def test_onset_pending_baseline_chatter_cannot_renew_preparation_deadline_or_start_trial(settings):
    replay, deadline = _pending_onset_with_two_counts(settings)
    base = replay.q[2] - 2*.0003834962844848633
    for n in range(600):
        state = replay.controller.status()["automatic_calibration"]
        if state["state"] not in {"onset_arm", "onset_scan"}:
            break
        q = list(replay.q)
        q[2] = base + (1, 0, 1, 2)[n % 4] * .0003834962844848633
        replay.tick(q)
        assert replay.delta == ZERO
        assert replay.audit["onset_capture"]["trial"] is None
        assert replay.controller.status()["automatic_calibration"]["onset_baseline"]["prepare_deadline_sec"] == deadline
    else:
        pytest.fail("pending start silently extended its deadline")
    assert deadline < replay.now <= deadline + .011
    replay.run(onset_move=False, motion_move=False)
    result = replay.controller.result_record()
    assert result["reason"] == "preparation_timeout"
    assert result["onset_trials"] == []
    assert replay.delta == ZERO


def test_pre_force_raw_count_confirmation_does_not_skip_trial_or_renew_deadline(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan", onset_move=False, motion_move=False)
    deadline = replay.controller.status()["automatic_calibration"]["onset_baseline"]["prepare_deadline_sec"]
    q = list(replay.q)
    q[2] += .0003834962844848633
    replay.tick(q, dt_sec=.0075)
    assert replay.audit["phase"] == "velocity_check"
    assert replay.delta == ZERO and replay.audit["onset_capture"]["trial"] is None
    for _ in range(50):
        replay.tick()
        state = replay.controller.status()["automatic_calibration"]
        assert state["onset_index"] == state["completed_onset_trials"] == 0
        assert state["onset_baseline"]["prepare_deadline_sec"] == deadline
        if replay.audit["onset_capture"]["trial"] is not None:
            break
        assert replay.delta == ZERO
    else:
        pytest.fail("zero-first confirmation could not return to pending baseline")
    assert replay.audit["onset_capture"]["trial"]["trial_sequence"] == 1
    assert replay.delta[2] < 0


@pytest.mark.parametrize("stop", ["human", "lease", "raw", "gap"])
def test_onset_pending_baseline_wait_keeps_original_cancellation_and_hard_guards(settings, stop, monkeypatch):
    replay, _ = _pending_onset_with_two_counts(settings)
    if stop == "human":
        from robot_motion.rebot_b601_rs import torque_experiment
        # The HTTP callback and synthetic owner feedback share one clock.
        monkeypatch.setattr(torque_experiment, "time", SimpleNamespace(monotonic=lambda: replay.now+.001))
        replay.controller.set_human_intervention(True)
    elif stop == "raw":
        q = list(replay.q)
        q[2] += .001
        replay.tick(q)
        assert replay.audit["phase"] == "fault"
    elif stop == "gap":
        replay.now += .05
    for n in range(200):
        state = replay.controller.status()["automatic_calibration"]
        if state["state"] in {"failed", "cancelled"}:
            break
        q = list(replay.q)
        if stop == "lease" and state["state"] == "onset_scan":
            # Keep baseline pending until the EXISTING lease actually expires;
            # withholding renewal does not retroactively revoke a valid lease.
            q[2] = (1, 0, 1, 2)[n % 4] * .0003834962844848633
        replay.tick(q, permit=stop != "lease")
        assert replay.delta == ZERO
        assert replay.audit["onset_capture"]["trial"] is None
    result = replay.controller.result_record()
    assert result["state"] in {"failed", "cancelled"}
    assert result["onset_trials"] == []
    assert replay.delta == ZERO


@pytest.mark.parametrize("mode", ["legacy", "release_braking", "acquisition"])
def test_complete_onset_and_motion_acquisition_reuses_native_ports(settings, mode):
    release_braking = mode != "legacy"
    replay = AcquisitionReplay(settings, config=AutomaticCalibrationConfig(
        onset_release_braking=release_braking,
        onset_joint_velocity_limit_rad_s=.15 if release_braking else None,
        sweep_excursion_rad=.010 if mode == "acquisition" else .008,
        motion_acquisition_arrival_tolerance_rad=.005 if mode == "acquisition" else None,
        max_sweep_leg_command_exposure_nm_sec=8. if mode == "acquisition" else None,
        max_sweep_command_exposure_nm_sec=80. if mode == "acquisition" else 20.,
        max_sweep_leg_sec=20. if mode == "acquisition" else 12.,
        max_batch_sec=300. if mode == "acquisition" else 150.,
    ))
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    assert result["state"] == "complete", result
    assert len(result["onset_trials"]) == 4
    assert [t["direction"] for t in result["onset_trials"]] == [-1, 1, 1, -1]
    assert len(result["motion"]["completed_legs"]) == 14
    report = summarize_automatic_acquisition(result)
    assert all(t["outcome"] == "confirmed_encoder_progress" for t in report["trials"])
    assert sorted(g["count"] for g in report["repeatability"]) == [2, 2]
    assert replay.delta == ZERO
    assert result["auto_apply"] is False
    assert report["acquisition_complete"] is True
    assert report["tracking_arrival_quality"]["outside_tracking_tolerance_legs"] == 0
    if mode == "acquisition":
        assert result["motion"]["budget_plan"]["step_count"] == 14
        assert result["motion"]["max_batch_command_exposure_nm_sec"] == 80.
        assert result["motion"]["max_leg_command_exposure_nm_sec"] == 8.
    assert result["onset_damping"]["base_restore_completed_at_sec"] is not None
    assert result["onset_damping"]["last_acknowledged_kd_nm_s_per_rad"] == ZERO
    assert all(
        t["active_driver_kd_nm_s_per_rad"] == (0.4, 0.6, 0.0, 0.6, 0.2, 0.1)
        and t["zero_assist_completed_at_sec"] is not None
        for t in result["onset_trials"]
    )


def test_held_probe_automatic_mode_reuses_native_complete_plan_and_keeps_default_off(settings):
    replay = AcquisitionReplay(settings,config=AutomaticCalibrationConfig(
        motion_position_resolution_rad=.0004,
        motion_responsive_tracking_brake=True,
        motion_breakaway_probe=True))
    assert settings.staged_assist.breakaway_probe is None
    probe_config = replay.controller.settings.staged_assist.breakaway_probe
    assert probe_config.hold_sec == .12
    assert probe_config.max_trial_sec == settings.staged_assist.breakaway_max_duration_sec
    assert probe_config.max_command_exposure_nm_sec == settings.staged_assist.breakaway_max_impulse_nm_sec
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    assert result["state"] == "complete",result
    assert len(result["motion"]["completed_legs"]) == 14
    assert replay.delta == ZERO
    assert result["motion_settings"]["breakaway_probe"]["hold_sec"] == .12
    assert any(row.get("breakaway_probe", {}).get("stage") == "hold"
        for row in replay.audits if row.get("breakaway_probe"))


@pytest.mark.parametrize("governor,arrival_stabilization", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("plan", ["motion_check", "local_identification"])
def test_continuous_motion_plans_reuse_owner_and_preserve_roles(settings, governor, arrival_stabilization, plan):
    config = AutomaticCalibrationConfig(
        motion_position_resolution_rad=.0004, motion_responsive_tracking_brake=True,
        motion_breakaway_probe=True, motion_tracking_recovery_cap_nm=.3,
        motion_reference_pacing=True, motion_onset_handoff="track", motion_plan=plan,
        motion_speed_governor=governor, motion_arrival_stabilization=arrival_stabilization,
        sweep_excursion_rad=.020, global_excursion_rad=.040)
    replay = AcquisitionReplay(settings, config=config)
    assert settings.staged_assist.onset_handoff == "stop_and_zero"
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    assert result["state"] == "complete", result
    assert len(result["onset_trials"]) == 4
    expected = 2 if plan == "motion_check" else 12
    assert len(result["motion"]["completed_legs"]) == expected
    assert result["settings"]["motion_plan"] == plan
    assert result["motion"]["step_count"] == expected
    legs = result["motion"]["completed_legs"]
    assert [leg["role"] for leg in legs] == (
        ["positioning_not_for_fit", "return_not_for_fit"] if plan == "motion_check" else ["training"]*8+["validation"]*4)
    handoffs = [a for a in replay.audits if a.get("reason") == "confirmed_onset_transferred_to_tracking"]
    assert len(handoffs) == expected
    assert all(a["tracking_recovery"]["handoff_seed_command_nm"] != 0 for a in handoffs)
    assert all(leg["final_zero_and_still_confirmed"] for leg in legs)
    assert replay.delta == ZERO


def test_automatic_speed_governor_continues_same_leg_without_new_pulse_or_budget_reset(settings):
    config = AutomaticCalibrationConfig(motion_position_resolution_rad=.0004,
        motion_responsive_tracking_brake=True, motion_breakaway_probe=True,
        motion_tracking_recovery_cap_nm=.3, motion_reference_pacing=True,
        motion_onset_handoff="track", motion_plan="motion_check", motion_speed_governor=True)
    replay = AcquisitionReplay(settings, config=config)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "motion" and replay.audit.get("phase") == "track")
    session = replay.controller._sweep
    started, step, exposure = session._leg_started, session._index, session._leg_exposure
    runtime = session.controller._runtime
    onset_started = runtime._breakaway_started_sec
    for _ in range(7):
        q = list(replay.q); q[2] -= .000384
        replay.tick(q)
        if (replay.audit.get("motion_control") or {}).get("speed_governor_active"):
            break
    assert replay.audit["motion_control"]["speed_governor_active"]
    for _ in range(65):
        replay.tick()
        if replay.audit.get("reason") == "speed_governor_resumed_original_target":
            break
    assert replay.audit["reason"] == "speed_governor_resumed_original_target"
    assert session._state == "moving" and session._index == step
    assert session._leg_started == started
    assert session._leg_exposure >= exposure
    assert runtime._breakaway_started_sec == onset_started
    assert not session._completed_legs


@pytest.mark.parametrize("changes", [
    {"motion_speed_governor":True}, {"motion_speed_governor":1},
    {"motion_arrival_stabilization": True},
    {"motion_arrival_stabilization": 1},
])
def test_automatic_speed_governor_is_explicit_and_bounded(changes):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(**changes)


@pytest.mark.parametrize("changes", [
    {"motion_onset_handoff": "invalid"}, {"motion_plan": "invalid"},
    {"motion_onset_handoff": "track"},
    {"feedback_velocity_timebase": "invalid"},
    {"feedback_velocity_timebase": "joint_receive"},
])
def test_automatic_rejects_invalid_or_unqualified_tracking_handoff(changes):
    with pytest.raises(ValueError):
        AutomaticCalibrationConfig(**changes)


@pytest.mark.parametrize("rate", [0., -1., True, float("nan"), float("inf")])
def test_automatic_motion_unload_rate_rejects_invalid_values(rate):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(motion_braking_slew_nm_s=rate)


def test_motion_unload_override_cannot_change_default_mode():
    assert AutomaticCalibrationConfig().motion_braking_slew_nm_s is None
    with pytest.raises(ValueError, match="explicit tracking handoff"):
        AutomaticCalibrationConfig(motion_braking_slew_nm_s=10.)


@pytest.mark.parametrize("state,reason,severity", [
    ("failed", "stopped_short_of_target", "warning"),
    ("failed", "leg_timeout", "warning"),
    ("failed", "active_command_exposure_budget", "warning"),
    ("failed", "withdrawal_not_confirmed_owner_protection_required", "error"),
    ("failed", "owner_fault:raw_velocity_hard_guard", "error"),
    ("failed", "incomplete_or_mismatched_dispatch", "error"),
    ("failed", "new_unknown_fault", "error"),
    ("cancelled", "human_contact_or_unknown", "info"),
    ("cancelled", "global_batch_excursion", "error"),
    ("cancelled", "new_unknown_stop", "error"),
    ("cancelled", "operator_permit_expired", "warning"),
    ("complete", "all_targets_and_withdrawals_confirmed", "info"),
])
def test_acquisition_outcome_never_hides_unknown_or_safety_failures(state, reason, severity):
    from control_station.b601_torque_console.calibration_session import (
        calibration_outcome,
    )
    outcome = calibration_outcome(state, reason)
    assert outcome["severity"] == severity
    assert outcome["scope"] == "acquisition"
    assert outcome["reason"] == reason


@pytest.mark.parametrize("value", [-.01, True, float("nan"), float("inf")])
def test_motion_velocity_headroom_rejects_invalid_or_implicit_override(value):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(motion_velocity_headroom_ratio=value)
    with pytest.raises(ValueError, match="explicit pacing"):
        AutomaticCalibrationConfig(motion_velocity_headroom_ratio=.30)


@pytest.mark.parametrize("value", [0., -.01, True, float("nan"), float("inf")])
def test_motion_braking_timeout_requires_finite_positive_explicit_handoff(value):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(motion_braking_timeout_sec=value)
    with pytest.raises(ValueError, match="tracking handoff"):
        AutomaticCalibrationConfig(motion_braking_timeout_sec=3.9)


@pytest.mark.parametrize("values", [
    {"onset_release_support_confirmation":1},
    {"onset_release_support_confirmation":True},
    {"onset_release_support_confirmation":True,"onset_release_braking":True},
])
def test_automatic_release_support_confirmation_requires_explicit_braking_and_limits(values):
    with pytest.raises((TypeError,ValueError)):
        AutomaticCalibrationConfig(**values)


def test_automatic_release_support_warning_reuses_damping_zero_receipts_and_original_trial(settings, tmp_path):
    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )
    replay=AcquisitionReplay(settings,config=AutomaticCalibrationConfig(
        onset_release_braking=True,onset_joint_velocity_limit_rad_s=.15,
        onset_release_support_confirmation=True))
    replay.begin()
    replay.run(until=lambda s:s["state"]=="onset_release")
    q=list(replay.q);q[3]+=2*.00038349442183971405
    replay.tick(q)
    observation=replay.audit["onset_capture"]["release_support_observation"]
    assert observation["state"]=="observing"
    trial_start=replay.audit["onset_capture"]["trial"]["started_at_sec"]
    assert replay.delta==ZERO
    assert replay.last_kd[3]==.6
    for _ in range(60):
        replay.tick()
        assert replay.delta==ZERO
        if replay.controller.status()["automatic_calibration"]["completed_onset_trials"]:break
    result=replay.controller.result_record()
    assert len(result["onset_trials"])==1
    trial=result["onset_trials"][0]
    assert trial["started_at_sec"]==trial_start
    assert trial["release_support_observation"]["state"]=="confirmed"
    assert trial["outcome"]=="motion_onset_confirmed"
    assert result["state"]!="failed" and result["dispatch_rejection"] is None
    (tmp_path/"b601_owner.debug.jsonl").write_text("".join(json.dumps(r)+"\n" for r in [*replay.records,result]))
    (tmp_path/"b601_owner.transparency.jsonl").write_text("")
    report=build_automatic_report(tmp_path)
    assert report["release_support_observations"][0]["state"]=="confirmed"
    assert report["acquisition_complete"] is False


def test_motion_headroom_defaults_do_not_mutate_ordinary_settings(settings):
    config = AutomaticCalibrationConfig()
    controller = B601AutomaticCalibrationController(settings, config)
    assert config.motion_velocity_headroom_ratio == 0.
    assert config.motion_braking_timeout_sec is None
    assert not config.motion_continuous_speed_regulation
    assert controller.settings == settings


@pytest.mark.parametrize("value", [None, 1, "true", [], float("nan")])
def test_continuous_acquisition_requires_explicit_boolean(value):
    with pytest.raises(TypeError, match="must be boolean"):
        AutomaticCalibrationConfig(motion_continuous_speed_regulation=value)


def test_continuous_acquisition_rejects_implicit_or_nonlocal_enablement():
    with pytest.raises(ValueError, match="explicit local governed tracking"):
        AutomaticCalibrationConfig(motion_continuous_speed_regulation=True)


@pytest.mark.parametrize("feedback_gain", [0.,40.])
@pytest.mark.parametrize("margin,excursion", [(0., .008), (.0008, .010)])
def test_automatic_tracking_recovery_is_explicit_and_reuses_native_plan(settings,feedback_gain,margin,excursion):
    original = settings
    settings = replace(settings, limits=replace(settings.limits,max_abs_assist_nm=1.),
        staged_assist=replace(settings.staged_assist,max_total_overlay_nm=1.))
    config = AutomaticCalibrationConfig(motion_position_resolution_rad=.0004,
        motion_responsive_tracking_brake=True,motion_breakaway_probe=True,
        motion_arrival_margin_rad=margin, sweep_excursion_rad=excursion,
        motion_tracking_recovery_cap_nm=1.,
        motion_tracking_recovery_velocity_feedback_nm_s_per_rad=feedback_gain)
    replay = AcquisitionReplay(settings, config=config)
    assert original.staged_assist.tracking_recovery_gain_nm_per_rad is None
    assert replay.controller.settings.staged_assist.tracking_overlay_cap_nm == 1.
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    assert result["state"] == "complete",result
    assert len(result["motion"]["completed_legs"]) == 14
    rows = [row["tracking_recovery"] for row in replay.audits if row.get("tracking_recovery")]
    assert rows and all(row["permitted"] for row in rows)
    assert replay.delta == ZERO
    for leg in result["motion"]["completed_legs"]:
        assert leg["final_zero_and_still_confirmed"] is True
        assert abs(leg["target_error_rad"]) <= original.staged_assist.settle_error_rad
    assert original.staged_assist.motion_control.arrival_margin_rad == 0.
    assert result["motion_settings"]["motion_control"]["arrival_margin_rad"] == margin
    assert original.staged_assist.motion_control.max_reference_lag_rad == .013
    assert result["motion_settings"]["motion_control"]["max_reference_lag_rad"] == .013


@pytest.mark.parametrize("value", [0, 1, None, "true"])
def test_automatic_reference_pacing_rejects_non_boolean(value):
    with pytest.raises(TypeError, match="reference pacing"):
        AutomaticCalibrationConfig(motion_reference_pacing=value)


def test_automatic_reference_pacing_requires_explicit_recovery():
    assert AutomaticCalibrationConfig().motion_reference_pacing is False
    with pytest.raises(ValueError, match="responsive tracking recovery"):
        AutomaticCalibrationConfig(motion_reference_pacing=True)


@pytest.mark.parametrize("value", [None, 0, 1, "true"])
def test_automatic_soft_confirmation_rejects_non_boolean(value):
    with pytest.raises(TypeError, match="soft withdrawal"):
        AutomaticCalibrationConfig(motion_soft_withdrawal_confirmation=value)


def test_automatic_soft_confirmation_requires_pacing():
    with pytest.raises(ValueError, match="explicit pacing"):
        AutomaticCalibrationConfig(motion_soft_withdrawal_confirmation=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_native_automatic_soft_withdrawal_quantized_progress_preserves_plan_and_batch(settings, enabled):
    """Paired encoder inputs exercise the real adapter, not physical dynamics."""
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j3_auto_calibration_recovery_v1.json"
    values = json.loads(profile.read_text())
    values.pop("motion_profile")
    values["motion_soft_withdrawal_confirmation"] = enabled
    settings = replace(settings, limits=replace(settings.limits,max_abs_assist_nm=1.),
        staged_assist=replace(settings.staged_assist,max_total_overlay_nm=1.))
    replay = AcquisitionReplay(settings, config=AutomaticCalibrationConfig(**values))
    replay.begin()
    replay.run(until=lambda s: s["stage"] == "motion" and replay.audit.get("phase") == "breakaway")
    anchor = replay.q[2]
    batch_id = replay.controller.result_record()["batch_id"]
    for count in (0,)*25+(1,0,1,1,2,2,3,4,4,5):
        q = list(replay.q)
        q[2] = anchor-count*.0003834962844848633
        replay.tick(q, dt_sec=.01007)
    reasons = [a.get("reason") for a in replay.audits]
    if enabled:
        assert "held_probe_soft_speed_confirmation_pending" in reasons
        assert "held_probe_post_withdrawal_progress_confirmed" in reasons
    else:
        assert "held_probe_soft_speed_withdrawn_unconfirmed" in reasons
    replay.run()
    result = replay.controller.result_record()
    assert result["batch_id"] == batch_id
    if enabled:
        assert result["state"] == "complete", result["motion"]
        assert len(result["motion"]["completed_legs"]) == 14
        assert all(leg["final_zero_and_still_confirmed"] for leg in result["motion"]["completed_legs"])
    else:
        assert result["state"] == "failed"
        assert result["motion"]["reason"] == "owner_terminal_withdrawal:held_probe_soft_speed_withdrawn_unconfirmed"
        assert not result["motion"]["completed_legs"]
    assert replay.delta == ZERO and replay.last_kd == ZERO


@pytest.mark.parametrize("stop", ["resume", "leg_budget", "deadline"])
def test_actual_recovery_profile_paces_post_onset_stall_without_renewing_trial(settings, stop):
    profile = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j3_auto_calibration_recovery_v1.json"
    values = json.loads(profile.read_text())
    values.pop("motion_profile")
    if stop == "leg_budget":
        values["max_sweep_leg_command_exposure_nm_sec"] = .5
    elif stop == "deadline":
        values["max_sweep_leg_sec"] = 4.
    settings = replace(settings, limits=replace(settings.limits,max_abs_assist_nm=1.),
        staged_assist=replace(settings.staged_assist,max_total_overlay_nm=1.))
    replay = AcquisitionReplay(settings, config=AutomaticCalibrationConfig(**values))
    assert settings.staged_assist.motion_control.reference_pacing is False
    assert replay.controller.settings.staged_assist.motion_control.reference_pacing is True
    replay.begin()
    replay.run(until=lambda s: s["stage"] == "motion" and replay.audit.get("phase") == "track")
    original_target = replay.audit["motion_control"]["target_position_rad"]
    original_batch = replay.controller.result_record()["batch_id"]
    start = replay.now
    maximum = 0.
    for _ in range(900):
        state = replay.controller.status()["automatic_calibration"]
        if state["state"] in {"failed", "complete", "cancelled"}:
            break
        if stop == "resume" and maximum > .6:
            q = list(replay.q)
            direction = 1 if original_target > q[2] else -1
            q[2] += direction*min(.00006,abs(original_target-q[2]))
            replay.tick(q)
            if replay.controller.result_record()["motion"]["completed_legs"]:
                break
        else:
            replay.tick()
        maximum = max(maximum,abs(replay.delta[2]))
        motion = replay.audit.get("motion_control")
        if motion and motion["phase"] == "track":
            assert abs(motion["q_reference_rad"]-replay.q[2]) <= .013
            assert motion["target_position_rad"] == original_target
        assert abs(replay.delta[2]) <= 1.
        assert replay.controller.result_record()["batch_id"] == original_batch
    result = replay.controller.result_record()
    if stop == "resume":
        assert maximum > .6
        assert len(result["motion"]["completed_legs"]) == 1, result["motion"]
        assert result["motion"]["completed_legs"][0]["final_zero_and_still_confirmed"]
        assert replay.delta == ZERO and replay.last_kd == ZERO
        replay.run()
        result = replay.controller.result_record()
        assert result["state"] == "complete", result["motion"]
        assert len(result["motion"]["completed_legs"]) == 14
        assert result["motion"]["batch_command_exposure_nm_sec"] > 0.
        assert replay.delta == ZERO and replay.last_kd == ZERO
    else:
        assert result["state"] == "failed", result
        expected = "leg_command_exposure_budget" if stop == "leg_budget" else "leg_timeout"
        assert result["motion"]["reason"] == expected
        assert not result["motion"]["completed_legs"]
        assert replay.now-start < 6.


@pytest.mark.parametrize("margin", [-.001, True, float("nan"), float("inf")])
def test_automatic_arrival_margin_rejects_invalid_values(margin):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(motion_position_resolution_rad=.0004, motion_arrival_margin_rad=margin)


def test_automatic_arrival_margin_requires_resolution_and_fits_tolerance(settings):
    with pytest.raises(ValueError, match="resolution"):
        AutomaticCalibrationConfig(motion_arrival_margin_rad=.0008)
    with pytest.raises(ValueError, match="below final"):
        B601AutomaticCalibrationController(settings, AutomaticCalibrationConfig(
            motion_position_resolution_rad=.0004, motion_arrival_margin_rad=settings.staged_assist.settle_error_rad,
        ))


@pytest.mark.parametrize("gain", [-1,True,float("nan"),float("inf")])
def test_recovery_velocity_feedback_config_rejects_invalid_values(gain):
    with pytest.raises((ValueError,TypeError)):
        AutomaticCalibrationConfig(motion_tracking_recovery_velocity_feedback_nm_s_per_rad=gain)


def test_recovery_velocity_feedback_cannot_enable_itself_outside_explicit_candidate():
    with pytest.raises(ValueError):
        AutomaticCalibrationConfig(motion_tracking_recovery_velocity_feedback_nm_s_per_rad=80.)


@pytest.mark.parametrize("cap", [0,1.01,True,float("nan")])
def test_automatic_tracking_recovery_invalid_cap_rejected_before_owner(cap):
    with pytest.raises((ValueError,TypeError)):
        AutomaticCalibrationConfig(motion_position_resolution_rad=.0004,
            motion_responsive_tracking_brake=True,motion_breakaway_probe=True,
            motion_tracking_recovery_cap_nm=cap)


# Keep every pair of exit cause, governed mode and feedback gain.
@pytest.mark.parametrize("governor,stop,feedback_gain", [
    (False, "human", 0.), (True, "human", 40.),
    (False, "lease", 40.), (True, "lease", 0.),
    (False, "deadline", 0.), (True, "deadline", 40.),
    (False, "exposure", 40.), (True, "exposure", 0.),
])
def test_automatic_tracking_recovery_keeps_original_session_exit_budgets(settings,stop,feedback_gain,governor):
    settings = replace(settings,limits=replace(settings.limits,max_abs_assist_nm=1.),
        staged_assist=replace(settings.staged_assist,max_total_overlay_nm=1.))
    config = AutomaticCalibrationConfig(motion_position_resolution_rad=.0004,
        motion_responsive_tracking_brake=True,motion_breakaway_probe=True,
        motion_tracking_recovery_cap_nm=1.,
        motion_tracking_recovery_velocity_feedback_nm_s_per_rad=feedback_gain,
        motion_reference_pacing=governor,
        motion_onset_handoff="track" if governor else "stop_and_zero",
        motion_speed_governor=governor,
        max_sweep_command_exposure_nm_sec=.5 if stop=="exposure" else 20.)
    replay = AcquisitionReplay(settings,config=config)
    replay.begin()
    replay.run(until=lambda state: state["stage"]=="motion" and replay.audit.get("phase")=="track")
    if stop=="human":
        replay.controller.set_human_intervention(True)
    for _ in range(1800):
        replay.tick(permit=stop!="lease")
        state = replay.controller.status()["automatic_calibration"]
        if state["state"] in {"failed","cancelled","complete"}:
            break
    result = replay.controller.result_record()
    assert result["state"] in {"failed","cancelled"},result
    assert len(result["motion"]["completed_legs"]) == 0
    assert replay.delta == ZERO
    assert replay.controller.status().get("tracking_recovery") is None
    reason = {"human":"human_intervention","lease":"gui_permit_expired",
        "deadline":"leg_timeout","exposure":"active_command_exposure_budget"}[stop]
    assert reason in str(result)


@pytest.mark.parametrize("invalid", [1,"true",None])
def test_held_probe_auto_configuration_requires_explicit_boolean(invalid):
    with pytest.raises(TypeError):
        AutomaticCalibrationConfig(motion_position_resolution_rad=.0004,motion_breakaway_probe=invalid)


def test_resolution_profile_is_automatic_only_and_completes_existing_native_plan(
    settings,
):
    assert settings.staged_assist.motion_control.position_resolution_rad is None
    replay = AcquisitionReplay(
        settings,
        config=AutomaticCalibrationConfig(motion_position_resolution_rad=0.0004),
    )
    assert (
        replay.controller.settings.staged_assist.motion_control.position_resolution_rad
        == 0.0004
    )
    assert settings.staged_assist.motion_control.position_resolution_rad is None
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    assert result["state"] == "complete", result
    assert len(result["motion"]["completed_legs"]) == 14
    assert (
        result["motion_settings"]["motion_control"]["position_resolution_rad"] == 0.0004
    )
    assert replay.delta == ZERO
    assert result["auto_apply"] is False


def test_responsive_profile_completes_ideal_feedback_plan_without_changing_legacy_settings(
    settings,
):
    replay = AcquisitionReplay(
        settings,
        config=AutomaticCalibrationConfig(
            motion_position_resolution_rad=0.0004,
            motion_responsive_tracking_brake=True,
        ),
    )
    assert settings.staged_assist.motion_control.responsive_tracking_brake is False
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    assert result["state"] == "complete", result
    assert len(result["motion"]["completed_legs"]) == 14
    assert (
        result["motion_settings"]["motion_control"]["responsive_tracking_brake"] is True
    )
    assert replay.delta == ZERO


def test_responsive_native_port_brakes_resolved_movement_before_raw_hard_guard(
    settings,
):
    replay = AcquisitionReplay(
        settings,
        config=AutomaticCalibrationConfig(
            motion_position_resolution_rad=0.0004,
            motion_responsive_tracking_brake=True,
        ),
    )
    replay.begin()
    replay.run(
        until=lambda state: (
            state["state"] == "motion" and replay.audit.get("phase") == "track"
        )
    )
    samples = []
    for _ in range(8):
        q = list(replay.q)
        q[2] -= 0.000383496
        replay.tick(q)
        samples.append(dict(replay.audit))
        if replay.audit.get("phase") == "brake":
            break
    assert samples[-1]["motion_control"]["reason"] == "resolved_tracking_speed_braking"
    assert samples[-1]["motion_control"][
        "tracking_speed_brake_limit_rad_s"
    ] == pytest.approx(0.0125)
    assert abs(samples[-1]["qvel_fd_rad_s"][2]) < 0.05
    assert samples[-1]["fault"] is None
    assert samples[-1]["motion_control"]["position_torque_nm"] == 0
    for _ in range(3):
        replay.tick()
    assert replay.delta[2] >= 0


def test_responsive_native_port_does_not_reverse_before_final_conditioner_zero(
    settings,
):
    replay = AcquisitionReplay(
        settings,
        config=AutomaticCalibrationConfig(
            motion_position_resolution_rad=0.0004,
            motion_responsive_tracking_brake=True,
        ),
    )
    replay.begin()
    replay.run(
        until=lambda state: (
            state["state"] == "motion" and replay.audit.get("phase") == "track"
        )
    )
    retained = (0.0, 0.0, -0.01, 0.0, 0.0, 0.0)
    for _ in range(8):
        q = list(replay.q)
        q[2] -= 0.000383496
        replay.tick(q, final_delta=retained)
        if replay.audit.get("phase") == "brake":
            break
    for _ in range(3):
        replay.tick(final_delta=retained)
        assert replay.delta == ZERO
        assert (
            replay.audit["motion_control"]["torque_transition_reason"]
            == "native_zero_pending_before_torque_reversal"
        )
    replay.tick(final_delta=ZERO)
    replay.tick()
    assert replay.delta[2] > 0


@pytest.mark.parametrize("invalid", [0, 1, "yes", None])
def test_responsive_auto_flag_rejects_non_boolean_configuration(invalid):
    with pytest.raises(TypeError, match="boolean"):
        AutomaticCalibrationConfig(
            motion_position_resolution_rad=0.0004,
            motion_responsive_tracking_brake=invalid,
        )


@pytest.mark.parametrize("resolution", [0.0, -0.0004, float("nan"), True, 0.0005])
def test_automatic_resolution_is_validated_before_any_owner_action(
    settings, resolution
):
    with pytest.raises(ValueError):
        B601AutomaticCalibrationController(
            settings,
            AutomaticCalibrationConfig(motion_position_resolution_rad=resolution),
        )


def test_no_motion_is_retained_as_censored_observation(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(onset_move=False, motion_move=False)
    result = replay.controller.result_record()
    assert len(result["onset_trials"]) == 4, result
    assert result["state"] == "failed"
    report = summarize_automatic_acquisition(result)
    assert all(
        t["outcome"] == "cap_reached_without_confirmation" for t in report["trials"]
    )
    assert report["repeatability"] == []
    assert replay.delta == ZERO


def test_manual_targets_are_rejected_and_start_is_explicit(settings):
    replay = AcquisitionReplay(settings)
    with pytest.raises(ValueError):
        replay.controller.submit("set", now_sec=1.2, torque_nm=0.2)
    with pytest.raises(ValueError):
        replay.controller.submit("set_target", now_sec=1.2, target_offset_rad=0.01)
    for _ in range(40):
        replay.tick()
    assert replay.controller.status()["automatic_calibration"]["state"] == "idle"
    assert replay.delta == ZERO


def test_native_receipt_and_json_roundtrip_have_identical_admission(settings):
    replay = AcquisitionReplay(settings)
    native = replay.receipt
    assert native["errors"] == native["failed_joint_names"] == ()
    assert native["sequence"] != native["source_sequence"]
    encoded = json.loads(json.dumps(native))
    assert encoded["errors"] == encoded["failed_joint_names"] == []
    assert complete_calibration_receipt(native, replay.sequence, None)
    assert complete_calibration_receipt(encoded, replay.sequence, None)


@pytest.mark.parametrize("updates,completion_delay,reason", [
    ({"accepted": False}, .0008, "native_dispatch_not_accepted"),
    ({"sent_joint_names": ("joint1",)}, .0008, "incomplete_six_axis_dispatch"),
    ({"errors": ("CAN error",)}, .0008, "native_dispatch_errors_or_failed_joints"),
    ({"source_sequence": -1}, .0008, "dispatch_source_sequence_mismatch"),
    ({"sequence": 1}, .0008, "dispatch_receipt_sequence_not_advancing"),
    ({"mode": "position"}, .0008, "native_command_identity_mismatch"),
    ({"completed_at_sec": 3.}, .0008, "dispatch_completion_timestamp_mismatch"),
    ({"requested_at_sec": 0.}, .0008, "dispatch_request_timestamp_order"),
    ({"requested_at_sec": float("nan")}, .0008, "invalid_dispatch_timestamp"),
    ({}, .040001, "dispatch_completion_deadline_exceeded"),
])
def test_rejected_dispatch_preserves_causal_receipt_without_advancing_ack(
    settings, updates, completion_delay, reason,
):
    replay = AcquisitionReplay(settings)
    replay.begin()
    previous = replay.controller._last_receipt.copy()
    previous_completed = replay.controller._last_completed
    with pytest.raises(ValueError, match=reason):
        replay.tick(receipt_updates=updates, completion_delay_sec=completion_delay)
    controller = replay.controller
    rejection = controller.result_record()["dispatch_rejection"]
    assert rejection["reason"] == reason
    assert rejection["stage"] == "after_dispatch"
    assert rejection["native_send_attempted"] is True
    assert rejection["expected_source_sequence"] == replay.sequence
    assert rejection["previous_receipt_sequence"] == previous["sequence"]
    assert rejection["max_control_gap_sec"] == .04
    assert controller._last_receipt == previous
    assert controller._last_completed == previous_completed
    assert controller._pending_sequence == replay.sequence
    assert controller._cancel_requested == "recording_failed:"+reason
    json.dumps(rejection, allow_nan=False)
    if reason == "dispatch_completion_deadline_exceeded":
        assert rejection["native_receipt"]["accepted"] is True
        assert len(rejection["native_receipt"]["sent_joint_names"]) == 6
        assert rejection["feedback_to_observation_sec"] > .04
    records = controller.drain_records()
    assert sum(r["kind"] == "automatic_calibration_dispatch_rejected" for r in records) == 1
    # Later owner shutdown or secondary failure cannot erase the first cause.
    controller.owner_stopped(now_sec=replay.now+.1, reason="operator_shutdown")
    assert controller.result_record()["dispatch_rejection"] == rejection


@pytest.mark.parametrize("delay,reason", [
    (.040001, "dispatch_start_deadline_exceeded"),
    (-.0001, "dispatch_request_precedes_feedback"),
])
def test_dispatch_start_rejection_occurs_before_native_send(settings, delay, reason):
    replay = AcquisitionReplay(settings)
    receipt_before = replay.receipt
    with pytest.raises(ValueError, match=reason):
        replay.tick(dispatch_start_delay_sec=delay)
    assert replay.receipt is receipt_before
    rejection = replay.controller.result_record()["dispatch_rejection"]
    assert rejection["stage"] == "before_dispatch"
    assert rejection["native_send_attempted"] is False
    assert rejection["native_receipt"] is None


def test_dispatch_deadline_boundary_remains_inclusive(settings):
    replay = AcquisitionReplay(settings)
    replay.tick(dispatch_start_delay_sec=.02, completion_delay_sec=.04)
    assert replay.controller.result_record()["dispatch_rejection"] is None
    assert replay.controller._last_completed == replay.now+.04


def test_incomplete_native_send_without_completion_time_is_preserved(settings):
    replay = AcquisitionReplay(settings)
    replay.tick(ack=False)
    native = dict(replay.receipt, accepted=False, completed_at_sec=None,
                  sent_joint_names=("joint1",), failed_joint_names=("joint2",))
    with pytest.raises(ValueError, match="native_dispatch_not_accepted"):
        replay.controller.record_dispatch(native_receipt=native, completed_at_sec=None)
    rejection = replay.controller.result_record()["dispatch_rejection"]
    assert rejection["native_receipt"] == native
    assert rejection["native_send_attempted"] is True
    assert replay.controller._last_receipt["sequence"] < native["sequence"]


@pytest.mark.parametrize("overrides,reason", [
    ({"receipt_sequence": -1}, "dispatch_receipt_sequence_mismatch"),
    ({"feedback_timestamp_sec": 0.}, "dispatch_feedback_timestamp_mismatch"),
])
def test_dispatch_observer_metadata_must_match_native_receipt(settings, overrides, reason, monkeypatch):
    replay = AcquisitionReplay(settings)
    record = replay.controller.record_dispatch

    def mismatched_observer(**values):
        record(**{**values, **overrides})

    monkeypatch.setattr(replay.controller, "record_dispatch", mismatched_observer)
    with pytest.raises(ValueError, match=reason):
        replay.tick()
    assert replay.controller.result_record()["dispatch_rejection"]["reason"] == reason


def test_live_dispatch_clock_is_sampled_after_acquiring_owner_lock(settings, monkeypatch):
    replay = AcquisitionReplay(settings)
    replay.tick(ack=False)
    controller = replay.controller

    def delayed_clock():
        assert controller._lock._is_owned()
        return replay.now+.040001

    monkeypatch.setattr("control_station.b601_torque_console.automatic_calibration.time.monotonic", delayed_clock)
    with pytest.raises(ValueError, match="dispatch_start_deadline_exceeded"):
        controller.validate_dispatch_start(
            feedback_sequence=replay.sequence, feedback_timestamp_sec=replay.now)
    assert controller.result_record()["dispatch_rejection"]["native_send_attempted"] is False


@pytest.mark.parametrize("field", ["errors", "failed_joint_names"])
@pytest.mark.parametrize("value", [None, "", {}, False, 0, ["failure"], ("failure",)])
def test_receipt_admission_rejects_missing_malformed_and_nonempty_fields(
    settings, field, value
):
    replay = AcquisitionReplay(settings)
    native = dict(replay.receipt)
    native[field] = value
    assert not complete_calibration_receipt(native, replay.sequence, None)
    native.pop(field)
    assert not complete_calibration_receipt(native, replay.sequence, None)


def test_operator_withdraw_waits_for_native_zero_before_terminal(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan")
    for _ in range(30):
        replay.tick()
    replay.controller.submit("withdraw", now_sec=replay.now + 0.001)
    replay.run()
    assert replay.controller.status()["automatic_calibration"]["state"] == "cancelled"
    assert replay.delta == ZERO


@pytest.mark.parametrize("cause", ["recording", "lease", "human"])
def test_recording_lease_and_human_changes_cancel_without_rearming(
    settings, monkeypatch, cause
):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan")
    for _ in range(30):
        replay.tick()
    if cause == "recording":
        replay.controller.recording_failed("disk full")
    elif cause == "human":
        monkeypatch.setattr(
            "robot_motion.rebot_b601_rs.torque_experiment.time.monotonic",
            lambda: replay.now,
        )
        replay.controller.set_human_intervention(True)
    for _ in range(300):
        replay.tick(permit=cause != "lease")
    result = replay.controller.result_record()
    assert result["state"] == "cancelled", result
    assert result["outcome"]["severity"] == ("error" if cause == "recording" else "warning" if cause == "lease" else "info")
    assert replay.delta == ZERO
    assert result["onset_damping"]["last_acknowledged_kd_nm_s_per_rad"] == ZERO
    assert result["onset_damping"]["base_restore_completed_at_sec"] is not None
    with pytest.raises(ValueError):
        replay.controller.submit("arm", now_sec=replay.now)


def test_cumulative_batch_excursion_cancels_without_resetting_anchor(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan")
    # Model a later leg whose fresh local anchor is already near the batch edge.
    replay.controller._anchor = (0.0, 0.0, -0.01995, 0.0, 0.0, 0.0)
    q = list(replay.q)
    q[2] += 0.0001
    replay.tick(q)
    replay.run(onset_move=False, motion_move=False)
    assert replay.controller.result_record()["reason"] == "global_batch_excursion"
    assert replay.delta == ZERO


def test_live_operator_permit_may_follow_feedback_but_not_observation(settings):
    replay = AcquisitionReplay(settings)
    replay.controller.submit("arm", now_sec=replay.now + 0.01005)
    replay.tick()
    assert (
        replay.controller.status()["automatic_calibration"]["state"] == "onset_prepare"
    )
    assert replay.delta == ZERO  # The new gain receipt gate precedes native arm.
    replay.run(until=lambda s: s["state"] == "onset_arm")


def test_report_does_not_invent_onset_force_for_micro_displacement(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    result["onset_trials"][0]["outcome"] = "probe_cumulative_displacement_withdrawn"
    report = summarize_automatic_acquisition(result)
    assert report["trials"][0]["outcome"] == "unconfirmed_micro_displacement"
    assert report["trials"][0]["confirmed_onset_command_nm"] is None
    assert report["onset_data_assessment"]["completed_trial_count"] == 4
    assert report["onset_data_assessment"]["confirmed_encoder_progress_trial_count"] == 3
    assert report["gravity_residual_and_friction_separated"] is False


def test_later_motion_failure_does_not_invalidate_completed_onset_trials(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    result["state"] = "failed"
    result["reason"] = "owner_terminal_withdrawal:held_probe_progress_returned_after_withdrawal"
    result["motion"] = {
        "state": "failed",
        "reason": result["reason"],
        "step_index": 0,
        "step_count": 14,
        "completed_legs": [],
    }

    report = summarize_automatic_acquisition(result)

    assert report["status"] == "ACQUISITION_PARTIAL"
    assert report["acquisition_complete"] is False
    assert report["tracking_arrival_quality"]["assessed_legs"] == 0
    assessment = report["onset_data_assessment"]
    assert assessment["status"] == "EXPLORATORY_RESPONSE_DATA_AVAILABLE"
    assert assessment["completed_trial_count"] == 4
    assert assessment["confirmed_encoder_progress_trial_count"] == 4
    assert assessment["expected_trial_count"] == 4
    assert assessment["scan_set_complete"] is True
    assert assessment["base_damping_restore_confirmed"] is True
    assert assessment["stage_complete"] is True
    assert assessment["later_motion_failure_invalidates_completed_trials"] is False


def test_support_damping_is_bounded_acknowledged_before_onset_and_restored(settings):
    unstarted = B601AutomaticCalibrationController(
        settings, AutomaticCalibrationConfig()
    )
    assert (
        unstarted.result_record()["onset_damping"]["last_acknowledged_kd_nm_s_per_rad"]
        is None
    )
    replay = AcquisitionReplay(settings)
    assert replay.controller.active_driver_damping(ZERO) == ZERO
    replay.begin()
    previous = ZERO
    for _ in range(40):
        replay.tick()
        actual = replay.controller.status()["automatic_calibration"]["support_damping"][
            "last_acknowledged_kd_nm_s_per_rad"
        ]
        assert actual[2] == 0
        assert all(abs(a - b) <= 0.020000001 for a, b in zip(actual, previous))
        if actual != (0.4, 0.6, 0.0, 0.6, 0.2, 0.1):
            assert replay.delta == ZERO
        previous = actual
    replay.run(until=lambda s: s["state"] == "onset_scan")
    assert previous == (0.4, 0.6, 0.0, 0.6, 0.2, 0.1)
    replay.controller.submit("withdraw", now_sec=replay.now + 0.001)
    replay.run()
    assert replay.controller.active_driver_damping(ZERO) == ZERO


def test_support_velocity_fault_is_not_hidden_by_damping(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan")
    for _ in range(30):
        replay.tick()
    replay.controller.submit("withdraw", now_sec=replay.now + 0.001)
    # Same continuous J4 count progression as the real withdrawal failure.
    for counts in (1, 1, 1, 2):
        q = list(replay.q)
        q[3] -= counts * 0.000383496
        replay.tick(q)
    assert replay.audit["phase"] == "fault"
    assert "joint4" in replay.audit["fault"]
    assert "0.050000" in replay.audit["fault"]


def test_missing_gain_receipt_cannot_arm_or_advance_gain_ramp(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.tick(ack=False)
    assert replay.delta == ZERO
    assert (
        replay.controller.status()["automatic_calibration"]["support_damping"][
            "last_acknowledged_kd_nm_s_per_rad"
        ]
        == ZERO
    )
    replay.run(onset_move=False, motion_move=False)
    result = replay.controller.result_record()
    assert result["state"] == "cancelled"
    assert result["onset_trials"] == []
    assert replay.delta == ZERO


def test_mismatched_motor_gain_receipt_is_rejected(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    with pytest.raises(ValueError, match="support damping receipt"):
        replay.tick(kd_override=(0.0, 0.0, 0.1, 0.0, 0.0, 0.0))
    assert replay.controller.result_record()["onset_trials"] == []


def test_support_ramp_timeout_is_rejected_before_owner_ready(settings):
    with pytest.raises(ValueError, match="ramp and settle"):
        B601AutomaticCalibrationController(
            settings, replace(AutomaticCalibrationConfig(), prepare_timeout_sec=0.3)
        )


def test_current_conditioned_tail_retains_support_damping_and_blocks_restoration(
    settings,
):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan")
    replay.controller.submit("withdraw", now_sec=replay.now + 0.001)
    replay.run(until=lambda s: s["state"] == "onset_restore")
    held = replay.last_kd
    for _ in range(15):
        replay.tick(final_delta=(0.0, 0.0, 0.02, 0.0, 0.0, 0.0))
        assert replay.last_kd == held
        assert (
            replay.controller.result_record()["onset_damping"][
                "base_restore_completed_at_sec"
            ]
            is None
        )
    replay.run(onset_move=False, motion_move=False)
    assert replay.controller.result_record()["state"] == "cancelled"
    assert replay.last_kd == ZERO


def test_nonstationary_restore_times_out_without_declaring_base_restored(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run(until=lambda s: s["state"] == "onset_scan")
    replay.controller.submit("withdraw", now_sec=replay.now + 0.001)
    replay.run(until=lambda s: s["state"] == "onset_restore")
    waveform = (
        0.0,
        0.0001,
        0.0002,
        0.0003,
        0.0002,
        0.0001,
        0.0,
        -0.0001,
        -0.0002,
        -0.0003,
        -0.0002,
        -0.0001,
    )
    for i in range(600):
        q = list(replay.q)
        q[3] = waveform[i % len(waveform)]
        replay.tick(q)
        if replay.audit["phase"] == "fault":
            break
    result = replay.controller.result_record()
    assert result["reason"] == "support_damping_restore_timeout", result
    assert result["onset_damping"]["base_restore_completed_at_sec"] is None
    assert replay.delta == ZERO


def test_onset_repeatability_never_merges_different_active_damping(settings):
    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run()
    result = replay.controller.result_record()
    result["onset_trials"][0]["active_driver_kd_nm_s_per_rad"] = ZERO
    report = summarize_automatic_acquisition(result)
    assert sorted(g["count"] for g in report["repeatability"]) == [1, 1, 2]


@pytest.mark.parametrize(
    "changes",
    [
        {"accepted": False},
        {"sent_joint_names": ["joint3"]},
        {"source_sequence": -1},
        {"completed_at_sec": 99.0},
    ],
)
def test_failed_or_mismatched_receipt_does_not_start_next_trial(settings, changes):
    replay = AcquisitionReplay(settings)
    replay.begin()
    with pytest.raises(ValueError):
        replay.tick(receipt_updates=changes)
    assert replay.controller.result_record()["onset_trials"] == []
    replay.run(onset_move=False, motion_move=False)
    assert replay.controller.result_record()["state"] == "cancelled"


def test_automatic_report_is_saved_after_journal_close(settings, tmp_path):
    from control_station.b601_torque_console.automatic_report import (
        save_automatic_report,
    )
    from control_station.b601_torque_console.report import main

    replay = AcquisitionReplay(settings)
    replay.begin()
    replay.run()
    records = [*replay.records, replay.controller.result_record()]
    (tmp_path / "b601_owner.debug.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )
    (tmp_path / "b601_owner.transparency.jsonl").write_text("")
    output = save_automatic_report(tmp_path)
    report = json.loads(output.read_text())
    assert report["status"] == "ACQUISITION_COMPLETE"
    assert report["running_resistance"]["auto_apply"] is False
    assert report["acquisition_complete"] is True
    assert report["acquisition_outcome"]["severity"] == "info"
    assert report["release_support_observations"] == []
    assert report["data_quality_outcome"]["severity"] == "warning"
    assert report["data_quality_outcome"]["scope"] == "model_fit"
    assert report["model_readiness"]["status"] == "INSUFFICIENT_LABELLED_WINDOWS"
    assert report["model_readiness"]["scope"] == "running_resistance_fit_only"
    assert report["model_readiness"]["onset_data_assessed_separately"] is True
    assert report["onset_data_assessment"]["stage_complete"] is True
    assert report["model_readiness"]["offline_candidates"] == 0
    assert report["motion_data_assessment"]["status"] == "NO_MOTION_WINDOWS"
    assert report["motion_data_assessment"]["recorded_command_windows"] == 0
    assert report["motion_data_assessment"]["model_eligible_windows"] == 0
    assert report["tracking_arrival_quality"]["outside_tracking_tolerance_legs"] == 0
    assert main(["--run-dir", str(tmp_path), "--automatic-calibration"]) == 0


def test_automatic_report_retains_first_dispatch_rejection(settings, tmp_path):
    from control_station.b601_torque_console.automatic_report import (
        save_automatic_report,
    )

    replay = AcquisitionReplay(settings)
    replay.begin()
    with pytest.raises(ValueError, match="dispatch_completion_deadline_exceeded"):
        replay.tick(completion_delay_sec=.040001)
    records = [*replay.records, *replay.controller.drain_records(), replay.controller.result_record()]
    (tmp_path / "b601_owner.debug.jsonl").write_text(
        "".join(json.dumps(record)+"\n" for record in records))
    (tmp_path / "b601_owner.transparency.jsonl").write_text("")
    report = json.loads(save_automatic_report(tmp_path).read_text())
    assert report["dispatch_rejection"]["reason"] == "dispatch_completion_deadline_exceeded"
    assert report["dispatch_rejection"]["native_receipt"]["accepted"] is True
    assert report["acquisition_complete"] is False
    assert report["acquisition_outcome"]["severity"] == "error"


@pytest.mark.parametrize("release_braking", [False, True])
def test_gui_auto_mode_has_one_start_no_manual_targets_and_batch_heartbeat(settings, release_braking):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node required for existing GUI contracts")
    replay = AcquisitionReplay(settings, config=AutomaticCalibrationConfig(
        onset_release_braking=release_braking,
        onset_joint_velocity_limit_rad_s=.15 if release_braking else None,
    ))
    status = replay.controller.status()
    status["transparency_assessment"] = {"available": True}
    source = Path("control_station/b601_torque_console/static/app.js").read_text()
    functions = source[
        source.index("function isStaged(") : source.index(
            "function preparationReasonLabel("
        )
    ]
    functions += source[source.index("function preparationReasonLabel("):
        source.index("function onsetCaptureDetails(")]
    script = """
const vm = require('node:vm'); const nodes={};
const c={state:JSON.parse(process.argv[2]),cycleAngleInitialized:false,el:id=>nodes[id] ||= {setAttribute(k,v){this[k]=v}},
  number:(v,n=3)=>Number.isFinite(v)?v.toFixed(n):'—'};
vm.runInNewContext(process.argv[1],c); c.renderAutomatic();
const idle={label:nodes.arm.textContent,disabled:nodes.arm.disabled,hidden:nodes.apply.hidden};
const fixedMode={title:nodes['automatic-mode-title'].textContent,note:nodes['automatic-mode-note'].textContent};
c.state.mode_selection={selected:'fixed_position',ready:false,locked:false};c.renderAutomatic();
const awaitingMode=nodes.arm.disabled;
c.state.mode_selection.ready=true;c.renderAutomatic();const modeReady=nodes.arm.disabled;
c.state.automatic_calibration.active=true;c.state.automatic_calibration.state='onset_scan';
c.state.automatic_calibration.onset_baseline={gate:'continuous_baseline_dwell',dwell_sec:.1,required_dwell_sec:.15};
c.renderAutomatic();const active={disabled:nodes.arm.disabled,label:nodes.phase.textContent};
const pending=nodes['automatic-progress'].textContent;
c.state.automatic_calibration.state='complete';c.renderAutomatic();
const end=nodes.phase.textContent;
c.state.automatic_calibration.stage='motion';
c.state.automatic_calibration.onset_trials=[{outcome:'motion_onset_confirmed'},
  {outcome:'probe_cumulative_displacement_withdrawn'}];
c.state.automatic_calibration.motion={completed_legs:[{tracking_target_reached:false}],
  acquisition_arrival_tolerance_rad:.005,tracking_arrival_tolerance_rad:.003,
  batch_budget_counts_withdrawal:true,batch_command_exposure_nm_sec:3.28,
  leg_command_exposure_nm_sec:0,max_leg_command_exposure_nm_sec:8,max_batch_command_exposure_nm_sec:80};
c.renderAutomatic();
const quality=nodes['automatic-progress'].textContent;
c.state.automatic_calibration.motion_plan='local_identification';
c.state.automatic_calibration.motion.step_count=12;
c.state.motion_control={arrival_stabilization_active:true,arrival_unloading:true,
  arrival_brake_error_rad:.0022,arrival_unload_exit_error_rad:.003};
c.renderAutomatic();const local=nodes['automatic-progress'].textContent;
const deviation=nodes.phase.textContent;
c.state.automatic_calibration.state='failed';
c.state.automatic_calibration.outcome={severity:'warning',scope:'acquisition'};
c.renderAutomatic();const warning=nodes['automatic-progress'].textContent;
const warningLabel=nodes.phase.textContent;
c.state.automatic_calibration.outcome={severity:'error',scope:'acquisition'};
c.renderAutomatic();const fault=nodes['automatic-progress'].textContent;
c.state.automatic_calibration.stage='onset';c.state.automatic_calibration.state='onset_release';
c.state.automatic_calibration.outcome={severity:'info'};
c.state.onset_capture={release_support_observation:{state:'observing',position_range_limit_rad:.000768}};
c.renderAutomatic();const releaseObservation=nodes['automatic-progress'].textContent;
c.state.automatic_calibration.cycle={phase:'positioning_out',origin_qpos_rad:[0,0,.1,0,0,0],
  current_qpos_rad:[0,0,.2,0,0,0],outbound_target_rad:.44906585,command_exposure_nm_sec:2};
c.state.automatic_calibration.state='positioning_out';c.state.automatic_calibration.stage='cycle';
c.state.automatic_calibration.cycle.positioning_control={phase:'brake',reason:'support_velocity_braking',
  target_error_rad:.34,speed_governor_resume_count:1,stop_capture_block_reason:'six_axis_window_velocity'};
c.state.automatic_calibration.cycle.positioning_limits={};
c.renderAutomatic();const cycleText=nodes['automatic-progress'].textContent;const cycleButton=nodes.arm.textContent;
const cycleMode={title:nodes['automatic-mode-title'].textContent,note:nodes['automatic-mode-note'].textContent};
c.state.mode_selection={selected:'supported_cycle',ready:true,locked:false,cycle_angle_deg:20,maximum_cycle_angle_deg:20};c.renderAutomatic();
nodes['cycle-angle'].value='7.5';c.renderAutomatic();
const draftAngle=nodes['cycle-angle'].value;
c.state.mode_selection.locked=true;c.state.mode_selection.cycle_angle_deg=12;c.renderAutomatic();
const angleUi={draft:draftAngle,locked:nodes['cycle-angle'].disabled,value:nodes['cycle-angle'].value};
const selectionUi={visible:!nodes['mode-selector'].hidden,fixed:nodes['mode-fixed-position']['aria-pressed'],
  full:nodes['mode-supported-cycle']['aria-pressed'],locked:nodes['mode-fixed-position'].disabled&&nodes['mode-supported-cycle'].disabled};
const clicks=[];c.intent=(action,extra)=>clicks.push({action,...extra});
vm.runInNewContext(process.argv[3],c);nodes['cycle-angle'].value='21';nodes.arm.onclick();
const invalidAngle={sent:clicks.length,text:nodes['cycle-angle-note'].textContent};
nodes['cycle-angle'].value='12';nodes.arm.onclick();nodes['mode-fixed-position'].onclick();nodes['mode-supported-cycle'].onclick();
console.log(JSON.stringify({idle,active,pending,end,joint:nodes['joint-select'].disabled,
  deviation,quality,local,warning,warningLabel,fault,releaseObservation,cycleText,cycleButton,fixedMode,cycleMode,awaitingMode,modeReady,selectionUi,angleUi,invalidAngle,clicks,
  dataStatus:nodes['automatic-data-status'].textContent}));
"""
    output = subprocess.run(
        [node, "-e", script, functions, json.dumps(status),
         source[source.index('el("arm").onclick'):source.index('el("authorize-breakaway").onclick')]],
        capture_output=True,
        text=True,
        check=True,
    )
    rendered = json.loads(output.stdout)
    assert "0.10 / 0.15 s" in rendered["pending"]
    assert "尚未开始新的起动试次" in rendered["pending"]
    if release_braking:
        assert "0.150 rad/s" in rendered["pending"]
        assert "限速升至 0.400" in rendered["pending"]
    else:
        assert "J3 保持零 Kd" in rendered["pending"]
    assert rendered["idle"] == {
        "label": "开始当前位置小范围采集",
        "disabled": False,
        "hidden": True,
    }
    assert rendered["active"]["disabled"] is True
    assert "观察" in rendered["active"]["label"]
    assert rendered["end"] == "采集批次完成" and rendered["joint"] is True
    assert rendered["deviation"] == "采集完成（含定位偏差）"
    assert "0.00500 rad" in rendered["quality"] and "0.00300 rad" in rendered["quality"]
    assert "3.28 / 80.00" in rendered["quality"]
    assert "2 个已结束试次，其中 1 次确认编码器进展" in rendered["dataStatus"]
    assert "不会回溯作废" in rendered["dataStatus"]
    assert "训练段与整段留出验证分开" in rendered["local"]
    assert "不自动修改补偿参数" in rendered["local"]
    assert "提前制动门 0.00220 rad，位置到位门 0.00300 rad" in rendered["local"]
    assert rendered["warningLabel"] == "采集告警：本批未完成"
    assert "不是已确认的硬件故障" in rendered["warning"]
    assert "不会自动重试" in rendered["warning"]
    assert "不能按普通采集告警忽略" in rendered["fault"]
    assert "保持零助力与阶段阻尼" in rendered["releaseObservation"]
    assert "不恢复原加力" in rendered["releaseObservation"]
    assert "缓慢移动 J3 至测试位置" in rendered["cycleText"]
    assert "J3 起点 0.10000 / 当前 0.20000" in rendered["cycleText"]
    assert "失能" in rendered["cycleButton"]
    assert "模式一" in rendered["fixedMode"]["title"]
    assert "不做抬升搬运" in rendered["fixedMode"]["note"]
    assert "不自动返回旧原位或自动失能" in rendered["fixedMode"]["note"]
    assert "模式二" in rendered["cycleMode"]["title"]
    assert "支撑起点" in rendered["cycleMode"]["note"]
    assert "support_velocity_braking" in rendered["cycleText"]
    assert "软制动续行次数 1" in rendered["cycleText"]
    assert "距目标 0.34000" in rendered["cycleText"]
    assert rendered["awaitingMode"] is True and rendered["modeReady"] is False
    assert rendered["selectionUi"] == {"visible": True, "fixed": "false", "full": "true", "locked": True}
    assert rendered["angleUi"] == {"draft": "7.5", "locked": True, "value": "12"}
    assert rendered["invalidAngle"]["sent"] == 0
    assert "尚未提交驱动" in rendered["invalidAngle"]["text"]
    assert rendered["clicks"] == [
        {"action": "arm", "operation_mode": "supported_cycle", "cycle_angle_deg": 12},
        {"action": "select_mode", "operation_mode": "fixed_position"},
        {"action": "select_mode", "operation_mode": "supported_cycle"},
    ]
    # Preparation/release must retain the external lease even when native arm is off.
    assert (
        "state.automatic_calibration.active || state.automatic_calibration.start_requested"
        in source
    )


@pytest.mark.parametrize("profile_name", ["b601_j3_auto_calibration_v1.json", "b601_j3_auto_calibration_recovery_v1.json", "b601_j3_continuous_motion_v1.json", "b601_j3_local_identification_v1.json"])
def test_auto_profile_parent_child_roundtrip_is_connection_free(tmp_path, monkeypatch, profile_name):
    profile = (
        Path(__file__).resolve().parents[1]
        / "docs/hardware/profiles" / profile_name
    )
    args = owner._parser().parse_args(
        [
            "--allow-motion",
            "--gravity-debug-mode",
            "--torque-console-auto-calibration-profile",
            str(profile),
            "--rate-hz",
            "100",
            "--torque-console-max-assist-nm",
            "1.0",
            "--torque-console-staged-target-offset-limit-rad",
            ".09" if profile_name == "b601_j3_local_identification_v1.json" else ".065",
            "--torque-console-max-joint-velocity-rad-s",
            "0.195" if profile_name == "b601_j3_local_identification_v1.json"
            else "0.10" if profile_name == "b601_j3_continuous_motion_v1.json" else "0.05",
            "--torque-console-confirmation",
            "B601_RS_TORQUE_CONSOLE",
            "--torque-console-staged-motion-onset-rad",
            ".0004",
            "--torque-console-staged-breakaway-duration-sec",
            "1.2",
            "--torque-console-staged-breakaway-impulse-nm-sec",
            ".6",
        ]
    )
    parsed = probe.build_parser().parse_args(
        owner.build_probe_command(args, run_dir=tmp_path)[2:]
    )
    probe._validate_cli(parsed)
    controller = probe._automatic_calibration_controller_from_args(parsed)
    assert controller.controller_mode == "automatic_calibration"
    assert parsed.kp == parsed.kd == 0
    assert tuple(parsed.driver_kd) == ZERO
    assert tuple(parsed.gravity_joint_scales) == (1.0, 0.96, 1.0, 1.0, 1.0, 1.0)
    assert tuple(parsed.gravity_joint_bias_nm) == (0.0, -0.22, -0.11, -0.08, 0.0, 0.0)
    assert controller.settings.staged_assist.motion_control.windowed_soft_velocity
    assert controller.settings.staged_assist.motion_control.position_resolution_rad == .0004
    assert controller.settings.staged_assist.motion_control.responsive_tracking_brake is True
    candidate = profile_name != "b601_j3_auto_calibration_v1.json"
    identification = profile_name == "b601_j3_local_identification_v1.json"
    assert controller.config.motion_continuous_speed_regulation is identification
    assert controller.settings.staged_assist.motion_control.continuous_speed_limit_rad_s == (
        .0585 if identification else None)
    continuous = profile_name == "b601_j3_continuous_motion_v1.json" or identification
    hard_speed = .195 if identification else .10 if continuous else .05
    assert controller.limits.max_joint_velocity_rad_s == hard_speed
    assert controller.settings.maximum_target_offset_rad == (.09 if identification else .065)
    assert controller.config.motion_soft_velocity_rad_s == (.0585 if identification else None)
    assert controller.settings.staged_assist.motion_control.max_reference_lag_rad == (.0195 if identification else .013)
    assert controller.config.onset_release_support_confirmation is continuous
    assert controller._direct.onset_capture.release_support_confirmation is continuous
    assert controller.config.feedback_velocity_timebase == (
        "joint_receive" if continuous else "snapshot")
    assert controller.status()["automatic_calibration"]["feedback_velocity_timebase"] == controller.config.feedback_velocity_timebase
    assert controller.config.motion_plan == ("local_identification" if identification else "motion_check" if continuous else "model_sweeps")
    assert controller.settings.staged_assist.onset_handoff == ("track" if continuous else "stop_and_zero")
    assert controller.settings.staged_assist.motion_control.braking_slew_nm_s == (10. if continuous else 2.)
    assert controller.settings.staged_assist.motion_control.speed_governor is continuous
    assert controller.settings.staged_assist.motion_control.arrival_stabilization is continuous
    assert controller.settings.staged_assist.motion_control.velocity_headroom_ratio == (.30 if continuous else 0.)
    assert controller.settings.staged_assist.motion_control.braking_timeout_sec == (6.591 if continuous else 3.9)
    assert controller.config.max_batch_sec == (390. if continuous else 300. if candidate else 150.)
    assert controller.settings.staged_assist.tracking_recovery_velocity_feedback_nm_s_per_rad == (
        10. if continuous else 40. if candidate else 0.)
    assert controller.status()["automatic_calibration"]["planned_motion_steps"] == (12 if identification else 2 if continuous else 14)
    assert controller.config.motion_reference_pacing is candidate
    assert controller.config.motion_soft_withdrawal_confirmation is candidate
    assert controller.settings.staged_assist.breakaway_soft_withdrawal_confirmation is candidate
    assert controller.settings.staged_assist.breakaway_probe.observation_limit_sec == (.78 if continuous else .6 if candidate else .2)
    assert controller.config.prepare_timeout_sec == (13. if continuous else 10. if candidate else 5.)
    assert controller.config.onset_settle_timeout_sec == (3.9 if continuous else 3. if candidate else 1.)
    assert controller.config.probe_max_displacement_rad == (.010 if candidate else .003)
    assert controller.config.sweep_excursion_rad == (
        .035 if identification else .020 if candidate else .008
    )
    assert controller.config.global_excursion_rad == (.040 if candidate else .02)
    if identification:
        # Leave enough planned travel beyond the intentionally broad arrival
        # tolerance to observe motion, while keeping the target inside the
        # existing global-excursion guard.
        assert controller.config.sweep_excursion_rad - controller.config.motion_acquisition_arrival_tolerance_rad == pytest.approx(.02148)
        assert controller.config.global_excursion_rad - controller.config.sweep_excursion_rad == pytest.approx(.005)
    assert controller.settings.staged_assist.motion_control.reference_pacing is candidate
    assert controller.config.onset_release_braking is candidate
    assert controller.config.onset_joint_velocity_limit_rad_s == (.15 if candidate else None)
    assert controller.config.motion_acquisition_arrival_tolerance_rad == (.01352 if identification else .00845 if profile_name == "b601_j3_continuous_motion_v1.json" else .0065 if candidate else None)
    assert controller.config.max_sweep_leg_command_exposure_nm_sec == (16. if continuous else 8. if candidate else None)
    assert controller.config.max_sweep_leg_sec == (39. if continuous else 20. if candidate else 12.)
    assert controller.config.max_sweep_command_exposure_nm_sec == (80. if candidate else 20.)
    assert controller.settings.staged_assist.settle_error_rad == pytest.approx(
        .01352 if identification else .00845 if continuous else .0039
    )
    assert controller.config.motion_tracking_arrival_tolerance_rad == (.01352 if identification else .00845 if continuous else None)
    monkeypatch.setattr(probe, "_resolve_model_and_limits", lambda source: (
        SimpleNamespace(urdf_path=tmp_path / "test.urdf"), ((-1., 1.),) * 6,
    ))
    monkeypatch.setattr(probe, "_resolve_effort_limits", lambda *args, **kwargs: (1.,) * 6)
    monkeypatch.setattr(probe.B601RSBackendConfig, "from_rs_yaml", classmethod(
        lambda cls, *args, **kwargs: SimpleNamespace(channel="can0", joints=[SimpleNamespace(name=j) for j in B601_RS_ARM_JOINT_NAMES])
    ))
    monkeypatch.setitem(sys.modules, "motorbridge", SimpleNamespace(abi_capabilities=lambda: {
        "features": {"robstride": ["parameter_reply_correlation_v1", "state_observation_metadata_v1"]},
    }))
    preflight = probe._preflight_probe(parsed)
    assert preflight["motion_settings"]["limits"]["max_joint_velocity_rad_s"] == hard_speed
    assert preflight["motion_settings"]["staged_assist"]["motion_control"]["position_resolution_rad"] == .0004
    assert preflight["motion_settings"]["staged_assist"]["motion_control"]["responsive_tracking_brake"] is True
    assert preflight["motion_settings"]["staged_assist"]["motion_control"]["reference_pacing"] is candidate
    assert "max_speed_governor_resumptions" not in preflight["motion_settings"]["staged_assist"]["motion_control"]
    assert preflight["motion_settings"]["staged_assist"]["motion_control"]["velocity_headroom_ratio"] == (.30 if continuous else 0.)
    assert preflight["motion_settings"]["staged_assist"]["motion_control"]["braking_timeout_sec"] == (6.591 if continuous else 3.9)
    assert preflight["can_opened"] is preflight["motor_command_sent"] is False


@pytest.mark.parametrize("value", [0., -1., True, float("nan"), float("inf")])
def test_tracking_arrival_override_requires_positive_finite_value(value):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(motion_tracking_arrival_tolerance_rad=value)


def test_tracking_arrival_override_preserves_default_settings_and_existing_bounds(settings):
    assert AutomaticCalibrationConfig().motion_tracking_arrival_tolerance_rad is None
    original = settings.staged_assist.settle_error_rad
    config = AutomaticCalibrationConfig(motion_tracking_arrival_tolerance_rad=.005,
        sweep_excursion_rad=.010, motion_acquisition_arrival_tolerance_rad=.005)
    controller = B601AutomaticCalibrationController(settings, config)
    assert controller.settings.staged_assist.settle_error_rad == .005
    assert original == pytest.approx(.0039)
    assert settings.staged_assist.settle_error_rad == original
    assert controller.settings.staged_assist.motion_control == settings.staged_assist.motion_control
    assert controller.limits == settings.limits
    with pytest.raises(ValueError, match="target envelope fraction"):
        B601AutomaticCalibrationController(settings, replace(config, motion_tracking_arrival_tolerance_rad=.02))
    with pytest.raises(ValueError, match="smaller than the tracking tolerance"):
        B601AutomaticCalibrationController(settings, replace(config, motion_acquisition_arrival_tolerance_rad=.004))


def test_local_identification_reuses_the_verified_continuous_drive_profile():
    root = Path(__file__).resolve().parents[1] / "docs/hardware/profiles"
    continuous = json.loads((root / "b601_j3_continuous_motion_v1.json").read_text())
    identification = json.loads((root / "b601_j3_local_identification_v1.json").read_text())
    assert continuous.pop("motion_plan") == "motion_check"
    assert identification.pop("motion_plan") == "local_identification"
    assert identification.pop("sweep_excursion_rad") == .035
    assert continuous.pop("sweep_excursion_rad") == .020
    assert identification.pop("motion_continuous_speed_regulation") is True
    assert identification.pop("motion_soft_velocity_rad_s") == .0585
    assert identification.pop("motion_max_reference_lag_rad") == .0195
    for key in ("motion_tracking_arrival_tolerance_rad", "motion_acquisition_arrival_tolerance_rad"):
        assert identification.pop(key) == .01352
        assert continuous.pop(key) == .00845
    assert "motion_continuous_speed_regulation" not in continuous
    assert identification == continuous


@pytest.mark.parametrize("value", [0., -1., True, float("nan"), float("inf")])
def test_acquisition_soft_speed_override_rejects_invalid_values(value):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(motion_soft_velocity_rad_s=value)


def test_acquisition_soft_speed_override_is_explicit_and_below_hard_guard(settings):
    assert AutomaticCalibrationConfig().motion_soft_velocity_rad_s is None
    with pytest.raises(ValueError, match="explicit continuous acquisition"):
        AutomaticCalibrationConfig(motion_soft_velocity_rad_s=.039)
    config = AutomaticCalibrationConfig(
        motion_plan="local_identification", motion_breakaway_probe=True,
        motion_position_resolution_rad=.0004, motion_responsive_tracking_brake=True,
        motion_reference_pacing=True, motion_onset_handoff="track", motion_speed_governor=True,
        motion_tracking_recovery_cap_nm=settings.limits.max_abs_assist_nm,
        motion_tracking_recovery_velocity_feedback_nm_s_per_rad=10.,
        motion_continuous_speed_regulation=True, motion_soft_velocity_rad_s=.039)
    original = settings.staged_assist.motion_control.soft_velocity_rad_s
    controller = B601AutomaticCalibrationController(settings, config)
    assert controller.settings.staged_assist.motion_control.soft_velocity_rad_s == .039
    assert settings.staged_assist.motion_control.soft_velocity_rad_s == original == .03
    assert controller.limits == settings.limits
    assert AutomaticCalibrationConfig().motion_max_reference_lag_rad is None
    with pytest.raises(ValueError, match="explicit continuous acquisition"):
        AutomaticCalibrationConfig(motion_max_reference_lag_rad=.0195)
    widened = B601AutomaticCalibrationController(settings, replace(config, motion_max_reference_lag_rad=.0195))
    assert widened.settings.staged_assist.motion_control.max_reference_lag_rad == .0195
    assert settings.staged_assist.motion_control.max_reference_lag_rad == .013
    for value in (0., -1., True, float("nan"), float("inf"), .0129,
                  settings.maximum_target_offset_rad+.001):
        with pytest.raises((ValueError, TypeError)):
            B601AutomaticCalibrationController(settings, replace(config, motion_max_reference_lag_rad=value))
    for value in (.029, settings.limits.max_joint_velocity_rad_s):
        with pytest.raises(ValueError, match="original soft floor"):
            B601AutomaticCalibrationController(settings, replace(config, motion_soft_velocity_rad_s=value))


def test_resolution_acquisition_motion_keeps_support_raw_hard_velocity_guard(settings):
    replay = AcquisitionReplay(
        settings, config=AutomaticCalibrationConfig(motion_position_resolution_rad=.0004)
    )
    replay.begin()
    replay.run(until=lambda state: state["state"] == "motion" and replay.audit.get("phase") == "cue")
    q = list(replay.q)
    q[3] += .0008
    replay.tick(q)
    assert "joint_velocity_exceeded: joint4" in replay.audit["fault"]
    assert replay.delta == ZERO


def test_unreachable_scan_cap_rejected_before_connection(settings):
    with pytest.raises(ValueError, match="unreachable"):
        B601AutomaticCalibrationController(
            replace(settings, limits=replace(settings.limits, max_abs_assist_nm=1.5)),
            AutomaticCalibrationConfig(),
        )


@pytest.mark.parametrize("options", [
    {"motion_acquisition_arrival_tolerance_rad": .005},
    {"sweep_excursion_rad": .010, "motion_acquisition_arrival_tolerance_rad": .006},
    {"motion_acquisition_arrival_tolerance_rad": float("nan")},
    {"motion_acquisition_arrival_tolerance_rad": True},
    {"max_sweep_leg_command_exposure_nm_sec": 21.},
    {"max_sweep_leg_command_exposure_nm_sec": -1.},
    {"max_batch_sec": 5., "max_sweep_leg_sec": 20.},
])
def test_acquisition_configuration_requires_explicit_finite_plan_and_budget_bounds(options):
    with pytest.raises((TypeError, ValueError)):
        AutomaticCalibrationConfig(**options)


@pytest.mark.parametrize("options", [
    {"motion_acquisition_arrival_tolerance_rad": .002},
    {"max_sweep_leg_command_exposure_nm_sec": .01},
])
def test_acquisition_connection_free_validation_keeps_tracking_and_withdrawal_envelopes(settings, options):
    with pytest.raises(ValueError):
        B601AutomaticCalibrationController(settings, AutomaticCalibrationConfig(**options))


def test_acquisition_tolerance_leaves_the_next_actual_anchor_inside_original_envelope(settings):
    settings = replace(settings, maximum_target_offset_rad=.020)
    with pytest.raises(ValueError, match="next leg's actual anchor"):
        B601AutomaticCalibrationController(settings, AutomaticCalibrationConfig(
            sweep_excursion_rad=.010, motion_acquisition_arrival_tolerance_rad=.005,
        ))


@pytest.mark.parametrize("reached,quality", [(False, "TARGET_DEVIATION_RECORDED"), (None, "NOT_FULLY_ASSESSED")])
def test_acquisition_completion_is_not_precision_or_model_validation(reached, quality):
    # A report contract fixture, not a successful physical experiment.
    leg = {} if reached is None else {"tracking_target_reached": reached}
    report = summarize_automatic_acquisition({
        "schema": "b601.automatic-joint-calibration.v1", "state": "complete",
        "onset_trials": [], "motion": {"completed_legs": [leg]},
    })
    assert report["acquisition_complete"] is True
    assert report["tracking_arrival_quality"]["status"] == quality
    assert report["auto_apply"] is False
    assert report["gravity_residual_and_friction_separated"] is False


def _onset_release_replay(settings, *, braking=True):
    replay = AcquisitionReplay(settings, config=AutomaticCalibrationConfig(
        onset_release_braking=braking,
        onset_joint_velocity_limit_rad_s=.15 if braking else None,
    ))
    replay.begin()
    replay.run(until=lambda state: state["state"] == "onset_release")
    assert replay.audit["phase"] == "onset_release"
    assert replay.delta == ZERO
    return replay


def test_onset_release_braking_is_zero_spring_bounded_and_restored_before_next_scan(settings):
    replay = _onset_release_replay(settings)
    gains = [replay.last_kd[2]]
    # Reproduce the three-count / 12.34 ms post-zero sample from 4oRIHu.
    q = list(replay.q)
    q[2] += .0011504888534545898
    replay.tick(q, dt_sec=.01234312099404633)
    assert replay.audit["phase"] == "onset_release"
    assert replay.delta == ZERO
    gains.append(replay.last_kd[2])
    for _ in range(40):
        replay.tick()
        assert replay.audit["phase"] != "fault"
        assert replay.delta == ZERO
        gains.append(replay.last_kd[2])
        if replay.controller.status()["automatic_calibration"]["state"] == "onset_arm":
            break
    assert replay.controller.status()["automatic_calibration"]["completed_onset_trials"] == 1
    assert max(gains) > 0 and max(gains) <= .4
    assert all(abs(a-b) <= .024687 for a,b in pairwise(gains))
    replay.run(onset_move=False, until=lambda state: state["state"] == "onset_scan")
    assert replay.last_kd[2] == 0
    trial = replay.controller.result_record()["onset_trials"][0]
    assert trial["active_driver_kd_nm_s_per_rad"][2] == 0
    # The sweep/staged path still has the original hard limit.
    assert replay.controller.settings.limits.max_joint_velocity_rad_s == .05


@pytest.mark.parametrize("braking,index,step,limit", [
    (False, 2, .0011504888534545898, "0.050000"),
    (True, 2, .00192, "0.150000"),
    (True, 3, .000767, "0.050000"),
])
def test_onset_velocity_relaxation_is_selected_joint_only_and_still_hard(settings, braking, index, step, limit):
    replay = _onset_release_replay(settings, braking=braking)
    q = list(replay.q)
    q[index] += step
    replay.tick(q, dt_sec=.01234312099404633 if not braking else .01)
    assert replay.audit["phase"] == "fault"
    assert f"joint{index+1}" in replay.audit["fault"]
    assert limit in replay.audit["fault"]
    assert replay.delta == ZERO


@pytest.mark.parametrize("braking", [False, True])
def test_onset_scan_velocity_override_does_not_skip_the_original_baseline(settings, braking):
    replay = AcquisitionReplay(settings, config=AutomaticCalibrationConfig(
        onset_release_braking=braking,
        onset_joint_velocity_limit_rad_s=.15 if braking else None,
    ))
    replay.begin()
    replay.run(onset_move=False, until=lambda state: state["state"] == "onset_scan")
    for _ in range(30):
        replay.tick()
        assert replay.last_kd[2] == 0
    assert replay.delta[2] != 0
    q = list(replay.q)
    q[2] -= .0011504888534545898
    replay.tick(q, dt_sec=.01234312099404633)
    assert (replay.audit["phase"] == "fault") is (not braking)
    assert replay.last_kd[2] <= .024687


def test_onset_release_braking_preserves_settle_deadline_and_feedback_watchdog(settings):
    replay = _onset_release_replay(settings)
    for _ in range(120):
        q = list(replay.q)
        q[2] += .0001
        replay.tick(q)
        assert replay.delta == ZERO
        if replay.audit["phase"] == "fault":
            break
    assert replay.audit["fault"] == "onset_capture_settle_timeout"
    assert replay.controller.result_record()["onset_trials"] == []
    replay = _onset_release_replay(settings)
    replay.tick(dt_sec=.041)
    assert replay.audit["fault"] == "velocity_debug_feedback_gap_exceeded"
    assert replay.delta == ZERO


@pytest.mark.parametrize("change", ["human", "withdraw", "permit"])
def test_onset_release_braking_cannot_increase_after_authority_withdrawal(settings, change):
    replay = _onset_release_replay(settings)
    for _ in range(3):
        q = list(replay.q)
        q[2] += .000383496
        replay.tick(q)
    previous = replay.last_kd[2]
    if change == "human":
        replay.controller.set_human_intervention(True)
    elif change == "withdraw":
        replay.controller.submit("withdraw", now_sec=replay.now + .001)
    elif change == "permit":
        # Expire the existing lease without a feedback gap or renewal.
        for _ in range(36):
            replay.tick(permit=False)
        previous = replay.last_kd[2]
    replay.tick(permit=change != "permit")
    assert replay.delta == ZERO
    assert replay.last_kd[2] <= previous
    replay.run(onset_move=False)
    assert replay.controller.result_record()["state"] == "cancelled"
    assert replay.last_kd == ZERO


def test_onset_release_braking_requires_current_final_zero_and_native_gain_receipt(settings):
    replay = _onset_release_replay(settings)
    previous = replay.last_kd
    tail = (0., 0., .02, 0., 0., 0.)
    # Reappearing propulsion after final zero remains a hard dispatch fault.
    with pytest.raises(ValueError, match="zero command cannot reappear"):
        replay.tick(final_delta=tail)
    assert replay.delta == ZERO
    assert replay.last_kd == previous
    assert replay.controller.active_driver_damping(ZERO, final_command_delta_nm=None) == previous
    replay = _onset_release_replay(settings)
    with pytest.raises(ValueError, match="support damping receipt"):
        replay.tick(kd_override=(.4, .6, .5, .6, .2, .1))


@pytest.mark.parametrize("options", [
    {"onset_joint_velocity_limit_rad_s": .15},
    {"onset_release_braking": True, "onset_joint_velocity_limit_rad_s": .151},
    {"onset_release_braking": True, "onset_joint_velocity_limit_rad_s": .049},
    {"onset_release_braking": True, "onset_joint_velocity_limit_rad_s": float("nan")},
    {"onset_release_braking": 1},
])
def test_onset_relaxation_requires_bounded_explicit_release_braking(options):
    with pytest.raises((ValueError, TypeError)):
        AutomaticCalibrationConfig(**options)


@pytest.mark.parametrize("gain", [0., .5])
def test_onset_release_braking_rejects_unestablished_selected_motor_gain(settings, gain):
    motion = replace(settings.staged_assist.motion_control,
                     drive_driver_kd_nm_s_per_rad=(.4, .6, gain, .6, .2, .1))
    settings = replace(settings, staged_assist=replace(settings.staged_assist, motion_control=motion))
    with pytest.raises(ValueError, match="existing J3 drive Kd"):
        B601AutomaticCalibrationController(settings, AutomaticCalibrationConfig(onset_release_braking=True))

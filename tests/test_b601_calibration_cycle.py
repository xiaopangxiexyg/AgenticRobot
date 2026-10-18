"""Prescribed encoder feedback through native positioning and acquisition."""

from __future__ import annotations

import io
import json
import math
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest
from test_b601_automatic_calibration import AcquisitionReplay
from test_b601_calibration_session import settings as shared_settings

from control_station.b601_torque_console import B601TorqueConsoleServer
from control_station.b601_torque_console.automatic_calibration import (
    AutomaticCalibrationConfig,
    B601AutomaticCalibrationController,
)
from control_station.b601_torque_console.calibration_cycle import (
    B601AutomaticCalibrationCycleController,
    CalibrationCycleConfig,
)
from control_station.b601_torque_console.mode_controller import (
    B601CalibrationModeController,
)
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

ZERO = (0.0,) * 6


@pytest.fixture
def settings(tmp_path):
    return shared_settings.__wrapped__(tmp_path)


def replay_cycle(settings, *, config_values=None, gui=False, **cycle_values):
    config = AutomaticCalibrationConfig(
        motion_plan="local_identification",
        motion_acquisition_arrival_tolerance_rad=0.0039,
        **(config_values or {}),
    )
    cycle = CalibrationCycleConfig(**cycle_values)
    base = B601AutomaticCalibrationController(settings, config)
    controller = B601AutomaticCalibrationCycleController(base, cycle)
    if gui:
        controller = B601CalibrationModeController(controller)
    return AcquisitionReplay(settings, controller=controller)


def advance(replay, *, ticks=40000, until=None, permit=True, move=True):
    for _ in range(ticks):
        state = replay.controller.status()
        batch = state["automatic_calibration"]
        if until is not None and until(state):
            return
        if batch["state"] in {"complete", "failed", "cancelled"}:
            return
        q = list(replay.q)
        if move:
            if batch["state"] == "onset_scan" and abs(replay.delta[2]) >= 0.18:
                q[2] += batch["direction"] * 0.0001
            motion = replay.audit.get("motion_control")
            if motion and motion["phase"] == "track":
                q[2] = motion["q_reference_rad"]
            elif state.get("target_offset_rad") and (
                replay.audit.get("phase") == "breakaway"
                or replay.audit.get("phase") == "releasing"
                and not replay.audit.get("release_terminal")
                and abs(q[2] - replay.audit["anchor_rad"]) < 0.001
            ):
                q[2] += 0.0001 if state["target_offset_rad"] > 0 else -0.0001
        replay.tick(q, permit=permit)
    pytest.fail(str(replay.controller.status()["automatic_calibration"]["cycle"]))


@pytest.mark.parametrize("offset", [0.02, math.radians(20)])
def test_native_cycle_positions_observes_runs_twelve_legs_returns_and_waits_for_gui_disable_confirmation(
    settings, offset
):
    replay = replay_cycle(
        settings,
        offset_rad=offset,
        supported_origin_disable_confirmation=True,
        gravity_observation_sec=0.1,
    )
    replay.begin()
    advance(replay)
    result = replay.controller.result_record()
    assert result["cycle"]["phase"] == "complete", result["cycle"]["reason"]
    assert result["state"] == "complete"
    assert len(result["onset_trials"]) == 4
    assert len(result["motion"]["completed_legs"]) == 12
    assert len(result["cycle"]["moves"]) == 2
    assert len(result["cycle"]["gravity_observations"]) == 2
    assert replay.controller.cycle_disable_confirmation_ready
    assert replay.delta == ZERO and replay.last_kd == ZERO
    assert abs(replay.q[2]) <= 0.0039
    assert result["cycle"]["outbound_target_rad"] == offset
    assert result["cycle"]["command_exposure_nm_sec"] > 0
    assert all(
        r["zero_assist_and_base_damping_confirmed"]
        for r in result["cycle"]["gravity_observations"]
    )
    stop = {"requested": False, "full_disable_requested": False}
    confirmation = {"confirmed": False, "confirmed_at_monotonic_sec": None, "source": None}
    assert stop == {"requested": False, "full_disable_requested": False}
    assert result["cycle"]["disable_confirmation_available"] is True
    # Completion alone must not issue the stop. Only the explicit confirmation
    # path requests full disable and leaves a record of its source/time.
    probe._confirm_completed_cycle_disable(
        replay.controller, stop, confirmation, confirmed_at_monotonic_sec=123.0
    )
    assert stop == {"requested": True, "full_disable_requested": True}
    assert confirmation == {
        "confirmed": True,
        "confirmed_at_monotonic_sec": 123.0,
        "source": "local_gui",
    }
    with pytest.raises(ValueError, match="already consumed"):
        probe._confirm_completed_cycle_disable(
            replay.controller, stop, confirmation, confirmed_at_monotonic_sec=124.0
        )
    replay.controller.owner_stopped(now_sec=replay.now, reason="confirmed_disable")
    assert replay.controller.result_record()["cycle"]["disable_confirmation_available"] is True


def test_completed_cycle_disable_confirmation_rejects_early_request(settings):
    replay = replay_cycle(
        settings,
        offset_rad=0.02,
        supported_origin_disable_confirmation=True,
    )
    stop = {"requested": False, "full_disable_requested": False}
    confirmation = {"confirmed": False, "confirmed_at_monotonic_sec": None, "source": None}
    with pytest.raises(ValueError, match="not complete"):
        probe._confirm_completed_cycle_disable(
            replay.controller, stop, confirmation, confirmed_at_monotonic_sec=1.0
        )
    assert stop == {"requested": False, "full_disable_requested": False}
    assert confirmation["confirmed"] is False


@pytest.mark.parametrize("cancel", [False, True])
def test_cycle_records_round_trip_through_owner_trace_writer(settings, cancel):
    replay = replay_cycle(settings, offset_rad=math.radians(20), gravity_observation_sec=0.1)
    stream = io.StringIO()
    trace = probe._DynamicsTrace(stream)

    def drain():
        # AcquisitionReplay already drains the real controller each tick.
        records = [*replay.records, *replay.controller.drain_records()]
        replay.records.clear()
        return records

    queued = SimpleNamespace(drain_records=drain)
    replay.begin()
    advance(replay, until=lambda s: abs(replay.delta[2]) > 0.1)
    # Exercise the real writer both online and after owner shutdown, rather
    # than validating the GUI/status representation of these same events.
    probe._flush_automatic_calibration_records(queued, trace)
    if not cancel:
        advance(replay)
    replay.controller.owner_stopped(now_sec=replay.now, reason="test_shutdown")
    probe._flush_automatic_calibration_records(queued, trace)
    before = stream.getvalue()
    probe._flush_automatic_calibration_records(queued, trace)
    assert stream.getvalue() == before
    assert trace.error is None
    records = [json.loads(line) for line in before.splitlines()]
    transitions = [r for r in records if r["kind"] == "automatic_calibration_cycle_transition"]
    assert transitions
    assert transitions[-1]["phase"] == ("cancelled" if cancel else "complete")
    assert all(r["transition_observed_at_sec"] == r["time_sec"] for r in transitions)
    assert all(math.isfinite(r["transition_observed_at_sec"]) for r in transitions)
    if not cancel:
        assert any(r["kind"] == "automatic_calibration_substage_result" for r in records)


@pytest.mark.parametrize("reason", ["human", "withdraw", "lease"])
def test_cycle_cancel_never_returns_or_auto_disables(settings, reason, monkeypatch):
    replay = replay_cycle(settings, offset_rad=0.02, supported_origin_disable_confirmation=True)
    replay.begin()
    advance(replay, until=lambda s: abs(replay.delta[2]) > 0.1)
    if reason == "human":
        monkeypatch.setattr(
            "robot_motion.rebot_b601_rs.torque_experiment.time.monotonic",
            lambda: replay.now + 0.001,
        )
        replay.controller.set_human_intervention(True)
    elif reason == "withdraw":
        replay.controller.submit("withdraw", now_sec=replay.now + 0.001)
    advance(replay, permit=reason != "lease", move=False)
    result = replay.controller.result_record()
    assert result["cycle"]["phase"] in {"cancelled", "failed"}
    assert not replay.controller.cycle_disable_confirmation_ready
    assert not any(
        e["phase"] == "returning" for e in replay.controller.status()["events"]
    )


def test_full_cycle_continues_through_background_with_bounded_gui_permit(settings):
    replay = replay_cycle(
        settings,
        offset_rad=0.02,
        background_permit_timeout_sec=70.0,
    )
    replay.controller.set_background_mode_enabled(True)
    replay.begin()
    advance(replay, until=lambda s: abs(replay.delta[2]) > 0.005)
    replay.controller.submit(
        "visibility", visible=False, now_sec=replay.now + 0.001
    )
    for _ in range(45):
        replay.tick(permit=False)
    status = replay.controller.status()["automatic_calibration"]
    assert status["cycle"]["page_visible"] is False
    assert status["cycle"]["background_permit_timeout_sec"] == pytest.approx(70.0)
    assert status["cycle"]["phase"] not in {"cancelled", "failed"}
    assert any(record["kind"] == "automatic_cycle_visibility" for record in replay.records)
    replay.controller.submit("visibility", visible=True, now_sec=replay.now + 0.001)
    replay.tick(permit=False)
    assert replay.controller.status()["automatic_calibration"]["cycle"]["page_visible"] is True


def test_page_unload_withdrawal_keeps_its_first_reason_after_permit_expires(settings):
    replay = replay_cycle(settings, offset_rad=0.02, background_permit_timeout_sec=70.0)
    replay.controller.set_background_mode_enabled(True)
    replay.begin()
    advance(replay, until=lambda s: abs(replay.delta[2]) > 0.005)
    replay.controller.submit(
        "withdraw", withdraw_source="page_unload", now_sec=replay.now + 0.001
    )
    advance(replay, permit=False, move=False)
    cycle = replay.controller.result_record()["cycle"]
    transitions = cycle["events"]
    assert transitions[-2]["reason"] == "page_unload"
    assert transitions[-1]["phase"] == "cancelled"
    assert transitions[-1]["reason"] == "page_unload"
    assert cycle["positioning"]["reason"] == "page_unload"


def test_mode_selection_applies_background_lease_only_to_full_cycle(settings):
    replay = replay_cycle(settings, gui=True, offset_rad=0.02)
    assert replay.controller.limits.command_timeout_sec == pytest.approx(0.35)
    replay.controller.submit(
        "select_mode", operation_mode="supported_cycle", now_sec=replay.now + 0.001
    )
    assert replay.controller.limits.command_timeout_sec == pytest.approx(70.0)
    replay.controller.submit(
        "select_mode", operation_mode="fixed_position", now_sec=replay.now + 0.002
    )
    assert replay.controller.limits.command_timeout_sec == pytest.approx(0.35)


def test_cycle_rejects_twenty_degree_target_outside_joint_limits_before_force(settings):
    replay = replay_cycle(settings)
    replay.controller._position_limits = (
        ((-1.0, 1.0),) * 2 + ((-0.1, 0.2),) + ((-1.0, 1.0),) * 3
    )
    replay.begin()
    replay.tick()
    assert replay.delta == ZERO and replay.audit["phase"] == "fault"
    assert replay.audit["fault"] == "cycle_target_and_test_range_outside_joint_limits"


def test_cycle_missing_dispatch_cannot_advance_or_claim_disable(settings):
    replay = replay_cycle(settings, offset_rad=0.02)
    replay.begin()
    replay.tick(ack=False)
    with pytest.raises(ValueError, match="not confirmed"):
        replay.tick()
    assert not replay.controller.cycle_disable_confirmation_ready


def test_cycle_positioning_dispatch_start_keeps_forty_ms_watchdog(settings):
    replay = replay_cycle(settings, offset_rad=0.02)
    replay.begin()
    replay.tick(ack=False)
    with pytest.raises(ValueError, match="deadline_exceeded"):
        replay.controller.validate_dispatch_start(
            feedback_sequence=replay.sequence,
            feedback_timestamp_sec=replay.now,
            now_sec=replay.now + 0.040001,
        )
    assert not replay.controller.cycle_disable_confirmation_ready


def test_cycle_gravity_observation_never_adds_a_position_hold_to_hide_drift(settings):
    replay = replay_cycle(settings, offset_rad=0.02, gravity_drift_limit_rad=0.001)
    replay.begin()
    advance(
        replay,
        until=lambda s: (
            s["automatic_calibration"]["cycle"]["phase"] == "gravity_observe"
        ),
    )
    replay.tick()
    for _ in range(30):
        q = list(replay.q)
        q[2] += 0.0001
        replay.tick(q)
        assert replay.delta == ZERO
        if replay.audit["phase"] == "fault":
            break
    assert replay.controller.result_record()["cycle"]["phase"] == "failed"
    assert not replay.controller.cycle_disable_confirmation_ready
    assert not replay.controller.result_record()["onset_trials"]
    assert (
        replay.controller.result_record()["cycle"]["gravity_observation"][
            "maximum_J3_rad"
        ]
        > replay.controller.result_record()["cycle"]["gravity_observation"][
            "anchor_qpos_rad"
        ][2]
        + 0.001
    )


@pytest.mark.parametrize("problem", ["raw_speed", "feedback_gap"])
def test_cycle_gravity_only_observation_retains_hard_speed_and_feedback_guards(
    settings, problem
):
    replay = replay_cycle(settings, offset_rad=0.02)
    replay.begin()
    advance(
        replay,
        until=lambda s: (
            s["automatic_calibration"]["cycle"]["phase"] == "gravity_observe"
        ),
    )
    q = list(replay.q)
    if problem == "raw_speed":
        q[2] += 0.001
    replay.tick(q, dt_sec=0.041 if problem == "feedback_gap" else 0.01)
    assert replay.audit["phase"] == "fault"
    assert replay.audit["fault"] == (
        "cycle_feedback_gap_exceeded"
        if problem == "feedback_gap"
        else "cycle_raw_velocity_exceeded"
    )
    assert not replay.controller.cycle_disable_confirmation_ready


@pytest.mark.parametrize("axis", range(6))
def test_cycle_keeps_positioning_raw_velocity_faults_and_does_not_return(
    settings, axis
):
    replay = replay_cycle(settings, offset_rad=0.02)
    replay.begin()
    advance(replay, until=lambda s: replay.audit.get("phase") == "cue")
    q = list(replay.q)
    q[axis] += 0.001
    replay.tick(q)
    assert replay.audit["phase"] == "fault"
    assert replay.audit["fault"] == "cycle_raw_velocity_exceeded"
    result = replay.controller.result_record()
    assert result["cycle"]["reason"] == "cycle_raw_velocity_exceeded"
    assert len([e for e in result["cycle"]["events"] if e["phase"] == "failed"]) == 1
    assert not replay.controller.cycle_disable_confirmation_ready
    assert not any(
        e["phase"] == "returning" for e in replay.controller.status()["events"]
    )


def test_cycle_acquisition_quality_stop_returns_without_rewriting_failed_acquisition(
    settings,
):
    replay = replay_cycle(
        settings,
        offset_rad=0.02,
        gravity_observation_sec=0.1,
        supported_origin_disable_confirmation=True,
        config_values={"max_sweep_leg_sec": 0.15},
    )
    replay.begin()
    advance(replay)
    result = replay.controller.result_record()
    assert result["cycle"]["phase"] == "complete", result["cycle"]["reason"]
    assert result["state"] == "failed" and result["reason"] == "leg_timeout"
    assert result["outcome"]["severity"] == "warning"
    assert len(result["onset_trials"]) == 4
    assert not result["motion"]["completed_legs"]
    assert len(result["cycle"]["moves"]) == 2
    assert replay.controller.cycle_disable_confirmation_ready
    assert (
        replay.controller.status()["automatic_calibration"]["outcome"]["severity"]
        == "warning"
    )


def test_cycle_control_fault_during_acquisition_does_not_automatically_return(settings):
    replay = replay_cycle(settings, offset_rad=0.02, gravity_observation_sec=0.1)
    replay.begin()
    advance(
        replay,
        until=lambda s: (
            s["automatic_calibration"]["state"] == "onset_scan"
            and abs(replay.delta[2]) > 0.1
        ),
    )
    q = list(replay.q)
    q[2] += 0.002
    replay.tick(q)
    assert replay.audit["phase"] == "fault"
    assert not replay.controller.cycle_disable_confirmation_ready
    assert not any(
        e["phase"] == "returning" for e in replay.controller.status()["events"]
    )


def test_cycle_bad_native_receipt_cannot_reach_test_or_disable(settings):
    replay = replay_cycle(settings, offset_rad=0.02)
    replay.begin()
    with pytest.raises(ValueError, match="incomplete_six_axis_dispatch"):
        replay.tick(receipt_updates={"sent_joint_names": ["joint3"]})
    assert replay.controller.result_record()["cycle"]["phase"] == "failed"
    assert not replay.controller.cycle_disable_confirmation_ready


@pytest.mark.parametrize(
    "values",
    [
        {"offset_rad": float("nan")},
        {"offset_rad": math.radians(21)},
        {"positioning_timeout_sec": 1.0},
        {"supported_origin_disable_confirmation": 1},
    ],
)
def test_cycle_invalid_envelopes_are_rejected_before_connection(values):
    with pytest.raises((ValueError, TypeError)):
        CalibrationCycleConfig(**values)


def _production_cycle_controller(tmp_path):
    root = Path(__file__).resolve().parents[1] / "docs/hardware/profiles"
    path = root / "b601_j3_supported_calibration_cycle_v1.json"
    args = owner._parser().parse_args(
        [
            "--allow-motion",
            "--gravity-debug-mode",
            "--torque-console-auto-calibration-profile",
            str(path),
            "--rate-hz",
            "100",
            "--torque-console-max-assist-nm",
            "1",
            "--torque-console-staged-target-offset-limit-rad",
            ".0845",
            "--torque-console-max-joint-velocity-rad-s",
            ".195",
            "--torque-console-staged-motion-onset-rad",
            ".0004",
            "--torque-console-staged-breakaway-duration-sec",
            "3",
            "--torque-console-staged-breakaway-impulse-nm-sec",
            "1.5",
            "--torque-console-staged-breakaway-torque-nm", "1",
            "--torque-console-staged-breakaway-rise-slew-nm-s", "1",
            "--torque-console-confirmation",
            "B601_RS_TORQUE_CONSOLE",
        ]
    )
    parsed = probe.build_parser().parse_args(
        owner.build_probe_command(args, run_dir=tmp_path)[2:]
    )
    probe._validate_cli(parsed)
    return probe._automatic_calibration_controller_from_args(parsed)


@pytest.mark.parametrize("idle_gap_sec", [.004, .01, 15.])
def test_production_gui_fixed_mode_sends_first_support_damping_via_owner_adapter(
    tmp_path, idle_gap_sec,
):
    controller = _production_cycle_controller(tmp_path)
    replay = AcquisitionReplay(controller.settings, controller=controller)
    controller.submit(
        "arm", now_sec=replay.now + idle_gap_sec - .001,
        operation_mode="fixed_position",
    )
    replay.tick(dt_sec=idle_gap_sec)
    assert replay.delta == ZERO
    increment = 2. * min(.01, idle_gap_sec)
    expected = (increment, increment, 0., increment, increment, increment)
    assert replay.last_kd == pytest.approx(expected)
    assert controller.status()["automatic_calibration"]["support_damping"][
        "last_acknowledged_kd_nm_s_per_rad"
    ] == pytest.approx(expected)


def test_production_gui_cycle_handoff_slews_damping_from_verified_base(tmp_path):
    controller = _production_cycle_controller(tmp_path)
    replay = AcquisitionReplay(controller.settings, controller=controller)
    controller.submit(
        "select_mode", now_sec=replay.now + .001,
        operation_mode="supported_cycle",
    )
    replay.tick()
    controller.submit(
        "arm", now_sec=replay.now + .001,
        operation_mode="supported_cycle", cycle_angle_deg=2.,
    )
    advance(replay, until=lambda view: view["automatic_calibration"]["cycle"]["phase"] == "calibration")
    assert controller.status()["automatic_calibration"]["cycle"]["phase"] == "calibration"
    assert replay.last_kd == ZERO
    assert replay.delta == ZERO
    observation = controller.result_record()["cycle"]["gravity_observations"][0]
    assert observation["zero_assist_and_base_damping_confirmed"]
    # The child was idle throughout transfer/observation. That elapsed time is
    # not an available damping-ramp interval in the first calibration packet.
    for tick in range(1, 6):
        replay.tick()
        expected = (.02 * tick, .02 * tick, 0., .02 * tick, .02 * tick, .02 * tick)
        assert replay.last_kd == pytest.approx(expected)
        assert replay.delta == ZERO
        assert controller.status()["automatic_calibration"]["support_damping"][
            "last_acknowledged_kd_nm_s_per_rad"
        ] == pytest.approx(expected)


@pytest.mark.parametrize("trial", ["sweep-1", "sweep-2"])
def test_production_local_track_regulates_without_spending_stop_resume_budget(tmp_path, trial):
    controller = _production_cycle_controller(tmp_path)
    replay = AcquisitionReplay(controller.settings, controller=controller)
    controller.submit("arm", now_sec=replay.now+.001, operation_mode="fixed_position")
    advance(replay, until=lambda view: (
        (replay.audit.get("calibration_trial") or {}).get("trial_id") == trial
        and (replay.audit.get("motion_control") or {}).get("phase") == "track"
    ))
    session = controller._fixed._sweep
    start, index, exposure = session._leg_started, session._index, session._leg_exposure
    target = replay.audit["motion_control"]["target_position_rad"]
    direction = math.copysign(1., target - replay.q[2])
    scales = []
    for _ in range(12):
        q = list(replay.q)
        q[2] += direction*.00023
        replay.tick(q)
        motion = replay.audit["motion_control"]
        # With the production arrival tolerance, the second sweep may reach
        # its arrival boundary during this synthetic sequence. Entering brake
        # there is expected; only assert tracking behavior while TRACK is
        # active rather than requiring it to continue inside the arrival band.
        if motion["phase"] != "track":
            assert motion["phase"] == "brake"
            break
        assert motion["speed_governor_resume_count"] == 0
        assert motion["target_position_rad"] == target
        assert motion["tracking_speed_brake_limit_rad_s"] == .0585
        assert abs(replay.delta[2]) <= 1.
        scales.append(motion["continuous_speed_scale"])
    assert scales
    if trial == "sweep-1":
        assert any(0. < scale < 1. for scale in scales)
    assert session._leg_started == start and session._index == index
    assert session._leg_exposure >= exposure
    controller.submit("withdraw", now_sec=replay.now+.001)
    advance(replay, move=False)
    assert replay.delta == ZERO and replay.last_kd == ZERO
    assert controller.status()["automatic_calibration"]["state"] == "cancelled"


def test_cycle_profile_reuses_local_identification_without_modifying_ordinary_profile(tmp_path):
    root = Path(__file__).resolve().parents[1] / "docs/hardware/profiles"
    values = probe._automatic_profile_values(root / "b601_j3_supported_calibration_cycle_v1.json")
    cycle = values.pop("cycle")
    assert values == json.loads((root / "b601_j3_local_identification_v1.json").read_text())
    controller = _production_cycle_controller(tmp_path)
    assert isinstance(controller, B601CalibrationModeController)
    assert as_cycle(controller) == cycle
    assert not controller.cycle_disable_confirmation_ready
    assert controller.limits.max_joint_velocity_rad_s == 0.195
    assert controller.settings.staged_assist.motion_control.support_soft_speed_recovery
    assert not controller._fixed.settings.staged_assist.motion_control.support_soft_speed_recovery
    assert controller.settings.staged_assist.motion_control.continuous_speed_limit_rad_s == .075
    assert controller._fixed.settings.staged_assist.motion_control.continuous_speed_limit_rad_s == .0585
    assert controller.settings.staged_assist.tracking_slew_nm_s == 1.
    assert controller.settings.limits.rise_slew_nm_s == 1.
    assert controller.settings.staged_assist.tracking_release_slew_nm_s == 8.
    assert controller.settings.staged_assist.motion_control.braking_slew_nm_s == 10.
    assert controller.settings.limits.max_excursion_rad == pytest.approx(
        math.radians(20) + .04 + .01352
    )
    assert controller.settings.limits.max_excursion_rad < .405
    assert controller._fixed.settings.staged_assist.tracking_release_slew_nm_s is None
    assert controller._fixed.settings.staged_assist.tracking_slew_nm_s == .15
    assert controller._fixed.settings.limits.rise_slew_nm_s == .15
    assert controller._supported._position._runtime.shared_config.max_assist_torque_rate_nm_s[2] == 1.
    assert controller._fixed._staged._runtime.shared_config.max_assist_torque_rate_nm_s[2] == .15
    assert controller.status()["automatic_calibration"]["operation_mode"] == "fixed_position"
    assert controller.result_record()["operation_mode"] == "fixed_position"
    controller.submit("select_mode", now_sec=1., operation_mode="supported_cycle")
    assert controller.status()["automatic_calibration"]["operation_mode"] == "supported_cycle"
    assert controller.result_record()["operation_mode"] == "supported_cycle"


def test_transfer_soft_ceiling_must_be_admitted_below_raw_hard_guard_before_connection(tmp_path):
    controller = _production_cycle_controller(tmp_path)
    with pytest.raises(ValueError, match="continuous speed ceiling"):
        replace(controller.settings, limits=replace(controller.settings.limits, max_joint_velocity_rad_s=.075))


@pytest.mark.parametrize("rate", [0., -.1, True, float("nan"), float("inf")])
def test_transfer_rise_override_rejects_invalid_configuration(rate):
    with pytest.raises(ValueError):
        CalibrationCycleConfig(positioning_torque_rise_nm_s=rate)


def test_transfer_rise_cannot_exceed_existing_release_rate(tmp_path):
    controller = _production_cycle_controller(tmp_path)
    with pytest.raises(ValueError, match="release slew"):
        B601AutomaticCalibrationCycleController(controller._fixed,
            replace(controller.cycle, positioning_torque_rise_nm_s=3.))


@pytest.mark.parametrize("rate", [0., -1., True, float("nan"), float("inf"), .5, 11.])
def test_normal_transfer_fall_is_finite_and_cannot_exceed_protective_braking_or_undercut_rise(tmp_path, rate):
    controller = _production_cycle_controller(tmp_path)
    with pytest.raises(ValueError):
        B601AutomaticCalibrationCycleController(controller._fixed,
            replace(controller.cycle, positioning_torque_fall_nm_s=rate))


@pytest.mark.parametrize("continuous,friction,residual", [
    (False, .2, 0.), (False, .6, 0.),
    (True, .2, 0.), (True, .6, 0.), (True, .2, .16), (True, .2, -.16),
])
def test_transfer_native_owner_with_quantized_stiction_plant_reaches_twenty_degrees(tmp_path, friction, continuous, residual, record_property):
    # Synthetic mechanical stress model, not identified B601 dynamics. Native
    # controller/receipt paths consume the ACTUAL prior dispatched command.
    controller = _production_cycle_controller(tmp_path)
    if not continuous:
        controller = B601CalibrationModeController(B601AutomaticCalibrationCycleController(
            controller._fixed, replace(controller.cycle, positioning_continuous_speed_limit_rad_s=None,
                positioning_torque_rise_nm_s=None, positioning_torque_fall_nm_s=None)))
    replay = AcquisitionReplay(controller.settings, controller=controller)
    controller.submit("select_mode", now_sec=replay.now, operation_mode="supported_cycle")
    replay.tick()
    controller.submit("arm", now_sec=replay.now+.001, operation_mode="supported_cycle")
    def plant_step(q, velocity):
        # Inertia + dry friction + angle-dependent load/cogging. Gravity is
        # residual can hold at rest but exceed kinetic friction after release,
        # exposing reverse motion without an initially unstable gravity pose.
        for _ in range(10):
            effort = replay.delta[2] - replay.last_kd[2]*velocity - .02*math.sin(25*q) - residual
            static = friction * (1.0 + .15*math.sin(40*q))
            if abs(velocity) < 1e-5 and abs(effort) <= static:
                velocity = 0.
            else:
                direction = math.copysign(1., velocity if abs(velocity)>1e-5 else effort)
                next_v = velocity + (effort - direction*.65*static - .5*velocity)/2.*.001
                velocity = 0. if next_v*velocity < 0 else next_v
                q += velocity*.001
        feedback = list(replay.q)
        feedback[2] = round(q/.000383496)*.000383496
        replay.tick(feedback)
        return q, velocity
    q, velocity = 0., 0.
    maximum_speed = maximum_torque = maximum_resume = regulation_ticks = feedback_ticks = 0
    for _ in range(24000):
        q, velocity = plant_step(q, velocity)
        maximum_speed = max(maximum_speed, abs(velocity))
        maximum_torque = max(maximum_torque, abs(replay.delta[2]))
        motion = replay.audit.get("motion_control") or {}
        maximum_resume = max(maximum_resume, motion.get("speed_governor_resume_count", 0))
        regulation_ticks += motion.get("continuous_speed_scale") is not None and motion["continuous_speed_scale"] < 1.
        feedback_ticks += (replay.audit.get("tracking_recovery") or {}).get("tracking_velocity_correction_nm", 0.) > 0.
        cycle = controller.status()["automatic_calibration"]["cycle"]
        replay.audits.clear()
        replay.records.clear()
        if cycle["phase"] != "positioning_out":
            break
    record_property("synthetic_transfer", {
        "friction_nm": friction, "residual_nm": residual, "continuous": continuous, "phase": cycle["phase"],
        "reason": cycle["reason"], "position_rad": q, "duration_sec": replay.now-1.12,
        "maximum_physical_speed_rad_s": maximum_speed, "maximum_overlay_nm": maximum_torque,
        "maximum_resumptions": maximum_resume, "regulation_ticks": regulation_ticks,
        "net_velocity_feedback_ticks": feedback_ticks,
    })
    if not continuous and friction == .6:
        # This stress plant exceeds the admitted 234 s positioning window;
        # it must terminate on the outer batch deadline, not a retry counter.
        assert cycle["phase"] == "failed"
        assert cycle["reason"] == "cycle_positioning:batch_timeout"
        assert maximum_resume > 7
        return
    assert cycle["phase"] == "gravity_observe", (cycle, q, velocity, maximum_resume)
    assert abs(q-math.radians(20)) <= controller.config.motion_tracking_arrival_tolerance_rad
    assert maximum_torque <= 1.
    assert maximum_speed < .10
    # Net velocity feedback may regulate before the independent PD scaling
    # ceiling is reached. Reaching saturation is not an arrival requirement.
    assert regulation_ticks+feedback_ticks > 0 if continuous else regulation_ticks+feedback_ticks == 0
    # Soft governor may pause and resume repeatedly; outer positioning/cycle
    # deadlines and the command-exposure budget bound the operation.
    assert controller._fixed.settings.staged_assist.motion_control.continuous_speed_limit_rad_s == .0585
    if continuous:
        # Intermediate calibration uses prescribed encoder motion, not the
        # mechanical model above. Then test the SAME native return controller
        # from that actual boundary with a negative physical-motion direction.
        advance(replay, until=lambda s: s["automatic_calibration"]["cycle"]["phase"] == "returning")
        cycle = controller.status()["automatic_calibration"]["cycle"]
        assert cycle["phase"] == "returning", cycle["reason"]
        q, velocity = replay.q[2], 0.
        for _ in range(18000):
            q, velocity = plant_step(q, velocity)
            cycle = controller.status()["automatic_calibration"]["cycle"]
            replay.audits.clear()
            replay.records.clear()
            if cycle["phase"] != "returning":
                break
        assert cycle["phase"] == "return_observe", (cycle, q, velocity)
        assert abs(q) <= controller.config.motion_tracking_arrival_tolerance_rad
        assert replay.delta == ZERO and replay.last_kd == ZERO


@pytest.mark.parametrize("enabled", [False, True])
def test_native_cycle_coupled_j4_soft_motion_has_bounded_positioning_only_recovery(settings, enabled):
    replay = replay_cycle(
        settings, offset_rad=0.02, positioning_support_soft_speed_recovery=enabled,
        config_values={
            "motion_position_resolution_rad": .0004,
            "motion_responsive_tracking_brake": True,
            "motion_reference_pacing": True,
            "motion_breakaway_probe": True,
            "motion_tracking_recovery_cap_nm": .3,
            "motion_tracking_recovery_gain_nm_per_rad": 20.,
            "motion_onset_handoff": "track",
            "motion_speed_governor": True,
        },
    )
    replay.begin()
    advance(replay, until=lambda s: (replay.audit.get("motion_control") or {}).get("phase") == "track")
    controller = replay.controller
    session = controller._session
    started = session._started
    exposure = controller._exposure
    # Quantized J4 coupling: one count per 10 ms, below the original .05 raw
    # guard and .5-degree support range, but above the .03 soft brake.
    for _ in range(10):
        q = list(replay.q)
        q[3] += .000383496
        motion = replay.audit.get("motion_control")
        if motion and motion["phase"] == "track":
            q[2] = motion["q_reference_rad"]
        replay.tick(q)
    assert replay.audit["motion_control"]["phase"] == "brake"
    assert replay.audit["motion_control"]["reason"] == "support_velocity_braking"
    assert replay.audit["motion_control"]["speed_governor_resume_count"] == 0
    advance(replay, until=lambda s: s["automatic_calibration"]["cycle"]["phase"] in {"gravity_observe", "failed"})
    result = controller.result_record()["cycle"]
    assert session._started == started
    assert controller._exposure >= exposure
    assert result["origin_qpos_rad"] == ZERO
    assert result["outbound_target_rad"] == .02
    if enabled:
        assert result["phase"] == "gravity_observe", result["reason"]
        assert result["completed_positioning_moves"] == 1
        assert result["positioning_control"]["speed_governor_resume_count"] == 1
        assert max((a.get("motion_control") or {}).get("speed_governor_resume_count", 0)
                   for a in replay.audits) == 1
        advance(replay)
        final = controller.result_record()
        assert final["cycle"]["phase"] == "complete", final["cycle"]["reason"]
        assert final["cycle"]["completed_positioning_moves"] == 2
        assert len(final["motion"]["completed_legs"]) == 12
        assert final["cycle"]["moves"][0]["motion_control"]["speed_governor_resume_count"] == 1
        before_stop = final["cycle"]["positioning_control"]
        controller.owner_stopped(now_sec=replay.now, reason="test_complete")
        assert controller.result_record()["cycle"]["positioning_control"] == before_stop
    else:
        assert result["phase"] == "failed"
        assert result["reason"] == "cycle_positioning:stopped_short_of_target"


def as_cycle(controller):
    return dict(vars(controller.cycle))


@pytest.mark.parametrize("mode", ["fixed_position", "supported_cycle"])
def test_gui_selection_executes_only_chosen_flow_and_locks_after_start(settings, mode):
    replay = replay_cycle(settings, gui=True, offset_rad=math.radians(20),
                          supported_origin_disable_confirmation=True, gravity_observation_sec=.1)
    controller = replay.controller
    assert controller.status()["mode_selection"]["selected"] == "fixed_position"
    controller.submit("select_mode", now_sec=replay.now+.001, operation_mode=mode)
    replay.tick()
    assert replay.delta == ZERO
    assert controller.status()["mode_selection"]["ready"]
    assert not controller.cycle_disable_confirmation_ready
    controller.submit("arm", now_sec=replay.now+.001, operation_mode=mode)
    assert controller.status()["mode_selection"]["locked"]
    with pytest.raises(ValueError, match="locked"):
        controller.submit("select_mode", now_sec=replay.now+.002,
                          operation_mode="fixed_position" if mode == "supported_cycle" else "supported_cycle")
    advance(replay)
    result = controller.result_record()
    assert result["state"] == "complete", result["reason"]
    assert result["operation_mode"] == mode
    assert len(result["motion"]["completed_legs"]) == 12
    assert replay.delta == ZERO and replay.last_kd == ZERO
    if mode == "fixed_position":
        assert "cycle" not in result
        assert not controller.cycle_disable_confirmation_ready
        assert all(abs(a["qpos_rad"][2]) < .05 for a in replay.audits)
    else:
        assert result["cycle"]["phase"] == "complete"
        assert len(result["cycle"]["moves"]) == 2
        assert controller.cycle_disable_confirmation_ready
        controller.owner_stopped(now_sec=replay.now, reason="waiting_for_gui_confirmation")
        final = controller.result_record()
        assert final["cycle"]["disable_confirmation_available"] is True
        assert controller.cycle_disable_confirmation_ready


def _ack_replay_pending(replay):
    replay.controller.record_dispatch(
        native_receipt=replay.receipt, joint_name="joint3",
        receipt_sequence=replay.receipt["sequence"], feedback_timestamp_sec=replay.now,
        completed_at_sec=replay.receipt["completed_at_sec"],
        overlay_after_profile_nm=replay.delta[2], gravity_nm=0.,
        requested_total_nm=replay.delta[2], conditioned_total_nm=replay.delta[2],
        joint_commands={"gravity_nm": ZERO, "overlay_nm": replay.delta,
                        "total_nm": replay.delta, "driver_kd_nm_s_per_rad": replay.last_kd},
        torque_rate_nm_s=(28.8,)*6,
    )


def test_gui_switch_during_idle_dispatch_pins_old_receipt_and_confirms_new_mode_next_cycle(settings):
    replay = replay_cycle(settings, gui=True)
    replay.tick(ack=False)
    controller = replay.controller
    controller.submit("select_mode", now_sec=replay.now+.001, operation_mode="supported_cycle")
    assert not controller.status()["mode_selection"]["ready"]
    with pytest.raises(ValueError, match="zero-assist cycle"):
        controller.submit("arm", now_sec=replay.now+.002, operation_mode="supported_cycle")
    _ack_replay_pending(replay)
    assert not controller.status()["mode_selection"]["ready"]
    assert controller._fixed._pending_sequence is None
    replay.tick()
    assert replay.delta == ZERO and controller.status()["mode_selection"]["ready"]
    with pytest.raises(ValueError, match="displayed mode changed"):
        controller.submit("arm", now_sec=replay.now+.001, operation_mode="fixed_position")
    assert not controller.status()["mode_selection"]["locked"]
    controller.submit("arm", now_sec=replay.now+.002, operation_mode="supported_cycle")
    assert controller.status()["mode_selection"]["locked"]


@pytest.mark.parametrize("invalid", [None, True, [], {}, "unknown"])
def test_gui_invalid_selection_or_stopped_owner_cannot_arm(settings, invalid):
    replay = replay_cycle(settings, gui=True)
    with pytest.raises(ValueError, match="choose"):
        replay.controller.submit("select_mode", now_sec=replay.now+.001, operation_mode=invalid)
    assert replay.delta == ZERO
    replay.controller.owner_stopped(now_sec=replay.now+.002, reason="test_shutdown")
    with pytest.raises(ValueError, match="locked"):
        replay.controller.submit("select_mode", now_sec=replay.now+.003, operation_mode="supported_cycle")
    with pytest.raises(ValueError, match="owner has stopped"):
        replay.controller.submit("arm", now_sec=replay.now+.003, operation_mode="fixed_position")


def test_gui_mode_change_records_use_real_owner_writer(settings):
    replay = replay_cycle(settings, gui=True)
    replay.controller.submit("select_mode", now_sec=replay.now+.001, operation_mode="supported_cycle")
    stream = io.StringIO()
    probe._flush_automatic_calibration_records(replay.controller, probe._DynamicsTrace(stream))
    record = json.loads(stream.getvalue())
    assert record["kind"] == "automatic_calibration_mode_selected"
    assert record["mode_selected_at_sec"] == replay.now+.001
    assert record["operation_mode"] == "supported_cycle"


def test_gui_repeated_idle_selections_keep_gravity_only_and_prime_inactive_feedback(settings):
    replay = replay_cycle(settings, gui=True)
    for mode in ["supported_cycle", "fixed_position"] * 3:
        replay.controller.submit("select_mode", now_sec=replay.now+.001, operation_mode=mode)
        for _ in range(50):
            replay.tick()
            assert replay.delta == ZERO and replay.audit["phase"] != "fault"
        status = replay.controller.status()
        assert status["mode_selection"]["ready"] and not status["mode_selection"]["locked"]
        assert status["automatic_calibration"]["state"] == "idle"


def test_gui_mode_buttons_use_authenticated_api_and_backend_rejects_mid_batch_switch(settings):
    replay = replay_cycle(settings, gui=True)
    server = B601TorqueConsoleServer(replay.controller)
    server.start()
    try:
        with urlopen(server.url+"/", timeout=2) as response:
            page = response.read().decode()
        assert 'id="mode-fixed-position"' in page and 'id="mode-supported-cycle"' in page
        token = page.split('name="csrf-token" content="', 1)[1].split('"', 1)[0]

        def post(payload):
            request = Request(server.url+"/api/intent", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json", "Origin": server.url,
                                       "X-B601-Console-Token": token})
            with urlopen(request, timeout=2) as response:
                return json.load(response)

        assert post({"action": "select_mode", "operation_mode": "supported_cycle"})["status"] == "accepted"
        assert replay.delta == ZERO and not replay.controller.status()["mode_selection"]["ready"]
        replay.tick()
        with pytest.raises(HTTPError) as stale:
            post({"action": "arm", "operation_mode": "fixed_position"})
        assert stale.value.code == 400
        for angle in [0., -1., 21., True, "10", None]:
            with pytest.raises(HTTPError) as invalid_angle:
                post({"action": "arm", "operation_mode": "supported_cycle", "cycle_angle_deg": angle})
            assert invalid_angle.value.code == 400
            assert not replay.controller.status()["mode_selection"]["locked"]
            assert math.degrees(replay.controller.cycle.offset_rad) == pytest.approx(20.)
        assert post({"action": "arm", "operation_mode": "supported_cycle", "cycle_angle_deg": 12.})["status"] == "accepted"
        assert math.degrees(replay.controller.cycle.offset_rad) == pytest.approx(12.)
        assert replay.controller.status()["mode_selection"]["cycle_angle_deg"] == pytest.approx(12.)
        assert replay.controller._supported.maximum_offset_rad == pytest.approx(math.radians(20))
        with pytest.raises(HTTPError) as locked:
            post({"action": "select_mode", "operation_mode": "fixed_position"})
        assert locked.value.code == 400
        with pytest.raises(HTTPError) as invalid:
            post({"action": "select_mode", "operation_mode": []})
        assert invalid.value.code == 400
        assert replay.controller.status()["mode_selection"]["selected"] == "supported_cycle"
    finally:
        server.close()


@pytest.mark.parametrize("angle", [5., 12., 20.])
def test_gui_selected_angle_changes_outbound_goal_not_admitted_envelope_or_return_origin(settings, angle):
    replay = replay_cycle(settings, gui=True)
    controller = replay.controller
    controller.submit("select_mode", now_sec=replay.now+.001, operation_mode="supported_cycle")
    replay.tick()
    envelope = controller.settings.maximum_target_offset_rad
    controller.submit("arm", now_sec=replay.now+.001,
        operation_mode="supported_cycle", cycle_angle_deg=angle)
    replay.tick()
    cycle = controller.result_record()["cycle"]
    assert cycle["settings"]["offset_rad"] == pytest.approx(math.radians(angle))
    assert cycle["outbound_target_rad"]-cycle["origin_qpos_rad"][2] == pytest.approx(math.radians(angle))
    assert controller.settings.maximum_target_offset_rad == envelope
    with pytest.raises(ValueError):
        controller.submit("arm", now_sec=replay.now+.001,
            operation_mode="supported_cycle", cycle_angle_deg=8.)
    assert controller.cycle.offset_rad == pytest.approx(math.radians(angle))
@pytest.mark.parametrize("invalid", [None, [], False])
def test_explicit_cycle_profile_cannot_silently_fall_back_to_an_ordinary_batch(
    tmp_path, invalid
):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"motion_profile": "unused.json"}))
    profile = tmp_path / "cycle.json"
    profile.write_text(json.dumps({"calibration_profile": base.name, "cycle": invalid}))
    with pytest.raises(TypeError, match="cycle settings must be a JSON object"):
        probe._automatic_profile_values(profile)

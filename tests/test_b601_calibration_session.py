"""Real controller + scripted feedback/receipts; not plant or hardware proof."""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from control_station.b601_torque_console.calibration_session import (
    B601AutomaticCalibrationSession,
    CalibrationBatchLimits,
)
from control_station.b601_torque_console.staged_controller import (
    B601StagedTorqueConsoleController,
)
from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition,
    plan_local_sweeps,
)
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

ZERO = (0.0,) * 6


@pytest.fixture
def settings(tmp_path):
    profile = (
        Path(__file__).resolve().parents[1]
        / "docs/hardware/profiles/b601_j3_interaction_v2.json"
    )
    args = owner._parser().parse_args(
        [
            "--allow-motion",
            "--torque-console",
            "--gravity-debug-mode",
            "--torque-console-confirmation",
            "B601_RS_TORQUE_CONSOLE",
            "--torque-console-motion-profile",
            str(profile),
            "--rate-hz",
            "100",
            "--torque-console-staged-motion-onset-rad",
            ".0004",
            "--torque-console-staged-motion-confirm-rad",
            ".0007",
            "--torque-console-staged-motion-confirm-samples",
            "2",
            # Explicit offline replay budget covers onset plus native stop dwell.
            # The production defaults are neither replaced nor expanded.
            "--torque-console-staged-breakaway-duration-sec",
            "1.2",
            "--torque-console-staged-breakaway-impulse-nm-sec",
            ".6",
        ]
    )
    parsed = probe.build_parser().parse_args(
        owner.build_probe_command(args, run_dir=tmp_path)[2:]
    )
    return probe._staged_torque_console_settings_from_args(parsed)


class Replay:
    """Native staged logic only; feedback motion is prescribed, not simulated."""

    def __init__(self, settings, *, limits=None, plan_change=None, acquisition_arrival_tolerance_rad=None, excursion_rad=.008):
        self.controller = B601StagedTorqueConsoleController(settings)
        self.condition = CalibrationCondition(
            "offline-contract-replay",
            "a" * 64,
            "b" * 64,
            1.0,
            (1.0, 0.96, 1.0, 1.0, 1.0, 1.0),
            ZERO,
            (0.0, -0.22, -0.11, -0.08, 0.0, 0.0),
            ZERO,
            0.02,
            ZERO,
        )
        self.plan = plan_local_sweeps(
            condition=self.condition,
            joint_name="joint3",
            anchor_qpos_rad=ZERO,
            position_limits_rad=((-1.0, 1.0),) * 6,
            excursion_rad=excursion_rad,
            speeds_rad_s=(0.0075, 0.015),
            max_reference_velocity_rad_s=0.015,
        )
        if plan_change is not None:
            plan_change(self.plan)
        self.session = B601AutomaticCalibrationSession(
            self.controller,
            plan=self.plan,
            condition=self.condition,
            position_limits_rad=((-1.0, 1.0),) * 6,
            limits=limits or CalibrationBatchLimits(120.0, 12.0, 2.0, 0.10, 0.04, 20.0),
            acquisition_arrival_tolerance_rad=acquisition_arrival_tolerance_rad,
        )
        self.now = 1.0
        self.sequence = 0
        self.q = ZERO
        self.audit = {}
        self.controller.owner_ready()
        self.history = []
        for _ in range(12):
            self.tick()

    def tick(
        self,
        *,
        q=None,
        human=False,
        permit=True,
        context=None,
        ack=True,
        command_delta=None,
    ):
        self.now = round(self.now + 0.01, 9)
        self.sequence += 1
        new_q = self.q if q is None else tuple(q)
        velocity = tuple((a - b) / 0.01 for a, b in zip(new_q, self.q))
        self.q = new_q
        overlay, self.audit = self.session.step(
            now_sec=self.now,
            feedback_sequence=self.sequence,
            qpos_rad=new_q,
            qvel_rad_s=velocity,
            condition=context or self.condition,
            human_intervention=human,
            operator_permit_timestamp_sec=self.now if permit else None,
        )
        delta = overlay if command_delta is None else command_delta
        kd = self.controller.active_driver_damping(ZERO, final_command_delta_nm=delta)
        self.receipt = {
            "accepted": True,
            "requested_joint_names": list(B601_RS_ARM_JOINT_NAMES),
            "sent_joint_names": list(B601_RS_ARM_JOINT_NAMES),
            "errors": [],
            "failed_joint_names": [],
            "mode": "mit",
            "frame_id": "b601_rs/joint_mit_reference",
            "source_sequence": self.sequence,
            "sequence": self.sequence,
            "requested_at_sec": self.now + 0.0002,
            "completed_at_sec": self.now + 0.0008,
        }
        self.commands = {
            "gravity_nm": ZERO,
            "overlay_nm": overlay,
            "total_nm": delta,
            "driver_kd_nm_s_per_rad": kd,
        }
        if ack:
            self.ack()
        self.history.append(
            {
                "audit": self.audit,
                "commands": self.commands,
                "state": self.session.status()["state"],
            }
        )

    def ack(self):
        self.session.record_dispatch(
            receipt=self.receipt,
            joint_commands=self.commands,
            torque_rate_nm_s=(28.8,) * 6,
        )

    def begin(self, *, breakaway=False):
        self.session.begin(
            now_sec=self.now + 0.001,
            allow_motion=True,
            human_intervention=False,
            allow_breakaway=breakaway,
        )

    def run_until(self, predicate, *, max_ticks=2000, move=False):
        for _ in range(max_ticks):
            if predicate():
                return
            q = list(self.q)
            view = self.controller.owner_snapshot()
            if move and self.session.status()["state"] == "moving":
                target = self.plan["steps"][self.session.status()["step_index"]][
                    "target_qpos_rad"
                ][2]
                # An ideal scripted servo follows the committed reference in
                # TRACK. Initial displacement is an explicit test stimulus.
                motion = self.audit.get("motion_control")
                if motion and motion["phase"] == "track":
                    q[2] = motion["q_reference_rad"]
                elif view["phase"] == "breakaway":
                    q[2] += max(-0.00010, min(0.00010, target - q[2]))
                elif (
                    view["phase"] == "releasing"
                    and not view["release_terminal"]
                    and abs(q[2] - view["anchor_qpos_rad"][2]) < 0.001
                ):
                    # Explicit post-pulse inertial displacement stimulus so the
                    # native multi-window onset confirmation can be exercised.
                    q[2] += max(-0.00010, min(0.00010, target - q[2]))
            self.tick(q=q)
        pytest.fail(
            f"bounded replay timed out: {self.session.status()} / {self.controller.owner_snapshot()}"
        )


def test_complete_bidirectional_multispeed_batch_uses_native_tracking_and_receipts(
    settings,
):
    replay = Replay(settings)
    original = copy.deepcopy(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session.status()["state"] in {"complete", "failed"},
        max_ticks=12000,
        move=True,
    )
    result = replay.session.result_record()
    assert result["state"] == "complete", result
    assert len(result["completed_legs"]) == len(replay.plan["steps"]) == 14
    assert replay.controller.limits == original.limits
    assert (
        replay.controller.settings.tracking_kp_nm_per_rad
        == original.tracking_kp_nm_per_rad
    )
    assert (
        replay.controller.settings.staged_assist.motion_control.drive_driver_kd_nm_s_per_rad
        == original.staged_assist.motion_control.drive_driver_kd_nm_s_per_rad
    )
    assert any(r["audit"].get("phase") == "track" for r in replay.history)
    assert replay.commands["total_nm"] == ZERO
    assert replay.commands["driver_kd_nm_s_per_rad"] == ZERO
    assert result["hardware_execution_allowed"] is False
    with pytest.raises(ValueError, match="one-shot"):
        replay.begin()


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"human": True}, "human_contact_or_unknown"),
        ({"human": None}, "human_contact_or_unknown"),
        ({"permit": False}, "operator_permit_expired"),
        ({"gravity": True}, "gravity_or_source_condition_changed"),
    ],
)
def test_abort_withdraws_without_next_leg_or_automatic_resume(settings, change, reason):
    replay = Replay(settings)
    replay.begin()
    replay.run_until(lambda: replay.session.status()["state"] == "moving")
    replay.tick()
    if change.pop("gravity", False):
        change["context"] = replace(replay.condition, gravity_scale=0.99)
    replay.tick(**change)
    replay.run_until(
        lambda: replay.session.status()["state"] in {"failed", "cancelled"}
    )
    state = replay.session.status()
    assert state["reason"] == reason
    assert state["completed_legs"] == []
    for _ in range(30):
        replay.tick()
    assert replay.controller.owner_snapshot()["operator_phase"] == "inactive"
    assert replay.commands["total_nm"] == ZERO


@pytest.mark.parametrize(
    "field,value",
    [
        ("accepted", False),
        ("sent_joint_names", ["joint3"]),
        ("source_sequence", 99999),
        ("sequence", 0),
        ("frame_id", "wrong"),
        ("completed_at_sec", 100.0),
    ],
)
def test_incomplete_or_wrong_dispatch_cannot_advance_batch(settings, field, value):
    replay = Replay(settings)
    replay.begin()
    replay.tick(ack=False)
    replay.receipt[field] = value
    with pytest.raises(ValueError):
        replay.ack()
    assert replay.session.status()["state"] == "releasing"
    assert replay.session.status()["step_index"] == 0
    assert replay.session.status()["completed_legs"] == []


def test_missing_receipt_never_arms_or_moves_next_leg(settings):
    replay = Replay(settings)
    replay.begin()
    replay.tick(ack=False)
    replay.tick()
    assert replay.session.status()["reason"] == "previous_dispatch_not_confirmed"
    assert replay.session.status()["completed_legs"] == []


def test_global_support_drift_is_not_reset_on_rearm(settings):
    replay = Replay(settings)
    replay.begin()
    q = list(ZERO)
    for _ in range(100):
        q[0] += 0.0001
        replay.tick(q=q)
        if replay.session.status()["state"] in {"releasing", "failed"}:
            break
    assert replay.session.status()["reason"] == "global_support_drift"
    assert replay.session.status()["outcome"]["severity"] == "error"


def test_no_motion_or_insufficient_authority_does_not_fake_completed_leg(settings):
    replay = Replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(
        lambda: replay.session.status()["state"] == "failed", max_ticks=1600
    )
    assert replay.session.status()["completed_legs"] == []
    assert (
        sum(
            e["action"] == "authorize_breakaway"
            for e in replay.controller.status()["events"]
        )
        == 1
    )


def test_stopping_short_during_cue_is_not_target_arrival(settings):
    replay = Replay(settings)
    replay.begin()
    replay.run_until(lambda: replay.audit.get("phase") == "cue")
    for i in range(1, 5):
        replay.tick(q=(0.0, 0.0, -i * 0.0001, 0.0, 0.0, 0.0))
    replay.run_until(lambda: replay.session.status()["state"] == "failed")
    assert replay.session.status()["reason"] == "stopped_short_of_target"
    assert replay.session.status()["completed_legs"] == []
    assert replay.session.status()["outcome"]["severity"] == "warning"
    assert replay.audit["phase"] == "inactive"
    assert replay.session.status()["pending_dispatch"] is False


def test_next_leg_waits_for_actual_zero_tail_and_restored_base_damping(settings):
    replay = Replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    # Final conditioner retains a nonzero command despite a zero request.
    # This is synthetic receipt input, never an actuator dispatch.
    for _ in range(30):
        replay.tick(command_delta=(0.0, 0.0, -0.02, 0.0, 0.0, 0.0))
        assert replay.session.status()["step_index"] == 0
        assert replay.session.status()["completed_legs"] == []
        assert replay.commands["driver_kd_nm_s_per_rad"] != ZERO
    replay.run_until(lambda: replay.session.status()["step_index"] == 1)
    assert replay.commands["total_nm"] == ZERO
    assert replay.commands["driver_kd_nm_s_per_rad"] == ZERO


@pytest.mark.parametrize("final_error,tolerance,expected", [
    (.00263106691837, None, "preparing"), (.00301456320286, None, "preparing"),
    (.0030970892310142467, None, "preparing"),
    (.0030970892310142467, .005, "preparing"), (.0051, .005, "failed"),
])
def test_arrival_uses_original_target_after_zero_and_new_still_dwell(settings, final_error, tolerance, expected):
    replay = Replay(settings, acquisition_arrival_tolerance_rad=tolerance, excursion_rad=.010)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    target = replay.plan["steps"][0]["target_qpos_rad"][2]
    q = list(replay.q)
    # This is a prescribed unloading/encoder response, not a physical model.
    final_q = target + final_error
    for _ in range(30):
        q[2] += max(-.0001, min(.0001, final_q - q[2]))
        replay.tick(q=q, command_delta=(0., 0., -.02, 0., 0., 0.))
        assert replay.session.status()["state"] == "releasing"
        assert replay.session.status()["completed_legs"] == []
    assert replay.q[2] == pytest.approx(final_q)
    replay.run_until(lambda: replay.controller.owner_snapshot()["operator_phase"] == "inactive")
    zero_at = replay.now
    for _ in range(10):
        replay.tick()
        assert replay.session.status()["state"] == "releasing"
        assert replay.session.status()["completed_legs"] == []
    replay.run_until(lambda: replay.session.status()["state"] != "releasing")
    state = replay.session.status()
    assert replay.now - zero_at >= settings.staged_assist.settle_dwell_sec
    assert state["state"] == expected
    assert replay.commands["total_nm"] == replay.commands["driver_kd_nm_s_per_rad"] == ZERO
    if expected == "failed":
        assert state["reason"] == "stopped_short_of_target"
        assert state["completed_legs"] == []
    else:
        leg = state["completed_legs"][0]
        assert leg["target_position_rad"] == target
        assert leg["measured_position_rad"] == pytest.approx(final_q)
        assert leg["target_error_rad"] == pytest.approx(-final_error)
        assert leg["final_zero_and_still_confirmed"] is True
        assert leg["tracking_target_reached"] is (final_error <= .0039)
        assert leg["acquisition_arrival_tolerance_rad"] == pytest.approx(tolerance or .0039)


def test_arrival_waits_for_stop_gate_not_only_looser_prepare_gate(settings):
    replay = Replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    replay.run_until(lambda: replay.controller.owner_snapshot()["operator_phase"] == "inactive")
    q = list(replay.q)
    for _ in range(25):
        q[0] += .0001  # Below preparation speed, above final stop speed.
        replay.tick(q=q)
        assert replay.session.status()["completed_legs"] == []
        assert replay.session.status()["state"] == "releasing"
    assert replay.controller.owner_snapshot()["prepare_gate"] == "ready"
    assert replay.controller.owner_snapshot()["arrival_stop_gate"] != "ready"
    stopped_at = replay.now
    replay.run_until(lambda: replay.session.status()["step_index"] == 1)
    assert replay.now - stopped_at >= settings.staged_assist.settle_dwell_sec
    assert replay.commands["total_nm"] == ZERO


@pytest.mark.parametrize("disturbance", ["tail", "human", "lease", "receipt"])
def test_arrival_dwell_cannot_survive_tail_or_revoked_receipt_and_permit(settings, disturbance):
    replay = Replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    replay.run_until(lambda: replay.controller.owner_snapshot()["operator_phase"] == "inactive")
    for _ in range(5):
        replay.tick()
    assert replay.session.status()["completed_legs"] == []
    if disturbance == "tail":
        replay.tick(command_delta=(0., 0., -.02, 0., 0., 0.))
        resumed_zero_at = replay.now
        for _ in range(10):
            replay.tick()
            assert replay.session.status()["completed_legs"] == []
        replay.run_until(lambda: replay.session.status()["step_index"] == 1)
        assert replay.now - resumed_zero_at >= settings.staged_assist.settle_dwell_sec
    elif disturbance == "receipt":
        replay.tick(ack=False)
        replay.receipt["sent_joint_names"] = ["joint3"]
        with pytest.raises(ValueError, match="complete advancing"):
            replay.ack()
        assert replay.session.status()["completed_legs"] == []
    else:
        replay.tick(human=disturbance == "human", permit=disturbance != "lease")
        assert replay.session.status()["state"] in {"cancelled", "failed"}
        assert replay.session.status()["completed_legs"] == []


def test_contact_during_release_does_not_restart_withdrawal_or_deadline(settings):
    replay = Replay(settings)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    before = replay.session.status()["events"]
    replay.tick(human=True)
    assert replay.session.status()["events"] == before
    replay.run_until(lambda: replay.session.status()["state"] == "cancelled")
    assert replay.session.status()["completed_legs"] == []


@pytest.mark.parametrize(
    "bad", ["duplicate_sequence", "nan_position", "nan_permit", "nan_clock"]
)
def test_invalid_feedback_latches_withdrawal_instead_of_resuming(settings, bad):
    replay = Replay(settings)
    replay.begin()
    replay.run_until(lambda: replay.session.status()["state"] == "moving")
    with pytest.raises(ValueError):
        replay.session.step(
            now_sec=float("nan") if bad == "nan_clock" else replay.now + 0.01,
            feedback_sequence=replay.sequence
            if bad == "duplicate_sequence"
            else replay.sequence + 1,
            qpos_rad=(float("nan"), *ZERO[1:]) if bad == "nan_position" else ZERO,
            qvel_rad_s=ZERO,
            condition=replay.condition,
            human_intervention=False,
            operator_permit_timestamp_sec=float("nan")
            if bad == "nan_permit"
            else replay.now + 0.01,
        )
    assert replay.session.status()["state"] == "releasing"
    replay.run_until(lambda: replay.session.status()["state"] == "failed")


def test_existing_raw_velocity_protection_still_stops_automatic_batch(settings):
    replay = Replay(settings)
    replay.begin()
    replay.run_until(lambda: replay.session.status()["state"] == "moving")
    replay.tick(
        q=(0.0, 0.0, 0.0006, 0.0, 0.0, 0.0)
    )  # .06 rad/s > original .05 hard guard
    assert "joint_velocity_exceeded" in replay.controller.owner_snapshot()["fault"]
    assert replay.session.status()["completed_legs"] == []
    assert replay.session.status()["state"] in {"releasing", "failed"}


def test_batch_exposure_budget_and_lease_only_withdraw_never_expand_authority(settings):
    replay = Replay(
        settings, limits=CalibrationBatchLimits(120.0, 12.0, 2.0, 0.1, 0.04, 0.05)
    )
    replay.begin()
    replay.tick()
    replay.run_until(lambda: replay.session.status()["state"] == "failed")
    assert replay.session.status()["reason"] == "active_command_exposure_budget"
    assert replay.session.status()["completed_legs"] == []


def test_acquisition_leg_budget_and_batch_budget_are_independent_and_finite(settings):
    replay = Replay(settings, limits=CalibrationBatchLimits(300., 20., 2., .1, .04, 20., .12))
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "failed")
    status = replay.session.status()
    assert status["reason"] == "leg_command_exposure_budget"
    assert status["batch_command_exposure_nm_sec"] < 20.
    assert status["completed_legs"] == []
    assert replay.commands["total_nm"] == ZERO

    replay = Replay(settings, limits=CalibrationBatchLimits(300., 20., 2., .1, .04, .12, .10))
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "failed", move=True)
    status = replay.session.status()
    assert status["reason"] == "active_command_exposure_budget"
    assert status["step_index"] > 0
    assert status["leg_command_exposure_nm_sec"] < .10
    assert status["batch_command_exposure_nm_sec"] > status["leg_command_exposure_nm_sec"]
    assert replay.commands["total_nm"] == ZERO


def test_acquisition_budget_counts_actual_withdrawal_tail_and_only_resets_completed_leg(settings):
    replay = Replay(settings, limits=CalibrationBatchLimits(300., 20., 2., .1, .04, 80., 8.))
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    before = replay.session.status()["batch_command_exposure_nm_sec"]
    for _ in range(20):
        replay.tick(command_delta=(0., 0., -.02, 0., 0., 0.))
        status = replay.session.status()
        assert status["step_index"] == 0
        assert status["leg_command_exposure_nm_sec"] == pytest.approx(status["batch_command_exposure_nm_sec"])
    assert status["withdrawal_command_exposure_nm_sec"] > 0
    assert status["batch_command_exposure_nm_sec"] > before
    replay.run_until(lambda: replay.session.status()["step_index"] == 1)
    status = replay.session.status()
    assert status["leg_command_exposure_nm_sec"] == 0
    assert status["batch_command_exposure_nm_sec"] > before
    assert status["completed_legs"][0]["command_exposure_nm_sec"] == pytest.approx(status["batch_command_exposure_nm_sec"])
    assert replay.commands["total_nm"] == replay.commands["driver_kd_nm_s_per_rad"] == ZERO
    assert status["budget_plan"]["maximum_leg_budgets_sum_nm_sec"] == 112.
    assert status["budget_plan"]["all_leg_caps_fit_batch"] is False


@pytest.mark.parametrize("tolerance", [.0029, .00501, 0., float("nan"), True])
def test_acquisition_arrival_tolerance_cannot_relax_plan_or_validate_empty_motion(settings, tolerance):
    with pytest.raises((ValueError, TypeError)):
        Replay(settings, acquisition_arrival_tolerance_rad=tolerance, excursion_rad=.010)


@pytest.mark.parametrize("leg_limit", [0., -.1, float("nan"), True, 81.])
def test_invalid_leg_exposure_limits_are_rejected(leg_limit):
    with pytest.raises((ValueError, TypeError)):
        CalibrationBatchLimits(300., 20., 2., .1, .04, 80., leg_limit)


def test_acquisition_leg_budget_cannot_spend_its_withdrawal_reserve(settings):
    with pytest.raises(ValueError, match="withdrawal reserve"):
        Replay(settings, limits=CalibrationBatchLimits(300., 20., 2., .1, .04, 80., .05))


def test_acquisition_tolerance_does_not_accept_no_motion_or_skip_original_guards(settings):
    replay = Replay(settings, acquisition_arrival_tolerance_rad=.005, excursion_rad=.010)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "failed")
    assert replay.session.status()["completed_legs"] == []
    assert replay.commands["total_nm"] == ZERO
    replay = Replay(settings, acquisition_arrival_tolerance_rad=.005, excursion_rad=.010)
    replay.begin()
    replay.run_until(lambda: replay.session.status()["state"] == "moving")
    replay.tick(q=(0., 0., .0006, 0., 0., 0.))
    assert "joint_velocity_exceeded" in replay.controller.owner_snapshot()["fault"]
    assert replay.session.status()["completed_legs"] == []


def test_acquisition_with_unloaded_position_deviation_continues_native_plan_and_records_quality(settings):
    replay = Replay(settings, acquisition_arrival_tolerance_rad=.005, excursion_rad=.010,
                    limits=CalibrationBatchLimits(300., 20., 2., .1, .04, 80., 8.))
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "releasing", move=True)
    target = replay.plan["steps"][0]["target_qpos_rad"][2]
    final = target + .0045
    for _ in range(30):
        q = list(replay.q)
        q[2] += max(-.0001, min(.0001, final-q[2]))
        replay.tick(q=q, command_delta=(0., 0., -.02, 0., 0., 0.))
        assert replay.session.status()["step_index"] == 0
    replay.run_until(lambda: replay.session.status()["step_index"] == 1)
    first = replay.session.status()["completed_legs"][0]
    assert first["tracking_target_reached"] is False
    assert first["target_error_rad"] == pytest.approx(-.0045)
    replay.run_until(lambda: replay.session.status()["state"] == "complete", max_ticks=10000, move=True)
    result = replay.session.result_record()
    assert len(result["completed_legs"]) == 14
    assert result["reason"] == "all_legs_collected_with_target_deviation"
    assert sum(not leg["tracking_target_reached"] for leg in result["completed_legs"]) == 1
    assert result["batch_command_exposure_nm_sec"] < 80.
    assert replay.commands["total_nm"] == replay.commands["driver_kd_nm_s_per_rad"] == ZERO


def test_short_original_breakaway_budget_remains_a_failure_not_an_auto_retry(settings):
    short = replace(
        settings,
        staged_assist=replace(settings.staged_assist, breakaway_max_duration_sec=0.20),
    )
    replay = Replay(short)
    replay.begin(breakaway=True)
    replay.run_until(lambda: replay.session.status()["state"] == "failed", move=True)
    assert "duration_exhausted" in replay.session.status()["reason"]
    assert replay.session.status()["completed_legs"] == []


def test_speed_configuration_requires_latest_zero_receipt_and_original_ceiling(
    settings,
):
    replay = Replay(settings)
    controller = replay.controller
    controller.configure_reference_velocity(0.0075)
    controller.configure_reference_velocity(0.015)
    with pytest.raises(ValueError, match="original motion profile"):
        controller.configure_reference_velocity(0.016)
    replay.tick(ack=False)
    with pytest.raises(ValueError, match="acknowledged zero"):
        controller.configure_reference_velocity(0.0075)
    replay.ack()
    replay.begin()
    replay.tick()
    with pytest.raises(ValueError, match="acknowledged zero"):
        controller.configure_reference_velocity(0.0075)


def test_offline_session_cannot_be_passed_to_hardware_owner(settings):
    replay = Replay(settings)
    with pytest.raises(probe.LowGainProbeError, match="offline calibration"):
        probe.run_probe(SimpleNamespace(), torque_console=replay.session)


def test_batch_requires_confirmed_motion_and_protected_withdrawal(settings):
    unconfirmed = replace(
        settings,
        staged_assist=replace(
            settings.staged_assist,
            motion_onset_confirm_displacement_rad=None,
        ),
    )
    with pytest.raises(ValueError, match="confirmed-motion"):
        Replay(unconfirmed)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.update(execution_enabled=True),
        lambda p: p["steps"][0].update(max_reference_velocity_rad_s=0.1),
        lambda p: p["steps"][0]["target_qpos_rad"].__setitem__(0, 0.01),
        lambda p: p["condition"].update(gravity_scale=0.99),
        lambda p: p["steps"][1].update(trial_id=p["steps"][0]["trial_id"]),
    ],
)
def test_invalid_plans_are_rejected_before_arming(settings, mutation):
    with pytest.raises(ValueError):
        Replay(settings, plan_change=mutation)

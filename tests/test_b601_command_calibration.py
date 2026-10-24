"""Synthetic identification and offline trace admission; never CAN or motion."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, replace

import pytest

from control_station.b601_torque_console.calibration_report import summarize_calibration
from control_station.b601_torque_console.report import main as report_main
from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition,
    CalibrationObservation,
    fit_local_command_model,
    plan_local_sweeps,
    predict_command_residual,
    reduce_command_window,
)


def condition(**changes):
    return CalibrationCondition(
        **{
            "run_id": "synthetic-run",
            "config_sha256": "a" * 64,
            "urdf_sha256": "b" * 64,
            "gravity_scale": 1.0,
            "gravity_joint_scales": (1, 0.96, 1, 1, 1, 1),
            "gravity_joint_offsets_rad": (0,) * 6,
            "gravity_joint_bias_nm": (0, -0.22, -0.11, -0.08, 0, 0),
            "gravity_coulomb_friction_nm": (0,) * 6,
            "gravity_coulomb_velocity_threshold_rad_s": 0.02,
            "driver_kd_nm_s_per_rad": (0,) * 6,
            **changes,
        }
    )


def observations(*, joint="joint3", context=None):
    context = context or condition()
    j = int(joint[-1]) - 1

    def sample(trial, velocity, load):
        q = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        q[j] += (load - 4) * 0.01
        gravity = [0, 7, 5, 1, 0.1, 0]
        gravity[j] = load
        direction = math.copysign(1, velocity)
        residual = (
            0.02
            + 0.12 * direction
            + 4 * velocity
            + 0.03 * load
            + 0.025 * direction * abs(load)
        )
        return CalibrationObservation(
            context, joint, trial, tuple(q), tuple(gravity), velocity, residual
        )

    train = [
        sample(f"train-{i}", v, g)
        for i, v in enumerate((-0.006, -0.012, 0.006, 0.012))
        for g in (3.0, 4.0, 5.0)
    ]
    valid = [
        sample(f"valid-{i}", v, g)
        for i, v in enumerate((-0.009, 0.009))
        for g in (3.5, 4.5)
    ]
    return train, valid


@pytest.mark.parametrize("joint", ["joint2", "joint3", "joint6"])
def test_local_fit_predicts_independent_trials_with_load_dependence(joint):
    train, valid = observations(joint=joint)
    fitted = fit_local_command_model(train, valid)
    assert fitted["status"] == "OFFLINE_CANDIDATE"
    assert fitted["load_dependence_identified"] is True
    assert fitted["coefficients"] == pytest.approx([0.02, 0.12, 4.0, 0.03, 0.025])
    assert fitted["validation_rmse_nm"] < 1e-10
    assert fitted["auto_apply"] is False
    assert fitted["hardware_approved"] is False
    # Persist and load the model: tuple/list differences must not lose identity.
    restored = json.loads(json.dumps(fitted, allow_nan=False))
    for sample in valid:
        assert predict_command_residual(restored, sample) == pytest.approx(
            sample.command_residual_nm
        )


@pytest.mark.parametrize(
    "change",
    [
        {"gravity_scale": 0.9},
        {"gravity_joint_scales": (1, 1, 1, 1, 1, 1)},
        {"gravity_joint_offsets_rad": (0, 0.1, 0, 0, 0, 0)},
        {"gravity_joint_bias_nm": (0, -0.2, -0.1, 0, 0, 0)},
        {"gravity_coulomb_friction_nm": (0, 0.1, 0, 0, 0, 0)},
        {"driver_kd_nm_s_per_rad": (0, 0.1, 0, 0, 0, 0)},
        {"config_sha256": "c" * 64},
        {"urdf_sha256": "d" * 64},
        {"run_id": "other-unknown-payload"},
    ],
)
def test_different_gravity_source_gain_or_unknown_payload_run_cannot_share_fit(change):
    train, valid = observations()
    changed = replace(valid[0], condition=condition(**change))
    result = fit_local_command_model(train, [changed, *valid[1:]])
    assert result["reasons"] == ["joint_or_gravity_or_source_condition_mismatch"]
    model = fit_local_command_model(train, valid)
    with pytest.raises(ValueError, match="condition"):
        predict_command_residual(model, changed)


def test_joint_identity_and_full_pose_not_just_j3_angle_are_required():
    train, valid = observations()
    changed = replace(valid[0], joint_name="joint2")
    assert (
        fit_local_command_model(train, [changed, *valid[1:]])["status"]
        == "INSUFFICIENT_DATA"
    )
    changed = replace(valid[0], qpos_rad=(0.2, *valid[0].qpos_rad[1:]))
    model = fit_local_command_model(train, valid)
    with pytest.raises(ValueError, match="coverage"):
        predict_command_residual(model, changed)
    changed = replace(
        valid[0], gravity_command_nm=(0, 8, *valid[0].gravity_command_nm[2:])
    )
    with pytest.raises(ValueError, match="coverage"):
        predict_command_residual(model, changed)


def test_whole_trial_validation_rejects_leakage_unseen_regions_and_bad_predictions():
    train, valid = observations()
    leaked = [replace(o, trial_id="train-0") for o in valid]
    assert (
        "whole_trial_validation_leakage"
        in fit_local_command_model(train, leaked)["reasons"]
    )
    outside = [
        replace(o, velocity_rad_s=math.copysign(0.02, o.velocity_rad_s)) for o in valid
    ]
    assert (
        "validation_outside_training_pose_load_or_speed_coverage"
        in fit_local_command_model(train, outside)["reasons"]
    )
    bad = [replace(o, command_residual_nm=o.command_residual_nm + 0.3) for o in valid]
    assert fit_local_command_model(train, bad)["status"] == "VALIDATION_FAILED"


def test_coverage_rejection_still_reports_original_whole_trial_prediction_errors():
    train, valid = observations()
    outside = [replace(o, velocity_rad_s=math.copysign(.02, o.velocity_rad_s),
                       command_residual_nm=o.command_residual_nm+.3) for o in valid]
    result = fit_local_command_model(train, outside)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["reasons"] == ["validation_outside_training_pose_load_or_speed_coverage"]
    diagnostic = result["validation_diagnostics"]
    assert diagnostic["covered_windows"] == 0
    assert diagnostic["total_windows"] == len(valid)
    assert diagnostic["whole_trial_split_preserved"]
    assert len(diagnostic["trial_errors"]) == 2
    assert result["validation_rmse_nm"] > .1
    assert not result["auto_apply"]


def test_closed_loop_window_uses_integrated_input_and_displacement_not_frame_medians():
    zero = (0.0,)*6
    args = {"condition": condition(), "joint_name": "joint3", "trial_id": "closed-loop",
        "times_sec": [0, .1, 1.0],
        "qpos_rad": [(0, 0, q, 0, 0, 0) for q in (0, .001, .01)],
        "gravity_command_nm": [zero]*3,
        "conditioned_command_nm": [(0, 0, u, 0, 0, 0) for u in (0, 1, 1)],
        "driver_kd_nm_s_per_rad": [zero]*3,
        "minimum_motion_rad": .002, "maximum_line_error_rad": .0008,
        "maximum_residual_span_nm": .15}
    assert reduce_command_window(**args)[1] == "changing_command_not_quasisteady_identification"
    args.update(reduction_method="closed_loop_average",
        interval_command_integrals_nm_sec=[zero, (0, 0, .9, 0, 0, 0)],
        interval_gravity_integrals_nm_sec=[zero]*2)
    observation, reason = reduce_command_window(**args)
    assert reason is None
    assert observation.command_residual_nm == pytest.approx(.9)
    assert observation.velocity_rad_s == pytest.approx(.01)
    assert observation.qpos_rad[2] == pytest.approx(.005)
    for integrals in (None, [zero], [zero, (float("nan"),)*6]):
        with pytest.raises(ValueError):
            reduce_command_window(**{**args, "interval_command_integrals_nm_sec": integrals})
    # A steady command cannot make coupled motion, reversal or a stop cruise.
    for positions, why in (([0, .004, .003], "nonsteady_or_coupled_motion"),
                           ([0, 0, .01], "stop_or_reversal_not_cruise")):
        # Use equal time intervals and a loose line residual only for the
        # explicit stop case, so its extra cruise condition is exercised.
        changed = {**args, "times_sec": [0, .5, 1],
                   "qpos_rad": [(0, 0, q, 0, 0, 0) for q in positions]}
        if why == "stop_or_reversal_not_cruise":
            changed["maximum_line_error_rad"] = .01
        assert reduce_command_window(**changed)[1] == why


def test_one_speed_or_one_direction_does_not_identify_friction_parameters():
    train, valid = observations()
    fixed_speed = [
        replace(o, velocity_rad_s=math.copysign(0.006, o.velocity_rad_s)) for o in train
    ]
    assert (
        "unidentifiable_features_add_speeds_or_load_conditions"
        in fit_local_command_model(fixed_speed, valid)["reasons"]
    )
    one_way = [o for o in train if o.velocity_rad_s > 0]
    assert fit_local_command_model(one_way, valid)["status"] == "INSUFFICIENT_DATA"


@pytest.mark.parametrize(
    "changes",
    [
        {"gravity_scale": float("nan")},
        {"gravity_scale": True},
        {"gravity_joint_scales": (1,)},
        {"config_sha256": "unknown"},
        {"run_id": ""},
        {"driver_kd_nm_s_per_rad": (-1,) * 6},
    ],
)
def test_calibration_context_is_complete_and_finite(changes):
    with pytest.raises((TypeError, ValueError)):
        condition(**changes)


def test_sweep_plan_is_bounded_reversible_and_has_no_execution_authority():
    result = plan_local_sweeps(
        condition=condition(),
        joint_name="joint3",
        anchor_qpos_rad=(0,) * 6,
        position_limits_rad=((-1, 1),) * 6,
        excursion_rad=0.02,
        speeds_rad_s=(0.006, 0.012),
        max_reference_velocity_rad_s=0.015,
    )
    assert result["execution_enabled"] is False
    assert len(result["steps"]) == 14
    assert result["steps"][0]["role"] == "positioning_not_for_fit"
    assert result["steps"][-1]["target_qpos_rad"] == [0] * 6
    assert {step["role"] for step in result["steps"]} == {
        "training",
        "validation",
        "positioning_not_for_fit",
        "return_not_for_fit",
    }
    for step in result["steps"]:
        assert all(q == 0 for i, q in enumerate(step["target_qpos_rad"]) if i != 2)
        assert abs(step["target_qpos_rad"][2]) <= 0.02
        assert step["max_reference_velocity_rad_s"] <= 0.015


def test_local_identification_reuses_excursion_return_with_preassigned_fit_roles():
    plan = plan_local_sweeps(condition=condition(), joint_name="joint3",
        anchor_qpos_rad=(0.,)*6, position_limits_rad=((-1., .005),)*6,
        excursion_rad=.02, speeds_rad_s=(.0075, .015), max_reference_velocity_rad_s=.015,
        purpose="local_identification")
    assert plan["execution_enabled"] is False
    steps = plan["steps"]
    assert len(steps) == 12
    assert [s["role"] for s in steps] == ["training"]*8+["validation"]*4
    for i, step in enumerate(steps):
        assert step["target_qpos_rad"] == [0., 0., -.02 if i % 2 == 0 else 0., 0., 0., 0.]
        assert step["max_reference_velocity_rad_s"] == (.0075 if i % 4 < 2 else .015)


@pytest.mark.parametrize(
    "changes",
    [
        {"joint_name": "joint7"},
        {"excursion_rad": 2.0},
        {"speeds_rad_s": (0.01,)},
        {"speeds_rad_s": (0.01, 0.2)},
        {"training_repeats": True},
        {"excursion_rad": float("inf")},
    ],
)
def test_sweep_plan_never_expands_approved_bounds(changes):
    args = {
        "condition": condition(),
        "joint_name": "joint3",
        "anchor_qpos_rad": (0,) * 6,
        "position_limits_rad": ((-1, 1),) * 6,
        "excursion_rad": 0.02,
        "speeds_rad_s": (0.006, 0.012),
        "max_reference_velocity_rad_s": 0.015,
    }
    with pytest.raises((ValueError, TypeError)):
        plan_local_sweeps(**(args | changes))


def trace_fixture():
    context = asdict(condition())
    rows = [
        {
            "kind": "calibration_context",
            "schema": "b601.command-calibration-context.v1",
            "profile": context,
            "planning_limits": {
                "position_limits_rad": [[-1, 1]] * 6,
                "joint_velocity_guard_rad_s": 0.05,
            },
        }
    ]
    events = []
    sequence = 0
    speeds = (0.006, -0.006, 0.012, -0.012, 0.006, -0.006, 0.012, -0.012, 0.009, -0.009)
    names = [f"joint{i}" for i in range(1, 7)]
    for trial, speed in enumerate(speeds, 1):
        start = 1 + (trial - 1) * 0.8
        meta = {
            "kind": "unloaded_response",
            "pair_label": str(trial),
            "condition_label": "current",
            "axis": "joint",
            "direction": "positive" if speed > 0 else "negative",
            "entry_direction": "unknown",
            "contact": "none",
            "note": "",
        }
        events += [
            {
                "schema": "b601.transparency-annotation.v1",
                "event": "begin",
                "sequence": len(events) + 1,
                "trial_id": trial,
                "recorded_monotonic_sec": start,
                "metadata": meta,
                "owner_snapshot": {"joint_name": "joint3", "qpos_rad": [0] * 6},
            },
            {
                "schema": "b601.transparency-annotation.v1",
                "event": "finish",
                "sequence": len(events) + 2,
                "trial_id": trial,
                "recorded_monotonic_sec": start + 0.7,
                "observation": {
                    "perceived_strength": "unknown",
                    "perceived_direction": "unknown",
                    "manual_force_n": None,
                    "force_instrument": "",
                    "force_observation": "none",
                    "note": "",
                },
            },
        ]
        for tick in range(1, 61):
            sequence += 1
            stamp = start + tick * 0.01
            residual = 0.02 + 0.12 * math.copysign(1, speed) + 4 * speed
            rows.append(
                {
                    "kind": "active_dispatch",
                    "recorded_monotonic_sec": stamp + 0.001,
                    "feedback_timestamp_sec": stamp,
                    "sequence": sequence,
                    "startup_state": "ACTIVE",
                    "gravity_ramp_fraction": 1.0,
                    "startup_handoff_fraction": 1.0,
                    "qpos_rad": [0, 0.1, 0.2 + speed * tick * 0.01, 0, 0, 0],
                    "gravity_tau_nm": [0, 7, 5, 1, 0, 0],
                    "conditioned_tau_nm": [0, 7, 5 + residual, 1, 0, 0],
                    "vendor_kp_nm_per_rad": [0] * 6,
                    "vendor_kd_nm_s_per_rad": [0] * 6,
                    "torque_limited": False,
                    "slew_limited": False,
                    "overlay_limited": False,
                    "torque_console": {
                        "phase": "armed",
                        "joint_name": "joint3",
                        "fault": None,
                    },
                    "host_observation": {"host_receive_freshness_verified": True},
                    "dispatch_receipt": {
                        "accepted": True,
                        "mode": "mit",
                        "frame_id": "b601_rs/joint_mit_reference",
                        "source_sequence": sequence,
                        "requested_at_sec": stamp + 0.0002,
                        "completed_at_sec": stamp + 0.0008,
                        "sent_joint_names": names,
                        "requested_joint_names": names,
                        "errors": [],
                        "failed_joint_names": [],
                    },
                }
            )
    return events, rows


def test_streaming_report_fits_commands_not_human_force_and_is_immutable():
    events, rows = trace_fixture()
    before = copy.deepcopy((events, rows))
    report = summarize_calibration(iter(events), iter(rows), run_id="test")
    assert report["status"] == "OFFLINE_CANDIDATE"
    assert report["offline_candidates"] == 1
    assert report["steady_motion_windows"] == 10
    assert report["groups"][0]["fit"]["coefficients"] == pytest.approx([0.02, 0.12, 4])
    assert report["motion_commands_sent"] is False
    (proposal,) = report["experiment_proposals"]
    assert proposal["status"] == "DRAFT_NOT_EXECUTABLE"
    assert proposal["plan"]["execution_enabled"] is False
    assert len(proposal["plan"]["steps"]) == 14
    assert (events, rows) == before


def automatic_trace_fixture():
    events, rows = trace_fixture()
    completed = []
    for i, event in enumerate(events[::2], 1):
        start = event["recorded_monotonic_sec"]
        end = start + 0.7
        role = "validation" if i <= 2 else "training"
        for row in rows[1:]:
            if start < row["feedback_timestamp_sec"] < end:
                row["vendor_kd_nm_s_per_rad"] = [0.4, 0.6, 0.4, 0.6, 0.2, 0.1]
                row["torque_console"].update(
                    controller_mode="staged_assist",
                    phase="track",
                    operator_armed=True,
                    motion_control={"phase": "track"},
                    calibration_trial={
                        "batch_id": "batch-1",
                        "trial_id": f"leg-{i}",
                        "role": role,
                        "state": "moving",
                    },
                )
        completed.append(
            {"trial_id": f"leg-{i}", "role": role, "completed_at_sec": end}
        )
    rows.append(
        {
            "kind": "calibration_batch_result",
            "schema": "b601.calibration-batch-result.v1",
            "state": "complete",
            "pending_dispatch": False,
            "batch_id": "batch-1",
            "completed_legs": completed,
            "recorded_monotonic_sec": completed[-1]["completed_at_sec"],
        }
    )
    return events, rows


def test_automatic_report_uses_reserved_validation_roles_and_actual_damping():
    events, rows = automatic_trace_fixture()
    report = summarize_calibration(events, rows, run_id="automatic")
    assert report["status"] == "OFFLINE_CANDIDATE", report
    model = report["groups"][0]["fit"]
    assert model["validation_trial_ids"] == ["1/batch-1/leg-1", "2/batch-1/leg-2"]
    assert model["active_driver_kd_nm_s_per_rad"] == (0.4, 0.6, 0.4, 0.6, 0.2, 0.1)
    assert "not_physical_friction" in model["quantity"]


def test_terminal_batch_may_be_followed_by_passive_owner_dispatch():
    import copy

    events, rows = automatic_trace_fixture()
    tail = copy.deepcopy(rows[-2])
    tail["feedback_timestamp_sec"] = rows[-1]["recorded_monotonic_sec"] + .01
    tail["recorded_monotonic_sec"] = tail["feedback_timestamp_sec"] + .002
    events[-1]["recorded_monotonic_sec"] = tail["recorded_monotonic_sec"] + .01
    tail["torque_console"]["calibration_trial"]["state"] = "complete"
    tail["torque_console"]["phase"] = "inactive"
    report = summarize_calibration(events, [*rows, tail], run_id="automatic")
    assert report["status"] == "OFFLINE_CANDIDATE"
    tail["torque_console"]["calibration_trial"]["state"] = "moving"
    with pytest.raises(ValueError, match="terminal batch"):
        summarize_calibration(events, [*rows, tail], run_id="automatic")


@pytest.mark.parametrize(
    "phase", ["cue", "breakaway", "capture", "brake", "soft_hold", "releasing"]
)
def test_report_never_fits_non_tracking_staged_phases(phase):
    events, rows = automatic_trace_fixture()
    for row in rows[1:-1]:
        row["torque_console"]["phase"] = phase
    result = summarize_calibration(events, rows, run_id="automatic")
    assert result["steady_motion_windows"] == 0
    assert result["rejected"]["not_steady_tracking_phase"] == 600


def test_automatic_report_requires_terminal_leg_identity_and_keeps_nonfit_windows_separate():
    events, rows = automatic_trace_fixture()
    result = summarize_calibration(events, rows[:-1], run_id="automatic")
    assert result["steady_motion_windows"] == 0
    assert result["rejected"]["automatic_leg_completion_not_confirmed"] == 10
    rows[-1]["state"] = "failed"
    rows[-1]["completed_legs"] = rows[-1]["completed_legs"][:1]
    result = summarize_calibration(events, rows, run_id="automatic")
    assert result["steady_motion_windows"] == 1
    for row in rows[1:-1]:
        row["torque_console"]["calibration_trial"]["role"] = "positioning_not_for_fit"
    rows[-1]["completed_legs"][0]["role"] = "positioning_not_for_fit"
    result = summarize_calibration(events, rows, run_id="automatic")
    assert result["steady_motion_windows"] == 0
    assert result["observed_steady_motion_windows"] == 10
    assert all(w["exclusion_reason"] == "automatic_nonfit_motion_window" for w in result["motion_windows"])


@pytest.mark.parametrize("terminal_state", ["failed", "cancelled"])
def test_completed_motion_window_survives_later_leg_failure(terminal_state):
    events, rows = automatic_trace_fixture()
    rows[-1].update(state=terminal_state, reason="stopped_short_of_target",
        completed_legs=rows[-1]["completed_legs"][:-1],
        calibration_trial={"batch_id": "batch-1", "trial_id": "leg-10", "role": "training", "state": terminal_state})
    report = summarize_calibration(events, rows, run_id="automatic")
    assert report["steady_motion_windows"] == 10
    assert report["retained_partial_leg_windows"] == 1
    window = report["motion_windows"][-1]
    assert window["source_leg_completed"] is False
    assert window["source_batch_state"] == terminal_state
    assert window["source_batch_reason"] == "stopped_short_of_target"
    assert window["model_eligible"] is True
    assert window["start_sec"] < window["end_sec"] < window["recorded_end_sec"]
    assert window["sample_count"] >= 40
    assert window["observation"]["active_driver_kd_nm_s_per_rad"] == (0.4, 0.6, 0.4, 0.6, 0.2, 0.1)
    assert report["status"] == "OFFLINE_CANDIDATE"
    assert report["auto_apply"] is report["hardware_approved"] is False


def test_partial_leg_retention_never_spans_a_rejected_dispatch_or_trains_its_tail():
    events, rows = automatic_trace_fixture()
    rows[-1].update(state="failed", reason="dispatch_rejected",
        completed_legs=rows[-1]["completed_legs"][:-1],
        calibration_trial={"batch_id": "batch-1", "trial_id": "leg-10", "role": "training", "state": "failed"})
    fault_index = 1+9*60+45
    fault_time = rows[fault_index]["recorded_monotonic_sec"]
    rows.insert(fault_index, {"kind": "automatic_calibration_dispatch_rejected", "recorded_monotonic_sec": fault_time})
    for row in rows[fault_index+1:-1]:
        row["torque_console"]["calibration_trial"]["state"] = "releasing"
    report = summarize_calibration(events, rows, run_id="automatic")
    last_window = report["motion_windows"][-1]
    assert report["retained_partial_leg_windows"] == 1
    assert last_window["recorded_end_sec"] < fault_time
    assert last_window["model_eligible"] is True
    # A fault before the first whole window leaves no valid prefix to retain.
    events, rows = automatic_trace_fixture()
    rows.insert(1+9*60+20, {"kind": "torque_console_guard"})
    report = summarize_calibration(events, rows, run_id="automatic")
    assert report["steady_motion_windows"] == 9


@pytest.mark.parametrize("marker_change", [{"trial_id": "other"}, {"role": "validation"}, {"state": "moving"}])
def test_partial_leg_needs_matching_terminal_identity(marker_change):
    events, rows = automatic_trace_fixture()
    rows[-1].update(state="failed", completed_legs=rows[-1]["completed_legs"][:-1],
        calibration_trial={"batch_id": "batch-1", "trial_id": "leg-10", "role": "training", "state": "failed", **marker_change})
    result = summarize_calibration(events, rows, run_id="automatic")
    assert result["steady_motion_windows"] == 9
    assert result["retained_partial_leg_windows"] == 0


def slow_quantized_trace():
    events, template = trace_fixture()
    events = events[:2]
    events[-1]["recorded_monotonic_sec"] = 3.3
    rows = [template[0]]
    for tick in range(1, 221):
        row = copy.deepcopy(template[1])
        stamp = 1.+tick*.01
        row.update(sequence=tick, feedback_timestamp_sec=stamp, recorded_monotonic_sec=stamp+.001,
            qpos_rad=[0, .1, .2+round(.0015*tick*.01/.000384)*.000384, 0, 0, 0])
        row["dispatch_receipt"].update(source_sequence=tick, requested_at_sec=stamp+.0002, completed_at_sec=stamp+.0008)
        rows.append(row)
    return events, rows


def test_adaptive_offline_window_resolves_slow_quantized_motion_without_lowering_displacement_gate():
    events, rows = slow_quantized_trace()
    baseline = summarize_calibration(events, rows, run_id="slow")
    candidate = summarize_calibration(events, rows, run_id="slow", max_window_sec=2.)
    assert baseline["steady_motion_windows"] == 0
    assert candidate["steady_motion_windows"] == 1
    window = candidate["motion_windows"][0]
    assert .4 < window["end_sec"]-window["start_sec"] <= 2.
    assert window["observation"]["velocity_rad_s"] == pytest.approx(.0015, abs=.0001)
    assert candidate["criteria"]["minimum_motion_rad"] == baseline["criteria"]["minimum_motion_rad"] == .002
    assert candidate["status"] == "INSUFFICIENT_DATA"  # One window cannot identify a model.


@pytest.mark.parametrize("fault", ["changing_command", "guard", "contact"])
def test_longer_analysis_window_keeps_stability_contact_and_continuity_gates(fault):
    events, rows = slow_quantized_trace()
    if fault == "changing_command":
        for row in rows[1:]:
            row["conditioned_tau_nm"][2] += .03*(row["feedback_timestamp_sec"]-1.)
    elif fault == "guard":
        rows.insert(101, {"kind": "torque_console_guard"})
    else:
        events[0]["metadata"]["contact"] = "unknown"
    result = summarize_calibration(events, rows, run_id="slow", max_window_sec=2.)
    assert result["steady_motion_windows"] == 0


@pytest.mark.parametrize("maximum", [.1, float("nan"), True])
def test_invalid_adaptive_window_rejected(maximum):
    events, rows = slow_quantized_trace()
    with pytest.raises(ValueError):
        summarize_calibration(events, rows, run_id="slow", max_window_sec=maximum)


def test_report_rejects_damping_ramp_and_separates_fixed_damping_regimes():
    events, rows = automatic_trace_fixture()
    for i, row in enumerate(rows[1:-1]):
        row["vendor_kd_nm_s_per_rad"][2] += i * 0.0001
    result = summarize_calibration(events, rows, run_id="automatic")
    assert result["steady_motion_windows"] == 0
    assert result["rejected"]["changing_motor_local_damping"] == 10
    events, rows = automatic_trace_fixture()
    for row in rows[301:-1]:
        row["vendor_kd_nm_s_per_rad"][2] = 0.5
    result = summarize_calibration(events, rows, run_id="automatic")
    assert len(result["groups"]) == 2
    assert {g["active_driver_kd_nm_s_per_rad"][2] for g in result["groups"]} == {
        0.4,
        0.5,
    }


def test_model_prediction_does_not_transfer_between_active_motor_damping():
    train, valid = observations()
    altered = replace(valid[0], active_driver_kd_nm_s_per_rad=(0.4,) * 6)
    result = fit_local_command_model(train, [altered, *valid[1:]])
    assert result["reasons"] == ["active_driver_damping_mismatch"]
    with pytest.raises(ValueError, match="damping"):
        predict_command_residual(fit_local_command_model(train, valid), altered)


@pytest.mark.parametrize("invalid", [None, "short", "misaligned", "clock", "damping"])
def test_numeric_window_boundary_is_independent_of_application_trace(invalid):
    args = {
        "condition": condition(),
        "joint_name": "joint3",
        "trial_id": "domain-window",
        "times_sec": [1.0, 1.1, 1.2],
        "qpos_rad": [(0.0, 0.0, q, 0.0, 0.0, 0.0) for q in (0.0, 0.001, 0.002)],
        "gravity_command_nm": [(0.0,) * 6] * 3,
        "conditioned_command_nm": [(0.0, 0.0, 0.1, 0.0, 0.0, 0.0)] * 3,
        "driver_kd_nm_s_per_rad": [(0.4,) * 6] * 3,
        "minimum_motion_rad": 0.001,
        "maximum_line_error_rad": 0.0005,
        "maximum_residual_span_nm": 0.02,
    }
    if invalid == "short":
        args["times_sec"] = [1.0]
    elif invalid == "misaligned":
        args["gravity_command_nm"] = []
    elif invalid == "clock":
        args["times_sec"] = [1.0, 1.0, 1.2]
    elif invalid == "damping":
        args["driver_kd_nm_s_per_rad"] = [(-0.1,) * 6] * 3
    if invalid:
        with pytest.raises(ValueError):
            reduce_command_window(**args)
    else:
        observation, reason = reduce_command_window(**args)
        assert reason is None
        assert observation.velocity_rad_s == pytest.approx(0.01)
        assert observation.command_residual_nm == pytest.approx(0.1)
        assert observation.active_driver_kd_nm_s_per_rad == (0.4,) * 6


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"startup_state": "GRAVITY_RAMP"}, "startup_or_missing_phase_provenance"),
        ({"torque_limited": True}, "limited_or_unknown_command"),
        ({"overlay_limited": True}, "limited_or_unknown_command"),
        (
            {"vendor_kp_nm_per_rad": [0.1] * 6},
            "motor_local_position_gain_not_supported",
        ),
        (
            {"dispatch_receipt": None},
            "missing_or_incomplete_same_sample_dispatch_receipt",
        ),
        ({"host_observation": {}}, "feedback_freshness_unverified"),
    ],
)
def test_no_model_from_startup_clipped_unmeasured_driver_or_unconfirmed_dispatch(
    change, reason
):
    events, rows = trace_fixture()
    for row in rows[1:]:
        row.update(change)
    report = summarize_calibration(events, rows, run_id="test")
    assert report["status"] == "INSUFFICIENT_DATA"
    assert report["rejected"][reason] == 600
    assert not report["groups"]


def test_contact_unknown_and_missing_context_never_default_to_safe_data():
    events, rows = trace_fixture()
    for event in events[::2]:
        event["metadata"]["contact"] = "unknown"
    report = summarize_calibration(events, rows, run_id="test")
    assert report["rejected"]["no_confirmed_human_free_drive_window"] == 600
    report = summarize_calibration(events, rows[1:], run_id="test")
    assert report["rejected"]["missing_run_gravity_context"] == 600


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_sequence", True),
        ("source_sequence", -1),
        ("sent_joint_names", ["joint3"]),
        ("mode", "position"),
    ],
)
def test_receipt_must_describe_this_complete_mit_dispatch(field, value):
    events, rows = trace_fixture()
    for row in rows[1:]:
        row["dispatch_receipt"][field] = value
    report = summarize_calibration(events, rows, run_id="test")
    assert (
        report["rejected"]["missing_or_incomplete_same_sample_dispatch_receipt"] == 600
    )


def test_receipt_before_feedback_or_after_trace_cannot_train_a_model():
    events, rows = trace_fixture()
    for row in rows[1:]:
        row["dispatch_receipt"]["requested_at_sec"] = (
            row["feedback_timestamp_sec"] - 0.01
        )
    report = summarize_calibration(events, rows, run_id="test")
    assert report["rejected"]["dispatch_receipt_not_causally_aligned"] == 600


@pytest.mark.parametrize(
    "field,value",
    [
        ("qpos_min_rad", [0, 0]),
        ("gravity_max_nm", [float("nan")] * 6),
        ("speed_min_rad_s", -1),
    ],
)
def test_persisted_model_cannot_drop_coverage_checks(field, value):
    train, valid = observations()
    fitted = fit_local_command_model(train, valid)
    fitted["coverage"][field] = value
    with pytest.raises(ValueError):
        predict_command_residual(fitted, valid[0])


def test_stiction_and_rising_commands_do_not_produce_fake_friction_calibration():
    events, rows = trace_fixture()
    for row in rows[1:]:
        row["qpos_rad"] = [0] * 6
    report = summarize_calibration(events, rows, run_id="test")
    assert report["rejected"]["insufficient_motion_not_zero_friction"] == 10
    assert report["recorded_command_windows"] == 10
    assert all(w["model_eligible"] is False and w["observation"] is None for w in report["motion_windows"])
    events, rows = trace_fixture()
    for row in rows[1:]:
        row["conditioned_tau_nm"][2] += row["feedback_timestamp_sec"]
    report = summarize_calibration(events, rows, run_id="test")
    assert report["rejected"]["changing_command_not_quasisteady_identification"] == 10
    assert report["recorded_command_windows"] == 10
    assert report["observed_steady_motion_windows"] == 0
    assert all(w["command_residual_end_nm"] > w["command_residual_start_nm"] for w in report["motion_windows"])


def test_profile_change_splits_models_even_with_same_joint_and_pose():
    events, rows = trace_fixture()
    changed = copy.deepcopy(rows[0])
    changed["profile"]["gravity_joint_scales"] = (1, 0.9, 1, 1, 1, 1)
    rows.insert(301, changed)
    report = summarize_calibration(events, rows, run_id="test")
    assert len(report["groups"]) == 2
    assert len(report["contexts"]) == 2
    assert report["status"] == "INSUFFICIENT_DATA"


def test_offline_cli_writes_a_separate_artifact_without_changing_source(
    tmp_path, capsys
):
    events, rows = trace_fixture()
    run = tmp_path / "run"
    run.mkdir()
    trace = run / "b601_owner.debug.jsonl"
    journal = run / "b601_owner.transparency.jsonl"
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    journal.write_text("".join(json.dumps(row) + "\n" for row in events))
    before = trace.read_bytes(), journal.read_bytes()
    assert (
        report_main(
            ["--run-dir", str(run), "--calibration", "--output-root", str(tmp_path)]
        )
        == 0
    )
    (output,) = tmp_path.glob("b601-torque-report-*/report.json")
    assert json.loads(output.read_text())["hardware_approved"] is False
    assert "no compensation applied" in capsys.readouterr().out
    assert before == (trace.read_bytes(), journal.read_bytes())


def test_offline_cli_reports_insufficient_legacy_data_not_success(tmp_path, capsys):
    events, rows = trace_fixture()
    (tmp_path / "b601_owner.debug.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows[1:])
    )
    (tmp_path / "b601_owner.transparency.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in events)
    )
    assert report_main(["--run-dir", str(tmp_path), "--calibration"]) == 1
    assert "INSUFFICIENT_DATA" in capsys.readouterr().out

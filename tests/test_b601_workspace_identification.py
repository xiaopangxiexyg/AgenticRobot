"""Synthetic command identification and scripted receipts, never real motion."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from test_b601_command_calibration import condition, observations
from test_b601_coordinated_holding import replay, scripted_coordinated_tick
from control_station.b601_torque_console.automatic_calibration import load_automatic_profile
from control_station.b601_torque_console.workspace_calibration import WorkspaceCalibrationConfig
from control_station.b601_torque_console.workspace_identification import (
    _dispatch_integrals,
    _local_repeat_prediction,
    _repeatability,
    organize_automatic_reports,
    summarize_workspace_identification,
)
from robot_motion.rebot_b601_rs.gravity_identification import (
    fit_paired_gravity_command, pair_motion_observations, solve_gravity_design,
)
from robot_motion.rebot_b601_rs.mit import B601_RS_JOINT_MIT_FRAME
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

pytestmark = [pytest.mark.unit, pytest.mark.contract]
ZERO = (0.0,)*6
PROFILE = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_continuous_identification_v1.json"


def test_continuous_plan_is_opt_in_multispeed_bidirectional_with_reserved_validation():
    values = load_automatic_profile(PROFILE)
    workspace = WorkspaceCalibrationConfig(**values["workspace"])
    cfg = probe._automatic_profile_values(PROFILE)
    assert WorkspaceCalibrationConfig().holding_sampling_plan == "detailed"
    from control_station.b601_torque_console.automatic_calibration import AutomaticCalibrationConfig
    calibration = AutomaticCalibrationConfig(**{k: v for k, v in cfg.items() if k not in {"motion_profile", "cycle", "workspace"}})
    for joint in ("joint2", "joint3"):
        plan = workspace.sweep_plan(condition=condition(), joint_name=joint,
            anchor_qpos_rad=(0, math.radians(5), math.radians(5), 0, 0, 0),
            position_limits_rad=((-2, 2),)*6, calibration=calibration)
        assert len(plan["steps"]) == 14
        assert [s["role"] for s in plan["steps"]] == ["positioning_not_for_fit"]+["training"]*8+["validation"]*4+["return_not_for_fit"]
        j = int(joint[-1])-1
        assert all(math.radians(2.9) < s["target_qpos_rad"][j] < math.radians(7.1) for s in plan["steps"])
        assert {s["max_reference_velocity_rad_s"] for s in plan["steps"]} == {.0075, .015}


def test_continuous_profile_resolves_through_the_real_launcher_before_backend_construction(tmp_path):
    args = owner._parser().parse_args([
        "--allow-motion", "--rate-hz", "100", "--preflight-only",
        "--torque-console-auto-calibration-profile", str(PROFILE),
        "--torque-console-confirmation", "B601_RS_TORQUE_CONSOLE", "--torque-console-joint", "joint3",
        "--torque-console-max-assist-nm", "1", "--torque-console-max-joint-velocity-rad-s", ".195",
        "--torque-console-max-excursion-rad", ".15", "--torque-console-staged-target-offset-limit-rad", ".09",
        "--torque-console-staged-breakaway-torque-nm", "1", "--torque-console-staged-breakaway-rise-slew-nm-s", "1",
        "--torque-console-staged-breakaway-duration-sec", "3", "--torque-console-staged-breakaway-impulse-nm-sec", "1.5",
        "--torque-console-staged-tracking-overlay-cap-nm", "1"])
    parsed = probe.build_parser().parse_args(owner.build_probe_command(args, run_dir=tmp_path)[2:])
    probe._validate_cli(parsed)
    controller = probe._automatic_calibration_controller_from_args(parsed)
    workspace = controller._workspace._coordinated
    assert workspace._continuous_identification
    assert workspace.pose.motion.max_control_gap_sec == .04
    assert workspace.workspace.coordinated_holding.total_assist_cap_nm == 1
    assert workspace._condition is None and not workspace._ready


def test_continuous_controller_completes_both_axes_without_force_bisection(tmp_path):
    r = replay(tmp_path, holding_sampling_plan="identification")
    r.controller.submit("arm", now_sec=r.now)
    markers = set()
    for _ in range(16000):
        scripted_coordinated_tick(r)
        marker = r.audit.get("calibration_trial")
        if marker:
            markers.add(marker["role"])
        core = r.controller._coordinated
        assert core._probe is None and core._search is None
        assert sum(abs(v) for v in r.delta) <= core.workspace.coordinated_holding.total_assist_cap_nm+1e-9
        assert not r.audit["holding_calibration"]["enabled"]
        if r.controller.result_record()["state"] in {"complete", "failed", "cancelled"}:
            break
    result = r.controller.result_record()
    assert result["state"] == "complete", result["reason"]
    assert len(result["motion"]["completed_legs"]) == 28
    assert all("holding_calibration" not in leg for leg in result["motion"]["completed_legs"])
    assert markers >= {"training", "validation", "positioning_not_for_fit", "return_not_for_fit"}
    assert r.controller.cycle_disable_confirmation_ready


def test_continuous_path_still_withdraws_on_human_and_keeps_hard_velocity(tmp_path):
    r = replay(tmp_path, holding_sampling_plan="identification")
    r.controller.submit("arm", now_sec=r.now)
    for _ in range(90):
        scripted_coordinated_tick(r)
    r.controller.set_human_intervention(True)
    for _ in range(300):
        r.tick()
        if r.controller.result_record()["state"] == "cancelled":
            break
    assert r.controller.result_record()["state"] == "cancelled"
    assert r.controller._coordinated._return_revoked == "human_intervention"
    r = replay(tmp_path, holding_sampling_plan="identification")
    r.controller.submit("arm", now_sec=r.now)
    r.tick()
    q = list(r.q)
    q[2] += .01
    r.tick(q)
    assert r.audit["fault"] == "joint_velocity_exceeded"


def paired_observations():
    train, valid = [], []
    for joint in ("joint2", "joint3"):
        a, b = observations(joint=joint)
        train += [replace(o, trial_id=joint+":"+o.trial_id) for o in a]
        valid += [replace(o, trial_id=joint+":"+o.trial_id) for o in b]
    return train, valid


def test_paired_gravity_fit_reuses_scale_bias_solver_but_does_not_claim_physical_gravity():
    train, valid = paired_observations()
    result = fit_paired_gravity_command(train, valid)
    assert result["status"] == "OFFLINE_CANDIDATE", result["reasons"]
    assert result["scale_correction"] == pytest.approx(.03)
    assert result["joint_intercept_nm"] == pytest.approx({"joint2": .02, "joint3": .02})
    assert result["joint_bias_correction_nm"] == pytest.approx({"joint2": .0134, "joint3": .0167})
    profile = result["candidate_profile_fields"]
    assert profile["gravity_scale"] == train[0].condition.gravity_scale
    assert profile["gravity_joint_scales"][1:3] == pytest.approx([.96*1.03, 1.03])
    for j in (0, 3, 4, 5):
        assert profile["gravity_joint_scales"][j] == train[0].condition.gravity_joint_scales[j]
        assert profile["gravity_joint_bias_nm"][j] == train[0].condition.gravity_joint_bias_nm[j]
    assert result["validation_rmse_nm"] < 1e-10
    assert not result["auto_apply"] and not result["identifies_unique_physical_gravity"]
    for pair in result["training_pairs"]:
        assert pair["maximum_pose_mismatch_rad"] == 0


def test_one_sided_mismatched_pose_and_leaked_trials_cannot_be_gravity_labels():
    train, valid = paired_observations()
    assert pair_motion_observations([o for o in train if o.velocity_rad_s > 0]) == []
    shifted = [replace(o, qpos_rad=tuple(v+.02 for v in o.qpos_rad)) if o.velocity_rad_s < 0 else o for o in train]
    assert pair_motion_observations(shifted) == []
    result = fit_paired_gravity_command(train, [train[0], *valid])
    assert result["reasons"] == ["whole_trial_validation_leakage"]
    result = fit_paired_gravity_command(train, [replace(o, command_residual_nm=o.command_residual_nm+.2) for o in valid])
    assert result["status"] == "VALIDATION_FAILED"


def test_shared_gravity_solver_preserves_scale_bias_and_payload_constraints():
    solution, rank, _ = solve_gravity_design([[1], [2]], [-3, -6], model="global-scale", nominal_scale=1)
    assert solution == pytest.approx([-1]) and rank == 1
    with pytest.raises(ValueError):
        solve_gravity_design([[float("nan")]], [0], model="global-scale", nominal_scale=1)


def motion_records():
    rows, sequence, clock = [], 0, 1.0
    context = condition()
    names = [f"joint{i}" for i in range(1, 7)]
    for joint in ("joint2", "joint3"):
        j = int(joint[-1])-1
        for role, repeats in (("training", 2), ("validation", 1)):
            for repeat in range(repeats):
                for magnitude in (.0075, .015):
                    for direction in (-1, 1):
                        trial = f"{joint}:{role}:{repeat}:{magnitude}:{direction}"
                        rows.append({"kind": "test_trial_boundary"})
                        ticks = round(.04/magnitude/.01)
                        velocity = direction*.04/(ticks*.01)
                        for tick in range(ticks+1):
                            sequence += 1
                            clock += .01
                            q = [0, .1, .1, 0, 0, 0]
                            q[j] = .08+tick/ticks*.04 if direction > 0 else .12-tick/ticks*.04
                            gravity = [0, 1+10*q[1], 2+8*q[2], 1, 0, 0]
                            residual = .02+.03*gravity[j]+.12*direction+4*velocity+.025*direction*abs(gravity[j])
                            total = gravity.copy()
                            total[j] += residual
                            rows.append({"kind": "active_dispatch", "recorded_monotonic_sec": clock+.003,
                                "feedback_timestamp_sec": clock, "sequence": sequence,
                                "startup_state": "ACTIVE", "gravity_ramp_fraction": 1, "startup_handoff_fraction": 1,
                                "qpos_rad": q, "gravity_tau_nm": gravity, "conditioned_tau_nm": total,
                                "vendor_kp_nm_per_rad": list(ZERO), "vendor_kd_nm_s_per_rad": list(ZERO),
                                "torque_limited": False, "slew_limited": False, "overlay_limited": False,
                                "host_observation": {"host_receive_freshness_verified": True,
                                    "joints": [{"name": name, "observed_at_sec": clock-.004} for name in names]},
                                "dispatch_receipt": {"sequence": sequence, "source_sequence": sequence,
                                    "accepted": True, "mode": "mit", "frame_id": B601_RS_JOINT_MIT_FRAME,
                                    "requested_joint_names": names, "sent_joint_names": names,
                                    "failed_joint_names": [], "errors": [], "requested_at_sec": clock+.001,
                                    "completed_at_sec": clock+.002},
                                "torque_console": {"phase": "track", "fault": None,
                                    "holding_calibration": {"enabled": False},
                                    "automatic_calibration": {"human_intervention": False, "workspace": {
                                        "current_task": {"kind": "sweep", "joint": joint, "pose": 0}}},
                                    "calibration_trial": {"batch_id": "test", "trial_id": trial,
                                        "role": role, "state": "moving", "reference_speed_rad_s": magnitude}}})
    return rows, context


def test_streaming_previous_dispatch_windows_feed_both_existing_fits_and_survive_later_failure():
    rows, context = motion_records()
    rows.append({"kind": "torque_console_guard", "reason": "test_later_stop"})
    result = summarize_workspace_identification(iter(rows), condition=asdict(context), workspace={"acquisitions": []})
    assert result["admitted_motion_windows"] > 20
    assert result["running_resistance"]["offline_candidates"] == 2
    assert result["gravity_fit"]["status"] == "OFFLINE_CANDIDATE", result["gravity_fit"]["reasons"]
    assert result["gravity_fit"]["scale_correction"] == pytest.approx(.03, abs=.002)
    assert result["later_stop_invalidates_completed_windows"] is False
    assert result["running_resistance"]["retained_partial_leg_windows"] > 0
    assert all(w["input_receipt_sequence_end"] > w["input_receipt_sequence_start"] for w in result["windows"])


@pytest.mark.parametrize("method", ["quasisteady", "closed_loop_average"])
def test_receive_clock_may_require_an_older_input_than_the_previous_snapshot(method):
    rows, context = motion_records()
    rows = rows[:220]
    for row in rows:
        if row["kind"] == "active_dispatch":
            for source in row["host_observation"]["joints"]:
                source["observed_at_sec"] = row["feedback_timestamp_sec"]-.018
    result = summarize_workspace_identification(rows, condition=asdict(context), workspace={"acquisitions": []},
                                               reduction_method=method)
    assert result["admitted_motion_windows"]
    # The first observation is source sequence 4, received before command 3
    # completed: command 2 is its newest available complete input.
    assert result["windows"][0]["input_receipt_sequence_start"] == 1
    assert result["device_sampling_timestamp_verified"] is False


def test_input_integral_uses_dispatch_events_and_never_backdates_future_input():
    def sent(t, value):
        return {"dispatch_receipt": {"completed_at_sec": t},
                "conditioned_tau_nm": [value]*6, "gravity_tau_nm": [1.0]*6}
    commands = [sent(.1, 2), sent(.3, 5), sent(.9, 100)]
    command, gravity = _dispatch_integrals(commands, .2, .5)
    assert command == pytest.approx([.1*2+.2*5]*6)
    assert gravity == pytest.approx([.3]*6)
    with pytest.raises(ValueError, match="missing causal command"):
        _dispatch_integrals(commands, 0, .5)


def test_repeatability_matches_disjoint_trials_without_using_command_for_selection():
    train, _ = observations(joint="joint2")
    a = train[0]
    b = replace(a, trial_id="repeat", command_residual_nm=a.command_residual_nm+.2)
    result = _repeatability([a, replace(a), b])
    assert result["matched_pairs"] == 1
    assert result["unmatched_windows"] == 1
    assert result["rms_pair_difference_nm"] == pytest.approx(.2)
    assert _repeatability([a, replace(b, velocity_rad_s=-b.velocity_rad_s)])["matched_pairs"] == 0
    assert _repeatability([a, replace(b, qpos_rad=(.9,)*6)])["matched_pairs"] == 0
    assert _repeatability([a, replace(b, condition=condition(run_id="other"))])["matched_pairs"] == 0


def test_local_prediction_uses_equal_training_trials_and_retains_uncovered_validation():
    train, _ = observations(joint="joint2")
    a = train[0]
    b = replace(a, trial_id="second-train", command_residual_nm=a.command_residual_nm+.2)
    target = replace(a, trial_id="valid", command_residual_nm=a.command_residual_nm+.1)
    outside = replace(target, qpos_rad=(.9,)*6)
    result = _local_repeat_prediction([a, replace(a), b], [target, outside])
    assert result["predicted_windows"] == 1 and result["unpredicted_windows"] == 1
    assert not result["all_validation_windows_predicted"]
    assert result["equal_trial_rmse_on_predicted_windows_nm"] == pytest.approx(0)
    assert result["status"] == "DIAGNOSTIC_ONLY" and not result["auto_apply"]
    assert _local_repeat_prediction([a], [target])["predicted_windows"] == 0
    with pytest.raises(ValueError, match="whole trial leakage"):
        _local_repeat_prediction([a], [a])


def test_changing_closed_loop_commands_can_be_observed_without_weakening_admission():
    rows, context = motion_records()
    for row in rows:
        if row["kind"] == "active_dispatch":
            j = int(row["torque_console"]["automatic_calibration"]["workspace"]["current_task"]["joint"][-1])-1
            # Changing control input with a zero-mean short periodic component;
            # this is synthetic command evidence, not a physics simulation.
            row["conditioned_tau_nm"][j] += .25*math.sin(row["sequence"]*math.pi/3)
    legacy = summarize_workspace_identification(rows, condition=asdict(context), workspace={})
    result = summarize_workspace_identification(iter(rows), condition=asdict(context), workspace={},
                                               reduction_method="closed_loop_average")
    assert legacy["admitted_motion_windows"] == 0
    assert result["admitted_motion_windows"] > 100
    assert result["running_resistance"]["offline_candidates"] == 2
    assert result["gravity_fit"]["status"] == "OFFLINE_CANDIDATE"
    assert not result["auto_apply"] and not result["device_sampling_timestamp_verified"]
    admitted = [w for w in result["windows"] if w["model_eligible"]]
    assert all(w["command_variation_exceeds_quasisteady_limit"] for w in admitted)
    for joint in ("joint2", "joint3"):
        for role in ("training", "validation"):
            assert {math.copysign(1, w["observation"]["velocity_rad_s"]) for w in admitted
                    if w["joint_name"] == joint and w["role"] == role} == {-1, 1}


def test_closed_loop_cli_preserves_sources_and_exposes_independent_reprocessing(tmp_path):
    from control_station.b601_torque_console.report import main
    rows, context = motion_records()
    trace = tmp_path/"b601_owner.debug.jsonl"
    trace.write_text("".join(json.dumps(r)+"\n" for r in rows))
    source = tmp_path/"automatic_calibration_report.json"
    source.write_text(json.dumps({"schema": "b601.automatic-calibration-report.v1",
        "batch_state": "failed", "acquisition_complete": False,
        "condition": asdict(context), "workspace": {"holding_sampling_plan": "identification"}}))
    original = (trace.read_bytes(), source.read_bytes())
    assert main(["--run-dir", str(tmp_path), "--closed-loop-identification",
                 "--output-root", str(tmp_path/"derived")]) == 0
    report = json.loads(next((tmp_path/"derived").glob("*/report.json")).read_text())
    assert report["source_batch_state"] == "failed"
    assert not report["source_acquisition_complete"]
    assert report["source_trace_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert report["source_report_sha256"] == hashlib.sha256(original[1]).hexdigest()
    assert report["running_resistance"]["offline_candidates"] == 2
    assert (trace.read_bytes(), source.read_bytes()) == original
    source.write_text(json.dumps({"schema": "b601.automatic-calibration-report.v1",
        "batch_state": "moving", "workspace": {"holding_sampling_plan": "identification"}}))
    assert main(["--run-dir", str(tmp_path), "--closed-loop-identification"]) == 2


def test_automatic_report_wires_continuous_fits_and_keeps_later_stop_separate(tmp_path):
    from control_station.b601_torque_console.automatic_report import build_automatic_report

    rows, context = motion_records()
    terminal = {"schema": "b601.automatic-joint-calibration.v1", "batch_id": "test",
        "state": "failed", "reason": "leg_timeout", "onset_trials": [],
        "condition": asdict(context), "outcome": {"severity": "warning", "reason": "leg_timeout"},
        "motion": {"completed_legs": []}, "workspace": {"holding_sampling_plan": "identification",
            "setup_label": "synthetic", "acquisitions": []}}
    trace = tmp_path/"b601_owner.debug.jsonl"
    with trace.open("x") as stream:
        for row in [{**terminal, "kind": "automatic_calibration_started"}, *rows,
                    {**terminal, "kind": "automatic_calibration_result"}]:
            stream.write(json.dumps(row)+"\n")
    report = build_automatic_report(tmp_path)
    assert not report["acquisition_complete"]
    assert report["model_readiness"]["offline_candidates"] == 2
    assert report["gravity_identification"]["status"] == "OFFLINE_CANDIDATE"
    assert report["motion_data_assessment"]["model_eligible_windows"] > 0
    assert not report["auto_apply"]


@pytest.mark.parametrize("method", ["quasisteady", "closed_loop_average"])
@pytest.mark.parametrize("fault", ["holding", "receipt", "future", "kp", "human", "saturation", "startup"])
def test_motion_admission_never_fits_unproven_context(fault, method):
    rows, context = motion_records()
    rows = rows[:200]
    for row in rows:
        if row["kind"] != "active_dispatch":
            continue
        if fault == "holding": row["torque_console"]["holding_calibration"]["enabled"] = True
        elif fault == "receipt": row["dispatch_receipt"]["sent_joint_names"] = ["joint3"]
        elif fault == "future": row["dispatch_receipt"]["completed_at_sec"] = row["feedback_timestamp_sec"]+.02
        elif fault == "kp": row["vendor_kp_nm_per_rad"][2] = .1
        elif fault == "human": row["torque_console"]["automatic_calibration"]["human_intervention"] = True
        elif fault == "saturation": row["overlay_limited"] = True
        elif fault == "startup": row["gravity_ramp_fraction"] = .5
    result = summarize_workspace_identification(rows, condition=asdict(context), workspace={"acquisitions": []},
                                               reduction_method=method)
    assert result["admitted_motion_windows"] == 0
    assert result["gravity_fit"]["status"] == "INSUFFICIENT_DATA"
    assert result["rejected"]


def test_static_dataset_organizer_preserves_censored_and_prior_valid_records_without_relabel(tmp_path):
    source = tmp_path/"automatic_calibration_report.json"
    report = {"schema": "b601.automatic-calibration-report.v1", "condition": asdict(condition()),
        "batch_state": "failed", "holding_calibration": {"points": [{"joint": "joint2", "pose": 0,
            "state": "complete", "trial_id": "hold", "sample_pose_rad": list(ZERO),
            "observations": [{"outcome": "stationary", "command_nm": 0, "final_qpos_rad": list(ZERO)}],
            "transitions": [{"outcome": "censored_plateau_outside_point", "command_nm": .1}]}]}}
    source.write_text(json.dumps(report))
    before = source.read_bytes()
    result = organize_automatic_reports([source])
    assert result["record_count"] == 2
    assert result["runs"][0]["verified_holds"] == 1
    assert result["runs"][0]["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert all(not r["identifies_unique_gravity"] for r in result["runs"][0]["records"])
    assert source.read_bytes() == before
    with pytest.raises(ValueError, match="duplicate"):
        organize_automatic_reports([source, source])


def test_normal_supported_return_is_not_a_timed_control_failure_but_needs_terminal_evidence():
    result = {"state": "complete", "pending_dispatch": None, "outcome": {"action": "complete"},
        "workspace": {"supported_return": {"completed": True, "revoked_reason": None}}}
    assert probe._control_sequence_completed("supported_origin_return_complete", result)
    assert not probe._control_sequence_completed("supported_origin_return_complete")
    assert not probe._control_sequence_completed("operator_interrupt", result)
    for changed in ({"state": "failed"}, {"pending_dispatch": {"receipt_confirmed": False}},
                    {"workspace": {"supported_return": {"completed": False}}}):
        assert not probe._control_sequence_completed("supported_origin_return_complete", {**result, **changed})

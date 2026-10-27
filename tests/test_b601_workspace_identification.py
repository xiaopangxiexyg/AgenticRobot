"""Synthetic command identification and scripted receipts, never real motion."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from test_b601_command_calibration import condition, observations
from test_b601_coordinated_holding import replay, scripted_coordinated_tick

from control_station.b601_torque_console.automatic_calibration import (
    load_automatic_profile,
)
from control_station.b601_torque_console.workspace_calibration import (
    WorkspaceCalibrationConfig,
)
from control_station.b601_torque_console.workspace_identification import (
    _dispatch_integrals,
    _local_repeat_prediction,
    _reference_speed_coverage,
    _repeatability,
    organize_automatic_reports,
    summarize_workspace_identification,
)
from robot_motion.rebot_b601_rs.command_field import (
    LocalCommandField,
    evaluate_local_command_field,
)
from robot_motion.rebot_b601_rs.gravity_identification import (
    fit_paired_gravity_command,
    fit_paired_resistance_command,
    pair_motion_observations,
    solve_gravity_design,
    summarize_pairing_coverage,
)
from robot_motion.rebot_b601_rs.mit import B601_RS_JOINT_MIT_FRAME
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe
from scripts.hardware import rebot_b601_rs_owner as owner

pytestmark = [pytest.mark.unit, pytest.mark.contract]
ZERO = (0.0,)*6
PROFILE = Path(__file__).resolve().parents[1]/"docs/hardware/profiles/b601_j2_j3_continuous_identification_v2.json"


def test_measured_speed_coverage_keeps_pose_direction_damping_and_holdout_separate():
    base = {"model_eligible":True, "pose_index":0, "joint_name":"joint3", "role":"training",
        "reference_speed_rad_s":.0075, "trial_id":"train-a",
        "observation":{"velocity_rad_s":.015, "active_driver_kd_nm_s_per_rad":[0,4.5,4.5,0,0,0]}}
    rows = [copy.deepcopy(base) for _ in range(6)]
    rows[1].update(role="validation", trial_id="validation-a")
    rows[1]["observation"]["velocity_rad_s"] = .03
    rows[2]["pose_index"] = 1
    rows[3]["observation"]["velocity_rad_s"] = -.015
    rows[4]["observation"]["active_driver_kd_nm_s_per_rad"][2] = 3.
    rows[5]["model_eligible"] = False
    original = copy.deepcopy(rows)
    summary = _reference_speed_coverage(rows)
    assert rows == original and len(summary["groups"]) == 5
    validation = next(g for g in summary["groups"] if g["role"] == "validation")
    assert validation["median_measured_to_reference_ratio"] == 4
    assert validation["window_count"] == validation["independent_trial_count"] == 1
    training = next(g for g in summary["groups"] if g["role"] == "training" and g["pose_index"] == 0
        and g["measured_direction"] == 1 and g["active_driver_kd_nm_s_per_rad"][2] == 4.5)
    assert training["median_measured_to_reference_ratio"] == 2
    assert summary["unavailable_windows"] == 0 and not summary["auto_apply"]
    rows[0]["observation"]["velocity_rad_s"] = 0
    assert _reference_speed_coverage(rows)["unavailable_windows"] == 1


def test_continuous_plan_is_opt_in_multispeed_bidirectional_with_reserved_validation():
    values = load_automatic_profile(PROFILE)
    workspace = WorkspaceCalibrationConfig(**values["workspace"])
    cfg = probe._automatic_profile_values(PROFILE)
    assert WorkspaceCalibrationConfig().holding_sampling_plan == "detailed"
    from control_station.b601_torque_console.automatic_calibration import (
        AutomaticCalibrationConfig,
    )
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


def test_four_centres_joint3_has_three_bidirectional_speeds_and_independent_holdouts(tmp_path):
    from test_b601_cruise_velocity import FOUR_CENTRES, controller_from_profile

    controller = controller_from_profile(tmp_path, profile=FOUR_CENTRES)
    core = controller._workspace._coordinated
    assert len(core.workspace.pose_centres()) == 4
    assert core.workspace.coordinated_holding.arrival_displacement_gains_nm_per_rad == (20, 40)
    for joint, expected, speeds in [('joint2', 14, {.0075, .015}), ('joint3', 20, {.0075, .01125, .015})]:
        plan = core.workspace.sweep_plan(condition=condition(), joint_name=joint,
            anchor_qpos_rad=(0,.06,.06,0,0,0), position_limits_rad=((-2,2),)*6, calibration=core.config)
        assert len(plan['steps']) == expected
        training, validation = set(), set()
        for role, ids, repeats in [('training', training, 2), ('validation', validation, 1)]:
            steps = [s for s in plan['steps'] if s['role'] == role]
            for speed in speeds:
                values = [s['target_qpos_rad'][int(joint[-1])-1] for s in steps if s['max_reference_velocity_rad_s'] == speed]
                assert len(values) == repeats*2 and len(set(values)) == 2
            ids.update(s['trial_id'] for s in steps)
        assert not training & validation
        assert max(s['max_reference_velocity_rad_s'] for s in plan['steps']) == .015
    with pytest.raises(ValueError, match='existing sweep ceiling'):
        workspace = replace(core.workspace, identification_joint3_speeds_rad_s=(.0075,.02))
        type(core)(core.settings,core.config,workspace)


@pytest.mark.parametrize('speeds', [True, [], [.0075], [.015,.0075], [.0075,.0075], [.0075,float('nan')], [.0075,True]])
def test_bad_joint3_speed_lists_fail_before_controller_construction(speeds):
    from test_b601_cruise_velocity import FOUR_CENTRES

    workspace = WorkspaceCalibrationConfig(**load_automatic_profile(FOUR_CENTRES)['workspace'])
    with pytest.raises(ValueError):
        replace(workspace, identification_joint3_speeds_rad_s=speeds)


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
    assert result["validation_rmse_nm"] < result["zero_correction_validation_rmse_nm"]
    assert not result["auto_apply"] and not result["identifies_unique_physical_gravity"]
    for pair in result["training_pairs"]:
        assert pair["maximum_pose_mismatch_rad"] == 0


def test_local_even_bias_freezes_scale_and_never_exports_a_gravity_profile():
    train, valid = paired_observations()

    def local(o):
        j = int(o.joint_name[-1])-1
        return replace(o, qpos_rad=(0, .06, .06, 0, 0, 0),
            command_residual_nm=(-.2 if j == 1 else .1)+math.copysign(.3, o.velocity_rad_s))

    train, valid = [local(o) for o in train], [local(o) for o in valid]
    result = fit_paired_gravity_command(train, valid, model="local-joint-bias")
    assert result["status"] == "OFFLINE_CANDIDATE", result["reasons"]
    assert result["scale_correction"] == 0 and result["gravity_scales_frozen"]
    assert result["local_command_bias_nm"] == pytest.approx({"joint2": -.2, "joint3": .1})
    assert result["validation_rmse_nm"] < 1e-12
    assert result["design_rank"] == 2
    assert "candidate_profile_fields" not in result and not result["profile_export_allowed"]
    assert not result["auto_apply"] and not result["identifies_unique_physical_gravity"]
    json.dumps(result, allow_nan=False)
    bad = [replace(o, command_residual_nm=o.command_residual_nm+.2) for o in valid]
    assert fit_paired_gravity_command(train, bad, model="local-joint-bias")["status"] == "VALIDATION_FAILED"
    outside = [replace(o, qpos_rad=(0, .15, .15, 0, 0, 0)) for o in valid]
    assert fit_paired_gravity_command(train, outside, model="local-joint-bias")["reasons"] == ["validation_outside_training_pose_coverage"]
    assert fit_paired_gravity_command(train, [train[0], *valid], model="local-joint-bias")["reasons"] == ["whole_trial_validation_leakage"]


def test_scale_information_detects_large_baseline_without_independent_scale_excitation():
    train, valid = paired_observations()
    flat = [replace(o, gravity_command_nm=tuple(6.0 if j == 2 else 1.0 for j in range(6))) for o in train]
    result = fit_paired_gravity_command(flat, valid)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["scale_information"]["fraction_after_joint_intercepts"] < 1e-12
    assert result["reasons"] == ["gravity_features_not_identifiable_expand_pose_information"]
    with pytest.raises(ValueError, match="unknown paired"):
        fit_paired_gravity_command(train, valid, model="automatic_fallback")


def test_one_sided_mismatched_pose_and_leaked_trials_cannot_be_gravity_labels():
    train, valid = paired_observations()
    assert pair_motion_observations([o for o in train if o.velocity_rad_s > 0]) == []
    shifted = [replace(o, qpos_rad=tuple(v+.02 for v in o.qpos_rad)) if o.velocity_rad_s < 0 else o for o in train]
    assert pair_motion_observations(shifted) == []
    result = fit_paired_gravity_command(train, [train[0], *valid])
    assert result["reasons"] == ["whole_trial_validation_leakage"]
    result = fit_paired_gravity_command(train, [replace(o, command_residual_nm=o.command_residual_nm+.2) for o in valid])
    assert result["status"] == "VALIDATION_FAILED"
    outside = [replace(o, qpos_rad=tuple(q+.02 for q in o.qpos_rad)) for o in valid]
    result = fit_paired_gravity_command(train, outside)
    assert result["reasons"] == ["validation_outside_training_pose_coverage"]
    assert result["covered_validation_pairs"] == 0
    assert result["outside_coverage_predictions_are_extrapolation"]
    assert "validation_rmse_nm" in result and not result["auto_apply"]


def test_paired_odd_resistance_uses_real_pairs_and_independent_speed_load_validation():
    train, valid = paired_observations()
    result = fit_paired_resistance_command(train, valid)
    assert result["status"] == "OFFLINE_CANDIDATE"
    assert not result["auto_apply"] and not result["identifies_physical_friction"]
    for fit in result["fits"]:
        assert fit["status"] == "OFFLINE_CANDIDATE", fit["reasons"]
        assert fit["reference_speed_rad_s"] == pytest.approx(.009)
        assert fit["reference_abs_gravity_command_nm"] == pytest.approx(4)
        assert fit["coefficients"] == pytest.approx([.256, 4, .025])
        assert fit["validation_rmse_nm"] < 1e-10
        assert fit["validation_rmse_nm"] < fit["zero_correction_validation_rmse_nm"]
        assert fit["covered_validation_pairs"] == fit["validation_pairs"]
    bad = [replace(o, command_residual_nm=o.command_residual_nm+math.copysign(.2, o.velocity_rad_s)) for o in valid]
    assert all(f["status"] == "VALIDATION_FAILED" for f in fit_paired_resistance_command(train, bad)["fits"])
    assert fit_paired_resistance_command(train, [train[0], *valid])["reasons"] == ["whole_trial_validation_leakage"]
    outside_speed = [replace(o, velocity_rad_s=3*o.velocity_rad_s) for o in valid]
    for fit in fit_paired_resistance_command(train, outside_speed)["fits"]:
        assert fit["status"] == "INSUFFICIENT_DATA"
        assert fit["covered_validation_pairs"] == 0
        assert fit["validation_rmse_nm"] > 0
    changed = [replace(o, condition=condition(run_id="other")) for o in valid]
    assert fit_paired_resistance_command(train, changed)["reasons"] == ["mixed_gravity_source_or_active_damping"]
    one_way = [o for o in train if o.velocity_rad_s > 0]
    assert all(f["status"] == "INSUFFICIENT_DATA" for f in fit_paired_resistance_command(one_way, valid)["fits"])


def test_pairing_coverage_distinguishes_pose_from_actual_speed_without_using_commands():
    train, _ = observations(joint="joint3")
    a = train[0]
    b = replace(a, trial_id="opposite", velocity_rad_s=-a.velocity_rad_s*2)
    group = summarize_pairing_coverage([a, b])["groups"][0]
    assert group["pose_matched_combinations"] == 1
    assert group["speed_matched_combinations"] == 0
    assert group["matched_disjoint_pairs"] == 0 and group["unmatched_windows"] == 2
    assert group["closest_pose_examples"][0]["speed_magnitude_ratio"] == 2
    b = replace(b, velocity_rad_s=-a.velocity_rad_s, qpos_rad=(.9,)*6)
    group = summarize_pairing_coverage([a, b])["groups"][0]
    assert group["pose_matched_combinations"] == 0 and group["speed_matched_combinations"] == 1


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
    assert result["cruise_execution_diagnostic"]["trials"] == []
    assert result["cruise_execution_diagnostic"]["unavailable_samples"] > 0
    assert result["running_resistance"]["retained_partial_leg_windows"] > 0
    assert all(w["input_receipt_sequence_end"] > w["input_receipt_sequence_start"] for w in result["windows"])


def test_local_even_reports_are_per_pose_and_damping_and_cruise_diagnostics_are_compact():
    rows, context = motion_records()
    count = 0
    for row in rows:
        if row.get("kind") != "active_dispatch":
            continue
        count += 1
        row["torque_console"]["coordinated_holding"] = {
            "combined_command_nm": [0, -.25, -.5, 0, 0, 0],
            "combined_cap_shortfall_nm": [0, .01, .02, 0, 0, 0], "saturated": True,
            "cruise_velocity_control": {"active": True, "integral_growth_blocked": False,
                "reason": "resolved_cruise_velocity_feedback"}}
    report = summarize_workspace_identification(rows, condition=asdict(context), workspace={"acquisitions": []},
        reduction_method="closed_loop_average", position_bin_width_rad=.0045)
    local = report["local_even_command_fit"]
    assert local["gravity_scales_frozen"] and not local["profile_export_allowed"]
    assert len(local["fits"]) == 1
    assert local["fits"][0]["pose_index"] == 0
    assert local["fits"][0]["analysis"]["model"] == "local-joint-bias"
    trials = report["cruise_execution_diagnostic"]["trials"]
    assert report["cruise_execution_diagnostic"]["unavailable_samples"] == 0
    assert len(trials) == 24
    assert sum(t["dispatch_samples"] for t in trials) == count
    for trial in trials:
        assert trial["cruise_active_samples"] == trial["combined_cap_samples"] == trial["dispatch_samples"]
        assert trial["integral_growth_blocked_samples"] == 0
        assert trial["cruise_telemetry_samples"] == trial["dispatch_samples"]
        assert trial["mean_abs_support_command_nm"] == pytest.approx(.5 if trial["joint_name"] == "joint2" else .25)
        assert trial["mean_abs_selected_cap_shortfall_nm"] == pytest.approx(.01 if trial["joint_name"] == "joint2" else .02)
        assert "sum_abs_selected_command_nm" not in trial


@pytest.mark.parametrize("endpoint_mode", ["native_brake", "completed_reference", "arrival_feedback"])
def test_native_arrival_frames_cannot_enter_motion_fits_through_outer_track_label(endpoint_mode):
    rows, context = motion_records()
    ends = {}
    for row in rows:
        if row.get("kind") == "active_dispatch":
            marker = row["torque_console"]["calibration_trial"]
            ends[marker["trial_id"]] = row["feedback_timestamp_sec"]
    cutoffs = {}
    for row in rows:
        if row.get("kind") != "active_dispatch":
            continue
        console = row["torque_console"]
        marker = console["calibration_trial"]
        trial = marker["trial_id"]
        cutoffs.setdefault(trial, (row["feedback_timestamp_sec"]+ends[trial])/2)
        selected = int(console["automatic_calibration"]["workspace"]["current_task"]["joint"][-1])-2
        axes = [{"phase": "brake", "reason": "target_reached_braking"} for _ in range(2)]
        moving = row["feedback_timestamp_sec"] < cutoffs[trial]
        if moving or endpoint_mode != "native_brake":
            axes[selected] = {"phase": "track", "reason": "bounded_reference_tracking"}
        if not moving and endpoint_mode == "completed_reference":
            axes[selected].update(target_position_rad=.1, q_reference_rad=.1, qvel_reference_rad_s=0.)
        console["coordinated_holding"] = {"axes": axes}
        if not moving and endpoint_mode == "arrival_feedback":
            console["coordinated_holding"]["cruise_velocity_control"] = {"arrival": True}
    result = summarize_workspace_identification(rows, condition=asdict(context), workspace={"acquisitions": []})
    assert result["admitted_motion_windows"] > 0
    assert result["rejected"]["endpoint_servo_not_cruise_identification"] > 0
    assert all(w["end_sec"] < cutoffs[w["trial_id"]] for w in result["windows"])


def test_position_cells_reduce_complete_measured_traversals_with_original_receipt_admission():
    rows, context = motion_records()
    result = summarize_workspace_identification(rows, condition=asdict(context), workspace={},
        reduction_method="closed_loop_average", position_bin_width_rad=.0045)
    admitted = [w for w in result["windows"] if w["model_eligible"]]
    assert len(admitted) > 100
    assert not result["settings"]["minimum_duration_gate_applied"]
    assert all(w["complete_position_traversal"] for w in admitted)
    for w in admitted:
        j = int(w["joint_name"][-1])-1
        lo, hi = w["position_bin_bounds_rad"]
        assert lo-1e-12 <= w["qpos_min_rad"][j] <= w["qpos_max_rad"][j] < hi+1e-12
        assert w["input_receipt_sequence_start"] < w["input_receipt_sequence_end"]
    assert result["gravity_fit"]["status"] == "OFFLINE_CANDIDATE", result["gravity_fit"]["reasons"]
    assert result["paired_resistance_fit"]["status"] == "OFFLINE_CANDIDATE"
    assert all(g["matched_disjoint_pairs"] > 0 for g in result["pairing_coverage"]["validation"]["groups"])


def test_position_tail_without_an_exit_never_becomes_a_complete_pairing_window():
    rows, context = motion_records()
    short = []
    for row in rows:
        short.append(row)
        if row.get("kind") == "active_dispatch" and row["qpos_rad"][1] <= .114:
            break
    result = summarize_workspace_identification(short, condition=asdict(context), workspace={},
        reduction_method="closed_loop_average", position_bin_width_rad=.0045)
    assert result["windows"]
    assert result["admitted_motion_windows"] == 0
    assert all(not w["complete_position_traversal"] for w in result["windows"])
    assert result["rejected"]["partial_position_bin_not_for_pairing"] > 0


def test_entering_and_leaving_the_same_cell_edge_is_not_a_full_position_traversal():
    rows, context = motion_records()
    active = [r for r in rows if r["kind"] == "active_dispatch"][:220]
    for i, row in enumerate(active):
        row["qpos_rad"][1] = .118-.00004*i if i <= 100 else .114+.00004*min(i-100, 100)
    result = summarize_workspace_identification(active, condition=asdict(context), workspace={},
        reduction_method="closed_loop_average", position_bin_width_rad=.0045)
    assert result["windows"]
    assert all(not w["complete_position_traversal"] for w in result["windows"])
    assert result["admitted_motion_windows"] == 0


@pytest.mark.parametrize("width", [0, -.01, .002, float("nan"), float("inf"), True])
def test_position_window_configuration_rejects_unresolved_or_invalid_bins(width):
    with pytest.raises(ValueError, match="position bins require"):
        summarize_workspace_identification([], condition=asdict(condition()), workspace={},
            reduction_method="closed_loop_average", position_bin_width_rad=width)


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
    assert result["validation_coverage_fraction"] == .5
    assert result["covered_subset_error_passed"]
    assert not result["physical_compensation_ready"]
    assert _local_repeat_prediction([a], [target])["predicted_windows"] == 0
    with pytest.raises(ValueError, match="whole trial leakage"):
        _local_repeat_prediction([a], [a])


def field_observations():
    train, _ = observations(joint="joint2")
    base = train[0]
    training, validation = [], []
    for position in (.02, .04):
        for speed in (.01, .02):
            for direction in (-1, 1):
                q = (0, position, .03, 0, 0, 0)
                value = .2+2*position+direction*(.12+4*speed)
                for repeat in range(3):
                    sample = replace(base, trial_id=f"{speed}:{direction}:{repeat}", qpos_rad=q,
                        velocity_rad_s=direction*speed, command_residual_nm=value)
                    (training if repeat < 2 else validation).append(sample)
    return training, validation


def test_native_window_witness_and_runtime_summary_do_not_create_extra_fit_admission():
    rows, context = motion_records()
    old = summarize_workspace_identification(rows, condition=asdict(context), workspace={}, reduction_method="closed_loop_average")
    assert not any(w["native_phase_verified"] for w in old["windows"])
    for row in rows:
        if row.get("kind") == "active_dispatch":
            row["torque_console"]["coordinated_holding"] = {
                "axes": [{"phase": "track", "reason": "bounded_reference_tracking"}]*2,
                "command_field": {"mode": "shadow", "predicted": True, "reason": "matched", "requested_increment_nm": 0.0}}
    result = summarize_workspace_identification(rows, condition=asdict(context), workspace={"setup_label": "test", "pose_centres_rad": [[.1, .1]]}, reduction_method="closed_loop_average")
    assert result["admitted_motion_windows"] == old["admitted_motion_windows"]
    assert all(w["native_phase_verified"] for w in result["windows"])
    assert result["command_field_runtime"]["predicted_samples"] > 0
    assert result["command_field_runtime"]["requested_increment_samples"] == 0
    assert result["command_field_context"]["setup_label"] == "test"
    active = next(r for r in rows if r.get("kind") == "active_dispatch")
    active["torque_console"]["coordinated_holding"]["command_field"]["requested_increment_nm"] = .1
    with pytest.raises(ValueError, match="shadow.*increment"):
        summarize_workspace_identification(rows, condition=asdict(context), workspace={}, reduction_method="closed_loop_average")


def test_command_field_predicts_pose_speed_and_direction_and_validates_components():
    training, validation = field_observations()
    field = LocalCommandField(tuple(training))
    target = validation[0]
    query = {"condition": target.condition, "qpos_rad": target.qpos_rad,
        "velocity_rad_s": target.velocity_rad_s,
        "active_driver_kd_nm_s_per_rad": target.active_driver_kd_nm_s_per_rad}
    prediction = field.predict(**query)
    assert prediction["predicted_command_residual_nm"] == pytest.approx(target.command_residual_nm)
    components = field.components(**query)
    assert components["even_command_nm"] == pytest.approx(.24)
    assert components["odd_command_nm"] == pytest.approx(.16)
    assert not components["identifies_physical_gravity_or_friction"]
    assert not components["usable_as_static_support"]
    result = evaluate_local_command_field(training, validation)
    assert result["status"] == "OFFLINE_CANDIDATE", result["validation_checks"]
    assert result["directional_validation"]["equal_trial_rmse_on_predicted_windows_nm"] < 1e-12
    assert result["directional_validation"]["command_prediction_error_reduction_fraction"] == pytest.approx(1)
    paired = result["paired_component_validation"]
    assert paired["status"] == "OFFLINE_CANDIDATE"
    assert paired["predicted_pairs"] == paired["validation_pairs"] == 4
    assert paired["equal_trial_even_rmse_on_predicted_pairs_nm"] < 1e-12
    assert paired["equal_trial_odd_rmse_on_predicted_pairs_nm"] < 1e-12
    assert not result["live_compensation_enabled"] and not result["hardware_approved"]
    # Held-out labels change error, never training neighbours or predictions.
    changed = evaluate_local_command_field(training, [replace(o, command_residual_nm=o.command_residual_nm+.2)
                                                      for o in validation])
    assert not changed["validation_checks"]["prediction_error_passed"]
    assert changed["directional_validation"]["predictions"][0]["predicted_command_residual_nm"] == prediction["predicted_command_residual_nm"]
    assert changed["paired_component_validation"]["equal_trial_even_rmse_on_predicted_pairs_nm"] == pytest.approx(.2)
    assert changed["paired_component_validation"]["equal_trial_odd_rmse_on_predicted_pairs_nm"] < 1e-12
    assert not changed["paired_component_validation"]["validation_checks"]["even_prediction_error_passed"]
    assert not changed["coverage_gaps"]


def test_command_field_abstains_at_rest_and_outside_actual_six_axis_speed_or_repeat_support():
    training, validation = field_observations()
    target = validation[0]
    field = LocalCommandField(tuple(training))
    query = {"condition": target.condition, "qpos_rad": target.qpos_rad,
        "velocity_rad_s": target.velocity_rad_s,
        "active_driver_kd_nm_s_per_rad": target.active_driver_kd_nm_s_per_rad}
    for change in ({"velocity_rad_s": 0}, {"velocity_rad_s": .1}, {"qpos_rad": (0, .02, .04, 0, 0, 0)}):
        result = field.predict(**{**query, **change})
        assert not result["predicted"] and result["predicted_command_residual_nm"] is None
    one_trial = [o for o in training if o.trial_id == target.trial_id.replace(":2", ":0")]
    assert not LocalCommandField(tuple(one_trial)).predict(**query)["predicted"]
    negative_only = LocalCommandField(tuple(o for o in training if o.velocity_rad_s < 0))
    assert negative_only.predict(**query)["predicted"]
    assert not negative_only.components(**query)["available"]
    outside = replace(target, qpos_rad=(0, .02, .04, 0, 0, 0))
    result = evaluate_local_command_field(training, [*validation, outside])
    assert result["status"] == "DIAGNOSTIC_ONLY"
    direct = result["directional_validation"]
    assert direct["predicted_windows"] == 8 and direct["unpredicted_windows"] == 1
    assert direct["validation_coverage_fraction"] == pytest.approx(8/9)
    assert direct["covered_subset_error_passed"]
    assert result["coverage_gaps"][0]["reason"] == "full_pose_outside_training_support"
    assert result["coverage_gaps"][0]["validation_windows"] == 1
    assert result["coverage_gaps"][0]["actual_abs_speed_range_rad_s"] == [.01, .01]


def test_command_field_enforces_source_damping_joint_and_trial_boundaries():
    training, validation = field_observations()
    field = LocalCommandField(tuple(training))
    target = validation[0]
    query = {"condition": target.condition, "qpos_rad": target.qpos_rad,
        "velocity_rad_s": target.velocity_rad_s,
        "active_driver_kd_nm_s_per_rad": target.active_driver_kd_nm_s_per_rad}
    for change in ({"condition": replace(target.condition, run_id="other-run")},
                   {"active_driver_kd_nm_s_per_rad": (1,)*6}):
        with pytest.raises(ValueError, match="mismatch"):
            field.predict(**{**query, **change})
    with pytest.raises(ValueError, match="whole trial leakage"):
        evaluate_local_command_field(training, [training[0]])
    with pytest.raises(ValueError, match="joint mismatch"):
        evaluate_local_command_field(training, [replace(target, joint_name="joint3")])
    with pytest.raises(ValueError, match="one joint"):
        LocalCommandField((*training, replace(training[0], joint_name="joint3")))
    with pytest.raises(TypeError, match="admitted observations"):
        evaluate_local_command_field([{}], validation)
    with pytest.raises(ValueError, match="finite"):
        field.predict(**{**query, "velocity_rad_s": float("nan")})
    # Remote poses in the bank never contribute to a local query.
    larger = LocalCommandField((*training, replace(training[0], qpos_rad=(1,)*6)))
    assert larger.predict(**query) == field.predict(**query)
    # Duplicate copies and long trials cannot acquire extra prediction weight.
    repeated = LocalCommandField((*training, *([training[0]]*100)))
    assert repeated.predict(**query) == field.predict(**query)


def test_validation_reports_both_coverage_and_error_failure_without_changing_admission():
    train, valid = paired_observations()
    outside = [replace(o, qpos_rad=tuple(q+.2 for q in o.qpos_rad),
        command_residual_nm=o.command_residual_nm+.2) for o in valid]
    gravity = fit_paired_gravity_command(train, outside)
    assert gravity["status"] == "INSUFFICIENT_DATA"
    assert gravity["validation_checks"] == {"coverage_passed": False, "prediction_error_passed": False}
    assert gravity["reasons"] == ["validation_outside_training_pose_coverage", "held_out_paired_command_prediction_error"]
    outside = [replace(o, qpos_rad=tuple(q+.2 for q in o.qpos_rad),
        command_residual_nm=o.command_residual_nm+math.copysign(.2, o.velocity_rad_s)) for o in valid]
    for fit in fit_paired_resistance_command(train, outside)["fits"]:
        assert fit["status"] == "INSUFFICIENT_DATA"
        assert fit["validation_checks"] == {"coverage_passed": False, "prediction_error_passed": False}
        assert fit["reasons"] == ["validation_outside_training_pose_load_or_speed_coverage", "held_out_odd_command_prediction_error"]


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


@pytest.mark.parametrize("width", [None, .0045])
def test_closed_loop_cli_preserves_sources_and_exposes_independent_reprocessing(tmp_path, width):
    from control_station.b601_torque_console.report import main
    rows, context = motion_records()
    trace = tmp_path/"b601_owner.debug.jsonl"
    trace.write_text("".join(json.dumps(r)+"\n" for r in rows))
    source = tmp_path/"automatic_calibration_report.json"
    source.write_text(json.dumps({"schema": "b601.automatic-calibration-report.v1",
        "batch_state": "failed", "acquisition_complete": False,
        "condition": asdict(context), "workspace": {"holding_sampling_plan": "identification"}}))
    original = (trace.read_bytes(), source.read_bytes())
    command = ["--run-dir", str(tmp_path), "--closed-loop-identification",
               "--output-root", str(tmp_path/"derived")]
    if width is not None:
        command += ["--position-bin-width-rad", str(width)]
    assert main(command) == 0
    report = json.loads(next((tmp_path/"derived").glob("*/report.json")).read_text())
    assert report["source_batch_state"] == "failed"
    assert not report["source_acquisition_complete"]
    assert report["source_trace_sha256"] == hashlib.sha256(original[0]).hexdigest()
    assert report["source_report_sha256"] == hashlib.sha256(original[1]).hexdigest()
    assert "command_field.py" in report["processor_sources_sha256"]
    assert report["running_resistance"]["offline_candidates"] == 2
    assert (trace.read_bytes(), source.read_bytes()) == original
    assert report["settings"]["position_bin_width_rad"] == width
    assert main(["--run-dir", str(tmp_path), "--position-bin-width-rad", ".0045"]) == 2
    source.write_text(json.dumps({"schema": "b601.automatic-calibration-report.v1",
        "batch_state": "moving", "workspace": {"holding_sampling_plan": "identification"}}))
    assert main(["--run-dir", str(tmp_path), "--closed-loop-identification"]) == 2


@pytest.mark.parametrize("width", [None, .0045])
def test_automatic_report_wires_continuous_fits_and_keeps_later_stop_separate(tmp_path, width):
    from control_station.b601_torque_console.automatic_report import (
        build_automatic_report,
    )

    rows, context = motion_records()
    terminal = {"schema": "b601.automatic-joint-calibration.v1", "batch_id": "test",
        "state": "failed", "reason": "leg_timeout", "onset_trials": [],
        "condition": asdict(context), "outcome": {"severity": "warning", "reason": "leg_timeout"},
        "motion": {"completed_legs": []}, "workspace": {"holding_sampling_plan": "identification",
            "setup_label": "synthetic", "acquisitions": [],
            "settings": {"identification_position_bin_width_rad": width}}}
    trace = tmp_path/"b601_owner.debug.jsonl"
    with trace.open("x") as stream:
        for row in [{**terminal, "kind": "automatic_calibration_started"}, *rows,
                    {**terminal, "kind": "automatic_calibration_result"}]:
            stream.write(json.dumps(row)+"\n")
    report = build_automatic_report(tmp_path)
    assert not report["acquisition_complete"]
    assert report["command_identification"]["settings"]["position_bin_width_rad"] == width
    if width is None:
        assert report["model_readiness"]["offline_candidates"] == 2
        assert report["gravity_identification"]["status"] == "OFFLINE_CANDIDATE"
        assert report["motion_data_assessment"]["model_eligible_windows"] > 0
    else:
        assert report["command_identification"]["reduction_method"] == "closed_loop_average"
        assert report["paired_resistance_identification"] == report["command_identification"]["paired_resistance_fit"]
    assert report["empirical_command_field"] == report["command_identification"]["empirical_command_field"]
    assert not report["empirical_command_field"]["live_compensation_enabled"]
    assert not report["auto_apply"]


@pytest.mark.parametrize("method", ["quasisteady", "closed_loop_average"])
@pytest.mark.parametrize("width", [None, .0045])
@pytest.mark.parametrize("fault", ["holding", "receipt", "future", "kp", "human", "saturation", "startup"])
def test_motion_admission_never_fits_unproven_context(fault, method, width):
    if width is not None and method == "quasisteady":
        with pytest.raises(ValueError, match="position bins require"):
            summarize_workspace_identification([], condition=asdict(condition()), workspace={},
                reduction_method=method, position_bin_width_rad=width)
        return
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
                                               reduction_method=method, position_bin_width_rad=width)
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

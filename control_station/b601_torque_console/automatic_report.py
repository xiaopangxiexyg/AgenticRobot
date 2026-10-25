"""Offline aggregation of the explicit automatic acquisition, never compensation."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from itertools import pairwise
from pathlib import Path

from robot_motion.rebot_b601_rs.calibration import summarize_automatic_acquisition

from .calibration_report import summarize_calibration
from .calibration_session import calibration_outcome, calibration_receipt_rejection
from .report import _jsonl, _resolve_jsonl_path
from .workspace_identification import summarize_workspace_identification


def _workspace_motion_windows(trace: Path, workspace: dict) -> dict:
    """Retain changing-input windows for later gravity-first identification.

    These are command/encoder observations, not a quasistatic friction fit.
    Keep the actual input variation and line error instead of pretending that
    a closed-loop input was constant or discarding its entire motion segment.
    """
    completed = {leg["trial_id"] for item in workspace["acquisitions"]
                 for leg in item["result"]["completed_legs"]}
    windows, buffer = [], []
    identity = None

    def flush():
        if len(buffer) < 3:
            buffer.clear()
            return
        times = [row["feedback_timestamp_sec"] for row in buffer]
        span = times[-1] - times[0]
        if span < .08:
            buffer.clear()
            return
        first, last = buffer[0], buffer[-1]
        task = first["torque_console"]["automatic_calibration"]["workspace"]["current_task"]
        marker = first["torque_console"]["calibration_trial"]
        j = int(task["joint"][-1]) - 1
        positions = [row["qpos_rad"][j] for row in buffer]
        time_mean, position_mean = statistics.mean(times), statistics.mean(positions)
        denominator = sum((t - time_mean) ** 2 for t in times)
        speed = sum((t - time_mean) * (q - position_mean) for t, q in zip(times, positions)) / denominator
        residuals = [row["conditioned_tau_nm"][j] - row["gravity_tau_nm"][j] for row in buffer]
        windows.append({"pose_index": task["pose"], "joint_name": task["joint"],
            "trial_id": marker["trial_id"], "role": marker["role"],
            "source_leg_completed": marker["trial_id"] in completed,
            "start_sec": times[0], "end_sec": times[-1], "sample_count": len(buffer),
            "qpos_start_rad": first["qpos_rad"], "qpos_end_rad": last["qpos_rad"],
            "velocity_rad_s": speed, "displacement_rad": positions[-1] - positions[0],
            "maximum_position_line_error_rad": max(abs(q - position_mean - speed * (t - time_mean)) for t, q in zip(times, positions)),
            "gravity_command_median_nm": statistics.median(row["gravity_tau_nm"][j] for row in buffer),
            "command_residual_median_nm": statistics.median(residuals),
            "command_residual_span_nm": max(residuals) - min(residuals),
            "driver_kd_start": first["vendor_kd_nm_s_per_rad"],
            "driver_kd_end": last["vendor_kd_nm_s_per_rad"],
            "command_limited": any(row.get("torque_limited") or row.get("slew_limited") or row.get("overlay_limited") for row in buffer),
            "maximum_feedback_gap_sec": max(b - a for a, b in pairwise(times)),
            "human_intervention": False, "measured_output_torque_nm": None})
        buffer.clear()

    for row in _jsonl(trace):
        if row.get("kind") != "active_dispatch":
            continue
        console = row.get("torque_console") or {}
        batch = console.get("automatic_calibration") or {}
        task = (batch.get("workspace") or {}).get("current_task") or {}
        marker = console.get("calibration_trial") or {}
        receipt = row.get("dispatch_receipt") or {}
        valid = (task.get("kind") == "sweep" and task.get("joint") in {"joint2", "joint3"}
            and batch.get("human_intervention") is False and not console.get("fault")
            and console.get("phase") == "track" and marker.get("state") == "moving"
            and calibration_receipt_rejection(receipt, row.get("sequence"), None) is None
            and row["feedback_timestamp_sec"] <= receipt["requested_at_sec"] <= receipt["completed_at_sec"] <= row["recorded_monotonic_sec"]
            and receipt["completed_at_sec"] - row["feedback_timestamp_sec"] <= .04
            and (row.get("host_observation") or {}).get("host_receive_freshness_verified") is True)
        key = (task.get("pose"), task.get("joint"), marker.get("trial_id")) if valid else None
        timestamp = row.get("feedback_timestamp_sec")
        if not valid or key != identity or buffer and not 0 < timestamp - buffer[-1]["feedback_timestamp_sec"] <= .04:
            flush()
        identity = key
        if valid:
            buffer.append(row)
            if timestamp - buffer[0]["feedback_timestamp_sec"] >= .4:
                flush()
    flush()
    return {"status": "COMMAND_MOTION_WINDOWS_AVAILABLE" if windows else "NO_TRACKING_WINDOWS",
        "windows": windows, "window_count": len(windows), "auto_apply": False,
        "semantics": "changing-input windows retained for paired gravity/resistance analysis; no gravity/friction separation or force calibration is claimed"}


def build_automatic_report(run_dir: Path) -> dict:
    trace = _resolve_jsonl_path(run_dir / "b601_owner.debug.jsonl")
    annotations = run_dir / "b601_owner.transparency.jsonl"
    result = None
    started = None
    trials = []
    dispatch_rejection = None
    first_guard = None
    cleanup_boundary = None
    stages, observations, last_workspace = [], [], None
    for row in _jsonl(trace):
        if row.get("kind") == "automatic_calibration_started":
            if started is not None:
                raise ValueError("multiple automatic batches in one owner run")
            started = row
        elif row.get("kind") == "automatic_calibration_onset_trial":
            trials.append(row["trial"])
        elif row.get("kind") == "automatic_calibration_dispatch_rejected" and dispatch_rejection is None:
            dispatch_rejection = row
        elif row.get("kind") == "workspace_calibration_stage_result":
            stages.append({k: v for k, v in row.items() if k in {"task", "result", "final_qpos_rad", "partial_stage"}})
        elif row.get("kind") == "workspace_gravity_observation":
            observations.append({k: v for k, v in row.items() if k != "gc_observation"})
        elif row.get("kind") == "torque_console_guard" and first_guard is None:
            first_guard = {k: row.get(k) for k in ("reason", "joint_name", "time_sec", "motion_fault")}
        elif row.get("kind") == "control_observation_complete":
            cleanup_boundary = {k: row.get(k) for k in ("boundary", "recorded_monotonic_sec", "cleanup_errors")}
        elif row.get("kind") == "active_dispatch":
            workspace = ((row.get("torque_console") or {}).get("automatic_calibration") or {}).get("workspace")
            if workspace is not None:
                last_workspace = workspace
        elif row.get("kind") == "automatic_calibration_result":
            # The owner emits a terminal record and a final lifecycle snapshot.
            # A later snapshot may add shutdown diagnostics, not replace trials.
            if result is not None and (
                row["batch_id"] != result["batch_id"]
                or row["onset_trials"] != result["onset_trials"]
            ):
                raise ValueError("conflicting automatic results")
            result = row
    recovered = result is None
    if recovered:
        if started is None:
            raise ValueError("no automatic acquisition record")
        result = {
            **started,
            "state": "failed",
            "reason": "terminal_record_missing",
            "outcome": calibration_outcome("failed", "terminal_record_missing"),
            "onset_trials": trials,
            "motion": None,
        }
        if started.get("workspace") is not None:
            # Recover only recorded evidence. A missing final record never
            # becomes successful execution or an invented completed sweep.
            workspace = {**started["workspace"], **(last_workspace or {})}
            acquisitions = [stage for stage in stages if stage["task"]["kind"] == "sweep"]
            workspace.update(
                acquisitions=acquisitions,
                moves=[stage for stage in stages if stage["task"]["kind"] != "sweep"],
                gravity_observations=observations,
                completed_joint_acquisitions=sum(stage["result"]["state"] == "complete" for stage in acquisitions),
                recovered_from_stage_records=True,
            )
            result["workspace"] = workspace
            result["motion"] = {"completed_legs": [
                {**leg, "joint_name": stage["task"]["joint"], "pose_index": stage["task"]["pose"]}
                for stage in acquisitions for leg in stage["result"]["completed_legs"]]}
    report = summarize_automatic_acquisition(result)
    if recovered:
        report["recovery"] = {
            "terminal_record_missing": True, "first_guard": first_guard,
            "control_cleanup_boundary": cleanup_boundary,
            "semantics": "recorded_stages_only; missing_terminal_result_is_not_success_or_disable_receipt",
        }
    # Do not reinterpret old records without an explicit acquisition outcome.
    report["acquisition_outcome"] = result.get("outcome")
    report["cycle"] = result.get("cycle")
    report["operation_mode"] = result.get(
        "operation_mode", "supported_cycle" if result.get("cycle") else "fixed_position"
    )
    report["mode_selection"] = result.get("mode_selection")
    if result.get("workspace") is not None:
        report["workspace"] = result["workspace"]
        if result["workspace"].get("holding_sampling_plan") == "identification":
            # Read the recorded acquisition configuration, not a profile that
            # may have changed after this physical run.
            width = (result["workspace"].get("settings") or {}).get("identification_position_bin_width_rad")
            identification = summarize_workspace_identification(_jsonl(trace),
                condition=report["condition"], workspace=result["workspace"],
                reduction_method="closed_loop_average" if width is not None else "quasisteady",
                position_bin_width_rad=width)
            report["command_identification"] = identification
            report["reference_speed_coverage"] = identification.get("reference_speed_coverage")
            report["local_even_command_identification"] = identification["local_even_command_fit"]
            report["local_command_predictability"] = identification["local_command_predictability"]
            report["empirical_command_field"] = identification["empirical_command_field"]
            report["command_field_runtime"] = identification["command_field_runtime"]
            if "paired_resistance_fit" in identification:
                report["paired_resistance_identification"] = identification["paired_resistance_fit"]
            report["workspace_motion"] = {"status": "CONTINUOUS_IDENTIFICATION_WINDOWS",
                "window_count": len(identification["windows"]), "auto_apply": False}
        else:
            report["workspace_motion"] = _workspace_motion_windows(trace, result["workspace"])
        report["onset_data_assessment"] = {
            "status": "NOT_REQUESTED_GRAVITY_FIRST", "completed_trial_count": 0,
            "expected_trial_count": 0, "semantics": "reacquire onset friction after the gravity baseline is optimized"}
        report["gravity_identification"] = {
            "status": "REQUIRES_PAIRED_MOTION_ANALYSIS", "auto_apply": False,
            "gravity_parameters_frozen": True, "setup_label": result["workspace"]["setup_label"]}
        if report.get("command_identification") is not None:
            report["gravity_identification"] = {**report["command_identification"]["gravity_fit"],
                "gravity_parameters_frozen": True,
                "setup_label": result["workspace"]["setup_label"]}
        holds = []
        for stage in result["workspace"].get("acquisitions", []):
            child = stage["result"]
            for leg in child.get("completed_legs", []):
                if leg.get("holding_calibration") is not None:
                    holds.append({"joint": stage["task"]["joint"], "pose": stage["task"]["pose"],
                        "trial_id": leg["trial_id"], "withdrawal_confirmed": leg.get("withdrawal_confirmed", True),
                        **leg["holding_calibration"]})
            pending = child.get("holding_calibration")
            if pending and pending.get("search"):
                # Completed observations survive a later withdrawal failure.
                holds.append({"joint": stage["task"]["joint"], "pose": stage["task"]["pose"],
                    "withdrawal_confirmed": False, "sample_pose_rad": pending["sample_pose_rad"],
                    "nominal_target_rad": pending["nominal_target_rad"], **pending["search"]})
        if holds:
            verified = sum(h["state"] == "complete" for h in holds)
            report["holding_calibration"] = {"status": "VERIFIED_COMMAND_HOLDS_AVAILABLE" if verified else "PARTIAL_COMMAND_OBSERVATIONS",
                "verified_points": verified, "points": holds, "auto_apply": False,
                "identifies_unique_gravity": False,
                "quantity": "software_command_not_measured_force"}
    report["release_support_observations"] = [
        {"trial_sequence": trial["trial_sequence"], **trial["release_support_observation"]}
        for trial in result["onset_trials"] if trial.get("release_support_observation") is not None
    ]
    report["dispatch_rejection"] = result.get("dispatch_rejection") or dispatch_rejection
    report["source_trace"] = str(trace)
    report["source_annotations"] = str(annotations)
    if report.get("command_identification"):
        # The new workspace mode owns role/authority labels in its trace.
        # Do not require an unrelated legacy manual-annotation journal.
        report["source_annotations"] = None
        report["running_resistance"] = report["command_identification"]["running_resistance"]
    else:
        events = list(_jsonl(annotations))
        report["running_resistance"] = (
            summarize_calibration(events, _jsonl(trace), run_id=str(run_dir.resolve()), max_window_sec=2.0)
            if events else {"status": "INSUFFICIENT_LABELLED_WINDOWS", "auto_apply": False})
    resistance = report["running_resistance"]
    if resistance.get("observed_steady_motion_windows", 0):
        data_status = "STEADY_WINDOWS_AVAILABLE"
    elif resistance.get("recorded_command_windows", 0):
        data_status = "DIAGNOSTIC_WINDOWS_AVAILABLE"
    else:
        data_status = "NO_MOTION_WINDOWS"
    report["motion_data_assessment"] = {
        "status": data_status,
        "recorded_command_windows": resistance.get("recorded_command_windows", 0),
        "observed_steady_motion_windows": resistance.get("observed_steady_motion_windows", 0),
        "model_eligible_windows": resistance.get("steady_motion_windows", 0),
        "retained_partial_leg_windows": resistance.get("retained_partial_leg_windows", 0),
        "semantics": "completed admitted intervals are independent of later arrival/withdrawal; explicit non-fit roles remain diagnostic only",
    }
    report["model_readiness"] = {
        "scope": "running_resistance_fit_only",
        "status": resistance["status"],
        "offline_candidates": resistance.get("offline_candidates", 0),
        "steady_motion_windows": resistance.get("steady_motion_windows", 0),
        "auto_apply": False,
        "onset_data_assessed_separately": True,
        "semantics": "running-resistance fit readiness uses independent window/held-out validation; it does not determine whether completed onset trials are usable, and no measured output force is available",
    }
    report["data_quality_outcome"] = {
        "severity": "info" if resistance.get("steady_motion_windows", 0) else "warning",
        "scope": "model_fit",
        "reason": "review_held_out_validation" if resistance.get("steady_motion_windows", 0) else "no_eligible_steady_motion_windows",
        "action": "retain_data_do_not_auto_apply",
    }
    return report


def save_automatic_report(run_dir: Path, *, output_dir: Path | None = None) -> Path:
    """Read closed journals; optionally recover into a separate fresh directory.

    Source paths and failure semantics remain those of the original run.
    Neither publication path may replace a retained report.
    """
    report = build_automatic_report(run_dir)
    destination = run_dir if output_dir is None else output_dir
    output = destination / "automatic_calibration_report.json"
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination,
                prefix=".automatic-report.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # Publish exclusively: never replace an existing retained report, and
        # never expose a partially serialized report to offline consumers.
        os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output

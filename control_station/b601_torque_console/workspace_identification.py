"""Offline workspace observations and fits; never a device or control owner."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition, fit_local_command_model, reduce_command_window, six,
)
from robot_motion.rebot_b601_rs.gravity_identification import fit_paired_gravity_command

from .calibration_session import calibration_receipt_rejection


def organize_automatic_reports(paths):
    """Create a compact, provenance-bound dataset without rewriting sources.

    A quiet command is a feasible interval observation, never a gravity label.
    Histories/configurations remain run-local; no cross-run fit is implied.
    """
    runs, seen = [], set()
    for path in paths:
        path = Path(path).resolve(strict=True)
        if path in seen:
            raise ValueError("duplicate automatic report input")
        seen.add(path)
        raw = path.read_bytes()
        report = json.loads(raw)
        if report.get("schema") != "b601.automatic-calibration-report.v1":
            raise ValueError("expected an automatic calibration report")
        condition = report.get("condition")
        if not isinstance(condition, dict):
            raise ValueError("automatic report is missing its source/gravity condition")
        records = []
        points = (report.get("holding_calibration") or {}).get("points", [])
        for point_index, point in enumerate(points):
            for source, observations in (("observations", point.get("observations", [])),
                                         ("transitions", point.get("transitions", []))):
                for index, observation in enumerate(observations):
                    records.append({"id": f"{point_index}:{source}:{index}",
                        "joint_name": point["joint"], "pose_index": point["pose"],
                        "trial_id": point.get("trial_id"), "search_state": point["state"],
                        "outcome": observation["outcome"], "source": source,
                        "qpos_rad": observation.get("final_qpos_rad"),
                        "sample_pose_rad": point.get("sample_pose_rad"),
                        "nominal_target_rad": observation.get("nominal_target_rad", point.get("nominal_target_rad")),
                        **{key: observation.get(key) for key in (
                            "command_nm", "requested_command_nm", "actual_delta_nm", "gravity_command_nm",
                            "driver_kd_nm_s_per_rad", "minimum_rad", "maximum_rad",
                            "stationary_minimum_rad", "stationary_maximum_rad", "stationary_dwell_sec",
                            "started_at_sec", "completed_at_sec", "first_causal_receipt_at_sec")},
                        "identifies_unique_gravity": False})
        runs.append({"source_report": str(path), "source_sha256": hashlib.sha256(raw).hexdigest(),
            "condition": condition, "setup_label": (report.get("workspace") or {}).get("setup_label"),
            "batch_state": report.get("batch_state"), "batch_reason": report.get("batch_reason"),
            "verified_holds": sum(p["state"] == "complete" for p in points),
            "records": records, "outcomes": dict(Counter(r["outcome"] for r in records)),
            "continuous_identification": report.get("command_identification")})
    return {"schema": "b601.organized-command-observations.v1", "runs": runs,
        "run_count": len(runs), "record_count": sum(len(r["records"]) for r in runs),
        "auto_apply": False, "quantity": "software_commands_and_encoder_response",
        "trials": [], "physical_payload_and_temperature_measured": False,
        "semantics": "retain completed observations despite later withdrawal failure; do not pool unknown physical conditions"}


def summarize_workspace_identification(records, *, condition, workspace):
    """Admit response windows, then reuse the existing resistance/gravity fits.

    A sample uses the latest complete dispatch preceding that joint's host
    receive time, not the new command computed from the snapshot. Host receive
    time still does not establish a device-side encoder sampling timestamp.
    Native support brakes, role changes and gaps split windows, never budgets.
    """
    condition = CalibrationCondition.from_mapping(condition, run_id=condition["run_id"])
    completed = {leg["trial_id"] for stage in workspace.get("acquisitions", [])
                 for leg in stage["result"].get("completed_legs", [])}
    rejected, windows, observations = Counter(), [], []
    buffer = []
    commands = deque(maxlen=16)
    previous_source_sequence = previous_receipt_sequence = None
    previous_receive_time = None
    identity = None
    settings = {"minimum_window_sec": .6, "maximum_window_sec": 2.0,
        "maximum_feedback_gap_sec": .04, "minimum_motion_rad": .002,
        "maximum_line_error_rad": .0008, "maximum_residual_span_nm": .15}

    def flush(force=False):
        if len(buffer) < 3:
            buffer.clear()
            return
        duration = buffer[-1]["time"]-buffer[0]["time"]
        if duration < settings["minimum_window_sec"]:
            rejected["short_interval_at_boundary"] += 1
            buffer.clear()
            return
        first, last = buffer[0], buffer[-1]
        observation, reason = reduce_command_window(condition=condition,
            joint_name=first["joint"], trial_id=first["trial"],
            times_sec=[r["time"] for r in buffer], qpos_rad=[r["q"] for r in buffer],
            gravity_command_nm=[r["gravity"] for r in buffer],
            conditioned_command_nm=[r["command"] for r in buffer],
            driver_kd_nm_s_per_rad=[r["kd"] for r in buffer],
            minimum_motion_rad=settings["minimum_motion_rad"],
            maximum_line_error_rad=settings["maximum_line_error_rad"],
            maximum_residual_span_nm=settings["maximum_residual_span_nm"])
        if (not force and reason == "insufficient_motion_not_zero_friction"
            and duration < settings["maximum_window_sec"]):
            return
        record = {"joint_name": first["joint"], "pose_index": first["pose"],
            "trial_id": first["trial"], "role": first["role"],
            "reference_speed_rad_s": first["speed"], "start_sec": first["time"],
            "end_sec": last["time"], "sample_count": len(buffer),
            "source_leg_completed": first["trial"] in completed,
            "input_receipt_sequence_start": first["input_sequence"],
            "input_receipt_sequence_end": last["input_sequence"],
            "model_eligible": reason is None, "exclusion_reason": reason}
        j = int(first["joint"][-1])-1
        residual = [r["command"][j]-r["gravity"][j] for r in buffer]
        record.update(qpos_start_rad=first["q"], qpos_end_rad=last["q"],
            qpos_min_rad=[min(r["q"][i] for r in buffer) for i in range(6)],
            qpos_max_rad=[max(r["q"][i] for r in buffer) for i in range(6)],
            command_residual_span_nm=max(residual)-min(residual))
        if observation is not None:
            record["observation"] = {k: v for k, v in asdict(observation).items() if k != "condition"}
            observations.append((first["pose"], first["role"], observation))
        else:
            rejected[reason] += 1
        windows.append(record)
        buffer.clear()

    def admission(row):
        console = row.get("torque_console") or {}
        batch = console.get("automatic_calibration") or {}
        task = (batch.get("workspace") or {}).get("current_task") or {}
        marker = console.get("calibration_trial") or {}
        receipt = row.get("dispatch_receipt") or {}
        if (row.get("startup_state") != "ACTIVE" or row.get("gravity_ramp_fraction") != 1
            or row.get("startup_handoff_fraction") != 1):
            return None, "startup_or_missing_phase_provenance"
        if (task.get("kind") != "sweep" or task.get("joint") not in {"joint2", "joint3"}
            or marker.get("role") not in {"training", "validation"} or marker.get("state") != "moving"
            or (console.get("holding_calibration") or {}).get("enabled") is True):
            return None, "non_identification_interval"
        if (batch.get("human_intervention") is not False or console.get("fault")
            or console.get("phase") != "track" or (batch.get("workspace") or {}).get("supported_soft_stop")):
            return None, "authority_or_braking_interval"
        if any(row.get(flag) is not False for flag in ("torque_limited", "slew_limited", "overlay_limited")):
            return None, "limited_or_unknown_command"
        if six(row.get("vendor_kp_nm_per_rad"), "driver kp") != (0.0,)*6:
            return None, "unmodelled_motor_local_position_gain"
        if calibration_receipt_rejection(receipt, row.get("sequence"), None) is not None:
            return None, "incomplete_or_mismatched_dispatch"
        stamp = row["feedback_timestamp_sec"]
        if not stamp <= receipt["requested_at_sec"] <= receipt["completed_at_sec"] <= row["recorded_monotonic_sec"]:
            return None, "invalid_dispatch_clock"
        if (receipt["completed_at_sec"]-stamp > settings["maximum_feedback_gap_sec"]
            or (row.get("host_observation") or {}).get("host_receive_freshness_verified") is not True):
            return None, "unverified_or_stale_feedback"
        return (task["pose"], task["joint"], marker["trial_id"], marker["role"]), None

    for row in records:
        if row.get("kind") != "active_dispatch":
            if row.get("kind") == "calibration_context" and (
                CalibrationCondition.from_mapping(row.get("profile"), run_id=condition.run_id) != condition
            ):
                raise ValueError("workspace gravity/source condition changed during one run")
            flush(force=True)
            commands.clear()
            identity = None
            previous_receive_time = None
            continue
        key, reason = admission(row)
        if reason:
            rejected[reason] += 1
        if key != identity or key is None:
            flush(force=True)
            commands.clear()
            previous_receive_time = None
        identity = key
        if key is None:
            continue
        receipt = row["dispatch_receipt"]
        if (previous_source_sequence is not None and row["sequence"] <= previous_source_sequence
            or previous_receipt_sequence is not None and receipt["sequence"] <= previous_receipt_sequence):
            raise ValueError("workspace source/dispatch sequences must advance")
        previous_source_sequence, previous_receipt_sequence = row["sequence"], receipt["sequence"]
        sources = (row.get("host_observation") or {}).get("joints") or []
        source = next((s for s in sources if s.get("name") == key[1]), {})
        stamp = source.get("observed_at_sec")
        if (not isinstance(stamp, (int, float)) or isinstance(stamp, bool)
            or not row["feedback_timestamp_sec"]-.04 <= stamp <= row["feedback_timestamp_sec"]):
            rejected["selected_joint_receive_clock_unavailable"] += 1
            flush(force=True)
            commands.clear()
            continue
        if previous_receive_time is not None and stamp == previous_receive_time:
            rejected["duplicate_selected_joint_receive"] += 1
            commands.append(row)
            continue
        if previous_receive_time is not None and not previous_receive_time < stamp <= previous_receive_time+.04:
            rejected["selected_joint_receive_gap"] += 1
            flush(force=True)
            commands.clear()
        previous_receive_time = stamp
        prior = next((r for r in reversed(commands)
            if r["dispatch_receipt"]["completed_at_sec"] < stamp
            and stamp-r["dispatch_receipt"]["completed_at_sec"] <= .04), None)
        if prior is not None:
            if row["vendor_kd_nm_s_per_rad"] != prior["vendor_kd_nm_s_per_rad"]:
                rejected["input_response_damping_change"] += 1
                flush(force=True)
                commands.clear()
            else:
                marker = row["torque_console"]["calibration_trial"]
                buffer.append({"time": stamp, "q": row["qpos_rad"],
                    "gravity": prior["gravity_tau_nm"], "command": prior["conditioned_tau_nm"],
                    "kd": prior["vendor_kd_nm_s_per_rad"], "input_sequence": prior["dispatch_receipt"]["sequence"],
                    "pose": key[0], "joint": key[1], "trial": key[2], "role": key[3],
                    "speed": marker.get("reference_speed_rad_s")})
                if buffer[-1]["time"]-buffer[0]["time"] >= settings["minimum_window_sec"]:
                    flush()
        commands.append(row)
    flush(force=True)
    groups = defaultdict(list)
    for pose, role, observation in observations:
        groups[(pose, observation.joint_name, observation.active_driver_kd_nm_s_per_rad)].append((role, observation))
    fits = []
    for (pose, joint, kd), group in sorted(groups.items()):
        fit = fit_local_command_model([o for role, o in group if role == "training"],
            [o for role, o in group if role == "validation"])
        fits.append({"pose_index": pose, "joint_name": joint,
            "active_driver_kd_nm_s_per_rad": kd, "fit": fit})
    gravity = fit_paired_gravity_command(
        [o for _, role, o in observations if role == "training"],
        [o for _, role, o in observations if role == "validation"])
    candidates = sum(f["fit"]["status"] == "OFFLINE_CANDIDATE" for f in fits)
    resistance = {"status": "OFFLINE_CANDIDATE" if candidates else "INSUFFICIENT_DATA",
        "offline_candidates": candidates, "steady_motion_windows": len(observations),
        "observed_steady_motion_windows": len(observations), "recorded_command_windows": len(windows),
        "retained_partial_leg_windows": sum(w["model_eligible"] and not w["source_leg_completed"] for w in windows),
        "fits": fits, "auto_apply": False}
    return {"schema": "b601.workspace-command-identification.v1", "auto_apply": False,
        "quantity": "software_command_not_measured_output_torque",
        "condition": asdict(condition), "settings": settings, "windows": windows,
        "admitted_motion_windows": len(observations), "rejected": dict(rejected),
        "input_alignment": "latest_complete_dispatch_before_selected_joint_host_receive",
        "device_sampling_timestamp_verified": False,
        "gravity_fit": gravity, "running_resistance": resistance,
        "motor_damping_semantics": "fixed active Kd is a fit condition; never subtract an unmeasured physical damping torque",
        "later_stop_invalidates_completed_windows": False}

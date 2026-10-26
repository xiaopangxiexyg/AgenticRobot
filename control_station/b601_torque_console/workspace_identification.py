"""Offline workspace observations and fits; never a device or control owner."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import asdict
from pathlib import Path

from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition,
    fit_local_command_model,
    reduce_command_window,
    six,
)
from robot_motion.rebot_b601_rs.command_field import evaluate_local_command_field
from robot_motion.rebot_b601_rs.gravity_identification import (
    fit_paired_gravity_command,
    fit_paired_resistance_command,
    summarize_pairing_coverage,
)

from .calibration_session import calibration_receipt_rejection


def _reference_speed_coverage(windows):
    """Describe achieved speed without changing fit admission or pooling splits."""
    groups, unavailable = defaultdict(list), 0
    for window in windows:
        if not window.get("model_eligible"):
            continue
        observation = window.get("observation") or {}
        reference, measured = window.get("reference_speed_rad_s"), observation.get("velocity_rad_s")
        if (any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in (reference, measured)) or reference <= 0 or measured == 0):
            unavailable += 1
            continue
        kd = observation.get("active_driver_kd_nm_s_per_rad")
        key = (window["pose_index"], window["joint_name"], window["role"], reference,
            1 if measured > 0 else -1, None if kd is None else tuple(kd))
        groups[key].append((abs(measured), window["trial_id"]))
    return {"quantity": "measured_encoder_window_speed_not_force",
        "scope": "admitted_command_windows_not_necessarily_quasisteady; no_fit_gate_changed",
        "unavailable_windows": unavailable, "auto_apply": False,
        "groups": [{"pose_index": key[0], "joint_name": key[1], "role": key[2],
            "reference_speed_rad_s": key[3], "measured_direction": key[4],
            "active_driver_kd_nm_s_per_rad": key[5], "window_count": len(values),
            "independent_trial_count": len({trial for _, trial in values}),
            "minimum_measured_speed_rad_s": min(v for v, _ in values),
            "median_measured_speed_rad_s": statistics.median(v for v, _ in values),
            "maximum_measured_speed_rad_s": max(v for v, _ in values),
            "median_measured_to_reference_ratio": statistics.median(v for v, _ in values)/key[3]}
            for key, values in sorted(groups.items(), key=lambda item: repr(item[0]))]}


def _matched_response_distance(a, b, *, pose_tolerance_rad=.0045, maximum_speed_ratio=1.25):
    if (a.trial_id == b.trial_id or a.condition != b.condition or a.joint_name != b.joint_name
        or a.active_driver_kd_nm_s_per_rad != b.active_driver_kd_nm_s_per_rad
        or a.velocity_rad_s*b.velocity_rad_s <= 0):
        return None
    distance = max(abs(x-y) for x, y in zip(a.qpos_rad, b.qpos_rad))
    ratio = max(abs(a.velocity_rad_s), abs(b.velocity_rad_s))/min(abs(a.velocity_rad_s), abs(b.velocity_rad_s))
    return (distance, ratio) if distance <= pose_tolerance_rad and ratio <= maximum_speed_ratio else None


def _repeatability(observations, *, pose_tolerance_rad=.0045, maximum_speed_ratio=1.25):
    """Disjoint same-direction, different-trial matches inside one split.

    This measures observed repeat differences, not sensor accuracy or an
    independent validation score. Matching uses no command/target labels.
    """
    used, pairs = set(), []
    for i, a in enumerate(observations):
        if i in used:
            continue
        choices = []
        for k in range(i+1, len(observations)):
            b = observations[k]
            if k in used:
                continue
            match = _matched_response_distance(a, b, pose_tolerance_rad=pose_tolerance_rad,
                                               maximum_speed_ratio=maximum_speed_ratio)
            if match is not None:
                choices.append((*match, k))
        if choices:
            distance, ratio, k = min(choices)
            b = observations[k]
            used.update((i, k))
            pairs.append({"first_trial_id": a.trial_id, "second_trial_id": b.trial_id,
                "first_qpos_rad": a.qpos_rad, "second_qpos_rad": b.qpos_rad,
                "direction": int(math.copysign(1, a.velocity_rad_s)),
                "maximum_pose_mismatch_rad": distance, "speed_magnitude_ratio": ratio,
                "command_difference_nm": b.command_residual_nm-a.command_residual_nm})
    return {"source_windows": len(observations), "matched_pairs": len(pairs),
        "unmatched_windows": len(observations)-len(used), "pairs": pairs,
        "rms_pair_difference_nm": (math.sqrt(sum(p["command_difference_nm"]**2 for p in pairs)/len(pairs))
            if pairs else None),
        "maximum_abs_pair_difference_nm": max((abs(p["command_difference_nm"]) for p in pairs), default=None),
        "pose_tolerance_rad": pose_tolerance_rad, "maximum_speed_ratio": maximum_speed_ratio,
        "semantics": "matched_encoder_pose_and_speed_not_identical_preload_temperature_or_output_force"}


def _local_repeat_prediction(training, validation):
    """Compatibility view of the reusable training-only command field."""
    return evaluate_local_command_field(training, validation)["directional_validation"]


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
            raise ValueError("automatic report is missing its source/gravity condition")  # noqa: TRY004 - preserve source-provenance refusal contract
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


def _dispatch_integrals(commands, start, end):
    """Integrate the latest-complete-command proxy over a receive interval.

    Six-joint completion is the only verified dispatch boundary in this trace.
    It is not the motor's sampling/actuation clock. Never interpolate a future
    command backwards into an earlier position response.
    """
    held = next((r for r in reversed(commands)
                 if r["dispatch_receipt"]["completed_at_sec"] < start), None)
    if held is None:
        raise ValueError("missing causal command at response interval start")
    total, gravity = [0.0]*6, [0.0]*6
    cursor = start
    events = [r for r in commands if start <= r["dispatch_receipt"]["completed_at_sec"] < end]
    for event in [*events, None]:
        boundary = end if event is None else event["dispatch_receipt"]["completed_at_sec"]
        dt = boundary-cursor
        command = six(held["conditioned_tau_nm"], "dispatched command")
        baseline = six(held["gravity_tau_nm"], "dispatched gravity")
        for j in range(6):
            total[j] += command[j]*dt
            gravity[j] += baseline[j]*dt
        cursor, held = boundary, event
    return total, gravity


def summarize_workspace_identification(records, *, condition, workspace,
                                       reduction_method="quasisteady", position_bin_width_rad=None):
    """Admit response windows, then reuse the existing resistance/gravity fits.

    A sample uses the latest complete dispatch preceding that joint's host
    receive time, not the new command computed from the snapshot. Host receive
    time still does not establish a device-side encoder sampling timestamp.
    Native support brakes, role changes and gaps split windows, never budgets.
    """
    if reduction_method not in {"quasisteady", "closed_loop_average"}:
        raise ValueError("unknown command window reduction method")
    if position_bin_width_rad is not None and (
            reduction_method != "closed_loop_average" or isinstance(position_bin_width_rad, bool)
            or not isinstance(position_bin_width_rad, (float, int))
            or not math.isfinite(position_bin_width_rad) or position_bin_width_rad <= .002):
        raise ValueError("position bins require closed-loop averaging and width greater than minimum motion .002 rad")
    condition = CalibrationCondition.from_mapping(condition, run_id=condition["run_id"])
    completed = {leg["trial_id"] for stage in workspace.get("acquisitions", [])
                 for leg in stage["result"].get("completed_legs", [])}
    rejected, windows, observations = Counter(), [], []
    execution = {}
    unavailable_execution_samples = 0
    buffer = []
    commands = deque(maxlen=16)
    previous_source_sequence = previous_receipt_sequence = None
    previous_receive_time = None
    identity = None
    field_runtime = {"dispatch_samples": 0, "predicted_samples": 0, "requested_increment_samples": 0,
        "modes": Counter(), "reasons": Counter(), "max_abs_requested_increment_nm": 0.0}
    settings = {"minimum_window_sec": .6, "maximum_window_sec": 2.0,
        "maximum_feedback_gap_sec": .04, "minimum_motion_rad": .002,
        "maximum_line_error_rad": .0008, "maximum_residual_span_nm": .15,
        "residual_span_is_admission_gate": reduction_method == "quasisteady",
        "window_basis": "position" if position_bin_width_rad is not None else "time",
        "position_bin_width_rad": position_bin_width_rad,
        "minimum_duration_gate_applied": position_bin_width_rad is None}

    def flush(force=False, position_exit=False):
        if len(buffer) < 3:
            buffer.clear()
            return
        duration = buffer[-1]["time"]-buffer[0]["time"]
        if position_bin_width_rad is None and duration < settings["minimum_window_sec"]:
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
            maximum_residual_span_nm=settings["maximum_residual_span_nm"],
            reduction_method=reduction_method,
            interval_command_integrals_nm_sec=[r["command_integral"] for r in buffer[1:]]
                if reduction_method == "closed_loop_average" else None,
            interval_gravity_integrals_nm_sec=[r["gravity_integral"] for r in buffer[1:]]
                if reduction_method == "closed_loop_average" else None)
        if (not force and reason == "insufficient_motion_not_zero_friction"
            and duration < settings["maximum_window_sec"]):
            return
        kinematic_reason = reason
        if position_bin_width_rad is not None and (not first["position_entry"] or not position_exit):
            # Retain partial boundary intervals, but do not label a tail or
            # interrupted cell as a complete position traversal.
            observation, reason = None, "partial_position_bin_not_for_pairing"
        record = {"joint_name": first["joint"], "pose_index": first["pose"],
            "trial_id": first["trial"], "role": first["role"],
            "reference_speed_rad_s": first["speed"], "start_sec": first["time"],
            "end_sec": last["time"], "sample_count": len(buffer),
            "source_leg_completed": first["trial"] in completed,
            "input_receipt_sequence_start": first["input_sequence"],
            "input_receipt_sequence_end": last["input_sequence"],
            "native_phase_verified": all(r["native_phase_verified"] for r in buffer),
            "model_eligible": reason is None, "exclusion_reason": reason}
        if position_bin_width_rad is not None:
            cell = first["position_cell"]
            record.update(position_bin_index=cell,
                position_bin_bounds_rad=[cell*position_bin_width_rad, (cell+1)*position_bin_width_rad],
                complete_position_traversal=first["position_entry"] and position_exit,
                kinematic_exclusion_reason=kinematic_reason)
        j = int(first["joint"][-1])-1
        residual = [r["command"][j]-r["gravity"][j] for r in buffer]
        record.update(qpos_start_rad=first["q"], qpos_end_rad=last["q"],
            qpos_min_rad=[min(r["q"][i] for r in buffer) for i in range(6)],
            qpos_max_rad=[max(r["q"][i] for r in buffer) for i in range(6)],
            command_residual_span_nm=max(residual)-min(residual),
            reduction_method=reduction_method,
            command_variation_exceeds_quasisteady_limit=max(residual)-min(residual) > settings["maximum_residual_span_nm"])
        if reduction_method == "closed_loop_average":
            mid = len(buffer)//2
            v_start = (buffer[mid]["q"][j]-first["q"][j])/(buffer[mid]["time"]-first["time"])
            v_end = (last["q"][j]-buffer[mid]["q"][j])/(last["time"]-buffer[mid]["time"])
            record.update(motion_class="linear_single_axis" if observation else reason,
                          half_window_speeds_rad_s=[v_start, v_end],
                          acceleration_proxy_rad_s2=2*(v_end-v_start)/duration,
                          inertia_identified=False)
            record["half_speed_magnitude_ratio"] = (max(abs(v_start), abs(v_end))/min(abs(v_start), abs(v_end))
                if v_start*v_end > 0 else None)
            ratio = record["half_speed_magnitude_ratio"]
            record["kinematic_phase_diagnostic"] = ("cruise_candidate" if ratio is not None and ratio <= 1.25
                else "speed_change_or_encoder_resolution_unresolved")
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
        holding = console.get("coordinated_holding") or {}
        if (holding.get("cruise_velocity_control") or {}).get("arrival") is True:
            return None, "endpoint_servo_not_cruise_identification"
        axes = holding.get("axes")
        if axes is not None:
            if not isinstance(axes, list) or len(axes) != 2 or not all(isinstance(a, dict) for a in axes):
                raise ValueError("invalid selected native tracking phase evidence")
            selected = axes[int(task["joint"][-1])-2]
            if selected.get("phase") != "track" or selected.get("reason") != "bounded_reference_tracking":
                return None, "endpoint_servo_not_cruise_identification"
            # Older traces may still label a completed reference as native
            # TRACK while the physical axis is stopped short of its target.
            # Its endpoint restoration is not a constant-speed experiment.
            if (selected.get("target_position_rad") is not None
                and selected.get("q_reference_rad") == selected["target_position_rad"]
                and selected.get("qvel_reference_rad_s") == 0.0):
                return None, "endpoint_servo_not_cruise_identification"
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
        field = ((row.get("torque_console") or {}).get("coordinated_holding") or {}).get("command_field")
        if field is not None:
            if not isinstance(field, dict) or field.get("mode") not in {"shadow", "candidate"} or type(field.get("predicted")) is not bool:
                raise ValueError("invalid runtime command field evidence")
            increment = field.get("requested_increment_nm")
            if isinstance(increment, bool) or not isinstance(increment, (int, float)) or not math.isfinite(increment):
                raise ValueError("invalid runtime command field increment")
            if field["mode"] == "shadow" and abs(increment) > 1e-12:
                raise ValueError("shadow command field cannot request a drive increment")
            field_runtime["dispatch_samples"] += 1
            field_runtime["predicted_samples"] += field["predicted"]
            field_runtime["requested_increment_samples"] += abs(increment) > 1e-12
            field_runtime["modes"][field["mode"]] += 1
            field_runtime["reasons"][field.get("reason", "not_recorded")] += 1
            field_runtime["max_abs_requested_increment_nm"] = max(field_runtime["max_abs_requested_increment_nm"], abs(increment))
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
        holding = (row.get("torque_console") or {}).get("coordinated_holding") or {}
        if all(name in holding for name in ("combined_command_nm", "combined_cap_shortfall_nm", "saturated")):
            j = int(key[1][-1])-1
            delta = six(holding["combined_command_nm"], "coordinated command diagnostic")
            shortfall = six(holding["combined_cap_shortfall_nm"], "coordinated cap diagnostic")
            cruise = holding.get("cruise_velocity_control") or {}
            item = execution.setdefault(key, {"pose_index": key[0], "joint_name": key[1],
                "trial_id": key[2], "role": key[3], "dispatch_samples": 0,
                "cruise_telemetry_samples": 0,
                "cruise_active_samples": 0, "combined_cap_samples": 0,
                "integral_growth_blocked_samples": 0, "cruise_reasons": Counter(),
                "sum_abs_selected_command_nm": 0.0, "sum_abs_support_command_nm": 0.0,
                "sum_abs_selected_cap_shortfall_nm": 0.0})
            item["dispatch_samples"] += 1
            item["cruise_telemetry_samples"] += type(cruise.get("active")) is bool
            item["cruise_active_samples"] += cruise.get("active") is True
            item["combined_cap_samples"] += holding.get("saturated") is True
            item["integral_growth_blocked_samples"] += cruise.get("integral_growth_blocked") is True
            item["cruise_reasons"][cruise.get("reason", "not_recorded")] += 1
            item["sum_abs_selected_command_nm"] += abs(delta[j])
            item["sum_abs_support_command_nm"] += abs(delta[3-j])
            item["sum_abs_selected_cap_shortfall_nm"] += abs(shortfall[j])
        else:
            # Historical traces may predate command-allocation telemetry.
            # Its absence is not evidence of either zero clipping or bad data.
            unavailable_execution_samples += 1
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
                cell, entry_direction = None, 0
                if position_bin_width_rad is not None:
                    j = int(key[1][-1])-1
                    cell = math.floor(six(row["qpos_rad"], "qpos_rad")[j]/position_bin_width_rad)
                    previous_cell = math.floor(commands[-1]["qpos_rad"][j]/position_bin_width_rad)
                    entry_direction = cell-previous_cell if abs(cell-previous_cell) == 1 else 0
                    if buffer and cell != buffer[0]["position_cell"]:
                        flush(force=True, position_exit=cell-buffer[0]["position_cell"] == buffer[0]["position_entry_direction"])
                integral = (_dispatch_integrals(commands, buffer[-1]["time"], stamp)
                    if buffer and reduction_method == "closed_loop_average" else (None, None))
                buffer.append({"time": stamp, "q": row["qpos_rad"],
                    "gravity": prior["gravity_tau_nm"], "command": prior["conditioned_tau_nm"],
                    "kd": prior["vendor_kd_nm_s_per_rad"], "input_sequence": prior["dispatch_receipt"]["sequence"],
                    "pose": key[0], "joint": key[1], "trial": key[2], "role": key[3],
                    "speed": marker.get("reference_speed_rad_s"),
                    "native_phase_verified": isinstance(holding.get("axes"), list),
                    "position_cell": cell, "position_entry": bool(entry_direction),
                    "position_entry_direction": entry_direction,
                    "command_integral": integral[0], "gravity_integral": integral[1]})
                if (position_bin_width_rad is None
                    and buffer[-1]["time"]-buffer[0]["time"] >= settings["minimum_window_sec"]):
                    flush()
        commands.append(row)
    flush(force=True)
    groups = defaultdict(list)
    for pose, role, observation in observations:
        groups[(pose, observation.joint_name, observation.active_driver_kd_nm_s_per_rad)].append((role, observation))
    fits, paired_models, fields = [], [], []
    for (pose, joint, kd), group in sorted(groups.items()):
        fit = fit_local_command_model([o for role, o in group if role == "training"],
            [o for role, o in group if role == "validation"])
        item = {"pose_index": pose, "joint_name": joint,
            "active_driver_kd_nm_s_per_rad": kd, "fit": fit}
        if reduction_method == "closed_loop_average":
            assessment = evaluate_local_command_field(
                [o for role, o in group if role == "training"],
                [o for role, o in group if role == "validation"])
            item.update(repeatability_by_split={role: _repeatability([o for r, o in group if r == role])
                for role in ("training", "validation")},
                local_repeat_prediction=assessment["directional_validation"])
            fields.append({"pose_index": pose, "joint_name": joint,
                "active_driver_kd_nm_s_per_rad": kd,
                **{k: v for k, v in assessment.items() if k != "directional_validation"},
                "directional_validation": {k: v for k, v in assessment["directional_validation"].items()
                    if k != "predictions"}})
            paired_models.append({"pose_index": pose, "joint_name": joint,
                "active_driver_kd_nm_s_per_rad": kd,
                "analysis": fit_paired_resistance_command(
                    [o for role, o in group if role == "training"],
                    [o for role, o in group if role == "validation"])})
        fits.append(item)
    gravity = fit_paired_gravity_command(
        [o for _, role, o in observations if role == "training"],
        [o for _, role, o in observations if role == "validation"])
    local_groups = defaultdict(list)
    for pose, role, observation in observations:
        local_groups[(pose, observation.active_driver_kd_nm_s_per_rad)].append((role, observation))
    local_even = [{"pose_index": pose, "active_driver_kd_nm_s_per_rad": kd,
        "analysis": fit_paired_gravity_command(
            [o for role, o in group if role == "training"],
            [o for role, o in group if role == "validation"], model="local-joint-bias")}
        for (pose, kd), group in sorted(local_groups.items())]
    candidates = sum(f["fit"]["status"] == "OFFLINE_CANDIDATE" for f in fits)
    cruise_count = (sum(w.get("kinematic_phase_diagnostic") == "cruise_candidate"
                       for w in windows if w["model_eligible"])
                    if reduction_method == "closed_loop_average" else len(observations))
    resistance = {"status": "OFFLINE_CANDIDATE" if candidates else "INSUFFICIENT_DATA",
        "fit_scope": "equivalent_window_command_prediction_not_identified_physical_friction",
        "offline_candidates": candidates, "steady_motion_windows": cruise_count,
        "observed_steady_motion_windows": cruise_count, "recorded_command_windows": len(windows),
        "admitted_command_response_windows": len(observations),
        "cruise_diagnostic_semantics": "half-speed ratio <= 1.25 is diagnostic only, not proof of zero inertia",
        "retained_partial_leg_windows": sum(w["model_eligible"] and not w["source_leg_completed"] for w in windows),
        "fits": fits, "auto_apply": False}
    result = {"schema": "b601.workspace-command-identification.v1", "auto_apply": False,
        "command_field_context": {"setup_label": workspace.get("setup_label"),
            "pose_centres_rad": workspace.get("pose_centres_rad")},
        "reduction_method": reduction_method,
        "quantity": "software_command_not_measured_output_torque",
        "condition": asdict(condition), "settings": settings, "windows": windows,
        "admitted_motion_windows": len(observations), "rejected": dict(rejected),
        "reference_speed_coverage": _reference_speed_coverage(windows),
        "input_alignment": "latest_complete_dispatch_before_selected_joint_host_receive",
        "device_sampling_timestamp_verified": False,
        "reduction_semantics": ("time_integral_of_latest_complete_dispatched_input_over_duration; "
            "velocity_is_measured_displacement_over_duration; fixed_Kd_condition; "
            "kinematic_linearity_is_not_proof_of_zero_inertia_or_physical_friction"
            if reduction_method == "closed_loop_average" else "median_quasisteady_command"),
        "gravity_fit": gravity, "running_resistance": resistance,
        "local_even_command_fit": {"scope": "separate_local_full_pose_and_fixed_damping_regions",
            "gravity_scales_frozen": True, "auto_apply": False, "profile_export_allowed": False,
            "fits": local_even},
        "local_command_predictability": {"status": "DIAGNOSTIC_ONLY", "auto_apply": False,
            "quantity": "direction_conditioned_software_command_not_separated_gravity_or_friction",
            "regions": [{"pose_index": f["pose_index"], "joint_name": f["joint_name"],
                "active_driver_kd_nm_s_per_rad": f["active_driver_kd_nm_s_per_rad"],
                **{k: v for k, v in f["local_repeat_prediction"].items() if k != "predictions"}}
                for f in fits if "local_repeat_prediction" in f]},
        "empirical_command_field": {"status": "OFFLINE_ONLY", "auto_apply": False,
            "live_compensation_enabled": False, "regions": fields,
            "scope": "separate_local_full_pose_and_fixed_damping_regions; no_static_or_breakaway_model"},
        "command_field_runtime": {**field_runtime, "modes": dict(field_runtime["modes"]), "reasons": dict(field_runtime["reasons"]),
            "scope": "recorded_active_dispatches_not_only_admitted_fit_windows",
            "query_velocity": "causal_committed_reference_window_not_measured_overspeed",
            "increments_are_requested_not_isolated_executed_force": True},
        "cruise_execution_diagnostic": {"quantity": "software_commands_not_measured_force",
            "scope": "admitted_complete_training_validation_dispatches_not_just_fit_windows",
            "unavailable_samples": unavailable_execution_samples,
            "trials": [{**{k: v for k, v in item.items() if not k.startswith("sum_")},
                "mean_abs_selected_command_nm": item["sum_abs_selected_command_nm"]/item["dispatch_samples"],
                "mean_abs_support_command_nm": item["sum_abs_support_command_nm"]/item["dispatch_samples"],
                "mean_abs_selected_cap_shortfall_nm": item["sum_abs_selected_cap_shortfall_nm"]/item["dispatch_samples"]}
                for item in execution.values()]},
        "kinematic_phase_counts": dict(Counter(w.get("kinematic_phase_diagnostic", "quasisteady")
            for w in windows if w["model_eligible"])),
        "motor_damping_semantics": "fixed active Kd is a fit condition; never subtract an unmeasured physical damping torque",
        "later_stop_invalidates_completed_windows": False}
    if reduction_method == "closed_loop_average":
        training = [o for _, role, o in observations if role == "training"]
        validation = [o for _, role, o in observations if role == "validation"]
        candidates = sum(f["status"] == "OFFLINE_CANDIDATE"
            for group in paired_models for f in group["analysis"]["fits"])
        result.update(paired_resistance_fit={"status": "OFFLINE_CANDIDATE" if candidates else "NOT_READY",
            "offline_candidates": candidates, "auto_apply": False, "hardware_approved": False,
            "quantity": "paired_odd_software_command_residual", "scope": "separate_local_pose_and_fixed_damping_regions",
            "fits": paired_models},
            pairing_coverage={role: summarize_pairing_coverage(subset)
                for role, subset in (("training", training), ("validation", validation))})
    return result

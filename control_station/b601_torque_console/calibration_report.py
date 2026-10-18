"""Streaming admission and local command fits from existing owner evidence.

This application adapter has no connection, control or model-installation path.
Human-free windows are operator declarations, not automatic contact detection.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from robot_motion.rebot_b601_rs.calibration import (
    CalibrationCondition,
    CalibrationObservation,
    finite,
    fit_local_command_model,
    plan_local_sweeps,
    reduce_command_window,
    six,
)
from robot_motion.rebot_b601_rs.joint_motion import B601JointMotionConfig
from robot_motion.rebot_b601_rs.mit import B601_RS_JOINT_MIT_FRAME
from robot_motion.rebot_b601_rs.model import B601_RS_ARM_JOINT_NAMES

from .transparency_report import load_trial_windows


@dataclass(frozen=True)
class _CommandWindow:
    observation: CalibrationObservation | None
    trial_id: str
    joint_name: str
    context_index: int
    start_sec: float
    end_sec: float
    recorded_end_sec: float
    sample_count: int
    qpos_start_rad: tuple[float, ...]
    qpos_end_rad: tuple[float, ...]
    command_residual_start_nm: float
    command_residual_end_nm: float
    quality_reason: str | None


def _admission_reason(row: dict[str, Any], trial: dict[str, Any]) -> str | None:
    if trial["end_sec"] is None or trial["end_reason"] not in (
        "context_change",
        "owner_closed",
        "finish",
    ):
        return "unfinished_or_aborted_annotation_window"
    meta = trial["metadata"]
    if meta.get("kind") == "joint_diagnostics":
        if (
            meta.get("human_intervention") is not False
            or meta.get("drive_enabled") is not True
        ):
            return "no_confirmed_human_free_drive_window"
    elif meta.get("contact") != "none" or any(
        c["contact"] != "none" for c in trial["contacts"]
    ):
        return "no_confirmed_human_free_drive_window"
    if (
        row.get("startup_state") != "ACTIVE"
        or row.get("gravity_ramp_fraction") != 1.0
        or row.get("startup_handoff_fraction") != 1.0
    ):
        return "startup_or_missing_phase_provenance"
    if (
        row.get("torque_limited") is not False
        or row.get("slew_limited") is not False
        or row.get("overlay_limited") is not False
    ):
        return "limited_or_unknown_command"
    # Position-spring torque is not in this command-residual model. Fixed
    # motor damping is instead a separate identification condition, never a
    # torque subtraction using unverified motor velocity.
    if six(row.get("vendor_kp_nm_per_rad"), "vendor_kp_nm_per_rad") != (0.0,) * 6:
        return "motor_local_position_gain_not_supported"
    if min(six(row.get("vendor_kd_nm_s_per_rad"), "vendor_kd_nm_s_per_rad")) < 0:
        return "invalid_motor_local_damping"
    receipt = row.get("dispatch_receipt")
    sequence = row.get("sequence")
    if (
        isinstance(sequence, bool)
        or not isinstance(sequence, int)
        or sequence < 0
        or not isinstance(receipt, dict)
        or receipt.get("accepted") is not True
        or receipt.get("mode") != "mit"
        or receipt.get("frame_id") != B601_RS_JOINT_MIT_FRAME
        or type(receipt.get("source_sequence")) is not int
        or tuple(receipt.get("sent_joint_names", ())) != B601_RS_ARM_JOINT_NAMES
        or tuple(receipt.get("requested_joint_names", ())) != B601_RS_ARM_JOINT_NAMES
        or receipt.get("errors") != []
        or receipt.get("failed_joint_names") != []
        or receipt.get("source_sequence") != row.get("sequence")
    ):
        return "missing_or_incomplete_same_sample_dispatch_receipt"
    requested = finite(receipt.get("requested_at_sec"), "receipt requested_at_sec")
    completed = finite(receipt.get("completed_at_sec"), "receipt completed_at_sec")
    feedback = finite(row.get("feedback_timestamp_sec"), "feedback_timestamp_sec")
    recorded = finite(row.get("recorded_monotonic_sec"), "recorded_monotonic_sec")
    if not feedback <= requested <= completed <= recorded:
        return "dispatch_receipt_not_causally_aligned"
    observed = row.get("host_observation")
    if (
        not isinstance(observed, dict)
        or observed.get("host_receive_freshness_verified") is not True
    ):
        return "feedback_freshness_unverified"
    console = row.get("torque_console")
    if (
        not isinstance(console, dict)
        or console.get("joint_name") != trial["joint_name"]
        or console.get("fault")
    ):
        return "console_joint_or_fault_mismatch"
    if isinstance(console.get("automatic_calibration"), dict) and console["automatic_calibration"].get("stage") == "onset":
        return "onset_characterization_not_running_resistance_fit"
    if console.get("controller_mode") == "staged_assist":
        motion = console.get("motion_control")
        if (
            console.get("operator_armed") is not True
            or console.get("phase") != "track"
            or not isinstance(motion, dict)
            or motion.get("phase") != "track"
        ):
            return "not_steady_tracking_phase"
    elif console.get("phase") != "armed":
        return "console_joint_or_fault_mismatch"
    return None


def summarize_calibration(
    events: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    run_id: str,
    window_sec: float = 0.4,
    max_window_sec: float | None = None,
    max_feedback_gap_sec: float = 0.04,
    minimum_motion_rad: float = 0.002,
    maximum_line_error_rad: float = 0.0005,
    pose_region_radius_rad: float = 0.05,
    max_validation_rmse_nm: float = 0.05,
    maximum_residual_span_nm: float = 0.02,
) -> dict[str, Any]:
    for name, value in (
        ("window_sec", window_sec),
        ("max_feedback_gap_sec", max_feedback_gap_sec),
        ("minimum_motion_rad", minimum_motion_rad),
        ("maximum_line_error_rad", maximum_line_error_rad),
        ("pose_region_radius_rad", pose_region_radius_rad),
        ("max_validation_rmse_nm", max_validation_rmse_nm),
        ("maximum_residual_span_nm", maximum_residual_span_nm),
    ):
        if finite(value, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if window_sec <= max_feedback_gap_sec:
        raise ValueError("calibration window must exceed the allowed feedback gap")
    max_window_sec = (
        window_sec if max_window_sec is None else finite(max_window_sec, "max_window_sec")
    )
    if max_window_sec < window_sec:
        raise ValueError("max_window_sec must not be shorter than window_sec")
    trials = load_trial_windows(events)
    rejected: Counter[str] = Counter()
    windows: list[_CommandWindow] = []
    condition: CalibrationCondition | None = None
    buffer: list[dict[str, Any]] = []
    pending_motion_resolution = False

    def clear_window() -> None:
        nonlocal pending_motion_resolution
        if pending_motion_resolution:
            rejected["insufficient_motion_before_interval_boundary"] += 1
        buffer.clear()
        pending_motion_resolution = False

    trial_index = 0
    last_trial_id = None
    previous_time = -math.inf
    previous_feedback = None
    dispatch_count = 0
    contexts: list[dict[str, Any]] = []
    planning_limits: dict[CalibrationCondition, dict[str, Any]] = {}
    planning_anchors: dict[tuple[CalibrationCondition, str], tuple[float, ...]] = {}
    automatic_trials: dict[str, tuple[str, str, str]] = {}
    completed_legs: dict[tuple[str, str], tuple[str, float]] = {}
    ended_batches: dict[str, dict[str, Any]] = {}
    for row in records:
        kind = row.get("kind")
        if kind == "calibration_batch_result":
            batch = row.get("batch_id")
            legs = row.get("completed_legs")
            if (
                row.get("schema") != "b601.calibration-batch-result.v1"
                or row.get("state") not in {"complete", "cancelled", "failed"}
                or row.get("pending_dispatch") is not False
                or not isinstance(batch, str)
                or not batch
                or batch in ended_batches
                or not isinstance(legs, list)
            ):
                raise ValueError("invalid or duplicate automatic batch result")
            ended_batches[batch] = row
            result_time = finite(row.get("recorded_monotonic_sec"), "batch result time")
            if result_time < previous_time:
                raise ValueError("automatic batch result precedes recorded dispatch")
            last_completion = -math.inf
            for leg in legs:
                if (
                    not isinstance(leg, dict)
                    or not isinstance(leg.get("trial_id"), str)
                    or not leg["trial_id"]
                ):
                    raise ValueError("invalid completed leg identity")
                completed = finite(leg.get("completed_at_sec"), "leg completion time")
                if not last_completion <= completed <= result_time or leg.get(
                    "role"
                ) not in {
                    "training",
                    "validation",
                    "positioning_not_for_fit",
                    "return_not_for_fit",
                }:
                    raise ValueError("invalid leg completion order or role")
                last_completion = completed
                key = (batch, leg["trial_id"])
                if key in completed_legs:
                    raise ValueError("duplicate completed calibration leg")
                completed_legs[key] = (leg["role"], completed)
            clear_window()
            previous_feedback = None
            continue
        if kind == "calibration_context":
            if row.get("schema") != "b601.command-calibration-context.v1":
                raise ValueError("unknown calibration context schema")
            condition = CalibrationCondition.from_mapping(
                row.get("profile"), run_id=run_id
            )
            contexts.append(asdict(condition))
            if not isinstance(row.get("planning_limits", {}), dict):
                raise TypeError("planning_limits must be an object")
            planning_limits[condition] = row.get("planning_limits", {})
            clear_window()
            previous_feedback = None
            continue
        if kind != "active_dispatch":
            # Guards, hold and quarantine break a steady-motion interval.
            clear_window()
            previous_feedback = None
            continue
        dispatch_count += 1
        time_sec = finite(row.get("recorded_monotonic_sec"), "recorded_monotonic_sec")
        if time_sec <= previous_time:
            raise ValueError("dispatch times must advance")
        previous_time = time_sec
        while (
            trial_index < len(trials)
            and trials[trial_index]["end_sec"] is not None
            and time_sec >= trials[trial_index]["end_sec"]
        ):
            trial_index += 1
        if condition is None:
            rejected["missing_run_gravity_context"] += 1
            continue
        if trial_index >= len(trials) or time_sec < trials[trial_index]["start_sec"]:
            rejected["outside_annotation_window"] += 1
            clear_window()
            continue
        trial = trials[trial_index]
        console = row.get("torque_console")
        marker = console.get("calibration_trial") if isinstance(console, dict) else None
        if marker is not None:
            if (
                not isinstance(marker, dict)
                or any(
                    not isinstance(marker.get(k), str) or not marker[k]
                    for k in ("batch_id", "trial_id")
                )
                or marker.get("role")
                not in {
                    "training",
                    "validation",
                    "positioning_not_for_fit",
                    "return_not_for_fit",
                }
            ):
                raise ValueError("invalid automatic calibration trial marker")
            if marker["batch_id"] in ended_batches and marker.get("state") == "moving":
                raise ValueError("automatic trial follows its terminal batch result")
            if marker.get("state") != "moving":
                rejected["automatic_positioning_release_or_nonfit_leg"] += 1
                clear_window()
                previous_feedback = None
                continue
            ident = f"{trial['trial_id']}/{marker['batch_id']}/{marker['trial_id']}"
            identity = (marker["batch_id"], marker["trial_id"], marker["role"])
            if automatic_trials.setdefault(ident, identity) != identity:
                raise ValueError("automatic calibration role changed within a trial")
            trial = {**trial, "trial_id": ident}
        if trial["trial_id"] != last_trial_id:
            clear_window()
            previous_feedback = None
            last_trial_id = trial["trial_id"]
        reason = _admission_reason(row, trial)
        if reason:
            rejected[reason] += 1
            clear_window()
            previous_feedback = None
            continue
        timestamp = finite(row.get("feedback_timestamp_sec"), "feedback_timestamp_sec")
        if (
            timestamp < trial["start_sec"]
            or not 0 <= time_sec - timestamp <= max_feedback_gap_sec
        ):
            rejected["feedback_outside_causal_annotation_or_freshness_window"] += 1
            clear_window()
            previous_feedback = None
            continue
        if (
            previous_feedback is not None
            and not 0 < timestamp - previous_feedback <= max_feedback_gap_sec
        ):
            rejected["feedback_gap_or_duplicate"] += 1
            clear_window()
            previous_feedback = timestamp
            continue
        previous_feedback = timestamp
        planning_anchors.setdefault(
            (condition, trial["joint_name"]), six(row.get("qpos_rad"), "qpos_rad")
        )
        buffer.append(row)
        if timestamp - buffer[0]["feedback_timestamp_sec"] < window_sec:
            continue
        observation, reason = reduce_command_window(
            condition=condition,
            joint_name=trial["joint_name"],
            trial_id=str(trial["trial_id"]),
            times_sec=[r["feedback_timestamp_sec"] for r in buffer],
            qpos_rad=[r.get("qpos_rad") for r in buffer],
            gravity_command_nm=[r.get("gravity_tau_nm") for r in buffer],
            conditioned_command_nm=[r.get("conditioned_tau_nm") for r in buffer],
            driver_kd_nm_s_per_rad=[r.get("vendor_kd_nm_s_per_rad") for r in buffer],
            minimum_motion_rad=minimum_motion_rad,
            maximum_line_error_rad=maximum_line_error_rad,
            maximum_residual_span_nm=maximum_residual_span_nm,
        )
        if (
            reason == "insufficient_motion_not_zero_friction"
            and timestamp - buffer[0]["feedback_timestamp_sec"] < max_window_sec
        ):
            pending_motion_resolution = True
            continue
        pending_motion_resolution = False
        j = B601_RS_ARM_JOINT_NAMES.index(trial["joint_name"])
        first, last = buffer[0], buffer[-1]
        windows.append(
            _CommandWindow(
                observation=observation,
                trial_id=str(trial["trial_id"]),
                joint_name=trial["joint_name"],
                context_index=len(contexts) - 1,
                start_sec=first["feedback_timestamp_sec"],
                end_sec=timestamp,
                recorded_end_sec=time_sec,
                sample_count=len(buffer),
                qpos_start_rad=six(first["qpos_rad"], "window start"),
                qpos_end_rad=six(last["qpos_rad"], "window end"),
                command_residual_start_nm=first["conditioned_tau_nm"][j] - first["gravity_tau_nm"][j],
                command_residual_end_nm=last["conditioned_tau_nm"][j] - last["gravity_tau_nm"][j],
                quality_reason=reason,
            )
        )
        if reason is not None:
            rejected[reason] += 1
        clear_window()

    clear_window()
    # A completed, admitted interval has its own evidence boundary. A later
    # stop cannot erase it, but missing terminal provenance and explicit
    # non-fit roles still prevent use as a model example.
    observations: list[CalibrationObservation] = []
    window_records = []
    for window in windows:
        observation = window.observation
        automatic = automatic_trials.get(window.trial_id)
        excluded = window.quality_reason
        source_batch = None
        leg_completed = None
        partial_leg = False
        if automatic is not None:
            if automatic[2] not in {"training", "validation"} and excluded is None:
                excluded = "automatic_nonfit_motion_window"
            source_batch = ended_batches.get(automatic[0])
            completion = completed_legs.get(automatic[:2])
            leg_completed = completion is not None
            if completion is not None:
                if (
                    completion[0] != automatic[2]
                    or completion[1] < window.recorded_end_sec
                ):
                    raise ValueError("automatic leg completion role/time differs from its observations")
            else:
                terminal_trial = (source_batch or {}).get("calibration_trial")
                partial_leg = (
                    source_batch is not None
                    and source_batch["state"] in {"failed", "cancelled"}
                    and isinstance(terminal_trial, dict)
                    and terminal_trial.get("state") == source_batch["state"]
                    and tuple(
                        terminal_trial.get(k) for k in ("batch_id", "trial_id", "role")
                    ) == automatic
                    and window.recorded_end_sec <= source_batch["recorded_monotonic_sec"]
                )
                if not partial_leg and excluded is None:
                    excluded = "automatic_leg_completion_not_confirmed"
        record = {
            **asdict(window),
            "role": automatic[2] if automatic else "manual_whole_trial_split",
            "model_eligible": excluded is None,
            "exclusion_reason": excluded,
            "source_leg_completed": leg_completed,
            "retained_partial_leg": partial_leg,
            "source_batch_state": source_batch["state"] if source_batch else None,
            "source_batch_reason": source_batch.get("reason") if source_batch else None,
        }
        window_records.append(record)
        if excluded is None:
            assert observation is not None
            observations.append(observation)
        elif window.quality_reason is None:
            rejected[excluded] += 1
    groups: list[list[CalibrationObservation]] = []
    for observation in observations:
        for group in groups:
            anchor = group[0]
            if (
                observation.condition == anchor.condition
                and observation.joint_name == anchor.joint_name
                and observation.active_driver_kd_nm_s_per_rad
                == anchor.active_driver_kd_nm_s_per_rad
                and (observation.trial_id in automatic_trials)
                == (anchor.trial_id in automatic_trials)
                and max(
                    abs(a - b) for a, b in zip(observation.qpos_rad, anchor.qpos_rad)
                )
                <= pose_region_radius_rad
            ):
                group.append(observation)
                break
        else:
            groups.append([observation])
    fits = []
    for group in groups:
        # Automatic batches reserve validation before execution. Legacy manual
        # journals hold out whole last trials, never nearby frames.
        if group[0].trial_id in automatic_trials:
            validation_ids = {
                o.trial_id
                for o in group
                if automatic_trials[o.trial_id][2] == "validation"
            }
        else:
            validation_ids = set()
            for direction in (-1, 1):
                ids = list(
                    dict.fromkeys(
                        o.trial_id for o in group if o.velocity_rad_s * direction > 0
                    )
                )
                if ids:
                    validation_ids.add(ids[-1])
        fit = fit_local_command_model(
            [o for o in group if o.trial_id not in validation_ids],
            [o for o in group if o.trial_id in validation_ids],
            max_pose_span_rad=2 * pose_region_radius_rad,
            max_validation_rmse_nm=max_validation_rmse_nm,
        )
        fits.append(
            {
                "condition": asdict(group[0].condition),
                "joint_name": group[0].joint_name,
                "active_driver_kd_nm_s_per_rad": group[0].active_driver_kd_nm_s_per_rad,
                "anchor_qpos_rad": group[0].qpos_rad,
                "fit": fit,
            }
        )
    candidates = sum(g["fit"]["status"] == "OFFLINE_CANDIDATE" for g in fits)
    proposals = []
    for (context, joint), anchor in planning_anchors.items():
        limits = planning_limits[context]
        if (
            "position_limits_rad" not in limits
            or "joint_velocity_guard_rad_s" not in limits
        ):
            proposals.append(
                {"joint_name": joint, "status": "MISSING_PLANNING_ENVELOPE"}
            )
            continue
        speed = min(
            B601JointMotionConfig().max_reference_velocity_rad_s,
            finite(limits["joint_velocity_guard_rad_s"], "joint_velocity_guard_rad_s")
            * 0.5,
        )
        try:
            plan = plan_local_sweeps(
                condition=context,
                joint_name=joint,
                anchor_qpos_rad=anchor,
                position_limits_rad=limits["position_limits_rad"],
                excursion_rad=0.02,
                speeds_rad_s=(speed * 0.5, speed),
                max_reference_velocity_rad_s=speed,
            )
        except ValueError as error:
            proposals.append(
                {
                    "joint_name": joint,
                    "status": "OUTSIDE_RECORDED_JOINT_ENVELOPE",
                    "reason": str(error),
                }
            )
        else:
            proposals.append(
                {
                    "joint_name": joint,
                    "status": "DRAFT_NOT_EXECUTABLE",
                    "plan": plan,
                    "unresolved": "workspace_collision_clearance_owner_adapter_and_trial_budgets_not_validated",
                }
            )
    return {
        "schema": "b601.conditioned-calibration-report.v1",
        "status": "OFFLINE_CANDIDATE" if candidates else "INSUFFICIENT_DATA",
        "auto_apply": False,
        "hardware_approved": False,
        "motion_commands_sent": False,
        "run_local_physical_setup": "external_load_mounting_and_human_contact_not_independently_measured",
        "trials": [
            {"trial_id": t["trial_id"], "joint_name": t["joint_name"]} for t in trials
        ],
        "contexts": contexts,
        "dispatch_rows": dispatch_count,
        "steady_motion_windows": len(observations),
        "recorded_command_windows": len(windows),
        "observed_steady_motion_windows": sum(w.observation is not None for w in windows),
        "retained_partial_leg_windows": sum(
            w["retained_partial_leg"] and w["quality_reason"] is None
            for w in window_records
        ),
        "motion_windows": window_records,
        "rejected": dict(rejected),
        "groups": fits,
        "offline_candidates": candidates,
        "experiment_proposals": proposals,
        "criteria": {
            "window_sec": window_sec,
            "max_window_sec": max_window_sec,
            "max_feedback_gap_sec": max_feedback_gap_sec,
            "minimum_motion_rad": minimum_motion_rad,
            "maximum_line_error_rad": maximum_line_error_rad,
            "pose_region_radius_rad": pose_region_radius_rad,
            "maximum_residual_span_nm": maximum_residual_span_nm,
            "history_scope": "entry_preload_and_temperature_not_yet_identified",
        },
        "next_experiment": "bounded_bidirectional_multispeed_sweeps_with_whole_trial_validation",
    }

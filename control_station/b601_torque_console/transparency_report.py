"""Offline assessment of labelled haptic trials, not friction identification.

Reported torques are software commands. Human contact and gauge observations
are operator declarations. Neither motion nor command torque is converted into
an inferred hand force. The existing trace is streamed once; raw data is kept.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import chain
from typing import Any

from .transparency import (
    CONTACT_STATES,
    OBSERVATION_FIELDS,
    finite_number,
    six_numbers,
    validate_observation,
    validate_trial_metadata,
)


def _new_trial(trial_id: int, snapshot: dict[str, Any], metadata: dict[str, Any], timestamp: float) -> dict[str, Any]:
    joint = snapshot.get("joint_name")
    if joint not in [f"joint{i}" for i in range(1, 7)]:
        raise ValueError("invalid recorded joint")
    return {
        "trial_id": trial_id, "metadata": metadata,
        "joint_name": joint, "start_sec": timestamp, "end_sec": None,
        "end_reason": "unfinished", "observation": None,
        "start_qpos_rad": six_numbers(snapshot.get("qpos_rad"), "start_qpos_rad"),
        "contacts": [{"time_sec": timestamp, "contact": metadata.get("contact", "unknown")}],
        "sample_count": 0, "first_sample_sec": None, "last_sample_sec": None,
        "maximum_trace_gap_sec": 0.0, "qpos_min_rad": None, "qpos_max_rad": None,
        "qpos_end_rad": None, "max_abs_velocity_fd_rad_s": [0.0] * 6,
        "max_abs_overlay_command_nm": [0.0] * 6,
        "max_abs_conditioned_minus_gravity_nm": [0.0] * 6,
        "gravity_command_min_nm": None, "gravity_command_max_nm": None,
        "limited_sample_count": 0, "phases": [], "issues": [], "hand_force_n": None,
    }


def _automatic_trials(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    trials = []
    previous_time = -float("inf")
    human = None
    closed = False
    for sequence, event in enumerate(events, start=1):
        if (event.get("schema") != "b601.joint-diagnostic-context.v1"
                or event.get("sequence") != sequence or closed):
            raise ValueError("diagnostic context schema, sequence or lifecycle mismatch")
        timestamp = finite_number(event.get("recorded_monotonic_sec"), "context time")
        if timestamp < previous_time:
            raise ValueError("context time must not go backwards")
        previous_time = timestamp
        kind = event.get("event")
        if kind == "declaration":
            if not isinstance(event.get("human_intervention"), bool):
                raise ValueError("human intervention declaration must be boolean")
            human = event["human_intervention"]
        elif kind == "context":
            if event.get("context_id") != len(trials) + 1 or event.get("human_intervention") is not human:
                raise ValueError("context identity or human declaration mismatch")
            if not isinstance(event.get("drive_enabled"), bool):
                raise ValueError("drive_enabled must be boolean")
            if not isinstance(event.get("feedback_fresh"), bool):
                raise ValueError("feedback_fresh must be boolean")
            snapshot = event.get("owner_snapshot")
            if not isinstance(snapshot, dict):
                raise ValueError("context is missing owner snapshot")
            expected_drive = bool(snapshot.get("operator_armed")) or snapshot.get("phase") == "armed"
            if event["drive_enabled"] != expected_drive:
                raise ValueError("drive declaration differs from recorded owner state")
            if trials:
                trials[-1].update(end_sec=timestamp, end_reason="context_change")
            trial = _new_trial(event["context_id"], snapshot,
                               {"kind": "joint_diagnostics", "human_intervention": human,
                                "drive_enabled": event["drive_enabled"]}, timestamp)
            if not event.get("feedback_fresh") or not snapshot.get("owner_ready") or snapshot.get("fault"):
                trial["issues"].append("owner_not_ready_or_feedback_stale")
            trials.append(trial)
        elif kind == "end" and trials and event.get("context_id") == len(trials):
            trials[-1].update(end_sec=timestamp, end_reason="owner_closed")
            closed = True
        else:
            raise ValueError("unexpected diagnostic context event")
    if not trials:
        raise ValueError("no owner feedback windows recorded")
    return trials


def load_trial_windows(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Read canonical annotation windows for offline report consumers."""
    iterator = iter(events)
    first = next(iterator, None)
    if first is None:
        raise ValueError("no labelled transparency trials recorded")
    events = chain((first,), iterator)
    if first.get("schema") == "b601.joint-diagnostic-context.v1":
        return _automatic_trials(events)
    trials: list[dict[str, Any]] = []
    active = None
    previous_time = -float("inf")
    for sequence, event in enumerate(events, start=1):
        if event.get("schema") != "b601.transparency-annotation.v1" or event.get("sequence") != sequence:
            raise ValueError("annotation schema or sequence mismatch")
        timestamp = finite_number(event.get("recorded_monotonic_sec"), "annotation time")
        if timestamp < previous_time:
            raise ValueError("annotation time must not go backwards")
        previous_time = timestamp
        kind = event.get("event")
        if kind == "begin":
            if active is not None or event.get("trial_id") != len(trials) + 1:
                raise ValueError("overlapping or out-of-order trial")
            snapshot, metadata = event.get("owner_snapshot"), event.get("metadata")
            if not isinstance(snapshot, dict) or not isinstance(metadata, dict):
                raise ValueError("trial is missing owner snapshot or metadata")
            metadata = validate_trial_metadata({**metadata, "action": "begin"})
            active = _new_trial(event["trial_id"], snapshot, metadata, timestamp)
            trials.append(active)
        elif active is None or event.get("trial_id") != active["trial_id"]:
            raise ValueError("annotation has no matching active trial")
        elif kind == "contact":
            if event.get("contact") not in CONTACT_STATES:
                raise ValueError("invalid recorded contact declaration")
            active["contacts"].append({"time_sec": timestamp, "contact": event["contact"]})
        elif kind in ("finish", "abort", "interrupted"):
            active["end_sec"] = timestamp
            active["end_reason"] = kind
            if kind == "finish":
                observation = event.get("observation")
                if not isinstance(observation, dict):
                    raise ValueError("finished trial is missing operator observation")
                active["observation"] = validate_observation(
                    {"action": "finish", **{key: observation[key] for key in OBSERVATION_FIELDS if key in observation}},
                    axis=active["metadata"]["axis"],
                )
            active = None
        else:
            raise ValueError(f"unknown annotation event: {kind}")
    if not trials:
        raise ValueError("no labelled transparency trials recorded")
    return trials


def _issue(trial: dict[str, Any], reason: str) -> None:
    if reason not in trial["issues"]:
        trial["issues"].append(reason)


def _sample(trial: dict[str, Any], row: dict[str, Any], timestamp: float) -> None:
    kind = row.get("kind")
    if kind != "active_dispatch":
        # Any protective-hold or guard sample inside the recording is not a
        # transparent interaction trial; retain the reason without hiding it.
        if kind == "active_stop" or "guard" in str(kind) or "hold" in str(kind):
            _issue(trial, f"owner_event:{kind}")
        return
    q = six_numbers(row.get("qpos_rad"), "qpos_rad")
    velocity = six_numbers(row.get("qvel_fd_rad_s"), "qvel_fd_rad_s")
    overlay = six_numbers(row.get("overlay_effective_tau_nm"), "overlay_effective_tau_nm")
    gravity = six_numbers(row.get("gravity_tau_nm"), "gravity_tau_nm")
    conditioned = six_numbers(row.get("conditioned_tau_nm"), "conditioned_tau_nm")
    console = row.get("torque_console")
    if not isinstance(console, dict):
        raise ValueError("active_dispatch is missing torque console audit")
    if console.get("joint_name") != trial["joint_name"]:
        _issue(trial, "selected_joint_changed")
    if console.get("fault"):
        _issue(trial, "console_fault")
    phase = console.get("phase")
    if not isinstance(phase, str) or not phase:
        raise ValueError("active_dispatch is missing console phase")
    if phase not in trial["phases"]:
        trial["phases"].append(phase)
    if trial["metadata"]["kind"] == "background_drag":
        if phase != "inactive" or max(abs(value) for value in overlay) > 1e-9:
            _issue(trial, "background_contaminated_by_assist_or_arming")
    trial["sample_count"] += 1
    if trial["first_sample_sec"] is None:
        trial["first_sample_sec"] = timestamp
        trial["qpos_min_rad"] = list(q)
        trial["qpos_max_rad"] = list(q)
        trial["gravity_command_min_nm"] = list(gravity)
        trial["gravity_command_max_nm"] = list(gravity)
        gap = timestamp - trial["start_sec"]
    else:
        gap = timestamp - trial["last_sample_sec"]
    trial["maximum_trace_gap_sec"] = max(trial["maximum_trace_gap_sec"], gap)
    trial["last_sample_sec"] = timestamp
    trial["qpos_end_rad"] = q
    for i in range(6):
        trial["qpos_min_rad"][i] = min(trial["qpos_min_rad"][i], q[i])
        trial["qpos_max_rad"][i] = max(trial["qpos_max_rad"][i], q[i])
        trial["max_abs_velocity_fd_rad_s"][i] = max(trial["max_abs_velocity_fd_rad_s"][i], abs(velocity[i]))
        trial["max_abs_overlay_command_nm"][i] = max(trial["max_abs_overlay_command_nm"][i], abs(overlay[i]))
        trial["max_abs_conditioned_minus_gravity_nm"][i] = max(
            trial["max_abs_conditioned_minus_gravity_nm"][i], abs(conditioned[i] - gravity[i]))
        trial["gravity_command_min_nm"][i] = min(trial["gravity_command_min_nm"][i], gravity[i])
        trial["gravity_command_max_nm"][i] = max(trial["gravity_command_max_nm"][i], gravity[i])
    if row.get("torque_limited") or row.get("slew_limited"):
        trial["limited_sample_count"] += 1


def _finish(trial: dict[str, Any], *, maximum_trace_gap_sec: float) -> None:
    automatic = trial["metadata"]["kind"] == "joint_diagnostics"
    if trial["end_reason"] not in (("context_change", "owner_closed") if automatic else ("finish",)):
        _issue(trial, trial["end_reason"])
    if not trial["sample_count"]:
        _issue(trial, "no_dispatch_samples")
    else:
        if trial["end_sec"] is not None:
            trial["maximum_trace_gap_sec"] = max(
                trial["maximum_trace_gap_sec"], trial["end_sec"] - trial["last_sample_sec"])
        if trial["maximum_trace_gap_sec"] > maximum_trace_gap_sec:
            _issue(trial, "incomplete_trace_coverage")
    contacts = {entry["contact"] for entry in trial["contacts"]}
    if (automatic and trial["metadata"]["human_intervention"] is None) or (not automatic and "unknown" in contacts):
        _issue(trial, "contact_not_declared")
    if len(contacts) != 1:
        _issue(trial, "contact_changed_split_trial_required")
    kind = trial["metadata"]["kind"]
    if kind == "unloaded_response" and contacts != {"none"}:
        _issue(trial, "not_confirmed_unloaded")
    if kind in ("background_drag", "cue_perception") and not contacts <= {"holding", "moving"}:
        _issue(trial, "human_interaction_not_declared")
    if kind == "cue_perception" and not any(trial["max_abs_overlay_command_nm"]):
        _issue(trial, "no_cue_command_recorded")
    if trial["limited_sample_count"]:
        _issue(trial, "command_conditioning_active")
    trial["record_usable_for_declared_trial"] = not trial["issues"]
    trial["force_measurement_status"] = (
        "manual_unsynchronised_observation"
        if trial["observation"] and trial["observation"].get("manual_force_n") is not None
        else "not_measured"
    )
    # This is motion evidence, not a torque-to-hand-force calibration.
    trial["delta_qpos_rad"] = (
        None if trial["qpos_end_rad"] is None else
        [end - start for end, start in zip(trial["qpos_end_rad"], trial["start_qpos_rad"], strict=True)])
    if not trial["sample_count"]:
        for key in ("max_abs_velocity_fd_rad_s", "max_abs_overlay_command_nm",
                    "max_abs_conditioned_minus_gravity_nm", "maximum_trace_gap_sec"):
            trial[key] = None
    if automatic:
        trial["joint_observations"] = _joint_observations(trial)


def _joint_observations(trial: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    human = trial["metadata"]["human_intervention"]
    for i in range(6):
        span = (None if not trial["sample_count"] else
                max(trial["qpos_max_rad"][i], trial["start_qpos_rad"][i]) -
                min(trial["qpos_min_rad"][i], trial["start_qpos_rad"][i]))
        overlay = None if not trial["sample_count"] else trial["max_abs_overlay_command_nm"][i]
        if span is None:
            response = "no_samples"
        elif human is None:
            response = "human_force_not_declared"
        elif human:
            response = "human_force_present"
        elif overlay > 0.001:
            response = "drive_with_motion" if span >= 0.001 else "drive_without_resolved_motion"
        else:
            response = "motion_without_overlay" if span >= 0.001 else "no_resolved_motion"
        delta = None if trial["delta_qpos_rad"] is None else trial["delta_qpos_rad"][i]
        result.append({"joint_name": f"joint{i + 1}", "motion_span_rad": span,
                       "net_displacement_rad": delta, "max_abs_overlay_command_nm": overlay,
                       "response": response, "friction_estimate_nm": None})
    return result


def _pairs(trials: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    pairs = []
    for i, first in enumerate(trials):
        for second in trials[i + 1:]:
            a, b = first["metadata"], second["metadata"]
            keys = ("pair_label", "condition_label", "kind", "axis")
            if any(a[key] != b[key] for key in keys) or a["direction"] == b["direction"]:
                continue
            reasons = []
            if not first["record_usable_for_declared_trial"] or not second["record_usable_for_declared_trial"]:
                reasons.append("trial_quality_issues")
            if first["joint_name"] != second["joint_name"]:
                reasons.append("different_selected_joint")
            pose_delta = max(abs(x - y) for x, y in zip(first["start_qpos_rad"], second["start_qpos_rad"], strict=True))
            if pose_delta > tolerance:
                reasons.append("different_full_arm_start_pose")
            if a["entry_direction"] == "unknown" or b["entry_direction"] == "unknown":
                reasons.append("entry_history_unknown")
            elif a["entry_direction"] != b["entry_direction"]:
                reasons.append("different_entry_history")
            if a["contact"] != b["contact"]:
                reasons.append("different_contact_state")
            pairs.append({"trial_ids": [first["trial_id"], second["trial_id"]],
                          "maximum_start_pose_difference_rad": pose_delta,
                          "comparable_recorded_conditions": not reasons, "issues": reasons,
                          "friction_estimate_nm": None,
                          "note": "Matched declarations are not proof of identical preload, temperature or physical force."})
    return pairs


def summarize_transparency(
    events: Iterable[dict[str, Any]],
    records: Iterable[dict[str, Any]],
    *,
    pair_pose_tolerance_rad: float = 0.03,
    maximum_trace_gap_sec: float = 0.15,
) -> dict[str, Any]:
    """Join label windows and dispatch evidence without rewriting either file."""
    tolerance = finite_number(pair_pose_tolerance_rad, "pair_pose_tolerance_rad")
    gap_limit = finite_number(maximum_trace_gap_sec, "maximum_trace_gap_sec")
    if tolerance <= 0 or gap_limit <= 0:
        raise ValueError("offline comparison tolerances must be positive")
    trials = load_trial_windows(events)
    automatic = trials[0]["metadata"]["kind"] == "joint_diagnostics"
    index = 0
    previous = -float("inf")
    for row in records:
        timestamp = finite_number(row.get("recorded_monotonic_sec"), "trace time")
        if timestamp < previous:
            raise ValueError("trace time must not go backwards")
        previous = timestamp
        while index < len(trials) and trials[index]["end_sec"] is not None and timestamp >= trials[index]["end_sec"]:
            index += 1
        if index == len(trials):
            break
        trial = trials[index]
        if timestamp >= trial["start_sec"]:
            _sample(trial, row, timestamp)
    for trial in trials:
        _finish(trial, maximum_trace_gap_sec=gap_limit)
    return {
        "schema": "b601.joint-diagnostics.v1" if automatic else "b601.transparency-assessment.v1",
        "semantics": {
            "torque": "software_commands_not_measured_motor_or_hand_force",
            "conditioned_minus_gravity": "command_difference_including_damping_and_conditioning_not_external_torque",
            "contact_and_perception": "operator_declared_not_detected",
            "manual_force": "optional_external_gauge_summary_not_a_synchronised_force_trace",
            "not_identified": ["static_friction", "hand_force_transfer_gain", "mechanical_transparency_bandwidth"],
            "response_criteria": {"motion_span_rad": 0.001, "nonzero_overlay_nm": 0.001},
        },
        "comparison_settings": {"pair_pose_tolerance_rad": tolerance, "maximum_trace_gap_sec": gap_limit},
        "trials": trials, "opposite_direction_pairs": [] if automatic else _pairs(trials, tolerance),
    }

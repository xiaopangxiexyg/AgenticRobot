"""Validate and compare dual-arm position baseline/admittance-shadow logs.

The analyzer is read-only.  It checks that both runs used the same 15/60/60/120
position chain and that the shadow observer was installed on the coupled
dual-arm pre-IK target path without changing any target.  The optional
single-live/single-Fix topology gate verifies that Fix was actually activated,
rather than merely selected in the GUI.  It also reports tracker, qref, IK, and
120 Hz stream statistics used to localize residual jitter.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SIDES = ("left", "right")
EXPECTED_POSITION_RATES = {
    "sample_hz": 15.0,
    "reference_hz": 60.0,
    "ik_hz": 60.0,
    "command_hz": 120.0,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _resolve_action_log(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        return resolved
    candidates = sorted(resolved.rglob("action_debug_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no action_debug_*.jsonl under {resolved}")
    return candidates[-1]


def _read_rows(path: Path) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _vector(value: Any, *, size: int) -> tuple[float, ...] | None:
    if isinstance(value, Mapping):
        value = value.get("deg")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != size or not all(math.isfinite(item) for item in result):
        return None
    return result


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return None
    position = (len(finite) - 1) * float(percentile) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return finite[lower]
    alpha = position - lower
    return finite[lower] * (1.0 - alpha) + finite[upper] * alpha


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "n": len(finite),
        "p50": _percentile(finite, 50.0),
        "p95": _percentile(finite, 95.0),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
    }


def _rotation_step_deg(previous: Sequence[Sequence[float]], current: Sequence[Sequence[float]]) -> float:
    # trace(R_prev^T R_current) without introducing a SciPy dependency.
    trace = sum(
        sum(float(previous[k][i]) * float(current[k][i]) for k in range(3))
        for i in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _position_rates(config: Mapping[str, Any]) -> dict[str, float | None]:
    raw = _mapping(_mapping(config.get("arm_execution_profile")).get("position_rates"))
    return {
        key: (None if raw.get(key) is None else float(raw[key]))
        for key in EXPECTED_POSITION_RATES
    }


def _control_fingerprint(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "arm_execution_profile": config.get("arm_execution_profile"),
        "position_control_profile": config.get("position_control_profile"),
        "position_ik_step_mode": config.get("position_ik_step_mode"),
        "position_joint_velocity_limit_profile": config.get(
            "position_joint_velocity_limit_profile"
        ),
        "position_post_ik_qpos_smoothing_alpha_effective": config.get(
            "position_post_ik_qpos_smoothing_alpha_effective"
        ),
        "arm_follow_qpos_alpha_effective": config.get(
            "arm_follow_qpos_alpha_effective"
        ),
        "arm_max_angvel_effective_rad_s": config.get(
            "arm_max_angvel_effective_rad_s"
        ),
        "canfd_trajectory_mode_effective": config.get(
            "canfd_trajectory_mode_effective"
        ),
        "canfd_radio_effective": config.get("canfd_radio_effective"),
    }


def _add_check(
    checks: list[dict[str, Any]], name: str, passed: bool, detail: Any
) -> None:
    checks.append({"name": name, "passed": bool(passed), "detail": detail})


def _tracker_metrics(frames: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    channel_name = f"{side}_wrist_pose"
    complete = 0
    with_window = 0
    translation_steps: list[float] = []
    rotation_steps: list[float] = []
    previous_xyz: tuple[float, ...] | None = None
    previous_rotation: Sequence[Sequence[float]] | None = None
    for frame in frames:
        channel = _mapping(_mapping(frame.get("tracker_channels")).get(channel_name))
        pose = _mapping(channel.get("pose"))
        xyz = _vector(pose.get("xyz_m"), size=3)
        rotation = pose.get("rotation_matrix")
        rotation_valid = (
            isinstance(rotation, Sequence)
            and len(rotation) == 3
            and all(_vector(row, size=3) is not None for row in rotation)
        )
        exact_stage = channel.get("pose_stage") == "observation_channel_before_retarget"
        if xyz is not None and rotation_valid and exact_stage:
            complete += 1
            if previous_xyz is not None:
                translation_steps.append(
                    math.sqrt(sum((xyz[i] - previous_xyz[i]) ** 2 for i in range(3)))
                )
                assert previous_rotation is not None
                rotation_steps.append(_rotation_step_deg(previous_rotation, rotation))
            previous_xyz = xyz
            previous_rotation = rotation
        if isinstance(channel.get("motion_window"), Mapping):
            with_window += 1
    count = len(frames)
    return {
        "complete_pose_count": complete,
        "complete_pose_coverage": complete / count if count else 0.0,
        "motion_window_count": with_window,
        "motion_window_coverage": with_window / count if count else 0.0,
        "translation_step_m": _stats(translation_steps),
        "rotation_step_deg": _stats(rotation_steps),
    }


def _qref_metrics(frames: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    steps: list[float] = []
    fixed_steps: list[float] = []
    fixed_frames = 0
    previous: tuple[float, ...] | None = None
    previous_fixed = False
    for frame in frames:
        command = _mapping(_mapping(frame.get("command_by_side")).get(side))
        qpos = _vector(command.get("value_qpos"), size=6)
        fix_state = _mapping(frame.get("fix_arm"))
        fixed = bool(fix_state.get("active")) and fix_state.get("fixed_side") == side
        if fixed:
            fixed_frames += 1
        if qpos is not None and previous is not None:
            step = max(abs(qpos[index] - previous[index]) for index in range(6))
            steps.append(step)
            if fixed and previous_fixed:
                fixed_steps.append(step)
        if qpos is not None:
            previous = qpos
            previous_fixed = fixed
    return {
        "fixed_frame_count": fixed_frames,
        "max_joint_step_deg": _stats(steps),
        "fixed_max_joint_step_deg": _stats(fixed_steps),
    }


def _stream_metrics(frames: Sequence[Mapping[str, Any]], side: str) -> dict[str, Any]:
    maxima = {key: 0 for key in ("failed_send_count", "deadline_miss_count", "stale_count")}
    fault_frames = 0
    unhealthy_frames = 0
    health_reasons: Counter[str] = Counter()
    for frame in frames:
        stream = _mapping(
            _mapping(_mapping(frame.get("jitter_focus_by_side")).get(side)).get(
                "position_background_streaming"
            )
        )
        for key in maxima:
            value = stream.get(key)
            if value is not None:
                maxima[key] = max(maxima[key], int(value))
        if stream.get("fault") not in (None, "", False):
            fault_frames += 1
        if stream.get("healthy") is False:
            unhealthy_frames += 1
            raw_reasons = stream.get("health_reasons", ()) or ()
            if isinstance(raw_reasons, Sequence) and not isinstance(
                raw_reasons, (str, bytes)
            ):
                health_reasons.update(str(reason) for reason in raw_reasons)
    return {
        **maxima,
        "fault_frame_count": fault_frames,
        "unhealthy_frame_count": unhealthy_frames,
        "health_reason_counts": dict(health_reasons),
    }


def _norm(value: Any, *, size: int = 3) -> float | None:
    vector = _vector(value, size=size)
    if vector is None:
        return None
    return math.sqrt(sum(item * item for item in vector))


def _admittance_side_metrics(
    frames: Sequence[Mapping[str, Any]], side: str
) -> dict[str, Any]:
    sample_age_ms: list[float] = []
    integration_dt_ms: list[float] = []
    tared_force_n: list[float] = []
    filtered_force_n: list[float] = []
    deadbanded_force_n: list[float] = []
    requested_offset_m: list[float] = []
    output_offset_m: list[float] = []
    velocity_m_s: list[float] = []
    direction_cosine: list[float] = []
    activity_reasons: Counter[str] = Counter()
    state_reasons: Counter[str] = Counter()
    sample_state_reasons: Counter[str] = Counter()
    active_count = 0
    contact_frame_count = 0
    new_sample_count = 0
    frozen_count = 0
    envelope_violation_count = 0
    for frame in frames:
        payload = _mapping(
            _mapping(_mapping(frame.get("position_admittance")).get("sides")).get(
                side
            )
        )
        if not payload:
            continue
        activity_reasons[str(payload.get("activity_reason") or "unknown")] += 1
        state_reasons[str(payload.get("state_reason") or "unknown")] += 1
        sample_state_reasons[
            str(payload.get("sample_state_reason") or "unknown")
        ] += 1
        active_count += int(bool(payload.get("active")))
        new_sample_count += int(bool(payload.get("new_sample_integrated")))
        frozen_count += int(bool(payload.get("frozen")))
        if payload.get("sample_age_sec") is not None:
            sample_age_ms.append(float(payload["sample_age_sec"]) * 1000.0)
        if payload.get("integration_dt_sec") is not None:
            integration_dt_ms.append(float(payload["integration_dt_sec"]) * 1000.0)
        snapshot_config = _mapping(frame.get("position_admittance")).get("config")
        snapshot_config = _mapping(snapshot_config)
        maximum_offset = _vector(snapshot_config.get("max_offset_m"), size=3)
        maximum_velocity = _vector(snapshot_config.get("max_velocity_m_s"), size=3)
        enabled_axes = snapshot_config.get("enabled_axes", (True, True, True))
        enabled_indexes = [
            index
            for index, enabled in enumerate(enabled_axes)
            if bool(enabled)
        ]
        requested_vector = _vector(payload.get("requested_offset_m"), size=3)
        output_vector = _vector(payload.get("output_offset_m"), size=3)
        velocity_vector = _vector(payload.get("velocity_m_s"), size=3)
        if enabled_indexes and maximum_offset is not None:
            norm_limit = max(maximum_offset[index] for index in enabled_indexes)
            for vector in (requested_vector, output_vector):
                if vector is not None and math.sqrt(
                    sum(vector[index] ** 2 for index in enabled_indexes)
                ) > norm_limit + 1e-9:
                    envelope_violation_count += 1
        if enabled_indexes and maximum_velocity is not None and velocity_vector is not None:
            velocity_limit = max(maximum_velocity[index] for index in enabled_indexes)
            if math.sqrt(
                sum(velocity_vector[index] ** 2 for index in enabled_indexes)
            ) > velocity_limit + 1e-9:
                envelope_violation_count += 1
        for key, destination in (
            ("filtered_force_control_n", filtered_force_n),
            ("deadbanded_force_control_n", deadbanded_force_n),
            ("requested_offset_m", requested_offset_m),
            ("output_offset_m", output_offset_m),
            ("velocity_m_s", velocity_m_s),
        ):
            norm = _norm(payload.get(key))
            if norm is not None:
                destination.append(norm)
        raw_tared = _vector(payload.get("tared_wrench_n_nm"), size=6)
        if raw_tared is not None:
            tared_force_n.append(
                math.sqrt(sum(raw_tared[index] ** 2 for index in range(3)))
            )
        force = _vector(payload.get("deadbanded_force_control_n"), size=3)
        offset = _vector(payload.get("output_offset_m"), size=3)
        if force is not None and offset is not None:
            force_norm = math.sqrt(sum(item * item for item in force))
            offset_norm = math.sqrt(sum(item * item for item in offset))
            if force_norm > 1e-12:
                contact_frame_count += 1
            if force_norm > 1e-12 and offset_norm > 1e-12:
                direction_cosine.append(
                    sum(force[index] * offset[index] for index in range(3))
                    / (force_norm * offset_norm)
                )
    return {
        "active_frame_count": active_count,
        "contact_frame_count": contact_frame_count,
        "new_sample_integrated_count": new_sample_count,
        "frozen_frame_count": frozen_count,
        "envelope_violation_count": envelope_violation_count,
        "activity_reason_counts": dict(activity_reasons),
        "state_reason_counts": dict(state_reasons),
        "sample_state_reason_counts": dict(sample_state_reasons),
        "sample_age_ms": _stats(sample_age_ms),
        "integration_dt_ms": _stats(integration_dt_ms),
        "tared_force_norm_n": _stats(tared_force_n),
        "filtered_force_norm_n": _stats(filtered_force_n),
        "deadbanded_force_norm_n": _stats(deadbanded_force_n),
        "requested_offset_norm_m": _stats(requested_offset_m),
        "output_offset_norm_m": _stats(output_offset_m),
        "velocity_norm_m_s": _stats(velocity_m_s),
        "force_output_direction_cosine": _stats(direction_cosine),
    }


def _shadow_metrics(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    semantic_failures = 0
    last_errors = 0
    max_applied_offset = {side: 0.0 for side in SIDES}
    present = 0
    for frame in frames:
        snapshot = _mapping(frame.get("position_admittance"))
        if not snapshot:
            continue
        present += 1
        if not (
            snapshot.get("mode") == "shadow"
            and snapshot.get("shadow_only") is True
            and snapshot.get("applied_to_target") is False
            and snapshot.get("target_identity_preserved") is True
        ):
            semantic_failures += 1
        if snapshot.get("last_error") not in (None, ""):
            last_errors += 1
        for side in SIDES:
            offset = _vector(_mapping(_mapping(snapshot.get("sides")).get(side)).get("applied_offset_m"), size=3)
            if offset is None:
                semantic_failures += 1
                continue
            max_applied_offset[side] = max(
                max_applied_offset[side], max(abs(value) for value in offset)
            )
    return {
        "present_frame_count": present,
        "semantic_failure_count": semantic_failures,
        "last_error_frame_count": last_errors,
        "max_abs_applied_offset_m": max_applied_offset,
    }


def _shadow_topology_metrics(
    frames: Sequence[Mapping[str, Any]],
    *,
    live_side: str,
    fixed_side: str,
) -> dict[str, Any]:
    """Describe one-live/one-Fix semantics on frames where Fix is active."""

    expected_fix_frames = 0
    unexpected_fix_frames = 0
    fix_available_frames = 0
    expected_selected_side_frames = 0
    live_active_frames = 0
    live_reason_frames = 0
    fixed_inactive_frames = 0
    fixed_frozen_frames = 0
    fixed_reason_frames = 0
    for frame in frames:
        fix_state = _mapping(frame.get("fix_arm"))
        fix_available_frames += int(bool(fix_state.get("available")))
        expected_selected_side_frames += int(
            fix_state.get("selected_side") == fixed_side
        )
        fix_active = bool(fix_state.get("active"))
        actual_fixed_side = str(fix_state.get("fixed_side") or "")
        expected_fix_active = fix_active and actual_fixed_side == fixed_side
        if expected_fix_active:
            expected_fix_frames += 1
        elif fix_active:
            unexpected_fix_frames += 1
        if not expected_fix_active:
            continue
        sides = _mapping(_mapping(frame.get("position_admittance")).get("sides"))
        live = _mapping(sides.get(live_side))
        fixed = _mapping(sides.get(fixed_side))
        live_active_frames += int(bool(live.get("active")))
        live_reason_frames += int(live.get("activity_reason") == "running_live_side")
        fixed_inactive_frames += int(not bool(fixed.get("active")))
        fixed_frozen_frames += int(bool(fixed.get("frozen")))
        fixed_reason_frames += int(fixed.get("activity_reason") == "fixed_side")
    return {
        "expected_live_side": live_side,
        "expected_fixed_side": fixed_side,
        "frame_count": len(frames),
        "fix_available_frame_count": fix_available_frames,
        "expected_selected_side_frame_count": expected_selected_side_frames,
        "expected_fix_active_frame_count": expected_fix_frames,
        "unexpected_fix_active_frame_count": unexpected_fix_frames,
        "live_active_frame_count_during_fix": live_active_frames,
        "live_reason_frame_count_during_fix": live_reason_frames,
        "fixed_inactive_frame_count_during_fix": fixed_inactive_frames,
        "fixed_frozen_frame_count_during_fix": fixed_frozen_frames,
        "fixed_reason_frame_count_during_fix": fixed_reason_frames,
    }


def _summarize(
    path: Path,
    *,
    expect_shadow: bool,
    min_frames: int,
    min_fixed_frames_per_side: int,
    min_tracker_pose_coverage: float,
    expected_position_profile: str | None,
    expected_shadow_live_side: str | None,
    expected_shadow_fixed_side: str | None,
) -> dict[str, Any]:
    log_path = _resolve_action_log(path)
    rows = _read_rows(log_path)
    configs = [_mapping(row.get("payload")) for row in rows if row.get("event") == "config"]
    frames = [row for row in rows if row.get("event") == "frame"]
    receipts = [
        _mapping(row.get("payload"))
        for row in rows
        if row.get("event") == "position_admittance_config"
    ]
    disabled = [row for row in rows if row.get("event") == "position_admittance_disabled"]
    config = configs[-1] if configs else {}
    rates = _position_rates(config)
    control_fingerprint = _control_fingerprint(config)
    trackers = {side: _tracker_metrics(frames, side) for side in SIDES}
    qref = {side: _qref_metrics(frames, side) for side in SIDES}
    streams = {side: _stream_metrics(frames, side) for side in SIDES}
    shadow = _shadow_metrics(frames)
    admittance = {
        side: _admittance_side_metrics(frames, side) for side in SIDES
    }
    topology = (
        None
        if not expect_shadow or expected_shadow_live_side is None
        else _shadow_topology_metrics(
            frames,
            live_side=expected_shadow_live_side,
            fixed_side=str(expected_shadow_fixed_side),
        )
    )
    fault_count = sum(len(frame.get("faults", ()) or ()) for frame in frames)
    ik_budget_exceeded = sum(
        bool(_mapping(frame.get("ik_timing")).get("time_budget_exceeded"))
        for frame in frames
    )
    checks: list[dict[str, Any]] = []
    _add_check(checks, "minimum_frames", len(frames) >= min_frames, len(frames))
    _add_check(checks, "dual_arm_frames", all(set(frame.get("active_sides", ())) == set(SIDES) for frame in frames), len(frames))
    _add_check(checks, "position_rates_15_60_60_120", rates == EXPECTED_POSITION_RATES, rates)
    profile_name = _mapping(config.get("position_control_profile")).get("name")
    if expected_position_profile is not None:
        _add_check(
            checks,
            "expected_position_profile",
            profile_name == expected_position_profile,
            profile_name,
        )
    _add_check(checks, "no_frame_errors", not any(row.get("event") == "frame_error" for row in rows), None)
    _add_check(checks, "no_runtime_faults", fault_count == 0, fault_count)
    for side in SIDES:
        _add_check(
            checks,
            f"{side}_tracker_pose_coverage",
            trackers[side]["complete_pose_coverage"] >= min_tracker_pose_coverage,
            trackers[side]["complete_pose_coverage"],
        )
        if expected_shadow_live_side is None:
            _add_check(
                checks,
                f"{side}_fixed_frames",
                qref[side]["fixed_frame_count"] >= min_fixed_frames_per_side,
                qref[side]["fixed_frame_count"],
            )
        _add_check(
            checks,
            f"{side}_stream_no_failed_send_or_fault",
            streams[side]["failed_send_count"] == 0 and streams[side]["fault_frame_count"] == 0,
            streams[side],
        )
    if expect_shadow:
        receipt = receipts[-1] if receipts else {}
        _add_check(
            checks,
            "shadow_startup_receipt",
            bool(receipt.get("enabled"))
            and receipt.get("requested_mode") == "shadow"
            and set(receipt.get("active_sides", ())) == set(SIDES)
            and receipt.get("coupled_pre_ik_target_filter_available") is True
            and not disabled,
            receipt or {"missing": True, "disabled_event_count": len(disabled)},
        )
        _add_check(
            checks,
            "shadow_present_on_every_frame",
            shadow["present_frame_count"] == len(frames),
            shadow["present_frame_count"],
        )
        _add_check(
            checks,
            "shadow_never_changes_target",
            shadow["semantic_failure_count"] == 0
            and shadow["last_error_frame_count"] == 0
            and all(value == 0.0 for value in shadow["max_abs_applied_offset_m"].values()),
            shadow,
        )
        if topology is not None:
            expected_fix_frames = topology["expected_fix_active_frame_count"]
            minimum_required_fix_frames = max(1, min_fixed_frames_per_side)
            topology["minimum_required_fix_frames"] = minimum_required_fix_frames
            _add_check(
                checks,
                "expected_single_arm_shadow_topology",
                topology["fix_available_frame_count"] == len(frames)
                and topology["expected_selected_side_frame_count"] == len(frames)
                and expected_fix_frames >= minimum_required_fix_frames
                and topology["unexpected_fix_active_frame_count"] == 0
                and topology["live_active_frame_count_during_fix"]
                == expected_fix_frames
                and topology["live_reason_frame_count_during_fix"]
                == expected_fix_frames
                and topology["fixed_inactive_frame_count_during_fix"]
                == expected_fix_frames
                and topology["fixed_frozen_frame_count_during_fix"]
                == expected_fix_frames
                and topology["fixed_reason_frame_count_during_fix"]
                == expected_fix_frames,
                topology,
            )
            fixed_qref = qref[str(expected_shadow_fixed_side)]
            fixed_step_max = fixed_qref["fixed_max_joint_step_deg"]["max"]
            _add_check(
                checks,
                "expected_fixed_side_qref_held",
                fixed_step_max is not None and fixed_step_max <= 1e-6,
                fixed_qref["fixed_max_joint_step_deg"],
            )
        force_response_sides = (
            SIDES
            if expected_shadow_live_side is None
            else (expected_shadow_live_side,)
        )
        for side in force_response_sides:
            side_metrics = admittance[side]
            _add_check(
                checks,
                f"{side}_shadow_force_response_observed",
                side_metrics["new_sample_integrated_count"] > 0
                and side_metrics["contact_frame_count"] > 0,
                {
                    "new_sample_integrated_count": side_metrics[
                        "new_sample_integrated_count"
                    ],
                    "contact_frame_count": side_metrics["contact_frame_count"],
                },
            )
            _add_check(
                checks,
                f"{side}_shadow_fresh_and_unfrozen",
                side_metrics["frozen_frame_count"] == 0
                and (
                    side_metrics["sample_age_ms"]["max"] is not None
                    and side_metrics["sample_age_ms"]["max"] <= 150.0
                ),
                {
                    "frozen_frame_count": side_metrics["frozen_frame_count"],
                    "sample_age_ms": side_metrics["sample_age_ms"],
                },
            )
            _add_check(
                checks,
                f"{side}_shadow_envelope",
                side_metrics["envelope_violation_count"] == 0,
                side_metrics["envelope_violation_count"],
            )
        for side in SIDES:
            if side in force_response_sides:
                continue
            side_metrics = admittance[side]
            _add_check(
                checks,
                f"{side}_shadow_envelope",
                side_metrics["envelope_violation_count"] == 0,
                side_metrics["envelope_violation_count"],
            )
    else:
        _add_check(
            checks,
            "baseline_has_no_position_admittance",
            not receipts and not disabled and shadow["present_frame_count"] == 0,
            {
                "config_receipt_count": len(receipts),
                "disabled_event_count": len(disabled),
                "frame_snapshot_count": shadow["present_frame_count"],
            },
        )
    return {
        "path": str(log_path),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "config": dict(config),
        "position_rates": rates,
        "control_fingerprint": control_fingerprint,
        "frame_count": len(frames),
        "fault_count": fault_count,
        "ik_time_budget_exceeded_count": ik_budget_exceeded,
        "ik_time_budget_exceeded_ratio": ik_budget_exceeded / len(frames) if frames else None,
        "tracker": trackers,
        "qref": qref,
        "position_stream": streams,
        "shadow": shadow,
        "admittance": admittance,
        "expected_shadow_topology": topology,
    }


def analyze_pair(
    baseline: Path,
    shadow: Path,
    *,
    min_frames: int = 30,
    min_fixed_frames_per_side: int = 10,
    min_tracker_pose_coverage: float = 0.9,
    expected_position_profile: str | None = None,
    expected_shadow_live_side: str | None = None,
    expected_shadow_fixed_side: str | None = None,
) -> dict[str, Any]:
    if min_fixed_frames_per_side < 0:
        raise ValueError("minimum fixed frames per side must be non-negative")
    if (expected_shadow_live_side is None) != (expected_shadow_fixed_side is None):
        raise ValueError(
            "expected shadow live/fixed sides must be provided together"
        )
    if expected_shadow_live_side is not None:
        if (
            expected_shadow_live_side not in SIDES
            or expected_shadow_fixed_side not in SIDES
        ):
            raise ValueError("expected shadow live/fixed sides must be left or right")
        if expected_shadow_live_side == expected_shadow_fixed_side:
            raise ValueError("expected shadow live/fixed sides must be different")
    baseline_report = _summarize(
        baseline,
        expect_shadow=False,
        min_frames=min_frames,
        min_fixed_frames_per_side=min_fixed_frames_per_side,
        min_tracker_pose_coverage=min_tracker_pose_coverage,
        expected_position_profile=expected_position_profile,
        expected_shadow_live_side=expected_shadow_live_side,
        expected_shadow_fixed_side=expected_shadow_fixed_side,
    )
    shadow_report = _summarize(
        shadow,
        expect_shadow=True,
        min_frames=min_frames,
        min_fixed_frames_per_side=min_fixed_frames_per_side,
        min_tracker_pose_coverage=min_tracker_pose_coverage,
        expected_position_profile=expected_position_profile,
        expected_shadow_live_side=expected_shadow_live_side,
        expected_shadow_fixed_side=expected_shadow_fixed_side,
    )
    rates_match = baseline_report["position_rates"] == shadow_report["position_rates"]
    control_match = (
        baseline_report["control_fingerprint"]
        == shadow_report["control_fingerprint"]
    )
    return {
        "schema": "dexteleop.position_admittance_shadow_ab/v1",
        "passed": bool(
            baseline_report["passed"]
            and shadow_report["passed"]
            and rates_match
            and control_match
        ),
        "pair_checks": [
            {
                "name": "matching_position_rates",
                "passed": rates_match,
                "detail": {
                    "baseline": baseline_report["position_rates"],
                    "shadow": shadow_report["position_rates"],
                },
            },
            {
                "name": "matching_control_fingerprint",
                "passed": control_match,
                "detail": {
                    "baseline": baseline_report["control_fingerprint"],
                    "shadow": shadow_report["control_fingerprint"],
                },
            },
        ],
        "baseline": baseline_report,
        "shadow": shadow_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--shadow", type=Path, required=True)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-fixed-frames-per-side", type=int, default=10)
    parser.add_argument("--min-tracker-pose-coverage", type=float, default=0.9)
    parser.add_argument(
        "--expected-position-profile",
        default="rm65_bimanual_position_v2_balanced",
    )
    parser.add_argument("--expected-shadow-live-side", choices=SIDES)
    parser.add_argument("--expected-shadow-fixed-side", choices=SIDES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze_pair(
        args.baseline,
        args.shadow,
        min_frames=args.min_frames,
        min_fixed_frames_per_side=args.min_fixed_frames_per_side,
        min_tracker_pose_coverage=args.min_tracker_pose_coverage,
        expected_position_profile=args.expected_position_profile,
        expected_shadow_live_side=args.expected_shadow_live_side,
        expected_shadow_fixed_side=args.expected_shadow_fixed_side,
    )
    encoded = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is not None:
        args.output.expanduser().write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

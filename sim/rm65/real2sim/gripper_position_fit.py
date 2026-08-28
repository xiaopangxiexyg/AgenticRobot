"""Fit a free-motion DM-Tac position actuator from identification traces.

This module is simulation-owned and offline-only. It reads evidence produced by
``hardware/DMTAC_gripper/examples/identify_gripper_actuator.py`` and fits a
causal delayed, acceleration/rate-limited normalized position response. It
does not import teleoperation runtime or initialize hardware.

The resulting artifact makes no contact-force claim. Current, force setting,
and continuous-grasp mode are retained only as source-condition evidence.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from sim.rm65.actuation.gripper import FIT_SCHEMA, MODEL_TYPE

@dataclass(frozen=True)
class MotionSegment:
    source_trace: str
    side: str
    trial_id: str
    segment_index: int | None
    stage: str
    target_position: float
    initial_position: float
    time_sec: np.ndarray
    measured_position: np.ndarray
    command_duration_sec: float
    force_setting: float | None
    speed_setting: float | None
    continuous_grasp: bool | None
    command_mode: str | None
    trace_schema: str

    @property
    def direction(self) -> str:
        return "close" if self.target_position > self.initial_position else "open"

    @property
    def commanded_distance(self) -> float:
        return abs(self.target_position - self.initial_position)

    @property
    def observed_distance(self) -> float:
        return abs(float(self.measured_position[-1]) - self.initial_position)

    @property
    def key(self) -> str:
        segment = "none" if self.segment_index is None else str(self.segment_index)
        return f"{self.side}:{self.trial_id}:{self.stage}:{segment}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _trace_and_metadata_path(source: str | Path) -> tuple[Path, Path]:
    path = Path(source).expanduser().resolve()
    trace_path = path / "gripper_trace.jsonl" if path.is_dir() else path
    metadata_path = trace_path.parent / "metadata.json"
    if not trace_path.is_file():
        raise FileNotFoundError(f"missing gripper trace: {trace_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing gripper metadata: {metadata_path}")
    return trace_path, metadata_path


def _row_monotonic_ns(row: Mapping[str, Any], *, state: bool = False) -> int:
    if state and row.get("read_midpoint_monotonic_ns") is not None:
        return int(row["read_midpoint_monotonic_ns"])
    if row.get("command_end_monotonic_ns") is not None:
        return int(row["command_end_monotonic_ns"])
    return int(row["monotonic_time_ns"])


def _trial_conditions(metadata: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in metadata.get("trials", ()):
        if isinstance(raw, Mapping) and raw.get("trial_id"):
            result[str(raw["trial_id"])] = dict(raw)
    return result


def _stage_pair(command_stage: str) -> tuple[str, str] | None:
    if command_stage == "motion_initial":
        return "motion_and_hold", "motion"
    if command_stage == "close_initial":
        return "close_and_hold", "motion"
    if command_stage == "release_open":
        return "release_open", "release"
    return None


def _trim_after_settle(
    times: np.ndarray,
    positions: np.ndarray,
    *,
    target: float,
) -> tuple[np.ndarray, np.ndarray]:
    tolerance = max(0.004, 0.01 * abs(float(target) - float(positions[0])))
    settled_index: int | None = None
    for index in range(max(0, len(positions) - 2)):
        window = positions[index : index + 3]
        if len(window) == 3 and np.all(np.abs(window - float(target)) <= tolerance):
            settled_index = min(len(positions), index + 8)
            break
    if settled_index is not None:
        times = times[:settled_index]
        positions = positions[:settled_index]
    within_window = times <= min(3.0, float(times[0]) + 2.0)
    return times[within_window], positions[within_window]


def load_motion_segments(source: str | Path) -> tuple[list[MotionSegment], dict[str, Any]]:
    """Load usable free-motion segments from trace schema v1 or v2."""

    trace_path, metadata_path = _trace_and_metadata_path(source)
    metadata = _load_object(metadata_path)
    endpoint = metadata.get("endpoint")
    if not isinstance(endpoint, Mapping) or str(endpoint.get("side", "")) not in {
        "left",
        "right",
    }:
        raise ValueError(f"{metadata_path}: missing left/right endpoint side")
    side = str(endpoint["side"])
    trial_conditions = _trial_conditions(metadata)
    rows: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{trace_path}:{line_number}: expected JSON object")
            rows.append(row)

    states = [row for row in rows if row.get("event") == "state"]
    command_begins = [row for row in rows if row.get("event") == "command_begin"]
    segments: list[MotionSegment] = []
    rejected: list[dict[str, Any]] = []
    for command in rows:
        if command.get("event") != "command_end":
            continue
        stage_pair = _stage_pair(str(command.get("stage", "")))
        if stage_pair is None:
            continue
        state_stage, stage_kind = stage_pair
        trial_id = str(command.get("trial_id", ""))
        segment_index = command.get("segment_index")
        segment_index = None if segment_index is None else int(segment_index)
        target = float(command["target_position"])
        command_end_ns = _row_monotonic_ns(command)
        command_begin_ns = command.get("command_begin_monotonic_ns")
        if command_begin_ns is None:
            matching_begins = [
                row
                for row in command_begins
                if str(row.get("trial_id", "")) == trial_id
                and str(row.get("stage", "")) == str(command.get("stage", ""))
                and (
                    segment_index is None
                    or row.get("segment_index") is not None
                    and int(row["segment_index"]) == segment_index
                )
                and _row_monotonic_ns(row) <= command_end_ns
            ]
            if matching_begins:
                command_begin_ns = _row_monotonic_ns(
                    max(matching_begins, key=_row_monotonic_ns)
                )
        command_begin_ns = (
            command_end_ns if command_begin_ns is None else int(command_begin_ns)
        )
        command_duration_sec = (command_end_ns - command_begin_ns) / 1e9
        if command_duration_sec < 0.0 or command_duration_sec > 1.0:
            raise ValueError(
                f"{trace_path}: invalid command duration for {trial_id}/{segment_index}"
            )
        matched: list[dict[str, Any]] = []
        for state_row in states:
            if str(state_row.get("trial_id", "")) != trial_id:
                continue
            if str(state_row.get("stage", "")) != state_stage:
                continue
            state_segment = state_row.get("segment_index")
            state_segment = None if state_segment is None else int(state_segment)
            if stage_kind == "motion" and segment_index != state_segment:
                continue
            if _row_monotonic_ns(state_row, state=True) < command_end_ns:
                continue
            matched.append(state_row)
        if len(matched) < 8:
            rejected.append(
                {
                    "trial_id": trial_id,
                    "segment_index": segment_index,
                    "stage": str(command.get("stage")),
                    "reason": "fewer_than_8_state_samples",
                }
            )
            continue
        matched.sort(key=lambda row: _row_monotonic_ns(row, state=True))
        times = np.asarray(
            [
                (_row_monotonic_ns(row, state=True) - command_begin_ns) / 1e9
                for row in matched
            ],
            dtype=float,
        )
        positions = np.asarray(
            [float(dict(row["state"])["joint_pos"]) for row in matched],
            dtype=float,
        )
        finite = np.isfinite(times) & np.isfinite(positions)
        times = times[finite]
        positions = positions[finite]
        if len(times) < 8 or np.any(np.diff(times) <= 0.0):
            rejected.append(
                {
                    "trial_id": trial_id,
                    "segment_index": segment_index,
                    "stage": str(command.get("stage")),
                    "reason": "invalid_or_non_monotonic_timestamps",
                }
            )
            continue
        times, positions = _trim_after_settle(times, positions, target=target)
        initial = float(positions[0])
        distance = abs(target - initial)
        observed = abs(float(positions[-1]) - initial)
        signed_steps = math.copysign(1.0, target - initial) * np.diff(positions)
        reverse_fraction = float(np.mean(signed_steps < -0.01)) if len(signed_steps) else 1.0
        reason: str | None = None
        if distance < 0.02:
            reason = "commanded_distance_below_0.02"
        elif observed < min(0.02, 0.4 * distance):
            reason = "insufficient_observed_motion"
        elif reverse_fraction > 0.15:
            reason = "non_monotonic_or_preempted_response"
        elif np.any((positions < -0.03) | (positions > 1.03)):
            reason = "position_out_of_tolerance"
        if reason is not None:
            rejected.append(
                {
                    "trial_id": trial_id,
                    "segment_index": segment_index,
                    "stage": str(command.get("stage")),
                    "reason": reason,
                }
            )
            continue
        conditions = trial_conditions.get(trial_id, {})
        force = conditions.get("force", command.get("force"))
        continuous = conditions.get(
            "continuous_grasp",
            command.get("continuous_grasp"),
        )
        speed = metadata.get("speed_fixed", command.get("speed"))
        metadata_args = metadata.get("args")
        command_mode = command.get("command_mode")
        if command_mode is None and isinstance(metadata_args, Mapping):
            command_mode = metadata_args.get("command_mode")
        segments.append(
            MotionSegment(
                source_trace=str(trace_path),
                side=side,
                trial_id=trial_id,
                segment_index=segment_index,
                stage=str(command.get("stage")),
                target_position=target,
                initial_position=initial,
                time_sec=times,
                measured_position=positions,
                command_duration_sec=float(command_duration_sec),
                force_setting=None if force is None else float(force),
                speed_setting=None if speed is None else float(speed),
                continuous_grasp=None if continuous is None else bool(continuous),
                command_mode=None if command_mode is None else str(command_mode),
                trace_schema=str(command.get("schema", metadata.get("schema", "unknown"))),
            )
        )
    source_evidence = {
        "trace_path": str(trace_path),
        "trace_sha256": _sha256_file(trace_path),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "side": side,
        "trace_schema": str(metadata.get("schema", "unknown")),
        "loaded_segment_count": len(segments),
        "rejected_segments": rejected,
    }
    return segments, source_evidence


def predict_step_response(
    time_sec: Sequence[float] | np.ndarray,
    *,
    initial_position: float,
    target_position: float,
    delay_sec: float,
    max_rate_per_sec: float,
    acceleration_per_sec2: float,
) -> np.ndarray:
    """Analytic zero-initial-rate trapezoidal/triangular position response."""

    times = np.asarray(time_sec, dtype=float)
    distance = abs(float(target_position) - float(initial_position))
    if distance <= 1e-12:
        return np.full_like(times, float(target_position), dtype=float)
    delay = max(0.0, float(delay_sec))
    max_rate = max(1e-9, float(max_rate_per_sec))
    acceleration = max(1e-9, float(acceleration_per_sec2))
    elapsed = np.maximum(times - delay, 0.0)
    acceleration_distance = 0.5 * max_rate * max_rate / acceleration
    travelled = np.zeros_like(elapsed)
    if 2.0 * acceleration_distance >= distance:
        acceleration_time = math.sqrt(distance / acceleration)
        total_time = 2.0 * acceleration_time
        accelerating = elapsed <= acceleration_time
        braking = (elapsed > acceleration_time) & (elapsed < total_time)
        travelled[accelerating] = 0.5 * acceleration * elapsed[accelerating] ** 2
        remaining_time = total_time - elapsed[braking]
        travelled[braking] = distance - 0.5 * acceleration * remaining_time**2
        travelled[elapsed >= total_time] = distance
    else:
        acceleration_time = max_rate / acceleration
        cruise_distance = distance - 2.0 * acceleration_distance
        cruise_time = cruise_distance / max_rate
        brake_start = acceleration_time + cruise_time
        total_time = brake_start + acceleration_time
        accelerating = elapsed <= acceleration_time
        cruising = (elapsed > acceleration_time) & (elapsed <= brake_start)
        braking = (elapsed > brake_start) & (elapsed < total_time)
        travelled[accelerating] = 0.5 * acceleration * elapsed[accelerating] ** 2
        travelled[cruising] = acceleration_distance + max_rate * (
            elapsed[cruising] - acceleration_time
        )
        remaining_time = total_time - elapsed[braking]
        travelled[braking] = distance - 0.5 * acceleration * remaining_time**2
        travelled[elapsed >= total_time] = distance
    direction = math.copysign(1.0, float(target_position) - float(initial_position))
    return np.clip(
        float(initial_position) + direction * travelled,
        min(float(initial_position), float(target_position)),
        max(float(initial_position), float(target_position)),
    )


def _segment_rmse(segment: MotionSegment, parameters: Sequence[float]) -> float:
    predicted = predict_step_response(
        segment.time_sec,
        initial_position=segment.initial_position,
        target_position=segment.target_position,
        delay_sec=float(parameters[0]),
        max_rate_per_sec=float(parameters[1]),
        acceleration_per_sec2=float(parameters[2]),
    )
    return float(np.sqrt(np.mean((predicted - segment.measured_position) ** 2)))


def _pooled_loss(segments: Sequence[MotionSegment], parameters: Sequence[float]) -> float:
    errors = np.asarray([_segment_rmse(segment, parameters) for segment in segments])
    if not len(errors):
        return math.inf
    cutoff = 0.03
    huber = np.where(
        errors <= cutoff,
        0.5 * errors**2,
        cutoff * (errors - 0.5 * cutoff),
    )
    return float(np.mean(huber))


def _initial_parameters(segments: Sequence[MotionSegment]) -> np.ndarray:
    delays: list[float] = []
    rates: list[float] = []
    for segment in segments:
        movement = np.abs(segment.measured_position - segment.initial_position)
        threshold = max(0.004, min(0.01, 0.04 * segment.commanded_distance))
        indices = np.flatnonzero(movement >= threshold)
        if len(indices):
            delays.append(max(0.0, float(segment.time_sec[int(indices[0])]) - 0.01))
        dt = np.diff(segment.time_sec)
        velocity = np.abs(np.diff(segment.measured_position) / dt)
        velocity = velocity[np.isfinite(velocity) & (velocity > 0.01) & (velocity < 8.0)]
        if len(velocity):
            rates.append(float(np.quantile(velocity, 0.70)))
    delay = statistics.median(delays) if delays else 0.10
    rate = statistics.median(rates) if rates else 1.4
    return np.asarray([np.clip(delay, 0.0, 0.4), np.clip(rate, 0.05, 5.0), 30.0])


def _coordinate_fit(
    segments: Sequence[MotionSegment],
    initial: Sequence[float] | None = None,
) -> np.ndarray:
    current = _initial_parameters(segments) if initial is None else np.asarray(initial, dtype=float)
    bounds = ((0.0, 0.4), (0.05, 5.0), (0.1, 300.0))
    refinements = (
        (0.08, 0.60, 0.90),
        (0.03, 0.30, 0.50),
        (0.012, 0.14, 0.25),
        (0.004, 0.06, 0.10),
        (0.0015, 0.025, 0.04),
    )
    for delay_span, rate_fraction, acceleration_fraction in refinements:
        delay_values = np.linspace(current[0] - delay_span, current[0] + delay_span, 9)
        rate_values = current[1] * np.linspace(1.0 - rate_fraction, 1.0 + rate_fraction, 9)
        acceleration_values = current[2] * np.linspace(
            1.0 - acceleration_fraction,
            1.0 + acceleration_fraction,
            9,
        )
        best = current.copy()
        best_loss = _pooled_loss(segments, best)
        for delay in np.clip(delay_values, *bounds[0]):
            for rate in np.clip(rate_values, *bounds[1]):
                for acceleration in np.clip(acceleration_values, *bounds[2]):
                    candidate = (float(delay), float(rate), float(acceleration))
                    loss = _pooled_loss(segments, candidate)
                    if loss < best_loss:
                        best_loss = loss
                        best = np.asarray(candidate)
        current = best
    return current


def fit_direction(segments: Sequence[MotionSegment]) -> dict[str, Any]:
    if not segments:
        raise ValueError("at least one motion segment is required")
    initial = _coordinate_fit(segments)
    initial_errors = np.asarray([_segment_rmse(segment, initial) for segment in segments])
    median = float(np.median(initial_errors))
    mad = float(np.median(np.abs(initial_errors - median)))
    outlier_threshold = median + max(0.01, 3.0 * 1.4826 * mad)
    kept = [
        segment
        for segment, error in zip(segments, initial_errors)
        if float(error) <= outlier_threshold
    ]
    rejected = [
        {"segment_key": segment.key, "rmse": float(error), "reason": "robust_rmse_outlier"}
        for segment, error in zip(segments, initial_errors)
        if float(error) > outlier_threshold
    ]
    parameters = _coordinate_fit(kept, initial=initial) if kept else initial
    errors = np.asarray([_segment_rmse(segment, parameters) for segment in kept])
    absolute_errors: list[float] = []
    for segment in kept:
        predicted = predict_step_response(
            segment.time_sec,
            initial_position=segment.initial_position,
            target_position=segment.target_position,
            delay_sec=float(parameters[0]),
            max_rate_per_sec=float(parameters[1]),
            acceleration_per_sec2=float(parameters[2]),
        )
        absolute_errors.extend(np.abs(predicted - segment.measured_position).tolist())
    return {
        "delay_sec": float(parameters[0]),
        "max_rate_per_sec": float(parameters[1]),
        "acceleration_per_sec2": float(parameters[2]),
        "segment_count": len(kept),
        "source_segment_count": len(segments),
        "segment_rmse_median": float(np.median(errors)),
        "segment_rmse_max": float(np.max(errors)),
        "sample_abs_error_p95": float(np.quantile(absolute_errors, 0.95)),
        "outlier_threshold_rmse": float(outlier_threshold),
        "used_segments": [segment.key for segment in kept],
        "rejected_segments": rejected,
        "status": "identified" if len(kept) >= 3 else "provisional",
    }


def _validate_conditions(
    segments: Sequence[MotionSegment],
    *,
    expected_speed: float | None,
    expected_force: float | None,
    expected_continuous: bool | None,
    expected_command_mode: str | None,
) -> None:
    def matches(observed: object, expected: object) -> bool:
        if observed is None:
            return False
        if isinstance(expected, bool):
            return bool(observed) is expected
        if isinstance(expected, (int, float)):
            return math.isclose(float(observed), float(expected), abs_tol=1e-6)
        return str(observed) == str(expected)

    mismatches: list[str] = []
    for name, values in (
        ("speed", {segment.speed_setting for segment in segments}),
        ("force", {segment.force_setting for segment in segments}),
        ("continuous", {segment.continuous_grasp for segment in segments}),
        ("command_mode", {segment.command_mode for segment in segments}),
    ):
        known = {value for value in values if value is not None}
        if len(known) > 1:
            mismatches.append(f"mixed {name} source conditions: {sorted(known)}")
    for segment in segments:
        checks = (
            ("speed", segment.speed_setting, expected_speed),
            ("force", segment.force_setting, expected_force),
            ("continuous", segment.continuous_grasp, expected_continuous),
            ("command_mode", segment.command_mode, expected_command_mode),
        )
        for name, observed, expected in checks:
            if expected is None:
                continue
            if not matches(observed, expected):
                mismatches.append(
                    f"{segment.key}:{name} observed={observed!r} expected={expected!r}"
                )
    if mismatches:
        raise ValueError("source condition mismatch: " + "; ".join(mismatches[:20]))


def build_fit_artifact(
    fit_sources: Sequence[str | Path],
    *,
    validation_sources: Sequence[str | Path] = (),
    expected_speed: float | None = 1.0,
    expected_force: float | None = None,
    expected_continuous: bool | None = None,
    expected_command_mode: str | None = "runtime-equivalent",
) -> dict[str, Any]:
    fit_segments: list[MotionSegment] = []
    fit_evidence: list[dict[str, Any]] = []
    for source in fit_sources:
        segments, evidence = load_motion_segments(source)
        fit_segments.extend(segments)
        fit_evidence.append(evidence)
    if not fit_segments:
        raise ValueError("no usable fit motion segments were found")
    _validate_conditions(
        fit_segments,
        expected_speed=expected_speed,
        expected_force=expected_force,
        expected_continuous=expected_continuous,
        expected_command_mode=expected_command_mode,
    )
    fit_by_side: dict[str, dict[str, Any]] = {}
    for side in sorted({segment.side for segment in fit_segments}):
        side_segments = [segment for segment in fit_segments if segment.side == side]
        fit_by_side[side] = {}
        for direction in ("close", "open"):
            selected = [segment for segment in side_segments if segment.direction == direction]
            if selected:
                fit_by_side[side][direction] = fit_direction(selected)

    validation_segments: list[MotionSegment] = []
    validation_evidence: list[dict[str, Any]] = []
    for source in validation_sources:
        segments, evidence = load_motion_segments(source)
        validation_segments.extend(segments)
        validation_evidence.append(evidence)
    _validate_conditions(
        [*fit_segments, *validation_segments],
        expected_speed=expected_speed,
        expected_force=expected_force,
        expected_continuous=expected_continuous,
        expected_command_mode=expected_command_mode,
    )
    validation: dict[str, Any] = {"status": "not_run", "groups": {}}
    if validation_segments:
        validation["status"] = "evaluated"
        for side, directions in fit_by_side.items():
            for direction, parameters in directions.items():
                selected = [
                    segment
                    for segment in validation_segments
                    if segment.side == side and segment.direction == direction
                ]
                if not selected:
                    continue
                values = (
                    parameters["delay_sec"],
                    parameters["max_rate_per_sec"],
                    parameters["acceleration_per_sec2"],
                )
                errors = [_segment_rmse(segment, values) for segment in selected]
                validation["groups"][f"{side}.{direction}"] = {
                    "segment_count": len(errors),
                    "segment_rmse_median": float(np.median(errors)),
                    "segment_rmse_max": float(np.max(errors)),
                    "segments": [segment.key for segment in selected],
                }
    payload: dict[str, Any] = {
        "schema": FIT_SCHEMA,
        "model_type": MODEL_TYPE,
        "scope": "free_motion_normalized_position_only",
        "fit_clock": "command_begin_to_state_read_midpoint_monotonic",
        "command_write_duration_sec": {
            "median": float(
                np.median([segment.command_duration_sec for segment in fit_segments])
            ),
            "p95": float(
                np.quantile(
                    [segment.command_duration_sec for segment in fit_segments],
                    0.95,
                )
            ),
            "max": max(segment.command_duration_sec for segment in fit_segments),
        },
        "source_conditions_required": {
            "speed_setting": expected_speed,
            "force_setting": expected_force,
            "continuous_grasp": expected_continuous,
            "command_mode": expected_command_mode,
        },
        "source_conditions_observed": {
            "speed_setting": sorted(
                {
                    segment.speed_setting
                    for segment in fit_segments
                    if segment.speed_setting is not None
                }
            ),
            "force_setting": sorted(
                {
                    segment.force_setting
                    for segment in fit_segments
                    if segment.force_setting is not None
                }
            ),
            "continuous_grasp": sorted(
                {
                    segment.continuous_grasp
                    for segment in fit_segments
                    if segment.continuous_grasp is not None
                }
            ),
            "command_mode": sorted(
                {
                    segment.command_mode
                    for segment in fit_segments
                    if segment.command_mode is not None
                }
            ),
        },
        "fit_by_side": fit_by_side,
        "fit_sources": fit_evidence,
        "validation": validation,
        "validation_sources": validation_evidence,
        "fidelity_claims": {
            "free_motion_position_response": (
                "identified" if all(
                    result["status"] == "identified"
                    for directions in fit_by_side.values()
                    for result in directions.values()
                ) else "provisional"
            ),
            "contact_force_newtons": "unavailable",
            "contact_stall_compliance": "unmodeled",
            "friction_and_slip": "unmodeled",
        },
        "limitations": [
            "No external force gauge is used; force/current settings are not Newtons.",
            "The primary fit assumes zero initial velocity per accepted step.",
            "Fast preemption/reversal segments that violate monotonic response are excluded.",
            "A side is not inferred from the other side; absent sides remain absent.",
        ],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _parse_optional_bool(value: str) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized == "any":
        return None
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected true, false, or any")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--validation-input", action="append", default=[], type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-speed", type=float, default=1.0)
    parser.add_argument("--expected-force", type=float, default=None)
    parser.add_argument(
        "--expected-continuous",
        type=_parse_optional_bool,
        default=None,
        metavar="{true,false,any}",
    )
    parser.add_argument(
        "--expected-command-mode",
        choices=("runtime-equivalent", "inline-config", "any"),
        default="runtime-equivalent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = build_fit_artifact(
        args.input,
        validation_sources=args.validation_input,
        expected_speed=args.expected_speed,
        expected_force=args.expected_force,
        expected_continuous=args.expected_continuous,
        expected_command_mode=(
            None if args.expected_command_mode == "any" else args.expected_command_mode
        ),
    )
    output = args.output.expanduser().resolve()
    _atomic_write_json(output, artifact)
    print(
        f"WROTE schema={FIT_SCHEMA} sides={sorted(artifact['fit_by_side'])} "
        f"status={artifact['fidelity_claims']['free_motion_position_response']} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FIT_SCHEMA",
    "MODEL_TYPE",
    "MotionSegment",
    "build_fit_artifact",
    "fit_direction",
    "load_motion_segments",
    "predict_step_response",
]

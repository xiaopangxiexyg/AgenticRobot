"""Replay recorded control-frame force through the current bounded C0 core.

Action-debug records are sampled at 15 Hz and do not contain every original
force-monitor sample.  They do contain the force after the runtime frame
transform and low-pass filter.  This replay therefore validates the current
profile-compliance response, release, and total output envelope against saved
force evidence without claiming to revalidate the raw tool-frame transform or
the complete 20 Hz producer stream.

This module is offline-only and owns no device, IK, safety, or dispatch.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.position_control.cartesian_admittance import (
    BoundedCartesianAdmittance,
    CartesianAdmittanceConfig,
    CartesianAdmittanceWrenchSample,
)


SIDES = ("left", "right")
DEFAULT_C0_MAX_OFFSET_M = 0.002
DEFAULT_C0_MAX_SPEED_M_S = 0.005
DEFAULT_REPLAY_HZ = 60.0
DEFAULT_RELEASE_TAIL_SEC = 6.0


@dataclass(frozen=True)
class _RecordedForceEvent:
    observed_at_sec: float
    source_time_sec: float
    filtered_force_control_n: tuple[float, float, float]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _vector3(value: Any, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain three finite values")
    vector = tuple(float(item) for item in value)
    if len(vector) != 3 or not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain three finite values")
    return vector


def _resolve_log(path: Path) -> Path:
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


def _position_admittance_receipt(
    rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    for row in rows:
        if row.get("event") != "position_admittance_config":
            continue
        payload = _mapping(row.get("payload"))
        if payload.get("enabled") is True:
            return payload
    return {}


def _force_events(
    rows: Sequence[Mapping[str, Any]],
    *,
    side: str,
) -> list[_RecordedForceEvent]:
    events: list[_RecordedForceEvent] = []
    previous_observed: float | None = None
    previous_source: float | None = None
    previous_force: tuple[float, float, float] | None = None
    for row in rows:
        if row.get("event") != "frame":
            continue
        payload = _mapping(
            _mapping(_mapping(row.get("position_admittance")).get("sides")).get(
                side
            )
        )
        if not payload:
            continue
        force = _vector3(
            payload.get("filtered_force_control_n"),
            name=f"{side}.filtered_force_control_n",
        )
        observed = float(payload.get("observed_at_monotonic_sec"))
        source = float(payload.get("source_time_sec"))
        if not math.isfinite(observed) or not math.isfinite(source):
            raise ValueError("force trace timestamps must be finite")
        if source > observed:
            raise ValueError("force trace source time cannot be in the future")
        if previous_observed is not None and observed <= previous_observed:
            raise ValueError("force trace observation time must increase")
        if previous_source is not None and source < previous_source:
            raise ValueError("sampled force trace source time must increase")
        if previous_source is not None and source == previous_source:
            if force != previous_force:
                raise ValueError(
                    "duplicate force source time must preserve the same force"
                )
        events.append(
            _RecordedForceEvent(
                observed_at_sec=observed,
                source_time_sec=source,
                filtered_force_control_n=force,
            )
        )
        previous_observed = observed
        previous_source = source
        previous_force = force
    return events


def _candidate_config(
    receipt: Mapping[str, Any],
    *,
    max_offset_m: float,
    max_speed_m_s: float,
    max_return_speed_m_s: float | None = None,
    max_output_slew_m_s: float | None = None,
) -> CartesianAdmittanceConfig:
    raw_config = dict(_mapping(receipt.get("config")))
    if not raw_config:
        raise ValueError("position-admittance receipt has no config")
    source = CartesianAdmittanceConfig(**raw_config)
    if source.response_model != "profile_compliance":
        raise ValueError("force replay requires profile_compliance evidence")
    limit = float(max_offset_m)
    speed = float(max_speed_m_s)
    return_speed = (
        speed
        if max_return_speed_m_s is None
        else float(max_return_speed_m_s)
    )
    output_slew = (
        speed
        if max_output_slew_m_s is None
        else float(max_output_slew_m_s)
    )
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("max_offset_m must be finite and positive")
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("max_speed_m_s must be finite and positive")
    if not math.isfinite(return_speed) or return_speed <= 0.0:
        raise ValueError("max_return_speed_m_s must be finite and positive")
    if not math.isfinite(output_slew) or output_slew <= 0.0:
        raise ValueError("max_output_slew_m_s must be finite and positive")
    return replace(
        source,
        # The log already stores force after the runtime low-pass and frame
        # transform.  Do not filter or rotate that evidence a second time.
        control_frame="solver_world",
        force_filter_tau_sec=0.0,
        force_sign=(1.0, 1.0, 1.0),
        max_offset_m=(limit, limit, limit),
        max_velocity_m_s=(speed, speed, speed),
        max_return_velocity_m_s=(return_speed, return_speed, return_speed),
        max_output_slew_m_s=(output_slew, output_slew, output_slew),
    )


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=float,
    )
    if finite.size == 0:
        return {"n": 0, "p50": None, "p95": None, "min": None, "max": None}
    return {
        "n": int(finite.size),
        "p50": float(np.percentile(finite, 50.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def _append_release_tail(
    events: Sequence[_RecordedForceEvent],
    *,
    duration_sec: float,
) -> list[_RecordedForceEvent]:
    result = list(events)
    if duration_sec <= 0.0 or not result:
        return result
    observed_deltas = np.diff(
        np.asarray([event.observed_at_sec for event in result], dtype=float)
    )
    source_deltas = np.diff(
        np.asarray([event.source_time_sec for event in result], dtype=float)
    )
    observed_dt = (
        float(np.median(observed_deltas))
        if observed_deltas.size
        else 1.0 / 15.0
    )
    positive_source_deltas = source_deltas[source_deltas > 0.0]
    source_dt = (
        float(np.median(positive_source_deltas))
        if positive_source_deltas.size
        else observed_dt
    )
    tail_dt = min(observed_dt, source_dt, 0.1)
    if not math.isfinite(tail_dt) or tail_dt <= 0.0:
        raise ValueError("force trace cannot determine a positive tail interval")
    count = int(math.ceil(float(duration_sec) / tail_dt))
    last = result[-1]
    for index in range(1, count + 1):
        result.append(
            _RecordedForceEvent(
                observed_at_sec=last.observed_at_sec + index * tail_dt,
                source_time_sec=last.source_time_sec + index * tail_dt,
                filtered_force_control_n=(0.0, 0.0, 0.0),
            )
        )
    return result


def replay_recorded_control_force(
    path: Path,
    *,
    side: str = "left",
    max_offset_m: float = DEFAULT_C0_MAX_OFFSET_M,
    max_speed_m_s: float = DEFAULT_C0_MAX_SPEED_M_S,
    max_return_speed_m_s: float | None = None,
    max_output_slew_m_s: float | None = None,
    replay_hz: float = DEFAULT_REPLAY_HZ,
    release_tail_sec: float = DEFAULT_RELEASE_TAIL_SEC,
) -> dict[str, Any]:
    """Run saved filtered control-force samples through the current core."""

    selected_side = str(side).strip().lower()
    if selected_side not in SIDES:
        raise ValueError("side must be left or right")
    resolved_hz = float(replay_hz)
    if not math.isfinite(resolved_hz) or resolved_hz <= 0.0:
        raise ValueError("replay_hz must be finite and positive")
    tail_sec = float(release_tail_sec)
    if not math.isfinite(tail_sec) or tail_sec < 0.0:
        raise ValueError("release_tail_sec must be finite and non-negative")

    log_path = _resolve_log(path)
    rows = _read_rows(log_path)
    receipt = _position_admittance_receipt(rows)
    recorded_events = _force_events(rows, side=selected_side)
    if len(recorded_events) < 2:
        raise ValueError("force replay requires at least two recorded frames")
    config = _candidate_config(
        receipt,
        max_offset_m=max_offset_m,
        max_speed_m_s=max_speed_m_s,
        max_return_speed_m_s=max_return_speed_m_s,
        max_output_slew_m_s=max_output_slew_m_s,
    )
    events = _append_release_tail(recorded_events, duration_sec=tail_sec)
    controller = BoundedCartesianAdmittance(config)
    tick_period = 1.0 / resolved_hz
    initial_source = recorded_events[0].source_time_sec - min(
        1.0 / 15.0,
        config.max_force_dt_sec * 0.5,
    )
    start_tick = recorded_events[0].observed_at_sec
    controller.reset(
        tare_sample=CartesianAdmittanceWrenchSample(
            values_n_nm=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            frame=config.control_frame,
            source_time_sec=initial_source,
            receive_time_sec=start_tick,
            tare_generation=1,
        ),
        output_time_sec=start_tick - tick_period,
    )

    end_tick = events[-1].observed_at_sec + tick_period
    tick_count = int(math.ceil((end_tick - start_tick) * resolved_hz)) + 1
    event_index = 0
    recorded_event_count_applied = 0
    reasons: Counter[str] = Counter()
    frozen_tick_count = 0
    contact_tick_count = 0
    envelope_violation_count = 0
    finite_violation_count = 0
    input_force_norms: list[float] = []
    deadbanded_force_norms: list[float] = []
    requested_norms: list[float] = []
    output_norms: list[float] = []
    requested_velocity_norms: list[float] = []
    output_slew_norms: list[float] = []
    direction_cosines: list[float] = []
    motion_direction_cosines: list[float] = []
    opposed_output_tick_count = 0
    opposed_motion_tick_count = 0
    current_opposed_output_run = 0
    max_opposed_output_run = 0
    max_opposed_output_norm_m = 0.0
    strong_axis_counts = [0, 0, 0]
    strong_axis_sign_mismatches = [0, 0, 0]
    strong_axis_mismatches_after_reversal = [0, 0, 0]
    last_strong_force_sign = [0, 0, 0]
    last_strong_sign_change_tick: list[int | None] = [None, None, None]
    previous_output = np.zeros(3, dtype=float)
    previous_tick = start_tick - tick_period
    tolerance = 1e-9

    for tick_index in range(tick_count):
        now = start_tick + tick_index * tick_period
        if now > end_tick + tolerance:
            break
        while (
            event_index < len(events)
            and events[event_index].observed_at_sec <= now + tolerance
        ):
            event = events[event_index]
            sample = CartesianAdmittanceWrenchSample(
                values_n_nm=(*event.filtered_force_control_n, 0.0, 0.0, 0.0),
                frame=config.control_frame,
                source_time_sec=event.source_time_sec,
                receive_time_sec=now,
                tare_generation=1,
            )
            controller.step_new_sample(sample, now_sec=now, active=True)
            if event_index < len(recorded_events):
                recorded_event_count_applied += 1
                input_force_norms.append(
                    float(np.linalg.norm(event.filtered_force_control_n))
                )
            event_index += 1
        state = controller.sample_output(now_sec=now, active=True)
        reasons[state.reason] += 1
        frozen_tick_count += int(state.frozen)
        requested = np.asarray(state.requested_offset_m, dtype=float)
        output = np.asarray(state.output_offset_m, dtype=float)
        velocity = np.asarray(state.velocity_m_s, dtype=float)
        deadbanded = np.asarray(state.deadbanded_force_control_n, dtype=float)
        vectors = (requested, output, velocity, deadbanded)
        if not all(np.all(np.isfinite(vector)) for vector in vectors):
            finite_violation_count += 1
            continue
        requested_norm = float(np.linalg.norm(requested))
        output_norm = float(np.linalg.norm(output))
        velocity_norm = float(np.linalg.norm(velocity))
        deadbanded_norm = float(np.linalg.norm(deadbanded))
        tick_dt = now - previous_tick
        output_slew = float(np.linalg.norm(output - previous_output) / tick_dt)
        requested_norms.append(requested_norm)
        output_norms.append(output_norm)
        requested_velocity_norms.append(velocity_norm)
        deadbanded_force_norms.append(deadbanded_norm)
        output_slew_norms.append(output_slew)
        if deadbanded_norm > 1e-12:
            contact_tick_count += 1
        if deadbanded_norm > 1e-12 and output_norm > 1e-12:
            direction_cosine = float(
                np.dot(deadbanded, output) / (deadbanded_norm * output_norm)
            )
            direction_cosines.append(direction_cosine)
            opposed = direction_cosine < -1e-9
            opposed_output_tick_count += int(opposed)
            if opposed:
                current_opposed_output_run += 1
                max_opposed_output_run = max(
                    max_opposed_output_run,
                    current_opposed_output_run,
                )
                max_opposed_output_norm_m = max(
                    max_opposed_output_norm_m,
                    output_norm,
                )
            else:
                current_opposed_output_run = 0
        else:
            current_opposed_output_run = 0
        for axis in range(3):
            if abs(deadbanded[axis]) > 1e-12:
                force_sign = 1 if deadbanded[axis] > 0.0 else -1
                if (
                    last_strong_force_sign[axis] != 0
                    and force_sign != last_strong_force_sign[axis]
                ):
                    last_strong_sign_change_tick[axis] = tick_index
                last_strong_force_sign[axis] = force_sign
            if abs(deadbanded[axis]) < 1.0:
                continue
            if abs(output[axis]) <= 1e-6:
                continue
            strong_axis_counts[axis] += 1
            mismatch = deadbanded[axis] * output[axis] < 0.0
            strong_axis_sign_mismatches[axis] += int(mismatch)
            change_tick = last_strong_sign_change_tick[axis]
            strong_axis_mismatches_after_reversal[axis] += int(
                mismatch
                and change_tick is not None
                and tick_index - change_tick <= math.ceil(0.2 * resolved_hz)
            )
        output_delta = output - previous_output
        output_delta_norm = float(np.linalg.norm(output_delta))
        if deadbanded_norm > 1e-12 and output_delta_norm > 1e-12:
            motion_direction_cosine = float(
                np.dot(deadbanded, output_delta)
                / (deadbanded_norm * output_delta_norm)
            )
            motion_direction_cosines.append(motion_direction_cosine)
            opposed_motion_tick_count += int(motion_direction_cosine < -1e-9)
        if (
            requested_norm > float(max_offset_m) + tolerance
            or output_norm > float(max_offset_m) + tolerance
            or velocity_norm > float(max_speed_m_s) + tolerance
            or output_slew
            > max(config.max_output_slew_m_s) + tolerance
        ):
            envelope_violation_count += 1
        previous_output = output.copy()
        previous_tick = now

    final_output_norm = float(np.linalg.norm(previous_output))
    release_limit_m = min(1e-5, float(max_offset_m) * 0.01)
    reversal_clearance_limit_sec = (
        float(max_offset_m) / float(max_speed_m_s) + tick_period
    )
    receipt_valid = bool(
        receipt
        and receipt.get("enabled") is True
        and selected_side in tuple(receipt.get("active_sides", SIDES))
        and _mapping(receipt.get("config")).get("response_model")
        == "profile_compliance"
    )
    checks = [
        {
            "name": "recorded_profile_compliance_receipt",
            "passed": receipt_valid,
            "detail": {
                "schema_version": receipt.get("schema_version"),
                "requested_mode": receipt.get("requested_mode"),
            },
        },
        {
            "name": "recorded_force_trace_coverage",
            "passed": recorded_event_count_applied == len(recorded_events),
            "detail": {
                "recorded": len(recorded_events),
                "applied": recorded_event_count_applied,
            },
        },
        {
            "name": "current_core_stays_finite_and_unfrozen",
            "passed": finite_violation_count == 0 and frozen_tick_count == 0,
            "detail": {
                "finite_violation_count": finite_violation_count,
                "frozen_tick_count": frozen_tick_count,
            },
        },
        {
            "name": "current_c0_total_envelope",
            "passed": envelope_violation_count == 0,
            "detail": {
                "violation_count": envelope_violation_count,
                "max_offset_m": float(max_offset_m),
                "max_speed_m_s": float(max_speed_m_s),
                "max_return_speed_m_s": max(
                    config.max_return_velocity_m_s
                ),
                "max_output_slew_m_s": max(config.max_output_slew_m_s),
            },
        },
        {
            "name": "passive_release_tail",
            "passed": tail_sec < 6.0 or final_output_norm <= release_limit_m,
            "detail": {
                "tail_sec": tail_sec,
                "final_output_norm_m": final_output_norm,
                "limit_m": release_limit_m,
            },
        },
        {
            "name": "bounded_force_reversal_transient",
            "passed": (
                max_opposed_output_run / resolved_hz
                <= reversal_clearance_limit_sec + tolerance
            ),
            "detail": {
                "max_opposed_output_run_sec": (
                    max_opposed_output_run / resolved_hz
                ),
                "clearance_limit_sec": reversal_clearance_limit_sec,
            },
        },
    ]
    return {
        "schema": "dexteleop.position_admittance_force_trace_replay/v1",
        "passed": all(check["passed"] for check in checks),
        "evidence_level": "offline_sampled_control_force_replay",
        "path": str(log_path),
        "side": selected_side,
        "input_contract": "recorded_filtered_force_control_n_after_frame_and_filter",
        "limitations": [
            "15 Hz action-debug samples do not contain every original force sample",
            "raw tool-to-solver frame transformation is not re-evaluated",
            "force-only replay does not prove IK, contact physics, or hardware motion",
        ],
        "checks": checks,
        "recorded_mode": receipt.get("requested_mode"),
        "recorded_schema_version": receipt.get("schema_version"),
        "recorded_event_count": len(recorded_events),
        "recorded_duration_sec": (
            recorded_events[-1].observed_at_sec
            - recorded_events[0].observed_at_sec
        ),
        "replay_hz": resolved_hz,
        "replay_tick_count": sum(reasons.values()),
        "release_tail_sec": tail_sec,
        "candidate_config": asdict(config),
        "reason_counts": dict(reasons),
        "contact_tick_count": contact_tick_count,
        "input_filtered_force_norm_n": _stats(input_force_norms),
        "replayed_deadbanded_force_norm_n": _stats(deadbanded_force_norms),
        "requested_offset_norm_m": _stats(requested_norms),
        "output_offset_norm_m": _stats(output_norms),
        "requested_velocity_norm_m_s": _stats(requested_velocity_norms),
        "output_slew_norm_m_s": _stats(output_slew_norms),
        "force_output_direction_cosine": _stats(direction_cosines),
        "force_motion_direction_cosine": _stats(motion_direction_cosines),
        "opposed_output_tick_count": opposed_output_tick_count,
        "opposed_motion_tick_count": opposed_motion_tick_count,
        "max_opposed_output_run_ticks": max_opposed_output_run,
        "max_opposed_output_run_sec": max_opposed_output_run / resolved_hz,
        "max_opposed_output_norm_m": max_opposed_output_norm_m,
        "strong_axis_force_output_sign": {
            axis: {
                "tick_count": strong_axis_counts[index],
                "mismatch_count": strong_axis_sign_mismatches[index],
                "mismatch_within_200ms_of_reversal": (
                    strong_axis_mismatches_after_reversal[index]
                ),
            }
            for index, axis in enumerate(("x", "y", "z"))
        },
        "final_output_norm_m": final_output_norm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--side", choices=SIDES, default="left")
    parser.add_argument("--max-offset-mm", type=float, default=2.0)
    parser.add_argument("--max-speed-mm-s", type=float, default=5.0)
    parser.add_argument(
        "--max-return-speed-mm-s",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-output-slew-mm-s",
        type=float,
        default=None,
    )
    parser.add_argument("--replay-hz", type=float, default=DEFAULT_REPLAY_HZ)
    parser.add_argument(
        "--release-tail-sec",
        type=float,
        default=DEFAULT_RELEASE_TAIL_SEC,
    )
    args = parser.parse_args()
    report = replay_recorded_control_force(
        args.log,
        side=args.side,
        max_offset_m=args.max_offset_mm / 1000.0,
        max_speed_m_s=args.max_speed_mm_s / 1000.0,
        max_return_speed_m_s=(
            None
            if args.max_return_speed_mm_s is None
            else args.max_return_speed_mm_s / 1000.0
        ),
        max_output_slew_m_s=(
            None
            if args.max_output_slew_mm_s is None
            else args.max_output_slew_mm_s / 1000.0
        ),
        replay_hz=args.replay_hz,
        release_tail_sec=args.release_tail_sec,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()

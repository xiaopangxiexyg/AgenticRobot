from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) * float(value) for value in values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def _stats_ms(values_sec: list[float]) -> dict[str, float]:
    values_ms = [value * 1000.0 for value in values_sec]
    return {
        "mean_ms": statistics.fmean(values_ms) if values_ms else math.nan,
        "p50_ms": _percentile(values_ms, 0.50),
        "p95_ms": _percentile(values_ms, 0.95),
        "p99_ms": _percentile(values_ms, 0.99),
        "max_ms": max(values_ms) if values_ms else math.nan,
    }


def _remember_event(events: dict[str, dict[str, float]], name: str, step: int, time_sec: float, value: float) -> None:
    if name not in events:
        events[name] = {
            "step": step,
            "time_sec": time_sec,
            "value": value,
        }


def _latency_ms(events: dict[str, dict[str, float]], later: str, earlier: str) -> float | None:
    if later not in events or earlier not in events:
        return None
    return (events[later]["time_sec"] - events[earlier]["time_sec"]) * 1000.0


def _fmt(value: float | None, suffix: str = "ms") -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.1f}{suffix}"


def _record_time_sec(records: list[dict[str, object]]) -> list[float]:
    if records and "time_sec" in records[0]:
        return [float(record["time_sec"]) for record in records]
    times = []
    current = 0.0
    for record in records:
        current += float(record.get("dt", 0.0))
        times.append(current)
    return times


def _analyze(data: dict[str, object], args) -> dict[str, object]:
    records = list(data.get("records", []))
    times = _record_time_sec(records)
    rows = []
    for index, record in enumerate(records):
        step = int(record.get("step", index + 1))
        time_sec = times[index]
        force_norm = float(record.get("force_norm_n", _norm(record.get("tare_corrected_wrench", [0.0, 0.0, 0.0])[:3])))
        active_norm = _norm(record.get("active_wrench", [0.0, 0.0, 0.0])[:3])
        target_norm = _norm(record.get("target_offset", [0.0, 0.0, 0.0])[:3])
        offset_norm = _norm(record.get("offset", [0.0, 0.0, 0.0])[:3])
        pose_norm = float(record.get("pose_linear_delta_m", 0.0))
        rows.append((step, time_sec, force_norm, active_norm, target_norm, offset_norm, pose_norm))

    events: dict[str, dict[str, float]] = {}
    force_anchor_step = None
    for step, time_sec, force_norm, *_ in rows:
        if force_norm >= args.force_threshold_n:
            force_anchor_step = step
            _remember_event(events, "force_threshold", step, time_sec, force_norm)
            break
    for step, time_sec, force_norm, active_norm, target_norm, offset_norm, pose_norm in rows:
        if force_anchor_step is not None and step < force_anchor_step:
            continue
        if active_norm >= args.active_threshold_n:
            _remember_event(events, "active_threshold", step, time_sec, active_norm)
        if target_norm >= args.offset_threshold_m:
            _remember_event(events, "target_threshold", step, time_sec, target_norm)
        if offset_norm >= args.offset_threshold_m:
            _remember_event(events, "offset_threshold", step, time_sec, offset_norm)
        if pose_norm >= args.pose_threshold_m:
            _remember_event(events, "pose_threshold", step, time_sec, pose_norm)

    timing_values: dict[str, list[float]] = {
        "state_read": [],
        "force_read": [],
        "compute": [],
        "command": [],
        "loop_elapsed": [],
        "sleep": [],
    }
    missed = 0
    for record in records:
        timing = record.get("timing")
        if not isinstance(timing, dict):
            continue
        timing_values["state_read"].append(float(timing.get("state_read_sec", 0.0)))
        timing_values["force_read"].append(float(timing.get("force_read_sec", 0.0)))
        timing_values["compute"].append(float(timing.get("compute_sec", 0.0)))
        timing_values["command"].append(float(timing.get("command_sec", 0.0)))
        timing_values["loop_elapsed"].append(float(timing.get("loop_elapsed_sec", 0.0)))
        timing_values["sleep"].append(float(timing.get("sleep_sec", 0.0)))
        if timing.get("missed_deadline"):
            missed += 1
    timing_summary = {
        name: _stats_ms(values)
        for name, values in timing_values.items()
        if values
    }

    latencies = {
        "force_to_active_ms": _latency_ms(events, "active_threshold", "force_threshold"),
        "force_to_offset_ms": _latency_ms(events, "offset_threshold", "force_threshold"),
        "force_to_pose_ms": _latency_ms(events, "pose_threshold", "force_threshold"),
        "active_to_offset_ms": _latency_ms(events, "offset_threshold", "active_threshold"),
        "target_to_offset_ms": _latency_ms(events, "offset_threshold", "target_threshold"),
        "target_to_pose_ms": _latency_ms(events, "pose_threshold", "target_threshold"),
        "offset_to_pose_ms": _latency_ms(events, "pose_threshold", "offset_threshold"),
    }
    return {
        "records": rows,
        "events": events,
        "latencies": latencies,
        "timing_summary": timing_summary,
        "timing_missed_deadlines": missed if timing_summary else None,
    }


def _print_window(rows: list[tuple[int, float, float, float, float, float, float]], center_step: int, window_steps: int) -> None:
    lo = center_step - window_steps
    hi = center_step + window_steps
    print("\nwindow around first force threshold:")
    print(" step   t_sec  force_N active_N target_mm offset_mm  pose_mm")
    for step, time_sec, force_norm, active_norm, target_norm, offset_norm, pose_norm in rows:
        if lo <= step <= hi:
            print(
                f"{step:5d} {time_sec:7.3f} {force_norm:8.3f} {active_norm:8.3f} "
                f"{target_norm * 1000.0:9.3f} {offset_norm * 1000.0:9.3f} {pose_norm * 1000.0:8.3f}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze RM65 pose admittance smoke-test latency from a JSON report.")
    parser.add_argument(
        "--input-json",
        default="hardware/rm65/utils/results/rm65_left_pose_admittance_smoke.json",
    )
    parser.add_argument("--force-threshold-n", type=float, default=0.50)
    parser.add_argument("--active-threshold-n", type=float, default=0.05)
    parser.add_argument("--offset-threshold-m", type=float, default=0.001)
    parser.add_argument("--pose-threshold-m", type=float, default=0.001)
    parser.add_argument("--window-steps", type=int, default=12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    result = _analyze(data, args)
    records = result["records"]
    events = result["events"]
    latencies = result["latencies"]
    print(f"input={args.input_json}")
    print(
        f"run hz={data.get('control_hz')} duration={data.get('duration_sec')} "
        f"records={len(records)} missed={data.get('missed_deadlines')}"
    )
    print(f"events={events}")
    print(
        "latency "
        f"force->active={_fmt(latencies['force_to_active_ms'])} "
        f"force->offset={_fmt(latencies['force_to_offset_ms'])} "
        f"force->pose={_fmt(latencies['force_to_pose_ms'])} "
        f"active->offset={_fmt(latencies['active_to_offset_ms'])} "
        f"target->offset={_fmt(latencies['target_to_offset_ms'])} "
        f"target->pose={_fmt(latencies['target_to_pose_ms'])} "
        f"offset->pose={_fmt(latencies['offset_to_pose_ms'])}"
    )
    if result["timing_summary"]:
        print("timing summary:")
        for name, stats in result["timing_summary"].items():
            print(
                f"  {name:12s} mean/p95/p99/max="
                f"{stats['mean_ms']:.3f}/{stats['p95_ms']:.3f}/"
                f"{stats['p99_ms']:.3f}/{stats['max_ms']:.3f} ms"
            )
        print(f"  missed_deadlines_from_records={result['timing_missed_deadlines']}")
    else:
        print("timing summary: unavailable in this report; rerun with the updated smoke script.")
    force_event = events.get("force_threshold")
    if force_event:
        _print_window(records, int(force_event["step"]), args.window_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

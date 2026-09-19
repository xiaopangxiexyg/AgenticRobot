"""Summarize RM65 observation to B601 MuJoCo target-dispatch latency.

The input is the JSONL ``gui.log`` emitted by
``rm65-b601-free-drive``.  This is an offline report only: it does not open a
RM65 connection, a CAN interface, or a MuJoCo viewer.  The primary latency is
measured from the accepted RM65 hardware-read timestamp to the tick where the
MuJoCo position driver accepts the retargeted qpos.  It excludes physical
actuator response and the RM65 measurement age before the read completed.

The report also exposes the post-validation ``source admission`` timestamp.
That second latency separates source/backend state assembly from the actual
target dispatch path while retaining the original field for compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _finite_non_negative(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0.0:
        return None
    return normalized


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (
        position - lower
    )


def _statistics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50_ms": _scale_ms(_percentile(values, 0.50)),
        "p95_ms": _scale_ms(_percentile(values, 0.95)),
        "p99_ms": _scale_ms(_percentile(values, 0.99)),
        "max_ms": _scale_ms(max(values) if values else None),
        "mean_ms": _scale_ms(sum(values) / len(values) if values else None),
    }


def _statistics_ms(values: list[float]) -> dict[str, float | int | None]:
    """Summarize values that the RM65 backend already reports in milliseconds."""

    if not values:
        return {
            "count": 0,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "mean_ms": None,
        }
    return {
        "count": len(values),
        "p50_ms": round(_percentile(values, 0.50) or 0.0, 3),
        "p95_ms": round(_percentile(values, 0.95) or 0.0, 3),
        "p99_ms": round(_percentile(values, 0.99) or 0.0, 3),
        "max_ms": round(max(values), 3),
        "mean_ms": round(sum(values) / len(values), 3),
    }


def _scale_ms(value: float | None) -> float | None:
    return None if value is None else round(value * 1000.0, 3)


def _diagnostics(values: list[float]) -> dict[str, float | int | None]:
    return {
        "min_ms": _scale_ms(min(values) if values else None),
        "over_10ms_count": sum(value > 0.010 for value in values),
        "over_20ms_count": sum(value > 0.020 for value in values),
    }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    timings: dict[str, list[float]] = {
        "source_to_target_dispatch_latency_sec": [],
        "source_admission_to_target_dispatch_latency_sec": [],
        "source_completion_to_consumption_latency_sec": [],
        "source_completion_to_target_dispatch_latency_sec": [],
        "source_interarrival_sec": [],
        "source_admission_interarrival_sec": [],
        "source_poll_lateness_sec": [],
        "source_step_elapsed_sec": [],
        "retarget_elapsed_sec": [],
        "position_step_elapsed_sec": [],
        "step_elapsed_sec": [],
    }
    source_read_phases_ms: dict[str, list[float]] = defaultdict(list)
    counts: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    previous_source_read_monotonic: float | None = None
    previous_source_admission_monotonic: float | None = None
    for record in records:
        if record.get("event") != "bridge_step":
            continue
        counts["bridge_steps"] += 1
        if record.get("source_admitted") is True:
            counts["source_admitted"] += 1
            source_read_monotonic = _finite_non_negative(
                record.get("source_state_read_timestamp_monotonic")
            )
            if (
                source_read_monotonic is not None
                and previous_source_read_monotonic is not None
            ):
                interval = source_read_monotonic - previous_source_read_monotonic
                if math.isfinite(interval) and interval >= 0.0:
                    timings["source_interarrival_sec"].append(interval)
            if source_read_monotonic is not None:
                previous_source_read_monotonic = source_read_monotonic
            source_admission_monotonic = _finite_non_negative(
                record.get("source_admission_monotonic_sec")
            )
            if (
                source_admission_monotonic is not None
                and previous_source_admission_monotonic is not None
            ):
                interval = (
                    source_admission_monotonic
                    - previous_source_admission_monotonic
                )
                if math.isfinite(interval) and interval >= 0.0:
                    timings["source_admission_interarrival_sec"].append(interval)
            if source_admission_monotonic is not None:
                previous_source_admission_monotonic = source_admission_monotonic
        if record.get("retarget_accepted") is True:
            counts["retarget_accepted"] += 1
        if record.get("command_applied") is True:
            counts["target_dispatches"] += 1
        if record.get("sim_slew_limited") is True:
            counts["sim_slew_limited"] += 1
        if record.get("source_poll_schedule_skipped") is True:
            counts["source_poll_schedule_skipped"] += 1
        reason = record.get("bridge_rejection_reason") or record.get(
            "retarget_rejection_reason"
        )
        if reason:
            rejection_reasons[str(reason)] += 1
        for name, values in timings.items():
            if name in {
                "source_step_elapsed_sec",
                "retarget_elapsed_sec",
            } and record.get("source_admitted") is None:
                continue
            # A background source result is represented by one bridge record
            # when it is consumed and another when its queued command is
            # dispatched.  Keep each phase tied to its owning event so the
            # reported sample counts remain comparable to source/dispatch
            # counts.
            if (
                name == "source_completion_to_consumption_latency_sec"
                and record.get("source_admitted") is None
            ):
                continue
            if (
                name == "source_completion_to_target_dispatch_latency_sec"
                and record.get("command_applied") is not True
            ):
                continue
            value = _finite_non_negative(record.get(name))
            if value is not None:
                values.append(value)
        if "source_admitted" in record:
            phase_timing = record.get("source_state_read_phase_timing")
            if isinstance(phase_timing, dict):
                for name, value in phase_timing.items():
                    if not str(name).endswith("_ms"):
                        continue
                    normalized = _finite_non_negative(value)
                    if normalized is not None:
                        source_read_phases_ms[str(name)].append(normalized)
    return {
        "schema": "rm65_b601_reverse_latency_report_v1",
        "counts": dict(sorted(counts.items())),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "timings": {
            name: _statistics(values) for name, values in timings.items()
        },
        "source_read_phase_timings_ms": {
            name: _statistics_ms(values)
            for name, values in sorted(source_read_phases_ms.items())
        },
        "diagnostics": {
            name: _diagnostics(values)
            for name, values in timings.items()
            if name
            in {
                "source_to_target_dispatch_latency_sec",
                "source_admission_to_target_dispatch_latency_sec",
                "source_poll_lateness_sec",
            }
        },
        "interpretation": {
            "primary_latency": (
                "source_to_target_dispatch_latency_sec is the interval from "
                "the accepted RM65 hardware-read completion timestamp to "
                "B601 MuJoCo position-driver command acceptance."
            ),
            "admission_latency": (
                "source_admission_to_target_dispatch_latency_sec starts when "
                "the source step has completed freshness/frame validation and "
                "ends at B601 MuJoCo position-driver command acceptance."
            ),
            "worker_queue_latency": (
                "source_completion_to_consumption_latency_sec is the delay "
                "between completion of the RM65 source read and consumption "
                "by the MuJoCo bridge loop."
            ),
            "excluded": (
                "RM65 measurement age before read completion, physical B601 "
                "actuator response, and viewer presentation latency."
            ),
        },
    }


def read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL") from exc
                if not isinstance(record, dict):
                    raise TypeError(f"{path}:{line_number}: expected a JSON object")
                yield record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = json.dumps(
        summarize(read_jsonl(args.jsonl)),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output is None:
        print(report, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

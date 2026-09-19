"""Summarize B601 hardware to RM65 MuJoCo forward-loop latency.

The input is the JSONL application log (usually ``gui.log``) emitted by
``control_station b601-rm65 --connect-read-only``; ``terminal.log`` is only
the human-readable stdout/stderr capture.  This is an offline report:
it never opens CAN, MotorBridge, or MuJoCo.  The primary latency is measured
from the accepted B601 host-receive timestamp to the RM65 target dispatch
timestamp in the same foreground tick.  Physical B601 feedback age before the
host receive cut and MuJoCo viewer presentation are reported separately or
excluded explicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_TIMING_FIELDS = (
    "source_poll_elapsed_sec",
    "feedback_poll_elapsed_sec",
    "feedback_cache_age_sec",
    "fk_elapsed_sec",
    "emission_elapsed_sec",
    "source_adaptation_elapsed_sec",
    "proposal_submission_elapsed_sec",
    "session_step_elapsed_sec",
    "target_step_elapsed_sec",
    "target_resolution_elapsed_sec",
    "target_pose_transform_elapsed_sec",
    "target_pose_smoothing_elapsed_sec",
    "target_pose_safety_filter_elapsed_sec",
    "target_pre_ik_elapsed_sec",
    "target_ik_elapsed_sec",
    "target_post_ik_elapsed_sec",
    "target_resolution_build_elapsed_sec",
    "target_command_resolve_elapsed_sec",
    "target_command_metadata_elapsed_sec",
    "target_resolution_cache_insert_elapsed_sec",
    "target_resolution_cache_release_elapsed_sec",
    "target_post_resolution_elapsed_sec",
    "source_to_target_dispatch_latency_sec",
    "source_interarrival_sec",
)


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
    def milliseconds(value: float | None) -> float | None:
        return None if value is None else round(value * 1000.0, 3)

    return {
        "count": len(values),
        "p50_ms": milliseconds(_percentile(values, 0.50)),
        "p95_ms": milliseconds(_percentile(values, 0.95)),
        "p99_ms": milliseconds(_percentile(values, 0.99)),
        "max_ms": milliseconds(max(values) if values else None),
        "mean_ms": milliseconds(sum(values) / len(values) if values else None),
    }


def _integer_summary(values: list[int]) -> dict[str, int]:
    """Summarize non-negative runtime counters without converting to time."""

    if not values:
        return {
            "count": 0,
            "first": 0,
            "last": 0,
            "min": 0,
            "max": 0,
        }
    return {
        "count": len(values),
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
    }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    timings = {name: [] for name in _TIMING_FIELDS}
    counts: Counter[str] = Counter()
    target_statuses: Counter[str] = Counter()
    target_resolution_statuses: Counter[str] = Counter()
    target_ik_statuses: Counter[str] = Counter()
    resolution_failures: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    resolution_cache_sizes_before: list[int] = []
    resolution_cache_sizes_after: list[int] = []
    resolution_cache_sizes_after_release: list[int] = []
    gc_count_before: list[tuple[int, int, int]] = []
    gc_count_after: list[tuple[int, int, int]] = []
    previous_source_receive: float | None = None

    for record in records:
        if record.get("event") != "bridge_step":
            continue
        counts["bridge_steps"] += 1
        if record.get("source_admitted") is True:
            counts["source_admitted"] += 1
            source_receive = _finite_non_negative(
                record.get("source_receive_timestamp_sec")
            )
            if source_receive is not None and previous_source_receive is not None:
                interval = source_receive - previous_source_receive
                if math.isfinite(interval) and interval >= 0.0:
                    timings["source_interarrival_sec"].append(interval)
            if source_receive is not None:
                previous_source_receive = source_receive
        if record.get("source_reused") is True:
            counts["source_reused"] += 1
        if record.get("proposal_accepted") is True:
            counts["proposal_accepted"] += 1
        if record.get("target_dispatch_accepted") is True:
            counts["target_dispatches"] += 1
        if record.get("target_resolution_invoked") is True:
            counts["target_resolution_invocations"] += 1
        resolution_status = record.get("target_resolution_status")
        if resolution_status:
            target_resolution_statuses[str(resolution_status)] += 1
        ik_status = record.get("target_ik_status")
        if ik_status:
            target_ik_statuses[str(ik_status)] += 1
        status = record.get("target_status")
        if status:
            target_statuses[str(status)] += 1
        rejection = record.get("proposal_rejection_code")
        if rejection:
            rejection_reasons[str(rejection)] += 1
        failure = record.get("target_resolution_failure")
        if isinstance(failure, dict) and failure.get("reason_code"):
            resolution_failures[str(failure["reason_code"])] += 1
        feedback_snapshot = record.get("feedback_snapshot")
        if isinstance(feedback_snapshot, dict):
            feedback_age = _finite_non_negative(
                feedback_snapshot.get("max_receive_age_sec")
            )
            if feedback_age is not None:
                timings["feedback_cache_age_sec"].append(feedback_age)
        for name in _TIMING_FIELDS:
            if name in {"source_interarrival_sec", "source_to_target_dispatch_latency_sec"}:
                continue
            value = _finite_non_negative(record.get(name))
            if value is not None:
                timings[name].append(value)
        for field_name, destination in (
            (
                "target_resolution_cache_size_before",
                resolution_cache_sizes_before,
            ),
            (
                "target_resolution_cache_size_after",
                resolution_cache_sizes_after,
            ),
            (
                "target_resolution_cache_size_after_release",
                resolution_cache_sizes_after_release,
            ),
        ):
            value = record.get(field_name)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                destination.append(value)
        for field_name, destination in (
            ("target_gc_count_before", gc_count_before),
            ("target_gc_count_after", gc_count_after),
        ):
            value = record.get(field_name)
            if (
                isinstance(value, (list, tuple))
                and len(value) == 3
                and all(
                    isinstance(item, int)
                    and not isinstance(item, bool)
                    and item >= 0
                    for item in value
                )
            ):
                destination.append(tuple(int(item) for item in value))
        source_receive = _finite_non_negative(
            record.get("source_receive_timestamp_sec")
        )
        target_dispatch = _finite_non_negative(
            record.get("target_dispatch_monotonic_sec")
        )
        if (
            record.get("target_dispatch_accepted") is True
            and source_receive is not None
            and target_dispatch is not None
            and target_dispatch >= source_receive
        ):
            timings["source_to_target_dispatch_latency_sec"].append(
                target_dispatch - source_receive
            )

    return {
        "schema": "b601_rm65_forward_latency_report_v3",
        "counts": dict(sorted(counts.items())),
        "target_statuses": dict(sorted(target_statuses.items())),
        "target_resolution_statuses": dict(
            sorted(target_resolution_statuses.items())
        ),
        "target_ik_statuses": dict(sorted(target_ik_statuses.items())),
        "proposal_rejection_reasons": dict(sorted(rejection_reasons.items())),
        "target_resolution_failure_reasons": dict(
            sorted(resolution_failures.items())
        ),
        "timings": {
            name: _statistics(values) for name, values in timings.items()
        },
        "runtime_diagnostics": {
            "resolution_cache_size_before": _integer_summary(
                resolution_cache_sizes_before
            ),
            "resolution_cache_size_after": _integer_summary(
                resolution_cache_sizes_after
            ),
            "resolution_cache_size_after_release": _integer_summary(
                resolution_cache_sizes_after_release
            ),
            "gc_count_before": {
                f"generation_{generation}": _integer_summary(
                    [sample[generation] for sample in gc_count_before]
                )
                for generation in range(3)
            },
            "gc_count_after": {
                f"generation_{generation}": _integer_summary(
                    [sample[generation] for sample in gc_count_after]
                )
                for generation in range(3)
            },
        },
        "interpretation": {
            "primary_latency": (
                "source_to_target_dispatch_latency_sec measures from the B601 "
                "host receive cut to RM65 MuJoCo target dispatch acceptance."
            ),
            "source_poll": (
                "source_poll_elapsed_sec is the complete host-fresh source "
                "projection; feedback_poll_elapsed_sec, fk_elapsed_sec and "
                "emission_elapsed_sec split its internal phases."
            ),
            "feedback_cache_age": (
                "feedback_cache_age_sec is the maximum per-joint host receive "
                "age reported by MotorBridge at the foreground tick; it is "
                "separate from source-to-target dispatch work."
            ),
            "target_resolution": (
                "target_resolution_elapsed_sec is the canonical MuJoCo pose "
                "resolver duration. Its pre_ik/pose-transform/smoothing/filter "
                "fields expose the preparation stages; target_ik_elapsed_sec "
                "is the coupled RM65 IK solver duration and "
                "target_post_ik_elapsed_sec is the complete post-IK segment. "
                "The resolution_build, command_resolve, command_metadata and "
                "cache insert/release fields split that segment further; "
                "runtime_diagnostics reports cache growth and Python GC counters. "
                "target_post_resolution_elapsed_sec "
                "is the remainder of the target step (dispatch, physics and "
                "measurement/trace work) after the resolver."
            ),
            "excluded": (
                "B601 feedback age before host receipt and MuJoCo viewer "
                "presentation latency are not included in the primary value."
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

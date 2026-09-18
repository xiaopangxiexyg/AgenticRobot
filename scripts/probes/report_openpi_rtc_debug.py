"""Summarize passive RTC timing evidence from deployment debug JSONL.

This report does not decide admission, readiness, safety, or action authority.
Cold/bootstrap and conditioned requests are deliberately kept separate.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _statistics(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "p50_ms": round(_percentile(values, 0.5) * 1000, 3),
        "p95_ms": round(_percentile(values, 0.95) * 1000, 3),
        "max_ms": round(max(values) * 1000, 3),
    }


def summarize(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    samples: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        event = record.get("event")
        if event == "rtc_transport_exchange":
            mode = record.get("conditioning_mode")
            if mode == "unconditioned_bootstrap_v1":
                group = "bootstrap"
            elif mode == "conditioned_prefix_v1":
                group = "conditioned"
            else:
                # Older logs lack the mode; zero prefix was the bootstrap path.
                group = (
                    "conditioned"
                    if int(record.get("selected_prefix_steps") or 0)
                    else "bootstrap"
                )
            counts[group]["exchanges"] += 1
            for source, target in (
                ("inference_duration_sec", "provider_exchange"),
                ("result_duration_sec", "provider_result"),
            ):
                value = record.get(source)
                if (
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value >= 0
                ):
                    samples[group][target].append(float(value))
            timing = record.get("client_timing_sec")
            if isinstance(timing, dict):
                for name, value in timing.items():
                    if (
                        isinstance(value, (int, float))
                        and math.isfinite(value)
                        and value >= 0
                    ):
                        samples[group][f"client_{name}"].append(float(value))
        elif (
            event == "rtc_frontend_step" and record.get("completion_status") is not None
        ):
            group = "preflight" if record.get("preflight_completion") else "active"
            counts[group]["completions"] += 1
            start = record.get("completion_worker_started_at")
            end = record.get("completion_worker_completed_at")
            release = record.get("completion_released_at")
            if (
                all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in (start, end)
                )
                and end >= start
            ):
                samples[group]["worker_execution"].append(float(end - start))
            if (
                all(
                    isinstance(v, (int, float)) and math.isfinite(v)
                    for v in (start, release)
                )
                and release >= start
            ):
                samples[group]["worker_to_release"].append(float(release - start))
            if record.get("plan_merge_accepted"):
                counts[group]["accepted_merges"] += 1
            counts[group]["admitted_keyframes"] += int(
                record.get("plan_merge_admitted_keyframe_count") or 0
            )
    return {
        group: {
            "counts": dict(counts[group]),
            "timings": {
                name: _statistics(values)
                for name, values in sorted(samples[group].items())
                if values
            },
        }
        for group in sorted(counts)
    }


def read_jsonl(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"{path}:{line_number}: invalid JSONL"
                        ) from exc
                    if not isinstance(record, dict):
                        raise ValueError(f"{path}:{line_number}: expected object")
                    yield record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(read_jsonl(args.jsonl)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

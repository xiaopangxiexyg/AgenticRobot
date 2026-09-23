"""Summarize teleop trace cadence in the monotonic window of one CPU capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _capture_window(capture_dir: Path) -> tuple[int, int]:
    rows = [
        json.loads(line)
        for line in (capture_dir / "procfs_cpu.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) < 2:
        raise ValueError("Capture needs at least two procfs samples")
    return int(rows[0]["monotonic_ns"]), int(rows[-1]["monotonic_ns"]) + 1_000_000_000


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    ms = values.astype(np.float64) / 1_000_000.0
    return {
        "count": int(ms.size),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "max_ms": float(ms.max()),
    }


def _stream_summary(
    rows: np.ndarray,
    timestamp_key: str,
    *,
    start_ns: int,
    end_ns: int,
    expected_hz: float | None,
    scheduled_key: str | None = None,
) -> dict[str, object]:
    if timestamp_key not in (rows.dtype.names or ()):
        raise ValueError(f"Trace stream lacks {timestamp_key}")
    selected = rows[(rows[timestamp_key] >= start_ns) & (rows[timestamp_key] < end_ns)]
    timestamps = selected[timestamp_key]
    gaps = np.diff(timestamps)
    summary: dict[str, object] = {
        "rows": int(len(selected)),
        "gap": _distribution(gaps),
        "non_monotonic_gap_count": int(np.count_nonzero(gaps <= 0)),
    }
    if expected_hz is not None:
        period_ns = 1e9 / expected_hz
        summary["expected_hz"] = expected_hz
        summary["gaps_over_1_5_period_count"] = int(np.count_nonzero(gaps > 1.5 * period_ns))
        summary["gaps_over_2_period_count"] = int(np.count_nonzero(gaps > 2 * period_ns))
    if scheduled_key is not None:
        summary["wakeup_lateness"] = _distribution(
            selected[timestamp_key] - selected[scheduled_key]
        )
    return summary


def summarize(trace_dir: Path, capture_dir: Path | None = None) -> dict[str, object]:
    manifest = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "dexteleop.rm65_actuator_command_trace.v4":
        raise ValueError("Unsupported actuator trace schema")
    start_ns, end_ns = (
        _capture_window(capture_dir)
        if capture_dir is not None
        else (int(manifest["started_monotonic_time_ns"]), int(manifest["stopped_monotonic_time_ns"]))
    )
    if end_ns <= start_ns:
        raise ValueError("Invalid capture window")
    config = manifest["config"]
    report: dict[str, object] = {
        "trace_dir": str(trace_dir),
        "capture_dir": None if capture_dir is None else str(capture_dir),
        "window_start_monotonic_ns": start_ns,
        "window_end_monotonic_ns": end_ns,
        "duration_sec": (end_ns - start_ns) / 1e9,
        "trace_result": manifest.get("result"),
        "streams": {},
    }
    with np.load(trace_dir / "command_trace.npz", allow_pickle=False) as data:
        streams = report["streams"]
        assert isinstance(streams, dict)
        for side in config["active_sides"]:
            streams[f"{side}_command"] = _stream_summary(
                data[side],
                "wakeup_monotonic_ns",
                start_ns=start_ns,
                end_ns=end_ns,
                expected_hz=float(config["command_hz"]),
                scheduled_key="scheduled_monotonic_ns",
            )
        streams["ik_shadow"] = _stream_summary(
            data["ik_shadow"],
            "monotonic_time_ns",
            start_ns=start_ns,
            end_ns=end_ns,
            expected_hz=float(config["reference_hz"]),
        )
        streams["vive_source"] = _stream_summary(
            data["vive_source"],
            "control_monotonic_time_ns",
            start_ns=start_ns,
            end_ns=end_ns,
            expected_hz=None,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.trace_dir, args.capture_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

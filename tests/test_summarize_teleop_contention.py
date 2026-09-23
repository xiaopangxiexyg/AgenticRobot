from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/probes/summarize_teleop_contention.py"
)
_SPEC = importlib.util.spec_from_file_location("summarize_teleop_contention", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_summarizes_only_capture_window(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    capture = tmp_path / "capture"
    trace.mkdir()
    capture.mkdir()
    (trace / "manifest.json").write_text(
        json.dumps({
            "schema": "dexteleop.rm65_actuator_command_trace.v4",
            "started_monotonic_time_ns": 1,
            "stopped_monotonic_time_ns": 8_000_000_000,
            "config": {"active_sides": ["left"], "command_hz": 2, "reference_hz": 2},
            "result": {"drop_count": 0},
        }),
        encoding="utf-8",
    )
    (capture / "procfs_cpu.jsonl").write_text(
        '{"monotonic_ns":1000000000}\n{"monotonic_ns":2000000000}\n',
        encoding="utf-8",
    )
    command = np.zeros(4, dtype=[("wakeup_monotonic_ns", "<i8"), ("scheduled_monotonic_ns", "<i8")])
    command["wakeup_monotonic_ns"] = [500_000_000, 1_000_000_000, 1_500_000_000, 2_500_000_000]
    command["scheduled_monotonic_ns"] = command["wakeup_monotonic_ns"] - 2_000_000
    ik = np.zeros(3, dtype=[("monotonic_time_ns", "<i8")])
    ik["monotonic_time_ns"] = [1_100_000_000, 1_600_000_000, 2_100_000_000]
    vive = np.zeros(2, dtype=[("control_monotonic_time_ns", "<i8")])
    vive["control_monotonic_time_ns"] = [1_200_000_000, 2_200_000_000]
    np.savez(trace / "command_trace.npz", left=command, ik_shadow=ik, vive_source=vive)

    report = _MODULE.summarize(trace, capture)
    assert report["streams"]["left_command"]["rows"] == 3
    assert report["streams"]["left_command"]["gap"]["max_ms"] == 1000.0
    assert report["streams"]["left_command"]["gaps_over_1_5_period_count"] == 1
    assert report["streams"]["left_command"]["wakeup_lateness"]["p99_ms"] == 2.0
    assert report["streams"]["ik_shadow"]["rows"] == 3

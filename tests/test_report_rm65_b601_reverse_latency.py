from __future__ import annotations

from scripts.probes.report_rm65_b601_reverse_latency import summarize


def test_reverse_latency_report_uses_dispatch_samples_and_rejections() -> None:
    report = summarize(
        [
            {
                "event": "bridge_step",
                "source_admitted": True,
                "retarget_accepted": True,
                "command_applied": True,
                "source_to_target_dispatch_latency_sec": 0.004,
                "source_admission_to_target_dispatch_latency_sec": 0.001,
                "source_admission_monotonic_sec": 10.003,
                "source_poll_lateness_sec": 0.002,
                "source_step_elapsed_sec": 0.006,
                "retarget_elapsed_sec": 0.001,
                "position_step_elapsed_sec": 0.0002,
                "step_elapsed_sec": 0.0075,
                "sim_slew_limited": True,
                "source_state_read_phase_timing": {
                    "total_wall_ms": 6.0,
                    "sdk_call_wall_ms": 5.0,
                },
                "source_state_read_timestamp_monotonic": 10.0,
            },
            {
                "event": "bridge_step",
                "source_admitted": True,
                "retarget_accepted": False,
                "command_applied": False,
                "bridge_rejection_reason": "ik_rejected:line_search_stalled",
                "source_state_read_timestamp_monotonic": 10.05,
            },
            {"event": "terminal", "message": "ignored"},
        ]
    )

    assert report["counts"] == {
        "bridge_steps": 2,
        "retarget_accepted": 1,
        "sim_slew_limited": 1,
        "source_admitted": 2,
        "target_dispatches": 1,
    }
    assert report["rejection_reasons"] == {
        "ik_rejected:line_search_stalled": 1,
    }
    assert report["source_read_phase_timings_ms"]["total_wall_ms"]["p95_ms"] == 6.0
    assert report["timings"]["source_interarrival_sec"]["p50_ms"] == 50.0
    assert report["timings"]["source_to_target_dispatch_latency_sec"] == {
        "count": 1,
        "mean_ms": 4.0,
        "max_ms": 4.0,
        "p50_ms": 4.0,
        "p95_ms": 4.0,
        "p99_ms": 4.0,
    }
    assert report["timings"]["source_admission_to_target_dispatch_latency_sec"] == {
        "count": 1,
        "mean_ms": 1.0,
        "max_ms": 1.0,
        "p50_ms": 1.0,
        "p95_ms": 1.0,
        "p99_ms": 1.0,
    }
    assert report["diagnostics"]["source_to_target_dispatch_latency_sec"] == {
        "min_ms": 4.0,
        "over_10ms_count": 0,
        "over_20ms_count": 0,
    }

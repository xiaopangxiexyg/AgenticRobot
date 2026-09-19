from __future__ import annotations

from scripts.probes.report_b601_rm65_forward_latency import summarize


def test_forward_latency_report_correlates_source_and_target_dispatch() -> None:
    report = summarize(
        [
            {
                "event": "bridge_step",
                "source_admitted": True,
                "source_reused": False,
                "source_receive_timestamp_sec": 10.000,
                "source_poll_elapsed_sec": 0.004,
                "source_adaptation_elapsed_sec": 0.001,
                "proposal_submission_elapsed_sec": 0.0005,
                "session_step_elapsed_sec": 0.006,
                "proposal_accepted": True,
                "target_dispatch_accepted": True,
                "target_dispatch_monotonic_sec": 10.012,
                "target_step_elapsed_sec": 0.008,
                "target_status": "dispatched",
                "feedback_snapshot": {"max_receive_age_sec": 0.025},
            },
            {
                "event": "bridge_step",
                "source_admitted": True,
                "source_reused": True,
                "source_receive_timestamp_sec": 10.050,
                "proposal_accepted": False,
                "proposal_rejection_code": "b601_motion_rejected",
                "target_dispatch_accepted": False,
                "target_status": "held",
                "target_resolution_failure": {
                    "reason_code": "ik_rejected:line_search_stalled"
                },
            },
        ]
    )

    assert report["counts"] == {
        "bridge_steps": 2,
        "proposal_accepted": 1,
        "source_admitted": 2,
        "source_reused": 1,
        "target_dispatches": 1,
    }
    assert report["proposal_rejection_reasons"] == {
        "b601_motion_rejected": 1,
    }
    assert report["target_resolution_failure_reasons"] == {
        "ik_rejected:line_search_stalled": 1,
    }
    assert report["timings"]["source_to_target_dispatch_latency_sec"] == {
        "count": 1,
        "mean_ms": 12.0,
        "max_ms": 12.0,
        "p50_ms": 12.0,
        "p95_ms": 12.0,
        "p99_ms": 12.0,
    }
    assert report["timings"]["source_interarrival_sec"]["p50_ms"] == 50.0
    assert report["timings"]["feedback_cache_age_sec"]["p50_ms"] == 25.0

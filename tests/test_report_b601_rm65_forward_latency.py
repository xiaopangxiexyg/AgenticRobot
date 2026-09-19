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
                "target_resolution_invoked": True,
                "target_resolution_elapsed_sec": 0.004,
                "target_ik_elapsed_sec": 0.003,
                "target_post_ik_elapsed_sec": 0.0008,
                "target_resolution_build_elapsed_sec": 0.0004,
                "target_command_resolve_elapsed_sec": 0.0002,
                "target_command_metadata_elapsed_sec": 0.0001,
                "target_resolution_cache_insert_elapsed_sec": 0.00005,
                "target_resolution_cache_size_before": 4,
                "target_resolution_cache_size_after": 5,
                "target_gc_count_before": [10, 2, 0],
                "target_gc_count_after": [12, 2, 0],
                "target_post_resolution_elapsed_sec": 0.004,
                "target_resolution_status": "accepted",
                "target_ik_status": "tracking",
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
        "target_resolution_invocations": 1,
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
    assert report["counts"]["target_resolution_invocations"] == 1
    assert report["target_resolution_statuses"] == {"accepted": 1}
    assert report["target_ik_statuses"] == {"tracking": 1}
    assert report["timings"]["target_ik_elapsed_sec"]["p50_ms"] == 3.0
    assert (
        report["timings"]["target_resolution_build_elapsed_sec"]["p50_ms"]
        == 0.4
    )
    assert (
        report["timings"]["target_command_resolve_elapsed_sec"]["p50_ms"]
        == 0.2
    )
    assert report["runtime_diagnostics"]["resolution_cache_size_before"] == {
        "count": 1,
        "first": 4,
        "last": 4,
        "min": 4,
        "max": 4,
    }
    assert report["runtime_diagnostics"]["resolution_cache_size_after"] == {
        "count": 1,
        "first": 5,
        "last": 5,
        "min": 5,
        "max": 5,
    }
    assert (
        report["runtime_diagnostics"]["gc_count_after"]["generation_0"]["last"]
        == 12
    )

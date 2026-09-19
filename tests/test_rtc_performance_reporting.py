from __future__ import annotations

import json

import pytest

from deployments.client.rtc_conditioned_client import RtcConditionedClientTiming
from scripts.probes.report_openpi_rtc_debug import read_jsonl, summarize


def test_report_separates_bootstrap_conditioned_and_preflight() -> None:
    records = [
        {
            "event": "rtc_transport_exchange",
            "conditioning_mode": "unconditioned_bootstrap_v1",
            "selected_prefix_steps": 0,
            "inference_duration_sec": 0.47,
        },
        {
            "event": "rtc_transport_exchange",
            "conditioning_mode": "conditioned_prefix_v1",
            "selected_prefix_steps": 3,
            "inference_duration_sec": 0.35,
            "client_timing_sec": {
                "encode_envelope": 0.01,
                "backend_call": 0.34,
                "result_decode": 0.02,
            },
        },
        {
            "event": "rtc_frontend_step",
            "preflight_completion": True,
            "completion_status": "ready",
            "completion_worker_started_at": 10.0,
            "completion_worker_completed_at": 10.55,
            "completion_released_at": 10.56,
            "plan_merge_accepted": False,
            "plan_merge_admitted_keyframe_count": 0,
        },
    ]

    report = summarize(records)

    assert report["bootstrap"]["timings"]["provider_exchange"]["p95_ms"] == 470
    assert report["conditioned"]["timings"]["provider_exchange"]["p50_ms"] == 350
    assert report["conditioned"]["timings"]["client_encode_envelope"]["max_ms"] == 10
    assert report["preflight"]["timings"]["worker_execution"]["max_ms"] == 550
    assert report["preflight"]["counts"]["admitted_keyframes"] == 0


def test_jsonl_reader_identifies_corrupt_line(tmp_path) -> None:
    path = tmp_path / "deployment.jsonl"
    path.write_text(json.dumps({"event": "rtc_transport_exchange"}) + "\n{broken\n")
    with pytest.raises(ValueError, match=r"deployment.jsonl:2: invalid JSONL"):
        list(read_jsonl([path]))


def test_client_timing_is_passive_value_only() -> None:
    timing = RtcConditionedClientTiming(0.001, 0.002, 0.4, 0.003, 0.004)
    assert timing.backend_call_sec == 0.4

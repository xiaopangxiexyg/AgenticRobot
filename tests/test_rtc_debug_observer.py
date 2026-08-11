from __future__ import annotations

import threading

import pytest

from teleop_runtime.deploy.rtc_debug import (
    RTC_DEBUG_LEVEL_SUMMARY,
    RTC_DEBUG_LEVEL_TRACE,
    RtcDebugEventObserver,
)
from teleop_runtime.deploy.run_policy import DeploymentDebugLogger


def test_summary_observer_filters_trace_and_drains_without_blocking() -> None:
    records: list[tuple[str, dict[str, object]]] = []
    observer = RtcDebugEventObserver(
        level=RTC_DEBUG_LEVEL_SUMMARY,
        sink=lambda event, payload: records.append((event, dict(payload))),
        queue_capacity=4,
    )
    assert observer.accepts(RTC_DEBUG_LEVEL_SUMMARY)
    assert not observer.accepts(RTC_DEBUG_LEVEL_TRACE)
    assert not observer.offer(
        "rtc_tick",
        {"tick": 1},
        minimum_level=RTC_DEBUG_LEVEL_TRACE,
    )
    assert observer.offer("rtc_transition", {"transition": "request_admitted"})
    status = observer.close()

    assert records == [
        (
            "rtc_transition",
            {
                "transition": "request_admitted",
                "non_authoritative": True,
            },
        )
    ]
    assert status.written_event_count == 1
    assert status.dropped_event_count == 0
    assert status.closed


def test_full_debug_queue_drops_diagnostics_without_blocking_producer() -> None:
    sink_entered = threading.Event()
    release_sink = threading.Event()
    records: list[int] = []

    def blocking_sink(_event: str, payload: dict[str, object]) -> None:
        sink_entered.set()
        assert release_sink.wait(timeout=2.0)
        records.append(int(payload["sequence"]))

    observer = RtcDebugEventObserver(
        level=RTC_DEBUG_LEVEL_TRACE,
        sink=blocking_sink,
        queue_capacity=1,
    )
    assert observer.offer("rtc_tick", {"sequence": 0})
    assert sink_entered.wait(timeout=1.0)
    assert observer.offer("rtc_tick", {"sequence": 1})
    assert not observer.offer("rtc_tick", {"sequence": 2})
    release_sink.set()
    status = observer.close()

    assert records == [0, 1]
    assert status.offered_event_count == 3
    assert status.written_event_count == 2
    assert status.dropped_event_count == 1


def test_debug_sink_failure_is_observational_only() -> None:
    def failing_sink(_event: str, _payload: dict[str, object]) -> None:
        raise RuntimeError("synthetic debug sink failure")

    observer = RtcDebugEventObserver(
        level=RTC_DEBUG_LEVEL_SUMMARY,
        sink=failing_sink,
        queue_capacity=2,
    )
    assert observer.offer("rtc_fault", {"fault": "test"})
    status = observer.close()

    assert status.sink_error_count == 1
    assert "synthetic debug sink failure" in status.last_sink_error


@pytest.mark.integration
def test_rtc_observer_and_batched_jsonl_flush_on_close(tmp_path) -> None:
    logger = DeploymentDebugLogger(
        enabled=True,
        root_dir=tmp_path,
        active_sides=("right",),
        flush_every_events=16,
        flush_interval_sec=10.0,
    )
    assert logger.path is not None
    observer = RtcDebugEventObserver(
        level=RTC_DEBUG_LEVEL_SUMMARY,
        sink=logger.log_event,
        queue_capacity=4,
    )
    assert observer.offer(
        "rtc_emission_publication",
        {"publication_key": "publication:test"},
    )
    observer.close()
    path = logger.path
    logger.close()

    text = path.read_text(encoding="utf-8")
    assert '"event":"rtc_emission_publication"' in text
    assert '"publication_key":"publication:test"' in text

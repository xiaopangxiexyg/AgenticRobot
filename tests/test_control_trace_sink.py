from __future__ import annotations

import threading
from pathlib import Path

import pytest

from dataset_contracts.control_trace import ControlTraceFact
from dataset_contracts.control_trace_wal import encode_control_trace_wal_frame
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    ControlTraceSpool,
    ControlTraceSpoolSink,
    ControlTraceSpoolWorker,
    ControlTraceWriterHealthLedger,
    RuntimeTraceSession,
)


def runtime() -> ControlTraceRuntime:
    return ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-1",
            runtime_session_id="runtime-1",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=64,
    )


def fact(index: int, *, terminal: bool = False) -> ControlTraceFact:
    if terminal:
        event_type = "runtime_trace_terminal"
        payload = {
            "terminal_status": "completed",
            "reason": "fixture_complete",
            "occurred_at": float(index),
        }
    else:
        event_type = "source_validation"
        payload = {
            "validation_id": f"validation-{index}",
            "source_event_id": f"source-{index}",
            "accepted": True,
            "reasons": (),
        }
    return ControlTraceFact.create(
        fact_id=f"fact-{index}",
        event_type=event_type,
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        occurred_at=float(index),
        clock_domain_id="local_monotonic",
        payload=payload,
    )


def test_spool_sink_acknowledges_only_post_fsync_records(tmp_path: Path) -> None:
    trace = runtime()
    assert trace.issue(fact(1)).accepted
    assert trace.issue(fact(2, terminal=True)).accepted
    with ControlTraceSpool(
        tmp_path / "trace.wal",
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as spool, ControlTraceWriterHealthLedger(
        tmp_path / "trace.health",
        trace_session_id="trace-1",
    ) as health:
        sink = ControlTraceSpoolSink(
            sink_id="spool-sink-1",
            runtime=trace,
            spool=spool,
            health_ledger=health,
            clock=lambda: 10.0,
        )
        report = sink.flush_pending()

        assert report.newly_durable_sequences == (1, 2)
        assert report.failed_sequence is None
        assert report.durable_high_watermark == 2
        assert health.records() == ()
        completeness = trace.completeness()
        assert completeness.in_memory_complete
        assert completeness.durability_supported
        assert completeness.durable_complete
        assert not completeness.capture_commit_eligible


def test_spool_sink_batches_one_durability_barrier_per_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = runtime()
    assert trace.issue(fact(1)).accepted
    assert trace.issue(fact(2, terminal=True)).accepted
    with ControlTraceSpool(
        tmp_path / "batched.wal",
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as spool, ControlTraceWriterHealthLedger(
        tmp_path / "batched.health",
        trace_session_id="trace-1",
    ) as health:
        import teleop_runtime.control_trace.spool as spool_module

        fsync_calls: list[int] = []
        real_fsync = spool_module.os.fsync

        def observed_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        monkeypatch.setattr(spool_module.os, "fsync", observed_fsync)
        sink = ControlTraceSpoolSink(
            sink_id="batched-sink",
            runtime=trace,
            spool=spool,
            health_ledger=health,
        )

        report = sink.flush_pending()

        assert report.newly_durable_sequences == (1, 2)
        assert report.failure_code is None
        assert len(fsync_calls) == 1


def test_spool_sink_scans_only_new_runtime_entries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = runtime()
    offsets: list[int] = []
    original = trace.journal_entries_from_offset

    def observed(offset: int):
        offsets.append(offset)
        return original(offset)

    monkeypatch.setattr(trace, "journal_entries_from_offset", observed)
    with ControlTraceSpool(
        tmp_path / "incremental.wal",
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as spool, ControlTraceWriterHealthLedger(
        tmp_path / "incremental.health",
        trace_session_id="trace-1",
    ) as health:
        sink = ControlTraceSpoolSink(
            sink_id="incremental-sink",
            runtime=trace,
            spool=spool,
            health_ledger=health,
        )
        assert trace.issue(fact(1)).accepted
        assert sink.flush_pending().newly_durable_sequences == (1,)
        assert trace.issue(fact(2)).accepted
        assert sink.flush_pending().newly_durable_sequences == (2,)
        assert sink.flush_pending().newly_durable_sequences == ()

    assert offsets == [0, 1, 2]


def test_spool_sink_hot_path_does_not_build_full_runtime_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = runtime()
    assert trace.issue(fact(1)).accepted
    with ControlTraceSpool(
        tmp_path / "no-snapshot.wal",
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as spool, ControlTraceWriterHealthLedger(
        tmp_path / "no-snapshot.health",
        trace_session_id="trace-1",
    ) as health:
        sink = ControlTraceSpoolSink(
            sink_id="no-snapshot-sink",
            runtime=trace,
            spool=spool,
            health_ledger=health,
        )

        def reject_snapshot():
            raise AssertionError("storage hot path built a full runtime snapshot")

        monkeypatch.setattr(trace, "snapshot", reject_snapshot)
        report = sink.flush_pending()

    assert report.newly_durable_sequences == (1,)
    assert report.durable_high_watermark == 1


def test_spool_failure_uses_independent_durable_health_ledger(tmp_path: Path) -> None:
    trace = runtime()
    for index in range(1, 8):
        assert trace.issue(fact(index)).accepted
    assert trace.issue(fact(8, terminal=True)).accepted
    health_path = tmp_path / "bounded.health"
    with ControlTraceSpool(
        tmp_path / "bounded.wal",
        trace_session_id="trace-1",
        max_bytes=4096,
        physical_preallocation=False,
    ) as spool, ControlTraceWriterHealthLedger(
        health_path,
        trace_session_id="trace-1",
        slot_count=4,
    ) as health:
        sink = ControlTraceSpoolSink(
            sink_id="bounded-sink",
            runtime=trace,
            spool=spool,
            health_ledger=health,
            clock=lambda: 20.0,
        )
        report = sink.flush_pending()

        assert report.failure_code == "spool_capacity_exhausted"
        assert report.failed_sequence is not None
        assert report.health_receipt is not None
        assert report.health_receipt.accepted and report.health_receipt.durable
        assert not trace.completeness().durable_complete

    with ControlTraceWriterHealthLedger(
        health_path,
        trace_session_id="trace-1",
        slot_count=4,
    ) as recovered:
        records = recovered.records()
        assert len(records) == 1
        assert records[0].code == "spool_capacity_exhausted"
        assert records[0].journal_sequence == report.failed_sequence


def test_runtime_rejects_forged_or_foreign_durable_acknowledgement() -> None:
    trace = runtime()
    receipt = trace.issue(fact(1))
    assert receipt.accepted
    record_hash = trace.records()[0].record_integrity_hash
    trace.register_durable_sink("sink-1")

    with pytest.raises(ValueError, match="unregistered sink"):
        trace.acknowledge_durable(
            sink_id="sink-2",
            journal_sequence=1,
            record_integrity_hash=record_hash,
        )
    with pytest.raises(ValueError, match="hash mismatch"):
        trace.acknowledge_durable(
            sink_id="sink-1",
            journal_sequence=1,
            record_integrity_hash="0" * 64,
        )


def test_sink_persists_wal_tail_recovery_evidence(tmp_path: Path) -> None:
    trace = runtime()
    assert trace.issue(fact(1)).accepted
    wal_path = tmp_path / "recover.wal"
    with ControlTraceSpool(
        wal_path,
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as spool:
        assert spool.append(trace.records()[0]).accepted
        valid_bytes = spool.write_offset
    torn = encode_control_trace_wal_frame(
        trace.records()[0]
    )
    with wal_path.open("r+b") as handle:
        handle.seek(valid_bytes)
        handle.write(torn[: len(torn) // 3])
        handle.flush()

    with ControlTraceSpool(
        wal_path,
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as recovered, ControlTraceWriterHealthLedger(
        tmp_path / "recover.health",
        trace_session_id="trace-1",
    ) as health:
        sink = ControlTraceSpoolSink(
            sink_id="recover-sink",
            runtime=trace,
            spool=recovered,
            health_ledger=health,
            clock=lambda: 30.0,
        )
        assert len(sink.startup_health_receipts) == 1
        assert health.records()[0].code == "wal_tail_recovered"


def test_background_worker_flushes_without_owning_runtime_lifecycle(tmp_path: Path) -> None:
    trace = runtime()
    with ControlTraceSpool(
        tmp_path / "worker.wal",
        trace_session_id="trace-1",
        max_bytes=1 << 20,
        physical_preallocation=False,
    ) as spool, ControlTraceWriterHealthLedger(
        tmp_path / "worker.health",
        trace_session_id="trace-1",
    ) as health:
        sink = ControlTraceSpoolSink(
            sink_id="worker-sink",
            runtime=trace,
            spool=spool,
            health_ledger=health,
        )
        worker = ControlTraceSpoolWorker(sink, poll_interval_sec=0.01)
        worker.start()
        try:
            assert trace.issue(fact(1)).accepted
            assert trace.issue(fact(2, terminal=True)).accepted
            worker.notify()
            waiter = threading.Event()
            for _ in range(100):
                if trace.snapshot().sink_durable_high_watermark == 2:
                    break
                waiter.wait(0.01)
            assert trace.snapshot().sink_durable_high_watermark == 2
            assert worker.snapshot().fatal_error_type is None
            assert not hasattr(worker, "begin_episode")
            assert not hasattr(worker, "commit_episode")
        finally:
            worker.stop(timeout_sec=1.0)
        assert worker.snapshot().stopped

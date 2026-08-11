from __future__ import annotations

from pathlib import Path

import pytest

from dataset_contracts.control_trace import ControlTraceFact, PersistedRecordEnvelope
from dataset_contracts.control_trace_wal import encode_control_trace_wal_frame
from teleop_runtime.control_trace.spool import ControlTraceSpool


pytestmark = pytest.mark.integration


def envelope(sequence: int, *, trace_session_id: str = "trace-1") -> PersistedRecordEnvelope:
    fact = ControlTraceFact.create(
        fact_id=f"terminal-{sequence}",
        event_type="runtime_trace_terminal",
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        occurred_at=float(sequence),
        clock_domain_id="local_monotonic",
        payload={
            "terminal_status": "completed",
            "reason": f"fixture-{sequence}",
            "occurred_at": float(sequence),
        },
    )
    return PersistedRecordEnvelope.wrap(
        trace_session_id=trace_session_id,
        journal_sequence=sequence,
        fact=fact,
        episode_membership=None,
    )


def spool(
    path: Path,
    *,
    max_bytes: int = 1 << 20,
    allow_growth: bool = False,
) -> ControlTraceSpool:
    return ControlTraceSpool(
        path,
        trace_session_id="trace-1",
        max_bytes=max_bytes,
        physical_preallocation=False,
        allow_growth=allow_growth,
        growth_bytes=max_bytes,
    )


def test_spool_append_reopen_and_durable_high_watermark(tmp_path: Path) -> None:
    path = tmp_path / "control-trace.wal"
    with spool(path) as sink:
        first = sink.append(envelope(1))
        third = sink.append(envelope(3))
        written = sink.write_offset

        assert first.accepted and first.durable
        assert third.accepted and third.durable
        assert third.durable_high_watermark == 1
        assert path.stat().st_size == 1 << 20

    with spool(path) as recovered:
        assert recovered.recovery_report.recovered_sequences == (1, 3)
        assert recovered.recovery_report.valid_bytes == written
        assert recovered.recovery_report.tail_status == "zero_filled"
        assert recovered.durable_high_watermark == 1
        assert recovered.append(envelope(4)).accepted


def test_spool_rejects_session_sequence_and_capacity_without_overwrite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bounded.wal"
    with spool(path, max_bytes=4096) as sink:
        first = sink.append(envelope(1))
        assert first.accepted
        before = sink.write_offset

        duplicate = sink.append(envelope(1))
        foreign = sink.append(envelope(2, trace_session_id="trace-other"))
        assert duplicate.rejection_code == "non_monotonic_sequence"
        assert foreign.rejection_code == "trace_session_mismatch"

        sequence = 2
        while True:
            receipt = sink.append(envelope(sequence))
            if not receipt.accepted:
                assert receipt.rejection_code == "spool_capacity_exhausted"
                break
            sequence += 1
        assert sink.write_offset >= before
        assert sink.write_offset <= 4096


def test_spool_can_grow_by_bounded_extents_and_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "growing.wal"
    sequence = 1
    with spool(path, max_bytes=4096, allow_growth=True) as sink:
        while sink.max_bytes == 4096:
            assert sink.append(envelope(sequence)).accepted
            sequence += 1
        expanded_bytes = sink.max_bytes
        written = sink.write_offset

    assert expanded_bytes >= 8192
    assert path.stat().st_size == expanded_bytes
    with spool(
        path,
        max_bytes=4096,
        allow_growth=True,
    ) as recovered:
        assert recovered.max_bytes == expanded_bytes
        assert recovered.recovery_report.valid_bytes == written
        assert recovered.append(envelope(sequence)).accepted


def test_spool_recovers_torn_tail_and_reuses_only_discarded_region(
    tmp_path: Path,
) -> None:
    path = tmp_path / "torn.wal"
    with spool(path) as sink:
        assert sink.append(envelope(1)).accepted
        valid_bytes = sink.write_offset

    torn_frame = encode_control_trace_wal_frame(envelope(2))
    with path.open("r+b") as handle:
        handle.seek(valid_bytes)
        handle.write(torn_frame[: len(torn_frame) // 2])
        handle.flush()

    with spool(path) as recovered:
        report = recovered.recovery_report
        assert report.recovered_sequences == (1,)
        assert report.tail_status in {"truncated", "corrupt"}
        assert report.discarded_byte_range == (valid_bytes, 1 << 20)
        receipt = recovered.append(envelope(2))
        assert receipt.accepted and receipt.frame_offset == valid_bytes

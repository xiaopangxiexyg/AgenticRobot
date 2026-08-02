"""Storage-worker bridge from in-memory journal records to the M4B spool."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from .health_ledger import (
    ControlTraceWriterHealthLedger,
    WriterHealthAppendReceipt,
)
from .runtime import ControlTraceRuntime
from .spool import ControlTraceSpool


@dataclass(frozen=True)
class ControlTraceSinkFlushReport:
    considered_sequences: tuple[int, ...]
    newly_durable_sequences: tuple[int, ...]
    failed_sequence: int | None
    failure_code: str | None
    health_receipt: WriterHealthAppendReceipt | None
    durable_high_watermark: int


class ControlTraceSpoolSink:
    """Synchronous storage-worker primitive; never called by control code directly."""

    def __init__(
        self,
        *,
        sink_id: str,
        runtime: ControlTraceRuntime,
        spool: ControlTraceSpool,
        health_ledger: ControlTraceWriterHealthLedger,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        normalized = str(sink_id).strip()
        if not normalized:
            raise ValueError("sink_id must be non-empty.")
        if spool.trace_session_id != runtime.session.trace_session_id:
            raise ValueError("Spool and ControlTraceRuntime trace sessions disagree.")
        if health_ledger.trace_session_id != runtime.session.trace_session_id:
            raise ValueError("Health ledger and ControlTraceRuntime sessions disagree.")
        self.sink_id = normalized
        self.runtime = runtime
        self.spool = spool
        self.health_ledger = health_ledger
        self._clock = clock
        self._acknowledged_sequences: set[int] = set()
        self._persisted_hash_by_sequence = {
            item.journal_sequence: item.record_integrity_hash
            for item in spool.recovered_envelopes
        }
        self._next_runtime_entry_offset = 0
        runtime.register_durable_sink(normalized)
        startup_health: list[WriterHealthAppendReceipt] = []
        recovery = spool.recovery_report
        if recovery.discarded_byte_range is not None:
            receipt = health_ledger.append(
                code="wal_tail_recovered",
                journal_sequence=(
                    None
                    if not recovery.recovered_sequences
                    else recovery.recovered_sequences[-1]
                ),
                observed_at=float(self._clock()),
                message=(
                    f"Discarded WAL tail {recovery.discarded_byte_range} after "
                    f"{recovery.tail_status}: {recovery.tail_error or 'unspecified'}"
                ),
            )
            if not receipt.accepted or not receipt.durable:
                raise ValueError("Unable to persist ControlTrace WAL recovery evidence.")
            startup_health.append(receipt)
        self.startup_health_receipts = tuple(startup_health)

    def _record_failure(
        self,
        *,
        code: str,
        sequence: int,
        message: str,
    ) -> WriterHealthAppendReceipt:
        return self.health_ledger.append(
            code=code,
            journal_sequence=sequence,
            observed_at=float(self._clock()),
            message=message,
        )

    def flush_pending(self) -> ControlTraceSinkFlushReport:
        considered: list[int] = []
        durable: list[int] = []
        entries = self.runtime.journal_entries_from_offset(
            self._next_runtime_entry_offset
        )
        # A new session has no recovered records.  Persist its pending tail as
        # one WAL batch so a high-rate ControlTrace pays one fsync per worker
        # drain, not one fsync per fact.
        if entries and all(
            entry.journal_sequence not in self._persisted_hash_by_sequence
            for entry in entries
        ):
            records = tuple(
                entry.materialize(
                    trace_session_id=self.runtime.session.trace_session_id
                )
                for entry in entries
                if entry.journal_sequence not in self._acknowledged_sequences
            )
            receipts = self.spool.append_batch(records)
            record_by_sequence = {
                record.journal_sequence: record for record in records
            }
            for receipt in receipts:
                sequence = receipt.journal_sequence
                considered.append(sequence)
                record = record_by_sequence[sequence]
                if not receipt.accepted or not receipt.durable:
                    code = str(
                        receipt.rejection_code
                        or "spool_append_not_durable"
                    )
                    health = self._record_failure(
                        code=code,
                        sequence=sequence,
                        message=receipt.message,
                    )
                    return ControlTraceSinkFlushReport(
                        considered_sequences=tuple(considered),
                        newly_durable_sequences=tuple(durable),
                        failed_sequence=sequence,
                        failure_code=code,
                        health_receipt=health,
                        durable_high_watermark=self.runtime.sink_durable_high_watermark,
                    )
                self._persisted_hash_by_sequence[sequence] = (
                    record.record_integrity_hash
                )
                try:
                    self.runtime.acknowledge_durable(
                        sink_id=self.sink_id,
                        journal_sequence=sequence,
                        record_integrity_hash=record.record_integrity_hash,
                    )
                except ValueError as exc:
                    health = self._record_failure(
                        code="runtime_durable_ack_failed",
                        sequence=sequence,
                        message=str(exc),
                    )
                    return ControlTraceSinkFlushReport(
                        considered_sequences=tuple(considered),
                        newly_durable_sequences=tuple(durable),
                        failed_sequence=sequence,
                        failure_code="runtime_durable_ack_failed",
                        health_receipt=health,
                        durable_high_watermark=self.runtime.sink_durable_high_watermark,
                    )
                self._acknowledged_sequences.add(sequence)
                durable.append(sequence)
                self._next_runtime_entry_offset += 1
            return ControlTraceSinkFlushReport(
                considered_sequences=tuple(considered),
                newly_durable_sequences=tuple(durable),
                failed_sequence=None,
                failure_code=None,
                health_receipt=None,
                durable_high_watermark=self.runtime.sink_durable_high_watermark,
            )
        for entry in entries:
            sequence = entry.journal_sequence
            if sequence in self._acknowledged_sequences:
                self._next_runtime_entry_offset += 1
                continue
            record = entry.materialize(
                trace_session_id=self.runtime.session.trace_session_id
            )
            considered.append(sequence)
            persisted_hash = self._persisted_hash_by_sequence.get(sequence)
            if persisted_hash is not None:
                if persisted_hash != record.record_integrity_hash:
                    health = self._record_failure(
                        code="recovered_record_hash_mismatch",
                        sequence=sequence,
                        message="Recovered WAL record hash disagrees with runtime record.",
                    )
                    return ControlTraceSinkFlushReport(
                        considered_sequences=tuple(considered),
                        newly_durable_sequences=tuple(durable),
                        failed_sequence=sequence,
                        failure_code="recovered_record_hash_mismatch",
                        health_receipt=health,
                        durable_high_watermark=self.runtime.sink_durable_high_watermark,
                    )
                append_durable = True
            else:
                receipt = self.spool.append(record)
                append_durable = receipt.accepted and receipt.durable
                if not append_durable:
                    code = str(receipt.rejection_code or "spool_append_not_durable")
                    health = self._record_failure(
                        code=code,
                        sequence=sequence,
                        message=receipt.message,
                    )
                    return ControlTraceSinkFlushReport(
                        considered_sequences=tuple(considered),
                        newly_durable_sequences=tuple(durable),
                        failed_sequence=sequence,
                        failure_code=code,
                        health_receipt=health,
                        durable_high_watermark=self.runtime.sink_durable_high_watermark,
                    )
                self._persisted_hash_by_sequence[sequence] = record.record_integrity_hash
            if append_durable:
                try:
                    self.runtime.acknowledge_durable(
                        sink_id=self.sink_id,
                        journal_sequence=sequence,
                        record_integrity_hash=record.record_integrity_hash,
                    )
                except ValueError as exc:
                    health = self._record_failure(
                        code="runtime_durable_ack_failed",
                        sequence=sequence,
                        message=str(exc),
                    )
                    return ControlTraceSinkFlushReport(
                        considered_sequences=tuple(considered),
                        newly_durable_sequences=tuple(durable),
                        failed_sequence=sequence,
                        failure_code="runtime_durable_ack_failed",
                        health_receipt=health,
                        durable_high_watermark=self.runtime.sink_durable_high_watermark,
                    )
                self._acknowledged_sequences.add(sequence)
                durable.append(sequence)
                self._next_runtime_entry_offset += 1
        return ControlTraceSinkFlushReport(
            considered_sequences=tuple(considered),
            newly_durable_sequences=tuple(durable),
            failed_sequence=None,
            failure_code=None,
            health_receipt=None,
            durable_high_watermark=self.runtime.sink_durable_high_watermark,
        )


__all__ = ["ControlTraceSinkFlushReport", "ControlTraceSpoolSink"]

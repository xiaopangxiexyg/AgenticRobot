"""Bounded local ControlTrace WAL spool; not wired to production frontends."""

from __future__ import annotations

import errno
import mmap
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.control_trace import PersistedRecordEnvelope
from dataset_contracts.control_trace_wal import (
    ControlTraceWalError,
    encode_control_trace_wal_frame,
    scan_control_trace_wal,
)


SPOOL_APPEND_REJECTION_CODES_V1 = (
    "closed",
    "faulted",
    "trace_session_mismatch",
    "non_monotonic_sequence",
    "spool_capacity_exhausted",
    "write_or_fsync_failed",
)


@dataclass(frozen=True)
class ControlTraceSpoolRecoveryReport:
    recovered_sequences: tuple[int, ...]
    recovered_record_count: int
    valid_bytes: int
    tail_status: str
    tail_error: str | None
    discarded_byte_range: tuple[int, int] | None
    preallocation_mode: str


@dataclass(frozen=True)
class ControlTraceSpoolAppendReceipt:
    journal_sequence: int
    accepted: bool
    durable: bool
    durable_high_watermark: int
    frame_offset: int | None
    frame_size: int | None
    rejection_code: str | None
    message: str


def _contiguous_high_watermark(sequences: set[int]) -> int:
    high = 0
    while high + 1 in sequences:
        high += 1
    return high


def _write_all(fd: int, payload: bytes, *, offset: int) -> None:
    written = 0
    while written < len(payload):
        count = os.pwrite(fd, payload[written:], offset + written)
        if count <= 0:
            raise OSError("ControlTrace spool pwrite made no progress.")
        written += count


class ControlTraceSpool:
    """Preallocated, append-only WAL owned by a future storage worker.

    The caller-facing ``ControlTraceRuntime.issue`` path does not use this
    class. M4B will connect it through a bounded worker/admission boundary only
    after latency and writer-health fixtures pass.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        trace_session_id: str,
        max_bytes: int,
        fsync_on_append: bool = True,
        recover_invalid_tail: bool = True,
        physical_preallocation: bool = True,
    ) -> None:
        session_id = str(trace_session_id).strip()
        if not session_id:
            raise ValueError("trace_session_id must be non-empty.")
        capacity = int(max_bytes)
        if capacity < 4096:
            raise ValueError("ControlTrace spool max_bytes must be at least 4096.")
        target = Path(path)
        if not target.parent.exists():
            raise ValueError("ControlTrace spool parent directory does not exist.")
        self.path = target
        self.trace_session_id = session_id
        self.max_bytes = capacity
        self.fsync_on_append = bool(fsync_on_append)
        self._lock = threading.RLock()
        self._closed = False
        self._faulted = False
        self._durable_sequences: set[int] = set()
        self._write_offset = 0
        self._last_sequence = 0
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        self._fd = os.open(target, flags, 0o600)
        try:
            current_size = os.fstat(self._fd).st_size
            if current_size == 0:
                os.ftruncate(self._fd, capacity)
            elif current_size != capacity:
                raise ValueError(
                    "Existing ControlTrace spool size does not match max_bytes."
                )
            self._preallocation_mode = "sparse_ftruncate"
            if physical_preallocation and hasattr(os, "posix_fallocate"):
                try:
                    os.posix_fallocate(self._fd, 0, capacity)
                    self._preallocation_mode = "posix_fallocate"
                except OSError as exc:
                    if exc.errno not in {
                        errno.EINVAL,
                        errno.ENOSYS,
                        errno.ENOTSUP,
                        errno.EOPNOTSUPP,
                    }:
                        raise
            scan = self._scan_file()
            self._recovered_envelopes = tuple(
                frame.envelope for frame in scan.frames
            )
            for frame in scan.frames:
                envelope = frame.envelope
                if envelope.trace_session_id != session_id:
                    raise ValueError(
                        "Recovered ControlTrace frame belongs to another trace session."
                    )
                self._durable_sequences.add(envelope.journal_sequence)
                self._last_sequence = envelope.journal_sequence
            self._write_offset = scan.valid_bytes
            discarded: tuple[int, int] | None = None
            if scan.tail_status not in {"zero_filled", "clean_eof"}:
                if not recover_invalid_tail:
                    raise ControlTraceWalError(
                        f"ControlTrace spool has invalid tail: {scan.tail_error}"
                    )
                discarded = (scan.valid_bytes, capacity)
                os.ftruncate(self._fd, scan.valid_bytes)
                os.ftruncate(self._fd, capacity)
                if physical_preallocation and hasattr(os, "posix_fallocate"):
                    try:
                        os.posix_fallocate(
                            self._fd,
                            scan.valid_bytes,
                            capacity - scan.valid_bytes,
                        )
                        self._preallocation_mode = "posix_fallocate"
                    except OSError as exc:
                        if exc.errno not in {
                            errno.EINVAL,
                            errno.ENOSYS,
                            errno.ENOTSUP,
                            errno.EOPNOTSUPP,
                        }:
                            raise
                os.fsync(self._fd)
            self.recovery_report = ControlTraceSpoolRecoveryReport(
                recovered_sequences=tuple(sorted(self._durable_sequences)),
                recovered_record_count=len(scan.frames),
                valid_bytes=scan.valid_bytes,
                tail_status=scan.tail_status,
                tail_error=scan.tail_error,
                discarded_byte_range=discarded,
                preallocation_mode=self._preallocation_mode,
            )
        except BaseException:
            os.close(self._fd)
            self._closed = True
            raise

    def _scan_file(self):
        mapping = mmap.mmap(self._fd, self.max_bytes, access=mmap.ACCESS_READ)
        try:
            return scan_control_trace_wal(mapping)
        finally:
            mapping.close()

    @property
    def durable_high_watermark(self) -> int:
        with self._lock:
            return _contiguous_high_watermark(self._durable_sequences)

    @property
    def recovered_envelopes(self) -> tuple[PersistedRecordEnvelope, ...]:
        return self._recovered_envelopes

    @property
    def write_offset(self) -> int:
        with self._lock:
            return self._write_offset

    def append(
        self,
        envelope: PersistedRecordEnvelope,
    ) -> ControlTraceSpoolAppendReceipt:
        if not isinstance(envelope, PersistedRecordEnvelope):
            raise TypeError("ControlTraceSpool.append requires PersistedRecordEnvelope.")
        sequence = envelope.journal_sequence
        with self._lock:
            def reject(code: str, message: str) -> ControlTraceSpoolAppendReceipt:
                return ControlTraceSpoolAppendReceipt(
                    journal_sequence=sequence,
                    accepted=False,
                    durable=False,
                    durable_high_watermark=_contiguous_high_watermark(
                        self._durable_sequences
                    ),
                    frame_offset=None,
                    frame_size=None,
                    rejection_code=code,
                    message=message,
                )

            if self._closed:
                return reject("closed", "ControlTrace spool is closed.")
            if self._faulted:
                return reject("faulted", "ControlTrace spool is faulted.")
            if envelope.trace_session_id != self.trace_session_id:
                return reject(
                    "trace_session_mismatch",
                    "Envelope belongs to another trace session.",
                )
            if sequence <= self._last_sequence:
                return reject(
                    "non_monotonic_sequence",
                    "WAL journal sequence must strictly increase.",
                )
            frame = encode_control_trace_wal_frame(envelope)
            offset = self._write_offset
            if offset + len(frame) > self.max_bytes:
                return reject(
                    "spool_capacity_exhausted",
                    "Preallocated ControlTrace spool capacity is exhausted.",
                )
            try:
                _write_all(self._fd, frame, offset=offset)
                if self.fsync_on_append:
                    os.fsync(self._fd)
            except OSError as exc:
                self._faulted = True
                return reject(
                    "write_or_fsync_failed",
                    f"ControlTrace spool write/fsync failed: {type(exc).__name__}.",
                )
            self._write_offset += len(frame)
            self._last_sequence = sequence
            if self.fsync_on_append:
                self._durable_sequences.add(sequence)
            return ControlTraceSpoolAppendReceipt(
                journal_sequence=sequence,
                accepted=True,
                durable=self.fsync_on_append,
                durable_high_watermark=_contiguous_high_watermark(
                    self._durable_sequences
                ),
                frame_offset=offset,
                frame_size=len(frame),
                rejection_code=None,
                message=(
                    "appended_and_fsynced"
                    if self.fsync_on_append
                    else "appended_not_durable"
                ),
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> "ControlTraceSpool":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


__all__ = [
    "SPOOL_APPEND_REJECTION_CODES_V1",
    "ControlTraceSpool",
    "ControlTraceSpoolAppendReceipt",
    "ControlTraceSpoolRecoveryReport",
]

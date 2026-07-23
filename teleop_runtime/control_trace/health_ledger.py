"""Small preallocated failure ledger independent from the main trace spool."""

from __future__ import annotations

import hashlib
import math
import mmap
import os
import struct
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path

from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads


WRITER_HEALTH_LEDGER_MAGIC_V1 = b"DXCTHLT1"
WRITER_HEALTH_LEDGER_DOMAIN_V1 = "dexteleop.control_trace.writer_health.v1"
WRITER_HEALTH_LEDGER_SLOT_BYTES_V1 = 1024
_SLOT_HEADER_PREFIX = struct.Struct(">8sQI32s")
_SLOT_HEADER = struct.Struct(">8sQI32sI")


@dataclass(frozen=True)
class WriterHealthEvidence:
    evidence_sequence: int
    trace_session_id: str
    code: str
    journal_sequence: int | None
    observed_at: float
    message: str

    def __post_init__(self) -> None:
        if int(self.evidence_sequence) <= 0:
            raise ValueError("evidence_sequence must be positive.")
        if not str(self.trace_session_id).strip() or not str(self.code).strip():
            raise ValueError("Writer health trace_session_id/code must be non-empty.")
        if self.journal_sequence is not None and int(self.journal_sequence) <= 0:
            raise ValueError("journal_sequence must be positive when present.")
        if not math.isfinite(float(self.observed_at)):
            raise ValueError("observed_at must be finite.")
        if not str(self.message).strip():
            raise ValueError("Writer health message must be non-empty.")

    def to_payload(self) -> dict[str, object]:
        return {
            "evidence_sequence": int(self.evidence_sequence),
            "trace_session_id": str(self.trace_session_id),
            "code": str(self.code),
            "journal_sequence": (
                None if self.journal_sequence is None else int(self.journal_sequence)
            ),
            "observed_at": float(self.observed_at),
            "message": str(self.message),
        }


@dataclass(frozen=True)
class WriterHealthAppendReceipt:
    evidence_sequence: int
    accepted: bool
    durable: bool
    slot_index: int | None
    rejection_code: str | None


def _encode_slot(evidence: WriterHealthEvidence) -> bytes:
    payload = canonical_bytes(
        evidence.to_payload(),
        domain=WRITER_HEALTH_LEDGER_DOMAIN_V1,
    )
    available = WRITER_HEALTH_LEDGER_SLOT_BYTES_V1 - _SLOT_HEADER.size
    if len(payload) > available:
        raise ValueError("Writer health evidence exceeds the fixed slot size.")
    digest = hashlib.sha256(payload).digest()
    prefix = _SLOT_HEADER_PREFIX.pack(
        WRITER_HEALTH_LEDGER_MAGIC_V1,
        evidence.evidence_sequence,
        len(payload),
        digest,
    )
    header = _SLOT_HEADER.pack(
        WRITER_HEALTH_LEDGER_MAGIC_V1,
        evidence.evidence_sequence,
        len(payload),
        digest,
        zlib.crc32(prefix) & 0xFFFFFFFF,
    )
    return header + payload + bytes(available - len(payload))


def _decode_slot(slot: bytes) -> WriterHealthEvidence | None:
    if not any(slot):
        return None
    if len(slot) != WRITER_HEALTH_LEDGER_SLOT_BYTES_V1:
        raise ValueError("Writer health slot size is invalid.")
    magic, sequence, payload_length, digest, crc = _SLOT_HEADER.unpack(
        slot[: _SLOT_HEADER.size]
    )
    if magic != WRITER_HEALTH_LEDGER_MAGIC_V1:
        raise ValueError("Writer health slot magic mismatch.")
    if zlib.crc32(slot[: _SLOT_HEADER_PREFIX.size]) & 0xFFFFFFFF != crc:
        raise ValueError("Writer health slot header CRC mismatch.")
    if payload_length > WRITER_HEALTH_LEDGER_SLOT_BYTES_V1 - _SLOT_HEADER.size:
        raise ValueError("Writer health slot payload length is invalid.")
    payload = slot[_SLOT_HEADER.size : _SLOT_HEADER.size + payload_length]
    if hashlib.sha256(payload).digest() != digest:
        raise ValueError("Writer health slot payload digest mismatch.")
    decoded = canonical_loads(
        payload,
        expected_domain=WRITER_HEALTH_LEDGER_DOMAIN_V1,
    )
    if canonical_bytes(decoded, domain=WRITER_HEALTH_LEDGER_DOMAIN_V1) != payload:
        raise ValueError("Writer health slot payload is not canonical.")
    evidence = WriterHealthEvidence(**decoded)
    if evidence.evidence_sequence != sequence:
        raise ValueError("Writer health slot sequence mismatch.")
    return evidence


class ControlTraceWriterHealthLedger:
    def __init__(
        self,
        path: str | Path,
        *,
        trace_session_id: str,
        slot_count: int = 64,
    ) -> None:
        session_id = str(trace_session_id).strip()
        if not session_id or int(slot_count) <= 0:
            raise ValueError("Writer health ledger session/slot_count is invalid.")
        target = Path(path)
        if not target.parent.exists():
            raise ValueError("Writer health ledger parent directory does not exist.")
        self.path = target
        self.trace_session_id = session_id
        self.slot_count = int(slot_count)
        self._size = self.slot_count * WRITER_HEALTH_LEDGER_SLOT_BYTES_V1
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        self._fd = os.open(target, flags, 0o600)
        self._lock = threading.RLock()
        self._closed = False
        self._records: list[WriterHealthEvidence] = []
        try:
            size = os.fstat(self._fd).st_size
            if size == 0:
                os.ftruncate(self._fd, self._size)
            elif size != self._size:
                raise ValueError("Existing writer health ledger size is incompatible.")
            mapping = mmap.mmap(self._fd, self._size, access=mmap.ACCESS_READ)
            try:
                for index in range(self.slot_count):
                    start = index * WRITER_HEALTH_LEDGER_SLOT_BYTES_V1
                    evidence = _decode_slot(
                        bytes(mapping[start : start + WRITER_HEALTH_LEDGER_SLOT_BYTES_V1])
                    )
                    if evidence is None:
                        break
                    if evidence.trace_session_id != session_id:
                        raise ValueError("Writer health evidence belongs to another session.")
                    if evidence.evidence_sequence != index + 1:
                        raise ValueError("Writer health evidence sequence is not contiguous.")
                    self._records.append(evidence)
            finally:
                mapping.close()
        except BaseException:
            os.close(self._fd)
            self._closed = True
            raise

    def records(self) -> tuple[WriterHealthEvidence, ...]:
        with self._lock:
            return tuple(self._records)

    def append(
        self,
        *,
        code: str,
        journal_sequence: int | None,
        observed_at: float,
        message: str,
    ) -> WriterHealthAppendReceipt:
        with self._lock:
            sequence = len(self._records) + 1
            if self._closed:
                return WriterHealthAppendReceipt(
                    sequence, False, False, None, "closed"
                )
            if len(self._records) >= self.slot_count:
                return WriterHealthAppendReceipt(
                    sequence, False, False, None, "health_ledger_capacity_exhausted"
                )
            evidence = WriterHealthEvidence(
                evidence_sequence=sequence,
                trace_session_id=self.trace_session_id,
                code=str(code),
                journal_sequence=journal_sequence,
                observed_at=float(observed_at),
                message=str(message),
            )
            slot = _encode_slot(evidence)
            offset = len(self._records) * WRITER_HEALTH_LEDGER_SLOT_BYTES_V1
            try:
                written = 0
                while written < len(slot):
                    count = os.pwrite(self._fd, slot[written:], offset + written)
                    if count <= 0:
                        raise OSError("Writer health pwrite made no progress.")
                    written += count
                os.fsync(self._fd)
            except OSError:
                return WriterHealthAppendReceipt(
                    sequence, False, False, None, "health_ledger_write_failed"
                )
            self._records.append(evidence)
            return WriterHealthAppendReceipt(
                sequence, True, True, len(self._records) - 1, None
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            os.close(self._fd)
            self._closed = True

    def __enter__(self) -> "ControlTraceWriterHealthLedger":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()


__all__ = [
    "WRITER_HEALTH_LEDGER_DOMAIN_V1",
    "WRITER_HEALTH_LEDGER_MAGIC_V1",
    "WRITER_HEALTH_LEDGER_SLOT_BYTES_V1",
    "ControlTraceWriterHealthLedger",
    "WriterHealthAppendReceipt",
    "WriterHealthEvidence",
]

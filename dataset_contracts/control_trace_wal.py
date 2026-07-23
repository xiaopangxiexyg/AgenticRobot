"""Versioned binary WAL framing for v2.4 ControlTrace logical envelopes.

This module is pure: it owns no files, threads, fsync policy, or episode
lifecycle.  It only defines frame bytes and deterministic recovery scanning.
"""

from __future__ import annotations

import hashlib
import struct
import zlib
from dataclasses import dataclass

from .canonical_encoding import canonical_bytes, canonical_loads
from .control_trace import PersistedRecordEnvelope


CONTROL_TRACE_WAL_MAGIC_V1 = b"DXCTWAL1"
CONTROL_TRACE_WAL_TRAILER_MAGIC_V1 = b"DXCTEND1"
CONTROL_TRACE_WAL_VERSION_V1 = 1
CONTROL_TRACE_WAL_PAYLOAD_DOMAIN_V1 = "dexteleop.control_trace.wal_payload.v1"
CONTROL_TRACE_WAL_MAX_PAYLOAD_BYTES_V1 = 64 * 1024 * 1024

# Exactly 64 bytes: magic, version, flags, journal sequence, payload length,
# SHA-256 payload digest, and CRC32 over the preceding 60 header bytes.
_HEADER_PREFIX = struct.Struct(">8sHHQQ32s")
_HEADER = struct.Struct(">8sHHQQ32sI")
_TRAILER = struct.Struct(">8sQ")


class ControlTraceWalError(ValueError):
    pass


class ControlTraceWalTruncatedError(ControlTraceWalError):
    pass


@dataclass(frozen=True)
class DecodedControlTraceWalFrame:
    envelope: PersistedRecordEnvelope
    offset: int
    frame_size: int
    payload_size: int
    payload_sha256: str


@dataclass(frozen=True)
class ControlTraceWalScanResult:
    frames: tuple[DecodedControlTraceWalFrame, ...]
    valid_bytes: int
    tail_status: str
    tail_error: str | None


def encode_control_trace_wal_frame(envelope: PersistedRecordEnvelope) -> bytes:
    if not isinstance(envelope, PersistedRecordEnvelope):
        raise TypeError("WAL framing requires PersistedRecordEnvelope.")
    payload = canonical_bytes(
        envelope.to_storage_payload(),
        domain=CONTROL_TRACE_WAL_PAYLOAD_DOMAIN_V1,
    )
    if len(payload) > CONTROL_TRACE_WAL_MAX_PAYLOAD_BYTES_V1:
        raise ValueError("ControlTrace WAL payload exceeds the v1 maximum.")
    digest = hashlib.sha256(payload).digest()
    prefix = _HEADER_PREFIX.pack(
        CONTROL_TRACE_WAL_MAGIC_V1,
        CONTROL_TRACE_WAL_VERSION_V1,
        0,
        envelope.journal_sequence,
        len(payload),
        digest,
    )
    header = _HEADER.pack(
        CONTROL_TRACE_WAL_MAGIC_V1,
        CONTROL_TRACE_WAL_VERSION_V1,
        0,
        envelope.journal_sequence,
        len(payload),
        digest,
        zlib.crc32(prefix) & 0xFFFFFFFF,
    )
    trailer = _TRAILER.pack(
        CONTROL_TRACE_WAL_TRAILER_MAGIC_V1,
        envelope.journal_sequence,
    )
    return header + payload + trailer


def decode_control_trace_wal_frame(
    encoded: bytes | bytearray | memoryview,
    *,
    offset: int = 0,
) -> DecodedControlTraceWalFrame:
    view = memoryview(encoded)
    start = int(offset)
    if start < 0 or start > len(view):
        raise ValueError("WAL frame offset is outside the buffer.")
    if len(view) - start < _HEADER.size:
        raise ControlTraceWalTruncatedError("ControlTrace WAL header is truncated.")
    header_bytes = bytes(view[start : start + _HEADER.size])
    magic, version, flags, sequence, payload_length, digest, header_crc = _HEADER.unpack(
        header_bytes
    )
    if magic != CONTROL_TRACE_WAL_MAGIC_V1:
        raise ControlTraceWalError("ControlTrace WAL frame magic mismatch.")
    if version != CONTROL_TRACE_WAL_VERSION_V1 or flags != 0:
        raise ControlTraceWalError("Unsupported ControlTrace WAL version/flags.")
    if zlib.crc32(header_bytes[: _HEADER_PREFIX.size]) & 0xFFFFFFFF != header_crc:
        raise ControlTraceWalError("ControlTrace WAL header CRC mismatch.")
    if payload_length > CONTROL_TRACE_WAL_MAX_PAYLOAD_BYTES_V1:
        raise ControlTraceWalError("ControlTrace WAL payload length exceeds the v1 maximum.")
    frame_size = _HEADER.size + int(payload_length) + _TRAILER.size
    end = start + frame_size
    if end > len(view):
        raise ControlTraceWalTruncatedError("ControlTrace WAL payload/trailer is truncated.")
    payload_start = start + _HEADER.size
    payload = bytes(view[payload_start : payload_start + payload_length])
    trailer_magic, trailer_sequence = _TRAILER.unpack(
        bytes(view[payload_start + payload_length : end])
    )
    if trailer_magic != CONTROL_TRACE_WAL_TRAILER_MAGIC_V1 or trailer_sequence != sequence:
        raise ControlTraceWalError("ControlTrace WAL trailer mismatch.")
    if hashlib.sha256(payload).digest() != digest:
        raise ControlTraceWalError("ControlTrace WAL payload digest mismatch.")
    logical_payload = canonical_loads(
        payload,
        expected_domain=CONTROL_TRACE_WAL_PAYLOAD_DOMAIN_V1,
    )
    if canonical_bytes(
        logical_payload,
        domain=CONTROL_TRACE_WAL_PAYLOAD_DOMAIN_V1,
    ) != payload:
        raise ControlTraceWalError("ControlTrace WAL payload is not canonical.")
    try:
        envelope = PersistedRecordEnvelope.from_storage_payload(logical_payload)
    except (TypeError, ValueError) as exc:
        raise ControlTraceWalError(
            f"ControlTrace WAL logical envelope is invalid: {exc}"
        ) from exc
    if envelope.journal_sequence != sequence:
        raise ControlTraceWalError("ControlTrace WAL header/envelope sequence mismatch.")
    return DecodedControlTraceWalFrame(
        envelope=envelope,
        offset=start,
        frame_size=frame_size,
        payload_size=int(payload_length),
        payload_sha256=digest.hex(),
    )


def scan_control_trace_wal(
    encoded: bytes | bytearray | memoryview,
) -> ControlTraceWalScanResult:
    """Scan complete frames and report, but never hide, an invalid tail."""

    view = memoryview(encoded)
    frames: list[DecodedControlTraceWalFrame] = []
    offset = 0
    while offset < len(view):
        remaining = len(view) - offset
        if not any(view[offset : offset + min(remaining, _HEADER.size)]):
            return ControlTraceWalScanResult(
                frames=tuple(frames),
                valid_bytes=offset,
                tail_status="zero_filled",
                tail_error=None,
            )
        try:
            frame = decode_control_trace_wal_frame(view, offset=offset)
        except ControlTraceWalTruncatedError as exc:
            return ControlTraceWalScanResult(
                frames=tuple(frames),
                valid_bytes=offset,
                tail_status="truncated",
                tail_error=str(exc),
            )
        except ControlTraceWalError as exc:
            return ControlTraceWalScanResult(
                frames=tuple(frames),
                valid_bytes=offset,
                tail_status="corrupt",
                tail_error=str(exc),
            )
        if frames and frame.envelope.journal_sequence <= frames[-1].envelope.journal_sequence:
            return ControlTraceWalScanResult(
                frames=tuple(frames),
                valid_bytes=offset,
                tail_status="non_monotonic_sequence",
                tail_error="ControlTrace WAL journal sequence must strictly increase.",
            )
        frames.append(frame)
        offset += frame.frame_size
    return ControlTraceWalScanResult(
        frames=tuple(frames),
        valid_bytes=offset,
        tail_status="clean_eof",
        tail_error=None,
    )


__all__ = [
    "CONTROL_TRACE_WAL_MAGIC_V1",
    "CONTROL_TRACE_WAL_MAX_PAYLOAD_BYTES_V1",
    "CONTROL_TRACE_WAL_PAYLOAD_DOMAIN_V1",
    "CONTROL_TRACE_WAL_TRAILER_MAGIC_V1",
    "CONTROL_TRACE_WAL_VERSION_V1",
    "ControlTraceWalError",
    "ControlTraceWalScanResult",
    "ControlTraceWalTruncatedError",
    "DecodedControlTraceWalFrame",
    "decode_control_trace_wal_frame",
    "encode_control_trace_wal_frame",
    "scan_control_trace_wal",
]

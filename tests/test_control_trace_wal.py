from __future__ import annotations

import struct

import pytest

from dataset_contracts.control_trace import (
    ControlTraceFact,
    EpisodeMembershipRef,
    PersistedRecordEnvelope,
)
from dataset_contracts.control_trace_wal import (
    ControlTraceWalError,
    decode_control_trace_wal_frame,
    encode_control_trace_wal_frame,
    scan_control_trace_wal,
)


def envelope(sequence: int, *, episode: bool = False) -> PersistedRecordEnvelope:
    fact = ControlTraceFact.create(
        fact_id=f"terminal-{sequence}",
        event_type="runtime_trace_terminal",
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        occurred_at=float(sequence),
        clock_domain_id="local_monotonic",
        payload={
            "terminal_status": "completed",
            "reason": f"fixture-{sequence}",
            "occurred_at": float(sequence),
        },
    )
    membership = (
        EpisodeMembershipRef(
            episode_uuid="episode-1",
            lifecycle_owner_id="fixture-owner",
            binding_event_id="bind-1",
            membership_epoch=1,
            bound_at=0.5,
        )
        if episode
        else None
    )
    return PersistedRecordEnvelope.wrap(
        trace_session_id="trace-1",
        journal_sequence=sequence,
        fact=fact,
        episode_membership=membership,
        schema_refs=("dexteleop_control_trace_v2.4",),
    )


def test_wal_frame_round_trip_is_deterministic_and_preserves_membership() -> None:
    source = envelope(1, episode=True)

    first = encode_control_trace_wal_frame(source)
    second = encode_control_trace_wal_frame(source)
    decoded = decode_control_trace_wal_frame(first)

    assert first == second
    assert decoded.frame_size == len(first)
    assert decoded.envelope.record_integrity_hash == source.record_integrity_hash
    assert decoded.envelope.fact.semantic_content_hash == source.fact.semantic_content_hash
    assert decoded.envelope.episode_membership is not None
    assert decoded.envelope.episode_membership.lifecycle_owner_id == "fixture-owner"


def test_wal_round_trip_hashes_the_frozen_nested_sequence_representation() -> None:
    fact = ControlTraceFact.create(
        fact_id="terminal-with-nested-sequence",
        event_type="runtime_trace_terminal",
        runtime_session_id="runtime-1",
        control_domain_id="robot-0",
        occurred_at=1.0,
        clock_domain_id="local_monotonic",
        payload={
            "terminal_status": "completed",
            "reason": ["nested", ["sequence"]],
            "occurred_at": 1.0,
        },
    )
    source = PersistedRecordEnvelope.wrap(
        trace_session_id="trace-1",
        journal_sequence=1,
        fact=fact,
        episode_membership=None,
    )

    decoded = decode_control_trace_wal_frame(
        encode_control_trace_wal_frame(source)
    ).envelope

    assert decoded.fact.semantic_content_hash == fact.semantic_content_hash
    assert decoded.fact.payload["reason"] == ("nested", ("sequence",))


def test_wal_scan_stops_at_zero_filled_preallocated_tail() -> None:
    first = encode_control_trace_wal_frame(envelope(1))
    second = encode_control_trace_wal_frame(envelope(3))

    scan = scan_control_trace_wal(first + second + bytes(4096))

    assert [frame.envelope.journal_sequence for frame in scan.frames] == [1, 3]
    assert scan.valid_bytes == len(first) + len(second)
    assert scan.tail_status == "zero_filled"
    assert scan.tail_error is None


def test_wal_scan_reports_torn_or_corrupt_tail_without_accepting_it() -> None:
    first = encode_control_trace_wal_frame(envelope(1))
    second = encode_control_trace_wal_frame(envelope(2))

    torn = scan_control_trace_wal(first + second[:-7])
    assert len(torn.frames) == 1
    assert torn.valid_bytes == len(first)
    assert torn.tail_status == "truncated"

    corrupt = bytearray(first + second)
    corrupt[len(first) + 80] ^= 0x01
    scan = scan_control_trace_wal(corrupt)
    assert len(scan.frames) == 1
    assert scan.valid_bytes == len(first)
    assert scan.tail_status == "corrupt"
    assert "digest mismatch" in str(scan.tail_error)


def test_wal_decoder_rejects_header_crc_and_sequence_tampering() -> None:
    encoded = bytearray(encode_control_trace_wal_frame(envelope(1)))
    encoded[12] ^= 0x01
    with pytest.raises(ControlTraceWalError, match="CRC mismatch"):
        decode_control_trace_wal_frame(encoded)

    encoded = bytearray(encode_control_trace_wal_frame(envelope(1)))
    # Trailer sequence is the final uint64.
    encoded[-8:] = struct.pack(">Q", 2)
    with pytest.raises(ControlTraceWalError, match="trailer mismatch"):
        decode_control_trace_wal_frame(encoded)

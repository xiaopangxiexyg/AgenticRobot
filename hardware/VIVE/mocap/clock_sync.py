"""Versioned UDP control-plane helpers for VIVE clock synchronization.

This module deliberately has no dependency on ``teleop_runtime`` or OpenVR so
the Windows VIVE sender can import it when launched directly as a script.  It
is a control-plane protocol only: VIVE pose packets remain
``teleop_runtime.vive/v1`` on their existing UDP port.

The four timestamp convention is:

``sender t1 -> collector t2/t3 -> sender t4``.

All timestamps are local ``perf_counter_ns`` values.  The resulting offset is
``collector_clock - sender_clock``; a sender capture timestamp maps to the
collector clock by adding that offset.  Neither value is suitable for causal
selection by itself: the receiver's local availability timestamp remains the
authority for ``latest <= T0``.
"""

from __future__ import annotations

import json
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Mapping


CLOCK_SYNC_PROTOCOL = "teleop_runtime.vive.clock_sync/v1"
CLOCK_SYNC_PROBE = "probe"
CLOCK_SYNC_REPLY = "reply"
CLOCK_SYNC_RESULT = "result"


@dataclass(frozen=True)
class ClockSyncProbe:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int


@dataclass(frozen=True)
class ClockSyncReply:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int


@dataclass(frozen=True)
class ClockSyncResult:
    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int
    t4_sender_monotonic_ns: int


@dataclass(frozen=True)
class ClockSyncMeasurement:
    """A validated NTP-style timing sample in integer nanoseconds."""

    sender_instance_id: str
    probe_id: int
    t1_sender_monotonic_ns: int
    t2_collector_monotonic_ns: int
    t3_collector_monotonic_ns: int
    t4_sender_monotonic_ns: int
    collector_minus_sender_offset_ns: int
    round_trip_ns: int
    uncertainty_ns: int

    def sender_to_collector_monotonic_ns(self, sender_timestamp_ns: int) -> int:
        return int(sender_timestamp_ns) + self.collector_minus_sender_offset_ns


@dataclass(frozen=True)
class ClockSyncEstimate:
    """The selected low-RTT mapping for one sender process instance."""

    sender_instance_id: str
    collector_minus_sender_offset_ns: int
    round_trip_ns: int
    uncertainty_ns: int
    sample_count: int
    reference_sender_monotonic_ns: int
    reference_collector_monotonic_ns: int

    def sender_to_collector_monotonic_ns(self, sender_timestamp_ns: int) -> int:
        return int(sender_timestamp_ns) + self.collector_minus_sender_offset_ns


class ClockSyncSampleStore:
    """Keep bounded valid samples and select the low-RTT recent estimate.

    Periodic fresh probes bound oscillator drift in this first protocol version.
    We intentionally do not claim a fitted drift model until enough real-host
    measurements exist to establish suitable quality gates.
    """

    def __init__(
        self,
        *,
        max_samples: int = 64,
        selection_window: int = 8,
        max_round_trip_ns: int | None = None,
    ):
        self.max_samples = max(1, int(max_samples))
        self.selection_window = max(1, int(selection_window))
        self.max_round_trip_ns = (
            None if max_round_trip_ns is None else max(0, int(max_round_trip_ns))
        )
        self._sender_instance_id: str | None = None
        self._samples: Deque[ClockSyncMeasurement] = deque(maxlen=self.max_samples)

    @property
    def sender_instance_id(self) -> str | None:
        return self._sender_instance_id

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._sender_instance_id = None
        self._samples.clear()

    def begin_instance(self, sender_instance_id: str) -> None:
        """Invalidate an old mapping as soon as a new sender process is seen."""

        if self._sender_instance_id != sender_instance_id:
            self._sender_instance_id = sender_instance_id
            self._samples.clear()

    def add(self, measurement: ClockSyncMeasurement) -> ClockSyncEstimate | None:
        self.begin_instance(measurement.sender_instance_id)
        if (
            self.max_round_trip_ns is not None
            and measurement.round_trip_ns > self.max_round_trip_ns
        ):
            return self.estimate()
        self._samples.append(measurement)
        return self.estimate()

    def estimate(self) -> ClockSyncEstimate | None:
        if not self._samples or self._sender_instance_id is None:
            return None
        recent_samples = list(self._samples)[-self.selection_window :]
        selected = min(
            recent_samples,
            key=lambda sample: (sample.round_trip_ns, -sample.probe_id),
        )
        # The low-RTT sample bounds symmetric-delay uncertainty.  The spread of
        # retained measurements catches queue asymmetry without over-fitting a
        # drift model from a short startup burst.
        offset_spread_ns = max(
            abs(sample.collector_minus_sender_offset_ns - selected.collector_minus_sender_offset_ns)
            for sample in recent_samples
        )
        return ClockSyncEstimate(
            sender_instance_id=self._sender_instance_id,
            collector_minus_sender_offset_ns=selected.collector_minus_sender_offset_ns,
            round_trip_ns=selected.round_trip_ns,
            uncertainty_ns=selected.uncertainty_ns + offset_spread_ns,
            sample_count=len(self._samples),
            reference_sender_monotonic_ns=(
                selected.t1_sender_monotonic_ns + selected.t4_sender_monotonic_ns
            ) // 2,
            reference_collector_monotonic_ns=(
                selected.t2_collector_monotonic_ns + selected.t3_collector_monotonic_ns
            ) // 2,
        )


def new_sender_instance_id() -> str:
    """Return a process/session identity that invalidates old clock mappings."""

    return uuid.uuid4().hex


def encode_clock_sync_message(message: Mapping[str, Any]) -> bytes:
    """Encode a compact control datagram after callers construct a valid message."""

    return json.dumps(message, allow_nan=False, separators=(",", ":")).encode("utf-8")


def make_probe(*, sender_instance_id: str, probe_id: int, t1_sender_monotonic_ns: int) -> bytes:
    return encode_clock_sync_message(
        {
            "protocol": CLOCK_SYNC_PROTOCOL,
            "type": CLOCK_SYNC_PROBE,
            "sender_instance_id": str(sender_instance_id),
            "probe_id": int(probe_id),
            "t1_sender_monotonic_ns": int(t1_sender_monotonic_ns),
        }
    )


def make_reply(
    *,
    sender_instance_id: str,
    probe_id: int,
    t1_sender_monotonic_ns: int,
    t2_collector_monotonic_ns: int,
    t3_collector_monotonic_ns: int,
) -> bytes:
    return encode_clock_sync_message(
        {
            "protocol": CLOCK_SYNC_PROTOCOL,
            "type": CLOCK_SYNC_REPLY,
            "sender_instance_id": str(sender_instance_id),
            "probe_id": int(probe_id),
            "t1_sender_monotonic_ns": int(t1_sender_monotonic_ns),
            "t2_collector_monotonic_ns": int(t2_collector_monotonic_ns),
            "t3_collector_monotonic_ns": int(t3_collector_monotonic_ns),
        }
    )


def make_result(
    *,
    sender_instance_id: str,
    probe_id: int,
    t1_sender_monotonic_ns: int,
    t2_collector_monotonic_ns: int,
    t3_collector_monotonic_ns: int,
    t4_sender_monotonic_ns: int,
) -> bytes:
    return encode_clock_sync_message(
        {
            "protocol": CLOCK_SYNC_PROTOCOL,
            "type": CLOCK_SYNC_RESULT,
            "sender_instance_id": str(sender_instance_id),
            "probe_id": int(probe_id),
            "t1_sender_monotonic_ns": int(t1_sender_monotonic_ns),
            "t2_collector_monotonic_ns": int(t2_collector_monotonic_ns),
            "t3_collector_monotonic_ns": int(t3_collector_monotonic_ns),
            "t4_sender_monotonic_ns": int(t4_sender_monotonic_ns),
        }
    )


def parse_probe(packet: bytes) -> ClockSyncProbe | None:
    message = _decode_message(packet, expected_type=CLOCK_SYNC_PROBE)
    if message is None:
        return None
    sender_instance_id = _required_nonempty_string(message, "sender_instance_id")
    probe_id = _required_nonnegative_int(message, "probe_id")
    t1 = _required_nonnegative_int(message, "t1_sender_monotonic_ns")
    if sender_instance_id is None or probe_id is None or t1 is None:
        return None
    return ClockSyncProbe(sender_instance_id, probe_id, t1)


def parse_reply(packet: bytes) -> ClockSyncReply | None:
    message = _decode_message(packet, expected_type=CLOCK_SYNC_REPLY)
    if message is None:
        return None
    sender_instance_id = _required_nonempty_string(message, "sender_instance_id")
    probe_id = _required_nonnegative_int(message, "probe_id")
    t1 = _required_nonnegative_int(message, "t1_sender_monotonic_ns")
    t2 = _required_nonnegative_int(message, "t2_collector_monotonic_ns")
    t3 = _required_nonnegative_int(message, "t3_collector_monotonic_ns")
    if (
        sender_instance_id is None
        or probe_id is None
        or t1 is None
        or t2 is None
        or t3 is None
        or t3 < t2
    ):
        return None
    return ClockSyncReply(sender_instance_id, probe_id, t1, t2, t3)


def parse_result(packet: bytes) -> ClockSyncResult | None:
    message = _decode_message(packet, expected_type=CLOCK_SYNC_RESULT)
    if message is None:
        return None
    sender_instance_id = _required_nonempty_string(message, "sender_instance_id")
    probe_id = _required_nonnegative_int(message, "probe_id")
    t1 = _required_nonnegative_int(message, "t1_sender_monotonic_ns")
    t2 = _required_nonnegative_int(message, "t2_collector_monotonic_ns")
    t3 = _required_nonnegative_int(message, "t3_collector_monotonic_ns")
    t4 = _required_nonnegative_int(message, "t4_sender_monotonic_ns")
    if (
        sender_instance_id is None
        or probe_id is None
        or t1 is None
        or t2 is None
        or t3 is None
        or t4 is None
        or t3 < t2
        or t4 < t1
    ):
        return None
    return ClockSyncResult(sender_instance_id, probe_id, t1, t2, t3, t4)


def measurement_from_result(result: ClockSyncResult) -> ClockSyncMeasurement | None:
    """Compute a validated four-timestamp offset from a sender result packet."""

    processing_ns = result.t3_collector_monotonic_ns - result.t2_collector_monotonic_ns
    sender_elapsed_ns = result.t4_sender_monotonic_ns - result.t1_sender_monotonic_ns
    round_trip_ns = sender_elapsed_ns - processing_ns
    if round_trip_ns < 0:
        return None

    # Integer rounding is deliberate.  A half-nanosecond is immaterial beside
    # real network uncertainty and avoids float loss for long-running clocks.
    offset_numerator = (
        result.t2_collector_monotonic_ns
        - result.t1_sender_monotonic_ns
        + result.t3_collector_monotonic_ns
        - result.t4_sender_monotonic_ns
    )
    offset_ns = int(round(offset_numerator / 2.0))
    return ClockSyncMeasurement(
        sender_instance_id=result.sender_instance_id,
        probe_id=result.probe_id,
        t1_sender_monotonic_ns=result.t1_sender_monotonic_ns,
        t2_collector_monotonic_ns=result.t2_collector_monotonic_ns,
        t3_collector_monotonic_ns=result.t3_collector_monotonic_ns,
        t4_sender_monotonic_ns=result.t4_sender_monotonic_ns,
        collector_minus_sender_offset_ns=offset_ns,
        round_trip_ns=round_trip_ns,
        uncertainty_ns=(round_trip_ns + 1) // 2,
    )


def _decode_message(packet: bytes, *, expected_type: str) -> dict[str, Any] | None:
    try:
        decoded = json.loads(packet.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    if decoded.get("protocol") != CLOCK_SYNC_PROTOCOL:
        return None
    if decoded.get("type") != expected_type:
        return None
    return decoded


def _required_nonempty_string(message: Mapping[str, Any], key: str) -> str | None:
    value = message.get(key)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _required_nonnegative_int(message: Mapping[str, Any], key: str) -> int | None:
    value = message.get(key)
    if isinstance(value, bool):
        return None
    if not isinstance(value, int):
        return None
    if value < 0 or value > (2**63 - 1):
        return None
    return int(value)

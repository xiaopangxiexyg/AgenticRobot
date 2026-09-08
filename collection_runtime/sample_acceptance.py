"""Runtime receipt for one sample accepted by the authoritative writer.

Acceptance is an admission boundary, not a durability or Episode-seal
receipt.  It means the sample contract was validated, the writer cursor
advanced, and every required write job was admitted by the writer that owns
the Episode.  Applications may use this receipt to align another runtime
state transition to the first formally accepted sample.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
import threading
from collections.abc import Mapping
from typing import Any


# The realtime control loop consumes terminal events even when the optional
# progress UI is disabled.  Keep a modest bounded queue so a writer fatal or
# contract rejection cannot disappear merely because no progress receipt
# consumer was configured.  The bound is intentionally finite: an unhealthy
# writer must fail closed rather than allowing unbounded metadata growth.
DEFAULT_SAMPLE_TERMINAL_EVENT_CAPACITY = 128


def _non_negative_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class AcceptedSampleReceipt:
    """Exact sample coordinate at the authoritative writer admission boundary."""

    episode_index: int
    sample_index: int
    episode_frame_index: int
    sample_timestamp: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "episode_index",
            _non_negative_int(self.episode_index, name="episode_index"),
        )
        object.__setattr__(
            self,
            "sample_index",
            _non_negative_int(self.sample_index, name="sample_index"),
        )
        object.__setattr__(
            self,
            "episode_frame_index",
            _non_negative_int(
                self.episode_frame_index,
                name="episode_frame_index",
            ),
        )
        object.__setattr__(
            self,
            "sample_timestamp",
            _finite(self.sample_timestamp, name="sample_timestamp"),
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class SampleTerminalEvent:
    """One terminal outcome for a sample admission.

    ``record_step_async`` may only establish queue admission.  The terminal
    event is emitted by the authoritative writer (or by the detached sample
    assembler when it cannot produce a sample) and is therefore the only
    boundary at which applications may advance a cursor or progress state.
    ``receipt`` is present for an accepted writer sample and absent for a
    rejected/failed sample.  Metadata is copied from the original admission so
    application leaves can correlate the event without owning writer state.
    """

    accepted: bool
    receipt: AcceptedSampleReceipt | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    terminal_kind: str = "accepted"

    def __post_init__(self) -> None:
        accepted = bool(self.accepted)
        receipt = self.receipt
        kind = str(self.terminal_kind).strip().lower()
        if kind not in {"accepted", "rejected", "fatal"}:
            raise ValueError("terminal_kind must be accepted, rejected, or fatal.")
        if accepted:
            if receipt is None:
                raise ValueError("accepted terminal event requires a receipt.")
            if kind != "accepted":
                raise ValueError("accepted terminal event must use terminal_kind=accepted.")
        elif kind == "accepted":
            raise ValueError("rejected terminal event cannot use terminal_kind=accepted.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "metadata", dict(self.metadata))
        object.__setattr__(self, "reason", str(self.reason or ""))
        object.__setattr__(self, "terminal_kind", kind)


class SampleTerminalEventBuffer:
    """Bounded, thread-safe retention for accepted/rejected sample outcomes."""

    def __init__(self, *, capacity: int = 0) -> None:
        self._capacity = _non_negative_int(capacity, name="capacity")
        self._events: deque[SampleTerminalEvent] = deque()
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, event: SampleTerminalEvent) -> None:
        if not isinstance(event, SampleTerminalEvent):
            raise TypeError("event must be SampleTerminalEvent.")
        if self._capacity <= 0:
            return
        with self._lock:
            if len(self._events) >= self._capacity:
                self._events.popleft()
                self._dropped += 1
            self._events.append(event)

    def drain(self) -> tuple[SampleTerminalEvent, ...]:
        with self._lock:
            if self._dropped:
                dropped = self._dropped
                self._events.clear()
                self._dropped = 0
                raise RuntimeError(
                    "collection sample terminal event buffer overflowed; "
                    f"dropped={dropped}"
                )
            events = tuple(self._events)
            self._events.clear()
            return events

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._dropped = 0


class SampleAcceptanceReceiptBuffer:
    """Thread-safe bounded retention for writer admission receipts.

    The writer remains the authority that decides when to call ``append``.
    This helper owns only bounded retention and fail-closed overflow handling;
    it does not infer acceptance from queue submission or counters.
    """

    def __init__(self, *, capacity: int = 0) -> None:
        self._capacity = _non_negative_int(capacity, name="capacity")
        self._receipts: deque[AcceptedSampleReceipt] = deque()
        self._dropped = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def append(self, receipt: AcceptedSampleReceipt) -> None:
        if not isinstance(receipt, AcceptedSampleReceipt):
            raise TypeError("receipt must be AcceptedSampleReceipt.")
        if self._capacity <= 0:
            return
        with self._lock:
            if len(self._receipts) >= self._capacity:
                self._receipts.popleft()
                self._dropped += 1
            self._receipts.append(receipt)

    def drain(self) -> tuple[AcceptedSampleReceipt, ...]:
        """Drain in acceptance order or fail if the exact boundary was lost."""

        with self._lock:
            if self._dropped:
                dropped = self._dropped
                self._dropped = 0
                self._receipts.clear()
                raise RuntimeError(
                    "collection sample acceptance receipt buffer overflowed; "
                    f"dropped={dropped}"
                )
            receipts = tuple(self._receipts)
            self._receipts.clear()
            return receipts

    def reset(self) -> None:
        """Forget receipts from a lifecycle that has already been closed."""

        with self._lock:
            self._receipts.clear()
            self._dropped = 0


__all__ = [
    "AcceptedSampleReceipt",
    "DEFAULT_SAMPLE_TERMINAL_EVENT_CAPACITY",
    "SampleTerminalEvent",
    "SampleTerminalEventBuffer",
    "SampleAcceptanceReceiptBuffer",
]

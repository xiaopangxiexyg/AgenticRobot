"""Bounded, rebuildable UI event cache for Control Station status clients."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StationUiEvent:
    event_sequence: int
    event_kind: str
    source_fact_id: str
    source_fact_hash: str
    state_revision: int
    status: str
    subject_id: str | None
    projected_at: float


@dataclass(frozen=True)
class StationUiEventSnapshot:
    events: tuple[StationUiEvent, ...]
    requested_after_sequence: int
    oldest_available_sequence: int | None
    latest_sequence: int | None
    next_sequence: int
    gap_detected: bool
    total_evicted: int


class BoundedStationEventLog:
    """A non-authoritative ring buffer; slow readers never backpressure runtime."""

    def __init__(
        self,
        *,
        capacity: int = 256,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or not 1 <= capacity <= 4096:
            raise ValueError("capacity must be an integer in [1, 4096].")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.capacity = capacity
        self.clock = clock
        self._events: deque[StationUiEvent] = deque(maxlen=capacity)
        self._next_sequence = 0
        self._total_evicted = 0
        self._lock = threading.Lock()

    def publish(
        self,
        *,
        event_kind: str,
        source_fact_id: str,
        source_fact_hash: str,
        state_revision: int,
        status: str,
        subject_id: str | None = None,
    ) -> StationUiEvent:
        kind = str(event_kind).strip()
        fact_id = str(source_fact_id).strip()
        fact_hash = str(source_fact_hash).strip()
        normalized_status = str(status).strip()
        if not kind or not fact_id or not fact_hash or not normalized_status:
            raise ValueError("UI event identity/status fields must be non-empty.")
        if isinstance(state_revision, bool) or not isinstance(state_revision, int) or state_revision < 0:
            raise ValueError("state_revision must be a non-negative integer.")
        normalized_subject = None if subject_id is None else str(subject_id).strip() or None
        with self._lock:
            projected_at = float(self.clock())
            if not math.isfinite(projected_at):
                raise ValueError("UI event clock must be finite.")
            if len(self._events) == self.capacity:
                self._total_evicted += 1
            event = StationUiEvent(
                event_sequence=self._next_sequence,
                event_kind=kind,
                source_fact_id=fact_id,
                source_fact_hash=fact_hash,
                state_revision=state_revision,
                status=normalized_status,
                subject_id=normalized_subject,
                projected_at=projected_at,
            )
            self._next_sequence += 1
            self._events.append(event)
            return event

    def snapshot(self, *, after_sequence: int = -1) -> StationUiEventSnapshot:
        if isinstance(after_sequence, bool) or not isinstance(after_sequence, int) or after_sequence < -1:
            raise ValueError("after_sequence must be an integer >= -1.")
        with self._lock:
            events = tuple(self._events)
            oldest = None if not events else events[0].event_sequence
            latest = None if not events else events[-1].event_sequence
            gap = oldest is not None and after_sequence < oldest - 1
            selected = tuple(
                event for event in events if event.event_sequence > after_sequence
            )
            return StationUiEventSnapshot(
                events=selected,
                requested_after_sequence=after_sequence,
                oldest_available_sequence=oldest,
                latest_sequence=latest,
                next_sequence=self._next_sequence,
                gap_detected=gap,
                total_evicted=self._total_evicted,
            )


__all__ = [
    "BoundedStationEventLog",
    "StationUiEvent",
    "StationUiEventSnapshot",
]

"""Bounded authoritative-source-neutral event journal for Skill protocol v2."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
import threading
from typing import Any

from agent_gateway.protocol._contract_validation import (
    finite_number,
    identifier,
    nonnegative_integer,
    positive_integer,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.event_contracts import (
    AgentGatewayEvent,
    AgentGatewayEventBatch,
)


class BoundedAgentGatewayEventJournal:
    """Assign monotonic sequence numbers and expose explicit replay gaps.

    Events are notification hints only.  A caller that observes a gap must
    reconcile through the authoritative Operation/catalog query surface.
    """

    def __init__(self, *, runtime_session_id: str, max_events: int = 4096) -> None:
        self.runtime_session_id = identifier(
            runtime_session_id,
            field_name="runtime_session_id",
        )
        self.max_events = positive_integer(max_events, field_name="max_events")
        self._lock = threading.RLock()
        self._events: deque[AgentGatewayEvent] = deque(maxlen=self.max_events)
        self._latest_sequence = 0

    @property
    def latest_sequence(self) -> int:
        with self._lock:
            return self._latest_sequence

    @property
    def oldest_available_sequence(self) -> int:
        with self._lock:
            if self._events:
                return self._events[0].sequence
            return self._latest_sequence + 1

    def append(
        self,
        *,
        event_kind: str,
        emitted_at: float,
        payload: Mapping[str, Any],
        operation_id: str | None = None,
        operation_hash: str | None = None,
        artifact_binding: AgentArtifactBinding | None = None,
        payload_schema_id: str | None = None,
    ) -> AgentGatewayEvent:
        timestamp = finite_number(emitted_at, field_name="emitted_at")
        with self._lock:
            event = AgentGatewayEvent(
                runtime_session_id=self.runtime_session_id,
                sequence=self._latest_sequence + 1,
                event_kind=event_kind,
                emitted_at=timestamp,
                operation_id=operation_id,
                operation_hash=operation_hash,
                artifact_binding=artifact_binding,
                payload_schema_id=payload_schema_id,
                payload=payload,
            )
            self._events.append(event)
            self._latest_sequence = event.sequence
            return event

    def replay(
        self,
        *,
        after_sequence: int,
        max_events: int,
    ) -> AgentGatewayEventBatch:
        cursor = nonnegative_integer(
            after_sequence,
            field_name="after_sequence",
        )
        limit = positive_integer(max_events, field_name="max_events")
        if limit > self.max_events:
            raise ValueError("max_events cannot exceed journal retention.")
        with self._lock:
            if cursor > self._latest_sequence:
                raise ValueError("after_sequence cannot exceed latest_sequence.")
            oldest = self.oldest_available_sequence
            events = tuple(
                event for event in self._events if event.sequence > cursor
            )[:limit]
            return AgentGatewayEventBatch(
                runtime_session_id=self.runtime_session_id,
                requested_after_sequence=cursor,
                oldest_available_sequence=oldest,
                latest_sequence=self._latest_sequence,
                gap_detected=cursor + 1 < oldest,
                events=events,
            )


__all__ = ["BoundedAgentGatewayEventJournal"]

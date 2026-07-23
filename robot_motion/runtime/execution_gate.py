"""Receiver-admitted execution-gate status and immutable snapshots."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from .control_admission import ControlAdmissionCoordinator, ControlCut, admitted_at_or_before
from .identity import RuntimeFactIdentity
from .immutable import deep_freeze


EXECUTION_GATE_EVENT_HASH_DOMAIN_V1 = "dexteleop.execution_gate_event.v1"
EXECUTION_GATE_SNAPSHOT_HASH_DOMAIN_V1 = "dexteleop.execution_gate_snapshot.v1"
EXECUTION_GATE_STATUSES = ("ready", "hold", "fault", "transitioning")


@dataclass(frozen=True)
class ExecutionGateEventDraft:
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    status: str
    produced_at: float
    produced_clock_domain_id: str
    reason: str
    source_event_ids: tuple[str, ...] = ()
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionGateEvent:
    event_id: str
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    gate_epoch: int
    status: str
    reason: str
    source_event_ids: tuple[str, ...]
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ExecutionGateSnapshot:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    included_admission_sequence_min: int | None
    included_admission_sequence_max: int | None
    status: str
    gate_epoch: int
    reason: str
    source_event_ids: tuple[str, ...]
    semantic_content_hash: str


class ExecutionGateStore:
    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        capacity: int = 1024,
        initial_status: str = "ready",
    ) -> None:
        if runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("ExecutionGateStore session must match coordinator.")
        status = str(initial_status).strip().lower()
        if status not in EXECUTION_GATE_STATUSES:
            raise ValueError("Unsupported initial execution gate status.")
        if int(capacity) <= 0:
            raise ValueError("ExecutionGateStore capacity must be positive.")
        self.runtime_session_id = runtime_session_id
        self.control_domain_id = coordinator.control_domain_id
        self.coordinator = coordinator
        self.capacity = int(capacity)
        self.initial_status = status
        self._records: list[ExecutionGateEvent] = []
        self._last_sequence = -1
        self._lock = threading.RLock()

    def admit(self, draft: ExecutionGateEventDraft) -> ExecutionGateEvent:
        if draft.runtime_session_id != self.runtime_session_id or draft.control_domain_id != self.control_domain_id:
            raise ValueError("Execution gate event belongs to another session/domain.")
        status = str(draft.status).strip().lower()
        if status not in EXECUTION_GATE_STATUSES:
            raise ValueError("Unsupported execution gate status.")
        if int(draft.sequence) < 0 or not math.isfinite(float(draft.produced_at)):
            raise ValueError("Execution gate sequence/time is invalid.")
        reason = str(draft.reason).strip()
        if not reason:
            raise ValueError("Execution gate reason must be non-empty.")
        if not str(draft.produced_clock_domain_id).strip():
            raise ValueError("Execution gate produced clock must be non-empty.")

        def build(available_at: float, admission_sequence: int) -> ExecutionGateEvent:
            with self._lock:
                gate_epoch = len(self._records) + 1
                identity = RuntimeFactIdentity(
                    runtime_session_id=self.runtime_session_id,
                    control_domain_id=self.control_domain_id,
                    producer_id="execution-gate",
                    producer_epoch=0,
                    sequence=int(draft.sequence),
                    namespace="execution_gate_event",
                )
                payload = {
                    "event_id": identity.fact_id,
                    "runtime_session_id": self.runtime_session_id,
                    "control_domain_id": self.control_domain_id,
                    "sequence": int(draft.sequence),
                    "gate_epoch": gate_epoch,
                    "status": status,
                    "reason": reason,
                    "source_event_ids": tuple(str(value) for value in draft.source_event_ids),
                    "produced_at": float(draft.produced_at),
                    "produced_clock_domain_id": str(draft.produced_clock_domain_id),
                    "available_at": available_at,
                    "available_clock_domain_id": self.coordinator.clock_domain.clock_domain_id,
                    "control_admission_sequence": admission_sequence,
                }
                return ExecutionGateEvent(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        payload, domain=EXECUTION_GATE_EVENT_HASH_DOMAIN_V1
                    ),
                    audit_metadata=deep_freeze(dict(draft.audit_metadata)),
                )

        return self.coordinator.commit(store=self, build=build)

    def _insert_admitted(self, record: ExecutionGateEvent) -> None:
        with self._lock:
            if len(self._records) >= self.capacity:
                raise ValueError("ExecutionGateStore is full.")
            if record.sequence <= self._last_sequence:
                raise ValueError("Execution gate event sequence must strictly increase.")
            self._records.append(record)
            self._last_sequence = record.sequence

    def records_at_or_before(self, cut: ControlCut) -> tuple[ExecutionGateEvent, ...]:
        with self._lock:
            return tuple(item for item in self._records if admitted_at_or_before(item, cut))

    def build_snapshot(
        self,
        *,
        cut: ControlCut,
        records: tuple[ExecutionGateEvent, ...] | None = None,
    ) -> ExecutionGateSnapshot:
        causal = self.records_at_or_before(cut) if records is None else tuple(records)
        latest = causal[-1] if causal else None
        sequences = tuple(item.control_admission_sequence for item in causal)
        payload = {
            "control_domain_id": self.control_domain_id,
            "cutoff_at": cut.cutoff_at,
            "cutoff_admission_sequence": cut.cutoff_admission_sequence,
            "included_admission_sequence_min": min(sequences) if sequences else None,
            "included_admission_sequence_max": max(sequences) if sequences else None,
            "status": self.initial_status if latest is None else latest.status,
            "gate_epoch": 0 if latest is None else latest.gate_epoch,
            "reason": "initial" if latest is None else latest.reason,
            "source_event_ids": () if latest is None else latest.source_event_ids,
        }
        return ExecutionGateSnapshot(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload, domain=EXECUTION_GATE_SNAPSHOT_HASH_DOMAIN_V1
            ),
        )


__all__ = [
    "EXECUTION_GATE_STATUSES",
    "ExecutionGateEvent",
    "ExecutionGateEventDraft",
    "ExecutionGateSnapshot",
    "ExecutionGateStore",
]

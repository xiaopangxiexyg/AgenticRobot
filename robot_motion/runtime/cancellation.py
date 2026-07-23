"""Per-control-domain execution cancellation and epoch contracts.

This module is deliberately independent of frontends and backend SDKs.  It
defines the in-memory linearization point and the receipts that production
workers will consume in later migration milestones; M1C does not wire these
objects into a real dispatch path.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from robot_motion.position_control.capability import (
    CANCELLATION_ACK_LEVELS,
    ExecutionCapabilityProfile,
)

from .identity import RuntimeFactIdentity
from .immutable import deep_freeze


EXECUTION_EPOCH_SNAPSHOT_HASH_DOMAIN_V1 = "dexteleop.execution_epoch_snapshot.v1"
EXECUTION_CANCELLATION_HASH_DOMAIN_V1 = "dexteleop.execution_cancellation.v1"
EXECUTION_RECOVERY_HASH_DOMAIN_V1 = "dexteleop.execution_recovery.v1"
EXECUTION_EPOCH_CHECK_HASH_DOMAIN_V1 = "dexteleop.execution_epoch_check.v1"
EXECUTION_CANCELLATION_ACK_HASH_DOMAIN_V1 = "dexteleop.execution_cancellation_ack.v1"
EXECUTION_FAULT_ESCALATION_HASH_DOMAIN_V1 = "dexteleop.execution_fault_escalation.v1"

CANCELLATION_REASONS = (
    "fault",
    "hold",
    "authority_handoff",
    "owner_rollback",
    "backend_not_ready",
    "manual_stop",
)
CANCELLATION_SCOPES = ("control_domain", "selection_groups")
EXECUTION_WORK_KINDS = (
    "reference_segment",
    "position_command",
    "backend_command",
    "safety_backend_command",
)
EPOCH_CHECKPOINTS = ("dequeue", "before_backend_call")
EPOCH_TRANSITION_KINDS = (
    "steady",
    "authority_update",
    "domain_cancellation",
    "authority_handoff_cancellation",
    "local_cancellation",
    "recovery",
)


def _identifier(value: str, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _finite_time(value: float, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


def _string_tuple(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(_identifier(value, name=name) for value in values)
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"{name} must not contain duplicates.")
    return normalized


def _generation_map(values: Mapping[str, int]) -> Mapping[str, int]:
    normalized = {
        _identifier(group, name="selection_group_id"): _non_negative_int(
            generation,
            name="segment_generation",
        )
        for group, generation in values.items()
    }
    return MappingProxyType(dict(sorted(normalized.items())))


@dataclass(frozen=True)
class EpochVector:
    authority_epoch: int
    execution_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_epoch",
            _non_negative_int(self.authority_epoch, name="authority_epoch"),
        )
        object.__setattr__(
            self,
            "execution_epoch",
            _non_negative_int(self.execution_epoch, name="execution_epoch"),
        )


def is_legal_epoch_transition(
    before: EpochVector,
    after: EpochVector,
    *,
    transition_kind: str,
) -> bool:
    """Validate the explicit v1 authority/execution transition matrix."""

    kind = str(transition_kind).strip()
    if kind not in EPOCH_TRANSITION_KINDS:
        raise ValueError("Unsupported epoch transition kind.")
    authority_delta = after.authority_epoch - before.authority_epoch
    execution_delta = after.execution_epoch - before.execution_epoch
    expected = {
        "steady": (0, 0),
        "authority_update": ("positive", 0),
        "domain_cancellation": (0, 1),
        "authority_handoff_cancellation": ("positive", 1),
        "local_cancellation": (0, 0),
        "recovery": (0, 1),
    }[kind]
    expected_authority, expected_execution = expected
    authority_matches = (
        authority_delta > 0
        if expected_authority == "positive"
        else authority_delta == expected_authority
    )
    return bool(authority_matches and execution_delta == expected_execution)


@dataclass(frozen=True)
class ExecutionEpochSnapshot:
    runtime_session_id: str
    control_domain_id: str
    execution_epoch: int
    segment_generation_by_group: Mapping[str, int]
    fault_latched: bool
    last_cancellation_id: str | None
    semantic_content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_generation_by_group",
            _generation_map(self.segment_generation_by_group),
        )


@dataclass(frozen=True)
class ExecutionCancellationRequest:
    reason: str
    authority_epoch: int
    requested_scope: str = "control_domain"
    selection_group_ids: tuple[str, ...] = ()
    invalidated_decision_ids: tuple[str, ...] = ()
    invalidated_reference_segment_ids: tuple[str, ...] = ()
    invalidated_queued_command_ids: tuple[str, ...] = ()
    source_event_ids: tuple[str, ...] = ()
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        reason = str(self.reason).strip()
        scope = str(self.requested_scope).strip()
        if reason not in CANCELLATION_REASONS:
            raise ValueError("Unsupported cancellation reason.")
        if scope not in CANCELLATION_SCOPES:
            raise ValueError("Unsupported cancellation scope.")
        groups = _string_tuple(tuple(self.selection_group_ids), name="selection_group_id")
        if scope == "selection_groups" and not groups:
            raise ValueError("selection_groups cancellation requires at least one group.")
        if scope == "control_domain" and groups:
            raise ValueError("control_domain cancellation cannot carry selection groups.")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "requested_scope", scope)
        object.__setattr__(self, "selection_group_ids", groups)
        object.__setattr__(
            self,
            "authority_epoch",
            _non_negative_int(self.authority_epoch, name="authority_epoch"),
        )
        for name in (
            "invalidated_decision_ids",
            "invalidated_reference_segment_ids",
            "invalidated_queued_command_ids",
            "source_event_ids",
        ):
            object.__setattr__(
                self,
                name,
                _string_tuple(tuple(getattr(self, name)), name=name),
            )
        object.__setattr__(self, "audit_metadata", deep_freeze(dict(self.audit_metadata)))


@dataclass(frozen=True)
class ExecutionCancellation:
    cancellation_id: str
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    reason: str
    requested_scope: str
    effective_scope: str
    selection_group_ids: tuple[str, ...]
    scope_escalation_reason: str | None
    previous_execution_epoch: int
    execution_epoch: int
    authority_epoch_evidence: int
    segment_generation_before: Mapping[str, int]
    segment_generation_after: Mapping[str, int]
    invalidated_decision_ids: tuple[str, ...]
    invalidated_reference_segment_ids: tuple[str, ...]
    invalidated_queued_command_ids: tuple[str, ...]
    source_event_ids: tuple[str, ...]
    effective_at: float
    effective_clock_domain_id: str
    acknowledgement_deadline_at: float
    required_ack_levels: tuple[str, ...]
    requests_backend_hold: bool
    fault_latched: bool
    affects_decision_rate: bool
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "segment_generation_before",
            _generation_map(self.segment_generation_before),
        )
        object.__setattr__(
            self,
            "segment_generation_after",
            _generation_map(self.segment_generation_after),
        )
        object.__setattr__(self, "audit_metadata", deep_freeze(dict(self.audit_metadata)))


@dataclass(frozen=True)
class ExecutionRecovery:
    recovery_id: str
    runtime_session_id: str
    control_domain_id: str
    sequence: int
    previous_execution_epoch: int
    execution_epoch: int
    authority_epoch_evidence: int
    recovered_at: float
    recovered_clock_domain_id: str
    source_event_ids: tuple[str, ...]
    semantic_content_hash: str


class ExecutionEpochRegistry:
    """Atomic execution epoch and selection-group generation registry."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        control_domain_id: str,
        clock_domain_id: str,
        initial_execution_epoch: int = 0,
        initial_segment_generation_by_group: Mapping[str, int] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.runtime_session_id = _identifier(
            runtime_session_id,
            name="runtime_session_id",
        )
        self.control_domain_id = _identifier(
            control_domain_id,
            name="control_domain_id",
        )
        self.clock_domain_id = _identifier(clock_domain_id, name="clock_domain_id")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._execution_epoch = _non_negative_int(
            initial_execution_epoch,
            name="initial_execution_epoch",
        )
        self._segment_generation_by_group = dict(
            _generation_map(initial_segment_generation_by_group or {})
        )
        self._fault_latched = False
        self._last_cancellation_id: str | None = None
        self._next_sequence = 0
        self._lock = threading.RLock()

    def snapshot(self) -> ExecutionEpochSnapshot:
        with self._lock:
            payload = {
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "execution_epoch": self._execution_epoch,
                "segment_generation_by_group": dict(self._segment_generation_by_group),
                "fault_latched": self._fault_latched,
                "last_cancellation_id": self._last_cancellation_id,
            }
            return ExecutionEpochSnapshot(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=EXECUTION_EPOCH_SNAPSHOT_HASH_DOMAIN_V1,
                ),
            )

    def cancel(
        self,
        request: ExecutionCancellationRequest,
        *,
        capability: ExecutionCapabilityProfile,
    ) -> ExecutionCancellation:
        if not isinstance(request, ExecutionCancellationRequest):
            raise TypeError("request must be an ExecutionCancellationRequest.")
        if not isinstance(capability, ExecutionCapabilityProfile):
            raise TypeError("capability must be an ExecutionCapabilityProfile.")

        with self._lock:
            effective_at = _finite_time(self._clock(), name="effective_at")
            previous_epoch = self._execution_epoch
            previous_generations = dict(self._segment_generation_by_group)
            next_epoch = previous_epoch
            next_generations = dict(previous_generations)
            effective_scope = request.requested_scope
            escalation_reason: str | None = None

            if request.reason == "fault" and request.requested_scope != "control_domain":
                effective_scope = "control_domain"
                escalation_reason = "fault_requires_domain_cancellation"
                next_epoch += 1
            elif request.requested_scope == "selection_groups":
                known_groups = set(capability.validated_selection_groups)
                unknown_groups = set(request.selection_group_ids).difference(known_groups)
                if unknown_groups:
                    raise ValueError(
                        f"Cancellation references unvalidated selection groups: {sorted(unknown_groups)!r}."
                    )
                if capability.supports_local_group_stop:
                    for group in request.selection_group_ids:
                        next_generations[group] = next_generations.get(group, 0) + 1
                else:
                    effective_scope = "control_domain"
                    escalation_reason = "backend_does_not_support_local_group_stop"
                    next_epoch += 1
            else:
                next_epoch += 1

            next_sequence = self._next_sequence
            identity = RuntimeFactIdentity(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                producer_id="execution-epoch-registry",
                producer_epoch=0,
                sequence=next_sequence,
                namespace="execution_cancellation",
            )
            fault_latched = self._fault_latched or request.reason == "fault"
            payload = {
                "cancellation_id": identity.fact_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "sequence": next_sequence,
                "reason": request.reason,
                "requested_scope": request.requested_scope,
                "effective_scope": effective_scope,
                "selection_group_ids": request.selection_group_ids,
                "scope_escalation_reason": escalation_reason,
                "previous_execution_epoch": previous_epoch,
                "execution_epoch": next_epoch,
                "authority_epoch_evidence": request.authority_epoch,
                "segment_generation_before": previous_generations,
                "segment_generation_after": next_generations,
                "invalidated_decision_ids": request.invalidated_decision_ids,
                "invalidated_reference_segment_ids": request.invalidated_reference_segment_ids,
                "invalidated_queued_command_ids": request.invalidated_queued_command_ids,
                "source_event_ids": request.source_event_ids,
                "effective_at": effective_at,
                "effective_clock_domain_id": self.clock_domain_id,
                "acknowledgement_deadline_at": (
                    effective_at + capability.cancellation_ack_deadline_sec
                ),
                "required_ack_levels": capability.required_cancellation_ack_levels,
                "requests_backend_hold": effective_scope == "control_domain",
                "fault_latched": fault_latched,
                "affects_decision_rate": False,
            }
            event = ExecutionCancellation(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=EXECUTION_CANCELLATION_HASH_DOMAIN_V1,
                ),
                audit_metadata=request.audit_metadata,
            )

            # This assignment is the domain-cancellation linearization point.
            # Local cancellation is linearized by the generation-map update
            # under the same lock.
            self._execution_epoch = next_epoch
            self._segment_generation_by_group = next_generations
            self._fault_latched = fault_latched
            self._last_cancellation_id = event.cancellation_id
            self._next_sequence += 1
            return event

    def recover(
        self,
        *,
        authority_epoch: int,
        source_event_ids: tuple[str, ...],
    ) -> ExecutionRecovery:
        normalized_authority_epoch = _non_negative_int(
            authority_epoch,
            name="authority_epoch",
        )
        normalized_source_events = _string_tuple(source_event_ids, name="source_event_ids")
        if not normalized_source_events:
            raise ValueError("Recovery requires explicit source-event evidence.")
        with self._lock:
            recovered_at = _finite_time(self._clock(), name="recovered_at")
            previous_epoch = self._execution_epoch
            next_epoch = previous_epoch + 1
            next_sequence = self._next_sequence
            identity = RuntimeFactIdentity(
                runtime_session_id=self.runtime_session_id,
                control_domain_id=self.control_domain_id,
                producer_id="execution-epoch-registry",
                producer_epoch=0,
                sequence=next_sequence,
                namespace="execution_recovery",
            )
            payload = {
                "recovery_id": identity.fact_id,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "sequence": next_sequence,
                "previous_execution_epoch": previous_epoch,
                "execution_epoch": next_epoch,
                "authority_epoch_evidence": normalized_authority_epoch,
                "recovered_at": recovered_at,
                "recovered_clock_domain_id": self.clock_domain_id,
                "source_event_ids": normalized_source_events,
            }
            event = ExecutionRecovery(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=EXECUTION_RECOVERY_HASH_DOMAIN_V1,
                ),
            )
            self._execution_epoch = next_epoch
            self._fault_latched = False
            self._next_sequence += 1
            return event


@dataclass(frozen=True)
class ExecutionWorkItem:
    """Epoch envelope for a reference, command tick, or backend command."""

    work_item_id: str
    work_kind: str
    control_domain_id: str
    authority_epoch: int
    execution_epoch: int
    segment_generation_by_group: Mapping[str, int]
    source_decision_id: str | None = None
    source_reference_segment_id: str | None = None
    source_safety_operation_id: str | None = None
    source_safety_tick_evidence_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "work_item_id", _identifier(self.work_item_id, name="work_item_id"))
        kind = str(self.work_kind).strip()
        if kind not in EXECUTION_WORK_KINDS:
            raise ValueError("Unsupported execution work kind.")
        object.__setattr__(self, "work_kind", kind)
        object.__setattr__(
            self,
            "control_domain_id",
            _identifier(self.control_domain_id, name="control_domain_id"),
        )
        object.__setattr__(
            self,
            "authority_epoch",
            _non_negative_int(self.authority_epoch, name="authority_epoch"),
        )
        object.__setattr__(
            self,
            "execution_epoch",
            _non_negative_int(self.execution_epoch, name="execution_epoch"),
        )
        object.__setattr__(
            self,
            "segment_generation_by_group",
            _generation_map(self.segment_generation_by_group),
        )
        for name in (
            "source_decision_id",
            "source_reference_segment_id",
            "source_safety_operation_id",
            "source_safety_tick_evidence_id",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _identifier(value, name=name))
        safety_lineage = (
            self.source_safety_operation_id,
            self.source_safety_tick_evidence_id,
        )
        if self.work_kind == "safety_backend_command":
            if any(value is None for value in safety_lineage):
                raise ValueError(
                    "safety_backend_command requires operation and tick evidence lineage."
                )
            if self.source_decision_id is not None:
                raise ValueError(
                    "safety_backend_command cannot inherit ordinary decision lineage."
                )
        elif any(value is not None for value in safety_lineage):
            raise ValueError(
                "Safety-operation lineage is only valid for safety_backend_command."
            )


@dataclass(frozen=True)
class ExecutionEpochCheckReceipt:
    work_item_id: str
    work_kind: str
    checkpoint: str
    checked_at: float
    control_domain_id: str
    item_execution_epoch: int
    current_execution_epoch: int
    accepted: bool
    status: str
    stale_selection_group_ids: tuple[str, ...]
    semantic_content_hash: str


class ExecutionEpochGuard:
    """Shared worker guard used at dequeue and immediately before SDK call."""

    def __init__(
        self,
        registry: ExecutionEpochRegistry,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(registry, ExecutionEpochRegistry):
            raise TypeError("registry must be an ExecutionEpochRegistry.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.registry = registry
        self._clock = clock

    def check_at_dequeue(self, item: ExecutionWorkItem) -> ExecutionEpochCheckReceipt:
        return self._check(item, checkpoint="dequeue")

    def check_before_backend_call(
        self,
        item: ExecutionWorkItem,
    ) -> ExecutionEpochCheckReceipt:
        return self._check(item, checkpoint="before_backend_call")

    def _check(
        self,
        item: ExecutionWorkItem,
        *,
        checkpoint: str,
    ) -> ExecutionEpochCheckReceipt:
        if not isinstance(item, ExecutionWorkItem):
            raise TypeError("item must be an ExecutionWorkItem.")
        current = self.registry.snapshot()
        checked_at = _finite_time(self._clock(), name="checked_at")
        stale_groups = tuple(
            sorted(
                group
                for group, generation in item.segment_generation_by_group.items()
                if current.segment_generation_by_group.get(group, 0) != generation
            )
        )
        if item.control_domain_id != current.control_domain_id:
            status = "wrong_control_domain"
        elif item.execution_epoch != current.execution_epoch:
            status = "execution_epoch_mismatch"
        elif stale_groups:
            status = "segment_generation_mismatch"
        elif current.fault_latched and item.work_kind != "safety_backend_command":
            status = "fault_latched"
        else:
            status = "accepted"
        payload = {
            "work_item_id": item.work_item_id,
            "work_kind": item.work_kind,
            "checkpoint": checkpoint,
            "checked_at": checked_at,
            "control_domain_id": current.control_domain_id,
            "item_execution_epoch": item.execution_epoch,
            "current_execution_epoch": current.execution_epoch,
            "accepted": status == "accepted",
            "status": status,
            "stale_selection_group_ids": stale_groups,
        }
        return ExecutionEpochCheckReceipt(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=EXECUTION_EPOCH_CHECK_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class CancellationAcknowledgement:
    cancellation_id: str
    acknowledgement_level: str
    acknowledged_at: float
    clock_domain_id: str
    source_id: str
    accepted: bool
    status: str
    semantic_content_hash: str


@dataclass(frozen=True)
class ExecutionCancellationReceipt:
    cancellation_id: str
    required_ack_levels: tuple[str, ...]
    acknowledged_levels: tuple[str, ...]
    pending_levels: tuple[str, ...]
    rejected_levels: tuple[str, ...]
    acknowledgement_deadline_at: float
    observed_at: float
    complete: bool
    timed_out: bool
    requires_fault_escalation: bool
    late_levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExecutionFaultEscalationRequest:
    escalation_id: str
    cancellation_id: str
    runtime_session_id: str
    control_domain_id: str
    execution_epoch: int
    authority_epoch_evidence: int
    reason: str
    requested_gate_status: str
    missing_ack_levels: tuple[str, ...]
    rejected_ack_levels: tuple[str, ...]
    observed_at: float
    semantic_content_hash: str


class CancellationAcknowledgementTracker:
    """Collect distinct ack evidence without conflating request and stop state."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic) -> None:
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._events: dict[str, ExecutionCancellation] = {}
        self._acks: dict[str, dict[str, CancellationAcknowledgement]] = {}
        self._lock = threading.RLock()

    def register(self, event: ExecutionCancellation) -> None:
        if not isinstance(event, ExecutionCancellation):
            raise TypeError("event must be an ExecutionCancellation.")
        with self._lock:
            if event.cancellation_id in self._events:
                raise ValueError("Cancellation is already registered.")
            self._events[event.cancellation_id] = event
            self._acks[event.cancellation_id] = {}

    def acknowledge(
        self,
        cancellation_id: str,
        *,
        acknowledgement_level: str,
        clock_domain_id: str,
        source_id: str,
        accepted: bool = True,
        status: str = "acknowledged",
        acknowledged_at: float | None = None,
    ) -> CancellationAcknowledgement:
        level = str(acknowledgement_level).strip()
        if level not in CANCELLATION_ACK_LEVELS:
            raise ValueError("Unsupported cancellation acknowledgement level.")
        with self._lock:
            event = self._events.get(cancellation_id)
            if event is None:
                raise KeyError("Unknown cancellation ID.")
            if level in self._acks[cancellation_id]:
                raise ValueError("Acknowledgement level is already recorded.")
            timestamp = _finite_time(
                self._clock() if acknowledged_at is None else acknowledged_at,
                name="acknowledged_at",
            )
            if timestamp < event.effective_at:
                raise ValueError("Acknowledgement cannot predate cancellation.")
            payload = {
                "cancellation_id": cancellation_id,
                "acknowledgement_level": level,
                "acknowledged_at": timestamp,
                "clock_domain_id": _identifier(clock_domain_id, name="clock_domain_id"),
                "source_id": _identifier(source_id, name="source_id"),
                "accepted": bool(accepted),
                "status": _identifier(status, name="status"),
            }
            acknowledgement = CancellationAcknowledgement(
                **payload,
                semantic_content_hash=canonical_content_hash(
                    payload,
                    domain=EXECUTION_CANCELLATION_ACK_HASH_DOMAIN_V1,
                ),
            )
            self._acks[cancellation_id][level] = acknowledgement
            return acknowledgement

    def receipt(
        self,
        cancellation_id: str,
        *,
        observed_at: float | None = None,
    ) -> ExecutionCancellationReceipt:
        with self._lock:
            event = self._events.get(cancellation_id)
            if event is None:
                raise KeyError("Unknown cancellation ID.")
            now = _finite_time(
                self._clock() if observed_at is None else observed_at,
                name="observed_at",
            )
            acknowledgements = self._acks[cancellation_id]
            required = event.required_ack_levels
            late = tuple(
                level
                for level in required
                if level in acknowledgements
                and acknowledgements[level].accepted
                and acknowledgements[level].acknowledged_at
                > event.acknowledgement_deadline_at
            )
            acknowledged = tuple(
                level
                for level in required
                if level in acknowledgements
                and acknowledgements[level].accepted
                and level not in late
            )
            rejected = tuple(
                level
                for level in required
                if level in acknowledgements and not acknowledgements[level].accepted
            )
            pending = tuple(
                level
                for level in required
                if level not in acknowledged and level not in rejected
            )
            complete = not pending and not rejected
            timed_out = bool(
                late or (not complete and now > event.acknowledgement_deadline_at)
            )
            return ExecutionCancellationReceipt(
                cancellation_id=cancellation_id,
                required_ack_levels=required,
                acknowledged_levels=acknowledged,
                pending_levels=pending,
                rejected_levels=rejected,
                acknowledgement_deadline_at=event.acknowledgement_deadline_at,
                observed_at=now,
                complete=complete,
                timed_out=timed_out,
                requires_fault_escalation=bool(rejected or timed_out),
                late_levels=late,
            )

    def fault_escalation_request(
        self,
        cancellation_id: str,
        *,
        observed_at: float | None = None,
    ) -> ExecutionFaultEscalationRequest | None:
        """Return a typed FAULT request; gate admission remains a later owner."""

        receipt = self.receipt(cancellation_id, observed_at=observed_at)
        if not receipt.requires_fault_escalation:
            return None
        with self._lock:
            event = self._events[cancellation_id]
            reason = (
                "cancellation_ack_rejected"
                if receipt.rejected_levels
                else "cancellation_ack_deadline_missed"
            )
            payload = {
                "cancellation_id": cancellation_id,
                "runtime_session_id": event.runtime_session_id,
                "control_domain_id": event.control_domain_id,
                "execution_epoch": event.execution_epoch,
                "authority_epoch_evidence": event.authority_epoch_evidence,
                "reason": reason,
                "requested_gate_status": "fault",
                "missing_ack_levels": receipt.pending_levels,
                "rejected_ack_levels": receipt.rejected_levels,
                "observed_at": receipt.observed_at,
            }
            semantic_hash = canonical_content_hash(
                payload,
                domain=EXECUTION_FAULT_ESCALATION_HASH_DOMAIN_V1,
            )
            return ExecutionFaultEscalationRequest(
                escalation_id=f"execution-fault-escalation:{semantic_hash}",
                **payload,
                semantic_content_hash=semantic_hash,
            )


__all__ = [
    "CANCELLATION_REASONS",
    "CANCELLATION_SCOPES",
    "CancellationAcknowledgement",
    "CancellationAcknowledgementTracker",
    "EpochVector",
    "ExecutionCancellation",
    "ExecutionCancellationReceipt",
    "ExecutionCancellationRequest",
    "ExecutionEpochCheckReceipt",
    "ExecutionEpochGuard",
    "ExecutionEpochRegistry",
    "ExecutionEpochSnapshot",
    "ExecutionFaultEscalationRequest",
    "ExecutionRecovery",
    "ExecutionWorkItem",
    "is_legal_epoch_transition",
]

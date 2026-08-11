"""Bounded in-memory projections for the B2a no-I/O Agent Gateway.

The store is deliberately not a work queue.  It retains at most one active
operation plus a bounded terminal history, and it never invokes a backend,
changes Station state, owns authority, or issues a motion permit.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .access_policy import SkillAccessDecision
from .contracts import AgentSkillOperation, AgentSkillRequest

AGENT_GATEWAY_OPERATION_ID_HASH_DOMAIN_V1 = (
    "dexteleop.agent_gateway.operation_id.v1"
)
AGENT_GATEWAY_CONTROL_KINDS_V1 = (
    "cancel",
    "disconnect",
    "hold",
    "service_close",
    "timeout",
)

_WORKER_STATUS_TRANSITIONS = {
    "admitted": frozenset({"activated", "failed"}),
    "activated": frozenset({"running", "failed"}),
    "running": frozenset(
        {"running", "awaiting_evaluation", "completed", "failed"}
    ),
}


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _finite_time(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("current_station_revision must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class OperationStoreOutcome:
    """One immutable execution_result from admission, lookup, or a projection CAS."""

    accepted: bool
    reason_code: str
    operation: AgentSkillOperation | None = None
    idempotent_replay: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean.")
        object.__setattr__(
            self,
            "reason_code",
            _required_text(self.reason_code, field_name="reason_code"),
        )
        if self.operation is not None and not isinstance(
            self.operation, AgentSkillOperation
        ):
            raise TypeError("operation must be an AgentSkillOperation or None.")
        if not isinstance(self.idempotent_replay, bool):
            raise TypeError("idempotent_replay must be a boolean.")
        if self.idempotent_replay and (not self.accepted or self.operation is None):
            raise ValueError(
                "An idempotent replay must accept and return its original operation."
            )


@dataclass(frozen=True)
class _ControlReservation:
    control_kind: str
    terminal_status: str


class BoundedInMemoryAgentSkillOperationStore:
    """Thread-safe one-active-operation projection store with no work queue."""

    def __init__(
        self,
        *,
        max_retained_operations: int = 64,
        max_total_admissions: int | None = None,
        max_admissions_by_route_hash: Mapping[str, int] | None = None,
    ) -> None:
        if (
            isinstance(max_retained_operations, bool)
            or not isinstance(max_retained_operations, int)
            or not 1 <= max_retained_operations <= 4096
        ):
            raise ValueError(
                "max_retained_operations must be an integer in [1, 4096]."
            )
        if max_total_admissions is not None and (
            isinstance(max_total_admissions, bool)
            or not isinstance(max_total_admissions, int)
            or not 1 <= max_total_admissions <= 4096
        ):
            raise ValueError(
                "max_total_admissions must be None or an integer in [1, 4096]."
            )
        route_limits: dict[str, int] = {}
        for raw_route_hash, raw_limit in dict(
            max_admissions_by_route_hash or {}
        ).items():
            route_hash = _required_text(
                raw_route_hash,
                field_name="max_admissions_by_route_hash key",
            )
            if (
                isinstance(raw_limit, bool)
                or not isinstance(raw_limit, int)
                or not 1 <= raw_limit <= 4096
            ):
                raise ValueError(
                    "Route admission limits must be integers in [1, 4096]."
                )
            route_limits[route_hash] = raw_limit
        self.max_retained_operations = max_retained_operations
        self.max_total_admissions = max_total_admissions
        self.max_admissions_by_route_hash = dict(route_limits)
        self._lock = threading.RLock()
        self._operations: OrderedDict[str, AgentSkillOperation] = OrderedDict()
        self._requests: dict[str, AgentSkillRequest] = {}
        self._operation_id_by_idempotency: dict[tuple[str, str], str] = {}
        self._request_hash_by_idempotency: dict[tuple[str, str], str] = {}
        self._last_client_sequence: dict[str, int] = {}
        self._control_reservations: dict[str, _ControlReservation] = {}
        self._active_operation_id: str | None = None
        self._latest_station_revision: int | None = None
        self._total_evicted = 0
        self._total_admitted = 0
        self._admissions_by_route_hash: dict[str, int] = {}

    @property
    def retained_operation_count(self) -> int:
        with self._lock:
            return len(self._operations)

    @property
    def total_evicted(self) -> int:
        with self._lock:
            return self._total_evicted

    @property
    def total_admitted(self) -> int:
        with self._lock:
            return self._total_admitted

    def admissions_for_route_hash(self, route_hash: str) -> int:
        normalized = _required_text(route_hash, field_name="route_hash")
        with self._lock:
            return self._admissions_by_route_hash.get(normalized, 0)

    @property
    def active_operation_id(self) -> str | None:
        with self._lock:
            return self._active_operation_id

    @property
    def latest_station_revision(self) -> int | None:
        with self._lock:
            return self._latest_station_revision

    def operations(self) -> tuple[AgentSkillOperation, ...]:
        with self._lock:
            return tuple(self._operations.values())

    def get(self, operation_id: str) -> AgentSkillOperation | None:
        normalized_id = _required_text(operation_id, field_name="operation_id")
        with self._lock:
            return self._operations.get(normalized_id)

    def active_operation(self) -> AgentSkillOperation | None:
        with self._lock:
            if self._active_operation_id is None:
                return None
            return self._operations[self._active_operation_id]

    def request_for_operation(self, operation_id: str) -> AgentSkillRequest | None:
        normalized_id = _required_text(operation_id, field_name="operation_id")
        with self._lock:
            return self._requests.get(normalized_id)

    def lookup_idempotency(self, request: AgentSkillRequest) -> OperationStoreOutcome | None:
        """Return an exact replay/conflict, or ``None`` for an unseen key."""

        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        key = (request.client_id, request.idempotency_key)
        request_hash = request.semantic_content_hash
        with self._lock:
            known_hash = self._request_hash_by_idempotency.get(key)
            if known_hash is None:
                return None
            operation_id = self._operation_id_by_idempotency[key]
            operation = self._operations[operation_id]
            if known_hash != request_hash:
                return OperationStoreOutcome(
                    accepted=False,
                    reason_code="idempotency_key_conflict",
                    operation=operation,
                )
            return OperationStoreOutcome(
                accepted=True,
                reason_code="idempotent_operation_lookup",
                operation=operation,
                idempotent_replay=True,
            )

    def admit(
        self,
        *,
        request: AgentSkillRequest,
        access_decision: SkillAccessDecision,
        current_station_revision: int,
        admitted_at: float,
    ) -> OperationStoreOutcome:
        """Atomically admit one request after a matching pure access decision."""

        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if not isinstance(access_decision, SkillAccessDecision):
            raise TypeError("access_decision must be a SkillAccessDecision.")
        revision = _revision(current_station_revision)
        timestamp = _finite_time(admitted_at, field_name="admitted_at")
        request_hash = request.semantic_content_hash
        key = (request.client_id, request.idempotency_key)

        with self._lock:
            replay = self.lookup_idempotency(request)
            if replay is not None:
                return replay
            if not access_decision.accepted:
                return OperationStoreOutcome(False, "access_not_accepted")
            if access_decision.request_hash != request_hash:
                return OperationStoreOutcome(False, "access_request_hash_mismatch")
            if access_decision.effective_timeout_ms is None:
                return OperationStoreOutcome(False, "access_timeout_missing")
            if request.expected_station_revision != revision:
                return OperationStoreOutcome(
                    False,
                    "station_revision_changed_after_access",
                )
            if (
                self._latest_station_revision is not None
                and revision < self._latest_station_revision
            ):
                return OperationStoreOutcome(
                    False,
                    "station_revision_regressed",
                )

            previous_sequence = self._last_client_sequence.get(request.client_id)
            if (
                previous_sequence is not None
                and request.client_sequence <= previous_sequence
            ):
                return OperationStoreOutcome(
                    False,
                    "duplicate_or_out_of_order_client_sequence",
                )
            if self._active_operation_id is not None:
                return OperationStoreOutcome(
                    False,
                    "active_operation_exists_no_queue",
                    self._operations[self._active_operation_id],
                )
            if (
                self.max_total_admissions is not None
                and self._total_admitted >= self.max_total_admissions
            ):
                return OperationStoreOutcome(
                    False,
                    "operation_admission_limit_reached",
                )
            route_limit = self.max_admissions_by_route_hash.get(
                request.route_hash
            )
            route_admissions = self._admissions_by_route_hash.get(
                request.route_hash,
                0,
            )
            if route_limit is not None and route_admissions >= route_limit:
                return OperationStoreOutcome(
                    False,
                    "route_admission_limit_reached",
                )
            if not self._reserve_retention_slot_locked():
                return OperationStoreOutcome(False, "operation_retention_full")

            operation_id = self._operation_id(request)
            if operation_id in self._operations:
                raise RuntimeError("Operation ID collision in bounded store.")
            deadline_at = timestamp + (
                access_decision.effective_timeout_ms / 1000.0
            )
            operation = AgentSkillOperation(
                operation_id=operation_id,
                request_hash=request_hash,
                runtime_session_id=request.runtime_session_id,
                status="admitted",
                reason_code="gateway_operation_admitted",
                station_revision_at_admission=revision,
                current_station_revision=revision,
                started_at=timestamp,
                updated_at=timestamp,
                deadline_at=deadline_at,
                execution_receipt_refs=(
                    "agent-access-decision:"
                    + access_decision.semantic_content_hash,
                ),
                progress_summary={"phase": "admitted"},
            )
            self._operations[operation_id] = operation
            self._total_admitted += 1
            self._admissions_by_route_hash[request.route_hash] = (
                route_admissions + 1
            )
            self._requests[operation_id] = request
            self._operation_id_by_idempotency[key] = operation_id
            self._request_hash_by_idempotency[key] = request_hash
            self._last_client_sequence[request.client_id] = request.client_sequence
            self._active_operation_id = operation_id
            self._latest_station_revision = revision
            return OperationStoreOutcome(
                accepted=True,
                reason_code="operation_admitted_not_executed",
                operation=operation,
            )

    def transition_worker_projection(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        next_status: str,
        reason_code: str,
        current_station_revision: int,
        updated_at: float,
        lower_operation_id: str | None = None,
        execution_receipt_refs: tuple[str, ...] = (),
        progress_summary: Mapping[str, Any] | None = None,
        execution_result: Mapping[str, Any] | None = None,
    ) -> OperationStoreOutcome:
        """CAS one worker projection; reserved safety control always wins."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        expected_hash = _required_text(
            expected_operation_hash,
            field_name="expected_operation_hash",
        )
        normalized_status = _required_text(next_status, field_name="next_status")
        normalized_reason = _required_text(reason_code, field_name="reason_code")
        revision = _revision(current_station_revision)
        timestamp = _finite_time(updated_at, field_name="updated_at")
        if lower_operation_id is not None:
            lower_operation_id = _required_text(
                lower_operation_id,
                field_name="lower_operation_id",
            )

        with self._lock:
            current = self._operations.get(normalized_id)
            if current is None:
                return OperationStoreOutcome(False, "operation_not_found")
            if current.semantic_content_hash != expected_hash:
                return OperationStoreOutcome(
                    False,
                    "operation_hash_mismatch",
                    current,
                )
            if current.terminal:
                return OperationStoreOutcome(False, "operation_terminal", current)
            if normalized_id in self._control_reservations:
                return OperationStoreOutcome(
                    False,
                    "operation_control_reserved",
                    current,
                )
            allowed = _WORKER_STATUS_TRANSITIONS.get(current.status, frozenset())
            if normalized_status not in allowed:
                return OperationStoreOutcome(
                    False,
                    "operation_status_transition_not_allowed",
                    current,
                )
            if revision < current.current_station_revision:
                return OperationStoreOutcome(
                    False,
                    "station_revision_regressed",
                    current,
                )
            if timestamp < current.updated_at:
                return OperationStoreOutcome(
                    False,
                    "operation_timestamp_regressed",
                    current,
                )
            if timestamp >= current.deadline_at and normalized_status in {
                "activated",
                "running",
                "awaiting_evaluation",
                "completed",
            }:
                return OperationStoreOutcome(
                    False,
                    "operation_deadline_expired",
                    current,
                )
            if current.lower_operation_id is not None and lower_operation_id not in {
                None,
                current.lower_operation_id,
            }:
                return OperationStoreOutcome(
                    False,
                    "lower_operation_id_mismatch",
                    current,
                )
            resolved_lower_id = current.lower_operation_id or lower_operation_id
            if normalized_status in {"awaiting_evaluation", "completed"} and (
                execution_result is None
            ):
                return OperationStoreOutcome(
                    False,
                    "terminal_result_required",
                    current,
                )
            if normalized_status not in {
                "awaiting_evaluation",
                "failed",
                "completed",
            } and execution_result is not None:
                return OperationStoreOutcome(
                    False,
                    "nonterminal_result_forbidden",
                    current,
                )

            next_operation = replace(
                current,
                status=normalized_status,
                reason_code=normalized_reason,
                execution_outcome=(
                    "completed"
                    if normalized_status in {"awaiting_evaluation", "completed"}
                    else (
                        "completed"
                        if normalized_status == "failed"
                        and execution_result is not None
                        else (
                            "failed"
                            if normalized_status == "failed"
                            else current.execution_outcome
                        )
                    )
                ),
                evaluation_status=(
                    "pending"
                    if normalized_status == "awaiting_evaluation"
                    else current.evaluation_status
                ),
                task_outcome=(
                    "pending"
                    if normalized_status == "awaiting_evaluation"
                    else current.task_outcome
                ),
                current_station_revision=revision,
                updated_at=timestamp,
                lower_operation_id=resolved_lower_id,
                execution_receipt_refs=tuple(
                    sorted({*current.execution_receipt_refs, *execution_receipt_refs})
                ),
                progress_summary=(
                    current.progress_summary
                    if progress_summary is None
                    else progress_summary
                ),
                execution_result=execution_result,
            )
            self._replace_operation_locked(next_operation)
            return OperationStoreOutcome(
                True,
                "operation_projection_updated",
                next_operation,
            )

    def transition_evaluation_projection(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        evaluation_status: str,
        task_outcome: str,
        reason_code: str,
        current_station_revision: int,
        updated_at: float,
        evaluator_receipt_refs: tuple[str, ...],
        aggregate_evaluation_receipt_ref: str,
        task_evaluation: Mapping[str, Any],
    ) -> OperationStoreOutcome:
        """CAS already-produced evaluator evidence onto an awaiting operation."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        expected_hash = _required_text(
            expected_operation_hash,
            field_name="expected_operation_hash",
        )
        normalized_evaluation_status = _required_text(
            evaluation_status,
            field_name="evaluation_status",
        )
        if normalized_evaluation_status not in {"completed", "timed_out"}:
            raise ValueError(
                "evaluation_status must be 'completed' or 'timed_out'."
            )
        normalized_task_outcome = _required_text(
            task_outcome,
            field_name="task_outcome",
        )
        normalized_reason = _required_text(reason_code, field_name="reason_code")
        aggregate_ref = _required_text(
            aggregate_evaluation_receipt_ref,
            field_name="aggregate_evaluation_receipt_ref",
        )
        revision = _revision(current_station_revision)
        timestamp = _finite_time(updated_at, field_name="updated_at")

        with self._lock:
            current = self._operations.get(normalized_id)
            if current is None:
                return OperationStoreOutcome(False, "operation_not_found")
            if current.semantic_content_hash != expected_hash:
                return OperationStoreOutcome(
                    False,
                    "operation_hash_mismatch",
                    current,
                )
            if current.terminal:
                return OperationStoreOutcome(False, "operation_terminal", current)
            if current.status != "awaiting_evaluation":
                return OperationStoreOutcome(
                    False,
                    "operation_not_awaiting_evaluation",
                    current,
                )
            if normalized_id in self._control_reservations:
                return OperationStoreOutcome(
                    False,
                    "operation_control_reserved",
                    current,
                )
            if revision < current.current_station_revision:
                return OperationStoreOutcome(
                    False,
                    "station_revision_regressed",
                    current,
                )
            if timestamp < current.updated_at:
                return OperationStoreOutcome(
                    False,
                    "operation_timestamp_regressed",
                    current,
                )
            if (
                timestamp >= current.deadline_at
                and normalized_evaluation_status == "completed"
            ):
                return OperationStoreOutcome(
                    False,
                    "operation_deadline_expired",
                    current,
                )
            next_operation = replace(
                current,
                status="completed",
                reason_code=normalized_reason,
                evaluation_status=normalized_evaluation_status,
                task_outcome=normalized_task_outcome,
                current_station_revision=revision,
                updated_at=timestamp,
                evaluator_receipt_refs=tuple(
                    sorted(
                        {
                            *current.evaluator_receipt_refs,
                            *evaluator_receipt_refs,
                        }
                    )
                ),
                aggregate_evaluation_receipt_ref=aggregate_ref,
                progress_summary={"phase": "evaluation_closed"},
                task_evaluation=task_evaluation,
            )
            self._replace_operation_locked(next_operation)
            return OperationStoreOutcome(
                True,
                "operation_evaluation_projection_updated",
                next_operation,
            )

    def reserve_control(
        self,
        *,
        operation_id: str,
        control_kind: str,
        terminal_status: str,
        reason_code: str,
        current_station_revision: int,
        updated_at: float,
    ) -> OperationStoreOutcome:
        """Invalidate outstanding worker hashes before a cancel/HOLD call."""

        normalized_id = _required_text(operation_id, field_name="operation_id")
        normalized_kind = _required_text(control_kind, field_name="control_kind")
        if normalized_kind not in AGENT_GATEWAY_CONTROL_KINDS_V1:
            raise ValueError(f"Unsupported control_kind: {normalized_kind!r}.")
        if terminal_status not in {"cancelled", "timed_out"}:
            raise ValueError("terminal_status must be 'cancelled' or 'timed_out'.")
        normalized_reason = _required_text(reason_code, field_name="reason_code")
        revision = _revision(current_station_revision)
        timestamp = _finite_time(updated_at, field_name="updated_at")

        with self._lock:
            current = self._operations.get(normalized_id)
            if current is None:
                return OperationStoreOutcome(False, "operation_not_found")
            if current.terminal:
                if current.status == terminal_status:
                    return OperationStoreOutcome(
                        True,
                        "operation_already_terminal",
                        current,
                    )
                return OperationStoreOutcome(False, "operation_terminal", current)
            existing = self._control_reservations.get(normalized_id)
            if existing is not None:
                if (
                    existing.control_kind != normalized_kind
                    or existing.terminal_status != terminal_status
                ):
                    return OperationStoreOutcome(
                        False,
                        "operation_control_reservation_conflict",
                        current,
                    )
                return OperationStoreOutcome(
                    True,
                    "operation_control_already_reserved",
                    current,
                )
            if revision < current.current_station_revision:
                return OperationStoreOutcome(
                    False,
                    "station_revision_regressed",
                    current,
                )
            if timestamp < current.updated_at:
                return OperationStoreOutcome(
                    False,
                    "operation_timestamp_regressed",
                    current,
                )
            progress = dict(current.progress_summary)
            progress["control_request"] = {
                "control_kind": normalized_kind,
                "terminal_status": terminal_status,
            }
            reserved = replace(
                current,
                reason_code=normalized_reason,
                current_station_revision=revision,
                updated_at=timestamp,
                progress_summary=progress,
            )
            self._control_reservations[normalized_id] = _ControlReservation(
                control_kind=normalized_kind,
                terminal_status=terminal_status,
            )
            self._operations[normalized_id] = reserved
            self._latest_station_revision = revision
            return OperationStoreOutcome(
                True,
                "operation_control_reserved",
                reserved,
            )

    def record_control_failure(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        reason_code: str,
        current_station_revision: int,
        updated_at: float,
        execution_receipt_refs: tuple[str, ...] = (),
    ) -> OperationStoreOutcome:
        """Keep an active operation fail-closed after lower control failure."""

        return self._update_reserved_control(
            operation_id=operation_id,
            expected_operation_hash=expected_operation_hash,
            reason_code=reason_code,
            current_station_revision=current_station_revision,
            updated_at=updated_at,
            execution_receipt_refs=execution_receipt_refs,
            finalize=False,
        )

    def finalize_control(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        reason_code: str,
        current_station_revision: int,
        updated_at: float,
        execution_receipt_refs: tuple[str, ...] = (),
    ) -> OperationStoreOutcome:
        """CAS the reserved terminal state after fake/null control confirms."""

        return self._update_reserved_control(
            operation_id=operation_id,
            expected_operation_hash=expected_operation_hash,
            reason_code=reason_code,
            current_station_revision=current_station_revision,
            updated_at=updated_at,
            execution_receipt_refs=execution_receipt_refs,
            finalize=True,
        )

    def due_active_operation(self, *, now: float) -> AgentSkillOperation | None:
        timestamp = _finite_time(now, field_name="now")
        with self._lock:
            active = self.active_operation()
            if active is None or timestamp < active.deadline_at:
                return None
            return active

    def _update_reserved_control(
        self,
        *,
        operation_id: str,
        expected_operation_hash: str,
        reason_code: str,
        current_station_revision: int,
        updated_at: float,
        execution_receipt_refs: tuple[str, ...],
        finalize: bool,
    ) -> OperationStoreOutcome:
        normalized_id = _required_text(operation_id, field_name="operation_id")
        expected_hash = _required_text(
            expected_operation_hash,
            field_name="expected_operation_hash",
        )
        normalized_reason = _required_text(reason_code, field_name="reason_code")
        revision = _revision(current_station_revision)
        timestamp = _finite_time(updated_at, field_name="updated_at")
        with self._lock:
            current = self._operations.get(normalized_id)
            if current is None:
                return OperationStoreOutcome(False, "operation_not_found")
            if current.semantic_content_hash != expected_hash:
                return OperationStoreOutcome(
                    False,
                    "operation_hash_mismatch",
                    current,
                )
            reservation = self._control_reservations.get(normalized_id)
            if reservation is None:
                return OperationStoreOutcome(
                    False,
                    "operation_control_not_reserved",
                    current,
                )
            if revision < current.current_station_revision:
                return OperationStoreOutcome(
                    False,
                    "station_revision_regressed",
                    current,
                )
            if timestamp < current.updated_at:
                return OperationStoreOutcome(
                    False,
                    "operation_timestamp_regressed",
                    current,
                )
            next_operation = replace(
                current,
                status=(reservation.terminal_status if finalize else current.status),
                reason_code=normalized_reason,
                execution_outcome=(
                    (
                        current.execution_outcome
                        if current.execution_outcome == "completed"
                        else reservation.terminal_status
                    )
                    if finalize
                    else current.execution_outcome
                ),
                current_station_revision=revision,
                updated_at=timestamp,
                execution_receipt_refs=tuple(
                    sorted({*current.execution_receipt_refs, *execution_receipt_refs})
                ),
                progress_summary=(
                    {"phase": reservation.terminal_status}
                    if finalize
                    else current.progress_summary
                ),
            )
            self._replace_operation_locked(next_operation)
            return OperationStoreOutcome(
                True,
                (
                    "operation_control_finalized"
                    if finalize
                    else "operation_control_failure_recorded"
                ),
                next_operation,
            )

    def _replace_operation_locked(self, operation: AgentSkillOperation) -> None:
        self._operations[operation.operation_id] = operation
        self._latest_station_revision = operation.current_station_revision
        if operation.terminal:
            if self._active_operation_id == operation.operation_id:
                self._active_operation_id = None
            self._control_reservations.pop(operation.operation_id, None)

    def _reserve_retention_slot_locked(self) -> bool:
        while len(self._operations) >= self.max_retained_operations:
            terminal_id = next(
                (
                    operation_id
                    for operation_id, operation in self._operations.items()
                    if operation.terminal
                ),
                None,
            )
            if terminal_id is None:
                return False
            self._evict_locked(terminal_id)
        return True

    def _evict_locked(self, operation_id: str) -> None:
        request = self._requests.pop(operation_id)
        key = (request.client_id, request.idempotency_key)
        self._operations.pop(operation_id)
        self._operation_id_by_idempotency.pop(key, None)
        self._request_hash_by_idempotency.pop(key, None)
        self._control_reservations.pop(operation_id, None)
        self._total_evicted += 1

    @staticmethod
    def _operation_id(request: AgentSkillRequest) -> str:
        digest = canonical_content_hash(
            {
                "client_id": request.client_id,
                "idempotency_key": request.idempotency_key,
                "request_hash": request.semantic_content_hash,
                "runtime_session_id": request.runtime_session_id,
            },
            domain=AGENT_GATEWAY_OPERATION_ID_HASH_DOMAIN_V1,
        )
        return "agent-skill-operation:" + digest


__all__ = [
    "AGENT_GATEWAY_CONTROL_KINDS_V1",
    "AGENT_GATEWAY_OPERATION_ID_HASH_DOMAIN_V1",
    "BoundedInMemoryAgentSkillOperationStore",
    "OperationStoreOutcome",
]

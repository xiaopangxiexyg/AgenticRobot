"""Bounded safety-control operations with per-call admitted tick evidence.

The first operation is RM65 quick-stop.  Operation admission freezes what may
be sent and for how long, but does not itself authorize a backend call.  Every
actual planner/SDK tick still requires a distinct one-shot dispatch permit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Callable

from dataset_contracts.canonical_encoding import canonical_content_hash

from .dispatch_owner import AdmittedSafetyControlEvidence, SAFETY_CONTROL_SOURCES
from .cancellation import ExecutionWorkItem


SAFETY_CONTROL_OPERATION_HASH_DOMAIN_V1 = (
    "dexteleop.safety_control_operation.v1"
)
SAFETY_CONTROL_OPERATION_TICK_HASH_DOMAIN_V1 = (
    "dexteleop.safety_control_operation_tick.v1"
)
SAFETY_CONTROL_OPERATION_TERMINAL_HASH_DOMAIN_V1 = (
    "dexteleop.safety_control_operation_terminal.v1"
)

RM65_QUICK_STOP_OPERATION_V1 = "rm65_quick_stop"
RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1 = "rm65_quick_stop_tick"
SAFETY_CONTROL_OPERATION_KINDS_V1 = frozenset({RM65_QUICK_STOP_OPERATION_V1})
SAFETY_CONTROL_OPERATION_TERMINAL_STATUSES_V1 = frozenset(
    {"completed", "canceled", "expired", "epoch_invalidated", "faulted"}
)


def _identifier(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class SafetyControlOperation:
    operation_id: str
    operation_kind: str
    runtime_session_id: str
    control_domain_id: str
    safety_source: str
    source_event_id: str
    target_execution_epoch: int
    affected_selection_groups: tuple[str, ...]
    frozen_stop_target_hash: str
    allowed_dispatch_operation: str
    maximum_duration_sec: float
    maximum_tick_count: int
    authorized_at: float
    semantic_content_hash: str

    @classmethod
    def create_rm65_quick_stop(
        cls,
        *,
        operation_id: str,
        runtime_session_id: str,
        control_domain_id: str,
        safety_source: str,
        source_event_id: str,
        target_execution_epoch: int,
        affected_selection_groups: tuple[str, ...],
        frozen_stop_target_hash: str,
        maximum_duration_sec: float,
        maximum_tick_count: int,
        authorized_at: float,
    ) -> "SafetyControlOperation":
        source = _identifier(safety_source, name="safety_source")
        if source not in SAFETY_CONTROL_SOURCES:
            raise ValueError("Unsupported safety-control source.")
        groups = tuple(
            _identifier(value, name="affected_selection_groups")
            for value in affected_selection_groups
        )
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("affected_selection_groups must be non-empty and unique.")
        epoch = int(target_execution_epoch)
        if isinstance(target_execution_epoch, bool) or epoch < 0:
            raise ValueError("target_execution_epoch must be non-negative.")
        duration = _finite(maximum_duration_sec, name="maximum_duration_sec")
        ticks = int(maximum_tick_count)
        if duration <= 0.0 or ticks <= 0:
            raise ValueError("Quick-stop duration/tick count must be positive.")
        payload = {
            "operation_id": _identifier(operation_id, name="operation_id"),
            "operation_kind": RM65_QUICK_STOP_OPERATION_V1,
            "runtime_session_id": _identifier(
                runtime_session_id,
                name="runtime_session_id",
            ),
            "control_domain_id": _identifier(
                control_domain_id,
                name="control_domain_id",
            ),
            "safety_source": source,
            "source_event_id": _identifier(source_event_id, name="source_event_id"),
            "target_execution_epoch": epoch,
            "affected_selection_groups": groups,
            "frozen_stop_target_hash": _identifier(
                frozen_stop_target_hash,
                name="frozen_stop_target_hash",
            ),
            "allowed_dispatch_operation": (
                RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1
            ),
            "maximum_duration_sec": duration,
            "maximum_tick_count": ticks,
            "authorized_at": _finite(authorized_at, name="authorized_at"),
        }
        return cls(
            **payload,
            semantic_content_hash=canonical_content_hash(
                payload,
                domain=SAFETY_CONTROL_OPERATION_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class SafetyControlOperationTickEvidence:
    tick_evidence_id: str
    operation_id: str
    operation_hash: str
    tick_sequence: int
    target_execution_epoch: int
    command_lineage_id: str
    admitted_at: float
    control_admission_sequence: int
    semantic_content_hash: str

    def as_dispatch_evidence(
        self,
        operation: SafetyControlOperation,
    ) -> AdmittedSafetyControlEvidence:
        if self.operation_id != operation.operation_id:
            raise ValueError("Tick evidence belongs to another safety operation.")
        return AdmittedSafetyControlEvidence(
            event_id=self.tick_evidence_id,
            runtime_session_id=operation.runtime_session_id,
            control_domain_id=operation.control_domain_id,
            execution_epoch=self.target_execution_epoch,
            evidence_kind="safety_operation_tick",
            safety_source=operation.safety_source,
            control_admission_sequence=self.control_admission_sequence,
        )

    def as_execution_work_item(
        self,
        operation: SafetyControlOperation,
        *,
        authority_epoch: int,
        segment_generation_by_group: dict[str, int],
    ) -> ExecutionWorkItem:
        """Build new-epoch safety work; never reuse canceled motion lineage."""

        if self.operation_id != operation.operation_id:
            raise ValueError("Tick evidence belongs to another safety operation.")
        if self.operation_hash != operation.semantic_content_hash:
            raise ValueError("Tick evidence operation hash does not match.")
        if self.target_execution_epoch != operation.target_execution_epoch:
            raise ValueError("Tick evidence execution epoch does not match operation.")
        return ExecutionWorkItem(
            work_item_id=f"safety-backend-command:{self.tick_evidence_id}",
            work_kind="safety_backend_command",
            control_domain_id=operation.control_domain_id,
            authority_epoch=authority_epoch,
            execution_epoch=operation.target_execution_epoch,
            segment_generation_by_group=segment_generation_by_group,
            source_safety_operation_id=operation.operation_id,
            source_safety_tick_evidence_id=self.tick_evidence_id,
        )


@dataclass(frozen=True)
class SafetyControlOperationTerminal:
    terminal_id: str
    operation_id: str
    operation_hash: str
    status: str
    reason: str
    terminal_at: float
    last_admitted_tick_sequence: int
    semantic_content_hash: str


class SafetyControlOperationRegistry:
    """In-process admission authority; it never calls a backend."""

    def __init__(
        self,
        *,
        runtime_session_id: str,
        control_domain_id: str,
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
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self._clock = clock
        self._operation: SafetyControlOperation | None = None
        self._terminal: SafetyControlOperationTerminal | None = None
        self._last_tick_sequence = 0
        self._admission_sequence = 0
        self._lock = threading.RLock()

    def admit(self, operation: SafetyControlOperation) -> None:
        if not isinstance(operation, SafetyControlOperation):
            raise TypeError("operation must be a SafetyControlOperation.")
        if (
            operation.runtime_session_id != self.runtime_session_id
            or operation.control_domain_id != self.control_domain_id
        ):
            raise ValueError("Safety operation belongs to another session/domain.")
        with self._lock:
            if self._operation is not None and self._terminal is None:
                raise RuntimeError("A safety-control operation is already active.")
            self._operation = operation
            self._terminal = None
            self._last_tick_sequence = 0

    def active_operation(self) -> SafetyControlOperation:
        """Return the immutable active operation without granting authority."""

        with self._lock:
            if self._operation is None:
                raise RuntimeError("No safety-control operation is active.")
            if self._terminal is not None:
                raise RuntimeError("Safety-control operation is terminal.")
            return self._operation

    def admit_tick(
        self,
        *,
        operation_id: str,
        tick_sequence: int,
        current_execution_epoch: int,
        command_lineage_id: str,
    ) -> SafetyControlOperationTickEvidence:
        with self._lock:
            operation = self._require_active_locked(operation_id)
            now = _finite(self._clock(), name="admitted_at")
            if int(current_execution_epoch) != operation.target_execution_epoch:
                raise RuntimeError("Safety operation execution epoch is no longer active.")
            if now > operation.authorized_at + operation.maximum_duration_sec:
                raise RuntimeError("Safety operation authorization has expired.")
            sequence = int(tick_sequence)
            if sequence != self._last_tick_sequence + 1:
                raise ValueError("Safety operation tick sequence must be contiguous.")
            if sequence > operation.maximum_tick_count:
                raise RuntimeError("Safety operation tick budget is exhausted.")
            self._admission_sequence += 1
            payload = {
                "operation_id": operation.operation_id,
                "operation_hash": operation.semantic_content_hash,
                "tick_sequence": sequence,
                "target_execution_epoch": operation.target_execution_epoch,
                "command_lineage_id": _identifier(
                    command_lineage_id,
                    name="command_lineage_id",
                ),
                "admitted_at": now,
                "control_admission_sequence": self._admission_sequence,
            }
            semantic_hash = canonical_content_hash(
                payload,
                domain=SAFETY_CONTROL_OPERATION_TICK_HASH_DOMAIN_V1,
            )
            self._last_tick_sequence = sequence
            return SafetyControlOperationTickEvidence(
                tick_evidence_id=f"safety-operation-tick:{semantic_hash}",
                **payload,
                semantic_content_hash=semantic_hash,
            )

    def terminate(
        self,
        *,
        operation_id: str,
        status: str,
        reason: str,
    ) -> SafetyControlOperationTerminal:
        normalized_status = str(status).strip()
        if normalized_status not in SAFETY_CONTROL_OPERATION_TERMINAL_STATUSES_V1:
            raise ValueError("Unsupported safety operation terminal status.")
        with self._lock:
            operation = self._require_active_locked(operation_id)
            payload = {
                "operation_id": operation.operation_id,
                "operation_hash": operation.semantic_content_hash,
                "status": normalized_status,
                "reason": _identifier(reason, name="reason"),
                "terminal_at": _finite(self._clock(), name="terminal_at"),
                "last_admitted_tick_sequence": self._last_tick_sequence,
            }
            semantic_hash = canonical_content_hash(
                payload,
                domain=SAFETY_CONTROL_OPERATION_TERMINAL_HASH_DOMAIN_V1,
            )
            terminal = SafetyControlOperationTerminal(
                terminal_id=f"safety-operation-terminal:{semantic_hash}",
                **payload,
                semantic_content_hash=semantic_hash,
            )
            self._terminal = terminal
            return terminal

    def _require_active_locked(self, operation_id: str) -> SafetyControlOperation:
        operation = self._operation
        if operation is None or operation.operation_id != str(operation_id):
            raise ValueError("Unknown safety-control operation.")
        if self._terminal is not None:
            raise RuntimeError("Safety-control operation is terminal.")
        return operation


__all__ = [
    "RM65_QUICK_STOP_OPERATION_V1",
    "RM65_QUICK_STOP_TICK_DISPATCH_OPERATION_V1",
    "SAFETY_CONTROL_OPERATION_HASH_DOMAIN_V1",
    "SAFETY_CONTROL_OPERATION_KINDS_V1",
    "SAFETY_CONTROL_OPERATION_TERMINAL_HASH_DOMAIN_V1",
    "SAFETY_CONTROL_OPERATION_TERMINAL_STATUSES_V1",
    "SAFETY_CONTROL_OPERATION_TICK_HASH_DOMAIN_V1",
    "SafetyControlOperation",
    "SafetyControlOperationRegistry",
    "SafetyControlOperationTerminal",
    "SafetyControlOperationTickEvidence",
]

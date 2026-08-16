"""Reusable bounded executor for read-only Agent Skills.

This executor never owns HOLD, motion, dispatch, a simulator loop, or a device.
One injected reader produces an already-measured result and immutable evidence
references.  The executor only drives the Gateway operation CAS around that
bounded read and can be reused for sequential operations.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..protocol.contracts import AgentSkillOperation, AgentSkillRequest
from .operation_port import AgentGatewayAuthoritativePortReceipt

PASSIVE_READ_EXECUTION_RECEIPT_HASH_DOMAIN_V1 = (
    "dexteleop.agent_gateway.passive_read_execution_receipt.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Station revision must be a non-negative integer.")
    return value


def _evidence_refs(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(
        sorted({_required(value, field_name="evidence_ref") for value in values})
    )
    if not normalized:
        raise ValueError("A passive read requires immutable evidence refs.")
    return normalized


@dataclass(frozen=True)
class PassiveReadResult:
    """One already-produced read result plus lower/application evidence refs."""

    execution_result: Mapping[str, Any]
    evidence_refs: tuple[str, ...]
    reason_code: str = "passive_read_completed"

    def __post_init__(self) -> None:
        if not isinstance(self.execution_result, Mapping):
            raise TypeError("execution_result must be a mapping.")
        object.__setattr__(self, "execution_result", dict(self.execution_result))
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs))
        object.__setattr__(
            self,
            "reason_code",
            _required(self.reason_code, field_name="reason_code"),
        )


@dataclass(frozen=True)
class PassiveReadExecutionReceipt:
    """Content-addressed proof of a completed passive result projection."""

    executor_id: str
    operation_id: str
    request_hash: str
    execution_result_hash: str
    completed_at: float
    evidence_refs: tuple[str, ...]
    succeeded: bool
    reason_code: str
    receipt_id: str = field(init=False)

    def __post_init__(self) -> None:
        for field_name in (
            "executor_id",
            "operation_id",
            "request_hash",
            "execution_result_hash",
            "reason_code",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        object.__setattr__(
            self,
            "completed_at",
            _finite(self.completed_at, field_name="completed_at"),
        )
        object.__setattr__(self, "evidence_refs", _evidence_refs(self.evidence_refs))
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        object.__setattr__(
            self,
            "receipt_id",
            "passive-read-execution-receipt:" + self.semantic_content_hash,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "executor_id": self.executor_id,
                "operation_id": self.operation_id,
                "request_hash": self.request_hash,
                "execution_result_hash": self.execution_result_hash,
                "completed_at": self.completed_at,
                "evidence_refs": self.evidence_refs,
                "succeeded": self.succeeded,
                "reason_code": self.reason_code,
                "proves_motion_or_hold": False,
            },
            domain=PASSIVE_READ_EXECUTION_RECEIPT_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PassiveReadLaunchReceipt:
    executor_id: str
    operation_id: str
    accepted: bool
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "executor_id",
            _required(self.executor_id, field_name="executor_id"),
        )
        object.__setattr__(
            self,
            "operation_id",
            _required(self.operation_id, field_name="operation_id"),
        )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean.")
        object.__setattr__(
            self,
            "status",
            _required(self.status, field_name="status"),
        )


class _PassiveReadGatewayService(Protocol):
    operation_store: Any

    def execution_port_for_operation(self, operation_id: str) -> object: ...

    def activate_operation(self, **kwargs: Any) -> Any: ...

    def complete_operation(self, **kwargs: Any) -> Any: ...

    def expire_due_operation(self, **kwargs: Any) -> Any: ...


class PassiveReadSkillExecutor:
    """Reusable thread-per-operation executor for sequential passive reads."""

    no_io = False

    def __init__(
        self,
        *,
        port_id: str,
        executor_id: str,
        read: Callable[[AgentSkillRequest, AgentSkillOperation], PassiveReadResult],
        current_station_revision: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
        max_retained_operations: int = 32,
    ) -> None:
        self.port_id = _required(port_id, field_name="port_id")
        self.executor_id = _required(executor_id, field_name="executor_id")
        if not callable(read):
            raise TypeError("read must be callable.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if (
            isinstance(max_retained_operations, bool)
            or not isinstance(max_retained_operations, int)
            or max_retained_operations <= 0
        ):
            raise ValueError("max_retained_operations must be positive.")
        self._read = read
        self._current_station_revision = current_station_revision
        self._clock = clock
        self._max_retained_operations = max_retained_operations
        self._lock = threading.RLock()
        self._receipt_lock = threading.Lock()
        self._threads: OrderedDict[str, threading.Thread] = OrderedDict()
        self._cancel_events: dict[str, threading.Event] = {}
        self._completion_by_operation: OrderedDict[str, Any] = OrderedDict()
        self._execution_receipts: OrderedDict[
            str, PassiveReadExecutionReceipt
        ] = OrderedDict()
        self._port_receipts: list[AgentGatewayAuthoritativePortReceipt] = []
        self._next_call_sequence = 0

    def _time(self) -> float:
        return _finite(self._clock(), field_name="clock result")

    def _revision(self) -> int:
        return _revision(self._current_station_revision())

    @property
    def port_receipts(self) -> tuple[AgentGatewayAuthoritativePortReceipt, ...]:
        with self._receipt_lock:
            return tuple(self._port_receipts)

    @property
    def execution_receipts(self) -> tuple[PassiveReadExecutionReceipt, ...]:
        with self._lock:
            return tuple(self._execution_receipts.values())

    def completion_for(self, operation_id: str) -> Any | None:
        normalized = _required(operation_id, field_name="operation_id")
        with self._lock:
            return self._completion_by_operation.get(normalized)

    def launch(
        self,
        *,
        service: _PassiveReadGatewayService,
        operation_id: str,
    ) -> PassiveReadLaunchReceipt:
        normalized = _required(operation_id, field_name="operation_id")
        if service.execution_port_for_operation(normalized) is not self:
            raise ValueError("Gateway operation is bound to another executor.")
        operation = service.operation_store.get(normalized)
        request = service.operation_store.request_for_operation(normalized)
        if operation is None or request is None:
            return PassiveReadLaunchReceipt(
                executor_id=self.executor_id,
                operation_id=normalized,
                accepted=False,
                status="operation_not_found",
            )
        if operation.status != "admitted":
            return PassiveReadLaunchReceipt(
                executor_id=self.executor_id,
                operation_id=normalized,
                accepted=False,
                status="operation_not_admitted",
            )
        with self._lock:
            existing = self._threads.get(normalized)
            if existing is not None:
                return PassiveReadLaunchReceipt(
                    executor_id=self.executor_id,
                    operation_id=normalized,
                    accepted=True,
                    status="operation_already_launched",
                )
            self._prune_finished_locked()
            cancel_event = threading.Event()
            thread = threading.Thread(
                target=self._run,
                args=(service, request, operation, cancel_event),
                name=f"agent-passive-read:{self.executor_id}",
                daemon=True,
            )
            self._cancel_events[normalized] = cancel_event
            self._threads[normalized] = thread
            thread.start()
        return PassiveReadLaunchReceipt(
            executor_id=self.executor_id,
            operation_id=normalized,
            accepted=True,
            status="passive_read_executor_launched",
        )

    def wait(self, operation_id: str, *, timeout_sec: float | None = None) -> bool:
        normalized = _required(operation_id, field_name="operation_id")
        if timeout_sec is not None:
            timeout_sec = _finite(timeout_sec, field_name="timeout_sec")
            if timeout_sec < 0.0:
                raise ValueError("timeout_sec cannot be negative.")
        with self._lock:
            thread = self._threads.get(normalized)
        if thread is None:
            return False
        thread.join(timeout=timeout_sec)
        return not thread.is_alive()

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> AgentGatewayAuthoritativePortReceipt:
        self._validate_request_operation(request=request, operation=operation)
        return self._record_port_receipt(
            operation_kind="activate",
            operation=operation,
            succeeded=True,
            status="passive_read_activation_confirmed",
            lower_operation_id=f"passive-read:{operation.operation_id}",
            evidence_refs=(
                f"passive-read-executor:{self.executor_id}:activated",
            ),
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        self._validate_request_operation(request=request, operation=operation)
        _required(reason, field_name="reason")
        with self._lock:
            event = self._cancel_events.get(operation.operation_id)
            if event is not None:
                event.set()
        return self._record_port_receipt(
            operation_kind="cancel",
            operation=operation,
            succeeded=True,
            status="passive_read_cancel_confirmed",
            lower_operation_id=operation.lower_operation_id,
            evidence_refs=(
                f"passive-read-executor:{self.executor_id}:cancelled",
            ),
        )

    @staticmethod
    def _validate_request_operation(
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> None:
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        if not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation.")
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("Operation request hash mismatch.")

    def _record_port_receipt(
        self,
        *,
        operation_kind: str,
        operation: AgentSkillOperation,
        succeeded: bool,
        status: str,
        lower_operation_id: str | None,
        evidence_refs: tuple[str, ...],
    ) -> AgentGatewayAuthoritativePortReceipt:
        with self._receipt_lock:
            receipt = AgentGatewayAuthoritativePortReceipt(
                port_id=self.port_id,
                operation_kind=operation_kind,
                operation_id=operation.operation_id,
                succeeded=succeeded,
                status=status,
                completed_at=self._time(),
                call_sequence=self._next_call_sequence,
                lower_operation_id=lower_operation_id,
                evidence_refs=evidence_refs,
                proves_motion_or_hold=False,
            )
            self._next_call_sequence += 1
            self._port_receipts.append(receipt)
            return receipt

    def _run(
        self,
        service: _PassiveReadGatewayService,
        request: AgentSkillRequest,
        admitted: AgentSkillOperation,
        cancel_event: threading.Event,
    ) -> None:
        try:
            activation = service.activate_operation(
                operation_id=admitted.operation_id,
                current_station_revision=self._revision(),
            )
            running = getattr(activation, "operation", None)
            if (
                not bool(getattr(activation, "accepted", False))
                or not isinstance(running, AgentSkillOperation)
                or running.status != "running"
            ):
                return
            if self._expire_or_cancelled(
                service=service,
                operation=running,
                cancel_event=cancel_event,
            ):
                return
            result = self._read(request, running)
            if not isinstance(result, PassiveReadResult):
                raise TypeError("Passive reader returned an invalid result.")
            if self._expire_or_cancelled(
                service=service,
                operation=running,
                cancel_event=cancel_event,
            ):
                return
            result_hash = canonical_content_hash(
                result.execution_result,
                domain="dexteleop.agent_gateway.passive_read_result.v1",
            )
            execution_receipt = PassiveReadExecutionReceipt(
                executor_id=self.executor_id,
                operation_id=running.operation_id,
                request_hash=request.semantic_content_hash,
                execution_result_hash=result_hash,
                completed_at=self._time(),
                evidence_refs=result.evidence_refs,
                succeeded=True,
                reason_code=result.reason_code,
            )
            outcome = service.complete_operation(
                operation_id=running.operation_id,
                expected_operation_hash=running.semantic_content_hash,
                succeeded=True,
                reason_code=result.reason_code,
                current_station_revision=self._revision(),
                execution_result=result.execution_result,
                execution_receipt_refs=(execution_receipt.receipt_id,),
            )
            if (
                not bool(getattr(outcome, "accepted", False))
                and getattr(outcome, "reason_code", None)
                == "execution_result_payload_rejected"
            ):
                outcome = service.complete_operation(
                    operation_id=running.operation_id,
                    expected_operation_hash=running.semantic_content_hash,
                    succeeded=False,
                    reason_code="passive_read_result_rejected",
                    current_station_revision=self._revision(),
                    execution_receipt_refs=(execution_receipt.receipt_id,),
                )
            with self._lock:
                self._execution_receipts[running.operation_id] = execution_receipt
                self._completion_by_operation[running.operation_id] = outcome
                self._trim_records_locked()
        except Exception as error:  # noqa: BLE001 - convert reader failure to CAS.
            current = service.operation_store.get(admitted.operation_id)
            if isinstance(current, AgentSkillOperation) and not current.terminal:
                outcome = service.complete_operation(
                    operation_id=current.operation_id,
                    expected_operation_hash=current.semantic_content_hash,
                    succeeded=False,
                    reason_code=(
                        "passive_read_executor_exception:"
                        + type(error).__name__
                    ),
                    current_station_revision=self._revision(),
                )
                with self._lock:
                    self._completion_by_operation[current.operation_id] = outcome
                    self._trim_records_locked()

    def _expire_or_cancelled(
        self,
        *,
        service: _PassiveReadGatewayService,
        operation: AgentSkillOperation,
        cancel_event: threading.Event,
    ) -> bool:
        now = self._time()
        if now >= operation.deadline_at:
            service.expire_due_operation(
                current_station_revision=self._revision(),
                now=now,
            )
            return True
        return cancel_event.is_set()

    def _prune_finished_locked(self) -> None:
        finished = [
            operation_id
            for operation_id, thread in self._threads.items()
            if not thread.is_alive()
        ]
        while (
            len(self._threads) >= self._max_retained_operations and finished
        ):
            operation_id = finished.pop(0)
            self._threads.pop(operation_id, None)
            self._cancel_events.pop(operation_id, None)

    def _trim_records_locked(self) -> None:
        for records in (
            self._completion_by_operation,
            self._execution_receipts,
        ):
            while len(records) > self._max_retained_operations:
                records.popitem(last=False)


__all__ = [
    "PASSIVE_READ_EXECUTION_RECEIPT_HASH_DOMAIN_V1",
    "PassiveReadExecutionReceipt",
    "PassiveReadLaunchReceipt",
    "PassiveReadResult",
    "PassiveReadSkillExecutor",
]

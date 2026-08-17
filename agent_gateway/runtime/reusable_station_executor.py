"""Reusable exact-Route executor built from one-use Station workers.

The Gateway still admits at most one active operation.  This adapter creates a
fresh lifecycle and bounded worker for each sequential operation while
remaining the single Route execution port and the single control-domain HOLD
port.  It owns no action queue, motion loop, safety, IK, or dispatch.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from ..protocol.contracts import AgentSkillOperation, AgentSkillRequest
from .operation_port import AgentGatewayAuthoritativePortReceipt
from .station_operation_worker import (
    BoundedStationAgentSkillOperationWorker,
    StationLifecycleControlEvidence,
    StationOperationLaunchReceipt,
    StationOperationWorkerBinding,
    StationSkillOperationLifecycle,
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


class StationOperationLifecycleFactory(Protocol):
    def __call__(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> StationSkillOperationLifecycle: ...


class _WorkerServiceProxy:
    """Make one child worker appear as the selected port to its launch check."""

    def __init__(self, service: Any, worker: Any) -> None:
        self._service = service
        self._worker = worker
        self.operation_port = worker
        self.operation_store = service.operation_store

    def execution_port_for_operation(self, operation_id: str) -> Any:
        request = self.operation_store.request_for_operation(operation_id)
        if request is None:
            raise KeyError("Operation has no immutable request.")
        return self._worker

    def activate_operation(self, **kwargs: Any) -> Any:
        return self._service.activate_operation(**kwargs)

    def complete_operation(self, **kwargs: Any) -> Any:
        return self._service.complete_operation(**kwargs)

    def complete_evaluation(self, **kwargs: Any) -> Any:
        return self._service.complete_evaluation(**kwargs)

    def expire_due_operation(self, **kwargs: Any) -> Any:
        return self._service.expire_due_operation(**kwargs)


class ReusableStationAgentSkillOperationExecutor:
    """Fresh one-use worker per sequential operation; no implicit queue.

    ``domain_hold`` is used before the first operation or after the active
    worker is unavailable.  It must call the same lower-owned HOLD boundary as
    every operation lifecycle and return exact lower evidence.
    """

    no_io = False
    control_effect_class = "motion"

    def __init__(
        self,
        *,
        port_id: str,
        binding: StationOperationWorkerBinding,
        lifecycle_factory: StationOperationLifecycleFactory,
        domain_hold: Callable[..., StationLifecycleControlEvidence],
        current_station_revision: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port_id = _required(port_id, field_name="port_id")
        if not isinstance(binding, StationOperationWorkerBinding):
            raise TypeError("binding must be a StationOperationWorkerBinding.")
        for name, value in (
            ("lifecycle_factory", lifecycle_factory),
            ("domain_hold", domain_hold),
            ("current_station_revision", current_station_revision),
            ("clock", clock),
        ):
            if not callable(value):
                raise TypeError(f"{name} must be callable.")
        self.binding = binding
        self._lifecycle_factory = lifecycle_factory
        self._domain_hold = domain_hold
        self._current_station_revision = current_station_revision
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._active_worker: BoundedStationAgentSkillOperationWorker | None = None
        self._active_operation_id: str | None = None
        self._retired_workers: list[BoundedStationAgentSkillOperationWorker] = []
        self._closed = False
        self._next_idle_hold_sequence = 0
        self._idle_hold_receipts: list[AgentGatewayAuthoritativePortReceipt] = []

    @property
    def active_operation_id(self) -> str | None:
        with self._condition:
            return self._active_operation_id

    @property
    def workers(self) -> tuple[BoundedStationAgentSkillOperationWorker, ...]:
        with self._condition:
            values = list(self._retired_workers)
            if self._active_worker is not None:
                values.append(self._active_worker)
            return tuple(values)

    @property
    def receipts(self) -> tuple[AgentGatewayAuthoritativePortReceipt, ...]:
        with self._condition:
            values = list(self._idle_hold_receipts)
            for worker in self._retired_workers:
                values.extend(worker.receipts)
            if self._active_worker is not None:
                values.extend(self._active_worker.receipts)
            return tuple(values)

    def _current_worker_for(
        self,
        operation: AgentSkillOperation,
    ) -> BoundedStationAgentSkillOperationWorker:
        if not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation.")
        with self._condition:
            worker = self._active_worker
            if worker is None or self._active_operation_id != operation.operation_id:
                raise ValueError("Reusable executor has no worker for this operation.")
            return worker

    def launch(self, *, service: Any, operation_id: str) -> StationOperationLaunchReceipt:
        normalized_id = _required(operation_id, field_name="operation_id")
        operation = service.operation_store.get(normalized_id)
        request = service.operation_store.request_for_operation(normalized_id)
        if operation is None or request is None:
            return StationOperationLaunchReceipt(
                worker_id=self.binding.worker_id,
                operation_id=normalized_id,
                accepted=False,
                status="operation_not_found",
            )
        self.binding.validate_request(request)
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("Operation request_hash does not match its request.")
        with self._condition:
            if self._closed:
                return StationOperationLaunchReceipt(
                    worker_id=self.binding.worker_id,
                    operation_id=normalized_id,
                    accepted=False,
                    status="reusable_executor_closed",
                )
            active = self._active_worker
            if active is not None and active.state not in {"finished", "closed"}:
                return StationOperationLaunchReceipt(
                    worker_id=self.binding.worker_id,
                    operation_id=normalized_id,
                    accepted=self._active_operation_id == normalized_id,
                    status=(
                        "operation_already_launched"
                        if self._active_operation_id == normalized_id
                        else "reusable_executor_busy"
                    ),
                )
            if active is not None:
                self._retired_workers.append(active)
            lifecycle = self._lifecycle_factory(
                request=request,
                operation=operation,
            )
            if not isinstance(lifecycle, StationSkillOperationLifecycle):
                raise TypeError(
                    "lifecycle_factory must return a StationSkillOperationLifecycle."
                )
            child = BoundedStationAgentSkillOperationWorker(
                port_id=self.port_id,
                binding=replace(
                    self.binding,
                    worker_id=f"{self.binding.worker_id}:{normalized_id}",
                ),
                lifecycle=lifecycle,
                current_station_revision=self._current_station_revision,
                clock=self._clock,
            )
            self._active_worker = child
            self._active_operation_id = normalized_id
            proxy = _WorkerServiceProxy(service, child)
            outcome = child.launch(service=proxy, operation_id=normalized_id)
            self._condition.notify_all()
            return outcome

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        with self._condition:
            worker = self._active_worker
        return False if worker is None else worker.wait(timeout_sec=timeout_sec)

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> AgentGatewayAuthoritativePortReceipt:
        return self._current_worker_for(operation).activate(
            request=request,
            operation=operation,
        )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        return self._current_worker_for(operation).cancel(
            request=request,
            operation=operation,
            reason=reason,
        )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        normalized_reason = _required(reason, field_name="reason")
        with self._condition:
            worker = self._active_worker
            active_id = self._active_operation_id
        if worker is not None and (
            operation is None or operation.operation_id == active_id
        ):
            return worker.request_hold(
                operation=operation,
                reason=normalized_reason,
            )
        evidence = self._domain_hold(
            operation=operation,
            reason=normalized_reason,
        )
        if not isinstance(evidence, StationLifecycleControlEvidence):
            raise TypeError("domain_hold returned invalid control evidence.")
        receipt = AgentGatewayAuthoritativePortReceipt(
            port_id=self.port_id,
            operation_kind="hold",
            operation_id=None if operation is None else operation.operation_id,
            succeeded=evidence.succeeded,
            status=evidence.status,
            completed_at=float(self._clock()),
            call_sequence=self._next_idle_hold_sequence,
            lower_operation_id=evidence.lower_operation_id,
            evidence_refs=evidence.evidence_refs,
            proves_motion_or_hold=evidence.lower_hold_confirmed,
        )
        with self._condition:
            self._next_idle_hold_sequence += 1
            self._idle_hold_receipts.append(receipt)
        return receipt

    def close_after_hold(
        self,
        *,
        hold_receipt: AgentGatewayAuthoritativePortReceipt,
    ) -> None:
        if not isinstance(hold_receipt, AgentGatewayAuthoritativePortReceipt):
            raise TypeError("hold_receipt must be authoritative.")
        if (
            hold_receipt.port_id != self.port_id
            or hold_receipt.operation_kind != "hold"
            or not hold_receipt.succeeded
            or not hold_receipt.proves_motion_or_hold
        ):
            raise ValueError("Executor close requires its successful lower HOLD.")
        with self._condition:
            self._closed = True
            self._condition.notify_all()


__all__ = [
    "ReusableStationAgentSkillOperationExecutor",
    "StationOperationLifecycleFactory",
]

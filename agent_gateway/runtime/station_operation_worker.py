"""Single-use authoritative Station lifecycle worker for Agent Skills.

The worker owns one application thread, never a motion/action queue.  Its
lifecycle adapter calls the existing Station application leaf, while all
cancel/HOLD requests remain out-of-band and race against result commits through
the Gateway store's CAS boundary.  This module imports no lower runtime.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..protocol.contracts import (
    AGENT_SKILL_EVALUATION_STATUSES_V1,
    AGENT_SKILL_TASK_OUTCOMES_V1,
    AgentSkillOperation,
    AgentSkillRequest,
)
from .operation_port import AgentGatewayAuthoritativePortReceipt

STATION_OPERATION_WORKER_STATES_V1 = (
    "idle",
    "launched",
    "activating",
    "running",
    "stopping",
    "finished",
    "closed",
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
        raise ValueError(
            "current_station_revision provider must return a non-negative integer."
        )
    return value


def _references(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(
        sorted({_required(value, field_name=f"{field_name} item") for value in values})
    )
    if len(normalized) > 128:
        raise ValueError(f"{field_name} cannot contain more than 128 references.")
    return normalized


def _freeze_json(value: Any, *, field_name: str, depth: int = 0) -> Any:
    if depth > 16:
        raise ValueError(f"{field_name} exceeds the maximum JSON depth.")
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return _finite(value, field_name=field_name)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} object keys must be strings.")
        return MappingProxyType(
            {
                key: _freeze_json(
                    value[key],
                    field_name=f"{field_name}.{key}",
                    depth=depth + 1,
                )
                for key in sorted(value)
            }
        )
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(
                item,
                field_name=f"{field_name} item",
                depth=depth + 1,
            )
            for item in value
        )
    raise TypeError(f"{field_name} must contain JSON-compatible values.")


def _freeze_json_object(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    frozen = _freeze_json(value, field_name=field_name)
    if not isinstance(frozen, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    return frozen


@dataclass(frozen=True)
class StationOperationWorkerBinding:
    """Exact one-session Route binding consumed by one worker instance."""

    worker_id: str
    executor_id: str
    runtime_session_id: str
    skill_id: str
    skill_version: str
    route_id: str
    route_version: str
    deployment_binding_hash: str

    def __post_init__(self) -> None:
        for field_name in (
            "worker_id",
            "executor_id",
            "runtime_session_id",
            "skill_id",
            "skill_version",
            "route_id",
            "route_version",
            "deployment_binding_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )

    def validate_request(self, request: AgentSkillRequest) -> None:
        if not isinstance(request, AgentSkillRequest):
            raise TypeError("request must be an AgentSkillRequest.")
        checks = (
            (request.runtime_session_id == self.runtime_session_id, "runtime session"),
            (request.skill_id == self.skill_id, "skill ID"),
            (request.skill_version == self.skill_version, "skill version"),
            (request.route_id == self.route_id, "Route ID"),
            (request.route_version == self.route_version, "Route version"),
            (
                request.deployment_binding_hash == self.deployment_binding_hash,
                "deployment binding",
            ),
        )
        for matches, label in checks:
            if not matches:
                raise ValueError(f"Station worker {label} mismatch.")


@dataclass(frozen=True)
class StationLifecycleActivationEvidence:
    succeeded: bool
    status: str
    lower_operation_id: str | None
    evidence_refs: tuple[str, ...]
    lower_hold_confirmed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        object.__setattr__(self, "status", _required(self.status, field_name="status"))
        if self.lower_operation_id is not None:
            object.__setattr__(
                self,
                "lower_operation_id",
                _required(
                    self.lower_operation_id,
                    field_name="lower_operation_id",
                ),
            )
        refs = _references(self.evidence_refs, field_name="evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        if not isinstance(self.lower_hold_confirmed, bool):
            raise TypeError("lower_hold_confirmed must be a boolean.")
        if self.succeeded and (self.lower_operation_id is None or not refs):
            raise ValueError(
                "Successful activation requires a lower operation ID and evidence."
            )
        if self.lower_hold_confirmed and not refs:
            raise ValueError("Confirmed lower HOLD requires evidence.")


@dataclass(frozen=True)
class StationLifecycleControlEvidence:
    succeeded: bool
    status: str
    lower_operation_id: str | None
    evidence_refs: tuple[str, ...]
    lower_hold_confirmed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        object.__setattr__(self, "status", _required(self.status, field_name="status"))
        if self.lower_operation_id is not None:
            object.__setattr__(
                self,
                "lower_operation_id",
                _required(
                    self.lower_operation_id,
                    field_name="lower_operation_id",
                ),
            )
        refs = _references(self.evidence_refs, field_name="evidence_refs")
        object.__setattr__(self, "evidence_refs", refs)
        if not isinstance(self.lower_hold_confirmed, bool):
            raise TypeError("lower_hold_confirmed must be a boolean.")
        if self.succeeded and not refs:
            raise ValueError("Successful control requires lower evidence.")
        if self.lower_hold_confirmed and not refs:
            raise ValueError("Confirmed lower HOLD requires evidence.")


@dataclass(frozen=True)
class StationLifecycleEvaluationCompletion:
    evaluation_status: str
    task_outcome: str
    reason_code: str
    evaluator_receipt_refs: tuple[str, ...]
    aggregate_evaluation_receipt_ref: str
    task_evaluation: Mapping[str, Any]

    def __post_init__(self) -> None:
        evaluation_status = _required(
            self.evaluation_status,
            field_name="evaluation_status",
        )
        if evaluation_status not in AGENT_SKILL_EVALUATION_STATUSES_V1:
            raise ValueError(f"Unsupported evaluation_status: {evaluation_status!r}.")
        if evaluation_status in {"not_requested", "pending"}:
            raise ValueError("A completion cannot retain a pending evaluation status.")
        object.__setattr__(self, "evaluation_status", evaluation_status)
        task_outcome = _required(self.task_outcome, field_name="task_outcome")
        if task_outcome not in AGENT_SKILL_TASK_OUTCOMES_V1:
            raise ValueError(f"Unsupported task_outcome: {task_outcome!r}.")
        if task_outcome in {"not_assessed", "pending"}:
            raise ValueError("A completion requires a terminal task outcome.")
        object.__setattr__(self, "task_outcome", task_outcome)
        object.__setattr__(
            self,
            "reason_code",
            _required(self.reason_code, field_name="reason_code"),
        )
        refs = _references(
            self.evaluator_receipt_refs,
            field_name="evaluator_receipt_refs",
        )
        if not refs:
            raise ValueError("Evaluation completion requires evaluator receipts.")
        object.__setattr__(self, "evaluator_receipt_refs", refs)
        object.__setattr__(
            self,
            "aggregate_evaluation_receipt_ref",
            _required(
                self.aggregate_evaluation_receipt_ref,
                field_name="aggregate_evaluation_receipt_ref",
            ),
        )
        object.__setattr__(
            self,
            "task_evaluation",
            _freeze_json_object(
                self.task_evaluation,
                field_name="task_evaluation",
            ),
        )


@dataclass(frozen=True)
class StationLifecycleExecutionCompletion:
    succeeded: bool
    reason_code: str
    execution_receipt_refs: tuple[str, ...]
    terminal_hold_confirmed: bool
    execution_result: Mapping[str, Any] | None = None
    evaluation: StationLifecycleEvaluationCompletion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")
        object.__setattr__(
            self,
            "reason_code",
            _required(self.reason_code, field_name="reason_code"),
        )
        refs = _references(
            self.execution_receipt_refs,
            field_name="execution_receipt_refs",
        )
        if not refs:
            raise ValueError("Execution completion requires lower evidence.")
        object.__setattr__(self, "execution_receipt_refs", refs)
        if not isinstance(self.terminal_hold_confirmed, bool):
            raise TypeError("terminal_hold_confirmed must be a boolean.")
        if self.succeeded and not self.terminal_hold_confirmed:
            raise ValueError(
                "Successful authoritative execution requires terminal lower HOLD."
            )
        if self.succeeded and self.execution_result is None:
            raise ValueError("Successful execution requires an execution_result.")
        if not self.succeeded and self.execution_result is not None:
            raise ValueError("Failed execution cannot claim an execution_result.")
        if self.execution_result is not None:
            object.__setattr__(
                self,
                "execution_result",
                _freeze_json_object(
                    self.execution_result,
                    field_name="execution_result",
                ),
            )
        if self.evaluation is not None and not isinstance(
            self.evaluation,
            StationLifecycleEvaluationCompletion,
        ):
            raise TypeError(
                "evaluation must be a StationLifecycleEvaluationCompletion or None."
            )
        if not self.succeeded and self.evaluation is not None:
            raise ValueError("Failed execution cannot claim task evaluation.")


@runtime_checkable
class StationSkillOperationLifecycle(Protocol):
    """One-use adapter around an already-built Station application graph."""

    executor_id: str

    def activate(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleActivationEvidence: ...

    def run(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        cancel_requested: Callable[[], bool],
    ) -> StationLifecycleExecutionCompletion: ...

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> StationLifecycleControlEvidence: ...

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class StationOperationLaunchReceipt:
    worker_id: str
    operation_id: str
    accepted: bool
    status: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "worker_id",
            _required(self.worker_id, field_name="worker_id"),
        )
        object.__setattr__(
            self,
            "operation_id",
            _required(self.operation_id, field_name="operation_id"),
        )
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be a boolean.")
        object.__setattr__(self, "status", _required(self.status, field_name="status"))


class _GatewayWorkerService(Protocol):
    operation_port: object
    operation_store: Any

    def execution_port_for_operation(self, operation_id: str) -> object: ...

    def activate_operation(self, **kwargs: Any) -> Any: ...

    def complete_operation(self, **kwargs: Any) -> Any: ...

    def complete_evaluation(self, **kwargs: Any) -> Any: ...

    def expire_due_operation(self, **kwargs: Any) -> Any: ...


class BoundedStationAgentSkillOperationWorker:
    """One thread and one immutable lifecycle; no queue and no reuse."""

    no_io = False
    control_effect_class = "motion"

    def __init__(
        self,
        *,
        port_id: str,
        binding: StationOperationWorkerBinding,
        lifecycle: StationSkillOperationLifecycle,
        current_station_revision: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port_id = _required(port_id, field_name="port_id")
        if not isinstance(binding, StationOperationWorkerBinding):
            raise TypeError("binding must be a StationOperationWorkerBinding.")
        if _required(
            getattr(lifecycle, "executor_id", None),
            field_name="lifecycle.executor_id",
        ) != binding.executor_id:
            raise ValueError("Lifecycle executor_id does not match worker binding.")
        for method_name in (
            "activate",
            "run",
            "cancel",
            "request_hold",
            "close",
        ):
            if not callable(getattr(lifecycle, method_name, None)):
                raise TypeError(
                    f"Lifecycle is missing required method {method_name}()."
                )
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")

        self.binding = binding
        self.lifecycle = lifecycle
        self._current_station_revision = current_station_revision
        self._clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._receipt_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._state = "idle"
        self._operation_id: str | None = None
        self._lower_operation_id: str | None = None
        self._thread: threading.Thread | None = None
        self._service: _GatewayWorkerService | None = None
        self._consumed = False
        self._next_call_sequence = 0
        self._receipts: list[AgentGatewayAuthoritativePortReceipt] = []
        self._last_activation_outcome: Any | None = None
        self._last_completion_outcome: Any | None = None
        self._last_evaluation_outcome: Any | None = None
        self._last_error: str | None = None

    @property
    def state(self) -> str:
        with self._condition:
            return self._state

    @property
    def operation_id(self) -> str | None:
        with self._condition:
            return self._operation_id

    @property
    def receipts(self) -> tuple[AgentGatewayAuthoritativePortReceipt, ...]:
        with self._receipt_lock:
            return tuple(self._receipts)

    @property
    def last_error(self) -> str | None:
        with self._condition:
            return self._last_error

    @property
    def last_completion_outcome(self) -> Any | None:
        with self._condition:
            return self._last_completion_outcome

    @property
    def last_evaluation_outcome(self) -> Any | None:
        with self._condition:
            return self._last_evaluation_outcome

    def _time(self) -> float:
        return _finite(self._clock(), field_name="clock result")

    def _revision(self) -> int:
        return _revision(self._current_station_revision())

    def _stop_or_deadline_requested(
        self,
        operation: AgentSkillOperation,
    ) -> bool:
        return self._stop_requested.is_set() or self._time() >= operation.deadline_at

    def _expire_if_due(
        self,
        *,
        service: _GatewayWorkerService,
        operation: AgentSkillOperation,
    ) -> Any | None:
        """Let the Gateway reserve timeout control before any worker result CAS."""

        now = self._time()
        if now < operation.deadline_at:
            return None
        outcome = service.expire_due_operation(
            current_station_revision=self._revision(),
            now=now,
        )
        with self._condition:
            self._last_completion_outcome = outcome
        return outcome

    def _set_state(self, state: str) -> None:
        if state not in STATION_OPERATION_WORKER_STATES_V1:
            raise ValueError(f"Unsupported worker state: {state!r}.")
        with self._condition:
            if self._state != "closed":
                self._state = state
            self._condition.notify_all()

    def _claim(self, operation_id: str) -> None:
        normalized = _required(operation_id, field_name="operation_id")
        with self._condition:
            if self._operation_id is None:
                self._operation_id = normalized
            elif self._operation_id != normalized:
                raise ValueError("Single-use worker is bound to another operation.")

    def _validate_operation_binding(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
    ) -> None:
        self.binding.validate_request(request)
        if not isinstance(operation, AgentSkillOperation):
            raise TypeError("operation must be an AgentSkillOperation.")
        if operation.request_hash != request.semantic_content_hash:
            raise ValueError("Operation request_hash does not match the request.")
        if operation.runtime_session_id != self.binding.runtime_session_id:
            raise ValueError("Operation runtime session does not match the worker.")
        self._claim(operation.operation_id)

    def launch(
        self,
        *,
        service: _GatewayWorkerService,
        operation_id: str,
    ) -> StationOperationLaunchReceipt:
        """Reserve the only worker slot and start outside the request thread."""

        normalized_id = _required(operation_id, field_name="operation_id")
        selector = getattr(service, "execution_port_for_operation", None)
        selected_port = (
            selector(normalized_id)
            if callable(selector)
            else getattr(service, "operation_port", None)
        )
        if selected_port is not self:
            raise ValueError("Gateway service is not bound to this worker port.")
        store = getattr(service, "operation_store", None)
        operation = None if store is None else store.get(normalized_id)
        request = None if store is None else store.request_for_operation(normalized_id)
        if operation is None or request is None:
            return StationOperationLaunchReceipt(
                worker_id=self.binding.worker_id,
                operation_id=normalized_id,
                accepted=False,
                status="operation_not_found",
            )
        self._validate_operation_binding(request=request, operation=operation)
        if operation.status != "admitted":
            return StationOperationLaunchReceipt(
                worker_id=self.binding.worker_id,
                operation_id=normalized_id,
                accepted=False,
                status="operation_not_admitted",
            )
        with self._condition:
            if self._state == "closed":
                return StationOperationLaunchReceipt(
                    worker_id=self.binding.worker_id,
                    operation_id=normalized_id,
                    accepted=False,
                    status="worker_closed",
                )
            if self._thread is not None:
                return StationOperationLaunchReceipt(
                    worker_id=self.binding.worker_id,
                    operation_id=normalized_id,
                    accepted=self._operation_id == normalized_id,
                    status=(
                        "operation_already_launched"
                        if self._operation_id == normalized_id
                        else "worker_single_use_consumed"
                    ),
                )
            if self._consumed:
                return StationOperationLaunchReceipt(
                    worker_id=self.binding.worker_id,
                    operation_id=normalized_id,
                    accepted=False,
                    status="worker_single_use_consumed",
                )
            self._consumed = True
            self._service = service
            self._state = "launched"
            self._thread = threading.Thread(
                target=self._run_operation,
                args=(service, request, operation),
                name=f"agent-skill:{self.binding.worker_id}",
                daemon=True,
            )
            self._thread.start()
        return StationOperationLaunchReceipt(
            worker_id=self.binding.worker_id,
            operation_id=normalized_id,
            accepted=True,
            status="operation_worker_launched",
        )

    def wait(self, *, timeout_sec: float | None = None) -> bool:
        if timeout_sec is not None:
            timeout_sec = _finite(timeout_sec, field_name="timeout_sec")
            if timeout_sec < 0.0:
                raise ValueError("timeout_sec cannot be negative.")
        with self._condition:
            thread = self._thread
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
        self._validate_operation_binding(request=request, operation=operation)
        self._set_state("activating")
        try:
            evidence = self.lifecycle.activate(
                request=request,
                operation=operation,
                cancel_requested=lambda: self._stop_or_deadline_requested(operation),
            )
            if not isinstance(evidence, StationLifecycleActivationEvidence):
                raise TypeError("Lifecycle returned invalid activation evidence.")
            service = self._service
            if service is not None and self._time() >= operation.deadline_at:
                timeout = self._expire_if_due(
                    service=service,
                    operation=operation,
                )
                timeout_receipt = getattr(timeout, "port_receipt", None)
                timeout_refs = (
                    ()
                    if timeout_receipt is None
                    else (timeout_receipt.receipt_id,)
                )
                return self._record_receipt(
                    operation_kind="activate",
                    operation_id=operation.operation_id,
                    succeeded=False,
                    status="station_activation_deadline_exceeded",
                    lower_operation_id=(
                        getattr(timeout_receipt, "lower_operation_id", None)
                        or evidence.lower_operation_id
                    ),
                    evidence_refs=tuple(
                        sorted({*evidence.evidence_refs, *timeout_refs})
                    ),
                    proves_motion_or_hold=bool(
                        getattr(timeout_receipt, "proves_motion_or_hold", False)
                    ),
                )
            if not evidence.succeeded and not evidence.lower_hold_confirmed:
                hold = self._call_lifecycle_hold(
                    operation=operation,
                    reason="activation_rejected",
                )
                return self._record_receipt(
                    operation_kind="activate",
                    operation_id=operation.operation_id,
                    succeeded=False,
                    status="station_activation_rejected_after_hold_attempt",
                    lower_operation_id=(
                        hold.lower_operation_id or evidence.lower_operation_id
                    ),
                    evidence_refs=tuple(
                        sorted({*evidence.evidence_refs, *hold.evidence_refs})
                    ),
                    proves_motion_or_hold=hold.lower_hold_confirmed,
                )
            if evidence.succeeded and self._stop_requested.is_set():
                hold = self._call_lifecycle_hold(
                    operation=operation,
                    reason="activation_preempted",
                )
                return self._record_receipt(
                    operation_kind="activate",
                    operation_id=operation.operation_id,
                    succeeded=False,
                    status="station_activation_preempted_after_lower_hold",
                    lower_operation_id=(
                        hold.lower_operation_id or evidence.lower_operation_id
                    ),
                    evidence_refs=tuple(
                        sorted({*evidence.evidence_refs, *hold.evidence_refs})
                    ),
                    proves_motion_or_hold=hold.lower_hold_confirmed,
                )
            return self._receipt_from_activation(operation, evidence)
        except Exception as error:  # noqa: BLE001 - establish lower HOLD first.
            hold = self._call_lifecycle_hold(
                operation=operation,
                reason="activation_exception",
            )
            return self._record_receipt(
                operation_kind="activate",
                operation_id=operation.operation_id,
                succeeded=False,
                status=f"station_activation_exception:{type(error).__name__}",
                lower_operation_id=hold.lower_operation_id,
                evidence_refs=hold.evidence_refs,
                proves_motion_or_hold=hold.lower_hold_confirmed,
            )

    def cancel(
        self,
        *,
        request: AgentSkillRequest,
        operation: AgentSkillOperation,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        self._validate_operation_binding(request=request, operation=operation)
        normalized_reason = _required(reason, field_name="reason")
        self._stop_requested.set()
        self._set_state("stopping")
        try:
            evidence = self.lifecycle.cancel(
                request=request,
                operation=operation,
                reason=normalized_reason,
            )
            if not isinstance(evidence, StationLifecycleControlEvidence):
                raise TypeError("Lifecycle returned invalid cancel evidence.")
            return self._receipt_from_control(
                operation_kind="cancel",
                operation=operation,
                evidence=evidence,
            )
        except Exception as error:  # noqa: BLE001 - fail closed in receipt.
            return self._record_receipt(
                operation_kind="cancel",
                operation_id=operation.operation_id,
                succeeded=False,
                status=f"station_cancel_exception:{type(error).__name__}",
            )

    def request_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> AgentGatewayAuthoritativePortReceipt:
        normalized_reason = _required(reason, field_name="reason")
        if operation is not None:
            if not isinstance(operation, AgentSkillOperation):
                raise TypeError("operation must be an AgentSkillOperation or None.")
            self._claim(operation.operation_id)
        self._stop_requested.set()
        self._set_state("stopping")
        evidence = self._call_lifecycle_hold(
            operation=operation,
            reason=normalized_reason,
        )
        return self._receipt_from_control(
            operation_kind="hold",
            operation=operation,
            evidence=evidence,
        )

    def _call_lifecycle_hold(
        self,
        *,
        operation: AgentSkillOperation | None,
        reason: str,
    ) -> StationLifecycleControlEvidence:
        try:
            evidence = self.lifecycle.request_hold(
                operation=operation,
                reason=_required(reason, field_name="reason"),
            )
            if not isinstance(evidence, StationLifecycleControlEvidence):
                raise TypeError("Lifecycle returned invalid HOLD evidence.")
            if evidence.succeeded and not evidence.lower_hold_confirmed:
                raise ValueError("Successful lifecycle HOLD lacks lower proof.")
            return evidence
        except Exception as error:  # noqa: BLE001 - encode failure, never claim HOLD.
            return StationLifecycleControlEvidence(
                succeeded=False,
                status=f"station_hold_exception:{type(error).__name__}",
                lower_operation_id=self._lower_operation_id,
                evidence_refs=(),
                lower_hold_confirmed=False,
            )

    def _receipt_from_activation(
        self,
        operation: AgentSkillOperation,
        evidence: StationLifecycleActivationEvidence,
    ) -> AgentGatewayAuthoritativePortReceipt:
        return self._record_receipt(
            operation_kind="activate",
            operation_id=operation.operation_id,
            succeeded=evidence.succeeded,
            status=evidence.status,
            lower_operation_id=evidence.lower_operation_id,
            evidence_refs=evidence.evidence_refs,
            proves_motion_or_hold=evidence.lower_hold_confirmed,
        )

    def _receipt_from_control(
        self,
        *,
        operation_kind: str,
        operation: AgentSkillOperation | None,
        evidence: StationLifecycleControlEvidence,
    ) -> AgentGatewayAuthoritativePortReceipt:
        return self._record_receipt(
            operation_kind=operation_kind,
            operation_id=None if operation is None else operation.operation_id,
            succeeded=evidence.succeeded,
            status=evidence.status,
            lower_operation_id=evidence.lower_operation_id,
            evidence_refs=evidence.evidence_refs,
            proves_motion_or_hold=evidence.lower_hold_confirmed,
        )

    def _record_receipt(
        self,
        *,
        operation_kind: str,
        operation_id: str | None,
        succeeded: bool,
        status: str,
        lower_operation_id: str | None = None,
        evidence_refs: tuple[str, ...] = (),
        proves_motion_or_hold: bool = False,
    ) -> AgentGatewayAuthoritativePortReceipt:
        with self._receipt_lock:
            if lower_operation_id is not None:
                normalized_lower_id = _required(
                    lower_operation_id,
                    field_name="lower_operation_id",
                )
                if self._lower_operation_id not in {None, normalized_lower_id}:
                    raise ValueError("Lifecycle lower operation identity changed.")
                self._lower_operation_id = normalized_lower_id
            receipt = AgentGatewayAuthoritativePortReceipt(
                port_id=self.port_id,
                operation_kind=operation_kind,
                operation_id=operation_id,
                succeeded=succeeded,
                status=status,
                completed_at=self._time(),
                call_sequence=self._next_call_sequence,
                lower_operation_id=lower_operation_id,
                evidence_refs=evidence_refs,
                proves_motion_or_hold=proves_motion_or_hold,
            )
            self._next_call_sequence += 1
            self._receipts.append(receipt)
            return receipt

    def _run_operation(
        self,
        service: _GatewayWorkerService,
        request: AgentSkillRequest,
        admitted: AgentSkillOperation,
    ) -> None:
        try:
            activation = service.activate_operation(
                operation_id=admitted.operation_id,
                current_station_revision=self._revision(),
            )
            with self._condition:
                self._last_activation_outcome = activation
            running = getattr(activation, "operation", None)
            if (
                not bool(getattr(activation, "accepted", False))
                or not isinstance(running, AgentSkillOperation)
                or running.status != "running"
            ):
                return
            self._set_state("running")
            completion = self.lifecycle.run(
                request=request,
                operation=running,
                cancel_requested=lambda: self._stop_or_deadline_requested(running),
            )
            if not isinstance(completion, StationLifecycleExecutionCompletion):
                raise TypeError("Lifecycle returned invalid execution completion.")
            if self._expire_if_due(service=service, operation=running) is not None:
                return
            receipt_refs = completion.execution_receipt_refs
            if not completion.terminal_hold_confirmed:
                hold_receipt = self.request_hold(
                    operation=running,
                    reason="execution_completed_without_terminal_hold",
                )
                receipt_refs = tuple(
                    sorted({*receipt_refs, hold_receipt.receipt_id})
                )
                if not hold_receipt.succeeded:
                    completion = StationLifecycleExecutionCompletion(
                        succeeded=False,
                        reason_code="terminal_hold_not_confirmed",
                        execution_receipt_refs=receipt_refs,
                        terminal_hold_confirmed=False,
                    )
            outcome = service.complete_operation(
                operation_id=running.operation_id,
                expected_operation_hash=running.semantic_content_hash,
                succeeded=completion.succeeded,
                reason_code=completion.reason_code,
                current_station_revision=self._revision(),
                execution_result=completion.execution_result,
                execution_receipt_refs=receipt_refs,
            )
            if (
                not bool(getattr(outcome, "accepted", False))
                and getattr(outcome, "reason_code", None)
                == "execution_result_payload_rejected"
            ):
                hold_receipt = self.request_hold(
                    operation=running,
                    reason="execution_result_payload_rejected",
                )
                fallback_refs = tuple(
                    sorted({*receipt_refs, hold_receipt.receipt_id})
                )
                outcome = service.complete_operation(
                    operation_id=running.operation_id,
                    expected_operation_hash=running.semantic_content_hash,
                    succeeded=False,
                    reason_code="station_worker_execution_result_rejected",
                    current_station_revision=self._revision(),
                    execution_receipt_refs=fallback_refs,
                )
            with self._condition:
                self._last_completion_outcome = outcome
            projected = getattr(outcome, "operation", None)
            if (
                bool(getattr(outcome, "accepted", False))
                and isinstance(projected, AgentSkillOperation)
                and projected.status == "awaiting_evaluation"
                and completion.evaluation is not None
            ):
                evaluation = completion.evaluation
                evaluation_outcome = service.complete_evaluation(
                    operation_id=projected.operation_id,
                    expected_operation_hash=projected.semantic_content_hash,
                    evaluation_status=evaluation.evaluation_status,
                    task_outcome=evaluation.task_outcome,
                    reason_code=evaluation.reason_code,
                    current_station_revision=self._revision(),
                    evaluator_receipt_refs=evaluation.evaluator_receipt_refs,
                    aggregate_evaluation_receipt_ref=(
                        evaluation.aggregate_evaluation_receipt_ref
                    ),
                    task_evaluation=evaluation.task_evaluation,
                )
                with self._condition:
                    self._last_evaluation_outcome = evaluation_outcome
        except Exception as error:  # noqa: BLE001 - HOLD then fail the live CAS.
            self._record_worker_error(error)
            current = service.operation_store.get(admitted.operation_id)
            deadline_operation = (
                current if isinstance(current, AgentSkillOperation) else admitted
            )
            if self._expire_if_due(
                service=service,
                operation=deadline_operation,
            ) is not None:
                return
            hold_receipt = self.request_hold(
                operation=(
                    current if isinstance(current, AgentSkillOperation) else admitted
                ),
                reason="station_operation_worker_exception",
            )
            if isinstance(current, AgentSkillOperation) and not current.terminal:
                outcome = service.complete_operation(
                    operation_id=current.operation_id,
                    expected_operation_hash=current.semantic_content_hash,
                    succeeded=False,
                    reason_code=(
                        "station_operation_worker_exception_after_hold"
                        if hold_receipt.succeeded
                        else "station_operation_worker_exception_hold_unconfirmed"
                    ),
                    current_station_revision=self._revision(),
                    execution_receipt_refs=(hold_receipt.receipt_id,),
                )
                with self._condition:
                    self._last_completion_outcome = outcome
        finally:
            try:
                self.lifecycle.close()
            except Exception as error:  # noqa: BLE001 - retain first failure.
                self._record_worker_error(error)
            self._set_state("finished")

    def _record_worker_error(self, error: BaseException) -> None:
        detail = f"{type(error).__name__}:{error}"
        with self._condition:
            if self._last_error is None:
                self._last_error = detail

    def close_after_hold(
        self,
        *,
        hold_receipt: AgentGatewayAuthoritativePortReceipt,
    ) -> None:
        """Fence reuse only after the Gateway/lower owner proved HOLD."""

        if not isinstance(hold_receipt, AgentGatewayAuthoritativePortReceipt):
            raise TypeError("hold_receipt must be authoritative.")
        if (
            hold_receipt.operation_kind != "hold"
            or not hold_receipt.succeeded
            or not hold_receipt.proves_motion_or_hold
        ):
            raise ValueError("Worker close requires a successful lower HOLD receipt.")
        if hold_receipt.port_id != self.port_id:
            raise ValueError("Worker close HOLD receipt port mismatch.")
        with self._condition:
            if (
                hold_receipt.operation_id is not None
                and self._operation_id not in {None, hold_receipt.operation_id}
            ):
                raise ValueError("Worker close HOLD receipt operation mismatch.")
            self._stop_requested.set()
            self._state = "closed"
            self._condition.notify_all()


__all__ = [
    "STATION_OPERATION_WORKER_STATES_V1",
    "BoundedStationAgentSkillOperationWorker",
    "StationLifecycleActivationEvidence",
    "StationLifecycleControlEvidence",
    "StationLifecycleEvaluationCompletion",
    "StationLifecycleExecutionCompletion",
    "StationOperationLaunchReceipt",
    "StationOperationWorkerBinding",
    "StationSkillOperationLifecycle",
]

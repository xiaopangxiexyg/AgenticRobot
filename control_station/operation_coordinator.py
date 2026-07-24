"""Bounded CS-1c coordinator for explicit no-I/O station adapters.

This coordinator owns neither a control loop nor a backend.  It reserves
bounded application capacity before an intent enters the station runtime,
executes only deterministic null/fake adapters on its own worker, and returns
their typed completion to the runtime admission seam.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .adapters import DeterministicNoIoStationAdapter, StationOperationAdapter
from .contracts import StationIntentDraft, StationOperationResultDraft
from .events import BoundedStationEventLog
from .runtime import ControlStationRuntime, StationOperationExecution
from .state_machine import (
    ControlStationOperationTransition,
    ControlStationTransition,
)


CS1C_COORDINATED_OPERATION_KINDS = frozenset(
    {
        "connect_source",
        "disconnect_source",
        "connect_target",
        "disconnect_target",
        "run_preflight",
        "start_shadow",
        "stop_shadow",
    }
)


class StationOperationCoordinatorUnavailable(RuntimeError):
    """The bounded coordinator cannot reserve work before intent admission."""

    def __init__(self, status: str) -> None:
        self.status = str(status).strip()
        super().__init__(self.status)


@dataclass(frozen=True)
class StationOperationEnqueueReceipt:
    coordinator_id: str
    source_intent_id: str | None
    accepted: bool
    status: str
    queue_capacity: int
    reserved_work_count: int


@dataclass(frozen=True)
class CoordinatedIntentSubmission:
    transition: ControlStationTransition
    enqueue_receipt: StationOperationEnqueueReceipt


@dataclass(frozen=True)
class StationOperationCompletion:
    coordinator_id: str
    adapter_id: str
    request_transition: ControlStationTransition
    completion_transition: ControlStationOperationTransition


class NoIoStationOperationCoordinator:
    """Execute null/fake station operations outside request threads.

    Capacity covers queued plus in-flight work.  A coordinated operation must
    reserve capacity before receiver admission so overload cannot strand a
    pending station operation.
    """

    def __init__(
        self,
        *,
        coordinator_id: str,
        runtime: ControlStationRuntime,
        adapters: tuple[DeterministicNoIoStationAdapter, ...],
        event_log: BoundedStationEventLog | None = None,
        queue_capacity: int = 16,
        clock: Callable[[], float] = time.monotonic,
        clock_domain_id: str = "station-coordinator-monotonic",
    ) -> None:
        normalized_id = str(coordinator_id).strip()
        normalized_clock_domain = str(clock_domain_id).strip()
        if not normalized_id or not normalized_clock_domain:
            raise ValueError("Coordinator identity/clock must be non-empty.")
        if not isinstance(runtime, ControlStationRuntime):
            raise TypeError("runtime must be a ControlStationRuntime.")
        if event_log is not None and not isinstance(event_log, BoundedStationEventLog):
            raise TypeError("event_log must be a BoundedStationEventLog or None.")
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or not 1 <= queue_capacity <= 1024
        ):
            raise ValueError("queue_capacity must be an integer in [1, 1024].")
        if not callable(clock):
            raise TypeError("clock must be callable.")

        normalized_adapters = tuple(adapters)
        if not normalized_adapters or any(
            not isinstance(adapter, DeterministicNoIoStationAdapter)
            for adapter in normalized_adapters
        ):
            raise TypeError(
                "CS-1c only permits DeterministicNoIoStationAdapter instances."
            )
        adapter_ids = tuple(adapter.adapter_id for adapter in normalized_adapters)
        subjects = tuple(adapter.subject_id for adapter in normalized_adapters)
        if len(set(adapter_ids)) != len(adapter_ids):
            raise ValueError("Coordinator adapter IDs must be unique.")
        if len(set(subjects)) != len(subjects):
            raise ValueError("Coordinator adapter subjects must be unique.")
        expected_subjects = {
            runtime.manifest.execution_binding.binding_id,
            *(slot.source_slot_id for slot in runtime.manifest.source_slots),
        }
        if set(subjects) != expected_subjects:
            missing = sorted(expected_subjects - set(subjects))
            unexpected = sorted(set(subjects) - expected_subjects)
            raise ValueError(
                "Coordinator adapters must cover the exact manifest subjects; "
                f"missing={missing}, unexpected={unexpected}."
            )
        self._validate_operation_coverage(runtime, normalized_adapters)

        self.coordinator_id = normalized_id
        self.runtime = runtime
        self.event_log = event_log
        self.queue_capacity = queue_capacity
        self.clock = clock
        self.clock_domain_id = normalized_clock_domain
        self.coordinated_operation_kinds = CS1C_COORDINATED_OPERATION_KINDS
        self._adapters_by_subject = {
            adapter.subject_id: adapter for adapter in normalized_adapters
        }
        self._queue: queue.Queue[ControlStationTransition] = queue.Queue(
            maxsize=queue_capacity
        )
        self._capacity = threading.BoundedSemaphore(queue_capacity)
        self._condition = threading.Condition()
        self._reserved_work_count = 0
        self._running = False
        self._stopping = False
        self._worker: threading.Thread | None = None
        self._completions: list[StationOperationCompletion] = []
        self._event_publish_failure_count = 0

    @staticmethod
    def _validate_operation_coverage(
        runtime: ControlStationRuntime,
        adapters: tuple[DeterministicNoIoStationAdapter, ...],
    ) -> None:
        target_id = runtime.manifest.execution_binding.binding_id
        target_required = {
            "connect_target",
            "disconnect_target",
            "run_preflight",
        } & set(runtime.manifest.allowed_intent_kinds)
        source_required = {
            "connect_source",
            "disconnect_source",
            "run_preflight",
            "start_shadow",
            "stop_shadow",
        } & set(runtime.manifest.allowed_intent_kinds)
        for adapter in adapters:
            required = target_required if adapter.subject_id == target_id else source_required
            missing = sorted(required - set(adapter.supported_operations))
            if missing:
                raise ValueError(
                    f"Adapter {adapter.adapter_id!r} lacks manifest operations: {missing}."
                )

    @property
    def is_running(self) -> bool:
        with self._condition:
            return self._running and not self._stopping

    @property
    def reserved_work_count(self) -> int:
        with self._condition:
            return self._reserved_work_count

    @property
    def completions(self) -> tuple[StationOperationCompletion, ...]:
        with self._condition:
            return tuple(self._completions)

    @property
    def event_publish_failure_count(self) -> int:
        with self._condition:
            return self._event_publish_failure_count

    def _time(self) -> float:
        value = float(self.clock())
        if not math.isfinite(value):
            raise ValueError("Coordinator clock must be finite.")
        return value

    def start(self) -> None:
        with self._condition:
            if self._running or self._worker is not None:
                raise RuntimeError("Coordinator can only be started once.")
            self._running = True
            self._worker = threading.Thread(
                target=self._run,
                name=f"control-station:{self.coordinator_id}",
                daemon=True,
            )
            self._worker.start()

    def submit_intent(self, draft: StationIntentDraft) -> CoordinatedIntentSubmission:
        """Reserve capacity, receiver-admit an intent, then enqueue if requested."""

        if not isinstance(draft, StationIntentDraft):
            raise TypeError("draft must be a StationIntentDraft.")
        requires_capacity = draft.intent_kind in self.coordinated_operation_kinds
        reserved = False
        if requires_capacity:
            with self._condition:
                if not self._running or self._stopping:
                    raise StationOperationCoordinatorUnavailable(
                        "operation_coordinator_not_running"
                    )
                reserved = self._capacity.acquire(blocking=False)
                if not reserved:
                    raise StationOperationCoordinatorUnavailable(
                        "operation_coordinator_queue_full"
                    )
                self._reserved_work_count += 1

        try:
            transition = self.runtime.submit_intent(draft)
            self._publish_intent(transition)
            self._after_intent_admission(transition)
            if transition.receipt.status == "operation_requested":
                if not reserved:
                    raise RuntimeError(
                        "Runtime requested an operation without coordinator capacity."
                    )
                try:
                    self._prepare_enqueued_operation(transition)
                except Exception as error:
                    completion = self._complete_prepare_failure(transition, error)
                    with self._condition:
                        self._completions.append(completion)
                    self._publish_completion(completion)
                    self._release_capacity()
                    reserved = False
                    receipt = self._enqueue_receipt(
                        transition=transition,
                        accepted=False,
                        status="operation_prepare_failed",
                    )
                    self._publish_enqueue(receipt, transition)
                    return CoordinatedIntentSubmission(transition, receipt)
                self._queue.put_nowait(transition)
                reserved = False  # the worker now owns the reserved capacity
                receipt = self._enqueue_receipt(
                    transition=transition,
                    accepted=True,
                    status="operation_enqueued",
                )
                self._publish_enqueue(receipt, transition)
                return CoordinatedIntentSubmission(transition, receipt)
            if reserved:
                self._release_capacity()
                reserved = False
            return CoordinatedIntentSubmission(
                transition,
                self._enqueue_receipt(
                    transition=transition,
                    accepted=False,
                    status="operation_not_requested",
                ),
            )
        except BaseException:
            if reserved:
                self._release_capacity()
            raise

    def _after_intent_admission(
        self,
        transition: ControlStationTransition,
    ) -> None:
        """Optional hook for accepted non-operation safety intents."""

        del transition

    def _prepare_enqueued_operation(
        self,
        transition: ControlStationTransition,
    ) -> None:
        """Prepare an admitted pending operation before queue publication.

        The caller converts ordinary preparation exceptions into a typed
        failed operation result.  Non-operation intents (for example lease
        acquisition) never reach this hook because they do not own a pending
        operation.
        """

        del transition

    def _complete_prepare_failure(
        self,
        transition: ControlStationTransition,
        error: Exception,
    ) -> StationOperationCompletion:
        intent = transition.intent
        adapter = self._adapters_by_subject[intent.subject_id]
        draft = StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=intent.subject_id,
            succeeded=False,
            completed_at=self._time(),
            completed_clock_domain_id=self.clock_domain_id,
            status=f"operation_prepare_exception:{type(error).__name__}",
            evidence_refs=(
                f"coordinator:{self.coordinator_id}:operation_prepare_exception",
            ),
        )
        completion = self.runtime.admit_operation_result(draft)
        return StationOperationCompletion(
            coordinator_id=self.coordinator_id,
            adapter_id=adapter.adapter_id,
            request_transition=transition,
            completion_transition=completion,
        )

    def wait_until_idle(self, *, timeout_sec: float = 2.0) -> bool:
        timeout = self._timeout(timeout_sec)
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._reserved_work_count:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def stop(self, *, timeout_sec: float = 2.0) -> bool:
        timeout = self._timeout(timeout_sec)
        with self._condition:
            if self._worker is None:
                return True
            self._stopping = True
            worker = self._worker
            self._condition.notify_all()
        worker.join(timeout=timeout)
        return not worker.is_alive()

    @staticmethod
    def _timeout(value: float) -> float:
        timeout = float(value)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("timeout_sec must be finite and non-negative.")
        return timeout

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    should_stop = self._stopping and self._reserved_work_count == 0
                if should_stop:
                    return
                try:
                    transition = self._queue.get(timeout=0.05)
                except queue.Empty:
                    continue
                try:
                    completion = self._perform(transition)
                    with self._condition:
                        self._completions.append(completion)
                    self._publish_completion(completion)
                finally:
                    self._queue.task_done()
                    self._release_capacity()
        finally:
            with self._condition:
                self._running = False
                self._condition.notify_all()

    def _perform(self, transition: ControlStationTransition) -> StationOperationCompletion:
        intent = transition.intent
        adapter = self._adapters_by_subject[intent.subject_id]
        try:
            draft = adapter.perform(intent)
            if (
                draft.source_intent_id != intent.intent_id
                or draft.operation_kind != intent.intent_kind
                or draft.subject_id != intent.subject_id
            ):
                raise ValueError("adapter_result_binding_mismatch")
        except Exception as error:  # adapter failure must not kill the worker
            draft = StationOperationResultDraft(
                source_intent_id=intent.intent_id,
                operation_kind=intent.intent_kind,
                subject_id=intent.subject_id,
                succeeded=False,
                completed_at=self._time(),
                completed_clock_domain_id=self.clock_domain_id,
                status=f"adapter_exception:{type(error).__name__}",
                evidence_refs=(f"coordinator:{self.coordinator_id}:adapter_exception",),
            )
        completion = self.runtime.admit_operation_result(draft)
        return StationOperationCompletion(
            coordinator_id=self.coordinator_id,
            adapter_id=adapter.adapter_id,
            request_transition=transition,
            completion_transition=completion,
        )

    def _release_capacity(self) -> None:
        self._capacity.release()
        with self._condition:
            self._reserved_work_count -= 1
            if self._reserved_work_count < 0:
                raise RuntimeError("Coordinator capacity accounting underflow.")
            self._condition.notify_all()

    def _enqueue_receipt(
        self,
        *,
        transition: ControlStationTransition,
        accepted: bool,
        status: str,
    ) -> StationOperationEnqueueReceipt:
        return StationOperationEnqueueReceipt(
            coordinator_id=self.coordinator_id,
            source_intent_id=transition.intent.intent_id,
            accepted=accepted,
            status=status,
            queue_capacity=self.queue_capacity,
            reserved_work_count=self.reserved_work_count,
        )

    def _publish_intent(self, transition: ControlStationTransition) -> None:
        self._publish_event(
            event_kind="station_intent_transition",
            source_fact_id=transition.intent.intent_id,
            source_fact_hash=transition.semantic_content_hash,
            state_revision=transition.state_after.state_revision,
            status=transition.receipt.status,
            subject_id=transition.intent.subject_id,
        )

    def _publish_enqueue(
        self,
        receipt: StationOperationEnqueueReceipt,
        transition: ControlStationTransition,
    ) -> None:
        self._publish_event(
            event_kind="station_operation_enqueued",
            source_fact_id=transition.intent.intent_id,
            source_fact_hash=transition.intent.semantic_content_hash,
            state_revision=transition.state_after.state_revision,
            status=receipt.status,
            subject_id=transition.intent.subject_id,
        )

    def _publish_completion(self, completion: StationOperationCompletion) -> None:
        transition = completion.completion_transition
        self._publish_event(
            event_kind="station_operation_completion",
            source_fact_id=transition.result.operation_result_id,
            source_fact_hash=transition.semantic_content_hash,
            state_revision=transition.state_after.state_revision,
            status=transition.status,
            subject_id=transition.result.subject_id,
        )

    def _publish_event(self, **payload: object) -> None:
        if self.event_log is None:
            return
        try:
            self.event_log.publish(**payload)
        except Exception:
            # This cache is explicitly non-authoritative.  Its failure cannot
            # alter runtime admission or adapter completion.
            with self._condition:
                self._event_publish_failure_count += 1


class SideEffectStationOperationCoordinator(NoIoStationOperationCoordinator):
    """Bounded worker for an explicit application-side operation executor.

    This class intentionally cannot be passed to the CS-1c web builder.  It is
    first used by scoped sim fixtures; a future web builder must opt into this
    maturity explicitly instead of widening the no-I/O server by accident.
    """

    def __init__(
        self,
        *,
        coordinator_id: str,
        runtime: ControlStationRuntime,
        adapters: tuple[StationOperationAdapter, ...],
        operation_executor: Callable[
            [ControlStationTransition], StationOperationExecution
        ],
        operation_prepare: Callable[[ControlStationTransition], None],
        intent_transition_observer: Callable[[ControlStationTransition], object]
        | None = None,
        operation_transition_observer: Callable[
            [ControlStationOperationTransition], object
        ]
        | None = None,
        event_log: BoundedStationEventLog | None = None,
        queue_capacity: int = 16,
        clock: Callable[[], float] = time.monotonic,
        clock_domain_id: str = "station-side-effect-coordinator-monotonic",
    ) -> None:
        normalized_id = str(coordinator_id).strip()
        normalized_clock_domain = str(clock_domain_id).strip()
        if not normalized_id or not normalized_clock_domain:
            raise ValueError("Coordinator identity/clock must be non-empty.")
        if not isinstance(runtime, ControlStationRuntime):
            raise TypeError("runtime must be a ControlStationRuntime.")
        if event_log is not None and not isinstance(event_log, BoundedStationEventLog):
            raise TypeError("event_log must be a BoundedStationEventLog or None.")
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or not 1 <= queue_capacity <= 1024
        ):
            raise ValueError("queue_capacity must be an integer in [1, 1024].")
        if (
            not callable(clock)
            or not callable(operation_executor)
            or not callable(operation_prepare)
        ):
            raise TypeError(
                "clock, operation_executor, and operation_prepare must be callable."
            )
        if intent_transition_observer is not None and not callable(
            intent_transition_observer
        ):
            raise TypeError("intent_transition_observer must be callable or None.")
        if operation_transition_observer is not None and not callable(
            operation_transition_observer
        ):
            raise TypeError("operation_transition_observer must be callable or None.")

        normalized_adapters = tuple(adapters)
        if not normalized_adapters:
            raise ValueError("Side-effect coordinator requires explicit adapters.")
        adapter_ids = tuple(str(adapter.adapter_id) for adapter in normalized_adapters)
        subjects = tuple(str(adapter.subject_id) for adapter in normalized_adapters)
        if any(not value for value in (*adapter_ids, *subjects)):
            raise ValueError("Adapter identity/subject must be non-empty.")
        if len(set(adapter_ids)) != len(adapter_ids):
            raise ValueError("Coordinator adapter IDs must be unique.")
        if len(set(subjects)) != len(subjects):
            raise ValueError("Coordinator adapter subjects must be unique.")
        expected_subjects = {
            runtime.manifest.execution_binding.binding_id,
            *(slot.source_slot_id for slot in runtime.manifest.source_slots),
        }
        if set(subjects) != expected_subjects:
            missing = sorted(expected_subjects - set(subjects))
            unexpected = sorted(set(subjects) - expected_subjects)
            raise ValueError(
                "Coordinator adapters must cover the exact manifest subjects; "
                f"missing={missing}, unexpected={unexpected}."
            )
        operations = frozenset(
            operation
            for adapter in normalized_adapters
            for operation in adapter.supported_operations
        )
        if not operations or not operations.issubset(
            runtime.manifest.allowed_intent_kinds
        ):
            raise ValueError(
                "Side-effect adapter operations must be allowed by the manifest."
            )

        self.coordinator_id = normalized_id
        self.runtime = runtime
        self.event_log = event_log
        self.queue_capacity = queue_capacity
        self.clock = clock
        self.clock_domain_id = normalized_clock_domain
        self.coordinated_operation_kinds = operations
        self._adapters_by_subject = {
            adapter.subject_id: adapter for adapter in normalized_adapters
        }
        self._operation_executor = operation_executor
        self._operation_prepare = operation_prepare
        self._intent_transition_observer = intent_transition_observer
        self._operation_transition_observer = operation_transition_observer
        self._transition_observer_failure_count = 0
        self._queue = queue.Queue(maxsize=queue_capacity)
        self._capacity = threading.BoundedSemaphore(queue_capacity)
        self._condition = threading.Condition()
        self._reserved_work_count = 0
        self._running = False
        self._stopping = False
        self._worker: threading.Thread | None = None
        self._completions: list[StationOperationCompletion] = []
        self._event_publish_failure_count = 0

    def _prepare_enqueued_operation(
        self,
        transition: ControlStationTransition,
    ) -> None:
        self._operation_prepare(transition)

    _NON_OPERATION_SAFETY_PREPARE_INTENTS = frozenset({"hold", "estop"})

    def _after_intent_admission(
        self,
        transition: ControlStationTransition,
    ) -> None:
        """Prepare accepted safety latches that have no pending operation.

        Lease acquisition/release is deliberately excluded: those intents
        only update receiver state and must not run a lower-operation hook.
        HOLD/ESTOP are the explicit non-operation safety seam; their prepare
        failure remains an exception because there is no pending operation
        that could be resolved into a typed result.
        """

        if (
            transition.receipt.accepted
            and transition.receipt.status != "operation_requested"
            and transition.intent.intent_kind
            in self._NON_OPERATION_SAFETY_PREPARE_INTENTS
        ):
            self._operation_prepare(transition)

    @property
    def transition_observer_failure_count(self) -> int:
        with self._condition:
            return self._transition_observer_failure_count

    def _publish_intent(self, transition: ControlStationTransition) -> None:
        super()._publish_intent(transition)
        observer = self._intent_transition_observer
        if observer is None:
            return
        try:
            observer(transition)
        except Exception:
            with self._condition:
                self._transition_observer_failure_count += 1

    def _publish_completion(self, completion: StationOperationCompletion) -> None:
        super()._publish_completion(completion)
        observer = self._operation_transition_observer
        if observer is None:
            return
        try:
            observer(completion.completion_transition)
        except Exception:
            with self._condition:
                self._transition_observer_failure_count += 1

    def _perform(self, transition: ControlStationTransition) -> StationOperationCompletion:
        intent = transition.intent
        adapter = self._adapters_by_subject[intent.subject_id]
        try:
            execution = self._operation_executor(transition)
            if execution.request_transition.intent.intent_id != intent.intent_id:
                raise ValueError("operation_executor_request_mismatch")
            completion = execution.completion_transition
            if (
                completion.result.source_intent_id != intent.intent_id
                or completion.result.operation_kind != intent.intent_kind
                or completion.result.subject_id != intent.subject_id
            ):
                raise ValueError("operation_executor_result_binding_mismatch")
        except Exception as error:  # fail closed without killing the worker
            draft = StationOperationResultDraft(
                source_intent_id=intent.intent_id,
                operation_kind=intent.intent_kind,
                subject_id=intent.subject_id,
                succeeded=False,
                completed_at=self._time(),
                completed_clock_domain_id=self.clock_domain_id,
                status=f"side_effect_executor_exception:{type(error).__name__}",
                evidence_refs=(
                    f"coordinator:{self.coordinator_id}:side_effect_executor_exception",
                ),
            )
            completion = self.runtime.admit_operation_result(draft)
        return StationOperationCompletion(
            coordinator_id=self.coordinator_id,
            adapter_id=adapter.adapter_id,
            request_transition=transition,
            completion_transition=completion,
        )


__all__ = [
    "CS1C_COORDINATED_OPERATION_KINDS",
    "CoordinatedIntentSubmission",
    "NoIoStationOperationCoordinator",
    "SideEffectStationOperationCoordinator",
    "StationOperationCompletion",
    "StationOperationCoordinatorUnavailable",
    "StationOperationEnqueueReceipt",
]

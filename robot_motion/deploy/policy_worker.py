"""Bounded, model-neutral OS-thread carrier for RTC policy inference.

This module belongs to the shared deploy runtime rather than a teleop or
Control Station frontend.  It owns only one policy
inference thread plus bounded request/completion mailboxes.  It does not own a
timed action plan, proposal sequence, execution epoch, worker-drain evidence,
safety, reference generation, dispatcher, or backend.  A stop receipt that
reports an executing call is therefore explicitly incomplete; callers must
invalidate the policy producer epoch and reject any later completion at the
plan merge boundary.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation import (
    normalize_model_continuation_contract,
)
from .inference import PolicyChunkResult, PolicyInferenceRequest
from .async_producer import (
    POLICY_WORKER_SESSION_ABANDONED_UNDRAINED,
    POLICY_WORKER_SESSION_ACTIVE,
    POLICY_WORKER_SESSION_DRAINED,
    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION,
    POLICY_WORKER_SESSION_STATES_V1,
)
from .observation_view import PolicyObservationView
from .rtc.conditioning import CanonicalRtcConditioningBundle
from .rtc.continuation import rtc_continuation_timeline_from_request


POLICY_WORK_ITEM_HASH_DOMAIN_V1 = "dexteleop.deploy.policy_work_item.v1"
POLICY_WORKER_SUBMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.policy_worker_submission.v1"
)
POLICY_WORKER_COMPLETION_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.policy_worker_completion.v1"
)
POLICY_WORKER_STOP_HASH_DOMAIN_V1 = "dexteleop.deploy.policy_worker_stop.v1"
POLICY_WORK_ITEM_HASH_DOMAIN_V2 = "dexteleop.deploy.policy_work_item.v2"
POLICY_WORK_ITEM_HASH_DOMAIN_V3 = "dexteleop.deploy.policy_work_item.v3"
POLICY_WORK_ITEM_HASH_DOMAIN_V4 = "dexteleop.deploy.policy_work_item.v4"
POLICY_WORKER_SUBMISSION_HASH_DOMAIN_V2 = (
    "dexteleop.deploy.policy_worker_submission.v2"
)
POLICY_WORKER_COMPLETION_HASH_DOMAIN_V2 = (
    "dexteleop.deploy.policy_worker_completion.v2"
)
POLICY_WORKER_STOP_HASH_DOMAIN_V2 = "dexteleop.deploy.policy_worker_stop.v2"
POLICY_WORKER_STOP_HASH_DOMAIN_V3 = "dexteleop.deploy.policy_worker_stop.v3"
POLICY_ACTOR_OBSERVATION_BINDING_HASH_DOMAIN_V2 = (
    "dexteleop.deploy.policy_worker_actor_observation_binding.v2"
)
POLICY_EXECUTOR_CANCELLATION_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.policy_executor_cancellation.v1"
)
POLICY_WORKER_SESSION_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.deploy.policy_worker_session_snapshot.v1"
)

POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED = "not_supported"
POLICY_EXECUTOR_CANCEL_PENDING = "pending"
POLICY_EXECUTOR_CANCEL_REQUESTED = "requested"
POLICY_EXECUTOR_CANCEL_DECLINED = "declined"
POLICY_EXECUTOR_CANCEL_FAILED = "failed"
POLICY_EXECUTOR_CANCELLATION_STATUSES_V1 = (
    POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED,
    POLICY_EXECUTOR_CANCEL_PENDING,
    POLICY_EXECUTOR_CANCEL_REQUESTED,
    POLICY_EXECUTOR_CANCEL_DECLINED,
    POLICY_EXECUTOR_CANCEL_FAILED,
)
POLICY_WORKER_FAULT_CODE_MAX_CHARS = 128
POLICY_WORKER_FAULT_DETAIL_MAX_CHARS = 512


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer.")
    normalized = int(value)
    if normalized != value or normalized < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return normalized


class PolicyInferenceExecutor(Protocol):
    """Blocking model/transport call run only by the policy worker thread."""

    def __call__(
        self,
        request: PolicyInferenceRequest,
        actor_observation: PolicyObservationView,
    ) -> PolicyChunkResult:
        ...


class PolicyInferenceWorkItemExecutor(PolicyInferenceExecutor, Protocol):
    """Opt-in executor that consumes the exact immutable queued work item."""

    def execute_work_item(
        self,
        work_item: "PolicyInferenceWorkItem",
    ) -> PolicyChunkResult:
        ...


class PolicyInferenceCancellationHook(Protocol):
    """Optional, non-blocking cancellation hook for a session executor.

    ``request_cancel`` must only signal cancellation / begin releasing
    session-scoped resources.  Returning ``True`` means the signal was
    accepted, not that the executing call or transport has drained.  Worker
    thread exit remains the sole drain fact owned by this module.
    """

    def request_cancel(self) -> bool:
        ...


@dataclass(frozen=True)
class PolicyInferenceWorkItem:
    worker_id: str
    worker_session_id: str
    submit_sequence: int
    request: PolicyInferenceRequest
    actor_observation: PolicyObservationView
    submitted_at: float
    conditioning_bundle: CanonicalRtcConditioningBundle | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _required(self.worker_id, field_name="worker_id"))
        object.__setattr__(
            self,
            "worker_session_id",
            _required(self.worker_session_id, field_name="worker_session_id"),
        )
        object.__setattr__(
            self,
            "submit_sequence",
            _nonnegative_integer(self.submit_sequence, field_name="submit_sequence"),
        )
        if not isinstance(self.request, PolicyInferenceRequest):
            raise TypeError("request must be a PolicyInferenceRequest.")
        if not isinstance(self.actor_observation, PolicyObservationView):
            raise TypeError("actor_observation must be a PolicyObservationView.")
        submitted = _finite(self.submitted_at, field_name="submitted_at")
        if submitted < self.request.admitted_at:
            raise ValueError("submitted_at cannot predate request admission.")
        object.__setattr__(self, "submitted_at", submitted)
        payload = self.actor_observation.payload
        if "timestamp" not in payload or "step_index" not in payload:
            raise ValueError(
                "Actor observation must expose timestamp and step_index identity."
            )
        if float(payload["timestamp"]) != self.request.observation_timestamp:
            raise ValueError("Actor observation/request timestamp mismatch.")
        if int(payload["step_index"]) != self.request.observation_step_index:
            raise ValueError("Actor observation/request step mismatch.")
        self._validate_conditioning_binding()

    def _validate_conditioning_binding(self) -> None:
        request = self.request
        bundle = self.conditioning_bundle
        if request.conditioning_binding_hash is None:
            if bundle is not None:
                raise ValueError(
                    "A conditioning bundle requires a causal request binding."
                )
            return
        if bundle is None:
            raise ValueError(
                "A causal conditioning request requires its immutable bundle "
                "in the work item."
            )
        if not isinstance(bundle, CanonicalRtcConditioningBundle):
            raise TypeError(
                "conditioning_bundle must be a "
                "CanonicalRtcConditioningBundle or None."
            )
        if request.conditioning_binding_hash != bundle.conditioning_binding_hash:
            raise ValueError("Request/conditioning bundle hash mismatch.")
        if request.session_id != bundle.runtime_session_id:
            raise ValueError("Request/conditioning runtime session mismatch.")
        if request.control_domain_id != bundle.control_domain_id:
            raise ValueError("Request/conditioning control domain mismatch.")
        if request.artifact_hash != bundle.artifact_hash:
            raise ValueError("Request/conditioning artifact mismatch.")
        if request.source_plan_hash != bundle.selection.source_plan_hash:
            raise ValueError("Request/conditioning source plan hash mismatch.")
        if (
            rtc_continuation_timeline_from_request(request)
            .semantic_content_hash
            != bundle.timeline.semantic_content_hash
        ):
            raise ValueError("Request/conditioning timeline mismatch.")
        context = request.action_representation_context
        if context is None:
            raise ValueError(
                "A canonical conditioning request requires an action "
                "representation context."
            )
        if (
            context.semantic_content_hash
            != bundle.action_representation_context.semantic_content_hash
        ):
            raise ValueError(
                "Request/conditioning action representation context mismatch."
            )
        contract = normalize_model_continuation_contract(
            bundle.continuation_contract
        ).runtime
        for name in (
            "action_codec_hash",
            "frame_contract_hash",
            "layout_hash",
            "normalization_hash",
        ):
            if getattr(request, name) != getattr(contract, name):
                raise ValueError(
                    f"Request/conditioning continuation {name} mismatch."
                )
        continuation_slice = bundle.selection.continuation_slice
        if continuation_slice is None:
            return
        expected = {
            "source_plan_id": request.source_plan_id,
            "source_plan_generation": request.source_plan_generation,
            "producer_id": request.producer_id,
            "producer_epoch": request.producer_epoch,
            "policy_node_id": request.policy_node_id,
        }
        for name, expected_value in expected.items():
            if getattr(continuation_slice, name) != expected_value:
                raise ValueError(
                    f"Request/conditioning continuation {name} mismatch."
                )

    @property
    def actor_observation_hash(self) -> str:
        """Hash the queued observation's causal binding, not bulk sensor bytes.

        The immutable in-process view is carried by this work item.  Exact
        model-facing content is prepared and attested later on the worker via
        ``model_input_hash``.
        """

        return canonical_content_hash(
            {
                "contract_id": self.actor_observation.contract_id,
                "contract_hash": self.actor_observation.contract_hash,
                "compatibility_adapter_id": (
                    self.actor_observation.compatibility_adapter_id
                ),
                "observation_cut_id": self.request.observation_cut_id,
                "observation_timestamp": self.actor_observation.payload["timestamp"],
                "observation_step_index": self.actor_observation.payload["step_index"],
            },
            domain=POLICY_ACTOR_OBSERVATION_BINDING_HASH_DOMAIN_V2,
        )

    @property
    def semantic_content_hash(self) -> str:
        payload = {
            "worker_id": self.worker_id,
            "worker_session_id": self.worker_session_id,
            "submit_sequence": self.submit_sequence,
            "request_hash": self.request.semantic_content_hash,
            "actor_observation_hash": self.actor_observation_hash,
            "submitted_at": self.submitted_at,
        }
        domain = POLICY_WORK_ITEM_HASH_DOMAIN_V3
        if self.conditioning_bundle is not None:
            payload["conditioning_bundle_hash"] = (
                self.conditioning_bundle.conditioning_binding_hash
            )
            domain = POLICY_WORK_ITEM_HASH_DOMAIN_V4
        return canonical_content_hash(
            payload,
            domain=domain,
        )


@dataclass(frozen=True)
class PolicyWorkerSubmissionReceipt:
    worker_id: str
    worker_session_id: str
    request_id: str
    work_item_hash: str
    occurred_at: float
    accepted: bool
    status: str

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "worker_id": self.worker_id,
                "worker_session_id": self.worker_session_id,
                "request_id": self.request_id,
                "work_item_hash": self.work_item_hash,
                "occurred_at": self.occurred_at,
                "accepted": self.accepted,
                "status": self.status,
            },
            domain=POLICY_WORKER_SUBMISSION_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class PolicyWorkerCompletion:
    worker_id: str
    worker_session_id: str
    request_id: str
    work_item_hash: str
    worker_started_at: float
    worker_completed_at: float
    released_at: float
    result: PolicyChunkResult

    @property
    def status(self) -> str:
        return self.result.status

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "worker_id": self.worker_id,
                "worker_session_id": self.worker_session_id,
                "request_id": self.request_id,
                "work_item_hash": self.work_item_hash,
                "worker_started_at": self.worker_started_at,
                "worker_completed_at": self.worker_completed_at,
                "released_at": self.released_at,
                "result_hash": self.result.semantic_content_hash,
            },
            domain=POLICY_WORKER_COMPLETION_HASH_DOMAIN_V2,
        )


@dataclass(frozen=True)
class PolicyExecutorCancellationReceipt:
    """Fact emitted when worker stop asks an executor to release its session."""

    executing_request_id: str | None
    hook_available: bool
    request_accepted: bool
    status: str
    failure_type: str | None = None
    failure_detail: str | None = None

    def __post_init__(self) -> None:
        request_id = self.executing_request_id
        if request_id is not None:
            object.__setattr__(
                self,
                "executing_request_id",
                _required(request_id, field_name="executing_request_id"),
            )
        status = _required(self.status, field_name="status")
        if status not in POLICY_EXECUTOR_CANCELLATION_STATUSES_V1:
            raise ValueError(f"Unsupported executor cancellation status: {status!r}.")
        object.__setattr__(self, "status", status)
        if self.request_accepted and status != POLICY_EXECUTOR_CANCEL_REQUESTED:
            raise ValueError("Accepted cancellation must have status='requested'.")
        if status == POLICY_EXECUTOR_CANCEL_REQUESTED and not self.request_accepted:
            raise ValueError("Cancellation status='requested' must be accepted.")
        if not self.hook_available and status != POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED:
            raise ValueError("Missing cancellation hook must have status='not_supported'.")
        if status == POLICY_EXECUTOR_CANCEL_FAILED:
            object.__setattr__(
                self,
                "failure_type",
                _required(self.failure_type, field_name="failure_type"),
            )
            object.__setattr__(
                self,
                "failure_detail",
                _required(self.failure_detail, field_name="failure_detail"),
            )
        elif self.failure_type is not None or self.failure_detail is not None:
            raise ValueError("Only failed cancellation may carry failure details.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "executing_request_id": self.executing_request_id,
                "hook_available": self.hook_available,
                "request_accepted": self.request_accepted,
                "status": self.status,
                "failure_type": self.failure_type,
                "failure_detail": self.failure_detail,
            },
            domain=POLICY_EXECUTOR_CANCELLATION_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class PolicyWorkerStopReceipt:
    worker_id: str
    worker_session_id: str
    stop_sequence: int
    requested_at: float
    returned_at: float
    thread_exited: bool
    active_request_id: str | None
    executing_request_id: str | None
    discarded_queued_request_ids: tuple[str, ...]
    pending_completion_count: int
    status: str
    previous_session_state: str
    session_state: str
    execution_drain_evidence: bool = False
    executor_cancellation: PolicyExecutorCancellationReceipt | None = None

    def __post_init__(self) -> None:
        if self.executor_cancellation is not None and not isinstance(
            self.executor_cancellation,
            PolicyExecutorCancellationReceipt,
        ):
            raise TypeError(
                "executor_cancellation must be a "
                "PolicyExecutorCancellationReceipt or None."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "worker_id": self.worker_id,
                "worker_session_id": self.worker_session_id,
                "stop_sequence": self.stop_sequence,
                "requested_at": self.requested_at,
                "returned_at": self.returned_at,
                "thread_exited": self.thread_exited,
                "active_request_id": self.active_request_id,
                "executing_request_id": self.executing_request_id,
                "discarded_queued_request_ids": self.discarded_queued_request_ids,
                "pending_completion_count": self.pending_completion_count,
                "status": self.status,
                "previous_session_state": self.previous_session_state,
                "session_state": self.session_state,
                "execution_drain_evidence": self.execution_drain_evidence,
                "executor_cancellation_hash": (
                    None
                    if self.executor_cancellation is None
                    else self.executor_cancellation.semantic_content_hash
                ),
            },
            domain=POLICY_WORKER_STOP_HASH_DOMAIN_V3,
        )


@dataclass(frozen=True)
class PolicyWorkerSessionSnapshot:
    worker_id: str
    worker_session_id: str
    session_state: str
    observed_at: float
    active_request_id: str | None
    executing_request_id: str | None
    pending_completion_count: int
    internal_fault: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "worker_id", _required(self.worker_id, field_name="worker_id"))
        object.__setattr__(
            self,
            "worker_session_id",
            _required(self.worker_session_id, field_name="worker_session_id"),
        )
        session_state = _required(self.session_state, field_name="session_state")
        if session_state not in POLICY_WORKER_SESSION_STATES_V1:
            raise ValueError(f"Unsupported worker session state: {session_state!r}.")
        object.__setattr__(self, "session_state", session_state)
        object.__setattr__(
            self,
            "observed_at",
            _finite(self.observed_at, field_name="observed_at"),
        )
        object.__setattr__(
            self,
            "pending_completion_count",
            _nonnegative_integer(
                self.pending_completion_count,
                field_name="pending_completion_count",
            ),
        )
        for name in ("active_request_id", "executing_request_id", "internal_fault"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required(value, field_name=name))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "worker_id": self.worker_id,
                "worker_session_id": self.worker_session_id,
                "session_state": self.session_state,
                "observed_at": self.observed_at,
                "active_request_id": self.active_request_id,
                "executing_request_id": self.executing_request_id,
                "pending_completion_count": self.pending_completion_count,
                "internal_fault": self.internal_fault,
            },
            domain=POLICY_WORKER_SESSION_SNAPSHOT_HASH_DOMAIN_V1,
        )


def _fault_result(
    request: PolicyInferenceRequest,
    *,
    inference_started_at: float,
    inference_completed_at: float,
    result_received_at: float,
    error: BaseException,
) -> PolicyChunkResult:
    error_type = type(error).__name__
    fault_code = f"policy_worker_exception:{error_type}"
    if len(fault_code) > POLICY_WORKER_FAULT_CODE_MAX_CHARS:
        fault_code = fault_code[:POLICY_WORKER_FAULT_CODE_MAX_CHARS]
    fault_detail = " ".join(str(error).split()) or error_type
    if len(fault_detail) > POLICY_WORKER_FAULT_DETAIL_MAX_CHARS:
        fault_detail = (
            fault_detail[: POLICY_WORKER_FAULT_DETAIL_MAX_CHARS - 3] + "..."
        )
    return PolicyChunkResult(
        request_id=request.request_id,
        request_hash=request.semantic_content_hash,
        inference_id=request.inference_id,
        rtc_mode=request.rtc_mode,
        session_id=request.session_id,
        control_domain_id=request.control_domain_id,
        producer_id=request.producer_id,
        producer_epoch=request.producer_epoch,
        policy_node_id=request.policy_node_id,
        artifact_hash=request.artifact_hash,
        action_codec_hash=request.action_codec_hash,
        frame_contract_hash=request.frame_contract_hash,
        layout_hash=request.layout_hash,
        normalization_hash=request.normalization_hash,
        observation_cut_id=request.observation_cut_id,
        observation_timestamp=request.observation_timestamp,
        observation_step_index=request.observation_step_index,
        policy_output_id=None,
        chunk_id=None,
        inference_started_at=inference_started_at,
        inference_completed_at=inference_completed_at,
        result_received_at=result_received_at,
        clock_domain_id=request.clock_domain_id,
        status="fault",
        source_chunk_size=0,
        decoded_absolute_actions=(),
        model_output_hash=None,
        protocol_version="bounded_policy_worker_v1",
        fault_code=fault_code,
        fault_detail=fault_detail,
        conditioning_binding_hash=request.conditioning_binding_hash,
    )


class BoundedPolicyInferenceWorker:
    """One blocking inference call at a time behind non-blocking mailboxes."""

    def __init__(
        self,
        *,
        worker_id: str,
        worker_session_id: str,
        executor: PolicyInferenceExecutor,
        result_capacity: int = 1,
        clock: Callable[[], float] = time.monotonic,
        idle_poll_sec: float = 0.01,
    ) -> None:
        self.worker_id = _required(worker_id, field_name="worker_id")
        self.worker_session_id = _required(
            worker_session_id,
            field_name="worker_session_id",
        )
        if not callable(executor):
            raise TypeError("executor must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        capacity = _nonnegative_integer(result_capacity, field_name="result_capacity")
        if capacity < 1:
            raise ValueError("result_capacity must be positive.")
        poll = _finite(idle_poll_sec, field_name="idle_poll_sec")
        if poll <= 0.0:
            raise ValueError("idle_poll_sec must be positive.")
        self.executor = executor
        self.clock = clock
        self.idle_poll_sec = poll
        self._requests: queue.Queue[PolicyInferenceWorkItem] = queue.Queue(maxsize=1)
        self._completions: queue.Queue[PolicyWorkerCompletion] = queue.Queue(
            maxsize=capacity
        )
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._state = "created"
        self._session_state = POLICY_WORKER_SESSION_ACTIVE
        self._active_request_id: str | None = None
        self._executing_request_id: str | None = None
        self._known_request_ids: set[str] = set()
        self._discarded_queued_request_ids: list[str] = []
        self._next_stop_sequence = 0
        self._internal_fault: str | None = None
        self._executor_cancel_started = False
        self._executor_cancel_hook_available = False
        self._executor_cancel_executing_request_id: str | None = None
        self._executor_cancel_outcome: tuple[bool, str | None, str | None] | None = (
            None
        )
        self._executor_cancel_thread: threading.Thread | None = None

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def active_request_id(self) -> str | None:
        with self._lock:
            return self._active_request_id

    @property
    def session_state(self) -> str:
        with self._lock:
            self._refresh_retired_session_locked()
            return self._session_state

    def _refresh_retired_session_locked(self) -> None:
        if (
            self._stop_event.is_set()
            and self._executing_request_id is None
            and (self._thread is None or not self._thread.is_alive())
        ):
            cancel_pending = (
                self._executor_cancel_hook_available
                and self._executor_cancel_outcome is None
            )
            self._session_state = (
                POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION
                if cancel_pending else POLICY_WORKER_SESSION_DRAINED
            )

    @property
    def executing_request_id(self) -> str | None:
        with self._lock:
            return self._executing_request_id

    @property
    def pending_completion_count(self) -> int:
        return self._completions.qsize()

    @property
    def internal_fault(self) -> str | None:
        with self._lock:
            return self._internal_fault

    def _time(self) -> float:
        return _finite(self.clock(), field_name="worker clock")

    def session_snapshot(self) -> PolicyWorkerSessionSnapshot:
        observed_at = self._time()
        with self._lock:
            self._refresh_retired_session_locked()
            return PolicyWorkerSessionSnapshot(
                worker_id=self.worker_id,
                worker_session_id=self.worker_session_id,
                session_state=self._session_state,
                observed_at=observed_at,
                active_request_id=self._active_request_id,
                executing_request_id=self._executing_request_id,
                pending_completion_count=self._completions.qsize(),
                internal_fault=self._internal_fault,
            )

    def start(self) -> None:
        with self._lock:
            if self._state != "created":
                raise RuntimeError("Policy inference worker can only be started once.")
            self._state = "running"
            thread = threading.Thread(
                target=self._run,
                name=f"dexteleop-policy-worker:{self.worker_id}",
                daemon=True,
            )
            self._thread = thread
            thread.start()

    def submit(self, work_item: PolicyInferenceWorkItem) -> PolicyWorkerSubmissionReceipt:
        if not isinstance(work_item, PolicyInferenceWorkItem):
            raise TypeError("work_item must be a PolicyInferenceWorkItem.")
        if work_item.worker_id != self.worker_id:
            raise ValueError("Work item belongs to another policy worker.")
        if work_item.worker_session_id != self.worker_session_id:
            raise ValueError("Work item belongs to another policy worker session.")
        occurred_at = self._time()
        with self._lock:
            if self._state != "running":
                status = "worker_not_running"
            elif self._active_request_id is not None:
                status = "in_flight_request_exists"
            elif work_item.request.request_id in self._known_request_ids:
                status = "duplicate_request_id"
            else:
                try:
                    self._requests.put_nowait(work_item)
                except queue.Full:
                    status = "request_mailbox_full"
                else:
                    self._active_request_id = work_item.request.request_id
                    self._known_request_ids.add(work_item.request.request_id)
                    status = "accepted"
        return PolicyWorkerSubmissionReceipt(
            worker_id=self.worker_id,
            worker_session_id=self.worker_session_id,
            request_id=work_item.request.request_id,
            work_item_hash=work_item.semantic_content_hash,
            occurred_at=occurred_at,
            accepted=status == "accepted",
            status=status,
        )

    def poll_completion(self) -> PolicyWorkerCompletion | None:
        try:
            completion = self._completions.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            if self._active_request_id == completion.request_id:
                self._active_request_id = None
        return completion

    def _run(self) -> None:
        try:
            while True:
                if self._stop_event.is_set() and self._requests.empty():
                    break
                try:
                    work_item = self._requests.get(timeout=self.idle_poll_sec)
                except queue.Empty:
                    continue
                with self._lock:
                    if self._stop_event.is_set():
                        request_id = work_item.request.request_id
                        self._discarded_queued_request_ids.append(request_id)
                        if self._active_request_id == request_id:
                            self._active_request_id = None
                        continue
                    self._executing_request_id = work_item.request.request_id
                started_at = max(self._time(), work_item.request.admitted_at)
                try:
                    if work_item.conditioning_bundle is None:
                        result = self.executor(
                            work_item.request,
                            work_item.actor_observation,
                        )
                    else:
                        execute_work_item = getattr(
                            self.executor,
                            "execute_work_item",
                            None,
                        )
                        if not callable(execute_work_item):
                            raise TypeError(
                                "A conditioning bundle requires an executor "
                                "with execute_work_item()."
                            )
                        result = execute_work_item(work_item)
                    if not isinstance(result, PolicyChunkResult):
                        raise TypeError("Policy executor must return PolicyChunkResult.")
                except Exception as error:  # noqa: BLE001 - converted to typed fault.
                    completed_at = max(self._time(), started_at)
                    received_at = max(self._time(), completed_at)
                    result = _fault_result(
                        work_item.request,
                        inference_started_at=started_at,
                        inference_completed_at=completed_at,
                        result_received_at=received_at,
                        error=error,
                    )
                completed_at = max(self._time(), started_at)
                released_at = max(self._time(), completed_at)
                completion = PolicyWorkerCompletion(
                    worker_id=self.worker_id,
                    worker_session_id=self.worker_session_id,
                    request_id=work_item.request.request_id,
                    work_item_hash=work_item.semantic_content_hash,
                    worker_started_at=started_at,
                    worker_completed_at=completed_at,
                    released_at=released_at,
                    result=result,
                )
                with self._lock:
                    self._executing_request_id = None
                try:
                    self._completions.put_nowait(completion)
                except queue.Full:
                    with self._lock:
                        self._internal_fault = "result_mailbox_full"
                        self._state = "faulted"
                    self._stop_event.set()
                    break
        finally:
            with self._lock:
                if self._state != "faulted":
                    self._state = "stopped"
                self._executing_request_id = None
                self._session_state = (
                    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION
                    if self._executor_cancel_hook_available
                    and self._executor_cancel_outcome is None
                    else POLICY_WORKER_SESSION_DRAINED
                )

    def _dispatch_executor_cancel(
        self,
        *,
        executing_request_id: str | None,
    ) -> None:
        hook = getattr(self.executor, "request_cancel", None)
        with self._lock:
            if self._executor_cancel_started:
                return
            self._executor_cancel_started = True
            self._executor_cancel_executing_request_id = executing_request_id
            self._executor_cancel_hook_available = callable(hook)
            if not callable(hook):
                return
            thread = threading.Thread(
                target=self._run_executor_cancel_hook,
                args=(hook,),
                name=f"dexteleop-policy-cancel:{self.worker_id}",
                daemon=True,
            )
            self._executor_cancel_thread = thread
        try:
            thread.start()
        except Exception as error:  # noqa: BLE001 - captured as typed evidence.
            with self._lock:
                self._executor_cancel_outcome = (
                    False,
                    type(error).__name__,
                    str(error) or repr(error),
                )

    def _run_executor_cancel_hook(self, hook: Callable[[], object]) -> None:
        try:
            accepted = hook()
            if not isinstance(accepted, bool):
                raise TypeError("executor request_cancel() must return bool.")
        except Exception as error:  # noqa: BLE001 - captured as typed evidence.
            outcome = (
                False,
                type(error).__name__,
                str(error) or repr(error),
            )
        else:
            outcome = (accepted, None, None)
        with self._lock:
            self._executor_cancel_outcome = outcome

    def _executor_cancel_receipt(self) -> PolicyExecutorCancellationReceipt:
        with self._lock:
            executing_request_id = self._executor_cancel_executing_request_id
            hook_available = self._executor_cancel_hook_available
            outcome = self._executor_cancel_outcome
        if not hook_available:
            return PolicyExecutorCancellationReceipt(
                executing_request_id=executing_request_id,
                hook_available=False,
                request_accepted=False,
                status=POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED,
            )
        if outcome is None:
            return PolicyExecutorCancellationReceipt(
                executing_request_id=executing_request_id,
                hook_available=True,
                request_accepted=False,
                status=POLICY_EXECUTOR_CANCEL_PENDING,
            )
        accepted, failure_type, failure_detail = outcome
        if failure_type is not None:
            return PolicyExecutorCancellationReceipt(
                executing_request_id=executing_request_id,
                hook_available=True,
                request_accepted=False,
                status=POLICY_EXECUTOR_CANCEL_FAILED,
                failure_type=failure_type,
                failure_detail=failure_detail,
            )
        return PolicyExecutorCancellationReceipt(
            executing_request_id=executing_request_id,
            hook_available=True,
            request_accepted=accepted,
            status=(
                POLICY_EXECUTOR_CANCEL_REQUESTED
                if accepted
                else POLICY_EXECUTOR_CANCEL_DECLINED
            ),
        )

    def stop(self, *, timeout_sec: float = 1.0) -> PolicyWorkerStopReceipt:
        timeout = _finite(timeout_sec, field_name="timeout_sec")
        if timeout < 0.0:
            raise ValueError("timeout_sec must be non-negative.")
        requested_at = self._time()
        with self._lock:
            previous_session_state = self._session_state
            stop_sequence = self._next_stop_sequence
            self._next_stop_sequence += 1
            if self._state == "created":
                self._state = "stopped"
                self._session_state = POLICY_WORKER_SESSION_DRAINED
            elif self._state == "running":
                self._state = "stopping"
                self._session_state = (
                    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION
                )
            self._stop_event.set()
            while True:
                try:
                    work_item = self._requests.get_nowait()
                except queue.Empty:
                    break
                request_id = work_item.request.request_id
                self._discarded_queued_request_ids.append(request_id)
                if (
                    self._active_request_id == request_id
                    and self._executing_request_id != request_id
                ):
                    self._active_request_id = None
            thread = self._thread
            executing_at_cancel = self._executing_request_id
        self._dispatch_executor_cancel(
            executing_request_id=executing_at_cancel,
        )
        stop_deadline = time.monotonic() + timeout
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout)
        with self._lock:
            cancel_thread = self._executor_cancel_thread
        if (
            cancel_thread is not None
            and cancel_thread is not threading.current_thread()
        ):
            cancel_thread.join(
                timeout=max(0.0, stop_deadline - time.monotonic())
            )
        executor_cancellation = self._executor_cancel_receipt()
        thread_exited = thread is None or not thread.is_alive()
        returned_at = max(self._time(), requested_at)
        with self._lock:
            active_request_id = self._active_request_id
            executing_request_id = self._executing_request_id
            discarded = tuple(self._discarded_queued_request_ids)
            state = self._state
            cancel_pending = (
                self._executor_cancel_hook_available
                and self._executor_cancel_outcome is None
            )
            if thread_exited and executing_request_id is None and not cancel_pending:
                self._session_state = POLICY_WORKER_SESSION_DRAINED
            elif executing_request_id is not None:
                self._session_state = POLICY_WORKER_SESSION_ABANDONED_UNDRAINED
            else:
                self._session_state = (
                    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION
                )
            session_state = self._session_state
        if thread_exited and executing_request_id is None and cancel_pending:
            status = "executor_cancel_pending"
        elif thread_exited and executing_request_id is None:
            status = "stopped"
        elif executing_request_id is not None:
            status = "executing_call_not_drained"
        else:
            status = "thread_exit_pending"
        if state == "faulted":
            status = "worker_faulted"
        return PolicyWorkerStopReceipt(
            worker_id=self.worker_id,
            worker_session_id=self.worker_session_id,
            stop_sequence=stop_sequence,
            requested_at=requested_at,
            returned_at=returned_at,
            thread_exited=thread_exited,
            active_request_id=active_request_id,
            executing_request_id=executing_request_id,
            discarded_queued_request_ids=discarded,
            pending_completion_count=self._completions.qsize(),
            status=status,
            previous_session_state=previous_session_state,
            session_state=session_state,
            executor_cancellation=executor_cancellation,
        )


__all__ = [
    "BoundedPolicyInferenceWorker",
    "POLICY_WORKER_COMPLETION_HASH_DOMAIN_V1",
    "POLICY_WORKER_COMPLETION_HASH_DOMAIN_V2",
    "POLICY_WORKER_SESSION_SNAPSHOT_HASH_DOMAIN_V1",
    "POLICY_WORKER_STOP_HASH_DOMAIN_V1",
    "POLICY_WORKER_STOP_HASH_DOMAIN_V2",
    "POLICY_WORKER_STOP_HASH_DOMAIN_V3",
    "POLICY_WORKER_SUBMISSION_HASH_DOMAIN_V1",
    "POLICY_WORKER_SUBMISSION_HASH_DOMAIN_V2",
    "POLICY_WORK_ITEM_HASH_DOMAIN_V1",
    "POLICY_WORK_ITEM_HASH_DOMAIN_V2",
    "POLICY_WORK_ITEM_HASH_DOMAIN_V3",
    "POLICY_WORK_ITEM_HASH_DOMAIN_V4",
    "POLICY_EXECUTOR_CANCELLATION_HASH_DOMAIN_V1",
    "POLICY_EXECUTOR_CANCELLATION_STATUSES_V1",
    "POLICY_EXECUTOR_CANCEL_DECLINED",
    "POLICY_EXECUTOR_CANCEL_FAILED",
    "POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED",
    "POLICY_EXECUTOR_CANCEL_PENDING",
    "POLICY_EXECUTOR_CANCEL_REQUESTED",
    "PolicyExecutorCancellationReceipt",
    "PolicyInferenceCancellationHook",
    "PolicyInferenceExecutor",
    "PolicyInferenceWorkItemExecutor",
    "PolicyInferenceWorkItem",
    "PolicyWorkerCompletion",
    "PolicyWorkerSessionSnapshot",
    "PolicyWorkerStopReceipt",
    "PolicyWorkerSubmissionReceipt",
]

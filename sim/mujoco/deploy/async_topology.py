"""Deterministic limited-async harness for the deployable MuJoCo topology.

This module deliberately models worker completion and receiver admission with a
bounded event queue instead of starting threads.  It is a causal scheduling
fixture for latency/jitter/restart/cutoff tests, not a production PolicyGraph,
RPC transport, or second control loop.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field, replace
from typing import Callable

from robot_motion.action import CompositionInputDraft, CompositionSpec
from robot_motion.action.proposal import freeze_robot_action
from robot_motion.deploy import PolicyContextDraft
from robot_motion.runtime.immutable import deep_freeze

from .runtime import MujocoDeployRuntime


ASYNC_JOB_KINDS_V1 = frozenset(
    {
        "policy_context",
        "composition_input",
        "compose",
        "decision",
        "producer_restart",
        "execution_gate",
    }
)


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


@dataclass(frozen=True)
class AsyncHarnessSubmissionReceipt:
    accepted: bool
    job_id: str
    queue_sequence: int | None
    rejection_code: str | None = None
    message: str = ""


@dataclass(frozen=True)
class AsyncHarnessRelease:
    job_id: str
    job_kind: str
    queue_sequence: int
    submitted_at: float
    completed_at: float
    deadline_at: float | None
    deadline_missed: bool
    status: str
    result: object | None = field(compare=False, repr=False)
    error: str | None = None


@dataclass(frozen=True)
class _ComposeOperation:
    spec: CompositionSpec
    target_effective_at: float
    required_context_channels: tuple[str, ...]
    policy_node_id: str


@dataclass(frozen=True)
class _DecisionOperation:
    physics_ticks_after_dispatch: int


@dataclass(frozen=True)
class _ProducerRestartOperation:
    producer_id: str
    producer_epoch: int
    reason: str


@dataclass(frozen=True)
class _ExecutionGateOperation:
    status: str
    reason: str
    source_event_ids: tuple[str, ...]


@dataclass(order=True, frozen=True)
class _ScheduledJob:
    completed_at: float
    queue_sequence: int
    job_id: str = field(compare=False)
    job_kind: str = field(compare=False)
    submitted_at: float = field(compare=False)
    deadline_at: float | None = field(compare=False)
    operation: object = field(compare=False, repr=False)


class DeterministicAsyncTopologyHarness:
    """Release immutable fake worker outputs at deterministic completion times."""

    def __init__(
        self,
        runtime: MujocoDeployRuntime,
        *,
        set_control_time: Callable[[float], None],
        start_at: float,
        capacity: int = 4096,
    ) -> None:
        if not isinstance(runtime, MujocoDeployRuntime):
            raise TypeError("runtime must be a MujocoDeployRuntime.")
        if not callable(set_control_time):
            raise TypeError("set_control_time must be callable.")
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive.")
        self.runtime = runtime
        self.set_control_time = set_control_time
        self.capacity = int(capacity)
        self.current_time = _finite(start_at, field_name="start_at")
        self._next_queue_sequence = 0
        self._jobs: list[_ScheduledJob] = []
        self._known_job_ids: set[str] = set()
        self._releases: list[AsyncHarnessRelease] = []
        self.set_control_time(self.current_time)

    @property
    def releases(self) -> tuple[AsyncHarnessRelease, ...]:
        return tuple(self._releases)

    @property
    def pending_count(self) -> int:
        return len(self._jobs)

    def _schedule(
        self,
        *,
        job_id: str,
        job_kind: str,
        operation: object,
        submitted_at: float,
        completed_at: float,
        deadline_at: float | None,
    ) -> AsyncHarnessSubmissionReceipt:
        normalized_id = _required(job_id, field_name="job_id")
        normalized_kind = _required(job_kind, field_name="job_kind")
        if normalized_kind not in ASYNC_JOB_KINDS_V1:
            raise ValueError(f"Unsupported async job kind {normalized_kind!r}.")
        submitted = _finite(submitted_at, field_name="submitted_at")
        completed = _finite(completed_at, field_name="completed_at")
        deadline = (
            None
            if deadline_at is None
            else _finite(deadline_at, field_name="deadline_at")
        )
        if completed < submitted:
            raise ValueError("completed_at cannot precede submitted_at.")
        if deadline is not None and deadline < submitted:
            raise ValueError("deadline_at cannot precede submitted_at.")
        if normalized_id in self._known_job_ids:
            return AsyncHarnessSubmissionReceipt(
                False,
                normalized_id,
                None,
                "duplicate_job_id",
                "job_id has already been scheduled.",
            )
        if len(self._jobs) >= self.capacity:
            return AsyncHarnessSubmissionReceipt(
                False,
                normalized_id,
                None,
                "capacity_exhausted",
                "async harness queue is full.",
            )
        queue_sequence = self._next_queue_sequence
        self._next_queue_sequence += 1
        heapq.heappush(
            self._jobs,
            _ScheduledJob(
                completed_at=completed,
                queue_sequence=queue_sequence,
                job_id=normalized_id,
                job_kind=normalized_kind,
                submitted_at=submitted,
                deadline_at=deadline,
                operation=operation,
            ),
        )
        self._known_job_ids.add(normalized_id)
        return AsyncHarnessSubmissionReceipt(True, normalized_id, queue_sequence)

    def schedule_policy_context(
        self,
        *,
        job_id: str,
        draft: PolicyContextDraft,
        submitted_at: float,
        completed_at: float,
        deadline_at: float | None = None,
    ) -> AsyncHarnessSubmissionReceipt:
        if not isinstance(draft, PolicyContextDraft):
            raise TypeError("draft must be a PolicyContextDraft.")
        frozen_draft = replace(
            draft,
            semantic_payload=deep_freeze(draft.semantic_payload),
            parent_context_ids=tuple(draft.parent_context_ids),
            blob_refs=tuple(draft.blob_refs),
            audit_metadata=deep_freeze(draft.audit_metadata),
        )
        return self._schedule(
            job_id=job_id,
            job_kind="policy_context",
            operation=frozen_draft,
            submitted_at=submitted_at,
            completed_at=completed_at,
            deadline_at=deadline_at,
        )

    def schedule_composition_input(
        self,
        *,
        job_id: str,
        draft: CompositionInputDraft,
        submitted_at: float,
        completed_at: float,
        deadline_at: float | None = None,
    ) -> AsyncHarnessSubmissionReceipt:
        if not isinstance(draft, CompositionInputDraft):
            raise TypeError("draft must be a CompositionInputDraft.")
        frozen_draft = replace(
            draft,
            action=freeze_robot_action(draft.action),
            context_ids=tuple(draft.context_ids),
            audit_metadata=deep_freeze(draft.audit_metadata),
        )
        return self._schedule(
            job_id=job_id,
            job_kind="composition_input",
            operation=frozen_draft,
            submitted_at=submitted_at,
            completed_at=completed_at,
            deadline_at=deadline_at,
        )

    def schedule_composition(
        self,
        *,
        job_id: str,
        spec: CompositionSpec,
        target_effective_at: float,
        submitted_at: float,
        completed_at: float,
        required_context_channels: tuple[str, ...] = (),
        policy_node_id: str = "fast-composer",
        deadline_at: float | None = None,
    ) -> AsyncHarnessSubmissionReceipt:
        if not isinstance(spec, CompositionSpec):
            raise TypeError("spec must be a CompositionSpec.")
        operation = _ComposeOperation(
            spec=spec,
            target_effective_at=_finite(
                target_effective_at, field_name="target_effective_at"
            ),
            required_context_channels=tuple(
                _required(value, field_name="required_context_channel")
                for value in required_context_channels
            ),
            policy_node_id=_required(policy_node_id, field_name="policy_node_id"),
        )
        return self._schedule(
            job_id=job_id,
            job_kind="compose",
            operation=operation,
            submitted_at=submitted_at,
            completed_at=completed_at,
            deadline_at=deadline_at,
        )

    def schedule_decision(
        self,
        *,
        job_id: str,
        decision_at: float,
        physics_ticks_after_dispatch: int = 0,
        deadline_at: float | None = None,
    ) -> AsyncHarnessSubmissionReceipt:
        ticks = int(physics_ticks_after_dispatch)
        if ticks < 0:
            raise ValueError("physics_ticks_after_dispatch must be non-negative.")
        return self._schedule(
            job_id=job_id,
            job_kind="decision",
            operation=_DecisionOperation(ticks),
            submitted_at=decision_at,
            completed_at=decision_at,
            deadline_at=deadline_at,
        )

    def schedule_producer_restart(
        self,
        *,
        job_id: str,
        producer_id: str,
        producer_epoch: int,
        submitted_at: float,
        completed_at: float,
        reason: str,
    ) -> AsyncHarnessSubmissionReceipt:
        operation = _ProducerRestartOperation(
            producer_id=_required(producer_id, field_name="producer_id"),
            producer_epoch=int(producer_epoch),
            reason=_required(reason, field_name="reason"),
        )
        if operation.producer_epoch < 0:
            raise ValueError("producer_epoch must be non-negative.")
        return self._schedule(
            job_id=job_id,
            job_kind="producer_restart",
            operation=operation,
            submitted_at=submitted_at,
            completed_at=completed_at,
            deadline_at=None,
        )

    def schedule_execution_gate(
        self,
        *,
        job_id: str,
        status: str,
        submitted_at: float,
        completed_at: float,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
    ) -> AsyncHarnessSubmissionReceipt:
        operation = _ExecutionGateOperation(
            status=_required(status, field_name="status"),
            reason=_required(reason, field_name="reason"),
            source_event_ids=tuple(
                _required(value, field_name="source_event_id")
                for value in source_event_ids
            ),
        )
        return self._schedule(
            job_id=job_id,
            job_kind="execution_gate",
            operation=operation,
            submitted_at=submitted_at,
            completed_at=completed_at,
            deadline_at=None,
        )

    def advance_to(self, target_time: float) -> tuple[AsyncHarnessRelease, ...]:
        target = _finite(target_time, field_name="target_time")
        if target < self.current_time:
            raise ValueError("async harness time cannot move backwards.")
        released: list[AsyncHarnessRelease] = []
        while self._jobs and self._jobs[0].completed_at <= target:
            job = heapq.heappop(self._jobs)
            self.current_time = job.completed_at
            self.set_control_time(self.current_time)
            release = self._release(job)
            self._releases.append(release)
            released.append(release)
        self.current_time = target
        self.set_control_time(target)
        return tuple(released)

    def run_all(self) -> tuple[AsyncHarnessRelease, ...]:
        if not self._jobs:
            return ()
        return self.advance_to(max(job.completed_at for job in self._jobs))

    def _release(self, job: _ScheduledJob) -> AsyncHarnessRelease:
        result: object | None = None
        status = "completed"
        error: str | None = None
        try:
            operation = job.operation
            if job.job_kind == "policy_context":
                result = self.runtime.submit_policy_context(operation)
                status = "admitted" if result.accepted else "rejected"
            elif job.job_kind == "composition_input":
                result = self.runtime.submit_composition_input(operation)
                status = "admitted" if result.accepted else "rejected"
            elif job.job_kind == "compose":
                result = self.runtime.compose_and_submit(
                    spec=operation.spec,
                    target_effective_at=operation.target_effective_at,
                    policy_node_id=operation.policy_node_id,
                    required_context_channels=operation.required_context_channels,
                )
                status = result.composition_result.output_status
            elif job.job_kind == "decision":
                result = self.runtime.step(
                    now=job.completed_at,
                    physics_ticks_after_dispatch=operation.physics_ticks_after_dispatch,
                )
                status = result.status
            elif job.job_kind == "producer_restart":
                self.runtime.activate_policy_node_producer(
                    operation.producer_id,
                    producer_epoch=operation.producer_epoch,
                    reason=operation.reason,
                )
                status = "activated"
            elif job.job_kind == "execution_gate":
                result = self.runtime.set_execution_gate(
                    operation.status,
                    reason=operation.reason,
                    source_event_ids=operation.source_event_ids,
                )
                status = operation.status
            else:  # pragma: no cover - guarded by _schedule
                raise AssertionError(f"Unhandled job kind {job.job_kind!r}.")
        except Exception as exc:  # intentional worker-isolation boundary
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        deadline_missed = (
            job.deadline_at is not None and job.completed_at > job.deadline_at
        )
        return AsyncHarnessRelease(
            job_id=job.job_id,
            job_kind=job.job_kind,
            queue_sequence=job.queue_sequence,
            submitted_at=job.submitted_at,
            completed_at=job.completed_at,
            deadline_at=job.deadline_at,
            deadline_missed=deadline_missed,
            status=status,
            result=result,
            error=error,
        )


__all__ = [
    "ASYNC_JOB_KINDS_V1",
    "AsyncHarnessRelease",
    "AsyncHarnessSubmissionReceipt",
    "DeterministicAsyncTopologyHarness",
]

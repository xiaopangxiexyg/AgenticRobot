"""Default-off Station worker port for allocation-context publication.

The reusable allocation runtime owns evaluation, exact retry identity, context
producer sequence, and receipt-driven reducer commit.  This application port
owns only a bounded wake-up worker and binds that runtime to one active Station
composition-session barrier.  It never polls a source, advances a cursor,
changes authority, publishes an action, or touches downstream execution.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
    ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
    ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
    ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED,
    ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
    ALLOCATION_PUBLICATION_OUTCOME_LATE,
    ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
    ALLOCATION_PUBLICATION_RUNTIME_PENDING,
    AllocationPublicationBasis,
    AllocationPublicationOutcome,
    AllocationPublicationRuntime,
    CompositionAllocationOverrideReceipt,
    CompositionAllocationOverrideRequest,
    CompositionAllocationOverrideState,
)
from control_composition.composition import (
    CompositionContextAdmissionReceipt,
    CompositionContextDraft,
    CompositionContextRecord,
)


_TERMINAL_ATTEMPT_OUTCOMES = frozenset(
    {
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_NEW,
        ALLOCATION_PUBLICATION_OUTCOME_ACCEPTED_EXISTING,
        ALLOCATION_PUBLICATION_OUTCOME_TERMINAL_REJECTION,
        ALLOCATION_PUBLICATION_OUTCOME_INVALIDATED,
        ALLOCATION_PUBLICATION_OUTCOME_LATE,
    }
)

from ..publication_barrier import StationPublicationBarrier


STATION_ALLOCATION_PUBLICATION_PROFILE_OFF = "off"
_ACTIVE_COMPOSITION_BARRIER_REASONS = frozenset(
    {
        "typed_composition_session_completion",
        "typed_composition_session_rearm_completion",
    }
)

ContextAdmitter = Callable[
    [CompositionContextDraft], CompositionContextAdmissionReceipt
]
ContextResolver = Callable[[str], CompositionContextRecord | None]


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _nonnegative_timeout(value: object, *, name: str) -> float:
    timeout = float(value)
    if not math.isfinite(timeout) or timeout < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return timeout


@dataclass(frozen=True)
class StationAllocationPublicationProfile:
    """Explicit switch for context publication; it can never send an action."""

    profile_id: str = STATION_ALLOCATION_PUBLICATION_PROFILE_OFF
    enabled: bool = False
    send_actions: bool = False
    outcome_cache_capacity: int = 32

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        if not isinstance(self.enabled, bool) or not isinstance(
            self.send_actions,
            bool,
        ):
            raise TypeError("enabled and send_actions must be booleans.")
        if self.send_actions:
            raise ValueError("Allocation context publication cannot send actions.")
        if self.enabled and profile_id == STATION_ALLOCATION_PUBLICATION_PROFILE_OFF:
            raise ValueError("An enabled allocation publication profile needs an id.")
        capacity = self.outcome_cache_capacity
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, int)
            or not 1 <= capacity <= 1024
        ):
            raise ValueError("outcome_cache_capacity must be in [1, 1024].")
        object.__setattr__(self, "profile_id", profile_id)


@dataclass(frozen=True)
class StationAllocationPublicationCycle:
    """Immediate receipt for one lazy Station-side request."""

    profile_id: str
    status: str
    outcome: AllocationPublicationOutcome | None = None
    worker_notified: bool = False

    def __post_init__(self) -> None:
        profile_id = _required(self.profile_id, name="profile_id")
        status = _required(self.status, name="status")
        if status not in {
            "disabled",
            "not_started",
            "closed",
            "basis_offered",
            "retry_requested",
            "retry_unavailable",
        }:
            raise ValueError("Unsupported allocation publication cycle status.")
        if status == "basis_offered":
            if not isinstance(self.outcome, AllocationPublicationOutcome):
                raise TypeError("A basis-offered cycle requires its typed outcome.")
        elif self.outcome is not None:
            raise ValueError("Only a basis-offered cycle carries an immediate outcome.")
        if self.worker_notified and status not in {
            "basis_offered",
            "retry_requested",
        }:
            raise ValueError("Only active publication work may notify the worker.")
        object.__setattr__(self, "profile_id", profile_id)
        object.__setattr__(self, "status", status)


@dataclass(frozen=True)
class StationAllocationPublicationSnapshot:
    profile_id: str
    enabled: bool
    running: bool
    closed: bool
    worker_alive: bool
    recorded_outcome_count: int
    dropped_outcome_count: int
    latest_outcome: AllocationPublicationOutcome | None
    allocation_runtime_status: str
    allocation_override_state: CompositionAllocationOverrideState | None = None


class StationAllocationPublicationPort:
    """Run exact context admission outside the application control loop."""

    def __init__(
        self,
        *,
        profile: StationAllocationPublicationProfile,
        allocation_runtime: AllocationPublicationRuntime | None = None,
        publication_barrier: StationPublicationBarrier | None = None,
        admit_context: ContextAdmitter | None = None,
        resolve_context: ContextResolver | None = None,
        worker_name: str = "station-allocation-publication",
    ) -> None:
        if not isinstance(profile, StationAllocationPublicationProfile):
            raise TypeError("profile must be StationAllocationPublicationProfile.")
        if profile.enabled:
            if not isinstance(allocation_runtime, AllocationPublicationRuntime):
                raise TypeError("Enabled publication requires an allocation runtime.")
            if not isinstance(publication_barrier, StationPublicationBarrier):
                raise TypeError("Enabled publication requires a Station barrier.")
            if not callable(admit_context):
                raise TypeError("Enabled publication requires a context admitter.")
            if resolve_context is not None and not callable(resolve_context):
                raise TypeError("resolve_context must be callable or None.")
            barrier_snapshot = publication_barrier.snapshot()
            if (
                not barrier_snapshot.enabled
                or barrier_snapshot.authority_operation_result_id is None
                or barrier_snapshot.transition_reason
                not in _ACTIVE_COMPOSITION_BARRIER_REASONS
            ):
                raise RuntimeError(
                    "Enabled allocation publication requires one active typed "
                    "composition-session barrier."
                )
            barrier_generation = barrier_snapshot.generation
            authority_result_id = barrier_snapshot.authority_operation_result_id
        else:
            barrier_generation = None
            authority_result_id = None
        self.profile = profile
        self.allocation_runtime = allocation_runtime
        self.publication_barrier = publication_barrier
        self._admit_context = admit_context
        self._resolve_context = resolve_context
        self._worker_name = _required(worker_name, name="worker_name")
        self._barrier_generation = barrier_generation
        self._authority_operation_result_id = authority_result_id
        self._condition = threading.Condition()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._outcomes: deque[AllocationPublicationOutcome] = deque(
            maxlen=profile.outcome_cache_capacity
        )
        self._latest_outcome: AllocationPublicationOutcome | None = None
        self._recorded_outcome_count = 0
        self._dropped_outcome_count = 0

    def _barrier_matches_activation(self) -> bool:
        barrier = self.publication_barrier
        if barrier is None:
            return False
        snapshot = barrier.snapshot()
        return bool(
            snapshot.enabled
            and snapshot.generation == self._barrier_generation
            and snapshot.authority_operation_result_id
            == self._authority_operation_result_id
            and snapshot.transition_reason in _ACTIVE_COMPOSITION_BARRIER_REASONS
        )

    def _record(self, outcome: AllocationPublicationOutcome) -> None:
        with self._condition:
            if len(self._outcomes) == self._outcomes.maxlen:
                self._dropped_outcome_count += 1
            self._outcomes.append(outcome)
            self._latest_outcome = outcome
            self._recorded_outcome_count += 1
            self._condition.notify_all()

    def snapshot(self) -> StationAllocationPublicationSnapshot:
        runtime = self.allocation_runtime
        with self._condition:
            worker = self._worker
            return StationAllocationPublicationSnapshot(
                profile_id=self.profile.profile_id,
                enabled=self.profile.enabled,
                running=self._running,
                closed=self._closed,
                worker_alive=bool(worker is not None and worker.is_alive()),
                recorded_outcome_count=self._recorded_outcome_count,
                dropped_outcome_count=self._dropped_outcome_count,
                latest_outcome=self._latest_outcome,
                allocation_runtime_status=(
                    "unbound" if runtime is None else runtime.runtime_status
                ),
                allocation_override_state=(
                    None if runtime is None else runtime.allocation_override_state
                ),
            )

    def drain_outcomes(self) -> tuple[AllocationPublicationOutcome, ...]:
        """Drain the bounded application cache; runtime state remains authoritative."""

        with self._condition:
            outcomes = tuple(self._outcomes)
            self._outcomes.clear()
            return outcomes

    @property
    def activation_active(self) -> bool:
        """Whether this port still belongs to one running Station barrier."""

        with self._condition:
            locally_active = self._running and not self._closed
        return bool(locally_active and self._barrier_matches_activation())

    def wait_for_outcomes(self, count: int, *, timeout_sec: float) -> bool:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer.")
        timeout = _nonnegative_timeout(timeout_sec, name="timeout_sec")
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._recorded_outcome_count < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def wait_for_terminal_attempt(
        self,
        attempt_id: str,
        *,
        timeout_sec: float,
    ) -> AllocationPublicationOutcome | None:
        """Wait for one exact worker attempt, never merely for queue progress."""

        normalized = _required(attempt_id, name="attempt_id")
        timeout = _nonnegative_timeout(timeout_sec, name="timeout_sec")
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                for outcome in reversed(self._outcomes):
                    if (
                        outcome.attempt_id == normalized
                        and outcome.outcome_kind in _TERMINAL_ATTEMPT_OUTCOMES
                    ):
                        return outcome
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(timeout=remaining)

    def start(self) -> StationAllocationPublicationSnapshot:
        if not self.profile.enabled:
            return self.snapshot()
        with self._condition:
            if self._closed:
                raise RuntimeError("Allocation publication port is closed.")
            if self._running:
                return self.snapshot()
            if not self._barrier_matches_activation():
                raise RuntimeError("Allocation publication barrier is no longer active.")
            self._running = True
            self._worker = threading.Thread(
                target=self._run,
                name=self._worker_name,
                daemon=True,
            )
            self._worker.start()
            return self.snapshot()

    def offer_basis(
        self,
        basis_factory: Callable[[], AllocationPublicationBasis],
    ) -> StationAllocationPublicationCycle:
        """Offer an already-produced basis without waiting for context admission."""

        if not callable(basis_factory):
            raise TypeError("basis_factory must be callable.")
        if not self.profile.enabled:
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        with self._condition:
            if self._closed:
                return StationAllocationPublicationCycle(
                    profile_id=self.profile.profile_id,
                    status="closed",
                )
            if not self._running:
                return StationAllocationPublicationCycle(
                    profile_id=self.profile.profile_id,
                    status="not_started",
                )
        if not self._barrier_matches_activation():
            self._fence(reason="station_composition_session_barrier_closed")
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="closed",
            )
        basis = basis_factory()
        if not isinstance(basis, AllocationPublicationBasis):
            raise TypeError("basis_factory must return AllocationPublicationBasis.")
        runtime = self.allocation_runtime
        assert runtime is not None
        outcome = runtime.offer_basis(basis)
        self._record(outcome)
        notify = outcome.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
        if notify:
            self._wake.set()
        return StationAllocationPublicationCycle(
            profile_id=self.profile.profile_id,
            status="basis_offered",
            outcome=outcome,
            worker_notified=notify,
        )

    def advance_anchor_generation(
        self,
        request_generation: int,
    ) -> AllocationPublicationOutcome:
        """Fence pending/coalesced work from an older residual anchor."""

        if not self.profile.enabled:
            raise RuntimeError("Allocation publication is disabled.")
        with self._condition:
            if self._closed:
                raise RuntimeError("Allocation publication port is closed.")
        if not self._barrier_matches_activation():
            self._fence(reason="station_composition_session_barrier_closed")
            raise RuntimeError("Allocation publication barrier is no longer active.")
        runtime = self.allocation_runtime
        assert runtime is not None
        outcome = runtime.advance_anchor_generation(request_generation)
        if outcome.outcome_kind != ALLOCATION_PUBLICATION_OUTCOME_GENERATION_FENCED:
            raise RuntimeError(
                "Allocation runtime could not accept the anchor generation fence."
            )
        self._record(outcome)
        return outcome

    def apply_override(
        self,
        request: CompositionAllocationOverrideRequest,
    ) -> CompositionAllocationOverrideReceipt:
        """Apply one typed persistent override behind the active barrier."""

        if not self.profile.enabled:
            raise RuntimeError("Allocation publication is disabled.")
        with self._condition:
            if self._closed or not self._running:
                raise RuntimeError("Allocation publication is not active.")
        if not self._barrier_matches_activation():
            self._fence(reason="station_composition_session_barrier_closed")
            raise RuntimeError("Allocation publication barrier is no longer active.")
        runtime = self.allocation_runtime
        assert runtime is not None
        return runtime.apply_override(request)

    def request_retry(self) -> StationAllocationPublicationCycle:
        """Request one exact retry; no new basis or source value is read."""

        if not self.profile.enabled:
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="disabled",
            )
        with self._condition:
            if self._closed:
                status = "closed"
            elif not self._running:
                status = "not_started"
            else:
                status = "retry_requested"
        if status != "retry_requested":
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status=status,
            )
        runtime = self.allocation_runtime
        assert runtime is not None
        if not self._barrier_matches_activation():
            self._fence(reason="station_composition_session_barrier_closed")
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="closed",
            )
        snapshot = runtime.snapshot()
        if (
            snapshot.runtime_status != ALLOCATION_PUBLICATION_RUNTIME_PENDING
            or snapshot.pending_attempt is None
            or snapshot.submission_in_progress
        ):
            return StationAllocationPublicationCycle(
                profile_id=self.profile.profile_id,
                status="retry_unavailable",
            )
        self._wake.set()
        return StationAllocationPublicationCycle(
            profile_id=self.profile.profile_id,
            status="retry_requested",
            worker_notified=True,
        )

    def invalidate(self, *, reason: str) -> AllocationPublicationOutcome | None:
        runtime = self.allocation_runtime
        if runtime is None:
            return None
        outcome = runtime.invalidate(reason=_required(reason, name="reason"))
        self._record(outcome)
        return outcome

    def _fence(self, *, reason: str) -> threading.Thread | None:
        """Fence without joining so control-facing checks never block."""

        with self._condition:
            newly_closed = not self._closed
            self._closed = True
            self._running = False
            worker = self._worker
        if newly_closed and self.profile.enabled:
            self.invalidate(reason=reason)
        self._stop.set()
        self._wake.set()
        return worker

    def close(
        self,
        *,
        reason: str = "station_allocation_publication_closed",
        timeout_sec: float = 2.0,
    ) -> bool:
        """Fence the activation before waiting for any in-flight callback."""

        timeout = _nonnegative_timeout(timeout_sec, name="timeout_sec")
        worker = self._fence(reason=reason)
        if worker is not None:
            worker.join(timeout=timeout)
        return bool(worker is None or not worker.is_alive())

    def _run(self) -> None:
        runtime = self.allocation_runtime
        assert runtime is not None
        assert self._admit_context is not None
        while True:
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                return
            if not self._barrier_matches_activation():
                self._fence(reason="station_composition_session_barrier_closed")
                return
            outcome = runtime.attempt_admission(
                self._admit_context,
                resolve_context=self._resolve_context,
            )
            if outcome is None:
                continue
            self._record(outcome)
            if runtime.runtime_status == ALLOCATION_PUBLICATION_RUNTIME_PENDING:
                # Unknown outcomes retain the exact attempt but never spin.
                continue
            prepared = runtime.prepare_latest_coalesced()
            if prepared is not None:
                self._record(prepared)
                if prepared.outcome_kind == (
                    ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
                ):
                    self._wake.set()


__all__ = [
    "STATION_ALLOCATION_PUBLICATION_PROFILE_OFF",
    "StationAllocationPublicationCycle",
    "StationAllocationPublicationPort",
    "StationAllocationPublicationProfile",
    "StationAllocationPublicationSnapshot",
]

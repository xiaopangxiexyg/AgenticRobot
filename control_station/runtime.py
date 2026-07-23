"""Entry-point-free CS-0b runtime composition.

The runtime serializes station intent/result admission.  It has no control
loop, action queue, source cursor, policy worker, dispatcher, or episode owner.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable

from .adapters import StationOperationAdapter
from .contracts import (
    ControlStationRunManifest,
    StationIntentDraft,
    StationOperationResultDraft,
)
from .state_machine import (
    ControlStationOperationTransition,
    ControlStationState,
    ControlStationTransition,
    admit_station_intent,
    admit_station_operation_result,
    apply_station_intent,
    apply_station_operation_result,
    initial_control_station_state,
)


@dataclass(frozen=True)
class StationOperationExecution:
    request_transition: ControlStationTransition
    completion_transition: ControlStationOperationTransition


class StationPendingOperationSuperseded(RuntimeError):
    """A safety or competing transition removed the requested operation."""


class ControlStationRuntime:
    """Serialize pure station facts without owning the data plane."""

    def __init__(
        self,
        *,
        manifest: ControlStationRunManifest,
        runtime_session_id: str,
        clock_domain_id: str = "control-monotonic",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(manifest, ControlStationRunManifest):
            raise TypeError("manifest must be a ControlStationRunManifest.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        self.manifest = manifest
        self.clock = clock
        self.clock_domain_id = str(clock_domain_id).strip()
        if not self.clock_domain_id:
            raise ValueError("clock_domain_id must be non-empty.")
        self._state = initial_control_station_state(
            manifest,
            runtime_session_id=runtime_session_id,
        )
        self._lock = threading.RLock()
        self._next_admission_sequence = 0

    @property
    def state(self) -> ControlStationState:
        with self._lock:
            return self._state

    @property
    def next_admission_sequence(self) -> int:
        with self._lock:
            return self._next_admission_sequence

    def _time(self) -> float:
        value = float(self.clock())
        if not math.isfinite(value):
            raise ValueError("Control Station receiver clock must be finite.")
        return value

    def submit_intent(self, draft: StationIntentDraft) -> ControlStationTransition:
        """Admit and reduce one operator request; never executes an adapter."""

        with self._lock:
            sequence = self._next_admission_sequence
            self._next_admission_sequence += 1
            intent = admit_station_intent(
                draft,
                runtime_session_id=self._state.runtime_session_id,
                control_domain_id=self._state.control_domain_id,
                available_at=self._time(),
                available_clock_domain_id=self.clock_domain_id,
                admission_sequence=sequence,
            )
            transition = apply_station_intent(self.manifest, self._state, intent)
            self._state = transition.state_after
            return transition

    def execute_pending_operation(
        self,
        request_transition: ControlStationTransition,
        adapter: StationOperationAdapter,
    ) -> StationOperationExecution:
        """Run one explicit no-I/O/test adapter and admit its typed result.

        Production adapters may execute asynchronously and should call a later
        result-admission method instead.  CS-0b deliberately exposes only this
        synchronous fixture seam.
        """

        if request_transition.receipt.status != "operation_requested":
            raise ValueError("Transition does not own a pending station operation.")
        # Never hold the station admission lock while an adapter performs
        # work.  Even CS-0b adapters are no-I/O; this ordering fixes the
        # boundary before later asynchronous adapters are introduced.
        draft = adapter.perform(request_transition.intent)
        completion = self.admit_operation_result(draft)
        return StationOperationExecution(
            request_transition=request_transition,
            completion_transition=completion,
        )

    def admit_operation_result(
        self,
        draft: StationOperationResultDraft,
    ) -> ControlStationOperationTransition:
        """Receiver-admit one adapter completion on the station timeline."""

        if not isinstance(draft, StationOperationResultDraft):
            raise TypeError("draft must be a StationOperationResultDraft.")
        with self._lock:
            return self._admit_operation_result_locked(draft)

    def finalize_pending_operation(
        self,
        request_transition: ControlStationTransition,
        finalizer: Callable[[], StationOperationResultDraft],
    ) -> ControlStationOperationTransition:
        """CAS one short lower finalization against the live pending request.

        Long-running acquisition must happen before this call.  The callable
        is reserved for the bounded final readiness/commit step that must not
        race a Station HOLD, ESTOP, stop, or competing operation.
        """

        if not isinstance(request_transition, ControlStationTransition):
            raise TypeError("request_transition must be a ControlStationTransition.")
        if request_transition.receipt.status != "operation_requested":
            raise ValueError("Transition does not own a pending station operation.")
        if not callable(finalizer):
            raise TypeError("finalizer must be callable.")
        intent = request_transition.intent
        with self._lock:
            matching = tuple(
                operation
                for operation in self._state.pending_operation_by_subject.values()
                if operation.source_intent_id == intent.intent_id
                and operation.operation_kind == intent.intent_kind
                and operation.subject_id == intent.subject_id
            )
            if (
                len(matching) != 1
                or self._state.lifecycle_state in {"hold", "fault", "stopped"}
            ):
                raise StationPendingOperationSuperseded(
                    "station_pending_operation_superseded"
                )
            draft = finalizer()
            if not isinstance(draft, StationOperationResultDraft):
                raise TypeError("finalizer must return StationOperationResultDraft.")
            if (
                draft.source_intent_id != intent.intent_id
                or draft.operation_kind != intent.intent_kind
                or draft.subject_id != intent.subject_id
            ):
                raise ValueError("finalizer_result_binding_mismatch")
            return self._admit_operation_result_locked(draft)

    def _admit_operation_result_locked(
        self,
        draft: StationOperationResultDraft,
    ) -> ControlStationOperationTransition:
        sequence = self._next_admission_sequence
        self._next_admission_sequence += 1
        result = admit_station_operation_result(
            draft,
            runtime_session_id=self._state.runtime_session_id,
            control_domain_id=self._state.control_domain_id,
            available_at=self._time(),
            available_clock_domain_id=self.clock_domain_id,
            admission_sequence=sequence,
        )
        completion = apply_station_operation_result(
            self.manifest,
            self._state,
            result,
        )
        self._state = completion.state_after
        return completion


__all__ = [
    "ControlStationRuntime",
    "StationOperationExecution",
    "StationPendingOperationSuperseded",
]

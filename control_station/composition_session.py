"""Lifecycle contracts and controller for one formal composition session.

The Station remains the application control plane and the composer remains the
only proposal owner.  Contributor slots are named here instead of being
misrepresented as parallel authority candidates in the generic Station source
state.  The controller owns only session identity/state and its publication
barrier; it owns no source, worker, queue, safety implementation, dispatcher,
backend, or episode lifecycle.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable

from control_composition.lifecycle.session import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    CompositionSessionRegistry,
    CompositionSessionTransition,
    advance_composition_session,
    initial_composition_session_state,
    validate_composition_session_rearm,
)

from .contracts import StationOperationResult
from .publication_barrier import (
    StationPublicationBarrier,
    StationPublicationBarrierObserverPort,
    StationPublicationBarrierSnapshot,
)


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


class CompositionSessionController(CompositionSessionRegistry):
    """Station adapter for the backend-neutral composition session registry.

    The inherited registry owns identity, epochs, and pure lifecycle state.
    This adapter binds Station operation results, the publication barrier, and
    lower HOLD/cancel ordering.
    """

    def __init__(
        self,
        *,
        composer_source_slot_id: str,
        contributor_source_slot_ids: tuple[str, ...],
        composition_spec_hash: str,
        authority_gate_spec_hash: str,
        execution_binding_hash: str,
        clock: Callable[[], float],
        clock_domain_id: str,
    ) -> None:
        super().__init__(
            composer_source_slot_id=composer_source_slot_id,
            contributor_source_slot_ids=contributor_source_slot_ids,
            composition_spec_hash=composition_spec_hash,
            authority_gate_spec_hash=authority_gate_spec_hash,
            execution_binding_hash=execution_binding_hash,
            clock=clock,
            clock_domain_id=clock_domain_id,
        )
        self.publication_barrier = StationPublicationBarrier(
            source_slot_id=self.composer_source_slot_id
        )
        self._barrier_observer_port = StationPublicationBarrierObserverPort(
            suppress_callback_errors=False
        )
        self._observer_failures: deque[tuple[str, str, str]] = deque(maxlen=128)
        self._observer_failure_count = 0
        self._observer_context = threading.local()

    @property
    def observer_failure_count(self) -> int:
        with self._lock:
            return self._observer_failure_count

    @property
    def observer_failures(self) -> tuple[tuple[str, str, str], ...]:
        with self._lock:
            return tuple(self._observer_failures)

    def _diagnostic_observer(
        self,
        observer_kind: str,
        observer: Callable[..., object],
    ) -> Callable[..., object]:
        if not callable(observer):
            raise TypeError(f"{observer_kind} observer must be callable.")

        def observe(*args: object, **kwargs: object) -> object | None:
            previous_depth = int(getattr(self._observer_context, "depth", 0))
            self._observer_context.depth = previous_depth + 1
            try:
                return observer(*args, **kwargs)
            except Exception as error:  # noqa: BLE001 - passive observer boundary
                with self._lock:
                    self._observer_failure_count += 1
                    self._observer_failures.append(
                        (
                            observer_kind,
                            type(error).__name__,
                            str(error),
                        )
                    )
                return None
            finally:
                self._observer_context.depth = previous_depth

        return observe

    def _require_authoritative_mutation_context(self) -> None:
        if int(getattr(self._observer_context, "depth", 0)) > 0:
            raise RuntimeError(
                "Session/publication mutation is forbidden from a passive observer."
            )
        if int(getattr(self._observer_context, "commit_depth", 0)) > 0:
            raise RuntimeError(
                "Session/publication mutation is forbidden from a passive "
                "commit observer."
            )

    def bind_session_trace_observers(
        self,
        *,
        spec_observer: Callable[[CompositionSessionOperationSpec], object],
        transition_observer: Callable[..., object],
        rearm_spec_observer: (
            Callable[[CompositionSessionRearmSpec], object] | None
        ) = None,
        rearm_receipt_observer: (
            Callable[[CompositionSessionRearmReceipt], object] | None
        ) = None,
    ) -> None:
        """Bind passive trace observers without transferring lifecycle authority.

        A trace sink can report health through its own typed control-plane path,
        but its callback exception cannot split an already-admitted Station
        operation from the session/publication transaction.
        """

        super().bind_session_trace_observers(
            spec_observer=self._diagnostic_observer(
                "session_spec",
                spec_observer,
            ),
            transition_observer=self._diagnostic_observer(
                "session_transition",
                transition_observer,
            ),
            rearm_spec_observer=(
                None
                if rearm_spec_observer is None
                else self._diagnostic_observer(
                    "session_rearm_spec",
                    rearm_spec_observer,
                )
            ),
            rearm_receipt_observer=(
                None
                if rearm_receipt_observer is None
                else self._diagnostic_observer(
                    "session_rearm_receipt",
                    rearm_receipt_observer,
                )
            ),
        )

    def bind_publication_barrier_observer(
        self,
        observer: Callable[..., object],
    ) -> None:
        with self._lock:
            self._barrier_observer_port.bind(
                self._diagnostic_observer(
                    "publication_barrier",
                    observer,
                )
            )

    def register(self, spec: CompositionSessionOperationSpec) -> str:
        self._require_authoritative_mutation_context()
        return super().register(spec)

    def register_rearm(self, spec: CompositionSessionRearmSpec) -> str:
        self._require_authoritative_mutation_context()
        return super().register_rearm(spec)

    def advance(
        self,
        *,
        transition: str,
        reason: str,
        source_fact_id: str,
        source_fact_hash: str,
        start_intent_id: str | None = None,
    ) -> CompositionSessionTransition:
        with self._lock:
            self._require_authoritative_mutation_context()
            return super().advance(
                transition=transition,
                reason=reason,
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                start_intent_id=start_intent_id,
            )

    def observe_publication_barrier(
        self,
        snapshot: StationPublicationBarrierSnapshot,
        *,
        causal_station_fact_id: str,
        occurred_at: float,
        clock_domain_id: str,
    ) -> None:
        if not isinstance(snapshot, StationPublicationBarrierSnapshot):
            raise TypeError("snapshot must be a StationPublicationBarrierSnapshot.")
        if snapshot.source_slot_id != self.composer_source_slot_id:
            raise ValueError("Publication barrier source slot mismatch.")
        self._barrier_observer_port.observe(
            snapshot,
            causal_station_fact_id=causal_station_fact_id,
            occurred_at=occurred_at,
            clock_domain_id=clock_domain_id,
        )

    def disable_publication(
        self,
        *,
        reason: str,
        causal_station_fact_id: str | None = None,
        occurred_at: float | None = None,
        clock_domain_id: str | None = None,
    ) -> StationPublicationBarrierSnapshot:
        with self._lock:
            self._require_authoritative_mutation_context()
            observation_fields = (
                causal_station_fact_id,
                occurred_at,
                clock_domain_id,
            )
            if any(value is not None for value in observation_fields) and any(
                value is None for value in observation_fields
            ):
                raise ValueError(
                    "Publication barrier observation metadata is incomplete."
                )
            snapshot = self.publication_barrier.disable(reason=reason)
            if any(value is not None for value in observation_fields):
                self.observe_publication_barrier(
                    snapshot,
                    causal_station_fact_id=str(causal_station_fact_id),
                    occurred_at=float(occurred_at),
                    clock_domain_id=str(clock_domain_id),
                )
            return snapshot

    def activate_from_operation_result(
        self,
        *,
        operation_spec_id: str | None,
        result: StationOperationResult,
        source_fact_id: str,
        source_fact_hash: str,
        start_intent_id: str,
        before_publication: Callable[[], object] | None = None,
    ) -> CompositionSessionTransition:
        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        if before_publication is not None and not callable(before_publication):
            raise TypeError("before_publication must be callable.")
        with self._lock:
            self._require_authoritative_mutation_context()
            spec = self.require_registered_spec(operation_spec_id)
            state_before = self._state_by_spec_id[spec.operation_spec_id]
            state_after, transition = advance_composition_session(
                state_before,
                transition="start",
                reason="station_composition_session_ready",
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                start_intent_id=start_intent_id,
            )
            transition_occurred_at = float(self.clock())

            def commit(snapshot: StationPublicationBarrierSnapshot):
                consumed_before = spec.operation_spec_id in self._consumed_spec_ids
                previous_commit_depth = int(
                    getattr(self._observer_context, "commit_depth", 0)
                )
                self._observer_context.commit_depth = previous_commit_depth + 1
                try:
                    if before_publication is not None:
                        before_publication()
                    self._state_by_spec_id[spec.operation_spec_id] = state_after
                    self._consumed_spec_ids.add(spec.operation_spec_id)
                    observer = self._transition_observer
                    if observer is not None:
                        observer(
                            transition,
                            occurred_at=transition_occurred_at,
                            clock_domain_id=self.clock_domain_id,
                        )
                    self.observe_publication_barrier(
                        snapshot,
                        causal_station_fact_id=result.operation_result_id,
                        occurred_at=result.available_at,
                        clock_domain_id=result.available_clock_domain_id,
                    )
                except BaseException:
                    self._state_by_spec_id[spec.operation_spec_id] = state_before
                    if not consumed_before:
                        self._consumed_spec_ids.discard(spec.operation_spec_id)
                    raise
                finally:
                    self._observer_context.commit_depth = previous_commit_depth
                return transition

            return self.publication_barrier.linearize_composition_session_enable(
                result,
                commit,
            )

    def prepare_safety_transition(
        self,
        *,
        intent_kind: str,
        causal_station_fact_id: str,
        source_fact_id: str,
        source_fact_hash: str,
        occurred_at: float,
        clock_domain_id: str,
        lower_hold_or_fault: Callable[[], object] | None = None,
    ) -> CompositionSessionTransition | None:
        """Close publication before a Station safety/lifecycle transition."""

        kind = _required(intent_kind, field_name="intent_kind")
        if kind not in {
            "stop_composition_session",
            "hold",
            "estop",
            "disconnect_target",
        }:
            raise ValueError("Intent does not close composition publication.")
        with self._lock:
            self._require_authoritative_mutation_context()
            self.disable_publication(
                reason=f"station_intent_admitted:{kind}",
                causal_station_fact_id=causal_station_fact_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            )
            state = self.current_state
            if state is None:
                return None
            if state.phase == "registered":
                if kind in {"hold", "estop", "disconnect_target"}:
                    return self.advance(
                        transition="start_failed",
                        reason=f"station_{kind}_before_session_start",
                        source_fact_id=source_fact_id,
                        source_fact_hash=source_fact_hash,
                    )
                return None
            if state.phase != "active":
                return None
            if lower_hold_or_fault is None:
                raise RuntimeError(
                    "Active composition safety transition requires lower HOLD/fault."
                )
            lower_hold_or_fault()
            return self.advance(
                transition="fault" if kind == "estop" else "hold",
                reason=f"station_{kind}",
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
            )

    def complete_start_operation(
        self,
        *,
        operation_spec_id: str | None,
        result: StationOperationResult,
        completion_accepted: bool,
        source_fact_id: str,
        source_fact_hash: str,
        start_intent_id: str,
        before_publication: Callable[[], object] | None = None,
    ) -> CompositionSessionTransition | None:
        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        self._require_authoritative_mutation_context()
        if completion_accepted and result.succeeded:
            return self.activate_from_operation_result(
                operation_spec_id=operation_spec_id,
                result=result,
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                start_intent_id=start_intent_id,
                before_publication=before_publication,
            )
        self.disable_publication(
            reason="composition_session_start_failed",
            causal_station_fact_id=result.operation_result_id,
            occurred_at=result.available_at,
            clock_domain_id=result.available_clock_domain_id,
        )
        state = self.current_state
        if state is None or state.phase != "registered":
            return None
        return self.advance(
            transition="start_failed",
            reason=result.status,
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
        )

    def complete_rearm_operation(
        self,
        *,
        rearm_spec_id: str | None,
        receipt: CompositionSessionRearmReceipt | None,
        result: StationOperationResult,
        completion_accepted: bool,
        source_fact_id: str,
        source_fact_hash: str,
        rearm_intent_id: str,
    ) -> tuple[CompositionSessionTransition, CompositionSessionTransition] | None:
        """Retire one held activation and start a fresh activation atomically."""

        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        self._require_authoritative_mutation_context()
        if not completion_accepted or not result.succeeded:
            self.disable_publication(
                reason="composition_session_rearm_failed",
                causal_station_fact_id=result.operation_result_id,
                occurred_at=result.available_at,
                clock_domain_id=result.available_clock_domain_id,
            )
            return None
        if receipt is None:
            raise ValueError("Successful composition rearm requires a typed receipt.")
        with self._lock:
            spec = self.require_rearm_spec(rearm_spec_id)
            assert self._current_spec_id is not None
            logical_spec = self._spec_by_id[self._current_spec_id]
            held_state = self._state_by_spec_id[self._current_spec_id]
            validate_composition_session_rearm(
                spec,
                receipt,
                logical_spec=logical_spec,
                held_state=held_state,
            )
            required_evidence = {
                spec.rearm_spec_id,
                spec.semantic_content_hash,
                receipt.receipt_id,
                receipt.semantic_content_hash,
            }
            if (
                result.operation_kind != "rearm_composition_session"
                or result.status != receipt.status
                or not required_evidence.issubset(set(result.evidence_refs))
            ):
                raise ValueError(
                    "Station rearm result does not carry exact typed evidence."
                )
            _retired_state, retired = advance_composition_session(
                held_state,
                transition="supersede",
                reason="typed_rearm_new_activation",
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
            )
            registered_state = initial_composition_session_state(
                logical_spec,
                session_epoch=receipt.next_session_epoch,
            )
            active_state, started = advance_composition_session(
                registered_state,
                transition="start",
                reason="station_composition_session_rearmed",
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                start_intent_id=rearm_intent_id,
            )
            next_epoch_before = self._next_session_epoch
            retired_occurred_at = float(self.clock())
            started_occurred_at = float(self.clock())

            def commit(snapshot: StationPublicationBarrierSnapshot):
                consumed_before = spec.rearm_spec_id in self._consumed_rearm_spec_ids
                previous_commit_depth = int(
                    getattr(self._observer_context, "commit_depth", 0)
                )
                self._observer_context.commit_depth = previous_commit_depth + 1
                try:
                    assert self._current_spec_id is not None
                    self._state_by_spec_id[self._current_spec_id] = active_state
                    self._next_session_epoch = max(
                        self._next_session_epoch,
                        receipt.next_session_epoch + 1,
                    )
                    self._consumed_rearm_spec_ids.add(spec.rearm_spec_id)
                    receipt_observer = self._rearm_receipt_observer
                    if receipt_observer is not None:
                        receipt_observer(receipt)
                    transition_observer = self._transition_observer
                    if transition_observer is not None:
                        transition_observer(
                            retired,
                            occurred_at=retired_occurred_at,
                            clock_domain_id=self.clock_domain_id,
                        )
                        transition_observer(
                            started,
                            occurred_at=started_occurred_at,
                            clock_domain_id=self.clock_domain_id,
                        )
                    self.observe_publication_barrier(
                        snapshot,
                        causal_station_fact_id=result.operation_result_id,
                        occurred_at=result.available_at,
                        clock_domain_id=result.available_clock_domain_id,
                    )
                except BaseException:
                    assert self._current_spec_id is not None
                    self._state_by_spec_id[self._current_spec_id] = held_state
                    self._next_session_epoch = next_epoch_before
                    if not consumed_before:
                        self._consumed_rearm_spec_ids.discard(spec.rearm_spec_id)
                    raise
                finally:
                    self._observer_context.commit_depth = previous_commit_depth
                return retired, started

            return self.publication_barrier.linearize_composition_session_rearm_enable(
                result,
                commit,
            )

    def complete_stop_operation(
        self,
        *,
        result: StationOperationResult,
        completion_accepted: bool,
        source_fact_id: str,
        source_fact_hash: str,
    ) -> CompositionSessionTransition | None:
        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        self._require_authoritative_mutation_context()
        if not completion_accepted:
            return None
        state = self.current_state
        if state is None or state.phase not in {"active", "held"}:
            return None
        return self.advance(
            transition="stop" if result.succeeded else "fault",
            reason=result.status,
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
        )

    def require_active(
        self,
        *,
        station_lifecycle_state: str,
        composer_runtime_state: str,
    ) -> None:
        state = self.current_state
        if (
            state is None
            or state.phase != "active"
            or station_lifecycle_state != "running"
            or composer_runtime_state != "active"
        ):
            raise RuntimeError("Composition session is not active.")

    def close(self) -> None:
        self.disable_publication(reason="composition_session_closed")


__all__ = [
    "CompositionSessionController",
]

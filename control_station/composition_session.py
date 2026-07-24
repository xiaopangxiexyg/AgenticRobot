"""Lifecycle contracts and controller for one formal composition session.

The Station remains the application control plane and the composer remains the
only proposal owner.  Contributor slots are named here instead of being
misrepresented as parallel authority candidates in the generic Station source
state.  The controller owns only session identity/state and its publication
barrier; it owns no source, worker, queue, safety implementation, dispatcher,
backend, or episode lifecycle.
"""

from __future__ import annotations

from typing import Callable

from control_composition.lifecycle.session import (
    COMPOSITION_SESSION_PHASES_V1,
    COMPOSITION_SESSION_REARM_RECEIPT_HASH_DOMAIN_V1,
    COMPOSITION_SESSION_REARM_SPEC_HASH_DOMAIN_V1,
    COMPOSITION_SESSION_SPEC_HASH_DOMAIN_V1,
    COMPOSITION_SESSION_STATE_HASH_DOMAIN_V1,
    COMPOSITION_SESSION_TRANSITION_HASH_DOMAIN_V1,
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    CompositionSessionRegistry,
    CompositionSessionState,
    CompositionSessionTransition,
    advance_composition_session,
    initial_composition_session_state,
    validate_composition_session_rearm,
)

from .contracts import StationOperationResult
from .publication_barrier import (
    StationPublicationBarrier,
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
        self._barrier_observer: Callable[..., object] | None = None
        self._observed_barrier_keys: set[tuple[str, str]] = set()

    def bind_publication_barrier_observer(
        self,
        observer: Callable[..., object],
    ) -> None:
        if not callable(observer):
            raise TypeError("Publication barrier observer must be callable.")
        with self._lock:
            if self._barrier_observer is not None:
                raise RuntimeError(
                    "Publication barrier observer is already bound."
                )
            self._barrier_observer = observer

    def observe_publication_barrier(
        self,
        snapshot: StationPublicationBarrierSnapshot,
        *,
        causal_station_fact_id: str,
        occurred_at: float,
        clock_domain_id: str,
    ) -> None:
        if not isinstance(snapshot, StationPublicationBarrierSnapshot):
            raise TypeError(
                "snapshot must be a StationPublicationBarrierSnapshot."
            )
        if snapshot.source_slot_id != self.composer_source_slot_id:
            raise ValueError("Publication barrier source slot mismatch.")
        observer = self._barrier_observer
        if observer is None:
            return
        key = (snapshot.semantic_content_hash, str(causal_station_fact_id))
        with self._lock:
            if key in self._observed_barrier_keys:
                return
            observer(
                snapshot,
                causal_station_fact_id=causal_station_fact_id,
                occurred_at=occurred_at,
                clock_domain_id=clock_domain_id,
            )
            self._observed_barrier_keys.add(key)

    def disable_publication(
        self,
        *,
        reason: str,
        causal_station_fact_id: str | None = None,
        occurred_at: float | None = None,
        clock_domain_id: str | None = None,
    ) -> StationPublicationBarrierSnapshot:
        observation_fields = (
            causal_station_fact_id,
            occurred_at,
            clock_domain_id,
        )
        if any(value is not None for value in observation_fields):
            if any(value is None for value in observation_fields):
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
    ) -> CompositionSessionTransition:
        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        spec = self.require_registered_spec(operation_spec_id)
        snapshot = (
            self.publication_barrier.enable_from_composition_session_result(
                result
            )
        )
        with self._lock:
            self._consumed_spec_ids.add(spec.operation_spec_id)
        self.observe_publication_barrier(
            snapshot,
            causal_station_fact_id=result.operation_result_id,
            occurred_at=result.available_at,
            clock_domain_id=result.available_clock_domain_id,
        )
        return self.advance(
            transition="start",
            reason="station_composition_session_ready",
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
            start_intent_id=start_intent_id,
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
    ) -> CompositionSessionTransition | None:
        if not isinstance(result, StationOperationResult):
            raise TypeError("result must be a StationOperationResult.")
        if completion_accepted and result.succeeded:
            return self.activate_from_operation_result(
                operation_spec_id=operation_spec_id,
                result=result,
                source_fact_id=source_fact_id,
                source_fact_hash=source_fact_hash,
                start_intent_id=start_intent_id,
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
        spec = self.require_rearm_spec(rearm_spec_id)
        with self._lock:
            assert self._current_spec_id is not None
            logical_spec = self._spec_by_id[self._current_spec_id]
            held_state = self._state_by_spec_id[self._current_spec_id]
        validate_composition_session_rearm(
            spec,
            receipt,
            logical_spec=logical_spec,
            held_state=held_state,
        )
        receipt_observer = self._rearm_receipt_observer
        if receipt_observer is not None:
            receipt_observer(receipt)
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
            raise ValueError("Station rearm result does not carry exact typed evidence.")
        retired = self.advance(
            transition="supersede",
            reason="typed_rearm_new_activation",
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
        )
        with self._lock:
            assert self._current_spec_id is not None
            next_state = initial_composition_session_state(
                logical_spec,
                session_epoch=receipt.next_session_epoch,
            )
            self._state_by_spec_id[self._current_spec_id] = next_state
            self._next_session_epoch = max(
                self._next_session_epoch,
                receipt.next_session_epoch + 1,
            )
            self._consumed_rearm_spec_ids.add(spec.rearm_spec_id)
        started = self.advance(
            transition="start",
            reason="station_composition_session_rearmed",
            source_fact_id=source_fact_id,
            source_fact_hash=source_fact_hash,
            start_intent_id=rearm_intent_id,
        )
        snapshot = (
            self.publication_barrier.enable_from_composition_session_rearm_result(
                result
            )
        )
        self.observe_publication_barrier(
            snapshot,
            causal_station_fact_id=result.operation_result_id,
            occurred_at=result.available_at,
            clock_domain_id=result.available_clock_domain_id,
        )
        return retired, started

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
    "COMPOSITION_SESSION_PHASES_V1",
    "COMPOSITION_SESSION_REARM_RECEIPT_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_REARM_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_STATE_HASH_DOMAIN_V1",
    "COMPOSITION_SESSION_TRANSITION_HASH_DOMAIN_V1",
    "CompositionSessionOperationSpec",
    "CompositionSessionController",
    "CompositionSessionRearmReceipt",
    "CompositionSessionRearmSpec",
    "CompositionSessionState",
    "CompositionSessionTransition",
    "advance_composition_session",
    "initial_composition_session_state",
    "validate_composition_session_rearm",
]

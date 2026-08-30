from __future__ import annotations

import threading

import pytest

from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
)
from control_station import (
    StationOperationResultDraft,
    admit_station_operation_result,
)
from control_station.composition_session import CompositionSessionController

pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _spec() -> CompositionSessionOperationSpec:
    return CompositionSessionOperationSpec(
        session_key="transaction-session",
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("human", "policy"),
        composition_spec_hash="composition:transaction",
        authority_gate_spec_hash="gate:transaction",
        execution_binding_hash="binding:transaction",
        capture_boundary_id="capture:transaction",
    )


def _controller(clock: _Clock) -> CompositionSessionController:
    return CompositionSessionController(
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("human", "policy"),
        composition_spec_hash="composition:transaction",
        authority_gate_spec_hash="gate:transaction",
        execution_binding_hash="binding:transaction",
        clock=clock,
        clock_domain_id="control.monotonic",
    )


def _result(
    *,
    operation_kind: str,
    status: str,
    evidence_refs: tuple[str, ...],
    sequence: int,
):
    return admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id=f"intent:{operation_kind}:{sequence}",
            operation_kind=operation_kind,
            subject_id="composer",
            succeeded=True,
            completed_at=10.0,
            completed_clock_domain_id="control.monotonic",
            status=status,
            evidence_refs=evidence_refs,
        ),
        runtime_session_id="runtime:transaction",
        control_domain_id="domain:transaction",
        available_at=10.0,
        available_clock_domain_id="control.monotonic",
        admission_sequence=sequence,
    )


def _start(
    controller: CompositionSessionController,
    spec: CompositionSessionOperationSpec,
    *,
    before_publication=None,
):
    result = _result(
        operation_kind="start_composition_session",
        status="composition_session_ready",
        evidence_refs=(spec.operation_spec_id,),
        sequence=1,
    )
    transition = controller.complete_start_operation(
        operation_spec_id=spec.operation_spec_id,
        result=result,
        completion_accepted=True,
        source_fact_id="station-operation:start",
        source_fact_hash="station-operation-hash:start",
        start_intent_id="intent:start",
        before_publication=before_publication,
    )
    assert transition is not None
    return result, transition


def test_start_prepublication_failure_restores_exact_retryable_state() -> None:
    clock = _Clock()
    controller = _controller(clock)
    spec = _spec()
    controller.register(spec)
    barrier_before = controller.publication_barrier.snapshot()
    calls = 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("synthetic activation trace failure")

    with pytest.raises(RuntimeError, match="activation trace failure"):
        _start(controller, spec, before_publication=fail_once)

    assert controller.current_state is not None
    assert controller.current_state.phase == "registered"
    assert controller.publication_barrier.snapshot() == barrier_before
    assert controller.require_registered_spec(spec.operation_spec_id) == spec

    _start(controller, spec, before_publication=fail_once)
    assert calls == 2
    assert controller.current_state is not None
    assert controller.current_state.phase == "active"
    assert controller.publication_barrier.snapshot().enabled


def test_start_hides_enabled_barrier_until_session_commit_finishes() -> None:
    clock = _Clock()
    controller = _controller(clock)
    spec = _spec()
    controller.register(spec)
    attempted = threading.Event()
    published = threading.Event()
    worker: threading.Thread | None = None

    def publish() -> None:
        attempted.set()
        controller.publication_barrier.execute_if_enabled(published.set)

    def before_publication() -> None:
        nonlocal worker
        worker = threading.Thread(target=publish, daemon=True)
        worker.start()
        assert attempted.wait(timeout=0.5)
        assert not published.wait(timeout=0.02)
        assert controller.current_state is not None
        assert controller.current_state.phase == "registered"

    _start(controller, spec, before_publication=before_publication)

    assert published.wait(timeout=0.5)
    assert worker is not None
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert controller.current_state is not None
    assert controller.current_state.phase == "active"


def test_start_serializes_a_concurrent_safety_transition() -> None:
    clock = _Clock()
    controller = _controller(clock)
    spec = _spec()
    controller.register(spec)
    attempted = threading.Event()
    finished = threading.Event()
    lower_calls: list[str] = []
    transitions = []
    worker: threading.Thread | None = None

    def hold() -> None:
        attempted.set()
        transitions.append(
            controller.prepare_safety_transition(
                intent_kind="hold",
                causal_station_fact_id="intent:concurrent-hold",
                source_fact_id="station-intent:concurrent-hold",
                source_fact_hash="station-intent-hash:concurrent-hold",
                occurred_at=10.0,
                clock_domain_id="control.monotonic",
                lower_hold_or_fault=lambda: lower_calls.append("hold"),
            )
        )
        finished.set()

    def before_publication() -> None:
        nonlocal worker
        worker = threading.Thread(target=hold, daemon=True)
        worker.start()
        assert attempted.wait(timeout=0.5)
        assert not finished.wait(timeout=0.02)

    _start(controller, spec, before_publication=before_publication)

    assert finished.wait(timeout=0.5)
    assert worker is not None
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert lower_calls == ["hold"]
    assert len(transitions) == 1
    assert transitions[0] is not None
    assert transitions[0].transition == "hold"
    assert controller.current_state is not None
    assert controller.current_state.phase == "held"
    assert not controller.publication_barrier.snapshot().enabled


def test_spec_observer_cannot_advance_a_registered_session() -> None:
    clock = _Clock()
    controller = _controller(clock)
    spec = _spec()

    def observe_spec(_spec) -> None:
        controller.advance(
            transition="start",
            reason="observer_must_not_start_session",
            source_fact_id="observer:spec",
            source_fact_hash="observer-spec-hash",
            start_intent_id="observer:start",
        )

    controller.bind_session_trace_observers(
        spec_observer=observe_spec,
        transition_observer=lambda _transition, **_metadata: None,
    )

    assert controller.register(spec) == spec.operation_spec_id
    assert controller.current_state is not None
    assert controller.current_state.phase == "registered"
    assert not controller.publication_barrier.snapshot().enabled
    assert controller.observer_failure_count == 1
    assert controller.observer_failures[0][0] == "session_spec"
    assert "passive observer" in controller.observer_failures[0][2]


def test_passive_observers_cannot_reenter_session_or_publication_mutation() -> None:
    clock = _Clock()
    controller = _controller(clock)
    spec = _spec()

    def observe_transition(transition, **_metadata) -> None:
        if transition.transition == "start":
            controller.advance(
                transition="hold",
                reason="observer_must_not_control_lifecycle",
                source_fact_id="observer:transition",
                source_fact_hash="observer-transition-hash",
            )

    def observe_barrier(snapshot, **_metadata) -> None:
        if snapshot.enabled:
            controller.disable_publication(
                reason="observer_must_not_control_publication"
            )

    controller.bind_session_trace_observers(
        spec_observer=lambda _spec: None,
        transition_observer=observe_transition,
    )
    controller.bind_publication_barrier_observer(observe_barrier)
    controller.register(spec)

    _start(controller, spec)

    assert controller.current_state is not None
    assert controller.current_state.phase == "active"
    assert controller.publication_barrier.snapshot().enabled
    assert controller.observer_failure_count == 2
    assert {failure[0] for failure in controller.observer_failures} == {
        "session_transition",
        "publication_barrier",
    }
    assert all("passive" in failure[2] for failure in controller.observer_failures)


def test_passive_observer_failures_cannot_split_start_or_rearm_commit() -> None:
    clock = _Clock()
    controller = _controller(clock)
    spec = _spec()

    def observe_transition(transition, **_metadata) -> None:
        if transition.transition in {"start", "supersede"}:
            raise RuntimeError(f"trace unavailable:{transition.transition}")

    def observe_barrier(snapshot, **_metadata) -> None:
        if snapshot.enabled:
            raise RuntimeError("barrier trace unavailable")

    controller.bind_session_trace_observers(
        spec_observer=lambda _spec: None,
        transition_observer=observe_transition,
        rearm_spec_observer=lambda _spec: None,
        rearm_receipt_observer=lambda _receipt: None,
    )
    controller.bind_publication_barrier_observer(observe_barrier)
    controller.register(spec)
    _start(controller, spec)

    assert controller.current_state is not None
    assert controller.current_state.phase == "active"
    assert controller.publication_barrier.snapshot().enabled

    controller.prepare_safety_transition(
        intent_kind="hold",
        causal_station_fact_id="intent:hold",
        source_fact_id="station-intent:hold",
        source_fact_hash="station-intent-hash:hold",
        occurred_at=10.0,
        clock_domain_id="control.monotonic",
        lower_hold_or_fault=lambda: None,
    )
    held = controller.current_state
    assert held is not None and held.phase == "held"
    rearm_spec = CompositionSessionRearmSpec(
        rearm_key="transaction-rearm",
        logical_operation_spec_id=spec.operation_spec_id,
        logical_operation_spec_hash=spec.semantic_content_hash,
        prior_session_epoch=held.session_epoch,
        prior_state_hash=held.semantic_content_hash,
        execution_binding_hash=spec.execution_binding_hash,
        readiness_observer_binding_id="readiness:transaction",
        readiness_timeout_sec=0.1,
    )
    controller.register_rearm(rearm_spec)
    receipt = CompositionSessionRearmReceipt(
        rearm_spec_id=rearm_spec.rearm_spec_id,
        rearm_spec_hash=rearm_spec.semantic_content_hash,
        logical_operation_spec_id=spec.operation_spec_id,
        logical_operation_spec_hash=spec.semantic_content_hash,
        prior_session_epoch=held.session_epoch,
        next_session_epoch=held.session_epoch + 1,
        prior_composer_producer_epoch=1,
        composer_producer_epoch=2,
        prior_authority_epoch=3,
        authority_epoch=4,
        prior_execution_epoch=5,
        execution_epoch=6,
        cancellation_receipt_id="cancel:transaction",
        measurement_id="measurement:transaction",
        measurement_hash="measurement-hash:transaction",
        producer_activation_id="producer-activation:transaction",
        authority_event_ids=("authority-event:transaction",),
        ready_gate_event_id="ready-gate:transaction",
        activation_input_barrier_time=10.0,
        completed_at=10.0,
        clock_domain_id="control.monotonic",
        satisfied_readiness_checks=rearm_spec.required_readiness_checks,
        status="composition_session_rearmed",
    )
    result = _result(
        operation_kind="rearm_composition_session",
        status="composition_session_rearmed",
        evidence_refs=(
            rearm_spec.rearm_spec_id,
            rearm_spec.semantic_content_hash,
            receipt.receipt_id,
            receipt.semantic_content_hash,
        ),
        sequence=2,
    )

    transitions = controller.complete_rearm_operation(
        rearm_spec_id=rearm_spec.rearm_spec_id,
        receipt=receipt,
        result=result,
        completion_accepted=True,
        source_fact_id="station-operation:rearm",
        source_fact_hash="station-operation-hash:rearm",
        rearm_intent_id="intent:rearm",
    )

    assert transitions is not None
    assert [transition.transition for transition in transitions] == [
        "supersede",
        "start",
    ]
    assert controller.current_state is not None
    assert controller.current_state.phase == "active"
    assert controller.current_state.session_epoch == held.session_epoch + 1
    assert controller.publication_barrier.snapshot().enabled
    assert controller.observer_failure_count == 5
    assert {failure[0] for failure in controller.observer_failures} == {
        "session_transition",
        "publication_barrier",
    }

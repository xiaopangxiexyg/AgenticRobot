from __future__ import annotations

import pytest

from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmSpec,
    CompositionSessionRegistry,
)


def _registry() -> CompositionSessionRegistry:
    return CompositionSessionRegistry(
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("human", "policy"),
        composition_spec_hash="composition-spec-hash",
        authority_gate_spec_hash="authority-gate-hash",
        execution_binding_hash="execution-binding-hash",
        clock=lambda: 10.0,
        clock_domain_id="control.monotonic",
    )


def _operation_spec() -> CompositionSessionOperationSpec:
    return CompositionSessionOperationSpec(
        session_key="session:transaction",
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("human", "policy"),
        composition_spec_hash="composition-spec-hash",
        authority_gate_spec_hash="authority-gate-hash",
        execution_binding_hash="execution-binding-hash",
        capture_boundary_id="capture-boundary:transaction",
    )


class _FailOnce:
    def __init__(self, message: str) -> None:
        self.message = message
        self.calls = 0

    def __call__(self, *_args, **_kwargs) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError(self.message)


def test_session_registration_rolls_back_when_required_trace_observer_fails() -> None:
    registry = _registry()
    observer = _FailOnce("spec trace failed")
    registry.bind_session_trace_observers(
        spec_observer=observer,
        transition_observer=lambda *_args, **_kwargs: None,
    )
    spec = _operation_spec()

    with pytest.raises(RuntimeError, match="spec trace failed"):
        registry.register(spec)

    assert registry.current_state is None
    assert registry.register(spec) == spec.operation_spec_id
    assert registry.current_state is not None
    assert registry.current_state.session_epoch == 0
    assert observer.calls == 2


def test_session_transition_rolls_back_when_required_trace_observer_fails() -> None:
    registry = _registry()
    observer = _FailOnce("transition trace failed")
    registry.bind_session_trace_observers(
        spec_observer=lambda _spec: None,
        transition_observer=observer,
    )
    registry.register(_operation_spec())

    transition_args = {
        "transition": "start",
        "reason": "ready",
        "source_fact_id": "operation-result:start",
        "source_fact_hash": "operation-result-hash:start",
        "start_intent_id": "intent:start",
    }
    with pytest.raises(RuntimeError, match="transition trace failed"):
        registry.advance(**transition_args)

    assert registry.current_state is not None
    assert registry.current_state.phase == "registered"
    event = registry.advance(**transition_args)
    assert event.transition == "start"
    assert registry.current_state.phase == "active"
    assert observer.calls == 2


def test_rearm_registration_rolls_back_when_required_trace_observer_fails() -> None:
    registry = _registry()
    observer = _FailOnce("rearm trace failed")
    registry.bind_session_trace_observers(
        spec_observer=lambda _spec: None,
        transition_observer=lambda *_args, **_kwargs: None,
        rearm_spec_observer=observer,
    )
    operation_spec = _operation_spec()
    registry.register(operation_spec)
    registry.advance(
        transition="start",
        reason="ready",
        source_fact_id="operation-result:start",
        source_fact_hash="operation-result-hash:start",
        start_intent_id="intent:start",
    )
    registry.advance(
        transition="hold",
        reason="operator hold",
        source_fact_id="intent:hold",
        source_fact_hash="intent-hash:hold",
    )
    held = registry.current_state
    assert held is not None
    rearm_spec = CompositionSessionRearmSpec(
        rearm_key="rearm:transaction",
        logical_operation_spec_id=operation_spec.operation_spec_id,
        logical_operation_spec_hash=operation_spec.semantic_content_hash,
        prior_session_epoch=held.session_epoch,
        prior_state_hash=held.semantic_content_hash,
        execution_binding_hash=operation_spec.execution_binding_hash,
        readiness_observer_binding_id="readiness:transaction",
        readiness_timeout_sec=1.0,
    )

    with pytest.raises(RuntimeError, match="rearm trace failed"):
        registry.register_rearm(rearm_spec)
    with pytest.raises(ValueError, match="Unknown.*rearm spec"):
        registry.require_rearm_spec(rearm_spec.rearm_spec_id)

    assert registry.register_rearm(rearm_spec) == rearm_spec.rearm_spec_id
    assert registry.require_rearm_spec(rearm_spec.rearm_spec_id) == rearm_spec
    assert observer.calls == 2

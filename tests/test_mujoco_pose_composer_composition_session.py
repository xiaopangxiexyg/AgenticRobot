from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from control_composition.activation import build_composition_activation_scope
from control_composition.lifecycle import CompositionSessionOperationSpec
from control_station.composition.mujoco_pose_composer import (
    MujocoPoseComposerAuthorityComposition,
    ScopedMujocoPoseCompositionSession,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _scope(*, epoch: int = 0):
    return build_composition_activation_scope(
        operation_spec_id="composition-session-operation:test",
        operation_spec_hash="composition-session-operation-hash:test",
        session_epoch=epoch,
        activated_at=10.0,
        clock_domain_id="control.monotonic",
        input_barrier_at=10.0,
        required_roles=("human", "policy"),
    )


def test_begin_activation_keeps_prior_state_when_trace_recording_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = object.__new__(MujocoPoseComposerAuthorityComposition)
    old_scope = _scope(epoch=0)
    old_receipts = {"human": object()}
    old_fact_times = {"emission:old": 9.0}
    old_steps = {"decision:old": object()}
    composition._lock = threading.RLock()
    composition._started = True
    composition._composition_activation_scope = old_scope
    composition._legacy_unscoped_input_barrier_time = None
    composition._input_receipt_by_role = old_receipts.copy()
    composition._human_fact_time_by_emission_id = old_fact_times.copy()
    composition._step_by_decision_key = old_steps.copy()
    composition.composition_spec = SimpleNamespace(required_roles=("human", "policy"))

    class _AuthoritativePort:
        trace_context = object()

        @staticmethod
        def record_external_facts(_facts) -> None:
            raise RuntimeError("synthetic trace recording failure")

    composition.runtime = SimpleNamespace(
        config=SimpleNamespace(clock_domain_id="control.monotonic"),
        authoritative_port=_AuthoritativePort(),
    )
    monkeypatch.setattr(
        "control_station.composition.mujoco_pose_composer."
        "project_composition_activation_scope",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(RuntimeError, match="trace recording failure"):
        composition.begin_composition_activation(scope=_scope(epoch=1))

    assert composition._composition_activation_scope == old_scope
    assert composition._input_receipt_by_role == old_receipts
    assert composition._human_fact_time_by_emission_id == old_fact_times
    assert composition._step_by_decision_key == old_steps


def test_begin_activation_retry_preserves_inputs_for_the_same_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composition = object.__new__(MujocoPoseComposerAuthorityComposition)
    initial_scope = _scope(epoch=0)
    next_scope = _scope(epoch=1)
    recorded_facts = []
    composition._lock = threading.RLock()
    composition._started = True
    composition._composition_activation_scope = initial_scope
    composition._legacy_unscoped_input_barrier_time = None
    composition._input_receipt_by_role = {"human": object()}
    composition._human_fact_time_by_emission_id = {"emission:old": 9.0}
    composition._step_by_decision_key = {"decision:old": object()}
    composition.composition_spec = SimpleNamespace(required_roles=("human", "policy"))

    class _AuthoritativePort:
        trace_context = object()

        @staticmethod
        def record_external_facts(facts) -> None:
            recorded_facts.extend(facts)

    composition.runtime = SimpleNamespace(
        config=SimpleNamespace(clock_domain_id="control.monotonic"),
        authoritative_port=_AuthoritativePort(),
    )
    monkeypatch.setattr(
        "control_station.composition.mujoco_pose_composer."
        "project_composition_activation_scope",
        lambda *_args, **_kwargs: object(),
    )

    assert composition.begin_composition_activation(scope=next_scope) == 10.0
    assert composition._composition_activation_scope == next_scope
    assert composition._input_receipt_by_role == {}
    assert len(recorded_facts) == 1

    new_receipt = object()
    composition._input_receipt_by_role["human"] = new_receipt

    assert composition.begin_composition_activation(scope=next_scope) == 10.0
    assert composition._input_receipt_by_role == {"human": new_receipt}
    assert len(recorded_facts) == 1


def test_station_completion_retries_finalization_without_reexecuting_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = object.__new__(ScopedMujocoPoseCompositionSession)
    spec = CompositionSessionOperationSpec(
        session_key="retry-finalization",
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("human", "policy"),
        composition_spec_hash="composition:retry",
        authority_gate_spec_hash="gate:retry",
        execution_binding_hash="binding:retry",
        capture_boundary_id="capture:retry",
    )

    class _SessionController:
        def __init__(self) -> None:
            self.current_state = SimpleNamespace(phase="registered", session_epoch=0)
            self.calls = 0

        @staticmethod
        def require_registered_spec(_operation_spec_id):
            return spec

        def complete_start_operation(self, *, before_publication, **_kwargs):
            self.calls += 1
            before_publication()
            if self.calls == 1:
                raise RuntimeError("synthetic finalization failure")
            self.current_state = SimpleNamespace(phase="active", session_epoch=0)
            return object()

    class _Station:
        def __init__(self, execution) -> None:
            self.execution = execution
            self.calls = 0

        def execute_pending_operation(self, _transition, _adapter):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("station operation cannot execute twice")
            return self.execution

    class _DataPlane:
        composition_spec = SimpleNamespace(required_roles=("human", "policy"))
        runtime = SimpleNamespace(
            config=SimpleNamespace(
                clock_domain_id="control.monotonic",
                embodiment_instance_id="embodiment:test",
            )
        )

        def __init__(self) -> None:
            self.activations = []

        def begin_composition_activation(self, *, scope) -> float:
            self.activations.append(scope)
            return scope.input_barrier_at

    result = SimpleNamespace(
        succeeded=True,
        available_at=10.0,
        operation_result_id="operation-result:start",
        semantic_content_hash="operation-result-hash:start",
    )
    completion = SimpleNamespace(accepted=True, result=result)
    execution = SimpleNamespace(completion_transition=completion)
    station = _Station(execution)
    controller = _SessionController()
    data_plane = _DataPlane()
    session._station = station
    session._data_plane = data_plane
    session.session_controller = controller
    session.composer_source_slot_id = "composer"
    session.human_source_slot_id = "human"
    session.replay_source_slot_id = "policy"
    session.target_adapter = SimpleNamespace(subject_id="target")
    session.human_adapter = SimpleNamespace(subject_id="human")
    session.replay_adapter = SimpleNamespace(subject_id="policy")
    session.composer_adapter = SimpleNamespace(subject_id="composer")
    session._session_operation_execution_by_intent_id = {}
    session._session_operation_transition_by_intent_id = {}
    session._finalized_session_operation_intent_ids = set()
    session._session_operation_lock = threading.RLock()
    transition = SimpleNamespace(
        receipt=SimpleNamespace(accepted=True),
        intent=SimpleNamespace(
            intent_id="intent:start",
            intent_kind="start_composition_session",
            subject_id="composer",
            operation_spec_id=spec.operation_spec_id,
        ),
    )
    monkeypatch.setattr(
        "control_station.composition.mujoco_pose_composer."
        "project_station_operation_transition",
        lambda *_args, **_kwargs: SimpleNamespace(
            fact_id="station-operation:start",
            semantic_content_hash="station-operation-hash:start",
        ),
    )

    with pytest.raises(RuntimeError, match="finalization failure"):
        session.complete_pending_operation(transition)
    assert station.calls == 1
    assert controller.current_state.phase == "registered"

    assert session.complete_pending_operation(transition) is execution
    assert station.calls == 1
    assert controller.calls == 2

    collision = SimpleNamespace(
        receipt=SimpleNamespace(accepted=True),
        intent=SimpleNamespace(
            intent_id="intent:start",
            intent_kind="start_composition_session",
            subject_id="composer",
            operation_spec_id="another-operation-spec",
        ),
    )
    with pytest.raises(RuntimeError, match="intent identity collision"):
        session.complete_pending_operation(collision)
    assert station.calls == 1
    assert controller.current_state.phase == "active"
    assert len(data_plane.activations) == 2

    assert session.complete_pending_operation(transition) is execution
    assert station.calls == 1
    assert controller.calls == 2

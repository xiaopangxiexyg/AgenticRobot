from __future__ import annotations

import pytest

from control_composition import (
    RESIDUAL_ANCHOR_LIFECYCLE_PENDING,
    RESIDUAL_ANCHOR_LIFECYCLE_READY,
    ResidualAnchorComponentBinding,
    ResidualAnchorLifecycleDraft,
    ResidualAnchorLifecycleSpec,
    ResidualAnchorLifecycleStore,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
    capture_residual_anchor_lifecycle_snapshot,
)
from robot_motion.runtime.control_admission import ControlAdmissionCoordinator
from robot_motion.runtime.identity import ClockDomainDescriptor
from robot_motion.runtime.producer_epoch import ProducerEpochRegistry


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _fixture():
    clock = _Clock()
    coordinator = ControlAdmissionCoordinator(
        "robot",
        clock_domain=ClockDomainDescriptor(
            clock_domain_id="control.monotonic",
            runtime_session_id="session",
            boot_id="boot",
            reset_epoch=0,
        ),
        clock=clock,
    )
    epochs = ProducerEpochRegistry(
        runtime_session_id="session",
        coordinator=coordinator,
    )
    epochs.activate(
        producer_id="anchor-lifecycle",
        producer_epoch=1,
        produced_at=9.8,
        produced_clock_domain_id="control.monotonic",
        reason="test_startup",
    )
    activation = build_composition_activation_scope(
        operation_spec_id="allocation",
        operation_spec_hash="allocation-hash",
        session_epoch=1,
        activated_at=9.9,
        clock_domain_id="control.monotonic",
        input_barrier_at=9.9,
        required_roles=("base", "human_residual", "human_absolute"),
    )
    spec = ResidualAnchorLifecycleSpec(
        spec_id="human-anchor",
        source_role="human_residual",
        component_ids=("arm",),
        absolute_action_schema_hash="absolute-schema",
        relative_action_schema_hash="relative-schema",
    )
    store = ResidualAnchorLifecycleStore(
        runtime_session_id="session",
        coordinator=coordinator,
        producer_epochs=epochs,
    )
    return clock, coordinator, epochs, activation, spec, store


def _request(activation, *, generation: int, previous=None, at: float):
    return ResidualAnchorRequest.for_activation(
        activation,
        request_generation=generation,
        source_role="human_residual",
        component_ids=("arm",),
        source_binding_id="human-device",
        source_binding_hash="human-device-hash",
        absolute_action_schema_hash="absolute-schema",
        relative_action_schema_hash="relative-schema",
        requested_at=at,
        previous_request=previous,
    )


def _receipt(request, *, at: float, suffix: str):
    return ResidualAnchorReceipt.capture(
        request,
        source_action_id=f"human:{suffix}",
        source_action_hash=f"human-hash:{suffix}",
        source_available_at=at,
        captured_at=at,
        component_bindings=(
            ResidualAnchorComponentBinding(
                component_id="arm",
                source_sample_id=f"sample:{suffix}",
                source_sample_hash=f"sample-hash:{suffix}",
                absolute_payload_schema_hash="arm-absolute-schema",
                anchor_id=f"anchor:{suffix}",
                anchor_hash=f"anchor-hash:{suffix}",
            ),
        ),
    )


def _draft(spec, request, *, sequence: int, phase: str, boundary: float, receipt=None):
    return ResidualAnchorLifecycleDraft(
        runtime_session_id="session",
        control_domain_id="robot",
        producer_id="anchor-lifecycle",
        producer_epoch=1,
        sequence=sequence,
        lifecycle_scope_id="human-allocation",
        lifecycle_spec_hash=spec.semantic_content_hash,
        phase=phase,
        effective_for_target_at=boundary,
        observed_at=max(10.0, request.requested_at),
        observed_clock_domain_id="control.monotonic",
        anchor_request=request,
        anchor_receipt=receipt,
    )


def _snapshot(coordinator, epochs, activation, spec, store, *, target: float):
    cut = coordinator.capture_cut(cutoff_at=10.8).cut
    return capture_residual_anchor_lifecycle_snapshot(
        store,
        cut,
        producer_epochs=epochs.snapshot(cut),
        spec=spec,
        activation_scope=activation,
        lifecycle_scope_id="human-allocation",
        target_effective_at=target,
    )


def test_pending_and_ready_are_selected_at_the_same_receiver_cut() -> None:
    clock, coordinator, epochs, activation, spec, store = _fixture()
    request = _request(activation, generation=0, at=10.0)
    pending = store.admit(
        _draft(spec, request, sequence=0, phase="pending", boundary=10.2),
        spec=spec,
        activation_scope=activation,
    )
    assert pending.accepted
    assert _snapshot(
        coordinator, epochs, activation, spec, store, target=10.1
    ).status == "missing"
    assert _snapshot(
        coordinator, epochs, activation, spec, store, target=10.2
    ).status == RESIDUAL_ANCHOR_LIFECYCLE_PENDING

    receipt = _receipt(request, at=10.15, suffix="initial")
    clock.value = 10.16
    ready = store.admit(
        _draft(
            spec,
            request,
            sequence=1,
            phase="ready",
            boundary=10.2,
            receipt=receipt,
        ),
        spec=spec,
        activation_scope=activation,
    )
    assert ready.accepted
    snapshot = _snapshot(
        coordinator, epochs, activation, spec, store, target=10.2
    )
    assert snapshot.status == RESIDUAL_ANCHOR_LIFECYCLE_READY
    assert snapshot.ready_receipt is receipt


def test_successor_pending_fences_only_targets_at_or_after_boundary() -> None:
    clock, coordinator, epochs, activation, spec, store = _fixture()
    initial = _request(activation, generation=0, at=10.0)
    initial_receipt = _receipt(initial, at=10.1, suffix="initial")
    for draft in (
        _draft(spec, initial, sequence=0, phase="pending", boundary=10.2),
        _draft(
            spec,
            initial,
            sequence=1,
            phase="ready",
            boundary=10.2,
            receipt=initial_receipt,
        ),
    ):
        assert store.admit(draft, spec=spec, activation_scope=activation).accepted
        clock.value += 0.01
    successor = _request(
        activation,
        generation=1,
        previous=initial,
        at=10.3,
    )
    assert store.admit(
        _draft(spec, successor, sequence=2, phase="pending", boundary=10.5),
        spec=spec,
        activation_scope=activation,
    ).accepted

    before = _snapshot(
        coordinator, epochs, activation, spec, store, target=10.49
    )
    boundary = _snapshot(
        coordinator, epochs, activation, spec, store, target=10.5
    )
    assert before.status == RESIDUAL_ANCHOR_LIFECYCLE_READY
    assert before.ready_receipt is initial_receipt
    assert boundary.status == RESIDUAL_ANCHOR_LIFECYCLE_PENDING
    assert boundary.selected_record.anchor_request_generation == 1


def test_exact_duplicate_is_idempotent_but_phase_order_cannot_regress() -> None:
    _clock, _coordinator, _epochs, activation, spec, store = _fixture()
    request = _request(activation, generation=0, at=10.0)
    draft = _draft(spec, request, sequence=0, phase="pending", boundary=10.2)
    first = store.admit(draft, spec=spec, activation_scope=activation)
    duplicate = store.admit(draft, spec=spec, activation_scope=activation)
    assert first.accepted and duplicate.accepted_existing

    another_pending = store.admit(
        _draft(spec, request, sequence=1, phase="pending", boundary=10.2),
        spec=spec,
        activation_scope=activation,
    )
    assert not another_pending.accepted
    assert another_pending.rejection_code == "pending_completion_mismatch"


def test_ready_must_preserve_the_pending_target_boundary() -> None:
    clock, _coordinator, _epochs, activation, spec, store = _fixture()
    request = _request(activation, generation=0, at=10.0)
    assert store.admit(
        _draft(spec, request, sequence=0, phase="pending", boundary=10.2),
        spec=spec,
        activation_scope=activation,
    ).accepted
    clock.value = 10.16
    rejected = store.admit(
        _draft(
            spec,
            request,
            sequence=1,
            phase="ready",
            boundary=10.21,
            receipt=_receipt(request, at=10.15, suffix="initial"),
        ),
        spec=spec,
        activation_scope=activation,
    )
    assert not rejected.accepted
    assert rejected.rejection_code == "pending_completion_mismatch"
    assert len(store) == 1

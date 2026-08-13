from __future__ import annotations

import pytest

from control_composition.allocation import (
    ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY,
    ALLOCATION_PUBLICATION_OUTCOME_LATE,
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
    COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC,
    AllocationPublicationRuntime,
    CompositionAllocationOverrideRequest,
    apply_composition_allocation_override,
    empty_composition_allocation_override_state,
    empty_composition_allocation_state,
    evaluate_composition_allocation,
)
from control_composition.composition import CompositionContextAdmissionReceipt
from tests.test_composition_allocation import _activity, _allocation_spec, _magnitude
from tests.test_composition_allocation_publication import Clock, _activation, _basis


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _override_state(spec, activation):
    return empty_composition_allocation_override_state(
        runtime_session_id="session",
        control_domain_id="robot",
        composition_activation_id=activation.activation_id,
        composition_activation_hash=activation.semantic_content_hash,
        allocation_spec_hash=spec.semantic_content_hash,
        clock_domain_id="control.monotonic",
        component_ids=spec.component_ids,
        allowed_modes=(
            COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        ),
    )


def _request(state, *, mode: str, at: float, suffix: str):
    return CompositionAllocationOverrideRequest(
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        composition_activation_id=state.composition_activation_id,
        composition_activation_hash=state.composition_activation_hash,
        allocation_spec_hash=state.allocation_spec_hash,
        component_ids=state.component_ids,
        requested_mode=mode,
        requested_at=at,
        available_at=at,
        clock_domain_id=state.clock_domain_id,
        source_fact_ids=(f"button:{suffix}",),
        source_fact_hashes=(f"button-hash:{suffix}",),
    )


def test_persistent_override_bypasses_dwell_but_keeps_fresh_evidence() -> None:
    spec = _allocation_spec()
    activation = _activation(activated_at=0.0)
    state = empty_composition_allocation_state("allocation", spec=spec)
    override = _override_state(spec, activation)
    request = _request(
        override,
        mode=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        at=1.0,
        suffix="relative",
    )
    receipt = apply_composition_allocation_override(
        override,
        request,
        admitted_at=1.0,
    )

    evaluation = evaluate_composition_allocation(
        state,
        spec=spec,
        activity_by_component={"tool": _activity("tool", level=0.0, evaluated_at=1.01)},
        residual_magnitude_by_component={
            "tool": _magnitude("tool", level=0.0, evaluated_at=1.01)
        },
        evaluated_at=1.01,
        clock_domain_id="control.monotonic",
        override_state=receipt.state_after,
    )

    assert evaluation.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY
    )
    assert evaluation.decision.selections[0].reason == (
        "persistent_allocation_override"
    )
    assert evaluation.decision.allocation_override_state_id == (
        receipt.state_after.state_id
    )


def test_automatic_request_releases_override_without_hiding_lineage() -> None:
    spec = _allocation_spec()
    activation = _activation(activated_at=0.0)
    initial = _override_state(spec, activation)
    forced = apply_composition_allocation_override(
        initial,
        _request(
            initial,
            mode=COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
            at=1.0,
            suffix="relative",
        ),
        admitted_at=1.0,
    ).state_after
    automatic = apply_composition_allocation_override(
        forced,
        _request(
            forced,
            mode=COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC,
            at=1.1,
            suffix="auto",
        ),
        admitted_at=1.1,
    ).state_after

    assert automatic.requested_mode == COMPOSITION_ALLOCATION_OVERRIDE_AUTOMATIC
    evaluation = evaluate_composition_allocation(
        empty_composition_allocation_state("allocation", spec=spec),
        spec=spec,
        activity_by_component={"tool": _activity("tool", level=0.0, evaluated_at=1.2)},
        residual_magnitude_by_component={
            "tool": _magnitude("tool", level=0.0, evaluated_at=1.2)
        },
        evaluated_at=1.2,
        clock_domain_id="control.monotonic",
        override_state=automatic,
    )
    assert evaluation.decision.selected_mode_by_component["tool"] == (
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY
    )
    assert evaluation.decision.allocation_override_state_hash == (
        automatic.semantic_content_hash
    )


def test_override_fences_pending_context_and_late_receipt_cannot_commit() -> None:
    spec = _allocation_spec()
    activation = _activation()
    clock = Clock(10.1)
    runtime = AllocationPublicationRuntime(
        allocation_scope_id="operator-tool",
        allocation_spec=spec,
        activation_scope=activation,
        runtime_session_id="session",
        control_domain_id="robot",
        control_clock_reset_epoch=0,
        context_channel="allocation",
        producer_id="allocation-producer",
        producer_epoch=1,
        initial_reducer_state=empty_composition_allocation_state(
            "allocation-producer",
            spec=spec,
        ),
        initial_override_state=_override_state(spec, activation),
        clock=clock,
    )
    attempt_outcome = runtime.offer_basis(
        _basis(activation, index=1, cut_at=10.05)
    )
    assert attempt_outcome.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    attempt = runtime.snapshot().pending_attempt
    assert attempt is not None

    clock.value = 10.11
    override = runtime.apply_override(
        _request(
            runtime.allocation_override_state,
            mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            at=10.1,
            suffix="base",
        )
    )
    assert override.status == "applied"
    assert runtime.snapshot().pending_attempt is None

    late = runtime.resolve_admission(
        attempt.attempt_id,
        CompositionContextAdmissionReceipt(
            accepted=False,
            rejection_code="test_late",
            message="late",
        ),
        occurred_at=10.12,
    )
    assert late.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_LATE
    assert late.reason == "receipt_belongs_to_superseded_allocation_override"


def test_absolute_override_is_rejected_by_scheme_a_allowlist() -> None:
    spec = _allocation_spec()
    activation = _activation(activated_at=0.0)
    state = _override_state(spec, activation)
    with pytest.raises(ValueError, match="non-promotable"):
        apply_composition_allocation_override(
            state,
            _request(
                state,
                mode="absolute_overlay_owner",
                at=1.0,
                suffix="absolute",
            ),
            admitted_at=1.0,
        )


def test_persistent_override_uses_a_separate_rolling_selection_lease() -> None:
    spec = _allocation_spec()
    activation = _activation()
    clock = Clock(10.1)
    runtime = AllocationPublicationRuntime(
        allocation_scope_id="operator-tool",
        allocation_spec=spec,
        activation_scope=activation,
        runtime_session_id="session",
        control_domain_id="robot",
        control_clock_reset_epoch=0,
        context_channel="allocation",
        producer_id="allocation-producer",
        producer_epoch=1,
        initial_reducer_state=empty_composition_allocation_state(
            "allocation-producer",
            spec=spec,
        ),
        initial_override_state=_override_state(spec, activation),
        persistent_override_selection_lease_sec=0.5,
        clock=clock,
    )
    runtime.apply_override(
        _request(
            runtime.allocation_override_state,
            mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
            at=10.05,
            suffix="base",
        )
    )

    outcome = runtime.offer_basis(
        _basis(
            activation,
            index=1,
            cut_at=10.05,
            valid_until=10.3,
            support_until=10.3,
        )
    )

    assert outcome.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    attempt = runtime.snapshot().pending_attempt
    assert attempt is not None
    assert attempt.context_draft.effective_from == pytest.approx(10.1)
    assert attempt.context_draft.effective_until == pytest.approx(10.6)
    assert attempt.allocation_decision.selections[0].valid_until == pytest.approx(
        10.6
    )
    assert attempt.activity_basis.support_until == pytest.approx(10.3)


def test_automatic_allocation_keeps_evidence_bounded_selection_lifetime() -> None:
    spec = _allocation_spec()
    activation = _activation()
    clock = Clock(10.1)
    runtime = AllocationPublicationRuntime(
        allocation_scope_id="operator-tool",
        allocation_spec=spec,
        activation_scope=activation,
        runtime_session_id="session",
        control_domain_id="robot",
        control_clock_reset_epoch=0,
        context_channel="allocation",
        producer_id="allocation-producer",
        producer_epoch=1,
        initial_reducer_state=empty_composition_allocation_state(
            "allocation-producer",
            spec=spec,
        ),
        initial_override_state=_override_state(spec, activation),
        persistent_override_selection_lease_sec=0.5,
        clock=clock,
    )

    outcome = runtime.offer_basis(
        _basis(
            activation,
            index=1,
            cut_at=10.05,
            valid_until=10.3,
            support_until=10.3,
        )
    )

    assert outcome.outcome_kind == ALLOCATION_PUBLICATION_OUTCOME_ATTEMPT_READY
    attempt = runtime.snapshot().pending_attempt
    assert attempt is not None
    assert attempt.context_draft.effective_until == pytest.approx(10.3)
    assert attempt.allocation_decision.selections[0].valid_until == pytest.approx(
        10.3
    )


def test_partial_override_does_not_extend_automatic_component_selection() -> None:
    spec = _allocation_spec(components=("arm", "gripper"))
    activation = _activation(activated_at=0.0)
    initial = _override_state(spec, activation)
    request = CompositionAllocationOverrideRequest(
        runtime_session_id=initial.runtime_session_id,
        control_domain_id=initial.control_domain_id,
        composition_activation_id=initial.composition_activation_id,
        composition_activation_hash=initial.composition_activation_hash,
        allocation_spec_hash=initial.allocation_spec_hash,
        component_ids=("arm",),
        requested_mode=COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        requested_at=1.0,
        available_at=1.0,
        clock_domain_id=initial.clock_domain_id,
        source_fact_ids=("button:arm-only",),
        source_fact_hashes=("button-hash:arm-only",),
    )
    override = apply_composition_allocation_override(
        initial,
        request,
        admitted_at=1.0,
    ).state_after

    evaluation = evaluate_composition_allocation(
        empty_composition_allocation_state("allocation", spec=spec),
        spec=spec,
        activity_by_component={
            component_id: _activity(
                component_id,
                level=0.0,
                evaluated_at=1.01,
                valid_until=1.3,
            )
            for component_id in spec.component_ids
        },
        residual_magnitude_by_component={
            component_id: _magnitude(
                component_id,
                level=0.0,
                evaluated_at=1.01,
                valid_until=1.3,
            )
            for component_id in spec.component_ids
        },
        evaluated_at=1.01,
        clock_domain_id="control.monotonic",
        override_state=override,
        persistent_override_valid_until=1.6,
    )

    valid_until = {
        selection.component_id: selection.valid_until
        for selection in evaluation.decision.selections
    }
    assert valid_until == {"arm": pytest.approx(1.6), "gripper": pytest.approx(1.3)}

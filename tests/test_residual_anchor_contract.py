from __future__ import annotations

import pytest

from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    ResidualAnchorState,
    build_composition_activation_scope,
)


def _activation(*, epoch: int = 1, barrier: float = 10.0):
    return build_composition_activation_scope(
        operation_spec_id="relative-profile",
        operation_spec_hash="relative-profile-hash",
        session_epoch=epoch,
        activated_at=barrier,
        clock_domain_id="receiver.monotonic",
        input_barrier_at=barrier,
        required_roles=("baseline", "residual"),
    )


def _request(
    *,
    generation: int = 0,
    previous: ResidualAnchorRequest | None = None,
    activation=None,
) -> ResidualAnchorRequest:
    return ResidualAnchorRequest.for_activation(
        _activation() if activation is None else activation,
        request_generation=generation,
        source_role="residual",
        component_ids=("arm", "hand"),
        source_binding_id="source-binding:1",
        source_binding_hash="source-binding-hash:1",
        absolute_action_schema_hash="absolute-schema-hash",
        relative_action_schema_hash="relative-schema-hash",
        requested_at=10.1 + generation,
        previous_request=previous,
    )


def _bindings() -> tuple[ResidualAnchorComponentBinding, ...]:
    return (
        ResidualAnchorComponentBinding(
            component_id="arm",
            source_sample_id="source-action:1:arm",
            source_sample_hash="source-action-hash:1:arm",
            absolute_payload_schema_hash="absolute-arm-schema-hash",
            anchor_id="pose-anchor:0",
            anchor_hash="pose-anchor-hash:0",
        ),
        ResidualAnchorComponentBinding(
            component_id="hand",
            source_sample_id="source-action:1:hand",
            source_sample_hash="source-action-hash:1:hand",
            absolute_payload_schema_hash="absolute-hand-schema-hash",
            anchor_id="joint-anchor:0",
            anchor_hash="joint-anchor-hash:0",
        ),
    )


def test_atomic_receipt_binds_role_activation_source_and_components() -> None:
    request = _request()
    receipt = ResidualAnchorReceipt.capture(
        request,
        source_action_id="source-action:1",
        source_action_hash="source-action-hash:1",
        source_available_at=10.2,
        captured_at=10.21,
        component_bindings=_bindings(),
    )
    state = ResidualAnchorState(request=request).accept(receipt)

    assert state.ready
    assert receipt.source_role == "residual"
    assert receipt.composition_activation_id == (
        request.composition_activation_id
    )
    assert set(receipt.component_binding_by_id) == {"arm", "hand"}
    assert state.accept(receipt) is state


def test_partial_or_extra_component_anchor_is_rejected() -> None:
    request = _request()
    with pytest.raises(ValueError, match="atomically cover exactly"):
        ResidualAnchorReceipt.capture(
            request,
            source_action_id="source-action:1",
            source_action_hash="source-action-hash:1",
            source_available_at=10.2,
            captured_at=10.21,
            component_bindings=_bindings()[:1],
        )


def test_pre_barrier_source_cannot_form_anchor() -> None:
    request = _request()
    with pytest.raises(ValueError, match="before its request barrier"):
        ResidualAnchorReceipt.capture(
            request,
            source_action_id="source-action:old",
            source_action_hash="source-action-hash:old",
            source_available_at=10.09,
            captured_at=10.2,
            component_bindings=_bindings(),
        )


def test_recenter_chain_advances_exactly_one_without_changing_role_or_coverage() -> None:
    first = _request()
    second = _request(generation=1, previous=first)

    assert second.previous_request_id == first.request_id
    assert second.previous_request_hash == first.semantic_content_hash
    with pytest.raises(ValueError, match="advance by exactly one"):
        _request(generation=2, previous=first)

    with pytest.raises(ValueError, match="source_role cannot change"):
        ResidualAnchorRequest.for_activation(
            _activation(epoch=2, barrier=12.0),
            request_generation=1,
            source_role="another-residual",
            component_ids=("arm", "hand"),
            source_binding_id="source-binding:1",
            source_binding_hash="source-binding-hash:1",
            absolute_action_schema_hash="absolute-schema-hash",
            relative_action_schema_hash="relative-schema-hash",
            requested_at=12.1,
            previous_request=first,
        )


def test_request_cannot_predate_activation_barrier() -> None:
    with pytest.raises(ValueError, match="cannot predate"):
        ResidualAnchorRequest.for_activation(
            _activation(barrier=10.0),
            request_generation=0,
            source_role="residual",
            component_ids=("arm",),
            source_binding_id="source-binding:1",
            source_binding_hash="source-binding-hash:1",
            absolute_action_schema_hash="absolute-schema-hash",
            relative_action_schema_hash="relative-schema-hash",
            requested_at=9.9,
        )


def test_one_request_cannot_accept_two_different_receipts() -> None:
    request = _request()
    first = ResidualAnchorReceipt.capture(
        request,
        source_action_id="source-action:1",
        source_action_hash="source-action-hash:1",
        source_available_at=10.2,
        captured_at=10.21,
        component_bindings=_bindings(),
    )
    second = ResidualAnchorReceipt.capture(
        request,
        source_action_id="source-action:2",
        source_action_hash="source-action-hash:2",
        source_available_at=10.3,
        captured_at=10.31,
        component_bindings=_bindings(),
    )
    state = ResidualAnchorState(request=request).accept(first)

    with pytest.raises(ValueError, match="already has another receipt"):
        state.accept(second)

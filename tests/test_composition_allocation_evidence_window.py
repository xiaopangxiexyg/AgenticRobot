from __future__ import annotations

import math

import pytest

from control_composition import (
    ResidualAnchorComponentBinding,
    ResidualAnchorReceipt,
    ResidualAnchorRequest,
    build_composition_activation_scope,
)
from control_composition.allocation import (
    AllocationEvidenceSamplePair,
    AllocationEvidenceWindow,
)
from robot_motion.action import ComponentAction
from robot_motion.runtime.control_admission import ControlCut
from tests.test_composition_allocation import (
    _action_schema,
    _allocation_spec,
    _relative_action_schema,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _activation(*, generation: int = 0):
    return build_composition_activation_scope(
        operation_spec_id="allocation-residual-evidence",
        operation_spec_hash=f"allocation-residual-evidence-hash:{generation}",
        session_epoch=generation + 1,
        activated_at=2.5 + generation,
        clock_domain_id="control.monotonic",
        input_barrier_at=2.5 + generation,
        required_roles=("base_absolute", "operator_relative"),
    )


def _anchor(activation, *, generation: int = 0):
    absolute_schema = _action_schema()
    relative_schema = _relative_action_schema()
    request = ResidualAnchorRequest.for_activation(
        activation,
        request_generation=generation,
        source_role="operator_relative",
        component_ids=("tool", "hand"),
        source_binding_id=f"operator-source:{generation}",
        source_binding_hash=f"operator-source-hash:{generation}",
        absolute_action_schema_hash=absolute_schema.semantic_content_hash,
        relative_action_schema_hash=relative_schema.semantic_content_hash,
        requested_at=activation.activated_at + 0.01,
    )
    return ResidualAnchorReceipt.capture(
        request,
        source_action_id=f"operator-anchor-action:{generation}",
        source_action_hash=f"operator-anchor-action-hash:{generation}",
        source_available_at=activation.activated_at + 0.02,
        captured_at=activation.activated_at + 0.03,
        component_bindings=tuple(
            ResidualAnchorComponentBinding(
                component_id=component_id,
                source_sample_id=f"operator-anchor:{component_id}:{generation}",
                source_sample_hash=(
                    f"operator-anchor-hash:{component_id}:{generation}"
                ),
                absolute_payload_schema_hash=(
                    absolute_schema.payload_schema_by_component[
                        component_id
                    ].semantic_content_hash
                ),
                anchor_id=f"anchor:{component_id}:{generation}",
                anchor_hash=f"anchor-hash:{component_id}:{generation}",
            )
            for component_id in ("tool", "hand")
        ),
    )


def _absolute_action(*, x: float, hand: float) -> ComponentAction:
    return ComponentAction(
        action_schema=_action_schema(),
        values_by_component={
            "tool": (x, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            "hand": (hand, -hand, hand, -hand, hand),
        },
    )


def _residual_action(*, x: float, hand: float) -> ComponentAction:
    return ComponentAction(
        action_schema=_relative_action_schema(),
        values_by_component={
            "tool": (
                x,
                0.0,
                0.0,
                0.0,
                0.0,
                math.sin(x / 2.0),
                math.cos(x / 2.0),
            ),
            "hand": (hand, -hand, hand, -hand, hand),
        },
    )


def _pair(
    receipt,
    *,
    index: int,
    sampled_at: float,
    available_at: float,
    admission_sequence: int,
) -> AllocationEvidenceSamplePair:
    return AllocationEvidenceSamplePair(
        runtime_session_id="session",
        control_domain_id="robot",
        clock_domain_id="control.monotonic",
        clock_reset_epoch=0,
        sampled_at=sampled_at,
        available_at=available_at,
        last_control_admission_sequence=admission_sequence,
        source_sample_id=f"human-sample:{index}",
        source_sample_hash=f"human-sample-hash:{index}",
        absolute_action=_absolute_action(x=0.01 * index, hand=0.02 * index),
        residual_action=_residual_action(x=0.01 * index, hand=0.02 * index),
        anchor_receipt_id=receipt.receipt_id,
        anchor_receipt_hash=receipt.semantic_content_hash,
        anchor_request_generation=receipt.request_generation,
    )


def _window():
    return AllocationEvidenceWindow(
        allocation_spec=_allocation_spec(components=("tool", "hand")),
        window_duration_sec=0.5,
        evidence_ttl_sec=0.25,
        activity_normalization_scale_by_component={
            "tool": {"translation_displacement_m": 0.02},
            "hand": {"coordinate_linf_displacement_native": 0.04},
        },
        residual_normalization_scale_by_component={
            "tool": {"translation_from_anchor_m": 0.02},
            "hand": {"coordinate_linf_from_anchor_native": 0.04},
        },
    )


def _cut(*, at: float, sequence: int) -> ControlCut:
    return ControlCut(
        runtime_session_id="session",
        control_domain_id="robot",
        clock_domain_id="control.monotonic",
        clock_reset_epoch=0,
        cutoff_at=at,
        cutoff_admission_sequence=sequence,
    )


def test_basis_requires_receiver_time_and_atomic_batch_sequence_visibility() -> None:
    activation = _activation()
    receipt = _anchor(activation)
    window = _window()
    window.bind_anchor(receipt)
    window.offer(
        _pair(
            receipt,
            index=0,
            sampled_at=2.8,
            available_at=2.81,
            admission_sequence=10,
        )
    )
    second_pair = _pair(
        receipt,
        index=1,
        sampled_at=2.9,
        available_at=3.10,
        admission_sequence=12,
    )
    window.offer(second_pair)

    assert window.capture_basis(
        allocation_scope_id="operator-allocation",
        control_cut=_cut(at=3.0, sequence=20),
        activation_scope=activation,
        requested_effective_from=3.0,
    ) is None
    assert window.capture_basis(
        allocation_scope_id="operator-allocation",
        control_cut=_cut(at=3.2, sequence=11),
        activation_scope=activation,
        requested_effective_from=3.2,
    ) is None

    basis = window.capture_basis(
        allocation_scope_id="operator-allocation",
        control_cut=_cut(at=3.10, sequence=12),
        activation_scope=activation,
        requested_effective_from=3.10,
    )
    assert basis is not None
    assert set(basis.activity_by_component) == {"tool", "hand"}
    assert basis.activity_by_component["tool"].activity_level == pytest.approx(0.5)
    assert basis.residual_magnitude_by_component[
        "hand"
    ].magnitude_level == pytest.approx(0.5)
    residual = basis.residual_magnitude_by_component["tool"]
    assert residual.residual_action_hash == (
        _residual_action(x=0.01, hand=0.02).semantic_content_hash
    )
    assert residual.source_fact_ids == (second_pair.pair_id,)


def test_rebase_clears_old_influence_and_binds_new_anchor_generation() -> None:
    activation = _activation()
    receipt = _anchor(activation)
    window = _window()
    window.bind_anchor(receipt)
    for index, sampled_at in enumerate((2.8, 2.9)):
        window.offer(
            _pair(
                receipt,
                index=index,
                sampled_at=sampled_at,
                available_at=sampled_at + 0.01,
                admission_sequence=10 + 2 * index,
            )
        )
    assert window.snapshot().retained_pair_count == 2

    next_activation = _activation(generation=1)
    next_receipt = _anchor(next_activation, generation=0)
    window.bind_anchor(next_receipt)

    snapshot = window.snapshot()
    assert snapshot.retained_pair_count == 0
    assert snapshot.anchor_receipt_id == next_receipt.receipt_id
    assert window.capture_basis(
        allocation_scope_id="operator-allocation",
        control_cut=_cut(at=4.0, sequence=30),
        activation_scope=next_activation,
        requested_effective_from=4.0,
    ) is None
    with pytest.raises(ValueError, match="another residual anchor"):
        window.offer(
            _pair(
                receipt,
                index=2,
                sampled_at=3.6,
                available_at=3.61,
                admission_sequence=32,
            )
        )


def test_stale_latest_pair_produces_no_basis() -> None:
    activation = _activation()
    receipt = _anchor(activation)
    window = _window()
    window.bind_anchor(receipt)
    for index, sampled_at in enumerate((2.8, 2.9)):
        window.offer(
            _pair(
                receipt,
                index=index,
                sampled_at=sampled_at,
                available_at=sampled_at + 0.01,
                admission_sequence=10 + 2 * index,
            )
        )

    assert window.capture_basis(
        allocation_scope_id="operator-allocation",
        control_cut=_cut(at=3.14, sequence=20),
        activation_scope=activation,
        requested_effective_from=3.15,
    ) is None


def test_high_rate_retention_is_bounded_by_time_not_pair_count() -> None:
    activation = _activation()
    receipt = _anchor(activation)
    window = _window()
    window.bind_anchor(receipt)
    for index in range(100):
        sampled_at = 2.6 + index * 0.01
        window.offer(
            _pair(
                receipt,
                index=index,
                sampled_at=sampled_at,
                available_at=sampled_at + 0.001,
                admission_sequence=10 + 2 * index,
            )
        )

    snapshot = window.snapshot()
    assert 50 <= snapshot.retained_pair_count <= 51
    assert snapshot.dropped_pair_count >= 49

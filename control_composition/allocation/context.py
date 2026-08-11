"""Adapt allocation decisions to the formal action/context cut channel.

The evidence basis cut precedes context admission.  Once admitted, the context
may be selected together with action inputs by a later formal composition cut;
it never rewrites the cut that produced the decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash

from control_composition.activation import CompositionActivationScope
from control_composition.composition.context import (
    CompositionContextDraft,
    CompositionContextRecord,
)

from .contracts import (
    COMPOSITION_ALLOCATION_MODES_DEVELOPMENT,
    COMPOSITION_ALLOCATION_STATUS_READY,
)
from .decision import CompositionAllocationDecision


COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT = canonical_content_hash(
    {
        "contract": "composition_allocation_decision_context",
        "version": "development",
        "semantics": (
            "non_authoritative_component_mode_selected_before_composition",
            "motion_and_anchor_relative_magnitude_are_distinct_evidence",
            "all_components_share_one_residual_anchor_receipt",
            "optional_receipt_driven_publication_basis_and_activation_binding",
        ),
    },
    domain="dexteleop.composition_allocation_context_contract.development",
)


@dataclass(frozen=True)
class AllocationResidualAnchorBinding:
    """Exact residual anchor selected by one validated allocation context."""

    receipt_id: str
    receipt_hash: str
    request_generation: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "receipt_id",
            _required(self.receipt_id, name="receipt_id"),
        )
        object.__setattr__(
            self,
            "receipt_hash",
            _required(self.receipt_hash, name="receipt_hash"),
        )
        generation = self.request_generation
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError("request_generation must be non-negative.")


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def allocation_decision_context_payload(
    decision: CompositionAllocationDecision,
    *,
    allocation_scope_id: str | None = None,
    activity_basis_id: str | None = None,
    activity_basis_hash: str | None = None,
    control_cut_hash: str | None = None,
    activation_scope: CompositionActivationScope | None = None,
) -> dict[str, object]:
    """Return the complete actual-data payload required by the strategy."""

    if not isinstance(decision, CompositionAllocationDecision):
        raise TypeError("decision must be a CompositionAllocationDecision.")
    if decision.output_status != COMPOSITION_ALLOCATION_STATUS_READY:
        raise ValueError("Only a ready allocation decision can become context.")
    publication_fields = (
        allocation_scope_id,
        activity_basis_id,
        activity_basis_hash,
        control_cut_hash,
        activation_scope,
    )
    publication_bound = any(value is not None for value in publication_fields)
    if publication_bound and any(value is None for value in publication_fields):
        raise ValueError(
            "Allocation publication binding must provide scope, basis, cut, "
            "and activation together."
        )
    if activation_scope is not None and not isinstance(
        activation_scope, CompositionActivationScope
    ):
        raise TypeError("activation_scope must be CompositionActivationScope.")
    if (
        activation_scope is not None
        and activation_scope.clock_domain_id != decision.clock_domain_id
    ):
        raise ValueError("Allocation decision and activation use another clock.")
    payload: dict[str, object] = {
        "decision_id": decision.decision_id,
        "decision_hash": decision.semantic_content_hash,
        "allocation_spec_hash": decision.allocation_spec_hash,
        "evaluated_at": decision.evaluated_at,
        "clock_domain_id": decision.clock_domain_id,
        "selected_mode_by_component": dict(decision.selected_mode_by_component),
        "valid_until_by_component": {
            selection.component_id: selection.valid_until
            for selection in decision.selections
        },
        "motion_observation_id_by_component": {
            selection.component_id: selection.motion_observation_id
            for selection in decision.selections
        },
        "motion_observation_hash_by_component": {
            selection.component_id: selection.motion_observation_hash
            for selection in decision.selections
        },
        "residual_magnitude_observation_id_by_component": {
            selection.component_id: selection.residual_magnitude_observation_id
            for selection in decision.selections
        },
        "residual_magnitude_observation_hash_by_component": {
            selection.component_id: (
                selection.residual_magnitude_observation_hash
            )
            for selection in decision.selections
        },
        "anchor_receipt_id_by_component": {
            selection.component_id: selection.anchor_receipt_id
            for selection in decision.selections
        },
        "anchor_receipt_hash_by_component": {
            selection.component_id: selection.anchor_receipt_hash
            for selection in decision.selections
        },
        "anchor_request_generation_by_component": {
            selection.component_id: selection.anchor_request_generation
            for selection in decision.selections
        },
        "authoritative": False,
    }
    if decision.allocation_override_state_id is not None:
        payload.update(
            {
                "allocation_override_state_id": (
                    decision.allocation_override_state_id
                ),
                "allocation_override_state_hash": (
                    decision.allocation_override_state_hash
                ),
            }
        )
    if publication_bound:
        assert activation_scope is not None
        payload.update(
            {
                "allocation_scope_id": _required(
                    allocation_scope_id,
                    name="allocation_scope_id",
                ),
                "activity_basis_id": _required(
                    activity_basis_id,
                    name="activity_basis_id",
                ),
                "activity_basis_hash": _required(
                    activity_basis_hash,
                    name="activity_basis_hash",
                ),
                "control_cut_hash": _required(
                    control_cut_hash,
                    name="control_cut_hash",
                ),
                "composition_activation_id": activation_scope.activation_id,
                "composition_activation_hash": (
                    activation_scope.semantic_content_hash
                ),
            }
        )
    return payload


def build_composition_allocation_context_draft(
    decision: CompositionAllocationDecision,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    producer_id: str,
    producer_epoch: int,
    sequence: int,
    context_channel: str,
    produced_at: float,
    produced_clock_domain_id: str,
    effective_from: float,
    effective_clock_domain_id: str,
    support_until: float | None = None,
    allocation_scope_id: str | None = None,
    activity_basis_id: str | None = None,
    activity_basis_hash: str | None = None,
    control_cut_hash: str | None = None,
    activation_scope: CompositionActivationScope | None = None,
) -> CompositionContextDraft:
    """Publish a ready decision through the ordinary context admission seam."""

    payload = allocation_decision_context_payload(
        decision,
        allocation_scope_id=allocation_scope_id,
        activity_basis_id=activity_basis_id,
        activity_basis_hash=activity_basis_hash,
        control_cut_hash=control_cut_hash,
        activation_scope=activation_scope,
    )
    valid_until = min(selection.valid_until for selection in decision.selections)
    if support_until is not None:
        valid_until = min(
            valid_until,
            _finite(support_until, name="support_until"),
        )
    start = _finite(effective_from, name="effective_from")
    produced = _finite(produced_at, name="produced_at")
    produced_clock = _required(
        produced_clock_domain_id,
        name="produced_clock_domain_id",
    )
    effective_clock = _required(
        effective_clock_domain_id,
        name="effective_clock_domain_id",
    )
    if produced_clock != decision.clock_domain_id or effective_clock != (
        decision.clock_domain_id
    ):
        raise ValueError(
            "Allocation decision and context draft must share one mapped clock."
        )
    if produced < decision.evaluated_at:
        raise ValueError("Allocation context cannot predate its decision.")
    if start < decision.evaluated_at:
        raise ValueError(
            "Allocation context effective_from cannot predate its decision."
        )
    if valid_until <= start:
        raise ValueError("Allocation decision expires before effective_from.")
    publication_bound = activity_basis_id is not None
    source_fact_ids = (decision.decision_id,)
    source_fact_hashes = (decision.semantic_content_hash,)
    if publication_bound:
        source_fact_ids += (_required(activity_basis_id, name="activity_basis_id"),)
        source_fact_hashes += (
            _required(activity_basis_hash, name="activity_basis_hash"),
        )
    return CompositionContextDraft(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        producer_id=producer_id,
        producer_epoch=producer_epoch,
        sequence=sequence,
        context_channel=context_channel,
        context_kind="state",
        semantic_payload=payload,
        produced_at=produced,
        produced_clock_domain_id=produced_clock,
        effective_from=start,
        effective_until=valid_until,
        effective_clock_domain_id=effective_clock,
        payload_contract_hash=(
            COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT
        ),
        source_fact_ids=source_fact_ids,
        source_fact_hashes=source_fact_hashes,
    )


def selected_allocation_modes_from_context(
    context: CompositionContextRecord,
    *,
    expected_allocation_spec_hash: str,
    expected_component_ids: tuple[str, ...],
    target_effective_at: float,
    expected_composition_activation_id: str | None = None,
    expected_composition_activation_hash: str | None = None,
) -> Mapping[str, str]:
    """Validate and parse one admitted allocation decision context."""

    if not isinstance(context, CompositionContextRecord):
        raise TypeError("context must be a CompositionContextRecord.")
    if (
        context.payload_contract_hash
        != COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT
    ):
        raise ValueError("Allocation context uses another payload contract.")
    payload = dict(context.semantic_payload)
    if payload.get("authoritative") is not False:
        raise ValueError("Allocation context cannot claim authority.")
    if payload.get("allocation_spec_hash") != _required(
        expected_allocation_spec_hash,
        name="expected_allocation_spec_hash",
    ):
        raise ValueError("Allocation context belongs to another spec.")
    decision_id = _required(payload.get("decision_id"), name="decision_id")
    decision_hash = _required(payload.get("decision_hash"), name="decision_hash")
    override_id = payload.get("allocation_override_state_id")
    override_hash = payload.get("allocation_override_state_hash")
    if (override_id is None) != (override_hash is None):
        raise ValueError("Allocation override context identity is incomplete.")
    if override_id is not None:
        _required(override_id, name="allocation_override_state_id")
        _required(override_hash, name="allocation_override_state_hash")
    expected_activation_pair = (
        expected_composition_activation_id,
        expected_composition_activation_hash,
    )
    if (expected_activation_pair[0] is None) != (
        expected_activation_pair[1] is None
    ):
        raise ValueError("Expected composition activation identity is incomplete.")
    payload_activation_pair = (
        payload.get("composition_activation_id"),
        payload.get("composition_activation_hash"),
    )
    if (payload_activation_pair[0] is None) != (payload_activation_pair[1] is None):
        raise ValueError("Allocation context activation identity is incomplete.")
    if expected_activation_pair != payload_activation_pair:
        raise ValueError("Allocation context belongs to another activation.")
    activity_basis_id = payload.get("activity_basis_id")
    activity_basis_hash = payload.get("activity_basis_hash")
    publication_values = (
        payload.get("allocation_scope_id"),
        activity_basis_id,
        activity_basis_hash,
        payload.get("control_cut_hash"),
        payload_activation_pair[0],
        payload_activation_pair[1],
    )
    publication_bound = any(value is not None for value in publication_values)
    if publication_bound and any(value is None for value in publication_values):
        raise ValueError("Allocation context publication binding is incomplete.")
    expected_lineage_ids = (decision_id,)
    expected_lineage_hashes = (decision_hash,)
    if publication_bound:
        expected_lineage_ids += (
            _required(activity_basis_id, name="activity_basis_id"),
        )
        expected_lineage_hashes += (
            _required(activity_basis_hash, name="activity_basis_hash"),
        )
        _required(payload.get("allocation_scope_id"), name="allocation_scope_id")
        _required(payload.get("control_cut_hash"), name="control_cut_hash")
    if context.source_fact_ids != expected_lineage_ids or (
        context.source_fact_hashes != expected_lineage_hashes
    ):
        raise ValueError("Allocation context decision lineage mismatch.")
    payload_clock = _required(
        payload.get("clock_domain_id"),
        name="allocation_clock_domain_id",
    )
    evaluated_at = _finite(payload.get("evaluated_at"), name="evaluated_at")
    if payload_clock != context.available_clock_domain_id:
        raise ValueError("Allocation context uses another mapped clock domain.")
    if evaluated_at > context.available_at:
        raise ValueError("Allocation decision was not available at admission.")
    expected = tuple(
        _required(value, name="expected_component_id")
        for value in expected_component_ids
    )
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("expected_component_ids must be non-empty and unique.")
    raw_modes = dict(payload.get("selected_mode_by_component", {}))
    raw_valid_until = dict(payload.get("valid_until_by_component", {}))
    raw_motion_ids = dict(payload.get("motion_observation_id_by_component", {}))
    raw_motion_hashes = dict(
        payload.get("motion_observation_hash_by_component", {})
    )
    raw_magnitude_ids = dict(
        payload.get("residual_magnitude_observation_id_by_component", {})
    )
    raw_magnitude_hashes = dict(
        payload.get("residual_magnitude_observation_hash_by_component", {})
    )
    raw_anchor_ids = dict(payload.get("anchor_receipt_id_by_component", {}))
    raw_anchor_hashes = dict(
        payload.get("anchor_receipt_hash_by_component", {})
    )
    raw_anchor_generations = dict(
        payload.get("anchor_request_generation_by_component", {})
    )
    if any(
        set(values) != set(expected)
        for values in (
            raw_modes,
            raw_valid_until,
            raw_motion_ids,
            raw_motion_hashes,
            raw_magnitude_ids,
            raw_magnitude_hashes,
            raw_anchor_ids,
            raw_anchor_hashes,
            raw_anchor_generations,
        )
    ):
        raise ValueError(
            "Allocation context must cover the expected component set exactly."
        )
    for component_id in expected:
        for name, values in (
            ("motion_observation_id", raw_motion_ids),
            ("motion_observation_hash", raw_motion_hashes),
            ("residual_magnitude_observation_id", raw_magnitude_ids),
            ("residual_magnitude_observation_hash", raw_magnitude_hashes),
            ("anchor_receipt_id", raw_anchor_ids),
            ("anchor_receipt_hash", raw_anchor_hashes),
        ):
            _required(values[component_id], name=f"{name}[{component_id!r}]")
        generation = raw_anchor_generations[component_id]
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise ValueError(
                "anchor_request_generation must be a non-negative integer."
            )
    if len(
        {
            (
                raw_anchor_ids[component_id],
                raw_anchor_hashes[component_id],
                raw_anchor_generations[component_id],
            )
            for component_id in expected
        }
    ) != 1:
        raise ValueError(
            "Allocation context components must share one residual anchor."
        )
    modes = {
        component_id: _required(
            raw_modes[component_id],
            name=f"selected_mode[{component_id!r}]",
        )
        for component_id in expected
    }
    if any(
        mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT for mode in modes.values()
    ):
        raise ValueError("Allocation context contains an unsupported mode.")
    target = _finite(target_effective_at, name="target_effective_at")
    if evaluated_at > target:
        raise ValueError(
            "Allocation decision was evaluated after the composition target."
        )
    if any(
        target
        >= _finite(
            raw_valid_until[component_id],
            name=f"valid_until[{component_id!r}]",
        )
        for component_id in expected
    ):
        raise ValueError("Allocation context expired before the target.")
    return MappingProxyType(modes)


def selected_allocation_anchor_from_context(
    context: CompositionContextRecord,
    *,
    expected_allocation_spec_hash: str,
    expected_component_ids: tuple[str, ...],
    target_effective_at: float,
    expected_composition_activation_id: str | None = None,
    expected_composition_activation_hash: str | None = None,
) -> AllocationResidualAnchorBinding:
    """Return the exact anchor after validating the complete context contract."""

    selected_allocation_modes_from_context(
        context,
        expected_allocation_spec_hash=expected_allocation_spec_hash,
        expected_component_ids=expected_component_ids,
        target_effective_at=target_effective_at,
        expected_composition_activation_id=(
            expected_composition_activation_id
        ),
        expected_composition_activation_hash=(
            expected_composition_activation_hash
        ),
    )
    payload = context.semantic_payload
    component_id = expected_component_ids[0]
    return AllocationResidualAnchorBinding(
        receipt_id=payload["anchor_receipt_id_by_component"][component_id],
        receipt_hash=payload["anchor_receipt_hash_by_component"][component_id],
        request_generation=(
            payload["anchor_request_generation_by_component"][component_id]
        ),
    )


__all__ = [
    "COMPOSITION_ALLOCATION_CONTEXT_CONTRACT_HASH_DEVELOPMENT",
    "AllocationResidualAnchorBinding",
    "allocation_decision_context_payload",
    "build_composition_allocation_context_draft",
    "selected_allocation_modes_from_context",
    "selected_allocation_anchor_from_context",
]

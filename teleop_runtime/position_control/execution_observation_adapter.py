"""Projection from the current position resampler observer to shared facts.

The current observer runs before resolver/planner processing.  Its projection
is therefore a stage-qualified point sample, never an active backend reference
interval.
"""

from __future__ import annotations

from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ExecutionFactReference,
    ExecutionReferenceFrameObservation,
)

from .reference_observation import LegacyReferenceDerivation, LegacyReferenceSourceFact


POSITION_PRE_RESOLVER_REFERENCE_STAGE = "position.pre_resolver_reference"
POSITION_PRE_RESOLVER_SOURCE_HASH_DOMAIN = (
    "dexteleop.position_pre_resolver_source_reference.development"
)


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _non_negative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _side_component_mapping(
    side_to_component_id: Mapping[str, str],
    *,
    expected_sides: tuple[str, ...],
) -> dict[str, str]:
    normalized = {
        _required(side, name="side"): _required(
            component_id,
            name=f"component_id[{side!r}]",
        )
        for side, component_id in side_to_component_id.items()
    }
    if (
        set(normalized) != set(expected_sides)
        or len(set(normalized.values())) != len(normalized)
    ):
        raise ValueError(
            "side_to_component_id must map every reference side to one "
            "unique component ID."
        )
    return normalized


def _source_parent(
    source: LegacyReferenceSourceFact,
    *,
    role: str,
    modality: str,
    observed_at: float,
    clock_domain_id: str,
) -> ExecutionFactReference:
    payload = {
        "role": role,
        "modality": modality,
        "source_index": source.source_index,
        "source_timestamp": source.source_timestamp,
        "source_stage": source.source_stage,
        "target_payload_hash": source.target_payload_hash,
    }
    fact_hash = canonical_content_hash(
        payload,
        domain="dexteleop.position_resampler_parent.development",
    )
    return ExecutionFactReference(
        stage=f"position.resampler_{role}_parent",
        fact_id=f"position-resampler-parent:{fact_hash}",
        fact_hash=fact_hash,
        observed_at=observed_at,
        clock_domain_id=clock_domain_id,
        status=source.source_stage,
    )


def project_position_pre_resolver_reference_frame(
    derivation: LegacyReferenceDerivation,
    *,
    control_domain_id: str,
    selection_group_id: str,
    execution_fence_sequence: int,
    reference_revision: int,
    target_fact_id: str,
    target_fact_hash: str,
    clock_domain_id: str,
    side_to_component_id: Mapping[str, str],
    component_schema_hash: str,
) -> ExecutionReferenceFrameObservation:
    """Project the exact resampler object without claiming backend activation."""

    if not isinstance(derivation, LegacyReferenceDerivation):
        raise TypeError("derivation must be a LegacyReferenceDerivation.")
    reference = derivation.reference
    if derivation.modality == "pose":
        values_by_side = reference.pose_by_side
        representation = "cartesian_pose"
    elif derivation.modality == "qpos":
        values_by_side = reference.qpos_by_side
        representation = "joint_position"
    else:  # pragma: no cover - LegacyReferenceDerivation validates this.
        raise ValueError("Unsupported reference modality.")
    sides = tuple(values_by_side)
    mapping = _side_component_mapping(
        side_to_component_id,
        expected_sides=sides,
    )
    schema_hash = _required(
        component_schema_hash,
        name="component_schema_hash",
    )
    component_values = {
        mapping[side]: tuple(float(value) for value in values_by_side[side])
        for side in sides
    }
    source_payload = {
        "legacy_worker_generation": derivation.legacy_worker_generation,
        "modality": derivation.modality,
        "reference_sequence": reference.sequence,
        "reference_timestamp": reference.timestamp,
        "left_source_index": reference.left_source_index,
        "right_source_index": reference.right_source_index,
        "left_source_timestamp": reference.left_source_timestamp,
        "right_source_timestamp": reference.right_source_timestamp,
        "interpolation_alpha": reference.interpolation_alpha,
        "component_values": component_values,
        "interpolation_contract_hash": derivation.interpolation_contract_hash,
    }
    source_hash = canonical_content_hash(
        source_payload,
        domain=POSITION_PRE_RESOLVER_SOURCE_HASH_DOMAIN,
    )
    source_id = f"position-pre-resolver-reference:{source_hash}"
    parents = (
        _source_parent(
            derivation.start_source_fact,
            role="start",
            modality=derivation.modality,
            observed_at=derivation.emitted_at,
            clock_domain_id=clock_domain_id,
        ),
        _source_parent(
            derivation.end_source_fact,
            role="end",
            modality=derivation.modality,
            observed_at=derivation.emitted_at,
            clock_domain_id=clock_domain_id,
        ),
    )
    return ExecutionReferenceFrameObservation(
        reference_frame_id=f"reference-frame:{source_hash}",
        reference_stage_id=POSITION_PRE_RESOLVER_REFERENCE_STAGE,
        source_reference_id=source_id,
        source_reference_hash=source_hash,
        target_fact_id=_required(target_fact_id, name="target_fact_id"),
        target_fact_hash=_required(target_fact_hash, name="target_fact_hash"),
        control_domain_id=_required(
            control_domain_id,
            name="control_domain_id",
        ),
        selection_group_id=_required(
            selection_group_id,
            name="selection_group_id",
        ),
        execution_fence_sequence=_non_negative_int(
            execution_fence_sequence,
            name="execution_fence_sequence",
        ),
        reference_revision=_non_negative_int(
            reference_revision,
            name="reference_revision",
        ),
        reference_sequence=reference.sequence,
        temporal_semantics="point_sample",
        scheduled_for=derivation.scheduled_for,
        generated_at=derivation.emitted_at,
        effective_from=None,
        effective_until=None,
        available_at=derivation.emitted_at,
        clock_domain_id=_required(
            clock_domain_id,
            name="clock_domain_id",
        ),
        value_by_component=component_values,
        representation_by_component={
            component_id: representation for component_id in component_values
        },
        schema_hash_by_component={
            component_id: schema_hash for component_id in component_values
        },
        interpolation_contract_hash=derivation.interpolation_contract_hash,
        parent_facts=parents,
    )


__all__ = [
    "POSITION_PRE_RESOLVER_REFERENCE_STAGE",
    "POSITION_PRE_RESOLVER_SOURCE_HASH_DOMAIN",
    "project_position_pre_resolver_reference_frame",
]

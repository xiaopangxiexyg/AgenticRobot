"""Passive MuJoCo adapter for reference-to-measurement differences.

This module translates MuJoCo-owned reference/measurement facts into the
backend-neutral provisional observation contracts.  Nothing here is called by
the production control path unless a consumer explicitly opts in.
"""

from __future__ import annotations

import math
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from control_composition.composition import (
    CompositionInputSnapshot,
    CompositionEvaluation,
)
from control_composition.observability import (
    DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
    DifferenceObservationEnvelope,
    MeasurementReferenceAssociation,
)
from robot_motion.action import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
    ExecutionReferenceFrameObservation,
    ReferenceSegment,
    ReferenceSegmentAdmissionReceipt,
    ReferenceSegmentTerminal,
)

from ..adapters import MujocoReferenceTargetActivationObservation
from .measurement import MujocoQposMeasurement


MUJOCO_QPOS_EXECUTION_DIFFERENCE_CAPABILITY_ID = (
    "sim.mujoco.qpos-reference-to-measured.provisional"
)
MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_ID = (
    "dexteleop.sim.mujoco.qpos_execution_difference.provisional"
)
MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_HASH = canonical_content_hash(
    {
        "difference_scope": "reference_to_measured",
        "association": "actual_effective_interval_at_measured_at",
        "representation": "joint_position",
        "component_topology": "component_keyed",
        "cross_clock_mapping": False,
        "authoritative": False,
    },
    domain=MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_ID,
)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _side_component_mapping(
    side_to_component_id: Mapping[str, str],
    *,
    expected_sides: tuple[str, ...],
) -> dict[str, str]:
    normalized = {
        str(side).strip(): str(component_id).strip()
        for side, component_id in side_to_component_id.items()
    }
    if (
        set(normalized) != set(expected_sides)
        or any(not side or not component_id for side, component_id in normalized.items())
        or len(set(normalized.values())) != len(normalized)
    ):
        raise ValueError(
            "side_to_component_id must map every measurement/reference side "
            "to one unique non-empty component ID."
        )
    return normalized


def _fact_reference(
    *,
    stage: str,
    fact_id: str,
    fact_hash: str,
    observed_at: float,
    clock_domain_id: str,
    status: str,
) -> ExecutionFactReference:
    return ExecutionFactReference(
        stage=stage,
        fact_id=fact_id,
        fact_hash=fact_hash,
        observed_at=observed_at,
        clock_domain_id=clock_domain_id,
        status=status,
    )


def project_mujoco_execution_reference_frame(
    *,
    segment: ReferenceSegment,
    admission: ReferenceSegmentAdmissionReceipt,
    activation: MujocoReferenceTargetActivationObservation,
    side_to_component_id: Mapping[str, str],
    component_schema_hash: str,
    terminal: ReferenceSegmentTerminal | None = None,
    observation_binding_id: str | None = None,
    observation_plan_hash: str | None = None,
) -> ExecutionReferenceFrameObservation:
    """Project one actually activated qref and its authoritative lineage."""

    if not isinstance(segment, ReferenceSegment):
        raise TypeError("segment must be a ReferenceSegment.")
    if not isinstance(admission, ReferenceSegmentAdmissionReceipt):
        raise TypeError("admission must be a ReferenceSegmentAdmissionReceipt.")
    if not isinstance(activation, MujocoReferenceTargetActivationObservation):
        raise TypeError(
            "activation must be a MujocoReferenceTargetActivationObservation."
        )
    if not admission.accepted:
        raise ValueError("A rejected segment admission cannot produce a reference.")
    if (
        admission.reference_segment_id != segment.reference_segment_id
        or admission.reference_segment_hash != segment.semantic_content_hash
        or admission.control_domain_id != segment.control_domain_id
        or admission.selection_group_id != segment.selection_group_id
        or admission.execution_epoch != segment.execution_epoch
        or admission.group_segment_generation != segment.group_segment_generation
        or admission.clock_domain_id != segment.clock_domain_id
    ):
        raise ValueError("Reference admission does not match the segment.")
    if (
        activation.reference_segment_id != segment.reference_segment_id
        or activation.execution_epoch != segment.execution_epoch
        or activation.group_segment_generation
        != segment.group_segment_generation
    ):
        raise ValueError("Reference activation does not match the segment lineage.")
    sides = tuple(activation.qref_by_side)
    mapping = _side_component_mapping(
        side_to_component_id,
        expected_sides=sides,
    )
    component_ids = tuple(mapping[side] for side in sides)
    if any(
        component_id not in segment.component_dependencies
        for component_id in component_ids
    ):
        raise ValueError(
            "Reference components must be declared segment dependencies."
        )
    unsupported_modes = {
        side: status
        for side, status in activation.activation_status_by_side.items()
        if status not in {"tracking", "degraded"}
    }
    if unsupported_modes:
        raise ValueError(
            "Reference activation did not establish a tracking/degraded qref: "
            f"{unsupported_modes!r}."
        )

    effective_until: float | None = None
    terminal_reference: ExecutionFactReference | None = None
    if terminal is not None:
        if not isinstance(terminal, ReferenceSegmentTerminal):
            raise TypeError("terminal must be a ReferenceSegmentTerminal.")
        if (
            terminal.reference_segment_id != segment.reference_segment_id
            or terminal.reference_segment_hash != segment.semantic_content_hash
            or terminal.control_domain_id != segment.control_domain_id
            or terminal.selection_group_id != segment.selection_group_id
            or terminal.execution_epoch != segment.execution_epoch
            or terminal.group_segment_generation
            != segment.group_segment_generation
            or terminal.clock_domain_id != segment.clock_domain_id
        ):
            raise ValueError("Reference terminal does not match the segment.")
        effective_until = terminal.effective_at
        terminal_reference = _fact_reference(
            stage="reference_segment_terminal",
            fact_id=terminal.terminal_id,
            fact_hash=terminal.semantic_content_hash,
            observed_at=terminal.available_at,
            clock_domain_id=terminal.clock_domain_id,
            status=terminal.status,
        )

    references = [
        _fact_reference(
            stage="reference_segment_admission",
            fact_id=admission.admission_id,
            fact_hash=admission.semantic_content_hash,
            observed_at=admission.admitted_at,
            clock_domain_id=admission.clock_domain_id,
            status="accepted",
        ),
        _fact_reference(
            stage="reference_activation",
            fact_id=activation.observation_id,
            fact_hash=activation.semantic_content_hash,
            observed_at=activation.observed_at,
            clock_domain_id=segment.clock_domain_id,
            status="activated",
        ),
    ]
    if terminal_reference is not None:
        references.append(terminal_reference)
    schema_hash = str(component_schema_hash).strip()
    if not schema_hash:
        raise ValueError("component_schema_hash must be non-empty.")
    return ExecutionReferenceFrameObservation(
        reference_frame_id=(
            f"mujoco-reference-frame:{activation.semantic_content_hash}"
        ),
        reference_stage_id="mujoco.active_qref",
        source_reference_id=activation.observation_id,
        source_reference_hash=activation.semantic_content_hash,
        target_fact_id=segment.reference_segment_id,
        target_fact_hash=segment.semantic_content_hash,
        control_domain_id=segment.control_domain_id,
        selection_group_id=segment.selection_group_id,
        execution_fence_sequence=segment.execution_epoch,
        reference_revision=segment.group_segment_generation,
        reference_sequence=activation.reference_sequence,
        temporal_semantics="active_interval",
        scheduled_for=activation.scheduled_for,
        generated_at=activation.activated_at_worker_time,
        effective_from=activation.activated_at_worker_time,
        effective_until=effective_until,
        available_at=activation.observed_at,
        clock_domain_id=segment.clock_domain_id,
        value_by_component={
            mapping[side]: tuple(float(value) for value in activation.qref_by_side[side])
            for side in sides
        },
        representation_by_component={
            component_id: "joint_position" for component_id in component_ids
        },
        schema_hash_by_component={
            component_id: schema_hash for component_id in component_ids
        },
        interpolation_contract_hash=segment.interpolation_contract_hash,
        observation_binding_id=observation_binding_id,
        observation_plan_hash=observation_plan_hash,
        parent_facts=tuple(references),
    )


def project_mujoco_execution_reference_sample(
    *,
    segment: ReferenceSegment,
    admission: ReferenceSegmentAdmissionReceipt,
    activation: MujocoReferenceTargetActivationObservation,
    sampled_at: float,
    available_at: float,
    side_to_component_id: Mapping[str, str],
    component_schema_hash: str,
    observation_binding_id: str,
    observation_plan_hash: str,
    selected_component_ids: tuple[str, ...] | None = None,
) -> ExecutionReferenceFrameObservation:
    """Project the exact active qref at one downstream measurement cut."""

    if not isinstance(segment, ReferenceSegment):
        raise TypeError("segment must be a ReferenceSegment.")
    if not isinstance(admission, ReferenceSegmentAdmissionReceipt):
        raise TypeError("admission must be a ReferenceSegmentAdmissionReceipt.")
    if not isinstance(activation, MujocoReferenceTargetActivationObservation):
        raise TypeError(
            "activation must be a MujocoReferenceTargetActivationObservation."
        )
    sample_time = _finite(sampled_at, name="sampled_at")
    observed_time = _finite(available_at, name="available_at")
    if observed_time + 1e-12 < sample_time:
        raise ValueError("Reference sample cannot be available before sampled_at.")
    if not admission.accepted:
        raise ValueError("A rejected segment admission cannot produce a reference.")
    if (
        admission.reference_segment_id != segment.reference_segment_id
        or admission.reference_segment_hash != segment.semantic_content_hash
        or activation.reference_segment_id != segment.reference_segment_id
        or activation.execution_epoch != segment.execution_epoch
        or activation.group_segment_generation
        != segment.group_segment_generation
    ):
        raise ValueError("Reference sample lineage does not match its segment.")
    if sample_time + 1e-12 < activation.activated_at_worker_time:
        raise ValueError("Reference sample precedes qref activation.")
    unsupported_modes = {
        side: status
        for side, status in activation.activation_status_by_side.items()
        if status not in {"tracking", "degraded"}
    }
    if unsupported_modes:
        raise ValueError(
            "Reference activation did not establish a tracking/degraded qref: "
            f"{unsupported_modes!r}."
        )
    sides = tuple(activation.qref_by_side)
    mapping = _side_component_mapping(
        side_to_component_id,
        expected_sides=sides,
    )
    requested_components = (
        tuple(mapping[side] for side in sides)
        if selected_component_ids is None
        else tuple(str(value).strip() for value in selected_component_ids)
    )
    if (
        not requested_components
        or any(not value for value in requested_components)
        or len(requested_components) != len(set(requested_components))
        or not set(requested_components).issubset(mapping.values())
    ):
        raise ValueError(
            "selected_component_ids must be a non-empty subset of active qref "
            "components."
        )
    selected_sides = tuple(
        side for side in sides if mapping[side] in requested_components
    )
    component_ids = tuple(mapping[side] for side in selected_sides)
    if any(
        component_id not in segment.component_dependencies
        for component_id in component_ids
    ):
        raise ValueError(
            "Reference components must be declared segment dependencies."
        )
    schema_hash = str(component_schema_hash).strip()
    if not schema_hash:
        raise ValueError("component_schema_hash must be non-empty.")
    source_hash = canonical_content_hash(
        {
            "activation_hash": activation.semantic_content_hash,
            "sampled_at": sample_time,
            "qref_by_side": activation.qref_by_side,
        },
        domain="dexteleop.sim.mujoco.active_qref_sample.development",
    )
    return ExecutionReferenceFrameObservation(
        reference_frame_id=f"mujoco-reference-sample:{source_hash}",
        reference_stage_id="mujoco.active_qref_at_measurement",
        source_reference_id=activation.observation_id,
        source_reference_hash=activation.semantic_content_hash,
        target_fact_id=segment.reference_segment_id,
        target_fact_hash=segment.semantic_content_hash,
        control_domain_id=segment.control_domain_id,
        selection_group_id=segment.selection_group_id,
        execution_fence_sequence=segment.execution_epoch,
        reference_revision=segment.group_segment_generation,
        reference_sequence=activation.reference_sequence,
        temporal_semantics="point_sample",
        scheduled_for=sample_time,
        generated_at=sample_time,
        effective_from=None,
        effective_until=None,
        available_at=observed_time,
        clock_domain_id=segment.clock_domain_id,
        value_by_component={
            mapping[side]: tuple(
                float(value) for value in activation.qref_by_side[side]
            )
            for side in selected_sides
        },
        representation_by_component={
            component_id: "joint_position" for component_id in component_ids
        },
        schema_hash_by_component={
            component_id: schema_hash for component_id in component_ids
        },
        interpolation_contract_hash=segment.interpolation_contract_hash,
        observation_binding_id=observation_binding_id,
        observation_plan_hash=observation_plan_hash,
        parent_facts=(
            _fact_reference(
                stage="reference_segment_admission",
                fact_id=admission.admission_id,
                fact_hash=admission.semantic_content_hash,
                observed_at=admission.admitted_at,
                clock_domain_id=admission.clock_domain_id,
                status="accepted",
            ),
            _fact_reference(
                stage="reference_activation",
                fact_id=activation.observation_id,
                fact_hash=activation.semantic_content_hash,
                observed_at=activation.observed_at,
                clock_domain_id=segment.clock_domain_id,
                status="activated",
            ),
        ),
    )


def project_mujoco_execution_measurement_observation(
    measurement: MujocoQposMeasurement,
    *,
    control_domain_id: str,
    side_to_component_id: Mapping[str, str],
    component_schema_hash: str,
    available_at: float | None = None,
    observation_binding_id: str | None = None,
    observation_plan_hash: str | None = None,
    selected_component_ids: tuple[str, ...] | None = None,
) -> ExecutionMeasurementObservation:
    """Project one raw MuJoCo measurement without inventing execution lineage."""

    if not isinstance(measurement, MujocoQposMeasurement):
        raise TypeError("measurement must be a MujocoQposMeasurement.")
    sides = tuple(measurement.qpos_by_side)
    mapping = _side_component_mapping(
        side_to_component_id,
        expected_sides=sides,
    )
    requested_components = (
        tuple(mapping[side] for side in sides)
        if selected_component_ids is None
        else tuple(str(value).strip() for value in selected_component_ids)
    )
    if (
        not requested_components
        or any(not value for value in requested_components)
        or len(requested_components) != len(set(requested_components))
        or not set(requested_components).issubset(mapping.values())
    ):
        raise ValueError(
            "selected_component_ids must be a non-empty subset of measurement "
            "components."
        )
    selected_sides = tuple(
        side for side in sides if mapping[side] in requested_components
    )
    component_ids = tuple(mapping[side] for side in selected_sides)
    schema_hash = str(component_schema_hash).strip()
    if not schema_hash:
        raise ValueError("component_schema_hash must be non-empty.")
    observed = (
        measurement.measured_at
        if available_at is None
        else _finite(available_at, name="available_at")
    )
    measurement_reference = _fact_reference(
        stage="measurement",
        fact_id=measurement.measurement_id,
        fact_hash=measurement.semantic_content_hash,
        observed_at=observed,
        clock_domain_id=measurement.clock_domain_id,
        status="measured",
    )
    return ExecutionMeasurementObservation(
        measurement_id=measurement.measurement_id,
        source_measurement_hash=measurement.semantic_content_hash,
        control_domain_id=str(control_domain_id).strip(),
        measured_at=measurement.measured_at,
        available_at=observed,
        clock_domain_id=measurement.clock_domain_id,
        value_by_component={
            mapping[side]: measurement.qpos_by_side[side]
            for side in selected_sides
        },
        representation_by_component={
            component_id: "joint_position" for component_id in component_ids
        },
        schema_hash_by_component={
            component_id: schema_hash for component_id in component_ids
        },
        observation_binding_id=observation_binding_id,
        observation_plan_hash=observation_plan_hash,
        parent_facts=(measurement_reference,),
    )


def build_mujoco_qpos_execution_difference_observation(
    *,
    input_snapshot: CompositionInputSnapshot,
    composition_evaluation: CompositionEvaluation,
    association: MeasurementReferenceAssociation,
    observed_at: float,
) -> DifferenceObservationEnvelope:
    """Wrap one passive association in the shared post-composition envelope."""

    if not isinstance(input_snapshot, CompositionInputSnapshot):
        raise TypeError("input_snapshot must be a CompositionInputSnapshot.")
    if not isinstance(composition_evaluation, CompositionEvaluation):
        raise TypeError("composition_evaluation must be a CompositionEvaluation.")
    if not isinstance(association, MeasurementReferenceAssociation):
        raise TypeError(
            "association must be a MeasurementReferenceAssociation."
        )
    if association.control_domain_id != input_snapshot.control_domain_id:
        raise ValueError(
            "Execution-difference association control domain does not match "
            "the composition input."
        )
    status = "available" if association.status == "available" else "unavailable"
    return DifferenceObservationEnvelope(
        module_id=DIFFERENCE_MODULE_POST_COMPOSITION_EXECUTION,
        capability_id=MUJOCO_QPOS_EXECUTION_DIFFERENCE_CAPABILITY_ID,
        status=status,
        reason=association.reason,
        control_domain_id=input_snapshot.control_domain_id,
        clock_domain_id=association.clock_domain_id,
        cutoff_at=input_snapshot.cutoff_at,
        cutoff_admission_sequence=input_snapshot.cutoff_admission_sequence,
        target_effective_at=input_snapshot.target_effective_at,
        observed_at=_finite(observed_at, name="observed_at"),
        valid_until=None,
        input_snapshot_id=input_snapshot.composition_input_snapshot_id,
        input_snapshot_hash=input_snapshot.semantic_content_hash,
        composition_evaluation_id=composition_evaluation.composition_evaluation_id,
        composition_evaluation_hash=composition_evaluation.semantic_content_hash,
        payload_contract_id=(
            MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_ID
        ),
        payload_contract_hash=(
            MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_HASH
        ),
        payload={
            "difference_scope": "reference_to_measured",
            "association_id": association.association_id,
            "association_hash": association.semantic_content_hash,
            "association_status": association.status,
            "measurement_id": association.measurement_id,
            "measurement_hash": association.measurement_hash,
            "reference_frame_id": association.reference_frame_id,
            "reference_frame_hash": association.reference_frame_hash,
            "reference_stage_id": association.reference_stage_id,
            "measured_at": association.measured_at,
            "execution_fence_sequence": association.execution_fence_sequence,
            "reference_revision": association.reference_revision,
            "component_difference": association.component_difference,
        },
        receipt_references=association.receipt_references,
    )


__all__ = [
    "MUJOCO_QPOS_EXECUTION_DIFFERENCE_CAPABILITY_ID",
    "MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_HASH",
    "MUJOCO_QPOS_EXECUTION_DIFFERENCE_PAYLOAD_CONTRACT_ID",
    "build_mujoco_qpos_execution_difference_observation",
    "project_mujoco_execution_measurement_observation",
    "project_mujoco_execution_reference_frame",
    "project_mujoco_execution_reference_sample",
]

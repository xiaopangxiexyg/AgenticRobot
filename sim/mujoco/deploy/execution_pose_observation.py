"""Typed FK pose projections for MuJoCo Reference and Measurement facts."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
    ExecutionReferenceFrameObservation,
)

from .pose_kinematics import MujocoPoseObservationProjector


MUJOCO_FK_POSE_REFERENCE_STAGE_ID = (
    "mujoco.active_qref_fk_pose_at_measurement"
)
MUJOCO_FK_POSE_MEASUREMENT_STAGE_ID = "mujoco.measured_qpos_fk_pose"
MUJOCO_FK_POSE_REFERENCE_SOURCE_HASH_DOMAIN = (
    "dexteleop.sim.mujoco.fk_pose_reference_source.development"
)
MUJOCO_FK_POSE_MEASUREMENT_SOURCE_HASH_DOMAIN = (
    "dexteleop.sim.mujoco.fk_pose_measurement_source.development"
)


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


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


def _pose_parent_facts(
    source_observation: ExecutionReferenceFrameObservation
    | ExecutionMeasurementObservation,
    projector: MujocoPoseObservationProjector,
    *,
    projected_at: float,
) -> tuple[ExecutionFactReference, ...]:
    if isinstance(source_observation, ExecutionReferenceFrameObservation):
        source_stage = source_observation.reference_stage_id
        source_id = source_observation.reference_frame_id
    else:
        source_stage = "mujoco.measured_qpos"
        source_id = source_observation.measurement_id
    values = (
        *source_observation.parent_facts,
        _fact_reference(
            stage=source_stage,
            fact_id=source_id,
            fact_hash=source_observation.semantic_content_hash,
            observed_at=source_observation.available_at,
            clock_domain_id=source_observation.clock_domain_id,
            status="observed",
        ),
        _fact_reference(
            stage="mujoco.fk_projection_contract",
            fact_id=projector.contract_id,
            fact_hash=projector.semantic_content_hash,
            observed_at=projected_at,
            clock_domain_id=source_observation.clock_domain_id,
            status="applied",
        ),
    )
    unique: dict[str, ExecutionFactReference] = {}
    for value in values:
        unique.setdefault(value.semantic_content_hash, value)
    return tuple(unique.values())


def project_mujoco_fk_pose_reference_observation(
    source: ExecutionReferenceFrameObservation,
    *,
    projector: MujocoPoseObservationProjector,
    model_qpos: Sequence[float],
    qpos_by_source: Mapping[str, Sequence[float]],
    component_ids: Sequence[str],
    projected_at: float,
    observation_binding_id: str,
    observation_plan_hash: str,
) -> ExecutionReferenceFrameObservation:
    """Derive ``FK(active qref)`` while preserving the exact qref parent."""

    if not isinstance(source, ExecutionReferenceFrameObservation):
        raise TypeError("source must be an ExecutionReferenceFrameObservation.")
    timestamp = _finite(projected_at, name="projected_at")
    if timestamp + 1e-12 < source.available_at:
        raise ValueError("Pose reference cannot precede its qref observation.")
    requested = tuple(component_ids)
    values = projector.project(
        model_qpos=model_qpos,
        qpos_by_source=qpos_by_source,
        component_ids=requested,
    )
    schema_hashes = {
        component_id: projector.schema_hash_by_component[component_id]
        for component_id in requested
    }
    source_hash = canonical_content_hash(
        {
            "qref_observation_hash": source.semantic_content_hash,
            "projector_hash": projector.semantic_content_hash,
            "value_by_component": values,
        },
        domain=MUJOCO_FK_POSE_REFERENCE_SOURCE_HASH_DOMAIN,
    )
    return ExecutionReferenceFrameObservation(
        reference_frame_id=f"mujoco-fk-pose-reference:{source_hash}",
        reference_stage_id=MUJOCO_FK_POSE_REFERENCE_STAGE_ID,
        source_reference_id=source.reference_frame_id,
        source_reference_hash=source.semantic_content_hash,
        target_fact_id=source.target_fact_id,
        target_fact_hash=source.target_fact_hash,
        control_domain_id=source.control_domain_id,
        selection_group_id=source.selection_group_id,
        execution_fence_sequence=source.execution_fence_sequence,
        reference_revision=source.reference_revision,
        reference_sequence=source.reference_sequence,
        temporal_semantics=source.temporal_semantics,
        scheduled_for=source.scheduled_for,
        generated_at=timestamp,
        effective_from=source.effective_from,
        effective_until=source.effective_until,
        available_at=timestamp,
        clock_domain_id=source.clock_domain_id,
        value_by_component=values,
        representation_by_component={
            component_id: "cartesian_pose" for component_id in requested
        },
        schema_hash_by_component=schema_hashes,
        interpolation_contract_hash=source.interpolation_contract_hash,
        observation_binding_id=observation_binding_id,
        observation_plan_hash=observation_plan_hash,
        parent_facts=_pose_parent_facts(
            source,
            projector,
            projected_at=timestamp,
        ),
    )


def project_mujoco_fk_pose_measurement_observation(
    source: ExecutionMeasurementObservation,
    *,
    projector: MujocoPoseObservationProjector,
    model_qpos: Sequence[float],
    qpos_by_source: Mapping[str, Sequence[float]],
    component_ids: Sequence[str],
    projected_at: float,
    observation_binding_id: str,
    observation_plan_hash: str,
) -> ExecutionMeasurementObservation:
    """Derive ``FK(measured qpos)`` while preserving the raw measurement."""

    if not isinstance(source, ExecutionMeasurementObservation):
        raise TypeError("source must be an ExecutionMeasurementObservation.")
    timestamp = _finite(projected_at, name="projected_at")
    if timestamp + 1e-12 < source.available_at:
        raise ValueError("Pose measurement cannot precede its qpos observation.")
    requested = tuple(component_ids)
    values = projector.project(
        model_qpos=model_qpos,
        qpos_by_source=qpos_by_source,
        component_ids=requested,
    )
    schema_hashes = {
        component_id: projector.schema_hash_by_component[component_id]
        for component_id in requested
    }
    source_hash = canonical_content_hash(
        {
            "qpos_observation_hash": source.semantic_content_hash,
            "projector_hash": projector.semantic_content_hash,
            "value_by_component": values,
        },
        domain=MUJOCO_FK_POSE_MEASUREMENT_SOURCE_HASH_DOMAIN,
    )
    return ExecutionMeasurementObservation(
        measurement_id=f"mujoco-fk-pose-measurement:{source_hash}",
        source_measurement_hash=source.semantic_content_hash,
        control_domain_id=source.control_domain_id,
        measured_at=source.measured_at,
        available_at=timestamp,
        clock_domain_id=source.clock_domain_id,
        value_by_component=values,
        representation_by_component={
            component_id: "cartesian_pose" for component_id in requested
        },
        schema_hash_by_component=schema_hashes,
        observation_binding_id=observation_binding_id,
        observation_plan_hash=observation_plan_hash,
        parent_facts=_pose_parent_facts(
            source,
            projector,
            projected_at=timestamp,
        ),
    )


__all__ = [
    "MUJOCO_FK_POSE_MEASUREMENT_STAGE_ID",
    "MUJOCO_FK_POSE_REFERENCE_STAGE_ID",
    "project_mujoco_fk_pose_measurement_observation",
    "project_mujoco_fk_pose_reference_observation",
]

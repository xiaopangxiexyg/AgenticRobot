"""Shared stateless numerical operations for typed component strategies."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .core import CompositionInputRecord
from .formal_strategy import FormalCompositionStrategyInput


def component_record_available(
    record: CompositionInputRecord,
    *,
    component_id: str,
    representation: str,
) -> bool:
    action = record.action
    if component_id not in action.values_by_component:
        return False
    family = action.action_schema.payload_schema_by_component[
        component_id
    ].representation_family
    return family == (
        "cartesian_pose" if representation == "pose" else "joint_position"
    )


def record_matches_declared_schema(
    strategy_input: FormalCompositionStrategyInput,
    record: CompositionInputRecord,
) -> bool:
    expected = strategy_input.input_snapshot.input_action_schema_hash_by_role.get(
        record.input_role,
        strategy_input.input_snapshot.action_schema_hash,
    )
    return record.action.action_schema.semantic_content_hash == expected


def relative_pose_component_value(
    baseline: CompositionInputRecord,
    delta: CompositionInputRecord,
    *,
    component_id: str,
) -> np.ndarray:
    baseline_schema = baseline.action.action_schema.payload_schema_by_component[
        component_id
    ]
    delta_schema = delta.action.action_schema.payload_schema_by_component[component_id]
    if (
        baseline_schema.representation_family != "cartesian_pose"
        or delta_schema.representation_family != "cartesian_pose"
        or baseline_schema.value_semantics != "absolute_target"
        or delta_schema.value_semantics != "relative_delta"
    ):
        raise ValueError("relative_pose_semantics_mismatch")
    if (
        baseline_schema.reference_frame != delta_schema.reference_frame
        or baseline_schema.controlled_frame != delta_schema.controlled_frame
    ):
        raise ValueError("relative_pose_frame_mismatch")
    baseline_value = np.asarray(
        baseline.action.value(component_id).values,
        dtype=np.float64,
    )
    delta_value = np.asarray(
        delta.action.value(component_id).values,
        dtype=np.float64,
    )
    translation = baseline_value[:3] + delta_value[:3]
    quaternion = (
        Rotation.from_quat(baseline_value[3:7]) * Rotation.from_quat(delta_value[3:7])
    ).as_quat()
    if float(np.dot(quaternion, baseline_value[3:7])) < 0.0:
        quaternion = -quaternion
    return np.concatenate((translation, quaternion)).astype(baseline_schema.dtype)


def relative_qpos_component_value(
    baseline: CompositionInputRecord,
    delta: CompositionInputRecord,
    *,
    component_id: str,
    baseline_value_contract_hash: str,
    delta_value_contract_hash: str,
) -> np.ndarray:
    baseline_schema = baseline.action.action_schema.payload_schema_by_component[
        component_id
    ]
    delta_schema = delta.action.action_schema.payload_schema_by_component[component_id]
    if (
        baseline_schema.representation_family != "joint_position"
        or delta_schema.representation_family != "joint_position"
        or baseline_schema.value_semantics != "absolute_target"
        or delta_schema.value_semantics != "relative_delta"
    ):
        raise ValueError("relative_qpos_semantics_mismatch")
    if (
        baseline_schema.semantic_content_hash != baseline_value_contract_hash
        or delta_schema.semantic_content_hash != delta_value_contract_hash
    ):
        raise ValueError("relative_qpos_value_contract_mismatch")
    if (
        baseline_schema.shape != delta_schema.shape
        or baseline_schema.coordinate_names != delta_schema.coordinate_names
        or baseline_schema.units != delta_schema.units
    ):
        raise ValueError("relative_qpos_layout_mismatch")
    return (
        np.asarray(
            baseline.action.value(component_id).values,
            dtype=np.float64,
        )
        + np.asarray(
            delta.action.value(component_id).values,
            dtype=np.float64,
        )
    ).astype(baseline_schema.dtype)


__all__ = [
    "component_record_available",
    "record_matches_declared_schema",
    "relative_pose_component_value",
    "relative_qpos_component_value",
]

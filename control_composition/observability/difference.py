"""Provisional difference observations bound to one composition cut.

Differences are diagnostic facts, never motion authority.  Signed values use
``reference - compared`` semantics.  Pose orientation error is the rotation
vector of ``R_compared^-1 * R_reference`` in xyzw quaternion convention.

This transitional carrier currently contains two future module boundaries:
``pairwise_differences`` is the same-cut source-to-source immediate view;
``output_residuals`` is the post-composition view.  The fields are retained for
local compatibility while the two observation modules are developed behind
explicit capabilities.  They are not a promise that this combined shape is a
long-term public schema.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash

from ..composition import (
    CompositionInputRecord,
    CompositionInputSnapshot,
    CompositionEvaluation,
    CompositionSpec,
)
from robot_motion.action.component_action import (
    ComponentAction,
    ComponentActionValue,
)


SOURCE_DIFFERENCE_OBSERVATION_HASH_DOMAIN_V1 = (
    "dexteleop.source_difference_observation.v1"
)
SOURCE_DIFFERENCE_SPEC_HASH_DOMAIN_V1 = "dexteleop.source_difference_spec.v1"


def _required(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


def _finite(value: object, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


@dataclass(frozen=True)
class ComponentDifferenceMetric:
    component_id: str
    representation: str
    reference_frame: str | None
    controlled_frame: str | None
    translation_delta_xyz_m: tuple[float, ...]
    translation_norm_m: float | None
    rotation_delta_rotvec_rad: tuple[float, ...]
    rotation_angle_rad: float | None
    qpos_active_indices: tuple[int, ...]
    qpos_delta: tuple[float, ...]
    qpos_l2_norm: float | None
    qpos_max_abs: float | None


@dataclass(frozen=True)
class ActionDifference:
    compared_role: str
    reference_role: str
    component_metrics: tuple[ComponentDifferenceMetric, ...]
    available_at_delta_sec: float | None
    effective_from_delta_sec: float | None
    valid_for_sec: float


@dataclass(frozen=True)
class SourceDifferenceSpec:
    """Declare which composition roles have comparable target semantics."""

    spec_id: str
    composition_spec_hash: str
    compared_reference_role_pairs: tuple[tuple[str, str], ...]
    output_residual_roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "spec_id", _required(self.spec_id, name="spec_id"))
        object.__setattr__(
            self,
            "composition_spec_hash",
            _required(
                self.composition_spec_hash,
                name="composition_spec_hash",
            ),
        )
        pairs = tuple(
            (
                _required(compared, name="compared_role"),
                _required(reference, name="reference_role"),
            )
            for compared, reference in self.compared_reference_role_pairs
        )
        if len(pairs) != len(set(pairs)):
            raise ValueError("Source difference role pairs must be unique.")
        if any(compared == reference for compared, reference in pairs):
            raise ValueError("Source difference role pairs must use distinct roles.")
        residual_roles = tuple(
            _required(role, name="output_residual_role")
            for role in self.output_residual_roles
        )
        if len(residual_roles) != len(set(residual_roles)):
            raise ValueError("Source difference output residual roles must be unique.")
        if not pairs and not residual_roles:
            raise ValueError("Source difference spec must request at least one metric.")
        object.__setattr__(self, "compared_reference_role_pairs", pairs)
        object.__setattr__(self, "output_residual_roles", residual_roles)

    def validate_composition_spec(self, spec: CompositionSpec) -> None:
        if spec.semantic_content_hash != self.composition_spec_hash:
            raise ValueError(
                "Source difference spec is bound to another composition spec."
            )
        roles = set(spec.input_roles)
        referenced = {
            role
            for pair in self.compared_reference_role_pairs
            for role in pair
        } | set(self.output_residual_roles)
        if not referenced.issubset(roles):
            raise ValueError(
                "Source difference spec references an unknown composition role."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "spec_id": self.spec_id,
                "composition_spec_hash": self.composition_spec_hash,
                "compared_reference_role_pairs": (
                    self.compared_reference_role_pairs
                ),
                "output_residual_roles": self.output_residual_roles,
                "difference_direction": "reference_minus_compared",
                "pose_rotation_error": (
                    "R_compared_inverse_times_R_reference_rotvec"
                ),
            },
            domain=SOURCE_DIFFERENCE_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class SourceDifferenceObservation:
    observation_id: str
    composition_input_snapshot_id: str
    composition_input_snapshot_hash: str
    composition_evaluation_id: str
    composition_evaluation_hash: str
    composition_spec_hash: str
    source_difference_spec_hash: str
    target_effective_at: float
    evaluated_at: float
    clock_domain_id: str
    valid_until: float
    pairwise_differences: tuple[ActionDifference, ...]
    output_residuals: tuple[ActionDifference, ...]
    semantic_content_hash: str


def _pose_metric(
    component: str,
    compared: ComponentActionValue,
    reference: ComponentActionValue,
) -> ComponentDifferenceMetric:
    compared_frame = compared.payload_schema.reference_frame
    reference_frame = reference.payload_schema.reference_frame
    compared_controlled = compared.payload_schema.controlled_frame
    reference_controlled = reference.payload_schema.controlled_frame
    if (
        compared_frame != reference_frame
        or compared_controlled != reference_controlled
    ):
        raise ValueError(f"source_difference_frame_mismatch:{component}")
    compared_pose = np.asarray(
        compared.values,
        dtype=np.float64,
    )
    reference_pose = np.asarray(
        reference.values,
        dtype=np.float64,
    )
    translation = reference_pose[:3] - compared_pose[:3]
    rotation = (
        Rotation.from_quat(compared_pose[3:7]).inv()
        * Rotation.from_quat(reference_pose[3:7])
    ).as_rotvec()
    return ComponentDifferenceMetric(
        component_id=component,
        representation="pose_xyz_quat_xyzw",
        reference_frame=compared_frame,
        controlled_frame=compared_controlled,
        translation_delta_xyz_m=tuple(float(value) for value in translation),
        translation_norm_m=float(np.linalg.norm(translation)),
        rotation_delta_rotvec_rad=tuple(float(value) for value in rotation),
        rotation_angle_rad=float(np.linalg.norm(rotation)),
        qpos_active_indices=(),
        qpos_delta=(),
        qpos_l2_norm=None,
        qpos_max_abs=None,
    )


def _qpos_metric(
    component: str,
    compared_values: np.ndarray,
    reference_values: np.ndarray,
    *,
    coordinate_indices: tuple[int, ...] | None = None,
) -> ComponentDifferenceMetric:
    compared_array = np.asarray(compared_values, dtype=np.float64).reshape(-1)
    reference_array = np.asarray(reference_values, dtype=np.float64).reshape(-1)
    if compared_array.shape != reference_array.shape:
        raise ValueError(f"source_difference_qpos_shape_mismatch:{component}")
    active = np.isfinite(compared_array) & np.isfinite(reference_array)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        raise ValueError(f"source_difference_no_common_qpos_slots:{component}")
    declared_indices = (
        np.arange(compared_array.size, dtype=np.int64)
        if coordinate_indices is None
        else np.asarray(coordinate_indices, dtype=np.int64)
    )
    if declared_indices.shape != compared_array.shape:
        raise ValueError(
            f"source_difference_qpos_coordinate_mismatch:{component}"
        )
    delta = reference_array[indices] - compared_array[indices]
    return ComponentDifferenceMetric(
        component_id=component,
        representation="qpos",
        reference_frame=None,
        controlled_frame=None,
        translation_delta_xyz_m=(),
        translation_norm_m=None,
        rotation_delta_rotvec_rad=(),
        rotation_angle_rad=None,
        qpos_active_indices=tuple(
            int(value) for value in declared_indices[indices]
        ),
        qpos_delta=tuple(float(value) for value in delta),
        qpos_l2_norm=float(np.linalg.norm(delta)),
        qpos_max_abs=float(np.max(np.abs(delta))),
    )


def _component_metrics(
    compared: ComponentAction,
    reference: ComponentAction,
) -> tuple[ComponentDifferenceMetric, ...]:
    if (
        compared.action_schema.semantic_content_hash
        != reference.action_schema.semantic_content_hash
    ):
        raise ValueError("source_difference_action_schema_mismatch")
    common = tuple(
        component
        for component in compared.component_coverage
        if component in set(reference.component_coverage)
    )
    if not common:
        raise ValueError("source_difference_no_common_components")
    metrics: list[ComponentDifferenceMetric] = []
    for component in common:
        compared_value = compared.value(component)
        reference_value = reference.value(component)
        compared_schema = compared_value.payload_schema
        reference_schema = reference_value.payload_schema
        if (
            compared_schema.semantic_content_hash
            != reference_schema.semantic_content_hash
        ):
            raise ValueError(
                f"source_difference_payload_schema_mismatch:{component}"
            )
        family = compared_schema.representation_family
        if family == "cartesian_pose":
            metrics.append(
                _pose_metric(component, compared_value, reference_value)
            )
        elif family == "joint_position":
            coordinate_indices = None
            hand_spec = compared_schema.semantic_parameters.get(
                "hand_action_spec"
            )
            if isinstance(hand_spec, Mapping):
                active_mask = np.asarray(
                    tuple(bool(value) for value in hand_spec["active_mask"]),
                    dtype=bool,
                )
                coordinate_indices = tuple(
                    int(value) for value in np.flatnonzero(active_mask)
                )
            metrics.append(
                _qpos_metric(
                    component,
                    compared_value.values,
                    reference_value.values,
                    coordinate_indices=coordinate_indices,
                )
            )
        else:
            raise ValueError(
                "source_difference_representation_unsupported:"
                f"{component}:{family}"
            )
    return tuple(metrics)


def _record_difference(
    compared_role: str,
    compared: CompositionInputRecord,
    reference_role: str,
    reference_action: ComponentAction,
    *,
    reference_record: CompositionInputRecord | None,
    evaluated_at: float,
) -> ActionDifference:
    available_delta = None
    effective_delta = None
    valid_until = compared.effective_until
    if reference_record is not None:
        if (
            compared.available_clock_domain_id
            == reference_record.available_clock_domain_id
        ):
            available_delta = (
                reference_record.available_at - compared.available_at
            )
        if (
            compared.effective_clock_domain_id
            == reference_record.effective_clock_domain_id
        ):
            effective_delta = (
                reference_record.effective_from - compared.effective_from
            )
        valid_until = min(valid_until, reference_record.effective_until)
    return ActionDifference(
        compared_role=_required(compared_role, name="compared_role"),
        reference_role=_required(reference_role, name="reference_role"),
        component_metrics=_component_metrics(compared.action, reference_action),
        available_at_delta_sec=available_delta,
        effective_from_delta_sec=effective_delta,
        valid_for_sec=max(0.0, valid_until - evaluated_at),
    )


def _metric_payload(metric: ComponentDifferenceMetric) -> Mapping[str, object]:
    return {
        "component_id": metric.component_id,
        "representation": metric.representation,
        "reference_frame": metric.reference_frame,
        "controlled_frame": metric.controlled_frame,
        "translation_delta_xyz_m": metric.translation_delta_xyz_m,
        "translation_norm_m": metric.translation_norm_m,
        "rotation_delta_rotvec_rad": metric.rotation_delta_rotvec_rad,
        "rotation_angle_rad": metric.rotation_angle_rad,
        "qpos_active_indices": metric.qpos_active_indices,
        "qpos_delta": metric.qpos_delta,
        "qpos_l2_norm": metric.qpos_l2_norm,
        "qpos_max_abs": metric.qpos_max_abs,
    }


def action_difference_semantic_payload(
    difference: ActionDifference,
) -> Mapping[str, object]:
    return {
        "compared_role": difference.compared_role,
        "reference_role": difference.reference_role,
        "component_metrics": tuple(
            _metric_payload(metric)
            for metric in difference.component_metrics
        ),
        "available_at_delta_sec": difference.available_at_delta_sec,
        "effective_from_delta_sec": difference.effective_from_delta_sec,
        "valid_for_sec": difference.valid_for_sec,
    }


def evaluate_composition_input_differences(
    snapshot: CompositionInputSnapshot,
    *,
    spec: CompositionSpec,
    difference_spec: SourceDifferenceSpec,
    evaluated_at: float,
    clock_domain_id: str,
) -> tuple[ActionDifference, ...]:
    """Evaluate only same-cut source-to-source differences.

    This is the computation seam for ``CompositionImmediateDifference``.  It
    deliberately has no composition evaluation argument and therefore cannot
    accidentally report a post-mix or execution fact.
    """

    if snapshot.composition_spec_hash != spec.semantic_content_hash:
        raise ValueError("Source difference snapshot/spec mismatch.")
    if not isinstance(difference_spec, SourceDifferenceSpec):
        raise TypeError("difference_spec must be a SourceDifferenceSpec.")
    difference_spec.validate_composition_spec(spec)
    evaluated = _finite(evaluated_at, name="evaluated_at")
    clock = _required(clock_domain_id, name="clock_domain_id")
    selected = snapshot.selected_input_by_role
    for record in selected.values():
        if record.effective_clock_domain_id != clock:
            raise ValueError(
                "Source difference evaluation uses another effective clock."
            )
    return tuple(
        _record_difference(
            compared_role,
            selected[compared_role],
            reference_role,
            selected[reference_role].action,
            reference_record=selected[reference_role],
            evaluated_at=evaluated,
        )
        for compared_role, reference_role
        in difference_spec.compared_reference_role_pairs
        if compared_role in selected and reference_role in selected
    )


def evaluate_composition_source_differences(
    snapshot: CompositionInputSnapshot,
    result: CompositionEvaluation,
    *,
    spec: CompositionSpec,
    difference_spec: SourceDifferenceSpec,
    evaluated_at: float,
    clock_domain_id: str,
) -> SourceDifferenceObservation:
    """Evaluate pairwise and source-to-output differences on one exact cut."""

    if (
        result.composition_spec_hash != spec.semantic_content_hash
        or result.composition_input_snapshot_id
        != snapshot.composition_input_snapshot_id
        or result.composition_input_snapshot_hash
        != snapshot.semantic_content_hash
    ):
        raise ValueError(
            "Source difference inputs must share one composition snapshot."
        )
    evaluated = _finite(evaluated_at, name="evaluated_at")
    clock = _required(clock_domain_id, name="clock_domain_id")
    selected = snapshot.selected_input_by_role
    pairwise = evaluate_composition_input_differences(
        snapshot,
        spec=spec,
        difference_spec=difference_spec,
        evaluated_at=evaluated,
        clock_domain_id=clock,
    )
    residuals: tuple[ActionDifference, ...] = ()
    if result.composed_candidate is not None:
        residuals = tuple(
            _record_difference(
                role,
                selected[role],
                "composed_output",
                result.composed_candidate,
                reference_record=None,
                evaluated_at=evaluated,
            )
            for role in difference_spec.output_residual_roles
            if role in selected
        )
    observed_roles = {
        role
        for pair in difference_spec.compared_reference_role_pairs
        for role in pair
        if role in selected
    } | {
        role
        for role in difference_spec.output_residual_roles
        if role in selected
    }
    valid_until = min(
        (selected[role].effective_until for role in observed_roles),
        default=evaluated,
    )
    payload = {
        "composition_input_snapshot_id": snapshot.composition_input_snapshot_id,
        "composition_input_snapshot_hash": snapshot.semantic_content_hash,
        "composition_evaluation_id": result.composition_evaluation_id,
        "composition_evaluation_hash": result.semantic_content_hash,
        "composition_spec_hash": spec.semantic_content_hash,
        "source_difference_spec_hash": (
            difference_spec.semantic_content_hash
        ),
        "target_effective_at": snapshot.target_effective_at,
        "evaluated_at": evaluated,
        "clock_domain_id": clock,
        "valid_until": valid_until,
        "pairwise_differences": tuple(
            action_difference_semantic_payload(item) for item in pairwise
        ),
        "output_residuals": tuple(
            action_difference_semantic_payload(item) for item in residuals
        ),
    }
    digest = canonical_content_hash(
        payload,
        domain=SOURCE_DIFFERENCE_OBSERVATION_HASH_DOMAIN_V1,
    )
    return SourceDifferenceObservation(
        observation_id=f"source-difference-observation:{digest}",
        pairwise_differences=pairwise,
        output_residuals=residuals,
        semantic_content_hash=digest,
        **{
            key: value
            for key, value in payload.items()
            if key not in {"pairwise_differences", "output_residuals"}
        },
    )


__all__ = [
    "SOURCE_DIFFERENCE_OBSERVATION_HASH_DOMAIN_V1",
    "SOURCE_DIFFERENCE_SPEC_HASH_DOMAIN_V1",
    "ActionDifference",
    "ComponentDifferenceMetric",
    "SourceDifferenceObservation",
    "SourceDifferenceSpec",
    "action_difference_semantic_payload",
    "evaluate_composition_input_differences",
    "evaluate_composition_source_differences",
]

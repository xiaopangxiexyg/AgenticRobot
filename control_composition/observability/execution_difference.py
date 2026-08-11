"""Non-authoritative difference between paired downstream observations.

The execution owner supplies the exact reference frame associated with a
measurement.  This module validates that pair and computes a metric; it never
selects from a reference timeline or reruns interpolation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.execution_observation import (
    ExecutionFactReference,
    ExecutionMeasurementObservation,
    ExecutionReferenceFrameObservation,
)
from robot_motion.runtime.immutable import deep_freeze

from .modules import DifferenceReceiptReference


MEASUREMENT_REFERENCE_ASSOCIATION_HASH_DOMAIN = (
    "dexteleop.measurement_reference_association.development"
)
MEASUREMENT_REFERENCE_ASSOCIATION_STATUSES = frozenset(
    {"available", "unavailable", "incomparable"}
)


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


def _sequence(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _receipt_references(
    values: Sequence[DifferenceReceiptReference],
) -> tuple[DifferenceReceiptReference, ...]:
    references = tuple(values)
    if any(
        not isinstance(reference, DifferenceReceiptReference)
        for reference in references
    ):
        raise TypeError(
            "receipt_references must contain DifferenceReceiptReference values."
        )
    hashes = tuple(reference.semantic_content_hash for reference in references)
    if len(hashes) != len(set(hashes)):
        raise ValueError("receipt_references must be unique.")
    return references


def _difference_reference(
    reference: ExecutionFactReference,
) -> DifferenceReceiptReference:
    return DifferenceReceiptReference(
        stage=reference.stage,
        receipt_id=reference.fact_id,
        receipt_hash=reference.fact_hash,
        observed_at=reference.observed_at,
        clock_domain_id=reference.clock_domain_id,
        status=reference.status,
    )


def _observation_receipts(
    measurement: ExecutionMeasurementObservation,
    reference: ExecutionReferenceFrameObservation | None,
) -> tuple[DifferenceReceiptReference, ...]:
    values: list[DifferenceReceiptReference] = []
    if reference is not None:
        values.extend(
            _difference_reference(parent) for parent in reference.parent_facts
        )
        values.append(
            DifferenceReceiptReference(
                stage="actual_reference_frame",
                receipt_id=reference.reference_frame_id,
                receipt_hash=reference.semantic_content_hash,
                observed_at=reference.available_at,
                clock_domain_id=reference.clock_domain_id,
                status="observed",
            )
        )
    values.extend(
        _difference_reference(parent) for parent in measurement.parent_facts
    )
    values.append(
        DifferenceReceiptReference(
            stage="measurement_observation",
            receipt_id=measurement.measurement_id,
            receipt_hash=measurement.semantic_content_hash,
            observed_at=measurement.available_at,
            clock_domain_id=measurement.clock_domain_id,
            status="observed",
        )
    )
    unique: dict[str, DifferenceReceiptReference] = {}
    for value in values:
        unique.setdefault(value.semantic_content_hash, value)
    return tuple(unique.values())


def _cartesian_pose_residual(
    reference_values: Sequence[float],
    measurement_values: Sequence[float],
    *,
    schema_hash: str,
) -> Mapping[str, object]:
    reference = np.asarray(reference_values, dtype=float).reshape(-1)
    measured = np.asarray(measurement_values, dtype=float).reshape(-1)
    if (
        reference.shape != (7,)
        or measured.shape != (7,)
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(measured))
    ):
        raise ValueError("Cartesian pose observations must contain finite xyzquat7.")
    reference_quaternion = reference[3:7]
    measured_quaternion = measured[3:7]
    reference_norm = float(np.linalg.norm(reference_quaternion))
    measured_norm = float(np.linalg.norm(measured_quaternion))
    if reference_norm <= 1e-12 or measured_norm <= 1e-12:
        raise ValueError("Cartesian pose quaternion norm must be positive.")
    reference_quaternion = reference_quaternion / reference_norm
    measured_quaternion = measured_quaternion / measured_norm
    translation = measured[:3] - reference[:3]
    relative_rotation = (
        Rotation.from_quat(reference_quaternion).inv()
        * Rotation.from_quat(measured_quaternion)
    )
    rotation_vector = relative_rotation.as_rotvec()
    return {
        "metric": "cartesian_pose_residual",
        "translation_signed_m": tuple(float(value) for value in translation),
        "translation_l2_m": float(np.linalg.norm(translation)),
        "translation_linf_m": float(np.max(np.abs(translation))),
        "rotation_vector_rad": tuple(
            float(value) for value in rotation_vector
        ),
        "rotation_angle_rad": float(np.linalg.norm(rotation_vector)),
        "rotation_vector_frame": "reference_pose_local",
        "reference": tuple(float(value) for value in reference),
        "measured": tuple(float(value) for value in measured),
        "representation": "cartesian_pose",
        "schema_hash": schema_hash,
    }


@dataclass(frozen=True)
class MeasurementReferenceAssociation:
    """Passive result for one downstream-paired reference and measurement."""

    status: str
    reason: str
    measurement_id: str
    measurement_hash: str
    measured_at: float
    clock_domain_id: str
    control_domain_id: str
    selection_group_id: str
    reference_frame_id: str | None
    reference_frame_hash: str | None
    reference_stage_id: str | None
    execution_fence_sequence: int | None
    reference_revision: int | None
    component_difference: Mapping[str, Mapping[str, object]]
    receipt_references: tuple[DifferenceReceiptReference, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        status = _required(self.status, name="status")
        if status not in MEASUREMENT_REFERENCE_ASSOCIATION_STATUSES:
            raise ValueError(
                f"Unsupported measurement/reference status {status!r}."
            )
        reference_id = self.reference_frame_id
        reference_hash = self.reference_frame_hash
        reference_stage_id = self.reference_stage_id
        fence = self.execution_fence_sequence
        revision = self.reference_revision
        if status == "available":
            if (
                reference_id is None
                or reference_hash is None
                or reference_stage_id is None
                or fence is None
                or revision is None
            ):
                raise ValueError(
                    "Available association requires exact reference lineage."
                )
        elif any(
            value is not None
            for value in (
                reference_id,
                reference_hash,
                reference_stage_id,
                fence,
                revision,
            )
        ):
            raise ValueError(
                "Unavailable/incomparable association cannot claim reference lineage."
            )
        if reference_id is not None:
            reference_id = _required(reference_id, name="reference_frame_id")
            reference_hash = _required(
                reference_hash,
                name="reference_frame_hash",
            )
            reference_stage_id = _required(
                reference_stage_id,
                name="reference_stage_id",
            )
            fence = _sequence(fence, name="execution_fence_sequence")
            revision = _sequence(revision, name="reference_revision")
        differences = deep_freeze(dict(self.component_difference))
        if status == "available" and not differences:
            raise ValueError(
                "Available association requires at least one component difference."
            )
        if status != "available" and differences:
            raise ValueError(
                "Unavailable/incomparable association cannot carry differences."
            )
        references = _receipt_references(self.receipt_references)
        payload = {
            "status": status,
            "reason": _required(self.reason, name="reason"),
            "measurement_id": _required(
                self.measurement_id,
                name="measurement_id",
            ),
            "measurement_hash": _required(
                self.measurement_hash,
                name="measurement_hash",
            ),
            "measured_at": _finite(self.measured_at, name="measured_at"),
            "clock_domain_id": _required(
                self.clock_domain_id,
                name="clock_domain_id",
            ),
            "control_domain_id": _required(
                self.control_domain_id,
                name="control_domain_id",
            ),
            "selection_group_id": _required(
                self.selection_group_id,
                name="selection_group_id",
            ),
            "reference_frame_id": reference_id,
            "reference_frame_hash": reference_hash,
            "reference_stage_id": reference_stage_id,
            "execution_fence_sequence": fence,
            "reference_revision": revision,
            "component_difference": differences,
            "receipt_reference_hashes": tuple(
                reference.semantic_content_hash for reference in references
            ),
        }
        for name, value in (
            *payload.items(),
            ("receipt_references", references),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=MEASUREMENT_REFERENCE_ASSOCIATION_HASH_DOMAIN,
            ),
        )

    @property
    def association_id(self) -> str:
        return f"measurement-reference-association:{self.semantic_content_hash}"


def associate_measurement_to_reference(
    measurement: ExecutionMeasurementObservation,
    reference: ExecutionReferenceFrameObservation | None,
    *,
    selection_group_id: str,
    component_ids: Sequence[str],
) -> MeasurementReferenceAssociation:
    """Validate one downstream-selected actual frame and compute residuals."""

    if not isinstance(measurement, ExecutionMeasurementObservation):
        raise TypeError(
            "measurement must be an ExecutionMeasurementObservation."
        )
    if reference is not None and not isinstance(
        reference,
        ExecutionReferenceFrameObservation,
    ):
        raise TypeError(
            "reference must be an ExecutionReferenceFrameObservation or None."
        )
    selection_group = _required(
        selection_group_id,
        name="selection_group_id",
    )
    required_components = tuple(
        _required(value, name="component_id") for value in component_ids
    )
    if (
        not required_components
        or len(set(required_components)) != len(required_components)
    ):
        raise ValueError("component_ids must be non-empty and unique.")
    if reference is None:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="unavailable",
            reason="reference_observation_not_provided",
        )
    if reference.selection_group_id != selection_group:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="incomparable",
            reason="selection_group_mismatch",
            reference=reference,
        )
    if reference.control_domain_id != measurement.control_domain_id:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="incomparable",
            reason="control_domain_mismatch",
            reference=reference,
        )
    if reference.clock_domain_id != measurement.clock_domain_id:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="incomparable",
            reason="clock_domain_mismatch",
            reference=reference,
        )
    if (
        reference.observation_binding_id
        != measurement.observation_binding_id
        or reference.observation_plan_hash
        != measurement.observation_plan_hash
    ):
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="incomparable",
            reason="observation_binding_mismatch",
            reference=reference,
        )
    if reference.temporal_semantics == "active_interval":
        reference_covers_measurement = reference.covers(measurement.measured_at)
    elif reference.observation_binding_id is not None:
        reference_covers_measurement = math.isclose(
            reference.scheduled_for,
            measurement.measured_at,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    else:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="incomparable",
            reason="reference_has_no_active_interval",
            reference=reference,
        )
    if not reference_covers_measurement:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="unavailable",
            reason="reference_not_effective_at_measurement",
            reference=reference,
        )
    if reference.available_at > measurement.available_at + 1e-12:
        return _association_without_reference(
            measurement,
            selection_group_id=selection_group,
            status="unavailable",
            reason="reference_fact_not_available_at_measurement_observation",
            reference=reference,
        )

    component_difference: dict[str, Mapping[str, object]] = {}
    for component_id in required_components:
        if (
            component_id not in reference.value_by_component
            or component_id not in measurement.value_by_component
        ):
            return _association_without_reference(
                measurement,
                selection_group_id=selection_group,
                status="incomparable",
                reason=f"component_missing:{component_id}",
                reference=reference,
            )
        reference_representation = reference.representation_by_component[
            component_id
        ]
        measurement_representation = measurement.representation_by_component[
            component_id
        ]
        if reference_representation != measurement_representation:
            return _association_without_reference(
                measurement,
                selection_group_id=selection_group,
                status="incomparable",
                reason=f"representation_mismatch:{component_id}",
                reference=reference,
            )
        if (
            reference.schema_hash_by_component[component_id]
            != measurement.schema_hash_by_component[component_id]
        ):
            return _association_without_reference(
                measurement,
                selection_group_id=selection_group,
                status="incomparable",
                reason=f"schema_mismatch:{component_id}",
                reference=reference,
            )
        reference_values = reference.value_by_component[component_id]
        measurement_values = measurement.value_by_component[component_id]
        if len(reference_values) != len(measurement_values):
            return _association_without_reference(
                measurement,
                selection_group_id=selection_group,
                status="incomparable",
                reason=f"dimension_mismatch:{component_id}",
                reference=reference,
            )
        if reference_representation == "joint_position":
            signed = tuple(
                measured - expected
                for expected, measured in zip(
                    reference_values,
                    measurement_values,
                )
            )
            component_difference[component_id] = {
                "metric": "joint_position_vector_residual",
                "dimension": len(signed),
                "signed": signed,
                "l2": math.sqrt(sum(value * value for value in signed)),
                "linf": max(abs(value) for value in signed),
                "reference": reference_values,
                "measured": measurement_values,
                "representation": reference_representation,
                "schema_hash": reference.schema_hash_by_component[
                    component_id
                ],
            }
        elif reference_representation == "cartesian_pose":
            try:
                component_difference[component_id] = (
                    _cartesian_pose_residual(
                        reference_values,
                        measurement_values,
                        schema_hash=reference.schema_hash_by_component[
                            component_id
                        ],
                    )
                )
            except ValueError:
                return _association_without_reference(
                    measurement,
                    selection_group_id=selection_group,
                    status="incomparable",
                    reason=f"invalid_cartesian_pose:{component_id}",
                    reference=reference,
                )
        else:
            return _association_without_reference(
                measurement,
                selection_group_id=selection_group,
                status="incomparable",
                reason=(
                    f"unsupported_metric:{component_id}:"
                    f"{reference_representation}"
                ),
                reference=reference,
            )

    return MeasurementReferenceAssociation(
        status="available",
        reason="downstream_actual_reference_at_measurement",
        measurement_id=measurement.measurement_id,
        measurement_hash=measurement.semantic_content_hash,
        measured_at=measurement.measured_at,
        clock_domain_id=measurement.clock_domain_id,
        control_domain_id=measurement.control_domain_id,
        selection_group_id=selection_group,
        reference_frame_id=reference.reference_frame_id,
        reference_frame_hash=reference.semantic_content_hash,
        reference_stage_id=reference.reference_stage_id,
        execution_fence_sequence=reference.execution_fence_sequence,
        reference_revision=reference.reference_revision,
        component_difference=component_difference,
        receipt_references=_observation_receipts(measurement, reference),
    )


def _association_without_reference(
    measurement: ExecutionMeasurementObservation,
    *,
    selection_group_id: str,
    status: str,
    reason: str,
    reference: ExecutionReferenceFrameObservation | None = None,
) -> MeasurementReferenceAssociation:
    return MeasurementReferenceAssociation(
        status=status,
        reason=reason,
        measurement_id=measurement.measurement_id,
        measurement_hash=measurement.semantic_content_hash,
        measured_at=measurement.measured_at,
        clock_domain_id=measurement.clock_domain_id,
        control_domain_id=measurement.control_domain_id,
        selection_group_id=selection_group_id,
        reference_frame_id=None,
        reference_frame_hash=None,
        reference_stage_id=None,
        execution_fence_sequence=None,
        reference_revision=None,
        component_difference={},
        receipt_references=_observation_receipts(measurement, reference),
    )


__all__ = [
    "MEASUREMENT_REFERENCE_ASSOCIATION_HASH_DOMAIN",
    "MEASUREMENT_REFERENCE_ASSOCIATION_STATUSES",
    "MeasurementReferenceAssociation",
    "associate_measurement_to_reference",
]

"""Standard activity metrics for pose and continuous component actions.

Metric extraction is separate from allocation thresholds so profiles can use
different devices, body topologies, units, and sensitivity without changing
the generic reducer.  Non-standard representations provide an explicit metric
provider instead of being coerced into pose/qpos semantics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.component_action import ComponentAction

from ..residual_anchor import ResidualAnchorReceipt
from .contracts import (
    ComponentResidualMagnitudeObservation,
    ComponentSourceActivityObservation,
)


SOURCE_ACTION_ACTIVITY_SAMPLE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.source_action_activity_sample.development"
)
METRIC_PROVIDER_CONTRACT_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_metric_provider_contract.development"
)
STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_ID = (
    "control_composition.standard_component_activity_metrics.v1"
)
STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_HASH = canonical_content_hash(
    {
        "provider_id": STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_ID,
        "input_contract": "bounded_component_action_window.v1",
        "supported_representation_families": (
            "cartesian_pose",
            "joint_position",
            "continuous_native",
        ),
        "output_contract": "nonnegative_named_metrics.v1",
    },
    domain=METRIC_PROVIDER_CONTRACT_HASH_DOMAIN_DEVELOPMENT,
)
STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_ID = (
    "control_composition.standard_component_residual_magnitude_metrics.v1"
)
STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_HASH = canonical_content_hash(
    {
        "provider_id": STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_ID,
        "input_contract": "anchor_bound_relative_component_action.v1",
        "supported_representation_families": (
            "cartesian_pose",
            "joint_position",
            "continuous_native",
        ),
        "output_contract": "nonnegative_named_metrics.v1",
    },
    domain=METRIC_PROVIDER_CONTRACT_HASH_DOMAIN_DEVELOPMENT,
)

ActivityMetricProvider = Callable[
    [tuple["SourceActionActivitySample", ...], str],
    Mapping[str, float],
]
ResidualMagnitudeMetricProvider = Callable[
    ["SourceActionActivitySample", str],
    Mapping[str, float],
]


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


@dataclass(frozen=True)
class SourceActionActivitySample:
    """One mapped-clock action sample used for allocation evidence."""

    sampled_at: float
    clock_domain_id: str
    action: ComponentAction
    source_fact_id: str
    source_fact_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        sampled_at = _finite(self.sampled_at, name="sampled_at")
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be a ComponentAction.")
        source_fact_id = _required(
            self.source_fact_id,
            name="source_fact_id",
        )
        source_fact_hash = _required(
            self.source_fact_hash,
            name="source_fact_hash",
        )
        payload = {
            "sampled_at": sampled_at,
            "clock_domain_id": clock,
            "action_hash": self.action.semantic_content_hash,
            "action_schema_hash": self.action.action_schema.semantic_content_hash,
            "source_fact_id": source_fact_id,
            "source_fact_hash": source_fact_hash,
        }
        object.__setattr__(self, "sampled_at", sampled_at)
        object.__setattr__(self, "clock_domain_id", clock)
        object.__setattr__(self, "source_fact_id", source_fact_id)
        object.__setattr__(self, "source_fact_hash", source_fact_hash)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=SOURCE_ACTION_ACTIVITY_SAMPLE_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def sample_id(self) -> str:
        return f"source-action-activity-sample:{self.semantic_content_hash}"


def _quaternion_angle(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise ValueError("Pose activity requires non-zero quaternions.")
    dot = abs(float(np.dot(left / left_norm, right / right_norm)))
    return 2.0 * math.acos(float(np.clip(dot, -1.0, 1.0)))


def _pose_metrics(
    values: tuple[np.ndarray, ...], times: tuple[float, ...]
) -> dict[str, float]:
    translation_path = 0.0
    rotation_path = 0.0
    translation_peak_speed = 0.0
    rotation_peak_speed = 0.0
    for previous, current, previous_at, current_at in zip(
        values,
        values[1:],
        times,
        times[1:],
    ):
        elapsed = current_at - previous_at
        translation = float(np.linalg.norm(current[:3] - previous[:3]))
        rotation = _quaternion_angle(previous[3:7], current[3:7])
        translation_path += translation
        rotation_path += rotation
        translation_peak_speed = max(
            translation_peak_speed,
            translation / elapsed,
        )
        rotation_peak_speed = max(rotation_peak_speed, rotation / elapsed)
    return {
        "translation_displacement_m": float(
            np.linalg.norm(values[-1][:3] - values[0][:3])
        ),
        "translation_path_m": translation_path,
        "translation_peak_speed_mps": translation_peak_speed,
        "rotation_displacement_rad": _quaternion_angle(
            values[0][3:7],
            values[-1][3:7],
        ),
        "rotation_path_rad": rotation_path,
        "rotation_peak_speed_radps": rotation_peak_speed,
    }


def _continuous_vector_metrics(
    values: tuple[np.ndarray, ...],
    times: tuple[float, ...],
) -> dict[str, float]:
    linf_path = 0.0
    l2_path = 0.0
    peak_rate = 0.0
    for previous, current, previous_at, current_at in zip(
        values,
        values[1:],
        times,
        times[1:],
    ):
        elapsed = current_at - previous_at
        delta = current - previous
        linf = float(np.max(np.abs(delta)))
        linf_path += linf
        l2_path += float(np.linalg.norm(delta))
        peak_rate = max(peak_rate, linf / elapsed)
    displacement = values[-1] - values[0]
    return {
        "coordinate_linf_displacement_native": float(np.max(np.abs(displacement))),
        "coordinate_l2_displacement_native": float(np.linalg.norm(displacement)),
        "coordinate_linf_path_native": linf_path,
        "coordinate_l2_path_native": l2_path,
        "coordinate_peak_rate_native_per_sec": peak_rate,
    }


def _pose_residual_magnitude_metrics(value: np.ndarray) -> dict[str, float]:
    if value.shape != (7,):
        raise ValueError("Cartesian pose residual magnitude requires shape (7,).")
    return {
        "translation_from_anchor_m": float(np.linalg.norm(value[:3])),
        "rotation_from_anchor_rad": _quaternion_angle(
            value[3:7],
            np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64),
        ),
    }


def _continuous_residual_magnitude_metrics(
    value: np.ndarray,
) -> dict[str, float]:
    return {
        "coordinate_linf_from_anchor_native": float(np.max(np.abs(value))),
        "coordinate_l2_from_anchor_native": float(np.linalg.norm(value)),
    }


def standard_component_activity_metrics(
    samples: tuple[SourceActionActivitySample, ...],
    component_id: str,
) -> Mapping[str, float]:
    """Extract standard metrics or require an explicit custom provider."""

    if len(samples) < 2:
        raise ValueError("Activity metrics require at least two samples.")
    component = _required(component_id, name="component_id")
    payload_schemas = tuple(
        sample.action.value(component).payload_schema for sample in samples
    )
    schema_hash = payload_schemas[0].semantic_content_hash
    if any(
        schema.semantic_content_hash != schema_hash for schema in payload_schemas[1:]
    ):
        raise ValueError("Activity samples changed the component payload schema.")
    family = payload_schemas[0].representation_family
    values = tuple(
        np.asarray(
            sample.action.value(component).values,
            dtype=np.float64,
        )
        for sample in samples
    )
    times = tuple(sample.sampled_at for sample in samples)
    if family == "cartesian_pose":
        return _pose_metrics(values, times)
    if family in {"joint_position", "continuous_native"}:
        return _continuous_vector_metrics(values, times)
    raise ValueError(
        f"Representation {family!r} requires an explicit activity metric provider."
    )


def standard_component_residual_magnitude_metrics(
    sample: SourceActionActivitySample,
    component_id: str,
) -> Mapping[str, float]:
    """Measure one relative action against the identity of its anchor."""

    if not isinstance(sample, SourceActionActivitySample):
        raise TypeError("sample must be SourceActionActivitySample.")
    component = _required(component_id, name="component_id")
    if component not in sample.action.component_coverage:
        raise ValueError(
            f"Residual magnitude sample lacks component {component!r}."
        )
    value = sample.action.value(component)
    schema = value.payload_schema
    if schema.value_semantics != "relative_delta":
        raise ValueError(
            "Residual magnitude requires relative_delta value semantics."
        )
    values = np.asarray(value.values, dtype=np.float64)
    if schema.representation_family == "cartesian_pose":
        return _pose_residual_magnitude_metrics(values)
    if schema.representation_family in {"joint_position", "continuous_native"}:
        return _continuous_residual_magnitude_metrics(values)
    raise ValueError(
        f"Representation {schema.representation_family!r} requires an explicit "
        "residual magnitude metric provider."
    )


def build_component_source_activity_observation(
    samples: Sequence[SourceActionActivitySample],
    *,
    source_role: str,
    component_id: str,
    available_at: float,
    valid_until: float,
    normalization_scale_by_metric: Mapping[str, float],
    metric_provider: ActivityMetricProvider | None = None,
    metric_provider_id: str | None = None,
    metric_provider_hash: str | None = None,
) -> ComponentSourceActivityObservation:
    """Build one replayable activity fact from a bounded action window."""

    ordered = tuple(samples)
    if len(ordered) < 2 or any(
        not isinstance(sample, SourceActionActivitySample) for sample in ordered
    ):
        raise TypeError("samples must contain at least two SourceActionActivitySample.")
    component = _required(component_id, name="component_id")
    clock = ordered[0].clock_domain_id
    previous_at: float | None = None
    for sample in ordered:
        if sample.clock_domain_id != clock:
            raise ValueError("Activity samples must share one mapped clock domain.")
        if component not in sample.action.component_coverage:
            raise ValueError(
                f"Activity sample does not contain component {component!r}."
            )
        if previous_at is not None and sample.sampled_at <= previous_at:
            raise ValueError("Activity sample times must increase strictly.")
        previous_at = sample.sampled_at
    if metric_provider is None:
        provider = standard_component_activity_metrics
        if metric_provider_id not in {
            None,
            STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_ID,
        } or metric_provider_hash not in {
            None,
            STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_HASH,
        }:
            raise ValueError(
                "The standard activity metric provider has one fixed identity."
            )
        provider_id = STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_ID
        provider_hash = STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_HASH
    else:
        provider = metric_provider
        if metric_provider_id is None or metric_provider_hash is None:
            raise ValueError(
                "A custom activity metric provider requires an explicit "
                "metric_provider_id and metric_provider_hash."
            )
        provider_id = _required(metric_provider_id, name="metric_provider_id")
        provider_hash = _required(
            metric_provider_hash,
            name="metric_provider_hash",
        )
    metrics = provider(ordered, component)
    if not isinstance(metrics, Mapping):
        raise TypeError("Activity metric provider must return a mapping.")
    family = ordered[0].action.value(component).payload_schema.representation_family
    return ComponentSourceActivityObservation(
        source_role=source_role,
        metric_provider_id=provider_id,
        metric_provider_hash=provider_hash,
        component_id=component,
        representation_family=family,
        window_started_at=ordered[0].sampled_at,
        window_ended_at=ordered[-1].sampled_at,
        available_at=available_at,
        valid_until=valid_until,
        clock_domain_id=clock,
        metric_values=metrics,
        normalization_scale_by_metric=normalization_scale_by_metric,
        source_fact_ids=tuple(sample.sample_id for sample in ordered),
        source_fact_hashes=tuple(sample.semantic_content_hash for sample in ordered),
    )


def build_component_residual_magnitude_observation(
    sample: SourceActionActivitySample,
    *,
    source_role: str,
    component_id: str,
    anchor_receipt: ResidualAnchorReceipt,
    available_at: float,
    valid_until: float,
    normalization_scale_by_metric: Mapping[str, float],
    metric_provider: ResidualMagnitudeMetricProvider | None = None,
    metric_provider_id: str | None = None,
    metric_provider_hash: str | None = None,
) -> ComponentResidualMagnitudeObservation:
    """Build one identity-relative fact bound to an exact anchor receipt."""

    if not isinstance(sample, SourceActionActivitySample):
        raise TypeError("sample must be SourceActionActivitySample.")
    if not isinstance(anchor_receipt, ResidualAnchorReceipt):
        raise TypeError("anchor_receipt must be ResidualAnchorReceipt.")
    role = _required(source_role, name="source_role")
    component = _required(component_id, name="component_id")
    if role != anchor_receipt.source_role:
        raise ValueError("Residual magnitude source role differs from its anchor.")
    if component not in anchor_receipt.component_binding_by_id:
        raise ValueError(
            "Residual magnitude component is not covered by its anchor receipt."
        )
    if component not in sample.action.component_coverage:
        raise ValueError(
            f"Residual magnitude sample lacks component {component!r}."
        )
    if sample.action.action_schema.semantic_content_hash != (
        anchor_receipt.relative_action_schema_hash
    ):
        raise ValueError(
            "Residual magnitude action schema differs from its anchor receipt."
        )
    if sample.clock_domain_id != anchor_receipt.clock_domain_id:
        raise ValueError(
            "Residual magnitude sample and anchor use another mapped clock."
        )
    if sample.sampled_at < anchor_receipt.captured_at:
        raise ValueError("Residual magnitude sample predates its active anchor.")
    if metric_provider is None:
        provider = standard_component_residual_magnitude_metrics
        if metric_provider_id not in {
            None,
            STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_ID,
        } or metric_provider_hash not in {
            None,
            STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_HASH,
        }:
            raise ValueError(
                "The standard residual magnitude provider has one fixed identity."
            )
        provider_id = STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_ID
        provider_hash = STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_HASH
    else:
        provider = metric_provider
        if metric_provider_id is None or metric_provider_hash is None:
            raise ValueError(
                "A custom residual magnitude provider requires an explicit "
                "metric_provider_id and metric_provider_hash."
            )
        provider_id = _required(metric_provider_id, name="metric_provider_id")
        provider_hash = _required(
            metric_provider_hash,
            name="metric_provider_hash",
        )
    metrics = provider(sample, component)
    if not isinstance(metrics, Mapping):
        raise TypeError("Residual magnitude metric provider must return a mapping.")
    family = sample.action.value(component).payload_schema.representation_family
    return ComponentResidualMagnitudeObservation(
        source_role=role,
        metric_provider_id=provider_id,
        metric_provider_hash=provider_hash,
        component_id=component,
        representation_family=family,
        sampled_at=sample.sampled_at,
        available_at=available_at,
        valid_until=valid_until,
        clock_domain_id=sample.clock_domain_id,
        anchor_receipt_id=anchor_receipt.receipt_id,
        anchor_receipt_hash=anchor_receipt.semantic_content_hash,
        anchor_request_generation=anchor_receipt.request_generation,
        residual_action_id=(
            f"component-action:{sample.action.semantic_content_hash}"
        ),
        residual_action_hash=sample.action.semantic_content_hash,
        metric_values=metrics,
        normalization_scale_by_metric=normalization_scale_by_metric,
        source_fact_ids=(sample.source_fact_id,),
        source_fact_hashes=(sample.source_fact_hash,),
    )


__all__ = [
    "METRIC_PROVIDER_CONTRACT_HASH_DOMAIN_DEVELOPMENT",
    "SOURCE_ACTION_ACTIVITY_SAMPLE_HASH_DOMAIN_DEVELOPMENT",
    "STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_HASH",
    "STANDARD_COMPONENT_ACTIVITY_METRIC_PROVIDER_ID",
    "STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_HASH",
    "STANDARD_COMPONENT_RESIDUAL_MAGNITUDE_PROVIDER_ID",
    "ActivityMetricProvider",
    "ResidualMagnitudeMetricProvider",
    "SourceActionActivitySample",
    "build_component_residual_magnitude_observation",
    "build_component_source_activity_observation",
    "standard_component_activity_metrics",
    "standard_component_residual_magnitude_metrics",
]

"""Backend-neutral contracts for activity-driven composition allocation.

Allocation selects how already-declared source roles participate in
composition.  It does not change motion authority, run canonical safety,
poll a device, or dispatch an action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash


COMPOSITION_ALLOCATION_MODE_BASE_ONLY = "base_only"
COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY = "relative_overlay"
COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER = "absolute_overlay_owner"
COMPOSITION_ALLOCATION_MODES_DEVELOPMENT = frozenset(
    {
        COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
        COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY,
        COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER,
    }
)
COMPOSITION_ALLOCATION_STATUS_READY = "ready"
COMPOSITION_ALLOCATION_STATUS_NO_DECISION = "no_decision"
COMPOSITION_ALLOCATION_STATUSES_DEVELOPMENT = frozenset(
    {
        COMPOSITION_ALLOCATION_STATUS_READY,
        COMPOSITION_ALLOCATION_STATUS_NO_DECISION,
    }
)

COMPONENT_SOURCE_ACTIVITY_OBSERVATION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_source_activity_observation.development"
)
COMPONENT_RESIDUAL_MAGNITUDE_OBSERVATION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_residual_magnitude_observation.development"
)
COMPONENT_ALLOCATION_EVIDENCE_POLICY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_allocation_evidence_policy.development"
)
COMPONENT_ALLOCATION_POLICY_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_allocation_policy.development"
)
COMPOSITION_ALLOCATION_SPEC_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_allocation_spec.development"
)
COMPONENT_ALLOCATION_SELECTION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.component_allocation_selection.development"
)
COMPOSITION_ALLOCATION_DECISION_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_allocation_decision.development"
)
COMPOSITION_ALLOCATION_STATE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.composition_allocation_state.development"
)


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


def _nonnegative(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(value)


def _metric_map(
    values: Mapping[str, float],
    *,
    name: str,
    positive: bool,
) -> Mapping[str, float]:
    normalized = {
        _required(metric, name=f"{name}.metric"): (
            _positive(value, name=f"{name}[{metric!r}]")
            if positive
            else _nonnegative(value, name=f"{name}[{metric!r}]")
        )
        for metric, value in dict(values).items()
    }
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return MappingProxyType(dict(sorted(normalized.items())))


def _lineage(
    ids: tuple[str, ...],
    hashes: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    normalized_ids = tuple(_required(value, name="source_fact_id") for value in ids)
    normalized_hashes = tuple(
        _required(value, name="source_fact_hash") for value in hashes
    )
    if not normalized_ids or len(normalized_ids) != len(normalized_hashes):
        raise ValueError(
            "source fact ids/hashes must be non-empty and have equal length."
        )
    if len(normalized_ids) != len(set(normalized_ids)):
        raise ValueError("source_fact_ids must be unique.")
    return normalized_ids, normalized_hashes


def _mode_map(
    values: Mapping[str, str],
    *,
    component_ids: tuple[str, ...],
) -> Mapping[str, str]:
    normalized = {
        _required(component, name="component_id"): _required(
            mode,
            name=f"mode_by_component[{component!r}]",
        )
        for component, mode in dict(values).items()
    }
    if set(normalized) != set(component_ids):
        raise ValueError("mode_by_component must cover every component exactly.")
    if any(
        mode not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT
        for mode in normalized.values()
    ):
        raise ValueError("mode_by_component contains an unsupported mode.")
    return MappingProxyType(
        {component: normalized[component] for component in component_ids}
    )


def _transition_time_map(
    values: Mapping[str, float | None],
    *,
    component_ids: tuple[str, ...],
) -> Mapping[str, float | None]:
    raw = dict(values)
    if set(raw) != set(component_ids):
        raise ValueError(
            "mode_entered_at_by_component must cover every component exactly."
        )
    normalized = {
        component: (
            None
            if raw[component] is None
            else _finite(
                raw[component],
                name=f"mode_entered_at_by_component[{component!r}]",
            )
        )
        for component in component_ids
    }
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class ComponentSourceActivityObservation:
    """One bounded-window activity fact for one source role/component.

    Metric normalization is explicit and replayable.  The generic allocation
    reducer consumes only ``activity_level``; raw metrics remain available for
    UI/debug without being hard-coded into the state machine.
    """

    source_role: str
    metric_provider_id: str
    metric_provider_hash: str
    component_id: str
    representation_family: str
    window_started_at: float
    window_ended_at: float
    available_at: float
    valid_until: float
    clock_domain_id: str
    metric_values: Mapping[str, float]
    normalization_scale_by_metric: Mapping[str, float]
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    activity_level: float = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_role = _required(self.source_role, name="source_role")
        provider_id = _required(
            self.metric_provider_id,
            name="metric_provider_id",
        )
        provider_hash = _required(
            self.metric_provider_hash,
            name="metric_provider_hash",
        )
        component_id = _required(self.component_id, name="component_id")
        family = _required(
            self.representation_family,
            name="representation_family",
        )
        started = _finite(self.window_started_at, name="window_started_at")
        ended = _finite(self.window_ended_at, name="window_ended_at")
        available = _finite(self.available_at, name="available_at")
        valid_until = _finite(self.valid_until, name="valid_until")
        if not started < ended <= available < valid_until:
            raise ValueError(
                "Activity timing must satisfy window_started_at < "
                "window_ended_at <= available_at < valid_until."
            )
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        metrics = _metric_map(
            self.metric_values,
            name="metric_values",
            positive=False,
        )
        scales = _metric_map(
            self.normalization_scale_by_metric,
            name="normalization_scale_by_metric",
            positive=True,
        )
        if not set(scales).issubset(metrics):
            raise ValueError(
                "Every normalization scale must reference an available metric."
            )
        activity_level = max(
            metrics[metric] / scale for metric, scale in scales.items()
        )
        fact_ids, fact_hashes = _lineage(
            tuple(self.source_fact_ids),
            tuple(self.source_fact_hashes),
        )
        payload = {
            "source_role": source_role,
            "metric_provider_id": provider_id,
            "metric_provider_hash": provider_hash,
            "component_id": component_id,
            "representation_family": family,
            "window_started_at": started,
            "window_ended_at": ended,
            "available_at": available,
            "valid_until": valid_until,
            "clock_domain_id": clock,
            "metric_values": metrics,
            "normalization_scale_by_metric": scales,
            "activity_level": activity_level,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
        }
        object.__setattr__(self, "source_role", source_role)
        object.__setattr__(self, "metric_provider_id", provider_id)
        object.__setattr__(self, "metric_provider_hash", provider_hash)
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "representation_family", family)
        object.__setattr__(self, "window_started_at", started)
        object.__setattr__(self, "window_ended_at", ended)
        object.__setattr__(self, "available_at", available)
        object.__setattr__(self, "valid_until", valid_until)
        object.__setattr__(self, "clock_domain_id", clock)
        object.__setattr__(self, "metric_values", metrics)
        object.__setattr__(self, "normalization_scale_by_metric", scales)
        object.__setattr__(self, "source_fact_ids", fact_ids)
        object.__setattr__(self, "source_fact_hashes", fact_hashes)
        object.__setattr__(self, "activity_level", activity_level)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(COMPONENT_SOURCE_ACTIVITY_OBSERVATION_HASH_DOMAIN_DEVELOPMENT),
            ),
        )

    @property
    def observation_id(self) -> str:
        return f"component-source-activity:{self.semantic_content_hash}"

    def is_fresh_at(self, target_at: float, *, clock_domain_id: str) -> bool:
        target = _finite(target_at, name="target_at")
        return (
            _required(clock_domain_id, name="clock_domain_id") == self.clock_domain_id
            and self.available_at <= target < self.valid_until
        )


@dataclass(frozen=True)
class ComponentResidualMagnitudeObservation:
    """One anchor-bound residual magnitude fact for one component.

    This fact answers whether the latest relative action is near the identity
    of its *current* residual anchor.  It deliberately does not infer whether
    the source is moving; bounded-window motion remains a separate activity
    observation.
    """

    source_role: str
    metric_provider_id: str
    metric_provider_hash: str
    component_id: str
    representation_family: str
    sampled_at: float
    available_at: float
    valid_until: float
    clock_domain_id: str
    anchor_receipt_id: str
    anchor_receipt_hash: str
    anchor_request_generation: int
    residual_action_id: str
    residual_action_hash: str
    metric_values: Mapping[str, float]
    normalization_scale_by_metric: Mapping[str, float]
    source_fact_ids: tuple[str, ...] = ()
    source_fact_hashes: tuple[str, ...] = ()
    magnitude_level: float = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        source_role = _required(self.source_role, name="source_role")
        provider_id = _required(
            self.metric_provider_id,
            name="metric_provider_id",
        )
        provider_hash = _required(
            self.metric_provider_hash,
            name="metric_provider_hash",
        )
        component_id = _required(self.component_id, name="component_id")
        family = _required(
            self.representation_family,
            name="representation_family",
        )
        sampled = _finite(self.sampled_at, name="sampled_at")
        available = _finite(self.available_at, name="available_at")
        valid_until = _finite(self.valid_until, name="valid_until")
        if not sampled <= available < valid_until:
            raise ValueError(
                "Residual magnitude timing must satisfy sampled_at <= "
                "available_at < valid_until."
            )
        clock = _required(self.clock_domain_id, name="clock_domain_id")
        receipt_id = _required(
            self.anchor_receipt_id,
            name="anchor_receipt_id",
        )
        receipt_hash = _required(
            self.anchor_receipt_hash,
            name="anchor_receipt_hash",
        )
        generation = _nonnegative_int(
            self.anchor_request_generation,
            name="anchor_request_generation",
        )
        action_id = _required(self.residual_action_id, name="residual_action_id")
        action_hash = _required(
            self.residual_action_hash,
            name="residual_action_hash",
        )
        metrics = _metric_map(
            self.metric_values,
            name="metric_values",
            positive=False,
        )
        scales = _metric_map(
            self.normalization_scale_by_metric,
            name="normalization_scale_by_metric",
            positive=True,
        )
        if not set(scales).issubset(metrics):
            raise ValueError(
                "Every magnitude normalization scale must reference an "
                "available metric."
            )
        if self.source_fact_ids or self.source_fact_hashes:
            fact_ids, fact_hashes = _lineage(
                tuple(self.source_fact_ids),
                tuple(self.source_fact_hashes),
            )
        else:
            fact_ids, fact_hashes = (), ()
        magnitude_level = max(
            metrics[metric] / scale for metric, scale in scales.items()
        )
        payload = {
            "source_role": source_role,
            "metric_provider_id": provider_id,
            "metric_provider_hash": provider_hash,
            "component_id": component_id,
            "representation_family": family,
            "sampled_at": sampled,
            "available_at": available,
            "valid_until": valid_until,
            "clock_domain_id": clock,
            "anchor_receipt_id": receipt_id,
            "anchor_receipt_hash": receipt_hash,
            "anchor_request_generation": generation,
            "residual_action_id": action_id,
            "residual_action_hash": action_hash,
            "metric_values": metrics,
            "normalization_scale_by_metric": scales,
            "source_fact_ids": fact_ids,
            "source_fact_hashes": fact_hashes,
            "magnitude_level": magnitude_level,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPONENT_RESIDUAL_MAGNITUDE_OBSERVATION_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )

    @property
    def observation_id(self) -> str:
        return f"component-residual-magnitude:{self.semantic_content_hash}"

    def is_fresh_at(self, target_at: float, *, clock_domain_id: str) -> bool:
        target = _finite(target_at, name="target_at")
        return (
            _required(clock_domain_id, name="clock_domain_id") == self.clock_domain_id
            and self.available_at <= target < self.valid_until
        )


@dataclass(frozen=True)
class ComponentAllocationEvidencePolicy:
    """Hysteresis thresholds for one independent evidence dimension."""

    relative_enter_level: float
    relative_exit_level: float
    absolute_enter_level: float
    absolute_exit_level: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        relative_enter = _nonnegative(
            self.relative_enter_level,
            name="relative_enter_level",
        )
        relative_exit = _nonnegative(
            self.relative_exit_level,
            name="relative_exit_level",
        )
        absolute_enter = _nonnegative(
            self.absolute_enter_level,
            name="absolute_enter_level",
        )
        absolute_exit = _nonnegative(
            self.absolute_exit_level,
            name="absolute_exit_level",
        )
        if not (relative_exit < relative_enter <= absolute_exit < absolute_enter):
            raise ValueError(
                "Evidence thresholds must satisfy relative_exit < "
                "relative_enter <= absolute_exit < absolute_enter."
            )
        payload = {
            "relative_enter_level": relative_enter,
            "relative_exit_level": relative_exit,
            "absolute_enter_level": absolute_enter,
            "absolute_exit_level": absolute_exit,
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=(
                    COMPONENT_ALLOCATION_EVIDENCE_POLICY_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class ComponentAllocationPolicy:
    """Two-evidence hysteretic three-mode policy for one component."""

    component_id: str
    motion_source_role: str
    residual_source_role: str
    motion_policy: ComponentAllocationEvidencePolicy
    residual_magnitude_policy: ComponentAllocationEvidencePolicy
    minimum_dwell_sec: float = 0.0
    automatic_mode_ceiling: str = (
        COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER
    )
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, name="component_id")
        motion_role = _required(
            self.motion_source_role,
            name="motion_source_role",
        )
        residual_role = _required(
            self.residual_source_role,
            name="residual_source_role",
        )
        if not isinstance(self.motion_policy, ComponentAllocationEvidencePolicy):
            raise TypeError(
                "motion_policy must be ComponentAllocationEvidencePolicy."
            )
        if not isinstance(
            self.residual_magnitude_policy,
            ComponentAllocationEvidencePolicy,
        ):
            raise TypeError(
                "residual_magnitude_policy must be "
                "ComponentAllocationEvidencePolicy."
            )
        dwell = _nonnegative(self.minimum_dwell_sec, name="minimum_dwell_sec")
        automatic_mode_ceiling = _required(
            self.automatic_mode_ceiling,
            name="automatic_mode_ceiling",
        )
        if automatic_mode_ceiling not in COMPOSITION_ALLOCATION_MODES_DEVELOPMENT:
            raise ValueError("automatic_mode_ceiling is not a supported mode.")
        payload = {
            "component_id": component_id,
            "motion_source_role": motion_role,
            "residual_source_role": residual_role,
            "motion_policy_hash": self.motion_policy.semantic_content_hash,
            "residual_magnitude_policy_hash": (
                self.residual_magnitude_policy.semantic_content_hash
            ),
            "minimum_dwell_sec": dwell,
            "automatic_mode_ceiling": automatic_mode_ceiling,
        }
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "motion_source_role", motion_role)
        object.__setattr__(self, "residual_source_role", residual_role)
        object.__setattr__(self, "minimum_dwell_sec", dwell)
        object.__setattr__(
            self,
            "automatic_mode_ceiling",
            automatic_mode_ceiling,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_ALLOCATION_POLICY_HASH_DOMAIN_DEVELOPMENT,
            ),
        )


@dataclass(frozen=True)
class CompositionAllocationSpec:
    """Predeclare source roles and every dynamically allocated component."""

    allocation_spec_id: str
    base_absolute_role: str
    relative_overlay_role: str
    absolute_overlay_role: str
    component_policies: tuple[ComponentAllocationPolicy, ...]
    transition_continuity_policy_hash: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        spec_id = _required(self.allocation_spec_id, name="allocation_spec_id")
        roles = (
            _required(self.base_absolute_role, name="base_absolute_role"),
            _required(self.relative_overlay_role, name="relative_overlay_role"),
            _required(self.absolute_overlay_role, name="absolute_overlay_role"),
        )
        if len(set(roles)) != 3:
            raise ValueError("Allocation source roles must be distinct.")
        policies = tuple(self.component_policies)
        if not policies or any(
            not isinstance(policy, ComponentAllocationPolicy) for policy in policies
        ):
            raise TypeError(
                "component_policies must contain ComponentAllocationPolicy."
            )
        components = tuple(policy.component_id for policy in policies)
        if len(components) != len(set(components)):
            raise ValueError("Allocation component IDs must be unique.")
        if any(
            policy.residual_source_role != roles[1]
            for policy in policies
        ):
            raise ValueError(
                "Every component residual_source_role must equal the "
                "predeclared relative_overlay_role."
            )
        continuity_hash = self.transition_continuity_policy_hash
        if continuity_hash is not None:
            continuity_hash = _required(
                continuity_hash,
                name="transition_continuity_policy_hash",
            )
        payload = {
            "allocation_spec_id": spec_id,
            "base_absolute_role": roles[0],
            "relative_overlay_role": roles[1],
            "absolute_overlay_role": roles[2],
            "component_policy_hashes": tuple(
                policy.semantic_content_hash for policy in policies
            ),
            "authority_semantics": "composer_stays_authoritative_soft_selection",
        }
        if continuity_hash is not None:
            payload["transition_continuity_policy_hash"] = continuity_hash
        object.__setattr__(self, "allocation_spec_id", spec_id)
        object.__setattr__(self, "base_absolute_role", roles[0])
        object.__setattr__(self, "relative_overlay_role", roles[1])
        object.__setattr__(self, "absolute_overlay_role", roles[2])
        object.__setattr__(self, "component_policies", policies)
        object.__setattr__(
            self,
            "transition_continuity_policy_hash",
            continuity_hash,
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPOSITION_ALLOCATION_SPEC_HASH_DOMAIN_DEVELOPMENT,
            ),
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(policy.component_id for policy in self.component_policies)

    @property
    def policy_by_component(self) -> Mapping[str, ComponentAllocationPolicy]:
        return MappingProxyType(
            {policy.component_id: policy for policy in self.component_policies}
        )


__all__ = [
    "COMPOSITION_ALLOCATION_MODES_DEVELOPMENT",
    "COMPOSITION_ALLOCATION_MODE_ABSOLUTE_OVERLAY_OWNER",
    "COMPOSITION_ALLOCATION_MODE_BASE_ONLY",
    "COMPOSITION_ALLOCATION_MODE_RELATIVE_OVERLAY",
    "COMPOSITION_ALLOCATION_STATUSES_DEVELOPMENT",
    "COMPOSITION_ALLOCATION_STATUS_NO_DECISION",
    "COMPOSITION_ALLOCATION_STATUS_READY",
    "ComponentAllocationEvidencePolicy",
    "ComponentAllocationPolicy",
    "ComponentResidualMagnitudeObservation",
    "ComponentSourceActivityObservation",
    "CompositionAllocationSpec",
]

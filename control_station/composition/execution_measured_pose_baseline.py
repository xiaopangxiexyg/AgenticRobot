"""Bind one fresh measured pose as a non-authoritative composition baseline.

The target runtime already owns measured-qpos capture and FK projection.  This
adapter consumes that typed ``ExecutionMeasurementObservation`` and freezes one
absolute pose baseline for an explicit composition activation/generation.  It
does not read a backend, evaluate IK, publish a proposal, run safety, or dispatch
a command.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

from control_composition.activation import CompositionActivationScope
from control_composition.composition import (
    CompositionInputDraft,
    build_composition_input_draft,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ComponentAction,
    ComponentActionSchema,
    ExecutionMeasurementObservation,
)

EXECUTION_MEASURED_POSE_BASELINE_PROFILE_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.execution_measured_pose_baseline_profile.development"
)
EXECUTION_MEASURED_POSE_BASELINE_ANCHOR_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.execution_measured_pose_baseline_anchor.development"
)
EXECUTION_MEASURED_POSE_BASELINE_SEMANTIC_BINDING_KIND = (
    "execution_measured_pose_baseline"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _finite(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite.")
    return normalized


@dataclass(frozen=True)
class ExecutionMeasuredPoseBaselineProfile:
    """Exact measurement and composition identities for one baseline source."""

    profile_id: str
    input_role: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    action_schema: ComponentActionSchema
    component_ids: tuple[str, ...]
    observation_binding_id: str
    observation_plan_hash: str
    maximum_measurement_age_sec: float
    enabled: bool = False
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "profile_id",
                "input_role",
                "runtime_session_id",
                "control_domain_id",
                "producer_id",
                "observation_binding_id",
                "observation_plan_hash",
            )
        }
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be bool.")
        epoch = int(self.producer_epoch)
        if (
            isinstance(self.producer_epoch, bool)
            or epoch != self.producer_epoch
            or epoch < 0
        ):
            raise ValueError("producer_epoch must be a non-negative integer.")
        if not isinstance(self.action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be ComponentActionSchema.")
        components = tuple(
            _required(value, name="component_id") for value in self.component_ids
        )
        if (
            not components
            or len(components) != len(set(components))
            or not set(components).issubset(self.action_schema.component_ids)
        ):
            raise ValueError(
                "component_ids must be a unique non-empty action-schema subset."
            )
        for component_id in components:
            payload = self.action_schema.payload_schema_by_component[component_id]
            if (
                payload.representation_family != "cartesian_pose"
                or payload.value_semantics != "absolute_target"
            ):
                raise ValueError(
                    "Measured pose baselines require absolute-target "
                    f"cartesian_pose components; got {component_id!r}."
                )
        maximum_age = _finite(
            self.maximum_measurement_age_sec,
            name="maximum_measurement_age_sec",
        )
        if maximum_age <= 0.0:
            raise ValueError("maximum_measurement_age_sec must be positive.")
        for name, value in text.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "producer_epoch", epoch)
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(self, "maximum_measurement_age_sec", maximum_age)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    **text,
                    "producer_epoch": epoch,
                    "action_schema_hash": self.action_schema.semantic_content_hash,
                    "component_ids": components,
                    "maximum_measurement_age_sec": maximum_age,
                    "enabled": self.enabled,
                },
                domain=(
                    EXECUTION_MEASURED_POSE_BASELINE_PROFILE_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


@dataclass(frozen=True)
class ExecutionMeasuredPoseBaselineAnchor:
    """One accepted measurement frozen as an activation-local pose anchor."""

    profile_hash: str
    composition_activation_id: str
    composition_activation_hash: str
    anchor_generation: int
    source_measurement_id: str
    source_measurement_hash: str
    measured_at: float
    available_at: float
    captured_at: float
    clock_domain_id: str
    action: ComponentAction
    semantic_content_hash: str = field(init=False)
    anchor_id: str = field(init=False)

    def __post_init__(self) -> None:
        text = {
            name: _required(getattr(self, name), name=name)
            for name in (
                "profile_hash",
                "composition_activation_id",
                "composition_activation_hash",
                "source_measurement_id",
                "source_measurement_hash",
                "clock_domain_id",
            )
        }
        generation = int(self.anchor_generation)
        if (
            isinstance(self.anchor_generation, bool)
            or generation != self.anchor_generation
            or generation < 0
        ):
            raise ValueError("anchor_generation must be a non-negative integer.")
        measured_at = _finite(self.measured_at, name="measured_at")
        available_at = _finite(self.available_at, name="available_at")
        captured_at = _finite(self.captured_at, name="captured_at")
        if available_at + 1e-12 < measured_at:
            raise ValueError(
                "Baseline measurement cannot arrive before it was measured."
            )
        if captured_at + 1e-12 < available_at:
            raise ValueError(
                "Baseline capture cannot precede measurement availability."
            )
        if not isinstance(self.action, ComponentAction):
            raise TypeError("action must be ComponentAction.")
        payload = {
            **text,
            "anchor_generation": generation,
            "measured_at": measured_at,
            "available_at": available_at,
            "captured_at": captured_at,
            "action_hash": self.action.semantic_content_hash,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=(EXECUTION_MEASURED_POSE_BASELINE_ANCHOR_HASH_DOMAIN_DEVELOPMENT),
        )
        for name, value in payload.items():
            if name != "action_hash":
                object.__setattr__(self, name, value)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "anchor_id",
            f"execution-measured-pose-baseline:{generation}:{semantic_hash}",
        )


@dataclass(frozen=True)
class ExecutionMeasuredPoseBaselineCapture:
    anchor: ExecutionMeasuredPoseBaselineAnchor
    input_draft: CompositionInputDraft

    def __post_init__(self) -> None:
        if not isinstance(self.anchor, ExecutionMeasuredPoseBaselineAnchor):
            raise TypeError("anchor must be ExecutionMeasuredPoseBaselineAnchor.")
        if not isinstance(self.input_draft, CompositionInputDraft):
            raise TypeError("input_draft must be CompositionInputDraft.")
        draft = self.input_draft
        if (
            draft.action.semantic_content_hash
            != self.anchor.action.semantic_content_hash
        ):
            raise ValueError("Baseline draft action differs from its anchor.")
        if draft.source_fact_ids != (self.anchor.source_measurement_id,) or (
            draft.source_fact_hashes != (self.anchor.source_measurement_hash,)
        ):
            raise ValueError("Baseline draft does not retain measurement lineage.")
        kind = EXECUTION_MEASURED_POSE_BASELINE_SEMANTIC_BINDING_KIND
        if (
            draft.semantic_binding_id_by_kind.get(kind) != self.anchor.anchor_id
            or draft.semantic_binding_hash_by_kind.get(kind)
            != self.anchor.semantic_content_hash
        ):
            raise ValueError("Baseline draft does not bind its anchor identity.")


class ExecutionMeasuredPoseBaselineController:
    """Capture exactly one fresh target baseline per explicit generation."""

    def __init__(self, profile: ExecutionMeasuredPoseBaselineProfile) -> None:
        if not isinstance(profile, ExecutionMeasuredPoseBaselineProfile):
            raise TypeError("profile must be ExecutionMeasuredPoseBaselineProfile.")
        self.profile = profile
        self._lock = threading.RLock()
        self._activation_scope: CompositionActivationScope | None = None
        self._current: ExecutionMeasuredPoseBaselineCapture | None = None
        self._next_sequence = 0

    @property
    def current(self) -> ExecutionMeasuredPoseBaselineCapture | None:
        with self._lock:
            return self._current

    @property
    def activation_scope(self) -> CompositionActivationScope | None:
        with self._lock:
            return self._activation_scope

    def activate(
        self,
        activation_scope: CompositionActivationScope,
    ) -> CompositionActivationScope:
        if not self.profile.enabled:
            raise RuntimeError("Measured pose baseline profile is disabled.")
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be CompositionActivationScope.")
        if self.profile.input_role not in activation_scope.required_roles:
            raise ValueError("Activation does not require the measured baseline role.")
        with self._lock:
            current = self._activation_scope
            if (
                current is not None
                and current.activation_id == activation_scope.activation_id
            ):
                if (
                    current.semantic_content_hash
                    != activation_scope.semantic_content_hash
                ):
                    raise ValueError(
                        "Composition activation ID has conflicting content."
                    )
                return current
            self._activation_scope = activation_scope
            self._current = None
            return activation_scope

    def capture(
        self,
        observation: ExecutionMeasurementObservation,
        *,
        anchor_generation: int,
        captured_at: float,
        effective_until: float,
        context_ids: tuple[str, ...] = (),
    ) -> ExecutionMeasuredPoseBaselineCapture:
        """Validate first, then atomically advance the retained baseline."""

        if not self.profile.enabled:
            raise RuntimeError("Measured pose baseline profile is disabled.")
        if not isinstance(observation, ExecutionMeasurementObservation):
            raise TypeError("observation must be ExecutionMeasurementObservation.")
        with self._lock:
            scope = self._activation_scope
            if scope is None:
                raise RuntimeError("activate() is required before baseline capture.")
            generation = int(anchor_generation)
            if (
                isinstance(anchor_generation, bool)
                or generation != anchor_generation
                or generation < 0
            ):
                raise ValueError("anchor_generation must be a non-negative integer.")
            expected_generation = (
                0
                if self._current is None
                else self._current.anchor.anchor_generation + 1
            )
            if generation != expected_generation:
                raise ValueError(
                    "Baseline capture generation must advance exactly once."
                )
            captured = _finite(captured_at, name="captured_at")
            expires = _finite(effective_until, name="effective_until")
            if expires <= captured:
                raise ValueError("effective_until must follow captured_at.")
            self._validate_observation(
                observation,
                scope=scope,
                captured_at=captured,
            )
            action = ComponentAction(
                action_schema=self.profile.action_schema,
                values_by_component={
                    component_id: observation.value_by_component[component_id]
                    for component_id in self.profile.component_ids
                },
            )
            anchor = ExecutionMeasuredPoseBaselineAnchor(
                profile_hash=self.profile.semantic_content_hash,
                composition_activation_id=scope.activation_id,
                composition_activation_hash=scope.semantic_content_hash,
                anchor_generation=generation,
                source_measurement_id=observation.measurement_id,
                source_measurement_hash=observation.semantic_content_hash,
                measured_at=observation.measured_at,
                available_at=observation.available_at,
                captured_at=captured,
                clock_domain_id=observation.clock_domain_id,
                action=action,
            )
            kind = EXECUTION_MEASURED_POSE_BASELINE_SEMANTIC_BINDING_KIND
            draft = build_composition_input_draft(
                runtime_session_id=self.profile.runtime_session_id,
                control_domain_id=self.profile.control_domain_id,
                producer_id=self.profile.producer_id,
                producer_epoch=self.profile.producer_epoch,
                sequence=self._next_sequence,
                input_role=self.profile.input_role,
                action=action,
                produced_at=observation.available_at,
                produced_clock_domain_id=observation.clock_domain_id,
                effective_from=captured,
                effective_until=expires,
                effective_clock_domain_id=scope.clock_domain_id,
                context_ids=context_ids,
                observation_cut_id=observation.measurement_id,
                source_fact_ids=(observation.measurement_id,),
                source_fact_hashes=(observation.semantic_content_hash,),
                semantic_binding_id_by_kind={kind: anchor.anchor_id},
                semantic_binding_hash_by_kind={kind: anchor.semantic_content_hash},
                composition_activation_id=scope.activation_id,
                composition_activation_hash=scope.semantic_content_hash,
                audit_metadata={
                    "composition_contributor": True,
                    "contributor_kind": "execution_measured_pose_baseline",
                    "baseline_profile_hash": self.profile.semantic_content_hash,
                    "anchor_generation": generation,
                    "observation_binding_id": observation.observation_binding_id,
                    "observation_plan_hash": observation.observation_plan_hash,
                },
            )
            result = ExecutionMeasuredPoseBaselineCapture(
                anchor=anchor,
                input_draft=draft,
            )
            self._current = result
            self._next_sequence += 1
            return result

    def _validate_observation(
        self,
        observation: ExecutionMeasurementObservation,
        *,
        scope: CompositionActivationScope,
        captured_at: float,
    ) -> None:
        profile = self.profile
        if observation.control_domain_id != profile.control_domain_id:
            raise ValueError("Measurement belongs to another control domain.")
        if (
            observation.observation_binding_id != profile.observation_binding_id
            or observation.observation_plan_hash != profile.observation_plan_hash
        ):
            raise ValueError("Measurement belongs to another observation binding.")
        if observation.clock_domain_id != scope.clock_domain_id:
            raise ValueError("Measurement and activation clock domains differ.")
        if tuple(observation.value_by_component) != profile.component_ids:
            raise ValueError(
                "Measurement component coverage/order differs from profile."
            )
        for component_id in profile.component_ids:
            payload = profile.action_schema.payload_schema_by_component[component_id]
            if (
                observation.representation_by_component[component_id]
                != "cartesian_pose"
                or observation.schema_hash_by_component[component_id]
                != payload.semantic_content_hash
            ):
                raise ValueError(
                    f"Measurement component {component_id!r} schema/frame binding differs."
                )
        if observation.measured_at + 1e-12 < scope.input_barrier_at or (
            observation.available_at + 1e-12 < scope.input_barrier_at
        ):
            raise ValueError("Measurement predates the composition activation barrier.")
        if captured_at + 1e-12 < observation.available_at:
            raise ValueError("captured_at precedes measurement availability.")
        age = captured_at - observation.measured_at
        if age > profile.maximum_measurement_age_sec + 1e-12:
            raise ValueError("Measurement is stale for baseline capture.")


__all__ = [
    "EXECUTION_MEASURED_POSE_BASELINE_ANCHOR_HASH_DOMAIN_DEVELOPMENT",
    "EXECUTION_MEASURED_POSE_BASELINE_PROFILE_HASH_DOMAIN_DEVELOPMENT",
    "EXECUTION_MEASURED_POSE_BASELINE_SEMANTIC_BINDING_KIND",
    "ExecutionMeasuredPoseBaselineAnchor",
    "ExecutionMeasuredPoseBaselineCapture",
    "ExecutionMeasuredPoseBaselineController",
    "ExecutionMeasuredPoseBaselineProfile",
]

"""Typed component-wise composition semantics.

These contracts describe *what* one composition operation is allowed to do.
They are independent from the strategy implementation that evaluates them.
The implementation identity, context requirements, and latency budget belong
to ``FormalCompositionStrategySpec`` instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action.hand_spec import HandActionSpec


FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2 = "formal_component_strategy_v2"
COMPONENT_COMPOSITION_RULE_SET_HASH_DOMAIN_V1 = (
    "dexteleop.component_composition_rule_set.v1"
)
FORMAL_COMPONENT_COMPOSITION_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.formal_component_composition_binding.v1"
)
HAND_ACTION_SPEC_HASH_DOMAIN_V1 = "dexteleop.hand_action_spec.semantic.v1"
COMPONENT_MISSING_BEHAVIORS_V1 = frozenset(
    {"no_action", "invalid", "renormalize_available", "use_baseline"}
)
ALLOCATION_CONTEXT_MISSING_BEHAVIORS_DEVELOPMENT = frozenset(
    {"no_action", "use_baseline"}
)
COMPONENT_SEMANTIC_PARAMETER_KEYS_V1 = frozenset(
    {
        "component_rule_set",
        "component_rule_set_hash",
        "rules",
        "weight_by_role",
        "weights_by_role",
        "missing_behavior",
        "missing_candidate_behavior",
        "max_translation_disagreement_m",
        "max_rotation_disagreement_rad",
        "value_contract_hash",
        "allocation_spec_hash",
        "allocation_context_channel",
        "context_missing_behavior",
        "base_absolute_role",
        "relative_overlay_role",
        "absolute_overlay_role",
    }
)


def _required(value: object, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be non-empty.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _roles_and_weights(
    contributing_roles: tuple[str, ...],
    weight_by_role: Mapping[str, float],
) -> tuple[tuple[str, ...], Mapping[str, float]]:
    roles = tuple(
        _required(value, field_name="contributing_role")
        for value in contributing_roles
    )
    if not roles or len(roles) != len(set(roles)):
        raise ValueError(
            "Component contributing_roles must be non-empty and unique."
        )
    weights: dict[str, float] = {}
    for raw_role, raw_weight in dict(weight_by_role).items():
        role = _required(raw_role, field_name="weighted_role")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                "Component role weights must be finite and non-negative."
            )
        weights[role] = weight
    if set(weights) != set(roles) or not any(
        value > 0.0 for value in weights.values()
    ):
        raise ValueError(
            "Component rules require one weight per contributing role and "
            "at least one positive weight."
        )
    return roles, MappingProxyType(weights)


def _missing_behavior(value: object) -> str:
    behavior = _required(value, field_name="missing_behavior")
    if behavior not in COMPONENT_MISSING_BEHAVIORS_V1:
        raise ValueError(f"Unsupported component missing behavior {behavior!r}.")
    return behavior


def _weighted_missing_behavior(value: object) -> str:
    behavior = _missing_behavior(value)
    if behavior == "use_baseline":
        raise ValueError(
            "use_baseline is only valid for relative-overlay rules."
        )
    return behavior


def _relative_missing_behavior(value: object) -> str:
    behavior = _missing_behavior(value)
    if behavior == "renormalize_available":
        raise ValueError(
            "renormalize_available is not valid for relative-overlay rules."
        )
    return behavior


def _allocation_missing_behavior(value: object) -> str:
    behavior = _missing_behavior(value)
    if behavior not in {"no_action", "invalid"}:
        raise ValueError(
            "Allocation-selected rules support only no_action or invalid "
            "for a missing selected input."
        )
    return behavior


@dataclass(frozen=True)
class PoseComponentCompositionRule:
    """One pose-valued component fusion rule.

    Pose values use the canonical ``xyz + quat_xyzw`` component payload.
    Frame and controlled-frame equality are checked by the numerical strategy
    before fusion.  Either disagreement limit may be ``None`` to keep the
    difference observable without turning it into a pre-composition rejection
    gate.  Canonical motion safety remains downstream.
    """

    component_id: str
    contributing_roles: tuple[str, ...]
    weight_by_role: Mapping[str, float]
    missing_behavior: str = "no_action"
    max_translation_disagreement_m: float | None = 0.20
    max_rotation_disagreement_rad: float | None = math.radians(60.0)

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, field_name="component_id")
        roles, weights = _roles_and_weights(
            self.contributing_roles,
            self.weight_by_role,
        )
        translation_limit = (
            None
            if self.max_translation_disagreement_m is None
            else float(self.max_translation_disagreement_m)
        )
        rotation_limit = (
            None
            if self.max_rotation_disagreement_rad is None
            else float(self.max_rotation_disagreement_rad)
        )
        if translation_limit is not None and (
            not math.isfinite(translation_limit) or translation_limit < 0.0
        ):
            raise ValueError(
                "Pose translation disagreement limit must be finite and "
                "non-negative when enabled."
            )
        if rotation_limit is not None and (
            not math.isfinite(rotation_limit)
            or rotation_limit < 0.0
            or rotation_limit >= math.pi
        ):
            raise ValueError(
                "Pose rotation disagreement limit must be in [0, pi) when "
                "enabled."
            )
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "contributing_roles", roles)
        object.__setattr__(self, "weight_by_role", weights)
        object.__setattr__(
            self,
            "missing_behavior",
            _weighted_missing_behavior(self.missing_behavior),
        )
        object.__setattr__(
            self,
            "max_translation_disagreement_m",
            translation_limit,
        )
        object.__setattr__(
            self,
            "max_rotation_disagreement_rad",
            rotation_limit,
        )

    @property
    def representation(self) -> str:
        return "pose"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "representation": self.representation,
            "contributing_roles": self.contributing_roles,
            "weight_by_role": self.weight_by_role,
            "missing_behavior": self.missing_behavior,
            "max_translation_disagreement_m": (
                self.max_translation_disagreement_m
            ),
            "max_rotation_disagreement_rad": (
                self.max_rotation_disagreement_rad
            ),
            "rotation_fusion": (
                "deterministic_reference_shortest_path_so3_tangent_mean_v1"
            ),
        }


@dataclass(frozen=True)
class QposComponentCompositionRule:
    """One qpos-valued component interpolation rule.

    ``value_contract_hash`` identifies the exact component layout and active
    mask. The current RobotAction-v1 adapter binds this to a
    ``HandActionSpec``; a future component-keyed carrier may bind another typed
    qpos layout without changing the generic composition semantics.
    """

    component_id: str
    contributing_roles: tuple[str, ...]
    weight_by_role: Mapping[str, float]
    value_contract_hash: str
    missing_behavior: str = "no_action"

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, field_name="component_id")
        roles, weights = _roles_and_weights(
            self.contributing_roles,
            self.weight_by_role,
        )
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "contributing_roles", roles)
        object.__setattr__(self, "weight_by_role", weights)
        object.__setattr__(
            self,
            "value_contract_hash",
            _required(
                self.value_contract_hash,
                field_name="value_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "missing_behavior",
            _weighted_missing_behavior(self.missing_behavior),
        )

    @property
    def representation(self) -> str:
        return "qpos"

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "representation": self.representation,
            "contributing_roles": self.contributing_roles,
            "weight_by_role": self.weight_by_role,
            "missing_behavior": self.missing_behavior,
            "value_contract_hash": self.value_contract_hash,
            "interpolation": "active_mask_weighted_linear_v1",
        }


@dataclass(frozen=True)
class RelativePoseComponentCompositionRule:
    """Apply one pose delta to a sampled absolute baseline.

    This rule is deliberately stateless. Translation is added in the
    baseline pose reference frame, while rotation is right-multiplied in the
    baseline controlled frame:

    ``p_out = p_baseline + delta_p``
    ``R_out = R_baseline * R_delta``

    Reusing the same delta record therefore never integrates it again.
    """

    component_id: str
    baseline_role: str
    delta_role: str
    missing_behavior: str = "use_baseline"

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, field_name="component_id")
        baseline_role = _required(
            self.baseline_role,
            field_name="baseline_role",
        )
        delta_role = _required(self.delta_role, field_name="delta_role")
        if baseline_role == delta_role:
            raise ValueError(
                "Relative pose baseline_role and delta_role must differ."
            )
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "baseline_role", baseline_role)
        object.__setattr__(self, "delta_role", delta_role)
        object.__setattr__(
            self,
            "missing_behavior",
            _relative_missing_behavior(self.missing_behavior),
        )

    @property
    def representation(self) -> str:
        return "pose"

    @property
    def contributing_roles(self) -> tuple[str, ...]:
        return (self.baseline_role, self.delta_role)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "representation": self.representation,
            "composition": "follow_baseline_relative_delta_v1",
            "baseline_role": self.baseline_role,
            "delta_role": self.delta_role,
            "missing_behavior": self.missing_behavior,
            "translation_application": "reference_frame_additive",
            "rotation_application": "baseline_local_right_multiply",
        }


@dataclass(frozen=True)
class RelativeQposComponentCompositionRule:
    """Apply one joint-position delta to a sampled absolute baseline."""

    component_id: str
    baseline_role: str
    delta_role: str
    baseline_value_contract_hash: str
    delta_value_contract_hash: str
    missing_behavior: str = "use_baseline"

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, field_name="component_id")
        baseline_role = _required(
            self.baseline_role,
            field_name="baseline_role",
        )
        delta_role = _required(self.delta_role, field_name="delta_role")
        if baseline_role == delta_role:
            raise ValueError(
                "Relative qpos baseline_role and delta_role must differ."
            )
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "baseline_role", baseline_role)
        object.__setattr__(self, "delta_role", delta_role)
        object.__setattr__(
            self,
            "baseline_value_contract_hash",
            _required(
                self.baseline_value_contract_hash,
                field_name="baseline_value_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "delta_value_contract_hash",
            _required(
                self.delta_value_contract_hash,
                field_name="delta_value_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "missing_behavior",
            _relative_missing_behavior(self.missing_behavior),
        )

    @property
    def representation(self) -> str:
        return "qpos"

    @property
    def contributing_roles(self) -> tuple[str, ...]:
        return (self.baseline_role, self.delta_role)

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "representation": self.representation,
            "composition": "follow_baseline_relative_delta_v1",
            "baseline_role": self.baseline_role,
            "delta_role": self.delta_role,
            "missing_behavior": self.missing_behavior,
            "baseline_value_contract_hash": (
                self.baseline_value_contract_hash
            ),
            "delta_value_contract_hash": self.delta_value_contract_hash,
            "application": "elementwise_additive",
        }


@dataclass(frozen=True)
class AllocatedPoseComponentCompositionRule:
    """Select base, relative overlay, or absolute overlay for one pose.

    The selected mode comes from one typed allocation decision context.  Even
    ``absolute_overlay_owner`` remains a composition choice; it does not change
    the runtime authority owner.
    """

    component_id: str
    base_absolute_role: str
    relative_overlay_role: str
    absolute_overlay_role: str
    allocation_spec_hash: str
    allocation_context_channel: str
    missing_behavior: str = "no_action"
    context_missing_behavior: str = "no_action"

    def __post_init__(self) -> None:
        component_id = _required(self.component_id, field_name="component_id")
        roles = (
            _required(
                self.base_absolute_role,
                field_name="base_absolute_role",
            ),
            _required(
                self.relative_overlay_role,
                field_name="relative_overlay_role",
            ),
            _required(
                self.absolute_overlay_role,
                field_name="absolute_overlay_role",
            ),
        )
        if len(set(roles)) != 3:
            raise ValueError("Allocated composition roles must be distinct.")
        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "base_absolute_role", roles[0])
        object.__setattr__(self, "relative_overlay_role", roles[1])
        object.__setattr__(self, "absolute_overlay_role", roles[2])
        object.__setattr__(
            self,
            "allocation_spec_hash",
            _required(
                self.allocation_spec_hash,
                field_name="allocation_spec_hash",
            ),
        )
        object.__setattr__(
            self,
            "allocation_context_channel",
            _required(
                self.allocation_context_channel,
                field_name="allocation_context_channel",
            ),
        )
        object.__setattr__(
            self,
            "missing_behavior",
            _allocation_missing_behavior(self.missing_behavior),
        )
        context_missing = _required(
            self.context_missing_behavior,
            field_name="context_missing_behavior",
        )
        if context_missing not in ALLOCATION_CONTEXT_MISSING_BEHAVIORS_DEVELOPMENT:
            raise ValueError("Unsupported allocation context missing behavior.")
        object.__setattr__(
            self,
            "context_missing_behavior",
            context_missing,
        )

    @property
    def representation(self) -> str:
        return "pose"

    @property
    def contributing_roles(self) -> tuple[str, ...]:
        return (
            self.base_absolute_role,
            self.relative_overlay_role,
            self.absolute_overlay_role,
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "representation": self.representation,
            "composition": "allocation_selected_pose_v1",
            "base_absolute_role": self.base_absolute_role,
            "relative_overlay_role": self.relative_overlay_role,
            "absolute_overlay_role": self.absolute_overlay_role,
            "allocation_spec_hash": self.allocation_spec_hash,
            "allocation_context_channel": self.allocation_context_channel,
            "missing_behavior": self.missing_behavior,
            **(
                {}
                if self.context_missing_behavior == "no_action"
                else {"context_missing_behavior": self.context_missing_behavior}
            ),
            "relative_translation_application": "reference_frame_additive",
            "relative_rotation_application": (
                "baseline_local_right_multiply"
            ),
            "authority_semantics": (
                "composer_stays_authoritative_soft_selection"
            ),
        }


@dataclass(frozen=True)
class AllocatedQposComponentCompositionRule:
    """Select base, relative overlay, or absolute overlay for one qpos."""

    component_id: str
    base_absolute_role: str
    relative_overlay_role: str
    absolute_overlay_role: str
    baseline_value_contract_hash: str
    delta_value_contract_hash: str
    allocation_spec_hash: str
    allocation_context_channel: str
    missing_behavior: str = "no_action"
    context_missing_behavior: str = "no_action"

    def __post_init__(self) -> None:
        pose_shape = AllocatedPoseComponentCompositionRule(
            component_id=self.component_id,
            base_absolute_role=self.base_absolute_role,
            relative_overlay_role=self.relative_overlay_role,
            absolute_overlay_role=self.absolute_overlay_role,
            allocation_spec_hash=self.allocation_spec_hash,
            allocation_context_channel=self.allocation_context_channel,
            missing_behavior=self.missing_behavior,
            context_missing_behavior=self.context_missing_behavior,
        )
        object.__setattr__(self, "component_id", pose_shape.component_id)
        object.__setattr__(
            self,
            "base_absolute_role",
            pose_shape.base_absolute_role,
        )
        object.__setattr__(
            self,
            "relative_overlay_role",
            pose_shape.relative_overlay_role,
        )
        object.__setattr__(
            self,
            "absolute_overlay_role",
            pose_shape.absolute_overlay_role,
        )
        object.__setattr__(
            self,
            "baseline_value_contract_hash",
            _required(
                self.baseline_value_contract_hash,
                field_name="baseline_value_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "delta_value_contract_hash",
            _required(
                self.delta_value_contract_hash,
                field_name="delta_value_contract_hash",
            ),
        )
        object.__setattr__(
            self,
            "allocation_spec_hash",
            pose_shape.allocation_spec_hash,
        )
        object.__setattr__(
            self,
            "allocation_context_channel",
            pose_shape.allocation_context_channel,
        )
        object.__setattr__(
            self,
            "missing_behavior",
            pose_shape.missing_behavior,
        )
        object.__setattr__(
            self,
            "context_missing_behavior",
            pose_shape.context_missing_behavior,
        )

    @property
    def representation(self) -> str:
        return "qpos"

    @property
    def contributing_roles(self) -> tuple[str, ...]:
        return (
            self.base_absolute_role,
            self.relative_overlay_role,
            self.absolute_overlay_role,
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "representation": self.representation,
            "composition": "allocation_selected_qpos_v1",
            "base_absolute_role": self.base_absolute_role,
            "relative_overlay_role": self.relative_overlay_role,
            "absolute_overlay_role": self.absolute_overlay_role,
            "baseline_value_contract_hash": (
                self.baseline_value_contract_hash
            ),
            "delta_value_contract_hash": self.delta_value_contract_hash,
            "allocation_spec_hash": self.allocation_spec_hash,
            "allocation_context_channel": self.allocation_context_channel,
            "missing_behavior": self.missing_behavior,
            **(
                {}
                if self.context_missing_behavior == "no_action"
                else {"context_missing_behavior": self.context_missing_behavior}
            ),
            "relative_application": "elementwise_additive",
            "authority_semantics": (
                "composer_stays_authoritative_soft_selection"
            ),
        }


ComponentCompositionRule = (
    PoseComponentCompositionRule
    | QposComponentCompositionRule
    | RelativePoseComponentCompositionRule
    | RelativeQposComponentCompositionRule
    | AllocatedPoseComponentCompositionRule
    | AllocatedQposComponentCompositionRule
)


@dataclass(frozen=True)
class ComponentCompositionRuleSet:
    """Complete component semantics for one additive CompositionSpec-v2 mode."""

    rule_set_id: str
    action_space: str
    rules: tuple[ComponentCompositionRule, ...]
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        rule_set_id = _required(self.rule_set_id, field_name="rule_set_id")
        action_space = _required(self.action_space, field_name="action_space")
        rules = tuple(self.rules)
        if not rules:
            raise ValueError("Component rule set must contain at least one rule.")
        if any(
            not isinstance(
                rule,
                (
                    PoseComponentCompositionRule,
                    QposComponentCompositionRule,
                    RelativePoseComponentCompositionRule,
                    RelativeQposComponentCompositionRule,
                    AllocatedPoseComponentCompositionRule,
                    AllocatedQposComponentCompositionRule,
                ),
            )
            for rule in rules
        ):
            raise TypeError("Component rule set contains an unsupported rule.")
        component_ids = tuple(rule.component_id for rule in rules)
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("Component rule IDs must be unique.")
        payload = {
            "rule_set_id": rule_set_id,
            "action_space": action_space,
            "rules": tuple(rule.semantic_payload() for rule in rules),
        }
        object.__setattr__(self, "rule_set_id", rule_set_id)
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=COMPONENT_COMPOSITION_RULE_SET_HASH_DOMAIN_V1,
            ),
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(rule.component_id for rule in self.rules)

    @property
    def contributing_roles(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                role
                for rule in self.rules
                for role in rule.contributing_roles
            )
        )

    def semantic_payload(self) -> dict[str, Any]:
        return {
            "rule_set_id": self.rule_set_id,
            "action_space": self.action_space,
            "rules": tuple(rule.semantic_payload() for rule in self.rules),
        }


@dataclass(frozen=True)
class FormalComponentCompositionBinding:
    """Bind semantics, typed rules, and one implementation without merging them."""

    composition_spec_hash: str
    component_rule_set_hash: str
    formal_strategy_spec_hash: str
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        payload = {
            "composition_spec_hash": _required(
                self.composition_spec_hash,
                field_name="composition_spec_hash",
            ),
            "component_rule_set_hash": _required(
                self.component_rule_set_hash,
                field_name="component_rule_set_hash",
            ),
            "formal_strategy_spec_hash": _required(
                self.formal_strategy_spec_hash,
                field_name="formal_strategy_spec_hash",
            ),
        }
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=FORMAL_COMPONENT_COMPOSITION_BINDING_HASH_DOMAIN_V1,
            ),
        )

    @property
    def binding_id(self) -> str:
        return (
            "formal-component-composition-binding:"
            f"{self.semantic_content_hash}"
        )


def hand_action_spec_semantic_hash(spec: HandActionSpec) -> str:
    if not isinstance(spec, HandActionSpec):
        raise TypeError("spec must be a HandActionSpec.")
    return canonical_content_hash(
        spec.to_metadata(),
        domain=HAND_ACTION_SPEC_HASH_DOMAIN_V1,
    )


__all__ = [
    "COMPONENT_COMPOSITION_RULE_SET_HASH_DOMAIN_V1",
    "COMPONENT_MISSING_BEHAVIORS_V1",
    "COMPONENT_SEMANTIC_PARAMETER_KEYS_V1",
    "FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2",
    "FORMAL_COMPONENT_COMPOSITION_BINDING_HASH_DOMAIN_V1",
    "HAND_ACTION_SPEC_HASH_DOMAIN_V1",
    "ComponentCompositionRule",
    "ComponentCompositionRuleSet",
    "AllocatedPoseComponentCompositionRule",
    "AllocatedQposComponentCompositionRule",
    "FormalComponentCompositionBinding",
    "PoseComponentCompositionRule",
    "QposComponentCompositionRule",
    "RelativePoseComponentCompositionRule",
    "RelativeQposComponentCompositionRule",
    "hand_action_spec_semantic_hash",
]

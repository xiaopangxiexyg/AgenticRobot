"""Reference formal strategy for typed pose/qpos component composition.

The numerical implementation consumes one immutable formal-strategy input and
returns one decision. It owns no authority, canonical safety, IK, reference
generation, dispatch, device state, or model-worker lifecycle.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.action.hand_spec import HandActionSpec
from robot_motion.action.component_action import ComponentAction

from .component_contract import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    RelativePoseComponentCompositionRule,
    RelativeQposComponentCompositionRule,
    hand_action_spec_semantic_hash,
)
from .core import CompositionInputRecord, CompositionSpec
from .formal_strategy import (
    CompositionStrategyDecision,
    FormalCompositionStrategyInput,
    FormalCompositionStrategySpec,
)
from .strategy_control import CompositionStrategyControlBinding


FORMAL_COMPONENT_COMPOSITION_STRATEGY_ID_V2 = (
    "typed_pose_qpos_component_composition_v2"
)
FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1 = (
    "weighted_component_fusion_v1"
)
FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1 = (
    "discrete_component_owner_v1"
)
FORMAL_COMPONENT_STRATEGY_MODES_V1 = (
    FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
    FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1,
)


def _positive_weights(
    weight_by_role: Mapping[str, float],
) -> dict[str, float]:
    return {
        role: float(weight)
        for role, weight in weight_by_role.items()
        if float(weight) > 0.0
    }


def _component_available(
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


def _fuse_pose_values(
    records: tuple[CompositionInputRecord, ...],
    *,
    component_id: str,
    weights: Mapping[str, float],
    max_translation_disagreement_m: float | None,
    max_rotation_disagreement_rad: float | None,
) -> np.ndarray:
    values = {
        record.input_role: np.asarray(
            record.action.value(component_id).values,
            dtype=np.float64,
        )
        for record in records
    }
    roles = tuple(sorted(values))
    for index, first in enumerate(roles):
        for second in roles[index + 1 :]:
            translation = float(
                np.linalg.norm(values[first][:3] - values[second][:3])
            )
            rotation = float(
                (
                    Rotation.from_quat(values[first][3:7]).inv()
                    * Rotation.from_quat(values[second][3:7])
                ).magnitude()
            )
            if (
                max_translation_disagreement_m is not None
                and translation > max_translation_disagreement_m
            ):
                raise ValueError("translation_disagreement")
            if (
                max_rotation_disagreement_rad is not None
                and rotation > max_rotation_disagreement_rad
            ):
                raise ValueError("rotation_disagreement")
    reference_role = min(
        roles,
        key=lambda role: (-float(weights[role]), role),
    )
    reference = Rotation.from_quat(values[reference_role][3:7])
    translation = sum(
        float(weights[role]) * values[role][:3] for role in roles
    )
    rotvec = sum(
        float(weights[role])
        * (
            reference.inv() * Rotation.from_quat(values[role][3:7])
        ).as_rotvec()
        for role in roles
    )
    quaternion = (reference * Rotation.from_rotvec(rotvec)).as_quat()
    if float(np.dot(quaternion, values[reference_role][3:7])) < 0.0:
        quaternion = -quaternion
    return np.concatenate((translation, quaternion)).astype(np.float32)


def _selected_records_and_weights(
    strategy_input: FormalCompositionStrategyInput,
    *,
    rule: PoseComponentCompositionRule | QposComponentCompositionRule,
) -> tuple[
    tuple[CompositionInputRecord, ...],
    Mapping[str, float],
    tuple[str, ...],
]:
    selected = strategy_input.input_snapshot.selected_input_by_role
    positive = _positive_weights(rule.weight_by_role)
    available = tuple(
        selected[role]
        for role in rule.contributing_roles
        if (
            role in positive
            and role in selected
            and _component_available(
                selected[role],
                component_id=rule.component_id,
                representation=rule.representation,
            )
        )
    )
    available_roles = {record.input_role for record in available}
    missing = tuple(
        role for role in positive if role not in available_roles
    )
    if not available:
        return (), MappingProxyType({}), missing
    total = sum(positive[record.input_role] for record in available)
    normalized = {
        record.input_role: positive[record.input_role] / total
        for record in available
    }
    return available, MappingProxyType(normalized), missing


def _record_matches_declared_schema(
    strategy_input: FormalCompositionStrategyInput,
    record: CompositionInputRecord,
) -> bool:
    expected = (
        strategy_input.input_snapshot.input_action_schema_hash_by_role.get(
            record.input_role,
            strategy_input.input_snapshot.action_schema_hash,
        )
    )
    return record.action.action_schema.semantic_content_hash == expected


def _relative_pose_value(
    baseline: CompositionInputRecord,
    delta: CompositionInputRecord,
    *,
    component_id: str,
) -> np.ndarray:
    baseline_schema = (
        baseline.action.action_schema.payload_schema_by_component[component_id]
    )
    delta_schema = (
        delta.action.action_schema.payload_schema_by_component[component_id]
    )
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
        Rotation.from_quat(baseline_value[3:7])
        * Rotation.from_quat(delta_value[3:7])
    ).as_quat()
    if float(np.dot(quaternion, baseline_value[3:7])) < 0.0:
        quaternion = -quaternion
    return np.concatenate((translation, quaternion)).astype(
        baseline_schema.dtype
    )


def _relative_qpos_value(
    baseline: CompositionInputRecord,
    delta: CompositionInputRecord,
    *,
    rule: RelativeQposComponentCompositionRule,
) -> np.ndarray:
    baseline_schema = (
        baseline.action.action_schema.payload_schema_by_component[
            rule.component_id
        ]
    )
    delta_schema = delta.action.action_schema.payload_schema_by_component[
        rule.component_id
    ]
    if (
        baseline_schema.representation_family != "joint_position"
        or delta_schema.representation_family != "joint_position"
        or baseline_schema.value_semantics != "absolute_target"
        or delta_schema.value_semantics != "relative_delta"
    ):
        raise ValueError("relative_qpos_semantics_mismatch")
    if (
        baseline_schema.semantic_content_hash
        != rule.baseline_value_contract_hash
        or delta_schema.semantic_content_hash
        != rule.delta_value_contract_hash
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
            baseline.action.value(rule.component_id).values,
            dtype=np.float64,
        )
        + np.asarray(
            delta.action.value(rule.component_id).values,
            dtype=np.float64,
        )
    ).astype(baseline_schema.dtype)


class FormalComponentCompositionStrategyV2:
    """Minimal deterministic evaluator for one typed component rule set.

    Component semantics come exclusively from ``ComponentCompositionRuleSet``.
    ``FormalCompositionStrategySpec`` contributes only implementation identity,
    context declarations, implementation dependencies, and latency budget.
    """

    def __init__(
        self,
        *,
        strategy_spec: FormalCompositionStrategySpec,
        composition_spec: CompositionSpec,
        component_rule_set: ComponentCompositionRuleSet,
        qpos_value_spec_by_component: Mapping[str, HandActionSpec] | None = None,
    ) -> None:
        if not isinstance(strategy_spec, FormalCompositionStrategySpec):
            raise TypeError(
                "strategy_spec must be a FormalCompositionStrategySpec."
            )
        if not isinstance(composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(component_rule_set, ComponentCompositionRuleSet):
            raise TypeError(
                "component_rule_set must be a ComponentCompositionRuleSet."
            )
        if (
            composition_spec.algebra
            != FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2
        ):
            raise ValueError(
                "Formal component strategy requires "
                "formal_component_strategy_v2."
            )
        if (
            composition_spec.component_rule_set_hash
            != component_rule_set.semantic_content_hash
        ):
            raise ValueError(
                "CompositionSpec is bound to another component rule set."
            )
        if (
            strategy_spec.composition_spec_hash
            != composition_spec.semantic_content_hash
        ):
            raise ValueError(
                "Formal strategy is bound to another CompositionSpec."
            )
        if (
            strategy_spec.implementation_contract_hashes
            != (component_rule_set.semantic_content_hash,)
        ):
            raise ValueError(
                "Formal strategy implementation must bind exactly the "
                "component rule set."
            )
        if set(component_rule_set.contributing_roles) != set(
            composition_spec.input_roles
        ):
            raise ValueError(
                "Component rules must cover every CompositionSpec input role "
                "and may not introduce other roles."
            )
        for rule in component_rule_set.rules:
            if not isinstance(
                rule,
                (
                    RelativePoseComponentCompositionRule,
                    RelativeQposComponentCompositionRule,
                ),
            ):
                continue
            if rule.baseline_role not in composition_spec.depends_on_roles_by_role.get(
                rule.delta_role,
                (),
            ):
                raise ValueError(
                    "A relative-overlay delta role must explicitly depend on "
                    "its baseline role in CompositionSpec."
                )
            if not composition_spec.input_action_schema_hash_by_role:
                raise ValueError(
                    "Relative-overlay rules require role-specific input "
                    "action schemas."
                )
        qpos_specs = dict(qpos_value_spec_by_component or {})
        for component_id, value_spec in qpos_specs.items():
            if not isinstance(value_spec, HandActionSpec):
                raise TypeError(
                    f"Qpos value spec for {component_id!r} must be a "
                    "HandActionSpec."
                )
            matching_rule = next(
                (
                    rule
                    for rule in component_rule_set.rules
                    if rule.component_id == component_id
                ),
                None,
            )
            if (
                isinstance(matching_rule, QposComponentCompositionRule)
                and hand_action_spec_semantic_hash(value_spec)
                != matching_rule.value_contract_hash
            ):
                raise ValueError(
                    f"Legacy qpos component {component_id!r} is bound to "
                    "another value contract."
                )

        self._spec = strategy_spec
        self.composition_spec = composition_spec
        self.component_rule_set = component_rule_set
        self.qpos_value_spec_by_component = MappingProxyType(qpos_specs)
        self.supported_strategy_modes = FORMAL_COMPONENT_STRATEGY_MODES_V1

    @property
    def spec(self) -> FormalCompositionStrategySpec:
        return self._spec

    def decide(
        self,
        strategy_input: FormalCompositionStrategyInput,
    ) -> CompositionStrategyDecision:
        if strategy_input.strategy_spec != self.spec:
            raise ValueError("Formal strategy input uses another strategy spec.")
        if (
            strategy_input.composition_spec.semantic_content_hash
            != self.composition_spec.semantic_content_hash
        ):
            raise ValueError(
                "Formal strategy input uses another CompositionSpec."
            )

        return self._decide(
            strategy_input,
            composition_spec=self.composition_spec,
            component_rule_set=self.component_rule_set,
            strategy_mode=FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
        )

    def decide_with_control(
        self,
        strategy_input: FormalCompositionStrategyInput,
        binding: CompositionStrategyControlBinding,
    ) -> CompositionStrategyDecision:
        """Evaluate one immutable hot-control binding on this same instance."""

        if not isinstance(binding, CompositionStrategyControlBinding):
            raise TypeError(
                "binding must be a CompositionStrategyControlBinding."
            )
        revision = binding.revision
        if (
            strategy_input.strategy_control_binding_id
            != binding.binding_id
            or strategy_input.strategy_control_binding_hash
            != binding.semantic_content_hash
            or strategy_input.strategy_spec != revision.strategy_spec
            or strategy_input.composition_spec != revision.composition_spec
            or revision.strategy_spec.strategy_id != self.spec.strategy_id
            or revision.strategy_spec.strategy_version
            != self.spec.strategy_version
        ):
            raise ValueError(
                "Formal strategy input uses another hot-control binding."
            )
        if revision.strategy_mode not in self.supported_strategy_modes:
            raise ValueError("Unsupported formal component strategy mode.")
        rules = revision.component_rule_set
        contract = revision.dynamic_parameter_contract
        discrete_tie_role: str | None = None
        if contract is not None:
            discrete_tie_role = contract.base_role
            effective_rules = []
            alpha = binding.parameter_snapshot.alpha_by_component
            for rule in rules.rules:
                if rule.component_id not in contract.component_ids:
                    effective_rules.append(rule)
                    continue
                if isinstance(
                    rule,
                    (
                        RelativePoseComponentCompositionRule,
                        RelativeQposComponentCompositionRule,
                    ),
                ):
                    raise ValueError(
                        "Alpha hot-control is not defined for relative-overlay "
                        "rules."
                    )
                value = float(alpha[rule.component_id])
                effective_rules.append(
                    replace(
                        rule,
                        weight_by_role={
                            contract.base_role: 1.0 - value,
                            contract.overlay_role: value,
                        },
                    )
                )
            rules = replace(rules, rules=tuple(effective_rules))
        return self._decide(
            strategy_input,
            composition_spec=revision.composition_spec,
            component_rule_set=rules,
            strategy_mode=revision.strategy_mode,
            discrete_tie_role=discrete_tie_role,
        )

    def _decide(
        self,
        strategy_input: FormalCompositionStrategyInput,
        *,
        composition_spec: CompositionSpec,
        component_rule_set: ComponentCompositionRuleSet,
        strategy_mode: str,
        discrete_tie_role: str | None = None,
    ) -> CompositionStrategyDecision:
        selected_by_role = (
            strategy_input.input_snapshot.selected_input_by_role
        )
        missing_required = tuple(
            role
            for role in composition_spec.required_roles
            if role not in selected_by_role
        )
        if missing_required:
            return CompositionStrategyDecision(
                output_status=composition_spec.missing_required_behavior,
                reason="missing_required:" + ",".join(missing_required),
                composed_candidate=None,
            )

        output_values: dict[str, np.ndarray] = {}
        renormalized_components: list[str] = []
        output_action_schema = None

        for rule in component_rule_set.rules:
            if isinstance(
                rule,
                (
                    RelativePoseComponentCompositionRule,
                    RelativeQposComponentCompositionRule,
                ),
            ):
                baseline = selected_by_role.get(rule.baseline_role)
                delta = selected_by_role.get(rule.delta_role)
                if baseline is None:
                    return CompositionStrategyDecision(
                        output_status=(
                            "no_action"
                            if rule.missing_behavior == "use_baseline"
                            else rule.missing_behavior
                        ),
                        reason=(
                            f"relative_baseline_missing:{rule.component_id}:"
                            f"{rule.baseline_role}"
                        ),
                        composed_candidate=None,
                    )
                if not _component_available(
                    baseline,
                    component_id=rule.component_id,
                    representation=rule.representation,
                ):
                    return CompositionStrategyDecision(
                        output_status="no_action",
                        reason=(
                            f"relative_baseline_component_unavailable:"
                            f"{rule.component_id}"
                        ),
                        composed_candidate=None,
                    )
                if not _record_matches_declared_schema(
                    strategy_input,
                    baseline,
                ):
                    return CompositionStrategyDecision(
                        output_status="invalid",
                        reason=(
                            f"action_schema_mismatch:{rule.component_id}:"
                            f"{rule.baseline_role}"
                        ),
                        composed_candidate=None,
                    )
                if (
                    baseline.action.action_schema.semantic_content_hash
                    != strategy_input.input_snapshot.action_schema_hash
                ):
                    return CompositionStrategyDecision(
                        output_status="invalid",
                        reason=(
                            f"relative_baseline_output_schema_mismatch:"
                            f"{rule.component_id}"
                        ),
                        composed_candidate=None,
                    )
                if output_action_schema is None:
                    output_action_schema = baseline.action.action_schema
                elif (
                    output_action_schema.semantic_content_hash
                    != baseline.action.action_schema.semantic_content_hash
                ):
                    return CompositionStrategyDecision(
                        output_status="invalid",
                        reason="relative_output_schema_mismatch",
                        composed_candidate=None,
                    )
                if delta is None or not _component_available(
                    delta,
                    component_id=rule.component_id,
                    representation=rule.representation,
                ):
                    if rule.missing_behavior != "use_baseline":
                        return CompositionStrategyDecision(
                            output_status=rule.missing_behavior,
                            reason=(
                                f"relative_delta_missing:{rule.component_id}:"
                                f"{rule.delta_role}"
                            ),
                            composed_candidate=None,
                        )
                    output_values[rule.component_id] = np.array(
                        baseline.action.value(rule.component_id).values,
                        copy=True,
                    )
                    continue
                if not _record_matches_declared_schema(
                    strategy_input,
                    delta,
                ):
                    return CompositionStrategyDecision(
                        output_status="invalid",
                        reason=(
                            f"action_schema_mismatch:{rule.component_id}:"
                            f"{rule.delta_role}"
                        ),
                        composed_candidate=None,
                    )
                try:
                    if isinstance(
                        rule,
                        RelativePoseComponentCompositionRule,
                    ):
                        output_values[rule.component_id] = (
                            _relative_pose_value(
                                baseline,
                                delta,
                                component_id=rule.component_id,
                            )
                        )
                    else:
                        output_values[rule.component_id] = (
                            _relative_qpos_value(
                                baseline,
                                delta,
                                rule=rule,
                            )
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    return CompositionStrategyDecision(
                        output_status="invalid",
                        reason=(
                            f"component_invalid:{rule.component_id}:{exc}"
                        ),
                        composed_candidate=None,
                    )
                continue

            records, weights, missing = _selected_records_and_weights(
                strategy_input,
                rule=rule,
            )
            if missing and rule.missing_behavior != "renormalize_available":
                return CompositionStrategyDecision(
                    output_status=rule.missing_behavior,
                    reason=(
                        f"component_missing:{rule.component_id}:"
                        + ",".join(missing)
                    ),
                    composed_candidate=None,
                )
            if not records:
                return CompositionStrategyDecision(
                    output_status="no_action",
                    reason=f"component_unavailable:{rule.component_id}",
                    composed_candidate=None,
                )
            if missing:
                renormalized_components.append(rule.component_id)
            if any(
                not _record_matches_declared_schema(strategy_input, record)
                for record in records
            ):
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason=f"action_schema_mismatch:{rule.component_id}",
                    composed_candidate=None,
                )
            if any(
                record.action.action_schema.semantic_content_hash
                != strategy_input.input_snapshot.action_schema_hash
                for record in records
            ):
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason=f"weighted_input_output_schema_mismatch:{rule.component_id}",
                    composed_candidate=None,
                )
            if output_action_schema is None:
                output_action_schema = records[0].action.action_schema

            try:
                if (
                    strategy_mode
                    == FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1
                ):
                    maximum_weight = max(
                        float(weights[record.input_role])
                        for record in records
                    )
                    tied = tuple(
                        record
                        for record in records
                        if float(weights[record.input_role])
                        == maximum_weight
                    )
                    owner = next(
                        (
                            record
                            for record in tied
                            if record.input_role == discrete_tie_role
                        ),
                        min(tied, key=lambda record: record.input_role),
                    )
                    output_values[rule.component_id] = np.array(
                        owner.action.value(rule.component_id).values,
                        copy=True,
                    )
                elif isinstance(rule, PoseComponentCompositionRule):
                    output_values[rule.component_id] = _fuse_pose_values(
                        records,
                        component_id=rule.component_id,
                        weights=weights,
                        max_translation_disagreement_m=rule.max_translation_disagreement_m,
                        max_rotation_disagreement_rad=rule.max_rotation_disagreement_rad,
                    )
                else:
                    payload_schema = records[0].action.action_schema.payload_schema_by_component[
                        rule.component_id
                    ]
                    if rule.value_contract_hash not in {
                        payload_schema.semantic_content_hash,
                        *(
                            hand_action_spec_semantic_hash(spec)
                            for spec in self.qpos_value_spec_by_component.values()
                        ),
                    }:
                        raise ValueError("value_contract_mismatch")
                    output_values[rule.component_id] = sum(
                        weights[record.input_role]
                        * np.asarray(
                            record.action.value(rule.component_id).values,
                            dtype=np.float64,
                        )
                        for record in records
                    ).astype(payload_schema.dtype)
            except (KeyError, TypeError, ValueError) as exc:
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason=f"component_invalid:{rule.component_id}:{exc}",
                    composed_candidate=None,
                )
        if output_action_schema is None:
            return CompositionStrategyDecision(
                output_status="no_action",
                reason="no_output_action_schema",
                composed_candidate=None,
            )
        action = ComponentAction(
            action_schema=output_action_schema,
            values_by_component=output_values,
        )
        reason = (
            "formal_component_composition_renormalized:"
            + ",".join(renormalized_components)
            if renormalized_components
            else "formal_component_composition"
        )
        if strategy_mode != FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1:
            reason += f":{strategy_mode}"
        return CompositionStrategyDecision(
            output_status="ready",
            reason=reason,
            composed_candidate=action,
        )


__all__ = [
    "FORMAL_COMPONENT_COMPOSITION_STRATEGY_ID_V2",
    "FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1",
    "FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1",
    "FORMAL_COMPONENT_STRATEGY_MODES_V1",
    "FormalComponentCompositionStrategyV2",
]

"""Reference formal strategy for typed pose/qpos component composition.

The numerical implementation consumes one immutable formal-strategy input and
returns one decision. It owns no authority, canonical safety, IK, reference
generation, dispatch, device state, or model-worker lifecycle.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from robot_motion.action.hand_spec import HandActionSpec
from robot_motion.action.execution_target import (
    project_robot_action_to_components,
)
from robot_motion.action.proposal import freeze_robot_action
from robot_motion.action.schema import RobotAction, split_effector_name

from .component_contract import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    hand_action_spec_semantic_hash,
)
from .core import CompositionInputRecord, CompositionSpec
from .formal_strategy import (
    CompositionStrategyDecision,
    FormalCompositionStrategyInput,
    FormalCompositionStrategySpec,
)
from .pose import fuse_weighted_se3_pose_actions


FORMAL_COMPONENT_COMPOSITION_STRATEGY_ID_V2 = (
    "typed_pose_qpos_component_composition_v2"
)


def _qpos_padding_value(spec: HandActionSpec) -> float:
    return (
        float("nan")
        if spec.invalid_slot_value.strip().lower() == "nan"
        else 0.0
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
    if component_id not in action.active_effectors:
        return False
    try:
        side, kind = split_effector_name(component_id)
    except ValueError:
        return False
    if representation == "pose":
        return kind == "arm" and side in action.arm_pose_by_side
    if representation == "qpos":
        return kind == "hand" and side in action.hand_qpos_by_side
    return False


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
        if component_rule_set.action_space.strip() == "":
            raise ValueError("Component rule set action_space must be non-empty.")

        qpos_specs = dict(qpos_value_spec_by_component or {})
        qpos_rules = {
            rule.component_id: rule
            for rule in component_rule_set.rules
            if isinstance(rule, QposComponentCompositionRule)
        }
        if set(qpos_specs) != set(qpos_rules):
            raise ValueError(
                "The RobotAction-v1 adapter requires one qpos value spec for "
                "every qpos component."
            )
        for component_id, value_spec in qpos_specs.items():
            if not isinstance(value_spec, HandActionSpec):
                raise TypeError(
                    f"Qpos value spec for {component_id!r} must be a "
                    "HandActionSpec."
                )
            try:
                side, kind = split_effector_name(component_id)
            except ValueError as exc:
                raise ValueError(
                    "RobotAction-v1 qpos components must use canonical "
                    "left/right hand component IDs."
                ) from exc
            if kind != "hand" or value_spec.side != side:
                raise ValueError(
                    "Qpos component ID and HandActionSpec side must match."
                )
            if (
                hand_action_spec_semantic_hash(value_spec)
                != qpos_rules[component_id].value_contract_hash
            ):
                raise ValueError(
                    f"Qpos component {component_id!r} is bound to another "
                    "value contract."
                )

        for rule in component_rule_set.rules:
            try:
                _, kind = split_effector_name(rule.component_id)
            except ValueError as exc:
                raise ValueError(
                    "The current RobotAction-v1 adapter supports canonical "
                    "left/right arm and hand component IDs only."
                ) from exc
            if (
                isinstance(rule, PoseComponentCompositionRule)
                and kind != "arm"
            ):
                raise ValueError(
                    "RobotAction-v1 pose rules must target arm components."
                )

        self._spec = strategy_spec
        self.composition_spec = composition_spec
        self.component_rule_set = component_rule_set
        self.qpos_value_spec_by_component = MappingProxyType(qpos_specs)

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

        selected_by_role = (
            strategy_input.input_snapshot.selected_input_by_role
        )
        missing_required = tuple(
            role
            for role in self.composition_spec.required_roles
            if role not in selected_by_role
        )
        if missing_required:
            return CompositionStrategyDecision(
                output_status=self.composition_spec.missing_required_behavior,
                reason="missing_required:" + ",".join(missing_required),
                composed_action=None,
            )

        pose_by_side: dict[str, np.ndarray] = {}
        hand_qpos_by_side: dict[str, np.ndarray] = {}
        frames: dict[str, str] = {}
        controlled_frames: dict[str, str] = {}
        hand_specs_by_side: dict[str, dict[str, object]] = {}
        active_effectors: list[str] = []
        renormalized_components: list[str] = []

        for rule in self.component_rule_set.rules:
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
                    composed_action=None,
                )
            if not records:
                return CompositionStrategyDecision(
                    output_status="no_action",
                    reason=f"component_unavailable:{rule.component_id}",
                    composed_action=None,
                )
            if missing:
                renormalized_components.append(rule.component_id)
            if any(
                record.action.action_space
                != self.component_rule_set.action_space
                for record in records
            ):
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason=f"action_space_mismatch:{rule.component_id}",
                    composed_action=None,
                )

            try:
                side, _ = split_effector_name(rule.component_id)
                if isinstance(rule, PoseComponentCompositionRule):
                    fused = fuse_weighted_se3_pose_actions(
                        {
                            record.input_role: (
                                project_robot_action_to_components(
                                    record.action,
                                    (rule.component_id,),
                                )
                            )
                            for record in records
                        },
                        weights_by_source_slot=weights,
                        target_effective_at=(
                            strategy_input.input_snapshot.target_effective_at
                        ),
                        max_translation_disagreement_m=(
                            rule.max_translation_disagreement_m
                        ),
                        max_rotation_disagreement_rad=(
                            rule.max_rotation_disagreement_rad
                        ),
                        component_dependencies=(rule.component_id,),
                        source=(
                            "formal_component_strategy:"
                            f"{self.spec.strategy_id}"
                        ),
                    )
                    pose_by_side[side] = fused.arm_pose_by_side[side]
                    frames[rule.component_id] = fused.frame_by_effector[
                        rule.component_id
                    ]
                    controlled_frames[rule.component_id] = (
                        fused.controlled_frame_by_effector[rule.component_id]
                    )
                else:
                    value_spec = self.qpos_value_spec_by_component[
                        rule.component_id
                    ]
                    active_values = sum(
                        weights[record.input_role]
                        * value_spec.active_values(
                            record.action.hand_qpos_by_side[side]
                        )
                        for record in records
                    )
                    output = np.full(
                        (value_spec.slot_count,),
                        _qpos_padding_value(value_spec),
                        dtype=np.float32,
                    )
                    output[
                        np.asarray(value_spec.active_mask, dtype=bool)
                    ] = active_values
                    hand_qpos_by_side[side] = output
                    hand_specs_by_side[side] = value_spec.to_metadata()
            except (KeyError, TypeError, ValueError) as exc:
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason=f"component_invalid:{rule.component_id}:{exc}",
                    composed_action=None,
                )
            active_effectors.append(rule.component_id)

        metadata = (
            {}
            if not hand_specs_by_side
            else {"hand_action_specs_by_side": hand_specs_by_side}
        )
        action = freeze_robot_action(
            RobotAction(
                action_space=self.component_rule_set.action_space,
                timestamp=(
                    strategy_input.input_snapshot.target_effective_at
                ),
                arm_pose_by_side=pose_by_side,
                hand_qpos_by_side=hand_qpos_by_side,
                active_effectors=tuple(active_effectors),
                frame_by_effector=frames,
                controlled_frame_by_effector=controlled_frames,
                source=(
                    "formal_component_strategy:"
                    f"{self.spec.strategy_id}"
                ),
                metadata=metadata,
            )
        )
        reason = (
            "formal_component_composition_renormalized:"
            + ",".join(renormalized_components)
            if renormalized_components
            else "formal_component_composition"
        )
        return CompositionStrategyDecision(
            output_status="ready",
            reason=reason,
            composed_action=action,
        )


__all__ = [
    "FORMAL_COMPONENT_COMPOSITION_STRATEGY_ID_V2",
    "FormalComponentCompositionStrategyV2",
]

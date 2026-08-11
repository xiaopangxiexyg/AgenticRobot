"""Generic formal-composition contracts for one allocation specification."""

from __future__ import annotations

from dataclasses import dataclass

from robot_motion.action import ComponentActionSchema
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleSpec,
)

from ..composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    AllocatedPoseComponentCompositionRule,
    AllocatedQposComponentCompositionRule,
    ComponentCompositionRuleSet,
    CompositionSpec,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
)
from .contracts import CompositionAllocationSpec
from .continuity_contracts import AllocationTransitionContinuityPolicy


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class AllocatedCompositionContracts:
    allocation_spec: CompositionAllocationSpec
    output_action_schema: ComponentActionSchema
    relative_action_schema: ComponentActionSchema
    component_rule_set: ComponentCompositionRuleSet
    composition_spec: CompositionSpec
    strategy_spec: FormalCompositionStrategySpec
    strategy: FormalComponentCompositionStrategyV2
    residual_anchor_lifecycle_spec: ResidualAnchorLifecycleSpec


def build_allocated_composition_contracts(
    *,
    contract_id: str,
    output_action_schema: ComponentActionSchema,
    relative_action_schema: ComponentActionSchema,
    allocation_spec: CompositionAllocationSpec,
    allocation_context_channel: str,
    required_roles: tuple[str, ...] | None = None,
    allow_missing_allocation_context: bool = False,
    action_space: str = "component-keyed",
    maximum_decision_latency_sec: float = 0.01,
    transition_continuity_policy: (
        AllocationTransitionContinuityPolicy | None
    ) = None,
) -> AllocatedCompositionContracts:
    """Build pose/qpos rules without choosing a source, backend, or authority."""

    identifier = _required(contract_id, name="contract_id")
    channel = _required(
        allocation_context_channel,
        name="allocation_context_channel",
    )
    if not isinstance(output_action_schema, ComponentActionSchema):
        raise TypeError("output_action_schema must be ComponentActionSchema.")
    if not isinstance(relative_action_schema, ComponentActionSchema):
        raise TypeError("relative_action_schema must be ComponentActionSchema.")
    if not isinstance(allocation_spec, CompositionAllocationSpec):
        raise TypeError("allocation_spec must be CompositionAllocationSpec.")
    if not isinstance(allow_missing_allocation_context, bool):
        raise TypeError("allow_missing_allocation_context must be a bool.")
    if transition_continuity_policy is not None:
        if not isinstance(
            transition_continuity_policy,
            AllocationTransitionContinuityPolicy,
        ):
            raise TypeError("transition_continuity_policy must be typed or None.")
        if (
            transition_continuity_policy.component_ids
            != allocation_spec.component_ids
            or transition_continuity_policy.semantic_content_hash
            != allocation_spec.transition_continuity_policy_hash
        ):
            raise ValueError(
                "Transition continuity policy differs from the allocation spec."
            )
    elif allocation_spec.transition_continuity_policy_hash is not None:
        raise ValueError("Allocation spec requires its transition continuity policy.")
    if output_action_schema.component_ids != allocation_spec.component_ids:
        raise ValueError(
            "Allocation policies must exactly cover the output components."
        )
    if relative_action_schema.component_ids != output_action_schema.component_ids:
        raise ValueError("Relative and output schemas must cover the same components.")
    lifecycle_spec = ResidualAnchorLifecycleSpec(
        spec_id=f"{identifier}.residual-anchor-lifecycle",
        source_role=allocation_spec.relative_overlay_role,
        component_ids=allocation_spec.component_ids,
        absolute_action_schema_hash=output_action_schema.semantic_content_hash,
        relative_action_schema_hash=relative_action_schema.semantic_content_hash,
    )
    rules = []
    for component_id in output_action_schema.component_ids:
        absolute = output_action_schema.payload_schema_by_component[component_id]
        relative = relative_action_schema.payload_schema_by_component[component_id]
        if absolute.value_semantics != "absolute_target":
            raise ValueError(
                f"Component {component_id!r} output must be absolute_target."
            )
        if relative.value_semantics != "relative_delta":
            raise ValueError(
                f"Component {component_id!r} overlay must be relative_delta."
            )
        common = {
            "component_id": component_id,
            "base_absolute_role": allocation_spec.base_absolute_role,
            "relative_overlay_role": allocation_spec.relative_overlay_role,
            "absolute_overlay_role": allocation_spec.absolute_overlay_role,
            "allocation_spec_hash": allocation_spec.semantic_content_hash,
            "allocation_context_channel": channel,
            "missing_behavior": "no_action",
            "context_missing_behavior": (
                "use_baseline"
                if allow_missing_allocation_context
                else "no_action"
            ),
        }
        if (
            absolute.representation_family == "cartesian_pose"
            and relative.representation_family == "cartesian_pose"
        ):
            rules.append(AllocatedPoseComponentCompositionRule(**common))
        elif (
            absolute.representation_family == "joint_position"
            and relative.representation_family == "joint_position"
        ):
            rules.append(
                AllocatedQposComponentCompositionRule(
                    **common,
                    baseline_value_contract_hash=(
                        absolute.semantic_content_hash
                    ),
                    delta_value_contract_hash=relative.semantic_content_hash,
                )
            )
        else:
            raise ValueError(
                f"Component {component_id!r} has unsupported/mismatched "
                "allocated representations."
            )
    rule_set = ComponentCompositionRuleSet(
        rule_set_id=f"{identifier}.rules",
        action_space=_required(action_space, name="action_space"),
        rules=tuple(rules),
    )
    roles = (
        allocation_spec.base_absolute_role,
        allocation_spec.relative_overlay_role,
        allocation_spec.absolute_overlay_role,
    )
    required = roles if required_roles is None else tuple(required_roles)
    composition_spec = CompositionSpec(
        spec_id=f"{identifier}.composition",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=roles,
        required_roles=required,
        depends_on_roles_by_role={
            allocation_spec.relative_overlay_role: (
                allocation_spec.base_absolute_role,
            ),
        },
        missing_required_behavior="no_action",
        component_rule_set_hash=rule_set.semantic_content_hash,
        action_schema_hash=output_action_schema.semantic_content_hash,
        input_action_schema_hash_by_role={
            allocation_spec.base_absolute_role: (
                output_action_schema.semantic_content_hash
            ),
            allocation_spec.relative_overlay_role: (
                relative_action_schema.semantic_content_hash
            ),
            allocation_spec.absolute_overlay_role: (
                output_action_schema.semantic_content_hash
            ),
        },
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id=f"{identifier}.strategy",
        strategy_version="development",
        composition_spec_hash=composition_spec.semantic_content_hash,
        required_context_channels=(
            () if allow_missing_allocation_context else (channel,)
        ),
        optional_context_channels=(
            (channel,) if allow_missing_allocation_context else ()
        ),
        required_residual_anchor_lifecycle_spec_hash=(
            lifecycle_spec.semantic_content_hash
        ),
        implementation_contract_hashes=(rule_set.semantic_content_hash,),
        maximum_decision_latency_sec=float(maximum_decision_latency_sec),
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        allocation_spec_by_hash={
            allocation_spec.semantic_content_hash: allocation_spec
        },
        residual_anchor_lifecycle_spec=lifecycle_spec,
        allocation_transition_continuity_policy_by_spec_hash=(
            {}
            if transition_continuity_policy is None
            else {
                allocation_spec.semantic_content_hash: (
                    transition_continuity_policy
                )
            }
        ),
    )
    return AllocatedCompositionContracts(
        allocation_spec=allocation_spec,
        output_action_schema=output_action_schema,
        relative_action_schema=relative_action_schema,
        component_rule_set=rule_set,
        composition_spec=composition_spec,
        strategy_spec=strategy_spec,
        strategy=strategy,
        residual_anchor_lifecycle_spec=lifecycle_spec,
    )


__all__ = [
    "AllocatedCompositionContracts",
    "build_allocated_composition_contracts",
]

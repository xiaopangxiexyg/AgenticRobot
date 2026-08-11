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

from control_composition.allocation.contracts import (
    COMPOSITION_ALLOCATION_MODE_BASE_ONLY,
    CompositionAllocationSpec,
)
from control_composition.allocation.context import (
    selected_allocation_modes_from_context,
)
from control_composition.allocation.continuity import (
    AllocationTransitionContinuityRuntime,
)
from control_composition.allocation.continuity_contracts import (
    AllocationTransitionContinuityPolicy,
)
from control_composition.activation import CompositionActivationScope
from control_composition.residual_anchor_lifecycle import (
    ResidualAnchorLifecycleSpec,
)
from control_composition.allocation.strategy import (
    evaluate_allocated_component_rule,
)

from .component_contract import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    AllocatedPoseComponentCompositionRule,
    AllocatedQposComponentCompositionRule,
    ComponentCompositionRuleSet,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    RelativePoseComponentCompositionRule,
    RelativeQposComponentCompositionRule,
    hand_action_spec_semantic_hash,
)
from .core import CompositionEvaluation, CompositionInputRecord, CompositionSpec
from .component_operations import (
    component_record_available,
    record_matches_declared_schema,
    relative_pose_component_value,
    relative_qpos_component_value,
)
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
            and component_record_available(
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
        allocation_spec_by_hash: (
            Mapping[str, CompositionAllocationSpec] | None
        ) = None,
        residual_anchor_lifecycle_spec: (
            ResidualAnchorLifecycleSpec | None
        ) = None,
        allocation_transition_continuity_policy_by_spec_hash: (
            Mapping[str, AllocationTransitionContinuityPolicy] | None
        ) = None,
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

        allocation_specs = dict(allocation_spec_by_hash or {})
        for spec_hash, allocation_spec in allocation_specs.items():
            if not isinstance(allocation_spec, CompositionAllocationSpec):
                raise TypeError(
                    "allocation_spec_by_hash values must be "
                    "CompositionAllocationSpec."
                )
            if spec_hash != allocation_spec.semantic_content_hash:
                raise ValueError("Allocation spec mapping key/hash mismatch.")
        allocated_rules = tuple(
            rule
            for rule in component_rule_set.rules
            if isinstance(
                rule,
                (
                    AllocatedPoseComponentCompositionRule,
                    AllocatedQposComponentCompositionRule,
                ),
            )
        )
        if allocated_rules:
            if not isinstance(
                residual_anchor_lifecycle_spec,
                ResidualAnchorLifecycleSpec,
            ):
                raise TypeError(
                    "Allocated component strategy requires a residual anchor "
                    "lifecycle spec."
                )
            if (
                strategy_spec.required_residual_anchor_lifecycle_spec_hash
                != residual_anchor_lifecycle_spec.semantic_content_hash
            ):
                raise ValueError(
                    "Allocated strategy and residual anchor lifecycle spec differ."
                )
        elif residual_anchor_lifecycle_spec is not None:
            raise ValueError(
                "A non-allocated strategy must not bind residual anchor lifecycle."
            )
        for rule in allocated_rules:
            allocation_spec = allocation_specs.get(rule.allocation_spec_hash)
            if allocation_spec is None:
                raise ValueError(
                    "Allocated component rule requires its typed allocation spec."
                )
            if (
                rule.base_absolute_role != allocation_spec.base_absolute_role
                or rule.relative_overlay_role
                != allocation_spec.relative_overlay_role
                or rule.absolute_overlay_role
                != allocation_spec.absolute_overlay_role
                or rule.component_id not in allocation_spec.component_ids
            ):
                raise ValueError(
                    "Allocated component rule disagrees with allocation spec."
                )
            if rule.allocation_context_channel not in (
                strategy_spec.context_channels
            ):
                raise ValueError(
                    "Allocated component rule requires its context channel in "
                    "FormalCompositionStrategySpec."
                )
            if rule.base_absolute_role not in (
                composition_spec.depends_on_roles_by_role.get(
                    rule.relative_overlay_role,
                    (),
                )
            ):
                raise ValueError(
                    "Allocated relative role must depend on its absolute base."
                )
            if not composition_spec.input_action_schema_hash_by_role:
                raise ValueError(
                    "Allocated rules require role-specific input action schemas."
                )
        for allocation_hash in {
            rule.allocation_spec_hash for rule in allocated_rules
        }:
            allocation_spec = allocation_specs[allocation_hash]
            bound_rules = tuple(
                rule
                for rule in allocated_rules
                if rule.allocation_spec_hash == allocation_hash
            )
            if tuple(rule.component_id for rule in bound_rules) != (
                allocation_spec.component_ids
            ):
                raise ValueError(
                    "Allocated rules must cover their allocation spec components "
                    "in declared order."
                )
            if len(
                {rule.allocation_context_channel for rule in bound_rules}
            ) != 1:
                raise ValueError(
                    "One allocation spec must use one context channel."
                )
        continuity_policies = dict(
            allocation_transition_continuity_policy_by_spec_hash or {}
        )
        for allocation_hash, allocation_spec in allocation_specs.items():
            policy = continuity_policies.get(allocation_hash)
            required_hash = allocation_spec.transition_continuity_policy_hash
            if required_hash is None:
                if policy is not None:
                    raise ValueError(
                        "Allocation strategy supplied undeclared continuity."
                    )
                continue
            if not isinstance(policy, AllocationTransitionContinuityPolicy):
                raise TypeError(
                    "Continuity-enabled allocation requires its typed policy."
                )
            if (
                policy.semantic_content_hash != required_hash
                or policy.component_ids != allocation_spec.component_ids
            ):
                raise ValueError(
                    "Allocation continuity policy differs from its spec."
                )
        if set(continuity_policies) - set(allocation_specs):
            raise ValueError("Continuity policy references an unknown allocation spec.")

        self._spec = strategy_spec
        self.composition_spec = composition_spec
        self.component_rule_set = component_rule_set
        self.qpos_value_spec_by_component = MappingProxyType(qpos_specs)
        self.allocation_spec_by_hash = MappingProxyType(allocation_specs)
        self.residual_anchor_lifecycle_spec = residual_anchor_lifecycle_spec
        self.allocation_transition_continuity_runtime_by_spec_hash = (
            MappingProxyType(
                {
                    spec_hash: AllocationTransitionContinuityRuntime(policy)
                    for spec_hash, policy in continuity_policies.items()
                }
            )
        )
        self.supported_strategy_modes = FORMAL_COMPONENT_STRATEGY_MODES_V1

    @property
    def spec(self) -> FormalCompositionStrategySpec:
        return self._spec

    def activate_allocation_transition_continuity(
        self,
        activation_scope: CompositionActivationScope,
    ) -> None:
        """Reset opt-in target-seam state for one exact composition activation."""

        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError("activation_scope must be typed.")
        for runtime in (
            self.allocation_transition_continuity_runtime_by_spec_hash.values()
        ):
            runtime.activate(
                composition_activation_id=activation_scope.activation_id,
                composition_activation_hash=activation_scope.semantic_content_hash,
            )

    def _continuity_publication_binding(
        self,
        evaluation: CompositionEvaluation,
    ) -> tuple[AllocationTransitionContinuityRuntime, str] | None:
        if not isinstance(evaluation, CompositionEvaluation):
            raise TypeError("evaluation must be a CompositionEvaluation.")
        receipt = evaluation.strategy_result_data.get(
            "allocation_transition_continuity"
        )
        runtimes = tuple(
            self.allocation_transition_continuity_runtime_by_spec_hash.values()
        )
        if receipt is None:
            if runtimes and evaluation.output_status == "ready":
                raise ValueError(
                    "Continuity-enabled ready evaluation has no preparation receipt."
                )
            return None
        if not isinstance(receipt, Mapping):
            raise TypeError("Continuity preparation receipt must be a mapping.")
        policy_hash = str(receipt.get("policy_hash", "")).strip()
        receipt_hash = str(receipt.get("receipt_hash", "")).strip()
        if not policy_hash or not receipt_hash:
            raise ValueError("Continuity preparation receipt identity is incomplete.")
        runtime = next(
            (
                item
                for item in runtimes
                if item.policy.semantic_content_hash == policy_hash
            ),
            None,
        )
        if runtime is None:
            raise ValueError("Continuity receipt names an unknown policy.")
        if (
            receipt.get("composition_activation_id")
            != evaluation.composition_activation_id
            or receipt.get("composition_activation_hash")
            != evaluation.composition_activation_hash
        ):
            raise ValueError("Continuity receipt belongs to another activation.")
        return runtime, receipt_hash

    def validate_publication_transaction(
        self,
        evaluation: CompositionEvaluation,
    ) -> None:
        """Validate prepared target state before an external proposal is admitted."""

        binding = self._continuity_publication_binding(evaluation)
        if binding is not None:
            runtime, receipt_hash = binding
            runtime.validate_prepared(receipt_hash=receipt_hash)

    def commit_publication_transaction(
        self,
        evaluation: CompositionEvaluation,
    ) -> None:
        """Advance target continuity after exact proposal admission succeeds."""

        binding = self._continuity_publication_binding(evaluation)
        if binding is not None:
            runtime, receipt_hash = binding
            runtime.commit_prepared(receipt_hash=receipt_hash)

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
            apply_transition_continuity=True,
        )

    def decide_projected(
        self,
        strategy_input: object,
    ) -> CompositionStrategyDecision:
        """Evaluate an explicit non-admitted projection with the same rules.

        The projection carrier intentionally lives in another identity domain
        and cannot enter formal result promotion.  Structural validation here
        is kept local to avoid making the production formal strategy protocol
        depend on a trajectory or preview package.
        """

        if getattr(strategy_input, "promotable", None) is not False:
            raise TypeError(
                "Projected strategy input must explicitly be non-promotable."
            )
        if getattr(strategy_input, "strategy_spec", None) != self.spec:
            raise ValueError("Projected strategy input uses another strategy spec.")
        composition_spec = getattr(strategy_input, "composition_spec", None)
        if not isinstance(composition_spec, CompositionSpec) or (
            composition_spec.semantic_content_hash
            != self.composition_spec.semantic_content_hash
        ):
            raise ValueError(
                "Projected strategy input uses another CompositionSpec."
            )
        input_snapshot = getattr(strategy_input, "input_snapshot", None)
        if getattr(input_snapshot, "promotable", None) is not False:
            raise TypeError("Projected input snapshot must be non-promotable.")
        return self._decide(
            strategy_input,
            composition_spec=self.composition_spec,
            component_rule_set=self.component_rule_set,
            strategy_mode=FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1,
            apply_transition_continuity=False,
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
                        AllocatedPoseComponentCompositionRule,
                        AllocatedQposComponentCompositionRule,
                    ),
                ):
                    raise ValueError(
                        "Alpha hot-control is not defined for relative or "
                        "allocation-selected rules."
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
            apply_transition_continuity=True,
        )

    def _decide(
        self,
        strategy_input: FormalCompositionStrategyInput,
        *,
        composition_spec: CompositionSpec,
        component_rule_set: ComponentCompositionRuleSet,
        strategy_mode: str,
        discrete_tie_role: str | None = None,
        apply_transition_continuity: bool,
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
        used_context_ids: set[str] = set()
        allocated_mode_by_component: dict[str, str] = {}
        output_action_schema = None

        for rule in component_rule_set.rules:
            if isinstance(
                rule,
                (
                    AllocatedPoseComponentCompositionRule,
                    AllocatedQposComponentCompositionRule,
                ),
            ):
                allocation_spec = self.allocation_spec_by_hash[
                    rule.allocation_spec_hash
                ]
                allocated = evaluate_allocated_component_rule(
                    strategy_input,
                    rule=rule,
                    allocation_component_ids=allocation_spec.component_ids,
                )
                if allocated.used_context_id is not None:
                    used_context_ids.add(allocated.used_context_id)
                if allocated.output_status != "ready":
                    return CompositionStrategyDecision(
                        output_status=allocated.output_status,
                        reason=(
                            f"allocated_component:{rule.component_id}:"
                            f"{allocated.reason}"
                        ),
                        composed_candidate=None,
                        used_context_ids=tuple(sorted(used_context_ids)),
                    )
                assert allocated.value is not None
                assert allocated.output_action_schema is not None
                assert allocated.selected_mode is not None
                allocated_mode_by_component[rule.component_id] = (
                    allocated.selected_mode
                )
                if output_action_schema is None:
                    output_action_schema = allocated.output_action_schema
                elif (
                    output_action_schema.semantic_content_hash
                    != allocated.output_action_schema.semantic_content_hash
                ):
                    return CompositionStrategyDecision(
                        output_status="invalid",
                        reason="allocated_output_schema_mismatch",
                        composed_candidate=None,
                        used_context_ids=tuple(sorted(used_context_ids)),
                    )
                output_values[rule.component_id] = allocated.value
                continue
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
                if not component_record_available(
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
                if not record_matches_declared_schema(
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
                if delta is None or not component_record_available(
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
                if not record_matches_declared_schema(
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
                            relative_pose_component_value(
                                baseline,
                                delta,
                                component_id=rule.component_id,
                            )
                        )
                    else:
                        output_values[rule.component_id] = (
                            relative_qpos_component_value(
                                baseline,
                                delta,
                                component_id=rule.component_id,
                                baseline_value_contract_hash=(
                                    rule.baseline_value_contract_hash
                                ),
                                delta_value_contract_hash=(
                                    rule.delta_value_contract_hash
                                ),
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
                not record_matches_declared_schema(strategy_input, record)
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
        result_data = (
            {}
            if not allocated_mode_by_component
            else {
                "allocation_selected_mode_by_component": dict(
                    allocated_mode_by_component
                )
            }
        )
        continuity_spec_hashes = tuple(
            dict.fromkeys(
                rule.allocation_spec_hash
                for rule in component_rule_set.rules
                if isinstance(
                    rule,
                    (
                        AllocatedPoseComponentCompositionRule,
                        AllocatedQposComponentCompositionRule,
                    ),
                )
                and rule.allocation_spec_hash
                in self.allocation_transition_continuity_runtime_by_spec_hash
            )
        )
        if apply_transition_continuity and continuity_spec_hashes:
            if len(continuity_spec_hashes) != 1:
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason="multiple_allocation_continuity_specs_unsupported",
                    composed_candidate=None,
                    used_context_ids=tuple(sorted(used_context_ids)),
                )
            allocation_spec = self.allocation_spec_by_hash[
                continuity_spec_hashes[0]
            ]
            context_snapshot = strategy_input.context_snapshot
            context = (
                None
                if context_snapshot is None
                else context_snapshot.selected_context_by_channel.get(
                    next(
                        rule.allocation_context_channel
                        for rule in component_rule_set.rules
                        if isinstance(
                            rule,
                            (
                                AllocatedPoseComponentCompositionRule,
                                AllocatedQposComponentCompositionRule,
                            ),
                        )
                        and rule.allocation_spec_hash == continuity_spec_hashes[0]
                    )
                )
            )
            baseline = selected_by_role.get(allocation_spec.base_absolute_role)
            activation_id = strategy_input.input_snapshot.expected_activation_id
            activation_hash = strategy_input.input_snapshot.expected_activation_hash
            if baseline is None or (
                activation_id is None or activation_hash is None
            ):
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason="allocation_continuity_basis_missing",
                    composed_candidate=None,
                    used_context_ids=tuple(sorted(used_context_ids)),
                )
            try:
                allocation_rules = tuple(
                    rule
                    for rule in component_rule_set.rules
                    if isinstance(
                        rule,
                        (
                            AllocatedPoseComponentCompositionRule,
                            AllocatedQposComponentCompositionRule,
                        ),
                    )
                    and rule.allocation_spec_hash == continuity_spec_hashes[0]
                )
                if context is None or context.availability != "available":
                    if any(
                        rule.context_missing_behavior != "use_baseline"
                        for rule in allocation_rules
                    ):
                        raise ValueError("allocation_continuity_context_missing")
                    modes = {
                        component_id: COMPOSITION_ALLOCATION_MODE_BASE_ONLY
                        for component_id in allocation_spec.component_ids
                    }
                else:
                    modes = selected_allocation_modes_from_context(
                        context,
                        expected_allocation_spec_hash=(
                            allocation_spec.semantic_content_hash
                        ),
                        expected_component_ids=allocation_spec.component_ids,
                        target_effective_at=(
                            strategy_input.input_snapshot.target_effective_at
                        ),
                        expected_composition_activation_id=activation_id,
                        expected_composition_activation_hash=activation_hash,
                    )
                continuity = self.allocation_transition_continuity_runtime_by_spec_hash[
                    continuity_spec_hashes[0]
                ].prepare(
                    raw_candidate=action,
                    baseline_candidate=baseline.action,
                    selected_mode_by_component=modes,
                    target_effective_at=(
                        strategy_input.input_snapshot.target_effective_at
                    ),
                    invocation_id=strategy_input.strategy_input_id,
                    invocation_hash=strategy_input.semantic_content_hash,
                )
            except (KeyError, RuntimeError, TypeError, ValueError) as exc:
                return CompositionStrategyDecision(
                    output_status="invalid",
                    reason=f"allocation_continuity_invalid:{exc}",
                    composed_candidate=None,
                    used_context_ids=tuple(sorted(used_context_ids)),
                )
            action = continuity.output
            result_data = {
                **result_data,
                "allocation_transition_continuity": (
                    continuity.receipt.semantic_payload()
                )
            }
        reason = (
            "formal_component_composition_renormalized:"
            + ",".join(renormalized_components)
            if renormalized_components
            else "formal_component_composition"
        )
        if strategy_mode != FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1:
            reason += f":{strategy_mode}"
        if result_data:
            reason += ":allocation_transition_continuity"
        return CompositionStrategyDecision(
            output_status="ready",
            reason=reason,
            composed_candidate=action,
            used_context_ids=tuple(sorted(used_context_ids)),
            result_data=result_data,
        )


__all__ = [
    "FORMAL_COMPONENT_COMPOSITION_STRATEGY_ID_V2",
    "FORMAL_COMPONENT_STRATEGY_MODE_DISCRETE_OWNER_V1",
    "FORMAL_COMPONENT_STRATEGY_MODE_WEIGHTED_V1",
    "FORMAL_COMPONENT_STRATEGY_MODES_V1",
    "FormalComponentCompositionStrategyV2",
]

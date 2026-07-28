"""Source-neutral baseline-plus-residual composition contract builder."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import ComponentActionSchema

from .component_contract import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    ComponentCompositionRuleSet,
    RelativePoseComponentCompositionRule,
    RelativeQposComponentCompositionRule,
)
from .component_strategy import FormalComponentCompositionStrategyV2
from .core import CompositionSpec
from .formal_strategy import FormalCompositionStrategySpec


RELATIVE_BASELINE_RESIDUAL_CONTRACTS_HASH_DOMAIN_DEVELOPMENT = (
    "dexteleop.relative_baseline_residual_contracts.development"
)


def _required(value: object, *, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


@dataclass(frozen=True)
class RelativeBaselineResidualContracts:
    """One immutable numerical strategy, independent of concrete source kinds."""

    baseline_role: str
    residual_role: str
    output_action_schema: ComponentActionSchema
    residual_action_schema: ComponentActionSchema
    component_rule_set: ComponentCompositionRuleSet
    composition_spec: CompositionSpec
    strategy_spec: FormalCompositionStrategySpec
    strategy: FormalComponentCompositionStrategyV2 = field(
        repr=False,
        compare=False,
    )
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        baseline = _required(self.baseline_role, name="baseline_role")
        residual = _required(self.residual_role, name="residual_role")
        if baseline == residual:
            raise ValueError("baseline_role and residual_role must differ.")
        if not isinstance(self.output_action_schema, ComponentActionSchema):
            raise TypeError(
                "output_action_schema must be ComponentActionSchema."
            )
        if not isinstance(self.residual_action_schema, ComponentActionSchema):
            raise TypeError(
                "residual_action_schema must be ComponentActionSchema."
            )
        if (
            self.composition_spec.component_rule_set_hash
            != self.component_rule_set.semantic_content_hash
        ):
            raise ValueError(
                "Composition spec and relative component rules disagree."
            )
        if (
            self.strategy_spec.composition_spec_hash
            != self.composition_spec.semantic_content_hash
        ):
            raise ValueError(
                "Relative strategy spec binds another composition spec."
            )
        object.__setattr__(self, "baseline_role", baseline)
        object.__setattr__(self, "residual_role", residual)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "baseline_role": baseline,
                    "residual_role": residual,
                    "output_action_schema_hash": (
                        self.output_action_schema.semantic_content_hash
                    ),
                    "residual_action_schema_hash": (
                        self.residual_action_schema.semantic_content_hash
                    ),
                    "component_rule_set_hash": (
                        self.component_rule_set.semantic_content_hash
                    ),
                    "composition_spec_hash": (
                        self.composition_spec.semantic_content_hash
                    ),
                    "strategy_spec_hash": (
                        self.strategy_spec.semantic_content_hash
                    ),
                },
                domain=(
                    RELATIVE_BASELINE_RESIDUAL_CONTRACTS_HASH_DOMAIN_DEVELOPMENT
                ),
            ),
        )


def build_relative_baseline_residual_contracts(
    *,
    contract_id: str,
    output_action_schema: ComponentActionSchema,
    residual_action_schema: ComponentActionSchema,
    baseline_role: str,
    residual_role: str,
    required_roles: Sequence[str] | None = None,
    pose_missing_behavior: str = "use_baseline",
    qpos_missing_behavior: str = "use_baseline",
    maximum_decision_latency_sec: float = 0.01,
) -> RelativeBaselineResidualContracts:
    """Build exact pose/qpos relative rules for arbitrary role names."""

    contract = _required(contract_id, name="contract_id")
    baseline = _required(baseline_role, name="baseline_role")
    residual = _required(residual_role, name="residual_role")
    if baseline == residual:
        raise ValueError("baseline_role and residual_role must differ.")
    if not isinstance(output_action_schema, ComponentActionSchema):
        raise TypeError(
            "output_action_schema must be ComponentActionSchema."
        )
    if not isinstance(residual_action_schema, ComponentActionSchema):
        raise TypeError(
            "residual_action_schema must be ComponentActionSchema."
        )
    if set(residual_action_schema.component_ids) != set(
        output_action_schema.component_ids
    ):
        raise ValueError(
            "Initial relative contract requires residual coverage to equal "
            "output component coverage."
        )
    rules = []
    for component_id in output_action_schema.component_ids:
        absolute = output_action_schema.payload_schema_by_component[
            component_id
        ]
        delta = residual_action_schema.payload_schema_by_component[
            component_id
        ]
        if (
            absolute.value_semantics != "absolute_target"
            or delta.value_semantics != "relative_delta"
            or absolute.representation_family != delta.representation_family
            or absolute.shape != delta.shape
            or absolute.coordinate_names != delta.coordinate_names
            or absolute.units != delta.units
        ):
            raise ValueError(
                f"Relative layout mismatch for component {component_id!r}."
            )
        if absolute.representation_family == "cartesian_pose":
            if (
                absolute.reference_frame != delta.reference_frame
                or absolute.controlled_frame != delta.controlled_frame
            ):
                raise ValueError(
                    f"Relative pose frames mismatch for {component_id!r}."
                )
            rules.append(
                RelativePoseComponentCompositionRule(
                    component_id=component_id,
                    baseline_role=baseline,
                    delta_role=residual,
                    missing_behavior=pose_missing_behavior,
                )
            )
        elif absolute.representation_family == "joint_position":
            rules.append(
                RelativeQposComponentCompositionRule(
                    component_id=component_id,
                    baseline_role=baseline,
                    delta_role=residual,
                    baseline_value_contract_hash=(
                        absolute.semantic_content_hash
                    ),
                    delta_value_contract_hash=delta.semantic_content_hash,
                    missing_behavior=qpos_missing_behavior,
                )
            )
        else:
            raise ValueError(
                "Relative baseline/residual contract supports only pose and "
                "joint-position components."
            )
    rule_set = ComponentCompositionRuleSet(
        rule_set_id=f"{contract}.component-rules",
        action_space="component-keyed",
        rules=tuple(rules),
    )
    required = (
        (baseline, residual)
        if required_roles is None
        else tuple(_required(value, name="required_role") for value in required_roles)
    )
    if len(required) != len(set(required)) or not set(required).issubset(
        {baseline, residual}
    ):
        raise ValueError(
            "required_roles must be a unique subset of baseline/residual roles."
        )
    if residual in required and baseline not in required:
        raise ValueError(
            "A required residual role requires its baseline role."
        )
    spec = CompositionSpec(
        spec_id=f"{contract}.composition",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=(baseline, residual),
        required_roles=required,
        depends_on_roles_by_role={residual: (baseline,)},
        missing_required_behavior="no_action",
        component_rule_set_hash=rule_set.semantic_content_hash,
        action_schema_hash=output_action_schema.semantic_content_hash,
        input_action_schema_hash_by_role={
            baseline: output_action_schema.semantic_content_hash,
            residual: residual_action_schema.semantic_content_hash,
        },
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id=f"{contract}.strategy",
        strategy_version="development",
        composition_spec_hash=spec.semantic_content_hash,
        implementation_contract_hashes=(rule_set.semantic_content_hash,),
        maximum_decision_latency_sec=maximum_decision_latency_sec,
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=spec,
        component_rule_set=rule_set,
    )
    return RelativeBaselineResidualContracts(
        baseline_role=baseline,
        residual_role=residual,
        output_action_schema=output_action_schema,
        residual_action_schema=residual_action_schema,
        component_rule_set=rule_set,
        composition_spec=spec,
        strategy_spec=strategy_spec,
        strategy=strategy,
    )


__all__ = [
    "RELATIVE_BASELINE_RESIDUAL_CONTRACTS_HASH_DOMAIN_DEVELOPMENT",
    "RelativeBaselineResidualContracts",
    "build_relative_baseline_residual_contracts",
]

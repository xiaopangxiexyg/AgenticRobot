"""Default-off MuJoCo Sim-HIL application composition.

This module is deliberately an application leaf.  It selects already-owned
policy and passive-human emissions, binds the formal component composition
contract, and drives one existing :class:`MujocoDeployRuntime`.  It does not
own policy inference, a policy queue, teleop devices, IK, safety, authority,
or dispatch.

The first profile is intentionally narrow: bimanual RM65 tool-TCP arm poses,
the two typed Q6 hand components, human plus policy contributors, and an
all-required compound proposal.  The profile is disabled unless a caller
explicitly opts in.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from control_composition.activation import (
    CompositionActivationScope,
    build_composition_activation_scope,
)
from control_composition.observability import SourceDifferenceSpec
from control_composition.eligibility import (
    CompositionSourceEligibilityDraft,
    CompositionSourceEligibilitySpec,
)
from control_composition.composition import (
    FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
    CompositionAuthorityDisposition,
    CompositionAuthorityGateSpec,
    CompositionDynamicParameterContract,
    CompositionDynamicParameterDraft,
    CompositionInputAdmissionReceipt,
    CompositionSpec,
    ComponentCompositionRuleSet,
    FormalComponentCompositionStrategyV2,
    FormalCompositionStrategySpec,
    PoseComponentCompositionRule,
    QposComponentCompositionRule,
    build_composition_input_draft,
    decide_composition_authority_disposition,
    hand_action_spec_semantic_hash,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.control_trace import ControlTraceFact
from dataset_interface.composition_control_trace import (
    project_composition_activation_scope,
    project_composition_authority_disposition,
)
from robot_motion.action import (
    BIMANUAL_EE_POSE_V1,
    ComponentAction,
    ComponentActionSchema,
    ComponentTargetCapabilitySpec,
    ConservativeCompoundAuthorityProfile,
    ExecutionGateTransitionOutcome,
    HandActionSpec,
    IMMEDIATE_QREF_SIM_V1,
)
from robot_motion.action.authoritative_control import AuthoritativeControlStep
from robot_motion.deploy import (
    PolicyActionEmission,
    project_policy_action_emission,
)
from robot_motion.runtime.control_admission import ControlCut
from sim.mujoco.deploy import MujocoDeployRuntime
from teleop_runtime.action.human_proposal import (
    HumanCommandEmission,
    HumanProposalAdapter,
)
from teleop_runtime.control_trace.projection import (
    project_human_proposal_adapter_result,
)

from ..sources.policy import (
    PolicyCompositionInputAdapter,
    PolicyCompositionInputBinding,
)
from ..sources.teleop_pose import PassiveTeleopComponentSourceSpec
from .mujoco_human import build_mujoco_component_human_proposal_adapter


MUJOCO_SIM_HIL_PROFILE_ID_V1 = "control_station_mujoco_sim_hil_compound_v1"
MUJOCO_SIM_HIL_POLICY_ROLE_V1 = "policy"
MUJOCO_SIM_HIL_HUMAN_ROLE_V1 = "human"
MUJOCO_SIM_HIL_COMPOSER_NODE_ID_V1 = "control-station-sim-hil-composer"
MUJOCO_SIM_HIL_CONTRACT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.mujoco_sim_hil_contract.v1"
)
MUJOCO_SIM_HIL_PREFLIGHT_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.mujoco_sim_hil_preflight.v1"
)
MUJOCO_SIM_HIL_READINESS_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.mujoco_sim_hil_readiness.v1"
)
MUJOCO_SIM_HIL_INPUT_SUBMISSION_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.mujoco_sim_hil_input_submission.v1"
)
MUJOCO_SIM_HIL_COMPOSITION_STEP_HASH_DOMAIN_V1 = (
    "dexteleop.control_station.mujoco_sim_hil_composition_step.v1"
)
MUJOCO_SIM_HIL_CYCLE_RETURN_HASH_DOMAIN = (
    "dexteleop.control_station.mujoco_sim_hil_cycle_return.development"
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


def _normalize_hand_specs(
    values: Mapping[str, HandActionSpec | Mapping[str, object]],
) -> dict[str, HandActionSpec]:
    result: dict[str, HandActionSpec] = {}
    for raw_side, raw_spec in values.items():
        side = _required(raw_side, name="hand side")
        spec = (
            raw_spec
            if isinstance(raw_spec, HandActionSpec)
            else HandActionSpec.from_metadata(raw_spec)
        )
        if spec.side != side:
            raise ValueError("Hand action spec key and side disagree.")
        result[side] = spec
    if set(result) != {"left", "right"}:
        raise ValueError(
            "The first MuJoCo Sim-HIL profile requires exactly left and right "
            "hand action specs."
        )
    return {side: result[side] for side in ("left", "right")}


@dataclass(frozen=True)
class MujocoSimHILContracts:
    """One immutable formal component and promotion contract."""

    composition_spec: CompositionSpec
    component_rule_set: ComponentCompositionRuleSet
    strategy_spec: FormalCompositionStrategySpec
    strategy: FormalComponentCompositionStrategyV2
    authority_profile: ConservativeCompoundAuthorityProfile
    authority_gate_spec: CompositionAuthorityGateSpec
    dynamic_parameter_contract: CompositionDynamicParameterContract | None = None
    initial_alpha_by_component: Mapping[str, float] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.composition_spec, CompositionSpec):
            raise TypeError("composition_spec must be a CompositionSpec.")
        if not isinstance(self.component_rule_set, ComponentCompositionRuleSet):
            raise TypeError(
                "component_rule_set must be a ComponentCompositionRuleSet."
            )
        if not isinstance(self.strategy_spec, FormalCompositionStrategySpec):
            raise TypeError(
                "strategy_spec must be a FormalCompositionStrategySpec."
            )
        if not isinstance(
            self.strategy,
            FormalComponentCompositionStrategyV2,
        ):
            raise TypeError(
                "strategy must be FormalComponentCompositionStrategyV2."
            )
        if not isinstance(
            self.authority_profile,
            ConservativeCompoundAuthorityProfile,
        ):
            raise TypeError(
                "authority_profile must be ConservativeCompoundAuthorityProfile."
            )
        if not isinstance(
            self.authority_gate_spec,
            CompositionAuthorityGateSpec,
        ):
            raise TypeError(
                "authority_gate_spec must be CompositionAuthorityGateSpec."
            )
        if self.composition_spec.algebra != FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2:
            raise ValueError("MuJoCo Sim-HIL requires formal component algebra v2.")
        if (
            self.composition_spec.component_rule_set_hash
            != self.component_rule_set.semantic_content_hash
        ):
            raise ValueError("CompositionSpec is bound to another rule set.")
        if (
            self.strategy_spec.composition_spec_hash
            != self.composition_spec.semantic_content_hash
            or self.strategy_spec.semantic_content_hash
            != self.authority_profile.formal_strategy_spec_hash
        ):
            raise ValueError("Formal strategy and authority profile do not bind.")
        if self.strategy.spec != self.strategy_spec:
            raise ValueError("Strategy implementation uses another strategy spec.")
        if (
            self.strategy.component_rule_set.semantic_content_hash
            != self.component_rule_set.semantic_content_hash
        ):
            raise ValueError("Strategy implementation uses another rule set.")
        if (
            self.authority_profile.composition_spec_hash
            != self.composition_spec.semantic_content_hash
            or self.authority_profile.component_rule_set_hash
            != self.component_rule_set.semantic_content_hash
        ):
            raise ValueError("Compound authority profile is bound to another contract.")
        if self.component_rule_set.component_ids != (
            self.authority_profile.component_coverage
        ):
            raise ValueError(
                "Formal component rules must cover the complete compound scope."
            )
        self.authority_gate_spec.validate_composition_spec(self.composition_spec)
        if self.authority_gate_spec.composer_producer_id != (
            self.authority_profile.producer_id
        ):
            raise ValueError("Authority gate and compound profile producer disagree.")
        dynamic_contract = self.dynamic_parameter_contract
        initial_alpha = {
            _required(component_id, name="initial alpha component"): _finite(
                value,
                name=f"initial_alpha_by_component[{component_id!r}]",
            )
            for component_id, value in dict(
                self.initial_alpha_by_component
            ).items()
        }
        if dynamic_contract is None:
            if initial_alpha:
                raise ValueError(
                    "Initial alpha values require a dynamic parameter contract."
                )
        else:
            if not isinstance(
                dynamic_contract,
                CompositionDynamicParameterContract,
            ):
                raise TypeError(
                    "dynamic_parameter_contract must be a "
                    "CompositionDynamicParameterContract or None."
                )
            if set(initial_alpha) != set(dynamic_contract.component_ids):
                raise ValueError(
                    "Initial alpha values must exactly cover the dynamic "
                    "parameter components."
                )
            if (
                dynamic_contract.base_role
                not in self.composition_spec.input_roles
                or dynamic_contract.overlay_role
                not in self.composition_spec.input_roles
            ):
                raise ValueError(
                    "Dynamic parameter roles must belong to the composition."
                )
        object.__setattr__(
            self,
            "initial_alpha_by_component",
            MappingProxyType(initial_alpha),
        )
        payload = {
            "composition_spec_hash": self.composition_spec.semantic_content_hash,
            "component_rule_set_hash": self.component_rule_set.semantic_content_hash,
            "strategy_spec_hash": self.strategy_spec.semantic_content_hash,
            "authority_profile_hash": self.authority_profile.semantic_content_hash,
            "authority_gate_spec_hash": self.authority_gate_spec.semantic_content_hash,
            "dynamic_parameter_contract_hash": (
                None
                if dynamic_contract is None
                else dynamic_contract.semantic_content_hash
            ),
            "initial_alpha_by_component": initial_alpha,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=MUJOCO_SIM_HIL_CONTRACT_HASH_DOMAIN_V1,
            ),
        )


def build_component_keyed_mujoco_sim_hil_contracts(
    action_schema: ComponentActionSchema,
    *,
    composer_producer_id: str,
    proposal_ttl_sec: float,
    profile_id: str = "control-station-mujoco-component-sim-hil-v1",
    human_role: str = MUJOCO_SIM_HIL_HUMAN_ROLE_V1,
    policy_role: str = MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    policy_active_components: Sequence[str] | None = None,
    human_weight: float = 0.5,
    policy_weight: float = 0.5,
    max_pose_translation_disagreement_m: float | None = None,
    max_pose_rotation_disagreement_rad: float | None = None,
) -> MujocoSimHILContracts:
    """Build a Sim-HIL contract from a native component action schema.

    This is an application profile selection helper, not a second action
    carrier.  Component cardinality, names, topology, representation, frames,
    units, and qpos layout come only from ``action_schema``.  ``None`` pose
    disagreement limits keep the difference observable without rejecting the
    pre-composition input; downstream target safety remains authoritative.
    """

    if not isinstance(action_schema, ComponentActionSchema):
        raise TypeError("action_schema must be a ComponentActionSchema.")
    profile_name = _required(profile_id, name="profile_id")
    human = _required(human_role, name="human_role")
    policy = _required(policy_role, name="policy_role")
    if human == policy:
        raise ValueError("human_role and policy_role must differ.")
    weights = {
        human: float(human_weight),
        policy: float(policy_weight),
    }
    if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
        raise ValueError("Sim-HIL role weights must be finite and non-negative.")
    if not any(value > 0.0 for value in weights.values()):
        raise ValueError("At least one Sim-HIL role weight must be positive.")
    component_ids = action_schema.component_ids
    if policy_active_components is None:
        policy_components = component_ids
    else:
        policy_components = tuple(
            _required(value, name="policy component id")
            for value in policy_active_components
        )
        if not policy_components or len(policy_components) != len(
            set(policy_components)
        ):
            raise ValueError(
                "policy_active_components must be non-empty and unique."
            )
        unknown = sorted(set(policy_components) - set(component_ids))
        if unknown:
            raise ValueError(
                f"policy_active_components contains unknown components: {unknown}."
            )
    policy_component_set = set(policy_components)
    dynamic_alpha_enabled = policy_component_set == set(component_ids)
    human_alpha = weights[human] / (weights[human] + weights[policy])

    rules: list[PoseComponentCompositionRule | QposComponentCompositionRule] = []
    for component_id in component_ids:
        payload_schema = action_schema.payload_schema_by_component[component_id]
        has_policy = component_id in policy_component_set
        roles = (human, policy) if has_policy else (human,)
        component_weights = (
            weights if has_policy else {human: 1.0}
        )
        # Positive-weight contributors are required for this cycle.  A role
        # whose hot-control weight is exactly zero is ignored by the formal
        # strategy, while temporary loss of an active role yields no_action
        # rather than silently changing the requested blend ratio.
        missing_behavior = "no_action"
        if payload_schema.representation_family == "cartesian_pose":
            rules.append(
                PoseComponentCompositionRule(
                    component_id=component_id,
                    contributing_roles=roles,
                    weight_by_role=component_weights,
                    missing_behavior=missing_behavior,
                    max_translation_disagreement_m=(
                        max_pose_translation_disagreement_m
                    ),
                    max_rotation_disagreement_rad=(
                        max_pose_rotation_disagreement_rad
                    ),
                )
            )
        elif payload_schema.representation_family == "joint_position":
            rules.append(
                QposComponentCompositionRule(
                    component_id=component_id,
                    contributing_roles=roles,
                    weight_by_role=component_weights,
                    value_contract_hash=payload_schema.semantic_content_hash,
                    missing_behavior=missing_behavior,
                )
            )
        else:
            raise ValueError(
                "The current minimal Sim-HIL strategy supports only "
                "cartesian_pose and joint_position components; "
                f"{component_id!r} uses "
                f"{payload_schema.representation_family!r}."
            )

    rule_set = ComponentCompositionRuleSet(
        rule_set_id=f"{profile_name}:rules",
        action_space=f"component_action:{action_schema.semantic_content_hash}",
        rules=tuple(rules),
    )
    composition_spec = CompositionSpec(
        spec_id=f"{profile_name}:composition",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=(human, policy),
        required_roles=(),
        missing_required_behavior="no_action",
        component_rule_set_hash=rule_set.semantic_content_hash,
        action_schema_hash=action_schema.semantic_content_hash,
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id=f"{profile_name}:strategy",
        strategy_version="development",
        composition_spec_hash=composition_spec.semantic_content_hash,
        implementation_contract_hashes=(rule_set.semantic_content_hash,),
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=composition_spec,
        component_rule_set=rule_set,
    )
    catalog = action_schema.component_catalog
    authority_profile = ConservativeCompoundAuthorityProfile(
        profile_id=f"{profile_name}:compound-authority",
        producer_id=_required(
            composer_producer_id,
            name="composer_producer_id",
        ),
        component_coverage=component_ids,
        selection_groups=catalog.selection_group_ids,
        composition_spec_hash=composition_spec.semantic_content_hash,
        component_rule_set_hash=rule_set.semantic_content_hash,
        formal_strategy_spec_hash=strategy_spec.semantic_content_hash,
    )
    authority_gate_spec = CompositionAuthorityGateSpec(
        gate_spec_id=f"{profile_name}:authority-gate",
        composition_spec_hash=composition_spec.semantic_content_hash,
        composer_producer_id=authority_profile.producer_id,
        missing_behavior_by_role={
            human: "no_action",
            policy: "no_action",
        },
        proposal_ttl_sec=_finite(proposal_ttl_sec, name="proposal_ttl_sec"),
        invalid_result_behavior="hold",
        composer_error_behavior="hold",
        all_inputs_missing_behavior="no_action",
        allow_parallel_source_proposals=False,
        canonical_safety_required=True,
    )
    dynamic_contract = (
        CompositionDynamicParameterContract(
            parameter_set_id=f"{profile_name}:human-alpha",
            base_role=policy,
            overlay_role=human,
            component_ids=component_ids,
            default_alpha_by_component={
                component_id: human_alpha for component_id in component_ids
            },
            lifecycle="latched",
        )
        if dynamic_alpha_enabled
        else None
    )
    return MujocoSimHILContracts(
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        strategy_spec=strategy_spec,
        strategy=strategy,
        authority_profile=authority_profile,
        authority_gate_spec=authority_gate_spec,
        dynamic_parameter_contract=dynamic_contract,
        initial_alpha_by_component=(
            {}
            if dynamic_contract is None
            else dynamic_contract.default_alpha_by_component
        ),
    )


def build_default_mujoco_sim_hil_contracts(
    hand_action_specs_by_side: Mapping[
        str,
        HandActionSpec | Mapping[str, object],
    ],
    *,
    composer_producer_id: str,
    proposal_ttl_sec: float,
    human_role: str = MUJOCO_SIM_HIL_HUMAN_ROLE_V1,
    policy_role: str = MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    policy_active_effectors: Sequence[str] | None = None,
    action_schema_hash: str | None = None,
) -> MujocoSimHILContracts:
    """Build the first narrow human+policy component profile.

    ``policy_active_effectors`` is an application-selected contributor scope,
    not an execution scope.  The compound authority still covers the complete
    bimanual arm+hand catalog.  ``None`` preserves the original full-policy
    contract; an explicit subset is useful for policies such as the temporary
    CubeLift PPO policy, which only emits a right-arm pose action while the
    passive human source owns both Q6 hand components.
    """

    human = _required(human_role, name="human_role")
    policy = _required(policy_role, name="policy_role")
    if human == policy:
        raise ValueError("human_role and policy_role must differ.")
    hand_specs = _normalize_hand_specs(hand_action_specs_by_side)
    catalog_effectors = (
        "left_arm",
        "right_arm",
        "left_hand",
        "right_hand",
    )
    if policy_active_effectors is None:
        selected_policy_effectors = catalog_effectors
    else:
        selected_policy_effectors = tuple(
            str(value).strip().lower() for value in policy_active_effectors
        )
        if not selected_policy_effectors:
            raise ValueError("policy_active_effectors must not be empty.")
        if len(set(selected_policy_effectors)) != len(selected_policy_effectors):
            raise ValueError("policy_active_effectors must not contain duplicates.")
        unknown = sorted(set(selected_policy_effectors) - set(catalog_effectors))
        if unknown:
            raise ValueError(
                "policy_active_effectors contains unknown components: "
                + ",".join(unknown)
            )

    full_policy_scope = selected_policy_effectors == catalog_effectors

    def _roles(component_id: str) -> tuple[str, ...]:
        return (
            (human, policy)
            if component_id in selected_policy_effectors
            else (human,)
        )

    def _weights(component_id: str) -> dict[str, float]:
        return (
            {human: 0.5, policy: 0.5}
            if component_id in selected_policy_effectors
            else {human: 1.0}
        )

    def _missing_behavior(component_id: str) -> str:
        # The historical full-scope contract deliberately used no_action.
        # A partial contributor scope must renormalize when its declared
        # component is temporarily absent, otherwise a human-only component
        # would be incorrectly held at zero by the weighted algebra.
        return "no_action" if full_policy_scope else (
            "renormalize_available"
            if component_id in selected_policy_effectors
            else "no_action"
        )

    rules = (
        *(
            PoseComponentCompositionRule(
                component_id=f"{side}_arm",
                contributing_roles=_roles(f"{side}_arm"),
                weight_by_role=_weights(f"{side}_arm"),
                missing_behavior=_missing_behavior(f"{side}_arm"),
                max_translation_disagreement_m=0.10,
                max_rotation_disagreement_rad=0.50,
            )
            for side in ("left", "right")
        ),
        *(
            QposComponentCompositionRule(
                component_id=f"{side}_hand",
                contributing_roles=_roles(f"{side}_hand"),
                weight_by_role=_weights(f"{side}_hand"),
                value_contract_hash=hand_action_spec_semantic_hash(
                    hand_specs[side]
                ),
                missing_behavior=_missing_behavior(f"{side}_hand"),
            )
            for side in ("left", "right")
        ),
    )
    rule_set = ComponentCompositionRuleSet(
        rule_set_id=(
            "mujoco-sim-hil-bimanual-dmtac-rules-v1"
            if full_policy_scope
            else "mujoco-sim-hil-bimanual-dmtac-rules-v1-policy-subset"
        ),
        action_space=BIMANUAL_EE_POSE_V1,
        rules=rules,
    )
    composition_spec = CompositionSpec(
        spec_id="mujoco-sim-hil-bimanual-dmtac-composition-v2",
        algebra=FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2,
        input_roles=(human, policy),
        required_roles=(human, policy),
        missing_required_behavior="no_action",
        component_rule_set_hash=rule_set.semantic_content_hash,
        action_schema_hash=(
            None
            if action_schema_hash is None
            else _required(action_schema_hash, name="action_schema_hash")
        ),
    )
    strategy_spec = FormalCompositionStrategySpec(
        strategy_id="mujoco-sim-hil-formal-component-strategy-v2",
        strategy_version="2.0",
        composition_spec_hash=composition_spec.semantic_content_hash,
        implementation_contract_hashes=(rule_set.semantic_content_hash,),
    )
    strategy = FormalComponentCompositionStrategyV2(
        strategy_spec=strategy_spec,
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        qpos_value_spec_by_component={
            f"{side}_hand": hand_specs[side]
            for side in ("left", "right")
        },
    )
    coverage = ("left_arm", "right_arm", "left_hand", "right_hand")
    groups = ("arms", "left_hand", "right_hand")
    profile = ConservativeCompoundAuthorityProfile(
        profile_id="mujoco-sim-hil-conservative-compound-authority-v1",
        producer_id=_required(
            composer_producer_id,
            name="composer_producer_id",
        ),
        component_coverage=coverage,
        selection_groups=groups,
        composition_spec_hash=composition_spec.semantic_content_hash,
        component_rule_set_hash=rule_set.semantic_content_hash,
        formal_strategy_spec_hash=strategy_spec.semantic_content_hash,
    )
    gate = CompositionAuthorityGateSpec(
        gate_spec_id="mujoco-sim-hil-formal-authority-gate-v1",
        composition_spec_hash=composition_spec.semantic_content_hash,
        composer_producer_id=profile.producer_id,
        missing_behavior_by_role={human: "hold", policy: "hold"},
        proposal_ttl_sec=_finite(proposal_ttl_sec, name="proposal_ttl_sec"),
        invalid_result_behavior="hold",
        composer_error_behavior="hold",
        all_inputs_missing_behavior="hold",
        allow_parallel_source_proposals=False,
        canonical_safety_required=True,
    )
    return MujocoSimHILContracts(
        composition_spec=composition_spec,
        component_rule_set=rule_set,
        strategy_spec=strategy_spec,
        strategy=strategy,
        authority_profile=profile,
        authority_gate_spec=gate,
    )


_READINESS_FIELDS = (
    "policy_source_ready",
    "human_device_ready",
    "calibration_ready",
    "home_ready",
    "rebase_ready",
    "deadman_ready",
    "policy_chunk_ttl_ready",
    "measured_state_ready",
)

# These fields are projected into the same-cut composition input/eligibility
# stores.  They are not execution-gate readiness and must not be interpreted
# by Station as permission to mutate the lower gate.
_COMPOSITION_INPUT_AVAILABILITY_FIELDS = frozenset(
    {
        "policy_source_ready",
        "human_device_ready",
        "policy_chunk_ttl_ready",
    }
)


@dataclass(frozen=True)
class MujocoSimHILReadinessEvidence:
    """Read-only readiness evidence supplied by lower-owned source profiles."""

    policy_source_ready: bool
    human_device_ready: bool
    calibration_ready: bool
    home_ready: bool
    rebase_ready: bool
    deadman_ready: bool
    policy_chunk_ttl_ready: bool
    measured_state_ready: bool
    detail_by_check: Mapping[str, str] = field(default_factory=dict)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        for name in _READINESS_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool.")
        details = {
            str(key).strip(): str(value).strip()
            for key, value in dict(self.detail_by_check).items()
        }
        if any(not key for key in details):
            raise ValueError("Readiness detail keys must be non-empty.")
        object.__setattr__(self, "detail_by_check", MappingProxyType(details))
        payload = {
            name: bool(getattr(self, name)) for name in _READINESS_FIELDS
        }
        payload["detail_by_check"] = self.detail_by_check
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=MUJOCO_SIM_HIL_READINESS_HASH_DOMAIN_V1,
            ),
        )

    @classmethod
    def all_ready(cls) -> "MujocoSimHILReadinessEvidence":
        return cls(**{name: True for name in _READINESS_FIELDS})

    @property
    def ready(self) -> bool:
        return all(bool(getattr(self, name)) for name in _READINESS_FIELDS)

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(
            name for name in _READINESS_FIELDS if not bool(getattr(self, name))
        )


class MujocoSimHILReadinessProvider(Protocol):
    """Lower-owned readiness provider; it must not mutate execution state."""

    def snapshot(
        self,
        *,
        now: float,
    ) -> MujocoSimHILReadinessEvidence:
        ...


@dataclass(frozen=True)
class MujocoSimHILPreflightReport:
    readiness: MujocoSimHILReadinessEvidence
    checks: Mapping[str, bool]
    reasons: tuple[str, ...] = ()
    passed: bool = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.readiness, MujocoSimHILReadinessEvidence):
            raise TypeError("readiness must be MujocoSimHILReadinessEvidence.")
        checks = {
            str(key).strip(): bool(value)
            for key, value in dict(self.checks).items()
        }
        if not checks or any(not key for key in checks):
            raise ValueError("Preflight checks must be non-empty and named.")
        reasons = tuple(str(value).strip() for value in self.reasons)
        if any(not value for value in reasons):
            raise ValueError("Preflight reasons must be non-empty.")
        # ``checks["required_readiness"]`` is computed from the active source
        # mix.  A human-only stage intentionally does not require policy
        # freshness, so the full evidence envelope may still contain false
        # policy fields without making this preflight fail.  The selected
        # readiness contract is represented by the checks map and must be the
        # sole admission criterion here.
        passed = bool(all(checks.values()))
        object.__setattr__(self, "checks", MappingProxyType(checks))
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "readiness_hash": self.readiness.semantic_content_hash,
                    "checks": self.checks,
                    "reasons": self.reasons,
                    "passed": self.passed,
                },
                domain=MUJOCO_SIM_HIL_PREFLIGHT_HASH_DOMAIN_V1,
            ),
        )


@dataclass(frozen=True)
class MujocoSimHILProfileConfig:
    """Application profile knobs; all execution semantics stay lower-owned."""

    profile_id: str = MUJOCO_SIM_HIL_PROFILE_ID_V1
    enabled: bool = False
    human_role: str = MUJOCO_SIM_HIL_HUMAN_ROLE_V1
    policy_role: str = MUJOCO_SIM_HIL_POLICY_ROLE_V1
    human_producer_id: str = "human-passive"
    human_producer_epoch: int = 1
    blend_control_producer_id: str = "control-station-sim-hil-blend-control"
    blend_control_producer_epoch: int = 1
    initial_human_alpha: float = 0.5
    max_human_emission_age_sec: float = 0.20
    max_policy_emission_age_sec: float = 0.20
    physics_ticks_after_dispatch: int = 0

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "human_role",
            "policy_role",
            "human_producer_id",
            "blend_control_producer_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), name=name),
            )
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")
        if self.human_role == self.policy_role:
            raise ValueError("human_role and policy_role must differ.")
        epoch = int(self.human_producer_epoch)
        if (
            isinstance(self.human_producer_epoch, bool)
            or epoch != self.human_producer_epoch
            or epoch < 0
        ):
            raise ValueError("human_producer_epoch must be a non-negative integer.")
        object.__setattr__(self, "human_producer_epoch", epoch)
        blend_epoch = int(self.blend_control_producer_epoch)
        if (
            isinstance(self.blend_control_producer_epoch, bool)
            or blend_epoch != self.blend_control_producer_epoch
            or blend_epoch < 0
        ):
            raise ValueError(
                "blend_control_producer_epoch must be a non-negative integer."
            )
        object.__setattr__(
            self,
            "blend_control_producer_epoch",
            blend_epoch,
        )
        alpha = _finite(
            self.initial_human_alpha,
            name="initial_human_alpha",
        )
        if alpha < 0.0 or alpha > 1.0:
            raise ValueError("initial_human_alpha must be in [0, 1].")
        object.__setattr__(self, "initial_human_alpha", alpha)
        for name in (
            "max_human_emission_age_sec",
            "max_policy_emission_age_sec",
        ):
            age = float(getattr(self, name))
            if not math.isfinite(age) or age <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, age)
        ticks = int(self.physics_ticks_after_dispatch)
        if (
            isinstance(self.physics_ticks_after_dispatch, bool)
            or ticks != self.physics_ticks_after_dispatch
            or ticks < 0
        ):
            raise ValueError("physics_ticks_after_dispatch must be non-negative.")
        object.__setattr__(self, "physics_ticks_after_dispatch", ticks)


@dataclass(frozen=True)
class MujocoSimHILInputSubmission:
    role: str
    source_emission_id: str
    input_receipt: CompositionInputAdmissionReceipt | None = None
    error: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _required(self.role, name="role"))
        object.__setattr__(
            self,
            "source_emission_id",
            _required(self.source_emission_id, name="source_emission_id"),
        )
        if self.input_receipt is not None and not isinstance(
            self.input_receipt,
            CompositionInputAdmissionReceipt,
        ):
            raise TypeError("input_receipt must be a composition admission receipt.")
        error = None if self.error is None else _required(self.error, name="error")
        if self.input_receipt is None and error is None:
            raise ValueError("A failed input submission must explain its error.")
        object.__setattr__(self, "error", error)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "role": self.role,
                    "source_emission_id": self.source_emission_id,
                    "input_receipt_hash": (
                        None
                        if self.input_receipt is None
                        else self.input_receipt.semantic_content_hash
                    ),
                    "error": self.error,
                },
                domain=MUJOCO_SIM_HIL_INPUT_SUBMISSION_HASH_DOMAIN_V1,
            ),
        )

    @property
    def accepted(self) -> bool:
        return bool(self.input_receipt is not None and self.input_receipt.accepted)


class MujocoSimHILHumanInputTransform(Protocol):
    """Optional source transform; it owns no authority, safety, or dispatch."""

    @property
    def input_action_schema(self) -> ComponentActionSchema: ...

    @property
    def output_action_schema(self) -> ComponentActionSchema: ...

    def activate(
        self,
        activation_scope: CompositionActivationScope,
        *,
        activated_at: float,
    ) -> Any: ...

    def transform(
        self,
        action: ComponentAction,
        *,
        source_action_id: str,
        source_action_hash: str,
        source_available_at: float,
        measured_at: float,
        measured_clock_domain_id: str,
        observed_at: float,
        observed_clock_domain_id: str,
    ) -> Any: ...


@dataclass(frozen=True)
class MujocoSimHILCompositionStep:
    decision_key: str
    target_effective_at: float
    control_cut: ControlCut | None
    readiness: MujocoSimHILReadinessEvidence | None
    evaluation: Any | None
    disposition: CompositionAuthorityDisposition | None
    proposal_submission: Any | None
    hold_transition: ExecutionGateTransitionOutcome | None
    blocked_reason: str | None = None
    error: str | None = None
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_key",
            _required(self.decision_key, name="decision_key"),
        )
        object.__setattr__(
            self,
            "target_effective_at",
            _finite(self.target_effective_at, name="target_effective_at"),
        )
        if self.control_cut is not None and not isinstance(self.control_cut, ControlCut):
            raise TypeError("control_cut must be a ControlCut or None.")
        if self.readiness is not None and not isinstance(
            self.readiness,
            MujocoSimHILReadinessEvidence,
        ):
            raise TypeError("readiness must be readiness evidence or None.")
        if self.disposition is not None and not isinstance(
            self.disposition,
            CompositionAuthorityDisposition,
        ):
            raise TypeError("disposition must be a CompositionAuthorityDisposition.")
        for name in ("blocked_reason", "error"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _required(value, name=name))
        result = None if self.evaluation is None else self.evaluation.composition_evaluation
        proposal = (
            None
            if self.proposal_submission is None
            else self.proposal_submission.proposal_admission
        )
        hold = None if self.hold_transition is None else self.hold_transition.event.event_id
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "decision_key": self.decision_key,
                    "target_effective_at": self.target_effective_at,
                    "control_cutoff_at": (
                        None if self.control_cut is None else self.control_cut.cutoff_at
                    ),
                    "control_cut_hash": (
                        None
                        if self.control_cut is None
                        else canonical_content_hash(
                            {
                                "runtime_session_id": self.control_cut.runtime_session_id,
                                "control_domain_id": self.control_cut.control_domain_id,
                                "clock_domain_id": self.control_cut.clock_domain_id,
                                "clock_reset_epoch": self.control_cut.clock_reset_epoch,
                                "cutoff_at": self.control_cut.cutoff_at,
                                "cutoff_admission_sequence": (
                                    self.control_cut.cutoff_admission_sequence
                                ),
                            },
                            domain="dexteleop.control_cut.v1",
                        )
                    ),
                    "readiness_hash": (
                        None
                        if self.readiness is None
                        else self.readiness.semantic_content_hash
                    ),
                    "composition_evaluation_id": (
                        None if result is None else result.composition_evaluation_id
                    ),
                    "composition_evaluation_hash": (
                        None if result is None else result.semantic_content_hash
                    ),
                    "disposition_id": (
                        None
                        if self.disposition is None
                        else self.disposition.disposition_id
                    ),
                    "proposal_id": (
                        None
                        if proposal is None
                        else proposal.proposal_id
                    ),
                    "hold_event_id": hold,
                    "blocked_reason": self.blocked_reason,
                    "error": self.error,
                },
                domain=MUJOCO_SIM_HIL_COMPOSITION_STEP_HASH_DOMAIN_V1,
            ),
        )

    @property
    def proposal_admission(self):
        if self.proposal_submission is None:
            return None
        return self.proposal_submission.proposal_admission


@dataclass(frozen=True)
class MujocoSimHILCycleResult:
    """Envelope of facts carried by ``runtime.step()``; no later receiver poll."""

    cycle_index: int
    human_step: Any | None
    human_submission: MujocoSimHILInputSubmission | None
    policy_submission: MujocoSimHILInputSubmission | None
    composition_step: MujocoSimHILCompositionStep
    execution: AuthoritativeControlStep
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        cycle = int(self.cycle_index)
        if isinstance(self.cycle_index, bool) or cycle < 0 or cycle != self.cycle_index:
            raise ValueError("cycle_index must be a non-negative integer.")
        object.__setattr__(self, "cycle_index", cycle)
        if not isinstance(self.composition_step, MujocoSimHILCompositionStep):
            raise TypeError("composition_step must be a MujocoSimHILCompositionStep.")
        if not isinstance(self.execution, AuthoritativeControlStep):
            raise TypeError("execution must be an AuthoritativeControlStep.")
        for name in ("human_submission", "policy_submission"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, MujocoSimHILInputSubmission):
                raise TypeError(f"{name} must be a MujocoSimHILInputSubmission.")
        target_dispatch = self.execution.target_dispatch_result
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                {
                    "evidence_cut": "authoritative_control_step_return",
                    "cycle_index": cycle,
                    "human_submission_hash": (
                        None
                        if self.human_submission is None
                        else self.human_submission.semantic_content_hash
                    ),
                    "policy_submission_hash": (
                        None
                        if self.policy_submission is None
                        else self.policy_submission.semantic_content_hash
                    ),
                    "composition_step_hash": self.composition_step.semantic_content_hash,
                    "execution_status": self.execution.status,
                    "execution_command_id": self.execution.command_id,
                    "execution_dispatch_receipt_id": self.execution.dispatch_receipt_id,
                    "target_execution_request_id": (
                        None
                        if self.execution.target_execution_request is None
                        else self.execution.target_execution_request.request_id
                    ),
                    "target_submission_receipt_id": (
                        None
                        if self.execution.target_submission_receipt is None
                        else self.execution.target_submission_receipt.receipt_id
                    ),
                    "target_completion_receipt_id": (
                        None
                        if self.execution.target_completion_receipt is None
                        else self.execution.target_completion_receipt.receipt_id
                    ),
                    "component_execution_target_id": (
                        None
                        if target_dispatch is None
                        else target_dispatch.target.target_id
                    ),
                    "component_target_admission_receipt_id": (
                        None
                        if target_dispatch is None
                        else target_dispatch.target_admission.receipt_id
                    ),
                    "component_target_activation_id_at_return": (
                        None
                        if target_dispatch is None
                        or target_dispatch.target_activation is None
                        else target_dispatch.target_activation.activation_id
                    ),
                },
                domain=MUJOCO_SIM_HIL_CYCLE_RETURN_HASH_DOMAIN,
            ),
        )


class MujocoSimHILComposition:
    """One Station-owned assembly over one existing MuJoCo runtime."""

    def __init__(
        self,
        *,
        runtime: MujocoDeployRuntime,
        contracts: MujocoSimHILContracts,
        policy_input_adapter: PolicyCompositionInputAdapter,
        human_source: Any,
        human_source_spec: PassiveTeleopComponentSourceSpec,
        human_adapter: HumanProposalAdapter,
        human_input_transform: MujocoSimHILHumanInputTransform | None = None,
        source_difference_spec: SourceDifferenceSpec | None = None,
        readiness_provider: MujocoSimHILReadinessProvider
        | Callable[..., MujocoSimHILReadinessEvidence]
        | None = None,
        config: MujocoSimHILProfileConfig | None = None,
    ) -> None:
        if not isinstance(runtime, MujocoDeployRuntime):
            raise TypeError("runtime must be a MujocoDeployRuntime.")
        if not isinstance(contracts, MujocoSimHILContracts):
            raise TypeError("contracts must be MujocoSimHILContracts.")
        if not isinstance(policy_input_adapter, PolicyCompositionInputAdapter):
            raise TypeError(
                "policy_input_adapter must be PolicyCompositionInputAdapter."
            )
        if not isinstance(
            human_source_spec,
            PassiveTeleopComponentSourceSpec,
        ):
            raise TypeError(
                "human_source_spec must be PassiveTeleopComponentSourceSpec."
            )
        if not isinstance(human_adapter, HumanProposalAdapter):
            raise TypeError("human_adapter must be HumanProposalAdapter.")
        self.runtime = runtime
        self.contracts = contracts
        self.policy_input_adapter = policy_input_adapter
        self.human_source = human_source
        self.human_source_spec = human_source_spec
        self.human_adapter = human_adapter
        self.human_input_transform = human_input_transform
        if source_difference_spec is not None and not isinstance(
            source_difference_spec,
            SourceDifferenceSpec,
        ):
            raise TypeError(
                "source_difference_spec must be a SourceDifferenceSpec or None."
            )
        if source_difference_spec is not None:
            source_difference_spec.validate_composition_spec(
                contracts.composition_spec
            )
        self.source_difference_spec = source_difference_spec
        self.config = config or MujocoSimHILProfileConfig()
        self._readiness_provider = readiness_provider
        self._lock = threading.RLock()
        self._started = False
        self._activation_scope: CompositionActivationScope | None = None
        self._activation_epoch = 0
        self._cycle_index = 0
        self._next_blend_sequence = 0
        self._human_alpha_by_component = dict(
            contracts.initial_alpha_by_component
        )
        self._blend_alpha_schedule: list[
            tuple[float, int, dict[str, float]]
        ] = []
        self._submission_by_key: dict[
            tuple[str, str],
            MujocoSimHILInputSubmission,
        ] = {}
        self._human_anchor_only_source_ids: set[str] = set()
        self._step_by_decision_key: dict[str, MujocoSimHILCompositionStep] = {}
        self._fact_hash_by_id: dict[str, str] = {}
        policy_binding = self.policy_input_adapter.binding
        if self.human_source_spec.source_id == policy_binding.source_slot_id:
            raise ValueError(
                "Human and policy contributors require distinct source slots."
            )
        self._source_eligibility_spec = CompositionSourceEligibilitySpec(
            spec_id=f"{self.config.profile_id}:source-eligibility",
            composition_spec_hash=(
                self.contracts.composition_spec.semantic_content_hash
            ),
            source_slot_by_role={
                self.config.human_role: self.human_source_spec.source_id,
                self.config.policy_role: policy_binding.source_slot_id,
            },
            producer_id_by_source_slot={
                self.human_source_spec.source_id: (
                    self.config.human_producer_id
                ),
                policy_binding.source_slot_id: policy_binding.producer_id,
            },
        )
        self._source_status_by_role: dict[str, str] = {}
        self._source_status_epoch_by_role: dict[str, int] = {}
        self._source_lease_expires_at_by_role: dict[str, float] = {}
        self._next_source_status_sequence_by_role = {
            self.config.human_role: 0,
            self.config.policy_role: 0,
        }
        self._validate_bindings()

    @property
    def started(self) -> bool:
        return self._started

    @property
    def activation_scope(self) -> CompositionActivationScope | None:
        return self._activation_scope

    def _validate_bindings(self) -> None:
        runtime = self.runtime
        config = self.config
        profile = self.contracts.authority_profile
        if not config.enabled:
            # Default-off is a lifecycle decision, not a weaker contract.
            pass
        if not runtime.pose_execution_enabled:
            raise ValueError("MuJoCo Sim-HIL requires shared pose execution.")
        if not runtime.explicit_shared_safety:
            raise ValueError("MuJoCo Sim-HIL requires explicit shared safety.")
        if runtime.policy_topology.source_eligibility is None:
            raise ValueError(
                "MuJoCo Sim-HIL requires source-eligibility admission."
            )
        self._source_eligibility_spec.validate_composition_spec(
            self.contracts.composition_spec
        )
        runtime_config = runtime.config
        if not runtime_config.enable_composer_authority:
            raise ValueError("MuJoCo Sim-HIL requires composer authority.")
        if runtime_config.enable_primary_policy_authority:
            raise ValueError(
                "MuJoCo Sim-HIL forbids raw primary policy authority."
            )
        if runtime_config.enable_replay_authority:
            raise ValueError("MuJoCo Sim-HIL forbids replay authority.")
        if not runtime_config.enable_component_execution_handoff:
            raise ValueError(
                "MuJoCo Sim-HIL requires component target execution handoff."
            )
        target_port = getattr(runtime, "target_execution_port", None)
        if target_port is None:
            raise ValueError(
                "MuJoCo Sim-HIL requires a target execution submission port."
            )
        if getattr(target_port, "activation_mode", None) != "omitted":
            raise ValueError(
                "MuJoCo Sim-HIL requires immediate target semantics without "
                "a separate activation receipt."
            )
        if runtime_config.composer_producer_id != profile.producer_id:
            raise ValueError("Composer producer does not match the formal profile.")
        if runtime_config.compound_composer_authority_profile != profile:
            raise ValueError(
                "Runtime compound authority profile does not match Sim-HIL contract."
            )
        if runtime_config.reference_activation_profile_id != (
            IMMEDIATE_QREF_SIM_V1
        ):
            raise ValueError(
                "MuJoCo Sim-HIL requires immediate qref activation."
            )
        if not any(
            payload.representation_family == "cartesian_pose"
            for payload in runtime.action_schema.payload_schema_by_component.values()
        ):
            raise ValueError(
                "MuJoCo pose-execution Sim-HIL requires at least one "
                "cartesian_pose component."
            )
        runtime_action_schema_hash = runtime.action_schema.semantic_content_hash
        runtime_capability = getattr(runtime, "component_target_capability", None)
        port_capability = getattr(target_port, "component_target_capability", None)
        if not isinstance(runtime_capability, ComponentTargetCapabilitySpec):
            raise ValueError(
                "MuJoCo Sim-HIL runtime lacks a component target capability."
            )
        if not isinstance(port_capability, ComponentTargetCapabilitySpec):
            raise ValueError(
                "MuJoCo Sim-HIL target port lacks a component target capability."
            )
        if (
            runtime_capability.semantic_content_hash
            != port_capability.semantic_content_hash
        ):
            raise ValueError(
                "MuJoCo Sim-HIL runtime and target port capabilities disagree."
            )
        if runtime_capability.action_schema_hash != runtime_action_schema_hash:
            raise ValueError(
                "MuJoCo Sim-HIL target capability does not bind the runtime "
                "ComponentActionSchema."
            )
        target_contract = getattr(runtime, "target_runtime_action_contract", None)
        target_contract_hash = getattr(target_contract, "semantic_content_hash", None)
        if not target_contract_hash:
            raise ValueError(
                "MuJoCo Sim-HIL runtime lacks a target action contract."
            )
        if any(
            route.output_target_schema_hash != target_contract_hash
            for route in runtime_capability.component_routes
        ):
            raise ValueError(
                "MuJoCo Sim-HIL target routes do not bind one target action "
                "contract."
            )
        if self.contracts.composition_spec.action_schema_hash is None:
            raise ValueError(
                "MuJoCo Sim-HIL composition spec must bind the runtime "
                "ComponentActionSchema."
            )
        if self.contracts.composition_spec.action_schema_hash != (
            runtime_action_schema_hash
        ):
            raise ValueError(
                "Sim-HIL composition spec does not bind the runtime "
                "ComponentActionSchema."
            )
        component_ids = tuple(item.component_id for item in runtime.catalog.components)
        if component_ids != profile.component_coverage:
            raise ValueError("Sim-HIL component scope does not match the runtime catalog.")
        if runtime.catalog.selection_group_ids != profile.selection_groups:
            raise ValueError("Sim-HIL group scope does not match the runtime catalog.")
        configured_strategy = runtime.composition_strategy
        if configured_strategy is None or not isinstance(
            configured_strategy,
            FormalComponentCompositionStrategyV2,
        ):
            raise ValueError("Runtime lacks the formal Sim-HIL component strategy.")
        if configured_strategy.spec != self.contracts.strategy_spec:
            raise ValueError("Runtime formal strategy spec does not match Sim-HIL.")
        if (
            configured_strategy.component_rule_set.semantic_content_hash
            != self.contracts.component_rule_set.semantic_content_hash
        ):
            raise ValueError("Runtime component rule set does not match Sim-HIL.")

        binding = self.policy_input_adapter.binding
        if binding.role != config.policy_role:
            raise ValueError("Policy binding role does not match Sim-HIL profile.")
        if binding.action_schema.semantic_content_hash != runtime_action_schema_hash:
            raise ValueError("Policy binding does not use the runtime action schema.")
        expected_effectors = tuple(
            component.effector_name for component in runtime.catalog.components
        )
        for name, expected in (
            ("runtime_session_id", runtime_config.runtime_session_id),
            ("control_domain_id", runtime_config.control_domain_id),
            ("producer_id", runtime_config.producer_id),
            ("producer_epoch", runtime_config.producer_epoch),
            ("policy_node_id", runtime_config.policy_node_id),
            ("policy_clock_domain_id", runtime_config.clock_domain_id),
            ("component_catalog_hash", runtime.catalog.semantic_content_hash),
        ):
            if getattr(binding, name) != expected:
                raise ValueError(f"Policy binding field {name} does not match runtime.")
        if (
            binding.component_projection_contract_hash is None
            and binding.action_space != runtime.catalog.action_space
        ):
            raise ValueError(
                "Identity policy binding action_space does not match runtime."
            )
        declared_policy_effectors = tuple(binding.required_active_effectors)
        if not declared_policy_effectors:
            raise ValueError("Policy binding must declare at least one contributor component.")
        if len(set(declared_policy_effectors)) != len(declared_policy_effectors):
            raise ValueError("Policy binding contributor components must be unique.")
        if not set(declared_policy_effectors).issubset(set(expected_effectors)):
            raise ValueError(
                "Policy binding contributor components must be a subset of the "
                "complete Sim-HIL scope."
            )

        source_receipt = getattr(self.human_source, "binding_receipt", None)
        if source_receipt is None:
            raise ValueError("Human source must expose a passive binding receipt.")
        for name, expected in (
            ("rig_absent", True),
            ("authoritative", False),
            ("observer_exclusive", True),
        ):
            if getattr(source_receipt, name, None) is not expected:
                raise ValueError(f"Human source binding is not passive: {name}.")
        if source_receipt.source_contract_hash != self.human_source_spec.semantic_content_hash:
            raise ValueError("Human source receipt does not bind its source spec.")
        source_spec = getattr(self.human_source, "spec", None)
        if source_spec is not None and (
            not isinstance(source_spec, PassiveTeleopComponentSourceSpec)
            or source_spec.semantic_content_hash
            != self.human_source_spec.semantic_content_hash
        ):
            raise ValueError("Human source object and source spec disagree.")
        for name in ("start", "step", "close"):
            if not callable(getattr(self.human_source, name, None)):
                raise TypeError(f"human_source must expose callable {name}().")

        adapter_config = self.human_adapter.config
        for name, expected in (
            ("runtime_session_id", runtime_config.runtime_session_id),
            ("control_domain_id", runtime_config.control_domain_id),
            ("embodiment_instance_id", runtime_config.embodiment_instance_id),
            ("producer_id", config.human_producer_id),
            ("producer_epoch", config.human_producer_epoch),
            ("produced_clock_domain_id", runtime_config.clock_domain_id),
            ("effective_clock_domain_id", runtime_config.clock_domain_id),
        ):
            if getattr(adapter_config, name) != expected:
                raise ValueError(f"Human adapter field {name} does not match Sim-HIL.")
        if (
            adapter_config.component_catalog.semantic_content_hash
            != runtime.catalog.semantic_content_hash
        ):
            raise ValueError("Human adapter does not bind the runtime catalog.")
        if adapter_config.action_schema.semantic_content_hash != runtime_action_schema_hash:
            raise ValueError("Human adapter does not use the runtime action schema.")
        if adapter_config.authority_request != "none":
            raise ValueError("Human adapter must remain non-authoritative.")
        expected_human_schema_hash = (
            self.contracts.composition_spec.input_action_schema_hash_by_role.get(
                config.human_role,
                runtime_action_schema_hash,
            )
        )
        transform = self.human_input_transform
        if transform is None:
            if expected_human_schema_hash != runtime_action_schema_hash:
                raise ValueError(
                    "The human role requires a non-default input schema but no "
                    "human_input_transform was installed."
                )
        else:
            for name in ("activate", "transform"):
                if not callable(getattr(transform, name, None)):
                    raise TypeError(
                        f"human_input_transform must expose callable {name}()."
                    )
            input_schema = getattr(transform, "input_action_schema", None)
            output_schema = getattr(transform, "output_action_schema", None)
            if not isinstance(input_schema, ComponentActionSchema):
                raise TypeError(
                    "human_input_transform must expose input_action_schema."
                )
            if not isinstance(output_schema, ComponentActionSchema):
                raise TypeError(
                    "human_input_transform must expose output_action_schema."
                )
            if (
                input_schema.semantic_content_hash
                != runtime_action_schema_hash
            ):
                raise ValueError(
                    "Human input transform does not consume the canonical "
                    "runtime action schema."
                )
            if (
                output_schema.semantic_content_hash
                != expected_human_schema_hash
            ):
                raise ValueError(
                    "Human input transform output schema does not match the "
                    "formal human-role input schema."
                )
        declared_role_schema_hashes = dict(
            self.contracts.composition_spec.input_action_schema_hash_by_role
        )
        installed_role_schema_hashes = {
            role: schema.semantic_content_hash
            for role, schema in (
                runtime.policy_topology.input_action_schema_by_role.items()
            )
        }
        if (
            declared_role_schema_hashes
            and installed_role_schema_hashes != declared_role_schema_hashes
        ):
            raise ValueError(
                "Runtime role-specific input schemas do not match the formal "
                "Sim-HIL contract."
            )
        if config.human_producer_id in {
            binding.producer_id,
            runtime_config.composer_producer_id,
        }:
            raise ValueError("Human contributor producer must be distinct from policy/composer.")
        if config.blend_control_producer_id in {
            binding.producer_id,
            config.human_producer_id,
            runtime_config.composer_producer_id,
        }:
            raise ValueError(
                "Blend-control producer must be distinct from source and "
                "composer producers."
            )
        dynamic_contract = self.contracts.dynamic_parameter_contract
        strategy_control = runtime.composition_strategy_control
        if (dynamic_contract is None) != (strategy_control is None):
            raise ValueError(
                "Sim-HIL dynamic blend contract/runtime control mismatch."
            )
        if dynamic_contract is not None:
            if any(
                not math.isclose(
                    value,
                    config.initial_human_alpha,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                for value in self.contracts.initial_alpha_by_component.values()
            ):
                raise ValueError(
                    "Profile initial_human_alpha does not match its contracts."
                )

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("MujocoSimHILComposition.start() is required.")

    def _readiness_failure(
        self,
        *,
        reason: str,
    ) -> MujocoSimHILReadinessEvidence:
        return MujocoSimHILReadinessEvidence(
            **{name: False for name in _READINESS_FIELDS},
            detail_by_check={name: reason for name in _READINESS_FIELDS},
        )

    def _readiness_snapshot(
        self,
        *,
        now: float,
    ) -> MujocoSimHILReadinessEvidence:
        provider = self._readiness_provider
        if provider is None:
            return self._readiness_failure(
                reason="readiness_provider_required"
            )
        try:
            snapshot = getattr(provider, "snapshot", None)
            if callable(snapshot):
                evidence = snapshot(now=now)
            elif callable(provider):
                evidence = provider(now=now)
            else:
                raise TypeError("readiness_provider is not callable or snapshot-capable.")
        except Exception as error:  # noqa: BLE001 - readiness is fail-closed.
            return self._readiness_failure(
                reason=f"provider_error:{type(error).__name__}:{error}"
            )
        if not isinstance(evidence, MujocoSimHILReadinessEvidence):
            return self._readiness_failure(reason="provider_returned_invalid_evidence")
        return evidence

    def preflight(self, *, now: float | None = None) -> MujocoSimHILPreflightReport:
        timestamp = _finite(
            self.runtime.clock() if now is None else now,
            name="now",
        )
        readiness = self._readiness_snapshot(now=timestamp)
        readiness_failures = self._mix_readiness_failures(
            readiness,
            target_effective_at=timestamp,
        )
        runtime = self.runtime
        profile = self.contracts.authority_profile
        checks = {
            "profile_enabled": self.config.enabled,
            "pose_execution": runtime.pose_execution_enabled,
            "explicit_shared_safety": runtime.explicit_shared_safety,
            "formal_strategy": (
                runtime.composition_strategy is not None
                and isinstance(
                    runtime.composition_strategy,
                    FormalComponentCompositionStrategyV2,
                )
            ),
            "compound_scope": (
                tuple(item.component_id for item in runtime.catalog.components)
                == profile.component_coverage
                and runtime.catalog.selection_group_ids == profile.selection_groups
            ),
            "single_execution_owner": (
                runtime.config.enable_composer_authority
                and not runtime.config.enable_primary_policy_authority
                and not runtime.config.enable_replay_authority
            ),
            "human_passive": bool(
                getattr(
                    getattr(self.human_source, "binding_receipt", None),
                    "rig_absent",
                    False,
                )
                and not getattr(
                    getattr(self.human_source, "binding_receipt", None),
                    "authoritative",
                    True,
                )
            ),
            "policy_contributor_only": not runtime.config.enable_primary_policy_authority,
            "required_readiness": not readiness_failures,
        }
        reasons = [
            f"check_failed:{name}"
            for name, passed in checks.items()
            if not passed
        ]
        reasons.extend(f"readiness_failed:{name}" for name in readiness_failures)
        for name in readiness_failures:
            detail = readiness.detail_by_check.get(name)
            if detail:
                reasons.append(f"readiness_detail:{name}:{detail}")
        return MujocoSimHILPreflightReport(
            readiness=readiness,
            checks=checks,
            reasons=tuple(dict.fromkeys(reasons)),
        )

    @property
    def human_alpha_by_component(self) -> Mapping[str, float]:
        with self._lock:
            return MappingProxyType(
                self._human_alpha_at(float(self.runtime.clock()))
            )

    def _human_alpha_at(self, effective_at: float) -> dict[str, float]:
        target = _finite(effective_at, name="blend effective_at")
        values = dict(self._human_alpha_by_component)
        for scheduled_at, _sequence, scheduled_values in sorted(
            self._blend_alpha_schedule,
            key=lambda item: (item[0], item[1]),
        ):
            if scheduled_at > target + 1e-12:
                break
            values = dict(scheduled_values)
        return values

    def _required_readiness_checks(
        self,
        *,
        target_effective_at: float | None = None,
    ) -> tuple[str, ...]:
        alpha_by_component = self._human_alpha_at(
            float(self.runtime.clock())
            if target_effective_at is None
            else target_effective_at
        )
        if not alpha_by_component:
            return _READINESS_FIELDS
        human_active = any(
            value > 1e-12
            for value in alpha_by_component.values()
        )
        policy_active = any(
            value < 1.0 - 1e-12
            for value in alpha_by_component.values()
        )
        required = {"measured_state_ready"}
        if human_active:
            required.update(
                {
                    "human_device_ready",
                    "calibration_ready",
                    "home_ready",
                    "rebase_ready",
                    "deadman_ready",
                }
            )
        if policy_active:
            required.update(
                {
                    "policy_source_ready",
                    "policy_chunk_ttl_ready",
                }
            )
        return tuple(
            name
            for name in _READINESS_FIELDS
            if name in required
        )

    def _mix_readiness_failures(
        self,
        readiness: MujocoSimHILReadinessEvidence,
        *,
        target_effective_at: float | None = None,
    ) -> tuple[str, ...]:
        required = self._required_readiness_checks(
            target_effective_at=target_effective_at,
        )
        return tuple(
            name
            for name in required
            if not bool(getattr(readiness, name))
        )

    def _runtime_gate_readiness_failures(
        self,
        readiness: MujocoSimHILReadinessEvidence,
        *,
        target_effective_at: float,
    ) -> tuple[str, ...]:
        """Return only failures that require a lower gate transition."""

        return tuple(
            name
            for name in self._mix_readiness_failures(
                readiness,
                target_effective_at=target_effective_at,
            )
            if name not in _COMPOSITION_INPUT_AVAILABILITY_FIELDS
        )

    def _publish_source_eligibility(
        self,
        *,
        readiness: MujocoSimHILReadinessEvidence,
        observed_at: float,
        target_effective_at: float,
    ) -> None:
        """Publish source liveness without granting authority or HOLD."""

        role_status = {
            self.config.human_role: (
                "available" if readiness.human_device_ready else "paused"
            ),
            self.config.policy_role: (
                "available"
                if (
                    readiness.policy_source_ready
                    and readiness.policy_chunk_ttl_ready
                )
                else "paused"
            ),
        }
        binding = self.policy_input_adapter.binding
        producer_by_role = {
            self.config.human_role: (
                self.config.human_producer_id,
                self.config.human_producer_epoch,
            ),
            self.config.policy_role: (
                binding.producer_id,
                binding.producer_epoch,
            ),
        }
        clock_domain_id = self.runtime.config.clock_domain_id
        effective_from = min(observed_at, target_effective_at)
        lease_expires_at = max(observed_at, target_effective_at) + (
            self.runtime.config.proposal_horizon_sec
        )
        for role in self.contracts.composition_spec.input_roles:
            status = role_status[role]
            previous_status = self._source_status_by_role.get(role)
            previous_lease = self._source_lease_expires_at_by_role.get(role)
            if (
                status == previous_status
                and previous_lease is not None
                and previous_lease
                >= target_effective_at
                + 0.5 * self.runtime.config.proposal_horizon_sec
            ):
                continue
            status_epoch = self._source_status_epoch_by_role.get(role, 0)
            if previous_status is not None and status != previous_status:
                status_epoch += 1
            sequence = self._next_source_status_sequence_by_role[role]
            producer_id, producer_epoch = producer_by_role[role]
            source_slot_id = (
                self._source_eligibility_spec.source_slot_by_role[role]
            )
            receipt = self.runtime.submit_composition_source_eligibility(
                CompositionSourceEligibilityDraft(
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    source_slot_id=source_slot_id,
                    producer_id=producer_id,
                    producer_epoch=producer_epoch,
                    status_epoch=status_epoch,
                    sequence=sequence,
                    status=status,
                    observed_at=observed_at,
                    observed_clock_domain_id=clock_domain_id,
                    effective_from=effective_from,
                    lease_expires_at=lease_expires_at,
                    effective_clock_domain_id=clock_domain_id,
                    rearm_required=False,
                    audit_metadata={
                        "composition_profile_id": self.config.profile_id,
                        "input_role": role,
                        "readiness_detail": (
                            {
                                name: readiness.detail_by_check.get(name)
                                for name in (
                                    ("human_device_ready",)
                                    if role == self.config.human_role
                                    else (
                                        "policy_source_ready",
                                        "policy_chunk_ttl_ready",
                                    )
                                )
                            }
                        ),
                    },
                )
            )
            if not receipt.accepted:
                raise RuntimeError(
                    "Source eligibility admission failed for "
                    f"{role!r}: {receipt.rejection_code or receipt.message}"
                )
            self._source_status_by_role[role] = status
            self._source_status_epoch_by_role[role] = status_epoch
            self._source_lease_expires_at_by_role[role] = lease_expires_at
            self._next_source_status_sequence_by_role[role] = sequence + 1

    def required_readiness_checks(
        self,
        *,
        target_effective_at: float | None = None,
    ) -> tuple[str, ...]:
        """Return evidence fields relevant to the selected source mix."""

        with self._lock:
            return self._required_readiness_checks(
                target_effective_at=target_effective_at,
            )

    def readiness_failures(
        self,
        readiness: MujocoSimHILReadinessEvidence,
        *,
        target_effective_at: float | None = None,
    ) -> tuple[str, ...]:
        """Return only the checks required by the currently selected mix."""

        if not isinstance(readiness, MujocoSimHILReadinessEvidence):
            raise TypeError("readiness must be MujocoSimHILReadinessEvidence.")
        with self._lock:
            return self._mix_readiness_failures(
                readiness,
                target_effective_at=target_effective_at,
            )

    def set_human_blend_alpha(
        self,
        alpha: float | Mapping[str, float],
        *,
        effective_from: float | None = None,
    ):
        """Admit one receiver-linearized human-overlay alpha vector.

        Alpha zero selects policy values, alpha one selects human values, and
        intermediate values use the installed pose/qpos component rules.  This
        changes no source, authority, safety, target, or dispatch owner.
        """

        with self._lock:
            self._require_started()
            contract = self.contracts.dynamic_parameter_contract
            strategy_control = self.runtime.composition_strategy_control
            if contract is None or strategy_control is None:
                raise RuntimeError(
                    "This Sim-HIL profile has no dynamic blend control."
                )
            if isinstance(alpha, Mapping):
                values = {
                    _required(key, name="blend component"): _finite(
                        value,
                        name=f"human alpha[{key!r}]",
                    )
                    for key, value in dict(alpha).items()
                }
            else:
                scalar = _finite(alpha, name="human alpha")
                values = {
                    component_id: scalar
                    for component_id in contract.component_ids
                }
            if set(values) != set(contract.component_ids):
                raise ValueError(
                    "Human alpha must exactly cover the dynamic components."
                )
            if any(value < 0.0 or value > 1.0 for value in values.values()):
                raise ValueError("Human alpha values must be in [0, 1].")
            now = float(self.runtime.clock())
            effective = (
                now
                if effective_from is None
                else _finite(effective_from, name="effective_from")
            )
            revision = strategy_control.latest_revision
            receipt = self.runtime.policy_topology.admit_dynamic_parameters(
                CompositionDynamicParameterDraft(
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    producer_id=self.config.blend_control_producer_id,
                    producer_generation=(
                        self.config.blend_control_producer_epoch
                    ),
                    sequence=self._next_blend_sequence,
                    strategy_revision_hash=revision.semantic_content_hash,
                    parameter_contract_hash=contract.semantic_content_hash,
                    alpha_by_component=values,
                    produced_at=now,
                    produced_clock_domain_id=(
                        self.runtime.config.clock_domain_id
                    ),
                    effective_from=effective,
                    effective_until=None,
                    effective_clock_domain_id=(
                        self.runtime.config.clock_domain_id
                    ),
                )
            )
            if receipt.accepted:
                self._next_blend_sequence += 1
                self._blend_alpha_schedule.append(
                    (
                        effective,
                        self._next_blend_sequence - 1,
                        dict(values),
                    )
                )
            return receipt

    def _record_facts(self, facts: tuple[ControlTraceFact, ...]) -> None:
        pending: list[ControlTraceFact] = []
        pending_hash_by_id: dict[str, str] = {}
        for fact in facts:
            if not isinstance(fact, ControlTraceFact):
                raise TypeError("facts must contain only ControlTraceFact values.")
            previous_hash = self._fact_hash_by_id.get(fact.fact_id)
            if previous_hash is None:
                previous_hash = pending_hash_by_id.get(fact.fact_id)
            if previous_hash is not None:
                if previous_hash != fact.semantic_content_hash:
                    raise ValueError("One trace fact ID was reused with another payload.")
                continue
            pending_hash_by_id[fact.fact_id] = fact.semantic_content_hash
            pending.append(fact)
        if pending:
            self.runtime.record_external_facts(tuple(pending))
            self._fact_hash_by_id.update(pending_hash_by_id)

    def _activate_composition(self, *, activated_at: float) -> None:
        self._activation_scope = build_composition_activation_scope(
            operation_spec_id=self.config.profile_id,
            operation_spec_hash=self.contracts.semantic_content_hash,
            session_epoch=self._activation_epoch,
            activated_at=activated_at,
            clock_domain_id=self.runtime.config.clock_domain_id,
            input_barrier_at=activated_at,
            required_roles=self.contracts.composition_spec.required_roles,
        )
        if self.human_input_transform is not None:
            self.human_input_transform.activate(
                self._activation_scope,
                activated_at=activated_at,
            )
        self._record_facts(
            (
                project_composition_activation_scope(
                    self.runtime.trace_context,
                    self._activation_scope,
                ),
            )
        )

    def start(self) -> MujocoSimHILPreflightReport:
        with self._lock:
            if self._started:
                return self.preflight()
            if not self.config.enabled:
                raise RuntimeError(
                    "MuJoCo Sim-HIL is default-off; construct the profile with "
                    "MujocoSimHILProfileConfig(enabled=True)."
                )
            report = self.preflight()
            if not report.passed:
                raise RuntimeError(
                    "MuJoCo Sim-HIL preflight failed: "
                    + ";".join(report.reasons)
                )
            self._submission_by_key.clear()
            self._human_anchor_only_source_ids.clear()
            self._step_by_decision_key.clear()
            self._cycle_index = 0
            source_started = False
            try:
                self.runtime.start()
                self.runtime.activate_policy_node_producer(
                    self.config.human_producer_id,
                    producer_epoch=self.config.human_producer_epoch,
                    reason="control-station sim-hil passive human contributor startup",
                )
                if self.contracts.dynamic_parameter_contract is not None:
                    self.runtime.activate_policy_node_producer(
                        self.config.blend_control_producer_id,
                        producer_epoch=(
                            self.config.blend_control_producer_epoch
                        ),
                        reason=(
                            "control-station sim-hil blend-control startup"
                        ),
                    )
                # The Station application may have started the passive source
                # during its readiness warm-up.  Keep this composition usable
                # on its own, but do not invoke the lower lifecycle twice when
                # the source already exposes the typed started state.
                if not bool(getattr(self.human_source, "started", False)):
                    self.human_source.start()
                    source_started = True
                self._activation_epoch += 1
                self._activate_composition(activated_at=float(self.runtime.clock()))
                self._started = True
                return report
            except Exception:
                if source_started:
                    try:
                        self.human_source.close()
                    except Exception:
                        pass
                self.runtime.close()
                self._activation_scope = None
                raise

    def close(self) -> None:
        with self._lock:
            if not self._started:
                return
            source_error: BaseException | None = None
            try:
                self.human_source.close()
            except BaseException as error:  # noqa: BLE001 - runtime still closes.
                source_error = error
            try:
                self.runtime.close()
            finally:
                self._started = False
                self._activation_scope = None
                self._submission_by_key.clear()
                self._human_anchor_only_source_ids.clear()
                self._step_by_decision_key.clear()
            self._cycle_index = 0
            self._next_blend_sequence = 0
            self._human_alpha_by_component = dict(
                self.contracts.initial_alpha_by_component
            )
            self._blend_alpha_schedule.clear()
            if source_error is not None:
                raise source_error

    def rearm(self, *, now: float | None = None) -> MujocoSimHILPreflightReport:
        """Create a new input activation only after an explicit HOLD.

        Recovery from a latched gate is intentionally not automatic.  A
        temporary contributor underrun does not make Station request HOLD;
        the lower persistence profile may still choose HOLD after its active
        target expires. The existing authoritative runtime remains the owner
        of gate/generation/cancellation evidence; this method only supplies
        the Station lifecycle boundary and a new composition activation scope.
        """

        with self._lock:
            self._require_started()
            timestamp = _finite(
                self.runtime.clock() if now is None else now,
                name="now",
            )
            report = self.preflight(now=timestamp)
            if not report.passed:
                self._hold(
                    reason=(
                        "sim_hil:rearm_readiness_not_ready:"
                        + ",".join(report.reasons)
                    )
                )
                raise RuntimeError(
                    "MuJoCo Sim-HIL rearm preflight failed: "
                    + ";".join(report.reasons)
                )
            gate = self.runtime.capture_execution_gate_snapshot(now=timestamp)
            if gate.status != "hold":
                raise RuntimeError(
                    "MuJoCo Sim-HIL rearm requires the existing execution gate "
                    f"to be HOLD, got {gate.status!r}."
                )
            self._submission_by_key.clear()
            self._human_anchor_only_source_ids.clear()
            self._step_by_decision_key.clear()
            self._activation_epoch += 1
            self._activate_composition(activated_at=timestamp)
            self.runtime.transition_execution_gate(
                "ready",
                reason="sim_hil:explicit_rearm",
                source_event_ids=(self._activation_scope.activation_id,),
            )
            return report

    def request_hold(
        self,
        *,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
        force_cancellation: bool = False,
    ) -> ExecutionGateTransitionOutcome | None:
        """Establish the lower-owned HOLD boundary without closing sources.

        Station lifecycle code calls this before pause, episode release, or
        shutdown.  An existing HOLD is idempotent unless the caller explicitly
        requests another cancellation fence; FAULT is never downgraded.
        """

        with self._lock:
            self._require_started()
            normalized_reason = _required(reason, name="reason")
            now = float(self.runtime.clock())
            gate = self.runtime.capture_execution_gate_snapshot(now=now)
            if gate.status == "fault":
                return None
            if gate.status == "hold" and not force_cancellation:
                return None
            return self.runtime.transition_execution_gate(
                "hold",
                reason=normalized_reason,
                source_event_ids=tuple(source_event_ids),
                force_cancellation=bool(force_cancellation),
            )

    def _submission_error(
        self,
        *,
        role: str,
        source_emission_id: str,
        error: str,
    ) -> MujocoSimHILInputSubmission:
        return MujocoSimHILInputSubmission(
            role=role,
            source_emission_id=source_emission_id,
            error=error,
        )

    def submit_policy_emission(
        self,
        emission: PolicyActionEmission,
        *,
        observed_at: float | None = None,
    ) -> MujocoSimHILInputSubmission:
        with self._lock:
            self._require_started()
            if not isinstance(emission, PolicyActionEmission):
                raise TypeError("emission must be a PolicyActionEmission.")
            key = (self.config.policy_role, emission.emission_id)
            cached = self._submission_by_key.get(key)
            if cached is not None:
                return cached
            now = _finite(
                self.runtime.clock() if observed_at is None else observed_at,
                name="observed_at",
            )
            try:
                facts = project_policy_action_emission(
                    emission,
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    embodiment_instance_id=self.runtime.config.embodiment_instance_id,
                    clock_domain_id=self.runtime.config.clock_domain_id,
                )
                self._record_facts(facts)
                self.policy_input_adapter.binding.validate_emission(emission)
                draft = self.policy_input_adapter.adapt(
                    emission,
                    activation_scope=self._activation_scope,
                    source_fact_ids=tuple(fact.fact_id for fact in facts),
                    source_fact_hashes=tuple(
                        fact.semantic_content_hash for fact in facts
                    ),
                )
                receipt = self.runtime.submit_composition_input(
                    draft,
                    activation_scope=self._activation_scope,
                )
                submission = MujocoSimHILInputSubmission(
                    role=self.config.policy_role,
                    source_emission_id=emission.emission_id,
                    input_receipt=receipt,
                    error=(
                        None
                        if receipt.accepted
                        else receipt.message
                        or receipt.rejection_code
                        or "composition_input_rejected"
                    ),
                )
            except Exception as error:  # noqa: BLE001 - contributor failure holds later.
                submission = self._submission_error(
                    role=self.config.policy_role,
                    source_emission_id=emission.emission_id,
                    error=f"policy_contributor_error:{type(error).__name__}:{error}",
                )
            self._submission_by_key[key] = submission
            return submission

    def submit_human_emission(
        self,
        emission: HumanCommandEmission,
        *,
        observed_at: float | None = None,
    ) -> MujocoSimHILInputSubmission | None:
        with self._lock:
            self._require_started()
            if not isinstance(emission, HumanCommandEmission):
                raise TypeError("emission must be a HumanCommandEmission.")
            source_id_hint = f"human-emission:{emission.emission_sequence}"
            key = (self.config.human_role, source_id_hint)
            if source_id_hint in self._human_anchor_only_source_ids:
                return None
            cached = self._submission_by_key.get(key)
            if cached is not None:
                return cached
            now = _finite(
                self.runtime.clock() if observed_at is None else observed_at,
                name="observed_at",
            )
            try:
                result = self.human_adapter.adapt(
                    emission,
                    effective_from=now,
                    effective_until=(
                        now + self.runtime.config.proposal_horizon_sec
                    ),
                    target_effective_at=now,
                )
                facts = project_human_proposal_adapter_result(
                    emission,
                    result,
                    runtime_session_id=self.runtime.config.runtime_session_id,
                    control_domain_id=self.runtime.config.control_domain_id,
                    embodiment_instance_id=self.runtime.config.embodiment_instance_id,
                    occurred_at=now,
                    clock_domain_id=self.runtime.config.clock_domain_id,
                )
                self._record_facts(facts)
                source_id = result.source_adapter_receipt.source_emission_id
                key = (self.config.human_role, source_id)
                if (
                    not result.source_validation.accepted
                    or result.proposal_draft is None
                ):
                    reason = ",".join(result.source_validation.reasons)
                    submission = self._submission_error(
                        role=self.config.human_role,
                        source_emission_id=source_id,
                        error=f"human_source_validation_failed:{reason or 'no_action'}",
                    )
                else:
                    proposal_draft = result.proposal_draft
                    action = proposal_draft.action
                    transform_audit_metadata: dict[str, Any] = {}
                    if self.human_input_transform is not None:
                        transform_step = self.human_input_transform.transform(
                            action,
                            source_action_id=source_id,
                            source_action_hash=action.semantic_content_hash,
                            source_available_at=now,
                            measured_at=proposal_draft.produced_at,
                            measured_clock_domain_id=(
                                proposal_draft.produced_clock_domain_id
                            ),
                            observed_at=now,
                            observed_clock_domain_id=(
                                self.runtime.config.clock_domain_id
                            ),
                        )
                        transformed_action = getattr(
                            transform_step,
                            "output_action",
                            None,
                        )
                        transform_hash = _required(
                            getattr(
                                transform_step,
                                "semantic_content_hash",
                                "",
                            ),
                            name="human_input_transform step hash",
                        )
                        if transformed_action is None:
                            self._human_anchor_only_source_ids.update(
                                (source_id_hint, source_id)
                            )
                            return None
                        if not isinstance(transformed_action, ComponentAction):
                            raise TypeError(
                                "human_input_transform output_action must be "
                                "a ComponentAction or None."
                            )
                        action = transformed_action
                        transform_audit_metadata = {
                            "human_input_transform_step_hash": transform_hash,
                        }
                    draft = build_composition_input_draft(
                        runtime_session_id=self.runtime.config.runtime_session_id,
                        control_domain_id=self.runtime.config.control_domain_id,
                        producer_id=self.config.human_producer_id,
                        producer_epoch=self.config.human_producer_epoch,
                        sequence=proposal_draft.sequence,
                        input_role=self.config.human_role,
                        action=action,
                        produced_at=proposal_draft.produced_at,
                        produced_clock_domain_id=(
                            proposal_draft.produced_clock_domain_id
                        ),
                        effective_from=proposal_draft.effective_from,
                        effective_until=proposal_draft.effective_until,
                        effective_clock_domain_id=(
                            proposal_draft.effective_clock_domain_id
                        ),
                        context_ids=proposal_draft.context_ids,
                        observation_cut_id=proposal_draft.observation_cut_id,
                        source_fact_ids=tuple(fact.fact_id for fact in facts),
                        source_fact_hashes=tuple(
                            fact.semantic_content_hash for fact in facts
                        ),
                        composition_activation_id=(
                            None
                            if self._activation_scope is None
                            else self._activation_scope.activation_id
                        ),
                        composition_activation_hash=(
                            None
                            if self._activation_scope is None
                            else self._activation_scope.semantic_content_hash
                        ),
                        audit_metadata={
                            **dict(proposal_draft.audit_metadata),
                            "composition_contributor": True,
                            "contributor_kind": "human",
                            "human_emission_id": source_id,
                            "human_emission_hash": emission.semantic_content_hash,
                            "representation_trace_id": (
                                result.source_adapter_receipt.representation_trace_id
                            ),
                            "source_adapter_receipt_hash": (
                                result.source_adapter_receipt.semantic_content_hash
                            ),
                            "passive_source_binding_hash": (
                                self.human_source.binding_receipt.semantic_content_hash
                            ),
                            **transform_audit_metadata,
                        },
                    )
                    receipt = self.runtime.submit_composition_input(
                        draft,
                        activation_scope=self._activation_scope,
                    )
                    submission = MujocoSimHILInputSubmission(
                        role=self.config.human_role,
                        source_emission_id=source_id,
                        input_receipt=receipt,
                        error=(
                            None
                            if receipt.accepted
                            else receipt.message
                            or receipt.rejection_code
                            or "composition_input_rejected"
                        ),
                    )
            except Exception as error:  # noqa: BLE001 - contributor failure holds later.
                submission = self._submission_error(
                    role=self.config.human_role,
                    source_emission_id=source_id_hint,
                    error=f"human_contributor_error:{type(error).__name__}:{error}",
                )
            self._submission_by_key[key] = submission
            return submission

    def poll_human(
        self,
        *,
        now: float | None = None,
    ) -> tuple[Any, MujocoSimHILInputSubmission | None]:
        with self._lock:
            self._require_started()
            timestamp = _finite(
                self.runtime.clock() if now is None else now,
                name="now",
            )
            step = self.human_source.step(now=timestamp)
            if getattr(step, "command_dispatched", False):
                raise RuntimeError("Passive human source reported command dispatch.")
            if getattr(step, "authoritative", True):
                raise RuntimeError("Passive human source reported authority eligibility.")
            emission = getattr(step, "emission", None)
            if emission is None:
                return step, None
            # ``HumanCommandEmission.emitted_at`` is captured inside the
            # source step.  Use a post-step observation cut for admission;
            # using the requested cut can make a valid synchronous emission
            # appear to come from the future by a few milliseconds.
            observed_at = max(
                timestamp,
                _finite(self.runtime.clock(), name="human.observed_at"),
            )
            return step, self.submit_human_emission(
                emission,
                observed_at=observed_at,
            )

    def _hold(
        self,
        *,
        reason: str,
        source_event_ids: tuple[str, ...] = (),
    ) -> ExecutionGateTransitionOutcome:
        """Request lower HOLD and preserve its typed outcome or failure.

        A missing/failed lower transition must never be represented as a
        successful composition cycle with absent cancellation evidence.
        """

        return self.runtime.transition_execution_gate(
            "hold",
            reason=reason,
            source_event_ids=source_event_ids,
            force_cancellation=True,
        )

    def compose_at(
        self,
        *,
        decision_key: str,
        target_effective_at: float,
    ) -> MujocoSimHILCompositionStep:
        with self._lock:
            self._require_started()
            key = _required(decision_key, name="decision_key")
            target = _finite(target_effective_at, name="target_effective_at")
            cached = self._step_by_decision_key.get(key)
            if cached is not None:
                if not math.isclose(
                    cached.target_effective_at,
                    target,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError("decision_key was reused with another target.")
                return cached
            now = float(self.runtime.clock())
            readiness = self._readiness_snapshot(now=now)
            control_cut: ControlCut | None = None
            evaluation = None
            disposition = None
            proposal = None
            hold = None
            blocked_reason = None
            error_text = None
            readiness_failures: tuple[str, ...] = ()
            runtime_gate_failures: tuple[str, ...] = ()
            try:
                gate = self.runtime.capture_execution_gate_snapshot(now=now)
                if gate.status != "ready":
                    blocked_reason = f"execution_gate_not_ready:{gate.status}"
                else:
                    readiness_failures = self._mix_readiness_failures(
                        readiness,
                        target_effective_at=target,
                    )
                    runtime_gate_failures = (
                        self._runtime_gate_readiness_failures(
                            readiness,
                            target_effective_at=target,
                        )
                    )
                if gate.status == "ready" and runtime_gate_failures:
                    blocked_reason = "manual_recovery_required"
                    hold = self._hold(
                        reason=(
                            "sim_hil:readiness_not_ready:"
                            + ",".join(runtime_gate_failures)
                        )
                    )
                elif gate.status == "ready":
                    self._publish_source_eligibility(
                        readiness=readiness,
                        observed_at=now,
                        target_effective_at=target,
                    )
                    # Eligibility admission obtains its receiver timestamp
                    # inside the lower runtime.  Capture the cut afterwards
                    # so the input records and their source state are visible
                    # on one causally closed cut.
                    control_cut = self.runtime.capture_control_cut(
                        cutoff_at=float(self.runtime.clock())
                    )
                    evaluation = self.runtime.evaluate_composition_shadow(
                        spec=self.contracts.composition_spec,
                        target_effective_at=target,
                        control_cut=control_cut,
                        source_difference_spec=self.source_difference_spec,
                        activation_scope=self._activation_scope,
                        source_eligibility_spec=(
                            self._source_eligibility_spec
                        ),
                    )
                    disposition = decide_composition_authority_disposition(
                        evaluation.input_snapshot,
                        evaluation.composition_evaluation,
                        spec=self.contracts.composition_spec,
                        gate_spec=self.contracts.authority_gate_spec,
                    )
                    self._record_facts(
                        (
                            project_composition_authority_disposition(
                                self.runtime.trace_context,
                                disposition,
                                occurred_at=now,
                            ),
                        )
                    )
                    if disposition.downstream_request == "submit_candidate":
                        proposal = self.runtime.submit_evaluated_composition(
                            evaluation,
                            policy_node_id=MUJOCO_SIM_HIL_COMPOSER_NODE_ID_V1,
                            component_coverage=(
                                self.contracts.authority_profile.component_coverage
                            ),
                            selection_groups=(
                                self.contracts.authority_profile.selection_groups
                            ),
                        )
                        receipt = proposal.proposal_admission
                        if receipt is None or not receipt.accepted:
                            raise RuntimeError(
                                "Sim-HIL compound proposal admission was rejected."
                            )
                    elif disposition.downstream_request == "no_action":
                        blocked_reason = (
                            f"composition_no_action:{disposition.reason}"
                        )
                    else:
                        hold = self._hold(
                            reason=f"sim_hil:{disposition.reason}",
                            source_event_ids=(disposition.disposition_id,),
                        )
            except Exception as error:  # noqa: BLE001 - composition must hold.
                error_text = f"{type(error).__name__}:{error}"
                if hold is None:
                    hold = self._hold(
                        reason=f"sim_hil:composition_error:{key}",
                    )
            result = MujocoSimHILCompositionStep(
                decision_key=key,
                target_effective_at=target,
                control_cut=control_cut,
                readiness=readiness,
                evaluation=evaluation,
                disposition=disposition,
                proposal_submission=proposal,
                hold_transition=hold,
                blocked_reason=blocked_reason,
                error=error_text,
            )
            self._step_by_decision_key[key] = result
            return result

    def step(
        self,
        *,
        now: float | None = None,
        policy_emission: PolicyActionEmission | None = None,
        poll_human: bool = True,
        target_effective_at: float | None = None,
        physics_ticks_after_dispatch: int | None = None,
    ) -> MujocoSimHILCycleResult:
        with self._lock:
            self._require_started()
            timestamp = _finite(
                self.runtime.clock() if now is None else now,
                name="now",
            )
            human_step = None
            human_submission = None
            if poll_human:
                human_step, human_submission = self.poll_human(now=timestamp)
            policy_submission = None
            if policy_emission is not None:
                policy_submission = self.submit_policy_emission(
                    policy_emission,
                    observed_at=timestamp,
                )
            requested_target = (
                timestamp
                + self.runtime.reference_activation_spec.target_lead_time_sec
                if target_effective_at is None
                else _finite(
                    target_effective_at,
                    name="target_effective_at",
                )
            )
            # A synchronous human source timestamps its new input at the
            # post-poll observation cut.  Immediate targets must use that same
            # cut; otherwise every fresh VIVE/hand sample is ineligible for
            # the current composition and the runtime silently reuses the
            # preceding sample.  Scheduled targets retain their explicitly
            # requested future activation and remain subject to lower late
            # admission.
            target = (
                max(
                    requested_target,
                    _finite(
                        self.runtime.clock(),
                        name="post_input_observed_at",
                    ),
                )
                if self.runtime.reference_activation_spec.activation_mode
                == "immediate_target"
                else requested_target
            )
            decision_key = f"{self.config.profile_id}:cycle:{self._cycle_index}"
            composition_step = self.compose_at(
                decision_key=decision_key,
                target_effective_at=target,
            )
            physics_ticks = (
                self.config.physics_ticks_after_dispatch
                if physics_ticks_after_dispatch is None
                else int(physics_ticks_after_dispatch)
            )
            if physics_ticks < 0:
                raise ValueError("physics_ticks_after_dispatch must be non-negative.")
            execution_now = max(timestamp, float(self.runtime.clock()))
            execution = self.runtime.step(
                now=execution_now,
                physics_ticks_after_dispatch=physics_ticks,
            )
            result = MujocoSimHILCycleResult(
                cycle_index=self._cycle_index,
                human_step=human_step,
                human_submission=human_submission,
                policy_submission=policy_submission,
                composition_step=composition_step,
                execution=execution,
            )
            self._cycle_index += 1
            return result


def build_mujoco_sim_hil_policy_binding(
    runtime: MujocoDeployRuntime,
    *,
    source_slot_id: str | None = None,
    role: str = MUJOCO_SIM_HIL_POLICY_ROLE_V1,
    policy_active_effectors: Sequence[str] | None = None,
    source_action_space: str | None = None,
    component_id_by_effector: Mapping[str, str] | None = None,
    component_projection_contract_hash: str | None = None,
) -> PolicyCompositionInputBinding:
    """Bind the lower policy emission seam to this exact MuJoCo catalog."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if (
        component_projection_contract_hash is None
        and runtime.catalog.action_space != BIMANUAL_EE_POSE_V1
    ):
        raise ValueError("Sim-HIL policy binding requires a pose target catalog.")
    expected_effectors = tuple(
        component.effector_name for component in runtime.catalog.components
    )
    selected_effectors = expected_effectors if policy_active_effectors is None else tuple(
        str(value).strip().lower() for value in policy_active_effectors
    )
    if not selected_effectors:
        raise ValueError("policy_active_effectors must not be empty.")
    if len(set(selected_effectors)) != len(selected_effectors):
        raise ValueError("policy_active_effectors must not contain duplicates.")
    if not set(selected_effectors).issubset(set(expected_effectors)):
        raise ValueError(
            "policy_active_effectors must be a subset of the runtime catalog."
        )
    mapping = (
        {effector: effector for effector in selected_effectors}
        if component_id_by_effector is None
        else {
            str(effector).strip(): str(component_id).strip()
            for effector, component_id in dict(
                component_id_by_effector
            ).items()
        }
    )
    if set(mapping) != set(selected_effectors):
        raise ValueError(
            "component_id_by_effector must exactly cover policy effectors."
        )
    required_components = tuple(mapping[effector] for effector in selected_effectors)
    return PolicyCompositionInputBinding(
        source_slot_id=(
            runtime.config.producer_id
            if source_slot_id is None
            else source_slot_id
        ),
        role=role,
        runtime_session_id=runtime.config.runtime_session_id,
        control_domain_id=runtime.config.control_domain_id,
        producer_id=runtime.config.producer_id,
        producer_epoch=runtime.config.producer_epoch,
        policy_node_id=runtime.config.policy_node_id,
        policy_clock_domain_id=runtime.config.clock_domain_id,
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        action_space=(
            runtime.catalog.action_space
            if source_action_space is None
            else str(source_action_space)
        ),
        required_active_effectors=selected_effectors,
        action_schema=runtime.action_schema,
        required_component_ids=required_components,
        component_projection_contract_hash=(
            component_projection_contract_hash
        ),
    )


def build_scoped_mujoco_sim_hil_composition(
    *,
    runtime: MujocoDeployRuntime,
    human_source: Any,
    human_source_spec: PassiveTeleopComponentSourceSpec,
    human_input_transform: MujocoSimHILHumanInputTransform | None = None,
    source_difference_spec: SourceDifferenceSpec | None = None,
    readiness_provider: MujocoSimHILReadinessProvider
    | Callable[..., MujocoSimHILReadinessEvidence]
    | None = None,
    policy_binding: PolicyCompositionInputBinding | None = None,
    contracts: MujocoSimHILContracts | None = None,
    config: MujocoSimHILProfileConfig | None = None,
    policy_active_effectors: Sequence[str] | None = None,
    policy_component_action_projector: Any | None = None,
    human_component_action_projector: Any | None = None,
    human_component_id_by_effector: Mapping[str, str] | None = None,
) -> MujocoSimHILComposition:
    """Assemble the default-off profile without constructing lower runtimes."""

    if not isinstance(runtime, MujocoDeployRuntime):
        raise TypeError("runtime must be a MujocoDeployRuntime.")
    if not isinstance(
        human_source_spec,
        PassiveTeleopComponentSourceSpec,
    ):
        raise TypeError(
            "human_source_spec must be PassiveTeleopComponentSourceSpec."
        )
    profile_config = config or MujocoSimHILProfileConfig()
    binding = policy_binding or build_mujoco_sim_hil_policy_binding(
        runtime,
        policy_active_effectors=policy_active_effectors,
    )
    if contracts is None:
        contracts = build_default_mujoco_sim_hil_contracts(
            runtime.config.compound_hand_action_specs_by_side,
            composer_producer_id=runtime.config.composer_producer_id,
            proposal_ttl_sec=runtime.config.proposal_horizon_sec,
            human_role=profile_config.human_role,
            policy_role=profile_config.policy_role,
            policy_active_effectors=policy_active_effectors,
            action_schema_hash=runtime.action_schema.semantic_content_hash,
        )
    policy_adapter = PolicyCompositionInputAdapter(
        binding,
        component_action_projector=policy_component_action_projector,
    )
    human_adapter = build_mujoco_component_human_proposal_adapter(
        runtime,
        human_source_spec,
        producer_id=profile_config.human_producer_id,
        producer_epoch=profile_config.human_producer_epoch,
        component_id_by_effector=human_component_id_by_effector,
        component_action_projector=human_component_action_projector,
    )
    return MujocoSimHILComposition(
        runtime=runtime,
        contracts=contracts,
        policy_input_adapter=policy_adapter,
        human_source=human_source,
        human_source_spec=human_source_spec,
        human_adapter=human_adapter,
        human_input_transform=human_input_transform,
        source_difference_spec=source_difference_spec,
        readiness_provider=readiness_provider,
        config=profile_config,
    )


__all__ = [
    "MUJOCO_SIM_HIL_COMPOSER_NODE_ID_V1",
    "MUJOCO_SIM_HIL_HUMAN_ROLE_V1",
    "MUJOCO_SIM_HIL_POLICY_ROLE_V1",
    "MUJOCO_SIM_HIL_PROFILE_ID_V1",
    "MujocoSimHILComposition",
    "MujocoSimHILCompositionStep",
    "MujocoSimHILContracts",
    "MujocoSimHILCycleResult",
    "MujocoSimHILInputSubmission",
    "MujocoSimHILHumanInputTransform",
    "MujocoSimHILPreflightReport",
    "MujocoSimHILProfileConfig",
    "MujocoSimHILReadinessEvidence",
    "MujocoSimHILReadinessProvider",
    "build_default_mujoco_sim_hil_contracts",
    "build_mujoco_sim_hil_policy_binding",
    "build_scoped_mujoco_sim_hil_composition",
]

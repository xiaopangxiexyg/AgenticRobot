"""Deterministic pre-arbitration action composition contracts.

Composition inputs are internal admitted facts, not authority candidates.  A
composer may publish exactly one new ``ActionProposalDraft`` carrying complete
typed input lineage; raw inputs never enter the global ProposalBuffer unless a
producer explicitly publishes a separate proposal fact.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.runtime.control_admission import (
    ControlAdmissionCoordinator,
    ControlCut,
    admitted_at_or_before,
)
from robot_motion.runtime.identity import RuntimeFactIdentity
from robot_motion.runtime.immutable import deep_freeze
from robot_motion.runtime.producer_epoch import (
    ProducerEpochError,
    ProducerEpochRegistry,
    ProducerEpochSnapshot,
)

from control_composition.retention import (
    TimeWindowRetentionSpec,
    TimeWindowStoreSnapshot,
)
from control_composition.activation import (
    CompositionActivationScope,
    evaluate_composition_input_activation,
)
from control_composition.eligibility import CompositionSourceEligibilitySnapshot
from control_composition.temporal import (
    DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC,
    CompositionTemporalSelectionSpec,
)

from .component_contract import FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2
from .pose import (
    POSE_MIXING_ARM_COMPONENTS_V1,
    fuse_weighted_se3_pose_actions,
    normalize_pose_component_dependencies,
)
from robot_motion.action.proposal import (
    ActionProposalDraft,
    freeze_robot_action,
)
from robot_motion.action.component_action import (
    ComponentAction,
    ComponentActionSchema,
)
from robot_motion.action.component_action_v1 import (
    materialize_robot_action_v1,
    project_robot_action_v1_to_component_action,
)
from robot_motion.action.schema import RobotAction, split_effector_name


COMPOSITION_INPUT_HASH_DOMAIN_V1 = "dexteleop.composition_input.semantic.v1"
COMPOSITION_INPUT_HASH_DOMAIN_V2 = "dexteleop.composition_input.semantic.v2"
COMPOSITION_INPUT_HASH_DOMAIN_V3 = "dexteleop.composition_input.semantic.v3"
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V1 = (
    "dexteleop.composition_input_snapshot.v1"
)
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V2 = (
    "dexteleop.composition_input_snapshot.v2"
)
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V3 = (
    "dexteleop.composition_input_snapshot.v3"
)
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V4 = (
    "dexteleop.composition_input_snapshot.v4"
)
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V5 = (
    "dexteleop.composition_input_snapshot.v5"
)
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V6 = (
    "dexteleop.composition_input_snapshot.v6"
)
COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V7 = (
    "dexteleop.composition_input_snapshot.v7"
)
COMPOSITION_SPEC_HASH_DOMAIN_V1 = "dexteleop.composition_spec.v1"
COMPOSITION_SPEC_HASH_DOMAIN_V2 = "dexteleop.composition_spec.v2"
COMPOSITION_SPEC_HASH_DOMAIN_V3 = "dexteleop.composition_spec.v3"
COMPOSITION_SPEC_HASH_DOMAIN_V4 = "dexteleop.composition_spec.v4"
COMPOSITION_EVALUATION_HASH_DOMAIN_V1 = "dexteleop.composition_evaluation.v1"
COMPOSITION_EVALUATION_HASH_DOMAIN_V2 = "dexteleop.composition_evaluation.v2"
COMPOSITION_EVALUATION_HASH_DOMAIN_V3 = "dexteleop.composition_evaluation.v3"
COMPOSITION_EVALUATION_HASH_DOMAIN_V4 = "dexteleop.composition_evaluation.v4"
COMPOSITION_EVALUATION_HASH_DOMAIN_V5 = "dexteleop.composition_evaluation.v5"
COMPOSITION_EVALUATION_HASH_DOMAIN_V6 = "dexteleop.composition_evaluation.v6"
COMPOSITION_EVALUATION_HASH_DOMAIN_V7 = "dexteleop.composition_evaluation.v7"
COMPOSITION_AUTHORITY_GATE_SPEC_HASH_DOMAIN_V1 = (
    "dexteleop.composition_authority_gate_spec.v1"
)
COMPOSITION_AUTHORITY_DISPOSITION_HASH_DOMAIN_V1 = (
    "dexteleop.composition_authority_disposition.v1"
)
COMPOSITION_ALGEBRAS_V1 = frozenset(
    {
        "component_union",
        "base_plus_bounded_residual",
        "weighted_se3_pose_fusion",
    }
)
COMPOSITION_ALGEBRAS_V2 = frozenset({FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2})


class CompositionAdmissionError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = str(code)


def _required(value: object, *, field_name: str) -> str:
    if value is None:
        raise CompositionAdmissionError(
            "invalid_field", f"{field_name} must be non-empty."
        )
    normalized = str(value).strip()
    if not normalized:
        raise CompositionAdmissionError(
            "invalid_field", f"{field_name} must be non-empty."
        )
    return normalized


def _finite(value: object, *, field_name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise CompositionAdmissionError(
            "invalid_time", f"{field_name} must be finite."
        )
    return normalized


@dataclass(frozen=True)
class CompositionInputDraft:
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    input_role: str
    action: ComponentAction
    produced_at: float
    produced_clock_domain_id: str
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    context_ids: tuple[str, ...] = ()
    observation_cut_id: str | None = None
    policy_output_id: str | None = None
    source_fact_ids: tuple[str, ...] = ()
    source_fact_hashes: tuple[str, ...] = ()
    composition_activation_id: str | None = None
    composition_activation_hash: str | None = None
    authority_candidate: bool = False
    audit_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompositionInputRecord:
    composition_input_id: str
    runtime_session_id: str
    control_domain_id: str
    producer_id: str
    producer_epoch: int
    sequence: int
    input_role: str
    action: ComponentAction
    produced_at: float
    produced_clock_domain_id: str
    available_at: float
    available_clock_domain_id: str
    control_admission_sequence: int
    effective_from: float
    effective_until: float
    effective_clock_domain_id: str
    context_ids: tuple[str, ...]
    observation_cut_id: str | None
    policy_output_id: str | None
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    composition_activation_id: str | None
    composition_activation_hash: str | None
    authority_candidate: bool
    semantic_content_hash: str
    audit_metadata: Mapping[str, Any]


@dataclass(frozen=True)
class CompositionInputAdmissionReceipt:
    accepted: bool
    rejection_code: str | None = None
    message: str = ""
    composition_input_id: str | None = None
    available_at: float | None = None
    available_clock_domain_id: str | None = None
    control_admission_sequence: int | None = None
    semantic_content_hash: str | None = None
    composition_input: CompositionInputRecord | None = None


@dataclass(frozen=True)
class CompositionSpec:
    spec_id: str
    algebra: str
    input_roles: tuple[str, ...]
    required_roles: tuple[str, ...] | None = None
    depends_on_roles_by_role: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    missing_required_behavior: str = "no_action"
    missing_residual_behavior: str = "zero_residual"
    residual_limit_by_component: Mapping[str, float] = field(default_factory=dict)
    weight_by_role: Mapping[str, float] = field(default_factory=dict)
    max_translation_disagreement_m: float | None = None
    max_rotation_disagreement_rad: float | None = None
    pose_component_dependencies: tuple[str, ...] = (
        POSE_MIXING_ARM_COMPONENTS_V1
    )
    component_rule_set_hash: str | None = None
    action_schema_hash: str | None = None
    input_action_schema_hash_by_role: Mapping[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _required(self.spec_id, field_name="spec_id")
        algebra = _required(self.algebra, field_name="algebra")
        if algebra not in COMPOSITION_ALGEBRAS_V1 | COMPOSITION_ALGEBRAS_V2:
            raise ValueError(f"Unsupported composition algebra {algebra!r}.")
        roles = tuple(_required(value, field_name="input_role") for value in self.input_roles)
        required = (
            roles
            if self.required_roles is None
            else tuple(
                _required(value, field_name="required_role")
                for value in self.required_roles
            )
        )
        if not roles or len(roles) != len(set(roles)):
            raise ValueError("Composition input_roles must be non-empty and unique.")
        if len(required) != len(set(required)):
            raise ValueError("required_roles must be unique.")
        if not set(required).issubset(roles):
            raise ValueError("required_roles must be a subset of input_roles.")
        dependencies: dict[str, tuple[str, ...]] = {}
        for raw_role, raw_dependencies in dict(
            self.depends_on_roles_by_role
        ).items():
            role = _required(raw_role, field_name="dependent_role")
            if role not in roles:
                raise ValueError(
                    "Dependency owners must belong to input_roles."
                )
            role_dependencies = tuple(
                _required(value, field_name=f"dependency for {role}")
                for value in raw_dependencies
            )
            if len(role_dependencies) != len(set(role_dependencies)):
                raise ValueError(
                    f"Dependencies for role {role!r} must be unique."
                )
            if role in role_dependencies:
                raise ValueError("A composition role cannot depend on itself.")
            if not set(role_dependencies).issubset(roles):
                raise ValueError(
                    "Role dependencies must belong to input_roles."
                )
            if role_dependencies:
                dependencies[role] = role_dependencies
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(role: str) -> None:
            if role in visiting:
                raise ValueError("Composition role dependencies must be acyclic.")
            if role in visited:
                return
            visiting.add(role)
            for dependency in dependencies.get(role, ()):
                visit(dependency)
            visiting.remove(role)
            visited.add(role)

        for role in roles:
            visit(role)
        required_set = set(required)
        for role in required:
            stack = list(dependencies.get(role, ()))
            while stack:
                dependency = stack.pop()
                if dependency not in required_set:
                    raise ValueError(
                        "A required role may depend only on required roles; "
                        f"{role!r} depends on optional role {dependency!r}."
                    )
                stack.extend(dependencies.get(dependency, ()))
        if self.missing_required_behavior not in {"no_action", "invalid"}:
            raise ValueError("Unsupported missing_required_behavior.")
        if self.missing_residual_behavior not in {
            "zero_residual",
            "no_action",
            "invalid",
        }:
            raise ValueError("Unsupported missing_residual_behavior.")
        limits: dict[str, float] = {}
        for component, raw in dict(self.residual_limit_by_component).items():
            key = _required(component, field_name="residual component")
            limit = float(raw)
            if not math.isfinite(limit) or limit <= 0.0:
                raise ValueError("Residual limits must be finite and positive.")
            limits[key] = limit
        if algebra == "base_plus_bounded_residual":
            if "base" not in roles or "residual" not in roles:
                raise ValueError("base_plus_bounded_residual requires base and residual roles.")
            if not limits:
                raise ValueError("base_plus_bounded_residual requires component limits.")
        weights: dict[str, float] = {}
        for role, raw in dict(self.weight_by_role).items():
            key = _required(role, field_name="weighted role")
            weight = float(raw)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError(
                    "Composition role weights must be finite and non-negative."
                )
            weights[key] = weight
        translation_limit = self.max_translation_disagreement_m
        rotation_limit = self.max_rotation_disagreement_rad
        pose_dependencies = normalize_pose_component_dependencies(
            self.pose_component_dependencies
        )
        component_rule_set_hash = self.component_rule_set_hash
        action_schema_hash = self.action_schema_hash
        input_schema_hashes: dict[str, str] = {}
        for raw_role, raw_hash in dict(
            self.input_action_schema_hash_by_role
        ).items():
            role = _required(raw_role, field_name="input schema role")
            if role not in roles:
                raise ValueError(
                    "Input action schema roles must belong to input_roles."
                )
            input_schema_hashes[role] = _required(
                raw_hash,
                field_name=f"input action schema hash for {role}",
            )
        if input_schema_hashes and set(input_schema_hashes) != set(roles):
            raise ValueError(
                "input_action_schema_hash_by_role must bind every input role "
                "when present."
            )
        if algebra == "weighted_se3_pose_fusion":
            if set(weights) != set(roles) or not any(
                weight > 0.0 for weight in weights.values()
            ):
                raise ValueError(
                    "weighted_se3_pose_fusion requires one role weight per input "
                    "and at least one positive weight."
                )
            if translation_limit is None or rotation_limit is None:
                raise ValueError(
                    "weighted_se3_pose_fusion requires explicit disagreement limits."
                )
            translation_limit = float(translation_limit)
            rotation_limit = float(rotation_limit)
            if not math.isfinite(translation_limit) or translation_limit < 0.0:
                raise ValueError(
                    "max_translation_disagreement_m must be finite and non-negative."
                )
            if (
                not math.isfinite(rotation_limit)
                or rotation_limit < 0.0
                or rotation_limit >= math.pi
            ):
                raise ValueError(
                    "max_rotation_disagreement_rad must be in [0, pi)."
                )
            if limits:
                raise ValueError(
                    "weighted_se3_pose_fusion does not use residual limits."
                )
        elif algebra == FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2:
            if action_schema_hash is not None:
                action_schema_hash = _required(
                    action_schema_hash,
                    field_name="action_schema_hash",
                )
            if component_rule_set_hash is None:
                raise ValueError(
                    "component_rule_set_hash must be non-empty."
                )
            component_rule_set_hash = _required(
                component_rule_set_hash,
                field_name="component_rule_set_hash",
            )
            if input_schema_hashes and action_schema_hash is None:
                raise ValueError(
                    "Role-specific input schemas require an explicit output "
                    "action_schema_hash."
                )
            if (
                limits
                or weights
                or translation_limit is not None
                or rotation_limit is not None
                or pose_dependencies != POSE_MIXING_ARM_COMPONENTS_V1
                or self.missing_residual_behavior != "zero_residual"
            ):
                raise ValueError(
                    "formal_component_strategy_v2 uses only its typed component "
                    "rule set; deterministic-v1 numerical fields are forbidden."
                )
        elif (
            weights
            or translation_limit is not None
            or rotation_limit is not None
            or pose_dependencies != POSE_MIXING_ARM_COMPONENTS_V1
        ):
            raise ValueError(
                "Pose fusion parameters require weighted_se3_pose_fusion."
            )
        if (
            algebra in COMPOSITION_ALGEBRAS_V1
            and component_rule_set_hash is not None
        ):
            raise ValueError(
                "component_rule_set_hash requires formal_component_strategy_v2."
            )
        if algebra in COMPOSITION_ALGEBRAS_V1 and action_schema_hash is not None:
            action_schema_hash = _required(
                action_schema_hash,
                field_name="action_schema_hash",
            )
        if algebra in COMPOSITION_ALGEBRAS_V1 and input_schema_hashes:
            raise ValueError(
                "Role-specific input action schemas require "
                "formal_component_strategy_v2."
            )
        object.__setattr__(self, "algebra", algebra)
        object.__setattr__(self, "input_roles", roles)
        object.__setattr__(self, "required_roles", required)
        object.__setattr__(
            self,
            "depends_on_roles_by_role",
            MappingProxyType(dependencies),
        )
        object.__setattr__(self, "residual_limit_by_component", MappingProxyType(limits))
        object.__setattr__(self, "weight_by_role", MappingProxyType(weights))
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
        object.__setattr__(
            self,
            "pose_component_dependencies",
            pose_dependencies,
        )
        object.__setattr__(
            self,
            "component_rule_set_hash",
            component_rule_set_hash,
        )
        object.__setattr__(self, "action_schema_hash", action_schema_hash)
        object.__setattr__(
            self,
            "input_action_schema_hash_by_role",
            MappingProxyType(input_schema_hashes),
        )

    @property
    def schema_version(self) -> int:
        if self.input_action_schema_hash_by_role:
            return 4
        if self.depends_on_roles_by_role:
            return 3
        return 2 if self.algebra in COMPOSITION_ALGEBRAS_V2 else 1

    @property
    def participation_requirement_by_role(self) -> Mapping[str, str]:
        required = set(self.required_roles)
        return MappingProxyType(
            {
                role: ("required" if role in required else "optional")
                for role in self.input_roles
            }
        )

    @property
    def semantic_content_hash(self) -> str:
        dependency_payload = (
            {
                "participation_requirement_by_role": (
                    self.participation_requirement_by_role
                ),
                "depends_on_roles_by_role": self.depends_on_roles_by_role,
            }
            if self.depends_on_roles_by_role
            else {}
        )
        if self.algebra == FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2:
            schema_payload = (
                {
                    "action_schema_hash": self.action_schema_hash,
                    "input_action_schema_hash_by_role": (
                        self.input_action_schema_hash_by_role
                    ),
                }
                if self.input_action_schema_hash_by_role
                else {}
            )
            return canonical_content_hash(
                {
                    "spec_id": self.spec_id,
                    "algebra": self.algebra,
                    "input_roles": self.input_roles,
                    "required_roles": self.required_roles,
                    "missing_required_behavior": (
                        self.missing_required_behavior
                    ),
                    "component_rule_set_hash": self.component_rule_set_hash,
                    **dependency_payload,
                    **schema_payload,
                },
                domain=(
                    COMPOSITION_SPEC_HASH_DOMAIN_V4
                    if schema_payload
                    else (
                        COMPOSITION_SPEC_HASH_DOMAIN_V3
                        if dependency_payload
                        else COMPOSITION_SPEC_HASH_DOMAIN_V2
                    )
                ),
            )
        payload = {
            "spec_id": self.spec_id,
            "algebra": self.algebra,
            "input_roles": self.input_roles,
            "required_roles": self.required_roles,
            "missing_required_behavior": self.missing_required_behavior,
            "missing_residual_behavior": self.missing_residual_behavior,
            "residual_limit_by_component": self.residual_limit_by_component,
        }
        if self.action_schema_hash is not None:
            payload["action_schema_hash"] = self.action_schema_hash
        if self.algebra == "weighted_se3_pose_fusion":
            payload.update(
                {
                    "weight_by_role": self.weight_by_role,
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
            )
            if (
                self.pose_component_dependencies
                != POSE_MIXING_ARM_COMPONENTS_V1
            ):
                payload["pose_component_dependencies"] = (
                    self.pose_component_dependencies
                )
        return canonical_content_hash(
            {**payload, **dependency_payload},
            domain=(
                COMPOSITION_SPEC_HASH_DOMAIN_V3
                if dependency_payload
                else COMPOSITION_SPEC_HASH_DOMAIN_V1
            ),
        )


@dataclass(frozen=True)
class CompositionAuthorityGateSpec:
    """Compatibility-named output-disposition policy for one composer.

    Contributors remain internal inputs.  Only a ready ``CompositionEvaluation``
    may be promoted, and that promotion belongs to ``composer_producer_id``.
    Missing required/deadman-like roles and numerical conflicts can request a
    lower-runtime HOLD instead of becoming implicit weight fallbacks.

    This object neither owns nor performs authority, canonical safety, HOLD,
    cancellation, or dispatch.  Its historic v1 name and hash payload are kept
    for trace compatibility.  In particular, ``canonical_safety_required`` is
    only a fixed legacy capability assertion; the actual target preflight,
    safety evaluation, execution-gate transition, and receipt remain
    downstream.
    """

    gate_spec_id: str
    composition_spec_hash: str
    composer_producer_id: str
    missing_behavior_by_role: Mapping[str, str]
    proposal_ttl_sec: float
    invalid_result_behavior: str = "hold"
    composer_error_behavior: str = "hold"
    all_inputs_missing_behavior: str = "hold"
    allow_parallel_source_proposals: bool = False
    canonical_safety_required: bool = True

    def __post_init__(self) -> None:
        for name in (
            "gate_spec_id",
            "composition_spec_hash",
            "composer_producer_id",
        ):
            _required(getattr(self, name), field_name=name)
        behaviors: dict[str, str] = {}
        for role, behavior in dict(self.missing_behavior_by_role).items():
            normalized_role = _required(role, field_name="contributor role")
            normalized_behavior = _required(
                behavior,
                field_name=f"missing behavior for {normalized_role}",
            )
            if normalized_behavior not in {"hold", "drop_and_renormalize"}:
                raise ValueError(
                    "Contributor missing behavior must be hold or "
                    "drop_and_renormalize."
                )
            behaviors[normalized_role] = normalized_behavior
        if not behaviors:
            raise ValueError("missing_behavior_by_role must be non-empty.")
        proposal_ttl = float(self.proposal_ttl_sec)
        if not math.isfinite(proposal_ttl) or proposal_ttl <= 0.0:
            raise ValueError("proposal_ttl_sec must be finite and positive.")
        for name in (
            "invalid_result_behavior",
            "composer_error_behavior",
            "all_inputs_missing_behavior",
        ):
            if getattr(self, name) != "hold":
                raise ValueError(f"{name} must be hold in formal composition v1.")
        if self.allow_parallel_source_proposals:
            raise ValueError(
                "Formal composition v1 forbids parallel contributor proposals."
            )
        if not self.canonical_safety_required:
            raise ValueError("Formal composition v1 requires canonical safety.")
        object.__setattr__(
            self,
            "missing_behavior_by_role",
            MappingProxyType(behaviors),
        )
        object.__setattr__(self, "proposal_ttl_sec", proposal_ttl)

    def validate_composition_spec(self, spec: CompositionSpec) -> None:
        if not isinstance(spec, CompositionSpec):
            raise TypeError("spec must be a CompositionSpec.")
        if spec.semantic_content_hash != self.composition_spec_hash:
            raise ValueError("Authority gate is bound to another composition spec.")
        if set(self.missing_behavior_by_role) != set(spec.input_roles):
            raise ValueError(
                "Authority gate must classify every composition input role exactly."
            )
        unsafe_required = {
            role
            for role in spec.required_roles
            if self.missing_behavior_by_role[role] != "hold"
        }
        if unsafe_required:
            raise ValueError(
                "Required formal contributors must HOLD when missing: "
                f"{sorted(unsafe_required)!r}."
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "gate_spec_id": self.gate_spec_id,
                "composition_spec_hash": self.composition_spec_hash,
                "composer_producer_id": self.composer_producer_id,
                "missing_behavior_by_role": self.missing_behavior_by_role,
                "proposal_ttl_sec": self.proposal_ttl_sec,
                "invalid_result_behavior": self.invalid_result_behavior,
                "composer_error_behavior": self.composer_error_behavior,
                "all_inputs_missing_behavior": self.all_inputs_missing_behavior,
                "allow_parallel_source_proposals": (
                    self.allow_parallel_source_proposals
                ),
                "canonical_safety_required": self.canonical_safety_required,
            },
            domain=COMPOSITION_AUTHORITY_GATE_SPEC_HASH_DOMAIN_V1,
        )


@dataclass(frozen=True)
class CompositionInputProjection:
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    records: tuple[CompositionInputRecord, ...]


@dataclass(frozen=True)
class CompositionInputSnapshot:
    composition_input_snapshot_id: str
    control_domain_id: str
    cutoff_at: float
    cutoff_admission_sequence: int
    target_effective_at: float
    composition_spec_hash: str
    action_schema_hash: str
    target_clock_domain_id: str
    selected_input_by_role: Mapping[str, CompositionInputRecord]
    missing_roles: tuple[str, ...]
    producer_epoch_snapshot_hash: str
    semantic_content_hash: str
    temporal_selection_spec_hash: str = (
        DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC.semantic_content_hash
    )
    expected_activation_id: str | None = None
    expected_activation_hash: str | None = None
    source_eligibility_snapshot_id: str | None = None
    source_eligibility_snapshot_hash: str | None = None
    source_ineligibility_reason_by_role: Mapping[str, str] = field(
        default_factory=dict
    )
    selected_input_before_dependency_by_role: Mapping[
        str, CompositionInputRecord
    ] = field(default_factory=dict)
    dependency_ineligibility_reason_by_role: Mapping[str, str] = field(
        default_factory=dict
    )
    required_roles: tuple[str, ...] = ()
    input_action_schema_hash_by_role: Mapping[str, str] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        _required(
            self.action_schema_hash,
            field_name="action_schema_hash",
        )
        _required(
            self.target_clock_domain_id,
            field_name="target_clock_domain_id",
        )
        _required(
            self.temporal_selection_spec_hash,
            field_name="temporal_selection_spec_hash",
        )
        object.__setattr__(
            self,
            "selected_input_by_role",
            MappingProxyType(dict(self.selected_input_by_role)),
        )
        object.__setattr__(
            self,
            "source_ineligibility_reason_by_role",
            MappingProxyType(dict(self.source_ineligibility_reason_by_role)),
        )
        object.__setattr__(
            self,
            "selected_input_before_dependency_by_role",
            MappingProxyType(
                dict(self.selected_input_before_dependency_by_role)
            ),
        )
        object.__setattr__(
            self,
            "dependency_ineligibility_reason_by_role",
            MappingProxyType(
                dict(self.dependency_ineligibility_reason_by_role)
            ),
        )
        input_schema_hashes = {
            _required(role, field_name="input schema role"): _required(
                schema_hash,
                field_name=f"input schema hash for {role}",
            )
            for role, schema_hash in dict(
                self.input_action_schema_hash_by_role
            ).items()
        }
        object.__setattr__(
            self,
            "input_action_schema_hash_by_role",
            MappingProxyType(input_schema_hashes),
        )
        required_roles = tuple(
            _required(value, field_name="required_role")
            for value in self.required_roles
        )
        if len(required_roles) != len(set(required_roles)):
            raise ValueError("Snapshot required_roles must be unique.")
        object.__setattr__(self, "required_roles", required_roles)

@dataclass(frozen=True)
class CompositionEvaluation:
    composition_evaluation_id: str
    composition_input_snapshot_id: str
    composition_input_snapshot_hash: str
    composition_spec_hash: str
    action_schema_hash: str
    target_effective_at: float
    target_clock_domain_id: str
    output_status: str
    reason: str
    composed_candidate: ComponentAction | None
    composed_candidate_hash: str | None
    input_ids: tuple[str, ...]
    input_hashes: tuple[str, ...]
    context_ids: tuple[str, ...]
    semantic_content_hash: str
    composition_activation_id: str | None = None
    composition_activation_hash: str | None = None
    strategy_id: str | None = None
    strategy_spec_hash: str | None = None
    strategy_input_id: str | None = None
    strategy_input_hash: str | None = None
    strategy_context_snapshot_id: str | None = None
    strategy_context_snapshot_hash: str | None = None
    strategy_information_plan_hash: str | None = None
    strategy_used_context_ids: tuple[str, ...] = ()
    strategy_control_binding_id: str | None = None
    strategy_control_binding_hash: str | None = None
    composition_supported_until: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "composition_evaluation_id",
            "composition_input_snapshot_id",
            "composition_input_snapshot_hash",
            "composition_spec_hash",
            "action_schema_hash",
            "target_clock_domain_id",
            "reason",
            "semantic_content_hash",
        ):
            _required(getattr(self, name), field_name=name)
        status = _required(self.output_status, field_name="output_status")
        if status not in {"ready", "no_action", "invalid"}:
            raise ValueError("Unsupported composition evaluation status.")
        if (status == "ready") != (self.composed_candidate is not None):
            raise ValueError(
                "Only a ready composition evaluation may contain a candidate."
            )
        if (self.composed_candidate is None) != (
            self.composed_candidate_hash is None
        ):
            raise ValueError(
                "Composed candidate and candidate hash must appear together."
            )
        target_effective_at = _finite(
            self.target_effective_at,
            field_name="target_effective_at",
        )
        supported_until = self.composition_supported_until
        if status == "ready":
            if supported_until is None:
                raise ValueError(
                    "A ready composition evaluation requires "
                    "composition_supported_until."
                )
            supported_until = _finite(
                supported_until,
                field_name="composition_supported_until",
            )
            if supported_until <= target_effective_at:
                raise ValueError(
                    "Composition support must extend beyond the target time."
                )
        elif supported_until is not None:
            raise ValueError(
                "Only a ready composition evaluation may carry support."
            )
        candidate = self.composed_candidate
        if candidate is not None:
            if not isinstance(candidate, ComponentAction):
                raise TypeError(
                    "CompositionEvaluation candidate must be a ComponentAction."
                )
            if (
                candidate.action_schema.semantic_content_hash
                != self.action_schema_hash
            ):
                raise ValueError("Composed candidate action schema mismatch.")
            if self.composed_candidate_hash != candidate.semantic_content_hash:
                raise ValueError("Composed candidate hash mismatch.")
        object.__setattr__(self, "output_status", status)
        object.__setattr__(self, "target_effective_at", target_effective_at)
        object.__setattr__(
            self,
            "composition_supported_until",
            supported_until,
        )
        object.__setattr__(self, "composed_candidate", candidate)
        identity_fields = (
            self.strategy_id,
            self.strategy_spec_hash,
            self.strategy_input_id,
            self.strategy_input_hash,
        )
        if any(value is not None for value in identity_fields):
            if any(value is None or not str(value).strip() for value in identity_fields):
                raise ValueError(
                    "Formal strategy result identity fields must appear together."
                )
        elif (
            self.strategy_context_snapshot_id is not None
            or self.strategy_context_snapshot_hash is not None
            or self.strategy_information_plan_hash is not None
            or self.strategy_used_context_ids
        ):
            raise ValueError(
                "Strategy lineage requires a formal strategy result identity."
            )
        if (
            self.strategy_control_binding_id is None
        ) != (
            self.strategy_control_binding_hash is None
        ):
            raise ValueError(
                "Strategy control binding ID/hash must appear together."
            )
        if self.strategy_control_binding_id is not None:
            if self.strategy_id is None:
                raise ValueError(
                    "Strategy control binding requires formal strategy identity."
                )
            object.__setattr__(
                self,
                "strategy_control_binding_id",
                _required(
                    self.strategy_control_binding_id,
                    field_name="strategy_control_binding_id",
                ),
            )
            object.__setattr__(
                self,
                "strategy_control_binding_hash",
                _required(
                    self.strategy_control_binding_hash,
                    field_name="strategy_control_binding_hash",
                ),
            )
        if (
            self.strategy_context_snapshot_id is None
        ) != (
            self.strategy_context_snapshot_hash is None
        ):
            raise ValueError(
                "Strategy context snapshot ID/hash must appear together."
            )
        used_context_ids = tuple(
            _required(value, field_name="strategy_used_context_id")
            for value in self.strategy_used_context_ids
        )
        if len(used_context_ids) != len(set(used_context_ids)):
            raise ValueError("strategy_used_context_ids must be unique.")
        object.__setattr__(
            self,
            "strategy_used_context_ids",
            used_context_ids,
        )


@dataclass(frozen=True)
class CompositionAuthorityDisposition:
    """Non-authoritative request bound to one exact composition evaluation.

    ``action="hold"`` means ``request_hold``.  It is not evidence that the
    lower execution gate entered HOLD.  Only a lower-runtime transition and
    cancellation receipt can establish that fact.  The legacy field names are
    retained so existing v1 hashes and stored traces remain readable.
    """

    disposition_id: str
    gate_spec_hash: str
    composition_evaluation_id: str
    composition_evaluation_hash: str
    action: str
    reason: str
    missing_roles: tuple[str, ...]
    semantic_content_hash: str

    @property
    def downstream_request(self) -> str:
        if self.action == "publish_proposal":
            return "submit_candidate"
        if self.action == "hold":
            return "request_hold"
        raise ValueError(
            f"Unsupported legacy composition disposition action {self.action!r}."
        )

    @property
    def authoritative(self) -> bool:
        return False


def decide_composition_authority_disposition(
    snapshot: CompositionInputSnapshot,
    result: CompositionEvaluation,
    *,
    spec: CompositionSpec,
    gate_spec: CompositionAuthorityGateSpec,
) -> CompositionAuthorityDisposition:
    """Request candidate submission or lower-runtime HOLD without side effects."""

    if not isinstance(snapshot, CompositionInputSnapshot):
        raise TypeError("snapshot must be a CompositionInputSnapshot.")
    if not isinstance(result, CompositionEvaluation):
        raise TypeError("result must be a CompositionEvaluation.")
    gate_spec.validate_composition_spec(spec)
    if (
        snapshot.composition_spec_hash != spec.semantic_content_hash
        or result.composition_spec_hash != spec.semantic_content_hash
        or result.composition_input_snapshot_id
        != snapshot.composition_input_snapshot_id
        or result.composition_input_snapshot_hash
        != snapshot.semantic_content_hash
        or result.composition_activation_id
        != snapshot.expected_activation_id
        or result.composition_activation_hash
        != snapshot.expected_activation_hash
    ):
        raise ValueError("Composition disposition inputs do not share one snapshot.")
    missing = tuple(snapshot.missing_roles)
    missing_required = tuple(
        role for role in spec.required_roles if role in missing
    )
    if missing_required:
        action = "hold"
        reason = (
            "all_inputs_missing"
            if len(missing) == len(spec.input_roles)
            else "required_contributor_missing"
        )
    elif result.output_status == "ready" and result.composed_candidate is not None:
        action = "publish_proposal"
        reason = "ready_composition"
    elif missing:
        missing_behaviors = tuple(
            gate_spec.missing_behavior_by_role[role] for role in missing
        )
        if len(missing) == len(spec.input_roles):
            action = gate_spec.all_inputs_missing_behavior
            reason = "all_inputs_missing"
        elif "hold" in missing_behaviors:
            action = "hold"
            reason = "required_contributor_missing"
        else:
            # A renormalizing numerical spec should have produced a ready
            # action.  Reaching no-action here is therefore fail-closed.
            action = "hold"
            reason = "renormalization_produced_no_action"
    elif result.output_status == "invalid":
        action = gate_spec.invalid_result_behavior
        reason = "composition_invalid"
    else:
        action = "hold"
        reason = "composition_not_publishable"
    payload = {
        "gate_spec_hash": gate_spec.semantic_content_hash,
        "composition_evaluation_id": result.composition_evaluation_id,
        "composition_evaluation_hash": result.semantic_content_hash,
        "action": action,
        "reason": reason,
        "missing_roles": missing,
    }
    digest = canonical_content_hash(
        payload,
        domain=COMPOSITION_AUTHORITY_DISPOSITION_HASH_DOMAIN_V1,
    )
    return CompositionAuthorityDisposition(
        disposition_id=f"composition-authority-disposition:{digest}",
        semantic_content_hash=digest,
        **payload,
    )


@dataclass(frozen=True)
class _PreparedInput:
    identity: RuntimeFactIdentity
    fact_id: str
    draft: CompositionInputDraft
    action: ComponentAction
    produced_at: float
    effective_from: float
    effective_until: float
    context_ids: tuple[str, ...]
    source_fact_ids: tuple[str, ...]
    source_fact_hashes: tuple[str, ...]
    audit_metadata: Mapping[str, Any]


def _prepare(draft: CompositionInputDraft) -> _PreparedInput:
    session = _required(draft.runtime_session_id, field_name="runtime_session_id")
    domain = _required(draft.control_domain_id, field_name="control_domain_id")
    producer = _required(draft.producer_id, field_name="producer_id")
    _required(draft.input_role, field_name="input_role")
    _required(draft.produced_clock_domain_id, field_name="produced_clock_domain_id")
    _required(draft.effective_clock_domain_id, field_name="effective_clock_domain_id")
    if draft.authority_candidate:
        raise CompositionAdmissionError(
            "authority_candidate_forbidden",
            "Internal composition inputs cannot be authority candidates.",
        )
    if int(draft.producer_epoch) < 0 or int(draft.sequence) < 0:
        raise CompositionAdmissionError(
            "invalid_identity", "producer_epoch and sequence must be non-negative."
        )
    produced_at = _finite(draft.produced_at, field_name="produced_at")
    effective_from = _finite(draft.effective_from, field_name="effective_from")
    effective_until = _finite(draft.effective_until, field_name="effective_until")
    if effective_until <= effective_from:
        raise CompositionAdmissionError(
            "invalid_effective_interval",
            "Composition input interval must be [effective_from, effective_until).",
        )
    for field_name in ("observation_cut_id", "policy_output_id"):
        value = getattr(draft, field_name)
        if value is not None:
            _required(value, field_name=field_name)
    activation_fields = (
        draft.composition_activation_id,
        draft.composition_activation_hash,
    )
    if any(value is None for value in activation_fields) and any(
        value is not None for value in activation_fields
    ):
        raise CompositionAdmissionError(
            "invalid_activation_binding",
            "Composition activation id and hash must be provided together.",
        )
    if draft.composition_activation_id is not None:
        _required(
            draft.composition_activation_id,
            field_name="composition_activation_id",
        )
        _required(
            draft.composition_activation_hash,
            field_name="composition_activation_hash",
        )
    context_ids = tuple(_required(value, field_name="context_id") for value in draft.context_ids)
    if len(context_ids) != len(set(context_ids)):
        raise CompositionAdmissionError("invalid_field", "context_ids must be unique.")
    source_fact_ids = tuple(
        _required(value, field_name="source_fact_id")
        for value in draft.source_fact_ids
    )
    source_fact_hashes = tuple(
        _required(value, field_name="source_fact_hash")
        for value in draft.source_fact_hashes
    )
    if len(source_fact_ids) != len(source_fact_hashes):
        raise CompositionAdmissionError(
            "invalid_field",
            "source_fact_ids and source_fact_hashes must have equal length.",
        )
    if len(source_fact_ids) != len(set(source_fact_ids)):
        raise CompositionAdmissionError(
            "invalid_field",
            "source_fact_ids must be unique.",
        )
    identity = RuntimeFactIdentity(
        runtime_session_id=session,
        control_domain_id=domain,
        producer_id=producer,
        producer_epoch=int(draft.producer_epoch),
        sequence=int(draft.sequence),
        namespace="composition_input",
    )
    fact_id = identity.fact_id
    if draft.composition_activation_id is not None:
        identity_digest = canonical_content_hash(
            {
                "runtime_session_id": session,
                "control_domain_id": domain,
                "producer_id": producer,
                "producer_epoch": int(draft.producer_epoch),
                "sequence": int(draft.sequence),
                "composition_activation_id": draft.composition_activation_id,
                "composition_activation_hash": (
                    draft.composition_activation_hash
                ),
            },
            domain="dexteleop.composition_input_identity.v2",
        )
        fact_id = f"composition-input:v2:{identity_digest}"
    if not isinstance(draft.action, ComponentAction):
        raise CompositionAdmissionError(
            "invalid_action_carrier",
            "Composition inputs must use the canonical ComponentAction carrier.",
        )
    return _PreparedInput(
        identity=identity,
        fact_id=fact_id,
        draft=draft,
        action=draft.action,
        produced_at=produced_at,
        effective_from=effective_from,
        effective_until=effective_until,
        context_ids=context_ids,
        source_fact_ids=source_fact_ids,
        source_fact_hashes=source_fact_hashes,
        audit_metadata=deep_freeze(dict(draft.audit_metadata)),
    )


def composition_input_semantic_payload(record: CompositionInputRecord) -> dict[str, Any]:
    payload = {
        "composition_input_id": record.composition_input_id,
        "runtime_session_id": record.runtime_session_id,
        "control_domain_id": record.control_domain_id,
        "producer_id": record.producer_id,
        "producer_epoch": record.producer_epoch,
        "sequence": record.sequence,
        "input_role": record.input_role,
        "action": record.action.semantic_payload(),
        "produced_at": record.produced_at,
        "produced_clock_domain_id": record.produced_clock_domain_id,
        "available_at": record.available_at,
        "available_clock_domain_id": record.available_clock_domain_id,
        "control_admission_sequence": record.control_admission_sequence,
        "effective_from": record.effective_from,
        "effective_until": record.effective_until,
        "effective_clock_domain_id": record.effective_clock_domain_id,
        "context_ids": record.context_ids,
        "observation_cut_id": record.observation_cut_id,
        "policy_output_id": record.policy_output_id,
        "authority_candidate": record.authority_candidate,
    }
    if record.source_fact_ids:
        payload["source_fact_ids"] = record.source_fact_ids
        payload["source_fact_hashes"] = record.source_fact_hashes
    if record.composition_activation_id is not None:
        payload["composition_activation_id"] = record.composition_activation_id
        payload["composition_activation_hash"] = (
            record.composition_activation_hash
        )
    return payload


class CompositionInputStore:
    def __init__(
        self,
        *,
        runtime_session_id: str,
        coordinator: ControlAdmissionCoordinator,
        producer_epochs: ProducerEpochRegistry,
        retention_spec: TimeWindowRetentionSpec,
        action_schema: ComponentActionSchema,
        input_action_schema_by_role: Mapping[
            str, ComponentActionSchema
        ] | None = None,
    ) -> None:
        if producer_epochs.coordinator is not coordinator:
            raise ValueError("Composition store and epochs must share one coordinator.")
        if not isinstance(retention_spec, TimeWindowRetentionSpec):
            raise TypeError("retention_spec must be a TimeWindowRetentionSpec.")
        if not isinstance(action_schema, ComponentActionSchema):
            raise TypeError("action_schema must be a ComponentActionSchema.")
        self.runtime_session_id = _required(runtime_session_id, field_name="runtime_session_id")
        if self.runtime_session_id != coordinator.clock_domain.runtime_session_id:
            raise ValueError("Composition store session must match the control clock session.")
        self.coordinator = coordinator
        self.control_domain_id = coordinator.control_domain_id
        self.producer_epochs = producer_epochs
        self.retention_spec = retention_spec
        self.action_schema = action_schema
        input_schemas: dict[str, ComponentActionSchema] = {}
        for raw_role, schema in dict(
            input_action_schema_by_role or {}
        ).items():
            role = _required(raw_role, field_name="input schema role")
            if not isinstance(schema, ComponentActionSchema):
                raise TypeError(
                    "input_action_schema_by_role values must be "
                    "ComponentActionSchema."
                )
            input_schemas[role] = schema
        self.input_action_schema_by_role = MappingProxyType(input_schemas)
        self._records: list[CompositionInputRecord] = []
        self._by_id: dict[str, CompositionInputRecord] = {}
        self._last_sequence: dict[tuple[str, int, str | None], int] = {}
        self._closed_target_through: float | None = None
        self._pruned_record_count = 0
        self._lock = threading.RLock()

    def admit(
        self,
        draft: CompositionInputDraft,
        *,
        activation_scope: CompositionActivationScope | None = None,
    ) -> CompositionInputAdmissionReceipt:
        input_id: str | None = None
        try:
            if activation_scope is not None and not isinstance(
                activation_scope,
                CompositionActivationScope,
            ):
                raise TypeError(
                    "activation_scope must be a CompositionActivationScope or None."
                )
            prepared = _prepare(draft)
            input_id = prepared.fact_id
            if prepared.identity.runtime_session_id != self.runtime_session_id:
                raise CompositionAdmissionError("session_mismatch", "Draft belongs to another session.")
            if prepared.identity.control_domain_id != self.control_domain_id:
                raise CompositionAdmissionError("domain_mismatch", "Draft belongs to another domain.")
            if draft.effective_clock_domain_id != self.coordinator.clock_domain.clock_domain_id:
                raise CompositionAdmissionError(
                    "clock_domain_mismatch",
                    "Composition input validity must use the receiver control clock.",
                )
            expected_action_schema = self.input_action_schema_by_role.get(
                draft.input_role,
                self.action_schema,
            )
            if (
                prepared.action.action_schema.semantic_content_hash
                != expected_action_schema.semantic_content_hash
            ):
                raise CompositionAdmissionError(
                    "action_schema_mismatch",
                    "Composition input action schema does not match its runtime "
                    "role.",
                )
            if (
                activation_scope is not None
                and activation_scope.clock_domain_id
                != self.coordinator.clock_domain.clock_domain_id
            ):
                raise CompositionAdmissionError(
                    "activation_receiver_clock_mismatch",
                    "Composition activation uses another receiver clock domain.",
                )

            def build(available_at: float, admission_sequence: int) -> CompositionInputRecord:
                if activation_scope is not None:
                    activation_verdict = evaluate_composition_input_activation(
                        activation_scope,
                        composition_activation_id=(
                            draft.composition_activation_id
                        ),
                        composition_activation_hash=(
                            draft.composition_activation_hash
                        ),
                        produced_at=prepared.produced_at,
                        produced_clock_domain_id=(
                            draft.produced_clock_domain_id
                        ),
                        available_at=available_at,
                        available_clock_domain_id=(
                            self.coordinator.clock_domain.clock_domain_id
                        ),
                        effective_from=prepared.effective_from,
                        effective_clock_domain_id=(
                            draft.effective_clock_domain_id
                        ),
                    )
                    if not activation_verdict.accepted:
                        raise CompositionAdmissionError(
                            str(activation_verdict.rejection_code),
                            activation_verdict.message,
                        )
                try:
                    self.retention_spec.validate_admitted_interval(
                        available_at=available_at,
                        effective_from=prepared.effective_from,
                        effective_until=prepared.effective_until,
                    )
                except ValueError as exc:
                    raise CompositionAdmissionError(
                        "retention_window_violation", str(exc)
                    ) from exc
                payload = {
                    "composition_input_id": prepared.fact_id,
                    "runtime_session_id": prepared.identity.runtime_session_id,
                    "control_domain_id": prepared.identity.control_domain_id,
                    "producer_id": prepared.identity.producer_id,
                    "producer_epoch": prepared.identity.producer_epoch,
                    "sequence": prepared.identity.sequence,
                    "input_role": draft.input_role,
                    "action": prepared.action,
                    "produced_at": prepared.produced_at,
                    "produced_clock_domain_id": draft.produced_clock_domain_id,
                    "available_at": available_at,
                    "available_clock_domain_id": self.coordinator.clock_domain.clock_domain_id,
                    "control_admission_sequence": admission_sequence,
                    "effective_from": prepared.effective_from,
                    "effective_until": prepared.effective_until,
                    "effective_clock_domain_id": draft.effective_clock_domain_id,
                    "context_ids": prepared.context_ids,
                    "observation_cut_id": draft.observation_cut_id,
                    "policy_output_id": draft.policy_output_id,
                    "source_fact_ids": prepared.source_fact_ids,
                    "source_fact_hashes": prepared.source_fact_hashes,
                    "composition_activation_id": (
                        draft.composition_activation_id
                    ),
                    "composition_activation_hash": (
                        draft.composition_activation_hash
                    ),
                    "authority_candidate": False,
                }
                semantic_payload = dict(payload)
                if not prepared.source_fact_ids:
                    semantic_payload.pop("source_fact_ids")
                    semantic_payload.pop("source_fact_hashes")
                hash_domain = COMPOSITION_INPUT_HASH_DOMAIN_V3
                if draft.composition_activation_id is None:
                    semantic_payload.pop("composition_activation_id")
                    semantic_payload.pop("composition_activation_hash")
                semantic_payload["action"] = prepared.action.semantic_payload()
                return CompositionInputRecord(
                    **payload,
                    semantic_content_hash=canonical_content_hash(
                        semantic_payload, domain=hash_domain
                    ),
                    audit_metadata=prepared.audit_metadata,
                )

            record = self.coordinator.commit(store=self, build=build)
            return CompositionInputAdmissionReceipt(
                accepted=True,
                composition_input_id=record.composition_input_id,
                available_at=record.available_at,
                available_clock_domain_id=record.available_clock_domain_id,
                control_admission_sequence=record.control_admission_sequence,
                semantic_content_hash=record.semantic_content_hash,
                composition_input=record,
            )
        except (CompositionAdmissionError, ProducerEpochError, TypeError, ValueError) as exc:
            code = exc.code if isinstance(exc, CompositionAdmissionError) else "validation_failed"
            return CompositionInputAdmissionReceipt(
                accepted=False,
                rejection_code=code,
                message=str(exc),
                composition_input_id=input_id,
            )

    def _insert_admitted(self, record: CompositionInputRecord) -> None:
        with self._lock:
            try:
                self.producer_epochs.assert_active(
                    producer_id=record.producer_id,
                    producer_epoch=record.producer_epoch,
                )
            except ProducerEpochError as exc:
                raise CompositionAdmissionError("inactive_producer_epoch", str(exc)) from exc
            if record.composition_input_id in self._by_id:
                raise CompositionAdmissionError("duplicate", "Composition input is already admitted.")
            key = (
                record.producer_id,
                record.producer_epoch,
                record.composition_activation_id,
            )
            previous = self._last_sequence.get(key)
            if previous is not None and record.sequence <= previous:
                code = "duplicate" if record.sequence == previous else "sequence_rollback"
                raise CompositionAdmissionError(
                    code, f"Producer sequence must increase beyond {previous}."
                )
            self._records.append(record)
            self._by_id[record.composition_input_id] = record
            self._last_sequence[key] = record.sequence

    def records_at_or_before(self, cut: ControlCut) -> tuple[CompositionInputRecord, ...]:
        if cut.control_domain_id != self.control_domain_id:
            raise ValueError("Control cut belongs to another domain.")
        with self._lock:
            return tuple(record for record in self._records if admitted_at_or_before(record, cut))

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def capture_snapshot(
        self,
        cut: ControlCut,
        *,
        producer_epochs: ProducerEpochSnapshot,
        spec: CompositionSpec,
        target_effective_at: float,
        activation_scope: CompositionActivationScope | None = None,
        source_eligibility_snapshot: (
            CompositionSourceEligibilitySnapshot | None
        ) = None,
        temporal_selection_spec: CompositionTemporalSelectionSpec = (
            DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC
        ),
    ) -> CompositionInputSnapshot:
        """Materialize one same-cut snapshot, then close its target-time window."""

        with self._lock:
            snapshot = build_composition_input_snapshot(
                CompositionInputProjection(
                    control_domain_id=cut.control_domain_id,
                    cutoff_at=cut.cutoff_at,
                    cutoff_admission_sequence=cut.cutoff_admission_sequence,
                    records=tuple(
                        record
                        for record in self._records
                        if admitted_at_or_before(record, cut)
                    ),
                ),
                producer_epochs=producer_epochs,
                spec=spec,
                target_effective_at=target_effective_at,
                activation_scope=activation_scope,
                source_eligibility_snapshot=source_eligibility_snapshot,
                temporal_selection_spec=temporal_selection_spec,
                action_schema_hash=self.action_schema.semantic_content_hash,
                input_action_schema_hash_by_role={
                    role: schema.semantic_content_hash
                    for role, schema in (
                        self.input_action_schema_by_role.items()
                    )
                },
                target_clock_domain_id=(
                    self.coordinator.clock_domain.clock_domain_id
                ),
            )
            self._advance_closed_target_locked(float(target_effective_at))
            return snapshot

    def _advance_closed_target_locked(self, target: float) -> None:
        self._validate_target_locked(target)
        self._closed_target_through = target
        prune_before = self.retention_spec.prune_before(
            closed_target_through=target
        )
        retained = [
            record
            for record in self._records
            if record.effective_until > prune_before
        ]
        self._pruned_record_count += len(self._records) - len(retained)
        self._records = retained
        self._by_id = {
            record.composition_input_id: record for record in retained
        }

    def _validate_target_locked(self, target: float) -> None:
        if not math.isfinite(target):
            raise ValueError("Composition target time must be finite.")
        if (
            self._closed_target_through is not None
            and target < self._closed_target_through
        ):
            raise ValueError(
                "Composition target time cannot move behind the closed window."
            )

    def validate_target(self, target_effective_at: float) -> None:
        with self._lock:
            self._validate_target_locked(float(target_effective_at))

    def retention_snapshot(self) -> TimeWindowStoreSnapshot:
        with self._lock:
            return TimeWindowStoreSnapshot(
                retention_spec_hash=self.retention_spec.semantic_content_hash,
                closed_target_through=self._closed_target_through,
                retained_record_count=len(self._records),
                pruned_record_count=self._pruned_record_count,
                producer_epoch_tombstone_count=len(self._last_sequence),
            )


def project_composition_inputs(
    store: CompositionInputStore, cut: ControlCut
) -> CompositionInputProjection:
    return CompositionInputProjection(
        control_domain_id=cut.control_domain_id,
        cutoff_at=cut.cutoff_at,
        cutoff_admission_sequence=cut.cutoff_admission_sequence,
        records=store.records_at_or_before(cut),
    )


def _epoch_hash(snapshot: ProducerEpochSnapshot) -> str:
    return canonical_content_hash(
        {
            "control_domain_id": snapshot.control_domain_id,
            "cutoff_at": snapshot.cutoff_at,
            "cutoff_admission_sequence": snapshot.cutoff_admission_sequence,
            "active_epoch_by_producer": snapshot.active_epoch_by_producer,
            "activation_id_by_producer": snapshot.activation_id_by_producer,
        },
        domain="dexteleop.composition_epoch_snapshot.v1",
    )


def build_composition_input_snapshot(
    projection: CompositionInputProjection,
    *,
    producer_epochs: ProducerEpochSnapshot,
    spec: CompositionSpec,
    target_effective_at: float,
    activation_scope: CompositionActivationScope | None = None,
    source_eligibility_snapshot: CompositionSourceEligibilitySnapshot | None = None,
    temporal_selection_spec: CompositionTemporalSelectionSpec = (
        DEFAULT_COMPOSITION_TEMPORAL_SELECTION_SPEC
    ),
    action_schema_hash: str | None = None,
    input_action_schema_hash_by_role: Mapping[str, str] | None = None,
    target_clock_domain_id: str | None = None,
) -> CompositionInputSnapshot:
    target = _finite(target_effective_at, field_name="target_effective_at")
    if not isinstance(
        temporal_selection_spec,
        CompositionTemporalSelectionSpec,
    ):
        raise TypeError(
            "temporal_selection_spec must be a "
            "CompositionTemporalSelectionSpec."
        )
    temporal_selection_spec.validate_target(
        cutoff_at=projection.cutoff_at,
        target_effective_at=target,
    )
    if producer_epochs.control_domain_id != projection.control_domain_id:
        raise ValueError("Composition projection and epochs belong to different domains.")
    if producer_epochs.cutoff_admission_sequence != projection.cutoff_admission_sequence:
        raise ValueError("Composition projection and epochs must share one committed cut.")
    if activation_scope is not None:
        if not isinstance(activation_scope, CompositionActivationScope):
            raise TypeError(
                "activation_scope must be a CompositionActivationScope or None."
            )
        if activation_scope.required_roles != spec.required_roles:
            raise ValueError(
                "Composition activation required roles mismatch the spec."
            )
    if source_eligibility_snapshot is not None:
        if not isinstance(
            source_eligibility_snapshot,
            CompositionSourceEligibilitySnapshot,
        ):
            raise TypeError(
                "source_eligibility_snapshot must be a "
                "CompositionSourceEligibilitySnapshot or None."
            )
        if (
            source_eligibility_snapshot.control_domain_id
            != projection.control_domain_id
            or source_eligibility_snapshot.cutoff_at != projection.cutoff_at
            or source_eligibility_snapshot.cutoff_admission_sequence
            != projection.cutoff_admission_sequence
            or source_eligibility_snapshot.target_effective_at != target
        ):
            raise ValueError(
                "Source eligibility and composition inputs must share one cut."
            )
    eligible_by_role: dict[str, list[CompositionInputRecord]] = {
        role: [] for role in spec.input_roles
    }
    for record in projection.records:
        if record.input_role not in spec.input_roles:
            continue
        if producer_epochs.active_epoch_by_producer.get(record.producer_id) != record.producer_epoch:
            continue
        if source_eligibility_snapshot is not None:
            source_state = (
                source_eligibility_snapshot.selected_state_by_role.get(
                    record.input_role
                )
            )
            if (
                not source_eligibility_snapshot.is_role_eligible(
                    record.input_role
                )
                or source_state is None
                or source_state.producer_id != record.producer_id
                or source_state.producer_epoch != record.producer_epoch
            ):
                continue
        if activation_scope is not None:
            activation_verdict = evaluate_composition_input_activation(
                activation_scope,
                composition_activation_id=record.composition_activation_id,
                composition_activation_hash=(
                    record.composition_activation_hash
                ),
                produced_at=record.produced_at,
                produced_clock_domain_id=record.produced_clock_domain_id,
                available_at=record.available_at,
                available_clock_domain_id=(
                    record.available_clock_domain_id
                ),
                effective_from=record.effective_from,
                effective_clock_domain_id=(
                    record.effective_clock_domain_id
                ),
            )
            if not activation_verdict.accepted:
                continue
        if not record.effective_from <= target < record.effective_until:
            continue
        eligible_by_role[record.input_role].append(record)
    selected: dict[str, CompositionInputRecord] = {}
    for role, eligible in eligible_by_role.items():
        if not eligible:
            continue
        if (
            temporal_selection_spec.overlap_behavior
            == "reject_multiple_covering"
            and len(eligible) > 1
        ):
            raise ValueError(
                "Multiple composition inputs cover one role/target under "
                f"reject_multiple_covering: {role!r}."
            )
        selected[role] = max(
            eligible,
            key=lambda record: (
                record.effective_from,
                record.control_admission_sequence,
                record.composition_input_id,
            ),
        )
    observed_schema_hashes = {
        record.action.action_schema.semantic_content_hash
        for record in projection.records
    }
    role_schema_hashes = dict(input_action_schema_hash_by_role or {})
    if not role_schema_hashes:
        role_schema_hashes = dict(spec.input_action_schema_hash_by_role)
    if role_schema_hashes:
        if set(role_schema_hashes) != set(spec.input_roles):
            raise ValueError(
                "Input action schema mapping must bind every composition role."
            )
        for record in projection.records:
            if record.input_role not in role_schema_hashes:
                continue
            if (
                record.action.action_schema.semantic_content_hash
                != role_schema_hashes[record.input_role]
            ):
                raise ValueError(
                    "Composition projection contains an action schema that "
                    "does not match its role."
                )
    if action_schema_hash is None:
        if spec.action_schema_hash is not None:
            action_schema_hash = spec.action_schema_hash
        elif len(observed_schema_hashes) == 1:
            action_schema_hash = next(iter(observed_schema_hashes))
        else:
            raise ValueError(
                "action_schema_hash is required when the projection does not "
                "identify exactly one output action schema."
            )
    action_schema_hash = _required(
        action_schema_hash,
        field_name="action_schema_hash",
    )
    if (
        not role_schema_hashes
        and observed_schema_hashes
        and observed_schema_hashes != {action_schema_hash}
    ):
        raise ValueError("Composition projection contains another action schema.")
    if spec.action_schema_hash is not None and (
        spec.action_schema_hash != action_schema_hash
    ):
        raise ValueError("Composition spec/action schema hash mismatch.")
    observed_clock_domains = {
        record.effective_clock_domain_id for record in projection.records
    }
    if target_clock_domain_id is None:
        if len(observed_clock_domains) != 1:
            raise ValueError(
                "target_clock_domain_id is required when the projection does "
                "not identify exactly one receiver clock."
            )
        target_clock_domain_id = next(iter(observed_clock_domains))
    target_clock_domain_id = _required(
        target_clock_domain_id,
        field_name="target_clock_domain_id",
    )
    if observed_clock_domains and observed_clock_domains != {
        target_clock_domain_id
    }:
        raise ValueError("Composition projection contains another target clock.")
    selected_before_dependency = dict(selected)
    dependency_ineligibility: dict[str, str] = {}
    dependency_changed = True
    while dependency_changed:
        dependency_changed = False
        for role in spec.input_roles:
            if role not in selected:
                continue
            missing_dependencies = tuple(
                dependency
                for dependency in spec.depends_on_roles_by_role.get(role, ())
                if dependency not in selected
            )
            if not missing_dependencies:
                continue
            del selected[role]
            dependency_ineligibility[role] = (
                "prerequisite_unavailable:" + ",".join(missing_dependencies)
            )
            dependency_changed = True
    missing = tuple(role for role in spec.input_roles if role not in selected)
    epoch_hash = _epoch_hash(producer_epochs)
    payload = {
        "control_domain_id": projection.control_domain_id,
        "cutoff_at": projection.cutoff_at,
        "cutoff_admission_sequence": projection.cutoff_admission_sequence,
        "target_effective_at": target,
        "composition_spec_hash": spec.semantic_content_hash,
        "action_schema_hash": action_schema_hash,
        "target_clock_domain_id": target_clock_domain_id,
        "temporal_selection_spec_hash": (
            temporal_selection_spec.semantic_content_hash
        ),
        "selected_input_ids_by_role": {
            role: selected[role].composition_input_id for role in spec.input_roles if role in selected
        },
        "selected_input_hashes_by_role": {
            role: selected[role].semantic_content_hash for role in spec.input_roles if role in selected
        },
        "missing_roles": missing,
        "producer_epoch_snapshot_hash": epoch_hash,
    }
    snapshot_hash_domain = COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V5
    if role_schema_hashes:
        payload["input_action_schema_hash_by_role"] = role_schema_hashes
        snapshot_hash_domain = COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V7
    if spec.depends_on_roles_by_role:
        payload["selected_input_ids_before_dependency_by_role"] = {
            role: selected_before_dependency[role].composition_input_id
            for role in spec.input_roles
            if role in selected_before_dependency
        }
        payload["selected_input_hashes_before_dependency_by_role"] = {
            role: selected_before_dependency[role].semantic_content_hash
            for role in spec.input_roles
            if role in selected_before_dependency
        }
        payload["dependency_ineligibility_reason_by_role"] = (
            dependency_ineligibility
        )
        if not role_schema_hashes:
            snapshot_hash_domain = COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V6
    if activation_scope is not None:
        payload["expected_activation_id"] = activation_scope.activation_id
        payload["expected_activation_hash"] = (
            activation_scope.semantic_content_hash
        )
    if source_eligibility_snapshot is not None:
        payload["source_eligibility_snapshot_id"] = (
            source_eligibility_snapshot.source_eligibility_snapshot_id
        )
        payload["source_eligibility_snapshot_hash"] = (
            source_eligibility_snapshot.semantic_content_hash
        )
        payload["source_ineligibility_reason_by_role"] = (
            source_eligibility_snapshot.ineligible_reason_by_role
        )
    digest = canonical_content_hash(payload, domain=snapshot_hash_domain)
    return CompositionInputSnapshot(
        composition_input_snapshot_id=f"composition-input-snapshot:{digest}",
        control_domain_id=projection.control_domain_id,
        cutoff_at=projection.cutoff_at,
        cutoff_admission_sequence=projection.cutoff_admission_sequence,
        target_effective_at=target,
        composition_spec_hash=spec.semantic_content_hash,
        action_schema_hash=action_schema_hash,
        target_clock_domain_id=target_clock_domain_id,
        selected_input_by_role=selected,
        missing_roles=missing,
        producer_epoch_snapshot_hash=epoch_hash,
        semantic_content_hash=digest,
        temporal_selection_spec_hash=(
            temporal_selection_spec.semantic_content_hash
        ),
        expected_activation_id=(
            None if activation_scope is None else activation_scope.activation_id
        ),
        expected_activation_hash=(
            None
            if activation_scope is None
            else activation_scope.semantic_content_hash
        ),
        source_eligibility_snapshot_id=(
            None
            if source_eligibility_snapshot is None
            else source_eligibility_snapshot.source_eligibility_snapshot_id
        ),
        source_eligibility_snapshot_hash=(
            None
            if source_eligibility_snapshot is None
            else source_eligibility_snapshot.semantic_content_hash
        ),
        source_ineligibility_reason_by_role=(
            {}
            if source_eligibility_snapshot is None
            else source_eligibility_snapshot.ineligible_reason_by_role
        ),
        selected_input_before_dependency_by_role=(
            selected_before_dependency
            if spec.depends_on_roles_by_role
            else selected
        ),
        dependency_ineligibility_reason_by_role=dependency_ineligibility,
        required_roles=spec.required_roles,
        input_action_schema_hash_by_role=role_schema_hashes,
    )


def _component_union(
    inputs: tuple[CompositionInputRecord, ...],
    target: float,
) -> ComponentAction:
    if not inputs:
        raise ValueError("component_union requires at least one input.")
    del target
    action_schema = inputs[0].action.action_schema
    values_by_component: dict[str, Any] = {}
    for item in inputs:
        action = item.action
        if (
            action.action_schema.semantic_content_hash
            != action_schema.semantic_content_hash
        ):
            raise ValueError("component_union action schemas do not match.")
        overlap = set(values_by_component) & set(action.values_by_component)
        if overlap:
            raise ValueError(f"component_union inputs overlap: {sorted(overlap)!r}.")
        values_by_component.update(action.values_by_component)
    return ComponentAction(
        action_schema=action_schema,
        values_by_component=values_by_component,
    )


def _base_plus_residual(
    base: CompositionInputRecord,
    residual: CompositionInputRecord | None,
    *,
    spec: CompositionSpec,
    target: float,
) -> RobotAction:
    base_action = materialize_robot_action_v1(
        base.action,
        timestamp=target,
        source="deterministic_composer:base",
    )
    if base_action.arm_pose_by_side or base_action.end_effector_targets_by_side:
        raise ValueError("v1 bounded residual supports qpos components only.")
    arm = {side: value.copy() for side, value in base_action.arm_qpos_by_side.items()}
    hand = {side: value.copy() for side, value in base_action.hand_qpos_by_side.items()}
    if residual is not None:
        residual_action = materialize_robot_action_v1(
            residual.action,
            timestamp=target,
            source="deterministic_composer:residual",
        )
        if residual_action.action_space != base_action.action_space:
            raise ValueError("Base and residual action spaces do not match.")
        if residual_action.arm_pose_by_side or residual_action.end_effector_targets_by_side:
            raise ValueError("v1 bounded residual supports qpos components only.")
        for side, value in residual_action.arm_qpos_by_side.items():
            component = f"{side}_arm"
            if side not in arm or component not in spec.residual_limit_by_component:
                raise ValueError(f"Residual component {component!r} has no compatible base/limit.")
            limit = spec.residual_limit_by_component[component]
            arm[side] = arm[side] + np.clip(value, -limit, limit)
        for side, value in residual_action.hand_qpos_by_side.items():
            component = f"{side}_hand"
            if side not in hand or component not in spec.residual_limit_by_component:
                raise ValueError(f"Residual component {component!r} has no compatible base/limit.")
            limit = spec.residual_limit_by_component[component]
            hand[side] = hand[side] + np.clip(value, -limit, limit)
    return freeze_robot_action(
        RobotAction(
            action_space=base_action.action_space,
            timestamp=target,
            arm_qpos_by_side=arm,
            hand_qpos_by_side=hand,
            active_effectors=base_action.active_effectors,
            frame_by_effector=base_action.frame_by_effector,
            controlled_frame_by_effector=base_action.controlled_frame_by_effector,
            source="deterministic_composer",
            metadata=base_action.metadata,
        )
    )


def _weighted_se3_pose_fusion(
    inputs_by_role: Mapping[str, CompositionInputRecord],
    *,
    spec: CompositionSpec,
    target: float,
) -> RobotAction:
    selected_weights = {
        role: spec.weight_by_role[role]
        for role in inputs_by_role
        if spec.weight_by_role[role] > 0.0
    }
    if not selected_weights:
        raise ValueError("No positive weighted pose input is available.")
    return freeze_robot_action(
        fuse_weighted_se3_pose_actions(
            {
                role: materialize_robot_action_v1(
                    inputs_by_role[role].action,
                    timestamp=target,
                    source=f"deterministic_composer:{role}",
                )
                for role in selected_weights
            },
            weights_by_source_slot=selected_weights,
            target_effective_at=target,
            max_translation_disagreement_m=float(
                spec.max_translation_disagreement_m
            ),
            max_rotation_disagreement_rad=float(
                spec.max_rotation_disagreement_rad
            ),
            component_dependencies=spec.pose_component_dependencies,
            source=f"deterministic_composer:{spec.spec_id}",
        )
    )


class DeterministicComposer:
    def compose(
        self,
        snapshot: CompositionInputSnapshot,
        *,
        spec: CompositionSpec,
        composition_supported_until: float | None = None,
    ) -> CompositionEvaluation:
        if snapshot.composition_spec_hash != spec.semantic_content_hash:
            raise ValueError("Composition snapshot/spec hash mismatch.")
        selected = snapshot.selected_input_by_role
        missing_required = tuple(role for role in spec.required_roles if role not in selected)
        status = "ready"
        reason = "composed"
        legacy_action: RobotAction | None = None
        action: ComponentAction | None = None
        if spec.algebra == FORMAL_COMPONENT_STRATEGY_ALGEBRA_V2:
            status = "invalid"
            reason = "formal_component_strategy_required"
        elif missing_required:
            status = spec.missing_required_behavior
            reason = "missing_required:" + ",".join(missing_required)
        elif spec.algebra == "base_plus_bounded_residual" and "residual" not in selected:
            if spec.missing_residual_behavior == "zero_residual":
                legacy_action = _base_plus_residual(
                    selected["base"], None, spec=spec, target=snapshot.target_effective_at
                )
                reason = "zero_residual"
            else:
                status = spec.missing_residual_behavior
                reason = "missing_residual"
        else:
            try:
                ordered = tuple(
                    selected[role] for role in spec.input_roles if role in selected
                )
                if spec.algebra == "component_union":
                    action = _component_union(
                        ordered,
                        snapshot.target_effective_at,
                    )
                elif spec.algebra == "base_plus_bounded_residual":
                    legacy_action = _base_plus_residual(
                        selected["base"],
                        selected["residual"],
                        spec=spec,
                        target=snapshot.target_effective_at,
                    )
                else:
                    legacy_action = _weighted_se3_pose_fusion(
                        selected,
                        spec=spec,
                        target=snapshot.target_effective_at,
                    )
                    reason = (
                        "weighted_se3_pose_fusion"
                        if not snapshot.missing_roles
                        else "weighted_se3_pose_fusion_renormalized"
                    )
            except (TypeError, ValueError) as exc:
                status = "invalid"
                reason = str(exc)
        if legacy_action is not None:
            action = project_robot_action_v1_to_component_action(
                legacy_action,
                action_schema=next(iter(selected.values())).action.action_schema,
            )
        ordered_inputs = tuple(
            selected[role] for role in spec.input_roles if role in selected
        )
        input_ids = tuple(item.composition_input_id for item in ordered_inputs)
        input_hashes = tuple(item.semantic_content_hash for item in ordered_inputs)
        context_ids = tuple(
            dict.fromkeys(
                context_id for item in ordered_inputs for context_id in item.context_ids
            )
        )
        action_hash = None if action is None else action.semantic_content_hash
        supported_until = None
        if status == "ready" and ordered_inputs:
            input_supported_until = min(
                item.effective_until for item in ordered_inputs
            )
            supported_until = (
                input_supported_until
                if composition_supported_until is None
                else min(
                    input_supported_until,
                    _finite(
                        composition_supported_until,
                        field_name="composition_supported_until",
                    ),
                )
            )
        payload = {
            "composition_input_snapshot_id": snapshot.composition_input_snapshot_id,
            "composition_input_snapshot_hash": snapshot.semantic_content_hash,
            "composition_spec_hash": spec.semantic_content_hash,
            "action_schema_hash": snapshot.action_schema_hash,
            "target_effective_at": snapshot.target_effective_at,
            "target_clock_domain_id": snapshot.target_clock_domain_id,
            "output_status": status,
            "reason": reason,
            "composed_candidate_hash": action_hash,
            "input_ids": input_ids,
            "input_hashes": input_hashes,
            "context_ids": context_ids,
            "composition_supported_until": supported_until,
        }
        result_hash_domain = COMPOSITION_EVALUATION_HASH_DOMAIN_V7
        if snapshot.expected_activation_id is not None:
            payload["composition_activation_id"] = (
                snapshot.expected_activation_id
            )
            payload["composition_activation_hash"] = (
                snapshot.expected_activation_hash
            )
        digest = canonical_content_hash(payload, domain=result_hash_domain)
        return CompositionEvaluation(
            composition_evaluation_id=f"composition-evaluation:{digest}",
            composed_candidate=action,
            semantic_content_hash=digest,
            **payload,
        )


def build_composite_action_proposal_draft(
    result: CompositionEvaluation,
    *,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    producer_id: str,
    producer_epoch: int,
    sequence: int,
    produced_at: float,
    produced_clock_domain_id: str,
    effective_from: float,
    effective_until: float,
    effective_clock_domain_id: str,
    policy_node_id: str | None = None,
) -> ActionProposalDraft:
    if result.output_status != "ready" or result.composed_candidate is None:
        raise ValueError("Only a ready composition evaluation can become a proposal draft.")
    requested_until = _finite(
        effective_until,
        field_name="effective_until",
    )
    supported_until = result.composition_supported_until
    if supported_until is None:
        raise ValueError("Ready composition result has no support deadline.")
    if requested_until > supported_until:
        raise ValueError(
            "Proposal effective_until cannot exceed composition support."
        )
    return ActionProposalDraft(
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        producer_kind="composer",
        producer_id=producer_id,
        producer_epoch=producer_epoch,
        sequence=sequence,
        action=result.composed_candidate,
        produced_at=produced_at,
        produced_clock_domain_id=produced_clock_domain_id,
        effective_from=effective_from,
        effective_until=requested_until,
        target_effective_at=result.target_effective_at,
        effective_clock_domain_id=effective_clock_domain_id,
        policy_node_id=policy_node_id,
        context_ids=result.context_ids,
        composition_evaluation_id=result.composition_evaluation_id,
        composition_evaluation_hash=result.semantic_content_hash,
        composition_input_ids=result.input_ids,
        composition_input_hashes=result.input_hashes,
        composition_spec_hash=result.composition_spec_hash,
    )


def build_blocked_composition_evaluation(
    snapshot: CompositionInputSnapshot,
    *,
    spec: CompositionSpec,
    status: str,
    reason: str,
) -> CompositionEvaluation:
    """Build a deterministic non-action result for an external dependency gate.

    Context freshness is owned by the policy-topology runtime rather than the
    numerical composer.  This helper lets that runtime preserve the exact
    composition-input snapshot while failing closed before proposal admission.
    """

    normalized_status = _required(status, field_name="composition status")
    if normalized_status not in {"no_action", "invalid"}:
        raise ValueError("Blocked composition status must be no_action or invalid.")
    normalized_reason = _required(reason, field_name="composition reason")
    if snapshot.composition_spec_hash != spec.semantic_content_hash:
        raise ValueError("Composition snapshot/spec hash mismatch.")
    ordered_inputs = tuple(
        snapshot.selected_input_by_role[role]
        for role in spec.input_roles
        if role in snapshot.selected_input_by_role
    )
    input_ids = tuple(item.composition_input_id for item in ordered_inputs)
    input_hashes = tuple(item.semantic_content_hash for item in ordered_inputs)
    context_ids = tuple(
        dict.fromkeys(
            context_id for item in ordered_inputs for context_id in item.context_ids
        )
    )
    payload = {
        "composition_input_snapshot_id": snapshot.composition_input_snapshot_id,
        "composition_input_snapshot_hash": snapshot.semantic_content_hash,
        "composition_spec_hash": spec.semantic_content_hash,
        "action_schema_hash": snapshot.action_schema_hash,
        "target_effective_at": snapshot.target_effective_at,
        "target_clock_domain_id": snapshot.target_clock_domain_id,
        "output_status": normalized_status,
        "reason": normalized_reason,
        "composed_candidate_hash": None,
        "input_ids": input_ids,
        "input_hashes": input_hashes,
        "context_ids": context_ids,
        "composition_supported_until": None,
    }
    result_hash_domain = COMPOSITION_EVALUATION_HASH_DOMAIN_V7
    if snapshot.expected_activation_id is not None:
        payload["composition_activation_id"] = snapshot.expected_activation_id
        payload["composition_activation_hash"] = (
            snapshot.expected_activation_hash
        )
    digest = canonical_content_hash(payload, domain=result_hash_domain)
    return CompositionEvaluation(
        composition_evaluation_id=f"composition-evaluation:{digest}",
        composed_candidate=None,
        semantic_content_hash=digest,
        **payload,
    )


__all__ = [
    "COMPOSITION_ALGEBRAS_V1",
    "COMPOSITION_ALGEBRAS_V2",
    "COMPOSITION_AUTHORITY_DISPOSITION_HASH_DOMAIN_V1",
    "COMPOSITION_AUTHORITY_GATE_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_INPUT_HASH_DOMAIN_V1",
    "COMPOSITION_INPUT_HASH_DOMAIN_V2",
    "COMPOSITION_INPUT_HASH_DOMAIN_V3",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V1",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V2",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V3",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V4",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V5",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V6",
    "COMPOSITION_INPUT_SNAPSHOT_HASH_DOMAIN_V7",
    "COMPOSITION_EVALUATION_HASH_DOMAIN_V1",
    "COMPOSITION_EVALUATION_HASH_DOMAIN_V2",
    "COMPOSITION_EVALUATION_HASH_DOMAIN_V3",
    "COMPOSITION_EVALUATION_HASH_DOMAIN_V4",
    "COMPOSITION_EVALUATION_HASH_DOMAIN_V5",
    "COMPOSITION_EVALUATION_HASH_DOMAIN_V7",
    "COMPOSITION_SPEC_HASH_DOMAIN_V1",
    "COMPOSITION_SPEC_HASH_DOMAIN_V2",
    "COMPOSITION_SPEC_HASH_DOMAIN_V3",
    "COMPOSITION_SPEC_HASH_DOMAIN_V4",
    "CompositionAdmissionError",
    "CompositionAuthorityDisposition",
    "CompositionAuthorityGateSpec",
    "CompositionInputAdmissionReceipt",
    "CompositionInputDraft",
    "CompositionInputProjection",
    "CompositionInputRecord",
    "CompositionInputSnapshot",
    "CompositionInputStore",
    "CompositionEvaluation",
    "CompositionSpec",
    "DeterministicComposer",
    "build_composite_action_proposal_draft",
    "build_blocked_composition_evaluation",
    "build_composition_input_snapshot",
    "composition_input_semantic_payload",
    "decide_composition_authority_disposition",
    "project_composition_inputs",
]

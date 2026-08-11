"""Pure CubeLift Skill/Route/binding/evaluator contracts for B3.

This module freezes the first ordinary authoritative-sim Skill without loading
MuJoCo, a policy checkpoint, or any device.  The Agent supplies no motion
parameters: the exact task, policy, lower manifests, evaluator, and stop
semantics are all server-bound by immutable hashes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .evaluation_contracts import (
    SkillEvaluationPlan,
    SkillEvaluatorSpec,
)
from .payload_validation import AgentSkillPayloadValidatorSet
from .skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)


CUBE_LIFT_SKILL_ID_V1 = "task.cube_lift"
CUBE_LIFT_SIM_ROUTE_ID_V1 = "sim.mujoco.cube_lift.policy_only"
CUBE_LIFT_PARAMETER_PROFILE_ID_V1 = "cube_lift.server_bound"
CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1 = (
    "sim.mujoco.cube_lift.terminal_stable_hold"
)
CUBE_LIFT_EVALUATION_PLAN_ID_V1 = (
    "sim.mujoco.cube_lift.required_environment"
)
CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1 = (
    "dexteleop.agent_skill.cube_lift_execution_result.v1"
)
CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1 = (
    "dexteleop.agent_skill.cube_lift_task_evaluation.v1"
)
CUBE_LIFT_EVALUATOR_EVIDENCE_SCHEMA_V1 = (
    "dexteleop.agent_skill.cube_lift_terminal_evidence.v1"
)
CUBE_LIFT_DEPLOYMENT_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.cube_lift_sim_deployment_binding.v1"
)
CUBE_LIFT_EVALUATOR_CONFIG_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.cube_lift_terminal_evaluator_config.v1"
)
CUBE_LIFT_EVALUATOR_INPUT_SCHEMA_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.cube_lift_terminal_evidence_schema.v1"
)
CUBE_LIFT_PARAMETER_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.cube_lift_parameter_profile.v1"
)
CUBE_LIFT_AGGREGATION_POLICY_ID_V1 = "required_environment_verdict.v1"
CUBE_LIFT_AGGREGATION_POLICY_HASH_V1 = canonical_content_hash(
    {
        "aggregation_policy_id": CUBE_LIFT_AGGREGATION_POLICY_ID_V1,
        "required_evaluator_count": 1,
        "required_evaluator_error": "inconclusive",
        "required_evaluator_missing": "inconclusive",
        "verdict_mapping": {
            "failed": "failed",
            "inconclusive": "inconclusive",
            "succeeded": "succeeded",
        },
    },
    domain="dexteleop.agent_skill.cube_lift_aggregation_policy.v1",
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _boolean(value: object, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean.")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field_name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{field_name} fields mismatch; missing={missing}, unknown={unknown}."
        )


@dataclass(frozen=True)
class CubeLiftTerminalEvaluatorConfig:
    """Exact measured terminal predicate configuration for one deployment."""

    task_config_hash: str
    object_body_name: str
    left_finger_body_name: str
    right_finger_body_name: str
    table_body_name: str
    lift_delta_m: float
    stable_hold_physics_ticks: int
    physics_hz: float
    success_requires_bilateral_contact: bool = True
    requires_no_table_contact: bool = True

    def __post_init__(self) -> None:
        for name in (
            "task_config_hash",
            "object_body_name",
            "left_finger_body_name",
            "right_finger_body_name",
            "table_body_name",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        lift_delta = _finite(self.lift_delta_m, field_name="lift_delta_m")
        if lift_delta <= 0.0:
            raise ValueError("lift_delta_m must be positive.")
        object.__setattr__(self, "lift_delta_m", lift_delta)
        object.__setattr__(
            self,
            "stable_hold_physics_ticks",
            _positive_integer(
                self.stable_hold_physics_ticks,
                field_name="stable_hold_physics_ticks",
            ),
        )
        physics_hz = _finite(self.physics_hz, field_name="physics_hz")
        if physics_hz <= 0.0:
            raise ValueError("physics_hz must be positive.")
        object.__setattr__(self, "physics_hz", physics_hz)
        for name in (
            "success_requires_bilateral_contact",
            "requires_no_table_contact",
        ):
            object.__setattr__(
                self,
                name,
                _boolean(getattr(self, name), field_name=name),
            )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(),
            domain=CUBE_LIFT_EVALUATOR_CONFIG_HASH_DOMAIN_V1,
        )

    def to_wire(self) -> dict[str, object]:
        return {
            "task_config_hash": self.task_config_hash,
            "object_body_name": self.object_body_name,
            "left_finger_body_name": self.left_finger_body_name,
            "right_finger_body_name": self.right_finger_body_name,
            "table_body_name": self.table_body_name,
            "lift_delta_m": self.lift_delta_m,
            "stable_hold_physics_ticks": self.stable_hold_physics_ticks,
            "physics_hz": self.physics_hz,
            "success_requires_bilateral_contact": (
                self.success_requires_bilateral_contact
            ),
            "requires_no_table_contact": self.requires_no_table_contact,
            "verification_gate": "lower_hold",
        }


@dataclass(frozen=True)
class CubeLiftSimDeploymentBinding:
    """Lower-owned exact binding for the first policy-only Sim-HIL Route."""

    deployment_binding_id: str
    deployment_binding_version: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    execution_domain_binding_hash: str
    target_runtime_action_contract_hash: str
    component_target_capability_hash: str
    action_schema_hash: str
    observation_plan_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    task_config_hash: str
    policy_checkpoint_hash: str
    evaluator_configuration_hash: str
    trace_requirement_profile_id: str
    executor_id: str
    explicit_exclusions: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "deployment_binding_id",
            "deployment_binding_version",
            "route_hash",
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "execution_domain_binding_hash",
            "target_runtime_action_contract_hash",
            "component_target_capability_hash",
            "action_schema_hash",
            "observation_plan_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "task_config_hash",
            "policy_checkpoint_hash",
            "evaluator_configuration_hash",
            "trace_requirement_profile_id",
            "executor_id",
        ):
            object.__setattr__(
                self,
                name,
                _required(getattr(self, name), field_name=name),
            )
        exclusions = tuple(
            sorted(
                {
                    _required(item, field_name="explicit_exclusions item")
                    for item in self.explicit_exclusions
                }
            )
        )
        if not exclusions:
            raise ValueError("CubeLift binding must declare explicit exclusions.")
        object.__setattr__(self, "explicit_exclusions", exclusions)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.to_wire(include_hash=False),
            domain=CUBE_LIFT_DEPLOYMENT_BINDING_HASH_DOMAIN_V1,
        )

    def to_wire(self, *, include_hash: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "deployment_binding_id": self.deployment_binding_id,
            "deployment_binding_version": self.deployment_binding_version,
            "route_hash": self.route_hash,
            "runtime_session_id": self.runtime_session_id,
            "control_domain_id": self.control_domain_id,
            "target_binding_id": self.target_binding_id,
            "target_kind": "sim",
            "station_manifest_hash": self.station_manifest_hash,
            "execution_manifest_hash": self.execution_manifest_hash,
            "execution_domain_binding_hash": (
                self.execution_domain_binding_hash
            ),
            "target_runtime_action_contract_hash": (
                self.target_runtime_action_contract_hash
            ),
            "component_target_capability_hash": (
                self.component_target_capability_hash
            ),
            "action_schema_hash": self.action_schema_hash,
            "observation_plan_hash": self.observation_plan_hash,
            "frame_graph_hash": self.frame_graph_hash,
            "tool_config_hash": self.tool_config_hash,
            "calibration_hash": self.calibration_hash,
            "task_config_hash": self.task_config_hash,
            "policy_checkpoint_hash": self.policy_checkpoint_hash,
            "evaluator_configuration_hash": (
                self.evaluator_configuration_hash
            ),
            "trace_requirement_profile_id": (
                self.trace_requirement_profile_id
            ),
            "executor_id": self.executor_id,
            "explicit_exclusions": self.explicit_exclusions,
        }
        if include_hash:
            payload["deployment_binding_hash"] = self.semantic_content_hash
        return payload

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="sim",
            execution_manifest_hash=self.execution_manifest_hash,
        )


def cube_lift_skill_definition_v1() -> SkillDefinition:
    return SkillDefinition(
        skill_id=CUBE_LIFT_SKILL_ID_V1,
        skill_version="1",
        skill_kind="policy_task",
        summary=(
            "Lift the server-bound cube with the right RM65 arm and gripper "
            "using the exact lower-bound policy deployment."
        ),
        intended_target_kind="sim",
        cross_target_deployment_policy="review_required",
        controlled_component_ids=("right_arm", "right_gripper"),
        parameter_semantics_id="server_bound_task.no_agent_motion_parameters.v1",
        execution_result_semantics_id=(
            "authoritative_execution_receipt_chain_and_terminal_hold.v1"
        ),
        task_outcome_semantics_id=(
            "cube_lift.measured_terminal_stable_hold.v1"
        ),
        entry_predicate_ids=(
            "policy_only_sim_hil_ready.v1",
            "station_gate_ready.v1",
            "task_reset_evidence_available.v1",
        ),
        task_outcome_predicate_ids=(
            "cube_above_resting_baseline.v1",
            "bilateral_finger_contact_stable.v1",
            "cube_not_in_table_contact.v1",
            "finite_state_stable_under_hold.v1",
        ),
    )


def _cube_lift_execution_result_schema_v1() -> dict[str, Any]:
    references = (
        "execution_evidence_chain_ref",
        "source_proposal_chain_hash",
        "authority_decision_chain_hash",
        "safety_decision_chain_hash",
        "target_request_chain_hash",
        "target_submission_receipt_chain_hash",
        "target_completion_receipt_chain_hash",
        "resolved_component_target_chain_hash",
        "component_admission_receipt_chain_hash",
        "dispatch_receipt_chain_hash",
        "measurement_lineage_chain_hash",
        "first_cycle_receipt_ref",
        "last_cycle_receipt_ref",
        "terminal_hold_receipt_ref",
        "terminal_state_hash",
        "trace_terminal_receipt_ref",
        "trace_commit_receipt_ref",
    )
    properties: dict[str, Any] = {
        "schema_version": {"const": CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1},
        "runtime_session_id": {"type": "string", "minLength": 1},
        "control_domain_id": {"type": "string", "minLength": 1},
        **{
            name: {"type": "string", "minLength": 1}
            for name in references
        },
        "cycle_count": {"type": "integer", "minimum": 1},
        "dispatched_cycle_count": {"type": "integer", "minimum": 1},
        "measurement_count": {"type": "integer", "minimum": 1},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _cube_lift_task_evaluation_schema_v1() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"const": CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1},
        "verdict": {
            "type": "string",
            "enum": ["failed", "inconclusive", "succeeded"],
        },
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "evaluated_cut_id": {"type": "string", "minLength": 1},
        "environment_evidence_hash": {"type": "string", "minLength": 1},
        "evaluator_configuration_hash": {"type": "string", "minLength": 1},
        "terminal_state_hash": {"type": "string", "minLength": 1},
        "baseline_resting_center_z_m": {"type": "number"},
        "required_lift_delta_m": {"type": "number", "exclusiveMinimum": 0},
        "minimum_verified_cube_center_z_m": {"type": "number"},
        "final_cube_center_z_m": {"type": "number"},
        "stable_hold_required_physics_ticks": {
            "type": "integer",
            "minimum": 1,
        },
        "stable_hold_observed_physics_ticks": {
            "type": "integer",
            "minimum": 0,
        },
        "finite_state_all_ticks": {"type": "boolean"},
        "bilateral_contact_all_ticks": {"type": "boolean"},
        "table_contact_any_tick": {"type": "boolean"},
        "terminal_hold_confirmed": {"type": "boolean"},
        "execution_lineage_complete": {"type": "boolean"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def cube_lift_schema_entry_v1(
    *,
    skill_definition_hash: str,
) -> AgentSkillSchemaEntry:
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }
    parameter_schema_hash = agent_skill_schema_document_hash(
        parameter_schema,
        schema_kind="parameter",
    )
    profile_hash = canonical_content_hash(
        {
            "parameter_profile_id": CUBE_LIFT_PARAMETER_PROFILE_ID_V1,
            "parameter_schema_hash": parameter_schema_hash,
            "agent_selectable_parameters": (),
            "binding_semantics": "server_bound_exact_deployment",
        },
        domain=CUBE_LIFT_PARAMETER_PROFILE_HASH_DOMAIN_V1,
    )
    return AgentSkillSchemaEntry(
        skill_id=CUBE_LIFT_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=_cube_lift_execution_result_schema_v1(),
        task_evaluation_schema=_cube_lift_task_evaluation_schema_v1(),
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=CUBE_LIFT_PARAMETER_PROFILE_ID_V1,
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=parameter_schema_hash,
            ),
        ),
    )


def validate_cube_lift_parameters_v1(parameters: Mapping[str, Any]) -> None:
    """The server-bound CubeLift profile exposes no Agent-selected fields."""

    _exact_fields(
        parameters,
        expected=frozenset(),
        field_name="parameters",
    )


def cube_lift_sim_route_v1(
    *,
    definition: SkillDefinition,
    schema_entry: AgentSkillSchemaEntry,
) -> SkillExecutionRoute:
    if definition.skill_id != CUBE_LIFT_SKILL_ID_V1:
        raise ValueError("CubeLift Route requires the CubeLift definition.")
    if schema_entry.skill_id != CUBE_LIFT_SKILL_ID_V1:
        raise ValueError("CubeLift Route requires the CubeLift schema entry.")
    if schema_entry.skill_definition_hash != definition.semantic_content_hash:
        raise ValueError("CubeLift schema does not bind the exact definition.")
    return SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=CUBE_LIFT_SIM_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema_entry.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=True,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=30_000,
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        max_concurrent_operations=1,
        requires_task_evaluation=True,
    )


def cube_lift_evaluator_input_schema_v1(
    config: CubeLiftTerminalEvaluatorConfig,
) -> dict[str, Any]:
    if not isinstance(config, CubeLiftTerminalEvaluatorConfig):
        raise TypeError("config must be a CubeLiftTerminalEvaluatorConfig.")
    sample_properties = {
        "physics_tick_sequence": {"type": "integer", "minimum": 0},
        "measured_at": {"type": "number"},
        "cube_center_z_m": {"type": "number"},
        "left_finger_cube_contact": {"type": "boolean"},
        "right_finger_cube_contact": {"type": "boolean"},
        "table_cube_contact": {"type": "boolean"},
        "finite_state": {"type": "boolean"},
        "source_measurement_ref": {"type": "string", "minLength": 1},
    }
    properties = {
        "schema_version": {"const": CUBE_LIFT_EVALUATOR_EVIDENCE_SCHEMA_V1},
        "runtime_session_id": {"type": "string", "minLength": 1},
        "control_domain_id": {"type": "string", "minLength": 1},
        "evaluated_cut_id": {"type": "string", "minLength": 1},
        "baseline_resting_center_z_m": {"type": "number"},
        "terminal_hold_receipt_ref": {"type": "string", "minLength": 1},
        "execution_evidence_chain_ref": {"type": "string", "minLength": 1},
        "execution_lineage_complete": {"type": "boolean"},
        "samples": {
            "type": "array",
            "minItems": config.stable_hold_physics_ticks,
            "maxItems": config.stable_hold_physics_ticks,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": sample_properties,
                "required": list(sample_properties),
            },
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def cube_lift_evaluator_input_schema_hash_v1(
    config: CubeLiftTerminalEvaluatorConfig,
) -> str:
    return canonical_content_hash(
        cube_lift_evaluator_input_schema_v1(config),
        domain=CUBE_LIFT_EVALUATOR_INPUT_SCHEMA_HASH_DOMAIN_V1,
    )


def cube_lift_environment_evaluator_spec_v1(
    *,
    config: CubeLiftTerminalEvaluatorConfig,
    task_evaluation_schema_hash: str,
) -> SkillEvaluatorSpec:
    return SkillEvaluatorSpec(
        evaluator_id=CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1,
        evaluator_version="1",
        evaluator_kind="environment",
        input_evidence_schema_hash=(
            cube_lift_evaluator_input_schema_hash_v1(config)
        ),
        output_schema_hash=_required(
            task_evaluation_schema_hash,
            field_name="task_evaluation_schema_hash",
        ),
        role="required",
        execution_location="lower_runtime",
        deadline_ms=5_000,
        failure_policy="veto",
        evidence_domain_id="sim.mujoco.scene.ground_truth",
        configuration_hash=config.semantic_content_hash,
    )


def cube_lift_evaluation_plan_v1(
    *,
    definition: SkillDefinition,
    route: SkillExecutionRoute,
    deployment_binding: CubeLiftSimDeploymentBinding,
    evaluator_config: CubeLiftTerminalEvaluatorConfig,
    task_evaluation_schema_hash: str,
) -> SkillEvaluationPlan:
    validate_cube_lift_sim_route_v1(route=route)
    if definition.semantic_content_hash != route.skill_definition_hash:
        raise ValueError("CubeLift plan definition does not match the Route.")
    if deployment_binding.route_hash != route.semantic_content_hash:
        raise ValueError("CubeLift deployment binding does not match the Route.")
    if (
        deployment_binding.evaluator_configuration_hash
        != evaluator_config.semantic_content_hash
    ):
        raise ValueError(
            "CubeLift deployment binding does not match evaluator config."
        )
    if deployment_binding.task_config_hash != evaluator_config.task_config_hash:
        raise ValueError("CubeLift evaluator and deployment task hashes differ.")
    evaluator = cube_lift_environment_evaluator_spec_v1(
        config=evaluator_config,
        task_evaluation_schema_hash=task_evaluation_schema_hash,
    )
    return SkillEvaluationPlan(
        evaluation_plan_id=CUBE_LIFT_EVALUATION_PLAN_ID_V1,
        evaluation_plan_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=deployment_binding.semantic_content_hash,
        evaluator_specs=(evaluator,),
        aggregation_policy_id=CUBE_LIFT_AGGREGATION_POLICY_ID_V1,
        aggregation_policy_hash=CUBE_LIFT_AGGREGATION_POLICY_HASH_V1,
        required_evaluation_deadline_ms=5_000,
        missing_evidence_policy="fail",
        disagreement_policy="veto",
        task_outcome_schema_hash=_required(
            task_evaluation_schema_hash,
            field_name="task_evaluation_schema_hash",
        ),
    )


def validate_cube_lift_sim_route_v1(
    *,
    route: SkillExecutionRoute,
    schema_entry: AgentSkillSchemaEntry | None = None,
) -> None:
    if not isinstance(route, SkillExecutionRoute):
        raise TypeError("route must be a SkillExecutionRoute.")
    expected = {
        "skill_id": CUBE_LIFT_SKILL_ID_V1,
        "skill_version": "1",
        "route_id": CUBE_LIFT_SIM_ROUTE_ID_V1,
        "route_version": "1",
        "control_effect_class": "motion",
        "privilege_tier": "t1_sim_or_no_motion",
        "supported_modes": ("sim_active",),
        "target_kind": "sim",
        "may_move_robot": True,
        "may_open_publication": True,
        "requires_operator_confirmation": False,
        "requires_deadman": False,
        "max_operation_duration_ms": 30_000,
        "cancellation_policy": "hold_on_cancel_disconnect_or_timeout",
        "max_concurrent_operations": 1,
        "requires_task_evaluation": True,
    }
    for field_name, expected_value in expected.items():
        if getattr(route, field_name) != expected_value:
            raise ValueError(
                f"CubeLift Route requires {field_name}={expected_value!r}."
            )
    if (
        tuple(profile.parameter_profile_id for profile in route.allowed_parameter_profiles)
        != (CUBE_LIFT_PARAMETER_PROFILE_ID_V1,)
    ):
        raise ValueError("CubeLift Route requires the server-bound profile only.")
    if schema_entry is None:
        return
    if route.skill_definition_hash != schema_entry.skill_definition_hash:
        raise ValueError("CubeLift Route definition hash mismatch.")
    for field_name in (
        "parameter_schema_hash",
        "execution_result_schema_hash",
        "task_evaluation_schema_hash",
    ):
        if getattr(route, field_name) != getattr(schema_entry, field_name):
            raise ValueError(f"CubeLift Route {field_name} mismatch.")
    if route.allowed_parameter_profiles != schema_entry.parameter_profiles:
        raise ValueError("CubeLift Route parameter profile mismatch.")


_CUBE_LIFT_EXECUTION_RESULT_FIELDS = frozenset(
    _cube_lift_execution_result_schema_v1()["properties"]
)
_CUBE_LIFT_TASK_EVALUATION_FIELDS = frozenset(
    _cube_lift_task_evaluation_schema_v1()["properties"]
)


def validate_cube_lift_execution_result_v1(
    result: Mapping[str, Any],
) -> None:
    _exact_fields(
        result,
        expected=_CUBE_LIFT_EXECUTION_RESULT_FIELDS,
        field_name="execution_result",
    )
    if result["schema_version"] != CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1:
        raise ValueError("Unsupported CubeLift execution result schema.")
    string_fields = _CUBE_LIFT_EXECUTION_RESULT_FIELDS - {
        "cycle_count",
        "dispatched_cycle_count",
        "measurement_count",
    }
    for field_name in string_fields:
        _required(result[field_name], field_name=field_name)
    cycle_count = _positive_integer(result["cycle_count"], field_name="cycle_count")
    dispatched_count = _positive_integer(
        result["dispatched_cycle_count"],
        field_name="dispatched_cycle_count",
    )
    measurement_count = _positive_integer(
        result["measurement_count"],
        field_name="measurement_count",
    )
    if dispatched_count > cycle_count:
        raise ValueError("dispatched_cycle_count cannot exceed cycle_count.")
    if measurement_count < dispatched_count:
        raise ValueError(
            "CubeLift completion requires measurement coverage for dispatches."
        )


def validate_cube_lift_task_evaluation_v1(
    *,
    evaluation: Mapping[str, Any],
    config: CubeLiftTerminalEvaluatorConfig,
) -> None:
    if not isinstance(config, CubeLiftTerminalEvaluatorConfig):
        raise TypeError("config must be a CubeLiftTerminalEvaluatorConfig.")
    _exact_fields(
        evaluation,
        expected=_CUBE_LIFT_TASK_EVALUATION_FIELDS,
        field_name="task_evaluation",
    )
    if evaluation["schema_version"] != CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1:
        raise ValueError("Unsupported CubeLift task evaluation schema.")
    verdict = _required(evaluation["verdict"], field_name="verdict")
    if verdict not in {"failed", "inconclusive", "succeeded"}:
        raise ValueError(f"Unsupported verdict: {verdict!r}.")
    reason_codes = evaluation["reason_codes"]
    if (
        not isinstance(reason_codes, (tuple, list))
        or not reason_codes
        or any(not isinstance(item, str) or not item for item in reason_codes)
        or len(set(reason_codes)) != len(reason_codes)
    ):
        raise ValueError("reason_codes must be a non-empty unique string list.")
    for field_name in (
        "evaluated_cut_id",
        "environment_evidence_hash",
        "evaluator_configuration_hash",
        "terminal_state_hash",
    ):
        _required(evaluation[field_name], field_name=field_name)
    if (
        evaluation["evaluator_configuration_hash"]
        != config.semantic_content_hash
    ):
        raise ValueError("Task evaluation binds another evaluator config.")
    baseline = _finite(
        evaluation["baseline_resting_center_z_m"],
        field_name="baseline_resting_center_z_m",
    )
    required_delta = _finite(
        evaluation["required_lift_delta_m"],
        field_name="required_lift_delta_m",
    )
    if abs(required_delta - config.lift_delta_m) > 1e-12:
        raise ValueError("Task evaluation lift delta does not match config.")
    minimum_z = _finite(
        evaluation["minimum_verified_cube_center_z_m"],
        field_name="minimum_verified_cube_center_z_m",
    )
    _finite(
        evaluation["final_cube_center_z_m"],
        field_name="final_cube_center_z_m",
    )
    required_ticks = _positive_integer(
        evaluation["stable_hold_required_physics_ticks"],
        field_name="stable_hold_required_physics_ticks",
    )
    if required_ticks != config.stable_hold_physics_ticks:
        raise ValueError("Task evaluation stable window does not match config.")
    observed_ticks = _nonnegative_integer(
        evaluation["stable_hold_observed_physics_ticks"],
        field_name="stable_hold_observed_physics_ticks",
    )
    booleans = {
        name: _boolean(evaluation[name], field_name=name)
        for name in (
            "finite_state_all_ticks",
            "bilateral_contact_all_ticks",
            "table_contact_any_tick",
            "terminal_hold_confirmed",
            "execution_lineage_complete",
        )
    }
    if verdict != "succeeded":
        return
    success_checks = (
        observed_ticks >= required_ticks,
        minimum_z >= baseline + required_delta,
        booleans["finite_state_all_ticks"],
        (
            booleans["bilateral_contact_all_ticks"]
            or not config.success_requires_bilateral_contact
        ),
        (
            not booleans["table_contact_any_tick"]
            or not config.requires_no_table_contact
        ),
        booleans["terminal_hold_confirmed"],
        booleans["execution_lineage_complete"],
    )
    if not all(success_checks):
        raise ValueError(
            "CubeLift success requires complete execution lineage and a "
            "measured stable terminal lift under lower HOLD."
        )


def cube_lift_payload_validator_set_v1(
    *,
    schema_entry: AgentSkillSchemaEntry,
    evaluator_config: CubeLiftTerminalEvaluatorConfig,
) -> AgentSkillPayloadValidatorSet:
    """Bind executable validation to the exact CubeLift schema hashes."""

    if not isinstance(schema_entry, AgentSkillSchemaEntry):
        raise TypeError("schema_entry must be an AgentSkillSchemaEntry.")
    if schema_entry.skill_id != CUBE_LIFT_SKILL_ID_V1:
        raise ValueError("CubeLift validators require the CubeLift schema entry.")
    if schema_entry.skill_version != "1":
        raise ValueError("CubeLift validators require schema version 1.")
    if not isinstance(evaluator_config, CubeLiftTerminalEvaluatorConfig):
        raise TypeError(
            "evaluator_config must be a CubeLiftTerminalEvaluatorConfig."
        )

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        validate_cube_lift_task_evaluation_v1(
            evaluation=payload,
            config=evaluator_config,
        )

    return AgentSkillPayloadValidatorSet(
        validator_id="dexteleop.agent_skill.cube_lift_payload_validator.v1",
        skill_id=CUBE_LIFT_SKILL_ID_V1,
        skill_version="1",
        route_id=CUBE_LIFT_SIM_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        validate_parameters=validate_cube_lift_parameters_v1,
        validate_execution_result=validate_cube_lift_execution_result_v1,
        validate_task_evaluation=validate_task_evaluation,
    )


__all__ = [
    "CUBE_LIFT_AGGREGATION_POLICY_HASH_V1",
    "CUBE_LIFT_AGGREGATION_POLICY_ID_V1",
    "CUBE_LIFT_DEPLOYMENT_BINDING_HASH_DOMAIN_V1",
    "CUBE_LIFT_ENVIRONMENT_EVALUATOR_ID_V1",
    "CUBE_LIFT_EVALUATION_PLAN_ID_V1",
    "CUBE_LIFT_EVALUATOR_EVIDENCE_SCHEMA_V1",
    "CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1",
    "CUBE_LIFT_PARAMETER_PROFILE_ID_V1",
    "CUBE_LIFT_SIM_ROUTE_ID_V1",
    "CUBE_LIFT_SKILL_ID_V1",
    "CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1",
    "CubeLiftSimDeploymentBinding",
    "CubeLiftTerminalEvaluatorConfig",
    "cube_lift_environment_evaluator_spec_v1",
    "cube_lift_evaluation_plan_v1",
    "cube_lift_evaluator_input_schema_hash_v1",
    "cube_lift_evaluator_input_schema_v1",
    "cube_lift_payload_validator_set_v1",
    "cube_lift_schema_entry_v1",
    "cube_lift_sim_route_v1",
    "cube_lift_skill_definition_v1",
    "validate_cube_lift_execution_result_v1",
    "validate_cube_lift_parameters_v1",
    "validate_cube_lift_sim_route_v1",
    "validate_cube_lift_task_evaluation_v1",
]

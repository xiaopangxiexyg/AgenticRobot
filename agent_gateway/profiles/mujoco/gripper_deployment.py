"""MuJoCo deployment contracts for passive and fixed-endpoint gripper Skills."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from agent_gateway.protocol.evaluation_contracts import (
    SkillEvaluationPlan,
    SkillEvaluatorSpec,
)
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.motion.definition import (
    CLOSE_GRIPPER_SKILL_ID_V1,
    OPEN_GRIPPER_SKILL_ID_V1,
    close_gripper_skill_definition_v1,
    open_gripper_skill_definition_v1,
)
from agent_gateway.skills.motion.gripper_schemas import (
    gripper_endpoint_schema_entry_v1,
    validate_gripper_endpoint_parameters_v1,
    validate_gripper_endpoint_result_v1,
    validate_gripper_endpoint_task_evaluation_v1,
)
from agent_gateway.skills.state.definition import read_gripper_state_definition_v1
from agent_gateway.skills.state.gripper_schemas import (
    read_gripper_state_schema_entry_v1,
    validate_no_task_evaluation_v1,
    validate_read_gripper_state_parameters_v1,
    validate_read_gripper_state_result_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

MUJOCO_OPEN_GRIPPER_ROUTE_ID_V1 = "sim.mujoco.motion.open_gripper"
MUJOCO_CLOSE_GRIPPER_ROUTE_ID_V1 = "sim.mujoco.motion.close_gripper"
MUJOCO_READ_GRIPPER_STATE_ROUTE_ID_V1 = "sim.mujoco.state.read_gripper_state"
MUJOCO_OPEN_GRIPPER_EXECUTOR_ID_V1 = "mujoco-open-gripper.v1"
MUJOCO_CLOSE_GRIPPER_EXECUTOR_ID_V1 = "mujoco-close-gripper.v1"
MUJOCO_READ_GRIPPER_STATE_EXECUTOR_ID_V1 = "mujoco-read-gripper-state.v1"


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


@dataclass(frozen=True)
class MujocoGripperProfile:
    gripper_name: str
    clock_domain_id: str
    component_id: str = "right_gripper"
    maximum_measurement_age_ms: int = 500
    open_endpoint_tolerance: float = 0.05
    settled_close_delta_tolerance: float = 0.005
    stable_measurement_count: int = 3
    max_control_cycles: int = 160
    physics_ticks_per_control_cycle: int = 6
    profile_id: str = "mujoco-fixed-right-gripper-endpoints-v1"

    def __post_init__(self) -> None:
        for name in ("gripper_name", "clock_domain_id", "component_id", "profile_id"):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        for name in (
            "maximum_measurement_age_ms",
            "stable_measurement_count",
            "max_control_cycles",
            "physics_ticks_per_control_cycle",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        for name in (
            "open_endpoint_tolerance",
            "settled_close_delta_tolerance",
        ):
            tolerance = float(getattr(self, name))
            if not 0.0 < tolerance < 1.0:
                raise ValueError(f"{name} must lie in (0, 1).")
            object.__setattr__(self, name, tolerance)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return (self.component_id,)

    @property
    def observation_binding_id(self) -> str:
        return f"sim.mujoco.measured_gripper:{self.gripper_name}"

    @property
    def observation_binding_hash(self) -> str:
        return canonical_content_hash(
            {
                "observation_binding_id": self.observation_binding_id,
                "component_id": self.component_id,
                "gripper_name": self.gripper_name,
                "clock_domain_id": self.clock_domain_id,
                "measurement": "mujoco_joint_qpos_to_normalized_close",
            },
            domain="dexteleop.agent_state.mujoco_gripper_observation_binding.v1",
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                name: getattr(self, name) for name in self.__dataclass_fields__
            }
            | {
                "command_surface": "fixed_open_or_close_only",
                "close_success": "settled_not_grasp_success",
            },
            domain="dexteleop.agent_skill.mujoco_gripper_profile.v1",
        )


@dataclass(frozen=True)
class MujocoGripperDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str
    binding_kind: str
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...]

    def __post_init__(self) -> None:
        for name in (
            "deployment_binding_id",
            "route_hash",
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "profile_hash",
            "executor_id",
            "binding_kind",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if self.binding_kind not in {"action", "state"}:
            raise ValueError("binding_kind must be action or state.")
        refs = tuple(self.safety_envelope_refs)
        if any(
            not isinstance(item, SkillDeploymentSafetyEnvelopeRef)
            for item in refs
        ):
            raise TypeError("safety_envelope_refs must contain typed refs.")
        if self.binding_kind == "action" and len(refs) != 1:
            raise ValueError("A gripper action requires one arm-hold envelope.")
        if self.binding_kind == "state" and refs:
            raise ValueError("A passive gripper-state binding has no envelope.")
        object.__setattr__(self, "safety_envelope_refs", refs)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "safety_envelope_refs"
            }
            | {
                "safety_envelope_refs": tuple(
                    item.to_wire() for item in self.safety_envelope_refs
                ),
                "target_kind": "sim",
            },
            domain="dexteleop.agent_skill.mujoco_gripper_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="sim",
            execution_manifest_hash=self.execution_manifest_hash,
            safety_envelope_refs=self.safety_envelope_refs,
        )


@dataclass(frozen=True)
class MujocoGripperActionRegistration:
    profile: MujocoGripperProfile
    skill: SkillRegistrationTemplate
    binding: MujocoGripperDeploymentBinding
    evaluation_plan: SkillEvaluationPlan
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


@dataclass(frozen=True)
class MujocoGripperStateRegistration:
    profile: MujocoGripperProfile
    skill: SkillRegistrationTemplate
    binding: MujocoGripperDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def _action_registration(
    *,
    skill_id: str,
    profile: MujocoGripperProfile,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
) -> MujocoGripperActionRegistration:
    if skill_id == OPEN_GRIPPER_SKILL_ID_V1:
        definition = open_gripper_skill_definition_v1(
            controlled_component_ids=profile.component_ids
        )
        route_id = MUJOCO_OPEN_GRIPPER_ROUTE_ID_V1
        executor_id = MUJOCO_OPEN_GRIPPER_EXECUTOR_ID_V1
    elif skill_id == CLOSE_GRIPPER_SKILL_ID_V1:
        definition = close_gripper_skill_definition_v1(
            controlled_component_ids=profile.component_ids
        )
        route_id = MUJOCO_CLOSE_GRIPPER_ROUTE_ID_V1
        executor_id = MUJOCO_CLOSE_GRIPPER_EXECUTOR_ID_V1
    else:
        raise ValueError("Unsupported MuJoCo gripper endpoint Skill.")
    schema = gripper_endpoint_schema_entry_v1(
        skill_id=skill_id,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=profile.component_ids,
    )
    io_contract = inline_only_skill_io_contract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=10_000,
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=True,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    skill = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = MujocoGripperDeploymentBinding(
        deployment_binding_id=f"sim.mujoco.gripper_action:{skill_id}:{runtime_session_id}",
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        executor_id=executor_id,
        binding_kind="action",
        safety_envelope_refs=safety_envelope_refs,
    )
    evaluator = SkillEvaluatorSpec(
        evaluator_id=(
            "motion.gripper_open_endpoint_rule"
            if skill_id == OPEN_GRIPPER_SKILL_ID_V1
            else "motion.gripper_close_settled_rule"
        ),
        evaluator_version="1",
        evaluator_kind="rule_based",
        input_evidence_schema_hash=canonical_content_hash(
            {"input": "measured_gripper_state_after_lower_hold"},
            domain="dexteleop.agent_skill.mujoco_gripper_evaluator_input.v1",
        ),
        output_schema_hash=schema.task_evaluation_schema_hash,
        role="required",
        execution_location="gateway_worker",
        deadline_ms=1_000,
        failure_policy="veto",
        evidence_domain_id=control_domain_id,
        configuration_hash=profile.semantic_content_hash,
    )
    aggregation_hash = canonical_content_hash(
        {"policy": "single_required_rule_based_evaluator"},
        domain="dexteleop.agent_skill.gripper_aggregation.v1",
    )
    evaluation_plan = SkillEvaluationPlan(
        evaluation_plan_id=evaluator.evaluator_id,
        evaluation_plan_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        evaluator_specs=(evaluator,),
        aggregation_policy_id="motion.single_required_rule.v1",
        aggregation_policy_hash=aggregation_hash,
        required_evaluation_deadline_ms=1_000,
        missing_evidence_policy="fail",
        disagreement_policy="aggregation_policy",
        task_outcome_schema_hash=schema.task_evaluation_schema_hash,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=f"payload-validator:mujoco-gripper-action:{skill_id}:{runtime_session_id}",
        skill_id=skill_id,
        skill_version="1",
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_gripper_endpoint_parameters_v1(
            payload, allowed_component_ids=profile.component_ids
        ),
        validate_execution_result=lambda payload: validate_gripper_endpoint_result_v1(
            payload, skill_id=skill_id
        ),
        validate_execution_result_for_parameters=lambda parameters, payload: (
            validate_gripper_endpoint_result_v1(
                payload,
                skill_id=skill_id,
                request_parameters=parameters,
            )
        ),
        validate_task_evaluation=lambda payload: (
            validate_gripper_endpoint_task_evaluation_v1(
                payload, skill_id=skill_id
            )
        ),
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_mujoco_runtime_ready",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.mujoco_gripper_action_availability.v1",
    )
    return MujocoGripperActionRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        evaluation_plan=evaluation_plan,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


def build_mujoco_gripper_registrations(
    *,
    gripper_name: str,
    clock_domain_id: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    safety_envelope_refs_by_route_id: Mapping[
        str, tuple[SkillDeploymentSafetyEnvelopeRef, ...]
    ],
    physics_ticks_per_control_cycle: int,
) -> tuple[
    MujocoGripperActionRegistration,
    MujocoGripperActionRegistration,
    MujocoGripperStateRegistration,
]:
    profile = MujocoGripperProfile(
        gripper_name=gripper_name,
        clock_domain_id=clock_domain_id,
        physics_ticks_per_control_cycle=physics_ticks_per_control_cycle,
    )
    common = {
        "profile": profile,
        "runtime_session_id": runtime_session_id,
        "control_domain_id": control_domain_id,
        "target_binding_id": target_binding_id,
        "station_manifest_hash": station_manifest_hash,
        "execution_manifest_hash": execution_manifest_hash,
    }
    open_registration = _action_registration(
        skill_id=OPEN_GRIPPER_SKILL_ID_V1,
        safety_envelope_refs=safety_envelope_refs_by_route_id[
            MUJOCO_OPEN_GRIPPER_ROUTE_ID_V1
        ],
        **common,
    )
    close_registration = _action_registration(
        skill_id=CLOSE_GRIPPER_SKILL_ID_V1,
        safety_envelope_refs=safety_envelope_refs_by_route_id[
            MUJOCO_CLOSE_GRIPPER_ROUTE_ID_V1
        ],
        **common,
    )
    definition = read_gripper_state_definition_v1(
        controlled_component_ids=profile.component_ids
    )
    schema = read_gripper_state_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id + ":state",
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=profile.component_ids,
        maximum_measurement_age_ms=profile.maximum_measurement_age_ms,
    )
    io_contract = inline_only_skill_io_contract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=MUJOCO_READ_GRIPPER_STATE_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    skill = SkillRegistrationTemplate(
        definition=definition,
        schema_entry=schema,
        io_contract=io_contract,
        routes=(route,),
    )
    binding = MujocoGripperDeploymentBinding(
        deployment_binding_id=f"sim.mujoco.gripper_state:{runtime_session_id}",
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        executor_id=MUJOCO_READ_GRIPPER_STATE_EXECUTOR_ID_V1,
        binding_kind="state",
        safety_envelope_refs=(),
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=f"payload-validator:mujoco-gripper-state:{runtime_session_id}",
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_read_gripper_state_parameters_v1(
            payload,
            allowed_component_ids=profile.component_ids,
            maximum_measurement_age_ms=profile.maximum_measurement_age_ms,
        ),
        validate_execution_result=lambda payload: validate_read_gripper_state_result_v1(
            payload,
            request_parameters={
                "component_ids": list(payload.get("component_ids", ())),
                "max_measurement_age_ms": profile.maximum_measurement_age_ms,
            },
            observation_binding_id=profile.observation_binding_id,
            observation_binding_hash=profile.observation_binding_hash,
            clock_domain_id=profile.clock_domain_id,
        ),
        validate_execution_result_for_parameters=lambda parameters, payload: (
            validate_read_gripper_state_result_v1(
                payload,
                request_parameters=parameters,
                observation_binding_id=profile.observation_binding_id,
                observation_binding_hash=profile.observation_binding_hash,
                clock_domain_id=profile.clock_domain_id,
            )
        ),
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    state_registration = MujocoGripperStateRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=canonical_content_hash(
            {
                "status": "configured_passive_mujoco_measurement",
                "route_hash": route.semantic_content_hash,
                "deployment_binding_hash": binding.semantic_content_hash,
            },
            domain="dexteleop.agent_state.mujoco_gripper_availability.v1",
        ),
    )
    return open_registration, close_registration, state_registration


__all__ = [
    "MUJOCO_CLOSE_GRIPPER_ROUTE_ID_V1",
    "MUJOCO_OPEN_GRIPPER_ROUTE_ID_V1",
    "MUJOCO_READ_GRIPPER_STATE_ROUTE_ID_V1",
    "MujocoGripperActionRegistration",
    "MujocoGripperProfile",
    "MujocoGripperStateRegistration",
    "build_mujoco_gripper_registrations",
]

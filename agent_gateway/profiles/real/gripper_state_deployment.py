"""Real read-only deployment contract for measured gripper state."""

from __future__ import annotations

from dataclasses import dataclass

from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOContract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.state.definition import read_gripper_state_definition_v1
from agent_gateway.skills.state.gripper_schemas import (
    read_gripper_state_schema_entry_v1,
    validate_no_task_evaluation_v1,
    validate_read_gripper_state_parameters_v1,
    validate_read_gripper_state_result_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

REAL_GRIPPER_STATE_ROUTE_ID_V1 = "real.passive_gripper_state"
REAL_GRIPPER_STATE_EXECUTOR_ID_V1 = "real-passive-gripper-state-reader.v1"


def _required(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty trimmed string.")
    return value


def _component_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(values)
    if (
        not normalized
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in normalized
        )
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("component_ids must contain unique non-empty strings.")
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class RealGripperStateProfile:
    component_ids: tuple[str, ...]
    observation_binding_id: str
    observation_binding_hash: str
    clock_domain_id: str
    maximum_measurement_age_ms: int = 200
    profile_id: str = "real-measured-gripper-state-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", _component_ids(self.component_ids))
        for name in (
            "observation_binding_id",
            "observation_binding_hash",
            "clock_domain_id",
            "profile_id",
        ):
            object.__setattr__(self, name, _required(getattr(self, name), name=name))
        if (
            isinstance(self.maximum_measurement_age_ms, bool)
            or not isinstance(self.maximum_measurement_age_ms, int)
            or self.maximum_measurement_age_ms <= 0
        ):
            raise ValueError("maximum_measurement_age_ms must be positive.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "component_ids": self.component_ids,
                "observation_binding_id": self.observation_binding_id,
                "observation_binding_hash": self.observation_binding_hash,
                "clock_domain_id": self.clock_domain_id,
                "maximum_measurement_age_ms": self.maximum_measurement_age_ms,
                "close_fraction_semantics": "0_open_1_closed",
                "source_semantics": "lower_owner_published_measured_state_only",
            },
            domain="dexteleop.agent_skill.real_gripper_state_profile.v1",
        )


@dataclass(frozen=True)
class RealGripperStateDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str = REAL_GRIPPER_STATE_EXECUTOR_ID_V1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _required(getattr(self, name), name=name))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__
            | {
                "target_kind": "real",
                "control_effect_class": "read_only",
                "hardware_refresh_from_skill": "forbidden",
            },
            domain="dexteleop.agent_skill.real_gripper_state_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class RealGripperStateRegistration:
    profile: RealGripperStateProfile
    skill: SkillRegistrationTemplate
    binding: RealGripperStateDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def build_real_gripper_state_registration(
    *,
    component_ids: tuple[str, ...],
    observation_binding_id: str,
    observation_binding_hash: str,
    clock_domain_id: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    maximum_measurement_age_ms: int = 200,
    current_mode: str = "real_active",
) -> RealGripperStateRegistration:
    if current_mode not in {"real_observe", "real_active"}:
        raise ValueError("gripper-state current_mode is unsupported.")
    profile = RealGripperStateProfile(
        component_ids=component_ids,
        observation_binding_id=observation_binding_id,
        observation_binding_hash=observation_binding_hash,
        clock_domain_id=clock_domain_id,
        maximum_measurement_age_ms=maximum_measurement_age_ms,
    )
    definition = read_gripper_state_definition_v1(
        controlled_component_ids=profile.component_ids
    )
    schema = read_gripper_state_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=profile.component_ids,
        maximum_measurement_age_ms=profile.maximum_measurement_age_ms,
    )
    io_contract = SkillIOContract(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        input_artifact_slots=(),
        output_artifact_slots=(),
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=REAL_GRIPPER_STATE_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=(current_mode,),
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=2_000,
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
    binding = RealGripperStateDeploymentBinding(
        deployment_binding_id="real.gripper_state:" + runtime_session_id,
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id="dexteleop.agent_skill.validator.real_gripper_state:"
        + runtime_session_id,
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
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_lower_owner_published_measurement",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.real_gripper_state_availability.v1",
    )
    return RealGripperStateRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


__all__ = [
    "REAL_GRIPPER_STATE_EXECUTOR_ID_V1",
    "REAL_GRIPPER_STATE_ROUTE_ID_V1",
    "RealGripperStateDeploymentBinding",
    "RealGripperStateProfile",
    "RealGripperStateRegistration",
    "build_real_gripper_state_registration",
]

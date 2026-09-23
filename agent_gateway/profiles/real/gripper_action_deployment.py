"""Real deployment contracts for fixed open/close gripper Skills."""

from __future__ import annotations

from dataclasses import dataclass

from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.skill_io_contracts import SkillIOContract
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
from dataset_contracts.canonical_encoding import canonical_content_hash

REAL_OPEN_GRIPPER_ROUTE_ID_V1 = "real.open_gripper"
REAL_CLOSE_GRIPPER_ROUTE_ID_V1 = "real.close_gripper"
REAL_OPEN_GRIPPER_EXECUTOR_ID_V1 = "real-open-gripper.v1"
REAL_CLOSE_GRIPPER_EXECUTOR_ID_V1 = "real-close-gripper.v1"


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
class RealGripperActionProfile:
    component_ids: tuple[str, ...]
    clock_domain_id: str
    maximum_measurement_age_ms: int = 250
    poll_interval_ms: int = 20
    stable_measurement_count: int = 2
    open_endpoint_tolerance: float = 0.05
    settled_close_delta_tolerance: float = 0.005
    profile_id: str = "real-fixed-gripper-endpoints-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_ids", _component_ids(self.component_ids))
        object.__setattr__(
            self,
            "clock_domain_id",
            _required(self.clock_domain_id, name="clock_domain_id"),
        )
        object.__setattr__(
            self, "profile_id", _required(self.profile_id, name="profile_id")
        )
        for name in (
            "maximum_measurement_age_ms",
            "poll_interval_ms",
            "stable_measurement_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        tolerance = float(self.open_endpoint_tolerance)
        if not 0.0 <= tolerance < 1.0:
            raise ValueError("open_endpoint_tolerance must lie in [0, 1).")
        object.__setattr__(self, "open_endpoint_tolerance", tolerance)
        settled_tolerance = float(self.settled_close_delta_tolerance)
        if not 0.0 < settled_tolerance < 1.0:
            raise ValueError(
                "settled_close_delta_tolerance must lie in (0, 1)."
            )
        object.__setattr__(
            self,
            "settled_close_delta_tolerance",
            settled_tolerance,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": self.profile_id,
                "component_ids": self.component_ids,
                "clock_domain_id": self.clock_domain_id,
                "maximum_measurement_age_ms": self.maximum_measurement_age_ms,
                "poll_interval_ms": self.poll_interval_ms,
                "stable_measurement_count": self.stable_measurement_count,
                "open_endpoint_tolerance": self.open_endpoint_tolerance,
                "settled_close_delta_tolerance": (
                    self.settled_close_delta_tolerance
                ),
                "close_fraction_semantics": "0_open_1_closed",
                "speed_and_force_semantics": "deployment_owned_not_agent_selectable",
            },
            domain="dexteleop.agent_skill.real_gripper_action_profile.v1",
        )


@dataclass(frozen=True)
class RealGripperActionDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    profile_hash: str
    executor_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _required(getattr(self, name), name=name))

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__
            | {
                "target_kind": "real",
                "control_effect_class": "motion",
                "dispatch_owner": "injected_lower_gripper_port",
            },
            domain="dexteleop.agent_skill.real_gripper_action_deployment.v1",
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
class RealGripperActionRegistration:
    profile: RealGripperActionProfile
    skill: SkillRegistrationTemplate
    binding: RealGripperActionDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str


def _build_registration(
    *,
    skill_id: str,
    profile: RealGripperActionProfile,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
) -> RealGripperActionRegistration:
    if skill_id == OPEN_GRIPPER_SKILL_ID_V1:
        definition = open_gripper_skill_definition_v1(
            controlled_component_ids=profile.component_ids
        )
        route_id = REAL_OPEN_GRIPPER_ROUTE_ID_V1
        executor_id = REAL_OPEN_GRIPPER_EXECUTOR_ID_V1
    elif skill_id == CLOSE_GRIPPER_SKILL_ID_V1:
        definition = close_gripper_skill_definition_v1(
            controlled_component_ids=profile.component_ids
        )
        route_id = REAL_CLOSE_GRIPPER_ROUTE_ID_V1
        executor_id = REAL_CLOSE_GRIPPER_EXECUTOR_ID_V1
    else:
        raise ValueError("Unsupported real gripper action Skill.")
    schema = gripper_endpoint_schema_entry_v1(
        skill_id=skill_id,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=profile.component_ids,
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
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t2_real_motion",
        supported_modes=("real_active",),
        target_kind="real",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=True,
        requires_deadman=False,
        max_operation_duration_ms=5_000,
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
    binding = RealGripperActionDeploymentBinding(
        deployment_binding_id=f"{route_id}:{runtime_session_id}",
        route_hash=route.semantic_content_hash,
        runtime_session_id=runtime_session_id,
        control_domain_id=control_domain_id,
        target_binding_id=target_binding_id,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        profile_hash=profile.semantic_content_hash,
        executor_id=executor_id,
    )
    validator = AgentSkillPayloadValidatorSet(
        validator_id=f"dexteleop.agent_skill.validator.{route_id}:{runtime_session_id}",
        skill_id=skill_id,
        skill_version="1",
        route_id=route_id,
        route_version="1",
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=lambda payload: validate_gripper_endpoint_parameters_v1(
            payload,
            allowed_component_ids=profile.component_ids,
        ),
        validate_execution_result=lambda payload: validate_gripper_endpoint_result_v1(
            payload,
            skill_id=skill_id,
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
                payload,
                skill_id=skill_id,
            )
        ),
    )
    availability_hash = canonical_content_hash(
        {
            "status": "configured_requires_exclusive_lower_gripper_owner",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.real_gripper_action_availability.v1",
    )
    return RealGripperActionRegistration(
        profile=profile,
        skill=skill,
        binding=binding,
        payload_validator=validator,
        availability_hash=availability_hash,
    )


def build_real_gripper_action_registrations(
    *,
    component_ids: tuple[str, ...],
    clock_domain_id: str,
    runtime_session_id: str,
    control_domain_id: str,
    target_binding_id: str,
    station_manifest_hash: str,
    execution_manifest_hash: str,
    maximum_measurement_age_ms: int = 250,
    poll_interval_ms: int = 20,
    stable_measurement_count: int = 2,
    open_endpoint_tolerance: float = 0.05,
    settled_close_delta_tolerance: float = 0.005,
) -> tuple[RealGripperActionRegistration, RealGripperActionRegistration]:
    """Build the paired registrations over one immutable deployment profile."""

    profile = RealGripperActionProfile(
        component_ids=component_ids,
        clock_domain_id=clock_domain_id,
        maximum_measurement_age_ms=maximum_measurement_age_ms,
        poll_interval_ms=poll_interval_ms,
        stable_measurement_count=stable_measurement_count,
        open_endpoint_tolerance=open_endpoint_tolerance,
        settled_close_delta_tolerance=settled_close_delta_tolerance,
    )
    common = {
        "profile": profile,
        "runtime_session_id": _required(runtime_session_id, name="runtime_session_id"),
        "control_domain_id": _required(control_domain_id, name="control_domain_id"),
        "target_binding_id": _required(target_binding_id, name="target_binding_id"),
        "station_manifest_hash": _required(
            station_manifest_hash,
            name="station_manifest_hash",
        ),
        "execution_manifest_hash": _required(
            execution_manifest_hash,
            name="execution_manifest_hash",
        ),
    }
    return (
        _build_registration(skill_id=OPEN_GRIPPER_SKILL_ID_V1, **common),
        _build_registration(skill_id=CLOSE_GRIPPER_SKILL_ID_V1, **common),
    )


__all__ = [
    "REAL_CLOSE_GRIPPER_EXECUTOR_ID_V1",
    "REAL_CLOSE_GRIPPER_ROUTE_ID_V1",
    "REAL_OPEN_GRIPPER_EXECUTOR_ID_V1",
    "REAL_OPEN_GRIPPER_ROUTE_ID_V1",
    "RealGripperActionDeploymentBinding",
    "RealGripperActionProfile",
    "RealGripperActionRegistration",
    "build_real_gripper_action_registrations",
]

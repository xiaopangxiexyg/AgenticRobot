"""Bound passive object feedback over a producer-owned measurement port."""

from collections.abc import Mapping
from dataclasses import dataclass, replace

from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import PassiveReadResult, PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.state.object_state import (
    IDENTITY_FIELDS, SKILL_ID, object_state_definition, object_state_schema,
    validate_object_state_parameters, validate_object_state_result,
    validate_object_state_result_for_parameters,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


@dataclass(frozen=True)
class ObjectStateProfile:
    object_id: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    clock_domain_id: str
    observation_source: str
    source_binding_hash: str
    release_supported: bool = False

    def __post_init__(self):
        for name in IDENTITY_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError("Object state requires explicit source/frame/clock identities")
        if type(self.release_supported) is not bool:
            raise ValueError("Release capability must be explicit")

    @property
    def identity(self):
        return {key: getattr(self, key) for key in IDENTITY_FIELDS}

    @property
    def semantic_content_hash(self):
        return canonical_content_hash({**self.identity, "release_supported": self.release_supported},
            domain="agenticrobot.bound_object_state.profile.v1")


@dataclass(frozen=True)
class ObjectStateRegistration:
    profile: ObjectStateProfile
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet


def build_object_state_registration(*, base_entry, profile):
    route = base_entry.route
    if (not isinstance(profile, ObjectStateProfile) or route.control_effect_class != "read_only"
            or route.privilege_tier != "t0_read_only" or route.cancellation_policy != "cancel_only"
            or route.may_move_robot or route.may_open_publication
            or route.requires_task_evaluation or base_entry.evaluation_plan_hash is not None):
        raise ValueError("Object state requires a strictly passive base registration")
    definition = object_state_definition()
    schema = object_state_schema(definition_hash=definition.semantic_content_hash, identity=profile.identity,
        profile_hash=profile.semantic_content_hash, release_supported=profile.release_supported)
    io = inline_only_skill_io_contract(skill_id=SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash)
    route = replace(route, skill_id=SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash, route_id=route.route_id + ".object_state",
        route_version="1", parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles, skill_io_contract_hash=io.semantic_content_hash)
    binding = replace(base_entry.deployment_binding,
        deployment_binding_id="object-state:" + base_entry.deployment_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash({"base": base_entry.deployment_binding.deployment_binding_hash,
            "route": route.semantic_content_hash, "profile": profile.semantic_content_hash},
            domain="agenticrobot.bound_object_state.binding.v1"))

    def parameters(value):
        validate_object_state_parameters(value, identity=profile.identity)

    def result(value):
        validate_object_state_result(value, identity=profile.identity,
            fields=schema.execution_result_schema["properties"], release_supported=profile.release_supported)

    def result_for(params, value):
        parameters(params)
        result(value)
        validate_object_state_result_for_parameters(params, value)

    validator = AgentSkillPayloadValidatorSet(
        validator_id="object-state:" + profile.semantic_content_hash,
        skill_id=SKILL_ID, skill_version="1", route_id=route.route_id, route_version=route.route_version,
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=parameters, validate_execution_result=result,
        validate_execution_result_for_parameters=result_for, validate_task_evaluation=validate_no_task_evaluation_v1)
    return ObjectStateRegistration(profile, SkillRegistrationTemplate(definition, schema, io, (route,)),
        SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
            availability_hash=canonical_content_hash({"profile": profile.semantic_content_hash,
                "route": route.semantic_content_hash}, domain="agenticrobot.object_state.availability.v1")), validator)


def build_object_state_executor(*, registration, capture, current_station_revision, clock,
                                max_retained_operations):
    if not callable(capture):
        raise TypeError("Object state requires a bound measurement reader")

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        measured = capture()
        if not isinstance(measured, Mapping):
            raise TypeError("Object-state producer must return measured fields")
        result = {**registration.profile.identity, **measured}
        registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
        digest = canonical_content_hash(result, domain="agenticrobot.object_state.measurement.v1")
        return PassiveReadResult(execution_result=result, evidence_refs=("object-state-measurement:" + digest,),
            reason_code="passive_object_state_read_completed")

    return PassiveReadSkillExecutor(port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.bound-object-state.v1", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations)

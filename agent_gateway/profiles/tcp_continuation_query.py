"""Simulation-neutral registration over an injected passive two-stage query port."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
from agent_gateway.protocol._contract_validation import required_text
from agent_gateway.protocol.skill_catalog import (
    SkillExecutionRoute,
    SkillRouteManifestEntry,
)
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.perception.tcp_continuation_query import (
    SKILL_ID,
    continuation_query_definition,
    continuation_query_schema,
    validate_continuation_parameters,
    validate_continuation_result,
    validate_continuation_result_for_parameters,
)
from dataset_contracts.canonical_encoding import canonical_content_hash


def build_tcp_continuation_query_registration(
    *, base_route: SkillExecutionRoute, base_binding: Any, component_id: str,
    reference_frame_id: str, planning_profile_hash: str,
) -> SpatialGoalRegistration:
    """Bind an explicit component/frame without constructing a target or planner."""
    if base_route.control_effect_class != "read_only" or base_route.target_kind != "sim":
        raise ValueError("TCP continuation queries require a passive simulation base route.")
    required_text(component_id, field_name="component_id")
    required_text(reference_frame_id, field_name="reference_frame_id")
    required_text(planning_profile_hash, field_name="planning_profile_hash")
    identity = canonical_content_hash(
        {"component_id": component_id, "reference_frame_id": reference_frame_id,
         "planning_profile_hash": planning_profile_hash},
        domain="agenticrobot.tcp_continuation_query.profile.v1",
    )
    definition = continuation_query_definition()
    schema = continuation_query_schema(definition_hash=definition.semantic_content_hash,
        profile_hash=identity, reference_frame_id=reference_frame_id, component_id=component_id)
    io = inline_only_skill_io_contract(
        skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = replace(
        base_route, skill_id=SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_id=base_route.route_id + ".tcp_continuation_query", route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io.semantic_content_hash, max_operation_duration_ms=40_000,
    )
    binding = base_binding.to_ref() if hasattr(base_binding, "to_ref") else base_binding
    binding = replace(binding,
        deployment_binding_id="tcp-continuation-query:" + binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.tcp_continuation_query.binding.v1",
        ),
    )

    def bound_identity(value):
        if value["component_id"] != component_id or value["reference_frame_id"] != reference_frame_id:
            raise ValueError("Continuation component/frame differs from the deployment binding.")

    def parameters(value):
        validate_continuation_parameters(value)
        bound_identity(value)

    def result(value):
        validate_continuation_result(value)
        bound_identity(value)

    def result_for(value, output):
        parameters(value)
        result(output)
        validate_continuation_result_for_parameters(value, output)

    validator = AgentSkillPayloadValidatorSet(
        validator_id="tcp-continuation-query:" + identity,
        skill_id=SKILL_ID, skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=parameters, validate_execution_result=result,
        validate_execution_result_for_parameters=result_for,
        validate_task_evaluation=validate_no_task_evaluation_v1,
    )
    return SpatialGoalRegistration(
        SkillRegistrationTemplate(definition, schema, io, (route,)),
        SkillRouteManifestEntry(route=route, deployment_binding=binding,
            evaluation_plan_hash=None, availability_hash=identity),
        validator,
    )


def build_tcp_continuation_query_executor(
    *, registration: SpatialGoalRegistration, query_port: Any,
    current_station_revision: Callable[[], int], clock: Callable[[], float], max_retained_operations: int,
) -> PassiveReadSkillExecutor:
    """Adapt query_continuations only; cancellation/deadlines confer no HOLD."""
    if registration.skill.definition.skill_id != SKILL_ID:
        raise ValueError("Continuation executor requires its exact Skill registration.")
    registration.payload_validator.validate_route(registration.entry.route)
    if not callable(getattr(query_port, "query_continuations", None)):
        raise TypeError("The passive port must expose query_continuations.")

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        result = query_port.query_continuations(
            request.parameters,
            cancel_requested=lambda: (
                executor.cancel_requested(operation.operation_id)
                or clock() >= operation.deadline_at
            ),
        )
        registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
        return PassiveReadResult(result, ("tcp-continuation-query:" + operation.operation_id,),
            reason_code="tcp_continuation_query_completed")

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.tcp-continuation-query.v1", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations,
    )
    return executor

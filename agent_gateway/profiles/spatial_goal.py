"""One model-independent relation calculation for both Gateway backends."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import PassiveReadResult, PassiveReadSkillExecutor
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from agent_gateway.skills.perception.spatial_goal import (
    PARAMETER_KEYS, SKILL_ID, spatial_goal_definition, spatial_goal_schema,
    validate_spatial_goal_result,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.spatial_goals import compute_spatial_goal


@dataclass(frozen=True)
class SpatialGoalRegistration:
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet


def build_spatial_goal_registration(*, base_route, base_binding, component_ids, reference_frame_id):
    """Clone only passive target identity; accept no controller or device port."""
    if base_route.control_effect_class != "read_only":
        raise ValueError("Spatial goal calculation requires a read-only base route.")
    components = tuple(component_ids)
    if (not components or len(set(components)) != len(components)
            or any(not isinstance(x, str) or not x or x.strip() != x for x in components)
            or not isinstance(reference_frame_id, str) or not reference_frame_id
            or reference_frame_id.strip() != reference_frame_id):
        raise ValueError("Spatial goal needs identified components and a public frame.")
    profile_hash = canonical_content_hash(
        {"component_ids": sorted(components), "reference_frame_id": reference_frame_id},
        domain="agenticrobot.spatial_goal.endpoint_profile.v1",
    )
    definition = spatial_goal_definition()
    schema = spatial_goal_schema(definition_hash=definition.semantic_content_hash,
        profile_hash=profile_hash, component_ids=components, reference_frame_id=reference_frame_id)
    io_contract = inline_only_skill_io_contract(skill_id=SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash, parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash)
    route = replace(base_route, skill_id=SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash, route_id=base_route.route_id+".spatial_goal",
        route_version="1", parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash, max_operation_duration_ms=5_000)
    binding = base_binding.to_ref() if hasattr(base_binding, "to_ref") else base_binding
    binding = replace(binding, deployment_binding_id="spatial-goal:"+binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.spatial_goal.binding.v1"))

    def validate_parameters(parameters):
        if not isinstance(parameters, Mapping) or set(parameters) != PARAMETER_KEYS:
            raise ValueError("Spatial goal parameters must match the endpoint relation schema.")
        if parameters["component_id"] not in components or parameters["reference_frame_id"] != reference_frame_id:
            raise ValueError("Spatial goal component/frame differs from the public binding.")
        _calculate(parameters)

    def validate_result(result):
        validate_spatial_goal_result(result)
        if result["component_id"] not in components or result["reference_frame_id"] != reference_frame_id:
            raise ValueError("Spatial goal result changed its component/frame.")

    def validate_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        expected = _calculate(parameters)
        for key, value in expected.items():
            if key == "pose_targets_by_component":
                if any(not math.isclose(expected, actual, abs_tol=1e-7, rel_tol=0)
                       for expected, actual in zip(value[parameters["component_id"]],
                                                   result[key][parameters["component_id"]])):
                    raise ValueError("Spatial goal pose differs from its relation.")
            elif isinstance(value, float):
                if not math.isclose(value, result[key], abs_tol=1e-9, rel_tol=0):
                    raise ValueError("Spatial goal diagnostics differ from its relation.")
            elif value != result[key]:
                raise ValueError("Spatial goal result differs from its request.")

    validator = AgentSkillPayloadValidatorSet(validator_id="spatial-goal:"+profile_hash,
        skill_id=SKILL_ID, skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=validate_parameters, validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    return SpatialGoalRegistration(SkillRegistrationTemplate(definition, schema, io_contract, (route,)),
        SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
            availability_hash=canonical_content_hash({"route": route.semantic_content_hash, "status": "pure_calculation"},
                domain="agenticrobot.spatial_goal.availability.v1")), validator)


def _calculate(parameters):
    goal = compute_spatial_goal(**{key: parameters[key] for key in PARAMETER_KEYS
                                  if key not in {"component_id", "reference_frame_id"}})
    return {"component_id": parameters["component_id"], "reference_frame_id": parameters["reference_frame_id"],
        "pose_targets_by_component": {parameters["component_id"]: list(goal.pose_xyz_quat_xyzw)},
        "translation_distance_m": goal.translation_distance_m, "rotation_distance_rad": goal.rotation_distance_rad,
        "axis_alignment_error_rad": goal.axis_alignment_error_rad, "constraint_scope": "endpoint_only",
        "input_source": "caller_supplied_geometry", "path_checked": False, "motion_executed": False}


def build_spatial_goal_executor(*, registration, current_station_revision, clock, max_retained_operations):
    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        return PassiveReadResult(_calculate(request.parameters),
            ("spatial-goal-calculation:"+operation.operation_id,), reason_code="spatial_goal_computed")
    return PassiveReadSkillExecutor(port_id="agent-gateway:"+registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.spatial-goal.v1", read=read, current_station_revision=current_station_revision,
        clock=clock, max_retained_operations=max_retained_operations)

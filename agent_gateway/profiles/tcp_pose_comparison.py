"""Read-only comparison with an authenticated retained absolute TCP command.

The original state producer still owns the single measured cut. The operation
store still owns historical requests. This adapter combines those two existing
ports; it owns no sampler, motion lifecycle, threshold, or success evaluator.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest
from agent_gateway.protocol.skill_catalog import (
    SkillRouteManifest,
    SkillRouteManifestEntry,
)
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.skills.state.tcp_pose_comparison import (
    tcp_pose_comparison_definition_v2,
    tcp_pose_comparison_schema_entry_v2,
    validate_tcp_pose_comparison_parameters_v2,
    validate_tcp_pose_comparison_result_shape_v2,
    validate_tcp_pose_comparison_result_v2,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.transforms import rotation_angle_rad, translation_distance


@dataclass(frozen=True)
class TcpPoseComparisonRegistration:
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet


def _base_parameters(parameters):
    return {key: value for key, value in parameters.items() if key != "target_operation_id"}


def _errors(measured, targets):
    translation, rotation = {}, {}
    for component, target in targets.items():
        current = xyzquat_to_pose_matrix(measured[component])
        goal = xyzquat_to_pose_matrix(target)
        translation[component] = translation_distance(current, goal)
        rotation[component] = rotation_angle_rad(current, goal)
    return translation, rotation


def build_tcp_pose_comparison_registration(*, base_definition, base_schema, base_entry, base_validator):
    """Extend only an already configured V1 passive state registration."""
    route = base_entry.route
    if ((route.skill_id, route.skill_version) != ("state.read_tcp_pose", "1")
            or route.control_effect_class != "read_only"
            or route.privilege_tier != "t0_read_only"
            or route.cancellation_policy != "cancel_only"
            or route.may_move_robot or route.may_open_publication
            or route.requires_task_evaluation or base_entry.evaluation_plan_hash is not None):
        raise ValueError("TCP comparison requires a passive state V1 base with no action authority.")
    base_validator.validate_route(route)
    if (base_definition.semantic_content_hash != route.skill_definition_hash
            or base_schema.parameter_schema_hash != route.parameter_schema_hash
            or base_schema.execution_result_schema_hash != route.execution_result_schema_hash
            or base_schema.task_evaluation_schema_hash != route.task_evaluation_schema_hash):
        raise ValueError("TCP comparison base definition, schema and Route disagree.")
    properties = base_schema.parameter_schema["properties"]
    components = tuple(properties["component_ids"]["items"]["enum"])
    frames = tuple(properties["reference_frame_id"]["enum"])
    max_age = properties["max_measurement_age_ms"]["maximum"]
    profile_hash = canonical_content_hash(
        {"base_route": route.semantic_content_hash,
         "base_binding": base_entry.deployment_binding.deployment_binding_hash,
         "comparison_source": "retained_same_session_absolute_safe_command"},
        domain="agenticrobot.tcp_pose_comparison.profile.v2",
    )
    definition = tcp_pose_comparison_definition_v2(controlled_component_ids=components)
    schema = tcp_pose_comparison_schema_entry_v2(
        base_schema=base_schema, definition_hash=definition.semantic_content_hash,
        profile_hash=profile_hash,
    )
    io_contract = inline_only_skill_io_contract(
        skill_id=definition.skill_id, skill_version="2",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
    )
    route = replace(
        route, skill_version="2", skill_definition_hash=definition.semantic_content_hash,
        route_id=route.route_id + ".comparison", route_version="2",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash,
    )
    binding = base_entry.deployment_binding
    binding = replace(
        binding, deployment_binding_id="tcp-comparison:" + binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.tcp_pose_comparison.binding.v2",
        ),
    )

    def validate_parameters(parameters):
        validate_tcp_pose_comparison_parameters_v2(
            parameters, allowed_component_ids=components,
            allowed_reference_frame_ids=frames, max_measurement_age_ms=max_age,
        )
        base_validator.validate_parameters(_base_parameters(parameters))

    def validate_result(result):
        if not isinstance(result, Mapping) or set(result) != {
            *base_schema.execution_result_schema["properties"], "target_comparison",
        }:
            raise ValueError("TCP comparison result schema mismatch.")
        base_validator.validate_execution_result(
            {key: value for key, value in result.items() if key != "target_comparison"}
        )
        validate_tcp_pose_comparison_result_shape_v2(result)
        comparison = result["target_comparison"]
        if comparison is not None:
            translation, rotation = _errors(result["pose_by_component"], comparison["target_pose_by_component"])
            for key, expected in (("translation_error_m_by_component", translation),
                                  ("rotation_error_rad_by_component", rotation)):
                if any(not math.isclose(comparison[key][component], error, rel_tol=0, abs_tol=1e-9)
                       for component, error in expected.items()):
                    raise ValueError("TCP comparison error differs from its measured and target poses.")

    def validate_result_for(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        validate_tcp_pose_comparison_result_v2(parameters, result, allowed_component_ids=components)
        if base_validator.validate_execution_result_for_parameters is not None:
            base_validator.validate_execution_result_for_parameters(
                _base_parameters(parameters),
                {key: value for key, value in result.items() if key != "target_comparison"},
            )

    validator = AgentSkillPayloadValidatorSet(
        validator_id="tcp-comparison:" + profile_hash, skill_id=route.skill_id,
        skill_version="2", route_id=route.route_id, route_version=route.route_version,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=validate_parameters, validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_result_for,
        validate_task_evaluation=base_validator.validate_task_evaluation,
    )
    return TcpPoseComparisonRegistration(
        SkillRegistrationTemplate(definition, schema, io_contract, (route,)),
        SkillRouteManifestEntry(
            route=route, deployment_binding=binding, evaluation_plan_hash=None,
            availability_hash=canonical_content_hash(
                {"base": base_entry.availability_hash, "route": route.semantic_content_hash},
                domain="agenticrobot.tcp_pose_comparison.availability.v2",
            ),
        ), validator,
    )


def _authenticate_target(*, request, active_operation_id, lookup_target, manifest, validators, current_entry):
    target_id = request.parameters["target_operation_id"]
    if target_id == active_operation_id:
        raise ValueError("A TCP read cannot compare itself as a historical target.")
    retained = lookup_target(target_id)
    if retained is None:
        raise ValueError("TCP target is unavailable or not terminal for this client/session/manifest.")
    source_request, source_operation = retained
    if not isinstance(source_request, AgentSkillRequest) or not isinstance(source_operation, AgentSkillOperation):
        raise TypeError("TCP target lookup must return the authoritative typed request and operation.")
    if (source_operation.operation_id != target_id or not source_operation.terminal
            or source_operation.request_hash != source_request.semantic_content_hash
            or source_operation.runtime_session_id != request.runtime_session_id
            or source_operation.station_revision_at_admission != source_request.expected_station_revision
            or source_request.client_id != request.client_id
            or source_request.runtime_session_id != request.runtime_session_id
            or source_request.skill_route_manifest_hash != request.skill_route_manifest_hash
            or request.skill_route_manifest_hash != manifest.semantic_content_hash
            or request.runtime_session_id != manifest.runtime_session_id):
        raise ValueError("TCP target is unavailable or not terminal for this client/session/manifest.")
    if (source_request.skill_id != "motion.move_tcp_to_pose_safe"
            or source_request.skill_version not in ("1", "2")):
        raise ValueError("Only an absolute safe TCP command can be a comparison target.")
    source_entry = manifest.route(
        source_request.skill_id, source_request.skill_version,
        route_id=source_request.route_id, route_version=source_request.route_version,
    )
    route, binding = source_entry.route, source_entry.deployment_binding
    if (route.semantic_content_hash != source_request.route_hash
            or route.skill_definition_hash != source_request.skill_definition_hash
            or binding.deployment_binding_hash != source_request.deployment_binding_hash
            or route.control_effect_class != "motion" or not route.may_move_robot
            or route.target_kind != current_entry.route.target_kind
            or binding.target_binding_id != current_entry.deployment_binding.target_binding_id
            or binding.execution_manifest_hash != current_entry.deployment_binding.execution_manifest_hash
            or not any(profile.parameter_profile_id == source_request.parameter_profile_id
                       and profile.parameter_profile_hash == source_request.parameter_profile_hash
                       for profile in route.allowed_parameter_profiles)):
        raise ValueError("TCP target request differs from its authoritative absolute motion Route/binding.")
    validators.validate_parameters(route=route, payload=source_request.parameters)
    targets = source_request.parameters["pose_targets_by_component"]
    if not targets or not set(targets).issubset(request.parameters["component_ids"]):
        raise ValueError("TCP comparison read must cover every component of the original target.")
    return source_request, source_operation


def build_tcp_pose_comparison_executor(
    *, registration: TcpPoseComparisonRegistration,
    capture_state: Callable[[Mapping, str], PassiveReadResult],
    lookup_target: Callable[[str], tuple[AgentSkillRequest, AgentSkillOperation] | None],
    route_manifest: SkillRouteManifest, payload_validators: AgentSkillPayloadValidatorRegistry,
    clock_domain_id: str, current_station_revision, clock, max_retained_operations,
):
    """Reuse a V1 single-cut reader and the owning service's bounded history."""
    if not isinstance(clock_domain_id, str) or not clock_domain_id or clock_domain_id != clock_domain_id.strip():
        raise ValueError("TCP comparison needs the original producer clock identity.")
    if route_manifest.route("state.read_tcp_pose", "2") != registration.entry:
        raise ValueError("TCP comparison registration differs from the fixed manifest.")

    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        source = None
        if request.parameters["target_operation_id"] is not None:
            source = _authenticate_target(
                request=request, active_operation_id=operation.operation_id,
                lookup_target=lookup_target, manifest=route_manifest,
                validators=payload_validators, current_entry=registration.entry,
            )
        measured = capture_state(_base_parameters(request.parameters), operation.operation_id)
        if not isinstance(measured, PassiveReadResult):
            raise TypeError("TCP comparison capture must reuse a passive state read result.")
        result = dict(measured.execution_result)
        if result["clock_domain_id"] != clock_domain_id:
            raise ValueError("TCP comparison measured cut changed its producer clock domain.")
        comparison = None
        evidence = measured.evidence_refs
        if source is not None:
            source_request, source_operation = source
            for key in ("reference_frame_id", "reference_frame_spec_hash", "controlled_frame_id",
                        "frame_graph_hash", "tool_config_hash", "calibration_hash"):
                if source_request.parameters[key] != result[key]:
                    raise ValueError(f"TCP comparison historical target {key} differs from this measured cut.")
            targets = source_request.parameters["pose_targets_by_component"]
            translation, rotation = _errors(result["pose_by_component"], targets)
            comparison = {
                "target_operation_id": source_operation.operation_id,
                "target_request_hash": source_request.semantic_content_hash,
                "target_route_hash": source_request.route_hash,
                "target_skill_version": source_request.skill_version,
                "target_status": source_operation.status,
                "target_execution_outcome": source_operation.execution_outcome,
                "target_task_outcome": source_operation.task_outcome,
                "target_pose_by_component": {key: list(value) for key, value in targets.items()},
                "translation_error_m_by_component": translation,
                "rotation_error_rad_by_component": rotation,
            }
            evidence += ("tcp-comparison-target-operation:" + source_operation.operation_id,
                         "tcp-comparison-target-request:" + source_request.semantic_content_hash)
        result["target_comparison"] = comparison
        registration.payload_validator.validate_execution_result_for_parameters(request.parameters, result)
        return PassiveReadResult(
            execution_result=result, evidence_refs=evidence,
            output_artifact_bindings=measured.output_artifact_bindings,
            reason_code="passive_tcp_pose_comparison_completed",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.tcp-pose-comparison.v2", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations,
    )

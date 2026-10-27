"""Authenticate retained sensor evidence without creating a second world cache.

An observation remains a historical location after station motion. Reuse is
scoped to one client/session/manifest/frame graph; it never asserts object
tracking, current obstacle clearance or an execution permission.
"""

from dataclasses import dataclass
import math

from agent_gateway.protocol.contracts import AgentSkillOperation, AgentSkillRequest


@dataclass(frozen=True)
class RetainedObservedPoint:
    operation_id: str
    request_hash: str
    point_index: int
    point_reference_m: tuple[float, float, float]
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    depth_content_hash: str
    capture_group_id: str
    station_revision: int
    acquired_at: float | None


def resolve_retained_observed_point(*, request, lookup, manifest, validators, current_entry):
    """Resolve v1/v2 back-projection from the service-owned request/operation pair."""
    source_id = request.parameters["observation_operation_id"]
    pair = lookup(source_id)
    if pair is None:
        raise ValueError("Observed target is missing or evicted; obtain a new observation.")
    source_request, operation = pair
    if not isinstance(source_request, AgentSkillRequest) or not isinstance(operation, AgentSkillOperation):
        raise TypeError("Observed target lookup requires the authoritative typed pair.")
    if (operation.operation_id != source_id or not operation.terminal
            or operation.status != "completed" or operation.execution_outcome != "completed"
            or operation.request_hash != source_request.semantic_content_hash
            or operation.runtime_session_id != request.runtime_session_id
            or source_request.runtime_session_id != request.runtime_session_id
            or source_request.client_id != request.client_id
            or operation.station_revision_at_admission != source_request.expected_station_revision
            or source_request.skill_route_manifest_hash != request.skill_route_manifest_hash
            or request.skill_route_manifest_hash != manifest.semantic_content_hash
            or request.runtime_session_id != manifest.runtime_session_id):
        raise ValueError("Observed target is not successful evidence from this client/session/manifest.")
    if source_request.skill_id != "perception.back_project" or source_request.skill_version not in ("1", "2"):
        raise ValueError("Observed target must be a completed sensor back-projection, not an inferred pose.")
    entry = manifest.route(source_request.skill_id, source_request.skill_version,
                           route_id=source_request.route_id, route_version=source_request.route_version)
    route, binding = entry.route, entry.deployment_binding
    if (route.semantic_content_hash != source_request.route_hash
            or route.skill_definition_hash != source_request.skill_definition_hash
            or binding.deployment_binding_hash != source_request.deployment_binding_hash
            or route.control_effect_class != "read_only" or route.may_move_robot
            or route.target_kind != current_entry.route.target_kind
            or binding.target_binding_id != current_entry.deployment_binding.target_binding_id
            or binding.execution_manifest_hash != current_entry.deployment_binding.execution_manifest_hash
            or not any(p.parameter_profile_id == source_request.parameter_profile_id
                       and p.parameter_profile_hash == source_request.parameter_profile_hash
                       for p in route.allowed_parameter_profiles)):
        raise ValueError("Observed target source Route/binding differs from its authoritative manifest.")
    validators.validate_parameters(route=route, payload=source_request.parameters)
    result = operation.execution_result
    validators.validate_execution_result(route=route, payload=result, request_parameters=source_request.parameters)
    # V1 captures live; V2 may back-project an older, retained RGB-D cut.
    if (result["station_revision"] > operation.station_revision_at_admission
            or (source_request.skill_version == "1"
                and result["station_revision"] != operation.station_revision_at_admission)):
        raise ValueError("Observed target result revision differs from its source contract.")
    index = request.parameters["point_index"]
    if type(index) is not int or index < 0:
        raise ValueError("Observed target point index must be a nonnegative integer.")
    if source_request.skill_version == "1":
        if index != 0:
            raise ValueError("Back-projection v1 contains exactly one point, index zero.")
        point = result["world_xyz_m"]
        acquired_at = None  # V1 does not publish the sensor acquisition timestamp.
    else:
        if index >= len(result["points"]):
            raise ValueError("Observed target point index exceeds the saved back-projection.")
        point = result["points"][index]["point_reference_m"]
        acquired_at = result["acquired_at"]
    if (not isinstance(point, (list, tuple)) or len(point) != 3
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in point)):
        raise ValueError("Observed target needs three finite measured coordinates.")
    return RetainedObservedPoint(source_id, source_request.semantic_content_hash, index,
        tuple(float(v) for v in point), result["reference_frame_id"], result["reference_frame_spec_hash"],
        result["frame_graph_hash"], result["depth_content_hash"], result["capture_group_id"],
        result["station_revision"], acquired_at)

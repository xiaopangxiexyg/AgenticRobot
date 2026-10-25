"""A compact relation action over a retained sensor-derived surface point."""

from collections.abc import Mapping
from dataclasses import replace
import math

from agent_gateway.protocol.skill_catalog import SkillDefinition
from agent_gateway.skills.motion.schemas import _schema_entry, _pose_schema, _pose

SKILL_ID = "motion.move_to_observed_point"
PARAMETERS = {"component_id", "observation_operation_id", "point_index", "offset_xyz_m", "offset_frame"}
POINT_SEMANTICS = "saved_surface_point_not_tracked_object_pose"


def observed_point_definition(*, controlled_component_ids):
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="1", skill_kind="motion",
        summary=("Move TCP to a saved perception.back_project point plus offset, preserving the "
                 "current TCP orientation. Reuse the completed observation operation; point_index "
                 "is zero for v1. Reference offsets use the deployed public motion frame; tool "
                 "offsets use the current TCP axes. The lower owner resolves frames, measures TCP, "
                 "checks a fresh RGB-D collision scene and executes the original safe path. "
                 "This is a saved surface location, not a tracked object centre/pose; refresh "
                 "observation if the object moved. No gripper or grasp action."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=tuple(controlled_component_ids),
        parameter_semantics_id="observed_point.relation.preserving_orientation.v1",
        execution_result_semantics_id="observed_point.safe_arrival.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "authenticated_retained_observation.v1",
                             "configured_scene_collision_free_path_found.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def observed_point_schema(*, definition, profile):
    properties = {
        "component_id": {"const": profile.component_id},
        "observation_operation_id": {"type": "string", "minLength": 1},
        "point_index": {"type": "integer", "minimum": 0, "maximum": 127},
        "offset_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        "offset_frame": {"enum": ["reference", "tool"],
                         "description": "reference=" + profile.reference_frame_id + "; tool=current TCP axes"},
    }
    entry = _schema_entry(skill_id=SKILL_ID, skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id, parameter_profile_hash=profile.semantic_content_hash,
        parameter_schema={"type": "object", "additionalProperties": False,
                          "properties": properties, "required": list(properties)})
    report_properties = {
        "observation_operation_id": {"type": "string", "minLength": 1},
        "source_request_hash": {"type": "string", "minLength": 1},
        "point_index": {"type": "integer", "minimum": 0, "maximum": 127},
        "point_semantics": {"const": POINT_SEMANTICS},
        "reference_frame_id": {"const": profile.reference_frame_id},
        "requested_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=profile.workspace_bounds),
        "achieved_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=None),
        "translation_error_m": {"type": "number", "minimum": 0},
        "rotation_error_rad": {"type": "number", "minimum": 0},
    }
    result = dict(entry.execution_result_schema)
    result["properties"] = {**result["properties"], "observed_point_report": {
        "type": "object", "additionalProperties": False,
        "properties": report_properties, "required": list(report_properties)}}
    result["required"] = [*result["required"], "observed_point_report"]
    return replace(entry, execution_result_schema=result)


def validate_observed_point_parameters(parameters, *, component_id):
    if not isinstance(parameters, Mapping) or set(parameters) != PARAMETERS:
        raise ValueError("Observed-point motion requires the exact compact relation schema.")
    identifier = parameters["observation_operation_id"]
    index = parameters["point_index"]
    offset = parameters["offset_xyz_m"]
    if (parameters["component_id"] != component_id or not isinstance(identifier, str)
            or not identifier or identifier.strip() != identifier
            or type(index) is not int or not 0 <= index < 128
            or parameters["offset_frame"] not in ("reference", "tool")
            or not isinstance(offset, (list, tuple)) or len(offset) != 3
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in offset)):
        raise ValueError("Observed-point reference, component or relation is invalid.")


def validate_observed_point_report(report, *, reference_frame_id):
    keys = {"observation_operation_id", "source_request_hash", "point_index", "point_semantics",
            "reference_frame_id", "requested_pose_xyz_quat_xyzw", "achieved_pose_xyz_quat_xyzw",
            "translation_error_m", "rotation_error_rad"}
    if not isinstance(report, Mapping) or set(report) != keys:
        raise ValueError("Observed-point report must contain exact provenance and measured arrival.")
    if (report["point_semantics"] != POINT_SEMANTICS or report["reference_frame_id"] != reference_frame_id
            or type(report["point_index"]) is not int or not 0 <= report["point_index"] < 128
            or any(not isinstance(report[k], str) or not report[k] for k in
                   ("observation_operation_id", "source_request_hash"))):
        raise ValueError("Observed-point report identity is invalid.")
    for key in ("requested_pose_xyz_quat_xyzw", "achieved_pose_xyz_quat_xyzw"):
        _pose(report[key], field_name=key)
    for key in ("translation_error_m", "rotation_error_rad"):
        value = report[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Observed-point report error must be finite and nonnegative.")

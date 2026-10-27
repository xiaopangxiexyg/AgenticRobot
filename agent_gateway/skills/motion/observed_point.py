"""A compact relation action over a retained sensor-derived surface point."""

from collections.abc import Mapping
from dataclasses import replace
import math

from agent_gateway.protocol.skill_catalog import SkillDefinition
from agent_gateway.skills.motion.schemas import _schema_entry, _pose_schema, _pose

SKILL_ID = "motion.move_to_observed_point"
PARAMETERS = {"component_id", "observation_operation_id", "point_index", "offset_xyz_m", "offset_frame"}
POINT_SEMANTICS = "saved_surface_point_not_tracked_object_pose"
ANCHOR_PARAMETERS_V2 = {"component_id", "surface_anchor_operation_id", "offset_xyz_m", "offset_frame"}
ANCHOR_REPORT_V2 = {
    "source_operation_id", "source_request_hash", "anchor_artifact_id", "point_semantics",
    "reference_frame_id", "requested_pose_xyz_quat_xyzw", "achieved_pose_xyz_quat_xyzw",
    "translation_error_m", "rotation_error_rad",
}
REGION_PARAMETERS_V3 = (ANCHOR_PARAMETERS_V2 - {"offset_frame"}) | {
    "lateral_radius_m", "height_extra_m"}
REGION_REPORT_V3 = ANCHOR_REPORT_V2 | {
    "selected_offset_xyz_m", "selected_candidate_index", "attempted_candidate_count",
}


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


def observed_anchor_definition_v2(*, controlled_component_ids):
    """Retained-anchor relation action; v1's back-project source is unchanged."""
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="2", skill_kind="motion",
        summary=("Move TCP to one authenticated capture_surface_anchor@2 point plus a bounded offset, "
                 "preserving the measured TCP orientation. Pass the completed capture operation ID; "
                 "the lower owner verifies its exact surface_anchor Artifact from this client/session "
                 "and current Station revision. Do not transcribe XYZ or a target pose. "
                 "The lower owner resolves the saved RGB-D point, measures current TCP, checks a fresh "
                 "collision scene and executes the original safe path/HOLD lifecycle. The point is a "
                 "historical visible surface location, not a tracked object pose; refresh after movement. "
                 "No gripper or grasp action."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=tuple(controlled_component_ids),
        parameter_semantics_id="authenticated_surface_anchor.relation.preserving_orientation.v2",
        execution_result_semantics_id="surface_anchor.safe_arrival.v2",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "authenticated_surface_anchor.v2",
                             "configured_scene_collision_free_path_found.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def observed_anchor_schema_v2(*, definition, profile):
    properties = {
        "component_id": {"const": profile.component_id},
        "surface_anchor_operation_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "offset_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3,
                         "items": {"type": "number"}},
        "offset_frame": {"enum": ["reference", "tool"],
                         "description": "reference=" + profile.reference_frame_id + "; tool=current TCP axes"},
    }
    entry = replace(_schema_entry(skill_id=SKILL_ID,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id, parameter_profile_hash=profile.semantic_content_hash,
        parameter_schema={"type": "object", "additionalProperties": False,
                          "properties": properties, "required": sorted(properties)}), skill_version="2")
    report_properties = {
        "source_operation_id": {"type": "string", "minLength": 1},
        "source_request_hash": {"type": "string", "minLength": 1},
        "anchor_artifact_id": {"type": "string", "pattern": "^artifact:[0-9a-f]{64}$"},
        "point_semantics": {"const": POINT_SEMANTICS},
        "reference_frame_id": {"const": profile.reference_frame_id},
        "requested_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=profile.workspace_bounds),
        "achieved_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=None),
        "translation_error_m": {"type": "number", "minimum": 0},
        "rotation_error_rad": {"type": "number", "minimum": 0},
    }
    result = dict(entry.execution_result_schema)
    result["properties"] = {**result["properties"], "surface_anchor_report": {
        "type": "object", "additionalProperties": False,
        "properties": report_properties, "required": sorted(report_properties)}}
    result["required"] = [*result["required"], "surface_anchor_report"]
    return replace(entry, execution_result_schema=result)


def validate_observed_anchor_parameters_v2(parameters, *, component_id):
    if not isinstance(parameters, Mapping) or set(parameters) != ANCHOR_PARAMETERS_V2:
        raise ValueError("Anchor motion requires its exact compact v2 relation schema.")
    offset = parameters["offset_xyz_m"]
    operation_id = parameters["surface_anchor_operation_id"]
    if (parameters["component_id"] != component_id
            or not isinstance(operation_id, str) or not operation_id
            or operation_id.strip() != operation_id or len(operation_id) > 256
            or parameters["offset_frame"] not in ("reference", "tool")
            or not isinstance(offset, (list, tuple)) or len(offset) != 3
            or any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
                   for v in offset)):
        raise ValueError("Anchor motion component or relation is invalid.")


def validate_observed_anchor_report_v2(report, *, reference_frame_id):
    if not isinstance(report, Mapping) or set(report) != ANCHOR_REPORT_V2:
        raise ValueError("Anchor motion report must retain exact source and measured arrival.")
    if (report["point_semantics"] != POINT_SEMANTICS or report["reference_frame_id"] != reference_frame_id
            or any(not isinstance(report[key], str) or not report[key] for key in
                   ("source_operation_id", "source_request_hash"))
            or not isinstance(report["anchor_artifact_id"], str)
            or not report["anchor_artifact_id"].startswith("artifact:")
            or len(report["anchor_artifact_id"]) != 73
            or any(c not in "0123456789abcdef" for c in report["anchor_artifact_id"][9:])):
        raise ValueError("Anchor motion report source identity is invalid.")
    for key in ("requested_pose_xyz_quat_xyzw", "achieved_pose_xyz_quat_xyzw"):
        _pose(report[key], field_name=key)
    for key in ("translation_error_m", "rotation_error_rad"):
        value = report[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("Anchor motion report error must be finite and nonnegative.")


def observed_anchor_region_offsets_v3(parameters):
    """Bounded Station-frame relation alternatives; identical inputs give identical order."""
    validate_observed_anchor_region_parameters_v3(parameters)
    x, y, z = (float(value) for value in parameters["offset_xyz_m"])
    radius = float(parameters["lateral_radius_m"])
    lift = float(parameters["height_extra_m"])
    variants = ((x, y, z), (x, y, z + lift),
                (x - radius, y, z + lift), (x + radius, y, z + lift),
                (x, y - radius, z + lift), (x, y + radius, z + lift))
    return tuple(dict.fromkeys(variants))


def validate_observed_anchor_region_parameters_v3(parameters, *, component_id=None):
    if not isinstance(parameters, Mapping) or set(parameters) != REGION_PARAMETERS_V3:
        raise ValueError("Anchor region motion requires its exact compact v3 relation schema.")
    component = parameters["component_id"]
    source = parameters["surface_anchor_operation_id"]
    offset = parameters["offset_xyz_m"]
    if (not isinstance(component, str) or not component
            or component_id is not None and component != component_id
            or not isinstance(source, str) or not source or source.strip() != source
            or len(source) > 256 or not isinstance(offset, (list, tuple)) or len(offset) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in offset)):
        raise ValueError("Anchor region source, component or offset is invalid.")
    for key, maximum in (("lateral_radius_m", .30), ("height_extra_m", .30)):
        value = parameters[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= maximum:
            raise ValueError(f"Anchor region {key} must be finite within 0..{maximum} m.")


def observed_anchor_region_definition_v3(*, controlled_component_ids):
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="3", skill_kind="motion",
        summary=("Move TCP near one authenticated capture_surface_anchor@2 point. Pass its completed "
                 "operation ID, a nominal Station-frame offset, maximum lateral radius and extra height. "
                 "The lower owner deterministically tests up to six offsets on one fresh RGB-D and measured-joint "
                 "scene, executes the first collision-free safe path, and reports the selected offset. "
                 "No model-authored XYZ, target pose, Artifact descriptor, gripper or grasp action. "
                 "The saved point is a historical surface location, not a tracked object pose."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=tuple(controlled_component_ids),
        parameter_semantics_id="authenticated_surface_anchor.region.reference_frame.v3",
        execution_result_semantics_id="surface_anchor.region_safe_arrival.v3",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "authenticated_surface_anchor.v2",
                             "configured_scene_collision_free_path_found.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )


def observed_anchor_region_schema_v3(*, definition, profile):
    properties = {
        "component_id": {"const": profile.component_id},
        "surface_anchor_operation_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "offset_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3,
                         "items": {"type": "number"}},
        "lateral_radius_m": {"type": "number", "minimum": 0, "maximum": .30},
        "height_extra_m": {"type": "number", "minimum": 0, "maximum": .30},
    }
    entry = replace(_schema_entry(skill_id=SKILL_ID,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id, parameter_profile_hash=profile.semantic_content_hash,
        parameter_schema={"type": "object", "additionalProperties": False,
                          "properties": properties, "required": sorted(properties)}), skill_version="3")
    report_properties = {
        "source_operation_id": {"type": "string", "minLength": 1},
        "source_request_hash": {"type": "string", "minLength": 1},
        "anchor_artifact_id": {"type": "string", "pattern": "^artifact:[0-9a-f]{64}$"},
        "point_semantics": {"const": POINT_SEMANTICS},
        "reference_frame_id": {"const": profile.reference_frame_id},
        "requested_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=profile.workspace_bounds),
        "achieved_pose_xyz_quat_xyzw": _pose_schema(workspace_bounds=None),
        "translation_error_m": {"type": "number", "minimum": 0},
        "rotation_error_rad": {"type": "number", "minimum": 0},
        "selected_offset_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3,
                                  "items": {"type": "number"}},
        "selected_candidate_index": {"type": "integer", "minimum": 0, "maximum": 5},
        "attempted_candidate_count": {"type": "integer", "minimum": 1, "maximum": 6},
    }
    result = dict(entry.execution_result_schema)
    result["properties"] = {**result["properties"], "surface_anchor_region_report": {
        "type": "object", "additionalProperties": False,
        "properties": report_properties, "required": sorted(report_properties)}}
    result["required"] = [*result["required"], "surface_anchor_region_report"]
    return replace(entry, execution_result_schema=result)


def validate_observed_anchor_region_report_v3(report, *, reference_frame_id):
    if not isinstance(report, Mapping) or set(report) != REGION_REPORT_V3:
        raise ValueError("Anchor region report must retain exact provenance and selected arrival.")
    validate_observed_anchor_report_v2({key: report[key] for key in ANCHOR_REPORT_V2},
                                       reference_frame_id=reference_frame_id)
    offset = report["selected_offset_xyz_m"]
    index = report["selected_candidate_index"]
    count = report["attempted_candidate_count"]
    if (not isinstance(offset, (list, tuple)) or len(offset) != 3
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in offset)
            or type(index) is not int or type(count) is not int
            or not 0 <= index < count <= 6):
        raise ValueError("Anchor region selected candidate evidence is invalid.")

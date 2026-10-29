"""Read-only relation goals; endpoint constraints do not certify a path."""

from __future__ import annotations

import math
from collections.abc import Mapping

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry, SkillDefinition, SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

SKILL_ID = "perception.compute_spatial_goal"
PARAMETER_KEYS = {
    "component_id", "reference_frame_id", "current_tcp_pose_xyz_quat_xyzw",
    "anchor_xyz_m", "offset_xyz_m", "offset_frame", "orientation_mode",
    "tool_axis", "axis_reference_xyz",
}
RESULT_KEYS = {
    "component_id", "reference_frame_id", "pose_targets_by_component",
    "translation_distance_m", "rotation_distance_rad", "axis_alignment_error_rad",
    "constraint_scope", "input_source", "path_checked", "motion_executed",
}


def finite_vector(value, size, *, name):
    """Validate public numeric arrays without importing a geometry backend."""
    if (not isinstance(value, (tuple, list)) or len(value) != size
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must contain {size} finite numbers.")
    return tuple(float(x) for x in value)


def validate_spatial_goal_geometry(*, offset_frame, orientation_mode, tool_axis, axis_reference_xyz):
    """Validate relation choices; profiles own endpoint construction and FK."""
    if offset_frame not in ("reference", "tool"):
        raise ValueError("offset_frame must be reference or tool.")
    if orientation_mode not in ("preserve", "align_axis"):
        raise ValueError("orientation_mode must be preserve or align_axis.")
    if orientation_mode == "preserve":
        if tool_axis is not None or axis_reference_xyz is not None:
            raise ValueError("Preserve mode cannot contain an axis constraint.")
    else:
        if tool_axis not in ("x", "y", "z"):
            raise ValueError("align_axis requires tool_axis x, y, or z.")
        direction = finite_vector(axis_reference_xyz, 3, name="axis_reference_xyz")
        norm = math.hypot(*direction)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("Axis constraint must have a nonzero finite length.")


def spatial_goal_geometry_properties():
    """Share endpoint relation parameters without introducing another Skill."""
    vector = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
    return {"offset_xyz_m": vector, "offset_frame": {"enum": ["reference", "tool"]},
        "orientation_mode": {"enum": ["preserve", "align_axis"]},
        "tool_axis": {"enum": [None, "x", "y", "z"]},
        "axis_reference_xyz": {"anyOf": [vector, {"type": "null"}]}}


def spatial_goal_definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="1", skill_kind="inspection",
        summary=("Compute a TCP endpoint proposal from caller-supplied geometry: anchor plus offset "
                 "in reference or input tool axes. Preserve orientation or align one tool axis "
                 "with minimum rotation. Not a fresh observation, path constraint, collision check, "
                 "or motion authorization; submit a proposal separately to a motion Skill."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(), parameter_semantics_id="spatial_goal.endpoint_relation.v1",
        execution_result_semantics_id="spatial_goal.caller_geometry_proposal.v1",
        task_outcome_semantics_id="no_task_outcome.v1", entry_predicate_ids=(),
        task_outcome_predicate_ids=("endpoint_proposal_computed_without_motion.v1",),
    )


def spatial_goal_schema(*, definition_hash: str, profile_hash: str,
                        component_ids: tuple[str, ...], reference_frame_id: str) -> AgentSkillSchemaEntry:
    def vector(size):
        return {"type": "array", "minItems": size, "maxItems": size, "items": {"type": "number"}}
    nullable_axis = {"anyOf": [vector(3), {"type": "null"}]}
    identity = {"component_id": {"type": "string", "enum": list(component_ids)},
                "reference_frame_id": {"type": "string", "const": reference_frame_id}}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        **identity, "current_tcp_pose_xyz_quat_xyzw": vector(7), "anchor_xyz_m": vector(3),
        "offset_xyz_m": vector(3), "offset_frame": {"enum": ["reference", "tool"]},
        "orientation_mode": {"enum": ["preserve", "align_axis"], "description": (
            "preserve keeps the current tool orientation; align_axis turns tool_axis onto "
            "axis_reference_xyz with the least rotation.")},
        "tool_axis": {"enum": [None, "x", "y", "z"], "description": (
            "Tool axis to align; z is the gripper approach axis. null with preserve.")},
        "axis_reference_xyz": {**nullable_axis, "description": (
            "Reference direction for tool_axis in the reference frame. The general gripper posture "
            "is horizontal forward: tool_axis z onto [1, 0, 0]. Use [0, 0, -1] (straight down) only "
            "when a top-down approach is explicitly required; exactly-down poses are often out of "
            "reach. null with preserve.")},
    }, "required": sorted(PARAMETER_KEYS)}
    result = {"type": "object", "additionalProperties": False, "properties": {
        **identity,
        "pose_targets_by_component": {"type": "object", "additionalProperties": False,
            "minProperties": 1, "maxProperties": 1,
            "properties": {component: vector(7) for component in component_ids}},
        "translation_distance_m": {"type": "number", "minimum": 0},
        "rotation_distance_rad": {"type": "number", "minimum": 0, "maximum": math.pi},
        "axis_alignment_error_rad": {"anyOf": [
            {"type": "number", "minimum": 0, "maximum": math.pi}, {"type": "null"}]},
        "constraint_scope": {"const": "endpoint_only"},
        "input_source": {"const": "caller_supplied_geometry"},
        "path_checked": {"const": False}, "motion_executed": {"const": False},
    }, "required": sorted(RESULT_KEYS)}
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="spatial-goal-endpoint-v1", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),),
    )


def validate_spatial_goal_result(payload: Mapping) -> None:
    if set(payload) != RESULT_KEYS:
        raise ValueError("Unexpected spatial goal result fields.")
    if (payload["constraint_scope"] != "endpoint_only"
            or payload["input_source"] != "caller_supplied_geometry"
            or payload["path_checked"] is not False or payload["motion_executed"] is not False):
        raise ValueError("A spatial goal cannot claim observation, path checking or execution.")
    component = payload["component_id"]
    targets = payload["pose_targets_by_component"]
    for key in ("component_id", "reference_frame_id"):
        value = payload[key]
        if not isinstance(value, str) or not 0 < len(value) <= 256 or value.strip() != value:
            raise ValueError("Invalid spatial goal component/frame identity.")
    if not isinstance(component, str) or not isinstance(targets, Mapping) or set(targets) != {component}:
        raise ValueError("Spatial goal must target exactly its requested component.")
    pose = targets[component]
    if (not isinstance(pose, (list, tuple)) or len(pose) != 7
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pose)
            or abs(math.hypot(*pose[3:]) - 1.0) > 1e-5):
        raise ValueError("Spatial goal pose must contain a finite xyz and unit xyzw quaternion.")
    for key in ("translation_distance_m", "rotation_distance_rad", "axis_alignment_error_rad"):
        value = payload[key]
        if value is None and key == "axis_alignment_error_rad":
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value < 0
                or (key != "translation_distance_m" and value > math.pi + 1e-12)):
            raise ValueError("Invalid spatial goal distance/angle.")

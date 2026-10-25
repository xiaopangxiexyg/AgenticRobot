"""Bounded, read-only path evidence for object-above endpoint samples."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.perception.spatial_goal import (
    finite_vector,
    validate_spatial_goal_geometry,
)

SKILL_ID = "perception.query_object_above_paths"
RELATION_SKILL_ID = "perception.query_relation_paths"
RELATION_FIELDS = {
    "component_id", "reference_frame_id", "offsets_xyz_m", "offset_frame",
    "orientation_mode", "tool_axis", "axis_reference_xyz", "tool_axis_rotation_offsets_rad", "time_budget_sec",
}
PARAMETER_KEYS = {
    "component_id",
    "reference_frame_id",
    "anchor_xyz_m",
    "heights_m",
    "tool_yaw_offsets_rad",
    "time_budget_sec",
}
RANKING = "max_joint_excursion_rad_then_duration_sec_then_total_joint_travel_rad"
RESULT_KEYS = {
    "component_id",
    "reference_frame_id",
    "anchor_xyz_m",
    "source_tcp_pose_xyz_quat_xyzw",
    "query_status",
    "scene_hash",
    "source_qpos_hash",
    "source_sim_time_sec",
    "pose_targets_by_component",
    "candidate_count",
    "checked_count",
    "all_candidates_checked",
    "selected_index",
    "candidate_diagnostics",
    "planning_wall_sec",
    "ranking_criteria",
    "motion_executed",
    "execution_authorized",
    "input_source",
    "constraint_scope",
}


def object_goal_query_definition():
    return SkillDefinition(
        skill_id=SKILL_ID,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Query collision/envelope-checked paths above a caller-supplied measured object point. "
            "Heights in public reference +Z; tool +Z points down; yaw offsets are radians about tool +Z. "
            "Uses one frozen scene/current TCP, <=64 samples and <=30 seconds. Returns best among completed "
            "candidates. time_budget_sec bounds the whole query; each candidate path search gets at most "
            "3 seconds and never more than the remaining query budget. Found paths were checked in that frozen scene; "
            "this is not global optimality, proof of unreachability, live revalidation or execution authorization. "
            "Submit any chosen pose separately through motion.move_tcp_to_pose_safe with fresh checks."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="object_above_path_query.v1",
        execution_result_semantics_id="frozen_scene_path_evidence.v1",
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=(),
        task_outcome_predicate_ids=("bounded_query_without_motion.v1",),
    )


def relation_query_definition():
    return replace(object_goal_query_definition(), skill_id=RELATION_SKILL_ID,
        parameter_semantics_id="artifact_relation_path_query.v1",
        execution_result_semantics_id="anchored_relation_path_evidence.v1",
        summary=("Query paths for general relation endpoints from one authenticated surface_anchor Artifact. "
            "Supply offsets in reference or original input-tool axes; preserve orientation, or align a tool axis "
            "and sample allowed rotations about that same axis in radians. Preserve permits only zero rotations. "
            "At most 64 offset/rotation combinations and 30 seconds for the whole query; partial search is explicit. "
            "Each candidate path search gets at most 3 seconds and the remaining query budget. "
            "Uses a fresh frozen measured TCP/scene, existing collision/envelope planner and joint-excursion ranking. "
            "Returns surface_anchor and path_query.pose_targets_by_component; no manual XYZ or current TCP input. "
            "When candidate_diagnostics includes pose_target_xyz_quat_xyzw, it is the exact generated "
            "proposal for that candidate_index, including cost-pruned candidates, not an additional path certificate. "
            "Reuse these unchanged proposals in a separately exposed continuation query when a followup is required; "
            "the first-stage winner alone need not be best for the complete task. "
            "Relation/orientation constraints are endpoint-only; full-path checks cover collision and the bound "
            "deployment envelope, not custom corridors or maintained transit orientation. No execution permission. "
            "Submit the selected pose separately to move_tcp_to_pose_safe for fresh checks. "
            "This general query covers above and lateral offsets without separate directional Skills."))


def validate_relation_query_parameters(parameters):
    if not isinstance(parameters, Mapping) or set(parameters) != RELATION_FIELDS:
        raise ValueError("Relation query parameters exclude caller XYZ and measured TCP.")
    budget = parameters["time_budget_sec"]
    if (isinstance(budget, bool) or not isinstance(budget, (int, float))
            or not math.isfinite(budget) or not 0.1 <= budget <= 30):
        raise ValueError("Query time budget must be 0.1..30 seconds.")
    offsets, turns = parameters["offsets_xyz_m"], parameters["tool_axis_rotation_offsets_rad"]
    if (not isinstance(offsets, (list, tuple)) or not offsets
            or not isinstance(turns, (list, tuple)) or not turns
            or len(offsets) * len(turns) > 64):
        raise ValueError("Relation sampling needs 1..64 explicit offset/rotation combinations.")
    for offset in offsets:
        finite_vector(offset, 3, name="offset_xyz_m")
    finite_vector(turns, len(turns), name="tool_axis_rotation_offsets_rad")
    validate_spatial_goal_geometry(**{key: parameters[key] for key in (
        "offset_frame", "orientation_mode", "tool_axis", "axis_reference_xyz")})
    if parameters["orientation_mode"] == "preserve" and any(
            abs((turn + math.pi) % math.tau - math.pi) > 1e-12 for turn in turns):
        raise ValueError("Preserve mode cannot contain a nonzero tool-axis rotation.")


def validate_query_parameters(parameters):
    if not isinstance(parameters, Mapping) or set(parameters) != PARAMETER_KEYS:
        raise ValueError("Object path query parameters must match the exact schema.")
    budget = parameters["time_budget_sec"]
    if (
        isinstance(budget, bool)
        or not isinstance(budget, (float, int))
        or not math.isfinite(budget)
        or not 0.1 <= budget <= 30
    ):
        raise ValueError("Query time budget must be 0.1..30 seconds.")
    finite_vector(parameters["anchor_xyz_m"], 3, name="anchor_xyz_m")
    heights, yaws = parameters["heights_m"], parameters["tool_yaw_offsets_rad"]
    if (not isinstance(heights, (list, tuple)) or not heights
            or not isinstance(yaws, (list, tuple)) or not yaws
            or len(heights) * len(yaws) > 64):
        raise ValueError("Above-goal sampling needs 1..64 explicit height/yaw combinations.")
    finite_vector(heights, len(heights), name="heights_m")
    finite_vector(yaws, len(yaws), name="tool_yaw_offsets_rad")
    if any(height < 0 for height in heights):
        raise ValueError("Above-goal heights must be non-negative.")


def validate_query_result(result):
    if not isinstance(result, Mapping) or set(result) != RESULT_KEYS:
        raise ValueError("Object path query result must match the exact schema.")
    if (
        result["motion_executed"] is not False
        or result["execution_authorized"] is not False
        or result["input_source"] != "caller_anchor_and_frozen_measured_scene"
        or result["constraint_scope"] != "frozen_scene_full_path"
        or result["ranking_criteria"] != RANKING
    ):
        raise ValueError(
            "A query must not claim execution or fresh object observation."
        )
    count, checked = result["candidate_count"], result["checked_count"]
    if (
        type(count) is not int
        or not 1 <= count <= 64
        or type(checked) is not int
        or not 0 <= checked <= count
        or type(result["all_candidates_checked"]) is not bool
        or result["all_candidates_checked"] != (checked == count)
        or len(result["candidate_diagnostics"]) != checked
    ):
        raise ValueError("Query candidate accounting is inconsistent.")
    for key in ("scene_hash", "source_qpos_hash"):
        value = result[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Query evidence needs content hashes.")
    for key in ("planning_wall_sec", "source_sim_time_sec"):
        value = result[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError("Query evidence time must be finite and nonnegative.")
    targets, index = result["pose_targets_by_component"], result["selected_index"]
    seen, found = set(), {}
    for item in result["candidate_diagnostics"]:
        if not isinstance(item, Mapping):
            raise TypeError("Candidate diagnostic must be an object.")
        i = item.get("candidate_index")
        wall = item.get("wall_sec")
        if (
            type(i) is not int
            or not 0 <= i < count
            or i in seen
            or isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(wall)
            or wall < 0
            or not isinstance(item.get("reason_code"), str)
            or not item["reason_code"]
            or item["reason_code"].strip() != item["reason_code"]
        ):
            raise ValueError("Invalid candidate diagnostic identity or time.")
        seen.add(i)
        if "pose_target_xyz_quat_xyzw" in item:
            pose = finite_vector(item["pose_target_xyz_quat_xyzw"], 7, name="diagnostic endpoint")
            if abs(math.hypot(*pose[3:]) - 1.0) > 1e-5:
                raise ValueError("Diagnostic endpoint quaternion must have unit norm.")
            if i == index and list(pose) != list(targets.get(result["component_id"], ())):
                raise ValueError("Selected endpoint differs from its diagnostic proposal.")
        if item["reason_code"] == "path_found":
            trajectory_hash = item.get("trajectory_hash")
            if (
                not isinstance(trajectory_hash, str)
                or len(trajectory_hash) != 64
                or any(c not in "0123456789abcdef" for c in trajectory_hash)
            ):
                raise ValueError(
                    "Found candidate must identify its checked trajectory."
                )
            cost = item.get("ranking_cost")
            if (
                not isinstance(cost, (list, tuple))
                or len(cost) != 3
                or any(
                    isinstance(x, bool)
                    or not isinstance(x, (int, float))
                    or not math.isfinite(x)
                    or x < 0
                    for x in cost
                )
            ):
                raise ValueError(
                    "Found candidate must report finite nonnegative ranking costs."
                )
            found[i] = tuple(cost)
        elif "ranking_cost" in item or "trajectory_hash" in item:
            raise ValueError(
                "Rejected candidate cannot contain successful path evidence."
            )
    if result["query_status"] == "path_found":
        if (
            type(index) is not int
            or not 0 <= index < count
            or set(targets) != {result["component_id"]}
        ):
            raise ValueError("Selected path must identify exactly one endpoint.")
        if index not in found or found[index] != min(found.values()):
            raise ValueError("Selected path must be the best completed candidate.")
        from agent_gateway.skills.perception.spatial_goal import (
            validate_spatial_goal_result,
        )

        validate_spatial_goal_result(
            {
                "component_id": result["component_id"],
                "reference_frame_id": result["reference_frame_id"],
                "pose_targets_by_component": targets,
                "translation_distance_m": 0.0,
                "rotation_distance_rad": 0.0,
                "axis_alignment_error_rad": None,
                "constraint_scope": "endpoint_only",
                "input_source": "caller_supplied_geometry",
                "path_checked": False,
                "motion_executed": False,
            }
        )
    elif (
        result["query_status"] != "no_path_found"
        or index is not None
        or targets != {}
        or found
    ):
        raise ValueError("No-path query cannot expose a selected endpoint.")


def object_goal_query_schema(*, definition_hash, profile_hash, reference_frame_id):
    def vector(size):
        return {
            "type": "array",
            "minItems": size,
            "maxItems": size,
            "items": {"type": "number"},
        }

    identity = {
        "component_id": {"const": "right_arm", "type": "string"},
        "reference_frame_id": {"const": reference_frame_id, "type": "string"},
    }
    samples = {
        "type": "array",
        "minItems": 1,
        "maxItems": 64,
        "items": {"type": "number"},
    }
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(PARAMETER_KEYS),
        "properties": {
            **identity,
            "anchor_xyz_m": vector(3),
            "heights_m": {**samples, "items": {"type": "number", "minimum": 0}},
            "tool_yaw_offsets_rad": samples,
            "time_budget_sec": {"type": "number", "minimum": 0.1, "maximum": 30},
        },
    }
    properties = {
        **identity,
        "anchor_xyz_m": vector(3),
        "source_tcp_pose_xyz_quat_xyzw": vector(7),
        "query_status": {"enum": ["path_found", "no_path_found"]},
        "scene_hash": {"type": "string"},
        "source_qpos_hash": {"type": "string"},
        "source_sim_time_sec": {"type": "number", "minimum": 0},
        "pose_targets_by_component": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"right_arm": vector(7)},
        },
        "candidate_count": {"type": "integer", "minimum": 1, "maximum": 64},
        "checked_count": {"type": "integer", "minimum": 0, "maximum": 64},
        "all_candidates_checked": {"type": "boolean"},
        "selected_index": {
            "anyOf": [
                {"type": "integer", "minimum": 0, "maximum": 63},
                {"type": "null"},
            ]
        },
        "candidate_diagnostics": {
            "type": "array",
            "maxItems": 64,
            "items": {"type": "object"},
        },
        "planning_wall_sec": {"type": "number", "minimum": 0},
        "ranking_criteria": {"const": RANKING},
        "motion_executed": {"const": False},
        "execution_authorized": {"const": False},
        "input_source": {"const": "caller_anchor_and_frozen_measured_scene"},
        "constraint_scope": {"const": "frozen_scene_full_path"},
    }
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID,
        skill_version="1",
        skill_definition_hash=definition_hash,
        parameter_schema=parameters,
        execution_result_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": sorted(RESULT_KEYS),
        },
        task_evaluation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="object-above-path-query-v1",
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=agent_skill_schema_document_hash(
                    parameters, schema_kind="parameter"
                ),
            ),
        ),
    )

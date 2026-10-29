"""Measured grasp approach for a named resting box-like object.

``perception.query_grasp_approach@1`` is read-only. One request measures the
object's top face and the surface under it from fresh RGB-D (text
segmentation, plane fit, footprint orientation), proposes face-aligned grasps
in a fixed preference order (the reviewed top-down grasp, the same grasp
turned a quarter turn either way, a horizontal grasp from the arm's own side,
a horizontal front grasp, the half-turned top-down grasp) and keeps the first
whose approach path from the current arm state and whose straight final
approach are collision-free. It returns the three tool_tcp poses consumed by
``motion.approach_grasp_pose`` (approach, pregrasp) and
``motion.approach_grasp_contact`` (contact). It never moves the arm, closes
the gripper or grants execution authority.
Semantics: ``agenticrobot.object.face_aligned_grasp_approach.v1``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

SKILL_ID = "perception.query_grasp_approach"
SEMANTICS = "agenticrobot.object.face_aligned_grasp_approach.v1"
APPROACH_FAMILIES = ("top_down", "side", "front")
STATUSES = ("grasp_approach_found", "no_grasp_approach")
REASONS = (
    "collision_free_approach_found",
    "gripper_not_open",
    "object_already_held",
    "object_not_segmented",
    "object_top_not_measured",
    "no_collision_free_approach",
    "query_budget_exhausted",
)
VIEW_OUTCOMES = ("measured", "not_segmented", "top_not_measured")
POSE_KEYS = ("approach_tcp_pose_xyz_quat_xyzw", "pregrasp_tcp_pose_xyz_quat_xyzw",
             "contact_tcp_pose_xyz_quat_xyzw")
PARAMETER_KEYS = frozenset({"object_id", "reference_frame_id"})
RESULT_KEYS = frozenset({
    "semantics", "status", "reason", "object_id", "reference_frame_id", "station_revision",
    *POSE_KEYS, "approach_family", "measured_object", "yaw_offset_rad", "candidate_count",
    "attempted_candidate_count", "selected_candidate_index", "candidate_rejections",
    "path_planning_wall_sec", "views", "motion_executed", "execution_authorized"})
MEASURED_KEYS = frozenset({"role", "language_prompt", "score", "top_center_xyz_m", "yaw_rad",
                           "extent_xy_m", "top_point_count", "support_z_m"})
REJECTION_KEYS = frozenset({"candidate", "reason_code"})
VIEW_KEYS = frozenset({"role", "outcome", "score"})


def definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="1", skill_kind="inspection",
        summary=(
            "Find a collision-checked grasp approach for a named box-like object resting on a "
            "surface. Requires the gripper open and the object not held. The Gateway segments "
            "the object in fresh RGB-D from each bound camera, measures its top face (center, "
            "height, yaw) and the surface under it, and tries face-aligned grasps in a fixed "
            "order: top-down, top-down turned a quarter turn either way, horizontal from the "
            "arm's own side (useful when a post or box blocks the open fingers from above), "
            "horizontal from the front, top-down turned a half turn. It returns "
            "the first whose path from the current arm state and whose straight final approach "
            "are collision-free (approach_family says which) and lists why earlier candidates "
            "were refused. Execute in order: "
            "motion.approach_grasp_pose to approach_tcp_pose_xyz_quat_xyzw, "
            "motion.approach_grasp_pose to pregrasp_tcp_pose_xyz_quat_xyzw, "
            "motion.approach_grasp_contact to contact_tcp_pose_xyz_quat_xyzw, then close the "
            "gripper and read the object state. After a failed or slipped grasp, open the "
            "gripper and query again: the object is measured afresh. Read-only; it never moves "
            "the arm or the gripper, and status=no_grasp_approach explains why."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="object.named_top_down_grasp.v1",
        execution_result_semantics_id=SEMANTICS,
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=("gripper_open_object_resting.v1",
                             "live_rgbd_text_segmentation_available.v1",
                             "collision_checked_planner_available.v1"),
        task_outcome_predicate_ids=("grasp_approach_assessed_without_motion.v1",),
    )


def _pose_schema():
    return {"type": "array", "minItems": 7, "maxItems": 7, "items": {"type": "number"},
            "description": "[x, y, z, qx, qy, qz, qw] tool_tcp pose in reference_frame_id."}


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def schema(*, definition_hash, profile_hash, prompt_by_object_id, reference_frame_id):
    prompts = dict(prompt_by_object_id)
    if not prompts:
        raise ValueError("Grasp queries need at least one deployed object.")
    objects = sorted(prompts)
    described = "; ".join(f"{name}: {prompts[name]}" for name in objects)
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "object_id": {"type": "string", "enum": objects,
                      "description": "Object to grasp (" + described + ")."},
        "reference_frame_id": {"const": reference_frame_id}},
        "required": ["object_id", "reference_frame_id"]}
    xyz = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
    xy = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    measured = {"type": "object", "additionalProperties": False, "properties": {
        "role": {"type": "string"}, "language_prompt": {"type": "string"},
        "score": {"type": "number"}, "top_center_xyz_m": xyz, "yaw_rad": {"type": "number"},
        "extent_xy_m": xy, "top_point_count": {"type": "integer", "minimum": 0},
        "support_z_m": {"type": "number"}},
        "required": sorted(MEASURED_KEYS)}
    result = {"type": "object", "additionalProperties": False, "properties": {
        "semantics": {"const": SEMANTICS},
        "status": {"enum": list(STATUSES)},
        "reason": {"enum": list(REASONS)},
        "object_id": {"type": "string", "enum": objects},
        "reference_frame_id": {"const": reference_frame_id},
        "station_revision": {"type": "integer", "minimum": 0},
        **{key: _nullable(_pose_schema()) for key in POSE_KEYS},
        "approach_family": {"enum": [*APPROACH_FAMILIES, None]},
        "measured_object": _nullable(measured),
        "yaw_offset_rad": {"type": ["number", "null"]},
        "candidate_count": {"type": "integer", "minimum": 0},
        "attempted_candidate_count": {"type": "integer", "minimum": 0},
        "selected_candidate_index": {"type": ["integer", "null"], "minimum": 0},
        "candidate_rejections": {"type": "array", "items": {"type": "object",
            "additionalProperties": False,
            "properties": {"candidate": {"type": "string"}, "reason_code": {"type": "string"}},
            "required": ["candidate", "reason_code"]}},
        "path_planning_wall_sec": {"type": "number", "minimum": 0},
        "views": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"role": {"type": "string"}, "outcome": {"enum": list(VIEW_OUTCOMES)},
                           "score": {"type": ["number", "null"]}},
            "required": sorted(VIEW_KEYS)}},
        "motion_executed": {"const": False},
        "execution_authorized": {"const": False}},
        "required": sorted(RESULT_KEYS)}
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema={"type": "object", "additionalProperties": False,
                                "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="grasp-approach-v1", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


def _finite(value, *, name, low=-math.inf, high=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}].")
    return float(value)


def _vector(value, size, *, name):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in value)):
        raise ValueError(f"{name} must contain {size} finite numbers.")


def _pose(value, *, name):
    _vector(value, 7, name=name)
    if abs(math.hypot(*value[3:]) - 1.) > 1e-6:
        raise ValueError(f"{name} must contain a unit xyzw quaternion.")


def _count(value, *, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def validate_parameters(value, *, object_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != PARAMETER_KEYS:
        raise ValueError("Grasp query parameters have unsupported keys.")
    if value["object_id"] not in tuple(object_ids):
        raise ValueError("Grasp query object is not a deployed grasp target.")
    if value["reference_frame_id"] != reference_frame_id:
        raise ValueError("Grasp query uses another reference frame.")


def validate_result(value, *, object_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != RESULT_KEYS:
        raise ValueError("Grasp query result has unsupported keys.")
    if (value["semantics"] != SEMANTICS or value["status"] not in STATUSES
            or value["reason"] not in REASONS or value["object_id"] not in tuple(object_ids)
            or value["reference_frame_id"] != reference_frame_id
            or value["motion_executed"] is not False or value["execution_authorized"] is not False):
        raise ValueError("Grasp query result identity or status is invalid.")
    _count(value["station_revision"], name="station_revision")
    _count(value["candidate_count"], name="candidate_count")
    _count(value["attempted_candidate_count"], name="attempted_candidate_count")
    _finite(value["path_planning_wall_sec"], name="path_planning_wall_sec", low=0.)
    if value["attempted_candidate_count"] > value["candidate_count"]:
        raise ValueError("Grasp query attempted more candidates than it proposed.")
    found = value["status"] == "grasp_approach_found"
    if found != (value["reason"] == "collision_free_approach_found"):
        raise ValueError("Grasp query status and reason disagree.")
    rejections = value["candidate_rejections"]
    if not isinstance(rejections, (list, tuple)) or len(rejections) > value["attempted_candidate_count"]:
        raise ValueError("Grasp query rejections must list at most the attempted candidates.")
    for item in rejections:
        if (not isinstance(item, Mapping) or set(item) != REJECTION_KEYS
                or not all(isinstance(item[key], str) and item[key] for key in REJECTION_KEYS)):
            raise ValueError("Grasp query rejection entries need a candidate and a reason code.")
    if found:
        if value["approach_family"] not in APPROACH_FAMILIES:
            raise ValueError("A found grasp approach must name its approach family.")
        for key in POSE_KEYS:
            _pose(value[key], name=key)
        if value["approach_family"] == "top_down":
            _finite(value["yaw_offset_rad"], name="yaw_offset_rad", low=-math.pi, high=math.pi)
        elif value["yaw_offset_rad"] is not None:
            raise ValueError("Only a top-down grasp reports a yaw offset.")
        index = value["selected_candidate_index"]
        _count(index, name="selected_candidate_index")
        if index >= value["attempted_candidate_count"]:
            raise ValueError("Selected grasp candidate was not attempted.")
        if value["measured_object"] is None:
            raise ValueError("A found grasp approach needs its object measurement.")
    elif (any(value[key] is not None for key in POSE_KEYS) or value["approach_family"] is not None
          or value["selected_candidate_index"] is not None or value["yaw_offset_rad"] is not None):
        raise ValueError("A rejected grasp query cannot publish approach poses.")
    measured = value["measured_object"]
    if measured is not None:
        if not isinstance(measured, Mapping) or set(measured) != MEASURED_KEYS:
            raise ValueError("Grasp query object measurement has unsupported keys.")
        _finite(measured["score"], name="score", low=0., high=1.)
        _vector(measured["top_center_xyz_m"], 3, name="top_center_xyz_m")
        _vector(measured["extent_xy_m"], 2, name="extent_xy_m")
        _finite(measured["yaw_rad"], name="yaw_rad", low=-math.pi, high=math.pi)
        _count(measured["top_point_count"], name="top_point_count")
        _finite(measured["support_z_m"], name="support_z_m")
        if measured["support_z_m"] >= measured["top_center_xyz_m"][2]:
            raise ValueError("The surface under the object must be below its top face.")
    views = value["views"]
    if not isinstance(views, (list, tuple)):
        raise ValueError("Grasp query views must be a list.")
    for view in views:
        if (not isinstance(view, Mapping) or set(view) != VIEW_KEYS
                or not isinstance(view["role"], str) or view["outcome"] not in VIEW_OUTCOMES):
            raise ValueError("Grasp query view outcome is invalid.")
        if view["score"] is not None:
            _finite(view["score"], name="view score", low=0., high=1.)
    if found and measured["role"] not in [view["role"] for view in views
                                          if view["outcome"] == "measured"]:
        raise ValueError("A found grasp approach must come from a measured view.")


def validate_result_for_parameters(parameters, result):
    if result["object_id"] != parameters["object_id"]:
        raise ValueError("Grasp query result names another object.")

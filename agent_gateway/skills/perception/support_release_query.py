"""Measured release pose for the held object above a named support.

``perception.query_support_release_pose@1`` is read-only. One request measures
the selected support's top face from a fresh RGB-D view (text segmentation of
the support, plane fit and a dense footprint check), reads the verified held
object, proposes on-support release poses and keeps the planner's ranked
choice among those whose complete payload path from the current arm state
is collision-free. The
result is an object-body pose for ``motion.move_grasped_object_to_pose``
(``phase=carry``); the query never moves the robot, opens the gripper or
grants execution authority. Semantics:
``agenticrobot.held_object.support_release_pose.v1``.
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

SKILL_ID = "perception.query_support_release_pose"
SEMANTICS = "agenticrobot.held_object.support_release_pose.v1"
MIN_CLEARANCE_M = 0.004
MAX_CLEARANCE_M = 0.03
DEFAULT_CLEARANCE_M = 0.006
STATUSES = ("release_pose_found", "no_release_pose")
REASONS = (
    "collision_free_release_path_found",
    "held_object_not_verified_lifted",
    "support_not_segmented",
    "support_top_not_measured",
    "held_footprint_does_not_fit_support",
    "no_collision_free_release_path",
    "query_budget_exhausted",
)
PARAMETER_KEYS = frozenset({"support_object_id", "reference_frame_id", "clearance_m"})
RESULT_KEYS = frozenset({
    "semantics", "status", "reason", "support_object_id", "reference_frame_id",
    "station_revision", "release_object_pose_xyz_quat_xyzw", "release_clearance_m",
    "held_object_pose_xyz_quat_xyzw", "yaw_change_rad", "support_top_z_m",
    "support_region_min_xy_m", "support_region_max_xy_m", "segmentation",
    "measured_support", "candidate_count", "attempted_candidate_count",
    "selected_candidate_index", "path_planning_wall_sec", "candidate_rejections", "views",
    "motion_executed", "execution_authorized"})
SEGMENTATION_KEYS = frozenset({"role", "language_prompt", "score", "mask_pixel_count",
                               "mask_bounds_xyxy"})
MEASURED_SUPPORT_KEYS = frozenset({"top_point_count", "plane_residual_m", "plane_tilt_deg",
                                   "footprint_sample_count", "observed_region_min_xy_m",
                                   "observed_region_max_xy_m"})
VIEW_KEYS = frozenset({"role", "outcome", "score", "certified_area_m2"})
# Why planned candidates were refused, e.g. safe_tcp_target_blocked (the
# endpoint collides), safe_tcp_path_search_exhausted (no collision-free path
# within budget), safe_tcp_endpoint_search_exhausted (out of reach).
REJECTIONS_SCHEMA = {"type": "array", "items": {"type": "object", "additionalProperties": False,
    "properties": {"reason_code": {"type": "string"}, "count": {"type": "integer", "minimum": 1}},
    "required": ["count", "reason_code"]}}


def validate_candidate_rejections(value, *, attempted):
    if not isinstance(value, (list, tuple)):
        raise ValueError("Candidate rejections must be a list.")
    total = 0
    for item in value:
        if (not isinstance(item, Mapping) or set(item) != {"reason_code", "count"}
                or not isinstance(item["reason_code"], str) or not item["reason_code"]
                or isinstance(item["count"], bool) or not isinstance(item["count"], int)
                or item["count"] < 1):
            raise ValueError("Candidate rejection entries need a reason code and a positive count.")
        total += item["count"]
    if total > attempted:
        raise ValueError("Candidate rejections exceed the attempted candidates.")
VIEW_OUTCOMES = ("measured", "not_segmented", "top_not_measured")


def definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version="1", skill_kind="inspection",
        summary=(
            "Find a release pose for the object currently held in the gripper above a named "
            "support (box top or mat). Requires the held object to be lifted clear of the table "
            "first (motion.move_grasped_object_to_pose phase=lift). The Gateway segments the "
            "support in fresh RGB-D from each bound camera (the wrist camera sees past the arm), "
            "keeps up to three fully observed top rectangles, largest first (the free strips "
            "beside an object already on the top count separately), proposes on-support "
            "positions (the center, the position nearest the held object and four interior "
            "corners) and yaw turns for the held object, and returns the planner's preferred "
            "one in the first rectangle that has one whose whole payload path from the current "
            "arm state is collision-free. Pass "
            "release_object_pose_xyz_quat_xyzw unchanged to motion.move_grasped_object_to_pose "
            "with phase=carry, then open the gripper with release_support_object_id set to the "
            "same support. Read-only; it never moves the arm or opens the gripper. "
            "status=no_release_pose gives the reason and candidate_rejections counts why "
            "planned poses were refused: safe_tcp_endpoint_search_exhausted means the arm "
            "found no joint solution for those poses from its current configuration (for "
            "example after a high lift near the edge of its reach); safe_tcp_target_blocked "
            "means the release pose itself collides; safe_tcp_path_search_exhausted means no "
            "collision-free path was found in time. Moving the held object lower or nearer the "
            "support with phase=carry and querying again changes all three."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="held_object.named_support_release.v1",
        execution_result_semantics_id=SEMANTICS,
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=("verified_lifted_held_object_available.v1",
                             "live_rgbd_text_segmentation_available.v1",
                             "payload_safe_planner_available.v1"),
        task_outcome_predicate_ids=("support_release_pose_assessed_without_motion.v1",),
    )


def _pose_schema():
    return {"type": "array", "minItems": 7, "maxItems": 7, "items": {"type": "number"},
            "description": "[x, y, z, qx, qy, qz, qw] object-body pose in reference_frame_id."}


def _nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def schema(*, definition_hash, profile_hash, support_object_ids, reference_frame_id):
    supports = list(support_object_ids)
    if not supports or len(set(supports)) != len(supports):
        raise ValueError("Release queries need distinct deployed support IDs.")
    xy = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "support_object_id": {"type": "string", "enum": supports,
                              "description": "Support to release the held object onto."},
        "reference_frame_id": {"const": reference_frame_id},
        "clearance_m": {"type": "number", "minimum": MIN_CLEARANCE_M, "maximum": MAX_CLEARANCE_M,
                        "description": "Gap between the held object's bottom and the support top "
                                       f"at the release pose; default {DEFAULT_CLEARANCE_M}."}},
        "required": ["support_object_id", "reference_frame_id"]}
    segmentation = {"type": "object", "additionalProperties": False, "properties": {
        "role": {"type": "string"}, "language_prompt": {"type": "string"},
        "score": {"type": "number"}, "mask_pixel_count": {"type": "integer", "minimum": 0},
        "mask_bounds_xyxy": {"type": "array", "minItems": 4, "maxItems": 4,
                             "items": {"type": "integer"}}},
        "required": sorted(SEGMENTATION_KEYS)}
    measured = {"type": "object", "additionalProperties": False, "properties": {
        "top_point_count": {"type": "integer", "minimum": 0},
        "plane_residual_m": {"type": "number", "minimum": 0},
        "plane_tilt_deg": {"type": "number", "minimum": 0},
        "footprint_sample_count": {"type": "integer", "minimum": 0},
        "observed_region_min_xy_m": xy, "observed_region_max_xy_m": xy},
        "required": sorted(MEASURED_SUPPORT_KEYS)}
    result = {"type": "object", "additionalProperties": False, "properties": {
        "semantics": {"const": SEMANTICS},
        "status": {"enum": list(STATUSES)},
        "reason": {"enum": list(REASONS)},
        "support_object_id": {"type": "string", "enum": supports},
        "reference_frame_id": {"const": reference_frame_id},
        "station_revision": {"type": "integer", "minimum": 0},
        "release_object_pose_xyz_quat_xyzw": _nullable(_pose_schema()),
        "release_clearance_m": {"type": "number", "minimum": MIN_CLEARANCE_M,
                                "maximum": MAX_CLEARANCE_M},
        "held_object_pose_xyz_quat_xyzw": _nullable(_pose_schema()),
        "yaw_change_rad": {"type": ["number", "null"]},
        "support_top_z_m": {"type": ["number", "null"]},
        "support_region_min_xy_m": _nullable(xy),
        "support_region_max_xy_m": _nullable(xy),
        "segmentation": _nullable(segmentation),
        "measured_support": _nullable(measured),
        "candidate_count": {"type": "integer", "minimum": 0},
        "attempted_candidate_count": {"type": "integer", "minimum": 0},
        "selected_candidate_index": {"type": ["integer", "null"], "minimum": 0},
        "path_planning_wall_sec": {"type": "number", "minimum": 0},
        "candidate_rejections": REJECTIONS_SCHEMA,
        "views": {"type": "array", "items": {"type": "object", "additionalProperties": False,
            "properties": {"role": {"type": "string"}, "outcome": {"enum": list(VIEW_OUTCOMES)},
                           "score": {"type": ["number", "null"]},
                           "certified_area_m2": {"type": ["number", "null"]}},
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
            parameter_profile_id="support-release-pose-v1", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


def _finite(value, *, name, low=-math.inf, high=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}].")
    return float(value)


def _pose(value, *, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 7
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in value)
            or abs(math.hypot(*value[3:]) - 1.) > 1e-6):
        raise ValueError(f"{name} must contain a finite xyz and a unit xyzw quaternion.")


def _xy(value, *, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                   for x in value)):
        raise ValueError(f"{name} must contain two finite numbers.")


def _count(value, *, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def validate_parameters(value, *, support_object_ids, reference_frame_id):
    if not isinstance(value, Mapping) or not {"support_object_id", "reference_frame_id"} <= set(value) \
            or not set(value) <= PARAMETER_KEYS:
        raise ValueError("Release query parameters have unsupported keys.")
    if value["support_object_id"] not in tuple(support_object_ids):
        raise ValueError("Release query support is not a deployed support.")
    if value["reference_frame_id"] != reference_frame_id:
        raise ValueError("Release query uses another reference frame.")
    if "clearance_m" in value:
        _finite(value["clearance_m"], name="clearance_m", low=MIN_CLEARANCE_M, high=MAX_CLEARANCE_M)


def requested_clearance_m(parameters) -> float:
    return float(parameters.get("clearance_m", DEFAULT_CLEARANCE_M))


def validate_result(value, *, support_object_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != RESULT_KEYS:
        raise ValueError("Release query result has unsupported keys.")
    if (value["semantics"] != SEMANTICS or value["status"] not in STATUSES
            or value["reason"] not in REASONS
            or value["support_object_id"] not in tuple(support_object_ids)
            or value["reference_frame_id"] != reference_frame_id
            or value["motion_executed"] is not False or value["execution_authorized"] is not False):
        raise ValueError("Release query result identity or status is invalid.")
    _count(value["station_revision"], name="station_revision")
    _finite(value["release_clearance_m"], name="release_clearance_m",
            low=MIN_CLEARANCE_M, high=MAX_CLEARANCE_M)
    _finite(value["path_planning_wall_sec"], name="path_planning_wall_sec", low=0.)
    _count(value["candidate_count"], name="candidate_count")
    _count(value["attempted_candidate_count"], name="attempted_candidate_count")
    if value["attempted_candidate_count"] > value["candidate_count"]:
        raise ValueError("Release query attempted more candidates than it proposed.")
    found = value["status"] == "release_pose_found"
    if found != (value["reason"] == "collision_free_release_path_found"):
        raise ValueError("Release query status and reason disagree.")
    if found:
        _pose(value["release_object_pose_xyz_quat_xyzw"], name="release_object_pose_xyz_quat_xyzw")
        _pose(value["held_object_pose_xyz_quat_xyzw"], name="held_object_pose_xyz_quat_xyzw")
        _finite(value["yaw_change_rad"], name="yaw_change_rad", low=-math.pi, high=math.pi)
        _finite(value["support_top_z_m"], name="support_top_z_m")
        _xy(value["support_region_min_xy_m"], name="support_region_min_xy_m")
        _xy(value["support_region_max_xy_m"], name="support_region_max_xy_m")
        index = value["selected_candidate_index"]
        _count(index, name="selected_candidate_index")
        if index >= value["attempted_candidate_count"]:
            raise ValueError("Selected release candidate was not attempted.")
        if value["segmentation"] is None or value["measured_support"] is None:
            raise ValueError("A found release pose needs its support measurement evidence.")
        support_z = value["support_top_z_m"]
        if value["release_object_pose_xyz_quat_xyzw"][2] <= support_z:
            raise ValueError("Release pose must be above the measured support top.")
    else:
        if (value["release_object_pose_xyz_quat_xyzw"] is not None
                or value["selected_candidate_index"] is not None
                or value["yaw_change_rad"] is not None):
            raise ValueError("A rejected release query cannot publish a release pose.")
        if value["held_object_pose_xyz_quat_xyzw"] is not None:
            _pose(value["held_object_pose_xyz_quat_xyzw"], name="held_object_pose_xyz_quat_xyzw")
        for name in ("support_region_min_xy_m", "support_region_max_xy_m"):
            if value[name] is not None:
                _xy(value[name], name=name)
        if value["support_top_z_m"] is not None:
            _finite(value["support_top_z_m"], name="support_top_z_m")
    segmentation = value["segmentation"]
    if segmentation is not None:
        if not isinstance(segmentation, Mapping) or set(segmentation) != SEGMENTATION_KEYS:
            raise ValueError("Release query segmentation evidence has unsupported keys.")
        _finite(segmentation["score"], name="segmentation score", low=0., high=1.)
        _count(segmentation["mask_pixel_count"], name="mask_pixel_count")
        bounds = segmentation["mask_bounds_xyxy"]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 4
                or any(isinstance(x, bool) or not isinstance(x, int) for x in bounds)):
            raise ValueError("Segmentation bounds must be four integers.")
    validate_candidate_rejections(value["candidate_rejections"],
                                  attempted=value["attempted_candidate_count"])
    views = value["views"]
    if not isinstance(views, (list, tuple)):
        raise ValueError("Release query views must be a list.")
    for view in views:
        if (not isinstance(view, Mapping) or set(view) != VIEW_KEYS
                or not isinstance(view["role"], str) or view["outcome"] not in VIEW_OUTCOMES):
            raise ValueError("Release query view outcome is invalid.")
        if view["score"] is not None:
            _finite(view["score"], name="view score", low=0., high=1.)
        if (view["certified_area_m2"] is None) != (view["outcome"] != "measured"):
            raise ValueError("Only a measured view reports a certified area.")
        if view["certified_area_m2"] is not None:
            _finite(view["certified_area_m2"], name="certified_area_m2", low=0.)
    if found and (value["segmentation"]["role"] not in
                  [view["role"] for view in views if view["outcome"] == "measured"]):
        raise ValueError("A found release pose must come from a measured view.")
    measured = value["measured_support"]
    if measured is not None:
        if not isinstance(measured, Mapping) or set(measured) != MEASURED_SUPPORT_KEYS:
            raise ValueError("Release query support measurement has unsupported keys.")
        _count(measured["top_point_count"], name="top_point_count")
        _count(measured["footprint_sample_count"], name="footprint_sample_count")
        _finite(measured["plane_residual_m"], name="plane_residual_m", low=0.)
        _finite(measured["plane_tilt_deg"], name="plane_tilt_deg", low=0.)
        _xy(measured["observed_region_min_xy_m"], name="observed_region_min_xy_m")
        _xy(measured["observed_region_max_xy_m"], name="observed_region_max_xy_m")


def validate_result_for_parameters(parameters, result):
    if result["support_object_id"] != parameters["support_object_id"]:
        raise ValueError("Release query result names another support.")
    if abs(result["release_clearance_m"] - requested_clearance_m(parameters)) > 1e-9:
        raise ValueError("Release query result uses another clearance.")

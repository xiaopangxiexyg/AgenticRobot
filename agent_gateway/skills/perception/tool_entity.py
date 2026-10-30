"""Measured gripper-object relations and gripper-entity hover path queries.

Target-agnostic definitions, strict I/O and validators shared by the RealMan
and MuJoCo Gateways. Semantics: ``agenticrobot.relation.tool_object.v1``
(relations) and ``agenticrobot.relation.tool_entity_hover.v1`` (hover).
Both Skills are read-only: they capture, segment, measure and plan, but never
move the robot or grant execution authority.
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

RELATION_SKILL_ID = "perception.describe_tool_object_relations"
HOVER_SKILL_ID = "perception.query_tool_hover_paths"
SEMANTICS_RELATION = "agenticrobot.relation.tool_object.v1"
SEMANTICS_HOVER = "agenticrobot.relation.tool_entity_hover.v1"
RELATION_SCOPE = "one_capture_time_node_no_execution"
MAX_RELATION_OBJECTS = 4
MAX_LABEL_CHARS = 64
MAX_PROMPT_CHARS = 256
MAX_HOVER_CANDIDATES = 8
MAX_HOVER_TIP_OFFSET_M = 0.25
MAX_HOVER_CLEARANCE_M = 0.30
MAX_TOOL_AXIS_TILT_RAD = math.radians(60.)
# Agent-facing order: the general horizontal approaches first, top-down last.
HOVER_ORIENTATION_POLICY_IDS = ("tool_axis_forward", "tool_axis_left", "tool_axis_right", "preserve",
                                "tool_axis_down")
RELATION_CLASSES = ("between_fingers_in_contact_band", "between_fingers", "in_contact_band", "near", "far",
                    "insufficient_surface")
UNMEASURED_REASONS = ("no_mask_above_threshold", "no_valid_depth", "relation_budget_exhausted")
HOVER_STATUSES = ("path_found", "path_rejected", "query_incomplete")
RELATION_PARAMETER_KEYS = frozenset({"objects", "roles", "min_score"})
RELATION_RESULT_KEYS = frozenset({
    "scope", "semantics", "reference_frame_id", "station_revision", "objects", "views",
    "evaluated_sides", "unbound_sides", "gripper_geometry_by_side", "unmeasured", "segments",
    "captures", "evidence_scope", "motion_executed"})
HOVER_PARAMETER_KEYS = frozenset({
    "component_id", "reference_frame_id", "tip_offset_bounds_xy_m", "clearance_bounds_m",
    "orientation_policy", "tool_axis_max_tilt_rad", "max_candidates"})
HOVER_RESULT_KEYS = frozenset({"surface_anchor", "anchor_artifact_id", "hover_query"})
HOVER_QUERY_KEYS = frozenset({
    "component_id", "reference_frame_id", "station_revision", "anchor_xyz_m",
    "source_tcp_pose_xyz_quat_xyzw", "semantics", "status", "reason", "candidate_policy",
    "orientation_policy", "tool_axis_max_tilt_rad", "tip_offset_bounds_xy_m", "clearance_bounds_m",
    "support_window_m", "support_margin_m", "region_budget_sec", "candidate_budget_sec",
    "candidate_count", "attempted_candidate_count", "unresolved_candidate_count", "candidate_attempts",
    "selected_candidate_index", "target_tcp_pose_xyz_quat_xyzw", "planned_endpoint_pose_xyz_quat_xyzw",
    "selected_tip_offset_xy_m", "selected_clearance_m", "achieved", "trajectory_duration_sec",
    "candidate_generation_wall_sec", "path_planning_wall_sec", "tool_entity", "observed_voxel_count",
    "start_clearance", "planning_frame_id", "motion_executed", "execution_authorized"})


def _text(value, *, name, limit):
    if not isinstance(value, str) or not value or value.strip() != value or len(value) > limit:
        raise ValueError(f"{name} must be a trimmed non-empty string of at most {limit} characters.")
    return value


def _number(value, *, name, low=-math.inf, high=math.inf):
    if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            or not low <= value <= high):
        raise ValueError(f"{name} must be a finite number in [{low}, {high}].")
    return float(value)


def _pose(value, *, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 7
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)
            or abs(math.hypot(*value[3:]) - 1.) > 1e-5):
        raise ValueError(f"{name} must contain a finite xyz and unit xyzw quaternion.")
    return [float(x) for x in value]


def _vector(value, size, *, name):
    if (not isinstance(value, (list, tuple)) or len(value) != size
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must contain {size} finite numbers.")
    return [float(x) for x in value]


def _interval(value, *, name, low, high):
    pair = _vector(value, 2, name=name)
    if pair[0] > pair[1] or pair[0] < low or pair[1] > high:
        raise ValueError(f"{name} must be an ordered interval within [{low}, {high}].")
    return pair


def _schema_entry(*, skill_id, definition_hash, profile_id, profile_hash, parameters, result):
    return AgentSkillSchemaEntry(
        skill_id=skill_id, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id=profile_id, parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


# --- perception.describe_tool_object_relations@1 -------------------------------------------------

def relation_definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=RELATION_SKILL_ID, skill_version="1", skill_kind="inspection",
        summary=(
            "Describe where each named object is relative to each bound arm's gripper, from one fresh "
            "stationary RGB-D capture of every selected camera view. The Gateway text-segments each "
            "object per view, keeps the largest measured surface cluster of the eroded mask, places the "
            "gripper solids at that frame's capture joint state, and reports per view and fused per "
            "object: between_fingers[_in_contact_band], in_contact_band, near, far or "
            "insufficient_surface, with robust distances. Read-only proximity evidence; never a "
            "contact, grasp, force, identity or task-success claim."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="text_prompted_objects.camera_roles.v1",
        execution_result_semantics_id=SEMANTICS_RELATION,
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=("live_stationary_rgbd_with_capture_qpos_available.v1",
                             "text_segmentation_available.v1", "reviewed_tool_entity_bound.v1"),
        task_outcome_predicate_ids=("gripper_object_relations_described_without_motion.v1",),
    )


def relation_schema(*, definition_hash, profile_hash, roles, sides, reference_frame_id):
    roles, sides = list(roles), list(sides)
    number = {"type": "number"}
    nullable_number = {"type": ["number", "null"]}
    count = {"type": "integer", "minimum": 0}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "objects": {"type": "array", "minItems": 1, "maxItems": MAX_RELATION_OBJECTS,
            "description": "Objects to describe; one label is one physical object seen from several views.",
            "items": {"type": "object", "additionalProperties": False, "properties": {
                "label": {"type": "string", "minLength": 1, "maxLength": MAX_LABEL_CHARS},
                "language_prompt": {"type": "string", "minLength": 1, "maxLength": MAX_PROMPT_CHARS}},
                "required": ["label", "language_prompt"]}},
        "roles": {"anyOf": [{"type": "array", "minItems": 1, "maxItems": len(roles), "uniqueItems": True,
                             "items": {"type": "string", "enum": roles}}, {"type": "null"}],
                  "description": "Camera roles to use; null selects every bound role."},
        "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }, "required": sorted(RELATION_PARAMETER_KEYS)}
    view = {"type": "object", "properties": {
        "side": {"enum": sides}, "target_reference": {"type": "string"}, "label": {"type": "string"},
        "role": {"enum": roles}, "relation": {"enum": list(RELATION_CLASSES)},
        "surface_distance_m": nullable_number, "sensor_surface_distance_m": nullable_number,
        "contact_band_point_count": count, "sensor_band_point_count": count,
        "grasp_region_point_count": count, "object_point_count": count,
        "gripper_pose_source": {"const": "frame_capture_qpos"},
        "surface_ambiguous": {"type": ["boolean", "null"]}, "surface_cluster": {"type": "object"}}}
    fused = {"type": "object", "properties": {
        "side": {"enum": sides}, "label": {"type": "string"}, "relation": {"enum": list(RELATION_CLASSES)},
        "surface_distance_m": nullable_number, "sensor_surface_distance_m": nullable_number,
        "view_count": count, "usable_view_count": count, "views_agree": {"type": "boolean"},
        "ambiguous_view_count": count}}
    result = {"type": "object", "additionalProperties": False, "properties": {
        "scope": {"const": RELATION_SCOPE}, "semantics": {"const": SEMANTICS_RELATION},
        "reference_frame_id": {"const": reference_frame_id}, "station_revision": count,
        "objects": {"type": "array", "items": fused}, "views": {"type": "array", "items": view},
        "evaluated_sides": {"type": "array", "items": {"enum": sides}},
        "unbound_sides": {"type": "array", "items": {"enum": sides}},
        "gripper_geometry_by_side": {"type": "object"},
        "unmeasured": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "role": {"enum": roles}, "reason": {"enum": list(UNMEASURED_REASONS)}}}},
        "segments": {"type": "array", "items": {"type": "object", "properties": {
            "label": {"type": "string"}, "role": {"enum": roles}, "score": number,
            "mask_pixel_count": count, "mask_bounds_xyxy": {"type": "array"},
            "mask_media_content_hash": {"type": "string"}, "valid_pixel_count": count}}},
        "captures": {"type": "array", "items": {"type": "object"}},
        "evidence_scope": {"type": "string"}, "motion_executed": {"const": False},
    }, "required": sorted(RELATION_RESULT_KEYS)}
    return _schema_entry(skill_id=RELATION_SKILL_ID, definition_hash=definition_hash,
        profile_id="tool-object-relations-v1", profile_hash=profile_hash, parameters=parameters, result=result)


def validate_relation_parameters(value, *, roles, validate_segment_prompt):
    """``validate_segment_prompt(role, language_prompt, min_score)`` applies the bound segmenter's rules."""
    if not isinstance(value, Mapping) or set(value) != RELATION_PARAMETER_KEYS:
        raise ValueError("Relation parameters must be exactly objects, roles and min_score.")
    objects = value["objects"]
    if not isinstance(objects, (list, tuple)) or not 1 <= len(objects) <= MAX_RELATION_OBJECTS:
        raise ValueError(f"objects must list 1 to {MAX_RELATION_OBJECTS} objects.")
    labels = []
    for item in objects:
        if not isinstance(item, Mapping) or set(item) != {"label", "language_prompt"}:
            raise ValueError("Each object needs exactly a label and a language_prompt.")
        labels.append(_text(item["label"], name="label", limit=MAX_LABEL_CHARS))
        _text(item["language_prompt"], name="language_prompt", limit=MAX_PROMPT_CHARS)
    if len(set(labels)) != len(labels):
        raise ValueError("Object labels must be unique; one label is one physical object.")
    selected = value["roles"]
    if selected is not None and (
            not isinstance(selected, (list, tuple)) or not selected or len(set(selected)) != len(selected)
            or not set(selected) <= set(roles)):
        raise ValueError("roles must be null or distinct bound camera roles.")
    _number(value["min_score"], name="min_score", low=0., high=1.)
    for role in (roles if selected is None else selected):
        for item in objects:
            validate_segment_prompt(role, item["language_prompt"], value["min_score"])


def validate_relation_result(value, *, roles, sides, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != RELATION_RESULT_KEYS:
        raise ValueError("Unexpected relation result fields.")
    if (value["scope"] != RELATION_SCOPE or value["semantics"] != SEMANTICS_RELATION
            or value["reference_frame_id"] != reference_frame_id or value["motion_executed"] is not False):
        raise ValueError("A relation result cannot change its scope, semantics, frame or claim motion.")
    revision = value["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Relation result must name its Station revision.")
    evaluated, unbound = list(value["evaluated_sides"]), list(value["unbound_sides"])
    if (sorted(evaluated + unbound) != sorted(sides) or evaluated != sorted(evaluated)
            or unbound != sorted(unbound)):
        raise ValueError("Every arm must be listed as evaluated or unbound.")
    if set(value["gripper_geometry_by_side"]) != set(evaluated):
        raise ValueError("Gripper geometry must be disclosed for every evaluated arm.")
    for item in value["views"]:
        if (not isinstance(item, Mapping) or item.get("side") not in evaluated or item.get("role") not in roles
                or item.get("relation") not in RELATION_CLASSES
                or item.get("gripper_pose_source") != "frame_capture_qpos"):
            raise ValueError("Relation views must use evaluated arms, bound roles and capture poses.")
    for item in value["objects"]:
        if (not isinstance(item, Mapping) or item.get("side") not in evaluated
                or item.get("relation") not in RELATION_CLASSES):
            raise ValueError("Fused relations must use evaluated arms and known classes.")
    for item in value["unmeasured"]:
        if (not isinstance(item, Mapping) or set(item) != {"label", "role", "reason"}
                or item["role"] not in roles or item["reason"] not in UNMEASURED_REASONS):
            raise ValueError("Unmeasured objects must name their label, role and reason.")
    for item in value["captures"]:
        if not isinstance(item, Mapping) or item.get("role") not in roles:
            raise ValueError("Capture provenance must use bound roles.")


def validate_relation_result_for_parameters(parameters, result, *, roles):
    """``roles`` are the registered roles; ``roles: null`` in a request selects all of them."""
    labels = {item["label"] for item in parameters["objects"]}
    requested = set(roles) if parameters["roles"] is None else set(parameters["roles"])
    captured = [item["role"] for item in result["captures"]]
    if len(set(captured)) != len(captured) or set(captured) != requested:
        raise ValueError("Relation result captured other camera roles than requested.")
    for name in ("views", "unmeasured", "segments"):
        for item in result[name]:
            if item["label"] not in labels or item["role"] not in captured:
                raise ValueError("Relation result names an unrequested object or view.")
    for item in result["objects"]:
        if item["label"] not in labels:
            raise ValueError("Relation result fused an unrequested object.")
    measured = {(item["label"], item["role"]) for item in result["views"]}
    missing = {(item["label"], item["role"]) for item in result["unmeasured"]}
    expected = {(label, role) for label in labels for role in captured}
    if measured & missing or measured | missing != expected:
        raise ValueError("Every requested object and view must be measured or listed as unmeasured.")


# --- perception.query_tool_hover_paths@1 -----------------------------------------------------------

def hover_definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=HOVER_SKILL_ID, skill_version="1", skill_kind="inspection",
        summary=(
            "Find a collision-checked path that hovers the bound arm's gripper box above an "
            "authenticated surface anchor: the box tip centre stays within explicit horizontal offsets of "
            "the anchor and the box's lowest point keeps an explicit clearance above the highest observed "
            "point under its footprint. The gripper normally approaches horizontally: tool_axis_forward "
            "turns its approach axis (TCP +Z) toward Station +X, from the robot side (the default); "
            "tool_axis_left / tool_axis_right turn it toward Station +Y / -Y, i.e. the right / left arm "
            "approaching from its own side, often better near the central camera pole, close to the arm or "
            "when the forward approach is rejected. A tilt cone accepts an approach axis within that angle "
            "of the policy axis. preserve keeps the current tool; tool_axis_down points the approach axis "
            "straight down (only when a top-down approach is explicitly required; often out of reach). "
            "\"Above the object\" names a position, not a top-down orientation. The Gateway captures the "
            "observed scene, plans each proposal with the deployment's safe-motion planner, re-measures "
            "the reached pose against the bounds and returns the first certified proposal "
            "(target_tcp_pose_xyz_quat_xyzw) with the pose it reached (planned_endpoint_pose_xyz_quat_xyzw), "
            "for which the bounds hold. Read-only: to execute, submit planned_endpoint_pose_xyz_quat_xyzw "
            "with pose_goal_policy=preserve_target to the safe TCP motion Skill, which replans and checks it "
            "again (the default arrival tolerance can shift the tip by about 2 cm). The surface_anchor "
            "input is the output Artifact of a completed perception.capture_surface_anchor@2 operation in "
            "this session. query_incomplete means the search budget ended first, not that no path exists."),
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="artifact_surface_anchor.tool_entity_hover_bounds.v1",
        execution_result_semantics_id=SEMANTICS_HOVER,
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=("authenticated_surface_anchor_available.v1",
                             "live_stationary_rgbd_with_capture_qpos_available.v1",
                             "reviewed_tool_entity_bound.v1", "safe_motion_planner_bound.v1"),
        task_outcome_predicate_ids=("hover_path_queried_without_motion.v1",),
    )


def hover_schema(*, definition_hash, profile_hash, component_ids, reference_frame_id, anchor_schema):
    components = list(component_ids)
    interval = {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "number"}}
    pose = {"anyOf": [{"type": "array", "minItems": 7, "maxItems": 7, "items": {"type": "number"}},
                      {"type": "null"}]}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "component_id": {"type": "string", "enum": components},
        "reference_frame_id": {"type": "string", "const": reference_frame_id},
        "tip_offset_bounds_xy_m": {"type": "array", "minItems": 2, "maxItems": 2, "items": interval,
            "description": ("Allowed horizontal [x, y] offsets of the gripper box tip centre from the "
                            f"anchor, each an ordered interval within +/-{MAX_HOVER_TIP_OFFSET_M} m.")},
        "clearance_bounds_m": {**interval, "description": (
            "Allowed vertical gap between the box's lowest point and the highest observed point under "
            f"its footprint, an ordered interval within [0, {MAX_HOVER_CLEARANCE_M}] m.")},
        "orientation_policy": {"type": "string", "enum": list(HOVER_ORIENTATION_POLICY_IDS),
            "description": ("tool_axis_forward: approach axis horizontal toward Station +X (from the "
                            "robot side), the default; tool_axis_left / tool_axis_right: approach axis "
                            "horizontal toward +Y / -Y (from the object's right / left side; right_arm "
                            "uses tool_axis_left, left_arm tool_axis_right to approach from its own "
                            "side, e.g. near the central pole); preserve: keep the measured tool "
                            "orientation; tool_axis_down: approach axis straight down, only for an "
                            "explicit top-down request.")},
        "tool_axis_max_tilt_rad": {"type": "number", "minimum": 0.0, "maximum": MAX_TOOL_AXIS_TILT_RAD,
            "description": ("Cone half-angle around the policy axis (the horizontal direction of "
                            "tool_axis_forward/left/right, straight down for tool_axis_down); 0 means "
                            "exactly on the axis and is required for preserve. With a cone, a tool "
                            "already inside it keeps its measured orientation; otherwise it turns only "
                            "to the cone boundary.")},
        "max_candidates": {"type": "integer", "minimum": 1, "maximum": MAX_HOVER_CANDIDATES},
    }, "required": sorted(HOVER_PARAMETER_KEYS)}
    hover_query = {"type": "object", "properties": {
        "component_id": {"enum": components}, "reference_frame_id": {"const": reference_frame_id},
        "status": {"enum": list(HOVER_STATUSES)}, "reason": {"type": "string"},
        "semantics": {"const": SEMANTICS_HOVER},
        "target_tcp_pose_xyz_quat_xyzw": pose, "planned_endpoint_pose_xyz_quat_xyzw": pose,
        "source_tcp_pose_xyz_quat_xyzw": pose,
        "anchor_xyz_m": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}},
        "selected_candidate_index": {"type": ["integer", "null"]},
        "candidate_attempts": {"type": "array", "items": {"type": "object"}},
        "achieved": {"type": ["object", "null"]},
        "motion_executed": {"const": False}, "execution_authorized": {"const": False}},
        "required": sorted(HOVER_QUERY_KEYS)}
    result = {"type": "object", "additionalProperties": False, "properties": {
        "surface_anchor": anchor_schema, "anchor_artifact_id": {"type": "string", "minLength": 1},
        "hover_query": hover_query}, "required": sorted(HOVER_RESULT_KEYS)}
    return _schema_entry(skill_id=HOVER_SKILL_ID, definition_hash=definition_hash,
        profile_id="tool-entity-hover-paths-v1", profile_hash=profile_hash, parameters=parameters, result=result)


def validate_hover_parameters(value, *, component_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != HOVER_PARAMETER_KEYS:
        raise ValueError("Hover parameters must match the tool-entity hover schema.")
    if value["component_id"] not in component_ids or value["reference_frame_id"] != reference_frame_id:
        raise ValueError("Hover component/frame differs from the deployment binding.")
    bounds = value["tip_offset_bounds_xy_m"]
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise ValueError("tip_offset_bounds_xy_m must contain X and Y intervals.")
    for interval, name in zip(bounds, ("tip X offset", "tip Y offset")):
        _interval(interval, name=name, low=-MAX_HOVER_TIP_OFFSET_M, high=MAX_HOVER_TIP_OFFSET_M)
    _interval(value["clearance_bounds_m"], name="clearance_bounds_m", low=0., high=MAX_HOVER_CLEARANCE_M)
    if value["orientation_policy"] not in HOVER_ORIENTATION_POLICY_IDS:
        raise ValueError("orientation_policy must be one of " + ", ".join(HOVER_ORIENTATION_POLICY_IDS) + ".")
    tilt = _number(value["tool_axis_max_tilt_rad"], name="tool_axis_max_tilt_rad", low=0.,
                   high=MAX_TOOL_AXIS_TILT_RAD)
    if value["orientation_policy"] == "preserve" and tilt:
        raise ValueError("A tool-axis tilt cone applies only to the tool-axis policies, not preserve.")
    count = value["max_candidates"]
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_HOVER_CANDIDATES:
        raise ValueError(f"max_candidates must be an integer from 1 to {MAX_HOVER_CANDIDATES}.")


def validate_hover_query(value, *, component_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != HOVER_QUERY_KEYS:
        raise ValueError("Unexpected hover query fields.")
    if (value["component_id"] not in component_ids or value["reference_frame_id"] != reference_frame_id
            or value["semantics"] != SEMANTICS_HOVER):
        raise ValueError("Hover query changed its component, frame or semantics.")
    if value["motion_executed"] is not False or value["execution_authorized"] is not False:
        raise ValueError("A hover query never executes or authorizes motion.")
    if value["status"] not in HOVER_STATUSES or not isinstance(value["reason"], str) or not value["reason"]:
        raise ValueError("Hover query needs a known status and a reason.")
    revision = value["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Hover query must name its Station revision.")
    _vector(value["anchor_xyz_m"], 3, name="anchor_xyz_m")
    _pose(value["source_tcp_pose_xyz_quat_xyzw"], name="source_tcp_pose_xyz_quat_xyzw")
    found = value["status"] == "path_found"
    for key in ("target_tcp_pose_xyz_quat_xyzw", "planned_endpoint_pose_xyz_quat_xyzw"):
        if found:
            _pose(value[key], name=key)
        elif value[key] is not None:
            raise ValueError("Only a found hover path carries target poses.")
    index = value["selected_candidate_index"]
    if found:
        if (isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < value["candidate_count"]
                or not isinstance(value["achieved"], Mapping)
                or value["achieved"].get("inside_requested_region") is not True):
            raise ValueError("A found hover path must name its certified candidate.")
    elif index is not None or value["achieved"] is not None:
        raise ValueError("An unfound hover path cannot name a selected candidate.")


def validate_hover_result_for_parameters(parameters, result):
    """Bind the result to its request; stored results may hold tuples instead of lists."""
    def nested(value):
        return [[float(x) for x in item] for item in value]

    query = result["hover_query"]
    for key in ("component_id", "reference_frame_id", "orientation_policy"):
        if query[key] != parameters[key]:
            raise ValueError("Hover query changed its requested component, frame or orientation.")
    if (nested(query["tip_offset_bounds_xy_m"]) != nested(parameters["tip_offset_bounds_xy_m"])
            or [float(x) for x in query["clearance_bounds_m"]] != [float(x) for x in parameters["clearance_bounds_m"]]
            or not math.isclose(query["tool_axis_max_tilt_rad"], parameters["tool_axis_max_tilt_rad"],
                                abs_tol=1e-12, rel_tol=0.)
            or query["candidate_count"] > parameters["max_candidates"]):
        raise ValueError("Hover query changed its requested bounds.")
    measurement = result["surface_anchor"]["measurement"]
    if [float(x) for x in query["anchor_xyz_m"]] != [float(x) for x in measurement["world_xyz_m"]]:
        raise ValueError("Hover query changed its measured anchor.")
    if query["station_revision"] != measurement["station_revision"]:
        raise ValueError("Hover query and its anchor belong to different Station revisions.")


__all__ = [
    "HOVER_ORIENTATION_POLICY_IDS",
    "HOVER_PARAMETER_KEYS",
    "HOVER_QUERY_KEYS",
    "HOVER_RESULT_KEYS",
    "HOVER_SKILL_ID",
    "HOVER_STATUSES",
    "MAX_RELATION_OBJECTS",
    "RELATION_CLASSES",
    "RELATION_PARAMETER_KEYS",
    "RELATION_RESULT_KEYS",
    "RELATION_SCOPE",
    "RELATION_SKILL_ID",
    "SEMANTICS_HOVER",
    "SEMANTICS_RELATION",
    "UNMEASURED_REASONS",
    "hover_definition",
    "hover_schema",
    "relation_definition",
    "relation_schema",
    "validate_hover_parameters",
    "validate_hover_query",
    "validate_hover_result_for_parameters",
    "validate_relation_parameters",
    "validate_relation_result",
    "validate_relation_result_for_parameters",
]

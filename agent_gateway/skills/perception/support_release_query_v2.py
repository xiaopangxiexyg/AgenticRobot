"""Measured release and pre-place poses for the held object above a named support.

``perception.query_support_release_pose@2`` is read-only, like version 1. It
returns two object-body poses for ``motion.move_grasped_object_to_pose``: a
pre-place pose straight above the release pose (``phase=carry``) and the
release pose itself (``phase=place``). The split serves deployments whose
place motion measures the support again from directly above before it lowers
the object (the RealMan place requires the holding arm's wrist camera to see
the support under the target). Both legs are checked with the deployment's
payload planner; nothing is moved, opened or authorized. Semantics:
``agenticrobot.held_object.support_release_pose.v2``.

Version 1 (``support_release_query``) is unchanged: one release pose for
``phase=carry`` followed by a named-support gripper release.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from . import support_release_query as v1

SKILL_ID = v1.SKILL_ID
SKILL_VERSION = "2"
SEMANTICS = "agenticrobot.held_object.support_release_pose.v2"
# Gap between the held object's bottom and the measured support at the release
# pose. A place motion that re-measures the support accepts a band around it;
# the default sits in the middle of the RealMan 0-2 cm band.
MIN_CLEARANCE_M = 0.005
MAX_CLEARANCE_M = 0.015
DEFAULT_CLEARANCE_M = 0.01
# The pre-place pose is straight above the release pose by a deployment-bound
# height in (0, MAX_PREPLACE_HEIGHT_M].
MAX_PREPLACE_HEIGHT_M = 0.15
MAX_RELEASE_LEAN_RAD = v1.MAX_RELEASE_LEAN_RAD
STATUSES = v1.STATUSES
REASONS = v1.REASONS
PARAMETER_KEYS = v1.PARAMETER_KEYS
PREPLACE_KEY = "preplace_object_pose_xyz_quat_xyzw"
RESULT_KEYS = v1.RESULT_KEYS | {PREPLACE_KEY}
VIEW_OUTCOMES = v1.VIEW_OUTCOMES
_POSE_ATOL = 1e-9

SUMMARY = (
    "Find where to put down the object held in the gripper on a named support (a box top or mat). "
    "Requires a verified grip that was lifted first (motion.move_grasped_object_to_pose phase=lift). "
    "The Gateway segments the support in fresh RGB-D from each bound camera (the holding arm's wrist "
    "camera first), keeps up to three fully observed top rectangles, largest first, proposes positions "
    "on them (the center, the position nearest the held object and four interior corners) with yaw "
    "turns of the held orientation about the support's normal, and checks two legs for each with the "
    "same payload planner as the motion: a carry from the current pose to a pre-place pose straight "
    "above the release pose, then a place that lowers the object to the release pose with "
    "clearance_m between its bottom and the support top. If no candidate at the held orientation "
    "passes, the same positions are tried at the current yaw with the object turned a few degrees "
    "further, top toward the arm (never more than 10 degrees from the support's normal); "
    "release_lean_rad reports that added lean, 0 otherwise. To use the result: "
    "motion.move_grasped_object_to_pose phase=carry with object_pose_xyz_quat_xyzw = "
    "preplace_object_pose_xyz_quat_xyzw, then phase=place with object_pose_xyz_quat_xyzw = "
    "release_object_pose_xyz_quat_xyzw (both unchanged), then open the gripper and lift the empty "
    "gripper. Read-only; it never moves the arm or the gripper. status=no_release_pose gives the "
    "reason; candidate_rejections counts why checked poses were refused, prefixed by the leg "
    "(carry: or place:), e.g. ik_search_exhausted (out of reach from here), endpoint_collision_blocked "
    "or path_search_exhausted. support_top_not_measured or a place: support refusal usually means the "
    "support is not seen well: carry the object nearer and above the support and query again.")


def definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=SKILL_ID, skill_version=SKILL_VERSION, skill_kind="inspection",
        summary=SUMMARY,
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="held_object.named_support_release.v2",
        execution_result_semantics_id=SEMANTICS,
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=("verified_lifted_held_object_available.v1",
                             "live_rgbd_text_segmentation_available.v1",
                             "payload_safe_planner_available.v1"),
        task_outcome_predicate_ids=("support_release_pose_assessed_without_motion.v1",),
    )


def _thawed(value):
    """A mutable JSON copy of a frozen schema document (read-only mappings, tuples)."""
    if isinstance(value, Mapping):
        return {key: _thawed(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thawed(item) for item in value]
    return copy.copy(value)


def schema(*, definition_hash, profile_hash, support_object_ids, reference_frame_id):
    base = v1.schema(definition_hash=definition_hash, profile_hash=profile_hash,
                     support_object_ids=support_object_ids, reference_frame_id=reference_frame_id)
    parameters = _thawed(base.parameter_schema)
    parameters["properties"]["clearance_m"] = {
        "type": "number", "minimum": MIN_CLEARANCE_M, "maximum": MAX_CLEARANCE_M,
        "description": ("Gap between the held object's bottom and the support top at the release "
                        f"pose; default {DEFAULT_CLEARANCE_M}.")}
    result = _thawed(base.execution_result_schema)
    properties = result["properties"]
    properties["semantics"] = {"const": SEMANTICS}
    properties["release_clearance_m"] = {"type": "number", "minimum": MIN_CLEARANCE_M,
                                         "maximum": MAX_CLEARANCE_M}
    properties["release_object_pose_xyz_quat_xyzw"] = v1._nullable({
        **v1._pose_schema(), "description": ("[x, y, z, qx, qy, qz, qw] object-body pose in "
                                            "reference_frame_id; submit with phase=place.")})
    properties[PREPLACE_KEY] = v1._nullable({
        **v1._pose_schema(), "description": ("Object-body pose straight above the release pose with the "
                                            "same orientation; submit with phase=carry first.")})
    result["required"] = sorted(RESULT_KEYS)
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version=SKILL_VERSION, skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema=_thawed(base.task_evaluation_schema),
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="support-release-pose-v2", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


def validate_parameters(value, *, support_object_ids, reference_frame_id):
    if (not isinstance(value, Mapping) or not {"support_object_id", "reference_frame_id"} <= set(value)
            or not set(value) <= PARAMETER_KEYS):
        raise ValueError("Release query parameters have unsupported keys.")
    if value["support_object_id"] not in tuple(support_object_ids):
        raise ValueError("Release query support is not a deployed support.")
    if value["reference_frame_id"] != reference_frame_id:
        raise ValueError("Release query uses another reference frame.")
    if "clearance_m" in value:
        v1._finite(value["clearance_m"], name="clearance_m", low=MIN_CLEARANCE_M, high=MAX_CLEARANCE_M)


def requested_clearance_m(parameters) -> float:
    return float(parameters.get("clearance_m", DEFAULT_CLEARANCE_M))


def validate_result(value, *, support_object_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != RESULT_KEYS:
        raise ValueError("Release query result has unsupported keys.")
    if value["semantics"] != SEMANTICS:
        raise ValueError("Release query result identity or status is invalid.")
    # Everything version 1 checks still holds; only the semantics, the clearance
    # band and the pre-place pose differ.
    subset = {key: value[key] for key in v1.RESULT_KEYS}
    subset["semantics"] = v1.SEMANTICS
    v1.validate_result(subset, support_object_ids=support_object_ids, reference_frame_id=reference_frame_id)
    v1._finite(value["release_clearance_m"], name="release_clearance_m",
               low=MIN_CLEARANCE_M, high=MAX_CLEARANCE_M)
    preplace = value[PREPLACE_KEY]
    if value["status"] != "release_pose_found":
        if preplace is not None:
            raise ValueError("A rejected release query cannot publish a pre-place pose.")
        return
    v1._pose(preplace, name=PREPLACE_KEY)
    release = value["release_object_pose_xyz_quat_xyzw"]
    rise = float(preplace[2]) - float(release[2])
    same_axis = all(abs(float(preplace[i]) - float(release[i])) <= _POSE_ATOL for i in (0, 1))
    same_orientation = all(abs(float(preplace[i]) - float(release[i])) <= _POSE_ATOL for i in range(3, 7))
    if not (same_axis and same_orientation and 0. < rise <= MAX_PREPLACE_HEIGHT_M + _POSE_ATOL):
        raise ValueError("The pre-place pose must be straight above the release pose with its orientation.")


def validate_result_for_parameters(parameters, result):
    if result["support_object_id"] != parameters["support_object_id"]:
        raise ValueError("Release query result names another support.")
    if abs(result["release_clearance_m"] - requested_clearance_m(parameters)) > 1e-9:
        raise ValueError("Release query result uses another clearance.")


__all__ = ["DEFAULT_CLEARANCE_M", "MAX_CLEARANCE_M", "MAX_PREPLACE_HEIGHT_M", "MAX_RELEASE_LEAN_RAD",
           "MIN_CLEARANCE_M", "PREPLACE_KEY", "REASONS", "RESULT_KEYS", "SEMANTICS", "SKILL_ID",
           "SKILL_VERSION", "STATUSES", "definition", "requested_clearance_m", "schema",
           "validate_parameters", "validate_result", "validate_result_for_parameters"]

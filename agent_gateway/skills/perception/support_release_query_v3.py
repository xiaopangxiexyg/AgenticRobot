"""One release pose with the selected support's measured orientation context.

Version 3 retains version 1's carry-then-named-release workflow. Its additional
angles describe the selected support, the held observation used to propose
candidates, and the selected release proposal. They are not execution or
stable-placement evidence. Version 1 and the two-leg version 2 are unchanged.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
)

from . import support_release_query as v1

SKILL_ID = v1.SKILL_ID
SKILL_VERSION = "3"
SEMANTICS = "agenticrobot.held_object.support_release_pose.v3"
MIN_CLEARANCE_M = v1.MIN_CLEARANCE_M
MAX_CLEARANCE_M = v1.MAX_CLEARANCE_M
DEFAULT_CLEARANCE_M = v1.DEFAULT_CLEARANCE_M
MAX_RELEASE_LEAN_RAD = v1.MAX_RELEASE_LEAN_RAD
STATUSES = v1.STATUSES
REASONS = v1.REASONS
PARAMETER_KEYS = v1.PARAMETER_KEYS
ANGLE_KEYS = frozenset({
    "support_tilt_deg", "held_tilt_from_support_normal_rad",
    "release_tilt_from_support_normal_rad", "release_leveling_rad",
})
RESULT_KEYS = v1.RESULT_KEYS | ANGLE_KEYS
VIEW_OUTCOMES = v1.VIEW_OUTCOMES

SUMMARY = (
    "Find a release pose for the verified, lifted held object above a named support. "
    "Fresh RGB-D text segmentation supplies up to three fully observed support regions, "
    "largest first. The existing payload planner checks bounded position and yaw candidates; "
    "inclined regions first try leveling to the support. An additional lean toward the arm "
    "is tried when the preceding passes find no path. Return the planner's preferred candidate "
    "in the first region with a collision-free payload path. Pass "
    "release_object_pose_xyz_quat_xyzw unchanged to motion.move_grasped_object_to_pose with "
    "phase=carry, then open the gripper with release_support_object_id set to the same support. "
    "Read-only; nothing is moved, opened or authorized. "
    "For status=release_pose_found, measured_support describes the selected region. "
    "support_tilt_deg is its upward normal's angle from reference-frame +Z, in degrees. "
    "held_tilt_from_support_normal_rad and release_tilt_from_support_normal_rad are the "
    "angles between that normal and the closest unoriented axis of the measured object box, "
    "at the reported held pose and selected release proposal respectively. "
    "release_leveling_rad is the candidate's leveling turn after its yaw turn; "
    "release_lean_rad remains the separate added lean. All four new angles are null when "
    "no release is selected. status=no_release_pose gives the reason and candidate_rejections "
    "counts planner refusals. Angles describe measured inputs and a proposal, not the achieved "
    "orientation or stable placement; release must still be observed after opening.")


def definition() -> SkillDefinition:
    base = v1.definition()
    return replace(
        base, skill_version=SKILL_VERSION, execution_result_semantics_id=SEMANTICS,
        summary=SUMMARY,
    )


def schema(*, definition_hash, profile_hash, support_object_ids, reference_frame_id):
    base = v1.schema(
        definition_hash=definition_hash, profile_hash=profile_hash,
        support_object_ids=support_object_ids, reference_frame_id=reference_frame_id)
    result = dict(base.execution_result_schema)
    properties = dict(result["properties"])
    properties["semantics"] = {"const": SEMANTICS}
    properties["support_tilt_deg"] = {
        "type": ["number", "null"], "minimum": 0., "maximum": 90.,
        "description": "Selected support region's upward-normal angle from reference-frame +Z, "
                       "in degrees; null without a selected release."}
    for key, pose in (("held_tilt_from_support_normal_rad", "reported held observation"),
                      ("release_tilt_from_support_normal_rad", "selected release proposal")):
        properties[key] = {
            "type": ["number", "null"], "minimum": 0., "maximum": math.pi / 2.,
            "description": "Angle in radians between the selected support normal and the closest "
                           f"unoriented measured-box axis at the {pose}; "
                           "null without a selected release."}
    properties["release_leveling_rad"] = {
        "type": ["number", "null"], "minimum": 0., "maximum": math.pi / 2.,
        "description": "Selected candidate's turn aligning the nearest box axis with the support "
                       "normal after its yaw turn; 0 when no leveling was applied, "
                       "null without a selected release."}
    result["properties"] = properties
    result["required"] = sorted(RESULT_KEYS)
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version=SKILL_VERSION, skill_definition_hash=definition_hash,
        parameter_schema=base.parameter_schema, execution_result_schema=result,
        task_evaluation_schema=base.task_evaluation_schema,
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="support-release-pose-v3", parameter_profile_hash=profile_hash,
            parameter_schema_hash=base.parameter_schema_hash),))


validate_parameters = v1.validate_parameters
requested_clearance_m = v1.requested_clearance_m
validate_result_for_parameters = v1.validate_result_for_parameters


def validate_result(value, *, support_object_ids, reference_frame_id):
    if not isinstance(value, Mapping) or set(value) != RESULT_KEYS:
        raise ValueError("Release query result has unsupported keys.")
    if value["semantics"] != SEMANTICS:
        raise ValueError("Release query result identity is invalid.")
    base = {key: value[key] for key in v1.RESULT_KEYS}
    base["semantics"] = v1.SEMANTICS
    v1.validate_result(base, support_object_ids=support_object_ids,
                       reference_frame_id=reference_frame_id)
    if value["status"] != "release_pose_found":
        if any(value[key] is not None for key in ANGLE_KEYS):
            raise ValueError("A rejected release query has no selected support or release angles.")
        return
    for key in ANGLE_KEYS:
        v1._finite(value[key], name=key, low=0.,
                   high=90. if key == "support_tilt_deg" else math.pi / 2.)


__all__ = ["ANGLE_KEYS", "DEFAULT_CLEARANCE_M", "MAX_CLEARANCE_M", "MAX_RELEASE_LEAN_RAD",
           "MIN_CLEARANCE_M", "PARAMETER_KEYS", "REASONS", "RESULT_KEYS", "SEMANTICS",
           "SKILL_ID", "SKILL_VERSION", "STATUSES", "definition", "requested_clearance_m",
           "schema", "validate_parameters", "validate_result", "validate_result_for_parameters"]

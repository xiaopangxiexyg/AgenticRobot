"""Leveled release proposals for the explicit yaw-free object-motion v3 phase."""
from dataclasses import replace
from collections.abc import Mapping

from . import support_release_query_v3 as v3
from .support_release_query_v3 import (
    ANGLE_KEYS, DEFAULT_CLEARANCE_M, MAX_CLEARANCE_M, MAX_RELEASE_LEAN_RAD,
    MIN_CLEARANCE_M, PARAMETER_KEYS, REASONS, RESULT_KEYS, STATUSES,
    requested_clearance_m, validate_parameters, validate_result_for_parameters,
)

SKILL_ID = v3.SKILL_ID
SKILL_VERSION = "4"
SEMANTICS = "agenticrobot.held_object.leveled_release_approach.v4"
SUMMARY = (
    "Find a leveled release pose for the verified lifted object above a named support. "
    "Fresh RGB-D regions and the actual hand-object relation determine the object goal. "
    "Try collision-checked position and yaw candidates with a box face aligned to the "
    "selected support normal; do not preserve an unsuitable in-hand lean as a fallback. "
    "Pass release_object_pose_xyz_quat_xyzw unchanged to motion.move_grasped_object_to_pose@3 "
    "with phase=release_approach and support_object_id set to this support. This explicit "
    "unsupported approach permits bounded inclination change toward the support, while "
    "ordinary lift/carry keep the grasped inclination and leave yaw free. Then open the "
    "gripper with the same release_support_object_id and read fresh stable object state. "
    "The returned angles describe the measured input and planned goal, not achieved "
    "placement. Read-only: no arm, gripper or HOLD command is performed."
)


def definition():
    return replace(v3.definition(), skill_version=SKILL_VERSION, summary=SUMMARY,
        execution_result_semantics_id=SEMANTICS)


def schema(**kwargs):
    base = v3.schema(**kwargs)
    result = dict(base.execution_result_schema)
    properties = dict(result["properties"])
    properties["semantics"] = {"const": SEMANTICS}
    result["properties"] = properties
    return replace(base, skill_version=SKILL_VERSION, execution_result_schema=result,
        parameter_profiles=tuple(replace(item, parameter_profile_id="support-release-pose-v4")
                                 for item in base.parameter_profiles))


def validate_result(value, **kwargs):
    if not isinstance(value, Mapping) or value.get("semantics") != SEMANTICS:
        raise ValueError("Release query v4 result identity is invalid.")
    v3.validate_result({**value, "semantics": v3.SEMANTICS}, **kwargs)

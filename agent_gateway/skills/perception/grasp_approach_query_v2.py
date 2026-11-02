"""Measured grasp approach with an optional, exclusive grasp-family choice.

Version 1 keeps its original parameters and automatic preference order.
Version 2 can restrict that same search to top-down, side or front grasps.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from . import grasp_approach_query as v1

SKILL_ID = v1.SKILL_ID
SKILL_VERSION = "2"
SEMANTICS = "agenticrobot.object.face_aligned_grasp_approach.v2"
APPROACH_FAMILIES = v1.APPROACH_FAMILIES
PARAMETER_KEYS = v1.PARAMETER_KEYS | {"preferred_family"}
RESULT_KEYS = v1.RESULT_KEYS | {"preferred_family"}
REASONS = (*v1.REASONS, "preferred_family_unavailable")
SUMMARY = (
    "Find a collision-checked grasp approach for a named resting box-like object. Requires the "
    "gripper open and the object not held. Fresh RGB-D and text segmentation measure its top "
    "face and the surface below it. Omit preferred_family to try the reviewed order: top-down, "
    "top-down turned a quarter turn either way, horizontal from the arm's own side, horizontal "
    "from the front, then top-down turned a half turn. Set preferred_family to top_down, side "
    "or front to try ONLY that family, keeping its original candidate order; there is no "
    "fallback to another family. Side/front require a measured underlying surface and enough "
    "object height for the pads to overlap while the hand clears the surface. "
    "Returns the first approach whose path, straight contact approach and pre-contact closure "
    "obstacle check pass, with earlier candidate refusals. Actual closure and lifting still "
    "require their own execution checks. Execute motion.approach_grasp_pose to "
    "approach_tcp_pose_xyz_quat_xyzw, then to pregrasp_tcp_pose_xyz_quat_xyzw, then "
    "motion.approach_grasp_contact to contact_tcp_pose_xyz_quat_xyzw, close the gripper and "
    "read the object state. After a failed or slipped grasp, open the gripper and query again "
    "for a fresh measurement. Read-only; it never moves the arm or gripper. "
    "status=no_grasp_approach explains the refusal. preferred_family_unavailable means the "
    "measurement cannot propose that family; no_collision_free_approach means no proposed "
    "candidate passed. candidate_count and selected_candidate_index refer to the searched "
    "family when one is specified, otherwise the automatic list."
)


def definition():
    return replace(v1.definition(), skill_version=SKILL_VERSION, summary=SUMMARY,
                   parameter_semantics_id="object.named_face_aligned_grasp.v2",
                   execution_result_semantics_id=SEMANTICS)


def schema(*, definition_hash, profile_hash, prompt_by_object_id, reference_frame_id):
    base = v1.schema(definition_hash=definition_hash, profile_hash=profile_hash,
                     prompt_by_object_id=prompt_by_object_id, reference_frame_id=reference_frame_id)
    parameters = {**base.parameter_schema, "properties": {
        **base.parameter_schema["properties"],
        "preferred_family": {"enum": list(APPROACH_FAMILIES), "description":
            "Optional exclusive family restriction. Omit for automatic selection; no cross-family fallback."},
    }}
    result = {**base.execution_result_schema, "properties": {
        **base.execution_result_schema["properties"],
        "semantics": {"const": SEMANTICS}, "reason": {"enum": list(REASONS)},
        "preferred_family": {"enum": [*APPROACH_FAMILIES, None], "description":
            "The requested family, or null for automatic selection."},
    }, "required": sorted(RESULT_KEYS)}
    return AgentSkillSchemaEntry(
        skill_id=SKILL_ID, skill_version=SKILL_VERSION, skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema=base.task_evaluation_schema,
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="grasp-approach-v2", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


def validate_parameters(value, *, object_ids, reference_frame_id):
    if (not isinstance(value, Mapping) or not v1.PARAMETER_KEYS <= set(value)
            or not set(value) <= PARAMETER_KEYS):
        raise ValueError("Grasp query v2 parameters have unsupported or missing keys.")
    v1.validate_parameters({key: value[key] for key in v1.PARAMETER_KEYS},
                           object_ids=object_ids, reference_frame_id=reference_frame_id)
    if "preferred_family" in value and value["preferred_family"] not in APPROACH_FAMILIES:
        raise ValueError("preferred_family must be top_down, side or front, or omitted.")


def validate_result(value, *, object_ids, reference_frame_id):
    if (not isinstance(value, Mapping) or set(value) != RESULT_KEYS
            or value["semantics"] != SEMANTICS
            or value["preferred_family"] not in (*APPROACH_FAMILIES, None)):
        raise ValueError("Grasp query v2 result identity or keys are invalid.")
    base = {key: value[key] for key in v1.RESULT_KEYS}
    base["semantics"] = v1.SEMANTICS
    if value["reason"] == "preferred_family_unavailable":
        if (value["preferred_family"] is None or value["candidate_count"] != 0
                or value["measured_object"] is None):
            raise ValueError("An unavailable requested family needs a measurement and zero candidates.")
        base["reason"] = "no_collision_free_approach"
    v1.validate_result(base, object_ids=object_ids, reference_frame_id=reference_frame_id)
    if (value["status"] == "grasp_approach_found" and value["preferred_family"] is not None
            and value["approach_family"] != value["preferred_family"]):
        raise ValueError("The found grasp must belong to the requested family.")


def validate_result_for_parameters(parameters, result):
    v1.validate_result_for_parameters(parameters, result)
    if result["preferred_family"] != parameters.get("preferred_family"):
        raise ValueError("Grasp query result names another requested family.")

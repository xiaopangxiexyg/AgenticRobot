"""Explicit v3: generate relation endpoints and check both stages once."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.perception.object_goal_query import (
    RELATION_FIELDS,
    validate_relation_query_parameters,
)
from agent_gateway.skills.perception.tcp_continuation_query import (
    continuation_query_definition,
    validate_continuation_parameters,
)

PARAMETER_KEYS = RELATION_FIELDS | {"continuation_delta_xyz_m", "path_constraint"}
POSES_KEY = "first_pose_candidates_xyz_quat_xyzw"


def relation_continuation_definition():
    return replace(continuation_query_definition(), skill_version="3",
        parameter_semantics_id="artifact_relation_continuation_query.v3",
        execution_result_semantics_id="anchored_relation_two_stage_evidence.v3",
        summary=("Generate 1..8 relation offset/rotation combinations from one authenticated surface_anchor v2 "
            "Artifact and the freshly frozen measured TCP; check each complete two-stage path in that same scene. "
            "Do not run query_relation_paths first or transcribe endpoint poses. Relations use reference or original "
            "input-tool offsets, preserve orientation or align a tool axis with allowed rotations about that axis. "
            "The relation/orientation sampling constrains endpoints only. Propagate the planned first endpoint qpos "
            "and FK into the second stage; add continuation_delta_xyz_m in the public reference frame, preserving "
            "its predicted quaternion. An optional preserve_start_tool_axis cone applies only to second-stage "
            "sampled references. The 0.1..30 second budget covers all candidates and both stages; each path search "
            "gets at most 3 seconds and the remaining budget. Rank completed chains by maximum joint excursion, "
            "duration, then total joint travel. Return exact generated first_pose_candidates_xyz_quat_xyzw, partial "
            "attempt accounting and selected targets. Budget exhaustion is not infeasibility. No motion, execution "
            "authority, grasp success or globally optimal/continuous-path guarantee. Replan each actual safe move "
            "from fresh measurements; compute the actual followup from measured arrival, not this prediction."))


def explicit_continuation_parameters(parameters, poses):
    return {**{key: parameters[key] for key in ("component_id", "reference_frame_id",
        "continuation_delta_xyz_m", "path_constraint", "time_budget_sec")}, POSES_KEY: poses}


def validate_continuation_fields(value):
    # Only validate shared input fields here. This constant is not a proposal;
    # profiles generate and validate every actual endpoint with the lower owner.
    validate_continuation_parameters(explicit_continuation_parameters(
        value, [[0., 0., 0., 0., 0., 0., 1.]]))


def validate_relation_continuation_parameters(value):
    if not isinstance(value, Mapping) or set(value) != PARAMETER_KEYS:
        raise ValueError("Relation continuation parameters exclude caller XYZ, TCP and pose arrays.")
    validate_relation_query_parameters({key: value[key] for key in RELATION_FIELDS})
    if len(value["offsets_xyz_m"]) * len(value["tool_axis_rotation_offsets_rad"]) > 8:
        raise ValueError("Relation continuation accepts at most 8 offset/rotation combinations before deduplication.")
    validate_continuation_fields(value)


def relation_continuation_schema(*, definition, capture, relation, continuation, profile_hash):
    """Compose existing schemas without changing their published versions."""
    def plain(value):
        if isinstance(value, Mapping):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        return value

    parameters = plain(relation.parameter_schema)
    for key in ("offsets_xyz_m", "tool_axis_rotation_offsets_rad"):
        parameters["properties"][key]["maxItems"] = 8
    parameters["properties"].update({key: plain(continuation.parameter_schema["properties"][key])
        for key in ("continuation_delta_xyz_m", "path_constraint")})
    parameters["required"] = sorted(PARAMETER_KEYS)
    path = plain(continuation.execution_result_schema)
    path["properties"][POSES_KEY] = plain(continuation.parameter_schema["properties"][POSES_KEY])
    path["required"] = sorted((*path["required"], POSES_KEY))
    result = {"type": "object", "additionalProperties": False,
        "required": ["surface_anchor", "anchor_artifact_id", "path_query", "relation_constraint_scope"],
        "properties": {"surface_anchor": plain(capture.execution_result_schema),
            "anchor_artifact_id": {"type": "string", "minLength": 1}, "path_query": path,
            "relation_constraint_scope": {"const": "endpoint_only"}}}
    return AgentSkillSchemaEntry(skill_id=definition.skill_id, skill_version="3",
        skill_definition_hash=definition.semantic_content_hash, parameter_schema=parameters,
        execution_result_schema=result, task_evaluation_schema={"type": "object", "additionalProperties": False,
            "properties": {}, "required": []}, parameter_profiles=(SkillParameterProfileRef(
                parameter_profile_id="relation-tcp-continuation-query-v3", parameter_profile_hash=profile_hash,
                parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))

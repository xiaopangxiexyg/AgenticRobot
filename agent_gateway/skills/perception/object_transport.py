"""Object-goal motion planning is an inspection Skill, not a move command."""
from __future__ import annotations

import math
from collections.abc import Mapping

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry, SkillDefinition, SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

SKILL_ID = "perception.plan_object_move"
PLAN_SCHEMA = "agenticrobot.object_transport_plan.v1"


def object_transport_definition():
    return SkillDefinition(skill_id=SKILL_ID, skill_version="1", skill_kind="inspection",
        summary="Plan motion of a verified held object from an observed hand/object relation and an object goal; no grasp, motion, release or HOLD.",
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(), parameter_semantics_id="held_object.object_goal.v1",
        execution_result_semantics_id="object_transport_plan.inspection_only.v1",
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=("verified_held_object_observation_available.v1", "complete_object_geometry_available.v1"),
        task_outcome_predicate_ids=("object_transport_assessed_without_motion.v1",))


def object_transport_schema(*, definition_hash, profile_hash):
    text = {"type": "string", "minLength": 1, "maxLength": 256}
    matrix = {"type": "array", "minItems": 16, "maxItems": 16, "items": {"type": "number"}}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "object_id": text, "side": {"enum": ["left", "right"]}, "reference_frame_id": text,
        "T_reference_from_object_goal": matrix, "speed_scale": {"type": "number", "exclusiveMinimum": 0, "maximum": 1}},
        "required": ["object_id", "side", "reference_frame_id", "T_reference_from_object_goal", "speed_scale"]}
    result = {"type": "object", "additionalProperties": False, "properties": {
        "status": {"enum": ["path_found", "rejected"]}, "reason_code": text,
        "object_id": text, "side": {"enum": ["left", "right"]}, "reference_frame_id": text,
        "scene_revision": {"type": "integer", "minimum": 0},
        "T_reference_from_object_goal": matrix, "T_reference_from_tool_goal": matrix,
        "T_tool_from_object": matrix,
        "trajectory_hash": {"anyOf": [{"type": "string", "pattern": "^[0-9a-f]{64}$"}, {"type": "null"}]},
        "duration_sec": {"type": "number", "minimum": 0}, "sample_count": {"type": "integer", "minimum": 0},
        "motion_executed": {"const": False}, "execution_authorized": {"const": False}},
        "required": ["status", "reason_code", "object_id", "side", "reference_frame_id", "scene_revision",
            "T_reference_from_object_goal", "T_reference_from_tool_goal", "T_tool_from_object", "trajectory_hash",
            "duration_sec", "sample_count", "motion_executed", "execution_authorized"]}
    return AgentSkillSchemaEntry(skill_id=SKILL_ID, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(parameter_profile_id="object-transport-v1",
            parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))


def _validate_matrix(matrix):
    if (not isinstance(matrix, (tuple, list)) or len(matrix) != 16
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in matrix)):
        raise ValueError("Object transform must contain sixteen finite numbers")
    if any(abs(matrix[12+i]-value) > 1e-8 for i, value in enumerate((0, 0, 0, 1))):
        raise ValueError("Object transform must be rigid")
    if any(abs(sum(matrix[4*k+i]*matrix[4*k+j] for k in range(3))-int(i == j)) > 1e-6
           for i in range(3) for j in range(3)):
        raise ValueError("Object transform rotation must be orthonormal")
    determinant = (matrix[0]*(matrix[5]*matrix[10]-matrix[6]*matrix[9])
                   - matrix[1]*(matrix[4]*matrix[10]-matrix[6]*matrix[8])
                   + matrix[2]*(matrix[4]*matrix[9]-matrix[5]*matrix[8]))
    if abs(determinant-1.) > 1e-6:
        raise ValueError("Object transform rotation must be proper")


def validate_object_transport_parameters(payload: Mapping):
    if set(payload) != {"object_id", "side", "reference_frame_id", "T_reference_from_object_goal", "speed_scale"}:
        raise ValueError("Object transport accepts object identity/goal only, not caller-authored grasp evidence")
    for key in ("object_id", "reference_frame_id"):
        if not isinstance(payload[key], str) or not 0 < len(payload[key]) <= 256 or payload[key].strip() != payload[key]:
            raise ValueError("Invalid object/frame identity")
    if payload["side"] not in {"left", "right"}:
        raise ValueError("Invalid arm side")
    scale = payload["speed_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Invalid speed scale")
    _validate_matrix(payload["T_reference_from_object_goal"])


def validate_object_transport_result(payload: Mapping):
    expected = {"status", "reason_code", "object_id", "side", "reference_frame_id", "scene_revision",
        "T_reference_from_object_goal", "T_reference_from_tool_goal", "T_tool_from_object", "trajectory_hash",
        "duration_sec", "sample_count", "motion_executed", "execution_authorized"}
    if set(payload) != expected or payload["status"] not in {"path_found", "rejected"}:
        raise ValueError("Invalid object transport result")
    if payload["motion_executed"] is not False or payload["execution_authorized"] is not False:
        raise ValueError("Object transport inspection cannot authorize or dispatch motion")
    validate_object_transport_parameters({key: payload[key] for key in (
        "object_id", "side", "reference_frame_id", "T_reference_from_object_goal")} | {"speed_scale": 1.})
    for key in ("scene_revision", "sample_count"):
        if isinstance(payload[key], bool) or not isinstance(payload[key], int) or payload[key] < 0:
            raise ValueError("Invalid transport revision/sample count")
    if (isinstance(payload["duration_sec"], bool) or not isinstance(payload["duration_sec"], (int, float))
            or not math.isfinite(payload["duration_sec"]) or payload["duration_sec"] < 0
            or not isinstance(payload["reason_code"], str) or not payload["reason_code"]):
        raise ValueError("Invalid transport duration/reason")
    for key in ("T_reference_from_tool_goal", "T_tool_from_object"):
        _validate_matrix(payload[key])
    tool, relation = payload["T_reference_from_tool_goal"], payload["T_tool_from_object"]
    composed = [sum(tool[4*i+k]*relation[4*k+j] for k in range(4)) for i in range(4) for j in range(4)]
    if any(abs(a-b) > 1e-6 for a, b in zip(composed, payload["T_reference_from_object_goal"])):
        raise ValueError("Object goal and hand/object relation disagree")
    if payload["status"] == "path_found":
        digest = payload["trajectory_hash"]
        if (not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest)
                or not math.isfinite(payload["duration_sec"]) or payload["duration_sec"] <= 0
                or payload["sample_count"] < 2):
            raise ValueError("Found transport path lacks a complete trajectory")
    elif payload["trajectory_hash"] is not None or payload["sample_count"] != 0 or payload["duration_sec"] != 0:
        raise ValueError("Rejected transport cannot expose a trajectory")


def validate_object_transport_result_for_parameters(parameters, result):
    if (any(result[key] != parameters[key] for key in ("object_id", "side", "reference_frame_id"))
            or tuple(result["T_reference_from_object_goal"]) != tuple(parameters["T_reference_from_object_goal"])):
        raise ValueError("Object transport result differs from its request")

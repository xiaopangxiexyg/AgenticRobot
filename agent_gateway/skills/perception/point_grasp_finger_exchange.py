"""Versioned, read-only parallel-jaw finger exchange proposal semantics."""

from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
)
from dataset_contracts.perception_artifacts import (
    PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1,
)

from .point_grasp_schemas import _mutable_json

FINGER_EXCHANGE_POSE_SOURCE = PARALLEL_JAW_FINGER_EXCHANGE_POSE_SOURCE_V1
FINGER_EXCHANGE_SKILL_VERSION = "5"


def finger_exchange_point_grasp_contract(base_definition, base_schema, *, profile_hash):
    """Extend a live/frozen schema without modifying any earlier version."""
    definition = replace(
        base_definition,
        skill_version=FINGER_EXCHANGE_SKILL_VERSION,
        summary="Generate read-only grasp proposals, including independently validated parallel-jaw finger exchanges; no robot motion.",
        execution_result_semantics_id="station_bound_grasp_proposals.finger_exchange.v5",
    )
    result = _mutable_json(base_schema.execution_result_schema)
    candidate = result["properties"]["candidates"]["items"]
    properties = candidate["properties"]
    properties["pose_source"]["enum"].append(FINGER_EXCHANGE_POSE_SOURCE)
    local_angle = _mutable_json(properties["orientation_variant_angle_deg"])
    properties["orientation_variant_angle_deg"] = {
        "anyOf": [local_angle, {"const": 180.0}]
    }
    candidate.setdefault("allOf", []).append(
        {
            "if": {
                "required": ["pose_source"],
                "properties": {"pose_source": {"const": FINGER_EXCHANGE_POSE_SOURCE}},
            },
            "then": {
                "required": [
                    "orientation_variant_axis",
                    "orientation_variant_angle_deg",
                ],
                "properties": {
                    "orientation_variant_axis": {"const": "z"},
                    "orientation_variant_angle_deg": {"const": 180.0},
                },
                "not": {"required": ["support_clearance_world_z_m"]},
            },
            "else": {"properties": {"orientation_variant_angle_deg": local_angle}},
        }
    )
    schema = AgentSkillSchemaEntry(
        skill_id=definition.skill_id,
        skill_version=FINGER_EXCHANGE_SKILL_VERSION,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=base_schema.parameter_schema,
        execution_result_schema=result,
        task_evaluation_schema=base_schema.task_evaluation_schema,
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="station-finger-exchange-v5",
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=base_schema.parameter_schema_hash,
            ),
        ),
    )
    return definition, schema


def legacy_candidate_validation_projection(payload: Mapping) -> dict:
    """Validate only the new provenance, then reuse ALL common legacy checks.

    This copy is used solely for validation. It never rewrites the published
    payload, candidate matrices, score, or retained execution target.
    """
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("candidates"), (list, tuple)
    ):
        raise TypeError("Point-grasp result must contain a candidate array.")
    if any(not isinstance(item, Mapping) for item in payload["candidates"]):
        raise ValueError("Point-grasp candidates must be objects.")
    projected = dict(payload)
    projected["candidates"] = [dict(item) for item in payload["candidates"]]
    for candidate in projected["candidates"]:
        if candidate.get("pose_source") != FINGER_EXCHANGE_POSE_SOURCE:
            continue
        angle = candidate.get("orientation_variant_angle_deg")
        if (
            candidate.get("orientation_variant_axis") != "z"
            or isinstance(angle, bool)
            or not isinstance(angle, (int, float))
            or angle != 180.0
            or "support_clearance_world_z_m" in candidate
        ):
            raise ValueError(
                "Finger exchange requires its exact +Z 180-degree provenance."
            )
        for field in (
            "pose_source",
            "orientation_variant_axis",
            "orientation_variant_angle_deg",
        ):
            candidate.pop(field)
    return projected

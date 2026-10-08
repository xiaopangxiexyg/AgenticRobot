"""Read-only frozen grasp observations; no device or planner selection here."""

from collections.abc import Mapping
from dataclasses import replace
import math

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry, SkillDefinition, SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

GRASP_OBSERVATION_SKILL_ID = "perception.capture_grasp_observation"


def grasp_observation_definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=GRASP_OBSERVATION_SKILL_ID, skill_version="1", skill_kind="inspection",
        summary="Retain correlated RGB-D views and capture-time robot geometry for target selection without moving the robot.",
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(), parameter_semantics_id="registered_roles.frozen_grasp_observation.v1",
        execution_result_semantics_id="correlated_rgbd_views.preview_artifacts.excluded_views.v1",
        task_outcome_semantics_id="passive_grasp_observation.v1",
        entry_predicate_ids=("registered_rgbd_geometry_available.v1",),
        task_outcome_predicate_ids=("grasp_observation_retained.v1",),
    )


def grasp_observation_schema(*, definition_hash: str, roles: tuple[str, ...], profile_hash: str) -> AgentSkillSchemaEntry:
    parameters = {"type": "object", "additionalProperties": False,
                  "properties": {"selected_roles": {"type": "array", "minItems": 1, "maxItems": 3,
                      "uniqueItems": True, "items": {"type": "string", "enum": list(roles)}}},
                  "required": ["selected_roles"]}
    frame_properties = {
        "role": {"type": "string", "enum": list(roles)}, "source_id": {"type": "string"},
        "ordinal": {"type": "integer", "minimum": 0, "maximum": 2},
        "width": {"type": "integer", "minimum": 1}, "height": {"type": "integer", "minimum": 1},
        "media_type": {"const": "image/jpeg"}, "media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "acquired_at": {"type": "number"},
    }
    result = {"type": "object", "additionalProperties": False, "properties": {
        "station_revision": {"type": "integer", "minimum": 0},
        "frames": {"type": "array", "minItems": 1, "maxItems": 3, "items": {
            "type": "object", "additionalProperties": False, "properties": frame_properties, "required": list(frame_properties)}},
        "excluded_views": {"type": "array", "maxItems": 3, "items": {"type": "object", "additionalProperties": False,
            "properties": {"role": {"type": "string", "enum": list(roles)}, "reason": {"type": "string"}},
            "required": ["role", "reason"]}},
    }, "required": ["station_revision", "frames", "excluded_views"]}
    return AgentSkillSchemaEntry(
        skill_id=GRASP_OBSERVATION_SKILL_ID, skill_version="1", skill_definition_hash=definition_hash,
        parameter_schema=parameters, execution_result_schema=result,
        task_evaluation_schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(parameter_profile_id="frozen-grasp-observation-v1",
            parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),),
    )


def validate_observation_parameters(payload: Mapping, *, roles: tuple[str, ...]) -> None:
    selected = payload.get("selected_roles")
    if (set(payload) != {"selected_roles"} or not isinstance(selected, (list, tuple))
            or not 1 <= len(selected) <= 3 or any(x not in roles for x in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("Select one to three registered RGB-D roles")


def validate_observation_result(payload: Mapping, *, roles: tuple[str, ...]) -> None:
    if set(payload) != {"station_revision", "frames", "excluded_views"}:
        raise ValueError("Invalid grasp observation result fields")
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Invalid observation station revision")
    frames, excluded = payload["frames"], payload["excluded_views"]
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= 3 or not isinstance(excluded, (list, tuple)):
        raise ValueError("Invalid grasp observation views")
    seen = []
    for ordinal, frame in enumerate(frames):
        if set(frame) != {"role", "source_id", "ordinal", "width", "height", "media_type", "media_content_hash", "acquired_at"}:
            raise ValueError("Invalid grasp observation frame")
        if (frame["ordinal"] != ordinal or frame["media_type"] != "image/jpeg"
                or not isinstance(frame["source_id"], str) or not frame["source_id"]
                or any(isinstance(frame[x], bool) or not isinstance(frame[x], int) or frame[x] <= 0 for x in ("width", "height"))
                or not math.isfinite(float(frame["acquired_at"]))
                or not isinstance(frame["media_content_hash"], str) or len(frame["media_content_hash"]) != 64
                or any(x not in "0123456789abcdef" for x in frame["media_content_hash"])):
            raise ValueError("Invalid grasp observation frame values")
        seen.append(frame["role"])
    for view in excluded:
        if set(view) != {"role", "reason"} or not isinstance(view["reason"], str) or not view["reason"]:
            raise ValueError("Invalid excluded observation view")
        seen.append(view["role"])
    if len(seen) > 3 or len(set(seen)) != len(seen) or any(x not in roles for x in seen):
        raise ValueError("Observation roles must be distinct and registered")


def validate_observation_result_for_parameters(parameters: Mapping, result: Mapping) -> None:
    observed = {x["role"] for x in (*result["frames"], *result["excluded_views"])}
    if observed != set(parameters["selected_roles"]):
        raise ValueError("Observation does not account for every requested role")


def frozen_point_grasp_contract(base_definition, base_schema, *, profile_hash):
    """Version the new snapshot-bound inputs without changing v2/v3 callers."""
    from .point_grasp_schemas import _mutable_json

    definition = replace(base_definition, skill_version="4",
        summary="Generate grasp proposals from a retained RGB-D observation, optionally fusing associated stationary views; no robot motion.",
        parameter_semantics_id="retained_rgbd_observation.role.pixel.view_policy.v4",
        execution_result_semantics_id="station_bound_grasp_proposals.view_fusion_evidence.v4")
    parameters = _mutable_json(base_schema.parameter_schema)
    parameters["properties"].update({
        "observation_operation_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "view_policy": {"type": "string", "enum": ["single", "auto"]},
    })
    parameters["required"].extend(("observation_operation_id", "view_policy"))
    result = _mutable_json(base_schema.execution_result_schema)
    result["properties"]["view_fusion"] = {"type": "object", "additionalProperties": False, "properties": {
        "observation_operation_id": {"type": "string", "minLength": 1},
        "view_policy": {"enum": ["single", "auto"]},
        "diagnostics_json": {"type": "string", "maxLength": 16384},
    }, "required": ["observation_operation_id", "view_policy", "diagnostics_json"]}
    result["required"].append("view_fusion")
    schema = AgentSkillSchemaEntry(skill_id=definition.skill_id, skill_version="4",
        skill_definition_hash=definition.semantic_content_hash, parameter_schema=parameters,
        execution_result_schema=result, task_evaluation_schema=base_schema.task_evaluation_schema,
        parameter_profiles=(SkillParameterProfileRef(parameter_profile_id="frozen-grasp-observation-v4",
            parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),))
    return definition, schema

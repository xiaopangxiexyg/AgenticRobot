"""Additive measured-pixel anchors; existing inline geometry Skills stay unchanged."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .object_goal_query import (
    RELATION_FIELDS,
    object_goal_query_definition,
    validate_query_parameters,
)
from .spatial_goal import PARAMETER_KEYS as SPATIAL_GOAL_FIELDS
from .spatial_goal import spatial_goal_definition, spatial_goal_geometry_properties

CAPTURE_SKILL_ID = "perception.capture_surface_anchor"
QUERY_SKILL_ID = "perception.query_object_above_paths_from_anchor"
GOAL_SKILL_ID = "perception.compute_spatial_goal_from_anchor"
GOAL_FIELDS = SPATIAL_GOAL_FIELDS - {"anchor_xyz_m"}
ANCHOR_FIELDS = {
    "measurement",
    "target_pixel_uv",
    "captured_at",
    "clock_domain_id",
    "source_operation_id",
    "source_request_hash",
    "runtime_session_id",
}
QUERY_FIELDS = {
    "component_id",
    "reference_frame_id",
    "heights_m",
    "tool_yaw_offsets_rad",
    "time_budget_sec",
}


def capture_definition(*, version="1"):
    definition = SkillDefinition(
        skill_id=CAPTURE_SKILL_ID,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Capture one measured depth surface point at role/target_pixel_uv and return "
            "an immutable surface_anchor Artifact with frame, depth, timestamp and producer identity. "
            "This is a single surface sample, not an object center, semantic recognition or 6D pose. "
            "It already includes metric back-projection; no preliminary back_project call is needed. "
            "Read required TCP identities before capture, then bind this exact output Artifact to "
            "a published anchored goal/query Skill without an intervening state read. "
            "Do not copy or edit XYZ. Current depth is captured once; no robot motion."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="measured_surface_pixel.v1",
        execution_result_semantics_id="measured_surface_anchor.v1",
        task_outcome_semantics_id="no_task_outcome.v1",
        entry_predicate_ids=(),
        task_outcome_predicate_ids=("measured_anchor_without_motion.v1",),
    )
    if version == "1":
        return definition
    if version != "2":
        raise ValueError("Unsupported surface anchor version.")
    return replace(definition, skill_version="2",
        parameter_semantics_id="selected_measured_surface_pixel.v2",
        execution_result_semantics_id="measured_surface_anchor.v2",
        summary=("Capture one measured surface anchor from a single correlated RGB-D cut. "
            "selection_mode=exact preserves target_pixel_uv; mask_interior requires exactly one "
            "point or text prompt and selects a valid-depth pixel farthest inside that mask. "
            "Use exact for an explicitly requested pixel; mask_interior is only for a free point "
            "on the selected visible object. Prefer a point prompt to disambiguate instances. "
            "Returns the actual selected pixel, depth and authenticated RGB/mask provenance, "
            "plus visible mask geometry (not a complete object extent, center, 6D pose or tracking ID). "
            "Consume this exact surface_anchor Artifact in the matching v2 anchored goal/query "
            "before another state read or motion changes the Station revision. No motion."))


def query_definition(*, version="1"):
    return replace(
        object_goal_query_definition(),
        skill_id=QUERY_SKILL_ID,
        summary=(
            "Query object-above paths using exactly one surface_anchor input Artifact from "
            "capture_surface_anchor in this same client/session/station revision. Supply its exact "
            "descriptor via the surface_anchor input Artifact binding, not XYZ or a file path. "
            "Reuses the existing frozen-scene path query and ranking. Missing/evicted provenance "
            "requires recapture. Result contains surface_anchor and path_query evidence. "
            "Ranking-pruned candidates have no path/collision verdict. No execution authorization; "
            "submit path_query.pose_targets_by_component separately to move_tcp_to_pose_safe."
        ),
        skill_version=version,
        parameter_semantics_id="artifact_surface_object_above_query.v" + version,
        execution_result_semantics_id="anchored_frozen_scene_path_evidence.v" + version,
    )


def goal_definition(*, version="1"):
    return replace(
        spatial_goal_definition(),
        skill_id=GOAL_SKILL_ID,
        summary=(
            "Compute an endpoint relation from one authenticated surface_anchor input Artifact "
            "from capture_surface_anchor in this client/session/station revision. No caller XYZ "
            "or second depth capture. Supply offset in reference or input-tool axes and caller-observed "
            "TCP orientation; preserve orientation or align a tool axis with least rotation. "
            "Returns surface_anchor provenance and spatial_goal.pose_targets_by_component. "
            "The TCP input remains caller geometry, not a new state observation. Constraints are "
            "endpoint-only, not maintained along a path. Submit the target separately to "
            "move_tcp_to_pose_safe for live collision/envelope/arrival checks."
        ),
        skill_version=version,
        parameter_semantics_id="artifact_surface_endpoint_relation.v" + version,
        execution_result_semantics_id="anchored_endpoint_proposal.v" + version,
    )


def _schema(*, definition, parameters, result, profile_hash):
    return AgentSkillSchemaEntry(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=parameters,
        execution_result_schema=result,
        task_evaluation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=definition.skill_id + ".v" + definition.skill_version,
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=agent_skill_schema_document_hash(
                    parameters, schema_kind="parameter"
                ),
            ),
        ),
    )


def capture_schema(*, definition, back_projection_schema, profile_hash):
    source = back_projection_schema.parameter_schema["properties"]
    # The source schema also accepts null; anchors deliberately only accept a pixel.
    pixel = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 0},
    }
    measurement = back_projection_schema.execution_result_schema
    measured_properties = dict(measurement["properties"])
    measured_properties["coordinate_mode"] = {"const": "pixel"}
    measured_properties["valid_point_count"] = {"const": 1}
    properties = {
        "measurement": {**dict(measurement), "properties": measured_properties},
        "target_pixel_uv": pixel,
        "captured_at": {"type": "number"},
        **{
            key: {"type": "string", "minLength": 1}
            for key in ("clock_domain_id", "source_operation_id", "runtime_session_id")
        },
        "source_request_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    return _schema(
        definition=definition,
        profile_hash=profile_hash,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": ["role", "target_pixel_uv"],
            "properties": {"role": source["role"], "target_pixel_uv": pixel},
        },
        result={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(ANCHOR_FIELDS),
            "properties": properties,
        },
    )


def query_schema(*, definition, capture, query, profile_hash):
    return _schema(
        definition=definition,
        profile_hash=profile_hash,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(QUERY_FIELDS),
            "properties": {
                key: query.parameter_schema["properties"][key] for key in QUERY_FIELDS
            },
        },
        result={
            "type": "object",
            "additionalProperties": False,
            "required": ["surface_anchor", "anchor_artifact_id", "path_query"],
            "properties": {
                "surface_anchor": capture.execution_result_schema,
                "anchor_artifact_id": {"type": "string", "minLength": 1},
                "path_query": query.execution_result_schema,
            },
        },
    )


def goal_schema(*, definition, capture, spatial_goal, profile_hash):
    return _schema(
        definition=definition,
        profile_hash=profile_hash,
        parameters={
            "type": "object",
            "additionalProperties": False,
            "required": sorted(GOAL_FIELDS),
            "properties": {
                key: spatial_goal.parameter_schema["properties"][key]
                for key in GOAL_FIELDS
            },
        },
        result={
            "type": "object",
            "additionalProperties": False,
            "required": ["surface_anchor", "anchor_artifact_id", "spatial_goal"],
            "properties": {
                "surface_anchor": capture.execution_result_schema,
                "anchor_artifact_id": {"type": "string", "minLength": 1},
                "spatial_goal": spatial_goal.execution_result_schema,
            },
        },
    )


def relation_query_schema(*, definition, capture, query, profile_hash):
    geometry = spatial_goal_geometry_properties()
    offset = geometry.pop("offset_xyz_m")
    return _schema(definition=definition, profile_hash=profile_hash,
        parameters={"type": "object", "additionalProperties": False,
            "required": sorted(RELATION_FIELDS), "properties": {
                **{key: query.parameter_schema["properties"][key]
                   for key in ("component_id", "reference_frame_id", "time_budget_sec")},
                **geometry,
                "offsets_xyz_m": {"type": "array", "minItems": 1, "maxItems": 64, "items": offset},
                "tool_axis_rotation_offsets_rad": {"type": "array", "minItems": 1, "maxItems": 64,
                    "items": {"type": "number"}},
            }},
        result={"type": "object", "additionalProperties": False,
            "required": ["surface_anchor", "anchor_artifact_id", "path_query", "relation_constraint_scope"],
            "properties": {"surface_anchor": capture.execution_result_schema,
                "anchor_artifact_id": {"type": "string", "minLength": 1},
                "path_query": query.execution_result_schema,
                "relation_constraint_scope": {"const": "endpoint_only"}}})


def validate_anchor(value):
    if not isinstance(value, Mapping) or set(value) != ANCHOR_FIELDS:
        raise ValueError("Surface anchor must match the exact measured schema.")
    pixel = value["target_pixel_uv"]
    if (
        not isinstance(pixel, (list, tuple))
        or len(pixel) != 2
        or any(type(x) is not int or x < 0 for x in pixel)
    ):
        raise ValueError("Surface anchor requires one nonnegative integer pixel.")
    measurement = value["measurement"]
    if (
        not isinstance(measurement, Mapping)
        or measurement.get("coordinate_mode") != "pixel"
        or measurement.get("valid_point_count") != 1
    ):
        raise ValueError("A region statistic is not a measured surface anchor.")
    captured = value["captured_at"]
    if (
        isinstance(captured, bool)
        or not isinstance(captured, (float, int))
        or not math.isfinite(captured)
    ):
        raise ValueError(
            "Anchor capture time must be finite in its named clock domain."
        )
    for key in ("clock_domain_id", "source_operation_id", "runtime_session_id"):
        text = value[key]
        if not isinstance(text, str) or not text or text != text.strip():
            raise ValueError("Anchor provenance must have nonempty identities.")
    digest = value["source_request_hash"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("Anchor must identify its measured request hash.")


def validate_anchor_query_parameters(value):
    if not isinstance(value, Mapping) or set(value) != QUERY_FIELDS:
        raise ValueError("Anchored query parameters exclude caller-supplied XYZ.")
    validate_query_parameters({**value, "anchor_xyz_m": [0.0, 0.0, 0.0]})

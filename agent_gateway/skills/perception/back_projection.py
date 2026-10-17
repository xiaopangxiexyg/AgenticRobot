"""Target-neutral frozen RGB-D and pixel-to-metric-point contracts."""

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
from agent_gateway.skills.perception.grasp_observation import (
    grasp_observation_definition,
    grasp_observation_schema,
)

CAPTURE_RGBD_SKILL_ID = "perception.capture_rgbd"
BACK_PROJECT_SKILL_ID = "perception.back_project"
BACK_PROJECT_SKILL_VERSION = "2"
RGBD_OBSERVATION_SLOT = "rgbd_observation"
MAX_BACK_PROJECT_PIXELS = 128


def capture_rgbd_definition() -> SkillDefinition:
    return replace(
        grasp_observation_definition(), skill_id=CAPTURE_RGBD_SKILL_ID,
        summary="Retain correlated RGB-D views and capture-time camera poses without a grasp model or robot motion.",
        parameter_semantics_id="registered_roles.frozen_rgbd_observation.v1",
        task_outcome_semantics_id="passive_rgbd_observation.v1",
        task_outcome_predicate_ids=("rgbd_observation_retained.v1",),
    )


def capture_rgbd_schema(*, roles: tuple[str, ...], profile_hash: str) -> AgentSkillSchemaEntry:
    definition = capture_rgbd_definition()
    return replace(
        grasp_observation_schema(definition_hash=definition.semantic_content_hash,
                                 roles=roles, profile_hash=profile_hash),
        skill_id=CAPTURE_RGBD_SKILL_ID,
    )


def back_project_definition() -> SkillDefinition:
    return SkillDefinition(
        skill_id=BACK_PROJECT_SKILL_ID, skill_version=BACK_PROJECT_SKILL_VERSION, skill_kind="inspection",
        summary="Back-project pixels from one retained RGB-D view to camera and public reference-frame points. No new capture, model inference, object pose or robot motion.",
        intended_target_kind="target_agnostic", cross_target_deployment_policy="not_applicable",
        controlled_component_ids=(),
        parameter_semantics_id="retained_rgbd_operation.role.pixels_uv.v1",
        execution_result_semantics_id="capture_bound_metric_surface_points.v1",
        task_outcome_semantics_id="passive_back_projection.v1",
        entry_predicate_ids=("retained_rgbd_observation_available.v1",),
        task_outcome_predicate_ids=("pixels_back_projected.v1",),
    )


def back_project_schema(*, roles: tuple[str, ...], profile_hash: str) -> AgentSkillSchemaEntry:
    definition = back_project_definition()
    pixel = {"type": "array", "minItems": 2, "maxItems": 2,
             "items": {"type": "integer", "minimum": 0}}
    vector = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "number"}}
    parameters = {"type": "object", "additionalProperties": False, "properties": {
        "observation_operation_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "role": {"type": "string", "enum": list(roles)},
        "pixels_uv": {"type": "array", "minItems": 1, "maxItems": MAX_BACK_PROJECT_PIXELS, "items": pixel},
    }, "required": ["observation_operation_id", "role", "pixels_uv"]}
    point = {"type": "object", "additionalProperties": False, "properties": {
        "pixel_uv": pixel, "depth_m": {"type": "number", "exclusiveMinimum": 0},
        "point_camera_m": vector, "point_reference_m": vector,
    }, "required": ["pixel_uv", "depth_m", "point_camera_m", "point_reference_m"]}
    properties = {
        "observation_operation_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "role": {"type": "string", "enum": list(roles)},
        "source_id": {"type": "string", "minLength": 1},
        "capture_group_id": {"type": "string", "minLength": 1},
        "acquired_at": {"type": "number"},
        "station_revision": {"type": "integer", "minimum": 0},
        "reference_frame_id": {"type": "string", "minLength": 1},
        "reference_frame_spec_hash": {"type": "string", "minLength": 1},
        "frame_graph_hash": {"type": "string", "minLength": 1},
        "rgb_content_hash": {"type": "string", "minLength": 1},
        "depth_content_hash": {"type": "string", "minLength": 1},
        "points": {"type": "array", "minItems": 1, "maxItems": MAX_BACK_PROJECT_PIXELS, "items": point},
        "point_semantics": {"const": "observed_surface_not_object_center_or_pose"},
    }
    return AgentSkillSchemaEntry(
        skill_id=BACK_PROJECT_SKILL_ID, skill_version=BACK_PROJECT_SKILL_VERSION, skill_definition_hash=definition.semantic_content_hash,
        parameter_schema=parameters,
        execution_result_schema={"type": "object", "additionalProperties": False,
                                 "properties": properties, "required": list(properties)},
        task_evaluation_schema={"type": "object", "additionalProperties": False, "properties": {}, "required": []},
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="retained-rgbd-back-projection-v2", parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),),
    )


def _identifier(value: object) -> bool:
    return isinstance(value, str) and 0 < len(value) <= 256 and value == value.strip()


def _pixels(values: object) -> bool:
    return (isinstance(values, (list, tuple)) and 1 <= len(values) <= MAX_BACK_PROJECT_PIXELS
            and all(isinstance(pixel, (list, tuple)) and len(pixel) == 2
                    and all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in pixel)
                    for pixel in values))


def validate_back_project_parameters(payload: Mapping, *, roles: tuple[str, ...]) -> None:
    if (set(payload) != {"observation_operation_id", "role", "pixels_uv"}
            or not _identifier(payload["observation_operation_id"])
            or payload["role"] not in roles or not _pixels(payload["pixels_uv"])):
        raise ValueError("Back projection requires a retained observation, registered role and bounded pixels_uv")


def validate_back_project_result(payload: Mapping, *, roles: tuple[str, ...]) -> None:
    fields = {"observation_operation_id", "role", "source_id", "capture_group_id", "acquired_at",
              "station_revision", "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
              "rgb_content_hash", "depth_content_hash", "points", "point_semantics"}
    if set(payload) != fields or payload["role"] not in roles:
        raise ValueError("Invalid back projection result fields")
    for key in fields - {"role", "acquired_at", "station_revision", "points", "point_semantics"}:
        if not _identifier(payload[key]):
            raise ValueError("Invalid back projection identity: " + key)
    revision = payload["station_revision"]
    if (isinstance(revision, bool) or not isinstance(revision, int) or revision < 0
            or not _finite_number(payload["acquired_at"])
            or payload["point_semantics"] != "observed_surface_not_object_center_or_pose"):
        raise ValueError("Invalid back projection observation metadata")
    points = payload["points"]
    if not isinstance(points, (list, tuple)) or not 1 <= len(points) <= MAX_BACK_PROJECT_PIXELS:
        raise ValueError("Invalid back projection point count")
    for point in points:
        if (not isinstance(point, Mapping)
                or set(point) != {"pixel_uv", "depth_m", "point_camera_m", "point_reference_m"}
                or not _pixels([point["pixel_uv"]]) or not _finite_number(point["depth_m"])
                or point["depth_m"] <= 0):
            raise ValueError("Invalid back projection point")
        for key in ("point_camera_m", "point_reference_m"):
            value = point[key]
            if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(_finite_number(x) for x in value):
                raise ValueError("Back projection vectors must be finite metric XYZ")


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_back_project_result_for_parameters(parameters: Mapping, result: Mapping) -> None:
    if (any(parameters[key] != result[key] for key in ("observation_operation_id", "role"))
            or [tuple(x) for x in parameters["pixels_uv"]] != [tuple(x["pixel_uv"]) for x in result["points"]]):
        raise ValueError("Back projection result differs from the requested observation, role or pixels")

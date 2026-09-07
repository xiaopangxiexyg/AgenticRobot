"""Strict schemas for rendering one retained Station point-grasp candidate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import RENDER_POINT_GRASP_CANDIDATE_SKILL_ID_V1


def _matrix_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 16,
        "maxItems": 16,
        "items": {"type": "number"},
    }


def _intrinsics_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "fx": {"type": "number", "exclusiveMinimum": 0},
            "fy": {"type": "number", "exclusiveMinimum": 0},
            "cx": {"type": "number", "minimum": 0},
            "cy": {"type": "number", "minimum": 0},
            "fovy_deg": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["width", "height", "fx", "fy", "cx", "cy", "fovy_deg"],
    }


def point_grasp_candidate_render_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    roles: tuple[str, ...],
    max_candidates: int,
) -> AgentSkillSchemaEntry:
    """Build the deployment-bound, single-candidate render schema."""

    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "point_grasp_operation_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "candidate_ordinal": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_candidates - 1,
            },
            "view_role": {"type": "string", "enum": list(roles)},
        },
        "required": [
            "point_grasp_operation_id",
            "candidate_ordinal",
            "view_role",
        ],
    }
    frame = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"const": 0},
            "role": {"type": "string", "enum": list(roles)},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "media_type": {"const": "image/jpeg"},
            "media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "ordinal",
            "role",
            "width",
            "height",
            "media_type",
            "media_content_hash",
        ],
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "point_grasp_operation_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "point_grasp_request_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "point_grasp_result_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "point_grasp_candidate_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "source_capture_group_id": {
                "type": "string",
                "pattern": "^real-aligned-rgbd:[0-9a-f]{64}$",
            },
            "source_station_revision": {"type": "integer", "minimum": 0},
            "candidate_ordinal": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_candidates - 1,
            },
            "view_role": {"type": "string", "enum": list(roles)},
            "view_source_id": {"type": "string", "minLength": 1},
            "view_acquired_at": {"type": "number"},
            "view_station_revision": {"type": "integer", "minimum": 0},
            "view_source_generation": {"type": "integer", "minimum": 0},
            "view_frame_index": {"type": "integer", "minimum": 0},
            "view_capture_group_id": {
                "type": "string",
                "pattern": "^real-rgb-geometry:[0-9a-f]{64}$",
            },
            "view_rgb_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "view_geometry_binding_hash": {
                "type": "string",
                "minLength": 1,
            },
            "view_geometry_cut_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "view_pose_measured_at": {"type": "number"},
            "view_pose_available_at": {"type": "number"},
            "view_pose_evidence_refs": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "source_binding_hash": {"type": "string", "minLength": 1},
            "frame_binding_hash": {"type": "string", "minLength": 1},
            "frame_graph_hash": {"type": "string", "minLength": 1},
            "source_frame_id": {"type": "string", "minLength": 1},
            "source_frame_spec_hash": {"type": "string", "minLength": 1},
            "reference_frame_id": {"type": "string", "minLength": 1},
            "reference_frame_spec_hash": {"type": "string", "minLength": 1},
            "camera_intrinsics": _intrinsics_schema(),
            "T_reference_from_tool_tcp": _matrix_schema(),
            "T_reference_from_camera": _matrix_schema(),
            "T_camera_from_tool_tcp": _matrix_schema(),
            "projection_visibility": {
                "type": "string",
                "enum": [
                    "visible",
                    "partially_visible",
                    "behind_camera",
                    "out_of_frame",
                ],
            },
            "projection_total_segment_count": {"type": "integer", "minimum": 1},
            "projection_front_segment_count": {"type": "integer", "minimum": 0},
            "projection_visible_segment_count": {"type": "integer", "minimum": 0},
            "projection_clipped_segment_count": {"type": "integer", "minimum": 0},
            "tcp_pixel_uv": {
                "anyOf": [
                    {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "integer", "minimum": 0},
                    },
                    {"type": "null"},
                ]
            },
            "depth_occlusion_tested": {"const": False},
            "downstream_control_effects": {"const": "forbidden"},
            "frames": {
                "type": "array",
                "minItems": 1,
                "maxItems": 1,
                "items": frame,
            },
        },
        "required": [
            "point_grasp_operation_id",
            "point_grasp_request_hash",
            "point_grasp_result_hash",
            "point_grasp_candidate_hash",
            "source_capture_group_id",
            "source_station_revision",
            "candidate_ordinal",
            "view_role",
            "view_source_id",
            "view_acquired_at",
            "view_station_revision",
            "view_source_generation",
            "view_frame_index",
            "view_capture_group_id",
            "view_rgb_content_hash",
            "view_geometry_binding_hash",
            "view_geometry_cut_hash",
            "view_pose_measured_at",
            "view_pose_available_at",
            "view_pose_evidence_refs",
            "source_binding_hash",
            "frame_binding_hash",
            "frame_graph_hash",
            "source_frame_id",
            "source_frame_spec_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "camera_intrinsics",
            "T_reference_from_tool_tcp",
            "T_reference_from_camera",
            "T_camera_from_tool_tcp",
            "projection_visibility",
            "projection_total_segment_count",
            "projection_front_segment_count",
            "projection_visible_segment_count",
            "projection_clipped_segment_count",
            "tcp_pixel_uv",
            "depth_occlusion_tested",
            "downstream_control_effects",
            "frames",
        ],
    }
    task_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }
    profile = SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema, schema_kind="parameter"
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=RENDER_POINT_GRASP_CANDIDATE_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=task_schema,
        parameter_profiles=(profile,),
    )


def _text(value: object, *, field: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ValueError(f"{field} must be a bounded trimmed string.")
    return value


def _sha256(value: object, *, field: str) -> str:
    text = _text(value, field=field, maximum=64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a SHA-256 digest.")
    return text


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _matrix(value: object, *, field: str) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 16:
        raise ValueError(f"{field} must contain 16 finite numbers.")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{field} must contain 16 finite numbers.")
    matrix = tuple(
        tuple(float(value[row * 4 + column]) for column in range(4)) for row in range(4)
    )
    if any(
        abs(actual - expected) > 1e-8
        for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        raise ValueError(f"{field} has an invalid homogeneous row.")
    rotation = tuple(
        tuple(matrix[row][column] for column in range(3)) for row in range(3)
    )
    for first in range(3):
        for second in range(3):
            product = sum(
                rotation[row][first] * rotation[row][second] for row in range(3)
            )
            expected = 1.0 if first == second else 0.0
            if abs(product - expected) > 1e-5:
                raise ValueError(f"{field} rotation is not orthonormal.")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if abs(determinant - 1.0) > 1e-5:
        raise ValueError(f"{field} rotation is not right-handed.")
    return matrix


def _inverse_rigid(
    matrix: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    rotation_t = tuple(
        tuple(matrix[column][row] for column in range(3)) for row in range(3)
    )
    translation = tuple(matrix[row][3] for row in range(3))
    inverse_translation = tuple(
        -sum(rotation_t[row][column] * translation[column] for column in range(3))
        for row in range(3)
    )
    return tuple(
        tuple(rotation_t[row][column] for column in range(3))
        + (inverse_translation[row],)
        for row in range(3)
    ) + ((0.0, 0.0, 0.0, 1.0),)


def _matmul(
    first: tuple[tuple[float, ...], ...],
    second: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            sum(first[row][item] * second[item][column] for item in range(4))
            for column in range(4)
        )
        for row in range(4)
    )


def validate_point_grasp_candidate_render_parameters_v1(
    payload: Mapping[str, Any],
    *,
    roles: tuple[str, ...],
    max_candidates: int,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "point_grasp_operation_id",
        "candidate_ordinal",
        "view_role",
    }:
        raise ValueError("Point-grasp candidate render parameter fields changed.")
    _text(
        payload["point_grasp_operation_id"],
        field="point_grasp_operation_id",
        maximum=256,
    )
    ordinal = _nonnegative_integer(
        payload["candidate_ordinal"], field="candidate_ordinal"
    )
    if ordinal >= max_candidates:
        raise ValueError("candidate_ordinal exceeds the deployment bound.")
    if _text(payload["view_role"], field="view_role", maximum=128) not in roles:
        raise ValueError("view_role is not a registered RGB geometry role.")


def validate_point_grasp_candidate_render_result_v1(
    payload: Mapping[str, Any],
    *,
    source_by_role: Mapping[str, str],
    dimensions_by_role: Mapping[str, tuple[int, int]],
    source_binding_hash: str,
    camera_geometry_binding_hash: str,
    frame_binding_hash: str,
    frame_graph_hash: str,
    source_frame_id: str,
    source_frame_spec_hash: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    max_candidates: int,
) -> None:
    expected = {
        "point_grasp_operation_id",
        "point_grasp_request_hash",
        "point_grasp_result_hash",
        "point_grasp_candidate_hash",
        "source_capture_group_id",
        "source_station_revision",
        "candidate_ordinal",
        "view_role",
        "view_source_id",
        "view_acquired_at",
        "view_station_revision",
        "view_source_generation",
        "view_frame_index",
        "view_capture_group_id",
        "view_rgb_content_hash",
        "view_geometry_binding_hash",
        "view_geometry_cut_hash",
        "view_pose_measured_at",
        "view_pose_available_at",
        "view_pose_evidence_refs",
        "source_binding_hash",
        "frame_binding_hash",
        "frame_graph_hash",
        "source_frame_id",
        "source_frame_spec_hash",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "camera_intrinsics",
        "T_reference_from_tool_tcp",
        "T_reference_from_camera",
        "T_camera_from_tool_tcp",
        "projection_visibility",
        "projection_total_segment_count",
        "projection_front_segment_count",
        "projection_visible_segment_count",
        "projection_clipped_segment_count",
        "tcp_pixel_uv",
        "depth_occlusion_tested",
        "downstream_control_effects",
        "frames",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Point-grasp candidate render result fields changed.")
    _text(
        payload["point_grasp_operation_id"],
        field="point_grasp_operation_id",
        maximum=256,
    )
    for field in (
        "point_grasp_request_hash",
        "point_grasp_result_hash",
        "point_grasp_candidate_hash",
        "view_rgb_content_hash",
        "view_geometry_cut_hash",
    ):
        _sha256(payload[field], field=field)
    source_group = _text(
        payload["source_capture_group_id"],
        field="source_capture_group_id",
        maximum=83,
    )
    if not source_group.startswith("real-aligned-rgbd:"):
        raise ValueError("source_capture_group_id has the wrong domain.")
    _sha256(
        source_group.removeprefix("real-aligned-rgbd:"),
        field="source_capture_group_id hash",
    )
    view_group = _text(
        payload["view_capture_group_id"],
        field="view_capture_group_id",
        maximum=82,
    )
    if not view_group.startswith("real-rgb-geometry:"):
        raise ValueError("view_capture_group_id has the wrong domain.")
    _sha256(
        view_group.removeprefix("real-rgb-geometry:"),
        field="view_capture_group_id hash",
    )
    for field in (
        "source_station_revision",
        "candidate_ordinal",
        "view_station_revision",
        "view_source_generation",
        "view_frame_index",
    ):
        _nonnegative_integer(payload[field], field=field)
    if payload["candidate_ordinal"] >= max_candidates:
        raise ValueError("candidate_ordinal exceeds the deployment bound.")
    if payload["source_station_revision"] != payload["view_station_revision"]:
        raise ValueError("Source and view Station revisions differ.")
    role = _text(payload["view_role"], field="view_role", maximum=128)
    if role not in source_by_role or payload["view_source_id"] != source_by_role[role]:
        raise ValueError("Rendered view source identity changed.")
    for field, expected_value in (
        ("source_binding_hash", source_binding_hash),
        ("view_geometry_binding_hash", camera_geometry_binding_hash),
        ("frame_binding_hash", frame_binding_hash),
        ("frame_graph_hash", frame_graph_hash),
        ("source_frame_id", source_frame_id),
        ("source_frame_spec_hash", source_frame_spec_hash),
        ("reference_frame_id", reference_frame_id),
        ("reference_frame_spec_hash", reference_frame_spec_hash),
    ):
        if payload[field] != expected_value:
            raise ValueError(f"Point-grasp candidate render {field} changed.")
    timestamps = {}
    for field in (
        "view_acquired_at",
        "view_pose_measured_at",
        "view_pose_available_at",
    ):
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{field} must be finite.")
        timestamps[field] = float(value)
    if timestamps["view_pose_available_at"] < timestamps["view_pose_measured_at"]:
        raise ValueError("Rendered camera pose availability precedes measurement.")
    evidence_refs = payload["view_pose_evidence_refs"]
    if (
        not isinstance(evidence_refs, (list, tuple))
        or not evidence_refs
        or len(evidence_refs) != len(set(evidence_refs))
        or any(not isinstance(item, str) or not item for item in evidence_refs)
    ):
        raise ValueError("Rendered camera pose requires unique evidence refs.")
    width, height = dimensions_by_role[role]
    intrinsics = payload["camera_intrinsics"]
    required_intrinsics = {"width", "height", "fx", "fy", "cx", "cy", "fovy_deg"}
    if not isinstance(intrinsics, Mapping) or set(intrinsics) != required_intrinsics:
        raise ValueError("Rendered view intrinsics fields changed.")
    if (intrinsics["width"], intrinsics["height"]) != (width, height):
        raise ValueError("Rendered view dimensions changed.")
    for field in ("fx", "fy", "cx", "cy", "fovy_deg"):
        value = intrinsics[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Rendered view intrinsics must be finite.")
    if (
        min(
            float(intrinsics["fx"]),
            float(intrinsics["fy"]),
            float(intrinsics["fovy_deg"]),
        )
        <= 0
    ):
        raise ValueError("Rendered view focal values must be positive.")
    reference_from_tcp = _matrix(
        payload["T_reference_from_tool_tcp"],
        field="T_reference_from_tool_tcp",
    )
    reference_from_camera = _matrix(
        payload["T_reference_from_camera"],
        field="T_reference_from_camera",
    )
    camera_from_tcp = _matrix(
        payload["T_camera_from_tool_tcp"],
        field="T_camera_from_tool_tcp",
    )
    derived = _matmul(_inverse_rigid(reference_from_camera), reference_from_tcp)
    if any(
        abs(derived[row][column] - camera_from_tcp[row][column]) > 1e-7
        for row in range(4)
        for column in range(4)
    ):
        raise ValueError("Rendered candidate transform chain is inconsistent.")
    if payload["projection_visibility"] not in {
        "visible",
        "partially_visible",
        "behind_camera",
        "out_of_frame",
    }:
        raise ValueError("projection_visibility is invalid.")
    total = _nonnegative_integer(
        payload["projection_total_segment_count"],
        field="projection_total_segment_count",
    )
    front = _nonnegative_integer(
        payload["projection_front_segment_count"],
        field="projection_front_segment_count",
    )
    visible = _nonnegative_integer(
        payload["projection_visible_segment_count"],
        field="projection_visible_segment_count",
    )
    clipped = _nonnegative_integer(
        payload["projection_clipped_segment_count"],
        field="projection_clipped_segment_count",
    )
    if total <= 0 or front > total or visible > front or clipped > total:
        raise ValueError("Projection segment counts are inconsistent.")
    tcp_pixel = payload["tcp_pixel_uv"]
    if tcp_pixel is not None and (
        not isinstance(tcp_pixel, (list, tuple))
        or len(tcp_pixel) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in tcp_pixel
        )
        or tcp_pixel[0] >= width
        or tcp_pixel[1] >= height
    ):
        raise ValueError("tcp_pixel_uv is outside the rendered view.")
    if (
        payload["depth_occlusion_tested"] is not False
        or payload["downstream_control_effects"] != "forbidden"
    ):
        raise ValueError("Rendered candidate authority semantics changed.")
    frames = payload["frames"]
    if not isinstance(frames, (list, tuple)) or len(frames) != 1:
        raise ValueError("Exactly one rendered frame is required.")
    frame = frames[0]
    frame_fields = {
        "ordinal",
        "role",
        "width",
        "height",
        "media_type",
        "media_content_hash",
    }
    if not isinstance(frame, Mapping) or set(frame) != frame_fields:
        raise ValueError("Rendered frame fields changed.")
    if (
        frame["ordinal"],
        frame["role"],
        frame["width"],
        frame["height"],
        frame["media_type"],
    ) != (0, role, width, height, "image/jpeg"):
        raise ValueError("Rendered frame identity changed.")
    _sha256(frame["media_content_hash"], field="frames[0].media_content_hash")


def validate_point_grasp_candidate_render_result_for_parameters_v1(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    for parameter_field, result_field in (
        ("point_grasp_operation_id", "point_grasp_operation_id"),
        ("candidate_ordinal", "candidate_ordinal"),
        ("view_role", "view_role"),
    ):
        if parameters.get(parameter_field) != result.get(result_field):
            raise ValueError(
                f"Rendered result {result_field} differs from the request."
            )


__all__ = [
    "point_grasp_candidate_render_schema_entry_v1",
    "validate_point_grasp_candidate_render_parameters_v1",
    "validate_point_grasp_candidate_render_result_for_parameters_v1",
    "validate_point_grasp_candidate_render_result_v1",
]

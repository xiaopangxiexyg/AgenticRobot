"""Strict public schemas for passive point-conditioned grasp planning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from .definition import PLAN_POINT_GRASP_SKILL_ID_V1


def _matrix_schema() -> dict[str, Any]:
    return {"type": "array", "minItems": 16, "maxItems": 16, "items": {"type": "number"}}


def _vector_schema(length: int) -> dict[str, Any]:
    return {"type": "array", "minItems": length, "maxItems": length, "items": {"type": "number"}}


def point_grasp_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    max_candidates: int,
    backend_id: str,
) -> AgentSkillSchemaEntry:
    nullable_path = {"anyOf": [{"type": "string", "minLength": 1, "maxLength": 4096}, {"type": "null"}]}
    intrinsics = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "fx": {"type": "number", "exclusiveMinimum": 0},
            "fy": {"type": "number", "exclusiveMinimum": 0},
            "cx": {"type": "number", "minimum": 0},
            "cy": {"type": "number", "minimum": 0},
        },
        "required": ["width", "height", "fx", "fy", "cx", "cy"],
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_source": {"type": "string", "enum": ["artifact", "local_path"]},
            "local_rgb_path": nullable_path,
            "local_depth_path": nullable_path,
            "target_pixel_uv": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0}},
            "camera_intrinsics": intrinsics,
            "camera_frame_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "base_frame_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "T_base_from_camera": _matrix_schema(),
            "max_candidates": {"type": "integer", "minimum": 1, "maximum": max_candidates},
        },
        "required": [
            "input_source", "local_rgb_path", "local_depth_path", "target_pixel_uv",
            "camera_intrinsics", "camera_frame_id", "base_frame_id",
            "T_base_from_camera", "max_candidates",
        ],
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"type": "integer", "minimum": 0},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "model_score": {"type": "number", "minimum": 0, "maximum": 1},
            "click_distance_m": {"type": "number", "minimum": 0},
            "gripper_width_m": {"type": "number", "minimum": 0.005, "maximum": 0.070},
            "T_camera_from_tcp": _matrix_schema(),
            "T_base_from_tcp": _matrix_schema(),
            "tcp_xyz_camera_m": _vector_schema(3),
            "tcp_xyz_base_m": _vector_schema(3),
            "visible_collision_free": {"const": True},
        },
        "required": [
            "ordinal", "score", "model_score", "click_distance_m", "gripper_width_m",
            "T_camera_from_tcp", "T_base_from_tcp", "tcp_xyz_camera_m", "tcp_xyz_base_m",
            "visible_collision_free",
        ],
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_source": {"type": "string", "enum": ["artifact", "local_path"]},
            "rgb_media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "depth_media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "target_pixel_uv": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "integer", "minimum": 0}},
            "target_point_camera_m": _vector_schema(3),
            "target_point_base_m": _vector_schema(3),
            "target_mask_bbox_xyxy": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "integer", "minimum": 0}},
            "camera_frame_id": {"type": "string", "minLength": 1},
            "base_frame_id": {"type": "string", "minLength": 1},
            "pose_semantics": {"const": "homogeneous_row_major_parent_from_tool_tcp_m"},
            "gripper_profile_id": {"const": "DMTAC_M"},
            "backend_id": {"const": backend_id},
            "candidate_count": {"type": "integer", "minimum": 1, "maximum": max_candidates},
            "selected_ordinal": {"type": "integer", "minimum": 0, "maximum": max_candidates - 1},
            "candidates": {"type": "array", "minItems": 1, "maxItems": max_candidates, "items": candidate},
            "model_profile_id": {"type": "string", "minLength": 1},
            "model_profile_hash": {"type": "string", "minLength": 1},
            "annotated_image_media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "required": [
            "input_source", "rgb_media_content_hash", "depth_media_content_hash", "target_pixel_uv",
            "target_point_camera_m", "target_point_base_m", "target_mask_bbox_xyxy",
            "camera_frame_id", "base_frame_id", "pose_semantics", "gripper_profile_id",
            "backend_id", "candidate_count", "selected_ordinal", "candidates", "model_profile_id",
            "model_profile_hash", "annotated_image_media_content_hash",
        ],
    }
    task_schema = {"type": "object", "additionalProperties": False, "properties": {}, "required": []}
    profile = SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(parameter_schema, schema_kind="parameter"),
    )
    return AgentSkillSchemaEntry(
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=task_schema,
        parameter_profiles=(profile,),
    )


def _text(value: object, *, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be a bounded trimmed string.")
    return value


def _matrix(value: object, *, field: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain 16 numbers.") from exc
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{field} must be a homogeneous transform.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError(f"{field} rotation is not rigid.")
    return matrix


def _vector(value: object, *, field: str, length: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must contain {length} finite numbers.") from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{field} must contain {length} finite numbers.")
    return vector


def validate_point_grasp_parameters_v1(payload: Mapping[str, Any], *, local_path_enabled: bool, max_candidates: int) -> None:
    fields = {
        "input_source", "local_rgb_path", "local_depth_path", "target_pixel_uv",
        "camera_intrinsics", "camera_frame_id", "base_frame_id", "T_base_from_camera", "max_candidates",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Point-grasp parameter fields do not match the schema.")
    source = payload["input_source"]
    if source not in {"artifact", "local_path"}:
        raise ValueError("Point-grasp input_source is unsupported.")
    paths = (payload["local_rgb_path"], payload["local_depth_path"])
    if source == "artifact" and paths != (None, None):
        raise ValueError("Artifact point-grasp input cannot include local paths.")
    if source == "local_path":
        if not local_path_enabled:
            raise ValueError("Local-path point-grasp input is disabled.")
        if any(not _text(value, field="local input path").startswith("/") for value in paths):
            raise ValueError("Local point-grasp paths must be absolute.")
    point = payload["target_pixel_uv"]
    if not isinstance(point, (list, tuple)) or len(point) != 2 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in point):
        raise ValueError("target_pixel_uv must contain two non-negative integers.")
    intrinsics = payload["camera_intrinsics"]
    if not isinstance(intrinsics, Mapping) or set(intrinsics) != {"width", "height", "fx", "fy", "cx", "cy"}:
        raise ValueError("camera_intrinsics fields changed.")
    width, height = intrinsics["width"], intrinsics["height"]
    if any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (width, height)):
        raise ValueError("Intrinsic dimensions must be positive integers.")
    if not (point[0] < width and point[1] < height):
        raise ValueError("target_pixel_uv is outside the calibrated image.")
    for field in ("fx", "fy", "cx", "cy"):
        value = intrinsics[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError("Camera intrinsics must be finite.")
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0 or intrinsics["cx"] < 0 or intrinsics["cy"] < 0:
        raise ValueError("Camera intrinsics are outside their bounds.")
    _text(payload["camera_frame_id"], field="camera_frame_id", maximum=128)
    _text(payload["base_frame_id"], field="base_frame_id", maximum=128)
    _matrix(payload["T_base_from_camera"], field="T_base_from_camera")
    requested = payload["max_candidates"]
    if isinstance(requested, bool) or not isinstance(requested, int) or not 1 <= requested <= max_candidates:
        raise ValueError("max_candidates is outside the deployment bound.")


def validate_point_grasp_result_v1(payload: Mapping[str, Any], *, backend_id: str, model_profile_id: str, model_profile_hash: str, max_candidates: int) -> None:
    expected = {
        "input_source", "rgb_media_content_hash", "depth_media_content_hash", "target_pixel_uv",
        "target_point_camera_m", "target_point_base_m", "target_mask_bbox_xyxy", "camera_frame_id",
        "base_frame_id", "pose_semantics", "gripper_profile_id", "backend_id", "candidate_count",
        "selected_ordinal", "candidates", "model_profile_id", "model_profile_hash",
        "annotated_image_media_content_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Point-grasp result fields changed.")
    if payload["model_profile_id"] != model_profile_id or payload["model_profile_hash"] != model_profile_hash:
        raise ValueError("Point-grasp model identity changed.")
    if payload["pose_semantics"] != "homogeneous_row_major_parent_from_tool_tcp_m" or payload["gripper_profile_id"] != "DMTAC_M":
        raise ValueError("Point-grasp pose or gripper semantics changed.")
    if payload["backend_id"] != backend_id:
        raise ValueError("Point-grasp backend identity changed.")
    for field in ("rgb_media_content_hash", "depth_media_content_hash", "annotated_image_media_content_hash"):
        value = payload[field]
        if not isinstance(value, str) or len(value) != 64 or any(x not in "0123456789abcdef" for x in value):
            raise ValueError(f"{field} must be a SHA-256 digest.")
    candidates = payload["candidates"]
    if not isinstance(candidates, (list, tuple)) or not 1 <= len(candidates) <= max_candidates or payload["candidate_count"] != len(candidates) or payload["selected_ordinal"] != 0:
        raise ValueError("Point-grasp candidate selection is invalid.")
    fields = {"ordinal", "score", "model_score", "click_distance_m", "gripper_width_m", "T_camera_from_tcp", "T_base_from_tcp", "tcp_xyz_camera_m", "tcp_xyz_base_m", "visible_collision_free"}
    previous_score = math.inf
    for ordinal, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or set(candidate) != fields or candidate["ordinal"] != ordinal:
            raise ValueError("Point-grasp candidate fields or ordinals changed.")
        camera = _matrix(candidate["T_camera_from_tcp"], field="T_camera_from_tcp")
        base = _matrix(candidate["T_base_from_tcp"], field="T_base_from_tcp")
        if not np.allclose(camera[:3, 3], candidate["tcp_xyz_camera_m"], atol=1e-8) or not np.allclose(base[:3, 3], candidate["tcp_xyz_base_m"], atol=1e-8):
            raise ValueError("Point-grasp candidate xyz differs from its transform.")
        if candidate["visible_collision_free"] is not True:
            raise ValueError("Point-grasp candidate is not visible-scene collision free.")
        for field in ("score", "model_score"):
            value = candidate[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
                raise ValueError("Point-grasp scores must be in [0, 1].")
        if float(candidate["score"]) > previous_score + 1e-12:
            raise ValueError("Point-grasp candidates are not sorted by score.")
        previous_score = float(candidate["score"])
        if not 0.005 <= float(candidate["gripper_width_m"]) <= 0.070 or float(candidate["click_distance_m"]) < 0:
            raise ValueError("Point-grasp geometry is outside DMTAC M bounds.")
    _vector(payload["target_point_camera_m"], field="target_point_camera_m", length=3)
    _vector(payload["target_point_base_m"], field="target_point_base_m", length=3)
    bbox = payload["target_mask_bbox_xyxy"]
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
        or not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3])
    ):
        raise ValueError("Point-grasp target mask bbox is invalid.")


def validate_point_grasp_result_for_parameters_v1(parameters: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    for field in ("input_source", "target_pixel_uv", "camera_frame_id", "base_frame_id"):
        if parameters.get(field) != result.get(field):
            raise ValueError(f"Point-grasp result {field} differs from the request.")
    if result.get("candidate_count", 0) > parameters.get("max_candidates", 0):
        raise ValueError("Point-grasp result exceeds the requested candidate bound.")
    intrinsics = parameters["camera_intrinsics"]
    x0, y0, x1, y1 = result["target_mask_bbox_xyxy"]
    if x1 > intrinsics["width"] or y1 > intrinsics["height"]:
        raise ValueError("Point-grasp target mask bbox exceeds the input image.")
    base_from_camera = _matrix(parameters["T_base_from_camera"], field="T_base_from_camera")
    target_camera = np.r_[_vector(result["target_point_camera_m"], field="target_point_camera_m", length=3), 1.0]
    target_base = _vector(result["target_point_base_m"], field="target_point_base_m", length=3)
    if not np.allclose((base_from_camera @ target_camera)[:3], target_base, atol=1e-7):
        raise ValueError("Point-grasp target base point differs from the calibrated transform.")
    for candidate in result["candidates"]:
        camera_from_tcp = _matrix(candidate["T_camera_from_tcp"], field="T_camera_from_tcp")
        base_from_tcp = _matrix(candidate["T_base_from_tcp"], field="T_base_from_tcp")
        if not np.allclose(base_from_camera @ camera_from_tcp, base_from_tcp, atol=1e-7):
            raise ValueError("Point-grasp base pose differs from the calibrated transform.")


__all__ = [
    "point_grasp_schema_entry_v1",
    "validate_point_grasp_parameters_v1",
    "validate_point_grasp_result_for_parameters_v1",
    "validate_point_grasp_result_v1",
]

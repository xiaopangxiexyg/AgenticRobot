"""Portable schemas and validators for passive depth capture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import CAPTURE_DEPTH_SKILL_ID_V1

DEPTH_FRAME_MEDIA_TYPE_V1 = "application/x-npy"
DEPTH_FRAME_SCHEMA_ID_V1 = "dexteleop.perception.depth_image_meters.v1"
DEPTH_FRAME_ENCODING_V1 = "npy.float32.c_order.v1"
DEPTH_FRAME_COMPRESSION_V1 = "none"


def _view_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(values)
    if (
        not normalized
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in normalized
        )
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("allowed_view_ids must contain unique non-empty strings.")
    return tuple(sorted(normalized))


def _pose_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 7,
        "maxItems": 7,
        "items": {"type": "number"},
    }


def _intrinsics_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "fx": {"type": "number", "exclusiveMinimum": 0.0},
            "fy": {"type": "number", "exclusiveMinimum": 0.0},
            "cx": {"type": "number", "minimum": 0.0},
            "cy": {"type": "number", "minimum": 0.0},
            "fovy_deg": {"type": "number", "exclusiveMinimum": 0.0},
        },
        "required": ["width", "height", "fx", "fy", "cx", "cy", "fovy_deg"],
    }


def capture_depth_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_view_ids: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    """Build the exact schema triple for one deployment view catalog."""

    views = _view_ids(allowed_view_ids)
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "view_id": {"type": "string", "enum": list(views)},
        },
        "required": ["view_id"],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "view_id": {"type": "string", "enum": list(views)},
            "captured_at": {"type": "number"},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "depth_unit": {"const": "meter"},
            "invalid_depth_value": {"const": 0.0},
            "camera_pose_world_xyz_quat_xyzw": _pose_schema(),
            "intrinsics": _intrinsics_schema(),
        },
        "required": [
            "view_id",
            "captured_at",
            "width",
            "height",
            "depth_unit",
            "invalid_depth_value",
            "camera_pose_world_xyz_quat_xyzw",
            "intrinsics",
        ],
    }
    task_evaluation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }
    profile = SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema,
            schema_kind="parameter",
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=CAPTURE_DEPTH_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_capture_depth_parameters_v1(
    payload: Mapping[str, Any],
    *,
    allowed_view_ids: tuple[str, ...],
) -> None:
    views = _view_ids(allowed_view_ids)
    if not isinstance(payload, Mapping) or set(payload) != {"view_id"}:
        raise ValueError("Depth capture parameters require exactly view_id.")
    if payload["view_id"] not in views:
        raise ValueError("Depth capture view_id is not registered.")


def validate_capture_depth_result_v1(
    payload: Mapping[str, Any],
    *,
    dimensions_by_view: Mapping[str, tuple[int, int]],
) -> None:
    fields = {
        "view_id",
        "captured_at",
        "width",
        "height",
        "depth_unit",
        "invalid_depth_value",
        "camera_pose_world_xyz_quat_xyzw",
        "intrinsics",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Depth capture result fields do not match the schema.")
    view_id = payload["view_id"]
    try:
        expected_width, expected_height = dimensions_by_view[view_id]
    except (KeyError, TypeError) as exc:
        raise ValueError("Depth capture result uses an unknown view.") from exc
    if any(
        isinstance(payload[field_name], bool)
        or not isinstance(payload[field_name], int)
        or payload[field_name] <= 0
        for field_name in ("width", "height")
    ):
        raise ValueError("Depth capture dimensions must be positive integers.")
    if (payload["width"], payload["height"]) != (
        expected_width,
        expected_height,
    ):
        raise ValueError("Depth capture dimensions do not match the view profile.")
    invalid_depth = payload["invalid_depth_value"]
    if (
        payload["depth_unit"] != "meter"
        or isinstance(invalid_depth, bool)
        or not isinstance(invalid_depth, (int, float))
        or invalid_depth != 0.0
    ):
        raise ValueError("Depth capture unit or invalid-value semantics changed.")
    captured_at = payload["captured_at"]
    if isinstance(captured_at, bool) or not isinstance(captured_at, (int, float)):
        raise TypeError("captured_at must be numeric.")
    if not math.isfinite(float(captured_at)):
        raise ValueError("captured_at must be finite.")
    pose = payload["camera_pose_world_xyz_quat_xyzw"]
    if (
        not isinstance(pose, (list, tuple))
        or len(pose) != 7
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in pose
        )
    ):
        raise ValueError("Camera pose must contain seven finite numbers.")
    quaternion_norm = math.sqrt(sum(float(item) ** 2 for item in pose[3:]))
    if abs(quaternion_norm - 1.0) > 1e-5:
        raise ValueError("Camera pose quaternion must be normalized quat_xyzw.")
    intrinsics = payload["intrinsics"]
    intrinsic_fields = {"width", "height", "fx", "fy", "cx", "cy", "fovy_deg"}
    if not isinstance(intrinsics, Mapping) or set(intrinsics) != intrinsic_fields:
        raise ValueError("Depth intrinsics fields do not match the schema.")
    if any(
        isinstance(intrinsics[field_name], bool)
        or not isinstance(intrinsics[field_name], int)
        or intrinsics[field_name] <= 0
        for field_name in ("width", "height")
    ):
        raise ValueError("Depth intrinsics dimensions must be positive integers.")
    if (intrinsics["width"], intrinsics["height"]) != (
        expected_width,
        expected_height,
    ):
        raise ValueError("Depth intrinsics dimensions do not match the frame.")
    for field_name in ("fx", "fy", "cx", "cy", "fovy_deg"):
        value = intrinsics[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"Depth intrinsic {field_name} must be finite.")
    if intrinsics["fx"] <= 0 or intrinsics["fy"] <= 0 or intrinsics["fovy_deg"] <= 0:
        raise ValueError("Depth focal lengths and fovy_deg must be positive.")
    if intrinsics["cx"] < 0 or intrinsics["cy"] < 0:
        raise ValueError("Depth principal point must be non-negative.")


def validate_capture_depth_result_for_parameters_v1(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if parameters.get("view_id") != result.get("view_id"):
        raise ValueError("Depth result view_id differs from the request.")


def validate_no_task_evaluation_v1(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or payload:
        raise ValueError("Passive depth capture has no task evaluation payload.")


__all__ = [
    "DEPTH_FRAME_COMPRESSION_V1",
    "DEPTH_FRAME_ENCODING_V1",
    "DEPTH_FRAME_MEDIA_TYPE_V1",
    "DEPTH_FRAME_SCHEMA_ID_V1",
    "capture_depth_schema_entry_v1",
    "validate_capture_depth_parameters_v1",
    "validate_capture_depth_result_for_parameters_v1",
    "validate_capture_depth_result_v1",
    "validate_no_task_evaluation_v1",
]

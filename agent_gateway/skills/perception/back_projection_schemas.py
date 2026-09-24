"""Portable schemas for live depth back-projection into a bound Station frame."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import BACK_PROJECT_SKILL_ID_V1

_PARAMETER_FIELDS = {
    "role",
    "target_pixel_uv",
    "pixel_region_xyxy",
    "world_z_min_m",
    "world_z_max_m",
}
_RESULT_FIELDS = {
    "source_id",
    "role",
    "captured_at",
    "station_revision",
    "capture_group_id",
    "depth_content_hash",
    "image_width",
    "image_height",
    "reference_frame_id",
    "reference_frame_spec_hash",
    "frame_graph_hash",
    "camera_pose_reference_xyz_quat_xyzw",
    "coordinate_mode",
    "target_pixel_uv",
    "pixel_region_xyxy",
    "world_z_min_m",
    "world_z_max_m",
    "world_center_xyz_m",
    "world_median_xyz_m",
    "world_bounds_xyz_m",
    "world_extent_xyz_m",
    "depth_m",
    "valid_point_count",
}


def _roles(values: tuple[str, ...]) -> tuple[str, ...]:
    roles = tuple(sorted(values))
    if (
        not roles
        or len(set(roles)) != len(roles)
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in roles
        )
    ):
        raise ValueError("allowed_roles must contain unique non-empty strings.")
    return roles


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _digest(value: object, *, field_name: str) -> str:
    text = _required_text(value, field_name=field_name)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return text


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _optional_finite(value: object, *, field_name: str) -> float | None:
    return None if value is None else _finite(value, field_name=field_name)


def _integer_vector(
    value: object,
    *,
    field_name: str,
    length: int,
) -> tuple[int, ...] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != length
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise ValueError(
            f"{field_name} must be null or {length} non-negative integers."
        )
    return tuple(int(item) for item in value)


def _vector(value: object, *, field_name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain {length} numbers.")
    return tuple(
        _finite(item, field_name=f"{field_name}[]") for item in value
    )


def _parameter_values(
    payload: Mapping[str, Any],
    *,
    allowed_roles: tuple[str, ...],
) -> tuple[
    tuple[int, int] | None,
    tuple[int, int, int, int] | None,
    float | None,
    float | None,
]:
    if not isinstance(payload, Mapping) or set(payload) != _PARAMETER_FIELDS:
        raise ValueError("Back-project parameter fields do not match the schema.")
    if payload["role"] not in _roles(allowed_roles):
        raise ValueError("Back-project role is not registered.")
    pixel = _integer_vector(
        payload["target_pixel_uv"], field_name="target_pixel_uv", length=2
    )
    region = _integer_vector(
        payload["pixel_region_xyxy"], field_name="pixel_region_xyxy", length=4
    )
    if (pixel is None) == (region is None):
        raise ValueError("Back-project requires exactly one pixel or region target.")
    z_min = _optional_finite(payload["world_z_min_m"], field_name="world_z_min_m")
    z_max = _optional_finite(payload["world_z_max_m"], field_name="world_z_max_m")
    if pixel is not None and (z_min is not None or z_max is not None):
        raise ValueError("Station Z filtering is supported only in region mode.")
    if z_min is not None and z_max is not None and z_min > z_max:
        raise ValueError("world_z_min_m must not exceed world_z_max_m.")
    if region is not None:
        x0, y0, x1, y1 = region
        if x0 >= x1 or y0 >= y1:
            raise ValueError("pixel_region_xyxy must be a non-empty half-open region.")
    return pixel, region, z_min, z_max


def back_project_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_roles: tuple[str, ...],
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
) -> AgentSkillSchemaEntry:
    """Build a provider-friendly schema; mode exclusivity stays in validators."""

    roles = _roles(allowed_roles)
    reference = _required_text(reference_frame_id, field_name="reference_frame_id")
    reference_hash = _required_text(
        reference_frame_spec_hash, field_name="reference_frame_spec_hash"
    )
    graph_hash = _required_text(frame_graph_hash, field_name="frame_graph_hash")
    nullable_number = {"type": ["number", "null"]}
    nullable_pixel = {
        "type": ["array", "null"],
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 0},
        "description": (
            "Zero-based [u, v] pixel. Set this or pixel_region_xyxy, but not both."
        ),
    }
    nullable_region = {
        "type": ["array", "null"],
        "minItems": 4,
        "maxItems": 4,
        "items": {"type": "integer", "minimum": 0},
        "description": (
            "Zero-based half-open [x0, y0, x1, y1] region. Set this or "
            "target_pixel_uv, but not both."
        ),
    }
    vector3 = {
        "type": "array",
        "minItems": 3,
        "maxItems": 3,
        "items": {"type": "number"},
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": list(roles)},
            "target_pixel_uv": nullable_pixel,
            "pixel_region_xyxy": nullable_region,
            "world_z_min_m": {
                **nullable_number,
                "description": "Optional Station-frame lower Z bound for region mode.",
            },
            "world_z_max_m": {
                **nullable_number,
                "description": "Optional Station-frame upper Z bound for region mode.",
            },
        },
        "required": sorted(_PARAMETER_FIELDS),
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "captured_at": {"type": "number"},
            "station_revision": {"type": "integer", "minimum": 0},
            "capture_group_id": {"type": "string", "minLength": 1},
            "depth_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "image_width": {"type": "integer", "minimum": 1},
            "image_height": {"type": "integer", "minimum": 1},
            "reference_frame_id": {"const": reference},
            "reference_frame_spec_hash": {"const": reference_hash},
            "frame_graph_hash": {"const": graph_hash},
            "camera_pose_reference_xyz_quat_xyzw": {
                "type": "array",
                "minItems": 7,
                "maxItems": 7,
                "items": {"type": "number"},
            },
            "coordinate_mode": {"type": "string", "enum": ["pixel", "region"]},
            "target_pixel_uv": nullable_pixel,
            "pixel_region_xyxy": nullable_region,
            "world_z_min_m": nullable_number,
            "world_z_max_m": nullable_number,
            "world_center_xyz_m": vector3,
            "world_median_xyz_m": vector3,
            "world_bounds_xyz_m": {
                "type": "object",
                "additionalProperties": False,
                "properties": {"low": vector3, "high": vector3},
                "required": ["low", "high"],
            },
            "world_extent_xyz_m": vector3,
            "depth_m": {"type": "number", "exclusiveMinimum": 0.0},
            "valid_point_count": {"type": "integer", "minimum": 1},
        },
        "required": sorted(_RESULT_FIELDS),
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
            parameter_schema, schema_kind="parameter"
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=BACK_PROJECT_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_back_project_parameters_v1(
    payload: Mapping[str, Any], *, allowed_roles: tuple[str, ...]
) -> None:
    _parameter_values(payload, allowed_roles=allowed_roles)


def validate_back_project_result_v1(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    dimensions_by_role: Mapping[str, tuple[int, int]],
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    minimum_region_valid_points: int,
    maximum_depth_m: float,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _RESULT_FIELDS:
        raise ValueError("Back-project result fields do not match the schema.")
    role = _required_text(payload["role"], field_name="role")
    if role not in source_id_by_role or payload["source_id"] != source_id_by_role[role]:
        raise ValueError("Back-project result changed its registered source binding.")
    width, height = dimensions_by_role[role]
    if payload["image_width"] != width or payload["image_height"] != height:
        raise ValueError("Back-project image dimensions changed.")
    if payload["reference_frame_id"] != reference_frame_id:
        raise ValueError("Back-project reference frame changed.")
    if payload["reference_frame_spec_hash"] != reference_frame_spec_hash:
        raise ValueError("Back-project reference-frame specification changed.")
    if payload["frame_graph_hash"] != frame_graph_hash:
        raise ValueError("Back-project frame graph changed.")
    _finite(payload["captured_at"], field_name="captured_at")
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("station_revision must be non-negative.")
    _required_text(payload["capture_group_id"], field_name="capture_group_id")
    _digest(payload["depth_content_hash"], field_name="depth_content_hash")
    pose = _vector(
        payload["camera_pose_reference_xyz_quat_xyzw"],
        field_name="camera_pose_reference_xyz_quat_xyzw",
        length=7,
    )
    if not math.isclose(
        math.sqrt(sum(value * value for value in pose[3:])),
        1.0,
        abs_tol=1e-5,
    ):
        raise ValueError("Back-project camera quaternion is not normalized.")
    parameter_projection = {
        "role": role,
        "target_pixel_uv": payload["target_pixel_uv"],
        "pixel_region_xyxy": payload["pixel_region_xyxy"],
        "world_z_min_m": payload["world_z_min_m"],
        "world_z_max_m": payload["world_z_max_m"],
    }
    pixel, region, _z_min, _z_max = _parameter_values(
        parameter_projection, allowed_roles=tuple(source_id_by_role)
    )
    mode = payload["coordinate_mode"]
    if mode != ("pixel" if pixel is not None else "region"):
        raise ValueError("Back-project coordinate mode and target disagree.")
    if pixel is not None and (pixel[0] >= width or pixel[1] >= height):
        raise ValueError("Back-project pixel is outside the image.")
    if region is not None and (region[2] > width or region[3] > height):
        raise ValueError("Back-project region is outside the image.")
    center = _vector(payload["world_center_xyz_m"], field_name="world_center_xyz_m", length=3)
    median = _vector(payload["world_median_xyz_m"], field_name="world_median_xyz_m", length=3)
    bounds = payload["world_bounds_xyz_m"]
    if not isinstance(bounds, Mapping) or set(bounds) != {"low", "high"}:
        raise ValueError("world_bounds_xyz_m must contain low and high.")
    low = _vector(bounds["low"], field_name="world_bounds_xyz_m.low", length=3)
    high = _vector(bounds["high"], field_name="world_bounds_xyz_m.high", length=3)
    extent = _vector(payload["world_extent_xyz_m"], field_name="world_extent_xyz_m", length=3)
    if any(lower > upper for lower, upper in zip(low, high, strict=True)):
        raise ValueError("Back-project world bounds are inverted.")
    if any(
        not math.isclose(value, upper - lower, abs_tol=2e-6)
        for value, lower, upper in zip(extent, low, high, strict=True)
    ):
        raise ValueError("Back-project extent differs from its world bounds.")
    if any(
        value < lower - 2e-6 or value > upper + 2e-6
        for values in (center, median)
        for value, lower, upper in zip(values, low, high, strict=True)
    ):
        raise ValueError("Back-project center or median lies outside its bounds.")
    depth = _finite(payload["depth_m"], field_name="depth_m")
    if not 0.0 < depth <= maximum_depth_m:
        raise ValueError("Back-project depth is outside the deployment bound.")
    count = payload["valid_point_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("valid_point_count must be positive.")
    if mode == "pixel" and count != 1:
        raise ValueError("Pixel back-projection must contain exactly one point.")
    if mode == "region" and count < minimum_region_valid_points:
        raise ValueError("Region back-projection has too few valid points.")


def validate_back_project_result_for_parameters_v1(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    for field_name in _PARAMETER_FIELDS:
        if result[field_name] != parameters[field_name]:
            raise ValueError(
                f"Back-project result field {field_name!r} differs from the request."
            )


__all__ = [
    "back_project_schema_entry_v1",
    "validate_back_project_parameters_v1",
    "validate_back_project_result_for_parameters_v1",
    "validate_back_project_result_v1",
]

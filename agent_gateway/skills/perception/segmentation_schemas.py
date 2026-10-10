"""Portable schemas and validators for passive live-image segmentation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import SEGMENT_SKILL_ID_V1


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


def _optional_text(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 256
    ):
        raise ValueError(f"{field_name} must be null or a bounded trimmed string.")
    return value


def _optional_pixel(value: object, *, field_name: str) -> tuple[int, int] | None:
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise ValueError(f"{field_name} must be null or two non-negative integers.")
    return int(value[0]), int(value[1])


def _digest(value: object, *, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    return value


def segment_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_roles: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    """Build a provider-friendly schema; exclusivity is enforced by validators."""

    roles = _roles(allowed_roles)
    nullable_prompt = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": 256,
        "description": (
            "Object description. Set this or target_pixel_uv, but not both."
        ),
    }
    nullable_pixel = {
        "type": ["array", "null"],
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 0},
        "description": (
            "Zero-based [u, v] pixel prompt. Set this or language_prompt, but not both."
        ),
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": list(roles)},
            "language_prompt": nullable_prompt,
            "target_pixel_uv": nullable_pixel,
            "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["role", "language_prompt", "target_pixel_uv", "min_score"],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "captured_at": {"type": "number"},
            "station_revision": {"type": "integer", "minimum": 0},
            "input_rgb_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "image_width": {"type": "integer", "minimum": 1},
            "image_height": {"type": "integer", "minimum": 1},
            "mode": {"type": "string", "enum": ["text", "point"]},
            "language_prompt": nullable_prompt,
            "target_pixel_uv": nullable_pixel,
            "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "label": nullable_prompt,
            "coordinate_format": {"const": "xyxy_pixel_0based_half_open"},
            "mask_bounds_xyxy": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0},
            },
            "mask_pixel_count": {"type": "integer", "minimum": 1},
            "model_profile_id": {"type": "string", "minLength": 1},
            "model_profile_hash": {"type": "string", "minLength": 1},
            "mask_media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "overlay_media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "source_id",
            "role",
            "captured_at",
            "station_revision",
            "input_rgb_content_hash",
            "image_width",
            "image_height",
            "mode",
            "language_prompt",
            "target_pixel_uv",
            "min_score",
            "score",
            "label",
            "coordinate_format",
            "mask_bounds_xyxy",
            "mask_pixel_count",
            "model_profile_id",
            "model_profile_hash",
            "mask_media_content_hash",
            "overlay_media_content_hash",
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
        skill_id=SEGMENT_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_segment_parameters_v1(
    payload: Mapping[str, Any], *, allowed_roles: tuple[str, ...]
) -> None:
    fields = {"role", "language_prompt", "target_pixel_uv", "min_score"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Segment parameter fields do not match the schema.")
    if payload["role"] not in _roles(allowed_roles):
        raise ValueError("Segment role is not registered.")
    prompt = _optional_text(payload["language_prompt"], field_name="language_prompt")
    pixel = _optional_pixel(payload["target_pixel_uv"], field_name="target_pixel_uv")
    if (prompt is None) == (pixel is None):
        raise ValueError("Segment requires exactly one language or pixel prompt.")
    score = payload["min_score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not math.isfinite(float(score))
        or not 0.0 <= float(score) <= 1.0
    ):
        raise ValueError("Segment min_score must be finite and in [0, 1].")


def validate_segment_result_v1(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    model_profile_id: str,
    model_profile_hash: str,
) -> None:
    fields = {
        "source_id",
        "role",
        "captured_at",
        "station_revision",
        "input_rgb_content_hash",
        "image_width",
        "image_height",
        "mode",
        "language_prompt",
        "target_pixel_uv",
        "min_score",
        "score",
        "label",
        "coordinate_format",
        "mask_bounds_xyxy",
        "mask_pixel_count",
        "model_profile_id",
        "model_profile_hash",
        "mask_media_content_hash",
        "overlay_media_content_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Segment result fields do not match the schema.")
    role = payload["role"]
    if source_id_by_role.get(role) != payload["source_id"]:
        raise ValueError("Segment source differs from its registered role.")
    captured_at = payload["captured_at"]
    if (
        isinstance(captured_at, bool)
        or not isinstance(captured_at, (int, float))
        or not math.isfinite(float(captured_at))
    ):
        raise ValueError("Segment captured_at must be finite.")
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Segment station_revision must be non-negative.")
    for name in (
        "input_rgb_content_hash",
        "mask_media_content_hash",
        "overlay_media_content_hash",
    ):
        _digest(payload[name], field_name=name)
    width = payload["image_width"]
    height = payload["image_height"]
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, height)
    ):
        raise ValueError("Segment image dimensions must be positive integers.")
    prompt = _optional_text(payload["language_prompt"], field_name="language_prompt")
    pixel = _optional_pixel(payload["target_pixel_uv"], field_name="target_pixel_uv")
    if (prompt is None) == (pixel is None):
        raise ValueError("Segment result requires exactly one prompt.")
    expected_mode = "text" if prompt is not None else "point"
    if payload["mode"] != expected_mode:
        raise ValueError("Segment mode differs from its prompt.")
    if pixel is not None and not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
        raise ValueError("Segment point prompt is outside the image.")
    for name in ("min_score", "score"):
        value = payload[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"Segment {name} must be finite and in [0, 1].")
    if float(payload["score"]) < float(payload["min_score"]):
        raise ValueError("Segment score is below the admitted threshold.")
    _optional_text(payload["label"], field_name="label")
    if payload["coordinate_format"] != "xyxy_pixel_0based_half_open":
        raise ValueError("Segment coordinate semantics changed.")
    box = payload["mask_bounds_xyxy"]
    if (
        not isinstance(box, (list, tuple))
        or len(box) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in box)
    ):
        raise ValueError("Segment mask bounds must contain four integers.")
    x0, y0, x1, y1 = box
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Segment mask bounds are outside the image.")
    count = payload["mask_pixel_count"]
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 1 <= count <= width * height
    ):
        raise ValueError("Segment mask pixel count is invalid.")
    if (
        payload["model_profile_id"] != model_profile_id
        or payload["model_profile_hash"] != model_profile_hash
    ):
        raise ValueError("Segment model identity changed.")


def validate_segment_result_for_parameters_v1(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    for field_name in ("role", "language_prompt", "min_score"):
        if parameters.get(field_name) != result.get(field_name):
            raise ValueError(f"Segment result {field_name} differs from request.")
    if _optional_pixel(
        parameters.get("target_pixel_uv"),
        field_name="target_pixel_uv",
    ) != _optional_pixel(
        result.get("target_pixel_uv"),
        field_name="target_pixel_uv",
    ):
        raise ValueError("Segment result target_pixel_uv differs from request.")


__all__ = [
    "segment_schema_entry_v1",
    "validate_segment_parameters_v1",
    "validate_segment_result_for_parameters_v1",
    "validate_segment_result_v1",
]

"""Portable schemas and validators for passive live-image segmentation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
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
    "segment_schema_entry_v2",
    "segment_schema_entry_v3",
    "validate_segment_parameters_v1",
    "validate_segment_parameters_v2",
    "validate_segment_parameters_v3",
    "validate_segment_result_for_parameters_v1",
    "validate_segment_result_for_parameters_v2",
    "validate_segment_result_for_parameters_v3",
    "validate_segment_result_v1",
    "validate_segment_result_v2",
    "validate_segment_result_v3",
]


_COMMON_FIELDS = (
    "source_id", "role", "captured_at", "station_revision", "input_rgb_content_hash",
    "image_width", "image_height", "model_profile_id", "model_profile_hash",
)
_PROMPT_FIELDS = ("language_prompt", "target_pixel_uv", "min_score")
_MASK_FIELDS = (
    "score", "label", "coordinate_format", "mask_bounds_xyxy", "mask_pixel_count",
    "mask_media_content_hash",
)


def _closed(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
        "properties": properties, "required": list(properties)}


def segment_schema_entry_v2(
    *, skill_definition_hash: str, parameter_profile_id: str,
    parameter_profile_hash: str, allowed_roles: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    """Version two processes ordered targets without changing v1 semantics."""
    original = segment_schema_entry_v1(
        skill_definition_hash=skill_definition_hash, parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash, allowed_roles=allowed_roles,
    )
    prompts = original.parameter_schema["properties"]
    result = original.execution_result_schema["properties"]
    parameters = _closed({
        "role": prompts["role"],
        "targets": {"type": "array", "minItems": 1, "maxItems": 3,
            "items": _closed({key: prompts[key] for key in _PROMPT_FIELDS})},
    })
    output = _closed({
        **{key: result[key] for key in _COMMON_FIELDS},
        "overlay_media_content_hash": result["overlay_media_content_hash"],
        "completed_count": {"type": "integer", "minimum": 0, "maximum": 3},
        "no_mask_count": {"type": "integer", "minimum": 0, "maximum": 3},
        "items": {"type": "array", "minItems": 1, "maxItems": 3, "items": _closed({
            "index": {"type": "integer", "minimum": 0, "maximum": 2},
            **{key: prompts[key] for key in _PROMPT_FIELDS},
            "status": {"type": "string", "enum": ["completed", "no_mask"]},
            "mask_artifact_ordinal": {"type": ["integer", "null"], "minimum": 0, "maximum": 2},
            "mask": {"anyOf": [{"type": "null"}, _closed({key: result[key] for key in _MASK_FIELDS})]},
        })},
    })
    return AgentSkillSchemaEntry(
        skill_id=SEGMENT_SKILL_ID_V1, skill_version="2", skill_definition_hash=skill_definition_hash,
        parameter_schema=parameters, execution_result_schema=output,
        task_evaluation_schema=original.task_evaluation_schema,
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id=parameter_profile_id, parameter_profile_hash=parameter_profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter"),
        ),),
    )


def validate_segment_parameters_v2(payload: Mapping[str, Any], *, allowed_roles: tuple[str, ...]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {"role", "targets"}:
        raise ValueError("Batch segment parameter fields do not match the schema.")
    targets = payload["targets"]
    if not isinstance(targets, (list, tuple)) or not 1 <= len(targets) <= 3:
        raise ValueError("Batch segment requires one to three ordered targets.")
    for target in targets:
        if not isinstance(target, Mapping) or set(target) != set(_PROMPT_FIELDS):
            raise ValueError("Batch target fields do not match the schema.")
        validate_segment_parameters_v1({"role": payload["role"], **target}, allowed_roles=allowed_roles)


def validate_segment_result_v2(
    payload: Mapping[str, Any], *, source_id_by_role: Mapping[str, str],
    model_profile_id: str, model_profile_hash: str,
) -> None:
    fields = {*_COMMON_FIELDS, "overlay_media_content_hash", "completed_count", "no_mask_count", "items"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Batch segment result fields do not match the schema.")
    if source_id_by_role.get(payload["role"]) != payload["source_id"]:
        raise ValueError("Batch source differs from its registered role.")
    if (isinstance(payload["captured_at"], bool) or not isinstance(payload["captured_at"], (int, float))
            or not math.isfinite(float(payload["captured_at"]))):
        raise ValueError("Batch capture time must be finite.")
    if type(payload["station_revision"]) is not int or payload["station_revision"] < 0:
        raise ValueError("Batch revision must be non-negative.")
    if any(type(payload[key]) is not int or payload[key] < 1 for key in ("image_width", "image_height")):
        raise ValueError("Batch image dimensions must be positive integers.")
    for key in ("input_rgb_content_hash", "overlay_media_content_hash"):
        _digest(payload[key], field_name=key)
    if payload["model_profile_id"] != model_profile_id or payload["model_profile_hash"] != model_profile_hash:
        raise ValueError("Batch model identity changed.")
    items = payload["items"]
    if not isinstance(items, (list, tuple)) or not 1 <= len(items) <= 3:
        raise ValueError("Batch result must preserve the ordered targets.")
    # Reuse the v1 value checks, including common identity, for every item.
    completed = 0
    for index, item in enumerate(items):
        if (not isinstance(item, Mapping) or set(item) != {
                "index", *_PROMPT_FIELDS, "status", "mask_artifact_ordinal", "mask"}
                or type(item["index"]) is not int or item["index"] != index):
            raise ValueError("Batch result changed its original target order.")
        validate_segment_parameters_v1(
            {"role": payload["role"], **{key: item[key] for key in _PROMPT_FIELDS}},
            allowed_roles=tuple(source_id_by_role),
        )
        pixel = item["target_pixel_uv"]
        width, height = payload["image_width"], payload["image_height"]
        mask = item["mask"]
        if item["status"] == "completed":
            if (type(item["mask_artifact_ordinal"]) is not int or item["mask_artifact_ordinal"] != completed
                    or not isinstance(mask, Mapping) or set(mask) != set(_MASK_FIELDS)):
                raise ValueError("Batch mask ordinals must densely map successful items.")
            completed += 1
            validate_segment_result_v1({
                **{key: payload[key] for key in _COMMON_FIELDS},
                **{key: item[key] for key in _PROMPT_FIELDS}, **mask,
                "mode": "text" if item["language_prompt"] is not None else "point",
                "overlay_media_content_hash": payload["overlay_media_content_hash"],
            }, source_id_by_role=source_id_by_role,
                model_profile_id=model_profile_id, model_profile_hash=model_profile_hash)
        elif item["status"] == "no_mask":
            if mask is not None or item["mask_artifact_ordinal"] is not None:
                raise ValueError("A no-mask item cannot advertise an Artifact.")
        else:
            raise ValueError("Unknown batch item status.")
        if pixel is not None and not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            raise ValueError("Batch point prompt is outside the image.")
    if (type(payload["completed_count"]) is not int or type(payload["no_mask_count"]) is not int
            or payload["completed_count"] != completed or payload["no_mask_count"] != len(items) - completed):
        raise ValueError("Batch result counts disagree with the item outcomes.")


def validate_segment_result_for_parameters_v2(parameters: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    if parameters.get("role") != result.get("role") or len(parameters["targets"]) != len(result["items"]):
        raise ValueError("Batch segmentation changed the requested role or target count.")
    for target, item in zip(parameters["targets"], result["items"]):
        validate_segment_result_for_parameters_v1({"role": parameters["role"], **target},
            {"role": result["role"], **item})


SEGMENT_GEOMETRY_METRIC_V3 = "euclidean_distance_between_occupied_pixel_centers"
SEGMENT_GEOMETRY_MAX_WIDTH_V3 = 1920
SEGMENT_GEOMETRY_MAX_HEIGHT_V3 = 1080
SEGMENT_RGB_PROVENANCE_FIELDS_V3 = (
    "rgb_capture_operation_id", "rgb_capture_request_hash",
    "input_rgb_artifact_id", "input_rgb_frame_ordinal",
)


def validate_segment_parameters_v3(payload: Mapping[str, Any], *, allowed_roles: tuple[str, ...]) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {"role", "targets", "rgb_capture_operation_id"}:
        raise ValueError("RGB-input segment parameter fields do not match the schema.")
    if _optional_text(payload["rgb_capture_operation_id"], field_name="rgb_capture_operation_id") is None:
        raise ValueError("RGB-input segment requires a capture producer operation ID.")
    validate_segment_parameters_v2(
        {key: payload[key] for key in ("role", "targets")}, allowed_roles=allowed_roles,
    )


def validate_segment_result_for_parameters_v3(parameters: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    validate_segment_result_for_parameters_v2(parameters, result)
    if parameters["rgb_capture_operation_id"] != result["rgb_capture_operation_id"]:
        raise ValueError("RGB-input segment changed its requested producer operation.")


def segment_schema_entry_v3(
    *, skill_definition_hash: str, parameter_profile_id: str,
    parameter_profile_hash: str, allowed_roles: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    """Bind a required RGB producer and add explicit historical 2D relations."""
    original = segment_schema_entry_v2(
        skill_definition_hash=skill_definition_hash,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        allowed_roles=allowed_roles,
    )
    pixel = {"type": "array", "minItems": 2, "maxItems": 2,
        "items": {"type": "integer", "minimum": 0}}
    properties = dict(original.execution_result_schema["properties"])
    identifier = {"type": "string", "minLength": 1, "maxLength": 256}
    parameters = _closed({**dict(original.parameter_schema["properties"]),
        "rgb_capture_operation_id": {**identifier,
            "description": "Completed capture_rgb operation that produced input_image; no fresh capture fallback."}})
    properties.update({
        "rgb_capture_operation_id": identifier,
        "rgb_capture_request_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "input_rgb_artifact_id": identifier,
        "input_rgb_frame_ordinal": {"type": "integer", "minimum": 0, "maximum": 7},
    })
    properties["image_width"] = {"type": "integer", "minimum": 1,
        "maximum": SEGMENT_GEOMETRY_MAX_WIDTH_V3}
    properties["image_height"] = {"type": "integer", "minimum": 1,
        "maximum": SEGMENT_GEOMETRY_MAX_HEIGHT_V3}
    properties["relations"] = {
        "type": "array", "minItems": 0, "maxItems": 3,
        "items": _closed({
            "indices": {"type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "integer", "minimum": 0, "maximum": 2}},
            "metric": {"const": SEGMENT_GEOMETRY_METRIC_V3},
            "status": {"type": "string", "enum": ["separated", "overlap", "unavailable"]},
            "unavailable_reason": {"enum": [None, "absent_mask"]},
            "distance_px": {"type": ["number", "null"], "minimum": 0},
            "closest_pixels_uv": {"anyOf": [{"type": "null"},
                {"type": "array", "minItems": 2, "maxItems": 2, "items": pixel}]},
            "semantic_identity_verified": {"const": False},
            "physical_clearance_m": {"type": "null"},
        }),
    }
    return replace(original, skill_version="3", parameter_schema=parameters,
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id=parameter_profile_id, parameter_profile_hash=parameter_profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter")),),
        execution_result_schema=_closed(properties))


def validate_segment_result_v3(
    payload: Mapping[str, Any], *, source_id_by_role: Mapping[str, str],
    model_profile_id: str, model_profile_hash: str,
) -> None:
    if not isinstance(payload, Mapping) or not {"relations", *SEGMENT_RGB_PROVENANCE_FIELDS_V3} <= set(payload):
        raise ValueError("Geometry segment result must include its relations and RGB provenance.")
    for key in ("rgb_capture_operation_id", "input_rgb_artifact_id"):
        if _optional_text(payload[key], field_name=key) is None:
            raise ValueError("Geometry RGB provenance identifiers cannot be null.")
    _digest(payload["rgb_capture_request_hash"], field_name="rgb_capture_request_hash")
    ordinal = payload["input_rgb_frame_ordinal"]
    if type(ordinal) is not int or not 0 <= ordinal <= 7:
        raise ValueError("Geometry RGB source ordinal is outside the capture bound.")
    validate_segment_result_v2(
        {key: value for key, value in payload.items() if key not in {"relations", *SEGMENT_RGB_PROVENANCE_FIELDS_V3}},
        source_id_by_role=source_id_by_role,
        model_profile_id=model_profile_id,
        model_profile_hash=model_profile_hash,
    )
    width, height = payload["image_width"], payload["image_height"]
    if width > SEGMENT_GEOMETRY_MAX_WIDTH_V3 or height > SEGMENT_GEOMETRY_MAX_HEIGHT_V3:
        raise ValueError("Geometry mask grid exceeds its reviewed bound.")
    items = payload["items"]
    pairs = [(one, two) for one in range(len(items)) for two in range(one + 1, len(items))]
    relations = payload["relations"]
    if not isinstance(relations, (list, tuple)) or len(relations) != len(pairs):
        raise ValueError("Geometry must report every unordered target pair exactly once.")
    fields = {"indices", "metric", "status", "unavailable_reason", "distance_px",
        "closest_pixels_uv", "semantic_identity_verified", "physical_clearance_m"}
    for pair, relation in zip(pairs, relations):
        if not isinstance(relation, Mapping) or set(relation) != fields:
            raise ValueError("Geometry relation fields do not match the schema.")
        indices = relation["indices"]
        if (not isinstance(indices, (list, tuple)) or len(indices) != 2
                or any(type(index) is not int for index in indices) or tuple(indices) != pair):
            raise ValueError("Geometry relations must retain canonical unordered pair order.")
        if (relation["metric"] != SEGMENT_GEOMETRY_METRIC_V3
                or relation["semantic_identity_verified"] is not False
                or relation["physical_clearance_m"] is not None):
            raise ValueError("Mask geometry cannot assert semantic identity or physical clearance.")
        if any(items[index]["status"] == "no_mask" for index in pair):
            if (relation["status"] != "unavailable"
                    or relation["unavailable_reason"] != "absent_mask"
                    or relation["distance_px"] is not None
                    or relation["closest_pixels_uv"] is not None):
                raise ValueError("Absent masks require an unavailable relation and null geometry.")
            continue
        distance = relation["distance_px"]
        if (isinstance(distance, bool) or not isinstance(distance, (int, float))
                or not math.isfinite(float(distance)) or distance < 0
                or relation["unavailable_reason"] is not None
                or relation["status"] != ("overlap" if distance == 0 else "separated")):
            raise ValueError("Available mask geometry must report an honest distance and status.")
        endpoints = relation["closest_pixels_uv"]
        if not isinstance(endpoints, (list, tuple)) or len(endpoints) != 2:
            raise ValueError("Available mask geometry requires two pixel endpoints.")
        for index, endpoint in zip(pair, endpoints):
            pixel = _optional_pixel(endpoint, field_name="closest_pixels_uv")
            if pixel is None or not (pixel[0] < width and pixel[1] < height):
                raise ValueError("Geometry endpoint is outside the capture grid.")
            x0, y0, x1, y1 = items[index]["mask"]["mask_bounds_xyxy"]
            if not (x0 <= pixel[0] < x1 and y0 <= pixel[1] < y1):
                raise ValueError("Geometry endpoint is outside its proposal bounds.")
        if not math.isclose(float(distance), math.hypot(
                endpoints[0][0] - endpoints[1][0], endpoints[0][1] - endpoints[1][1]),
                rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Geometry distance disagrees with its pixel endpoints.")

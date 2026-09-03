"""Portable schemas and validators for passive perception capture."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)

from .definition import (
    CAPTURE_DEPTH_SKILL_ID_V1,
    CAPTURE_RGB_SKILL_ID_V1,
    LOCALIZE_OBJECTS_SKILL_ID_V1,
)


def _semantic_roles(values: tuple[str, ...]) -> tuple[str, ...]:
    normalized = tuple(values)
    if (
        not normalized
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in normalized
        )
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError("allowed_roles must contain unique non-empty strings.")
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
    allowed_roles: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    """Build the exact schema triple for one deployment role catalog."""

    roles = _semantic_roles(allowed_roles)
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": list(roles)},
        },
        "required": ["role"],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "captured_at": {"type": "number"},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "depth_unit": {"const": "meter"},
            "invalid_depth_value": {"const": 0.0},
            "camera_pose_world_xyz_quat_xyzw": _pose_schema(),
            "intrinsics": _intrinsics_schema(),
        },
        "required": [
            "source_id",
            "role",
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


def capture_depth_schema_entry_v2(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_roles: tuple[str, ...],
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
) -> AgentSkillSchemaEntry:
    """Build depth v2 with one exact consumer reference-frame identity."""

    roles = _semantic_roles(allowed_roles)
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": list(roles)},
        },
        "required": ["role"],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "captured_at": {"type": "number"},
            "station_revision": {"type": "integer", "minimum": 0},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "depth_unit": {"const": "meter"},
            "invalid_depth_value": {"const": 0.0},
            "reference_frame_id": {"const": reference_frame_id},
            "reference_frame_spec_hash": {
                "const": reference_frame_spec_hash,
            },
            "frame_graph_hash": {"const": frame_graph_hash},
            "camera_pose_reference_xyz_quat_xyzw": _pose_schema(),
            "intrinsics": _intrinsics_schema(),
        },
        "required": [
            "source_id",
            "role",
            "captured_at",
            "station_revision",
            "width",
            "height",
            "depth_unit",
            "invalid_depth_value",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "camera_pose_reference_xyz_quat_xyzw",
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
        skill_version="2",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_capture_depth_parameters_v1(
    payload: Mapping[str, Any],
    *,
    allowed_roles: tuple[str, ...],
) -> None:
    roles = _semantic_roles(allowed_roles)
    if not isinstance(payload, Mapping) or set(payload) != {"role"}:
        raise ValueError("Depth capture parameters require exactly role.")
    if payload["role"] not in roles:
        raise ValueError("Depth capture role is not registered.")


def validate_capture_depth_result_v1(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    dimensions_by_role: Mapping[str, tuple[int, int]],
) -> None:
    fields = {
        "source_id",
        "role",
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
    role = payload["role"]
    try:
        source_id = source_id_by_role[role]
        expected_width, expected_height = dimensions_by_role[role]
    except (KeyError, TypeError) as exc:
        raise ValueError("Depth capture result uses an unknown role.") from exc
    if payload["source_id"] != source_id:
        raise ValueError("Depth capture source differs from its registered role.")
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
        raise ValueError("Depth capture dimensions do not match the role profile.")
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
    if parameters.get("role") != result.get("role"):
        raise ValueError("Depth result role differs from the request.")


def validate_capture_depth_result_v2(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    dimensions_by_role: Mapping[str, tuple[int, int]],
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
) -> None:
    fields = {
        "source_id",
        "role",
        "captured_at",
        "station_revision",
        "width",
        "height",
        "depth_unit",
        "invalid_depth_value",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "frame_graph_hash",
        "camera_pose_reference_xyz_quat_xyzw",
        "intrinsics",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Depth v2 result fields do not match the schema.")
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("Depth station_revision must be non-negative.")
    expected_identity = {
        "reference_frame_id": reference_frame_id,
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
    }
    if any(
        payload[field_name] != expected
        for field_name, expected in expected_identity.items()
    ):
        raise ValueError("Depth reference-frame identity differs from its deployment.")
    validate_capture_depth_result_v1(
        {
            "source_id": payload["source_id"],
            "role": payload["role"],
            "captured_at": payload["captured_at"],
            "width": payload["width"],
            "height": payload["height"],
            "depth_unit": payload["depth_unit"],
            "invalid_depth_value": payload["invalid_depth_value"],
            "camera_pose_world_xyz_quat_xyzw": payload[
                "camera_pose_reference_xyz_quat_xyzw"
            ],
            "intrinsics": payload["intrinsics"],
        },
        source_id_by_role=source_id_by_role,
        dimensions_by_role=dimensions_by_role,
    )


def validate_capture_depth_result_for_parameters_v2(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    validate_capture_depth_result_for_parameters_v1(parameters, result)


def validate_no_task_evaluation_v1(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or payload:
        raise ValueError("Passive perception capture has no task evaluation payload.")


def localize_objects_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    max_detections: int,
    local_path_enabled: bool,
) -> AgentSkillSchemaEntry:
    """Build strict schemas for automatic or text-prompted bounding boxes."""

    if (
        isinstance(max_detections, bool)
        or not isinstance(max_detections, int)
        or not 1 <= max_detections <= 256
    ):
        raise ValueError("max_detections must be between one and 256.")
    if not isinstance(local_path_enabled, bool):
        raise TypeError("local_path_enabled must be boolean.")
    nullable_prompt = {
        "anyOf": [
            {"type": "string", "minLength": 1, "maxLength": 256},
            {"type": "null"},
        ]
    }
    nullable_path = {
        "anyOf": [
            {"type": "string", "minLength": 1, "maxLength": 4096},
            {"type": "null"},
        ]
    }
    input_source_schema = (
        {"type": "string", "enum": ["artifact", "local_path"]}
        if local_path_enabled
        else {"const": "artifact"}
    )
    local_image_path_schema = nullable_path if local_path_enabled else {"type": "null"}
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_source": input_source_schema,
            "local_image_path": local_image_path_schema,
            "language_prompt": nullable_prompt,
            "max_detections": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_detections,
            },
        },
        "required": [
            "input_source",
            "local_image_path",
            "language_prompt",
            "max_detections",
        ],
    }
    detection_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"type": "integer", "minimum": 0},
            "box_xyxy": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0},
            },
            "score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "label": nullable_prompt,
        },
        "required": ["ordinal", "box_xyxy", "score", "label"],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_source": input_source_schema,
            "input_media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "mode": {"type": "string", "enum": ["automatic", "text"]},
            "language_prompt": nullable_prompt,
            "image_width": {"type": "integer", "minimum": 1},
            "image_height": {"type": "integer", "minimum": 1},
            "coordinate_format": {"const": "xyxy_pixel_0based_half_open"},
            "detections": {
                "type": "array",
                "minItems": 0,
                "maxItems": max_detections,
                "items": detection_schema,
            },
            "detection_count": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_detections,
            },
            "model_profile_id": {"type": "string", "minLength": 1},
            "model_profile_hash": {"type": "string", "minLength": 1},
            "annotated_image_media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "input_source",
            "input_media_content_hash",
            "mode",
            "language_prompt",
            "image_width",
            "image_height",
            "coordinate_format",
            "detections",
            "detection_count",
            "model_profile_id",
            "model_profile_hash",
            "annotated_image_media_content_hash",
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
        skill_id=LOCALIZE_OBJECTS_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def _optional_trimmed_text(
    value: object,
    *,
    field_name: str,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ValueError(f"{field_name} must be null or a bounded trimmed string.")
    return value


def validate_localize_objects_parameters_v1(
    payload: Mapping[str, Any],
    *,
    max_detections: int,
    local_path_enabled: bool,
) -> None:
    fields = {
        "input_source",
        "local_image_path",
        "language_prompt",
        "max_detections",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError(
            "Object localization parameter fields do not match the schema."
        )
    source = payload["input_source"]
    if source not in {"artifact", "local_path"}:
        raise ValueError("Object localization input_source is unsupported.")
    path = _optional_trimmed_text(
        payload["local_image_path"],
        field_name="local_image_path",
        maximum=4096,
    )
    if source == "artifact" and path is not None:
        raise ValueError("Artifact input cannot also supply local_image_path.")
    if source == "local_path":
        if not local_path_enabled:
            raise ValueError("Local-path test input is disabled for this deployment.")
        if path is None or not path.startswith("/"):
            raise ValueError("Local-path test input requires one absolute path.")
    _optional_trimmed_text(
        payload["language_prompt"],
        field_name="language_prompt",
        maximum=256,
    )
    requested = payload["max_detections"]
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 1 <= requested <= max_detections
    ):
        raise ValueError("max_detections is outside the deployment bound.")


def validate_localize_objects_result_v1(
    payload: Mapping[str, Any],
    *,
    max_detections: int,
    model_profile_id: str,
    model_profile_hash: str,
) -> None:
    fields = {
        "input_source",
        "input_media_content_hash",
        "mode",
        "language_prompt",
        "image_width",
        "image_height",
        "coordinate_format",
        "detections",
        "detection_count",
        "model_profile_id",
        "model_profile_hash",
        "annotated_image_media_content_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Object localization result fields do not match the schema.")
    if payload["input_source"] not in {"artifact", "local_path"}:
        raise ValueError("Object localization result input_source changed.")
    for field_name in (
        "input_media_content_hash",
        "annotated_image_media_content_hash",
    ):
        digest = payload[field_name]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{field_name} must be a lowercase SHA-256 digest.")
    prompt = _optional_trimmed_text(
        payload["language_prompt"],
        field_name="language_prompt",
        maximum=256,
    )
    expected_mode = "text" if prompt is not None else "automatic"
    if payload["mode"] != expected_mode:
        raise ValueError("Object localization mode disagrees with language_prompt.")
    dimensions = []
    for field_name in ("image_width", "image_height"):
        value = payload[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("Object localization image dimensions must be positive.")
        dimensions.append(value)
    width, height = dimensions
    if payload["coordinate_format"] != "xyxy_pixel_0based_half_open":
        raise ValueError("Object localization coordinate semantics changed.")
    detections = payload["detections"]
    if (
        not isinstance(detections, (list, tuple))
        or len(detections) > max_detections
        or payload["detection_count"] != len(detections)
    ):
        raise ValueError("Object localization detection count is invalid.")
    detection_fields = {"ordinal", "box_xyxy", "score", "label"}
    for ordinal, detection in enumerate(detections):
        if not isinstance(detection, Mapping) or set(detection) != detection_fields:
            raise ValueError("Object localization detection fields changed.")
        if detection["ordinal"] != ordinal:
            raise ValueError(
                "Object localization detection ordinals must be contiguous."
            )
        box = detection["box_xyxy"]
        if (
            not isinstance(box, (list, tuple))
            or len(box) != 4
            or any(
                isinstance(value, bool) or not isinstance(value, int) for value in box
            )
        ):
            raise ValueError("Object localization boxes must contain four integers.")
        x_min, y_min, x_max, y_max = box
        if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
            raise ValueError("Object localization box is outside the image.")
        score = detection["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not 0.0 <= float(score) <= 1.0
        ):
            raise ValueError("Object localization score must be in [0, 1].")
        label = _optional_trimmed_text(
            detection["label"],
            field_name="detection.label",
            maximum=256,
        )
        if prompt is None and label is not None:
            raise ValueError(
                "Automatic localization detections must not invent labels."
            )
    if payload["model_profile_id"] != model_profile_id:
        raise ValueError("Object localization model_profile_id changed.")
    if payload["model_profile_hash"] != model_profile_hash:
        raise ValueError("Object localization model_profile_hash changed.")


def validate_localize_objects_result_for_parameters_v1(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if parameters.get("input_source") != result.get("input_source"):
        raise ValueError(
            "Object localization result input_source differs from request."
        )
    if parameters.get("language_prompt") != result.get("language_prompt"):
        raise ValueError("Object localization result prompt differs from request.")
    if result.get("detection_count", 0) > parameters.get("max_detections", 0):
        raise ValueError(
            "Object localization result exceeds requested detection bound."
        )


def capture_rgb_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_roles: tuple[str, ...],
    max_total_bytes: int,
) -> AgentSkillSchemaEntry:
    """Build the exact RGB schema triple for one deployment role catalog."""

    roles = _semantic_roles(allowed_roles)
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_roles": {
                "type": "array",
                "minItems": 1,
                "maxItems": min(8, len(roles)),
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(roles)},
            },
            "width": {"type": "integer", "minimum": 1, "maximum": 1920},
            "height": {"type": "integer", "minimum": 1, "maximum": 1080},
            "max_total_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_total_bytes,
            },
        },
        "required": ["selected_roles", "width", "height", "max_total_bytes"],
    }
    frame_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"type": "integer", "minimum": 0},
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "media_type": {"const": RGB_FRAME_MEDIA_TYPE_V1},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "ordinal",
            "source_id",
            "role",
            "media_type",
            "width",
            "height",
            "media_content_hash",
        ],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "snapshot_id": {"type": "string", "minLength": 1},
            "station_revision": {"type": "integer", "minimum": 0},
            "captured_at": {"type": "number"},
            "frames": {
                "type": "array",
                "minItems": 1,
                "maxItems": min(8, len(roles)),
                "items": frame_schema,
            },
            "total_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_total_bytes,
            },
        },
        "required": [
            "snapshot_id",
            "station_revision",
            "captured_at",
            "frames",
            "total_bytes",
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
        skill_id=CAPTURE_RGB_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def capture_rgb_schema_entry_v2(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_roles: tuple[str, ...],
    max_total_bytes: int,
) -> AgentSkillSchemaEntry:
    """Build RGB v2 with an explicit bounded multi-camera capture window."""

    roles = _semantic_roles(allowed_roles)
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "selected_roles": {
                "type": "array",
                "minItems": 1,
                "maxItems": min(8, len(roles)),
                "uniqueItems": True,
                "items": {"type": "string", "enum": list(roles)},
            },
            "width": {"type": "integer", "minimum": 1, "maximum": 1920},
            "height": {"type": "integer", "minimum": 1, "maximum": 1080},
            "max_total_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_total_bytes,
            },
        },
        "required": ["selected_roles", "width", "height", "max_total_bytes"],
    }
    frame_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"type": "integer", "minimum": 0},
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "acquired_at": {"type": "number"},
            "media_type": {"const": RGB_FRAME_MEDIA_TYPE_V1},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "ordinal",
            "source_id",
            "role",
            "acquired_at",
            "media_type",
            "width",
            "height",
            "media_content_hash",
        ],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "snapshot_id": {"type": "string", "minLength": 1},
            "station_revision": {"type": "integer", "minimum": 0},
            "capture_window_started_at": {"type": "number"},
            "capture_window_ended_at": {"type": "number"},
            "frames": {
                "type": "array",
                "minItems": 1,
                "maxItems": min(8, len(roles)),
                "items": frame_schema,
            },
            "total_bytes": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_total_bytes,
            },
        },
        "required": [
            "snapshot_id",
            "station_revision",
            "capture_window_started_at",
            "capture_window_ended_at",
            "frames",
            "total_bytes",
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
        skill_id=CAPTURE_RGB_SKILL_ID_V1,
        skill_version="2",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_capture_rgb_parameters_v1(
    payload: Mapping[str, Any],
    *,
    allowed_roles: tuple[str, ...],
    max_total_bytes: int,
) -> None:
    fields = {"selected_roles", "width", "height", "max_total_bytes"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("RGB capture parameter fields do not match the schema.")
    roles = payload["selected_roles"]
    allowed = frozenset(_semantic_roles(allowed_roles))
    if (
        not isinstance(roles, (list, tuple))
        or not 1 <= len(roles) <= min(8, len(allowed))
        or any(not isinstance(role, str) for role in roles)
    ):
        raise ValueError("RGB selected_roles must be unique registered roles.")
    if len(set(roles)) != len(roles) or any(role not in allowed for role in roles):
        raise ValueError("RGB selected_roles must be unique registered roles.")
    for field_name, maximum in (("width", 1920), ("height", 1080)):
        value = payload[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(f"RGB {field_name} is outside its bounded range.")
    requested_bytes = payload["max_total_bytes"]
    if (
        isinstance(requested_bytes, bool)
        or not isinstance(requested_bytes, int)
        or not 1 <= requested_bytes <= max_total_bytes
    ):
        raise ValueError("RGB max_total_bytes is outside the deployment bound.")


def validate_capture_rgb_result_v1(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    max_total_bytes: int,
) -> None:
    fields = {"snapshot_id", "station_revision", "captured_at", "frames", "total_bytes"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("RGB capture result fields do not match the schema.")
    if not isinstance(payload["snapshot_id"], str) or not payload["snapshot_id"]:
        raise ValueError("RGB snapshot_id must be non-empty.")
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("RGB station_revision must be non-negative.")
    captured_at = payload["captured_at"]
    if (
        isinstance(captured_at, bool)
        or not isinstance(captured_at, (int, float))
        or not math.isfinite(float(captured_at))
    ):
        raise ValueError("RGB captured_at must be finite.")
    frames = payload["frames"]
    if not isinstance(frames, (list, tuple)) or not 1 <= len(frames) <= 8:
        raise ValueError("RGB frames must contain between one and eight entries.")
    if len(frames) > len(source_id_by_role):
        raise ValueError("RGB frames exceed the registered role catalog.")
    frame_fields = {
        "ordinal",
        "source_id",
        "role",
        "media_type",
        "width",
        "height",
        "media_content_hash",
    }
    for ordinal, frame in enumerate(frames):
        if not isinstance(frame, Mapping) or set(frame) != frame_fields:
            raise ValueError("RGB frame fields do not match the schema.")
        role = frame["role"]
        if (
            frame["ordinal"] != ordinal
            or source_id_by_role.get(role) != frame["source_id"]
        ):
            raise ValueError("RGB frame ordinal or registered role binding changed.")
        if frame["media_type"] != RGB_FRAME_MEDIA_TYPE_V1:
            raise ValueError("RGB frame media type changed.")
        for field_name in ("width", "height"):
            value = frame[field_name]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("RGB frame dimensions must be positive integers.")
        digest = frame["media_content_hash"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("RGB frame content hash must be lowercase SHA-256.")
    roles = tuple(frame["role"] for frame in frames)
    sources = tuple(frame["source_id"] for frame in frames)
    if len(set(roles)) != len(roles) or len(set(sources)) != len(sources):
        raise ValueError("RGB frame roles and sources must be unique.")
    total_bytes = payload["total_bytes"]
    if (
        isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or not 1 <= total_bytes <= max_total_bytes
    ):
        raise ValueError("RGB total_bytes is outside the deployment bound.")


def validate_capture_rgb_result_v2(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    max_total_bytes: int,
) -> None:
    """Validate ordered RGB frames and their finite acquisition window."""

    fields = {
        "snapshot_id",
        "station_revision",
        "capture_window_started_at",
        "capture_window_ended_at",
        "frames",
        "total_bytes",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("RGB v2 capture result fields do not match the schema.")
    window = []
    for field_name in (
        "capture_window_started_at",
        "capture_window_ended_at",
    ):
        value = payload[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"RGB {field_name} must be finite.")
        window.append(float(value))
    window_started_at, window_ended_at = window
    if window_started_at > window_ended_at:
        raise ValueError("RGB capture window start must not exceed its end.")
    frames = payload["frames"]
    if not isinstance(frames, (list, tuple)):
        raise TypeError("RGB frames must contain between one and eight entries.")
    projected_frames = []
    for frame in frames:
        if not isinstance(frame, Mapping) or "acquired_at" not in frame:
            raise ValueError("RGB v2 frame fields do not match the schema.")
        acquired_at = frame["acquired_at"]
        if (
            isinstance(acquired_at, bool)
            or not isinstance(acquired_at, (int, float))
            or not math.isfinite(float(acquired_at))
        ):
            raise ValueError("RGB frame acquired_at must be finite.")
        if not window_started_at <= float(acquired_at) <= window_ended_at:
            raise ValueError("RGB frame acquired_at is outside the capture window.")
        projected_frames.append(
            {key: value for key, value in frame.items() if key != "acquired_at"}
        )
    validate_capture_rgb_result_v1(
        {
            "snapshot_id": payload["snapshot_id"],
            "station_revision": payload["station_revision"],
            "captured_at": window_ended_at,
            "frames": projected_frames,
            "total_bytes": payload["total_bytes"],
        },
        source_id_by_role=source_id_by_role,
        max_total_bytes=max_total_bytes,
    )


def validate_capture_rgb_result_for_parameters_v1(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    frames = result.get("frames")
    roles = (
        tuple(frame.get("role") for frame in frames)
        if isinstance(frames, (list, tuple))
        else ()
    )
    if roles != tuple(parameters.get("selected_roles", ())):
        raise ValueError("RGB result roles differ from the request.")
    for frame in frames or ():
        if frame.get("width") != parameters.get("width") or frame.get(
            "height"
        ) != parameters.get("height"):
            raise ValueError("RGB result dimensions differ from the request.")
    if result.get("total_bytes", 0) > parameters.get("max_total_bytes", 0):
        raise ValueError("RGB result exceeds the requested byte bound.")


def validate_capture_rgb_result_for_parameters_v2(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    frames = result.get("frames")
    projected_frames = (
        [
            {key: value for key, value in frame.items() if key != "acquired_at"}
            if isinstance(frame, Mapping)
            else frame
            for frame in frames
        ]
        if isinstance(frames, (list, tuple))
        else frames
    )
    validate_capture_rgb_result_for_parameters_v1(
        parameters,
        {**result, "frames": projected_frames},
    )


__all__ = [
    "DEPTH_FRAME_COMPRESSION_V1",
    "DEPTH_FRAME_ENCODING_V1",
    "DEPTH_FRAME_MEDIA_TYPE_V1",
    "DEPTH_FRAME_SCHEMA_ID_V1",
    "RGB_FRAME_COMPRESSION_V1",
    "RGB_FRAME_ENCODING_V1",
    "RGB_FRAME_MEDIA_TYPE_V1",
    "RGB_FRAME_SCHEMA_ID_V1",
    "capture_depth_schema_entry_v1",
    "capture_depth_schema_entry_v2",
    "capture_rgb_schema_entry_v1",
    "capture_rgb_schema_entry_v2",
    "localize_objects_schema_entry_v1",
    "validate_capture_depth_parameters_v1",
    "validate_capture_depth_result_for_parameters_v1",
    "validate_capture_depth_result_for_parameters_v2",
    "validate_capture_depth_result_v1",
    "validate_capture_depth_result_v2",
    "validate_capture_rgb_parameters_v1",
    "validate_capture_rgb_result_for_parameters_v1",
    "validate_capture_rgb_result_for_parameters_v2",
    "validate_capture_rgb_result_v1",
    "validate_capture_rgb_result_v2",
    "validate_localize_objects_parameters_v1",
    "validate_localize_objects_result_for_parameters_v1",
    "validate_localize_objects_result_v1",
    "validate_no_task_evaluation_v1",
]

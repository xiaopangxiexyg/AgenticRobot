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

from .definition import CAPTURE_DEPTH_SKILL_ID_V1, CAPTURE_RGB_SKILL_ID_V1


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
    if any(payload[field_name] != expected for field_name, expected in expected_identity.items()):
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
    "validate_capture_depth_parameters_v1",
    "validate_capture_depth_result_for_parameters_v1",
    "validate_capture_depth_result_for_parameters_v2",
    "validate_capture_depth_result_v1",
    "validate_capture_depth_result_v2",
    "validate_capture_rgb_parameters_v1",
    "validate_capture_rgb_result_for_parameters_v1",
    "validate_capture_rgb_result_v1",
    "validate_no_task_evaluation_v1",
]

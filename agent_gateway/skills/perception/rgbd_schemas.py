"""Neutral optional same-cut RGBD contracts, preserving RGB v1/v2 identities."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import capture_rgb_skill_definition_v2
from .schemas import (
    capture_rgb_schema_entry_v2,
    validate_capture_rgb_parameters_v1,
    validate_capture_rgb_result_for_parameters_v2,
    validate_capture_rgb_result_v2,
)


def capture_rgb_skill_definition_v3() -> SkillDefinition:
    return replace(
        capture_rgb_skill_definition_v2(),
        skill_version="3",
        summary=(
            "Capture ordered deployment-registered RGB roles. With include_depth "
            "true, capture exactly one aligned RGB/depth observation at its native "
            "grid, preserving original JPEG/NPY bytes and producer-bound metadata; "
            "requested width/height must equal that grid. With include_depth false, "
            "retain RGB-only capture behavior. This historical observation does "
            "not certify current object identity, freshness, or physical safety."
        ),
        parameter_semantics_id="registered_roles.optional_same_cut_rgbd.v3",
        execution_result_semantics_id="registered_roles.rgbd_artifact_provenance.v3",
    )


def capture_rgb_schema_entry_v3(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_roles: tuple[str, ...],
    max_total_bytes: int,
    depth_dimensions_by_role: Mapping[str, tuple[int, int]],
) -> AgentSkillSchemaEntry:
    if (
        not isinstance(depth_dimensions_by_role, Mapping)
        or not depth_dimensions_by_role
        or any(
            role not in allowed_roles
            or not isinstance(size, (tuple, list))
            or len(size) != 2
            or any(type(value) is not int or value <= 0 for value in size)
            or size[0] > 1920
            or size[1] > 1080
            for role, size in depth_dimensions_by_role.items()
        )
    ):
        raise ValueError(
            "RGB-D schema requires explicit registered native depth grids."
        )
    # Thaw only the existing canonical v2 documents; never mutate their entries.
    base = capture_rgb_schema_entry_v2(
        skill_definition_hash=skill_definition_hash,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        allowed_roles=allowed_roles,
        max_total_bytes=max_total_bytes,
    ).to_wire()
    parameters = base["parameter_schema"]
    grids = "; ".join(
        f"{role}: width={size[0]}, height={size[1]}"
        for role, size in sorted(depth_dimensions_by_role.items())
    )
    parameters["properties"]["include_depth"] = {
        "type": "boolean",
        "description": (
            "True requires one of these exact native RGB-D grids: "
            + grids
            + ". False keeps RGB-only multi-role and requested-resolution capture."
        ),
    }
    parameters["required"].append("include_depth")
    result = base["execution_result_schema"]
    pair_fields = {
        "role": {"type": "string", "enum": sorted(allowed_roles)},
        "source_id": {"type": "string", "minLength": 1},
        "rgb_frame_ordinal": {"type": "integer", "minimum": 0, "maximum": 0},
        "depth_artifact_ordinal": {"type": "integer", "minimum": 0, "maximum": 0},
        "observation_artifact_ordinal": {"type": "integer", "minimum": 0, "maximum": 0},
        "observation_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    }
    result["properties"].update(
        {
            "include_depth": {"type": "boolean"},
            "paired_observations": {
                "type": "array",
                "minItems": 0,
                "maxItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": pair_fields,
                    "required": list(pair_fields),
                },
            },
        }
    )
    result["required"].extend(("include_depth", "paired_observations"))
    return AgentSkillSchemaEntry(
        skill_id=base["skill_id"],
        skill_version="3",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameters,
        execution_result_schema=result,
        task_evaluation_schema=base["task_evaluation_schema"],
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=parameter_profile_id,
                parameter_profile_hash=parameter_profile_hash,
                parameter_schema_hash=agent_skill_schema_document_hash(
                    parameters, schema_kind="parameter"
                ),
            ),
        ),
    )


def validate_capture_rgb_parameters_v3(
    payload: Mapping[str, Any],
    *,
    allowed_roles: tuple[str, ...],
    max_total_bytes: int,
    depth_dimensions_by_role: Mapping[str, tuple[int, int]],
) -> None:
    fields = {"selected_roles", "width", "height", "max_total_bytes", "include_depth"}
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("RGB v3 capture parameter fields do not match the schema.")
    if not isinstance(payload["include_depth"], bool):
        raise TypeError("include_depth must be a boolean.")
    validate_capture_rgb_parameters_v1(
        {key: value for key, value in payload.items() if key != "include_depth"},
        allowed_roles=allowed_roles,
        max_total_bytes=max_total_bytes,
    )
    if payload["include_depth"]:
        roles = payload["selected_roles"]
        if len(roles) != 1 or roles[0] not in depth_dimensions_by_role:
            raise ValueError(
                "RGBD capture requires exactly one registered aligned-depth role."
            )
        if (payload["width"], payload["height"]) != depth_dimensions_by_role[roles[0]]:
            raise ValueError(
                "RGBD capture requires the exact native grid; resizing is forbidden."
            )


def validate_capture_rgb_result_v3(
    payload: Mapping[str, Any],
    *,
    source_id_by_role: Mapping[str, str],
    max_total_bytes: int,
    depth_dimensions_by_role: Mapping[str, tuple[int, int]],
) -> None:
    extra = {"include_depth", "paired_observations"}
    base_fields = {
        "snapshot_id",
        "station_revision",
        "capture_window_started_at",
        "capture_window_ended_at",
        "frames",
        "total_bytes",
    }
    if not isinstance(payload, Mapping) or set(payload) != base_fields | extra:
        raise ValueError("RGB v3 capture result fields do not match the schema.")
    if not isinstance(payload["include_depth"], bool):
        raise TypeError("include_depth must be a boolean.")
    validate_capture_rgb_result_v2(
        {key: value for key, value in payload.items() if key not in extra},
        source_id_by_role=source_id_by_role,
        max_total_bytes=max_total_bytes,
    )
    # The legacy validator compares ordinals by value; v3 also rejects bool aliases.
    for frame in payload["frames"]:
        if type(frame["ordinal"]) is not int:
            raise ValueError("RGB frame ordinal must be an integer, not a boolean.")
    pairs = payload["paired_observations"]
    if not isinstance(pairs, (list, tuple)):
        raise TypeError("paired_observations must be an array.")
    if not payload["include_depth"]:
        if pairs:
            raise ValueError("RGB-only capture must not advertise paired observations.")
        return
    if len(payload["frames"]) != 1 or len(pairs) != 1:
        raise ValueError(
            "RGBD capture must contain one frame and one paired observation."
        )
    pair = pairs[0]
    fields = {
        "role",
        "source_id",
        "rgb_frame_ordinal",
        "depth_artifact_ordinal",
        "observation_artifact_ordinal",
        "observation_sha256",
    }
    if not isinstance(pair, Mapping) or set(pair) != fields:
        raise ValueError("Paired observation fields do not match the schema.")
    frame = payload["frames"][0]
    role = pair["role"]
    if (
        not isinstance(role, str)
        or role not in depth_dimensions_by_role
        or (pair["role"], pair["source_id"]) != (frame["role"], frame["source_id"])
        or (frame["width"], frame["height"]) != depth_dimensions_by_role[role]
    ):
        raise ValueError(
            "Paired observation changed its registered RGB/depth role or grid."
        )
    for name in (
        "rgb_frame_ordinal",
        "depth_artifact_ordinal",
        "observation_artifact_ordinal",
    ):
        if type(pair[name]) is not int or pair[name] != 0:
            raise ValueError("Paired observation ordinals must be integer zero.")
    digest = pair["observation_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("observation_sha256 must be lowercase SHA-256.")
    if payload["capture_window_started_at"] != payload["capture_window_ended_at"]:
        raise ValueError("RGBD capture must describe one same-cut observation.")


def validate_capture_rgb_result_for_parameters_v3(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    if result["include_depth"] is not parameters["include_depth"]:
        raise ValueError("RGBD result include_depth differs from the request.")
    validate_capture_rgb_result_for_parameters_v2(parameters, result)


__all__ = [
    "capture_rgb_schema_entry_v3",
    "capture_rgb_skill_definition_v3",
    "validate_capture_rgb_parameters_v3",
    "validate_capture_rgb_result_for_parameters_v3",
    "validate_capture_rgb_result_v3",
]

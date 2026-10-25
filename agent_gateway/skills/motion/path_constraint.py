"""Target-neutral safe TCP v2 semantics, schemas, and offline validators.

The lower owner derives the positive selected tool axis from the fresh
activation calibrated TCP in the public declared reference frame. The cone
applies to sampled planned and live references, with rotation about that axis
free. This contract neither implements execution nor proves continuous motion.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.motion.definition import (
    SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    safe_tcp_pose_target_skill_definition_v1,
)
from agent_gateway.skills.motion.schemas import (
    tcp_pose_target_schema_entry_v1,
    validate_agent_motion_parameters_v1,
)

SAFE_TCP_POSE_TARGET_SKILL_ID_V2 = SAFE_TCP_POSE_TARGET_SKILL_ID_V1
_KIND = "preserve_start_tool_axis"
_SCOPE = "sampled_planned_and_live_references"
_CONSTRAINT_FIELDS = frozenset(("kind", "tool_axis", "max_deviation_rad"))
_EVIDENCE_FIELDS = _CONSTRAINT_FIELDS | {
    "activation_axis_in_reference_frame",
    "constraint_identity",
    "planned_max_deviation_rad",
    "validated_reference_count",
    "scope",
}
_PARAMETER_FIELDS = frozenset((
    "pose_targets_by_component",
    "reference_frame_id",
    "reference_frame_spec_hash",
    "controlled_frame_id",
    "frame_graph_hash",
    "tool_config_hash",
    "calibration_hash",
    "path_constraint",
))
_OPTIONAL_PARAMETER_FIELDS = frozenset(("requested_duration_ms", "speed_scale", "pose_goal_policy"))
_CONSTRAINT_UNSET = object()
_ANGLE_TOLERANCE_RAD = 1e-9
_UNIT_TOLERANCE = 1e-9


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field_name} must be finite.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite.")
    return result


def validate_path_constraint(value: object) -> None:
    """Validate null or the exact axis-cone policy without changing it."""
    if value is None:
        return
    if not isinstance(value, Mapping) or set(value) != _CONSTRAINT_FIELDS:
        raise ValueError("path_constraint must be null or the exact policy object.")
    if value["kind"] != _KIND:
        raise ValueError("Unsupported path_constraint kind.")
    if value["tool_axis"] not in ("x", "y", "z"):
        raise ValueError("path_constraint tool_axis must be x, y, or z.")
    bound = _finite(value["max_deviation_rad"], field_name="max_deviation_rad")
    if not 0.0 < bound <= math.pi:
        raise ValueError("max_deviation_rad must satisfy 0 < value <= pi.")


def validate_path_constraint_evidence(
    evidence: object, *, constraint: object = _CONSTRAINT_UNSET,
) -> None:
    """Validate evidence, optionally requiring it to match the request policy.

    An omitted constraint checks evidence alone. Explicit None requires null
    evidence; an explicit policy requires non-null evidence with identical
    policy fields. Unit-vector norm tolerance is 1e-9; the planned deviation
    may exceed the requested bound by at most 1e-9 radians. The lower owner
    produces constraint_identity; this validator checks its 64-hex shape only.
    """
    if constraint is not _CONSTRAINT_UNSET:
        validate_path_constraint(constraint)
    if evidence is None:
        if constraint is not _CONSTRAINT_UNSET and constraint is not None:
            raise ValueError("A requested path_constraint requires non-null evidence.")
        return
    if constraint is None:
        raise ValueError("A null path_constraint requires null evidence.")
    if not isinstance(evidence, Mapping) or set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("path_constraint_evidence must be the exact evidence object.")
    validate_path_constraint({key: evidence[key] for key in _CONSTRAINT_FIELDS})
    if constraint is not _CONSTRAINT_UNSET and any(
        evidence[key] != constraint[key] for key in _CONSTRAINT_FIELDS
    ):
        raise ValueError("Evidence must match the requested path_constraint.")
    axis = evidence["activation_axis_in_reference_frame"]
    if not isinstance(axis, (list, tuple)) or len(axis) != 3:
        raise ValueError("activation_axis_in_reference_frame must have three values.")
    coordinates = tuple(_finite(item, field_name="activation_axis") for item in axis)
    if any(abs(item) > 1.0 for item in coordinates) or (
        abs(math.hypot(*coordinates) - 1.0) > _UNIT_TOLERANCE
    ):
        raise ValueError("activation_axis_in_reference_frame must be a unit vector.")
    identity = evidence["constraint_identity"]
    if not isinstance(identity, str) or re.fullmatch(r"[0-9a-fA-F]{64}", identity) is None:
        raise ValueError("constraint_identity must contain exactly 64 hex characters.")
    maximum = _finite(
        evidence["planned_max_deviation_rad"], field_name="planned_max_deviation_rad",
    )
    if not 0.0 <= maximum <= evidence["max_deviation_rad"] + _ANGLE_TOLERANCE_RAD:
        raise ValueError("planned_max_deviation_rad exceeds the requested cone.")
    count = evidence["validated_reference_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("validated_reference_count must be an integer >= 1.")
    if evidence["scope"] != _SCOPE:
        raise ValueError("Evidence scope must be sampled_planned_and_live_references.")


def safe_tcp_pose_target_skill_definition_v2(
    *, controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    """Return the same safe TCP Skill identity at semantic version 2."""
    definition = safe_tcp_pose_target_skill_definition_v1(
        controlled_component_ids=controlled_component_ids,
    )
    return replace(
        definition,
        skill_version="2",
        parameter_semantics_id="tcp_pose_target.absolute.xyz_quat_xyzw.path_constraint.v2",
        execution_result_semantics_id="component_action_execution.path_constraint_evidence.v2",
        summary=(
            definition.summary + " A required null path_constraint adds no axis policy. "
            "For preserve_start_tool_axis, the lower owner derives the positive selected "
            "tool x/y/z axis from the fresh activation calibrated TCP in the public "
            "declared reference frame and preserves it within max_deviation_rad along "
            "sampled planned and live references. Rotation about that axis remains free. "
            "No implicit relaxation or straight-line substitution is permitted. "
            "Sampled checks provide no mathematical continuous-path guarantee."
        ),
    )


def _constraint_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "kind": {"const": _KIND},
            "tool_axis": {"type": "string", "enum": ["x", "y", "z"]},
            "max_deviation_rad": {
                "type": "number", "exclusiveMinimum": 0.0, "maximum": math.pi,
            },
        },
        "required": ["kind", "tool_axis", "max_deviation_rad"],
    }


def _evidence_schema() -> dict[str, Any]:
    schema = _constraint_schema()
    schema["properties"].update({
        "activation_axis_in_reference_frame": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": {"type": "number", "minimum": -1.0, "maximum": 1.0},
            "description": "Unit vector in the public declared reference frame.",
        },
        "constraint_identity": {
            "type": "string", "minLength": 64, "maxLength": 64,
            "pattern": "^[0-9a-fA-F]{64}$",
        },
        "planned_max_deviation_rad": {
            "type": "number", "minimum": 0.0,
            "maximum": math.pi + _ANGLE_TOLERANCE_RAD,
            "description": "At most max_deviation_rad + 1e-9 radians.",
        },
        "validated_reference_count": {"type": "integer", "minimum": 1},
        "scope": {"const": _SCOPE},
    })
    schema["required"] += [
        "activation_axis_in_reference_frame", "constraint_identity",
        "planned_max_deviation_rad", "validated_reference_count", "scope",
    ]
    return schema


def safe_tcp_pose_target_schema_entry_v2(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    max_requested_duration_ms: int | None,
    workspace_bounds_by_component: Mapping[
        str, tuple[float, float, float, float, float, float],
    ] | None = None,
    min_speed_scale: float | None = None,
    max_speed_scale: float | None = None,
    parameter_description: str | None = None,
    max_target_components: int | None = None,
) -> AgentSkillSchemaEntry:
    """Clone v1, adding required policy/evidence and rebinding profile hashes."""
    entry = tcp_pose_target_schema_entry_v1(
        skill_id=SAFE_TCP_POSE_TARGET_SKILL_ID_V2,
        skill_definition_hash=skill_definition_hash,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        allowed_component_ids=allowed_component_ids,
        allowed_reference_frame_ids=allowed_reference_frame_ids,
        max_requested_duration_ms=max_requested_duration_ms,
        workspace_bounds_by_component=workspace_bounds_by_component,
        min_speed_scale=min_speed_scale,
        max_speed_scale=max_speed_scale,
        parameter_description=parameter_description,
        max_target_components=max_target_components,
    )
    wire = entry.to_wire()
    parameters = wire["parameter_schema"]
    parameters["properties"]["path_constraint"] = {
        "anyOf": [{"type": "null"}, _constraint_schema()],
    }
    parameters["required"].append("path_constraint")
    result = wire["execution_result_schema"]
    result["properties"]["path_constraint_evidence"] = {
        "anyOf": [{"type": "null"}, _evidence_schema()],
    }
    result["required"].append("path_constraint_evidence")
    schema_hash = agent_skill_schema_document_hash(parameters, schema_kind="parameter")
    return replace(
        entry,
        skill_version="2",
        parameter_schema=parameters,
        execution_result_schema=result,
        parameter_profiles=tuple(
            replace(profile, parameter_schema_hash=schema_hash)
            for profile in entry.parameter_profiles
        ),
    )


def validate_safe_tcp_pose_parameters_v2(
    payload: Mapping[str, Any], allowed_component_ids: tuple[str, ...],
) -> None:
    """Validate exact v2 parameters and delegate absolute-pose checks to v1.

    The caller's schema still enforces its exact frame, workspace, duration,
    and speed profile bounds. This function does not derive deployment facts.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("Safe TCP parameters must be a mapping.")
    fields = set(payload)
    if not _PARAMETER_FIELDS <= fields or fields - (
        _PARAMETER_FIELDS | _OPTIONAL_PARAMETER_FIELDS
    ):
        raise ValueError("Safe TCP v2 parameters must match the exact contract.")
    validate_path_constraint(payload["path_constraint"])
    for key in _PARAMETER_FIELDS - {"pose_targets_by_component", "path_constraint"}:
        value = payload[key]
        if not isinstance(value, str) or not value or value.strip() != value:
            raise ValueError(f"{key} must be a non-empty string without surrounding whitespace.")
    if "requested_duration_ms" in payload:
        duration = payload["requested_duration_ms"]
        if isinstance(duration, bool) or not isinstance(duration, int) or duration < 1:
            raise ValueError("requested_duration_ms must be an integer >= 1.")
    validate_agent_motion_parameters_v1(
        schema_kind="tcp_pose_target",
        parameters={key: value for key, value in payload.items() if key != "path_constraint"},
        allowed_component_ids=allowed_component_ids,
        # Profile-specific limits are narrower; the semantic fraction is (0, 1].
        speed_scale_bounds=(math.nextafter(0.0, 1.0), 1.0),
    )


__all__ = [
    "SAFE_TCP_POSE_TARGET_SKILL_ID_V2",
    "safe_tcp_pose_target_schema_entry_v2",
    "safe_tcp_pose_target_skill_definition_v2",
    "validate_path_constraint",
    "validate_path_constraint_evidence",
    "validate_safe_tcp_pose_parameters_v2",
]

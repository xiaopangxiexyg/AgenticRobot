"""Portable schema and validators for passive placement geometry planning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import PLAN_PLACEMENT_SKILL_ID_V1

_PARAMETER_FIELDS = {
    "held_segment_operation_id",
    "destination_segment_operation_id",
    "clearance_m",
}
_RESULT_FIELDS = {
    "planning_status",
    "reason_code",
    "held_segment_operation_id",
    "destination_segment_operation_id",
    "station_revision",
    "component_id",
    "reference_frame_id",
    "reference_frame_spec_hash",
    "frame_graph_hash",
    "target_eef_xy",
    "minimum_lateral_eef_z",
    "fits_with_clearance",
    "clearance_m",
    "held_gripper_close_fraction",
    "held_gripper_state_checked",
    "destination_geometry_provisional",
    "geometry_eligible_for_motion_planning",
    "execution_requires_explicit_motion_skill",
    "motion_or_hold_performed",
}


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")
    return number


def _vector(
    value: object,
    *,
    field_name: str,
    length: int,
    nullable: bool = False,
) -> tuple[float, ...] | None:
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field_name} must contain {length} numbers.")
    return tuple(_finite(item, field_name=f"{field_name}[]") for item in value)


def plan_placement_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    component_id: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    maximum_clearance_m: float,
) -> AgentSkillSchemaEntry:
    """Build one strict inline-only plan schema."""

    component = _text(component_id, field_name="component_id")
    reference = _text(reference_frame_id, field_name="reference_frame_id")
    reference_hash = _text(
        reference_frame_spec_hash,
        field_name="reference_frame_spec_hash",
    )
    graph_hash = _text(frame_graph_hash, field_name="frame_graph_hash")
    clearance_limit = _finite(
        maximum_clearance_m,
        field_name="maximum_clearance_m",
    )
    if clearance_limit <= 0.0:
        raise ValueError("maximum_clearance_m must be positive.")
    operation_id = {
        "type": "string",
        "minLength": 1,
        "description": "Operation ID returned by a completed text-prompt segment call.",
    }
    parameters = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "held_segment_operation_id": operation_id,
            "destination_segment_operation_id": operation_id,
            "clearance_m": {
                "type": "number",
                "minimum": 0.0,
                "maximum": clearance_limit,
                "description": "Required lateral boundary and top clearance in meters.",
            },
        },
        "required": sorted(_PARAMETER_FIELDS),
    }
    vec2 = {
        "type": ["array", "null"],
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "number"},
    }
    nullable_number = {"type": ["number", "null"]}
    nullable_boolean = {"type": ["boolean", "null"]}
    results = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "planning_status": {"type": "string", "enum": ["planned", "rejected"]},
            "reason_code": {"type": "string", "minLength": 1},
            "held_segment_operation_id": operation_id,
            "destination_segment_operation_id": operation_id,
            "station_revision": {"type": "integer", "minimum": 0},
            "component_id": {"const": component},
            "reference_frame_id": {"const": reference},
            "reference_frame_spec_hash": {"const": reference_hash},
            "frame_graph_hash": {"const": graph_hash},
            "target_eef_xy": vec2,
            "minimum_lateral_eef_z": nullable_number,
            "fits_with_clearance": nullable_boolean,
            "clearance_m": {
                "type": "number",
                "minimum": 0.0,
                "maximum": clearance_limit,
            },
            "held_gripper_close_fraction": {
                "type": ["number", "null"],
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "held_gripper_state_checked": {"type": "boolean"},
            "destination_geometry_provisional": {"type": "boolean"},
            "geometry_eligible_for_motion_planning": {"type": "boolean"},
            "execution_requires_explicit_motion_skill": {"const": True},
            "motion_or_hold_performed": {"const": False},
        },
        "required": sorted(_RESULT_FIELDS),
    }
    task_evaluation = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }
    profile = SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameters,
            schema_kind="parameter",
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=PLAN_PLACEMENT_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameters,
        execution_result_schema=results,
        task_evaluation_schema=task_evaluation,
        parameter_profiles=(profile,),
    )


def validate_plan_placement_parameters_v1(
    payload: Mapping[str, Any],
    *,
    maximum_clearance_m: float,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _PARAMETER_FIELDS:
        raise ValueError("Placement parameters do not match the schema.")
    held = _text(
        payload["held_segment_operation_id"],
        field_name="held_segment_operation_id",
    )
    destination = _text(
        payload["destination_segment_operation_id"],
        field_name="destination_segment_operation_id",
    )
    if held == destination:
        raise ValueError("Held and destination segment operations must differ.")
    clearance = _finite(payload["clearance_m"], field_name="clearance_m")
    if not 0.0 <= clearance <= maximum_clearance_m:
        raise ValueError("clearance_m is outside the deployment bound.")


def validate_plan_placement_result_v1(
    payload: Mapping[str, Any],
    *,
    component_id: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    maximum_clearance_m: float,
    target_kind: str,
    minimum_held_close_fraction: float,
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != _RESULT_FIELDS:
        raise ValueError("Placement result fields do not match the schema.")
    for name, expected in (
        ("component_id", component_id),
        ("reference_frame_id", reference_frame_id),
        ("reference_frame_spec_hash", reference_frame_spec_hash),
        ("frame_graph_hash", frame_graph_hash),
    ):
        if payload[name] != expected:
            raise ValueError(f"Placement result changed {name}.")
    for name in (
        "held_segment_operation_id",
        "destination_segment_operation_id",
        "reason_code",
    ):
        _text(payload[name], field_name=name)
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("station_revision must be non-negative.")
    clearance = _finite(payload["clearance_m"], field_name="clearance_m")
    if not 0.0 <= clearance <= maximum_clearance_m:
        raise ValueError("Placement result clearance is outside its bound.")
    status = payload["planning_status"]
    if status not in {"planned", "rejected"}:
        raise ValueError("Placement planning_status is invalid.")
    target_xy = _vector(
        payload["target_eef_xy"],
        field_name="target_eef_xy",
        length=2,
        nullable=True,
    )
    minimum_z = payload["minimum_lateral_eef_z"]
    if minimum_z is not None:
        _finite(minimum_z, field_name="minimum_lateral_eef_z")
    complete_geometry = all(
        value is not None
        for value in (target_xy, minimum_z, payload["fits_with_clearance"])
    )
    if (status == "planned") != complete_geometry:
        raise ValueError("Placement status and geometry completeness disagree.")
    close = payload["held_gripper_close_fraction"]
    if close is not None and not 0.0 <= _finite(
        close,
        field_name="held_gripper_close_fraction",
    ) <= 1.0:
        raise ValueError("Held gripper close fraction is outside [0, 1].")
    checked = payload["held_gripper_state_checked"]
    if not isinstance(checked, bool) or checked != (close is not None):
        raise ValueError("Gripper checked flag differs from its measurement.")
    for name in (
        "destination_geometry_provisional",
        "geometry_eligible_for_motion_planning",
    ):
        if not isinstance(payload[name], bool):
            raise TypeError(f"{name} must be boolean.")
    eligible = payload["geometry_eligible_for_motion_planning"]
    if eligible and (
        status != "planned"
        or payload["fits_with_clearance"] is not True
        or payload["destination_geometry_provisional"]
    ):
        raise ValueError("Placement eligibility contradicts its geometry result.")
    minimum_close = _finite(
        minimum_held_close_fraction,
        field_name="minimum_held_close_fraction",
    )
    if target_kind not in {"real", "sim"} or not 0.0 <= minimum_close <= 1.0:
        raise ValueError("Placement target or gripper threshold is invalid.")
    if target_kind == "real" and eligible and (
        not checked or close is None or float(close) <= minimum_close
    ):
        raise ValueError(
            "Real placement eligibility requires a measured closed gripper."
        )
    if payload["execution_requires_explicit_motion_skill"] is not True:
        raise ValueError("Placement cannot authorize motion.")
    if payload["motion_or_hold_performed"] is not False:
        raise ValueError("Placement result cannot claim a control effect.")


def validate_plan_placement_result_for_parameters_v1(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    for name in _PARAMETER_FIELDS:
        if result[name] != parameters[name]:
            raise ValueError(f"Placement result {name} differs from the request.")


__all__ = [
    "plan_placement_schema_entry_v1",
    "validate_plan_placement_parameters_v1",
    "validate_plan_placement_result_for_parameters_v1",
    "validate_plan_placement_result_v1",
]

"""Frozen-observation first-acceptable grasp planning without motion or HOLD."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from agent_gateway.protocol._contract_validation import (
    finite_number,
    freeze_json_object,
    nonnegative_integer,
    required_text,
    sha256_digest,
    unique_sorted_text,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

PLAN_GRASP_FAST_SKILL_ID_V1 = "perception.plan_grasp_fast"
_PLAN_FIELDS = (
    "T_reference_from_tool_tcp",
    "gripper_width_m",
    "grasp_approach_plan_id",
    "grasp_approach_plan_hash",
    "trajectory_hash",
    "trajectory_sample_count",
    "trajectory_duration_sec",
    "plan_created_at",
    "plan_expires_at",
)


def grasp_path_attempts_json(attempts: Sequence[Mapping[str, Any]]) -> str:
    """Keep the complete query history within the existing inline contract.

    Six decimal places are diagnostic precision only, never a planner clock.
    Native stage/endpoint details belong in the provider log, not duplicated in
    this bounded cross-target summary. Never truncate query reasons or rows.
    """
    rows = []
    for attempt in attempts:
        elapsed = attempt["elapsed_sec"]
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or (
            not math.isfinite(elapsed) or elapsed < 0
        ):
            raise ValueError("Path attempt elapsed time must be nonnegative and finite.")
        rows.append({
            "index": attempt["index"],
            "reason_code": attempt["reason_code"],
            "elapsed_sec": round(elapsed, 6),
        })
    encoded = json.dumps(rows, separators=(",", ":"), allow_nan=False)
    if len(encoded) > 8192:
        raise ValueError("Complete path attempt history exceeds its inline contract.")
    return encoded


def fast_grasp_plan_skill_definition_v1(
    *,
    controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    """Describe one bounded full-path search, never physical task success."""
    components = unique_sorted_text(
        controlled_component_ids,
        field_name="controlled_component_ids",
        allow_empty=False,
    )
    return SkillDefinition(
        skill_id=PLAN_GRASP_FAST_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Plan from one frozen observation and native target pixel using "
            "first_acceptable_v1: return the first immutable grasp approach plan "
            "that passes the existing complete path checks within one bounded "
            "search, without candidate selection input, rendering, motion or HOLD. "
            "Planning completion is not physical grasp success or an execution grant."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=components,
        parameter_semantics_id="frozen_observation.native_pixel.first_acceptable.v1",
        execution_result_semantics_id="immutable_first_acceptable_grasp_plan.v1",
        task_outcome_semantics_id="passive_bounded_grasp_path_assessment.v1",
        entry_predicate_ids=(
            "retained_frozen_grasp_observation_available.v1",
            "fresh_measured_qpos_available.v1",
            "complete_grasp_path_checks_available.v1",
        ),
        task_outcome_predicate_ids=("grasp_path_assessed_without_motion.v1",),
    )


def fast_grasp_plan_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    component_id: str,
    side: str,
    roles: tuple[str, ...],
    min_speed_scale: float,
    max_speed_scale: float,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
) -> AgentSkillSchemaEntry:
    """Bind strict semantic schemas to explicit deployment-owned identities."""
    identities = {
        "component_id": component_id,
        "reference_frame_id": reference_frame_id,
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "tool_config_hash": tool_config_hash,
        "calibration_hash": calibration_hash,
    }
    for name, value in identities.items():
        required_text(value, field_name=name)
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right.")
    selected_roles = unique_sorted_text(roles, field_name="roles", allow_empty=False)
    minimum = finite_number(min_speed_scale, field_name="min_speed_scale")
    maximum = finite_number(max_speed_scale, field_name="max_speed_scale")
    if not 0 < minimum <= maximum <= 1:
        raise ValueError("speed scales must satisfy 0 < min <= max <= 1.")
    text = {"type": "string", "minLength": 1, "maxLength": 256}
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    pixel = {
        "type": "array",
        "minItems": 2,
        "maxItems": 2,
        "items": {"type": "integer", "minimum": 0},
    }
    parameter_properties = {
        "component_id": {"const": component_id},
        "observation_operation_id": text,
        "role": {"type": "string", "enum": list(selected_roles)},
        "target_pixel_uv": pixel,
        "view_policy": {"type": "string", "enum": ["single", "auto"]},
        "speed_scale": {"type": "number", "minimum": minimum, "maximum": maximum},
        "planning_budget_sec": {"type": "number", "minimum": 1, "maximum": 30},
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": parameter_properties,
        "required": list(parameter_properties),
    }
    plan_properties = {
        "T_reference_from_tool_tcp": {
            "type": "array",
            "minItems": 16,
            "maxItems": 16,
            "items": {"type": "number"},
        },
        "gripper_width_m": {"type": "number", "minimum": 0},
        "grasp_approach_plan_id": text,
        "grasp_approach_plan_hash": digest,
        "trajectory_hash": digest,
        "trajectory_sample_count": {"type": "integer", "minimum": 2},
        "trajectory_duration_sec": {"type": "number", "exclusiveMinimum": 0},
        "plan_created_at": {"type": "number"},
        "plan_expires_at": {"type": "number"},
    }
    result_properties = {
        "planning_status": {"type": "string", "enum": ["path_found", "rejected"]},
        "reason_code": text,
        **{name: {"const": value} for name, value in identities.items()},
        "side": {"const": side},
        "observation_operation_id": text,
        "observation_content_hash": digest,
        "role": parameter_properties["role"],
        "target_pixel_uv": pixel,
        "source_station_revision": {"type": "integer", "minimum": 0},
        **{
            name: {"anyOf": [value, {"type": "null"}]}
            for name, value in plan_properties.items()
        },
        "eligible_for_execution": {"type": "boolean"},
        "execution_requires_explicit_motion_skill": {"const": True},
        "motion_or_hold_performed": {"const": False},
        "planning_policy": {"const": "first_acceptable_v1"},
        "diagnostics": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pool_count": {"type": "integer", "minimum": 0, "maximum": 65536},
                "path_query_count": {"type": "integer", "minimum": 0, "maximum": 65536},
                "elapsed_sec": {"type": "number", "minimum": 0, "maximum": 3600},
                "attempts_json": {"type": "string", "maxLength": 8192},
            },
        },
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": result_properties,
        "required": list(result_properties),
        "allOf": [
            {
                "if": {"properties": {"planning_status": {"const": "path_found"}}},
                "then": {
                    "properties": {
                        **plan_properties,
                        "eligible_for_execution": {"const": True},
                    }
                },
                "else": {
                    "properties": {
                        **{name: {"type": "null"} for name in plan_properties},
                        "eligible_for_execution": {"const": False},
                    }
                },
            }
        ],
    }
    return AgentSkillSchemaEntry(
        skill_id=PLAN_GRASP_FAST_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema={
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        },
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=parameter_profile_id,
                parameter_profile_hash=parameter_profile_hash,
                parameter_schema_hash=agent_skill_schema_document_hash(
                    parameter_schema,
                    schema_kind="parameter",
                ),
            ),
        ),
    )


def _closed_payload(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> Mapping:
    value = freeze_json_object(payload, field_name="fast grasp payload")
    if set(value) - set(schema["properties"]) or set(schema.get("required", ())) - set(
        value
    ):
        raise ValueError("Fast grasp payload fields do not match the schema.")
    for name, spec in schema["properties"].items():
        if "const" in spec and (
            type(value[name]) is not type(spec["const"]) or value[name] != spec["const"]
        ):
            raise ValueError(f"{name} differs from its deployed constant.")
    return value


def _pixel(value: object) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("target_pixel_uv must contain two native integers.")
    for item in value:
        nonnegative_integer(item, field_name="target_pixel_uv")
    return tuple(value)


def _source(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    required_text(
        payload["observation_operation_id"],
        field_name="observation_operation_id",
        maximum_length=256,
    )
    if payload["role"] not in schema["properties"]["role"]["enum"]:
        raise ValueError("role is not supported by this deployment.")
    _pixel(payload["target_pixel_uv"])


def _bounded_number(value: object, spec: Mapping[str, Any], *, field: str) -> None:
    number = finite_number(value, field_name=field)
    if number < spec.get("minimum", number) or number > spec.get("maximum", number):
        raise ValueError(f"{field} is outside the deployed bounds.")


def validate_fast_grasp_plan_parameters_v1(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    """Validate frozen-source parameters without importing a runtime or model."""
    value = _closed_payload(payload, schema)
    _source(value, schema)
    if value["view_policy"] not in schema["properties"]["view_policy"]["enum"]:
        raise ValueError("view_policy must be single or auto.")
    for field in ("speed_scale", "planning_budget_sec"):
        _bounded_number(value[field], schema["properties"][field], field=field)


def validate_fast_grasp_plan_result_v1(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    """Validate the published structure and success/rejection plan boundary."""
    value = _closed_payload(payload, schema)
    _source(value, schema)
    required_text(value["reason_code"], field_name="reason_code", maximum_length=256)
    sha256_digest(
        value["observation_content_hash"], field_name="observation_content_hash"
    )
    nonnegative_integer(
        value["source_station_revision"], field_name="source_station_revision"
    )
    diagnostics_schema = schema["properties"]["diagnostics"]
    diagnostics = _closed_payload(value["diagnostics"], diagnostics_schema)
    for field, item in diagnostics.items():
        spec = diagnostics_schema["properties"][field]
        if spec["type"] == "integer":
            nonnegative_integer(item, field_name=field, maximum=spec["maximum"])
        elif spec["type"] == "number":
            _bounded_number(item, spec, field=field)
        elif not isinstance(item, str) or len(item) > spec["maxLength"]:
            raise ValueError("attempts_json exceeds its bounded string contract.")
    status = value["planning_status"]
    if (
        status not in {"path_found", "rejected"}
        or type(value["eligible_for_execution"]) is not bool
    ):
        raise ValueError("Invalid fast grasp planning status or eligibility.")
    success = status == "path_found"
    if value["eligible_for_execution"] != success or any(
        (value[field] is not None) != success for field in _PLAN_FIELDS
    ):
        raise ValueError(
            "Planning status, eligibility and nullable plan fields disagree."
        )
    if not success:
        return
    pose = value["T_reference_from_tool_tcp"]
    if not isinstance(pose, (tuple, list)) or len(pose) != 16:
        raise ValueError("T_reference_from_tool_tcp must contain 16 numbers.")
    for item in pose:
        finite_number(item, field_name="T_reference_from_tool_tcp")
    _bounded_number(value["gripper_width_m"], {"minimum": 0}, field="gripper_width_m")
    for field in ("grasp_approach_plan_hash", "trajectory_hash"):
        sha256_digest(value[field], field_name=field)
    # Opaque plan IDs belong to the target's shared store; the public contract
    # must not derive a real/sim prefix or reconstruct an ID from its hash.
    required_text(
        value["grasp_approach_plan_id"],
        field_name="grasp_approach_plan_id",
        maximum_length=256,
    )
    if (
        nonnegative_integer(
            value["trajectory_sample_count"], field_name="trajectory_sample_count"
        )
        < 2
    ):
        raise ValueError("trajectory_sample_count must be at least two.")
    if (
        finite_number(
            value["trajectory_duration_sec"], field_name="trajectory_duration_sec"
        )
        <= 0
    ):
        raise ValueError("trajectory_duration_sec must be positive.")
    created = finite_number(value["plan_created_at"], field_name="plan_created_at")
    if finite_number(value["plan_expires_at"], field_name="plan_expires_at") <= created:
        raise ValueError("plan_expires_at must follow plan_created_at.")


def validate_fast_grasp_plan_result_for_parameters_v1(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Preserve the exact requested component and frozen view/pixel lineage."""
    for field in ("component_id", "observation_operation_id", "role"):
        requested = required_text(
            parameters[field], field_name=field, maximum_length=256
        )
        if result[field] != requested:
            raise ValueError(f"Fast grasp result {field} differs from the request.")
    if _pixel(result["target_pixel_uv"]) != _pixel(parameters["target_pixel_uv"]):
        raise ValueError("Fast grasp result target_pixel_uv differs from the request.")


__all__ = [
    "PLAN_GRASP_FAST_SKILL_ID_V1",
    "fast_grasp_plan_schema_entry_v1",
    "fast_grasp_plan_skill_definition_v1",
    "validate_fast_grasp_plan_parameters_v1",
    "validate_fast_grasp_plan_result_for_parameters_v1",
    "validate_fast_grasp_plan_result_v1",
]

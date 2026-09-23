"""Strict portable schema and validators for measured gripper state."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import READ_GRIPPER_STATE_SKILL_ID_V1


def _ids(values: tuple[str, ...], *, name: str) -> tuple[str, ...]:
    normalized = tuple(values)
    if (
        not normalized
        or any(
            not isinstance(item, str) or not item or item != item.strip()
            for item in normalized
        )
        or len(set(normalized)) != len(normalized)
    ):
        raise ValueError(f"{name} must contain unique non-empty strings.")
    return tuple(sorted(normalized))


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _component_map(
    value: object,
    *,
    components: tuple[str, ...],
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != set(components):
        raise ValueError(f"{name} must exactly cover requested components.")
    return value


def _map_schema(
    component_ids: tuple[str, ...], value_schema: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {component: dict(value_schema) for component in component_ids},
    }


def read_gripper_state_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    maximum_measurement_age_ms: int,
) -> AgentSkillSchemaEntry:
    components = _ids(allowed_component_ids, name="allowed_component_ids")
    if (
        isinstance(maximum_measurement_age_ms, bool)
        or not isinstance(maximum_measurement_age_ms, int)
        or maximum_measurement_age_ms <= 0
    ):
        raise ValueError("maximum_measurement_age_ms must be positive.")
    component_array = {
        "type": "array",
        "minItems": 1,
        "maxItems": len(components),
        "uniqueItems": True,
        "items": {"type": "string", "enum": list(components)},
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_ids": component_array,
            "max_measurement_age_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": maximum_measurement_age_ms,
            },
        },
        "required": ["component_ids", "max_measurement_age_ms"],
    }
    result_properties = {
        "component_ids": component_array,
        "station_revision": {"type": "integer", "minimum": 0},
        "measurement_id": {"type": "string", "minLength": 1},
        "measurement_hash": {"type": "string", "minLength": 1},
        "measured_at_by_component": _map_schema(components, {"type": "number"}),
        "available_at": {"type": "number"},
        "clock_domain_id": {"type": "string", "minLength": 1},
        "close_fraction_by_component": _map_schema(
            components, {"type": "number", "minimum": 0.0, "maximum": 1.0}
        ),
        "state_source_by_component": _map_schema(
            components, {"type": "string", "minLength": 1}
        ),
        "observation_binding_id": {"type": "string", "minLength": 1},
        "observation_binding_hash": {"type": "string", "minLength": 1},
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": result_properties,
        "required": list(result_properties),
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
        skill_id=READ_GRIPPER_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_read_gripper_state_parameters_v1(
    payload: Mapping[str, Any],
    *,
    allowed_component_ids: tuple[str, ...],
    maximum_measurement_age_ms: int,
) -> None:
    components = _ids(allowed_component_ids, name="allowed_component_ids")
    if not isinstance(payload, Mapping) or set(payload) != {
        "component_ids",
        "max_measurement_age_ms",
    }:
        raise ValueError(
            "Gripper-state parameters require exactly component_ids and max_measurement_age_ms."
        )
    requested = payload["component_ids"]
    if (
        not isinstance(requested, (list, tuple))
        or not requested
        or len(set(requested)) != len(requested)
        or not set(requested) <= set(components)
    ):
        raise ValueError("component_ids must be a unique non-empty registered subset.")
    age = payload["max_measurement_age_ms"]
    if (
        isinstance(age, bool)
        or not isinstance(age, int)
        or not 1 <= age <= maximum_measurement_age_ms
    ):
        raise ValueError("max_measurement_age_ms is outside the deployment bound.")


def validate_read_gripper_state_result_v1(
    payload: Mapping[str, Any],
    *,
    request_parameters: Mapping[str, Any],
    observation_binding_id: str,
    observation_binding_hash: str,
    clock_domain_id: str,
) -> None:
    expected_fields = {
        "component_ids",
        "station_revision",
        "measurement_id",
        "measurement_hash",
        "measured_at_by_component",
        "available_at",
        "clock_domain_id",
        "close_fraction_by_component",
        "state_source_by_component",
        "observation_binding_id",
        "observation_binding_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_fields:
        raise ValueError("Gripper-state result fields do not match the schema.")
    requested = tuple(sorted(request_parameters["component_ids"]))
    result_components = tuple(sorted(payload["component_ids"]))
    if result_components != requested:
        raise ValueError("Result component_ids do not match the request.")
    if payload["clock_domain_id"] != clock_domain_id:
        raise ValueError("Gripper-state clock domain mismatch.")
    if (
        payload["observation_binding_id"] != observation_binding_id
        or payload["observation_binding_hash"] != observation_binding_hash
    ):
        raise ValueError("Gripper-state observation binding mismatch.")
    if (
        isinstance(payload["station_revision"], bool)
        or not isinstance(payload["station_revision"], int)
        or payload["station_revision"] < 0
    ):
        raise ValueError("station_revision must be non-negative.")
    for name in ("measurement_id", "measurement_hash"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"{name} must be non-empty.")
    available_at = _finite(payload["available_at"], name="available_at")
    measured = _component_map(
        payload["measured_at_by_component"],
        components=requested,
        name="measured_at_by_component",
    )
    close = _component_map(
        payload["close_fraction_by_component"],
        components=requested,
        name="close_fraction_by_component",
    )
    sources = _component_map(
        payload["state_source_by_component"],
        components=requested,
        name="state_source_by_component",
    )
    maximum_age = request_parameters["max_measurement_age_ms"] / 1_000.0
    for component in requested:
        measured_at = _finite(
            measured[component], name=f"measured_at_by_component.{component}"
        )
        if (
            measured_at > available_at
            or available_at - measured_at > maximum_age + 1e-9
        ):
            raise ValueError("Gripper-state measurement is unavailable or stale.")
        fraction = _finite(
            close[component], name=f"close_fraction_by_component.{component}"
        )
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Gripper close fraction must lie in [0, 1].")
        if not isinstance(sources[component], str) or not sources[component].strip():
            raise ValueError("Gripper state source must be non-empty.")


def validate_no_task_evaluation_v1(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or payload:
        raise ValueError("Passive gripper-state reads do not produce task evaluation.")


__all__ = [
    "read_gripper_state_schema_entry_v1",
    "validate_no_task_evaluation_v1",
    "validate_read_gripper_state_parameters_v1",
    "validate_read_gripper_state_result_v1",
]

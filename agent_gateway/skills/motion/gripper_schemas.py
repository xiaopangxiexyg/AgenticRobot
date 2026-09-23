"""Strict portable schemas for fixed open/close gripper Skills."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.motion.definition import (
    CLOSE_GRIPPER_SKILL_ID_V1,
    OPEN_GRIPPER_SKILL_ID_V1,
)

GRIPPER_ENDPOINT_SKILL_IDS_V1 = (
    CLOSE_GRIPPER_SKILL_ID_V1,
    OPEN_GRIPPER_SKILL_ID_V1,
)


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


def _skill_id(value: object) -> str:
    if value not in GRIPPER_ENDPOINT_SKILL_IDS_V1:
        raise ValueError("skill_id is not a fixed gripper endpoint Skill.")
    return str(value)


def gripper_endpoint_schema_entry_v1(
    *,
    skill_id: str,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    selected_skill_id = _skill_id(skill_id)
    components = _ids(allowed_component_ids, name="allowed_component_ids")
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_id": {"type": "string", "enum": list(components)},
        },
        "required": ["component_id"],
    }
    result_properties = {
        "component_id": {"type": "string", "enum": list(components)},
        "commanded_close_fraction": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "initial_close_fraction": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "final_close_fraction": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "final_measured_at": {"type": "number"},
        "completed_at": {"type": "number"},
        "station_revision": {"type": "integer", "minimum": 0},
        "command_receipt_ref": {"type": "string", "minLength": 1},
        "final_measurement_hash": {"type": "string", "minLength": 1},
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": result_properties,
        "required": list(result_properties),
    }
    task_properties = {
        "aggregate_evaluation_receipt_ref": {
            "type": "string",
            "minLength": 1,
        },
        "task_outcome": {
            "type": "string",
            "enum": ["failed", "inconclusive", "succeeded"],
        },
        "predicate": {
            "type": "string",
            "enum": [
                "gripper_close_command_settled.v1",
                "gripper_open_endpoint_reached.v1",
            ],
        },
    }
    task_evaluation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": task_properties,
        "required": list(task_properties),
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
        skill_id=selected_skill_id,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(profile,),
    )


def validate_gripper_endpoint_parameters_v1(
    payload: Mapping[str, Any],
    *,
    allowed_component_ids: tuple[str, ...],
) -> None:
    components = _ids(allowed_component_ids, name="allowed_component_ids")
    if not isinstance(payload, Mapping) or set(payload) != {"component_id"}:
        raise ValueError("Gripper action parameters require exactly component_id.")
    if payload["component_id"] not in components:
        raise ValueError("component_id is not allowed by this deployment.")


def validate_gripper_endpoint_result_v1(
    payload: Mapping[str, Any],
    *,
    skill_id: str,
    request_parameters: Mapping[str, Any] | None = None,
) -> None:
    selected_skill_id = _skill_id(skill_id)
    expected = {
        "component_id",
        "commanded_close_fraction",
        "initial_close_fraction",
        "final_close_fraction",
        "final_measured_at",
        "completed_at",
        "station_revision",
        "command_receipt_ref",
        "final_measurement_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Gripper action result fields do not match the schema.")
    if request_parameters is not None and payload[
        "component_id"
    ] != request_parameters.get("component_id"):
        raise ValueError("Result component_id does not match the request.")
    expected_target = 0.0 if selected_skill_id == OPEN_GRIPPER_SKILL_ID_V1 else 1.0
    target = _finite(
        payload["commanded_close_fraction"],
        name="commanded_close_fraction",
    )
    if target != expected_target:
        raise ValueError("Result target does not match the fixed Skill endpoint.")
    for name in ("initial_close_fraction", "final_close_fraction"):
        value = _finite(payload[name], name=name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must lie in [0, 1].")
    measured_at = _finite(payload["final_measured_at"], name="final_measured_at")
    completed_at = _finite(payload["completed_at"], name="completed_at")
    if measured_at > completed_at:
        raise ValueError("Final measurement cannot occur after completion.")
    revision = payload["station_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("station_revision must be a non-negative integer.")
    for name in ("component_id", "command_receipt_ref", "final_measurement_hash"):
        if not isinstance(payload[name], str) or not payload[name].strip():
            raise ValueError(f"{name} must be a non-empty string.")


def validate_gripper_endpoint_task_evaluation_v1(
    payload: Mapping[str, Any],
    *,
    skill_id: str,
) -> None:
    selected_skill_id = _skill_id(skill_id)
    if not isinstance(payload, Mapping) or set(payload) != {
        "aggregate_evaluation_receipt_ref",
        "task_outcome",
        "predicate",
    }:
        raise ValueError("Gripper action task evaluation fields are invalid.")
    if payload["task_outcome"] not in {"failed", "inconclusive", "succeeded"}:
        raise ValueError("Unsupported gripper action task outcome.")
    expected_predicate = (
        "gripper_open_endpoint_reached.v1"
        if selected_skill_id == OPEN_GRIPPER_SKILL_ID_V1
        else "gripper_close_command_settled.v1"
    )
    if payload["predicate"] != expected_predicate:
        raise ValueError("Task predicate does not match the gripper Skill.")
    receipt = payload["aggregate_evaluation_receipt_ref"]
    if not isinstance(receipt, str) or not receipt.strip():
        raise ValueError("aggregate_evaluation_receipt_ref must be non-empty.")


__all__ = [
    "GRIPPER_ENDPOINT_SKILL_IDS_V1",
    "gripper_endpoint_schema_entry_v1",
    "validate_gripper_endpoint_parameters_v1",
    "validate_gripper_endpoint_result_v1",
    "validate_gripper_endpoint_task_evaluation_v1",
]

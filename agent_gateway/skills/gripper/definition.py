"""Target-neutral definition and payload schemas for atomic gripper control."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

SET_GRIPPER_STATE_SKILL_ID_V1 = "manipulation.set_gripper_state"
SET_GRIPPER_STATE_PARAMETER_PROFILE_ID_V1 = "gripper_state.open_or_closed.v1"
SET_GRIPPER_STATE_RESULT_SCHEMA_V1 = "dexteleop.agent_skill.set_gripper_state_result.v1"
SET_GRIPPER_STATE_TARGETS_V1 = ("closed", "open")
SET_GRIPPER_STATE_COMPLETIONS_V1 = (
    "settled_before_full_close",
    "target_reached",
)
GRIPPER_COMPONENT_IDS = ("left_gripper", "right_gripper")
SET_GRIPPER_STATE_RESULT_SCHEMA_V2 = "dexteleop.agent_skill.set_gripper_state_result.v2"


def set_gripper_state_skill_definition_v3() -> SkillDefinition:
    return replace(
        set_gripper_state_skill_definition_v2(),
        skill_version="3",
        summary=("Open or close one selected deployment-bound gripper. An optional "
                 "named support may be bound only when opening the right hand "
                 "after a verified grasp; it selects the release observation target "
                 "without moving the arm. Command completion is not stable release; "
                 "read the object state after its full observation window."),
        parameter_semantics_id="gripper_state.selected_optional_named_release.v3",
    )


def set_gripper_state_schema_entry_v3(
    *, skill_definition_hash: str, support_object_ids: tuple[str, ...]
) -> AgentSkillSchemaEntry:
    if isinstance(support_object_ids, str):
        raise ValueError("v3 support identities must be explicit objects.")
    names = tuple(support_object_ids)
    if (not names or any(not isinstance(name, str) or not name or name != name.strip()
                         for name in names) or len(set(names)) != len(names)):
        raise ValueError("v3 requires distinct deployment-bound support identities.")
    base = set_gripper_state_schema_entry_v2(
        skill_definition_hash=skill_definition_hash)
    parameters = {
        **base.parameter_schema,
        "properties": {**base.parameter_schema["properties"],
                       "release_support_object_id": {"type": "string", "enum": list(names)}},
        "allOf": [{"if": {"required": ["release_support_object_id"]},
                   "then": {"properties": {
                       "component_id": {"const": "right_gripper"},
                       "target_state": {"const": "open"}}}}],
    }
    schema_hash = agent_skill_schema_document_hash(parameters, schema_kind="parameter")
    profile_id = "gripper_state.selected_optional_named_release.v3"
    profile_hash = canonical_content_hash({
        "parameter_profile_id": profile_id,
        "parameter_schema_hash": schema_hash,
        "agent_selectable_parameters": ("component_id", "target_state", "release_support_object_id"),
    }, domain="dexteleop.agent_skill.gripper_state_parameter_profile.v3")
    return replace(base, skill_version="3", parameter_schema=parameters,
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id=profile_id, parameter_profile_hash=profile_hash,
            parameter_schema_hash=schema_hash),))


def validate_set_gripper_state_parameters_v3(
    parameters: Mapping[str, Any], *, support_object_ids: tuple[str, ...]
) -> None:
    if not isinstance(parameters, Mapping) or set(parameters) not in (
            {"component_id", "target_state"},
            {"component_id", "target_state", "release_support_object_id"}):
        raise ValueError("v3 requires a selected gripper/state and optional named release support.")
    validate_set_gripper_state_parameters_v2({
        key: parameters[key] for key in ("component_id", "target_state")})
    if "release_support_object_id" in parameters and (
            parameters["component_id"] != "right_gripper"
            or parameters["target_state"] != "open"
            or parameters["release_support_object_id"] not in support_object_ids):
        raise ValueError("Named release requires the admitted right-hand support and open state.")


def validate_set_gripper_state_result_for_parameters_v3(
    parameters: Mapping[str, Any], result: Mapping[str, Any],
    *, support_object_ids: tuple[str, ...]
) -> None:
    validate_set_gripper_state_parameters_v3(
        parameters, support_object_ids=support_object_ids)
    validate_set_gripper_state_result_for_parameters_v2(
        {key: parameters[key] for key in ("component_id", "target_state")}, result)


def set_gripper_state_skill_definition_v2() -> SkillDefinition:
    return replace(
        set_gripper_state_skill_definition_v1(),
        skill_version="2",
        controlled_component_ids=GRIPPER_COMPONENT_IDS,
        parameter_semantics_id="gripper_state.selected_component_open_or_closed.v2",
    )


def set_gripper_state_schema_entry_v2(
    *, skill_definition_hash: str
) -> AgentSkillSchemaEntry:
    base = set_gripper_state_schema_entry_v1(
        skill_definition_hash=skill_definition_hash
    )
    parameters = {
        **base.parameter_schema,
        "properties": {
            **base.parameter_schema["properties"],
            "component_id": {"type": "string", "enum": list(GRIPPER_COMPONENT_IDS)},
        },
        "required": ["component_id", "target_state"],
    }
    result = {
        **base.execution_result_schema,
        "properties": {
            **base.execution_result_schema["properties"],
            "schema_version": {"const": SET_GRIPPER_STATE_RESULT_SCHEMA_V2},
            "component_id": {"type": "string", "enum": list(GRIPPER_COMPONENT_IDS)},
        },
    }
    schema_hash = agent_skill_schema_document_hash(parameters, schema_kind="parameter")
    profile_id = "gripper_state.selected_open_or_closed.v2"
    profile_hash = canonical_content_hash(
        {
            "parameter_profile_id": profile_id,
            "parameter_schema_hash": schema_hash,
            "agent_selectable_parameters": ("component_id", "target_state"),
        },
        domain="dexteleop.agent_skill.gripper_state_parameter_profile.v2",
    )
    return AgentSkillSchemaEntry(
        skill_id=SET_GRIPPER_STATE_SKILL_ID_V1,
        skill_version="2",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameters,
        execution_result_schema=result,
        task_evaluation_schema=base.task_evaluation_schema,
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=profile_id,
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=schema_hash,
            ),
        ),
    )


def validate_set_gripper_state_parameters_v2(parameters: Mapping[str, Any]) -> None:
    if not isinstance(parameters, Mapping) or set(parameters) != {
        "component_id",
        "target_state",
    }:
        raise ValueError(
            "v2 parameters must contain exactly component_id and target_state."
        )
    if parameters["component_id"] not in GRIPPER_COMPONENT_IDS:
        raise ValueError("component_id must identify the left or right gripper.")
    validate_set_gripper_state_parameters_v1(
        {"target_state": parameters["target_state"]}
    )


def validate_set_gripper_state_result_v2(result: Mapping[str, Any]) -> None:
    _validate_set_gripper_state_result(
        result, schema_version=SET_GRIPPER_STATE_RESULT_SCHEMA_V2
    )
    if result["component_id"] not in GRIPPER_COMPONENT_IDS:
        raise ValueError("result component_id is not a gripper.")


def validate_set_gripper_state_result_for_parameters_v2(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    validate_set_gripper_state_parameters_v2(parameters)
    validate_set_gripper_state_result_v2(result)
    if (
        result["component_id"] != parameters["component_id"]
        or result["commanded_state"] != parameters["target_state"]
    ):
        raise ValueError("result does not match the selected gripper and target state.")


def set_gripper_state_skill_definition_v1() -> SkillDefinition:
    return SkillDefinition(
        skill_id=SET_GRIPPER_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_kind="motion",
        summary=(
            "Open or close the deployment-bound gripper without commanding "
            "arm motion."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=("right_gripper",),
        parameter_semantics_id="gripper_state.discrete_open_or_closed.v1",
        execution_result_semantics_id=("measured_gripper_settle_and_terminal_hold.v1"),
        task_outcome_semantics_id="not_applicable.no_task_evaluation.v1",
        entry_predicate_ids=("deployment_bound_gripper_ready.v1",),
        task_outcome_predicate_ids=(),
    )


def set_gripper_state_schema_entry_v1(
    *,
    skill_definition_hash: str,
) -> AgentSkillSchemaEntry:
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_state": {
                "type": "string",
                "enum": list(SET_GRIPPER_STATE_TARGETS_V1),
            }
        },
        "required": ["target_state"],
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": SET_GRIPPER_STATE_RESULT_SCHEMA_V1},
            "component_id": {"type": "string", "minLength": 1},
            "commanded_state": {
                "type": "string",
                "enum": list(SET_GRIPPER_STATE_TARGETS_V1),
            },
            "measured_close": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "completion_kind": {
                "type": "string",
                "enum": list(SET_GRIPPER_STATE_COMPLETIONS_V1),
            },
            "terminal_hold_confirmed": {"const": True},
        },
        "required": [
            "schema_version",
            "component_id",
            "commanded_state",
            "measured_close",
            "completion_kind",
            "terminal_hold_confirmed",
        ],
    }
    task_evaluation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
        "required": [],
    }
    parameter_schema_hash = agent_skill_schema_document_hash(
        parameter_schema,
        schema_kind="parameter",
    )
    profile_hash = canonical_content_hash(
        {
            "parameter_profile_id": SET_GRIPPER_STATE_PARAMETER_PROFILE_ID_V1,
            "parameter_schema_hash": parameter_schema_hash,
            "agent_selectable_parameters": ("target_state",),
        },
        domain="dexteleop.agent_skill.gripper_state_parameter_profile.v1",
    )
    return AgentSkillSchemaEntry(
        skill_id=SET_GRIPPER_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=task_evaluation_schema,
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=SET_GRIPPER_STATE_PARAMETER_PROFILE_ID_V1,
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=parameter_schema_hash,
            ),
        ),
    )


def validate_set_gripper_state_parameters_v1(
    parameters: Mapping[str, Any],
) -> None:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be an object.")
    if set(parameters) != {"target_state"}:
        raise ValueError("parameters must contain exactly target_state.")
    if parameters["target_state"] not in SET_GRIPPER_STATE_TARGETS_V1:
        raise ValueError("target_state must be 'open' or 'closed'.")


def validate_set_gripper_state_result_v1(result: Mapping[str, Any]) -> None:
    _validate_set_gripper_state_result(
        result, schema_version=SET_GRIPPER_STATE_RESULT_SCHEMA_V1
    )


def _validate_set_gripper_state_result(
    result: Mapping[str, Any], *, schema_version: str
) -> None:
    expected = {
        "schema_version",
        "component_id",
        "commanded_state",
        "measured_close",
        "completion_kind",
        "terminal_hold_confirmed",
    }
    if not isinstance(result, Mapping) or set(result) != expected:
        raise ValueError("execution result fields do not match the v1 schema.")
    if result["schema_version"] != schema_version:
        raise ValueError("unsupported gripper execution result schema.")
    component_id = result["component_id"]
    if not isinstance(component_id, str) or not component_id.strip():
        raise ValueError("component_id must be non-empty.")
    if result["commanded_state"] not in SET_GRIPPER_STATE_TARGETS_V1:
        raise ValueError("commanded_state is invalid.")
    if result["completion_kind"] not in SET_GRIPPER_STATE_COMPLETIONS_V1:
        raise ValueError("completion_kind is invalid.")
    measured = result["measured_close"]
    if (
        isinstance(measured, bool)
        or not isinstance(measured, (int, float))
        or not math.isfinite(float(measured))
        or not 0.0 <= float(measured) <= 1.0
    ):
        raise ValueError("measured_close must be finite and in [0, 1].")
    if result["terminal_hold_confirmed"] is not True:
        raise ValueError("successful result requires terminal HOLD confirmation.")
    if (
        result["commanded_state"] == "open"
        and result["completion_kind"] != "target_reached"
    ):
        raise ValueError("opening cannot complete before the open target is reached.")


def validate_set_gripper_state_result_for_parameters_v1(
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    validate_set_gripper_state_parameters_v1(parameters)
    validate_set_gripper_state_result_v1(result)
    if result["commanded_state"] != parameters["target_state"]:
        raise ValueError("result commanded_state does not match target_state.")


def validate_no_gripper_task_evaluation_v1(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping) or payload:
        raise ValueError("this Skill has no task-evaluation payload.")


__all__ = [
    "GRIPPER_COMPONENT_IDS",
    "SET_GRIPPER_STATE_COMPLETIONS_V1",
    "SET_GRIPPER_STATE_PARAMETER_PROFILE_ID_V1",
    "SET_GRIPPER_STATE_RESULT_SCHEMA_V1",
    "SET_GRIPPER_STATE_RESULT_SCHEMA_V2",
    "SET_GRIPPER_STATE_SKILL_ID_V1",
    "SET_GRIPPER_STATE_TARGETS_V1",
    "set_gripper_state_schema_entry_v1",
    "set_gripper_state_schema_entry_v2",
    "set_gripper_state_schema_entry_v3",
    "set_gripper_state_skill_definition_v1",
    "set_gripper_state_skill_definition_v2",
    "set_gripper_state_skill_definition_v3",
    "validate_no_gripper_task_evaluation_v1",
    "validate_set_gripper_state_parameters_v1",
    "validate_set_gripper_state_parameters_v2",
    "validate_set_gripper_state_parameters_v3",
    "validate_set_gripper_state_result_for_parameters_v1",
    "validate_set_gripper_state_result_for_parameters_v2",
    "validate_set_gripper_state_result_for_parameters_v3",
    "validate_set_gripper_state_result_v1",
    "validate_set_gripper_state_result_v2",
]

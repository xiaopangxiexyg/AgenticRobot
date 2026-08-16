"""Target-independent CubeLift Skill semantics and payload schemas.

The definition is stable above concrete simulator or real-robot Routes.  It
contains no MuJoCo deployment binding, evaluator implementation or runtime
owner.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

CUBE_LIFT_SKILL_ID_V1 = "task.cube_lift"
CUBE_LIFT_PARAMETER_PROFILE_ID_V1 = "cube_lift.server_bound"
CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1 = (
    "dexteleop.agent_skill.cube_lift_execution_result.v1"
)
CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1 = (
    "dexteleop.agent_skill.cube_lift_task_evaluation.v1"
)
CUBE_LIFT_PARAMETER_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.agent_skill.cube_lift_parameter_profile.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _exact_fields(
    value: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field_name: str,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    actual = frozenset(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            f"{field_name} fields mismatch; missing={missing}, unknown={unknown}."
        )


def cube_lift_skill_definition_v1() -> SkillDefinition:
    return SkillDefinition(
        skill_id=CUBE_LIFT_SKILL_ID_V1,
        skill_version="1",
        skill_kind="policy_task",
        summary=(
            "Lift the server-bound cube with the right RM65 arm and gripper "
            "using the exact lower-bound policy deployment."
        ),
        intended_target_kind="sim",
        cross_target_deployment_policy="review_required",
        controlled_component_ids=("right_arm", "right_gripper"),
        parameter_semantics_id="server_bound_task.no_agent_motion_parameters.v1",
        execution_result_semantics_id=(
            "authoritative_execution_receipt_chain_and_terminal_hold.v1"
        ),
        task_outcome_semantics_id=(
            "cube_lift.measured_terminal_stable_hold.v1"
        ),
        entry_predicate_ids=(
            "policy_only_sim_hil_ready.v1",
            "station_gate_ready.v1",
            "task_reset_evidence_available.v1",
        ),
        task_outcome_predicate_ids=(
            "cube_above_resting_baseline.v1",
            "bilateral_finger_contact_stable.v1",
            "cube_not_in_table_contact.v1",
            "finite_state_stable_under_hold.v1",
        ),
    )


def _cube_lift_execution_result_schema_v1() -> dict[str, Any]:
    references = (
        "execution_evidence_chain_ref",
        "source_proposal_chain_hash",
        "authority_decision_chain_hash",
        "safety_decision_chain_hash",
        "target_request_chain_hash",
        "target_submission_receipt_chain_hash",
        "target_completion_receipt_chain_hash",
        "resolved_component_target_chain_hash",
        "component_admission_receipt_chain_hash",
        "dispatch_receipt_chain_hash",
        "measurement_lineage_chain_hash",
        "first_cycle_receipt_ref",
        "last_cycle_receipt_ref",
        "terminal_hold_receipt_ref",
        "terminal_state_hash",
        "trace_terminal_receipt_ref",
        "trace_commit_receipt_ref",
    )
    properties: dict[str, Any] = {
        "schema_version": {"const": CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1},
        "runtime_session_id": {"type": "string", "minLength": 1},
        "control_domain_id": {"type": "string", "minLength": 1},
        **{
            name: {"type": "string", "minLength": 1}
            for name in references
        },
        "cycle_count": {"type": "integer", "minimum": 1},
        "dispatched_cycle_count": {"type": "integer", "minimum": 1},
        "measurement_count": {"type": "integer", "minimum": 1},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def _cube_lift_task_evaluation_schema_v1() -> dict[str, Any]:
    properties: dict[str, Any] = {
        "schema_version": {"const": CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1},
        "verdict": {
            "type": "string",
            "enum": ["failed", "inconclusive", "succeeded"],
        },
        "reason_codes": {
            "type": "array",
            "minItems": 1,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1},
        },
        "evaluated_cut_id": {"type": "string", "minLength": 1},
        "environment_evidence_hash": {"type": "string", "minLength": 1},
        "evaluator_configuration_hash": {"type": "string", "minLength": 1},
        "terminal_state_hash": {"type": "string", "minLength": 1},
        "baseline_resting_center_z_m": {"type": "number"},
        "required_lift_delta_m": {"type": "number", "exclusiveMinimum": 0},
        "minimum_verified_cube_center_z_m": {"type": "number"},
        "final_cube_center_z_m": {"type": "number"},
        "stable_hold_required_physics_ticks": {
            "type": "integer",
            "minimum": 1,
        },
        "stable_hold_observed_physics_ticks": {
            "type": "integer",
            "minimum": 0,
        },
        "finite_state_all_ticks": {"type": "boolean"},
        "bilateral_contact_all_ticks": {"type": "boolean"},
        "table_contact_any_tick": {"type": "boolean"},
        "terminal_hold_confirmed": {"type": "boolean"},
        "execution_lineage_complete": {"type": "boolean"},
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


def cube_lift_task_evaluation_fields_v1() -> frozenset[str]:
    """Return the exact semantic evaluation field set for Route validation."""

    return frozenset(_cube_lift_task_evaluation_schema_v1()["properties"])


def cube_lift_schema_entry_v1(
    *,
    skill_definition_hash: str,
) -> AgentSkillSchemaEntry:
    parameter_schema = {
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
            "parameter_profile_id": CUBE_LIFT_PARAMETER_PROFILE_ID_V1,
            "parameter_schema_hash": parameter_schema_hash,
            "agent_selectable_parameters": (),
            "binding_semantics": "server_bound_exact_deployment",
        },
        domain=CUBE_LIFT_PARAMETER_PROFILE_HASH_DOMAIN_V1,
    )
    return AgentSkillSchemaEntry(
        skill_id=CUBE_LIFT_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=_cube_lift_execution_result_schema_v1(),
        task_evaluation_schema=_cube_lift_task_evaluation_schema_v1(),
        parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id=CUBE_LIFT_PARAMETER_PROFILE_ID_V1,
                parameter_profile_hash=profile_hash,
                parameter_schema_hash=parameter_schema_hash,
            ),
        ),
    )


def validate_cube_lift_parameters_v1(parameters: Mapping[str, Any]) -> None:
    """The server-bound CubeLift profile exposes no Agent-selected fields."""

    _exact_fields(
        parameters,
        expected=frozenset(),
        field_name="parameters",
    )


_CUBE_LIFT_EXECUTION_RESULT_FIELDS = frozenset(
    _cube_lift_execution_result_schema_v1()["properties"]
)


def validate_cube_lift_execution_result_v1(
    result: Mapping[str, Any],
) -> None:
    _exact_fields(
        result,
        expected=_CUBE_LIFT_EXECUTION_RESULT_FIELDS,
        field_name="execution_result",
    )
    if result["schema_version"] != CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1:
        raise ValueError("Unsupported CubeLift execution result schema.")
    string_fields = _CUBE_LIFT_EXECUTION_RESULT_FIELDS - {
        "cycle_count",
        "dispatched_cycle_count",
        "measurement_count",
    }
    for field_name in string_fields:
        _required(result[field_name], field_name=field_name)
    cycle_count = _positive_integer(result["cycle_count"], field_name="cycle_count")
    dispatched_count = _positive_integer(
        result["dispatched_cycle_count"],
        field_name="dispatched_cycle_count",
    )
    measurement_count = _positive_integer(
        result["measurement_count"],
        field_name="measurement_count",
    )
    if dispatched_count > cycle_count:
        raise ValueError("dispatched_cycle_count cannot exceed cycle_count.")
    if measurement_count < dispatched_count:
        raise ValueError(
            "CubeLift completion requires measurement coverage for dispatches."
        )


__all__ = [
    "CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1",
    "CUBE_LIFT_PARAMETER_PROFILE_ID_V1",
    "CUBE_LIFT_SKILL_ID_V1",
    "CUBE_LIFT_TASK_EVALUATION_SCHEMA_V1",
    "cube_lift_schema_entry_v1",
    "cube_lift_skill_definition_v1",
    "validate_cube_lift_execution_result_v1",
    "validate_cube_lift_parameters_v1",
]

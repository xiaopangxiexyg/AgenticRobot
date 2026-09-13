"""Read-only contract for assessing a retained grasp candidate's full path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

PLAN_GRASP_APPROACH_SKILL_ID_V1 = "perception.plan_grasp_approach"


def plan_grasp_approach_skill_definition_v1(
    *, controlled_component_ids: tuple[str, ...]
) -> SkillDefinition:
    """Describe full lower-owned path assessment without motion or HOLD."""

    return SkillDefinition(
        skill_id=PLAN_GRASP_APPROACH_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary=(
            "Assess one retained point-grasp candidate with the deployed RM65 "
            "target envelope, endpoint IK, production URDF/FCL scene, and full "
            "joint-path planner without granting or dispatching motion."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=tuple(sorted(controlled_component_ids)),
        parameter_semantics_id=(
            "retained_point_grasp.candidate_ordinal.speed_scale.v1"
        ),
        execution_result_semantics_id=(
            "immutable_session_local_grasp_path_assessment.v1"
        ),
        task_outcome_semantics_id="passive_grasp_path_assessment.v1",
        entry_predicate_ids=(
            "retained_station_point_grasp_result_available.v1",
            "fresh_measured_qpos_available.v1",
            "production_urdf_fcl_planner_available.v1",
        ),
        task_outcome_predicate_ids=("grasp_path_assessed_without_motion.v1",),
    )


def _nullable(schema: Mapping[str, Any]) -> dict[str, Any]:
    return {"anyOf": [dict(schema), {"type": "null"}]}


def grasp_approach_plan_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    source_backend_id: str,
    source_model_profile_id: str,
    source_model_profile_hash: str,
    max_candidates: int,
    min_speed_scale: float,
    max_speed_scale: float,
) -> AgentSkillSchemaEntry:
    """Build the strict paired planning schema."""

    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "point_grasp_operation_id": {
                "type": "string",
                "minLength": 1,
                "maxLength": 256,
            },
            "candidate_ordinal": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_candidates - 1,
            },
            "speed_scale": {
                "type": "number",
                "minimum": min_speed_scale,
                "maximum": max_speed_scale,
            },
        },
        "required": [
            "point_grasp_operation_id",
            "candidate_ordinal",
            "speed_scale",
        ],
    }
    digest = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    matrix = {
        "type": "array",
        "minItems": 16,
        "maxItems": 16,
        "items": {"type": "number"},
    }
    envelope_projection = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "translation_modified": {"type": "boolean"},
            "rotation_modified": {"type": "boolean"},
            "translation_adjustment_m": {"type": "number", "minimum": 0},
            "rotation_adjustment_deg": {
                "type": "number",
                "minimum": 0,
                "maximum": 180.000001,
            },
            "violated_translation_constraints": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "active_translation_constraints": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "raw_target_xyz_m": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "number"},
            },
            "effective_target_xyz_m": {
                "type": "array",
                "minItems": 3,
                "maxItems": 3,
                "items": {"type": "number"},
            },
        },
    }
    diagnostics = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "cycles": {"type": "integer", "minimum": 0},
            "best_position_residual_m": {"type": "number", "minimum": 0},
            "best_orientation_residual_deg": {"type": "number", "minimum": 0},
            "modified_sides": {
                "type": "array",
                "uniqueItems": True,
                "items": {"type": "string", "enum": ["left", "right"]},
            },
            "path_status": {"type": "string", "minLength": 1},
            "segment_index": {"type": "integer", "minimum": 0},
            "edge_sample_index": {"type": "integer", "minimum": 0},
            "exception_type": {"type": "string", "minLength": 1},
            "envelope_projection_by_side": {
                "type": "object",
                "minProperties": 1,
                "additionalProperties": False,
                "properties": {
                    "left": envelope_projection,
                    "right": envelope_projection,
                },
            },
            "envelope_ingress_rejection": {
                "type": "string",
                "minLength": 1,
            },
        },
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "planning_status": {
                "type": "string",
                "enum": ["path_found", "rejected"],
            },
            "reason_code": {"type": "string", "minLength": 1},
            "point_grasp_operation_id": {"type": "string", "minLength": 1},
            "point_grasp_request_hash": digest,
            "point_grasp_result_hash": digest,
            "point_grasp_candidate_hash": digest,
            "source_station_revision": {"type": "integer", "minimum": 0},
            "source_backend_id": {"const": source_backend_id},
            "source_model_profile_id": {"const": source_model_profile_id},
            "source_model_profile_hash": {"const": source_model_profile_hash},
            "candidate_ordinal": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_candidates - 1,
            },
            "candidate_score": {"type": "number"},
            "component_id": {"type": "string", "minLength": 1},
            "side": {"type": "string", "enum": ["left", "right"]},
            "reference_frame_id": {"type": "string", "minLength": 1},
            "reference_frame_spec_hash": {"type": "string", "minLength": 1},
            "execution_reference_frame_id": {
                "type": "string",
                "minLength": 1,
            },
            "execution_reference_frame_spec_hash": {
                "type": "string",
                "minLength": 1,
            },
            "frame_graph_hash": {"type": "string", "minLength": 1},
            "tool_config_hash": {"type": "string", "minLength": 1},
            "calibration_hash": {"type": "string", "minLength": 1},
            "T_reference_from_tool_tcp": matrix,
            "T_execution_from_tool_tcp": matrix,
            "speed_scale": {
                "type": "number",
                "minimum": min_speed_scale,
                "maximum": max_speed_scale,
            },
            "measured_qpos_revision": _nullable({"type": "integer", "minimum": 0}),
            "measured_qpos_hash": _nullable(digest),
            "measured_qpos_at": _nullable({"type": "number"}),
            "measured_qpos_age_ms": _nullable({"type": "number", "minimum": 0}),
            "planner_evidence_refs": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "grasp_approach_plan_id": _nullable(
                {
                    "type": "string",
                    "pattern": "^real-grasp-approach-plan:[0-9a-f]{64}$",
                }
            ),
            "grasp_approach_plan_hash": _nullable(digest),
            "trajectory_hash": _nullable(digest),
            "trajectory_sample_count": _nullable({"type": "integer", "minimum": 2}),
            "trajectory_duration_sec": _nullable(
                {"type": "number", "exclusiveMinimum": 0}
            ),
            "plan_created_at": _nullable({"type": "number"}),
            "plan_expires_at": _nullable({"type": "number"}),
            "eligible_for_execution": {"type": "boolean"},
            "execution_requires_explicit_motion_skill": {"const": True},
            "motion_or_hold_performed": {"const": False},
            "diagnostics": diagnostics,
        },
        "required": [
            "planning_status",
            "reason_code",
            "point_grasp_operation_id",
            "point_grasp_request_hash",
            "point_grasp_result_hash",
            "point_grasp_candidate_hash",
            "source_station_revision",
            "source_backend_id",
            "source_model_profile_id",
            "source_model_profile_hash",
            "candidate_ordinal",
            "candidate_score",
            "component_id",
            "side",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "T_reference_from_tool_tcp",
            "T_execution_from_tool_tcp",
            "speed_scale",
            "measured_qpos_revision",
            "measured_qpos_hash",
            "measured_qpos_at",
            "measured_qpos_age_ms",
            "planner_evidence_refs",
            "grasp_approach_plan_id",
            "grasp_approach_plan_hash",
            "trajectory_hash",
            "trajectory_sample_count",
            "trajectory_duration_sec",
            "plan_created_at",
            "plan_expires_at",
            "eligible_for_execution",
            "execution_requires_explicit_motion_skill",
            "motion_or_hold_performed",
            "diagnostics",
        ],
    }
    task_schema = {
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
        skill_id=PLAN_GRASP_APPROACH_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=task_schema,
        parameter_profiles=(profile,),
    )


def _text(value: object, *, field: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ValueError(f"{field} must be a bounded trimmed string.")
    return value


def _digest(value: object, *, field: str) -> str:
    text = _text(value, field=field, maximum=64)
    if len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be a SHA-256 digest.")
    return text


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite.")
    return number


def _nonnegative_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _matrix(value: object, *, field: str) -> None:
    if not isinstance(value, (tuple, list)) or len(value) != 16:
        raise ValueError(f"{field} must contain 16 finite values.")
    if any(not math.isfinite(_finite(item, field=field)) for item in value):
        raise ValueError(f"{field} must contain 16 finite values.")


def validate_grasp_approach_plan_parameters_v1(
    payload: Mapping[str, Any],
    *,
    max_candidates: int,
    speed_scale_bounds: tuple[float, float],
) -> None:
    if not isinstance(payload, Mapping) or set(payload) != {
        "point_grasp_operation_id",
        "candidate_ordinal",
        "speed_scale",
    }:
        raise ValueError("Grasp-approach planning parameter fields changed.")
    _text(
        payload["point_grasp_operation_id"],
        field="point_grasp_operation_id",
        maximum=256,
    )
    ordinal = _nonnegative_integer(
        payload["candidate_ordinal"], field="candidate_ordinal"
    )
    if ordinal >= max_candidates:
        raise ValueError("candidate_ordinal exceeds the deployment bound.")
    speed = _finite(payload["speed_scale"], field="speed_scale")
    if not speed_scale_bounds[0] <= speed <= speed_scale_bounds[1]:
        raise ValueError("speed_scale is outside the deployment bounds.")


def validate_grasp_approach_plan_result_v1(
    payload: Mapping[str, Any],
    *,
    component_id: str,
    side: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    execution_reference_frame_id: str,
    execution_reference_frame_spec_hash: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
    source_backend_id: str,
    source_model_profile_id: str,
    source_model_profile_hash: str,
    max_candidates: int,
    speed_scale_bounds: tuple[float, float],
) -> None:
    if not isinstance(payload, Mapping):
        raise TypeError("Grasp-approach planning result must be an object.")
    required = {
        "planning_status",
        "reason_code",
        "point_grasp_operation_id",
        "point_grasp_request_hash",
        "point_grasp_result_hash",
        "point_grasp_candidate_hash",
        "source_station_revision",
        "source_backend_id",
        "source_model_profile_id",
        "source_model_profile_hash",
        "candidate_ordinal",
        "candidate_score",
        "component_id",
        "side",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "execution_reference_frame_id",
        "execution_reference_frame_spec_hash",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
        "T_reference_from_tool_tcp",
        "T_execution_from_tool_tcp",
        "speed_scale",
        "measured_qpos_revision",
        "measured_qpos_hash",
        "measured_qpos_at",
        "measured_qpos_age_ms",
        "planner_evidence_refs",
        "grasp_approach_plan_id",
        "grasp_approach_plan_hash",
        "trajectory_hash",
        "trajectory_sample_count",
        "trajectory_duration_sec",
        "plan_created_at",
        "plan_expires_at",
        "eligible_for_execution",
        "execution_requires_explicit_motion_skill",
        "motion_or_hold_performed",
        "diagnostics",
    }
    if set(payload) != required:
        raise ValueError("Grasp-approach planning result fields changed.")
    exact = {
        "component_id": component_id,
        "side": side,
        "reference_frame_id": reference_frame_id,
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "execution_reference_frame_id": execution_reference_frame_id,
        "execution_reference_frame_spec_hash": execution_reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "tool_config_hash": tool_config_hash,
        "calibration_hash": calibration_hash,
        "source_backend_id": source_backend_id,
        "source_model_profile_id": source_model_profile_id,
        "source_model_profile_hash": source_model_profile_hash,
        "execution_requires_explicit_motion_skill": True,
        "motion_or_hold_performed": False,
    }
    if any(payload[name] != expected for name, expected in exact.items()):
        raise ValueError("Grasp-approach planning deployment identity changed.")
    for field in (
        "point_grasp_request_hash",
        "point_grasp_result_hash",
        "point_grasp_candidate_hash",
    ):
        _digest(payload[field], field=field)
    for field in (
        "point_grasp_operation_id",
        "reason_code",
        "source_backend_id",
        "source_model_profile_id",
        "source_model_profile_hash",
    ):
        _text(payload[field], field=field)
    ordinal = _nonnegative_integer(
        payload["candidate_ordinal"], field="candidate_ordinal"
    )
    if ordinal >= max_candidates:
        raise ValueError("candidate_ordinal exceeds the deployment bound.")
    _nonnegative_integer(
        payload["source_station_revision"], field="source_station_revision"
    )
    _finite(payload["candidate_score"], field="candidate_score")
    speed = _finite(payload["speed_scale"], field="speed_scale")
    if not speed_scale_bounds[0] <= speed <= speed_scale_bounds[1]:
        raise ValueError("speed_scale is outside the deployment bounds.")
    qpos_fields = (
        "measured_qpos_revision",
        "measured_qpos_hash",
        "measured_qpos_at",
        "measured_qpos_age_ms",
    )
    qpos_present = tuple(payload[field] is not None for field in qpos_fields)
    if any(qpos_present) and not all(qpos_present):
        raise ValueError("Measured qpos identity must be wholly present or absent.")
    if all(qpos_present):
        _nonnegative_integer(
            payload["measured_qpos_revision"], field="measured_qpos_revision"
        )
        _digest(payload["measured_qpos_hash"], field="measured_qpos_hash")
        _finite(payload["measured_qpos_at"], field="measured_qpos_at")
        if _finite(payload["measured_qpos_age_ms"], field="measured_qpos_age_ms") < 0:
            raise ValueError("measured_qpos_age_ms cannot be negative.")
    _matrix(payload["T_reference_from_tool_tcp"], field="T_reference_from_tool_tcp")
    _matrix(payload["T_execution_from_tool_tcp"], field="T_execution_from_tool_tcp")
    refs = payload["planner_evidence_refs"]
    if (
        not isinstance(refs, (tuple, list))
        or not refs
        or len(refs) != len(set(refs))
        or any(not isinstance(item, str) or not item for item in refs)
    ):
        raise ValueError("planner_evidence_refs must be unique and non-empty.")
    diagnostics = payload["diagnostics"]
    allowed_diagnostics = {
        "cycles",
        "best_position_residual_m",
        "best_orientation_residual_deg",
        "modified_sides",
        "path_status",
        "segment_index",
        "edge_sample_index",
        "exception_type",
        "envelope_projection_by_side",
        "envelope_ingress_rejection",
    }
    if not isinstance(diagnostics, Mapping) or set(diagnostics) - allowed_diagnostics:
        raise ValueError("diagnostics contains unsupported fields.")
    status = payload["planning_status"]
    plan_fields = (
        "grasp_approach_plan_id",
        "grasp_approach_plan_hash",
        "trajectory_hash",
        "trajectory_sample_count",
        "trajectory_duration_sec",
        "plan_created_at",
        "plan_expires_at",
    )
    if status == "path_found":
        if (
            payload["reason_code"] != "path_found"
            or not payload["eligible_for_execution"]
        ):
            raise ValueError("Path-found result eligibility is inconsistent.")
        if any(payload[field] is None for field in plan_fields):
            raise ValueError("Path-found result must include immutable plan identity.")
        if not all(qpos_present):
            raise ValueError("Path-found result must bind a measured qpos cut.")
        plan_hash = _digest(
            payload["grasp_approach_plan_hash"], field="grasp_approach_plan_hash"
        )
        if payload["grasp_approach_plan_id"] != "real-grasp-approach-plan:" + plan_hash:
            raise ValueError("Grasp-approach plan ID/hash mismatch.")
        _digest(payload["trajectory_hash"], field="trajectory_hash")
        count = payload["trajectory_sample_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("trajectory_sample_count must be at least two.")
        if (
            _finite(payload["trajectory_duration_sec"], field="trajectory_duration_sec")
            <= 0
        ):
            raise ValueError("trajectory_duration_sec must be positive.")
        created = _finite(payload["plan_created_at"], field="plan_created_at")
        if _finite(payload["plan_expires_at"], field="plan_expires_at") <= created:
            raise ValueError("plan_expires_at must follow plan_created_at.")
    elif status == "rejected":
        if payload["eligible_for_execution"] or any(
            payload[field] is not None for field in plan_fields
        ):
            raise ValueError("Rejected result cannot expose an executable plan.")
    else:
        raise ValueError("Unsupported planning_status.")


def validate_grasp_approach_plan_result_for_parameters_v1(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    if result["point_grasp_operation_id"] != parameters["point_grasp_operation_id"]:
        raise ValueError("Planning result source operation differs from request.")
    if result["candidate_ordinal"] != parameters["candidate_ordinal"]:
        raise ValueError("Planning result candidate differs from request.")
    if result["speed_scale"] != parameters["speed_scale"]:
        raise ValueError("Planning result speed_scale differs from request.")


__all__ = [
    "PLAN_GRASP_APPROACH_SKILL_ID_V1",
    "grasp_approach_plan_schema_entry_v1",
    "plan_grasp_approach_skill_definition_v1",
    "validate_grasp_approach_plan_parameters_v1",
    "validate_grasp_approach_plan_result_for_parameters_v1",
    "validate_grasp_approach_plan_result_v1",
]

"""Strict public schemas for passive point-conditioned grasp planning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

from .definition import PLAN_POINT_GRASP_SKILL_ID_V1


def _matrix_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 16,
        "maxItems": 16,
        "items": {"type": "number"},
    }


def _vector_schema(length: int) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": length,
        "maxItems": length,
        "items": {"type": "number"},
    }


def _mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_mutable_json(item) for item in value]
    return value


def point_grasp_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    max_candidates: int,
    backend_id: str,
) -> AgentSkillSchemaEntry:
    nullable_path = {
        "anyOf": [
            {"type": "string", "minLength": 1, "maxLength": 4096},
            {"type": "null"},
        ]
    }
    intrinsics = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "fx": {"type": "number", "exclusiveMinimum": 0},
            "fy": {"type": "number", "exclusiveMinimum": 0},
            "cx": {"type": "number", "minimum": 0},
            "cy": {"type": "number", "minimum": 0},
        },
        "required": ["width", "height", "fx", "fy", "cx", "cy"],
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_source": {"type": "string", "enum": ["artifact", "local_path"]},
            "local_rgb_path": nullable_path,
            "local_depth_path": nullable_path,
            "target_pixel_uv": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "camera_intrinsics": intrinsics,
            "camera_frame_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "base_frame_id": {"type": "string", "minLength": 1, "maxLength": 128},
            "T_base_from_camera": _matrix_schema(),
            "max_candidates": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_candidates,
            },
        },
        "required": [
            "input_source",
            "local_rgb_path",
            "local_depth_path",
            "target_pixel_uv",
            "camera_intrinsics",
            "camera_frame_id",
            "base_frame_id",
            "T_base_from_camera",
            "max_candidates",
        ],
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"type": "integer", "minimum": 0},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "model_score": {"type": "number", "minimum": 0, "maximum": 1},
            "click_distance_m": {"type": "number", "minimum": 0},
            "gripper_width_m": {"type": "number", "minimum": 0.005, "maximum": 0.070},
            "T_camera_from_tcp": _matrix_schema(),
            "T_base_from_tcp": _matrix_schema(),
            "tcp_xyz_camera_m": _vector_schema(3),
            "tcp_xyz_base_m": _vector_schema(3),
            "visible_collision_free": {"const": True},
            "pose_source": {
                "type": "string",
                "enum": [
                    "dmtac_arm_side_approach_variant_v1",
                    "dmtac_local_orientation_variant_v1",
                    "dmtac_support_clearance_variant_v1",
                ],
            },
            "orientation_variant_axis": {"type": "string", "enum": ["x", "y", "z"]},
            "orientation_variant_angle_deg": {
                "type": "number",
                "minimum": -45,
                "maximum": 45,
            },
            "support_clearance_world_z_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 0.030,
            },
        },
        "required": [
            "ordinal",
            "score",
            "model_score",
            "click_distance_m",
            "gripper_width_m",
            "T_camera_from_tcp",
            "T_base_from_tcp",
            "tcp_xyz_camera_m",
            "tcp_xyz_base_m",
            "visible_collision_free",
        ],
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "input_source": {"type": "string", "enum": ["artifact", "local_path"]},
            "rgb_media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "depth_media_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "target_pixel_uv": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "target_point_camera_m": _vector_schema(3),
            "target_point_base_m": _vector_schema(3),
            "target_mask_bbox_xyxy": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0},
            },
            "camera_frame_id": {"type": "string", "minLength": 1},
            "base_frame_id": {"type": "string", "minLength": 1},
            "pose_semantics": {"const": "homogeneous_row_major_parent_from_tool_tcp_m"},
            "gripper_profile_id": {"const": "DMTAC_M"},
            "backend_id": {"const": backend_id},
            "candidate_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_candidates,
            },
            "selected_ordinal": {
                "type": "integer",
                "minimum": 0,
                "maximum": max_candidates - 1,
            },
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_candidates,
                "items": candidate,
            },
            "model_profile_id": {"type": "string", "minLength": 1},
            "model_profile_hash": {"type": "string", "minLength": 1},
            "annotated_image_media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "input_source",
            "rgb_media_content_hash",
            "depth_media_content_hash",
            "target_pixel_uv",
            "target_point_camera_m",
            "target_point_base_m",
            "target_mask_bbox_xyxy",
            "camera_frame_id",
            "base_frame_id",
            "pose_semantics",
            "gripper_profile_id",
            "backend_id",
            "candidate_count",
            "selected_ordinal",
            "candidates",
            "model_profile_id",
            "model_profile_hash",
            "annotated_image_media_content_hash",
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
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=task_schema,
        parameter_profiles=(profile,),
    )


def point_grasp_schema_entry_v2(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    roles: tuple[str, ...],
    max_candidates: int,
    backend_id: str,
) -> AgentSkillSchemaEntry:
    """Build the live, deployment-bound point-grasp schema."""

    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "role": {"type": "string", "enum": list(roles)},
            "target_pixel_uv": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "max_candidates": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_candidates,
            },
        },
        "required": ["role", "target_pixel_uv", "max_candidates"],
    }
    intrinsics = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "fx": {"type": "number", "exclusiveMinimum": 0},
            "fy": {"type": "number", "exclusiveMinimum": 0},
            "cx": {"type": "number", "minimum": 0},
            "cy": {"type": "number", "minimum": 0},
            "fovy_deg": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": ["width", "height", "fx", "fy", "cx", "cy", "fovy_deg"],
    }
    candidate = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "ordinal": {"type": "integer", "minimum": 0},
            "score": {"type": "number", "minimum": 0, "maximum": 1},
            "model_score": {"type": "number", "minimum": 0, "maximum": 1},
            "click_distance_m": {"type": "number", "minimum": 0},
            "gripper_width_m": {
                "type": "number",
                "minimum": 0.005,
                "maximum": 0.070,
            },
            "T_camera_from_tool_tcp": _matrix_schema(),
            "T_reference_from_tool_tcp": _matrix_schema(),
            "tool_tcp_xyz_camera_m": _vector_schema(3),
            "tool_tcp_xyz_reference_m": _vector_schema(3),
            "visible_scene_collision_free": {"const": True},
            "pose_source": {
                "type": "string",
                "enum": [
                    "dmtac_arm_side_approach_variant_v1",
                    "dmtac_local_orientation_variant_v1",
                    "dmtac_support_clearance_variant_v1",
                ],
            },
            "orientation_variant_axis": {"type": "string", "enum": ["x", "y", "z"]},
            "orientation_variant_angle_deg": {
                "type": "number",
                "minimum": -45,
                "maximum": 45,
            },
            "support_clearance_world_z_m": {
                "type": "number",
                "exclusiveMinimum": 0,
                "maximum": 0.030,
            },
        },
        "required": [
            "ordinal",
            "score",
            "model_score",
            "click_distance_m",
            "gripper_width_m",
            "T_camera_from_tool_tcp",
            "T_reference_from_tool_tcp",
            "tool_tcp_xyz_camera_m",
            "tool_tcp_xyz_reference_m",
            "visible_scene_collision_free",
        ],
    }
    result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "source_id": {"type": "string", "minLength": 1},
            "role": {"type": "string", "enum": list(roles)},
            "acquired_at": {"type": "number"},
            "station_revision": {"type": "integer", "minimum": 0},
            "source_generation": {"type": "integer", "minimum": 0},
            "frame_index": {"type": "integer", "minimum": 0},
            "capture_group_id": {
                "type": "string",
                "pattern": "^real-aligned-rgbd:[0-9a-f]{64}$",
            },
            "source_binding_hash": {"type": "string", "minLength": 1},
            "frame_binding_hash": {"type": "string", "minLength": 1},
            "frame_graph_hash": {"type": "string", "minLength": 1},
            "source_frame_id": {"type": "string", "minLength": 1},
            "source_frame_spec_hash": {"type": "string", "minLength": 1},
            "reference_frame_id": {"type": "string", "minLength": 1},
            "reference_frame_spec_hash": {"type": "string", "minLength": 1},
            "width": {"type": "integer", "minimum": 1},
            "height": {"type": "integer", "minimum": 1},
            "camera_intrinsics": intrinsics,
            "T_source_from_camera": _matrix_schema(),
            "T_reference_from_camera": _matrix_schema(),
            "rgb_content_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "depth_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "target_pixel_uv": {
                "type": "array",
                "minItems": 2,
                "maxItems": 2,
                "items": {"type": "integer", "minimum": 0},
            },
            "target_point_camera_m": _vector_schema(3),
            "target_point_reference_m": _vector_schema(3),
            "target_mask_bbox_xyxy": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "integer", "minimum": 0},
            },
            "pose_semantics": {"const": "homogeneous_row_major_parent_from_tool_tcp_m"},
            "gripper_profile_id": {"const": "DMTAC_M"},
            "backend_id": {"const": backend_id},
            "candidate_count": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_candidates,
            },
            "candidates": {
                "type": "array",
                "minItems": 1,
                "maxItems": max_candidates,
                "items": candidate,
            },
            "model_profile_id": {"type": "string", "minLength": 1},
            "model_profile_hash": {"type": "string", "minLength": 1},
            "inference_implementation_hash": {"type": "string", "minLength": 1},
            "annotated_image_media_content_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
            "proposal_semantics": {
                "const": "ranked_unprivileged_motion_proposals_only"
            },
            "collision_scope": {
                "const": "visible_scene_only_not_full_scene_or_robot_safety"
            },
            "downstream_control_effects": {"const": "forbidden"},
        },
        "required": [
            "source_id",
            "role",
            "acquired_at",
            "station_revision",
            "source_generation",
            "frame_index",
            "capture_group_id",
            "source_binding_hash",
            "frame_binding_hash",
            "frame_graph_hash",
            "source_frame_id",
            "source_frame_spec_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "width",
            "height",
            "camera_intrinsics",
            "T_source_from_camera",
            "T_reference_from_camera",
            "rgb_content_hash",
            "depth_content_hash",
            "target_pixel_uv",
            "target_point_camera_m",
            "target_point_reference_m",
            "target_mask_bbox_xyxy",
            "pose_semantics",
            "gripper_profile_id",
            "backend_id",
            "candidate_count",
            "candidates",
            "model_profile_id",
            "model_profile_hash",
            "inference_implementation_hash",
            "annotated_image_media_content_hash",
            "proposal_semantics",
            "collision_scope",
            "downstream_control_effects",
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
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="2",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=task_schema,
        parameter_profiles=(profile,),
    )


def point_grasp_schema_entry_v3(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    roles: tuple[str, ...],
    max_candidates: int,
    backend_id: str,
    arm_side: str,
    evaluator_profile_id: str,
    evaluator_profile_hash: str,
    candidate_reference_frame_id: str,
    solver_reference_frame_id: str,
    solver_reference_frame_spec_hash: str,
    frame_graph_hash: str,
    time_budget_ms: float,
    max_solve_cycles: int,
    max_measured_qpos_age_ms: float,
    standard_position_tolerance_m: float,
    standard_orientation_tolerance_deg: float,
    basic_position_tolerance_m: float,
    basic_orientation_tolerance_deg: float,
) -> AgentSkillSchemaEntry:
    """Build v3 with request-gated, annotation-only arm IK reachability."""

    base = point_grasp_schema_entry_v2(
        skill_definition_hash=skill_definition_hash,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        roles=roles,
        max_candidates=max_candidates,
        backend_id=backend_id,
    )
    parameter_schema = _mutable_json(base.parameter_schema)
    parameter_schema["properties"]["include_arm_reachability"] = {"type": "boolean"}
    parameter_schema["required"].append("include_arm_reachability")

    annotation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "classification": {
                "type": "string",
                "enum": [
                    "standard_reachable",
                    "basically_reachable",
                    "difficult_to_reach",
                ],
            },
            "position_residual_m": {"type": "number", "minimum": 0},
            "orientation_residual_deg": {
                "type": "number",
                "minimum": 0,
                "maximum": 180,
            },
            "solve_cycles": {
                "type": "integer",
                "minimum": 1,
                "maximum": max_solve_cycles,
            },
            "elapsed_ms": {"type": "number", "minimum": 0},
            "termination_reason": {
                "type": "string",
                "enum": [
                    "standard_tolerance_met",
                    "time_budget_exhausted",
                    "cycle_budget_exhausted",
                    "seed_budget_exhausted",
                ],
            },
        },
        "required": [
            "classification",
            "position_residual_m",
            "orientation_residual_deg",
            "solve_cycles",
            "elapsed_ms",
            "termination_reason",
        ],
    }
    result_schema = _mutable_json(base.execution_result_schema)
    result_schema["properties"]["arm_reachability_context"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "arm_side": {"const": arm_side},
            "evaluator_profile_id": {"const": evaluator_profile_id},
            "evaluator_profile_hash": {"const": evaluator_profile_hash},
            "candidate_reference_frame_id": {"const": candidate_reference_frame_id},
            "solver_reference_frame_id": {"const": solver_reference_frame_id},
            "solver_reference_frame_spec_hash": {
                "const": solver_reference_frame_spec_hash
            },
            "frame_graph_hash": {"const": frame_graph_hash},
            "time_budget_ms": {"const": time_budget_ms},
            "max_solve_cycles": {"const": max_solve_cycles},
            "max_measured_qpos_age_ms": {"const": max_measured_qpos_age_ms},
            "standard_position_tolerance_m": {"const": standard_position_tolerance_m},
            "standard_orientation_tolerance_deg": {
                "const": standard_orientation_tolerance_deg
            },
            "basic_position_tolerance_m": {"const": basic_position_tolerance_m},
            "basic_orientation_tolerance_deg": {
                "const": basic_orientation_tolerance_deg
            },
            "measured_qpos_revision": {"type": "integer", "minimum": 0},
            "measured_qpos_at": {"type": "number"},
            "measured_qpos_age_ms": {
                "type": "number",
                "minimum": 0,
                "maximum": max_measured_qpos_age_ms,
            },
            "evaluation_scope": {
                "const": "finite_time_endpoint_ik_not_path_or_grasp_success"
            },
            "candidate_order_effect": {"const": "annotation_only_order_unchanged"},
        },
        "required": [
            "arm_side",
            "evaluator_profile_id",
            "evaluator_profile_hash",
            "candidate_reference_frame_id",
            "solver_reference_frame_id",
            "solver_reference_frame_spec_hash",
            "frame_graph_hash",
            "time_budget_ms",
            "max_solve_cycles",
            "max_measured_qpos_age_ms",
            "standard_position_tolerance_m",
            "standard_orientation_tolerance_deg",
            "basic_position_tolerance_m",
            "basic_orientation_tolerance_deg",
            "measured_qpos_revision",
            "measured_qpos_at",
            "measured_qpos_age_ms",
            "evaluation_scope",
            "candidate_order_effect",
        ],
    }
    result_schema["properties"]["candidates"]["items"]["properties"][
        "arm_reachability"
    ] = annotation_schema
    profile = SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema, schema_kind="parameter"
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=PLAN_POINT_GRASP_SKILL_ID_V1,
        skill_version="3",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=result_schema,
        task_evaluation_schema=_mutable_json(base.task_evaluation_schema),
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


def _finite_values(value: object, *, field: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain {length} finite numbers.")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise ValueError(f"{field} must contain {length} finite numbers.")
    return tuple(float(item) for item in value)


def _matrix(value: object, *, field: str) -> tuple[tuple[float, ...], ...]:
    flat = _finite_values(value, field=field, length=16)
    matrix = tuple(flat[index : index + 4] for index in range(0, 16, 4))
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-8)
        for actual, expected in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True)
    ):
        raise ValueError(f"{field} must be a homogeneous transform.")
    rotation = tuple(row[:3] for row in matrix[:3])
    for row_or_column in (
        rotation,
        tuple(tuple(rotation[row][column] for row in range(3)) for column in range(3)),
    ):
        for left in range(3):
            for right in range(3):
                dot = sum(
                    row_or_column[left][axis] * row_or_column[right][axis]
                    for axis in range(3)
                )
                expected = 1.0 if left == right else 0.0
                if not math.isclose(dot, expected, rel_tol=0.0, abs_tol=1e-5):
                    raise ValueError(f"{field} rotation is not rigid.")
    determinant = (
        rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1]
        * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2]
        * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{field} rotation is not rigid.")
    return matrix


def _vector(value: object, *, field: str, length: int) -> tuple[float, ...]:
    return _finite_values(value, field=field, length=length)


def _allclose(
    actual: tuple[float, ...],
    expected: tuple[float, ...],
    *,
    tolerance: float,
) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)
        for left, right in zip(actual, expected, strict=True)
    )


def _transform_point(
    matrix: tuple[tuple[float, ...], ...],
    point: tuple[float, ...],
) -> tuple[float, float, float]:
    return tuple(  # type: ignore[return-value]
        sum(matrix[row][column] * point[column] for column in range(3)) + matrix[row][3]
        for row in range(3)
    )


def _matmul4(
    left: tuple[tuple[float, ...], ...],
    right: tuple[tuple[float, ...], ...],
) -> tuple[tuple[float, ...], ...]:
    return tuple(
        tuple(
            sum(left[row][axis] * right[axis][column] for axis in range(4))
            for column in range(4)
        )
        for row in range(4)
    )


def validate_point_grasp_parameters_v1(
    payload: Mapping[str, Any], *, local_path_enabled: bool, max_candidates: int
) -> None:
    fields = {
        "input_source",
        "local_rgb_path",
        "local_depth_path",
        "target_pixel_uv",
        "camera_intrinsics",
        "camera_frame_id",
        "base_frame_id",
        "T_base_from_camera",
        "max_candidates",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("Point-grasp parameter fields do not match the schema.")
    source = payload["input_source"]
    if source not in {"artifact", "local_path"}:
        raise ValueError("Point-grasp input_source is unsupported.")
    paths = (payload["local_rgb_path"], payload["local_depth_path"])
    if source == "artifact" and paths != (None, None):
        raise ValueError("Artifact point-grasp input cannot include local paths.")
    if source == "local_path":
        if not local_path_enabled:
            raise ValueError("Local-path point-grasp input is disabled.")
        if any(
            not _text(value, field="local input path").startswith("/")
            for value in paths
        ):
            raise ValueError("Local point-grasp paths must be absolute.")
    point = payload["target_pixel_uv"]
    if (
        not isinstance(point, (list, tuple))
        or len(point) != 2
        or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in point)
    ):
        raise ValueError("target_pixel_uv must contain two non-negative integers.")
    intrinsics = payload["camera_intrinsics"]
    if not isinstance(intrinsics, Mapping) or set(intrinsics) != {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
    }:
        raise ValueError("camera_intrinsics fields changed.")
    width, height = intrinsics["width"], intrinsics["height"]
    if any(
        isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in (width, height)
    ):
        raise ValueError("Intrinsic dimensions must be positive integers.")
    if not (point[0] < width and point[1] < height):
        raise ValueError("target_pixel_uv is outside the calibrated image.")
    for field in ("fx", "fy", "cx", "cy"):
        value = intrinsics[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Camera intrinsics must be finite.")
    if (
        intrinsics["fx"] <= 0
        or intrinsics["fy"] <= 0
        or intrinsics["cx"] < 0
        or intrinsics["cy"] < 0
    ):
        raise ValueError("Camera intrinsics are outside their bounds.")
    _text(payload["camera_frame_id"], field="camera_frame_id", maximum=128)
    _text(payload["base_frame_id"], field="base_frame_id", maximum=128)
    _matrix(payload["T_base_from_camera"], field="T_base_from_camera")
    requested = payload["max_candidates"]
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 1 <= requested <= max_candidates
    ):
        raise ValueError("max_candidates is outside the deployment bound.")


def validate_point_grasp_result_v1(
    payload: Mapping[str, Any],
    *,
    backend_id: str,
    model_profile_id: str,
    model_profile_hash: str,
    max_candidates: int,
) -> None:
    expected = {
        "input_source",
        "rgb_media_content_hash",
        "depth_media_content_hash",
        "target_pixel_uv",
        "target_point_camera_m",
        "target_point_base_m",
        "target_mask_bbox_xyxy",
        "camera_frame_id",
        "base_frame_id",
        "pose_semantics",
        "gripper_profile_id",
        "backend_id",
        "candidate_count",
        "selected_ordinal",
        "candidates",
        "model_profile_id",
        "model_profile_hash",
        "annotated_image_media_content_hash",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Point-grasp result fields changed.")
    if (
        payload["model_profile_id"] != model_profile_id
        or payload["model_profile_hash"] != model_profile_hash
    ):
        raise ValueError("Point-grasp model identity changed.")
    if (
        payload["pose_semantics"] != "homogeneous_row_major_parent_from_tool_tcp_m"
        or payload["gripper_profile_id"] != "DMTAC_M"
    ):
        raise ValueError("Point-grasp pose or gripper semantics changed.")
    if payload["backend_id"] != backend_id:
        raise ValueError("Point-grasp backend identity changed.")
    for field in (
        "rgb_media_content_hash",
        "depth_media_content_hash",
        "annotated_image_media_content_hash",
    ):
        value = payload[field]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(x not in "0123456789abcdef" for x in value)
        ):
            raise ValueError(f"{field} must be a SHA-256 digest.")
    candidates = payload["candidates"]
    if (
        not isinstance(candidates, (list, tuple))
        or not 1 <= len(candidates) <= max_candidates
        or payload["candidate_count"] != len(candidates)
        or payload["selected_ordinal"] != 0
    ):
        raise ValueError("Point-grasp candidate selection is invalid.")
    fields = {
        "ordinal",
        "score",
        "model_score",
        "click_distance_m",
        "gripper_width_m",
        "T_camera_from_tcp",
        "T_base_from_tcp",
        "tcp_xyz_camera_m",
        "tcp_xyz_base_m",
        "visible_collision_free",
    }
    orientation_variant_fields = {
        "pose_source",
        "orientation_variant_axis",
        "orientation_variant_angle_deg",
    }
    support_clearance_variant_fields = {
        "pose_source",
        "support_clearance_world_z_m",
    }
    pose_source_fields = {"pose_source"}
    previous_score = math.inf
    for ordinal, candidate in enumerate(candidates):
        if (
            not isinstance(candidate, Mapping)
            or frozenset(candidate)
            not in {
                frozenset(fields),
                frozenset(fields | pose_source_fields),
                frozenset(fields | orientation_variant_fields),
                frozenset(fields | support_clearance_variant_fields),
            }
            or candidate["ordinal"] != ordinal
        ):
            raise ValueError("Point-grasp candidate fields or ordinals changed.")
        camera = _matrix(candidate["T_camera_from_tcp"], field="T_camera_from_tcp")
        base = _matrix(candidate["T_base_from_tcp"], field="T_base_from_tcp")
        camera_xyz = tuple(camera[row][3] for row in range(3))
        base_xyz = tuple(base[row][3] for row in range(3))
        candidate_camera_xyz = _vector(
            candidate["tcp_xyz_camera_m"], field="tcp_xyz_camera_m", length=3
        )
        candidate_base_xyz = _vector(
            candidate["tcp_xyz_base_m"], field="tcp_xyz_base_m", length=3
        )
        if not _allclose(
            camera_xyz, candidate_camera_xyz, tolerance=1e-8
        ) or not _allclose(base_xyz, candidate_base_xyz, tolerance=1e-8):
            raise ValueError("Point-grasp candidate xyz differs from its transform.")
        if candidate["visible_collision_free"] is not True:
            raise ValueError(
                "Point-grasp candidate is not visible-scene collision free."
            )
        if candidate.get("pose_source") == "dmtac_local_orientation_variant_v1":
            if not orientation_variant_fields.issubset(candidate):
                raise ValueError("Point-grasp orientation variant fields are incomplete.")
            if candidate["orientation_variant_axis"] not in {"x", "y", "z"}:
                raise ValueError("Point-grasp orientation variant axis is invalid.")
            angle = candidate["orientation_variant_angle_deg"]
            if (
                isinstance(angle, bool)
                or not isinstance(angle, (int, float))
                or not math.isfinite(float(angle))
                or not 0 < abs(float(angle)) <= 45
            ):
                raise ValueError("Point-grasp orientation variant is invalid.")
        elif candidate.get("pose_source") == "dmtac_support_clearance_variant_v1":
            if not support_clearance_variant_fields.issubset(candidate):
                raise ValueError("Point-grasp support-clearance fields are incomplete.")
            clearance = candidate["support_clearance_world_z_m"]
            if (
                isinstance(clearance, bool)
                or not isinstance(clearance, (int, float))
                or not math.isfinite(float(clearance))
                or not 0 < float(clearance) <= 0.030
            ):
                raise ValueError("Point-grasp support clearance is invalid.")
        elif candidate.get("pose_source") == "dmtac_arm_side_approach_variant_v1":
            pass
        elif "pose_source" in candidate:
            raise ValueError("Point-grasp pose source is invalid.")
        for field in ("score", "model_score"):
            value = candidate[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError("Point-grasp scores must be in [0, 1].")
        if float(candidate["score"]) > previous_score + 1e-12:
            raise ValueError("Point-grasp candidates are not sorted by score.")
        previous_score = float(candidate["score"])
        if (
            not 0.005 <= float(candidate["gripper_width_m"]) <= 0.070
            or float(candidate["click_distance_m"]) < 0
        ):
            raise ValueError("Point-grasp geometry is outside DMTAC M bounds.")
    _vector(payload["target_point_camera_m"], field="target_point_camera_m", length=3)
    _vector(payload["target_point_base_m"], field="target_point_base_m", length=3)
    bbox = payload["target_mask_bbox_xyxy"]
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
        or not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3])
    ):
        raise ValueError("Point-grasp target mask bbox is invalid.")


def validate_point_grasp_result_for_parameters_v1(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    for field in (
        "input_source",
        "target_pixel_uv",
        "camera_frame_id",
        "base_frame_id",
    ):
        if parameters.get(field) != result.get(field):
            raise ValueError(f"Point-grasp result {field} differs from the request.")
    if result.get("candidate_count", 0) > parameters.get("max_candidates", 0):
        raise ValueError("Point-grasp result exceeds the requested candidate bound.")
    intrinsics = parameters["camera_intrinsics"]
    _, _, x1, y1 = result["target_mask_bbox_xyxy"]
    if x1 > intrinsics["width"] or y1 > intrinsics["height"]:
        raise ValueError("Point-grasp target mask bbox exceeds the input image.")
    base_from_camera = _matrix(
        parameters["T_base_from_camera"], field="T_base_from_camera"
    )
    target_camera = _vector(
        result["target_point_camera_m"], field="target_point_camera_m", length=3
    )
    target_base = _vector(
        result["target_point_base_m"], field="target_point_base_m", length=3
    )
    if not _allclose(
        _transform_point(base_from_camera, target_camera),
        target_base,
        tolerance=1e-7,
    ):
        raise ValueError(
            "Point-grasp target base point differs from the calibrated transform."
        )
    for candidate in result["candidates"]:
        camera_from_tcp = _matrix(
            candidate["T_camera_from_tcp"], field="T_camera_from_tcp"
        )
        base_from_tcp = _matrix(candidate["T_base_from_tcp"], field="T_base_from_tcp")
        composed = _matmul4(base_from_camera, camera_from_tcp)
        if any(
            not _allclose(actual, expected, tolerance=1e-7)
            for actual, expected in zip(composed, base_from_tcp, strict=True)
        ):
            raise ValueError(
                "Point-grasp base pose differs from the calibrated transform."
            )


def validate_point_grasp_parameters_v2(
    payload: Mapping[str, Any],
    *,
    dimensions_by_role: Mapping[str, tuple[int, int]],
    max_candidates: int,
) -> None:
    """Validate the semantic-only live planning request."""

    if not isinstance(payload, Mapping) or set(payload) != {
        "role",
        "target_pixel_uv",
        "max_candidates",
    }:
        raise ValueError("Station point-grasp parameter fields changed.")
    role = _text(payload["role"], field="role", maximum=128)
    if role not in dimensions_by_role:
        raise ValueError("Station point-grasp role is not registered.")
    point = payload["target_pixel_uv"]
    if (
        not isinstance(point, (list, tuple))
        or len(point) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in point
        )
    ):
        raise ValueError("target_pixel_uv must contain two non-negative integers.")
    width, height = dimensions_by_role[role]
    if not (point[0] < width and point[1] < height):
        raise ValueError("target_pixel_uv is outside the registered RGB-D frame.")
    requested = payload["max_candidates"]
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 1 <= requested <= max_candidates
    ):
        raise ValueError("max_candidates is outside the deployment bound.")


def _sha256_digest(value: object, *, field: str) -> str:
    text = _text(value, field=field, maximum=64)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a SHA-256 digest.")
    return text


def validate_point_grasp_result_v2(
    payload: Mapping[str, Any],
    *,
    source_by_role: Mapping[str, str],
    dimensions_by_role: Mapping[str, tuple[int, int]],
    source_binding_hash: str,
    frame_binding_hash: str,
    frame_graph_hash: str,
    source_frame_id: str,
    source_frame_spec_hash: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    backend_id: str,
    model_profile_id: str,
    model_profile_hash: str,
    inference_implementation_hash: str,
    max_candidates: int,
) -> None:
    """Validate identity, geometry, and proposal-only semantics for live v2."""

    expected = {
        "source_id",
        "role",
        "acquired_at",
        "station_revision",
        "source_generation",
        "frame_index",
        "capture_group_id",
        "source_binding_hash",
        "frame_binding_hash",
        "frame_graph_hash",
        "source_frame_id",
        "source_frame_spec_hash",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "width",
        "height",
        "camera_intrinsics",
        "T_source_from_camera",
        "T_reference_from_camera",
        "rgb_content_hash",
        "depth_content_hash",
        "target_pixel_uv",
        "target_point_camera_m",
        "target_point_reference_m",
        "target_mask_bbox_xyxy",
        "pose_semantics",
        "gripper_profile_id",
        "backend_id",
        "candidate_count",
        "candidates",
        "model_profile_id",
        "model_profile_hash",
        "inference_implementation_hash",
        "annotated_image_media_content_hash",
        "proposal_semantics",
        "collision_scope",
        "downstream_control_effects",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Station point-grasp result fields changed.")
    role = _text(payload["role"], field="role", maximum=128)
    if role not in source_by_role or payload["source_id"] != source_by_role[role]:
        raise ValueError("Station point-grasp source identity changed.")
    width, height = dimensions_by_role[role]
    if (payload["width"], payload["height"]) != (width, height):
        raise ValueError("Station point-grasp dimensions changed.")
    for field, expected_value in (
        ("source_binding_hash", source_binding_hash),
        ("frame_binding_hash", frame_binding_hash),
        ("frame_graph_hash", frame_graph_hash),
        ("source_frame_id", source_frame_id),
        ("source_frame_spec_hash", source_frame_spec_hash),
        ("reference_frame_id", reference_frame_id),
        ("reference_frame_spec_hash", reference_frame_spec_hash),
        ("backend_id", backend_id),
        ("model_profile_id", model_profile_id),
        ("model_profile_hash", model_profile_hash),
        ("inference_implementation_hash", inference_implementation_hash),
    ):
        if payload[field] != expected_value:
            raise ValueError(f"Station point-grasp {field} changed.")
    if (
        payload["pose_semantics"] != "homogeneous_row_major_parent_from_tool_tcp_m"
        or payload["gripper_profile_id"] != "DMTAC_M"
        or payload["proposal_semantics"] != "ranked_unprivileged_motion_proposals_only"
        or payload["collision_scope"]
        != "visible_scene_only_not_full_scene_or_robot_safety"
        or payload["downstream_control_effects"] != "forbidden"
    ):
        raise ValueError("Station point-grasp authority or geometry semantics changed.")
    for field in ("acquired_at",):
        value = payload[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError(f"{field} must be finite.")
    for field in ("station_revision", "source_generation", "frame_index"):
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer.")
    capture_group_id = _text(
        payload["capture_group_id"], field="capture_group_id", maximum=83
    )
    prefix = "real-aligned-rgbd:"
    if not capture_group_id.startswith(prefix):
        raise ValueError("capture_group_id has the wrong domain.")
    _sha256_digest(capture_group_id[len(prefix) :], field="capture_group_id hash")
    for field in (
        "rgb_content_hash",
        "depth_content_hash",
        "annotated_image_media_content_hash",
    ):
        _sha256_digest(payload[field], field=field)
    intrinsics = payload["camera_intrinsics"]
    if not isinstance(intrinsics, Mapping) or set(intrinsics) != {
        "width",
        "height",
        "fx",
        "fy",
        "cx",
        "cy",
        "fovy_deg",
    }:
        raise ValueError("Station point-grasp intrinsics fields changed.")
    if (intrinsics["width"], intrinsics["height"]) != (width, height):
        raise ValueError("Station point-grasp intrinsics dimensions changed.")
    for field in ("fx", "fy", "cx", "cy", "fovy_deg"):
        value = intrinsics[field]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("Station point-grasp intrinsics must be finite.")
    if (
        float(intrinsics["fx"]) <= 0
        or float(intrinsics["fy"]) <= 0
        or float(intrinsics["cx"]) < 0
        or float(intrinsics["cy"]) < 0
        or float(intrinsics["fovy_deg"]) <= 0
    ):
        raise ValueError("Station point-grasp intrinsics are outside their bounds.")
    _matrix(payload["T_source_from_camera"], field="T_source_from_camera")
    reference_from_camera = _matrix(
        payload["T_reference_from_camera"], field="T_reference_from_camera"
    )
    point = payload["target_pixel_uv"]
    if (
        not isinstance(point, (list, tuple))
        or len(point) != 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in point
        )
        or not (point[0] < width and point[1] < height)
    ):
        raise ValueError("Station point-grasp target pixel is invalid.")
    target_camera = _vector(
        payload["target_point_camera_m"], field="target_point_camera_m", length=3
    )
    target_reference = _vector(
        payload["target_point_reference_m"],
        field="target_point_reference_m",
        length=3,
    )
    if not _allclose(
        _transform_point(reference_from_camera, target_camera),
        target_reference,
        tolerance=1e-7,
    ):
        raise ValueError("Station point-grasp target transform is inconsistent.")
    bbox = payload["target_mask_bbox_xyxy"]
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
        or not (0 <= bbox[0] < bbox[2] <= width)
        or not (0 <= bbox[1] < bbox[3] <= height)
    ):
        raise ValueError("Station point-grasp mask bbox is invalid.")
    candidates = payload["candidates"]
    if (
        not isinstance(candidates, (list, tuple))
        or not 1 <= len(candidates) <= max_candidates
        or payload["candidate_count"] != len(candidates)
    ):
        raise ValueError("Station point-grasp candidate count is invalid.")
    candidate_fields = {
        "ordinal",
        "score",
        "model_score",
        "click_distance_m",
        "gripper_width_m",
        "T_camera_from_tool_tcp",
        "T_reference_from_tool_tcp",
        "tool_tcp_xyz_camera_m",
        "tool_tcp_xyz_reference_m",
        "visible_scene_collision_free",
    }
    orientation_variant_fields = {
        "pose_source",
        "orientation_variant_axis",
        "orientation_variant_angle_deg",
    }
    support_clearance_variant_fields = {
        "pose_source",
        "support_clearance_world_z_m",
    }
    pose_source_fields = {"pose_source"}
    previous_score = math.inf
    for ordinal, candidate in enumerate(candidates):
        if (
            not isinstance(candidate, Mapping)
            or frozenset(candidate)
            not in {
                frozenset(candidate_fields),
                frozenset(candidate_fields | pose_source_fields),
                frozenset(candidate_fields | orientation_variant_fields),
                frozenset(candidate_fields | support_clearance_variant_fields),
            }
            or candidate["ordinal"] != ordinal
        ):
            raise ValueError("Station point-grasp candidate fields changed.")
        camera_from_tcp = _matrix(
            candidate["T_camera_from_tool_tcp"], field="T_camera_from_tool_tcp"
        )
        reference_from_tcp = _matrix(
            candidate["T_reference_from_tool_tcp"],
            field="T_reference_from_tool_tcp",
        )
        composed = _matmul4(reference_from_camera, camera_from_tcp)
        if any(
            not _allclose(actual, expected_row, tolerance=1e-7)
            for actual, expected_row in zip(composed, reference_from_tcp, strict=True)
        ):
            raise ValueError("Station point-grasp candidate transform is inconsistent.")
        if not _allclose(
            tuple(camera_from_tcp[row][3] for row in range(3)),
            _vector(
                candidate["tool_tcp_xyz_camera_m"],
                field="tool_tcp_xyz_camera_m",
                length=3,
            ),
            tolerance=1e-8,
        ) or not _allclose(
            tuple(reference_from_tcp[row][3] for row in range(3)),
            _vector(
                candidate["tool_tcp_xyz_reference_m"],
                field="tool_tcp_xyz_reference_m",
                length=3,
            ),
            tolerance=1e-8,
        ):
            raise ValueError("Station point-grasp candidate xyz is inconsistent.")
        if candidate["visible_scene_collision_free"] is not True:
            raise ValueError("Candidate did not pass the bounded visible-scene check.")
        if candidate.get("pose_source") == "dmtac_local_orientation_variant_v1":
            if not orientation_variant_fields.issubset(candidate):
                raise ValueError(
                    "Station point-grasp orientation variant fields are incomplete."
                )
            if candidate["orientation_variant_axis"] not in {"x", "y", "z"}:
                raise ValueError("Station point-grasp orientation variant axis is invalid.")
            angle = candidate["orientation_variant_angle_deg"]
            if (
                isinstance(angle, bool)
                or not isinstance(angle, (int, float))
                or not math.isfinite(float(angle))
                or not 0 < abs(float(angle)) <= 45
            ):
                raise ValueError("Station point-grasp orientation variant is invalid.")
        elif candidate.get("pose_source") == "dmtac_support_clearance_variant_v1":
            if not support_clearance_variant_fields.issubset(candidate):
                raise ValueError(
                    "Station point-grasp support-clearance fields are incomplete."
                )
            clearance = candidate["support_clearance_world_z_m"]
            if (
                isinstance(clearance, bool)
                or not isinstance(clearance, (int, float))
                or not math.isfinite(float(clearance))
                or not 0 < float(clearance) <= 0.030
            ):
                raise ValueError("Station point-grasp support clearance is invalid.")
        elif candidate.get("pose_source") == "dmtac_arm_side_approach_variant_v1":
            pass
        elif "pose_source" in candidate:
            raise ValueError("Station point-grasp pose source is invalid.")
        for field in ("score", "model_score"):
            value = candidate[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= float(value) <= 1
            ):
                raise ValueError("Station point-grasp scores must be in [0, 1].")
        if float(candidate["score"]) > previous_score + 1e-12:
            raise ValueError("Station point-grasp candidates are not score-ranked.")
        previous_score = float(candidate["score"])
        if (
            not 0.005 <= float(candidate["gripper_width_m"]) <= 0.070
            or float(candidate["click_distance_m"]) < 0
        ):
            raise ValueError("Station point-grasp geometry exceeds DMTAC M bounds.")


def validate_point_grasp_result_for_parameters_v2(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    if parameters.get("role") != result.get("role"):
        raise ValueError("Station point-grasp result role differs from the request.")
    parameter_pixel = parameters.get("target_pixel_uv")
    result_pixel = result.get("target_pixel_uv")
    if (
        not isinstance(parameter_pixel, (list, tuple))
        or not isinstance(result_pixel, (list, tuple))
        or tuple(parameter_pixel) != tuple(result_pixel)
    ):
        raise ValueError("Station point-grasp result pixel differs from the request.")
    if result.get("candidate_count", 0) > parameters.get("max_candidates", 0):
        raise ValueError("Station point-grasp result exceeds the requested bound.")


def validate_point_grasp_parameters_v3(
    payload: Mapping[str, Any],
    *,
    dimensions_by_role: Mapping[str, tuple[int, int]],
    max_candidates: int,
) -> None:
    """Validate v3 while preserving the exact v2 request when disabled."""

    expected = {
        "role",
        "target_pixel_uv",
        "max_candidates",
        "include_arm_reachability",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("Station point-grasp v3 parameter fields changed.")
    if not isinstance(payload["include_arm_reachability"], bool):
        raise TypeError("include_arm_reachability must be boolean.")
    validate_point_grasp_parameters_v2(
        {key: payload[key] for key in expected - {"include_arm_reachability"}},
        dimensions_by_role=dimensions_by_role,
        max_candidates=max_candidates,
    )


def _finite_nonnegative(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"{field} must be finite and non-negative.")
    return float(value)


def validate_point_grasp_result_v3(
    payload: Mapping[str, Any],
    *,
    source_by_role: Mapping[str, str],
    dimensions_by_role: Mapping[str, tuple[int, int]],
    source_binding_hash: str,
    frame_binding_hash: str,
    frame_graph_hash: str,
    source_frame_id: str,
    source_frame_spec_hash: str,
    reference_frame_id: str,
    reference_frame_spec_hash: str,
    backend_id: str,
    model_profile_id: str,
    model_profile_hash: str,
    inference_implementation_hash: str,
    max_candidates: int,
    arm_side: str,
    evaluator_profile_id: str,
    evaluator_profile_hash: str,
    solver_reference_frame_id: str,
    solver_reference_frame_spec_hash: str,
    time_budget_ms: float,
    max_solve_cycles: int,
    max_measured_qpos_age_ms: float,
    standard_position_tolerance_m: float,
    standard_orientation_tolerance_deg: float,
    basic_position_tolerance_m: float,
    basic_orientation_tolerance_deg: float,
) -> None:
    """Validate optional reachability without changing proposal order/authority."""

    if not isinstance(payload, Mapping):
        raise TypeError("Station point-grasp v3 result must be a mapping.")
    context = payload.get("arm_reachability_context")
    candidate_payloads = payload.get("candidates")
    if not isinstance(candidate_payloads, (list, tuple)):
        raise TypeError("Station point-grasp v3 candidates are invalid.")
    annotation_presence = tuple(
        isinstance(candidate, Mapping) and "arm_reachability" in candidate
        for candidate in candidate_payloads
    )
    annotated = bool(context is not None)
    if annotation_presence and any(annotation_presence) != all(annotation_presence):
        raise ValueError("Arm reachability annotations must cover every candidate.")
    if bool(annotation_presence and all(annotation_presence)) != annotated:
        raise ValueError("Arm reachability context and annotations must agree.")

    base_payload = dict(payload)
    base_payload.pop("arm_reachability_context", None)
    base_candidates = []
    for candidate in candidate_payloads:
        if not isinstance(candidate, Mapping):
            raise TypeError("Station point-grasp v3 candidate must be a mapping.")
        base_candidate = dict(candidate)
        base_candidate.pop("arm_reachability", None)
        base_candidates.append(base_candidate)
    base_payload["candidates"] = base_candidates
    validate_point_grasp_result_v2(
        base_payload,
        source_by_role=source_by_role,
        dimensions_by_role=dimensions_by_role,
        source_binding_hash=source_binding_hash,
        frame_binding_hash=frame_binding_hash,
        frame_graph_hash=frame_graph_hash,
        source_frame_id=source_frame_id,
        source_frame_spec_hash=source_frame_spec_hash,
        reference_frame_id=reference_frame_id,
        reference_frame_spec_hash=reference_frame_spec_hash,
        backend_id=backend_id,
        model_profile_id=model_profile_id,
        model_profile_hash=model_profile_hash,
        inference_implementation_hash=inference_implementation_hash,
        max_candidates=max_candidates,
    )
    if not annotated:
        return
    assert isinstance(context, Mapping)
    expected_context = {
        "arm_side",
        "evaluator_profile_id",
        "evaluator_profile_hash",
        "candidate_reference_frame_id",
        "solver_reference_frame_id",
        "solver_reference_frame_spec_hash",
        "frame_graph_hash",
        "time_budget_ms",
        "max_solve_cycles",
        "max_measured_qpos_age_ms",
        "standard_position_tolerance_m",
        "standard_orientation_tolerance_deg",
        "basic_position_tolerance_m",
        "basic_orientation_tolerance_deg",
        "measured_qpos_revision",
        "measured_qpos_at",
        "measured_qpos_age_ms",
        "evaluation_scope",
        "candidate_order_effect",
    }
    if set(context) != expected_context:
        raise ValueError("Arm reachability context fields changed.")
    expected_values = {
        "arm_side": arm_side,
        "evaluator_profile_id": evaluator_profile_id,
        "evaluator_profile_hash": evaluator_profile_hash,
        "candidate_reference_frame_id": reference_frame_id,
        "solver_reference_frame_id": solver_reference_frame_id,
        "solver_reference_frame_spec_hash": solver_reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "time_budget_ms": time_budget_ms,
        "max_solve_cycles": max_solve_cycles,
        "max_measured_qpos_age_ms": max_measured_qpos_age_ms,
        "standard_position_tolerance_m": standard_position_tolerance_m,
        "standard_orientation_tolerance_deg": standard_orientation_tolerance_deg,
        "basic_position_tolerance_m": basic_position_tolerance_m,
        "basic_orientation_tolerance_deg": basic_orientation_tolerance_deg,
        "evaluation_scope": "finite_time_endpoint_ik_not_path_or_grasp_success",
        "candidate_order_effect": "annotation_only_order_unchanged",
    }
    for field, expected in expected_values.items():
        if context[field] != expected:
            raise ValueError(f"Arm reachability {field} changed.")
    revision = context["measured_qpos_revision"]
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise ValueError("measured_qpos_revision must be non-negative.")
    _finite_nonnegative(context["measured_qpos_at"], field="measured_qpos_at")
    age_ms = _finite_nonnegative(
        context["measured_qpos_age_ms"], field="measured_qpos_age_ms"
    )
    if age_ms > max_measured_qpos_age_ms:
        raise ValueError("Arm reachability used stale measured qpos.")

    annotation_fields = {
        "classification",
        "position_residual_m",
        "orientation_residual_deg",
        "solve_cycles",
        "elapsed_ms",
        "termination_reason",
    }
    for candidate in candidate_payloads:
        annotation = candidate["arm_reachability"]
        if not isinstance(annotation, Mapping) or set(annotation) != annotation_fields:
            raise ValueError("Arm reachability annotation fields changed.")
        position = _finite_nonnegative(
            annotation["position_residual_m"], field="position_residual_m"
        )
        orientation = _finite_nonnegative(
            annotation["orientation_residual_deg"],
            field="orientation_residual_deg",
        )
        if orientation > 180.0:
            raise ValueError("orientation_residual_deg exceeds 180 degrees.")
        cycles = annotation["solve_cycles"]
        if (
            isinstance(cycles, bool)
            or not isinstance(cycles, int)
            or not 1 <= cycles <= max_solve_cycles
        ):
            raise ValueError("solve_cycles is outside the profile bound.")
        _finite_nonnegative(annotation["elapsed_ms"], field="elapsed_ms")
        classification = annotation["classification"]
        standard = (
            position <= standard_position_tolerance_m
            and orientation <= standard_orientation_tolerance_deg
        )
        basic = (
            position <= basic_position_tolerance_m
            and orientation <= basic_orientation_tolerance_deg
        )
        expected_classification = (
            "standard_reachable"
            if standard
            else "basically_reachable"
            if basic
            else "difficult_to_reach"
        )
        if classification != expected_classification:
            raise ValueError(
                "Arm reachability classification disagrees with residuals."
            )
        reason = annotation["termination_reason"]
        if classification == "standard_reachable":
            if reason != "standard_tolerance_met":
                raise ValueError("Standard reachability has the wrong termination.")
        elif reason not in {
            "time_budget_exhausted",
            "cycle_budget_exhausted",
            "seed_budget_exhausted",
        }:
            raise ValueError("Non-standard reachability has the wrong termination.")


def validate_point_grasp_result_for_parameters_v3(
    parameters: Mapping[str, Any], result: Mapping[str, Any]
) -> None:
    base_parameters = {
        key: value
        for key, value in parameters.items()
        if key != "include_arm_reachability"
    }
    validate_point_grasp_result_for_parameters_v2(base_parameters, result)
    requested = parameters.get("include_arm_reachability")
    if not isinstance(requested, bool):
        raise TypeError("include_arm_reachability must be boolean.")
    present = "arm_reachability_context" in result
    if present != requested:
        raise ValueError("Arm reachability output does not match the request.")


__all__ = [
    "point_grasp_schema_entry_v1",
    "point_grasp_schema_entry_v2",
    "point_grasp_schema_entry_v3",
    "validate_point_grasp_parameters_v1",
    "validate_point_grasp_parameters_v2",
    "validate_point_grasp_parameters_v3",
    "validate_point_grasp_result_for_parameters_v1",
    "validate_point_grasp_result_for_parameters_v2",
    "validate_point_grasp_result_for_parameters_v3",
    "validate_point_grasp_result_v1",
    "validate_point_grasp_result_v2",
    "validate_point_grasp_result_v3",
]

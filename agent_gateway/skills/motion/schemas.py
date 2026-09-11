"""Exact Agent-facing TCP pose proposal schemas and offline validators.

These schemas describe bounded proposals only.  They do not expose joint
commands, qpos, velocity/effort control, SDK methods, action streaming, IK,
collision checking, authority, a motion permit, or dispatch.  Real historical
trajectory input is an immutable server-side reference; inline waypoints are a
separate sim-only Skill and schema.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.skills.motion.definition import (
    CARDINAL_TCP_STEP_DIRECTIONS_V1,
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_SUMMARY_BY_ID_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    INLINE_TCP_TRAJECTORY_SKILL_ID_V1,
    REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1,
    TCP_CARDINAL_STEP_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
    fixed_level_tcp_orientation_skill_definition_v1,
)

AGENT_MOTION_SCHEMA_KINDS_V1 = (
    "cardinal_tcp_step",
    "fixed_level_tcp_orientation",
    "inline_tcp_pose_trajectory",
    "registered_tcp_pose_trajectory",
    "tcp_pose_target",
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if normalized != value:
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return normalized


def _positive_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _component_ids(values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError("allowed_component_ids must be a non-empty tuple.")
    normalized = tuple(
        _required(item, field_name="allowed_component_id") for item in values
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("allowed_component_ids must be unique.")
    return tuple(sorted(normalized))


def _pose_schema(
    *,
    workspace_bounds: tuple[float, float, float, float, float, float] | None,
) -> dict[str, Any]:
    if workspace_bounds is None:
        xyz = ({"type": "number"},) * 3
    else:
        if len(workspace_bounds) != 6:
            raise ValueError("workspace_bounds must contain xyz min/max pairs.")
        values = tuple(
            _finite(item, field_name="workspace_bound") for item in workspace_bounds
        )
        if not (
            values[0] < values[1] and values[2] < values[3] and values[4] < values[5]
        ):
            raise ValueError("Each workspace minimum must be below its maximum.")
        xyz = (
            {"type": "number", "minimum": values[0], "maximum": values[1]},
            {"type": "number", "minimum": values[2], "maximum": values[3]},
            {"type": "number", "minimum": values[4], "maximum": values[5]},
        )
    quaternion = tuple(
        {"type": "number", "minimum": -1.0, "maximum": 1.0} for _ in range(4)
    )
    return {
        "type": "array",
        "minItems": 7,
        "maxItems": 7,
        "prefixItems": [*xyz, *quaternion],
        "items": False,
    }


def _poses_by_component_schema(
    *,
    component_ids: tuple[str, ...],
    workspace_bounds_by_component: Mapping[
        str,
        tuple[float, float, float, float, float, float],
    ]
    | None = None,
) -> dict[str, Any]:
    bounds = (
        {}
        if workspace_bounds_by_component is None
        else dict(workspace_bounds_by_component)
    )
    unknown = sorted(set(bounds) - set(component_ids))
    if unknown:
        raise ValueError(
            f"Workspace bounds reference unsupported components: {unknown}."
        )
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {
            component_id: _pose_schema(workspace_bounds=bounds.get(component_id))
            for component_id in component_ids
        },
    }


def _execution_result_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "execution_receipt_refs": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1},
            },
            "final_measured_state_hash": {"type": "string", "minLength": 1},
        },
        "required": ["execution_receipt_refs", "final_measured_state_hash"],
    }


def _task_evaluation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "aggregate_evaluation_receipt_ref": {
                "type": "string",
                "minLength": 1,
            },
            "task_outcome": {
                "type": "string",
                "enum": ["failed", "inconclusive", "succeeded"],
            },
        },
        "required": ["aggregate_evaluation_receipt_ref", "task_outcome"],
    }


def _schema_entry(
    *,
    skill_id: str,
    skill_definition_hash: str,
    parameter_schema: Mapping[str, Any],
    parameter_profile_id: str,
    parameter_profile_hash: str,
) -> AgentSkillSchemaEntry:
    profile = SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema,
            schema_kind="parameter",
        ),
    )
    return AgentSkillSchemaEntry(
        skill_id=skill_id,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=_execution_result_schema(),
        task_evaluation_schema=_task_evaluation_schema(),
        parameter_profiles=(profile,),
    )


def tcp_pose_target_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    max_requested_duration_ms: int | None,
    workspace_bounds_by_component: Mapping[
        str,
        tuple[float, float, float, float, float, float],
    ]
    | None = None,
    min_speed_scale: float | None = None,
    max_speed_scale: float | None = None,
    skill_id: str = TCP_POSE_TARGET_SKILL_ID_V1,
) -> AgentSkillSchemaEntry:
    selected_skill_id = _required(skill_id, field_name="skill_id")
    if selected_skill_id not in {
        TCP_POSE_TARGET_SKILL_ID_V1,
        GRASP_APPROACH_SKILL_ID_V1,
    }:
        raise ValueError("skill_id is not an absolute TCP-pose Skill.")
    components = _component_ids(allowed_component_ids)
    frames = _component_ids(allowed_reference_frame_ids)
    maximum_duration = (
        None
        if max_requested_duration_ms is None
        else _positive_integer(
            max_requested_duration_ms,
            field_name="max_requested_duration_ms",
        )
    )
    speed_scale_bounds: tuple[float, float] | None = None
    if min_speed_scale is not None or max_speed_scale is not None:
        if min_speed_scale is None or max_speed_scale is None:
            raise ValueError(
                "min_speed_scale and max_speed_scale must be provided together."
            )
        minimum_speed = _finite(min_speed_scale, field_name="min_speed_scale")
        maximum_speed = _finite(max_speed_scale, field_name="max_speed_scale")
        if not 0.0 < minimum_speed <= maximum_speed <= 1.0:
            raise ValueError("speed_scale bounds must satisfy 0 < min <= max <= 1.")
        speed_scale_bounds = (minimum_speed, maximum_speed)
    properties = {
        "pose_targets_by_component": _poses_by_component_schema(
            component_ids=components,
            workspace_bounds_by_component=workspace_bounds_by_component,
        ),
        "reference_frame_id": {"type": "string", "enum": list(frames)},
        "reference_frame_spec_hash": {"type": "string", "minLength": 1},
        "controlled_frame_id": {"const": "tool_tcp"},
        "frame_graph_hash": {"type": "string", "minLength": 1},
        "tool_config_hash": {"type": "string", "minLength": 1},
        "calibration_hash": {"type": "string", "minLength": 1},
    }
    if maximum_duration is not None:
        properties["requested_duration_ms"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": maximum_duration,
        }
    if speed_scale_bounds is not None:
        properties["speed_scale"] = {
            "type": "number",
            "minimum": speed_scale_bounds[0],
            "maximum": speed_scale_bounds[1],
            "default": 1.0,
            "description": (
                "Optional fraction of the lower reviewed trajectory-speed limits. "
                "It can only reduce, never expand, those limits."
            ),
        }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": [
            "pose_targets_by_component",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ],
    }
    return _schema_entry(
        skill_id=selected_skill_id,
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )


def grasp_approach_plan_execution_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
) -> AgentSkillSchemaEntry:
    """Build the motion schema that accepts only one immutable plan reference."""

    components = _component_ids(allowed_component_ids)
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_id": {"type": "string", "enum": list(components)},
            "grasp_approach_plan_id": {
                "type": "string",
                "pattern": "^real-grasp-approach-plan:[0-9a-f]{64}$",
            },
            "grasp_approach_plan_hash": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "required": [
            "component_id",
            "grasp_approach_plan_id",
            "grasp_approach_plan_hash",
        ],
    }
    return _schema_entry(
        skill_id=EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )


def cardinal_tcp_step_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
    primitive_catalog_hash: str,
    allowed_directions: tuple[str, ...] = CARDINAL_TCP_STEP_DIRECTIONS_V1,
    skill_id: str = TCP_CARDINAL_STEP_SKILL_ID_V1,
) -> AgentSkillSchemaEntry:
    """Publish one bounded semantic step over a lower-owned primitive catalog.

    The request names only the component and semantic direction.  Exact frame,
    calibration and primitive-catalog identities are immutable profile facts;
    callers repeat them for lineage but cannot choose an alternative value.
    Distance, target derivation, safety, IK and dispatch remain lower-owned.
    """

    selected_skill_id = _required(skill_id, field_name="skill_id")
    if selected_skill_id != TCP_CARDINAL_STEP_SKILL_ID_V1 and (
        selected_skill_id not in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
    ):
        raise ValueError("skill_id is not a cardinal TCP-step Skill.")
    components = _component_ids(allowed_component_ids)
    frames = _component_ids(allowed_reference_frame_ids)
    directions = tuple(
        _required(value, field_name="allowed_direction") for value in allowed_directions
    )
    if not directions or len(set(directions)) != len(directions):
        raise ValueError("allowed_directions must be non-empty and unique.")
    unknown = sorted(set(directions) - set(CARDINAL_TCP_STEP_DIRECTIONS_V1))
    if unknown:
        raise ValueError(f"Unsupported cardinal TCP directions: {unknown}.")
    exact_hashes = {
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "tool_config_hash": tool_config_hash,
        "calibration_hash": calibration_hash,
        "primitive_catalog_hash": primitive_catalog_hash,
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_id": {"type": "string", "enum": list(components)},
            "direction": {"type": "string", "enum": list(directions)},
            "reference_frame_id": {"type": "string", "enum": list(frames)},
            "reference_frame_spec_hash": {
                "const": _required(
                    exact_hashes["reference_frame_spec_hash"],
                    field_name="reference_frame_spec_hash",
                )
            },
            "controlled_frame_id": {"const": "tool_tcp"},
            "frame_graph_hash": {
                "const": _required(
                    exact_hashes["frame_graph_hash"],
                    field_name="frame_graph_hash",
                )
            },
            "tool_config_hash": {
                "const": _required(
                    exact_hashes["tool_config_hash"],
                    field_name="tool_config_hash",
                )
            },
            "calibration_hash": {
                "const": _required(
                    exact_hashes["calibration_hash"],
                    field_name="calibration_hash",
                )
            },
            "primitive_catalog_hash": {
                "const": _required(
                    exact_hashes["primitive_catalog_hash"],
                    field_name="primitive_catalog_hash",
                )
            },
        },
        "required": [
            "component_id",
            "direction",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "primitive_catalog_hash",
        ],
    }
    return _schema_entry(
        skill_id=selected_skill_id,
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )


def fixed_cardinal_tcp_step_schema_entry_v1(
    *,
    skill_id: str,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
    primitive_catalog_hash: str,
) -> AgentSkillSchemaEntry:
    """Publish one direction-fixed TCP step over the shared primitive catalog."""

    selected_skill_id = _required(skill_id, field_name="skill_id")
    try:
        direction = FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1[selected_skill_id]
    except KeyError as exc:
        raise ValueError("skill_id is not a fixed cardinal TCP-step Skill.") from exc
    return cardinal_tcp_step_schema_entry_v1(
        skill_id=selected_skill_id,
        skill_definition_hash=skill_definition_hash,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        allowed_component_ids=allowed_component_ids,
        allowed_reference_frame_ids=allowed_reference_frame_ids,
        reference_frame_spec_hash=reference_frame_spec_hash,
        frame_graph_hash=frame_graph_hash,
        tool_config_hash=tool_config_hash,
        calibration_hash=calibration_hash,
        primitive_catalog_hash=primitive_catalog_hash,
        allowed_directions=(direction,),
    )


def fixed_level_tcp_orientation_schema_entry_v1(
    *,
    skill_id: str,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    reference_frame_spec_hash: str,
    frame_graph_hash: str,
    tool_config_hash: str,
    calibration_hash: str,
    primitive_hash: str,
) -> AgentSkillSchemaEntry:
    """Publish one fixed absolute level TCP heading.

    The Skill identity fixes the reviewed heading.  A caller may select only
    the controlled component and repeat immutable lineage facts; it cannot
    provide an angle, a relative rotation, or an orientation increment.
    """

    selected_skill_id = _required(skill_id, field_name="skill_id")
    if selected_skill_id not in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
        raise ValueError("skill_id is not a fixed level TCP orientation Skill.")
    components = _component_ids(allowed_component_ids)
    frames = _component_ids(allowed_reference_frame_ids)
    exact_hashes = {
        "reference_frame_spec_hash": reference_frame_spec_hash,
        "frame_graph_hash": frame_graph_hash,
        "tool_config_hash": tool_config_hash,
        "calibration_hash": calibration_hash,
        "primitive_hash": primitive_hash,
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_id": {"type": "string", "enum": list(components)},
            "reference_frame_id": {"type": "string", "enum": list(frames)},
            "reference_frame_spec_hash": {
                "const": _required(
                    exact_hashes["reference_frame_spec_hash"],
                    field_name="reference_frame_spec_hash",
                )
            },
            "controlled_frame_id": {"const": "tool_tcp"},
            "frame_graph_hash": {
                "const": _required(
                    exact_hashes["frame_graph_hash"],
                    field_name="frame_graph_hash",
                )
            },
            "tool_config_hash": {
                "const": _required(
                    exact_hashes["tool_config_hash"],
                    field_name="tool_config_hash",
                )
            },
            "calibration_hash": {
                "const": _required(
                    exact_hashes["calibration_hash"],
                    field_name="calibration_hash",
                )
            },
            "primitive_hash": {
                "const": _required(
                    exact_hashes["primitive_hash"],
                    field_name="primitive_hash",
                )
            },
        },
        "required": [
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "primitive_hash",
        ],
    }
    return _schema_entry(
        skill_id=selected_skill_id,
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )


def registered_tcp_trajectory_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_schedule_profile_ids: tuple[str, ...],
    allowed_time_scale_profile_ids: tuple[str, ...] = (),
) -> AgentSkillSchemaEntry:
    schedules = _component_ids(allowed_schedule_profile_ids)
    time_scales = tuple(sorted(allowed_time_scale_profile_ids))
    if len(set(time_scales)) != len(time_scales):
        raise ValueError("allowed_time_scale_profile_ids must be unique.")
    properties: dict[str, Any] = {
        "trajectory_id": {"type": "string", "minLength": 1},
        "trajectory_hash": {"type": "string", "minLength": 1},
        "revision": {"type": "integer", "minimum": 1},
        "source_contract_id": {"type": "string", "minLength": 1},
        "source_contract_hash": {"type": "string", "minLength": 1},
        "source_frame_contract_hash": {"type": "string", "minLength": 1},
        "source_component_schema_hash": {"type": "string", "minLength": 1},
        "sample_begin": {"type": "integer", "minimum": 0},
        "sample_end_exclusive": {"type": "integer", "minimum": 1},
        "schedule_profile_id": {"type": "string", "enum": list(schedules)},
        "schedule_profile_hash": {"type": "string", "minLength": 1},
    }
    if time_scales:
        properties.update(
            {
                "time_scale_profile_id": {
                    "type": "string",
                    "enum": list(time_scales),
                },
                "time_scale_profile_hash": {"type": "string", "minLength": 1},
            }
        )
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": [
            "trajectory_id",
            "trajectory_hash",
            "revision",
            "source_contract_id",
            "source_contract_hash",
            "source_frame_contract_hash",
            "source_component_schema_hash",
            "sample_begin",
            "sample_end_exclusive",
            "schedule_profile_id",
            "schedule_profile_hash",
        ],
    }
    return _schema_entry(
        skill_id=REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1,
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )


@dataclass(frozen=True)
class InlineTcpTrajectoryLimits:
    max_waypoints: int
    max_duration_ms: int
    min_dt_ms: int
    max_dt_ms: int
    max_translation_step_m: float
    max_rotation_step_rad: float
    max_payload_bytes: int

    def __post_init__(self) -> None:
        for field_name in (
            "max_waypoints",
            "max_duration_ms",
            "min_dt_ms",
            "max_dt_ms",
            "max_payload_bytes",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name=field_name),
            )
        if self.max_waypoints < 2:
            raise ValueError("max_waypoints must be at least 2.")
        if self.max_dt_ms < self.min_dt_ms:
            raise ValueError("max_dt_ms cannot be below min_dt_ms.")
        for field_name in (
            "max_translation_step_m",
            "max_rotation_step_rad",
        ):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)


def inline_tcp_trajectory_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    limits: InlineTcpTrajectoryLimits,
    workspace_bounds_by_component: Mapping[
        str,
        tuple[float, float, float, float, float, float],
    ]
    | None = None,
) -> AgentSkillSchemaEntry:
    if not isinstance(limits, InlineTcpTrajectoryLimits):
        raise TypeError("limits must be InlineTcpTrajectoryLimits.")
    components = _component_ids(allowed_component_ids)
    frames = _component_ids(allowed_reference_frame_ids)
    waypoint = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "timestamp_ms": {
                "type": "integer",
                "minimum": 0,
                "maximum": limits.max_duration_ms,
            },
            "pose_targets_by_component": _poses_by_component_schema(
                component_ids=components,
                workspace_bounds_by_component=workspace_bounds_by_component,
            ),
        },
        "required": ["timestamp_ms", "pose_targets_by_component"],
    }
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "waypoints": {
                "type": "array",
                "minItems": 2,
                "maxItems": limits.max_waypoints,
                "items": waypoint,
            },
            "reference_frame_id": {"type": "string", "enum": list(frames)},
            "reference_frame_spec_hash": {"type": "string", "minLength": 1},
            "controlled_frame_id": {"const": "tool_tcp"},
            "frame_graph_hash": {"type": "string", "minLength": 1},
            "tool_config_hash": {"type": "string", "minLength": 1},
            "calibration_hash": {"type": "string", "minLength": 1},
        },
        "required": [
            "waypoints",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "controlled_frame_id",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ],
        "x-dexteleop-inline-trajectory-limits": {
            "min_dt_ms": limits.min_dt_ms,
            "max_dt_ms": limits.max_dt_ms,
            "max_translation_step_m": limits.max_translation_step_m,
            "max_rotation_step_rad": limits.max_rotation_step_rad,
            "max_payload_bytes": limits.max_payload_bytes,
        },
    }
    return _schema_entry(
        skill_id=INLINE_TCP_TRAJECTORY_SKILL_ID_V1,
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )


def _pose(value: object, *, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 7:
        raise ValueError(f"{field_name} must be xyz+quat_xyzw with 7 values.")
    pose = tuple(
        _finite(item, field_name=f"{field_name}[{index}]")
        for index, item in enumerate(value)
    )
    quaternion_norm = math.sqrt(sum(item * item for item in pose[3:]))
    if abs(quaternion_norm - 1.0) > 1e-5:
        raise ValueError(f"{field_name} quaternion must be normalized quat_xyzw.")
    return pose


def _poses_by_component(
    value: object,
    *,
    allowed_component_ids: tuple[str, ...],
    field_name: str,
) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{field_name} must be a non-empty object.")
    unknown = sorted(set(value) - set(allowed_component_ids))
    if unknown:
        raise ValueError(f"{field_name} contains unsupported components: {unknown}.")
    return {
        str(component): _pose(pose, field_name=f"{field_name}.{component}")
        for component, pose in value.items()
    }


def _quaternion_distance(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    dot = abs(sum(a * b for a, b in zip(first[3:], second[3:])))
    return 2.0 * math.acos(min(1.0, max(-1.0, dot)))


def validate_agent_motion_parameters_v1(
    *,
    schema_kind: str,
    parameters: Mapping[str, Any],
    allowed_component_ids: tuple[str, ...] = (),
    allowed_cardinal_directions: tuple[str, ...] = CARDINAL_TCP_STEP_DIRECTIONS_V1,
    limits: InlineTcpTrajectoryLimits | None = None,
    speed_scale_bounds: tuple[float, float] | None = None,
) -> None:
    """Validate cross-field constraints JSON Schema cannot express."""

    if schema_kind not in AGENT_MOTION_SCHEMA_KINDS_V1:
        raise ValueError(f"Unsupported schema_kind: {schema_kind!r}.")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping.")
    if "controlled_frame_id" in parameters and (
        parameters["controlled_frame_id"] != "tool_tcp"
    ):
        raise ValueError("controlled_frame_id must be exactly 'tool_tcp'.")
    if schema_kind == "registered_tcp_pose_trajectory":
        forbidden = {"waypoints", "path", "file_path", "raw_actions"} & set(parameters)
        if forbidden:
            raise ValueError(
                "Registered real trajectory parameters cannot carry inline/path "
                f"content: {sorted(forbidden)}."
            )
        begin = parameters.get("sample_begin")
        end = parameters.get("sample_end_exclusive")
        if (
            isinstance(begin, bool)
            or not isinstance(begin, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or begin < 0
            or end <= begin
        ):
            raise ValueError("Trajectory sample slice must be non-empty and bounded.")
        return

    components = _component_ids(allowed_component_ids)
    if schema_kind == "cardinal_tcp_step":
        component_id = parameters.get("component_id")
        if component_id not in components:
            raise ValueError("Cardinal TCP step selects an unsupported component.")
        directions = tuple(allowed_cardinal_directions)
        if (
            not directions
            or len(set(directions)) != len(directions)
            or set(directions) - set(CARDINAL_TCP_STEP_DIRECTIONS_V1)
        ):
            raise ValueError(
                "allowed_cardinal_directions must be a non-empty cardinal subset."
            )
        if parameters.get("direction") not in directions:
            raise ValueError("Cardinal TCP step selects an unsupported direction.")
        return
    if schema_kind == "fixed_level_tcp_orientation":
        if parameters.get("component_id") not in components:
            raise ValueError(
                "Fixed level TCP orientation selects an unsupported component."
            )
        forbidden = {
            "angle",
            "angle_deg",
            "delta_rotation",
            "heading",
            "rotation",
            "yaw",
            "yaw_deg",
        } & set(parameters)
        if forbidden:
            raise ValueError(
                "Fixed level TCP orientation cannot carry an angle or relative "
                f"rotation: {sorted(forbidden)}."
            )
        return
    if schema_kind == "tcp_pose_target":
        _poses_by_component(
            parameters.get("pose_targets_by_component"),
            allowed_component_ids=components,
            field_name="pose_targets_by_component",
        )
        if "speed_scale" in parameters:
            if speed_scale_bounds is None or len(speed_scale_bounds) != 2:
                raise ValueError(
                    "TCP pose speed_scale requires exact profile speed_scale_bounds."
                )
            minimum = _finite(
                speed_scale_bounds[0],
                field_name="speed_scale_bounds[0]",
            )
            maximum = _finite(
                speed_scale_bounds[1],
                field_name="speed_scale_bounds[1]",
            )
            if not 0.0 < minimum <= maximum <= 1.0:
                raise ValueError("speed_scale_bounds must satisfy 0 < min <= max <= 1.")
            speed_scale = _finite(
                parameters["speed_scale"],
                field_name="speed_scale",
            )
            if not minimum <= speed_scale <= maximum:
                raise ValueError("speed_scale is outside the exact profile bounds.")
        return

    if limits is None or not isinstance(limits, InlineTcpTrajectoryLimits):
        raise TypeError("Inline trajectory validation requires typed limits.")
    encoded = json.dumps(
        parameters,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limits.max_payload_bytes:
        raise ValueError("Inline trajectory payload exceeds max_payload_bytes.")
    waypoints = parameters.get("waypoints")
    if not isinstance(waypoints, (list, tuple)) or not (
        2 <= len(waypoints) <= limits.max_waypoints
    ):
        raise ValueError("Inline trajectory waypoint count is outside profile bounds.")
    previous_time: int | None = None
    previous_poses: dict[str, tuple[float, ...]] | None = None
    first_time: int | None = None
    for index, waypoint in enumerate(waypoints):
        if not isinstance(waypoint, Mapping):
            raise TypeError(f"waypoints[{index}] must be an object.")
        timestamp = waypoint.get("timestamp_ms")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ValueError(f"waypoints[{index}].timestamp_ms must be non-negative.")
        poses = _poses_by_component(
            waypoint.get("pose_targets_by_component"),
            allowed_component_ids=components,
            field_name=f"waypoints[{index}].pose_targets_by_component",
        )
        if first_time is None:
            first_time = timestamp
        if previous_time is not None:
            dt = timestamp - previous_time
            if not limits.min_dt_ms <= dt <= limits.max_dt_ms:
                raise ValueError("Inline trajectory dt is outside profile bounds.")
            assert previous_poses is not None
            if set(poses) != set(previous_poses):
                raise ValueError("Inline trajectory component coverage must be stable.")
            for component_id, pose in poses.items():
                previous_pose = previous_poses[component_id]
                translation_step = math.sqrt(
                    sum((a - b) ** 2 for a, b in zip(pose[:3], previous_pose[:3]))
                )
                if translation_step > limits.max_translation_step_m:
                    raise ValueError(
                        f"Inline trajectory translation step exceeds profile for "
                        f"{component_id}."
                    )
                if (
                    _quaternion_distance(previous_pose, pose)
                    > limits.max_rotation_step_rad
                ):
                    raise ValueError(
                        f"Inline trajectory rotation step exceeds profile for "
                        f"{component_id}."
                    )
        previous_time = timestamp
        previous_poses = poses
    assert first_time is not None and previous_time is not None
    if previous_time - first_time > limits.max_duration_ms:
        raise ValueError("Inline trajectory duration exceeds profile bounds.")


def validate_agent_motion_route_schema_v1(
    *,
    route: SkillExecutionRoute,
    schema_entry: AgentSkillSchemaEntry,
) -> None:
    """Reject cross-mode reuse, especially inline schema on a real Route."""

    if not isinstance(route, SkillExecutionRoute):
        raise TypeError("route must be a SkillExecutionRoute.")
    if not isinstance(schema_entry, AgentSkillSchemaEntry):
        raise TypeError("schema_entry must be an AgentSkillSchemaEntry.")
    if (route.skill_id, route.skill_version) != (
        schema_entry.skill_id,
        schema_entry.skill_version,
    ):
        raise ValueError("Route and motion schema Skill identities must match.")
    schema_hashes_match = (
        route.parameter_schema_hash == schema_entry.parameter_schema_hash,
        route.execution_result_schema_hash == schema_entry.execution_result_schema_hash,
        route.task_evaluation_schema_hash == schema_entry.task_evaluation_schema_hash,
    )
    if not all(schema_hashes_match):
        raise ValueError("Route does not bind the exact motion schema hashes.")
    if route.skill_id == INLINE_TCP_TRAJECTORY_SKILL_ID_V1 and (
        route.target_kind != "sim" or "sim_active" not in route.supported_modes
    ):
        raise ValueError("Inline TCP trajectory schema is sim-only.")
    if route.target_kind == "real" and (
        route.skill_id == INLINE_TCP_TRAJECTORY_SKILL_ID_V1
    ):
        raise ValueError("Real Routes must reject inline TCP trajectory schemas.")


__all__ = [
    "AGENT_MOTION_SCHEMA_KINDS_V1",
    "CARDINAL_TCP_STEP_DIRECTIONS_V1",
    "EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1",
    "FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1",
    "FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1",
    "FIXED_LEVEL_TCP_ORIENTATION_SKILL_SUMMARY_BY_ID_V1",
    "INLINE_TCP_TRAJECTORY_SKILL_ID_V1",
    "REGISTERED_TCP_TRAJECTORY_SKILL_ID_V1",
    "TCP_CARDINAL_STEP_SKILL_ID_V1",
    "TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1",
    "TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1",
    "TCP_LEVEL_FORWARD_SKILL_ID_V1",
    "TCP_POSE_TARGET_SKILL_ID_V1",
    "InlineTcpTrajectoryLimits",
    "cardinal_tcp_step_schema_entry_v1",
    "fixed_cardinal_tcp_step_schema_entry_v1",
    "fixed_level_tcp_orientation_schema_entry_v1",
    "fixed_level_tcp_orientation_skill_definition_v1",
    "grasp_approach_plan_execution_schema_entry_v1",
    "inline_tcp_trajectory_schema_entry_v1",
    "registered_tcp_trajectory_schema_entry_v1",
    "tcp_pose_target_schema_entry_v1",
    "validate_agent_motion_parameters_v1",
    "validate_agent_motion_route_schema_v1",
]

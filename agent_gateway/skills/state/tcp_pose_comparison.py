"""Fresh TCP state relative to an authenticated historical absolute safe target.

The fixed assembly resolver authenticates the retained operation/request and
fixed source, frame, tool and calibration identities. Its profile computes
geometry errors. This module owns only portable schemas and validation; it never
reads a backend or selects a Route, target, tolerance, authority, or outcome.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
from typing import Any

from agent_gateway.protocol.contracts import (
    AGENT_SKILL_EXECUTION_OUTCOMES_V1,
    AGENT_SKILL_TASK_OUTCOMES_V1,
    AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)
from agent_gateway.protocol.state_schemas import (
    TCP_POSE_STATE_SKILL_ID_V1,
    tcp_pose_state_definition_v1,
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
)

_PARAMETER_FIELDS = {
    "component_ids", "reference_frame_id", "controlled_frame_id",
    "max_measurement_age_ms", "target_operation_id",
}
_COMPARISON_FIELDS = {
    "target_operation_id", "target_request_hash", "target_route_hash",
    "target_skill_version", "target_status", "target_execution_outcome",
    "target_task_outcome",
    "target_pose_by_component", "translation_error_m_by_component",
    "rotation_error_rad_by_component",
}
# V1 semantic validators own field values; this set only closes the V2 envelope.
_STATE_RESULT_FIELDS = {
    "observation_cut_id", "observation_cut_hash", "cutoff_at",
    "cutoff_clock_domain_id", "component_ids", "reference_frame_id",
    "reference_frame_spec_hash", "controlled_frame_id", "frame_graph_hash",
    "tool_config_hash", "calibration_hash", "measurement_id",
    "measurement_observation_hash", "source_measurement_hash", "measured_at",
    "available_at", "clock_domain_id", "pose_by_component",
    "pose_schema_hash_by_component", "representation_by_component",
    "observation_binding_id", "observation_plan_hash",
}
_ROTATION_MAX_RAD = math.pi + 1e-12


def _operation_id(value: object) -> None:
    if (not isinstance(value, str) or not 0 < len(value) <= 128
            or value != value.strip()):
        raise ValueError("target_operation_id must be a non-empty trimmed opaque string of at most 128 characters.")


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{field_name} must be finite.") from error
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")
    return number


def _finite_nonnegative(value: object, *, field_name: str) -> float:
    number = _finite(value, field_name=field_name)
    if number < 0:
        raise ValueError(f"{field_name} must be non-negative.")
    return number


def _base_parameters(
    parameters: Mapping[str, Any], *, allowed_component_ids: tuple[str, ...],
) -> dict[str, Any]:
    if not isinstance(parameters, Mapping) or set(parameters) != _PARAMETER_FIELDS:
        raise ValueError("V2 requires exactly the four V1 fields and explicit target_operation_id.")
    if parameters["target_operation_id"] is not None:
        _operation_id(parameters["target_operation_id"])
    base = {key: value for key, value in parameters.items() if key != "target_operation_id"}
    validate_agent_state_parameters_v1(
        schema_kind="tcp_pose", parameters=base,
        allowed_component_ids=allowed_component_ids,
    )
    return base


def tcp_pose_comparison_definition_v2(
    *, controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    """Extend V1 semantics without changing its definition or default selection."""
    return replace(
        tcp_pose_state_definition_v1(controlled_component_ids=controlled_component_ids),
        skill_version="2",
        summary=(
            "Read one fresh measured tool_tcp cut, optionally comparing it with an "
            "authenticated retained absolute motion.move_tcp_to_pose_safe v1/v2 target "
            "in the same fixed source, frame, tool and calibration context. "
            "Explicit null requests state only. Report translation errors in metres, "
            "full rotation errors in radians and original terminal status/outcomes. A current "
            "single-cut comparison proves no historical success, stability, HOLD, "
            "object effect or motion authorization. No motion or new task evaluation."
        ),
        parameter_semantics_id="dexteleop.agent_state.tcp_pose_comparison_query.v2",
        execution_result_semantics_id="dexteleop.agent_state.measured_tcp_pose_comparison.v2",
    )


def tcp_pose_comparison_schema_entry_v2(
    *, base_schema: AgentSkillSchemaEntry, definition_hash: str, profile_hash: str,
) -> AgentSkillSchemaEntry:
    """Extend a configured V1 state schema through a detached wire copy."""
    if not isinstance(base_schema, AgentSkillSchemaEntry):
        raise TypeError("base_schema must be an AgentSkillSchemaEntry.")
    if (base_schema.skill_id, base_schema.skill_version) != (TCP_POSE_STATE_SKILL_ID_V1, "1"):
        raise ValueError("base_schema must be state.read_tcp_pose@1.")
    # to_wire thaws deep-frozen mappings; never deepcopy their mapping proxies.
    wire = base_schema.to_wire()
    parameters = wire["parameter_schema"]
    operation_id = {"type": "string", "minLength": 1, "maxLength": 128}
    parameters["properties"]["target_operation_id"] = {"anyOf": [operation_id, {"type": "null"}]}
    parameters["required"].append("target_operation_id")
    result = wire["execution_result_schema"]
    poses = deepcopy(result["properties"]["pose_by_component"])
    errors = {}
    for field, maximum, unit in (
        ("translation_error_m_by_component", None, "metres"),
        ("rotation_error_rad_by_component", _ROTATION_MAX_RAD, "radians"),
    ):
        mapping = deepcopy(poses)
        number = {"type": "number", "minimum": 0.0, "description": f"Error in {unit}."}
        if maximum is not None:
            number["maximum"] = maximum
        mapping["properties"] = {key: deepcopy(number) for key in poses["properties"]}
        errors[field] = mapping
    hash_schema = {"type": "string", "pattern": "^[0-9a-f]{64}$", "minLength": 64, "maxLength": 64}
    comparison = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "target_operation_id": deepcopy(operation_id),
            "target_request_hash": deepcopy(hash_schema),
            "target_route_hash": deepcopy(hash_schema),
            "target_skill_version": {"type": "string", "enum": ["1", "2"]},
            "target_status": {"type": "string", "enum": list(AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1)},
            "target_execution_outcome": {"type": "string", "enum": list(AGENT_SKILL_EXECUTION_OUTCOMES_V1)},
            "target_task_outcome": {"type": "string", "enum": list(AGENT_SKILL_TASK_OUTCOMES_V1)},
            "target_pose_by_component": poses,
            **errors,
        },
        "required": sorted(_COMPARISON_FIELDS),
    }
    result["properties"]["target_comparison"] = {"anyOf": [comparison, {"type": "null"}]}
    result["required"].append("target_comparison")
    return AgentSkillSchemaEntry(
        skill_id=TCP_POSE_STATE_SKILL_ID_V1, skill_version="2",
        skill_definition_hash=definition_hash, parameter_schema=parameters,
        execution_result_schema=result, task_evaluation_schema=wire["task_evaluation_schema"],
        parameter_profiles=(SkillParameterProfileRef(
            parameter_profile_id="tcp-pose-comparison-v2",
            parameter_profile_hash=profile_hash,
            parameter_schema_hash=agent_skill_schema_document_hash(parameters, schema_kind="parameter"),
        ),),
    )


def validate_tcp_pose_comparison_parameters_v2(
    parameters: Mapping[str, Any], *, allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...], max_measurement_age_ms: int,
) -> None:
    """Validate the explicit V2 request and the registration's frame/age bounds."""
    _base_parameters(parameters, allowed_component_ids=allowed_component_ids)
    if (not isinstance(allowed_reference_frame_ids, tuple) or not allowed_reference_frame_ids
            or any(not isinstance(frame, str) or not frame or frame != frame.strip() for frame in allowed_reference_frame_ids)
            or len(set(allowed_reference_frame_ids)) != len(allowed_reference_frame_ids)):
        raise ValueError("allowed_reference_frame_ids must be a non-empty unique tuple of frame IDs.")
    if isinstance(max_measurement_age_ms, bool) or not isinstance(max_measurement_age_ms, int) or max_measurement_age_ms <= 0:
        raise ValueError("max_measurement_age_ms must be a positive integer profile bound.")
    if parameters["reference_frame_id"] not in allowed_reference_frame_ids:
        raise ValueError("reference_frame_id is not allowed by the registration.")
    if parameters["max_measurement_age_ms"] > max_measurement_age_ms:
        raise ValueError("max_measurement_age_ms exceeds the registration bound.")


def validate_tcp_pose_comparison_result_shape_v2(result: Mapping[str, Any]) -> None:
    """Validate only added comparison shape; the base validator owns V1 shape.

    No request, age policy, or target authentication is inferred from the result.
    Comparison maps share a non-empty subset of result.component_ids. The fixed
    resolver must match that subset to all original request targets; no target
    is invented for other measured components. Request matching and current-cut
    clock/freshness validation belong to the contextual validator.
    """
    if not isinstance(result, Mapping) or "target_comparison" not in result:
        raise ValueError("V2 result requires explicit target_comparison.")
    comparison = result["target_comparison"]
    if comparison is None:
        return
    if not isinstance(comparison, Mapping) or set(comparison) != _COMPARISON_FIELDS:
        raise ValueError("A target reference requires the exact non-null target_comparison fields.")
    _operation_id(comparison["target_operation_id"])
    for field in ("target_request_hash", "target_route_hash"):
        value = comparison[field]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"{field} must be a lowercase SHA256 digest.")
    for field, allowed in (
        ("target_skill_version", ("1", "2")),
        ("target_status", AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1),
        ("target_execution_outcome", AGENT_SKILL_EXECUTION_OUTCOMES_V1),
        ("target_task_outcome", AGENT_SKILL_TASK_OUTCOMES_V1),
    ):
        if not isinstance(comparison[field], str) or comparison[field] not in allowed:
            raise ValueError(f"Unsupported {field}.")
    components = result.get("component_ids")
    if (not isinstance(components, (list, tuple)) or not components
            or any(not isinstance(c, str) or not c or c != c.strip() for c in components)
            or len(set(components)) != len(components)):
        raise ValueError("Result component_ids must be non-empty and unique.")
    targets = comparison["target_pose_by_component"]
    if not isinstance(targets, Mapping) or not targets or not set(targets) <= set(components):
        raise ValueError("Comparison components must be a non-empty subset of state.component_ids.")
    compared = set(targets)
    for field in ("target_pose_by_component", "translation_error_m_by_component", "rotation_error_rad_by_component"):
        mapping = comparison[field]
        if not isinstance(mapping, Mapping) or set(mapping) != compared:
            raise ValueError(f"{field} must exactly cover the comparison components.")
    for component in compared:
        pose = comparison["target_pose_by_component"][component]
        if not isinstance(pose, (list, tuple)) or len(pose) != 7:
            raise ValueError("Target poses require seven xyz+quat_xyzw numbers.")
        numbers = [_finite(v, field_name="target_pose_by_component") for v in pose]
        # Match the public V1 state validator's quaternion normalization bound.
        if abs(math.hypot(*numbers[3:]) - 1.0) > 1e-5:
            raise ValueError("Target quaternion must be normalized quat_xyzw.")
        _finite_nonnegative(comparison["translation_error_m_by_component"][component], field_name="translation_error_m_by_component")
        angle = _finite_nonnegative(comparison["rotation_error_rad_by_component"][component], field_name="rotation_error_rad_by_component")
        if angle > _ROTATION_MAX_RAD:
            raise ValueError("rotation_error_rad_by_component exceeds pi plus numeric roundoff.")


def validate_tcp_pose_comparison_result_v2(
    parameters: Mapping[str, Any], result: Mapping[str, Any], *,
    allowed_component_ids: tuple[str, ...],
) -> None:
    """Validate portable facts, not target authentication or error recomputation.

    The registration must separately validate its allowed frame and age context.
    V1 retains ownership of the current measurement's clock and freshness gates.
    """
    base_parameters = _base_parameters(parameters, allowed_component_ids=allowed_component_ids)
    if not isinstance(result, Mapping) or set(result) != _STATE_RESULT_FIELDS | {"target_comparison"}:
        raise ValueError("V2 result requires exactly the V1 state fields and target_comparison.")
    base_result = {key: value for key, value in result.items() if key != "target_comparison"}
    validate_agent_state_result_v1(
        schema_kind="tcp_pose", parameters=base_parameters, result=base_result,
        allowed_component_ids=allowed_component_ids,
    )
    validate_tcp_pose_comparison_result_shape_v2(result)
    comparison = result["target_comparison"]
    target = parameters["target_operation_id"]
    if target is None:
        if comparison is not None:
            raise ValueError("Null target_operation_id requires null target_comparison.")
        return
    if comparison is None or comparison["target_operation_id"] != target:
        raise ValueError("Result target_operation_id does not match the query.")


__all__ = [
    "tcp_pose_comparison_definition_v2",
    "tcp_pose_comparison_schema_entry_v2",
    "validate_tcp_pose_comparison_parameters_v2",
    "validate_tcp_pose_comparison_result_shape_v2",
    "validate_tcp_pose_comparison_result_v2",
]

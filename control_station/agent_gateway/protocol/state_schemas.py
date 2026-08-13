"""Exact read-only Agent schemas for measured TCP pose state.

The records in this module are portable projections of lower-owned facts.  A
pose result references an ``ExecutionMeasurementObservation`` and an
``ObservationCutRecord``; a history result is a bounded slice projected from a
lower-owned causal history.  This module does not read a backend, own a
history, open publication, grant authority, or prove that either projection is
currently wired into a simulator or real target.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dataset_contracts.canonical_encoding import canonical_content_hash

from .skill_catalog import (
    AgentSkillSchemaEntry,
    SkillDefinition,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    agent_skill_schema_document_hash,
)

TCP_POSE_STATE_SKILL_ID_V1 = "state.read_tcp_pose"
TCP_POSE_HISTORY_STATE_SKILL_ID_V1 = "state.read_tcp_pose_history"

AGENT_STATE_SCHEMA_KINDS_V1 = (
    "tcp_pose",
    "tcp_pose_history",
)
AGENT_STATE_HISTORY_SLICE_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.tcp_pose_history_slice.v1"
)
AGENT_STATE_HISTORY_SAMPLING_POLICIES_V1 = (
    "latest",
    "uniform",
    "boundary_only",
    "boundary_plus_uniform",
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


def _nonnegative_integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _component_ids(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not values:
        raise ValueError(f"{field_name} must be a non-empty tuple.")
    normalized = tuple(_required(item, field_name=field_name) for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique.")
    return tuple(sorted(normalized))


def _requested_component_ids(
    value: object,
    *,
    allowed_component_ids: tuple[str, ...],
    field_name: str = "component_ids",
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field_name} must be a non-empty array.")
    normalized = tuple(_required(item, field_name=field_name) for item in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{field_name} must be unique.")
    unknown = sorted(set(normalized) - set(allowed_component_ids))
    if unknown:
        raise ValueError(f"{field_name} contains unsupported components: {unknown}.")
    return tuple(sorted(normalized))


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


def _exact_component_mapping(
    value: object,
    *,
    component_ids: tuple[str, ...],
    field_name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    if set(value) != set(component_ids):
        raise ValueError(f"{field_name} must exactly cover requested components.")
    return value


def _component_id_array_schema(component_ids: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 1,
        "maxItems": len(component_ids),
        "uniqueItems": True,
        "items": {"type": "string", "enum": list(component_ids)},
    }


def _pose_schema() -> dict[str, Any]:
    return {
        "type": "array",
        "minItems": 7,
        "maxItems": 7,
        "prefixItems": [
            {"type": "number"},
            {"type": "number"},
            {"type": "number"},
            *(
                {"type": "number", "minimum": -1.0, "maximum": 1.0}
                for _ in range(4)
            ),
        ],
        "items": False,
    }


def _component_object_schema(
    *,
    component_ids: tuple[str, ...],
    value_schema: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "minProperties": 1,
        "properties": {
            component_id: dict(value_schema) for component_id in component_ids
        },
    }


def _measurement_sample_schema(
    *,
    component_ids: tuple[str, ...],
    include_history_sequence: bool,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "measurement_id": {"type": "string", "minLength": 1},
        "measurement_observation_hash": {"type": "string", "minLength": 1},
        "source_measurement_hash": {"type": "string", "minLength": 1},
        "measured_at": {"type": "number"},
        "available_at": {"type": "number"},
        "clock_domain_id": {"type": "string", "minLength": 1},
        "pose_by_component": _component_object_schema(
            component_ids=component_ids,
            value_schema=_pose_schema(),
        ),
        "pose_schema_hash_by_component": _component_object_schema(
            component_ids=component_ids,
            value_schema={"type": "string", "minLength": 1},
        ),
        "representation_by_component": _component_object_schema(
            component_ids=component_ids,
            value_schema={"const": "cartesian_pose"},
        ),
        "observation_binding_id": {"type": "string", "minLength": 1},
        "observation_plan_hash": {"type": "string", "minLength": 1},
    }
    required = list(properties)
    if include_history_sequence:
        properties["history_sequence"] = {"type": "integer", "minimum": 1}
        required.append("history_sequence")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": required,
    }


def _frame_result_properties() -> dict[str, Any]:
    return {
        "reference_frame_id": {"type": "string", "minLength": 1},
        "reference_frame_spec_hash": {"type": "string", "minLength": 1},
        "controlled_frame_id": {"const": "tool_tcp"},
        "frame_graph_hash": {"type": "string", "minLength": 1},
        "tool_config_hash": {"type": "string", "minLength": 1},
        "calibration_hash": {"type": "string", "minLength": 1},
    }


def _observation_cut_result_properties() -> dict[str, Any]:
    return {
        "observation_cut_id": {"type": "string", "minLength": 1},
        "observation_cut_hash": {"type": "string", "minLength": 1},
        "cutoff_at": {"type": "number"},
        "cutoff_clock_domain_id": {"type": "string", "minLength": 1},
    }


def _parameter_profile(
    *,
    parameter_schema: Mapping[str, Any],
    parameter_profile_id: str,
    parameter_profile_hash: str,
) -> SkillParameterProfileRef:
    return SkillParameterProfileRef(
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
        parameter_schema_hash=agent_skill_schema_document_hash(
            parameter_schema,
            schema_kind="parameter",
        ),
    )


def _no_task_evaluation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    }


def tcp_pose_state_definition_v1(
    *,
    controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    components = _component_ids(
        controlled_component_ids,
        field_name="controlled_component_ids",
    )
    return SkillDefinition(
        skill_id=TCP_POSE_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary="Read one fresh measured tool_tcp pose projection.",
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=components,
        parameter_semantics_id="dexteleop.agent_state.tcp_pose_query.v1",
        execution_result_semantics_id=(
            "dexteleop.agent_state.measured_tcp_pose_projection.v1"
        ),
        task_outcome_semantics_id="dexteleop.agent_state.not_evaluated.v1",
        entry_predicate_ids=("agent_state.measured_tcp_pose_available",),
    )


def tcp_pose_history_state_definition_v1(
    *,
    controlled_component_ids: tuple[str, ...],
) -> SkillDefinition:
    components = _component_ids(
        controlled_component_ids,
        field_name="controlled_component_ids",
    )
    return SkillDefinition(
        skill_id=TCP_POSE_HISTORY_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_kind="inspection",
        summary="Read one bounded immutable measured tool_tcp pose history slice.",
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=components,
        parameter_semantics_id="dexteleop.agent_state.tcp_pose_history_query.v1",
        execution_result_semantics_id=(
            "dexteleop.agent_state.measured_tcp_pose_history_slice.v1"
        ),
        task_outcome_semantics_id="dexteleop.agent_state.not_evaluated.v1",
        entry_predicate_ids=("agent_state.measured_tcp_pose_history_available",),
    )


def tcp_pose_state_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    max_measurement_age_ms: int,
) -> AgentSkillSchemaEntry:
    components = _component_ids(
        allowed_component_ids,
        field_name="allowed_component_ids",
    )
    frames = _component_ids(
        allowed_reference_frame_ids,
        field_name="allowed_reference_frame_ids",
    )
    maximum_age = _positive_integer(
        max_measurement_age_ms,
        field_name="max_measurement_age_ms",
    )
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_ids": _component_id_array_schema(components),
            "reference_frame_id": {"type": "string", "enum": list(frames)},
            "controlled_frame_id": {"const": "tool_tcp"},
            "max_measurement_age_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": maximum_age,
            },
        },
        "required": [
            "component_ids",
            "reference_frame_id",
            "controlled_frame_id",
            "max_measurement_age_ms",
        ],
    }
    sample = _measurement_sample_schema(
        component_ids=components,
        include_history_sequence=False,
    )
    result_properties = {
        **_observation_cut_result_properties(),
        "component_ids": _component_id_array_schema(components),
        **_frame_result_properties(),
        **sample["properties"],
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": result_properties,
        "required": list(result_properties),
    }
    profile = _parameter_profile(
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )
    return AgentSkillSchemaEntry(
        skill_id=TCP_POSE_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=_no_task_evaluation_schema(),
        parameter_profiles=(profile,),
    )


@dataclass(frozen=True)
class TcpPoseHistoryLimits:
    max_samples: int
    max_window_ms: int
    max_payload_bytes: int

    def __post_init__(self) -> None:
        for field_name in ("max_samples", "max_window_ms", "max_payload_bytes"):
            object.__setattr__(
                self,
                field_name,
                _positive_integer(getattr(self, field_name), field_name=field_name),
            )


def tcp_pose_history_state_schema_entry_v1(
    *,
    skill_definition_hash: str,
    parameter_profile_id: str,
    parameter_profile_hash: str,
    allowed_component_ids: tuple[str, ...],
    allowed_reference_frame_ids: tuple[str, ...],
    limits: TcpPoseHistoryLimits,
) -> AgentSkillSchemaEntry:
    if not isinstance(limits, TcpPoseHistoryLimits):
        raise TypeError("limits must be TcpPoseHistoryLimits.")
    components = _component_ids(
        allowed_component_ids,
        field_name="allowed_component_ids",
    )
    frames = _component_ids(
        allowed_reference_frame_ids,
        field_name="allowed_reference_frame_ids",
    )
    parameter_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "component_ids": _component_id_array_schema(components),
            "reference_frame_id": {"type": "string", "enum": list(frames)},
            "controlled_frame_id": {"const": "tool_tcp"},
            "history_window_ms": {
                "type": "integer",
                "minimum": 1,
                "maximum": limits.max_window_ms,
            },
            "max_samples": {
                "type": "integer",
                "minimum": 1,
                "maximum": limits.max_samples,
            },
        },
        "required": [
            "component_ids",
            "reference_frame_id",
            "controlled_frame_id",
            "history_window_ms",
            "max_samples",
        ],
    }
    result_properties = {
        **_observation_cut_result_properties(),
        "history_source_id": {"type": "string", "minLength": 1},
        "history_source_contract_hash": {"type": "string", "minLength": 1},
        "history_slice_hash": {"type": "string", "minLength": 1},
        "window_begin_at": {"type": "number"},
        "window_end_at": {"type": "number"},
        "sampling_policy": {
            "type": "string",
            "enum": list(AGENT_STATE_HISTORY_SAMPLING_POLICIES_V1),
        },
        "sample_count": {
            "type": "integer",
            "minimum": 1,
            "maximum": limits.max_samples,
        },
        "component_ids": _component_id_array_schema(components),
        **_frame_result_properties(),
        "samples": {
            "type": "array",
            "minItems": 1,
            "maxItems": limits.max_samples,
            "items": _measurement_sample_schema(
                component_ids=components,
                include_history_sequence=True,
            ),
        },
        "truncated": {"type": "boolean"},
    }
    execution_result_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": result_properties,
        "required": list(result_properties),
        "x-dexteleop-history-limits": {
            "max_samples": limits.max_samples,
            "max_window_ms": limits.max_window_ms,
            "max_payload_bytes": limits.max_payload_bytes,
        },
    }
    profile = _parameter_profile(
        parameter_schema=parameter_schema,
        parameter_profile_id=parameter_profile_id,
        parameter_profile_hash=parameter_profile_hash,
    )
    return AgentSkillSchemaEntry(
        skill_id=TCP_POSE_HISTORY_STATE_SKILL_ID_V1,
        skill_version="1",
        skill_definition_hash=skill_definition_hash,
        parameter_schema=parameter_schema,
        execution_result_schema=execution_result_schema,
        task_evaluation_schema=_no_task_evaluation_schema(),
        parameter_profiles=(profile,),
    )


def _validate_state_parameters(
    *,
    schema_kind: str,
    parameters: Mapping[str, Any],
    allowed_component_ids: tuple[str, ...],
    limits: TcpPoseHistoryLimits | None,
) -> tuple[str, ...]:
    if schema_kind not in AGENT_STATE_SCHEMA_KINDS_V1:
        raise ValueError(f"Unsupported schema_kind: {schema_kind!r}.")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping.")
    components = _component_ids(
        allowed_component_ids,
        field_name="allowed_component_ids",
    )
    requested = _requested_component_ids(
        parameters.get("component_ids"),
        allowed_component_ids=components,
    )
    if parameters.get("controlled_frame_id") != "tool_tcp":
        raise ValueError("controlled_frame_id must be exactly 'tool_tcp'.")
    _required(parameters.get("reference_frame_id"), field_name="reference_frame_id")
    if schema_kind == "tcp_pose":
        _positive_integer(
            parameters.get("max_measurement_age_ms"),
            field_name="max_measurement_age_ms",
        )
        return requested
    if not isinstance(limits, TcpPoseHistoryLimits):
        raise TypeError("TCP pose history validation requires typed limits.")
    window = _positive_integer(
        parameters.get("history_window_ms"),
        field_name="history_window_ms",
    )
    requested_samples = _positive_integer(
        parameters.get("max_samples"),
        field_name="max_samples",
    )
    if window > limits.max_window_ms:
        raise ValueError("history_window_ms exceeds the profile bound.")
    if requested_samples > limits.max_samples:
        raise ValueError("max_samples exceeds the profile bound.")
    return requested


def validate_agent_state_parameters_v1(
    *,
    schema_kind: str,
    parameters: Mapping[str, Any],
    allowed_component_ids: tuple[str, ...],
    limits: TcpPoseHistoryLimits | None = None,
) -> None:
    """Validate state-query constraints that JSON Schema cannot express."""

    _validate_state_parameters(
        schema_kind=schema_kind,
        parameters=parameters,
        allowed_component_ids=allowed_component_ids,
        limits=limits,
    )


def _validate_frame_and_cut(
    *,
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[float, str]:
    for field_name in (
        "observation_cut_id",
        "observation_cut_hash",
        "cutoff_clock_domain_id",
        "reference_frame_spec_hash",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
    ):
        _required(result.get(field_name), field_name=field_name)
    if result.get("reference_frame_id") != parameters.get("reference_frame_id"):
        raise ValueError("Result reference_frame_id does not match the query.")
    if result.get("controlled_frame_id") != "tool_tcp":
        raise ValueError("Result controlled_frame_id must be exactly 'tool_tcp'.")
    cutoff = _finite(result.get("cutoff_at"), field_name="cutoff_at")
    return cutoff, str(result["cutoff_clock_domain_id"])


def _validate_measurement_sample(
    *,
    sample: Mapping[str, Any],
    component_ids: tuple[str, ...],
    cutoff_at: float,
    cutoff_clock_domain_id: str,
    field_name: str,
) -> tuple[float, float]:
    if not isinstance(sample, Mapping):
        raise TypeError(f"{field_name} must be an object.")
    for name in (
        "measurement_id",
        "measurement_observation_hash",
        "source_measurement_hash",
        "clock_domain_id",
        "observation_binding_id",
        "observation_plan_hash",
    ):
        _required(sample.get(name), field_name=f"{field_name}.{name}")
    if sample["clock_domain_id"] != cutoff_clock_domain_id:
        raise ValueError("Measurement and observation cut clocks must match.")
    measured_at = _finite(
        sample.get("measured_at"),
        field_name=f"{field_name}.measured_at",
    )
    available_at = _finite(
        sample.get("available_at"),
        field_name=f"{field_name}.available_at",
    )
    if available_at < measured_at:
        raise ValueError("Measurement cannot be available before measured_at.")
    if measured_at > cutoff_at or available_at > cutoff_at:
        raise ValueError("Measurement must be causally available at the cut.")
    poses = _exact_component_mapping(
        sample.get("pose_by_component"),
        component_ids=component_ids,
        field_name=f"{field_name}.pose_by_component",
    )
    for component_id, pose in poses.items():
        _pose(pose, field_name=f"{field_name}.pose_by_component.{component_id}")
    schemas = _exact_component_mapping(
        sample.get("pose_schema_hash_by_component"),
        component_ids=component_ids,
        field_name=f"{field_name}.pose_schema_hash_by_component",
    )
    representations = _exact_component_mapping(
        sample.get("representation_by_component"),
        component_ids=component_ids,
        field_name=f"{field_name}.representation_by_component",
    )
    for component_id in component_ids:
        _required(
            schemas[component_id],
            field_name=f"{field_name}.pose_schema_hash_by_component.{component_id}",
        )
        if representations[component_id] != "cartesian_pose":
            raise ValueError("State results require cartesian_pose representation.")
    return measured_at, available_at


def agent_state_history_slice_content_hash_v1(
    result: Mapping[str, Any],
) -> str:
    """Hash a history result excluding its self-referential slice hash."""

    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    payload = dict(result)
    payload.pop("history_slice_hash", None)
    return canonical_content_hash(
        payload,
        domain=AGENT_STATE_HISTORY_SLICE_HASH_DOMAIN_V1,
    )


def validate_agent_state_result_v1(
    *,
    schema_kind: str,
    parameters: Mapping[str, Any],
    result: Mapping[str, Any],
    allowed_component_ids: tuple[str, ...],
    limits: TcpPoseHistoryLimits | None = None,
) -> None:
    """Validate causal, frame, component, freshness, and history invariants."""

    requested = _validate_state_parameters(
        schema_kind=schema_kind,
        parameters=parameters,
        allowed_component_ids=allowed_component_ids,
        limits=limits,
    )
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping.")
    result_components = _requested_component_ids(
        result.get("component_ids"),
        allowed_component_ids=allowed_component_ids,
        field_name="result.component_ids",
    )
    if result_components != requested:
        raise ValueError("Result component_ids do not match the query.")
    cutoff, cutoff_clock = _validate_frame_and_cut(
        parameters=parameters,
        result=result,
    )

    if schema_kind == "tcp_pose":
        measured_at, _ = _validate_measurement_sample(
            sample=result,
            component_ids=requested,
            cutoff_at=cutoff,
            cutoff_clock_domain_id=cutoff_clock,
            field_name="result",
        )
        maximum_age_ms = _positive_integer(
            parameters.get("max_measurement_age_ms"),
            field_name="max_measurement_age_ms",
        )
        if (cutoff - measured_at) * 1_000.0 > maximum_age_ms + 1e-6:
            raise ValueError("Measured TCP pose exceeds max_measurement_age_ms.")
        return

    assert isinstance(limits, TcpPoseHistoryLimits)
    encoded = json.dumps(
        result,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limits.max_payload_bytes:
        raise ValueError("TCP pose history result exceeds max_payload_bytes.")
    for field_name in ("history_source_id", "history_source_contract_hash"):
        _required(result.get(field_name), field_name=field_name)
    if result.get("sampling_policy") not in AGENT_STATE_HISTORY_SAMPLING_POLICIES_V1:
        raise ValueError("History result has an unsupported sampling_policy.")
    window_begin = _finite(result.get("window_begin_at"), field_name="window_begin_at")
    window_end = _finite(result.get("window_end_at"), field_name="window_end_at")
    if window_end != cutoff:
        raise ValueError("History window must end at the observation cut.")
    if window_begin > window_end:
        raise ValueError("History window cannot be negative.")
    requested_window_ms = _positive_integer(
        parameters.get("history_window_ms"),
        field_name="history_window_ms",
    )
    if (window_end - window_begin) * 1_000.0 > requested_window_ms + 1e-6:
        raise ValueError("History result exceeds the requested time window.")
    samples = result.get("samples")
    if not isinstance(samples, (list, tuple)) or not samples:
        raise ValueError("History result must contain at least one sample.")
    requested_samples = _positive_integer(
        parameters.get("max_samples"),
        field_name="max_samples",
    )
    if len(samples) > requested_samples or len(samples) > limits.max_samples:
        raise ValueError("History result exceeds the requested sample bound.")
    if _nonnegative_integer(
        result.get("sample_count"),
        field_name="sample_count",
    ) != len(samples):
        raise ValueError("sample_count does not match samples.")
    if not isinstance(result.get("truncated"), bool):
        raise TypeError("truncated must be a boolean.")
    previous_sequence: int | None = None
    previous_measured_at: float | None = None
    previous_available_at: float | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"samples[{index}] must be an object.")
        sequence = _positive_integer(
            sample.get("history_sequence"),
            field_name=f"samples[{index}].history_sequence",
        )
        measured_at, available_at = _validate_measurement_sample(
            sample=sample,
            component_ids=requested,
            cutoff_at=cutoff,
            cutoff_clock_domain_id=cutoff_clock,
            field_name=f"samples[{index}]",
        )
        if measured_at < window_begin or measured_at > window_end:
            raise ValueError("History sample lies outside the declared window.")
        if previous_sequence is not None and sequence <= previous_sequence:
            raise ValueError("History sequences must be strictly increasing.")
        if previous_measured_at is not None and measured_at < previous_measured_at:
            raise ValueError("History measured_at values must be monotonic.")
        if previous_available_at is not None and available_at < previous_available_at:
            raise ValueError("History available_at values must be monotonic.")
        previous_sequence = sequence
        previous_measured_at = measured_at
        previous_available_at = available_at
    expected_hash = agent_state_history_slice_content_hash_v1(result)
    if result.get("history_slice_hash") != expected_hash:
        raise ValueError("history_slice_hash does not match the immutable slice.")


def validate_agent_state_route_schema_v1(
    *,
    route: SkillExecutionRoute,
    schema_entry: AgentSkillSchemaEntry,
) -> None:
    """Require exact T0, read-only, no-evaluation sim/real Route semantics."""

    if not isinstance(route, SkillExecutionRoute):
        raise TypeError("route must be a SkillExecutionRoute.")
    if not isinstance(schema_entry, AgentSkillSchemaEntry):
        raise TypeError("schema_entry must be an AgentSkillSchemaEntry.")
    if route.skill_id not in {
        TCP_POSE_STATE_SKILL_ID_V1,
        TCP_POSE_HISTORY_STATE_SKILL_ID_V1,
    }:
        raise ValueError("Route is not a TCP pose state Skill.")
    if (route.skill_id, route.skill_version) != (
        schema_entry.skill_id,
        schema_entry.skill_version,
    ):
        raise ValueError("Route and state schema Skill identities must match.")
    schema_hashes_match = (
        route.parameter_schema_hash == schema_entry.parameter_schema_hash,
        route.execution_result_schema_hash
        == schema_entry.execution_result_schema_hash,
        route.task_evaluation_schema_hash
        == schema_entry.task_evaluation_schema_hash,
    )
    if not all(schema_hashes_match):
        raise ValueError("Route does not bind the exact state schema hashes.")
    if route.target_kind not in {"sim", "real"}:
        raise ValueError("TCP pose state Routes must bind a sim or real target.")
    required = (
        route.control_effect_class == "read_only",
        route.privilege_tier == "t0_read_only",
        not route.may_move_robot,
        not route.may_open_publication,
        not route.requires_operator_confirmation,
        not route.requires_deadman,
        not route.requires_task_evaluation,
        route.cancellation_policy == "cancel_only",
    )
    if not all(required):
        raise ValueError("TCP pose state Routes must be strictly T0 read-only.")


__all__ = [
    "AGENT_STATE_HISTORY_SAMPLING_POLICIES_V1",
    "AGENT_STATE_HISTORY_SLICE_HASH_DOMAIN_V1",
    "AGENT_STATE_SCHEMA_KINDS_V1",
    "TCP_POSE_HISTORY_STATE_SKILL_ID_V1",
    "TCP_POSE_STATE_SKILL_ID_V1",
    "TcpPoseHistoryLimits",
    "agent_state_history_slice_content_hash_v1",
    "tcp_pose_history_state_definition_v1",
    "tcp_pose_history_state_schema_entry_v1",
    "tcp_pose_state_definition_v1",
    "tcp_pose_state_schema_entry_v1",
    "validate_agent_state_parameters_v1",
    "validate_agent_state_result_v1",
    "validate_agent_state_route_schema_v1",
]

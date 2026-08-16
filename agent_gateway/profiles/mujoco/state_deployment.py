"""Reusable passive TCP-pose state Route deployment contracts for MuJoCo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
)
from agent_gateway.protocol.state_schemas import (
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorSet,
)
from dataset_contracts.canonical_encoding import canonical_content_hash

MUJOCO_TCP_POSE_STATE_EXECUTOR_ID_V1 = (
    "control-station.mujoco-passive-tcp-pose-reader.v1"
)
MUJOCO_TCP_POSE_STATE_ROUTE_ID_V1 = "sim.mujoco.rm65.passive_tcp_pose"
MUJOCO_TCP_POSE_STATE_PARAMETER_PROFILE_ID_V1 = (
    "right-arm-tool-tcp-pose-read-v1"
)
MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1 = 500
MUJOCO_TCP_POSE_STATE_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.mujoco_tcp_pose_deployment_binding.v1"
)
MUJOCO_TCP_POSE_STATE_PROFILE_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.mujoco_tcp_pose_parameter_profile.v1"
)
MUJOCO_TCP_POSE_STATE_AVAILABILITY_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.mujoco_tcp_pose_availability.v1"
)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class MujocoPassiveTcpPoseStateDeploymentBinding:
    """Exact passive state-reader binding; no policy/evaluator/motion owner."""

    deployment_binding_id: str
    deployment_binding_version: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    observation_binding_id: str
    observation_binding_hash: str
    observation_plan_hash: str
    observation_capability_hash: str
    pose_schema_hash: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    executor_id: str
    explicit_exclusions: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "deployment_binding_id",
            "deployment_binding_version",
            "route_hash",
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "observation_binding_id",
            "observation_binding_hash",
            "observation_plan_hash",
            "observation_capability_hash",
            "pose_schema_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "executor_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        exclusions = tuple(
            sorted(
                {
                    _required(item, field_name="explicit_exclusions item")
                    for item in self.explicit_exclusions
                }
            )
        )
        if not exclusions:
            raise ValueError("State deployment must declare explicit exclusions.")
        object.__setattr__(self, "explicit_exclusions", exclusions)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "deployment_binding_id": self.deployment_binding_id,
                "deployment_binding_version": self.deployment_binding_version,
                "route_hash": self.route_hash,
                "runtime_session_id": self.runtime_session_id,
                "control_domain_id": self.control_domain_id,
                "target_binding_id": self.target_binding_id,
                "station_manifest_hash": self.station_manifest_hash,
                "execution_manifest_hash": self.execution_manifest_hash,
                "observation_binding_id": self.observation_binding_id,
                "observation_binding_hash": self.observation_binding_hash,
                "observation_plan_hash": self.observation_plan_hash,
                "observation_capability_hash": self.observation_capability_hash,
                "pose_schema_hash": self.pose_schema_hash,
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "frame_graph_hash": self.frame_graph_hash,
                "tool_config_hash": self.tool_config_hash,
                "calibration_hash": self.calibration_hash,
                "executor_id": self.executor_id,
                "controlled_component_ids": ("right_arm",),
                "controlled_frame_id": "tool_tcp",
                "target_kind": "sim",
                "read_semantics": "application_locked_passive_cut",
                "explicit_exclusions": self.explicit_exclusions,
            },
            domain=MUJOCO_TCP_POSE_STATE_BINDING_HASH_DOMAIN_V1,
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="sim",
            execution_manifest_hash=self.execution_manifest_hash,
        )


def build_mujoco_tcp_pose_state_payload_validator_set(
    *,
    route: SkillExecutionRoute,
    schema_entry: Any,
    exact: Mapping[str, str],
) -> AgentSkillPayloadValidatorSet:
    parameter_fields = {
        "component_ids",
        "reference_frame_id",
        "controlled_frame_id",
        "max_measurement_age_ms",
    }
    result_fields = set(schema_entry.execution_result_schema["properties"])

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError("TCP pose query must exactly match its parameter schema.")
        validate_agent_state_parameters_v1(
            schema_kind="tcp_pose",
            parameters=payload,
            allowed_component_ids=("right_arm",),
        )
        if tuple(payload["component_ids"]) != ("right_arm",):
            raise ValueError("TCP pose query must select exactly right_arm.")
        if payload["reference_frame_id"] != exact["reference_frame_id"]:
            raise ValueError("TCP pose query uses another reference frame.")
        maximum_age = payload["max_measurement_age_ms"]
        if (
            isinstance(maximum_age, bool)
            or not isinstance(maximum_age, int)
            or maximum_age
            > MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1
        ):
            raise ValueError("TCP pose query exceeds the measurement-age bound.")

    def validate_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != result_fields:
            raise ValueError("TCP pose result must exactly match its result schema.")
        validate_agent_state_result_v1(
            schema_kind="tcp_pose",
            parameters={
                "component_ids": ["right_arm"],
                "reference_frame_id": exact["reference_frame_id"],
                "controlled_frame_id": "tool_tcp",
                "max_measurement_age_ms": (
                    MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1
                ),
            },
            result=payload,
            allowed_component_ids=("right_arm",),
        )
        exact_checks = {
            "reference_frame_spec_hash": exact["reference_frame_spec_hash"],
            "frame_graph_hash": exact["frame_graph_hash"],
            "tool_config_hash": exact["tool_config_hash"],
            "calibration_hash": exact["calibration_hash"],
            "observation_binding_id": exact["observation_binding_id"],
            "observation_plan_hash": exact["observation_plan_hash"],
        }
        for field_name, expected in exact_checks.items():
            if payload[field_name] != expected:
                raise ValueError(f"TCP pose result {field_name} mismatch.")
        if payload["pose_schema_hash_by_component"] != {
            "right_arm": exact["pose_schema_hash"]
        }:
            raise ValueError("TCP pose result schema identity mismatch.")

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or payload:
            raise ValueError("TCP pose reads do not produce task evaluation.")

    def validate_result_for_parameters(
        parameters: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        validate_parameters(parameters)
        validate_agent_state_result_v1(
            schema_kind="tcp_pose",
            parameters=parameters,
            result=payload,
            allowed_component_ids=("right_arm",),
        )

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:mujoco-passive-tcp-pose:"
            + exact["runtime_session_id"]
        ),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters,
        validate_execution_result=validate_result,
        validate_task_evaluation=validate_task_evaluation,
        validate_execution_result_for_parameters=(
            validate_result_for_parameters
        ),
    )


__all__ = [
    "MUJOCO_TCP_POSE_STATE_AVAILABILITY_HASH_DOMAIN_V1",
    "MUJOCO_TCP_POSE_STATE_BINDING_HASH_DOMAIN_V1",
    "MUJOCO_TCP_POSE_STATE_EXECUTOR_ID_V1",
    "MUJOCO_TCP_POSE_STATE_MAX_MEASUREMENT_AGE_MS_V1",
    "MUJOCO_TCP_POSE_STATE_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_TCP_POSE_STATE_PROFILE_HASH_DOMAIN_V1",
    "MUJOCO_TCP_POSE_STATE_ROUTE_ID_V1",
    "MujocoPassiveTcpPoseStateDeploymentBinding",
    "build_mujoco_tcp_pose_state_payload_validator_set",
]

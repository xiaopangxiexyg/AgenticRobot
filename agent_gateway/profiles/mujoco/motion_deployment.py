"""Exact no-model Sim Route and deployment for Agent TCP-pose motion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from collection_runtime.source_observation_history import (
    SOURCE_OBSERVATION_HISTORY_SAMPLING_POLICIES_V1,
    SourceObservationHistoryBinding,
    SourceObservationHistoryStoreLimits,
)
from agent_gateway.protocol.evaluation_contracts import (
    SkillEvaluationPlan,
    SkillEvaluatorSpec,
)
from agent_gateway.protocol.motion_schemas import (
    CARDINAL_TCP_STEP_DIRECTIONS_V1,
    cardinal_tcp_step_schema_entry_v1,
    tcp_pose_target_schema_entry_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinition,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.protocol.state_schemas import (
    TcpPoseHistoryLimits,
    tcp_pose_history_state_definition_v1,
    tcp_pose_history_state_schema_entry_v1,
    tcp_pose_state_definition_v1,
    tcp_pose_state_schema_entry_v1,
    validate_agent_state_parameters_v1,
    validate_agent_state_result_v1,
    validate_agent_state_route_schema_v1,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from control_station.contracts import (
    ControlStationRunManifest,
    ExecutionDomainBinding,
    SourceSlotSpec,
)
from control_station.targets.authoritative import (
    authoritative_authority_epoch_source_id,
    authoritative_execution_epoch_source_id,
    authoritative_owner_registry_id,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ExecutionProfileExclusion,
    build_control_execution_manifest,
)
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    TCPPoseDisplacementPrimitive,
    cardinal_tcp_short_step,
)

from .state_deployment import MujocoPassiveTcpPoseStateDeploymentBinding

MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1 = "sim.mujoco.rm65.agent_skill_tcp_pose"
MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1 = (
    "control-station.mujoco-agent-tcp-pose.v1"
)
MUJOCO_AGENT_TCP_POSE_PARAMETER_PROFILE_ID_V1 = (
    "right-arm-bounded-tool-tcp-target-v1"
)
MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_tcp_cardinal_step"
)
MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1 = (
    "control-station.mujoco-agent-tcp-cardinal-step.v1"
)
MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1 = (
    "right-arm-cardinal-tool-tcp-short-step-v1"
)
MUJOCO_AGENT_TCP_POSE_STATE_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_passive_tcp_pose"
)
MUJOCO_AGENT_TCP_POSE_STATE_EXECUTOR_ID_V1 = (
    "control-station.mujoco-agent-passive-tcp-pose.v1"
)
MUJOCO_AGENT_TCP_POSE_STATE_PROFILE_ID_V1 = (
    "right-arm-agent-motion-tcp-pose-read-v1"
)
MUJOCO_AGENT_TCP_POSE_HISTORY_ROUTE_ID_V1 = (
    "sim.mujoco.rm65.agent_skill_passive_tcp_pose_history"
)
MUJOCO_AGENT_TCP_POSE_HISTORY_EXECUTOR_ID_V1 = (
    "control-station.mujoco-agent-passive-tcp-pose-history.v1"
)
MUJOCO_AGENT_TCP_POSE_HISTORY_PROFILE_ID_V1 = (
    "right-arm-agent-motion-tcp-pose-history-v1"
)
MUJOCO_AGENT_TCP_POSE_HISTORY_SOURCE_ID_V1 = (
    "planning-history.right-arm.tool-tcp-pose.v1"
)
MUJOCO_AGENT_TCP_POSE_HISTORY_BINDING_HASH_DOMAIN_V1 = (
    "dexteleop.agent_state.mujoco_tcp_pose_history_deployment_binding.v1"
)
MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1 = 500
MUJOCO_AGENT_TCP_POSE_TRACE_PROFILE_ID_V1 = (
    "deployable_mujoco_agent_tcp_pose_no_model_v1"
)
MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1 = (
    -0.60,
    0.60,
    -0.80,
    0.80,
    0.0,
    0.90,
)
MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1 = 0.10
MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1 = math.radians(15.0)


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a finite number.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _hash(value: object, *, field_name: str) -> str:
    return _required(
        getattr(value, "semantic_content_hash", None),
        field_name=field_name,
    )


@dataclass(frozen=True)
class MujocoAgentTcpPoseMotionProfile:
    """Immutable admission, execution cadence and terminal predicate profile."""

    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    component_id: str = "right_arm"
    workspace_bounds: tuple[float, float, float, float, float, float] = (
        MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1
    )
    max_translation_delta_m: float = (
        MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1
    )
    max_rotation_delta_rad: float = (
        MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1
    )
    translation_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = math.radians(5.0)
    stable_measurement_count: int = 3
    max_control_cycles: int = 120
    physics_ticks_per_control_cycle: int = 1

    def __post_init__(self) -> None:
        for field_name in (
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "component_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.component_id not in {"left_arm", "right_arm"}:
            raise ValueError("component_id must be left_arm or right_arm.")
        bounds = tuple(
            _finite(value, field_name="workspace_bounds item")
            for value in self.workspace_bounds
        )
        if len(bounds) != 6 or not (
            bounds[0] < bounds[1]
            and bounds[2] < bounds[3]
            and bounds[4] < bounds[5]
        ):
            raise ValueError("workspace_bounds must contain ordered xyz pairs.")
        object.__setattr__(self, "workspace_bounds", bounds)
        for field_name in (
            "max_translation_delta_m",
            "max_rotation_delta_rad",
            "translation_tolerance_m",
            "rotation_tolerance_rad",
        ):
            value = _finite(getattr(self, field_name), field_name=field_name)
            if value <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        for field_name in (
            "stable_measurement_count",
            "max_control_cycles",
            "physics_ticks_per_control_cycle",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": MUJOCO_AGENT_TCP_POSE_PARAMETER_PROFILE_ID_V1,
                "component_ids": (self.component_id,),
                "controlled_frame_id": "tool_tcp",
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "frame_graph_hash": self.frame_graph_hash,
                "tool_config_hash": self.tool_config_hash,
                "calibration_hash": self.calibration_hash,
                "workspace_bounds": self.workspace_bounds,
                "max_translation_delta_m": self.max_translation_delta_m,
                "max_rotation_delta_rad": self.max_rotation_delta_rad,
                "translation_tolerance_m": self.translation_tolerance_m,
                "rotation_tolerance_rad": self.rotation_tolerance_rad,
                "stable_measurement_count": self.stable_measurement_count,
                "max_control_cycles": self.max_control_cycles,
                "physics_ticks_per_control_cycle": (
                    self.physics_ticks_per_control_cycle
                ),
                "requested_duration_semantics": "not_supported",
            },
            domain="dexteleop.agent_skill.mujoco_tcp_pose_profile.v1",
        )


@dataclass(frozen=True)
class MujocoAgentTcpCardinalStepProfile:
    """Exact lower primitive catalog and frame binding for cardinal TCP steps."""

    component_id: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    primitives: tuple[TCPPoseDisplacementPrimitive, ...] = (
        CARDINAL_TCP_SHORT_STEP_PRIMITIVES
    )

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.component_id not in {"left_arm", "right_arm"}:
            raise ValueError("Cardinal-step component must be left_arm or right_arm.")
        primitives = tuple(self.primitives)
        if any(
            not isinstance(item, TCPPoseDisplacementPrimitive)
            for item in primitives
        ):
            raise TypeError("primitives must contain TCP displacement primitives.")
        directions = tuple(item.direction.value for item in primitives)
        if set(directions) != set(CARDINAL_TCP_STEP_DIRECTIONS_V1) or len(
            directions
        ) != len(set(directions)):
            raise ValueError(
                "Cardinal-step profile requires one primitive per direction."
            )
        object.__setattr__(
            self,
            "primitives",
            tuple(sorted(primitives, key=lambda item: item.direction.value)),
        )

    @property
    def primitive_catalog_hash(self) -> str:
        return canonical_content_hash(
            tuple(item.to_payload() for item in self.primitives),
            domain="dexteleop.robot_motion.tcp_displacement_catalog.v1",
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": (
                    MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1
                ),
                "component_id": self.component_id,
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "controlled_frame_id": "tool_tcp",
                "frame_graph_hash": self.frame_graph_hash,
                "tool_config_hash": self.tool_config_hash,
                "calibration_hash": self.calibration_hash,
                "primitive_catalog_hash": self.primitive_catalog_hash,
                "primitives": tuple(item.to_payload() for item in self.primitives),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
            },
            domain="dexteleop.agent_skill.mujoco_tcp_cardinal_step_profile.v1",
        )

    def primitive(self, direction: str) -> TCPPoseDisplacementPrimitive:
        selected = cardinal_tcp_short_step(direction)
        matching = tuple(
            item
            for item in self.primitives
            if item.semantic_content_hash == selected.semantic_content_hash
        )
        if len(matching) != 1:
            raise ValueError("Direction is absent from the exact primitive profile.")
        return matching[0]


@dataclass(frozen=True)
class MujocoAgentTcpPoseHistoryProfile:
    """Bounded disk/history/query policy; construction never starts capture."""

    max_samples: int = 16
    max_window_ms: int = 60_000
    max_result_payload_bytes: int = 128 * 1024
    retention_sec: float = 3_600.0
    max_records: int = 7_200
    max_record_payload_bytes: int = 16 * 1024
    memory_buffer_max_records: int = 64
    sampling_policy: str = "boundary_plus_uniform"
    capture_interval_sec: float = 1.0
    max_operation_duration_ms: int = 1_000

    def __post_init__(self) -> None:
        for field_name in (
            "max_samples",
            "max_window_ms",
            "max_result_payload_bytes",
            "max_records",
            "max_record_payload_bytes",
            "memory_buffer_max_records",
            "max_operation_duration_ms",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        retention = _finite(self.retention_sec, field_name="retention_sec")
        interval = _finite(
            self.capture_interval_sec,
            field_name="capture_interval_sec",
        )
        if retention <= 0.0 or interval <= 0.0:
            raise ValueError("History retention and capture interval must be positive.")
        if self.max_window_ms > round(retention * 1_000.0):
            raise ValueError("max_window_ms cannot exceed disk retention.")
        if self.max_record_payload_bytes > self.max_result_payload_bytes:
            raise ValueError(
                "max_record_payload_bytes cannot exceed the result payload bound."
            )
        if self.sampling_policy not in SOURCE_OBSERVATION_HISTORY_SAMPLING_POLICIES_V1:
            raise ValueError("Unsupported planning history sampling policy.")
        object.__setattr__(self, "retention_sec", retention)
        object.__setattr__(self, "capture_interval_sec", interval)

    @property
    def schema_limits(self) -> TcpPoseHistoryLimits:
        return TcpPoseHistoryLimits(
            max_samples=self.max_samples,
            max_window_ms=self.max_window_ms,
            max_payload_bytes=self.max_result_payload_bytes,
        )

    @property
    def store_limits(self) -> SourceObservationHistoryStoreLimits:
        return SourceObservationHistoryStoreLimits(
            retention_sec=self.retention_sec,
            max_records=self.max_records,
            max_record_payload_bytes=self.max_record_payload_bytes,
            memory_buffer_max_records=self.memory_buffer_max_records,
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                "profile_id": MUJOCO_AGENT_TCP_POSE_HISTORY_PROFILE_ID_V1,
                "component_ids": ("right_arm",),
                "controlled_frame_id": "tool_tcp",
                "max_samples": self.max_samples,
                "max_window_ms": self.max_window_ms,
                "max_result_payload_bytes": self.max_result_payload_bytes,
                "retention_sec": self.retention_sec,
                "max_records": self.max_records,
                "max_record_payload_bytes": self.max_record_payload_bytes,
                "memory_buffer_max_records": self.memory_buffer_max_records,
                "sampling_policy": self.sampling_policy,
                "capture_interval_sec": self.capture_interval_sec,
                "max_operation_duration_ms": self.max_operation_duration_ms,
            },
            domain="dexteleop.agent_state.mujoco_tcp_pose_history_profile.v1",
        )


@dataclass(frozen=True)
class MujocoAgentTcpPoseDeploymentBinding:
    deployment_binding_id: str
    deployment_binding_version: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    execution_domain_binding_hash: str
    target_runtime_action_contract_hash: str
    component_target_capability_hash: str
    action_schema_hash: str
    observation_binding_id: str
    observation_binding_hash: str
    observation_plan_hash: str
    observation_capability_hash: str
    pose_schema_hash: str
    motion_profile_hash: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    trace_requirement_profile_id: str
    executor_id: str
    explicit_exclusions: tuple[str, ...]
    component_id: str = "right_arm"
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...] = ()

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
            "execution_domain_binding_hash",
            "target_runtime_action_contract_hash",
            "component_target_capability_hash",
            "action_schema_hash",
            "observation_binding_id",
            "observation_binding_hash",
            "observation_plan_hash",
            "observation_capability_hash",
            "pose_schema_hash",
            "motion_profile_hash",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "trace_requirement_profile_id",
            "executor_id",
            "component_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.component_id not in {"left_arm", "right_arm"}:
            raise ValueError("component_id must be left_arm or right_arm.")
        exclusions = tuple(sorted(set(self.explicit_exclusions)))
        if not exclusions or any(not str(item).strip() for item in exclusions):
            raise ValueError("explicit_exclusions must contain non-empty values.")
        object.__setattr__(self, "explicit_exclusions", exclusions)
        safety_refs = tuple(self.safety_envelope_refs)
        if any(
            not isinstance(item, SkillDeploymentSafetyEnvelopeRef)
            for item in safety_refs
        ):
            raise TypeError("safety_envelope_refs must contain typed refs.")
        object.__setattr__(
            self,
            "safety_envelope_refs",
            tuple(sorted(safety_refs, key=lambda item: item.object_id)),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **{
                    field_name: getattr(self, field_name)
                    for field_name in self.__dataclass_fields__
                    if field_name not in {"component_id", "safety_envelope_refs"}
                },
                "safety_envelope_refs": tuple(
                    item.to_wire() for item in self.safety_envelope_refs
                ),
                "controlled_component_ids": (self.component_id,),
                "controlled_frame_id": "tool_tcp",
                "target_kind": "sim",
                "source_kind": "agent_skill",
            },
            domain="dexteleop.agent_skill.mujoco_tcp_pose_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="sim",
            execution_manifest_hash=self.execution_manifest_hash,
            safety_envelope_refs=self.safety_envelope_refs,
        )


@dataclass(frozen=True)
class MujocoPassiveTcpPoseHistoryDeploymentBinding:
    """Exact passive history Route binding; it owns no motion or source clock."""

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
    history_source_id: str
    history_source_contract_hash: str
    history_profile_hash: str
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
            "history_source_id",
            "history_source_contract_hash",
            "history_profile_hash",
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
            raise ValueError("History deployment must declare explicit exclusions.")
        object.__setattr__(self, "explicit_exclusions", exclusions)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **{
                    field_name: getattr(self, field_name)
                    for field_name in self.__dataclass_fields__
                },
                "controlled_component_ids": ("right_arm",),
                "controlled_frame_id": "tool_tcp",
                "target_kind": "sim",
                "read_semantics": "bounded_disk_primary_causal_history",
            },
            domain=MUJOCO_AGENT_TCP_POSE_HISTORY_BINDING_HASH_DOMAIN_V1,
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="sim",
            execution_manifest_hash=self.execution_manifest_hash,
        )


@dataclass(frozen=True)
class MujocoAgentTcpPoseContractBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    route_manifest: SkillRouteManifest
    deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    cardinal_step_deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    state_deployment_binding: MujocoPassiveTcpPoseStateDeploymentBinding
    motion_profile: MujocoAgentTcpPoseMotionProfile
    cardinal_step_profile: MujocoAgentTcpCardinalStepProfile
    evaluation_plan: SkillEvaluationPlan
    cardinal_step_evaluation_plan: SkillEvaluationPlan
    payload_validators: AgentSkillPayloadValidatorRegistry
    station_manifest: ControlStationRunManifest
    execution_manifest: Any
    history_profile: MujocoAgentTcpPoseHistoryProfile | None = None
    history_source_binding: SourceObservationHistoryBinding | None = None
    history_deployment_binding: (
        MujocoPassiveTcpPoseHistoryDeploymentBinding | None
    ) = None

    def __post_init__(self) -> None:
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
        )
        motion_entry = self.route_manifest.route(
            "motion.move_tcp_to_pose",
            "1",
        )
        cardinal_step_entry = self.route_manifest.route(
            "motion.step_tcp_cardinal",
            "1",
        )
        state_entry = self.route_manifest.route("state.read_tcp_pose", "1")
        if motion_entry.deployment_binding.deployment_binding_hash != (
            self.deployment_binding.semantic_content_hash
        ):
            raise ValueError("Motion Route uses another deployment binding.")
        if state_entry.deployment_binding.deployment_binding_hash != (
            self.state_deployment_binding.semantic_content_hash
        ):
            raise ValueError("State Route uses another deployment binding.")
        if cardinal_step_entry.deployment_binding.deployment_binding_hash != (
            self.cardinal_step_deployment_binding.semantic_content_hash
        ):
            raise ValueError("Cardinal-step Route uses another deployment binding.")
        if self.deployment_binding.component_id != self.motion_profile.component_id:
            raise ValueError("Motion deployment and profile component mismatch.")
        if (
            self.cardinal_step_deployment_binding.component_id
            != self.cardinal_step_profile.component_id
        ):
            raise ValueError(
                "Cardinal-step deployment and profile component mismatch."
            )
        if motion_entry.evaluation_plan_hash != (
            self.evaluation_plan.semantic_content_hash
        ):
            raise ValueError("Motion Route uses another evaluation plan.")
        if cardinal_step_entry.evaluation_plan_hash != (
            self.cardinal_step_evaluation_plan.semantic_content_hash
        ):
            raise ValueError("Cardinal-step Route uses another evaluation plan.")
        if self.cardinal_step_profile.reference_frame_id != (
            self.motion_profile.reference_frame_id
        ):
            raise ValueError("Cardinal-step and motion profiles use another frame.")
        history_parts = (
            self.history_profile,
            self.history_source_binding,
            self.history_deployment_binding,
        )
        history_routes = tuple(
            entry
            for entry in self.route_manifest.skill_routes
            if entry.route.skill_id == "state.read_tcp_pose_history"
        )
        if all(item is None for item in history_parts):
            if history_routes:
                raise ValueError("History Route requires the complete history binding.")
        elif any(item is None for item in history_parts):
            raise ValueError("History profile, source, and deployment are all-or-none.")
        else:
            if not isinstance(
                self.history_profile,
                MujocoAgentTcpPoseHistoryProfile,
            ):
                raise TypeError("history_profile must use the typed history profile.")
            if not isinstance(
                self.history_source_binding,
                SourceObservationHistoryBinding,
            ):
                raise TypeError(
                    "history_source_binding must use the typed source binding."
                )
            if not isinstance(
                self.history_deployment_binding,
                MujocoPassiveTcpPoseHistoryDeploymentBinding,
            ):
                raise TypeError(
                    "history_deployment_binding must use the typed deployment binding."
                )
            if len(history_routes) != 1:
                raise ValueError("Enabled history requires exactly one history Route.")
            history_entry = history_routes[0]
            if history_entry.deployment_binding.deployment_binding_hash != (
                self.history_deployment_binding.semantic_content_hash
            ):
                raise ValueError("History Route uses another deployment binding.")
            if self.history_source_binding.semantic_content_hash != (
                self.history_deployment_binding.history_source_contract_hash
            ):
                raise ValueError("History deployment uses another source contract.")
            if self.history_profile.semantic_content_hash != (
                self.history_deployment_binding.history_profile_hash
            ):
                raise ValueError("History deployment uses another profile.")
            if self.history_source_binding.capture_profile_hash != (
                self.history_profile.semantic_content_hash
            ):
                raise ValueError("History source uses another capture profile.")
        self.payload_validators.validate_manifest(self.route_manifest)


def _build_station_manifests(application: Any) -> tuple[Any, Any]:
    runtime = application.runtime
    config = runtime.config
    direct = config.direct_component_proposal_source
    if (
        direct is None
        or direct.producer_kind != "agent_skill"
        or config.enable_primary_policy_authority
        or config.enable_composer_authority
        or config.enable_replay_authority
        or config.initial_control_mode != "agent_skill"
        or tuple(runtime.action_schema.component_ids) != ("right_arm",)
    ):
        raise ValueError("Runtime is not the exact no-model Agent motion profile.")
    target = ExecutionDomainBinding(
        binding_id=f"mujoco-agent-target:{config.control_domain_id}",
        target_kind="sim",
        control_domain_id=config.control_domain_id,
        embodiment_instance_id=config.embodiment_instance_id,
        target_runtime_action_contract_hash=(
            runtime.target_runtime_action_contract.semantic_content_hash
        ),
        execution_capability_profile_hash=(
            runtime.execution_capability.semantic_content_hash
        ),
        component_catalog_hash=runtime.catalog.semantic_content_hash,
        selection_group_ids=runtime.authoritative_port.profile.selection_group_ids,
        owner_registry_id=authoritative_owner_registry_id(runtime),
        authority_epoch_source_id=authoritative_authority_epoch_source_id(runtime),
        execution_epoch_source_id=authoritative_execution_epoch_source_id(runtime),
        trace_requirement_profile_id=MUJOCO_AGENT_TCP_POSE_TRACE_PROFILE_ID_V1,
        explicit_exclusions=(
            "hardware_device_initialization",
            "hand_control",
            "model_inference",
            "real_backend",
            "second_dispatch_owner",
        ),
    )
    source_contract_hash = canonical_content_hash(
        {
            "source_kind": "agent_skill",
            "producer_id": direct.producer_id,
            "producer_epoch": direct.producer_epoch,
            "action_schema_hash": runtime.action_schema.semantic_content_hash,
            "submission_seam": (
                "sim.mujoco.deploy.MujocoDeployRuntime.submit_component_action"
            ),
        },
        domain="dexteleop.control_station.agent_skill_component_source.v1",
    )
    epoch_policy_hash = canonical_content_hash(
        {
            "producer_id": direct.producer_id,
            "producer_epoch": direct.producer_epoch,
            "activation": "fixed_on_runtime_start",
            "reuse": "sequential_operations_same_session",
        },
        domain="dexteleop.control_station.agent_skill_producer_epoch.v1",
    )
    station = ControlStationRunManifest(
        manifest_id="control_station_mujoco_agent_tcp_pose_v1",
        manifest_version="1",
        execution_binding=target,
        source_slots=(
            SourceSlotSpec(
                source_slot_id=f"agent-skill-slot:{direct.producer_id}",
                source_kind="agent_skill",
                producer_id=direct.producer_id,
                source_action_contract_hash=source_contract_hash,
                producer_epoch_policy_hash=epoch_policy_hash,
                supported_uses=("authority", "candidate"),
                expected_emission_seam=(
                    "sim.mujoco.deploy.MujocoDeployRuntime.submit_component_action"
                ),
                required_binding_refs=(
                    runtime.action_schema.semantic_content_hash,
                    runtime.component_target_capability.semantic_content_hash,
                    target.semantic_content_hash,
                ),
                trace_requirement_refs=(
                    "action_proposal",
                    "proposal_admission",
                    "arbitration_decision",
                    "safety_decision",
                    "target_dispatch",
                    "measurement_alignment",
                ),
                explicit_exclusions=(
                    "agent_selected_producer_identity",
                    "hand_component",
                    "policy_lineage",
                    "raw_action_stream",
                    "second_dispatch_owner",
                ),
            ),
        ),
        authority_policy_hash=runtime.authoritative_port.profile.authority_policy_hash,
        decision_schedule_hash=runtime.authoritative_port.profile.decision_schedule_hash,
        trace_requirement_profile_id=MUJOCO_AGENT_TCP_POSE_TRACE_PROFILE_ID_V1,
        capture_policy_id="agent_tcp_pose_lower_trace_owner_v1",
        allowed_intent_kinds=(
            "connect_target",
            "hold",
            "run_preflight",
            "stop_runtime",
        ),
        explicit_exclusions=(
            "agent_estop",
            "agent_operator_lease",
            "agent_rearm",
            "agent_resume",
            "hardware_device_initialization",
            "hand_control",
            "model_inference",
            "real_motion",
            "second_dispatch_owner",
        ),
    )
    exclusions = tuple(
        ExecutionProfileExclusion.create(
            capability_id=capability_id,
            reason="agent_tcp_pose_no_model_sim",
            evidence_ids=(station.semantic_content_hash,),
        )
        for capability_id in (
            "hand_control",
            "hardware_device_initialization",
            "human_motion_source",
            "model_inference",
            "real_backend",
            "replay_authority",
            "second_dispatch_owner",
        )
    )
    execution = build_control_execution_manifest(
        manifest_id="mujoco_agent_tcp_pose_no_model_v1",
        manifest_version="v1",
        action_space=runtime.catalog.action_space,
        action_schema=runtime.action_schema,
        component_target_capability=runtime.component_target_capability,
        catalog=runtime.catalog,
        motion_authority_sources=(f"agent_skill:{direct.producer_id}",),
        human_motion_proposal_status="not_applicable",
        safety_control_sources=("runtime_safety_supervisor",),
        safety_control_requests=("hold",),
        lifecycle_control_sources=("agent_skill_operation_executor",),
        lifecycle_control_requests=("hold", "start", "stop"),
        capability=runtime.execution_capability,
        activation=runtime.reference_activation_spec,
        actual_reference_frame_producer="mujoco_pose_to_qpos_resolver",
        exclusions=exclusions,
    )
    return station, execution


def _exact_application(application: Any) -> dict[str, str]:
    runtime = application.runtime
    runtime_config = runtime.config
    observation_plan = runtime_config.execution_observation_plan
    observation_binding = runtime.execution_observation_binding
    observation_capability = runtime.execution_observation_capability
    if (
        not bool(getattr(observation_plan, "enabled", False))
        or getattr(observation_plan, "measurement_mode", None) == "off"
        or tuple(getattr(observation_plan, "component_ids", ()))
        != ("right_arm",)
        or dict(getattr(observation_plan, "required_representation_by_component", {}))
        != {"right_arm": "cartesian_pose"}
        or getattr(observation_binding, "plan_hash", None)
        != getattr(observation_plan, "semantic_content_hash", None)
        or dict(getattr(observation_capability, "representation_by_component", {}))
        != {"right_arm": "cartesian_pose"}
    ):
        raise ValueError("Agent motion requires exact right-arm TCP observations.")
    target_profile = application.target_frame_profile
    frame_graph = target_profile.frame_graph
    reference_frame_id = target_profile.reference_frame_by_component["right_arm"]
    frame_spec = frame_graph.frame_by_id[reference_frame_id]
    tool_profile = application.tool_frame_profile
    if tool_profile.binding_by_component["right_arm"].controlled_frame_id != (
        "tool_tcp"
    ):
        raise ValueError("Agent motion requires the right-arm tool_tcp binding.")
    pose_schemas = dict(observation_capability.schema_hash_by_component)
    return {
        "runtime_session_id": _required(
            runtime_config.runtime_session_id,
            field_name="runtime_session_id",
        ),
        "control_domain_id": _required(
            runtime_config.control_domain_id,
            field_name="control_domain_id",
        ),
        "clock_domain_id": _required(
            runtime_config.clock_domain_id,
            field_name="clock_domain_id",
        ),
        "observation_binding_id": _required(
            observation_binding.binding_id,
            field_name="observation_binding_id",
        ),
        "observation_binding_hash": _hash(
            observation_binding,
            field_name="observation_binding_hash",
        ),
        "observation_plan_hash": _hash(
            observation_plan,
            field_name="observation_plan_hash",
        ),
        "observation_capability_hash": _hash(
            observation_capability,
            field_name="observation_capability_hash",
        ),
        "pose_schema_hash": _required(
            pose_schemas["right_arm"],
            field_name="pose_schema_hash",
        ),
        "reference_frame_id": _required(
            reference_frame_id,
            field_name="reference_frame_id",
        ),
        "reference_frame_spec_hash": _hash(
            frame_spec,
            field_name="reference_frame_spec_hash",
        ),
        "frame_graph_hash": _hash(frame_graph, field_name="frame_graph_hash"),
        "tool_config_hash": _hash(
            tool_profile,
            field_name="tool_config_hash",
        ),
        "calibration_hash": _required(
            application.calibration_hash,
            field_name="calibration_hash",
        ),
    }


def _motion_validators(
    *,
    route: SkillExecutionRoute,
    schema_entry: Any,
    profile: MujocoAgentTcpPoseMotionProfile,
) -> AgentSkillPayloadValidatorSet:
    parameter_fields = {
        "pose_targets_by_component",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "controlled_frame_id",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
    }

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError("TCP pose target must exactly match its profile schema.")
        validate_agent_motion_parameters_v1(
            schema_kind="tcp_pose_target",
            parameters=payload,
            allowed_component_ids=("right_arm",),
        )
        poses = payload["pose_targets_by_component"]
        if not isinstance(poses, Mapping) or set(poses) != {"right_arm"}:
            raise ValueError("TCP pose target must exactly cover right_arm.")
        exact = {
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
        }
        for field_name, expected in exact.items():
            if payload[field_name] != expected:
                raise ValueError(f"TCP pose target {field_name} mismatch.")
        pose = tuple(float(item) for item in poses["right_arm"])
        bounds = profile.workspace_bounds
        if not (
            bounds[0] <= pose[0] <= bounds[1]
            and bounds[2] <= pose[1] <= bounds[3]
            and bounds[4] <= pose[2] <= bounds[5]
        ):
            raise ValueError("TCP pose target is outside the profile workspace.")

    def validate_execution_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "execution_receipt_refs",
            "final_measured_state_hash",
        }:
            raise ValueError("TCP pose execution result schema mismatch.")
        refs = payload["execution_receipt_refs"]
        if (
            not isinstance(refs, (tuple, list))
            or not refs
            or len(refs) != len(set(refs))
            or any(not isinstance(item, str) or not item for item in refs)
        ):
            raise ValueError("execution_receipt_refs must be non-empty and unique.")
        _required(
            payload["final_measured_state_hash"],
            field_name="final_measured_state_hash",
        )

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "aggregate_evaluation_receipt_ref",
            "task_outcome",
        }:
            raise ValueError("TCP pose task evaluation schema mismatch.")
        _required(
            payload["aggregate_evaluation_receipt_ref"],
            field_name="aggregate_evaluation_receipt_ref",
        )
        if payload["task_outcome"] not in {
            "failed",
            "inconclusive",
            "succeeded",
        }:
            raise ValueError("Unsupported TCP pose task_outcome.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:mujoco-agent-tcp-pose:"
            + profile.semantic_content_hash
        ),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters,
        validate_execution_result=validate_execution_result,
        validate_task_evaluation=validate_task_evaluation,
    )


def _cardinal_step_validators(
    *,
    route: SkillExecutionRoute,
    profile: MujocoAgentTcpCardinalStepProfile,
) -> AgentSkillPayloadValidatorSet:
    parameter_fields = {
        "component_id",
        "direction",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "controlled_frame_id",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
        "primitive_catalog_hash",
    }

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError(
                "Cardinal TCP step must exactly match its profile schema."
            )
        validate_agent_motion_parameters_v1(
            schema_kind="cardinal_tcp_step",
            parameters=payload,
            allowed_component_ids=(profile.component_id,),
            allowed_cardinal_directions=tuple(
                item.direction.value for item in profile.primitives
            ),
        )
        exact = {
            "component_id": profile.component_id,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "primitive_catalog_hash": profile.primitive_catalog_hash,
        }
        for field_name, expected in exact.items():
            if payload[field_name] != expected:
                raise ValueError(f"Cardinal TCP step {field_name} mismatch.")
        profile.primitive(str(payload["direction"]))

    def validate_execution_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "execution_receipt_refs",
            "final_measured_state_hash",
        }:
            raise ValueError("Cardinal TCP step result schema mismatch.")
        refs = payload["execution_receipt_refs"]
        if (
            not isinstance(refs, (tuple, list))
            or not refs
            or len(refs) != len(set(refs))
            or any(not isinstance(item, str) or not item for item in refs)
        ):
            raise ValueError("execution_receipt_refs must be non-empty and unique.")
        _required(
            payload["final_measured_state_hash"],
            field_name="final_measured_state_hash",
        )

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "aggregate_evaluation_receipt_ref",
            "task_outcome",
        }:
            raise ValueError("Cardinal TCP step task evaluation schema mismatch.")
        _required(
            payload["aggregate_evaluation_receipt_ref"],
            field_name="aggregate_evaluation_receipt_ref",
        )
        if payload["task_outcome"] not in {
            "failed",
            "inconclusive",
            "succeeded",
        }:
            raise ValueError("Unsupported cardinal TCP step task_outcome.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:mujoco-agent-tcp-cardinal-step:"
            + profile.semantic_content_hash
        ),
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters,
        validate_execution_result=validate_execution_result,
        validate_task_evaluation=validate_task_evaluation,
    )


def _state_validators(
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
            raise ValueError("TCP pose query parameter schema mismatch.")
        validate_agent_state_parameters_v1(
            schema_kind="tcp_pose",
            parameters=payload,
            allowed_component_ids=("right_arm",),
        )
        if tuple(payload["component_ids"]) != ("right_arm",):
            raise ValueError("TCP pose query must exactly select right_arm.")
        if payload["reference_frame_id"] != exact["reference_frame_id"]:
            raise ValueError("TCP pose query reference frame mismatch.")
        age = payload["max_measurement_age_ms"]
        if (
            isinstance(age, bool)
            or not isinstance(age, int)
            or age > MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1
        ):
            raise ValueError("TCP pose query exceeds the age bound.")

    def validate_result_for(
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

    def validate_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != result_fields:
            raise ValueError("TCP pose result schema mismatch.")
        validate_result_for(
            {
                "component_ids": ["right_arm"],
                "reference_frame_id": exact["reference_frame_id"],
                "controlled_frame_id": "tool_tcp",
                "max_measurement_age_ms": (
                    MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1
                ),
            },
            payload,
        )
        expected = {
            "reference_frame_spec_hash": exact["reference_frame_spec_hash"],
            "frame_graph_hash": exact["frame_graph_hash"],
            "tool_config_hash": exact["tool_config_hash"],
            "calibration_hash": exact["calibration_hash"],
            "observation_binding_id": exact["observation_binding_id"],
            "observation_plan_hash": exact["observation_plan_hash"],
        }
        for field_name, value in expected.items():
            if payload[field_name] != value:
                raise ValueError(f"TCP pose result {field_name} mismatch.")
        if payload["pose_schema_hash_by_component"] != {
            "right_arm": exact["pose_schema_hash"]
        }:
            raise ValueError("TCP pose result pose schema mismatch.")

    def validate_task(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or payload:
            raise ValueError("Passive TCP pose reads have no task evaluation.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:mujoco-agent-passive-tcp-pose:"
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
        validate_task_evaluation=validate_task,
        validate_execution_result_for_parameters=validate_result_for,
    )


def _history_validators(
    *,
    route: SkillExecutionRoute,
    schema_entry: Any,
    exact: Mapping[str, str],
    profile: MujocoAgentTcpPoseHistoryProfile,
    source_binding: SourceObservationHistoryBinding,
) -> AgentSkillPayloadValidatorSet:
    parameter_fields = {
        "component_ids",
        "reference_frame_id",
        "controlled_frame_id",
        "history_window_ms",
        "max_samples",
    }
    result_fields = set(schema_entry.execution_result_schema["properties"])

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError("TCP pose history query parameter schema mismatch.")
        validate_agent_state_parameters_v1(
            schema_kind="tcp_pose_history",
            parameters=payload,
            allowed_component_ids=("right_arm",),
            limits=profile.schema_limits,
        )
        if tuple(payload["component_ids"]) != ("right_arm",):
            raise ValueError("TCP pose history query must exactly select right_arm.")
        if payload["reference_frame_id"] != exact["reference_frame_id"]:
            raise ValueError("TCP pose history query reference frame mismatch.")

    def validate_result_for(
        parameters: Mapping[str, Any],
        payload: Mapping[str, Any],
    ) -> None:
        validate_parameters(parameters)
        validate_agent_state_result_v1(
            schema_kind="tcp_pose_history",
            parameters=parameters,
            result=payload,
            allowed_component_ids=("right_arm",),
            limits=profile.schema_limits,
        )
        expected = {
            "history_source_id": source_binding.source_id,
            "history_source_contract_hash": source_binding.semantic_content_hash,
            "reference_frame_spec_hash": exact["reference_frame_spec_hash"],
            "frame_graph_hash": exact["frame_graph_hash"],
            "tool_config_hash": exact["tool_config_hash"],
            "calibration_hash": exact["calibration_hash"],
        }
        for field_name, value in expected.items():
            if payload[field_name] != value:
                raise ValueError(f"TCP pose history result {field_name} mismatch.")
        for sample in payload["samples"]:
            if sample["pose_schema_hash_by_component"] != {
                "right_arm": exact["pose_schema_hash"]
            }:
                raise ValueError("TCP pose history sample pose schema mismatch.")
            if sample["observation_binding_id"] != exact["observation_binding_id"]:
                raise ValueError("TCP pose history observation binding mismatch.")
            if sample["observation_plan_hash"] != exact["observation_plan_hash"]:
                raise ValueError("TCP pose history observation plan mismatch.")

    def validate_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != result_fields:
            raise ValueError("TCP pose history result schema mismatch.")
        validate_result_for(
            {
                "component_ids": ["right_arm"],
                "reference_frame_id": exact["reference_frame_id"],
                "controlled_frame_id": "tool_tcp",
                "history_window_ms": profile.max_window_ms,
                "max_samples": profile.max_samples,
            },
            payload,
        )

    def validate_task(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or payload:
            raise ValueError("Passive TCP pose history reads have no task evaluation.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:mujoco-agent-passive-tcp-pose-history:"
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
        validate_task_evaluation=validate_task,
        validate_execution_result_for_parameters=validate_result_for,
    )


def build_mujoco_agent_tcp_pose_contract_bundle(
    *,
    application: Any,
    history_profile: MujocoAgentTcpPoseHistoryProfile | None = None,
) -> MujocoAgentTcpPoseContractBundle:
    """Project one already-constructed no-model application without starting it."""

    if history_profile is not None and not isinstance(
        history_profile,
        MujocoAgentTcpPoseHistoryProfile,
    ):
        raise TypeError("history_profile must be typed or None.")
    exact = _exact_application(application)
    station, execution = _build_station_manifests(application)
    runtime = application.runtime
    target_binding = station.execution_binding
    physics_hz = float(application.task_config.physics_hz)
    decision_hz = float(runtime.config.decision_hz)
    profile = MujocoAgentTcpPoseMotionProfile(
        reference_frame_id=exact["reference_frame_id"],
        reference_frame_spec_hash=exact["reference_frame_spec_hash"],
        frame_graph_hash=exact["frame_graph_hash"],
        tool_config_hash=exact["tool_config_hash"],
        calibration_hash=exact["calibration_hash"],
        physics_ticks_per_control_cycle=max(1, round(physics_hz / decision_hz)),
    )
    cardinal_step_profile = MujocoAgentTcpCardinalStepProfile(
        component_id="right_arm",
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
    )
    if any(
        item.distance_m > profile.max_translation_delta_m
        for item in cardinal_step_profile.primitives
    ):
        raise ValueError(
            "Cardinal TCP primitive exceeds the lower motion delta profile."
        )
    definition = SkillDefinition(
        skill_id="motion.move_tcp_to_pose",
        skill_version="1",
        skill_kind="motion",
        summary="Move right_arm tool_tcp to one bounded absolute pose proposal.",
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=("right_arm",),
        parameter_semantics_id="tcp_pose_target.absolute.xyz_quat_xyzw.v1",
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "measured_start_pose_valid.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )
    schema_entry = tcp_pose_target_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=MUJOCO_AGENT_TCP_POSE_PARAMETER_PROFILE_ID_V1,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=(profile.reference_frame_id,),
        max_requested_duration_ms=None,
        workspace_bounds_by_component={"right_arm": profile.workspace_bounds},
    )
    route = SkillExecutionRoute(
        skill_id=definition.skill_id,
        skill_version=definition.skill_version,
        skill_definition_hash=definition.semantic_content_hash,
        route_id=MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=schema_entry.parameter_schema_hash,
        execution_result_schema_hash=schema_entry.execution_result_schema_hash,
        task_evaluation_schema_hash=schema_entry.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema_entry.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=10_000,
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=True,
    )
    validate_agent_motion_route_schema_v1(route=route, schema_entry=schema_entry)
    cardinal_step_definition = SkillDefinition(
        skill_id="motion.step_tcp_cardinal",
        skill_version="1",
        skill_kind="motion",
        summary=(
            "Move one tool_tcp by the reviewed lower short-step primitive in "
            "a semantic workspace direction."
        ),
        intended_target_kind="target_agnostic",
        cross_target_deployment_policy="not_applicable",
        controlled_component_ids=("right_arm",),
        parameter_semantics_id="tcp_cardinal_short_step.direction.v1",
        execution_result_semantics_id="component_action_execution.v1",
        task_outcome_semantics_id="tcp_pose_arrival.v1",
        entry_predicate_ids=("route_ready.v1", "measured_start_pose_valid.v1"),
        task_outcome_predicate_ids=("tcp_pose_within_tolerance.v1",),
    )
    cardinal_step_schema = cardinal_tcp_step_schema_entry_v1(
        skill_definition_hash=cardinal_step_definition.semantic_content_hash,
        parameter_profile_id=(
            MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1
        ),
        parameter_profile_hash=cardinal_step_profile.semantic_content_hash,
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=(profile.reference_frame_id,),
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
        primitive_catalog_hash=cardinal_step_profile.primitive_catalog_hash,
        allowed_directions=tuple(
            item.direction.value for item in cardinal_step_profile.primitives
        ),
    )
    cardinal_step_route = SkillExecutionRoute(
        skill_id=cardinal_step_definition.skill_id,
        skill_version=cardinal_step_definition.skill_version,
        skill_definition_hash=(
            cardinal_step_definition.semantic_content_hash
        ),
        route_id=MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=cardinal_step_schema.parameter_schema_hash,
        execution_result_schema_hash=(
            cardinal_step_schema.execution_result_schema_hash
        ),
        task_evaluation_schema_hash=(
            cardinal_step_schema.task_evaluation_schema_hash
        ),
        allowed_parameter_profiles=cardinal_step_schema.parameter_profiles,
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=10_000,
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=True,
    )
    validate_agent_motion_route_schema_v1(
        route=cardinal_step_route,
        schema_entry=cardinal_step_schema,
    )
    safety_runtime = getattr(application, "tcp_pose_envelope_runtime", None)
    safety_envelope_refs = (
        ()
        if safety_runtime is None
        else tuple(
            SkillDeploymentSafetyEnvelopeRef(
                object_id=entry.envelope_id,
                schema="dexteleop.deployment_safety_envelope.v1",
                content_hash=entry.semantic_content_hash,
                constraint_set_hash=entry.constraint_set_hash,
                allowed_route_ids=entry.allowed_route_ids,
            )
            for entry in safety_runtime.catalog_entries
        )
    )
    binding = MujocoAgentTcpPoseDeploymentBinding(
        deployment_binding_id=(
            "sim.mujoco.agent_tcp_pose:" + exact["runtime_session_id"]
        ),
        deployment_binding_version="1",
        route_hash=route.semantic_content_hash,
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        target_binding_id=target_binding.binding_id,
        station_manifest_hash=station.semantic_content_hash,
        execution_manifest_hash=execution.semantic_content_hash,
        execution_domain_binding_hash=target_binding.semantic_content_hash,
        target_runtime_action_contract_hash=(
            target_binding.target_runtime_action_contract_hash
        ),
        component_target_capability_hash=(
            runtime.component_target_capability.semantic_content_hash
        ),
        action_schema_hash=runtime.action_schema.semantic_content_hash,
        observation_binding_id=exact["observation_binding_id"],
        observation_binding_hash=exact["observation_binding_hash"],
        observation_plan_hash=exact["observation_plan_hash"],
        observation_capability_hash=exact["observation_capability_hash"],
        pose_schema_hash=exact["pose_schema_hash"],
        motion_profile_hash=profile.semantic_content_hash,
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
        trace_requirement_profile_id=station.trace_requirement_profile_id,
        executor_id=MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1,
        explicit_exclusions=(
            "hand_control",
            "hardware_backend",
            "model_inference",
            "requested_duration",
            "second_dispatch_owner",
            "trajectory_streaming",
        ),
        safety_envelope_refs=safety_envelope_refs,
    )
    cardinal_step_binding = MujocoAgentTcpPoseDeploymentBinding(
        deployment_binding_id=(
            "sim.mujoco.agent_tcp_cardinal_step:" + exact["runtime_session_id"]
        ),
        deployment_binding_version="1",
        route_hash=cardinal_step_route.semantic_content_hash,
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        target_binding_id=target_binding.binding_id,
        station_manifest_hash=station.semantic_content_hash,
        execution_manifest_hash=execution.semantic_content_hash,
        execution_domain_binding_hash=target_binding.semantic_content_hash,
        target_runtime_action_contract_hash=(
            target_binding.target_runtime_action_contract_hash
        ),
        component_target_capability_hash=(
            runtime.component_target_capability.semantic_content_hash
        ),
        action_schema_hash=runtime.action_schema.semantic_content_hash,
        observation_binding_id=exact["observation_binding_id"],
        observation_binding_hash=exact["observation_binding_hash"],
        observation_plan_hash=exact["observation_plan_hash"],
        observation_capability_hash=exact["observation_capability_hash"],
        pose_schema_hash=exact["pose_schema_hash"],
        motion_profile_hash=profile.semantic_content_hash,
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
        trace_requirement_profile_id=station.trace_requirement_profile_id,
        executor_id=MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1,
        explicit_exclusions=(
            "agent_authored_distance",
            "hand_control",
            "hardware_backend",
            "model_inference",
            "second_dispatch_owner",
            "trajectory_streaming",
            "upper_target_derivation",
        ),
        safety_envelope_refs=safety_envelope_refs,
    )
    evaluator_input_hash = canonical_content_hash(
        {
            "input": "measured_tcp_pose_after_lower_hold",
            "motion_profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.tcp_pose_evaluator_input.v1",
    )
    aggregation_hash = canonical_content_hash(
        {"policy": "single_required_rule_based_evaluator"},
        domain="dexteleop.agent_skill.tcp_pose_aggregation.v1",
    )
    evaluator = SkillEvaluatorSpec(
        evaluator_id="motion.tcp_pose_arrival_rule",
        evaluator_version="1",
        evaluator_kind="rule_based",
        input_evidence_schema_hash=evaluator_input_hash,
        output_schema_hash=schema_entry.task_evaluation_schema_hash,
        role="required",
        execution_location="gateway_worker",
        deadline_ms=1_000,
        failure_policy="veto",
        evidence_domain_id=exact["control_domain_id"],
        configuration_hash=profile.semantic_content_hash,
    )
    evaluation_plan = SkillEvaluationPlan(
        evaluation_plan_id="motion.tcp_pose_arrival_rule",
        evaluation_plan_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        evaluator_specs=(evaluator,),
        aggregation_policy_id="motion.single_required_rule.v1",
        aggregation_policy_hash=aggregation_hash,
        required_evaluation_deadline_ms=1_000,
        missing_evidence_policy="fail",
        disagreement_policy="aggregation_policy",
        task_outcome_schema_hash=schema_entry.task_evaluation_schema_hash,
    )
    cardinal_evaluator = SkillEvaluatorSpec(
        evaluator_id="motion.tcp_cardinal_step_arrival_rule",
        evaluator_version="1",
        evaluator_kind="rule_based",
        input_evidence_schema_hash=evaluator_input_hash,
        output_schema_hash=cardinal_step_schema.task_evaluation_schema_hash,
        role="required",
        execution_location="gateway_worker",
        deadline_ms=1_000,
        failure_policy="veto",
        evidence_domain_id=exact["control_domain_id"],
        configuration_hash=canonical_content_hash(
            {
                "motion_profile_hash": profile.semantic_content_hash,
                "cardinal_step_profile_hash": (
                    cardinal_step_profile.semantic_content_hash
                ),
            },
            domain="dexteleop.agent_skill.tcp_cardinal_step_evaluator.v1",
        ),
    )
    cardinal_step_evaluation_plan = SkillEvaluationPlan(
        evaluation_plan_id="motion.tcp_cardinal_step_arrival_rule",
        evaluation_plan_version="1",
        skill_definition_hash=(
            cardinal_step_definition.semantic_content_hash
        ),
        route_hash=cardinal_step_route.semantic_content_hash,
        deployment_binding_hash=cardinal_step_binding.semantic_content_hash,
        evaluator_specs=(cardinal_evaluator,),
        aggregation_policy_id="motion.single_required_rule.v1",
        aggregation_policy_hash=aggregation_hash,
        required_evaluation_deadline_ms=1_000,
        missing_evidence_policy="fail",
        disagreement_policy="aggregation_policy",
        task_outcome_schema_hash=(
            cardinal_step_schema.task_evaluation_schema_hash
        ),
    )

    state_definition = tcp_pose_state_definition_v1(
        controlled_component_ids=("right_arm",)
    )
    state_profile_hash = canonical_content_hash(
        {
            "profile_id": MUJOCO_AGENT_TCP_POSE_STATE_PROFILE_ID_V1,
            "component_ids": ("right_arm",),
            "reference_frame_id": profile.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": (
                MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1
            ),
        },
        domain="dexteleop.agent_state.mujoco_agent_tcp_pose_profile.v1",
    )
    state_schema = tcp_pose_state_schema_entry_v1(
        skill_definition_hash=state_definition.semantic_content_hash,
        parameter_profile_id=MUJOCO_AGENT_TCP_POSE_STATE_PROFILE_ID_V1,
        parameter_profile_hash=state_profile_hash,
        allowed_component_ids=("right_arm",),
        allowed_reference_frame_ids=(profile.reference_frame_id,),
        max_measurement_age_ms=MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1,
    )
    state_route = SkillExecutionRoute(
        skill_id=state_definition.skill_id,
        skill_version=state_definition.skill_version,
        skill_definition_hash=state_definition.semantic_content_hash,
        route_id=MUJOCO_AGENT_TCP_POSE_STATE_ROUTE_ID_V1,
        route_version="1",
        parameter_schema_hash=state_schema.parameter_schema_hash,
        execution_result_schema_hash=state_schema.execution_result_schema_hash,
        task_evaluation_schema_hash=state_schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=state_schema.parameter_profiles,
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("dry_run", "sim_active"),
        target_kind="sim",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="cancel_only",
        requires_task_evaluation=False,
    )
    validate_agent_state_route_schema_v1(
        route=state_route,
        schema_entry=state_schema,
    )
    state_binding = MujocoPassiveTcpPoseStateDeploymentBinding(
        deployment_binding_id=(
            "sim.mujoco.agent_passive_tcp_pose:" + exact["runtime_session_id"]
        ),
        deployment_binding_version="1",
        route_hash=state_route.semantic_content_hash,
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        target_binding_id=target_binding.binding_id,
        station_manifest_hash=station.semantic_content_hash,
        execution_manifest_hash=execution.semantic_content_hash,
        observation_binding_id=exact["observation_binding_id"],
        observation_binding_hash=exact["observation_binding_hash"],
        observation_plan_hash=exact["observation_plan_hash"],
        observation_capability_hash=exact["observation_capability_hash"],
        pose_schema_hash=exact["pose_schema_hash"],
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
        executor_id=MUJOCO_AGENT_TCP_POSE_STATE_EXECUTOR_ID_V1,
        explicit_exclusions=(
            "control_authority",
            "device_start",
            "motion",
            "physics_step",
            "publication_open",
            "task_evaluation",
        ),
    )
    history_definition = None
    history_schema = None
    history_route = None
    history_source_binding = None
    history_binding = None
    history_availability = None
    if history_profile is not None:
        history_definition = tcp_pose_history_state_definition_v1(
            controlled_component_ids=("right_arm",)
        )
        history_schema = tcp_pose_history_state_schema_entry_v1(
            skill_definition_hash=history_definition.semantic_content_hash,
            parameter_profile_id=MUJOCO_AGENT_TCP_POSE_HISTORY_PROFILE_ID_V1,
            parameter_profile_hash=history_profile.semantic_content_hash,
            allowed_component_ids=("right_arm",),
            allowed_reference_frame_ids=(profile.reference_frame_id,),
            limits=history_profile.schema_limits,
        )
        history_route = SkillExecutionRoute(
            skill_id=history_definition.skill_id,
            skill_version=history_definition.skill_version,
            skill_definition_hash=history_definition.semantic_content_hash,
            route_id=MUJOCO_AGENT_TCP_POSE_HISTORY_ROUTE_ID_V1,
            route_version="1",
            parameter_schema_hash=history_schema.parameter_schema_hash,
            execution_result_schema_hash=(
                history_schema.execution_result_schema_hash
            ),
            task_evaluation_schema_hash=history_schema.task_evaluation_schema_hash,
            allowed_parameter_profiles=history_schema.parameter_profiles,
            control_effect_class="read_only",
            privilege_tier="t0_read_only",
            supported_modes=("dry_run", "sim_active"),
            target_kind="sim",
            may_move_robot=False,
            may_open_publication=False,
            requires_operator_confirmation=False,
            requires_deadman=False,
            max_operation_duration_ms=history_profile.max_operation_duration_ms,
            cancellation_policy="cancel_only",
            requires_task_evaluation=False,
        )
        validate_agent_state_route_schema_v1(
            route=history_route,
            schema_entry=history_schema,
        )
        history_source_binding = SourceObservationHistoryBinding(
            source_id=MUJOCO_AGENT_TCP_POSE_HISTORY_SOURCE_ID_V1,
            runtime_session_id=exact["runtime_session_id"],
            clock_domain_id=exact["clock_domain_id"],
            channels=("tcp_pose",),
            capture_profile_hash=history_profile.semantic_content_hash,
            source_contract={
                "contract_id": "dexteleop.planning_history.tcp_pose_source.v1",
                "component_ids": ["right_arm"],
                "channel": "tcp_pose",
                "representation_by_component": {
                    "right_arm": "cartesian_pose"
                },
                "pose_schema_hash_by_component": {
                    "right_arm": exact["pose_schema_hash"]
                },
                "reference_frame_id": exact["reference_frame_id"],
                "reference_frame_spec_hash": exact["reference_frame_spec_hash"],
                "controlled_frame_id": "tool_tcp",
                "frame_graph_hash": exact["frame_graph_hash"],
                "tool_config_hash": exact["tool_config_hash"],
                "calibration_hash": exact["calibration_hash"],
                "observation_binding_id": exact["observation_binding_id"],
                "observation_plan_hash": exact["observation_plan_hash"],
                "read_semantics": "application_locked_passive_cut",
            },
        )
        history_binding = MujocoPassiveTcpPoseHistoryDeploymentBinding(
            deployment_binding_id=(
                "sim.mujoco.agent_passive_tcp_pose_history:"
                + exact["runtime_session_id"]
            ),
            deployment_binding_version="1",
            route_hash=history_route.semantic_content_hash,
            runtime_session_id=exact["runtime_session_id"],
            control_domain_id=exact["control_domain_id"],
            target_binding_id=target_binding.binding_id,
            station_manifest_hash=station.semantic_content_hash,
            execution_manifest_hash=execution.semantic_content_hash,
            observation_binding_id=exact["observation_binding_id"],
            observation_binding_hash=exact["observation_binding_hash"],
            observation_plan_hash=exact["observation_plan_hash"],
            observation_capability_hash=exact["observation_capability_hash"],
            pose_schema_hash=exact["pose_schema_hash"],
            reference_frame_id=profile.reference_frame_id,
            reference_frame_spec_hash=profile.reference_frame_spec_hash,
            frame_graph_hash=profile.frame_graph_hash,
            tool_config_hash=profile.tool_config_hash,
            calibration_hash=profile.calibration_hash,
            history_source_id=history_source_binding.source_id,
            history_source_contract_hash=(
                history_source_binding.semantic_content_hash
            ),
            history_profile_hash=history_profile.semantic_content_hash,
            executor_id=MUJOCO_AGENT_TCP_POSE_HISTORY_EXECUTOR_ID_V1,
            explicit_exclusions=(
                "control_authority",
                "device_start",
                "implicit_recorder_start",
                "memory_only_history",
                "motion",
                "physics_step",
                "publication_open",
                "task_evaluation",
            ),
        )
        history_availability = canonical_content_hash(
            {
                "status": "configured_requires_exact_bound_store",
                "route_hash": history_route.semantic_content_hash,
                "deployment_binding_hash": history_binding.semantic_content_hash,
                "history_source_contract_hash": (
                    history_source_binding.semantic_content_hash
                ),
                "history_profile_hash": history_profile.semantic_content_hash,
            },
            domain="dexteleop.agent_state.mujoco_tcp_pose_history_availability.v1",
        )

    definition_items = [definition, cardinal_step_definition, state_definition]
    schema_items = [schema_entry, cardinal_step_schema, state_schema]
    if history_definition is not None and history_schema is not None:
        definition_items.append(history_definition)
        schema_items.append(history_schema)
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.agent_tcp_pose_session",
        catalog_version="1",
        skill_definitions=tuple(definition_items),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.agent_tcp_pose_session",
        catalog_version="1",
        schema_entries=tuple(schema_items),
    )
    motion_availability = canonical_content_hash(
        {
            "status": "configured_requires_activation_preflight",
            "route_hash": route.semantic_content_hash,
            "deployment_binding_hash": binding.semantic_content_hash,
            "motion_profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.agent_skill.mujoco_tcp_pose_availability.v1",
    )
    state_availability = canonical_content_hash(
        {
            "status": "configured_passive_read_ready",
            "route_hash": state_route.semantic_content_hash,
            "deployment_binding_hash": state_binding.semantic_content_hash,
            "observation_binding_hash": exact["observation_binding_hash"],
        },
        domain="dexteleop.agent_state.mujoco_agent_tcp_pose_availability.v1",
    )
    route_items = [
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=binding.to_ref(),
            evaluation_plan_hash=evaluation_plan.semantic_content_hash,
            availability_hash=motion_availability,
        ),
        SkillRouteManifestEntry(
            route=cardinal_step_route,
            deployment_binding=cardinal_step_binding.to_ref(),
            evaluation_plan_hash=(
                cardinal_step_evaluation_plan.semantic_content_hash
            ),
            availability_hash=canonical_content_hash(
                {
                    "status": "configured_requires_activation_preflight",
                    "route_hash": cardinal_step_route.semantic_content_hash,
                    "deployment_binding_hash": (
                        cardinal_step_binding.semantic_content_hash
                    ),
                    "cardinal_step_profile_hash": (
                        cardinal_step_profile.semantic_content_hash
                    ),
                },
                domain=(
                    "dexteleop.agent_skill.mujoco_tcp_cardinal_step_availability.v1"
                ),
            ),
        ),
        SkillRouteManifestEntry(
            route=state_route,
            deployment_binding=state_binding.to_ref(),
            evaluation_plan_hash=None,
            availability_hash=state_availability,
        ),
    ]
    if (
        history_route is not None
        and history_binding is not None
        and history_availability is not None
    ):
        route_items.append(
            SkillRouteManifestEntry(
                route=history_route,
                deployment_binding=history_binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=history_availability,
            )
        )
    manifest = SkillRouteManifest(
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        station_manifest_hash=station.semantic_content_hash,
        execution_manifest_hash=execution.semantic_content_hash,
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        target_binding_id=target_binding.binding_id,
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=(
            "own_domain_hold_v1",
            "own_operation_cancel_v1",
            "own_operation_status_v1",
            "idempotency_v1",
            "authoritative_port_receipt_v1",
            *(
                ("safety_profile_selection_v1",)
                if getattr(application, "tcp_pose_envelope_runtime", None)
                is not None
                else ()
            ),
        ),
        skill_routes=tuple(route_items),
        explicit_exclusions=(
            "agent_estop",
            "agent_rearm",
            "agent_resume",
            "hand_control",
            "hardware_backend",
            "model_inference",
            "raw_action_submission",
            "second_dispatch_owner",
        ),
    )
    validator_items = [
        _motion_validators(
            route=route,
            schema_entry=schema_entry,
            profile=profile,
        ),
        _cardinal_step_validators(
            route=cardinal_step_route,
            profile=cardinal_step_profile,
        ),
        _state_validators(
            route=state_route,
            schema_entry=state_schema,
            exact=exact,
        ),
    ]
    if (
        history_route is not None
        and history_schema is not None
        and history_profile is not None
        and history_source_binding is not None
    ):
        validator_items.append(
            _history_validators(
                route=history_route,
                schema_entry=history_schema,
                exact=exact,
                profile=history_profile,
                source_binding=history_source_binding,
            )
        )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id=(
            "dexteleop.agent_skill_payload_validators.agent_tcp_pose_session:"
            + exact["runtime_session_id"]
        ),
        validators=tuple(validator_items),
    )
    return MujocoAgentTcpPoseContractBundle(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
        deployment_binding=binding,
        cardinal_step_deployment_binding=cardinal_step_binding,
        state_deployment_binding=state_binding,
        motion_profile=profile,
        cardinal_step_profile=cardinal_step_profile,
        evaluation_plan=evaluation_plan,
        cardinal_step_evaluation_plan=cardinal_step_evaluation_plan,
        payload_validators=validators,
        station_manifest=station,
        execution_manifest=execution,
        history_profile=history_profile,
        history_source_binding=history_source_binding,
        history_deployment_binding=history_binding,
    )


__all__ = [
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_HISTORY_BINDING_HASH_DOMAIN_V1",
    "MUJOCO_AGENT_TCP_POSE_HISTORY_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_HISTORY_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_HISTORY_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_HISTORY_SOURCE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1",
    "MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1",
    "MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1",
    "MUJOCO_AGENT_TCP_POSE_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_STATE_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_STATE_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_STATE_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_POSE_WORKSPACE_BOUNDS_V1",
    "MujocoAgentTcpCardinalStepProfile",
    "MujocoAgentTcpPoseContractBundle",
    "MujocoAgentTcpPoseDeploymentBinding",
    "MujocoAgentTcpPoseHistoryProfile",
    "MujocoAgentTcpPoseMotionProfile",
    "MujocoPassiveTcpPoseHistoryDeploymentBinding",
    "build_mujoco_agent_tcp_pose_contract_bundle",
]

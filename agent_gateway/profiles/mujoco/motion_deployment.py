"""Exact no-model Sim Route and deployment for Agent TCP-pose motion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gateway.profiles.object_state import ObjectStateRegistration
    from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
    from agent_gateway.profiles.tcp_pose_comparison import TcpPoseComparisonRegistration

from agent_gateway.profiles.placement import (
    PlacementRegistration,
    build_placement_registration,
)
from agent_gateway.profiles.segmentation import (
    SegmentationRegistration,
    build_segmentation_registration,
)
from agent_gateway.protocol.evaluation_contracts import (
    SkillEvaluationPlan,
    SkillEvaluatorSpec,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    validate_skill_route_catalog_closure,
)
from agent_gateway.protocol.skill_io_contracts import (
    SkillIOCatalog,
    inline_only_skill_io_contract,
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
from agent_gateway.skills.gripper.deployment import GripperSkillContractBundle
from agent_gateway.skills.motion.definition import (
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    cardinal_tcp_step_skill_definition_v1,
    fixed_cardinal_tcp_step_skill_definition_v1,
    fixed_level_tcp_orientation_skill_definition_v1,
    grasp_approach_skill_definition_v1,
    grasp_contact_approach_skill_definition_v1,
    safe_tcp_pose_target_skill_definition_v1,
    tcp_pose_target_skill_definition_v1,
)
from agent_gateway.skills.motion.grasped_object import (
    GRASPED_OBJECT_MOVE_SKILL_ID,
    grasped_object_move_definition,
    grasped_object_move_schema,
    validate_grasped_object_move,
    validate_object_motion_state,
)
from agent_gateway.skills.motion.schemas import (
    cardinal_tcp_step_schema_entry_v1,
    fixed_cardinal_tcp_step_schema_entry_v1,
    fixed_level_tcp_orientation_schema_entry_v1,
    tcp_pose_target_schema_entry_v1,
    validate_safe_tcp_pose_goal_report_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)
from collection_runtime.source_observation_history import (
    SOURCE_OBSERVATION_HISTORY_SAMPLING_POLICIES_V1,
    SourceObservationHistoryBinding,
    SourceObservationHistoryStoreLimits,
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
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import level_tcp_orientation
from sim.mujoco.deploy.grasp_contact import MujocoGraspContactBinding

from .depth_deployment import (
    MujocoDepthCaptureRegistration,
    build_mujoco_depth_capture_registration_v2,
)
from .motion_primitive_profiles import (
    MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1,
    MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1,
    MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_DIRECTION_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_REFERENCE_STEP_RAD_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_ROTATION_DELTA_RAD_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_TRANSLATION_DELTA_M_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1,
    MUJOCO_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1,
    MUJOCO_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1,
    MujocoAgentTcpCardinalStepProfile,
    MujocoAgentTcpFixedOrientationProfile,
)
from .rgb_deployment import (
    MujocoRgbCaptureRegistration,
    build_mujoco_rgb_capture_registration_v2,
)
from .state_deployment import MujocoPassiveTcpPoseStateDeploymentBinding
from .visual_observation import mujoco_rgb_role_for_source

MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1 = "sim.mujoco.motion.move_tcp_to_pose_safe.v1"
MUJOCO_AGENT_SAFE_TCP_POSE_EXECUTOR_ID_V1 = "mujoco-agent-safe-tcp-pose-executor-v1"
MUJOCO_AGENT_GRASP_APPROACH_ROUTE_ID_V1 = "sim.mujoco.motion.approach_grasp_pose.v1"
MUJOCO_AGENT_GRASP_CONTACT_ROUTE_ID_V1 = "sim.mujoco.motion.approach_grasp_contact.v1"
MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V1 = "sim.mujoco.motion.move_grasped_object_to_pose.v1"
MUJOCO_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1 = "mujoco-agent-grasp-approach-executor-v1"

MUJOCO_AGENT_TCP_POSE_ROUTE_ID_V1 = "sim.mujoco.rm65.agent_skill_tcp_pose"
MUJOCO_AGENT_TCP_POSE_EXECUTOR_ID_V1 = (
    "control-station.mujoco-agent-tcp-pose.v1"
)
MUJOCO_AGENT_TCP_POSE_PARAMETER_PROFILE_ID_V1 = (
    "right-arm-bounded-tool-tcp-target-v1"
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
    execution_reference_frame_id: str | None = None
    execution_reference_frame_spec_hash: str | None = None
    profile_id: str = MUJOCO_AGENT_TCP_POSE_PARAMETER_PROFILE_ID_V1
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
    contact_policy_hash: str | None = None

    def __post_init__(self) -> None:
        if self.contact_policy_hash is not None:
            _required(self.contact_policy_hash, field_name="contact_policy_hash")
        for field_name in (
            "reference_frame_id",
            "reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "profile_id",
            "component_id",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        execution_reference_frame_id = (
            self.reference_frame_id
            if self.execution_reference_frame_id is None
            else _required(
                self.execution_reference_frame_id,
                field_name="execution_reference_frame_id",
            )
        )
        execution_reference_frame_spec_hash = (
            self.reference_frame_spec_hash
            if self.execution_reference_frame_spec_hash is None
            else _required(
                self.execution_reference_frame_spec_hash,
                field_name="execution_reference_frame_spec_hash",
            )
        )
        object.__setattr__(
            self,
            "execution_reference_frame_id",
            execution_reference_frame_id,
        )
        object.__setattr__(
            self,
            "execution_reference_frame_spec_hash",
            execution_reference_frame_spec_hash,
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
                "profile_id": self.profile_id,
                "component_ids": (self.component_id,),
                "controlled_frame_id": "tool_tcp",
                "reference_frame_id": self.reference_frame_id,
                "reference_frame_spec_hash": self.reference_frame_spec_hash,
                "execution_reference_frame_id": (
                    self.execution_reference_frame_id
                ),
                "execution_reference_frame_spec_hash": (
                    self.execution_reference_frame_spec_hash
                ),
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
                **({"contact_policy_hash": self.contact_policy_hash}
                   if self.contact_policy_hash is not None else {}),
            },
            domain="dexteleop.agent_skill.mujoco_tcp_pose_profile.v1",
        )


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
class MujocoAgentTcpFixedCardinalStepContract:
    skill_id: str
    direction: str
    deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    evaluation_plan: SkillEvaluationPlan

    def __post_init__(self) -> None:
        if self.skill_id not in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1:
            raise ValueError("skill_id is not a fixed cardinal TCP-step Skill.")
        if FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1[self.skill_id] != (
            self.direction
        ):
            raise ValueError("Fixed cardinal Skill and direction disagree.")
        if not isinstance(
            self.deployment_binding,
            MujocoAgentTcpPoseDeploymentBinding,
        ):
            raise TypeError("deployment_binding must be typed.")
        if not isinstance(self.evaluation_plan, SkillEvaluationPlan):
            raise TypeError("evaluation_plan must be typed.")


@dataclass(frozen=True)
class MujocoAgentTcpFixedOrientationContract:
    profile: MujocoAgentTcpFixedOrientationProfile
    motion_profile: MujocoAgentTcpPoseMotionProfile
    deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    evaluation_plan: SkillEvaluationPlan

    def __post_init__(self) -> None:
        if not isinstance(self.profile, MujocoAgentTcpFixedOrientationProfile):
            raise TypeError("profile must be a fixed-orientation profile.")
        if not isinstance(self.motion_profile, MujocoAgentTcpPoseMotionProfile):
            raise TypeError("motion_profile must be typed.")
        if not isinstance(
            self.deployment_binding,
            MujocoAgentTcpPoseDeploymentBinding,
        ):
            raise TypeError("deployment_binding must be typed.")
        if not isinstance(self.evaluation_plan, SkillEvaluationPlan):
            raise TypeError("evaluation_plan must be typed.")


@dataclass(frozen=True)
class MujocoAgentSafeTcpPoseContract:
    deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    evaluation_plan: SkillEvaluationPlan
    motion_profile: MujocoAgentTcpPoseMotionProfile
    path_search_time_budget_sec: float = 3.0
    contact_binding: MujocoGraspContactBinding | None = None

    def __post_init__(self):
        if self.contact_binding is not None:
            if not isinstance(self.contact_binding, MujocoGraspContactBinding):
                raise TypeError("contact_binding must be a lower target contact binding.")
            if self.contact_binding.semantic_content_hash != self.motion_profile.contact_policy_hash:
                raise ValueError("Contact binding differs from the admitted motion profile.")
        elif self.motion_profile.contact_policy_hash is not None:
            raise ValueError("Contact motion profile requires its exact lower binding.")


@dataclass(frozen=True)
class MujocoAgentTcpPoseContractBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    io_catalog: SkillIOCatalog
    route_manifest: SkillRouteManifest
    deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    cardinal_step_deployment_binding: MujocoAgentTcpPoseDeploymentBinding
    state_deployment_binding: MujocoPassiveTcpPoseStateDeploymentBinding
    motion_profile: MujocoAgentTcpPoseMotionProfile
    cardinal_step_profile: MujocoAgentTcpCardinalStepProfile
    evaluation_plan: SkillEvaluationPlan
    cardinal_step_evaluation_plan: SkillEvaluationPlan
    fixed_cardinal_step_contracts: tuple[
        MujocoAgentTcpFixedCardinalStepContract, ...
    ]
    fixed_orientation_contracts: tuple[
        MujocoAgentTcpFixedOrientationContract, ...
    ]
    payload_validators: AgentSkillPayloadValidatorRegistry
    station_manifest: ControlStationRunManifest
    execution_manifest: Any
    safe_tcp_pose_contract: MujocoAgentSafeTcpPoseContract | None = None
    depth_capture: MujocoDepthCaptureRegistration | None = None
    rgb_capture: MujocoRgbCaptureRegistration | None = None
    segmentation: SegmentationRegistration | None = None
    placement: PlacementRegistration | None = None
    history_profile: MujocoAgentTcpPoseHistoryProfile | None = None
    history_source_binding: SourceObservationHistoryBinding | None = None
    history_deployment_binding: (
        MujocoPassiveTcpPoseHistoryDeploymentBinding | None
    ) = None
    spatial_goal_registration: SpatialGoalRegistration | None = None
    tcp_pose_comparison_registration: TcpPoseComparisonRegistration | None = None
    object_state_registration: ObjectStateRegistration | None = None
    gripper_contract: GripperSkillContractBundle | None = None
    grasp_approach_contract: MujocoAgentSafeTcpPoseContract | None = None
    grasp_contact_contract: MujocoAgentSafeTcpPoseContract | None = None
    object_motion_contract: MujocoAgentSafeTcpPoseContract | None = None

    def __post_init__(self) -> None:
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
            io_catalog=self.io_catalog,
        )
        if self.spatial_goal_registration is not None:
            if self.route_manifest.route("perception.compute_spatial_goal", "1") != self.spatial_goal_registration.entry:
                raise ValueError("Spatial goal registration differs from its manifest entry.")
        if (self.tcp_pose_comparison_registration is not None
                and self.route_manifest.route("state.read_tcp_pose", "2") != self.tcp_pose_comparison_registration.entry):
            raise ValueError("TCP comparison registration differs from its manifest entry.")
        if (self.object_state_registration is not None
                and self.route_manifest.route("state.read_object_state", "1") != self.object_state_registration.entry):
            raise ValueError("Object-state registration differs from its manifest entry.")
        if self.gripper_contract is not None:
            hand = self.gripper_contract
            for name in ("runtime_session_id", "control_domain_id", "target_binding_id", "target_kind", "current_mode",
                         "station_manifest_hash", "execution_manifest_hash"):
                if getattr(hand.route_manifest, name) != getattr(self.route_manifest, name):
                    raise ValueError("Composed gripper and motion session identity mismatch.")
            entry = self.route_manifest.route("manipulation.set_gripper_state", "2")
            if (entry.route != hand.route_manifest.skill_routes[0].route
                    or entry.deployment_binding.deployment_binding_hash != hand.deployment_binding.semantic_content_hash):
                raise ValueError("Composed gripper Route changed.")
        for skill_id, safe in (
            ("motion.move_tcp_to_pose_safe", self.safe_tcp_pose_contract),
            ("motion.approach_grasp_pose", self.grasp_approach_contract),
            ("motion.approach_grasp_contact", self.grasp_contact_contract),
            (GRASPED_OBJECT_MOVE_SKILL_ID, self.object_motion_contract),
        ):
            if safe is None:
                continue
            entry = self.route_manifest.route(skill_id, "1")
            if (
                entry.deployment_binding.deployment_binding_hash
                != safe.deployment_binding.semantic_content_hash
                or entry.evaluation_plan_hash
                != safe.evaluation_plan.semantic_content_hash
                or safe.evaluation_plan.route_hash != entry.route.semantic_content_hash
                or safe.evaluation_plan.deployment_binding_hash
                != safe.deployment_binding.semantic_content_hash
                or safe.deployment_binding.motion_profile_hash
                != safe.motion_profile.semantic_content_hash
            ):
                raise ValueError("Safe TCP Route, deployment and evaluator disagree.")
        motion_entry = self.route_manifest.route(
            "motion.move_tcp_to_pose",
            "1",
        )
        cardinal_step_entry = self.route_manifest.route(
            "motion.step_tcp_cardinal",
            "1",
        )
        state_entry = self.route_manifest.route("state.read_tcp_pose", "1")
        depth_capture = self.depth_capture
        if depth_capture is not None:
            if not isinstance(depth_capture, MujocoDepthCaptureRegistration):
                raise TypeError("depth_capture must be a typed registration.")
            depth_entry = self.route_manifest.route(
                depth_capture.registration.definition.skill_id,
                depth_capture.registration.definition.skill_version,
            )
            if depth_entry.deployment_binding.deployment_binding_hash != (
                depth_capture.deployment_binding.semantic_content_hash
            ):
                raise ValueError("Depth Route uses another deployment binding.")
        rgb_capture = self.rgb_capture
        if rgb_capture is not None:
            if not isinstance(rgb_capture, MujocoRgbCaptureRegistration):
                raise TypeError("rgb_capture must be a typed registration.")
            rgb_entry = self.route_manifest.route(
                rgb_capture.registration.definition.skill_id,
                rgb_capture.registration.definition.skill_version,
            )
            if rgb_entry.deployment_binding.deployment_binding_hash != (
                rgb_capture.deployment_binding.semantic_content_hash
            ):
                raise ValueError("RGB Route uses another deployment binding.")
        segmentation = self.segmentation
        if segmentation is not None:
            if not isinstance(segmentation, SegmentationRegistration):
                raise TypeError("segmentation must be a typed registration.")
            segment_entry = self.route_manifest.route(
                segmentation.skill.definition.skill_id,
                segmentation.skill.definition.skill_version,
            )
            if segment_entry.deployment_binding.deployment_binding_hash != (
                segmentation.binding.semantic_content_hash
            ):
                raise ValueError("Segmentation Route uses another deployment binding.")
        placement = self.placement
        if placement is not None:
            if not isinstance(placement, PlacementRegistration):
                raise TypeError("placement must be a typed registration.")
            placement_entry = self.route_manifest.route(
                placement.skill.definition.skill_id,
                placement.skill.definition.skill_version,
            )
            if placement_entry.deployment_binding.deployment_binding_hash != (
                placement.binding.semantic_content_hash
            ):
                raise ValueError("Placement Route uses another deployment binding.")
            if segmentation is None or depth_capture is None:
                raise ValueError("Placement requires segmentation and depth capture.")
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
        if self.cardinal_step_profile.execution_reference_frame_id != (
            self.motion_profile.execution_reference_frame_id
        ):
            raise ValueError(
                "Cardinal-step execution frame and motion profile disagree."
            )
        fixed_cardinal_contracts = tuple(self.fixed_cardinal_step_contracts)
        if {
            item.skill_id for item in fixed_cardinal_contracts
        } != set(FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1) or len(
            fixed_cardinal_contracts
        ) != len(FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1):
            raise ValueError(
                "Bundle requires exactly six fixed cardinal-step contracts."
            )
        for item in fixed_cardinal_contracts:
            entry = self.route_manifest.route(item.skill_id, "1")
            if entry.deployment_binding.deployment_binding_hash != (
                item.deployment_binding.semantic_content_hash
            ):
                raise ValueError(
                    "Fixed cardinal-step Route uses another deployment binding."
                )
            if entry.evaluation_plan_hash != (
                item.evaluation_plan.semantic_content_hash
            ):
                raise ValueError(
                    "Fixed cardinal-step Route uses another evaluation plan."
                )
        orientation_contracts = tuple(self.fixed_orientation_contracts)
        if len(orientation_contracts) != len(
            FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
        ):
            raise ValueError(
                "Bundle requires exactly three fixed-orientation contracts."
            )
        if {
            item.profile.skill_id for item in orientation_contracts
        } != set(FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1):
            raise ValueError("Fixed-orientation contract Skill IDs are incomplete.")
        for item in orientation_contracts:
            entry = self.route_manifest.route(item.profile.skill_id, "1")
            if entry.deployment_binding.deployment_binding_hash != (
                item.deployment_binding.semantic_content_hash
            ):
                raise ValueError(
                    "Fixed-orientation Route uses another deployment binding."
                )
            if entry.evaluation_plan_hash != (
                item.evaluation_plan.semantic_content_hash
            ):
                raise ValueError(
                    "Fixed-orientation Route uses another evaluation plan."
                )
            if item.deployment_binding.motion_profile_hash != (
                item.motion_profile.semantic_content_hash
            ):
                raise ValueError(
                    "Fixed-orientation deployment uses another motion profile."
                )
            if item.profile.execution_reference_frame_id != (
                item.motion_profile.execution_reference_frame_id
            ):
                raise ValueError(
                    "Fixed-orientation execution frame and motion profile disagree."
                )
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


def _application_component_id(application: Any) -> str:
    components = tuple(application.runtime.action_schema.component_ids)
    if len(components) != 1 or components[0] not in {"left_arm", "right_arm"}:
        raise ValueError("Agent motion requires exactly one configured arm component.")
    return components[0]


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
        or tuple(runtime.action_schema.component_ids)
        not in {("left_arm",), ("right_arm",)}
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
    component = _application_component_id(application)
    runtime_config = runtime.config
    observation_plan = runtime_config.execution_observation_plan
    observation_binding = runtime.execution_observation_binding
    observation_capability = runtime.execution_observation_capability
    if (
        not bool(getattr(observation_plan, "enabled", False))
        or getattr(observation_plan, "measurement_mode", None) == "off"
        or tuple(getattr(observation_plan, "component_ids", ())) != (component,)
        or dict(getattr(observation_plan, "required_representation_by_component", {}))
        != {component: "cartesian_pose"}
        or getattr(observation_binding, "plan_hash", None)
        != getattr(observation_plan, "semantic_content_hash", None)
        or dict(getattr(observation_capability, "representation_by_component", {}))
        != {component: "cartesian_pose"}
    ):
        raise ValueError(
            f"Agent motion requires exact {component.replace('_', '-')} TCP observations."
        )
    target_profile = application.target_frame_profile
    frame_graph = target_profile.frame_graph
    reference_frame_id = target_profile.reference_frame_by_component[component]
    frame_spec = frame_graph.frame_by_id[reference_frame_id]
    primitive_reference_frame_id = target_profile.canonical_reference_frame_id
    primitive_frame_spec = frame_graph.frame_by_id[primitive_reference_frame_id]
    if primitive_frame_spec.semantic_parameters.get("axis_contract") != (
        STATION_RETARGET_FRAME_CONTRACT_V1
    ):
        raise ValueError(
            "Agent primitives require the canonical station axis contract."
        )
    source_frame_specs = tuple(
        item
        for item in frame_graph.frames
        if item.semantic_parameters.get("frame_family") == "simulation_world"
    )
    if len(source_frame_specs) > 1:
        raise ValueError(
            "Agent perception permits at most one simulation-world frame."
        )
    # Legacy frame graphs treated MuJoCo world and canonical Station as the
    # same identity.  Preserve that contract while new graphs expose the
    # explicit world-to-Station adapter.
    source_frame_spec = (
        source_frame_specs[0] if source_frame_specs else primitive_frame_spec
    )
    frame_graph.resolve(reference_frame_id, primitive_reference_frame_id)
    frame_graph.resolve(source_frame_spec.frame_id, primitive_reference_frame_id)
    tool_profile = application.tool_frame_profile
    if tool_profile.binding_by_component[component].controlled_frame_id != ("tool_tcp"):
        raise ValueError(
            f"Agent motion requires the {component.replace('_', '-')} tool_tcp binding."
        )
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
            pose_schemas[component],
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
        "primitive_reference_frame_id": _required(
            primitive_reference_frame_id,
            field_name="primitive_reference_frame_id",
        ),
        "primitive_reference_frame_spec_hash": _hash(
            primitive_frame_spec,
            field_name="primitive_reference_frame_spec_hash",
        ),
        "perception_source_frame_id": _required(
            source_frame_spec.frame_id,
            field_name="perception_source_frame_id",
        ),
        "perception_source_frame_spec_hash": _hash(
            source_frame_spec,
            field_name="perception_source_frame_spec_hash",
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
    object_id: str | None = None,
    object_clock_domain_id: str | None = None,
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
        if object_id is not None:
            validate_grasped_object_move(payload, profile=profile, object_id=object_id)
            return
        optional = {"pose_goal_policy"} if route.skill_id == "motion.move_tcp_to_pose_safe" else set()
        if not isinstance(payload, Mapping) or not parameter_fields <= set(payload) <= parameter_fields | optional:
            raise ValueError("TCP pose target must exactly match its profile schema.")
        validate_agent_motion_parameters_v1(
            schema_kind="tcp_pose_target",
            parameters=payload,
            allowed_component_ids=(profile.component_id,),
        )
        poses = payload["pose_targets_by_component"]
        if not isinstance(poses, Mapping) or set(poses) != {profile.component_id}:
            raise ValueError(
                f"TCP pose target must exactly cover {profile.component_id}."
            )
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
        pose = tuple(float(item) for item in poses[profile.component_id])
        bounds = profile.workspace_bounds
        if not (
            bounds[0] <= pose[0] <= bounds[1]
            and bounds[2] <= pose[1] <= bounds[3]
            and bounds[4] <= pose[2] <= bounds[5]
        ):
            raise ValueError("TCP pose target is outside the profile workspace.")

    def validate_execution_result(payload: Mapping[str, Any]) -> None:
        required = {"execution_receipt_refs", "final_measured_state_hash"}
        if object_id is not None:
            required.add("object_state")
        optional = {"pose_goal_report"} if route.skill_id == "motion.move_tcp_to_pose_safe" else set()
        if not isinstance(payload, Mapping) or not required <= set(payload) <= required | optional:
            raise ValueError("TCP pose execution result schema mismatch.")
        if "pose_goal_report" in payload:
            validate_safe_tcp_pose_goal_report_v1(payload["pose_goal_report"])
        if object_id is not None:
            validate_object_motion_state(payload["object_state"], profile=profile,
                object_id=object_id, observation_source="mujoco_native_physics",
                clock_domain_id=object_clock_domain_id)
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

    def validate_execution_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_execution_result(result)
        if result["object_state"]["phase"] != parameters["phase"]:
            raise ValueError("Object terminal phase differs from the admitted request phase.")

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
        validate_execution_result_for_parameters=(
            validate_execution_result_for_parameters if object_id is not None else None),
        validate_task_evaluation=validate_task_evaluation,
    )


def _cardinal_step_validators(
    *,
    route: SkillExecutionRoute,
    profile: MujocoAgentTcpCardinalStepProfile,
    allowed_directions: tuple[str, ...] | None = None,
) -> AgentSkillPayloadValidatorSet:
    directions = (
        tuple(item.direction.value for item in profile.primitives)
        if allowed_directions is None
        else tuple(allowed_directions)
    )
    if not directions or set(directions) - {
        item.direction.value for item in profile.primitives
    }:
        raise ValueError("Cardinal validator directions are outside the profile.")
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
            allowed_cardinal_directions=directions,
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
            + (
                ""
                if allowed_directions is None
                else ":" + route.skill_id
            )
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


def _fixed_orientation_validators(
    *,
    route: SkillExecutionRoute,
    profile: MujocoAgentTcpFixedOrientationProfile,
) -> AgentSkillPayloadValidatorSet:
    parameter_fields = {
        "component_id",
        "reference_frame_id",
        "reference_frame_spec_hash",
        "controlled_frame_id",
        "frame_graph_hash",
        "tool_config_hash",
        "calibration_hash",
        "primitive_hash",
    }

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != parameter_fields:
            raise ValueError(
                "Fixed TCP orientation must exactly match its profile schema."
            )
        validate_agent_motion_parameters_v1(
            schema_kind="fixed_level_tcp_orientation",
            parameters=payload,
            allowed_component_ids=(profile.component_id,),
        )
        exact = {
            "component_id": profile.component_id,
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
            "primitive_hash": profile.primitive.semantic_content_hash,
        }
        for field_name, expected in exact.items():
            if payload[field_name] != expected:
                raise ValueError(
                    f"Fixed TCP orientation {field_name} mismatch."
                )

    def validate_execution_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "execution_receipt_refs",
            "final_measured_state_hash",
        }:
            raise ValueError("Fixed TCP orientation result schema mismatch.")
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
            raise ValueError(
                "Fixed TCP orientation task evaluation schema mismatch."
            )
        _required(
            payload["aggregate_evaluation_receipt_ref"],
            field_name="aggregate_evaluation_receipt_ref",
        )
        if payload["task_outcome"] not in {
            "failed",
            "inconclusive",
            "succeeded",
        }:
            raise ValueError("Unsupported fixed TCP orientation task_outcome.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:mujoco-agent-tcp-fixed-orientation:"
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
    component_id: str = "right_arm",
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
            allowed_component_ids=(component_id,),
        )
        if tuple(payload["component_ids"]) != (component_id,):
            raise ValueError(f"TCP pose query must exactly select {component_id}.")
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
            allowed_component_ids=(component_id,),
        )

    def validate_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != result_fields:
            raise ValueError("TCP pose result schema mismatch.")
        validate_result_for(
            {
                "component_ids": [component_id],
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
            component_id: exact["pose_schema_hash"]
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
    enable_artifact_perception: bool = False,
    segmentation_model_profile_id: str | None = None,
    segmentation_model_profile_hash: str | None = None,
    segmentation_inference_implementation_hash: str | None = None,
    enable_safe_tcp_pose: bool = False,
    enable_spatial_goals: bool = False,
    enable_tcp_pose_comparison: bool = False,
    enable_object_state: bool = False,
    enable_object_follow_observation: bool = False,
    enable_object_motion: bool = False,
    enable_grasp_approach: bool = False,
    grasp_contact_binding: MujocoGraspContactBinding | None = None,
) -> MujocoAgentTcpPoseContractBundle:
    """Project one already-constructed no-model application without starting it."""

    if history_profile is not None and not isinstance(
        history_profile,
        MujocoAgentTcpPoseHistoryProfile,
    ):
        raise TypeError("history_profile must be typed or None.")
    if not isinstance(enable_artifact_perception, bool):
        raise TypeError("enable_artifact_perception must be boolean.")
    segmentation_identity = (
        segmentation_model_profile_id,
        segmentation_model_profile_hash,
        segmentation_inference_implementation_hash,
    )
    if any(item is not None for item in segmentation_identity) and not all(
        item is not None for item in segmentation_identity
    ):
        raise ValueError("MuJoCo segmentation model identity is all-or-none.")
    if segmentation_model_profile_id is not None and not enable_artifact_perception:
        raise ValueError("MuJoCo segmentation requires Artifact perception.")
    if not isinstance(enable_safe_tcp_pose, bool):
        raise TypeError("enable_safe_tcp_pose must be boolean.")
    if not isinstance(enable_spatial_goals, bool):
        raise TypeError("enable_spatial_goals must be boolean.")
    if not isinstance(enable_tcp_pose_comparison, bool):
        raise TypeError("enable_tcp_pose_comparison must be boolean.")
    if type(enable_object_state) is not bool:
        raise TypeError("enable_object_state must be boolean.")
    if type(enable_object_follow_observation) is not bool:
        raise TypeError("enable_object_follow_observation must be boolean.")
    if enable_object_follow_observation and not enable_object_state:
        raise ValueError("Object follow observation requires explicit object-state feedback.")
    if enable_object_state and grasp_contact_binding is None:
        raise ValueError("Object-state feedback requires an explicit bound contact target.")
    if type(enable_object_motion) is not bool:
        raise TypeError("enable_object_motion must be boolean.")
    if enable_object_motion and (not enable_object_follow_observation or not enable_grasp_approach
                                 or grasp_contact_binding is None):
        raise ValueError("Object motion requires explicit follow observation, grasp approach and contact target.")
    exact = _exact_application(application)
    component = _application_component_id(application)
    side = component.removesuffix("_arm")
    if side == "left" and (
        history_profile is not None
        or enable_spatial_goals
        or enable_grasp_approach
        or grasp_contact_binding is not None
        or any(item is not None for item in segmentation_identity)
    ):
        raise ValueError(
            "Left motion mode does not expose history, spatial, grasp or segmentation Routes."
        )
    if not isinstance(enable_grasp_approach, bool):
        raise TypeError("enable_grasp_approach must be boolean.")
    if (
        enable_safe_tcp_pose or enable_grasp_approach
    ) and application.target.planned_tcp_motion is None:
        raise ValueError("Safe TCP Route requires the exact lower planned-motion adapter.")
    station, execution = _build_station_manifests(application)
    runtime = application.runtime
    target_binding = station.execution_binding
    physics_hz = float(application.task_config.physics_hz)
    decision_hz = float(runtime.config.decision_hz)
    coupled_ik = getattr(
        getattr(runtime, "pose_resolver", None),
        "ik",
        None,
    )
    orientation_reference_hz = float(
        getattr(
            coupled_ik,
            "reference_hz",
            decision_hz,
        )
    )
    profile = MujocoAgentTcpPoseMotionProfile(
        component_id=component,
        profile_id=f"{side}-arm-bounded-tool-tcp-target-v1",
        reference_frame_id=exact["primitive_reference_frame_id"],
        reference_frame_spec_hash=exact["primitive_reference_frame_spec_hash"],
        execution_reference_frame_id=exact["reference_frame_id"],
        execution_reference_frame_spec_hash=exact["reference_frame_spec_hash"],
        frame_graph_hash=exact["frame_graph_hash"],
        tool_config_hash=exact["tool_config_hash"],
        calibration_hash=exact["calibration_hash"],
        physics_ticks_per_control_cycle=max(1, round(physics_hz / decision_hz)),
    )
    cardinal_step_profile = MujocoAgentTcpCardinalStepProfile(
        component_id=component,
        reference_frame_id=exact["primitive_reference_frame_id"],
        reference_frame_spec_hash=(exact["primitive_reference_frame_spec_hash"]),
        execution_reference_frame_id=profile.execution_reference_frame_id,
        execution_reference_frame_spec_hash=(
            profile.execution_reference_frame_spec_hash
        ),
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
    )
    depth_capture = (
        build_mujoco_depth_capture_registration_v2(
            source_catalog=tuple(
                {
                    "source_id": source["view_id"],
                    "role": mujoco_rgb_role_for_source(source["view_id"]),
                    "native_width": source["native_width"],
                    "native_height": source["native_height"],
                }
                for source in application.depth_source_catalog()
            ),
            runtime_session_id=exact["runtime_session_id"],
            control_domain_id=exact["control_domain_id"],
            target_binding_id=target_binding.binding_id,
            station_manifest_hash=station.semantic_content_hash,
            execution_manifest_hash=execution.semantic_content_hash,
            source_frame_id=exact["perception_source_frame_id"],
            source_frame_spec_hash=exact[
                "perception_source_frame_spec_hash"
            ],
            reference_frame_id=exact["primitive_reference_frame_id"],
            reference_frame_spec_hash=exact[
                "primitive_reference_frame_spec_hash"
            ],
            frame_graph_hash=exact["frame_graph_hash"],
        )
        if enable_artifact_perception
        else None
    )
    rgb_capture = (
        build_mujoco_rgb_capture_registration_v2(
            source_catalog=tuple(
                {
                    **dict(source),
                    "role": mujoco_rgb_role_for_source(source["source_id"]),
                }
                for source in application.visual_source_catalog()
            ),
            runtime_session_id=exact["runtime_session_id"],
            control_domain_id=exact["control_domain_id"],
            target_binding_id=target_binding.binding_id,
            station_manifest_hash=station.semantic_content_hash,
            execution_manifest_hash=execution.semantic_content_hash,
        )
        if enable_artifact_perception
        else None
    )
    segmentation = (
        build_segmentation_registration(
            target_kind="sim",
            current_mode="sim_active",
            source_id_by_role={
                mujoco_rgb_role_for_source(source["source_id"]): str(
                    source["source_id"]
                )
                for source in application.visual_source_catalog()
            },
            model_profile_id=segmentation_model_profile_id,
            model_profile_hash=segmentation_model_profile_hash,
            inference_implementation_hash=(
                segmentation_inference_implementation_hash
            ),
            runtime_session_id=exact["runtime_session_id"],
            control_domain_id=exact["control_domain_id"],
            target_binding_id=target_binding.binding_id,
            station_manifest_hash=station.semantic_content_hash,
            execution_manifest_hash=execution.semantic_content_hash,
        )
        if segmentation_model_profile_id is not None
        else None
    )
    placement = (
        build_placement_registration(
            segmentation=segmentation,
            back_projection=depth_capture.back_projection,
            component_id="right_arm",
            destination_role="main",
            runtime_session_id=exact["runtime_session_id"],
            control_domain_id=exact["control_domain_id"],
            target_binding_id=target_binding.binding_id,
            station_manifest_hash=station.semantic_content_hash,
            execution_manifest_hash=execution.semantic_content_hash,
        )
        if segmentation is not None
        and depth_capture is not None
        and depth_capture.back_projection is not None
        else None
    )
    if any(
        item.distance_m > profile.max_translation_delta_m
        for item in cardinal_step_profile.primitives
    ):
        raise ValueError(
            "Cardinal TCP primitive exceeds the lower motion delta profile."
        )
    definition = tcp_pose_target_skill_definition_v1(
        controlled_component_ids=(component,),
    )
    schema_entry = tcp_pose_target_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        allowed_component_ids=(component,),
        allowed_reference_frame_ids=(profile.reference_frame_id,),
        max_requested_duration_ms=None,
        workspace_bounds_by_component={component: profile.workspace_bounds},
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
    cardinal_step_definition = cardinal_tcp_step_skill_definition_v1(
        controlled_component_ids=(component,),
    )
    cardinal_step_schema = cardinal_tcp_step_schema_entry_v1(
        skill_definition_hash=cardinal_step_definition.semantic_content_hash,
        parameter_profile_id=(cardinal_step_profile.parameter_profile_id),
        parameter_profile_hash=cardinal_step_profile.semantic_content_hash,
        allowed_component_ids=(component,),
        allowed_reference_frame_ids=(cardinal_step_profile.reference_frame_id,),
        reference_frame_spec_hash=(cardinal_step_profile.reference_frame_spec_hash),
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
    fixed_cardinal_definitions: list[Any] = []
    fixed_cardinal_schemas: list[Any] = []
    fixed_cardinal_routes: list[SkillExecutionRoute] = []
    for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1:
        fixed_definition = fixed_cardinal_tcp_step_skill_definition_v1(
            skill_id=skill_id,
            controlled_component_ids=(component,),
        )
        fixed_schema = fixed_cardinal_tcp_step_schema_entry_v1(
            skill_id=skill_id,
            skill_definition_hash=fixed_definition.semantic_content_hash,
            parameter_profile_id=(cardinal_step_profile.parameter_profile_id),
            parameter_profile_hash=cardinal_step_profile.semantic_content_hash,
            allowed_component_ids=(component,),
            allowed_reference_frame_ids=(cardinal_step_profile.reference_frame_id,),
            reference_frame_spec_hash=(cardinal_step_profile.reference_frame_spec_hash),
            frame_graph_hash=profile.frame_graph_hash,
            tool_config_hash=profile.tool_config_hash,
            calibration_hash=profile.calibration_hash,
            primitive_catalog_hash=cardinal_step_profile.primitive_catalog_hash,
        )
        fixed_route = SkillExecutionRoute(
            skill_id=fixed_definition.skill_id,
            skill_version=fixed_definition.skill_version,
            skill_definition_hash=fixed_definition.semantic_content_hash,
            route_id=(
                MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1[
                    skill_id
                ]
            ),
            route_version="1",
            parameter_schema_hash=fixed_schema.parameter_schema_hash,
            execution_result_schema_hash=(
                fixed_schema.execution_result_schema_hash
            ),
            task_evaluation_schema_hash=(
                fixed_schema.task_evaluation_schema_hash
            ),
            allowed_parameter_profiles=fixed_schema.parameter_profiles,
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
            route=fixed_route,
            schema_entry=fixed_schema,
        )
        fixed_cardinal_definitions.append(fixed_definition)
        fixed_cardinal_schemas.append(fixed_schema)
        fixed_cardinal_routes.append(fixed_route)
    orientation_motion_profile = MujocoAgentTcpPoseMotionProfile(
        component_id=component,
        reference_frame_id=profile.reference_frame_id,
        reference_frame_spec_hash=profile.reference_frame_spec_hash,
        execution_reference_frame_id=profile.execution_reference_frame_id,
        execution_reference_frame_spec_hash=(
            profile.execution_reference_frame_spec_hash
        ),
        frame_graph_hash=profile.frame_graph_hash,
        tool_config_hash=profile.tool_config_hash,
        calibration_hash=profile.calibration_hash,
        profile_id=f"{side}-arm-fixed-level-tool-tcp-orientation-v1",
        max_translation_delta_m=(
            MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_TRANSLATION_DELTA_M_V1
        ),
        max_rotation_delta_rad=(
            MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_ROTATION_DELTA_RAD_V1
        ),
        max_control_cycles=(
            math.ceil(
                MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_ROTATION_DELTA_RAD_V1
                / MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_REFERENCE_STEP_RAD_V1
            )
            + round(2.0 * orientation_reference_hz)
        ),
        physics_ticks_per_control_cycle=max(
            1,
            round(physics_hz / orientation_reference_hz),
        ),
    )
    orientation_profiles: list[MujocoAgentTcpFixedOrientationProfile] = []
    orientation_definitions: list[Any] = []
    orientation_schemas: list[Any] = []
    orientation_routes: list[SkillExecutionRoute] = []
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
        primitive = level_tcp_orientation(
            MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1[skill_id]
        )
        orientation_profile = MujocoAgentTcpFixedOrientationProfile(
            skill_id=skill_id,
            component_id=component,
            reference_frame_id=exact["primitive_reference_frame_id"],
            reference_frame_spec_hash=(exact["primitive_reference_frame_spec_hash"]),
            execution_reference_frame_id=profile.execution_reference_frame_id,
            execution_reference_frame_spec_hash=(
                profile.execution_reference_frame_spec_hash
            ),
            frame_graph_hash=profile.frame_graph_hash,
            tool_config_hash=profile.tool_config_hash,
            calibration_hash=profile.calibration_hash,
            primitive=primitive,
        )
        orientation_definition = fixed_level_tcp_orientation_skill_definition_v1(
            skill_id=skill_id,
        )
        if side == "left":
            # Scope only the new deployment; preserve legacy right definition bytes.
            orientation_definition = replace(
                orientation_definition, controlled_component_ids=(component,)
            )
        orientation_schema = fixed_level_tcp_orientation_schema_entry_v1(
            skill_id=skill_id,
            skill_definition_hash=(orientation_definition.semantic_content_hash),
            parameter_profile_id=orientation_profile.parameter_profile_id,
            parameter_profile_hash=orientation_profile.semantic_content_hash,
            allowed_component_ids=(component,),
            allowed_reference_frame_ids=(orientation_profile.reference_frame_id,),
            reference_frame_spec_hash=(orientation_profile.reference_frame_spec_hash),
            frame_graph_hash=orientation_profile.frame_graph_hash,
            tool_config_hash=orientation_profile.tool_config_hash,
            calibration_hash=orientation_profile.calibration_hash,
            primitive_hash=primitive.semantic_content_hash,
        )
        orientation_route = SkillExecutionRoute(
            skill_id=orientation_definition.skill_id,
            skill_version=orientation_definition.skill_version,
            skill_definition_hash=(
                orientation_definition.semantic_content_hash
            ),
            route_id=(
                MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1[
                    skill_id
                ]
            ),
            route_version="1",
            parameter_schema_hash=orientation_schema.parameter_schema_hash,
            execution_result_schema_hash=(
                orientation_schema.execution_result_schema_hash
            ),
            task_evaluation_schema_hash=(
                orientation_schema.task_evaluation_schema_hash
            ),
            allowed_parameter_profiles=orientation_schema.parameter_profiles,
            control_effect_class="motion",
            privilege_tier="t1_sim_or_no_motion",
            supported_modes=("sim_active",),
            target_kind="sim",
            may_move_robot=True,
            may_open_publication=False,
            requires_operator_confirmation=False,
            requires_deadman=False,
            max_operation_duration_ms=30_000,
            cancellation_policy="hold_on_cancel_disconnect_or_timeout",
            requires_task_evaluation=True,
        )
        validate_agent_motion_route_schema_v1(
            route=orientation_route,
            schema_entry=orientation_schema,
        )
        orientation_profiles.append(orientation_profile)
        orientation_definitions.append(orientation_definition)
        orientation_schemas.append(orientation_schema)
        orientation_routes.append(orientation_route)
    safety_runtime = getattr(application, "tcp_pose_envelope_runtime", None)

    def safety_envelope_refs_for_route(
        route_id: str,
    ) -> tuple[SkillDeploymentSafetyEnvelopeRef, ...]:
        if safety_runtime is None:
            return ()
        refs = tuple(
            SkillDeploymentSafetyEnvelopeRef(
                object_id=entry.envelope_id,
                schema="dexteleop.deployment_safety_envelope.v1",
                content_hash=entry.semantic_content_hash,
                constraint_set_hash=entry.constraint_set_hash,
                allowed_route_ids=entry.allowed_route_ids,
            )
            for entry in safety_runtime.catalog_entries
            if route_id in entry.allowed_route_ids
        )
        if len(refs) != 1:
            raise ValueError(
                "Each Agent motion Route requires exactly one lower safety envelope."
            )
        return refs

    binding = MujocoAgentTcpPoseDeploymentBinding(
        component_id=component,
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
        safety_envelope_refs=safety_envelope_refs_for_route(route.route_id),
    )
    cardinal_step_binding = MujocoAgentTcpPoseDeploymentBinding(
        component_id=component,
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
        reference_frame_id=cardinal_step_profile.reference_frame_id,
        reference_frame_spec_hash=(cardinal_step_profile.reference_frame_spec_hash),
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
        safety_envelope_refs=safety_envelope_refs_for_route(
            cardinal_step_route.route_id
        ),
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

    def planned_motion_contract(
        motion_definition,
        motion_profile,
        *,
        route_id,
        executor_id,
        binding_prefix,
        duration_ms,
        path_search_time_budget_sec=3.0,
        contact_binding=None,
    ):
        motion_schema = (grasped_object_move_schema(
            definition=motion_definition, profile=motion_profile,
            object_id=grasp_contact_binding.target_body_name,
            observation_source="mujoco_native_physics",
            clock_domain_id="mujoco-native-physics:" + exact["runtime_session_id"],
        ) if motion_definition.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID else tcp_pose_target_schema_entry_v1(
            skill_id=motion_definition.skill_id,
            skill_definition_hash=motion_definition.semantic_content_hash,
            parameter_profile_id=motion_profile.profile_id,
            parameter_profile_hash=motion_profile.semantic_content_hash,
            allowed_component_ids=(component,),
            allowed_reference_frame_ids=(profile.reference_frame_id,),
            max_requested_duration_ms=None,
            workspace_bounds_by_component={component: profile.workspace_bounds},
            parameter_description=(
                None
                if contact_binding is None
                else "Terminal contact target="
                + contact_binding.target_body_name
                + "; sensor geoms="
                + ",".join(contact_binding.sensor_geom_names)
                + "; max approach metres="
                + str(contact_binding.max_approach_m)
                + "; no other contact permission; no automatic closure or carry."
            ),
        ))
        motion_route = replace(
            route,
            skill_id=motion_definition.skill_id,
            skill_definition_hash=motion_definition.semantic_content_hash,
            route_id=route_id,
            skill_io_contract_hash=None,
            parameter_schema_hash=motion_schema.parameter_schema_hash,
            execution_result_schema_hash=motion_schema.execution_result_schema_hash,
            task_evaluation_schema_hash=motion_schema.task_evaluation_schema_hash,
            allowed_parameter_profiles=motion_schema.parameter_profiles,
            max_operation_duration_ms=duration_ms,
        )
        motion_binding = replace(
            binding,
            deployment_binding_id=binding_prefix + exact["runtime_session_id"],
            route_hash=motion_route.semantic_content_hash,
            executor_id=executor_id,
            motion_profile_hash=motion_profile.semantic_content_hash,
            safety_envelope_refs=safety_envelope_refs_for_route(motion_route.route_id),
        )
        motion_contract = MujocoAgentSafeTcpPoseContract(
            deployment_binding=motion_binding,
            evaluation_plan=replace(
                evaluation_plan,
                skill_definition_hash=motion_definition.semantic_content_hash,
                route_hash=motion_route.semantic_content_hash,
                deployment_binding_hash=motion_binding.semantic_content_hash,
                evaluator_specs=(
                    replace(
                        evaluator,
                        configuration_hash=motion_profile.semantic_content_hash,
                    ),
                ),
            ),
            motion_profile=motion_profile,
            path_search_time_budget_sec=path_search_time_budget_sec,
            contact_binding=contact_binding,
        )
        validate_agent_motion_route_schema_v1(
            route=motion_route, schema_entry=motion_schema
        )
        return motion_definition, motion_schema, motion_route, motion_contract

    planned_motions = []
    safe_contract = grasp_contract = contact_contract = object_contract = None
    if enable_safe_tcp_pose:
        item = planned_motion_contract(
            safe_tcp_pose_target_skill_definition_v1(
                controlled_component_ids=(component,)
            ),
            replace(
                profile,
                profile_id="mujoco.agent_safe_tcp_pose.v1",
                max_control_cycles=300,
            ),
            route_id=MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1,
            executor_id=MUJOCO_AGENT_SAFE_TCP_POSE_EXECUTOR_ID_V1,
            binding_prefix="sim.mujoco.agent_safe_tcp_pose:",
            duration_ms=30_000,
        )
        safe_contract = item[3]
        planned_motions.append(item)
    if enable_grasp_approach:
        item = planned_motion_contract(
            grasp_approach_skill_definition_v1(controlled_component_ids=("right_arm",)),
            replace(
                profile,
                profile_id="mujoco.agent_grasp_approach.v1",
                max_control_cycles=1800,
                max_translation_delta_m=math.dist(
                    profile.workspace_bounds[::2], profile.workspace_bounds[1::2]
                ),
                max_rotation_delta_rad=math.pi,
                translation_tolerance_m=0.003,
                rotation_tolerance_rad=math.radians(2),
            ),
            route_id=MUJOCO_AGENT_GRASP_APPROACH_ROUTE_ID_V1,
            executor_id=MUJOCO_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1,
            binding_prefix="sim.mujoco.agent_grasp_approach:",
            duration_ms=120_000,
            path_search_time_budget_sec=15.0,
        )
        grasp_contract = item[3]
        planned_motions.append(item)
    if grasp_contact_binding is not None:
        if not enable_grasp_approach:
            raise ValueError("Terminal contact requires the separate grasp approach Route.")
        item = planned_motion_contract(
            grasp_contact_approach_skill_definition_v1(controlled_component_ids=("right_arm",)),
            replace(
                profile,
                profile_id="mujoco.agent_grasp_contact.v1",
                contact_policy_hash=grasp_contact_binding.semantic_content_hash,
                max_control_cycles=600,
                max_translation_delta_m=grasp_contact_binding.max_approach_m,
                max_rotation_delta_rad=grasp_contact_binding.rotation_tolerance_rad,
                translation_tolerance_m=0.001,
                rotation_tolerance_rad=math.radians(1),
            ),
            route_id=MUJOCO_AGENT_GRASP_CONTACT_ROUTE_ID_V1,
            executor_id="mujoco-grasp-contact-v1",
            binding_prefix="sim.mujoco.agent_grasp_contact:",
            duration_ms=60_000,
            path_search_time_budget_sec=3.0,
            contact_binding=grasp_contact_binding,
        )
        contact_contract = item[3]
        planned_motions.append(item)
    if enable_object_motion:
        item = planned_motion_contract(
            grasped_object_move_definition(controlled_component_ids=(component,)),
            replace(grasp_contract.motion_profile,
                profile_id="mujoco.agent_object_motion." + grasp_contact_binding.semantic_content_hash,
                stable_measurement_count=60),
            route_id=MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V1,
            executor_id="mujoco-grasped-object-motion-v1",
            binding_prefix="sim.mujoco.agent_object_motion:",
            duration_ms=120_000, path_search_time_budget_sec=15.0,
        )
        object_contract = item[3]
        planned_motions.append(item)
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
    fixed_cardinal_contracts: list[
        MujocoAgentTcpFixedCardinalStepContract
    ] = []
    for fixed_definition, fixed_schema, fixed_route in zip(
        fixed_cardinal_definitions,
        fixed_cardinal_schemas,
        fixed_cardinal_routes,
        strict=True,
    ):
        skill_id = fixed_definition.skill_id
        direction = FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1[skill_id]
        fixed_binding = MujocoAgentTcpPoseDeploymentBinding(
            component_id=component,
            deployment_binding_id=(
                "sim.mujoco.agent_tcp_fixed_cardinal_step:"
                + skill_id
                + ":"
                + exact["runtime_session_id"]
            ),
            deployment_binding_version="1",
            route_hash=fixed_route.semantic_content_hash,
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
            reference_frame_id=cardinal_step_profile.reference_frame_id,
            reference_frame_spec_hash=(cardinal_step_profile.reference_frame_spec_hash),
            frame_graph_hash=profile.frame_graph_hash,
            tool_config_hash=profile.tool_config_hash,
            calibration_hash=profile.calibration_hash,
            trace_requirement_profile_id=station.trace_requirement_profile_id,
            executor_id=(
                MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1[
                    skill_id
                ]
            ),
            explicit_exclusions=(
                "agent_authored_direction",
                "agent_authored_distance",
                "hand_control",
                "hardware_backend",
                "model_inference",
                "second_dispatch_owner",
                "trajectory_streaming",
                "upper_target_derivation",
            ),
            safety_envelope_refs=safety_envelope_refs_for_route(fixed_route.route_id),
        )
        fixed_evaluator = SkillEvaluatorSpec(
            evaluator_id=f"motion.tcp_fixed_{direction}_step_arrival_rule",
            evaluator_version="1",
            evaluator_kind="rule_based",
            input_evidence_schema_hash=evaluator_input_hash,
            output_schema_hash=fixed_schema.task_evaluation_schema_hash,
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
                    "direction": direction,
                },
                domain=(
                    "dexteleop.agent_skill.tcp_fixed_cardinal_step_evaluator.v1"
                ),
            ),
        )
        fixed_evaluation_plan = SkillEvaluationPlan(
            evaluation_plan_id=fixed_evaluator.evaluator_id,
            evaluation_plan_version="1",
            skill_definition_hash=fixed_definition.semantic_content_hash,
            route_hash=fixed_route.semantic_content_hash,
            deployment_binding_hash=fixed_binding.semantic_content_hash,
            evaluator_specs=(fixed_evaluator,),
            aggregation_policy_id="motion.single_required_rule.v1",
            aggregation_policy_hash=aggregation_hash,
            required_evaluation_deadline_ms=1_000,
            missing_evidence_policy="fail",
            disagreement_policy="aggregation_policy",
            task_outcome_schema_hash=fixed_schema.task_evaluation_schema_hash,
        )
        fixed_cardinal_contracts.append(
            MujocoAgentTcpFixedCardinalStepContract(
                skill_id=skill_id,
                direction=direction,
                deployment_binding=fixed_binding,
                evaluation_plan=fixed_evaluation_plan,
            )
        )
    orientation_contracts: list[MujocoAgentTcpFixedOrientationContract] = []
    for (
        orientation_profile,
        orientation_definition,
        orientation_schema,
        orientation_route,
    ) in zip(
        orientation_profiles,
        orientation_definitions,
        orientation_schemas,
        orientation_routes,
        strict=True,
    ):
        skill_id = orientation_profile.skill_id
        orientation_binding = MujocoAgentTcpPoseDeploymentBinding(
            component_id=component,
            deployment_binding_id=(
                "sim.mujoco.agent_tcp_fixed_orientation:"
                + skill_id
                + ":"
                + exact["runtime_session_id"]
            ),
            deployment_binding_version="1",
            route_hash=orientation_route.semantic_content_hash,
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
            motion_profile_hash=orientation_motion_profile.semantic_content_hash,
            reference_frame_id=orientation_profile.reference_frame_id,
            reference_frame_spec_hash=(orientation_profile.reference_frame_spec_hash),
            frame_graph_hash=orientation_profile.frame_graph_hash,
            tool_config_hash=orientation_profile.tool_config_hash,
            calibration_hash=orientation_profile.calibration_hash,
            trace_requirement_profile_id=station.trace_requirement_profile_id,
            executor_id=(
                MUJOCO_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1[skill_id]
            ),
            explicit_exclusions=(
                "agent_authored_angle",
                "hand_control",
                "hardware_backend",
                "model_inference",
                "relative_rotation",
                "second_dispatch_owner",
                "trajectory_streaming",
                "upper_target_derivation",
            ),
            safety_envelope_refs=safety_envelope_refs_for_route(
                orientation_route.route_id
            ),
        )
        orientation_evaluator = SkillEvaluatorSpec(
            evaluator_id=(
                "motion.tcp_fixed_orientation_arrival_rule."
                + skill_id.removeprefix("motion.")
            ),
            evaluator_version="1",
            evaluator_kind="rule_based",
            input_evidence_schema_hash=canonical_content_hash(
                {
                    "input": "measured_tcp_pose_after_lower_hold",
                    "motion_profile_hash": (
                        orientation_motion_profile.semantic_content_hash
                    ),
                },
                domain=(
                    "dexteleop.agent_skill.tcp_fixed_orientation_evaluator_input.v1"
                ),
            ),
            output_schema_hash=(
                orientation_schema.task_evaluation_schema_hash
            ),
            role="required",
            execution_location="gateway_worker",
            deadline_ms=1_000,
            failure_policy="veto",
            evidence_domain_id=exact["control_domain_id"],
            configuration_hash=canonical_content_hash(
                {
                    "motion_profile_hash": (
                        orientation_motion_profile.semantic_content_hash
                    ),
                    "orientation_profile_hash": (
                        orientation_profile.semantic_content_hash
                    ),
                },
                domain=(
                    "dexteleop.agent_skill.tcp_fixed_orientation_evaluator.v1"
                ),
            ),
        )
        orientation_evaluation_plan = SkillEvaluationPlan(
            evaluation_plan_id=orientation_evaluator.evaluator_id,
            evaluation_plan_version="1",
            skill_definition_hash=(
                orientation_definition.semantic_content_hash
            ),
            route_hash=orientation_route.semantic_content_hash,
            deployment_binding_hash=orientation_binding.semantic_content_hash,
            evaluator_specs=(orientation_evaluator,),
            aggregation_policy_id="motion.single_required_rule.v1",
            aggregation_policy_hash=aggregation_hash,
            required_evaluation_deadline_ms=1_000,
            missing_evidence_policy="fail",
            disagreement_policy="aggregation_policy",
            task_outcome_schema_hash=(
                orientation_schema.task_evaluation_schema_hash
            ),
        )
        orientation_contracts.append(
            MujocoAgentTcpFixedOrientationContract(
                profile=orientation_profile,
                motion_profile=orientation_motion_profile,
                deployment_binding=orientation_binding,
                evaluation_plan=orientation_evaluation_plan,
            )
        )

    state_definition = tcp_pose_state_definition_v1(
        controlled_component_ids=(component,)
    )
    state_profile_id = f"{side}-arm-agent-motion-tcp-pose-read-v1"
    state_profile_hash = canonical_content_hash(
        {
            "profile_id": state_profile_id,
            "component_ids": (component,),
            "reference_frame_id": profile.reference_frame_id,
            "controlled_frame_id": "tool_tcp",
            "max_measurement_age_ms": (MUJOCO_AGENT_TCP_POSE_MAX_MEASUREMENT_AGE_MS_V1),
        },
        domain="dexteleop.agent_state.mujoco_agent_tcp_pose_profile.v1",
    )
    state_schema = tcp_pose_state_schema_entry_v1(
        skill_definition_hash=state_definition.semantic_content_hash,
        parameter_profile_id=state_profile_id,
        parameter_profile_hash=state_profile_hash,
        allowed_component_ids=(component,),
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
        component_id=component,
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
                "reference_frame_id": profile.reference_frame_id,
                "reference_frame_spec_hash": profile.reference_frame_spec_hash,
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

    spatial_goal_registration = None
    if enable_spatial_goals:
        from agent_gateway.profiles.spatial_goal import build_spatial_goal_registration

        spatial_goal_registration = build_spatial_goal_registration(
            base_route=state_route, base_binding=state_binding,
            component_ids=state_definition.controlled_component_ids,
            reference_frame_id=profile.reference_frame_id,
        )
    # Use exactly the original passive state identity and freshness validator.
    public_state_exact = {
        **exact,
        "reference_frame_id": state_binding.reference_frame_id,
        "reference_frame_spec_hash": state_binding.reference_frame_spec_hash,
    }
    state_validator = _state_validators(
        route=state_route,
        schema_entry=state_schema,
        exact=public_state_exact,
        component_id=component,
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
    state_entry = SkillRouteManifestEntry(
        route=state_route, deployment_binding=state_binding.to_ref(),
        evaluation_plan_hash=None, availability_hash=state_availability,
    )
    tcp_pose_comparison_registration = None
    if enable_tcp_pose_comparison:
        from agent_gateway.profiles.tcp_pose_comparison import (
            build_tcp_pose_comparison_registration,
        )

        tcp_pose_comparison_registration = build_tcp_pose_comparison_registration(
            base_definition=state_definition, base_schema=state_schema,
            base_entry=state_entry, base_validator=state_validator,
        )
    object_state_registration = None
    if enable_object_state:
        from agent_gateway.profiles.object_state import ObjectStateProfile, build_object_state_registration

        object_profile = ObjectStateProfile(
            object_id=grasp_contact_binding.target_body_name,
            reference_frame_id=profile.reference_frame_id,
            reference_frame_spec_hash=profile.reference_frame_spec_hash,
            frame_graph_hash=profile.frame_graph_hash,
            clock_domain_id="mujoco-native-physics:" + exact["runtime_session_id"],
            observation_source="mujoco_native_physics",
            source_binding_hash=canonical_content_hash({
                "state_source_binding": state_binding.semantic_content_hash,
                "contact_binding": grasp_contact_binding.semantic_content_hash,
                "support_body": application.task_config.table_body_name,
                "measurement": "native_body_origin_jacobian_contact_private_copy.v1",
                **({"follow_producer": "rig_tick_frozen_contact_lift_release.v1"}
                   if enable_object_follow_observation else {}),
            }, domain="agenticrobot.mujoco_object_state.source.v1"),
            release_supported=enable_object_follow_observation)
        object_state_registration = build_object_state_registration(base_entry=state_entry, profile=object_profile)
    definition_items = [
        definition,
        cardinal_step_definition,
        *fixed_cardinal_definitions,
        *orientation_definitions,
        state_definition,
    ]
    schema_items = [
        schema_entry,
        cardinal_step_schema,
        *fixed_cardinal_schemas,
        *orientation_schemas,
        state_schema,
    ]
    if spatial_goal_registration is not None:
        definition_items.append(spatial_goal_registration.skill.definition)
        schema_items.append(spatial_goal_registration.skill.schema_entry)
    if tcp_pose_comparison_registration is not None:
        definition_items.append(tcp_pose_comparison_registration.skill.definition)
        schema_items.append(tcp_pose_comparison_registration.skill.schema_entry)
    if object_state_registration is not None:
        definition_items.append(object_state_registration.skill.definition)
        schema_items.append(object_state_registration.skill.schema_entry)
    for motion_definition, motion_schema, _, _ in planned_motions:
        definition_items.append(motion_definition)
        schema_items.append(motion_schema)
    if history_definition is not None and history_schema is not None:
        definition_items.append(history_definition)
        schema_items.append(history_schema)
    if depth_capture is not None:
        definition_items.append(depth_capture.registration.definition)
        schema_items.append(depth_capture.registration.schema_entry)
        if depth_capture.back_projection is not None:
            definition_items.append(depth_capture.back_projection.skill.definition)
            schema_items.append(depth_capture.back_projection.skill.schema_entry)
    if rgb_capture is not None:
        definition_items.append(rgb_capture.registration.definition)
        schema_items.append(rgb_capture.registration.schema_entry)
    if segmentation is not None:
        definition_items.append(segmentation.skill.definition)
        schema_items.append(segmentation.skill.schema_entry)
    if placement is not None:
        definition_items.append(placement.skill.definition)
        schema_items.append(placement.skill.schema_entry)
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
        state_entry,
    ]
    if tcp_pose_comparison_registration is not None:
        route_items.append(tcp_pose_comparison_registration.entry)
    if object_state_registration is not None:
        route_items.append(object_state_registration.entry)
    for _, _, safe_route, planned_contract in planned_motions:
        route_items.append(
            SkillRouteManifestEntry(
                route=safe_route,
                deployment_binding=planned_contract.deployment_binding.to_ref(),
                evaluation_plan_hash=planned_contract.evaluation_plan.semantic_content_hash,
                availability_hash=canonical_content_hash(
                    {
                        "route": safe_route.semantic_content_hash,
                        "planner": "mujoco_collision_checked_joint_path_v1",
                        "planning_clearance_padding_m": (
                            application.target.planned_tcp_motion.planning_clearance_padding_m
                        ),
                        "path_search_time_budget_sec": planned_contract.path_search_time_budget_sec,
                    },
                    domain="dexteleop.mujoco.safe_tcp_availability.v1",
                ),
            )
        )
    for fixed_route, fixed_contract in zip(
        fixed_cardinal_routes,
        fixed_cardinal_contracts,
        strict=True,
    ):
        route_items.insert(
            -1,
            SkillRouteManifestEntry(
                route=fixed_route,
                deployment_binding=fixed_contract.deployment_binding.to_ref(),
                evaluation_plan_hash=(
                    fixed_contract.evaluation_plan.semantic_content_hash
                ),
                availability_hash=canonical_content_hash(
                    {
                        "status": "configured_requires_activation_preflight",
                        "route_hash": fixed_route.semantic_content_hash,
                        "deployment_binding_hash": (
                            fixed_contract.deployment_binding.semantic_content_hash
                        ),
                        "cardinal_step_profile_hash": (
                            cardinal_step_profile.semantic_content_hash
                        ),
                        "fixed_direction": fixed_contract.direction,
                    },
                    domain=(
                        "dexteleop.agent_skill."
                        "mujoco_tcp_fixed_cardinal_step_availability.v1"
                    ),
                ),
            ),
        )
    for orientation_route, orientation_contract in zip(
        orientation_routes,
        orientation_contracts,
        strict=True,
    ):
        route_items.insert(
            -1,
            SkillRouteManifestEntry(
                route=orientation_route,
                deployment_binding=(
                    orientation_contract.deployment_binding.to_ref()
                ),
                evaluation_plan_hash=(
                    orientation_contract.evaluation_plan.semantic_content_hash
                ),
                availability_hash=canonical_content_hash(
                    {
                        "status": "configured_requires_activation_preflight",
                        "route_hash": orientation_route.semantic_content_hash,
                        "deployment_binding_hash": (
                            orientation_contract.deployment_binding.semantic_content_hash
                        ),
                        "orientation_profile_hash": (
                            orientation_contract.profile.semantic_content_hash
                        ),
                    },
                    domain=(
                        "dexteleop.agent_skill."
                        "mujoco_tcp_fixed_orientation_availability.v1"
                    ),
                ),
            ),
        )
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
    if depth_capture is not None:
        depth_route = depth_capture.registration.routes[0]
        route_items.append(
            SkillRouteManifestEntry(
                route=depth_route,
                deployment_binding=depth_capture.deployment_binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=depth_capture.availability_hash,
            )
        )
        if depth_capture.back_projection is not None:
            back_projection = depth_capture.back_projection
            route_items.append(
                SkillRouteManifestEntry(
                    route=back_projection.skill.routes[0],
                    deployment_binding=back_projection.binding.to_ref(),
                    evaluation_plan_hash=None,
                    availability_hash=back_projection.availability_hash,
                )
            )
    if rgb_capture is not None:
        rgb_route = rgb_capture.registration.routes[0]
        route_items.append(
            SkillRouteManifestEntry(
                route=rgb_route,
                deployment_binding=rgb_capture.deployment_binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=rgb_capture.availability_hash,
            )
        )
    if segmentation is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=segmentation.skill.routes[0],
                deployment_binding=segmentation.binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=segmentation.availability_hash,
            )
        )
    if placement is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=placement.skill.routes[0],
                deployment_binding=placement.binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=placement.availability_hash,
            )
        )
    if spatial_goal_registration is not None:
        route_items.append(spatial_goal_registration.entry)
    io_contract_by_skill = {}
    for item in route_items:
        route_item = item.route
        key = (route_item.skill_id, route_item.skill_version)
        explicit_contracts = tuple(
            registration.registration.io_contract
            for registration in (depth_capture, rgb_capture)
            if registration is not None
            and key
            == (
                registration.registration.definition.skill_id,
                registration.registration.definition.skill_version,
            )
        )
        if (
            depth_capture is not None
            and depth_capture.back_projection is not None
            and key
            == (
                depth_capture.back_projection.skill.definition.skill_id,
                depth_capture.back_projection.skill.definition.skill_version,
            )
        ):
            explicit_contracts += (depth_capture.back_projection.skill.io_contract,)
        if segmentation is not None and key == (
            segmentation.skill.definition.skill_id,
            segmentation.skill.definition.skill_version,
        ):
            explicit_contracts += (segmentation.skill.io_contract,)
        if placement is not None and key == (
            placement.skill.definition.skill_id,
            placement.skill.definition.skill_version,
        ):
            explicit_contracts += (placement.skill.io_contract,)
        contract = (
            explicit_contracts[0]
            if explicit_contracts
            else inline_only_skill_io_contract(
                skill_id=route_item.skill_id,
                skill_version=route_item.skill_version,
                skill_definition_hash=route_item.skill_definition_hash,
                parameter_schema_hash=route_item.parameter_schema_hash,
                execution_result_schema_hash=(route_item.execution_result_schema_hash),
                task_evaluation_schema_hash=(route_item.task_evaluation_schema_hash),
            )
        )
        previous = io_contract_by_skill.get(key)
        if previous is not None and previous != contract:
            raise ValueError("Routes for one Skill disagree on the I/O contract.")
        io_contract_by_skill[key] = contract
    io_catalog = SkillIOCatalog(
        catalog_id="dexteleop.agent_skill_io",
        catalog_version="1",
        io_contracts=tuple(
            io_contract_by_skill[key] for key in sorted(io_contract_by_skill)
        ),
    )
    manifest = SkillRouteManifest(
        runtime_session_id=exact["runtime_session_id"],
        control_domain_id=exact["control_domain_id"],
        station_manifest_hash=station.semantic_content_hash,
        execution_manifest_hash=execution.semantic_content_hash,
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        skill_io_catalog_hash=io_catalog.semantic_content_hash,
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
            *(("model_inference",) if segmentation is None else ()),
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
        state_validator,
    ]
    validator_items[2:2] = [
        _cardinal_step_validators(
            route=fixed_route,
            profile=cardinal_step_profile,
            allowed_directions=(fixed_contract.direction,),
        )
        for fixed_route, fixed_contract in zip(
            fixed_cardinal_routes,
            fixed_cardinal_contracts,
            strict=True,
        )
    ]
    validator_items[2:2] = [
        _fixed_orientation_validators(
            route=orientation_route,
            profile=orientation_contract.profile,
        )
        for orientation_route, orientation_contract in zip(
            orientation_routes,
            orientation_contracts,
            strict=True,
        )
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
                exact=public_state_exact,
                profile=history_profile,
                source_binding=history_source_binding,
            )
        )
    if depth_capture is not None:
        validator_items.append(depth_capture.payload_validator)
        if depth_capture.back_projection is not None:
            validator_items.append(depth_capture.back_projection.payload_validator)
    if rgb_capture is not None:
        validator_items.append(rgb_capture.payload_validator)
    if spatial_goal_registration is not None:
        validator_items.append(spatial_goal_registration.payload_validator)
    if tcp_pose_comparison_registration is not None:
        validator_items.append(tcp_pose_comparison_registration.payload_validator)
    if object_state_registration is not None:
        validator_items.append(object_state_registration.payload_validator)
    if segmentation is not None:
        validator_items.append(segmentation.payload_validator)
    if placement is not None:
        validator_items.append(placement.payload_validator)
    for _, safe_schema, safe_route, planned_contract in planned_motions:
        validator_items.append(
            _motion_validators(
                route=safe_route,
                schema_entry=safe_schema,
                profile=planned_contract.motion_profile,
                **({"object_id": grasp_contact_binding.target_body_name,
                    "object_clock_domain_id": "mujoco-native-physics:" + exact["runtime_session_id"]}
                   if safe_route.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID else {}),
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
        io_catalog=io_catalog,
        route_manifest=manifest,
        deployment_binding=binding,
        cardinal_step_deployment_binding=cardinal_step_binding,
        state_deployment_binding=state_binding,
        motion_profile=profile,
        cardinal_step_profile=cardinal_step_profile,
        evaluation_plan=evaluation_plan,
        cardinal_step_evaluation_plan=cardinal_step_evaluation_plan,
        fixed_cardinal_step_contracts=tuple(fixed_cardinal_contracts),
        fixed_orientation_contracts=tuple(orientation_contracts),
        payload_validators=validators,
        station_manifest=station,
        execution_manifest=execution,
        safe_tcp_pose_contract=safe_contract,
        spatial_goal_registration=spatial_goal_registration,
        tcp_pose_comparison_registration=tcp_pose_comparison_registration,
        object_state_registration=object_state_registration,
        grasp_approach_contract=grasp_contract,
        grasp_contact_contract=contact_contract,
        object_motion_contract=object_contract,
        depth_capture=depth_capture,
        rgb_capture=rgb_capture,
        segmentation=segmentation,
        placement=placement,
        history_profile=history_profile,
        history_source_binding=history_source_binding,
        history_deployment_binding=history_binding,
    )


__all__ = [
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_DIRECTION_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_ROTATION_DELTA_RAD_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_MAX_TRANSLATION_DELTA_M_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1",
    "MUJOCO_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1",
    "MUJOCO_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1",
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
    "MujocoAgentTcpFixedCardinalStepContract",
    "MujocoAgentTcpFixedOrientationContract",
    "MujocoAgentTcpFixedOrientationProfile",
    "MujocoAgentTcpPoseContractBundle",
    "MujocoAgentTcpPoseDeploymentBinding",
    "MujocoAgentTcpPoseHistoryProfile",
    "MujocoAgentTcpPoseMotionProfile",
    "MujocoPassiveTcpPoseHistoryDeploymentBinding",
    "build_mujoco_agent_tcp_pose_contract_bundle",
]

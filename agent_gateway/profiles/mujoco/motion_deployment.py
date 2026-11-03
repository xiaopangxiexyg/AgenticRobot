"""Exact no-model Sim Route and deployment for Agent TCP-pose motion."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gateway.profiles.object_state import ObjectStateRegistration
    from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
    from agent_gateway.profiles.tcp_pose_comparison import TcpPoseComparisonRegistration

from agent_gateway.profiles.placement import (
    PlacementRegistration,
    build_placement_registration,
)
from agent_gateway.profiles.localization_deployment import (
    ObjectLocalizationRegistration,
    build_object_localization_registration,
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
    AgentSkillParameterRejection,
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.skills.gripper.deployment import GripperSkillContractBundle
from agent_gateway.skills.motion.definition import (
    ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1,
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    grasp_approach_skill_definition_v1,
    grasp_contact_approach_skill_definition_v1,
    safe_tcp_pose_target_skill_definition_v1,
    tcp_pose_target_skill_definition_v1,
)
from agent_gateway.skills.motion.grasped_object import (
    GRASPED_OBJECT_MOVE_SKILL_ID,
    grasped_object_move_definition,
    grasped_object_move_schema,
    grasped_object_place_definition_v2,
    grasped_object_place_schema_v2,
    validate_grasped_object_move,
    grasped_object_move_definition_v3, grasped_object_move_schema_v3,
    validate_grasped_object_move_v3, validate_object_motion_state_v3,
    OBJECT_INCLINATION_DEVIATION_RAD_V3,
    validate_grasped_object_place_v2,
    validate_object_motion_state,
    validate_object_motion_state_v2,
)
from agent_gateway.skills.motion.path_constraint import (
    safe_tcp_pose_target_skill_definition_v2,
    safe_tcp_pose_target_schema_entry_v2,
    validate_safe_tcp_pose_parameters_v2,
    validate_path_constraint_evidence,
)
from agent_gateway.skills.motion.schemas import (
    tcp_pose_target_schema_entry_v1,
    validate_safe_tcp_pose_goal_report_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)
from agent_gateway.skills.motion.observed_point import (
    SKILL_ID as OBSERVED_POINT_SKILL_ID, observed_point_definition, observed_point_schema,
    observed_anchor_definition_v2, observed_anchor_schema_v2,
    validate_observed_anchor_parameters_v2, validate_observed_anchor_report_v2,
    observed_anchor_region_definition_v3, observed_anchor_region_schema_v3,
    observed_anchor_region_offsets_v3, validate_observed_anchor_region_parameters_v3,
    validate_observed_anchor_region_report_v3,
    observed_anchor_hover_definition_v4, observed_anchor_hover_schema_v4,
    observed_anchor_hover_offsets_v4, validate_observed_anchor_hover_parameters_v4,
    validate_observed_point_parameters, validate_observed_point_report,
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
from sim.mujoco.deploy.grasp_contact import MujocoGraspContactBinding

from .depth_deployment import (
    MujocoDepthCaptureRegistration,
    build_mujoco_depth_capture_registration_v2,
)
from .motion_primitive_profiles import (
    MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1,
    MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1,
    MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_DIRECTION_BY_SKILL_ID_V1 as MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_DIRECTION_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1,
    MUJOCO_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1,
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

MUJOCO_OBSERVED_POINT_ROUTE_ID = "sim.mujoco.agent_observed_point_safe"
MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V2 = "sim.mujoco.agent_observed_anchor_safe.v2"
MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V3 = "sim.mujoco.agent_observed_anchor_region_safe.v3"
MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V4 = "sim.mujoco.agent_observed_anchor_hover_safe.v4"
MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1 = "sim.mujoco.motion.move_tcp_to_pose_safe.v1"
MUJOCO_AGENT_SAFE_TCP_POSE_EXECUTOR_ID_V1 = "mujoco-agent-safe-tcp-pose-executor-v1"
MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V2 = "sim.mujoco.motion.move_tcp_to_pose_safe.v2"
MUJOCO_AGENT_SAFE_TCP_POSE_EXECUTOR_ID_V2 = "mujoco-agent-safe-tcp-pose-executor-v2"
MUJOCO_AGENT_GRASP_APPROACH_ROUTE_ID_V1 = "sim.mujoco.motion.approach_grasp_pose.v1"
MUJOCO_AGENT_GRASP_CONTACT_ROUTE_ID_V1 = "sim.mujoco.motion.approach_grasp_contact.v1"
MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V1 = "sim.mujoco.motion.move_grasped_object_to_pose.v1"
MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V2 = "sim.mujoco.motion.move_grasped_object_to_pose.v2"
MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V3 = "sim.mujoco.motion.move_grasped_object_to_pose.v3"
MUJOCO_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1 = "mujoco-agent-grasp-approach-executor-v1"
MUJOCO_AGENT_OBJECT_MOVE_MAX_OPERATION_MS = 240_000
# Simulated time a held object (and the TCP) must stay inside the goal
# tolerance before an object motion succeeds. Was 60 control cycles (about 4 s
# at the deployed decision rate), which dominated the wall time of every lift,
# carry and place under the per-step held-object guards. Holding verification
# keeps its own 0.5 s unsupported-following window.
MUJOCO_AGENT_OBJECT_MOVE_STABLE_SEC = 0.5

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


@dataclass(frozen=True)
class MujocoSafeTcpMotionLimits:
    max_translation_delta_m: float
    max_rotation_delta_rad: float
    max_control_cycles: int
    max_operation_duration_ms: int
    translation_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = math.radians(5.0)
    normalize_ik_objective: bool = False


MUJOCO_SAFE_TCP_MOTION_PROFILES = {
    "local_step": MujocoSafeTcpMotionLimits(
        MUJOCO_AGENT_TCP_POSE_MAX_TRANSLATION_DELTA_M_V1,
        MUJOCO_AGENT_TCP_POSE_MAX_ROTATION_DELTA_RAD_V1, 300, 30_000),
    "object_relation": MujocoSafeTcpMotionLimits(0.50, math.pi, 900, 90_000),
    "object_relation_precise": MujocoSafeTcpMotionLimits(
        0.50, math.pi, 900, 90_000, 0.003, math.radians(1.0), True),
}


def resolve_safe_tcp_motion_limits(name: str, *, enabled: bool) -> MujocoSafeTcpMotionLimits:
    """Select an explicit sim-only deployment; never widen ordinary Routes."""
    if not isinstance(name, str) or name not in MUJOCO_SAFE_TCP_MOTION_PROFILES:
        raise ValueError("Unknown MuJoCo safe TCP motion profile.")
    if not enabled and name != "local_step":
        raise ValueError("A non-default safe TCP profile requires enable_safe_tcp_pose.")
    return MUJOCO_SAFE_TCP_MOTION_PROFILES[name]


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

    normalize_ik_objective: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.normalize_ik_objective, bool):
            raise TypeError("normalize_ik_objective must be a bool.")
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
                **({"ik_objective_mode": "calibrated_tcp_tolerance_normalized"}
                   if self.normalize_ik_objective else {}),
                **({"contact_policy_hash": self.contact_policy_hash}
                   if self.contact_policy_hash is not None else {}),
            },
            domain="dexteleop.agent_skill.mujoco_tcp_pose_profile.v1",
        )

    def to_payload(self) -> dict[str, Any]:
        """Preserve the original default export and publish opt-in IK mode explicitly."""
        return {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "normalize_ik_objective" or self.normalize_ik_objective
        }


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
    state_deployment_binding: MujocoPassiveTcpPoseStateDeploymentBinding
    motion_profile: MujocoAgentTcpPoseMotionProfile
    evaluation_plan: SkillEvaluationPlan
    payload_validators: AgentSkillPayloadValidatorRegistry
    station_manifest: ControlStationRunManifest
    execution_manifest: Any
    safe_tcp_pose_contract: MujocoAgentSafeTcpPoseContract | None = None
    safe_tcp_pose_v2_contract: MujocoAgentSafeTcpPoseContract | None = None
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
    object_support_motion_contract_v2: MujocoAgentSafeTcpPoseContract | None = None
    object_attitude_motion_contract_v3: MujocoAgentSafeTcpPoseContract | None = None
    object_support_body_names: tuple[str, ...] = ()
    fast_grasp_contract: Any | None = None
    observed_point_contract: MujocoAgentSafeTcpPoseContract | None = None
    observed_anchor_contract_v2: MujocoAgentSafeTcpPoseContract | None = None
    observed_anchor_region_contract_v3: MujocoAgentSafeTcpPoseContract | None = None
    observed_anchor_hover_contract_v4: MujocoAgentSafeTcpPoseContract | None = None
    rgbd_capture: SpatialGoalRegistration | None = None
    segmentation_batch: SegmentationRegistration | None = None
    segmentation_geometry: SegmentationRegistration | None = None
    localization: ObjectLocalizationRegistration | None = None

    object_goal_query_registration: SpatialGoalRegistration | None = None
    surface_anchor_registrations: tuple[SpatialGoalRegistration, ...] = ()
    tcp_continuation_query_registration: SpatialGoalRegistration | None = None
    candidate_query_registrations: tuple[SpatialGoalRegistration, ...] = ()
    # Shared tool-entity relations/hover Skills (agent_gateway.profiles.tool_entity).
    tool_entity_registrations: tuple[Any, ...] = ()
    # Selected perception.query_support_release_pose@1/@3 (mujoco/support_release_query.py).
    support_release_query_registration: Any | None = None
    # Selected perception.query_grasp_approach@1/@2 (mujoco/grasp_approach_query.py).
    grasp_approach_query_registration: Any | None = None

    def __post_init__(self) -> None:
        published_skill_ids = {
            *(item.skill_id for item in self.definition_catalog.skill_definitions),
            *(item.skill_id for item in self.schema_catalog.schema_entries),
            *(item.route.skill_id for item in self.route_manifest.skill_routes),
            *(item.skill_id for item in self.io_catalog.io_contracts),
        }
        if published_skill_ids.intersection(ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1):
            raise ValueError("Archived debug motion Skills cannot be published.")
        if bool(self.object_support_motion_contract_v2) != bool(self.object_support_body_names):
            raise ValueError("Object support v2 contract requires its exact support allowlist.")
        if (self.object_goal_query_registration is not None
                and self.route_manifest.route("perception.query_object_above_paths", "1") != self.object_goal_query_registration.entry):
            raise ValueError("Object goal query registration differs from its manifest entry.")

        for registration in (*self.surface_anchor_registrations, *self.candidate_query_registrations):
            if self.route_manifest.route(registration.skill.definition.skill_id,
                    registration.skill.definition.skill_version) != registration.entry:
                raise ValueError("Surface anchor registration differs from its manifest entry.")
        for registration in self.tool_entity_registrations:
            if self.route_manifest.route(registration.skill.definition.skill_id,
                    registration.skill.definition.skill_version) != registration.entry:
                raise ValueError("Tool-entity registration differs from its manifest entry.")
            if registration.skill.definition.skill_id not in {
                    "perception.describe_tool_object_relations", "perception.query_tool_hover_paths"}:
                raise ValueError("Unknown tool-entity registration.")
        grasp_query = self.grasp_approach_query_registration
        if grasp_query is not None and (
                self.route_manifest.route("perception.query_grasp_approach",
                    grasp_query.skill.definition.skill_version) != grasp_query.entry
                or self.grasp_approach_contract is None):
            raise ValueError("Grasp approach query registration differs from its manifest entry.")
        release_query = self.support_release_query_registration
        if release_query is not None:
            if self.route_manifest.route("perception.query_support_release_pose",
                    release_query.skill.definition.skill_version) != release_query.entry:
                raise ValueError("Support release query registration differs from its manifest entry.")
            if (self.object_motion_contract is None
                    or not set(release_query.support_object_ids) <= set(self.object_support_body_names)):
                raise ValueError("Support release queries need object motion and admitted supports.")

        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
            io_catalog=self.io_catalog,
        )
        if self.localization is not None:
            if not isinstance(self.localization, ObjectLocalizationRegistration):
                raise TypeError("localization must be a typed registration.")
            entry = self.route_manifest.route("perception.localize_objects", "1")
            if (entry.route != self.localization.skill.routes[0]
                    or entry.deployment_binding != self.localization.binding.to_ref()
                    or entry.availability_hash != self.localization.availability_hash
                    or self.localization.binding.target_kind != "sim"):
                raise ValueError("Object localization differs from its simulation manifest entry.")
            if self.segmentation is None or self.localization.profile.local_path_enabled:
                raise ValueError("Simulation localization requires the shared segmentation model and Artifact-only input.")
            for name in ("model_profile_id", "model_profile_hash", "inference_implementation_hash"):
                if getattr(self.localization.profile, name) != getattr(self.segmentation.profile, name):
                    raise ValueError("Object localization must share the exact segmentation model identity.")
            for name in ("runtime_session_id", "control_domain_id", "target_binding_id",
                         "station_manifest_hash", "execution_manifest_hash", "target_kind"):
                if getattr(self.localization.binding, name) != getattr(self.segmentation.binding, name):
                    raise ValueError("Object localization must share the exact segmentation deployment context.")
        elif any(entry.route.skill_id == "perception.localize_objects" for entry in self.route_manifest.skill_routes):
            raise ValueError("A localization Route requires its typed registration.")
        if self.fast_grasp_contract is not None:
            from .grasp_deployment import MujocoFastGraspContract

            if not isinstance(self.fast_grasp_contract, MujocoFastGraspContract):
                raise TypeError("fast_grasp_contract must be a native typed contract.")
            fast = self.fast_grasp_contract
            if (self.grasp_approach_contract is None or self.grasp_contact_contract is None
                    or self.gripper_contract is None or self.rgb_capture is None):
                raise ValueError("Fast grasp requires the complete native observation/contact graph.")
            for registration in (fast.observation, fast.fast, fast.execute):
                route = registration.skill.routes[0]
                if self.route_manifest.route(route.skill_id, route.skill_version) != registration.entry:
                    raise ValueError("Fast grasp registration differs from its manifest entry.")
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
            entry = self.route_manifest.route(
                "manipulation.set_gripper_state", hand.deployment_binding.skill_version)
            if (entry.route != hand.route_manifest.skill_routes[0].route
                    or entry.deployment_binding.deployment_binding_hash != hand.deployment_binding.semantic_content_hash):
                raise ValueError("Composed gripper Route changed.")
        if self.tcp_continuation_query_registration is not None:
            if self.safe_tcp_pose_v2_contract is None:
                raise ValueError("Continuation queries require the matching safe TCP v2 deployment.")
            if self.route_manifest.route("perception.query_tcp_continuation_paths", "1") != self.tcp_continuation_query_registration.entry:
                raise ValueError("Continuation query registration differs from its manifest entry.")
        if self.candidate_query_registrations and (
                self.tcp_continuation_query_registration is None or not any(
                    item.skill.definition.skill_id == "perception.capture_surface_anchor"
                    and item.skill.definition.skill_version == "2" for item in self.surface_anchor_registrations)):
            raise ValueError("Candidate query registrations require their anchor and continuation owners.")
        if self.candidate_query_registrations and tuple(
                (item.skill.definition.skill_id, item.skill.definition.skill_version)
                for item in self.candidate_query_registrations) != (
                    ("perception.query_relation_paths", "3"), ("perception.query_tcp_continuation_paths", "2")):
            raise ValueError("Candidate query registrations must contain the ordered producer/consumer pair.")
        if self.safe_tcp_pose_v2_contract is not None and self.safe_tcp_pose_contract is None:
            raise ValueError("Safe TCP v2 requires the existing safe TCP deployment.")
        if self.safe_tcp_pose_v2_contract is not None and self.safe_tcp_pose_contract is not None:
            baseline = self.safe_tcp_pose_contract.motion_profile
            if replace(self.safe_tcp_pose_v2_contract.motion_profile,
                    profile_id=baseline.profile_id) != baseline:
                raise ValueError("Safe TCP v2 must share the exact baseline numeric and geometry profile.")
        if (self.observed_anchor_contract_v2 is not None
                or self.observed_anchor_region_contract_v3 is not None
                or self.observed_anchor_hover_contract_v4 is not None) and not any(
                item.skill.definition.skill_id == "perception.capture_surface_anchor"
                and item.skill.definition.skill_version == "2"
                for item in self.surface_anchor_registrations):
            raise ValueError("Anchor motion v2 requires its authenticated v2 producer.")
        for skill_id, version, safe in (
            ("motion.move_tcp_to_pose_safe", "1", self.safe_tcp_pose_contract),
            ("motion.move_tcp_to_pose_safe", "2", self.safe_tcp_pose_v2_contract),
            (OBSERVED_POINT_SKILL_ID, "1", self.observed_point_contract),
            (OBSERVED_POINT_SKILL_ID, "2", self.observed_anchor_contract_v2),
            (OBSERVED_POINT_SKILL_ID, "3", self.observed_anchor_region_contract_v3),
            (OBSERVED_POINT_SKILL_ID, "4", self.observed_anchor_hover_contract_v4),
            ("motion.approach_grasp_pose", "1", self.grasp_approach_contract),
            ("motion.approach_grasp_contact", "1", self.grasp_contact_contract),
            (GRASPED_OBJECT_MOVE_SKILL_ID, "1", self.object_motion_contract),
            (GRASPED_OBJECT_MOVE_SKILL_ID, "2", self.object_support_motion_contract_v2),
            (GRASPED_OBJECT_MOVE_SKILL_ID, "3", self.object_attitude_motion_contract_v3),
        ):
            if safe is None:
                continue
            entry = self.route_manifest.route(skill_id, version)
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
        for version, batch in (("2", self.segmentation_batch), ("3", self.segmentation_geometry)):
            if batch is None:
                if any(entry.route.skill_id == "perception.segment" and entry.route.skill_version == version
                       for entry in self.route_manifest.skill_routes):
                    raise ValueError("A segmentation batch Route requires its typed registration.")
                continue
            if not isinstance(batch, SegmentationRegistration):
                raise TypeError("segmentation_batch must be a typed registration.")
            if (segmentation is None or depth_capture is None or rgb_capture is None
                    or segmentation.skill.definition.skill_version != "1"
                    or batch.skill.definition.skill_version != version
                    or batch.skill.definition.skill_id != segmentation.skill.definition.skill_id):
                raise ValueError(f"Segmentation batch v{version} requires the v1 model and Artifact capture registrations.")
            if replace(batch.profile, profile_id=segmentation.profile.profile_id,
                       rgb_capture_route_hash=None) != segmentation.profile:
                raise ValueError("Segmentation batch must share the exact v1 model, sources and byte bounds.")
            if version == "3" and batch.profile.rgb_capture_route_hash != (
                    self.rgbd_capture.entry.route.semantic_content_hash if self.rgbd_capture is not None else None):
                raise ValueError("Segmentation v3 must bind its exact selected RGB producer Route.")
            for name in ("runtime_session_id", "control_domain_id", "target_binding_id",
                         "station_manifest_hash", "execution_manifest_hash", "target_kind"):
                if getattr(batch.binding, name) != getattr(segmentation.binding, name):
                    raise ValueError("Segmentation batch must share the exact v1 deployment context.")
            entry = self.route_manifest.route(batch.skill.definition.skill_id, version)
            if (entry.route != batch.skill.routes[0] or entry.deployment_binding != batch.binding.to_ref()
                    or entry.availability_hash != batch.availability_hash or entry.evaluation_plan_hash is not None):
                raise ValueError("Segmentation batch Route differs from its registration.")
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
        if self.rgbd_capture is not None:
            if (depth_capture is None or rgb_capture is None
                    or depth_capture.back_projection is None
                    or self.route_manifest.route("perception.capture_rgb", "3") != self.rgbd_capture.entry):
                raise ValueError("Paired RGB-D capture requires its exact RGB, depth and manifest registrations.")
        elif any(item.route.skill_id == "perception.capture_rgb" and item.route.skill_version == "3"
                 for item in self.route_manifest.skill_routes):
            raise ValueError("RGB-D capture v3 Route has no registration.")
        if motion_entry.deployment_binding.deployment_binding_hash != (
            self.deployment_binding.semantic_content_hash
        ):
            raise ValueError("Motion Route uses another deployment binding.")
        if state_entry.deployment_binding.deployment_binding_hash != (
            self.state_deployment_binding.semantic_content_hash
        ):
            raise ValueError("State Route uses another deployment binding.")
        if self.deployment_binding.component_id != self.motion_profile.component_id:
            raise ValueError("Motion deployment and profile component mismatch.")
        if motion_entry.evaluation_plan_hash != (
            self.evaluation_plan.semantic_content_hash
        ):
            raise ValueError("Motion Route uses another evaluation plan.")
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
    object_support_body_names: tuple[str, ...] = (),
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

    is_safe_v2 = (route.skill_id == "motion.move_tcp_to_pose_safe" and route.skill_version == "2")
    if is_safe_v2:
        parameter_fields.add("path_constraint")
    is_anchor_v2 = (route.skill_id == OBSERVED_POINT_SKILL_ID and route.skill_version == "2")
    is_anchor_v3 = (route.skill_id == OBSERVED_POINT_SKILL_ID and route.skill_version == "3")
    is_anchor_v4 = (route.skill_id == OBSERVED_POINT_SKILL_ID and route.skill_version == "4")
    is_anchor_region = is_anchor_v3 or is_anchor_v4
    is_object_support_v2 = (route.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID
                            and route.skill_version == "2")
    is_object_attitude_v3 = (route.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID
                             and route.skill_version == "3")

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        if route.skill_id == OBSERVED_POINT_SKILL_ID:
            if is_anchor_v4:
                validate_observed_anchor_hover_parameters_v4(payload, component_id=profile.component_id)
            elif is_anchor_v3:
                validate_observed_anchor_region_parameters_v3(payload, component_id=profile.component_id)
            elif is_anchor_v2:
                validate_observed_anchor_parameters_v2(payload, component_id=profile.component_id)
            else:
                validate_observed_point_parameters(payload, component_id=profile.component_id)
            return
        if object_id is not None:
            if is_object_attitude_v3:
                validate_grasped_object_move_v3(payload, profile=profile, object_id=object_id,
                    support_object_ids=object_support_body_names)
            elif is_object_support_v2:
                validate_grasped_object_place_v2(payload, profile=profile,
                    object_id=object_id, support_object_ids=object_support_body_names)
            else:
                validate_grasped_object_move(payload, profile=profile, object_id=object_id)
            return
        optional = {"pose_goal_policy"} if route.skill_id == "motion.move_tcp_to_pose_safe" else set()
        if not isinstance(payload, Mapping) or not parameter_fields <= set(payload) <= parameter_fields | optional:
            raise ValueError("TCP pose target must exactly match its profile schema.")
        if is_safe_v2:
            validate_safe_tcp_pose_parameters_v2(payload, (profile.component_id,))
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
                raise AgentSkillParameterRejection(
                    f"parameter_payload_rejected_{field_name}_mismatch",
                    f"TCP pose target {field_name} mismatch.",
                )
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
        if is_safe_v2:
            required.add("path_constraint_evidence")
        if route.skill_id == OBSERVED_POINT_SKILL_ID:
            required.add("surface_anchor_region_report" if is_anchor_region else
                         "surface_anchor_report" if is_anchor_v2 else "observed_point_report")
        if object_id is not None:
            required.add("object_state")
        optional = {"pose_goal_report"} if route.skill_id == "motion.move_tcp_to_pose_safe" else set()
        if not isinstance(payload, Mapping) or not required <= set(payload) <= required | optional:
            raise ValueError("TCP pose execution result schema mismatch.")
        if is_safe_v2:
            validate_path_constraint_evidence(payload["path_constraint_evidence"])
        if "pose_goal_report" in payload:
            validate_safe_tcp_pose_goal_report_v1(payload["pose_goal_report"])
        if route.skill_id == OBSERVED_POINT_SKILL_ID:
            if is_anchor_region:
                validate_observed_anchor_region_report_v3(payload["surface_anchor_region_report"],
                                                          reference_frame_id=profile.reference_frame_id)
            elif is_anchor_v2:
                validate_observed_anchor_report_v2(payload["surface_anchor_report"],
                                                   reference_frame_id=profile.reference_frame_id)
            else:
                validate_observed_point_report(payload["observed_point_report"],
                                               reference_frame_id=profile.reference_frame_id)
        if object_id is not None:
            state = payload["object_state"]
            if is_object_support_v2:
                if state.get("support_object_id") not in object_support_body_names:
                    raise ValueError("Object terminal support is outside the deployed allowlist.")
                state = {key: value for key, value in state.items()
                         if key != "support_object_id"}
            if is_object_attitude_v3:
                validate_object_motion_state_v3(state, profile=profile, object_id=object_id,
                    observation_source="mujoco_native_physics", clock_domain_id=object_clock_domain_id,
                    support_object_ids=object_support_body_names)
            else:
                validate_object_motion_state(state, profile=profile,
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
        if is_safe_v2:
            validate_path_constraint_evidence(result["path_constraint_evidence"],
                constraint=parameters["path_constraint"])
            return
        if route.skill_id == OBSERVED_POINT_SKILL_ID:
            report = result["surface_anchor_region_report" if is_anchor_region else
                            "surface_anchor_report" if is_anchor_v2 else "observed_point_report"]
            if is_anchor_region:
                offsets = (observed_anchor_hover_offsets_v4(parameters) if is_anchor_v4
                           else observed_anchor_region_offsets_v3(parameters))
                if (report["source_operation_id"] != parameters["surface_anchor_operation_id"]
                        or report["selected_candidate_index"] >= len(offsets)
                        or list(offsets[report["selected_candidate_index"]]) != list(report["selected_offset_xyz_m"])
                        or report["attempted_candidate_count"] != report["selected_candidate_index"] + 1):
                    raise ValueError("Anchor region result differs from the requested source/offset search.")
            elif is_anchor_v2:
                if report["source_operation_id"] != parameters["surface_anchor_operation_id"]:
                    raise ValueError("Anchor motion result differs from the requested capture operation.")
            else:
                if any(report[key] != parameters[key] for key in ("observation_operation_id", "point_index")):
                    raise ValueError("Observed-point result does not match the requested reference.")
            return
        if is_object_attitude_v3:
            validate_object_motion_state_v3(result["object_state"], parameters=parameters,
                profile=profile, object_id=object_id, observation_source="mujoco_native_physics",
                clock_domain_id=object_clock_domain_id, support_object_ids=object_support_body_names)
        if is_object_support_v2:
            validate_object_motion_state_v2(result["object_state"],
                parameters=parameters, profile=profile, object_id=object_id,
                observation_source="mujoco_native_physics",
                clock_domain_id=object_clock_domain_id)
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
            validate_execution_result_for_parameters if is_safe_v2 or object_id is not None or route.skill_id == OBSERVED_POINT_SKILL_ID else None),
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
    enable_object_localization: bool = False,
    enable_rgbd_capture_v3: bool = False,
    segmentation_model_profile_id: str | None = None,
    segmentation_model_profile_hash: str | None = None,
    segmentation_inference_implementation_hash: str | None = None,
    segmentation_inference_execution_mode: str = "in_process",
    enable_segmentation_batch_v2: bool = False,
    enable_segmentation_geometry_v3: bool = False,
    enable_safe_tcp_pose: bool = False,
    enable_safe_tcp_path_constraints: bool = False,
    enable_tcp_continuation_queries: bool = False,
    enable_tcp_candidate_artifacts: bool = False,
    enable_relation_tcp_continuation_queries: bool = False,
    enable_region_tcp_continuation_queries: bool = False,
    region_tcp_continuation_version: str = "4",
    safe_motion_observation_profile_hash: str | None = None,
    enable_observed_point_motion: bool = False,
    enable_anchor_motion_v2: bool = False,
    enable_anchor_region_motion_v3: bool = False,
    enable_anchor_hover_motion_v4: bool = False,
    enable_spatial_goals: bool = False,
    enable_object_goal_queries: bool = False,
    enable_surface_anchors: bool = False,
    enable_surface_anchor_v2: bool = False,
    enable_surface_anchor_v3: bool = False,
    surface_anchor_selection_policy: str = "prompt_nearest",
    enable_relation_goal_queries: bool = False,
    safe_tcp_motion_profile: str = "local_step",
    enable_tcp_pose_comparison: bool = False,
    enable_object_state: bool = False,
    enable_object_follow_observation: bool = False,
    enable_object_motion: bool = False,
    enable_object_support_motion_v2: bool = False,
    enable_object_attitude_motion_v3: bool = False,
    object_support_body_names: tuple[str, ...] = (),
    enable_grasp_approach: bool = False,
    grasp_contact_binding: MujocoGraspContactBinding | None = None,
    enable_tool_object_relations: bool = False,
    enable_tool_hover_queries: bool = False,
    tool_entity_by_side: Mapping[str, Any] | None = None,
    tool_hover_roles: tuple[str, ...] = (),
    enable_support_release_queries: bool = False,
    support_release_prompt_by_id: Mapping[str, str] | None = None,
    support_release_roles: tuple[str, ...] | None = None,
    support_release_query_version: str = "1",
    enable_grasp_approach_queries: bool = False,
    grasp_query_prompt_by_id: Mapping[str, str] | None = None,
    grasp_query_roles: tuple[str, ...] | None = None,
    grasp_approach_query_version: str = "1",
    tool_hover_region_budget_sec: float | None = None,
    tool_hover_candidate_budget_sec: float | None = None,
) -> MujocoAgentTcpPoseContractBundle:
    """Project one already-constructed no-model application without starting it."""

    if history_profile is not None and not isinstance(
        history_profile,
        MujocoAgentTcpPoseHistoryProfile,
    ):
        raise TypeError("history_profile must be typed or None.")
    if not isinstance(enable_artifact_perception, bool):
        raise TypeError("enable_artifact_perception must be boolean.")
    if type(enable_object_localization) is not bool:
        raise TypeError("enable_object_localization must be boolean.")
    if enable_object_localization and (
        not enable_artifact_perception or segmentation_model_profile_id is None
    ):
        raise ValueError("Object localization requires Artifact perception and the shared model identity.")
    if type(enable_rgbd_capture_v3) is not bool or (enable_rgbd_capture_v3 and not enable_artifact_perception):
        raise ValueError("RGB-D capture v3 requires explicit Artifact perception.")
    if type(enable_observed_point_motion) is not bool:
        raise TypeError("enable_observed_point_motion must be boolean.")
    if enable_observed_point_motion and (not enable_safe_tcp_pose or not enable_artifact_perception
                                         or safe_motion_observation_profile_hash is None):
        raise ValueError("Observed-point motion requires sensor-backed safe motion and original RGB-D perception.")
    if type(enable_anchor_motion_v2) is not bool or (enable_anchor_motion_v2
            and not (enable_observed_point_motion and enable_surface_anchor_v2)):
        raise ValueError("Anchor motion v2 requires observed safe motion and surface anchor v2.")
    if type(enable_anchor_region_motion_v3) is not bool or (enable_anchor_region_motion_v3
            and not enable_anchor_motion_v2):
        raise ValueError("Anchor region v3 requires the opt-in anchor v2 family.")
    if type(enable_anchor_hover_motion_v4) is not bool or (enable_anchor_hover_motion_v4
            and not enable_anchor_motion_v2):
        raise ValueError("Anchor hover v4 requires the opt-in anchor v2 family.")
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
    for name, enabled, version in (
        ("enable_segmentation_batch_v2", enable_segmentation_batch_v2, "2"),
        ("enable_segmentation_geometry_v3", enable_segmentation_geometry_v3, "3"),
    ):
        if type(enabled) is not bool:
            raise TypeError(f"{name} must be boolean.")
        if enabled and (not enable_artifact_perception or segmentation_model_profile_id is None):
            raise ValueError(f"Segmentation batch v{version} requires Artifact perception and a segmentation model.")
        if enabled and any(not isinstance(value, str) or not value or value != value.strip()
                           for value in segmentation_identity):
            raise ValueError(f"Segmentation batch v{version} requires typed model identity strings.")
    if not isinstance(enable_safe_tcp_pose, bool):
        raise TypeError("enable_safe_tcp_pose must be boolean.")
    if safe_motion_observation_profile_hash is not None:
        if (not isinstance(safe_motion_observation_profile_hash, str)
                or len(safe_motion_observation_profile_hash) != 64
                or any(c not in "0123456789abcdef" for c in safe_motion_observation_profile_hash)):
            raise ValueError("Observed safe motion requires an exact observation profile SHA-256.")
        if not enable_safe_tcp_pose or not enable_artifact_perception:
            raise ValueError("Observed safe motion requires the safe Route and RGB-D resources.")
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
    if type(enable_object_support_motion_v2) is not bool or (
            enable_object_support_motion_v2 != bool(object_support_body_names)):
        raise ValueError("Object support v2 requires its explicit nonempty allowlist.")
    if enable_object_support_motion_v2 and not enable_object_motion:
        raise ValueError("Object support v2 requires the original object motion chain.")
    object_support_body_names = tuple(object_support_body_names)
    if (len(set(object_support_body_names)) != len(object_support_body_names)
            or any(not isinstance(name, str) or not name or name.strip() != name
                   for name in object_support_body_names)):
        raise ValueError("Object supports require distinct native identifiers.")
    if not isinstance(enable_safe_tcp_path_constraints, bool) or (
            enable_safe_tcp_path_constraints and not enable_safe_tcp_pose):
        raise ValueError("Safe TCP path constraints require explicitly enabled safe TCP motion.")
    if not isinstance(enable_tcp_continuation_queries, bool) or (enable_tcp_continuation_queries
            and not enable_safe_tcp_path_constraints):
        raise ValueError("Continuation queries require explicitly enabled safe TCP path constraints.")
    if not isinstance(enable_tcp_candidate_artifacts, bool) or (enable_tcp_candidate_artifacts
            and not (enable_tcp_continuation_queries and enable_surface_anchor_v2 and enable_relation_goal_queries)):
        raise ValueError("Candidate Artifacts require continuation queries and the v2 anchor/relation family.")
    if not isinstance(enable_relation_tcp_continuation_queries, bool) or (enable_relation_tcp_continuation_queries
            and not (enable_tcp_continuation_queries and enable_surface_anchor_v2 and enable_relation_goal_queries)):
        raise ValueError("Relation continuation v3 requires continuation queries and the v2 anchor/relation family.")
    if not isinstance(enable_region_tcp_continuation_queries, bool) or (enable_region_tcp_continuation_queries
            and not enable_relation_tcp_continuation_queries):
        raise ValueError("Region continuation v4 requires the explicitly enabled relation continuation v3 family.")
    if (type(region_tcp_continuation_version) is not str or region_tcp_continuation_version not in {"4", "5"}
            or region_tcp_continuation_version != "4" and not enable_region_tcp_continuation_queries):
        raise ValueError("Region continuation version 5 requires explicitly enabled region queries; choose '4' or '5'.")
    if enable_surface_anchor_v3 and (enable_tcp_candidate_artifacts or enable_relation_tcp_continuation_queries):
        raise ValueError("Surface anchor v3 cannot be combined with candidate Artifacts or relation continuations.")

    from agent_gateway.skills.perception.surface_anchor_v2 import INTERIOR_SELECTION_POLICIES
    if type(enable_surface_anchor_v3) is not bool or (enable_surface_anchor_v3 and not (
            enable_surface_anchor_v2 and enable_rgbd_capture_v3 and enable_segmentation_geometry_v3)):
        raise ValueError("Surface anchor v3 requires explicit RGB-D, segmentation v3 and compatible v2 consumers.")

    if not isinstance(enable_object_goal_queries, bool) or (enable_object_goal_queries and not enable_safe_tcp_pose):
        raise ValueError("Object goal queries require explicitly enabled safe TCP geometry.")

    if not isinstance(enable_surface_anchors, bool) or (enable_surface_anchors and not (enable_artifact_perception and enable_object_goal_queries)):
        raise ValueError("Surface anchors require Artifact perception and object goal queries.")

    if not isinstance(enable_relation_goal_queries, bool) or (enable_relation_goal_queries and not enable_surface_anchors):
        raise ValueError("Relation goal queries require explicitly enabled surface anchors.")

    if not isinstance(enable_surface_anchor_v2, bool) or (enable_surface_anchor_v2
            and not (enable_surface_anchors and segmentation_model_profile_id is not None)):
        raise ValueError("Surface anchor v2 requires surface anchors and a segmentation model.")

    if (not isinstance(surface_anchor_selection_policy, str)
            or surface_anchor_selection_policy not in INTERIOR_SELECTION_POLICIES
            or surface_anchor_selection_policy != "prompt_nearest" and not enable_surface_anchor_v2):
        raise ValueError("A non-default surface selection policy requires explicitly enabled v2 anchors.")

    safe_limits = resolve_safe_tcp_motion_limits(safe_tcp_motion_profile, enabled=enable_safe_tcp_pose)
    exact = _exact_application(application)
    component = _application_component_id(application)
    side = component.removesuffix("_arm")
    if side == "left" and (
        history_profile is not None
        or enable_spatial_goals
        or enable_object_goal_queries
        or enable_surface_anchors
        or enable_surface_anchor_v2
        or enable_surface_anchor_v3
        or enable_object_localization
        or enable_safe_tcp_path_constraints
        or enable_tcp_continuation_queries
        or enable_tcp_candidate_artifacts
        or enable_relation_tcp_continuation_queries
        or enable_region_tcp_continuation_queries
        or enable_relation_goal_queries
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
    rgbd_capture = None
    if enable_rgbd_capture_v3:
        from agent_gateway.profiles.rgbd_capture import build_rgbd_capture_registration

        rgbd_capture = build_rgbd_capture_registration(
            rgb_registration=rgb_capture, back_projection=depth_capture.back_projection)
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
    def same_cut_registration(version: str) -> SegmentationRegistration:
        assert segmentation is not None
        return build_segmentation_registration(
            skill_version=version,
            rgb_capture_route_hash=(rgbd_capture.entry.route.semantic_content_hash
                if version == "3" and rgbd_capture is not None else None),
            target_kind=segmentation.profile.target_kind, current_mode=segmentation.profile.current_mode,
            source_id_by_role=segmentation.profile.source_id_by_role,
            model_profile_id=segmentation.profile.model_profile_id,
            model_profile_hash=segmentation.profile.model_profile_hash,
            inference_implementation_hash=segmentation.profile.inference_implementation_hash,
            runtime_session_id=segmentation.binding.runtime_session_id,
            control_domain_id=segmentation.binding.control_domain_id,
            target_binding_id=segmentation.binding.target_binding_id,
            station_manifest_hash=segmentation.binding.station_manifest_hash,
            execution_manifest_hash=segmentation.binding.execution_manifest_hash)

    segmentation_batch = same_cut_registration("2") if enable_segmentation_batch_v2 else None
    segmentation_geometry = same_cut_registration("3") if enable_segmentation_geometry_v3 else None
    localization = (
        build_object_localization_registration(
            target_kind="sim", current_mode="sim_active", local_path_enabled=False,
            inference_execution_mode=segmentation_inference_execution_mode,
            model_profile_id=segmentation_model_profile_id,
            model_profile_hash=segmentation_model_profile_hash,
            inference_implementation_hash=segmentation_inference_implementation_hash,
            runtime_session_id=exact["runtime_session_id"],
            control_domain_id=exact["control_domain_id"],
            target_binding_id=target_binding.binding_id,
            station_manifest_hash=station.semantic_content_hash,
            execution_manifest_hash=execution.semantic_content_hash,
        ) if enable_object_localization else None
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
        binding_suffix="",
    ):
        is_safe_v2 = (motion_definition.skill_id == "motion.move_tcp_to_pose_safe"
                      and motion_definition.skill_version == "2")
        is_anchor_v2 = (motion_definition.skill_id == OBSERVED_POINT_SKILL_ID
                        and motion_definition.skill_version == "2")
        is_anchor_v3 = (motion_definition.skill_id == OBSERVED_POINT_SKILL_ID
                       and motion_definition.skill_version == "3")
        is_anchor_v4 = (motion_definition.skill_id == OBSERVED_POINT_SKILL_ID
                       and motion_definition.skill_version == "4")
        schema_builder = (safe_tcp_pose_target_schema_entry_v2 if is_safe_v2
                          else tcp_pose_target_schema_entry_v1)
        def object_motion_schema():
            kwargs = dict(
                definition=motion_definition, profile=motion_profile,
                object_id=grasp_contact_binding.target_body_name,
                observation_source="mujoco_native_physics",
                clock_domain_id="mujoco-native-physics:" + exact["runtime_session_id"],
            )
            if motion_definition.skill_version == "3":
                return grasped_object_move_schema_v3(
                    **kwargs, support_object_ids=object_support_body_names)
            if motion_definition.skill_version == "2":
                return grasped_object_place_schema_v2(
                    **kwargs, support_object_ids=object_support_body_names)
            return grasped_object_move_schema(**kwargs)

        motion_schema = (observed_anchor_hover_schema_v4(definition=motion_definition, profile=motion_profile)
            if is_anchor_v4 else observed_anchor_region_schema_v3(definition=motion_definition, profile=motion_profile)
            if is_anchor_v3 else observed_anchor_schema_v2(definition=motion_definition, profile=motion_profile)
            if is_anchor_v2 else observed_point_schema(definition=motion_definition, profile=motion_profile)
            if motion_definition.skill_id == OBSERVED_POINT_SKILL_ID else object_motion_schema()
            if motion_definition.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID else schema_builder(
            **({} if is_safe_v2 else {"skill_id": motion_definition.skill_id}),
            skill_definition_hash=motion_definition.semantic_content_hash,
            parameter_profile_id=motion_profile.profile_id,
            parameter_profile_hash=motion_profile.semantic_content_hash,
            allowed_component_ids=(component,),
            allowed_reference_frame_ids=(profile.reference_frame_id,),
            max_requested_duration_ms=None,
            workspace_bounds_by_component={component: profile.workspace_bounds},
            # Grasp approach and contact accept only this deployment's frame
            # lineage; publish it as constants so clients need not copy hashes.
            **({"exact_lineage": {name: getattr(motion_profile, name) for name in (
                "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
                "tool_config_hash", "calibration_hash")}}
               if motion_definition.skill_id in {"motion.approach_grasp_pose",
                                                 "motion.approach_grasp_contact"} else {}),
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
            skill_version=motion_definition.skill_version,
            route_version=motion_definition.skill_version,
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
            deployment_binding_id=binding_prefix + exact["runtime_session_id"] + binding_suffix,
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
    safe_contract = safe_contract_v2 = grasp_contract = contact_contract = object_contract = object_support_contract_v2 = observed_point_contract = observed_anchor_contract_v2 = observed_anchor_region_contract_v3 = observed_anchor_hover_contract_v4 = None
    if enable_safe_tcp_pose:
        local_safe_profile = replace(profile,
            profile_id=("mujoco.agent_safe_tcp_pose.v1" if safe_motion_observation_profile_hash is None
                else "mujoco.agent_safe_tcp_pose.rgbd." + safe_motion_observation_profile_hash),
            max_control_cycles=300)
        safe_profile = replace(local_safe_profile,
            profile_id=(local_safe_profile.profile_id
                if safe_tcp_motion_profile == "local_step" or safe_motion_observation_profile_hash is not None
                else f"mujoco.agent_safe_tcp_pose.{safe_tcp_motion_profile}.v1"),
            max_translation_delta_m=safe_limits.max_translation_delta_m,
            max_rotation_delta_rad=safe_limits.max_rotation_delta_rad,
            max_control_cycles=safe_limits.max_control_cycles,
            translation_tolerance_m=safe_limits.translation_tolerance_m,
            rotation_tolerance_rad=safe_limits.rotation_tolerance_rad,
            normalize_ik_objective=safe_limits.normalize_ik_objective)
        item = planned_motion_contract(
            safe_tcp_pose_target_skill_definition_v1(
                controlled_component_ids=(component,)
            ),
            safe_profile,
            route_id=MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V1,
            executor_id=MUJOCO_AGENT_SAFE_TCP_POSE_EXECUTOR_ID_V1,
            binding_prefix="sim.mujoco.agent_safe_tcp_pose:",
            duration_ms=safe_limits.max_operation_duration_ms,
        )
        safe_contract = item[3]
        planned_motions.append(item)
        if enable_safe_tcp_path_constraints:
            safe_profile_v2 = replace(safe_profile, profile_id=safe_profile.profile_id + ".path_axis.v2")
            item = planned_motion_contract(
                safe_tcp_pose_target_skill_definition_v2(controlled_component_ids=(component,)),
                safe_profile_v2, route_id=MUJOCO_AGENT_SAFE_TCP_POSE_ROUTE_ID_V2,
                executor_id=MUJOCO_AGENT_SAFE_TCP_POSE_EXECUTOR_ID_V2,
                binding_prefix="sim.mujoco.agent_safe_tcp_pose:", binding_suffix=".v2",
                duration_ms=safe_limits.max_operation_duration_ms)
            safe_contract_v2 = item[3]
            planned_motions.append(item)
    if enable_observed_point_motion:
        item = planned_motion_contract(
            observed_point_definition(controlled_component_ids=(component,)),
            local_safe_profile,
            route_id=MUJOCO_OBSERVED_POINT_ROUTE_ID,
            executor_id="mujoco-observed-point-safe-v1",
            binding_prefix="sim.mujoco.agent_observed_point:", duration_ms=30_000)
        observed_point_contract = item[3]
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
                stable_measurement_count=max(
                    3, math.ceil(MUJOCO_AGENT_OBJECT_MOVE_STABLE_SEC * decision_hz))),
            route_id=MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V1,
            executor_id="mujoco-grasped-object-motion-v1",
            binding_prefix="sim.mujoco.agent_object_motion:",
            # Debug-phase bound: per-step held-object guards make one carry
            # take ~60-110 s of wall time; tighten once the guard cost drops.
            duration_ms=MUJOCO_AGENT_OBJECT_MOVE_MAX_OPERATION_MS,
            path_search_time_budget_sec=15.0,
        )
        object_contract = item[3]
        planned_motions.append(item)
    object_attitude_contract_v3 = None
    if enable_object_attitude_motion_v3:
        if not enable_object_motion or not object_support_body_names:
            raise ValueError("Object inclination v3 requires observed object motion and named supports.")
        attitude_identity = canonical_content_hash({
            "target_contact_binding": grasp_contact_binding.to_payload(),
            "maximum_inclination_deviation_rad": OBJECT_INCLINATION_DEVIATION_RAD_V3,
            "yaw": "free_about_world_up", "anchor": "original_grasp_generation",
        }, domain="mujoco.object_inclination_policy.v3")
        item = planned_motion_contract(
            grasped_object_move_definition_v3(controlled_component_ids=(component,)),
            replace(object_contract.motion_profile,
                profile_id="mujoco.agent_object_attitude.v3." + attitude_identity),
            route_id=MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V3,
            executor_id="mujoco-grasped-object-attitude-v3",
            binding_prefix="sim.mujoco.agent_object_attitude_v3:",
            duration_ms=MUJOCO_AGENT_OBJECT_MOVE_MAX_OPERATION_MS,
            path_search_time_budget_sec=15.0,
        )
        object_attitude_contract_v3 = item[3]
        planned_motions.append(item)
    if enable_object_support_motion_v2:
        item = planned_motion_contract(
            grasped_object_place_definition_v2(controlled_component_ids=(component,)),
            replace(object_contract.motion_profile,
                profile_id="mujoco.agent_object_support_motion.v2." +
                           grasp_contact_binding.semantic_content_hash),
            route_id=MUJOCO_AGENT_OBJECT_MOVE_ROUTE_ID_V2,
            executor_id="mujoco-grasped-object-support-motion-v2",
            binding_prefix="sim.mujoco.agent_object_support_motion_v2:",
            duration_ms=MUJOCO_AGENT_OBJECT_MOVE_MAX_OPERATION_MS,
            path_search_time_budget_sec=15.0,
        )
        object_support_contract_v2 = item[3]
        planned_motions.append(item)
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

    object_goal_query_registration = None
    if enable_object_goal_queries:
        from .object_goal_query import build_object_goal_query_registration

        object_goal_query_registration = build_object_goal_query_registration(
            base_route=state_route, base_binding=state_binding,
            reference_frame_id=profile.reference_frame_id,
            planning_profile_hash=safe_contract.motion_profile.semantic_content_hash,
        )

    tcp_continuation_query_registration = None
    if enable_tcp_continuation_queries:
        from .tcp_continuation_query import build_tcp_continuation_query_registration

        tcp_continuation_query_registration = build_tcp_continuation_query_registration(
            base_route=state_route, base_binding=state_binding,
            reference_frame_id=profile.reference_frame_id,
            planning_profile_hash=safe_contract_v2.motion_profile.semantic_content_hash)
    surface_anchor_registrations = ()
    if enable_surface_anchors:
        from agent_gateway.profiles.surface_anchor import (
            build_surface_anchor_registrations,
        )

        surface_anchor_registrations = build_surface_anchor_registrations(
            back_projection=depth_capture.back_projection,
            object_query=object_goal_query_registration,
            spatial_goal=spatial_goal_registration,
            enable_relation_queries=enable_relation_goal_queries,
        )
        if enable_surface_anchor_v2:
            surface_anchor_registrations += build_surface_anchor_registrations(
                back_projection=depth_capture.back_projection,
                object_query=object_goal_query_registration,
                spatial_goal=spatial_goal_registration,
                enable_relation_queries=enable_relation_goal_queries,
                version="2", segmentation=segmentation,
                selection_policy=surface_anchor_selection_policy,
            )
        if enable_surface_anchor_v3:
            from agent_gateway.profiles.surface_anchor import (
                build_multi_source_anchor_consumer,
            )
            from agent_gateway.profiles.surface_anchor_v3 import (
                build_surface_anchor_registration_v3,
            )

            paired_anchor = build_surface_anchor_registration_v3(
                back_projection=depth_capture.back_projection, segmentation=segmentation_geometry,
                capture_rgb=rgbd_capture, selection_policy=surface_anchor_selection_policy)
            original_anchor = next(item for item in surface_anchor_registrations
                if item.skill.definition.skill_id == "perception.capture_surface_anchor"
                and item.skill.definition.skill_version == "2")
            surface_anchor_registrations = tuple(
                build_multi_source_anchor_consumer(registration=item,
                    capture_registrations=(original_anchor, paired_anchor))
                if item.skill.definition.skill_version == "2"
                and item.skill.definition.skill_id != "perception.capture_surface_anchor" else item
                for item in surface_anchor_registrations) + (paired_anchor,)

    if enable_anchor_motion_v2:
        capture = [item for item in surface_anchor_registrations
                   if item.skill.definition.skill_id == "perception.capture_surface_anchor"
                   and item.skill.definition.skill_version == "2"]
        if len(capture) != 1:
            raise ValueError("Anchor motion v2 requires one exact surface anchor v2 producer.")
        item = planned_motion_contract(
            observed_anchor_definition_v2(controlled_component_ids=(component,)),
            safe_profile,
            route_id=MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V2,
            executor_id="mujoco-observed-anchor-safe-v2",
            binding_prefix="sim.mujoco.agent_observed_anchor:", binding_suffix=".v2",
            duration_ms=safe_limits.max_operation_duration_ms)
        observed_anchor_contract_v2 = item[3]
        planned_motions.append(item)

    if enable_anchor_region_motion_v3:
        item = planned_motion_contract(
            observed_anchor_region_definition_v3(controlled_component_ids=(component,)),
            safe_profile,
            route_id=MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V3,
            executor_id="mujoco-observed-anchor-region-safe-v3",
            binding_prefix="sim.mujoco.agent_observed_anchor_region:", binding_suffix=".v3",
            duration_ms=safe_limits.max_operation_duration_ms)
        observed_anchor_region_contract_v3 = item[3]
        planned_motions.append(item)

    if enable_anchor_hover_motion_v4:
        item = planned_motion_contract(
            observed_anchor_hover_definition_v4(controlled_component_ids=(component,)),
            safe_profile,
            route_id=MUJOCO_OBSERVED_ANCHOR_ROUTE_ID_V4,
            executor_id="mujoco-observed-anchor-hover-safe-v4",
            binding_prefix="sim.mujoco.agent_observed_anchor_hover:", binding_suffix=".v4",
            duration_ms=safe_limits.max_operation_duration_ms)
        observed_anchor_hover_contract_v4 = item[3]
        planned_motions.append(item)

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
    candidate_query_registrations = ()
    if enable_tcp_candidate_artifacts:
        from agent_gateway.profiles.pose_candidate_set import (
            build_candidate_continuation_registration,
            build_relation_candidate_registration,
        )

        relation = next(item for item in surface_anchor_registrations
            if item.skill.definition.skill_id == "perception.query_relation_paths"
            and item.skill.definition.skill_version == "2")
        producer = build_relation_candidate_registration(relation_registration=relation)
        consumer = build_candidate_continuation_registration(
            continuation_registration=tcp_continuation_query_registration, relation_registration=producer)
        candidate_query_registrations = (producer, consumer)
    if enable_relation_tcp_continuation_queries:
        from agent_gateway.profiles.relation_continuation_query import (
            build_relation_continuation_registration,
        )

        capture = next(item for item in surface_anchor_registrations
            if item.skill.definition.skill_id == "perception.capture_surface_anchor"
            and item.skill.definition.skill_version == "2")
        relation = next(item for item in surface_anchor_registrations
            if item.skill.definition.skill_id == "perception.query_relation_paths"
            and item.skill.definition.skill_version == "2")
        surface_anchor_registrations += (build_relation_continuation_registration(
            capture=capture, relation=relation, continuation=tcp_continuation_query_registration),)
        if enable_region_tcp_continuation_queries:
            from agent_gateway.profiles.region_continuation_query import (
                build_region_continuation_registration,
            )

            surface_anchor_registrations += (build_region_continuation_registration(
                capture=capture, relation=relation, continuation=tcp_continuation_query_registration,
                version=region_tcp_continuation_version),)
    tool_entity_registrations = ()
    if enable_tool_object_relations or enable_tool_hover_queries:
        from agent_gateway.profiles.tool_entity import (
            build_tool_hover_registration,
            build_tool_object_relation_registration,
        )

        if (segmentation is None or depth_capture is None or depth_capture.back_projection is None
                or not tool_entity_by_side):
            raise ValueError("MuJoCo tool-entity Skills need segmentation, aligned RGB-D and a tool entity.")
        if enable_tool_object_relations:
            tool_entity_registrations += (build_tool_object_relation_registration(
                segmentation=segmentation, back_projection=depth_capture.back_projection,
                roles=depth_capture.back_projection.profile.roles, tool_entity_by_side=tool_entity_by_side),)
        if enable_tool_hover_queries:
            anchors = [item for item in surface_anchor_registrations
                       if item.skill.definition.skill_id == "perception.capture_surface_anchor"
                       and item.skill.definition.skill_version == "2"]
            safe_entry = next((item for item in planned_motions
                               if item[2].skill_id == "motion.move_tcp_to_pose_safe"
                               and item[2].skill_version == "1"), None)
            if (len(anchors) != 1 or safe_entry is None or safe_motion_observation_profile_hash is None
                    or not tool_hover_roles):
                raise ValueError("MuJoCo tool hover queries need surface anchor v2, the safe TCP Route "
                                 "and the observed-scene planner.")
            hover_side = component.removesuffix("_arm")
            tool_entity_registrations += (build_tool_hover_registration(
                capture_registration=anchors[0], back_projection=depth_capture.back_projection,
                component_side_by_id={component: hover_side}, roles=tuple(tool_hover_roles),
                tool_entity_by_side=tool_entity_by_side,
                # The private plan-only planner shares the safe Route's observed
                # scene, motion profile and envelope; both identities bind here.
                planner_profile_hash=canonical_content_hash({
                    "planner": "mujoco_private_collision_checked_joint_path_v1",
                    "observed_scene": safe_motion_observation_profile_hash,
                    "safe_motion_route": safe_entry[2].semantic_content_hash},
                    domain="agenticrobot.mujoco.tool_hover_planner_binding.v1"),
                **({} if tool_hover_region_budget_sec is None
                   else {"region_budget_sec": tool_hover_region_budget_sec}),
                **({} if tool_hover_candidate_budget_sec is None
                   else {"candidate_budget_sec": tool_hover_candidate_budget_sec})),)
            if tool_entity_registrations[-1].reference_frame_id != safe_entry[3].motion_profile.reference_frame_id:
                # The selected pose is submitted unchanged to the safe TCP Route.
                raise ValueError("MuJoCo tool hover results must use the safe TCP Route's public frame.")
    support_release_query_registration = None
    if type(enable_support_release_queries) is not bool:
        raise TypeError("enable_support_release_queries must be boolean.")
    if enable_support_release_queries:
        from .support_release_query import build_support_release_query_registration

        prompts = dict(support_release_prompt_by_id or {})
        object_entry = next((item for item in planned_motions
                             if item[2].skill_id == "motion.move_grasped_object_to_pose"
                             and item[2].skill_version == ("3" if support_release_query_version == "4" else "1")), None)
        if (segmentation is None or depth_capture is None or depth_capture.back_projection is None
                or object_entry is None or not object_support_body_names):
            raise ValueError("Support release queries need segmentation, aligned RGB-D, object motion "
                             "and named supports.")
        if not prompts or not set(prompts) <= set(object_support_body_names):
            raise ValueError("Support release prompts must name admitted support bodies.")
        roles = tuple(depth_capture.back_projection.profile.roles)
        if support_release_roles is None:
            # The held hand's wrist camera sees the support past the arm; keep
            # every other bound view as a measured alternative.
            wrist = "wrist_" + component.removesuffix("_arm")
            roles = tuple(sorted(roles, key=lambda role: role != wrist))
        else:
            roles = tuple(support_release_roles)
        support_release_query_registration = build_support_release_query_registration(
            segmentation=segmentation, back_projection=depth_capture.back_projection, roles=roles,
            prompt_by_support_id=prompts, object_motion_contract=object_entry[3],
            object_motion_route=object_entry[2], skill_version=support_release_query_version)
    elif support_release_prompt_by_id or support_release_roles is not None:
        raise ValueError("Support release prompts and role require enable_support_release_queries.")
    grasp_approach_query_registration = None
    if type(enable_grasp_approach_queries) is not bool:
        raise TypeError("enable_grasp_approach_queries must be boolean.")
    if enable_grasp_approach_queries:
        from .grasp_approach_query import build_grasp_approach_query_registration

        approach_entry = next((item for item in planned_motions
                               if item[2].skill_id == "motion.approach_grasp_pose"
                               and item[2].skill_version == "1"), None)
        if (segmentation is None or depth_capture is None or depth_capture.back_projection is None
                or approach_entry is None or not grasp_query_prompt_by_id):
            raise ValueError("Grasp approach queries need segmentation, aligned RGB-D, the grasp "
                             "approach Route and one prompt per object.")
        grasp_approach_query_registration = build_grasp_approach_query_registration(
            segmentation=segmentation, back_projection=depth_capture.back_projection,
            roles=(tuple(grasp_query_roles) if grasp_query_roles is not None
                   else tuple(depth_capture.back_projection.profile.roles)),
            prompt_by_object_id=grasp_query_prompt_by_id,
            grasp_approach_contract=approach_entry[3], grasp_approach_route=approach_entry[2],
            skill_version=grasp_approach_query_version)
    elif grasp_query_prompt_by_id or grasp_query_roles is not None:
        raise ValueError("Grasp query prompts and roles require enable_grasp_approach_queries.")
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
        state_definition,
    ]
    schema_items = [
        schema_entry,
        state_schema,
    ]
    if spatial_goal_registration is not None:
        definition_items.append(spatial_goal_registration.skill.definition)
        schema_items.append(spatial_goal_registration.skill.schema_entry)
    if object_goal_query_registration is not None:
        definition_items.append(object_goal_query_registration.skill.definition)
        schema_items.append(object_goal_query_registration.skill.schema_entry)
    for registration in (*surface_anchor_registrations, *candidate_query_registrations, *tool_entity_registrations,
                         *((support_release_query_registration,)
                           if support_release_query_registration is not None else ()),
                         *((grasp_approach_query_registration,)
                           if grasp_approach_query_registration is not None else ())):
        definition_items.append(registration.skill.definition)
        schema_items.append(registration.skill.schema_entry)
    if tcp_continuation_query_registration is not None:
        definition_items.append(tcp_continuation_query_registration.skill.definition)
        schema_items.append(tcp_continuation_query_registration.skill.schema_entry)
    if localization is not None:
        definition_items.append(localization.skill.definition)
        schema_items.append(localization.skill.schema_entry)
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
    if rgbd_capture is not None:
        definition_items.append(rgbd_capture.skill.definition)
        schema_items.append(rgbd_capture.skill.schema_entry)
    if segmentation is not None:
        definition_items.append(segmentation.skill.definition)
        schema_items.append(segmentation.skill.schema_entry)
    for batch in (segmentation_batch, segmentation_geometry):
        if batch is not None:
            definition_items.append(batch.skill.definition)
            schema_items.append(batch.skill.schema_entry)
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
    if rgbd_capture is not None:
        route_items.append(rgbd_capture.entry)
    if segmentation is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=segmentation.skill.routes[0],
                deployment_binding=segmentation.binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=segmentation.availability_hash,
            )
        )
    for batch in (segmentation_batch, segmentation_geometry):
        if batch is not None:
            route_items.append(SkillRouteManifestEntry(route=batch.skill.routes[0],
                deployment_binding=batch.binding.to_ref(), evaluation_plan_hash=None,
                availability_hash=batch.availability_hash))
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
    if object_goal_query_registration is not None:
        route_items.append(object_goal_query_registration.entry)
    route_items.extend(registration.entry for registration in (*surface_anchor_registrations, *candidate_query_registrations,
                                                               *tool_entity_registrations))
    if support_release_query_registration is not None:
        route_items.append(support_release_query_registration.entry)
    if grasp_approach_query_registration is not None:
        route_items.append(grasp_approach_query_registration.entry)
    if tcp_continuation_query_registration is not None:
        route_items.append(tcp_continuation_query_registration.entry)
    if localization is not None:
        route_items.append(SkillRouteManifestEntry(
            route=localization.skill.routes[0],
            deployment_binding=localization.binding.to_ref(),
            evaluation_plan_hash=None, availability_hash=localization.availability_hash))
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
        for batch in (segmentation_batch, segmentation_geometry):
            if batch is not None and key == (batch.skill.definition.skill_id, batch.skill.definition.skill_version):
                explicit_contracts += (batch.skill.io_contract,)
        if rgbd_capture is not None and key == ("perception.capture_rgb", "3"):
            explicit_contracts += (rgbd_capture.skill.io_contract,)
        explicit_contracts += tuple(registration.skill.io_contract
            for registration in (object_goal_query_registration, localization, tcp_continuation_query_registration,
                                 *surface_anchor_registrations, *candidate_query_registrations,
                                 *tool_entity_registrations, support_release_query_registration,
                                 grasp_approach_query_registration)
            if registration is not None and key == (
                registration.skill.definition.skill_id, registration.skill.definition.skill_version))
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
        state_validator,
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
    if rgbd_capture is not None:
        validator_items.append(rgbd_capture.payload_validator)
    if spatial_goal_registration is not None:
        validator_items.append(spatial_goal_registration.payload_validator)
    if object_goal_query_registration is not None:
        validator_items.append(object_goal_query_registration.payload_validator)
    validator_items.extend(registration.payload_validator for registration in (*surface_anchor_registrations, *candidate_query_registrations,
                                                                               *tool_entity_registrations))
    if support_release_query_registration is not None:
        validator_items.append(support_release_query_registration.payload_validator)
    if grasp_approach_query_registration is not None:
        validator_items.append(grasp_approach_query_registration.payload_validator)
    if tcp_continuation_query_registration is not None:
        validator_items.append(tcp_continuation_query_registration.payload_validator)
    if localization is not None:
        validator_items.append(localization.payload_validator)
    if tcp_pose_comparison_registration is not None:
        validator_items.append(tcp_pose_comparison_registration.payload_validator)
    if object_state_registration is not None:
        validator_items.append(object_state_registration.payload_validator)
    if segmentation is not None:
        validator_items.append(segmentation.payload_validator)
    for batch in (segmentation_batch, segmentation_geometry):
        if batch is not None:
            validator_items.append(batch.payload_validator)
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
                   | ({"object_support_body_names": object_support_body_names}
                      if safe_route.skill_version == "2" else {})
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
        state_deployment_binding=state_binding,
        motion_profile=profile,
        evaluation_plan=evaluation_plan,
        payload_validators=validators,
        station_manifest=station,
        execution_manifest=execution,
        safe_tcp_pose_contract=safe_contract,
        safe_tcp_pose_v2_contract=safe_contract_v2,
        observed_point_contract=observed_point_contract,
        observed_anchor_contract_v2=observed_anchor_contract_v2,
        observed_anchor_region_contract_v3=observed_anchor_region_contract_v3,
        observed_anchor_hover_contract_v4=observed_anchor_hover_contract_v4,
        spatial_goal_registration=spatial_goal_registration,
        object_goal_query_registration=object_goal_query_registration,
        tcp_continuation_query_registration=tcp_continuation_query_registration,
        candidate_query_registrations=candidate_query_registrations,
        surface_anchor_registrations=surface_anchor_registrations,
        tool_entity_registrations=tool_entity_registrations,
        support_release_query_registration=support_release_query_registration,
        grasp_approach_query_registration=grasp_approach_query_registration,
        tcp_pose_comparison_registration=tcp_pose_comparison_registration,
        object_state_registration=object_state_registration,
        grasp_approach_contract=grasp_contract,
        grasp_contact_contract=contact_contract,
        object_motion_contract=object_contract,
        object_support_motion_contract_v2=object_support_contract_v2,
        object_attitude_motion_contract_v3=object_attitude_contract_v3,
        object_support_body_names=object_support_body_names,
        depth_capture=depth_capture,
        rgb_capture=rgb_capture,
        rgbd_capture=rgbd_capture,
        segmentation=segmentation,
        segmentation_batch=segmentation_batch,
        segmentation_geometry=segmentation_geometry,
        localization=localization,
        placement=placement,
        history_profile=history_profile,
        history_source_binding=history_source_binding,
        history_deployment_binding=history_binding,
    )


__all__ = [
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1",
    "MUJOCO_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1",
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

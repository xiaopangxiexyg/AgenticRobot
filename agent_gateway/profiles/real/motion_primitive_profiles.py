"""Exact real-Robot Routes and bindings for reviewed TCP motion Skills.

Construction is pure and default-off.  It requires an existing real-observe
attachment, lower motion-port binding, exact frame graph, and at least one
lower-owned safety-envelope reference.  No device, control loop, socket, or
motion operation is started here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

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
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.skills.motion.definition import (
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
    cardinal_tcp_step_skill_definition_v1,
    fixed_cardinal_tcp_step_skill_definition_v1,
    fixed_level_tcp_orientation_skill_definition_v1,
    grasp_approach_skill_definition_v1,
    tcp_pose_target_skill_definition_v1,
)
from agent_gateway.skills.motion.schemas import (
    CARDINAL_TCP_STEP_DIRECTIONS_V1,
    cardinal_tcp_step_schema_entry_v1,
    fixed_cardinal_tcp_step_schema_entry_v1,
    fixed_level_tcp_orientation_schema_entry_v1,
    tcp_pose_target_schema_entry_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    LevelTCPOrientationPrimitive,
    TCPPoseDisplacementPrimitive,
    cardinal_tcp_short_step,
    level_tcp_orientation,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPortConfig

from .grasp_approach_plan import (
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    REAL_AGENT_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    RealGraspApproachPlanProfile,
    RealGraspApproachPlanRegistration,
    build_real_grasp_approach_plan_registration,
)
from .localization_deployment import RealObjectLocalizationRegistration
from .observation_gateway import (
    RealTcpPoseObserveAttachmentSpec,
    RealTcpPoseObserveContractBundle,
    build_real_tcp_pose_observe_contract_bundle,
)
from .perception_deployment import RealPerceptionCaptureRegistration
from .station_point_grasp_deployment import StationBoundPointGraspRegistration
from .station_point_grasp_render_deployment import (
    StationBoundPointGraspRenderRegistration,
)

REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1 = "real.rm65.agent_skill_tcp_cardinal_step"
REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1 = "real.rm65.agent_skill_tcp_pose_target"
REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1 = "real.rm65.agent_skill_grasp_approach"
REAL_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1 = (
    "control-station.real-agent-grasp-approach.v1"
)
REAL_AGENT_TCP_POSE_TARGET_EXECUTOR_ID_V1 = (
    "control-station.real-agent-tcp-pose-target.v1"
)
REAL_AGENT_TCP_POSE_TARGET_PARAMETER_PROFILE_ID_V1 = (
    "real-tool-tcp-absolute-grasp-pose-v1"
)
REAL_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1 = (
    "control-station.real-agent-tcp-cardinal-step.v1"
)
REAL_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1 = (
    "real-tool-tcp-cardinal-short-step-v1"
)
REAL_AGENT_PUBLIC_TCP_POSE_SCHEMA_HASH_DOMAIN_V1 = (
    "dexteleop.real_agent_motion.public_tcp_pose_schema.v1"
)
REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1 = {
    skill_id: ("real.rm65.agent_skill_" + skill_id.removeprefix("motion."))
    for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
}
REAL_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "control-station.real-agent-"
        + skill_id.removeprefix("motion.").replace("_", "-")
        + ".v1"
    )
    for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
}
REAL_AGENT_TCP_FIXED_CARDINAL_STEP_DIRECTION_BY_SKILL_ID_V1 = dict(
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1
)
REAL_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1 = "real.rm65.agent_skill_tcp_level_forward"
REAL_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_level_forward_left_30"
)
REAL_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_level_forward_right_30"
)
REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1 = {
    TCP_LEVEL_FORWARD_SKILL_ID_V1: REAL_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: (
        REAL_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1
    ),
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: (
        REAL_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1
    ),
}
REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1 = {
    TCP_LEVEL_FORWARD_SKILL_ID_V1: "center_forward",
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1: "forward_left_30",
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1: "forward_right_30",
}
REAL_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "control-station.real-agent-"
        + skill_id.removeprefix("motion.").replace("_", "-")
        + ".v1"
    )
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
}
REAL_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1 = {
    skill_id: (
        "real-tool-tcp-" + skill_id.removeprefix("motion.").replace("_", "-") + "-v1"
    )
    for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
}


def real_agent_motion_route_ids_v1(
    *,
    include_grasp_approach: bool = False,
    include_grasp_approach_plan: bool = False,
) -> tuple[str, ...]:
    """Return the single authoritative real-motion Route allowlist."""

    optional_routes = (
        *((REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1,) if include_grasp_approach else ()),
        *(
            (REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,)
            if include_grasp_approach_plan
            else ()
        ),
    )
    return tuple(
        sorted(
            {
                REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1,
                REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
                *optional_routes,
                *REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1.values(),
                *REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1.values(),
            }
        )
    )


def _required(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return value


@dataclass(frozen=True)
class RealAgentPrimitiveConvergenceProfile:
    translation_tolerance_m: float = 0.01
    rotation_tolerance_rad: float = 0.08726646259971647
    stable_measurement_count: int = 3
    max_measurement_age_ms: int = 200
    poll_interval_sec: float = 0.01
    hold_confirmation_timeout_sec: float = 2.0

    def __post_init__(self) -> None:
        for field_name in (
            "translation_tolerance_m",
            "rotation_tolerance_rad",
            "poll_interval_sec",
            "hold_confirmation_timeout_sec",
        ):
            value = float(getattr(self, field_name))
            if not math.isfinite(value) or not value > 0.0:
                raise ValueError(f"{field_name} must be positive.")
            object.__setattr__(self, field_name, value)
        for field_name in ("stable_measurement_count", "max_measurement_age_ms"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            self.__dict__,
            domain="dexteleop.real_agent_motion.convergence_profile.v1",
        )


@dataclass(frozen=True)
class RealAgentTcpCardinalStepProfile:
    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    convergence_profile_hash: str
    primitives: tuple[TCPPoseDisplacementPrimitive, ...] = (
        CARDINAL_TCP_SHORT_STEP_PRIMITIVES
    )

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "convergence_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        primitives = tuple(self.primitives)
        if any(
            not isinstance(item, TCPPoseDisplacementPrimitive) for item in primitives
        ):
            raise TypeError("primitives must contain TCP displacement primitives.")
        directions = tuple(item.direction.value for item in primitives)
        if set(directions) != set(CARDINAL_TCP_STEP_DIRECTIONS_V1) or len(
            directions
        ) != len(set(directions)):
            raise ValueError("Profile requires one primitive per cardinal direction.")
        object.__setattr__(
            self,
            "primitives",
            tuple(sorted(primitives, key=lambda item: item.direction.value)),
        )

    @property
    def parameter_profile_id(self) -> str:
        return REAL_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1

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
                **self.__dict__,
                "profile_id": self.parameter_profile_id,
                "controlled_frame_id": "tool_tcp",
                "primitive_catalog_hash": self.primitive_catalog_hash,
                "primitives": tuple(item.to_payload() for item in self.primitives),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
            },
            domain="dexteleop.agent_skill.real_tcp_cardinal_step_profile.v1",
        )

    def primitive(self, direction: str) -> TCPPoseDisplacementPrimitive:
        selected = cardinal_tcp_short_step(direction)
        for item in self.primitives:
            if item.semantic_content_hash == selected.semantic_content_hash:
                return item
        raise ValueError("Direction is absent from the exact primitive profile.")


@dataclass(frozen=True)
class RealAgentTcpPoseTargetProfile:
    """Exact real absolute tool-TCP target profile for grasp-pose proposals."""

    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    convergence_profile_hash: str
    min_speed_scale: float = 0.05
    max_speed_scale: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "convergence_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        minimum = float(self.min_speed_scale)
        maximum = float(self.max_speed_scale)
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or not 0.0 < minimum <= maximum <= 1.0
        ):
            raise ValueError("speed scales must satisfy 0 < min <= max <= 1.")
        object.__setattr__(self, "min_speed_scale", minimum)
        object.__setattr__(self, "max_speed_scale", maximum)

    @property
    def skill_id(self) -> str:
        return TCP_POSE_TARGET_SKILL_ID_V1

    @property
    def parameter_profile_id(self) -> str:
        return REAL_AGENT_TCP_POSE_TARGET_PARAMETER_PROFILE_ID_V1

    @property
    def speed_scale_bounds(self) -> tuple[float, float]:
        return (self.min_speed_scale, self.max_speed_scale)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "profile_id": self.parameter_profile_id,
                "controlled_frame_id": "tool_tcp",
                "target_semantics": "absolute_pose_in_public_reference_frame",
                "execution_transform": "single_frame_graph_projection_at_activation",
                "speed_semantics": "fraction_of_lower_reviewed_trajectory_limits",
            },
            domain="dexteleop.agent_skill.real_tcp_pose_target_profile.v1",
        )


@dataclass(frozen=True)
class RealAgentTcpFixedOrientationProfile:
    skill_id: str
    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    convergence_profile_hash: str
    primitive: LevelTCPOrientationPrimitive

    def __post_init__(self) -> None:
        for field_name in (
            "skill_id",
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "convergence_profile_hash",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        if self.skill_id not in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1:
            raise ValueError("Unsupported fixed-orientation Skill ID.")
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        if not isinstance(self.primitive, LevelTCPOrientationPrimitive):
            raise TypeError("primitive must be a fixed TCP orientation primitive.")
        expected = REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1[
            self.skill_id
        ]
        if self.primitive.heading.value != expected:
            raise ValueError("Skill ID and fixed-orientation primitive disagree.")

    @property
    def parameter_profile_id(self) -> str:
        return REAL_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1[
            self.skill_id
        ]

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **self.__dict__,
                "profile_id": self.parameter_profile_id,
                "controlled_frame_id": "tool_tcp",
                "primitive": self.primitive.to_payload(),
                "anchor_semantics": "lower_measured_pose_at_skill_activation",
                "target_semantics": "absolute_heading_preserve_translation",
            },
            domain="dexteleop.agent_skill.real_tcp_fixed_orientation_profile.v1",
        )


@dataclass(frozen=True)
class RealAgentPrimitiveDeploymentBinding:
    deployment_binding_id: str
    route_hash: str
    runtime_session_id: str
    control_domain_id: str
    target_binding_id: str
    station_manifest_hash: str
    execution_manifest_hash: str
    attachment_hash: str
    motion_port_binding_hash: str
    parameter_profile_hash: str
    convergence_profile_hash: str
    component_id: str
    side: str
    reference_frame_id: str
    reference_frame_spec_hash: str
    execution_reference_frame_id: str
    execution_reference_frame_spec_hash: str
    frame_graph_hash: str
    tool_config_hash: str
    calibration_hash: str
    executor_id: str
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...]
    deployment_binding_version: str = "1"

    def __post_init__(self) -> None:
        for field_name in (
            "deployment_binding_id",
            "route_hash",
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
            "attachment_hash",
            "motion_port_binding_hash",
            "parameter_profile_hash",
            "convergence_profile_hash",
            "component_id",
            "reference_frame_id",
            "reference_frame_spec_hash",
            "execution_reference_frame_id",
            "execution_reference_frame_spec_hash",
            "frame_graph_hash",
            "tool_config_hash",
            "calibration_hash",
            "executor_id",
            "deployment_binding_version",
        ):
            object.__setattr__(
                self,
                field_name,
                _required(getattr(self, field_name), field_name=field_name),
            )
        side = str(self.side).strip().lower()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right.")
        object.__setattr__(self, "side", side)
        refs = tuple(self.safety_envelope_refs)
        if not refs or any(
            not isinstance(item, SkillDeploymentSafetyEnvelopeRef) for item in refs
        ):
            raise ValueError("Real motion binding requires typed safety-envelope refs.")
        object.__setattr__(
            self,
            "safety_envelope_refs",
            tuple(sorted(refs, key=lambda item: item.object_id)),
        )

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            {
                **{
                    name: getattr(self, name)
                    for name in self.__dataclass_fields__
                    if name != "safety_envelope_refs"
                },
                "safety_envelope_refs": tuple(
                    item.to_wire() for item in self.safety_envelope_refs
                ),
                "target_kind": "real",
                "controlled_frame_id": "tool_tcp",
                "source_kind": "agent_skill_via_lower_owned_motion_port",
            },
            domain="dexteleop.agent_skill.real_tcp_primitive_deployment.v1",
        )

    def to_ref(self) -> SkillDeploymentBindingRef:
        return SkillDeploymentBindingRef(
            deployment_binding_id=self.deployment_binding_id,
            deployment_binding_hash=self.semantic_content_hash,
            target_binding_id=self.target_binding_id,
            target_kind="real",
            execution_manifest_hash=self.execution_manifest_hash,
            safety_envelope_refs=self.safety_envelope_refs,
        )


@dataclass(frozen=True)
class RealAgentPrimitiveContractBundle:
    definition_catalog: SkillDefinitionCatalog
    schema_catalog: AgentSkillSchemaCatalog
    route_manifest: SkillRouteManifest
    io_catalog: SkillIOCatalog
    cardinal_profile: RealAgentTcpCardinalStepProfile
    pose_target_profile: RealAgentTcpPoseTargetProfile | None
    orientation_profiles: tuple[RealAgentTcpFixedOrientationProfile, ...]
    deployment_bindings: tuple[RealAgentPrimitiveDeploymentBinding, ...]
    convergence_profile: RealAgentPrimitiveConvergenceProfile
    payload_validators: AgentSkillPayloadValidatorRegistry
    frame_graph: CoordinateFrameGraph
    attachment: RealTcpPoseObserveAttachmentSpec
    tcp_pose_observe: RealTcpPoseObserveContractBundle
    perception_capture: RealPerceptionCaptureRegistration | None = None
    localization_registration: RealObjectLocalizationRegistration | None = None
    station_point_grasp_registration: StationBoundPointGraspRegistration | None = None
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    ) = None
    grasp_approach_plan_registration: RealGraspApproachPlanRegistration | None = None

    def __post_init__(self) -> None:
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
            io_catalog=self.io_catalog,
        )
        self.payload_validators.validate_manifest(self.route_manifest)
        pose_target_enabled = self.pose_target_profile is not None
        grasp_approach_enabled = any(
            entry.route.skill_id == GRASP_APPROACH_SKILL_ID_V1
            for entry in self.route_manifest.skill_routes
        )
        grasp_approach_plan_enabled = self.grasp_approach_plan_registration is not None
        paired_plan_routes_present = (
            any(
                entry.route.skill_id == EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1
                for entry in self.route_manifest.skill_routes
            ),
            any(
                entry.route.route_id == REAL_AGENT_GRASP_APPROACH_PLAN_ROUTE_ID_V1
                for entry in self.route_manifest.skill_routes
            ),
        )
        if paired_plan_routes_present != (
            grasp_approach_plan_enabled,
            grasp_approach_plan_enabled,
        ):
            raise ValueError("Paired grasp-approach planning Routes must coexist.")
        expected_binding_count = (
            10
            + int(pose_target_enabled)
            + int(grasp_approach_enabled)
            + int(grasp_approach_plan_enabled)
        )
        if (
            len(self.orientation_profiles) != 3
            or len(self.deployment_bindings) != expected_binding_count
        ):
            raise ValueError(
                "Real primitive bundle requires one generic cardinal, six "
                "fixed-cardinal and three orientation bindings, plus exactly one "
                "absolute-pose, legacy grasp-approach, and paired-plan execution "
                "binding when their lower envelopes authorize them."
            )
        for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1:
            self.route_manifest.route(skill_id, "1")
        if (
            self.route_manifest.skill_io_catalog_hash
            != self.io_catalog.semantic_content_hash
        ):
            raise ValueError("Real primitive bundle I/O catalog mismatch.")
        if self.tcp_pose_observe.attachment != self.attachment:
            raise ValueError("Real primitive and TCP-pose attachments disagree.")
        public_frame = (
            self.attachment.reference_frame_id,
            self.attachment.reference_frame_spec_hash,
        )
        profile_public_frame = (
            self.cardinal_profile.reference_frame_id,
            self.cardinal_profile.reference_frame_spec_hash,
        )
        if public_frame != profile_public_frame:
            raise ValueError("Real primitive public Station-frame binding mismatch.")
        execution_frame = (
            self.attachment.execution_reference_frame_id,
            self.attachment.execution_reference_frame_spec_hash,
        )
        profile_execution_frame = (
            self.cardinal_profile.execution_reference_frame_id,
            self.cardinal_profile.execution_reference_frame_spec_hash,
        )
        if execution_frame != profile_execution_frame:
            raise ValueError("Real primitive lower execution-frame binding mismatch.")
        if self.pose_target_profile is not None and (
            (
                self.pose_target_profile.reference_frame_id,
                self.pose_target_profile.reference_frame_spec_hash,
            )
            != public_frame
            or (
                self.pose_target_profile.execution_reference_frame_id,
                self.pose_target_profile.execution_reference_frame_spec_hash,
            )
            != execution_frame
        ):
            raise ValueError("Real absolute-pose profile frame bindings disagree.")
        if self.grasp_approach_plan_registration is not None:
            plan_profile = self.grasp_approach_plan_registration.profile
            if (
                plan_profile.reference_frame_id,
                plan_profile.reference_frame_spec_hash,
            ) != public_frame or (
                plan_profile.execution_reference_frame_id,
                plan_profile.execution_reference_frame_spec_hash,
            ) != execution_frame:
                raise ValueError("Grasp-approach plan profile frame bindings disagree.")
        if any(
            (
                profile.reference_frame_id,
                profile.reference_frame_spec_hash,
            )
            != public_frame
            or (
                profile.execution_reference_frame_id,
                profile.execution_reference_frame_spec_hash,
            )
            != execution_frame
            for profile in self.orientation_profiles
        ):
            raise ValueError("Real primitive profile frame bindings disagree.")
        if self.attachment.frame_graph_hash != self.frame_graph.semantic_content_hash:
            raise ValueError("Real primitive attachment frame graph mismatch.")
        optional_grasp = (
            self.localization_registration,
            self.station_point_grasp_registration,
            self.station_point_grasp_render_registration,
        )
        if any(item is not None for item in optional_grasp) and not all(
            item is not None for item in optional_grasp
        ):
            raise ValueError(
                "Real localization, point-grasp, and render registrations must coexist."
            )
        if (
            all(item is not None for item in optional_grasp)
            and self.perception_capture is None
        ):
            raise ValueError("Real point-grasp Routes require perception capture.")
        if grasp_approach_plan_enabled and not all(
            item is not None for item in optional_grasp
        ):
            raise ValueError("Grasp-approach planning requires retained point grasps.")
        expected_route_count = (
            (11 + int(pose_target_enabled) + int(grasp_approach_enabled))
            + (2 if self.perception_capture is not None else 0)
            + (3 if self.localization_registration is not None else 0)
            + (2 if grasp_approach_plan_enabled else 0)
        )
        if len(self.route_manifest.skill_routes) != expected_route_count:
            raise ValueError("Real perception registration and manifest disagree.")

    def binding_for_skill(self, skill_id: str) -> RealAgentPrimitiveDeploymentBinding:
        route = self.route_manifest.route(skill_id, "1").route
        for binding in self.deployment_bindings:
            if binding.route_hash == route.semantic_content_hash:
                return binding
        raise KeyError(skill_id)

    def orientation_profile(
        self,
        skill_id: str,
    ) -> RealAgentTcpFixedOrientationProfile:
        for profile in self.orientation_profiles:
            if profile.skill_id == skill_id:
                return profile
        raise KeyError(skill_id)


def _payload_validator(
    *,
    route: SkillExecutionRoute,
    profile: RealAgentTcpCardinalStepProfile
    | RealAgentTcpFixedOrientationProfile
    | RealAgentTcpPoseTargetProfile
    | RealGraspApproachPlanProfile,
    fixed_cardinal_direction: str | None = None,
) -> AgentSkillPayloadValidatorSet:
    cardinal = isinstance(profile, RealAgentTcpCardinalStepProfile)
    pose_target = isinstance(profile, RealAgentTcpPoseTargetProfile)
    plan_execution = isinstance(profile, RealGraspApproachPlanProfile)
    parameter_fields = (
        {
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
        if cardinal
        else (
            {
                "pose_targets_by_component",
                "reference_frame_id",
                "reference_frame_spec_hash",
                "controlled_frame_id",
                "frame_graph_hash",
                "tool_config_hash",
                "calibration_hash",
            }
            if pose_target
            else (
                {
                    "component_id",
                    "grasp_approach_plan_id",
                    "grasp_approach_plan_hash",
                }
                if plan_execution
                else {
                    "component_id",
                    "reference_frame_id",
                    "reference_frame_spec_hash",
                    "controlled_frame_id",
                    "frame_graph_hash",
                    "tool_config_hash",
                    "calibration_hash",
                    "primitive_hash",
                }
            )
        )
    )

    def validate_parameters(payload: Mapping[str, Any]) -> None:
        payload_fields = set(payload) if isinstance(payload, Mapping) else set()
        allowed_parameter_fields = (
            parameter_fields | {"speed_scale"} if pose_target else parameter_fields
        )
        if (
            not isinstance(payload, Mapping)
            or not parameter_fields <= payload_fields <= allowed_parameter_fields
        ):
            raise ValueError("Real primitive parameters do not match the exact schema.")
        if plan_execution:
            if payload["component_id"] != profile.component_id:
                raise ValueError("Grasp-approach plan component mismatch.")
            plan_hash = payload["grasp_approach_plan_hash"]
            if (
                not isinstance(plan_hash, str)
                or len(plan_hash) != 64
                or any(char not in "0123456789abcdef" for char in plan_hash)
                or payload["grasp_approach_plan_id"]
                != "real-grasp-approach-plan:" + plan_hash
            ):
                raise ValueError("Grasp-approach plan ID/hash mismatch.")
            return
        validation_kwargs: dict[str, Any] = {
            "schema_kind": (
                "cardinal_tcp_step"
                if cardinal
                else (
                    "tcp_pose_target" if pose_target else "fixed_level_tcp_orientation"
                )
            ),
            "parameters": payload,
            "allowed_component_ids": (profile.component_id,),
        }
        if isinstance(profile, RealAgentTcpCardinalStepProfile):
            validation_kwargs["allowed_cardinal_directions"] = (
                tuple(item.direction.value for item in profile.primitives)
                if fixed_cardinal_direction is None
                else (fixed_cardinal_direction,)
            )
        elif isinstance(profile, RealAgentTcpPoseTargetProfile):
            validation_kwargs["speed_scale_bounds"] = profile.speed_scale_bounds
        validate_agent_motion_parameters_v1(**validation_kwargs)
        exact = {
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
        }
        if not pose_target:
            exact["component_id"] = profile.component_id
            exact["primitive_catalog_hash" if cardinal else "primitive_hash"] = (
                profile.primitive_catalog_hash
                if cardinal
                else profile.primitive.semantic_content_hash
            )
        for field_name, expected in exact.items():
            if payload[field_name] != expected:
                raise ValueError(f"Real primitive {field_name} mismatch.")
        if pose_target:
            targets = payload["pose_targets_by_component"]
            if not isinstance(targets, Mapping) or set(targets) != {
                profile.component_id
            }:
                raise ValueError(
                    "Real absolute-pose target must select exactly the bound component."
                )
        if cardinal:
            profile.primitive(str(payload["direction"]))
            if (
                fixed_cardinal_direction is not None
                and payload["direction"] != fixed_cardinal_direction
            ):
                raise ValueError("Fixed cardinal Skill direction changed.")

    def validate_execution_result(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "execution_receipt_refs",
            "final_measured_state_hash",
        }:
            raise ValueError("Real primitive result schema mismatch.")
        refs = payload["execution_receipt_refs"]
        if (
            not isinstance(refs, (tuple, list))
            or not refs
            or len(refs) != len(set(refs))
            or any(not isinstance(item, str) or not item for item in refs)
        ):
            raise ValueError("execution_receipt_refs must be unique and non-empty.")
        _required(
            payload["final_measured_state_hash"],
            field_name="final_measured_state_hash",
        )

    def validate_task_evaluation(payload: Mapping[str, Any]) -> None:
        if not isinstance(payload, Mapping) or set(payload) != {
            "aggregate_evaluation_receipt_ref",
            "task_outcome",
        }:
            raise ValueError("Real primitive task-evaluation schema mismatch.")
        _required(
            payload["aggregate_evaluation_receipt_ref"],
            field_name="aggregate_evaluation_receipt_ref",
        )
        if payload["task_outcome"] not in {"failed", "inconclusive", "succeeded"}:
            raise ValueError("Unsupported real primitive task_outcome.")

    return AgentSkillPayloadValidatorSet(
        validator_id=(
            "payload-validator:real-agent-primitive:"
            + profile.semantic_content_hash
            + ("" if fixed_cardinal_direction is None else ":" + route.skill_id)
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


def build_real_agent_primitive_contract_bundle(
    *,
    attachment: RealTcpPoseObserveAttachmentSpec,
    port_config: RealAgentMotionPortConfig,
    frame_graph: CoordinateFrameGraph,
    primitive_reference_frame_id: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
    perception_capture: RealPerceptionCaptureRegistration | None = None,
    localization_registration: RealObjectLocalizationRegistration | None = None,
    station_point_grasp_registration: StationBoundPointGraspRegistration | None = None,
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    ) = None,
    domain_emergency_stop_available: bool = False,
) -> RealAgentPrimitiveContractBundle:
    """Build exact motion and optional passive Routes without runtime effects."""

    if not isinstance(attachment, RealTcpPoseObserveAttachmentSpec):
        raise TypeError("attachment must be typed.")
    if not isinstance(port_config, RealAgentMotionPortConfig):
        raise TypeError("port_config must be typed.")
    if not isinstance(frame_graph, CoordinateFrameGraph):
        raise TypeError("frame_graph must be typed.")
    if not isinstance(domain_emergency_stop_available, bool):
        raise TypeError("domain_emergency_stop_available must be boolean.")
    refs = tuple(safety_envelope_refs)
    if not refs:
        raise ValueError("Real primitive Routes require lower safety-envelope refs.")
    if attachment.frame_graph_hash != frame_graph.semantic_content_hash:
        raise ValueError("Attachment and primitive frame graph disagree.")
    primitive_frame_id = _required(
        primitive_reference_frame_id,
        field_name="primitive_reference_frame_id",
    )
    try:
        primitive_frame = frame_graph.frame_by_id[primitive_frame_id]
    except KeyError as exc:
        raise ValueError("Primitive reference frame is absent from the graph.") from exc
    if (
        primitive_frame.semantic_parameters.get("axis_contract")
        != STATION_RETARGET_FRAME_CONTRACT_V1
    ):
        raise ValueError(
            "Primitive reference frame must declare the canonical Station axis contract."
        )
    try:
        execution_frame = frame_graph.frame_by_id[port_config.execution_frame_id]
    except KeyError as exc:
        raise ValueError("Lower execution frame is absent from the graph.") from exc
    if execution_frame.semantic_content_hash != port_config.execution_frame_spec_hash:
        raise ValueError("Motion-port execution-frame spec is not the graph member.")
    if attachment.execution_reference_frame_id != port_config.execution_frame_id:
        raise ValueError("Real observation execution_frame_id mismatch.")
    if (
        attachment.execution_reference_frame_spec_hash
        != port_config.execution_frame_spec_hash
    ):
        raise ValueError("Real observation execution-frame spec mismatch.")

    # Compatibility for already-reviewed composition callers that predate the
    # public/execution split: they described the lower execution frame in the
    # old public fields.  Upgrade only that exact legacy shape.  Any other
    # public-frame mismatch remains fail-closed.
    if attachment.reference_frame_id == port_config.execution_frame_id:
        if (
            attachment.reference_frame_spec_hash
            != port_config.execution_frame_spec_hash
        ):
            raise ValueError("Legacy real observation execution-frame spec mismatch.")
        attachment = replace(
            attachment,
            reference_frame_id=primitive_frame.frame_id,
            reference_frame_spec_hash=primitive_frame.semantic_content_hash,
            pose_schema_hash=canonical_content_hash(
                {
                    "representation": "cartesian_pose",
                    "layout": "xyz_quat_xyzw",
                    "quaternion_order": "xyzw",
                    "reference_frame_id": primitive_frame.frame_id,
                    "reference_frame_spec_hash": (
                        primitive_frame.semantic_content_hash
                    ),
                    "controlled_frame_id": "tool_tcp",
                    "frame_graph_hash": frame_graph.semantic_content_hash,
                    "execution_pose_schema_hash": attachment.pose_schema_hash,
                    "projection": "T_public_execution@T_execution_tool_tcp",
                },
                domain=REAL_AGENT_PUBLIC_TCP_POSE_SCHEMA_HASH_DOMAIN_V1,
            ),
            execution_reference_frame_id=port_config.execution_frame_id,
            execution_reference_frame_spec_hash=(port_config.execution_frame_spec_hash),
        )
    elif attachment.reference_frame_id != primitive_frame.frame_id:
        raise ValueError(
            "Real observation public frame must be the primitive Station frame."
        )
    elif attachment.reference_frame_spec_hash != primitive_frame.semantic_content_hash:
        raise ValueError("Real observation public Station-frame spec mismatch.")
    frame_graph.resolve(
        attachment.execution_reference_frame_id,
        attachment.reference_frame_id,
    )

    if set(attachment.component_ids) != set(port_config.component_id_by_side.values()):
        raise ValueError("Observation and motion-port component sets disagree.")
    if len(attachment.component_ids) != 1:
        raise ValueError(
            "The v1 real primitive profile binds exactly one arm component."
        )
    component_id = attachment.component_ids[0]
    side = next(
        side
        for side, bound_component in port_config.component_id_by_side.items()
        if bound_component == component_id
    )
    convergence = convergence_profile or RealAgentPrimitiveConvergenceProfile()
    tcp_pose_observe = build_real_tcp_pose_observe_contract_bundle(
        attachment,
        current_mode="real_active",
    )
    cardinal_profile = RealAgentTcpCardinalStepProfile(
        component_id=component_id,
        side=side,
        reference_frame_id=primitive_frame.frame_id,
        reference_frame_spec_hash=primitive_frame.semantic_content_hash,
        execution_reference_frame_id=port_config.execution_frame_id,
        execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash=attachment.tool_config_hash,
        calibration_hash=attachment.calibration_hash,
        convergence_profile_hash=convergence.semantic_content_hash,
    )
    pose_target_profile = RealAgentTcpPoseTargetProfile(
        component_id=component_id,
        side=side,
        reference_frame_id=primitive_frame.frame_id,
        reference_frame_spec_hash=primitive_frame.semantic_content_hash,
        execution_reference_frame_id=port_config.execution_frame_id,
        execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash=attachment.tool_config_hash,
        calibration_hash=attachment.calibration_hash,
        convergence_profile_hash=convergence.semantic_content_hash,
    )
    grasp_approach_plan_enabled = bool(
        station_point_grasp_registration is not None
        and any(
            REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1 in ref.allowed_route_ids
            for ref in refs
        )
    )
    grasp_approach_plan_registration = (
        build_real_grasp_approach_plan_registration(
            point_grasp_registration=station_point_grasp_registration,
            component_id=component_id,
            side=side,
            reference_frame_id=primitive_frame.frame_id,
            reference_frame_spec_hash=primitive_frame.semantic_content_hash,
            execution_reference_frame_id=port_config.execution_frame_id,
            execution_reference_frame_spec_hash=(port_config.execution_frame_spec_hash),
            frame_graph_hash=frame_graph.semantic_content_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            convergence_profile_hash=convergence.semantic_content_hash,
            safety_envelope_refs=refs,
            max_measured_qpos_age_ms=convergence.max_measurement_age_ms,
            min_speed_scale=pose_target_profile.min_speed_scale,
            max_speed_scale=pose_target_profile.max_speed_scale,
        )
        if grasp_approach_plan_enabled
        else None
    )
    pose_target_definition = tcp_pose_target_skill_definition_v1(
        controlled_component_ids=(component_id,),
    )
    pose_target_schema = tcp_pose_target_schema_entry_v1(
        skill_definition_hash=pose_target_definition.semantic_content_hash,
        parameter_profile_id=pose_target_profile.parameter_profile_id,
        parameter_profile_hash=pose_target_profile.semantic_content_hash,
        allowed_component_ids=(component_id,),
        allowed_reference_frame_ids=(pose_target_profile.reference_frame_id,),
        max_requested_duration_ms=None,
        min_speed_scale=pose_target_profile.min_speed_scale,
        max_speed_scale=pose_target_profile.max_speed_scale,
    )
    grasp_approach_definition = grasp_approach_skill_definition_v1(
        controlled_component_ids=(component_id,),
    )
    grasp_approach_schema = tcp_pose_target_schema_entry_v1(
        skill_id=GRASP_APPROACH_SKILL_ID_V1,
        skill_definition_hash=grasp_approach_definition.semantic_content_hash,
        parameter_profile_id=pose_target_profile.parameter_profile_id,
        parameter_profile_hash=pose_target_profile.semantic_content_hash,
        allowed_component_ids=(component_id,),
        allowed_reference_frame_ids=(pose_target_profile.reference_frame_id,),
        max_requested_duration_ms=None,
        min_speed_scale=pose_target_profile.min_speed_scale,
        max_speed_scale=pose_target_profile.max_speed_scale,
    )
    cardinal_definition = cardinal_tcp_step_skill_definition_v1(
        controlled_component_ids=(component_id,),
    )
    cardinal_schema = cardinal_tcp_step_schema_entry_v1(
        skill_definition_hash=cardinal_definition.semantic_content_hash,
        parameter_profile_id=cardinal_profile.parameter_profile_id,
        parameter_profile_hash=cardinal_profile.semantic_content_hash,
        allowed_component_ids=(component_id,),
        allowed_reference_frame_ids=(cardinal_profile.reference_frame_id,),
        reference_frame_spec_hash=cardinal_profile.reference_frame_spec_hash,
        frame_graph_hash=cardinal_profile.frame_graph_hash,
        tool_config_hash=cardinal_profile.tool_config_hash,
        calibration_hash=cardinal_profile.calibration_hash,
        primitive_catalog_hash=cardinal_profile.primitive_catalog_hash,
        allowed_directions=tuple(
            item.direction.value for item in cardinal_profile.primitives
        ),
    )
    fixed_cardinal_definitions = tuple(
        fixed_cardinal_tcp_step_skill_definition_v1(
            skill_id=skill_id,
            controlled_component_ids=(component_id,),
        )
        for skill_id in FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1
    )
    fixed_cardinal_schemas = tuple(
        fixed_cardinal_tcp_step_schema_entry_v1(
            skill_id=definition.skill_id,
            skill_definition_hash=definition.semantic_content_hash,
            parameter_profile_id=cardinal_profile.parameter_profile_id,
            parameter_profile_hash=cardinal_profile.semantic_content_hash,
            allowed_component_ids=(component_id,),
            allowed_reference_frame_ids=(cardinal_profile.reference_frame_id,),
            reference_frame_spec_hash=cardinal_profile.reference_frame_spec_hash,
            frame_graph_hash=cardinal_profile.frame_graph_hash,
            tool_config_hash=cardinal_profile.tool_config_hash,
            calibration_hash=cardinal_profile.calibration_hash,
            primitive_catalog_hash=cardinal_profile.primitive_catalog_hash,
        )
        for definition in fixed_cardinal_definitions
    )

    orientation_profiles = tuple(
        RealAgentTcpFixedOrientationProfile(
            skill_id=skill_id,
            component_id=component_id,
            side=side,
            reference_frame_id=primitive_frame.frame_id,
            reference_frame_spec_hash=primitive_frame.semantic_content_hash,
            execution_reference_frame_id=port_config.execution_frame_id,
            execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
            frame_graph_hash=frame_graph.semantic_content_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            convergence_profile_hash=convergence.semantic_content_hash,
            primitive=level_tcp_orientation(
                REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1[skill_id]
            ),
        )
        for skill_id in FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1
    )
    orientation_definitions = tuple(
        fixed_level_tcp_orientation_skill_definition_v1(skill_id=profile.skill_id)
        for profile in orientation_profiles
    )
    orientation_schemas = tuple(
        fixed_level_tcp_orientation_schema_entry_v1(
            skill_id=profile.skill_id,
            skill_definition_hash=definition.semantic_content_hash,
            parameter_profile_id=profile.parameter_profile_id,
            parameter_profile_hash=profile.semantic_content_hash,
            allowed_component_ids=(component_id,),
            allowed_reference_frame_ids=(profile.reference_frame_id,),
            reference_frame_spec_hash=profile.reference_frame_spec_hash,
            frame_graph_hash=profile.frame_graph_hash,
            tool_config_hash=profile.tool_config_hash,
            calibration_hash=profile.calibration_hash,
            primitive_hash=profile.primitive.semantic_content_hash,
        )
        for profile, definition in zip(
            orientation_profiles,
            orientation_definitions,
            strict=True,
        )
    )

    def route_for(
        *,
        definition: Any,
        schema: Any,
        route_id: str,
        max_operation_duration_ms: int = 10_000,
    ) -> SkillExecutionRoute:
        route = SkillExecutionRoute(
            skill_id=definition.skill_id,
            skill_version=definition.skill_version,
            skill_definition_hash=definition.semantic_content_hash,
            route_id=route_id,
            route_version="1",
            parameter_schema_hash=schema.parameter_schema_hash,
            execution_result_schema_hash=schema.execution_result_schema_hash,
            task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
            allowed_parameter_profiles=schema.parameter_profiles,
            control_effect_class="motion",
            privilege_tier="t2_real_motion",
            supported_modes=("real_active",),
            target_kind="real",
            may_move_robot=True,
            may_open_publication=False,
            requires_operator_confirmation=True,
            requires_deadman=False,
            max_operation_duration_ms=max_operation_duration_ms,
            cancellation_policy="hold_on_cancel_disconnect_or_timeout",
            requires_task_evaluation=False,
        )
        validate_agent_motion_route_schema_v1(route=route, schema_entry=schema)
        return route

    cardinal_route = route_for(
        definition=cardinal_definition,
        schema=cardinal_schema,
        route_id=REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1,
    )
    pose_target_route = route_for(
        definition=pose_target_definition,
        schema=pose_target_schema,
        route_id=REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1,
        max_operation_duration_ms=30_000,
    )
    grasp_approach_route = route_for(
        definition=grasp_approach_definition,
        schema=grasp_approach_schema,
        route_id=REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1,
        max_operation_duration_ms=60_000,
    )
    pose_target_enabled = any(
        REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1 in ref.allowed_route_ids for ref in refs
    )
    grasp_approach_enabled = any(
        REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1 in ref.allowed_route_ids for ref in refs
    )
    execute_grasp_approach_plan_route = (
        None
        if grasp_approach_plan_registration is None
        else grasp_approach_plan_registration.execute_skill.routes[0]
    )
    fixed_cardinal_routes = tuple(
        route_for(
            definition=definition,
            schema=schema,
            route_id=(
                REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1[
                    definition.skill_id
                ]
            ),
        )
        for definition, schema in zip(
            fixed_cardinal_definitions,
            fixed_cardinal_schemas,
            strict=True,
        )
    )
    orientation_routes = tuple(
        route_for(
            definition=definition,
            schema=schema,
            route_id=REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1[
                profile.skill_id
            ],
        )
        for profile, definition, schema in zip(
            orientation_profiles,
            orientation_definitions,
            orientation_schemas,
            strict=True,
        )
    )
    all_routes = (
        *((pose_target_route,) if pose_target_enabled else ()),
        *((grasp_approach_route,) if grasp_approach_enabled else ()),
        *(
            ()
            if execute_grasp_approach_plan_route is None
            else (execute_grasp_approach_plan_route,)
        ),
        cardinal_route,
        *fixed_cardinal_routes,
        *orientation_routes,
    )
    expected_route_ids = set(
        real_agent_motion_route_ids_v1(
            include_grasp_approach=grasp_approach_enabled,
            include_grasp_approach_plan=grasp_approach_plan_enabled,
        )
    )
    if not pose_target_enabled:
        expected_route_ids.remove(REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1)
    if {route.route_id for route in all_routes} != expected_route_ids:
        raise RuntimeError(
            "Real motion Route construction and safety allowlist drifted."
        )
    for route in all_routes:
        if not any(route.route_id in ref.allowed_route_ids for ref in refs):
            raise ValueError(
                f"Real primitive Route {route.route_id!r} is absent from all lower envelopes."
            )

    motion_port_binding_hash = canonical_content_hash(
        {
            "active_sides": port_config.active_sides,
            "component_id_by_side": dict(port_config.component_id_by_side),
            "execution_frame_id": port_config.execution_frame_id,
            "execution_frame_spec_hash": port_config.execution_frame_spec_hash,
            "exclusive_command_claim": port_config.exclusive_command_claim,
            "command_path": "teleop_session_filter_guard_admission_dispatch",
            "hold_path": "teleop_session_owner_hold",
        },
        domain="dexteleop.real_agent_motion.port_binding.v1",
    )
    profile_by_route_hash = {
        **(
            {pose_target_route.semantic_content_hash: pose_target_profile}
            if pose_target_enabled
            else {}
        ),
        **(
            {grasp_approach_route.semantic_content_hash: pose_target_profile}
            if grasp_approach_enabled
            else {}
        ),
        **(
            {
                execute_grasp_approach_plan_route.semantic_content_hash: (
                    grasp_approach_plan_registration.profile
                )
            }
            if execute_grasp_approach_plan_route is not None
            and grasp_approach_plan_registration is not None
            else {}
        ),
        cardinal_route.semantic_content_hash: cardinal_profile,
        **{
            route.semantic_content_hash: cardinal_profile
            for route in fixed_cardinal_routes
        },
        **{
            route.semantic_content_hash: profile
            for route, profile in zip(
                orientation_routes,
                orientation_profiles,
                strict=True,
            )
        },
    }
    executor_by_route_hash = {
        **(
            {
                pose_target_route.semantic_content_hash: (
                    REAL_AGENT_TCP_POSE_TARGET_EXECUTOR_ID_V1
                )
            }
            if pose_target_enabled
            else {}
        ),
        **(
            {
                grasp_approach_route.semantic_content_hash: (
                    REAL_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1
                )
            }
            if grasp_approach_enabled
            else {}
        ),
        **(
            {
                execute_grasp_approach_plan_route.semantic_content_hash: (
                    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1
                )
            }
            if execute_grasp_approach_plan_route is not None
            else {}
        ),
        cardinal_route.semantic_content_hash: REAL_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1,
        **{
            route.semantic_content_hash: (
                REAL_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1[
                    route.skill_id
                ]
            )
            for route in fixed_cardinal_routes
        },
        **{
            route.semantic_content_hash: (
                REAL_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1[
                    profile.skill_id
                ]
            )
            for route, profile in zip(
                orientation_routes,
                orientation_profiles,
                strict=True,
            )
        },
    }
    bindings = tuple(
        RealAgentPrimitiveDeploymentBinding(
            deployment_binding_id=(
                "real-agent-primitive:"
                + route.route_id
                + ":"
                + attachment.runtime_session_id
            ),
            route_hash=route.semantic_content_hash,
            runtime_session_id=attachment.runtime_session_id,
            control_domain_id=attachment.control_domain_id,
            target_binding_id=attachment.target_binding_id,
            station_manifest_hash=attachment.station_manifest_hash,
            execution_manifest_hash=attachment.execution_manifest_hash,
            attachment_hash=attachment.semantic_content_hash,
            motion_port_binding_hash=motion_port_binding_hash,
            parameter_profile_hash=(
                profile_by_route_hash[route.semantic_content_hash].semantic_content_hash
            ),
            convergence_profile_hash=convergence.semantic_content_hash,
            component_id=component_id,
            side=side,
            reference_frame_id=primitive_frame.frame_id,
            reference_frame_spec_hash=primitive_frame.semantic_content_hash,
            execution_reference_frame_id=port_config.execution_frame_id,
            execution_reference_frame_spec_hash=port_config.execution_frame_spec_hash,
            frame_graph_hash=frame_graph.semantic_content_hash,
            tool_config_hash=attachment.tool_config_hash,
            calibration_hash=attachment.calibration_hash,
            executor_id=executor_by_route_hash[route.semantic_content_hash],
            safety_envelope_refs=refs,
        )
        for route in all_routes
    )
    binding_by_route_hash = {item.route_hash: item for item in bindings}
    optional_grasp = (
        localization_registration,
        station_point_grasp_registration,
        station_point_grasp_render_registration,
    )
    if any(item is not None for item in optional_grasp) and not all(
        item is not None for item in optional_grasp
    ):
        raise ValueError(
            "Real localization, point-grasp, and render registrations must coexist."
        )
    if localization_registration is not None and perception_capture is None:
        raise ValueError("Real point-grasp Routes require perception capture.")
    perception_registrations = list(
        ()
        if perception_capture is None
        else (perception_capture.rgb, perception_capture.depth)
    )
    if localization_registration is not None:
        assert station_point_grasp_registration is not None
        assert station_point_grasp_render_registration is not None
        perception_registrations.extend(
            (
                localization_registration.skill,
                station_point_grasp_registration.skill,
                station_point_grasp_render_registration.skill,
            )
        )
    if grasp_approach_plan_registration is not None:
        perception_registrations.append(grasp_approach_plan_registration.skill)
    perception_registrations = tuple(perception_registrations)
    tcp_pose_definition = tcp_pose_observe.definition_catalog.skill_definitions[0]
    tcp_pose_schema = tcp_pose_observe.schema_catalog.schema_entries[0]
    tcp_pose_entry = tcp_pose_observe.route_manifest.skill_routes[0]
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.real_motion_primitives",
        catalog_version="1",
        skill_definitions=(
            tcp_pose_definition,
            *((pose_target_definition,) if pose_target_enabled else ()),
            *((grasp_approach_definition,) if grasp_approach_enabled else ()),
            *(
                ()
                if grasp_approach_plan_registration is None
                else (grasp_approach_plan_registration.execute_skill.definition,)
            ),
            cardinal_definition,
            *fixed_cardinal_definitions,
            *orientation_definitions,
            *(item.definition for item in perception_registrations),
        ),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.real_motion_primitives",
        catalog_version="1",
        schema_entries=(
            tcp_pose_schema,
            *((pose_target_schema,) if pose_target_enabled else ()),
            *((grasp_approach_schema,) if grasp_approach_enabled else ()),
            *(
                ()
                if grasp_approach_plan_registration is None
                else (grasp_approach_plan_registration.execute_skill.schema_entry,)
            ),
            cardinal_schema,
            *fixed_cardinal_schemas,
            *orientation_schemas,
            *(item.schema_entry for item in perception_registrations),
        ),
    )
    route_items = [tcp_pose_entry]
    route_items.extend(
        SkillRouteManifestEntry(
            route=route,
            deployment_binding=(
                binding_by_route_hash[route.semantic_content_hash].to_ref()
            ),
            evaluation_plan_hash=None,
            availability_hash=canonical_content_hash(
                {
                    "status": "default_off_requires_exclusive_real_skill_owner",
                    "route_hash": route.semantic_content_hash,
                    "deployment_binding_hash": (
                        binding_by_route_hash[
                            route.semantic_content_hash
                        ].semantic_content_hash
                    ),
                },
                domain="dexteleop.agent_skill.real_primitive_availability.v1",
            ),
        )
        for route in all_routes
    )
    if perception_capture is not None:
        for registration, deployment_binding in (
            (perception_capture.rgb, perception_capture.rgb_binding),
            (perception_capture.depth, perception_capture.depth_binding),
        ):
            perception_route = registration.routes[0]
            route_items.append(
                SkillRouteManifestEntry(
                    route=perception_route,
                    deployment_binding=deployment_binding.to_ref(),
                    evaluation_plan_hash=None,
                    availability_hash=(
                        perception_capture.availability_hash_by_route_hash[
                            perception_route.semantic_content_hash
                        ]
                    ),
                )
            )
    if localization_registration is not None:
        assert station_point_grasp_registration is not None
        assert station_point_grasp_render_registration is not None
        for registration, deployment_binding, availability_hash in (
            (
                localization_registration.skill,
                localization_registration.binding,
                localization_registration.availability_hash,
            ),
            (
                station_point_grasp_registration.skill,
                station_point_grasp_registration.binding,
                station_point_grasp_registration.availability_hash,
            ),
            (
                station_point_grasp_render_registration.skill,
                station_point_grasp_render_registration.binding,
                station_point_grasp_render_registration.availability_hash,
            ),
        ):
            route_items.append(
                SkillRouteManifestEntry(
                    route=registration.routes[0],
                    deployment_binding=deployment_binding.to_ref(),
                    evaluation_plan_hash=None,
                    availability_hash=availability_hash,
                )
            )
    if grasp_approach_plan_registration is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=grasp_approach_plan_registration.skill.routes[0],
                deployment_binding=(grasp_approach_plan_registration.binding.to_ref()),
                evaluation_plan_hash=None,
                availability_hash=(grasp_approach_plan_registration.availability_hash),
            )
        )
    artifact_contract_by_skill = {
        (item.definition.skill_id, item.definition.skill_version): item.io_contract
        for item in perception_registrations
    }
    io_catalog = SkillIOCatalog(
        catalog_id="dexteleop.agent_skill_io",
        catalog_version="1",
        io_contracts=tuple(
            artifact_contract_by_skill.get(
                (item.route.skill_id, item.route.skill_version),
                inline_only_skill_io_contract(
                    skill_id=item.route.skill_id,
                    skill_version=item.route.skill_version,
                    skill_definition_hash=item.route.skill_definition_hash,
                    parameter_schema_hash=item.route.parameter_schema_hash,
                    execution_result_schema_hash=(
                        item.route.execution_result_schema_hash
                    ),
                    task_evaluation_schema_hash=(
                        item.route.task_evaluation_schema_hash
                    ),
                ),
            )
            for item in sorted(
                route_items,
                key=lambda entry: (entry.route.skill_id, entry.route.skill_version),
            )
        ),
    )
    manifest = SkillRouteManifest(
        runtime_session_id=attachment.runtime_session_id,
        control_domain_id=attachment.control_domain_id,
        station_manifest_hash=attachment.station_manifest_hash,
        execution_manifest_hash=attachment.execution_manifest_hash,
        skill_definition_catalog_hash=definitions.semantic_content_hash,
        skill_schema_catalog_hash=schemas.semantic_content_hash,
        skill_io_catalog_hash=io_catalog.semantic_content_hash,
        target_binding_id=attachment.target_binding_id,
        target_kind="real",
        current_mode="real_active",
        supported_transport_features=tuple(
            sorted(
                {
                    "authoritative_port_receipt_v1",
                    "idempotency_v1",
                    "own_domain_hold_v1",
                    "own_operation_cancel_v1",
                    "own_operation_status_v1",
                    "safety_profile_selection_v1",
                    *(
                        ("own_domain_emergency_stop_v1",)
                        if domain_emergency_stop_available
                        else ()
                    ),
                }
            )
        ),
        skill_routes=tuple(route_items),
        explicit_exclusions=tuple(
            sorted(
                {
                    "agent_rearm",
                    "agent_resume",
                    "direct_hardware_refresh",
                    "raw_action_submission",
                    "second_dispatch_owner",
                    *(("agent_estop",) if not domain_emergency_stop_available else ()),
                }
            )
        ),
    )
    validators = AgentSkillPayloadValidatorRegistry(
        registry_id=(
            "dexteleop.agent_payload_validators.real_motion_primitives:"
            + attachment.runtime_session_id
        ),
        validators=(
            tcp_pose_observe.payload_validators.validator_for(tcp_pose_entry.route),
            *(
                (
                    _payload_validator(
                        route=pose_target_route,
                        profile=pose_target_profile,
                    ),
                )
                if pose_target_enabled
                else ()
            ),
            *(
                (
                    _payload_validator(
                        route=execute_grasp_approach_plan_route,
                        profile=grasp_approach_plan_registration.profile,
                    ),
                )
                if execute_grasp_approach_plan_route is not None
                and grasp_approach_plan_registration is not None
                else ()
            ),
            *(
                (
                    _payload_validator(
                        route=grasp_approach_route,
                        profile=pose_target_profile,
                    ),
                )
                if grasp_approach_enabled
                else ()
            ),
            _payload_validator(route=cardinal_route, profile=cardinal_profile),
            *tuple(
                _payload_validator(
                    route=route,
                    profile=cardinal_profile,
                    fixed_cardinal_direction=(
                        FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1[route.skill_id]
                    ),
                )
                for route in fixed_cardinal_routes
            ),
            *tuple(
                _payload_validator(route=route, profile=profile)
                for route, profile in zip(
                    orientation_routes,
                    orientation_profiles,
                    strict=True,
                )
            ),
            *(
                ()
                if perception_capture is None
                else perception_capture.payload_validators
            ),
            *(
                ()
                if localization_registration is None
                else (
                    localization_registration.payload_validator,
                    station_point_grasp_registration.payload_validator,
                    station_point_grasp_render_registration.payload_validator,
                )
            ),
            *(
                ()
                if grasp_approach_plan_registration is None
                else (grasp_approach_plan_registration.payload_validator,)
            ),
        ),
    )
    return RealAgentPrimitiveContractBundle(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
        io_catalog=io_catalog,
        cardinal_profile=cardinal_profile,
        pose_target_profile=(pose_target_profile if pose_target_enabled else None),
        orientation_profiles=orientation_profiles,
        deployment_bindings=bindings,
        convergence_profile=convergence,
        payload_validators=validators,
        frame_graph=frame_graph,
        attachment=attachment,
        tcp_pose_observe=tcp_pose_observe,
        perception_capture=perception_capture,
        localization_registration=localization_registration,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        grasp_approach_plan_registration=grasp_approach_plan_registration,
    )


__all__ = [
    "REAL_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1",
    "REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1",
    "REAL_AGENT_TCP_CARDINAL_STEP_EXECUTOR_ID_V1",
    "REAL_AGENT_TCP_CARDINAL_STEP_PARAMETER_PROFILE_ID_V1",
    "REAL_AGENT_TCP_CARDINAL_STEP_ROUTE_ID_V1",
    "REAL_AGENT_TCP_FIXED_CARDINAL_STEP_DIRECTION_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_CARDINAL_STEP_EXECUTOR_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_CARDINAL_STEP_ROUTE_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_EXECUTOR_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_HEADING_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_PARAMETER_PROFILE_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_FIXED_ORIENTATION_ROUTE_ID_BY_SKILL_ID_V1",
    "REAL_AGENT_TCP_LEVEL_FORWARD_LEFT_30_ROUTE_ID_V1",
    "REAL_AGENT_TCP_LEVEL_FORWARD_RIGHT_30_ROUTE_ID_V1",
    "REAL_AGENT_TCP_LEVEL_FORWARD_ROUTE_ID_V1",
    "REAL_AGENT_TCP_POSE_TARGET_EXECUTOR_ID_V1",
    "REAL_AGENT_TCP_POSE_TARGET_PARAMETER_PROFILE_ID_V1",
    "REAL_AGENT_TCP_POSE_TARGET_ROUTE_ID_V1",
    "RealAgentPrimitiveContractBundle",
    "RealAgentPrimitiveConvergenceProfile",
    "RealAgentPrimitiveDeploymentBinding",
    "RealAgentTcpCardinalStepProfile",
    "RealAgentTcpFixedOrientationProfile",
    "RealAgentTcpPoseTargetProfile",
    "build_real_agent_primitive_contract_bundle",
    "real_agent_motion_route_ids_v1",
]

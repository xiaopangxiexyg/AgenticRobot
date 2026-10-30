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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_gateway.profiles.spatial_goal import SpatialGoalRegistration
    from agent_gateway.profiles.tcp_pose_comparison import TcpPoseComparisonRegistration

from agent_gateway.profiles.segmentation import SegmentationRegistration
from agent_gateway.profiles.placement import (
    PlacementRegistration,
    build_placement_registration,
)
from agent_gateway.profiles.tool_entity import (
    ToolEntityRegistration,
    build_tool_hover_registration,
    build_tool_object_relation_registration,
)
from agent_gateway.protocol.skill_catalog import (
    AgentSkillSchemaCatalog,
    SkillDefinitionCatalog,
    SkillDeploymentBindingRef,
    SkillDeploymentSafetyEnvelopeRef,
    SkillExecutionRoute,
    SkillRouteManifest,
    SkillRouteManifestEntry,
    agent_skill_schema_document_hash,
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
from agent_gateway.skills.gripper.deployment import GripperSkillContractBundle
from agent_gateway.skills.motion.definition import (
    ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1,
    EXECUTE_GRASP_APPROACH_PLAN_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_DIRECTION_BY_SKILL_ID_V1,
    FIXED_CARDINAL_TCP_STEP_SKILL_IDS_V1,
    FIXED_LEVEL_TCP_ORIENTATION_SKILL_IDS_V1,
    GRASP_APPROACH_SKILL_ID_V1,
    SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_LEFT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_RIGHT_30_SKILL_ID_V1,
    TCP_LEVEL_FORWARD_SKILL_ID_V1,
    TCP_POSE_TARGET_SKILL_ID_V1,
    grasp_approach_skill_definition_v1,
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
    CARDINAL_TCP_STEP_DIRECTIONS_V1,
    tcp_pose_target_schema_entry_v1,
    validate_safe_tcp_pose_goal_report_v1,
    validate_agent_motion_parameters_v1,
    validate_agent_motion_route_schema_v1,
)
from agent_gateway.skills.motion.observed_point import (
    SKILL_ID as OBSERVED_POINT_SKILL_ID,
    observed_anchor_definition_v2,
    observed_anchor_schema_v2,
    observed_point_definition,
    observed_point_schema,
    validate_observed_anchor_parameters_v2,
    validate_observed_anchor_report_v2,
    validate_observed_point_parameters,
    validate_observed_point_report,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import (
    CARDINAL_TCP_SHORT_STEP_PRIMITIVES,
    LevelTCPOrientationPrimitive,
    TCPPoseDisplacementPrimitive,
    cardinal_tcp_short_step,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPortConfig

from .back_projection import RgbdGeometryRegistration
from .grasp_approach_plan import (
    DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_EXECUTOR_ID_V1,
    REAL_AGENT_EXECUTE_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    REAL_AGENT_GRASP_APPROACH_PLAN_ROUTE_ID_V1,
    RealGraspApproachPlanProfile,
    RealGraspApproachPlanRegistration,
    build_real_grasp_approach_plan_registration,
)
from .grasp_fast_plan import RealFastGraspPlanRegistration, build_real_fast_grasp_plan_registration
from .localization_deployment import RealObjectLocalizationRegistration
from .object_transport import ObjectTransportRegistration
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
REAL_AGENT_SAFE_TCP_POSE_TARGET_ROUTE_ID_V1 = (
    "real.rm65.agent_skill_tcp_pose_target_safe"
)
REAL_AGENT_SAFE_TCP_POSE_TARGET_EXECUTOR_ID_V1 = (
    "control-station.real-agent-tcp-pose-target-safe.v1"
)
REAL_AGENT_OBSERVED_POINT_ROUTE_ID_V1 = "real.rm65.agent_skill_observed_point"
REAL_AGENT_OBSERVED_POINT_EXECUTOR_ID_V1 = "real-agent-observed-point.v1"
REAL_AGENT_OBSERVED_ANCHOR_ROUTE_ID_V2 = "real.rm65.agent_skill_observed_anchor_v2"
REAL_AGENT_OBSERVED_ANCHOR_EXECUTOR_ID_V2 = "real-agent-observed-anchor.v2"
# Held-object motion of whatever the bound arm holds (measured per request).
REAL_AGENT_OBJECT_MOTION_ROUTE_ID_V1 = "real.rm65.agent_skill_move_grasped_object"
REAL_AGENT_OBJECT_MOTION_EXECUTOR_ID_V1 = "real-agent-object-motion.v1"
REAL_AGENT_OBJECT_MOTION_OBSERVATION_SOURCE_V1 = "real_rgbd_held_object_measurement"
# The real Route measures whatever the bound arm holds; the summary is what the
# Agent reads, so it states the measured semantics and the recovery reasons.
REAL_OBJECT_MOTION_SUMMARY_V1 = (
    "Move the object held by this arm's gripper (object_id names 'whatever this arm holds') to an "
    "object-body pose or by translation_delta_xyz_m in the public frame. Before planning the lower "
    "measures the held object from fresh RGB-D views (the points between the closed pads) and the "
    "gripper closure, and refuses unless the object is between the fingers with the gripper closed on it "
    "(object_motion_object_not_between_fingers, object_motion_gripper_not_closed_on_object). The measured "
    "box, fixed in the tool frame, is carried along the whole collision-checked path (table, camera poles, "
    "the other arm and observed obstacles). Phases: lift raises a just-grasped object from its support "
    "(use a translation_delta_xyz_m like [0, 0, 0.05]); carry moves it in free space, keep the object above "
    "obstacles; place lowers it onto the support measured under the goal, whose box bottom must end 0-2 cm "
    "above that support (otherwise object_motion_place_target_below_support / _above_support with the "
    "measured gap). After arrival the object is measured again: the result reports the measured object_state "
    "and held_object_report (box size, grip_assessment centered or crooked with offsets and tilt), or the "
    "operation fails with "
    "object_motion_holding_lost_after_move (dropped or slipped) or object_motion_place_not_on_support. Path "
    "or endpoint rejections (for example path_search_exhausted, endpoint_collision_blocked) mean that goal "
    "is blocked: lift higher first, choose another place pose, or approach from the arm's own side. Recovery "
    "after a drop or a crooked grasp: observe again (perception.describe_tool_object_relations), open the "
    "gripper, re-plan the grasp. Never opens or closes the gripper and never certifies a stable release; "
    "open it with the gripper Skill after place and lift the empty gripper with the safe TCP Skill.")
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
    include_safe_pose: bool = False,
    include_grasp_approach: bool = False,
    include_grasp_approach_plan: bool = False,
    include_observed_point: bool = False,
    include_observed_anchor_v2: bool = False,
    include_object_motion: bool = False,
) -> tuple[str, ...]:
    """Return the single authoritative real-motion Route allowlist."""

    optional_routes = (
        *((REAL_AGENT_OBJECT_MOTION_ROUTE_ID_V1,) if include_object_motion else ()),
        *((REAL_AGENT_OBSERVED_POINT_ROUTE_ID_V1,) if include_observed_point else ()),
        *((REAL_AGENT_OBSERVED_ANCHOR_ROUTE_ID_V2,) if include_observed_anchor_v2 else ()),
        *((REAL_AGENT_SAFE_TCP_POSE_TARGET_ROUTE_ID_V1,) if include_safe_pose else ()),
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
                *optional_routes,
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
    additional_components: tuple[tuple[str, str], ...] = ()

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
        extra = tuple(sorted(tuple(pair) for pair in self.additional_components))
        components = ((self.component_id, self.side), *extra)
        if (any(len(pair) != 2 or not isinstance(pair[0], str) or not pair[0]
                or pair[1] not in {"left", "right"} for pair in components)
                or len({pair[0] for pair in components}) != len(components)
                or len({pair[1] for pair in components}) != len(components)):
            raise ValueError("Pose profile requires unique component/side bindings.")
        object.__setattr__(self, "additional_components", extra)

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(sorted((self.component_id, *(item[0] for item in self.additional_components))))

    def side_for_component(self, component_id: str) -> str:
        return dict(((self.component_id, self.side), *self.additional_components))[component_id]

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
                **{key: value for key, value in self.__dict__.items()
                   if key != "additional_components" or value},
                "profile_id": self.parameter_profile_id,
                "controlled_frame_id": "tool_tcp",
                "target_semantics": "absolute_pose_in_public_reference_frame",
                "execution_transform": "single_frame_graph_projection_at_activation",
                "speed_semantics": "fraction_of_lower_reviewed_trajectory_limits",
            },
            domain="dexteleop.agent_skill.real_tcp_pose_target_profile.v1",
        )


@dataclass(frozen=True)
class RealAgentObservedPointProfile(RealAgentTcpPoseTargetProfile):
    """Bind relation resolution to the existing safe TCP limits."""

    speed_scale: float = 0.2

    def __post_init__(self) -> None:
        super().__post_init__()
        if (isinstance(self.speed_scale, bool)
                or not isinstance(self.speed_scale, (int, float))
                or not math.isfinite(self.speed_scale)
                or not self.min_speed_scale <= self.speed_scale <= self.max_speed_scale):
            raise ValueError("Observed-point speed must be within the bound TCP limits.")

    @property
    def skill_id(self) -> str:
        return OBSERVED_POINT_SKILL_ID

    @property
    def parameter_profile_id(self) -> str:
        return "real-observed-point-v1"

    @property
    def profile_id(self) -> str:
        return self.parameter_profile_id

    @property
    def workspace_bounds(self) -> None:
        # Real motion uses the lower deployment envelope, not a second AABB.
        return None

    @property
    def pose_goal_policy(self) -> str:
        return "preserve_target"

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash({
            **self.__dict__, "profile_id": self.parameter_profile_id,
            "controlled_frame_id": "tool_tcp",
            "target_semantics": "authenticated_surface_point_plus_offset_preserve_activation_orientation",
            "source_contracts": ("back_project@1", "back_project@2"),
            "pose_goal_policy": self.pose_goal_policy,
        }, domain="agenticrobot.real_observed_point.profile.v1")


@dataclass(frozen=True)
class RealAgentObservedAnchorProfile(RealAgentObservedPointProfile):
    """The same lower safe limits with a v2 authenticated anchor source."""

    @property
    def parameter_profile_id(self) -> str:
        return "real-observed-anchor-v2"

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash({
            **self.__dict__, "profile_id": self.parameter_profile_id,
            "controlled_frame_id": "tool_tcp",
            "target_semantics": "authenticated_surface_anchor_plus_offset_preserve_activation_orientation",
            "source_contracts": ("capture_surface_anchor@2",),
            "pose_goal_policy": self.pose_goal_policy,
        }, domain="agenticrobot.real_observed_anchor.profile.v2")


@dataclass(frozen=True)
class RealAgentObjectMotionProfile(RealAgentObservedPointProfile):
    """Safe TCP limits plus the bound held-object identity and object-pose bounds."""

    object_id: str = "right_held_object"
    object_workspace_bounds: tuple[float, float, float, float, float, float] = (
        -0.45, 0.45, -0.70, 0.70, -0.02, 0.80)
    observation_source: str = REAL_AGENT_OBJECT_MOTION_OBSERVATION_SOURCE_V1
    clock_domain_id: str = "real-station-wall-clock"

    def __post_init__(self) -> None:
        super().__post_init__()
        bounds = tuple(float(value) for value in self.object_workspace_bounds)
        if (len(bounds) != 6 or not all(math.isfinite(value) for value in bounds)
                or not (bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5])):
            raise ValueError("Object workspace bounds must be ordered finite xyz pairs.")
        object.__setattr__(self, "object_workspace_bounds", bounds)
        for name in ("object_id", "observation_source", "clock_domain_id"):
            _required(getattr(self, name), field_name=name)

    @property
    def skill_id(self) -> str:
        return GRASPED_OBJECT_MOVE_SKILL_ID

    @property
    def parameter_profile_id(self) -> str:
        return "real-object-motion-v1"

    @property
    def profile_id(self) -> str:
        return self.parameter_profile_id

    @property
    def workspace_bounds(self) -> tuple[float, ...]:
        return self.object_workspace_bounds

    @property
    def pose_goal_policy(self) -> str:
        return "within_arrival_tolerance"

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash({
            **self.__dict__, "profile_id": self.parameter_profile_id, "controlled_frame_id": "object_body",
            "target_semantics": "measured_held_object_box_pose_in_public_frame",
            "held_object_identity": "object_between_the_closing_pads_of_the_bound_arm",
            "measurement": "stationary_rgbd_closing_region_point_prompt_plus_dmtac_closure",
            "pose_goal_policy": self.pose_goal_policy,
        }, domain="agenticrobot.real.object_motion_profile.v1")


_HELD_OBJECT_REPORT_SCHEMA = {
    "type": "object",
    "description": ("Lower held-object measurement: box size in the tool frame, the box bottom below the "
                    "reported object pose, start/terminal observations (relation, closure, grip offsets) "
                    "and, for place, the measured support plane."),
}


def real_object_motion_schema_v1(*, definition, profile: RealAgentObjectMotionProfile):
    """The shared object-motion schema plus the lower measurement report.

    The shared root ``oneOf`` (exactly one of the two targets) moves into the
    target descriptions; the lower validator still enforces it. Some model
    providers reject a root-level ``oneOf`` beside ``properties`` in tools.
    """
    entry = grasped_object_move_schema(
        definition=definition, profile=profile, object_id=profile.object_id,
        observation_source=profile.observation_source, clock_domain_id=profile.clock_domain_id)
    parameters = {key: value for key, value in entry.parameter_schema.items() if key != "oneOf"}
    properties = dict(parameters["properties"])
    properties["object_pose_xyz_quat_xyzw"] = {
        **properties["object_pose_xyz_quat_xyzw"],
        "description": ("Absolute object-body goal [x, y, z, qx, qy, qz, qw] in the public frame. Give exactly one "
                        "of object_pose_xyz_quat_xyzw or translation_delta_xyz_m.")}
    properties["translation_delta_xyz_m"] = {
        **properties["translation_delta_xyz_m"],
        "description": ("Offset [dx, dy, dz] of the currently measured object in the public frame, keeping its "
                        "orientation. Give exactly one of translation_delta_xyz_m or object_pose_xyz_quat_xyzw.")}
    parameters["properties"] = properties
    result = dict(entry.execution_result_schema)
    result["properties"] = {**result["properties"], "held_object_report": _HELD_OBJECT_REPORT_SCHEMA}
    result["required"] = [*result["required"], "held_object_report"]
    schema_hash = agent_skill_schema_document_hash(parameters, schema_kind="parameter")
    return replace(entry, parameter_schema=parameters, execution_result_schema=result,
                   parameter_profiles=tuple(replace(item, parameter_schema_hash=schema_hash)
                                            for item in entry.parameter_profiles))


def _object_motion_payload_validator(*, route: SkillExecutionRoute,
                                     profile: RealAgentObjectMotionProfile) -> AgentSkillPayloadValidatorSet:
    def validate_parameters(payload):
        validate_grasped_object_move(payload, profile=profile, object_id=profile.object_id)

    def validate_execution_result(payload):
        required = {"execution_receipt_refs", "final_measured_state_hash", "object_state", "held_object_report"}
        if not isinstance(payload, Mapping) or set(payload) != required:
            raise ValueError("Real object-motion result schema mismatch.")
        refs = payload["execution_receipt_refs"]
        if (not isinstance(refs, (tuple, list)) or not refs or len(refs) != len(set(refs))
                or any(not isinstance(item, str) or not item for item in refs)):
            raise ValueError("execution_receipt_refs must be unique and non-empty.")
        _required(payload["final_measured_state_hash"], field_name="final_measured_state_hash")
        validate_object_motion_state(payload["object_state"], profile=profile, object_id=profile.object_id,
                                     observation_source=profile.observation_source,
                                     clock_domain_id=profile.clock_domain_id)
        if not isinstance(payload["held_object_report"], Mapping):
            # Payload validators report every schema mismatch as ValueError.
            raise ValueError("Real object-motion report must be an object.")  # noqa: TRY004

    def validate_task_evaluation(payload):
        if not isinstance(payload, Mapping) or set(payload) != {"aggregate_evaluation_receipt_ref", "task_outcome"}:
            raise ValueError("Real primitive task-evaluation schema mismatch.")

    def validate_result_for_parameters(parameters, result):
        if result["object_state"]["phase"] != parameters["phase"]:
            raise ValueError("Object-motion result phase differs from its request.")

    return AgentSkillPayloadValidatorSet(
        validator_id="payload-validator:real-agent-object-motion:" + profile.semantic_content_hash,
        skill_id=route.skill_id, skill_version=route.skill_version, route_id=route.route_id,
        route_version=route.route_version, parameter_schema_hash=route.parameter_schema_hash,
        execution_result_schema_hash=route.execution_result_schema_hash,
        task_evaluation_schema_hash=route.task_evaluation_schema_hash,
        validate_parameters=validate_parameters, validate_execution_result=validate_execution_result,
        validate_task_evaluation=validate_task_evaluation,
        validate_execution_result_for_parameters=validate_result_for_parameters)


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
    segmentation_registration: SegmentationRegistration | None = None
    surface_anchor_capture_registration: SpatialGoalRegistration | None = None
    placement_registration: PlacementRegistration | None = None
    station_point_grasp_registration: StationBoundPointGraspRegistration | None = None
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    ) = None
    grasp_approach_plan_registration: RealGraspApproachPlanRegistration | None = None
    object_transport_registration: ObjectTransportRegistration | None = None
    gripper_contract: GripperSkillContractBundle | None = None
    spatial_goal_registration: SpatialGoalRegistration | None = None
    tcp_pose_comparison_registration: TcpPoseComparisonRegistration | None = None
    rgbd_geometry_registrations: tuple[RgbdGeometryRegistration, ...] = ()
    grasp_fast_plan_registration: RealFastGraspPlanRegistration | None = None
    observed_point_profile: RealAgentObservedPointProfile | None = None
    observed_anchor_profile_v2: RealAgentObservedAnchorProfile | None = None
    object_motion_profile: RealAgentObjectMotionProfile | None = None
    tool_object_relation_registration: ToolEntityRegistration | None = None
    tool_hover_registration: ToolEntityRegistration | None = None

    def __post_init__(self) -> None:
        published_skill_ids = {
            *(item.skill_id for item in self.definition_catalog.skill_definitions),
            *(item.skill_id for item in self.schema_catalog.schema_entries),
            *(item.route.skill_id for item in self.route_manifest.skill_routes),
            *(item.skill_id for item in self.io_catalog.io_contracts),
        }
        if published_skill_ids.intersection(ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1):
            raise ValueError("Archived debug motion Skills cannot be published.")
        validate_skill_route_catalog_closure(
            definition_catalog=self.definition_catalog,
            schema_catalog=self.schema_catalog,
            route_manifest=self.route_manifest,
            io_catalog=self.io_catalog,
        )
        self.payload_validators.validate_manifest(self.route_manifest)
        if (self.tcp_pose_comparison_registration is not None
                and self.route_manifest.route("state.read_tcp_pose", "2") != self.tcp_pose_comparison_registration.entry):
            raise ValueError("TCP comparison registration differs from its manifest entry.")
        if self.spatial_goal_registration is not None:
            if self.route_manifest.route("perception.compute_spatial_goal", "1") != self.spatial_goal_registration.entry:
                raise ValueError("Spatial goal registration differs from its manifest entry.")
        if self.rgbd_geometry_registrations and self.perception_capture is None:
            raise ValueError("RGB-D geometry requires composed perception capture.")
        if self.gripper_contract is not None:
            binding = self.gripper_contract.deployment_binding
            if binding.skill_version == "1" and self.cardinal_profile.side != "right":
                raise ValueError("Gripper v1 requires the bound right-arm session.")
            for name in (
                "runtime_session_id",
                "control_domain_id",
                "target_binding_id",
                "station_manifest_hash",
                "execution_manifest_hash",
                "target_kind",
                "current_mode",
            ):
                if getattr(binding, name) != getattr(self.route_manifest, name):
                    raise ValueError("Gripper and arm deployment identities disagree.")
        pose_target_enabled = any(
            entry.route.skill_id == TCP_POSE_TARGET_SKILL_ID_V1
            for entry in self.route_manifest.skill_routes
        )
        safe_pose_enabled = any(
            entry.route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1
            for entry in self.route_manifest.skill_routes
        )
        observed_point_enabled = self.observed_point_profile is not None
        if observed_point_enabled != any(
                entry.route.skill_id == OBSERVED_POINT_SKILL_ID
                and entry.route.skill_version == "1"
                for entry in self.route_manifest.skill_routes):
            raise ValueError("Observed-point profile and Route must coexist.")
        observed_anchor_enabled = self.observed_anchor_profile_v2 is not None
        if observed_anchor_enabled != any(
                entry.route.skill_id == OBSERVED_POINT_SKILL_ID
                and entry.route.skill_version == "2"
                for entry in self.route_manifest.skill_routes):
            raise ValueError("Observed-anchor v2 profile and Route must coexist.")
        if observed_anchor_enabled and self.surface_anchor_capture_registration is None:
            raise ValueError("Observed-anchor motion requires its v2 capture producer.")
        object_motion_enabled = self.object_motion_profile is not None
        if object_motion_enabled != any(
                entry.route.skill_id == GRASPED_OBJECT_MOVE_SKILL_ID and entry.route.skill_version == "1"
                for entry in self.route_manifest.skill_routes):
            raise ValueError("Object-motion profile and Route must coexist.")
        if object_motion_enabled and (not safe_pose_enabled or self.perception_capture is None):
            raise ValueError("Real object motion needs the observed safe TCP Route and RGB-D capture.")
        if observed_point_enabled and (self.perception_capture is None
                or self.perception_capture.back_projection is None):
            raise ValueError("Observed-point motion requires bound back-projection.")
        grasp_approach_enabled = any(
            entry.route.skill_id == GRASP_APPROACH_SKILL_ID_V1
            for entry in self.route_manifest.skill_routes
        )
        if (pose_target_enabled or safe_pose_enabled or grasp_approach_enabled
                or observed_point_enabled or observed_anchor_enabled or object_motion_enabled) != (
            self.pose_target_profile is not None
        ):
            raise ValueError("Absolute-pose Routes require their bound target profile.")
        grasp_approach_plan_enabled = self.grasp_approach_plan_registration is not None
        fast = self.grasp_fast_plan_registration
        fast_entry = next((entry for entry in self.route_manifest.skill_routes
            if entry.route.skill_id == "perception.plan_grasp_fast"), None)
        if (fast is not None) != (fast_entry is not None):
            raise ValueError("Fast grasp registration and manifest must coexist.")
        if fast is not None:
            point = self.station_point_grasp_registration
            if (not grasp_approach_plan_enabled or point is None or not point.profile.frozen_observation
                    or point.profile.arm_reachability_profile is None):
                raise ValueError("Fast grasp requires frozen embodied grasp and paired execution.")
            if (fast_entry.route != fast.skill.routes[0]
                    or fast_entry.deployment_binding != fast.binding
                    or fast_entry.availability_hash != fast.availability_hash):
                raise ValueError("Fast grasp registration differs from its manifest entry.")
        transport_enabled = self.object_transport_registration is not None
        if transport_enabled != any(entry.route.skill_id == "perception.plan_object_move"
                                    for entry in self.route_manifest.skill_routes):
            raise ValueError("Object transport registration and manifest disagree.")
        if transport_enabled and self.perception_capture is None:
            raise ValueError("Object transport needs the observation Artifact store.")
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
            0
            + int(pose_target_enabled)
            + int(safe_pose_enabled)
            + int(observed_point_enabled)
            + int(observed_anchor_enabled)
            + int(object_motion_enabled)
            + int(grasp_approach_enabled)
            + int(grasp_approach_plan_enabled)
        )
        if (
            self.orientation_profiles
            or len(self.deployment_bindings) != expected_binding_count
        ):
            raise ValueError(
                "Real motion bindings must match only the retained absolute-pose, "
                "grasp-approach, and paired-plan execution Routes."
            )
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
        optional_grasp = (self.station_point_grasp_registration,
                          self.station_point_grasp_render_registration)
        if any(item is not None for item in optional_grasp) != all(
            item is not None for item in optional_grasp
        ) or (any(item is not None for item in optional_grasp)
              and self.localization_registration is None):
            raise ValueError(
                "Real localization, point-grasp, and render registrations must coexist."
            )
        if (
            all(item is not None for item in optional_grasp)
            and self.perception_capture is None
        ):
            raise ValueError("Real point-grasp Routes require perception capture.")
        if (
            self.segmentation_registration is not None
            and self.perception_capture is None
        ):
            raise ValueError("Real segmentation Route requires perception capture.")
        if self.surface_anchor_capture_registration is not None and (
            self.segmentation_registration is None
            or self.perception_capture is None
            or self.perception_capture.back_projection is None
        ):
            raise ValueError("Real surface anchor v2 requires segmentation and back projection.")
        if self.surface_anchor_capture_registration is not None and (
            self.route_manifest.route("perception.capture_surface_anchor", "2")
            != self.surface_anchor_capture_registration.entry
        ):
            raise ValueError("Real surface anchor differs from its manifest entry.")
        if self.placement_registration is not None and (
            self.segmentation_registration is None
            or self.perception_capture is None
            or self.perception_capture.back_projection is None
        ):
            raise ValueError(
                "Real placement Route requires segmentation and back-projection."
            )
        if self.placement_registration is not None:
            placement_profile = self.placement_registration.profile
            if (
                placement_profile.component_id != self.cardinal_profile.component_id
                or placement_profile.reference_frame_id != public_frame[0]
                or placement_profile.reference_frame_spec_hash != public_frame[1]
                or placement_profile.frame_graph_hash
                != self.frame_graph.semantic_content_hash
            ):
                raise ValueError("Real placement frame/component binding mismatch.")
        if grasp_approach_plan_enabled and not all(
            item is not None for item in optional_grasp
        ):
            raise ValueError("Grasp-approach planning requires retained point grasps.")
        if self.tool_object_relation_registration is not None and (
            self.segmentation_registration is None or self.perception_capture is None
            or self.perception_capture.back_projection is None
            or self.route_manifest.route("perception.describe_tool_object_relations", "1")
            != self.tool_object_relation_registration.entry
        ):
            raise ValueError("Real tool-object relations need segmentation, aligned RGB-D and their manifest entry.")
        if self.tool_hover_registration is not None and (
            self.surface_anchor_capture_registration is None or not safe_pose_enabled
            or self.route_manifest.route("perception.query_tool_hover_paths", "1")
            != self.tool_hover_registration.entry
        ):
            raise ValueError("Real tool hover queries need surface anchor v2, the safe TCP Route and their entry.")
        expected_route_count = (
            (1 + int(pose_target_enabled) + int(grasp_approach_enabled)
             + int(safe_pose_enabled) + int(observed_point_enabled)
             + int(observed_anchor_enabled) + int(object_motion_enabled))
            + (2 if self.perception_capture is not None else 0)
            + int(
                self.perception_capture is not None
                and self.perception_capture.back_projection is not None
            )
            + len(self.rgbd_geometry_registrations)
            + int(self.localization_registration is not None)
            + 2 * int(self.station_point_grasp_registration is not None)
            + int(self.segmentation_registration is not None)
            + int(self.surface_anchor_capture_registration is not None)
            + int(self.tool_object_relation_registration is not None)
            + int(self.tool_hover_registration is not None)
            + int(self.placement_registration is not None)
            + (2 if grasp_approach_plan_enabled else 0)
            + int(transport_enabled)
            + int(self.gripper_contract is not None)
            + int(self.spatial_goal_registration is not None)
            + int(self.tcp_pose_comparison_registration is not None)
            + int(fast is not None)
            + int(
                self.station_point_grasp_registration is not None
                and self.station_point_grasp_registration.observation_registration
                is not None
            )
        )
        if len(self.route_manifest.skill_routes) != expected_route_count:
            raise ValueError("Real perception registration and manifest disagree.")

    def binding_for_skill(self, skill_id: str, skill_version: str = "1") -> RealAgentPrimitiveDeploymentBinding:
        route = self.route_manifest.route(skill_id, skill_version).route
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
    observed_anchor = isinstance(profile, RealAgentObservedAnchorProfile)
    observed_point = isinstance(profile, RealAgentObservedPointProfile) and not observed_anchor
    pose_components = (profile.component_ids if pose_target and route.skill_id in {
        TCP_POSE_TARGET_SKILL_ID_V1, SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
    } else (profile.component_id,))
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
        if observed_point or observed_anchor:
            component = payload.get("component_id") if isinstance(payload, Mapping) else None
            if component not in profile.component_ids:
                raise ValueError("Observed-point component differs from its bound profile.")
            if observed_anchor:
                validate_observed_anchor_parameters_v2(payload, component_id=component)
            else:
                validate_observed_point_parameters(payload, component_id=component)
            return
        payload_fields = set(payload) if isinstance(payload, Mapping) else set()
        allowed_parameter_fields = (
            parameter_fields | {"speed_scale"} if pose_target else parameter_fields
        )
        if route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1:
            allowed_parameter_fields |= {"pose_goal_policy"}
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
            "allowed_component_ids": pose_components,
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
            if (not isinstance(targets, Mapping) or len(targets) != 1
                    or not set(targets) <= set(pose_components)):
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
        required = {"execution_receipt_refs", "final_measured_state_hash"}
        allowed = required | ({"pose_goal_report"}
            if route.skill_id == SAFE_TCP_POSE_TARGET_SKILL_ID_V1 else set())
        if observed_point or observed_anchor:
            required = required | {"surface_anchor_report" if observed_anchor else "observed_point_report"}
            allowed = required
        if not isinstance(payload, Mapping) or not required <= set(payload) <= allowed:
            raise ValueError("Real primitive result schema mismatch.")
        if "pose_goal_report" in payload:
            validate_safe_tcp_pose_goal_report_v1(payload["pose_goal_report"])
        if observed_point:
            validate_observed_point_report(payload["observed_point_report"],
                                          reference_frame_id=profile.reference_frame_id)
        if observed_anchor:
            validate_observed_anchor_report_v2(payload["surface_anchor_report"],
                                               reference_frame_id=profile.reference_frame_id)
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

    def validate_result_for_parameters(parameters, result):
        if observed_anchor:
            if result["surface_anchor_report"]["source_operation_id"] != parameters["surface_anchor_operation_id"]:
                raise ValueError("Observed-anchor result differs from its request.")
            return
        report = result["observed_point_report"]
        if any(report[name] != parameters[name] for name in ("observation_operation_id", "point_index")):
            raise ValueError("Observed-point result differs from its request.")

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
        validate_execution_result_for_parameters=(validate_result_for_parameters
            if observed_point or observed_anchor else None),
    )


def build_real_agent_primitive_contract_bundle(
    *,
    attachment: RealTcpPoseObserveAttachmentSpec,
    port_config: RealAgentMotionPortConfig,
    primary_component_id: str | None = None,
    frame_graph: CoordinateFrameGraph,
    primitive_reference_frame_id: str,
    safety_envelope_refs: tuple[SkillDeploymentSafetyEnvelopeRef, ...],
    convergence_profile: RealAgentPrimitiveConvergenceProfile | None = None,
    perception_capture: RealPerceptionCaptureRegistration | None = None,
    rgbd_geometry_registrations: tuple[RgbdGeometryRegistration, ...] = (),
    object_transport_registration: ObjectTransportRegistration | None = None,
    gripper_contract: GripperSkillContractBundle | None = None,
    localization_registration: RealObjectLocalizationRegistration | None = None,
    segmentation_registration: SegmentationRegistration | None = None,
    enable_surface_anchor_v2: bool = False,
    enable_placement: bool = True,
    station_point_grasp_registration: StationBoundPointGraspRegistration | None = None,
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    ) = None,
    grasp_execution_operation_timeout_sec: float = DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    domain_emergency_stop_available: bool = False,
    enable_spatial_goals: bool = False,
    enable_tcp_pose_comparison: bool = False,
    enable_tool_object_relations: bool = False,
    enable_tool_hover_queries: bool = False,
    tool_entity_by_side: Mapping[str, object] | None = None,
    tool_hover_planner_profile_hash: str | None = None,
    tool_hover_roles: tuple[str, ...] | None = None,
    tool_hover_region_budget_sec: float | None = None,
    tool_hover_candidate_budget_sec: float | None = None,
    object_motion: Mapping[str, object] | None = None,
) -> RealAgentPrimitiveContractBundle:
    """Build exact motion and optional passive Routes without runtime effects.

    ``object_motion`` (``object_id``, ``object_workspace_bounds``,
    ``clock_domain_id``) publishes the real held-object motion Route when the
    lower envelopes allow it.
    """

    if not isinstance(attachment, RealTcpPoseObserveAttachmentSpec):
        raise TypeError("attachment must be typed.")
    if not isinstance(port_config, RealAgentMotionPortConfig):
        raise TypeError("port_config must be typed.")
    if not isinstance(frame_graph, CoordinateFrameGraph):
        raise TypeError("frame_graph must be typed.")
    if not isinstance(domain_emergency_stop_available, bool):
        raise TypeError("domain_emergency_stop_available must be boolean.")
    if not isinstance(enable_spatial_goals, bool):
        raise TypeError("enable_spatial_goals must be boolean.")
    if not isinstance(enable_tcp_pose_comparison, bool):
        raise TypeError("enable_tcp_pose_comparison must be boolean.")
    if not isinstance(enable_surface_anchor_v2, bool) or not isinstance(enable_placement, bool):
        raise TypeError("Optional real perception switches must be boolean.")
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
    if len(attachment.component_ids) != 1 and primary_component_id is None:
        raise ValueError("Multiple arms require an explicit primary component for grasp/fixed primitives.")
    component_id = (attachment.component_ids[0] if primary_component_id is None
                    else _required(primary_component_id, field_name="primary_component_id"))
    if component_id not in attachment.component_ids:
        raise ValueError("Primary component is absent from the motion attachment.")
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
        additional_components=tuple((component, arm_side)
            for arm_side, component in port_config.component_id_by_side.items()
            if component != component_id),
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
            execution_operation_timeout_sec=grasp_execution_operation_timeout_sec,
        )
        if grasp_approach_plan_enabled
        else None
    )
    grasp_fast_plan_registration = (
        build_real_fast_grasp_plan_registration(
            paired=grasp_approach_plan_registration,
            point_grasp_registration=station_point_grasp_registration)
        if (grasp_approach_plan_registration is not None
            and station_point_grasp_registration.profile.frozen_observation
            and station_point_grasp_registration.profile.backend_id == "graspgenx"
            and station_point_grasp_registration.profile.arm_reachability_profile is not None)
        else None
    )
    pose_target_definition = tcp_pose_target_skill_definition_v1(
        controlled_component_ids=pose_target_profile.component_ids,
    )
    pose_target_schema = tcp_pose_target_schema_entry_v1(
        max_target_components=1 if pose_target_profile.additional_components else None,
        skill_definition_hash=pose_target_definition.semantic_content_hash,
        parameter_profile_id=pose_target_profile.parameter_profile_id,
        parameter_profile_hash=pose_target_profile.semantic_content_hash,
        allowed_component_ids=pose_target_profile.component_ids,
        allowed_reference_frame_ids=(pose_target_profile.reference_frame_id,),
        max_requested_duration_ms=None,
        min_speed_scale=pose_target_profile.min_speed_scale,
        max_speed_scale=pose_target_profile.max_speed_scale,
    )
    grasp_approach_definition = grasp_approach_skill_definition_v1(
        controlled_component_ids=(component_id,),
    )
    safe_pose_definition = safe_tcp_pose_target_skill_definition_v1(
        controlled_component_ids=pose_target_profile.component_ids,
    )
    observed_point_enabled = any(REAL_AGENT_OBSERVED_POINT_ROUTE_ID_V1 in ref.allowed_route_ids
                                 for ref in refs)
    observed_point_profile = None
    observed_definition = observed_schema = None
    if observed_point_enabled:
        if perception_capture is None or perception_capture.back_projection is None:
            raise ValueError("Observed-point motion requires bound back-projection.")
        observed_point_profile = RealAgentObservedPointProfile(**pose_target_profile.__dict__)
        observed_definition = observed_point_definition(
            controlled_component_ids=observed_point_profile.component_ids)
        observed_schema = observed_point_schema(definition=observed_definition,
                                                profile=observed_point_profile)
        if len(observed_point_profile.component_ids) > 1:
            parameters = dict(observed_schema.parameter_schema)
            parameters["properties"] = {**parameters["properties"], "component_id": {
                "type": "string", "enum": list(observed_point_profile.component_ids)}}
            observed_schema = replace(observed_schema, parameter_schema=parameters,
                parameter_profiles=(replace(observed_schema.parameter_profiles[0],
                    parameter_schema_hash=agent_skill_schema_document_hash(
                        parameters, schema_kind="parameter")),))
    observed_anchor_enabled = any(
        REAL_AGENT_OBSERVED_ANCHOR_ROUTE_ID_V2 in ref.allowed_route_ids for ref in refs
    )
    observed_anchor_profile_v2 = None
    observed_anchor_definition = observed_anchor_schema = None
    if observed_anchor_enabled:
        if not observed_point_enabled or not enable_surface_anchor_v2:
            raise ValueError("Observed-anchor v2 requires v1 safe motion and v2 capture.")
        observed_anchor_profile_v2 = RealAgentObservedAnchorProfile(**pose_target_profile.__dict__)
        observed_anchor_definition = observed_anchor_definition_v2(
            controlled_component_ids=observed_anchor_profile_v2.component_ids)
        observed_anchor_schema = observed_anchor_schema_v2(
            definition=observed_anchor_definition, profile=observed_anchor_profile_v2)
        if len(observed_anchor_profile_v2.component_ids) > 1:
            parameters = dict(observed_anchor_schema.parameter_schema)
            parameters["properties"] = {**parameters["properties"], "component_id": {
                "type": "string", "enum": list(observed_anchor_profile_v2.component_ids)}}
            observed_anchor_schema = replace(observed_anchor_schema, parameter_schema=parameters,
                parameter_profiles=(replace(observed_anchor_schema.parameter_profiles[0],
                    parameter_schema_hash=agent_skill_schema_document_hash(
                        parameters, schema_kind="parameter")),))
    object_motion_enabled = any(REAL_AGENT_OBJECT_MOTION_ROUTE_ID_V1 in ref.allowed_route_ids for ref in refs)
    object_motion_profile = object_motion_definition = object_motion_schema = None
    if object_motion_enabled:
        if object_motion is None or perception_capture is None:
            raise ValueError("Real object motion needs its held-object binding and RGB-D capture.")
        object_motion_profile = RealAgentObjectMotionProfile(
            **pose_target_profile.__dict__, object_id=str(object_motion["object_id"]),
            object_workspace_bounds=tuple(object_motion["object_workspace_bounds"]),
            clock_domain_id=str(object_motion["clock_domain_id"]))
        object_motion_definition = replace(grasped_object_move_definition(
            controlled_component_ids=(object_motion_profile.component_id,)),
            summary=REAL_OBJECT_MOTION_SUMMARY_V1)
        object_motion_schema = real_object_motion_schema_v1(
            definition=object_motion_definition, profile=object_motion_profile)
    safe_pose_schema = tcp_pose_target_schema_entry_v1(
        max_target_components=1 if pose_target_profile.additional_components else None,
        skill_id=SAFE_TCP_POSE_TARGET_SKILL_ID_V1,
        skill_definition_hash=safe_pose_definition.semantic_content_hash,
        parameter_profile_id=pose_target_profile.parameter_profile_id,
        parameter_profile_hash=pose_target_profile.semantic_content_hash,
        allowed_component_ids=pose_target_profile.component_ids,
        allowed_reference_frame_ids=(pose_target_profile.reference_frame_id,),
        max_requested_duration_ms=None,
        min_speed_scale=pose_target_profile.min_speed_scale,
        max_speed_scale=pose_target_profile.max_speed_scale,
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
            route_version=definition.skill_version,
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
    safe_pose_route = route_for(
        definition=safe_pose_definition, schema=safe_pose_schema,
        route_id=REAL_AGENT_SAFE_TCP_POSE_TARGET_ROUTE_ID_V1,
        max_operation_duration_ms=120_000,
    )
    observed_route = (None if not observed_point_enabled else route_for(
        definition=observed_definition, schema=observed_schema,
        route_id=REAL_AGENT_OBSERVED_POINT_ROUTE_ID_V1, max_operation_duration_ms=120_000))
    observed_anchor_route = (None if not observed_anchor_enabled else route_for(
        definition=observed_anchor_definition, schema=observed_anchor_schema,
        route_id=REAL_AGENT_OBSERVED_ANCHOR_ROUTE_ID_V2,
        max_operation_duration_ms=120_000))
    # Measurement before planning and after arrival (RGB-D + segmentation per
    # view) on top of the safe-move planning and execution budget.
    object_motion_route = (None if not object_motion_enabled else route_for(
        definition=object_motion_definition, schema=object_motion_schema,
        route_id=REAL_AGENT_OBJECT_MOTION_ROUTE_ID_V1, max_operation_duration_ms=180_000))
    safe_pose_enabled = any(
        REAL_AGENT_SAFE_TCP_POSE_TARGET_ROUTE_ID_V1 in ref.allowed_route_ids for ref in refs
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
    if object_motion_enabled and not safe_pose_enabled:
        raise ValueError("Real object motion reuses the observed safe TCP Route's planner.")
    all_routes = (
        *((observed_route,) if observed_route is not None else ()),
        *((observed_anchor_route,) if observed_anchor_route is not None else ()),
        *((object_motion_route,) if object_motion_route is not None else ()),
        *((safe_pose_route,) if safe_pose_enabled else ()),
        *((pose_target_route,) if pose_target_enabled else ()),
        *((grasp_approach_route,) if grasp_approach_enabled else ()),
        *(
            ()
            if execute_grasp_approach_plan_route is None
            else (execute_grasp_approach_plan_route,)
        ),
    )
    expected_route_ids = set(
        real_agent_motion_route_ids_v1(
            include_safe_pose=safe_pose_enabled,
            include_observed_point=observed_point_enabled,
            include_observed_anchor_v2=observed_anchor_enabled,
            include_object_motion=object_motion_enabled,
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
        **({observed_route.semantic_content_hash: observed_point_profile}
           if observed_route is not None else {}),
        **({observed_anchor_route.semantic_content_hash: observed_anchor_profile_v2}
           if observed_anchor_route is not None else {}),
        **({object_motion_route.semantic_content_hash: object_motion_profile}
           if object_motion_route is not None else {}),
        **({safe_pose_route.semantic_content_hash: pose_target_profile}
           if safe_pose_enabled else {}),
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
    }
    executor_by_route_hash = {
        **({observed_route.semantic_content_hash: REAL_AGENT_OBSERVED_POINT_EXECUTOR_ID_V1}
           if observed_route is not None else {}),
        **({observed_anchor_route.semantic_content_hash: REAL_AGENT_OBSERVED_ANCHOR_EXECUTOR_ID_V2}
           if observed_anchor_route is not None else {}),
        **({object_motion_route.semantic_content_hash: REAL_AGENT_OBJECT_MOTION_EXECUTOR_ID_V1}
           if object_motion_route is not None else {}),
        **({safe_pose_route.semantic_content_hash: REAL_AGENT_SAFE_TCP_POSE_TARGET_EXECUTOR_ID_V1}
           if safe_pose_enabled else {}),
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
    optional_grasp = (station_point_grasp_registration,
                      station_point_grasp_render_registration)
    if any(item is not None for item in optional_grasp) != all(
        item is not None for item in optional_grasp
    ) or (any(item is not None for item in optional_grasp)
          and localization_registration is None):
        raise ValueError(
            "Real localization, point-grasp, and render registrations must coexist."
        )
    if localization_registration is not None and perception_capture is None:
        raise ValueError("Real point-grasp Routes require perception capture.")
    if segmentation_registration is not None and perception_capture is None:
        raise ValueError("Real segmentation Route requires perception capture.")
    placement_registration = (
        build_placement_registration(
            segmentation=segmentation_registration,
            back_projection=perception_capture.back_projection,
            component_id=component_id,
            destination_role="main",
            runtime_session_id=attachment.runtime_session_id,
            control_domain_id=attachment.control_domain_id,
            target_binding_id=attachment.target_binding_id,
            station_manifest_hash=attachment.station_manifest_hash,
            execution_manifest_hash=attachment.execution_manifest_hash,
        )
        if enable_placement and segmentation_registration is not None
        and perception_capture is not None
        and perception_capture.back_projection is not None
        else None
    )
    surface_anchor_capture_registration = None
    if enable_surface_anchor_v2:
        if (segmentation_registration is None or perception_capture is None
                or perception_capture.back_projection is None):
            raise ValueError("Real surface anchor v2 requires segmentation and back projection.")
        from agent_gateway.profiles.surface_anchor import build_surface_anchor_registrations

        (surface_anchor_capture_registration,) = build_surface_anchor_registrations(
            back_projection=perception_capture.back_projection,
            segmentation=segmentation_registration, version="2",
        )
    tool_object_relation_registration = None
    tool_hover_registration = None
    if enable_tool_object_relations or enable_tool_hover_queries:
        if (segmentation_registration is None or perception_capture is None
                or perception_capture.back_projection is None or not tool_entity_by_side):
            raise ValueError("Real tool-entity Skills need segmentation, aligned RGB-D and a reviewed tool entity.")
    if enable_tool_object_relations:
        tool_object_relation_registration = build_tool_object_relation_registration(
            segmentation=segmentation_registration, back_projection=perception_capture.back_projection,
            roles=perception_capture.back_projection.profile.roles, tool_entity_by_side=tool_entity_by_side)
    if enable_tool_hover_queries:
        if (surface_anchor_capture_registration is None or tool_hover_planner_profile_hash is None
                or not tool_hover_roles or not safe_pose_enabled):
            raise ValueError("Real tool hover queries need surface anchor v2 and the observed safe-move planner.")
        hover_sides = {component: side for side, component in port_config.component_id_by_side.items()
                       if side in tool_entity_by_side}
        tool_hover_registration = build_tool_hover_registration(
            capture_registration=surface_anchor_capture_registration,
            back_projection=perception_capture.back_projection, component_side_by_id=hover_sides,
            roles=tuple(tool_hover_roles), tool_entity_by_side=tool_entity_by_side,
            # The checked path belongs to the same safe-move Route the Agent
            # later submits the selected pose to.
            planner_profile_hash=canonical_content_hash(
                {"planner": tool_hover_planner_profile_hash,
                 "safe_motion_route": safe_pose_route.semantic_content_hash},
                domain="agenticrobot.real.tool_hover_planner_binding.v1"),
            **({} if tool_hover_region_budget_sec is None else {"region_budget_sec": tool_hover_region_budget_sec}),
            **({} if tool_hover_candidate_budget_sec is None
               else {"candidate_budget_sec": tool_hover_candidate_budget_sec}))
        if tool_hover_registration.reference_frame_id != pose_target_profile.reference_frame_id:
            # The selected pose is submitted unchanged to the safe TCP Route.
            raise ValueError("Real tool hover results must use the safe TCP Route's public frame.")
    perception_registrations = list(
        ()
        if perception_capture is None
        else (perception_capture.rgb, perception_capture.depth)
    )
    if (
        perception_capture is not None
        and perception_capture.back_projection is not None
    ):
        perception_registrations.append(perception_capture.back_projection.skill)
    if rgbd_geometry_registrations and perception_capture is None:
        raise ValueError("RGB-D geometry requires composed perception capture.")
    perception_registrations.extend(item.skill for item in rgbd_geometry_registrations)
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
    if segmentation_registration is not None:
        perception_registrations.append(segmentation_registration.skill)
    if surface_anchor_capture_registration is not None:
        perception_registrations.append(surface_anchor_capture_registration.skill)
    if tool_object_relation_registration is not None:
        perception_registrations.append(tool_object_relation_registration.skill)
    if tool_hover_registration is not None:
        perception_registrations.append(tool_hover_registration.skill)
    if placement_registration is not None:
        perception_registrations.append(placement_registration.skill)
    if grasp_approach_plan_registration is not None:
        perception_registrations.append(grasp_approach_plan_registration.skill)
    if grasp_fast_plan_registration is not None:
        perception_registrations.append(grasp_fast_plan_registration.skill)
    grasp_observation_registration = (None if station_point_grasp_registration is None
                                      else station_point_grasp_registration.observation_registration)
    if grasp_observation_registration is not None:
        perception_registrations.append(grasp_observation_registration.skill)
    if object_transport_registration is not None:
        if perception_capture is None:
            raise ValueError("Object transport needs perception registration.")
        perception_registrations.append(object_transport_registration.skill)
    perception_registrations = tuple(perception_registrations)
    tcp_pose_definition = tcp_pose_observe.definition_catalog.skill_definitions[0]
    tcp_pose_schema = tcp_pose_observe.schema_catalog.schema_entries[0]
    tcp_pose_entry = tcp_pose_observe.route_manifest.skill_routes[0]
    tcp_pose_comparison_registration = None
    if enable_tcp_pose_comparison:
        from agent_gateway.profiles.tcp_pose_comparison import (
            build_tcp_pose_comparison_registration,
        )

        tcp_pose_comparison_registration = build_tcp_pose_comparison_registration(
            base_definition=tcp_pose_definition, base_schema=tcp_pose_schema,
            base_entry=tcp_pose_entry,
            base_validator=tcp_pose_observe.payload_validators.validator_for(tcp_pose_entry.route),
        )
        perception_registrations = (*perception_registrations, tcp_pose_comparison_registration.skill)
    spatial_goal_registration = None
    if enable_spatial_goals:
        from agent_gateway.profiles.spatial_goal import build_spatial_goal_registration

        spatial_goal_registration = build_spatial_goal_registration(
            base_route=tcp_pose_entry.route, base_binding=tcp_pose_entry.deployment_binding,
            component_ids=attachment.component_ids, reference_frame_id=primitive_reference_frame_id,
        )
        perception_registrations = (*perception_registrations, spatial_goal_registration.skill)
    definitions = SkillDefinitionCatalog(
        catalog_id="dexteleop.agent_skill_definitions.real_motion_primitives",
        catalog_version="1",
        skill_definitions=(
            tcp_pose_definition,
            *((observed_definition,) if observed_point_enabled else ()),
            *((observed_anchor_definition,) if observed_anchor_enabled else ()),
            *((object_motion_definition,) if object_motion_enabled else ()),
            *((safe_pose_definition,) if safe_pose_enabled else ()),
            *((pose_target_definition,) if pose_target_enabled else ()),
            *((grasp_approach_definition,) if grasp_approach_enabled else ()),
            *(
                ()
                if grasp_approach_plan_registration is None
                else (grasp_approach_plan_registration.execute_skill.definition,)
            ),
            *(item.definition for item in perception_registrations),
            *(
                (gripper_contract.definition_catalog.skill_definitions)
                if gripper_contract is not None
                else ()
            ),
        ),
    )
    schemas = AgentSkillSchemaCatalog(
        catalog_id="dexteleop.agent_skill_schemas.real_motion_primitives",
        catalog_version="1",
        schema_entries=(
            tcp_pose_schema,
            *((observed_schema,) if observed_point_enabled else ()),
            *((observed_anchor_schema,) if observed_anchor_enabled else ()),
            *((object_motion_schema,) if object_motion_enabled else ()),
            *((safe_pose_schema,) if safe_pose_enabled else ()),
            *((pose_target_schema,) if pose_target_enabled else ()),
            *((grasp_approach_schema,) if grasp_approach_enabled else ()),
            *(
                ()
                if grasp_approach_plan_registration is None
                else (grasp_approach_plan_registration.execute_skill.schema_entry,)
            ),
            *(item.schema_entry for item in perception_registrations),
            *(
                (gripper_contract.schema_catalog.schema_entries)
                if gripper_contract is not None
                else ()
            ),
        ),
    )
    route_items = [tcp_pose_entry]
    if gripper_contract is not None:
        hand_binding = gripper_contract.deployment_binding
        entry = gripper_contract.route_manifest.skill_routes[0]
        # The composed manifest requires a selected lower constraint binding
        # for every motion Route. This is the hand profile, not a TCP box.
        hand_envelope = SkillDeploymentSafetyEnvelopeRef(
            object_id="real-dmtac-gripper:" + hand_binding.port_binding_hash,
            schema="dexteleop.deployment_safety_envelope.v1",
            content_hash=hand_binding.port_binding_hash,
            constraint_set_hash=gripper_contract.schema_catalog.schema_entries[
                0
            ].parameter_schema_hash,
            allowed_route_ids=(entry.route.route_id,),
        )
        route_items.append(
            replace(
                entry,
                deployment_binding=replace(
                    entry.deployment_binding, safety_envelope_refs=(hand_envelope,)
                ),
            )
        )
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
        if perception_capture.back_projection is not None:
            back_projection = perception_capture.back_projection
            route_items.append(
                SkillRouteManifestEntry(
                    route=back_projection.skill.routes[0],
                    deployment_binding=back_projection.binding.to_ref(),
                    evaluation_plan_hash=None,
                    availability_hash=back_projection.availability_hash,
                )
            )
    for item in rgbd_geometry_registrations:
        route_items.append(SkillRouteManifestEntry(
            route=item.skill.routes[0], deployment_binding=item.binding.to_ref(),
            evaluation_plan_hash=None, availability_hash=item.availability_hash))
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
    if segmentation_registration is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=segmentation_registration.skill.routes[0],
                deployment_binding=segmentation_registration.binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=segmentation_registration.availability_hash,
            )
        )
    if placement_registration is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=placement_registration.skill.routes[0],
                deployment_binding=placement_registration.binding.to_ref(),
                evaluation_plan_hash=None,
                availability_hash=placement_registration.availability_hash,
            )
        )
    if surface_anchor_capture_registration is not None:
        route_items.append(surface_anchor_capture_registration.entry)
    if tool_object_relation_registration is not None:
        route_items.append(tool_object_relation_registration.entry)
    if tool_hover_registration is not None:
        route_items.append(tool_hover_registration.entry)
    if grasp_approach_plan_registration is not None:
        route_items.append(
            SkillRouteManifestEntry(
                route=grasp_approach_plan_registration.skill.routes[0],
                deployment_binding=(grasp_approach_plan_registration.binding.to_ref()),
                evaluation_plan_hash=None,
                availability_hash=(grasp_approach_plan_registration.availability_hash),
            )
        )
    if grasp_observation_registration is not None:
        route_items.append(SkillRouteManifestEntry(
            route=grasp_observation_registration.skill.routes[0],
            deployment_binding=grasp_observation_registration.binding, evaluation_plan_hash=None,
            availability_hash=grasp_observation_registration.availability_hash))
    if grasp_fast_plan_registration is not None:
        route_items.append(SkillRouteManifestEntry(
            route=grasp_fast_plan_registration.skill.routes[0],
            deployment_binding=grasp_fast_plan_registration.binding,
            evaluation_plan_hash=None,
            availability_hash=grasp_fast_plan_registration.availability_hash))
    if object_transport_registration is not None:
        route_items.append(SkillRouteManifestEntry(
            route=object_transport_registration.skill.routes[0],
            deployment_binding=object_transport_registration.binding,
            evaluation_plan_hash=None,
            availability_hash=object_transport_registration.binding.deployment_binding_hash))
    if spatial_goal_registration is not None:
        route_items.append(spatial_goal_registration.entry)
    if tcp_pose_comparison_registration is not None:
        route_items.append(tcp_pose_comparison_registration.entry)
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
                    *ARCHIVED_DEBUG_MOTION_SKILL_IDS_V1,
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
            *((_payload_validator(route=safe_pose_route, profile=pose_target_profile),)
              if safe_pose_enabled else ()),
            *((_payload_validator(route=observed_route, profile=observed_point_profile),)
              if observed_point_enabled else ()),
            *((_payload_validator(route=observed_anchor_route, profile=observed_anchor_profile_v2),)
              if observed_anchor_enabled else ()),
            *((_object_motion_payload_validator(route=object_motion_route, profile=object_motion_profile),)
              if object_motion_enabled else ()),
            *(
                ()
                if perception_capture is None
                else (
                    *perception_capture.payload_validators,
                    *(
                        ()
                        if perception_capture.back_projection is None
                        else (perception_capture.back_projection.payload_validator,)
                    ),
                )
            ),
            *(item.payload_validator for item in rgbd_geometry_registrations),
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
                if segmentation_registration is None
                else (segmentation_registration.payload_validator,)
            ),
            *((surface_anchor_capture_registration.payload_validator,)
              if surface_anchor_capture_registration is not None else ()),
            *((tool_object_relation_registration.payload_validator,)
              if tool_object_relation_registration is not None else ()),
            *((tool_hover_registration.payload_validator,)
              if tool_hover_registration is not None else ()),
            *(
                ()
                if placement_registration is None
                else (placement_registration.payload_validator,)
            ),
            *(
                ()
                if grasp_approach_plan_registration is None
                else (grasp_approach_plan_registration.payload_validator,)
            ),
            *((grasp_fast_plan_registration.payload_validator,)
              if grasp_fast_plan_registration is not None else ()),
            *(
                (grasp_observation_registration.payload_validator,)
                if grasp_observation_registration is not None
                else ()
            ),
            *(
                (object_transport_registration.payload_validator,)
                if object_transport_registration is not None
                else ()
            ),
            *((spatial_goal_registration.payload_validator,) if spatial_goal_registration is not None else ()),
            *((tcp_pose_comparison_registration.payload_validator,) if tcp_pose_comparison_registration is not None else ()),
            *(
                (
                    gripper_contract.payload_validators.validator_for(
                        gripper_contract.route_manifest.skill_routes[0].route
                    ),
                )
                if gripper_contract is not None
                else ()
            ),
        ),
    )
    return RealAgentPrimitiveContractBundle(
        definition_catalog=definitions,
        schema_catalog=schemas,
        route_manifest=manifest,
        io_catalog=io_catalog,
        cardinal_profile=cardinal_profile,
        pose_target_profile=(
            pose_target_profile
            if pose_target_enabled or safe_pose_enabled or grasp_approach_enabled
            or observed_point_enabled or observed_anchor_enabled or object_motion_enabled
            else None
        ),
        orientation_profiles=(),
        deployment_bindings=bindings,
        convergence_profile=convergence,
        payload_validators=validators,
        frame_graph=frame_graph,
        attachment=attachment,
        tcp_pose_observe=tcp_pose_observe,
        perception_capture=perception_capture,
        rgbd_geometry_registrations=rgbd_geometry_registrations,
        observed_point_profile=observed_point_profile,
        observed_anchor_profile_v2=observed_anchor_profile_v2,
        object_motion_profile=object_motion_profile,
        localization_registration=localization_registration,
        segmentation_registration=segmentation_registration,
        surface_anchor_capture_registration=surface_anchor_capture_registration,
        tool_object_relation_registration=tool_object_relation_registration,
        tool_hover_registration=tool_hover_registration,
        placement_registration=placement_registration,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        grasp_approach_plan_registration=grasp_approach_plan_registration,
        grasp_fast_plan_registration=grasp_fast_plan_registration,
        object_transport_registration=object_transport_registration,
        spatial_goal_registration=spatial_goal_registration,
        tcp_pose_comparison_registration=tcp_pose_comparison_registration,
        gripper_contract=gripper_contract,
    )


__all__ = [
    "REAL_AGENT_OBSERVED_POINT_EXECUTOR_ID_V1",
    "REAL_AGENT_OBSERVED_POINT_ROUTE_ID_V1",
    "RealAgentObservedPointProfile",
    "REAL_AGENT_GRASP_APPROACH_EXECUTOR_ID_V1",
    "REAL_AGENT_GRASP_APPROACH_ROUTE_ID_V1",
    "REAL_AGENT_SAFE_TCP_POSE_TARGET_EXECUTOR_ID_V1",
    "REAL_AGENT_SAFE_TCP_POSE_TARGET_ROUTE_ID_V1",
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

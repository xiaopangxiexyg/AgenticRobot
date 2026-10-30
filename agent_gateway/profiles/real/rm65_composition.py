"""Trusted composition root for one independent real RM65 Skill process.

This module binds the neutral Agent Gateway Skill contracts to an already
assembled lower RM65 owner.  Construction is default-off: it opens no device,
socket, thread, or motion authority.  RealTeleop, VIVE, collection, GUI, and
``control_composition`` are deliberately absent from this dependency graph.
"""

from __future__ import annotations

import time
import json
import threading
import math
from functools import partial
from collections.abc import Callable, Mapping
from dataclasses import dataclass, asdict, replace
from pathlib import Path

import numpy as np

from agent_gateway.profiles.segmentation import (
    SegmentationRegistration,
    build_segmentation_registration,
)
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentSafetyEnvelopeRef,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from .motion_observed_scene import (
    MotionPlanningGeometryResources, ObservedSafeMovePlanner, SafeMotionObservationConfig,
)
from .motion_observed_scene_debug import validate_debug_directory
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from agent_gateway.transport.unix_transport import (
    validate_agent_gateway_unix_group_parent,
    validate_agent_gateway_unix_server_preflight,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.coordinate_frames import CoordinateFrameGraph
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    RM65CapturedRobotObservationFilterConfig,
)
from teleop_runtime.skill_deploy import (
    RM65_SKILL_SCENE_WORLD_FRAME_ID_V1,
    ExclusiveRM65SkillLowerAssembly,
    RealAgentMotionPortConfig,
    build_exclusive_rm65_skill_target_builder,
)
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner,
    RM65GraspReachabilityConfig,
    RM65GraspTrajectoryPlanningConfig,
)

from .exclusive_application import (
    ExclusiveRealSkillApplication,
    ExclusiveRealSkillApplicationConfig,
)
from .exclusive_factory import (
    ExclusiveRealAgentPrimitiveDeployment,
    assemble_exclusive_rm65_agent_primitive_deployment,
)
from .grasp_approach_plan import (
    DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
)
from .gripper_skill import ExclusiveRealGripperResources, RealDmtacGripperControlPort
from .localization_deployment import (
    RealObjectLocalizationRegistration,
)
from agent_gateway.profiles.localization_deployment import (
    build_object_localization_registration,
)
from .localization_gateway import ObjectLocalizationProvider
from .motion_primitive_profiles import (
    RealAgentPrimitiveConvergenceProfile,
    real_agent_motion_route_ids_v1,
)
from .observation_gateway import RealTcpPoseObserveAttachmentSpec
from .one_shot_authorization import (
    RealAgentPrimitiveOneShotAuthorization,
)
from .perception_deployment import (
    REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
    RealPerceptionCaptureRegistration,
    build_real_perception_capture_registration_v2,
)
from .perception_frames import (
    REAL_PERCEPTION_SOURCE_FRAME_ID_V1,
    REAL_PERCEPTION_STATION_FRAME_ID_V1,
    RealPerceptionFrameBinding,
    build_real_perception_station_frame_binding,
)
from .realsense_perception import (
    ExclusiveRealPerceptionResources,
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
    build_real_perception_source_binding,
)
from .rm65_camera_geometry import (
    RM65RgbCameraGeometry,
    build_rm65_rgb_camera_geometry,
    load_rm65_wrist_camera_side_axis_tool,
)
from .rm65_grasp_reachability import (
    RM65PointGraspReachabilityAdapter,
    arm_base_origin_in_candidate_frame,
)
from .session_authorization import validate_real_agent_primitive_action
from .station_point_grasp_deployment import (
    StationBoundPointGraspRegistration,
    StationPointGraspArmReachabilityProfile,
    build_station_bound_point_grasp_registration,
)
from .station_point_grasp_render_deployment import (
    StationBoundPointGraspRenderRegistration,
    build_station_bound_point_grasp_render_registration,
)

RM65_SKILL_EXECUTION_FRAME_ID_V1 = RM65_SKILL_SCENE_WORLD_FRAME_ID_V1
RM65_SKILL_STATION_FRAME_ID_V1 = "canonical_station"
RM65_SKILL_CLOCK_DOMAIN_ID_V1 = "python-time-time-wall-clock"
REAL_SURFACE_ANCHOR_MAX_SEGMENT_TIMEOUT_SEC = 45.0


def _required(value: object, *, field_name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    if str(value) != normalized:
        raise ValueError(f"{field_name} cannot contain surrounding whitespace.")
    return normalized


def _hash(payload: object, *, domain: str) -> str:
    return canonical_content_hash(payload, domain=domain)


def _public_tool_profiles(lower: ExclusiveRM65SkillLowerAssembly) -> dict[str, object]:
    return {
        side: {
            key: value
            for key, value in lower.profile.tool_profile_by_side[side].items()
            if not str(key).startswith("_")
        }
        for side in ("left", "right")
    }


def _route_ids(*, include_grasp_approach_plan: bool = False, include_safe_pose: bool = False,
               include_observed_point: bool = False,
               include_observed_anchor_v2: bool = False,
               include_object_motion: bool = False) -> tuple[str, ...]:
    return real_agent_motion_route_ids_v1(
        include_safe_pose=include_safe_pose,
        include_observed_point=include_observed_point,
        include_observed_anchor_v2=include_observed_anchor_v2,
        include_object_motion=include_object_motion,
        include_grasp_approach=True,
        include_grasp_approach_plan=include_grasp_approach_plan,
    )


def _public_object_workspace_bounds(*, public_from_geometry, collision_scene_profile):
    """Held-object pose bounds: the fixed table's footprint, up to 0.8 m above its top."""
    from robot_motion.rm65 import build_dual_arm_scene_profile
    from robot_motion.rm65.scene_profiles import fixed_scene_aabb_obstacles

    table = next(item for item in fixed_scene_aabb_obstacles(build_dual_arm_scene_profile(
        collision_scene_profile or "dual_arm_hand_proxy_dmtac_grasp")) if item.obstacle_id == "fixed-scene:table")
    centre, size = np.asarray(table.center_xyz_m, dtype=float), np.asarray(table.size_xyz_m, dtype=float)
    corners = np.array([[x, y, centre[2] + size[2] / 2.0] for x in (-0.5, 0.5) for y in (-0.5, 0.5)])
    corners[:, :2] = corners[:, :2] * size[:2] + centre[:2]
    transform = np.asarray(public_from_geometry, dtype=float).reshape(4, 4)
    public = corners @ transform[:3, :3].T + transform[:3, 3]
    low, high = public.min(axis=0), public.max(axis=0)
    return (float(low[0]), float(high[0]), float(low[1]), float(high[1]),
            float(low[2]) - 0.02, float(high[2]) + 0.80)


@dataclass(frozen=True)
class ExclusiveRM65AgentPrimitiveComposition:
    """Inspectible result of one exact, still-default-off composition."""

    lower: ExclusiveRM65SkillLowerAssembly
    frame_graph: CoordinateFrameGraph
    attachment: RealTcpPoseObserveAttachmentSpec
    port_config: RealAgentMotionPortConfig
    safety_envelope_ref: SkillDeploymentSafetyEnvelopeRef
    deployment: ExclusiveRealAgentPrimitiveDeployment
    authorization: RealAgentPrimitiveOneShotAuthorization
    application: ExclusiveRealSkillApplication
    resources: object
    perception_source_binding: RealPerceptionSourceBinding | None
    perception_frame_binding: RealPerceptionFrameBinding | None
    perception_capture: RealPerceptionCaptureRegistration | None
    rgb_camera_geometry: RM65RgbCameraGeometry | None
    localization_registration: RealObjectLocalizationRegistration | None
    localization_provider: ObjectLocalizationProvider | None
    segmentation_registration: SegmentationRegistration | None
    station_point_grasp_registration: StationBoundPointGraspRegistration | None
    station_point_grasp_render_registration: (
        StationBoundPointGraspRenderRegistration | None
    )
    point_grasp_planner: object | None
    socket_path: Path
    authorized_side: str
    authorized_motion_sides: tuple[str, ...]
    authorized_action: str | None
    authorized_actions: tuple[str, ...]
    client_id: str
    client_uid: int
    read_only_client_ids: tuple[str, ...]
    socket_mode: int

    def __post_init__(self) -> None:
        if not isinstance(self.lower, ExclusiveRM65SkillLowerAssembly):
            raise TypeError("lower must be an ExclusiveRM65SkillLowerAssembly.")
        if self.deployment.control_runtime.motion_port.config is not self.port_config:
            raise ValueError("deployment and composition must share one port config.")
        if self.application.deployment is not self.deployment:
            raise ValueError("application must own the exact composed deployment.")
        if self.authorization.gateway_session is not self.deployment.gateway_session:
            raise ValueError("authorization must bind the exact Gateway session.")
        if self.application.resources is not self.resources:
            raise ValueError("application must own the exact composed resources.")
        owned_resources = self.resources
        if isinstance(owned_resources, ExclusiveRealGripperResources):
            if owned_resources.ports != self.deployment.gateway_session.gripper_ports:
                raise ValueError("Gripper resources must own the advertised adapter.")
            owned_resources = owned_resources.lower
        elif self.deployment.gateway_session.gripper_ports:
            raise ValueError("Advertised gripper requires application-owned resources.")
        if self.perception_capture is None:
            if owned_resources is not self.lower.ports:
                raise ValueError("motion-only composition must own lower resources.")
            if (
                self.perception_source_binding is not None
                or self.perception_frame_binding is not None
            ):
                raise ValueError(
                    "motion-only composition cannot retain perception bindings."
                )
            if any(
                item is not None
                for item in (
                    self.localization_registration,
                    self.localization_provider,
                    self.segmentation_registration,
                    self.station_point_grasp_registration,
                    self.station_point_grasp_render_registration,
                    self.point_grasp_planner,
                )
            ):
                raise ValueError(
                    "motion-only composition cannot retain grasp resources."
                )
        elif not isinstance(owned_resources, ExclusiveRealPerceptionResources):
            raise ValueError("perception composition requires aggregate resources.")
        elif (
            self.perception_source_binding is None
            or self.perception_frame_binding is None
            or self.perception_capture.frame_binding
            is not self.perception_frame_binding
        ):
            raise ValueError("perception composition requires one exact frame binding.")
        grasp_items = (
            self.localization_registration,
            self.localization_provider,
            self.station_point_grasp_registration,
            self.station_point_grasp_render_registration,
            self.point_grasp_planner,
        )
        if any(item is not None for item in grasp_items) and not all(
            item is not None for item in grasp_items
        ):
            raise ValueError("RM65 point-grasp resources must be assembled together.")
        if any(item is not None for item in grasp_items) and self.rgb_camera_geometry is None:
            raise ValueError("RM65 point-grasp requires camera geometry.")
        if self.segmentation_registration is not None and self.rgb_camera_geometry is None:
            raise ValueError("RM65 segmentation requires camera geometry.")
        if self.application.startup_admission is not self.lower.startup_admission:
            raise ValueError("application must use the exact lower startup gate.")


def assemble_exclusive_rm65_agent_primitive_composition(
    *,
    lower: ExclusiveRM65SkillLowerAssembly,
    right_gripper_backend: object | None = None,
    gripper_backends: Mapping[str, object] | None = None,
    authorized_gripper_sides: tuple[str, ...] | None = None,
    gripper_startup_reconnect_on_fault16: bool = False,
    socket_path: str | Path,
    authorized_side: str,
    authorized_motion_sides: tuple[str, ...] | None = None,
    authorized_action: str | None,
    confirmation_token: str,
    client_id: str,
    client_uid: int,
    ssh_read_only_client_id_by_uid: Mapping[int, str] | None = None,
    authorized_actions: tuple[str, ...] | None = None,
    runtime_session_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    grasp_execution_operation_timeout_sec: float = DEFAULT_REAL_GRASP_EXECUTION_OPERATION_TIMEOUT_SEC,
    max_retained_operations: int = 16,
    peer_idle_timeout_sec: float = 10.0,
    grant_lifetime_sec: float = 5.0,
    camera_config_path: str | Path | None = None,
    camera_registry_path: str | Path | None = None,
    camera_single_start: bool = False,
    artifact_root: str | Path | None = None,
    planned_trajectory_dir: str | Path | None = None,
    camera_mount_urdf_path: str | Path | None = None,
    captured_robot_filter_config: RM65CapturedRobotObservationFilterConfig | None = None,
    captured_robot_observation_urdf_path: str | Path | None = None,
    self_observation_proxy_urdf_path: str | Path | None = None,
    object_localizer: object | None = None,
    point_grasp_planner: object | None = None,
    enable_grasp_observation: bool = False,
    enable_spatial_goals: bool = False,
    enable_tcp_pose_comparison: bool = False,
    enable_back_projection: bool = False,
    enable_surface_anchor_v2: bool = False,
    enable_anchor_motion_v2: bool = False,
    enable_tool_object_relations: bool = False,
    enable_tool_hover_queries: bool = False,
    tool_hover_region_budget_sec: float | None = None,
    tool_hover_candidate_budget_sec: float | None = None,
    enable_point_grasp_finger_exchange: bool = False,
    debug_render_object_coordinates: bool = False,
    enable_point_grasp_arm_reachability: bool = False,
    grasp_trajectory_config: RM65GraspTrajectoryPlanningConfig | None = None,
    safe_motion_observation_config: SafeMotionObservationConfig | None = None,
    safe_motion_debug_dir: str | Path | None = None,
    clock: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
) -> ExclusiveRM65AgentPrimitiveComposition:
    """Bind one lower RM65 graph to the existing real Skill/Gateway contracts."""

    if not isinstance(lower, ExclusiveRM65SkillLowerAssembly):
        raise TypeError("lower must be an ExclusiveRM65SkillLowerAssembly.")
    if not isinstance(debug_render_object_coordinates, bool):
        raise TypeError("debug_render_object_coordinates must be bool.")
    if not isinstance(enable_point_grasp_arm_reachability, bool):
        raise TypeError("enable_point_grasp_arm_reachability must be bool.")
    if not isinstance(enable_grasp_observation, bool):
        raise TypeError("enable_grasp_observation must be bool.")
    if not isinstance(enable_spatial_goals, bool):
        raise TypeError("enable_spatial_goals must be bool.")
    if not isinstance(enable_tcp_pose_comparison, bool):
        raise TypeError("enable_tcp_pose_comparison must be bool.")
    if not isinstance(enable_back_projection, bool):
        raise TypeError("enable_back_projection must be bool.")
    if not isinstance(enable_surface_anchor_v2, bool):
        raise TypeError("enable_surface_anchor_v2 must be bool.")
    if not isinstance(enable_anchor_motion_v2, bool):
        raise TypeError("enable_anchor_motion_v2 must be bool.")
    if not isinstance(enable_tool_object_relations, bool) or not isinstance(enable_tool_hover_queries, bool):
        raise TypeError("Tool-entity Skill switches must be bool.")
    if not enable_tool_hover_queries and (tool_hover_region_budget_sec is not None
                                          or tool_hover_candidate_budget_sec is not None):
        raise ValueError("Tool hover search budgets require enabled tool hover queries.")
    if not isinstance(enable_point_grasp_finger_exchange, bool):
        raise TypeError("enable_point_grasp_finger_exchange must be bool.")
    if grasp_trajectory_config is not None and not isinstance(
        grasp_trajectory_config,
        RM65GraspTrajectoryPlanningConfig,
    ):
        raise TypeError(
            "grasp_trajectory_config must be RM65GraspTrajectoryPlanningConfig "
            "or None."
        )
    side = str(authorized_side).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError("authorized_side must be left or right.")
    motion_sides = tuple(sorted(set((side,) if authorized_motion_sides is None
                                   else authorized_motion_sides)))
    if not motion_sides or side not in motion_sides or not set(motion_sides) <= {"left", "right"}:
        raise ValueError("authorized_motion_sides must include the primary side and select left/right.")
    motion_components = tuple(arm + "_arm" for arm in motion_sides)
    if authorized_actions is None:
        if authorized_action is None:
            raise ValueError("authorized_action or authorized_actions is required.")
        requested_actions = (authorized_action,)
    else:
        if authorized_action is not None:
            raise ValueError(
                "authorized_action and authorized_actions are mutually exclusive."
            )
        requested_actions = tuple(authorized_actions)
    normalized_actions = tuple(
        sorted({str(item).strip() for item in requested_actions})
    )
    if not normalized_actions:
        raise ValueError("authorized_actions contains an unreviewed primitive.")
    for item in normalized_actions:
        validate_real_agent_primitive_action(item)
    bilateral = gripper_backends is not None
    if bilateral and right_gripper_backend is not None:
        raise ValueError("Select existing right-gripper v1 or explicit bilateral v2.")
    if not bilateral and right_gripper_backend is not None and side != "right":
        raise ValueError("Gripper v1 is bound to a right-arm session.")
    gripper_backends = (dict(gripper_backends) if bilateral else
                        {} if right_gripper_backend is None else {"right": right_gripper_backend})
    if bilateral and set(gripper_backends) != {"left", "right"}:
        raise ValueError("Gripper v2 requires both left and right backend configurations.")
    if gripper_startup_reconnect_on_fault16 and not gripper_backends:
        raise ValueError("Startup gripper reconnect requires configured gripper backends.")
    if "set_gripper_state" in normalized_actions and not gripper_backends:
        raise ValueError("set_gripper_state requires configured gripper backends.")
    gripper_ports = {
        key + "_gripper": RealDmtacGripperControlPort(
            backend=value, clock=clock, component_id=key+"_gripper")
        for key, value in gripper_backends.items()
    }
    gripper_port = None if bilateral else gripper_ports.get("right_gripper")
    client = _required(client_id, field_name="client_id")
    runtime_id = _required(runtime_session_id, field_name="runtime_session_id")
    domain_id = _required(control_domain_id, field_name="control_domain_id")
    embodiment_id = _required(
        embodiment_instance_id,
        field_name="embodiment_instance_id",
    )
    uid = int(client_uid)
    if isinstance(client_uid, bool) or uid < 0:
        raise ValueError("client_uid must be a non-negative integer.")
    if ssh_read_only_client_id_by_uid is None:
        read_only_pairs: tuple[tuple[str, int], ...] = ()
    elif isinstance(ssh_read_only_client_id_by_uid, Mapping):
        read_only_pairs = tuple(
            (client_id, read_only_uid)
            for read_only_uid, client_id in ssh_read_only_client_id_by_uid.items()
        )
    else:
        raise TypeError("ssh_read_only_client_id_by_uid must be a mapping or None.")
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id=client,
        primary_client_uid=uid,
        ssh_read_only_clients=read_only_pairs,
    )
    if client_bindings.read_only_client_ids:
        validate_agent_gateway_unix_group_parent(socket_path=socket_path)
    endpoint_value, peer_timeout = validate_agent_gateway_unix_server_preflight(
        socket_path=socket_path,
        peer_idle_timeout_sec=peer_idle_timeout_sec,
    )
    endpoint = Path(endpoint_value)
    if not callable(clock) or not callable(sleep):
        raise TypeError("clock and sleep must be callable.")
    perception_values = (
        camera_config_path,
        camera_registry_path,
        artifact_root,
    )
    perception_enabled = all(value is not None for value in perception_values)
    if type(camera_single_start) is not bool:
        raise TypeError("camera_single_start must be boolean.")
    if camera_single_start and not perception_enabled:
        raise ValueError("camera_single_start requires the composed perception resources.")
    if perception_enabled != any(value is not None for value in perception_values):
        raise ValueError(
            "camera_config_path, camera_registry_path, and artifact_root must be "
            "provided together."
        )
    grasp_enabled = point_grasp_planner is not None
    if enable_surface_anchor_v2 and (not enable_back_projection or object_localizer is None):
        raise ValueError("Surface anchor v2 requires back projection and a bound segmenter.")
    if enable_tool_object_relations and (not enable_back_projection or object_localizer is None):
        raise ValueError("Tool-object relations require back projection and a bound segmenter.")
    if enable_surface_anchor_v2 or enable_tool_object_relations:
        timeout_sec = getattr(object_localizer, "timeout_sec", None)
        if (not isinstance(timeout_sec, (int, float)) or isinstance(timeout_sec, bool)
                or not math.isfinite(float(timeout_sec))
                or not 0 < timeout_sec <= REAL_SURFACE_ANCHOR_MAX_SEGMENT_TIMEOUT_SEC):
            raise ValueError("Surface anchor v2 requires a bounded segmenter timeout of at most 45s.")
    observed_point_enabled = "move_to_observed_point" in normalized_actions
    observed_safe_enabled = safe_motion_observation_config is not None
    object_motion_enabled = "move_grasped_object_to_pose" in normalized_actions
    if object_motion_enabled and (not enable_tool_object_relations or not observed_safe_enabled
                                  or "move_tcp_to_pose_safe" not in normalized_actions):
        raise ValueError("Held-object motion requires tool-object relations, the observed safe scene "
                         "and authorized move_tcp_to_pose_safe.")
    if enable_tool_hover_queries and (not enable_surface_anchor_v2 or not observed_safe_enabled
                                      or "move_tcp_to_pose_safe" not in normalized_actions):
        raise ValueError("Tool hover queries require surface anchor v2, the observed safe scene "
                         "and authorized move_tcp_to_pose_safe.")
    inherited_start_clearance_enabled = (
        grasp_trajectory_config is not None
        and grasp_trajectory_config.enable_inherited_start_clearance
    )
    if inherited_start_clearance_enabled and not observed_safe_enabled:
        raise ValueError("Inherited start clearance requires an explicit observed safe scene.")
    ignore_start_collisions_enabled = (
        grasp_trajectory_config is not None and grasp_trajectory_config.ignore_start_collisions)
    if ignore_start_collisions_enabled and not observed_safe_enabled:
        raise ValueError("Ignoring start collisions requires an explicit observed safe scene.")
    if enable_anchor_motion_v2 and (
        not enable_surface_anchor_v2 or not observed_point_enabled or not observed_safe_enabled
    ):
        raise ValueError("Observed-anchor motion v2 requires its capture, action, and safe scene.")
    if observed_point_enabled and (not enable_back_projection or not observed_safe_enabled):
        raise ValueError("Observed-point motion requires back projection and the RGB-D safe scene.")
    if observed_safe_enabled and (not isinstance(safe_motion_observation_config, SafeMotionObservationConfig)
            or not perception_enabled or not {"move_tcp_to_pose_safe", "move_to_observed_point"}.intersection(normalized_actions)):
        raise ValueError("Observed safe motion requires its typed policy, composed cameras and explicit safe move authorization.")
    if safe_motion_debug_dir is not None:
        if not observed_safe_enabled:
            raise ValueError("Safe-motion debug records require observed safe motion.")
        safe_motion_debug_dir = validate_debug_directory(safe_motion_debug_dir)
    if enable_point_grasp_finger_exchange and (not enable_point_grasp_arm_reachability
            or getattr(point_grasp_planner, "backend_id", None) != "graspgenx"):
        raise ValueError("Finger exchange requires GraspGenX and arm reachability.")
    if bool(getattr(point_grasp_planner, "enable_finger_exchange", False)) != enable_point_grasp_finger_exchange:
        raise ValueError("Finger-exchange generator and registered contract differ.")
    if enable_grasp_observation and (
        not grasp_enabled or getattr(point_grasp_planner, "backend_id", None) != "graspgenx"
    ):
        raise ValueError("Frozen grasp observations require a GraspGenX planner.")
    if grasp_enabled and object_localizer is None:
        raise ValueError(
            "Point-grasp requires its bound object localizer."
        )
    if object_localizer is not None and not (grasp_enabled or enable_surface_anchor_v2
                                             or enable_tool_object_relations):
        raise ValueError("Object localizer requires point grasp, surface anchor v2 or tool-object relations.")
    if grasp_enabled and not perception_enabled:
        raise ValueError("Point-grasp requires the composed camera resources.")
    if enable_back_projection and not perception_enabled:
        raise ValueError("Back projection requires composed cameras and capture-time geometry.")
    if (grasp_enabled or enable_back_projection or observed_safe_enabled) != (camera_mount_urdf_path is not None):
        raise ValueError(
            "Point-grasp three-view rendering requires one explicit camera mount URDF."
        )
    if debug_render_object_coordinates and not grasp_enabled:
        raise ValueError(
            "Debug object-coordinate rendering requires point-grasp resources."
        )
    if enable_point_grasp_arm_reachability and not grasp_enabled:
        raise ValueError("Point-grasp arm reachability requires point-grasp resources.")
    if (
        grasp_enabled
        and getattr(point_grasp_planner, "point_segmenter", None)
        is not object_localizer
    ):
        raise ValueError("Point-grasp must share the exact object-localizer instance.")

    envelope_profile = lower.target_safety.envelope_profile
    for arm in motion_sides:
        constraint = envelope_profile.side_config_by_side.get(arm)
        if constraint is None or not constraint.enabled:
            raise ValueError(f"selected lower envelope does not actively constrain {arm!r}.")

    profile = lower.profile
    if camera_mount_urdf_path is None and any(value is not None for value in (
        captured_robot_filter_config, captured_robot_observation_urdf_path,
        self_observation_proxy_urdf_path,
    )):
        raise ValueError("Captured-robot filter options require a physical camera mount URDF.")
    captured_robot_observation_filter = (
        None if camera_mount_urdf_path is None else
        RM65CapturedRobotObservationFilter.for_binding(
            urdf_path=camera_mount_urdf_path, config=captured_robot_filter_config,
            observation_urdf_path=captured_robot_observation_urdf_path,
            self_observation_proxy_urdf_path=self_observation_proxy_urdf_path,
            tool_tcp_pose_flange_m_rad_by_side=profile.tool_tcp_pose_flange_m_rad_by_side,
        )
    )
    if profile.station_world_frame_id != RM65_SKILL_EXECUTION_FRAME_ID_V1:
        raise ValueError(
            "lower Station world frame does not match the real Skill execution frame"
        )
    if RM65_SKILL_EXECUTION_FRAME_ID_V1 != REAL_PERCEPTION_SOURCE_FRAME_ID_V1:
        raise ValueError("RM65 execution and Real Depth source frames disagree.")
    if RM65_SKILL_STATION_FRAME_ID_V1 != REAL_PERCEPTION_STATION_FRAME_ID_V1:
        raise ValueError("RM65 and Real Depth Station frame identities disagree.")
    tool_config_hash = _hash(
        {
            "profiles": _public_tool_profiles(lower),
            "tcp_pose_flange_m_rad_by_side": {
                name: tuple(
                    float(value)
                    for value in profile.tool_tcp_pose_flange_m_rad_by_side[name]
                )
                for name in ("left", "right")
            },
        },
        domain="dexteleop.real_skill.rm65_tool_inventory.v1",
    )
    backend_inventory_hash = _hash(
        {
            "backend_kind": "dual_rm65_position",
            "arm_ip_assignment": dict(profile.arm_ip_by_side),
            "position_profile": profile.position_profile.to_metadata(),
            "tool_config_hash": tool_config_hash,
        },
        domain="dexteleop.real_skill.rm65_backend_inventory.v1",
    )
    scene_mount_hash = _hash(
        {
            name: profile.scene_world_from_arm_base_by_side[name].tolist()
            for name in ("left", "right")
        },
        domain="dexteleop.real_skill.rm65_scene_mounts.v1",
    )
    station_geometry_hash = _hash(
        {
            "source": "collision_scene_urdf_fixed_joint_chain",
            "world_frame_id": profile.station_world_frame_id,
            "collision_scene_profile": profile.collision_scene_profile,
            "scene_mount_hash": scene_mount_hash,
            "scene_world_from_canonical_station": (
                profile.scene_world_from_canonical_station.tolist()
            ),
        },
        domain="dexteleop.real_skill.rm65_station_geometry.v1",
    )
    station_frame_binding = build_real_perception_station_frame_binding(
        collision_scene_profile=profile.collision_scene_profile,
        scene_mount_hash=scene_mount_hash,
        scene_world_from_canonical_station=(profile.scene_world_from_canonical_station),
    )
    frame_graph = station_frame_binding.frame_graph
    execution_frame = frame_graph.frame_by_id[RM65_SKILL_EXECUTION_FRAME_ID_V1]
    station_frame = frame_graph.frame_by_id[RM65_SKILL_STATION_FRAME_ID_V1]
    station_manifest_hash = _hash(
        {
            "frame_graph_hash": frame_graph.semantic_content_hash,
            "scene_mount_hash": scene_mount_hash,
            "station_geometry_hash": station_geometry_hash,
            "deployment_profile_hash": profile.semantic_content_hash,
        },
        domain="dexteleop.real_skill.rm65_station_manifest.v1",
    )
    execution_manifest_hash = _hash(
        {
            "lower_owner": "teleop_runtime.skill_deploy",
            "profile_hash": profile.semantic_content_hash,
            "backend_inventory_hash": backend_inventory_hash,
            "frame_graph_hash": frame_graph.semantic_content_hash,
            "target_safety_profile_hash": envelope_profile.semantic_content_hash,
            "active_side": side,
            "inactive_side_semantics": "coupled_solver_hard_hold",
            "authorized_motion_sides": motion_sides,
            "command_owner_count": 1,
            **({"captured_robot_filter": {
                "config_hash": captured_robot_observation_filter.config.semantic_content_hash,
                "input_binding_hash": captured_robot_observation_filter.input_binding_hash,
            }} if captured_robot_observation_filter is not None
                and captured_robot_observation_filter.config.geometry_mode != "legacy" else {}),
            **({"safe_motion_observation": safe_motion_observation_config.identity_payload()}
               if safe_motion_observation_config is not None else {}),
            **({"safe_motion_start_clearance": "ignore-start-contacts-v1"}
               if ignore_start_collisions_enabled else
               {"safe_motion_start_clearance": "inherited-positive-v1"}
               if inherited_start_clearance_enabled else {}),
        },
        domain="dexteleop.real_skill.rm65_execution_manifest.v1",
    )
    target_binding_hash = _hash(
        {
            "embodiment_instance_id": embodiment_id,
            "profile_hash": profile.semantic_content_hash,
            "active_side": side,
            "authorized_motion_sides": motion_sides,
        },
        domain="dexteleop.real_skill.rm65_target_binding.v1",
    )
    target_binding_id = "real-rm65-exclusive:" + target_binding_hash
    source_binding_id = "rm65-hardware-qpos:" + backend_inventory_hash
    source_binding_hash = _hash(
        {
            "source_binding_id": source_binding_id,
            "backend_inventory_hash": backend_inventory_hash,
            "read_semantics": (
                "independently_sampled_cached_authoritative_dual_hardware_qpos"
                if lower.ports.profile.background_measurement_sampling
                else "synchronous_authoritative_dual_hardware_qpos"
            ),
            "state_refresh_hz": lower.ports.profile.state_refresh_hz,
            "max_measurement_age_sec": (lower.ports.profile.max_measurement_age_sec),
        },
        domain="dexteleop.real_skill.rm65_source_binding.v1",
    )
    observation_plan_hash = _hash(
        {
            "source_binding_hash": source_binding_hash,
            "projection": "coupled_fk_to_tool_tcp_world",
            "component_ids": motion_components,
            "pull_semantics": (
                "continuous_lower_sampling_plus_skill_cycle_consumption"
                if lower.ports.profile.background_measurement_sampling
                else "gateway_request_or_skill_cycle"
            ),
        },
        domain="dexteleop.real_skill.rm65_observation_plan.v1",
    )
    observation_capability_hash = _hash(
        {
            "observation_kind": "tcp_pose",
            "representation": "xyz_quat_xyzw",
            "controlled_frame": "tool_tcp",
            "reference_frame": execution_frame.frame_id,
            "bounded_components": motion_components,
        },
        domain="dexteleop.real_skill.rm65_observation_capability.v1",
    )
    observation_binding_id = "rm65-tool-tcp-fk:" + observation_plan_hash
    observation_binding_hash = _hash(
        {
            "observation_binding_id": observation_binding_id,
            "observation_plan_hash": observation_plan_hash,
            "observation_capability_hash": observation_capability_hash,
            "tool_config_hash": tool_config_hash,
            "station_geometry_hash": station_geometry_hash,
        },
        domain="dexteleop.real_skill.rm65_observation_binding.v1",
    )
    pose_schema_hash = _hash(
        {
            "representation": "cartesian_pose",
            "layout": "xyz_quat_xyzw",
            "quaternion_order": "xyzw",
            "reference_frame_id": execution_frame.frame_id,
            "controlled_frame_id": "tool_tcp",
        },
        domain="dexteleop.real_skill.rm65_tcp_pose_schema.v1",
    )
    maximum_age_ms = min(
        200,
        max(1, round(lower.ports.profile.max_measurement_age_sec * 1000.0)),
    )
    attachment = RealTcpPoseObserveAttachmentSpec(
        runtime_session_id=runtime_id,
        control_domain_id=domain_id,
        embodiment_instance_id=embodiment_id,
        target_binding_id=target_binding_id,
        component_ids=motion_components,
        station_manifest_hash=station_manifest_hash,
        execution_manifest_hash=execution_manifest_hash,
        backend_inventory_hash=backend_inventory_hash,
        source_binding_id=source_binding_id,
        source_binding_hash=source_binding_hash,
        observation_binding_id=observation_binding_id,
        observation_binding_hash=observation_binding_hash,
        observation_plan_hash=observation_plan_hash,
        observation_capability_hash=observation_capability_hash,
        pose_schema_hash=pose_schema_hash,
        reference_frame_id=execution_frame.frame_id,
        reference_frame_spec_hash=execution_frame.semantic_content_hash,
        frame_graph_hash=frame_graph.semantic_content_hash,
        tool_config_hash=tool_config_hash,
        # ``calibration_hash`` is a v2 wire compatibility field. Independent
        # Skill deployment binds it to lower-owned Station geometry; it never
        # hashes or loads a TeleopCalibration.
        calibration_hash=station_geometry_hash,
        clock_domain_id=RM65_SKILL_CLOCK_DOMAIN_ID_V1,
        producer_epoch=1,
        max_measurement_age_ms=maximum_age_ms,
    )
    port_config = RealAgentMotionPortConfig(
        active_sides=motion_sides,
        component_id_by_side={arm: f"{arm}_arm" for arm in motion_sides},
        execution_frame_id=execution_frame.frame_id,
        execution_frame_spec_hash=execution_frame.semantic_content_hash,
        exclusive_command_claim=True,
    )
    constraint_set_hash = _hash(
        {
            "profile_id": envelope_profile.profile_id,
            "profile_hash": envelope_profile.semantic_content_hash,
            "active_side": side,
            "side_constraints": {arm: envelope_profile.side_config_by_side[arm].semantic_payload()
                                 for arm in motion_sides},
            "application": "active_final_pre_ik",
        },
        domain="dexteleop.real_skill.rm65_safety_constraint_set.v1",
    )
    envelope_ref = SkillDeploymentSafetyEnvelopeRef(
        object_id=f"real-rm65-envelope:{envelope_profile.profile_id}:{'+'.join(motion_sides)}",
        schema="dexteleop.deployment_safety_envelope.v1",
        content_hash=envelope_profile.semantic_content_hash,
        constraint_set_hash=constraint_set_hash,
        allowed_route_ids=_route_ids(
            include_safe_pose="move_tcp_to_pose_safe" in normalized_actions,
            include_observed_point=observed_point_enabled,
            include_observed_anchor_v2=enable_anchor_motion_v2,
            include_grasp_approach_plan=grasp_enabled,
            include_object_motion=object_motion_enabled,
        ),
    )
    perception_source_binding = None
    perception_frame_binding = None
    perception_capture = None
    rgb_camera_geometry = None
    rgbd_geometry_registrations = ()
    localization_registration = None
    localization_provider = None
    segmentation_registration = None
    station_point_grasp_registration = None
    station_point_grasp_render_registration = None
    artifact_store = None
    provider_factory = None
    arm_reachability_profile = None
    reachability_config = (
        RM65GraspReachabilityConfig() if enable_point_grasp_arm_reachability else None
    )
    resources: object = lower.ports
    if perception_enabled:
        assert camera_config_path is not None
        assert camera_registry_path is not None
        assert artifact_root is not None
        root = Path(artifact_root)
        if not root.is_absolute():
            raise ValueError("artifact_root must be an absolute path.")
        perception_source_binding = build_real_perception_source_binding(
            camera_config_path=camera_config_path,
            camera_registry_path=camera_registry_path,
            scene_world_from_arm_base_by_side=(
                lower.profile.scene_world_from_arm_base_by_side
            ),
            single_start=camera_single_start,
        )
        perception_frame_binding = station_frame_binding
        if grasp_enabled or enable_back_projection or observed_safe_enabled:
            assert camera_mount_urdf_path is not None
            rgb_camera_geometry = build_rm65_rgb_camera_geometry(
                source_binding=perception_source_binding,
                target_builder=lower.target_builder,
                camera_registry_path=camera_registry_path,
                camera_mount_urdf_path=camera_mount_urdf_path,
                scene_world_from_arm_base_by_side=(
                    lower.profile.scene_world_from_arm_base_by_side
                ),
                scene_world_from_canonical_station=(
                    lower.profile.scene_world_from_canonical_station
                ),
            )
        catalog_provider = RealSenseAgentPerceptionProvider(
            source_binding=perception_source_binding,
            current_station_revision=lambda: 0,
            rgb_camera_geometry=rgb_camera_geometry,
        )
        perception_capture = build_real_perception_capture_registration_v2(
            rgb_source_catalog=catalog_provider.rgb_source_catalog(),
            depth_source_catalog=catalog_provider.depth_source_catalog(),
            aligned_rgbd_source_catalog=(
                catalog_provider.aligned_rgbd_source_catalog()
            ),
            runtime_session_id=runtime_id,
            control_domain_id=domain_id,
            target_binding_id=target_binding_id,
            station_manifest_hash=station_manifest_hash,
            execution_manifest_hash=execution_manifest_hash,
            source_binding_hash=perception_source_binding.semantic_content_hash,
            frame_binding=perception_frame_binding,
        )
        if enable_back_projection:
            from .back_projection import build_rgbd_geometry_registrations

            rgbd_geometry_registrations = build_rgbd_geometry_registrations(
                perception_capture, roles=tuple(row["role"]
                    for row in catalog_provider.aligned_rgbd_source_catalog()))
        if object_localizer is not None:
            assert object_localizer is not None
            assert rgb_camera_geometry is not None
            localization_spec = getattr(object_localizer, "model_spec", None)
            localization_model_id = str(
                getattr(localization_spec, "model_profile_id", "")
            ).strip()
            localization_model_hash = str(
                getattr(localization_spec, "semantic_content_hash", "")
            ).strip()
            localizer_hash = str(
                getattr(object_localizer, "semantic_content_hash", "")
            ).strip()
            if not all((localization_model_id, localization_model_hash, localizer_hash)):
                raise ValueError("Object localizer lacks verified profile identity.")
            segmentation_registration = build_segmentation_registration(
                target_kind="real",
                current_mode="real_active",
                source_id_by_role={
                    str(item["role"]): str(item["source_id"])
                    for item in catalog_provider.rgb_source_catalog()
                },
                model_profile_id=localization_model_id,
                model_profile_hash=localization_model_hash,
                inference_implementation_hash=localizer_hash,
                runtime_session_id=runtime_id,
                control_domain_id=domain_id,
                target_binding_id=target_binding_id,
                station_manifest_hash=station_manifest_hash,
                execution_manifest_hash=execution_manifest_hash,
            )
        if grasp_enabled:
            assert point_grasp_planner is not None
            planner_spec = getattr(point_grasp_planner, "model_spec", None)
            planner_model_id = str(
                getattr(planner_spec, "model_profile_id", "")
            ).strip()
            planner_model_hash = str(
                getattr(planner_spec, "semantic_content_hash", "")
            ).strip()
            planner_hash = str(
                getattr(point_grasp_planner, "semantic_content_hash", "")
            ).strip()
            backend_id = str(getattr(point_grasp_planner, "backend_id", "")).strip()
            if not all(
                (
                    localization_model_id,
                    localization_model_hash,
                    localizer_hash,
                    planner_model_id,
                    planner_model_hash,
                    planner_hash,
                    backend_id,
                )
            ):
                raise ValueError("Point-grasp models lack verified profile identity.")
            localization_registration = build_object_localization_registration(
                target_kind="real",
                inference_execution_mode=getattr(object_localizer, "inference_execution_mode", "in_process"),
                model_profile_id=localization_model_id,
                model_profile_hash=localization_model_hash,
                inference_implementation_hash=localizer_hash,
                local_path_enabled=False,
                runtime_session_id=runtime_id,
                control_domain_id=domain_id,
                target_binding_id=target_binding_id,
                station_manifest_hash=station_manifest_hash,
                execution_manifest_hash=execution_manifest_hash,
                current_mode="real_active",
            )
            localization_provider = ObjectLocalizationProvider(
                localizer=object_localizer,
                registration=localization_registration,
                allowed_local_image_roots=(),
            )
            station_point_grasp_registration = (
                build_station_bound_point_grasp_registration(
                    aligned_rgbd_source_catalog=tuple(
                        dict(item)
                        for item in catalog_provider.aligned_rgbd_source_catalog()
                    ),
                    source_binding_hash=(
                        perception_source_binding.semantic_content_hash
                    ),
                    frame_binding_hash=perception_frame_binding.semantic_content_hash,
                    frame_graph_hash=frame_graph.semantic_content_hash,
                    source_frame_id=perception_frame_binding.source_frame_id,
                    source_frame_spec_hash=(
                        perception_frame_binding.source_frame_spec_hash
                    ),
                    reference_frame_id=perception_frame_binding.reference_frame_id,
                    reference_frame_spec_hash=(
                        perception_frame_binding.reference_frame_spec_hash
                    ),
                    backend_id=backend_id,
                    model_profile_id=planner_model_id,
                    model_profile_hash=planner_model_hash,
                    inference_implementation_hash=planner_hash,
                    runtime_session_id=runtime_id,
                    control_domain_id=domain_id,
                    target_binding_id=target_binding_id,
                    station_manifest_hash=station_manifest_hash,
                    execution_manifest_hash=execution_manifest_hash,
                    current_mode="real_active",
                    frozen_observation=enable_grasp_observation,
                    finger_exchange=enable_point_grasp_finger_exchange,
                    arm_reachability_profile=(
                        StationPointGraspArmReachabilityProfile(
                            arm_side=side,
                            wrist_camera_side_axis_tool=(
                                load_rm65_wrist_camera_side_axis_tool(
                                    camera_mount_urdf_path=camera_mount_urdf_path,
                                    side=side,
                                    tool_tcp_pose_flange_m_rad=lower.profile.tool_tcp_pose_flange_m_rad_by_side[side],
                                ) if backend_id == "graspgenx" else None
                            ),
                            evaluator_profile_id=(
                                "rm65.global_fk_multistart_reachability.v1"
                            ),
                            evaluator_profile_hash=(
                                reachability_config.semantic_content_hash
                            ),
                            solver_reference_frame_id=execution_frame.frame_id,
                            solver_reference_frame_spec_hash=(
                                execution_frame.semantic_content_hash
                            ),
                            frame_graph_hash=frame_graph.semantic_content_hash,
                            time_budget_ms=reachability_config.time_budget_ms,
                            max_solve_cycles=reachability_config.max_solve_cycles,
                            max_measured_qpos_age_ms=float(maximum_age_ms),
                            standard_position_tolerance_m=(
                                reachability_config.standard_position_tolerance_m
                            ),
                            standard_orientation_tolerance_deg=(
                                reachability_config.standard_orientation_tolerance_deg
                            ),
                            basic_position_tolerance_m=(
                                reachability_config.basic_position_tolerance_m
                            ),
                            basic_orientation_tolerance_deg=(
                                reachability_config.basic_orientation_tolerance_deg
                            ),
                        )
                        if reachability_config is not None
                        else None
                    ),
                )
            )
            arm_reachability_profile = (
                station_point_grasp_registration.profile.arm_reachability_profile
            )
            station_point_grasp_render_registration = (
                build_station_bound_point_grasp_render_registration(
                    point_grasp_registration=station_point_grasp_registration,
                    render_rgb_source_catalog=tuple(
                        dict(item)
                        for item in catalog_provider.rgb_geometry_source_catalog()
                    ),
                    camera_geometry_binding_hash=(
                        rgb_camera_geometry.semantic_content_hash
                    ),
                    debug_target_coordinate_overlay=(debug_render_object_coordinates),
                )
            )
        partition = _hash(
            {"runtime_session_id": runtime_id},
            domain="dexteleop.agent_gateway.artifact_store_partition.v1",
        )
        artifact_store = RecoverableAgentArtifactStore(
            root=root / partition,
            quota_bytes=512 * 1024 * 1024,
            max_artifact_bytes=REAL_CAPTURE_MAX_ARTIFACT_BYTES_V1,
            max_chunk_bytes=1024 * 1024,
        )

        def provider_factory(port, current_station_revision):
            assert perception_source_binding is not None
            return RealSenseAgentPerceptionProvider(
                source_binding=perception_source_binding,
                current_station_revision=current_station_revision,
                max_frame_age_ms=perception_capture.profile.max_frame_age_ms,
                max_inter_camera_skew_ms=(
                    perception_capture.profile.max_inter_camera_skew_ms
                ),
                rgb_camera_geometry=rgb_camera_geometry,
                clock=clock,
            )

    def measured_qpos_observer(cut, *, derived_available_at):
        if rgb_camera_geometry is None:
            return
        rgb_camera_geometry.publish_measured_qpos(
            qpos_by_side=cut.qpos_by_side,
            measured_at=cut.measured_at,
            # FK is derived only after this control cycle has consumed and
            # synchronized the lower qpos cut.  It cannot inherit the earlier
            # source availability timestamp.
            available_at=derived_available_at,
            evidence_refs=cut.evidence_refs,
        )

    try:
        convergence_profile = RealAgentPrimitiveConvergenceProfile()
        grasp_trajectory_planner = RM65GraspJointTrajectoryPlanner(
            profile=lower.profile,
            planning_target_builder=build_exclusive_rm65_skill_target_builder(
                lower.profile
            ),
            target_envelope_filter=(lower.target_builder.pre_ik_target_safety_filter),
            config=grasp_trajectory_config,
            reachability_config=reachability_config,
            trajectory_record_dir=planned_trajectory_dir,
        )
        safe_pose_trajectory_planner = None
        safe_pose_trajectory_planner_factory = None
        planning_geometry_resources = None
        if captured_robot_observation_filter is not None:
            planning_geometry_resources = MotionPlanningGeometryResources(
                self_filter=captured_robot_observation_filter)
        if {"move_tcp_to_pose_safe", "move_to_observed_point"}.intersection(normalized_actions):
            safe_config = (grasp_trajectory_config or RM65GraspTrajectoryPlanningConfig()).for_tcp_pose_move(
                translation_tolerance_m=convergence_profile.translation_tolerance_m,
                rotation_tolerance_rad=convergence_profile.rotation_tolerance_rad)
            if observed_safe_enabled:
                safe_config = replace(safe_config,
                    minimum_observed_clearance_m=safe_motion_observation_config.minimum_clearance_m)

            safe_static_planner = None
            safe_static_planner_lock = threading.Lock()

            def safe_planner_factory(cloud=None):
                nonlocal safe_static_planner
                with safe_static_planner_lock:
                    if safe_static_planner is None:
                        safe_static_planner = RM65GraspJointTrajectoryPlanner(profile=lower.profile,
                            planning_target_builder=build_exclusive_rm65_skill_target_builder(lower.profile),
                            target_envelope_filter=lower.target_builder.pre_ik_target_safety_filter,
                            config=safe_config, reachability_config=reachability_config,
                            trajectory_record_dir=planned_trajectory_dir)
                return (safe_static_planner if cloud is None else
                        partial(safe_static_planner.plan_with_observed_cloud, cloud))

            if observed_safe_enabled:
                assert captured_robot_observation_filter is not None
                planning_geometry_resources.planner_factory = safe_planner_factory
                assert perception_source_binding is not None
                aligned_roles = {item["role"] for item in catalog_provider.aligned_rgbd_source_catalog()}
                if not set(safe_motion_observation_config.roles) <= aligned_roles:
                    raise ValueError("Required safe-motion views are absent from the camera profile.")
                if lower.profile.station_world_frame_id != perception_frame_binding.source_frame_id:
                    raise ValueError("Safe obstacle camera geometry and lower world frame differ.")

                def safe_pose_trajectory_planner_factory(provider, current_revision):
                    context_files = None
                    if safe_motion_debug_dir is not None:
                        context_files = {
                            "camera-config.json": camera_config_path, "camera-registry.json": camera_registry_path,
                            "robot.urdf": camera_mount_urdf_path, "envelope-profile.json": lower.profile.envelope_profile_path,
                            **{f"{side}-tool-profile.json": path
                               for side, path in lower.profile.tool_profile_path_by_side.items()},
                        }
                        deployment_metadata = Path(__file__).resolve().parents[3] / "DEPLOYMENT.json"
                        if deployment_metadata.is_file():
                            context_files["deployment.json"] = deployment_metadata
                    return ObservedSafeMovePlanner(provider=provider, current_station_revision=current_revision,
                        reference_frame_id=lower.profile.station_world_frame_id,
                        self_filter=captured_robot_observation_filter, config=safe_motion_observation_config,
                        planner_factory=safe_planner_factory, clock=clock,
                        non_motion_read_window=lower.ports.non_motion_read_window,
                        debug_record_dir=safe_motion_debug_dir,
                        debug_context={"runtime_session_id": runtime_session_id,
                                       "deployment_profile_hash": lower.profile.semantic_content_hash},
                        debug_context_files=context_files,
                        observation_record_dir=(None if planned_trajectory_dir is None
                            else Path(planned_trajectory_dir) / "observations"))
            else:
                safe_pose_trajectory_planner = safe_planner_factory()

        tool_entity_options = None
        if enable_tool_object_relations or enable_tool_hover_queries:
            from deployments.perception.dmtac_tool_entity import default_tool_entity_by_side

            from .tool_entity import RealToolEntityOptions, RealToolHoverPlanner

            assert camera_mount_urdf_path is not None
            tool_entities = default_tool_entity_by_side(profile.tool_tcp_pose_flange_m_rad_by_side,
                profile.tool_profile_by_side, urdf_path=camera_mount_urdf_path)
            if not tool_entities:
                raise ValueError("Tool-entity Skills need the reviewed DMTAC_M tool on at least one arm.")
            public_from_geometry, _ = frame_graph.transform_pose_matrix(np.eye(4),
                source_reference_frame_id=execution_frame.frame_id,
                target_reference_frame_id=station_frame.frame_id)
            # A request-owned FK builder, created on first use; the control
            # loop's builder keeps its own scratch.
            tool_fk_lock = threading.Lock()
            tool_fk_builders = []

            def tool_tcp_poses(qpos_by_side):
                with tool_fk_lock:
                    if not tool_fk_builders:
                        tool_fk_builders.append(build_exclusive_rm65_skill_target_builder(lower.profile))
                    return tool_fk_builders[0].tool_tcp_world_targets_for_qpos_pair(
                        left_qpos=np.asarray(qpos_by_side["left"], dtype=float),
                        right_qpos=np.asarray(qpos_by_side["right"], dtype=float))

            tool_hover_planner = None
            if enable_tool_hover_queries:
                hover_config = safe_config

                def tool_hover_planner_factory():
                    # Same reviewed safe-move configuration, separate instance:
                    # a query never holds the motion planner's lock.
                    return RM65GraspJointTrajectoryPlanner(profile=lower.profile,
                        planning_target_builder=build_exclusive_rm65_skill_target_builder(lower.profile),
                        target_envelope_filter=lower.target_builder.pre_ik_target_safety_filter,
                        config=hover_config, reachability_config=reachability_config)

                tool_hover_planner = RealToolHoverPlanner(planner_factory=tool_hover_planner_factory,
                    self_filter=captured_robot_observation_filter,
                    observation_config=safe_motion_observation_config,
                    geometry_frame_id=lower.profile.station_world_frame_id,
                    translation_tolerance_m=convergence_profile.translation_tolerance_m,
                    rotation_tolerance_rad=convergence_profile.rotation_tolerance_rad,
                    start_clearance=("ignore-start-contacts-v1" if hover_config.ignore_start_collisions
                                     else "inherited-positive-v1" if hover_config.enable_inherited_start_clearance
                                     else "strict"),
                    max_inter_camera_skew_sec=perception_capture.profile.max_inter_camera_skew_ms / 1000.0,
                    profile_identity={"execution_manifest_hash": execution_manifest_hash,
                        "planning_config": json.loads(json.dumps(asdict(hover_config))),
                        "observation": json.loads(json.dumps(safe_motion_observation_config.identity_payload()))})
            tool_entity_options = RealToolEntityOptions(enable_relations=enable_tool_object_relations,
                enable_hover=enable_tool_hover_queries, tool_entity_by_side=tool_entities,
                tool_tcp_poses=tool_tcp_poses, public_from_geometry=public_from_geometry,
                hover_planner=tool_hover_planner,
                hover_roles=(() if not enable_tool_hover_queries
                             else tuple(safe_motion_observation_config.roles)),
                hover_region_budget_sec=tool_hover_region_budget_sec,
                hover_candidate_budget_sec=tool_hover_candidate_budget_sec,
                object_motion=(None if not object_motion_enabled else {
                    "object_id": f"{side}_held_object",
                    "object_workspace_bounds": _public_object_workspace_bounds(
                        public_from_geometry=public_from_geometry,
                        collision_scene_profile=lower.profile.collision_scene_profile),
                    "clock_domain_id": attachment.clock_domain_id}))

        def reachability_evaluator_factory(port):
            if arm_reachability_profile is None:
                raise RuntimeError("Arm reachability profile is unavailable.")
            return RM65PointGraspReachabilityAdapter(
                profile=arm_reachability_profile,
                planner=grasp_trajectory_planner,
                motion_port=port,
                frame_graph=frame_graph,
                candidate_reference_frame_id=station_frame.frame_id,
                arm_base_origin_candidate_m=arm_base_origin_in_candidate_frame(
                    scene_world_from_candidate=(
                        lower.profile.scene_world_from_canonical_station
                    ),
                    scene_world_from_arm_base=(
                        lower.profile.scene_world_from_arm_base_by_side[
                            arm_reachability_profile.arm_side
                        ]
                    ),
                ),
                clock=clock,
            )

        deployment = assemble_exclusive_rm65_agent_primitive_deployment(
            enable_spatial_goals=enable_spatial_goals,
            enable_tcp_pose_comparison=enable_tcp_pose_comparison,
            tool_entity_options=tool_entity_options,
            gripper_port=gripper_port,
            gripper_ports=gripper_ports if bilateral else None,
            primary_component_id=f"{side}_arm",
            target_builder=lower.target_builder,
            rm65_ports=lower.ports,
            port_config=port_config,
            observation_attachment=attachment,
            frame_graph=frame_graph,
            primitive_reference_frame_id=station_frame.frame_id,
            safety_envelope_refs=(envelope_ref,),
            allowed_client_ids=client_bindings.allowed_client_ids,
            convergence_profile=convergence_profile,
            perception_capture=perception_capture,
            rgbd_geometry_registrations=rgbd_geometry_registrations,
            localization_registration=localization_registration,
            segmentation_registration=segmentation_registration,
            enable_surface_anchor_v2=enable_surface_anchor_v2,
            enable_placement=grasp_enabled,
            localization_provider=localization_provider,
            segmentation_localizer=(
                object_localizer if segmentation_registration is not None else None
            ),
            station_point_grasp_registration=station_point_grasp_registration,
            station_point_grasp_render_registration=(
                station_point_grasp_render_registration
            ),
            station_point_grasp_planner=point_grasp_planner,
            station_point_grasp_reachability_evaluator_factory=(
                reachability_evaluator_factory
                if arm_reachability_profile is not None
                else None
            ),
            grasp_trajectory_planner=grasp_trajectory_planner,
            safe_pose_trajectory_planner=safe_pose_trajectory_planner,
            safe_pose_trajectory_planner_factory=safe_pose_trajectory_planner_factory,
            captured_robot_observation_filter=(captured_robot_observation_filter),
            non_motion_read_window=lower.ports.non_motion_read_window,
            perception_provider_factory=provider_factory,
            measured_qpos_observer=(
                measured_qpos_observer if rgb_camera_geometry is not None else None
            ),
            artifact_store=artifact_store,
            # The aggregate application resource closes the shared store after
            # Gateway executors/models and cameras have settled.
            owns_artifact_store=False,
            grasp_execution_operation_timeout_sec=grasp_execution_operation_timeout_sec,
            max_retained_operations=max_retained_operations,
            clock=clock,
            sleep=sleep,
        )
    except Exception:
        if artifact_store is not None:
            artifact_store.close()
        raise
    if perception_source_binding is not None:
        additional_resources = tuple(
            resource
            for resource in (
                localization_provider if localization_provider is not None else object_localizer,
                deployment.gateway_session.station_point_grasp_provider,
            )
            if resource is not None
        )
        resources = ExclusiveRealPerceptionResources(
            lower=lower.ports,
            camera_sources=perception_source_binding.camera_sources,
            artifact_store=artifact_store,
            additional_resources=additional_resources,
            prepare_static_geometry=(
                planning_geometry_resources.start
                if planning_geometry_resources is not None else None
            ),
        )
    if gripper_ports:
        resources = ExclusiveRealGripperResources(lower=resources,
            startup_reconnect_on_fault16=gripper_startup_reconnect_on_fault16,
            **({"ports": gripper_ports} if bilateral else {"port": gripper_port}))
    authorization = RealAgentPrimitiveOneShotAuthorization(
        authorized_motion_sides=motion_sides,
        authorized_gripper_sides=authorized_gripper_sides,
        gateway_session=deployment.gateway_session,
        side=side,
        authorized_actions=normalized_actions,
        client_id=client,
        confirmation_token=confirmation_token,
        grant_lifetime_sec=grant_lifetime_sec,
        clock=clock,
    )

    def server_factory():
        return deployment.gateway_session.build_unix_json_server(
            socket_path=str(endpoint),
            client_id_by_uid=client_bindings.client_id_by_uid,
            read_only_client_ids=client_bindings.read_only_client_ids,
            socket_mode=client_bindings.socket_mode,
            peer_idle_timeout_sec=peer_timeout,
            grant_resolver=authorization.grant_resolver,
        )

    application = ExclusiveRealSkillApplication(
        deployment=deployment,
        resources=resources,
        startup_admission=lower.startup_admission,
        server_factory=server_factory,
        config=ExclusiveRealSkillApplicationConfig(
            control_hz=profile.position_profile.rates.reference_hz,
        ),
    )
    return ExclusiveRM65AgentPrimitiveComposition(
        lower=lower,
        frame_graph=frame_graph,
        attachment=attachment,
        port_config=port_config,
        safety_envelope_ref=envelope_ref,
        deployment=deployment,
        authorization=authorization,
        application=application,
        resources=resources,
        perception_source_binding=perception_source_binding,
        perception_frame_binding=perception_frame_binding,
        perception_capture=perception_capture,
        rgb_camera_geometry=rgb_camera_geometry,
        localization_registration=localization_registration,
        localization_provider=localization_provider,
        segmentation_registration=segmentation_registration,
        station_point_grasp_registration=station_point_grasp_registration,
        station_point_grasp_render_registration=(
            station_point_grasp_render_registration
        ),
        point_grasp_planner=point_grasp_planner,
        socket_path=endpoint,
        authorized_side=side,
        authorized_motion_sides=motion_sides,
        authorized_action=authorization.action,
        authorized_actions=authorization.authorized_actions,
        client_id=client,
        client_uid=uid,
        read_only_client_ids=tuple(sorted(client_bindings.read_only_client_ids)),
        socket_mode=client_bindings.socket_mode,
    )


__all__ = [
    "RM65_SKILL_CLOCK_DOMAIN_ID_V1",
    "RM65_SKILL_EXECUTION_FRAME_ID_V1",
    "RM65_SKILL_STATION_FRAME_ID_V1",
    "ExclusiveRM65AgentPrimitiveComposition",
    "assemble_exclusive_rm65_agent_primitive_composition",
]

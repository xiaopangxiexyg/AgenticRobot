"""RealMan ports for the shared tool-entity relation and hover Skills.

Captures come from the running RealSense provider with ``require_stationary``
so every view carries its capture-time dual-arm qpos. Hover planning uses a
dedicated instance of the deployment's safe-move planner configuration over
the same observed-cloud builder as ``motion.move_tcp_to_pose_safe@1``; it never
dispatches, never enters the motion read window and never retains a plan.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from PIL import Image

from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.profiles.tool_entity import ToolHoverScene
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.tool_entity_scene import HoverPlanOutcome
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspTrajectoryPlanningError,
)

from .motion_observed_scene import obstacle_cloud_from_captures

# Planner rejections that are specific to one proposed endpoint or path; any
# other reason (for example a blocked start) applies to every proposal.
REAL_HOVER_RETRYABLE_REASONS = frozenset({
    "endpoint_collision_blocked", "endpoint_clearance_blocked", "ik_search_exhausted",
    "path_collision_blocked", "path_target_safety_envelope_blocked", "grasp_path_motion_quality_exceeded",
    "safe_tcp_endpoint_pose_tolerance_exceeded"})
REAL_HOVER_SPEED_SCALE = 0.2
REAL_HOVER_MAX_INTER_CAMERA_SKEW_SEC = 0.1


@dataclass(frozen=True)
class RealToolEntityOptions:
    """Composition-owned inputs for the default-off tool-entity Skills."""

    enable_relations: bool
    enable_hover: bool
    tool_entity_by_side: Mapping[str, object]
    tool_tcp_poses: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]
    public_from_geometry: np.ndarray
    hover_planner: RealToolHoverPlanner | None = None
    hover_roles: tuple[str, ...] = ()
    hover_region_budget_sec: float | None = None
    hover_candidate_budget_sec: float | None = None

    def __post_init__(self):
        if type(self.enable_relations) is not bool or type(self.enable_hover) is not bool:
            raise TypeError("Tool-entity switches must be bool.")
        if not (self.enable_relations or self.enable_hover) or not self.tool_entity_by_side:
            raise ValueError("Tool-entity options need an enabled Skill and a reviewed tool entity.")
        if not callable(self.tool_tcp_poses):
            raise TypeError("tool_tcp_poses must be callable.")
        if self.enable_hover != (self.hover_planner is not None) or (self.enable_hover and not self.hover_roles):
            raise ValueError("Hover queries need their planner and observed-scene roles.")


class RealToolScenePort:
    """Stationary aligned RGB-D views plus lower tool-TCP kinematics in the Station world.

    ``tool_tcp_poses`` must be a request-safe FK owner (a dedicated, locked
    target builder), never the control loop's mutable builder.
    """

    def __init__(self, *, perception_provider, tool_tcp_poses, public_from_geometry):
        if not callable(getattr(perception_provider, "capture_aligned_rgbd", None)):
            raise TypeError("perception_provider must capture aligned RGB-D.")
        if not callable(tool_tcp_poses):
            raise TypeError("tool_tcp_poses must be callable.")
        self.perception_provider = perception_provider
        self.tool_tcp_poses = tool_tcp_poses
        self.public_from_geometry = np.asarray(public_from_geometry, dtype=float).reshape(4, 4).copy()

    def capture_views(self, *, roles, expected_station_revision, cancel_requested):
        captures = []
        for role in roles:
            if cancel_requested():
                raise InterruptedError("Tool-entity capture cancelled.")
            capture = self.perception_provider.capture_aligned_rgbd(
                role=role, expected_station_revision=expected_station_revision, require_stationary=True)
            if capture.role != role or capture.capture_qpos_by_side is None:
                raise RuntimeError("Stationary capture did not return its role with capture-time qpos.")
            captures.append(capture)
        return tuple(captures)

    @staticmethod
    def segmentation_input(capture):
        return LiveRgbSegmentationCapture(source_id=capture.source_id, role=capture.role,
            captured_at=capture.acquired_at, station_revision=capture.station_revision,
            image=Image.fromarray(capture.rgb), rgb_content_hash=capture.rgb_content_hash)

    def tcp_pose_by_side(self, qpos_by_side):
        poses = self.tool_tcp_poses({side: np.asarray(qpos_by_side[side], dtype=float)
                                     for side in ("left", "right")})
        return {side: [float(x) for x in pose_matrix_to_xyzquat(poses[side], name="tool_entity.tcp")]
                for side in ("left", "right")}


class RealToolHoverPlanner:
    """Plan-only hover checks with the deployment's observed safe-move semantics."""

    def __init__(self, *, planner_factory: Callable[[], object], self_filter, observation_config,
                 geometry_frame_id: str, translation_tolerance_m: float, rotation_tolerance_rad: float,
                 start_clearance: str, profile_identity: Mapping[str, object]):
        if not callable(planner_factory) or self_filter is None:
            raise TypeError("Hover planning needs a planner factory and the captured-robot filter.")
        self._planner_factory = planner_factory
        self._planner = None
        self._lock = threading.Lock()
        self.self_filter = self_filter
        self.observation_config = observation_config
        self.geometry_frame_id = geometry_frame_id
        self.translation_tolerance_m = float(translation_tolerance_m)
        self.rotation_tolerance_rad = float(rotation_tolerance_rad)
        self.start_clearance = start_clearance
        self.profile_hash = canonical_content_hash({**dict(profile_identity),
            "speed_scale": REAL_HOVER_SPEED_SCALE, "start_clearance": start_clearance,
            "translation_tolerance_m": self.translation_tolerance_m,
            "rotation_tolerance_rad": self.rotation_tolerance_rad},
            domain="agenticrobot.real.tool_hover_planner.v1")

    def _instance(self):
        with self._lock:
            if self._planner is None:
                self._planner = self._planner_factory()
            return self._planner

    def prepare_scene(self, *, captures, planning_qpos_by_side, cancel_requested):
        if cancel_requested():
            raise InterruptedError("Hover query cancelled.")
        # Build the planner before the search so its one-time preparation is
        # never charged to a single proposal's budget.
        self._instance()
        cloud = obstacle_cloud_from_captures(captures=captures, config=self.observation_config,
            reference_frame_id=self.geometry_frame_id, self_filter=self.self_filter,
            planning_qpos_by_side=planning_qpos_by_side,
            max_inter_camera_skew_sec=REAL_HOVER_MAX_INTER_CAMERA_SKEW_SEC)
        return ToolHoverScene(frame_id=self.geometry_frame_id, points_m=np.asarray(cloud.points_m, dtype=float),
            quantization_margin_m=float(cloud.quantization_margin_m), planning_input=cloud)

    def plan(self, *, scene, qpos_by_side, side, target_pose_xyz_quat_xyzw, time_budget_sec, cancel_requested):
        if cancel_requested():
            raise InterruptedError("Hover query cancelled.")
        planner = self._instance()
        try:
            trajectory = planner.plan_with_observed_cloud(scene.planning_input,
                qpos_by_side={s: np.asarray(q, dtype=float).copy() for s, q in qpos_by_side.items()},
                side=side, target_tool_tcp_world=xyzquat_to_pose_matrix(target_pose_xyz_quat_xyzw),
                speed_scale=REAL_HOVER_SPEED_SCALE, require_grasp_approach=False,
                time_budget_sec=time_budget_sec, pose_goal_policy="within_arrival_tolerance")
        except RM65GraspTrajectoryPlanningError as error:
            if error.reason_code == "path_planning_time_budget_exhausted":
                return HoverPlanOutcome(status="query_incomplete", reason=error.reason_code)
            return HoverPlanOutcome(status="path_rejected", reason=error.reason_code,
                                    retryable=error.reason_code in REAL_HOVER_RETRYABLE_REASONS)
        last = {s: np.asarray(trajectory.qpos_by_side[s][-1], dtype=float) for s in ("left", "right")}
        actual = planner.tool_tcp_poses_for_qpos_pair(qpos_by_side=last)[side]
        return HoverPlanOutcome(status="path_found", reason="planned_not_executed",
            endpoint_pose_xyz_quat_xyzw=tuple(float(x) for x in pose_matrix_to_xyzquat(
                actual, name="tool_hover.endpoint")),
            trajectory_duration_sec=float(trajectory.time_from_start_sec[-1]))


__all__ = ["REAL_HOVER_RETRYABLE_REASONS", "RealToolEntityOptions", "RealToolHoverPlanner", "RealToolScenePort"]

"""Read-only RM65 object transport using the existing IK/path/time owners.

Construct one planner per frozen request. No live solver/oracle is modified;
no grasp-contact allowances from the approach planner apply to a carry path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from robot_motion.joint_limits import load_joint_velocity_limit_profiles
from robot_motion.object_transport import GraspedObjectObservation, rigid_pose
from robot_motion.pose import pose_delta_metrics
from robot_motion.rm65.obstacle_collision import (
    AttachedBoxCollisionProxy, RM65_D405_ATTACHED_COLLISION_PROXIES,
    build_rm65_profile_obstacle_collision_oracle,
)
from robot_motion.rm65.scene_profiles import build_dual_arm_scene_profile
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from dataset_contracts.canonical_encoding import canonical_content_hash
from teleop_runtime.skill_deploy.motion_port import RealAgentJointTrajectory
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import (
    RM65GraspJointTrajectoryPlanner, RM65GraspTrajectoryPlanningError,
)


@dataclass(frozen=True)
class ObjectTransportPlan:
    trajectory: RealAgentJointTrajectory
    details: dict


class RM65ObjectTransportPlanner(RM65GraspJointTrajectoryPlanner):
    """Planning only; a result is not a grasp execution plan or motion grant."""

    def plan_object_move(self, *, observation: GraspedObjectObservation,
                         object_goal: np.ndarray, world_scene: ObstacleSceneSnapshot,
                         speed_scale: float, monotonic=time.perf_counter) -> ObjectTransportPlan:
        if getattr(self, "_transport_request_started", False):
            raise ValueError("Object transport planner is request-local; construct a fresh planner")
        self._transport_request_started = True
        side = observation.side
        goal = rigid_pose(object_goal, name="object goal")
        if (observation.reference_frame_id != self.scene.reference_frame_id
                or world_scene.reference_frame_id != self.scene.reference_frame_id
                or world_scene.reference_frame_spec_hash != self.scene.reference_frame_spec_hash
                or world_scene.revision != observation.scene_revision):
            raise ValueError("Object, world and RM65 planning frames/revisions differ")
        if not np.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("speed_scale must be in (0, 1]")
        current = observation.qpos_by_side
        tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=current["left"], right_qpos=current["right"])[side]
        position_error, rotation_error = pose_delta_metrics(tcp, observation.reference_from_tool)
        if position_error > 1e-5 or rotation_error > 1e-3:
            raise ValueError("Observed tool pose disagrees with the measured-qpos FK/tool profile")
        # The scene owner supplies identified object geometry. Remove only the
        # held object's old world identity, never a spatial neighbourhood.
        fixed = {box.obstacle_id: box for box in self.scene.obstacles}
        if observation.object_id in fixed:
            raise ValueError("A fixed fixture cannot become a held object")
        for box in world_scene.obstacles:
            if box.obstacle_id == observation.object_id:
                continue
            if box.obstacle_id in fixed and box != fixed[box.obstacle_id]:
                raise ValueError("World scene attempts to alter a canonical fixed fixture")
            fixed[box.obstacle_id] = box
        self.scene = replace(self.scene, revision=world_scene.revision,
            obstacles=tuple(fixed.values()),
            minimum_clearance_m=max(self.scene.minimum_clearance_m, world_scene.minimum_clearance_m))
        solver_pose = self.target_builder.solver_world_targets_for_qpos_pair(
            left_qpos=current["left"], right_qpos=current["right"])[side]
        solver_from_box = np.linalg.inv(solver_pose) @ observation.reference_from_object @ observation.object_from_box
        profile = build_dual_arm_scene_profile(self.path_collision_scene_profile)
        # Hand proxies include the actual gripping contact volume. Only this
        # arm's reviewed hand proxies are contact partners; link3, cameras,
        # the other arm and all world obstacles remain checked.
        touch_names = {box.geometry_name for box in profile.hand_proxy_boxes
                       if box.geometry_name.startswith(side+"_hand_proxy_")}
        protected = {name for pair in profile.collision_pairs for name in pair} - touch_names
        protected.update(proxy.geometry_name for proxy in RM65_D405_ATTACHED_COLLISION_PROXIES)
        carried = AttachedBoxCollisionProxy(
            geometry_name="held_object::"+observation.object_id,
            parent_frame_name="arm_link6" if side == "left" else "right_arm_link6",
            center_xyz_m=tuple(solver_from_box[:3, 3]), size_xyz_m=observation.box_size_m,
            rotation_row_major=tuple(solver_from_box[:3, :3].reshape(-1)),
            collision_geometry_names=tuple(sorted(protected)))
        self.oracle = build_rm65_profile_obstacle_collision_oracle(
            scene_profile=self.path_collision_scene_profile, scene=self.scene,
            expected_reference_frame_id=self.scene.reference_frame_id,
            expected_reference_frame_spec_hash=self.scene.reference_frame_spec_hash,
            attached_box_proxies=(*RM65_D405_ATTACHED_COLLISION_PROXIES, carried),
            use_profile_proxy_geometry=True)
        target = observation.tool_goal(goal)
        deadline = monotonic() + self.config.path_planning_time_budget_sec
        self._preflight_start_envelope(qpos_by_side=current, side=side)
        # Reuse the free-space leg. Unlike grasp approach, this has no final
        # target-contact exemption, pregrasp offset or gripper command.
        leg = self._plan_path_leg(qpos_by_side=current, side=side,
            target_tool_tcp_world=target, planning_phase="pregrasp_target",
            prefer_closest_tolerance=False, deadline=deadline, monotonic=monotonic)
        waypoints = tuple(leg.path.waypoints)
        if len(waypoints) == 1:
            waypoints = (waypoints[0], waypoints[0])
        velocity = load_joint_velocity_limit_profiles()[
            self.profile.position_profile.ik_execution.joint_velocity_limit_profile]
        continuous, _, tcp_scale = self._continuous_leg_trajectory(
            waypoints=waypoints, moving_side=side, speed_scale=speed_scale,
            joint_velocity_rad_s=np.deg2rad(velocity.max_velocity_deg_s)*speed_scale,
            reference_hz=float(self.profile.position_profile.rates.ik_hz))
        final_waypoints = tuple({"left": row[:6], "right": row[6:]} for row in continuous.q_rad)
        # Certify the actual smoothed curve, including the carried object, at
        # the same dense resolution used for the original grasp planner.
        validation = self._revalidate(final_waypoints, moving_side=side,
                                      deadline=deadline, monotonic=monotonic)
        final_tcp = self.target_builder.tool_tcp_world_targets_for_qpos_pair(
            left_qpos=continuous.q_rad[-1, :6], right_qpos=continuous.q_rad[-1, 6:])[side]
        object_error = pose_delta_metrics(final_tcp @ observation.tool_from_object, goal)
        if (object_error[0] > self.config.endpoint_position_tolerance_m
                or object_error[1] > self.config.endpoint_orientation_tolerance_deg):
            raise RM65GraspTrajectoryPlanningError("object_goal_residual_exceeded")
        inactive = "right" if side == "left" else "left"
        qpos = {"left": continuous.q_rad[:, :6], "right": continuous.q_rad[:, 6:]}
        if not np.allclose(qpos[inactive], current[inactive], atol=1e-12, rtol=0):
            raise RM65GraspTrajectoryPlanningError("stationary_arm_changed")
        metadata = {"planning_kind": "object_transport", "execution_authorized": False,
            "object_id": observation.object_id, "side": side, "scene_hash": self.scene.semantic_content_hash,
            "scene_revision": observation.scene_revision, "reference_frame_id": observation.reference_frame_id,
            "T_tool_from_object": observation.tool_from_object.tolist(),
            "T_reference_from_object_goal": goal.tolist(), "T_reference_from_tool_goal": target.tolist(),
            "object_box_size_m": list(observation.box_size_m),
            "T_object_from_box": observation.object_from_box.tolist(),
            "pose_evidence_ref": observation.pose_evidence_ref, "grasp_evidence_ref": observation.grasp_evidence_ref,
            "contact_proxy_names": sorted(touch_names), "carried_collision_pairs": list(carried.collision_geometry_names),
            "dense_revalidation": validation, "object_position_error_m": object_error[0],
            "object_orientation_error_deg": object_error[1], "tcp_time_scale": tcp_scale,
            "path_metadata": dict(leg.path.metadata)}
        trajectory = RealAgentJointTrajectory(time_from_start_sec=continuous.timestamps,
            qpos_by_side=qpos, planner_evidence_refs=(
                "object-transport:"+canonical_content_hash(metadata, domain="agenticrobot.object_transport.v1"),
                observation.pose_evidence_ref, observation.grasp_evidence_ref))
        return ObjectTransportPlan(trajectory, metadata)

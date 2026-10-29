"""MuJoCo ports for the shared tool-entity relation and hover Skills.

Views come from one frozen native multi-view cut taken while the motion
application is HOLD with no active operation (the same passive capture the
grasp observation uses), so every view shares one dual-arm qpos. Hover paths
are planned by a private, never-activated MujocoPlannedTcpMotion over the same
observed RGB-D scene, oracle and envelope as ``motion.move_tcp_to_pose_safe@1``.
"""

from __future__ import annotations

import math
import threading
import time
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.profiles.tool_entity import ToolHoverScene
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.rm65.tool_profiles import (
    endpoint_to_tool_transform,
    load_rm65_tool_profile,
)
from robot_motion.tool_entity_scene import (
    HoverPlanOutcome,
    pose_matrix,
    pose_xyz_quat_xyzw,
)
from sim.mujoco.deploy.planned_tcp_motion import (
    MujocoPlannedTcpMotion,
    MujocoTcpPlanRejected,
)

from .motion_observed_scene import mujoco_rgbd_captures_from_cut

# Rejections specific to one proposed endpoint/path (MujocoObjectGoalQuery's
# ordinary rejections plus the per-path inherited egress proofs).
MUJOCO_HOVER_RETRYABLE_REASONS = frozenset({
    "safe_tcp_endpoint_search_exhausted", "safe_tcp_target_blocked", "safe_tcp_retimed_path_blocked",
    "safe_tcp_path_exceeds_control_budget", "safe_tcp_target_envelope_blocked", "safe_tcp_target_exceeds_profile",
    "safe_tcp_motion_quality_exceeded", "safe_tcp_inherited_start_clearance_regressed",
    "safe_tcp_inherited_start_clearance_not_exited", "safe_tcp_ik_failed"})
# Budget ends inside the planner, not a property of the proposal.
MUJOCO_HOVER_INCOMPLETE_REASONS = frozenset({"safe_tcp_path_search_exhausted", "safe_tcp_planning_cancelled"})


def _identity_pose():
    return (0., 0., 0., 0., 0., 0., 1.)


class MujocoToolScenePort:
    """Frozen multi-view RGB-D in the URDF/FCL world plus native tool-TCP FK."""

    def __init__(self, *, application, capture_port, source_binding_hash, tool_profile_path_by_side):
        self.application = application
        self.capture_port = capture_port
        self.source_binding_hash = source_binding_hash
        scene = application.scene
        self._model = scene.model
        self._mujoco = scene.mujoco
        self._addresses = {side: np.asarray([scene.joint_qpos_addr(name) for name in scene.arm_specs[side].joint_names],
                                            dtype=int) for side in ("left", "right")}
        self._link6 = {side: self._mujoco.mj_name2id(self._model, self._mujoco.mjtObj.mjOBJ_BODY, side + "_arm_link6")
                       for side in ("left", "right")}
        if any(index < 0 for index in self._link6.values()):
            raise ValueError("Tool FK requires both arm flange bodies.")
        self._flange_from_tcp = {side: endpoint_to_tool_transform(load_rm65_tool_profile(path, side=side))
                                 for side, path in tool_profile_path_by_side.items()}
        self._data = self._mujoco.MjData(self._model)
        self._lock = threading.Lock()
        self.geometry_frame_id = capture_port.source_frame_id
        # Same frame resolution as the observed safe-motion provider.
        self.geometry_from_depth_reference = xyzquat_to_pose_matrix(application.transform_tcp_pose_xyzquat(
            _identity_pose(), source_frame_id=capture_port.reference_frame_id, target_frame_id=self.geometry_frame_id))
        self.public_from_geometry = xyzquat_to_pose_matrix(application.transform_tcp_pose_xyzquat(
            _identity_pose(), source_frame_id=self.geometry_frame_id, target_frame_id=capture_port.reference_frame_id))

    def capture_views(self, *, roles, expected_station_revision, cancel_requested):
        if cancel_requested():
            raise InterruptedError("Tool-entity capture cancelled.")
        cut = self.application.capture_frozen_rgbd_cut(
            source_ids=tuple(self.capture_port.source_id_by_role[role] for role in roles))
        return tuple(mujoco_rgbd_captures_from_cut(cut, roles=tuple(roles), capture_port=self.capture_port,
            geometry_from_depth_reference=self.geometry_from_depth_reference,
            source_binding_hash=self.source_binding_hash, expected_station_revision=expected_station_revision))

    @staticmethod
    def segmentation_input(capture):
        from PIL import Image

        if capture.rgb is None:
            raise RuntimeError("Tool-object relations need the RGB of each view.")
        return LiveRgbSegmentationCapture(source_id=capture.source_id, role=capture.role,
            captured_at=capture.acquired_at, station_revision=capture.station_revision,
            image=Image.fromarray(capture.rgb), rgb_content_hash=capture.rgb_content_hash)

    def tcp_matrices(self, qpos_by_side):
        """World tool-TCP poses from private model data (never the live scene)."""
        with self._lock:
            data = self._data
            data.qpos[:] = self._model.qpos0
            for side, indices in self._addresses.items():
                data.qpos[indices] = np.asarray(qpos_by_side[side], dtype=float)
            self._mujoco.mj_kinematics(self._model, data)
            poses = {}
            for side, body in self._link6.items():
                world_from_flange = np.eye(4)
                world_from_flange[:3, :3] = np.asarray(data.xmat[body], dtype=float).reshape(3, 3)
                world_from_flange[:3, 3] = np.asarray(data.xpos[body], dtype=float)
                poses[side] = world_from_flange @ self._flange_from_tcp[side]
            return poses

    def tcp_pose_by_side(self, qpos_by_side):
        return {side: pose_xyz_quat_xyzw(pose) for side, pose in self.tcp_matrices(qpos_by_side).items()}


class MujocoToolHoverPlanner:
    """Private plan-only safe-TCP checks on the observed scene; nothing is activated."""

    def __init__(self, *, provider, application, scene_port, motion_profile, prepare_projection):
        self.provider = provider
        self.application = application
        self.scene_port = scene_port
        self.profile = motion_profile
        self.prepare_projection = prepare_projection
        self.translation_tolerance_m = float(motion_profile.translation_tolerance_m)
        self.rotation_tolerance_rad = float(motion_profile.rotation_tolerance_rad)
        self.start_clearance = provider.config.start_clearance
        self.motion_side = application.target.planned_tcp_motion.motion_side
        self.profile_hash = canonical_content_hash({
            "observed_scene": provider.semantic_content_hash, "motion_profile": motion_profile.semantic_content_hash,
            "planner": "mujoco_private_collision_checked_joint_path_v1", "start_clearance": self.start_clearance},
            domain="agenticrobot.mujoco.tool_hover_planner.v1")

    def prepare_scene(self, *, captures, planning_qpos_by_side, cancel_requested):
        if cancel_requested():
            raise InterruptedError("Hover query cancelled.")
        cloud, source = self.provider.planning_input_from_captures(captures)
        return ToolHoverScene(frame_id=self.provider.geometry_frame_id, points_m=np.asarray(cloud.points_m, dtype=float),
            quantization_margin_m=float(cloud.quantization_margin_m), planning_input=source)

    def plan(self, *, scene, qpos_by_side, side, target_pose_xyz_quat_xyzw, time_budget_sec, cancel_requested):
        if side != self.motion_side:
            return HoverPlanOutcome(status="path_rejected", reason="tool_hover_side_not_bound")
        app = self.application
        source = app.target.planned_tcp_motion
        rig = app.runtime.rig
        geometry, execution = self.scene_port.geometry_frame_id, self.profile.execution_reference_frame_id
        target = app.transform_tcp_pose_xyzquat(tuple(target_pose_xyz_quat_xyzw), source_frame_id=geometry,
                                                target_frame_id=execution)
        current = self.scene_port.tcp_pose_by_side(qpos_by_side)[side]
        current_execution = app.transform_tcp_pose_xyzquat(tuple(current), source_frame_id=geometry,
                                                           target_frame_id=execution)
        delta = float(np.linalg.norm(np.asarray(target[:3]) - np.asarray(current_execution[:3])))
        relative = pose_matrix(current_execution)[:3, :3].T @ pose_matrix(target)[:3, :3]
        angle = float(np.arccos(np.clip((np.trace(relative) - 1.) / 2., -1., 1.)))
        if delta > self.profile.max_translation_delta_m or angle > self.profile.max_rotation_delta_rad:
            return HoverPlanOutcome(status="path_rejected", reason="safe_tcp_target_exceeds_profile", retryable=True)
        planner = MujocoPlannedTcpMotion(rig=SimpleNamespace(scene=app.scene, config=rig.config,
                capture_scene=rig.capture_scene), task_config=app.task_config, pose_projector=source.pose_projector,
            reference_hz=source.reference_hz, planning_clearance_padding_m=source.planning_clearance_padding_m,
            motion_side=side)
        project, envelope_identity = self.prepare_projection()

        def constraint(model_qpos, pair):
            pose = planner.pose_for_configuration(model_qpos, pair)
            projection = project({side: xyzquat_to_pose_matrix(pose)})
            if projection.modified_sides:
                return JointConfigurationCheck(safe=False, reason="tcp_envelope_blocked",
                                               metadata=projection.to_payload())
            return JointConfigurationCheck(safe=True)

        deadline = time.perf_counter() + max(1e-3, float(time_budget_sec))
        try:
            plan = planner.plan(tuple(target), translation_tolerance_m=self.translation_tolerance_m,
                rotation_tolerance_rad=self.rotation_tolerance_rad,
                cancel_requested=lambda: cancel_requested() or time.perf_counter() >= deadline,
                configuration_constraint=constraint, constraint_identity=envelope_identity,
                path_search_time_budget_sec=max(1e-3, float(time_budget_sec)),
                observed_planning_input=scene.planning_input, deadline_monotonic=deadline,
                normalize_ik_objective=self.profile.normalize_ik_objective)
        except MujocoTcpPlanRejected as error:
            if cancel_requested():
                raise InterruptedError("Hover query cancelled.") from error
            if error.reason_code in MUJOCO_HOVER_INCOMPLETE_REASONS or time.perf_counter() >= deadline:
                return HoverPlanOutcome(status="query_incomplete", reason=error.reason_code)
            return HoverPlanOutcome(status="path_rejected", reason=error.reason_code,
                                    retryable=error.reason_code in MUJOCO_HOVER_RETRYABLE_REASONS)
        finally:
            planner.clear()
        if len(plan.reference_poses) + self.profile.stable_measurement_count >= self.profile.max_control_cycles:
            return HoverPlanOutcome(status="path_rejected", reason="safe_tcp_path_exceeds_control_budget",
                                    retryable=True)
        final = {**{s: np.asarray(q, dtype=float) for s, q in qpos_by_side.items()},
                 side: np.asarray(plan.trajectory.q_rad[-1], dtype=float)}
        endpoint = self.scene_port.tcp_matrices(final)[side]
        duration = float(len(plan.reference_poses)) / float(source.reference_hz)
        if not math.isfinite(duration):
            duration = None
        return HoverPlanOutcome(status="path_found", reason="planned_not_executed",
            endpoint_pose_xyz_quat_xyzw=tuple(pose_xyz_quat_xyzw(endpoint)), trajectory_duration_sec=duration)


__all__ = ["MUJOCO_HOVER_RETRYABLE_REASONS", "MujocoToolHoverPlanner", "MujocoToolScenePort"]

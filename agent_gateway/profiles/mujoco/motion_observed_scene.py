"""Sensor-shaped scene assembly for the original MuJoCo safe TCP owner.

Only calibrated RGB-D and measured arm joints reach the planning oracle. The
native plant/dispatch guard remains independent and is never used to select a
path. No object body pose, segmentation label or native obstacle list is read.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.observed_scene import SafeMotionObservationConfig, obstacle_cloud_from_captures
from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilter,
    RM65CapturedRobotObservationFilterConfig,
)
from robot_motion.rm65.scene_profiles import build_dual_arm_scene_profile
from robot_motion.rm65.obstacle_collision import (
    RM65_D405_ATTACHED_COLLISION_PROXIES,
    build_rm65_profile_obstacle_collision_oracle,
)
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from sim.mujoco.deploy.planned_tcp_motion import (
    OBSERVED_START_CLEARANCE_MODES,
    OBSERVED_START_CLEARANCE_STRICT,
    MujocoObservedJointPlanningInput,
)


def _urdf_geometry_identity(path):
    """Hash the selected local assembly and its actual mesh resources once."""
    path = Path(path).resolve(strict=True)
    urdf = path.read_bytes()
    meshes = {}
    for mesh in ElementTree.fromstring(urdf).iter("mesh"):
        name = mesh.attrib["filename"]
        if "://" in name:
            raise ValueError("Observed planning requires resolved local URDF mesh paths.")
        resource = (path.parent / name).resolve(strict=True)
        meshes[name] = hashlib.sha256(resource.read_bytes()).hexdigest()
    return {"urdf_sha256": hashlib.sha256(urdf).hexdigest(), "meshes": meshes}


@dataclass(frozen=True)
class MujocoObservedSafeMotionConfig:
    robot_urdf_path: Path
    collision_scene_profile: str
    observation: SafeMotionObservationConfig
    maximum_start_drift_rad: float = 0.01
    maximum_inter_camera_skew_sec: float = 0.1
    # Strict keeps every prior identity. The reviewed RealMan egress mode
    # enters the safe profile identity only when selected.
    start_clearance: str = OBSERVED_START_CLEARANCE_STRICT
    _semantic_content_hash: str = field(init=False, repr=False)
    _collision_model_hash: str = field(init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.robot_urdf_path, Path) or not self.robot_urdf_path.is_absolute()
                or not self.robot_urdf_path.is_file()):
            raise ValueError("Observed planning requires an explicit local robot URDF.")
        if (not isinstance(self.collision_scene_profile, str) or not self.collision_scene_profile
                or self.collision_scene_profile.strip() != self.collision_scene_profile):
            raise ValueError("Observed planning requires an explicit robot collision profile.")
        if not isinstance(self.observation, SafeMotionObservationConfig):
            raise TypeError("observation must be SafeMotionObservationConfig.")
        for name in ("maximum_start_drift_rad", "maximum_inter_camera_skew_sec"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if type(self.start_clearance) is not str or self.start_clearance not in OBSERVED_START_CLEARANCE_MODES:
            raise ValueError("start_clearance must be 'strict' or 'inherited-positive-v1'.")
        identity, collision = self._content_identities()
        object.__setattr__(self, "_semantic_content_hash", identity)
        object.__setattr__(self, "_collision_model_hash", collision)

    def _content_identities(self):
        collision_profile = build_dual_arm_scene_profile(self.collision_scene_profile)
        collision = canonical_content_hash({
            "collision_scene_profile": self.collision_scene_profile,
            "collision_geometry": _urdf_geometry_identity(collision_profile.urdf_path),
            "collision_pairs": collision_profile.collision_pairs,
            "fixed_scene_overrides": [asdict(box) for box in collision_profile.fixed_scene_box_overrides],
            "attached_camera_boxes": [asdict(box) for box in RM65_D405_ATTACHED_COLLISION_PROXIES],
        }, domain="agenticrobot.mujoco.observed_collision_model.v1")
        identity = canonical_content_hash({
            "robot_geometry": _urdf_geometry_identity(self.robot_urdf_path),
            "collision_model": collision,
            "observation": asdict(self.observation),
            "maximum_start_drift_rad": self.maximum_start_drift_rad,
            "maximum_inter_camera_skew_sec": self.maximum_inter_camera_skew_sec,
            "self_filter": asdict(RM65CapturedRobotObservationFilterConfig()),
            "planning_inputs": "rgbd_and_measured_joints_only",
            "observed_collision_geometry": "existing_profile_boxes_and_D405",
            **({"start_clearance": self.start_clearance}
               if self.start_clearance != OBSERVED_START_CLEARANCE_STRICT else {}),
        }, domain="agenticrobot.mujoco.observed_safe_motion.v1")
        return identity, collision

    @property
    def semantic_content_hash(self):
        return self._semantic_content_hash

    def require_resources_unchanged(self):
        if self._content_identities() != (self._semantic_content_hash, self._collision_model_hash):
            raise ValueError("Observed planning resources changed after configuration.")

    def bound_profile_hash(self, *, minimum_clearance_m, planning_clearance_padding_m):
        for value in (minimum_clearance_m, planning_clearance_padding_m):
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
                raise ValueError("Observed planning margins must be finite and nonnegative.")
        return canonical_content_hash({"observation": self.semantic_content_hash,
            "minimum_clearance_m": float(minimum_clearance_m),
            "planning_clearance_padding_m": float(planning_clearance_padding_m)},
            domain="agenticrobot.mujoco.bound_observed_safe_motion.v1")


class MujocoObservedSafeMotionProvider:
    def __init__(self, *, config, capture_port, rig, profile, current_station_revision, clock,
                 planning_clearance_padding_m, source_binding_hash):
        if rig.config.realtime_worker is not False:
            raise ValueError("Observed planning requires the deterministic physics owner.")
        if config.robot_urdf_path.resolve() != Path(rig.scene.load_result.conversion.urdf_path).resolve():
            raise ValueError("Self-filter URDF must match the session's known robot assembly.")
        if not set(config.observation.roles) <= set(capture_port.source_id_by_role):
            raise ValueError("Observed planning requires every configured RGB-D role.")
        config.require_resources_unchanged()
        self.config, self.capture_port, self.rig = config, capture_port, rig
        if (not isinstance(source_binding_hash, str) or len(source_binding_hash) != 64
                or any(c not in "0123456789abcdef" for c in source_binding_hash)):
            raise ValueError("Observed planning requires the original RGB-D source profile hash.")
        self.source_binding_hash = source_binding_hash
        self.profile = profile
        self.semantic_content_hash = config.bound_profile_hash(
            minimum_clearance_m=rig.scene.scene_spec.collision.safety_minimum_clearance_m,
            planning_clearance_padding_m=planning_clearance_padding_m)
        self.current_station_revision, self.clock = current_station_revision, clock
        self.self_filter = RM65CapturedRobotObservationFilter(urdf_path=config.robot_urdf_path)
        self.self_filter.prepare_geometry()
        # The public TCP frame may be an arm base, and depth is normally in
        # Station axes. URDF/FCL and the self filter use the known URDF world.
        # Resolve that calibration once, never from a native object's pose.
        self.geometry_frame_id = capture_port.source_frame_id
        frame = capture_port.application.target_frame_profile.frame_graph.frame_by_id[self.geometry_frame_id]
        self.geometry_frame_hash = frame.semantic_content_hash
        self.geometry_from_depth_reference = xyzquat_to_pose_matrix(
            capture_port.application.transform_tcp_pose_xyzquat(
                (0., 0., 0., 0., 0., 0., 1.), source_frame_id=capture_port.reference_frame_id,
                target_frame_id=self.geometry_frame_id))
        # This is known robot/fixture geometry, not a native scene export.
        self.oracle = build_rm65_profile_obstacle_collision_oracle(
            scene_profile=config.collision_scene_profile,
            scene=ObstacleSceneSnapshot(reference_frame_id=self.geometry_frame_id,
                reference_frame_spec_hash=self.geometry_frame_hash, revision=0,
                minimum_clearance_m=(rig.scene.scene_spec.collision.safety_minimum_clearance_m
                    + planning_clearance_padding_m)),
            expected_reference_frame_id=self.geometry_frame_id,
            expected_reference_frame_spec_hash=self.geometry_frame_hash,
            attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
            use_profile_proxy_geometry=True, model_content_hash=config._collision_model_hash,
            validate_model_resources=config.require_resources_unchanged)
        # Every later query uses the materialized immutable model/mesh data.
        # Do not reread resource paths on every frame or every planner edge.
        config.require_resources_unchanged()

    def capture(self, *, operation_id, expected_station_revision, cancel_requested):
        from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

        def require_request():
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            if self.current_station_revision() != expected_station_revision:
                raise MujocoTcpPlanRejected("observed_planning_station_changed")

        require_request()
        # Enter through the existing source owner. Never hold app/rig while
        # waiting for another capture worker to release this source.
        cut = self.capture_port.application.capture_planning_rgbd_cut(
            operation_id=operation_id,
            source_ids=tuple(self.capture_port.source_id_by_role[role]
                for role in self.config.observation.roles),
            cancel_requested=cancel_requested)
        require_request()
        captures = self.captures_from_cut(cut, roles=self.config.observation.roles,
            expected_station_revision=expected_station_revision, require_request=require_request)
        _, source = self.planning_input_from_captures(captures)
        require_request()
        return source

    def captures_from_cut(self, cut, *, roles, expected_station_revision, require_request=lambda: None):
        """Typed RGB-D views of one native cut in the URDF/FCL geometry frame."""
        return mujoco_rgbd_captures_from_cut(cut, roles=roles, capture_port=self.capture_port,
            geometry_from_depth_reference=self.geometry_from_depth_reference,
            source_binding_hash=self.source_binding_hash, expected_station_revision=expected_station_revision,
            require_request=require_request)

    def planning_input_from_captures(self, captures):
        """Observed cloud plus the request-owned planning input for the current measured arms."""
        from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

        current = self.rig.capture_scene(lambda scene: scene.arm_qpos_by_side())
        try:
            cloud = obstacle_cloud_from_captures(captures=captures, config=self.config.observation,
                reference_frame_id=self.geometry_frame_id, self_filter=self.self_filter,
                planning_qpos_by_side=current,
                max_inter_camera_skew_sec=self.config.maximum_inter_camera_skew_sec,
                max_qpos_drift_rad=self.config.maximum_start_drift_rad)
            source = MujocoObservedJointPlanningInput(oracle=self.oracle.for_observed_cloud(cloud),
                qpos_by_component=current, acquired_at=min(c.acquired_at for c in captures),
                maximum_age_sec=self.config.observation.maximum_scene_age_sec,
                maximum_start_drift_rad=self.config.maximum_start_drift_rad, clock=self.clock,
                start_clearance=self.config.start_clearance)
            source.require_current(current)
        except ValueError as error:
            raise MujocoTcpPlanRejected("observed_planning_input_rejected", {"error": str(error)}) from error
        return cloud, source


def mujoco_rgbd_captures_from_cut(cut, *, roles, capture_port, geometry_from_depth_reference, source_binding_hash,
                                  expected_station_revision, require_request=lambda: None):
    """Convert one native multi-view cut into RGB-D captures with capture-time dual-arm qpos.

    ``source_from_camera`` is in the URDF/FCL geometry frame (the capture
    port's source frame); RGB is kept for segmentation consumers.
    """
    from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
    from .back_projection import live_aligned_depth_capture_from_array

    pair = cut.arm_qpos_by_side
    captures = []
    for role, view in zip(roles, cut.views, strict=True):
        require_request()
        source_id = capture_port.source_id_by_role[role]
        if view.view_id != source_id:
            raise MujocoTcpPlanRejected("observed_planning_rgbd_cut_mismatch")
        depth = live_aligned_depth_capture_from_array(
            view.depth_m, captured_at=cut.captured_at,
            camera_pose_world_xyz_quat_xyzw=view.optical_camera_pose_world_xyz_quat_xyzw,
            intrinsics=view.intrinsics, source_id=source_id, role=role,
            station_revision=expected_station_revision,
            application=capture_port.application,
            source_frame_id=capture_port.source_frame_id,
            reference_frame_id=capture_port.reference_frame_id)
        rgb = getattr(view, "rgb", None)
        rgb = None if rgb is None else np.ascontiguousarray(rgb)
        captures.append(SimpleNamespace(
            role=role, source_id=depth.source_id, acquired_at=depth.captured_at,
            station_revision=depth.station_revision, source_generation=None,
            frame_index=None,
            source_binding_hash=source_binding_hash,
            width=depth.depth_m.shape[1], height=depth.depth_m.shape[0],
            depth_m=depth.depth_m, depth_content_hash=depth.depth_content_hash,
            intrinsics=depth.intrinsics, source_from_camera=(
                geometry_from_depth_reference @ np.asarray(depth.reference_from_camera).reshape(4, 4)),
            capture_qpos_by_side=pair, pose_measured_at=cut.captured_at,
            pose_evidence_refs=(), rgb=rgb,
            rgb_content_hash=None if rgb is None else hashlib.sha256(rgb.tobytes(order="C")).hexdigest()))
    return captures

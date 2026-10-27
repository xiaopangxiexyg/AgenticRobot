"""Borrow the one native rig to bind a fixed support's complete box geometry.

The visual/depth owner remains separate. This adapter supplies catalog-like
geometry and a fixed frame pose; it never renders, steps physics, or dispatches
motion. Moving support bodies require a correlated tracked-pose adapter instead.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from agent_gateway.profiles.back_projection import CorrelatedPlacementCapture
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import CalibratedBoxSupportObservation
from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1

from .back_projection import MujocoBackProjectionCapturePort
from .motion_application import MujocoAgentSkillMotionApplication


class MujocoFixedBoxSupportPort:
    """Read an allowlisted fixed collision box through the existing rig lock."""

    def __init__(
        self,
        *,
        application,
        depth_capture_port: MujocoBackProjectionCapturePort,
        allowed_support_body_names: tuple[str, ...],
        reference_frame_id: str,
        depth_role: str,
        current_station_revision: Callable[[], int],
    ) -> None:
        if not isinstance(application, MujocoAgentSkillMotionApplication):
            raise TypeError("Fixed support must borrow the deployed MuJoCo application")
        names = tuple(allowed_support_body_names)
        if (not names or len(set(names)) != len(names)
                or any(not isinstance(name, str) or not name.strip() for name in names)):
            raise ValueError("Fixed support names must be a unique nonempty allowlist")
        for name, value in (("reference_frame_id", reference_frame_id),
                            ("depth_role", depth_role)):
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(f"Fixed support needs an explicit {name}")
        if not callable(current_station_revision):
            raise TypeError("Fixed support needs the Station revision owner")
        if (not isinstance(depth_capture_port, MujocoBackProjectionCapturePort)
                or depth_capture_port.application is not application
                or depth_capture_port.reference_frame_id != reference_frame_id
                or depth_capture_port.source_frame_id != MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1
                or depth_role not in depth_capture_port.source_id_by_role):
            raise ValueError("Fixed support must borrow this application's bound depth owner")
        graph = application.target_frame_profile.frame_graph
        graph.resolve(MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, reference_frame_id)
        self.application = application
        self.depth_capture_port = depth_capture_port
        self.allowed_support_body_names = names
        self.reference_frame_id = reference_frame_id
        self.depth_role = depth_role
        self.current_station_revision = current_station_revision

    def capture_fixed_support(
        self,
        *,
        support_id: str,
        expected_station_revision: int,
    ) -> tuple[CorrelatedPlacementCapture, CalibratedBoxSupportObservation]:
        if support_id not in self.allowed_support_body_names:
            raise ValueError("Support identity is outside the deployed allowlist")
        revision = self.current_station_revision()
        if isinstance(expected_station_revision, bool) or expected_station_revision != revision:
            raise ValueError("Fixed support request uses another Station revision")
        paired_rgbd = self.depth_capture_port.capture_placement_rgbd(
            role=self.depth_role, expected_station_revision=revision)
        paired_depth = paired_rgbd.depth

        def read(scene):
            model = scene.model
            body = int(model.body(support_id).id)
            ancestor = body
            while ancestor != 0:
                if (int(model.body_jntnum[ancestor]) != 0
                        or int(model.body_mocapid[ancestor]) >= 0):
                    raise ValueError("Moving support requires correlated tracked pose")
                ancestor = int(model.body_parentid[ancestor])
            geoms = np.flatnonzero((model.geom_bodyid == body)
                & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
            if len(geoms) != 1:
                raise ValueError("Fixed support requires one complete collision box")
            geom = int(geoms[0])
            if model.geom_type[geom] != scene.mujoco.mjtGeom.mjGEOM_BOX:
                raise ValueError("Fixed support collision geometry is not a box")
            world_from_box = np.eye(4)
            world_from_box[:3, :3] = scene.data.geom_xmat[geom].reshape(3, 3)
            world_from_box[:3, 3] = scene.data.geom_xpos[geom]
            reference_from_world = self.application.target_frame_profile.frame_graph.resolve(
                MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, self.reference_frame_id
            ).matrix_target_source
            box_size = tuple(float(2. * value) for value in model.geom_size[geom])
            reference_from_box = reference_from_world @ world_from_box
            return reference_from_box, box_size

        cut = self.application.capture_passive_application_cut(scene_projector=read)
        if cut.lifecycle_state not in {"running", "held"} or cut.scene_projection is None:
            raise RuntimeError("Fixed support read needs a running or held application")
        reference_from_box, box_size = cut.scene_projection
        if self.current_station_revision() != revision:
            raise ValueError("Station revision changed during fixed support capture")
        geometry_ref = "mujoco-fixed-support-box:" + canonical_content_hash(
            {"support_id": support_id, "box_size_m": box_size,
             "reference_frame_id": self.reference_frame_id},
            domain="dexteleop.mujoco.fixed_support_box.v1",
        )
        pose_ref = "mujoco-fixed-support-pose:" + canonical_content_hash(
            {"support_id": support_id,
             "reference_from_box": reference_from_box.tolist(),
             "capture_group_id": paired_depth.capture_group_id,
             "station_revision": revision},
            domain="dexteleop.mujoco.fixed_support_pose.v1",
        )
        box = CalibratedBoxSupportObservation(
            support_id=support_id, reference_frame_id=self.reference_frame_id,
            scene_revision=revision, measured_at=paired_depth.captured_at,
            reference_from_box=reference_from_box, box_size_m=box_size,
            geometry_evidence_ref=geometry_ref, pose_evidence_ref=pose_ref,
            geometry_complete=True,
        )
        return paired_rgbd, box

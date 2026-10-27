"""Read a verified native held box through the existing object and application owners.

No close command, lift, reanchor, physics step, or motion authority is created.
The reference-frame and wall-clock observation are bracketed by application
cuts; a moving or closing application invalidates the passive read.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import GraspedObjectObservation
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy import capture_mujoco_sim_hil_tool_tcp_pose_by_side
from sim.mujoco.deploy.grasped_object import MujocoGraspedObjectObserver
from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1

from .motion_application import MujocoAgentSkillMotionApplication


def _complete_box_geom(model, mujoco, target_body: int) -> int:
    """Allow an exact visual duplicate; reject all unbounded target additions."""
    target_geoms = []
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        ancestor = body
        while ancestor and ancestor != target_body:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == target_body:
            target_geoms.append(geom)
    collision = [geom for geom in target_geoms
                 if model.geom_contype[geom] or model.geom_conaffinity[geom]]
    if len(collision) != 1:
        raise ValueError("Verified held box needs one complete target geometry")
    geom = collision[0]
    if (model.geom_bodyid[geom] != target_body
            or model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX):
        raise ValueError("Verified held geometry is not a known box")
    for other in target_geoms:
        if other == geom:
            continue
        if (model.geom_bodyid[other] != target_body
                or model.geom_contype[other] or model.geom_conaffinity[other]
                or model.geom_type[other] != mujoco.mjtGeom.mjGEOM_BOX
                or not np.allclose(model.geom_size[other], model.geom_size[geom], atol=1e-9, rtol=0)
                or not np.allclose(model.geom_pos[other], model.geom_pos[geom], atol=1e-9, rtol=0)
                or not np.allclose(model.geom_quat[other], model.geom_quat[geom], atol=1e-9, rtol=0)):
            raise ValueError("Verified held box has an unmatched additional shape")
    return geom


class MujocoHeldBoxObservationPort:
    """Borrow one native follow observer and bound application for a read-only cut."""

    def __init__(
        self,
        *,
        application: MujocoAgentSkillMotionApplication,
        observer: MujocoGraspedObjectObserver,
        object_id: str,
        side: str,
        reference_frame_id: str,
        current_station_revision: Callable[[], int],
    ) -> None:
        if (not isinstance(application, MujocoAgentSkillMotionApplication)
                or not isinstance(observer, MujocoGraspedObjectObserver)
                or observer.rig is not application.runtime.rig):
            raise TypeError("Held box needs the deployed application and its one observer")
        expected_gripper = (application.task_config.right_gripper_name
                            if side == "right" else application.task_config.left_gripper_name)
        if (observer.object_body_name != object_id
                or side not in {"left", "right"}
                or side != application.motion_side
                or observer.gripper_name != expected_gripper):
            raise ValueError("Held box identity or side differs from its observer")
        if not isinstance(reference_frame_id, str) or not reference_frame_id:
            raise ValueError("Held box needs an explicit reference frame")
        if not callable(current_station_revision):
            raise TypeError("Held box needs the Station revision owner")
        application.target_frame_profile.frame_graph.resolve(
            MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, reference_frame_id)
        self.application = application
        self.observer = observer
        self.object_id = object_id
        self.side = side
        self.reference_frame_id = reference_frame_id
        self.current_station_revision = current_station_revision

    def capture_held_box(self, *, expected_station_revision: int) -> GraspedObjectObservation:
        revision = self.current_station_revision()
        if (isinstance(expected_station_revision, bool)
                or revision != expected_station_revision):
            raise ValueError("Held-box request uses another Station revision")
        if self.application.closed:
            raise RuntimeError("Held-object application is closed")

        def read(scene):
            model = scene.model
            body = int(model.body(self.object_id).id)
            geom = _complete_box_geom(model, scene.mujoco, body)
            quat = model.geom_quat[geom]
            object_from_box = xyzquat_to_pose_matrix((*model.geom_pos[geom],
                float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])))
            world_from_tool = self.observer.world_tcp_from_scene(scene).copy()
            expected_world_from_tool = (scene.body_pose(self.side + "_arm_base_link")
                @ capture_mujoco_sim_hil_tool_tcp_pose_by_side(
                    scene, self.application.task_config)[self.side])
            if not np.allclose(world_from_tool, expected_world_from_tool,
                               atol=1e-6, rtol=0):
                raise ValueError("Held observer TCP differs from the deployed arm calibration")
            return dict(
                sim_time_s=float(scene.data.time),
                world_from_object=scene.body_pose(self.object_id).copy(),
                world_from_tool=world_from_tool,
                qpos_by_side={side: q.copy() for side, q in scene.arm_qpos_by_side().items()},
                box_size_m=tuple(float(2. * value) for value in model.geom_size[geom]),
                object_from_box=object_from_box,
            )

        before = self.application.capture_passive_application_cut(scene_projector=read)
        if before.lifecycle_state not in {"running", "held"} or before.scene_projection is None:
            raise RuntimeError("Held-object observation needs a running or held application")
        observed = self.observer.observe(phase="carry")
        after = self.application.capture_passive_application_cut(scene_projector=read)
        if after.lifecycle_state not in {"running", "held"} or after.scene_projection is None:
            raise RuntimeError("Application closed during held-object observation")
        first, last = before.scene_projection, after.scene_projection
        if (after.captured_at < before.captured_at
                or after.captured_at - before.captured_at > .05
                or first["sim_time_s"] != last["sim_time_s"]
                or observed.sim_time_s != last["sim_time_s"]
                or any(not np.array_equal(first["qpos_by_side"][side],
                                           last["qpos_by_side"][side])
                       for side in ("left", "right"))
                or not np.allclose(first["world_from_object"], last["world_from_object"],
                                   atol=1e-6, rtol=0)
                or not np.allclose(first["world_from_tool"], last["world_from_tool"],
                                   atol=1e-6, rtol=0)
                or not np.allclose(observed.world_from_object, last["world_from_object"],
                                   atol=1e-6, rtol=0)
                or not np.allclose(observed.world_from_tcp, last["world_from_tool"],
                                   atol=1e-6, rtol=0)):
            raise ValueError("Held-object scene changed across its correlated observation")
        if (observed.status != "held_verified" or observed.support_contact
                or observed.other_contact or observed.sensor_contacts != (True, True)):
            raise ValueError("Held-object observation lacks verified unsupported grasp")
        if self.current_station_revision() != revision:
            raise ValueError("Station revision changed during held-object capture")
        reference_from_world = self.application.target_frame_profile.frame_graph.resolve(
            MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, self.reference_frame_id
        ).matrix_target_source
        object_pose = reference_from_world @ last["world_from_object"]
        tool_pose = reference_from_world @ last["world_from_tool"]
        pose_ref = "mujoco-held-box-pose:" + canonical_content_hash(
            {"object_id": self.object_id, "side": self.side,
             "reference_frame_id": self.reference_frame_id,
             "reference_from_object": object_pose.tolist(),
             "reference_from_tool": tool_pose.tolist(),
             "measured_at": after.captured_at,
             "station_revision": revision},
            domain="dexteleop.mujoco.held_box_pose.v1",
        )
        grasp_ref = "mujoco-held-follow:" + canonical_content_hash(
            {"object_id": self.object_id,
             "command_generation": observed.command_generation,
             "sim_time_s": observed.sim_time_s,
             "frozen_tool_from_object": observed.tool_from_object.tolist(),
             "sensor_contacts": observed.sensor_contacts,
             "relative_translation_error_m": observed.relative_translation_error_m,
             "relative_rotation_error_deg": observed.relative_rotation_error_deg},
            domain="dexteleop.mujoco.held_box_follow.v1",
        )
        return GraspedObjectObservation(
            object_id=self.object_id, side=self.side,
            reference_frame_id=self.reference_frame_id,
            scene_revision=revision, measured_at=after.captured_at,
            reference_from_object=object_pose, reference_from_tool=tool_pose,
            qpos_by_side=last["qpos_by_side"], box_size_m=last["box_size_m"],
            object_from_box=last["object_from_box"], geometry_complete=True,
            pose_evidence_ref=pose_ref, grasp_evidence_ref=grasp_ref,
        )

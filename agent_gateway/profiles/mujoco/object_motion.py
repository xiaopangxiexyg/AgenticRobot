"""Opt-in object goal composition on the original safe motion lifecycle."""

import copy
import itertools
import logging
import math
from dataclasses import replace

import numpy as np

from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
)
from agent_gateway.skills.motion.grasped_object import OBJECT_MOVE_LINEAGE_FIELDS
from robot_motion.object_transport import rigid_pose, tool_goal_from_object_goal
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.payload_motion import MujocoPayloadBinding, MujocoPayloadSupport
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

from .motion_skill import MujocoAgentSafeTcpPoseSkillLifecycle


class MujocoAgentObjectMotionLifecycle(MujocoAgentSafeTcpPoseSkillLifecycle):
    """Resolve once from a correlated cut; no second planner or physics owner."""

    def __init__(self, *, object_observer, **kwargs):
        self._object_observer = object_observer
        self._payload = None
        self._object_goal_world = None
        self._object_terminal = None
        self._phase = kwargs["request"].parameters["phase"]
        self._object_id = kwargs["request"].parameters["object_id"]
        if self._object_id != object_observer.object_body_name:
            raise ValueError("Object motion differs from the deployed observed object.")
        lineage = {name: kwargs["request"].parameters[name] for name in OBJECT_MOVE_LINEAGE_FIELDS}
        super().__init__(target_resolver=self._resolve_object_goal, parameter_lineage=lineage, **kwargs)

    def _public_from_world(self):
        from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1

        return self.application.target_frame_profile.frame_graph.resolve(
            MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, self.motion_profile.reference_frame_id
        ).matrix_target_source

    def _resolve_object_goal(self, _measured):
        observer, parameters = self._object_observer, self.request.parameters
        cut = observer.observe(phase=self._phase, establish=True)
        public_from_world = self._public_from_world()
        current = public_from_world @ cut.world_from_object
        if "translation_delta_xyz_m" in parameters:
            goal = current.copy()
            goal[:3, 3] += np.asarray(parameters["translation_delta_xyz_m"], dtype=float)
        else:
            goal = xyzquat_to_pose_matrix(parameters["object_pose_xyz_quat_xyzw"])
        bounds = self.motion_profile.workspace_bounds
        if any(not bounds[2 * i] <= goal[i, 3] <= bounds[2 * i + 1] for i in range(3)):
            raise MujocoTcpPlanRejected("object_motion_goal_outside_workspace")
        world_goal = rigid_pose(np.linalg.inv(public_from_world) @ goal, name="world object goal")
        if self._phase in {"lift", "place"}:
            delta = world_goal[:3, 3] - cut.world_from_object[:3, 3]
            _, rotation = pose_delta_metrics(world_goal, cut.world_from_object)
            signed = delta[2] if self._phase == "lift" else -delta[2]
            if (not 0 <= signed <= .08 or np.linalg.norm(delta[:2]) > .003
                    or rotation > 2.):
                raise MujocoTcpPlanRejected("object_motion_support_stage_outside_normal_corridor")
        self._object_goal_world = world_goal

        def bind(scene):
            support = None
            if self._phase in {"lift", "place"}:
                model = scene.model
                table = int(model.body(self.application.task_config.table_body_name).id)
                geoms = [int(g) for g in np.flatnonzero((model.geom_bodyid == table)
                         & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))]
                if not geoms:
                    raise ValueError("Object support stage requires native support geometry.")
                points = []
                for geom in geoms:
                    if model.geom_type[geom] != scene.mujoco.mjtGeom.mjGEOM_BOX:
                        raise ValueError("Current object support binding requires finite native boxes.")
                    vertices = np.asarray(tuple(itertools.product((-1., 1.), repeat=3))) * model.geom_size[geom]
                    points.extend(vertices @ scene.data.geom_xmat[geom].reshape(3, 3).T
                                  + scene.data.geom_xpos[geom])
                points = np.asarray(points)
                lower, upper = points.min(axis=0), points.max(axis=0)
                lower[2] = upper[2]
                support = MujocoPayloadSupport(
                    tuple(model.geom(g).name for g in geoms), (0., 0., 1.), tuple(lower), tuple(upper),
                    self._phase, scene.collision_projector().policy.hard_penetration_m)

            def view(data):
                result = copy.copy(scene)
                result.data = data
                return result

            def measured_guard(data):
                try:
                    observer.observe(phase=self._phase, source_data=data)
                except ValueError as error:
                    return JointConfigurationCheck(safe=False, reason="object_follow_evidence_invalid",
                                                   metadata={"message": str(error)})
                return JointConfigurationCheck(safe=True)

            return MujocoPayloadBinding.capture(
                scene, scene.data, target_body_name=self._object_id,
                sensor_geom_names=observer.sensor_geom_names,
                tcp_pose_from_data=lambda data: observer.world_tcp_from_scene(view(data)),
                gripper_closed_from_data=lambda data: (
                    observer.command_state() == (cut.command_generation, 1.)
                    and observer.measured_closed(
                        view(data).get_gripper_close(observer.gripper_name))),
                tool_from_object=cut.tool_from_object, support=support,
                measured_guard=measured_guard, guard_identity=self.motion_profile.semantic_content_hash)

        self._payload = self.application.runtime.rig.capture_scene(bind)
        tool_world_goal = tool_goal_from_object_goal(world_goal, cut.tool_from_object)
        tool_public_goal = public_from_world @ tool_world_goal
        target = self.application.transform_tcp_pose_xyzquat(
            tuple(pose_to_xyz_quat_xyzw(tool_public_goal)),
            source_frame_id=self.motion_profile.reference_frame_id,
            target_frame_id=self.motion_profile.execution_reference_frame_id)
        return MotionPrimitiveTargetResolution(
            target_pose=target,
            evidence_refs=("mujoco-measured-payload-binding:" + self._payload.semantic_content_hash,))

    def activate(self, **kwargs):
        try:
            evidence = super().activate(**kwargs)
            if evidence.succeeded:
                # Commit the stage only after the owned plan and guard were
                # activated. Scratch preflights do not change tick policy.
                self.application.runtime.rig.capture_scene(lambda scene:
                    self._object_observer.observe(phase=self._phase, source_data=scene.data))
            return evidence
        except (ValueError, MujocoTcpPlanRejected) as error:
            logging.getLogger(__name__).warning("Object motion activation rejected: %s", error)
            reason = (error.reason_code if isinstance(error, MujocoTcpPlanRejected)
                      else "object_motion_measured_relation_unavailable")
            hold = self._hold(reason=reason)
            return StationLifecycleActivationEvidence(
                succeeded=False, status=reason, lower_operation_id=self._lower_operation_id,
                evidence_refs=hold.evidence_refs, lower_hold_confirmed=hold.lower_hold_confirmed)

    def _planning_payload_binding(self):
        return self._payload

    def _capture_pose(self):
        pose = super()._capture_pose()
        if self._payload is not None:
            check = self.application.runtime.rig.capture_scene(self._payload.check_measured)
            if not check.safe:
                raise MujocoTcpPlanRejected("object_motion_measured_state_blocked", check.to_dict())
        return pose

    def _within_terminal_tolerance(self, pose):
        if not super()._within_terminal_tolerance(pose):
            return False
        cut = self._object_observer.observe(phase=self._phase)
        translation, rotation = pose_delta_metrics(self._object_goal_world, cut.world_from_object)
        return (translation <= self.motion_profile.translation_tolerance_m
                and math.radians(rotation) <= self.motion_profile.rotation_tolerance_rad
                and (cut.status == "held_verified" if self._phase == "lift" else True)
                and (cut.support_contact if self._phase == "place" else not cut.support_contact))

    def _terminal_state_evidence(self, public_final_pose, measured_at):
        cut = self._object_observer.observe(phase=self._phase)
        self._object_terminal = {
            "object_id": self._object_id, "phase": self._phase,
            "reference_frame_id": self.motion_profile.reference_frame_id,
            "pose_xyz_quat_xyzw": list(pose_to_xyz_quat_xyzw(self._public_from_world() @ cut.world_from_object)),
            "measured_at": cut.sim_time_s,
            "clock_domain_id": "mujoco-native-physics:" + self.request.runtime_session_id,
            "holding_verified": cut.status == "held_verified",
            "support_contact": cut.support_contact, "source": "mujoco_native_physics",
        }
        return {**super()._terminal_state_evidence(public_final_pose, measured_at),
                "object_state": self._object_terminal}

    def _success(self, **kwargs):
        completion = super()._success(**kwargs)
        return replace(completion, reason_code="grasped_object_target_reached_and_held",
                       execution_result={**completion.execution_result, "object_state": self._object_terminal})

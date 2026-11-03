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
from agent_gateway.skills.motion.grasped_object import (
    OBJECT_MOVE_LINEAGE_FIELDS, OBJECT_INCLINATION_DEVIATION_RAD_V3,
)
from robot_motion.object_attitude import ObjectInclinationBand, up_facing_box_axis
from robot_motion.object_transport import rigid_pose, tool_goal_from_object_goal
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.actuation.executor import MujocoPhysicalContactFault
from sim.mujoco.deploy.payload_motion import MujocoPayloadBinding, MujocoPayloadSupport
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

from .motion_skill import MujocoAgentSafeTcpPoseSkillLifecycle
from .held_box import _complete_box_geom

# Once the TCP has completed its checked path, the measured TCP and object
# must reach their goal state within this much simulation time. Lift
# verification itself needs 0.5 s of unsupported following.
OBJECT_TERMINAL_GRACE_S = 1.5
# Before holding is verified, a lift goal must exceed the observer's remaining
# verification rise by the TCP arrival tolerance plus this margin.
LIFT_VERIFICATION_MARGIN_M = .002


class _ObjectTerminalUnmet(Exception):
    """The TCP arrived but the measured object state did not; fail with HOLD."""

    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


def _native_support_for_object_motion(scene, *, body_name, phase,
                                      max_penetration_m, dynamic):
    """Bind the current top face; a free support need not be exactly level."""
    model = scene.model
    body = int(model.body(body_name).id)
    geoms = [int(g) for g in np.flatnonzero((model.geom_bodyid == body)
             & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))]
    if not geoms:
        raise ValueError("Object support stage requires native support geometry.")
    data = scene.data
    if dynamic:
        data = scene.mujoco.MjData(model)
        scene.mujoco.mj_copyData(data, model, scene.data)
        scene.mujoco.mj_kinematics(model, data)
    points, normals, top_heights = [], [], []
    for geom in geoms:
        if model.geom_type[geom] != scene.mujoco.mjtGeom.mjGEOM_BOX:
            raise ValueError("Current object support binding requires finite native boxes.")
        rotation = data.geom_xmat[geom].reshape(3, 3)
        center = data.geom_xpos[geom]
        size = model.geom_size[geom]
        if dynamic:
            # A free box may rest on any native face. Select its upward face
            # from the measured orientation, without inferring it from a
            # privileged object pose or changing the native collision model.
            axis = int(np.argmax(np.abs(rotation[2, :])))
            sign = float(np.sign(rotation[2, axis]))
            normal = rotation[:, axis] * sign
            if normal[2] < math.cos(math.radians(1.)):
                raise ValueError("Dynamic support top face is not near-horizontal and upward.")
            normals.append(normal)
            tangents = [i for i in range(3) if i != axis]
            face_local = np.empty((4, 3), dtype=float)
            face_local[:, axis] = sign * size[axis]
            face_local[:, tangents] = (np.asarray(tuple(itertools.product(
                (-1., 1.), repeat=2)), dtype=float) * size[tangents])
            face = face_local @ rotation.T + center
            points.extend(face)
            top_heights.append(float(np.dot(normal, center + normal * size[axis])))
        else:
            vertices = np.asarray(tuple(itertools.product((-1., 1.), repeat=3))) * size
            points.extend(vertices @ rotation.T + center)
    if dynamic and (any(not np.allclose(value, normals[0], atol=1e-7, rtol=0.)
                        for value in normals[1:])
                    or max(top_heights) - min(top_heights) > max_penetration_m):
        raise ValueError("Dynamic support boxes must have one measured top face.")
    region = np.asarray(points)
    lower, upper = region.min(axis=0), region.max(axis=0)
    if not dynamic:
        lower[2] = upper[2]
    frozen = {}
    if dynamic:
        pose = np.eye(4)
        pose[:3, :3] = data.xmat[body].reshape(3, 3)
        pose[:3, 3] = data.xpos[body]
        frozen = {"dynamic_body_name": body_name,
                  "frozen_pose_xyz_quat_xyzw": tuple(pose_to_xyz_quat_xyzw(pose))}
    return MujocoPayloadSupport(
        tuple(model.geom(g).name for g in geoms),
        tuple(normals[0]) if dynamic else (0., 0., 1.),
        tuple(lower), tuple(upper), phase, max_penetration_m, **frozen)


class MujocoAgentObjectMotionLifecycle(MujocoAgentSafeTcpPoseSkillLifecycle):
    """Resolve once from a correlated cut; no second planner or physics owner."""

    def __init__(self, *, object_observer, **kwargs):
        self._object_observer = object_observer
        self._payload = None
        self._object_goal_world = None
        self._object_terminal = None
        self._object_unmet_since = None
        self._public_phase = kwargs["request"].parameters["phase"]
        self._attitude_v3 = kwargs["route"].skill_version == "3"
        self._release_approach = self._attitude_v3 and self._public_phase == "release_approach"
        self._phase = "carry" if self._release_approach else self._public_phase
        self._inclination_band = None
        self._grasp_generation = None
        self._object_id = kwargs["request"].parameters["object_id"]
        self._support_object_id = kwargs["request"].parameters.get("support_object_id")
        if self._object_id != object_observer.object_body_name:
            raise ValueError("Object motion differs from the deployed observed object.")
        if (self._attitude_v3 and (
                self._public_phase not in {"lift", "carry", "release_approach"}
                or (self._release_approach and self._support_object_id not in object_observer.support_body_names[1:])
                or (not self._release_approach and self._support_object_id is not None))) or (
                kwargs["route"].skill_version == "2" and (
                self._phase != "place" or self._support_object_id not in object_observer.support_body_names[1:]
        )) or (kwargs["route"].skill_version == "1" and self._support_object_id is not None):
            raise ValueError("Object support identity differs from the admitted Route.")
        lineage = {name: kwargs["request"].parameters[name] for name in OBJECT_MOVE_LINEAGE_FIELDS}
        super().__init__(target_resolver=self._resolve_object_goal, parameter_lineage=lineage, **kwargs)

    def _public_from_world(self):
        from sim.mujoco.deploy.target_measurement import MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1

        return self.application.target_frame_profile.frame_graph.resolve(
            MUJOCO_SIM_HIL_WORLD_FRAME_ID_V1, self.motion_profile.reference_frame_id
        ).matrix_target_source

    def _resolve_object_goal(self, _measured):
        observer, parameters = self._object_observer, self.request.parameters
        support_selection = ({} if self._support_object_id is None or self._release_approach else
                             {"place_support_body_name": self._support_object_id})
        cut = observer.observe(phase=self._phase, establish=True, **support_selection)
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
        if self._phase == "lift" and cut.status != "held_verified":
            # Holding verifies only after the observer's measured rise. A
            # shorter lift could arrive and then never verify; refuse it now.
            required = (observer.holding_verification_rise_remaining_m(cut)
                        + self.motion_profile.translation_tolerance_m + LIFT_VERIFICATION_MARGIN_M)
            requested = float(world_goal[2, 3] - cut.world_from_object[2, 3])
            if requested < required:
                raise MujocoTcpPlanRejected(
                    "object_motion_lift_below_holding_verification_height",
                    {"requested_rise_m": requested, "required_rise_m": required})
        # Lift and place carry no fixed vertical corridor: the measured payload
        # binding, its support allowance and the follow guard bound every path.
        self._object_goal_world = world_goal

        def bind(scene):
            support = None
            if self._phase in {"lift", "place"}:
                support_name = (self._support_object_id or self.application.task_config.table_body_name)
                support = _native_support_for_object_motion(
                    scene, body_name=support_name, phase=self._phase,
                    max_penetration_m=scene.collision_projector().policy.hard_penetration_m,
                    dynamic=self._support_object_id is not None)

            if self._attitude_v3:
                anchor = observer.grasp_attitude_anchor(command_generation=cut.command_generation)
                body = int(scene.model.body(self._object_id).id)
                geom = _complete_box_geom(scene.model, scene.mujoco, body)
                quat = scene.model.geom_quat[geom]
                object_from_box = xyzquat_to_pose_matrix((*scene.model.geom_pos[geom],
                    float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])))
                axis = up_facing_box_axis(anchor, object_from_box, (0., 0., 1.))
                if self._release_approach:
                    support_body = int(scene.model.body(self._support_object_id).id)
                    support_geom = _complete_box_geom(scene.model, scene.mujoco, support_body)
                    rotation = scene.data.geom_xmat[support_geom].reshape(3, 3)
                    index = int(np.argmax(np.abs(rotation[2])))
                    normal = rotation[:, index] * (1. if rotation[2, index] >= 0. else -1.)
                    if float((world_goal[:3, :3] @ axis) @ normal) < math.cos(
                            OBJECT_INCLINATION_DEVIATION_RAD_V3):
                        raise MujocoTcpPlanRejected("object_release_goal_not_aligned_with_named_support")
                self._inclination_band = ObjectInclinationBand.from_poses(
                    anchor,
                    axis_object_xyz=axis, up_world_xyz=(0., 0., 1.),
                    max_deviation_rad=OBJECT_INCLINATION_DEVIATION_RAD_V3,
                    goal_world_from_object=world_goal if self._release_approach else None)
                if not self._inclination_band.accepts(world_goal):
                    raise MujocoTcpPlanRejected("object_goal_changes_grasped_inclination")
                self._grasp_generation = cut.command_generation

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
                measured_guard=measured_guard, guard_identity=self.motion_profile.semantic_content_hash,
                **({"inclination_band": self._inclination_band} if self._attitude_v3 else {}))

        self._payload = self.application.runtime.rig.capture_scene(bind)
        # Convert the goal through the hand-object transform measured in this
        # cut. The frozen relation still bounds slip (binding and observer), but
        # converting through it would turn accumulated slip into an unrequested
        # TCP rotation and leave the object off its goal by that slip.
        measured_relation = rigid_pose(np.linalg.inv(cut.world_from_tcp) @ cut.world_from_object,
                                       name="measured tool_from_object")
        tool_world_goal = tool_goal_from_object_goal(world_goal, measured_relation)
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
                def commit(scene):
                    if self._inclination_band is not None:
                        self._object_observer.bind_grasp_inclination_band(
                            command_generation=self._grasp_generation, band=self._inclination_band)
                    return self._object_observer.observe(
                        phase=self._phase, source_data=scene.data,
                        **({} if self._support_object_id is None or self._release_approach else
                           {"place_support_body_name": self._support_object_id}))
                self.application.runtime.rig.capture_scene(commit)
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
            if not self._path_completed:
                self._object_unmet_since = None
                return False
            # The TCP finished its checked path but does not settle inside the
            # goal tolerance (e.g. a stretched arm sagging under the payload).
            # Fail with HOLD after the same bounded settle; waiting for the
            # operation timeout left the station busy and unobservable.
            return self._terminal_unmet(self._object_observer.observe(phase=self._phase),
                                        "object_motion_tcp_pose_outside_tolerance")
        cut = self._object_observer.observe(phase=self._phase)
        translation, rotation = pose_delta_metrics(self._object_goal_world, cut.world_from_object)
        if (translation > self.motion_profile.translation_tolerance_m
                or math.radians(rotation) > self.motion_profile.rotation_tolerance_rad):
            reason = "object_motion_object_pose_outside_tolerance"
        elif self._phase == "lift" and cut.status != "held_verified":
            reason = "object_motion_lift_did_not_verify_holding"
        elif self._phase == "place" and not cut.support_contact:
            reason = "object_motion_support_contact_missing"
        elif self._phase != "place" and cut.support_contact:
            reason = "object_motion_support_contact_unexpected"
        else:
            self._object_unmet_since = None
            return True
        # The TCP finished its checked path; give the measured object a short,
        # bounded settle instead of waiting for the operation timeout.
        return self._terminal_unmet(cut, reason)

    def _terminal_unmet(self, cut, reason):
        """One bounded settle after the checked path, shared by TCP and object goals."""
        if self._object_unmet_since is None:
            self._object_unmet_since = cut.sim_time_s
        elif cut.sim_time_s - self._object_unmet_since >= OBJECT_TERMINAL_GRACE_S:
            raise _ObjectTerminalUnmet(reason)
        return False

    def _terminal_state_evidence(self, public_final_pose, measured_at):
        cut = self._object_observer.observe(phase=self._phase)
        self._object_terminal = {
            "object_id": self._object_id, "phase": self._public_phase,
            "reference_frame_id": self.motion_profile.reference_frame_id,
            "pose_xyz_quat_xyzw": list(pose_to_xyz_quat_xyzw(self._public_from_world() @ cut.world_from_object)),
            "measured_at": cut.sim_time_s,
            "clock_domain_id": "mujoco-native-physics:" + self.request.runtime_session_id,
            "holding_verified": cut.status == "held_verified",
            "support_contact": cut.support_contact, "source": "mujoco_native_physics",
        }
        if self._support_object_id is not None:
            self._object_terminal["support_object_id"] = self._support_object_id
        return {**super()._terminal_state_evidence(public_final_pose, measured_at),
                "object_state": self._object_terminal}

    def _success(self, **kwargs):
        completion = super()._success(**kwargs)
        return replace(completion, reason_code="grasped_object_target_reached_and_held",
                       execution_result={**completion.execution_result, "object_state": self._object_terminal})

    def run(self, **kwargs):
        try:
            return super().run(**kwargs)
        except _ObjectTerminalUnmet as error:
            hold = self._hold(reason=error.reason_code)
            return self._failure(reason_code=error.reason_code, hold=hold)
        except MujocoPhysicalContactFault as error:
            # The lower executor has already latched the same hard fault.
            # Classify only its typed payload pair; all other collisions keep
            # the original worker exception and HOLD path.
            reason = {
                "payload_relation": "object_follow_relation_lost_after_hold",
                "payload_support_moved": "object_support_moved_after_hold",
            }.get(error.check.pair_class)
            if reason is None:
                raise
            hold = self._hold(reason=reason)
            return self._failure(reason_code=reason, hold=hold)

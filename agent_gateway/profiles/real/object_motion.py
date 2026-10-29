"""Real held-object motion on the existing observed safe-move lifecycle.

``motion.move_grasped_object_to_pose@1`` on the RealMan station. The bound
object is "whatever this arm holds": every request first measures it from
fresh stationary RGB-D views (point prompts inside the gripper's closing
region, ``RealHeldObjectObserver``) and the DMTAC closure, and refuses unless
it is between the fingers with the gripper closed on it. The measured box,
fixed in the tool frame, converts the object-body goal into the TCP goal and
is carried by the observed safe-move planner along the whole path. ``place``
also measures the support plane under the target footprint and lets only that
support top approach the box. After arrival the object is measured again;
the operation reports that measured state, or stops with a reason when the
object is no longer held. It never opens or closes a gripper.

Real measurement differs from the simulator's native object state: it is a
camera cut after the arm stopped, not a continuous contact signal.
``support_contact`` for ``place`` means the measured box bottom is within the
place gap of the measured support plane, not a sensed contact force.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable

import numpy as np

from agent_gateway.profiles.motion_primitive_resolution import (
    MotionPrimitiveTargetResolution,
)
from agent_gateway.runtime.station_operation_worker import (
    StationLifecycleActivationEvidence,
    StationLifecycleExecutionCompletion,
)
from agent_gateway.skills.motion.grasped_object import GRASPED_OBJECT_MOVE_SKILL_ID
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import tool_goal_from_object_goal
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from teleop_runtime.skill_deploy.rm65_grasp_trajectory import RM65CarriedObjectSupport

from .held_object import RealHeldObjectObservation, RealHeldObjectObserver
from .motion_primitive_profiles import (
    REAL_AGENT_OBJECT_MOTION_EXECUTOR_ID_V1,
    REAL_AGENT_OBJECT_MOTION_ROUTE_ID_V1,
    RealAgentObjectMotionProfile,
)
from .motion_primitive_skill import RealAgentPrimitiveSkillLifecycle

_LOGGER = logging.getLogger(__name__)

# The observed multi-view table surface is a band about 2.5 cm thick (voxels,
# multi-camera disagreement); support voxels up to 15 mm above the fitted
# plane under the target footprint are the support itself.
REAL_OBJECT_MOTION_SUPPORT_HEIGHT_TOLERANCE_M = 0.015
REAL_OBJECT_MOTION_SUPPORT_MARGIN_M = 0.04
REAL_OBJECT_MOTION_SUPPORT_MIN_POINTS = 150
REAL_OBJECT_MOTION_SUPPORT_MAX_RESIDUAL_M = 0.008
# A placed box bottom within this gap of the measured plane is on the support.
REAL_OBJECT_MOTION_PLACE_CONTACT_GAP_M = 0.02


def fit_support_plane_under(*, captures, footprint_min_xy, footprint_max_xy, below_z_m: float,
                            exclude_box=None, margin_m: float = REAL_OBJECT_MOTION_SUPPORT_MARGIN_M):
    """Plane of the measured surface around a target footprint, below ``below_z_m``.

    Uses every view's valid depth inside the footprint widened by ``margin_m``
    (the held object usually hides the footprint itself), drops points inside
    ``exclude_box`` (the held object's own box, ``(world_from_box, size)``) and
    fits z = a x + b y + c with one 2.5-sigma trim. Returns ``None`` when the
    surface is unmeasured or not planar.
    """
    low = np.asarray(footprint_min_xy, dtype=float) - margin_m
    high = np.asarray(footprint_max_xy, dtype=float) + margin_m
    selected = []
    for capture in captures:
        depth = np.asarray(capture.depth_m, dtype=float)
        rows, cols = np.mgrid[0:depth.shape[0]:2, 0:depth.shape[1]:2]
        z = depth[rows, cols]
        valid = np.isfinite(z) & (z > 0) & (z <= 1.5)
        rows, cols, z = rows[valid], cols[valid], z[valid]
        intrinsics = capture.intrinsics
        camera = np.column_stack(((cols - float(intrinsics["cx"])) * z / float(intrinsics["fx"]),
                                  (rows - float(intrinsics["cy"])) * z / float(intrinsics["fy"]), z))
        pose = np.asarray(capture.source_from_camera, dtype=float).reshape(4, 4)
        world = camera @ pose[:3, :3].T + pose[:3, 3]
        keep = np.all((world[:, :2] >= low) & (world[:, :2] <= high), axis=1) & (world[:, 2] < below_z_m)
        selected.append(world[keep])
    points = np.concatenate(selected) if selected else np.zeros((0, 3))
    if exclude_box is not None and len(points):
        world_from_box, size = exclude_box
        local = (points - world_from_box[:3, 3]) @ world_from_box[:3, :3]
        points = points[~np.all(np.abs(local) <= np.asarray(size) / 2.0 + 0.01, axis=1)]
    if len(points) < REAL_OBJECT_MOTION_SUPPORT_MIN_POINTS:
        return None, {"support_point_count": len(points)}
    design = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    coef, *_ = np.linalg.lstsq(design, points[:, 2], rcond=None)
    residual = points[:, 2] - design @ coef
    keep = np.abs(residual) <= 2.5 * max(float(np.std(residual)), 1e-4)
    coef, *_ = np.linalg.lstsq(design[keep], points[keep, 2], rcond=None)
    residual = points[keep, 2] - design[keep] @ coef
    normal = np.array([-coef[0], -coef[1], 1.0])
    normal /= np.linalg.norm(normal)
    centre = (np.asarray(footprint_min_xy, dtype=float) + np.asarray(footprint_max_xy, dtype=float)) / 2.0
    anchor = np.array([centre[0], centre[1], coef[0] * centre[0] + coef[1] * centre[1] + coef[2]])
    stats = {"support_point_count": int(np.count_nonzero(keep)),
             "plane_residual_p95_m": float(np.percentile(np.abs(residual), 95)),
             "plane_tilt_deg": float(math.degrees(math.acos(min(1.0, normal[2])))),
             "plane_anchor_xyz_m": anchor.tolist()}
    if (stats["plane_residual_p95_m"] > REAL_OBJECT_MOTION_SUPPORT_MAX_RESIDUAL_M
            or stats["plane_tilt_deg"] > 10.0):
        return None, stats
    return (anchor, normal, low, high), stats


class RealHeldObjectPort:
    """Lower-owned held-object measurement for the object-motion lifecycle."""

    def __init__(self, *, observer: RealHeldObjectObserver, public_from_geometry,
                 current_station_revision: Callable[[], int], clock: Callable[[], float],
                 min_score: float = 0.2):
        if not isinstance(observer, RealHeldObjectObserver) or not callable(current_station_revision):
            raise TypeError("Held-object port needs the observer and the Station revision.")
        self.observer = observer
        self.public_from_geometry = np.asarray(public_from_geometry, dtype=float).reshape(4, 4).copy()
        self.current_station_revision = current_station_revision
        self.clock = clock
        self.min_score = float(min_score)
        self.last_captures = ()

    def observe(self, *, side: str, cancel_requested: Callable[[], bool]) -> RealHeldObjectObservation:
        revision = int(self.current_station_revision())
        captures = tuple(self.observer.scene_port.capture_views(
            roles=self.observer.roles, expected_station_revision=revision, cancel_requested=cancel_requested))
        self.last_captures = captures
        return self.observer.observe(side=side, label="held object", language_prompt=None,
                                     min_score=self.min_score, expected_station_revision=revision,
                                     cancel_requested=cancel_requested, captures=captures)


class _ObjectMotionUnavailable(ValueError):
    def __init__(self, reason: str, message: str, diagnostic: dict | None = None):
        super().__init__(message)
        self.reason = reason
        self.diagnostic = dict(diagnostic or {})


def _box_world_corners(world_from_box, size):
    signs = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
    return (signs * np.asarray(size)) @ world_from_box[:3, :3].T + world_from_box[:3, 3]


class RealObjectMotionLifecycle(RealAgentPrimitiveSkillLifecycle):
    """Measured held object -> carried safe move -> measured terminal object state."""

    def __init__(self, *, held_object_port: RealHeldObjectPort, **kwargs):
        if (kwargs["route"].skill_id != GRASPED_OBJECT_MOVE_SKILL_ID or kwargs["route"].skill_version != "1"
                or not isinstance(kwargs["profile"], RealAgentObjectMotionProfile)
                or not isinstance(held_object_port, RealHeldObjectPort)):
            raise ValueError("Real object motion requires its exact Route, profile and held-object port.")
        self._held = held_object_port
        self._phase = kwargs["request"].parameters["phase"]
        self._carried = self._support = None
        self._start = self._terminal = self._terminal_failure = None
        self._public_goal = None
        self._report: dict = {}
        super().__init__(**kwargs)

    # Target resolution -------------------------------------------------
    def _resolve_target(self, measured_pose):
        try:
            return self._resolve_object_goal()
        except _ObjectMotionUnavailable:
            raise
        except (ValueError, RuntimeError, KeyError) as error:
            raise _ObjectMotionUnavailable("object_motion_measured_relation_unavailable", str(error)) from error

    def _observation_summary(self, observation: RealHeldObjectObservation) -> dict:
        measurement = observation.measurement
        return {"status": measurement.status, "reason": measurement.reason,
                "relation": measurement.relation.get("relation"),
                "holding_verified": observation.holding_verified,
                "gripper_closed_on_object": observation.gripper_closed_on_object,
                "gripper_measured_close": None if observation.gripper is None
                else observation.gripper.get("measured_close"),
                "views": [dict(item) for item in observation.segments],
                "unmeasured": [dict(item) for item in observation.unmeasured],
                "grip_metrics": {key: value for key, value in measurement.metrics.items()
                                 if key in ("centroid_offset_closing_axis_m", "centroid_offset_pad_axis_m",
                                            "principal_axis_tilt_deg", "measured_point_count")}}

    def _resolve_object_goal(self):
        observation = self._held.observe(side=self.side, cancel_requested=lambda: False)
        self._start = observation
        summary = self._observation_summary(observation)
        if not observation.holding_verified:
            reason = ("object_motion_object_not_between_fingers" if observation.measurement.status
                      == "not_between_fingers" else "object_motion_gripper_not_closed_on_object"
                      if observation.gripper_closed_on_object is False
                      else "object_motion_held_object_unmeasured")
            raise _ObjectMotionUnavailable(reason, "The held object is not verified between the closed fingers.",
                                           {"start_observation": summary})
        tool_from_object = np.asarray(observation.measurement.tool_from_box)
        public_from_tool = self._held.public_from_geometry @ observation.world_from_tcp
        current = public_from_tool @ tool_from_object
        parameters = self.request.parameters
        if "translation_delta_xyz_m" in parameters:
            goal = current.copy()
            goal[:3, 3] += np.asarray(parameters["translation_delta_xyz_m"], dtype=float)
        else:
            goal = xyzquat_to_pose_matrix(parameters["object_pose_xyz_quat_xyzw"], name="object_goal")
        bounds = self.profile.object_workspace_bounds
        if any(not bounds[2 * i] <= goal[i, 3] <= bounds[2 * i + 1] for i in range(3)):
            raise _ObjectMotionUnavailable("object_motion_goal_outside_workspace",
                                           "The object goal is outside the deployed object workspace.")
        tool_goal_public = tool_goal_from_object_goal(goal, tool_from_object)
        execution, resolution = self.contract_bundle.frame_graph.transform_pose_matrix(
            tool_goal_public, source_reference_frame_id=self.profile.reference_frame_id,
            target_reference_frame_id=self.profile.execution_reference_frame_id)
        self._carried = observation.carried_object_box(object_id=self.profile.object_id)
        size = np.asarray(observation.measurement.size_xyz_m)
        geometry_from_public = np.linalg.inv(self._held.public_from_geometry)
        target_box_world = geometry_from_public @ goal
        corners = _box_world_corners(target_box_world, size)
        self._report = {"phase": self._phase, "start_observation": summary,
                        "box_size_xyz_m": size.tolist(),
                        "box_bottom_below_object_pose_m": float(
                            goal[2, 3] - _box_world_corners(goal, size)[:, 2].min()),
                        "start_object_pose_xyz_quat_xyzw": [float(v) for v in pose_matrix_to_xyzquat(current)],
                        "requested_object_pose_xyz_quat_xyzw": [float(v) for v in pose_matrix_to_xyzquat(goal)]}
        if self._phase == "place":
            current_box_world = observation.world_from_tcp @ tool_from_object
            fitted, stats = fit_support_plane_under(
                captures=self._held.last_captures, footprint_min_xy=corners[:, :2].min(axis=0),
                footprint_max_xy=corners[:, :2].max(axis=0),
                below_z_m=float(corners[:, 2].min()) + 0.01, exclude_box=(current_box_world, size))
            self._report["support_measurement"] = stats
            if fitted is None:
                raise _ObjectMotionUnavailable("object_motion_support_not_measured",
                                               "No planar support surface was measured under the place footprint.",
                                               {"support_measurement": stats})
            anchor, normal, low, high = fitted
            self._support = RM65CarriedObjectSupport(
                support_id="measured_support_under_target", plane_anchor_xyz_m=tuple(anchor),
                normal_xyz=tuple(normal), region_min_xy_m=tuple(low), region_max_xy_m=tuple(high),
                evidence_ref=observation.evidence_ref,
                height_tolerance_m=REAL_OBJECT_MOTION_SUPPORT_HEIGHT_TOLERANCE_M)
            gap = float(np.min((corners - anchor) @ normal))
            self._report["target_box_bottom_above_support_m"] = gap
            if gap > REAL_OBJECT_MOTION_PLACE_CONTACT_GAP_M:
                raise _ObjectMotionUnavailable(
                    "object_motion_place_target_above_support",
                    f"The place goal leaves the box bottom {gap * 100:.1f} cm above the measured support; "
                    f"place requires at most {REAL_OBJECT_MOTION_PLACE_CONTACT_GAP_M * 100:.0f} cm.",
                    {"target_box_bottom_above_support_m": gap})
        self._public_goal = goal
        digest = canonical_content_hash({
            "profile_hash": self.profile.semantic_content_hash, "component_id": self.component_id,
            "phase": self._phase, "observation": observation.evidence_ref,
            "object_goal": goal, "tool_goal_public": tool_goal_public,
            "frame_resolution_hash": resolution.semantic_content_hash,
            "support": None if self._support is None else self._support.to_dict()},
            domain="agenticrobot.real.object_motion.target.v1")
        return (MotionPrimitiveTargetResolution(
            target_pose=tuple(float(v) for v in pose_matrix_to_xyzquat(execution, name="object_motion_tool_goal")),
            evidence_refs=(observation.evidence_ref, "real-object-motion-target:" + digest)),
            GRASPED_OBJECT_MOVE_SKILL_ID + "@1", digest)

    def _extra_planning_kwargs(self) -> dict:
        return {"carried_object": self._carried,
                **({} if self._support is None else {"carried_object_support": self._support})}

    def activate(self, *, request, operation, **kwargs):
        try:
            return super().activate(request=request, operation=operation, **kwargs)
        except _ObjectMotionUnavailable as error:
            _LOGGER.warning("Real object motion refused before activation: %s (%s) %r",
                            error.reason, error, error.diagnostic)
            hold = self.request_hold(operation=operation, reason=error.reason)
            return StationLifecycleActivationEvidence(
                succeeded=False, status=error.reason, lower_operation_id=self.lower_operation_id,
                evidence_refs=hold.evidence_refs, lower_hold_confirmed=hold.lower_hold_confirmed,
                failure_diagnostic={"reason_code": error.reason, "side": self.side, "message": str(error)})

    # Terminal state -----------------------------------------------------
    def _extend_arrival_result(self, result, measured_pose):
        try:
            observation = self._held.observe(side=self.side, cancel_requested=lambda: False)
        except (ValueError, RuntimeError) as error:
            self._terminal_failure = ("object_motion_terminal_state_unmeasured", str(error))
            return result
        self._terminal = observation
        summary = self._observation_summary(observation)
        self._report["terminal_observation"] = summary
        if not observation.holding_verified:
            self._terminal_failure = (
                "object_motion_holding_lost_after_move",
                ("The hand arrived but the object is no longer measured between the closed fingers "
                 "(dropped or slipped)."))
            return result
        size = np.asarray(observation.measurement.size_xyz_m)
        public = (self._held.public_from_geometry @ observation.world_from_tcp
                  @ np.asarray(observation.measurement.tool_from_box))
        support_contact = self._phase == "place"
        if self._support is not None:
            corners = _box_world_corners(observation.world_from_tcp @ np.asarray(
                observation.measurement.tool_from_box), size)
            gap = float(np.min((corners - np.asarray(self._support.plane_anchor_xyz_m))
                               @ np.asarray(self._support.normal_xyz)))
            self._report["terminal_box_bottom_above_support_m"] = gap
            if gap > REAL_OBJECT_MOTION_PLACE_CONTACT_GAP_M:
                self._terminal_failure = ("object_motion_place_not_on_support",
                                          f"The measured box bottom stays {gap * 100:.1f} cm above the support.")
                return result
        captured_at = max(float(capture.acquired_at) for capture in self._held.last_captures)
        state = {"object_id": self.profile.object_id, "phase": self._phase,
                 "reference_frame_id": self.profile.reference_frame_id,
                 "pose_xyz_quat_xyzw": [float(v) for v in pose_matrix_to_xyzquat(public)],
                 "measured_at": captured_at, "clock_domain_id": self.profile.clock_domain_id,
                 "holding_verified": True, "support_contact": support_contact,
                 "source": self.profile.observation_source}
        return {**result, "object_state": state, "held_object_report": self._report}

    def run(self, **kwargs):
        completion = super().run(**kwargs)
        if not completion.succeeded or self._terminal_failure is None:
            return completion
        reason, message = self._terminal_failure
        _LOGGER.warning("Real object motion arrived without its object state: %s (%s)", reason, message)
        return StationLifecycleExecutionCompletion(
            succeeded=False, reason_code=reason, execution_receipt_refs=completion.execution_receipt_refs,
            terminal_hold_confirmed=completion.terminal_hold_confirmed,
            failure_diagnostic={"reason_code": reason, "side": self.side, "message": message})


__all__ = ["REAL_AGENT_OBJECT_MOTION_EXECUTOR_ID_V1", "REAL_AGENT_OBJECT_MOTION_ROUTE_ID_V1",
           "RealAgentObjectMotionProfile", "RealHeldObjectPort", "RealObjectMotionLifecycle",
           "fit_support_plane_under"]

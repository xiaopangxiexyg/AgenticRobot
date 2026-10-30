"""MuJoCo registration and port for ``perception.query_grasp_approach@1``.

One read-only request, with the motion application HOLD and no active
operation:

1. require an open gripper and an object that is not held;
2. per bound camera role, text-segment the object with its deployment-bound
   prompt and measure its top face from depth only (center, height, yaw);
3. build face-aligned grasps on the measured top in a fixed preference order:
   the reviewed DMTAC top-down grasp, the same grasp turned a quarter turn
   either way, a horizontal grasp approaching from the motion arm's own side,
   a horizontal front grasp, the half-turned top-down grasp;
4. keep the first whose approach pose is reachable from the current arm state
   and whose pregrasp is reachable from that approach configuration, both
   with a private planner bound to the grasp approach Route's profile and
   envelope. Every refused candidate is reported with its reason.

Nothing is commanded, retained as a plan or authorized for execution.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.measured_support import measure_object_top, measure_surface_below
from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception import grasp_approach_query as skill
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.planned_tcp_motion import MujocoPlannedTcpMotion, MujocoTcpPlanRejected
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

GRASP_APPROACH_MAX_OPERATION_MS = 90_000
# Reviewed DMTAC_M top-down grasp of a 35 mm box: tool_tcp in the object's
# top-face frame (origin at the measured top center, z up, x along the
# measured yaw, axes otherwise aligned with the MuJoCo station frame: x
# forward from the robot, y to its left). The qualified scripted pregrasp
# that secures bilateral sensor contact on the place-suite cube was recorded
# in world axes (x toward the robot): this is that grasp turned into the
# station-aligned top frame.
_TOP_FROM_GRASP_WORLD_AXES = np.array([
    [0.998539, -0.021993, -0.049362, -0.029317],
    [-0.022996, -0.999539, -0.019843, 0.012864],
    [-0.048903, 0.020949, -0.998584, 0.098497],
    [0.0, 0.0, 0.0, 1.0],
])
# Offsets along the tool axis from the template grasp pose.
APPROACH_OFFSET_M, PREGRASP_OFFSET_M, CONTACT_OFFSET_M = -0.07, -0.025, 0.01
# When the nominal approach pose has no endpoint (near the arm's reach, IK
# convergence there is sensitive to sub-millimetre goal changes), is blocked,
# or plans only near an arm singularity, retry it closer along the same tool
# axis, at most down to the pregrasp pose; the pregrasp and contact poses are
# unchanged.
APPROACH_SHORTENINGS_M = (0., 0.01, 0.02, 0.03, 0.045)
APPROACH_RETRY_REASONS = frozenset({"safe_tcp_endpoint_search_exhausted", "safe_tcp_target_blocked",
                                    "near_arm_singularity"})
# Smallest singular value of the 6x6 flange Jacobian (m/rad and rad/rad rows)
# a planned approach or pregrasp configuration must keep. Near the stretched
# elbow it falls to ~0 and the TCP cannot settle within tolerance (an approach
# planned there timed out); a normal reach keeps about 0.06.
MINIMUM_STAGE_JACOBIAN_SIGMA = 0.02
# Side and front grasps. The DMTAC_M pads close along tool y; tool x points
# down. Collision boxes of the open hand in tool_tcp (x, y, z ranges, from the
# deployed URDF): the wrist flange reaches 61 mm along tool x, so a level
# approach must keep tool_tcp at least 65 mm above the surface. Tilting the
# approach down lifts the flange (it sits 9-17 cm behind tool_tcp).
HAND_BOXES_IN_TCP = (
    ((-0.054, 0.061), (-0.047, 0.043), (-0.170, -0.091)),   # gripper flange
    ((-0.030, 0.040), (-0.038, 0.038), (-0.170, -0.138)),   # arm link 6
    ((-0.058, -0.013), (-0.022, 0.022), (-0.134, -0.106)),  # wrist camera
    ((0.011, 0.039), (-0.045, 0.045), (-0.133, -0.023)),    # gripper base
    ((0.015, 0.034), (-0.057, 0.057), (-0.047, 0.031)),     # finger links
    ((0.003, 0.043), (-0.085, 0.086), (0.022, 0.099)),      # sensor pads
)
PAD_BOX_IN_TCP = ((0.003, 0.043), (0.022, 0.099))  # (tool x, tool z) of the pads
# Object center along the pads: tool x 23 mm, tool z 65 mm.
HORIZONTAL_GRASP_POINT_IN_TCP = (0.023, 0.0, 0.065)
HORIZONTAL_SURFACE_CLEARANCE_M = 0.004
HORIZONTAL_MINIMUM_PAD_OVERLAP_M = 0.008
HORIZONTAL_PREGRASP_BACKOFF_M = 0.035
HORIZONTAL_APPROACH_BACKOFF_M = 0.08
TILTED_PITCH_RAD = math.radians(35.)
# Fixed preference order: (label, family, top-down yaw offset or None, pad shift).
# Pads stay centered on the face: a 1 cm shift cleared a nearby post but its
# 26 mm contact overlap exceeded the 1 mm hard penetration bound in the lift.
# Horizontal entries carry their downward pitch instead of a top-down yaw.
CANDIDATES = (
    ("top_down", "top_down", 0., 0.),
    ("top_down_turned_left", "top_down", math.pi / 2., 0.),
    ("top_down_turned_right", "top_down", -math.pi / 2., 0.),
    ("side", "side", 0., 0.),
    ("side_tilted", "side", TILTED_PITCH_RAD, 0.),
    ("front_tilted", "front", TILTED_PITCH_RAD, 0.),
    ("top_down_half_turn", "top_down", math.pi, 0.),
)
SEGMENTATION_MIN_SCORE = 0.3
PLANNING_TIME_BUDGET_SEC = 45.
PATH_SEARCH_BUDGET_SEC = 8.
_RESULT_RESERVE_SEC = 1.0


def _orthonormal(matrix):
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    result = matrix.copy()
    result[:3, :3] = rotation
    return result


def _z_turn(angle):
    rotation = np.eye(4)
    rotation[:2, :2] = ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    return rotation


TOP_FROM_GRASP = _orthonormal(_z_turn(math.pi) @ _TOP_FROM_GRASP_WORLD_AXES)


def _along_tool(pose, distance):
    offset = np.eye(4)
    offset[2, 3] = distance
    return pose @ offset


def _yaw(angle):
    return _z_turn(angle)


def _box_corner_z(rotation, translation, box):
    corners = np.array([(x, y, z) for x in box[0] for y in box[1] for z in box[2]])
    return corners @ rotation[2] + translation[2]


def horizontal_grasp(center_xy, *, approach_xy, object_top_z, surface_z, pitch_rad=0.):
    """Contact tool_tcp pose for a face grasp approaching along approach_xy, or None.

    The approach is pitched down by pitch_rad about the horizontal closing
    axis. The grasp point rises from the object's mid-height until every hand
    box clears the surface, and is refused when the pads then overlap less
    than 8 mm of the object's height.
    """
    approach = np.array([approach_xy[0], approach_xy[1], 0.])
    approach /= np.linalg.norm(approach)
    level = np.empty((3, 3))
    level[:, 0] = (0., 0., -1.)
    level[:, 2] = approach
    level[:, 1] = np.cross(level[:, 2], level[:, 0])
    c, s_ = math.cos(pitch_rad), math.sin(pitch_rad)
    rotation = level @ np.array([[c, 0., s_], [0., 1., 0.], [-s_, 0., c]])
    grasp = np.asarray(HORIZONTAL_GRASP_POINT_IN_TCP)
    object_height = object_top_z - surface_z
    # The grasp point may rise above the top face while the pads still
    # overlap the object's upper part; the overlap check below bounds it.
    for height in np.arange(object_height / 2., object_height + .05, .002):
        point = np.array([center_xy[0], center_xy[1], surface_z + height])
        translation = point - rotation @ grasp
        lowest = min(float(np.min(_box_corner_z(rotation, translation, box))) for box in HAND_BOXES_IN_TCP)
        if lowest < surface_z + HORIZONTAL_SURFACE_CLEARANCE_M:
            continue
        pad = _box_corner_z(rotation, translation, (PAD_BOX_IN_TCP[0], (0.,), PAD_BOX_IN_TCP[1]))
        overlap = min(object_top_z, float(np.max(pad))) - max(surface_z, float(np.min(pad)))
        if overlap < HORIZONTAL_MINIMUM_PAD_OVERLAP_M:
            return None
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        return pose
    return None


def candidate_grasps(measured_top, *, surface_z, motion_side):
    """(label, family, yaw offset or None, approach, pregrasp, contact) in preference order."""
    yaw = float(measured_top.yaw_rad)
    reference_from_top = _yaw(yaw)
    reference_from_top[:3, 3] = (measured_top.center_xy_m[0], measured_top.center_xy_m[1],
                                 measured_top.top_z_m)
    side_sign = -1. if motion_side == "left" else 1.
    out = []
    for label, family, offset, pad_shift in CANDIDATES:
        if family == "top_down":
            shift = np.eye(4)
            shift[0, 3] = pad_shift
            grasp = reference_from_top @ _yaw(offset) @ TOP_FROM_GRASP @ shift
            out.append((label, family, float(offset), _along_tool(grasp, APPROACH_OFFSET_M),
                        _along_tool(grasp, PREGRASP_OFFSET_M), _along_tool(grasp, CONTACT_OFFSET_M)))
            continue
        if surface_z is None or measured_top.top_z_m - surface_z < 0.01:
            continue
        # Face normals of the measured box: the one nearest the arm's inward
        # direction (right arm points left) or nearest straight ahead.
        local = (0., side_sign) if family == "side" else (1., 0.)
        approach_xy = (math.cos(yaw) * local[0] - math.sin(yaw) * local[1],
                       math.sin(yaw) * local[0] + math.cos(yaw) * local[1])
        contact = horizontal_grasp(measured_top.center_xy_m, approach_xy=approach_xy,
                                   object_top_z=measured_top.top_z_m, surface_z=surface_z,
                                   pitch_rad=float(offset))
        if contact is None:
            continue
        out.append((label, family, None, _along_tool(contact, -HORIZONTAL_APPROACH_BACKOFF_M),
                    _along_tool(contact, -HORIZONTAL_PREGRASP_BACKOFF_M), contact))
    return tuple(out)


@dataclass(frozen=True)
class GraspApproachQueryRegistration:
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet
    profile_hash: str
    roles: tuple[str, ...]
    prompt_by_object_id: Mapping[str, str]
    reference_frame_id: str


def build_grasp_approach_query_registration(*, segmentation, back_projection, roles,
                                            prompt_by_object_id, grasp_approach_contract,
                                            grasp_approach_route):
    """Bind the query to the segmenter, ordered RGB-D roles and the grasp approach Route."""
    prompts = {str(key): str(value) for key, value in dict(prompt_by_object_id).items()}
    roles = tuple(roles)
    if not prompts or any(not key for key in prompts):
        raise ValueError("Grasp queries need at least one named object prompt.")
    if not roles or len(set(roles)) != len(roles) or any(
            role not in back_projection.profile.roles
            or segmentation.profile.source_id_by_role.get(role)
            != back_projection.profile.source_id_by_role.get(role) for role in roles):
        raise ValueError("Grasp query roles need matching segmentation and aligned RGB-D sources.")
    for key in ("runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash",
                "execution_manifest_hash", "target_kind"):
        if getattr(segmentation.binding, key) != getattr(back_projection.binding, key):
            raise ValueError("Grasp query sources belong to another deployment.")
    frame = back_projection.profile.reference_frame_id
    motion_profile = grasp_approach_contract.motion_profile
    if frame != motion_profile.reference_frame_id:
        raise ValueError("Grasp query must use the grasp approach Route's public frame.")
    for prompt in prompts.values():
        for role in roles:
            segmentation.payload_validator.validate_parameters({
                "role": role, "language_prompt": prompt, "target_pixel_uv": None,
                "min_score": SEGMENTATION_MIN_SCORE})
    profile_hash = canonical_content_hash({
        "segmentation_route": segmentation.skill.routes[0].semantic_content_hash,
        "segmentation_profile": segmentation.profile.semantic_content_hash,
        "back_projection_profile": back_projection.profile.semantic_content_hash,
        "roles": list(roles), "view_selection": "most_measured_top_points",
        "prompt_by_object_id": dict(sorted(prompts.items())),
        "segmentation_min_score": SEGMENTATION_MIN_SCORE,
        "grasp_approach_route": grasp_approach_route.semantic_content_hash,
        "grasp_approach_profile": motion_profile.semantic_content_hash,
        "top_from_grasp": TOP_FROM_GRASP.round(9).tolist(),
        "tool_axis_offsets_m": [APPROACH_OFFSET_M, PREGRASP_OFFSET_M, CONTACT_OFFSET_M],
        "approach_shortenings_m": list(APPROACH_SHORTENINGS_M),
        "approach_retry_reasons": sorted(APPROACH_RETRY_REASONS),
        "minimum_stage_jacobian_sigma": MINIMUM_STAGE_JACOBIAN_SIGMA,
        "candidates": [list(item) for item in CANDIDATES],
        "horizontal": {"hand_boxes_in_tcp": [list(map(list, box)) for box in HAND_BOXES_IN_TCP],
                       "pad_box_in_tcp": [list(item) for item in PAD_BOX_IN_TCP],
                       "grasp_point_in_tcp": list(HORIZONTAL_GRASP_POINT_IN_TCP),
                       "surface_clearance_m": HORIZONTAL_SURFACE_CLEARANCE_M,
                       "minimum_pad_overlap_m": HORIZONTAL_MINIMUM_PAD_OVERLAP_M,
                       "backoff_m": [HORIZONTAL_PREGRASP_BACKOFF_M, HORIZONTAL_APPROACH_BACKOFF_M]},
        "selection": "first_feasible_in_preference_order",
        "planning_time_budget_sec": PLANNING_TIME_BUDGET_SEC,
        "path_search_budget_sec": PATH_SEARCH_BUDGET_SEC,
        "object_measurement": "depth_only_top_plane_minimum_area_rectangle_surface_ring_v1",
        "planner": "mujoco_private_two_stage_candidate_planner_v1",
    }, domain="agenticrobot.grasp_approach_query.profile.v1")
    definition = skill.definition()
    schema = skill.schema(definition_hash=definition.semantic_content_hash, profile_hash=profile_hash,
                          prompt_by_object_id=prompts, reference_frame_id=frame)
    io_contract = inline_only_skill_io_contract(
        skill_id=skill.SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash)
    base_route = back_projection.skill.routes[0]
    if base_route.control_effect_class != "read_only" or base_route.may_move_robot:
        raise ValueError("Grasp queries require a read-only base Route.")
    route = replace(base_route, skill_id=skill.SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_id=base_route.route_id + ".grasp_approach", route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash,
        max_operation_duration_ms=GRASP_APPROACH_MAX_OPERATION_MS)
    base_binding = back_projection.binding.to_ref()
    binding = replace(base_binding,
        deployment_binding_id="grasp-approach:" + base_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": base_binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.grasp_approach_query.binding.v1"))
    objects = tuple(sorted(prompts))

    def validate_parameters(value):
        skill.validate_parameters(value, object_ids=objects, reference_frame_id=frame)

    def validate_result(value):
        skill.validate_result(value, object_ids=objects, reference_frame_id=frame)

    def validate_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        skill.validate_result_for_parameters(parameters, result)

    validator = AgentSkillPayloadValidatorSet(
        validator_id="grasp-approach:" + profile_hash, skill_id=skill.SKILL_ID,
        skill_version="1", route_id=route.route_id, route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        validate_parameters=validate_parameters, validate_execution_result=validate_result,
        validate_execution_result_for_parameters=validate_result_for_parameters,
        validate_task_evaluation=validate_no_task_evaluation_v1)
    entry = SkillRouteManifestEntry(route=route, deployment_binding=binding, evaluation_plan_hash=None,
        availability_hash=canonical_content_hash({
            "route": route.semantic_content_hash, "profile": profile_hash,
            "status": "configured_requires_open_gripper_live_rgbd_text_segmentation_and_planner"},
            domain="agenticrobot.grasp_approach_query.availability.v1"))
    return GraspApproachQueryRegistration(
        SkillRegistrationTemplate(definition, schema, io_contract, (route,)), entry, validator,
        profile_hash, roles, prompts, frame)


def _rejected(*, parameters, revision, frame, reason, measured=None, views=(),
              candidate_count=0, attempted=0, planning_wall_sec=0., rejections=()):
    return {
        "semantics": skill.SEMANTICS, "status": "no_grasp_approach", "reason": reason,
        "object_id": parameters["object_id"], "reference_frame_id": frame,
        "station_revision": revision, **{key: None for key in skill.POSE_KEYS},
        "approach_family": None, "measured_object": measured, "yaw_offset_rad": None,
        "candidate_count": candidate_count, "attempted_candidate_count": attempted,
        "selected_candidate_index": None, "candidate_rejections": list(rejections),
        "path_planning_wall_sec": float(planning_wall_sec), "views": list(views),
        "motion_executed": False, "execution_authorized": False}


class MujocoGraspApproachQuery:
    """Measured object top, quarter-turn grasp proposals and two-stage private planning."""

    def __init__(self, *, registration: GraspApproachQueryRegistration, application,
                 observer, gripper_name: str, segmentation_provider, capture_port, motion_profile,
                 prepare_projection, maximum_depth_m: float,
                 current_station_revision: Callable[[], int], clock: Callable[[], float]) -> None:
        if not isinstance(registration, GraspApproachQueryRegistration):
            raise TypeError("Grasp query needs its typed registration.")
        if segmentation_provider is None or capture_port is None:
            raise ValueError("Grasp query needs the segmenter and RGB-D port.")
        if not callable(prepare_projection) or not callable(clock):
            raise TypeError("Grasp query needs the approach envelope projector and the Gateway clock.")
        self.registration = registration
        self.application = application
        self.observer = observer
        self.gripper_name = gripper_name
        self.segmentation_provider = segmentation_provider
        self.capture_port = capture_port
        self.motion_profile = motion_profile
        self.prepare_projection = prepare_projection
        self.maximum_depth_m = float(maximum_depth_m)
        self.current_station_revision = current_station_revision
        self.clock = clock
        self._lock = threading.Lock()

    def query(self, parameters: Mapping, *, expected_station_revision: int,
              cancel_requested: Callable[[], bool], deadline: float) -> tuple[dict, tuple[str, ...]]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Grasp approach query is already running.")
        try:
            return self._query(parameters, revision=expected_station_revision,
                               cancel_requested=cancel_requested, deadline=deadline)
        finally:
            self._lock.release()

    def _gripper_state(self):
        rig = self.application.runtime.rig
        command = rig.latest_gripper_command(self.gripper_name)
        target = None if command is None or command[1] is None else float(command[1])
        measured = float(rig.capture_scene(lambda scene: scene.get_gripper_close(self.gripper_name)))
        held = False
        if self.observer is not None:
            held = self.observer.diagnostic_snapshot().get("holding_verified") is True
        return target, measured, held

    def _query(self, parameters, *, revision, cancel_requested, deadline):
        registration = self.registration
        frame = registration.reference_frame_id
        common = dict(parameters=parameters, revision=revision, frame=frame)
        target, measured_close, held = self._gripper_state()
        if held:
            return _rejected(reason="object_already_held", **common), ("grasp-query:object-held",)
        if (target is not None and target > .5) or measured_close > .2:
            return _rejected(reason="gripper_not_open", **common), ("grasp-query:gripper-closed",)
        prompt = registration.prompt_by_object_id[parameters["object_id"]]
        evidence, views, best = [], [], None
        for role in registration.roles:
            if cancel_requested():
                raise InterruptedError("Grasp approach query cancelled.")
            paired = self.capture_port.capture_placement_rgbd(role=role, expected_station_revision=revision)
            evidence.append("rgbd-capture:" + role + ":" + paired.depth.depth_content_hash)
            prepared = self.segmentation_provider.try_segment_capture(capture=paired.rgb, parameters={
                "role": role, "language_prompt": prompt, "target_pixel_uv": None,
                "min_score": SEGMENTATION_MIN_SCORE})
            if prepared is None:
                views.append({"role": role, "outcome": "not_segmented", "score": None})
                continue
            result = prepared.execution_result
            mask = decode_segmentation_mask_npy(
                prepared.mask_npy, width=int(result["image_width"]),
                height=int(result["image_height"]), expected_count=int(result["mask_pixel_count"]),
                expected_bounds=tuple(int(v) for v in result["mask_bounds_xyxy"]))
            evidence.append("object-mask:" + role + ":" + str(result["mask_media_content_hash"]))
            try:
                top = measure_object_top(paired.depth, mask, maximum_depth_m=self.maximum_depth_m)
            except ValueError:
                views.append({"role": role, "outcome": "top_not_measured", "score": float(result["score"])})
                continue
            views.append({"role": role, "outcome": "measured", "score": float(result["score"])})
            surface = measure_surface_below(paired.depth, mask, top_z_m=top.top_z_m,
                                            maximum_depth_m=self.maximum_depth_m)
            if best is None or top.top_point_count > best[1].top_point_count:
                best = (role, top, float(result["score"]), surface)
        if best is None:
            reason = ("object_not_segmented" if all(view["outcome"] == "not_segmented" for view in views)
                      else "object_top_not_measured")
            return _rejected(reason=reason, views=views, **common), tuple(evidence)
        role, top, score, surface = best
        measured = {"role": role, "language_prompt": prompt, "score": score,
                    "top_center_xyz_m": [top.center_xy_m[0], top.center_xy_m[1], top.top_z_m],
                    "yaw_rad": top.yaw_rad, "extent_xy_m": list(top.extent_xy_m),
                    "top_point_count": top.top_point_count,
                    # Without a measured ring the surface is not assumed: only
                    # top-down candidates are proposed.
                    "support_z_m": float(surface if surface is not None
                                         else top.top_z_m - max(top.extent_xy_m))}
        common.update(measured=measured, views=views)
        candidates = candidate_grasps(top, surface_z=surface, motion_side=self.application.motion_side)
        began = time.monotonic()
        rejections = []
        for index, (label, family, offset, approach, pregrasp, contact) in enumerate(candidates):
            if cancel_requested():
                raise InterruptedError("Grasp approach query cancelled.")
            if float(self.clock()) >= deadline:
                return _rejected(reason="query_budget_exhausted", candidate_count=len(candidates),
                                 attempted=index, planning_wall_sec=time.monotonic() - began,
                                 rejections=rejections, **common), tuple(evidence)
            reason, approach_q = None, None
            for shortening in APPROACH_SHORTENINGS_M:
                trial = _along_tool(approach, shortening)
                stage_reason, approach_q = self._plan_stage(
                    trial, start_q=None, cancel_requested=cancel_requested, deadline=deadline)
                if stage_reason is None and self._jacobian_sigma_min(approach_q) < MINIMUM_STAGE_JACOBIAN_SIGMA:
                    stage_reason, approach_q = "near_arm_singularity", None
                if stage_reason is None:
                    approach = trial
                    break
                # Report the nominal approach's refusal if no shorter one plans.
                reason = reason or stage_reason
                if stage_reason not in APPROACH_RETRY_REASONS:
                    break
            if approach_q is not None:
                reason, pregrasp_q = self._plan_stage(pregrasp, start_q=approach_q,
                                                      cancel_requested=cancel_requested, deadline=deadline)
                if reason is None and self._jacobian_sigma_min(pregrasp_q) < MINIMUM_STAGE_JACOBIAN_SIGMA:
                    reason = "near_arm_singularity"
                if reason is not None:
                    reason = "pregrasp_" + reason
            else:
                reason = "approach_" + reason
            if reason is not None:
                rejections.append({"candidate": label, "reason_code": reason})
                continue
            poses = {key: [float(v) for v in pose_to_xyz_quat_xyzw(pose)] for key, pose in zip(
                skill.POSE_KEYS, (approach, pregrasp, contact))}
            evidence.append("grasp-approach-plan:" + canonical_content_hash(
                {"candidate": label, "poses": poses}, domain="agenticrobot.grasp_approach_query.plan.v1"))
            return {
                "semantics": skill.SEMANTICS, "status": "grasp_approach_found",
                "reason": "collision_free_approach_found", "object_id": parameters["object_id"],
                "reference_frame_id": frame, "station_revision": revision, **poses,
                "approach_family": family, "measured_object": measured,
                "yaw_offset_rad": None if offset is None else float(offset),
                "candidate_count": len(candidates), "attempted_candidate_count": index + 1,
                "selected_candidate_index": int(index), "candidate_rejections": rejections,
                "path_planning_wall_sec": float(time.monotonic() - began), "views": views,
                "motion_executed": False, "execution_authorized": False}, tuple(evidence)
        return _rejected(reason="no_collision_free_approach", candidate_count=len(candidates),
                         attempted=len(candidates), planning_wall_sec=time.monotonic() - began,
                         rejections=rejections, **common), tuple(evidence)

    def _planner(self, *, source_model_qpos=None):
        app = self.application
        rig = app.runtime.rig
        source = app.target.planned_tcp_motion
        options = {} if source_model_qpos is None else {"source_model_qpos": source_model_qpos}
        return MujocoPlannedTcpMotion(
            rig=SimpleNamespace(scene=app.scene, config=rig.config, capture_scene=rig.capture_scene),
            task_config=app.task_config, pose_projector=source.pose_projector,
            reference_hz=source.reference_hz,
            planning_clearance_padding_m=source.planning_clearance_padding_m,
            motion_side=app.motion_side, **options)

    def _goals(self, poses):
        app = self.application
        return tuple(app.transform_tcp_pose_xyzquat(
            tuple(float(v) for v in pose_to_xyz_quat_xyzw(pose)),
            source_frame_id=self.registration.reference_frame_id,
            target_frame_id=self.motion_profile.execution_reference_frame_id) for pose in poses)

    def _constraint(self, planner):
        project, identity = self.prepare_projection()
        side = self.application.motion_side

        def constraint(model_qpos, pair):
            pose = planner.pose_for_configuration(model_qpos, pair)
            projection = project({side: xyzquat_to_pose_matrix(pose)})
            if projection.modified_sides:
                return JointConfigurationCheck(safe=False, reason="tcp_envelope_blocked",
                                               metadata=projection.to_payload())
            return JointConfigurationCheck(safe=True)

        return constraint, identity

    def _select(self, planner, goals, *, cancel_requested, deadline):
        profile = self.motion_profile
        constraint, identity = self._constraint(planner)
        budget = min(PLANNING_TIME_BUDGET_SEC, max(1., float(deadline) - float(self.clock())))
        return planner.plan_candidates(
            goals, translation_tolerance_m=profile.translation_tolerance_m,
            rotation_tolerance_rad=profile.rotation_tolerance_rad,
            cancel_requested=lambda: cancel_requested() or float(self.clock()) >= deadline,
            time_budget_sec=budget, path_search_time_budget_sec=PATH_SEARCH_BUDGET_SEC,
            max_reference_count=profile.max_control_cycles - profile.stable_measurement_count,
            configuration_constraint=constraint, constraint_identity=identity,
            normalize_ik_objective=profile.normalize_ik_objective)

    def _jacobian_sigma_min(self, q):
        """Smallest singular value of the motion arm's flange Jacobian at q (private copy)."""
        app = self.application
        names = app.scene.arm_specs[app.motion_side].joint_names
        if len(names) != len(q):
            raise ValueError("Stage configuration differs from the motion-side arm.")

        def compute(live):
            model, mujoco = live.model, live.mujoco
            data = mujoco.MjData(model)
            mujoco.mj_copyData(data, model, live.data)
            for name, value in zip(names, q):
                data.qpos[live.joint_qpos_addr(name)] = float(value)
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            dofs = [int(model.joint(name).dofadr[0]) for name in names]
            linear, angular = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, linear, angular, int(model.joint(names[-1]).bodyid[0]))
            jacobian = np.vstack([linear[:, dofs], angular[:, dofs]])
            return float(np.linalg.svd(jacobian, compute_uv=False)[-1])

        return app.runtime.rig.capture_scene(compute)

    def _plan_stage(self, pose, *, start_q, cancel_requested, deadline):
        """(None, final arm q) when a path to pose exists from start_q (None: the live state)."""
        source = None
        if start_q is not None:
            app = self.application
            scene = app.scene
            source = app.runtime.rig.capture_scene(lambda live: live.data.qpos.copy())
            names = scene.arm_specs[app.motion_side].joint_names
            if len(names) != len(start_q):
                raise ValueError("Stage configuration differs from the motion-side arm.")
            for name, value in zip(names, start_q):
                source[scene.joint_qpos_addr(name)] = value
        planner = self._planner(source_model_qpos=source)
        try:
            selection = self._select(planner, self._goals((pose,)), cancel_requested=cancel_requested,
                                     deadline=deadline)
            return None, np.asarray(selection.plan.trajectory.q_rad[-1], dtype=float)
        except MujocoTcpPlanRejected as error:
            if cancel_requested():
                raise InterruptedError("Grasp approach query cancelled.") from None
            details = error.details.get("candidates") or ()
            reasons = [item.get("reason_code") for item in details if item.get("reason_code")]
            return (reasons[0] if reasons else error.reason_code), None
        finally:
            planner.clear()


def build_grasp_approach_query_executor(*, registration, query_port, current_station_revision,
                                        clock, max_retained_operations):
    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        revision = request.expected_station_revision
        result, evidence = query_port.query(
            request.parameters, expected_station_revision=revision,
            cancel_requested=lambda: executor.cancel_requested(operation.operation_id),
            deadline=operation.deadline_at - _RESULT_RESERVE_SEC)
        if current_station_revision() != revision:
            raise RuntimeError("Station revision changed during the grasp query; retry.")
        return PassiveReadResult(result, ("grasp-approach:" + operation.operation_id, *evidence),
                                 reason_code="grasp_approach_assessed")

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.grasp-approach.v1", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations)
    return executor


__all__ = ["CANDIDATES", "GRASP_APPROACH_MAX_OPERATION_MS", "GraspApproachQueryRegistration",
           "MujocoGraspApproachQuery", "TOP_FROM_GRASP", "build_grasp_approach_query_executor",
           "build_grasp_approach_query_registration", "candidate_grasps", "horizontal_grasp"]

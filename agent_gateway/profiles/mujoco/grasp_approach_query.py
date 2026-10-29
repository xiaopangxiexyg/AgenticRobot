"""MuJoCo registration and port for ``perception.query_grasp_approach@1``.

One read-only request, with the motion application HOLD and no active
operation:

1. require an open gripper and an object that is not held;
2. per bound camera role, text-segment the object with its deployment-bound
   prompt and measure its top face from depth only (center, height, yaw);
3. place the reviewed DMTAC top-down grasp template on the measured top,
   turned by quarter turns about the object's vertical axis, and derive the
   approach, pregrasp and contact tool_tcp poses along the tool axis;
4. rank the approach poses with a private planner bound to the grasp approach
   Route's profile and envelope, then require the straight pregrasp move from
   the selected approach configuration to be collision-free as well.

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

from agent_gateway.profiles.measured_support import measure_object_top
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
# measured yaw). Derived from the qualified scripted pregrasp that secures
# bilateral sensor contact on the place-suite cube.
TOP_FROM_GRASP = np.array([
    [0.998539, -0.021993, -0.049362, -0.029317],
    [-0.022996, -0.999539, -0.019843, 0.012864],
    [-0.048903, 0.020949, -0.998584, 0.098497],
    [0.0, 0.0, 0.0, 1.0],
])
# Offsets along the tool axis from the template grasp pose.
APPROACH_OFFSET_M, PREGRASP_OFFSET_M, CONTACT_OFFSET_M = -0.07, -0.025, 0.01
YAW_OFFSETS_RAD = (0., math.pi / 2., math.pi, -math.pi / 2.)
SEGMENTATION_MIN_SCORE = 0.3
PLANNING_TIME_BUDGET_SEC = 30.
PATH_SEARCH_BUDGET_SEC = 3.
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


TOP_FROM_GRASP = _orthonormal(TOP_FROM_GRASP)


def _along_tool(pose, distance):
    offset = np.eye(4)
    offset[2, 3] = distance
    return pose @ offset


def _yaw(angle):
    rotation = np.eye(4)
    rotation[:2, :2] = ((math.cos(angle), -math.sin(angle)), (math.sin(angle), math.cos(angle)))
    return rotation


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
        "yaw_offsets_rad": list(YAW_OFFSETS_RAD),
        "planning_time_budget_sec": PLANNING_TIME_BUDGET_SEC,
        "path_search_budget_sec": PATH_SEARCH_BUDGET_SEC,
        "object_measurement": "depth_only_top_plane_minimum_area_rectangle_v1",
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
              candidate_count=0, attempted=0, planning_wall_sec=0.):
    return {
        "semantics": skill.SEMANTICS, "status": "no_grasp_approach", "reason": reason,
        "object_id": parameters["object_id"], "reference_frame_id": frame,
        "station_revision": revision, **{key: None for key in skill.POSE_KEYS},
        "measured_object": measured, "yaw_offset_rad": None, "candidate_count": candidate_count,
        "attempted_candidate_count": attempted, "selected_candidate_index": None,
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
            if best is None or top.top_point_count > best[1].top_point_count:
                best = (role, top, float(result["score"]))
        if best is None:
            reason = ("object_not_segmented" if all(view["outcome"] == "not_segmented" for view in views)
                      else "object_top_not_measured")
            return _rejected(reason=reason, views=views, **common), tuple(evidence)
        role, top, score = best
        measured = {"role": role, "language_prompt": prompt, "score": score,
                    "top_center_xyz_m": [top.center_xy_m[0], top.center_xy_m[1], top.top_z_m],
                    "yaw_rad": top.yaw_rad, "extent_xy_m": list(top.extent_xy_m),
                    "top_point_count": top.top_point_count}
        common.update(measured=measured, views=views)
        reference_from_top = _yaw(top.yaw_rad)
        reference_from_top[:3, 3] = (top.center_xy_m[0], top.center_xy_m[1], top.top_z_m)
        candidates = []
        for offset in YAW_OFFSETS_RAD:
            grasp = reference_from_top @ _yaw(offset) @ TOP_FROM_GRASP
            candidates.append((offset, _along_tool(grasp, APPROACH_OFFSET_M),
                               _along_tool(grasp, PREGRASP_OFFSET_M), _along_tool(grasp, CONTACT_OFFSET_M)))
        began = time.monotonic()
        remaining = list(range(len(candidates)))
        planned = dict(candidate_count=len(candidates), attempted=len(candidates))
        while remaining:
            if cancel_requested():
                raise InterruptedError("Grasp approach query cancelled.")
            if float(self.clock()) >= deadline:
                return _rejected(reason="query_budget_exhausted", planning_wall_sec=time.monotonic() - began,
                                 **planned, **common), tuple(evidence)
            try:
                choice, approach_q = self._plan_approach([candidates[i][1] for i in remaining],
                                                         cancel_requested=cancel_requested, deadline=deadline)
            except MujocoTcpPlanRejected:
                if cancel_requested():
                    raise InterruptedError("Grasp approach query cancelled.") from None
                break
            index = remaining[choice]
            if self._straight_move_feasible(approach_q, candidates[index][2],
                                            cancel_requested=cancel_requested, deadline=deadline):
                offset, approach, pregrasp, contact = candidates[index]
                poses = {key: [float(v) for v in pose_to_xyz_quat_xyzw(pose)] for key, pose in zip(
                    skill.POSE_KEYS, (approach, pregrasp, contact))}
                evidence.append("grasp-approach-plan:" + canonical_content_hash(
                    {"candidate_index": index, "poses": poses},
                    domain="agenticrobot.grasp_approach_query.plan.v1"))
                return {
                    "semantics": skill.SEMANTICS, "status": "grasp_approach_found",
                    "reason": "collision_free_approach_found", "object_id": parameters["object_id"],
                    "reference_frame_id": frame, "station_revision": revision, **poses,
                    "measured_object": measured, "yaw_offset_rad": float(offset),
                    "candidate_count": len(candidates), "attempted_candidate_count": len(candidates),
                    "selected_candidate_index": int(index),
                    "path_planning_wall_sec": float(time.monotonic() - began), "views": views,
                    "motion_executed": False, "execution_authorized": False}, tuple(evidence)
            remaining.remove(index)
        return _rejected(reason="no_collision_free_approach", planning_wall_sec=time.monotonic() - began,
                         **planned, **common), tuple(evidence)

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

    def _plan_approach(self, poses, *, cancel_requested, deadline):
        planner = self._planner()
        try:
            selection = self._select(planner, self._goals(poses), cancel_requested=cancel_requested,
                                     deadline=deadline)
            return selection.candidate_index, np.asarray(selection.plan.trajectory.q_rad[-1], dtype=float)
        finally:
            planner.clear()

    def _straight_move_feasible(self, approach_q, pregrasp, *, cancel_requested, deadline):
        app = self.application
        scene = app.scene
        qpos = app.runtime.rig.capture_scene(lambda live: live.data.qpos.copy())
        names = scene.arm_specs[app.motion_side].joint_names
        if len(names) != len(approach_q):
            raise ValueError("Approach configuration differs from the motion-side arm.")
        for name, value in zip(names, approach_q):
            qpos[scene.joint_qpos_addr(name)] = value
        planner = self._planner(source_model_qpos=qpos)
        try:
            self._select(planner, self._goals((pregrasp,)), cancel_requested=cancel_requested,
                         deadline=deadline)
            return True
        except MujocoTcpPlanRejected:
            if cancel_requested():
                raise InterruptedError("Grasp approach query cancelled.") from None
            return False
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


__all__ = ["GRASP_APPROACH_MAX_OPERATION_MS", "GraspApproachQueryRegistration",
           "MujocoGraspApproachQuery", "TOP_FROM_GRASP", "build_grasp_approach_query_executor",
           "build_grasp_approach_query_registration"]

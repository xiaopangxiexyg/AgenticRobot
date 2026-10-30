"""MuJoCo registration and port for ``perception.query_support_release_pose@1``.

One read-only request, with the motion application HOLD and no active
operation:

1. require a verified, unsupported held object (the same lifted state that
   ``motion.move_grasped_object_to_pose`` phase=carry requires);
2. capture correlated RGB-D for the bound camera role, text-segment the
   selected support with its deployment-bound prompt and measure a certified
   top rectangle from depth only (``agent_gateway.profiles.measured_support``);
3. re-capture depth, require the support pixels unchanged, then read the held
   box and propose on-support release poses with bounded yaw turns;
4. check every proposal with a private, never-activated planner that uses the
   carry Route's motion profile, envelope and measured payload binding, and
   keep the planner's ranked selection.

Native body poses never enter the support measurement. Nothing is commanded,
retained as a plan or authorized for execution.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from types import SimpleNamespace

import numpy as np

from agent_gateway.profiles.back_projection import LiveAlignedDepthCapture
from agent_gateway.profiles.measured_support import measure_support_top
from agent_gateway.profiles.segmentation import decode_segmentation_mask_npy
from agent_gateway.protocol.skill_catalog import SkillRouteManifestEntry
from agent_gateway.protocol.skill_io_contracts import inline_only_skill_io_contract
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception import support_release_query as skill
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.object_transport import plan_release_approach_candidates
from robot_motion.path_planning import JointConfigurationCheck
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.payload_motion import MujocoPayloadBinding
from sim.mujoco.deploy.planned_tcp_motion import MujocoPlannedTcpMotion, MujocoTcpPlanRejected
from sim.mujoco.transforms import pose_to_xyz_quat_xyzw

from .held_box import MujocoHeldBoxObservationPort

SUPPORT_RELEASE_MAX_OPERATION_MS = 90_000
# Bounded yaw turns of the held object about the support normal. The current
# orientation stays first; a symmetric 8-angle sample is the planner's limit.
RELEASE_YAW_OFFSETS_RAD = (0., -math.pi / 4., math.pi / 4., -math.pi / 2., math.pi / 2.,
                           -3. * math.pi / 4., 3. * math.pi / 4., math.pi)
RELEASE_LATERAL_MARGIN_M = 0.003
MAXIMUM_RELEASE_CLEARANCE_M = 0.05
PLANNING_TIME_BUDGET_SEC = 45.
PATH_SEARCH_BUDGET_SEC = 8.
SEGMENTATION_MIN_SCORE = 0.3
DEPTH_UNCHANGED_TOLERANCE_M = 0.001
# Leave the executor time to validate and retain its result before the deadline.
_RESULT_RESERVE_SEC = 1.0


@dataclass(frozen=True)
class SupportReleaseQueryRegistration:
    skill: SkillRegistrationTemplate
    entry: SkillRouteManifestEntry
    payload_validator: AgentSkillPayloadValidatorSet
    profile_hash: str
    roles: tuple[str, ...]
    prompt_by_support_id: Mapping[str, str]
    support_object_ids: tuple[str, ...]
    reference_frame_id: str


def build_support_release_query_registration(*, segmentation, back_projection, roles,
                                             prompt_by_support_id, object_motion_contract,
                                             object_motion_route):
    """Bind the query to the segmenter, ordered aligned RGB-D roles and the carry Route."""
    prompts = {str(key): str(value) for key, value in dict(prompt_by_support_id).items()}
    supports = tuple(sorted(prompts))
    roles = tuple(roles)
    if not supports:
        raise ValueError("Release queries need at least one support prompt.")
    if not roles or len(set(roles)) != len(roles) or any(
            role not in back_projection.profile.roles
            or segmentation.profile.source_id_by_role.get(role)
            != back_projection.profile.source_id_by_role.get(role) for role in roles):
        raise ValueError("Release query roles need matching segmentation and aligned RGB-D sources.")
    for key in ("runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash",
                "execution_manifest_hash", "target_kind"):
        if getattr(segmentation.binding, key) != getattr(back_projection.binding, key):
            raise ValueError("Release query sources belong to another deployment.")
    frame = back_projection.profile.reference_frame_id
    motion_profile = object_motion_contract.motion_profile
    if frame != motion_profile.reference_frame_id:
        # The returned pose is submitted unchanged to the carry Route.
        raise ValueError("Release query must use the object motion Route's public frame.")
    for support, prompt in prompts.items():
        if not support:
            raise ValueError("Release query support IDs must be non-empty.")
        for role in roles:
            segmentation.payload_validator.validate_parameters({
                "role": role, "language_prompt": prompt, "target_pixel_uv": None,
                "min_score": SEGMENTATION_MIN_SCORE})
    profile_hash = canonical_content_hash({
        "segmentation_route": segmentation.skill.routes[0].semantic_content_hash,
        "segmentation_profile": segmentation.profile.semantic_content_hash,
        "back_projection_profile": back_projection.profile.semantic_content_hash,
        "roles": list(roles), "view_selection": "largest_certified_top_rectangle",
        "prompt_by_support_id": dict(sorted(prompts.items())),
        "segmentation_min_score": SEGMENTATION_MIN_SCORE,
        "object_motion_route": object_motion_route.semantic_content_hash,
        "object_motion_profile": motion_profile.semantic_content_hash,
        "yaw_offsets_rad": list(RELEASE_YAW_OFFSETS_RAD),
        "lateral_margin_m": RELEASE_LATERAL_MARGIN_M,
        "planning_time_budget_sec": PLANNING_TIME_BUDGET_SEC,
        "path_search_budget_sec": PATH_SEARCH_BUDGET_SEC,
        "support_measurement": "depth_only_certified_top_rectangle_v1",
        "planner": "mujoco_private_payload_candidate_planner_v1",
    }, domain="agenticrobot.support_release_query.profile.v1")
    definition = skill.definition()
    schema = skill.schema(definition_hash=definition.semantic_content_hash, profile_hash=profile_hash,
                          support_object_ids=supports, reference_frame_id=frame)
    io_contract = inline_only_skill_io_contract(
        skill_id=skill.SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash)
    base_route = back_projection.skill.routes[0]
    if base_route.control_effect_class != "read_only" or base_route.may_move_robot:
        raise ValueError("Release queries require a read-only base Route.")
    route = replace(base_route, skill_id=skill.SKILL_ID, skill_version="1",
        skill_definition_hash=definition.semantic_content_hash,
        route_id=base_route.route_id + ".support_release_pose", route_version="1",
        parameter_schema_hash=schema.parameter_schema_hash,
        execution_result_schema_hash=schema.execution_result_schema_hash,
        task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
        allowed_parameter_profiles=schema.parameter_profiles,
        skill_io_contract_hash=io_contract.semantic_content_hash,
        max_operation_duration_ms=SUPPORT_RELEASE_MAX_OPERATION_MS)
    base_binding = back_projection.binding.to_ref()
    binding = replace(base_binding,
        deployment_binding_id="support-release-pose:" + base_binding.deployment_binding_id,
        deployment_binding_hash=canonical_content_hash(
            {"base": base_binding.deployment_binding_hash, "route": route.semantic_content_hash},
            domain="agenticrobot.support_release_query.binding.v1"))

    def validate_parameters(value):
        skill.validate_parameters(value, support_object_ids=supports, reference_frame_id=frame)

    def validate_result(value):
        skill.validate_result(value, support_object_ids=supports, reference_frame_id=frame)

    def validate_result_for_parameters(parameters, result):
        validate_parameters(parameters)
        validate_result(result)
        skill.validate_result_for_parameters(parameters, result)

    validator = AgentSkillPayloadValidatorSet(
        validator_id="support-release-pose:" + profile_hash, skill_id=skill.SKILL_ID,
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
            "status": "configured_requires_held_object_live_rgbd_text_segmentation_and_payload_planner"},
            domain="agenticrobot.support_release_query.availability.v1"))
    return SupportReleaseQueryRegistration(
        SkillRegistrationTemplate(definition, schema, io_contract, (route,)), entry, validator,
        profile_hash, roles, prompts, supports, frame)


def summarize_rejections(diagnostics):
    """Count planner refusals by reason; feasible candidates are not refusals."""
    counts = {}
    for item in diagnostics or ():
        reason = str(item.get("reason_code") or "unknown")
        if reason != "path_found":
            counts[reason] = counts.get(reason, 0) + 1
    return [{"reason_code": reason, "count": count}
            for reason, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]


def _rejected(*, parameters, revision, frame, reason, held_pose=None, segmentation=None,
              measured=None, candidate_count=0, attempted=0, planning_wall_sec=0., views=(),
              rejections=()):
    return {
        "semantics": skill.SEMANTICS, "status": "no_release_pose", "reason": reason,
        "support_object_id": parameters["support_object_id"], "reference_frame_id": frame,
        "station_revision": revision, "release_object_pose_xyz_quat_xyzw": None,
        "release_clearance_m": skill.requested_clearance_m(parameters),
        "held_object_pose_xyz_quat_xyzw": held_pose, "yaw_change_rad": None,
        "support_top_z_m": None if measured is None else measured["top_z_m"],
        "support_region_min_xy_m": None if measured is None else measured["region_min_xy_m"],
        "support_region_max_xy_m": None if measured is None else measured["region_max_xy_m"],
        "segmentation": segmentation,
        "measured_support": None if measured is None else measured["evidence"],
        "candidate_count": candidate_count, "attempted_candidate_count": attempted,
        "selected_candidate_index": None, "path_planning_wall_sec": float(planning_wall_sec),
        "candidate_rejections": list(rejections),
        "views": list(views), "motion_executed": False, "execution_authorized": False}


class MujocoSupportReleaseQuery:
    """Measured support, verified held box and private payload planning in one read."""

    def __init__(self, *, registration: SupportReleaseQueryRegistration, application, observer,
                 segmentation_provider, capture_port, motion_profile, prepare_projection,
                 maximum_depth_m: float, current_station_revision: Callable[[], int],
                 clock: Callable[[], float]) -> None:
        if not isinstance(registration, SupportReleaseQueryRegistration):
            raise TypeError("Release query needs its typed registration.")
        if observer is None or segmentation_provider is None or capture_port is None:
            raise ValueError("Release query needs the held-object observer, segmenter and RGB-D port.")
        if not set(registration.support_object_ids) <= set(observer.support_body_names[1:]):
            raise ValueError("Release query supports must be admitted named-release supports.")
        if not callable(prepare_projection) or not callable(clock):
            raise TypeError("Release query needs the carry envelope projector and the Gateway clock.")
        self.clock = clock
        self.registration = registration
        self.application = application
        self.observer = observer
        self.segmentation_provider = segmentation_provider
        self.capture_port = capture_port
        self.motion_profile = motion_profile
        self.prepare_projection = prepare_projection
        self.maximum_depth_m = float(maximum_depth_m)
        self.current_station_revision = current_station_revision
        self.held_port = MujocoHeldBoxObservationPort(
            application=application, observer=observer, object_id=observer.object_body_name,
            side=application.motion_side, reference_frame_id=registration.reference_frame_id,
            current_station_revision=current_station_revision)
        self._lock = threading.Lock()

    def _held_verified_without_observing(self) -> bool:
        # Do not run a carry-phase observation on a supported or open hand:
        # that would latch a relation failure for the next motion.
        _generation, target = self.observer.command_state()
        snapshot = self.observer.diagnostic_snapshot()
        return (target == 1. and snapshot.get("holding_verified") is True
                and snapshot.get("other_contact") is False
                and tuple(snapshot.get("sensor_contacts", ())) == (True, True))

    def query(self, parameters: Mapping, *, expected_station_revision: int,
              cancel_requested: Callable[[], bool], deadline: float) -> tuple[dict, tuple[str, ...]]:
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Support release query is already running.")
        try:
            return self._query(parameters, revision=expected_station_revision,
                               cancel_requested=cancel_requested, deadline=deadline)
        finally:
            self._lock.release()

    def _query(self, parameters, *, revision, cancel_requested, deadline):
        registration = self.registration
        frame = registration.reference_frame_id
        support_id = parameters["support_object_id"]
        clearance = skill.requested_clearance_m(parameters)
        common = dict(parameters=parameters, revision=revision, frame=frame)
        if not self._held_verified_without_observing():
            return _rejected(reason="held_object_not_verified_lifted", **common), (
                "support-release-precondition:held-not-verified",)
        prompt = registration.prompt_by_support_id[support_id]
        evidence, views, best = [], [], None
        for role in registration.roles:
            if cancel_requested():
                raise InterruptedError("Support release query cancelled.")
            paired = self.capture_port.capture_placement_rgbd(
                role=role, expected_station_revision=revision)
            evidence.append("rgbd-capture:" + role + ":" + paired.depth.depth_content_hash)
            prepared = self.segmentation_provider.try_segment_capture(capture=paired.rgb, parameters={
                "role": role, "language_prompt": prompt, "target_pixel_uv": None,
                "min_score": SEGMENTATION_MIN_SCORE})
            if prepared is None:
                views.append({"role": role, "outcome": "not_segmented", "score": None,
                              "certified_area_m2": None})
                continue
            result = prepared.execution_result
            mask = decode_segmentation_mask_npy(
                prepared.mask_npy, width=int(result["image_width"]),
                height=int(result["image_height"]), expected_count=int(result["mask_pixel_count"]),
                expected_bounds=tuple(int(v) for v in result["mask_bounds_xyxy"]))
            segmentation = {"role": role, "language_prompt": prompt, "score": float(result["score"]),
                            "mask_pixel_count": int(result["mask_pixel_count"]),
                            "mask_bounds_xyxy": [int(v) for v in result["mask_bounds_xyxy"]]}
            evidence.append("support-mask:" + role + ":" + str(result["mask_media_content_hash"]))
            try:
                top = measure_support_top(paired.depth, mask, maximum_depth_m=self.maximum_depth_m)
            except ValueError:
                views.append({"role": role, "outcome": "top_not_measured",
                              "score": segmentation["score"], "certified_area_m2": None})
                continue
            area = float((top.region_max_xy_m[0] - top.region_min_xy_m[0])
                         * (top.region_max_xy_m[1] - top.region_min_xy_m[1]))
            views.append({"role": role, "outcome": "measured", "score": segmentation["score"],
                          "certified_area_m2": area})
            if best is None or area > best[0]:
                best = (area, role, paired, mask, top, segmentation,
                        str(result["mask_media_content_hash"]))
        if best is None:
            reason = ("support_not_segmented"
                      if all(view["outcome"] == "not_segmented" for view in views)
                      else "support_top_not_measured")
            return _rejected(reason=reason, views=views, **common), tuple(evidence)
        _area, role, paired, mask, top, segmentation, mask_hash = best
        result = {"mask_media_content_hash": mask_hash}
        measured = {
            "top_z_m": top.top_z_m, "region_min_xy_m": list(top.region_min_xy_m),
            "region_max_xy_m": list(top.region_max_xy_m),
            "evidence": {"top_point_count": int(top.plane.top_point_count),
                         "plane_residual_m": float(top.plane.maximum_plane_residual_m),
                         "plane_tilt_deg": float(top.tilt_deg),
                         "footprint_sample_count": int(top.coverage.sample_count),
                         "observed_region_min_xy_m": list(top.observed_min_xy_m),
                         "observed_region_max_xy_m": list(top.observed_max_xy_m)}}
        common["views"] = views
        if cancel_requested():
            raise InterruptedError("Support release query cancelled.")
        # Segmentation is slow: the support must be unchanged in a fresh depth
        # cut, and that fresh cut carries the support time for the planner.
        fresh = self.capture_port.capture_aligned_depth(role=role, expected_station_revision=revision)
        self._require_unchanged(paired.depth, fresh, mask)
        try:
            held = self.held_port.capture_held_box(expected_station_revision=revision)
        except ValueError:
            return _rejected(reason="held_object_not_verified_lifted", segmentation=segmentation,
                             measured=measured, **common), tuple(evidence)
        held_pose = [float(v) for v in pose_to_xyz_quat_xyzw(held.reference_from_object)]
        support_ref = "support-measured-top:" + canonical_content_hash({
            "support_id": support_id, "depth": fresh.depth_content_hash,
            "mask": str(result["mask_media_content_hash"]),
            "region": [list(top.region_min_xy_m), list(top.region_max_xy_m)],
            "top_z_m": top.top_z_m}, domain="agenticrobot.support_release_query.support.v1")
        support = top.observation(support_id=support_id, reference_frame_id=frame,
                                  scene_revision=revision, measured_at=float(fresh.captured_at),
                                  evidence_ref=support_ref)
        evidence += [support_ref, held.pose_evidence_ref, held.grasp_evidence_ref]
        try:
            candidates = plan_release_approach_candidates(
                held, support, now=float(self.application.runtime.clock()), clearance_m=clearance,
                minimum_planning_clearance_m=skill.MIN_CLEARANCE_M,
                maximum_clearance_m=MAXIMUM_RELEASE_CLEARANCE_M,
                lateral_margin_m=RELEASE_LATERAL_MARGIN_M, yaw_offsets_rad=RELEASE_YAW_OFFSETS_RAD)
        except ValueError as error:
            if "fits" not in str(error):
                raise
            return _rejected(reason="held_footprint_does_not_fit_support", held_pose=held_pose,
                             segmentation=segmentation, measured=measured, **common), tuple(evidence)
        indices = self._admissible(candidates, held)
        admissible = [candidates[index] for index in indices]
        filtered = len(candidates) - len(indices)
        limit_rejections = ([{"reason_code": "carry_target_outside_route_limits", "count": filtered}]
                            if filtered else [])
        began = time.monotonic()
        try:
            selection = self._plan(admissible, cancel_requested=cancel_requested, deadline=deadline)
        except MujocoTcpPlanRejected as error:
            if cancel_requested():
                raise InterruptedError("Support release query cancelled.") from error
            reason = ("query_budget_exhausted" if error.reason_code in {
                "safe_tcp_path_search_exhausted", "safe_tcp_planning_cancelled"}
                else "no_collision_free_release_path")
            return _rejected(reason=reason, held_pose=held_pose, segmentation=segmentation,
                             measured=measured, candidate_count=len(candidates),
                             attempted=len(candidates), planning_wall_sec=time.monotonic() - began,
                             rejections=[*limit_rejections,
                                         *summarize_rejections(error.details.get("candidates"))],
                             **common), tuple(evidence)
        wall = time.monotonic() - began
        if selection is None:
            return _rejected(reason="no_collision_free_release_path", held_pose=held_pose,
                             segmentation=segmentation, measured=measured,
                             candidate_count=len(candidates), attempted=len(candidates),
                             rejections=limit_rejections, **common), tuple(evidence)
        chosen = admissible[selection.candidate_index]
        release = [float(v) for v in pose_to_xyz_quat_xyzw(chosen.proposal.object_goal)]
        evidence.append("support-release-plan:" + canonical_content_hash({
            "candidate_index": int(selection.candidate_index), "release_pose": release,
            "diagnostics_count": len(selection.diagnostics)},
            domain="agenticrobot.support_release_query.plan.v1"))
        return {
            "semantics": skill.SEMANTICS, "status": "release_pose_found",
            "reason": "collision_free_release_path_found", "support_object_id": support_id,
            "reference_frame_id": frame, "station_revision": revision,
            "release_object_pose_xyz_quat_xyzw": release, "release_clearance_m": clearance,
            "held_object_pose_xyz_quat_xyzw": held_pose,
            "yaw_change_rad": float(chosen.yaw_offset_rad),
            "support_top_z_m": float(top.top_z_m),
            "support_region_min_xy_m": list(top.region_min_xy_m),
            "support_region_max_xy_m": list(top.region_max_xy_m),
            "segmentation": segmentation, "measured_support": measured["evidence"],
            "candidate_count": len(candidates), "attempted_candidate_count": len(candidates),
            "selected_candidate_index": int(indices[selection.candidate_index]),
            "path_planning_wall_sec": float(wall),
            "candidate_rejections": [*limit_rejections, *summarize_rejections(selection.diagnostics)],
            "views": views,
            "motion_executed": False, "execution_authorized": False}, tuple(evidence)

    @staticmethod
    def _require_unchanged(source: LiveAlignedDepthCapture, fresh: LiveAlignedDepthCapture, mask):
        if (fresh.source_id != source.source_id
                or dict(fresh.intrinsics) != dict(source.intrinsics)
                or not np.allclose(fresh.reference_from_camera, source.reference_from_camera,
                                   atol=1e-9, rtol=0)
                or fresh.depth_m.shape != source.depth_m.shape):
            raise RuntimeError("Support camera binding changed during the release query; retry.")
        before, after = source.depth_m[mask], fresh.depth_m[mask]
        if before.size == 0 or float(np.max(np.abs(after - before))) > DEPTH_UNCHANGED_TOLERANCE_M:
            raise RuntimeError("Support moved during segmentation; retry the release query.")

    def _admissible(self, candidates, held):
        """Indices of proposals the carry Route would admit (workspace and delta limits)."""
        profile = self.motion_profile
        bounds = profile.workspace_bounds
        tool = held.reference_from_tool
        kept = []
        for index, candidate in enumerate(candidates):
            goal = candidate.proposal.object_goal
            tool_goal = candidate.proposal.tool_goal
            relative = tool[:3, :3].T @ tool_goal[:3, :3]
            angle = math.acos(max(-1., min(1., (float(np.trace(relative)) - 1.) / 2.)))
            if (all(bounds[2 * i] <= goal[i, 3] <= bounds[2 * i + 1] for i in range(3))
                    and float(np.linalg.norm(tool_goal[:3, 3] - tool[:3, 3]))
                    <= profile.max_translation_delta_m
                    and angle <= profile.max_rotation_delta_rad):
                kept.append(index)
        return tuple(kept[:64])

    def _plan(self, candidates, *, cancel_requested, deadline):
        if not candidates:
            return None
        app = self.application
        rig = app.runtime.rig
        source = app.target.planned_tcp_motion
        profile = self.motion_profile
        observer = self.observer
        object_id = observer.object_body_name
        cut = observer.observe(phase="carry")
        public = self.registration.reference_frame_id

        def bind(scene):
            def view(data):
                value = copy.copy(scene)
                value.data = data
                return value

            def measured_guard(data):
                try:
                    observer.observe(phase="carry", source_data=data)
                except ValueError as error:
                    return JointConfigurationCheck(safe=False, reason="object_follow_evidence_invalid",
                                                   metadata={"message": str(error)})
                return JointConfigurationCheck(safe=True)

            return MujocoPayloadBinding.capture(
                scene, scene.data, target_body_name=object_id,
                sensor_geom_names=observer.sensor_geom_names,
                tcp_pose_from_data=lambda data: observer.world_tcp_from_scene(view(data)),
                gripper_closed_from_data=lambda data: (
                    observer.command_state() == (cut.command_generation, 1.)
                    and observer.measured_closed(view(data).get_gripper_close(observer.gripper_name))),
                tool_from_object=cut.tool_from_object, support=None,
                measured_guard=measured_guard, guard_identity=profile.semantic_content_hash)

        payload = rig.capture_scene(bind)
        goals = tuple(app.transform_tcp_pose_xyzquat(
            tuple(float(v) for v in pose_to_xyz_quat_xyzw(item.proposal.tool_goal)),
            source_frame_id=public, target_frame_id=profile.execution_reference_frame_id)
            for item in candidates)
        project, identity = self.prepare_projection()
        side = app.motion_side
        planner = MujocoPlannedTcpMotion(
            rig=SimpleNamespace(scene=app.scene, config=rig.config, capture_scene=rig.capture_scene),
            task_config=app.task_config, pose_projector=source.pose_projector,
            reference_hz=source.reference_hz,
            planning_clearance_padding_m=source.planning_clearance_padding_m, motion_side=side)

        def constraint(model_qpos, pair):
            pose = planner.pose_for_configuration(model_qpos, pair)
            projection = project({side: xyzquat_to_pose_matrix(pose)})
            if projection.modified_sides:
                return JointConfigurationCheck(safe=False, reason="tcp_envelope_blocked",
                                               metadata=projection.to_payload())
            return JointConfigurationCheck(safe=True)

        budget = min(PLANNING_TIME_BUDGET_SEC, max(1., float(deadline) - float(self.clock())))
        try:
            return planner.plan_candidates(
                goals, translation_tolerance_m=profile.translation_tolerance_m,
                rotation_tolerance_rad=profile.rotation_tolerance_rad,
                cancel_requested=lambda: cancel_requested() or float(self.clock()) >= deadline,
                time_budget_sec=budget,
                path_search_time_budget_sec=PATH_SEARCH_BUDGET_SEC,
                max_reference_count=profile.max_control_cycles - profile.stable_measurement_count,
                configuration_constraint=constraint, constraint_identity=identity,
                normalize_ik_objective=profile.normalize_ik_objective, payload_binding=payload)
        finally:
            planner.clear()


def build_support_release_query_executor(*, registration, query_port, current_station_revision,
                                         clock, max_retained_operations):
    def read(request, operation):
        registration.payload_validator.validate_parameters(request.parameters)
        revision = request.expected_station_revision
        result, evidence = query_port.query(
            request.parameters, expected_station_revision=revision,
            cancel_requested=lambda: executor.cancel_requested(operation.operation_id),
            deadline=operation.deadline_at - _RESULT_RESERVE_SEC)
        if current_station_revision() != revision:
            raise RuntimeError("Station revision changed during the release query; retry.")
        return PassiveReadResult(result, ("support-release-pose:" + operation.operation_id, *evidence),
                                 reason_code="support_release_pose_assessed")

    executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:" + registration.entry.deployment_binding.deployment_binding_id,
        executor_id="passive.support-release-pose.v1", read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations)
    return executor


__all__ = ["MujocoSupportReleaseQuery", "SUPPORT_RELEASE_MAX_OPERATION_MS",
           "SupportReleaseQueryRegistration", "build_support_release_query_executor",
           "build_support_release_query_registration"]

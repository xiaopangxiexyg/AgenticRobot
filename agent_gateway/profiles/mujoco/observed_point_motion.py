"""Observed-reference composition on the original safe lifecycle, not a planner."""

from dataclasses import replace
import logging
import math
import time

import numpy as np

from agent_gateway.profiles.motion_primitive_resolution import MotionPrimitiveTargetResolution
from agent_gateway.profiles.observed_target import resolve_retained_observed_point
from agent_gateway.profiles.surface_anchor import resolve_surface_anchor
from agent_gateway.runtime.station_operation_worker import StationLifecycleActivationEvidence
from agent_gateway.skills.motion.observed_point import (
    SKILL_ID, POINT_SEMANTICS, validate_observed_anchor_parameters_v2,
    validate_observed_anchor_region_parameters_v3, observed_anchor_region_offsets_v3,
    validate_observed_point_parameters,
)
from robot_motion.action import ComponentAction
from robot_motion.spatial_goals import compute_spatial_goal
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import xyzquat_to_pose_matrix
from sim.mujoco.deploy.planned_tcp_motion import MujocoTcpPlanRejected

from .motion_skill import MujocoAgentSafeTcpPoseSkillLifecycle


class MujocoObservedPointMotionLifecycle(MujocoAgentSafeTcpPoseSkillLifecycle):
    def __init__(self, *, lookup_observation, route_manifest, payload_validators, **kwargs):
        if kwargs["route"].skill_id != SKILL_ID or kwargs["observed_planning_provider"] is None:
            raise ValueError("Observed-point motion requires its explicit Route and sensor-shaped safe planner.")
        self._lookup_observation = lookup_observation
        self._manifest = route_manifest
        self._validators = payload_validators
        self._source = None
        validate_observed_point_parameters(kwargs["request"].parameters,
                                          component_id=kwargs["motion_profile"].component_id)
        super().__init__(target_resolver=self._resolve_goal, **kwargs)

    def _validate_parameter_lineage(self, parameters, lineage):
        # This new schema binds frames in its exact Route/profile, not in copied
        # caller numbers. The base validates Route, binding and profile hashes.
        # The retained source frame is checked independently before activation.
        if any(key in parameters for key in lineage):
            raise ValueError("Observed-point motion does not accept caller-supplied frame lineage.")

    def _resolve_goal(self, measured):
        parameters, profile = self.request.parameters, self.motion_profile
        source = resolve_retained_observed_point(request=self.request, lookup=self._lookup_observation,
            manifest=self._manifest, validators=self._validators,
            current_entry=self._manifest.route(SKILL_ID, "1",
                route_id=self.request.route_id, route_version=self.request.route_version))
        graph = self.application.target_frame_profile.frame_graph
        frame = graph.frame_by_id.get(source.reference_frame_id)
        if (frame is None or frame.semantic_content_hash != source.reference_frame_spec_hash
                or source.frame_graph_hash != profile.frame_graph_hash):
            raise ValueError("Observed target frame identity differs from the active calibrated frame graph.")
        anchor = self.application.transform_tcp_pose_xyzquat(
            (*source.point_reference_m, 0., 0., 0., 1.),
            source_frame_id=source.reference_frame_id, target_frame_id=profile.reference_frame_id)
        current = self.application.transform_tcp_pose_xyzquat(measured,
            source_frame_id=profile.execution_reference_frame_id, target_frame_id=profile.reference_frame_id)
        goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(current),
            anchor_xyz_m=list(anchor[:3]), offset_xyz_m=parameters["offset_xyz_m"],
            offset_frame=parameters["offset_frame"], orientation_mode="preserve")
        bounds = profile.workspace_bounds
        if any(not bounds[2*i] <= goal.pose_xyz_quat_xyzw[i] <= bounds[2*i+1] for i in range(3)):
            raise ValueError("Observed relation goal is outside the deployed workspace.")
        self._source = source
        target = self.application.transform_tcp_pose_xyzquat(goal.pose_xyz_quat_xyzw,
            source_frame_id=profile.reference_frame_id, target_frame_id=profile.execution_reference_frame_id)
        return MotionPrimitiveTargetResolution(target_pose=target, evidence_refs=(
            "observed-point-operation:" + source.operation_id,
            "observed-point-request:" + source.request_hash,
            "observed-point-depth:" + source.depth_content_hash))

    def activate(self, **kwargs):
        try:
            return super().activate(**kwargs)
        except (ValueError, KeyError) as error:
            logging.getLogger(__name__).warning("Observed-point activation rejected: %s", error)
            hold = self._hold(reason="observed_point_reference_unavailable")
            return StationLifecycleActivationEvidence(succeeded=False,
                status="observed_point_reference_unavailable", lower_operation_id=self._lower_operation_id,
                evidence_refs=hold.evidence_refs, lower_hold_confirmed=hold.lower_hold_confirmed)

    def _success(self, *, operation, request, final_pose):
        completion = super()._success(operation=operation, request=request, final_pose=final_pose)
        final = self.application.transform_tcp_pose_xyzquat(final_pose,
            source_frame_id=self.motion_profile.execution_reference_frame_id,
            target_frame_id=self.motion_profile.reference_frame_id)
        translation, degrees = pose_delta_metrics(xyzquat_to_pose_matrix(final),
                                                  xyzquat_to_pose_matrix(self.public_target_pose))
        report = {"observation_operation_id": self._source.operation_id,
            "source_request_hash": self._source.request_hash, "point_index": self._source.point_index,
            "point_semantics": POINT_SEMANTICS, "reference_frame_id": self.motion_profile.reference_frame_id,
            "requested_pose_xyz_quat_xyzw": list(self.public_target_pose),
            "achieved_pose_xyz_quat_xyzw": list(final), "translation_error_m": translation,
            "rotation_error_rad": math.radians(degrees)}
        return replace(completion, execution_result={**completion.execution_result, "observed_point_report": report})


class MujocoAnchorMotionLifecycle(MujocoObservedPointMotionLifecycle):
    """Reuse the safe motion owner with an authenticated v2 anchor input."""

    def __init__(self, *, store, lookup_operation, capture_registration,
                 current_station_revision, clock_domain_id, **kwargs):
        route = kwargs["route"]
        if (route.skill_id != SKILL_ID or route.skill_version not in {"2", "3"}
                or kwargs["observed_planning_provider"] is None):
            raise ValueError("Anchor motion requires its explicit v2/v3 Route and sensor-shaped safe planner.")
        if route.skill_version == "3":
            validate_observed_anchor_region_parameters_v3(kwargs["request"].parameters,
                component_id=kwargs["motion_profile"].component_id)
        else:
            validate_observed_anchor_parameters_v2(kwargs["request"].parameters,
                component_id=kwargs["motion_profile"].component_id)
        self._store = store
        self._lookup_operation = lookup_operation
        self._capture_registration = capture_registration
        self._current_station_revision = current_station_revision
        self._clock_domain_id = clock_domain_id
        self._source = None
        self._artifact_id = None
        MujocoAgentSafeTcpPoseSkillLifecycle.__init__(
            self, target_resolver=self._resolve_goal, **kwargs)

    def _resolve_goal(self, measured):
        parameters, profile = self.request.parameters, self.motion_profile
        operation_id = parameters["surface_anchor_operation_id"]
        retained = self._lookup_operation(operation_id)
        if retained is None or retained[0] is None or retained[1] is None:
            raise ValueError("Surface anchor producer is unavailable; recapture the pixel.")
        bindings = [binding for binding in retained[1].output_artifact_bindings
                    if binding.slot_id == "surface_anchor" and binding.ordinal == 0]
        if len(bindings) != 1:
            raise ValueError("Surface anchor producer lacks one exact output Artifact.")
        # The model names only the source operation. Reconstruct the transient
        # input binding from its retained output, then reuse the common CAS and
        # producer-authentication boundary without creating another source owner.
        bound_request = replace(self.request, input_artifact_bindings=(
            replace(bindings[0], production_receipt_ref=None),))
        anchor, artifact_id = resolve_surface_anchor(
            request=bound_request, store=self._store,
            lookup_operation=self._lookup_operation,
            capture_registration=self._capture_registration,
            current_station_revision=self._current_station_revision,
            clock_domain_id=self._clock_domain_id)
        if anchor["source_operation_id"] != operation_id:
            raise ValueError("Surface anchor differs from the requested source operation.")
        source = anchor["measurement"]
        graph = self.application.target_frame_profile.frame_graph
        frame = graph.frame_by_id.get(source["reference_frame_id"])
        if (frame is None or frame.semantic_content_hash != source["reference_frame_spec_hash"]
                or source["frame_graph_hash"] != profile.frame_graph_hash):
            raise ValueError("Surface anchor frame differs from the active calibrated frame graph.")
        anchor_pose = self.application.transform_tcp_pose_xyzquat(
            (*source["world_xyz_m"], 0., 0., 0., 1.),
            source_frame_id=source["reference_frame_id"],
            target_frame_id=profile.reference_frame_id)
        current = self.application.transform_tcp_pose_xyzquat(measured,
            source_frame_id=profile.execution_reference_frame_id,
            target_frame_id=profile.reference_frame_id)
        self._anchor_public_xyz = tuple(anchor_pose[:3])
        self._activation_public_pose = tuple(current)
        bounds = profile.workspace_bounds
        if self.route.skill_version == "3":
            # The inherited lifecycle checks its provisional target before
            # search. Bind the first admissible region member there so a
            # nominal point outside bounds cannot suppress valid alternatives.
            goal = None
            for offset in observed_anchor_region_offsets_v3(parameters):
                candidate = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(current),
                    anchor_xyz_m=list(anchor_pose[:3]), offset_xyz_m=list(offset),
                    offset_frame="reference", orientation_mode="preserve")
                if (all(bounds[2*i] <= candidate.pose_xyz_quat_xyzw[i] <= bounds[2*i+1]
                        for i in range(3))
                        and candidate.translation_distance_m <= profile.max_translation_delta_m
                        and candidate.rotation_distance_rad <= profile.max_rotation_delta_rad):
                    goal = candidate
                    break
            if goal is None:
                raise ValueError("Surface anchor region has no target within the deployed profile.")
        else:
            goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(current),
                anchor_xyz_m=list(anchor_pose[:3]), offset_xyz_m=parameters["offset_xyz_m"],
                offset_frame=parameters["offset_frame"], orientation_mode="preserve")
            if any(not bounds[2*i] <= goal.pose_xyz_quat_xyzw[i] <= bounds[2*i+1] for i in range(3)):
                raise ValueError("Surface anchor relation goal is outside the deployed workspace.")
        self._source, self._artifact_id = anchor, artifact_id
        target = self.application.transform_tcp_pose_xyzquat(goal.pose_xyz_quat_xyzw,
            source_frame_id=profile.reference_frame_id,
            target_frame_id=profile.execution_reference_frame_id)
        return MotionPrimitiveTargetResolution(target_pose=target, evidence_refs=(
            "surface-anchor-operation:" + anchor["source_operation_id"],
            "surface-anchor-request:" + anchor["source_request_hash"],
            "surface-anchor-artifact:" + artifact_id))

    def _prepare_plan(self, *, constraint, constraint_identity, cancel_requested):
        if self.route.skill_version == "2":
            return super()._prepare_plan(constraint=constraint,
                constraint_identity=constraint_identity, cancel_requested=cancel_requested)
        if cancel_requested():
            raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
        observed = self._observed_planning_provider.capture(
            operation_id=self.operation.operation_id,
            expected_station_revision=self.request.expected_station_revision,
            cancel_requested=cancel_requested)
        offsets = observed_anchor_region_offsets_v3(self.request.parameters)
        deadline = time.perf_counter() + min(20., self._path_search_time_budget_sec * len(offsets))
        translation_tolerance, rotation_tolerance = self._planning_goal_tolerances()
        bounds = self.motion_profile.workspace_bounds
        failures = []
        for index, offset in enumerate(offsets):
            if cancel_requested():
                raise MujocoTcpPlanRejected("safe_tcp_planning_cancelled")
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise MujocoTcpPlanRejected("safe_tcp_region_budget_exhausted",
                    {"candidate_count": len(offsets), "considered_count": len(failures)})
            goal = compute_spatial_goal(current_tcp_pose_xyz_quat_xyzw=list(self._activation_public_pose),
                anchor_xyz_m=list(self._anchor_public_xyz), offset_xyz_m=list(offset),
                offset_frame="reference", orientation_mode="preserve")
            public = goal.pose_xyz_quat_xyzw
            if (any(not bounds[2 * axis] <= public[axis] <= bounds[2 * axis + 1]
                    for axis in range(3))
                    or goal.translation_distance_m > self.motion_profile.max_translation_delta_m
                    or goal.rotation_distance_rad > self.motion_profile.max_rotation_delta_rad):
                failures.append((index, "safe_tcp_target_exceeds_profile"))
                continue
            target = self.application.transform_tcp_pose_xyzquat(public,
                source_frame_id=self.motion_profile.reference_frame_id,
                target_frame_id=self.motion_profile.execution_reference_frame_id)
            try:
                plan = self._planner.plan(target,
                    translation_tolerance_m=translation_tolerance,
                    rotation_tolerance_rad=rotation_tolerance,
                    normalize_ik_objective=self.motion_profile.normalize_ik_objective,
                    cancel_requested=cancel_requested,
                    configuration_constraint=constraint,
                    constraint_identity=constraint_identity,
                    observed_planning_input=observed,
                    deadline_monotonic=deadline,
                    path_search_time_budget_sec=min(self._path_search_time_budget_sec, remaining))
            except MujocoTcpPlanRejected as error:
                if (error.reason_code == "safe_tcp_planning_cancelled"
                        and not cancel_requested() and time.perf_counter() >= deadline):
                    raise MujocoTcpPlanRejected("safe_tcp_region_budget_exhausted",
                        {"candidate_count": len(offsets), "considered_count": index + 1}) from error
                if error.reason_code not in {
                    "safe_tcp_target_blocked", "safe_tcp_endpoint_search_exhausted",
                    "safe_tcp_path_search_exhausted", "safe_tcp_retimed_path_blocked",
                    "safe_tcp_motion_quality_exceeded", "safe_tcp_target_envelope_blocked",
                }:
                    raise
                failures.append((index, error.reason_code))
                continue
            if (len(plan.reference_poses) + self.motion_profile.stable_measurement_count
                    >= self.motion_profile.max_control_cycles):
                # The planner has a pending, not yet activated handle. Discard
                # it before another candidate; never promote an overlong path.
                self._planner.clear()
                failures.append((index, "safe_tcp_path_exceeds_control_budget"))
                continue
            self.target_pose = tuple(target)
            self.public_target_pose = tuple(public)
            self.action = ComponentAction(action_schema=self.application.runtime.action_schema,
                values_by_component={self.component_id: np.asarray(target, dtype=np.float32)})
            self._selected_offset = tuple(offset)
            self._selected_index = index
            self._considered_count = index + 1
            return plan
        raise MujocoTcpPlanRejected("safe_tcp_no_candidate_path",
            {"candidate_count": len(offsets), "considered_count": len(failures),
             "candidate_reasons": tuple(failures)})

    def _success(self, *, operation, request, final_pose):
        completion = MujocoAgentSafeTcpPoseSkillLifecycle._success(
            self, operation=operation, request=request, final_pose=final_pose)
        final = self.application.transform_tcp_pose_xyzquat(final_pose,
            source_frame_id=self.motion_profile.execution_reference_frame_id,
            target_frame_id=self.motion_profile.reference_frame_id)
        translation, degrees = pose_delta_metrics(xyzquat_to_pose_matrix(final),
                                                  xyzquat_to_pose_matrix(self.public_target_pose))
        report = {"source_operation_id": self._source["source_operation_id"],
            "source_request_hash": self._source["source_request_hash"],
            "anchor_artifact_id": self._artifact_id,
            "point_semantics": POINT_SEMANTICS,
            "reference_frame_id": self.motion_profile.reference_frame_id,
            "requested_pose_xyz_quat_xyzw": list(self.public_target_pose),
            "achieved_pose_xyz_quat_xyzw": list(final),
            "translation_error_m": translation,
            "rotation_error_rad": math.radians(degrees)}
        if self.route.skill_version == "3":
            report.update(selected_offset_xyz_m=list(self._selected_offset),
                          selected_candidate_index=self._selected_index,
                          attempted_candidate_count=self._considered_count)
            return replace(completion, execution_result={**completion.execution_result,
                "surface_anchor_region_report": report})
        return replace(completion, execution_result={**completion.execution_result,
                                                     "surface_anchor_report": report})

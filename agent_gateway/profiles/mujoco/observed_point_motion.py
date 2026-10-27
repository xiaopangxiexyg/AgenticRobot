"""Observed-reference composition on the original safe lifecycle, not a planner."""

from dataclasses import replace
import logging
import math

from agent_gateway.profiles.motion_primitive_resolution import MotionPrimitiveTargetResolution
from agent_gateway.profiles.observed_target import resolve_retained_observed_point
from agent_gateway.runtime.station_operation_worker import StationLifecycleActivationEvidence
from agent_gateway.skills.motion.observed_point import SKILL_ID, POINT_SEMANTICS, validate_observed_point_parameters
from robot_motion.spatial_goals import compute_spatial_goal
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import xyzquat_to_pose_matrix

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

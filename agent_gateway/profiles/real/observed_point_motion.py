"""Saved sensor points resolved on the existing real safe-motion lifecycle."""

from dataclasses import replace
import logging
import math

from agent_gateway.profiles.motion_primitive_resolution import MotionPrimitiveTargetResolution
from agent_gateway.profiles.observed_target import resolve_retained_observed_point
from agent_gateway.profiles.surface_anchor import resolve_surface_anchor
from agent_gateway.runtime.artifact_store import ArtifactStoreError
from agent_gateway.runtime.station_operation_worker import StationLifecycleActivationEvidence
from agent_gateway.skills.motion.observed_point import SKILL_ID, POINT_SEMANTICS
from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.observed_scene import SafeMotionObservationConfig
from robot_motion.pose import pose_delta_metrics
from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix
from robot_motion.spatial_goals import compute_spatial_goal

from .motion_primitive_profiles import (
    RealAgentObservedAnchorProfile,
    RealAgentObservedPointProfile,
)
from .motion_primitive_skill import RealAgentPrimitiveSkillLifecycle


class _ObservedReferenceUnavailable(ValueError):
    pass


class RealObservedPointMotionLifecycle(RealAgentPrimitiveSkillLifecycle):
    """Change target resolution and feedback only; the base owns execution/HOLD."""

    def __init__(self, *, lookup_observation, observation_config, **kwargs):
        if (kwargs["route"].skill_id != SKILL_ID or kwargs["route"].skill_version != "1"
                or not isinstance(kwargs["profile"], RealAgentObservedPointProfile)
                or not callable(lookup_observation)):
            raise ValueError("Observed-point motion requires its exact Route and evidence lookup.")
        if not isinstance(observation_config, SafeMotionObservationConfig):
            raise TypeError("Observed-point motion requires the existing safe observation policy.")
        self._observation_config = observation_config
        self._lookup = lookup_observation
        self._source = self._public_target = None
        super().__init__(**kwargs)

    def _resolve_target(self, measured_pose):
        try:
            return self._resolve_observed_target(measured_pose)
        except (ValueError, KeyError, TypeError) as error:
            raise _ObservedReferenceUnavailable(str(error)) from error

    def _read_source(self):
        manifest = self.contract_bundle.route_manifest
        pair = self._lookup(self.request.parameters["observation_operation_id"])
        source = resolve_retained_observed_point(request=self.request, lookup=lambda _: pair,
            manifest=manifest, validators=self.contract_bundle.payload_validators,
            current_entry=manifest.route(SKILL_ID, "1",
                route_id=self.request.route_id, route_version=self.request.route_version))
        now = float(self.clock())
        if pair[1].started_at > now or pair[1].updated_at > now:
            raise ValueError("Observed target is expired or from the future; obtain a new observation.")
        # V1 reads a live provider's bounded cache but has no sensor timestamp.
        # Include that provider's admitted cache age; operation time alone could
        # underestimate sensor age. V2 uses acquisition, never re-projection time.
        observed_at = source.acquired_at
        if observed_at is None:
            capture_profile = self.contract_bundle.perception_capture.profile
            observed_at = pair[1].started_at - capture_profile.max_frame_age_ms / 1000.
        age = now - observed_at
        if not math.isfinite(age) or age < 0 or age > self._observation_config.maximum_scene_age_sec:
            raise ValueError("Observed target is expired or from the future; obtain a new observation.")
        return source

    def _validate_target_before_activation(self):
        try:
            if self._source is None or self._read_source() != self._source:
                raise ValueError("Observed target changed or was evicted during planning.")
        except (ValueError, KeyError, TypeError) as error:
            raise _ObservedReferenceUnavailable(str(error)) from error

    def _resolve_observed_target(self, measured_pose):
        source = self._read_source()
        graph = self.contract_bundle.frame_graph
        frame = graph.frame_by_id.get(source.reference_frame_id)
        if (frame is None or frame.semantic_content_hash != source.reference_frame_spec_hash
                or source.frame_graph_hash != self.profile.frame_graph_hash):
            raise ValueError("Observed target differs from the calibrated frame graph.")
        source_pose = xyzquat_to_pose_matrix((*source.point_reference_m, 0., 0., 0., 1.))
        anchor, _ = graph.transform_pose_matrix(source_pose,
            source_reference_frame_id=source.reference_frame_id,
            target_reference_frame_id=self.profile.reference_frame_id)
        measured_public, _ = graph.transform_pose_matrix(xyzquat_to_pose_matrix(measured_pose),
            source_reference_frame_id=self.profile.execution_reference_frame_id,
            target_reference_frame_id=self.profile.reference_frame_id)
        goal = compute_spatial_goal(
            current_tcp_pose_xyz_quat_xyzw=pose_matrix_to_xyzquat(measured_public).tolist(),
            anchor_xyz_m=anchor[:3, 3].tolist(),
            offset_xyz_m=self.request.parameters["offset_xyz_m"],
            offset_frame=self.request.parameters["offset_frame"], orientation_mode="preserve")
        execution, resolution = graph.transform_pose_matrix(
            xyzquat_to_pose_matrix(goal.pose_xyz_quat_xyzw),
            source_reference_frame_id=self.profile.reference_frame_id,
            target_reference_frame_id=self.profile.execution_reference_frame_id)
        self._source, self._public_target = source, tuple(goal.pose_xyz_quat_xyzw)
        digest = canonical_content_hash({"source_request_hash": source.request_hash,
            "point_index": source.point_index, "component_id": self.component_id,
            "public_target": self._public_target, "profile_hash": self.profile.semantic_content_hash,
            "frame_resolution_hash": resolution.semantic_content_hash},
            domain="agenticrobot.real_observed_point.target.v1")
        return (MotionPrimitiveTargetResolution(target_pose=tuple(pose_matrix_to_xyzquat(execution)),
            evidence_refs=("observed-point-operation:" + source.operation_id,
                "observed-point-depth:" + source.depth_content_hash,
                "observed-point-target:" + digest)), SKILL_ID + "@1", digest)

    def activate(self, *, request, operation, **kwargs):
        try:
            return super().activate(request=request, operation=operation, **kwargs)
        except _ObservedReferenceUnavailable as error:
            logging.getLogger(__name__).warning("Observed-point activation rejected: %s", error)
            hold = self.request_hold(operation=operation, reason="observed_point_reference_unavailable")
            return StationLifecycleActivationEvidence(succeeded=False,
                status="observed_point_reference_unavailable", lower_operation_id=self.lower_operation_id,
                evidence_refs=hold.evidence_refs, lower_hold_confirmed=hold.lower_hold_confirmed)

    def _extend_arrival_result(self, result, measured_pose):
        measured, _ = self.contract_bundle.frame_graph.transform_pose_matrix(
            xyzquat_to_pose_matrix(measured_pose),
            source_reference_frame_id=self.profile.execution_reference_frame_id,
            target_reference_frame_id=self.profile.reference_frame_id)
        achieved = pose_matrix_to_xyzquat(measured).tolist()
        translation, rotation = pose_delta_metrics(measured, xyzquat_to_pose_matrix(self._public_target))
        return {**result, "observed_point_report": {
            "observation_operation_id": self._source.operation_id,
            "source_request_hash": self._source.request_hash, "point_index": self._source.point_index,
            "point_semantics": POINT_SEMANTICS, "reference_frame_id": self.profile.reference_frame_id,
            "requested_pose_xyz_quat_xyzw": list(self._public_target),
            "achieved_pose_xyz_quat_xyzw": achieved, "translation_error_m": translation,
            "rotation_error_rad": math.radians(rotation)}}


class RealObservedAnchorMotionLifecycle(RealObservedPointMotionLifecycle):
    """Resolve an authenticated v2 producer, then use the original safe owner."""

    def __init__(self, *, lookup_observation, observation_config,
                 artifact_store, capture_registration, current_station_revision,
                 **kwargs):
        if (kwargs["route"].skill_id != SKILL_ID
                or kwargs["route"].skill_version != "2"
                or not isinstance(kwargs["profile"], RealAgentObservedAnchorProfile)
                or not callable(lookup_observation)
                or not callable(current_station_revision)
                or artifact_store is None or capture_registration is None):
            raise ValueError("Observed-anchor motion requires its exact v2 producer and safe Route.")
        if not isinstance(observation_config, SafeMotionObservationConfig):
            raise TypeError("Observed-anchor motion requires the existing safe observation policy.")
        self._observation_config = observation_config
        self._lookup = lookup_observation
        self._store = artifact_store
        self._capture_registration = capture_registration
        self._current_station_revision = current_station_revision
        self._source = self._public_target = None
        RealAgentPrimitiveSkillLifecycle.__init__(self, **kwargs)

    def _resolve_target(self, measured_pose):
        try:
            return super()._resolve_target(measured_pose)
        except (ArtifactStoreError, OSError) as error:
            raise _ObservedReferenceUnavailable(str(error)) from error

    def _validate_target_before_activation(self):
        try:
            return super()._validate_target_before_activation()
        except (ArtifactStoreError, OSError) as error:
            raise _ObservedReferenceUnavailable(str(error)) from error

    def _read_source(self):
        operation_id = self.request.parameters["surface_anchor_operation_id"]
        pair = self._lookup(operation_id)
        if pair is None or pair[0] is None or pair[1] is None:
            raise ValueError("Surface anchor producer is unavailable; recapture the point.")
        bindings = [binding for binding in pair[1].output_artifact_bindings
                    if binding.slot_id == "surface_anchor" and binding.ordinal == 0]
        if len(bindings) != 1:
            raise ValueError("Surface anchor producer lacks one exact output Artifact.")
        bound_request = replace(self.request, input_artifact_bindings=(
            replace(bindings[0], production_receipt_ref=None),))
        anchor, artifact_id = resolve_surface_anchor(
            request=bound_request,
            store=self._store,
            lookup_operation=self._lookup,
            capture_registration=self._capture_registration,
            current_station_revision=self._current_station_revision,
            clock_domain_id=self.contract_bundle.attachment.clock_domain_id,
        )
        if anchor["source_operation_id"] != operation_id:
            raise ValueError("Surface anchor differs from the requested producer.")
        now = float(self.clock())
        acquired_at = float(anchor["captured_at"])
        if (pair[1].started_at > now or pair[1].updated_at > now
                or not math.isfinite(now - acquired_at)
                or not 0 <= now - acquired_at <= self._observation_config.maximum_scene_age_sec):
            raise ValueError("Surface anchor is expired or from the future; recapture the point.")
        return anchor, artifact_id

    def _resolve_observed_target(self, measured_pose):
        anchor, artifact_id = self._read_source()
        source = anchor["measurement"]
        graph = self.contract_bundle.frame_graph
        frame = graph.frame_by_id.get(source["reference_frame_id"])
        if (frame is None or frame.semantic_content_hash != source["reference_frame_spec_hash"]
                or source["frame_graph_hash"] != self.profile.frame_graph_hash):
            raise ValueError("Surface anchor differs from the calibrated frame graph.")
        source_pose = xyzquat_to_pose_matrix((*source["world_xyz_m"], 0., 0., 0., 1.))
        public_anchor, _ = graph.transform_pose_matrix(source_pose,
            source_reference_frame_id=source["reference_frame_id"],
            target_reference_frame_id=self.profile.reference_frame_id)
        measured_public, _ = graph.transform_pose_matrix(xyzquat_to_pose_matrix(measured_pose),
            source_reference_frame_id=self.profile.execution_reference_frame_id,
            target_reference_frame_id=self.profile.reference_frame_id)
        goal = compute_spatial_goal(
            current_tcp_pose_xyz_quat_xyzw=pose_matrix_to_xyzquat(measured_public).tolist(),
            anchor_xyz_m=public_anchor[:3, 3].tolist(),
            offset_xyz_m=self.request.parameters["offset_xyz_m"],
            offset_frame=self.request.parameters["offset_frame"], orientation_mode="preserve")
        execution, resolution = graph.transform_pose_matrix(
            xyzquat_to_pose_matrix(goal.pose_xyz_quat_xyzw),
            source_reference_frame_id=self.profile.reference_frame_id,
            target_reference_frame_id=self.profile.execution_reference_frame_id)
        self._source, self._public_target = (anchor, artifact_id), tuple(goal.pose_xyz_quat_xyzw)
        digest = canonical_content_hash({
            "source_request_hash": anchor["source_request_hash"],
            "anchor_artifact_id": artifact_id,
            "component_id": self.component_id,
            "public_target": self._public_target,
            "profile_hash": self.profile.semantic_content_hash,
            "frame_resolution_hash": resolution.semantic_content_hash,
        }, domain="agenticrobot.real_observed_anchor.target.v2")
        return (MotionPrimitiveTargetResolution(
            target_pose=tuple(pose_matrix_to_xyzquat(execution)),
            evidence_refs=("surface-anchor-operation:" + anchor["source_operation_id"],
                "surface-anchor-artifact:" + artifact_id,
                "surface-anchor-depth:" + source["depth_content_hash"],
                "surface-anchor-target:" + digest)), SKILL_ID + "@2", digest)

    def _extend_arrival_result(self, result, measured_pose):
        measured, _ = self.contract_bundle.frame_graph.transform_pose_matrix(
            xyzquat_to_pose_matrix(measured_pose),
            source_reference_frame_id=self.profile.execution_reference_frame_id,
            target_reference_frame_id=self.profile.reference_frame_id)
        achieved = pose_matrix_to_xyzquat(measured).tolist()
        translation, rotation = pose_delta_metrics(
            measured, xyzquat_to_pose_matrix(self._public_target))
        anchor, artifact_id = self._source
        return {**result, "surface_anchor_report": {
            "source_operation_id": anchor["source_operation_id"],
            "source_request_hash": anchor["source_request_hash"],
            "anchor_artifact_id": artifact_id,
            "point_semantics": POINT_SEMANTICS,
            "reference_frame_id": self.profile.reference_frame_id,
            "requested_pose_xyz_quat_xyzw": list(self._public_target),
            "achieved_pose_xyz_quat_xyzw": achieved,
            "translation_error_m": translation,
            "rotation_error_rad": math.radians(rotation)}}

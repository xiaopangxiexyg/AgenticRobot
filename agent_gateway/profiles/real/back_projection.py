"""Real aligned-depth ports and versioned retained RGB-D projection adapters."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, replace
import logging
import time

from PIL import Image
import numpy as np

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
)
from agent_gateway.profiles.segmentation import LiveRgbSegmentationCapture
from agent_gateway.protocol.skill_io_contracts import (
    SkillArtifactSlotSpec,
    SkillIOContract,
)
from agent_gateway.protocol.skill_registration import SkillRegistrationTemplate
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import AgentSkillPayloadValidatorSet
from agent_gateway.skills.perception.back_projection import (
    MAX_BACK_PROJECT_PIXELS,
    RGBD_OBSERVATION_SLOT,
    back_project_definition,
    back_project_schema,
    capture_rgbd_definition,
    capture_rgbd_schema,
    validate_back_project_parameters,
    validate_back_project_result,
    validate_back_project_result_for_parameters,
)
from agent_gateway.skills.perception.grasp_observation import (
    validate_observation_parameters,
    validate_observation_result,
    validate_observation_result_for_parameters,
)
from agent_gateway.skills.perception.schemas import validate_no_task_evaluation_v1
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_ENCODING,
    GRASP_MULTIVIEW_MAX_BYTES,
    GRASP_MULTIVIEW_SCHEMA_ID,
)
from dataset_contracts.perception_artifacts import (
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
from robot_motion.relative_action import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)

from .grasp_observation import (
    build_grasp_observation_executor,
    read_retained_grasp_observation,
)
from .perception_deployment import (
    RealPerceptionCaptureRegistration,
    RealPerceptionDeploymentBinding,
)
from .perception_frames import RealPerceptionFrameBinding
from .realsense_perception import RealSenseAgentPerceptionProvider


def live_aligned_depth_capture(
    capture,
    *,
    frame_binding: RealPerceptionFrameBinding,
) -> LiveAlignedDepthCapture:
    """Project one deployment-owned aligned RGB-D cut into the Station frame."""

    source_from_camera = np.asarray(
        capture.source_from_camera, dtype=np.float64
    ).reshape(4, 4)
    source_pose = pose_matrix_to_xyzquat(
        source_from_camera,
        name="real_back_projection.source_from_camera",
    )
    reference_pose = frame_binding.camera_pose_reference_xyz_quat_xyzw(source_pose)
    reference_from_camera = xyzquat_to_pose_matrix(
        reference_pose,
        name="real_back_projection.reference_from_camera",
    )
    return LiveAlignedDepthCapture(
        source_id=capture.source_id,
        role=capture.role,
        captured_at=capture.acquired_at,
        station_revision=capture.station_revision,
        capture_group_id=capture.capture_group_id,
        depth_content_hash=capture.depth_content_hash,
        intrinsics=capture.intrinsics,
        reference_from_camera=reference_from_camera,
        depth_m=capture.depth_m,
    )


class RealSenseBackProjectionCapturePort:
    def __init__(
        self,
        *,
        perception_provider: RealSenseAgentPerceptionProvider,
        frame_binding: RealPerceptionFrameBinding,
    ) -> None:
        if not isinstance(perception_provider, RealSenseAgentPerceptionProvider):
            raise TypeError("perception_provider must be typed.")
        if not isinstance(frame_binding, RealPerceptionFrameBinding):
            raise TypeError("frame_binding must be typed.")
        self.perception_provider = perception_provider
        self.frame_binding = frame_binding

    def capture_aligned_depth(
        self, *, role: str, expected_station_revision: int
    ) -> LiveAlignedDepthCapture:
        capture = self.perception_provider.capture_aligned_rgbd(
            role=role,
            expected_station_revision=expected_station_revision,
        )
        return live_aligned_depth_capture(
            capture,
            frame_binding=self.frame_binding,
        )

    def capture_placement_rgbd(
        self, *, role: str, expected_station_revision: int
    ) -> CorrelatedPlacementCapture:
        capture = self.perception_provider.capture_aligned_rgbd(
            role=role,
            expected_station_revision=expected_station_revision,
        )
        return CorrelatedPlacementCapture(
            rgb=LiveRgbSegmentationCapture(
                source_id=capture.source_id,
                role=capture.role,
                captured_at=capture.acquired_at,
                station_revision=capture.station_revision,
                image=Image.fromarray(capture.rgb),
                rgb_content_hash=capture.rgb_content_hash,
            ),
            depth=live_aligned_depth_capture(
                capture,
                frame_binding=self.frame_binding,
            ),
        )


_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class RgbdGeometryRegistration:
    skill: SkillRegistrationTemplate
    binding: RealPerceptionDeploymentBinding
    payload_validator: AgentSkillPayloadValidatorSet
    availability_hash: str
    executor_id: str


def build_rgbd_geometry_registrations(
    capture: RealPerceptionCaptureRegistration,
    *, roles: tuple[str, ...] | None = None,
) -> tuple[RgbdGeometryRegistration, RgbdGeometryRegistration]:
    """Bind model-free geometry only to explicitly registered metric depth roles."""
    if capture.frame_binding is None:
        raise ValueError("RGB-D projection requires a typed public frame binding")
    roles = capture.profile.depth_roles if roles is None else roles
    if (not isinstance(roles, tuple) or not roles or len(roles) > 3
            or len(set(roles)) != len(roles) or not set(roles) <= set(capture.profile.rgb_roles)):
        raise ValueError("RGB-D roles must be distinct registered RGB roles with trusted geometry")
    profile_hash = canonical_content_hash({
        "source_binding_hash": capture.profile.source_binding_hash,
        "frame_binding_hash": capture.frame_binding.semantic_content_hash,
        "roles": roles, "capture_owner": "existing_aligned_rgbd_provider",
        "projection": "positive_exact_pixel_depth_no_hole_fill",
    }, domain="dexteleop.real_perception.rgbd_geometry.v1")
    slots = (
        SkillArtifactSlotSpec(slot_id=RGBD_OBSERVATION_SLOT, direction="output", min_count=1, max_count=1,
            allowed_media_types=("application/zip",), allowed_schema_ids=(GRASP_MULTIVIEW_SCHEMA_ID,),
            allowed_encodings=(GRASP_MULTIVIEW_ENCODING,), allowed_compressions=("zip",),
            max_artifact_bytes=GRASP_MULTIVIEW_MAX_BYTES, max_total_bytes=GRASP_MULTIVIEW_MAX_BYTES),
        SkillArtifactSlotSpec(slot_id="rgb_frames", direction="output", min_count=1, max_count=3,
            allowed_media_types=("image/jpeg",), allowed_schema_ids=(RGB_FRAME_SCHEMA_ID_V1,),
            allowed_encodings=(RGB_FRAME_ENCODING_V1,), allowed_compressions=(RGB_FRAME_COMPRESSION_V1,),
            max_artifact_bytes=4*1024*1024, max_total_bytes=12*1024*1024),
    )
    result = []
    for definition, schema, output_slots, validators in (
        (capture_rgbd_definition(), capture_rgbd_schema(roles=roles, profile_hash=profile_hash), slots,
         (lambda x: validate_observation_parameters(x, roles=roles),
          lambda x: validate_observation_result(x, roles=roles), validate_observation_result_for_parameters)),
        (back_project_definition(), back_project_schema(roles=roles, profile_hash=profile_hash), (),
         (lambda x: validate_back_project_parameters(x, roles=roles),
          lambda x: validate_back_project_result(x, roles=roles), validate_back_project_result_for_parameters)),
    ):
        suffix = definition.skill_id.split(".")[-1]
        executor_id = "real.passive." + suffix + ".v" + definition.skill_version
        io_contract = SkillIOContract(
            skill_id=definition.skill_id, skill_version=definition.skill_version, skill_definition_hash=definition.semantic_content_hash,
            parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
            task_evaluation_schema_hash=schema.task_evaluation_schema_hash, input_artifact_slots=(),
            output_artifact_slots=tuple(sorted(output_slots, key=lambda slot: slot.slot_id)))
        route = replace(capture.depth.routes[0], skill_id=definition.skill_id, skill_version=definition.skill_version,
            skill_definition_hash=definition.semantic_content_hash,
            route_id="real.realsense.perception." + suffix, route_version=definition.skill_version,
            parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
            task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
            allowed_parameter_profiles=schema.parameter_profiles, skill_io_contract_hash=io_contract.semantic_content_hash,
            max_operation_duration_ms=10_000)
        binding = replace(capture.rgb_binding, deployment_binding_id="real-rgbd-geometry:" + suffix + ":" + capture.rgb_binding.runtime_session_id,
                          route_hash=route.semantic_content_hash, profile_hash=profile_hash, executor_id=executor_id)
        validator = AgentSkillPayloadValidatorSet(
            validator_id="real.rgbd-geometry:" + suffix + ":" + profile_hash,
            skill_id=definition.skill_id, skill_version=definition.skill_version, route_id=route.route_id, route_version=route.route_version,
            parameter_schema_hash=schema.parameter_schema_hash, execution_result_schema_hash=schema.execution_result_schema_hash,
            task_evaluation_schema_hash=schema.task_evaluation_schema_hash,
            validate_parameters=validators[0], validate_execution_result=validators[1],
            validate_execution_result_for_parameters=validators[2], validate_task_evaluation=validate_no_task_evaluation_v1)
        result.append(RgbdGeometryRegistration(
            skill=SkillRegistrationTemplate(definition=definition, schema_entry=schema, io_contract=io_contract, routes=(route,)),
            binding=binding, payload_validator=validator, executor_id=executor_id,
            availability_hash=canonical_content_hash({"profile": profile_hash, "route": route.semantic_content_hash,
                "binding": binding.semantic_content_hash}, domain="dexteleop.real_perception.rgbd_geometry_availability.v1")))
    return tuple(result)


def project_retained_pixels(*, arrays, metadata, pixels_uv, frame_binding):
    """Vectorized pinhole projection; never fill holes or estimate an object pose."""
    if (metadata.get("frame_binding_hash") != frame_binding.semantic_content_hash
            or metadata.get("reference_frame_id") != frame_binding.reference_frame_id):
        raise ValueError("Retained RGB-D view has another frame binding")
    depth = arrays["depth_m"]
    intrinsics = metadata["camera_intrinsics"]
    if (intrinsics["width"], intrinsics["height"]) != (depth.shape[1], depth.shape[0]):
        raise ValueError("Retained RGB-D intrinsics dimensions differ")
    values = np.asarray([intrinsics[x] for x in ("fx", "fy", "cx", "cy")], dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values[:2] <= 0):
        raise ValueError("Retained RGB-D intrinsics are invalid")
    pixels = np.asarray(pixels_uv)
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or not 1 <= len(pixels) <= MAX_BACK_PROJECT_PIXELS
            or not np.issubdtype(pixels.dtype, np.integer)
            or np.any(pixels < 0) or np.any(pixels[:, 0] >= depth.shape[1])
            or np.any(pixels[:, 1] >= depth.shape[0])):
        raise ValueError("Back projection pixel is outside the retained view")
    z = depth[pixels[:, 1], pixels[:, 0]].astype(float)
    if not np.all(np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("Back projection pixel has invalid or missing depth; no hole filling")
    camera = np.column_stack(((pixels[:, 0]-values[2])*z/values[0],
                              (pixels[:, 1]-values[3])*z/values[1], z))
    transform, _ = frame_binding.frame_graph.transform_pose_matrix(
        np.asarray(metadata["T_source_from_camera"]).reshape(4, 4),
        source_reference_frame_id=frame_binding.source_frame_id,
        target_reference_frame_id=frame_binding.reference_frame_id)
    if not np.allclose(transform, np.asarray(metadata["T_reference_from_camera"]).reshape(4, 4), atol=1e-9, rtol=0):
        raise ValueError("Retained RGB-D reference transform disagrees with its frame graph")
    reference = camera @ transform[:3, :3].T + transform[:3, 3]
    return [{"pixel_uv": [int(x) for x in pixel], "depth_m": float(distance),
             "point_camera_m": point.tolist(), "point_reference_m": projected.tolist()}
            for pixel, distance, point, projected in zip(pixels, z, camera, reference, strict=True)]


def build_rgbd_geometry_executors(*, registrations, perception_provider, frame_binding,
                                  artifact_store, runtime_session_id, current_station_revision,
                                  max_retained_operations, clock, source_request_for_operation):
    if max_retained_operations < 2:
        raise ValueError("RGB-D capture and back projection require max_retained_operations >= 2")
    capture_registration, projection_registration = registrations
    capture_executor = build_grasp_observation_executor(
        registration=capture_registration, perception_provider=perception_provider, frame_binding=frame_binding,
        artifact_store=artifact_store, runtime_session_id=runtime_session_id,
        current_station_revision=current_station_revision, max_retained_operations=max_retained_operations,
        clock=clock, non_motion_read_window=nullcontext, require_stationary=False,
        observation_slot=RGBD_OBSERVATION_SLOT)

    def read(request, operation):
        began = time.perf_counter()
        parameters = request.parameters
        source_id = parameters["observation_operation_id"]
        source_request = source_request_for_operation(source_id)
        if source_request is None:
            raise ValueError("RGB-D observation operation is unknown or no longer retained")
        if source_request.client_id != request.client_id:
            raise ValueError("RGB-D observation does not belong to this client")
        completion = capture_executor.completion_for(source_id)
        if (completion is None or not completion.accepted
                or completion.operation.status != "completed"
                or completion.operation.execution_outcome != "completed"):
            raise ValueError("RGB-D observation is no longer retained")
        source_revision = completion.operation.execution_result["station_revision"]
        views, _ = read_retained_grasp_observation(
            executor=capture_executor, operation_id=source_id, artifact_store=artifact_store,
            runtime_session_id=runtime_session_id, expected_station_revision=source_revision,
            frame_binding_hash=frame_binding.semantic_content_hash, observation_slot=RGBD_OBSERVATION_SLOT,
            decode_arrays=True)
        decode_done = time.perf_counter()
        selected = [view for view in views if view[1].get("role") == parameters["role"]]
        if len(selected) != 1:
            raise ValueError("Selected RGB-D view was excluded or is not retained")
        arrays, metadata = selected[0]
        if metadata.get("station_revision") != source_revision:
            raise ValueError("RGB-D view differs from its observation revision")
        points = project_retained_pixels(arrays=arrays, metadata=metadata,
                                         pixels_uv=parameters["pixels_uv"], frame_binding=frame_binding)
        result = {
            "observation_operation_id": source_id, "role": parameters["role"],
            **{key: metadata[key] for key in ("source_id", "capture_group_id", "acquired_at", "station_revision",
                                             "reference_frame_id", "rgb_content_hash", "depth_content_hash")},
            "reference_frame_spec_hash": frame_binding.reference_frame_spec_hash,
            "frame_graph_hash": frame_binding.frame_graph.semantic_content_hash,
            "points": points, "point_semantics": "observed_surface_not_object_center_or_pose",
        }
        _LOG.info("Back projection timing: operation_id=%s retained_decode_ms=%.3f projection_ms=%.3f pixels=%s",
                  operation.operation_id, (decode_done-began)*1000, (time.perf_counter()-decode_done)*1000, len(points))
        return PassiveReadResult(execution_result=result,
            evidence_refs=("rgbd-back-projection:" + operation.operation_id, "rgbd-observation:" + source_id),
            reason_code="passive_rgbd_back_projection_completed")

    projection_executor = PassiveReadSkillExecutor(
        port_id="agent-gateway:back-projection:" + runtime_session_id,
        executor_id=projection_registration.executor_id, read=read,
        current_station_revision=current_station_revision, clock=clock,
        max_retained_operations=max_retained_operations)
    return capture_executor, projection_executor


__all__ = ["RealSenseBackProjectionCapturePort", "live_aligned_depth_capture",
           "RgbdGeometryRegistration", "build_rgbd_geometry_registrations",
           "build_rgbd_geometry_executors", "project_retained_pixels"]

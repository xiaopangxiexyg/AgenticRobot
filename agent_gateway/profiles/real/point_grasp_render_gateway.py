"""Render one retained Station point-grasp proposal into one live camera view."""

from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.contracts import AgentSkillOperation
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.service import AgentGatewayOperationOutcome
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
)
from deployments.perception.dmtac_grasp_overlay import (
    DmtacGraspOverlayCandidate,
    render_dmtac_grasp_overlays,
)

from .perception_frames import RealPerceptionFrameBinding
from .realsense_perception import RealSenseAgentPerceptionProvider
from .station_point_grasp_deployment import StationBoundPointGraspRegistration
from .station_point_grasp_render_deployment import (
    REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1,
    StationBoundPointGraspRenderRegistration,
)


@dataclass(frozen=True)
class PreparedPointGraspCandidateRender:
    execution_result: dict[str, Any]
    annotated_jpeg: bytes


_DEBUG_TARGET_AXIS_LENGTH_M = 0.05
_DEBUG_TARGET_COLORS = {
    "target": (0, 255, 255),
    "x": (255, 64, 64),
    "y": (64, 255, 64),
    "z": (64, 128, 255),
    "text": (255, 255, 255),
}


def _project_reference_point(
    point_reference_m: np.ndarray,
    *,
    camera_from_reference: np.ndarray,
    intrinsics: Mapping[str, Any],
) -> tuple[float, float] | None:
    point_h = np.append(np.asarray(point_reference_m, dtype=np.float64), 1.0)
    point_camera = camera_from_reference @ point_h
    depth = float(point_camera[2])
    if not np.isfinite(depth) or depth <= 1e-6:
        return None
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    u = fx * float(point_camera[0]) / depth + cx
    v = fy * float(point_camera[1]) / depth + cy
    if not np.isfinite((u, v)).all():
        return None
    return u, v


def _render_target_coordinate_debug_overlay(
    image: Image.Image,
    *,
    target_point_reference_m: Any,
    reference_from_camera: Any,
    intrinsics: Mapping[str, Any],
) -> Image.Image:
    """Draw the depth-localized target with canonical Station axes.

    This is deliberately a position diagnostic.  The axes use the Station
    orientation because point-grasp localization does not estimate an object
    orientation.
    """

    target = np.asarray(target_point_reference_m, dtype=np.float64)
    if target.shape != (3,) or not np.all(np.isfinite(target)):
        raise ValueError("target_point_reference_m must contain three finite values.")
    reference_from_camera_matrix = np.asarray(reference_from_camera, dtype=np.float64)
    if reference_from_camera_matrix.shape != (4, 4) or not np.all(
        np.isfinite(reference_from_camera_matrix)
    ):
        raise ValueError("reference_from_camera must be a finite 4x4 matrix.")
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    if image.size != (width, height):
        raise ValueError("image dimensions differ from camera intrinsics.")
    camera_from_reference = np.linalg.inv(reference_from_camera_matrix)
    origin = _project_reference_point(
        target,
        camera_from_reference=camera_from_reference,
        intrinsics=intrinsics,
    )
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)
    label = "TARGET Station XYZ " + " ".join(f"{value:+.3f}" for value in target)
    draw.rectangle((0, 0, min(width - 1, 330), min(height - 1, 13)), fill=(0, 0, 0))
    draw.text((2, 2), label, fill=_DEBUG_TARGET_COLORS["text"])
    if origin is None:
        draw.text((2, min(height - 1, 15)), "TARGET behind camera", fill=(255, 191, 0))
        return rendered
    u, v = origin
    if not (0.0 <= u < width and 0.0 <= v < height):
        draw.text((2, min(height - 1, 15)), "TARGET outside image", fill=(255, 191, 0))
        return rendered
    center = (round(u), round(v))
    radius = 6
    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline=_DEBUG_TARGET_COLORS["target"],
        width=2,
    )
    draw.line(
        (center[0] - 9, center[1], center[0] + 9, center[1]),
        fill=_DEBUG_TARGET_COLORS["target"],
        width=2,
    )
    draw.line(
        (center[0], center[1] - 9, center[0], center[1] + 9),
        fill=_DEBUG_TARGET_COLORS["target"],
        width=2,
    )
    for axis_index, axis_name in enumerate(("x", "y", "z")):
        endpoint_reference = target.copy()
        endpoint_reference[axis_index] += _DEBUG_TARGET_AXIS_LENGTH_M
        endpoint = _project_reference_point(
            endpoint_reference,
            camera_from_reference=camera_from_reference,
            intrinsics=intrinsics,
        )
        if endpoint is None:
            continue
        endpoint_u, endpoint_v = endpoint
        clipped_endpoint = (
            round(min(max(endpoint_u, 0.0), width - 1.0)),
            round(min(max(endpoint_v, 0.0), height - 1.0)),
        )
        color = _DEBUG_TARGET_COLORS[axis_name]
        draw.line((center, clipped_endpoint), fill=color, width=3)
        if 0.0 <= endpoint_u < width and 0.0 <= endpoint_v < height:
            draw.text(clipped_endpoint, axis_name.upper(), fill=color)
    return rendered


class StationBoundPointGraspRenderProvider:
    """Resolve a retained proposal and project it with deployment-owned geometry."""

    def __init__(
        self,
        *,
        registration: StationBoundPointGraspRenderRegistration,
        point_grasp_registration: StationBoundPointGraspRegistration,
        point_grasp_executor: PassiveReadSkillExecutor,
        perception_provider: RealSenseAgentPerceptionProvider,
        frame_binding: RealPerceptionFrameBinding,
        runtime_session_id: str,
        current_station_revision: Callable[[], int],
        solid_geometry: Any = None,
    ) -> None:
        if not isinstance(registration, StationBoundPointGraspRenderRegistration):
            raise TypeError("registration must be typed.")
        if not isinstance(point_grasp_registration, StationBoundPointGraspRegistration):
            raise TypeError("point_grasp_registration must be typed.")
        if not isinstance(point_grasp_executor, PassiveReadSkillExecutor):
            raise TypeError("point_grasp_executor must be typed.")
        if not isinstance(perception_provider, RealSenseAgentPerceptionProvider):
            raise TypeError("perception_provider must be typed.")
        if not isinstance(frame_binding, RealPerceptionFrameBinding):
            raise TypeError("frame_binding must be typed.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        session_id = str(runtime_session_id).strip()
        if not session_id or session_id != registration.binding.runtime_session_id:
            raise ValueError("Render runtime session differs from registration.")
        source_route_hash = point_grasp_registration.skill.routes[
            0
        ].semantic_content_hash
        profile = registration.profile
        if (
            profile.point_grasp_route_hash != source_route_hash
            or profile.point_grasp_profile_hash
            != point_grasp_registration.profile.semantic_content_hash
            or profile.point_grasp_deployment_binding_hash
            != point_grasp_registration.binding.semantic_content_hash
        ):
            raise ValueError("Render Route is bound to another point-grasp contract.")
        if (
            point_grasp_executor.executor_id
            != point_grasp_registration.binding.executor_id
        ):
            raise ValueError("Render provider received another point-grasp executor.")
        if (
            perception_provider.source_binding.semantic_content_hash
            != profile.source_binding_hash
        ):
            raise ValueError("Render camera source binding differs from registration.")
        if (
            frame_binding.semantic_content_hash != profile.frame_binding_hash
            or frame_binding.frame_graph.semantic_content_hash
            != profile.frame_graph_hash
        ):
            raise ValueError("Render frame binding differs from registration.")
        self.registration = registration
        self.point_grasp_registration = point_grasp_registration
        self.point_grasp_executor = point_grasp_executor
        self.perception_provider = perception_provider
        self.frame_binding = frame_binding
        self.runtime_session_id = session_id
        self._current_station_revision = current_station_revision
        self.solid_geometry = solid_geometry

    def _retained_candidate(
        self,
        *,
        operation_id: str,
        ordinal: int,
        expected_station_revision: int,
    ) -> tuple[AgentSkillOperation, Mapping[str, Any], Mapping[str, Any]]:
        completion = self.point_grasp_executor.completion_for(operation_id)
        if completion is None:
            raise ValueError("Point-grasp operation is not retained by this session.")
        if not isinstance(completion, AgentGatewayOperationOutcome):
            raise TypeError("Retained point-grasp completion has an invalid type.")
        operation = completion.operation
        if not completion.accepted or not isinstance(operation, AgentSkillOperation):
            raise ValueError("Retained point-grasp completion was not accepted.")
        if operation.runtime_session_id != self.runtime_session_id:
            raise ValueError("Point-grasp operation belongs to another session.")
        if (
            operation.status != "completed"
            or operation.execution_outcome != "completed"
        ):
            raise ValueError("Point-grasp operation did not complete successfully.")
        result = operation.execution_result
        if result is None:
            raise ValueError("Point-grasp operation has no retained result.")
        if not isinstance(result, Mapping):
            raise TypeError("Retained point-grasp result has an invalid type.")
        self.point_grasp_registration.payload_validator.validate_execution_result(
            result
        )
        if result["station_revision"] != expected_station_revision:
            raise ValueError("Point-grasp result belongs to another Station revision.")
        candidates = result["candidates"]
        if not isinstance(candidates, (list, tuple)) or ordinal >= len(candidates):
            raise ValueError("candidate_ordinal is absent from the retained result.")
        candidate = candidates[ordinal]
        if not isinstance(candidate, Mapping) or candidate.get("ordinal") != ordinal:
            raise ValueError("Retained point-grasp candidate ordering is inconsistent.")
        return operation, result, candidate

    def render_request(self, request: Any) -> PreparedPointGraspCandidateRender:
        if request.input_artifact_bindings:
            raise ValueError(
                "Point-grasp candidate rendering accepts no input Artifacts."
            )
        expected_revision = request.expected_station_revision
        operation_id = str(request.parameters["point_grasp_operation_id"])
        ordinal = int(request.parameters["candidate_ordinal"])
        view_role = str(request.parameters["view_role"])
        operation, point_grasp_result, candidate = self._retained_candidate(
            operation_id=operation_id,
            ordinal=ordinal,
            expected_station_revision=expected_revision,
        )
        capture = self.perception_provider.capture_rgb_geometry(
            role=view_role,
            expected_station_revision=expected_revision,
        )
        source_from_camera = np.asarray(
            capture.source_from_camera, dtype=np.float64
        ).reshape(4, 4)
        reference_from_camera, resolution = (
            self.frame_binding.frame_graph.transform_pose_matrix(
                source_from_camera,
                source_reference_frame_id=self.frame_binding.source_frame_id,
                target_reference_frame_id=self.frame_binding.reference_frame_id,
            )
        )
        if resolution.frame_graph_hash != self.registration.profile.frame_graph_hash:
            raise RuntimeError("Render frame resolution identity changed.")
        reference_from_tcp = np.asarray(
            candidate["T_reference_from_tool_tcp"], dtype=np.float64
        ).reshape(4, 4)
        overlay_candidate = DmtacGraspOverlayCandidate.from_pose(
            ordinal=ordinal,
            score=float(candidate["score"]),
            gripper_width_m=float(candidate["gripper_width_m"]),
            reference_from_tcp=reference_from_tcp,
            selected=True,
        )
        image, projections = render_dmtac_grasp_overlays(
            Image.fromarray(capture.rgb, mode="RGB"),
            candidates=(overlay_candidate,),
            reference_from_camera=reference_from_camera,
            intrinsics=capture.intrinsics,
            solid_geometry=self.solid_geometry,
        )
        if self.registration.profile.debug_target_coordinate_overlay:
            image = _render_target_coordinate_debug_overlay(
                image,
                target_point_reference_m=point_grasp_result["target_point_reference_m"],
                reference_from_camera=reference_from_camera,
                intrinsics=capture.intrinsics,
            )
        if len(projections) != 1 or projections[0].ordinal != ordinal:
            raise RuntimeError(
                "Renderer did not return exactly the requested candidate."
            )
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=90, optimize=True)
        annotated = stream.getvalue()
        if len(annotated) > self.registration.profile.max_annotated_bytes:
            raise RuntimeError("Rendered point-grasp image exceeds its bound.")
        if self._current_station_revision() != expected_revision:
            raise RuntimeError("Station revision changed during candidate rendering.")
        projection = projections[0]
        point_grasp_result_hash = canonical_content_hash(
            point_grasp_result,
            domain="dexteleop.agent_gateway.passive_read_result.v1",
        )
        candidate_hash = canonical_content_hash(
            candidate,
            domain="dexteleop.agent_skill.station_point_grasp_candidate.v2",
        )
        media_content_hash = hashlib.sha256(annotated).hexdigest()
        result = {
            "point_grasp_operation_id": operation.operation_id,
            "point_grasp_request_hash": operation.request_hash,
            "point_grasp_result_hash": point_grasp_result_hash,
            "point_grasp_candidate_hash": candidate_hash,
            "source_capture_group_id": point_grasp_result["capture_group_id"],
            "source_station_revision": point_grasp_result["station_revision"],
            "candidate_ordinal": ordinal,
            "view_role": capture.role,
            "view_source_id": capture.source_id,
            "view_acquired_at": capture.acquired_at,
            "view_station_revision": capture.station_revision,
            "view_source_generation": capture.source_generation,
            "view_frame_index": capture.frame_index,
            "view_capture_group_id": capture.capture_group_id,
            "view_rgb_content_hash": capture.rgb_content_hash,
            "view_geometry_binding_hash": capture.geometry_binding_hash,
            "view_geometry_cut_hash": capture.geometry_cut_hash,
            "view_pose_measured_at": capture.pose_measured_at,
            "view_pose_available_at": capture.pose_available_at,
            "view_pose_evidence_refs": list(capture.pose_evidence_refs),
            "source_binding_hash": capture.source_binding_hash,
            "frame_binding_hash": self.frame_binding.semantic_content_hash,
            "frame_graph_hash": self.frame_binding.frame_graph.semantic_content_hash,
            "source_frame_id": self.frame_binding.source_frame_id,
            "source_frame_spec_hash": self.frame_binding.source_frame_spec_hash,
            "reference_frame_id": self.frame_binding.reference_frame_id,
            "reference_frame_spec_hash": self.frame_binding.reference_frame_spec_hash,
            "camera_intrinsics": dict(capture.intrinsics),
            "T_reference_from_tool_tcp": [
                float(value) for value in reference_from_tcp.reshape(-1)
            ],
            "T_reference_from_camera": [
                float(value) for value in reference_from_camera.reshape(-1)
            ],
            "T_camera_from_tool_tcp": list(projection.camera_from_tcp),
            "projection_visibility": projection.visibility,
            "projection_total_segment_count": projection.total_segment_count,
            "projection_front_segment_count": projection.front_segment_count,
            "projection_visible_segment_count": projection.visible_segment_count,
            "projection_clipped_segment_count": projection.clipped_segment_count,
            "tcp_pixel_uv": (
                None
                if projection.tcp_pixel_uv is None
                else list(projection.tcp_pixel_uv)
            ),
            "depth_occlusion_tested": False,
            "downstream_control_effects": "forbidden",
            "frames": [
                {
                    "ordinal": 0,
                    "role": capture.role,
                    "width": capture.width,
                    "height": capture.height,
                    "media_type": LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
                    "media_content_hash": media_content_hash,
                }
            ],
        }
        return PreparedPointGraspCandidateRender(result, annotated)


def build_station_bound_point_grasp_render_executor(
    *,
    registration: StationBoundPointGraspRenderRegistration,
    provider: StationBoundPointGraspRenderProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    """Bind single-candidate rendering without motion or selection authority."""

    def read(request: Any, operation: Any) -> PassiveReadResult:
        prepared = provider.render_request(request)
        expected_revision = request.expected_station_revision
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed before render Artifact commit."
            )
        artifact = artifact_ref_for_bytes(
            prepared.annotated_jpeg,
            media_type=LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
            schema_id=POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
            encoding=LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
            compression=LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
        )
        if (
            prepared.execution_result["frames"][0]["media_content_hash"]
            != artifact.sha256
        ):
            raise RuntimeError("Rendered frame hash differs from its Artifact.")
        committed = commit_artifact_bytes(
            store=artifact_store,
            data=prepared.annotated_jpeg,
            artifact=artifact,
            producer_id="agent_gateway.provider.station_point_grasp_render",
            runtime_session_id=runtime_session_id,
            idempotency_key="station-point-grasp-render-output:"
            + operation.operation_id,
            now=float(clock()),
        )
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed during render Artifact commit."
            )
        identity = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "result": prepared.execution_result,
                "artifact": committed.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.station_point_grasp_render_receipt.v1",
        )
        evidence = "station-point-grasp-render:" + identity
        return PassiveReadResult(
            execution_result=prepared.execution_result,
            evidence_refs=(evidence,),
            output_artifact_bindings=(
                AgentArtifactBinding(
                    slot_id=REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1,
                    ordinal=0,
                    artifact=committed,
                    production_receipt_ref=evidence,
                ),
            ),
            reason_code="passive_station_point_grasp_candidate_render_completed",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:station-point-grasp-render:" + runtime_session_id,
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "PreparedPointGraspCandidateRender",
    "StationBoundPointGraspRenderProvider",
    "build_station_bound_point_grasp_render_executor",
]

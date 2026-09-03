"""Aligned RGB-D input provider and passive point-grasp Gateway executor."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
    LOCALIZATION_INPUT_PNG_ENCODING_V1,
    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
    POINT_GRASP_DEPTH_COMPRESSION_V1,
    POINT_GRASP_DEPTH_ENCODING_V1,
    POINT_GRASP_DEPTH_MEDIA_TYPE_V1,
    POINT_GRASP_DEPTH_SCHEMA_ID_V1,
    POINT_GRASP_RGB_SCHEMA_ID_V1,
)

from .localization_gateway import (
    decode_bounded_image,
    read_bounded_regular_file,
    safe_local_path,
)
from .point_grasp_deployment import (
    REAL_POINT_GRASP_DEPTH_SLOT_ID_V1,
    REAL_POINT_GRASP_OUTPUT_SLOT_ID_V1,
    REAL_POINT_GRASP_RGB_SLOT_ID_V1,
    RealPointGraspRegistration,
)


@dataclass(frozen=True)
class PreparedPointGrasp:
    execution_result: dict[str, Any]
    annotated_jpeg: bytes


def _decode_depth(data: bytes, *, width: int, height: int) -> np.ndarray:
    try:
        value = np.load(io.BytesIO(data), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError("Point-grasp depth input is not a valid NPY array.") from exc
    if (
        value.dtype != np.float32
        or value.shape != (height, width)
        or not value.flags.c_contiguous
    ):
        raise ValueError(
            "Point-grasp depth must be C-order float32 HxW aligned meters."
        )
    if not np.all(np.isfinite(value)) or np.any(value < 0) or np.any(value > 10.0):
        raise ValueError("Point-grasp depth contains invalid meter values.")
    return value


def _annotate(
    image: Image.Image,
    pixel_uv: tuple[int, int],
    prepared: Any,
    intrinsics: dict[str, Any],
) -> bytes:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    u, v = pixel_uv
    draw.ellipse((u - 5, v - 5, u + 5, v + 5), outline=(255, 255, 0), width=3)
    x0, y0, x1, y1 = prepared.target_mask_bbox_xyxy
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(64, 220, 255), width=2)
    for ordinal, candidate in enumerate(prepared.candidates):
        matrix = np.asarray(candidate.camera_from_tcp).reshape(4, 4)
        point = matrix[:3, 3]
        if point[2] <= 0:
            continue
        candidate_u = round(
            float(intrinsics["fx"]) * point[0] / point[2] + float(intrinsics["cx"])
        )
        candidate_v = round(
            float(intrinsics["fy"]) * point[1] / point[2] + float(intrinsics["cy"])
        )
        if not (0 <= candidate_u < output.width and 0 <= candidate_v < output.height):
            continue
        color = (255, 32, 32) if ordinal == 0 else (255, 160, 32)
        draw.line(
            (candidate_u - 6, candidate_v, candidate_u + 6, candidate_v),
            fill=color,
            width=2,
        )
        draw.line(
            (candidate_u, candidate_v - 6, candidate_u, candidate_v + 6),
            fill=color,
            width=2,
        )
        draw.text(
            (candidate_u + 7, candidate_v + 4),
            f"{ordinal}: {candidate.score:.3f} w={candidate.gripper_width_m:.3f}m",
            fill=color,
        )
    stream = io.BytesIO()
    output.save(stream, format="JPEG", quality=90, optimize=True)
    return stream.getvalue()


class PointGraspProvider:
    def __init__(
        self,
        *,
        planner: object,
        registration: RealPointGraspRegistration,
        allowed_local_image_roots: tuple[str | Path, ...] = (),
    ) -> None:
        if not isinstance(registration, RealPointGraspRegistration):
            raise TypeError("registration must be typed.")
        for name in ("start", "close", "plan"):
            if not callable(getattr(planner, name, None)):
                raise TypeError(f"planner must expose {name}().")
        if (
            getattr(planner, "semantic_content_hash", None)
            != registration.profile.inference_implementation_hash
        ):
            raise ValueError(
                "Point-grasp planner identity differs from its registration."
            )
        roots = tuple(
            sorted(
                (
                    Path(value).expanduser().resolve(strict=True)
                    for value in allowed_local_image_roots
                ),
                key=str,
            )
        )
        if (
            any(not root.is_dir() for root in roots)
            or bool(roots) != registration.profile.local_path_enabled
        ):
            raise ValueError(
                "Point-grasp local roots disagree with the deployment profile."
            )
        self.planner = planner
        self.registration = registration
        self.allowed_local_image_roots = roots
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    def start(self) -> None:
        self.planner.start()
        self._started = True

    def close(self) -> None:
        try:
            self.planner.close()
        finally:
            self._started = False

    def _inputs(
        self, request: Any, store: RecoverableAgentArtifactStore
    ) -> tuple[bytes, bytes, str | None]:
        profile = self.registration.profile
        if request.parameters["input_source"] == "local_path":
            if request.input_artifact_bindings:
                raise ValueError("Local point-grasp input cannot bind Artifacts.")
            rgb_path = safe_local_path(
                request.parameters["local_rgb_path"],
                self.allowed_local_image_roots,
                field_name="local_rgb_path",
            )
            depth_path = safe_local_path(
                request.parameters["local_depth_path"],
                self.allowed_local_image_roots,
                field_name="local_depth_path",
            )
            return (
                read_bounded_regular_file(rgb_path, maximum=profile.max_rgb_bytes),
                read_bounded_regular_file(depth_path, maximum=profile.max_depth_bytes),
                None,
            )
        bindings = {
            binding.slot_id: binding for binding in request.input_artifact_bindings
        }
        if set(bindings) != {
            REAL_POINT_GRASP_RGB_SLOT_ID_V1,
            REAL_POINT_GRASP_DEPTH_SLOT_ID_V1,
        }:
            raise ValueError(
                "Artifact point-grasp input requires one RGB and one depth binding."
            )
        rgb = bindings[REAL_POINT_GRASP_RGB_SLOT_ID_V1].artifact
        depth = bindings[REAL_POINT_GRASP_DEPTH_SLOT_ID_V1].artifact
        if (
            rgb.schema_id != POINT_GRASP_RGB_SCHEMA_ID_V1
            or rgb.compression != LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1
            or (rgb.media_type, rgb.encoding)
            not in {
                (
                    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
                    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
                ),
                (
                    LOCALIZATION_INPUT_PNG_MEDIA_TYPE_V1,
                    LOCALIZATION_INPUT_PNG_ENCODING_V1,
                ),
            }
        ):
            raise ValueError("Point-grasp RGB Artifact identity is invalid.")
        if (
            depth.media_type != POINT_GRASP_DEPTH_MEDIA_TYPE_V1
            or depth.schema_id != POINT_GRASP_DEPTH_SCHEMA_ID_V1
            or depth.encoding != POINT_GRASP_DEPTH_ENCODING_V1
            or depth.compression != POINT_GRASP_DEPTH_COMPRESSION_V1
        ):
            raise ValueError("Point-grasp depth Artifact identity is invalid.")
        return (
            read_bounded_regular_file(
                store.managed_read_path(rgb.artifact_id), maximum=profile.max_rgb_bytes
            ),
            read_bounded_regular_file(
                store.managed_read_path(depth.artifact_id),
                maximum=profile.max_depth_bytes,
            ),
            rgb.media_type,
        )

    def plan_request(
        self, request: Any, *, artifact_store: RecoverableAgentArtifactStore
    ) -> PreparedPointGrasp:
        rgb_data, depth_data, media_type = self._inputs(request, artifact_store)
        intrinsics = request.parameters["camera_intrinsics"]
        image = decode_bounded_image(
            rgb_data, expected_media_type=media_type, maximum_pixels=16_777_216
        )
        if image.size != (intrinsics["width"], intrinsics["height"]):
            raise ValueError("RGB dimensions differ from camera_intrinsics.")
        depth = _decode_depth(depth_data, width=image.width, height=image.height)
        pixel_uv = tuple(request.parameters["target_pixel_uv"])
        planned = self.planner.plan(
            image,
            depth,
            pixel_uv=pixel_uv,
            intrinsics=intrinsics,
            base_from_camera=np.asarray(
                request.parameters["T_base_from_camera"], dtype=float
            ).reshape(4, 4),
            max_candidates=request.parameters["max_candidates"],
        )
        annotated = _annotate(image, pixel_uv, planned, intrinsics)
        if len(annotated) > self.registration.profile.max_annotated_bytes:
            raise RuntimeError("Annotated point-grasp image exceeds its bound.")
        candidates = []
        for ordinal, candidate in enumerate(planned.candidates):
            camera = list(candidate.camera_from_tcp)
            base = list(candidate.base_from_tcp)
            candidates.append(
                {
                    "ordinal": ordinal,
                    "score": candidate.score,
                    "model_score": candidate.model_score,
                    "click_distance_m": candidate.click_distance_m,
                    "gripper_width_m": candidate.gripper_width_m,
                    "T_camera_from_tcp": camera,
                    "T_base_from_tcp": base,
                    "tcp_xyz_camera_m": [camera[3], camera[7], camera[11]],
                    "tcp_xyz_base_m": [base[3], base[7], base[11]],
                    "visible_collision_free": candidate.visible_collision_free,
                }
            )
        result = {
            "input_source": request.parameters["input_source"],
            "rgb_media_content_hash": hashlib.sha256(rgb_data).hexdigest(),
            "depth_media_content_hash": hashlib.sha256(depth_data).hexdigest(),
            "target_pixel_uv": list(pixel_uv),
            "target_point_camera_m": list(planned.target_point_camera_m),
            "target_point_base_m": list(planned.target_point_base_m),
            "target_mask_bbox_xyxy": list(planned.target_mask_bbox_xyxy),
            "camera_frame_id": request.parameters["camera_frame_id"],
            "base_frame_id": request.parameters["base_frame_id"],
            "pose_semantics": "homogeneous_row_major_parent_from_tool_tcp_m",
            "gripper_profile_id": "DMTAC_M",
            "backend_id": self.registration.profile.backend_id,
            "candidate_count": len(candidates),
            "selected_ordinal": 0,
            "candidates": candidates,
            "model_profile_id": self.registration.profile.model_profile_id,
            "model_profile_hash": self.registration.profile.model_profile_hash,
            "annotated_image_media_content_hash": hashlib.sha256(annotated).hexdigest(),
        }
        return PreparedPointGrasp(result, annotated)


def build_real_point_grasp_executor(
    *,
    registration: RealPointGraspRegistration,
    provider: PointGraspProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Any,
    max_retained_operations: int,
    clock: Any = time.time,
) -> PassiveReadSkillExecutor:
    def read(request: Any, operation: Any) -> PassiveReadResult:
        prepared = provider.plan_request(request, artifact_store=artifact_store)
        artifact = artifact_ref_for_bytes(
            prepared.annotated_jpeg,
            media_type=LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
            schema_id=POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
            encoding=LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
            compression=LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
        )
        committed = commit_artifact_bytes(
            store=artifact_store,
            data=prepared.annotated_jpeg,
            artifact=artifact,
            producer_id="agent_gateway.provider.real_point_grasp",
            runtime_session_id=runtime_session_id,
            idempotency_key="point-grasp-output:" + operation.operation_id,
            now=float(clock()),
        )
        identity = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "result": prepared.execution_result,
                "artifact": committed.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.real_point_grasp_receipt.v1",
        )
        evidence = "real-point-grasp:" + identity
        return PassiveReadResult(
            execution_result=prepared.execution_result,
            evidence_refs=(evidence,),
            output_artifact_bindings=(
                AgentArtifactBinding(
                    slot_id=REAL_POINT_GRASP_OUTPUT_SLOT_ID_V1,
                    ordinal=0,
                    artifact=committed,
                    production_receipt_ref=evidence,
                ),
            ),
            reason_code="passive_point_grasp_planning_completed",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:real-passive-point-grasp:" + runtime_session_id,
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "PointGraspProvider",
    "PreparedPointGrasp",
    "build_real_point_grasp_executor",
]

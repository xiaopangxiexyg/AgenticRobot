"""Aligned RGB-D input provider and passive point-grasp Gateway executor."""

from __future__ import annotations

import hashlib
import io
import logging
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

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
from dataset_contracts.grasp_rgbd_snapshot import (
    encode_grasp_rgbd_snapshot, GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
    GRASP_RGBD_SNAPSHOT_MEDIA_TYPE, GRASP_RGBD_SNAPSHOT_ENCODING,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPort
from robot_motion.grasp_visible_scene import (
    ParallelJawSolidGeometry,
    VisibleGraspScene,
)
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilterResult,
)
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
from deployments.perception.dmtac_grasp_overlay import (
    DmtacGraspOverlayCandidate,
    render_dmtac_grasp_overlays,
)

from .localization_gateway import (
    decode_bounded_image,
    read_bounded_regular_file,
    safe_local_path,
)
from .perception_frames import RealPerceptionFrameBinding
from .point_grasp_deployment import (
    REAL_POINT_GRASP_DEPTH_SLOT_ID_V1,
    REAL_POINT_GRASP_OUTPUT_SLOT_ID_V1,
    REAL_POINT_GRASP_RGB_SLOT_ID_V1,
    RealPointGraspRegistration,
)
from .realsense_perception import RealSenseAgentPerceptionProvider
from .station_point_grasp_deployment import (
    REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
    REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID,
    StationBoundPointGraspRegistration,
)


@dataclass(frozen=True)
class PreparedPointGrasp:
    execution_result: dict[str, Any]
    annotated_jpeg: bytes
    rgbd_snapshot: bytes | None = None


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
    *,
    solid_geometry: Any = None,
) -> bytes:
    # The result carries every ranked proposal; the preview stays legible by
    # rendering only selected ordinal 0.
    candidates = tuple(
        DmtacGraspOverlayCandidate.from_pose(
            ordinal=ordinal,
            score=candidate.score,
            gripper_width_m=candidate.gripper_width_m,
            reference_from_tcp=candidate.camera_from_tcp,
            selected=ordinal == 0,
        )
        for ordinal, candidate in enumerate(prepared.candidates[:1])
    )
    output, _projections = render_dmtac_grasp_overlays(
        image,
        candidates=candidates,
        reference_from_camera=np.eye(4),
        intrinsics=intrinsics,
        target_pixel_uv=pixel_uv,
        target_bbox_xyxy=tuple(prepared.target_mask_bbox_xyxy),
        solid_geometry=solid_geometry,
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
        annotated = _annotate(
            image,
            pixel_uv,
            planned,
            intrinsics,
            solid_geometry=getattr(self.planner, "solid_geometry", None),
        )
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


class StationBoundPointGraspProvider:
    """Plan from one trusted live RGB-D cut and return proposals only."""

    def __init__(
        self,
        *,
        planner: object,
        registration: StationBoundPointGraspRegistration,
        perception_provider: RealSenseAgentPerceptionProvider,
        frame_binding: RealPerceptionFrameBinding,
        current_station_revision: Callable[[], int],
        arm_reachability_evaluator: (
            Callable[
                [tuple[np.ndarray, ...]],
                tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]],
            ]
            | None
        ) = None,
        motion_port: RealAgentMotionPort | None = None,
        captured_robot_observation_filter: Callable[..., object] | None = None,
    ) -> None:
        if not isinstance(registration, StationBoundPointGraspRegistration):
            raise TypeError("registration must be typed.")
        if not isinstance(perception_provider, RealSenseAgentPerceptionProvider):
            raise TypeError("perception_provider must be typed.")
        if not isinstance(frame_binding, RealPerceptionFrameBinding):
            raise TypeError("frame_binding must be typed.")
        if not callable(current_station_revision):
            raise TypeError("current_station_revision must be callable.")
        reachability_profile = registration.profile.arm_reachability_profile
        if (reachability_profile is None) != (arm_reachability_evaluator is None):
            raise ValueError(
                "Arm reachability registration and evaluator must be supplied together."
            )
        if arm_reachability_evaluator is not None and not callable(
            arm_reachability_evaluator
        ):
            raise TypeError("arm_reachability_evaluator must be callable or None.")
        for name in ("start", "close", "plan"):
            if not callable(getattr(planner, name, None)):
                raise TypeError(f"planner must expose {name}().")
        if (
            getattr(planner, "semantic_content_hash", None)
            != registration.profile.inference_implementation_hash
        ):
            raise ValueError("Point-grasp planner identity differs from registration.")
        if (
            perception_provider.source_binding.semantic_content_hash
            != registration.profile.source_binding_hash
        ):
            raise ValueError("Live RGB-D source binding differs from registration.")
        if (
            frame_binding.semantic_content_hash
            != registration.profile.frame_binding_hash
        ):
            raise ValueError("Station frame binding differs from registration.")
        if (
            frame_binding.frame_graph.semantic_content_hash
            != registration.profile.frame_graph_hash
        ):
            raise ValueError("Station frame graph differs from registration.")
        self.planner = planner
        self.registration = registration
        self.perception_provider = perception_provider
        self.frame_binding = frame_binding
        self._current_station_revision = current_station_revision
        self.arm_reachability_evaluator = arm_reachability_evaluator
        if motion_port is not None and not isinstance(motion_port, RealAgentMotionPort):
            raise TypeError("motion_port must be typed or None")
        self.motion_port = motion_port
        if (
            captured_robot_observation_filter is not None
            and not callable(captured_robot_observation_filter)
        ):
            raise TypeError(
                "captured_robot_observation_filter must be callable or None"
            )
        self.captured_robot_observation_filter = captured_robot_observation_filter
        self._lock = threading.Lock()
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def semantic_content_hash(self) -> str:
        return self.registration.profile.inference_implementation_hash

    def start(self) -> None:
        self.planner.start()
        self._started = True

    def close(self) -> None:
        try:
            self.planner.close()
        finally:
            self._started = False

    def plan_request(self, request: Any, *, input_snapshot_sink: Callable[[bytes], None] | None = None) -> PreparedPointGrasp:
        if request.input_artifact_bindings:
            raise ValueError("Live Station point-grasp accepts no input Artifacts.")
        expected_revision = request.expected_station_revision
        with self._lock:
            capture = self.perception_provider.capture_aligned_rgbd(
                role=str(request.parameters["role"]),
                expected_station_revision=expected_revision,
            )
            # Read the lower-owned mailbox; never reconstruct or advance control.
            qpos_cut = None if self.motion_port is None else self.motion_port.latest_measured_qpos()
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
            if (
                resolution.frame_graph_hash
                != self.registration.profile.frame_graph_hash
            ):
                raise RuntimeError(
                    "Station point-grasp frame resolution identity changed."
                )
            image = Image.fromarray(capture.rgb, mode="RGB")
            pixel_uv = tuple(request.parameters["target_pixel_uv"])
            if input_snapshot_sink is not None and self.registration.profile.backend_id == "graspgenx":
                # Persist before segmentation/inference so a failed operation is
                # reproducible too. An unavailable mask is explicitly marked;
                # it cannot pass the completed-capture planning binding.
                input_data = encode_grasp_rgbd_snapshot(
                    rgb=capture.rgb, depth_m=capture.depth_m,
                    target_mask=np.zeros(capture.depth_m.shape, dtype=bool),
                    qpos_by_side=None if qpos_cut is None else qpos_cut.qpos_by_side,
                    metadata={
                        "schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                        "target_mask_available": False,
                        "target_pixel_uv": list(pixel_uv),
                        "camera_intrinsics": dict(capture.intrinsics),
                        "T_reference_from_camera": reference_from_camera.tolist(),
                        "reference_frame_id": self.frame_binding.reference_frame_id,
                        "capture_group_id": capture.capture_group_id,
                        "acquired_at": capture.acquired_at,
                        "station_revision": capture.station_revision,
                        "measured_qpos": None if qpos_cut is None else {
                            "hash": qpos_cut.semantic_content_hash,
                            "measured_at": qpos_cut.measured_at,
                            "available_at": qpos_cut.available_at,
                            "revision": qpos_cut.revision,
                            "evidence_refs": list(qpos_cut.evidence_refs),
                        },
                        "scope": "pre_inference_input_not_a_validated_grasp_plan",
                    },
                )
                input_snapshot_sink(input_data)
            if self.motion_port is not None and (
                qpos_cut is None or abs(qpos_cut.measured_at-capture.acquired_at) > 0.2
            ):
                raise RuntimeError("Point-grasp snapshot lacks a camera-matched measured qpos cut")
            visible_scene_factory = None
            if (
                self.registration.profile.backend_id == "graspgenx"
                and self.captured_robot_observation_filter is not None
            ):
                if qpos_cut is None:
                    raise RuntimeError(
                        "GraspGenX terminal collision prefilter requires a "
                        "camera-matched measured qpos cut"
                    )
                capture_qpos_by_side = qpos_cut.qpos_by_side
                capture_filter = self.captured_robot_observation_filter
                capture_source_from_camera = source_from_camera.copy()

                def build_visible_scene(
                    *,
                    depth_m: np.ndarray,
                    target_mask: np.ndarray,
                    intrinsics: Mapping[str, float | int],
                    geometry: ParallelJawSolidGeometry,
                ) -> VisibleGraspScene:
                    filtered = capture_filter(
                        depth_m=depth_m,
                        target_mask=target_mask,
                        intrinsics=intrinsics,
                        source_from_camera=capture_source_from_camera,
                        capture_qpos_by_side=capture_qpos_by_side,
                    )
                    if not isinstance(
                        filtered, RM65CapturedRobotObservationFilterResult
                    ):
                        raise TypeError(
                            "captured_robot_observation_filter returned the wrong type"
                        )
                    # Candidate poses inside GraspGenX are camera-frame poses;
                    # keep the scene in that same frame. The factory receives
                    # the zero-offset grasp geometry used by the model, while
                    # the output TCP/tool offset is restored by the planner.
                    return VisibleGraspScene.from_depth(
                        depth_m=filtered.depth_m,
                        target_mask=np.asarray(target_mask, dtype=np.bool_),
                        intrinsics=intrinsics,
                        reference_from_camera=np.eye(4, dtype=np.float64),
                        geometry=geometry,
                        capture_qpos_by_side=capture_qpos_by_side,
                        capture_self_observation_evidence=filtered.evidence,
                    )

                # Wire the capture-filtered visible scene into GraspGenX
                # candidate generation. Without this assignment the factory
                # above is defined but the planner silently receives None and
                # terminal background prefiltering is skipped.
                visible_scene_factory = build_visible_scene

            planned = self.planner.plan(
                image,
                capture.depth_m,
                pixel_uv=pixel_uv,
                intrinsics=capture.intrinsics,
                base_from_camera=reference_from_camera,
                max_candidates=request.parameters["max_candidates"],
                **(
                    {"visible_scene_factory": visible_scene_factory}
                    if visible_scene_factory is not None
                    else {}
                ),
            )
            if self._current_station_revision() != expected_revision:
                raise RuntimeError(
                    "Station revision changed during point-grasp model inference."
                )
            annotated = _annotate(
                image,
                pixel_uv,
                planned,
                dict(capture.intrinsics),
                solid_geometry=getattr(self.planner, "solid_geometry", None),
            )
            if len(annotated) > self.registration.profile.max_annotated_bytes:
                raise RuntimeError(
                    "Annotated Station point-grasp image exceeds its bound."
                )
            candidates = []
            for ordinal, candidate in enumerate(planned.candidates):
                camera = list(candidate.camera_from_tcp)
                reference = list(candidate.base_from_tcp)
                candidates.append(
                    {
                        "ordinal": ordinal,
                        "score": candidate.score,
                        "model_score": candidate.model_score,
                        "click_distance_m": candidate.click_distance_m,
                        "gripper_width_m": candidate.gripper_width_m,
                        "T_camera_from_tool_tcp": camera,
                        "T_reference_from_tool_tcp": reference,
                        "tool_tcp_xyz_camera_m": [
                            camera[3],
                            camera[7],
                            camera[11],
                        ],
                        "tool_tcp_xyz_reference_m": [
                            reference[3],
                            reference[7],
                            reference[11],
                        ],
                        "visible_scene_collision_free": (
                            candidate.visible_collision_free
                        ),
                    }
                )
            reachability_context = None
            if request.parameters.get("include_arm_reachability", False):
                if self.arm_reachability_evaluator is None:
                    raise RuntimeError("Arm reachability evaluator is unavailable.")
                reachability_context, annotations = self.arm_reachability_evaluator(
                    tuple(
                        np.asarray(
                            candidate["T_reference_from_tool_tcp"],
                            dtype=np.float64,
                        ).reshape(4, 4)
                        for candidate in candidates
                    )
                )
                if len(annotations) != len(candidates):
                    raise RuntimeError(
                        "Arm reachability evaluator returned the wrong candidate count."
                    )
                reachability_context = dict(reachability_context)
                for candidate, annotation in zip(candidates, annotations, strict=True):
                    candidate["arm_reachability"] = dict(annotation)
            if self._current_station_revision() != expected_revision:
                raise RuntimeError(
                    "Station revision changed before point-grasp result commit."
                )
            profile = self.registration.profile
            result = {
                "source_id": capture.source_id,
                "role": capture.role,
                "acquired_at": capture.acquired_at,
                "station_revision": capture.station_revision,
                "source_generation": capture.source_generation,
                "frame_index": capture.frame_index,
                "capture_group_id": capture.capture_group_id,
                "source_binding_hash": capture.source_binding_hash,
                "frame_binding_hash": self.frame_binding.semantic_content_hash,
                "frame_graph_hash": self.frame_binding.frame_graph.semantic_content_hash,
                "source_frame_id": self.frame_binding.source_frame_id,
                "source_frame_spec_hash": self.frame_binding.source_frame_spec_hash,
                "reference_frame_id": self.frame_binding.reference_frame_id,
                "reference_frame_spec_hash": (
                    self.frame_binding.reference_frame_spec_hash
                ),
                "width": capture.width,
                "height": capture.height,
                "camera_intrinsics": dict(capture.intrinsics),
                "T_source_from_camera": list(capture.source_from_camera),
                "T_reference_from_camera": [
                    float(value) for value in reference_from_camera.reshape(-1)
                ],
                "rgb_content_hash": capture.rgb_content_hash,
                "depth_content_hash": capture.depth_content_hash,
                "target_pixel_uv": list(pixel_uv),
                "target_point_camera_m": list(planned.target_point_camera_m),
                "target_point_reference_m": list(planned.target_point_base_m),
                "target_mask_bbox_xyxy": list(planned.target_mask_bbox_xyxy),
                "pose_semantics": "homogeneous_row_major_parent_from_tool_tcp_m",
                "gripper_profile_id": "DMTAC_M",
                "backend_id": profile.backend_id,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "model_profile_id": profile.model_profile_id,
                "model_profile_hash": profile.model_profile_hash,
                "inference_implementation_hash": (
                    profile.inference_implementation_hash
                ),
                "annotated_image_media_content_hash": hashlib.sha256(
                    annotated
                ).hexdigest(),
                "proposal_semantics": "ranked_unprivileged_motion_proposals_only",
                "collision_scope": (
                    "visible_scene_only_not_full_scene_or_robot_safety"
                ),
                "downstream_control_effects": "forbidden",
            }
            if reachability_context is not None:
                result["arm_reachability_context"] = reachability_context
            snapshot = None
            mask = getattr(self.planner, "last_target_mask", None)
            geometry = getattr(self.planner, "solid_geometry", None)
            if profile.backend_id == "graspgenx":
                if mask is None or geometry is None:
                    raise RuntimeError("GraspGenX replay snapshot geometry or mask is unavailable")
                metadata = {
                    "schema_id": GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                    "capture": result,
                    "solid_geometry": asdict(geometry),
                    "measured_qpos": None if qpos_cut is None else {
                        "hash": qpos_cut.semantic_content_hash,
                        "revision": qpos_cut.revision,
                        "measured_at": qpos_cut.measured_at,
                        "available_at": qpos_cut.available_at,
                        "evidence_refs": list(qpos_cut.evidence_refs),
                        "camera_skew_ms": (qpos_cut.measured_at-capture.acquired_at)*1000,
                    },
                    "scope": "visible_nominal_open_gripper_only_no_closure_or_attached_retreat",
                }
                snapshot = encode_grasp_rgbd_snapshot(
                    rgb=capture.rgb, depth_m=capture.depth_m,
                    target_mask=np.asarray(mask), metadata=metadata,
                    qpos_by_side=None if qpos_cut is None else qpos_cut.qpos_by_side,
                )
            return PreparedPointGrasp(result, annotated, snapshot)


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


def build_station_bound_point_grasp_executor(
    *,
    registration: StationBoundPointGraspRegistration,
    provider: StationBoundPointGraspProvider,
    artifact_store: RecoverableAgentArtifactStore,
    runtime_session_id: str,
    current_station_revision: Callable[[], int],
    max_retained_operations: int,
    clock: Callable[[], float] = time.time,
) -> PassiveReadSkillExecutor:
    """Bind the live proposal Route without granting downstream execution."""

    def read(request: Any, operation: Any) -> PassiveReadResult:
        def retain_input(data: bytes) -> None:
            ref = artifact_ref_for_bytes(
                data, media_type=GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,
                schema_id=GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                encoding=GRASP_RGBD_SNAPSHOT_ENCODING, compression="zip",
            )
            committed_input = commit_artifact_bytes(
                store=artifact_store, data=data, artifact=ref,
                producer_id="agent_gateway.provider.station_point_grasp_input",
                runtime_session_id=runtime_session_id,
                idempotency_key="station-point-grasp-input:"+operation.operation_id,
                now=float(clock()),
            )
            logging.getLogger(__name__).warning(
                "Point-grasp input retained before inference: operation_id=%s sha256=%s path=%s",
                operation.operation_id, committed_input.sha256,
                artifact_store.managed_read_path(committed_input.artifact_id),
            )
        prepared = provider.plan_request(request, input_snapshot_sink=retain_input)
        expected_revision = request.expected_station_revision
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed before point-grasp Artifact commit."
            )
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
            producer_id="agent_gateway.provider.station_bound_point_grasp",
            runtime_session_id=runtime_session_id,
            idempotency_key="station-point-grasp-output:" + operation.operation_id,
            now=float(clock()),
        )
        if current_station_revision() != expected_revision:
            raise RuntimeError(
                "Station revision changed during point-grasp Artifact commit."
            )
        identity = canonical_content_hash(
            {
                "operation_id": operation.operation_id,
                "request_hash": request.semantic_content_hash,
                "result": prepared.execution_result,
                "artifact": committed.semantic_content_hash,
            },
            domain="dexteleop.agent_skill.station_point_grasp_receipt.v2",
        )
        evidence = "station-point-grasp:" + identity
        outputs = [AgentArtifactBinding(
            slot_id=REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2, ordinal=0,
            artifact=committed, production_receipt_ref=evidence,
        )]
        if prepared.rgbd_snapshot is not None:
            snapshot_ref = artifact_ref_for_bytes(
                prepared.rgbd_snapshot, media_type=GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,
                schema_id=GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                encoding=GRASP_RGBD_SNAPSHOT_ENCODING, compression="zip",
            )
            snapshot_ref = commit_artifact_bytes(
                store=artifact_store, data=prepared.rgbd_snapshot, artifact=snapshot_ref,
                producer_id="agent_gateway.provider.station_bound_point_grasp",
                runtime_session_id=runtime_session_id,
                idempotency_key="station-point-grasp-snapshot:"+operation.operation_id,
                now=float(clock()),
            )
            outputs.append(AgentArtifactBinding(
                slot_id=REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID, ordinal=0,
                artifact=snapshot_ref, production_receipt_ref=evidence,
            ))
        return PassiveReadResult(
            execution_result=prepared.execution_result,
            evidence_refs=(evidence,),
            output_artifact_bindings=tuple(outputs),
            reason_code="passive_station_point_grasp_planning_completed",
        )

    return PassiveReadSkillExecutor(
        port_id="agent-gateway:station-point-grasp:" + runtime_session_id,
        executor_id=registration.binding.executor_id,
        read=read,
        current_station_revision=current_station_revision,
        clock=clock,
        max_retained_operations=max_retained_operations,
    )


__all__ = [
    "PointGraspProvider",
    "PreparedPointGrasp",
    "StationBoundPointGraspProvider",
    "build_real_point_grasp_executor",
    "build_station_bound_point_grasp_executor",
]

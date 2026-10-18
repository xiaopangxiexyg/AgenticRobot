"""Aligned RGB-D input provider and passive point-grasp Gateway executor."""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import logging
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
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
    GRASP_RGBD_SNAPSHOT_ENCODING,
    GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,
    GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
    encode_grasp_rgbd_snapshot,
    decode_grasp_rgbd_snapshot,
)
from dataset_contracts.grasp_multiview_snapshot import (
    GRASP_MULTIVIEW_SCHEMA_ID, GRASP_MULTIVIEW_ENCODING, encode_grasp_multiview_arrays,
)
from deployments.perception.grasp_multiview import GraspRgbdView, fuse_grasp_views
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
from robot_motion.grasp_visible_scene import (
    PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
    ParallelJawSolidGeometry,
    VisibleGraspScene,
)
from robot_motion.rm65.captured_robot_observations import (
    RM65CapturedRobotObservationFilterResult,
)
from teleop_runtime.skill_deploy.motion_port import RealAgentMotionPort, RealAgentMotionMeasuredQposCut

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
from .realsense_perception import RealAlignedRgbdCapture, RealSenseAgentPerceptionProvider
from .station_point_grasp_deployment import (
    REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
    REAL_STATION_POINT_GRASP_SNAPSHOT_SLOT_ID,
    StationBoundPointGraspRegistration,
)

_LOGGER = logging.getLogger(__name__)


@contextmanager
def _record_elapsed_ms(timings, phase):
    started = time.perf_counter()
    try:
        yield
    finally:
        timings[phase] = (time.perf_counter() - started) * 1_000.


@dataclass(frozen=True)
class PreparedPointGrasp:
    execution_result: dict[str, Any]
    annotated_jpeg: bytes
    rgbd_snapshot: bytes | None = None
    multiview_snapshot: bool = False


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
            store.read_bytes(rgb, maximum=profile.max_rgb_bytes),
            store.read_bytes(depth, maximum=profile.max_depth_bytes),
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
            payload = {
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
            if candidate.pose_source != "model_scored":
                payload["pose_source"] = candidate.pose_source
                if candidate.orientation_variant_axis is not None:
                    payload.update(
                        {
                            "orientation_variant_axis": (
                                candidate.orientation_variant_axis
                            ),
                            "orientation_variant_angle_deg": (
                                candidate.orientation_variant_angle_deg
                            ),
                        }
                    )
                if candidate.support_clearance_world_z_m > 0.0:
                    payload["support_clearance_world_z_m"] = (
                        candidate.support_clearance_world_z_m
                    )
            candidates.append(payload)
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


def _call_arm_reachability_evaluator(
    evaluator: Callable[..., object],
    candidates: tuple[np.ndarray, ...],
    *,
    measured_qpos_cut: object | None,
    measurement_reference_time: float | None,
    endpoint_seeds: Mapping[tuple[float, ...], tuple[float, ...]] | None = None,
) -> object:
    optional: dict[str, object] = {}
    if measured_qpos_cut is not None or endpoint_seeds:
        try:
            parameters = inspect.signature(evaluator).parameters
        except (TypeError, ValueError):
            parameters = {}
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if measured_qpos_cut is not None and (accepts_kwargs or "measured_qpos_cut" in parameters):
            optional["measured_qpos_cut"] = measured_qpos_cut
        if endpoint_seeds and "endpoint_seeds" in parameters:
            optional["endpoint_seeds"] = endpoint_seeds
        if (
            measurement_reference_time is not None
            and measured_qpos_cut is not None
            and (accepts_kwargs or "measurement_reference_time" in parameters)
        ):
            optional["measurement_reference_time"] = float(measurement_reference_time)
    return evaluator(candidates, **optional)


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
        non_motion_read_window: Callable[[], object] | None = None,
        observation_reader: Callable[..., object] | None = None,
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
        if (
            non_motion_read_window is not None
            and not callable(non_motion_read_window)
        ):
            raise TypeError("non_motion_read_window must be callable or None")
        self.non_motion_read_window = non_motion_read_window
        self.observation_reader = observation_reader
        if registration.profile.frozen_observation and not callable(observation_reader):
            raise ValueError("Frozen point-grasp requires its retained observation reader")
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
        started = time.perf_counter()
        timings: dict[str, Any] = {}
        completed = False
        try:
            window_factory = self.non_motion_read_window
            window = (
                window_factory()
                if window_factory is not None
                else nullcontext()
            )
            with window:
                prepared = self._plan_request_unwindowed(
                    request, input_snapshot_sink=input_snapshot_sink, timings=timings)
            completed = True
            return prepared
        finally:
            timings["provider_total"] = (time.perf_counter() - started) * 1_000.
            _LOGGER.info("Point-grasp provider timing: %s", json.dumps({
                "request_id": getattr(request, "request_id", None),
                "status": "completed" if completed else "failed",
                "timing_ms": timings}, sort_keys=True))

    def _plan_request_unwindowed(
        self,
        request: Any,
        *,
        input_snapshot_sink: Callable[[bytes], None] | None = None,
        timings: dict[str, Any],
    ) -> PreparedPointGrasp:
        if request.input_artifact_bindings:
            raise ValueError("Live Station point-grasp accepts no input Artifacts.")
        expected_revision = request.expected_station_revision
        preparation_started = time.perf_counter()
        with self._lock:
            timings["provider_lock_wait"] = (time.perf_counter() - preparation_started) * 1_000.
            with _record_elapsed_ms(timings, "capture_and_preparation"):
                frozen_views = None
                fused_observation = None
                if self.registration.profile.frozen_observation:
                    view_blobs, observation_metadata = self.observation_reader(
                        request.parameters["observation_operation_id"], expected_revision)
                    frozen_views = [decode_grasp_rgbd_snapshot(blob) for blob in view_blobs]
                    for arrays, metadata in frozen_views:
                        if (metadata["source_binding_hash"] != self.registration.profile.source_binding_hash
                                or metadata["frame_binding_hash"] != self.frame_binding.semantic_content_hash
                                or metadata["station_revision"] != expected_revision
                                or metadata.get("stationary_only") is not True
                                or arrays["left_qpos"].shape != (6,)):
                            raise ValueError("Frozen RGB-D view geometry or provenance differs")
                    selected = [(a, m) for a, m in frozen_views if m["role"] == request.parameters["role"]]
                    if len(selected) != 1:
                        raise ValueError("Selected role is unavailable in the retained observation")
                    arrays, metadata = selected[0]
                    capture = RealAlignedRgbdCapture(
                        source_id=metadata["source_id"], role=metadata["role"], acquired_at=metadata["acquired_at"],
                        station_revision=expected_revision, source_generation=metadata["source_generation"],
                        frame_index=metadata["frame_index"], capture_group_id=metadata["capture_group_id"],
                        source_binding_hash=metadata["source_binding_hash"], width=arrays["rgb"].shape[1], height=arrays["rgb"].shape[0],
                        intrinsics=metadata["camera_intrinsics"], source_from_camera=tuple(np.asarray(metadata["T_source_from_camera"]).reshape(-1)),
                        rgb=arrays["rgb"], depth_m=arrays["depth_m"], rgb_content_hash=metadata["rgb_content_hash"],
                        depth_content_hash=metadata["depth_content_hash"])
                    qpos = metadata["measured_qpos"]
                    qpos_cut = RealAgentMotionMeasuredQposCut(revision=0, measured_at=qpos["measured_at"], available_at=qpos["available_at"],
                        qpos_by_side={side: arrays[side+"_qpos"] for side in ("left", "right")},
                        source_evidence_refs=tuple(qpos["source_evidence_refs"]))
                else:
                    capture = self.perception_provider.capture_aligned_rgbd(
                        role=str(request.parameters["role"]), expected_station_revision=expected_revision)
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
                            depth_filter=PRODUCTION_VISIBLE_GRASP_DEPTH_FILTER,
                        )

                    # Wire the capture-filtered visible scene into GraspGenX
                    # candidate generation. Without this assignment the factory
                    # above is defined but the planner silently receives None and
                    # terminal background prefiltering is skipped.
                    visible_scene_factory = build_visible_scene

                if frozen_views is not None:
                    if self.captured_robot_observation_filter is None:
                        raise RuntimeError("Frozen multi-view inference requires capture-time robot filtering")
                    camera_from_reference = np.linalg.inv(reference_from_camera)
                    by_role = {m["role"]: (a, m) for a, m in frozen_views}
                    views = {role: GraspRgbdView(role=role, rgb=a["rgb"], depth_m=a["depth_m"],
                        intrinsics=m["camera_intrinsics"], acquired_at=m["acquired_at"],
                        reference_from_camera=camera_from_reference @ np.asarray(m["T_reference_from_camera"]).reshape(4, 4))
                        for role, (a, m) in by_role.items()}

                    def clean_depth(view, mask):
                        a, m = by_role[view.role]
                        filtered = self.captured_robot_observation_filter(
                            depth_m=view.depth_m, target_mask=mask, intrinsics=view.intrinsics,
                            source_from_camera=np.asarray(m["T_source_from_camera"]).reshape(4, 4),
                            capture_qpos_by_side={side: a[side+"_qpos"] for side in ("left", "right")})
                        if not isinstance(filtered, RM65CapturedRobotObservationFilterResult):
                            raise TypeError("Invalid captured-robot filter result")
                        return filtered.depth_m

                    def build_fused_scene(*, depth_m, target_mask, intrinsics, geometry):
                        nonlocal fused_observation
                        fused_observation = fuse_grasp_views(
                            anchor=views[capture.role], anchor_mask=target_mask,
                            anchor_pixel_uv=pixel_uv,
                            auxiliary=tuple(v for role, v in views.items() if role != capture.role)
                                if request.parameters["view_policy"] == "auto" else (),
                            geometry=geometry, segment_point=self.planner.point_segmenter.segment_point,
                            clean_depth=clean_depth)
                        return fused_observation.scene

                    visible_scene_factory = build_fused_scene

                candidate_selector = None
                # Proposal seeds belong only to this frozen request. No cached
                # reachability label or execution authority crosses the boundary.
                endpoint_seeds: dict[tuple[float, ...], tuple[float, ...]] = {}
                if (
                    request.parameters.get("include_arm_reachability", False)
                    and self.registration.profile.backend_id == "graspgenx"
                    and self.arm_reachability_evaluator is not None
                ):
                    selector = getattr(
                        self.arm_reachability_evaluator, "select_candidates", None
                    )
                    if callable(selector):
                        try:
                            accepts_endpoint_seeds = "endpoint_seeds" in inspect.signature(selector).parameters
                        except (TypeError, ValueError):
                            accepts_endpoint_seeds = False
                        def select_candidates(
                            candidates,
                            target_point_reference_m,
                            maximum,
                        ):
                            return selector(
                                candidates,
                                target_point_reference_m,
                                maximum,
                                measured_qpos_cut=qpos_cut,
                                measurement_reference_time=float(capture.acquired_at),
                                **({"endpoint_seeds": endpoint_seeds} if accepts_endpoint_seeds else {}),
                            )

                        candidate_selector = select_candidates

            proposal_started = time.perf_counter()
            try:
                planned = self.planner.plan(
                    image, capture.depth_m, pixel_uv=pixel_uv, intrinsics=capture.intrinsics,
                    base_from_camera=reference_from_camera,
                    max_candidates=request.parameters["max_candidates"],
                    **({"use_visible_target_points": True} if frozen_views is not None else {}),
                    **({"visible_scene_factory": visible_scene_factory} if visible_scene_factory is not None else {}),
                    **({"candidate_selector": candidate_selector} if candidate_selector is not None else {}))
            finally:
                timings["proposal_planning"] = (time.perf_counter() - proposal_started) * 1_000.
                stages = getattr(self.planner, "last_plan_timing_ms", None)
                if isinstance(stages, Mapping):
                    timings["proposal_stages"] = dict(stages)
            filter_summary = getattr(self.planner, "last_filter_summary", None)
            if isinstance(filter_summary, Mapping):
                _LOGGER.info(
                    "Point-grasp candidate filter summary: %s",
                    json.dumps(dict(filter_summary), sort_keys=True),
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
                payload = {
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
                if candidate.pose_source != "model_scored":
                    payload["pose_source"] = candidate.pose_source
                    if candidate.orientation_variant_axis is not None:
                        payload.update(
                            {
                                "orientation_variant_axis": (
                                    candidate.orientation_variant_axis
                                ),
                                "orientation_variant_angle_deg": (
                                    candidate.orientation_variant_angle_deg
                                ),
                            }
                        )
                    if candidate.support_clearance_world_z_m > 0.0:
                        payload["support_clearance_world_z_m"] = (
                            candidate.support_clearance_world_z_m
                        )
                candidates.append(payload)
            reachability_context = None
            if request.parameters.get("include_arm_reachability", False):
                if self.arm_reachability_evaluator is None:
                    raise RuntimeError("Arm reachability evaluator is unavailable.")
                candidate_poses = tuple(
                    np.asarray(
                        candidate["T_reference_from_tool_tcp"],
                        dtype=np.float64,
                    ).reshape(4, 4)
                    for candidate in candidates
                )
                annotation_started = time.perf_counter()
                try:
                    reachability_context, annotations = _call_arm_reachability_evaluator(
                        self.arm_reachability_evaluator, candidate_poses, measured_qpos_cut=qpos_cut,
                        measurement_reference_time=float(capture.acquired_at), endpoint_seeds=endpoint_seeds)
                finally:
                    timings["endpoint_annotation"] = (time.perf_counter() - annotation_started) * 1_000.
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
            with _record_elapsed_ms(timings, "result_encoding"):
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
                if fused_observation is not None:
                    result["view_fusion"] = {
                        "observation_operation_id": request.parameters["observation_operation_id"],
                        "view_policy": request.parameters["view_policy"],
                        "diagnostics_json": json.dumps(fused_observation.view_diagnostics),
                    }
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
                        "candidate_filter_summary": (
                            dict(filter_summary)
                            if isinstance(filter_summary, Mapping)
                            else None
                        ),
                        "scope": "visible_nominal_open_gripper_only_no_closure_or_attached_retreat",
                    }
                    snapshot = encode_grasp_rgbd_snapshot(
                        rgb=capture.rgb, depth_m=capture.depth_m,
                        target_mask=np.asarray(mask), metadata=metadata,
                        qpos_by_side=None if qpos_cut is None else qpos_cut.qpos_by_side,
                    )
                    if fused_observation is not None:
                        retained_views = []
                        for a, m in frozen_views:
                            target_mask = fused_observation.target_masks.get(m["role"])
                            retained_views.append(dict(
                                rgb=a["rgb"], depth_m=a["depth_m"],
                                target_mask=np.zeros_like(a["target_mask"]) if target_mask is None else target_mask,
                                qpos_by_side={side: a[side+"_qpos"] for side in ("left", "right")},
                                metadata={**m, "target_mask_available": target_mask is not None}))
                        snapshot = encode_grasp_multiview_arrays(views=retained_views,
                            metadata={**metadata, "schema_id": GRASP_MULTIVIEW_SCHEMA_ID, "scene_frame": "anchor_camera"},
                            scene_points=fused_observation.scene.points,
                            scene_target_mask=fused_observation.scene.target_mask)
                return PreparedPointGrasp(result, annotated, snapshot, fused_observation is not None)


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

    def read_and_publish(request: Any, operation: Any, timings: dict[str, float]) -> PassiveReadResult:
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
        with _record_elapsed_ms(timings, "artifact_publication"):
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
                    prepared.rgbd_snapshot, media_type="application/zip" if prepared.multiview_snapshot else GRASP_RGBD_SNAPSHOT_MEDIA_TYPE,
                    schema_id=GRASP_MULTIVIEW_SCHEMA_ID if prepared.multiview_snapshot else GRASP_RGBD_SNAPSHOT_SCHEMA_ID,
                    encoding=GRASP_MULTIVIEW_ENCODING if prepared.multiview_snapshot else GRASP_RGBD_SNAPSHOT_ENCODING, compression="zip",
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

    def read(request: Any, operation: Any) -> PassiveReadResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        completed = False
        try:
            result = read_and_publish(request, operation, timings)
            completed = True
            return result
        finally:
            timings["producer_total"] = (time.perf_counter() - started) * 1_000.
            _LOGGER.info("Point-grasp producer timing: %s", json.dumps({
                "operation_id": getattr(operation, "operation_id", None),
                "request_id": getattr(request, "request_id", None),
                "status": "completed" if completed else "failed",
                "timing_ms": timings}, sort_keys=True))

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

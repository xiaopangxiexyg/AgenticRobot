"""RealSense resource and passive capture boundary for the real Gateway."""

from __future__ import annotations

import hashlib
import io
import math
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np
from PIL import Image

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
)
from robot_motion.relative_action import pose_matrix_to_xyzquat
from teleop_runtime.cameras import load_camera_registry
from teleop_runtime.sources.camera import (
    REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    RealSenseRGBSource,
    load_realsense_camera_config_map,
)

from .perception_deployment import RealPerceptionCaptureProfile

_ROLE_BY_SOURCE_ID = {
    "front": "main",
    "left_wrist": "wrist_left",
    "right_wrist": "wrist_right",
}


class _LifecycleResource(Protocol):
    def start(self) -> None: ...

    def close(self) -> None: ...


class _CloseResource(Protocol):
    def close(self) -> None: ...


def _apply_formal_camera_startup_defaults(configs: Mapping[str, Any]) -> None:
    # Keep the optional perception attachment out of motion-only Gateway
    # imports. The actual lifecycle remains solely owned by teleop_runtime.
    from teleop_runtime.real_teleop.camera_runtime import (
        apply_real_teleop_camera_startup_defaults,
    )

    apply_real_teleop_camera_startup_defaults(
        configs,
        color_format="bgr8",
        warmup_policy=REALSENSE_STARTUP_WARMUP_BOUNDED_FIRST_FRAME,
    )


def _formal_start_camera_sources(camera_sources: Mapping[str, Any]) -> None:
    from teleop_runtime.deploy.camera_runtime import start_camera_sources

    start_camera_sources(camera_sources)


def _formal_camera_startup_self_check(
    camera_sources: Mapping[str, Any],
) -> None:
    from teleop_runtime.deploy.camera_runtime import run_camera_startup_self_check

    run_camera_startup_self_check(camera_sources)


def _formal_close_camera_sources(camera_sources: Mapping[str, Any]) -> None:
    from teleop_runtime.deploy.camera_runtime import close_camera_sources

    close_camera_sources(camera_sources)


def real_camera_role_for_source(source_id: str) -> str:
    source = _required_text(source_id, field_name="source_id")
    return _ROLE_BY_SOURCE_ID.get(source, f"auxiliary_{source}")


def _required_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field_name} must be a non-empty trimmed string.")
    return value


def _finite_timestamp(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be numeric.")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite.")
    return normalized


def _pose_matrix(value: object, *, field_name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must be a finite 4x4 transform.")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-9, rtol=0.0):
        raise ValueError(f"{field_name} has an invalid homogeneous row.")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.0):
        raise ValueError(f"{field_name} rotation is not orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{field_name} rotation is not right-handed.")
    return matrix.copy()


@dataclass(frozen=True)
class EncodedRealRgbFrame:
    source_id: str
    role: str
    acquired_at: float
    width: int
    height: int
    data: bytes
    media_type: str = RGB_FRAME_MEDIA_TYPE_V1


@dataclass(frozen=True)
class RealRgbCapture:
    capture_window_started_at: float
    capture_window_ended_at: float
    station_revision: int
    frames: tuple[EncodedRealRgbFrame, ...]

    @property
    def captured_at(self) -> float:
        """Compatibility projection for v1 callers."""

        return self.capture_window_ended_at


@dataclass(frozen=True)
class EncodedRealDepthFrame:
    source_id: str
    role: str
    captured_at: float
    width: int
    height: int
    camera_pose_world_xyz_quat_xyzw: tuple[float, ...]
    intrinsics: Mapping[str, float | int]
    data: bytes
    media_type: str = DEPTH_FRAME_MEDIA_TYPE_V1
    schema_id: str = DEPTH_FRAME_SCHEMA_ID_V1
    encoding: str = DEPTH_FRAME_ENCODING_V1
    compression: str = DEPTH_FRAME_COMPRESSION_V1


@dataclass(frozen=True)
class RealAlignedRgbdCapture:
    """One correlated live frame with deployment-owned metric geometry."""

    source_id: str
    role: str
    acquired_at: float
    station_revision: int
    source_generation: int
    frame_index: int
    capture_group_id: str
    source_binding_hash: str
    width: int
    height: int
    intrinsics: Mapping[str, float | int]
    source_from_camera: tuple[float, ...]
    rgb: np.ndarray
    depth_m: np.ndarray
    rgb_content_hash: str
    depth_content_hash: str


@dataclass(frozen=True)
class RealRgbCameraGeometryCut:
    """One deployment-owned camera pose matched to a published RGB frame."""

    source_id: str
    source_from_camera: tuple[float, ...]
    measured_at: float
    available_at: float
    geometry_binding_hash: str
    geometry_cut_hash: str
    evidence_refs: tuple[str, ...]


@runtime_checkable
class RealRgbCameraGeometryProvider(Protocol):
    """Trusted static/dynamic camera geometry prepared by the application."""

    @property
    def semantic_content_hash(self) -> str: ...

    def source_catalog(self) -> tuple[Mapping[str, object], ...]: ...

    def pose_cut(
        self,
        *,
        source_id: str,
        frame_acquired_at: float,
        now: float,
    ) -> RealRgbCameraGeometryCut: ...


@dataclass(frozen=True)
class RealRgbGeometryCapture:
    """One live RGB frame and its deployment-owned optical-frame pose."""

    source_id: str
    role: str
    acquired_at: float
    station_revision: int
    source_generation: int
    frame_index: int
    capture_group_id: str
    source_binding_hash: str
    width: int
    height: int
    intrinsics: Mapping[str, float | int]
    source_from_camera: tuple[float, ...]
    rgb: np.ndarray
    rgb_content_hash: str
    geometry_binding_hash: str
    geometry_cut_hash: str
    pose_measured_at: float
    pose_available_at: float
    pose_evidence_refs: tuple[str, ...]


@dataclass(frozen=True)
class RealPerceptionSourceBinding:
    """Preflighted camera inventory and trusted static depth poses."""

    camera_sources: Mapping[str, object]
    world_from_depth_camera_by_source: Mapping[str, np.ndarray]
    semantic_content_hash: str

    def __post_init__(self) -> None:
        sources = dict(self.camera_sources)
        if not sources:
            raise ValueError("Real perception source binding must not be empty.")
        for source_id, source in sources.items():
            _required_text(source_id, field_name="camera_sources key")
            if getattr(source, "config", None) is None:
                raise TypeError(f"camera source {source_id!r} has no config.")
        depth_sources = {
            source_id
            for source_id, source in sources.items()
            if bool(source.config.enable_depth)
        }
        if set(self.world_from_depth_camera_by_source) != depth_sources:
            raise ValueError(
                "Every enabled depth source requires exactly one trusted pose."
            )
        poses = {}
        for source_id, value in self.world_from_depth_camera_by_source.items():
            matrix = _pose_matrix(
                value,
                field_name=f"world_from_depth_camera_by_source.{source_id}",
            )
            matrix.setflags(write=False)
            poses[source_id] = matrix
        object.__setattr__(self, "camera_sources", MappingProxyType(sources))
        object.__setattr__(
            self,
            "world_from_depth_camera_by_source",
            MappingProxyType(poses),
        )
        object.__setattr__(
            self,
            "semantic_content_hash",
            _required_text(
                self.semantic_content_hash,
                field_name="semantic_content_hash",
            ),
        )


def build_real_perception_source_binding(
    *,
    camera_config_path: str | Path,
    camera_registry_path: str | Path,
    scene_world_from_arm_base_by_side: Mapping[str, np.ndarray],
) -> RealPerceptionSourceBinding:
    """Prepare sources and static depth geometry without opening hardware."""

    configs = load_realsense_camera_config_map(camera_config_path)
    if not configs:
        raise ValueError("Real perception camera config must not be empty.")
    for camera_key, config in configs.items():
        if config.startup_hardware_reset:
            raise ValueError(
                f"camera {camera_key!r} cannot request a startup hardware reset."
            )
    _apply_formal_camera_startup_defaults(configs)
    for config in configs.values():
        config.keep_last = max(4, int(config.keep_last))
    registry = load_camera_registry(camera_registry_path)
    registry_cameras = registry["cameras"]
    world_from_base = {
        side: _pose_matrix(
            scene_world_from_arm_base_by_side[side],
            field_name=f"scene_world_from_arm_base_by_side.{side}",
        )
        for side in ("left", "right")
    }
    static_depth_poses: dict[str, np.ndarray] = {}
    binding_cameras = []
    for camera_key, config in configs.items():
        spec = registry_cameras.get(camera_key)
        if not isinstance(spec, Mapping):
            raise ValueError(  # noqa: TRY004 - registry validation contract
                f"camera {camera_key!r} is absent from the registry."
            )
        if str(spec.get("serial")) != str(config.serial):
            raise ValueError(f"camera {camera_key!r} serial differs from the registry.")
        profile = spec.get("profile")
        if not isinstance(profile, Mapping) or any(
            int(profile.get(field_name, -1)) != int(getattr(config, field_name))
            for field_name in ("width", "height", "fps")
        ):
            raise ValueError(
                f"camera {camera_key!r} profile differs from the registry."
            )
        pose_payload = None
        if config.enable_depth:
            frames = (spec.get("extrinsics") or {}).get("frames")
            candidates = tuple(
                value
                for value in (frames.values() if isinstance(frames, Mapping) else ())
                if isinstance(value, Mapping) and "T_camera_to_parent" in value
            )
            if len(candidates) != 1:
                raise ValueError(
                    f"depth camera {camera_key!r} requires exactly one trusted "
                    "static T_camera_to_parent binding."
                )
            frame = candidates[0]
            parent = str(frame.get("parent_frame") or "")
            side_by_parent = {
                "left_arm_base_link": "left",
                "right_arm_base_link": "right",
            }
            if parent not in side_by_parent:
                raise ValueError(
                    f"depth camera {camera_key!r} parent is not a fixed arm base."
                )
            parent_from_camera = _pose_matrix(
                frame["T_camera_to_parent"],
                field_name=f"camera_registry.{camera_key}.T_camera_to_parent",
            )
            world_from_camera = (
                world_from_base[side_by_parent[parent]] @ parent_from_camera
            )
            static_depth_poses[camera_key] = _pose_matrix(
                world_from_camera,
                field_name=f"world_from_depth_camera.{camera_key}",
            )
            pose_payload = static_depth_poses[camera_key].tolist()
        binding_cameras.append(
            {
                "source_id": camera_key,
                "role": real_camera_role_for_source(camera_key),
                "config": config.to_dict(),
                "registry_serial": str(spec["serial"]),
                "world_from_depth_camera": pose_payload,
            }
        )
    binding_hash = canonical_content_hash(
        {
            "camera_registry_schema": registry["schema_version"],
            "cameras": binding_cameras,
            "pose_semantics": "world_from_camera_optical_frame",
        },
        domain="dexteleop.real_realsense_perception_source_binding.v1",
    )
    return RealPerceptionSourceBinding(
        camera_sources={
            key: RealSenseRGBSource(config) for key, config in configs.items()
        },
        world_from_depth_camera_by_source=static_depth_poses,
        semantic_content_hash=binding_hash,
    )


class ExclusiveRealPerceptionResources:
    """One transaction over cameras and the existing lower RM65 resources."""

    def __init__(
        self,
        *,
        lower: _LifecycleResource,
        camera_sources: Mapping[str, object],
        artifact_store: _CloseResource | None = None,
        additional_resources: tuple[_LifecycleResource, ...] = (),
        start_cameras: Callable[[Mapping[str, Any]], None] | None = None,
        admit_cameras: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not callable(getattr(lower, "start", None)) or not callable(
            getattr(lower, "close", None)
        ):
            raise TypeError("lower must expose start and close.")
        if not camera_sources:
            raise ValueError("camera_sources must not be empty.")
        if artifact_store is not None and not callable(
            getattr(artifact_store, "close", None)
        ):
            raise TypeError("artifact_store must expose close.")
        resources = tuple(additional_resources)
        if any(
            not callable(getattr(resource, "start", None))
            or not callable(getattr(resource, "close", None))
            for resource in resources
        ):
            raise TypeError("additional_resources must expose start and close.")
        start_cameras = (
            start_realsense_sources if start_cameras is None else start_cameras
        )
        admit_cameras = (
            admit_realsense_sources if admit_cameras is None else admit_cameras
        )
        if not callable(start_cameras) or not callable(admit_cameras):
            raise TypeError("camera lifecycle callbacks must be callable.")
        self.lower = lower
        self.camera_sources = dict(camera_sources)
        self.artifact_store = artifact_store
        self.additional_resources = resources
        self._start_cameras = start_cameras
        self._admit_cameras = admit_cameras
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._start_attempted = False
        self._started = False
        self._closed = False
        self._lower_closed = False
        self._additional_closed = [False] * len(self.additional_resources)
        self._cameras_closed = False
        self._artifact_store_closed = self.artifact_store is None

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("closed real perception resources cannot restart.")
            if self._start_attempted:
                if self._started:
                    return
                raise RuntimeError(
                    "real perception resource startup is already in progress or "
                    "failed; cannot restart."
                )
            self._start_attempted = True
        cameras_attempted = False
        lower_attempted = False
        attempted_resource_indexes: list[int] = []
        try:
            cameras_attempted = True
            self._start_cameras(self.camera_sources)
            self._admit_cameras(self.camera_sources)
            for index, resource in enumerate(self.additional_resources):
                attempted_resource_indexes.append(index)
                resource.start()
            lower_attempted = True
            self.lower.start()
        except Exception:
            if lower_attempted:
                try:
                    self.lower.close()
                except Exception:  # noqa: BLE001, S110 - preserve the original startup failure
                    pass
                else:
                    self._lower_closed = True
            else:
                self._lower_closed = True
            for index in reversed(attempted_resource_indexes):
                resource = self.additional_resources[index]
                try:
                    resource.close()
                except Exception:  # noqa: BLE001, S110 - preserve startup failure
                    pass
                else:
                    self._additional_closed[index] = True
            for index in range(
                len(attempted_resource_indexes), len(self.additional_resources)
            ):
                self._additional_closed[index] = True
            if cameras_attempted:
                self._close_cameras()
            if self.artifact_store is not None:
                try:
                    self.artifact_store.close()
                except Exception:  # noqa: BLE001, S110 - preserve the original startup failure
                    pass
                else:
                    self._artifact_store_closed = True
            with self._lock:
                self._closed = bool(
                    self._lower_closed
                    and all(self._additional_closed)
                    and self._cameras_closed
                    and self._artifact_store_closed
                )
            raise
        with self._lock:
            self._started = True

    def _close_cameras(self) -> list[str]:
        with self._lock:
            if self._cameras_closed:
                return []
        try:
            _formal_close_camera_sources(self.camera_sources)
        except Exception as exc:  # noqa: BLE001 - formal cleanup already tried all cameras.
            return [f"cameras:{type(exc).__name__}:{exc}"]
        with self._lock:
            self._cameras_closed = True
        return []

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
            errors = []
            if not self._lower_closed:
                try:
                    self.lower.close()
                except Exception as exc:  # noqa: BLE001 - cameras still must close.
                    errors.append(f"lower:{type(exc).__name__}:{exc}")
                else:
                    self._lower_closed = True
            for index in reversed(range(len(self.additional_resources))):
                if self._additional_closed[index]:
                    continue
                resource = self.additional_resources[index]
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001 - finish all cleanup stages.
                    errors.append(
                        f"additional_resource[{index}]:{type(exc).__name__}:{exc}"
                    )
                else:
                    self._additional_closed[index] = True
            errors.extend(self._close_cameras())
            if not self._artifact_store_closed:
                assert self.artifact_store is not None
                try:
                    self.artifact_store.close()
                except Exception as exc:  # noqa: BLE001 - complete resource close.
                    errors.append(f"artifact_store:{type(exc).__name__}:{exc}")
                else:
                    self._artifact_store_closed = True
            with self._lock:
                self._closed = bool(
                    self._lower_closed
                    and all(self._additional_closed)
                    and self._cameras_closed
                    and self._artifact_store_closed
                )
                complete = self._closed
            if not complete and not errors:
                errors.append("close_transaction_incomplete")
            if errors:
                raise RuntimeError(
                    "real perception resource close failed: " + "; ".join(errors)
                )


class CameraOnlyRealPerceptionResources:
    """Own cameras, optional read-only models, and Artifact storage."""

    def __init__(
        self,
        *,
        camera_sources: Mapping[str, object],
        artifact_store: _CloseResource,
        additional_resources: tuple[_LifecycleResource, ...] = (),
        start_cameras: Callable[[Mapping[str, Any]], None] | None = None,
        admit_cameras: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not camera_sources:
            raise ValueError("camera_sources must not be empty.")
        if not callable(getattr(artifact_store, "close", None)):
            raise TypeError("artifact_store must expose close.")
        resources = tuple(additional_resources)
        if any(
            not callable(getattr(resource, "start", None))
            or not callable(getattr(resource, "close", None))
            for resource in resources
        ):
            raise TypeError("additional_resources must expose start and close.")
        start_cameras = (
            start_realsense_sources if start_cameras is None else start_cameras
        )
        admit_cameras = (
            admit_realsense_sources if admit_cameras is None else admit_cameras
        )
        if not callable(start_cameras) or not callable(admit_cameras):
            raise TypeError("camera lifecycle callbacks must be callable.")
        self.camera_sources = dict(camera_sources)
        self.artifact_store = artifact_store
        self.additional_resources = resources
        self._start_cameras = start_cameras
        self._admit_cameras = admit_cameras
        self._lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._start_attempted = False
        self._started = False
        self._closed = False
        self._additional_closed = [False] * len(self.additional_resources)
        self._cameras_closed = False
        self._artifact_store_closed = False

    @property
    def started(self) -> bool:
        with self._lock:
            return self._started

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("closed camera-only resources cannot restart.")
            if self._start_attempted:
                if self._started:
                    return
                raise RuntimeError(
                    "camera-only resource startup is already in progress or failed; "
                    "cannot restart."
                )
            self._start_attempted = True
        attempted_resource_indexes = []
        try:
            self._start_cameras(self.camera_sources)
            self._admit_cameras(self.camera_sources)
            for index, resource in enumerate(self.additional_resources):
                attempted_resource_indexes.append(index)
                resource.start()
        except Exception as start_error:
            cleanup_errors = []
            for index in reversed(attempted_resource_indexes):
                resource = self.additional_resources[index]
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001 - aggregate startup cleanup.
                    cleanup_errors.append(
                        f"additional_resource[{index}]:{type(exc).__name__}:{exc}"
                    )
                else:
                    self._additional_closed[index] = True
            for index in range(
                len(attempted_resource_indexes), len(self.additional_resources)
            ):
                self._additional_closed[index] = True
            cleanup_errors.extend(self._close_cameras())
            try:
                self.artifact_store.close()
            except Exception as exc:  # noqa: BLE001 - preserve all cleanup evidence.
                cleanup_errors.append(f"artifact_store:{type(exc).__name__}:{exc}")
            else:
                self._artifact_store_closed = True
            with self._lock:
                self._closed = bool(
                    all(self._additional_closed)
                    and self._cameras_closed
                    and self._artifact_store_closed
                )
            if cleanup_errors:
                raise RuntimeError(
                    "camera-only perception startup cleanup failed: "
                    + "; ".join(cleanup_errors)
                ) from start_error
            raise
        with self._lock:
            self._started = True

    def _close_cameras(self) -> list[str]:
        with self._lock:
            if self._cameras_closed:
                return []
        try:
            _formal_close_camera_sources(self.camera_sources)
        except Exception as exc:  # noqa: BLE001 - formal close covers all cameras.
            return [f"cameras:{type(exc).__name__}:{exc}"]
        with self._lock:
            self._cameras_closed = True
        return []

    def close(self) -> None:
        with self._close_lock:
            with self._lock:
                if self._closed:
                    return
            errors = []
            for index in reversed(range(len(self.additional_resources))):
                if self._additional_closed[index]:
                    continue
                resource = self.additional_resources[index]
                try:
                    resource.close()
                except Exception as exc:  # noqa: BLE001 - finish all cleanup stages.
                    errors.append(
                        f"additional_resource[{index}]:{type(exc).__name__}:{exc}"
                    )
                else:
                    self._additional_closed[index] = True
            errors.extend(self._close_cameras())
            if not self._artifact_store_closed:
                try:
                    self.artifact_store.close()
                except Exception as exc:  # noqa: BLE001 - cameras were already attempted.
                    errors.append(f"artifact_store:{type(exc).__name__}:{exc}")
                else:
                    self._artifact_store_closed = True
            with self._lock:
                self._closed = bool(
                    all(self._additional_closed)
                    and self._cameras_closed
                    and self._artifact_store_closed
                )
                complete = self._closed
            if not complete and not errors:
                errors.append("close_transaction_incomplete")
            if errors:
                raise RuntimeError(
                    "camera-only perception resource close failed: "
                    + "; ".join(errors)
                )


def start_realsense_sources(camera_sources: Mapping[str, Any]) -> None:
    """Compatibility name delegating to formal real-teleop startup."""

    _formal_start_camera_sources(camera_sources)


def admit_realsense_sources(camera_sources: Mapping[str, Any]) -> None:
    """Compatibility name delegating to formal real-teleop admission."""

    _formal_camera_startup_self_check(camera_sources)


class RealSenseAgentPerceptionProvider:
    """Read already-published frames; never starts devices or touches control."""

    def __init__(
        self,
        *,
        source_binding: RealPerceptionSourceBinding,
        current_station_revision: Callable[[], int],
        max_frame_age_ms: int = 200,
        max_inter_camera_skew_ms: int = 100,
        rgb_camera_geometry: RealRgbCameraGeometryProvider | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not isinstance(source_binding, RealPerceptionSourceBinding):
            raise TypeError("source_binding must be typed.")
        if not callable(current_station_revision) or not callable(clock):
            raise TypeError("revision and clock providers must be callable.")
        for field_name, value in (
            ("max_frame_age_ms", max_frame_age_ms),
            ("max_inter_camera_skew_ms", max_inter_camera_skew_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer.")
        self.source_binding = source_binding
        self._current_station_revision = current_station_revision
        self.max_frame_age_ms = max_frame_age_ms
        self.max_inter_camera_skew_ms = max_inter_camera_skew_ms
        if rgb_camera_geometry is not None and not isinstance(
            rgb_camera_geometry, RealRgbCameraGeometryProvider
        ):
            raise TypeError("rgb_camera_geometry must implement the trusted provider")
        self.rgb_camera_geometry = rgb_camera_geometry
        self._clock = clock
        self._profile: RealPerceptionCaptureProfile | None = None

    def rgb_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "source_id": source_id,
                "role": real_camera_role_for_source(source_id),
                "native_width": int(source.config.width),
                "native_height": int(source.config.height),
            }
            for source_id, source in self.source_binding.camera_sources.items()
        )

    def depth_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        return tuple(
            {
                "source_id": source_id,
                "role": real_camera_role_for_source(source_id),
                "native_width": int(
                    source.config.width
                    if source.config.align_depth_to_color
                    else source.config.depth_width
                ),
                "native_height": int(
                    source.config.height
                    if source.config.align_depth_to_color
                    else source.config.depth_height
                ),
            }
            for source_id, source in self.source_binding.camera_sources.items()
            if source.config.enable_depth
        )

    def aligned_rgbd_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        """Expose only static depth sources aligned to their color stream."""

        return tuple(
            item
            for item in self.depth_source_catalog()
            if self.source_binding.camera_sources[
                str(item["source_id"])
            ].config.align_depth_to_color
        )

    def rgb_geometry_source_catalog(self) -> tuple[Mapping[str, object], ...]:
        """Expose only RGB roles with deployment-owned camera geometry."""

        if self.rgb_camera_geometry is None:
            return self.aligned_rgbd_source_catalog()
        catalog = tuple(self.rgb_camera_geometry.source_catalog())
        rgb_by_source = {
            str(item["source_id"]): item for item in self.rgb_source_catalog()
        }
        if not catalog or any(
            str(item.get("source_id")) not in rgb_by_source
            or str(item.get("role"))
            != str(rgb_by_source[str(item.get("source_id"))]["role"])
            or int(item.get("native_width", -1))
            != int(rgb_by_source[str(item.get("source_id"))]["native_width"])
            or int(item.get("native_height", -1))
            != int(rgb_by_source[str(item.get("source_id"))]["native_height"])
            for item in catalog
        ):
            raise RuntimeError("RGB camera geometry catalog differs from live sources")
        return catalog

    def bind_profile(self, profile: RealPerceptionCaptureProfile) -> None:
        rgb_sources = tuple(
            sorted(
                (
                    (
                        item["source_id"],
                        item["role"],
                        item["native_width"],
                        item["native_height"],
                    )
                    for item in self.rgb_source_catalog()
                ),
                key=lambda item: item[1],
            )
        )
        if tuple(profile.rgb_sources) != rgb_sources:
            raise ValueError("Real RGB provider does not match the registered profile.")
        if tuple(profile.depth_sources) != tuple(
            sorted(
                (
                    (
                        item["source_id"],
                        item["role"],
                        item["native_width"],
                        item["native_height"],
                    )
                    for item in self.depth_source_catalog()
                ),
                key=lambda item: item[1],
            )
        ):
            raise ValueError(
                "Real depth provider does not match the registered profile."
            )
        if profile.source_binding_hash != self.source_binding.semantic_content_hash:
            raise ValueError("Real perception source binding hash changed.")
        self._profile = profile

    @staticmethod
    def _expected_revision(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("expected_station_revision must be non-negative.")
        return value

    def _frame(self, source_id: str, *, now: float):
        source = self.source_binding.camera_sources[source_id]
        frame = source.get_latest_frame()
        if frame is None:
            raise RuntimeError(f"real_perception_frame_unavailable:{source_id}")
        source_at = _finite_timestamp(
            frame.source_timestamp,
            field_name=f"{source_id}.source_timestamp",
        )
        available_at = _finite_timestamp(
            frame.available_timestamp,
            field_name=f"{source_id}.available_timestamp",
        )
        if source_at > now or available_at > now:
            raise RuntimeError(f"real_perception_frame_from_future:{source_id}")
        if available_at < source_at:
            raise RuntimeError(f"real_perception_frame_clock_order_invalid:{source_id}")
        if (now - source_at) * 1000.0 > self.max_frame_age_ms:
            raise RuntimeError(f"real_perception_frame_stale:{source_id}")
        info = source.device_info()
        if (
            info.get("running") is not True
            or info.get("lifecycle_state") != "running"
            or info.get("worker_alive") is not True
        ):
            raise RuntimeError(f"real_perception_source_not_running:{source_id}")
        serial = info.get("serial_number")
        if serial is None or str(serial) != str(source.config.serial):
            raise RuntimeError(f"real_perception_serial_mismatch:{source_id}")
        metadata = frame.metadata
        if not isinstance(metadata, Mapping):
            raise RuntimeError(  # noqa: TRY004 - runtime capture contract
                f"real_perception_frame_identity_missing:{source_id}"
            )
        if metadata.get("camera_key") != source_id or str(
            metadata.get("serial")
        ) != str(source.config.serial):
            raise RuntimeError(f"real_perception_frame_identity_mismatch:{source_id}")
        generation = info.get("generation")
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or metadata.get("source_generation") != generation
        ):
            raise RuntimeError(f"real_perception_frame_generation_mismatch:{source_id}")
        return frame, source_at, info

    def capture_rgb(
        self,
        parameters: Mapping[str, Any],
        *,
        expected_station_revision: int,
    ) -> RealRgbCapture:
        profile = self._profile
        if profile is None:
            raise RuntimeError("real_perception_provider_not_bound")
        fields = {"selected_roles", "width", "height", "max_total_bytes"}
        if not isinstance(parameters, Mapping) or set(parameters) != fields:
            raise ValueError("RGB capture parameters must use the exact schema fields.")
        expected_station_revision = self._expected_revision(expected_station_revision)
        roles_raw = parameters.get("selected_roles")
        if isinstance(roles_raw, (str, bytes)) or not isinstance(roles_raw, Sequence):
            raise TypeError("selected_roles must be an array.")
        roles = tuple(
            _required_text(value, field_name="selected_roles[]") for value in roles_raw
        )
        if not roles or len(roles) > 8 or len(set(roles)) != len(roles):
            raise ValueError("selected_roles must contain one to eight unique roles.")
        width = parameters["width"]
        height = parameters["height"]
        max_total_bytes = parameters["max_total_bytes"]
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                width,
                height,
                max_total_bytes,
            )
        ) or not (
            1 <= width <= 1920
            and 1 <= height <= 1080
            and 1 <= max_total_bytes <= profile.max_artifact_bytes
        ):
            raise ValueError("RGB capture dimensions or byte bound are invalid.")
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgb_capture_station_revision_mismatch")
        source_by_role = {
            real_camera_role_for_source(source_id): (source_id, source)
            for source_id, source in self.source_binding.camera_sources.items()
        }
        now = _finite_timestamp(self._clock(), field_name="capture clock")
        frames = []
        source_timestamps = []
        total_bytes = 0
        for role in roles:
            if role not in source_by_role:
                raise ValueError(f"RGB role {role!r} is not registered.")
            source_id, source = source_by_role[role]
            frame, source_at, _info = self._frame(source_id, now=now)
            image = np.asarray(frame.image)
            expected_shape = (int(source.config.height), int(source.config.width), 3)
            if image.shape != expected_shape or image.dtype != np.uint8:
                raise RuntimeError(f"real_rgb_frame_contract_mismatch:{source_id}")
            if source.config.color_space == "bgr":
                image = image[..., ::-1]
            encoded = _encode_rgb_jpeg(image, width=width, height=height)
            total_bytes += len(encoded)
            if total_bytes > max_total_bytes:
                raise RuntimeError("rgb_capture_artifact_limit_exceeded")
            frames.append(
                EncodedRealRgbFrame(
                    source_id=source_id,
                    role=role,
                    acquired_at=source_at,
                    width=width,
                    height=height,
                    data=encoded,
                )
            )
            source_timestamps.append(source_at)
        skew_ms = (max(source_timestamps) - min(source_timestamps)) * 1000.0
        if skew_ms > self.max_inter_camera_skew_ms:
            raise RuntimeError("rgb_capture_inter_camera_skew_exceeded")
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgb_capture_station_revision_changed")
        return RealRgbCapture(
            capture_window_started_at=min(source_timestamps),
            capture_window_ended_at=max(source_timestamps),
            station_revision=expected_station_revision,
            frames=tuple(frames),
        )

    def capture_depth(
        self,
        *,
        role: str,
        expected_station_revision: int,
    ) -> EncodedRealDepthFrame:
        profile = self._profile
        if profile is None:
            raise RuntimeError("real_perception_provider_not_bound")
        expected_station_revision = self._expected_revision(expected_station_revision)
        selected_role = _required_text(role, field_name="role")
        source_by_role = {
            real_camera_role_for_source(source_id): (source_id, source)
            for source_id, source in self.source_binding.camera_sources.items()
            if source.config.enable_depth
        }
        if selected_role not in source_by_role:
            raise ValueError("Depth role is not registered.")
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("depth_capture_station_revision_mismatch")
        source_id, source = source_by_role[selected_role]
        now = _finite_timestamp(self._clock(), field_name="capture clock")
        frame, source_at, info = self._frame(source_id, now=now)
        depth = np.asarray(frame.depth_image)
        metadata = frame.depth_metadata
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise RuntimeError(f"real_depth_frame_contract_mismatch:{source_id}")
        if (int(depth.shape[1]), int(depth.shape[0])) != (
            profile.depth_dimensions_by_role[selected_role]
        ):
            raise RuntimeError(f"real_depth_dimensions_changed:{source_id}")
        scale = metadata.get("depth_scale_m_per_unit")
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise RuntimeError(  # noqa: TRY004 - runtime capture contract
                f"real_depth_scale_unavailable:{source_id}"
            )
        scale = float(scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise RuntimeError(f"real_depth_scale_invalid:{source_id}")
        expected_alignment = "color" if source.config.align_depth_to_color else "none"
        if metadata.get("aligned_to") != expected_alignment:
            raise RuntimeError(f"real_depth_alignment_changed:{source_id}")
        aligned = expected_alignment == "color"
        intrinsics_raw = info.get("intrinsics" if aligned else "depth_intrinsics")
        intrinsics = _depth_intrinsics(intrinsics_raw, depth.shape)
        output = io.BytesIO()
        np.save(output, np.asarray(depth, dtype=np.float32) * scale, allow_pickle=False)
        world_from_camera = self.source_binding.world_from_depth_camera_by_source[
            source_id
        ]
        pose = tuple(
            float(value)
            for value in pose_matrix_to_xyzquat(
                world_from_camera,
                name=f"real_depth_camera_pose.{source_id}",
            )
        )
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("depth_capture_station_revision_changed")
        data = output.getvalue()
        if len(data) > profile.max_artifact_bytes:
            raise RuntimeError("depth_capture_artifact_limit_exceeded")
        return EncodedRealDepthFrame(
            source_id=source_id,
            role=selected_role,
            captured_at=source_at,
            width=int(depth.shape[1]),
            height=int(depth.shape[0]),
            camera_pose_world_xyz_quat_xyzw=pose,
            intrinsics=intrinsics,
            data=data,
        )

    def capture_aligned_rgbd(
        self,
        *,
        role: str,
        expected_station_revision: int,
    ) -> RealAlignedRgbdCapture:
        """Read RGB and aligned depth once from the same published VideoFrame."""

        profile = self._profile
        if profile is None:
            raise RuntimeError("real_perception_provider_not_bound")
        expected_station_revision = self._expected_revision(expected_station_revision)
        selected_role = _required_text(role, field_name="role")
        source_by_role = {
            real_camera_role_for_source(source_id): (source_id, source)
            for source_id, source in self.source_binding.camera_sources.items()
            if source.config.enable_depth and source.config.align_depth_to_color
        }
        if selected_role not in source_by_role:
            raise ValueError("Aligned RGB-D role is not registered.")
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgbd_capture_station_revision_mismatch")
        source_id, source = source_by_role[selected_role]
        now = _finite_timestamp(self._clock(), field_name="capture clock")
        frame, source_at, info = self._frame(source_id, now=now)
        rgb = np.asarray(frame.image)
        expected_rgb_shape = (
            int(source.config.height),
            int(source.config.width),
            3,
        )
        if rgb.shape != expected_rgb_shape or rgb.dtype != np.uint8:
            raise RuntimeError(f"real_rgb_frame_contract_mismatch:{source_id}")
        if source.config.color_space == "bgr":
            rgb = rgb[..., ::-1]
        rgb = np.ascontiguousarray(rgb).copy()
        depth_raw = np.asarray(frame.depth_image)
        metadata = frame.depth_metadata
        if depth_raw.ndim != 2 or depth_raw.dtype != np.uint16:
            raise RuntimeError(f"real_depth_frame_contract_mismatch:{source_id}")
        if depth_raw.shape != rgb.shape[:2]:
            raise RuntimeError(f"real_rgbd_alignment_dimensions_changed:{source_id}")
        if (int(depth_raw.shape[1]), int(depth_raw.shape[0])) != (
            profile.depth_dimensions_by_role[selected_role]
        ):
            raise RuntimeError(f"real_depth_dimensions_changed:{source_id}")
        if not isinstance(metadata, Mapping) or metadata.get("aligned_to") != "color":
            raise RuntimeError(f"real_depth_alignment_changed:{source_id}")
        scale = metadata.get("depth_scale_m_per_unit")
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(float(scale))
            or float(scale) <= 0.0
        ):
            raise RuntimeError(f"real_depth_scale_invalid:{source_id}")
        intrinsics = _depth_intrinsics(info.get("intrinsics"), depth_raw.shape)
        depth_m = np.ascontiguousarray(
            depth_raw.astype(np.float32) * float(scale),
            dtype=np.float32,
        )
        if not np.all(np.isfinite(depth_m)) or np.any(depth_m < 0.0):
            raise RuntimeError(f"real_depth_metric_values_invalid:{source_id}")
        generation = info.get("generation")
        frame_index = getattr(frame, "frame_index", None)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
        ):
            raise RuntimeError(f"real_perception_generation_invalid:{source_id}")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise RuntimeError(f"real_perception_frame_index_invalid:{source_id}")
        rgb_content_hash = canonical_content_hash(
            {
                "dtype": "uint8",
                "shape": list(rgb.shape),
                "byte_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
            },
            domain="dexteleop.real_aligned_rgbd.rgb_content.v1",
        )
        depth_content_hash = canonical_content_hash(
            {
                "dtype": "float32",
                "unit": "meter",
                "invalid_value": 0.0,
                "shape": list(depth_m.shape),
                "byte_sha256": hashlib.sha256(depth_m.tobytes(order="C")).hexdigest(),
            },
            domain="dexteleop.real_aligned_rgbd.depth_content.v1",
        )
        capture_hash = canonical_content_hash(
            {
                "source_binding_hash": self.source_binding.semantic_content_hash,
                "source_id": source_id,
                "role": selected_role,
                "serial": str(info.get("serial_number")),
                "source_generation": generation,
                "frame_index": frame_index,
                "acquired_at": source_at,
                "station_revision": expected_station_revision,
                "rgb_content_hash": rgb_content_hash,
                "depth_content_hash": depth_content_hash,
            },
            domain="dexteleop.real_aligned_rgbd.capture_group.v1",
        )
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgbd_capture_station_revision_changed")
        rgb.setflags(write=False)
        depth_m.setflags(write=False)
        source_from_camera = tuple(
            float(value)
            for value in self.source_binding.world_from_depth_camera_by_source[
                source_id
            ].reshape(-1)
        )
        return RealAlignedRgbdCapture(
            source_id=source_id,
            role=selected_role,
            acquired_at=source_at,
            station_revision=expected_station_revision,
            source_generation=generation,
            frame_index=frame_index,
            capture_group_id="real-aligned-rgbd:" + capture_hash,
            source_binding_hash=self.source_binding.semantic_content_hash,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            intrinsics=MappingProxyType(dict(intrinsics)),
            source_from_camera=source_from_camera,
            rgb=rgb,
            depth_m=depth_m,
            rgb_content_hash=rgb_content_hash,
            depth_content_hash=depth_content_hash,
        )

    def capture_rgb_geometry(
        self,
        *,
        role: str,
        expected_station_revision: int,
    ) -> RealRgbGeometryCapture:
        """Read one RGB frame and resolve its trusted static or measured pose."""

        profile = self._profile
        if profile is None:
            raise RuntimeError("real_perception_provider_not_bound")
        expected_station_revision = self._expected_revision(expected_station_revision)
        selected_role = _required_text(role, field_name="role")
        catalog = self.rgb_geometry_source_catalog()
        source_by_role = {str(item["role"]): str(item["source_id"]) for item in catalog}
        if selected_role not in source_by_role:
            raise ValueError("RGB geometry role is not registered.")
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgb_geometry_capture_station_revision_mismatch")
        source_id = source_by_role[selected_role]
        source = self.source_binding.camera_sources[source_id]
        now = _finite_timestamp(self._clock(), field_name="capture clock")
        frame, source_at, info = self._frame(source_id, now=now)
        rgb = np.asarray(frame.image)
        expected_shape = (int(source.config.height), int(source.config.width), 3)
        if rgb.shape != expected_shape or rgb.dtype != np.uint8:
            raise RuntimeError(f"real_rgb_frame_contract_mismatch:{source_id}")
        if source.config.color_space == "bgr":
            rgb = rgb[..., ::-1]
        rgb = np.ascontiguousarray(rgb).copy()
        intrinsics = _depth_intrinsics(info.get("intrinsics"), rgb.shape[:2])
        generation = info.get("generation")
        frame_index = getattr(frame, "frame_index", None)
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation < 0
            or isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise RuntimeError(f"real_rgb_frame_sequence_invalid:{source_id}")
        rgb_content_hash = canonical_content_hash(
            {
                "dtype": "uint8",
                "shape": list(rgb.shape),
                "byte_sha256": hashlib.sha256(rgb.tobytes(order="C")).hexdigest(),
            },
            domain="dexteleop.real_rgb_geometry.rgb_content.v1",
        )
        if self.rgb_camera_geometry is None:
            source_from_camera = self.source_binding.world_from_depth_camera_by_source[
                source_id
            ]
            geometry_binding_hash = self.source_binding.semantic_content_hash
            geometry_cut_hash = canonical_content_hash(
                {
                    "source_id": source_id,
                    "source_from_camera": source_from_camera.tolist(),
                    "source_binding_hash": geometry_binding_hash,
                },
                domain="dexteleop.real_rgb_geometry.static_cut.v1",
            )
            geometry = RealRgbCameraGeometryCut(
                source_id=source_id,
                source_from_camera=tuple(
                    float(v) for v in source_from_camera.reshape(-1)
                ),
                measured_at=source_at,
                available_at=source_at,
                geometry_binding_hash=geometry_binding_hash,
                geometry_cut_hash=geometry_cut_hash,
                evidence_refs=("real-rgb-static-camera:" + geometry_cut_hash,),
            )
        else:
            geometry = self.rgb_camera_geometry.pose_cut(
                source_id=source_id,
                frame_acquired_at=source_at,
                now=now,
            )
        capture_hash = canonical_content_hash(
            {
                "source_binding_hash": self.source_binding.semantic_content_hash,
                "source_id": source_id,
                "role": selected_role,
                "serial": str(info.get("serial_number")),
                "source_generation": generation,
                "frame_index": frame_index,
                "acquired_at": source_at,
                "station_revision": expected_station_revision,
                "rgb_content_hash": rgb_content_hash,
                "geometry_cut_hash": geometry.geometry_cut_hash,
            },
            domain="dexteleop.real_rgb_geometry.capture_group.v1",
        )
        if self._current_station_revision() != expected_station_revision:
            raise RuntimeError("rgb_geometry_capture_station_revision_changed")
        rgb.setflags(write=False)
        return RealRgbGeometryCapture(
            source_id=source_id,
            role=selected_role,
            acquired_at=source_at,
            station_revision=expected_station_revision,
            source_generation=generation,
            frame_index=frame_index,
            capture_group_id="real-rgb-geometry:" + capture_hash,
            source_binding_hash=self.source_binding.semantic_content_hash,
            width=int(rgb.shape[1]),
            height=int(rgb.shape[0]),
            intrinsics=MappingProxyType(dict(intrinsics)),
            source_from_camera=geometry.source_from_camera,
            rgb=rgb,
            rgb_content_hash=rgb_content_hash,
            geometry_binding_hash=geometry.geometry_binding_hash,
            geometry_cut_hash=geometry.geometry_cut_hash,
            pose_measured_at=geometry.measured_at,
            pose_available_at=geometry.available_at,
            pose_evidence_refs=geometry.evidence_refs,
        )


def _encode_rgb_jpeg(image: np.ndarray, *, width: int, height: int) -> bytes:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise RuntimeError("Real RGB source did not return uint8 HWC RGB.")
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = Image.fromarray(np.ascontiguousarray(image), mode="RGB").resize(
        (width, height),
        resample=resampling,
    )
    output = io.BytesIO()
    resized.save(output, format="JPEG", quality=85, optimize=True)
    return output.getvalue()


def _depth_intrinsics(
    payload: object,
    shape: tuple[int, ...],
) -> Mapping[str, float | int]:
    if not isinstance(payload, Mapping):
        raise RuntimeError(  # noqa: TRY004 - runtime capture contract
            "real_depth_intrinsics_unavailable"
        )
    if len(shape) != 2:
        raise RuntimeError("real_depth_shape_invalid")
    width, height = int(shape[1]), int(shape[0])
    values = {
        "width": int(payload.get("width", 0)),
        "height": int(payload.get("height", 0)),
        "fx": float(payload.get("fx", 0.0)),
        "fy": float(payload.get("fy", 0.0)),
        "cx": float(payload.get("cx", payload.get("ppx", -1.0))),
        "cy": float(payload.get("cy", payload.get("ppy", -1.0))),
    }
    if (
        (values["width"], values["height"]) != (width, height)
        or not all(math.isfinite(float(value)) for value in values.values())
        or values["fx"] <= 0.0
        or values["fy"] <= 0.0
        or values["cx"] < 0.0
        or values["cy"] < 0.0
    ):
        raise RuntimeError("real_depth_intrinsics_invalid")
    values["fovy_deg"] = math.degrees(
        2.0 * math.atan(height / (2.0 * float(values["fy"])))
    )
    return values


__all__ = [
    "CameraOnlyRealPerceptionResources",
    "EncodedRealDepthFrame",
    "EncodedRealRgbFrame",
    "ExclusiveRealPerceptionResources",
    "RealAlignedRgbdCapture",
    "RealPerceptionSourceBinding",
    "RealRgbCameraGeometryCut",
    "RealRgbCameraGeometryProvider",
    "RealRgbCapture",
    "RealRgbGeometryCapture",
    "RealSenseAgentPerceptionProvider",
    "admit_realsense_sources",
    "build_real_perception_source_binding",
    "real_camera_role_for_source",
    "start_realsense_sources",
]

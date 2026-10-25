"""Bounded native-resolution depth snapshots from one MuJoCo scene cut."""

from __future__ import annotations

import io
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
)

from .pointcloud import render_intrinsics_from_fovy
from .rgb_snapshot import EncodedMujocoRgbFrame, _encode_jpeg
from .scene import MujocoCameraRenderer
from .snapshot_renderer import SnapshotRendererCache


@dataclass(frozen=True)
class EncodedMujocoDepthFrame:
    """Depth image whose camera pose uses optical x-right/y-down/z-forward axes."""

    view_id: str
    captured_at: float
    width: int
    height: int
    media_type: str
    schema_id: str
    encoding: str
    compression: str
    data: bytes
    camera_pose_world_xyz_quat_xyzw: tuple[float, ...]
    intrinsics: Mapping[str, float | int]


@dataclass(frozen=True)
class MujocoRawRgbdView:
    """Native pixels and optical geometry from the same frozen scene cut."""

    view_id: str
    rgb: np.ndarray
    depth_m: np.ndarray
    optical_camera_pose_world_xyz_quat_xyzw: tuple[float, ...]
    intrinsics: Mapping[str, float | int]

    def __post_init__(self):
        if (
            not isinstance(self.view_id, str)
            or not self.view_id
            or self.view_id != self.view_id.strip()
        ):
            raise ValueError("RGB-D view ID must be a nonempty trimmed string.")
        rgb = np.asarray(self.rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2:] != (3,):
            raise RuntimeError(
                "MuJoCo RGB-D color source must return native RGB uint8."
            )
        if not 0 < rgb.shape[0] * rgb.shape[1] <= 1920 * 1080:
            raise RuntimeError(
                "MuJoCo RGB-D frame dimensions exceed the capture bound."
            )
        depth = _normalized_depth(self.depth_m, width=rgb.shape[1], height=rgb.shape[0])
        rgb = np.ascontiguousarray(rgb).copy()
        rgb.setflags(write=False)
        depth.setflags(write=False)
        pose = tuple(float(v) for v in self.optical_camera_pose_world_xyz_quat_xyzw)
        if (
            len(pose) != 7
            or not all(math.isfinite(v) for v in pose)
            or abs(math.sqrt(sum(v * v for v in pose[3:])) - 1) > 1e-5
        ):
            raise RuntimeError(
                "MuJoCo RGB-D optical pose must be finite and normalized."
            )
        intrinsics = dict(self.intrinsics)
        if (
            intrinsics.get("width") != rgb.shape[1]
            or intrinsics.get("height") != rgb.shape[0]
            or any(
                not math.isfinite(intrinsics.get(n, float("nan")))
                for n in ("fx", "fy", "cx", "cy")
            )
            or intrinsics["fx"] <= 0
            or intrinsics["fy"] <= 0
        ):
            raise RuntimeError("MuJoCo RGB-D intrinsics do not match native pixels.")
        object.__setattr__(self, "rgb", rgb)
        object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "optical_camera_pose_world_xyz_quat_xyzw", pose)
        object.__setattr__(self, "intrinsics", MappingProxyType(intrinsics))



@dataclass(frozen=True)
class MujocoRawRgbdCut:
    captured_at: float
    simulation_time_sec: float
    model_qpos: np.ndarray
    model_qvel: np.ndarray
    arm_qpos_by_side: Mapping[str, np.ndarray]
    views: tuple[MujocoRawRgbdView, ...]

    def __post_init__(self):
        if not math.isfinite(self.captured_at) or not math.isfinite(
            self.simulation_time_sec
        ):
            raise RuntimeError("MuJoCo RGB-D capture clock is non-finite.")
        for name in ("model_qpos", "model_qvel"):
            vector = np.asarray(getattr(self, name), dtype=float).copy()
            if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
                raise RuntimeError("MuJoCo RGB-D capture requires finite model state.")
            vector.setflags(write=False)
            object.__setattr__(self, name, vector)
        pair = {}
        if set(self.arm_qpos_by_side) != {"left", "right"}:
            raise RuntimeError("MuJoCo RGB-D requires both measured arm states.")
        for side, values in self.arm_qpos_by_side.items():
            q = np.asarray(values, dtype=float).copy()
            if q.shape != (6,) or not np.isfinite(q).all():
                raise RuntimeError("MuJoCo RGB-D arms require six finite radians.")
            q.setflags(write=False)
            pair[side] = q
        views = tuple(self.views)
        if (
            not 1 <= len(views) <= 3
            or any(not isinstance(v, MujocoRawRgbdView) for v in views)
            or len({v.view_id for v in views}) != len(views)
        ):
            raise ValueError(
                "MuJoCo RGB-D requires one to three distinct native views."
            )
        object.__setattr__(self, "arm_qpos_by_side", MappingProxyType(pair))
        object.__setattr__(self, "views", views)



class MujocoDepthSnapshotSource:
    """Own renderer resources for passive depth capture only."""

    def __init__(
        self,
        *,
        capture_scene: Callable[[Callable[[Any], Any]], Any],
        clock: Callable[[], float],
        renderer_factory: Callable[..., Any] = MujocoCameraRenderer,
        max_renderer_dimensions: int = 4,
    ) -> None:
        if not callable(capture_scene) or not callable(clock):
            raise TypeError("capture_scene and clock must be callable.")
        if not callable(renderer_factory):
            raise TypeError("renderer_factory must be callable.")
        if (
            isinstance(max_renderer_dimensions, bool)
            or not isinstance(max_renderer_dimensions, int)
            or max_renderer_dimensions < 1
        ):
            raise ValueError("max_renderer_dimensions must be positive.")
        self._capture_scene = capture_scene
        self._clock = clock
        self._renderer_cache = SnapshotRendererCache(
            renderer_factory, max_renderer_dimensions
        )
        self._lock = threading.RLock()
        self._closed = False

    def source_catalog(self) -> tuple[Mapping[str, object], ...]:
        def capture(scene: Any) -> tuple[Mapping[str, object], ...]:
            metadata = scene.camera_metadata()
            return tuple(
                {
                    "view_id": name,
                    "native_width": int(metadata[name]["width"]),
                    "native_height": int(metadata[name]["height"]),
                }
                for name in scene.camera_names
            )

        with self._lock:
            self._ensure_open()
            return self._capture_scene(capture)

    def capture_npy_frame(self, *, view_id: str) -> EncodedMujocoDepthFrame:
        _validate_view_id(view_id)

        def render(scene: Any) -> tuple[np.ndarray, Mapping[str, object], float]:
            rendered, metadata, captured_at = self._render_registered_view(
                scene,
                view_id=view_id,
                render=lambda renderer: renderer.render_depth(view_id),
            )
            return rendered, metadata, captured_at

        with self._lock:
            self._ensure_open()
            raw_depth, metadata, captured_at = self._capture_scene(render)
        return _encoded_depth_frame(
            view_id=view_id,
            raw_depth=raw_depth,
            metadata=metadata,
            captured_at=captured_at,
        )

    def capture_jpeg_npy_cut(
        self,
        *,
        view_id: str,
    ) -> tuple[EncodedMujocoRgbFrame, EncodedMujocoDepthFrame]:
        """Render registered RGB and depth inside one scene-capture callback."""

        _validate_view_id(view_id)

        def render(
            scene: Any,
        ) -> tuple[np.ndarray, np.ndarray, Mapping[str, object], float]:
            rendered, metadata, captured_at = self._render_registered_view(
                scene,
                view_id=view_id,
                render=lambda renderer: (
                    renderer.render(view_id),
                    renderer.render_depth(view_id),
                ),
            )
            raw_rgb, raw_depth = rendered
            return (
                raw_rgb,
                raw_depth,
                metadata,
                captured_at,
            )

        with self._lock:
            self._ensure_open()
            raw_rgb, raw_depth, metadata, captured_at = self._capture_scene(render)
        depth = _encoded_depth_frame(
            view_id=view_id,
            raw_depth=raw_depth,
            metadata=metadata,
            captured_at=captured_at,
        )
        return (
            EncodedMujocoRgbFrame(
                source_id=view_id,
                width=depth.width,
                height=depth.height,
                media_type=RGB_FRAME_MEDIA_TYPE_V1,
                data=_encode_jpeg(raw_rgb),
            ),
            depth,
        )

    def capture_rgbd_cut(
        self,
        *,
        view_ids: tuple[str, ...],
        require_capture_state: Callable[[], None],
    ) -> MujocoRawRgbdCut:
        """Copy RGB, metric depth, camera geometry and qpos under one scene lock.

        Reuse this source's existing bounded renderer cache and close owner.
        The application supplies its state guard; this method never starts or
        advances a simulator and never infers stationarity from frame timestamps.
        """
        view_ids = tuple(view_ids)
        if (
            not 1 <= len(view_ids) <= 3
            or len(set(view_ids)) != len(view_ids)
            or any(not isinstance(v, str) or not v or v != v.strip() for v in view_ids)
        ):
            raise ValueError("RGB-D requires one to three distinct native view IDs.")
        if not callable(require_capture_state):
            raise TypeError("require_capture_state must be callable.")

        def capture(scene):
            require_capture_state()
            if any(v not in scene.camera_names for v in view_ids):
                raise ValueError("Unknown native RGB-D view.")
            # Fail before entering renderer/native code on an invalid plant.
            qpos, qvel = scene.data.qpos.copy(), scene.data.qvel.copy()
            stamp = float(scene.data.time)
            if (
                not np.isfinite(qpos).all()
                or not np.isfinite(qvel).all()
                or not math.isfinite(stamp)
            ):
                raise RuntimeError("MuJoCo RGB-D capture requires finite model state.")
            pair = {side: q.copy() for side, q in scene.arm_qpos_by_side().items()}
            metadata = scene.camera_metadata()
            if any(v not in metadata for v in view_ids):
                raise ValueError("Unknown native RGB-D view.")
            views = []
            for view_id in view_ids:
                require_capture_state()
                meta = metadata[view_id]
                width, height = int(meta["width"]), int(meta["height"])
                if not 0 < width * height <= 1920 * 1080 or width <= 0 or height <= 0:
                    raise RuntimeError(
                        "MuJoCo RGB-D frame dimensions exceed the capture bound."
                    )
                rgb, depth = self._renderer_cache.render(
                    scene, width=width, height=height,
                    draw=lambda renderer: (renderer.render(view_id), renderer.render_depth(view_id)),
                )
                intrinsics = render_intrinsics_from_fovy(
                    width=width, height=height, fovy_deg=float(meta["fovy_deg"])
                )
                views.append(
                    MujocoRawRgbdView(
                        view_id=view_id,
                        rgb=rgb,
                        depth_m=depth,
                        optical_camera_pose_world_xyz_quat_xyzw=meta[
                            "optical_camera_pose_world_xyz_quat_xyzw"
                        ],
                        intrinsics={
                            "width": width,
                            "height": height,
                            "fx": intrinsics.fx,
                            "fy": intrinsics.fy,
                            "cx": intrinsics.cx,
                            "cy": intrinsics.cy,
                            "fovy_deg": intrinsics.fovy_deg,
                        },
                    )
                )
            require_capture_state()
            if (
                not np.array_equal(qpos, scene.data.qpos)
                or not np.array_equal(qvel, scene.data.qvel)
                or stamp != float(scene.data.time)
            ):
                raise RuntimeError("MuJoCo scene changed during atomic RGB-D capture.")
            return MujocoRawRgbdCut(
                captured_at=float(self._clock()),
                simulation_time_sec=stamp,
                model_qpos=qpos,
                model_qvel=qvel,
                arm_qpos_by_side=pair,
                views=tuple(views),
            )

        with self._lock:
            self._ensure_open()
            return self._capture_scene(capture)


    def _render_registered_view(
        self,
        scene: Any,
        *,
        view_id: str,
        render: Callable[[Any], Any],
    ) -> tuple[Any, Mapping[str, object], float]:
        if view_id not in frozenset(scene.camera_names):
            raise ValueError(f"unknown depth view: {view_id!r}")
        metadata = scene.camera_metadata()[view_id]
        key = (int(metadata["width"]), int(metadata["height"]))
        rendered = self._renderer_cache.render(
            scene, width=key[0], height=key[1], draw=render,
        )
        return rendered, metadata, float(self._clock())

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._renderer_cache.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuJoCo depth snapshot source is closed.")


def _encoded_depth_frame(
    *,
    view_id: str,
    raw_depth: object,
    metadata: Mapping[str, object],
    captured_at: float,
) -> EncodedMujocoDepthFrame:
    if not math.isfinite(captured_at):
        raise RuntimeError("MuJoCo depth capture clock is non-finite.")
    width = int(metadata["width"])
    height = int(metadata["height"])
    depth = _normalized_depth(raw_depth, width=width, height=height)
    render_pose = tuple(
        float(item) for item in metadata["camera_pose_world_xyz_quat_xyzw"]
    )
    if len(render_pose) != 7 or not all(math.isfinite(item) for item in render_pose):
        raise RuntimeError("MuJoCo depth camera pose is invalid.")
    quaternion_norm = math.sqrt(sum(item * item for item in render_pose[3:]))
    if abs(quaternion_norm - 1.0) > 1e-5:
        raise RuntimeError("MuJoCo depth camera quaternion is not normalized.")
    # MJCF stores the render camera at optical->OpenGL Rx(pi). Depth pixels
    # use optical axes (+x right, +y down, +z forward), so undo that local
    # rotation when publishing the camera pose: T_world_optical is
    # T_world_render * Rx(pi). Quaternion composition is right multiplication.
    x, y, z, w = render_pose[3:]
    pose = (*render_pose[:3], w, z, -y, -x)
    intrinsics = render_intrinsics_from_fovy(
        width=width,
        height=height,
        fovy_deg=float(metadata["fovy_deg"]),
    )
    payload = io.BytesIO()
    np.save(payload, depth, allow_pickle=False)
    return EncodedMujocoDepthFrame(
        view_id=view_id,
        captured_at=captured_at,
        width=width,
        height=height,
        media_type=DEPTH_FRAME_MEDIA_TYPE_V1,
        schema_id=DEPTH_FRAME_SCHEMA_ID_V1,
        encoding=DEPTH_FRAME_ENCODING_V1,
        compression=DEPTH_FRAME_COMPRESSION_V1,
        data=payload.getvalue(),
        camera_pose_world_xyz_quat_xyzw=pose,
        intrinsics={
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "fovy_deg": intrinsics.fovy_deg,
        },
    )


def _validate_view_id(view_id: object) -> None:
    if not isinstance(view_id, str) or not view_id or view_id != view_id.strip():
        raise ValueError("view_id must be a non-empty trimmed string.")


def _normalized_depth(image: object, *, width: int, height: int) -> np.ndarray:
    depth = np.asarray(image, dtype=np.float32)
    if depth.shape != (height, width):
        raise RuntimeError("MuJoCo depth source returned an unexpected shape.")
    normalized = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    return np.ascontiguousarray(normalized, dtype=np.float32)


__all__ = [
    "EncodedMujocoDepthFrame",
    "MujocoDepthSnapshotSource",
    "MujocoRawRgbdCut",
    "MujocoRawRgbdView",
]

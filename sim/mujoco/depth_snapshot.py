"""Bounded native-resolution depth snapshots from one MuJoCo scene cut."""

from __future__ import annotations

import io
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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
    pose = tuple(float(item) for item in metadata["camera_pose_world_xyz_quat_xyzw"])
    if len(pose) != 7 or not all(math.isfinite(item) for item in pose):
        raise RuntimeError("MuJoCo depth camera pose is invalid.")
    quaternion_norm = math.sqrt(sum(item * item for item in pose[3:]))
    if abs(quaternion_norm - 1.0) > 1e-5:
        raise RuntimeError("MuJoCo depth camera quaternion is not normalized.")
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
]

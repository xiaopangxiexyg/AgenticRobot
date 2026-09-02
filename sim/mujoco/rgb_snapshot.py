"""Bounded RGB snapshot capture for a MuJoCo scene.

Rendering is performed inside the caller-provided scene capture boundary.  RGB
validation and JPEG encoding happen after that boundary returns so image
compression does not extend the physics-owner critical section.
"""

from __future__ import annotations

import io
import math
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from perception_contracts.artifacts import RGB_FRAME_MEDIA_TYPE_V1

from .scene import MujocoCameraRenderer


@dataclass(frozen=True)
class EncodedMujocoRgbFrame:
    """One validated, encoded frame from a named MuJoCo camera."""

    source_id: str
    width: int
    height: int
    media_type: str
    data: bytes


@dataclass(frozen=True)
class EncodedMujocoRgbSnapshot:
    """Frames from one scene cut plus its lower runtime timestamp."""

    captured_at: float
    frames: Mapping[str, EncodedMujocoRgbFrame]


class MujocoRgbSnapshotSource:
    """Own bounded renderer resources for on-demand MuJoCo RGB snapshots."""

    def __init__(
        self,
        *,
        capture_scene: Callable[[Callable[[Any], Any]], Any],
        clock: Callable[[], float] = time.monotonic,
        renderer_factory: Callable[..., Any] = MujocoCameraRenderer,
        max_renderer_dimensions: int = 4,
    ) -> None:
        if not callable(capture_scene):
            raise TypeError("capture_scene must be callable.")
        if not callable(clock):
            raise TypeError("clock must be callable.")
        if not callable(renderer_factory):
            raise TypeError("renderer_factory must be callable.")
        if (
            isinstance(max_renderer_dimensions, bool)
            or not isinstance(max_renderer_dimensions, int)
            or max_renderer_dimensions < 1
        ):
            raise ValueError("max_renderer_dimensions must be a positive integer.")
        self._capture_scene = capture_scene
        self._clock = clock
        self._renderer_factory = renderer_factory
        self._max_renderer_dimensions = max_renderer_dimensions
        self._renderers: dict[tuple[int, int], Any] = {}
        self._lock = threading.RLock()
        self._closed = False

    def source_catalog(self) -> tuple[Mapping[str, object], ...]:
        """Describe currently available cameras without rendering a frame."""

        def capture(scene: Any) -> tuple[Mapping[str, object], ...]:
            metadata = scene.camera_metadata()
            return tuple(
                {
                    "source_id": name,
                    "native_width": int(metadata[name]["width"]),
                    "native_height": int(metadata[name]["height"]),
                }
                for name in scene.camera_names
            )

        with self._lock:
            self._ensure_open()
            return self._capture_scene(capture)

    def capture_jpeg_snapshot(
        self,
        *,
        source_ids: tuple[str, ...],
        width: int,
        height: int,
    ) -> EncodedMujocoRgbSnapshot:
        """Render one scene cut and encode it after releasing the capture lock."""

        _validate_request(source_ids=source_ids, width=width, height=height)

        def render(scene: Any) -> tuple[Mapping[str, Any], float]:
            available = frozenset(scene.camera_names)
            missing = tuple(name for name in source_ids if name not in available)
            if missing:
                raise ValueError(f"unknown visual sources: {list(missing)}")
            key = (width, height)
            renderer = self._renderers.get(key)
            if renderer is None:
                if len(self._renderers) >= self._max_renderer_dimensions:
                    raise RuntimeError("visual renderer dimension cache is full.")
                renderer = self._renderer_factory(
                    scene,
                    width=width,
                    height=height,
                )
                self._renderers[key] = renderer
            return renderer.render_all(source_ids), float(self._clock())

        with self._lock:
            self._ensure_open()
            rendered, captured_at = self._capture_scene(render)

        if not isinstance(rendered, Mapping):
            raise TypeError("MuJoCo renderer did not return a frame mapping.")
        if not math.isfinite(captured_at):
            raise RuntimeError("MuJoCo RGB capture clock is non-finite.")
        encoded: dict[str, EncodedMujocoRgbFrame] = {}
        for source_id in source_ids:
            if source_id not in rendered:
                raise RuntimeError(
                    f"MuJoCo renderer omitted visual source {source_id!r}."
                )
            data = _encode_jpeg(rendered[source_id])
            encoded[source_id] = EncodedMujocoRgbFrame(
                source_id=source_id,
                width=width,
                height=height,
                media_type=RGB_FRAME_MEDIA_TYPE_V1,
                data=data,
            )
        return EncodedMujocoRgbSnapshot(
            captured_at=captured_at,
            frames=encoded,
        )

    def close(self) -> None:
        """Close every cached renderer. Repeated calls are harmless."""

        with self._lock:
            if self._closed:
                return
            renderers = tuple(self._renderers.values())
            self._renderers.clear()
            self._closed = True
        first_error: Exception | None = None
        for renderer in renderers:
            try:
                renderer.close()
            except Exception as exc:  # noqa: BLE001  # pragma: no cover - backend-specific cleanup
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuJoCo RGB snapshot source is closed.")


def _validate_request(
    *,
    source_ids: tuple[str, ...],
    width: int,
    height: int,
) -> None:
    if not source_ids or len(source_ids) > 8:
        raise ValueError("source_ids must contain between one and eight entries.")
    if any(not isinstance(item, str) or not item for item in source_ids):
        raise ValueError("source_ids must contain non-empty strings.")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source_ids must not contain duplicates.")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or not 1 <= width <= 1920
        or isinstance(height, bool)
        or not isinstance(height, int)
        or not 1 <= height <= 1080
    ):
        raise ValueError("visual dimensions exceed the bounded RGB contract.")


def _encode_jpeg(image: object) -> bytes:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise RuntimeError("MuJoCo visual source did not return uint8 HWC RGB.")
    output = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(
        output,
        format="JPEG",
        quality=85,
        optimize=True,
    )
    return output.getvalue()


__all__ = [
    "EncodedMujocoRgbFrame",
    "EncodedMujocoRgbSnapshot",
    "MujocoRgbSnapshotSource",
]

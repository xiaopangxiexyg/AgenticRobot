"""Neutral immutable camera-frame history contracts.

This owner is intentionally outside ``teleop_runtime.collection`` so camera
sources and independent deployments do not initialize the collection graph.
The historical collection module re-exports these exact objects.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class VideoFrame:
    image: Any
    source_timestamp: float
    receive_timestamp: float | None = None
    available_timestamp: float | None = None
    frame_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    depth_image: Any | None = None
    depth_metadata: dict[str, Any] = field(default_factory=dict)


class NearestFrameBuffer:
    """Keep a bounded frame history for causal timestamp matching."""

    def __init__(self, *, keep_last: int = 2):
        if int(keep_last) <= 0:
            raise ValueError("keep_last must be positive.")
        self._frames: deque[VideoFrame] = deque(maxlen=int(keep_last))
        self._lock = threading.Lock()

    def push(self, frame: VideoFrame) -> None:
        with self._lock:
            if frame.available_timestamp is None:
                frame = replace(frame, available_timestamp=float(time.time()))
            self._frames.append(frame)

    def publish(self, frame: VideoFrame) -> VideoFrame:
        with self._lock:
            self._frames.append(frame)
            published_at = time.time()
            if frame.available_timestamp is not None:
                try:
                    published_at = max(published_at, float(frame.available_timestamp))
                except (TypeError, ValueError):
                    pass
            published = replace(frame, available_timestamp=float(published_at))
            self._frames[-1] = published
            return published

    def get_latest(self) -> VideoFrame | None:
        with self._lock:
            return None if not self._frames else self._frames[-1]

    def get_nearest(self, timestamp: float) -> VideoFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return min(
                self._frames,
                key=lambda frame: abs(float(frame.source_timestamp) - float(timestamp)),
            )

    def get_latest_before_or_at(self, timestamp: float) -> VideoFrame | None:
        target = float(timestamp)
        if not math.isfinite(target):
            raise ValueError("timestamp must be finite.")
        with self._lock:
            for frame in reversed(self._frames):
                try:
                    source_value = float(frame.source_timestamp)
                    available_value = float(frame.available_timestamp)
                except (TypeError, ValueError):
                    continue
                if (
                    math.isfinite(source_value)
                    and math.isfinite(available_value)
                    and source_value <= target
                    and available_value <= target
                ):
                    return frame
            return None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)


__all__ = ["NearestFrameBuffer", "VideoFrame"]

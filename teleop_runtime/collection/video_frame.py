from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class VideoFrame:
    image: Any
    source_timestamp: float
    receive_timestamp: float | None = None
    frame_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    depth_image: Any | None = None
    depth_metadata: dict[str, Any] = field(default_factory=dict)


class NearestFrameBuffer:
    """Keeps a tiny sliding window of camera frames for timestamped sample matching."""

    def __init__(self, *, keep_last: int = 2):
        if int(keep_last) <= 0:
            raise ValueError("keep_last must be positive.")
        self._frames: deque[VideoFrame] = deque(maxlen=int(keep_last))
        self._lock = threading.Lock()

    def push(self, frame: VideoFrame) -> None:
        with self._lock:
            self._frames.append(frame)

    def get_latest(self) -> VideoFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return self._frames[-1]

    def get_nearest(self, timestamp: float) -> VideoFrame | None:
        with self._lock:
            if not self._frames:
                return None
            return min(
                self._frames,
                key=lambda frame: abs(float(frame.source_timestamp) - float(timestamp)),
            )

    def get_latest_before_or_at(self, timestamp: float) -> VideoFrame | None:
        """Return the newest frame whose source timestamp does not look ahead."""
        target = float(timestamp)
        with self._lock:
            for frame in reversed(self._frames):
                if float(frame.source_timestamp) <= target:
                    return frame
            return None

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

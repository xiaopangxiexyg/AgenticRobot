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


class NearestFrameBuffer:
    """Keeps only a tiny sliding window so each sample can select the nearest frame."""

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

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

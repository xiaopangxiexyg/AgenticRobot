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
    # Local wall time at which the immutable frame became visible in the
    # causal history.  Producers may omit it before push(); the buffer stamps
    # local publication in that case.
    available_timestamp: float | None = None
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
        """Append a frame, stamping local availability when it is absent."""
        with self._lock:
            if frame.available_timestamp is None:
                frame = replace(frame, available_timestamp=float(time.time()))
            self._frames.append(frame)

    def publish(self, frame: VideoFrame) -> VideoFrame:
        """Publish ``frame`` and return the immutable, availability-stamped copy."""

        with self._lock:
            # Append first, then stamp/replace while holding the same lock.  A
            # causal reader can therefore never observe the unstamped entry,
            # and available_timestamp denotes completed local publication.
            self._frames.append(frame)
            published_at = time.time()
            upstream_available = frame.available_timestamp
            if upstream_available is not None:
                try:
                    published_at = max(published_at, float(upstream_available))
                except (TypeError, ValueError):
                    pass
            published = replace(frame, available_timestamp=float(published_at))
            self._frames[-1] = published
            return published

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
        """Return the newest frame satisfying both causal timestamp bounds."""
        target = float(timestamp)
        if not math.isfinite(target):
            raise ValueError("timestamp must be finite.")
        with self._lock:
            for frame in reversed(self._frames):
                available_timestamp = frame.available_timestamp
                try:
                    source_value = float(frame.source_timestamp)
                    available_value = float(available_timestamp)
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

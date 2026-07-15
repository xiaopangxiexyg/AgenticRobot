from __future__ import annotations

import math
import threading
from collections import deque

from .contracts import PositionReferenceFrame


class OutOfOrderReferenceError(ValueError):
    """Raised when an old or duplicate reference is republished in an epoch."""


class ReferenceTimeline:
    """Thread-safe bounded history used for causal cross-rate sampling.

    Selection is based on ``published_wall_time`` because dataset sample cuts
    use wall timestamps. Publication order is protected independently by epoch,
    sequence, and monotonic timestamps, so a wall-clock correction is allowed.
    """

    def __init__(self, max_frames: int = 256):
        if isinstance(max_frames, bool) or int(max_frames) != max_frames or int(max_frames) <= 0:
            raise ValueError("max_frames must be a positive integer")
        self._max_frames = int(max_frames)
        self._frames: deque[PositionReferenceFrame] = deque(maxlen=self._max_frames)
        self._high_water: dict[int, tuple[int, float]] = {}
        self._lock = threading.RLock()

    @property
    def max_frames(self) -> int:
        return self._max_frames

    def __len__(self) -> int:
        with self._lock:
            return len(self._frames)

    def publish(self, frame: PositionReferenceFrame) -> None:
        if not isinstance(frame, PositionReferenceFrame):
            raise TypeError("frame must be a PositionReferenceFrame")
        with self._lock:
            high_water = self._high_water.get(frame.epoch)
            if high_water is not None:
                last_sequence, last_published_monotonic = high_water
                if frame.sequence <= last_sequence:
                    raise OutOfOrderReferenceError(
                        f"epoch {frame.epoch} reference sequence {frame.sequence} is not newer "
                        f"than {last_sequence}"
                    )
                if frame.published_monotonic_time < last_published_monotonic:
                    raise OutOfOrderReferenceError(
                        f"epoch {frame.epoch} published_monotonic_time "
                        f"{frame.published_monotonic_time} is older than "
                        f"{last_published_monotonic}"
                    )
            self._frames.append(frame)
            self._high_water[frame.epoch] = (
                frame.sequence,
                frame.published_monotonic_time,
            )

    def latest_at_or_before(
        self,
        sample_time: float,
        epoch: int,
        max_age: float | None,
    ) -> PositionReferenceFrame | None:
        """Return the newest valid frame published no later than ``sample_time``.

        ``sample_time`` and ``max_age`` are wall-time seconds. Invalid frames are
        skipped, matching the action-label contract ``latest_valid(FK(q_ref))``.
        ``epoch`` is mandatory so a new session can never consume an old frame.
        """

        sample = float(sample_time)
        if not math.isfinite(sample):
            raise ValueError("sample_time must be finite")
        if isinstance(epoch, bool) or int(epoch) != epoch or int(epoch) < 0:
            raise ValueError("epoch must be a non-negative integer")
        normalized_epoch = int(epoch)
        if max_age is None:
            normalized_max_age = None
        else:
            normalized_max_age = float(max_age)
            if not math.isfinite(normalized_max_age) or normalized_max_age < 0.0:
                raise ValueError("max_age must be finite and >= 0, or None")

        with self._lock:
            candidates = tuple(self._frames)
        selected: PositionReferenceFrame | None = None
        for frame in candidates:
            if frame.epoch != normalized_epoch or not frame.valid:
                continue
            if frame.published_wall_time > sample:
                continue
            # Wall time is only the causal eligibility gate. Once eligible,
            # producer order is authoritative so an NTP/manual wall-clock
            # rewind cannot make an older sequence look newer.
            if selected is None or (
                frame.published_monotonic_time,
                frame.sequence,
            ) > (
                selected.published_monotonic_time,
                selected.sequence,
            ):
                selected = frame
        if selected is None:
            return None
        age = sample - selected.published_wall_time
        if normalized_max_age is not None and age > normalized_max_age:
            return None
        return selected

    def snapshot(self, *, epoch: int | None = None) -> tuple[PositionReferenceFrame, ...]:
        with self._lock:
            frames = tuple(self._frames)
        if epoch is None:
            return frames
        return tuple(frame for frame in frames if frame.epoch == int(epoch))

    def clear(self, *, epoch: int | None = None) -> None:
        with self._lock:
            if epoch is None:
                self._frames.clear()
                self._high_water.clear()
                return
            normalized_epoch = int(epoch)
            self._frames = deque(
                (frame for frame in self._frames if frame.epoch != normalized_epoch),
                maxlen=self._max_frames,
            )
            self._high_water.pop(normalized_epoch, None)


__all__ = ["OutOfOrderReferenceError", "ReferenceTimeline"]

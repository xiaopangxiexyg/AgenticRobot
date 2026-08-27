"""Clock-domain measurements used by B601 hardware commissioning.

The meter does not infer a control rate from one aggregate CAN stream.  Callers
record independent ``feedback``, ``compute``, ``tx`` and ``haptic`` events and
inspect each series separately.  This keeps a 440 Hz mixed bus observation
from being misreported as a 500 Hz command or haptic loop.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
import time


_EVENT_NAMES = ("feedback", "compute", "tx", "haptic")


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] + weight * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class FrequencyStats:
    """Summary for one independently measured event clock."""

    event: str
    sample_count: int
    elapsed_sec: float | None
    observed_hz: float | None
    p50_interval_sec: float | None
    p95_interval_sec: float | None
    p99_interval_sec: float | None
    max_interval_sec: float | None
    deadline_misses: int
    target_hz: float | None
    deadline_factor: float

    def as_dict(self) -> dict[str, object]:
        return {
            "event": self.event,
            "sample_count": self.sample_count,
            "elapsed_sec": self.elapsed_sec,
            "observed_hz": self.observed_hz,
            "p50_interval_sec": self.p50_interval_sec,
            "p95_interval_sec": self.p95_interval_sec,
            "p99_interval_sec": self.p99_interval_sec,
            "max_interval_sec": self.max_interval_sec,
            "deadline_misses": self.deadline_misses,
            "target_hz": self.target_hz,
            "deadline_factor": self.deadline_factor,
        }


def summarize_timestamps(
    event: str,
    timestamps_sec: Iterable[float],
    *,
    target_hz: float | None = None,
    deadline_factor: float = 2.0,
) -> FrequencyStats:
    """Summarize a monotonic event series without requiring NumPy."""

    normalized_event = str(event).strip()
    if normalized_event not in _EVENT_NAMES:
        raise ValueError(f"event must be one of {_EVENT_NAMES}")
    timestamps = [_finite(value, name="timestamp") for value in timestamps_sec]
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("timestamps must be monotonic")
    if target_hz is not None:
        target_hz = _finite(target_hz, name="target_hz")
        if target_hz <= 0.0:
            raise ValueError("target_hz must be positive")
    deadline_factor = _finite(deadline_factor, name="deadline_factor")
    if deadline_factor <= 0.0:
        raise ValueError("deadline_factor must be positive")
    intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
    elapsed = timestamps[-1] - timestamps[0] if len(timestamps) >= 2 else None
    observed_hz = (
        (len(timestamps) - 1) / elapsed
        if elapsed is not None and elapsed > 0.0
        else None
    )
    deadline = 1.0 / target_hz * deadline_factor if target_hz else None
    return FrequencyStats(
        event=normalized_event,
        sample_count=len(timestamps),
        elapsed_sec=elapsed,
        observed_hz=observed_hz,
        p50_interval_sec=_percentile(intervals, 0.50),
        p95_interval_sec=_percentile(intervals, 0.95),
        p99_interval_sec=_percentile(intervals, 0.99),
        max_interval_sec=max(intervals) if intervals else None,
        deadline_misses=(sum(interval > deadline for interval in intervals) if deadline else 0),
        target_hz=target_hz,
        deadline_factor=deadline_factor,
    )


@dataclass
class FrequencyLedger:
    """Record independent event clocks for a single commissioning run."""

    _timestamps: dict[str, list[float]] = field(
        default_factory=lambda: {event: [] for event in _EVENT_NAMES}
    )

    def mark(self, event: str, timestamp_sec: float | None = None) -> float:
        normalized_event = str(event).strip()
        if normalized_event not in _EVENT_NAMES:
            raise ValueError(f"event must be one of {_EVENT_NAMES}")
        timestamp = _finite(
            time.monotonic() if timestamp_sec is None else timestamp_sec,
            name="timestamp",
        )
        timestamps = self._timestamps[normalized_event]
        if timestamps and timestamp < timestamps[-1]:
            raise ValueError(f"{normalized_event} timestamps must be monotonic")
        timestamps.append(timestamp)
        return timestamp

    def timestamps(self, event: str) -> tuple[float, ...]:
        normalized_event = str(event).strip()
        if normalized_event not in _EVENT_NAMES:
            raise ValueError(f"event must be one of {_EVENT_NAMES}")
        return tuple(self._timestamps[normalized_event])

    def summarize(
        self,
        *,
        target_hz: float | None = None,
        deadline_factor: float = 2.0,
    ) -> dict[str, FrequencyStats]:
        return {
            event: summarize_timestamps(
                event,
                self._timestamps[event],
                target_hz=target_hz,
                deadline_factor=deadline_factor,
            )
            for event in _EVENT_NAMES
        }

    def as_dict(
        self,
        *,
        target_hz: float | None = None,
        deadline_factor: float = 2.0,
    ) -> dict[str, dict[str, object]]:
        return {
            event: stats.as_dict()
            for event, stats in self.summarize(
                target_hz=target_hz,
                deadline_factor=deadline_factor,
            ).items()
        }


__all__ = ["FrequencyLedger", "FrequencyStats", "summarize_timestamps"]

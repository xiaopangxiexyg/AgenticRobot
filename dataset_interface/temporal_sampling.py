"""Timestamp-based window selection for fixed-cadence training.

An Episode MP4 has one frame per accepted Parquet row, but its container PTS
is encoded at nominal FPS. The Parquet ``timestamp`` column, not MP4 PTS or
row number, is the authoritative capture clock for temporal targets.
"""

from __future__ import annotations

import math
from typing import Sequence


class TemporalSamplingError(ValueError):
    """A sample clock or requested training cadence is ambiguous."""


def select_fixed_cadence_window_starts(
    timestamps: Sequence[float],
    *,
    target_fps: float,
    horizon: int,
    step_tolerance_fraction: float = 0.25,
    window_drift_fraction: float = 0.10,
    continuity_segments: Sequence[int | None] | None = None,
) -> tuple[int, ...]:
    """Admit contiguous action chunks only when the capture clock fits FPS.

    The per-step bound rejects dropped/duplicated sample time; the full-window
    bound prevents small same-direction errors from silently changing action
    speed. No frame or action is duplicated, interpolated, or fabricated.
    Consumers still resolve video via the row's ``video_frame_index`` column.
    """

    try:
        times = tuple(float(value) for value in timestamps)
        fps = float(target_fps)
        step_tolerance = float(step_tolerance_fraction)
        window_drift = float(window_drift_fraction)
    except (TypeError, ValueError) as exc:
        raise TemporalSamplingError("timestamps and cadence must be numeric") from exc
    if (
        not math.isfinite(fps)
        or fps <= 0.0
        or not math.isfinite(step_tolerance)
        or not 0.0 <= step_tolerance < 1.0
        or not math.isfinite(window_drift)
        or not 0.0 <= window_drift < 1.0
        or isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon <= 0
    ):
        raise TemporalSamplingError("target_fps, horizon or cadence tolerance is invalid")
    if any(not math.isfinite(value) for value in times) or any(
        next_time <= current for current, next_time in zip(times, times[1:])
    ):
        raise TemporalSamplingError("sample timestamps must be finite and strictly increasing")
    if continuity_segments is None:
        segments: tuple[int | None, ...] = (0,) * len(times)
    else:
        segments = tuple(continuity_segments)
        if len(segments) != len(times):
            raise TemporalSamplingError("continuity segment count differs from timestamps")

    period = 1.0 / fps
    minimum_step = period * (1.0 - step_tolerance)
    maximum_step = period * (1.0 + step_tolerance)
    bad_prefix = [0]
    for index in range(1, len(times)):
        elapsed = times[index] - times[index - 1]
        bad = (
            segments[index] is None
            or segments[index - 1] is None
            or segments[index] != segments[index - 1]
            or elapsed < minimum_step - 1e-9
            or elapsed > maximum_step + 1e-9
        )
        bad_prefix.append(bad_prefix[-1] + int(bad))

    starts: list[int] = []
    for start in range(max(0, len(times) - horizon + 1)):
        end = start + horizon - 1
        if segments[start] is None or bad_prefix[end] != bad_prefix[start]:
            continue
        expected_span = (horizon - 1) * period
        actual_span = times[end] - times[start]
        if abs(actual_span - expected_span) <= window_drift * expected_span + 1e-9:
            starts.append(start)
    return tuple(starts)

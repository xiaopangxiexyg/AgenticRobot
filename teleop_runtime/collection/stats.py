from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class EpisodeStats:
    episode_index: int
    frame_count: int
    start_timestamp: float | None
    end_timestamp: float | None
    duration_sec: float
    average_dt_sec: float | None
    average_hz: float | None
    mean_control_hz: float | None
    nominal_fps: float
    active_sides: tuple[str, ...]
    arm_state_source: str | None
    camera_keys: tuple[str, ...]
    fault_count: int


class EpisodeStatsAccumulator:
    def __init__(self, episode_index: int):
        self.episode_index = int(episode_index)
        self._frame_count = 0
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._last_row_timestamp: float | None = None
        self._intervals_sec: list[float] = []
        self._control_hz_values: list[float] = []
        self._active_sides: tuple[str, ...] = ()
        self._arm_state_source: str | None = None
        self._camera_keys: set[str] = set()
        self._fault_count = 0

    def update(self, row: Mapping[str, Any]) -> None:
        timestamp = row.get("timestamp")
        if timestamp is not None:
            timestamp = float(timestamp)
            if self._first_timestamp is None:
                self._first_timestamp = timestamp
            if self._last_row_timestamp is not None and timestamp > self._last_row_timestamp:
                self._intervals_sec.append(timestamp - self._last_row_timestamp)
            self._last_row_timestamp = timestamp
            self._last_timestamp = timestamp
        control_hz = row.get("control_hz")
        if control_hz is not None:
            self._control_hz_values.append(float(control_hz))
        active_sides = row.get("active_sides")
        if isinstance(active_sides, (list, tuple)):
            self._active_sides = tuple(str(side) for side in active_sides)
        arm_state_source = row.get("arm_state_source")
        if arm_state_source is not None:
            self._arm_state_source = str(arm_state_source)
        camera_keys = row.get("camera_keys")
        if isinstance(camera_keys, (list, tuple)):
            self._camera_keys.update(str(camera_key) for camera_key in camera_keys)
        fault_codes = row.get("fault_codes")
        if isinstance(fault_codes, (list, tuple)):
            self._fault_count += len(fault_codes)
        self._frame_count += 1

    def finalize(self, *, nominal_fps: float) -> EpisodeStats:
        average_dt_sec = (
            float(np.mean(np.asarray(self._intervals_sec, dtype=float)))
            if self._intervals_sec
            else None
        )
        average_hz = (
            None if average_dt_sec is None or average_dt_sec <= 1e-9 else float(1.0 / average_dt_sec)
        )
        mean_control_hz = (
            float(np.mean(np.asarray(self._control_hz_values, dtype=float)))
            if self._control_hz_values
            else None
        )
        duration_sec = (
            0.0
            if self._first_timestamp is None or self._last_timestamp is None
            else float(self._last_timestamp - self._first_timestamp)
        )
        return EpisodeStats(
            episode_index=int(self.episode_index),
            frame_count=int(self._frame_count),
            start_timestamp=self._first_timestamp,
            end_timestamp=self._last_timestamp,
            duration_sec=duration_sec,
            average_dt_sec=average_dt_sec,
            average_hz=average_hz,
            mean_control_hz=mean_control_hz,
            nominal_fps=float(nominal_fps),
            active_sides=tuple(self._active_sides),
            arm_state_source=self._arm_state_source,
            camera_keys=tuple(sorted(self._camera_keys)),
            fault_count=int(self._fault_count),
        )

"""Dataset-row integrity and episode statistics shared by all data flows."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


_FORCE_LIMIT_COLLISION_LABEL = "detected_collision"
_FORCE_LIMIT_CLEAN_LABEL = "clean"
_FORCE_LIMIT_TEXT_RE = re.compile(
    r"\b(?:force|torque)\s+"
    r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?\s*(?:nm|n)\s*>\s*"
    r"(?:hard_stop_(?:force_n|torque_nm)|[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


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
    pose_quality_status_counts: Mapping[str, int]
    pose_quality_bad_count: int
    pose_quality_warning_count: int
    pose_quality_min_score: float | None
    pose_quality_mean_score: float | None
    force_collision_detected: bool
    force_collision_label: str


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _strings_in_payload(value: Any) -> Iterable[str]:
    if value is None:
        return
    if isinstance(value, str):
        yield value
        try:
            decoded = json.loads(value)
        except (TypeError, ValueError):
            return
        if decoded is not value:
            yield from _strings_in_payload(decoded)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _strings_in_payload(item)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _strings_in_payload(item)


def _has_force_limit_collision_text(*values: Any) -> bool:
    for value in values:
        for text in _strings_in_payload(value):
            if _FORCE_LIMIT_TEXT_RE.search(text):
                return True
    return False


def _row_has_force_limit_collision(row: Mapping[str, Any]) -> bool:
    values: list[Any] = [row.get("faults_json")]
    for side in ("left", "right"):
        prefix = f"observation.arm.{side}"
        values.extend(
            [
                row.get(f"{prefix}.hardware_result_text"),
                row.get(f"{prefix}.send_result_text"),
                row.get(f"{prefix}.fault"),
                row.get(f"{prefix}.actual_hardware_result_text"),
                row.get(f"{prefix}.actual_state_error"),
            ]
        )
    return _has_force_limit_collision_text(*values)


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
        self._pose_quality_status_counts: dict[str, int] = {}
        self._pose_quality_scores: list[float] = []
        self._pose_quality_bad_count = 0
        self._pose_quality_warning_count = 0
        self._force_collision_detected = False

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
        pose_status = row.get("teleop.pose_quality.status")
        if pose_status is not None:
            status_key = str(pose_status)
            self._pose_quality_status_counts[status_key] = (
                self._pose_quality_status_counts.get(status_key, 0) + 1
            )
            if status_key == "bad":
                self._pose_quality_bad_count += 1
            elif status_key == "warning":
                self._pose_quality_warning_count += 1
        pose_bad = row.get("teleop.pose_quality.bad")
        if pose_status is None and _truthy(pose_bad):
            self._pose_quality_bad_count += 1
        pose_score = _finite_float(row.get("teleop.pose_quality.score"))
        if pose_score is not None:
            self._pose_quality_scores.append(pose_score)
        legacy_force_collision = _truthy(row.get("teleop.force_limit_exceeded")) or _truthy(
            row.get("teleop.force_collision_detected")
        )
        if legacy_force_collision or _row_has_force_limit_collision(row):
            self._force_collision_detected = True
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
        pose_scores = np.asarray(self._pose_quality_scores, dtype=float)
        pose_quality_min_score = float(np.min(pose_scores)) if pose_scores.size else None
        pose_quality_mean_score = float(np.mean(pose_scores)) if pose_scores.size else None
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
            pose_quality_status_counts=dict(self._pose_quality_status_counts),
            pose_quality_bad_count=int(self._pose_quality_bad_count),
            pose_quality_warning_count=int(self._pose_quality_warning_count),
            pose_quality_min_score=pose_quality_min_score,
            pose_quality_mean_score=pose_quality_mean_score,
            force_collision_detected=bool(self._force_collision_detected),
            force_collision_label=(
                _FORCE_LIMIT_COLLISION_LABEL
                if self._force_collision_detected
                else _FORCE_LIMIT_CLEAN_LABEL
            ),
        )


__all__ = ["EpisodeStats", "EpisodeStatsAccumulator"]

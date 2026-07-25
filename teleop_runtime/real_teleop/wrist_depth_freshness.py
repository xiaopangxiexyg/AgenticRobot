"""Wrist-depth freshness detection without teleop runtime dependencies.

This module owns only observation freshness classification.  It neither starts
camera devices nor changes collection/control state, so callers can test it
without real hardware.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np


DEFAULT_STALE_REPEAT_FAILURE_FRAMES = 3
DEFAULT_EPOCH_TIMESTAMP_STALE_SEC = 2.0
WRIST_DEPTH_CAMERA_KEYS = ("left_wrist", "right_wrist")
DEFAULT_WRIST_DEPTH_SAVE_FPS = 15
DEFAULT_WRIST_DEPTH_STREAM_FPS = 15
DEFAULT_WRIST_DEPTH_FPS = DEFAULT_WRIST_DEPTH_SAVE_FPS
DEFAULT_WRIST_DEPTH_MIN_M = 0.05
DEFAULT_WRIST_DEPTH_MAX_M = 0.5


def _json_safe_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.astype(float, copy=False).reshape(-1).round(4).tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


def _safe_float_value(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _safe_int_value(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_depth_signature_value(value: object, *, digits: int) -> float | None:
    numeric = _safe_float_value(value)
    if numeric is None:
        return None
    return round(float(numeric), int(digits))


def _wrist_depth_signature(
    depth_metadata: Mapping[str, object],
) -> tuple[tuple[str, object], ...] | None:
    parts: list[tuple[str, object]] = []
    frame_number = _safe_int_value(depth_metadata.get("frame_number"))
    if frame_number is not None:
        parts.append(("frame_number", int(frame_number)))
    camera_timestamp_sec = _normalize_depth_signature_value(
        depth_metadata.get("camera_timestamp_sec"),
        digits=6,
    )
    if camera_timestamp_sec is not None:
        parts.append(("camera_timestamp_sec", camera_timestamp_sec))
    sensor_timestamp_ms = _normalize_depth_signature_value(
        depth_metadata.get("sensor_timestamp_ms"),
        digits=3,
    )
    if sensor_timestamp_ms is not None:
        parts.append(("sensor_timestamp_ms", sensor_timestamp_ms))
    return tuple(parts) if parts else None


def _frame_source_marker(frame: object) -> tuple[str, object] | None:
    frame_index = _safe_int_value(getattr(frame, "frame_index", None))
    if frame_index is not None:
        return ("frame_index", int(frame_index))
    source_timestamp = _normalize_depth_signature_value(
        getattr(frame, "source_timestamp", None),
        digits=6,
    )
    if source_timestamp is not None:
        return ("source_timestamp", source_timestamp)
    return None


def _epoch_depth_timestamp_age_sec(
    frame: object,
    depth_metadata: Mapping[str, object],
) -> float | None:
    source_timestamp = _safe_float_value(getattr(frame, "source_timestamp", None))
    camera_timestamp_sec = _safe_float_value(depth_metadata.get("camera_timestamp_sec"))
    if source_timestamp is None or camera_timestamp_sec is None:
        return None
    if source_timestamp < 1_000_000_000.0 or camera_timestamp_sec < 1_000_000_000.0:
        return None
    return float(source_timestamp) - float(camera_timestamp_sec)


@dataclass
class WristDepthFreshnessState:
    first_seen_at: float | None = None
    last_seen_at: float | None = None
    last_changed_at: float | None = None
    last_signature: tuple[tuple[str, object], ...] | None = None
    last_source_marker: tuple[str, object] | None = None
    seen_signatures: set[tuple[tuple[str, object], ...]] = field(default_factory=set)
    repeated_source_frame_count: int = 0
    total_depth_frames_seen: int = 0


class WristDepthFreshnessMonitor:
    def __init__(
        self,
        *,
        stale_repeat_threshold: int = DEFAULT_STALE_REPEAT_FAILURE_FRAMES,
    ) -> None:
        self.stale_repeat_threshold = max(1, int(stale_repeat_threshold))
        self._states: dict[str, WristDepthFreshnessState] = {}
        self._lock = threading.RLock()

    def reset(self, camera_key: str | None = None) -> None:
        with self._lock:
            if camera_key is None:
                self._states.clear()
            else:
                self._states.pop(str(camera_key), None)

    def update(
        self,
        camera_key: str,
        frame: object,
        *,
        now: float,
        depth_metadata: Mapping[str, object],
    ) -> dict[str, object]:
        signature = _wrist_depth_signature(depth_metadata)
        source_marker = _frame_source_marker(frame)
        depth_timestamp_age_sec = _epoch_depth_timestamp_age_sec(frame, depth_metadata)
        with self._lock:
            state = self._states.setdefault(str(camera_key), WristDepthFreshnessState())
            if state.first_seen_at is None:
                state.first_seen_at = float(now)
            state.last_seen_at = float(now)
            state.total_depth_frames_seen += 1
            source_advanced = (
                source_marker is not None
                and state.last_source_marker is not None
                and source_marker != state.last_source_marker
            )
            if signature is not None:
                if signature != state.last_signature:
                    state.last_changed_at = float(now)
                    state.repeated_source_frame_count = 0
                elif source_advanced:
                    state.repeated_source_frame_count += 1
                state.seen_signatures.add(signature)
                state.last_signature = signature
            if source_marker is not None:
                state.last_source_marker = source_marker
            first_seen_at = state.first_seen_at
            last_changed_at = state.last_changed_at
            unique_count = len(state.seen_signatures)
            repeated_count = int(state.repeated_source_frame_count)
            total_seen = int(state.total_depth_frames_seen)

        first_seen_age_sec = (
            None if first_seen_at is None else max(0.0, float(now) - float(first_seen_at))
        )
        last_changed_age_sec = (
            None
            if last_changed_at is None
            else max(0.0, float(now) - float(last_changed_at))
        )
        stale_by_repeat = repeated_count >= int(self.stale_repeat_threshold)
        stale_by_timestamp = (
            depth_timestamp_age_sec is not None
            and depth_timestamp_age_sec >= DEFAULT_EPOCH_TIMESTAMP_STALE_SEC
        )
        status_level = "ok"
        status_label = "Depth 新鲜"
        status_message = ""
        if signature is None:
            status_level = "error"
            status_label = "Depth metadata 缺失"
            status_message = "无法读取 depth frame_number/timestamp，不能验证 wrist depth 是否在更新。"
        elif stale_by_timestamp:
            status_level = "error"
            status_label = "Wrist Depth 停滞"
            status_message = (
                "depth camera timestamp stale: "
                f"age={depth_timestamp_age_sec:.2f}s >= {DEFAULT_EPOCH_TIMESTAMP_STALE_SEC:.2f}s"
            )
        elif stale_by_repeat:
            status_level = "error"
            status_label = "Wrist Depth 停滞"
            status_message = (
                "RGB/source frame 在更新，但 depth frame_number/timestamp 连续 "
                f"{repeated_count} 帧未变化。"
            )
        elif unique_count < 2:
            status_level = "waiting"
            status_label = "等待 Depth 更新"
            status_message = "已收到 wrist depth，正在确认 frame_number/timestamp 是否连续变化。"

        return {
            "depth_signature_available": signature is not None,
            "depth_signature_unique_count": int(unique_count),
            "depth_signature_repeat_count": int(repeated_count),
            "depth_signature_total_seen": int(total_seen),
            "depth_signature_first_seen_age_sec": first_seen_age_sec,
            "depth_signature_last_changed_age_sec": last_changed_age_sec,
            "depth_source_marker": _json_safe_value(source_marker),
            "depth_signature": _json_safe_value(signature),
            "depth_timestamp_age_sec": depth_timestamp_age_sec,
            "depth_freshness_status_level": status_level,
            "depth_freshness_status_label": status_label,
            "depth_freshness_status_message": status_message,
        }


__all__ = [
    "DEFAULT_EPOCH_TIMESTAMP_STALE_SEC",
    "DEFAULT_STALE_REPEAT_FAILURE_FRAMES",
    "DEFAULT_WRIST_DEPTH_FPS",
    "DEFAULT_WRIST_DEPTH_MAX_M",
    "DEFAULT_WRIST_DEPTH_MIN_M",
    "DEFAULT_WRIST_DEPTH_SAVE_FPS",
    "DEFAULT_WRIST_DEPTH_STREAM_FPS",
    "WRIST_DEPTH_CAMERA_KEYS",
    "WristDepthFreshnessMonitor",
    "WristDepthFreshnessState",
]

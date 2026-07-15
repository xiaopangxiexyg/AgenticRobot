"""Pure tracker-status payloads and publication for the real-teleop GUI.

The functions transform already-read tracker channels into GUI-safe status
data.  They do not acquire tracker data, update calibration, or change teleop
state, so source/lifecycle ownership stays with the real-teleop coordinator.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation as SciPyRotation

from ..transforms import as_pose_matrix


TRACKER_GUI_WARN_AGE_SEC = 0.25
TRACKER_GUI_STALE_AGE_SEC = 0.50


class _GuiPublisher(Protocol):
    def publish(self, **payload: object) -> None: ...


def tracker_gui_status(
    *,
    enabled: bool,
    valid: bool,
    age_sec: float | None,
    warn_age_sec: float = TRACKER_GUI_WARN_AGE_SEC,
    stale_age_sec: float = TRACKER_GUI_STALE_AGE_SEC,
) -> tuple[str, str, bool]:
    if not enabled:
        return "disabled", "未启用", False
    if age_sec is None:
        return "no-data", "无数据", False
    if not valid:
        return "invalid", "无效", False
    if age_sec > stale_age_sec:
        return "stale", f"STALE {age_sec:.2f}s", True
    if age_sec >= warn_age_sec:
        return "warn", f"延迟 {age_sec:.2f}s", False
    return "ok", "有效", False


def pose_xyz_rpy_summary(
    channel: object | None,
    *,
    now: float,
    enabled: bool = True,
    warn_age_sec: float = TRACKER_GUI_WARN_AGE_SEC,
    stale_age_sec: float = TRACKER_GUI_STALE_AGE_SEC,
) -> dict[str, object]:
    if not enabled:
        status_level, status_label, stale = tracker_gui_status(
            enabled=False,
            valid=False,
            age_sec=None,
            warn_age_sec=warn_age_sec,
            stale_age_sec=stale_age_sec,
        )
        return {
            "enabled": False,
            "valid": False,
            "xyz_m": None,
            "rpy_deg": None,
            "age_sec": None,
            "stale": stale,
            "status_level": status_level,
            "status_label": status_label,
            "stale_timeout_sec": stale_age_sec,
            "tracker_name": "",
            "tracker_role": "",
            "seq": None,
        }
    if channel is None:
        status_level, status_label, stale = tracker_gui_status(
            enabled=True,
            valid=False,
            age_sec=None,
            warn_age_sec=warn_age_sec,
            stale_age_sec=stale_age_sec,
        )
        return {
            "enabled": True,
            "valid": False,
            "xyz_m": None,
            "rpy_deg": None,
            "age_sec": None,
            "stale": stale,
            "status_level": status_level,
            "status_label": status_label,
            "stale_timeout_sec": stale_age_sec,
            "tracker_name": "",
            "tracker_role": "",
            "seq": None,
        }
    metadata = getattr(channel, "metadata", None) or {}
    pose = as_pose_matrix(getattr(channel, "value"))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rpy_deg = SciPyRotation.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=True)
    except ValueError:
        rpy_deg = np.full(3, np.nan, dtype=float)
    rpy_deg = np.round(rpy_deg.astype(float), 1)
    rpy_values = [float(value) if np.isfinite(value) else None for value in rpy_deg]
    timestamp = metadata.get("receiver_timestamp")
    if timestamp is None:
        timestamp = getattr(channel, "timestamp", None)
    age_sec = max(0.0, now - float(timestamp)) if timestamp is not None else None
    valid = bool(getattr(channel, "valid", False))
    status_level, status_label, stale = tracker_gui_status(
        enabled=True,
        valid=valid,
        age_sec=age_sec,
        warn_age_sec=warn_age_sec,
        stale_age_sec=stale_age_sec,
    )
    return {
        "enabled": True,
        "valid": valid,
        "xyz_m": np.round(pose[:3, 3].astype(float), 4).tolist(),
        "rpy_deg": rpy_values,
        "age_sec": round(age_sec, 3) if age_sec is not None else None,
        "stale": stale,
        "status_level": status_level,
        "status_label": status_label,
        "stale_timeout_sec": stale_age_sec,
        "tracker_name": str(metadata.get("tracker_name") or ""),
        "tracker_role": str(metadata.get("tracker_role") or ""),
        "seq": metadata.get("seq"),
    }


def raw_tracker_pose_state_from_channels(
    channels: Mapping[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float,
    warn_age_sec: float = TRACKER_GUI_WARN_AGE_SEC,
    stale_age_sec: float = TRACKER_GUI_STALE_AGE_SEC,
) -> dict[str, object]:
    active = set(active_sides)
    return {
        "left": pose_xyz_rpy_summary(
            channels.get("left_wrist_pose"),
            now=now,
            enabled="left" in active,
            warn_age_sec=warn_age_sec,
            stale_age_sec=stale_age_sec,
        ),
        "right": pose_xyz_rpy_summary(
            channels.get("right_wrist_pose"),
            now=now,
            enabled="right" in active,
            warn_age_sec=warn_age_sec,
            stale_age_sec=stale_age_sec,
        ),
    }


def publish_raw_tracker_pose_state(
    ui: _GuiPublisher | None,
    channels: Mapping[str, object],
    *,
    active_sides: tuple[str, ...],
    now: float | None = None,
    warn_age_sec: float = TRACKER_GUI_WARN_AGE_SEC,
    stale_age_sec: float = TRACKER_GUI_STALE_AGE_SEC,
) -> None:
    if ui is None:
        return
    current_time = time.time() if now is None else float(now)
    ui.publish(
        raw_tracker_poses=raw_tracker_pose_state_from_channels(
            channels,
            active_sides=active_sides,
            now=current_time,
            warn_age_sec=warn_age_sec,
            stale_age_sec=stale_age_sec,
        )
    )


__all__ = [
    "TRACKER_GUI_STALE_AGE_SEC",
    "TRACKER_GUI_WARN_AGE_SEC",
    "pose_xyz_rpy_summary",
    "publish_raw_tracker_pose_state",
    "raw_tracker_pose_state_from_channels",
    "tracker_gui_status",
]

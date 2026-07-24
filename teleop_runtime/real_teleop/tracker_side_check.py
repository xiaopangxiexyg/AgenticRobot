"""Read-only left/right placement check for dual VIVE wrist trackers.

The VIVE desk calibration defines the operator lateral axis by sweeping from
the operator's right side to the left side, so ``+Y`` means "operator left".
The production retarget frame may then apply a post-yaw rotation (180 degrees
by default).  This check deliberately removes that post-yaw before comparing
the two raw tracker positions; otherwise the default rotation would invert the
left/right result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..calibration import DEFAULT_VIRTUAL_DESK_POST_YAW_DEG
from ..transforms import (
    as_pose_matrix,
    identity_pose,
    invert_pose,
    validate_pose_matrix,
)


TRACKER_SIDE_MIN_LATERAL_SEPARATION_M = 0.05
TRACKER_SIDE_MAX_SAMPLE_AGE_SEC = 0.50


def _yaw_pose_deg(yaw_deg: float) -> np.ndarray:
    yaw_rad = np.deg2rad(float(yaw_deg))
    pose = identity_pose()
    pose[:3, :3] = np.array(
        [
            [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
            [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return pose


def _unavailable(message: str) -> dict[str, object]:
    return {
        "status": "unavailable",
        "warning_required": False,
        "operator_lateral_axis": "+Y=left",
        "basis": "vive_virtual_desk_before_post_yaw",
        "message": str(message),
    }


def evaluate_tracker_side_assignment(
    channels: Mapping[str, object],
    *,
    calibration: object,
    active_sides: Sequence[str],
    now: float | None = None,
    post_yaw_deg: float | None = None,
    min_lateral_separation_m: float = TRACKER_SIDE_MIN_LATERAL_SEPARATION_M,
    max_sample_age_sec: float = TRACKER_SIDE_MAX_SAMPLE_AGE_SEC,
) -> dict[str, object]:
    """Classify the logical wrist trackers using the calibrated lateral axis.

    The input channels are expected to have already passed through the current
    logical tracker mapping.  The function only observes that mapping; it does
    not mutate or automatically swap it.
    """

    if set(str(side) for side in active_sides) != {"left", "right"}:
        payload = _unavailable("仅双侧 wrist tracker 启用时检查左右位置。")
        payload["status"] = "not_applicable"
        return payload

    left_channel = channels.get("left_wrist_pose")
    right_channel = channels.get("right_wrist_pose")
    if left_channel is None or right_channel is None:
        return _unavailable("等待左右 wrist tracker 数据。")
    if not bool(getattr(left_channel, "valid", False)) or not bool(
        getattr(right_channel, "valid", False)
    ):
        return _unavailable("等待左右 wrist tracker 同时有效。")
    if now is not None:
        current_time = float(now)
        max_age_sec = max(0.0, float(max_sample_age_sec))
        for side, channel in (
            ("left", left_channel),
            ("right", right_channel),
        ):
            metadata = getattr(channel, "metadata", None) or {}
            timestamp = metadata.get("receiver_timestamp")
            if timestamp is None:
                timestamp = getattr(channel, "timestamp", None)
            if timestamp is None:
                return _unavailable(
                    f"等待 {side} wrist tracker 的接收时间戳。"
                )
            try:
                age_sec = current_time - float(timestamp)
            except (TypeError, ValueError):
                return _unavailable(
                    f"{side} wrist tracker 的接收时间戳无效。"
                )
            if age_sec < -0.05 or age_sec > max_age_sec:
                return _unavailable(
                    f"{side} wrist tracker 数据不新鲜：age={age_sec:.3f}s，"
                    f"要求 0–{max_age_sec:.3f}s。"
                )

    world_to_desk_pose = getattr(calibration, "world_to_desk_pose", None)
    if not callable(world_to_desk_pose):
        return _unavailable("当前 calibration 不提供 VIVE desk 坐标变换。")
    try:
        retarget_pose_in_world = world_to_desk_pose()
        if retarget_pose_in_world is None:
            return _unavailable("尚未完成 VIVE desk 坐标系标定。")
        metadata = getattr(calibration, "metadata", None) or {}
        resolved_post_yaw_deg = float(
            metadata.get(
                "virtual_desk_post_yaw_deg",
                DEFAULT_VIRTUAL_DESK_POST_YAW_DEG
                if post_yaw_deg is None
                else post_yaw_deg,
            )
        )
        post_yaw_pose = _yaw_pose_deg(resolved_post_yaw_deg)
        retarget_pose_in_world = as_pose_matrix(retarget_pose_in_world)
        left_pose_in_world = as_pose_matrix(getattr(left_channel, "value"))
        right_pose_in_world = as_pose_matrix(getattr(right_channel, "value"))
        if not all(
            np.all(np.isfinite(pose))
            for pose in (
                retarget_pose_in_world,
                left_pose_in_world,
                right_pose_in_world,
            )
        ):
            return _unavailable(
                "Tracker 左右位置检查输入包含非有限坐标。"
            )
        for label, pose in (
            ("VIVE desk calibration", retarget_pose_in_world),
            ("left wrist tracker", left_pose_in_world),
            ("right wrist tracker", right_pose_in_world),
        ):
            valid_pose, reason = validate_pose_matrix(pose)
            if not valid_pose:
                return _unavailable(f"{label} 位姿无效：{reason}")
        base_desk_pose_in_world = (
            retarget_pose_in_world @ invert_pose(post_yaw_pose)
        )
        world_in_base_desk = invert_pose(base_desk_pose_in_world)
        left_y_m = float(
            (world_in_base_desk @ left_pose_in_world)[1, 3]
        )
        right_y_m = float(
            (world_in_base_desk @ right_pose_in_world)[1, 3]
        )
    except (TypeError, ValueError, AttributeError, np.linalg.LinAlgError) as exc:
        return _unavailable(
            f"Tracker 左右位置检查暂不可用：{type(exc).__name__}: {exc}"
        )

    lateral_gap_m = left_y_m - right_y_m
    min_gap_m = max(0.0, float(min_lateral_separation_m))
    if lateral_gap_m < 0.0:
        status = "swapped"
        warning_required = True
        message = (
            "检测到左右 Tracker 的空间顺序可能相反。"
            "VIVE desk 标定规定“身体右侧 → 身体左侧”为 +Y，"
            f"当前 logical_left Y={left_y_m:+.3f}m，"
            f"logical_right Y={right_y_m:+.3f}m，"
            f"左-右={lateral_gap_m:+.3f}m。"
            "请确认 Tracker 佩戴位置和当前逻辑映射后再开始 Home。"
        )
    elif lateral_gap_m < min_gap_m:
        status = "ambiguous"
        warning_required = True
        message = (
            "左右 Tracker 的横向间距过小，暂时无法可靠确认左右。"
            "VIVE desk 标定规定“身体右侧 → 身体左侧”为 +Y，"
            f"当前 logical_left Y={left_y_m:+.3f}m，"
            f"logical_right Y={right_y_m:+.3f}m，"
            f"左-右={lateral_gap_m:+.3f}m，"
            f"可靠判定至少需要 {min_gap_m:.3f}m。"
            "请把双手自然分开放在身体左右两侧后重试。"
        )
    else:
        status = "correct"
        warning_required = False
        message = (
            "Tracker 左右位置检查通过："
            f"logical_left Y={left_y_m:+.3f}m > "
            f"logical_right Y={right_y_m:+.3f}m。"
        )

    return {
        "status": status,
        "warning_required": warning_required,
        "operator_lateral_axis": "+Y=left",
        "basis": "vive_virtual_desk_before_post_yaw",
        "post_yaw_deg": resolved_post_yaw_deg,
        "left_y_m": round(left_y_m, 4),
        "right_y_m": round(right_y_m, 4),
        "lateral_gap_m": round(lateral_gap_m, 4),
        "min_lateral_separation_m": min_gap_m,
        "message": message,
    }


__all__ = [
    "TRACKER_SIDE_MIN_LATERAL_SEPARATION_M",
    "TRACKER_SIDE_MAX_SAMPLE_AGE_SEC",
    "evaluate_tracker_side_assignment",
]

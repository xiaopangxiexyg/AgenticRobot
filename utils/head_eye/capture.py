from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from utils.calibration_common.capture import (
    max_distance as _max_distance,
    mean_vector as _mean_vector,
    most_common as _most_common,
    now_utc as _now_utc,
)

from .calibration import CalibrationSample
from .transforms import average_transforms


@dataclass(frozen=True)
class CaptureBurstConfig:
    duration_sec: float = 0.6
    sample_interval_sec: float = 0.03
    min_detections: int = 3
    max_robot_motion_m: float = 0.0015
    max_robot_motion_deg: float = 0.25
    max_target_motion_m: float = 0.003
    max_target_motion_deg: float = 0.8
    accept_unstable: bool = False


def capture_burst(
    *,
    camera: Any,
    detector: Any,
    robot: Any,
    config: CaptureBurstConfig,
    sample_index: int,
) -> tuple[CalibrationSample, dict[str, float | int | bool]]:
    """Collect a short stable burst and average camera/robot poses."""

    robot_poses = []
    robot_pose6 = []
    robot_joints = []
    robot_tools = []
    target_poses = []
    reprojections = []
    corner_counts = []
    frame_count = 0
    deadline = time.time() + max(0.05, float(config.duration_sec))
    while time.time() < deadline:
        frame = camera.read()
        frame_count += 1
        detection, _ = detector.detect(
            frame.color_bgr,
            frame.intrinsics_matrix,
            frame.distortion_coeffs,
            draw=False,
        )
        if detection is not None:
            robot_sample = robot.read()
            robot_poses.append(robot_sample.T_gripper_to_base)
            robot_pose6.append(robot_sample.pose_xyz_rpy)
            robot_joints.append(robot_sample.joint_deg)
            if robot_sample.tool_frame:
                robot_tools.append(robot_sample.tool_frame)
            target_poses.append(detection.T_target_to_camera)
            if detection.reprojection_rmse_px is not None:
                reprojections.append(float(detection.reprojection_rmse_px))
            corner_counts.append(int(detection.corner_count))
        sleep_sec = float(config.sample_interval_sec)
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    if len(target_poses) < int(config.min_detections):
        raise RuntimeError(
            f"only {len(target_poses)} valid detections in {frame_count} frames; "
            f"need at least {config.min_detections}."
        )

    T_robot_mean = average_transforms(robot_poses)
    T_target_mean = average_transforms(target_poses)
    robot_motion = _max_distance(T_robot_mean, robot_poses)
    target_motion = _max_distance(T_target_mean, target_poses)
    stable = (
        robot_motion[0] <= float(config.max_robot_motion_m)
        and robot_motion[1] <= float(config.max_robot_motion_deg)
        and target_motion[0] <= float(config.max_target_motion_m)
        and target_motion[1] <= float(config.max_target_motion_deg)
    )
    if not stable and not config.accept_unstable:
        raise RuntimeError(
            "burst was not stable: "
            f"robot max {robot_motion[0] * 1000.0:.2f}mm/{robot_motion[1]:.2f}deg, "
            f"target max {target_motion[0] * 1000.0:.2f}mm/{target_motion[1]:.2f}deg."
        )

    sample = CalibrationSample(
        T_gripper_to_base=T_robot_mean,
        T_target_to_camera=T_target_mean,
        robot_pose_xyz_rpy=_mean_vector(robot_pose6),
        robot_joint_deg=_mean_vector(robot_joints),
        target_reprojection_rmse_px=float(np.mean(reprojections)) if reprojections else None,
        target_corner_count=int(round(float(np.mean(corner_counts)))) if corner_counts else None,
        timestamp_utc=_now_utc(),
        metadata={
            "sample_index": int(sample_index),
            "burst_detection_count": len(target_poses),
            "burst_frame_count": frame_count,
            "burst_stable": stable,
            "robot_motion_max_m": robot_motion[0],
            "robot_motion_max_deg": robot_motion[1],
            "target_motion_max_m": target_motion[0],
            "target_motion_max_deg": target_motion[1],
            "robot_tool_frame": _most_common(robot_tools),
        },
    )
    stats = {
        "detections": len(target_poses),
        "frames": frame_count,
        "stable": stable,
        "robot_motion_max_m": robot_motion[0],
        "robot_motion_max_deg": robot_motion[1],
        "target_motion_max_m": target_motion[0],
        "target_motion_max_deg": target_motion[1],
        "reprojection_rmse_px": float(np.mean(reprojections)) if reprojections else -1.0,
    }
    return sample, stats

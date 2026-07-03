from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import time
from typing import Any

import numpy as np

from .calibration import CalibrationSample
from .transforms import average_transforms, transform_distance


@dataclass(frozen=True)
class CaptureBurstConfig:
    duration_sec: float = 2.0
    sample_interval_sec: float = 0.0
    min_detections: int = 20
    max_robot_motion_m: float = 0.0015
    max_robot_motion_deg: float = 0.25
    max_target_motion_m: float = 0.003
    max_target_motion_deg: float = 0.8
    max_reprojection_rmse_px: float | None = 0.8
    reprojection_mad_scale: float = 3.5
    filter_motion_outliers: bool = True
    save_target_points: bool = True
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

    observations: list[dict[str, Any]] = []
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
            observations.append(
                {
                    "T_gripper_to_base": robot_sample.T_gripper_to_base,
                    "pose_xyz_rpy": robot_sample.pose_xyz_rpy,
                    "joint_deg": robot_sample.joint_deg,
                    "tool_frame": robot_sample.tool_frame,
                    "T_target_to_camera": detection.T_target_to_camera,
                    "reprojection_rmse_px": detection.reprojection_rmse_px,
                    "corner_count": int(detection.corner_count),
                    "target_center_px": detection.target_center_px,
                    "image_size_px": detection.image_size_px,
                    "object_points": detection.object_points,
                    "image_points": detection.image_points,
                    "intrinsics_matrix": frame.intrinsics_matrix,
                    "distortion_coeffs": frame.distortion_coeffs,
                }
            )
        sleep_sec = float(config.sample_interval_sec)
        if sleep_sec > 0.0:
            time.sleep(sleep_sec)

    if len(observations) < int(config.min_detections):
        raise RuntimeError(
            f"only {len(observations)} valid detections in {frame_count} frames; "
            f"need at least {config.min_detections}."
        )

    before_reprojection = len(observations)
    observations = _filter_reprojection_outliers(observations, config)
    rejected_reprojection = before_reprojection - len(observations)
    if len(observations) < int(config.min_detections):
        raise RuntimeError(
            f"only {len(observations)} detections remain after reprojection filtering; "
            f"rejected {rejected_reprojection}, need at least {config.min_detections}."
        )

    rejected_motion = 0
    if config.filter_motion_outliers:
        before_motion = len(observations)
        observations = _filter_motion_outliers(observations, config)
        rejected_motion = before_motion - len(observations)
        if len(observations) < int(config.min_detections):
            raise RuntimeError(
                f"only {len(observations)} detections remain after motion filtering; "
                f"rejected {rejected_motion}, need at least {config.min_detections}."
            )

    robot_poses = [obs["T_gripper_to_base"] for obs in observations]
    robot_pose6 = [obs["pose_xyz_rpy"] for obs in observations]
    robot_joints = [obs["joint_deg"] for obs in observations]
    robot_tools = [obs["tool_frame"] for obs in observations if obs["tool_frame"]]
    target_poses = [obs["T_target_to_camera"] for obs in observations]
    reprojections = [
        float(obs["reprojection_rmse_px"])
        for obs in observations
        if obs["reprojection_rmse_px"] is not None
    ]
    corner_counts = [int(obs["corner_count"]) for obs in observations]
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

    best_observation = _best_reprojection_observation(observations)
    center_norm = _mean_center_norm(observations)
    target_distance_m = float(np.mean([np.linalg.norm(obs["T_target_to_camera"][:3, 3]) for obs in observations]))
    target_z_m = float(np.mean([obs["T_target_to_camera"][2, 3] for obs in observations]))
    extra_metadata: dict[str, Any] = {}
    if center_norm is not None:
        extra_metadata["target_center_norm"] = center_norm
    extra_metadata["target_distance_m"] = target_distance_m
    extra_metadata["target_z_m"] = target_z_m
    if config.save_target_points and best_observation is not None:
        extra_metadata.update(_target_point_metadata(best_observation))

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
            "burst_detection_count": len(observations),
            "burst_frame_count": frame_count,
            "burst_raw_detection_count": before_reprojection,
            "burst_rejected_reprojection_count": rejected_reprojection,
            "burst_rejected_motion_count": rejected_motion,
            "burst_stable": stable,
            "robot_motion_max_m": robot_motion[0],
            "robot_motion_max_deg": robot_motion[1],
            "target_motion_max_m": target_motion[0],
            "target_motion_max_deg": target_motion[1],
            "robot_tool_frame": _most_common(robot_tools),
            **extra_metadata,
        },
    )
    stats = {
        "detections": len(observations),
        "frames": frame_count,
        "stable": stable,
        "rejected_reprojection": rejected_reprojection,
        "rejected_motion": rejected_motion,
        "robot_motion_max_m": robot_motion[0],
        "robot_motion_max_deg": robot_motion[1],
        "target_motion_max_m": target_motion[0],
        "target_motion_max_deg": target_motion[1],
        "reprojection_rmse_px": float(np.mean(reprojections)) if reprojections else -1.0,
    }
    return sample, stats


def _filter_reprojection_outliers(
    observations: list[dict[str, Any]],
    config: CaptureBurstConfig,
) -> list[dict[str, Any]]:
    kept = list(observations)
    threshold = config.max_reprojection_rmse_px
    if threshold is not None:
        kept = [
            obs
            for obs in kept
            if obs["reprojection_rmse_px"] is None
            or float(obs["reprojection_rmse_px"]) <= float(threshold)
        ]
    values = np.asarray(
        [float(obs["reprojection_rmse_px"]) for obs in kept if obs["reprojection_rmse_px"] is not None],
        dtype=np.float64,
    )
    if values.size >= 6 and config.reprojection_mad_scale > 0.0:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        if mad > 1e-9:
            robust_threshold = median + float(config.reprojection_mad_scale) * 1.4826 * mad
            kept = [
                obs
                for obs in kept
                if obs["reprojection_rmse_px"] is None
                or float(obs["reprojection_rmse_px"]) <= robust_threshold
            ]
    return kept


def _filter_motion_outliers(
    observations: list[dict[str, Any]],
    config: CaptureBurstConfig,
) -> list[dict[str, Any]]:
    if len(observations) < 3:
        return observations
    T_robot_mean = average_transforms(obs["T_gripper_to_base"] for obs in observations)
    T_target_mean = average_transforms(obs["T_target_to_camera"] for obs in observations)
    kept = []
    for obs in observations:
        robot_motion = transform_distance(T_robot_mean, obs["T_gripper_to_base"])
        target_motion = transform_distance(T_target_mean, obs["T_target_to_camera"])
        if (
            robot_motion[0] <= float(config.max_robot_motion_m)
            and robot_motion[1] <= float(config.max_robot_motion_deg)
            and target_motion[0] <= float(config.max_target_motion_m)
            and target_motion[1] <= float(config.max_target_motion_deg)
        ):
            kept.append(obs)
    return kept


def _best_reprojection_observation(observations: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not observations:
        return None
    return min(
        observations,
        key=lambda obs: float(obs["reprojection_rmse_px"])
        if obs["reprojection_rmse_px"] is not None
        else float("inf"),
    )


def _mean_center_norm(observations: list[dict[str, Any]]) -> list[float] | None:
    values = []
    for obs in observations:
        center = obs.get("target_center_px")
        size = obs.get("image_size_px")
        if center is None or size is None:
            continue
        width, height = float(size[0]), float(size[1])
        if width <= 0.0 or height <= 0.0:
            continue
        values.append([float(center[0]) / width, float(center[1]) / height])
    if not values:
        return None
    return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()


def _target_point_metadata(observation: dict[str, Any]) -> dict[str, Any]:
    object_points = observation.get("object_points")
    image_points = observation.get("image_points")
    if object_points is None or image_points is None:
        return {}
    metadata = {
        "target_object_points": np.asarray(object_points, dtype=np.float64).reshape(-1, 3).tolist(),
        "target_image_points": np.asarray(image_points, dtype=np.float64).reshape(-1, 2).tolist(),
        "target_image_size_px": list(observation.get("image_size_px") or []),
        "reprojection_T_gripper_to_base": np.asarray(
            observation["T_gripper_to_base"],
            dtype=np.float64,
        ).tolist(),
        "reprojection_T_target_to_camera": np.asarray(
            observation["T_target_to_camera"],
            dtype=np.float64,
        ).tolist(),
        "reprojection_rmse_px_representative": (
            float(observation["reprojection_rmse_px"])
            if observation.get("reprojection_rmse_px") is not None
            else None
        ),
    }
    metadata["camera_matrix"] = np.asarray(observation["intrinsics_matrix"], dtype=np.float64).tolist()
    metadata["distortion_coeffs"] = np.asarray(observation["distortion_coeffs"], dtype=np.float64).reshape(-1).tolist()
    return metadata


def _most_common(values: list[str]) -> str:
    if not values:
        return ""
    return max(set(values), key=values.count)


def _max_distance(reference: np.ndarray, values: list[np.ndarray]) -> tuple[float, float]:
    trans = 0.0
    rot = 0.0
    for value in values:
        current_trans, current_rot = transform_distance(reference, value)
        trans = max(trans, current_trans)
        rot = max(rot, current_rot)
    return float(trans), float(rot)


def _mean_vector(values: list[list[float]]) -> list[float] | None:
    if not values:
        return None
    return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()


def _now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

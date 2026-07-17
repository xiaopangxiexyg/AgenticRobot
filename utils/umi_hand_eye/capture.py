from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.hand_eye.capture import CaptureBurstConfig, capture_burst
from utils.hand_eye.robot import RobotPoseSample

from .calibration import UmiCalibrationSample


@dataclass
class ViveTrackerPoseReader:
    """Adapt a read-only VivePoseSource channel to the hand-eye burst API."""

    source: Any
    channel_name: str
    tracker_name: str
    max_age_sec: float = 0.25

    def read(self) -> RobotPoseSample:
        channel = self.source.get_latest_channels().get(self.channel_name)
        if channel is None:
            raise RuntimeError(f"Vive channel {self.channel_name!r} has no sample")
        if not channel.valid:
            raise RuntimeError(f"Vive channel {self.channel_name!r} is invalid")
        actual_name = str(channel.metadata.get("tracker_name") or "")
        if actual_name != self.tracker_name:
            raise RuntimeError(
                f"Vive channel {self.channel_name!r} resolved tracker {actual_name!r}, "
                f"expected {self.tracker_name!r}"
            )
        age = time.time() - float(channel.timestamp)
        if age < -0.5 or age > float(self.max_age_sec):
            raise RuntimeError(
                f"Vive tracker sample is stale or clock-invalid: age={age:.3f}s, "
                f"limit={self.max_age_sec:.3f}s"
            )
        pose = np.asarray(channel.value, dtype=float)
        if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
            raise RuntimeError("Vive tracker pose is not a finite 4x4 matrix")
        return RobotPoseSample(
            T_gripper_to_base=pose.copy(),
            pose_xyz_rpy=[],
            joint_deg=[],
            timestamp_sec=float(channel.timestamp),
            arm_err=0,
            sys_err=0,
            tool_frame=f"vive_tracker:{actual_name}",
        )


def capture_umi_burst(
    *,
    camera: Any,
    detector: Any,
    tracker: ViveTrackerPoseReader,
    config: CaptureBurstConfig,
    sample_index: int,
) -> tuple[UmiCalibrationSample, dict[str, float | int | bool]]:
    legacy_sample, stats = capture_burst(
        camera=camera,
        detector=detector,
        robot=tracker,
        config=config,
        sample_index=sample_index,
    )
    metadata = dict(legacy_sample.metadata)
    metadata.update(
        {
            "pairing_method": "static_burst_camera_detection_plus_latest_vive_pose",
            "tracker_name": tracker.tracker_name,
            "tracker_channel_name": tracker.channel_name,
        }
    )
    return (
        UmiCalibrationSample(
            T_tracker_to_vive_world=legacy_sample.T_gripper_to_base,
            T_board_to_camera=legacy_sample.T_target_to_camera,
            timestamp_utc=legacy_sample.timestamp_utc,
            target_reprojection_rmse_px=legacy_sample.target_reprojection_rmse_px,
            target_corner_count=legacy_sample.target_corner_count,
            metadata=metadata,
        ),
        stats,
    )

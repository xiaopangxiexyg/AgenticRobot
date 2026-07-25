"""Stable frontend defaults shared by real-teleop configuration and runtime."""

from __future__ import annotations

from pathlib import Path

from project_paths import default_dataset_root

from ..configs.camera_paths import DEFAULT_REALSENSE_DEVICE_CONFIG_PATH


DEFAULT_CAMERA_CONFIG_PATH = DEFAULT_REALSENSE_DEVICE_CONFIG_PATH
DEFAULT_DATASET_ROOT = default_dataset_root()
DEFAULT_INIT_ARM_PROFILE_PATH = (
    Path(__file__).resolve().parents[2]
    / "robot_motion"
    / "rm65"
    / "arm_init"
    / "qpos_targets"
    / "measured_init_pose.json"
)
INIT_ARM_QPOS_SPEED_BASELINE_PROFILE = "rm65_init_slow"
DEFAULT_INIT_ARM_QPOS_SPEED_SCALE = 5.0
DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2 = (7.0, 7.0, 9.0, 15.0, 15.0, 20.0)
DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M = 0.45
DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG = 160.0


__all__ = [
    "DEFAULT_CAMERA_CONFIG_PATH",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_INIT_ARM_PROFILE_PATH",
    "DEFAULT_INIT_ARM_QPOS_SPEED_SCALE",
    "DEFAULT_POSITION_MAX_ACCELERATION_RAD_S2",
    "DEFAULT_REAL_TELEOP_MAX_TRACKER_ROTATION_JUMP_DEG",
    "DEFAULT_REAL_TELEOP_MAX_TRACKER_TRANSLATION_JUMP_M",
    "INIT_ARM_QPOS_SPEED_BASELINE_PROFILE",
]

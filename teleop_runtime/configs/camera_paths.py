"""Canonical project paths for the real camera configuration inputs.

The device assignment is a teleop station concern.  The RM65 camera registry
is a shared robot/Rig profile so simulation and offline geometry tools can
consume it without depending on ``teleop_runtime``.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = Path(__file__).resolve().parent

DEFAULT_REALSENSE_DEVICE_CONFIG_PATH = CONFIG_ROOT / "devices" / "realsense.json"
DEFAULT_REALSENSE_FRONT_DEVICE_CONFIG_PATH = (
    CONFIG_ROOT / "devices" / "realsense_front.json"
)
DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH = (
    PROJECT_ROOT
    / "robot_motion"
    / "configs"
    / "rm65_camera_registry.json"
)


def realsense_device_config_path() -> Path:
    return DEFAULT_REALSENSE_DEVICE_CONFIG_PATH


def realsense_camera_registry_path() -> Path:
    return DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH


__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH",
    "DEFAULT_REALSENSE_DEVICE_CONFIG_PATH",
    "DEFAULT_REALSENSE_FRONT_DEVICE_CONFIG_PATH",
    "PROJECT_ROOT",
    "realsense_camera_registry_path",
    "realsense_device_config_path",
]

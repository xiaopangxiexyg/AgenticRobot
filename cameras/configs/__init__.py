from __future__ import annotations

from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parent
RM65_CONFIG_DIR = CONFIG_ROOT / "rm65"
RM65_CAMERA_CONFIG_DIR = RM65_CONFIG_DIR / "cameras"
RM65_DEVICE_CONFIG_DIR = RM65_CONFIG_DIR / "devices"

DEFAULT_REALSENSE_DEVICE_CONFIG_PATH = RM65_DEVICE_CONFIG_DIR / "realsense.json"
DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH = RM65_CAMERA_CONFIG_DIR / "realsense_registry.json"


def realsense_device_config_path() -> Path:
    return DEFAULT_REALSENSE_DEVICE_CONFIG_PATH


def realsense_camera_registry_path() -> Path:
    return DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH


__all__ = [
    "CONFIG_ROOT",
    "DEFAULT_REALSENSE_CAMERA_REGISTRY_PATH",
    "DEFAULT_REALSENSE_DEVICE_CONFIG_PATH",
    "RM65_CAMERA_CONFIG_DIR",
    "RM65_CONFIG_DIR",
    "RM65_DEVICE_CONFIG_DIR",
    "realsense_camera_registry_path",
    "realsense_device_config_path",
]

"""Compatibility exports for runtime-independent scene layout conventions."""

from __future__ import annotations

from robot_motion.scene_layout import (
    CameraPose,
    rm65_base_y_for_side,
    rm65_base_yaw_for_side,
    rm65_viewer_camera_pose,
)

__all__ = [
    "CameraPose",
    "rm65_base_y_for_side",
    "rm65_base_yaw_for_side",
    "rm65_viewer_camera_pose",
]

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraPose:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]


def rm65_base_y_for_side(side: str, left_right_base_distance: float) -> float:
    half_distance = float(left_right_base_distance) * 0.5
    normalized_side = str(side).strip().lower()
    if normalized_side == "left":
        return -half_distance
    if normalized_side == "right":
        return half_distance
    raise ValueError(f"Unsupported side '{side}'. Expected 'left' or 'right'.")


def rm65_base_yaw_for_side(side: str) -> float:
    normalized_side = str(side).strip().lower()
    if normalized_side == "left":
        return -math.pi / 2.0
    if normalized_side == "right":
        return math.pi / 2.0
    raise ValueError(f"Unsupported side '{side}'. Expected 'left' or 'right'.")


def rm65_viewer_camera_pose() -> CameraPose:
    x = 0.9
    z = 0.55
    return CameraPose(
        # Place the viewer on desk +X looking toward the origin (-X).
        # In this viewpoint, screen-left matches desk -Y / logical left_arm,
        # and screen-right matches desk +Y / logical right_arm.
        xyz=(x, 0.0, z),
        rpy=(0.0, -math.atan2(z, x), math.pi),
    )

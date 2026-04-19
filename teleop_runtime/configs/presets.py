from __future__ import annotations

import math

import numpy as np

from ..calibration import ArmCalibration, TeleopCalibration
from ..scene_layout import rm65_base_y_for_side, rm65_base_yaw_for_side
from ..transforms import identity_pose


ANNOTATED_DUAL_RM65_BASE_DISTANCE_M = 0.525


def _pose_xyz_rz(x: float, y: float, z: float, yaw_rad: float) -> np.ndarray:
    pose = identity_pose()
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    pose[:3, :3] = np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pose[:3, 3] = np.array([x, y, z], dtype=float)
    return pose


def build_annotated_dual_rm65_calibration(
    *,
    left_right_base_distance: float = ANNOTATED_DUAL_RM65_BASE_DISTANCE_M,
    workspace_min: np.ndarray | tuple[float, float, float] = (-0.60, -0.80, 0.00),
    workspace_max: np.ndarray | tuple[float, float, float] = (0.60, 0.80, 0.90),
) -> TeleopCalibration:
    workspace_min = np.asarray(workspace_min, dtype=float)
    workspace_max = np.asarray(workspace_max, dtype=float)

    return TeleopCalibration(
        T_W_O=identity_pose(),
        T_O_D=identity_pose(),
        arms={
            "left_arm": ArmCalibration(
                name="left_arm",
                side="left",
                T_D_B=_pose_xyz_rz(
                    0.0,
                    rm65_base_y_for_side("left", left_right_base_distance),
                    0.0,
                    rm65_base_yaw_for_side("left"),
                ),
                workspace_min=workspace_min.copy(),
                workspace_max=workspace_max.copy(),
                metadata={
                    "preset": "annotated_dual_rm65",
                    "base_distance_m": float(left_right_base_distance),
                    "desk_slot": "-y(screen_left)",
                },
            ),
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=_pose_xyz_rz(
                    0.0,
                    rm65_base_y_for_side("right", left_right_base_distance),
                    0.0,
                    rm65_base_yaw_for_side("right"),
                ),
                workspace_min=workspace_min.copy(),
                workspace_max=workspace_max.copy(),
                metadata={
                    "preset": "annotated_dual_rm65",
                    "base_distance_m": float(left_right_base_distance),
                    "desk_slot": "+y(screen_right)",
                },
            ),
        },
        metadata={
            "preset": "annotated_dual_rm65",
            "base_distance_m": float(left_right_base_distance),
            "note": (
                "Dual-arm desk/base relation for teleop_runtime where desk axes are "
                "+x=forward, +y=left, +z=up, while the default viewer places "
                "left_arm at desk -Y (screen-left) and right_arm at desk +Y (screen-right)."
            ),
        },
    )

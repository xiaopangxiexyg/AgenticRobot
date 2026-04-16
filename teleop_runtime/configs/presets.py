from __future__ import annotations

import math

import numpy as np

from ..calibration import ArmCalibration, TeleopCalibration
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
    half_distance = float(left_right_base_distance) * 0.5

    return TeleopCalibration(
        T_W_O=identity_pose(),
        T_O_D=identity_pose(),
        arms={
            "left_arm": ArmCalibration(
                name="left_arm",
                side="left",
                T_D_B=_pose_xyz_rz(0.0, -half_distance, 0.0, -math.pi / 2.0),
                workspace_min=workspace_min.copy(),
                workspace_max=workspace_max.copy(),
                metadata={
                    "preset": "annotated_dual_rm65",
                    "base_distance_m": float(left_right_base_distance),
                },
            ),
            "right_arm": ArmCalibration(
                name="right_arm",
                side="right",
                T_D_B=_pose_xyz_rz(0.0, half_distance, 0.0, math.pi / 2.0),
                workspace_min=workspace_min.copy(),
                workspace_max=workspace_max.copy(),
                metadata={
                    "preset": "annotated_dual_rm65",
                    "base_distance_m": float(left_right_base_distance),
                },
            ),
        },
        metadata={
            "preset": "annotated_dual_rm65",
            "base_distance_m": float(left_right_base_distance),
            "note": (
                "Dual-arm desk/base relation aligned with the previous annotated "
                "bimanual visualization setup."
            ),
        },
    )

from __future__ import annotations

from typing import Any

import numpy as np

from .transforms import as_pose_matrix


DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M = 0.04


# Local rotation that maps the calibrated tracker-hand control frame to the
# RM65 hand_base TCP frame used by IK / SAPIEN / real-arm state.
#
# Observed mapping:
# - tracker +X -> tcp -Z
# - tracker +Y -> tcp -Y
# - tracker +Z -> tcp -X
TRACKER_HAND_TO_RM65_TCP_ROTATION = np.array(
    [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def tracker_hand_to_rm65_tcp_rotation() -> np.ndarray:
    return TRACKER_HAND_TO_RM65_TCP_ROTATION.copy()


def vive_tracker_height_correction_offset(
    height_m: float = DEFAULT_VIVE_TRACKER_HEIGHT_CORRECTION_M,
) -> np.ndarray:
    return np.asarray([0.0, 0.0, -float(height_m)], dtype=float)


def apply_local_pose_rotation(pose: Any, local_rotation: Any) -> np.ndarray:
    pose_matrix = as_pose_matrix(pose)
    rotation = np.asarray(local_rotation, dtype=float).reshape(3, 3)
    corrected = pose_matrix.copy()
    corrected[:3, :3] = corrected[:3, :3] @ rotation
    return corrected

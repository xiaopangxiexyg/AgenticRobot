from __future__ import annotations

from typing import Any

import numpy as np

from .transforms import as_pose_matrix


LEGACY_CANONICAL_HAND_RPY_DEG = np.array([0.0, 0.0, 0.0], dtype=float)
DEFAULT_CANONICAL_HAND_RPY_DEG = LEGACY_CANONICAL_HAND_RPY_DEG.copy()
DEFAULT_TRACKER_TO_HAND_ROTATION = np.eye(3, dtype=float)
TRACKER_OFFSET_CALIBRATION_VERSION = 1


def canonical_hand_rpy_is_legacy(value: Any) -> bool:
    if value is None:
        return True
    array = np.asarray(value, dtype=float).reshape(3)
    return bool(np.allclose(array, LEGACY_CANONICAL_HAND_RPY_DEG, atol=1e-6))


def canonical_hand_rpy_default() -> np.ndarray:
    return DEFAULT_CANONICAL_HAND_RPY_DEG.copy()


def canonical_tracker_to_hand_rotation() -> np.ndarray:
    return DEFAULT_TRACKER_TO_HAND_ROTATION.copy()


def apply_tracker_hand_frame_correction(pose: Any) -> np.ndarray:
    return as_pose_matrix(pose)

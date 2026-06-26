"""RM65 eye-to-hand calibration utilities.

The package keeps import-time dependencies light. Camera, OpenCV, RealSense,
and RM65 SDK modules are imported lazily by the command-line tools that need
them.
"""

from .calibration import (
    CalibrationResult,
    CalibrationSample,
    evaluate_eye_to_hand,
    load_result,
    load_samples,
    quality_warnings,
    save_result,
    save_samples,
    solve_eye_to_hand,
)
from .camera import CAMERA_IDS, DEFAULT_CAMERA_SERIALS
from .transforms import (
    average_transforms,
    invert_transform,
    make_transform,
    pose6_xyz_rpy_to_matrix,
    rotation_angle_deg,
    transform_distance,
)

__all__ = [
    "CalibrationResult",
    "CalibrationSample",
    "CAMERA_IDS",
    "DEFAULT_CAMERA_SERIALS",
    "average_transforms",
    "evaluate_eye_to_hand",
    "invert_transform",
    "load_result",
    "load_samples",
    "make_transform",
    "pose6_xyz_rpy_to_matrix",
    "quality_warnings",
    "rotation_angle_deg",
    "save_result",
    "save_samples",
    "solve_eye_to_hand",
    "transform_distance",
]

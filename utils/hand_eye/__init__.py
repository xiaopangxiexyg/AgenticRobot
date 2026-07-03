"""RM65 eye-in-hand calibration utilities.

The package keeps import-time dependencies light. Camera, OpenCV, RealSense,
and RM65 SDK modules are imported lazily by the command-line tools that need
them.
"""

from .calibration import (
    CalibrationResult,
    CalibrationSample,
    evaluate_eye_in_hand,
    evaluate_hand_eye,
    load_result,
    load_samples,
    quality_warnings,
    refine_eye_in_hand,
    sample_plan_summary,
    save_result,
    save_samples,
    solve_eye_in_hand,
    solve_hand_eye,
)
from .camera import CAMERA_IDS, DEFAULT_CAMERA_SERIALS
from .transforms import (
    average_transforms,
    compose_transforms,
    invert_transform,
    make_transform,
    pose6_xyz_rpy_to_matrix,
    rotation_matrix_from_euler_xyz,
    rotation_angle_deg,
    transform_distance,
)

__all__ = [
    "CalibrationResult",
    "CalibrationSample",
    "CAMERA_IDS",
    "DEFAULT_CAMERA_SERIALS",
    "average_transforms",
    "compose_transforms",
    "evaluate_eye_in_hand",
    "evaluate_hand_eye",
    "invert_transform",
    "load_result",
    "load_samples",
    "make_transform",
    "pose6_xyz_rpy_to_matrix",
    "quality_warnings",
    "refine_eye_in_hand",
    "rotation_matrix_from_euler_xyz",
    "rotation_angle_deg",
    "sample_plan_summary",
    "save_result",
    "save_samples",
    "solve_eye_in_hand",
    "solve_hand_eye",
    "transform_distance",
]

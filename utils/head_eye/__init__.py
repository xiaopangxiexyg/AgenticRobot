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
    load_samples_with_metadata,
    quality_warnings,
    save_result,
    save_samples,
    solve_eye_to_hand,
)
from .camera import (
    CAMERA_IDS,
    DEFAULT_CAMERA_SERIALS,
    VideoStreamProfileInfo,
    available_video_profiles,
)
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
    "VideoStreamProfileInfo",
    "average_transforms",
    "available_video_profiles",
    "compose_transforms",
    "evaluate_eye_to_hand",
    "invert_transform",
    "load_result",
    "load_samples",
    "load_samples_with_metadata",
    "make_transform",
    "pose6_xyz_rpy_to_matrix",
    "quality_warnings",
    "rotation_matrix_from_euler_xyz",
    "rotation_angle_deg",
    "save_result",
    "save_samples",
    "solve_eye_to_hand",
    "transform_distance",
]

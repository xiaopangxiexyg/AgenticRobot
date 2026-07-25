"""Compatibility exports for the historical debug calibration-flow path."""

from ..calibration_flow import (
    AxisFitQuality,
    SignedAxisFit,
    VirtualDeskCalibrationResult,
    _angle_deg,
    _normalize,
    average_pose,
    compose_tracker_offset,
    euler_xyz_deg_to_matrix,
    fit_signed_axis_from_poses,
    fit_virtual_desk_frame,
    pose_from_rotation_translation,
    tracker_offset_from_neutral_pose,
)

__all__ = [
    "AxisFitQuality",
    "SignedAxisFit",
    "VirtualDeskCalibrationResult",
    "average_pose",
    "compose_tracker_offset",
    "euler_xyz_deg_to_matrix",
    "fit_signed_axis_from_poses",
    "fit_virtual_desk_frame",
    "pose_from_rotation_translation",
    "tracker_offset_from_neutral_pose",
]

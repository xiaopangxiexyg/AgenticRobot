from .triad_openvr import (
    convert_to_euler,
    convert_to_quaternion,
    convert_to_quaternion_robust,
    convert_to_quaternion_tfs,
    get_pose,
    matrix_to_flat_list,
    pose_sample_buffer,
    triad_openvr as _triad_openvr_class,
    update_text,
    vr_tracked_device,
    vr_tracking_reference,
)

triad_openvr = _triad_openvr_class
triad_openvr.triad_openvr = _triad_openvr_class

__all__ = [
    "convert_to_euler",
    "convert_to_quaternion",
    "convert_to_quaternion_robust",
    "convert_to_quaternion_tfs",
    "get_pose",
    "matrix_to_flat_list",
    "pose_sample_buffer",
    "triad_openvr",
    "update_text",
    "vr_tracked_device",
    "vr_tracking_reference",
]

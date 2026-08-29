"""Compatibility exports for shared calibration rigid-transform helpers."""

from utils.calibration_common import transforms as _shared

EPS = _shared.EPS
_as_float_array = _shared._as_float_array
as_transform = _shared.as_transform
average_quaternion_xyzw = _shared.average_quaternion_xyzw
average_transforms = _shared.average_transforms
compose_transforms = _shared.compose_transforms
invert_transform = _shared.invert_transform
make_transform = _shared.make_transform
matrix_to_quaternion_xyzw = _shared.matrix_to_quaternion_xyzw
pose6_xyz_rpy_to_matrix = _shared.pose6_xyz_rpy_to_matrix
project_to_rotation = _shared.project_to_rotation
quaternion_xyzw_to_matrix = _shared.quaternion_xyzw_to_matrix
rotation_angle_deg = _shared.rotation_angle_deg
rotation_matrix_from_euler_xyz = _shared.rotation_matrix_from_euler_xyz
transform_distance = _shared.transform_distance
transform_to_list = _shared.transform_to_list

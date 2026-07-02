from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def as_pose_matrix(pose: np.ndarray) -> np.ndarray:
    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"pose must be a 4x4 matrix, got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("pose matrix contains non-finite values")
    return matrix


def pose_delta_metrics(reference_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    reference = as_pose_matrix(reference_pose)
    target = as_pose_matrix(target_pose)
    translation_m = float(np.linalg.norm(target[:3, 3] - reference[:3, 3]))
    relative_rot = target[:3, :3] @ reference[:3, :3].T
    rotation_deg = float(np.rad2deg(R.from_matrix(relative_rot).magnitude()))
    return translation_m, rotation_deg


def pose6_to_matrix(pose6: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    values = np.asarray(pose6, dtype=float).reshape(6)
    matrix = np.eye(4, dtype=float)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = R.from_euler("xyz", values[3:6], degrees=False).as_matrix()
    return matrix


def matrix_to_pose6(matrix: np.ndarray) -> list[float]:
    pose = as_pose_matrix(matrix)
    rpy = R.from_matrix(pose[:3, :3]).as_euler("xyz", degrees=False)
    return [
        float(pose[0, 3]),
        float(pose[1, 3]),
        float(pose[2, 3]),
        float(rpy[0]),
        float(rpy[1]),
        float(rpy[2]),
    ]


def matrix_to_rm_quat_pose(matrix: np.ndarray) -> list[float]:
    pose = as_pose_matrix(matrix)
    quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat()
    return [
        float(pose[0, 3]),
        float(pose[1, 3]),
        float(pose[2, 3]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def relax_pose_toward_reference(
    target_pose: np.ndarray,
    reference_pose: np.ndarray,
    *,
    max_position_m: float,
    max_orientation_deg: float,
) -> np.ndarray:
    target = as_pose_matrix(target_pose)
    reference = as_pose_matrix(reference_pose)
    relaxed = target.copy()

    delta_t = reference[:3, 3] - target[:3, 3]
    delta_norm = float(np.linalg.norm(delta_t))
    max_position = max(0.0, float(max_position_m))
    if delta_norm > 1e-12 and max_position > 0.0:
        relaxed[:3, 3] = target[:3, 3] + delta_t * min(1.0, max_position / delta_norm)

    target_rot = R.from_matrix(target[:3, :3])
    reference_rot = R.from_matrix(reference[:3, :3])
    delta_rot = reference_rot * target_rot.inv()
    delta_angle = float(delta_rot.magnitude())
    max_angle = np.deg2rad(max(0.0, float(max_orientation_deg)))
    if delta_angle > 1e-12 and max_angle > 0.0:
        step_angle = min(delta_angle, max_angle)
        step_rot = R.from_rotvec(delta_rot.as_rotvec() / delta_angle * step_angle)
        relaxed[:3, :3] = (step_rot * target_rot).as_matrix()

    return relaxed

from __future__ import annotations

from typing import Any

import numpy as np


def identity_pose() -> np.ndarray:
    return np.eye(4, dtype=float)


def _quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def as_pose_matrix(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape == (4, 4):
        return pose.copy()
    if pose.shape == (16,):
        return pose.reshape(4, 4).copy()
    if pose.shape == (12,):
        out = identity_pose()
        out[:3, 3] = pose[:3]
        out[:3, :3] = pose[3:].reshape(3, 3)
        return out
    if pose.shape == (7,):
        out = identity_pose()
        out[:3, 3] = pose[:3]
        out[:3, :3] = _quaternion_to_matrix(*pose[3:7])
        return out
    raise ValueError(f"Unsupported pose shape: {pose.shape}")


def rotation_is_orthonormal(rotation: np.ndarray, atol: float = 1e-3) -> bool:
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape != (3, 3):
        return False
    should_be_identity = rotation.T @ rotation
    determinant = np.linalg.det(rotation)
    return np.allclose(should_be_identity, np.eye(3), atol=atol) and np.isclose(
        determinant, 1.0, atol=atol
    )


def validate_pose_matrix(value: Any, atol: float = 1e-3) -> tuple[bool, str]:
    try:
        pose = as_pose_matrix(value)
    except ValueError as exc:
        return False, str(exc)
    if not np.all(np.isfinite(pose)):
        return False, "Pose contains non-finite values."
    if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        return False, "Pose last row is not [0, 0, 0, 1]."
    if not rotation_is_orthonormal(pose[:3, :3], atol=atol):
        return False, "Rotation matrix is not orthonormal."
    return True, ""


def invert_pose(value: Any) -> np.ndarray:
    pose = as_pose_matrix(value)
    out = identity_pose()
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ pose[:3, 3]
    return out


def compose_poses(*poses: Any) -> np.ndarray:
    out = identity_pose()
    for pose in poses:
        out = out @ as_pose_matrix(pose)
    return out


def translation_distance(a: Any, b: Any) -> float:
    pose_a = as_pose_matrix(a)
    pose_b = as_pose_matrix(b)
    return float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))


def rotation_angle_rad(a: Any, b: Any) -> float:
    rot_a = as_pose_matrix(a)[:3, :3]
    rot_b = as_pose_matrix(b)[:3, :3]
    delta = rot_a.T @ rot_b
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(trace))


def clip_pose_translation(
    pose: Any, min_xyz: np.ndarray | None = None, max_xyz: np.ndarray | None = None
) -> np.ndarray:
    out = as_pose_matrix(pose)
    if min_xyz is not None:
        out[:3, 3] = np.maximum(out[:3, 3], np.asarray(min_xyz, dtype=float))
    if max_xyz is not None:
        out[:3, 3] = np.minimum(out[:3, 3], np.asarray(max_xyz, dtype=float))
    return out

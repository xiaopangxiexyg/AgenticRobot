from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


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


def parse_pose_matrix(value: Any) -> np.ndarray:
    """Parse supported pose encodings into a 4x4 matrix copy.

    Accepted inputs are 4x4 matrices, flattened 16-value matrices,
    12-value xyz+row-major-rotation vectors, and 7-value xyz+quat_xyzw poses.
    Use robot_motion.pose.as_strict_pose_matrix when only finite 4x4 matrices
    should be accepted.
    """
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


as_pose_matrix = parse_pose_matrix


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
        pose = parse_pose_matrix(value)
    except ValueError as exc:
        return False, str(exc)
    if not np.all(np.isfinite(pose)):
        return False, "Pose contains non-finite values."
    if not np.allclose(pose[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=atol):
        return False, "Pose last row is not [0, 0, 0, 1]."
    if not rotation_is_orthonormal(pose[:3, :3], atol=atol):
        return False, "Rotation matrix is not orthonormal."
    return True, ""


def project_near_rigid_pose(
    value: Any,
    *,
    name: str = "pose",
    max_orthogonality_error: float = 1e-2,
    max_determinant_error: float = 1e-2,
    max_projection_frobenius: float = 1e-2,
    homogeneous_atol: float = 1e-5,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Project bounded matrix drift onto SO(3) without accepting bad poses.

    Device pose protocols often encode a rigid rotation as nine floating-point
    values.  Small transport/backend drift is representational noise, while a
    reflection or a materially distorted basis is an invalid pose.  This
    helper distinguishes the two and returns explicit projection diagnostics.
    """

    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.all(np.isfinite(pose)):
        raise ValueError(f"{name}: Pose contains non-finite values.")
    if not np.allclose(
        pose[3],
        np.array([0.0, 0.0, 0.0, 1.0]),
        atol=float(homogeneous_atol),
        rtol=0.0,
    ):
        raise ValueError(f"{name}: Pose last row is not [0, 0, 0, 1].")
    limits = {
        "max_orthogonality_error": float(max_orthogonality_error),
        "max_determinant_error": float(max_determinant_error),
        "max_projection_frobenius": float(max_projection_frobenius),
    }
    if any(not np.isfinite(limit) or limit < 0.0 for limit in limits.values()):
        raise ValueError("Rigid-pose projection limits must be finite and non-negative.")

    raw_rotation = pose[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(raw_rotation.T @ raw_rotation - np.eye(3)))
    )
    determinant = float(np.linalg.det(raw_rotation))
    determinant_error = abs(determinant - 1.0)
    if determinant <= 0.0:
        raise ValueError(
            f"{name}: Rotation determinant must be positive, got {determinant:.6g}."
        )
    try:
        u, _singular_values, vt = np.linalg.svd(raw_rotation)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name}: Rotation SVD failed: {exc}") from exc
    projected = u @ vt
    if float(np.linalg.det(projected)) <= 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    projection_frobenius = float(
        np.linalg.norm(raw_rotation - projected, ord="fro")
    )
    if (
        orthogonality_error > limits["max_orthogonality_error"]
        or determinant_error > limits["max_determinant_error"]
        or projection_frobenius > limits["max_projection_frobenius"]
    ):
        raise ValueError(
            f"{name}: Rotation is too far from SO(3) "
            f"(orth_err={orthogonality_error:.6g}, det={determinant:.6g}, "
            f"projection_fro={projection_frobenius:.6g})."
        )
    normalized = pose.copy()
    normalized[:3, :3] = projected
    normalized[3] = np.array([0.0, 0.0, 0.0, 1.0])
    valid, reason = validate_pose_matrix(normalized, atol=1e-8)
    if not valid:
        raise ValueError(f"{name}: {reason}")
    return normalized, {
        "rotation_projected": bool(projection_frobenius > 1e-12),
        "rotation_orthogonality_error": orthogonality_error,
        "rotation_determinant": determinant,
        "rotation_projection_frobenius": projection_frobenius,
    }


def invert_pose(value: Any) -> np.ndarray:
    pose = parse_pose_matrix(value)
    out = identity_pose()
    out[:3, :3] = pose[:3, :3].T
    out[:3, 3] = -out[:3, :3] @ pose[:3, 3]
    return out


def compose_poses(*poses: Any) -> np.ndarray:
    out = identity_pose()
    for pose in poses:
        out = out @ parse_pose_matrix(pose)
    return out


def translation_distance(a: Any, b: Any) -> float:
    pose_a = parse_pose_matrix(a)
    pose_b = parse_pose_matrix(b)
    return float(np.linalg.norm(pose_a[:3, 3] - pose_b[:3, 3]))


def rotation_angle_rad(a: Any, b: Any) -> float:
    rot_a = parse_pose_matrix(a)[:3, :3]
    rot_b = parse_pose_matrix(b)[:3, :3]
    delta = rot_a.T @ rot_b
    trace = np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(trace))


def clip_pose_translation(
    pose: Any, min_xyz: np.ndarray | None = None, max_xyz: np.ndarray | None = None
) -> np.ndarray:
    out = parse_pose_matrix(pose)
    if min_xyz is not None:
        out[:3, 3] = np.maximum(out[:3, 3], np.asarray(min_xyz, dtype=float))
    if max_xyz is not None:
        out[:3, 3] = np.minimum(out[:3, 3], np.asarray(max_xyz, dtype=float))
    return out


def interpolate_pose(start_pose: Any, end_pose: Any, alpha: float) -> np.ndarray:
    start = parse_pose_matrix(start_pose)
    end = parse_pose_matrix(end_pose)
    blend = float(np.clip(alpha, 0.0, 1.0))
    if blend <= 0.0:
        return start
    if blend >= 1.0:
        return end

    out = identity_pose()
    out[:3, 3] = (1.0 - blend) * start[:3, 3] + blend * end[:3, 3]
    key_rots = R.from_matrix(np.stack((start[:3, :3], end[:3, :3]), axis=0))
    out[:3, :3] = Slerp([0.0, 1.0], key_rots)([blend]).as_matrix()[0]
    return out


__all__ = [
    "as_pose_matrix",
    "clip_pose_translation",
    "compose_poses",
    "identity_pose",
    "interpolate_pose",
    "invert_pose",
    "parse_pose_matrix",
    "project_near_rigid_pose",
    "rotation_angle_rad",
    "rotation_is_orthonormal",
    "translation_distance",
    "validate_pose_matrix",
]

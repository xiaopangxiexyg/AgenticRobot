"""Portable rigid-transform helpers for robot and WebXR pose payloads.

The payload convention is inherited from AR5 ``tcp_pose`` messages:
position is metres and orientation is an ``xyzw`` quaternion. NumPy matrices
use the usual row/column indexing; ``matrix_to_column_major`` prepares them for
WebXR/WebGL JavaScript arrays.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if vector.shape != (length,):
        raise ValueError(f"{name} must contain exactly {length} values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert a finite, non-zero xyzw quaternion to a 3x3 rotation."""
    x, y, z, w = _finite_vector(quaternion, 4, "orientation_quat_xyzw")
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm < 1e-12:
        raise ValueError("orientation_quat_xyzw must be non-zero")
    x, y, z, w = (np.asarray([x, y, z, w]) / norm).tolist()
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """Convert a valid 3x3 rotation matrix to a normalized xyzw quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            )
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 - matrix[0, 0] + matrix[1, 1] - matrix[2, 2])
            )
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = 2.0 * np.sqrt(
                max(0.0, 1.0 - matrix[0, 0] - matrix[1, 1] + matrix[2, 2])
            )
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray([x, y, z, w], dtype=float)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("rotation cannot be converted to a quaternion")
    return (quaternion / norm).tolist()


def pose_payload_to_matrix(payload: dict[str, Any]) -> np.ndarray:
    """Validate an AR5-style pose payload and return ``T_parent_from_child``."""
    if not isinstance(payload, dict):
        raise TypeError("pose payload must be a mapping")
    position = _finite_vector(payload.get("position"), 3, "position")
    rotation = quaternion_xyzw_to_matrix(payload.get("orientation_quat_xyzw"))
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def matrix_to_pose_payload(transform: np.ndarray) -> dict[str, list[float]]:
    matrix = validate_rigid_transform(transform)
    return {
        "position": matrix[:3, 3].tolist(),
        "orientation_quat_xyzw": matrix_to_quaternion_xyzw(matrix[:3, :3]),
    }


def validate_rigid_transform(
    transform: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=tolerance):
        raise ValueError("transform last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance):
        raise ValueError("transform rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=tolerance):
        raise ValueError("transform rotation must have determinant +1")
    return matrix.copy()


def rigid_inverse(transform: np.ndarray) -> np.ndarray:
    matrix = validate_rigid_transform(transform)
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def transform_points(
    transform: np.ndarray, points: Sequence[Sequence[float]]
) -> np.ndarray:
    matrix = validate_rigid_transform(transform)
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("points must be a finite Nx3 array")
    return (matrix[:3, :3] @ values.T).T + matrix[:3, 3]


def matrix_to_column_major(transform: np.ndarray) -> list[float]:
    """Flatten a rigid matrix for ``Float32Array``/WebGL consumption."""
    matrix = validate_rigid_transform(transform)
    return matrix.reshape(16, order="F").tolist()

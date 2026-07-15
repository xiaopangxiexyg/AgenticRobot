from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (r[2, 1] - r[1, 2]) / scale
        qy = (r[0, 2] - r[2, 0]) / scale
        qz = (r[1, 0] - r[0, 1]) / scale
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        scale = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / scale
        qx = 0.25 * scale
        qy = (r[0, 1] + r[1, 0]) / scale
        qz = (r[0, 2] + r[2, 0]) / scale
    elif r[1, 1] > r[2, 2]:
        scale = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / scale
        qx = (r[0, 1] + r[1, 0]) / scale
        qy = 0.25 * scale
        qz = (r[1, 2] + r[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / scale
        qx = (r[0, 2] + r[2, 0]) / scale
        qy = (r[1, 2] + r[2, 1]) / scale
        qz = 0.25 * scale
    quat = np.array([qw, qx, qy, qz], dtype=float)
    norm = float(np.linalg.norm(quat))
    if norm <= 0.0:
        raise ValueError("Cannot convert rotation matrix to quaternion.")
    return quat / norm


def quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = np.asarray(quat, dtype=float).reshape(4)
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0:
        raise ValueError("Quaternion norm must be positive.")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def pose_from_xyz_rpy(xyz: Sequence[float], rpy: Sequence[float]) -> np.ndarray:
    pose = np.eye(4, dtype=float)
    pose[:3, :3] = rpy_to_matrix(rpy)
    pose[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return pose


def pose_from_xyz_quat_xyzw(values: Sequence[float]) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = np.asarray(values, dtype=float).reshape(7)
    pose = np.eye(4, dtype=float)
    pose[:3, 3] = [x, y, z]
    pose[:3, :3] = quat_wxyz_to_matrix([qw, qx, qy, qz])
    return pose


def pose_to_xyz_quat_xyzw(pose: np.ndarray) -> list[float]:
    matrix = np.asarray(pose, dtype=float).reshape(4, 4)
    qw, qx, qy, qz = matrix_to_quat_wxyz(matrix[:3, :3])
    return [
        float(matrix[0, 3]),
        float(matrix[1, 3]),
        float(matrix[2, 3]),
        float(qx),
        float(qy),
        float(qz),
        float(qw),
    ]


def pose_to_pos_quat_text(pose: np.ndarray) -> tuple[str, str]:
    matrix = np.asarray(pose, dtype=float).reshape(4, 4)
    pos = matrix[:3, 3]
    quat = matrix_to_quat_wxyz(matrix[:3, :3])
    return format_float_list(pos), format_float_list(quat)


def format_float_list(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def parse_float_list(text: str | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if text is None or not str(text).strip():
        return default
    return tuple(float(part) for part in str(text).replace(",", " ").split())


def rotation_error_world(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    current_r = np.asarray(current, dtype=float).reshape(3, 3)
    target_r = np.asarray(target, dtype=float).reshape(3, 3)
    return 0.5 * (
        np.cross(current_r[:, 0], target_r[:, 0])
        + np.cross(current_r[:, 1], target_r[:, 1])
        + np.cross(current_r[:, 2], target_r[:, 2])
    )


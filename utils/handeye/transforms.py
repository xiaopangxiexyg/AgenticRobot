from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


EPS = 1e-12


def _as_float_array(value: object, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def make_transform(rotation: object, translation: object) -> np.ndarray:
    """Return a 4x4 transform from a 3x3 rotation and 3D translation."""

    rot = _as_float_array(rotation, shape=(3, 3), name="rotation")
    trans = np.asarray(translation, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(trans)):
        raise ValueError("translation contains non-finite values.")
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = project_to_rotation(rot)
    out[:3, 3] = trans
    return out


def as_transform(value: object, *, name: str = "transform") -> np.ndarray:
    matrix = _as_float_array(value, shape=(4, 4), name=name).copy()
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError(f"{name} must have homogeneous last row [0, 0, 0, 1].")
    matrix[:3, :3] = project_to_rotation(matrix[:3, :3])
    matrix[3] = np.array([0.0, 0.0, 0.0, 1.0])
    return matrix


def project_to_rotation(rotation: object) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3)."""

    rot = _as_float_array(rotation, shape=(3, 3), name="rotation")
    u, _, vt = np.linalg.svd(rot)
    projected = u @ vt
    if np.linalg.det(projected) < 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    return projected


def invert_transform(value: object) -> np.ndarray:
    matrix = as_transform(value)
    out = np.eye(4, dtype=np.float64)
    rot = matrix[:3, :3]
    trans = matrix[:3, 3]
    out[:3, :3] = rot.T
    out[:3, 3] = -rot.T @ trans
    return out


def compose_transforms(*values: object) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    for index, value in enumerate(values):
        out = out @ as_transform(value, name=f"transform[{index}]")
    return as_transform(out)


def rotation_matrix_from_euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """RM65/static XYZ Euler angles in radians, equivalent to scipy ``xyz``."""

    sx, cx = math.sin(float(rx)), math.cos(float(rx))
    sy, cy = math.sin(float(ry)), math.cos(float(ry))
    sz, cz = math.sin(float(rz)), math.cos(float(rz))
    rx_mat = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_mat = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_mat = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_mat @ ry_mat @ rx_mat


def pose6_xyz_rpy_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """Convert RM65 ``[x, y, z, rx, ry, rz]`` pose to a 4x4 matrix.

    Translation is meters. Rotation angles are radians.
    """

    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.shape != (6,):
        raise ValueError(f"pose must have shape (6,), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("pose contains non-finite values.")
    return make_transform(rotation_matrix_from_euler_xyz(*values[3:6]), values[:3])


def matrix_to_quaternion_xyzw(rotation: object) -> np.ndarray:
    rot = project_to_rotation(rotation)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rot[2, 1] - rot[1, 2]) / scale
        qy = (rot[0, 2] - rot[2, 0]) / scale
        qz = (rot[1, 0] - rot[0, 1]) / scale
    else:
        diag = np.diag(rot)
        axis = int(np.argmax(diag))
        if axis == 0:
            scale = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / scale
            qx = 0.25 * scale
            qy = (rot[0, 1] + rot[1, 0]) / scale
            qz = (rot[0, 2] + rot[2, 0]) / scale
        elif axis == 1:
            scale = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / scale
            qx = (rot[0, 1] + rot[1, 0]) / scale
            qy = 0.25 * scale
            qz = (rot[1, 2] + rot[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / scale
            qx = (rot[0, 2] + rot[2, 0]) / scale
            qy = (rot[1, 2] + rot[2, 1]) / scale
            qz = 0.25 * scale
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= EPS:
        raise ValueError("rotation produced a zero quaternion.")
    return quat / norm


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= EPS:
        raise ValueError("quaternion norm is zero.")
    x, y, z, w = quat / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_quaternion_xyzw(quaternions: Iterable[Sequence[float]]) -> np.ndarray:
    accumulator = np.zeros((4, 4), dtype=np.float64)
    count = 0
    reference: np.ndarray | None = None
    for quaternion in quaternions:
        quat = np.asarray(quaternion, dtype=np.float64).reshape(4)
        norm = float(np.linalg.norm(quat))
        if norm <= EPS:
            continue
        quat = quat / norm
        if reference is None:
            reference = quat
        elif float(np.dot(reference, quat)) < 0.0:
            quat = -quat
        accumulator += np.outer(quat, quat)
        count += 1
    if count == 0:
        raise ValueError("no valid quaternions to average.")
    _, eigenvectors = np.linalg.eigh(accumulator / float(count))
    quat = eigenvectors[:, -1]
    if reference is not None and float(np.dot(reference, quat)) < 0.0:
        quat = -quat
    return quat / float(np.linalg.norm(quat))


def average_transforms(transforms: Iterable[object]) -> np.ndarray:
    matrices = [as_transform(transform) for transform in transforms]
    if not matrices:
        raise ValueError("no transforms to average.")
    translations = np.stack([matrix[:3, 3] for matrix in matrices], axis=0)
    quaternions = [matrix_to_quaternion_xyzw(matrix[:3, :3]) for matrix in matrices]
    return make_transform(
        quaternion_xyzw_to_matrix(average_quaternion_xyzw(quaternions)),
        np.mean(translations, axis=0),
    )


def rotation_angle_deg(first: object, second: object | None = None) -> float:
    first_rot = project_to_rotation(first)
    if second is not None:
        second_rot = project_to_rotation(second)
        delta = second_rot @ first_rot.T
    else:
        delta = first_rot
    cos_angle = (float(np.trace(delta)) - 1.0) * 0.5
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return float(math.degrees(math.acos(cos_angle)))


def transform_distance(first: object, second: object) -> tuple[float, float]:
    first_matrix = as_transform(first, name="first")
    second_matrix = as_transform(second, name="second")
    translation_m = float(np.linalg.norm(first_matrix[:3, 3] - second_matrix[:3, 3]))
    rotation_deg = rotation_angle_deg(first_matrix[:3, :3], second_matrix[:3, :3])
    return translation_m, rotation_deg


def transform_to_list(value: object) -> list[list[float]]:
    return as_transform(value).tolist()


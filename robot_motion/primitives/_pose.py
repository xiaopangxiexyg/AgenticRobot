"""Shared private rigid-pose validation for TCP motion primitives."""

from __future__ import annotations

import math

import numpy as np

from robot_motion.runtime.immutable import deep_freeze


_POSE_ATOL = 1e-9


def _rigid_pose(value: object, *, name: str) -> np.ndarray:
    try:
        pose = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric 4x4 pose.") from exc
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 pose.")
    if not np.allclose(
        pose[3],
        (0.0, 0.0, 0.0, 1.0),
        rtol=0.0,
        atol=_POSE_ATOL,
    ):
        raise ValueError(f"{name} must be homogeneous.")
    rotation = pose[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        rtol=0.0,
        atol=_POSE_ATOL,
    ):
        raise ValueError(f"{name} rotation must be orthonormal.")
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=_POSE_ATOL,
    ):
        raise ValueError(f"{name} rotation determinant must be +1.")
    return deep_freeze(np.ascontiguousarray(pose, dtype=np.float64))

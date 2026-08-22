"""Robot-world to Quest-world point alignment.

The solver is extracted from the Kabsch implementation in
``AR5_real/tools/estimate_workstation_y_rotation.py`` and strengthened with
input/rank checks suitable for interactive calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

try:  # Support both direct script-path imports and package imports.
    from .transforms import transform_points, validate_rigid_transform
except ImportError:  # pragma: no cover - direct import compatibility.
    from transforms import transform_points, validate_rigid_transform


@dataclass(frozen=True)
class AlignmentResult:
    quest_from_robot: np.ndarray
    residuals_m: np.ndarray
    rms_error_m: float
    max_error_m: float
    sample_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "quest_from_robot": self.quest_from_robot.tolist(),
            "residuals_m": self.residuals_m.tolist(),
            "rms_error_m": self.rms_error_m,
            "max_error_m": self.max_error_m,
            "sample_count": self.sample_count,
        }


def fit_quest_from_robot(
    robot_points: Sequence[Sequence[float]],
    quest_points: Sequence[Sequence[float]],
) -> AlignmentResult:
    """Fit ``p_quest ~= R @ p_robot + t`` without scale."""
    robot = np.asarray(robot_points, dtype=float)
    quest = np.asarray(quest_points, dtype=float)
    if robot.ndim != 2 or robot.shape[1:] != (3,):
        raise ValueError("robot_points must be an Nx3 array")
    if quest.shape != robot.shape:
        raise ValueError("quest_points must have the same Nx3 shape")
    if robot.shape[0] < 3:
        raise ValueError("at least three point pairs are required")
    if not np.all(np.isfinite(robot)) or not np.all(np.isfinite(quest)):
        raise ValueError("calibration points must be finite")

    robot_center = robot.mean(axis=0)
    quest_center = quest.mean(axis=0)
    robot_zero = robot - robot_center
    quest_zero = quest - quest_center
    if np.linalg.matrix_rank(robot_zero, tol=1e-8) < 2:
        raise ValueError("robot calibration points must not be collinear")
    if np.linalg.matrix_rank(quest_zero, tol=1e-8) < 2:
        raise ValueError("Quest calibration points must not be collinear")

    u, _, vt = np.linalg.svd(robot_zero.T @ quest_zero)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    translation = quest_center - rotation @ robot_center

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    transform = validate_rigid_transform(transform)
    predicted = transform_points(transform, robot)
    residuals = np.linalg.norm(predicted - quest, axis=1)
    return AlignmentResult(
        quest_from_robot=transform,
        residuals_m=residuals,
        rms_error_m=float(np.sqrt(np.mean(residuals**2))),
        max_error_m=float(np.max(residuals)),
        sample_count=int(robot.shape[0]),
    )


def alignment_acceptable(
    result: AlignmentResult,
    *,
    max_rms_error_m: float,
    max_point_error_m: float,
) -> bool:
    return bool(
        result.rms_error_m <= float(max_rms_error_m)
        and result.max_error_m <= float(max_point_error_m)
    )

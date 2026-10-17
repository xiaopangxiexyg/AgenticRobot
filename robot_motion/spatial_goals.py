"""Endpoint relation geometry shared by simulated and physical TCP targets.

This module computes proposals, not observations, IK, paths, or motion permits.
Tool-relative offsets use the input TCP orientation, before axis alignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.relative_action import pose_matrix_to_xyzquat, xyzquat_to_pose_matrix


def finite_vector(value: Sequence[float], size: int, *, name: str) -> np.ndarray:
    if (not isinstance(value, (tuple, list)) or len(value) != size
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"{name} must contain {size} finite numbers.")
    return np.asarray(value, dtype=float)


@dataclass(frozen=True)
class SpatialGoal:
    pose_xyz_quat_xyzw: tuple[float, ...]
    translation_distance_m: float
    rotation_distance_rad: float
    axis_alignment_error_rad: float | None


def compute_spatial_goal(
    *, current_tcp_pose_xyz_quat_xyzw: Sequence[float],
    anchor_xyz_m: Sequence[float], offset_xyz_m: Sequence[float],
    offset_frame: str, orientation_mode: str,
    tool_axis: str | None = None, axis_reference_xyz: Sequence[float] | None = None,
) -> SpatialGoal:
    """Offset an explicit anchor, optionally aligning one TCP axis with least turn.

    Alignment leaves roll unconstrained and chooses the rotation nearest the
    input orientation. Antiparallel vectors use the next input TCP axis as a
    deterministic half-turn axis; no joint or path feasibility is implied.
    """
    pose = finite_vector(current_tcp_pose_xyz_quat_xyzw, 7, name="current TCP pose")
    if abs(math.hypot(*pose[3:]) - 1.0) > 1e-5:
        raise ValueError("Current TCP quaternion must have unit norm.")
    anchor = finite_vector(anchor_xyz_m, 3, name="anchor_xyz_m")
    offset = finite_vector(offset_xyz_m, 3, name="offset_xyz_m")
    if offset_frame not in ("reference", "tool"):
        raise ValueError("offset_frame must be reference or tool.")
    if orientation_mode not in ("preserve", "align_axis"):
        raise ValueError("orientation_mode must be preserve or align_axis.")
    current = xyzquat_to_pose_matrix(pose)
    goal = current.copy()
    goal[:3, 3] = anchor + (offset if offset_frame == "reference" else current[:3, :3] @ offset)
    axis_error = None
    if orientation_mode == "preserve":
        if tool_axis is not None or axis_reference_xyz is not None:
            raise ValueError("Preserve mode cannot contain an axis constraint.")
    else:
        if tool_axis not in ("x", "y", "z"):
            raise ValueError("align_axis requires tool_axis x, y, or z.")
        direction = finite_vector(axis_reference_xyz, 3, name="axis_reference_xyz")
        norm = math.hypot(*direction)
        if not math.isfinite(norm) or norm < 1e-12:
            raise ValueError("Axis constraint must have a nonzero finite length.")
        direction /= norm
        index = ("x", "y", "z").index(tool_axis)
        source = current[:3, index]
        cross = np.cross(source, direction)
        sine = float(np.linalg.norm(cross))
        cosine = float(np.clip(source @ direction, -1.0, 1.0))
        if sine > 1e-12:
            turn = Rotation.from_rotvec(cross / sine * math.atan2(sine, cosine))
        elif cosine < 0:
            turn = Rotation.from_rotvec(current[:3, (index + 1) % 3] * math.pi)
        else:
            turn = Rotation.identity()
        goal[:3, :3] = turn.as_matrix() @ current[:3, :3]
        axis_error = float(math.atan2(
            np.linalg.norm(np.cross(goal[:3, index], direction)),
            np.clip(goal[:3, index] @ direction, -1.0, 1.0),
        ))
    if not np.isfinite(goal).all():
        raise ValueError("Computed goal is not finite.")
    encoded = pose_matrix_to_xyzquat(goal).astype(float)
    # Preserve metric coordinates without the shared encoder's float32 rounding.
    encoded[:3] = goal[:3, 3]
    distance = math.hypot(*(goal[:3, 3] - pose[:3]))
    if not math.isfinite(distance):
        raise ValueError("Computed translation distance is not finite.")
    return SpatialGoal(
        tuple(float(x) for x in encoded), distance,
        float(Rotation.from_matrix(goal[:3, :3] @ current[:3, :3].T).magnitude()),
        axis_error,
    )

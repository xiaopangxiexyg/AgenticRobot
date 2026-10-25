"""Endpoint relation geometry shared by simulated and physical TCP targets.

This module computes proposals, not observations, IK, paths, or motion permits.
Tool-relative offsets use the input TCP orientation, before axis alignment.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

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


@dataclass(frozen=True)
class ObjectAboveGoalCandidate:
    """One explicit height/yaw sample, with no feasibility or motion claim."""

    height_m: float
    tool_yaw_rad: float
    goal: SpatialGoal


@dataclass(frozen=True)
class RelationGoalCandidate:
    """One endpoint sample, not a transit-orientation or motion certificate."""

    offset_xyz_m: tuple[float, ...]
    tool_axis_rotation_rad: float
    goal: SpatialGoal


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


def generate_object_above_goals(
    *,
    current_tcp_pose_xyz_quat_xyzw: Sequence[float],
    anchor_xyz_m: Sequence[float],
    heights_m: Sequence[float],
    tool_yaw_offsets_rad: Sequence[float],
) -> tuple[ObjectAboveGoalCandidate, ...]:
    """Sample above an observed anchor with tool +Z pointing along reference -Z.

    Heights are measured along reference +Z, not from an inferred object centre.
    Yaw is about the aligned tool's own +Z axis. Periodic/duplicate samples are
    removed in input order. At most 64 input combinations are accepted; callers
    must explicitly choose the region and discretization. No IK is performed.
    """
    if (not isinstance(heights_m, (tuple, list)) or not heights_m
            or not isinstance(tool_yaw_offsets_rad, (tuple, list)) or not tool_yaw_offsets_rad
            or len(heights_m) * len(tool_yaw_offsets_rad) > 64):
        raise ValueError("Above-goal sampling needs 1..64 explicit height/yaw combinations.")
    heights = finite_vector(heights_m, len(heights_m), name="heights_m")
    yaws = finite_vector(tool_yaw_offsets_rad, len(tool_yaw_offsets_rad), name="tool_yaw_offsets_rad")
    if np.any(heights < 0):
        raise ValueError("Above-goal heights must be non-negative.")
    return tuple(ObjectAboveGoalCandidate(candidate.offset_xyz_m[2],
        candidate.tool_axis_rotation_rad, candidate.goal)
        for candidate in generate_relation_goals(
            current_tcp_pose_xyz_quat_xyzw=current_tcp_pose_xyz_quat_xyzw,
            anchor_xyz_m=anchor_xyz_m,
            offsets_xyz_m=[[0., 0., float(height)] for height in heights],
            offset_frame="reference", orientation_mode="align_axis", tool_axis="z",
            axis_reference_xyz=[0., 0., -1.], tool_axis_rotation_offsets_rad=yaws.tolist()))


def generate_relation_goals(
    *, current_tcp_pose_xyz_quat_xyzw: Sequence[float], anchor_xyz_m: Sequence[float],
    offsets_xyz_m: Sequence[Sequence[float]], offset_frame: str, orientation_mode: str,
    tool_axis: str | None, axis_reference_xyz: Sequence[float] | None,
    tool_axis_rotation_offsets_rad: Sequence[float],
) -> tuple[RelationGoalCandidate, ...]:
    """Sample explicit relation endpoints; offsets use the original input-tool axes.

    Alignment permits rotations about the chosen aligned tool axis, not only Z.
    Preserve mode accepts only zero/periodic-zero rotations. Input combinations
    are bounded before deduplication; no IK, collision or transit check occurs.
    """
    if (not isinstance(offsets_xyz_m, (list, tuple)) or not offsets_xyz_m
            or not isinstance(tool_axis_rotation_offsets_rad, (list, tuple))
            or not tool_axis_rotation_offsets_rad
            or len(offsets_xyz_m) * len(tool_axis_rotation_offsets_rad) > 64):
        raise ValueError("Relation sampling needs 1..64 explicit offset/rotation combinations.")
    offsets = [finite_vector(value, 3, name="offset_xyz_m") for value in offsets_xyz_m]
    turns = finite_vector(tool_axis_rotation_offsets_rad, len(tool_axis_rotation_offsets_rad),
                          name="tool_axis_rotation_offsets_rad")
    wrapped_turns = [float((float(turn) + math.pi) % (2 * math.pi) - math.pi) for turn in turns]
    if orientation_mode == "preserve" and any(abs(turn) > 1e-12 for turn in wrapped_turns):
        raise ValueError("Preserve mode cannot contain a nonzero tool-axis rotation.")
    current = xyzquat_to_pose_matrix(finite_vector(current_tcp_pose_xyz_quat_xyzw, 7, name="current TCP pose"))
    candidates, seen = [], set()
    for offset in offsets:
        base = compute_spatial_goal(
            current_tcp_pose_xyz_quat_xyzw=current_tcp_pose_xyz_quat_xyzw,
            anchor_xyz_m=anchor_xyz_m, offset_xyz_m=offset.tolist(), offset_frame=offset_frame,
            orientation_mode=orientation_mode, tool_axis=tool_axis, axis_reference_xyz=axis_reference_xyz)
        for turn in wrapped_turns:
            identity = (*offset, round(turn, 12))
            if identity in seen:
                continue
            seen.add(identity)
            if orientation_mode == "preserve":
                goal = base
            else:
                index = ("x", "y", "z").index(tool_axis)
                rotvec = np.zeros(3)
                rotvec[index] = turn
                pose = xyzquat_to_pose_matrix(base.pose_xyz_quat_xyzw)
                pose[:3, :3] = pose[:3, :3] @ Rotation.from_rotvec(rotvec).as_matrix()
                encoded = pose_matrix_to_xyzquat(pose).astype(float)
                encoded[:3] = pose[:3, 3]
                if np.dot(encoded[3:], current_tcp_pose_xyz_quat_xyzw[3:]) < 0:
                    encoded[3:] *= -1
                direction = np.asarray(axis_reference_xyz, dtype=float)
                direction /= math.hypot(*direction)
                goal = SpatialGoal(tuple(float(x) for x in encoded), base.translation_distance_m,
                    float(Rotation.from_matrix(pose[:3, :3] @ current[:3, :3].T).magnitude()),
                    float(math.acos(np.clip(pose[:3, index] @ direction, -1., 1.))))
            candidates.append(RelationGoalCandidate(tuple(float(x) for x in offset), turn, goal))
    return tuple(candidates)

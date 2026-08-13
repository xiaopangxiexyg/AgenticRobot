"""Pure demonstration editing and time-indexed nominal reference sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .contracts import (
    DemonstrationEdit,
    DemonstrationTrajectory,
    EditedDemonstrationTrajectory,
)


@dataclass(frozen=True)
class DemonstrationReference:
    """One interpolated nominal reference, before safety or execution."""

    requested_time_sec: float
    resolved_time_sec: float
    left_index: int
    right_index: int
    interpolation_alpha: float
    tool_pose_xyzquat_xyzw: np.ndarray
    hand_target: np.ndarray
    reference_frame: str
    controlled_frame: str
    hand_component_id: str
    hand_target_representation: str
    hand_target_names: tuple[str, ...]
    hand_target_units: tuple[str, ...]
    source_trajectory_hash: str
    edit_hash: str
    edit_sample_id: str
    edited_trajectory_hash: str

    def __post_init__(self) -> None:
        for name in (
            "requested_time_sec",
            "resolved_time_sec",
            "interpolation_alpha",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if not 0.0 <= self.interpolation_alpha <= 1.0:
            raise ValueError("interpolation_alpha must lie in [0, 1].")
        for name in ("left_index", "right_index"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)
        pose = np.asarray(self.tool_pose_xyzquat_xyzw, dtype=np.float64).reshape(-1).copy()
        hand = np.asarray(self.hand_target, dtype=np.float64).reshape(-1).copy()
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise ValueError("tool_pose_xyzquat_xyzw must contain seven finite values.")
        if not hand.size or not np.all(np.isfinite(hand)):
            raise ValueError("hand_target must contain finite values.")
        pose.setflags(write=False)
        hand.setflags(write=False)
        object.__setattr__(self, "tool_pose_xyzquat_xyzw", pose)
        object.__setattr__(self, "hand_target", hand)
        for name in (
            "reference_frame",
            "controlled_frame",
            "hand_component_id",
            "hand_target_representation",
            "edit_sample_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be non-empty.")
            object.__setattr__(self, name, value)
        for name in ("hand_target_names", "hand_target_units"):
            values = tuple(str(value).strip() for value in getattr(self, name))
            if len(values) != hand.shape[0] or any(not value for value in values):
                raise ValueError(f"{name} must match the hand target dimension.")
            object.__setattr__(self, name, values)
        for name in (
            "source_trajectory_hash",
            "edit_hash",
            "edited_trajectory_hash",
        ):
            value = str(getattr(self, name)).strip().lower()
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(
                    f"{name} must be a 64-character SHA-256 hex digest."
                )
            object.__setattr__(self, name, value)


def _hand_ramp_weights(trajectory: DemonstrationTrajectory) -> np.ndarray:
    weights = np.zeros(trajectory.step_count, dtype=np.float64)
    start = trajectory.hand_edit_start_index
    end = trajectory.hand_edit_end_index
    if start == end:
        weights[start:] = 1.0
        return weights
    weights[start : end + 1] = np.linspace(0.0, 1.0, end - start + 1)
    weights[end + 1 :] = 1.0
    return weights


def apply_demonstration_edit(
    trajectory: DemonstrationTrajectory,
    edit: DemonstrationEdit,
) -> EditedDemonstrationTrajectory:
    """Apply one high-level edit without invoking policy, safety, or physics."""

    if not isinstance(trajectory, DemonstrationTrajectory):
        raise TypeError("trajectory must be a DemonstrationTrajectory.")
    if not isinstance(edit, DemonstrationEdit):
        raise TypeError("edit must be a DemonstrationEdit.")
    if trajectory.hand_dim != edit.hand_dim:
        raise ValueError(
            "Demonstration edit hand dimension does not match the trajectory: "
            f"edit={edit.hand_dim}, trajectory={trajectory.hand_dim}."
        )

    # SciPy 1.13's Cython buffer adapter rejects read-only arrays.  Contracts
    # remain immutable; only the local numerical inputs are copied.
    delta_rotation = Rotation.from_rotvec(edit.rotation_rotvec_rad.copy())
    source_pose = trajectory.tool_pose_xyzquat_xyzw
    edited_position = (
        delta_rotation.apply(source_pose[:, :3]) + edit.translation_m[None, :]
    )
    edited_rotation = (
        delta_rotation * Rotation.from_quat(source_pose[:, 3:7].copy())
    ).as_quat()
    edited_pose = np.concatenate((edited_position, edited_rotation), axis=1)

    hand_weights = _hand_ramp_weights(trajectory)
    edited_hand = trajectory.hand_targets + (
        hand_weights[:, None] * edit.hand_final_delta[None, :]
    )

    return EditedDemonstrationTrajectory(
        trajectory_id=f"{trajectory.trajectory_id}:edit:{edit.edit_id}",
        reference_frame=trajectory.reference_frame,
        controlled_frame=trajectory.controlled_frame,
        hand_component_id=trajectory.hand_component_id,
        hand_target_representation=trajectory.hand_target_representation,
        hand_target_names=trajectory.hand_target_names,
        hand_target_units=trajectory.hand_target_units,
        timestamps_sec=trajectory.timestamps_sec,
        tool_pose_xyzquat_xyzw=edited_pose,
        hand_targets=edited_hand,
        hand_edit_start_index=trajectory.hand_edit_start_index,
        hand_edit_end_index=trajectory.hand_edit_end_index,
        source_trajectory_hash=trajectory.semantic_content_hash,
        edit_hash=edit.semantic_content_hash,
        edit_sample_id=edit.sample_id,
    )


def _interpolate_quaternion(
    left: np.ndarray,
    right: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha <= 0.0:
        return left.copy()
    if alpha >= 1.0:
        return right.copy()
    rotations = Rotation.from_quat(np.stack((left, right), axis=0))
    quaternion = Slerp([0.0, 1.0], rotations)([alpha]).as_quat()[0]
    if float(np.dot(quaternion, left)) < 0.0:
        quaternion *= -1.0
    return quaternion


def sample_edited_trajectory(
    trajectory: EditedDemonstrationTrajectory,
    time_sec: float,
) -> DemonstrationReference:
    """Sample a nominal reference by time, holding the two endpoints.

    The requested time is preserved for diagnostics.  ``resolved_time_sec`` is
    clamped to the demonstration interval so downstream scheduling never has
    to infer an implicit extrapolation policy.
    """

    if not isinstance(trajectory, EditedDemonstrationTrajectory):
        raise TypeError("trajectory must be an EditedDemonstrationTrajectory.")
    requested = float(time_sec)
    if not math.isfinite(requested):
        raise ValueError("time_sec must be finite.")
    resolved = float(np.clip(requested, 0.0, trajectory.duration_sec))
    timestamps = trajectory.timestamps_sec

    if resolved <= 0.0:
        left_index = right_index = 0
        alpha = 0.0
    elif resolved >= trajectory.duration_sec:
        left_index = right_index = trajectory.step_count - 1
        alpha = 0.0
    else:
        right_index = int(np.searchsorted(timestamps, resolved, side="right"))
        left_index = right_index - 1
        left_time = float(timestamps[left_index])
        right_time = float(timestamps[right_index])
        alpha = (resolved - left_time) / (right_time - left_time)

    if left_index == right_index:
        pose = trajectory.tool_pose_xyzquat_xyzw[left_index].copy()
        hand = trajectory.hand_targets[left_index].copy()
    else:
        left_pose = trajectory.tool_pose_xyzquat_xyzw[left_index]
        right_pose = trajectory.tool_pose_xyzquat_xyzw[right_index]
        position = (1.0 - alpha) * left_pose[:3] + alpha * right_pose[:3]
        quaternion = _interpolate_quaternion(
            left_pose[3:7],
            right_pose[3:7],
            alpha,
        )
        pose = np.concatenate((position, quaternion))
        hand = (
            (1.0 - alpha) * trajectory.hand_targets[left_index]
            + alpha * trajectory.hand_targets[right_index]
        )

    return DemonstrationReference(
        requested_time_sec=requested,
        resolved_time_sec=resolved,
        left_index=left_index,
        right_index=right_index,
        interpolation_alpha=float(alpha),
        tool_pose_xyzquat_xyzw=pose,
        hand_target=hand,
        reference_frame=trajectory.reference_frame,
        controlled_frame=trajectory.controlled_frame,
        hand_component_id=trajectory.hand_component_id,
        hand_target_representation=trajectory.hand_target_representation,
        hand_target_names=trajectory.hand_target_names,
        hand_target_units=trajectory.hand_target_units,
        source_trajectory_hash=trajectory.source_trajectory_hash,
        edit_hash=trajectory.edit_hash,
        edit_sample_id=trajectory.edit_sample_id,
        edited_trajectory_hash=trajectory.semantic_content_hash,
    )


__all__ = [
    "DemonstrationReference",
    "apply_demonstration_edit",
    "sample_edited_trajectory",
]

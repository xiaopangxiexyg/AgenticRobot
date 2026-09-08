"""Pure relative replay rebasing at a rule-based segment boundary.

The source trajectory remains immutable.  A source chunk's absolute pose
targets are converted into deltas from its recorded actual-TCP start, then
those deltas are applied to a runtime boundary state.  Dispatch and safety
remain outside this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from robot_motion.action import RobotAction
from robot_motion.action.schema import normalize_side
from robot_motion.relative_action import (
    absolute_pose,
    pose_matrix_to_xyzquat,
    relative_pose,
)

from .action_frame import replay_action_pose_frame, replay_robot_action_at_index
from .trajectory import ReplayTrajectoryGroup

RULE_BASED_RELATIVE_REPLAY_SCHEMA_V1 = (
    "dexteleop.rule_based_relative_replay.v1"
)


class RuleBasedRelativeReplayError(ValueError):
    """Raised when a boundary rebase cannot be proved safe and unambiguous."""


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class RuleBasedBoundaryState:
    """Measured TCP state that anchors a subsequent relative replay segment."""

    tcp_pose_by_side: Mapping[str, Any]
    pose_frame_by_side: Mapping[str, str]
    source: str = "runtime_actual_tcp"
    timestamp: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_poses = dict(self.tcp_pose_by_side)
        raw_frames = dict(self.pose_frame_by_side)
        if not raw_poses:
            raise ValueError("tcp_pose_by_side must contain at least one side.")
        poses: dict[str, np.ndarray] = {}
        frames: dict[str, str] = {}
        normalized_frames: dict[str, str] = {}
        for raw_side, value in raw_frames.items():
            side = normalize_side(str(raw_side))
            if side in normalized_frames:
                raise ValueError(f"Duplicate boundary frame side {side!r}.")
            normalized_frames[side] = str(value)
        for raw_side, value in raw_poses.items():
            side = normalize_side(str(raw_side))
            if side in poses:
                raise ValueError(f"Duplicate boundary side {side!r}.")
            try:
                pose = pose_matrix_to_xyzquat(value)
            except (TypeError, ValueError) as exc:
                raise RuleBasedRelativeReplayError(
                    f"Invalid boundary TCP pose for {side!r}: {exc}"
                ) from exc
            frame = _required_text(
                normalized_frames.get(side),
                name=f"pose_frame_by_side[{side}]",
            )
            poses[side] = pose.copy()
            frames[side] = frame
        if set(normalized_frames) - set(poses):
            unknown = sorted(set(normalized_frames) - set(poses))
            raise ValueError(f"pose_frame_by_side contains unknown side(s): {unknown}.")
        if self.timestamp is not None:
            timestamp = float(self.timestamp)
            if not np.isfinite(timestamp):
                raise ValueError("timestamp must be finite when provided.")
            object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "tcp_pose_by_side", poses)
        object.__setattr__(self, "pose_frame_by_side", frames)
        object.__setattr__(self, "source", _required_text(self.source, name="source"))
        object.__setattr__(self, "metadata", dict(self.metadata))


def _actual_tcp_at(
    group: ReplayTrajectoryGroup,
    *,
    side: str,
    index: int,
) -> np.ndarray:
    values = group.arm_observation_actual_tcp_pose.get(str(side))
    if values is None:
        raise RuleBasedRelativeReplayError(
            "Relative replay requires recorded actual TCP history for "
            f"{side!r}; none was found in group {group.key!r}."
        )
    array = np.asarray(values)
    if array.ndim not in {2, 3}:
        raise RuleBasedRelativeReplayError(
            f"Actual TCP history for {side!r} has invalid shape {array.shape}."
        )
    if array.ndim == 2 and array.shape[1] != 7:
        raise RuleBasedRelativeReplayError(
            f"Actual TCP history for {side!r} must have width 7, got {array.shape}."
        )
    if array.ndim == 3 and array.shape[1:] != (4, 4):
        raise RuleBasedRelativeReplayError(
            f"Actual TCP history for {side!r} must have shape [N, 4, 4], got {array.shape}."
        )
    if int(index) < 0 or int(index) >= array.shape[0]:
        raise RuleBasedRelativeReplayError(
            f"Actual TCP history for {side!r} does not contain frame {int(index)}."
        )
    return np.asarray(array[int(index)]).copy()


def rebase_replay_action_at_boundary(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int,
    boundary: RuleBasedBoundaryState,
    source_start_index: int = 0,
    include_end_effectors: bool = False,
    end_effector_sides: Sequence[str] | None = None,
    source: str = "rule_based_relative_replay",
) -> RobotAction:
    """Rebase one pose-replay action on a measured runtime boundary.

    For each side this computes ``inverse(source_actual_start) @ source_target``
    and applies it to ``boundary.tcp_pose_by_side``.  The returned action is an
    absolute canonical pose action in the boundary's existing frame, so the
    normal lower runtime can perform its usual admission, safety, IK, and
    dispatch steps.
    """

    if not isinstance(group, ReplayTrajectoryGroup):
        raise TypeError("group must be ReplayTrajectoryGroup.")
    if not isinstance(boundary, RuleBasedBoundaryState):
        raise TypeError("boundary must be RuleBasedBoundaryState.")
    sides = tuple(normalize_side(str(side)) for side in active_sides)
    if not sides or len(set(sides)) != len(sides):
        raise ValueError("active_sides must be non-empty and contain no duplicates.")
    start = int(source_start_index)
    frame_index = int(index)
    if start < 0 or frame_index < 0 or start > frame_index:
        raise IndexError(
            "source_start_index and index must satisfy "
            f"0 <= source_start_index <= index; got {start}, {frame_index}."
        )
    if frame_index >= int(group.frame_count):
        raise IndexError(
            f"Replay frame index {frame_index} is outside [0, {group.frame_count})."
        )
    if start >= int(group.frame_count):
        raise IndexError(
            f"Replay source_start_index {start} is outside [0, {group.frame_count})."
        )
    missing = sorted(set(sides) - set(boundary.tcp_pose_by_side))
    if missing:
        raise RuleBasedRelativeReplayError(
            f"Boundary is missing active side(s): {missing}."
        )

    hand_sides = tuple(sides if end_effector_sides is None else end_effector_sides)
    if any(side not in {"left", "right"} for side in hand_sides):
        raise ValueError("end_effector_sides must contain only left/right sides.")
    source_action = replay_robot_action_at_index(
        group,
        sides,
        index=frame_index,
        include_end_effectors=bool(include_end_effectors),
        end_effector_sides=hand_sides,
        source=source,
        timestamp=group.relative_time_sec(frame_index),
    )
    source_pose_frame = replay_action_pose_frame(group)
    rebased_by_side: dict[str, np.ndarray] = {}
    for side in sides:
        effector = f"{side}_arm"
        action_frame = str(source_action.frame_by_effector.get(effector) or "")
        boundary_frame = boundary.pose_frame_by_side[side]
        if action_frame != boundary_frame:
            raise RuleBasedRelativeReplayError(
                f"Frame mismatch for {side!r}: source action uses {action_frame!r}, "
                f"boundary uses {boundary_frame!r}; source pose marker={source_pose_frame!r}."
            )
        source_anchor = _actual_tcp_at(group, side=side, index=start)
        delta = relative_pose(source_anchor, source_action.arm_pose_by_side[side])
        rebased_matrix = absolute_pose(boundary.tcp_pose_by_side[side], delta)
        rebased_by_side[side] = pose_matrix_to_xyzquat(rebased_matrix)

    metadata = {
        **dict(source_action.metadata),
        "rule_based_relative_replay_schema": RULE_BASED_RELATIVE_REPLAY_SCHEMA_V1,
        "relative_replay_source_group": str(group.key),
        "relative_replay_source_start_index": int(start),
        "relative_replay_frame_index": int(frame_index),
        "relative_replay_anchor": "runtime_boundary_actual_tcp",
        "relative_replay_boundary_source": boundary.source,
        "relative_replay_boundary_timestamp": boundary.timestamp,
        "relative_replay_boundary_metadata": dict(boundary.metadata),
    }
    return RobotAction(
        action_space=source_action.action_space,
        timestamp=source_action.timestamp,
        arm_pose_by_side=rebased_by_side,
        hand_qpos_by_side=source_action.hand_qpos_by_side,
        active_effectors=source_action.active_effectors,
        frame_by_effector=source_action.frame_by_effector,
        controlled_frame_by_effector=source_action.controlled_frame_by_effector,
        source=str(source),
        metadata=metadata,
    )


__all__ = [
    "RULE_BASED_RELATIVE_REPLAY_SCHEMA_V1",
    "RuleBasedBoundaryState",
    "RuleBasedRelativeReplayError",
    "rebase_replay_action_at_boundary",
]

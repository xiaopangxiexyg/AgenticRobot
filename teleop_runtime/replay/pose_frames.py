"""Pose-frame bridge between canonical v2.3 replay and RM65 hardware.

V2.3 dataset loading deliberately leaves ``ReplayTrajectoryGroup.actions`` in
the dataset-wide canonical desk frame.  RM65 controllers and the replay
collision planner, however, consume a tool-TCP pose in each arm's controller
base frame.  This module owns that final, explicit frame boundary.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.pose_frames import PoseFrameContract, PoseFrameContractError
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES
from robot_motion.pose_frame_profiles import default_rm65_workspace_near_boundary_runtime_pose_frame_contract
from robot_motion.replay.action_frame import (
    CANONICAL_DESK_ACTION_FRAME,
    HARDWARE_ARM_BASE_ACTION_FRAME,
    replay_action_pose_frame,
)
from robot_motion.replay.trajectory import ReplayTrajectoryGroup, replay_action_poses_at_index


@lru_cache(maxsize=1)
def _runtime_contract() -> PoseFrameContract:
    return default_rm65_workspace_near_boundary_runtime_pose_frame_contract()


def _action_metadata(group: ReplayTrajectoryGroup) -> Mapping[str, Any]:
    value = group.metadata.get("action_metadata")
    return value if isinstance(value, Mapping) else {}


def replay_action_poses_in_hardware_base_at_index(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int,
) -> dict[str, np.ndarray]:
    """Return tool-TCP targets in the controller base frame for each side."""

    poses = replay_action_poses_at_index(group, active_sides, index=int(index))
    if replay_action_pose_frame(group) == HARDWARE_ARM_BASE_ACTION_FRAME:
        return poses
    source_contract = PoseFrameContract.from_dict(
        _action_metadata(group)["pose_frame_contract"]
    )
    runtime = _runtime_contract()
    converted: dict[str, np.ndarray] = {}
    for side, pose in poses.items():
        if side not in source_contract.source.active_sides:
            raise RuntimeError(
                f"Replay pose_frame_contract does not declare active side {side!r}."
            )
        if side not in runtime.source.active_sides:
            raise RuntimeError(
                f"RM65 runtime desk contract does not declare replay side {side!r}."
            )
        try:
            converted[side] = runtime.canonical_desk_pose_vector_to_storage(
                pose, side=side
            )
        except (PoseFrameContractError, ValueError) as exc:
            raise RuntimeError(
                f"{side} canonical-desk replay pose cannot be converted to arm base: {exc}"
            ) from exc
    return converted


def replay_group_in_hardware_base(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
) -> ReplayTrajectoryGroup:
    """Build a detached arm-base view for command resolvers that take a group."""

    if replay_action_pose_frame(group) == HARDWARE_ARM_BASE_ACTION_FRAME:
        return group
    actions = np.asarray(group.actions, dtype=np.float32).copy()
    for index in range(int(group.frame_count)):
        poses = replay_action_poses_in_hardware_base_at_index(
            group, active_sides, index=index
        )
        for side, pose in poses.items():
            actions[index, ACTION_ARM_POSE_SLICES[side]] = pose
    metadata = dict(group.metadata)
    action_metadata = dict(_action_metadata(group))
    action_metadata["pose_action_absolute_frame"] = HARDWARE_ARM_BASE_ACTION_FRAME
    action_metadata["pose_frame_bridge"] = "canonical_desk_to_rm65_controller_base"
    metadata["action_metadata"] = action_metadata
    source_metadata = dict(metadata.get("source_metadata") or {})
    episode_metadata = dict(source_metadata.get("episode_metadata") or {})
    episode_metadata["arm_reference_frame_by_side"] = {
        str(side): f"{side}_arm_base" for side in active_sides
    }
    episode_metadata["arm_controlled_frame_by_side"] = {
        str(side): "tool_tcp" for side in active_sides
    }
    episode_metadata["arm_action_specs_by_side"] = {
        str(side): {
            "side": str(side),
            "controlled_frame": "tool_tcp",
            "reference_frame": f"{side}_arm_base",
            "representation": "xyz_quat",
            "dof": 7,
            "semantics": "tool_tcp_pose",
            "metadata": {
                "pose_frame_bridge": "canonical_desk_to_rm65_controller_base"
            },
        }
        for side in active_sides
    }
    source_metadata["episode_metadata"] = episode_metadata
    metadata["source_metadata"] = source_metadata
    return replace(group, actions=actions, metadata=metadata)


__all__ = [
    "CANONICAL_DESK_ACTION_FRAME",
    "HARDWARE_ARM_BASE_ACTION_FRAME",
    "replay_action_pose_frame",
    "replay_action_poses_in_hardware_base_at_index",
    "replay_group_in_hardware_base",
]

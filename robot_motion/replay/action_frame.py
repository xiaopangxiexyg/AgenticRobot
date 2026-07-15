from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np

from robot_motion.action.hand_spec import hand_action_specs_from_metadata
from robot_motion.action.executor import RobotActionResolver
from robot_motion.action.schema import RobotAction
from robot_motion.deploy.schema import ACTION_HAND_SLICES, HAND_DOF
from robot_motion.runtime import TeleopCommand

from .trajectory import ReplayTrajectoryGroup, replay_action_poses_at_index


COMMON_REPLAY_MIGRATION_NOTE = (
    "This module is hardware-free. It is the shared replay action-construction "
    "surface used by sim and real dispatchers."
)


def _source_metadata(group: ReplayTrajectoryGroup) -> Mapping[str, Any]:
    source = group.metadata.get("source_metadata")
    return source if isinstance(source, Mapping) else {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _explicit_hand_replay_target(
    group: ReplayTrajectoryGroup,
    *,
    side: str,
    index: int,
) -> np.ndarray | None:
    values = group.hand_joint_targets.get(str(side))
    if values is None:
        return None
    try:
        targets = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if (
        targets.ndim != 2
        or targets.shape[1] != HAND_DOF
        or targets.shape[0] <= int(index)
    ):
        return None
    target = np.asarray(targets[int(index)], dtype=np.float32).reshape(-1)
    if target.shape != (HAND_DOF,) or not np.all(np.isfinite(target)):
        return None
    return target.astype(np.float32, copy=True)


def replay_action_metadata(
    group: ReplayTrajectoryGroup,
    *,
    active_sides: Sequence[str],
) -> dict[str, Any]:
    sides = {str(side) for side in active_sides}
    source = _source_metadata(group)
    dataset_info = _mapping(source.get("dataset_info"))
    episode_metadata = _mapping(source.get("episode_metadata"))
    merged: dict[str, Any] = {}
    for key in (
        "arm_action_specs_by_side",
        "arm_controlled_frame_by_side",
        "arm_reference_frame_by_side",
        "hand_action_specs_by_side",
        "hand_active_mask_by_side",
    ):
        values: dict[str, Any] = {}
        for parent in (dataset_info, episode_metadata):
            payload = parent.get(key)
            if isinstance(payload, Mapping):
                values.update({str(side): item for side, item in payload.items() if str(side) in sides})
        if values:
            merged[key] = values
    merged.setdefault("replay_common_migration_note", COMMON_REPLAY_MIGRATION_NOTE)
    return merged


def replay_end_effector_targets_at_index(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int,
) -> dict[str, np.ndarray]:
    actions = np.asarray(group.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] <= int(index):
        raise RuntimeError(f"Replay action trajectory has invalid shape {actions.shape}.")
    targets: dict[str, np.ndarray] = {}
    metadata = replay_action_metadata(group, active_sides=active_sides)
    hand_specs = hand_action_specs_from_metadata(metadata)
    for side in active_sides:
        side_text = str(side)
        target_slice = ACTION_HAND_SLICES[side_text]
        if actions.shape[1] < int(target_slice.stop or 0):
            raise RuntimeError(
                f"{side} replay end-effector target requires action width >= {int(target_slice.stop or 0)}, "
                f"got {actions.shape[1]}."
            )
        target = np.asarray(actions[int(index), target_slice], dtype=np.float32).reshape(-1)
        if target.shape != (HAND_DOF,):
            raise RuntimeError(f"{side} replay end-effector target at frame {int(index)} has invalid shape.")
        explicit_replay_target = _explicit_hand_replay_target(
            group,
            side=side_text,
            index=int(index),
        )
        spec = hand_specs.get(side_text)
        if spec is not None:
            if int(spec.slot_count) != HAND_DOF:
                raise RuntimeError(
                    f"{side} replay end-effector spec slot_count={int(spec.slot_count)} "
                    f"does not match dataset hand slot width {HAND_DOF}."
                )
            active_mask = np.asarray(spec.active_mask, dtype=bool)
            if active_mask.shape != (HAND_DOF,):
                raise RuntimeError(f"{side} replay end-effector active mask has invalid shape.")
            active = target[active_mask]
            if active.shape != (int(spec.active_dof),) or not np.all(np.isfinite(active)):
                if explicit_replay_target is None:
                    raise RuntimeError(
                        f"{side} replay end-effector active target at frame {int(index)} is invalid."
                    )
                normalized = explicit_replay_target
            else:
                normalized = target.astype(np.float32, copy=True)
            normalized[~active_mask & ~np.isfinite(normalized)] = 0.0
            if not np.all(np.isfinite(normalized)):
                raise RuntimeError(
                    f"{side} replay end-effector target at frame {int(index)} contains infinite values."
                )
            targets[side_text] = normalized
            continue
        if not np.all(np.isfinite(target)):
            if explicit_replay_target is None:
                raise RuntimeError(
                    f"{side} replay end-effector target at frame {int(index)} is invalid; "
                    "partial/NaN hand slots require hand_action_specs_by_side metadata "
                    "or a finite action.replay hand target."
                )
            target = explicit_replay_target
        targets[side_text] = target.astype(np.float32, copy=True)
    return targets


def replay_robot_action_at_index(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int,
    include_arms: bool = True,
    include_end_effectors: bool = False,
    source: str = "replay_action",
) -> RobotAction:
    arm_pose_by_side = (
        replay_action_poses_at_index(group, active_sides, index=int(index))
        if bool(include_arms)
        else {}
    )
    hand_qpos_by_side = (
        replay_end_effector_targets_at_index(group, active_sides, index=int(index))
        if bool(include_end_effectors)
        else {}
    )
    if not arm_pose_by_side and not hand_qpos_by_side:
        raise RuntimeError("Replay frame produced no active arm or end-effector action.")
    return RobotAction.bimanual_ee_pose(
        left_arm=arm_pose_by_side.get("left"),
        right_arm=arm_pose_by_side.get("right"),
        left_hand=hand_qpos_by_side.get("left"),
        right_hand=hand_qpos_by_side.get("right"),
        timestamp=time.time(),
        source=str(source),
        metadata={
            **replay_action_metadata(group, active_sides=active_sides),
            "replay_source": str(source),
            "replay_frame_index": int(index),
        },
    )


def annotate_replay_command_metadata(command: TeleopCommand, *, source: str) -> TeleopCommand:
    command.metadata["source"] = str(source)
    for effector_command in command.commands.values():
        effector_command.metadata["source"] = str(source)
        effector_command.metadata["replay_source"] = str(source)
        if effector_command.command_type == "cartesian_pose":
            pose = np.asarray(effector_command.value, dtype=np.float32).reshape(4, 4)
            controlled_frame = str(
                effector_command.metadata.get("arm_controlled_frame")
                or effector_command.metadata.get("target_pose_controlled_frame")
                or "tool_tcp"
            )
            effector_command.metadata.setdefault("source_target_pose_base", pose.copy())
            effector_command.metadata.setdefault("target_pose_tcp_base", pose.copy())
            effector_command.metadata.setdefault("target_pose_retarget_frame", controlled_frame)
    return command


def replay_teleop_command_at_index(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int,
    include_arms: bool = True,
    include_end_effectors: bool = False,
    source: str = "replay_action",
    resolver: RobotActionResolver | None = None,
) -> TeleopCommand:
    resolved = resolver if resolver is not None else RobotActionResolver()
    action = replay_robot_action_at_index(
        group,
        active_sides,
        index=int(index),
        include_arms=bool(include_arms),
        include_end_effectors=bool(include_end_effectors),
        source=str(source),
    )
    return annotate_replay_command_metadata(resolved.resolve(action), source=str(source))


__all__ = [
    "COMMON_REPLAY_MIGRATION_NOTE",
    "annotate_replay_command_metadata",
    "replay_action_metadata",
    "replay_end_effector_targets_at_index",
    "replay_robot_action_at_index",
    "replay_teleop_command_at_index",
]

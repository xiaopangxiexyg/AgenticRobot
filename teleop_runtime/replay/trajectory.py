"""Legacy replay trajectory entrypoint.

Pure replay trajectory contracts live in robot_motion.replay.trajectory. This
module keeps teleop-specific dataset/parquet loading wrappers plus compatibility
re-exports for old teleop_runtime.replay.trajectory imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.replay.trajectory import (
    QPOS_INIT_AUTO,
    QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
    QPOS_INIT_OBSERVATION_CURRENT_QPOS,
    QPOS_INIT_REPLAY_QPOS,
    QPOS_INIT_SOURCES,
    REPLAY_TRAJECTORY_ACTION_POSE,
    REPLAY_TRAJECTORY_QPOS_TARGET,
    REPLAY_TRAJECTORY_SOURCES,
    ReplayGroupSpec,
    ReplayInitEpisodeSpec,
    ReplayTrajectoryGroup,
    ReplayTrajectoryPool,
    normalize_qpos_init_source,
    normalize_replay_trajectory_source,
    parse_episode_range_token,
    parse_replay_group_key,
    parse_replay_group_spec,
    parse_replay_index_token,
    parse_replay_init_episode_spec,
    replay_action_poses_at_index,
    replay_active_arm_poses_are_finite,
    replay_qpos_init_candidates_at_index,
    select_replay_qpos_init_targets,
)

from ..deploy.replay import (
    infer_replay_action_hz,
    load_replay_actions_from_parquet,
    load_replay_arm_joint_targets_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    load_replay_frame_times_from_parquet,
    load_replay_hand_joint_targets_from_parquet,
    resolve_replay_parquet_path,
)


@dataclass(frozen=True)
class ReplayInitEpisodePlan:
    group: ReplayTrajectoryGroup
    move_time_s: float
    to_record_time_s: float


def read_replay_dataset_episode_indices(dataset_root: str | Path) -> list[int]:
    root = Path(dataset_root).expanduser().resolve()
    episodes_path = root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(
            f"Balanced replay requires dataset metadata: {episodes_path}"
        )
    indices: list[int] = []
    for line in episodes_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        row = json.loads(text)
        try:
            indices.append(int(row["episode_index"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid replay episode row in {episodes_path}: {row!r}") from exc
    if not indices:
        raise ValueError(f"Replay dataset has no episodes in {episodes_path}.")
    return sorted(set(indices))


def resolve_replay_episode_indices(args: Any) -> list[int]:
    dataset = str(args.replay_dataset).strip()
    if not dataset:
        return [int(args.replay_episode)]
    if not bool(getattr(args, "replay_all_episodes", True)):
        return [int(args.replay_episode)]
    return read_replay_dataset_episode_indices(dataset)


def build_replay_group_for_episode(
    args: Any,
    *,
    episode_index: int,
    group_spec: ReplayGroupSpec,
    balanced_episode_count: int,
    require_arm_joint_targets: bool = False,
) -> ReplayTrajectoryGroup:
    source_path, resolved_metadata = resolve_replay_parquet_path(
        dataset_root=str(args.replay_dataset).strip() or None,
        parquet_path=str(args.replay_parquet).strip() or None,
        episode_index=int(episode_index),
    )
    all_actions = load_replay_actions_from_parquet(
        source_path,
        action_column=str(args.replay_action_column or "action"),
        start_index=0,
        end_index=None,
    )
    total_frames = int(all_actions.shape[0])
    start_index = int(group_spec.start_index)
    end_index = int(group_spec.end_index)
    if end_index < 0:
        end_index = total_frames
    if start_index < 0 or start_index >= total_frames:
        raise ValueError(
            f"replay group '{group_spec.key}' start_index={start_index} "
            f"is outside source episode {episode_index} frame range [0, {total_frames})."
        )
    if end_index <= start_index or end_index > total_frames:
        raise ValueError(
            f"replay group '{group_spec.key}' end_index={end_index} must satisfy "
            f"start < end <= {total_frames} for source episode {episode_index}."
        )
    action_hz_info = infer_replay_action_hz(
        parquet_path=source_path,
        dataset_root=str(args.replay_dataset).strip() or None,
        episode_index=int(episode_index),
        start_index=start_index,
        end_index=end_index,
        resolved_metadata=resolved_metadata,
    )
    frame_times_sec, frame_timing_info = load_replay_frame_times_from_parquet(
        source_path,
        start_index=start_index,
        end_index=end_index,
        fallback_hz=float(action_hz_info["hz"]),
    )
    actions = all_actions[start_index:end_index].astype(np.float32, copy=True)
    arm_joint_targets = load_replay_arm_joint_targets_from_parquet(
        source_path,
        active_sides=args.active_sides,
        start_index=start_index,
        end_index=end_index,
        required=bool(require_arm_joint_targets),
    )
    hand_joint_targets = load_replay_hand_joint_targets_from_parquet(
        source_path,
        active_sides=args.active_sides,
        start_index=start_index,
        end_index=end_index,
    )
    arm_observation_current_qpos = load_replay_arm_observation_qpos_from_parquet(
        source_path,
        active_sides=args.active_sides,
        field_name="current_qpos",
        start_index=start_index,
        end_index=end_index,
    )
    arm_observation_actual_qpos = load_replay_arm_observation_qpos_from_parquet(
        source_path,
        active_sides=args.active_sides,
        field_name="actual_hardware_qpos",
        start_index=start_index,
        end_index=end_index,
    )
    source_episode_index = int(resolved_metadata.get("episode_index", int(episode_index)))
    return ReplayTrajectoryGroup(
        key=group_spec.key,
        label=f"episodes_{group_spec.episode_range}",
        source_path=str(source_path),
        source_episode_index=source_episode_index,
        start_index=int(start_index),
        end_index=int(end_index),
        frame_count=int(actions.shape[0]),
        action_hz=float(action_hz_info["hz"]),
        actions=actions,
        frame_times_sec=frame_times_sec,
        arm_joint_targets=arm_joint_targets,
        hand_joint_targets=hand_joint_targets,
        arm_observation_current_qpos=arm_observation_current_qpos,
        arm_observation_actual_qpos=arm_observation_actual_qpos,
        metadata={
            "action_hz_estimate": dict(action_hz_info),
            "frame_timing": dict(frame_timing_info),
            "source_metadata": dict(resolved_metadata),
            "action_column": str(args.replay_action_column or "action"),
            "balanced_replay": bool(str(args.replay_dataset).strip())
            and bool(getattr(args, "replay_all_episodes", True)),
            "balanced_episode_count": int(balanced_episode_count),
            "source_episode_index": source_episode_index,
            "episode_range_spec": str(group_spec.episode_range),
        },
    )


def build_replay_groups_from_args(args: Any) -> dict[str, ReplayTrajectoryPool]:
    if not (str(args.replay_dataset).strip() or str(args.replay_parquet).strip()):
        return {}
    group_specs = list(args.replay_group or [])
    if not group_specs:
        group_specs = ["a:all"]
    available_episode_indices = resolve_replay_episode_indices(args)
    pools: dict[str, ReplayTrajectoryPool] = {}
    for spec in group_specs:
        parsed_spec = parse_replay_group_spec(
            spec,
            available_episode_indices=available_episode_indices,
        )
        if str(args.replay_parquet).strip() and parsed_spec.episode_range not in {"all", "*"}:
            raise ValueError(
                "With --replay-parquet, --replay-group only supports 'key:all' because "
                "there is only one source parquet. Use --replay-dataset to select episode index ranges."
            )
        selected_episode_indices = (
            tuple(available_episode_indices)
            if str(args.replay_parquet).strip()
            else parse_episode_range_token(
                parsed_spec.episode_range,
                available_indices=available_episode_indices,
            )
        )
        candidates = [
            build_replay_group_for_episode(
                args,
                episode_index=episode_index,
                group_spec=parsed_spec,
                balanced_episode_count=len(selected_episode_indices),
            )
            for episode_index in selected_episode_indices
        ]
        key = candidates[0].key
        label = candidates[0].label
        if key in pools:
            raise ValueError(f"Duplicate replay group key '{key}'.")
        if any(candidate.key != key for candidate in candidates):
            raise ValueError(f"Replay group spec {spec!r} produced inconsistent keys.")
        pools[key] = ReplayTrajectoryPool(
            key=key,
            label=label,
            groups=candidates,
            episode_indices=tuple(int(index) for index in selected_episode_indices),
            episode_range_spec=str(parsed_spec.episode_range),
        )
    return pools


def build_replay_init_plan_from_args(args: Any) -> ReplayInitEpisodePlan | None:
    spec_text = str(getattr(args, "replay_init_episode", "") or "").strip()
    if not spec_text:
        return None
    init_dataset_root = str(getattr(args, "replay_init_dataset", "") or "").strip()
    init_parquet_path = str(getattr(args, "replay_init_parquet", "") or "").strip()
    source_dataset_root = init_dataset_root or str(args.replay_dataset).strip()
    source_parquet_path = init_parquet_path or str(args.replay_parquet).strip()
    episode_token = spec_text.split(":", 1)[0].strip()
    if not episode_token:
        raise ValueError("replay init episode spec must start with an episode index.")
    episode_index = int(episode_token)
    init_episode_path, _ = resolve_replay_parquet_path(
        dataset_root=source_dataset_root or None,
        parquet_path=source_parquet_path or None,
        episode_index=episode_index,
    )
    total_frames = int(
        load_replay_actions_from_parquet(
            init_episode_path,
            action_column=str(args.replay_action_column or "action"),
            start_index=0,
            end_index=None,
        ).shape[0]
    )
    parsed = parse_replay_init_episode_spec(
        spec_text,
        default_total_frames=total_frames,
    )
    group_spec = ReplayGroupSpec(
        key="h",
        start_index=int(parsed.start_index),
        end_index=int(parsed.end_index),
        episode_range=str(parsed.episode_index),
    )
    group_args = _namespace_copy(args)
    group_args.replay_dataset = source_dataset_root
    group_args.replay_parquet = source_parquet_path
    group = build_replay_group_for_episode(
        group_args,
        episode_index=int(parsed.episode_index),
        group_spec=group_spec,
        balanced_episode_count=1,
        require_arm_joint_targets=True,
    )
    return ReplayInitEpisodePlan(
        group=group,
        move_time_s=float(args.replay_init_episode_move_time),
        to_record_time_s=float(args.replay_init_episode_to_record_time),
    )


def _namespace_copy(args: Any) -> Any:
    try:
        import argparse

        return argparse.Namespace(**vars(args))
    except TypeError:
        class _Args:
            pass

        copied = _Args()
        for key in dir(args):
            if key.startswith("_"):
                continue
            try:
                setattr(copied, key, getattr(args, key))
            except Exception:
                pass
        return copied


__all__ = [
    "QPOS_INIT_AUTO",
    "QPOS_INIT_OBSERVATION_ACTUAL_QPOS",
    "QPOS_INIT_OBSERVATION_CURRENT_QPOS",
    "QPOS_INIT_REPLAY_QPOS",
    "QPOS_INIT_SOURCES",
    "REPLAY_TRAJECTORY_ACTION_POSE",
    "REPLAY_TRAJECTORY_QPOS_TARGET",
    "REPLAY_TRAJECTORY_SOURCES",
    "ReplayGroupSpec",
    "ReplayInitEpisodePlan",
    "ReplayInitEpisodeSpec",
    "ReplayTrajectoryGroup",
    "ReplayTrajectoryPool",
    "build_replay_group_for_episode",
    "build_replay_groups_from_args",
    "build_replay_init_plan_from_args",
    "normalize_qpos_init_source",
    "normalize_replay_trajectory_source",
    "parse_episode_range_token",
    "parse_replay_group_key",
    "parse_replay_group_spec",
    "parse_replay_index_token",
    "parse_replay_init_episode_spec",
    "read_replay_dataset_episode_indices",
    "replay_action_poses_at_index",
    "replay_active_arm_poses_are_finite",
    "replay_qpos_init_candidates_at_index",
    "resolve_replay_episode_indices",
    "select_replay_qpos_init_targets",
]

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..deploy.replay import (
    infer_replay_action_hz,
    load_replay_actions_from_parquet,
    load_replay_arm_joint_targets_from_parquet,
    load_replay_arm_observation_qpos_from_parquet,
    resolve_replay_parquet_path,
)
from ..deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ARM_JOINT_DOF


REPLAY_TRAJECTORY_ACTION_POSE = "action_pose"
REPLAY_TRAJECTORY_QPOS_TARGET = "qpos_target"
REPLAY_TRAJECTORY_SOURCES = (
    REPLAY_TRAJECTORY_ACTION_POSE,
    REPLAY_TRAJECTORY_QPOS_TARGET,
)

QPOS_INIT_AUTO = "auto"
QPOS_INIT_REPLAY_QPOS = "replay_qpos"
QPOS_INIT_OBSERVATION_CURRENT_QPOS = "observation_current_qpos"
QPOS_INIT_OBSERVATION_ACTUAL_QPOS = "observation_actual_qpos"
QPOS_INIT_SOURCES = (
    QPOS_INIT_AUTO,
    QPOS_INIT_REPLAY_QPOS,
    QPOS_INIT_OBSERVATION_CURRENT_QPOS,
    QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
)


def normalize_replay_trajectory_source(value: str | None) -> str:
    source = str(value or REPLAY_TRAJECTORY_ACTION_POSE).strip().lower()
    aliases = {
        "action": REPLAY_TRAJECTORY_ACTION_POSE,
        "pose": REPLAY_TRAJECTORY_ACTION_POSE,
        "action-pose": REPLAY_TRAJECTORY_ACTION_POSE,
        "cartesian_pose": REPLAY_TRAJECTORY_ACTION_POSE,
        "cartesian-pose": REPLAY_TRAJECTORY_ACTION_POSE,
        "qpos": REPLAY_TRAJECTORY_QPOS_TARGET,
        "joint": REPLAY_TRAJECTORY_QPOS_TARGET,
        "joint_vector": REPLAY_TRAJECTORY_QPOS_TARGET,
        "joint-vector": REPLAY_TRAJECTORY_QPOS_TARGET,
        "qpos-target": REPLAY_TRAJECTORY_QPOS_TARGET,
    }
    source = aliases.get(source, source)
    if source not in REPLAY_TRAJECTORY_SOURCES:
        raise ValueError(
            "trajectory_source must be one of "
            f"{REPLAY_TRAJECTORY_SOURCES}, got {value!r}."
        )
    return source


def normalize_qpos_init_source(value: str | None) -> str:
    source = str(value or QPOS_INIT_AUTO).strip().lower()
    aliases = {
        "replay": QPOS_INIT_REPLAY_QPOS,
        "qpos": QPOS_INIT_REPLAY_QPOS,
        "joint": QPOS_INIT_REPLAY_QPOS,
        "current": QPOS_INIT_OBSERVATION_CURRENT_QPOS,
        "current_qpos": QPOS_INIT_OBSERVATION_CURRENT_QPOS,
        "observation_current": QPOS_INIT_OBSERVATION_CURRENT_QPOS,
        "actual": QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
        "actual_qpos": QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
        "hardware_qpos": QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
        "actual_hardware_qpos": QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
    }
    source = aliases.get(source, source)
    if source not in QPOS_INIT_SOURCES:
        raise ValueError(
            f"qpos_init_source must be one of {QPOS_INIT_SOURCES}, got {value!r}."
        )
    return source


def parse_replay_group_key(text: str) -> str:
    key = str(text).strip().lower()
    if len(key) != 1:
        raise ValueError("replay group key must be a single character.")
    if key in {"p", "z", "c", "h", "q", "x", "r"}:
        raise ValueError(
            f"replay group key '{key}' conflicts with an existing control key."
        )
    return key


def parse_replay_index_token(
    text: str,
    *,
    total_frames: int,
    default: int,
) -> int:
    token = str(text).strip().lower()
    if not token:
        return int(default)
    if token == "start":
        return 0
    if token in {"end", "last"}:
        return int(total_frames)
    if token in {"mid", "half"}:
        return int(total_frames // 2)
    value = int(token)
    if value < 0:
        value = int(total_frames) + value
    return int(value)


def parse_episode_range_token(text: str, *, available_indices: Sequence[int]) -> tuple[int, ...]:
    token = str(text).strip().lower()
    available = tuple(sorted({int(index) for index in available_indices}))
    if not available:
        raise ValueError("No replay source episodes are available.")
    if not token or token in {"all", "*"}:
        return available
    separators = ("-", "..")
    for separator in separators:
        if separator in token:
            left, right = [part.strip() for part in token.split(separator, 1)]
            if not left or not right:
                raise ValueError(f"Invalid replay episode range: {text!r}")
            start = int(left)
            end = int(right)
            if end < start:
                raise ValueError(f"Replay episode range end must be >= start: {text!r}")
            selected = tuple(index for index in available if start <= index <= end)
            if not selected:
                raise ValueError(
                    f"Replay episode range {text!r} selected no available episodes; "
                    f"available=[{available[0]}..{available[-1]}]."
                )
            return selected
    selected_values = tuple(
        int(part.strip())
        for part in token.split(",")
        if part.strip()
    )
    if not selected_values:
        raise ValueError(f"Invalid replay episode range: {text!r}")
    available_set = set(available)
    missing = sorted(set(selected_values) - available_set)
    if missing:
        raise ValueError(
            f"Replay episode list {text!r} contains unavailable episode(s): {missing}."
        )
    return tuple(index for index in available if index in set(selected_values))


@dataclass(frozen=True)
class ReplayGroupSpec:
    key: str
    start_index: int
    end_index: int
    episode_range: str = "all"


@dataclass(frozen=True)
class ReplayInitEpisodeSpec:
    episode_index: int
    start_index: int
    end_index: int


def parse_replay_group_spec(
    spec: str,
    *,
    available_episode_indices: Sequence[int],
) -> ReplayGroupSpec:
    parts = [part.strip() for part in str(spec).split(":")]
    if len(parts) != 2:
        raise ValueError(
            "replay group spec must be 'key:episode_range', "
            "for example 'a:0-49' or 'd:50-99'."
        )
    key = parse_replay_group_key(parts[0])
    episode_range = parts[1] if parts[1] else "all"
    parse_episode_range_token(episode_range, available_indices=available_episode_indices)
    return ReplayGroupSpec(
        key=key,
        start_index=0,
        end_index=-1,
        episode_range=str(episode_range),
    )


def parse_replay_init_episode_spec(
    spec: str,
    *,
    default_total_frames: int,
) -> ReplayInitEpisodeSpec:
    text = str(spec).strip()
    if not text:
        raise ValueError("replay init episode spec is empty.")
    parts = [part.strip() for part in text.split(":")]
    if len(parts) not in {1, 3}:
        raise ValueError(
            "replay init episode spec must be 'episode' or 'episode:start:end', "
            "for example '0' or '3:0:end'."
        )
    episode_index = int(parts[0])
    start = 0
    end = int(default_total_frames)
    if len(parts) == 3:
        start = parse_replay_index_token(
            parts[1],
            total_frames=default_total_frames,
            default=0,
        )
        end = parse_replay_index_token(
            parts[2],
            total_frames=default_total_frames,
            default=default_total_frames,
        )
    if start < 0 or start >= default_total_frames:
        raise ValueError(
            f"replay init episode start_index={start} is outside [0, {default_total_frames})."
        )
    if end <= start or end > default_total_frames:
        raise ValueError(
            f"replay init episode end_index={end} must satisfy start < end <= {default_total_frames}."
        )
    return ReplayInitEpisodeSpec(
        episode_index=int(episode_index),
        start_index=int(start),
        end_index=int(end),
    )


@dataclass(frozen=True)
class ReplayTrajectoryGroup:
    key: str
    label: str
    source_path: str
    source_episode_index: int
    start_index: int
    end_index: int
    frame_count: int
    action_hz: float
    actions: np.ndarray
    arm_joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    arm_observation_current_qpos: dict[str, np.ndarray] = field(default_factory=dict)
    arm_observation_actual_qpos: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def first_action(self) -> np.ndarray:
        return self.actions[0].astype(np.float32, copy=True)


def _finite_qpos_row(values: Any, index: int) -> np.ndarray | None:
    try:
        array = np.asarray(values, dtype=float)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[1] != ARM_JOINT_DOF or array.shape[0] <= int(index):
        return None
    row = np.asarray(array[int(index)], dtype=float).reshape(-1)
    if row.shape != (ARM_JOINT_DOF,) or not np.all(np.isfinite(row)):
        return None
    return row.astype(float, copy=True)


def replay_qpos_init_candidates_at_index(
    group: ReplayTrajectoryGroup,
    side: str,
    index: int = 0,
    *,
    source: str = QPOS_INIT_AUTO,
) -> list[tuple[str, np.ndarray]]:
    normalized_source = normalize_qpos_init_source(source)
    source_arrays = {
        QPOS_INIT_REPLAY_QPOS: group.arm_joint_targets,
        QPOS_INIT_OBSERVATION_CURRENT_QPOS: group.arm_observation_current_qpos,
        QPOS_INIT_OBSERVATION_ACTUAL_QPOS: group.arm_observation_actual_qpos,
    }
    order = (
        (QPOS_INIT_REPLAY_QPOS, QPOS_INIT_OBSERVATION_CURRENT_QPOS, QPOS_INIT_OBSERVATION_ACTUAL_QPOS)
        if normalized_source == QPOS_INIT_AUTO
        else (normalized_source,)
    )
    candidates: list[tuple[str, np.ndarray]] = []
    for candidate_source in order:
        values = source_arrays.get(candidate_source, {}).get(str(side))
        row = _finite_qpos_row(values, int(index))
        if row is not None:
            candidates.append((candidate_source, row))
    return candidates


def select_replay_qpos_init_targets(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int = 0,
    source: str = QPOS_INIT_AUTO,
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    targets: dict[str, np.ndarray] = {}
    sources: dict[str, str] = {}
    for side in active_sides:
        candidates = replay_qpos_init_candidates_at_index(
            group,
            str(side),
            index=int(index),
            source=source,
        )
        if not candidates:
            continue
        selected_source, target = candidates[0]
        targets[str(side)] = target
        sources[str(side)] = selected_source
    return targets, sources


def replay_action_poses_at_index(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    index: int,
) -> dict[str, np.ndarray]:
    actions = np.asarray(group.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or actions.shape[0] <= int(index):
        raise RuntimeError(
            f"Replay action trajectory has invalid shape {actions.shape}; "
            f"expected (N, {ACTION_DIM})."
        )
    poses: dict[str, np.ndarray] = {}
    for side in active_sides:
        pose = np.asarray(actions[int(index), ACTION_ARM_POSE_SLICES[str(side)]], dtype=np.float32).reshape(-1)
        if pose.shape != (7,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"{side} replay action pose at frame {int(index)} is invalid.")
        if float(np.linalg.norm(np.asarray(pose[3:7], dtype=float))) <= 1e-12:
            raise RuntimeError(f"{side} replay action pose at frame {int(index)} has zero quaternion.")
        poses[str(side)] = pose.astype(np.float32, copy=True)
    return poses


@dataclass
class ReplayTrajectoryPool:
    key: str
    label: str
    groups: list[ReplayTrajectoryGroup]
    episode_indices: tuple[int, ...] = ()
    episode_range_spec: str = "all"
    cursor: int = 0
    started_count: int = 0

    def __post_init__(self) -> None:
        self.groups = list(self.groups)
        if not self.groups:
            raise ValueError(f"ReplayTrajectoryPool '{self.key}' requires at least one group.")

    @property
    def size(self) -> int:
        return int(len(self.groups))

    def next_group(self) -> ReplayTrajectoryGroup:
        index = int(self.cursor) % len(self.groups)
        group = self.groups[index]
        self.cursor = (index + 1) % len(self.groups)
        self.started_count += 1
        return group


@dataclass(frozen=True)
class ReplayInitEpisodePlan:
    group: ReplayTrajectoryGroup
    move_time_s: float
    to_record_time_s: float


def replay_active_arm_poses_are_finite(
    action: np.ndarray,
    active_sides: tuple[str, ...],
) -> bool:
    vector = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
    for side in active_sides:
        pose = vector[ACTION_ARM_POSE_SLICES[side]]
        if not np.all(np.isfinite(pose)):
            return False
    return True


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
    actions = all_actions[start_index:end_index].astype(np.float32, copy=True)
    arm_joint_targets = load_replay_arm_joint_targets_from_parquet(
        source_path,
        active_sides=args.active_sides,
        start_index=start_index,
        end_index=end_index,
        required=bool(require_arm_joint_targets),
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
        arm_joint_targets=arm_joint_targets,
        arm_observation_current_qpos=arm_observation_current_qpos,
        arm_observation_actual_qpos=arm_observation_actual_qpos,
        metadata={
            "action_hz_estimate": dict(action_hz_info),
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

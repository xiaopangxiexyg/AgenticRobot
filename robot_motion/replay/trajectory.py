from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES, ACTION_DIM, ARM_JOINT_DOF
from robot_motion.relative_action import (
    ACTION_POSE_MODE_ABSOLUTE,
    ACTION_POSE_MODE_RELATIVE,
    compare_pose_chunks,
    convert_arm_pose_chunk,
    normalize_action_pose_mode,
)


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
QPOS_INIT_OBSERVATION_ARM_QPOS = "observation_arm_qpos"
QPOS_INIT_SOURCES = (
    QPOS_INIT_AUTO,
    QPOS_INIT_REPLAY_QPOS,
    QPOS_INIT_OBSERVATION_CURRENT_QPOS,
    QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
    QPOS_INIT_OBSERVATION_ARM_QPOS,
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
        "audit": QPOS_INIT_OBSERVATION_ARM_QPOS,
        "arm_qpos": QPOS_INIT_OBSERVATION_ARM_QPOS,
        "observation_qpos": QPOS_INIT_OBSERVATION_ARM_QPOS,
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
    frame_times_sec: tuple[float, ...] = ()
    arm_joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    hand_joint_targets: dict[str, np.ndarray] = field(default_factory=dict)
    arm_observation_current_qpos: dict[str, np.ndarray] = field(default_factory=dict)
    arm_observation_actual_qpos: dict[str, np.ndarray] = field(default_factory=dict)
    arm_observation_actual_tcp_pose: dict[str, np.ndarray] = field(default_factory=dict)
    arm_observation_qpos_audit: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        times = tuple(float(value) for value in self.frame_times_sec)
        if times:
            if len(times) != int(self.frame_count):
                raise ValueError(
                    "frame_times_sec length must match ReplayTrajectoryGroup.frame_count"
                )
            values = np.asarray(times, dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("frame_times_sec must contain only finite values")
            values = values - values[0]
            if values.size > 1 and np.any(np.diff(values) <= 0.0):
                raise ValueError("frame_times_sec must be strictly increasing")
            times = tuple(float(value) for value in values)
        object.__setattr__(self, "frame_times_sec", times)

    def first_action(self) -> np.ndarray:
        return self.actions[0].astype(np.float32, copy=True)

    def relative_time_sec(self, index: int) -> float:
        frame_index = int(index)
        if frame_index < 0 or frame_index >= int(self.frame_count):
            raise IndexError(
                f"replay frame index {frame_index} is outside [0, {self.frame_count})"
            )
        if self.frame_times_sec:
            return float(self.frame_times_sec[frame_index])
        rate = float(self.action_hz)
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("action_hz must be finite and positive when frame times are absent")
        return float(frame_index) / rate


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
        QPOS_INIT_OBSERVATION_ARM_QPOS: group.arm_observation_qpos_audit,
    }
    order = (
        (
            QPOS_INIT_REPLAY_QPOS,
            QPOS_INIT_OBSERVATION_ARM_QPOS,
            QPOS_INIT_OBSERVATION_CURRENT_QPOS,
            QPOS_INIT_OBSERVATION_ACTUAL_QPOS,
        )
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


def replay_action_chunk_at_index(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    start_index: int,
    chunk_size: int,
    source_mode: str = ACTION_POSE_MODE_ABSOLUTE,
    target_mode: str = ACTION_POSE_MODE_ABSOLUTE,
) -> np.ndarray:
    """Return one replay chunk with an optional fixed actual-TCP conversion.

    The anchor is always the actual measured TCP at ``start_index``.  When a
    relative chunk is requested, every action row in that chunk shares that
    anchor.  The default absolute-to-absolute path is intentionally a pure
    independent copy and preserves existing replay behavior.
    """

    start = int(start_index)
    if start < 0 or start >= int(group.frame_count):
        raise IndexError(
            f"replay chunk start_index={start} is outside [0, {int(group.frame_count)})."
        )
    size = max(1, int(chunk_size))
    stop = min(int(group.frame_count), start + size)
    actions = np.asarray(group.actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < stop or actions.shape[1] != ACTION_DIM:
        raise RuntimeError(
            f"Replay action trajectory has invalid shape {actions.shape}; expected (N, {ACTION_DIM})."
        )
    source = normalize_action_pose_mode(source_mode)
    target = normalize_action_pose_mode(target_mode)
    chunk = actions[start:stop].copy()
    if source == target:
        return chunk

    anchors: dict[str, np.ndarray] = {}
    for side in active_sides:
        side_text = str(side).strip().lower()
        values = group.arm_observation_actual_tcp_pose.get(side_text)
        if values is None:
            raise RuntimeError(
                f"Replay relative conversion requires actual TCP history for {side_text!r}."
            )
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[1] == 7 and array.shape[0] > start:
            anchors[side_text] = np.asarray(array[start], dtype=np.float32).copy()
        elif array.ndim == 3 and array.shape[1:] == (4, 4) and array.shape[0] > start:
            anchors[side_text] = np.asarray(array[start], dtype=float).copy()
        else:
            raise RuntimeError(
                f"Replay actual TCP history for {side_text!r} has invalid shape {array.shape}."
            )
    return convert_arm_pose_chunk(
        chunk,
        anchors_by_side=anchors,
        source_mode=source,
        target_mode=target,
        active_sides=tuple(str(side).strip().lower() for side in active_sides),
    )


def replay_action_chunk_round_trip_metrics(
    group: ReplayTrajectoryGroup,
    active_sides: Sequence[str],
    *,
    start_index: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Report absolute -> relative -> absolute replay pose errors for one chunk."""

    absolute = replay_action_chunk_at_index(
        group,
        active_sides,
        start_index=int(start_index),
        chunk_size=int(chunk_size),
        source_mode=ACTION_POSE_MODE_ABSOLUTE,
        target_mode=ACTION_POSE_MODE_ABSOLUTE,
    )
    relative = replay_action_chunk_at_index(
        group,
        active_sides,
        start_index=int(start_index),
        chunk_size=int(chunk_size),
        source_mode=ACTION_POSE_MODE_ABSOLUTE,
        target_mode=ACTION_POSE_MODE_RELATIVE,
    )
    relative_group = ReplayTrajectoryGroup(
        key=group.key,
        label=group.label,
        source_path=group.source_path,
        source_episode_index=group.source_episode_index,
        start_index=0,
        end_index=int(relative.shape[0]),
        frame_count=int(relative.shape[0]),
        action_hz=group.action_hz,
        actions=relative,
        frame_times_sec=tuple(group.frame_times_sec[int(start_index) : int(start_index) + relative.shape[0]])
        if group.frame_times_sec
        else (),
        arm_observation_actual_tcp_pose={
            side: np.asarray(values)[
                int(start_index) : int(start_index) + int(relative.shape[0])
            ].copy()
            for side, values in group.arm_observation_actual_tcp_pose.items()
        },
        metadata=dict(group.metadata),
    )
    reconstructed = replay_action_chunk_at_index(
        relative_group,
        active_sides,
        start_index=0,
        chunk_size=int(relative.shape[0]),
        source_mode=ACTION_POSE_MODE_RELATIVE,
        target_mode=ACTION_POSE_MODE_ABSOLUTE,
    )
    metrics = compare_pose_chunks(absolute, reconstructed, active_sides=active_sides)
    return {
        "start_index": int(start_index),
        "chunk_size": int(relative.shape[0]),
        "source_mode": ACTION_POSE_MODE_ABSOLUTE,
        "relative_mode": ACTION_POSE_MODE_RELATIVE,
        "metrics_by_side": metrics,
    }


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


__all__ = [
    "QPOS_INIT_AUTO",
    "QPOS_INIT_OBSERVATION_ACTUAL_QPOS",
    "QPOS_INIT_OBSERVATION_ARM_QPOS",
    "QPOS_INIT_OBSERVATION_CURRENT_QPOS",
    "QPOS_INIT_REPLAY_QPOS",
    "QPOS_INIT_SOURCES",
    "REPLAY_TRAJECTORY_ACTION_POSE",
    "REPLAY_TRAJECTORY_QPOS_TARGET",
    "REPLAY_TRAJECTORY_SOURCES",
    "ReplayGroupSpec",
    "ReplayInitEpisodeSpec",
    "ReplayTrajectoryGroup",
    "ReplayTrajectoryPool",
    "normalize_qpos_init_source",
    "normalize_replay_trajectory_source",
    "parse_episode_range_token",
    "parse_replay_group_key",
    "parse_replay_group_spec",
    "parse_replay_index_token",
    "parse_replay_init_episode_spec",
    "replay_action_poses_at_index",
    "replay_action_chunk_at_index",
    "replay_action_chunk_round_trip_metrics",
    "replay_active_arm_poses_are_finite",
    "replay_qpos_init_candidates_at_index",
    "select_replay_qpos_init_targets",
]

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .async_writer import WriteJob
from .common import (
    EffectorCommand,
    ObservationChannel,
    nan_vector,
    to_serializable,
    vector_list,
    vector_or_none,
)

ARM_DOF = 7
HAND_DOF = 6
HAND_VECTOR_ORDER = ("right", "left")
ACTION_DIM = ARM_DOF + HAND_DOF * 2
OBSERVATION_STATE_DIM = ARM_DOF + HAND_DOF * 2

ROW_JOB_KIND = "hand_only.row"
FINALIZE_EPISODE_JOB_KIND = "hand_only.finalize_episode"


def episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def chunk_id_from_episode_index(index: int, *, chunk_size: int) -> str:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    return f"chunk-{int(index) // int(chunk_size):03d}"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(payload), ensure_ascii=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text:
            rows.append(json.loads(text))
    return rows


def write_rows_to_parquet(
    rows: list[Mapping[str, Any]],
    path: str | Path,
    *,
    compression: str = "zstd",
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Parquet export requires pyarrow to be installed.") from exc

    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([to_serializable(dict(row)) for row in rows])
    pq.write_table(table, parquet_path, compression=compression)


@dataclass(frozen=True)
class HandSample:
    index: int
    episode_index: int
    frame_index: int
    timestamp: float
    session_state: str
    control_hz: float
    active_sides: tuple[str, ...]
    action: np.ndarray
    observation_state: np.ndarray
    observation_last_action: np.ndarray
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    commands: dict[str, EffectorCommand] = field(default_factory=dict)
    hand_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class HandSampleBuilder:
    def __init__(
        self,
        *,
        active_sides: tuple[str, ...],
        episode_index: int = 0,
        sample_index_start: int = 0,
    ):
        self.active_sides = tuple(str(side) for side in active_sides)
        self._episode_index = int(episode_index)
        self._frame_index = 0
        self._sample_index = int(sample_index_start)
        self._last_action = nan_vector(ACTION_DIM)

    @property
    def frame_index(self) -> int:
        return int(self._frame_index)

    @property
    def sample_index(self) -> int:
        return int(self._sample_index)

    def set_episode(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)
        self._frame_index = 0
        self._last_action = nan_vector(ACTION_DIM)

    def _hand_command_or_nan(
        self,
        commands: Mapping[str, EffectorCommand],
        side: str,
    ) -> np.ndarray:
        command = commands.get(f"{side}_hand")
        if command is None or side not in self.active_sides:
            return nan_vector(HAND_DOF)
        value = vector_or_none(command.value, length=HAND_DOF)
        return value if value is not None else nan_vector(HAND_DOF)

    def _hand_state_or_nan(
        self,
        hand_states: Mapping[str, Mapping[str, Any] | None],
        side: str,
    ) -> np.ndarray:
        if side not in self.active_sides:
            return nan_vector(HAND_DOF)
        state = hand_states.get(side) or {}
        value = vector_or_none(state.get("joint_pos"), length=HAND_DOF)
        return value if value is not None else nan_vector(HAND_DOF)

    def _build_action_vector(self, commands: Mapping[str, EffectorCommand]) -> np.ndarray:
        return np.concatenate(
            (
                nan_vector(ARM_DOF),
                *(self._hand_command_or_nan(commands, side) for side in HAND_VECTOR_ORDER),
            ),
            dtype=np.float32,
        )

    def _build_observation_state(
        self,
        hand_states: Mapping[str, Mapping[str, Any] | None],
    ) -> np.ndarray:
        return np.concatenate(
            (
                nan_vector(ARM_DOF),
                *(self._hand_state_or_nan(hand_states, side) for side in HAND_VECTOR_ORDER),
            ),
            dtype=np.float32,
        )

    def build(
        self,
        *,
        timestamp: float,
        session_state: str,
        control_hz: float,
        channels: Mapping[str, ObservationChannel],
        commands: Mapping[str, EffectorCommand],
        hand_states: Mapping[str, Mapping[str, Any] | None],
        faults: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HandSample:
        copied_commands = {
            name: command.copy() for name, command in commands.items()
        }
        action = self._build_action_vector(copied_commands)
        observation_state = self._build_observation_state(hand_states)
        sample = HandSample(
            index=int(self._sample_index),
            episode_index=int(self._episode_index),
            frame_index=int(self._frame_index),
            timestamp=float(timestamp),
            session_state=str(session_state),
            control_hz=float(control_hz),
            active_sides=tuple(self.active_sides),
            action=action,
            observation_state=observation_state,
            observation_last_action=self._last_action.copy(),
            channels={
                name: channel.copy() for name, channel in channels.items()
            },
            commands=copied_commands,
            hand_states={
                str(side): dict(state or {}) for side, state in hand_states.items()
            },
            faults=[] if faults is None else [str(fault) for fault in faults],
            metadata={} if metadata is None else dict(to_serializable(dict(metadata))),
        )
        self._last_action = action.copy()
        self._sample_index += 1
        self._frame_index += 1
        return sample


def _serialize_channel(channel: ObservationChannel) -> dict[str, Any]:
    return {
        "timestamp": float(channel.timestamp),
        "source_name": channel.source_name,
        "side": channel.side,
        "valid": bool(channel.valid),
        "metadata": {} if channel.metadata is None else dict(to_serializable(channel.metadata)),
    }


def _split_action(action: np.ndarray) -> dict[str, list[float]]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"HandSample.action must have shape ({ACTION_DIM},).")
    result: dict[str, list[float]] = {"arm": vector[0:ARM_DOF].tolist()}
    for index, side in enumerate(HAND_VECTOR_ORDER):
        start = ARM_DOF + index * HAND_DOF
        result[f"{side}_hand"] = vector[start : start + HAND_DOF].tolist()
    return result


def _split_observation_state(observation_state: np.ndarray) -> dict[str, list[float]]:
    vector = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    if vector.shape != (OBSERVATION_STATE_DIM,):
        raise ValueError(
            f"HandSample.observation_state must have shape ({OBSERVATION_STATE_DIM},)."
        )
    result: dict[str, list[float]] = {"arm": vector[0:ARM_DOF].tolist()}
    for index, side in enumerate(HAND_VECTOR_ORDER):
        start = ARM_DOF + index * HAND_DOF
        result[f"{side}_hand"] = vector[start : start + HAND_DOF].tolist()
    return result


def hand_sample_to_row(sample: HandSample) -> dict[str, Any]:
    action_split = _split_action(sample.action)
    observation_split = _split_observation_state(sample.observation_state)
    command_metadata_by_effector = {
        name: {} if command.metadata is None else dict(command.metadata)
        for name, command in sample.commands.items()
    }
    row: dict[str, Any] = {
        "index": int(sample.index),
        "episode_index": int(sample.episode_index),
        "frame_index": int(sample.frame_index),
        "timestamp": float(sample.timestamp),
        "session_state": sample.session_state,
        "control_hz": float(sample.control_hz),
        "active_sides": list(sample.active_sides),
        "arm_state_source": "reserved_single_arm_7dof",
        "camera_keys": [],
        "fault_codes": list(sample.faults),
        "action": np.asarray(sample.action, dtype=np.float32).tolist(),
        "action.arm.joint_target": action_split["arm"],
        "action.right_hand.joint_target": action_split["right_hand"],
        "action.left_hand.joint_target": action_split["left_hand"],
        "observation.state": np.asarray(sample.observation_state, dtype=np.float32).tolist(),
        "observation.last_action": np.asarray(sample.observation_last_action, dtype=np.float32).tolist(),
        "observation.arm.qpos": observation_split["arm"],
        "observation.right_hand.qpos": observation_split["right_hand"],
        "observation.left_hand.qpos": observation_split["left_hand"],
        "sample_metadata_json": json.dumps(
            to_serializable(sample.metadata),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "channels_json": json.dumps(
            {name: _serialize_channel(channel) for name, channel in sample.channels.items()},
            ensure_ascii=True,
            sort_keys=True,
        ),
        "faults_json": json.dumps(to_serializable(sample.faults), ensure_ascii=True, sort_keys=True),
    }
    for side in HAND_VECTOR_ORDER:
        hand_state = sample.hand_states.get(side) or {}
        command_metadata = command_metadata_by_effector.get(f"{side}_hand", {})
        row[f"observation.hand.{side}.joint_pos"] = vector_list(
            hand_state.get("joint_pos"),
            length=HAND_DOF,
        )
        row[f"observation.hand.{side}.joint_timestamp"] = hand_state.get("joint_timestamp")
        row[f"observation.hand.{side}.tactile"] = vector_list(hand_state.get("tactile"))
        row[f"observation.hand.{side}.tactile_timestamp"] = hand_state.get("tactile_timestamp")
        row[f"observation.hand.{side}.telemetry_timestamp"] = hand_state.get("telemetry_timestamp")
        row[f"observation.hand.{side}.tactile_regions"] = list(
            hand_state.get("tactile_regions", ())
        )
        row[f"action.replay.{side}_hand_joint_target"] = vector_list(
            hand_state.get("last_sent_command"),
            length=HAND_DOF,
        )
        row[f"teleop.action_source.{side}_hand_q6"] = vector_list(
            command_metadata.get("source_q6"),
            length=HAND_DOF,
        )
    return row


@dataclass
class WriterConfig:
    root_dir: str | Path
    chunk_size: int = 1000
    nominal_fps: float = 30.0
    parquet_compression: str = "zstd"
    task_id: str = "hand_only"
    task_description: str = ""
    dataset_name: str = ""
    format_name: str = "hand_only_lerobot_v2.1_style"
    write_parquet: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        if float(self.nominal_fps) <= 0.0:
            raise ValueError("nominal_fps must be positive.")


@dataclass
class _EpisodeState:
    episode_index: int
    episode_id: str
    chunk_id: str
    metadata: dict[str, Any]
    rows_by_frame: dict[int, dict[str, Any]] = field(default_factory=dict)
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    control_hz_values: list[float] = field(default_factory=list)


class HandOnlyLeRobotWriter:
    def __init__(self, config: WriterConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.meta_dir = self.root_dir / "meta"
        self.data_dir = self.root_dir / "data"
        self._lock = threading.RLock()
        self._episodes: dict[int, _EpisodeState] = {}
        self._next_episode_index = 0
        self._finalized_episode_count = 0
        self._finalized_frame_count = 0
        self._prepare_dataset_root()

    @property
    def info_path(self) -> Path:
        return self.meta_dir / "info.json"

    @property
    def episodes_path(self) -> Path:
        return self.meta_dir / "episodes.jsonl"

    @property
    def tasks_path(self) -> Path:
        return self.meta_dir / "tasks.jsonl"

    @property
    def episode_stats_path(self) -> Path:
        return self.meta_dir / "episodes_stats.jsonl"

    @property
    def next_episode_index(self) -> int:
        with self._lock:
            return int(self._next_episode_index)

    def _prepare_dataset_root(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.episodes_path, self.tasks_path, self.episode_stats_path):
            path.touch(exist_ok=True)
        self._load_existing_dataset_state()
        self._ensure_task_manifest()
        self._write_info_locked()

    def _load_existing_dataset_state(self) -> None:
        episode_rows = _read_jsonl(self.episodes_path)
        stats_rows = _read_jsonl(self.episode_stats_path)
        self._finalized_episode_count = len(episode_rows)
        self._finalized_frame_count = (
            sum(int(row.get("frame_count", 0)) for row in stats_rows)
            if stats_rows
            else sum(int(row.get("frame_count", 0)) for row in episode_rows)
        )
        max_episode_index = -1
        for row in episode_rows:
            try:
                max_episode_index = max(max_episode_index, int(row.get("episode_index", -1)))
            except (TypeError, ValueError):
                continue
        self._next_episode_index = max_episode_index + 1

    def _ensure_task_manifest(self) -> None:
        if not self.config.task_id:
            return
        existing = self.tasks_path.read_text(encoding="utf-8").splitlines()
        task_prefix = f"\"task_id\": \"{self.config.task_id}\""
        if any(task_prefix in line for line in existing):
            return
        _append_jsonl(
            self.tasks_path,
            {
                "task_id": self.config.task_id,
                "description": self.config.task_description,
            },
        )

    def _write_info_locked(self) -> None:
        _write_json(
            self.info_path,
            {
                "format": self.config.format_name,
                "dataset_name": self.config.dataset_name,
                "nominal_fps": float(self.config.nominal_fps),
                "chunk_size": int(self.config.chunk_size),
                "episode_count": int(self._finalized_episode_count),
                "frame_count": int(self._finalized_frame_count),
                "write_parquet": bool(self.config.write_parquet),
                "task_id": self.config.task_id,
                "task_description": self.config.task_description,
                "reserved_spaces": {
                    "single_arm_dof": ARM_DOF,
                    "action.arm.joint_target": {"shape": [ARM_DOF], "slice": [0, ARM_DOF]},
                    "observation.arm.qpos": {"shape": [ARM_DOF], "slice": [0, ARM_DOF]},
                },
                "vector_layout": {
                    "action": {
                        "arm": [0, ARM_DOF],
                        "right_hand": [ARM_DOF, ARM_DOF + HAND_DOF],
                        "left_hand": [ARM_DOF + HAND_DOF, ACTION_DIM],
                    },
                    "observation.state": {
                        "arm": [0, ARM_DOF],
                        "right_hand": [ARM_DOF, ARM_DOF + HAND_DOF],
                        "left_hand": [ARM_DOF + HAND_DOF, OBSERVATION_STATE_DIM],
                    },
                    "hand_vector_order": list(HAND_VECTOR_ORDER),
                },
                "features": {
                    "action": {"shape": [ACTION_DIM], "dtype": "float32"},
                    "observation.state": {"shape": [OBSERVATION_STATE_DIM], "dtype": "float32"},
                    "observation.last_action": {"shape": [ACTION_DIM], "dtype": "float32"},
                    "action.arm.joint_target": {"shape": [ARM_DOF], "dtype": "float32"},
                    "observation.arm.qpos": {"shape": [ARM_DOF], "dtype": "float32"},
                    "action.right_hand.joint_target": {"shape": [HAND_DOF], "dtype": "float32"},
                    "action.left_hand.joint_target": {"shape": [HAND_DOF], "dtype": "float32"},
                    "observation.right_hand.qpos": {"shape": [HAND_DOF], "dtype": "float32"},
                    "observation.left_hand.qpos": {"shape": [HAND_DOF], "dtype": "float32"},
                },
            },
        )

    def start_episode(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        episode_index: int | None = None,
    ) -> int:
        with self._lock:
            resolved_index = int(self._next_episode_index) if episode_index is None else int(episode_index)
            self._ensure_episode_state(resolved_index, metadata=metadata)
            return resolved_index

    def _ensure_episode_state(
        self,
        episode_index: int,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> _EpisodeState:
        if episode_index in self._episodes:
            state = self._episodes[episode_index]
            if metadata:
                state.metadata.update(to_serializable(dict(metadata)))
            return state
        episode_id = episode_id_from_index(episode_index)
        chunk_id = chunk_id_from_episode_index(episode_index, chunk_size=self.config.chunk_size)
        state = _EpisodeState(
            episode_index=int(episode_index),
            episode_id=episode_id,
            chunk_id=chunk_id,
            metadata={} if metadata is None else dict(to_serializable(dict(metadata))),
        )
        self._episodes[episode_index] = state
        self._next_episode_index = max(self._next_episode_index, episode_index + 1)
        return state

    def _episode_parquet_relative_path(self, episode_index: int) -> str:
        chunk_id = chunk_id_from_episode_index(episode_index, chunk_size=self.config.chunk_size)
        return str(Path("data") / chunk_id / f"{episode_id_from_index(episode_index)}.parquet")

    def plan_sample_jobs(self, sample: HandSample) -> list[WriteJob]:
        with self._lock:
            self._ensure_episode_state(sample.episode_index)
        return [
            WriteJob(
                kind=ROW_JOB_KIND,
                payload={
                    "episode_index": int(sample.episode_index),
                    "frame_index": int(sample.frame_index),
                    "row": hand_sample_to_row(sample),
                },
            )
        ]

    def plan_finalize_episode(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> list[WriteJob]:
        return [
            WriteJob(
                kind=FINALIZE_EPISODE_JOB_KIND,
                payload={
                    "episode_index": int(episode_index),
                    "summary": {} if summary is None else dict(to_serializable(dict(summary))),
                },
            )
        ]

    def handle_job(self, job: WriteJob) -> None:
        with self._lock:
            if job.kind == ROW_JOB_KIND:
                self._handle_row_job(job.payload)
                return
            if job.kind == FINALIZE_EPISODE_JOB_KIND:
                self._finalize_episode_locked(
                    int(job.payload["episode_index"]),
                    summary=job.payload.get("summary"),
                )
                return
        raise ValueError(f"Unsupported writer job kind '{job.kind}'.")

    def _handle_row_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        frame_index = int(payload["frame_index"])
        row = dict(payload["row"])
        episode.rows_by_frame[frame_index] = row
        timestamp = row.get("timestamp")
        if timestamp is not None:
            timestamp = float(timestamp)
            if episode.start_timestamp is None:
                episode.start_timestamp = timestamp
            episode.end_timestamp = timestamp
        control_hz = row.get("control_hz")
        if control_hz is not None:
            episode.control_hz_values.append(float(control_hz))

    def discard_episode(self, episode_index: int) -> bool:
        with self._lock:
            episode = self._episodes.pop(int(episode_index), None)
            self._next_episode_index = max(
                [int(row.get("episode_index", -1)) for row in _read_jsonl(self.episodes_path)] + [-1]
            ) + 1
            return episode is not None

    def _finalize_episode_locked(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        episode = self._ensure_episode_state(int(episode_index))
        rows = [episode.rows_by_frame[index] for index in sorted(episode.rows_by_frame)]
        if not rows:
            raise RuntimeError(f"Episode {episode.episode_id} does not contain any rows.")
        parquet_relative_path = self._episode_parquet_relative_path(episode.episode_index)
        if self.config.write_parquet:
            write_rows_to_parquet(
                rows,
                self.root_dir / parquet_relative_path,
                compression=self.config.parquet_compression,
            )
        frame_count = len(rows)
        duration_sec = (
            0.0
            if episode.start_timestamp is None or episode.end_timestamp is None
            else float(episode.end_timestamp - episode.start_timestamp)
        )
        mean_control_hz = (
            None
            if not episode.control_hz_values
            else float(np.mean(np.asarray(episode.control_hz_values, dtype=float)))
        )
        episode_record = {
            "episode_index": int(episode.episode_index),
            "episode_id": episode.episode_id,
            "frame_count": int(frame_count),
            "start_timestamp": episode.start_timestamp,
            "end_timestamp": episode.end_timestamp,
            "duration_sec": float(duration_sec),
            "parquet_path": parquet_relative_path if self.config.write_parquet else None,
            "video_paths": {},
            "metadata": dict(episode.metadata),
            "summary": {} if summary is None else dict(to_serializable(dict(summary))),
        }
        episode_stats_record = {
            "episode_index": int(episode.episode_index),
            "frame_count": int(frame_count),
            "start_timestamp": episode.start_timestamp,
            "end_timestamp": episode.end_timestamp,
            "duration_sec": float(duration_sec),
            "mean_control_hz": mean_control_hz,
            "nominal_fps": float(self.config.nominal_fps),
            "active_sides": list(rows[0].get("active_sides", [])),
            "arm_state_source": "reserved_single_arm_7dof",
            "camera_keys": [],
            "fault_count": int(sum(len(row.get("fault_codes", [])) for row in rows)),
        }
        _append_jsonl(self.episodes_path, episode_record)
        _append_jsonl(self.episode_stats_path, episode_stats_record)
        self._finalized_episode_count += 1
        self._finalized_frame_count += int(frame_count)
        self._write_info_locked()
        del self._episodes[episode_index]

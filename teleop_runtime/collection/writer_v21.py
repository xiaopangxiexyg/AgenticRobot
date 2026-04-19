from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..recording import _to_serializable
from .async_writer import WriteJob
from .sample_assembler import CollectorSample
from .schema_v21 import (
    camera_video_channel,
    chunk_id_from_episode_index,
    collector_sample_to_row,
    episode_id_from_index,
    write_rows_to_parquet,
)
from .stats import EpisodeStatsAccumulator


ROW_JOB_KIND = "lerobot_v21.row"
VIDEO_FRAME_JOB_KIND = "lerobot_v21.video_frame"
FINALIZE_EPISODE_JOB_KIND = "lerobot_v21.finalize_episode"


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("LeRobot video export requires opencv-python.") from exc
    return cv2


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_serializable(payload), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_to_serializable(payload), ensure_ascii=True) + "\n")


def _coerce_video_frame(image: Any, *, color_space: str = "rgb") -> np.ndarray:
    cv2 = _require_cv2()
    frame = np.asarray(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame.astype(np.uint8, copy=False), cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 3:
        frame = frame.astype(np.uint8, copy=False)
        if str(color_space).lower() == "rgb":
            frame = frame[..., ::-1]
        elif str(color_space).lower() != "bgr":
            raise ValueError(f"Unsupported video color_space '{color_space}'.")
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = frame.astype(np.uint8, copy=False)
        if str(color_space).lower() == "rgba":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        elif str(color_space).lower() == "bgra":
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported 4-channel video color_space '{color_space}'.")
    else:
        raise ValueError(f"Unsupported video frame shape {frame.shape}.")
    return np.ascontiguousarray(frame)


@dataclass
class LeRobotV21WriterConfig:
    root_dir: str | Path
    chunk_size: int = 1000
    nominal_fps: float = 30.0
    parquet_compression: str = "zstd"
    video_codec: str = "mp4v"
    task_id: str = "default"
    task_description: str = ""
    dataset_name: str = ""
    format_name: str = "lerobot_v2.1_style"
    write_parquet: bool = True
    write_video: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        if float(self.nominal_fps) <= 0.0:
            raise ValueError("nominal_fps must be positive.")


@dataclass
class _VideoState:
    camera_key: str
    relative_path: str
    path: Path
    expected_by_sample: dict[int, bool] = field(default_factory=dict)
    pending_frames: dict[int, np.ndarray] = field(default_factory=dict)
    writer: Any | None = None
    frame_size: tuple[int, int] | None = None
    next_sample_index: int = 0
    next_video_frame_index: int = 0


@dataclass
class _EpisodeState:
    episode_index: int
    episode_id: str
    chunk_id: str
    metadata: dict[str, Any]
    rows_by_frame: dict[int, dict[str, Any]] = field(default_factory=dict)
    stats: EpisodeStatsAccumulator | None = None
    video_states: dict[str, _VideoState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stats is None:
            self.stats = EpisodeStatsAccumulator(self.episode_index)


class LeRobotV21Writer:
    def __init__(self, config: LeRobotV21WriterConfig):
        self.config = config
        self.root_dir = Path(config.root_dir)
        self.meta_dir = self.root_dir / "meta"
        self.data_dir = self.root_dir / "data"
        self.videos_dir = self.root_dir / "videos"
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

    def _prepare_dataset_root(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.episodes_path, self.tasks_path, self.episode_stats_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        self._ensure_task_manifest()
        self._write_info_locked()

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
                "write_video": bool(self.config.write_video),
                "task_id": self.config.task_id,
                "task_description": self.config.task_description,
                "features": {
                    "action": {"shape": [26], "dtype": "float32"},
                    "observation.state": {"shape": [38], "dtype": "float32"},
                    "observation.last_action": {"shape": [26], "dtype": "float32"},
                },
            },
        )

    def write_calibration_snapshot(self, name: str, payload: Any) -> Path:
        path = self.meta_dir / "calibration" / f"{name}.json"
        _write_json(path, payload)
        return path

    def write_device_config(self, name: str, payload: Any) -> Path:
        path = self.meta_dir / "devices" / f"{name}.json"
        _write_json(path, payload)
        return path

    def start_episode(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        episode_index: int | None = None,
    ) -> int:
        with self._lock:
            resolved_index = (
                int(self._next_episode_index)
                if episode_index is None
                else int(episode_index)
            )
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
                state.metadata.update(_to_serializable(dict(metadata)))
            return state
        episode_id = episode_id_from_index(episode_index)
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        state = _EpisodeState(
            episode_index=int(episode_index),
            episode_id=episode_id,
            chunk_id=chunk_id,
            metadata={} if metadata is None else dict(_to_serializable(dict(metadata))),
        )
        self._episodes[episode_index] = state
        self._next_episode_index = max(self._next_episode_index, episode_index + 1)
        return state

    def _episode_parquet_relative_path(self, episode_index: int) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(Path("data") / chunk_id / f"{episode_id_from_index(episode_index)}.parquet")

    def _episode_video_relative_path(self, episode_index: int, camera_key: str) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(
            Path("videos")
            / camera_video_channel(camera_key)
            / chunk_id
            / f"{episode_id_from_index(episode_index)}.mp4"
        )

    def _ensure_video_state(self, episode: _EpisodeState, camera_key: str) -> _VideoState:
        if camera_key in episode.video_states:
            return episode.video_states[camera_key]
        relative_path = self._episode_video_relative_path(episode.episode_index, camera_key)
        state = _VideoState(
            camera_key=str(camera_key),
            relative_path=relative_path,
            path=self.root_dir / relative_path,
        )
        episode.video_states[camera_key] = state
        return state

    def plan_sample_jobs(self, sample: CollectorSample) -> list[WriteJob]:
        with self._lock:
            self._ensure_episode_state(sample.episode_index)
        camera_video_paths = (
            {
                camera_key: self._episode_video_relative_path(sample.episode_index, camera_key)
                for camera_key in sample.camera_frames
            }
            if self.config.write_video
            else None
        )
        row, camera_expectations = collector_sample_to_row(
            sample,
            camera_video_paths=camera_video_paths,
        )
        if not self.config.write_video:
            camera_expectations = {camera_key: False for camera_key in camera_expectations}
        jobs = [
            WriteJob(
                kind=ROW_JOB_KIND,
                payload={
                    "episode_index": int(sample.episode_index),
                    "frame_index": int(sample.frame_index),
                    "row": row,
                    "camera_expectations": dict(camera_expectations),
                },
            )
        ]
        if self.config.write_video:
            for camera_key, match in sample.camera_frames.items():
                if match.frame is None:
                    continue
                jobs.append(
                    WriteJob(
                        kind=VIDEO_FRAME_JOB_KIND,
                        payload={
                            "episode_index": int(sample.episode_index),
                            "frame_index": int(sample.frame_index),
                            "camera_key": str(camera_key),
                            "image": match.frame.image,
                            "color_space": match.frame.metadata.get("color_space", "rgb"),
                        },
                    )
                )
        return jobs

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
                    "summary": {} if summary is None else dict(_to_serializable(dict(summary))),
                },
            )
        ]

    def handle_job(self, job: WriteJob) -> None:
        with self._lock:
            if job.kind == ROW_JOB_KIND:
                self._handle_row_job(job.payload)
                return
            if job.kind == VIDEO_FRAME_JOB_KIND:
                self._handle_video_frame_job(job.payload)
                return
            if job.kind == FINALIZE_EPISODE_JOB_KIND:
                self._finalize_episode_locked(
                    int(job.payload["episode_index"]),
                    summary=job.payload.get("summary"),
                )
                return
        raise ValueError(f"Unsupported LeRobotV21Writer job kind '{job.kind}'.")

    def _handle_row_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        frame_index = int(payload["frame_index"])
        row = dict(payload["row"])
        episode.rows_by_frame[frame_index] = row
        assert episode.stats is not None
        episode.stats.update(row)
        for camera_key, expects_frame in dict(payload.get("camera_expectations", {})).items():
            video_state = self._ensure_video_state(episode, str(camera_key))
            video_state.expected_by_sample[frame_index] = bool(expects_frame)
            self._drain_video_state(episode, video_state)

    def _handle_video_frame_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        camera_key = str(payload["camera_key"])
        video_state = self._ensure_video_state(episode, camera_key)
        video_state.pending_frames[int(payload["frame_index"])] = _coerce_video_frame(
            payload["image"],
            color_space=str(payload.get("color_space", "rgb")),
        )
        self._drain_video_state(episode, video_state)

    def _open_video_writer(self, state: _VideoState, frame: np.ndarray) -> None:
        cv2 = _require_cv2()
        height, width = frame.shape[:2]
        state.path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(state.path),
            cv2.VideoWriter_fourcc(*self.config.video_codec),
            float(self.config.nominal_fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for '{state.path}'.")
        state.writer = writer
        state.frame_size = (int(width), int(height))

    def _drain_video_state(self, episode: _EpisodeState, state: _VideoState) -> None:
        while state.next_sample_index in state.expected_by_sample:
            sample_index = int(state.next_sample_index)
            expects_frame = bool(state.expected_by_sample[sample_index])
            if not expects_frame:
                del state.expected_by_sample[sample_index]
                state.next_sample_index += 1
                continue
            if sample_index not in state.pending_frames:
                return
            frame = state.pending_frames.pop(sample_index)
            if state.writer is None:
                self._open_video_writer(state, frame)
            assert state.frame_size is not None
            expected_width, expected_height = state.frame_size
            if frame.shape[1] != expected_width or frame.shape[0] != expected_height:
                raise ValueError(
                    f"Video frame for camera '{state.camera_key}' has shape {frame.shape[:2]}, "
                    f"expected {(expected_height, expected_width)}."
                )
            assert state.writer is not None
            state.writer.write(frame)
            row = episode.rows_by_frame.get(sample_index)
            if row is not None:
                row[f"{camera_video_channel(state.camera_key)}.video_frame_index"] = int(
                    state.next_video_frame_index
                )
            del state.expected_by_sample[sample_index]
            state.next_video_frame_index += 1
            state.next_sample_index += 1

    def _close_video_state(self, state: _VideoState) -> None:
        if state.writer is not None:
            state.writer.release()
            state.writer = None

    def _finalize_episode_locked(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        episode = self._ensure_episode_state(int(episode_index))
        for state in episode.video_states.values():
            self._drain_video_state(episode, state)
            unresolved = [
                sample_index
                for sample_index, expects_frame in sorted(state.expected_by_sample.items())
                if expects_frame
            ]
            if unresolved:
                raise RuntimeError(
                    f"Episode {episode.episode_id} camera '{state.camera_key}' is missing "
                    f"selected frames for sample indices {unresolved[:8]}."
                )
            self._close_video_state(state)

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

        assert episode.stats is not None
        stats = episode.stats.finalize(nominal_fps=self.config.nominal_fps)
        video_paths = {
            camera_key: state.relative_path
            for camera_key, state in episode.video_states.items()
            if state.next_video_frame_index > 0
        }
        episode_record = {
            "episode_index": int(episode.episode_index),
            "episode_id": episode.episode_id,
            "frame_count": int(stats.frame_count),
            "start_timestamp": stats.start_timestamp,
            "end_timestamp": stats.end_timestamp,
            "duration_sec": float(stats.duration_sec),
            "parquet_path": parquet_relative_path if self.config.write_parquet else None,
            "video_paths": video_paths,
            "metadata": dict(episode.metadata),
            "summary": {} if summary is None else dict(_to_serializable(dict(summary))),
        }
        episode_stats_record = {
            "episode_index": int(stats.episode_index),
            "frame_count": int(stats.frame_count),
            "start_timestamp": stats.start_timestamp,
            "end_timestamp": stats.end_timestamp,
            "duration_sec": float(stats.duration_sec),
            "average_dt_sec": stats.average_dt_sec,
            "average_hz": stats.average_hz,
            "mean_control_hz": stats.mean_control_hz,
            "nominal_fps": float(stats.nominal_fps),
            "active_sides": list(stats.active_sides),
            "arm_state_source": stats.arm_state_source,
            "camera_keys": list(stats.camera_keys),
            "fault_count": int(stats.fault_count),
        }
        _append_jsonl(self.episodes_path, episode_record)
        _append_jsonl(self.episode_stats_path, episode_stats_record)
        self._finalized_episode_count += 1
        self._finalized_frame_count += int(stats.frame_count)
        self._write_info_locked()
        del self._episodes[episode_index]

    def finalize_episode(
        self,
        episode_index: int,
        *,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._finalize_episode_locked(episode_index, summary=summary)

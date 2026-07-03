from __future__ import annotations

import json
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..action import (
    arm_action_specs_metadata,
    hand_action_specs_metadata,
    normalize_arm_action_specs,
    normalize_hand_action_specs,
)
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
TACTILE_FRAME_JOB_KIND = "lerobot_v21.tactile_frame"
FINALIZE_EPISODE_JOB_KIND = "lerobot_v21.finalize_episode"
TACTILE_STORAGE_EXTERNAL = "external"
TACTILE_STORAGE_PARQUET = "parquet"
TACTILE_STREAM_FORMAT = "binary_stream_v1"


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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        rows.append(json.loads(text))
    return rows


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


def _safe_path_token(value: Any) -> str:
    text = str(value).strip() or "unnamed"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unnamed"


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
    tactile_storage: str = TACTILE_STORAGE_EXTERNAL
    allow_missing_video_frames: bool = False
    arm_action_specs: Mapping[str, Any] | None = None
    hand_action_specs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        if float(self.nominal_fps) <= 0.0:
            raise ValueError("nominal_fps must be positive.")
        storage = str(self.tactile_storage).strip().lower().replace("-", "_")
        if storage not in {TACTILE_STORAGE_PARQUET, TACTILE_STORAGE_EXTERNAL}:
            raise ValueError("tactile_storage must be parquet or external.")
        self.tactile_storage = storage
        self.arm_action_specs = normalize_arm_action_specs(self.arm_action_specs)
        self.hand_action_specs = normalize_hand_action_specs(self.hand_action_specs)


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
    expected_frame_count: int = 0
    written_frame_count: int = 0
    missing_frame_count: int = 0


@dataclass
class _TactileSensorStream:
    sensor_key: str
    relative_path: str
    path: Path
    dtype: str | None = None
    shape: tuple[int, ...] | None = None
    frame_count: int = 0
    handle: Any | None = None

    def append(self, value: Any) -> None:
        array = np.asarray(value)
        if array.dtype == np.dtype("O"):
            raise ValueError(f"Tactile sensor '{self.sensor_key}' has unsupported object dtype.")
        dtype = str(array.dtype)
        if self.dtype is None:
            self.dtype = dtype
        elif str(self.dtype) != dtype:
            raise ValueError(
                f"Tactile sensor '{self.sensor_key}' dtype changed from "
                f"{self.dtype} to {dtype}."
            )
        if self.shape is None:
            self.shape = tuple(int(dim) for dim in array.shape)
        elif tuple(int(dim) for dim in array.shape) != self.shape:
            raise ValueError(
                f"Tactile sensor '{self.sensor_key}' shape changed from "
                f"{self.shape} to {tuple(array.shape)}."
            )
        if self.handle is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.handle = self.path.open("ab")
        contiguous = np.ascontiguousarray(array)
        self.handle.write(contiguous.tobytes(order="C"))
        self.frame_count += 1

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None


@dataclass
class _TactileSideState:
    side: str
    relative_dir: str
    path_dir: Path
    manifest_relative_path: str
    manifest_path: Path
    expected_by_sample: dict[int, bool] = field(default_factory=dict)
    pending_frames: dict[int, Mapping[str, Any]] = field(default_factory=dict)
    streams: dict[str, _TactileSensorStream] = field(default_factory=dict)
    sensor_key_order: tuple[str, ...] | None = None
    frame_records: list[dict[str, Any]] = field(default_factory=list)
    next_sample_index: int = 0
    next_tactile_frame_index: int = 0


@dataclass
class _EpisodeState:
    episode_index: int
    episode_id: str
    chunk_id: str
    metadata: dict[str, Any]
    rows_by_frame: dict[int, dict[str, Any]] = field(default_factory=dict)
    stats: EpisodeStatsAccumulator | None = None
    video_states: dict[str, _VideoState] = field(default_factory=dict)
    tactile_states: dict[str, _TactileSideState] = field(default_factory=dict)

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

    @property
    def next_episode_index(self) -> int:
        with self._lock:
            return int(self._next_episode_index)

    def _prepare_dataset_root(self) -> None:
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.episodes_path, self.tasks_path, self.episode_stats_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch(exist_ok=True)
        self._load_existing_dataset_state()
        self._ensure_task_manifest()
        self._write_info_locked()

    def _load_existing_dataset_state(self) -> None:
        episode_rows = _read_jsonl(self.episodes_path)
        stats_rows = _read_jsonl(self.episode_stats_path)
        self._finalized_episode_count = len(episode_rows)
        if stats_rows:
            self._finalized_frame_count = sum(
                int(row.get("frame_count", 0))
                for row in stats_rows
            )
        else:
            self._finalized_frame_count = sum(
                int(row.get("frame_count", 0))
                for row in episode_rows
            )
        max_episode_index = -1
        for row in episode_rows:
            try:
                max_episode_index = max(max_episode_index, int(row.get("episode_index", -1)))
            except (TypeError, ValueError):
                continue
        self._next_episode_index = max(self._next_episode_index, max_episode_index + 1)

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
        arm_specs = arm_action_specs_metadata(self.config.arm_action_specs)
        hand_specs = hand_action_specs_metadata(self.config.hand_action_specs)
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
                "tactile_storage": str(self.config.tactile_storage),
                "tactile_stream_format": (
                    TACTILE_STREAM_FORMAT
                    if self.config.tactile_storage == TACTILE_STORAGE_EXTERNAL
                    else None
                ),
                "task_id": self.config.task_id,
                "task_description": self.config.task_description,
                "arm_action_specs_by_side": arm_specs,
                "arm_controlled_frame_by_side": {
                    side: spec["controlled_frame"] for side, spec in arm_specs.items()
                },
                "arm_reference_frame_by_side": {
                    side: spec["reference_frame"] for side, spec in arm_specs.items()
                },
                "arm_spec_required": True,
                "hand_action_specs_by_side": hand_specs,
                "hand_active_mask_by_side": {
                    side: list(spec["active_mask"]) for side, spec in hand_specs.items()
                },
                "hand_spec_required": True,
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

    def tactile_frame_counts(self, episode_index: int) -> dict[str, int]:
        with self._lock:
            episode = self._episodes.get(int(episode_index))
            if episode is None:
                return {}
            return {
                side: int(state.next_tactile_frame_index)
                for side, state in episode.tactile_states.items()
                if int(state.next_tactile_frame_index) > 0
            }

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
            state.metadata.setdefault(
                "arm_action_specs_by_side",
                arm_action_specs_metadata(self.config.arm_action_specs),
            )
            state.metadata.setdefault(
                "hand_action_specs_by_side",
                hand_action_specs_metadata(self.config.hand_action_specs),
            )
            return state
        episode_id = episode_id_from_index(episode_index)
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        episode_metadata = {} if metadata is None else dict(_to_serializable(dict(metadata)))
        episode_metadata.setdefault(
            "arm_action_specs_by_side",
            arm_action_specs_metadata(self.config.arm_action_specs),
        )
        episode_metadata.setdefault(
            "arm_controlled_frame_by_side",
            {
                side: spec.controlled_frame
                for side, spec in self.config.arm_action_specs.items()
            },
        )
        episode_metadata.setdefault(
            "arm_reference_frame_by_side",
            {
                side: spec.reference_frame
                for side, spec in self.config.arm_action_specs.items()
            },
        )
        episode_metadata.setdefault("arm_spec_required", True)
        episode_metadata.setdefault(
            "hand_action_specs_by_side",
            hand_action_specs_metadata(self.config.hand_action_specs),
        )
        episode_metadata.setdefault(
            "hand_active_mask_by_side",
            {
                side: list(spec.active_mask or ())
                for side, spec in self.config.hand_action_specs.items()
            },
        )
        episode_metadata.setdefault("hand_spec_required", True)
        state = _EpisodeState(
            episode_index=int(episode_index),
            episode_id=episode_id,
            chunk_id=chunk_id,
            metadata=episode_metadata,
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

    def _episode_tactile_relative_dir(self, episode_index: int, side: str) -> str:
        chunk_id = chunk_id_from_episode_index(
            episode_index,
            chunk_size=self.config.chunk_size,
        )
        return str(
            Path("tactile")
            / f"observation.hand.{str(side)}.tactile"
            / chunk_id
            / episode_id_from_index(episode_index)
        )

    def _episode_tactile_manifest_relative_path(self, episode_index: int, side: str) -> str:
        return str(Path(self._episode_tactile_relative_dir(episode_index, side)) / "manifest.json")

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

    def _ensure_tactile_state(self, episode: _EpisodeState, side: str) -> _TactileSideState:
        side = str(side)
        if side in episode.tactile_states:
            return episode.tactile_states[side]
        relative_dir = self._episode_tactile_relative_dir(episode.episode_index, side)
        manifest_relative_path = self._episode_tactile_manifest_relative_path(
            episode.episode_index,
            side,
        )
        state = _TactileSideState(
            side=side,
            relative_dir=relative_dir,
            path_dir=self.root_dir / relative_dir,
            manifest_relative_path=manifest_relative_path,
            manifest_path=self.root_dir / manifest_relative_path,
        )
        episode.tactile_states[side] = state
        return state

    def _tactile_arrays_from_hand_state(
        self,
        hand_state: Mapping[str, Any],
    ) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        tactile_dict = hand_state.get("tactile_dict")
        if isinstance(tactile_dict, Mapping):
            for key, value in tactile_dict.items():
                if value is None:
                    continue
                array = np.asarray(value)
                if array.size <= 0:
                    continue
                arrays[str(key)] = array
        if arrays:
            return arrays
        tactile = hand_state.get("tactile")
        if tactile is None:
            return {}
        array = np.asarray(tactile, dtype=np.float32).reshape(-1)
        if array.size <= 0:
            return {}
        return {"vector": array}

    def _plan_external_tactile_jobs(
        self,
        sample: CollectorSample,
    ) -> tuple[list[WriteJob], dict[str, dict[str, Any]], dict[str, bool]]:
        jobs: list[WriteJob] = []
        refs: dict[str, dict[str, Any]] = {}
        expectations: dict[str, bool] = {}
        for side in ("left", "right"):
            hand_state = sample.hand_states.get(side) or {}
            arrays = self._tactile_arrays_from_hand_state(hand_state)
            has_tactile = bool(arrays)
            expectations[side] = has_tactile
            if not has_tactile:
                continue
            manifest_relative_path = self._episode_tactile_manifest_relative_path(
                sample.episode_index,
                side,
            )
            sensor_keys = sorted(arrays)
            refs[side] = {
                "format": TACTILE_STREAM_FORMAT,
                "path": manifest_relative_path,
                "frame_index": None,
                "sensor_keys": sensor_keys,
            }
            jobs.append(
                WriteJob(
                    kind=TACTILE_FRAME_JOB_KIND,
                    payload={
                        "episode_index": int(sample.episode_index),
                        "frame_index": int(sample.frame_index),
                        "side": side,
                        "arrays": arrays,
                        "timestamp": hand_state.get("tactile_timestamp"),
                        "telemetry_timestamp": hand_state.get("telemetry_timestamp"),
                        "regions": list(
                            hand_state.get("tactile_regions", hand_state.get("tactile_devices", ()))
                        ),
                        "frame_ids": dict(hand_state.get("tactile_frame_ids", {}) or {}),
                        "shapes": dict(hand_state.get("tactile_shapes", {}) or {}),
                        "field_by_key": dict(hand_state.get("tactile_field_by_key", {}) or {}),
                        "fields": list(hand_state.get("tactile_fields", ()) or ()),
                        "version": hand_state.get("tactile_version"),
                    },
                )
            )
        return jobs, refs, expectations

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
        tactile_jobs: list[WriteJob] = []
        tactile_refs: dict[str, dict[str, Any]] | None = None
        tactile_expectations: dict[str, bool] = {}
        inline_tactile = self.config.tactile_storage != TACTILE_STORAGE_EXTERNAL
        if self.config.tactile_storage == TACTILE_STORAGE_EXTERNAL:
            tactile_jobs, tactile_refs, tactile_expectations = self._plan_external_tactile_jobs(
                sample
            )
        row, camera_expectations = collector_sample_to_row(
            sample,
            camera_video_paths=camera_video_paths,
            tactile_storage_refs=tactile_refs,
            inline_tactile=inline_tactile,
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
                    "tactile_expectations": dict(tactile_expectations),
                },
            )
        ]
        jobs.extend(tactile_jobs)
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
                        critical=False,
                        metadata={
                            "storage_queue": "video",
                            "camera_key": str(camera_key),
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
            if job.kind == TACTILE_FRAME_JOB_KIND:
                self._handle_tactile_frame_job(job.payload)
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
            if bool(expects_frame):
                video_state.expected_frame_count += 1
            self._drain_video_state(episode, video_state)
        for side, expects_frame in dict(payload.get("tactile_expectations", {})).items():
            tactile_state = self._ensure_tactile_state(episode, str(side))
            tactile_state.expected_by_sample[frame_index] = bool(expects_frame)
            self._drain_tactile_state(episode, tactile_state)
            self._apply_latest_tactile_ref_to_row(row, tactile_state)

    def _handle_video_frame_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        camera_key = str(payload["camera_key"])
        video_state = self._ensure_video_state(episode, camera_key)
        video_state.pending_frames[int(payload["frame_index"])] = _coerce_video_frame(
            payload["image"],
            color_space=str(payload.get("color_space", "rgb")),
        )
        self._drain_video_state(episode, video_state)

    def _handle_tactile_frame_job(self, payload: Mapping[str, Any]) -> None:
        episode = self._ensure_episode_state(int(payload["episode_index"]))
        side = str(payload["side"])
        tactile_state = self._ensure_tactile_state(episode, side)
        tactile_state.pending_frames[int(payload["frame_index"])] = payload
        self._drain_tactile_state(episode, tactile_state)

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
            state.written_frame_count += 1
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

    def _sensor_stream_for(
        self,
        state: _TactileSideState,
        sensor_key: str,
        dtype: str,
    ) -> _TactileSensorStream:
        if sensor_key in state.streams:
            return state.streams[sensor_key]
        safe_name = _safe_path_token(sensor_key)
        used_names = {
            Path(stream.relative_path).name.rsplit(".", 2)[0]
            for stream in state.streams.values()
        }
        base_name = safe_name
        suffix = 1
        while safe_name in used_names:
            suffix += 1
            safe_name = f"{base_name}_{suffix}"
        safe_dtype = _safe_path_token(dtype)
        relative_path = str(Path(state.relative_dir) / f"{safe_name}.{safe_dtype}.bin")
        stream = _TactileSensorStream(
            sensor_key=str(sensor_key),
            relative_path=relative_path,
            path=self.root_dir / relative_path,
            dtype=str(dtype),
        )
        state.streams[sensor_key] = stream
        return stream

    def _append_tactile_payload(
        self,
        state: _TactileSideState,
        *,
        sample_index: int,
        payload: Mapping[str, Any],
    ) -> tuple[int, tuple[str, ...]]:
        raw_arrays = payload.get("arrays", {})
        if not isinstance(raw_arrays, Mapping):
            raise ValueError("Tactile frame payload arrays must be a mapping.")
        arrays = {
            str(key): np.asarray(value)
            for key, value in raw_arrays.items()
            if value is not None and np.asarray(value).size > 0
        }
        if not arrays:
            raise ValueError("Tactile frame payload does not contain any arrays.")
        sensor_keys = tuple(sorted(arrays))
        if state.sensor_key_order is None:
            state.sensor_key_order = sensor_keys
        elif sensor_keys != state.sensor_key_order:
            raise ValueError(
                f"Tactile sensor set for side '{state.side}' changed from "
                f"{state.sensor_key_order} to {sensor_keys}."
            )

        tactile_frame_index = int(state.next_tactile_frame_index)
        for sensor_key in sensor_keys:
            array = arrays[sensor_key]
            if array.dtype == np.dtype("O"):
                raise ValueError(f"Tactile sensor '{sensor_key}' has unsupported object dtype.")
            self._sensor_stream_for(state, sensor_key, str(array.dtype)).append(array)
        dtypes = {sensor_key: str(arrays[sensor_key].dtype) for sensor_key in sensor_keys}
        state.frame_records.append(
            {
                "sample_frame_index": int(sample_index),
                "tactile_frame_index": tactile_frame_index,
                "timestamp": payload.get("timestamp"),
                "telemetry_timestamp": payload.get("telemetry_timestamp"),
                "regions": list(payload.get("regions", ())),
                "frame_ids": dict(payload.get("frame_ids", {}) or {}),
                "shapes": dict(payload.get("shapes", {}) or {}),
                "dtypes": dtypes,
                "field_by_key": dict(payload.get("field_by_key", {}) or {}),
                "fields": list(payload.get("fields", ()) or ()),
                "version": payload.get("version"),
            }
        )
        return tactile_frame_index, sensor_keys

    def _apply_latest_tactile_ref_to_row(
        self,
        row: dict[str, Any],
        state: _TactileSideState,
    ) -> None:
        latest_frame_index = int(state.next_tactile_frame_index) - 1
        if latest_frame_index < 0:
            return
        prefix = f"observation.hand.{state.side}.tactile"
        row[f"{prefix}_storage_format"] = TACTILE_STREAM_FORMAT
        row[f"{prefix}_path"] = state.manifest_relative_path
        row[f"{prefix}_frame_index"] = latest_frame_index
        if state.sensor_key_order is not None:
            row[f"{prefix}_sensor_keys"] = list(state.sensor_key_order)

    def _drain_tactile_state(self, episode: _EpisodeState, state: _TactileSideState) -> None:
        while state.next_sample_index in state.expected_by_sample:
            sample_index = int(state.next_sample_index)
            expects_frame = bool(state.expected_by_sample[sample_index])
            if not expects_frame:
                del state.expected_by_sample[sample_index]
                state.next_sample_index += 1
                continue
            if sample_index not in state.pending_frames:
                return
            payload = state.pending_frames.pop(sample_index)
            tactile_frame_index, sensor_keys = self._append_tactile_payload(
                state,
                sample_index=sample_index,
                payload=payload,
            )
            row = episode.rows_by_frame.get(sample_index)
            if row is not None:
                prefix = f"observation.hand.{state.side}.tactile"
                row[f"{prefix}_storage_format"] = TACTILE_STREAM_FORMAT
                row[f"{prefix}_path"] = state.manifest_relative_path
                row[f"{prefix}_frame_index"] = int(tactile_frame_index)
                row[f"{prefix}_sensor_keys"] = list(sensor_keys)
            del state.expected_by_sample[sample_index]
            state.next_tactile_frame_index += 1
            state.next_sample_index += 1

    def _close_tactile_state(self, state: _TactileSideState) -> str | None:
        for stream in state.streams.values():
            stream.close()
        if not state.frame_records:
            return None
        streams = {
            sensor_key: {
                "path": stream.relative_path,
                "dtype": stream.dtype,
                "shape": [] if stream.shape is None else list(stream.shape),
                "frame_count": int(stream.frame_count),
            }
            for sensor_key, stream in state.streams.items()
        }
        _write_json(
            state.manifest_path,
            {
                "format": TACTILE_STREAM_FORMAT,
                "side": state.side,
                "relative_dir": state.relative_dir,
                "frame_count": len(state.frame_records),
                "sensor_keys": list(state.sensor_key_order or ()),
                "streams": streams,
                "frames": list(state.frame_records),
            },
        )
        return state.manifest_relative_path

    def _recompute_next_episode_index_locked(self) -> None:
        max_episode_index = -1
        for row in _read_jsonl(self.episodes_path):
            try:
                max_episode_index = max(max_episode_index, int(row.get("episode_index", -1)))
            except (TypeError, ValueError):
                continue
        if self._episodes:
            max_episode_index = max(max_episode_index, max(self._episodes))
        self._next_episode_index = max_episode_index + 1

    def discard_episode(self, episode_index: int) -> bool:
        with self._lock:
            episode = self._episodes.pop(int(episode_index), None)
            if episode is None:
                self._recompute_next_episode_index_locked()
                return False
            for state in episode.video_states.values():
                self._close_video_state(state)
                try:
                    state.path.unlink(missing_ok=True)
                except OSError:
                    pass
            for state in episode.tactile_states.values():
                self._close_tactile_state(state)
                shutil.rmtree(state.path_dir, ignore_errors=True)
            self._recompute_next_episode_index_locked()
            return True

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
                state.missing_frame_count += len(unresolved)
                for sample_index in unresolved:
                    state.expected_by_sample.pop(sample_index, None)
                if not bool(self.config.allow_missing_video_frames):
                    raise RuntimeError(
                        f"Episode {episode.episode_id} camera '{state.camera_key}' is missing "
                        f"selected frames for sample indices {unresolved[:8]}."
                    )
            self._close_video_state(state)
        tactile_paths: dict[str, str] = {}
        for state in episode.tactile_states.values():
            self._drain_tactile_state(episode, state)
            unresolved = [
                sample_index
                for sample_index, expects_frame in sorted(state.expected_by_sample.items())
                if expects_frame
            ]
            if unresolved:
                raise RuntimeError(
                    f"Episode {episode.episode_id} tactile side '{state.side}' is missing "
                    f"frames for sample indices {unresolved[:8]}."
                )
            manifest_path = self._close_tactile_state(state)
            if manifest_path is not None:
                tactile_paths[state.side] = manifest_path

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
        summary_payload = {} if summary is None else dict(_to_serializable(dict(summary)))
        storage_queue_stats = summary_payload.get("storage_queue_stats")
        storage_routes = (
            storage_queue_stats.get("routes", {})
            if isinstance(storage_queue_stats, Mapping)
            else {}
        )
        video_frame_stats = {}
        for camera_key, state in episode.video_states.items():
            route_stats = storage_routes.get(camera_key, {})
            dropped = (
                int(route_stats.get("dropped", 0) or 0)
                if isinstance(route_stats, Mapping)
                else 0
            )
            video_frame_stats[camera_key] = {
                "expected": int(state.expected_frame_count),
                "written": int(state.written_frame_count),
                "dropped": dropped,
                "missing": int(state.missing_frame_count),
                "path": state.relative_path if state.next_video_frame_index > 0 else None,
            }
        if video_frame_stats:
            summary_payload.setdefault("video_frame_stats", video_frame_stats)
        episode_record = {
            "episode_index": int(episode.episode_index),
            "episode_id": episode.episode_id,
            "frame_count": int(stats.frame_count),
            "start_timestamp": stats.start_timestamp,
            "end_timestamp": stats.end_timestamp,
            "duration_sec": float(stats.duration_sec),
            "parquet_path": parquet_relative_path if self.config.write_parquet else None,
            "video_paths": video_paths,
            "tactile_paths": tactile_paths,
            "metadata": dict(episode.metadata),
            "summary": summary_payload,
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

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..scene import MujocoScene
from ..transforms import pose_to_xyz_quat_xyzw
from .trajectory import GripperReplaySample, QposSample


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_to_serializable(item) for item in value]
    return value


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
        if text:
            value = json.loads(text)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _write_rows_to_parquet(
    rows: list[Mapping[str, Any]],
    path: Path,
    *,
    compression: str = "zstd",
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Sim replay dataset export requires pyarrow.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([_to_serializable(dict(row)) for row in rows])
    pq.write_table(table, path, compression=compression)


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Sim replay video export requires opencv-python.") from exc
    return cv2


def _episode_id(index: int) -> str:
    return f"episode_{int(index):06d}"


def _chunk_id(index: int, *, chunk_size: int) -> str:
    return f"chunk-{int(index) // max(1, int(chunk_size)):03d}"


def _next_episode_index(root_dir: Path) -> int:
    records = _read_jsonl(root_dir / "meta" / "episodes.jsonl")
    indices = []
    for record in records:
        try:
            indices.append(int(record.get("episode_index")))
        except Exception:
            continue
    return (max(indices) + 1) if indices else 0


def _coerce_video_frame_rgb_to_bgr(image: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    frame = np.asarray(image, dtype=np.uint8)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return np.ascontiguousarray(frame[..., ::-1])
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported sim camera frame shape {frame.shape}.")


def _zero_vector(length: int) -> list[float]:
    return [0.0 for _ in range(int(length))]


@dataclass(frozen=True)
class SimReplayDatasetWriterConfig:
    root_dir: Path
    dataset_name: str = ""
    task_id: str = "mujoco_real2sim_replay"
    task_description: str = "MuJoCo real2sim replay"
    nominal_fps: float = 30.0
    chunk_size: int = 1000
    episode_index: int | None = None
    video_codec: str = "mp4v"
    parquet_compression: str = "zstd"
    source_dataset_root: Path | None = None
    source_episode_id: str = ""
    source_parquet_path: Path | None = None
    urdf_path: Path | None = None
    mode: str = "dual"
    qpos_source: str = "auto"
    gripper_command_dim: int = 1


class SimReplayDatasetWriter:
    def __init__(
        self,
        scene: MujocoScene,
        config: SimReplayDatasetWriterConfig,
        *,
        camera_names: Sequence[str],
    ):
        self.scene = scene
        self.config = config
        self.root_dir = Path(config.root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.camera_names = tuple(str(name) for name in camera_names)
        self.episode_index = (
            int(config.episode_index)
            if config.episode_index is not None
            else _next_episode_index(self.root_dir)
        )
        self.episode_id = _episode_id(self.episode_index)
        self.chunk_id = _chunk_id(self.episode_index, chunk_size=int(config.chunk_size))
        self.parquet_relative_path = str(
            Path("data") / self.chunk_id / f"{self.episode_id}.parquet"
        )
        self.video_relative_paths = {
            camera_name: str(
                Path("videos")
                / f"observation.images.{camera_name}"
                / self.chunk_id
                / f"{self.episode_id}.mp4"
            )
            for camera_name in self.camera_names
        }
        self.rows: list[dict[str, Any]] = []
        self._video_writers: dict[str, Any] = {}
        self._video_sizes: dict[str, tuple[int, int]] = {}
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._fallback_timestamp_origin = time.time()
        self._closed = False
        self._write_initial_metadata()

    def _write_initial_metadata(self) -> None:
        (self.root_dir / "meta").mkdir(parents=True, exist_ok=True)
        tasks_path = self.root_dir / "meta" / "tasks.jsonl"
        task_id = str(self.config.task_id or "mujoco_real2sim_replay")
        existing_task_ids = {
            str(row.get("task_id") or row.get("task_index") or "")
            for row in _read_jsonl(tasks_path)
        }
        if task_id not in existing_task_ids:
            _append_jsonl(
                tasks_path,
                {
                    "task_index": 0,
                    "task_id": task_id,
                    "task": str(self.config.task_description or task_id),
                },
            )
        _write_json(self.root_dir / "meta" / "sim_replay_export.json", self._export_metadata())
        self._write_info()

    def _export_metadata(self) -> dict[str, Any]:
        return {
            "created_at": self._started_at,
            "format": "dexteleop_mujoco_real2sim_replay_dataset",
            "source_dataset_root": self.config.source_dataset_root,
            "source_episode_id": self.config.source_episode_id,
            "source_parquet_path": self.config.source_parquet_path,
            "urdf_path": self.config.urdf_path,
            "mjcf_path": self.scene.conversion.mjcf_path,
            "mode": self.config.mode,
            "qpos_source": self.config.qpos_source,
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "camera_names": list(self.camera_names),
            "camera_metadata": self.scene.camera_metadata(),
        }

    def _write_info(self) -> None:
        records = _read_jsonl(self.root_dir / "meta" / "episodes.jsonl")
        frame_count = 0
        for record in records:
            try:
                frame_count += int(record.get("frame_count") or 0)
            except Exception:
                continue
        info = {
            "format": "lerobot_v2.1_style",
            "dataset_name": self.config.dataset_name or self.root_dir.name,
            "nominal_fps": float(self.config.nominal_fps),
            "chunk_size": int(self.config.chunk_size),
            "episode_count": len(records),
            "frame_count": int(frame_count),
            "write_parquet": True,
            "write_video": True,
            "tactile_storage": "none",
            "task_id": self.config.task_id,
            "task_description": self.config.task_description,
            "features": {
                "action": {"shape": [26], "dtype": "float32"},
                "observation.state": {"shape": [38], "dtype": "float32"},
                "observation.last_action": {"shape": [26], "dtype": "float32"},
            },
            "sim_replay": self._export_metadata(),
        }
        _write_json(self.root_dir / "meta" / "info.json", info)

    def _open_video_writer(self, camera_name: str, image: np.ndarray) -> None:
        cv2 = _require_cv2()
        frame = _coerce_video_frame_rgb_to_bgr(image)
        height, width = frame.shape[:2]
        path = self.root_dir / self.video_relative_paths[camera_name]
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*str(self.config.video_codec)),
            float(self.config.nominal_fps),
            (int(width), int(height)),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open sim replay video writer: {path}")
        self._video_writers[camera_name] = writer
        self._video_sizes[camera_name] = (int(width), int(height))
        writer.write(frame)

    def _write_video_frame(self, camera_name: str, image: np.ndarray) -> None:
        if camera_name not in self._video_writers:
            self._open_video_writer(camera_name, image)
            return
        frame = _coerce_video_frame_rgb_to_bgr(image)
        expected = self._video_sizes[camera_name]
        if (int(frame.shape[1]), int(frame.shape[0])) != expected:
            raise ValueError(
                f"Camera {camera_name!r} frame size changed from {expected} to "
                f"{(int(frame.shape[1]), int(frame.shape[0]))}."
            )
        self._video_writers[camera_name].write(frame)

    def _side_pose(self, side: str) -> list[float]:
        spec = self.scene.arm_specs[side]
        return pose_to_xyz_quat_xyzw(self.scene.body_pose(spec.controlled_body))

    def _hand_qpos_by_side(
        self,
        gripper_close_by_side: Mapping[str, float] | None,
    ) -> dict[str, list[float]]:
        out = {"left": _zero_vector(6), "right": _zero_vector(6)}
        if not gripper_close_by_side:
            return out
        index = int(np.clip(int(self.config.gripper_command_dim), 0, 5))
        for side, close in gripper_close_by_side.items():
            if side not in out:
                continue
            q6 = _zero_vector(6)
            q6[index] = float(np.clip(float(close), 0.0, 1.0))
            out[str(side)] = q6
        return out

    def _observation_state(
        self,
        qpos_by_side: Mapping[str, Sequence[float]],
        hand_qpos_by_side: Mapping[str, Sequence[float]],
    ) -> list[float]:
        left_qpos = list(qpos_by_side.get("left", _zero_vector(6)))
        right_qpos = list(qpos_by_side.get("right", _zero_vector(6)))
        left_pose = self._side_pose("left") if "left" in self.scene.arm_specs else _zero_vector(7)
        right_pose = self._side_pose("right") if "right" in self.scene.arm_specs else _zero_vector(7)
        left_hand = list(hand_qpos_by_side.get("left", _zero_vector(6)))
        right_hand = list(hand_qpos_by_side.get("right", _zero_vector(6)))
        return [
            *left_qpos,
            *left_pose,
            *left_hand,
            *right_qpos,
            *right_pose,
            *right_hand,
        ]

    def _action_vector(self, hand_qpos_by_side: Mapping[str, Sequence[float]]) -> list[float]:
        left_pose = self._side_pose("left") if "left" in self.scene.arm_specs else _zero_vector(7)
        right_pose = self._side_pose("right") if "right" in self.scene.arm_specs else _zero_vector(7)
        left_hand = list(hand_qpos_by_side.get("left", _zero_vector(6)))
        right_hand = list(hand_qpos_by_side.get("right", _zero_vector(6)))
        return [*left_pose, *right_pose, *left_hand, *right_hand]

    def record_frame(
        self,
        *,
        sample: QposSample,
        qpos_result: Mapping[str, Sequence[float]],
        gripper_sample: GripperReplaySample | None = None,
        gripper_result: Mapping[str, float] | None = None,
        images: Mapping[str, np.ndarray],
        playback_time_sec: float,
        active_sides: Sequence[str],
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record to a closed sim replay dataset writer.")
        missing_images = [name for name in self.camera_names if name not in images]
        if missing_images:
            raise ValueError(f"Missing rendered sim camera frame(s): {missing_images}")

        frame_index = len(self.rows)
        timestamp = (
            float(sample.timestamp)
            if sample.timestamp is not None
            else self._fallback_timestamp_origin + float(playback_time_sec)
        )
        for camera_name in self.camera_names:
            self._write_video_frame(camera_name, images[camera_name])

        qpos_by_side = {
            side: [float(value) for value in values]
            for side, values in dict(qpos_result).items()
        }
        gripper_close_by_side = (
            {}
            if gripper_sample is None
            else {side: float(value) for side, value in gripper_sample.close_by_side.items()}
        )
        hand_qpos_by_side = self._hand_qpos_by_side(gripper_close_by_side)
        observation_state = self._observation_state(qpos_by_side, hand_qpos_by_side)
        action = self._action_vector(hand_qpos_by_side)
        row: dict[str, Any] = {
            "index": frame_index,
            "episode_index": self.episode_index,
            "frame_index": frame_index,
            "source_frame_index": int(sample.frame_index),
            "timestamp": timestamp,
            "relative_time_sec": float(sample.relative_time_sec),
            "playback_time_sec": float(playback_time_sec),
            "session_state": "sim_replay",
            "control_hz": float(self.config.nominal_fps),
            "arm_state_source": "mujoco_qpos_replay",
            "active_sides": list(active_sides),
            "camera_keys": list(self.camera_names),
            "fault_codes": [],
            "action": action,
            "action.left_arm.pose": action[0:7],
            "action.right_arm.pose": action[7:14],
            "action.left_hand.joint_target": hand_qpos_by_side["left"],
            "action.right_hand.joint_target": hand_qpos_by_side["right"],
            "observation.state": observation_state,
            "observation.last_action": action,
            "sample_metadata_json": json.dumps(
                _to_serializable(
                    {
                        "sim_replay": {
                            "source_frame_index": sample.frame_index,
                            "qpos_source_by_side": sample.source_by_side,
                            "gripper_source_by_name": {}
                            if gripper_sample is None
                            else gripper_sample.source_by_name,
                            "gripper_source_by_side": {}
                            if gripper_sample is None
                            else gripper_sample.source_by_side,
                            "camera_metadata": self.scene.camera_metadata(),
                        }
                    }
                ),
                ensure_ascii=True,
                sort_keys=True,
            ),
            "channels_json": "{}",
            "faults_json": "[]",
        }

        for side in ("left", "right"):
            qpos = qpos_by_side.get(side, _zero_vector(6))
            hand_qpos = list(hand_qpos_by_side.get(side, _zero_vector(6)))
            gripper_close = gripper_close_by_side.get(side)
            pose = self._side_pose(side) if side in self.scene.arm_specs else None
            row[f"observation.{side}_arm.qpos"] = qpos
            row[f"observation.{side}_arm.pose"] = pose
            row[f"observation.{side}_hand.qpos"] = hand_qpos
            row[f"observation.arm.{side}.current_qpos"] = qpos
            row[f"observation.arm.{side}.current_tcp_pose"] = pose
            row[f"observation.arm.{side}.actual_qpos"] = qpos
            row[f"observation.arm.{side}.actual_tcp_pose"] = pose
            row[f"observation.arm.{side}.actual_state_source"] = "mujoco"
            row[f"observation.arm.{side}.actual_state_timestamp"] = timestamp
            row[f"action.replay.{side}_arm_joint_target"] = qpos
            row[f"observation.hand.{side}.joint_pos"] = hand_qpos
            row[f"observation.hand.{side}.measured_joint_pos"] = hand_qpos
            row[f"observation.hand.{side}.commanded_joint_pos"] = hand_qpos
            row[f"action.replay.{side}_hand_joint_target"] = hand_qpos
            row[f"observation.gripper.{side}.state"] = gripper_close
            row[f"observation.gripper.{side}.measured"] = gripper_close
            row[f"observation.gripper.{side}.commanded"] = gripper_close
            row[f"action.gripper.{side}.target"] = gripper_close
            row[f"action.gripper.{side}.output_dim"] = int(self.config.gripper_command_dim)

        for camera_name in self.camera_names:
            prefix = f"observation.images.{camera_name}"
            row[f"{prefix}.source_timestamp"] = timestamp
            row[f"{prefix}.receive_timestamp"] = timestamp
            row[f"{prefix}.age_sec"] = 0.0
            row[f"{prefix}.source_frame_index"] = int(sample.frame_index)
            row[f"{prefix}.video_frame_index"] = frame_index
            row[f"{prefix}.path"] = self.video_relative_paths[camera_name]
            row[f"observation.sim_camera.{camera_name}.pose_world"] = pose_to_xyz_quat_xyzw(
                self.scene.camera_pose(camera_name)
            )

        self.rows.append(row)

    def close(self, *, summary: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()
        if not self.rows:
            return

        parquet_path = self.root_dir / self.parquet_relative_path
        _write_rows_to_parquet(
            self.rows,
            parquet_path,
            compression=str(self.config.parquet_compression),
        )
        start_timestamp = float(self.rows[0]["timestamp"])
        end_timestamp = float(self.rows[-1]["timestamp"])
        duration_sec = max(0.0, end_timestamp - start_timestamp)
        video_paths = {
            camera_name: relative_path
            for camera_name, relative_path in self.video_relative_paths.items()
            if (self.root_dir / relative_path).exists()
        }
        episode_record = {
            "episode_index": self.episode_index,
            "episode_id": self.episode_id,
            "frame_count": len(self.rows),
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "duration_sec": duration_sec,
            "parquet_path": self.parquet_relative_path,
            "video_paths": video_paths,
            "tactile_paths": {},
            "metadata": {
                "source_entrypoint": "sim.mujoco_real2sim.replay.runner",
                "arm_mode": "mujoco",
                "active_sides": sorted({side for row in self.rows for side in row.get("active_sides", [])}),
                "camera_keys": list(self.camera_names),
                "hz": float(self.config.nominal_fps),
                "sim_replay": self._export_metadata(),
            },
            "summary": {} if summary is None else dict(_to_serializable(dict(summary))),
        }
        stats_record = {
            "episode_index": self.episode_index,
            "frame_count": len(self.rows),
            "start_timestamp": start_timestamp,
            "end_timestamp": end_timestamp,
            "duration_sec": duration_sec,
            "average_dt_sec": duration_sec / max(1, len(self.rows) - 1),
            "average_hz": (len(self.rows) - 1) / duration_sec if duration_sec > 0.0 and len(self.rows) > 1 else 0.0,
            "mean_control_hz": float(self.config.nominal_fps),
            "nominal_fps": float(self.config.nominal_fps),
            "active_sides": episode_record["metadata"]["active_sides"],
            "arm_state_source": "mujoco_qpos_replay",
            "camera_keys": list(self.camera_names),
            "fault_count": 0,
        }
        _append_jsonl(self.root_dir / "meta" / "episodes.jsonl", episode_record)
        _append_jsonl(self.root_dir / "meta" / "episodes_stats.jsonl", stats_record)
        self._write_info()

    def __enter__(self) -> "SimReplayDatasetWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

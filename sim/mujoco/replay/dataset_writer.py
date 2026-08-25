from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.wrist_depth import (
    DepthRangeMaskConfig,
    depth_range_mask,
    mask_manifest_payload,
    pack_binary_mask,
)

from ..pointcloud import pointcloud_from_depth, pointcloud_manifest_payload
from ..scene import MujocoScene
from ..transforms import pose_to_xyz_quat_xyzw
from .trajectory import GripperReplaySample, QposSample

WRIST_DEPTH_STREAM_FORMAT = "binary_stream_v1"
WRIST_DEPTH_VIDEO_STREAM_FORMAT = "video_stream_v1"
WRIST_DEPTH_VIDEO_FORMAT = "dexteleop_wrist_depth_video_u8_linear_v1"


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


def _camera_video_channel(camera_name: str) -> str:
    return f"observation.images.{str(camera_name)}"


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


def _positive_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0.0:
        return None
    return float(number)


def _depth_video_manifest_payload(
    *,
    camera_key: str,
    relative_dir: str,
    stream_path: str,
    frame_count: int,
    height: int,
    width: int,
    records: list[Mapping[str, Any]],
    codec: str,
    fps: float,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, Any]:
    min_m = float(depth_min_m)
    max_m = float(depth_max_m)
    span_m = max_m - min_m
    quant_step = span_m / 254.0
    return {
        "format": WRIST_DEPTH_VIDEO_FORMAT,
        "camera_key": str(camera_key),
        "relative_dir": str(relative_dir),
        "stream": {
            "path": str(stream_path),
            "encoding": "mp4_bgr_gray_u8_linear_depth_range",
            "codec": str(codec),
            "dtype": "uint8",
            "shape": [int(height), int(width)],
            "frame_count": int(frame_count),
            "fps": float(fps),
            "channels": "BGR channels carry the same intended uint8 depth code before MP4 compression.",
        },
        "semantics": {
            "name": "aligned_depth_video_u8",
            "unit": "meters along the RGB camera optical z axis in the MuJoCo render camera frame",
            "visible_range_m": [min_m, max_m],
            "value_0": "invalid depth, background/far clip, or outside visible_range_m",
            "value_1_to_255": (
                "linear depth code; decode approximately as "
                f"depth_m = {min_m:.12g} + (value - 1) * {quant_step:.12g}"
            ),
            "quantization_step_m": quant_step,
            "lossy_note": (
                "MP4 compression can perturb decoded pixel values. Use the formula for the "
                "intended pre-compression code; this stream is compact and visualization-oriented, "
                "not a lossless replacement for z16 raw depth."
            ),
        },
        "frames": [dict(record) for record in records],
    }


@dataclass
class _WristDepthState:
    camera_name: str
    stream_kind: str
    relative_dir: str
    manifest_relative_path: str
    manifest_path: Path
    stream_relative_path: str
    stream_path: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    shape: tuple[int, int] | None = None
    valid_fraction_sum: float = 0.0
    foreground_fraction_sum: float = 0.0
    handle: Any | None = None
    video_writer: Any | None = None
    video_size: tuple[int, int] | None = None

    def close_handle(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None


@dataclass
class _PointCloudState:
    camera_name: str
    relative_dir: str
    manifest_relative_path: str
    manifest_path: Path
    records: list[dict[str, Any]] = field(default_factory=list)
    point_count_sum: int = 0
    valid_fraction_sum: float = 0.0


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
    write_wrist_depth_raw: bool = False
    write_wrist_depth_mask: bool = False
    wrist_depth_camera_names: tuple[str, ...] = ("left_wrist", "right_wrist")
    wrist_depth_scale_m_per_unit: float = 0.001
    wrist_depth_min_m: float = 0.05
    wrist_depth_max_m: float = 0.5
    wrist_depth_video_min_m: float = 0.05
    wrist_depth_video_max_m: float = 1.0
    wrist_depth_metadata_by_camera: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    rgb_postprocess: Mapping[str, Any] = field(default_factory=dict)
    write_pointcloud: bool = False
    pointcloud_camera_names: tuple[str, ...] = ("left_wrist", "right_wrist")
    pointcloud_depth_min_m: float = 0.05
    pointcloud_depth_max_m: float = 1.0
    pointcloud_pixel_stride: int = 1
    pointcloud_frame_stride: int = 1


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
                / _camera_video_channel(camera_name)
                / self.chunk_id
                / f"{self.episode_id}.mp4"
            )
            for camera_name in self.camera_names
        }
        self.wrist_depth_camera_names = tuple(
            name for name in self.config.wrist_depth_camera_names if name in self.camera_names
        )
        self.pointcloud_camera_names = tuple(
            name for name in self.config.pointcloud_camera_names if name in self.camera_names
        )
        self.rows: list[dict[str, Any]] = []
        self._video_writers: dict[str, Any] = {}
        self._video_sizes: dict[str, tuple[int, int]] = {}
        self._wrist_depth_states: dict[tuple[str, str], _WristDepthState] = {}
        self._pointcloud_states: dict[str, _PointCloudState] = {}
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
            "rgb_postprocess": _to_serializable(dict(self.config.rgb_postprocess)),
            "wrist_depth": {
                "raw": bool(self.config.write_wrist_depth_raw),
                "mask": bool(self.config.write_wrist_depth_mask),
                "camera_names": list(self.wrist_depth_camera_names),
                "depth_scale_m_per_unit": float(self.config.wrist_depth_scale_m_per_unit),
                "mask_config": DepthRangeMaskConfig(
                    depth_min_m=float(self.config.wrist_depth_min_m),
                    depth_max_m=float(self.config.wrist_depth_max_m),
                ).to_dict(),
                "video_encoding": {
                    "format": WRIST_DEPTH_VIDEO_FORMAT,
                    "visible_range_m": [
                        float(self.config.wrist_depth_video_min_m),
                        float(self.config.wrist_depth_video_max_m),
                    ],
                    "value_0": "invalid depth, background/far clip, or outside visible_range_m",
                    "value_1_to_255": "linear depth code before MP4 compression",
                },
                "source_metadata": _to_serializable(dict(self.config.wrist_depth_metadata_by_camera)),
            },
            "pointcloud": {
                "enabled": bool(self.config.write_pointcloud),
                "camera_names": list(self.pointcloud_camera_names),
                "format": "dexteleop_sim_pointcloud_npz_v1",
                "depth_range_m": [
                    float(self.config.pointcloud_depth_min_m),
                    float(self.config.pointcloud_depth_max_m),
                ],
                "pixel_stride": int(self.config.pointcloud_pixel_stride),
                "frame_stride": int(self.config.pointcloud_frame_stride),
                "source": "MuJoCo metric depth render at exported replay action frames",
            },
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

    @property
    def wrist_depth_enabled(self) -> bool:
        return bool(self.config.write_wrist_depth_raw or self.config.write_wrist_depth_mask)

    @property
    def pointcloud_enabled(self) -> bool:
        return bool(self.config.write_pointcloud)

    def _wrist_depth_stream_kinds(self) -> tuple[str, ...]:
        kinds: list[str] = []
        if self.config.write_wrist_depth_mask:
            kinds.append("mask")
        if self.config.write_wrist_depth_raw:
            kinds.append("raw")
        return tuple(kinds)

    def _wrist_depth_relative_dir(self, camera_name: str, stream_kind: str) -> str:
        return str(
            Path("wrist_depth")
            / str(stream_kind)
            / _camera_video_channel(camera_name)
            / self.chunk_id
            / self.episode_id
        )

    def _wrist_depth_stream_relative_path(self, camera_name: str, stream_kind: str) -> str:
        suffix = "bitpack.bin" if stream_kind == "mask" else "mp4"
        return str(Path(self._wrist_depth_relative_dir(camera_name, stream_kind)) / f"stream.{suffix}")

    def _wrist_depth_manifest_relative_path(self, camera_name: str, stream_kind: str) -> str:
        return str(Path(self._wrist_depth_relative_dir(camera_name, stream_kind)) / "manifest.json")

    def _wrist_depth_state(self, camera_name: str, stream_kind: str) -> _WristDepthState:
        key = (str(camera_name), str(stream_kind))
        if key in self._wrist_depth_states:
            return self._wrist_depth_states[key]
        relative_dir = self._wrist_depth_relative_dir(key[0], key[1])
        stream_relative_path = self._wrist_depth_stream_relative_path(key[0], key[1])
        manifest_relative_path = self._wrist_depth_manifest_relative_path(key[0], key[1])
        state = _WristDepthState(
            camera_name=key[0],
            stream_kind=key[1],
            relative_dir=relative_dir,
            manifest_relative_path=manifest_relative_path,
            manifest_path=self.root_dir / manifest_relative_path,
            stream_relative_path=stream_relative_path,
            stream_path=self.root_dir / stream_relative_path,
        )
        self._wrist_depth_states[key] = state
        return state

    def _pointcloud_relative_dir(self, camera_name: str) -> str:
        return str(
            Path("pointcloud")
            / _camera_video_channel(camera_name)
            / self.chunk_id
            / self.episode_id
        )

    def _pointcloud_manifest_relative_path(self, camera_name: str) -> str:
        return str(Path(self._pointcloud_relative_dir(camera_name)) / "manifest.json")

    def _pointcloud_frame_relative_path(self, camera_name: str, frame_index: int) -> str:
        return str(Path(self._pointcloud_relative_dir(camera_name)) / f"frame_{int(frame_index):06d}.npz")

    def _pointcloud_state(self, camera_name: str) -> _PointCloudState:
        key = str(camera_name)
        if key in self._pointcloud_states:
            return self._pointcloud_states[key]
        relative_dir = self._pointcloud_relative_dir(key)
        manifest_relative_path = self._pointcloud_manifest_relative_path(key)
        state = _PointCloudState(
            camera_name=key,
            relative_dir=relative_dir,
            manifest_relative_path=manifest_relative_path,
            manifest_path=self.root_dir / manifest_relative_path,
        )
        self._pointcloud_states[key] = state
        return state

    def _camera_fovy_deg(self, camera_name: str) -> float:
        for mount in self.scene.camera_mounts:
            if mount.name == camera_name:
                return float(mount.fovy_deg)
        metadata = self.scene.camera_metadata().get(str(camera_name), {})
        fovy = _positive_float_or_none(metadata.get("fovy_deg"))
        if fovy is None:
            raise KeyError(f"no fovy metadata for camera {camera_name!r}")
        return float(fovy)

    def _wrist_depth_metadata(self, camera_name: str, *, frame_index: int, timestamp: float) -> dict[str, Any]:
        raw = self.config.wrist_depth_metadata_by_camera.get(str(camera_name), {})
        metadata = dict(raw) if isinstance(raw, Mapping) else {}
        metadata.setdefault("depth_scale_m_per_unit", float(self.config.wrist_depth_scale_m_per_unit))
        metadata.setdefault("camera_timestamp_sec", float(timestamp))
        metadata.setdefault("sensor_timestamp_ms", float(timestamp) * 1000.0)
        metadata.setdefault("frame_number", int(frame_index))
        metadata.setdefault("align_depth_to_color", True)
        metadata.setdefault("sim_depth_source", "mujoco_metric_depth")
        return metadata

    def _depth_scale_for_camera(self, camera_name: str) -> float:
        metadata = self._wrist_depth_metadata(camera_name, frame_index=0, timestamp=0.0)
        scale = _positive_float_or_none(metadata.get("depth_scale_m_per_unit"))
        if scale is not None:
            return scale
        fallback = _positive_float_or_none(self.config.wrist_depth_scale_m_per_unit)
        return 0.001 if fallback is None else fallback

    def _depth_far_clip_m(self) -> float | None:
        try:
            extent = float(self.scene.model.stat.extent)
            zfar = float(self.scene.model.vis.map.zfar)
        except Exception:
            return None
        if not np.isfinite(extent) or not np.isfinite(zfar) or extent <= 0.0 or zfar <= 0.0:
            return None
        return float(extent * zfar)

    def _metric_depth_to_z16(self, depth_m: np.ndarray, *, camera_name: str) -> np.ndarray:
        scale = self._depth_scale_for_camera(camera_name)
        depth = np.asarray(depth_m, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.0)
        far_clip = self._depth_far_clip_m()
        if far_clip is not None:
            valid &= depth < np.float32(float(far_clip) * 0.999)
        values = np.zeros(depth.shape, dtype=np.float32)
        values[valid] = depth[valid] / np.float32(scale)
        return np.clip(np.rint(values), 0, np.iinfo(np.uint16).max).astype(np.uint16)

    def _depth_z16_to_video_u8(self, depth_z16: np.ndarray, *, camera_name: str) -> tuple[np.ndarray, float]:
        scale = self._depth_scale_for_camera(camera_name)
        depth = np.asarray(depth_z16, dtype=np.uint16)
        depth_m = depth.astype(np.float32, copy=False) * np.float32(scale)
        min_m = float(self.config.wrist_depth_video_min_m)
        max_m = float(self.config.wrist_depth_video_max_m)
        span_m = max(max_m - min_m, 1e-9)
        valid = (depth > 0) & np.isfinite(depth_m) & (depth_m >= min_m) & (depth_m <= max_m)
        code = np.zeros(depth.shape, dtype=np.uint8)
        values = 1.0 + (depth_m[valid] - np.float32(min_m)) * np.float32(254.0 / span_m)
        code[valid] = np.clip(np.rint(values), 1, 255).astype(np.uint8)
        total = int(depth.size)
        valid_fraction = 0.0 if total <= 0 else float(np.count_nonzero(valid)) / float(total)
        return code, valid_fraction

    def _write_depth_video_code_frame(self, state: _WristDepthState, code: np.ndarray) -> None:
        cv2 = _require_cv2()
        frame = cv2.cvtColor(np.asarray(code, dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        height, width = frame.shape[:2]
        if state.video_writer is None:
            state.stream_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(state.stream_path),
                cv2.VideoWriter_fourcc(*str(self.config.video_codec)),
                float(self.config.nominal_fps),
                (int(width), int(height)),
            )
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open sim replay wrist depth video writer: {state.stream_path}")
            state.video_writer = writer
            state.video_size = (int(width), int(height))
        elif state.video_size != (int(width), int(height)):
            raise ValueError(
                f"Wrist depth video for camera {state.camera_name!r} changed size "
                f"from {state.video_size} to {(int(width), int(height))}."
            )
        assert state.video_writer is not None
        state.video_writer.write(frame)

    def _write_wrist_depth_frame(
        self,
        *,
        row: dict[str, Any],
        camera_name: str,
        stream_kind: str,
        depth_z16: np.ndarray,
        frame_index: int,
        timestamp: float,
        source_frame_index: int,
    ) -> None:
        state = self._wrist_depth_state(camera_name, stream_kind)
        depth = np.ascontiguousarray(np.asarray(depth_z16, dtype=np.uint16))
        if depth.ndim != 2:
            raise ValueError(f"Wrist depth for camera {camera_name!r} must be 2D, got {depth.shape!r}.")
        height, width = int(depth.shape[0]), int(depth.shape[1])
        shape = (height, width)
        if state.shape is None:
            state.shape = shape
        elif state.shape != shape:
            raise ValueError(
                f"Wrist depth for camera {camera_name!r} changed shape from {state.shape} to {shape}."
            )

        depth_metadata = self._wrist_depth_metadata(
            camera_name,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        record: dict[str, Any] = {
            "sample_frame_index": int(frame_index),
            "depth_frame_index": int(len(state.records)),
            "source_timestamp": float(timestamp),
            "receive_timestamp": float(timestamp),
            "source_frame_index": int(source_frame_index),
            "depth_camera_timestamp_sec": depth_metadata.get("camera_timestamp_sec"),
            "depth_sensor_timestamp_ms": depth_metadata.get("sensor_timestamp_ms"),
            "depth_frame_number": depth_metadata.get("frame_number"),
            "depth_scale_m_per_unit": depth_metadata.get("depth_scale_m_per_unit"),
            "shape": [height, width],
        }
        if stream_kind == "mask":
            if state.handle is None:
                state.stream_path.parent.mkdir(parents=True, exist_ok=True)
                state.handle = state.stream_path.open("ab")
            scale = _positive_float_or_none(depth_metadata.get("depth_scale_m_per_unit"))
            if scale is None:
                raise ValueError(f"Wrist depth mask for camera {camera_name!r} is missing depth scale.")
            result = depth_range_mask(
                depth,
                depth_scale_m_per_unit=scale,
                config=DepthRangeMaskConfig(
                    depth_min_m=float(self.config.wrist_depth_min_m),
                    depth_max_m=float(self.config.wrist_depth_max_m),
                ),
            )
            assert state.handle is not None
            state.handle.write(pack_binary_mask(result.mask))
            record.update(result.stats_dict())
            state.valid_fraction_sum += float(result.valid_fraction)
            state.foreground_fraction_sum += float(result.foreground_fraction)
        elif stream_kind == "raw":
            code, valid_fraction = self._depth_z16_to_video_u8(depth, camera_name=camera_name)
            self._write_depth_video_code_frame(state, code)
            record["valid_fraction"] = valid_fraction
            record["depth_video_format"] = WRIST_DEPTH_VIDEO_FORMAT
            record["depth_video_min_m"] = float(self.config.wrist_depth_video_min_m)
            record["depth_video_max_m"] = float(self.config.wrist_depth_video_max_m)
            record["depth_video_quantization_step_m"] = (
                float(self.config.wrist_depth_video_max_m) - float(self.config.wrist_depth_video_min_m)
            ) / 254.0
            state.valid_fraction_sum += float(record["valid_fraction"])
        else:
            raise ValueError(f"Unsupported wrist depth stream kind {stream_kind!r}.")

        state.records.append(record)
        prefix = f"{_camera_video_channel(camera_name)}.wrist_depth_{stream_kind}"
        row[f"{prefix}_storage_format"] = (
            WRIST_DEPTH_VIDEO_STREAM_FORMAT if stream_kind == "raw" else WRIST_DEPTH_STREAM_FORMAT
        )
        row[f"{prefix}_path"] = state.manifest_relative_path
        row[f"{prefix}_frame_index"] = int(record["depth_frame_index"])
        row[f"{prefix}_source_timestamp"] = record["source_timestamp"]
        row[f"{prefix}_receive_timestamp"] = record["receive_timestamp"]
        row[f"{prefix}_source_frame_index"] = record["source_frame_index"]
        row[f"{prefix}_depth_camera_timestamp_sec"] = record["depth_camera_timestamp_sec"]
        row[f"{prefix}_depth_sensor_timestamp_ms"] = record["depth_sensor_timestamp_ms"]
        row[f"{prefix}_depth_frame_number"] = record["depth_frame_number"]
        row[f"{prefix}_valid_fraction"] = record.get("valid_fraction")
        if stream_kind == "mask":
            row[f"{prefix}_foreground_fraction"] = record.get("foreground_fraction")

    def _write_pointcloud_frame(
        self,
        *,
        row: dict[str, Any],
        camera_name: str,
        depth_m: np.ndarray,
        frame_index: int,
        timestamp: float,
        source_frame_index: int,
    ) -> None:
        frame_stride = max(1, int(self.config.pointcloud_frame_stride))
        prefix = f"{_camera_video_channel(camera_name)}.pointcloud"
        if int(frame_index) % frame_stride != 0:
            row[f"{prefix}_path"] = None
            row[f"{prefix}_frame_index"] = None
            row[f"{prefix}_point_count"] = 0
            row[f"{prefix}_valid_fraction"] = 0.0
            return

        state = self._pointcloud_state(camera_name)
        depth = np.asarray(depth_m, dtype=np.float32)
        if depth.ndim != 2:
            raise ValueError(f"Pointcloud depth for camera {camera_name!r} must be 2D, got {depth.shape!r}.")
        height, width = int(depth.shape[0]), int(depth.shape[1])
        pointcloud = pointcloud_from_depth(
            depth,
            fovy_deg=self._camera_fovy_deg(camera_name),
            T_world_camera=self.scene.camera_link_pose(camera_name),
            depth_min_m=float(self.config.pointcloud_depth_min_m),
            depth_max_m=float(self.config.pointcloud_depth_max_m),
            pixel_stride=int(self.config.pointcloud_pixel_stride),
        )
        relative_path = self._pointcloud_frame_relative_path(camera_name, frame_index)
        path = self.root_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points_camera=pointcloud.points_camera,
            points_world=pointcloud.points_world,
            uv=pointcloud.uv,
            depth_m=pointcloud.depth_m,
            valid_mask=pointcloud.valid_mask,
            K_render=pointcloud.intrinsics.camera_matrix().astype(np.float32),
            T_world_camera=pointcloud.T_world_camera.astype(np.float32),
            frame_index=np.asarray([int(frame_index)], dtype=np.int32),
            source_frame_index=np.asarray([int(source_frame_index)], dtype=np.int32),
            timestamp=np.asarray([float(timestamp)], dtype=np.float64),
        )
        valid_fraction = 0.0 if height <= 0 or width <= 0 else float(pointcloud.point_count) / float(height * width)
        record = {
            "sample_frame_index": int(frame_index),
            "pointcloud_frame_index": int(len(state.records)),
            "source_timestamp": float(timestamp),
            "receive_timestamp": float(timestamp),
            "source_frame_index": int(source_frame_index),
            "path": relative_path,
            "shape": [height, width],
            "point_count": int(pointcloud.point_count),
            "valid_fraction": float(valid_fraction),
            "depth_min_m": float(self.config.pointcloud_depth_min_m),
            "depth_max_m": float(self.config.pointcloud_depth_max_m),
            "pixel_stride": int(self.config.pointcloud_pixel_stride),
            "intrinsics": pointcloud.intrinsics.to_dict(),
            "camera_pose_world_xyz_quat_xyzw": pose_to_xyz_quat_xyzw(self.scene.camera_link_pose(camera_name)),
        }
        state.records.append(record)
        state.point_count_sum += int(pointcloud.point_count)
        state.valid_fraction_sum += float(valid_fraction)

        row[f"{prefix}_storage_format"] = "npz_per_frame"
        row[f"{prefix}_path"] = relative_path
        row[f"{prefix}_manifest_path"] = state.manifest_relative_path
        row[f"{prefix}_frame_index"] = int(record["pointcloud_frame_index"])
        row[f"{prefix}_source_timestamp"] = record["source_timestamp"]
        row[f"{prefix}_receive_timestamp"] = record["receive_timestamp"]
        row[f"{prefix}_source_frame_index"] = record["source_frame_index"]
        row[f"{prefix}_point_count"] = int(record["point_count"])
        row[f"{prefix}_valid_fraction"] = float(record["valid_fraction"])

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
        depth_images: Mapping[str, np.ndarray] | None = None,
        playback_time_sec: float,
        active_sides: Sequence[str],
        frame_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot record to a closed sim replay dataset writer.")
        missing_images = [name for name in self.camera_names if name not in images]
        if missing_images:
            raise ValueError(f"Missing rendered sim camera frame(s): {missing_images}")
        depth_images = {} if depth_images is None else depth_images
        if self.wrist_depth_enabled:
            missing_depth = [name for name in self.wrist_depth_camera_names if name not in depth_images]
            if missing_depth:
                raise ValueError(f"Missing rendered sim depth frame(s): {missing_depth}")
        if self.pointcloud_enabled:
            missing_depth = [name for name in self.pointcloud_camera_names if name not in depth_images]
            if missing_depth:
                raise ValueError(f"Missing rendered sim depth frame(s) for pointcloud: {missing_depth}")

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
                            "frame_metadata": {}
                            if frame_metadata is None
                            else _to_serializable(dict(frame_metadata)),
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

        if self.wrist_depth_enabled:
            for camera_name in self.wrist_depth_camera_names:
                depth_z16 = self._metric_depth_to_z16(depth_images[camera_name], camera_name=camera_name)
                for stream_kind in self._wrist_depth_stream_kinds():
                    self._write_wrist_depth_frame(
                        row=row,
                        camera_name=camera_name,
                        stream_kind=stream_kind,
                        depth_z16=depth_z16,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        source_frame_index=int(sample.frame_index),
                    )

        if self.pointcloud_enabled:
            for camera_name in self.pointcloud_camera_names:
                self._write_pointcloud_frame(
                    row=row,
                    camera_name=camera_name,
                    depth_m=depth_images[camera_name],
                    frame_index=frame_index,
                    timestamp=timestamp,
                    source_frame_index=int(sample.frame_index),
                )

        self.rows.append(row)

    def _close_wrist_depth_state(self, state: _WristDepthState) -> str | None:
        state.close_handle()
        if not state.records or state.shape is None:
            return None
        height, width = state.shape
        if state.stream_kind == "mask":
            payload = mask_manifest_payload(
                camera_key=state.camera_name,
                relative_dir=state.relative_dir,
                stream_path=state.stream_relative_path,
                frame_count=len(state.records),
                height=height,
                width=width,
                records=state.records,
                config=DepthRangeMaskConfig(
                    depth_min_m=float(self.config.wrist_depth_min_m),
                    depth_max_m=float(self.config.wrist_depth_max_m),
                ),
            )
        elif state.stream_kind == "raw":
            payload = _depth_video_manifest_payload(
                camera_key=state.camera_name,
                relative_dir=state.relative_dir,
                stream_path=state.stream_relative_path,
                frame_count=len(state.records),
                height=height,
                width=width,
                records=state.records,
                codec=str(self.config.video_codec),
                fps=float(self.config.nominal_fps),
                depth_min_m=float(self.config.wrist_depth_video_min_m),
                depth_max_m=float(self.config.wrist_depth_video_max_m),
            )
        else:
            raise ValueError(f"Unsupported wrist depth stream kind {state.stream_kind!r}.")
        _write_json(state.manifest_path, payload)
        return state.manifest_relative_path

    def _close_pointcloud_state(self, state: _PointCloudState) -> str | None:
        if not state.records:
            return None
        payload = pointcloud_manifest_payload(
            camera_key=state.camera_name,
            relative_dir=state.relative_dir,
            frame_count=len(state.records),
            records=state.records,
            depth_min_m=float(self.config.pointcloud_depth_min_m),
            depth_max_m=float(self.config.pointcloud_depth_max_m),
            pixel_stride=int(self.config.pointcloud_pixel_stride),
        )
        _write_json(state.manifest_path, payload)
        return state.manifest_relative_path

    def close(self, *, summary: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        for writer in self._video_writers.values():
            writer.release()
        self._video_writers.clear()
        wrist_depth_mask_paths: dict[str, str] = {}
        wrist_depth_raw_paths: dict[str, str] = {}
        wrist_depth_frame_stats: dict[str, dict[str, Any]] = {}
        pointcloud_paths: dict[str, str] = {}
        pointcloud_frame_stats: dict[str, dict[str, Any]] = {}
        for state in self._wrist_depth_states.values():
            manifest_path = self._close_wrist_depth_state(state)
            if manifest_path is not None:
                if state.stream_kind == "mask":
                    wrist_depth_mask_paths[state.camera_name] = manifest_path
                elif state.stream_kind == "raw":
                    wrist_depth_raw_paths[state.camera_name] = manifest_path
            key = f"{state.camera_name}.{state.stream_kind}"
            frame_count = int(len(state.records))
            wrist_depth_frame_stats[key] = {
                "expected": len(self.rows),
                "written": frame_count,
                "missing": max(0, len(self.rows) - frame_count),
                "path": manifest_path,
                "valid_fraction_mean": (
                    0.0 if frame_count <= 0 else float(state.valid_fraction_sum) / float(frame_count)
                ),
            }
            if state.stream_kind == "mask":
                wrist_depth_frame_stats[key]["foreground_fraction_mean"] = (
                    0.0 if frame_count <= 0 else float(state.foreground_fraction_sum) / float(frame_count)
                )
        for state in self._pointcloud_states.values():
            manifest_path = self._close_pointcloud_state(state)
            if manifest_path is not None:
                pointcloud_paths[state.camera_name] = manifest_path
            frame_count = int(len(state.records))
            pointcloud_frame_stats[state.camera_name] = {
                "expected": len(self.rows),
                "written": frame_count,
                "missing": max(0, len(self.rows) - frame_count),
                "path": manifest_path,
                "point_count_mean": (
                    0.0 if frame_count <= 0 else float(state.point_count_sum) / float(frame_count)
                ),
                "valid_fraction_mean": (
                    0.0 if frame_count <= 0 else float(state.valid_fraction_sum) / float(frame_count)
                ),
            }
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
            "wrist_depth_mask_paths": wrist_depth_mask_paths,
            "wrist_depth_raw_paths": wrist_depth_raw_paths,
            "pointcloud_paths": pointcloud_paths,
            "metadata": {
                "source_entrypoint": "sim.mujoco.replay.runner",
                "arm_mode": "mujoco",
                "active_sides": sorted({side for row in self.rows for side in row.get("active_sides", [])}),
                "camera_keys": list(self.camera_names),
                "hz": float(self.config.nominal_fps),
                "sim_replay": self._export_metadata(),
            },
            "summary": {
                **({} if summary is None else dict(_to_serializable(dict(summary)))),
                **({"wrist_depth_frame_stats": wrist_depth_frame_stats} if wrist_depth_frame_stats else {}),
                **({"pointcloud_frame_stats": pointcloud_frame_stats} if pointcloud_frame_stats else {}),
            },
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

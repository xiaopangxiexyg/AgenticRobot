from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..async_writer import WriteJob
from ..common import (
    EffectorCommand,
    ObservationChannel,
    nan_vector,
    to_serializable,
    vector_list,
    vector_or_none,
)
from ..video_frame import NearestFrameBuffer, VideoFrame

ARM_DOF = 7
HAND_DOF = 6
HAND_VECTOR_ORDER = ("right", "left")
ACTION_DIM = ARM_DOF + HAND_DOF * 2
OBSERVATION_STATE_DIM = ARM_DOF + HAND_DOF * 2

ROW_JOB_KIND = "hand_only.row"
VIDEO_FRAME_JOB_KIND = "hand_only.video_frame"
FINALIZE_EPISODE_JOB_KIND = "hand_only.finalize_episode"


def _require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Video export requires opencv-python.") from exc
    return cv2


def episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def chunk_id_from_episode_index(index: int, *, chunk_size: int) -> str:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    return f"chunk-{int(index) // int(chunk_size):03d}"


def camera_video_channel(camera_key: str) -> str:
    return f"observation.images.{str(camera_key)}"


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
class CameraFrameMatch:
    camera_key: str
    frame: VideoFrame | None
    selected_timestamp: float | None
    age_sec: float | None


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
    arm_state: dict[str, Any] = field(default_factory=dict)
    hand_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: list[str] = field(default_factory=list)
    camera_frames: dict[str, CameraFrameMatch] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class HandSampleBuilder:
    def __init__(
        self,
        *,
        active_sides: tuple[str, ...],
        episode_index: int = 0,
        sample_index_start: int = 0,
        camera_buffers: Mapping[str, NearestFrameBuffer] | None = None,
    ):
        self.active_sides = tuple(str(side) for side in active_sides)
        self.camera_buffers = {
            str(camera_key): buffer for camera_key, buffer in dict(camera_buffers or {}).items()
        }
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

    def _arm_command_or_nan(
        self,
        commands: Mapping[str, EffectorCommand],
        arm_state: Mapping[str, Any] | None,
    ) -> np.ndarray:
        state = dict(arm_state or {})
        actual_sent = vector_or_none(state.get("last_sent_command"), length=ARM_DOF)
        if actual_sent is not None:
            return actual_sent
        command = commands.get("arm")
        if command is None:
            return nan_vector(ARM_DOF)
        value = vector_or_none(command.value, length=ARM_DOF)
        return value if value is not None else nan_vector(ARM_DOF)

    def _arm_state_or_nan(
        self,
        arm_state: Mapping[str, Any] | None,
    ) -> np.ndarray:
        state = dict(arm_state or {})
        value = vector_or_none(state.get("qpos"), length=ARM_DOF)
        return value if value is not None else nan_vector(ARM_DOF)

    def _build_action_vector(
        self,
        commands: Mapping[str, EffectorCommand],
        arm_state: Mapping[str, Any] | None,
    ) -> np.ndarray:
        return np.concatenate(
            (
                self._arm_command_or_nan(commands, arm_state),
                *(self._hand_command_or_nan(commands, side) for side in HAND_VECTOR_ORDER),
            ),
            dtype=np.float32,
        )

    def _build_observation_state(
        self,
        arm_state: Mapping[str, Any] | None,
        hand_states: Mapping[str, Mapping[str, Any] | None],
    ) -> np.ndarray:
        return np.concatenate(
            (
                self._arm_state_or_nan(arm_state),
                *(self._hand_state_or_nan(hand_states, side) for side in HAND_VECTOR_ORDER),
            ),
            dtype=np.float32,
        )

    def _camera_matches(
        self,
        *,
        timestamp: float,
        camera_frames: Mapping[str, VideoFrame | None] | None,
    ) -> dict[str, CameraFrameMatch]:
        if camera_frames is None:
            camera_frames = {
                camera_key: buffer.get_nearest(timestamp)
                for camera_key, buffer in self.camera_buffers.items()
            }
        matches: dict[str, CameraFrameMatch] = {}
        for camera_key, frame in camera_frames.items():
            key = str(camera_key)
            if frame is None:
                matches[key] = CameraFrameMatch(
                    camera_key=key,
                    frame=None,
                    selected_timestamp=None,
                    age_sec=None,
                )
                continue
            matches[key] = CameraFrameMatch(
                camera_key=key,
                frame=frame,
                selected_timestamp=float(frame.source_timestamp),
                age_sec=float(timestamp - float(frame.source_timestamp)),
            )
        return matches

    def build(
        self,
        *,
        timestamp: float,
        session_state: str,
        control_hz: float,
        channels: Mapping[str, ObservationChannel],
        commands: Mapping[str, EffectorCommand],
        arm_state: Mapping[str, Any] | None,
        hand_states: Mapping[str, Mapping[str, Any] | None],
        camera_frames: Mapping[str, VideoFrame | None] | None = None,
        faults: list[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> HandSample:
        copied_commands = {
            name: command.copy() for name, command in commands.items()
        }
        action = self._build_action_vector(copied_commands, arm_state)
        observation_state = self._build_observation_state(arm_state, hand_states)
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
            arm_state={} if arm_state is None else dict(to_serializable(dict(arm_state))),
            hand_states={
                str(side): _compact_hand_state_for_sample(dict(state or {}))
                for side, state in hand_states.items()
            },
            faults=[] if faults is None else [str(fault) for fault in faults],
            camera_frames=self._camera_matches(
                timestamp=float(timestamp),
                camera_frames=camera_frames,
            ),
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


def _compact_hand_state_for_sample(hand_state: Mapping[str, Any]) -> dict[str, Any]:
    compact = dict(hand_state)
    if _uses_dmtac_gripper_tactile(compact):
        _apply_dmtac_gripper_tactile_save_pooling(compact)
    tactile_dict = compact.get("tactile_dict")
    if isinstance(tactile_dict, Mapping):
        compact["tactile_dict"] = {
            str(name): _compact_tactile_dict_value(value)
            for name, value in dict(tactile_dict).items()
        }
    return compact


def _uses_dmtac_gripper_tactile(hand_state: Mapping[str, Any]) -> bool:
    return str(hand_state.get("backend_type") or "") == "dmtac_gripper"


def _mean_pool_spatial_2x2(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim < 2:
        return arr.astype(np.float32, copy=True)
    pooled_h = int(arr.shape[0]) // 2
    pooled_w = int(arr.shape[1]) // 2
    if pooled_h <= 0 or pooled_w <= 0:
        return arr.astype(np.float32, copy=True)
    trimmed = arr[: pooled_h * 2, : pooled_w * 2, ...]
    pooled_shape = (pooled_h, 2, pooled_w, 2, *arr.shape[2:])
    pooled = trimmed.reshape(pooled_shape).mean(axis=(1, 3), dtype=np.float32)
    return pooled.astype(np.float32, copy=False)


def _apply_dmtac_gripper_tactile_save_pooling(hand_state: dict[str, Any]) -> None:
    tactile_dict = hand_state.get("tactile_dict")
    if not isinstance(tactile_dict, Mapping) or not tactile_dict:
        return
    pooled_dict: dict[str, np.ndarray] = {
        str(name): _mean_pool_spatial_2x2(value)
        for name, value in dict(tactile_dict).items()
    }
    pooled_vectors = [value.reshape(-1) for value in pooled_dict.values()]
    if pooled_vectors:
        hand_state["tactile"] = np.concatenate(pooled_vectors).astype(
            np.float32,
            copy=False,
        )
    hand_state["tactile_dict"] = pooled_dict


def _compact_tactile_dict_value(value: Any) -> Any:
    arr = np.asarray(value)
    if arr.ndim >= 3 or arr.size > 5000:
        return {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    return to_serializable(value)


def hand_sample_to_row(
    sample: HandSample,
    *,
    camera_video_paths: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    action_split = _split_action(sample.action)
    observation_split = _split_observation_state(sample.observation_state)
    arm_state = sample.arm_state or {}
    command_metadata_by_effector = {
        name: {} if command.metadata is None else dict(command.metadata)
        for name, command in sample.commands.items()
    }
    arm_command_metadata = command_metadata_by_effector.get("arm", {})
    arm_side = str(
        arm_state.get("arm_side")
        or arm_command_metadata.get("arm_side")
        or sample.metadata.get("arm_side")
        or "right"
    )
    arm_enabled = bool(
        arm_state.get("arm_enabled")
        or sample.metadata.get("arm_enabled")
        or ("arm" in sample.commands)
    )
    arm_state_source = str(
        arm_state.get("state_source")
        or ("fr3_joint_control_client.cached" if arm_enabled else "disabled")
    )
    clip_mask = arm_command_metadata.get("clip_mask")
    if clip_mask is not None:
        clip_mask = [bool(value) for value in list(clip_mask)]
    row: dict[str, Any] = {
        "index": int(sample.index),
        "episode_index": int(sample.episode_index),
        "frame_index": int(sample.frame_index),
        "timestamp": float(sample.timestamp),
        "session_state": sample.session_state,
        "control_hz": float(sample.control_hz),
        "active_sides": list(sample.active_sides),
        "arm_enabled": bool(arm_enabled),
        "arm_side": arm_side,
        "arm_state_source": arm_state_source,
        "camera_keys": sorted(sample.camera_frames.keys()),
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
    row["observation.arm.dq"] = vector_list(arm_state.get("dq"), length=ARM_DOF)
    row["observation.arm.controller_timestamp"] = arm_state.get("controller_timestamp")
    row["observation.arm.receive_timestamp"] = arm_state.get("state_receive_timestamp")
    row["observation.arm.cache_age_sec"] = arm_state.get("state_age_sec")
    row["observation.arm.server_age_sec"] = arm_state.get("server_age_s")
    row["observation.arm.state_error"] = arm_state.get("state_error")
    row["observation.arm.raw_state_json"] = json.dumps(
        to_serializable(arm_state.get("raw_state")),
        ensure_ascii=True,
        sort_keys=True,
    )
    row["action.replay.arm_joint_target"] = vector_list(
        arm_state.get("last_sent_command"),
        length=ARM_DOF,
    )
    row["action.arm.last_sent_timestamp"] = arm_state.get("last_sent_command_timestamp")
    row["observation.arm.startup_alignment_enabled"] = arm_state.get(
        "startup_alignment_enabled"
    )
    row["observation.arm.startup_alignment_active"] = arm_state.get(
        "startup_alignment_active"
    )
    row["observation.arm.startup_alignment_complete"] = arm_state.get(
        "startup_alignment_complete"
    )
    row["observation.arm.startup_alignment_progress"] = arm_state.get(
        "startup_alignment_progress"
    )
    row["observation.arm.startup_alignment_duration_s"] = arm_state.get(
        "startup_alignment_duration_s"
    )
    row["observation.arm.startup_alignment_start_max_delta_rad"] = arm_state.get(
        "startup_alignment_start_max_delta_rad"
    )
    row["observation.arm.startup_alignment_start_qpos"] = vector_list(
        arm_state.get("startup_alignment_start_qpos"),
        length=ARM_DOF,
    )
    row["observation.arm.startup_alignment_initial_target"] = vector_list(
        arm_state.get("startup_alignment_initial_target"),
        length=ARM_DOF,
    )
    row["observation.arm.startup_alignment_latest_target"] = vector_list(
        arm_state.get("startup_alignment_latest_target"),
        length=ARM_DOF,
    )
    row["teleop.action_source.arm_exo_qpos"] = vector_list(
        arm_command_metadata.get("source_qpos"),
        length=ARM_DOF,
    )
    row["teleop.action_source.arm_source_timestamp"] = arm_command_metadata.get("source_timestamp")
    row["teleop.action_source.arm_preclip_joint_target"] = vector_list(
        arm_command_metadata.get("preclip_joint_target"),
        length=ARM_DOF,
    )
    row["teleop.action_source.arm_clip_mask"] = clip_mask
    row["teleop.action_source.arm_clipped"] = arm_command_metadata.get("clipped")
    row["teleop.action_source.arm_mapping"] = arm_command_metadata.get("mapping")
    row["teleop.action_source.arm_shrink_deg"] = arm_command_metadata.get("shrink_deg")
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
        row[f"observation.hand.{side}.tactile_dict_json"] = json.dumps(
            to_serializable(hand_state.get("tactile_dict")),
            ensure_ascii=True,
            sort_keys=True,
        )
        row[f"action.replay.{side}_hand_joint_target"] = vector_list(
            hand_state.get("last_sent_command"),
            length=HAND_DOF,
        )
        row[f"teleop.action_source.{side}_hand_q6"] = vector_list(
            command_metadata.get("source_q6"),
            length=HAND_DOF,
        )
    camera_expectations: dict[str, bool] = {}
    for camera_key, match in sample.camera_frames.items():
        prefix = camera_video_channel(camera_key)
        frame = match.frame
        camera_expectations[str(camera_key)] = frame is not None
        row[f"{prefix}.source_timestamp"] = match.selected_timestamp
        row[f"{prefix}.receive_timestamp"] = None if frame is None else frame.receive_timestamp
        row[f"{prefix}.age_sec"] = match.age_sec
        row[f"{prefix}.source_frame_index"] = None if frame is None else frame.frame_index
        row[f"{prefix}.video_frame_index"] = None
        if camera_video_paths is not None and camera_key in camera_video_paths:
            row[f"{prefix}.path"] = str(camera_video_paths[camera_key])
    return row, camera_expectations


@dataclass
class WriterConfig:
    root_dir: str | Path
    chunk_size: int = 1000
    nominal_fps: float = 30.0
    parquet_compression: str = "zstd"
    video_codec: str = "mp4v"
    task_name: str = ""
    task_id: str = "hand_only"
    task_description: str = ""
    dataset_name: str = ""
    format_name: str = "hand_only_lerobot_v2.1_style"
    write_parquet: bool = True
    write_video: bool = True
    hand_tactile_dim: int = 1062

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.task_name = str(self.task_name).strip()
        self.task_id = str(self.task_id).strip()
        self.task_description = str(self.task_description).strip()
        self.dataset_name = str(self.dataset_name).strip() or self.root_dir.name
        if int(self.chunk_size) <= 0:
            raise ValueError("chunk_size must be positive.")
        if float(self.nominal_fps) <= 0.0:
            raise ValueError("nominal_fps must be positive.")
        if int(self.hand_tactile_dim) < 0:
            raise ValueError("hand_tactile_dim must be non-negative.")


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
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    control_hz_values: list[float] = field(default_factory=list)
    video_states: dict[str, _VideoState] = field(default_factory=dict)


class HandOnlyLeRobotWriter:
    def __init__(self, config: WriterConfig):
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
        task_name = self.config.task_name or self.config.task_description or self.config.task_id
        if not task_name:
            return
        existing_rows = _read_jsonl(self.tasks_path)
        for row in existing_rows:
            if row.get("task") == task_name:
                return
            if self.config.task_id and row.get("task_id") == self.config.task_id:
                return
        next_task_index = max(
            [int(row.get("task_index", -1)) for row in existing_rows] + [-1]
        ) + 1
        task_record = {
            "task_index": int(next_task_index),
            "task": task_name,
        }
        if self.config.task_id:
            task_record["task_id"] = self.config.task_id
        if self.config.task_description:
            task_record["description"] = self.config.task_description
        _append_jsonl(self.tasks_path, task_record)

    def _task_manifest_rows(self) -> list[dict[str, Any]]:
        rows = _read_jsonl(self.tasks_path)
        if rows:
            return rows
        task_name = self.config.task_name or self.config.task_description or self.config.task_id
        if not task_name:
            return []
        task_record = {
            "task_index": 0,
            "task": task_name,
        }
        if self.config.task_id:
            task_record["task_id"] = self.config.task_id
        if self.config.task_description:
            task_record["description"] = self.config.task_description
        return [task_record]

    def _task_names(self) -> list[str]:
        tasks: list[str] = []
        for row in self._task_manifest_rows():
            task = str(row.get("task", "")).strip()
            if task and task not in tasks:
                tasks.append(task)
        return tasks

    def _primary_task_index(self) -> int | None:
        task_name = self.config.task_name or self.config.task_description or self.config.task_id
        if not task_name:
            return None
        for row in self._task_manifest_rows():
            if row.get("task") == task_name:
                try:
                    return int(row.get("task_index"))
                except (TypeError, ValueError):
                    return None
        return None

    def _primary_task_name(self) -> str:
        return self.config.task_name or self.config.task_description or self.config.task_id

    def _write_info_locked(self) -> None:
        task_names = self._task_names()
        primary_task_index = self._primary_task_index()
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
                "video_codec": str(self.config.video_codec),
                "task_name": self._primary_task_name(),
                "task_id": self.config.task_id,
                "task_description": self.config.task_description,
                "task_index": primary_task_index,
                "tasks": task_names,
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
                    "action.replay.arm_joint_target": {"shape": [ARM_DOF], "dtype": "float32"},
                    "observation.arm.dq": {"shape": [ARM_DOF], "dtype": "float32"},
                    "teleop.action_source.arm_exo_qpos": {"shape": [ARM_DOF], "dtype": "float32"},
                    "teleop.action_source.arm_preclip_joint_target": {
                        "shape": [ARM_DOF],
                        "dtype": "float32",
                    },
                    "teleop.action_source.arm_clip_mask": {"shape": [ARM_DOF], "dtype": "bool"},
                    "action.right_hand.joint_target": {"shape": [HAND_DOF], "dtype": "float32"},
                    "action.left_hand.joint_target": {"shape": [HAND_DOF], "dtype": "float32"},
                    "observation.right_hand.qpos": {"shape": [HAND_DOF], "dtype": "float32"},
                    "observation.left_hand.qpos": {"shape": [HAND_DOF], "dtype": "float32"},
                    "observation.hand.right.tactile": {
                        "shape": [int(self.config.hand_tactile_dim)],
                        "dtype": "float32",
                    },
                    "observation.hand.left.tactile": {
                        "shape": [int(self.config.hand_tactile_dim)],
                        "dtype": "float32",
                    },
                },
            },
        )

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

    def _episode_video_relative_path(self, episode_index: int, camera_key: str) -> str:
        chunk_id = chunk_id_from_episode_index(episode_index, chunk_size=self.config.chunk_size)
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

    def plan_sample_jobs(self, sample: HandSample) -> list[WriteJob]:
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
        row, camera_expectations = hand_sample_to_row(
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
                    "summary": {} if summary is None else dict(to_serializable(dict(summary))),
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

    def discard_episode(self, episode_index: int) -> bool:
        with self._lock:
            episode = self._episodes.pop(int(episode_index), None)
            if episode is not None:
                for state in episode.video_states.values():
                    self._close_video_state(state)
                    try:
                        state.path.unlink(missing_ok=True)
                    except OSError:
                        pass
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
        for state in episode.video_states.values():
            self._drain_video_state(episode, state)
            unresolved = [
                frame_index
                for frame_index, expects_frame in sorted(state.expected_by_sample.items())
                if expects_frame
            ]
            if unresolved:
                raise RuntimeError(
                    f"Episode {episode.episode_id} camera '{state.camera_key}' is missing "
                    f"selected frames for frame indices {unresolved[:8]}."
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
        camera_keys = sorted(
            {
                str(camera_key)
                for row in rows
                for camera_key in row.get("camera_keys", [])
            }
        )
        video_paths = {
            camera_key: state.relative_path
            for camera_key, state in episode.video_states.items()
            if state.next_video_frame_index > 0
        }
        episode_record = {
            "episode_index": int(episode.episode_index),
            "episode_id": episode.episode_id,
            "frame_count": int(frame_count),
            "start_timestamp": episode.start_timestamp,
            "end_timestamp": episode.end_timestamp,
            "duration_sec": float(duration_sec),
            "parquet_path": parquet_relative_path if self.config.write_parquet else None,
            "video_paths": video_paths,
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
            "arm_enabled": bool(rows[0].get("arm_enabled", False)),
            "arm_side": rows[0].get("arm_side"),
            "arm_state_source": rows[0].get("arm_state_source"),
            "camera_keys": camera_keys,
            "fault_count": int(sum(len(row.get("fault_codes", [])) for row in rows)),
        }
        _append_jsonl(self.episodes_path, episode_record)
        _append_jsonl(self.episode_stats_path, episode_stats_record)
        self._finalized_episode_count += 1
        self._finalized_frame_count += int(frame_count)
        self._write_info_locked()
        del self._episodes[episode_index]

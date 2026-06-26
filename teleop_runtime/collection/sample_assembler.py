from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..recording import _to_serializable
from ..schemas import EffectorCommand, ObservationChannel, TeleopCommand
from ..session import SessionStepResult
from .action_record import collection_action_record_metadata
from .build_collect_components import CollectorComponents
from .video_frame import NearestFrameBuffer, VideoFrame


def _nan_vector(length: int) -> np.ndarray:
    return np.full(int(length), np.nan, dtype=np.float32)


def _copy_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, EffectorCommand):
        return value.copy()
    if isinstance(value, ObservationChannel):
        return value.copy()
    if isinstance(value, dict):
        return {key: _copy_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    return value


def _vector_or_nan(value: Any, *, length: int) -> np.ndarray:
    if value is None:
        return _nan_vector(length)
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape != (int(length),):
        return _nan_vector(length)
    if not np.all(np.isfinite(array)):
        return _nan_vector(length)
    return array.astype(np.float32, copy=True)


def _pose_xyz_quat_or_nan(value: Any) -> np.ndarray:
    if value is None:
        return _nan_vector(7)
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return _nan_vector(7)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return _nan_vector(7)
    quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)
    xyz = pose[:3, 3].astype(np.float32, copy=True)
    return np.concatenate((xyz, quat_xyzw), dtype=np.float32)


def _command_joint_or_nan(command: EffectorCommand | None, *, length: int = 6) -> np.ndarray:
    if command is None or command.command_type != "joint_vector":
        return _nan_vector(length)
    return _vector_or_nan(command.value, length=length)


def _command_pose_or_nan(command: EffectorCommand | None) -> np.ndarray:
    if command is None:
        return _nan_vector(7)
    if command.command_type == "cartesian_pose":
        return _pose_xyz_quat_or_nan(command.value)
    if command.command_type == "joint_vector":
        return _pose_xyz_quat_or_nan(command.metadata.get("source_target_pose_base"))
    return _nan_vector(7)


def _state_tcp_pose_or_nan(state: Mapping[str, Any] | None) -> np.ndarray:
    if not state:
        return _nan_vector(7)
    current_tcp_pose = state.get("current_tcp_pose")
    if current_tcp_pose is not None:
        return _pose_xyz_quat_or_nan(current_tcp_pose)
    last_command = state.get("last_command")
    if isinstance(last_command, EffectorCommand):
        if last_command.command_type == "cartesian_pose":
            return _pose_xyz_quat_or_nan(last_command.value)
        if last_command.command_type == "joint_vector":
            return _pose_xyz_quat_or_nan(last_command.metadata.get("source_target_pose_base"))
    return _nan_vector(7)


@dataclass(frozen=True)
class CameraFrameMatch:
    camera_key: str
    frame: VideoFrame | None
    selected_timestamp: float | None
    age_sec: float | None


@dataclass(frozen=True)
class CollectorSample:
    index: int
    episode_index: int
    frame_index: int
    timestamp: float
    session_state: str
    control_hz: float
    action: np.ndarray
    observation_state: np.ndarray
    observation_last_action: np.ndarray
    arm_state_source: str
    active_sides: tuple[str, ...]
    channels: dict[str, ObservationChannel] = field(default_factory=dict)
    arm_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    hand_states: dict[str, dict[str, Any]] = field(default_factory=dict)
    faults: list[dict[str, Any]] = field(default_factory=list)
    camera_frames: dict[str, CameraFrameMatch] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SampleAssemblerConfig:
    active_sides: tuple[str, ...] = ("left", "right")

    def __post_init__(self) -> None:
        normalized = tuple(str(side) for side in self.active_sides)
        invalid = set(normalized) - {"left", "right"}
        if invalid:
            raise ValueError(f"Unsupported sides in active_sides: {sorted(invalid)}")
        self.active_sides = normalized


MetadataBuilder = Callable[
    [SessionStepResult, CollectorComponents, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any] | None,
]


@dataclass
class CollectorSampleBuilderConfig:
    active_sides: tuple[str, ...] = ("left", "right")
    arm_backend_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_arm", "right": "right_arm"}
    )
    hand_backend_names: dict[str, str] = field(
        default_factory=lambda: {"left": "left_hand", "right": "right_hand"}
    )
    camera_buffers: dict[str, NearestFrameBuffer] = field(default_factory=dict)
    episode_index: int = 0
    frame_index_start: int = 0
    sample_index_start: int = 0

    def __post_init__(self) -> None:
        self.active_sides = tuple(str(side) for side in self.active_sides)
        self.arm_backend_names = {
            str(side): str(name) for side, name in dict(self.arm_backend_names).items()
        }
        self.hand_backend_names = {
            str(side): str(name) for side, name in dict(self.hand_backend_names).items()
        }
        self.camera_buffers = {
            str(camera_key): buffer for camera_key, buffer in dict(self.camera_buffers).items()
        }


class SampleAssembler:
    def __init__(self, config: SampleAssemblerConfig | None = None):
        self.config = config if config is not None else SampleAssemblerConfig()
        self._last_action = _nan_vector(26)

    def reset_last_action(self) -> None:
        self._last_action = _nan_vector(26)

    def _side_enabled(self, side: str) -> bool:
        return str(side) in self.config.active_sides

    def _build_action_vector(self, command: TeleopCommand | None) -> np.ndarray:
        if command is None:
            return _nan_vector(26)
        commands = command.commands
        return np.concatenate(
            (
                _command_pose_or_nan(commands.get("left_arm")),
                _command_pose_or_nan(commands.get("right_arm")),
                _command_joint_or_nan(commands.get("left_hand")),
                _command_joint_or_nan(commands.get("right_hand")),
            ),
            dtype=np.float32,
        )

    def _build_observation_state(
        self,
        *,
        arm_states: Mapping[str, Mapping[str, Any] | None],
        hand_states: Mapping[str, Mapping[str, Any] | None],
    ) -> np.ndarray:
        state_parts: list[np.ndarray] = []
        for side in ("left", "right"):
            if not self._side_enabled(side):
                state_parts.extend((_nan_vector(6), _nan_vector(7), _nan_vector(6)))
                continue
            arm_state = arm_states.get(side) or {}
            hand_state = hand_states.get(side) or {}
            state_parts.append(_vector_or_nan(arm_state.get("current_qpos"), length=6))
            state_parts.append(_state_tcp_pose_or_nan(arm_state))
            state_parts.append(_vector_or_nan(hand_state.get("joint_pos"), length=6))
        return np.concatenate(state_parts, dtype=np.float32)

    def _camera_matches(
        self,
        *,
        sample_timestamp: float,
        camera_frames: Mapping[str, VideoFrame | None] | None,
    ) -> dict[str, CameraFrameMatch]:
        if not camera_frames:
            return {}
        matches: dict[str, CameraFrameMatch] = {}
        for camera_key, frame in camera_frames.items():
            camera_name = str(camera_key)
            if frame is None:
                matches[camera_name] = CameraFrameMatch(
                    camera_key=camera_name,
                    frame=None,
                    selected_timestamp=None,
                    age_sec=None,
                )
                continue
            age_sec = float(sample_timestamp - float(frame.source_timestamp))
            matches[camera_name] = CameraFrameMatch(
                camera_key=camera_name,
                frame=frame,
                selected_timestamp=float(frame.source_timestamp),
                age_sec=age_sec,
            )
        return matches

    def build_sample(
        self,
        *,
        step_result: SessionStepResult,
        arm_states: Mapping[str, Mapping[str, Any] | None],
        hand_states: Mapping[str, Mapping[str, Any] | None],
        camera_frames: Mapping[str, VideoFrame | None] | None,
        index: int,
        episode_index: int,
        frame_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> CollectorSample:
        action = self._build_action_vector(step_result.command)
        observation_state = self._build_observation_state(
            arm_states=arm_states,
            hand_states=hand_states,
        )
        sample = CollectorSample(
            index=int(index),
            episode_index=int(episode_index),
            frame_index=int(frame_index),
            timestamp=float(step_result.timestamp),
            session_state=str(step_result.state.value),
            control_hz=float(step_result.control_hz),
            action=action,
            observation_state=observation_state,
            observation_last_action=self._last_action.copy(),
            arm_state_source="last_commanded_action",
            active_sides=tuple(self.config.active_sides),
            channels={
                name: channel.copy()
                for name, channel in step_result.channels.items()
            },
            arm_states={
                str(side): _copy_value(state) if state is not None else {}
                for side, state in arm_states.items()
            },
            hand_states={
                str(side): _copy_value(state) if state is not None else {}
                for side, state in hand_states.items()
            },
            faults=[
                _to_serializable(fault)
                for fault in step_result.faults
            ],
            camera_frames=self._camera_matches(
                sample_timestamp=float(step_result.timestamp),
                camera_frames=camera_frames,
            ),
            metadata={} if metadata is None else dict(_to_serializable(metadata)),
        )
        self._last_action = action.copy()
        return sample


class CollectorSampleBuilder:
    def __init__(
        self,
        config: CollectorSampleBuilderConfig | None = None,
        *,
        assembler: SampleAssembler | None = None,
        metadata_builder: MetadataBuilder | None = None,
    ):
        self.config = config if config is not None else CollectorSampleBuilderConfig()
        self.assembler = assembler if assembler is not None else SampleAssembler(
            SampleAssemblerConfig(active_sides=self.config.active_sides)
        )
        self.metadata_builder = metadata_builder
        self._episode_index = int(self.config.episode_index)
        self._frame_index = int(self.config.frame_index_start)
        self._sample_index = int(self.config.sample_index_start)

    @property
    def episode_index(self) -> int:
        return int(self._episode_index)

    @property
    def frame_index(self) -> int:
        return int(self._frame_index)

    @property
    def sample_index(self) -> int:
        return int(self._sample_index)

    def next_episode(self) -> int:
        self._episode_index += 1
        self._frame_index = 0
        return int(self._episode_index)

    def set_episode(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)
        self._frame_index = 0
        self.assembler.reset_last_action()

    def _snapshot_arm_states(self, components: CollectorComponents) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            backend_name = self.config.arm_backend_names.get(side)
            if backend_name is None:
                states[side] = {}
                continue
            backend = components.backends.get(backend_name)
            if backend is None or not hasattr(backend, "get_state"):
                states[side] = {}
                continue
            state = dict(getattr(backend, "get_state")())
            compute_tcp_pose = getattr(backend, "compute_tcp_pose", None)
            if callable(compute_tcp_pose):
                try:
                    state["current_tcp_pose"] = compute_tcp_pose()
                except Exception:
                    state["current_tcp_pose"] = None
            states[side] = state
        return states

    def _snapshot_hand_states(self, components: CollectorComponents) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        for side in ("left", "right"):
            backend_name = self.config.hand_backend_names.get(side)
            if backend_name is None:
                states[side] = {}
                continue
            backend = components.backends.get(backend_name)
            if backend is None or not hasattr(backend, "get_state"):
                states[side] = {}
                continue
            states[side] = dict(getattr(backend, "get_state")())
        return states

    def _snapshot_camera_frames(self, timestamp: float) -> dict[str, VideoFrame | None]:
        frames: dict[str, VideoFrame | None] = {}
        for camera_key, buffer in self.config.camera_buffers.items():
            frames[str(camera_key)] = buffer.get_nearest(timestamp)
        return frames

    @staticmethod
    def _default_metadata(
        step_result: SessionStepResult,
        _components: CollectorComponents,
        _arm_states: Mapping[str, Any],
        _hand_states: Mapping[str, Any],
        camera_frames: Mapping[str, VideoFrame | None],
    ) -> dict[str, Any]:
        command = step_result.command
        command_metadata = {} if command is None else _to_serializable(command.metadata)
        command_metadata_by_effector = (
            {}
            if command is None
            else {
                effector_name: _to_serializable(effector_command.metadata)
                for effector_name, effector_command in command.commands.items()
            }
        )
        channel_metadata = {
            channel_name: {
                "timestamp": float(channel.timestamp),
                "source_name": channel.source_name,
                "frame_name": channel.frame_name,
                "side": channel.side,
                "capability": channel.capability,
                "valid": bool(channel.valid),
                "sender_timestamp": channel.metadata.get("sender_timestamp"),
                "receiver_timestamp": channel.metadata.get("receiver_timestamp"),
            }
            for channel_name, channel in step_result.channels.items()
        }
        return {
            "canonical_action_record": collection_action_record_metadata(step_result),
            "command_metadata": command_metadata,
            "command_metadata_by_effector": command_metadata_by_effector,
            "channel_metadata": channel_metadata,
            "camera_keys": sorted(str(camera_key) for camera_key in camera_frames),
        }

    def __call__(
        self,
        step_result: SessionStepResult,
        components: CollectorComponents,
    ) -> CollectorSample:
        arm_states = self._snapshot_arm_states(components)
        hand_states = self._snapshot_hand_states(components)
        camera_frames = self._snapshot_camera_frames(float(step_result.timestamp))
        metadata_builder = (
            self.metadata_builder
            if self.metadata_builder is not None
            else self._default_metadata
        )
        metadata = metadata_builder(
            step_result,
            components,
            arm_states,
            hand_states,
            camera_frames,
        )
        sample = self.assembler.build_sample(
            step_result=step_result,
            arm_states=arm_states,
            hand_states=hand_states,
            camera_frames=camera_frames,
            index=self._sample_index,
            episode_index=self._episode_index,
            frame_index=self._frame_index,
            metadata=metadata,
        )
        self._sample_index += 1
        self._frame_index += 1
        return sample

from __future__ import annotations

import time
from collections.abc import Sequence
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

_COMPACT_COMMAND_METADATA_KEYS = frozenset(
    {
        "arm_action_spec",
        "arm_controlled_frame",
        "arm_reference_frame",
        "controlled_frame",
        "frame_name",
        "hand_action_spec",
        "invert",
        "output_dim",
        "source_command_type",
        "source_dim",
        "source_q6",
        "source_target_pose_base",
        "source_target_pose_desk",
        "source_target_pose_operator",
        "source_value",
        "target_pose_base",
        "target_pose_controlled_frame",
        "target_pose_desk",
        "target_pose_operator",
        "target_pose_tcp_base",
        "tcp_frame",
    }
)

_POSE_QUALITY_SCHEMA = "dexteleop_pose_quality_v1"


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _boolish(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


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


def _get_state_arg_names(get_state: Any) -> set[str]:
    code = getattr(get_state, "__code__", None)
    if code is None:
        return set()
    arg_count = int(getattr(code, "co_argcount", 0) or 0) + int(
        getattr(code, "co_kwonlyargcount", 0) or 0
    )
    return {str(name) for name in code.co_varnames[:arg_count]}


def _hand_backend_state_for_sample(
    backend: Any,
    *,
    include_tactile_arrays: bool | None = None,
    copy_tactile_arrays: bool | None = None,
) -> dict[str, Any]:
    get_state = getattr(backend, "get_state")
    arg_names = _get_state_arg_names(get_state)
    kwargs: dict[str, Any] = {}
    if include_tactile_arrays is not None and "include_tactile_arrays" in arg_names:
        kwargs["include_tactile_arrays"] = bool(include_tactile_arrays)
    if copy_tactile_arrays is not None and "copy_tactile_arrays" in arg_names:
        kwargs["copy_tactile_arrays"] = bool(copy_tactile_arrays)
    if "include_tactile_vector" in arg_names:
        kwargs["include_tactile_vector"] = False
    return dict(get_state(**kwargs))


def _backend_supports_tactile_array_filter(backend: Any) -> bool:
    get_state = getattr(backend, "get_state", None)
    if get_state is None:
        return False
    return "include_tactile_arrays" in _get_state_arg_names(get_state)


def _tactile_snapshot_key(state: Mapping[str, Any]) -> tuple[str, Any] | None:
    version = state.get("tactile_version")
    if version is not None:
        return ("version", version)
    frame_ids = state.get("tactile_frame_ids")
    if isinstance(frame_ids, Mapping) and frame_ids:
        return (
            "frame_ids",
            tuple(sorted((str(key), value) for key, value in frame_ids.items())),
        )
    timestamp = state.get("tactile_timestamp")
    if timestamp is not None:
        return ("timestamp", timestamp)
    return None


def _state_has_tactile_metadata(state: Mapping[str, Any]) -> bool:
    return (
        state.get("tactile_timestamp") is not None
        or bool(state.get("tactile_devices"))
        or bool(state.get("tactile_frame_ids"))
        or bool(state.get("tactile_shapes"))
    )


def _state_has_tactile_arrays(state: Mapping[str, Any]) -> bool:
    tactile_dict = state.get("tactile_dict")
    if isinstance(tactile_dict, Mapping) and bool(tactile_dict):
        return True
    tactile = state.get("tactile")
    if tactile is None:
        return False
    try:
        return np.asarray(tactile).size > 0
    except Exception:
        return True


def _camera_debug_snapshot(
    *,
    sample_timestamp: float,
    camera_frames: Mapping[str, VideoFrame | None] | None,
) -> dict[str, Any]:
    if not camera_frames:
        return {}
    debug: dict[str, Any] = {}
    for camera_key, frame in camera_frames.items():
        name = str(camera_key)
        if frame is None:
            debug[name] = {"missing": True}
            continue
        source_timestamp = float(frame.source_timestamp)
        receive_timestamp = (
            None if frame.receive_timestamp is None else float(frame.receive_timestamp)
        )
        debug[name] = {
            "missing": False,
            "frame_index": frame.frame_index,
            "source_timestamp": source_timestamp,
            "receive_timestamp": receive_timestamp,
            "source_age_sec": float(sample_timestamp - source_timestamp),
            "receive_age_sec": (
                None
                if receive_timestamp is None
                else float(sample_timestamp - receive_timestamp)
            ),
        }
    return debug


def _hand_debug_snapshot(
    *,
    sample_timestamp: float,
    hand_states: Mapping[str, Mapping[str, Any] | None],
) -> dict[str, Any]:
    debug: dict[str, Any] = {}
    for side, state in hand_states.items():
        hand_state = state or {}
        frame_ids = hand_state.get("tactile_frame_ids", {})
        timings = hand_state.get("tactile_timings_ms", {})
        cpu_affinity_value = hand_state.get("tactile_process_cpu_affinity")
        if cpu_affinity_value is None:
            cpu_affinity = []
        else:
            try:
                cpu_affinity = list(cpu_affinity_value)
            except TypeError:
                cpu_affinity = [cpu_affinity_value]

        def age(timestamp: Any) -> float | None:
            if timestamp is None:
                return None
            try:
                return float(sample_timestamp - float(timestamp))
            except (TypeError, ValueError):
                return None

        debug[str(side)] = {
            "joint_age_sec": age(hand_state.get("joint_timestamp")),
            "tactile_age_sec": age(hand_state.get("tactile_timestamp")),
            "telemetry_age_sec": age(hand_state.get("telemetry_timestamp")),
            "gripper_read_elapsed_s": hand_state.get("gripper_read_elapsed_s"),
            "tactile_read_elapsed_s": hand_state.get("tactile_read_elapsed_s"),
            "tactile_gate_wait_s": hand_state.get("tactile_gate_wait_s"),
            "telemetry_elapsed_s": hand_state.get("telemetry_elapsed_s"),
            "tactile_version": hand_state.get("tactile_version"),
            "tactile_devices": list(hand_state.get("tactile_devices", ())),
            "tactile_frame_ids": dict(frame_ids) if isinstance(frame_ids, Mapping) else {},
            "tactile_timings_ms": dict(timings) if isinstance(timings, Mapping) else {},
            "tactile_read_hz": hand_state.get("tactile_read_hz"),
            "tactile_read_skipped": hand_state.get("tactile_read_skipped"),
            "tactile_process_read": hand_state.get("tactile_process_read"),
            "tactile_process_nice": hand_state.get("tactile_process_nice"),
            "tactile_process_cpu_affinity": cpu_affinity,
            "tactile_process_queue_size": hand_state.get("tactile_process_queue_size"),
            "tactile_worker_alive": hand_state.get("tactile_worker_alive"),
            "tactile_worker_receiver_alive": hand_state.get("tactile_worker_receiver_alive"),
            "tactile_parallel_read": hand_state.get("tactile_parallel_read"),
            "tactile_global_read_lock": hand_state.get("tactile_global_read_lock"),
            "telemetry_error": hand_state.get("telemetry_error"),
        }
    return debug


def _compact_command_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        str(key): _to_serializable(value)
        for key, value in metadata.items()
        if str(key) in _COMPACT_COMMAND_METADATA_KEYS
    }


def _command_metadata_values(command: TeleopCommand | None, key: str) -> list[Any]:
    if command is None:
        return []
    values: list[Any] = []
    if key in (command.metadata or {}):
        values.append(command.metadata.get(key))
    for effector_command in command.commands.values():
        metadata = effector_command.metadata or {}
        if key in metadata:
            values.append(metadata.get(key))
    return values


def _first_command_metadata(command: TeleopCommand | None, key: str) -> Any:
    for value in _command_metadata_values(command, key):
        if value is not None:
            return value
    return None


def _pose_error_summary(value: Any) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        return {"max_translation_m": None, "max_rotation_deg": None}
    translations: list[float] = []
    rotations: list[float] = []
    for metrics in value.values():
        if not isinstance(metrics, Mapping):
            continue
        translation = _finite_float(metrics.get("translation_m"))
        rotation = _finite_float(metrics.get("rotation_deg"))
        if translation is not None:
            translations.append(translation)
        if rotation is not None:
            rotations.append(rotation)
    return {
        "max_translation_m": max(translations) if translations else None,
        "max_rotation_deg": max(rotations) if rotations else None,
    }


def _joint_margin_summary(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {"nearest_margin_deg": None, "near_limit": False}
    margins: list[float] = []
    near_limit = False
    by_side: dict[str, Any] = {}
    for side, payload in value.items():
        if not isinstance(payload, Mapping):
            continue
        margin = _finite_float(payload.get("nearest_margin_deg"))
        if margin is not None:
            margins.append(margin)
        side_near = _boolish(payload.get("near_limit", False))
        near_limit = near_limit or side_near
        by_side[str(side)] = {
            "nearest_margin_deg": margin,
            "nearest_joint": payload.get("nearest_joint"),
            "near_limit": side_near,
        }
    return {
        "nearest_margin_deg": min(margins) if margins else None,
        "near_limit": near_limit,
        "by_side": by_side,
    }


def _build_pose_quality_metadata(command: TeleopCommand | None) -> dict[str, Any]:
    if command is None:
        return {
            "schema": _POSE_QUALITY_SCHEMA,
            "available": False,
            "status": "unknown",
            "score": None,
            "source": "no_command",
            "bad": False,
            "reasons": ["no_command"],
            "metrics": {},
        }

    collision_status = _first_command_metadata(command, "collision_avoidance_status")
    collision_distance_m = _finite_float(_first_command_metadata(command, "collision_distance_m"))
    collision_min_distance_m = _finite_float(
        _first_command_metadata(command, "collision_min_distance_m")
    )
    fallback_reason = _first_command_metadata(command, "collision_avoidance_fallback_reason")
    branch_candidate = _first_command_metadata(command, "ik_branch_candidate")
    jump_gate_status = _first_command_metadata(command, "ik_jump_gate_status")
    time_budget_exceeded = _boolish(
        _first_command_metadata(command, "ik_time_budget_exceeded")
    )
    execution_step = _first_command_metadata(command, "execution_step_metadata")
    execution_step_limited = (
        isinstance(execution_step, Mapping)
        and _boolish(execution_step.get("step_limited"))
    )
    pose_error = _pose_error_summary(_first_command_metadata(command, "ik_solve_pose_error"))
    joint_margin = _joint_margin_summary(_first_command_metadata(command, "joint_limit_margin"))

    has_signal = any(
        value is not None
        for value in (
            collision_status,
            collision_distance_m,
            collision_min_distance_m,
            fallback_reason,
            branch_candidate,
            jump_gate_status,
            pose_error["max_translation_m"],
            joint_margin["nearest_margin_deg"],
        )
    )
    if not has_signal:
        return {
            "schema": _POSE_QUALITY_SCHEMA,
            "available": False,
            "status": "unknown",
            "score": None,
            "source": "command_without_quality_metadata",
            "bad": False,
            "reasons": ["quality_metadata_unavailable"],
            "metrics": {},
        }

    bad_reasons: list[str] = []
    warning_reasons: list[str] = []
    status_text = "" if collision_status is None else str(collision_status)
    if status_text in {"hold_last_safe", "unsafe_candidate", "jump_gate_hold"}:
        bad_reasons.append(f"collision_status:{status_text}")
    elif status_text == "boundary_active":
        warning_reasons.append("collision_boundary_active")
    if fallback_reason:
        bad_reasons.append(f"fallback:{fallback_reason}")
    if jump_gate_status not in {None, "pass"}:
        warning_reasons.append(f"jump_gate:{jump_gate_status}")
    if collision_distance_m is not None and collision_min_distance_m is not None:
        margin = collision_distance_m - collision_min_distance_m
        if margin < 0.0:
            bad_reasons.append("collision_distance_below_min")
        elif margin <= 0.02:
            warning_reasons.append("collision_distance_near_min")
    nearest_margin = _finite_float(joint_margin.get("nearest_margin_deg"))
    if nearest_margin is not None:
        if nearest_margin < 3.0:
            bad_reasons.append("joint_limit_margin_lt_3deg")
        elif nearest_margin < 10.0 or _boolish(joint_margin.get("near_limit")):
            warning_reasons.append("joint_limit_margin_low")
    max_translation = _finite_float(pose_error.get("max_translation_m"))
    max_rotation = _finite_float(pose_error.get("max_rotation_deg"))
    if max_translation is not None:
        if max_translation > 0.02:
            bad_reasons.append("ik_translation_error_gt_2cm")
        elif max_translation > 0.005:
            warning_reasons.append("ik_translation_error_gt_5mm")
    if max_rotation is not None:
        if max_rotation > 5.0:
            bad_reasons.append("ik_rotation_error_gt_5deg")
        elif max_rotation > 2.0:
            warning_reasons.append("ik_rotation_error_gt_2deg")
    if execution_step_limited:
        warning_reasons.append("execution_step_limited")
    if time_budget_exceeded:
        warning_reasons.append("ik_time_budget_exceeded")

    score = 1.0
    score -= 0.35 * len(bad_reasons)
    score -= 0.12 * len(warning_reasons)
    score = max(0.0, min(1.0, float(score)))
    if bad_reasons:
        quality_status = "bad"
    elif warning_reasons:
        quality_status = "warning"
    else:
        quality_status = "good"
    return {
        "schema": _POSE_QUALITY_SCHEMA,
        "available": True,
        "status": quality_status,
        "score": score,
        "bad": bool(bad_reasons),
        "source": "existing_command_metadata",
        "reasons": bad_reasons + warning_reasons,
        "bad_reasons": bad_reasons,
        "warning_reasons": warning_reasons,
        "metrics": {
            "collision_avoidance_status": collision_status,
            "collision_distance_m": collision_distance_m,
            "collision_min_distance_m": collision_min_distance_m,
            "collision_margin_m": (
                None
                if collision_distance_m is None or collision_min_distance_m is None
                else float(collision_distance_m - collision_min_distance_m)
            ),
            "collision_avoidance_fallback_reason": fallback_reason,
            "ik_branch_candidate": branch_candidate,
            "ik_jump_gate_status": jump_gate_status,
            "ik_solve_pose_error": pose_error,
            "joint_limit_margin": joint_margin,
            "execution_step_limited": execution_step_limited,
            "ik_time_budget_exceeded": time_budget_exceeded,
        },
    }


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
    array = _vector_or_nan(command.value, length=length)
    if np.all(np.isfinite(array)):
        return array
    raw = np.asarray(command.value, dtype=np.float32).reshape(-1)
    if raw.shape != (1,) or not np.all(np.isfinite(raw)):
        return _nan_vector(length)
    spec_payload = (command.metadata or {}).get("hand_action_spec")
    if not isinstance(spec_payload, Mapping):
        return _nan_vector(length)
    active_mask = spec_payload.get("active_mask")
    if not isinstance(active_mask, Sequence) or len(active_mask) != int(length):
        return _nan_vector(length)
    active_indices = [index for index, enabled in enumerate(active_mask) if bool(enabled)]
    if len(active_indices) != 1:
        return _nan_vector(length)
    padded = np.zeros(int(length), dtype=np.float32)
    padded[int(active_indices[0])] = float(raw[0])
    return padded


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
ArmStateMetadataProvider = Callable[
    [CollectorComponents, float],
    Mapping[str, Mapping[str, Any]] | None,
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
    include_debug_metadata: bool = False
    tactile_array_snapshot_policy: str = "on_change"

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
        self.include_debug_metadata = bool(self.include_debug_metadata)
        policy = str(self.tactile_array_snapshot_policy).strip().lower().replace("-", "_")
        if policy not in {"always", "on_change", "never"}:
            raise ValueError("tactile_array_snapshot_policy must be always, on_change, or never.")
        self.tactile_array_snapshot_policy = policy


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
        arm_state_metadata_provider: ArmStateMetadataProvider | None = None,
    ):
        self.config = config if config is not None else CollectorSampleBuilderConfig()
        self.assembler = assembler if assembler is not None else SampleAssembler(
            SampleAssemblerConfig(active_sides=self.config.active_sides)
        )
        self.metadata_builder = metadata_builder
        self.arm_state_metadata_provider = arm_state_metadata_provider
        self.include_debug_metadata = bool(self.config.include_debug_metadata)
        self._episode_index = int(self.config.episode_index)
        self._frame_index = int(self.config.frame_index_start)
        self._sample_index = int(self.config.sample_index_start)
        self._last_tactile_snapshot_key_by_side: dict[str, tuple[str, Any] | None] = {}

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
        self._last_tactile_snapshot_key_by_side.clear()
        return int(self._episode_index)

    def set_episode(self, episode_index: int) -> None:
        self._episode_index = int(episode_index)
        self._frame_index = 0
        self.assembler.reset_last_action()
        self._last_tactile_snapshot_key_by_side.clear()

    def _snapshot_arm_state_metadata(
        self,
        components: CollectorComponents,
        *,
        sample_timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        provider = self.arm_state_metadata_provider
        if provider is None:
            return {}
        try:
            metadata = provider(components, float(sample_timestamp))
        except Exception as exc:  # noqa: BLE001 - annotation failures should not block collection.
            return {
                side: {
                    "actual_state_provider_error": f"{type(exc).__name__}: {exc}",
                }
                for side in ("left", "right")
            }
        if not isinstance(metadata, Mapping):
            return {}
        return {
            str(side): dict(_to_serializable(value))
            for side, value in metadata.items()
            if isinstance(value, Mapping)
        }

    def _snapshot_arm_states(
        self,
        components: CollectorComponents,
        *,
        sample_timestamp: float,
    ) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        actual_metadata = self._snapshot_arm_state_metadata(
            components,
            sample_timestamp=sample_timestamp,
        )
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
            extra = actual_metadata.get(side)
            if extra:
                state.update(_copy_value(extra))
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
            states[side] = self._snapshot_hand_state(side, backend)
        return states

    def _snapshot_hand_state(self, side: str, backend: Any) -> dict[str, Any]:
        policy = str(self.config.tactile_array_snapshot_policy)
        if policy == "always" or not _backend_supports_tactile_array_filter(backend):
            return _hand_backend_state_for_sample(backend, copy_tactile_arrays=False)
        if policy == "never":
            return _hand_backend_state_for_sample(backend, include_tactile_arrays=False)

        light_state = _hand_backend_state_for_sample(
            backend,
            include_tactile_arrays=False,
            copy_tactile_arrays=False,
        )
        snapshot_key = _tactile_snapshot_key(light_state)
        if snapshot_key is None or not _state_has_tactile_metadata(light_state):
            return light_state
        if self._last_tactile_snapshot_key_by_side.get(side) == snapshot_key:
            return light_state

        full_state = _hand_backend_state_for_sample(
            backend,
            include_tactile_arrays=True,
            copy_tactile_arrays=False,
        )
        if _state_has_tactile_arrays(full_state):
            self._last_tactile_snapshot_key_by_side[side] = snapshot_key
            return full_state
        return light_state

    def _snapshot_camera_frames(self, timestamp: float) -> dict[str, VideoFrame | None]:
        frames: dict[str, VideoFrame | None] = {}
        for camera_key, buffer in self.config.camera_buffers.items():
            frames[str(camera_key)] = buffer.get_latest_before_or_at(timestamp)
        return frames

    @staticmethod
    def _default_metadata(
        step_result: SessionStepResult,
        _components: CollectorComponents,
        _arm_states: Mapping[str, Any],
        _hand_states: Mapping[str, Any],
        camera_frames: Mapping[str, VideoFrame | None],
        *,
        include_debug_metadata: bool = False,
    ) -> dict[str, Any]:
        command = step_result.command
        command_metadata = (
            {}
            if command is None
            else (
                _to_serializable(command.metadata)
                if include_debug_metadata
                else _compact_command_metadata(command.metadata)
            )
        )
        command_metadata_by_effector = (
            {}
            if command is None
            else {
                effector_name: (
                    _to_serializable(effector_command.metadata)
                    if include_debug_metadata
                    else _compact_command_metadata(effector_command.metadata)
                )
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
                "sender_read_elapsed_ms": channel.metadata.get("sender_read_elapsed_ms"),
            }
            for channel_name, channel in step_result.channels.items()
        }
        return {
            "canonical_action_record": collection_action_record_metadata(
                step_result,
                include_debug_metadata=include_debug_metadata,
            ),
            "pose_quality": _build_pose_quality_metadata(command),
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
        timing_started = time.perf_counter()
        arm_states = self._snapshot_arm_states(
            components,
            sample_timestamp=float(step_result.timestamp),
        )
        arm_snapshot_s = time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        hand_states = self._snapshot_hand_states(components)
        hand_snapshot_s = time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        camera_frames = self._snapshot_camera_frames(float(step_result.timestamp))
        camera_snapshot_s = time.perf_counter() - timing_started
        timing_started = time.perf_counter()
        if self.metadata_builder is None:
            metadata = self._default_metadata(
                step_result,
                components,
                arm_states,
                hand_states,
                camera_frames,
                include_debug_metadata=bool(self.include_debug_metadata),
            )
        else:
            metadata = self.metadata_builder(
                step_result,
                components,
                arm_states,
                hand_states,
                camera_frames,
            )
        metadata = {} if metadata is None else dict(metadata)
        metadata["sample_builder_debug"] = {
            "arm_snapshot_s": float(arm_snapshot_s),
            "hand_snapshot_s": float(hand_snapshot_s),
            "camera_snapshot_s": float(camera_snapshot_s),
            "metadata_build_s": float(time.perf_counter() - timing_started),
            "camera_frames": _camera_debug_snapshot(
                sample_timestamp=float(step_result.timestamp),
                camera_frames=camera_frames,
            ),
            "hand_states": _hand_debug_snapshot(
                sample_timestamp=float(step_result.timestamp),
                hand_states=hand_states,
            ),
        }
        timing_started = time.perf_counter()
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
        sample.metadata.setdefault("sample_builder_debug", {})["build_sample_s"] = float(
            time.perf_counter() - timing_started
        )
        self._sample_index += 1
        self._frame_index += 1
        return sample

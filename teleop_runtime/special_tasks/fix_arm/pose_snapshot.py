from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...schemas import EffectorCommand, ObservationChannel, TeleopCommand

POSE_SNAPSHOT_SCHEMA = "dexteleop_fix_arm_pose_snapshot_v1"


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result):
        return None
    return result


def _pose_or_none(value: Any) -> list[list[float]] | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except (TypeError, ValueError):
        return None
    return json_safe(pose)


def _array_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return json_safe(value)
    return json_safe(array)


def serialize_channel(channel: ObservationChannel | Any) -> dict[str, Any]:
    metadata = getattr(channel, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "channel_name": str(getattr(channel, "channel_name", "") or ""),
        "capability": str(getattr(channel, "capability", "") or ""),
        "side": getattr(channel, "side", None),
        "valid": bool(getattr(channel, "valid", False)),
        "critical": bool(getattr(channel, "critical", False)),
        "frame_name": getattr(channel, "frame_name", None),
        "timestamp": _float_or_none(getattr(channel, "timestamp", None)),
        "source_name": str(getattr(channel, "source_name", "") or ""),
        "value": json_safe(getattr(channel, "value", None)),
        "metadata": json_safe(metadata),
    }


def serialize_channels(channels: Mapping[str, ObservationChannel] | None) -> dict[str, Any]:
    if not isinstance(channels, Mapping):
        return {}
    return {
        str(name): serialize_channel(channel)
        for name, channel in sorted(channels.items(), key=lambda item: str(item[0]))
    }


def serialize_effector_command(command: EffectorCommand | Any) -> dict[str, Any]:
    metadata = getattr(command, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "effector_name": str(getattr(command, "effector_name", "") or ""),
        "side": getattr(command, "side", None),
        "command_type": str(getattr(command, "command_type", "") or ""),
        "frame_name": getattr(command, "frame_name", None),
        "value": json_safe(getattr(command, "value", None)),
        "metadata": json_safe(metadata),
    }


def serialize_command(command: TeleopCommand | None) -> dict[str, Any] | None:
    if command is None:
        return None
    commands = getattr(command, "commands", {}) or {}
    if not isinstance(commands, Mapping):
        commands = {}
    metadata = getattr(command, "metadata", {}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "timestamp": _float_or_none(getattr(command, "timestamp", None)),
        "metadata": json_safe(metadata),
        "commands": {
            str(name): serialize_effector_command(effector)
            for name, effector in sorted(commands.items(), key=lambda item: str(item[0]))
        },
    }


def _faults_payload(faults: Sequence[Any] | None) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for fault in faults or ():
        severity = getattr(fault, "severity", None)
        metadata = getattr(fault, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        payload.append(
            {
                "code": str(getattr(fault, "code", fault) or ""),
                "channel_name": getattr(fault, "channel_name", None),
                "component": getattr(fault, "component", None),
                "severity": str(getattr(severity, "value", severity) or ""),
                "message": str(getattr(fault, "message", "") or ""),
                "timestamp": _float_or_none(getattr(fault, "timestamp", None)),
                "metadata": json_safe(metadata),
            }
        )
    return payload


def _retargeter_home_payload(retargeter: Any) -> dict[str, Any]:
    if retargeter is None:
        return {
            "home_human_pose_desk": None,
            "home_robot_pose_desk": None,
            "last_target_pose_desk": None,
        }
    return {
        "home_human_pose_desk": _pose_or_none(
            getattr(retargeter, "home_human_pose_desk", None)
        ),
        "home_robot_pose_desk": _pose_or_none(
            getattr(retargeter, "home_robot_pose_desk", None)
        ),
        "last_target_pose_desk": _pose_or_none(
            getattr(retargeter, "last_target_pose_desk", None)
        ),
    }


def _calibration_payload(calibration: Any) -> dict[str, Any] | None:
    if calibration is None:
        return None
    to_dict = getattr(calibration, "to_dict", None)
    if callable(to_dict):
        try:
            return json_safe(to_dict())
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}
    return json_safe(calibration)


def build_fix_arm_pose_snapshot_payload(
    *,
    timestamp: float,
    monotonic_time: float,
    step_index: int,
    active_sides: Sequence[str],
    teleop_state: str,
    control_hz: float,
    movement_multiple: float,
    tcp_mode: str,
    calibration: Any,
    raw_channels: Mapping[str, ObservationChannel] | None,
    filtered_channels: Mapping[str, ObservationChannel] | None,
    observation: Any,
    step_command: TeleopCommand | None,
    active_command: TeleopCommand | None,
    hand_command: TeleopCommand | None,
    gripper_command: TeleopCommand | None,
    command_dispatched: bool,
    faults: Sequence[Any],
    fix_arm_state: Mapping[str, Any],
    left_retargeter: Any,
    right_retargeter: Any,
    left_preview_qpos: Any,
    right_preview_qpos: Any,
    left_hardware_state: Mapping[str, Any] | None,
    right_hardware_state: Mapping[str, Any] | None,
    left_command_state: Mapping[str, Any] | None,
    right_command_state: Mapping[str, Any] | None,
    gripper_status: Mapping[str, Any] | None,
    pose_quality: Mapping[str, Any] | None = None,
    ik_timing: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    observation_timestamp = None
    if observation is not None:
        observation_timestamp = _float_or_none(getattr(observation, "timestamp", None))
    payload = {
        "schema": POSE_SNAPSHOT_SCHEMA,
        "timestamp": float(timestamp),
        "monotonic_time": float(monotonic_time),
        "step_index": int(step_index),
        "runtime": {
            "active_sides": [str(side) for side in active_sides],
            "teleop_state": str(teleop_state),
            "control_hz": float(control_hz),
            "movement_multiple": float(movement_multiple),
            "tcp_mode": str(tcp_mode),
            "command_dispatched": bool(command_dispatched),
        },
        "fix_arm": json_safe(dict(fix_arm_state)),
        "calibration": _calibration_payload(calibration),
        "home_anchors": {
            "left": _retargeter_home_payload(left_retargeter),
            "right": _retargeter_home_payload(right_retargeter),
        },
        "tracker": {
            "raw_channels": serialize_channels(raw_channels),
            "filtered_channels": serialize_channels(filtered_channels),
        },
        "observation": {
            "timestamp": observation_timestamp,
            "metadata": json_safe(getattr(observation, "metadata", {}) or {}),
        },
        "commands": {
            "step_command": serialize_command(step_command),
            "active_command": serialize_command(active_command),
            "hand_command": serialize_command(hand_command),
            "gripper_command": serialize_command(gripper_command),
        },
        "robot_state": {
            "preview_qpos": {
                "left": _array_or_none(left_preview_qpos),
                "right": _array_or_none(right_preview_qpos),
            },
            "hardware": {
                "left": json_safe(left_hardware_state or {}),
                "right": json_safe(right_hardware_state or {}),
            },
            "command_backend": {
                "left": json_safe(left_command_state or {}),
                "right": json_safe(right_command_state or {}),
            },
        },
        "quality": {
            "faults": _faults_payload(faults),
            "gripper_status": json_safe(gripper_status or {}),
            "pose_quality": json_safe(pose_quality or {}),
            "ik_timing": json_safe(ik_timing or {}),
        },
    }
    if extra:
        payload["extra"] = json_safe(dict(extra))
    return payload


@dataclass(frozen=True)
class PoseSnapshotRecord:
    sequence: int
    path: Path
    timestamp: float


class FixArmPoseSnapshotRecorder:
    def __init__(self, root_dir: str | Path) -> None:
        self.root_dir = Path(root_dir).expanduser()
        self._lock = threading.Lock()
        self._count = 0

    @property
    def count(self) -> int:
        return int(self._count)

    @property
    def index_path(self) -> Path:
        return self.root_dir / "index.jsonl"

    def record(self, payload: Mapping[str, Any], *, now: float | None = None) -> PoseSnapshotRecord:
        timestamp = float(now if now is not None else payload.get("timestamp", 0.0) or 0.0)
        with self._lock:
            self.root_dir.mkdir(parents=True, exist_ok=True)
            sequence = self._count + 1
            stamp = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S_%f")
            path = self.root_dir / f"pose_{sequence:06d}_{stamp}.json"
            full_payload = dict(payload)
            full_payload.setdefault("snapshot_sequence", sequence)
            full_payload.setdefault("snapshot_path", str(path))
            path.write_text(
                json.dumps(json_safe(full_payload), ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            index_record = {
                "sequence": sequence,
                "timestamp": timestamp,
                "path": str(path),
                "schema": full_payload.get("schema", POSE_SNAPSHOT_SCHEMA),
                "step_index": full_payload.get("step_index"),
                "fix_arm": full_payload.get("fix_arm", {}),
            }
            with self.index_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(json_safe(index_record), ensure_ascii=False, sort_keys=True) + "\n")
            self._count = sequence
        return PoseSnapshotRecord(sequence=sequence, path=path, timestamp=timestamp)

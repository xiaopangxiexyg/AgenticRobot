from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as R

from ..recording import _to_serializable
from ..schemas import EffectorCommand, ObservationChannel
from .sample_assembler import CollectorSample


ACTION_DIM = 26
OBSERVATION_STATE_DIM = 38


def episode_id_from_index(index: int) -> str:
    return f"episode_{int(index):06d}"


def chunk_id_from_episode_index(index: int, *, chunk_size: int) -> str:
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive.")
    return f"chunk-{int(index) // int(chunk_size):03d}"


def camera_video_channel(camera_key: str) -> str:
    return f"observation.images.{str(camera_key)}"


def _vector_list(value: Any, *, length: int | None = None) -> list[float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if length is not None and array.shape != (int(length),):
        return None
    return array.astype(np.float32, copy=False).tolist()


def _pose_matrix_to_xyzquat_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    try:
        pose = np.asarray(value, dtype=float).reshape(4, 4)
    except Exception:
        return None
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        return None
    quat_xyzw = R.from_matrix(pose[:3, :3]).as_quat().astype(np.float32)
    xyz = pose[:3, 3].astype(np.float32, copy=False)
    return np.concatenate((xyz, quat_xyzw), dtype=np.float32).tolist()


def _serialize_channel(channel: ObservationChannel) -> dict[str, Any]:
    return {
        "timestamp": float(channel.timestamp),
        "source_name": channel.source_name,
        "frame_name": channel.frame_name,
        "side": channel.side,
        "capability": channel.capability,
        "valid": bool(channel.valid),
        "critical": bool(channel.critical),
        "sender_timestamp": channel.metadata.get("sender_timestamp"),
        "receiver_timestamp": channel.metadata.get("receiver_timestamp"),
    }


def _command_from_state(state: Mapping[str, Any] | None) -> EffectorCommand | None:
    if not state:
        return None
    command = state.get("last_command")
    return command if isinstance(command, EffectorCommand) else None


def _command_pose_metadata(command: EffectorCommand | None, metadata_key: str) -> list[float] | None:
    if command is None:
        return None
    if metadata_key == "source_target_pose_base" and command.command_type == "cartesian_pose":
        return _pose_matrix_to_xyzquat_list(command.value)
    if metadata_key == "target_pose_base" and command.command_type == "cartesian_pose":
        return _pose_matrix_to_xyzquat_list(command.value)
    return _pose_matrix_to_xyzquat_list(command.metadata.get(metadata_key))


def _command_vector_metadata(
    command: EffectorCommand | None,
    metadata_key: str,
    *,
    length: int | None = None,
) -> list[float] | None:
    if command is None:
        return None
    if metadata_key == "command_value" and command.command_type == "joint_vector":
        return _vector_list(command.value, length=length)
    return _vector_list(command.metadata.get(metadata_key), length=length)


def _mapping_pose_metadata(mapping: Mapping[str, Any] | None, metadata_key: str) -> list[float] | None:
    if not mapping:
        return None
    return _pose_matrix_to_xyzquat_list(mapping.get(metadata_key))


def _split_action(action: np.ndarray) -> dict[str, list[float]]:
    vector = np.asarray(action, dtype=np.float32).reshape(-1)
    if vector.shape != (ACTION_DIM,):
        raise ValueError(f"CollectorSample.action must have shape ({ACTION_DIM},).")
    return {
        "left_arm": vector[0:7].tolist(),
        "right_arm": vector[7:14].tolist(),
        "left_hand": vector[14:20].tolist(),
        "right_hand": vector[20:26].tolist(),
    }


def _split_observation_state(observation_state: np.ndarray) -> dict[str, list[float]]:
    vector = np.asarray(observation_state, dtype=np.float32).reshape(-1)
    if vector.shape != (OBSERVATION_STATE_DIM,):
        raise ValueError(
            f"CollectorSample.observation_state must have shape ({OBSERVATION_STATE_DIM},)."
        )
    return {
        "left_arm_qpos": vector[0:6].tolist(),
        "left_arm_pose": vector[6:13].tolist(),
        "left_hand_qpos": vector[13:19].tolist(),
        "right_arm_qpos": vector[19:25].tolist(),
        "right_arm_pose": vector[25:32].tolist(),
        "right_hand_qpos": vector[32:38].tolist(),
    }


def collector_sample_to_row(
    sample: CollectorSample,
    *,
    camera_video_paths: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, bool]]:
    action_split = _split_action(sample.action)
    observation_split = _split_observation_state(sample.observation_state)
    row: dict[str, Any] = {
        "index": int(sample.index),
        "episode_index": int(sample.episode_index),
        "frame_index": int(sample.frame_index),
        "timestamp": float(sample.timestamp),
        "session_state": sample.session_state,
        "control_hz": float(sample.control_hz),
        "arm_state_source": sample.arm_state_source,
        "active_sides": list(sample.active_sides),
        "camera_keys": sorted(sample.camera_frames.keys()),
        "fault_codes": [
            str(fault.get("code"))
            for fault in sample.faults
            if isinstance(fault, Mapping) and fault.get("code") is not None
        ],
        "action": np.asarray(sample.action, dtype=np.float32).tolist(),
        "action.left_arm.pose": action_split["left_arm"],
        "action.right_arm.pose": action_split["right_arm"],
        "action.left_hand.joint_target": action_split["left_hand"],
        "action.right_hand.joint_target": action_split["right_hand"],
        "observation.state": np.asarray(sample.observation_state, dtype=np.float32).tolist(),
        "observation.last_action": np.asarray(
            sample.observation_last_action, dtype=np.float32
        ).tolist(),
        "observation.left_arm.qpos": observation_split["left_arm_qpos"],
        "observation.left_arm.pose": observation_split["left_arm_pose"],
        "observation.left_hand.qpos": observation_split["left_hand_qpos"],
        "observation.right_arm.qpos": observation_split["right_arm_qpos"],
        "observation.right_arm.pose": observation_split["right_arm_pose"],
        "observation.right_hand.qpos": observation_split["right_hand_qpos"],
        "sample_metadata_json": json.dumps(
            _to_serializable(sample.metadata),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "channels_json": json.dumps(
            {name: _serialize_channel(channel) for name, channel in sample.channels.items()},
            ensure_ascii=True,
            sort_keys=True,
        ),
        "faults_json": json.dumps(_to_serializable(sample.faults), ensure_ascii=True, sort_keys=True),
    }
    camera_expectations: dict[str, bool] = {}
    command_metadata_by_effector = sample.metadata.get("command_metadata_by_effector", {})

    for side in ("left", "right"):
        arm_state = sample.arm_states.get(side) or {}
        hand_state = sample.hand_states.get(side) or {}
        arm_command = _command_from_state(arm_state)
        hand_command = _command_from_state(hand_state)
        arm_command_metadata = command_metadata_by_effector.get(f"{side}_arm", {})
        hand_command_metadata = command_metadata_by_effector.get(f"{side}_hand", {})

        row[f"observation.arm.{side}.current_qpos"] = _vector_list(
            arm_state.get("current_qpos"),
            length=6,
        )
        row[f"observation.arm.{side}.current_tcp_pose"] = _pose_matrix_to_xyzquat_list(
            arm_state.get("current_tcp_pose")
        )
        row[f"observation.arm.{side}.last_target_qpos"] = _vector_list(
            arm_state.get("last_target_qpos"),
            length=6,
        )
        row[f"observation.arm.{side}.last_command_timestamp"] = arm_state.get(
            "last_command_timestamp"
        )
        row[f"observation.arm.{side}.effective_step_scale"] = arm_state.get("effective_step_scale")
        row[f"action.replay.{side}_arm_joint_target"] = _vector_list(
            arm_state.get("last_target_qpos"),
            length=6,
        )
        row[f"teleop.action_source.{side}_wrist_pose_operator"] = _command_pose_metadata(
            arm_command,
            "source_target_pose_operator",
        ) or _command_pose_metadata(arm_command, "target_pose_operator") or _mapping_pose_metadata(
            arm_command_metadata,
            "source_target_pose_operator",
        ) or _mapping_pose_metadata(
            arm_command_metadata,
            "target_pose_operator",
        )
        row[f"teleop.action_source.{side}_wrist_pose_desk"] = _command_pose_metadata(
            arm_command,
            "source_target_pose_desk",
        ) or _command_pose_metadata(arm_command, "target_pose_desk") or _mapping_pose_metadata(
            arm_command_metadata,
            "source_target_pose_desk",
        ) or _mapping_pose_metadata(
            arm_command_metadata,
            "target_pose_desk",
        )
        row[f"teleop.action_source.{side}_wrist_pose_base"] = _command_pose_metadata(
            arm_command,
            "source_target_pose_base",
        ) or _command_pose_metadata(arm_command, "target_pose_base") or _mapping_pose_metadata(
            arm_command_metadata,
            "source_target_pose_base",
        )

        row[f"observation.hand.{side}.joint_pos"] = _vector_list(
            hand_state.get("joint_pos"),
            length=6,
        )
        row[f"observation.hand.{side}.joint_timestamp"] = hand_state.get("joint_timestamp")
        row[f"observation.hand.{side}.tactile"] = _vector_list(hand_state.get("tactile"))
        row[f"observation.hand.{side}.tactile_timestamp"] = hand_state.get("tactile_timestamp")
        row[f"observation.hand.{side}.telemetry_timestamp"] = hand_state.get("telemetry_timestamp")
        row[f"observation.hand.{side}.tactile_regions"] = list(
            hand_state.get("tactile_regions", ())
        )
        row[f"action.replay.{side}_hand_joint_target"] = _vector_list(
            hand_state.get("last_sent_command"),
            length=6,
        )
        row[f"teleop.action_source.{side}_hand_q6"] = _command_vector_metadata(
            hand_command,
            "source_q6",
            length=6,
        ) or _vector_list(hand_command_metadata.get("source_q6"), length=6)

    for channel_name, channel in sample.channels.items():
        row[f"teleop.channels.{channel_name}.timestamp"] = float(channel.timestamp)
        row[f"teleop.channels.{channel_name}.sender_timestamp"] = channel.metadata.get(
            "sender_timestamp"
        )
        row[f"teleop.channels.{channel_name}.receiver_timestamp"] = channel.metadata.get(
            "receiver_timestamp"
        )
        row[f"teleop.channels.{channel_name}.valid"] = bool(channel.valid)

    for camera_key, match in sample.camera_frames.items():
        prefix = f"{camera_video_channel(camera_key)}"
        frame = match.frame
        camera_expectations[camera_key] = frame is not None
        row[f"{prefix}.source_timestamp"] = match.selected_timestamp
        row[f"{prefix}.receive_timestamp"] = None if frame is None else frame.receive_timestamp
        row[f"{prefix}.age_sec"] = match.age_sec
        row[f"{prefix}.source_frame_index"] = None if frame is None else frame.frame_index
        row[f"{prefix}.video_frame_index"] = None
        if camera_video_paths is not None and camera_key in camera_video_paths:
            row[f"{prefix}.path"] = str(camera_video_paths[camera_key])

    return row, camera_expectations


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
        raise ModuleNotFoundError(
            "LeRobot parquet export requires pyarrow to be installed."
        ) from exc

    parquet_path = Path(path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_rows = [_to_serializable(dict(row)) for row in rows]
    table = pa.Table.from_pylist(normalized_rows)
    pq.write_table(table, parquet_path, compression=compression)

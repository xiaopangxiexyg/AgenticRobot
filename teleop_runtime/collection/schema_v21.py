from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

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


def _hand_specs_from_sample_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    canonical = metadata.get("canonical_action_record")
    if isinstance(canonical, Mapping):
        action = canonical.get("action")
        if isinstance(action, Mapping):
            action_metadata = action.get("metadata")
            if isinstance(action_metadata, Mapping):
                specs = action_metadata.get("hand_action_specs_by_side")
                if isinstance(specs, Mapping):
                    return dict(_to_serializable(specs))
    specs = metadata.get("hand_action_specs_by_side")
    if isinstance(specs, Mapping):
        return dict(_to_serializable(specs))
    return {}


def _arm_specs_from_sample_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    canonical = metadata.get("canonical_action_record")
    if isinstance(canonical, Mapping):
        action = canonical.get("action")
        if isinstance(action, Mapping):
            action_metadata = action.get("metadata")
            if isinstance(action_metadata, Mapping):
                specs = action_metadata.get("arm_action_specs_by_side")
                if isinstance(specs, Mapping):
                    return dict(_to_serializable(specs))
    specs = metadata.get("arm_action_specs_by_side")
    if isinstance(specs, Mapping):
        return dict(_to_serializable(specs))
    return {}


def _hand_active_masks_from_specs(specs: Mapping[str, Any]) -> dict[str, list[bool]]:
    masks: dict[str, list[bool]] = {}
    for side, spec in specs.items():
        if isinstance(spec, Mapping):
            active_mask = spec.get("active_mask")
            if isinstance(active_mask, list | tuple):
                masks[str(side)] = [bool(value) for value in active_mask]
    return masks


def _gripper_slot_from_metadata(
    hand_command_metadata: Mapping[str, Any],
    hand_spec: Mapping[str, Any] | None,
) -> int:
    for key in ("gripper_slot", "output_dim"):
        value = hand_command_metadata.get(key)
        try:
            slot = int(value)
        except (TypeError, ValueError):
            continue
        if 0 <= slot < 6:
            return slot
    if isinstance(hand_spec, Mapping):
        metadata = hand_spec.get("metadata")
        if isinstance(metadata, Mapping):
            try:
                slot = int(metadata.get("gripper_slot"))
            except (TypeError, ValueError):
                slot = -1
            if 0 <= slot < 6:
                return slot
        slot_map = hand_spec.get("slot_map")
        if isinstance(slot_map, Mapping):
            try:
                slot = int(slot_map.get("gripper_close"))
            except (TypeError, ValueError):
                slot = -1
            if 0 <= slot < 6:
                return slot
    return 0


def _gripper_target_from_action(
    target_from_action: Sequence[float] | None,
    *,
    slot: int,
    metadata: Mapping[str, Any],
) -> float | None:
    explicit_target = metadata.get("gripper_joint_target")
    try:
        value = float(explicit_target)
    except (TypeError, ValueError):
        value = float("nan")
    if np.isfinite(value):
        return value
    if target_from_action is None or not 0 <= int(slot) < len(target_from_action):
        return None
    try:
        value = float(target_from_action[int(slot)])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def collector_sample_to_row(
    sample: CollectorSample,
    *,
    camera_video_paths: Mapping[str, str] | None = None,
    tactile_storage_refs: Mapping[str, Mapping[str, Any]] | None = None,
    inline_tactile: bool = True,
) -> tuple[dict[str, Any], dict[str, bool]]:
    action_split = _split_action(sample.action)
    observation_split = _split_observation_state(sample.observation_state)
    arm_specs = _arm_specs_from_sample_metadata(sample.metadata)
    hand_specs = _hand_specs_from_sample_metadata(sample.metadata)
    hand_active_masks = _hand_active_masks_from_specs(hand_specs)
    pose_quality = sample.metadata.get("pose_quality", {})
    if not isinstance(pose_quality, Mapping):
        pose_quality = {}
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
        "arm_action_specs_json": json.dumps(
            _to_serializable(arm_specs),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "hand_action_specs_json": json.dumps(
            _to_serializable(hand_specs),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "hand_active_mask_by_side_json": json.dumps(
            _to_serializable(hand_active_masks),
            ensure_ascii=True,
            sort_keys=True,
        ),
        "channels_json": json.dumps(
            {name: _serialize_channel(channel) for name, channel in sample.channels.items()},
            ensure_ascii=True,
            sort_keys=True,
        ),
        "faults_json": json.dumps(_to_serializable(sample.faults), ensure_ascii=True, sort_keys=True),
        "teleop.pose_quality.score": pose_quality.get("score"),
        "teleop.pose_quality.status": pose_quality.get("status"),
        "teleop.pose_quality.bad": int(bool(pose_quality.get("bad", False))),
        "teleop.pose_quality.available": int(bool(pose_quality.get("available", False))),
        "teleop.pose_quality_json": json.dumps(
            _to_serializable(pose_quality),
            ensure_ascii=True,
            sort_keys=True,
        ),
    }
    camera_expectations: dict[str, bool] = {}
    command_metadata_by_effector = sample.metadata.get("command_metadata_by_effector", {})

    for side in ("left", "right"):
        arm_state = sample.arm_states.get(side) or {}
        hand_state = sample.hand_states.get(side) or {}
        raw_tactile_ref = (
            tactile_storage_refs.get(side)
            if tactile_storage_refs is not None
            else None
        )
        tactile_ref = dict(raw_tactile_ref) if isinstance(raw_tactile_ref, Mapping) else {}
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
        row[f"observation.arm.{side}.actual_qpos"] = _vector_list(
            arm_state.get("actual_qpos"),
            length=6,
        )
        row[f"observation.arm.{side}.actual_tcp_pose"] = _pose_matrix_to_xyzquat_list(
            arm_state.get("actual_tcp_pose")
        )
        row[f"observation.arm.{side}.actual_hardware_qpos"] = _vector_list(
            arm_state.get("actual_hardware_qpos"),
            length=6,
        )
        row[f"observation.arm.{side}.actual_hardware_pose_6d"] = _vector_list(
            arm_state.get("actual_hardware_pose_6d"),
            length=6,
        )
        row[f"observation.arm.{side}.actual_hardware_refresh_time"] = arm_state.get(
            "actual_hardware_refresh_time"
        )
        row[f"observation.arm.{side}.actual_hardware_result_code"] = arm_state.get(
            "actual_hardware_result_code"
        )
        row[f"observation.arm.{side}.actual_hardware_result_text"] = arm_state.get(
            "actual_hardware_result_text"
        )
        row[f"observation.arm.{side}.actual_hardware_arm_err"] = arm_state.get(
            "actual_hardware_arm_err"
        )
        row[f"observation.arm.{side}.actual_hardware_sys_err"] = arm_state.get(
            "actual_hardware_sys_err"
        )
        row[f"observation.arm.{side}.actual_state_source"] = arm_state.get(
            "actual_state_source"
        )
        row[f"observation.arm.{side}.actual_state_timestamp"] = arm_state.get(
            "actual_state_timestamp"
        )
        row[f"observation.arm.{side}.actual_state_read_start_time"] = arm_state.get(
            "actual_state_read_start_time"
        )
        row[f"observation.arm.{side}.actual_state_read_end_time"] = arm_state.get(
            "actual_state_read_end_time"
        )
        row[f"observation.arm.{side}.actual_state_read_elapsed_s"] = arm_state.get(
            "actual_state_read_elapsed_s"
        )
        row[f"observation.arm.{side}.actual_sample_age_sec"] = arm_state.get(
            "actual_sample_age_sec"
        )
        row[f"observation.arm.{side}.actual_state_stale"] = arm_state.get(
            "actual_state_stale"
        )
        row[f"observation.arm.{side}.actual_state_error"] = arm_state.get(
            "actual_state_error"
        )
        row[f"observation.arm.{side}.actual_state_provider_error"] = arm_state.get(
            "actual_state_provider_error"
        )
        row[f"observation.wrist_camera.{side}.pose_base"] = _pose_matrix_to_xyzquat_list(
            arm_state.get("wrist_camera_pose_base")
        )
        row[f"observation.wrist_camera.{side}.timestamp"] = arm_state.get(
            "wrist_camera_pose_timestamp"
        )
        row[f"observation.wrist_camera.{side}.age_sec"] = arm_state.get(
            "wrist_camera_pose_age_sec"
        )
        row[f"observation.wrist_camera.{side}.sample_time_offset_sec"] = arm_state.get(
            "wrist_camera_pose_sample_time_offset_sec"
        )
        row[f"observation.wrist_camera.{side}.read_elapsed_s"] = arm_state.get(
            "wrist_camera_pose_read_elapsed_s"
        )
        row[f"observation.wrist_camera.{side}.source"] = arm_state.get(
            "wrist_camera_pose_source"
        )
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
        row[f"observation.hand.{side}.measured_joint_pos"] = _vector_list(
            hand_state.get("measured_joint_pos"),
            length=6,
        )
        row[f"observation.hand.{side}.commanded_joint_pos"] = _vector_list(
            hand_state.get("commanded_joint_pos"),
            length=6,
        )
        row[f"observation.hand.{side}.joint_timestamp"] = hand_state.get("joint_timestamp")
        row[f"observation.hand.{side}.tactile"] = (
            _vector_list(hand_state.get("tactile")) if inline_tactile else None
        )
        row[f"observation.hand.{side}.tactile_timestamp"] = hand_state.get("tactile_timestamp")
        row[f"observation.hand.{side}.telemetry_timestamp"] = hand_state.get("telemetry_timestamp")
        row[f"observation.hand.{side}.gripper_read_elapsed_s"] = hand_state.get(
            "gripper_read_elapsed_s"
        )
        row[f"observation.hand.{side}.tactile_read_elapsed_s"] = hand_state.get(
            "tactile_read_elapsed_s"
        )
        row[f"observation.hand.{side}.tactile_gate_wait_s"] = hand_state.get(
            "tactile_gate_wait_s"
        )
        row[f"observation.hand.{side}.telemetry_elapsed_s"] = hand_state.get(
            "telemetry_elapsed_s"
        )
        row[f"observation.hand.{side}.tactile_timings_ms_json"] = json.dumps(
            _to_serializable(hand_state.get("tactile_timings_ms", {})),
            ensure_ascii=True,
            sort_keys=True,
        )
        row[f"observation.hand.{side}.tactile_wait_for_new"] = hand_state.get(
            "tactile_wait_for_new"
        )
        row[f"observation.hand.{side}.tactile_read_hz"] = hand_state.get(
            "tactile_read_hz"
        )
        row[f"observation.hand.{side}.tactile_read_skipped"] = hand_state.get(
            "tactile_read_skipped"
        )
        row[f"observation.hand.{side}.tactile_process_read"] = hand_state.get(
            "tactile_process_read"
        )
        row[f"observation.hand.{side}.tactile_process_nice"] = hand_state.get(
            "tactile_process_nice"
        )
        row[f"observation.hand.{side}.tactile_process_cpu_affinity"] = hand_state.get(
            "tactile_process_cpu_affinity"
        )
        row[f"observation.hand.{side}.tactile_process_queue_size"] = hand_state.get(
            "tactile_process_queue_size"
        )
        row[f"observation.hand.{side}.tactile_worker_alive"] = hand_state.get(
            "tactile_worker_alive"
        )
        row[f"observation.hand.{side}.tactile_worker_receiver_alive"] = hand_state.get(
            "tactile_worker_receiver_alive"
        )
        row[f"observation.hand.{side}.tactile_parallel_read"] = hand_state.get(
            "tactile_parallel_read"
        )
        row[f"observation.hand.{side}.tactile_global_read_lock"] = hand_state.get(
            "tactile_global_read_lock"
        )
        row[f"observation.hand.{side}.tactile_compute_profile"] = hand_state.get(
            "tactile_compute_profile"
        )
        row[f"observation.hand.{side}.tactile_fields"] = list(
            hand_state.get("tactile_fields", ()) or ()
        )
        row[f"observation.hand.{side}.tactile_max_fps"] = hand_state.get(
            "tactile_max_fps"
        )
        row[f"observation.hand.{side}.tactile_denoise"] = hand_state.get(
            "tactile_denoise"
        )
        row[f"observation.hand.{side}.tactile_store_flat_vector"] = hand_state.get(
            "tactile_store_flat_vector"
        )
        row[f"observation.hand.{side}.tactile_regions"] = list(
            hand_state.get("tactile_regions", ())
        )
        row[f"observation.hand.{side}.tactile_storage_format"] = tactile_ref.get(
            "format",
            "parquet_inline" if inline_tactile and hand_state.get("tactile") is not None else None,
        )
        row[f"observation.hand.{side}.tactile_path"] = tactile_ref.get("path")
        row[f"observation.hand.{side}.tactile_frame_index"] = tactile_ref.get("frame_index")
        row[f"observation.hand.{side}.tactile_sensor_keys"] = list(
            tactile_ref.get(
                "sensor_keys",
                hand_state.get("tactile_devices", hand_state.get("tactile_regions", ())),
            )
            or ()
        )
        row[f"observation.hand.{side}.tactile_frame_ids_json"] = json.dumps(
            _to_serializable(hand_state.get("tactile_frame_ids", {})),
            ensure_ascii=True,
            sort_keys=True,
        )
        row[f"observation.hand.{side}.tactile_shapes_json"] = json.dumps(
            _to_serializable(hand_state.get("tactile_shapes", {})),
            ensure_ascii=True,
            sort_keys=True,
        )
        row[f"observation.hand.{side}.tactile_field_by_key_json"] = json.dumps(
            _to_serializable(hand_state.get("tactile_field_by_key", {})),
            ensure_ascii=True,
            sort_keys=True,
        )
        row[f"observation.hand.{side}.tactile_version"] = hand_state.get("tactile_version")
        row[f"observation.hand.{side}.tactile_raw_shape_per_sensor"] = (
            list(hand_state.get("tactile_raw_shape_per_sensor"))
            if hand_state.get("tactile_raw_shape_per_sensor") is not None
            else None
        )
        row[f"action.replay.{side}_hand_joint_target"] = _vector_list(
            hand_state.get("last_sent_command"),
            length=6,
        )
        row[f"observation.gripper.{side}.state"] = hand_state.get("gripper_state_scalar")
        row[f"observation.gripper.{side}.measured"] = hand_state.get("gripper_measured_scalar")
        row[f"observation.gripper.{side}.commanded"] = hand_state.get("gripper_commanded_scalar")
        target_from_action = action_split[f"{side}_hand"]
        gripper_slot = _gripper_slot_from_metadata(
            hand_command_metadata,
            hand_specs.get(side) if isinstance(hand_specs.get(side), Mapping) else None,
        )
        row[f"action.gripper.{side}.target"] = _gripper_target_from_action(
            target_from_action,
            slot=gripper_slot,
            metadata=hand_command_metadata,
        )
        row[f"action.gripper.{side}.source_value"] = hand_command_metadata.get("source_value")
        row[f"action.gripper.{side}.source_dim"] = hand_command_metadata.get("source_dim")
        row[f"action.gripper.{side}.output_dim"] = hand_command_metadata.get("output_dim")
        row[f"action.gripper.{side}.slot_index"] = gripper_slot
        row[f"action.gripper.{side}.action_dim"] = hand_command_metadata.get("gripper_action_dim", 1)
        row[f"action.gripper.{side}.invert"] = hand_command_metadata.get("invert")
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
        row[f"teleop.channels.{channel_name}.sender_read_elapsed_ms"] = channel.metadata.get(
            "sender_read_elapsed_ms"
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

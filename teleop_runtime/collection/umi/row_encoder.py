"""Native v2.3 row encoder for arm-free UMI samples."""

from __future__ import annotations

import json
import threading
from typing import Any, Mapping

import numpy as np

from dataset_contracts.layout import camera_video_channel
from dataset_contracts.v22 import ARM_ACTION_DIM, PolicyIOLayout

from ..arm_mode_labels import flatten_arm_mode_columns
from ..schema_v21 import (
    bind_hand_runtime_diagnostic_fields,
    bind_hand_state_snapshot_fields,
)
from ..schema_v22 import validate_strict_v22_measured_state
from .types import UmiSample


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return value


def _camera_fields(
    sample: UmiSample,
    row: dict[str, Any],
    *,
    paths: Mapping[str, str] | None,
) -> dict[str, bool]:
    expectations: dict[str, bool] = {}
    for key, match in sample.camera_frames.items():
        frame = getattr(match, "frame", match)
        expectations[str(key)] = frame is not None
        prefix = camera_video_channel(str(key))
        row[f"{prefix}.source_timestamp"] = getattr(match, "selected_timestamp", None)
        row[f"{prefix}.receive_timestamp"] = getattr(frame, "receive_timestamp", None)
        row[f"{prefix}.available_timestamp"] = getattr(frame, "available_timestamp", None)
        metadata = getattr(frame, "metadata", {}) if frame is not None else {}
        metadata = metadata if isinstance(metadata, Mapping) else {}
        row[f"{prefix}.source_clock"] = (
            None if frame is None else metadata.get("source_clock", "receiver_wall_time")
        )
        row[f"{prefix}.available_clock"] = (
            None if frame is None else metadata.get("available_clock", "local_wall_time")
        )
        row[f"{prefix}.camera_timestamp_sec"] = metadata.get("camera_timestamp_sec")
        row[f"{prefix}.timestamp_domain"] = metadata.get("timestamp_domain")
        row[f"{prefix}.sensor_timestamp_ms"] = metadata.get("sensor_timestamp_ms")
        row[f"{prefix}.sensor_frame_number"] = metadata.get("color_frame_number")
        row[f"{prefix}.age_sec"] = getattr(match, "age_sec", None)
        row[f"{prefix}.source_frame_index"] = getattr(frame, "frame_index", None)
        row[f"{prefix}.video_frame_index"] = None
        if paths is not None and key in paths:
            row[f"{prefix}.path"] = str(paths[key])
    return expectations


class UmiV23RowEncoder:
    """Pack current observations with next-observation arm pose actions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._last_action_by_episode: dict[int, np.ndarray] = {}

    def reset_episode(self, episode_index: int) -> None:
        with self._lock:
            self._last_action_by_episode.pop(int(episode_index), None)

    def encode_row(
        self,
        sample: UmiSample,
        *,
        layout: PolicyIOLayout | None,
        camera_video_paths: Mapping[str, str] | None,
        tactile_storage_refs: Mapping[str, Mapping[str, Any]] | None,
        inline_tactile: bool,
    ) -> tuple[dict[str, Any], dict[str, bool]]:
        if layout is None:
            raise ValueError("UMI v2.3 row encoding requires a PolicyIOLayout")
        if layout.action_dim != 16 or layout.state_dim != 16:
            raise ValueError(
                f"UMI layout must define 16D action/state, got {layout.action_dim}/{layout.state_dim}"
            )
        validate_strict_v22_measured_state(
            sample.hand_states,
            layout=layout,
            sample_timestamp=float(sample.timestamp),
            # UMI snapshot assembly already rejects feedback more than 1 ms
            # from the causal cutoff in the future.  Preserve that stricter
            # boundary for direct row-encoder callers as well.
            future_tolerance_sec=1e-3,
        )
        if sample.arm_action_poses is None:
            raise ValueError(
                "UMI arm action requires the next accepted observation; "
                "the recorder must align samples before row encoding"
            )
        raw_poses = {
            side: np.asarray(sample.poses[side].raw_tcp_pose7, dtype=np.float32)
            for side in ("left", "right")
        }
        action_poses = {
            side: np.asarray(
                sample.arm_action_poses[side].raw_tcp_pose7,
                dtype=np.float32,
            )
            for side in ("left", "right")
        }
        command_values = {
            side: [float(sample.gripper_commands[side])]
            for side in ("left", "right")
        }
        measured_values: dict[str, list[float]] = {}
        for side in ("left", "right"):
            measured = sample.hand_states[side].get("gripper_measured_scalar")
            measured_values[side] = [float(measured)]

        action = layout.pack_action(
            arm_pose_by_side=action_poses,
            end_effector_by_side=command_values,
            strict=True,
        ).continuous
        state = layout.pack_observation_state(
            arm_pose_by_side=raw_poses,
            end_effector_by_side=measured_values,
            strict=True,
        ).continuous
        if not np.all(np.isfinite(action)) or not np.all(np.isfinite(state)):
            raise ValueError("UMI model vectors must be finite")
        causal_initial_action = layout.pack_action(
            arm_pose_by_side=raw_poses,
            end_effector_by_side=command_values,
            strict=True,
        ).continuous
        with self._lock:
            last_action = self._last_action_by_episode.get(int(sample.episode_index))
            if last_action is None:
                last_action = causal_initial_action
            self._last_action_by_episode[int(sample.episode_index)] = action.copy()

        label_payload = sample.operator_labels.arm_mode_compatibility_payload()
        row: dict[str, Any] = {
            "episode_index": int(sample.episode_index),
            "frame_index": int(sample.frame_index),
            "timestamp": float(sample.timestamp),
            "index": int(sample.frame_index),
            "dexteleop.dataset_schema": "dexteleop_dataset_v2.3",
            "policy_io_layout_hash": layout.layout_hash,
            "action": action.tolist(),
            "action.arm.pose": action[:ARM_ACTION_DIM].tolist(),
            "action.left_arm.pose": action_poses["left"].tolist(),
            "action.right_arm.pose": action_poses["right"].tolist(),
            "action.arm.temporal_alignment": "next_accepted_observation",
            "action.arm.source_frame_index": int(
                sample.arm_action_source_frame_index
            ),
            "action.arm.source_timestamp": float(sample.arm_action_source_timestamp),
            "observation.state": state.tolist(),
            "observation.arm.pose": state[:ARM_ACTION_DIM].tolist(),
            "observation.arm.qpos": [float("nan")] * 12,
            "observation.arm.qpos_valid": [False] * 12,
            "observation.last_action": last_action.tolist(),
            "observation.left_arm.pose": raw_poses["left"].tolist(),
            "observation.right_arm.pose": raw_poses["right"].tolist(),
            "action.end_effector.left.target": command_values["left"],
            "action.end_effector.right.target": command_values["right"],
            "observation.end_effector.left.state": measured_values["left"],
            "observation.end_effector.right.state": measured_values["right"],
            "teleop.arm_mode_json": json.dumps(
                _jsonable(label_payload), ensure_ascii=True, sort_keys=True
            ),
            "teleop.operator_labels_json": json.dumps(
                _jsonable(sample.operator_labels), ensure_ascii=True, sort_keys=True
            ),
            "sample_metadata_json": json.dumps(
                _jsonable(sample.metadata), ensure_ascii=True, sort_keys=True
            ),
            "faults_json": json.dumps(_jsonable(sample.faults), ensure_ascii=True, sort_keys=True),
            "active_sides": ["left", "right"],
            "camera_keys": sorted(str(key) for key in sample.camera_frames),
            "fault_codes": [],
        }
        row.update(flatten_arm_mode_columns(label_payload))

        for side in ("left", "right"):
            pose = sample.poses[side]
            prefix = f"observation.arm.{side}."
            row.update(
                {
                    f"{prefix}actual_tcp_pose": pose.raw_tcp_pose7.tolist(),
                    f"{prefix}actual_tcp_pose_source": "vive_tracker_tcp_extrinsic",
                    f"{prefix}actual_state_read_success": True,
                    f"{prefix}actual_state_valid": True,
                    f"{prefix}actual_state_stale": False,
                    f"{prefix}causal_measured_state_eligible": True,
                    f"{prefix}causal_measured_state_rejection_reason": "",
                    f"{prefix}actual_state_sequence": int(pose.sequence),
                    f"{prefix}actual_state_timestamp": float(pose.source_timestamp),
                    f"{prefix}source_timestamp": float(pose.source_timestamp),
                    f"{prefix}available_timestamp": float(pose.available_timestamp),
                    f"{prefix}source_clock": pose.source_clock,
                    f"{prefix}available_clock": pose.available_clock,
                    f"{prefix}state_source": "vive_umi",
                }
            )
            hand = sample.hand_states[side]
            row[f"observation.hand.{side}.gripper_commanded_scalar"] = float(
                sample.gripper_commands[side]
            )
            row[f"observation.hand.{side}.gripper_measured_scalar"] = float(
                hand["gripper_measured_scalar"]
            )
            row[f"observation.hand.{side}.joint_timestamp"] = hand.get("joint_timestamp")
            bind_hand_state_snapshot_fields(row, side=side, hand_state=hand)
            bind_hand_runtime_diagnostic_fields(row, side=side, hand_state=hand)
            tactile_ref = (
                tactile_storage_refs.get(side)
                if tactile_storage_refs is not None
                else None
            )
            row[f"observation.hand.{side}.tactile_storage_ref_json"] = json.dumps(
                _jsonable(tactile_ref or {}), ensure_ascii=True, sort_keys=True
            )
            if inline_tactile:
                tactile = hand.get("tactile")
                row[f"observation.hand.{side}.tactile"] = (
                    None if tactile is None else np.asarray(tactile).reshape(-1).tolist()
                )
        expectations = _camera_fields(
            sample, row, paths=camera_video_paths
        )
        return row, expectations


__all__ = ["UmiV23RowEncoder"]

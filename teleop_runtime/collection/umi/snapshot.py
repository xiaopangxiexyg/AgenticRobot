"""Causal T0 snapshot assembly for UMI sources."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np

from ..operator_labels import OperatorLabelStateMachine
from ..hand_state_polling import hand_state_tactile_arrays
from .config import UmiCollectionConfig
from .pose_adapter import UmiPoseAdapterPort
from .topology import CAMERA_KEYS, SIDES
from .types import SnapshotRejection, UmiPoseMeasurement, UmiSample


CameraSelector = Callable[[float], Any | None]


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _frame_timestamp(value: Any) -> float | None:
    for name in ("selected_timestamp", "available_timestamp", "receive_timestamp", "timestamp"):
        result = _finite(getattr(value, name, None))
        if result is not None:
            return result
    frame = getattr(value, "frame", None)
    if frame is not None:
        for name in ("available_timestamp", "receive_timestamp", "source_timestamp"):
            result = _finite(getattr(frame, name, None))
            if result is not None:
                return result
    return None


def _gripper_measurement_timing(
    state: Mapping[str, Any],
    *,
    cutoff_timestamp: float,
    max_age_sec: float,
) -> tuple[float | None, str | None]:
    joint_timestamp = _finite(state.get("joint_timestamp"))
    if joint_timestamp is None:
        return None, "missing_gripper_measurement_timestamp"
    age = float(cutoff_timestamp) - joint_timestamp
    if age < -1e-3:
        return age, f"future_gripper_measurement:{age:.3f}"
    if age > float(max_age_sec):
        return age, f"stale_gripper_measurement:{age:.3f}"
    return age, None


class UmiSnapshotAssembler:
    def __init__(
        self,
        *,
        config: UmiCollectionConfig,
        pose_adapter: UmiPoseAdapterPort,
        vive_source: Any,
        camera_selectors: Mapping[str, CameraSelector],
        hand_state_poller: Any,
        gripper_command_provider: Callable[[], Mapping[str, float]],
        labels: OperatorLabelStateMachine,
    ) -> None:
        self.config = config
        self.pose_adapter = pose_adapter
        self.vive_source = vive_source
        self.camera_selectors = dict(camera_selectors)
        self.hand_state_poller = hand_state_poller
        self.gripper_command_provider = gripper_command_provider
        self.labels = labels

    def assemble(
        self,
        *,
        episode_index: int,
        frame_index: int,
        timestamp: float,
    ) -> UmiSample | SnapshotRejection:
        t0 = float(timestamp)
        reasons: list[str] = []
        channels = self.vive_source.get_latest_channels_at_or_before(t0)
        poses: dict[str, UmiPoseMeasurement] = {}
        vive_rotation_diagnostics: dict[str, dict[str, float | bool]] = {}
        for side in SIDES:
            side_config = self.config.side(side)
            channel = channels.get(side_config.tracker_channel)
            if channel is None:
                reasons.append(f"{side}:missing_vive")
                continue
            metadata = dict(getattr(channel, "metadata", {}) or {})
            source_timestamp = _finite(metadata.get("source_timestamp"))
            available_timestamp = _finite(metadata.get("available_timestamp"))
            sequence = metadata.get("seq")
            if not bool(getattr(channel, "valid", False)):
                reasons.append(f"{side}:invalid_vive")
                continue
            if source_timestamp is None or available_timestamp is None:
                reasons.append(f"{side}:missing_vive_timestamp")
                continue
            if source_timestamp > t0 + 1e-6 or available_timestamp > t0 + 1e-6:
                reasons.append(f"{side}:future_vive")
                continue
            age = t0 - source_timestamp
            if age > self.config.timing.vive_max_age_sec:
                reasons.append(f"{side}:stale_vive:{age:.3f}")
                continue
            try:
                sequence_int = int(sequence)
                normalized_tracker, rotation_diagnostics = (
                    self.pose_adapter.normalize_tracker_pose(channel.value)
                )
                raw_pose = self.pose_adapter.raw_pose7(side, normalized_tracker)
                desk_pose = self.pose_adapter.desk_pose7(side, normalized_tracker)
                vive_rotation_diagnostics[side] = rotation_diagnostics
                poses[side] = UmiPoseMeasurement(
                    side=side,
                    raw_tcp_pose7=raw_pose,
                    desk_tcp_pose7=desk_pose,
                    source_timestamp=source_timestamp,
                    available_timestamp=available_timestamp,
                    source_clock=str(metadata.get("source_clock") or ""),
                    available_clock=str(metadata.get("available_clock") or ""),
                    sequence=sequence_int,
                    tracker_name=str(metadata.get("tracker_name") or side_config.tracker_name),
                    tracker_role=str(metadata.get("tracker_role") or side_config.tracker_role),
                    age_sec=age,
                )
            except (TypeError, ValueError) as exc:
                reasons.append(f"{side}:invalid_vive_pose:{exc}")

        camera_frames: dict[str, Any] = {}
        for camera_key in CAMERA_KEYS:
            if camera_key in self.config.ignored_camera_keys:
                continue
            selector = self.camera_selectors.get(camera_key)
            match = None if selector is None else selector(t0)
            selected_timestamp = _frame_timestamp(match)
            if match is None or getattr(match, "frame", match) is None:
                reasons.append(f"{camera_key}:missing_camera")
                continue
            if selected_timestamp is None or selected_timestamp > t0 + 1e-6:
                reasons.append(f"{camera_key}:noncausal_camera")
                continue
            age = t0 - selected_timestamp
            if age > self.config.timing.camera_max_age_sec:
                reasons.append(f"{camera_key}:stale_camera:{age:.3f}")
                continue
            camera_frames[camera_key] = deepcopy(match)

        hand_states = {
            str(side): dict(state)
            for side, state in self.hand_state_poller.latest_at_or_before(t0).items()
            if isinstance(state, Mapping)
        }
        for side in SIDES:
            state = hand_states.get(side)
            if not isinstance(state, Mapping):
                reasons.append(f"{side}:missing_gripper_state")
                continue
            measured = _finite(state.get("gripper_measured_scalar"))
            if measured is None:
                reasons.append(f"{side}:missing_gripper_measurement")
            joint_age, timing_rejection = _gripper_measurement_timing(
                state,
                cutoff_timestamp=t0,
                max_age_sec=self.config.timing.gripper_max_age_sec,
            )
            state["joint_age_sec"] = joint_age
            if timing_rejection is not None:
                reasons.append(f"{side}:{timing_rejection}")
            tactile_age = _finite(state.get("tactile_snapshot_age_sec"))
            if side not in self.config.ignored_tactile_keys and self.config.tactile_required and (
                not hand_state_tactile_arrays(state)
                or tactile_age is None
                or tactile_age > self.config.timing.tactile_max_age_sec
            ):
                reasons.append(f"{side}:missing_or_stale_tactile")
            if side in self.config.ignored_tactile_keys:
                state["tactile"] = None
                state["tactile_dict"] = None
                state["tactile_timestamp"] = None
                state["tactile_frame_ids"] = {}
                state["tactile_shapes"] = {}
                state["tactile_devices"] = []
                state["tactile_ignored_by_dataset_contract"] = True

        commands = {side: float(value) for side, value in self.gripper_command_provider().items()}
        if set(commands) != set(SIDES) or any(
            not np.isfinite(value) or not 0.0 <= value <= 1.0
            for value in commands.values()
        ):
            reasons.append("invalid_gripper_commands")

        if reasons:
            return SnapshotRejection(timestamp=t0, reasons=tuple(reasons))
        return UmiSample(
            episode_index=int(episode_index),
            frame_index=int(frame_index),
            timestamp=t0,
            poses=poses,
            gripper_commands=commands,
            hand_states=deepcopy(hand_states),
            operator_labels=self.labels.snapshot(),
            camera_frames=camera_frames,
            channels=deepcopy(channels),
            metadata={
                "embodiment": "umi_dual_wrist",
                "embodiment_has_arm": False,
                "pose_action_semantics": "current_measured_tcp_absolute",
                "vive_rotation_normalization_by_side": vive_rotation_diagnostics,
            },
        )


__all__ = ["UmiSnapshotAssembler"]

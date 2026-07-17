"""Causal T0 snapshot assembly for UMI sources."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

import numpy as np

from ..operator_labels import OperatorLabelStateMachine
from .config import UmiCollectionConfig
from .pose_adapter import UmiPoseAdapter
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


class UmiSnapshotAssembler:
    def __init__(
        self,
        *,
        config: UmiCollectionConfig,
        pose_adapter: UmiPoseAdapter,
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
        for side in ("left", "right"):
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
                raw_pose = self.pose_adapter.raw_pose7(side, channel.value)
                desk_pose = self.pose_adapter.desk_pose7(side, channel.value)
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
        for camera_key in ("left_wrist", "right_wrist"):
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

        hand_states = self.hand_state_poller.latest_at_or_before(t0)
        for side in ("left", "right"):
            state = hand_states.get(side)
            if not isinstance(state, Mapping):
                reasons.append(f"{side}:missing_gripper_state")
                continue
            measured = _finite(state.get("gripper_measured_scalar"))
            telemetry_age = _finite(state.get("telemetry_snapshot_age_sec"))
            if measured is None:
                reasons.append(f"{side}:missing_gripper_measurement")
            if telemetry_age is None or telemetry_age > self.config.timing.gripper_max_age_sec:
                reasons.append(f"{side}:stale_gripper")
            tactile_age = _finite(state.get("tactile_snapshot_age_sec"))
            if self.config.tactile_required and (
                not state.get("tactile_dict")
                or tactile_age is None
                or tactile_age > self.config.timing.tactile_max_age_sec
            ):
                reasons.append(f"{side}:missing_or_stale_tactile")

        commands = {side: float(value) for side, value in self.gripper_command_provider().items()}
        if set(commands) != {"left", "right"} or any(
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
            },
        )


__all__ = ["UmiSnapshotAssembler"]

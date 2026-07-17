"""UMI control and recording state mapped onto the existing formal GUI."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Iterable, MutableMapping

import numpy as np

from robot_motion.runtime.contracts import TeleopObservation, merge_teleop_commands

from ...gui.calibration_web_ui import CalibrationWebUi
from ...real_teleop.gui_tracker_status import raw_tracker_pose_state_from_channels
from ..operator_labels import OperatorLabelStateMachine
from .calibration import UmiCalibrationBinding
from .components import UmiComponents
from .config import UmiCollectionConfig
from .recorder import UmiDatasetRecorder
from .snapshot import UmiSnapshotAssembler
from .types import SnapshotRejection


class UmiCollectionController:
    def __init__(
        self,
        *,
        config: UmiCollectionConfig,
        calibration: UmiCalibrationBinding,
        components: UmiComponents,
        recorder: UmiDatasetRecorder,
        ui: CalibrationWebUi,
        labels: OperatorLabelStateMachine,
        snapshot_assembler: UmiSnapshotAssembler,
        last_commands: MutableMapping[str, float] | None = None,
    ) -> None:
        self.config = config
        self.calibration = calibration
        self.components = components
        self.recorder = recorder
        self.ui = ui
        self.labels = labels
        self.snapshot_assembler = snapshot_assembler
        self.should_quit = False
        self.last_commands = (
            last_commands
            if last_commands is not None
            else {"left": 0.0, "right": 0.0}
        )
        self.last_rejection: tuple[str, ...] = ()
        self._next_sample_at = time.perf_counter()
        self._selected_label_side = {"fix": "left", "init": "left"}

    def _events(self) -> list[tuple[str, str, float]]:
        now = time.time()
        events = [(str(event), "web_gui", now) for event in self.ui.consume_events()]
        if self.components.foot_pad is not None:
            events.extend(
                (str(event), "foot_pad", now)
                for event in self.components.foot_pad.consume_events()
            )
        return events

    def _handle_events(self, events: Iterable[tuple[str, str, float]]) -> None:
        for event, source, timestamp in events:
            if event == "q":
                self.should_quit = True
            elif event.startswith("fix_arm_select:"):
                side = event.split(":", 1)[1]
                if side in {"left", "right"}:
                    self._selected_label_side["fix"] = side
            elif event.startswith("init_arm_select:"):
                side = event.split(":", 1)[1]
                if side in {"left", "right"}:
                    self._selected_label_side["init"] = side
            elif event in {"c", "fix_arm_toggle"}:
                self.labels.toggle(
                    self._selected_label_side["fix"], "fixed_hold",
                    source=source, timestamp=timestamp,
                )
            elif event in {"g", "init_arm_toggle"}:
                self.labels.toggle(
                    self._selected_label_side["init"], "init_pose",
                    source=source, timestamp=timestamp,
                )
            elif event in {"fix_arm_left", "fix_arm_right"}:
                self.labels.toggle(
                    event.rsplit("_", 1)[1], "fixed_hold",
                    source=source, timestamp=timestamp,
                )
            elif event in {"init_arm_left", "init_arm_right"}:
                self.labels.toggle(
                    event.rsplit("_", 1)[1], "init_pose",
                    source=source, timestamp=timestamp,
                )
            elif event == "labels_reset":
                self.labels.reset(source=source, timestamp=timestamp)
            elif event == "episode_start" or event.startswith("episode_start:"):
                if not self.recorder.recording:
                    self.calibration.require_valid(self.config)
                    self.labels.reset(source="episode_start", timestamp=timestamp)
                    self.recorder.start_episode(
                        {
                            "embodiment": "umi_dual_wrist",
                            "embodiment_has_arm": False,
                            "calibration_sha256": self.calibration.sha256,
                            "tcp_extrinsic_by_side": {
                                side: {
                                    "T_tcp_to_tracker": self.config.side(side).tcp_transform.tolist(),
                                    "status": self.config.side(side).tcp_extrinsic_status,
                                }
                                for side in ("left", "right")
                            },
                        }
                    )
            elif event in {"episode_stop", "episode_cancel"}:
                if self.recorder.recording:
                    self.recorder.stop_episode(reason=event)
                    self.labels.reset(source=event, timestamp=timestamp)

    def _update_grippers(self, now: float) -> tuple[bool, str]:
        channels = self.components.angle_trigger.get_latest_channels()
        required = {f"{side}_angle_trigger" for side in ("left", "right")}
        if not required.issubset(channels):
            return False, "missing_angle_trigger"
        for name in required:
            channel = channels[name]
            age = now - float(channel.timestamp)
            if (
                not channel.valid
                or age < -1e-3
                or age > self.config.timing.angle_trigger_max_age_sec
            ):
                return False, f"invalid_or_stale_angle_trigger:{name}:{age:.3f}"
        observation = TeleopObservation(channels=channels, timestamp=now)
        merged = merge_teleop_commands(
            [
                self.components.retargeters[side].retarget(observation)
                for side in ("left", "right")
            ]
        )
        if set(merged.commands) != {"left_hand", "right_hand"}:
            return False, "incomplete_gripper_command"
        pending: dict[str, float] = {}
        for side in ("left", "right"):
            scalar = float(
                merged.commands[f"{side}_hand"].metadata["gripper_joint_target"]
            )
            if not np.isfinite(scalar):
                return False, f"nonfinite_gripper_command:{side}"
            pending[side] = scalar
        self.components.rig.send_commands(merged)
        self.last_commands.update(pending)
        return True, ""

    def _sample_if_due(self, wall_now: float) -> None:
        monotonic_now = time.perf_counter()
        if monotonic_now < self._next_sample_at:
            return
        period = 1.0 / self.config.timing.sample_hz
        self._next_sample_at = max(self._next_sample_at + period, monotonic_now)
        if not self.recorder.recording:
            return
        assert self.recorder.episode_index is not None
        result = self.snapshot_assembler.assemble(
            episode_index=self.recorder.episode_index,
            frame_index=self.recorder.frame_count,
            timestamp=wall_now,
        )
        if isinstance(result, SnapshotRejection):
            self.last_rejection = result.reasons
            return
        self.last_rejection = ()
        self.recorder.submit(result)

    def _publish(self, now: float, *, gripper_ready: bool, gripper_reason: str) -> None:
        remaining = (
            self.calibration.expires_at - datetime.now(timezone.utc)
        ).total_seconds()
        label_state = self.labels.snapshot()
        left_label = label_state.labels_by_side["left"]
        right_label = label_state.labels_by_side["right"]
        vive_channels = self.components.vive.get_latest_channels_at_or_before(now)
        gui_vive_channels = {
            f"{side}_wrist_pose": vive_channels.get(
                self.config.side(side).tracker_channel
            )
            for side in ("left", "right")
        }
        hand_states = self.components.hand_state_poller.latest_at_or_before(now)
        angle_channels = self.components.angle_trigger.get_latest_channels()
        self.ui.publish(
            ui_mode="umi_collection",
            stage="UMI 数采中" if self.recorder.recording else "UMI 就绪",
            control_hint=gripper_reason,
            teleop_running=True,
            home_ready=True,
            episode_recording_available=True,
            episode_recording=self.recorder.recording,
            episode_dir=str(self.config.dataset_root),
            episode_frame_count=self.recorder.frame_count,
            episode_sample_hz=self.config.timing.sample_hz,
            episode_target_hz=self.config.timing.sample_hz,
            episode_event_status="recording" if self.recorder.recording else "ready",
            episode_event_message=gripper_reason,
            episode_last_episode_index=self.recorder.episode_index,
            vive_calibration_profile_ready=remaining >= 0.0,
            vive_calibration_profile_loaded=True,
            vive_calibration_profile_path=str(self.calibration.path),
            vive_calibration_profile_status=(
                f"expires={self.calibration.expires_at.isoformat()} "
                f"remaining={max(0.0, remaining):.1f}s hash={self.calibration.sha256}"
            ),
            raw_tracker_poses=raw_tracker_pose_state_from_channels(
                gui_vive_channels,
                active_sides=("left", "right"),
                now=now,
            ),
            camera_status={
                key: camera.device_info()
                for key, camera in self.components.cameras.items()
            },
            gripper_control_available=True,
            gripper_control_enabled=gripper_ready,
            gripper_status={
                side: {
                    "configured": True,
                    "is_connected": self.components.grippers[side].is_ready(),
                    "input_source": "angle_trigger",
                    "q6_control_available": True,
                    "q6_valid": bool(
                        angle_channels.get(f"{side}_angle_trigger")
                        and angle_channels[f"{side}_angle_trigger"].valid
                    ),
                    "q6_age_sec": (
                        None
                        if angle_channels.get(f"{side}_angle_trigger") is None
                        else now - angle_channels[f"{side}_angle_trigger"].timestamp
                    ),
                    "q6_source_dim": None,
                    "q6_source_value": (
                        None
                        if angle_channels.get(f"{side}_angle_trigger") is None
                        else float(angle_channels[f"{side}_angle_trigger"].value)
                    ),
                    "q6_target_preview": self.last_commands[side],
                    "q6_values": [],
                    "q6_command": {
                        "status": "sent" if gripper_ready else "blocked",
                        "sent": gripper_ready,
                        "target": self.last_commands[side],
                        "error": gripper_reason,
                    },
                    "commanded": self.last_commands[side],
                    "measured": hand_states.get(side, {}).get("gripper_measured_scalar"),
                    "state": hand_states.get(side, {}).get("gripper_state_scalar"),
                    "max_action": 1.0,
                    "speed": self.components.grippers[side].config.speed,
                    "torque": self.components.grippers[side].config.torque,
                    "tactile_devices": list(
                        hand_states.get(side, {}).get("tactile_devices", ())
                    ),
                    "telemetry_error": hand_states.get(side, {}).get("telemetry_error"),
                }
                for side in ("left", "right")
            },
            tactile_camera_status={
                side: {
                    "timestamp": state.get("tactile_timestamp"),
                    "devices": list(state.get("tactile_devices", ())),
                }
                for side, state in hand_states.items()
            },
            visualization_status=(
                f"last_rejection={list(self.last_rejection)} "
                f"writer={self.recorder.stats()}"
            ),
            fix_arm_available=True,
            fix_arm_selected_side=self._selected_label_side["fix"],
            fix_arm_fixed_side=(
                "left" if left_label == "fixed_hold" else
                "right" if right_label == "fixed_hold" else ""
            ),
            fix_arm_active=(
                left_label == "fixed_hold" or right_label == "fixed_hold"
            ),
            fix_arm_status=f"UMI annotation only: left={left_label}, right={right_label}",
            init_arm_available=True,
            init_arm_selected_side=self._selected_label_side["init"],
            init_arm_active_side=(
                "left" if left_label == "init_pose" else
                "right" if right_label == "init_pose" else ""
            ),
            init_arm_active=(
                left_label == "init_pose" or right_label == "init_pose"
            ),
            init_arm_status=f"UMI annotation only: left={left_label}, right={right_label}",
            special_arm_control_mode="manual",
            active_controls=[
                "episode_start", "episode_stop", "q", "c", "g",
                "fix_arm_toggle", "init_arm_toggle",
                "fix_arm_select", "init_arm_select",
            ],
        )

    def run(self) -> None:
        period = 1.0 / self.config.timing.control_hz
        last_publish = 0.0
        while not self.should_quit:
            started = time.perf_counter()
            wall_now = time.time()
            self._handle_events(self._events())
            try:
                self.calibration.require_valid(self.config)
            except RuntimeError:
                if self.recorder.recording:
                    self.recorder.stop_episode(reason="calibration_expired")
                raise
            gripper_ready, gripper_reason = self._update_grippers(wall_now)
            if gripper_ready:
                self._sample_if_due(wall_now)
            elif self.recorder.recording:
                self.last_rejection = (gripper_reason,)
            if wall_now - last_publish >= 0.2:
                self._publish(
                    wall_now,
                    gripper_ready=gripper_ready,
                    gripper_reason=gripper_reason,
                )
                last_publish = wall_now
            remaining = period - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)


__all__ = ["UmiCollectionController"]

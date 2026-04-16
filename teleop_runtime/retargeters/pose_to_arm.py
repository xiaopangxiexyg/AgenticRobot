from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..calibration import TeleopCalibration
from ..clutch import DeltaClutchController
from ..interfaces import Retargeter
from ..registry import register_retargeter
from ..schemas import EffectorCommand, TeleopCommand, TeleopObservation
from ..transforms import as_pose_matrix, invert_pose


def _rotation_about_z(yaw_rad: float) -> np.ndarray:
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    return np.array(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )


@dataclass
class PoseToArmRetargeterConfig:
    side: str
    input_channel_name: str
    effector_name: str | None = None
    tracker_role: str | None = None
    clutch_channel_names: tuple[str, ...] = ()
    control_mode: str = "delta_clutch"
    orientation_mode: str = "full"
    translation_scale: float = 1.0
    workspace_clip: bool = True
    output_frame_name: str = "arm_base"

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if not self.clutch_channel_names:
            self.clutch_channel_names = (f"{self.side}_clutch_button", "clutch_button")


@register_retargeter("pose_to_arm")
class PoseToArmRetargeter(Retargeter):
    def __init__(
        self,
        config: PoseToArmRetargeterConfig,
        *,
        clutch_controller: DeltaClutchController | None = None,
        fixed_orientation_desk: np.ndarray | None = None,
    ):
        self.config = config
        self.clutch_controller = (
            clutch_controller if clutch_controller is not None else DeltaClutchController()
        )
        self.fixed_orientation_desk = None
        if fixed_orientation_desk is not None:
            orientation = np.asarray(fixed_orientation_desk, dtype=float)
            if orientation.shape != (3, 3):
                raise ValueError(
                    "fixed_orientation_desk must have shape (3, 3) when provided."
                )
            self.fixed_orientation_desk = orientation.copy()
        self._last_target_pose_desk: np.ndarray | None = None
        self._home_human_pose_desk: np.ndarray | None = None
        self._home_robot_pose_desk: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"pose_to_arm_{self.config.side}"

    def set_home_anchors(self, *, human_pose_desk: np.ndarray, robot_pose_desk: np.ndarray) -> None:
        self._home_human_pose_desk = as_pose_matrix(human_pose_desk)
        self._home_robot_pose_desk = as_pose_matrix(robot_pose_desk)
        self._last_target_pose_desk = self._home_robot_pose_desk.copy()

    def clear_home_anchors(self) -> None:
        self._home_human_pose_desk = None
        self._home_robot_pose_desk = None
        self._last_target_pose_desk = None

    def retarget(
        self, observation: TeleopObservation, calibration: TeleopCalibration | None = None
    ) -> TeleopCommand:
        command = TeleopCommand(timestamp=observation.timestamp)
        channel = observation.get_channel(self.config.input_channel_name)
        if channel is None or not channel.valid:
            command.metadata["skipped"] = self.config.input_channel_name
            return command
        if calibration is None:
            raise RuntimeError(f"{self.name} requires a TeleopCalibration instance.")

        tcp_mode = str(observation.metadata.get("tcp_mode", "hand_base"))
        target_tcp_frame = calibration.target_frame_name(self.config.side, tcp_mode)
        pose_desk, pose_operator = self._resolve_pose_desk_and_operator(
            channel=channel,
            calibration=calibration,
        )

        clutch_pressed = self._resolve_clutch_pressed(observation)
        if self.config.control_mode == "absolute":
            target_pose_desk = self._apply_absolute_mapping(pose_desk)
        elif self.config.control_mode in {"delta_clutch", "delta_clutch_scaled"}:
            if self._last_target_pose_desk is None and not clutch_pressed:
                command.metadata.update(
                    {
                        "skipped": "waiting_for_clutch_anchor",
                        "input_channel": self.config.input_channel_name,
                    }
                )
                return command
            robot_anchor = (
                self._last_target_pose_desk.copy()
                if self._last_target_pose_desk is not None
                else self._apply_absolute_mapping(pose_desk)
            )
            human_reference = self._apply_absolute_mapping(pose_desk)
            target_pose_desk = self.clutch_controller.update(
                side=self.config.side,
                clutch_pressed=clutch_pressed,
                human_pose_reference=human_reference,
                robot_pose_target=robot_anchor,
                timestamp=float(channel.timestamp),
                scale=self.config.translation_scale,
            )
        else:
            raise ValueError(f"Unsupported control_mode: {self.config.control_mode}")

        target_pose_desk = self._apply_orientation_policy(target_pose_desk)
        if self.config.workspace_clip:
            target_pose_desk = calibration.clip_pose_to_workspace(
                target_pose_desk,
                side=self.config.side,
            )

        self._last_target_pose_desk = target_pose_desk.copy()
        target_pose_base = calibration.desk_to_base(target_pose_desk, side=self.config.side)

        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="cartesian_pose",
                value=target_pose_base,
                frame_name=self.config.output_frame_name,
                metadata={
                    "tcp_mode": tcp_mode,
                    "tcp_frame": target_tcp_frame,
                    "control_mode": self.config.control_mode,
                    "orientation_mode": self.config.orientation_mode,
                    "input_channel": self.config.input_channel_name,
                    "tracker_role": self._resolve_tracker_role(channel),
                    "clutch_pressed": clutch_pressed,
                    "target_pose_operator": pose_operator,
                    "target_pose_desk": target_pose_desk,
                },
            )
        )
        return command

    def _resolve_clutch_pressed(self, observation: TeleopObservation) -> bool:
        for channel_name in self.config.clutch_channel_names:
            channel = observation.get_channel(channel_name)
            if channel is None:
                continue
            return bool(channel.value)
        return False

    def _resolve_tracker_role(self, channel) -> str | None:
        if self.config.tracker_role is not None:
            return self.config.tracker_role
        metadata = channel.metadata or {}
        tracker_role = metadata.get("tracker_role")
        if tracker_role:
            return str(tracker_role)
        tracker_name = metadata.get("tracker_name")
        return str(tracker_name) if tracker_name else None

    def _resolve_pose_desk_and_operator(
        self,
        *,
        channel,
        calibration: TeleopCalibration,
    ) -> tuple[np.ndarray, np.ndarray]:
        pose = as_pose_matrix(channel.value)
        frame_name = channel.frame_name or ""
        tracker_role = self._resolve_tracker_role(channel)

        if frame_name in {"vive_world", "world"}:
            pose_operator = calibration.world_to_operator(pose, tracker_role=tracker_role)
            pose_desk = calibration.operator_to_desk(pose_operator)
            return pose_desk, pose_operator
        if frame_name.endswith("_operator") or frame_name == "operator":
            pose_operator = pose
            pose_desk = calibration.operator_to_desk(pose_operator)
            return pose_desk, pose_operator
        if frame_name in {calibration.desk_frame.name, "desk"}:
            pose_desk = pose
            if calibration.T_O_D is None:
                raise RuntimeError("Cannot recover operator-frame pose without T_O_D calibration.")
            pose_operator = calibration.T_O_D @ pose_desk
            return pose_desk, pose_operator
        raise ValueError(
            f"{self.name} does not know how to map input frame '{frame_name or '<none>'}' to desk."
        )

    def _apply_absolute_mapping(self, pose_desk: np.ndarray) -> np.ndarray:
        current_human_pose = as_pose_matrix(pose_desk)
        if self._home_human_pose_desk is None or self._home_robot_pose_desk is None:
            target = current_human_pose.copy()
            target[:3, 3] = target[:3, 3] * float(self.config.translation_scale)
            return target

        relative_pose = invert_pose(self._home_human_pose_desk) @ current_human_pose
        relative_pose[:3, 3] = relative_pose[:3, 3] * float(self.config.translation_scale)
        return self._home_robot_pose_desk @ relative_pose

    def _apply_orientation_policy(self, pose_desk: np.ndarray) -> np.ndarray:
        target = as_pose_matrix(pose_desk)
        if self.config.orientation_mode == "full":
            return target
        if self.config.orientation_mode == "position_only":
            if self._last_target_pose_desk is not None:
                target[:3, :3] = self._last_target_pose_desk[:3, :3]
            return target
        if self.config.orientation_mode == "fixed_orientation":
            if self.fixed_orientation_desk is None:
                raise RuntimeError(
                    f"{self.name} uses fixed_orientation but fixed_orientation_desk is not set."
                )
            target[:3, :3] = self.fixed_orientation_desk
            return target
        if self.config.orientation_mode == "yaw_only":
            yaw_rad = float(np.arctan2(target[1, 0], target[0, 0]))
            target[:3, :3] = _rotation_about_z(yaw_rad)
            return target
        raise ValueError(f"Unsupported orientation_mode: {self.config.orientation_mode}")

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from robot_motion.action import ArmActionSpec

from ..arm_frames import TOOL_TCP_FRAME
from ..calibration import TeleopCalibration
from ..hand_tcp_alignment import apply_local_pose_rotation
from ..interfaces import Retargeter
from ..registry import register_retargeter
from ..schemas import EffectorCommand, TeleopCommand, TeleopObservation
from ..sources.pose.vive_motion import (
    TrackerMotionIntentConfig,
    TrackerMotionIntentStateMachine,
)
from ..transforms import (
    as_pose_matrix,
    project_near_rigid_pose,
    rotation_is_orthonormal,
)


DEFAULT_RETARGET_CONTROL_MODE = "absolute"


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
    # Logical role used to look up the calibration offset.  The observation
    # metadata still identifies the upstream tracker that produced the sample.
    tracker_role: str | None = None
    control_mode: str = DEFAULT_RETARGET_CONTROL_MODE
    orientation_mode: str = "full"
    translation_scale: float = 1.0
    workspace_clip: bool = True
    output_frame_name: str = "arm_base"
    input_local_translation_offset: np.ndarray | Sequence[float] | None = None
    input_local_rotation_offset: np.ndarray | Sequence[float] | None = None
    tracker_motion_intent_enabled: bool = False
    tracker_motion_intent_config: TrackerMotionIntentConfig = field(
        default_factory=TrackerMotionIntentConfig
    )

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if self.control_mode != DEFAULT_RETARGET_CONTROL_MODE:
            raise ValueError(
                "delta_clutch has been hard-archived; PoseToArmRetargeter only supports "
                f"{DEFAULT_RETARGET_CONTROL_MODE!r}."
            )
        if self.input_local_translation_offset is not None:
            translation = np.asarray(
                self.input_local_translation_offset,
                dtype=float,
            ).reshape(3)
            if not np.all(np.isfinite(translation)):
                raise ValueError("input_local_translation_offset must contain finite values.")
            self.input_local_translation_offset = translation.copy()
        if self.input_local_rotation_offset is not None:
            rotation = np.asarray(self.input_local_rotation_offset, dtype=float).reshape(3, 3)
            if not rotation_is_orthonormal(rotation):
                raise ValueError(
                    "input_local_rotation_offset must be an orthonormal 3x3 rotation matrix."
                )
            self.input_local_rotation_offset = rotation.copy()


@register_retargeter("pose_to_arm")
class PoseToArmRetargeter(Retargeter):
    def __init__(
        self,
        config: PoseToArmRetargeterConfig,
        *,
        fixed_orientation_desk: np.ndarray | None = None,
    ):
        self.config = config
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
        if self.config.input_local_translation_offset is None:
            self._input_local_translation_offset = None
        else:
            self._input_local_translation_offset = np.asarray(
                self.config.input_local_translation_offset,
                dtype=float,
            ).reshape(3).copy()
        if self.config.input_local_rotation_offset is None:
            self._input_local_rotation_offset = None
        else:
            self._input_local_rotation_offset = np.asarray(
                self.config.input_local_rotation_offset,
                dtype=float,
            ).reshape(3, 3).copy()
        self._tracker_motion_intent = TrackerMotionIntentStateMachine(
            self.config.tracker_motion_intent_config
        )

    @property
    def name(self) -> str:
        return f"pose_to_arm_{self.config.side}"

    def set_home_anchors(self, *, human_pose_desk: np.ndarray, robot_pose_desk: np.ndarray) -> None:
        """Bind raw desk-frame tracker Home to robot Home.

        ``human_pose_desk`` must not be pre-adjusted.  Tracker-local translation
        and rotation offsets are applied here exactly once, matching
        :meth:`retarget`.
        """

        self._home_human_pose_desk = self.prepare_input_pose_desk(human_pose_desk)
        self._home_robot_pose_desk = as_pose_matrix(robot_pose_desk)
        self._last_target_pose_desk = self._home_robot_pose_desk.copy()
        self._tracker_motion_intent.reset("home_anchors_set")

    def clear_home_anchors(self) -> None:
        self._home_human_pose_desk = None
        self._home_robot_pose_desk = None
        self._last_target_pose_desk = None
        self._tracker_motion_intent.reset("home_anchors_cleared")

    @property
    def last_target_pose_desk(self) -> np.ndarray | None:
        return None if self._last_target_pose_desk is None else self._last_target_pose_desk.copy()

    @property
    def home_human_pose_desk(self) -> np.ndarray | None:
        if self._home_human_pose_desk is None:
            return None
        return self._home_human_pose_desk.copy()

    @property
    def home_robot_pose_desk(self) -> np.ndarray | None:
        if self._home_robot_pose_desk is None:
            return None
        return self._home_robot_pose_desk.copy()

    def resolve_absolute_target_from_observation(
        self,
        observation: TeleopObservation,
        calibration: TeleopCalibration,
    ) -> dict[str, object] | None:
        channel = observation.get_channel(self.config.input_channel_name)
        if channel is None or not channel.valid:
            return None
        (
            pose_desk,
            pose_operator,
            input_pose_normalization,
        ) = self._resolve_pose_desk_and_operator(
            channel=channel,
            calibration=calibration,
        )
        target_pose_desk = self._apply_absolute_mapping(pose_desk)
        target_pose_desk = self._apply_orientation_policy(target_pose_desk)
        if self.config.workspace_clip:
            target_pose_desk = calibration.clip_pose_to_workspace(
                target_pose_desk,
                side=self.config.side,
            )
        return {
            "channel": channel,
            "clutch_pressed": False,
            "pose_operator": pose_operator,
            "target_pose_desk": target_pose_desk,
            "input_pose_normalization": input_pose_normalization,
            "tracker_role": self._resolve_tracker_role(channel),
            "source_tracker_role": self._resolve_source_tracker_role(channel),
        }

    def build_command_from_target_pose_desk(
        self,
        *,
        target_pose_desk: np.ndarray,
        observation: TeleopObservation,
        calibration: TeleopCalibration,
        pose_operator: np.ndarray | None = None,
        input_pose_normalization: dict[str, float | bool] | None = None,
    ) -> TeleopCommand:
        command = TeleopCommand(timestamp=observation.timestamp)
        channel = observation.get_channel(self.config.input_channel_name)
        if channel is None or not channel.valid:
            command.metadata["skipped"] = self.config.input_channel_name
            return command
        channel_metadata = dict(channel.metadata or {})
        fixed_side = str(
            channel_metadata.get("fix_arm_fixed_side", self.config.side) or ""
        ).strip().lower()
        fix_arm_authority_metadata = (
            {
                "fix_arm_frozen": True,
                "fix_arm_fixed_side": fixed_side,
            }
            if bool(channel_metadata.get("fix_arm_frozen", False))
            and fixed_side == self.config.side
            else {}
        )

        target_pose_tcp_desk = as_pose_matrix(target_pose_desk)
        self._last_target_pose_desk = target_pose_tcp_desk.copy()
        target_pose_base = calibration.desk_to_base(
            target_pose_tcp_desk,
            side=self.config.side,
        )
        tracker_motion_intent = self._resolve_tracker_motion_intent(
            channel=channel,
        )
        calibration_tracker_role = self._resolve_tracker_role(channel)
        source_tracker_role = self._resolve_source_tracker_role(channel)
        tracker_lineage_metadata = {
            "tracker_role": calibration_tracker_role,
            **(
                {"source_tracker_role": source_tracker_role}
                if source_tracker_role != calibration_tracker_role
                else {}
            ),
        }

        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="cartesian_pose",
                value=target_pose_base,
                frame_name=self.config.output_frame_name,
                metadata={
                    "arm_controlled_frame": TOOL_TCP_FRAME,
                    "arm_reference_frame": self.config.output_frame_name,
                    "arm_action_spec": ArmActionSpec.tool_tcp(
                        self.config.side,
                        reference_frame=self.config.output_frame_name,
                    ).to_metadata(),
                    "control_mode": self.config.control_mode,
                    "orientation_mode": self.config.orientation_mode,
                    "input_channel": self.config.input_channel_name,
                    **(
                        {}
                        if input_pose_normalization is None
                        else {
                            "input_pose_normalization": dict(
                                input_pose_normalization
                            )
                        }
                    ),
                    **tracker_lineage_metadata,
                    **fix_arm_authority_metadata,
                    "clutch_pressed": False,
                    "tracker_motion_intent": tracker_motion_intent,
                    "target_pose_operator": None
                    if pose_operator is None
                    else as_pose_matrix(pose_operator),
                    "target_pose_desk": target_pose_tcp_desk,
                    "target_pose_tcp_desk": target_pose_tcp_desk.copy(),
                    "target_pose_base": target_pose_base.copy(),
                    "target_pose_tcp_base": target_pose_base.copy(),
                    "target_pose_controlled_frame": TOOL_TCP_FRAME,
                    "target_pose_retarget_frame": TOOL_TCP_FRAME,
                    "target_pose_desk_retarget_frame": target_pose_tcp_desk.copy(),
                    "input_local_translation_offset_m": None
                    if self._input_local_translation_offset is None
                    else self._input_local_translation_offset.copy(),
                },
            )
        )
        return command

    def _resolve_tracker_motion_intent(
        self,
        *,
        channel,
    ) -> dict[str, object]:
        if not bool(self.config.tracker_motion_intent_enabled):
            return {
                "schema": "teleop_runtime.tracker_motion_intent/v1",
                "state": "DISABLED",
                "inactive": False,
                "hold_weight": 0.0,
                "transition_reason": "disabled",
            }
        metadata = dict(channel.metadata or {})
        motion_window = metadata.get("tracker_motion_window")
        return self._tracker_motion_intent.update(
            motion_window=(
                dict(motion_window) if isinstance(motion_window, dict) else None
            ),
            tracker_pose=as_pose_matrix(channel.value),
            clutch_pressed=False,
        )

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

        (
            pose_desk,
            pose_operator,
            input_pose_normalization,
        ) = self._resolve_pose_desk_and_operator(
            channel=channel,
            calibration=calibration,
        )

        target_pose_desk = self._apply_absolute_mapping(pose_desk)

        target_pose_desk = self._apply_orientation_policy(target_pose_desk)
        if self.config.workspace_clip:
            target_pose_desk = calibration.clip_pose_to_workspace(
                target_pose_desk,
                side=self.config.side,
            )
        return self.build_command_from_target_pose_desk(
            target_pose_desk=target_pose_desk,
            observation=observation,
            calibration=calibration,
            pose_operator=pose_operator,
            input_pose_normalization=input_pose_normalization,
        )

    def _resolve_tracker_role(self, channel) -> str | None:
        if self.config.tracker_role is not None:
            return self.config.tracker_role
        return self._resolve_source_tracker_role(channel)

    @staticmethod
    def _resolve_source_tracker_role(channel) -> str | None:
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
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
        pose, normalization = project_near_rigid_pose(
            channel.value,
            name=f"{self.config.input_channel_name} input pose",
        )
        frame_name = channel.frame_name or ""
        tracker_role = self._resolve_tracker_role(channel)

        if frame_name in {"vive_world", "world"}:
            pose_operator = calibration.world_to_operator(pose, tracker_role=tracker_role)
            pose_desk = calibration.operator_to_desk(pose_operator)
            return (
                self.prepare_input_pose_desk(pose_desk),
                self._apply_input_pose_adjustments(pose_operator),
                normalization,
            )
        if frame_name.endswith("_operator") or frame_name == "operator":
            pose_operator = pose
            pose_desk = calibration.operator_to_desk(pose_operator)
            return (
                self.prepare_input_pose_desk(pose_desk),
                self._apply_input_pose_adjustments(pose_operator),
                normalization,
            )
        if frame_name in {calibration.desk_frame.name, "desk"}:
            pose_desk = pose
            if calibration.T_O_D is None:
                raise RuntimeError("Cannot recover operator-frame pose without T_O_D calibration.")
            pose_operator = calibration.T_O_D @ pose_desk
            return (
                self.prepare_input_pose_desk(pose_desk),
                self._apply_input_pose_adjustments(pose_operator),
                normalization,
            )
        raise ValueError(
            f"{self.name} does not know how to map input frame '{frame_name or '<none>'}' to desk."
        )

    def prepare_input_pose_desk(self, pose_desk: np.ndarray) -> np.ndarray:
        return self._apply_input_pose_adjustments(pose_desk)

    def _apply_input_pose_adjustments(self, pose: np.ndarray) -> np.ndarray:
        corrected = self._apply_input_local_translation(pose)
        return self._apply_input_local_rotation(corrected)

    def _apply_input_local_translation(self, pose: np.ndarray) -> np.ndarray:
        pose_matrix = as_pose_matrix(pose)
        if self._input_local_translation_offset is None:
            return pose_matrix
        corrected = pose_matrix.copy()
        corrected[:3, 3] = (
            corrected[:3, 3] + corrected[:3, :3] @ self._input_local_translation_offset
        )
        return corrected

    def _apply_input_local_rotation(self, pose: np.ndarray) -> np.ndarray:
        if self._input_local_rotation_offset is None:
            return as_pose_matrix(pose)
        return apply_local_pose_rotation(pose, self._input_local_rotation_offset)

    def _apply_home_relative_translation(self, pose_desk: np.ndarray) -> np.ndarray:
        current_human_pose = as_pose_matrix(pose_desk)
        target = current_human_pose.copy()
        translation_scale = float(self.config.translation_scale)
        if self._home_human_pose_desk is None or self._home_robot_pose_desk is None:
            target[:3, 3] = current_human_pose[:3, 3] * translation_scale
            return target

        delta_translation = (
            current_human_pose[:3, 3] - self._home_human_pose_desk[:3, 3]
        ) * translation_scale
        target[:3, 3] = self._home_robot_pose_desk[:3, 3] + delta_translation
        return target

    def _apply_absolute_mapping(self, pose_desk: np.ndarray) -> np.ndarray:
        current_human_pose = as_pose_matrix(pose_desk)
        target = self._apply_home_relative_translation(current_human_pose)
        if self._home_human_pose_desk is None or self._home_robot_pose_desk is None:
            return target

        target[:3, :3] = (
            self._home_robot_pose_desk[:3, :3]
            @ self._home_human_pose_desk[:3, :3].T
            @ current_human_pose[:3, :3]
        )
        return target

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

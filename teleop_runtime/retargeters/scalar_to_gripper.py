from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robot_motion.action.gripper import (
    DEFAULT_GRIPPER_INPUT_MAX,
    DEFAULT_GRIPPER_INPUT_MIN,
    map_scalar_to_gripper_target,
)

from ..backends.grippers import GRIPPER_ACTION_MAX
from ..calibration import TeleopCalibration
from ..action import HandActionSpec
from ..interfaces import Retargeter
from ..registry import register_retargeter
from ..schemas import EffectorCommand, TeleopCommand, TeleopObservation


@dataclass(frozen=True)
class ScalarToGripperRetargeterConfig:
    side: str
    input_channel_name: str
    effector_name: str | None = None
    source_dim: int | None = 2
    output_dim: int = 0
    invert: bool = False
    alpha: float = 0.95
    input_min: float = DEFAULT_GRIPPER_INPUT_MIN
    input_max: float = DEFAULT_GRIPPER_INPUT_MAX
    output_min: float = 0.0
    output_max: float = GRIPPER_ACTION_MAX

    def __post_init__(self) -> None:
        if self.effector_name is None:
            object.__setattr__(self, "effector_name", f"{self.side}_hand")
        if self.source_dim is not None and not 0 <= int(self.source_dim) < 6:
            raise ValueError("source_dim must be in [0, 5], or None for a scalar input channel.")
        if int(self.output_dim) != 0:
            raise ValueError("DM-Tac gripper output_dim is fixed to hand slot 0.")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if not 0.0 <= float(self.input_min) < float(self.input_max) <= 1.0:
            raise ValueError("input_min/input_max must satisfy 0 <= input_min < input_max <= 1.")
        if float(self.output_min) < 0.0:
            raise ValueError("output_min must be non-negative.")
        if float(self.output_max) > GRIPPER_ACTION_MAX:
            raise ValueError(f"output_max must be <= {GRIPPER_ACTION_MAX}.")
        if float(self.output_min) > float(self.output_max):
            raise ValueError("output_min cannot exceed output_max.")


@register_retargeter("scalar_to_gripper")
class ScalarToGripperRetargeter(Retargeter):
    """Map a Q6 dimension or scalar source channel into the 1D DM-Tac gripper slot."""

    def __init__(self, config: ScalarToGripperRetargeterConfig):
        self.config = config
        self._filtered_value: float | None = None

    @property
    def name(self) -> str:
        return f"scalar_to_gripper_{self.config.side}"

    def retarget(
        self,
        observation: TeleopObservation,
        calibration: TeleopCalibration | None = None,
    ) -> TeleopCommand:
        del calibration
        command = TeleopCommand(timestamp=observation.timestamp)
        channel = observation.get_channel(self.config.input_channel_name)
        if channel is None or not channel.valid:
            command.metadata["skipped"] = self.config.input_channel_name
            return command

        source = np.asarray(channel.value, dtype=np.float32).reshape(-1)
        if self.config.source_dim is None:
            if source.shape != (1,):
                raise ValueError(
                    f"{self.name} expects one scalar input value, got {source.shape}."
                )
            source_value = float(np.clip(source[0], 0.0, 1.0))
        else:
            if source.shape != (6,):
                raise ValueError(
                    f"{self.name} expects q6 input with shape (6,), got {source.shape}."
                )
            source = np.clip(source, 0.0, 1.0)
            source_value = float(source[int(self.config.source_dim)])
        target_value, normalized_value = map_scalar_to_gripper_target(
            source_value,
            input_min=float(self.config.input_min),
            input_max=float(self.config.input_max),
            invert=bool(self.config.invert),
            output_min=float(self.config.output_min),
            output_max=float(self.config.output_max),
        )
        if self._filtered_value is None or self.config.alpha >= 1.0:
            self._filtered_value = target_value
        else:
            alpha = float(self.config.alpha)
            self._filtered_value = alpha * target_value + (1.0 - alpha) * self._filtered_value
        self._filtered_value = float(
            np.clip(
                self._filtered_value,
                float(self.config.output_min),
                float(self.config.output_max),
            )
        )

        command_value = np.zeros(6, dtype=np.float32)
        gripper_slot = int(self.config.output_dim)
        command_value[gripper_slot] = float(self._filtered_value)
        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="joint_vector",
                value=command_value,
                frame_name=None,
                metadata={
                    "input_channel": self.config.input_channel_name,
                    "source_dim": (
                        None if self.config.source_dim is None else int(self.config.source_dim)
                    ),
                    "output_dim": gripper_slot,
                    "gripper_slot": gripper_slot,
                    "gripper_action_dim": 1,
                    "invert": bool(self.config.invert),
                    "alpha": float(self.config.alpha),
                    "input_min": float(self.config.input_min),
                    "input_max": float(self.config.input_max),
                    "source_q6": source.copy(),  # Backward-compatible name for recorded Q6 metadata.
                    "source_value_vector": source.copy(),
                    "source_value": float(source_value),
                    "source_normalized_value": float(normalized_value),
                    "gripper_joint_target": float(self._filtered_value),
                    "hand_action_spec": HandActionSpec.dmtac_gripper(
                        self.config.side,
                        gripper_slot=gripper_slot,
                    ).to_metadata(),
                    "gripper_command_semantics": (
                        "Six-wide compatibility vector; only gripper_slot is a 1D gripper actuator."
                    ),
                },
            )
        )
        return command

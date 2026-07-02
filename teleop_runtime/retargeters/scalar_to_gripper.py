from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    source_dim: int = 2
    output_dim: int = 1
    invert: bool = False
    alpha: float = 1.0
    output_min: float = 0.0
    output_max: float = GRIPPER_ACTION_MAX

    def __post_init__(self) -> None:
        if self.effector_name is None:
            object.__setattr__(self, "effector_name", f"{self.side}_hand")
        if not 0 <= int(self.source_dim) < 6:
            raise ValueError("source_dim must be in [0, 5].")
        if not 0 <= int(self.output_dim) < 6:
            raise ValueError("output_dim must be in [0, 5].")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        if float(self.output_min) < 0.0:
            raise ValueError("output_min must be non-negative.")
        if float(self.output_max) > GRIPPER_ACTION_MAX:
            raise ValueError(f"output_max must be <= {GRIPPER_ACTION_MAX}.")
        if float(self.output_min) > float(self.output_max):
            raise ValueError("output_min cannot exceed output_max.")


@register_retargeter("scalar_to_gripper")
class ScalarToGripperRetargeter(Retargeter):
    """Map one q6 source dimension into a 6-wide gripper command slot."""

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
        if source.shape != (6,):
            raise ValueError(
                f"{self.name} expects q6 input with shape (6,), got {source.shape}."
            )
        source = np.clip(source, 0.0, 1.0)
        source_value = float(source[int(self.config.source_dim)])
        target_value = 1.0 - source_value if self.config.invert else source_value
        target_value = float(
            np.clip(
                target_value,
                float(self.config.output_min),
                float(self.config.output_max),
            )
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
        command_value[int(self.config.output_dim)] = float(self._filtered_value)
        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="joint_vector",
                value=command_value,
                frame_name=None,
                metadata={
                    "input_channel": self.config.input_channel_name,
                    "source_dim": int(self.config.source_dim),
                    "output_dim": int(self.config.output_dim),
                    "invert": bool(self.config.invert),
                    "alpha": float(self.config.alpha),
                    "source_q6": source.copy(),
                    "source_value": float(source_value),
                    "gripper_joint_target": float(self._filtered_value),
                    "hand_action_spec": HandActionSpec.dmtac_gripper(
                        self.config.side,
                        gripper_slot=int(self.config.output_dim),
                    ).to_metadata(),
                    "gripper_command_semantics": (
                        "Six-wide compatibility vector; only output_dim is a gripper actuator."
                    ),
                },
            )
        )
        return command

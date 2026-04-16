from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..calibration import TeleopCalibration
from ..interfaces import Retargeter
from ..registry import register_retargeter
from ..schemas import EffectorCommand, TeleopCommand, TeleopObservation


@dataclass
class JointsToDexHandRetargeterConfig:
    side: str
    input_channel_name: str
    effector_name: str | None = None
    joint_order: tuple[int, ...] = (5, 4, 3, 2, 0, 1)
    invert: bool = True
    gain: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    alpha: float = 1.0
    output_min: float = 0.0
    output_max: float = 1.0

    def __post_init__(self) -> None:
        if self.effector_name is None:
            self.effector_name = f"{self.side}_hand"
        if len(self.joint_order) != 6:
            raise ValueError("joint_order must contain exactly 6 indices for q6 input.")
        if len(self.gain) != 6 or len(self.bias) != 6:
            raise ValueError("gain and bias must both contain exactly 6 values.")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1].")


@register_retargeter("joints_to_dexhand")
class JointsToDexHandRetargeter(Retargeter):
    def __init__(self, config: JointsToDexHandRetargeterConfig):
        self.config = config
        self._joint_order = np.asarray(config.joint_order, dtype=np.int64)
        self._gain = np.asarray(config.gain, dtype=np.float32)
        self._bias = np.asarray(config.bias, dtype=np.float32)
        self._filtered_cmd: np.ndarray | None = None

    @property
    def name(self) -> str:
        return f"joints_to_dexhand_{self.config.side}"

    def retarget(
        self, observation: TeleopObservation, calibration: TeleopCalibration | None = None
    ) -> TeleopCommand:
        del calibration
        command = TeleopCommand(timestamp=observation.timestamp)
        channel = observation.get_channel(self.config.input_channel_name)
        if channel is None or not channel.valid:
            command.metadata["skipped"] = self.config.input_channel_name
            return command

        source = np.asarray(channel.value, dtype=np.float32)
        if source.shape != (6,):
            raise ValueError(
                f"{self.name} expects q6 input with shape (6,), got {source.shape}."
            )

        source = np.clip(source, 0.0, 1.0)
        source = source[self._joint_order]

        target = 1.0 - source if self.config.invert else source.copy()
        target = target * self._gain + self._bias
        target = np.clip(target, self.config.output_min, self.config.output_max)

        if self._filtered_cmd is None or self.config.alpha >= 1.0:
            self._filtered_cmd = target
        else:
            self._filtered_cmd = (
                self.config.alpha * target + (1.0 - self.config.alpha) * self._filtered_cmd
            )

        command.add_command(
            EffectorCommand(
                effector_name=str(self.config.effector_name),
                side=self.config.side,
                command_type="joint_vector",
                value=self._filtered_cmd.astype(np.float32),
                frame_name=None,
                metadata={
                    "input_channel": self.config.input_channel_name,
                    "joint_order": self._joint_order.copy(),
                    "invert": self.config.invert,
                    "gain": self._gain.copy(),
                    "bias": self._bias.copy(),
                    "alpha": self.config.alpha,
                    "source_q6": source.copy(),
                },
            )
        )
        return command

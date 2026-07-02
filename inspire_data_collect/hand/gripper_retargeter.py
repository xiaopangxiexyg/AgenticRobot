from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import EffectorCommand, ObservationChannel


GRIPPER_ACTION_MAX = 0.99


@dataclass(frozen=True)
class GripperRetargeterConfig:
    side: str
    input_channel_name: str
    effector_name: str
    source_dim: int = 1
    output_dim: int = 1
    invert: bool = True
    alpha: float = 1.0
    output_min: float = 0.0
    output_max: float = GRIPPER_ACTION_MAX

    def __post_init__(self) -> None:
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


class GripperRetargeter:
    """Map one exo-hand q6 dimension to a compatibility q6 gripper command.

    The collection writer stores hand actions in a fixed 6-wide slot. For the
    gripper mode only `output_dim` is meaningful; the remaining slots are held
    at zero for stable vector shape.
    """

    def __init__(self, config: GripperRetargeterConfig):
        self.config = config
        self._filtered_value: float | None = None

    def retarget(
        self,
        channels: dict[str, ObservationChannel],
        *,
        timestamp: float,
    ) -> EffectorCommand | None:
        channel = channels.get(self.config.input_channel_name)
        if channel is None or not channel.valid:
            return None
        source = np.asarray(channel.value, dtype=np.float32).reshape(-1)
        if source.shape != (6,):
            return None
        source = np.clip(source, 0.0, 1.0)
        source_value = float(source[int(self.config.source_dim)])
        target_value = 1.0 - source_value if self.config.invert else source_value
        target_value = float(
            np.clip(target_value, float(self.config.output_min), float(self.config.output_max))
        )
        if self._filtered_value is None or self.config.alpha >= 1.0:
            self._filtered_value = target_value
        else:
            alpha = float(self.config.alpha)
            self._filtered_value = alpha * target_value + (1.0 - alpha) * self._filtered_value
        self._filtered_value = float(
            np.clip(self._filtered_value, float(self.config.output_min), float(self.config.output_max))
        )

        command_value = np.zeros(6, dtype=np.float32)
        command_value[int(self.config.output_dim)] = float(self._filtered_value)
        return EffectorCommand(
            effector_name=self.config.effector_name,
            side=self.config.side,
            value=command_value,
            timestamp=float(timestamp),
            metadata={
                "input_channel": self.config.input_channel_name,
                "source_dim": int(self.config.source_dim),
                "output_dim": int(self.config.output_dim),
                "invert": bool(self.config.invert),
                "alpha": float(self.config.alpha),
                "source_q6": source.copy(),
                "source_value": float(source_value),
                "gripper_joint_target": float(self._filtered_value),
            },
        )

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import EffectorCommand, ObservationChannel
from .hand_mapping import DEFAULT_INSPIRE_HAND_JOINT_ORDER


@dataclass(frozen=True)
class HandRetargeterConfig:
    side: str
    input_channel_name: str
    effector_name: str
    joint_order: tuple[int, ...] = DEFAULT_INSPIRE_HAND_JOINT_ORDER
    invert: bool = True
    alpha: float = 1.0
    output_min: float = 0.0
    output_max: float = 1.0

    def __post_init__(self) -> None:
        if len(self.joint_order) != 6:
            raise ValueError("joint_order must contain exactly 6 indices.")
        if not 0.0 < float(self.alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1].")


class HandRetargeter:
    def __init__(self, config: HandRetargeterConfig):
        self.config = config
        self._joint_order = np.asarray(config.joint_order, dtype=np.int64)
        self._filtered_cmd: np.ndarray | None = None

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
        source = source[self._joint_order]
        target = 1.0 - source if self.config.invert else source.copy()
        target = np.clip(target, self.config.output_min, self.config.output_max)
        if self._filtered_cmd is None or self.config.alpha >= 1.0:
            self._filtered_cmd = target
        else:
            alpha = float(self.config.alpha)
            self._filtered_cmd = alpha * target + (1.0 - alpha) * self._filtered_cmd
        return EffectorCommand(
            effector_name=self.config.effector_name,
            side=self.config.side,
            value=self._filtered_cmd.astype(np.float32, copy=True),
            timestamp=float(timestamp),
            metadata={
                "input_channel": self.config.input_channel_name,
                "joint_order": self._joint_order.copy(),
                "invert": bool(self.config.invert),
                "alpha": float(self.config.alpha),
                "source_q6": source.copy(),
            },
        )

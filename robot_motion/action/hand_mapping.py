from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


SOURCE_HAND_Q6_ORDER = (
    "thumb_yaw",
    "thumb_flex",
    "index",
    "middle",
    "ring",
    "pinky",
)

# InspireTac q6 expects the four fingers first, then thumb yaw/flex.
DEFAULT_INSPIRE_HAND_JOINT_ORDER = (5, 4, 3, 2, 0, 1)

TARGET_INSPIRE_HAND_Q6_ORDER = tuple(
    SOURCE_HAND_Q6_ORDER[index] for index in DEFAULT_INSPIRE_HAND_JOINT_ORDER
)


@dataclass(frozen=True)
class HandQ6MappingResult:
    source_q6: np.ndarray
    target_q6: np.ndarray


def map_q6_to_dexhand_target(
    source_q6: Sequence[float] | np.ndarray,
    *,
    joint_order: Sequence[int] = DEFAULT_INSPIRE_HAND_JOINT_ORDER,
    invert: bool = True,
    gain: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    bias: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    output_min: float = 0.0,
    output_max: float = 1.0,
) -> HandQ6MappingResult:
    source = np.asarray(source_q6, dtype=np.float32)
    if source.shape != (6,):
        raise ValueError(f"source_q6 must have shape (6,), got {source.shape}.")

    order = np.asarray(joint_order, dtype=np.int64)
    if order.shape != (6,):
        raise ValueError("joint_order must contain exactly 6 indices for q6 input.")
    gain_array = np.asarray(gain, dtype=np.float32)
    bias_array = np.asarray(bias, dtype=np.float32)
    if gain_array.shape != (6,) or bias_array.shape != (6,):
        raise ValueError("gain and bias must both contain exactly 6 values.")

    reordered_source = np.clip(source, 0.0, 1.0)[order]
    target = 1.0 - reordered_source if invert else reordered_source.copy()
    target = target * gain_array + bias_array
    target = np.clip(target, float(output_min), float(output_max))
    return HandQ6MappingResult(
        source_q6=reordered_source.copy(),
        target_q6=target.astype(np.float32, copy=False),
    )


__all__ = [
    "DEFAULT_INSPIRE_HAND_JOINT_ORDER",
    "HandQ6MappingResult",
    "SOURCE_HAND_Q6_ORDER",
    "TARGET_INSPIRE_HAND_Q6_ORDER",
    "map_q6_to_dexhand_target",
]

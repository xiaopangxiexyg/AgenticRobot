from __future__ import annotations

import numpy as np


DEFAULT_GRIPPER_INPUT_MIN = 0.1
DEFAULT_GRIPPER_INPUT_MAX = 0.9
DEFAULT_GRIPPER_ACTION_MAX = 1.0


def map_scalar_to_gripper_target(
    source_value: float,
    *,
    input_min: float = DEFAULT_GRIPPER_INPUT_MIN,
    input_max: float = DEFAULT_GRIPPER_INPUT_MAX,
    invert: bool = False,
    output_min: float = 0.0,
    output_max: float = DEFAULT_GRIPPER_ACTION_MAX,
) -> tuple[float, float]:
    source = float(np.clip(float(source_value), 0.0, 1.0))
    lower = float(input_min)
    upper = float(input_max)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("input_min must be finite and smaller than input_max.")
    normalized = float(np.clip((source - lower) / (upper - lower), 0.0, 1.0))
    if invert:
        normalized = 1.0 - normalized
    target = float(output_min) + normalized * (float(output_max) - float(output_min))
    return float(target), float(normalized)


__all__ = [
    "DEFAULT_GRIPPER_ACTION_MAX",
    "DEFAULT_GRIPPER_INPUT_MAX",
    "DEFAULT_GRIPPER_INPUT_MIN",
    "map_scalar_to_gripper_target",
]

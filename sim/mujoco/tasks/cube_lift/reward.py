"""Pure Cube-Lift reward and termination predicates."""

from __future__ import annotations

import numpy as np

from .config import CubeLiftRewardConfig


def lift_progress(*, cube_z_m: float, resting_center_z_m: float, lift_delta_m: float) -> float:
    return float(np.clip((cube_z_m - resting_center_z_m) / lift_delta_m, 0.0, 1.0))


def cube_out_of_bounds(
    cube_position_m: np.ndarray,
    *,
    lower_m: tuple[float, float, float],
    upper_m: tuple[float, float, float],
) -> bool:
    position = np.asarray(cube_position_m, dtype=float).reshape(3)
    return bool(np.any(position < np.asarray(lower_m)) or np.any(position > np.asarray(upper_m)))


def reward_for_step(
    *,
    progress: float,
    action: np.ndarray,
    success: bool,
    config: CubeLiftRewardConfig,
) -> float:
    values = np.asarray(action, dtype=float).reshape(7)
    value = float(progress) - config.action_penalty * float(np.mean(np.square(values)))
    if success:
        value += config.success_bonus
    return float(value)


__all__ = ["cube_out_of_bounds", "lift_progress", "reward_for_step"]

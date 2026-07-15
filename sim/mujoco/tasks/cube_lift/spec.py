"""Task-local RL spaces; intentionally separate from real/deploy contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


CUBE_LIFT_ACTION_LAYOUT_VERSION = "right_qpos_delta6_right_gripper_delta1.v0"
CUBE_LIFT_OBSERVATION_LAYOUT_VERSION = (
    "right_qpos6_right_close1_cube_pose7_cube_velocity6_"
    "cube_to_grasp3_last_action7_progress1.v0"
)


def require_gymnasium():
    """Import Gymnasium lazily so config/physics utilities remain lightweight."""

    try:
        import gymnasium
        from gymnasium import spaces
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "CubeLiftEnv requires gymnasium. Install it in dexteleop310 with "
            "`python -m pip install gymnasium`."
        ) from exc
    return gymnasium, spaces


@dataclass(frozen=True)
class CubeLiftActionSpec:
    """Seven normalized deltas consumed by ``CubeLiftEnv`` at policy rate."""

    layout_version: str = CUBE_LIFT_ACTION_LAYOUT_VERSION
    dtype: np.dtype = np.dtype(np.float32)
    dim: int = 7

    def __post_init__(self) -> None:
        if self.layout_version != CUBE_LIFT_ACTION_LAYOUT_VERSION or self.dim != 7:
            raise ValueError("Unsupported CubeLift action layout.")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    def coerce(self, action: Any) -> np.ndarray:
        values = np.asarray(action, dtype=self.dtype).reshape(-1)
        if values.shape != (self.dim,) or not np.all(np.isfinite(values)):
            raise ValueError("CubeLift action must be a finite 7-D vector.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("CubeLift action values must lie in [-1, 1].")
        return values.copy()

    def gym_space(self):
        _gymnasium, spaces = require_gymnasium()
        return spaces.Box(low=-1.0, high=1.0, shape=(self.dim,), dtype=self.dtype)


@dataclass(frozen=True)
class CubeLiftObservationSpec:
    """Fixed 31-D privileged state layout for the first headless task."""

    layout_version: str = CUBE_LIFT_OBSERVATION_LAYOUT_VERSION
    dtype: np.dtype = np.dtype(np.float32)
    field_names: tuple[str, ...] = (
        "right_qpos_0",
        "right_qpos_1",
        "right_qpos_2",
        "right_qpos_3",
        "right_qpos_4",
        "right_qpos_5",
        "right_gripper_close",
        "cube_x_m",
        "cube_y_m",
        "cube_z_m",
        "cube_quat_w",
        "cube_quat_x",
        "cube_quat_y",
        "cube_quat_z",
        "cube_linear_velocity_x_mps",
        "cube_linear_velocity_y_mps",
        "cube_linear_velocity_z_mps",
        "cube_angular_velocity_x_radps",
        "cube_angular_velocity_y_radps",
        "cube_angular_velocity_z_radps",
        "cube_to_right_grasp_x_m",
        "cube_to_right_grasp_y_m",
        "cube_to_right_grasp_z_m",
        "last_action_0",
        "last_action_1",
        "last_action_2",
        "last_action_3",
        "last_action_4",
        "last_action_5",
        "last_action_6",
        "episode_progress",
    )

    def __post_init__(self) -> None:
        if self.layout_version != CUBE_LIFT_OBSERVATION_LAYOUT_VERSION:
            raise ValueError("Unsupported CubeLift observation layout.")
        if len(self.field_names) != 31:
            raise ValueError("CubeLift observation layout must remain 31-D.")
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def dim(self) -> int:
        return len(self.field_names)

    def coerce(self, observation: Any) -> np.ndarray:
        values = np.asarray(observation, dtype=self.dtype).reshape(-1)
        if values.shape != (self.dim,) or not np.all(np.isfinite(values)):
            raise RuntimeError("CubeLift observation became non-finite or changed layout.")
        return values.copy()

    def gym_space(self):
        _gymnasium, spaces = require_gymnasium()
        # Keep finite declared bounds so Gymnasium's API checker can catch
        # non-finite observations rather than accepting them through inf bounds.
        return spaces.Box(low=-1e6, high=1e6, shape=(self.dim,), dtype=self.dtype)


__all__ = [
    "CUBE_LIFT_ACTION_LAYOUT_VERSION",
    "CUBE_LIFT_OBSERVATION_LAYOUT_VERSION",
    "CubeLiftActionSpec",
    "CubeLiftObservationSpec",
    "require_gymnasium",
]

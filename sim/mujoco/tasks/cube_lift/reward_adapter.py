"""Named simulator-only reward state for the Cube-Lift task.

The temporary PPO reward path historically receives a flat 31-D vector.  The
layout remains accepted as a compatibility boundary, but all task reward code
can use this named projection instead of scattering fixed indices through the
generic training loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CubeLiftRewardState:
    """One batched transition view of the simulator-only reward state."""

    right_qpos: np.ndarray
    right_gripper_close: np.ndarray
    cube_qpos: np.ndarray
    cube_qvel: np.ndarray
    cube_to_grasp: np.ndarray
    last_action: np.ndarray
    episode_progress: np.ndarray

    @classmethod
    def from_flat(cls, values: Any) -> "CubeLiftRewardState":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != 31:
            raise ValueError(
                "CubeLift reward state compatibility input must have shape (N, 31)."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("CubeLift reward state must be finite.")
        return cls(
            right_qpos=np.ascontiguousarray(array[:, 0:6]),
            right_gripper_close=np.ascontiguousarray(array[:, 6]),
            cube_qpos=np.ascontiguousarray(array[:, 7:14]),
            cube_qvel=np.ascontiguousarray(array[:, 14:20]),
            cube_to_grasp=np.ascontiguousarray(array[:, 20:23]),
            last_action=np.ascontiguousarray(array[:, 23:30]),
            episode_progress=np.ascontiguousarray(array[:, 30]),
        )

    @property
    def batch_size(self) -> int:
        return int(self.right_qpos.shape[0])

    @property
    def cube_position(self) -> np.ndarray:
        return self.cube_qpos[:, :3]

    @property
    def cube_z(self) -> np.ndarray:
        return self.cube_qpos[:, 2]

    def validate_batch(self, expected_size: int) -> None:
        if self.batch_size != int(expected_size):
            raise ValueError(
                f"CubeLift reward state batch mismatch: expected {expected_size}, "
                f"got {self.batch_size}."
            )


__all__ = ["CubeLiftRewardState"]

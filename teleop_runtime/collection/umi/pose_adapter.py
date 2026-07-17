"""Explicit tracker/TCP/desk pose transforms for UMI collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.transforms import validate_pose_matrix


def as_rigid_transform(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    valid, reason = validate_pose_matrix(result, atol=1e-5)
    if not valid:
        raise ValueError(f"{name}: {reason}")
    return result.copy()


def transform_to_pose7(transform: object) -> np.ndarray:
    matrix = as_rigid_transform(transform, name="transform")
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    quaternion /= np.linalg.norm(quaternion)
    return np.concatenate([matrix[:3, 3], quaternion]).astype(np.float64)


@dataclass(frozen=True)
class UmiPoseAdapter:
    T_vive_world_to_desk: np.ndarray
    T_tcp_to_tracker_by_side: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "T_vive_world_to_desk",
            as_rigid_transform(self.T_vive_world_to_desk, name="T_vive_world_to_desk"),
        )
        transforms = {
            side: as_rigid_transform(value, name=f"T_tcp_to_tracker_by_side[{side}]")
            for side, value in dict(self.T_tcp_to_tracker_by_side).items()
        }
        if set(transforms) != {"left", "right"}:
            raise ValueError("T_tcp_to_tracker_by_side must define left and right")
        object.__setattr__(self, "T_tcp_to_tracker_by_side", transforms)

    def tcp_to_vive_world(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        tracker = as_rigid_transform(
            T_tracker_to_vive_world, name="T_tracker_to_vive_world"
        )
        return tracker @ self.T_tcp_to_tracker_by_side[str(side)]

    def tcp_to_desk(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        return self.T_vive_world_to_desk @ self.tcp_to_vive_world(
            side, T_tracker_to_vive_world
        )

    def raw_pose7(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        return transform_to_pose7(self.tcp_to_vive_world(side, T_tracker_to_vive_world))

    def desk_pose7(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        return transform_to_pose7(self.tcp_to_desk(side, T_tracker_to_vive_world))


__all__ = ["UmiPoseAdapter", "as_rigid_transform", "transform_to_pose7"]

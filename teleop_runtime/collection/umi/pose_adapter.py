"""Explicit tracker/TCP/desk pose transforms for UMI collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.transforms import project_near_rigid_pose, validate_pose_matrix
from .topology import SIDES


class UmiPoseAdapterPort(Protocol):
    """Map an external tracker pose into the frozen UMI raw/desk pose fields."""

    def normalize_tracker_pose(
        self, tracker_pose: Any
    ) -> tuple[np.ndarray, dict[str, float | bool]]: ...

    def raw_pose7(self, side: str, tracker_pose: Any) -> np.ndarray: ...

    def desk_pose7(self, side: str, tracker_pose: Any) -> np.ndarray: ...


MAX_VIVE_ROTATION_ORTHOGONALITY_ERROR = 1e-2
MAX_VIVE_ROTATION_DETERMINANT_ERROR = 1e-2
MAX_VIVE_ROTATION_PROJECTION_FROBENIUS = 1e-2


def as_rigid_transform(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    valid, reason = validate_pose_matrix(result, atol=1e-5)
    if not valid:
        raise ValueError(f"{name}: {reason}")
    # A matrix can pass the bounded validation above while still carrying enough
    # floating-point drift to fail again after several rigid-transform products.
    # Project only already-valid rotations so malformed calibration is not hidden.
    u, _singular_values, vt = np.linalg.svd(result[:3, :3])
    projected = u @ vt
    if float(np.linalg.det(projected)) <= 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    normalized = result.copy()
    normalized[:3, :3] = projected
    normalized[3] = np.array([0.0, 0.0, 0.0, 1.0])
    return normalized


def normalize_vive_tracker_transform(
    value: object,
    *,
    name: str = "T_tracker_to_vive_world",
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Project small OpenVR float drift onto SO(3), rejecting malformed poses."""

    return project_near_rigid_pose(
        value,
        name=name,
        max_orthogonality_error=MAX_VIVE_ROTATION_ORTHOGONALITY_ERROR,
        max_determinant_error=MAX_VIVE_ROTATION_DETERMINANT_ERROR,
        max_projection_frobenius=MAX_VIVE_ROTATION_PROJECTION_FROBENIUS,
    )


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
        if set(transforms) != set(SIDES):
            raise ValueError("T_tcp_to_tracker_by_side must define left and right")
        object.__setattr__(self, "T_tcp_to_tracker_by_side", transforms)

    def tcp_to_vive_world(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        tracker, _diagnostics = normalize_vive_tracker_transform(
            T_tracker_to_vive_world,
            name="T_tracker_to_vive_world",
        )
        return tracker @ self.T_tcp_to_tracker_by_side[str(side)]

    def normalize_tracker_pose(
        self, T_tracker_to_vive_world: object
    ) -> tuple[np.ndarray, dict[str, float | bool]]:
        return normalize_vive_tracker_transform(T_tracker_to_vive_world)

    def tcp_to_desk(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        return self.T_vive_world_to_desk @ self.tcp_to_vive_world(
            side, T_tracker_to_vive_world
        )

    def raw_pose7(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        return transform_to_pose7(self.tcp_to_vive_world(side, T_tracker_to_vive_world))

    def desk_pose7(self, side: str, T_tracker_to_vive_world: object) -> np.ndarray:
        return transform_to_pose7(self.tcp_to_desk(side, T_tracker_to_vive_world))


__all__ = [
    "UmiPoseAdapter",
    "UmiPoseAdapterPort",
    "as_rigid_transform",
    "normalize_vive_tracker_transform",
    "transform_to_pose7",
]

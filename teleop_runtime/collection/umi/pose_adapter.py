"""Explicit tracker/TCP/desk pose transforms for UMI collection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from robot_motion.transforms import validate_pose_matrix


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

    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4):
        raise ValueError(f"{name} must be a 4x4 matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: Pose contains non-finite values.")
    if not np.allclose(result[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        raise ValueError(f"{name}: Pose last row is not [0, 0, 0, 1].")

    raw_rotation = result[:3, :3]
    orthogonality_error = float(
        np.max(np.abs(raw_rotation.T @ raw_rotation - np.eye(3)))
    )
    determinant = float(np.linalg.det(raw_rotation))
    determinant_error = abs(determinant - 1.0)
    if determinant <= 0.0:
        raise ValueError(
            f"{name}: Rotation determinant must be positive, got {determinant:.6g}."
        )
    try:
        u, _singular_values, vt = np.linalg.svd(raw_rotation)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{name}: Rotation SVD failed: {exc}") from exc
    projected = u @ vt
    if float(np.linalg.det(projected)) <= 0.0:
        u[:, -1] *= -1.0
        projected = u @ vt
    projection_frobenius = float(np.linalg.norm(raw_rotation - projected, ord="fro"))
    if (
        orthogonality_error > MAX_VIVE_ROTATION_ORTHOGONALITY_ERROR
        or determinant_error > MAX_VIVE_ROTATION_DETERMINANT_ERROR
        or projection_frobenius > MAX_VIVE_ROTATION_PROJECTION_FROBENIUS
    ):
        raise ValueError(
            f"{name}: Rotation is too far from SO(3) "
            f"(orth_err={orthogonality_error:.6g}, det={determinant:.6g}, "
            f"projection_fro={projection_frobenius:.6g})."
        )
    normalized = result.copy()
    normalized[:3, :3] = projected
    valid, reason = validate_pose_matrix(normalized, atol=1e-8)
    if not valid:
        raise ValueError(f"{name}: {reason}")
    return normalized, {
        "rotation_projected": bool(projection_frobenius > 1e-12),
        "rotation_orthogonality_error": orthogonality_error,
        "rotation_determinant": determinant,
        "rotation_projection_frobenius": projection_frobenius,
    }


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
    "as_rigid_transform",
    "normalize_vive_tracker_transform",
    "transform_to_pose7",
]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..transforms import as_pose_matrix, identity_pose


def _normalize(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-9:
        raise ValueError("Cannot normalize a near-zero vector.")
    return array / norm


def pose_from_rotation_translation(rotation: Any, translation: Any) -> np.ndarray:
    pose = identity_pose()
    pose[:3, :3] = np.asarray(rotation, dtype=float).reshape(3, 3)
    pose[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return pose


def euler_xyz_deg_to_matrix(euler_deg: Any) -> np.ndarray:
    roll_deg, pitch_deg, yaw_deg = np.asarray(euler_deg, dtype=float).reshape(3)
    roll = np.deg2rad(roll_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw = np.deg2rad(yaw_deg)

    cx, sx = float(np.cos(roll)), float(np.sin(roll))
    cy, sy = float(np.cos(pitch)), float(np.sin(pitch))
    cz, sz = float(np.cos(yaw)), float(np.sin(yaw))

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cx, -sx],
            [0.0, sx, cx],
        ],
        dtype=float,
    )
    ry = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=float,
    )
    rz = np.array(
        [
            [cz, -sz, 0.0],
            [sz, cz, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rz @ ry @ rx


def average_pose(poses: list[np.ndarray]) -> np.ndarray:
    if not poses:
        raise ValueError("Cannot average an empty pose list.")
    pose_stack = [as_pose_matrix(pose) for pose in poses]
    mean_pose = identity_pose()
    mean_pose[:3, 3] = np.mean([pose[:3, 3] for pose in pose_stack], axis=0)
    rot_sum = np.sum([pose[:3, :3] for pose in pose_stack], axis=0)
    u, _, vt = np.linalg.svd(rot_sum)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    mean_pose[:3, :3] = rotation
    return mean_pose


@dataclass(frozen=True)
class AxisFitQuality:
    sample_count: int
    extent_m: float
    variance_ratio: float
    start_end_distance_m: float


@dataclass(frozen=True)
class SignedAxisFit:
    axis_world: np.ndarray
    mean_world: np.ndarray
    quality: AxisFitQuality

    def to_metadata(self) -> dict[str, Any]:
        return {
            "axis_world": self.axis_world.tolist(),
            "mean_world": self.mean_world.tolist(),
            "quality": {
                "sample_count": self.quality.sample_count,
                "extent_m": self.quality.extent_m,
                "variance_ratio": self.quality.variance_ratio,
                "start_end_distance_m": self.quality.start_end_distance_m,
            },
        }


@dataclass(frozen=True)
class VirtualDeskCalibrationResult:
    T_W_V: np.ndarray
    x_fit: SignedAxisFit
    y_fit: SignedAxisFit
    z_fit: SignedAxisFit
    origin_world: np.ndarray
    x_axis_world: np.ndarray
    y_axis_world: np.ndarray
    z_axis_world: np.ndarray
    xy_non_orthogonality_deg: float
    xz_non_orthogonality_deg: float
    yz_non_orthogonality_deg: float

    def to_metadata(self) -> dict[str, Any]:
        return {
            "T_W_V": self.T_W_V.tolist(),
            "origin_world": self.origin_world.tolist(),
            "x_axis_world": self.x_axis_world.tolist(),
            "y_axis_world": self.y_axis_world.tolist(),
            "z_axis_world": self.z_axis_world.tolist(),
            "xy_non_orthogonality_deg": self.xy_non_orthogonality_deg,
            "xz_non_orthogonality_deg": self.xz_non_orthogonality_deg,
            "yz_non_orthogonality_deg": self.yz_non_orthogonality_deg,
            "x_fit": self.x_fit.to_metadata(),
            "y_fit": self.y_fit.to_metadata(),
            "z_fit": self.z_fit.to_metadata(),
        }


def fit_signed_axis_from_poses(poses: list[np.ndarray]) -> SignedAxisFit:
    if len(poses) < 8:
        raise ValueError("At least 8 pose samples are required to fit a sweep axis.")
    pose_stack = [as_pose_matrix(pose) for pose in poses]
    positions = np.asarray([pose[:3, 3] for pose in pose_stack], dtype=float)
    centered = positions - np.mean(positions, axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    axis = _normalize(vh[0])

    edge_count = max(1, min(len(positions) // 5, 10))
    start_mean = np.mean(positions[:edge_count], axis=0)
    end_mean = np.mean(positions[-edge_count:], axis=0)
    displacement = end_mean - start_mean
    if float(np.dot(displacement, axis)) < 0.0:
        axis = -axis

    squared = singular_values * singular_values
    variance_ratio = float(squared[0] / max(np.sum(squared), 1e-12))
    projections = centered @ axis
    extent_m = float(np.max(projections) - np.min(projections))
    start_end_distance_m = float(np.linalg.norm(displacement))

    return SignedAxisFit(
        axis_world=axis,
        mean_world=np.mean(positions, axis=0),
        quality=AxisFitQuality(
            sample_count=len(poses),
            extent_m=extent_m,
            variance_ratio=variance_ratio,
            start_end_distance_m=start_end_distance_m,
        ),
    )


def fit_virtual_desk_frame(
    *,
    x_sweep_poses: list[np.ndarray],
    y_sweep_poses: list[np.ndarray],
    z_sweep_poses: list[np.ndarray],
    neutral_poses: list[np.ndarray],
) -> VirtualDeskCalibrationResult:
    x_fit = fit_signed_axis_from_poses(x_sweep_poses)
    y_fit = fit_signed_axis_from_poses(y_sweep_poses)
    z_fit = fit_signed_axis_from_poses(z_sweep_poses)

    x_axis = _normalize(x_fit.axis_world)
    y_raw = _normalize(y_fit.axis_world)
    y_axis = y_raw - float(np.dot(y_raw, x_axis)) * x_axis
    y_axis = _normalize(y_axis)
    z_axis = _normalize(np.cross(x_axis, y_axis))

    z_hint = _normalize(z_fit.axis_world)
    if float(np.dot(z_axis, z_hint)) < 0.0:
        y_axis = -y_axis
        z_axis = -z_axis

    origin_world = np.mean(
        np.asarray([as_pose_matrix(pose)[:3, 3] for pose in neutral_poses], dtype=float),
        axis=0,
    )
    T_W_V = pose_from_rotation_translation(
        np.column_stack([x_axis, y_axis, z_axis]),
        origin_world,
    )

    xy_non_orthogonality_deg = abs(90.0 - np.rad2deg(np.arccos(np.clip(np.dot(x_axis, y_raw), -1.0, 1.0))))
    xz_non_orthogonality_deg = abs(90.0 - np.rad2deg(np.arccos(np.clip(np.dot(x_axis, z_hint), -1.0, 1.0))))
    yz_non_orthogonality_deg = abs(90.0 - np.rad2deg(np.arccos(np.clip(np.dot(y_axis, z_axis), -1.0, 1.0))))

    return VirtualDeskCalibrationResult(
        T_W_V=T_W_V,
        x_fit=x_fit,
        y_fit=y_fit,
        z_fit=z_fit,
        origin_world=origin_world,
        x_axis_world=x_axis,
        y_axis_world=y_axis,
        z_axis_world=z_axis,
        xy_non_orthogonality_deg=float(xy_non_orthogonality_deg),
        xz_non_orthogonality_deg=float(xz_non_orthogonality_deg),
        yz_non_orthogonality_deg=float(yz_non_orthogonality_deg),
    )


def tracker_offset_from_neutral_pose(
    tracker_pose_virtual: np.ndarray,
    *,
    hand_rotation_virtual: np.ndarray | None = None,
    translation_tracker_to_hand: Any = (0.0, 0.0, 0.0),
) -> np.ndarray:
    T_V_T = as_pose_matrix(tracker_pose_virtual)
    desired_rotation = np.eye(3, dtype=float)
    if hand_rotation_virtual is not None:
        desired_rotation = np.asarray(hand_rotation_virtual, dtype=float).reshape(3, 3)
    rotation_tracker_to_hand = T_V_T[:3, :3].T @ desired_rotation
    return pose_from_rotation_translation(
        rotation_tracker_to_hand,
        np.asarray(translation_tracker_to_hand, dtype=float).reshape(3),
    )


def compose_tracker_offset(base_offset: np.ndarray, tune_euler_deg: Any) -> np.ndarray:
    tuned = as_pose_matrix(base_offset)
    delta_rotation = euler_xyz_deg_to_matrix(tune_euler_deg)
    tuned[:3, :3] = tuned[:3, :3] @ delta_rotation
    return tuned

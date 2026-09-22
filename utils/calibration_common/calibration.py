"""Shared data and stateless helpers for camera/robot calibration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Sequence

import numpy as np

from .capture import now_utc
from .transforms import as_transform, transform_distance, transform_to_list


@dataclass
class CalibrationSample:
    """One synchronized robot/end-effector and target/camera observation."""

    T_gripper_to_base: np.ndarray
    T_target_to_camera: np.ndarray
    robot_pose_xyz_rpy: list[float] | None = None
    robot_joint_deg: list[float] | None = None
    target_reprojection_rmse_px: float | None = None
    target_corner_count: int | None = None
    timestamp_utc: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_gripper_to_base = as_transform(
            self.T_gripper_to_base,
            name="T_gripper_to_base",
        )
        self.T_target_to_camera = as_transform(
            self.T_target_to_camera,
            name="T_target_to_camera",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_gripper_to_base": transform_to_list(self.T_gripper_to_base),
            "T_target_to_camera": transform_to_list(self.T_target_to_camera),
            "robot_pose_xyz_rpy": list_or_none(self.robot_pose_xyz_rpy),
            "robot_joint_deg": list_or_none(self.robot_joint_deg),
            "target_reprojection_rmse_px": float_or_none(
                self.target_reprojection_rmse_px
            ),
            "target_corner_count": (
                int(self.target_corner_count) if self.target_corner_count is not None else None
            ),
            "timestamp_utc": self.timestamp_utc,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibrationSample":
        return cls(
            T_gripper_to_base=np.asarray(payload["T_gripper_to_base"], dtype=np.float64),
            T_target_to_camera=np.asarray(payload["T_target_to_camera"], dtype=np.float64),
            robot_pose_xyz_rpy=list_or_none(payload.get("robot_pose_xyz_rpy")),
            robot_joint_deg=list_or_none(payload.get("robot_joint_deg")),
            target_reprojection_rmse_px=float_or_none(
                payload.get("target_reprojection_rmse_px")
            ),
            target_corner_count=payload.get("target_corner_count"),
            timestamp_utc=payload.get("timestamp_utc"),
            metadata=dict(payload.get("metadata") or {}),
        )


def pose_span(transforms: Sequence[object]) -> tuple[float, float]:
    matrices = [as_transform(value) for value in transforms]
    if len(matrices) < 2:
        return 0.0, 0.0
    translation_span = 0.0
    rotation_span = 0.0
    for index, first in enumerate(matrices):
        for second in matrices[index + 1 :]:
            translation, rotation = transform_distance(first, second)
            translation_span = max(translation_span, translation)
            rotation_span = max(rotation_span, rotation)
    return float(translation_span), float(rotation_span)


def rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array * array)))


def max_or_zero(values: Sequence[float]) -> float:
    return float(max(values)) if values else 0.0


def timestamp_for_filename() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_or_none(values: Any) -> list[float] | None:
    if values is None:
        return None
    return [float(value) for value in list(values)]


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


__all__ = [
    "CalibrationSample",
    "float_or_none",
    "jsonable",
    "list_or_none",
    "max_or_zero",
    "now_utc",
    "pose_span",
    "rms",
    "timestamp_for_filename",
]

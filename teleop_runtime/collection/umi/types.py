"""Immutable samples at the UMI collection/writer boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from collection_runtime.source_cut import SnapshotRejection

from ..operator_labels import OperatorLabelSnapshot
from .topology import SIDES


@dataclass(frozen=True)
class UmiPoseMeasurement:
    side: str
    raw_tcp_pose7: np.ndarray
    desk_tcp_pose7: np.ndarray
    source_timestamp: float
    available_timestamp: float
    source_clock: str
    available_clock: str
    sequence: int
    tracker_name: str
    tracker_role: str
    age_sec: float

    def __post_init__(self) -> None:
        for name in ("raw_tcp_pose7", "desk_tcp_pose7"):
            value = np.asarray(getattr(self, name), dtype=np.float64).reshape(-1)
            if value.shape != (7,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite pose7")
            if not np.isclose(np.linalg.norm(value[3:]), 1.0, atol=1e-5):
                raise ValueError(f"{name} quaternion must have unit norm")
            object.__setattr__(self, name, value.copy())
        if self.side not in set(SIDES):
            raise ValueError("side must be left or right")
        if int(self.sequence) <= 0:
            raise ValueError("Vive sequence must be positive")


@dataclass
class UmiSample:
    episode_index: int
    frame_index: int
    timestamp: float
    poses: Mapping[str, UmiPoseMeasurement]
    gripper_commands: Mapping[str, float]
    hand_states: Mapping[str, Mapping[str, Any]]
    operator_labels: OperatorLabelSnapshot
    arm_action_poses: Mapping[str, UmiPoseMeasurement] | None = None
    arm_action_source_frame_index: int | None = None
    arm_action_source_timestamp: float | None = None
    camera_frames: Mapping[str, Any] = field(default_factory=dict)
    channels: Mapping[str, Any] = field(default_factory=dict)
    faults: tuple[Any, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if set(self.poses) != set(SIDES):
            raise ValueError("UMI samples require left and right poses")
        if set(self.gripper_commands) != set(SIDES):
            raise ValueError("UMI samples require left and right gripper commands")
        if self.arm_action_poses is None and (
            self.arm_action_source_frame_index is not None
            or self.arm_action_source_timestamp is not None
        ):
            raise ValueError("UMI arm action alignment fields must be provided together")
        if self.arm_action_poses is not None:
            if set(self.arm_action_poses) != set(SIDES):
                raise ValueError("UMI arm action poses require left and right poses")
            if self.arm_action_source_frame_index is None:
                raise ValueError("UMI aligned arm actions require a source frame index")
            if int(self.arm_action_source_frame_index) != int(self.frame_index) + 1:
                raise ValueError(
                    "UMI arm action source must be the next accepted observation frame"
                )
            if self.arm_action_source_timestamp is None or not np.isfinite(
                float(self.arm_action_source_timestamp)
            ):
                raise ValueError("UMI aligned arm actions require a finite source timestamp")
            if float(self.arm_action_source_timestamp) <= float(self.timestamp):
                raise ValueError("UMI arm action source timestamp must follow the row timestamp")
        for side, value in self.gripper_commands.items():
            scalar = float(value)
            if not np.isfinite(scalar) or not 0.0 <= scalar <= 1.0:
                raise ValueError(f"{side} gripper command must be in [0, 1]")

__all__ = ["SnapshotRejection", "UmiPoseMeasurement", "UmiSample"]

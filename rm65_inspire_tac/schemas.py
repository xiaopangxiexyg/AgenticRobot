from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _as_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=float)


@dataclass
class HandMotionInput:
    wrist_pose: np.ndarray | None = None
    hand_keypoints: np.ndarray | None = None
    finger_matrices: np.ndarray | None = None
    wrist_in_operator_frame: bool = True
    hand_keypoints_in_operator_frame: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.wrist_pose = _as_array(self.wrist_pose)
        self.hand_keypoints = _as_array(self.hand_keypoints)
        self.finger_matrices = _as_array(self.finger_matrices)


@dataclass
class TeleopFrame:
    left: HandMotionInput | None = None
    right: HandMotionInput | None = None
    timestamp: Any = None
    raw_data: Any = None

    def active_sides(self) -> tuple[str, ...]:
        active = []
        if self.left is not None:
            active.append("left")
        if self.right is not None:
            active.append("right")
        return tuple(active)


@dataclass
class HandRetargetingResult:
    side: str
    wrist_pose_robot: np.ndarray
    retargeted_hand_qpos: np.ndarray
    hardware_hand_qpos: np.ndarray
    hand_keypoints: np.ndarray | None = None
    wrist_pose_operator: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BimanualRetargetingResult:
    left: HandRetargetingResult | None = None
    right: HandRetargetingResult | None = None
    timestamp: Any = None
    raw_frame: Any = None


@dataclass
class ArmExecutionResult:
    side: str
    arm_qpos: np.ndarray
    wrist_pose_robot: np.ndarray
    retargeted_hand_qpos: np.ndarray
    hardware_hand_qpos: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BimanualExecutionResult:
    left: ArmExecutionResult | None = None
    right: ArmExecutionResult | None = None
    rendered_rgb: np.ndarray | None = None
    timestamp: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

"""Archived legacy delta-clutch controller.

Kept for existing teleop_runtime delta_clutch compatibility. This module is
intentionally not migrated to robot_motion; new shared sim/real motion code
should use the absolute tool-TCP action contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .transforms import as_pose_matrix


@dataclass
class ClutchAnchor:
    human_pose_reference: np.ndarray
    robot_pose_target: np.ndarray
    timestamp: float


@dataclass
class ClutchState:
    anchors: dict[str, ClutchAnchor] = field(default_factory=dict)
    pressed: dict[str, bool] = field(default_factory=dict)

    def is_pressed(self, side: str) -> bool:
        return bool(self.pressed.get(side, False))

    def clear(self, side: str | None = None) -> None:
        if side is None:
            self.anchors.clear()
            self.pressed.clear()
            return
        self.anchors.pop(side, None)
        self.pressed.pop(side, None)


class DeltaClutchController:
    def __init__(self, state: ClutchState | None = None):
        self.state = state if state is not None else ClutchState()

    def reanchor(
        self,
        side: str,
        *,
        human_pose_reference: np.ndarray,
        robot_pose_target: np.ndarray,
        timestamp: float,
        clutch_pressed: bool | None = None,
    ) -> None:
        self.state.anchors[side] = ClutchAnchor(
            human_pose_reference=as_pose_matrix(human_pose_reference),
            robot_pose_target=as_pose_matrix(robot_pose_target),
            timestamp=float(timestamp),
        )
        if clutch_pressed is not None:
            self.state.pressed[side] = bool(clutch_pressed)

    def update(
        self,
        side: str,
        clutch_pressed: bool,
        human_pose_reference: np.ndarray,
        robot_pose_target: np.ndarray,
        timestamp: float,
        scale: float = 1.0,
    ) -> np.ndarray:
        human_pose_reference = as_pose_matrix(human_pose_reference)
        robot_pose_target = as_pose_matrix(robot_pose_target)
        was_pressed = self.state.is_pressed(side)

        if clutch_pressed and not was_pressed:
            self.state.anchors[side] = ClutchAnchor(
                human_pose_reference=human_pose_reference.copy(),
                robot_pose_target=robot_pose_target.copy(),
                timestamp=timestamp,
            )

        self.state.pressed[side] = clutch_pressed

        if not clutch_pressed:
            return robot_pose_target.copy()

        if side not in self.state.anchors:
            self.state.anchors[side] = ClutchAnchor(
                human_pose_reference=human_pose_reference.copy(),
                robot_pose_target=robot_pose_target.copy(),
                timestamp=timestamp,
            )

        anchor = self.state.anchors[side]
        delta_translation = (
            human_pose_reference[:3, 3] - anchor.human_pose_reference[:3, 3]
        ) * float(scale)

        target_pose = anchor.robot_pose_target.copy()
        target_pose[:3, 3] = anchor.robot_pose_target[:3, 3] + delta_translation
        target_pose[:3, :3] = human_pose_reference[:3, :3]
        return target_pose

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ik import IKResult, PoseTarget, RM65CanonicalIK
from .scene import MujocoScene


class QposController:
    def __init__(self, scene: MujocoScene):
        self.scene = scene

    def apply(self, qpos_by_side: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        self.scene.set_qpos_by_side(qpos_by_side)
        return self.scene.arm_qpos_by_side()


class GripperController:
    def __init__(self, scene: MujocoScene):
        self.scene = scene

    def apply(self, close_by_name: Mapping[str, float] | float) -> dict[str, float]:
        if isinstance(close_by_name, (float, int)):
            self.scene.set_gripper_close(None, float(close_by_name))
        else:
            for name, close in close_by_name.items():
                self.scene.set_gripper_close(str(name), float(close))
        return self.scene.gripper_close_by_name()


class PoseIKController:
    """Apply one group-level canonical RM65 pose-planning result atomically."""

    def __init__(self, scene: MujocoScene, ik_solver: RM65CanonicalIK):
        self.scene = scene
        self.ik_solver = ik_solver

    def apply(self, pose_by_side: Mapping[str, np.ndarray]) -> IKResult:
        targets = {
            side: PoseTarget(side=side, pose=np.asarray(pose, dtype=float).reshape(4, 4))
            for side, pose in pose_by_side.items()
        }
        result = self.ik_solver.solve(targets)
        if result.success:
            self.scene.set_qpos_by_side(result.qpos_by_side)
        return result

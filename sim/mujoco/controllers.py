from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .ik import IKResult, PoseTarget, RM65CanonicalIK
from .scene import MujocoScene


class QposController:
    def __init__(self, scene: MujocoScene, *, collision_mode: str = "safe"):
        if collision_mode not in {"safe", "audit", "raw"}:
            raise ValueError("collision_mode must be 'safe', 'audit', or 'raw'.")
        self.scene = scene
        self.collision_mode = collision_mode
        self.last_safety_result = None

    def apply(self, qpos_by_side: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
        if self.collision_mode == "safe":
            self.last_safety_result = self.scene.apply_safe_kinematic_target(qpos_by_side)
        else:
            if self.collision_mode == "audit" and self.scene.collision_safety_available:
                start = self.scene.kinematic_controlled_qpos()
                requested = self.scene.kinematic_target_vector(start, qpos_by_side)
                projector = self.scene.collision_projector()
                assert projector is not None
                self.last_safety_result = projector.check_qpos(self.scene.data, requested)
            self.scene.set_qpos_by_side(qpos_by_side)
        return self.scene.arm_qpos_by_side()


class GripperController:
    def __init__(self, scene: MujocoScene, *, collision_mode: str = "safe"):
        if collision_mode not in {"safe", "audit", "raw"}:
            raise ValueError("collision_mode must be 'safe', 'audit', or 'raw'.")
        self.scene = scene
        self.collision_mode = collision_mode
        self.last_safety_result = None

    def apply(self, close_by_name: Mapping[str, float] | float) -> dict[str, float]:
        if isinstance(close_by_name, (float, int)):
            name = self.scene._resolve_gripper_name(None)
            targets = {name: float(close_by_name)}
        else:
            targets = {str(name): float(close) for name, close in close_by_name.items()}
        if self.collision_mode == "safe":
            self.last_safety_result = self.scene.apply_safe_kinematic_target(
                {}, gripper_close_by_name=targets
            )
        else:
            if self.collision_mode == "audit" and self.scene.collision_safety_available:
                start = self.scene.kinematic_controlled_qpos()
                requested = self.scene.kinematic_target_vector(
                    start, {}, gripper_close_by_name=targets
                )
                projector = self.scene.collision_projector()
                assert projector is not None
                self.last_safety_result = projector.check_qpos(self.scene.data, requested)
            for name, close in targets.items():
                self.scene.set_gripper_close(name, close)
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
            if not hasattr(self.scene, "apply_safe_kinematic_target"):
                self.scene.set_qpos_by_side(result.qpos_by_side)
                return result
            safety = self.scene.apply_safe_kinematic_target(result.qpos_by_side)
            if safety is not None and safety.modified:
                result = IKResult(
                    **{
                        **result.__dict__,
                        "success": False,
                        "hold": True,
                        "status": "mujoco_collision_projection_required",
                        "message": str(safety.to_dict()),
                    }
                )
        return result

"""Teleoperation adapter for the shared RM65 dual-arm target solver.

The collision-aware target-pose to qref algorithm lives in ``robot_motion``.
This module intentionally keeps the historical import path and adds only the
operator observation adapters and registry integration required by
``teleop_runtime``.
"""

from __future__ import annotations

from typing import Any

from robot_motion.rm65 import dual_arm_target_solver as _core
from robot_motion.rm65.dual_arm_target_solver import (
    BimanualIKResult,
    BimanualIKSideResult,
    BimanualPoseTarget,
    BimanualPoseTargetSource,
    DualArmCollisionAwareRetargeter as _SharedDualArmCollisionAwareRetargeter,
    DualArmCollisionAwareRetargeterConfig,
    DualArmQposCollisionCheck,
    DualArmTargetContext,
    _DualArmCandidate,
    _DualArmEvaluatedCandidate,
)
from ..registry import register_retargeter
from .pose_to_arm import PoseToArmRetargeter, PoseToArmRetargeterConfig


build_rm65_dual_arm_collision_ik_solver = _core.build_rm65_dual_arm_collision_ik_solver
build_rm65_dual_arm_scene_profile = _core.build_rm65_dual_arm_scene_profile


@register_retargeter("dual_arm_collision_aware")
class DualArmCollisionAwareRetargeter(_SharedDualArmCollisionAwareRetargeter):
    """Backward-compatible operator adapter around the shared motion core."""

    def __init__(
        self,
        config: DualArmCollisionAwareRetargeterConfig,
        *,
        left_pose_retargeter: PoseToArmRetargeter | None = None,
        right_pose_retargeter: PoseToArmRetargeter | None = None,
        ik_solver: Any | None = None,
    ) -> None:
        left_adapter = left_pose_retargeter or PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="left",
                input_channel_name=config.left_input_channel_name,
                effector_name=config.left_effector_name,
                control_mode=config.control_mode,
                orientation_mode=config.orientation_mode,
                translation_scale=config.translation_scale,
                workspace_clip=config.workspace_clip,
            )
        )
        right_adapter = right_pose_retargeter or PoseToArmRetargeter(
            PoseToArmRetargeterConfig(
                side="right",
                input_channel_name=config.right_input_channel_name,
                effector_name=config.right_effector_name,
                control_mode=config.control_mode,
                orientation_mode=config.orientation_mode,
                translation_scale=config.translation_scale,
                workspace_clip=config.workspace_clip,
            )
        )
        # Inject the historical module monkeypatch seams without mutating the
        # shared module's globals. This remains safe if real and simulation
        # composition roots happen to construct solvers concurrently.
        super().__init__(
            config,
            left_pose_retargeter=left_adapter,
            right_pose_retargeter=right_adapter,
            ik_solver=ik_solver,
            ik_solver_builder=build_rm65_dual_arm_collision_ik_solver,
            scene_profile_builder=build_rm65_dual_arm_scene_profile,
        )

# Tests and legacy diagnostics monkeypatch the module-level clock and construct
# candidate values directly. Keep those aliases pointed at the shared module
# so the compatibility surface remains behaviorally identical.
time = _core.time


__all__ = [
    "BimanualIKResult",
    "BimanualIKSideResult",
    "BimanualPoseTarget",
    "BimanualPoseTargetSource",
    "DualArmCollisionAwareRetargeter",
    "DualArmCollisionAwareRetargeterConfig",
    "DualArmQposCollisionCheck",
    "DualArmTargetContext",
]

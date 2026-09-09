"""URDF-only planning context for a registered real RM65 grasp trajectory."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Any

import numpy as np

from robot_motion.rm65 import DEFAULT_GRASP_PATH_COLLISION_SCENE
from robot_motion.rm65.obstacle_collision import (
    RM65_D405_ATTACHED_COLLISION_PROXIES,
    RM65ObstacleCollisionOracle,
    build_rm65_profile_obstacle_collision_oracle,
)
from robot_motion.scene_obstacles import ObstacleSceneSnapshot
from robot_motion.shadow_execution import AuthoritativeRobotStateSnapshot

from .rm65_profile import ResolvedExclusiveRM65SkillDeploymentProfile

_SIDES = ("left", "right")


class _MeasuredArmScene:
    def __init__(self, qpos_by_side: Mapping[str, np.ndarray]) -> None:
        self._qpos_by_side = {
            side: np.asarray(qpos_by_side[side], dtype=float).reshape(6).copy()
            for side in _SIDES
        }

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self._qpos_by_side[str(side)].copy()


class RM65UrdfGraspPlanningContext:
    """Duck-typed planning inputs backed only by the selected RM65 URDF."""

    def __init__(
        self,
        *,
        profile: ResolvedExclusiveRM65SkillDeploymentProfile,
        snapshot: AuthoritativeRobotStateSnapshot,
        obstacle_scene: ObstacleSceneSnapshot,
        target_builder: Any,
    ) -> None:
        if not isinstance(profile, ResolvedExclusiveRM65SkillDeploymentProfile):
            raise TypeError("profile must be resolved and typed.")
        if not isinstance(snapshot, AuthoritativeRobotStateSnapshot):
            raise TypeError("snapshot must be AuthoritativeRobotStateSnapshot.")
        if not isinstance(obstacle_scene, ObstacleSceneSnapshot):
            raise TypeError("obstacle_scene must be ObstacleSceneSnapshot.")
        if not callable(getattr(target_builder, "set_current_qpos_pair", None)):
            raise TypeError("target_builder must accept measured dual-arm qpos.")
        target_builder.set_current_qpos_pair(
            left_qpos=snapshot.qpos_by_side["left"],
            right_qpos=snapshot.qpos_by_side["right"],
            reset_tracking=True,
        )
        world_to_base = {
            side: np.linalg.inv(profile.scene_world_from_arm_base_by_side[side])
            for side in _SIDES
        }
        self.scene = _MeasuredArmScene(snapshot.qpos_by_side)
        self.retargeter = target_builder
        self.world_to_base_by_side = MappingProxyType(world_to_base)
        self.world_from_desk = np.asarray(
            profile.scene_world_from_canonical_station, dtype=float
        ).reshape(4, 4).copy()
        self.contract = SimpleNamespace(
            timing=SimpleNamespace(ik_hz=profile.position_profile.rates.ik_hz)
        )
        self.config = SimpleNamespace(
            joint_velocity_limit_profile=(
                profile.position_profile.ik_execution.joint_velocity_limit_profile
            )
        )
        self.obstacle_oracle = RM65ObstacleCollisionOracle.from_solver(
            target_builder.ik_solver,
            scene=obstacle_scene,
            solution_layout=target_builder.profile.solution_layout,
            expected_reference_frame_id=obstacle_scene.reference_frame_id,
            expected_reference_frame_spec_hash=(
                obstacle_scene.reference_frame_spec_hash
            ),
            attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
        )
        self.path_collision_profile_name = DEFAULT_GRASP_PATH_COLLISION_SCENE
        self.path_obstacle_oracle = build_rm65_profile_obstacle_collision_oracle(
            scene_profile=self.path_collision_profile_name,
            scene=obstacle_scene,
            expected_reference_frame_id=obstacle_scene.reference_frame_id,
            expected_reference_frame_spec_hash=(
                obstacle_scene.reference_frame_spec_hash
            ),
            attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
            use_profile_proxy_geometry=True,
        )
        self.path_revalidation_obstacle_oracle = (
            build_rm65_profile_obstacle_collision_oracle(
                scene_profile=self.path_collision_profile_name,
                scene=obstacle_scene,
                expected_reference_frame_id=obstacle_scene.reference_frame_id,
                expected_reference_frame_spec_hash=(
                    obstacle_scene.reference_frame_spec_hash
                ),
                attached_box_proxies=RM65_D405_ATTACHED_COLLISION_PROXIES,
                stop_at_first_clearance_violation=True,
            )
        )

    def close(self) -> None:
        """The context owns no hardware or simulation resource."""


__all__ = ["RM65UrdfGraspPlanningContext"]

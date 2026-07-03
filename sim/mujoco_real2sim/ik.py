from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .config import IKConfig
from .scene import MujocoScene
from .transforms import rotation_error_world


@dataclass(frozen=True)
class PoseTarget:
    side: str
    pose: np.ndarray
    body_name: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose", np.asarray(self.pose, dtype=float).reshape(4, 4))


@dataclass(frozen=True)
class IKResult:
    success: bool
    iterations: int
    qpos_by_side: dict[str, list[float]]
    position_error_m: dict[str, float] = field(default_factory=dict)
    orientation_error_rad: dict[str, float] = field(default_factory=dict)
    message: str = ""


class DampedLeastSquaresIK:
    """Small MuJoCo Jacobian IK solver for position-controlled real2sim checks."""

    def __init__(self, scene: MujocoScene, config: IKConfig | None = None):
        self.scene = scene
        self.config = config if config is not None else IKConfig()

    def solve(self, targets: Mapping[str, PoseTarget]) -> IKResult:
        if not targets:
            return IKResult(
                success=True,
                iterations=0,
                qpos_by_side=self.scene.arm_qpos_by_side(),
                message="no targets",
            )
        sides = tuple(targets)
        active_joint_names: list[str] = []
        for side in sides:
            if side not in self.scene.arm_specs:
                raise KeyError(f"Scene has no arm spec for side '{side}'.")
            active_joint_names.extend(self.scene.arm_specs[side].joint_names)
        active_dofs = np.asarray(
            [self.scene.joint_dof_addr(name) for name in active_joint_names],
            dtype=int,
        )
        active_qpos = np.asarray(
            [self.scene.joint_qpos_addr(name) for name in active_joint_names],
            dtype=int,
        )

        final_position_errors: dict[str, float] = {}
        final_orientation_errors: dict[str, float] = {}
        success = False
        message = "max_iterations"
        iterations = 0
        for iteration in range(1, int(self.config.max_iterations) + 1):
            self.scene.forward()
            error_blocks: list[np.ndarray] = []
            jac_blocks: list[np.ndarray] = []
            final_position_errors.clear()
            final_orientation_errors.clear()
            for side, target in targets.items():
                body_name = target.body_name or self.scene.arm_specs[side].controlled_body
                body_id = self.scene.body_id(body_name)
                current_pose = self.scene.body_pose(body_name)
                pos_error = target.pose[:3, 3] - current_pose[:3, 3]
                rot_error = rotation_error_world(current_pose[:3, :3], target.pose[:3, :3])
                final_position_errors[side] = float(np.linalg.norm(pos_error))
                final_orientation_errors[side] = float(np.linalg.norm(rot_error))

                jacp = np.zeros((3, self.scene.model.nv), dtype=float)
                jacr = np.zeros((3, self.scene.model.nv), dtype=float)
                self.scene.mujoco.mj_jacBody(self.scene.model, self.scene.data, jacp, jacr, body_id)
                error_blocks.append(
                    np.concatenate(
                        [
                            pos_error,
                            float(self.config.orientation_weight) * rot_error,
                        ]
                    )
                )
                jac_blocks.append(
                    np.vstack(
                        [
                            jacp[:, active_dofs],
                            float(self.config.orientation_weight) * jacr[:, active_dofs],
                        ]
                    )
                )
            if (
                all(value <= float(self.config.position_tolerance_m) for value in final_position_errors.values())
                and all(
                    value <= float(self.config.orientation_tolerance_rad)
                    for value in final_orientation_errors.values()
                )
            ):
                success = True
                message = "converged"
                iterations = iteration - 1
                break

            error = np.concatenate(error_blocks)
            jacobian = np.vstack(jac_blocks)
            damping = float(self.config.damping)
            lhs = jacobian @ jacobian.T + (damping * damping) * np.eye(jacobian.shape[0])
            dq = jacobian.T @ np.linalg.solve(lhs, error)
            max_step = float(self.config.max_joint_step_rad)
            dq = np.clip(dq, -max_step, max_step)
            for qpos_addr, delta in zip(active_qpos, dq):
                self.scene.data.qpos[qpos_addr] += float(delta)
            self._clip_joint_limits(active_joint_names)
            iterations = iteration

        self.scene.forward()
        return IKResult(
            success=success,
            iterations=iterations,
            qpos_by_side=self.scene.arm_qpos_by_side(),
            position_error_m=dict(final_position_errors),
            orientation_error_rad=dict(final_orientation_errors),
            message=message,
        )

    def _clip_joint_limits(self, joint_names: list[str]) -> None:
        model = self.scene.model
        mujoco = self.scene.mujoco
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            if not bool(model.jnt_limited[joint_id]):
                continue
            qpos_addr = int(model.jnt_qposadr[joint_id])
            lower, upper = model.jnt_range[joint_id]
            self.scene.data.qpos[qpos_addr] = float(
                np.clip(self.scene.data.qpos[qpos_addr], lower, upper)
            )


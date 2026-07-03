from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from teleop_runtime.interfaces import EffectorBackend
from teleop_runtime.schemas import EffectorCommand

from .ik import DampedLeastSquaresIK, PoseTarget
from .scene import MujocoScene


@dataclass
class MuJoCoArmBackendConfig:
    side: str
    effector_name: str | None = None
    controlled_body: str | None = None
    initial_qpos: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.side not in {"left", "right"}:
            raise ValueError("MuJoCoArmBackendConfig.side must be 'left' or 'right'.")
        if self.effector_name is None:
            self.effector_name = f"{self.side}_arm"
        if self.initial_qpos is not None and len(self.initial_qpos) != 6:
            raise ValueError("initial_qpos must contain 6 joint values.")


class MuJoCoArmBackend(EffectorBackend):
    """Position-controlled arm backend compatible with TeleopSession.

    This backend intentionally does not model force sensors, impedance control,
    gravity compensation, or hardware streaming. It applies qpos directly or
    resolves cartesian poses through MuJoCo Jacobian IK, then forwards the scene.
    """

    def __init__(
        self,
        scene: MujocoScene,
        config: MuJoCoArmBackendConfig,
        *,
        ik_solver: DampedLeastSquaresIK | None = None,
    ):
        self.scene = scene
        self.config = config
        self.ik_solver = ik_solver if ik_solver is not None else DampedLeastSquaresIK(scene)
        self._last_command: EffectorCommand | None = None
        self._last_target_qpos: np.ndarray | None = None
        self._last_command_timestamp: float | None = None
        if config.side not in self.scene.arm_specs:
            raise KeyError(f"Scene has no arm spec for side '{config.side}'.")
        if config.controlled_body is not None:
            spec = self.scene.arm_specs[config.side]
            self.scene.arm_specs[config.side] = type(spec)(
                side=spec.side,
                joint_names=spec.joint_names,
                controlled_body=str(config.controlled_body),
                effector_name=spec.effector_name,
            )

    @property
    def name(self) -> str:
        assert self.config.effector_name is not None
        return str(self.config.effector_name)

    def start(self) -> None:
        if self.config.initial_qpos is not None:
            self.scene.set_arm_qpos(self.config.side, self.config.initial_qpos)

    def send_command(self, command: EffectorCommand) -> None:
        if command.effector_name != self.name:
            raise ValueError(f"{self.name} received command for {command.effector_name!r}.")
        if command.command_type == "joint_vector":
            target = np.asarray(command.value, dtype=float).reshape(6)
            self.scene.set_arm_qpos(self.config.side, target)
        elif command.command_type == "cartesian_pose":
            target_pose = np.asarray(command.value, dtype=float).reshape(4, 4)
            result = self.ik_solver.solve(
                {
                    self.config.side: PoseTarget(
                        side=self.config.side,
                        pose=target_pose,
                        body_name=self.scene.arm_specs[self.config.side].controlled_body,
                    )
                }
            )
            target = np.asarray(result.qpos_by_side[self.config.side], dtype=float)
            if not result.success:
                command.metadata.setdefault("mujoco_ik_warning", result.message)
                command.metadata.setdefault(
                    "mujoco_ik_position_error_m",
                    result.position_error_m.get(self.config.side),
                )
                command.metadata.setdefault(
                    "mujoco_ik_orientation_error_rad",
                    result.orientation_error_rad.get(self.config.side),
                )
        else:
            raise ValueError(f"{self.name} does not support command_type={command.command_type!r}.")

        self._last_command = command.copy()
        self._last_target_qpos = target.copy()
        self._last_command_timestamp = time.time()

    def hold_motion(self, reason: str = "manual hold") -> None:
        del reason
        self._last_target_qpos = None

    def reset_motion_state(self, qpos: Sequence[float] | np.ndarray | None = None) -> None:
        if qpos is not None:
            self.scene.set_arm_qpos(self.config.side, np.asarray(qpos, dtype=float).reshape(6))
        self._last_command = None
        self._last_target_qpos = None
        self._last_command_timestamp = None

    def compute_tcp_pose(self, qpos: Sequence[float] | np.ndarray | None = None) -> np.ndarray:
        if qpos is None:
            return self.scene.body_pose(self.scene.arm_specs[self.config.side].controlled_body)
        current = self.scene.get_arm_qpos(self.config.side)
        self.scene.set_arm_qpos(self.config.side, np.asarray(qpos, dtype=float).reshape(6))
        pose = self.scene.body_pose(self.scene.arm_specs[self.config.side].controlled_body)
        self.scene.set_arm_qpos(self.config.side, current)
        return pose

    def get_state(self) -> dict[str, object]:
        qpos = self.scene.get_arm_qpos(self.config.side)
        return {
            "current_qpos": qpos.copy(),
            "current_tcp_pose": self.compute_tcp_pose(),
            "current_endpoint_pose": self.compute_tcp_pose(),
            "last_target_qpos": None if self._last_target_qpos is None else self._last_target_qpos.copy(),
            "last_command": None if self._last_command is None else self._last_command.copy(),
            "last_command_timestamp": self._last_command_timestamp,
            "backend": "mujoco_position",
            "controlled_body": self.scene.arm_specs[self.config.side].controlled_body,
        }


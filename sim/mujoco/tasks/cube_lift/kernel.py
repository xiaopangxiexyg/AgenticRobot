"""Low-allocation scalar kernel for the temporary Cube-Lift task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ...scene import MujocoScene
from .config import CubeLiftTaskConfig
from .env import DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH


InfoMode = Literal["compact", "terminal"]


@dataclass(frozen=True)
class CubeLiftKernelStep:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class CubeLiftKernel:
    """Task-specialized fast path with the reference environment's semantics.

    Names and rich objects are resolved once during construction.  The physics
    loop only writes indexed arrays, advances MuJoCo, and performs integer-ID
    contact checks.  This remains the explicitly non-identified direct-qpos
    debug execution mode.
    """

    action_dim = 7
    observation_dim = 31

    def __init__(
        self,
        config: CubeLiftTaskConfig | str | Path = DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH,
        *,
        scene: MujocoScene | None = None,
        info_mode: InfoMode = "terminal",
        use_settled_snapshot: bool = True,
        use_model_cache: bool = True,
        model_cache_dir: str | Path | None = None,
    ) -> None:
        if info_mode not in {"compact", "terminal"}:
            raise ValueError("info_mode must be 'compact' or 'terminal'.")
        self.config = (
            config if isinstance(config, CubeLiftTaskConfig) else CubeLiftTaskConfig.load(config)
        )
        contract = self.config.base_contract()
        self.model_artifact = None
        if scene is not None:
            self.scene = scene
        elif use_model_cache:
            from ...artifacts import ModelArtifactCache

            self.scene, self.model_artifact = ModelArtifactCache(model_cache_dir).load_scene(contract)
        else:
            self.scene = MujocoScene.from_contract(contract)
        self.model = self.scene.model
        self.data = self.scene.data
        self.mujoco = self.scene.mujoco
        self.info_mode = info_mode
        self.use_settled_snapshot = bool(use_settled_snapshot)

        self._left_qpos_idx, self._left_dof_idx = self._arm_indices(self.config.fixed_side)
        self._right_qpos_idx, self._right_dof_idx = self._arm_indices(
            self.config.controlled_side
        )
        self._controlled_dof_idx = np.concatenate((self._left_dof_idx, self._right_dof_idx))
        self._gripper_qpos_idx: list[np.ndarray] = []
        self._gripper_dof_idx: list[np.ndarray] = []
        self._gripper_open: list[np.ndarray] = []
        self._gripper_span: list[np.ndarray] = []
        self._gripper_names = (self.config.left_gripper_name, self.config.right_gripper_name)
        for name in self._gripper_names:
            spec = self.scene.gripper_specs[name]
            self._gripper_qpos_idx.append(
                np.asarray([self.scene.joint_qpos_addr(joint) for joint in spec.joint_names])
            )
            self._gripper_dof_idx.append(
                np.asarray([self.scene.joint_dof_addr(joint) for joint in spec.joint_names])
            )
            opened = np.asarray(spec.open_qpos, dtype=np.float64)
            self._gripper_open.append(opened)
            self._gripper_span.append(np.asarray(spec.closed_qpos, dtype=np.float64) - opened)
        self._controlled_dof_idx = np.concatenate(
            (self._controlled_dof_idx, *self._gripper_dof_idx)
        )

        free_joint_id = self.scene.joint_id(self.config.object_freejoint_name)
        self._cube_qpos_addr = int(self.model.jnt_qposadr[free_joint_id])
        self._cube_dof_addr = int(self.model.jnt_dofadr[free_joint_id])
        self._cube_body_id = self.scene.body_id(self.config.object_body_name)
        self._table_body_id = self.scene.body_id(self.config.table_body_name)
        self._grasp_body_id = self.scene.body_id(self.config.right_grasp_body_name)

        self._left_home = np.asarray(self.config.left_home_qpos, dtype=np.float64)
        self._right_home = np.asarray(self.config.right_home_qpos, dtype=np.float64)
        self._right_lower = np.asarray(self.config.right_qpos_lower, dtype=np.float64)
        self._right_upper = np.asarray(self.config.right_qpos_upper, dtype=np.float64)
        self._right_delta = np.asarray(self.config.right_joint_delta_rad, dtype=np.float64)
        self._bounds_lower = np.asarray(self.config.out_of_bounds_xyz_lower_m, dtype=np.float64)
        self._bounds_upper = np.asarray(self.config.out_of_bounds_xyz_upper_m, dtype=np.float64)
        self._spawn_qpos = np.asarray(self.config.cube_spawn_qpos, dtype=np.float64)
        self._zero_cube_qvel = np.zeros(6, dtype=np.float64)

        self._right_target = self._right_home.copy()
        self._gripper_close = np.asarray(
            [self.config.initial_gripper_close_by_name[name] for name in self._gripper_names],
            dtype=np.float64,
        )
        self._last_action = np.zeros(7, dtype=np.float32)
        self._observation_buffer = np.empty(31, dtype=np.float32)
        self._physics_tick = 0
        self._next_command_time = 0.0
        self._next_policy_time: float | None = None
        self._episode_steps = 0
        self._success_hold_ticks = 0
        self._termination_reason: str | None = None
        self._resting_center_z = float("nan")
        self.last_physics_ticks = 0
        self.last_command_ticks = 0
        self._settled_snapshot: dict[str, np.ndarray | float] | None = None

    def _arm_indices(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        joints = self.scene.arm_specs[side].joint_names
        return (
            np.asarray([self.scene.joint_qpos_addr(name) for name in joints]),
            np.asarray([self.scene.joint_dof_addr(name) for name in joints]),
        )

    @property
    def physics_time(self) -> float:
        return self._physics_tick / self.config.physics_hz

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        # RNG is reserved now so adding randomization does not change the API.
        self._rng = np.random.default_rng(seed)
        self._right_target[:] = self._right_home
        self._gripper_close[:] = (
            self.config.initial_gripper_close_by_name[self._gripper_names[0]],
            self.config.initial_gripper_close_by_name[self._gripper_names[1]],
        )
        self._last_action.fill(0.0)
        self._episode_steps = 0
        self._success_hold_ticks = 0
        self._termination_reason = None

        if self.use_settled_snapshot and self._settled_snapshot is not None:
            self._restore_settled_snapshot()
            self.last_physics_ticks = 0
            self.last_command_ticks = 0
        else:
            self.scene.reset_data()
            self.data.qpos[self._cube_qpos_addr : self._cube_qpos_addr + 7] = self._spawn_qpos
            self.data.qvel[self._cube_dof_addr : self._cube_dof_addr + 6] = self._zero_cube_qvel
            self._physics_tick = 0
            self._next_command_time = 1.0 / self.config.command_hz
            self.last_physics_ticks, self.last_command_ticks = self._advance_ticks(
                self.config.settle_physics_ticks,
                check_success=False,
            )
            if self._termination_reason is not None:
                raise RuntimeError(
                    "CubeLift fast reset did not produce a valid settled scene: "
                    f"{self._termination_reason}."
                )
            if self.use_settled_snapshot:
                self._capture_settled_snapshot()

        self._resting_center_z = float(self.data.qpos[self._cube_qpos_addr + 2])
        self._next_policy_time = self.physics_time + 1.0 / self.config.policy_hz
        return self._observation(), self._info(reset=True, terminal=True)

    def step(self, action: Any) -> CubeLiftKernelStep:
        if self._next_policy_time is None:
            raise RuntimeError("CubeLiftKernel.reset() must be called before step().")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("CubeLift action must be a finite 7-D vector.")
        if np.any(values < -1.0) or np.any(values > 1.0):
            raise ValueError("CubeLift action values must lie in [-1, 1].")
        self._last_action[:] = values
        np.clip(
            self._right_target + values[:6] * self._right_delta,
            self._right_lower,
            self._right_upper,
            out=self._right_target,
        )
        self._gripper_close[1] = np.clip(
            self._gripper_close[1] + float(values[6]) * self.config.right_gripper_delta,
            0.0,
            1.0,
        )
        self._episode_steps += 1
        self._termination_reason = None
        physics_ticks = 0
        command_ticks = 0
        while self.physics_time + 1e-12 < self._next_policy_time:
            due = self.physics_time + 1e-12 >= self._next_command_time
            if due:
                command_ticks += 1
                period = 1.0 / self.config.command_hz
                while self._next_command_time <= self.physics_time + 1e-12:
                    self._next_command_time += period
            self._step_physics()
            physics_ticks += 1
            if self._check_termination(check_success=True):
                break
        self._next_policy_time += 1.0 / self.config.policy_hz
        self.last_physics_ticks = physics_ticks
        self.last_command_ticks = command_ticks

        terminated = self._termination_reason is not None
        truncated = not terminated and self._episode_steps >= self.config.max_episode_steps
        success = self._termination_reason == "height_only_lift_success"
        cube_z = float(self.data.qpos[self._cube_qpos_addr + 2])
        progress = float(
            np.clip((cube_z - self._resting_center_z) / self.config.lift_delta_m, 0.0, 1.0)
        )
        reward = progress - self.config.reward.action_penalty * float(
            np.mean(np.square(values, dtype=np.float32))
        )
        if success:
            reward += self.config.reward.success_bonus
        if truncated:
            self._termination_reason = "time_limit"

        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            if not terminated:
                self._termination_reason = "non_finite_simulation_state"
                terminated = True
                truncated = False
        terminal = terminated or truncated
        return CubeLiftKernelStep(
            observation=self._observation(),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=self._info(reset=False, terminal=terminal),
        )

    def _write_direct_state(self) -> None:
        self.data.qpos[self._left_qpos_idx] = self._left_home
        self.data.qpos[self._right_qpos_idx] = self._right_target
        for index, opened, span, close in zip(
            self._gripper_qpos_idx,
            self._gripper_open,
            self._gripper_span,
            self._gripper_close,
        ):
            self.data.qpos[index] = opened + close * span
        self.data.qvel[self._controlled_dof_idx] = 0.0

    def _step_physics(self) -> None:
        self._write_direct_state()
        # mj_step performs the pre-integration forward pipeline.  One post-step
        # forward keeps body poses and contacts current for this task's checks.
        self.mujoco.mj_step(self.model, self.data)
        self.mujoco.mj_forward(self.model, self.data)
        self._physics_tick += 1

    def _advance_ticks(self, ticks: int, *, check_success: bool) -> tuple[int, int]:
        command_ticks = 0
        advanced = 0
        for _ in range(ticks):
            if self.physics_time + 1e-12 >= self._next_command_time:
                command_ticks += 1
                period = 1.0 / self.config.command_hz
                while self._next_command_time <= self.physics_time + 1e-12:
                    self._next_command_time += period
            self._step_physics()
            advanced += 1
            if self._check_termination(check_success=check_success):
                break
        return advanced, command_ticks

    def _critical_state_is_finite(self) -> bool:
        return bool(
            np.all(np.isfinite(self.data.qpos[self._right_qpos_idx]))
            and np.all(
                np.isfinite(
                    self.data.qpos[self._cube_qpos_addr : self._cube_qpos_addr + 7]
                )
            )
            and np.all(
                np.isfinite(
                    self.data.qvel[self._cube_dof_addr : self._cube_dof_addr + 6]
                )
            )
        )

    def _cube_touches_table(self) -> bool:
        geom_body_id = self.model.geom_bodyid
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            body1 = int(geom_body_id[int(contact.geom1)])
            body2 = int(geom_body_id[int(contact.geom2)])
            if (body1 == self._cube_body_id and body2 == self._table_body_id) or (
                body2 == self._cube_body_id and body1 == self._table_body_id
            ):
                return True
        return False

    def _check_termination(self, *, check_success: bool) -> bool:
        if not self._critical_state_is_finite():
            self._termination_reason = "non_finite_simulation_state"
            return True
        cube_position = self.data.qpos[self._cube_qpos_addr : self._cube_qpos_addr + 3]
        if np.any(cube_position < self._bounds_lower) or np.any(cube_position > self._bounds_upper):
            self._termination_reason = "cube_out_of_bounds"
            return True
        if not check_success:
            return False
        above_target = float(cube_position[2]) >= (
            self._resting_center_z + self.config.lift_delta_m
        )
        if above_target and not self._cube_touches_table():
            self._success_hold_ticks += 1
        else:
            self._success_hold_ticks = 0
        if self._success_hold_ticks >= self.config.success_hold_physics_ticks:
            self._termination_reason = "height_only_lift_success"
            return True
        return False

    def _observation(self) -> np.ndarray:
        out = self._observation_buffer
        cube_qpos = self.data.qpos[self._cube_qpos_addr : self._cube_qpos_addr + 7]
        cube_qvel = self.data.qvel[self._cube_dof_addr : self._cube_dof_addr + 6]
        out[0:6] = self.data.qpos[self._right_qpos_idx]
        span = self._gripper_span[1]
        active = np.abs(span) > 1e-12
        closes = (
            self.data.qpos[self._gripper_qpos_idx[1]][active] - self._gripper_open[1][active]
        ) / span[active]
        out[6] = np.clip(np.mean(closes), 0.0, 1.0)
        out[7:14] = cube_qpos
        out[14:20] = cube_qvel
        out[20:23] = cube_qpos[:3] - self.data.xpos[self._grasp_body_id]
        out[23:30] = self._last_action
        out[30] = self._episode_steps / self.config.max_episode_steps
        if not np.all(np.isfinite(out)):
            raise RuntimeError("CubeLift fast observation became non-finite.")
        return out.copy()

    def _info(self, *, reset: bool, terminal: bool) -> dict[str, Any]:
        compact: dict[str, Any] = {
            "reset": bool(reset),
            "episode_step": self._episode_steps,
            "physics_ticks": self.last_physics_ticks,
            "command_ticks": self.last_command_ticks,
            "success_hold_physics_ticks": self._success_hold_ticks,
            "termination_reason": self._termination_reason,
            "execution_mode": self.config.execution_mode,
            "robot_dynamics": self.config.robot_dynamics,
            "runtime": "cube_lift_fast_kernel_v1",
            "model_artifact_key": (
                None if self.model_artifact is None else self.model_artifact.key
            ),
            "model_cache_hit": (
                None if self.model_artifact is None else self.model_artifact.cache_hit
            ),
        }
        if self.info_mode == "compact" or not terminal:
            return compact
        cube = self.data.qpos[self._cube_qpos_addr : self._cube_qpos_addr + 7]
        return {
            **compact,
            "resting_center_z_m": self._resting_center_z,
            "cube_position_m": cube[:3].astype(float).tolist(),
            "cube_quaternion_wxyz": cube[3:].astype(float).tolist(),
            "cube_z_m": float(cube[2]),
            "table_contact": self._cube_touches_table(),
        }

    def _capture_settled_snapshot(self) -> None:
        self._settled_snapshot = {
            "time": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "act": self.data.act.copy(),
            "ctrl": self.data.ctrl.copy(),
            "qacc_warmstart": self.data.qacc_warmstart.copy(),
        }

    def _restore_settled_snapshot(self) -> None:
        assert self._settled_snapshot is not None
        self.mujoco.mj_resetData(self.model, self.data)
        self.data.time = float(self._settled_snapshot["time"])
        for name in ("qpos", "qvel", "act", "ctrl", "qacc_warmstart"):
            getattr(self.data, name)[:] = self._settled_snapshot[name]
        self.mujoco.mj_forward(self.model, self.data)
        self._physics_tick = self.config.settle_physics_ticks
        self._next_command_time = (
            np.floor(self.physics_time * self.config.command_hz + 1e-12) + 1.0
        ) / self.config.command_hz

    def close(self) -> None:
        """The scalar kernel owns no renderer or external process."""


__all__ = ["CubeLiftKernel", "CubeLiftKernelStep", "InfoMode"]

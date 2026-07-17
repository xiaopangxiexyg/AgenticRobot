"""A minimal state-privileged Cube-Lift Gymnasium debug environment."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim.mujoco.control import DirectQposCommand, DirectQposDebugController
from sim.mujoco.scene import MujocoScene

from .config import CubeLiftTaskConfig
from .contacts import CubeLiftContactSummary, summarize_cube_contacts
from .grasp_latch import ContactGraspLatch
from .reward import cube_out_of_bounds, lift_progress, reward_for_step
from .scheduler import CubeLiftPolicyScheduler, PolicyAdvance
from .spec import CubeLiftActionSpec, CubeLiftObservationSpec, require_gymnasium


DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH = Path(
    "sim/configs/tasks/rm65_right_cube_lift_rl_debug_v0.json"
)

try:
    _gymnasium, _spaces = require_gymnasium()
    _EnvBase = _gymnasium.Env
except ModuleNotFoundError:
    _gymnasium = None
    _spaces = None
    _EnvBase = object


class CubeLiftEnv(_EnvBase):
    """Right-arm joint-delta task: lift the center-table cube above a threshold.

    The robot is held through ``kinematic_direct_qpos_debug``. The temporary
    task uses bilateral contact to acquire a debug grasp latch, after which the
    cube follows the grasp frame until the gripper opens. This environment is
    useful for RL API, reset, observation, clock and contact-pipeline
    development—not sim2real or physical-grasp claims.
    """

    metadata = {"render_modes": [], "render_fps": 15}

    def __init__(
        self,
        config: CubeLiftTaskConfig | str | Path = DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH,
        *,
        scene: MujocoScene | None = None,
    ) -> None:
        if _gymnasium is None:
            require_gymnasium()
        self.config = (
            config if isinstance(config, CubeLiftTaskConfig) else CubeLiftTaskConfig.load(config)
        )
        contract = self.config.base_contract()
        self._validate_base_contract(contract)
        self.scene = scene if scene is not None else MujocoScene.from_contract(contract)
        self._validate_scene()

        self.action_spec = CubeLiftActionSpec()
        self.observation_spec = CubeLiftObservationSpec()
        self.action_space = self.action_spec.gym_space()
        self.observation_space = self.observation_spec.gym_space()
        self.controller = DirectQposDebugController(
            self.scene,
            contract.timing,
            active_sides=(self.config.fixed_side, self.config.controlled_side),
        )
        self.scheduler = CubeLiftPolicyScheduler(
            self.controller,
            policy_hz=self.config.policy_hz,
        )
        self._grasp_latch = ContactGraspLatch(
            self.scene,
            object_body_name=self.config.object_body_name,
            object_freejoint_name=self.config.object_freejoint_name,
            grasp_body_name=self.config.right_grasp_body_name,
            left_finger_body_name=self.config.grasp_latch_left_finger_body_name,
            right_finger_body_name=self.config.grasp_latch_right_finger_body_name,
            acquire_close=self.config.grasp_latch_acquire_close,
            release_close=self.config.grasp_latch_release_close,
            acquire_distance_m=self.config.grasp_latch_acquire_distance_m,
        )

        self._rng = np.random.default_rng()
        self._needs_reset = True
        self._command_sequence = 0
        self._episode_steps = 0
        self._success_hold_ticks = 0
        self._termination_reason: str | None = None
        self._right_target_qpos = np.asarray(self.config.right_home_qpos, dtype=float)
        self._gripper_close_by_name = dict(self.config.initial_gripper_close_by_name)
        self._last_action = np.zeros(self.action_spec.dim, dtype=np.float32)
        self._grasp_latch.reset()
        self._resting_center_z_m = float("nan")
        self._last_advance: PolicyAdvance | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if _gymnasium is None:
            require_gymnasium()
        super().reset(seed=seed)
        if options:
            unknown = sorted(set(options) - {"unused"})
            if unknown:
                raise ValueError(f"CubeLiftEnv.reset does not support options={unknown}.")
        self._rng = np.random.default_rng(seed)
        self.scene.reset_data()
        self._command_sequence = 0
        self._episode_steps = 0
        self._success_hold_ticks = 0
        self._termination_reason = None
        self._right_target_qpos = np.asarray(self.config.right_home_qpos, dtype=float)
        self._gripper_close_by_name = dict(self.config.initial_gripper_close_by_name)
        self._last_action = np.zeros(self.action_spec.dim, dtype=np.float32)
        self._grasp_latch.reset()

        self.scene.set_freejoint_state(
            self.config.object_freejoint_name,
            qpos=self.config.cube_spawn_qpos,
            qvel=np.zeros(6, dtype=float),
            forward=False,
        )
        initial_command = self._new_command(source="cube_lift_reset")
        self.scheduler.reset(initial_command)
        self._last_advance = self.scheduler.advance_physics_ticks(
            self.config.settle_physics_ticks,
            command_supplier=lambda: self._new_command(source="cube_lift_reset_hold"),
            after_tick=self._check_physics_termination,
        )
        if self._termination_reason is not None:
            raise RuntimeError(
                "CubeLift reset did not produce a valid settled scene: "
                f"{self._termination_reason}."
            )
        self._resting_center_z_m = float(self._cube_qpos()[2])
        self.scheduler.start_episode_clock()
        self._needs_reset = False
        observation = self._observation()
        return observation, self._info(reset=True)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._needs_reset:
            raise RuntimeError("CubeLiftEnv.reset() must be called before step().")
        values = self.action_spec.coerce(action)
        previous_right_target = self._right_target_qpos.copy()
        previous_right_close = float(
            self._gripper_close_by_name[self.config.right_gripper_name]
        )
        self._right_target_qpos = np.clip(
            self._right_target_qpos
            + values[:6].astype(np.float64) * np.asarray(
                self.config.right_joint_delta_rad, dtype=np.float64
            ),
            np.asarray(self.config.right_qpos_lower, dtype=float),
            np.asarray(self.config.right_qpos_upper, dtype=float),
        )
        self._gripper_close_by_name[self.config.right_gripper_name] = float(
            np.clip(
                self._gripper_close_by_name[self.config.right_gripper_name]
                + values[6] * self.config.right_gripper_delta,
                0.0,
                self.config.right_gripper_max_close,
            )
        )
        self._episode_steps += 1
        self._termination_reason = None
        self._last_advance = self.scheduler.advance_one_policy_step(
            command_supplier=lambda: self._new_command(source="cube_lift_policy"),
            after_tick=self._check_physics_termination,
        )
        # Invalid requested deltas must not accumulate behind the safety
        # boundary. The next policy action starts from the accepted target.
        self._right_target_qpos = self.controller.accepted_arm_qpos(
            self.config.controlled_side
        )
        self._gripper_close_by_name[self.config.right_gripper_name] = (
            self.controller.accepted_gripper_close(self.config.right_gripper_name)
        )
        self._last_action[:6] = np.clip(
            (self._right_target_qpos - previous_right_target)
            / np.asarray(self.config.right_joint_delta_rad, dtype=float),
            -1.0,
            1.0,
        )
        self._last_action[6] = np.clip(
            (
                self._gripper_close_by_name[self.config.right_gripper_name]
                - previous_right_close
            )
            / self.config.right_gripper_delta,
            -1.0,
            1.0,
        )

        terminated = self._termination_reason is not None
        truncated = not terminated and self._episode_steps >= self.config.max_episode_steps
        success = self._termination_reason == "height_only_lift_success"
        cube_position = self._cube_qpos()[:3]
        progress = lift_progress(
            cube_z_m=float(cube_position[2]),
            resting_center_z_m=self._resting_center_z_m,
            lift_delta_m=self.config.lift_delta_m,
        )
        reward = reward_for_step(
            progress=progress,
            action=values,
            success=success,
            config=self.config.reward,
        )
        if truncated:
            self._termination_reason = "time_limit"
        observation = self._observation()
        info = self._info(reset=False)
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """Keep the Gymnasium lifecycle explicit; this env owns no renderer yet."""

    def _validate_base_contract(self, contract) -> None:
        timing = contract.timing
        expected = {
            "policy_hz": self.config.policy_hz,
            "command_hz": self.config.command_hz,
            "physics_hz": self.config.physics_hz,
        }
        actual = {
            "policy_hz": timing.policy_hz,
            "command_hz": timing.command_hz,
            "physics_hz": timing.physics_hz,
        }
        mismatches = {
            key: (expected[key], actual[key])
            for key in expected
            if not np.isclose(expected[key], actual[key], rtol=0.0, atol=1e-12)
        }
        if mismatches:
            raise ValueError(f"CubeLift task clock disagrees with base contract: {mismatches}")
        object_spec = next(
            (item for item in contract.scene.objects if item.name == self.config.object_body_name),
            None,
        )
        if object_spec is None:
            raise ValueError(
                f"Base contract has no object {self.config.object_body_name!r} required by CubeLift."
            )
        if not np.allclose(object_spec.size_m, self.config.object_size_m, atol=1e-12):
            raise ValueError("CubeLift object_size_m disagrees with the base contract object.")

    def _validate_scene(self) -> None:
        missing_arms = {
            self.config.fixed_side,
            self.config.controlled_side,
        } - set(self.scene.arm_specs)
        if missing_arms:
            raise ValueError(f"CubeLift scene is missing arm side(s): {sorted(missing_arms)}")
        missing_grippers = {
            self.config.left_gripper_name,
            self.config.right_gripper_name,
        } - set(self.scene.gripper_specs)
        if missing_grippers:
            raise ValueError(
                f"CubeLift scene is missing configured gripper(s): {sorted(missing_grippers)}"
            )
        for body_name in (
            self.config.object_body_name,
            self.config.table_body_name,
            self.config.right_grasp_body_name,
            self.config.grasp_latch_left_finger_body_name,
            self.config.grasp_latch_right_finger_body_name,
        ):
            if not self.scene.has_body(body_name):
                raise ValueError(f"CubeLift scene is missing required body {body_name!r}.")
        # Calls through the public scene helper to validate the freejoint name/type.
        self.scene.freejoint_qpos(self.config.object_freejoint_name)

    def _new_command(self, *, source: str) -> DirectQposCommand:
        command = DirectQposCommand(
            sequence=self._command_sequence,
            qpos_by_side={
                self.config.fixed_side: self.config.left_home_qpos,
                self.config.controlled_side: self._right_target_qpos,
            },
            gripper_close_by_name=self._gripper_close_by_name,
            source=source,
        )
        self._command_sequence += 1
        return command

    def _cube_qpos(self) -> np.ndarray:
        return self.scene.freejoint_qpos(self.config.object_freejoint_name)

    def _cube_qvel(self) -> np.ndarray:
        return self.scene.freejoint_qvel(self.config.object_freejoint_name)

    def _contact_summary(self) -> CubeLiftContactSummary:
        return summarize_cube_contacts(
            self.scene.contacts(),
            object_body_name=self.config.object_body_name,
            table_body_name=self.config.table_body_name,
        )

    def _check_physics_termination(self, _tick) -> bool:
        if not self.scene.state_is_finite():
            self._termination_reason = "non_finite_simulation_state"
            return True
        self._grasp_latch.update(
            self.scene.get_gripper_close(self.config.right_gripper_name)
        )
        cube_position = self._cube_qpos()[:3]
        if cube_out_of_bounds(
            cube_position,
            lower_m=self.config.out_of_bounds_xyz_lower_m,
            upper_m=self.config.out_of_bounds_xyz_upper_m,
        ):
            self._termination_reason = "cube_out_of_bounds"
            return True
        active_collision = self.controller.last_active_collision_check
        if active_collision is not None and not active_collision.safe:
            self._termination_reason = "robot_environment_penetration"
            return True
        contacts = self._contact_summary()
        above_target = float(cube_position[2]) >= (
            self._resting_center_z_m + self.config.lift_delta_m
        )
        if above_target and not contacts.table_contact:
            self._success_hold_ticks += 1
        else:
            self._success_hold_ticks = 0
        if self._success_hold_ticks >= self.config.success_hold_physics_ticks:
            self._termination_reason = "height_only_lift_success"
            return True
        return False

    def _observation(self) -> np.ndarray:
        cube_qpos = self._cube_qpos()
        cube_qvel = self._cube_qvel()
        grasp_position = self.scene.body_pose(self.config.right_grasp_body_name)[:3, 3]
        progress = self._episode_steps / self.config.max_episode_steps
        observation = np.concatenate(
            (
                self.scene.get_arm_qpos(self.config.controlled_side),
                np.asarray(
                    [self.scene.get_gripper_close(self.config.right_gripper_name)],
                    dtype=float,
                ),
                cube_qpos,
                cube_qvel,
                cube_qpos[:3] - grasp_position,
                self._last_action,
                np.asarray([progress], dtype=float),
            )
        )
        return self.observation_spec.coerce(observation)

    def _info(self, *, reset: bool) -> dict[str, Any]:
        cube_qpos = self._cube_qpos()
        contacts = self._contact_summary()
        last_tick = self.controller.ticks[-1] if self.controller.ticks else None
        return {
            **self.config.to_metadata(),
            "reset": bool(reset),
            "episode_step": self._episode_steps,
            "resting_center_z_m": self._resting_center_z_m,
            "cube_position_m": cube_qpos[:3].astype(float).tolist(),
            "cube_quaternion_wxyz": cube_qpos[3:].astype(float).tolist(),
            "cube_z_m": float(cube_qpos[2]),
            "termination_reason": self._termination_reason,
            "success_hold_physics_ticks": self._success_hold_ticks,
            "grasp_latched": self._grasp_latch.active,
            "grasp_latch_acquisition_count": self._grasp_latch.acquisition_count,
            "controller_sequence": None if last_tick is None else last_tick.active_command_sequence,
            "controller_physics_tick": None if last_tick is None else last_tick.physics_tick,
            "scheduler": None
            if self._last_advance is None
            else {
                "physics_ticks": self._last_advance.physics_ticks,
                "command_ticks": self._last_advance.command_ticks,
                "stopped_early": self._last_advance.stopped_early,
                "final_physics_time_sec": self._last_advance.final_physics_time_sec,
            },
            "contacts": contacts.to_info(),
            "safety": (
                None
                if self.controller.last_safety_result is None
                else self.controller.last_safety_result.to_dict()
            ),
            "active_collision": (
                None
                if self.controller.last_active_collision_check is None
                else self.controller.last_active_collision_check.to_dict()
            ),
        }


__all__ = ["CubeLiftEnv", "DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH"]

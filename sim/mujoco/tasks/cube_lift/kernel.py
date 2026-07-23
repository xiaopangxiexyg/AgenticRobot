"""Low-allocation scalar kernel for the temporary Cube-Lift task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from robot_motion.runtime import EffectorCommand, TeleopCommand
from sim.contracts import TimingSpec

from ...adapters import MujocoPositionExecutionConfig, MujocoPositionExecutionRig
from ...collision_safety import project_grippers_then_arms
from ...scene import MujocoCameraRenderer, MujocoScene
from .config import (
    CONTACT_LATCH_DEBUG_GRASP_MODE,
    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
    NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1,
    CubeLiftTaskConfig,
)
from .env import DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH
from .grasp_latch import ContactGraspLatch
from .pose_action import CubeLiftPoseActionController


InfoMode = Literal["compact", "terminal"]


def _wrist_rgb_health(image: np.ndarray) -> dict[str, float | bool]:
    """Return cheap image-signal diagnostics without touching policy inputs.

    These checks deliberately do not use simulator object masks or privileged
    geometry.  They only catch transport/render failures such as an all-black,
    all-flat, or fully saturated frame; object visibility still needs a manual
    camera probe at representative arm poses.
    """

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("wrist RGB health expects an HWC RGB image.")
    finite = bool(np.all(np.isfinite(values)))
    if not finite:
        return {
            "wrist_rgb_mean": float("nan"),
            "wrist_rgb_std": float("nan"),
            "wrist_rgb_nonzero_fraction": 0.0,
            "wrist_rgb_saturated_fraction": 0.0,
            "wrist_rgb_health_ok": False,
        }
    nonzero_fraction = float(np.mean(np.any(values > 2.0, axis=-1)))
    saturated_fraction = float(np.mean(np.all(values >= 250.0, axis=-1)))
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "wrist_rgb_mean": mean,
        "wrist_rgb_std": std,
        "wrist_rgb_nonzero_fraction": nonzero_fraction,
        "wrist_rgb_saturated_fraction": saturated_fraction,
        "wrist_rgb_health_ok": bool(
            mean >= 10.0
            and std >= 5.0
            and nonzero_fraction >= 0.05
            and saturated_fraction < 0.99
        ),
    }


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
    contact checks. The execution backend is selected by the versioned task
    config: legacy direct-qpos remains available, while actuated-position uses
    the shared 15/60/120 Hz planning path and physical MuJoCo arm actuators.
    Physical normal/expert policies emit TCP pose increments; only the
    explicitly retained legacy task emits joint deltas.
    """

    action_dim = 7
    def __init__(
        self,
        config: CubeLiftTaskConfig | str | Path = DEFAULT_CUBE_LIFT_TASK_CONFIG_PATH,
        *,
        scene: MujocoScene | None = None,
        info_mode: InfoMode = "terminal",
        use_settled_snapshot: bool = True,
        use_model_cache: bool = True,
        model_cache_dir: str | Path | None = None,
        include_training_reward_state: bool = False,
    ) -> None:
        if info_mode not in {"compact", "terminal"}:
            raise ValueError("info_mode must be 'compact' or 'terminal'.")
        self.config = (
            config if isinstance(config, CubeLiftTaskConfig) else CubeLiftTaskConfig.load(config)
        )
        contract = self.config.base_contract()
        position_profile = self.config.position_actuation_profile()
        gripper_profile = self.config.gripper_actuation_profile()
        self.model_artifact = None
        if scene is not None:
            self.scene = scene
        elif use_model_cache:
            from ...artifacts import ModelArtifactCache

            self.scene, self.model_artifact = ModelArtifactCache(model_cache_dir).load_scene(
                contract,
                position_actuation_profile=position_profile,
                gripper_actuation_profile=gripper_profile,
            )
        else:
            self.scene = MujocoScene.from_contract(
                contract,
                position_actuation_profile=position_profile,
                gripper_actuation_profile=gripper_profile,
            )
        self.model = self.scene.model
        self.data = self.scene.data
        self.mujoco = self.scene.mujoco
        self.info_mode = info_mode
        self.use_settled_snapshot = bool(use_settled_snapshot)
        self.include_training_reward_state = bool(include_training_reward_state)

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
        self.observation_dim = self.config.observation_dim
        self._grasp_latch = (
            ContactGraspLatch(
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
            if self.config.grasp_mode == CONTACT_LATCH_DEBUG_GRASP_MODE
            else None
        )

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
        self._observation_buffer = np.empty(self.observation_dim, dtype=np.float32)
        self._training_reward_state_buffer = np.empty(31, dtype=np.float32)
        self._previous_observation_right_qpos: np.ndarray | None = None
        self._previous_observation_right_close: float | None = None
        self._last_wrist_rgb_health: dict[str, float | bool] = {
            "wrist_rgb_mean": float("nan"),
            "wrist_rgb_std": float("nan"),
            "wrist_rgb_nonzero_fraction": 0.0,
            "wrist_rgb_saturated_fraction": 0.0,
            "wrist_rgb_health_ok": False,
        }
        self._wrist_rgb_renderer: MujocoCameraRenderer | None = None
        self._wrist_rgb_proprio_dim = self.config.proprio_observation_dim
        if self.config.wrist_rgb_enabled:
            if self.config.wrist_rgb_camera_name not in self.scene.camera_names:
                raise ValueError(
                    "CubeLift wrist RGB camera is not present in the base scene: "
                    f"requested={self.config.wrist_rgb_camera_name!r}, "
                    f"available={self.scene.camera_names!r}."
                )
            self._wrist_rgb_renderer = MujocoCameraRenderer(
                self.scene,
                width=self.config.wrist_rgb_width,
                height=self.config.wrist_rgb_height,
            )
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
        self._collision_projector = self.scene.collision_projector()
        self._accepted_controlled_qpos = self.scene.kinematic_controlled_qpos()
        self._last_safety_result = None
        self._last_active_collision_check = None
        self._actuated_rig: MujocoPositionExecutionRig | None = None
        self._pose_action_controller: CubeLiftPoseActionController | None = None
        if self.config.uses_pose_action:
            self._pose_action_controller = CubeLiftPoseActionController(
                self.scene,
                self.config,
            )
        self._last_pose_action_step = None
        if self._grasp_latch is not None:
            self._grasp_latch.reset()

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
        self._last_safety_result = None
        self._last_active_collision_check = None
        if self._grasp_latch is not None:
            self._grasp_latch.reset()
        self._close_actuated_rig()

        if self.use_settled_snapshot and self._settled_snapshot is not None:
            self._restore_settled_snapshot()
            self._start_actuated_rig()
            self.last_physics_ticks = 0
            self.last_command_ticks = 0
        else:
            self.scene.reset_data()
            self.data.qpos[self._cube_qpos_addr : self._cube_qpos_addr + 7] = self._spawn_qpos
            self.data.qvel[self._cube_dof_addr : self._cube_dof_addr + 6] = self._zero_cube_qvel
            self._physics_tick = 0
            self._next_command_time = 1.0 / self.config.command_hz
            self._accepted_controlled_qpos = self.scene.kinematic_target_vector(
                self.scene.kinematic_controlled_qpos(),
                {
                    self.config.fixed_side: self._left_home,
                    self.config.controlled_side: self._right_home,
                },
                gripper_close_by_name={
                    name: float(close)
                    for name, close in zip(self._gripper_names, self._gripper_close)
                },
            )
            if self._collision_projector is not None:
                initial_check = self._collision_projector.check_qpos(
                    self.data, self._accepted_controlled_qpos
                )
                if not initial_check.safe:
                    raise RuntimeError(
                        "CubeLift fast reset target is collision-unsafe: "
                        f"{initial_check.to_dict()}"
                    )
            if self.config.uses_actuated_position:
                # Reset is the only point where an actuated rollout may place
                # the robot directly. During rollout only data.ctrl and
                # mj_step are used for the arm; the gripper remains the
                # explicitly declared prescribed-qpos debug subsystem.
                self.data.qpos[self.scene.kinematic_qpos_indices] = (
                    self._accepted_controlled_qpos
                )
                self.data.qvel[self._controlled_dof_idx] = 0.0
                self.mujoco.mj_forward(self.model, self.data)
            self._start_actuated_rig()
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

        if self._pose_action_controller is not None:
            # Anchor both the canonical pose planner and the policy's held TCP
            # target after the settled physical actuator state is available.
            self._pose_action_controller.reset()
            self._last_pose_action_step = None
            self._right_target[:] = self.scene.get_arm_qpos(self.config.controlled_side)

        self._resting_center_z = float(self.data.qpos[self._cube_qpos_addr + 2])
        self._previous_observation_right_qpos = self.scene.get_arm_qpos(
            self.config.controlled_side
        ).astype(np.float64, copy=True)
        self._previous_observation_right_close = self._measured_gripper_close(1)
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
        previous_right_target = self._right_target.copy()
        previous_right_close = float(self._gripper_close[1])
        if self._pose_action_controller is not None:
            pose_step = self._pose_action_controller.apply(values)
            self._last_pose_action_step = pose_step
            planned_right = np.asarray(
                pose_step.ik_result.qpos_by_side.get(
                    self.config.controlled_side,
                    previous_right_target,
                ),
                dtype=np.float64,
            ).reshape(6)
            if pose_step.ik_result.hold or not pose_step.ik_result.success:
                planned_right = previous_right_target
            np.clip(planned_right, self._right_lower, self._right_upper, out=self._right_target)
        else:
            np.clip(
                self._right_target + values[:6] * self._right_delta,
                self._right_lower,
                self._right_upper,
                out=self._right_target,
            )
        self._gripper_close[1] = np.clip(
            self._gripper_close[1] + float(values[6]) * self.config.right_gripper_delta,
            0.0,
            self.config.right_gripper_max_close,
        )
        qpos_by_side = {
            self.config.fixed_side: self._left_home,
            self.config.controlled_side: self._right_target,
        }
        gripper_close_by_name = {
            name: float(close)
            for name, close in zip(self._gripper_names, self._gripper_close)
        }
        measured_controlled_qpos = self.scene.kinematic_controlled_qpos()
        projection_start = (
            measured_controlled_qpos
            if self.config.uses_actuated_position
            else self._accepted_controlled_qpos
        )
        requested = self.scene.kinematic_target_vector(
            projection_start,
            qpos_by_side,
            gripper_close_by_name=gripper_close_by_name,
        )
        clearance_violation = False
        if np.array_equal(requested, projection_start):
            self._accepted_controlled_qpos = requested.copy()
            self._last_safety_result = None
        elif self._collision_projector is None:
            self._accepted_controlled_qpos = requested
            self._last_safety_result = None
        else:
            self._last_safety_result = project_grippers_then_arms(
                self.scene,
                self._collision_projector,
                self.data,
                start_qpos=projection_start,
                qpos_by_side=qpos_by_side,
                gripper_close_by_name=gripper_close_by_name,
            )
            if self._last_safety_result.status == "invalid_safe_state":
                if self.config.uses_actuated_gripper:
                    # Physical tracking/contact may leave the measured state
                    # just inside the preventive clearance envelope without
                    # hard penetration. Reject the complete new target and
                    # command a measured-state hold so the policy can retreat
                    # on its next step. The active-contact oracle still
                    # terminates any actual hard penetration every physics
                    # tick.
                    self._accepted_controlled_qpos = projection_start.copy()
                else:
                    # A prescribed state has no physical lag to recover on a
                    # later tick, so retain the legacy fail-safe terminal.
                    clearance_violation = True
            else:
                self._accepted_controlled_qpos = (
                    self._last_safety_result.accepted_qpos.copy()
                )
        self._right_target[:] = self.scene.arm_qpos_from_kinematic_vector(
            self._accepted_controlled_qpos, self.config.controlled_side
        )
        self._gripper_close[:] = [
            self.scene.gripper_close_from_kinematic_vector(
                self._accepted_controlled_qpos, name
            )
            for name in self._gripper_names
        ]
        if self._pose_action_controller is not None:
            # ``last_action`` is policy lineage in pose coordinates.  It must
            # not silently become a qpos delta merely because the downstream
            # executor consumes qref.
            self._last_action[:6] = values[:6]
        else:
            self._last_action[:6] = np.clip(
                (self._right_target - previous_right_target) / self._right_delta,
                -1.0,
                1.0,
            )
        self._last_action[6] = np.clip(
            (self._gripper_close[1] - previous_right_close)
            / self.config.right_gripper_delta,
            -1.0,
            1.0,
        )
        self._episode_steps += 1
        self._termination_reason = (
            "robot_environment_clearance_violation"
            if clearance_violation
            else None
        )
        if not clearance_violation:
            self._submit_actuated_reference()
        physics_ticks = 0
        command_ticks = 0
        while (
            not clearance_violation
            and self.physics_time + 1e-12 < self._next_policy_time
        ):
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
        if (
            self._pose_action_controller is not None
            and not clearance_violation
            and self._termination_reason is None
        ):
            # The simulator's full-scene clearance projection is authoritative
            # for MuJoCo.  Re-anchor the shared pose planner to the measured
            # accepted state before the next policy reference while retaining
            # the held TCP target. MuJoCo's soft joint-range constraint can
            # leave a tiny numerical residual outside the canonical RM65
            # limits; the sim adapter clips only that residual for the planner
            # snapshot. A larger violation ends this episode instead of
            # allowing a worker exception to abort the whole vector rollout.
            planner_sync = self._pose_action_controller.sync_planner_from_scene()
            if not planner_sync.safe:
                self._termination_reason = (
                    "sim_collision_margin_violation"
                    if planner_sync.failure_reason == "collision_margin_low"
                    else "sim_joint_position_limit_violation"
                )
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
        self.data.qpos[self.scene.kinematic_qpos_indices] = self._accepted_controlled_qpos
        self.data.qvel[self._controlled_dof_idx] = 0.0

    def _write_prescribed_grippers(self) -> None:
        for index, close in enumerate(self._gripper_close):
            self.data.qpos[self._gripper_qpos_idx[index]] = (
                self._gripper_open[index] + close * self._gripper_span[index]
            )
            self.data.qvel[self._gripper_dof_idx[index]] = 0.0

    def _measured_gripper_close(self, index: int) -> float:
        return float(self.scene.get_gripper_close(self._gripper_names[index]))

    def _start_actuated_rig(self) -> None:
        if not self.config.uses_actuated_position:
            return
        profile = self.config.position_actuation_profile()
        assert profile is not None
        timing = TimingSpec(
            sample_hz=self.config.policy_hz,
            policy_hz=self.config.policy_hz,
            reference_hz=self.config.reference_hz,
            ik_hz=self.config.reference_hz,
            command_hz=self.config.command_hz,
            physics_hz=self.config.physics_hz,
            parameter_source="cube_lift_actuated_task_config",
        )
        self._actuated_rig = MujocoPositionExecutionRig(
            self.scene,
            MujocoPositionExecutionConfig(
                timing=timing,
                active_sides=(self.config.fixed_side, self.config.controlled_side),
                actuation_profile=profile,
                gripper_actuation_profile=self.config.gripper_actuation_profile(),
                realtime_worker=False,
            ),
            monotonic_clock=lambda: self.physics_time,
            sleep=lambda _seconds: None,
        )
        self._actuated_rig.start()

    def _close_actuated_rig(self) -> None:
        if self._actuated_rig is not None:
            self._actuated_rig.close()
            self._actuated_rig = None

    def _submit_actuated_reference(self) -> None:
        if not self.config.uses_actuated_position:
            return
        if self._actuated_rig is None:
            raise RuntimeError("Actuated CubeLift execution rig is not initialized.")
        command = TeleopCommand(timestamp=self.physics_time)
        for side, target in (
            (self.config.fixed_side, self._left_home),
            (self.config.controlled_side, self._right_target),
        ):
            command.add_command(
                EffectorCommand(
                    effector_name=f"{side}_arm",
                    side=side,
                    command_type="joint_vector",
                    value=target,
                    metadata={
                        "source": (
                            "cube_lift_policy_tcp_pose_ik_qref"
                            if self._pose_action_controller is not None
                            else "cube_lift_policy_qref"
                        ),
                        "policy_action_layout": self.config.action_layout,
                    },
                )
            )
        if self.config.uses_actuated_gripper:
            for side, close in (
                (self.config.fixed_side, self._gripper_close[0]),
                (self.config.controlled_side, self._gripper_close[1]),
            ):
                command.add_command(
                    EffectorCommand(
                        effector_name=f"{side}_hand",
                        side=side,
                        command_type="gripper_scalar",
                        value=(float(close),),
                        metadata={"source": "cube_lift_policy_gripper_target"},
                    )
                )
        self._actuated_rig.send_commands(command)

    def _step_physics(self) -> None:
        if self.config.uses_actuated_position:
            if self._actuated_rig is None:
                raise RuntimeError("Actuated CubeLift execution rig is not initialized.")
            if not self.config.uses_actuated_gripper:
                self._write_prescribed_grippers()
            self._actuated_rig.step_physics()
            if not self.config.uses_actuated_gripper:
                self._write_prescribed_grippers()
                self.mujoco.mj_forward(self.model, self.data)
        else:
            self._write_direct_state()
            # mj_step performs the pre-integration forward pipeline. One
            # post-step rewrite keeps the debug robot truly prescribed while
            # contact forces still advance the free cube.
            self.mujoco.mj_step(self.model, self.data)
            self._write_direct_state()
            self.mujoco.mj_forward(self.model, self.data)
        latch_close = (
            self._measured_gripper_close(1)
            if self.config.uses_actuated_gripper
            else float(self._gripper_close[1])
        )
        if self._grasp_latch is not None:
            self._grasp_latch.update(latch_close)
        self._last_active_collision_check = (
            None
            if self._collision_projector is None
            else self._collision_projector.oracle.check_contacts(self.data)
        )
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
        if (
            self._last_active_collision_check is not None
            and not self._last_active_collision_check.safe
        ):
            self._termination_reason = "robot_environment_penetration"
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
        privileged = self._training_reward_state()
        if self.config.policy_class == "normal":
            out = self._observation_buffer
            out[:7] = privileged[0:7]
            if self.config.normal_proprio_profile in {
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1,
                NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_NO_EPISODE_CLOCK_V2,
            }:
                measured_qpos = privileged[0:6].astype(np.float64, copy=False)
                measured_close = float(privileged[6])
                previous_qpos = (
                    measured_qpos
                    if self._previous_observation_right_qpos is None
                    else self._previous_observation_right_qpos
                )
                previous_close = (
                    measured_close
                    if self._previous_observation_right_close is None
                    else self._previous_observation_right_close
                )
                out[7:13] = measured_qpos - previous_qpos
                out[13] = measured_close - previous_close
                out[14:20] = self._right_target - measured_qpos
                out[20] = float(self._gripper_close[1] - measured_close)
                out[21:28] = privileged[23:30]
                if (
                    self.config.normal_proprio_profile
                    == NORMAL_PROPRIO_PROFILE_MOTION_TRACKING_V1
                ):
                    out[28] = privileged[30]
                self._previous_observation_right_qpos = measured_qpos.copy()
                self._previous_observation_right_close = measured_close
            else:
                out[7:14] = privileged[23:30]
                out[14] = privileged[30]
        else:
            self._observation_buffer[:31] = privileged
        if self._wrist_rgb_renderer is not None:
            # Renderer output is HWC uint8.  The policy contract uses CHW
            # float32 pixels in [0, 1], appended after the proprio/state part.
            image = self._wrist_rgb_renderer.render(self.config.wrist_rgb_camera_name)
            expected = (self.config.wrist_rgb_height, self.config.wrist_rgb_width, 3)
            if image.shape != expected:
                raise RuntimeError(
                    "CubeLift wrist RGB renderer changed shape: "
                    f"expected={expected}, got={image.shape}."
                )
            image_values = np.asarray(image, dtype=np.float32).transpose(2, 0, 1)
            image_values *= np.float32(1.0 / 255.0)
            self._observation_buffer[self._wrist_rgb_proprio_dim :] = image_values.reshape(-1)
            self._last_wrist_rgb_health = _wrist_rgb_health(image)
        else:
            # State-only layouts are exactly the historical 15/31-D vectors.
            self._last_wrist_rgb_health = {
                "wrist_rgb_mean": float("nan"),
                "wrist_rgb_std": float("nan"),
                "wrist_rgb_nonzero_fraction": 0.0,
                "wrist_rgb_saturated_fraction": 0.0,
                "wrist_rgb_health_ok": False,
            }
        return self._observation_buffer.copy()

    def _training_reward_state(self) -> np.ndarray:
        """Return simulator-only reward features, never the normal policy input."""

        out = self._training_reward_state_buffer
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
            raise RuntimeError("CubeLift training reward state became non-finite.")
        return out.copy()

    def _info(self, *, reset: bool, terminal: bool) -> dict[str, Any]:
        tracking_error = self._right_target - self.data.qpos[self._right_qpos_idx]
        measured_right_close = self._measured_gripper_close(1)
        compact: dict[str, Any] = {
            "reset": bool(reset),
            "episode_step": self._episode_steps,
            "physics_ticks": self.last_physics_ticks,
            "command_ticks": self.last_command_ticks,
            "success_hold_physics_ticks": self._success_hold_ticks,
            "termination_reason": self._termination_reason,
            "policy_class": self.config.policy_class,
            "observation_mode": self.config.observation_mode,
            "observation_layout": self.config.observation_layout,
            "normal_proprio_profile": self.config.normal_proprio_profile,
            "wrist_rgb_enabled": self.config.wrist_rgb_enabled,
            "wrist_rgb_camera_name": (
                self.config.wrist_rgb_camera_name
                if self.config.wrist_rgb_enabled
                else None
            ),
            "wrist_rgb_spatial_pool_size": self.config.wrist_rgb_spatial_pool_size,
            "action_layout": self.config.action_layout,
            "pose_control_frame": self.config.pose_control_frame,
            "grasp_latched": bool(
                self._grasp_latch is not None and self._grasp_latch.active
            ),
            "grasp_latch_acquisition_count": (
                0 if self._grasp_latch is None else self._grasp_latch.acquisition_count
            ),
            "execution_mode": self.config.execution_mode,
            "robot_dynamics": self.config.robot_dynamics,
            "gripper_dynamics": self.config.gripper_dynamics,
            "right_gripper_target_close": float(self._gripper_close[1]),
            "right_gripper_measured_close": measured_right_close,
            "right_gripper_tracking_error": float(
                self._gripper_close[1] - measured_right_close
            ),
            "gripper_actuation_profile_name": (
                self.config.gripper_actuation_profile_name
            ),
            "position_actuation_profile_name": (
                self.config.position_actuation_profile_name
            ),
            "pose_ik_status": (
                None
                if self._last_pose_action_step is None
                else self._last_pose_action_step.ik_result.status
            ),
            "pose_ik_hold": (
                None
                if self._last_pose_action_step is None
                else bool(self._last_pose_action_step.ik_result.hold)
            ),
            "pose_ik_solver": (
                None
                if self._pose_action_controller is None
                else self._pose_action_controller.solver_name
            ),
            "pose_ik_limit_recovered": bool(
                self._pose_action_controller is not None
                and self._pose_action_controller.last_planner_sync.recovered
            ),
            "pose_ik_limit_violation_rad": float(
                0.0
                if self._pose_action_controller is None
                else self._pose_action_controller.last_planner_sync.max_violation_rad
            ),
            "pose_ik_limit_violation_side": (
                None
                if self._pose_action_controller is None
                else self._pose_action_controller.last_planner_sync.violating_side
            ),
            "pose_ik_limit_violation_joint": (
                None
                if self._pose_action_controller is None
                else self._pose_action_controller.last_planner_sync.violating_joint_index
            ),
            "pose_ik_collision_min_distance_m": (
                None
                if self._pose_action_controller is None
                else float(self._pose_action_controller.planner_collision_min_distance_m)
            ),
            "pose_ik_sync_failure_reason": (
                None
                if self._pose_action_controller is None
                else self._pose_action_controller.last_planner_sync.failure_reason
            ),
            "arm_tracking_error_rad_mean": float(np.mean(np.abs(tracking_error))),
            "arm_tracking_error_rad_max": float(np.max(np.abs(tracking_error))),
            "runtime": "cube_lift_fast_kernel_v2",
            "model_artifact_key": (
                None if self.model_artifact is None else self.model_artifact.key
            ),
            "model_cache_hit": (
                None if self.model_artifact is None else self.model_artifact.cache_hit
            ),
            "safety_modified": bool(
                self._last_safety_result is not None
                and self._last_safety_result.modified
            ),
            "safety_recovery": bool(
                self._last_safety_result is not None
                and self._last_safety_result.status == "recovered"
            ),
            "safety_projection_alpha": float(
                1.0
                if self._last_safety_result is None
                else self._last_safety_result.projection_alpha
            ),
            "safety_minimum_distance_m": float(
                float("inf")
                if self._last_safety_result is None
                else self._last_safety_result.check.minimum_distance_m
            ),
        }
        if self.config.wrist_rgb_enabled:
            compact.update(self._last_wrist_rgb_health)
        if self.include_training_reward_state:
            compact["training_reward_state"] = self._training_reward_state()
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
            "safety": (
                None
                if self._last_safety_result is None
                else self._last_safety_result.to_dict()
            ),
            "active_collision": (
                None
                if self._last_active_collision_check is None
                else self._last_active_collision_check.to_dict()
            ),
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
        self._accepted_controlled_qpos = self.scene.kinematic_controlled_qpos()
        self._physics_tick = self.config.settle_physics_ticks
        self._next_command_time = (
            np.floor(self.physics_time * self.config.command_hz + 1e-12) + 1.0
        ) / self.config.command_hz

    def close(self) -> None:
        self._close_actuated_rig()
        if self._wrist_rgb_renderer is not None:
            self._wrist_rgb_renderer.close()
            self._wrist_rgb_renderer = None


__all__ = ["CubeLiftKernel", "CubeLiftKernelStep", "InfoMode"]

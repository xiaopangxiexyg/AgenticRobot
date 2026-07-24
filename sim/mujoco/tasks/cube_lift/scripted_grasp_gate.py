"""Deterministic simulator-only feasibility gate for physical Cube-Lift.

This controller reads privileged cube/grasp state and is therefore a
diagnostic oracle, never a normal-policy observation or deployable policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    BILATERAL_CONTACT_LIFT_SUCCESS_REASON,
    PHYSICAL_CONTACT_GRASP_MODE,
    CubeLiftTaskConfig,
)
from .fast_env import FastCubeLiftEnv


@dataclass(frozen=True)
class ScriptedPhysicalGraspResult:
    success: bool
    steps: int
    bilateral_contact_steps: int
    max_cube_lift_m: float
    termination_reason: str | None
    pose_ik_solver: str | None


def run_scripted_physical_grasp_gate(
    config: CubeLiftTaskConfig | str | Path,
    *,
    seed: int = 1,
    required_contact_steps: int = 5,
    lift_step_m: float = 0.004,
    use_model_cache: bool = False,
) -> ScriptedPhysicalGraspResult:
    """Approach, close on bilateral contact, then lift through the task goal."""

    task_config = (
        config if isinstance(config, CubeLiftTaskConfig) else CubeLiftTaskConfig.load(config)
    )
    if task_config.grasp_mode != PHYSICAL_CONTACT_GRASP_MODE:
        raise ValueError("The physical grasp gate requires grasp_mode='physical_contact'.")
    if not task_config.success_requires_bilateral_contact:
        raise ValueError("The physical grasp gate requires bilateral-contact success.")
    if required_contact_steps <= 0:
        raise ValueError("required_contact_steps must be positive.")
    if not np.isfinite(lift_step_m) or not 0.0 < lift_step_m <= max(
        task_config.right_tcp_delta_m
    ):
        raise ValueError("lift_step_m must be positive and within the TCP action bound.")
    closed_height = float(
        task_config.reward_shaping.get("closed_grasp_height_offset_m", 0.0)
    )
    if closed_height <= 0.0:
        raise ValueError(
            "reward_shaping.closed_grasp_height_offset_m must be positive."
        )

    env = FastCubeLiftEnv(
        task_config,
        include_training_reward_state=True,
        use_model_cache=use_model_cache,
    )
    try:
        _observation, info = env.reset(seed=seed)
        base_rotation = env.kernel.scene.body_pose("right_arm_base_link")[:3, :3]
        reset_has_contact = bool(
            info.get("scripted_contact_reset", False)
            and info.get("bilateral_finger_cube_contact", False)
        )
        phase = "lift" if reset_has_contact else "approach"
        consecutive_contact_steps = (
            required_contact_steps if reset_has_contact else 0
        )
        bilateral_contact_steps = 0
        resting_cube_z = float(info["training_reward_state"][9])
        max_cube_z = resting_cube_z
        termination_reason: str | None = None

        for step_index in range(task_config.max_episode_steps):
            state = np.asarray(info["training_reward_state"])
            cube_position = state[7:10]
            grasp_position = cube_position - state[20:23]
            grasp_target = cube_position + np.asarray(
                (0.0, 0.0, closed_height)
            )
            if phase == "approach":
                target = grasp_target
                if np.linalg.norm(target - grasp_position) < 0.035:
                    phase = "close"
            elif phase == "close":
                target = grasp_target
                if bool(info["bilateral_finger_cube_contact"]):
                    consecutive_contact_steps += 1
                else:
                    consecutive_contact_steps = 0
                if consecutive_contact_steps >= required_contact_steps:
                    phase = "lift"
            else:
                target = grasp_position + np.asarray(
                    (0.0, 0.0, lift_step_m)
                )

            action = np.zeros(7, dtype=np.float32)
            action[:3] = np.clip(
                (base_rotation.T @ (target - grasp_position))
                / np.asarray(task_config.right_tcp_delta_m),
                -1.0,
                1.0,
            )
            if phase == "close":
                action[6] = 1.0

            _observation, _reward, terminated, truncated, info = env.step(action)
            bilateral_contact_steps += int(
                bool(info["bilateral_finger_cube_contact"])
            )
            max_cube_z = max(
                max_cube_z, float(info["training_reward_state"][9])
            )
            termination_reason = info.get("termination_reason")
            if terminated or truncated:
                steps = step_index + 1
                break
        else:
            steps = task_config.max_episode_steps

        return ScriptedPhysicalGraspResult(
            success=(
                termination_reason
                == BILATERAL_CONTACT_LIFT_SUCCESS_REASON
            ),
            steps=steps,
            bilateral_contact_steps=bilateral_contact_steps,
            max_cube_lift_m=max_cube_z - resting_cube_z,
            termination_reason=termination_reason,
            pose_ik_solver=info.get("pose_ik_solver"),
        )
    finally:
        env.close()


__all__ = [
    "ScriptedPhysicalGraspResult",
    "run_scripted_physical_grasp_gate",
]

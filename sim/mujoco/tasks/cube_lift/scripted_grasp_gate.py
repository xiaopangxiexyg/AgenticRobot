"""Deterministic simulator-only feasibility gate for physical Cube-Lift.

This controller reads privileged cube/grasp state and is therefore a
diagnostic oracle, never a normal-policy observation or deployable policy.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (
    PHYSICAL_CONTACT_GRASP_MODE,
    REACH_TASK_OBJECTIVE,
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
    safety_intervention_steps: int
    fully_rejected_steps: int
    safety_intervention_rate: float
    fully_rejected_rate: float
    safety_intervention_steps_by_phase: tuple[tuple[str, int], ...]
    safety_blocking_category_counts: tuple[tuple[str, int], ...]
    safety_blocking_phase_category_counts: tuple[tuple[str, int], ...]
    safety_blocking_body_pair_counts: tuple[tuple[str, int], ...]
    safety_blocking_phase_body_pair_counts: tuple[tuple[str, int], ...]


def run_scripted_physical_grasp_gate(
    config: CubeLiftTaskConfig | str | Path,
    *,
    seed: int = 1,
    required_contact_steps: int = 5,
    lift_step_m: float = 0.004,
    use_model_cache: bool = False,
) -> ScriptedPhysicalGraspResult:
    """Drive a privileged oracle through the configured reach/contact/lift goal."""

    task_config = (
        config if isinstance(config, CubeLiftTaskConfig) else CubeLiftTaskConfig.load(config)
    )
    if task_config.grasp_mode != PHYSICAL_CONTACT_GRASP_MODE:
        raise ValueError("The physical grasp gate requires grasp_mode='physical_contact'.")
    if (
        task_config.task_objective != REACH_TASK_OBJECTIVE
        and not task_config.success_requires_bilateral_contact
    ):
        raise ValueError("The physical grasp gate requires bilateral-contact success.")
    if required_contact_steps <= 0:
        raise ValueError("required_contact_steps must be positive.")
    if not np.isfinite(lift_step_m) or not 0.0 < lift_step_m <= max(
        task_config.right_tcp_delta_m
    ):
        raise ValueError("lift_step_m must be positive and within the TCP action bound.")
    grasp_height = float(
        task_config.reach_target_height_offset_m
        if task_config.task_objective == REACH_TASK_OBJECTIVE
        else task_config.reward_shaping.get("closed_grasp_height_offset_m", 0.0)
    )
    if grasp_height <= 0.0:
        raise ValueError(
            "The configured scripted grasp/reach height must be positive."
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
        last_public_gripper_action = 0.0
        consecutive_contact_steps = (
            required_contact_steps if reset_has_contact else 0
        )
        bilateral_contact_steps = 0
        safety_intervention_steps = 0
        fully_rejected_steps = 0
        phase_interventions: Counter[str] = Counter()
        blocking_categories: Counter[str] = Counter()
        blocking_phase_categories: Counter[str] = Counter()
        blocking_body_pairs: Counter[str] = Counter()
        blocking_phase_body_pairs: Counter[str] = Counter()
        # Use the task's pre-scripted-reset settled height, exactly like the
        # lift success predicate.  A scripted pregrasp/contact reset can move
        # the cube by a fraction of a millimetre; using that later pose as the
        # diagnostic baseline made a successful lift appear slightly short.
        resting_cube_z = float(info["resting_center_z_m"])
        max_cube_z = resting_cube_z
        termination_reason: str | None = None

        for step_index in range(task_config.max_episode_steps):
            entered_lift = False
            state = np.asarray(info["training_reward_state"])
            cube_position = state[7:10]
            grasp_position = cube_position - state[20:23]
            grasp_target = cube_position + np.asarray(
                (0.0, 0.0, grasp_height)
            )
            if task_config.task_objective == REACH_TASK_OBJECTIVE:
                target = grasp_target
            elif phase == "approach":
                if np.linalg.norm(grasp_target - grasp_position) < 0.035:
                    phase = "close"
            elif phase == "close":
                if bool(info["bilateral_finger_cube_contact"]):
                    consecutive_contact_steps += 1
                else:
                    consecutive_contact_steps = 0
                if bool(info.get("post_contact_phase_active", False)) or (
                    not task_config.post_contact_phase_enabled
                    and consecutive_contact_steps >= required_contact_steps
                ):
                    phase = "lift"
                    entered_lift = True
            if (
                task_config.task_objective != REACH_TASK_OBJECTIVE
                and phase == "lift"
            ):
                if task_config.post_contact_phase_enabled:
                    # Keep deployable cube/TCP xy centering active for the
                    # direct Pick profile. Use the previous close request to
                    # hold one observable settling cycle before lifting.
                    target = grasp_target.copy()
                    target[2] = grasp_position[2] + (
                        0.0 if last_public_gripper_action > 0.5 else lift_step_m
                    )
                elif entered_lift:
                    # Preserve the legacy oracle's final close cycle for
                    # profiles without a task-owned post-contact lock.
                    target = grasp_target
                else:
                    target = grasp_position + np.asarray(
                        (0.0, 0.0, lift_step_m)
                    )
            else:
                target = grasp_target

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
            safety_intervention = bool(
                info.get(
                    "safety_intervention",
                    info.get("safety_modified", False),
                )
            )
            safety_intervention_steps += int(safety_intervention)
            phase_interventions[phase] += int(safety_intervention)
            fully_rejected_steps += int(
                float(info.get("safety_projection_alpha", 1.0)) <= 1e-8
            )
            if safety_intervention:
                body1 = info.get("safety_blocking_body1")
                body2 = info.get("safety_blocking_body2")
                pair = " <-> ".join(
                    sorted(
                        (
                            "<unknown>" if body1 is None else str(body1),
                            "<unknown>" if body2 is None else str(body2),
                        )
                    )
                )
                blocking_body_pairs[pair] += 1
                blocking_phase_body_pairs[f"{phase}: {pair}"] += 1
                category = str(
                    info.get("safety_blocking_category") or "unknown"
                )
                blocking_categories[category] += 1
                blocking_phase_categories[f"{phase}: {category}"] += 1
            last_public_gripper_action = float(action[6])
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
            success=(termination_reason == task_config.success_termination_reason),
            steps=steps,
            bilateral_contact_steps=bilateral_contact_steps,
            max_cube_lift_m=max_cube_z - resting_cube_z,
            termination_reason=termination_reason,
            pose_ik_solver=info.get("pose_ik_solver"),
            safety_intervention_steps=safety_intervention_steps,
            fully_rejected_steps=fully_rejected_steps,
            safety_intervention_rate=safety_intervention_steps / max(steps, 1),
            fully_rejected_rate=fully_rejected_steps / max(steps, 1),
            safety_intervention_steps_by_phase=tuple(
                sorted(phase_interventions.items())
            ),
            safety_blocking_category_counts=tuple(
                sorted(blocking_categories.items())
            ),
            safety_blocking_phase_category_counts=tuple(
                sorted(blocking_phase_categories.items())
            ),
            safety_blocking_body_pair_counts=tuple(
                sorted(
                    blocking_body_pairs.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            safety_blocking_phase_body_pair_counts=tuple(
                sorted(
                    blocking_phase_body_pairs.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
        )
    finally:
        env.close()


__all__ = [
    "ScriptedPhysicalGraspResult",
    "run_scripted_physical_grasp_gate",
]

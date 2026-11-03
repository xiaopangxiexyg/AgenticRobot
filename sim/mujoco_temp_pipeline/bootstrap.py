"""Optional actor-only initialization from deployable goal-delta features."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from sim.mujoco.runtime import CubeLiftVectorExecutor
from sim.mujoco.tasks.cube_lift.config import (
    CONTACT_TASK_OBJECTIVE,
    LIFT_TASK_OBJECTIVE,
    REACH_TASK_OBJECTIVE,
    CubeLiftTaskConfig,
    is_cube_lift_success_reason,
)
from sim.mujoco.tasks.cube_lift.learning import (
    build_learning_metadata,
    build_policy_action_transform,
    encode_policy_observation,
    project_flat_observation,
)
from sim.mujoco.tasks.cube_lift.reach_guidance import (
    contact_close_translation_multiplier,
)

from .config import GoalDeltaActorBootstrapConfig
from .normalization import RunningMeanStd
from .policy import ActorCritic


GOAL_DELTA_ACTOR_BOOTSTRAP_ID = "dexteleop.sim.goal_delta_actor_bootstrap.v3"
GOAL_DELTA_ACTOR_FIELDS = ("proprio", "reach_target_delta")
GOAL_DELTA_XYZ_ACTION_COMPONENTS = (
    "right_tcp_delta_x",
    "right_tcp_delta_y",
    "right_tcp_delta_z",
)
GOAL_DELTA_CONTACT_ACTION_COMPONENTS = (
    *GOAL_DELTA_XYZ_ACTION_COMPONENTS,
    "right_gripper_delta",
)


@dataclass(frozen=True)
class GoalDeltaBootstrapDataset:
    actor_observations: np.ndarray
    target_actions: np.ndarray
    collection_metadata: dict[str, Any]


def validate_goal_delta_bootstrap_task(config: CubeLiftTaskConfig) -> None:
    """Fail closed unless the teacher uses the deployed actor contract."""

    if config.resolved_policy_observation_fields != GOAL_DELTA_ACTOR_FIELDS:
        raise ValueError(
            "Goal-delta actor bootstrap requires policy_observation_fields="
            f"{GOAL_DELTA_ACTOR_FIELDS!r}; got "
            f"{config.resolved_policy_observation_fields!r}."
        )
    components = config.resolved_policy_action_components
    valid_reach = bool(
        config.task_objective == REACH_TASK_OBJECTIVE
        and components == GOAL_DELTA_XYZ_ACTION_COMPONENTS
    )
    valid_contact = bool(
        config.task_objective == CONTACT_TASK_OBJECTIVE
        and components == GOAL_DELTA_CONTACT_ACTION_COMPONENTS
        and config.contact_close_translation_scaling_enabled
    )
    valid_pick = bool(
        config.task_objective == LIFT_TASK_OBJECTIVE
        and components == GOAL_DELTA_CONTACT_ACTION_COMPONENTS
        and config.post_contact_phase_enabled
        and not build_policy_action_transform(config).observation_conditioning_enabled
        and config.resolved_policy_action_scale == (1.0,) * config.policy_action_dim
        and config.resolved_policy_action_bias == (0.0,) * config.policy_action_dim
    )
    if not (valid_reach or valid_contact or valid_pick):
        raise ValueError(
            "Goal-delta actor bootstrap requires Reach xyz, Contact "
            "goal-delta xyz-plus-gripper precision, or the stationary direct "
            "Pick xyz-plus-gripper profile; "
            f"got objective={config.task_objective!r}, components={components!r}."
        )
    metadata = build_learning_metadata(config)
    if bool(metadata["actor_uses_privileged"]):
        raise ValueError(
            "Goal-delta actor bootstrap cannot initialize a privileged actor."
        )


def goal_delta_teacher_action(
    config: CubeLiftTaskConfig,
    task_observation: np.ndarray,
    *,
    lift_step_m: float = 0.004,
    pick_close_distance_m: float = 0.035,
    pick_lift_active: Any | None = None,
    pick_lift_lock_xy: bool = False,
) -> np.ndarray:
    """Map actor-visible target/closure state to the task's latent action.

    The target is first expressed in the public normalized pose action, then
    inverted through the task-owned affine and optional distance-dependent
    decoder. This keeps actor supervision correct when a task deliberately
    slows its public TCP increment near the target. Pick lift labels require an
    explicit optional teacher phase; this simulator signal is never appended
    to the fitted actor observation.
    """

    validate_goal_delta_bootstrap_task(config)
    projected = project_flat_observation(config, task_observation)
    goal_delta = np.asarray(
        projected["reach_target_delta"],
        dtype=np.float32,
    )
    tcp_scale = np.asarray(config.right_tcp_delta_m, dtype=np.float32)
    task_action = np.clip(goal_delta / tcp_scale, -1.0, 1.0)
    policy_scale = np.asarray(
        config.resolved_policy_action_scale,
        dtype=np.float32,
    )
    policy_bias = np.asarray(
        config.resolved_policy_action_bias,
        dtype=np.float32,
    )
    transform = build_policy_action_transform(config)
    decoder_scale = np.asarray(
        transform.translation_action_scale(task_observation),
        dtype=np.float32,
    )
    desired_task_action = np.zeros(
        (*goal_delta.shape[:-1], config.policy_action_dim),
        dtype=np.float32,
    )
    desired_task_action[..., :3] = task_action
    if config.task_objective == CONTACT_TASK_OBJECTIVE:
        proprio = np.asarray(projected["proprio"], dtype=np.float32)
        close_xy_multiplier = contact_close_translation_multiplier(
            goal_delta,
            proprio[..., 6],
            translation_scale=config.contact_close_translation_scale,
            close_distance_m=config.contact_close_distance_m,
            keep_measured_close=(
                config.contact_close_keep_measured_close
            ),
        )
        if config.contact_close_anisotropic_translation_scaling_enabled:
            close_z_multiplier = contact_close_translation_multiplier(
                goal_delta,
                proprio[..., 6],
                translation_scale=(
                    config.resolved_contact_close_translation_scales[2]
                ),
                close_distance_m=config.contact_close_distance_m,
                keep_measured_close=(
                    config.contact_close_keep_measured_close
                ),
            )
        else:
            close_z_multiplier = close_xy_multiplier
        close_translation_multiplier = np.stack(
            (
                close_xy_multiplier,
                close_xy_multiplier,
                close_z_multiplier,
            ),
            axis=-1,
        )
        desired_task_action[..., :3] *= close_translation_multiplier
        desired_task_action[..., 3] = (
            close_xy_multiplier < 1.0 - 1e-6
        ).astype(np.float32)
    elif config.task_objective == LIFT_TASK_OBJECTIVE:
        lift_step = float(lift_step_m)
        close_distance = float(pick_close_distance_m)
        if (
            not np.isfinite(lift_step)
            or lift_step <= 0.0
            or not np.isfinite(close_distance)
            or close_distance <= 0.0
        ):
            raise ValueError(
                "Pick bootstrap lift step and close distance must be positive "
                "and finite."
            )
        if pick_lift_active is None:
            raise ValueError(
                "Pick bootstrap requires an explicit optional teacher lift phase."
            )
        lift_active = np.asarray(pick_lift_active, dtype=bool)
        try:
            lift_active = np.broadcast_to(lift_active, goal_delta.shape[:-1])
        except ValueError as exc:
            raise ValueError(
                "Pick teacher lift phase must broadcast to the observation batch."
            ) from exc
        proprio = np.asarray(projected["proprio"], dtype=np.float32)
        close_requested = np.logical_or(
            np.linalg.norm(goal_delta, axis=-1) <= close_distance,
            np.logical_or(
                proprio[..., 6] >= config.grasp_latch_release_close,
                proprio[..., 27] > 0.5,
            ),
        )
        # The first post-contact observation still carries the close request
        # as its last public gripper action.  Hold position for that one actor-
        # visible cycle before lifting so the locked gripper target can settle;
        # this avoids turning simulator-only phase into a hidden timing state.
        lift_motion_active = np.logical_and(
            lift_active,
            proprio[..., 27] <= 0.5,
        )
        if pick_lift_lock_xy:
            # Label zero requested base-frame XY even in the first settling
            # cycle. The fitted actor still receives no simulator phase field.
            desired_task_action[..., :2] = np.where(
                lift_active[..., None], np.float32(0.0), desired_task_action[..., :2]
            )
        desired_task_action[..., 2] = np.where(
            lift_motion_active,
            np.clip(lift_step / tcp_scale[2], 0.0, 1.0),
            np.where(lift_active, np.float32(0.0), desired_task_action[..., 2]),
        )
        desired_task_action[..., 3] = np.where(
            lift_active,
            np.float32(0.0),
            close_requested.astype(np.float32),
        )
    component_decoder_scale = np.ones_like(desired_task_action)
    component_decoder_scale[..., :3] = decoder_scale
    action = np.clip(
        (
            desired_task_action / component_decoder_scale
            - policy_bias
        )
        / policy_scale,
        -1.0,
        1.0,
    )
    return np.ascontiguousarray(action, dtype=np.float32)


def collect_goal_delta_bootstrap_dataset(
    config: GoalDeltaActorBootstrapConfig,
    *,
    task_config_path: str,
    num_envs: int,
    seed: int,
    start_method: str,
    model_cache_dir: str | None = None,
) -> GoalDeltaBootstrapDataset:
    """Collect a bounded actor-visible behavior-cloning initialization set."""

    if not config.enabled:
        raise ValueError("Goal-delta bootstrap collection is disabled.")
    if int(num_envs) <= 0:
        raise ValueError("Goal-delta bootstrap num_envs must be positive.")
    task_config = CubeLiftTaskConfig.load(task_config_path)
    validate_goal_delta_bootstrap_task(task_config)
    transform = build_policy_action_transform(task_config)
    executor = CubeLiftVectorExecutor(
        int(num_envs),
        config_path=task_config_path,
        model_cache_dir=model_cache_dir,
        auto_reset=True,
        master_seed=int(seed),
        start_method=start_method,
    )
    observations: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    teacher_episodes = 0
    teacher_successes = 0
    safety_projection_count = 0
    fully_rejected_count = 0
    hard_penetration_count = 0
    blocking_body_pairs: Counter[str] = Counter()
    collected = 0
    started = time.monotonic()
    try:
        task_observation, current_infos = executor.reset()
        while collected < int(config.transitions):
            actor_observation = encode_policy_observation(
                task_config,
                task_observation,
            )
            target_action = goal_delta_teacher_action(
                task_config,
                task_observation,
                lift_step_m=config.lift_step_m,
                pick_close_distance_m=config.pick_close_distance_m,
                pick_lift_lock_xy=config.pick_lift_lock_xy,
                pick_lift_active=(
                    np.asarray(
                        [
                            bool(info.get("post_contact_phase_active", False))
                            for info in current_infos
                        ],
                        dtype=bool,
                    )
                    if task_config.task_objective == LIFT_TASK_OBJECTIVE
                    else None
                ),
            )
            environment_action = transform.transform(
                target_action,
                observation=task_observation,
            )
            (
                next_task_observation,
                _reward,
                terminated,
                truncated,
                infos,
            ) = executor.step(environment_action)
            done = np.logical_or(terminated, truncated)
            transition_infos = tuple(
                info.get("final_info", info) if done[env_id] else info
                for env_id, info in enumerate(infos)
            )
            take = min(int(num_envs), int(config.transitions) - collected)
            observations.append(actor_observation[:take].copy())
            targets.append(target_action[:take].copy())
            for env_id in range(take):
                info = transition_infos[env_id]
                alpha = float(info.get("safety_projection_alpha", 1.0))
                safety_projection_count += int(
                    bool(info.get("safety_modified", False))
                    or alpha < 1.0 - 1e-6
                )
                if bool(
                    info.get(
                        "safety_intervention",
                        info.get("safety_modified", False),
                    )
                ):
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
                fully_rejected_count += int(alpha <= 1e-8)
                hard_penetration_count += int(
                    info.get("termination_reason")
                    == "robot_environment_penetration"
                )
                if done[env_id]:
                    teacher_episodes += 1
                    teacher_successes += int(
                        is_cube_lift_success_reason(
                            info.get("termination_reason")
                        )
                    )
            collected += take
            task_observation = next_task_observation
            current_infos = [
                info.get("reset_info", info) if done[env_id] else info
                for env_id, info in enumerate(infos)
            ]
    finally:
        executor.close()
    elapsed = time.monotonic() - started
    actor_observations = np.concatenate(observations, axis=0)
    target_actions = np.concatenate(targets, axis=0)
    if actor_observations.shape[0] != int(config.transitions):
        raise RuntimeError("Goal-delta bootstrap collection changed size.")
    return GoalDeltaBootstrapDataset(
        actor_observations=np.ascontiguousarray(
            actor_observations,
            dtype=np.float32,
        ),
        target_actions=np.ascontiguousarray(target_actions, dtype=np.float32),
        collection_metadata={
            "bootstrap_id": GOAL_DELTA_ACTOR_BOOTSTRAP_ID,
            "pick_lift_xy_mode": (
                "zero_requested_base_xy" if config.pick_lift_lock_xy
                else "target_centering"
            ) if task_config.task_objective == LIFT_TASK_OBJECTIVE else "not_applicable",
            "teacher_source": (
                "optional_sim_contact_phase_teacher_plus_actor_observation"
                ".proprio_and_reach_target_delta"
                if task_config.task_objective == LIFT_TASK_OBJECTIVE
                else (
                    "actor_observation.proprio_measured_close_and_reach_target_delta"
                    if task_config.task_objective == CONTACT_TASK_OBJECTIVE
                    else "actor_observation.reach_target_delta"
                )
            ),
            "target_action_source": "requested_latent_action",
            "transitions": int(config.transitions),
            "teacher_episodes": teacher_episodes,
            "teacher_successes": teacher_successes,
            "teacher_success_rate": (
                teacher_successes / teacher_episodes
                if teacher_episodes
                else 0.0
            ),
            "teacher_safety_projection_rate": (
                safety_projection_count / int(config.transitions)
            ),
            "teacher_fully_rejected_action_rate": (
                fully_rejected_count / int(config.transitions)
            ),
            "teacher_hard_penetration_count": hard_penetration_count,
            "teacher_safety_blocking_body_pair_counts": dict(
                sorted(
                    blocking_body_pairs.items(),
                    key=lambda item: (-item[1], item[0]),
                )
            ),
            "collection_seconds": elapsed,
            "actor_uses_privileged": False,
            "teacher_uses_privileged": (
                task_config.task_objective == LIFT_TASK_OBJECTIVE
            ),
            "privileged_teacher_optional": True,
            "policy_requires_privileged": False,
        },
    )


def fit_goal_delta_actor_bootstrap(
    policy: ActorCritic,
    dataset: GoalDeltaBootstrapDataset,
    config: GoalDeltaActorBootstrapConfig,
    *,
    device: torch.device,
) -> tuple[RunningMeanStd, dict[str, Any]]:
    """Fit only actor parameters and return its deployment normalizer."""

    observations = np.asarray(dataset.actor_observations, dtype=np.float32)
    targets = np.asarray(dataset.target_actions, dtype=np.float32)
    expected_observations = (int(config.transitions), policy.observation_dim)
    expected_targets = (int(config.transitions), policy.action_dim)
    if observations.shape != expected_observations:
        raise ValueError(
            "Goal-delta bootstrap observation shape mismatch: "
            f"expected={expected_observations}, got={observations.shape}."
        )
    if targets.shape != expected_targets:
        raise ValueError(
            "Goal-delta bootstrap target shape mismatch: "
            f"expected={expected_targets}, got={targets.shape}."
        )
    if not np.all(np.isfinite(observations)) or not np.all(
        np.isfinite(targets)
    ):
        raise ValueError("Goal-delta bootstrap dataset must be finite.")
    if np.any(targets < -1.0) or np.any(targets > 1.0):
        raise ValueError("Goal-delta bootstrap targets must lie in [-1, 1].")

    normalizer = RunningMeanStd.create((policy.observation_dim,))
    normalizer.update(observations)
    normalized = torch.as_tensor(
        normalizer.normalize(observations),
        dtype=torch.float32,
        device=device,
    )
    target_tensor = torch.as_tensor(
        targets,
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam(
        policy.actor_parameters(),
        lr=float(config.learning_rate),
    )

    def loss_value() -> torch.Tensor:
        prediction, _ = policy.predict_action(normalized, deterministic=True)
        return torch.mean(torch.square(prediction - target_tensor))

    policy.train()
    with torch.no_grad():
        initial_mse = float(loss_value().cpu())
    final_mse = initial_mse
    epochs_completed = 0
    started = time.monotonic()
    for epoch in range(int(config.epochs)):
        permutation = torch.randperm(normalized.shape[0], device=device)
        for start in range(0, normalized.shape[0], int(config.minibatch_size)):
            indices = permutation[start : start + int(config.minibatch_size)]
            prediction, _ = policy.predict_action(
                normalized[indices],
                deterministic=True,
            )
            loss = torch.mean(
                torch.square(prediction - target_tensor[indices])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                policy.actor_parameters(),
                float(config.max_grad_norm),
            )
            optimizer.step()
        epochs_completed = epoch + 1
        with torch.no_grad():
            final_mse = float(loss_value().cpu())
        if final_mse <= float(config.target_mse):
            break
    policy.reset_critic()
    return normalizer, {
        **dataset.collection_metadata,
        "config": config.to_dict(),
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "epochs_completed": epochs_completed,
        "fit_seconds": time.monotonic() - started,
        "critic_initialized_from_teacher": False,
    }


__all__ = [
    "GOAL_DELTA_ACTOR_BOOTSTRAP_ID",
    "GoalDeltaBootstrapDataset",
    "collect_goal_delta_bootstrap_dataset",
    "fit_goal_delta_actor_bootstrap",
    "goal_delta_teacher_action",
    "validate_goal_delta_bootstrap_task",
]

"""A deliberately small PPO implementation wired to CubeLiftVectorExecutor."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from sim.mujoco.collision_safety import FORBIDDEN_COLLISION_CATEGORIES
from sim.mujoco.runtime import CubeLiftVectorExecutor
from sim.mujoco.tasks.cube_lift.config import (
    CubeLiftTaskConfig,
    is_cube_lift_success_reason,
)
from sim.mujoco.tasks.cube_lift.learning import (
    build_actor_observation_contract,
    build_model_input_binding,
    build_observation_contract,
    build_policy_action_binding,
    build_policy_action_transform,
    build_task_action_binding,
    encode_policy_observation,
)
from sim.mujoco.tasks.cube_lift.reward_adapter import CubeLiftRewardState

from .action_diagnostics import ActionDiagnostics
from .config import PPOConfig, RewardShapingConfig
from .normalization import RunningMeanStd
from .policy import ActorCritic, resolve_device
from .reward import shape_rewards


UNSAFE_TERMINATION_REASONS = frozenset(
    {
        "sim_joint_position_limit_violation",
        "sim_collision_margin_violation",
        "robot_environment_penetration",
        "robot_environment_clearance_violation",
    }
)


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    critic_observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_values: torch.Tensor
    raw_reward_mean: float
    shaped_reward_mean: float
    completed_episodes: int
    successful_episodes: int
    episode_raw_returns: tuple[float, ...]
    episodes: tuple[dict[str, Any], ...]
    action_diagnostics: dict[str, float | None]
    safety_projection_rate: float
    safety_rejection_rate: float
    safety_recovery_rate: float
    safety_projection_alpha_mean: float
    safety_collision_category_rates: dict[str, float]
    effective_action_override_rate: float
    effective_action_delta_mean: float
    post_contact_phase_active_rate: float
    post_contact_action_modified_rate: float
    scripted_pregrasp_action_modified_rate: float
    scripted_contact_action_modified_rate: float
    safety_minimum_distance_m: float | None
    hard_penetration_count: int
    clearance_violation_count: int
    arm_tracking_error_rad_mean: float
    arm_tracking_error_rad_max: float
    gripper_tracking_error_mean: float
    gripper_tracking_error_max: float
    wrist_rgb_mean: float | None
    wrist_rgb_std: float | None
    wrist_rgb_health_rate: float | None
    reward_term_means: dict[str, float]


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    bootstrap_values: np.ndarray,
    terminated: np.ndarray,
    episode_done: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GAE while bootstrapping time-limit truncations only."""

    shape = rewards.shape
    if rewards.ndim != 2:
        raise ValueError(f"rewards must be [steps, envs], got shape {shape}.")
    for name, array in {
        "values": values,
        "bootstrap_values": bootstrap_values,
        "terminated": terminated,
        "episode_done": episode_done,
    }.items():
        if np.asarray(array).shape != shape:
            raise ValueError(f"{name} must match reward shape {shape}.")
    advantages = np.zeros(shape, dtype=np.float32)
    last_gae = np.zeros(shape[1], dtype=np.float32)
    for step in reversed(range(shape[0])):
        bootstrap_mask = 1.0 - terminated[step]
        continuation_mask = 1.0 - episode_done[step]
        delta = (
            rewards[step]
            + gamma * bootstrap_values[step] * bootstrap_mask
            - values[step]
        )
        last_gae = (
            delta
            + gamma * gae_lambda * continuation_mask * last_gae
        )
        advantages[step] = last_gae
    return advantages, advantages + values


def compute_clipped_value_loss(
    new_values: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    *,
    clip_coef: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return PPO's clipped critic loss and the clipped-sample fraction.

    When enabled, the rollout-time value prediction is the critic trust-region
    anchor. This optional constraint is independent of policy clipping; a
    shared backbone otherwise lets value gradients move actor features behind
    PPO's ratio guard. ``clip_coef=0`` selects the established squared-error
    baseline.
    """

    if (
        new_values.shape != old_values.shape
        or new_values.shape != returns.shape
    ):
        raise ValueError(
            "new_values, old_values, and returns must have identical shapes."
        )
    coefficient = float(clip_coef)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("clip_coef must be finite and non-negative.")
    loss_unclipped = torch.square(new_values - returns)
    if coefficient == 0.0:
        return 0.5 * loss_unclipped.mean(), torch.zeros(
            (), dtype=new_values.dtype, device=new_values.device
        )
    value_delta = new_values - old_values
    clipped_values = old_values + value_delta.clamp(
        -coefficient,
        coefficient,
    )
    loss_clipped = torch.square(clipped_values - returns)
    loss = 0.5 * torch.maximum(loss_unclipped, loss_clipped).mean()
    clip_fraction = (value_delta.abs() > coefficient).float().mean()
    return loss, clip_fraction


def clip_actor_critic_grad_norms(
    policy: ActorCritic,
    *,
    max_grad_norm: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Clip independent actor/critic gradients without cross-suppression.

    In the shared-backbone topology a parameter can carry both losses, so a
    combined clip is the only coherent behavior. Once the critic is
    structurally separate, clipping the combined parameter set would still let
    a large value gradient shrink an otherwise valid policy update.
    """

    maximum = float(max_grad_norm)
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("max_grad_norm must be finite and positive.")
    if not policy.separate_critic_backbone:
        combined = nn.utils.clip_grad_norm_(policy.parameters(), maximum)
        return combined, combined
    actor_norm = nn.utils.clip_grad_norm_(policy.actor_parameters(), maximum)
    critic_norm = nn.utils.clip_grad_norm_(policy.critic_parameters(), maximum)
    return actor_norm, critic_norm


def compute_actor_reference_mse(
    policy: ActorCritic,
    reference_policy: ActorCritic,
    observations: torch.Tensor,
) -> torch.Tensor:
    """Measure deterministic latent-action drift from a frozen actor.

    Both actors receive the same already-normalized deployable observation.
    The reference path is detached, so this regularizes only the live actor
    and never introduces a simulator-only observation or critic dependency.
    """

    with torch.no_grad():
        reference_action, _ = reference_policy.predict_action(
            observations,
            deterministic=True,
        )
    current_action, _ = policy.predict_action(
        observations,
        deterministic=True,
    )
    return torch.square(current_action - reference_action).mean()


class PPOSmokeTrainer:
    def __init__(
        self,
        ppo_config: PPOConfig,
        reward_config: RewardShapingConfig,
        *,
        task_config_path: str,
        model_cache_dir: str | None = None,
        require_wrist_rgb_health: bool = False,
    ) -> None:
        self.config = ppo_config
        self.reward_config = reward_config
        self.task_config_path = str(task_config_path)
        self.task_config = CubeLiftTaskConfig.load(task_config_path)
        self.task_observation_dim = self.task_config.observation_dim
        self.learning_task_observation_contract = build_observation_contract(
            self.task_config
        )
        self.learning_observation_contract = build_actor_observation_contract(
            self.task_config,
            self.learning_task_observation_contract,
        )
        self.learning_model_input_binding = build_model_input_binding(
            self.task_config,
            self.learning_task_observation_contract,
        )
        self.observation_dim = (
            self.learning_model_input_binding.expected_flat_dim(
                self.learning_observation_contract
            )
        )
        self.actor_uses_wrist_rgb = (
            "wrist_rgb" in self.learning_model_input_binding.input_fields
        )
        actor_image_shape = (
            self.task_config.wrist_rgb_shape
            if self.actor_uses_wrist_rgb
            else None
        )
        actor_image_dim = (
            0
            if actor_image_shape is None
            else int(np.prod(actor_image_shape, dtype=np.int64))
        )
        self.actor_proprio_dim = self.observation_dim - actor_image_dim
        self.critic_observation_mode = ppo_config.critic_observation_mode
        self.critic_observation_dim = (
            31
            if self.critic_observation_mode == "privileged_training_state"
            else self.observation_dim
        )
        self.environment_action_dim = self.task_config.action_dim
        self.active_action_indices = (
            self.task_config.policy_active_action_indices
        )
        self.action_dim = self.task_config.policy_action_dim
        self.device = resolve_device(ppo_config.device)
        np.random.seed(ppo_config.seed)
        torch.manual_seed(ppo_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ppo_config.seed)

        self.policy = ActorCritic(
            self.observation_dim,
            self.action_dim,
            ppo_config.hidden_dim,
            proprio_dim=self.actor_proprio_dim,
            image_shape=actor_image_shape,
            image_pool_size=self.task_config.wrist_rgb_spatial_pool_size,
            critic_observation_dim=self.critic_observation_dim,
            separate_critic_backbone=(
                ppo_config.critic_backbone_mode == "separate"
            ),
            environment_action_dim=self.environment_action_dim,
            active_action_indices=self.active_action_indices,
            trainable_log_std=False,
            initial_log_std=ppo_config.initial_log_std,
        ).to(self.device)
        self.critic_backbone_mode = (
            "separate"
            if self.policy.separate_critic_backbone
            else "shared_actor"
        )
        self.optimizer = torch.optim.Adam(
            (
                parameter
                for parameter in self.policy.parameters()
                if parameter.requires_grad
            ),
            lr=ppo_config.learning_rate,
            eps=1e-5,
        )
        self._actor_reference_policy: ActorCritic | None = None
        self._actor_reference_source: str | None = None
        self._exploration_initial_log_std = np.full(
            self.action_dim,
            float(ppo_config.initial_log_std),
            dtype=np.float32,
        )
        self._exploration_final_log_std = np.full(
            self.action_dim,
            float(ppo_config.final_log_std),
            dtype=np.float32,
        )
        self._exploration_schedule_source = "ppo_config"
        self.observation_normalizer = RunningMeanStd.create((self.observation_dim,))
        self.critic_observation_normalizer = (
            RunningMeanStd.create((self.critic_observation_dim,))
            if self.critic_observation_mode == "privileged_training_state"
            else None
        )
        self.env = CubeLiftVectorExecutor(
            ppo_config.num_envs,
            config_path=task_config_path,
            model_cache_dir=model_cache_dir,
            auto_reset=True,
            master_seed=ppo_config.seed,
            start_method=ppo_config.start_method,
        )
        task_observation, reset_infos = self.env.reset()
        self.task_observation = task_observation
        self.observation = encode_policy_observation(
            self.task_config,
            task_observation,
        )
        self.learning_action_binding = build_policy_action_binding(
            self.task_config
        )
        self.task_action_binding = build_task_action_binding(self.task_config)
        self.policy_action_transform = build_policy_action_transform(
            self.task_config
        )
        if self.learning_action_binding.output_dim != self.action_dim:
            self.env.close()
            raise RuntimeError(
                "CubeLift policy action binding dimension does not match "
                f"the actor: binding={self.learning_action_binding.output_dim}, "
                f"actor={self.action_dim}."
            )
        if self.task_action_binding.output_dim != self.environment_action_dim:
            self.env.close()
            raise RuntimeError(
                "CubeLift task action binding dimension does not match "
                f"the environment: binding={self.task_action_binding.output_dim}, "
                f"environment={self.environment_action_dim}."
            )
        if self.observation.shape != (
            ppo_config.num_envs,
            self.observation_dim,
        ):
            self.env.close()
            raise RuntimeError(
                "CubeLift actor projection changed shape unexpectedly: "
                f"expected={(ppo_config.num_envs, self.observation_dim)}, "
                f"got={self.observation.shape}."
            )
        if require_wrist_rgb_health and self.actor_uses_wrist_rgb:
            unhealthy = [
                {
                    "env_id": index,
                    "mean": info.get("wrist_rgb_mean"),
                    "std": info.get("wrist_rgb_std"),
                    "nonzero_fraction": info.get("wrist_rgb_nonzero_fraction"),
                }
                for index, info in enumerate(reset_infos)
                if not bool(info.get("wrist_rgb_health_ok", False))
            ]
            if unhealthy:
                self.env.close()
                raise RuntimeError(
                    "Wrist RGB health check failed at reset; inspect camera pose/"
                    f"lighting before training: {unhealthy}"
                )
        self._reward_observation = np.stack(
            [self._training_reward_state(info) for info in reset_infos]
        )
        self.pose_ik_solvers = tuple(
            sorted(
                {
                    str(info["pose_ik_solver"])
                    for info in reset_infos
                    if info.get("pose_ik_solver") is not None
                }
            )
        )
        self.observation_normalizer.update(self.observation)
        if self.critic_observation_normalizer is not None:
            self.critic_observation_normalizer.update(self._reward_observation)
        self.global_step = 0
        self.update_index = 0
        self._episode_raw_return = np.zeros(ppo_config.num_envs, dtype=np.float64)
        self._episode_shaped_return = np.zeros(ppo_config.num_envs, dtype=np.float64)
        self._episode_length = np.zeros(ppo_config.num_envs, dtype=np.int32)
        self._episode_resting_cube_z = CubeLiftRewardState.from_flat(
            self._reward_observation
        ).cube_z.astype(np.float32, copy=True)
        self._previous_bilateral_contact = np.asarray(
            [
                bool(info.get("bilateral_finger_cube_contact", False))
                for info in reset_infos
            ],
            dtype=np.float32,
        )
        self._previous_inner_contact_score = np.asarray(
            [float(info.get("gripper_inner_contact_score", 0.0)) for info in reset_infos],
            dtype=np.float32,
        )
        self._previous_bilateral_inner_contact_score = np.asarray(
            [
                float(info.get("gripper_bilateral_inner_contact_score", 0.0))
                for info in reset_infos
            ],
            dtype=np.float32,
        )
        self._apply_exploration_schedule()

    def capture_actor_reference(self, *, source: str) -> dict[str, object]:
        """Freeze the current deployable actor as an optional PPO anchor."""

        normalized_source = str(source).strip()
        if not normalized_source:
            raise ValueError("Actor reference source must be non-empty.")
        reference = copy.deepcopy(self.policy).to(self.device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        self._actor_reference_policy = reference
        self._actor_reference_source = normalized_source
        return {
            "enabled": True,
            "source": normalized_source,
            "mse_coefficient": float(
                self.config.actor_reference_mse_coef
            ),
            "uses_privileged_observation": False,
        }

    @staticmethod
    def _training_reward_state(info: dict[str, Any]) -> np.ndarray:
        state = np.asarray(info["training_reward_state"], dtype=np.float32).reshape(-1)
        if state.shape != (31,) or not np.all(np.isfinite(state)):
            raise RuntimeError("CubeLift training_reward_state must be finite and 31-D.")
        return state

    def close(self) -> None:
        self.env.close()

    def configure_exploration_schedule(
        self,
        initial_log_std: Any,
        final_log_std: Any,
        *,
        source: str,
    ) -> dict[str, object]:
        """Install one checkpoint-aware schedule in latent component order."""

        initial = np.asarray(initial_log_std, dtype=np.float32).reshape(-1)
        final = np.asarray(final_log_std, dtype=np.float32).reshape(-1)
        expected = (self.action_dim,)
        if initial.shape != expected or final.shape != expected:
            raise ValueError(
                "Exploration schedules must match the latent action dimension: "
                f"initial={initial.shape}, final={final.shape}, expected={expected}."
            )
        if not np.all(np.isfinite(initial)) or not np.all(np.isfinite(final)):
            raise ValueError("Exploration schedules must be finite.")
        if np.any(initial < -5.0) or np.any(initial > 1.0):
            raise ValueError("Initial log std must lie within [-5, 1].")
        if np.any(final < -5.0) or np.any(final > initial):
            raise ValueError(
                "Final component log std must lie within [-5, initial]."
            )
        normalized_source = str(source).strip()
        if not normalized_source:
            raise ValueError("Exploration schedule source must be non-empty.")
        self._exploration_initial_log_std = initial.copy()
        self._exploration_final_log_std = final.copy()
        self._exploration_schedule_source = normalized_source
        self._apply_exploration_schedule()
        return self.exploration_schedule_metadata

    @property
    def exploration_schedule_metadata(self) -> dict[str, object]:
        components = tuple(
            self.task_config.resolved_policy_action_components
        )
        current = self.policy.log_std.detach().cpu().numpy()
        return {
            "source": self._exploration_schedule_source,
            "components": list(components),
            "initial_log_std": self._exploration_initial_log_std.astype(
                float
            ).tolist(),
            "final_log_std": self._exploration_final_log_std.astype(
                float
            ).tolist(),
            "current_log_std": current.astype(float).tolist(),
        }

    def _apply_exploration_schedule(self) -> float:
        anneal_steps = max(
            float(self.config.total_timesteps)
            * float(self.config.std_anneal_fraction),
            1.0,
        )
        progress = min(max(float(self.global_step) / anneal_steps, 0.0), 1.0)
        log_std = (
            self._exploration_initial_log_std
            + progress
            * (
                self._exploration_final_log_std
                - self._exploration_initial_log_std
            )
        )
        with torch.no_grad():
            self.policy.log_std.copy_(
                torch.as_tensor(
                    log_std,
                    dtype=self.policy.log_std.dtype,
                    device=self.policy.log_std.device,
                )
            )
        return float(np.mean(log_std))

    def collect_rollout(self) -> RolloutBatch:
        cfg = self.config
        self._apply_exploration_schedule()
        shape = (cfg.rollout_steps, cfg.num_envs)
        observations = np.empty((*shape, self.observation_dim), dtype=np.float32)
        critic_observations = np.empty(
            (*shape, self.critic_observation_dim),
            dtype=np.float32,
        )
        actions = np.empty((*shape, self.action_dim), dtype=np.float32)
        log_probs = np.empty(shape, dtype=np.float32)
        values = np.empty(shape, dtype=np.float32)
        rewards = np.empty(shape, dtype=np.float32)
        raw_rewards_buffer = np.empty(shape, dtype=np.float32)
        terminated_buffer = np.empty(shape, dtype=np.float32)
        episode_done_buffer = np.empty(shape, dtype=np.float32)
        bootstrap_values = np.empty(shape, dtype=np.float32)
        episode_returns: list[float] = []
        episode_records: list[dict[str, Any]] = []
        completed = 0
        successful = 0
        safety_modified_count = 0
        safety_rejected_count = 0
        safety_recovery_count = 0
        safety_alpha_sum = 0.0
        safety_collision_category_counts = {
            category: 0 for category in FORBIDDEN_COLLISION_CATEGORIES
        }
        safety_sample_count = 0
        effective_action_override_count = 0
        effective_action_delta_sum = 0.0
        post_contact_phase_active_count = 0
        post_contact_action_modified_count = 0
        scripted_pregrasp_action_modified_count = 0
        scripted_contact_action_modified_count = 0
        safety_minimum_distance = float("inf")
        hard_penetration_count = 0
        clearance_violation_count = 0
        arm_tracking_error_sum = 0.0
        arm_tracking_error_max = 0.0
        gripper_tracking_error_sum = 0.0
        gripper_tracking_error_max = 0.0
        wrist_rgb_mean_sum = 0.0
        wrist_rgb_std_sum = 0.0
        wrist_rgb_health_count = 0
        wrist_rgb_sample_count = 0
        reward_term_sums: dict[str, float] = {}
        reward_term_samples = 0
        action_diagnostics = ActionDiagnostics()

        self.policy.eval()
        for step in range(cfg.rollout_steps):
            normalized = self.observation_normalizer.normalize(self.observation)
            normalized_critic = (
                normalized
                if self.critic_observation_normalizer is None
                else self.critic_observation_normalizer.normalize(
                    self._reward_observation
                )
            )
            observation_tensor = torch.as_tensor(normalized, device=self.device)
            critic_observation_tensor = (
                None
                if self.critic_observation_normalizer is None
                else torch.as_tensor(normalized_critic, device=self.device)
            )
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = self.policy.act(
                    observation_tensor,
                    critic_observation=critic_observation_tensor,
                )
            action = action_tensor.cpu().numpy().astype(np.float32, copy=False)
            action_scale = self.policy_action_transform.action_scale(
                self.task_observation
            )
            environment_action = self.policy_action_transform.transform(
                action,
                observation=self.task_observation,
            )
            action_diagnostics.update(
                action,
                environment_action,
                action_scale,
            )
            (
                next_task_observation,
                raw_reward,
                terminated,
                truncated,
                infos,
            ) = self.env.step(environment_action)
            next_observation = encode_policy_observation(
                self.task_config,
                next_task_observation,
            )
            done = np.logical_or(terminated, truncated)
            transition_infos = tuple(
                info.get("final_info", info) if done[env_id] else info
                for env_id, info in enumerate(infos)
            )
            safety_alpha = np.asarray(
                [
                    float(info.get("safety_projection_alpha", 1.0))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            safety_modified = np.asarray(
                [bool(info.get("safety_modified", False)) for info in transition_infos],
                dtype=bool,
            )
            safety_intervention = np.asarray(
                [
                    bool(
                        info.get(
                            "safety_intervention",
                            info.get("safety_modified", False),
                        )
                    )
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            safety_collision_category = np.asarray(
                [
                    info.get("safety_blocking_category")
                    for info in transition_infos
                ],
                dtype=object,
            )
            for category in FORBIDDEN_COLLISION_CATEGORIES:
                safety_collision_category_counts[category] += int(
                    np.count_nonzero(
                        (safety_collision_category == category)
                        & safety_intervention.astype(bool)
                    )
                )
            post_contact_phase_active = np.asarray(
                [
                    bool(info.get("post_contact_phase_active", False))
                    for info in transition_infos
                ],
                dtype=bool,
            )
            post_contact_action_modified = np.asarray(
                [
                    bool(info.get("post_contact_phase_action_modified", False))
                    for info in transition_infos
                ],
                dtype=bool,
            )
            scripted_pregrasp_action_modified = np.asarray(
                [
                    bool(info.get("scripted_pregrasp_action_modified", False))
                    for info in transition_infos
                ],
                dtype=bool,
            )
            scripted_contact_action_modified = np.asarray(
                [
                    bool(info.get("scripted_contact_action_modified", False))
                    for info in transition_infos
                ],
                dtype=bool,
            )
            effective_action_override = (
                safety_intervention.astype(bool)
                | (safety_alpha < (1.0 - 1e-6))
                | post_contact_action_modified
                | scripted_pregrasp_action_modified
                | scripted_contact_action_modified
            )
            effective_action_override_count += int(
                np.count_nonzero(effective_action_override)
            )
            post_contact_phase_active_count += int(
                np.count_nonzero(post_contact_phase_active)
            )
            post_contact_action_modified_count += int(
                np.count_nonzero(post_contact_action_modified)
            )
            scripted_pregrasp_action_modified_count += int(
                np.count_nonzero(scripted_pregrasp_action_modified)
            )
            scripted_contact_action_modified_count += int(
                np.count_nonzero(scripted_contact_action_modified)
            )
            safety_modified_count += sum(
                bool(info.get("safety_modified", False))
                for info in transition_infos
            )
            safety_rejected_count += int(np.count_nonzero(safety_alpha <= 1e-8))
            safety_recovery_count += sum(
                bool(info.get("safety_recovery", False))
                for info in transition_infos
            )
            safety_alpha_sum += float(np.sum(safety_alpha))
            safety_sample_count += cfg.num_envs
            for info in transition_infos:
                tracking_mean = float(info.get("arm_tracking_error_rad_mean", 0.0))
                tracking_max = float(info.get("arm_tracking_error_rad_max", 0.0))
                arm_tracking_error_sum += tracking_mean
                arm_tracking_error_max = max(arm_tracking_error_max, tracking_max)
                gripper_error = abs(
                    float(info.get("right_gripper_tracking_error", 0.0))
                )
                gripper_tracking_error_sum += gripper_error
                gripper_tracking_error_max = max(
                    gripper_tracking_error_max, gripper_error
                )
                if self.task_config.wrist_rgb_enabled:
                    wrist_rgb_mean_sum += float(info.get("wrist_rgb_mean", 0.0))
                    wrist_rgb_std_sum += float(info.get("wrist_rgb_std", 0.0))
                    wrist_rgb_health_count += int(
                        bool(info.get("wrist_rgb_health_ok", False))
                    )
                    wrist_rgb_sample_count += 1
                distance = float(info.get("safety_minimum_distance_m", float("inf")))
                if np.isfinite(distance):
                    safety_minimum_distance = min(safety_minimum_distance, distance)
                hard_penetration_count += int(
                    info.get("termination_reason") == "robot_environment_penetration"
                )
                clearance_violation_count += int(
                    info.get("termination_reason")
                    in {
                        "robot_environment_clearance_violation",
                        "sim_collision_margin_violation",
                    }
                )

            transition_reward_next = np.stack(
                [self._training_reward_state(info) for info in transition_infos]
            )
            succeeded = np.asarray(
                [
                    is_cube_lift_success_reason(
                        info.get("termination_reason")
                    )
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            termination_reasons = [
                info.get("termination_reason") for info in transition_infos
            ]
            failure_termination = (
                terminated.astype(np.float32) * (1.0 - succeeded)
            )
            unsafe_termination = np.asarray(
                [
                    reason in UNSAFE_TERMINATION_REASONS
                    for reason in termination_reasons
                ],
                dtype=np.float32,
            )
            grasp_latched = np.asarray(
                [
                    bool(info.get("grasp_latched", False))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            bilateral_contact = np.asarray(
                [
                    bool(info.get("bilateral_finger_cube_contact", False))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            side_contact_score = np.asarray(
                [
                    float(info.get("cube_side_contact_score", 0.0))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            top_bottom_contact_score = np.asarray(
                [
                    float(info.get("cube_top_bottom_contact_score", 0.0))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            inner_contact_score = np.asarray(
                [
                    float(info.get("gripper_inner_contact_score", 0.0))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            bilateral_inner_contact_score = np.asarray(
                [
                    float(
                        info.get("gripper_bilateral_inner_contact_score", 0.0)
                    )
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            outer_contact_score = np.asarray(
                [
                    float(info.get("gripper_outer_contact_score", 0.0))
                    for info in transition_infos
                ],
                dtype=np.float32,
            )
            task_conditioned_action = np.stack(
                [
                    np.asarray(
                        info.get(
                            "task_conditioned_action",
                            info.get(
                                "requested_action",
                                np.zeros(7, dtype=np.float32),
                            ),
                        ),
                        dtype=np.float32,
                    ).reshape(7)
                    for info in transition_infos
                ]
            )
            admitted_action = np.stack(
                [
                    np.asarray(
                        info.get(
                            "admitted_action",
                            info.get(
                                "accepted_action",
                                np.zeros(7, dtype=np.float32),
                            ),
                        ),
                        dtype=np.float32,
                    ).reshape(7)
                    for info in transition_infos
                ]
            )
            effective_action_delta_sum += float(
                np.sum(np.abs(admitted_action - task_conditioned_action))
            )
            approach_action = np.stack(
                [
                    np.asarray(
                        info.get(
                            "scripted_approach_action",
                            np.zeros(7, dtype=np.float32),
                        ),
                        dtype=np.float32,
                    ).reshape(7)
                    for info in transition_infos
                ]
            )
            shaped_reward, terms = shape_rewards(
                self._reward_observation,
                transition_reward_next,
                raw_reward,
                self.reward_config,
                safety_projection_alpha=safety_alpha,
                safety_intervention=safety_intervention,
                safety_collision_category=safety_collision_category,
                resting_cube_z_m=self._episode_resting_cube_z,
                lift_delta_m=self.task_config.lift_delta_m,
                success=succeeded,
                grasp_latched=grasp_latched,
                bilateral_contact=bilateral_contact,
                previous_bilateral_contact=self._previous_bilateral_contact,
                side_contact_score=side_contact_score,
                top_bottom_contact_score=top_bottom_contact_score,
                inner_contact_score=inner_contact_score,
                bilateral_inner_contact_score=bilateral_inner_contact_score,
                outer_contact_score=outer_contact_score,
                previous_inner_contact_score=self._previous_inner_contact_score,
                previous_bilateral_inner_contact_score=(
                    self._previous_bilateral_inner_contact_score
                ),
                task_action=task_conditioned_action,
                approach_action=approach_action,
                time_limit=truncated,
                failure_termination=failure_termination,
                unsafe_termination=unsafe_termination,
            )
            for name, term_values in terms.items():
                reward_term_sums[name] = reward_term_sums.get(name, 0.0) + float(
                    np.sum(term_values)
                )
            reward_term_samples += cfg.num_envs

            observations[step] = normalized
            critic_observations[step] = normalized_critic
            # PPO credit belongs to the sampled latent action.  Task
            # conditioning, IK, safety projection, and plant response are
            # downstream transition dynamics and must not replace this sample.
            actions[step] = action
            log_probs[step] = log_prob_tensor.cpu().numpy()
            values[step] = value_tensor.cpu().numpy()
            rewards[step] = shaped_reward
            raw_rewards_buffer[step] = raw_reward
            terminated_buffer[step] = terminated.astype(np.float32)
            episode_done_buffer[step] = done.astype(np.float32)

            bootstrap_task_observation = next_task_observation.copy()
            for env_id in np.flatnonzero(done):
                final_observation = infos[env_id].get("final_observation")
                if final_observation is not None:
                    bootstrap_task_observation[env_id] = np.asarray(
                        final_observation, dtype=np.float32
                    )
            bootstrap_observation = encode_policy_observation(
                self.task_config,
                bootstrap_task_observation,
            )
            bootstrap_critic_observation = (
                None
                if self.critic_observation_normalizer is None
                else torch.as_tensor(
                    self.critic_observation_normalizer.normalize(
                        transition_reward_next
                    ),
                    device=self.device,
                )
            )
            with torch.no_grad():
                bootstrap_values[step] = self.policy.value(
                    torch.as_tensor(
                        self.observation_normalizer.normalize(
                            bootstrap_observation
                        ),
                        device=self.device,
                    ),
                    critic_observation=bootstrap_critic_observation,
                ).cpu().numpy()

            self._episode_raw_return += raw_reward
            self._episode_shaped_return += shaped_reward
            self._episode_length += 1
            for env_id in np.flatnonzero(done):
                completed += 1
                episode_returns.append(float(self._episode_raw_return[env_id]))
                transition_info = transition_infos[env_id]
                termination_reason = transition_info.get("termination_reason")
                success = is_cube_lift_success_reason(termination_reason)
                if success:
                    successful += 1
                episode_records.append(
                    {
                        "end_global_step": self.global_step + cfg.num_envs,
                        "env_id": int(env_id),
                        "episode_id": int(transition_info.get("episode_id", -1)),
                        "length": int(self._episode_length[env_id]),
                        "raw_return": float(self._episode_raw_return[env_id]),
                        "shaped_return": float(self._episode_shaped_return[env_id]),
                        "success": bool(success),
                        "termination_reason": termination_reason,
                    }
                )
                self._episode_raw_return[env_id] = 0.0
                self._episode_shaped_return[env_id] = 0.0
                self._episode_length[env_id] = 0

            next_reward_observation = transition_reward_next.copy()
            for env_id in np.flatnonzero(done):
                reset_info = infos[env_id].get("reset_info")
                if reset_info is not None:
                    next_reward_observation[env_id] = self._training_reward_state(
                        reset_info
                    )
                    bilateral_contact[env_id] = float(
                        bool(
                            reset_info.get(
                                "bilateral_finger_cube_contact", False
                            )
                        )
                    )
                    inner_contact_score[env_id] = float(
                        reset_info.get("gripper_inner_contact_score", 0.0)
                    )
                    bilateral_inner_contact_score[env_id] = float(
                        reset_info.get(
                            "gripper_bilateral_inner_contact_score", 0.0
                        )
                    )
            self._episode_resting_cube_z[done] = CubeLiftRewardState.from_flat(
                next_reward_observation[done]
            ).cube_z
            self._previous_bilateral_contact = bilateral_contact
            self._previous_inner_contact_score = inner_contact_score
            self._previous_bilateral_inner_contact_score = (
                bilateral_inner_contact_score
            )

            self.task_observation = next_task_observation
            self.observation = next_observation
            self._reward_observation = next_reward_observation
            self.observation_normalizer.update(next_observation)
            if self.critic_observation_normalizer is not None:
                self.critic_observation_normalizer.update(
                    next_reward_observation
                )
            self.global_step += cfg.num_envs

        advantages, returns = compute_gae(
            rewards,
            values,
            bootstrap_values,
            terminated_buffer,
            episode_done_buffer,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        def tensor(values_: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                values_.reshape((-1, *values_.shape[2:])),
                dtype=torch.float32,
                device=self.device,
            )

        return RolloutBatch(
            observations=tensor(observations),
            critic_observations=tensor(critic_observations),
            actions=tensor(actions),
            old_log_probs=tensor(log_probs),
            advantages=tensor(advantages),
            returns=tensor(returns),
            old_values=tensor(values),
            raw_reward_mean=float(np.mean(raw_rewards_buffer)),
            shaped_reward_mean=float(np.mean(rewards)),
            completed_episodes=completed,
            successful_episodes=successful,
            episode_raw_returns=tuple(episode_returns),
            episodes=tuple(episode_records),
            action_diagnostics=action_diagnostics.summary(),
            safety_projection_rate=safety_modified_count / max(safety_sample_count, 1),
            safety_rejection_rate=safety_rejected_count / max(safety_sample_count, 1),
            safety_recovery_rate=safety_recovery_count / max(safety_sample_count, 1),
            safety_projection_alpha_mean=safety_alpha_sum / max(safety_sample_count, 1),
            safety_collision_category_rates={
                category: count / max(safety_sample_count, 1)
                for category, count in safety_collision_category_counts.items()
            },
            effective_action_override_rate=(
                effective_action_override_count / max(safety_sample_count, 1)
            ),
            effective_action_delta_mean=(
                effective_action_delta_sum
                / max(safety_sample_count * self.environment_action_dim, 1)
            ),
            post_contact_phase_active_rate=(
                post_contact_phase_active_count / max(safety_sample_count, 1)
            ),
            post_contact_action_modified_rate=(
                post_contact_action_modified_count / max(safety_sample_count, 1)
            ),
            scripted_pregrasp_action_modified_rate=(
                scripted_pregrasp_action_modified_count
                / max(safety_sample_count, 1)
            ),
            scripted_contact_action_modified_rate=(
                scripted_contact_action_modified_count
                / max(safety_sample_count, 1)
            ),
            safety_minimum_distance_m=(
                None if not np.isfinite(safety_minimum_distance) else safety_minimum_distance
            ),
            hard_penetration_count=hard_penetration_count,
            clearance_violation_count=clearance_violation_count,
            arm_tracking_error_rad_mean=(
                arm_tracking_error_sum / max(safety_sample_count, 1)
            ),
            arm_tracking_error_rad_max=arm_tracking_error_max,
            gripper_tracking_error_mean=(
                gripper_tracking_error_sum / max(safety_sample_count, 1)
            ),
            gripper_tracking_error_max=gripper_tracking_error_max,
            wrist_rgb_mean=(
                None
                if wrist_rgb_sample_count == 0
                else wrist_rgb_mean_sum / wrist_rgb_sample_count
            ),
            wrist_rgb_std=(
                None
                if wrist_rgb_sample_count == 0
                else wrist_rgb_std_sum / wrist_rgb_sample_count
            ),
            wrist_rgb_health_rate=(
                None
                if wrist_rgb_sample_count == 0
                else wrist_rgb_health_count / wrist_rgb_sample_count
            ),
            reward_term_means={
                f"reward_{name}_mean": total / max(reward_term_samples, 1)
                for name, total in reward_term_sums.items()
            },
        )

    def update(self, batch: RolloutBatch) -> dict[str, float]:
        cfg = self.config
        count = batch.observations.shape[0]
        advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std(unbiased=False) + 1e-8
        )
        # A corrupted bootstrap or a rare extreme terminal return must not
        # poison the actor update.  The clip is only on the normalized GAE;
        # value targets remain untouched for critic fitting.
        advantages = torch.nan_to_num(
            advantages,
            nan=0.0,
            posinf=float(cfg.advantage_clip),
            neginf=-float(cfg.advantage_clip),
        ).clamp(-float(cfg.advantage_clip), float(cfg.advantage_clip))
        with torch.no_grad():
            return_variance = torch.var(batch.returns, unbiased=False)
            if torch.isfinite(return_variance) and float(return_variance) > 1e-8:
                explained_variance = 1.0 - torch.var(
                    batch.returns - batch.old_values,
                    unbiased=False,
                ) / return_variance
                explained_variance_value = float(explained_variance.cpu())
            else:
                explained_variance_value = 0.0
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "actor_reference_mse": [],
            "value_loss": [],
            "value_clip_fraction": [],
            "actor_grad_norm": [],
            "critic_grad_norm": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        early_stop_count = 0
        minibatch_count = 0
        stop_update = False
        self.policy.train()
        for _epoch in range(cfg.update_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, cfg.minibatch_size):
                minibatch_count += 1
                index = permutation[start : start + cfg.minibatch_size]
                new_log_prob, entropy, new_value = self.policy.evaluate_actions(
                    batch.observations[index],
                    batch.actions[index],
                    critic_observation=(
                        batch.critic_observations[index]
                        if self.critic_observation_normalizer is not None
                        else None
                    ),
                )
                raw_log_ratio = new_log_prob - batch.old_log_probs[index]
                if not torch.isfinite(raw_log_ratio).all():
                    early_stop_count += 1
                    stop_update = True
                    break
                log_ratio = torch.clamp(
                    raw_log_ratio,
                    -float(cfg.max_log_ratio),
                    float(cfg.max_log_ratio),
                )
                ratio = log_ratio.exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                # Once this minibatch is already outside the trust region,
                # do not apply another gradient step.  Without this guard a
                # single projected-action outlier can make exp(log_ratio)
                # dominate the remaining epochs and eventually become NaN.
                if (
                    not torch.isfinite(approx_kl)
                    or float(approx_kl) > float(cfg.target_kl)
                ):
                    if torch.isfinite(approx_kl):
                        metrics["approx_kl"].append(
                            float(approx_kl.detach().cpu())
                        )
                    early_stop_count += 1
                    stop_update = True
                    break
                minibatch_advantage = advantages[index]
                unclipped = ratio * minibatch_advantage
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                ) * minibatch_advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss, value_clip_fraction = compute_clipped_value_loss(
                    new_value,
                    batch.old_values[index],
                    batch.returns[index],
                    clip_coef=cfg.value_clip_coef,
                )
                entropy_mean = entropy.mean()
                if float(cfg.actor_reference_mse_coef) > 0.0:
                    if self._actor_reference_policy is None:
                        raise RuntimeError(
                            "actor_reference_mse_coef requires a captured "
                            "post-initialization actor reference."
                        )
                    actor_reference_mse = compute_actor_reference_mse(
                        self.policy,
                        self._actor_reference_policy,
                        batch.observations[index],
                    )
                else:
                    actor_reference_mse = torch.zeros(
                        (),
                        dtype=policy_loss.dtype,
                        device=policy_loss.device,
                    )
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                    + cfg.actor_reference_mse_coef * actor_reference_mse
                )
                if not torch.isfinite(loss):
                    early_stop_count += 1
                    stop_update = True
                    break
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad_norm, critic_grad_norm = (
                    clip_actor_critic_grad_norms(
                        self.policy,
                        max_grad_norm=cfg.max_grad_norm,
                    )
                )
                if not torch.isfinite(actor_grad_norm) or not torch.isfinite(
                    critic_grad_norm
                ):
                    self.optimizer.zero_grad(set_to_none=True)
                    early_stop_count += 1
                    stop_update = True
                    break
                self.optimizer.step()

                with torch.no_grad():
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()
                    )
                metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                metrics["actor_reference_mse"].append(
                    float(actor_reference_mse.detach().cpu())
                )
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["value_clip_fraction"].append(
                    float(value_clip_fraction.detach().cpu())
                )
                metrics["actor_grad_norm"].append(
                    float(actor_grad_norm.detach().cpu())
                )
                metrics["critic_grad_norm"].append(
                    float(critic_grad_norm.detach().cpu())
                )
                metrics["entropy"].append(float(entropy_mean.detach().cpu()))
                metrics["approx_kl"].append(float(approx_kl.detach().cpu()))
                metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))
            if stop_update:
                break
        self.update_index += 1
        if not metrics["policy_loss"]:
            # The first minibatch can be rejected when resuming from a policy
            # that already drifted.  Returning finite diagnostics lets the
            # outer loop checkpoint and continue instead of destroying the
            # last valid checkpoint with NaNs.
            metrics = {
                name: [0.0] if not values else values
                for name, values in metrics.items()
            }
        result = {name: float(np.mean(values_)) for name, values_ in metrics.items()}
        result["ppo_early_stop_rate"] = early_stop_count / max(minibatch_count, 1)
        result["explained_variance"] = explained_variance_value
        return result


__all__ = [
    "PPOSmokeTrainer",
    "RolloutBatch",
    "UNSAFE_TERMINATION_REASONS",
    "clip_actor_critic_grad_norms",
    "compute_actor_reference_mse",
    "compute_clipped_value_loss",
    "compute_gae",
]

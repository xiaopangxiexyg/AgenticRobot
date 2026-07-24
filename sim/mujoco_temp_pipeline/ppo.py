"""A deliberately small PPO implementation wired to CubeLiftVectorExecutor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from sim.mujoco.runtime import CubeLiftVectorExecutor
from sim.mujoco.tasks.cube_lift.config import (
    CubeLiftTaskConfig,
    is_cube_lift_success_reason,
)
from sim.mujoco.learning import encode_model_input
from sim.mujoco.tasks.cube_lift.learning import (
    build_action_binding,
    build_model_input_binding,
    build_observation_contract,
    project_flat_observation,
)
from sim.mujoco.tasks.cube_lift.reward_adapter import CubeLiftRewardState

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
    safety_projection_rate: float
    safety_rejection_rate: float
    safety_recovery_rate: float
    safety_projection_alpha_mean: float
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
        self.observation_dim = self.task_config.observation_dim
        self.environment_action_dim = self.task_config.action_dim
        self.active_action_indices = (
            self.task_config.policy_active_action_indices
        )
        self.action_dim = len(self.active_action_indices)
        self.device = resolve_device(ppo_config.device)
        np.random.seed(ppo_config.seed)
        torch.manual_seed(ppo_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ppo_config.seed)

        self.policy = ActorCritic(
            self.observation_dim,
            self.action_dim,
            ppo_config.hidden_dim,
            proprio_dim=self.task_config.proprio_observation_dim,
            image_shape=self.task_config.wrist_rgb_shape,
            image_pool_size=self.task_config.wrist_rgb_spatial_pool_size,
            environment_action_dim=self.environment_action_dim,
            active_action_indices=self.active_action_indices,
            trainable_log_std=False,
            initial_log_std=ppo_config.initial_log_std,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            (
                parameter
                for parameter in self.policy.parameters()
                if parameter.requires_grad
            ),
            lr=ppo_config.learning_rate,
            eps=1e-5,
        )
        self.observation_normalizer = RunningMeanStd.create((self.observation_dim,))
        self.env = CubeLiftVectorExecutor(
            ppo_config.num_envs,
            config_path=task_config_path,
            model_cache_dir=model_cache_dir,
            auto_reset=True,
            master_seed=ppo_config.seed,
            start_method=ppo_config.start_method,
        )
        self.observation, reset_infos = self.env.reset()
        self.learning_observation_contract = build_observation_contract(self.task_config)
        self.learning_model_input_binding = build_model_input_binding(
            self.task_config,
            self.learning_observation_contract,
        )
        self.learning_action_binding = build_action_binding(self.task_config)
        if self.learning_action_binding.output_dim != self.environment_action_dim:
            self.env.close()
            raise RuntimeError(
                "CubeLift environment action binding dimension does not match "
                f"the task: binding={self.learning_action_binding.output_dim}, "
                f"environment={self.environment_action_dim}."
            )
        expected_input_dim = self.learning_model_input_binding.expected_flat_dim(
            self.learning_observation_contract
        )
        if expected_input_dim != self.observation_dim:
            self.env.close()
            raise RuntimeError(
                "CubeLift learning binding dimension does not match policy input: "
                f"binding={expected_input_dim}, policy={self.observation_dim}."
            )
        encoded_reset = encode_model_input(
            project_flat_observation(self.task_config, self.observation),
            self.learning_observation_contract,
            self.learning_model_input_binding,
        )
        if not np.array_equal(encoded_reset.flat, self.observation):
            self.env.close()
            raise RuntimeError(
                "CubeLift learning binding changed the reset observation; "
                "refuse to start PPO with sim/real input drift."
            )
        if require_wrist_rgb_health and self.task_config.wrist_rgb_enabled:
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
        self._apply_exploration_schedule()

    @staticmethod
    def _training_reward_state(info: dict[str, Any]) -> np.ndarray:
        state = np.asarray(info["training_reward_state"], dtype=np.float32).reshape(-1)
        if state.shape != (31,) or not np.all(np.isfinite(state)):
            raise RuntimeError("CubeLift training_reward_state must be finite and 31-D.")
        return state

    def close(self) -> None:
        self.env.close()

    def _apply_exploration_schedule(self) -> float:
        anneal_steps = max(
            float(self.config.total_timesteps)
            * float(self.config.std_anneal_fraction),
            1.0,
        )
        progress = min(max(float(self.global_step) / anneal_steps, 0.0), 1.0)
        log_std = (
            float(self.config.initial_log_std)
            + progress
            * (
                float(self.config.final_log_std)
                - float(self.config.initial_log_std)
            )
        )
        with torch.no_grad():
            self.policy.log_std.fill_(log_std)
        return log_std

    def collect_rollout(self) -> RolloutBatch:
        cfg = self.config
        self._apply_exploration_schedule()
        shape = (cfg.rollout_steps, cfg.num_envs)
        observations = np.empty((*shape, self.observation_dim), dtype=np.float32)
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
        safety_sample_count = 0
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

        self.policy.eval()
        for step in range(cfg.rollout_steps):
            normalized = self.observation_normalizer.normalize(self.observation)
            observation_tensor = torch.as_tensor(normalized, device=self.device)
            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = self.policy.act(
                    observation_tensor
                )
                environment_action_tensor = self.policy.expand_action(action_tensor)
            action = action_tensor.cpu().numpy().astype(np.float32, copy=False)
            environment_action = (
                environment_action_tensor.cpu().numpy().astype(
                    np.float32, copy=False
                )
            )
            next_observation, raw_reward, terminated, truncated, infos = self.env.step(
                environment_action
            )
            done = np.logical_or(terminated, truncated)
            safety_alpha = np.asarray(
                [float(info.get("safety_projection_alpha", 1.0)) for info in infos],
                dtype=np.float32,
            )
            safety_modified_count += sum(
                bool(info.get("safety_modified", False)) for info in infos
            )
            safety_rejected_count += int(np.count_nonzero(safety_alpha <= 1e-8))
            safety_recovery_count += sum(
                bool(info.get("safety_recovery", False)) for info in infos
            )
            safety_alpha_sum += float(np.sum(safety_alpha))
            safety_sample_count += cfg.num_envs
            for info in infos:
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
                [self._training_reward_state(info) for info in infos]
            )
            for env_id, info in enumerate(infos):
                if done[env_id] and "final_info" in info:
                    transition_reward_next[env_id] = self._training_reward_state(
                        info["final_info"]
                    )
            succeeded = np.asarray(
                [
                    is_cube_lift_success_reason(
                        info.get("termination_reason")
                    )
                    for info in infos
                ],
                dtype=np.float32,
            )
            termination_reasons = [
                info.get("termination_reason") for info in infos
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
                [bool(info.get("grasp_latched", False)) for info in infos],
                dtype=np.float32,
            )
            bilateral_contact = np.asarray(
                [
                    bool(info.get("bilateral_finger_cube_contact", False))
                    for info in infos
                ],
                dtype=np.float32,
            )
            shaped_reward, terms = shape_rewards(
                self._reward_observation,
                transition_reward_next,
                raw_reward,
                self.reward_config,
                safety_projection_alpha=safety_alpha,
                resting_cube_z_m=self._episode_resting_cube_z,
                lift_delta_m=self.task_config.lift_delta_m,
                success=succeeded,
                grasp_latched=grasp_latched,
                bilateral_contact=bilateral_contact,
                previous_bilateral_contact=self._previous_bilateral_contact,
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
            actions[step] = action
            log_probs[step] = log_prob_tensor.cpu().numpy()
            values[step] = value_tensor.cpu().numpy()
            rewards[step] = shaped_reward
            raw_rewards_buffer[step] = raw_reward
            terminated_buffer[step] = terminated.astype(np.float32)
            episode_done_buffer[step] = done.astype(np.float32)

            bootstrap_observation = next_observation.copy()
            for env_id in np.flatnonzero(done):
                final_observation = infos[env_id].get("final_observation")
                if final_observation is not None:
                    bootstrap_observation[env_id] = np.asarray(
                        final_observation, dtype=np.float32
                    )
            with torch.no_grad():
                bootstrap_values[step] = self.policy.value(
                    torch.as_tensor(
                        self.observation_normalizer.normalize(
                            bootstrap_observation
                        ),
                        device=self.device,
                    )
                ).cpu().numpy()

            self._episode_raw_return += raw_reward
            self._episode_shaped_return += shaped_reward
            self._episode_length += 1
            for env_id in np.flatnonzero(done):
                completed += 1
                episode_returns.append(float(self._episode_raw_return[env_id]))
                termination_reason = infos[env_id].get("termination_reason")
                success = is_cube_lift_success_reason(termination_reason)
                if success:
                    successful += 1
                episode_records.append(
                    {
                        "end_global_step": self.global_step + cfg.num_envs,
                        "env_id": int(env_id),
                        "episode_id": int(infos[env_id].get("episode_id", -1)),
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
            self._episode_resting_cube_z[done] = CubeLiftRewardState.from_flat(
                next_reward_observation[done]
            ).cube_z
            self._previous_bilateral_contact = bilateral_contact

            self.observation = next_observation
            self._reward_observation = next_reward_observation
            self.observation_normalizer.update(next_observation)
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
            safety_projection_rate=safety_modified_count / max(safety_sample_count, 1),
            safety_rejection_rate=safety_rejected_count / max(safety_sample_count, 1),
            safety_recovery_rate=safety_recovery_count / max(safety_sample_count, 1),
            safety_projection_alpha_mean=safety_alpha_sum / max(safety_sample_count, 1),
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
        metrics: dict[str, list[float]] = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "approx_kl": [],
            "clip_fraction": [],
        }
        self.policy.train()
        for _epoch in range(cfg.update_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, cfg.minibatch_size):
                index = permutation[start : start + cfg.minibatch_size]
                new_log_prob, entropy, new_value = self.policy.evaluate_actions(
                    batch.observations[index], batch.actions[index]
                )
                log_ratio = new_log_prob - batch.old_log_probs[index]
                ratio = log_ratio.exp()
                minibatch_advantage = advantages[index]
                unclipped = ratio * minibatch_advantage
                clipped = torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef
                ) * minibatch_advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * torch.mean(
                    torch.square(new_value - batch.returns[index])
                )
                entropy_mean = entropy.mean()
                loss = (
                    policy_loss
                    + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy_mean
                )
                if not torch.isfinite(loss):
                    raise RuntimeError("PPO loss became non-finite.")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > cfg.clip_coef).float().mean()
                    )
                metrics["policy_loss"].append(float(policy_loss.detach().cpu()))
                metrics["value_loss"].append(float(value_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy_mean.detach().cpu()))
                metrics["approx_kl"].append(float(approx_kl.detach().cpu()))
                metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))
        self.update_index += 1
        return {name: float(np.mean(values_)) for name, values_ in metrics.items()}


__all__ = [
    "PPOSmokeTrainer",
    "RolloutBatch",
    "UNSAFE_TERMINATION_REASONS",
    "compute_gae",
]

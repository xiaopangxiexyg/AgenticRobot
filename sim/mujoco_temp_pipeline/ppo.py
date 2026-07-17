"""A deliberately small PPO implementation wired to CubeLiftVectorExecutor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from sim.mujoco.runtime import CubeLiftVectorExecutor
from sim.mujoco.tasks.cube_lift.config import CubeLiftTaskConfig

from .config import PPOConfig, RewardShapingConfig
from .normalization import RunningMeanStd
from .policy import ActorCritic, resolve_device
from .reward import shape_rewards


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
    safety_projection_alpha_mean: float
    safety_minimum_distance_m: float | None
    hard_penetration_count: int
    clearance_violation_count: int
    reward_term_means: dict[str, float]


class PPOSmokeTrainer:
    observation_dim = 31
    action_dim = 7

    def __init__(
        self,
        ppo_config: PPOConfig,
        reward_config: RewardShapingConfig,
        *,
        task_config_path: str,
        model_cache_dir: str | None = None,
    ) -> None:
        self.config = ppo_config
        self.reward_config = reward_config
        self.task_config_path = str(task_config_path)
        self.task_config = CubeLiftTaskConfig.load(task_config_path)
        self.device = resolve_device(ppo_config.device)
        np.random.seed(ppo_config.seed)
        torch.manual_seed(ppo_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(ppo_config.seed)

        self.policy = ActorCritic(
            self.observation_dim,
            self.action_dim,
            ppo_config.hidden_dim,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
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
        self.observation, _infos = self.env.reset()
        self.observation_normalizer.update(self.observation)
        self.global_step = 0
        self.update_index = 0
        self._episode_raw_return = np.zeros(ppo_config.num_envs, dtype=np.float64)
        self._episode_shaped_return = np.zeros(ppo_config.num_envs, dtype=np.float64)
        self._episode_length = np.zeros(ppo_config.num_envs, dtype=np.int32)
        self._episode_resting_cube_z = self.observation[:, 9].astype(
            np.float32, copy=True
        )

    def close(self) -> None:
        self.env.close()

    def collect_rollout(self) -> RolloutBatch:
        cfg = self.config
        shape = (cfg.rollout_steps, cfg.num_envs)
        observations = np.empty((*shape, self.observation_dim), dtype=np.float32)
        actions = np.empty((*shape, self.action_dim), dtype=np.float32)
        log_probs = np.empty(shape, dtype=np.float32)
        values = np.empty(shape, dtype=np.float32)
        rewards = np.empty(shape, dtype=np.float32)
        raw_rewards_buffer = np.empty(shape, dtype=np.float32)
        dones = np.empty(shape, dtype=np.float32)
        episode_returns: list[float] = []
        episode_records: list[dict[str, Any]] = []
        completed = 0
        successful = 0
        safety_modified_count = 0
        safety_rejected_count = 0
        safety_alpha_sum = 0.0
        safety_sample_count = 0
        safety_minimum_distance = float("inf")
        hard_penetration_count = 0
        clearance_violation_count = 0
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
            action = action_tensor.cpu().numpy().astype(np.float32, copy=False)
            next_observation, raw_reward, terminated, truncated, infos = self.env.step(action)
            done = np.logical_or(terminated, truncated)
            safety_alpha = np.asarray(
                [float(info.get("safety_projection_alpha", 1.0)) for info in infos],
                dtype=np.float32,
            )
            safety_modified_count += sum(
                bool(info.get("safety_modified", False)) for info in infos
            )
            safety_rejected_count += int(np.count_nonzero(safety_alpha <= 1e-8))
            safety_alpha_sum += float(np.sum(safety_alpha))
            safety_sample_count += cfg.num_envs
            for info in infos:
                distance = float(info.get("safety_minimum_distance_m", float("inf")))
                if np.isfinite(distance):
                    safety_minimum_distance = min(safety_minimum_distance, distance)
                hard_penetration_count += int(
                    info.get("termination_reason") == "robot_environment_penetration"
                )
                clearance_violation_count += int(
                    info.get("termination_reason")
                    == "robot_environment_clearance_violation"
                )

            transition_next = next_observation.copy()
            for env_id, info in enumerate(infos):
                if done[env_id] and "final_observation" in info:
                    transition_next[env_id] = np.asarray(
                        info["final_observation"], dtype=np.float32
                    )
            succeeded = np.asarray(
                [
                    info.get("termination_reason") == "height_only_lift_success"
                    for info in infos
                ],
                dtype=np.float32,
            )
            grasp_latched = np.asarray(
                [bool(info.get("grasp_latched", False)) for info in infos],
                dtype=np.float32,
            )
            shaped_reward, terms = shape_rewards(
                self.observation,
                transition_next,
                raw_reward,
                self.reward_config,
                safety_projection_alpha=safety_alpha,
                resting_cube_z_m=self._episode_resting_cube_z,
                lift_delta_m=self.task_config.lift_delta_m,
                success=succeeded,
                grasp_latched=grasp_latched,
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
            dones[step] = done.astype(np.float32)

            self._episode_raw_return += raw_reward
            self._episode_shaped_return += shaped_reward
            self._episode_length += 1
            for env_id in np.flatnonzero(done):
                completed += 1
                episode_returns.append(float(self._episode_raw_return[env_id]))
                termination_reason = infos[env_id].get("termination_reason")
                success = termination_reason == "height_only_lift_success"
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

            self._episode_resting_cube_z[done] = next_observation[done, 9]

            self.observation = next_observation
            self.observation_normalizer.update(next_observation)
            self.global_step += cfg.num_envs

        with torch.no_grad():
            next_value = self.policy.value(
                torch.as_tensor(
                    self.observation_normalizer.normalize(self.observation),
                    device=self.device,
                )
            ).cpu().numpy()

        advantages = np.zeros(shape, dtype=np.float32)
        last_gae = np.zeros(cfg.num_envs, dtype=np.float32)
        for step in reversed(range(cfg.rollout_steps)):
            next_values = next_value if step == cfg.rollout_steps - 1 else values[step + 1]
            next_non_terminal = 1.0 - dones[step]
            delta = (
                rewards[step]
                + cfg.gamma * next_values * next_non_terminal
                - values[step]
            )
            last_gae = (
                delta
                + cfg.gamma * cfg.gae_lambda * next_non_terminal * last_gae
            )
            advantages[step] = last_gae
        returns = advantages + values

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
            safety_projection_alpha_mean=safety_alpha_sum / max(safety_sample_count, 1),
            safety_minimum_distance_m=(
                None if not np.isfinite(safety_minimum_distance) else safety_minimum_distance
            ),
            hard_penetration_count=hard_penetration_count,
            clearance_violation_count=clearance_violation_count,
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


__all__ = ["PPOSmokeTrainer", "RolloutBatch"]

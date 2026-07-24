"""Small, explicit configuration objects for the temporary PPO pipeline."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardShapingConfig:
    enabled: bool = True
    # Legacy v0 keeps the historical narrow grasp gate.  The staged profile
    # used by normal_rgb_v3 exposes an earlier close gate while retaining the
    # strict gate for lift confidence and success.
    staged_grasp: bool = False
    reach_reward_scale: float = 0.25
    reach_sigma_m: float = 0.25
    reach_progress_scale: float = 4.0
    grasp_height_offset_m: float = 0.03
    closed_grasp_height_offset_m: float = 0.025
    grasp_distance_m: float = 0.05
    grasp_gate_width_m: float = 0.01
    close_gate_distance_m: float = 0.12
    close_gate_width_m: float = 0.03
    near_reach_decay: float = 0.75
    grasp_region_reward_scale: float = 0.15
    gripper_reward_scale: float = 0.25
    gripper_progress_scale: float = 2.0
    premature_close_penalty: float = 0.02
    grasp_acquire_close: float = 0.45
    grasp_acquire_gate: float = 0.25
    grasp_acquire_bonus: float = 1.0
    grasp_close_confidence_width: float = 0.05
    bilateral_contact_reward_scale: float = 0.0
    bilateral_contact_acquire_bonus: float = 0.0
    bilateral_contact_missing_penalty: float = 0.0
    require_bilateral_contact_for_lift_reward: bool = False
    ungated_negative_lift_progress: bool = False
    latch_reward_scale: float = 0.50
    lift_reward_scale: float = 6.0
    lift_progress_scale: float = 30.0
    success_bonus: float = 20.0
    action_penalty: float = 0.01
    safety_projection_penalty: float = 0.25
    time_limit_penalty: float = 0.0
    failure_termination_penalty: float = 0.0
    unsafe_termination_penalty: float = 0.0

    def __post_init__(self) -> None:
        non_negative = {
            "reach_reward_scale": self.reach_reward_scale,
            "reach_progress_scale": self.reach_progress_scale,
            "grasp_region_reward_scale": self.grasp_region_reward_scale,
            "gripper_reward_scale": self.gripper_reward_scale,
            "gripper_progress_scale": self.gripper_progress_scale,
            "latch_reward_scale": self.latch_reward_scale,
            "premature_close_penalty": self.premature_close_penalty,
            "lift_reward_scale": self.lift_reward_scale,
            "lift_progress_scale": self.lift_progress_scale,
            "success_bonus": self.success_bonus,
            "action_penalty": self.action_penalty,
            "safety_projection_penalty": self.safety_projection_penalty,
            "grasp_acquire_bonus": self.grasp_acquire_bonus,
            "bilateral_contact_reward_scale": self.bilateral_contact_reward_scale,
            "bilateral_contact_acquire_bonus": self.bilateral_contact_acquire_bonus,
            "bilateral_contact_missing_penalty": (
                self.bilateral_contact_missing_penalty
            ),
            "time_limit_penalty": self.time_limit_penalty,
            "failure_termination_penalty": self.failure_termination_penalty,
            "unsafe_termination_penalty": self.unsafe_termination_penalty,
        }
        invalid = {name: value for name, value in non_negative.items() if value < 0.0}
        if invalid:
            raise ValueError(f"Reward scales must be non-negative: {invalid}.")
        positive = {
            "reach_sigma_m": self.reach_sigma_m,
            "grasp_distance_m": self.grasp_distance_m,
            "grasp_gate_width_m": self.grasp_gate_width_m,
            "close_gate_distance_m": self.close_gate_distance_m,
            "close_gate_width_m": self.close_gate_width_m,
            "grasp_close_confidence_width": self.grasp_close_confidence_width,
            "grasp_height_offset_m": self.grasp_height_offset_m,
            "closed_grasp_height_offset_m": self.closed_grasp_height_offset_m,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0.0}
        if invalid:
            raise ValueError(f"Reward distances must be positive: {invalid}.")
        if self.closed_grasp_height_offset_m > self.grasp_height_offset_m:
            raise ValueError(
                "closed_grasp_height_offset_m cannot exceed grasp_height_offset_m."
            )
        unit_values = {
            "near_reach_decay": self.near_reach_decay,
            "grasp_acquire_close": self.grasp_acquire_close,
            "grasp_acquire_gate": self.grasp_acquire_gate,
        }
        if any(
            not 0.0 <= float(value) <= 1.0
            for value in unit_values.values()
        ):
            raise ValueError(
                "Staged reward unit values must lie in [0, 1]: "
                f"{unit_values}."
            )
        object.__setattr__(
            self,
            "require_bilateral_contact_for_lift_reward",
            bool(self.require_bilateral_contact_for_lift_reward),
        )
        object.__setattr__(
            self,
            "ungated_negative_lift_progress",
            bool(self.ungated_negative_lift_progress),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RewardShapingConfig":
        return cls(**dict(values))


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 200_000
    num_envs: int = 16
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden_dim: int = 64
    seed: int = 0
    checkpoint_interval: int = 50_000
    start_method: str = "spawn"
    device: str = "auto"
    initial_log_std: float = -0.5
    final_log_std: float = -2.5
    std_anneal_fraction: float = 0.75

    def __post_init__(self) -> None:
        positive_ints = {
            "total_timesteps": self.total_timesteps,
            "num_envs": self.num_envs,
            "rollout_steps": self.rollout_steps,
            "update_epochs": self.update_epochs,
            "minibatch_size": self.minibatch_size,
            "hidden_dim": self.hidden_dim,
            "checkpoint_interval": self.checkpoint_interval,
        }
        invalid = {name: value for name, value in positive_ints.items() if int(value) <= 0}
        if invalid:
            raise ValueError(f"PPO positive integer settings required: {invalid}.")
        if self.minibatch_size > self.num_envs * self.rollout_steps:
            raise ValueError("minibatch_size cannot exceed one rollout batch.")
        unit_interval = {
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_coef": self.clip_coef,
        }
        if any(not 0.0 < float(value) <= 1.0 for value in unit_interval.values()):
            raise ValueError(f"PPO unit-interval settings invalid: {unit_interval}.")
        if self.learning_rate <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive.")
        if not math.isfinite(float(self.initial_log_std)) or not math.isfinite(
            float(self.final_log_std)
        ):
            raise ValueError("PPO log standard deviations must be finite.")
        if self.final_log_std > self.initial_log_std:
            raise ValueError(
                "final_log_std cannot exceed initial_log_std for exploration annealing."
            )
        if not 0.0 < float(self.std_anneal_fraction) <= 1.0:
            raise ValueError("std_anneal_fraction must lie in (0, 1].")

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PPOConfig":
        return cls(**dict(values))


__all__ = ["PPOConfig", "RewardShapingConfig"]

"""Minimal tanh-Gaussian actor critic used by the temporary PPO trainer."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn
from torch.distributions import Normal


LOG_PROB_EPS = 1e-6


def resolve_device(requested: str) -> torch.device:
    value = str(requested).lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false. "
            "Check the NVIDIA driver and the locked CUDA 12.6 PyTorch wheel."
        )
    return device


class ActorCritic(nn.Module):
    def __init__(
        self,
        observation_dim: int = 31,
        action_dim: int = 7,
        hidden_dim: int = 64,
        proprio_dim: int | None = None,
        image_shape: Sequence[int] | None = None,
        image_pool_size: int = 1,
        environment_action_dim: int | None = None,
        active_action_indices: Sequence[int] | None = None,
        trainable_log_std: bool = True,
        initial_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.environment_action_dim = (
            self.action_dim
            if environment_action_dim is None
            else int(environment_action_dim)
        )
        self.active_action_indices = (
            tuple(range(self.action_dim))
            if active_action_indices is None
            else tuple(int(value) for value in active_action_indices)
        )
        if self.action_dim <= 0 or self.environment_action_dim <= 0:
            raise ValueError("Policy and environment action dimensions must be positive.")
        if len(self.active_action_indices) != self.action_dim:
            raise ValueError(
                "active_action_indices length must match the policy action dimension."
            )
        if (
            len(set(self.active_action_indices)) != self.action_dim
            or any(
                value < 0 or value >= self.environment_action_dim
                for value in self.active_action_indices
            )
        ):
            raise ValueError(
                "active_action_indices must be unique environment action indices."
            )
        self.trainable_log_std = bool(trainable_log_std)
        self.image_shape = None if image_shape is None else tuple(int(v) for v in image_shape)
        self.image_pool_size = int(image_pool_size)
        if self.image_shape is not None:
            if len(self.image_shape) != 3 or self.image_shape[0] <= 0:
                raise ValueError("image_shape must be a positive CHW image shape.")
            if self.image_shape[0] % 3 != 0:
                raise ValueError("image_shape channel count must be a multiple of RGB channels.")
            if any(value <= 0 for value in self.image_shape):
                raise ValueError("image_shape dimensions must be positive.")
            if self.image_pool_size not in {1, 3}:
                raise ValueError("image_pool_size must be 1 or 3.")
            self.image_dim = int(math.prod(self.image_shape))
            self.proprio_dim = (
                self.observation_dim - self.image_dim
                if proprio_dim is None
                else int(proprio_dim)
            )
            if self.proprio_dim <= 0 or self.proprio_dim + self.image_dim != self.observation_dim:
                raise ValueError(
                    "proprio_dim + image_shape product must equal observation_dim."
                )
            image_embedding_dim = max(16, self.hidden_dim // 2)
            self.proprio_encoder = nn.Sequential(
                nn.Linear(self.proprio_dim, self.hidden_dim),
                nn.Tanh(),
            )
            self.image_encoder = nn.Sequential(
                nn.Conv2d(self.image_shape[0], 16, kernel_size=5, stride=2, padding=2),
                nn.Tanh(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.Tanh(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.Tanh(),
                nn.AdaptiveAvgPool2d((self.image_pool_size, self.image_pool_size)),
                nn.Flatten(),
                nn.Linear(64 * self.image_pool_size * self.image_pool_size, image_embedding_dim),
                nn.Tanh(),
            )
            self.backbone = nn.Sequential(
                nn.Linear(self.hidden_dim + image_embedding_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
        else:
            self.image_dim = 0
            self.proprio_dim = self.observation_dim if proprio_dim is None else int(proprio_dim)
            if self.proprio_dim != self.observation_dim:
                raise ValueError("proprio_dim is only separate when image_shape is provided.")
            self.proprio_encoder = None
            self.image_encoder = None
            self.backbone = nn.Sequential(
                nn.Linear(self.observation_dim, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
            )
        self.actor_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(
            torch.full((self.action_dim,), float(initial_log_std)),
            requires_grad=self.trainable_log_std,
        )
        self.apply(self._initialize_layer)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    @staticmethod
    def _initialize_layer(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2.0))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _features(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dim {self.observation_dim}, got {observation.shape}."
            )
        if self.image_shape is None:
            return self.backbone(observation)
        leading_shape = observation.shape[:-1]
        flat = observation.reshape(-1, self.observation_dim)
        proprio = flat[:, : self.proprio_dim]
        image = flat[:, self.proprio_dim :].reshape(-1, *self.image_shape)
        combined = torch.cat(
            (self.proprio_encoder(proprio), self.image_encoder(image)), dim=-1
        )
        features = self.backbone(combined)
        return features.reshape(*leading_shape, self.hidden_dim)

    def _distribution_and_value(
        self, observation: torch.Tensor
    ) -> tuple[Normal, torch.Tensor]:
        features = self._features(observation)
        mean = self.actor_mean(features)
        std = torch.exp(torch.clamp(self.log_std, -5.0, 1.0)).expand_as(mean)
        return Normal(mean, std), self.critic(features).squeeze(-1)

    @staticmethod
    def _squashed_log_prob(
        distribution: Normal,
        pre_tanh: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        correction = torch.log(torch.clamp(1.0 - action.square(), min=LOG_PROB_EPS))
        return (distribution.log_prob(pre_tanh) - correction).sum(dim=-1)

    def act(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self._distribution_and_value(observation)
        pre_tanh = distribution.mean if deterministic else distribution.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, action)
        return action, log_prob, value

    def expand_action(self, action: torch.Tensor) -> torch.Tensor:
        """Expand compact learned actions into the environment's public layout."""

        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected compact action dim {self.action_dim}, got {action.shape}."
            )
        if (
            self.environment_action_dim == self.action_dim
            and self.active_action_indices == tuple(range(self.action_dim))
        ):
            return action
        expanded = torch.zeros(
            (*action.shape[:-1], self.environment_action_dim),
            dtype=action.dtype,
            device=action.device,
        )
        expanded[..., list(self.active_action_indices)] = action
        return expanded

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        distribution, value = self._distribution_and_value(observation)
        bounded = torch.clamp(action, -1.0 + LOG_PROB_EPS, 1.0 - LOG_PROB_EPS)
        pre_tanh = torch.atanh(bounded)
        log_prob = self._squashed_log_prob(distribution, pre_tanh, bounded)
        # Base-Normal entropy is a stable exploration diagnostic. Exact
        # tanh-distribution entropy has no simple closed form.
        entropy = distribution.entropy().sum(dim=-1)
        return log_prob, entropy, value

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        _distribution, value = self._distribution_and_value(observation)
        return value


__all__ = ["ActorCritic", "resolve_device"]

"""Minimal tanh-Gaussian actor critic used by the temporary PPO trainer."""

from __future__ import annotations

import math

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
    ) -> None:
        super().__init__()
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.observation_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(self.hidden_dim, self.action_dim)
        self.critic = nn.Linear(self.hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), -0.5))
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

    def _features(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.observation_dim:
            raise ValueError(
                f"Expected observation dim {self.observation_dim}, got {observation.shape}."
            )
        return self.backbone(observation)

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

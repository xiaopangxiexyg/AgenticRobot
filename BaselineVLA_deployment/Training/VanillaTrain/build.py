from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from Models.Policies.diffusion_policy import BasicDiffusionPolicy
from Models.Policies.vanilla_mlp_policy import BasicVanillaMLPPolicy
from Models.Policies.vanilla_tactile_vla_policy import VanillaTactileVLAPolicy
from Models.Policies.vla_policy import BasicVLAPolicy
from Training.build_utils import (
    build_adam_optimizer,
    resolve_model_family,
    resolve_tactile_observation_dim,
    resolve_train_dimensions,
)


VANILLA_POLICY_FAMILIES = {
    "mlp": BasicVanillaMLPPolicy,
    "vla": BasicVLAPolicy,
    "diffusion": BasicDiffusionPolicy,
}
def _build_vanilla_policy(state_dim: int, action_dim: int, action_chunk: int, config: Any) -> nn.Module:
    family = resolve_model_family(config)
    if family not in VANILLA_POLICY_FAMILIES:
        raise KeyError(
            f"Unsupported VanillaTrain model family '{family}'. Available families: {', '.join(sorted(VANILLA_POLICY_FAMILIES))}"
        )
    return VANILLA_POLICY_FAMILIES[family](
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk=action_chunk,
        config=config,
    )


def build_model_and_optimizer(config: Any, data_summary: Dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
    state_dim, action_dim, action_chunk = resolve_train_dimensions(data_summary)
    model = _build_vanilla_policy(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)
    optimizer = build_adam_optimizer(config, model)
    return model, optimizer


def build_tactile_model_and_optimizer(config: Any, data_summary: Dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
    family = resolve_model_family(config)
    if family != "vla":
        raise KeyError("vanilla-tactile currently supports only model.family=vla.")
    state_dim, action_dim, action_chunk = resolve_train_dimensions(data_summary)
    tactile_dim = resolve_tactile_observation_dim(data_summary)
    model = VanillaTactileVLAPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk=action_chunk,
        tactile_dim=tactile_dim,
        config=config,
    )
    optimizer = build_adam_optimizer(config, model)
    return model, optimizer

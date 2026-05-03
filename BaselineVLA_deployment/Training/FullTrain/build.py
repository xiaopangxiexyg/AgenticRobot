from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from Models.Policies.fulltrain_diffusion_policy import FullTrainDiffusionPolicy
from Models.Policies.fulltrain_mlp_policy import FullTrainMLPPolicy
from Models.Policies.fulltrain_vla_policy import FullTrainVLAPolicy
from Training.build_utils import build_adam_optimizer, resolve_med_spec, resolve_model_family, resolve_train_dimensions


FULLTRAIN_POLICY_FAMILIES = {
    "mlp": FullTrainMLPPolicy,
    "vla": FullTrainVLAPolicy,
    "diffusion": FullTrainDiffusionPolicy,
}


def _build_fulltrain_policy(state_dim: int, action_dim: int, med_spec: Dict[str, Any], action_chunk: int, config: Any) -> nn.Module:
    family = resolve_model_family(config)
    if family not in FULLTRAIN_POLICY_FAMILIES:
        raise KeyError(
            f"Unsupported FullTrain model family '{family}'. Available families: {', '.join(sorted(FULLTRAIN_POLICY_FAMILIES))}"
        )
    return FULLTRAIN_POLICY_FAMILIES[family](
        state_dim=state_dim,
        action_dim=action_dim,
        med_spec=med_spec,
        action_chunk=action_chunk,
        config=config,
    )


def build_model_and_optimizer(config: Any, data_summary: Dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
    state_dim, action_dim, action_chunk = resolve_train_dimensions(data_summary)
    med_spec = resolve_med_spec(config, data_summary, experiment_name="FullTrain")
    model = _build_fulltrain_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        med_spec=med_spec,
        action_chunk=action_chunk,
        config=config,
    )
    optimizer = build_adam_optimizer(config, model)
    return model, optimizer

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from Models.Policies.gtmed_diffusion_policy import GTMedDiffusionPolicy
from Models.Policies.gtmed_mlp_policy import GTMedMLPPolicy
from Models.Policies.gtmed_vla_policy import GTMedVLAPolicy
from Training.build_utils import build_adam_optimizer, resolve_med_spec, resolve_model_family, resolve_train_dimensions


GTMED_POLICY_FAMILIES = {
    "mlp": GTMedMLPPolicy,
    "vla": GTMedVLAPolicy,
    "diffusion": GTMedDiffusionPolicy,
}


def _build_gtmed_policy(state_dim: int, action_dim: int, med_spec: Dict[str, Any], action_chunk: int, config: Any) -> nn.Module:
    family = resolve_model_family(config)
    if family not in GTMED_POLICY_FAMILIES:
        raise KeyError(
            f"Unsupported GTMed model family '{family}'. Available families: {', '.join(sorted(GTMED_POLICY_FAMILIES))}"
        )
    return GTMED_POLICY_FAMILIES[family](
        state_dim=state_dim,
        action_dim=action_dim,
        med_spec=med_spec,
        action_chunk=action_chunk,
        config=config,
    )


def build_model_and_optimizer(config: Any, data_summary: Dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
    state_dim, action_dim, action_chunk = resolve_train_dimensions(data_summary)
    med_spec = resolve_med_spec(config, data_summary, experiment_name="GTMed")
    model = _build_gtmed_policy(
        state_dim=state_dim,
        action_dim=action_dim,
        med_spec=med_spec,
        action_chunk=action_chunk,
        config=config,
    )
    optimizer = build_adam_optimizer(config, model)
    return model, optimizer

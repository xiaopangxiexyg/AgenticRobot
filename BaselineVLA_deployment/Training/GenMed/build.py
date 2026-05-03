from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from Models.Policies.genmed_diffusion_policy import GenMedDiffusionPolicy
from Models.Policies.genmed_mlp_policy import GenMedMLPPolicy
from Models.Policies.genmed_vla_policy import GenMedVLAPolicy
from Training.build_utils import build_adam_optimizer, resolve_med_spec, resolve_model_family, resolve_state_dim


GENMED_MODEL_FAMILIES = {
    "diffusion": GenMedDiffusionPolicy,
    "mlp": GenMedMLPPolicy,
    "vla": GenMedVLAPolicy,
}


def _build_genmed_model(state_dim: int, med_spec: Dict[str, Any], config: Any) -> nn.Module:
    family = resolve_model_family(config)
    if family not in GENMED_MODEL_FAMILIES:
        raise KeyError(
            f"Unsupported GenMed model family '{family}'. Available families: {', '.join(sorted(GENMED_MODEL_FAMILIES))}"
        )
    return GENMED_MODEL_FAMILIES[family](
        state_dim=state_dim,
        med_spec=med_spec,
        config=config,
    )


def build_model_and_optimizer(config: Any, data_summary: Dict[str, Any]) -> tuple[nn.Module, torch.optim.Optimizer]:
    state_dim = resolve_state_dim(data_summary)
    med_spec = resolve_med_spec(config, data_summary, experiment_name="GenMed")
    model = _build_genmed_model(state_dim=state_dim, med_spec=med_spec, config=config)
    optimizer = build_adam_optimizer(config, model)
    return model, optimizer

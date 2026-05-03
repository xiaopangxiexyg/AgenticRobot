from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.med_support import build_med_projector

from .diffusion_core import DiffusionPolicyCore
from .policy_support import feature_tokens


class GTMedDiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, med_spec: dict[str, Any], action_chunk: int, config: Any) -> None:
        super().__init__()
        self.core = DiffusionPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)
        self.med_projector = build_med_projector(config=config, med_spec=med_spec, feature_dim=self.core.context_dim)

    def _encode_med_tokens(self, med: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return feature_tokens(self.med_projector(med))

    def forward(
        self,
        state: torch.Tensor,
        med: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        med_tokens, med_mask = self._encode_med_tokens(med)
        return self.core(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_id=task_id,
            task_tokens=task_tokens,
            task_mask=task_mask,
            condition_tokens=med_tokens,
            condition_mask=med_mask,
        )

    def compute_custom_loss(self, model_inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        med_tokens, med_mask = self._encode_med_tokens(model_inputs["med"])
        encoded = self.core.encode_context(
            state=model_inputs["state"],
            vision_tokens=model_inputs.get("vision_tokens"),
            vision_mask=model_inputs.get("vision_mask"),
            task_id=model_inputs.get("task_id"),
            task_tokens=model_inputs.get("task_tokens"),
            task_mask=model_inputs.get("task_mask"),
            condition_tokens=med_tokens,
            condition_mask=med_mask,
        )
        return self.core.compute_loss_from_context(context_features=encoded["pooled"], target_action=targets["action"])

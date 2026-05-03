from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models.med_support import build_med_predictor, build_med_projector

from .diffusion_core import DiffusionPolicyCore
from .policy_support import feature_tokens, format_action_output


class FullTrainDiffusionPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, med_spec: dict[str, Any], action_chunk: int, config: Any) -> None:
        super().__init__()
        self.core = DiffusionPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)
        self.med_loss_weight = float(config.experiment.loss_weights.get("med", 1.0))
        self.med_predictor = build_med_predictor(config=config, med_spec=med_spec, input_dim=self.core.context_dim)
        self.med_projector = build_med_projector(config=config, med_spec=med_spec, feature_dim=self.core.context_dim)

    def _prepare_conditioned_context(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        task_tokens, task_mask = self.core.resolve_task_tokens(
            state=state,
            task_id=task_id,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        base_context = self.core.encode_context(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        med_prediction = self.med_predictor(base_context["pooled"])
        med_tokens, med_mask = feature_tokens(self.med_projector(med_prediction))
        conditioned_context = self.core.encode_context(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_tokens=task_tokens,
            task_mask=task_mask,
            condition_tokens=med_tokens,
            condition_mask=med_mask,
        )
        return base_context, med_prediction, conditioned_context

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        base_context, med_prediction, conditioned_context = self._prepare_conditioned_context(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_id=task_id,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        action_sequence = self.core.sample_action_sequence(
            conditioned_context["pooled"],
            dtype=state.dtype,
            device=state.device,
        )
        return {
            "med": med_prediction,
            "action": format_action_output(action_sequence),
            "action_sequence": action_sequence,
            "context_features": conditioned_context["pooled"],
            "base_context_features": base_context["pooled"],
        }

    def compute_custom_loss(self, model_inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        _, med_prediction, conditioned_context = self._prepare_conditioned_context(
            state=model_inputs["state"],
            vision_tokens=model_inputs.get("vision_tokens"),
            vision_mask=model_inputs.get("vision_mask"),
            task_id=model_inputs.get("task_id"),
            task_tokens=model_inputs.get("task_tokens"),
            task_mask=model_inputs.get("task_mask"),
        )
        outputs, action_total_loss, component_losses = self.core.compute_loss_from_context(
            context_features=conditioned_context["pooled"],
            target_action=targets["action"],
        )
        med_loss = F.mse_loss(med_prediction, targets["med"])
        total_loss = action_total_loss + self.med_loss_weight * med_loss
        outputs["med"] = med_prediction
        component_losses["med"] = med_loss
        return outputs, total_loss, component_losses

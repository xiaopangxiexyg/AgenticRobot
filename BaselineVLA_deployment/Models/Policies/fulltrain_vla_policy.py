from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.med_support import build_med_predictor, build_med_projector

from .policy_support import feature_tokens, format_action_output, resolve_hidden_dims
from .vla_core import VLAPolicyCore


class FullTrainVLAPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, med_spec: dict[str, Any], action_chunk: int, config: Any) -> None:
        super().__init__()
        self.core = VLAPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)
        self.med_predictor = build_med_predictor(config=config, med_spec=med_spec, input_dim=self.core.token_dim)
        self.med_projector = build_med_projector(config=config, med_spec=med_spec, feature_dim=self.core.token_dim)

    def _prepare_conditioned_context(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        task_tokens, task_mask = self.core.resolve_task_tokens(
            state=state,
            task_text=task_text,
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
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        base_context, med_prediction, conditioned_context = self._prepare_conditioned_context(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
        )
        action_sequence = self.core.decode_action(conditioned_context)
        return {
            "med": med_prediction,
            "action": format_action_output(action_sequence),
            "action_sequence": action_sequence,
            "context_features": conditioned_context["pooled"],
            "base_context_features": base_context["pooled"],
        }

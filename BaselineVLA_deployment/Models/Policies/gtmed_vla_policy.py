from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.med_support import build_med_projector

from .policy_support import feature_tokens
from .vla_core import VLAPolicyCore


class GTMedVLAPolicy(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, med_spec: dict[str, Any], action_chunk: int, config: Any) -> None:
        super().__init__()
        self.core = VLAPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)
        self.med_projector = build_med_projector(config=config, med_spec=med_spec, feature_dim=self.core.token_dim)

    def forward(
        self,
        state: torch.Tensor,
        med: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        med_tokens, med_mask = feature_tokens(self.med_projector(med))
        return self.core(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
            condition_tokens=med_tokens,
            condition_mask=med_mask,
        )

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .diffusion_core import DiffusionPolicyCore


class BasicDiffusionPolicy(nn.Module):
    """A lightweight diffusion-style action policy for low-dimensional VanillaTrain."""

    def __init__(self, state_dim: int, action_dim: int, action_chunk: int, config: Any) -> None:
        super().__init__()
        self.core = DiffusionPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return self.core(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_id=task_id,
            task_tokens=task_tokens,
            task_mask=task_mask,
            **kwargs,
        )

    def compute_custom_loss(self, model_inputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]):
        return self.core.compute_custom_loss(model_inputs=model_inputs, targets=targets)

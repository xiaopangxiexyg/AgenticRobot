from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from .vla_core import VLAPolicyCore


class BasicVLAPolicy(nn.Module):
    """A lightweight multimodal token policy that keeps a VLA-style interface."""

    def __init__(self, state_dim: int, action_dim: int, action_chunk: int, config: Any) -> None:
        super().__init__()
        self.core = VLAPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        return self.core(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
            **kwargs,
        )

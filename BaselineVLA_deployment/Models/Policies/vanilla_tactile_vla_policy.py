from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp

from .vla_core import VLAPolicyCore


def _resolve_tactile_hidden_dims(config: Any) -> list[int]:
    tactile_config = getattr(config.model, "tactile", None)
    if tactile_config is not None:
        hidden_dims = getattr(tactile_config, "projector_hidden_dims", None)
        if hidden_dims is not None:
            return [int(dim) for dim in hidden_dims]
    hidden_dims = getattr(config.model, "tactile_projector_hidden_dims", None)
    if hidden_dims is not None:
        return [int(dim) for dim in hidden_dims]
    return []


class VanillaTactileVLAPolicy(nn.Module):
    """Vanilla VLA policy augmented with per-frame tactile vector tokens."""

    def __init__(self, state_dim: int, action_dim: int, action_chunk: int, tactile_dim: int, config: Any) -> None:
        super().__init__()
        self.core = VLAPolicyCore(state_dim=state_dim, action_dim=action_dim, action_chunk=action_chunk, config=config)
        self.tactile_dim = int(tactile_dim)
        self.tactile_projector = build_mlp(
            input_dim=self.tactile_dim,
            hidden_dims=_resolve_tactile_hidden_dims(config),
            output_dim=self.core.token_dim,
            activation=str(config.model.activation),
            dropout=float(config.model.dropout),
        )

    def _project_tactile_tokens(
        self,
        tactile: torch.Tensor,
        tactile_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tactile.dim() == 2:
            tactile = tactile.unsqueeze(1)
        if tactile.dim() != 3:
            raise ValueError(f"tactile must have shape [B, D] or [B, T, D], got {tuple(tactile.shape)}")
        if int(tactile.shape[-1]) != self.tactile_dim:
            raise ValueError(f"Expected tactile feature dim={self.tactile_dim}, got {int(tactile.shape[-1])}")

        tokens = self.tactile_projector(tactile)
        if tactile_mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            if tactile_mask.dim() == 1:
                tactile_mask = tactile_mask.unsqueeze(1)
            if tactile_mask.dim() != 2:
                raise ValueError(f"tactile_mask must have shape [B] or [B, T], got {tuple(tactile_mask.shape)}")
            mask = tactile_mask.to(device=tokens.device, dtype=torch.bool)
        return tokens, mask

    def forward(
        self,
        state: torch.Tensor,
        tactile: torch.Tensor,
        tactile_mask: Optional[torch.Tensor] = None,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        task_text: Optional[list[str]] = None,
        task_tokens: Optional[torch.Tensor] = None,
        task_mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        tactile_tokens, tactile_token_mask = self._project_tactile_tokens(tactile=tactile, tactile_mask=tactile_mask)
        return self.core(
            state=state,
            vision_tokens=vision_tokens,
            vision_mask=vision_mask,
            task_text=task_text,
            task_tokens=task_tokens,
            task_mask=task_mask,
            condition_tokens=tactile_tokens,
            condition_mask=tactile_token_mask,
        )

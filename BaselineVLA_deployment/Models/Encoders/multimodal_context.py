from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp
from Models.transformer_utils import resolve_transformer_activation


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denom


class MultimodalContextEncoder(nn.Module):
    """Encode state and optional tokenized observations into shared context features."""

    def __init__(
        self,
        state_dim: int,
        token_dim: int,
        state_hidden_dims: list[int],
        vision_token_dim: int = 4,
        vision_hidden_dims: Optional[list[int]] = None,
        num_attention_heads: int = 4,
        num_transformer_layers: int = 2,
        transformer_ffn_dim: int = 256,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_vision_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.use_vision_tokens = bool(use_vision_tokens)
        self.state_tokenizer = build_mlp(
            input_dim=state_dim,
            hidden_dims=state_hidden_dims,
            output_dim=token_dim,
            activation=activation,
            dropout=dropout,
        )
        self.vision_tokenizer = build_mlp(
            input_dim=int(vision_token_dim),
            hidden_dims=[] if vision_hidden_dims is None else vision_hidden_dims,
            output_dim=token_dim,
            activation=activation,
            dropout=dropout,
        )
        self.state_modality = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.vision_modality = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.extra_modality = nn.Parameter(torch.zeros(1, 1, token_dim))
        self.encoder: nn.Module | None = None
        if int(num_transformer_layers) > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=int(num_attention_heads),
                dim_feedforward=int(transformer_ffn_dim),
                dropout=float(dropout),
                activation=resolve_transformer_activation(activation),
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=int(num_transformer_layers))
        self.final_norm = nn.LayerNorm(token_dim)

    def forward(
        self,
        state: torch.Tensor,
        vision_tokens: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
        extra_tokens: Optional[torch.Tensor] = None,
        extra_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(state.shape[0])
        device = state.device

        token_chunks = [self.state_tokenizer(state).unsqueeze(1) + self.state_modality]
        mask_chunks = [torch.ones(batch_size, 1, dtype=torch.bool, device=device)]

        if self.use_vision_tokens and vision_tokens is not None:
            if vision_tokens.dim() != 3:
                raise ValueError(f"vision_tokens must have shape [B, N, D], got {tuple(vision_tokens.shape)}")
            projected_vision = self.vision_tokenizer(vision_tokens) + self.vision_modality
            if vision_mask is None:
                projected_mask = torch.ones(projected_vision.shape[:2], dtype=torch.bool, device=device)
            else:
                projected_mask = vision_mask.to(device=device, dtype=torch.bool)
            token_chunks.append(projected_vision)
            mask_chunks.append(projected_mask)

        if extra_tokens is not None:
            if extra_tokens.dim() != 3 or int(extra_tokens.shape[-1]) != int(self.token_dim):
                raise ValueError(
                    f"extra_tokens must have shape [B, N, {self.token_dim}], got {tuple(extra_tokens.shape)}"
                )
            if extra_mask is None:
                projected_extra_mask = torch.ones(extra_tokens.shape[:2], dtype=torch.bool, device=device)
            else:
                projected_extra_mask = extra_mask.to(device=device, dtype=torch.bool)
            token_chunks.append(extra_tokens + self.extra_modality)
            mask_chunks.append(projected_extra_mask)

        tokens = torch.cat(token_chunks, dim=1)
        mask = torch.cat(mask_chunks, dim=1)
        if self.encoder is not None:
            tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        tokens = self.final_norm(tokens)
        pooled = _masked_mean(tokens, mask)
        return {
            "tokens": tokens,
            "mask": mask,
            "pooled": pooled,
        }

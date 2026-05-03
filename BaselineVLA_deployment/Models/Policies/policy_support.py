from __future__ import annotations

from typing import Any, Sequence

import torch


def format_action_output(action_sequence: torch.Tensor) -> torch.Tensor:
    if int(action_sequence.shape[1]) == 1:
        return action_sequence[:, 0]
    return action_sequence


def resolve_hidden_dims(config_section: Any, field_name: str, default: Sequence[int] | None = None) -> list[int]:
    value = getattr(config_section, field_name, None)
    if value is None:
        return [] if default is None else [int(dim) for dim in default]
    return [int(dim) for dim in value]


def feature_tokens(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.dim() == 2:
        tokens = features.unsqueeze(1)
    elif features.dim() == 3:
        tokens = features
    else:
        raise ValueError(f"Expected [B, D] or [B, N, D] features, got {tuple(features.shape)}")
    mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=features.device)
    return tokens, mask


def combine_condition_sequences(
    *items: tuple[torch.Tensor | None, torch.Tensor | None],
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    token_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    for tokens, mask in items:
        if tokens is None:
            continue
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(1)
        if tokens.dim() != 3:
            raise ValueError(f"Condition tokens must have shape [B, N, D], got {tuple(tokens.shape)}")
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        else:
            mask = mask.to(device=tokens.device, dtype=torch.bool)
        token_chunks.append(tokens)
        mask_chunks.append(mask)
    if not token_chunks:
        return None, None
    return torch.cat(token_chunks, dim=1), torch.cat(mask_chunks, dim=1)

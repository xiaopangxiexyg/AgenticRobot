from __future__ import annotations

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp


class StateMLPEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], feature_dim: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.network = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=feature_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

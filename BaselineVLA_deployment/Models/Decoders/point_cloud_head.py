from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp


def _product(values: Sequence[int]) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


class PointCloudQueryHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_shape: Sequence[int],
        hidden_dims: Sequence[int],
        query_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        normalized_shape = tuple(int(value) for value in output_shape)
        if len(normalized_shape) < 2:
            raise ValueError(f"PointCloudQueryHead expects a point-cloud output shape, got {normalized_shape}")
        self.output_shape = normalized_shape
        self.point_count = _product(normalized_shape[:-1])
        self.point_feature_dim = int(normalized_shape[-1])
        self.query_tokens = nn.Parameter(torch.randn(1, self.point_count, int(query_dim)) * 0.02)
        self.context_projection = nn.Linear(int(input_dim), int(query_dim))
        self.point_head = build_mlp(
            input_dim=int(query_dim),
            hidden_dims=[int(dim) for dim in hidden_dims],
            output_dim=self.point_feature_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.query_tokens.expand(features.shape[0], -1, -1)
        hidden = hidden + self.context_projection(features).unsqueeze(1)
        point_features = self.point_head(hidden)
        return point_features.view(features.shape[0], *self.output_shape)

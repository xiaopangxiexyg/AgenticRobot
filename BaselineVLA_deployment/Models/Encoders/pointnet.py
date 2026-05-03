from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp


def flatten_point_tensor(points: torch.Tensor) -> torch.Tensor:
    if points.dim() == 3:
        return points
    if points.dim() == 4:
        batch_size, time_steps, point_count, feature_dim = points.shape
        return points.reshape(batch_size, time_steps * point_count, feature_dim)
    raise ValueError(f"Point-cloud tensor must have shape [B, N, C] or [B, T, N, C], got {tuple(points.shape)}")


class PointNetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        normalized_hidden_dims = [int(dim) for dim in hidden_dims]
        local_dim = int(output_dim) if not normalized_hidden_dims else int(normalized_hidden_dims[-1])
        self.point_mlp = build_mlp(
            input_dim=input_dim,
            hidden_dims=normalized_hidden_dims[:-1],
            output_dim=local_dim,
            activation=activation,
            dropout=dropout,
        )
        self.output_projection = nn.Identity() if local_dim == int(output_dim) else nn.Linear(local_dim, int(output_dim))

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        flattened_points = flatten_point_tensor(points)
        point_features = self.point_mlp(flattened_points)
        pooled = point_features.max(dim=1).values
        return self.output_projection(pooled)

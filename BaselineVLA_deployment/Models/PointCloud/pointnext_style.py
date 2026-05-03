from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


def _activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU(inplace=True)
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported activation: {name}")


def _build_linear_stack(input_dim: int, hidden_dims: Sequence[int], output_dim: int, activation: str, dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, int(hidden_dim)))
        layers.append(nn.LayerNorm(int(hidden_dim)))
        layers.append(_activation(activation))
        if float(dropout) > 0.0:
            layers.append(nn.Dropout(float(dropout)))
        previous_dim = int(hidden_dim)
    layers.append(nn.Linear(previous_dim, int(output_dim)))
    return nn.Sequential(*layers)


def _gather_neighbors(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch_size, point_count, channels = values.shape
    _, query_count, neighbor_count = indices.shape
    expanded = values.unsqueeze(1).expand(batch_size, query_count, point_count, channels)
    gather_index = indices.unsqueeze(-1).expand(batch_size, query_count, neighbor_count, channels)
    return torch.gather(expanded, dim=2, index=gather_index)


def _knn_indices(xyz: torch.Tensor, valid_mask: torch.Tensor | None, k_neighbors: int) -> torch.Tensor:
    point_count = int(xyz.shape[1])
    k = min(max(1, int(k_neighbors)), point_count)
    distances = torch.cdist(xyz, xyz)
    if valid_mask is not None:
        support_valid = valid_mask.to(dtype=torch.bool).unsqueeze(1)
        distances = distances.masked_fill(~support_valid, torch.finfo(distances.dtype).max)
    return torch.topk(distances, k=k, dim=-1, largest=False).indices


def masked_mean_max_pool(features: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
    if valid_mask is None:
        return torch.cat([features.mean(dim=1), features.max(dim=1).values], dim=-1)
    mask = valid_mask.to(device=features.device, dtype=features.dtype).unsqueeze(-1)
    count = mask.sum(dim=1).clamp_min(1.0)
    mean = (features * mask).sum(dim=1) / count
    masked_features = features.masked_fill(mask <= 0.0, torch.finfo(features.dtype).min)
    max_values = masked_features.max(dim=1).values
    has_valid = valid_mask.to(dtype=torch.bool, device=features.device).any(dim=1, keepdim=True)
    max_values = torch.where(has_valid, max_values, torch.zeros_like(max_values))
    return torch.cat([mean, max_values], dim=-1)


class PointNeXtStyleBlock(nn.Module):
    """Small pure-PyTorch local aggregation block inspired by PointNeXt.

    This intentionally avoids OpenPoints CUDA extensions so the training path is
    usable in the existing ManipRep_310 environment.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        k_neighbors: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.k_neighbors = int(k_neighbors)
        edge_dim = int(feature_dim) * 2 + 3
        self.edge_mlp = _build_linear_stack(
            input_dim=edge_dim,
            hidden_dims=[int(hidden_dim)],
            output_dim=int(feature_dim),
            activation=activation,
            dropout=dropout,
        )
        self.output_norm = nn.LayerNorm(int(feature_dim))
        self.output_activation = _activation(activation)

    def forward(self, xyz: torch.Tensor, features: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        neighbor_indices = _knn_indices(xyz=xyz, valid_mask=valid_mask, k_neighbors=self.k_neighbors)
        neighbor_xyz = _gather_neighbors(xyz, neighbor_indices)
        neighbor_features = _gather_neighbors(features, neighbor_indices)
        center_xyz = xyz.unsqueeze(2)
        center_features = features.unsqueeze(2).expand_as(neighbor_features)
        edge_features = torch.cat([center_features, neighbor_features, neighbor_xyz - center_xyz], dim=-1)
        aggregated = self.edge_mlp(edge_features).max(dim=2).values
        updated = self.output_activation(self.output_norm(features + aggregated))
        if valid_mask is not None:
            updated = updated * valid_mask.to(dtype=updated.dtype, device=updated.device).unsqueeze(-1)
        return updated


class PointNeXtStyleEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        width: int,
        depth: int,
        k_neighbors: int,
        output_dim: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_projection = _build_linear_stack(
            input_dim=int(input_dim),
            hidden_dims=[],
            output_dim=int(width),
            activation=activation,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                PointNeXtStyleBlock(
                    feature_dim=int(width),
                    hidden_dim=int(width),
                    k_neighbors=int(k_neighbors),
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(int(depth))
            ]
        )
        self.output_projection = _build_linear_stack(
            input_dim=int(width) * 2,
            hidden_dims=[int(width)],
            output_dim=int(output_dim),
            activation=activation,
            dropout=dropout,
        )

    def forward(self, points: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        xyz = points[..., :3].contiguous()
        features = self.input_projection(points)
        if valid_mask is not None:
            features = features * valid_mask.to(dtype=features.dtype, device=features.device).unsqueeze(-1)
        for block in self.blocks:
            features = block(xyz=xyz, features=features, valid_mask=valid_mask)
        pooled = masked_mean_max_pool(features, valid_mask=valid_mask)
        return self.output_projection(pooled)


class PointNeXtStyleQueryDecoder(nn.Module):
    def __init__(
        self,
        point_count: int,
        latent_dim: int,
        query_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        refinement_blocks: int,
        k_neighbors: int,
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.point_count = int(point_count)
        self.query_embedding = nn.Parameter(torch.randn(1, self.point_count, int(query_dim)) * 0.02)
        self.query_xyz = nn.Parameter(torch.randn(1, self.point_count, 3) * 0.02)
        self.latent_projection = nn.Linear(int(latent_dim), int(query_dim))
        self.refinement_blocks = nn.ModuleList(
            [
                PointNeXtStyleBlock(
                    feature_dim=int(query_dim),
                    hidden_dim=int(query_dim),
                    k_neighbors=int(k_neighbors),
                    activation=activation,
                    dropout=dropout,
                )
                for _ in range(int(refinement_blocks))
            ]
        )
        self.output_head = _build_linear_stack(
            input_dim=int(query_dim),
            hidden_dims=[int(dim) for dim in hidden_dims],
            output_dim=int(output_dim),
            activation=activation,
            dropout=dropout,
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        batch_size = int(latent.shape[0])
        features = self.query_embedding.expand(batch_size, -1, -1)
        features = features + self.latent_projection(latent).unsqueeze(1)
        xyz = self.query_xyz.expand(batch_size, -1, -1)
        for block in self.refinement_blocks:
            features = block(xyz=xyz, features=features, valid_mask=None)
        return self.output_head(features)

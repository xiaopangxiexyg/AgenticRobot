from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp


class TaskIDEncoder(nn.Module):
    """Encode discrete task identifiers into features or a single conditioning token."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        output_dim: int,
        hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(int(num_embeddings), int(embedding_dim))
        self.projector = build_mlp(
            input_dim=int(embedding_dim),
            hidden_dims=[] if hidden_dims is None else hidden_dims,
            output_dim=int(output_dim),
            activation=activation,
            dropout=dropout,
        )

    def forward(self, task_id: torch.Tensor) -> dict[str, torch.Tensor]:
        if task_id.dim() > 1:
            task_id = task_id.view(task_id.shape[0], -1)[:, 0]
        task_features = self.projector(self.embedding(task_id.long()))
        return {
            "features": task_features,
            "tokens": task_features.unsqueeze(1),
            "mask": torch.ones(task_features.shape[0], 1, dtype=torch.bool, device=task_features.device),
        }

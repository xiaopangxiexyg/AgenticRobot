from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from Models.mlp_blocks import build_mlp


class MedMLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        med_dim: int,
        activation: str,
        dropout: float,
        output_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.output_shape = None if output_shape is None else tuple(int(value) for value in output_shape)
        self.network = build_mlp(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=med_dim,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.network(features)
        if self.output_shape is not None:
            return output.view(features.shape[0], *self.output_shape)
        return output

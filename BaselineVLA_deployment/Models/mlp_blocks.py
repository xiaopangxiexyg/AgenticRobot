from __future__ import annotations

from typing import Iterable, List

import torch.nn as nn


def _build_activation(name: str) -> nn.Module:
    normalized = str(name).lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise KeyError(f"Unsupported activation: {name}")


def build_mlp(
    input_dim: int,
    hidden_dims: Iterable[int],
    output_dim: int,
    activation: str = "relu",
    dropout: float = 0.0,
) -> nn.Sequential:
    dims: List[int] = [int(input_dim), *[int(hidden) for hidden in hidden_dims], int(output_dim)]
    layers = []
    for layer_index in range(len(dims) - 1):
        in_dim = dims[layer_index]
        out_dim = dims[layer_index + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        is_last_layer = layer_index == len(dims) - 2
        if not is_last_layer:
            layers.append(_build_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)

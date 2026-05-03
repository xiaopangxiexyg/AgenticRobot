from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F


def resolve_transformer_activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    normalized = str(name).lower()
    if normalized == "relu":
        return F.relu
    if normalized == "gelu":
        return F.gelu
    if normalized == "silu":
        return F.silu
    raise KeyError(f"Unsupported transformer activation: {name}")

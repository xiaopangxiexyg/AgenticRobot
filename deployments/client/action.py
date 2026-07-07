from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class UnifiedActionChunk:
    """Dense action chunk in the deployment unified action space.

    The chunk is model-agnostic. Model-specific clients are responsible for
    converting their server response into this dense [horizon, action_dim]
    representation before a real or sim launcher consumes it.
    """

    actions: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 1D or 2D, got shape {actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain non-finite values.")
        object.__setattr__(self, "actions", actions.astype(np.float32, copy=True))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def action_chunk(self) -> np.ndarray:
        return self.actions

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    def iter_vectors(self):
        for index in range(self.horizon):
            yield self.actions[index].astype(np.float32, copy=True)


# Compatibility name for policy-loader code and tests during the transition.
PolicyActionChunk = UnifiedActionChunk


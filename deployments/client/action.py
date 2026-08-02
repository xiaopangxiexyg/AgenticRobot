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
    discrete_actions: np.ndarray | None = None
    arm_hold_probabilities: np.ndarray | None = None
    arm_hold_contract_hash: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2:
            raise ValueError(f"actions must be 1D or 2D, got shape {actions.shape}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions contain non-finite values.")
        discrete_actions = self.discrete_actions
        if discrete_actions is not None:
            raw_tokens = np.asarray(discrete_actions)
            if raw_tokens.ndim == 1:
                raw_tokens = raw_tokens.reshape(1, -1)
            if raw_tokens.ndim != 2:
                raise ValueError(
                    "discrete_actions must be 1D or 2D [horizon, token_dim], "
                    f"got shape {raw_tokens.shape}."
                )
            if raw_tokens.shape[0] != actions.shape[0]:
                raise ValueError(
                    "discrete_actions horizon must match actions: "
                    f"{raw_tokens.shape[0]} != {actions.shape[0]}."
                )
            tokens = np.empty(raw_tokens.shape, dtype=np.int64)
            for index, value in np.ndenumerate(raw_tokens):
                if isinstance(value, (bool, np.bool_)):
                    raise ValueError("discrete_actions must contain integer token ids, not booleans.")
                try:
                    token = int(value)
                    if float(value) != float(token):
                        raise ValueError
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"discrete_actions{index} must be an exact integer token id."
                    ) from exc
                tokens[index] = token
            discrete_actions = tokens
        arm_hold_probabilities = self.arm_hold_probabilities
        arm_hold_contract_hash = self.arm_hold_contract_hash
        if (arm_hold_probabilities is None) != (arm_hold_contract_hash is None):
            raise ValueError(
                "arm_hold_probabilities and arm_hold_contract_hash must be "
                "supplied together."
            )
        if arm_hold_probabilities is not None:
            probabilities = np.asarray(
                arm_hold_probabilities,
                dtype=np.float32,
            )
            if probabilities.ndim == 1:
                probabilities = probabilities.reshape(1, -1)
            if probabilities.shape != (actions.shape[0], 2):
                raise ValueError(
                    "arm_hold_probabilities must have shape "
                    f"{(actions.shape[0], 2)}, got {probabilities.shape}."
                )
            if (
                not np.all(np.isfinite(probabilities))
                or np.any(probabilities < 0.0)
                or np.any(probabilities > 1.0)
            ):
                raise ValueError(
                    "arm_hold_probabilities must contain probabilities in [0, 1]."
                )
            arm_hold_probabilities = probabilities.astype(
                np.float32,
                copy=True,
            )
            arm_hold_contract_hash = str(arm_hold_contract_hash).strip()
            if not arm_hold_contract_hash:
                raise ValueError(
                    "arm_hold_contract_hash must be non-empty when supplied."
                )
        object.__setattr__(self, "actions", actions.astype(np.float32, copy=True))
        object.__setattr__(
            self,
            "discrete_actions",
            None if discrete_actions is None else discrete_actions.astype(np.int64, copy=True),
        )
        object.__setattr__(
            self,
            "arm_hold_probabilities",
            arm_hold_probabilities,
        )
        object.__setattr__(
            self,
            "arm_hold_contract_hash",
            arm_hold_contract_hash,
        )
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

    @property
    def action_discrete_dim(self) -> int:
        return 0 if self.discrete_actions is None else int(self.discrete_actions.shape[1])

    def iter_vectors(self):
        for index in range(self.horizon):
            yield self.actions[index].astype(np.float32, copy=True)


# Compatibility name for policy-loader code and tests during the transition.
PolicyActionChunk = UnifiedActionChunk

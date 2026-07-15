"""Shared state-input modes for training artifacts and deployment adapters.

The on-disk v2.2 ``observation.state`` layout is always the base continuous
state.  A policy artifact may opt into a model-facing representation that
appends an availability mask after the model's state normalization step.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


STATE_INPUT_MODE_RAW = "raw_state_v1"
STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT = "masked_normalized_concat_v1"
StateInputMode = Literal["raw_state_v1", "masked_normalized_concat_v1"]


def normalize_state_input_mode(value: Any) -> StateInputMode:
    mode = str(value or STATE_INPUT_MODE_RAW).strip().lower()
    if mode == STATE_INPUT_MODE_RAW:
        return STATE_INPUT_MODE_RAW
    if mode == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT:
        return STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT
    raise ValueError(
        "Unsupported state_input_mode "
        f"{value!r}; expected {STATE_INPUT_MODE_RAW!r} or "
        f"{STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT!r}."
    )


def state_input_dim(base_state_dim: int, mode: StateInputMode | str = STATE_INPUT_MODE_RAW) -> int:
    base_dim = int(base_state_dim)
    if base_dim < 0:
        raise ValueError("base_state_dim must be non-negative.")
    normalized_mode = normalize_state_input_mode(mode)
    return base_dim if normalized_mode == STATE_INPUT_MODE_RAW else base_dim * 2


def uses_state_mask(mode: StateInputMode | str = STATE_INPUT_MODE_RAW) -> bool:
    return normalize_state_input_mode(mode) == STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT


def concatenate_state_and_mask(state: Any, mask: Any) -> np.ndarray:
    """Return ``[..., state_dim * 2]`` while preserving float32 layout.

    The caller is responsible for normalizing the first half when the policy
    uses ``masked_normalized_concat_v1``.  Deployment uses this helper with an
    all-one mask; its model/server adapter must normalize only the base-state
    half before consuming the concatenated representation.
    """

    values = np.asarray(state, dtype=np.float32)
    availability = np.asarray(mask, dtype=np.float32)
    if values.ndim not in (1, 2):
        raise ValueError(f"state must be [dim] or [batch, dim], got {values.shape}.")
    if availability.shape != values.shape:
        raise ValueError(
            f"state mask shape {availability.shape} must equal state shape {values.shape}."
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(availability)):
        raise ValueError("state and state mask must be finite.")
    if np.any((availability != 0.0) & (availability != 1.0)):
        raise ValueError("state mask values must be binary 0 or 1.")
    return np.concatenate((values, availability), axis=-1).astype(np.float32, copy=False)


def deployment_state_input(
    base_state: Any,
    mode: StateInputMode | str = STATE_INPUT_MODE_RAW,
) -> np.ndarray:
    """Prepare a deployment state with an all-available mask when required."""

    values = np.asarray(base_state, dtype=np.float32)
    if values.ndim not in (1, 2):
        raise ValueError(f"base_state must be [dim] or [batch, dim], got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("base_state must be finite.")
    if not uses_state_mask(mode):
        return values.astype(np.float32, copy=True)
    return concatenate_state_and_mask(values, np.ones_like(values, dtype=np.float32))


__all__ = [
    "STATE_INPUT_MODE_MASKED_NORMALIZED_CONCAT",
    "STATE_INPUT_MODE_RAW",
    "StateInputMode",
    "concatenate_state_and_mask",
    "deployment_state_input",
    "normalize_state_input_mode",
    "state_input_dim",
    "uses_state_mask",
]

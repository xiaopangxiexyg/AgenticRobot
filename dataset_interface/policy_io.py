"""Typed v2.2 low-dimensional samples for training readers.

Image augmentation remains in :mod:`dataset_interface.pipeline`.  This module
only validates and presents the configuration-bound policy I/O tensors so model
code cannot accidentally treat discrete categories as continuous floats.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from dataset_contracts.v22 import (
    ARM_QPOS_AUDIT_DIM,
    INVALID_DISCRETE_TOKEN,
    DatasetLayoutBindingError,
    PolicyIOLayout,
)


def _float_vector(row: Mapping[str, Any], key: str, *, length: int) -> np.ndarray:
    try:
        value = np.asarray(row[key], dtype=np.float32).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetLayoutBindingError(f"v2.2 row is missing a valid {key!r} vector.") from exc
    if value.shape != (int(length),):
        raise DatasetLayoutBindingError(
            f"v2.2 row {key!r} has shape {value.shape}; expected ({int(length)},)."
        )
    return value.astype(np.float32, copy=True)


def _token_vector(row: Mapping[str, Any], key: str, *, length: int) -> np.ndarray:
    if int(length) == 0:
        return np.zeros((0,), dtype=np.int64)
    try:
        raw = np.asarray(row[key]).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise DatasetLayoutBindingError(f"v2.2 row is missing a valid {key!r} token vector.") from exc
    if raw.shape != (int(length),):
        raise DatasetLayoutBindingError(
            f"v2.2 row {key!r} has shape {raw.shape}; expected ({int(length)},)."
        )
    tokens = np.empty((int(length),), dtype=np.int64)
    for index, value in enumerate(raw):
        if isinstance(value, (bool, np.bool_)):
            raise DatasetLayoutBindingError(f"v2.2 row {key!r} token {index} must be an integer id.")
        try:
            numeric = int(value)
        except (TypeError, ValueError) as exc:
            raise DatasetLayoutBindingError(
                f"v2.2 row {key!r} token {index} must be an integer id."
            ) from exc
        if numeric != INVALID_DISCRETE_TOKEN:
            try:
                if float(value) != float(numeric):
                    raise DatasetLayoutBindingError(
                        f"v2.2 row {key!r} token {index} is not an exact integer id."
                    )
            except (TypeError, ValueError) as exc:
                raise DatasetLayoutBindingError(
                    f"v2.2 row {key!r} token {index} must be numeric."
                ) from exc
        tokens[index] = numeric
    return tokens


@dataclass(frozen=True)
class PolicyIOFrame:
    """One model-ready v2.2 frame with separated float and token channels."""

    action: np.ndarray
    action_discrete: np.ndarray
    observation_state: np.ndarray
    observation_state_discrete: np.ndarray
    arm_qpos_audit: np.ndarray
    layout_hash: str

    def model_dict(self) -> dict[str, np.ndarray | str]:
        return {
            "action": self.action.copy(),
            "action_discrete": self.action_discrete.copy(),
            "observation.state": self.observation_state.copy(),
            "observation.state.discrete": self.observation_state_discrete.copy(),
            "policy_io_layout_hash": self.layout_hash,
        }


@dataclass(frozen=True)
class PolicyIOBatch:
    """Stacked training batch with float features and int64 tokens kept typed."""

    action: np.ndarray
    action_discrete: np.ndarray
    observation_state: np.ndarray
    observation_state_discrete: np.ndarray
    arm_qpos_audit: np.ndarray
    layout_hash: str

    def __post_init__(self) -> None:
        action = np.asarray(self.action, dtype=np.float32)
        action_discrete = np.asarray(self.action_discrete, dtype=np.int64)
        observation_state = np.asarray(self.observation_state, dtype=np.float32)
        observation_state_discrete = np.asarray(self.observation_state_discrete, dtype=np.int64)
        arm_qpos_audit = np.asarray(self.arm_qpos_audit, dtype=np.float32)
        if action.ndim != 2 or observation_state.ndim != 2:
            raise DatasetLayoutBindingError("PolicyIOBatch float arrays must be 2D [batch, dim].")
        if action_discrete.ndim != 2 or observation_state_discrete.ndim != 2:
            raise DatasetLayoutBindingError("PolicyIOBatch token arrays must be 2D [batch, token_dim].")
        if arm_qpos_audit.ndim != 2:
            raise DatasetLayoutBindingError("PolicyIOBatch arm qpos audit must be 2D [batch, 12].")
        batch_size = action.shape[0]
        if any(
            value.shape[0] != batch_size
            for value in (action_discrete, observation_state, observation_state_discrete, arm_qpos_audit)
        ):
            raise DatasetLayoutBindingError("PolicyIOBatch arrays must have the same batch dimension.")
        object.__setattr__(self, "action", action.copy())
        object.__setattr__(self, "action_discrete", action_discrete.copy())
        object.__setattr__(self, "observation_state", observation_state.copy())
        object.__setattr__(self, "observation_state_discrete", observation_state_discrete.copy())
        object.__setattr__(self, "arm_qpos_audit", arm_qpos_audit.copy())
        object.__setattr__(self, "layout_hash", str(self.layout_hash))

    def model_dict(self) -> dict[str, np.ndarray | str]:
        return {
            "action": self.action.copy(),
            "action_discrete": self.action_discrete.copy(),
            "observation.state": self.observation_state.copy(),
            "observation.state.discrete": self.observation_state_discrete.copy(),
            "policy_io_layout_hash": self.layout_hash,
        }


def policy_io_frame_from_row(row: Mapping[str, Any], layout: PolicyIOLayout) -> PolicyIOFrame:
    """Validate a parquet row against the declared dataset/model layout."""

    row_hash = row.get("policy_io_layout_hash")
    if row_hash is not None and str(row_hash) != layout.layout_hash:
        raise DatasetLayoutBindingError(
            f"v2.2 row layout hash {row_hash!r} differs from expected {layout.layout_hash!r}."
        )
    action = _float_vector(row, "action", length=layout.action_dim)
    observation_state = _float_vector(row, "observation.state", length=layout.state_dim)
    action_discrete = _token_vector(row, "action.discrete", length=layout.action_discrete_dim)
    observation_state_discrete = _token_vector(
        row,
        "observation.state.discrete",
        length=layout.state_discrete_dim,
    )
    arm_qpos_audit = _float_vector(row, "observation.arm.qpos", length=ARM_QPOS_AUDIT_DIM)
    return PolicyIOFrame(
        action=action,
        action_discrete=action_discrete,
        observation_state=observation_state,
        observation_state_discrete=observation_state_discrete,
        arm_qpos_audit=arm_qpos_audit,
        layout_hash=layout.layout_hash,
    )


def policy_io_batch_from_rows(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    layout: PolicyIOLayout,
) -> PolicyIOBatch:
    """Build one typed training batch from v2.2 rows.

    The returned model dictionary intentionally excludes arm qpos: qpos stays
    an audit-only tensor while continuous state and discrete state tokens are
    fed through separate model inputs.
    """

    frames = [policy_io_frame_from_row(row, layout) for row in rows]
    if not frames:
        raise DatasetLayoutBindingError("Cannot build a PolicyIOBatch from zero rows.")
    return PolicyIOBatch(
        action=np.stack([frame.action for frame in frames], axis=0),
        action_discrete=np.stack([frame.action_discrete for frame in frames], axis=0),
        observation_state=np.stack([frame.observation_state for frame in frames], axis=0),
        observation_state_discrete=np.stack(
            [frame.observation_state_discrete for frame in frames], axis=0
        ),
        arm_qpos_audit=np.stack([frame.arm_qpos_audit for frame in frames], axis=0),
        layout_hash=layout.layout_hash,
    )


__all__ = [
    "PolicyIOBatch",
    "PolicyIOFrame",
    "policy_io_batch_from_rows",
    "policy_io_frame_from_row",
]

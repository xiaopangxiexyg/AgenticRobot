"""Pure contracts for causal policy-state history snapshots.

The history is sampled on the control timeline.  Model-specific normalization
and tokenization are intentionally outside this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

STATE_HISTORY_DIM = 28
STATE_HISTORY_FRAME_OFFSETS = (4, 8, 16)


def _frame_offsets(values: Sequence[int]) -> tuple[int, ...]:
    offsets = tuple(int(value) for value in values)
    if not offsets or any(value <= 0 for value in offsets):
        raise ValueError("state-history frame offsets must be positive.")
    if offsets != tuple(sorted(set(offsets))):
        raise ValueError("state-history frame offsets must be sorted and unique.")
    return offsets


@dataclass(frozen=True)
class PolicyStateHistorySpec:
    """Runtime sampling specification, independent of any checkpoint identity."""

    frame_offsets: tuple[int, ...] = STATE_HISTORY_FRAME_OFFSETS
    state_dim: int = STATE_HISTORY_DIM

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_offsets", _frame_offsets(self.frame_offsets))
        if int(self.state_dim) != STATE_HISTORY_DIM:
            raise ValueError(f"policy state history must be {STATE_HISTORY_DIM}-D.")
        object.__setattr__(self, "state_dim", int(self.state_dim))

    @property
    def capacity(self) -> int:
        return self.frame_offsets[-1] + 1


@dataclass(frozen=True)
class PolicyStateHistorySnapshot:
    """Immutable values selected relative to one request observation anchor."""

    frame_offsets: tuple[int, ...]
    states: tuple[tuple[float, ...], ...]
    ages_sec: tuple[float, ...]
    anchor_timestamp: float
    anchor_step_index: int
    continuity_segment: int

    def __post_init__(self) -> None:
        offsets = _frame_offsets(self.frame_offsets)
        object.__setattr__(self, "frame_offsets", offsets)
        states = tuple(tuple(float(value) for value in row) for row in self.states)
        if len(states) != len(offsets):
            raise ValueError("state-history rows must match frame offsets.")
        if any(len(row) != STATE_HISTORY_DIM for row in states):
            raise ValueError(f"every state-history row must be {STATE_HISTORY_DIM}-D.")
        if any(not math.isfinite(value) for row in states for value in row):
            raise ValueError("state-history rows must contain finite values.")
        object.__setattr__(self, "states", states)
        ages = tuple(float(value) for value in self.ages_sec)
        if len(ages) != len(offsets) or any(not math.isfinite(value) or value <= 0.0 for value in ages):
            raise ValueError("state-history ages must be finite positive seconds.")
        if any(left >= right for left, right in pairwise(ages)):
            raise ValueError("state-history ages must increase with frame offset.")
        object.__setattr__(self, "ages_sec", ages)
        timestamp = float(self.anchor_timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("state-history anchor timestamp must be finite.")
        object.__setattr__(self, "anchor_timestamp", timestamp)
        if int(self.anchor_step_index) < 0:
            raise ValueError("state-history anchor step index must be non-negative.")
        object.__setattr__(self, "anchor_step_index", int(self.anchor_step_index))
        if int(self.continuity_segment) < 0:
            raise ValueError("state-history continuity segment must be non-negative.")
        object.__setattr__(self, "continuity_segment", int(self.continuity_segment))


__all__ = [
    "STATE_HISTORY_DIM",
    "STATE_HISTORY_FRAME_OFFSETS",
    "PolicyStateHistorySnapshot",
    "PolicyStateHistorySpec",
]

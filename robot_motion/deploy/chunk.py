"""Shared action-chunk execution policy.

Policy servers are allowed to return a horizon longer than the amount of
motion the runtime should consume before requesting the next prediction.  The
selection rule lives here so the real and simulated deployment frontends use
the same boundary.  ``None`` intentionally preserves the historical behavior
of consuming the complete received chunk.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from typing import Sequence, TypeVar


_T = TypeVar("_T")


def normalize_action_chunk_execution_steps(value: object | None) -> int | None:
    """Normalize the optional prefix length used for one received chunk."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("action_chunk_execution_steps must be a positive integer or None.")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError(
                "action_chunk_execution_steps must be a positive integer or None."
            )
        try:
            steps = int(text, 10)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "action_chunk_execution_steps must be a positive integer or None."
            ) from exc
    else:
        try:
            # operator.index accepts Python and numpy integer scalars without
            # silently truncating a float or another fractional numeric value.
            steps = int(operator.index(value))
        except (TypeError, ValueError, OverflowError):
            try:
                numeric = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    "action_chunk_execution_steps must be a positive integer or None."
                ) from exc
            if not math.isfinite(numeric) or not numeric.is_integer():
                raise ValueError(
                    "action_chunk_execution_steps must be a positive integer or None."
                )
            steps = int(numeric)
    if steps <= 0:
        raise ValueError("action_chunk_execution_steps must be positive when enabled.")
    return steps


@dataclass(frozen=True)
class ActionChunkExecutionPolicy:
    """Select the executable prefix of each policy output chunk.

    The policy is deliberately stateless: callers can use it for immutable
    model output in either a synchronous loop or an asynchronous producer.  A
    value larger than the received horizon simply keeps the whole chunk.
    """

    execution_steps: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "execution_steps",
            normalize_action_chunk_execution_steps(self.execution_steps),
        )

    @property
    def enabled(self) -> bool:
        return self.execution_steps is not None

    def select(self, actions: Sequence[_T]) -> tuple[_T, ...]:
        """Return only the configured prefix, without mutating ``actions``."""

        values = tuple(actions)
        if self.execution_steps is None:
            return values
        return values[: self.execution_steps]


__all__ = [
    "ActionChunkExecutionPolicy",
    "normalize_action_chunk_execution_steps",
]

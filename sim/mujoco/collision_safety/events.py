"""Task-independent collision-event accounting for learning and evaluation.

The safety projector reports a counterfactual command admission decision while
the active-contact oracle reports physical contacts that actually occurred.
Keeping those signals separate prevents a trainer from treating projection
fraction as penetration depth or from charging every step of one sustained
intervention as a new collision event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class CollisionEventBatch:
    """One vector step of task-independent collision cost components."""

    preventive_active: np.ndarray
    preventive_onset: np.ndarray
    preventive_blocked_fraction: np.ndarray
    preventive_streak_steps: np.ndarray
    preventive_category: np.ndarray
    actual_forbidden_contact: np.ndarray
    actual_category: np.ndarray
    hard_violation: np.ndarray


class CollisionEventTracker:
    """Track bounded collision events across independent environment streams.

    This class owns no policy observation, task reward, or safety authority.
    It only turns already-authoritative admission/contact facts into separate
    cost components suitable for logging or a future constrained optimizer.
    """

    def __init__(self, stream_count: int) -> None:
        count = int(stream_count)
        if count <= 0:
            raise ValueError("CollisionEventTracker stream_count must be positive.")
        self._stream_count = count
        self._previous_preventive_active = np.zeros(count, dtype=bool)
        self._preventive_streak_steps = np.zeros(count, dtype=np.int32)

    @property
    def stream_count(self) -> int:
        return self._stream_count

    def reset(self, stream_ids: Sequence[int] | np.ndarray | None = None) -> None:
        if stream_ids is None:
            self._previous_preventive_active.fill(False)
            self._preventive_streak_steps.fill(0)
            return
        indices = np.asarray(stream_ids, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self._stream_count):
            raise IndexError("CollisionEventTracker reset stream id is out of range.")
        self._previous_preventive_active[indices] = False
        self._preventive_streak_steps[indices] = 0

    def observe(
        self,
        *,
        preventive_intervention: Any,
        projection_alpha: Any,
        preventive_category: Any | None = None,
        actual_forbidden_contact: Any | None = None,
        actual_category: Any | None = None,
        hard_violation: Any | None = None,
        episode_done: Any | None = None,
    ) -> CollisionEventBatch:
        active = self._bool_vector(
            preventive_intervention,
            name="preventive_intervention",
        )
        alpha = self._float_vector(projection_alpha, name="projection_alpha")
        if np.any(alpha < 0.0) or np.any(alpha > 1.0):
            raise ValueError("projection_alpha must lie in [0, 1].")
        categories = self._category_vector(
            preventive_category,
            name="preventive_category",
        )
        actual = self._optional_bool_vector(
            actual_forbidden_contact,
            name="actual_forbidden_contact",
        )
        actual_categories = self._category_vector(
            actual_category,
            name="actual_category",
        )
        hard = self._optional_bool_vector(
            hard_violation,
            name="hard_violation",
        )
        done = self._optional_bool_vector(episode_done, name="episode_done")

        onset = active & ~self._previous_preventive_active
        streak = np.where(
            active,
            self._preventive_streak_steps + 1,
            0,
        ).astype(np.int32, copy=False)
        blocked_fraction = np.where(active, 1.0 - alpha, 0.0).astype(
            np.float32,
            copy=False,
        )
        categories = np.where(active, categories, None)
        actual_categories = np.where(actual, actual_categories, None)

        self._previous_preventive_active[:] = active
        self._preventive_streak_steps[:] = streak
        if np.any(done):
            self.reset(np.flatnonzero(done))

        return CollisionEventBatch(
            preventive_active=active.copy(),
            preventive_onset=onset,
            preventive_blocked_fraction=blocked_fraction,
            preventive_streak_steps=streak.copy(),
            preventive_category=np.asarray(categories, dtype=object),
            actual_forbidden_contact=actual,
            actual_category=np.asarray(actual_categories, dtype=object),
            hard_violation=hard,
        )

    def _bool_vector(self, values: Any, *, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=bool).reshape(-1)
        if result.shape != (self._stream_count,):
            raise ValueError(
                f"{name} must have shape ({self._stream_count},), got {result.shape}."
            )
        return result

    def _optional_bool_vector(self, values: Any | None, *, name: str) -> np.ndarray:
        if values is None:
            return np.zeros(self._stream_count, dtype=bool)
        return self._bool_vector(values, name=name)

    def _float_vector(self, values: Any, *, name: str) -> np.ndarray:
        result = np.asarray(values, dtype=np.float32).reshape(-1)
        if result.shape != (self._stream_count,):
            raise ValueError(
                f"{name} must have shape ({self._stream_count},), got {result.shape}."
            )
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must be finite.")
        return result

    def _category_vector(self, values: Any | None, *, name: str) -> np.ndarray:
        if values is None:
            return np.full(self._stream_count, None, dtype=object)
        result = np.asarray(values, dtype=object).reshape(-1)
        if result.shape != (self._stream_count,):
            raise ValueError(
                f"{name} must have shape ({self._stream_count},), got {result.shape}."
            )
        return np.asarray(
            [
                None
                if value is None or not str(value).strip()
                else str(value).strip()
                for value in result
            ],
            dtype=object,
        )


__all__ = ["CollisionEventBatch", "CollisionEventTracker"]

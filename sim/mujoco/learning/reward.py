"""Generic embodiment/execution reward components.

Task progress belongs in a task adapter.  This module only measures costs
that should remain meaningful when a policy moves from MuJoCo to the real
execution path: effort, command jitter, tracking error, limit proximity,
safety intervention, stale observations, and elapsed control time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class EmbodimentRewardConfig:
    """Non-negative scales for generic execution costs."""

    action_effort_scale: float = 0.0
    action_delta_scale: float = 0.0
    tracking_error_scale: float = 0.0
    limit_proximity_scale: float = 0.0
    safety_intervention_scale: float = 0.0
    stale_observation_scale: float = 0.0
    time_cost_scale: float = 0.0

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class RewardResult:
    """Reward value with an auditable generic/task split."""

    total: float
    embodiment_terms: Mapping[str, float]
    task_terms: Mapping[str, float]

    @property
    def embodiment_total(self) -> float:
        return float(sum(self.embodiment_terms.values()))

    @property
    def task_total(self) -> float:
        return float(sum(self.task_terms.values()))


class TaskRewardAdapter(Protocol):
    """Task-owned reward adapter; it must not be used for actor observation."""

    def terms(
        self,
        previous: Mapping[str, Any],
        current: Mapping[str, Any],
    ) -> Mapping[str, float]:
        """Return named task progress/event terms for one transition."""


def _mean_square(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("reward inputs must be finite")
    return float(np.mean(np.square(array)))


def _mean_abs(value: Any) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return 0.0
    if not np.all(np.isfinite(array)):
        raise ValueError("reward inputs must be finite")
    return float(np.mean(np.abs(array)))


def compose_embodiment_reward(
    config: EmbodimentRewardConfig,
    *,
    action: Any | None = None,
    previous_action: Any | None = None,
    tracking_error: Any | None = None,
    limit_margin: Any | None = None,
    safety_intervention: float = 0.0,
    stale_observation: float = 0.0,
    dt_seconds: float = 0.0,
) -> RewardResult:
    """Compute only generic execution/embodiment costs.

    ``limit_margin`` is normalized to ``[0, 1]`` where one is comfortably
    inside limits.  All returned generic terms are non-positive costs.
    """

    terms: dict[str, float] = {}
    if action is not None:
        terms["action_effort"] = -config.action_effort_scale * _mean_square(action)
    if action is not None and previous_action is not None:
        current = np.asarray(action, dtype=np.float64)
        previous = np.asarray(previous_action, dtype=np.float64)
        if current.shape != previous.shape:
            raise ValueError("action and previous_action must have the same shape")
        terms["action_delta"] = -config.action_delta_scale * _mean_square(current - previous)
    if tracking_error is not None:
        terms["tracking_error"] = -config.tracking_error_scale * _mean_abs(tracking_error)
    if limit_margin is not None:
        margin = np.asarray(limit_margin, dtype=np.float64)
        if margin.size and (not np.all(np.isfinite(margin))):
            raise ValueError("limit_margin must be finite")
        proximity = np.square(1.0 - np.clip(margin, 0.0, 1.0))
        terms["limit_proximity"] = -config.limit_proximity_scale * float(
            np.mean(proximity) if proximity.size else 0.0
        )
    if not np.isfinite(float(safety_intervention)) or safety_intervention < 0.0:
        raise ValueError("safety_intervention must be finite and non-negative")
    if not np.isfinite(float(stale_observation)) or stale_observation < 0.0:
        raise ValueError("stale_observation must be finite and non-negative")
    if not np.isfinite(float(dt_seconds)) or dt_seconds < 0.0:
        raise ValueError("dt_seconds must be finite and non-negative")
    terms["safety_intervention"] = -config.safety_intervention_scale * float(safety_intervention)
    terms["stale_observation"] = -config.stale_observation_scale * float(stale_observation)
    terms["time_cost"] = -config.time_cost_scale * float(dt_seconds)
    return RewardResult(total=float(sum(terms.values())), embodiment_terms=terms, task_terms={})


def compose_reward(
    embodiment: RewardResult,
    *,
    task_terms: Mapping[str, float] | None = None,
) -> RewardResult:
    """Add task-owned terms without mixing them into generic diagnostics."""

    if embodiment.task_terms:
        raise ValueError("embodiment result must not already contain task terms")
    normalized: dict[str, float] = {}
    for name, value in (task_terms or {}).items():
        key = str(name).strip()
        if not key:
            raise ValueError("task reward term names must be non-empty")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"task reward term {key!r} must be finite")
        normalized[key] = numeric
    return RewardResult(
        total=float(embodiment.total + sum(normalized.values())),
        embodiment_terms=dict(embodiment.embodiment_terms),
        task_terms=normalized,
    )


__all__ = [
    "EmbodimentRewardConfig",
    "RewardResult",
    "TaskRewardAdapter",
    "compose_embodiment_reward",
    "compose_reward",
]

"""Task-neutral local action-set sampling and executability summaries.

The functions in this module do not own a policy, task reward, IK solver, or
collision model.  A task adapter supplies exact candidate outcomes after
running its normal action path.  Keeping the geometry search separate from the
adapter makes the resulting metrics reusable without turning simulator-only
facts into actor inputs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


LOCAL_EXECUTABILITY_SCHEMA = "dexteleop.sim.local_executability.v1"


@dataclass(frozen=True)
class LocalActionCandidate:
    """One bounded policy-space candidate around a proposed action."""

    label: str
    family: str
    policy_action: tuple[float, ...]
    radial_scale: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "family": self.family,
            "policy_action": list(self.policy_action),
            "radial_scale": self.radial_scale,
        }


@dataclass(frozen=True)
class LocalActionOutcome:
    """Observed result after one candidate traverses the exact action path."""

    candidate: LocalActionCandidate
    task_action: tuple[float, ...]
    ik_feasible: bool
    projector_identity: bool
    executable: bool
    projection_alpha: float
    safety_status: str | None = None
    fully_rejected: bool = False
    actual_forbidden_contact: bool = False
    hard_violation: bool = False
    objective_progress_delta: float = 0.0
    goal_distance_progress_m: float | None = None
    blocking_category: str | None = None
    blocking_body_pair: str | None = None
    termination_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            **self.candidate.to_dict(),
            "task_action": list(self.task_action),
            "ik_feasible": bool(self.ik_feasible),
            "projector_identity": bool(self.projector_identity),
            "executable": bool(self.executable),
            "projection_alpha": float(self.projection_alpha),
            "safety_status": self.safety_status,
            "fully_rejected": bool(self.fully_rejected),
            "actual_forbidden_contact": bool(self.actual_forbidden_contact),
            "hard_violation": bool(self.hard_violation),
            "objective_progress_delta": float(self.objective_progress_delta),
            "goal_distance_progress_m": self.goal_distance_progress_m,
            "blocking_category": self.blocking_category,
            "blocking_body_pair": self.blocking_body_pair,
            "termination_reason": self.termination_reason,
        }


def generate_local_action_candidates(
    policy_action: Sequence[float],
    *,
    translation_indices: Sequence[int],
    radial_scales: Sequence[float] = (0.0, 0.25, 0.5, 0.75),
    repair_radii: Sequence[float] = (0.25, 0.5),
    sphere_directions: int = 8,
) -> tuple[LocalActionCandidate, ...]:
    """Generate deterministic scale and direction-repair candidates.

    Non-translation components are preserved exactly.  This is important for
    manipulation probes: a collision repair must not silently change the
    gripper command and then attribute the result to Cartesian feasibility.
    """

    anchor = np.asarray(policy_action, dtype=np.float64).reshape(-1)
    indices = tuple(int(value) for value in translation_indices)
    if anchor.size == 0 or not np.all(np.isfinite(anchor)):
        raise ValueError("policy_action must be a non-empty finite vector.")
    if np.any(anchor < -1.0) or np.any(anchor > 1.0):
        raise ValueError("policy_action must lie in [-1, 1].")
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError("translation_indices must contain three unique indices.")
    if any(index < 0 or index >= anchor.size for index in indices):
        raise ValueError("translation_indices are outside the policy action.")
    if int(sphere_directions) < 0:
        raise ValueError("sphere_directions must be non-negative.")

    candidates: list[LocalActionCandidate] = []
    seen: set[tuple[float, ...]] = set()

    def append(
        values: np.ndarray,
        *,
        label: str,
        family: str,
        radial_scale: float | None = None,
    ) -> None:
        clipped = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
        key = tuple(float(value) for value in np.round(clipped, decimals=10))
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            LocalActionCandidate(
                label=label,
                family=family,
                policy_action=tuple(float(value) for value in clipped),
                radial_scale=radial_scale,
            )
        )

    append(anchor, label="raw", family="raw", radial_scale=1.0)
    translation = anchor[list(indices)].copy()
    for scale_value in radial_scales:
        scale = float(scale_value)
        if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
            raise ValueError("radial_scales must lie in [0, 1].")
        values = anchor.copy()
        values[list(indices)] = translation * scale
        append(
            values,
            label=f"radial_scale:{scale:.6g}",
            family="radial_scale",
            radial_scale=scale,
        )

    directions = _local_directions(int(sphere_directions))
    for radius_value in repair_radii:
        radius = float(radius_value)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError("repair_radii must contain positive finite values.")
        for direction_index, direction in enumerate(directions):
            values = anchor.copy()
            values[list(indices)] += radius * direction
            append(
                values,
                label=f"local:r={radius:.6g}:d={direction_index}",
                family="local_repair",
            )
    return tuple(candidates)


def summarize_local_outcomes(
    outcomes: Iterable[LocalActionOutcome],
    *,
    translation_indices: Sequence[int],
) -> dict[str, object]:
    """Summarize one blocked state without merging safety into task reward."""

    values = tuple(outcomes)
    if not values:
        raise ValueError("At least one local action outcome is required.")
    indices = tuple(int(value) for value in translation_indices)
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError("translation_indices must contain three unique indices.")
    raw = next((value for value in values if value.candidate.family == "raw"), None)
    if raw is None:
        raise ValueError("Local outcomes must contain the raw policy action.")

    raw_action = np.asarray(raw.candidate.policy_action, dtype=np.float64)
    executable = tuple(value for value in values if value.executable)
    nearest = min(
        executable,
        key=lambda value: _action_distance(raw_action, value.candidate.policy_action),
        default=None,
    )
    nearest_distance = (
        None
        if nearest is None
        else _action_distance(raw_action, nearest.candidate.policy_action)
    )
    correction_angle = (
        None
        if nearest is None
        else _direction_angle_deg(
            raw_action[list(indices)],
            np.asarray(nearest.candidate.policy_action, dtype=np.float64)[list(indices)],
        )
    )
    radial_safe_scales = [
        float(value.candidate.radial_scale)
        for value in executable
        if value.candidate.family in {"raw", "radial_scale"}
        and value.candidate.radial_scale is not None
    ]
    blockers = Counter(
        value.blocking_category
        for value in values
        if value.blocking_category is not None
    )
    body_pairs = Counter(
        value.blocking_body_pair
        for value in values
        if value.blocking_body_pair is not None
    )
    progressing_feasible = sum(
        value.executable and value.objective_progress_delta > 0.0
        for value in values
    )
    goal_progressing_feasible = sum(
        value.executable
        and value.goal_distance_progress_m is not None
        and value.goal_distance_progress_m > 0.0
        for value in values
    )
    return {
        "schema": LOCAL_EXECUTABILITY_SCHEMA,
        "candidate_count": len(values),
        "executable_count": len(executable),
        "executable_ratio": len(executable) / len(values),
        "raw_executable": bool(raw.executable),
        "raw_projection_alpha": float(raw.projection_alpha),
        "ik_infeasible_count": sum(not value.ik_feasible for value in values),
        "projector_modified_count": sum(
            not value.projector_identity for value in values
        ),
        "fully_rejected_count": sum(value.fully_rejected for value in values),
        "actual_forbidden_contact_count": sum(
            value.actual_forbidden_contact for value in values
        ),
        "hard_violation_count": sum(value.hard_violation for value in values),
        "progressing_feasible_count": int(progressing_feasible),
        "progressing_feasible_ratio": progressing_feasible / len(values),
        "goal_progressing_feasible_count": int(goal_progressing_feasible),
        "goal_progressing_feasible_ratio": (
            goal_progressing_feasible / len(values)
        ),
        "maximum_executable_radial_scale": (
            None if not radial_safe_scales else max(radial_safe_scales)
        ),
        "nearest_executable_label": (
            None if nearest is None else nearest.candidate.label
        ),
        "nearest_executable_policy_action": (
            None if nearest is None else list(nearest.candidate.policy_action)
        ),
        "nearest_executable_distance": nearest_distance,
        "nearest_executable_correction_angle_deg": correction_angle,
        "blocking_category_counts": dict(sorted(blockers.items())),
        "blocking_body_pair_counts": dict(sorted(body_pairs.items())),
    }


def _local_directions(sphere_directions: int) -> tuple[np.ndarray, ...]:
    axes = tuple(
        direction
        for axis in np.eye(3, dtype=np.float64)
        for direction in (axis, -axis)
    )
    if sphere_directions == 0:
        return axes
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    sphere: list[np.ndarray] = []
    for index in range(sphere_directions):
        y = 1.0 - 2.0 * ((index + 0.5) / sphere_directions)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * index
        sphere.append(
            np.asarray(
                (math.cos(theta) * radius, y, math.sin(theta) * radius),
                dtype=np.float64,
            )
        )
    return (*axes, *sphere)


def _action_distance(reference: np.ndarray, candidate: Sequence[float]) -> float:
    values = np.asarray(candidate, dtype=np.float64)
    return float(np.linalg.norm(values - reference))


def _direction_angle_deg(reference: np.ndarray, candidate: np.ndarray) -> float | None:
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    if reference_norm <= 1e-12 or candidate_norm <= 1e-12:
        return None
    cosine = float(
        np.clip(
            np.dot(reference, candidate) / (reference_norm * candidate_norm),
            -1.0,
            1.0,
        )
    )
    return float(np.rad2deg(np.arccos(cosine)))


__all__ = [
    "LOCAL_EXECUTABILITY_SCHEMA",
    "LocalActionCandidate",
    "LocalActionOutcome",
    "generate_local_action_candidates",
    "summarize_local_outcomes",
]

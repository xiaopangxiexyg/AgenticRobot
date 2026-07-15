"""Small, dependency-free primitives for composing motion safety bounds.

The helpers in this module deliberately operate on numeric vectors only.  The
callers decide how to latch a fault or hold hardware; these functions make the
underlying constraint intersection explicit so a later rate limit cannot undo
an earlier absolute bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_EPSILON = 1e-12


@dataclass(frozen=True)
class TranslationConstraintResult:
    """Result of intersecting a soft workspace with an L2 step ball."""

    value: np.ndarray | None
    target_workspace_distance_m: float
    reference_workspace_distance_m: float | None
    workspace_limited: bool
    step_limited: bool
    infeasible: bool


@dataclass(frozen=True)
class JointConstraintResult:
    """Result of intersecting hard joint limits with per-joint step bounds."""

    value: np.ndarray | None
    absolute_limited: bool
    step_limited: bool
    infeasible: bool
    infeasible_indices: tuple[int, ...] = ()


def workspace_distance_l2(
    point: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> float:
    """Return L2 distance from ``point`` to the closed axis-aligned box."""

    point_array, lower_array, upper_array = _workspace_inputs(point, lower, upper)
    return float(np.linalg.norm(point_array - np.clip(point_array, lower_array, upper_array)))


def project_to_soft_workspace(
    point: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
    *,
    margin_m: float,
) -> np.ndarray:
    """Project ``point`` onto ``box + L2_ball(margin_m)``.

    Expanding each coordinate independently would allow a corner to exceed the
    requested allowance by ``sqrt(3)``.  This projection keeps the Euclidean
    distance to the hard workspace at or below the configured soft margin.
    """

    point_array, lower_array, upper_array = _workspace_inputs(point, lower, upper)
    margin = _finite_non_negative(margin_m, name="margin_m")
    nearest_hard = np.clip(point_array, lower_array, upper_array)
    outward = point_array - nearest_hard
    distance = float(np.linalg.norm(outward))
    if distance <= margin + _EPSILON:
        return point_array.copy()
    if margin <= 0.0:
        return nearest_hard
    return nearest_hard + outward * (margin / distance)


def constrain_translation_l2(
    target: Sequence[float] | np.ndarray,
    *,
    reference: Sequence[float] | np.ndarray | None,
    workspace_lower: Sequence[float] | np.ndarray,
    workspace_upper: Sequence[float] | np.ndarray,
    workspace_soft_margin_m: float,
    max_step_m: float,
) -> TranslationConstraintResult:
    """Constrain a translation by a soft workspace and an L2 step bound.

    The workspace is the Minkowski sum of the hard box and a sphere of the
    requested soft margin.  When the previous reference is already inside that
    convex set, ordinary L2 rate limiting preserves membership.  When it is
    outside, the closest legal re-entry point is used only if it is reachable
    in one permitted step; otherwise the two constraints have no intersection.
    """

    target_array, lower, upper = _workspace_inputs(target, workspace_lower, workspace_upper)
    margin = _finite_non_negative(workspace_soft_margin_m, name="workspace_soft_margin_m")
    max_step = _finite_non_negative(max_step_m, name="max_step_m")
    target_distance = workspace_distance_l2(target_array, lower, upper)
    projected_target = project_to_soft_workspace(
        target_array,
        lower,
        upper,
        margin_m=margin,
    )
    workspace_limited = not np.allclose(projected_target, target_array, atol=_EPSILON, rtol=0.0)

    if reference is None or max_step <= 0.0:
        return TranslationConstraintResult(
            value=projected_target,
            target_workspace_distance_m=target_distance,
            reference_workspace_distance_m=None,
            workspace_limited=workspace_limited,
            step_limited=False,
            infeasible=False,
        )

    reference_array, _, _ = _workspace_inputs(reference, lower, upper)
    reference_distance = workspace_distance_l2(reference_array, lower, upper)
    if reference_distance <= margin + _EPSILON:
        delta = projected_target - reference_array
        distance = float(np.linalg.norm(delta))
        if distance <= max_step + _EPSILON:
            value = projected_target
            step_limited = False
        else:
            value = reference_array + delta * (max_step / distance)
            step_limited = True
        # The soft workspace is convex; the segment above is therefore safe.
        return TranslationConstraintResult(
            value=value,
            target_workspace_distance_m=target_distance,
            reference_workspace_distance_m=reference_distance,
            workspace_limited=workspace_limited,
            step_limited=step_limited,
            infeasible=False,
        )

    reentry = project_to_soft_workspace(
        reference_array,
        lower,
        upper,
        margin_m=margin,
    )
    reentry_distance = float(np.linalg.norm(reentry - reference_array))
    if reentry_distance > max_step + _EPSILON:
        return TranslationConstraintResult(
            value=None,
            target_workspace_distance_m=target_distance,
            reference_workspace_distance_m=reference_distance,
            workspace_limited=workspace_limited,
            step_limited=False,
            infeasible=True,
        )
    # The reference is outside the domain.  A full re-entry step is the only
    # conservative choice that is guaranteed to satisfy both constraints.
    return TranslationConstraintResult(
        value=reentry,
        target_workspace_distance_m=target_distance,
        reference_workspace_distance_m=reference_distance,
        workspace_limited=workspace_limited,
        step_limited=not np.allclose(reentry, projected_target, atol=_EPSILON, rtol=0.0),
        infeasible=False,
    )


def constrain_joint_vector(
    target: Sequence[float] | np.ndarray,
    *,
    reference: Sequence[float] | np.ndarray | None,
    lower: Sequence[float] | np.ndarray | None,
    upper: Sequence[float] | np.ndarray | None,
    max_step_rad: float | Sequence[float] | np.ndarray,
) -> JointConstraintResult:
    """Intersect hard qpos bounds with a per-joint step interval.

    A non-empty result satisfies both bounds by construction.  An empty
    interval is reported rather than emitting a value that violates either
    side; callers must fail closed.
    """

    target_array = _finite_vector(target, name="target")
    max_step_array = np.asarray(max_step_rad, dtype=float).reshape(-1)
    if max_step_array.size == 1:
        max_step_array = np.full(target_array.shape, float(max_step_array[0]), dtype=float)
    if (
        max_step_array.shape != target_array.shape
        or not np.all(np.isfinite(max_step_array))
        or np.any(max_step_array < 0.0)
    ):
        raise ValueError("max_step_rad must be finite, non-negative, and scalar or target-shaped")
    if (lower is None) != (upper is None):
        raise ValueError("lower and upper must be provided together")

    lower_array: np.ndarray | None = None
    upper_array: np.ndarray | None = None
    bounded_target = target_array.copy()
    absolute_limited = False
    if lower is not None and upper is not None:
        lower_array = _finite_vector(lower, name="lower")
        upper_array = _finite_vector(upper, name="upper")
        if lower_array.shape != target_array.shape or upper_array.shape != target_array.shape:
            raise ValueError("joint limit shapes must match target")
        if np.any(lower_array > upper_array):
            raise ValueError("joint lower bounds must not exceed upper bounds")
        bounded_target = np.clip(target_array, lower_array, upper_array)
        absolute_limited = not np.allclose(
            bounded_target,
            target_array,
            atol=_EPSILON,
            rtol=0.0,
        )

    if reference is None or np.all(max_step_array <= 0.0):
        return JointConstraintResult(
            value=bounded_target,
            absolute_limited=absolute_limited,
            step_limited=False,
            infeasible=False,
        )

    reference_array = _finite_vector(reference, name="reference")
    if reference_array.shape != target_array.shape:
        raise ValueError("target/reference joint shapes must match")
    interval_lower = reference_array - max_step_array
    interval_upper = reference_array + max_step_array
    if lower_array is not None and upper_array is not None:
        interval_lower = np.maximum(interval_lower, lower_array)
        interval_upper = np.minimum(interval_upper, upper_array)
    infeasible_mask = interval_lower > interval_upper + _EPSILON
    if np.any(infeasible_mask):
        return JointConstraintResult(
            value=None,
            absolute_limited=absolute_limited,
            step_limited=False,
            infeasible=True,
            infeasible_indices=tuple(int(index) for index in np.flatnonzero(infeasible_mask)),
        )
    value = np.clip(bounded_target, interval_lower, interval_upper)
    step_limited = not np.allclose(value, bounded_target, atol=_EPSILON, rtol=0.0)
    # Clip once more to absorb floating point roundoff at an interval endpoint.
    if lower_array is not None and upper_array is not None:
        value = np.minimum(np.maximum(value, lower_array), upper_array)
    return JointConstraintResult(
        value=value,
        absolute_limited=absolute_limited,
        step_limited=step_limited,
        infeasible=False,
    )


def _workspace_inputs(
    point: Sequence[float] | np.ndarray,
    lower: Sequence[float] | np.ndarray,
    upper: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    point_array = _finite_vector(point, name="point")
    lower_array = _finite_vector(lower, name="lower")
    upper_array = _finite_vector(upper, name="upper")
    if point_array.shape != (3,) or lower_array.shape != (3,) or upper_array.shape != (3,):
        raise ValueError("workspace point and bounds must each contain three values")
    if np.any(lower_array > upper_array):
        raise ValueError("workspace lower bounds must not exceed upper bounds")
    return point_array, lower_array, upper_array


def _finite_vector(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.copy()


def _finite_non_negative(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


__all__ = [
    "JointConstraintResult",
    "TranslationConstraintResult",
    "constrain_joint_vector",
    "constrain_translation_l2",
    "project_to_soft_workspace",
    "workspace_distance_l2",
]

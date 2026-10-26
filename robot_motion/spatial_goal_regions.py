"""Bounded, deterministic region samples for endpoint relation proposals."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import qmc

from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.spatial_goals import (
    RelationGoalCandidate,
    finite_vector,
    generate_relation_goals,
)


def _closed_interval(value: Sequence[float], *, name: str) -> tuple[float, float]:
    try:
        lower, upper = finite_vector(value, 2, name=name).tolist()
    except OverflowError as exc:
        raise ValueError(f"{name} endpoints must be finite numbers.") from exc
    if lower > upper or not math.isfinite(upper - lower):
        raise ValueError(f"{name} must be ordered with a finite width.")
    return lower, upper


def generate_relation_region_goals(
    *,
    current_tcp_pose_xyz_quat_xyzw: Sequence[float],
    anchor_xyz_m: Sequence[float],
    offset_bounds_xyz_m: Sequence[Sequence[float]],
    offset_frame: str,
    orientation_mode: str,
    tool_axis: str | None,
    axis_reference_xyz: Sequence[float] | None,
    tool_axis_rotation_bounds_rad: Sequence[float],
    max_candidates: int,
) -> tuple[RelationGoalCandidate, ...]:
    """Return up to 1..8 unique endpoint proposals from explicit closed bounds.

    The interval midpoint comes first, followed by exactly ``max_candidates - 1``
    unscrambled Halton draws over nondegenerate dimensions, ordered X, Y, Z,
    then tool-axis rotation. Fixed dimensions consume no Halton coordinates;
    an entirely fixed region returns one candidate. Duplicate samples are
    omitted without replacement draws, so the count can be below the request.
    Each draw pairs one offset with one rotation, without a Cartesian product.

    Offsets are relative to the supplied anchor, with no inferred object centre.
    Tool-frame offsets use the ORIGINAL input TCP orientation before alignment
    or tool-axis rotation. Constraints describe endpoints only, not transit
    orientation, measurements, IK, collision-free paths, or execution authority.

    Rotation bounds span at most one full turn. Returned rotations use the
    existing relation generator's periodic normalization and duplicate identity;
    they can lie outside the numeric input interval while being period aliases
    of angles inside it. Preserve mode requires a fixed zero/periodic-zero
    rotation, with the existing generator's zero tolerance.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= 8:
        raise ValueError("max_candidates must be an integer in 1..8.")
    if not isinstance(offset_bounds_xyz_m, (list, tuple)) or len(offset_bounds_xyz_m) != 3:
        raise ValueError("offset_bounds_xyz_m must contain three lower/upper pairs.")
    bounds = [
        _closed_interval(value, name=f"offset_bounds_xyz_m[{index}]")
        for index, value in enumerate(offset_bounds_xyz_m)
    ]
    rotation_bounds = _closed_interval(
        tool_axis_rotation_bounds_rad, name="tool_axis_rotation_bounds_rad"
    )
    if rotation_bounds[1] - rotation_bounds[0] > math.tau:
        raise ValueError("Tool-axis rotation bounds must span at most 2*pi.")
    if orientation_mode == "preserve" and rotation_bounds[0] != rotation_bounds[1]:
        raise ValueError("Preserve mode requires a fixed zero/periodic-zero rotation.")
    bounds.append(rotation_bounds)
    active = [index for index, (lower, upper) in enumerate(bounds) if lower != upper]
    samples = [[lower + (upper - lower) / 2 for lower, upper in bounds]]
    if active and max_candidates > 1:
        for draw in qmc.Halton(d=len(active), scramble=False).random(max_candidates - 1):
            sample = [lower for lower, _ in bounds]
            for index, unit in zip(active, draw):
                lower, upper = bounds[index]
                # Clamp only rounding at the closed interval endpoints.
                sample[index] = min(upper, max(lower, lower + (upper - lower) * float(unit)))
            samples.append(sample)

    candidates: list[RelationGoalCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for sample in samples:
        try:
            with np.errstate(over="raise", invalid="raise"):
                candidate, = generate_relation_goals(
                    current_tcp_pose_xyz_quat_xyzw=current_tcp_pose_xyz_quat_xyzw,
                    anchor_xyz_m=anchor_xyz_m,
                    offsets_xyz_m=[sample[:3]],
                    offset_frame=offset_frame,
                    orientation_mode=orientation_mode,
                    tool_axis=tool_axis,
                    axis_reference_xyz=axis_reference_xyz,
                    tool_axis_rotation_offsets_rad=[sample[3]],
                )
        except (OverflowError, FloatingPointError) as exc:
            raise ValueError("Region goal computation must remain finite without overflow.") from exc
        pose = finite_vector(candidate.goal.pose_xyz_quat_xyzw, 7, name="generated TCP pose")
        if abs(math.hypot(*pose[3:]) - 1.0) > 1e-5:
            raise ValueError("Generated TCP quaternion must have unit norm.")
        if any(not lower <= value <= upper
               for value, (lower, upper) in zip(candidate.offset_xyz_m, bounds[:3])):
            raise ValueError("Generated offset must remain inside its supplied bounds.")
        # Use the canonical turn and precision owned by generate_relation_goals.
        identity = (*candidate.offset_xyz_m, round(candidate.tool_axis_rotation_rad, 12))
        if identity not in seen:
            seen.add(identity)
            candidates.append(candidate)
    return tuple(candidates)


def generate_continuation_region_goals(
    *,
    current_tcp_pose_xyz_quat_xyzw: Sequence[float],
    anchor_xyz_m: Sequence[float],
    offset_bounds_xyz_m: Sequence[Sequence[float]],
    offset_frame: str,
    orientation_mode: str,
    tool_axis: str | None,
    axis_reference_xyz: Sequence[float] | None,
    tool_axis_rotation_bounds_rad: Sequence[float],
    max_candidates: int,
    continuation_delta_xyz_m: Sequence[float],
) -> tuple[RelationGoalCandidate, ...]:
    """Return centre, negative/positive continuation extrema, then baseline draws.

    Validate all original inputs with the unchanged region generator first.
    Project the public-reference continuation direction into nondegenerate
    offset axes, using the ORIGINAL input TCP R.T for tool-frame offsets.
    Intersect the centre line with the offset box in both signs. The new
    endpoints use the rotation midpoint and the existing relation generator.

    Zero delta or zero free-axis projection returns the exact baseline. Keep
    the existing canonical-turn/12-decimal duplicate identity and the original
    1..8 limit: extrema displace trailing Halton proposals, not increase the
    budget or draw replacement samples. Rotation values retain the original
    generator's closed-interval period aliases. These endpoint proposals do
    not establish IK, collision-free paths, or execution authority.
    """
    geometry = {
        "current_tcp_pose_xyz_quat_xyzw": current_tcp_pose_xyz_quat_xyzw,
        "anchor_xyz_m": anchor_xyz_m,
        "offset_frame": offset_frame,
        "orientation_mode": orientation_mode,
        "tool_axis": tool_axis,
        "axis_reference_xyz": axis_reference_xyz,
    }
    baseline = generate_relation_region_goals(
        **geometry,
        offset_bounds_xyz_m=offset_bounds_xyz_m,
        tool_axis_rotation_bounds_rad=tool_axis_rotation_bounds_rad,
        max_candidates=max_candidates,
    )
    try:
        delta = finite_vector(continuation_delta_xyz_m, 3, name="continuation_delta_xyz_m")
    except OverflowError as exc:
        raise ValueError("continuation_delta_xyz_m must contain three finite numbers.") from exc
    if not np.any(delta) or max_candidates == 1:
        return baseline

    if offset_frame == "tool":
        rotation = xyzquat_to_pose_matrix(current_tcp_pose_xyz_quat_xyzw)[:3, :3]
        # Scale before R.T to avoid overflow; only the direction is needed.
        delta = rotation.T @ (delta / float(np.max(np.abs(delta))))
    direction = np.array([
        float(value) if lower != upper else 0.0
        for value, (lower, upper) in zip(delta, offset_bounds_xyz_m, strict=True)
    ])
    scale = float(np.max(np.abs(direction)))
    if scale == 0.0:
        return baseline
    direction /= scale

    centre = baseline[0].offset_xyz_m
    extrema = []
    for sign in (-1.0, 1.0):
        signed = [sign * float(value) for value in direction]
        distance = min(
            ((upper - value) if component > 0.0 else (value - lower)) / abs(component)
            for value, component, (lower, upper) in zip(
                centre, signed, offset_bounds_xyz_m, strict=True
            )
            if component != 0.0
        )
        # Clamp only rounding at the supplied closed endpoints.
        extrema.append([
            min(upper, max(lower, value + distance * component))
            for value, component, (lower, upper) in zip(
                centre, signed, offset_bounds_xyz_m, strict=True
            )
        ])

    lower_turn, upper_turn = tool_axis_rotation_bounds_rad
    midpoint_turn = lower_turn + (upper_turn - lower_turn) / 2.0
    try:
        with np.errstate(over="raise", invalid="raise"):
            endpoints = generate_relation_goals(
                **geometry,
                offsets_xyz_m=extrema,
                tool_axis_rotation_offsets_rad=[midpoint_turn],
            )
    except (OverflowError, FloatingPointError) as exc:
        raise ValueError("Extrema goal computation must remain finite without overflow.") from exc
    for candidate in endpoints:
        pose = finite_vector(candidate.goal.pose_xyz_quat_xyzw, 7, name="generated TCP pose")
        if abs(math.hypot(*pose[3:]) - 1.0) > 1e-5:
            raise ValueError("Generated TCP quaternion must have unit norm.")
        if any(not lower <= value <= upper for value, (lower, upper) in zip(
                candidate.offset_xyz_m, offset_bounds_xyz_m, strict=True)):
            raise ValueError("Generated offset must remain inside its supplied bounds.")

    candidates: list[RelationGoalCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for candidate in (baseline[0], *endpoints, *baseline[1:]):
        # The original relation generator owns periodic normalization/precision.
        identity = (*candidate.offset_xyz_m, round(candidate.tool_axis_rotation_rad, 12))
        if identity not in seen:
            seen.add(identity)
            candidates.append(candidate)
            if len(candidates) == max_candidates:
                break
    return tuple(candidates)

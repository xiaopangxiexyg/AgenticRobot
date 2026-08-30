from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from .profiles import IKSafetyProfile


def local_step_scale_factors(profile: IKSafetyProfile) -> tuple[float, ...]:
    raw_scales = tuple(
        float(value) for value in getattr(profile, "local_step_scale_factors", ()) or ()
    )
    if not raw_scales:
        raw_scales = (1.0,)
    scales: list[float] = []
    for value in raw_scales:
        if not np.isfinite(value) or value <= 0.0:
            continue
        scale = max(1e-6, min(1.0, float(value)))
        if any(abs(scale - existing) < 1e-9 for existing in scales):
            continue
        scales.append(scale)
    if not scales:
        scales = [1.0]
    scales.sort(reverse=True)
    return tuple(scales)


def local_step_worsening_limits(
    *,
    profile: IKSafetyProfile,
    before_position_m: float,
    before_orientation_deg: float,
) -> dict[str, float | bool]:
    base_position_m = max(0.0, float(profile.local_step_max_position_worsen_m))
    base_orientation_deg = max(0.0, float(profile.local_step_max_orientation_worsen_deg))
    enabled = bool(getattr(profile, "local_step_dynamic_worsen_enabled", False))
    start_m = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_error_start_m", 0.05)))
    start_deg = max(0.0, float(getattr(profile, "local_step_dynamic_worsen_error_start_deg", 15.0)))
    active = bool(
        enabled
        and (
            float(before_position_m) >= start_m
            or float(before_orientation_deg) >= start_deg
        )
    )
    position_fraction = max(
        0.0, float(getattr(profile, "local_step_dynamic_worsen_position_fraction", 0.10))
    )
    orientation_fraction = max(
        0.0, float(getattr(profile, "local_step_dynamic_worsen_orientation_fraction", 0.10))
    )
    max_position_m = max(
        0.0, float(getattr(profile, "local_step_dynamic_worsen_max_position_m", 0.02))
    )
    max_orientation_deg = max(
        0.0, float(getattr(profile, "local_step_dynamic_worsen_max_orientation_deg", 8.0))
    )
    position_m = base_position_m
    orientation_deg = base_orientation_deg
    if active:
        position_m = max(base_position_m, max(0.0, before_position_m) * position_fraction)
        orientation_deg = max(
            base_orientation_deg, max(0.0, before_orientation_deg) * orientation_fraction
        )
        if max_position_m > 0.0:
            position_m = min(position_m, max_position_m)
        if max_orientation_deg > 0.0:
            orientation_deg = min(orientation_deg, max_orientation_deg)
    return {
        "position_m": float(position_m),
        "orientation_deg": float(orientation_deg),
        "base_position_m": float(base_position_m),
        "base_orientation_deg": float(base_orientation_deg),
        "enabled": bool(enabled),
        "active": bool(active),
        "position_fraction": float(position_fraction),
        "orientation_fraction": float(orientation_fraction),
    }


def prefer_tracking_candidate_pool(
    evaluated: list[Any],
    *,
    is_auxiliary_without_pose_progress: Callable[[dict[str, object]], bool],
) -> tuple[list[Any], dict[str, object]]:
    preferred = [
        item
        for item in evaluated
        if not is_auxiliary_without_pose_progress(item.metadata)
    ]
    pool = preferred if preferred else evaluated
    fallback = not preferred
    return pool, {
        "name": "prefer_tracking_over_auxiliary_posture_recovery",
        "accepted_count": len(evaluated),
        "selection_count": len(pool),
        "posture_wrist_auxiliary_deprioritized_count": (
            0 if fallback else len(evaluated) - len(preferred)
        ),
        "fallback_to_auxiliary_recovery": fallback,
    }

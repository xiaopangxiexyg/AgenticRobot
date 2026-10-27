"""Internal timing predicates shared by MuJoCo actual-contact guards."""

from __future__ import annotations

import math


def is_single_physics_step_elapsed(previous_time_s, current_time_s, timestep_s) -> bool:
    """Return true only for one elapsed, finite native physics timestep."""
    values = (previous_time_s, current_time_s, timestep_s)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in values):
        return False
    if timestep_s <= 0 or current_time_s < previous_time_s:
        return False
    tolerance = max(timestep_s * 1e-6, 1e-12)
    elapsed = current_time_s - previous_time_s
    return timestep_s - tolerance <= elapsed <= timestep_s + tolerance

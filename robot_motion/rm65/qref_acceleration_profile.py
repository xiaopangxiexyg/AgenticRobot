"""Versioned RM65 joint-acceleration envelopes for planned qref paths.

These profiles scale only the planned accepted-qref acceleration and its
derived jerk envelope.  The downstream high-rate command planner keeps its
independent hard execution envelope, so selecting a comfort profile cannot
weaken or silently retune teleoperation or the final qcmd safety boundary.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Sequence

import numpy as np


RM65_QREF_ACCELERATION_HARD_V1 = "hard_v1"
RM65_QREF_ACCELERATION_COMFORT_75_V1 = "comfort_75_v1"
RM65_QREF_ACCELERATION_COMFORT_50_V1 = "comfort_50_v1"
RM65_QREF_ACCELERATION_COMFORT_35_V1 = "comfort_35_v1"

RM65_QREF_ACCELERATION_PROFILE_SCALES = MappingProxyType(
    {
        RM65_QREF_ACCELERATION_HARD_V1: 1.0,
        RM65_QREF_ACCELERATION_COMFORT_75_V1: 0.75,
        RM65_QREF_ACCELERATION_COMFORT_50_V1: 0.50,
        RM65_QREF_ACCELERATION_COMFORT_35_V1: 0.35,
    }
)
RM65_QREF_ACCELERATION_PROFILE_CHOICES = tuple(
    RM65_QREF_ACCELERATION_PROFILE_SCALES
)


def resolve_rm65_qref_acceleration_profile(
    profile: str,
    *,
    hard_max_acceleration_rad_s2: Sequence[float],
) -> tuple[float, tuple[float, ...]]:
    """Return ``(scale, qref_limit)`` for one fixed runtime profile.

    The hard vector is supplied by the selected position execution profile or
    its explicit CLI override.  Keeping it as an input makes the rollback
    exact even when a deployment uses a station-specific hard envelope.
    """

    normalized = str(profile).strip().lower()
    try:
        scale = float(RM65_QREF_ACCELERATION_PROFILE_SCALES[normalized])
    except KeyError as exc:
        raise ValueError(
            "unsupported RM65 qref acceleration profile "
            f"{profile!r}; expected one of "
            f"{RM65_QREF_ACCELERATION_PROFILE_CHOICES!r}."
        ) from exc
    hard = np.asarray(hard_max_acceleration_rad_s2, dtype=float).reshape(-1)
    if hard.shape != (6,) or not np.all(np.isfinite(hard)) or np.any(hard <= 0.0):
        raise ValueError(
            "hard_max_acceleration_rad_s2 must contain six finite positive values."
        )
    if not math.isfinite(scale) or not 0.0 < scale <= 1.0:
        raise RuntimeError("registered RM65 qref acceleration scale is invalid.")
    return scale, tuple(float(value) for value in hard * scale)


__all__ = [
    "RM65_QREF_ACCELERATION_COMFORT_35_V1",
    "RM65_QREF_ACCELERATION_COMFORT_50_V1",
    "RM65_QREF_ACCELERATION_COMFORT_75_V1",
    "RM65_QREF_ACCELERATION_HARD_V1",
    "RM65_QREF_ACCELERATION_PROFILE_CHOICES",
    "RM65_QREF_ACCELERATION_PROFILE_SCALES",
    "resolve_rm65_qref_acceleration_profile",
]

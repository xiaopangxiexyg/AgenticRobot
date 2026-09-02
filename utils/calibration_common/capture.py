"""Stateless burst-capture helpers shared by calibration tools."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from .transforms import transform_distance


def most_common(values: list[str]) -> str:
    if not values:
        return ""
    return max(set(values), key=values.count)


def max_distance(reference: np.ndarray, values: list[np.ndarray]) -> tuple[float, float]:
    trans = 0.0
    rot = 0.0
    for value in values:
        current_trans, current_rot = transform_distance(reference, value)
        trans = max(trans, current_trans)
        rot = max(rot, current_rot)
    return float(trans), float(rot)


def mean_vector(values: list[list[float]]) -> list[float] | None:
    if not values:
        return None
    return np.mean(np.asarray(values, dtype=np.float64), axis=0).tolist()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


__all__ = ["max_distance", "mean_vector", "most_common", "now_utc"]

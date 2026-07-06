from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .constants import ARM_DOF, SIDES


def normalize_side(side: str) -> str:
    value = str(side).strip().lower()
    if value not in SIDES:
        raise ValueError(f"RM65 side must be one of {SIDES}, got {side!r}.")
    return value


def normalize_sides(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"both", "dual", "all"}:
            requested = SIDES
        else:
            requested = tuple(part.strip().lower() for part in text.split(",") if part.strip())
    else:
        requested = tuple(str(part).strip().lower() for part in value)
    normalized = tuple(side for side in SIDES if side in {normalize_side(side) for side in requested})
    if not normalized:
        raise ValueError("RM65 side selection must include left, right, or both.")
    return normalized


def as_rm65_qpos(values: Any, *, label: str = "qpos") -> np.ndarray:
    try:
        qpos = np.asarray(values, dtype=float).reshape(-1)
    except Exception as exc:  # noqa: BLE001 - callers get an annotated validation error.
        raise ValueError(f"{label} must be a numeric RM65 qpos vector.") from exc
    if qpos.shape != (ARM_DOF,) or not np.all(np.isfinite(qpos)):
        raise ValueError(f"{label} must contain {ARM_DOF} finite radians, got shape={qpos.shape}.")
    return qpos.astype(float, copy=True)


def validate_qpos_by_side(qpos_by_side: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(qpos_by_side, Mapping) or not qpos_by_side:
        raise ValueError("qpos_by_side must be a non-empty mapping.")
    out: dict[str, np.ndarray] = {}
    for raw_side, values in qpos_by_side.items():
        side = normalize_side(str(raw_side))
        if side in out:
            raise ValueError(f"duplicate RM65 side {side!r}.")
        out[side] = as_rm65_qpos(values, label=f"{side}.qpos")
    return out

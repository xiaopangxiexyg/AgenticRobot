"""Passive MuJoCo viewer geometry shared by TCP primitive overlays."""

from __future__ import annotations

from typing import Any

import numpy as np


def write_sphere(
    geom: Any,
    *,
    position: np.ndarray,
    radius: float,
    rgba: np.ndarray,
    label: str,
) -> None:
    import mujoco

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.full(3, float(radius), dtype=float),
        np.asarray(position, dtype=float),
        np.eye(3, dtype=float).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    geom.label = label


def write_arrow(
    geom: Any,
    *,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: np.ndarray,
    label: str,
) -> None:
    import mujoco

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        np.eye(3, dtype=float).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        float(width),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    geom.rgba = np.asarray(rgba, dtype=np.float32)
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    geom.label = label

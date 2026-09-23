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


def write_capsule(
    geom: Any,
    *,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
    label: str,
) -> None:
    """Draw a capsule, including an intentional zero-length trail segment."""

    import mujoco

    color = np.asarray(rgba, dtype=np.float32)
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3, dtype=float),
        np.zeros(3, dtype=float),
        np.eye(3, dtype=float).reshape(-1),
        color,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        float(radius),
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    geom.rgba = color
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    geom.label = label


def write_segment(
    geom: Any,
    *,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    rgba: np.ndarray,
    label: str,
) -> None:
    """Preserve the visualizers' sphere fallback for coincident endpoints."""

    start_point = np.asarray(start, dtype=float)
    end_point = np.asarray(end, dtype=float)
    if float(np.linalg.norm(end_point - start_point)) <= 1e-12:
        write_sphere(
            geom,
            position=start_point,
            radius=radius,
            rgba=rgba,
            label=label,
        )
    else:
        write_capsule(
            geom,
            start=start_point,
            end=end_point,
            radius=radius,
            rgba=rgba,
            label=label,
        )

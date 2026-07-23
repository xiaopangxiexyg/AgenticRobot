"""Passive-viewer overlay for replay tool-TCP targets.

The overlay owns only MuJoCo ``user_scn`` decoration geometry. It never reads
or writes ``MjData`` and therefore cannot affect replay control or physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from robot_motion.transforms import as_pose_matrix


_SIDES = ("left", "right")
_AXIS_COLORS_RGBA = (
    np.array([1.0, 0.12, 0.12, 1.0], dtype=np.float32),
    np.array([0.12, 1.0, 0.12, 1.0], dtype=np.float32),
    np.array([0.20, 0.45, 1.0, 1.0], dtype=np.float32),
)
_ORIGIN_COLORS_RGBA = {
    "left": np.array([1.0, 0.20, 0.80, 1.0], dtype=np.float32),
    "right": np.array([0.10, 0.90, 1.0, 1.0], dtype=np.float32),
}


@dataclass(frozen=True)
class TcpTargetVisualizationConfig:
    enabled: bool = True
    axis_length_m: float = 0.08
    axis_width_m: float = 0.003
    origin_radius_m: float = 0.008

    def __post_init__(self) -> None:
        for name in ("axis_length_m", "axis_width_m", "origin_radius_m"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "enabled", bool(self.enabled))


class TcpTargetVisualizer:
    """Draw left/right target positions and orientation axes in a passive viewer."""

    _GEOMS_PER_SIDE = 4

    def __init__(self, config: TcpTargetVisualizationConfig | None = None) -> None:
        self.config = config or TcpTargetVisualizationConfig()
        self._base_geom_index: int | None = None

    @property
    def geom_count(self) -> int:
        return len(_SIDES) * self._GEOMS_PER_SIDE

    def update(
        self,
        viewer_handle,
        target_pose_world_by_side: Mapping[str, np.ndarray],
    ) -> bool:
        """Update overlay geometry without touching the simulation model or data."""

        if not self.config.enabled or viewer_handle is None:
            return False
        if set(target_pose_world_by_side) != set(_SIDES):
            raise ValueError("TCP target visualization requires left and right poses")
        poses = {
            side: as_pose_matrix(target_pose_world_by_side[side]).copy()
            for side in _SIDES
        }
        lock = getattr(viewer_handle, "lock", None)
        if not callable(lock):
            raise TypeError("viewer_handle must expose lock()")
        with lock():
            user_scene = getattr(viewer_handle, "user_scn", None)
            if user_scene is None:
                raise TypeError("viewer_handle must expose a user_scn decoration scene")
            if self._base_geom_index is None:
                self._base_geom_index = int(user_scene.ngeom)
            stop = self._base_geom_index + self.geom_count
            if stop > int(user_scene.maxgeom):
                raise RuntimeError(
                    "MuJoCo viewer user_scn has insufficient geometry capacity for "
                    f"TCP target visualization: need={stop} max={user_scene.maxgeom}"
                )
            for side_index, side in enumerate(_SIDES):
                pose = poses[side]
                origin = pose[:3, 3]
                offset = self._base_geom_index + side_index * self._GEOMS_PER_SIDE
                self._write_origin(user_scene.geoms[offset], side=side, origin=origin)
                for axis_index in range(3):
                    endpoint = (
                        origin
                        + pose[:3, axis_index] * float(self.config.axis_length_m)
                    )
                    self._write_axis(
                        user_scene.geoms[offset + 1 + axis_index],
                        side=side,
                        axis_index=axis_index,
                        origin=origin,
                        endpoint=endpoint,
                    )
            user_scene.ngeom = max(int(user_scene.ngeom), stop)
        return True

    def _write_origin(self, geom, *, side: str, origin: np.ndarray) -> None:
        import mujoco

        radius = float(self.config.origin_radius_m)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=float),
            np.asarray(origin, dtype=float),
            np.eye(3, dtype=float).reshape(-1),
            _ORIGIN_COLORS_RGBA[side],
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = f"{side}_tcp_target"

    def _write_axis(
        self,
        geom,
        *,
        side: str,
        axis_index: int,
        origin: np.ndarray,
        endpoint: np.ndarray,
    ) -> None:
        import mujoco

        rgba = _AXIS_COLORS_RGBA[axis_index]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            np.zeros(3, dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(-1),
            rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            float(self.config.axis_width_m),
            np.asarray(origin, dtype=float),
            np.asarray(endpoint, dtype=float),
        )
        geom.rgba = rgba
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = f"{side}_tcp_target_{'xyz'[axis_index]}"


__all__ = ["TcpTargetVisualizationConfig", "TcpTargetVisualizer"]

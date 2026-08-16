"""Passive MuJoCo decoration for level TCP-orientation primitive review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_motion.primitives import LEVEL_TCP_ORIENTATION_PRIMITIVES
from robot_motion.transforms import as_pose_matrix

_SIDES = ("left", "right")
_SIDE_COLORS = {
    "left": np.array([1.0, 0.10, 0.78, 0.95], dtype=np.float32),
    "right": np.array([0.05, 0.88, 1.0, 0.95], dtype=np.float32),
}
_HEADING_COLORS = {
    "center_forward": np.array([1.0, 0.18, 0.12, 0.82], dtype=np.float32),
    "forward_left_30": np.array([0.12, 1.0, 0.18, 0.82], dtype=np.float32),
    "forward_right_30": np.array([0.16, 0.42, 1.0, 0.82], dtype=np.float32),
}
_ACTIVE_TARGET_COLOR = np.array([1.0, 0.88, 0.04, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class LevelTcpOrientationVisualizationConfig:
    anchor_radius_m: float = 0.006
    heading_arrow_length_m: float = 0.10
    heading_arrow_width_m: float = 0.0025
    current_arrow_width_m: float = 0.0035
    active_target_arrow_width_m: float = 0.0040

    def __post_init__(self) -> None:
        for name in (
            "anchor_radius_m",
            "heading_arrow_length_m",
            "heading_arrow_width_m",
            "current_arrow_width_m",
            "active_target_arrow_width_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


class LevelTcpOrientationVisualizer:
    """Draw target headings and current/active tool +Z forward directions."""

    def __init__(
        self,
        *,
        anchor_pose_world_by_side: dict[str, np.ndarray],
        world_from_desk: np.ndarray,
        actual_pose_world_by_side: dict[str, tuple[np.ndarray, ...]],
        target_pose_world_by_frame: tuple[dict[str, np.ndarray], ...],
        active_side_by_frame: tuple[str | None, ...],
        heading_by_frame: tuple[str | None, ...],
        config: LevelTcpOrientationVisualizationConfig | None = None,
    ) -> None:
        if set(anchor_pose_world_by_side) != set(_SIDES):
            raise ValueError("anchor poses must contain left and right")
        if set(actual_pose_world_by_side) != set(_SIDES):
            raise ValueError("actual poses must contain left and right")
        self.config = config or LevelTcpOrientationVisualizationConfig()
        self.anchor_pose_world_by_side = {
            side: as_pose_matrix(anchor_pose_world_by_side[side]).copy()
            for side in _SIDES
        }
        self.world_from_desk = as_pose_matrix(world_from_desk).copy()
        self.actual_pose_world_by_side = {
            side: tuple(as_pose_matrix(pose).copy() for pose in poses)
            for side, poses in actual_pose_world_by_side.items()
        }
        frame_counts = {len(self.actual_pose_world_by_side[side]) for side in _SIDES}
        if len(frame_counts) != 1:
            raise ValueError("left and right actual traces must align")
        self.frame_count = frame_counts.pop()
        if len(target_pose_world_by_frame) != self.frame_count:
            raise ValueError("target frames must align with actual traces")
        if len(active_side_by_frame) != self.frame_count:
            raise ValueError("active-side frames must align with actual traces")
        if len(heading_by_frame) != self.frame_count:
            raise ValueError("heading frames must align with actual traces")
        self.target_pose_world_by_frame = tuple(
            {side: as_pose_matrix(frame[side]).copy() for side in _SIDES}
            for frame in target_pose_world_by_frame
        )
        self.active_side_by_frame = tuple(active_side_by_frame)
        self.heading_by_frame = tuple(heading_by_frame)
        self.static_forward_endpoint_by_side_and_heading = {
            (side, primitive.heading.value): (
                self.anchor_pose_world_by_side[side][:3, 3]
                + self.config.heading_arrow_length_m
                * (
                    self.world_from_desk[:3, :3]
                    @ primitive.target_rotation_reference[:, 2]
                )
            )
            for side in _SIDES
            for primitive in LEVEL_TCP_ORIENTATION_PRIMITIVES
        }
        self._base_geom_index: int | None = None

    @property
    def geom_count(self) -> int:
        anchors = len(_SIDES)
        static_heading_arrows = len(_SIDES) * len(LEVEL_TCP_ORIENTATION_PRIMITIVES)
        current_forward_arrows = len(_SIDES)
        active_target_forward_arrow = 1
        return (
            anchors
            + static_heading_arrows
            + current_forward_arrows
            + active_target_forward_arrow
        )

    def update(self, viewer_handle: Any, *, frame_index: int) -> bool:
        if viewer_handle is None:
            return False
        index = int(frame_index)
        if isinstance(frame_index, bool) or not 0 <= index < self.frame_count:
            raise ValueError(f"frame_index must be within [0, {self.frame_count - 1}]")
        lock = getattr(viewer_handle, "lock", None)
        if not callable(lock):
            raise TypeError("viewer_handle must expose lock()")
        with lock():
            scene = getattr(viewer_handle, "user_scn", None)
            if scene is None:
                raise TypeError("viewer_handle must expose user_scn")
            if self._base_geom_index is None:
                self._base_geom_index = int(scene.ngeom)
            stop = self._base_geom_index + self.geom_count
            if stop > int(scene.maxgeom):
                raise RuntimeError(
                    "MuJoCo viewer user_scn has insufficient geometry capacity: "
                    f"need={stop} max={scene.maxgeom}"
                )
            cursor = self._base_geom_index
            for side in _SIDES:
                self._write_sphere(
                    scene.geoms[cursor],
                    position=self.anchor_pose_world_by_side[side][:3, 3],
                    radius=self.config.anchor_radius_m,
                    rgba=_SIDE_COLORS[side],
                    label=f"{side}_orientation_anchor",
                )
                cursor += 1
            for side in _SIDES:
                origin = self.anchor_pose_world_by_side[side][:3, 3]
                for primitive in LEVEL_TCP_ORIENTATION_PRIMITIVES:
                    heading = primitive.heading.value
                    self._write_arrow(
                        scene.geoms[cursor],
                        start=origin,
                        end=self.static_forward_endpoint_by_side_and_heading[
                            (side, heading)
                        ],
                        width=self.config.heading_arrow_width_m,
                        rgba=_HEADING_COLORS[heading],
                        label=f"{side}_{heading}_tool_plus_z",
                    )
                    cursor += 1
            for side in _SIDES:
                actual = self.actual_pose_world_by_side[side][index]
                self._write_arrow(
                    scene.geoms[cursor],
                    start=actual[:3, 3],
                    end=(
                        actual[:3, 3]
                        + self.config.heading_arrow_length_m * actual[:3, 2]
                    ),
                    width=self.config.current_arrow_width_m,
                    rgba=_SIDE_COLORS[side],
                    label=f"{side}_actual_tool_plus_z",
                )
                cursor += 1
            active_side = self.active_side_by_frame[index]
            heading = self.heading_by_frame[index]
            target_side = active_side if active_side in _SIDES else "left"
            target = self.target_pose_world_by_frame[index][target_side]
            target_color = (
                _ACTIVE_TARGET_COLOR
                if active_side in _SIDES and heading in _HEADING_COLORS
                else np.array([1.0, 0.88, 0.04, 0.0], dtype=np.float32)
            )
            self._write_arrow(
                scene.geoms[cursor],
                start=target[:3, 3],
                end=(
                    target[:3, 3] + self.config.heading_arrow_length_m * target[:3, 2]
                ),
                width=self.config.active_target_arrow_width_m,
                rgba=target_color,
                label=(
                    "idle_orientation_target"
                    if active_side not in _SIDES or heading is None
                    else f"active_{active_side}_{heading}_target_tool_plus_z"
                ),
            )
            cursor += 1
            scene.ngeom = max(int(scene.ngeom), stop)
            if cursor != stop:
                raise RuntimeError("Level TCP orientation geom count drifted")
        return True

    @staticmethod
    def _write_sphere(
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

    @staticmethod
    def _write_arrow(
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


__all__ = [
    "LevelTcpOrientationVisualizationConfig",
    "LevelTcpOrientationVisualizer",
]

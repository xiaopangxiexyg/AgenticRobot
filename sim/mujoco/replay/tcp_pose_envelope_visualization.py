"""Passive-viewer decoration for angular TCP-pose limits and trajectories.

This module owns only MuJoCo ``user_scn`` geometry. It never advances physics,
changes model state, performs IK, or dispatches commands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from robot_motion.transforms import as_pose_matrix


_PATH_NAMES = ("raw", "effective", "actual")
_PATH_COLORS = {
    "raw": np.array([1.0, 0.28, 0.03, 0.92], dtype=np.float32),
    "effective": np.array([0.10, 1.0, 0.22, 0.95], dtype=np.float32),
    "actual": np.array([0.92, 0.10, 1.0, 0.95], dtype=np.float32),
}
_CONE_COLORS = (
    np.array([1.0, 0.08, 0.08, 0.48], dtype=np.float32),
    np.array([0.08, 1.0, 0.08, 0.48], dtype=np.float32),
    np.array([0.12, 0.42, 1.0, 0.52], dtype=np.float32),
)
_TRANSLATION_BOX_COLOR = np.array([0.12, 0.62, 1.0, 0.10], dtype=np.float32)
_TRANSLATION_EDGE_COLOR = np.array([0.20, 0.78, 1.0, 0.72], dtype=np.float32)


@dataclass(frozen=True)
class TcpPoseEnvelopeVisualizationConfig:
    path_width_m: float = 0.0018
    pose_origin_radius_m: float = 0.006
    orientation_cone_length_m: float = 0.105
    orientation_cone_line_width_m: float = 0.00075
    orientation_direction_width_m: float = 0.0015
    translation_edge_width_m: float = 0.0014
    orientation_cone_segment_count: int = 48

    def __post_init__(self) -> None:
        for name in (
            "path_width_m",
            "pose_origin_radius_m",
            "orientation_cone_length_m",
            "orientation_cone_line_width_m",
            "orientation_direction_width_m",
            "translation_edge_width_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        segment_count = self.orientation_cone_segment_count
        if (
            isinstance(segment_count, bool)
            or int(segment_count) != segment_count
            or int(segment_count) < 6
        ):
            raise ValueError("orientation_cone_segment_count must be an integer >= 6")
        object.__setattr__(
            self,
            "orientation_cone_segment_count",
            int(segment_count),
        )


def _pose_sequence(
    values: Sequence[np.ndarray],
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    poses = tuple(as_pose_matrix(value).copy() for value in values)
    if len(poses) < 2:
        raise ValueError(f"{name} must contain at least two poses")
    return poses


class TcpPoseEnvelopeVisualizer:
    """Render orientation cones, trails, and common-origin pose directions."""

    _POSITION_GEOMS = 1
    _DIRECTION_GEOMS_PER_POSE = 3

    def __init__(
        self,
        *,
        anchor_pose_world: np.ndarray,
        max_direction_angle_rad: float,
        trajectory_pose_world_by_kind: Mapping[str, Sequence[np.ndarray]],
        translation_frame_world: np.ndarray | None = None,
        translation_min_xyz_m: Sequence[float] | None = None,
        translation_max_xyz_m: Sequence[float] | None = None,
        config: TcpPoseEnvelopeVisualizationConfig | None = None,
    ) -> None:
        self.config = config or TcpPoseEnvelopeVisualizationConfig()
        self.anchor_pose_world = as_pose_matrix(anchor_pose_world).copy()
        angle = float(max_direction_angle_rad)
        if not np.isfinite(angle) or not 0.0 < angle < np.pi:
            raise ValueError(
                "max_direction_angle_rad must be finite and within (0, pi)"
            )
        self.max_direction_angle_rad = angle
        if (translation_min_xyz_m is None) != (translation_max_xyz_m is None):
            raise ValueError(
                "translation_min_xyz_m and translation_max_xyz_m must be "
                "provided together"
            )
        self.translation_frame_world: np.ndarray | None = None
        self.translation_min_xyz_m: np.ndarray | None = None
        self.translation_max_xyz_m: np.ndarray | None = None
        if translation_min_xyz_m is not None:
            frame = (
                self.anchor_pose_world
                if translation_frame_world is None
                else as_pose_matrix(translation_frame_world)
            )
            lower = np.asarray(translation_min_xyz_m, dtype=float)
            upper = np.asarray(translation_max_xyz_m, dtype=float)
            if (
                lower.shape != (3,)
                or upper.shape != (3,)
                or not np.all(np.isfinite(lower))
                or not np.all(np.isfinite(upper))
                or np.any(lower >= upper)
            ):
                raise ValueError(
                    "translation bounds must be finite non-empty 3D bounds"
                )
            self.translation_frame_world = frame.copy()
            self.translation_min_xyz_m = lower.copy()
            self.translation_max_xyz_m = upper.copy()
        if set(trajectory_pose_world_by_kind) != set(_PATH_NAMES):
            raise ValueError("trajectory poses must contain raw, effective, and actual")
        self.trajectory_pose_world_by_kind = {
            kind: _pose_sequence(trajectory_pose_world_by_kind[kind], name=kind)
            for kind in _PATH_NAMES
        }
        counts = {len(values) for values in self.trajectory_pose_world_by_kind.values()}
        if len(counts) != 1:
            raise ValueError("all trajectory pose sequences must have equal length")
        self.frame_count = counts.pop()
        self._base_geom_index: int | None = None

    @property
    def geom_count(self) -> int:
        translation_geoms = 13 if self.translation_frame_world is not None else 0
        trail_geoms = len(_PATH_NAMES) * (self.frame_count - 1)
        current_position_geoms = len(_PATH_NAMES) * self._POSITION_GEOMS
        current_direction_geoms = (
            len(_PATH_NAMES) * self._DIRECTION_GEOMS_PER_POSE
        )
        orientation_cone_geoms = (
            3 * 2 * int(self.config.orientation_cone_segment_count)
        )
        return (
            translation_geoms
            + orientation_cone_geoms
            + trail_geoms
            + current_position_geoms
            + current_direction_geoms
        )

    def update(self, viewer_handle, *, frame_index: int) -> bool:
        if viewer_handle is None:
            return False
        index = int(frame_index)
        if isinstance(frame_index, bool) or not 0 <= index < self.frame_count:
            raise ValueError(
                f"frame_index must be within [0, {self.frame_count - 1}]"
            )
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
            if self.translation_frame_world is not None:
                cursor = self._write_translation_envelope(scene, cursor)
            cursor = self._write_orientation_cones(scene, cursor)
            for kind in _PATH_NAMES:
                poses = self.trajectory_pose_world_by_kind[kind]
                color = _PATH_COLORS[kind]
                for start, end in zip(poses[:-1], poses[1:]):
                    self._write_segment(
                        scene.geoms[cursor],
                        start=start[:3, 3],
                        end=end[:3, 3],
                        radius=float(self.config.path_width_m),
                        rgba=color,
                        label="",
                    )
                    cursor += 1
            for kind in _PATH_NAMES:
                self._write_position(
                    scene.geoms[cursor],
                    pose=self.trajectory_pose_world_by_kind[kind][index],
                    kind=kind,
                )
                cursor += self._POSITION_GEOMS
            for kind in _PATH_NAMES:
                cursor = self._write_orientation_directions(
                    scene,
                    cursor,
                    pose=self.trajectory_pose_world_by_kind[kind][index],
                    kind=kind,
                )
            scene.ngeom = max(int(scene.ngeom), stop)
        return True

    def _write_translation_envelope(self, scene, cursor: int) -> int:
        import mujoco

        assert self.translation_frame_world is not None
        assert self.translation_min_xyz_m is not None
        assert self.translation_max_xyz_m is not None
        lower = self.translation_min_xyz_m
        upper = self.translation_max_xyz_m
        center_local = 0.5 * (lower + upper)
        half_size = 0.5 * (upper - lower)
        rotation = self.translation_frame_world[:3, :3]
        center_world = (
            self.translation_frame_world[:3, 3] + rotation @ center_local
        )
        mujoco.mjv_initGeom(
            scene.geoms[cursor],
            mujoco.mjtGeom.mjGEOM_BOX,
            half_size,
            center_world,
            rotation.reshape(-1),
            _TRANSLATION_BOX_COLOR,
        )
        scene.geoms[cursor].category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        scene.geoms[cursor].label = "tcp_relative_translation_envelope"
        cursor += 1
        corners = tuple(
            self.translation_frame_world[:3, 3]
            + rotation
            @ np.array([x, y, z], dtype=float)
            for x in (float(lower[0]), float(upper[0]))
            for y in (float(lower[1]), float(upper[1]))
            for z in (float(lower[2]), float(upper[2]))
        )
        for first, second in (
            (0, 1), (0, 2), (0, 4),
            (1, 3), (1, 5),
            (2, 3), (2, 6),
            (3, 7),
            (4, 5), (4, 6),
            (5, 7), (6, 7),
        ):
            self._write_segment(
                scene.geoms[cursor],
                start=corners[first],
                end=corners[second],
                radius=float(self.config.translation_edge_width_m),
                rgba=_TRANSLATION_EDGE_COLOR,
                label="",
            )
            cursor += 1
        return cursor

    def _write_orientation_cones(self, scene, cursor: int) -> int:
        """Draw allowed anchor-axis deflection as spherical-cap boundaries.

        The underlying safety contract is a total SO(3) rotation bound. Each
        wire fan shows its directional consequence for one anchor basis vector.
        Boundary rays have fixed length and terminate on the exact small circle
        at the configured angular distance. Unlike a planar ``tan(angle)``
        cone, this remains finite and semantically correct at and beyond 90
        degrees. Axial twist remains visible on the moving XYZ pose axes rather
        than being misrepresented as part of the spherical cap.
        """

        origin = self.anchor_pose_world[:3, 3]
        length = float(self.config.orientation_cone_length_m)
        segment_count = int(self.config.orientation_cone_segment_count)
        for axis_index in range(3):
            direction = self.anchor_pose_world[:3, axis_index]
            matrix = self._frame_with_z_axis(direction)
            circle = self._spherical_direction_boundary(
                origin=origin,
                axis_frame=matrix,
                angle_rad=self.max_direction_angle_rad,
                radius=length,
                segment_count=segment_count,
            )
            label = f"tcp_orientation_{'xyz'[axis_index]}_cone"
            for boundary_index, point in enumerate(circle):
                self._write_segment(
                    scene.geoms[cursor],
                    start=origin,
                    end=point,
                    radius=float(self.config.orientation_cone_line_width_m),
                    rgba=_CONE_COLORS[axis_index],
                    label=label if boundary_index == 0 else "",
                )
                cursor += 1
            for point, successor in zip(circle, circle[1:] + circle[:1]):
                self._write_segment(
                    scene.geoms[cursor],
                    start=point,
                    end=successor,
                    radius=float(self.config.orientation_cone_line_width_m),
                    rgba=_CONE_COLORS[axis_index],
                    label="",
                )
                cursor += 1
        return cursor

    @staticmethod
    def _spherical_direction_boundary(
        *,
        origin: np.ndarray,
        axis_frame: np.ndarray,
        angle_rad: float,
        radius: float,
        segment_count: int,
    ) -> tuple[np.ndarray, ...]:
        """Return the exact fixed-radius small circle for one axis limit."""

        center = np.asarray(origin, dtype=float)
        frame = np.asarray(axis_frame, dtype=float)
        axial = math.cos(float(angle_rad))
        transverse = math.sin(float(angle_rad))
        return tuple(
            center
            + float(radius)
            * (
                axial * frame[:, 2]
                + transverse
                * (
                    math.cos(2.0 * math.pi * index / int(segment_count))
                    * frame[:, 0]
                    + math.sin(2.0 * math.pi * index / int(segment_count))
                    * frame[:, 1]
                )
            )
            for index in range(int(segment_count))
        )

    @staticmethod
    def _frame_with_z_axis(z_axis: np.ndarray) -> np.ndarray:
        z = np.asarray(z_axis, dtype=float)
        z /= np.linalg.norm(z)
        helper = (
            np.array([0.0, 0.0, 1.0], dtype=float)
            if abs(float(z[2])) < 0.9
            else np.array([0.0, 1.0, 0.0], dtype=float)
        )
        x = np.cross(helper, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        return np.column_stack((x, y, z))

    @staticmethod
    def _write_segment(
        geom,
        *,
        start: np.ndarray,
        end: np.ndarray,
        radius: float,
        rgba: np.ndarray,
        label: str,
    ) -> None:
        import mujoco

        start_point = np.asarray(start, dtype=float)
        end_point = np.asarray(end, dtype=float)
        if float(np.linalg.norm(end_point - start_point)) <= 1e-12:
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius], dtype=float),
                start_point,
                np.eye(3, dtype=float).reshape(-1),
                rgba,
            )
        else:
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.eye(3, dtype=float).reshape(-1),
                rgba,
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                radius,
                start_point,
                end_point,
            )
            geom.rgba = rgba
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = label

    def _write_position(
        self,
        geom,
        *,
        pose: np.ndarray,
        kind: str,
    ) -> None:
        import mujoco

        origin = pose[:3, 3]
        radius = float(self.config.pose_origin_radius_m)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, radius, radius], dtype=float),
            origin,
            np.eye(3, dtype=float).reshape(-1),
            _PATH_COLORS[kind],
        )
        geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
        geom.label = f"{kind}_tcp_position"

    def _write_orientation_directions(
        self,
        scene,
        cursor: int,
        *,
        pose: np.ndarray,
        kind: str,
    ) -> int:
        origin = self.anchor_pose_world[:3, 3]
        length = float(self.config.orientation_cone_length_m)
        for axis_index in range(3):
            endpoint = origin + pose[:3, axis_index] * length
            self._write_segment(
                scene.geoms[cursor],
                start=origin,
                end=endpoint,
                radius=float(self.config.orientation_direction_width_m),
                rgba=_PATH_COLORS[kind],
                label="",
            )
            cursor += 1
        return cursor


__all__ = [
    "TcpPoseEnvelopeVisualizationConfig",
    "TcpPoseEnvelopeVisualizer",
]

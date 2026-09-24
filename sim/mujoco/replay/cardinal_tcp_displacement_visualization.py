"""Passive MuJoCo decoration for cardinal TCP primitive review.

This overlay only writes ``user_scn`` geometry.  It does not read or modify
MuJoCo state, advance physics, solve IK, or dispatch commands.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_motion.coordinate_frames import CoordinateFrameSpec
from robot_motion.coordinate_profiles import STATION_RETARGET_FRAME_CONTRACT_V1
from robot_motion.primitives import CARDINAL_TCP_SHORT_STEP_PRIMITIVES
from robot_motion.transforms import as_pose_matrix

from ._viewer_geometry import write_arrow, write_capsule, write_sphere

_SIDES = ("left", "right")
_DESK_REFERENCE_FRAME_SPEC = CoordinateFrameSpec(
    frame_id="desk",
    frame_version="1",
    semantic_parameters={
        "axis_contract": STATION_RETARGET_FRAME_CONTRACT_V1,
    },
)
_SIDE_COLORS = {
    "left": np.array([1.0, 0.10, 0.78, 0.90], dtype=np.float32),
    "right": np.array([0.05, 0.88, 1.0, 0.90], dtype=np.float32),
}
_DIRECTION_COLORS = {
    "forward": np.array([1.0, 0.12, 0.12, 0.95], dtype=np.float32),
    "backward": np.array([0.58, 0.04, 0.04, 0.95], dtype=np.float32),
    "left": np.array([0.12, 1.0, 0.12, 0.95], dtype=np.float32),
    "right": np.array([0.04, 0.55, 0.04, 0.95], dtype=np.float32),
    "up": np.array([0.16, 0.46, 1.0, 0.95], dtype=np.float32),
    "down": np.array([0.05, 0.18, 0.62, 0.95], dtype=np.float32),
}
_ACTIVE_TARGET_COLOR = np.array([1.0, 0.86, 0.05, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class CardinalTcpPrimitiveVisualizationConfig:
    anchor_radius_m: float = 0.006
    primitive_arrow_width_m: float = 0.0022
    trail_width_m: float = 0.0015
    current_tcp_radius_m: float = 0.0045
    active_target_radius_m: float = 0.006

    def __post_init__(self) -> None:
        for name in (
            "anchor_radius_m",
            "primitive_arrow_width_m",
            "trail_width_m",
            "current_tcp_radius_m",
            "active_target_radius_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)


def _pose_sequence(
    values: Sequence[np.ndarray],
    *,
    name: str,
) -> tuple[np.ndarray, ...]:
    poses = tuple(as_pose_matrix(value).copy() for value in values)
    if not poses:
        raise ValueError(f"{name} must not be empty")
    return poses


class CardinalTcpPrimitiveVisualizer:
    """Draw both six-direction target stars and physical TCP trajectories."""

    def __init__(
        self,
        *,
        anchor_pose_world_by_side: Mapping[str, np.ndarray],
        actual_pose_world_by_side: Mapping[str, Sequence[np.ndarray]],
        target_pose_world_by_frame: Sequence[Mapping[str, np.ndarray]],
        active_side_by_frame: Sequence[str | None],
        direction_by_frame: Sequence[str | None],
        world_from_desk: np.ndarray,
        reference_frame_spec: CoordinateFrameSpec = _DESK_REFERENCE_FRAME_SPEC,
        config: CardinalTcpPrimitiveVisualizationConfig | None = None,
    ) -> None:
        if set(anchor_pose_world_by_side) != set(_SIDES):
            raise ValueError("anchor poses must contain left and right")
        if set(actual_pose_world_by_side) != set(_SIDES):
            raise ValueError("actual poses must contain left and right")
        if not isinstance(reference_frame_spec, CoordinateFrameSpec):
            raise TypeError("reference_frame_spec must be a CoordinateFrameSpec")
        self.config = config or CardinalTcpPrimitiveVisualizationConfig()
        self.anchor_pose_world_by_side = {
            side: as_pose_matrix(anchor_pose_world_by_side[side]).copy()
            for side in _SIDES
        }
        self.world_from_desk = as_pose_matrix(world_from_desk).copy()
        desk_from_world = np.linalg.inv(self.world_from_desk)
        self.actual_pose_world_by_side = {
            side: _pose_sequence(
                actual_pose_world_by_side[side],
                name=f"actual_pose_world_by_side.{side}",
            )
            for side in _SIDES
        }
        counts = {
            len(self.actual_pose_world_by_side[side]) for side in _SIDES
        }
        if len(counts) != 1:
            raise ValueError("left and right actual pose traces must align")
        self.frame_count = counts.pop()
        if len(target_pose_world_by_frame) != self.frame_count:
            raise ValueError("target pose frames must align with actual traces")
        if len(active_side_by_frame) != self.frame_count:
            raise ValueError("active-side frames must align with actual traces")
        if len(direction_by_frame) != self.frame_count:
            raise ValueError("direction frames must align with actual traces")
        self.target_pose_world_by_frame = tuple(
            {
                side: as_pose_matrix(frame[side]).copy() for side in _SIDES
            }
            for frame in target_pose_world_by_frame
        )
        self.active_side_by_frame = tuple(active_side_by_frame)
        self.direction_by_frame = tuple(direction_by_frame)
        self.primitive_target_by_side_and_direction = {
            (side, primitive.direction.value): self.world_from_desk
            @ np.asarray(
                primitive.apply(
                    desk_from_world @ self.anchor_pose_world_by_side[side],
                    reference_frame_spec=reference_frame_spec,
                ).target_pose,
                dtype=float,
            )
            for side in _SIDES
            for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES
        }
        self._trail_segments_by_side = {
            side: tuple(
                (start[:3, 3].copy(), end[:3, 3].copy())
                for start, end in zip(
                    self.actual_pose_world_by_side[side][:-1],
                    self.actual_pose_world_by_side[side][1:],
                )
                if float(np.linalg.norm(end[:3, 3] - start[:3, 3])) > 1e-9
            )
            for side in _SIDES
        }
        self._base_geom_index: int | None = None

    @property
    def geom_count(self) -> int:
        anchors = len(_SIDES)
        primitive_arrows = len(_SIDES) * len(CARDINAL_TCP_SHORT_STEP_PRIMITIVES)
        trails = sum(
            len(segments) for segments in self._trail_segments_by_side.values()
        )
        current_tcp_markers = len(_SIDES)
        active_target_marker = 1
        return (
            anchors
            + primitive_arrows
            + trails
            + current_tcp_markers
            + active_target_marker
        )

    def update(self, viewer_handle: Any, *, frame_index: int) -> bool:
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
            for side in _SIDES:
                write_sphere(
                    scene.geoms[cursor],
                    position=self.anchor_pose_world_by_side[side][:3, 3],
                    radius=self.config.anchor_radius_m,
                    rgba=_SIDE_COLORS[side],
                    label=f"{side}_tcp_primitive_anchor",
                )
                cursor += 1
            for side in _SIDES:
                origin = self.anchor_pose_world_by_side[side][:3, 3]
                for primitive in CARDINAL_TCP_SHORT_STEP_PRIMITIVES:
                    direction = primitive.direction.value
                    endpoint = self.primitive_target_by_side_and_direction[
                        (side, direction)
                    ][:3, 3]
                    write_arrow(
                        scene.geoms[cursor],
                        start=origin,
                        end=endpoint,
                        width=self.config.primitive_arrow_width_m,
                        rgba=_DIRECTION_COLORS[direction],
                        label=f"{side}_{direction}_3cm",
                    )
                    cursor += 1
            for side in _SIDES:
                trail_color = _SIDE_COLORS[side].copy()
                trail_color[3] = 0.42
                for start, end in self._trail_segments_by_side[side]:
                    write_capsule(
                        scene.geoms[cursor],
                        start=start,
                        end=end,
                        radius=self.config.trail_width_m,
                        rgba=trail_color,
                        label="",
                    )
                    cursor += 1
            for side in _SIDES:
                write_sphere(
                    scene.geoms[cursor],
                    position=self.actual_pose_world_by_side[side][index][:3, 3],
                    radius=self.config.current_tcp_radius_m,
                    rgba=_SIDE_COLORS[side],
                    label=f"{side}_actual_tcp",
                )
                cursor += 1
            active_side = self.active_side_by_frame[index]
            direction = self.direction_by_frame[index]
            target_side = active_side if active_side in _SIDES else "left"
            target_color = (
                _ACTIVE_TARGET_COLOR
                if active_side in _SIDES and direction in _DIRECTION_COLORS
                else np.array([1.0, 0.86, 0.05, 0.0], dtype=np.float32)
            )
            write_sphere(
                scene.geoms[cursor],
                position=self.target_pose_world_by_frame[index][target_side][
                    :3, 3
                ],
                radius=self.config.active_target_radius_m,
                rgba=target_color,
                label=(
                    "idle_tcp_target"
                    if active_side not in _SIDES or direction is None
                    else f"active_{active_side}_{direction}_target"
                ),
            )
            cursor += 1
            scene.ngeom = max(int(scene.ngeom), stop)
            if cursor != stop:
                raise RuntimeError("Cardinal TCP visualizer geometry count drifted")
        return True

__all__ = [
    "CardinalTcpPrimitiveVisualizationConfig",
    "CardinalTcpPrimitiveVisualizer",
]

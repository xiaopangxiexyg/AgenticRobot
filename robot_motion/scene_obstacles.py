"""Immutable obstacle-scene contracts for offline motion planning.

The first scene representation is intentionally small: axis-aligned boxes in
one explicitly named reference frame.  It carries no execution authority and
does not initialize a simulator or a hardware backend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from dataset_contracts.canonical_encoding import canonical_content_hash


_AABB_HASH_DOMAIN = "dexteleop.robot_motion.axis_aligned_box.v1"
_SCENE_HASH_DOMAIN = "dexteleop.robot_motion.obstacle_scene_snapshot.v1"


def _finite_vector3(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite xyz vector with shape (3,).")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, order=True)
class AxisAlignedBoxObstacle:
    """One solid axis-aligned box expressed in the scene reference frame."""

    obstacle_id: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    semantic_content_hash: str = field(init=False, compare=False)

    def __post_init__(self) -> None:
        obstacle_id = str(self.obstacle_id)
        if not obstacle_id or obstacle_id.strip() != obstacle_id:
            raise ValueError("obstacle_id must be a non-empty trimmed string.")
        center = _finite_vector3(self.center_xyz_m, name="center_xyz_m")
        size = _finite_vector3(self.size_xyz_m, name="size_xyz_m")
        if any(item <= 0.0 for item in size):
            raise ValueError("size_xyz_m values must be positive.")
        object.__setattr__(self, "obstacle_id", obstacle_id)
        object.__setattr__(self, "center_xyz_m", center)
        object.__setattr__(self, "size_xyz_m", size)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(self.to_dict(), domain=_AABB_HASH_DOMAIN),
        )

    @property
    def minimum_xyz_m(self) -> tuple[float, float, float]:
        return tuple(center - 0.5 * size for center, size in zip(self.center_xyz_m, self.size_xyz_m))

    @property
    def maximum_xyz_m(self) -> tuple[float, float, float]:
        return tuple(center + 0.5 * size for center, size in zip(self.center_xyz_m, self.size_xyz_m))

    def to_dict(self) -> dict[str, Any]:
        return {
            "obstacle_id": self.obstacle_id,
            "center_xyz_m": list(self.center_xyz_m),
            "size_xyz_m": list(self.size_xyz_m),
        }


@dataclass(frozen=True)
class ObstacleSceneSnapshot:
    """Hash-stable static AABB scene used by one planning request."""

    reference_frame_id: str
    reference_frame_spec_hash: str
    revision: int
    obstacles: tuple[AxisAlignedBoxObstacle, ...] = ()
    minimum_clearance_m: float = 0.0
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        frame_id = str(self.reference_frame_id)
        frame_hash = str(self.reference_frame_spec_hash)
        if not frame_id or frame_id.strip() != frame_id:
            raise ValueError("reference_frame_id must be a non-empty trimmed string.")
        if not frame_hash or frame_hash.strip() != frame_hash:
            raise ValueError("reference_frame_spec_hash must be a non-empty trimmed string.")
        revision = int(self.revision)
        if revision < 0:
            raise ValueError("revision must be non-negative.")
        clearance = float(self.minimum_clearance_m)
        if not np.isfinite(clearance) or clearance < 0.0:
            raise ValueError("minimum_clearance_m must be finite and non-negative.")
        supplied_obstacles = tuple(self.obstacles)
        if any(not isinstance(item, AxisAlignedBoxObstacle) for item in supplied_obstacles):
            raise TypeError("obstacles must contain AxisAlignedBoxObstacle values.")
        obstacles = tuple(sorted(supplied_obstacles, key=lambda item: item.obstacle_id))
        obstacle_ids = tuple(item.obstacle_id for item in obstacles)
        if len(set(obstacle_ids)) != len(obstacle_ids):
            raise ValueError("obstacle_id values must be unique within a scene.")
        object.__setattr__(self, "reference_frame_id", frame_id)
        object.__setattr__(self, "reference_frame_spec_hash", frame_hash)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "obstacles", obstacles)
        object.__setattr__(self, "minimum_clearance_m", clearance)
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(self.to_dict(), domain=_SCENE_HASH_DOMAIN),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_frame_id": self.reference_frame_id,
            "reference_frame_spec_hash": self.reference_frame_spec_hash,
            "revision": self.revision,
            "minimum_clearance_m": self.minimum_clearance_m,
            "obstacles": [item.to_dict() for item in self.obstacles],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ObstacleSceneSnapshot":
        return cls(
            reference_frame_id=str(value["reference_frame_id"]),
            reference_frame_spec_hash=str(value["reference_frame_spec_hash"]),
            revision=int(value["revision"]),
            minimum_clearance_m=float(value.get("minimum_clearance_m", 0.0)),
            obstacles=tuple(
                AxisAlignedBoxObstacle(
                    obstacle_id=str(item["obstacle_id"]),
                    center_xyz_m=tuple(item["center_xyz_m"]),
                    size_xyz_m=tuple(item["size_xyz_m"]),
                )
                for item in value.get("obstacles", ())
            ),
        )


__all__ = ["AxisAlignedBoxObstacle", "ObstacleSceneSnapshot"]

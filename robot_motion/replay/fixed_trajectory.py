"""Hardware-independent rule-based fixed trajectories.

The catalog contains canonical 2D TCP centerlines.  This module only expands
one continuous path into canonical-desk pose targets; it does not own IK,
world-frame conversion, dispatch, capture, or device lifecycle.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from robot_motion.position_control import TimestampedMotionTarget
from robot_motion.relative_action import xyzquat_to_pose_matrix

FIXED_TRAJECTORY_CATALOG_SCHEMA_V1 = "agentic_robot.rule_based_fixed_trajectories.v1"
DEFAULT_FIXED_TRAJECTORY_CATALOG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "rule_based_fixed_trajectories_v1.json"
)

# The authored glyph +Y axis is called ``up``.  These four labels describe the
# direction of that axis in canonical_desk, so the mapping remains explicit
# when the same path is moved between simulation and the calibrated real desk.
FIXED_TRAJECTORY_DIRECTIONS = ("up", "right", "down", "left")
_DIRECTION_YAW_RAD = {
    "up": 0.0,
    "right": -math.pi / 2.0,
    "down": math.pi,
    "left": math.pi / 2.0,
}


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _finite_float(value: object, *, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def normalize_fixed_trajectory_direction(value: object) -> str:
    """Validate and normalize one of the four canonical desk directions."""

    direction = _required_text(value, name="direction").lower()
    if direction not in FIXED_TRAJECTORY_DIRECTIONS:
        choices = ", ".join(FIXED_TRAJECTORY_DIRECTIONS)
        raise ValueError(f"direction must be one of {choices}, got {value!r}")
    return direction


def fixed_trajectory_direction_yaw_rad(direction: object) -> float:
    """Return the canonical-desk XY yaw for a named four-way direction."""

    return float(_DIRECTION_YAW_RAD[normalize_fixed_trajectory_direction(direction)])


def _angle_delta(left: float, right: float) -> float:
    return math.atan2(math.sin(left - right), math.cos(left - right))


def _resolve_direction(
    direction: object | None,
    yaw_rad: object | None,
) -> tuple[str, float]:
    """Resolve the named direction while retaining a low-level yaw seam."""

    normalized_direction = (
        normalize_fixed_trajectory_direction(direction)
        if direction is not None
        else None
    )
    if yaw_rad is None:
        resolved_direction = normalized_direction or "up"
        return resolved_direction, fixed_trajectory_direction_yaw_rad(resolved_direction)
    if normalized_direction not in (None, "up"):
        raise ValueError("direction and yaw_rad cannot both specify a non-default orientation")
    resolved_yaw = _finite_float(yaw_rad, name="yaw_rad")
    for name, candidate in _DIRECTION_YAW_RAD.items():
        if abs(_angle_delta(resolved_yaw, candidate)) <= 1e-9:
            return name, candidate
    return "custom", resolved_yaw


@dataclass(frozen=True)
class FixedTrajectorySpec:
    """One deterministic, connected centerline in normalized local XY."""

    trajectory_id: str
    family: str
    symbol: str
    stroke_count: int
    points: tuple[tuple[float, float], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_id", _required_text(self.trajectory_id, name="trajectory_id"))
        object.__setattr__(self, "family", _required_text(self.family, name="family"))
        object.__setattr__(self, "symbol", _required_text(self.symbol, name="symbol"))
        stroke_count = int(self.stroke_count)
        if stroke_count != self.stroke_count or stroke_count <= 0:
            raise ValueError("stroke_count must be a positive integer")
        object.__setattr__(self, "stroke_count", stroke_count)
        points = tuple(
            (
                _finite_float(point[0], name="point.x"),
                _finite_float(point[1], name="point.y"),
            )
            for point in self.points
        )
        if len(points) < 2:
            raise ValueError("fixed trajectory must contain at least two points")
        if any(
            np.linalg.norm(np.subtract(right, left)) <= 1e-12
            for left, right in pairwise(points)
        ):
            raise ValueError(f"fixed trajectory {self.trajectory_id!r} contains a zero-length segment")
        height = max(point[1] for point in points) - min(point[1] for point in points)
        if height <= 1e-12:
            raise ValueError(f"fixed trajectory {self.trajectory_id!r} must have positive height")
        object.__setattr__(self, "points", points)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def bounding_height(self) -> float:
        return max(point[1] for point in self.points) - min(point[1] for point in self.points)

    @property
    def normalized_points(self) -> np.ndarray:
        points = np.asarray(self.points, dtype=float)
        return (points - points[0]) / float(self.bounding_height)


def _catalog_file_payload(catalog_path: Path) -> tuple[dict[str, Any], str]:
    raw = catalog_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"fixed trajectory catalog must be a JSON object: {catalog_path}")
    if payload.get("schema") != FIXED_TRAJECTORY_CATALOG_SCHEMA_V1:
        raise ValueError(
            f"unsupported fixed trajectory catalog schema: {payload.get('schema')!r}"
        )
    if payload.get("path_frame") != "glyph_local_xy":
        raise ValueError("fixed trajectory catalog path_frame must be glyph_local_xy")
    if payload.get("normalization") != "bounding_height_is_one":
        raise ValueError("fixed trajectory catalog normalization is unsupported")
    return payload, hashlib.sha256(raw).hexdigest()


@lru_cache(maxsize=4)
def _load_fixed_trajectory_specs(catalog_path_text: str) -> tuple[tuple[FixedTrajectorySpec, ...], str]:
    catalog_path = Path(catalog_path_text)
    payload, catalog_hash = _catalog_file_payload(catalog_path)
    raw_items = payload.get("trajectories")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("fixed trajectory catalog must contain a non-empty trajectories list")
    specs: list[FixedTrajectorySpec] = []
    seen: set[str] = set()
    for raw_item in raw_items:
        if not isinstance(raw_item, Mapping):
            raise TypeError("fixed trajectory entries must be objects")
        trajectory_id = _required_text(raw_item.get("trajectory_id"), name="trajectory_id")
        if trajectory_id in seen:
            raise ValueError(f"duplicate fixed trajectory id: {trajectory_id!r}")
        seen.add(trajectory_id)
        raw_points = raw_item.get("points")
        if not isinstance(raw_points, list):
            raise TypeError(f"fixed trajectory {trajectory_id!r} points must be a list")
        specs.append(
            FixedTrajectorySpec(
                trajectory_id=trajectory_id,
                family=raw_item.get("family", ""),
                symbol=raw_item.get("symbol", ""),
                stroke_count=raw_item.get("stroke_count", 0),
                points=tuple(tuple(point) for point in raw_points),
                metadata={
                    key: value
                    for key, value in raw_item.items()
                    if key not in {"trajectory_id", "family", "symbol", "stroke_count", "points"}
                },
            )
        )
    return tuple(specs), catalog_hash


def _catalog_path(path: str | Path | None) -> Path:
    return Path(path or DEFAULT_FIXED_TRAJECTORY_CATALOG_PATH).expanduser().resolve()


def fixed_trajectory_catalog_hash(catalog_path: str | Path | None = None) -> str:
    """Return the content hash recorded with generated targets."""

    _, catalog_hash = _load_fixed_trajectory_specs(str(_catalog_path(catalog_path)))
    return catalog_hash


def available_fixed_trajectory_ids(
    *, family: str | None = None, catalog_path: str | Path | None = None
) -> tuple[str, ...]:
    """List deterministic catalog ids, optionally restricted to a family."""

    specs, _ = _load_fixed_trajectory_specs(str(_catalog_path(catalog_path)))
    normalized_family = None if family is None else _required_text(family, name="family")
    return tuple(
        spec.trajectory_id
        for spec in specs
        if normalized_family is None or spec.family == normalized_family
    )


def get_fixed_trajectory(
    trajectory_id: str, *, catalog_path: str | Path | None = None
) -> FixedTrajectorySpec:
    """Resolve one checked-in trajectory or fail closed."""

    normalized_id = _required_text(trajectory_id, name="trajectory_id")
    specs, _ = _load_fixed_trajectory_specs(str(_catalog_path(catalog_path)))
    for spec in specs:
        if spec.trajectory_id == normalized_id:
            return spec
    available = ", ".join(spec.trajectory_id for spec in specs)
    raise KeyError(f"unknown fixed trajectory {normalized_id!r}; available: {available}")


def _boundary_poses(boundary_pose_by_side: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if not isinstance(boundary_pose_by_side, Mapping) or not boundary_pose_by_side:
        raise ValueError("boundary_pose_by_side must be a non-empty mapping")
    normalized: dict[str, np.ndarray] = {}
    for raw_side, raw_pose in boundary_pose_by_side.items():
        side = _required_text(raw_side, name="boundary side")
        pose = np.asarray(raw_pose, dtype=float).reshape(-1)
        if pose.shape == (7,):
            normalized[side] = xyzquat_to_pose_matrix(pose, name=f"boundary pose for {side!r}")
            continue
        if pose.shape != (16,) or not np.all(np.isfinite(pose)):
            raise ValueError(
                f"boundary pose for {side!r} must be a finite 4x4 matrix or xyzquat"
            )
        normalized[side] = pose.reshape(4, 4).copy()
    return normalized


def build_fixed_trajectory_targets(
    boundary_pose_by_side: Mapping[str, Any],
    trajectory_id: str,
    *,
    scale_m: float,
    linear_speed_m_s: float,
    side: str = "right",
    direction: str = "up",
    yaw_rad: float | None = None,
    catalog_path: str | Path | None = None,
) -> list[TimestampedMotionTarget]:
    """Build a one-stroke canonical-desk target sequence from an actual boundary.

    ``scale_m`` is the authored glyph bounding-box height before rotation.  The
    first centerline point is anchored exactly at the supplied side's actual
    boundary pose.  ``direction`` rotates the local +Y glyph axis into one of
    four canonical-desk directions.  ``yaw_rad`` remains a low-level
    compatibility seam for callers that need a non-cardinal angle; the MuJoCo
    CLI intentionally exposes only the four named choices.  All other sides
    are held at their supplied boundary poses.  World-frame conversion is
    intentionally left to the existing position controller.
    """

    spec = get_fixed_trajectory(trajectory_id, catalog_path=catalog_path)
    if spec.stroke_count != 1:
        raise ValueError(f"fixed trajectory {spec.trajectory_id!r} is not a one-stroke path")
    scale = _finite_float(scale_m, name="scale_m")
    speed = _finite_float(linear_speed_m_s, name="linear_speed_m_s")
    if scale <= 0.0:
        raise ValueError("scale_m must be positive")
    if speed <= 0.0:
        raise ValueError("linear_speed_m_s must be positive")
    resolved_direction, resolved_yaw = _resolve_direction(direction, yaw_rad)
    normalized_side = _required_text(side, name="side")
    boundary = _boundary_poses(boundary_pose_by_side)
    if normalized_side not in boundary:
        raise KeyError(f"boundary has no side {normalized_side!r}")

    local_xy = spec.normalized_points * scale
    cosine = math.cos(resolved_yaw)
    sine = math.sin(resolved_yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)), dtype=float)
    local_xy = local_xy @ rotation.T
    path_length_m = float(np.linalg.norm(np.diff(local_xy, axis=0), axis=1).sum())
    if path_length_m <= 1e-12:
        raise ValueError(f"fixed trajectory {spec.trajectory_id!r} has zero path length")
    duration_sec = path_length_m / speed
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(local_xy, axis=0), axis=1))))
    timestamps = duration_sec * cumulative / path_length_m
    base_position = boundary[normalized_side][:3, 3].copy()
    catalog_hash = fixed_trajectory_catalog_hash(catalog_path)
    metadata = {
        "phase": "rule_based_fixed_trajectory",
        "rule": "fixed_trajectory",
        "trajectory_id": spec.trajectory_id,
        "symbol": spec.symbol,
        "family": spec.family,
        "stroke_count": int(spec.stroke_count),
        "one_stroke_verified": True,
        "pose_frame": "canonical_desk",
        "path_frame": "glyph_local_xy",
        "direction_frame": "canonical_desk",
        "direction": resolved_direction,
        "direction_yaw_deg": math.degrees(resolved_yaw),
        "anchor": "actual_boundary_start",
        "anchor_side": normalized_side,
        "size_m": scale,
        "scale_m": scale,
        "yaw_rad": resolved_yaw,
        "linear_speed_m_s": speed,
        "path_length_m": path_length_m,
        "duration_sec": duration_sec,
        "speed_source": "explicit_linear_speed_m_s",
        "catalog_hash": catalog_hash,
    }
    targets: list[TimestampedMotionTarget] = []
    for index, (offset, timestamp) in enumerate(zip(local_xy, timestamps)):
        pose_by_side = {side_name: pose.copy() for side_name, pose in boundary.items()}
        pose_by_side[normalized_side][:3, 3] = base_position + np.asarray(
            (offset[0], offset[1], 0.0), dtype=float
        )
        targets.append(
            TimestampedMotionTarget(
                timestamp=float(timestamp),
                available_timestamp=float(timestamp),
                source_index=index,
                pose_by_side=pose_by_side,
                metadata=dict(metadata),
            )
        )
    return targets


__all__ = [
    "DEFAULT_FIXED_TRAJECTORY_CATALOG_PATH",
    "FIXED_TRAJECTORY_CATALOG_SCHEMA_V1",
    "FIXED_TRAJECTORY_DIRECTIONS",
    "FixedTrajectorySpec",
    "available_fixed_trajectory_ids",
    "build_fixed_trajectory_targets",
    "fixed_trajectory_catalog_hash",
    "fixed_trajectory_direction_yaw_rad",
    "get_fixed_trajectory",
    "normalize_fixed_trajectory_direction",
]

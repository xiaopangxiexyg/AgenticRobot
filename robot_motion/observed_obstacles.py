"""Frozen occupied RGB-D voxels; unknown space is not certified as free."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree

from dataset_contracts.canonical_encoding import canonical_content_hash


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class ObservedObstacleCloud:
    reference_frame_id: str
    points_m: np.ndarray
    evidence: Mapping[str, object]
    voxel_size_m: float = 0.004
    semantic_content_hash: str = field(init=False)

    def __post_init__(self):
        points = np.asarray(self.points_m, dtype=float)
        size = float(self.voxel_size_m)
        if (not self.reference_frame_id or points.ndim != 2 or points.shape[1] != 3
                or not len(points) or not np.isfinite(points).all()
                or not np.isfinite(size) or size <= 0):
            raise ValueError("Observed obstacles require a named frame and finite nonempty xyz points/voxel size.")
        # Keep every occupied voxel, not a random/downsampled subset. The
        # half-diagonal below conservatively covers quantization of its points.
        cells = np.ascontiguousarray(np.floor(points / size))
        # Byte-key deduplication avoids sorting three-field records for every
        # input point. Normalize signed zero, then restore numeric row order so
        # the occupied cells and their content identity remain unchanged.
        cells[cells == 0.0] = 0.0
        keys = cells.view(np.dtype((np.void, cells.dtype.itemsize * 3))).reshape(-1)
        _, first = np.unique(keys, return_index=True)
        occupied = cells[first]
        order = np.lexsort((occupied[:, 2], occupied[:, 1], occupied[:, 0]))
        centers = (occupied[order] + 0.5) * size
        centers.setflags(write=False)
        evidence = _plain(self.evidence)
        object.__setattr__(self, "points_m", centers)
        object.__setattr__(self, "evidence", _freeze(evidence))
        object.__setattr__(self, "voxel_size_m", size)
        object.__setattr__(self, "semantic_content_hash", canonical_content_hash({
            "reference_frame_id": self.reference_frame_id, "points_m": centers,
            "voxel_size_m": size, "evidence": evidence,
        }, domain="dexteleop.observed_obstacle_cloud.v1"))

    @property
    def quantization_margin_m(self) -> float:
        return float(np.sqrt(3) * self.voxel_size_m / 2)

    def evidence_payload(self):
        return _plain(self.evidence)


class ObservedBoxDistance:
    """Exact signed distance from occupied centers to sparse oriented boxes."""

    def __init__(self, cloud: ObservedObstacleCloud):
        self.cloud = cloud
        self.tree = cKDTree(cloud.points_m)

    def check_boxes(self, boxes, *, clearance_m: float):
        # Only points within this conservative circumsphere can violate the
        # requested margin. A miss proves a lower bound, not global distance.
        required = float(clearance_m) + self.cloud.quantization_margin_m
        minimum, nearest = float(np.nextafter(clearance_m, np.inf)), None
        for name, world_from_box, half_size in boxes:
            transform = np.asarray(world_from_box, dtype=float)
            half = np.asarray(half_size, dtype=float)
            candidates = self.tree.query_ball_point(
                transform[:3, 3], float(np.linalg.norm(half)) + required)
            if not candidates:
                continue
            local = (self.cloud.points_m[candidates] - transform[:3, 3]) @ transform[:3, :3]
            delta = np.abs(local) - half
            signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
            index = int(np.argmin(signed))
            distance = float(signed[index]) - self.cloud.quantization_margin_m
            if distance < minimum:
                minimum = distance
                nearest = {"geometry_name": name,
                           "occupied_center_world_m": self.cloud.points_m[candidates[index]].tolist()}
        return minimum, nearest

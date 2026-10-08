"""Remove capture-time robot observations from one aligned RGB-D frame.

The filter is deliberately separate from path collision checking.  It explains
only background-labelled depth pixels with the measured arm configuration and
the selected URDF.  Target labels are never removed, and the resulting point
cloud remains fixed while the robot moves in the planner.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from dataset_contracts.canonical_encoding import canonical_content_hash


@dataclass(frozen=True)
class RM65CapturedRobotObservationFilterConfig:
    silhouette_radius_px: float = 12.0
    depth_tolerance_m: float = 0.015
    depth_consistent_neighbor_candidates: int = 8
    finger_sweep_q_rad: tuple[float, ...] = (0.0, 0.216, 0.432, 0.648, 0.864)

    def __post_init__(self) -> None:
        radius = float(self.silhouette_radius_px)
        depth = float(self.depth_tolerance_m)
        if (
            isinstance(self.silhouette_radius_px, bool)
            or not math.isfinite(radius)
            or not 0.0 <= radius <= 16.0
        ):
            raise ValueError("silhouette_radius_px must be finite and in [0, 16]")
        if (
            isinstance(self.depth_tolerance_m, bool)
            or not math.isfinite(depth)
            or not 0.0 < depth <= 0.015
        ):
            raise ValueError("depth_tolerance_m must be finite and in (0, 0.015]")
        neighbor_candidates = self.depth_consistent_neighbor_candidates
        if (
            isinstance(neighbor_candidates, bool)
            or not isinstance(neighbor_candidates, int)
            or not 1 <= neighbor_candidates <= 32
        ):
            raise ValueError(
                "depth_consistent_neighbor_candidates must be an integer in [1, 32]"
            )
        samples = tuple(float(value) for value in self.finger_sweep_q_rad)
        if (
            not samples
            or len(samples) > 9
            or any(not math.isfinite(value) or not 0.0 <= value <= 0.864 for value in samples)
            or tuple(sorted(set(samples))) != samples
            or samples[0] != 0.0
            or samples[-1] != 0.864
        ):
            raise ValueError(
                "finger_sweep_q_rad must be 1-9 unique ascending samples from 0 to 0.864"
            )
        object.__setattr__(self, "silhouette_radius_px", radius)
        object.__setattr__(self, "depth_tolerance_m", depth)
        object.__setattr__(
            self,
            "depth_consistent_neighbor_candidates",
            neighbor_candidates,
        )
        object.__setattr__(self, "finger_sweep_q_rad", samples)

    @property
    def semantic_content_hash(self) -> str:
        return canonical_content_hash(
            asdict(self), domain="dexteleop.rm65.captured_robot_filter_config.v1"
        )


@dataclass(frozen=True)
class RM65CapturedRobotObservationFilterResult:
    depth_m: np.ndarray
    removed_mask: np.ndarray
    evidence: Mapping[str, object]

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth_m, dtype=float)
        removed = np.asarray(self.removed_mask)
        if (
            depth.ndim != 2
            or removed.shape != depth.shape
            or removed.dtype != np.bool_
            or not np.all(np.isfinite(depth))
            or np.any(depth < 0)
        ):
            raise ValueError("filtered depth and removed mask must be aligned and finite")
        frozen_depth = depth.copy()
        frozen_removed = removed.copy()
        frozen_depth.setflags(write=False)
        frozen_removed.setflags(write=False)
        object.__setattr__(self, "depth_m", frozen_depth)
        object.__setattr__(self, "removed_mask", frozen_removed)
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


def captured_robot_observation_mask(
    *,
    depth_m: np.ndarray,
    target_mask: np.ndarray,
    rendered_depth_maps_m: Sequence[np.ndarray],
    silhouette_radius_px: float,
    depth_tolerance_m: float,
    depth_consistent_neighbor_candidates: int = 8,
) -> np.ndarray:
    """Return background pixels explained by one of the rendered robot states."""

    depth = np.asarray(depth_m, dtype=float)
    target = np.asarray(target_mask)
    radius = float(silhouette_radius_px)
    tolerance = float(depth_tolerance_m)
    if (
        depth.ndim != 2
        or target.shape != depth.shape
        or target.dtype != np.bool_
        or not np.all(np.isfinite(depth))
        or np.any(depth < 0)
    ):
        raise ValueError("depth_m and target_mask must be aligned and finite")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("silhouette_radius_px must be finite and nonnegative")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("depth_tolerance_m must be finite and positive")
    neighbor_candidates = depth_consistent_neighbor_candidates
    if (
        isinstance(neighbor_candidates, bool)
        or not isinstance(neighbor_candidates, int)
        or neighbor_candidates < 1
    ):
        raise ValueError(
            "depth_consistent_neighbor_candidates must be a positive integer"
        )
    maps = tuple(np.asarray(value, dtype=float) for value in rendered_depth_maps_m)
    if not maps:
        raise ValueError("rendered_depth_maps_m must not be empty")
    remove = np.zeros(depth.shape, dtype=bool)
    for rendered in maps:
        if rendered.shape != depth.shape:
            raise ValueError("rendered robot depth must align with captured depth")
        valid = np.isfinite(rendered) & (rendered > 0)
        if np.any(np.isinf(rendered)) or np.any(
            np.isfinite(rendered) & (rendered <= 0)
        ):
            raise ValueError("rendered robot depth must contain positive meters or NaN")
        if not np.any(valid):
            continue
        pixel_distance, nearest_indices = distance_transform_edt(
            ~valid, return_indices=True
        )
        nearest_depth = rendered[nearest_indices[0], nearest_indices[1]]
        remove |= (
            (depth > 0)
            & ~target
            & (pixel_distance <= radius)
            & np.isfinite(nearest_depth)
            & (np.abs(depth - nearest_depth) <= tolerance)
        )
        if radius <= 0.0:
            continue

        # A projected silhouette may contain several robot surfaces at nearby
        # pixels.  The nearest silhouette pixel is not necessarily the surface
        # that produced the captured depth (for example where a DMTAC module
        # overlaps a finger in the image).  Search a bounded set of nearby
        # rendered samples in normalized (row, column, depth) space, then apply
        # the original exact pixel/depth limits before removing anything.  This
        # can only add depth-consistent background pixels inside the existing
        # cylinder; the finite neighbour count is deliberately conservative.
        captured_rows, captured_columns = np.nonzero(
            (depth > 0) & ~target & ~remove
        )
        if not len(captured_rows):
            continue
        rendered_rows, rendered_columns = np.nonzero(valid)
        rendered_depth = rendered[rendered_rows, rendered_columns]
        rendered_samples = np.column_stack(
            (
                rendered_rows / radius,
                rendered_columns / radius,
                rendered_depth / tolerance,
            )
        )
        captured_samples = np.column_stack(
            (
                captured_rows / radius,
                captured_columns / radius,
                depth[captured_rows, captured_columns] / tolerance,
            )
        )
        candidate_count = min(neighbor_candidates, len(rendered_samples))
        _distances, candidate_indices = cKDTree(rendered_samples).query(
            captured_samples,
            k=candidate_count,
            distance_upper_bound=math.sqrt(2.0) + 1e-12,
        )
        if candidate_count == 1:
            candidate_indices = candidate_indices[:, np.newaxis]
        valid_candidate = candidate_indices < len(rendered_samples)
        safe_indices = np.minimum(candidate_indices, len(rendered_samples) - 1)
        candidate_rows = rendered_rows[safe_indices]
        candidate_columns = rendered_columns[safe_indices]
        candidate_depth = rendered_depth[safe_indices]
        pixel_distance_squared = (
            (candidate_rows - captured_rows[:, np.newaxis]) ** 2
            + (candidate_columns - captured_columns[:, np.newaxis]) ** 2
        )
        depth_difference = np.abs(
            candidate_depth - depth[captured_rows, captured_columns, np.newaxis]
        )
        depth_consistent = np.any(
            valid_candidate
            & (pixel_distance_squared <= radius * radius)
            & (depth_difference <= tolerance),
            axis=1,
        )
        remove[captured_rows[depth_consistent], captured_columns[depth_consistent]] = (
            True
        )
    remove &= ~target
    return remove


def _is_robot_visual_mesh(name: str, *, finger_only: bool) -> bool:
    if finger_only:
        # The DMTAC modules are fixed to the moving 4B2 fingertip links.  They
        # therefore have to follow every sampled gripper state just like the
        # 4B2 CAD meshes.  Rendering them only in the first (fully-open) state
        # leaves capture-time tactile-sensor points in the static obstacle
        # cloud whenever the physical gripper is not exactly at that state.
        return "_eg2_4b2_cad_" in name or "_dm_tac_w2m_" in name
    return name.startswith(("left_", "right_"))


class RM65CapturedRobotObservationFilter:
    """Render capture-time robot CAD and remove only depth-consistent background."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        config: RM65CapturedRobotObservationFilterConfig | None = None,
    ) -> None:
        logical_path = Path(urdf_path).absolute()
        physical_path = logical_path.resolve(strict=True)
        if not physical_path.is_file():
            raise ValueError("captured-robot filter URDF must be a regular file")
        self.logical_urdf_path = logical_path
        self.physical_urdf_path = physical_path
        self.urdf_sha256 = hashlib.sha256(physical_path.read_bytes()).hexdigest()
        self.config = config or RM65CapturedRobotObservationFilterConfig()
        self._lock = threading.Lock()
        self._runtime: tuple[Any, ...] | None = None

    def _runtime_components(self) -> tuple[Any, ...]:
        if self._runtime is not None:
            return self._runtime
        try:
            import pinocchio as pin  # type: ignore
            import trimesh  # type: ignore
            from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore
        except ImportError as error:
            raise RuntimeError(
                "captured robot depth rendering requires the reviewed Pinocchio, "
                "trimesh, and Embree runtime"
            ) from error
        model, _, visual_model = pin.buildModelsFromUrdf(
            str(self.physical_urdf_path),
            package_dirs=[str(self.physical_urdf_path.parent)],
        )
        if int(model.nq) != 24:
            raise ValueError("captured-robot filter requires the reviewed 24-DoF URDF")
        selected = tuple(
            index
            for index, item in enumerate(visual_model.geometryObjects)
            if _is_robot_visual_mesh(item.name, finger_only=False)
            and hasattr(item.geometry, "vertices")
            and int(getattr(item.geometry, "num_vertices", 0)) > 0
            and int(getattr(item.geometry, "num_tris", 0)) > 0
        )
        if not selected or not any(
            "_eg2_4b2_cad_" in visual_model.geometryObjects[index].name
            for index in selected
        ):
            raise ValueError("selected URDF lacks the reviewed RM65 gripper CAD")
        self._runtime = (
            pin,
            trimesh,
            RayMeshIntersector,
            model,
            visual_model,
            selected,
        )
        return self._runtime

    @staticmethod
    def _validated_pose(value: object, *, field_name: str) -> np.ndarray:
        pose = np.asarray(value, dtype=float)
        if (
            pose.shape != (4, 4)
            or not np.all(np.isfinite(pose))
            or not np.allclose(pose[3], (0, 0, 0, 1), atol=1e-9)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1e-6)
        ):
            raise ValueError(f"{field_name} must be one rigid 4x4 pose")
        return pose.copy()

    def _render_depth_maps(
        self,
        *,
        depth_m: np.ndarray,
        intrinsics: Mapping[str, float],
        source_from_camera: np.ndarray,
        capture_qpos_by_side: Mapping[str, np.ndarray],
    ) -> tuple[tuple[np.ndarray, ...], dict[str, object]]:
        pin, trimesh, intersector_type, model, visual_model, selected = (
            self._runtime_components()
        )
        if set(capture_qpos_by_side) != {"left", "right"}:
            raise ValueError("capture qpos must contain exactly left and right")
        arm_qpos = {
            side: np.asarray(capture_qpos_by_side[side], dtype=float)
            for side in ("left", "right")
        }
        if any(
            value.shape != (6,) or not np.all(np.isfinite(value))
            for value in arm_qpos.values()
        ):
            raise ValueError("capture qpos must contain six finite radians per arm")
        fx, fy, cx, cy = (
            float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy")
        )
        if not all(math.isfinite(value) for value in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
            raise ValueError("invalid capture camera intrinsics")
        source_from_camera = self._validated_pose(
            source_from_camera, field_name="source_from_camera"
        )
        camera_from_source = np.linalg.inv(source_from_camera)
        rows, columns = np.nonzero(depth_m > 0)
        directions = np.column_stack(
            ((columns - cx) / fx, (rows - cy) / fy, np.ones(len(rows)))
        )
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = np.zeros_like(directions)
        q = np.concatenate(
            (arm_qpos["left"], np.zeros(6), arm_qpos["right"], np.zeros(6))
        )
        data = model.createData()
        visual_data = pin.GeometryData(visual_model)
        rendered_depth_maps: list[np.ndarray] = []
        triangle_counts: list[int] = []
        geometry_counts: list[int] = []
        for sample_index, finger_q_rad in enumerate(self.config.finger_sweep_q_rad):
            sample_q = q.copy()
            sample_q[6:12] = finger_q_rad
            sample_q[18:24] = finger_q_rad
            pin.forwardKinematics(model, data, sample_q)
            pin.updateFramePlacements(model, data)
            pin.updateGeometryPlacements(
                model, data, visual_model, visual_data, sample_q
            )
            vertices_parts: list[np.ndarray] = []
            faces_parts: list[np.ndarray] = []
            vertex_offset = 0
            geometry_count = 0
            for geometry_index in selected:
                item = visual_model.geometryObjects[geometry_index]
                if sample_index and not _is_robot_visual_mesh(
                    item.name, finger_only=True
                ):
                    continue
                shape = item.geometry
                local_vertices = np.asarray(shape.vertices(), dtype=float)
                source_from_geometry = visual_data.oMg[geometry_index]
                source_vertices = (
                    local_vertices @ np.asarray(source_from_geometry.rotation).T
                    + np.asarray(source_from_geometry.translation)
                )
                camera_vertices = (
                    source_vertices @ camera_from_source[:3, :3].T
                    + camera_from_source[:3, 3]
                )
                faces = np.asarray(
                    [
                        (
                            int(shape.tri_indices(index)[0]),
                            int(shape.tri_indices(index)[1]),
                            int(shape.tri_indices(index)[2]),
                        )
                        for index in range(int(shape.num_tris))
                    ],
                    dtype=np.int64,
                )
                vertices_parts.append(camera_vertices)
                faces_parts.append(faces + vertex_offset)
                vertex_offset += len(local_vertices)
                geometry_count += 1
            if not vertices_parts:
                raise RuntimeError("captured-robot renderer selected no mesh geometry")
            vertices = np.concatenate(vertices_parts, axis=0)
            faces = np.concatenate(faces_parts, axis=0)
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            locations, ray_indices, _ = intersector_type(mesh).intersects_location(
                origins, directions, multiple_hits=False
            )
            rendered = np.full(depth_m.shape, np.nan, dtype=float)
            rendered[rows[ray_indices], columns[ray_indices]] = locations[:, 2]
            rendered_depth_maps.append(rendered)
            triangle_counts.append(len(faces))
            geometry_counts.append(geometry_count)
        return tuple(rendered_depth_maps), {
            "rendered_depth_map_count": len(rendered_depth_maps),
            "rendered_geometry_count_by_state": tuple(geometry_counts),
            "rendered_triangle_count_by_state": tuple(triangle_counts),
        }

    def __call__(
        self,
        *,
        depth_m: np.ndarray,
        target_mask: np.ndarray,
        intrinsics: Mapping[str, float],
        source_from_camera: np.ndarray,
        capture_qpos_by_side: Mapping[str, np.ndarray],
    ) -> RM65CapturedRobotObservationFilterResult:
        depth = np.asarray(depth_m, dtype=float)
        target = np.asarray(target_mask)
        if (
            depth.ndim != 2
            or target.shape != depth.shape
            or target.dtype != np.bool_
            or not np.all(np.isfinite(depth))
            or np.any(depth < 0)
            or not np.any(depth > 0)
        ):
            raise ValueError("captured depth and target mask must be aligned and finite")
        with self._lock:
            rendered, render_evidence = self._render_depth_maps(
                depth_m=depth,
                intrinsics=intrinsics,
                source_from_camera=source_from_camera,
                capture_qpos_by_side=capture_qpos_by_side,
            )
        remove = captured_robot_observation_mask(
            depth_m=depth,
            target_mask=target,
            rendered_depth_maps_m=rendered,
            silhouette_radius_px=self.config.silhouette_radius_px,
            depth_tolerance_m=self.config.depth_tolerance_m,
            depth_consistent_neighbor_candidates=(
                self.config.depth_consistent_neighbor_candidates
            ),
        )
        if np.any(remove & target):
            raise RuntimeError("captured-robot filter attempted to remove target pixels")
        filtered = depth.copy()
        filtered[remove] = 0.0
        semantic_evidence = {
            "schema": "dexteleop.rm65.captured_robot_observation_filter.v1",
            "method": (
                "urdf_cad_depth_finger_sweep_depth_consistent_neighborhood_v2"
            ),
            "urdf_sha256": self.urdf_sha256,
            "filter_config_hash": self.config.semantic_content_hash,
            "silhouette_radius_px": self.config.silhouette_radius_px,
            "depth_tolerance_m": self.config.depth_tolerance_m,
            "depth_consistent_neighbor_candidates": (
                self.config.depth_consistent_neighbor_candidates
            ),
            "finger_sweep_q_rad": self.config.finger_sweep_q_rad,
            "rendered_depth_map_count": render_evidence["rendered_depth_map_count"],
            "rendered_geometry_count_by_state": render_evidence[
                "rendered_geometry_count_by_state"
            ],
            "rendered_triangle_count_by_state": render_evidence[
                "rendered_triangle_count_by_state"
            ],
            "removed_background_points": int(np.count_nonzero(remove)),
            "removed_target_points": 0,
            "target_depth_points_before": int(np.count_nonzero(target & (depth > 0))),
            "target_depth_points_after": int(np.count_nonzero(target & (filtered > 0))),
        }
        semantic_evidence["semantic_content_hash"] = canonical_content_hash(
            semantic_evidence,
            domain="dexteleop.rm65.captured_robot_observation_filter_result.v1",
        )
        return RM65CapturedRobotObservationFilterResult(
            depth_m=filtered,
            removed_mask=remove,
            evidence=semantic_evidence,
        )


__all__ = [
    "RM65CapturedRobotObservationFilter",
    "RM65CapturedRobotObservationFilterConfig",
    "RM65CapturedRobotObservationFilterResult",
    "captured_robot_observation_mask",
]

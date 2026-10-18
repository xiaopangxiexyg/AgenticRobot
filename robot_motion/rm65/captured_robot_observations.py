"""Remove capture-time robot observations from one aligned RGB-D frame.

The filter is deliberately separate from path collision checking.  It explains
only background-labelled depth pixels with the measured arm configuration and
the selected URDF.  Target labels are never removed, and the resulting point
cloud remains fixed while the robot moves in the planner.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.pose import pose6_to_matrix
from robot_motion.rm65.scene_profiles import HandProxyBoxSpec, load_dmtac_self_observation_proxy_boxes


@dataclass(frozen=True)
class RM65CapturedRobotObservationFilterConfig:
    silhouette_radius_px: float = 12.0
    depth_tolerance_m: float = 0.015
    depth_consistent_neighbor_candidates: int = 8
    finger_sweep_q_rad: tuple[float, ...] = (0.0, 0.216, 0.432, 0.648, 0.864)
    geometry_mode: str = "legacy"
    tool_surface_bounds_m: tuple[tuple[float, float], ...] | None = None
    tool_surface_normalized_squared_limit: float = 1.05
    # Larger capture-only shells belong to the explicitly selected policy.
    body_surface_tolerance_m: float = 0.0

    @classmethod
    def from_json_file(cls, path: str | Path):
        """Load explicitly selected declarative filter parameters."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("captured-robot filter config must be a JSON object")
        return cls(**payload)

    @classmethod
    def for_urdf(cls, urdf_path, *, has_self_envelope: bool):
        """Explicitly select the extended policy, retaining CAD for unknown accessories."""
        if not isinstance(has_self_envelope, bool):
            raise TypeError("has_self_envelope must be bool")
        links = ET.parse(urdf_path).getroot().findall('link')
        names = [link.attrib['name'] + '_0' for link in links
                 if link.find('visual/geometry/mesh') is not None
                 and link.attrib['name'].startswith(('left_', 'right_'))]
        supported = bool(names) and all(_has_primitive_rule(name) for name in names)
        return cls(geometry_mode='link_primitives'
                   if has_self_envelope and supported else 'cad_surface',
                   depth_tolerance_m=0.025,
                   tool_surface_bounds_m=((-0.045, 0.045), (-0.050, 0.050), (-0.050, 0.050)),
                   body_surface_tolerance_m=0.050)

    def __post_init__(self) -> None:
        if self.geometry_mode not in ("legacy", "cad_surface", "link_primitives"):
            raise ValueError("geometry_mode must be legacy, cad_surface or link_primitives")
        radius = float(self.silhouette_radius_px)
        depth = float(self.depth_tolerance_m)
        depth_limit = 0.015 if self.geometry_mode == "legacy" else 0.030
        if (
            isinstance(self.silhouette_radius_px, bool)
            or not math.isfinite(radius)
            or not 0.0 <= radius <= 16.0
        ):
            raise ValueError("silhouette_radius_px must be finite and in [0, 16]")
        if (
            isinstance(self.depth_tolerance_m, bool)
            or not math.isfinite(depth)
            or not 0.0 < depth <= depth_limit
        ):
            raise ValueError(f"depth_tolerance_m must be finite and in (0, {depth_limit}]")
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
        if self.tool_surface_bounds_m is not None:
            bounds = np.asarray(self.tool_surface_bounds_m, dtype=float)
            if (bounds.shape != (3, 2) or not np.all(np.isfinite(bounds))
                    or np.any(bounds[:, 0] >= 0) or np.any(bounds[:, 1] <= 0)
                    or np.any(np.abs(bounds) > 0.050)):
                raise ValueError("tool_surface_bounds_m requires signed XYZ bounds within 50 mm")
            object.__setattr__(self, "tool_surface_bounds_m", tuple(map(tuple, bounds.tolist())))
        limit = float(self.tool_surface_normalized_squared_limit)
        if (isinstance(self.tool_surface_normalized_squared_limit, bool)
                or not math.isfinite(limit) or not 1.0 <= limit <= 1.10):
            raise ValueError("tool_surface_normalized_squared_limit must be finite and in [1.0, 1.10]")
        object.__setattr__(self, "tool_surface_normalized_squared_limit", limit)
        body_tolerance = float(self.body_surface_tolerance_m)
        if (isinstance(self.body_surface_tolerance_m, bool)
                or not math.isfinite(body_tolerance) or not 0 <= body_tolerance <= 0.050):
            raise ValueError("body_surface_tolerance_m must be finite and in [0, 0.050]")
        object.__setattr__(self, "body_surface_tolerance_m", body_tolerance)
        if self.geometry_mode == "legacy" and (
            self.tool_surface_bounds_m is not None or body_tolerance != 0.0 or limit != 1.05
        ):
            raise ValueError("extended surface rules require an explicit geometry_mode")

    @property
    def semantic_content_hash(self) -> str:
        payload = asdict(self)
        if self.geometry_mode == "legacy":
            payload = {name: payload[name] for name in (
                "silhouette_radius_px", "depth_tolerance_m",
                "depth_consistent_neighbor_candidates", "finger_sweep_q_rad",
            )}
        return canonical_content_hash(
            payload, domain="dexteleop.rm65.captured_robot_filter_config.v1"
        )


def validate_observation_urdf_candidate(
    reference_urdf_path: str | Path, observation_urdf_path: str | Path,
) -> Path:
    """Require the observation clone to preserve the bound physical model."""
    reference = Path(reference_urdf_path).resolve(strict=True)
    observation = Path(observation_urdf_path).resolve(strict=True)
    base, candidate = ET.parse(reference).getroot(), ET.parse(observation).getroot()
    original, actual = list(base), list(candidate)
    if base.tag != candidate.tag or base.attrib != candidate.attrib or len(actual) < len(original):
        raise ValueError("observation URDF must preserve the bound physical model")
    for root, parent in ((base, reference.parent), (candidate, observation.parent)):
        for mesh in root.iter("mesh"):
            filename = mesh.attrib["filename"]
            if "://" in filename:
                raise ValueError("observation clone requires resolved filesystem mesh paths")
            mesh.set("filename", str((parent / filename).resolve(strict=True)))
    if any(ET.tostring(a) != ET.tostring(b) for a, b in zip(original, actual, strict=False)):
        raise ValueError("observation URDF changed an original link, joint, or collision element")
    base_links = {link.attrib["name"] for link in base.findall("link")}
    base_joints = {joint.attrib["name"] for joint in base.findall("joint")}
    added_links, added_children, added_joint_names = set(), set(), set()
    for element in actual[len(original):]:
        if element.tag == "link":
            name = element.attrib["name"]
            if (not _is_observation_attachment(name) or name in base_links or name in added_links
                    or element.find("collision") is not None or element.find("inertial") is not None
                    or element.find("visual/geometry/mesh") is None):
                raise ValueError("observation additions must be unique visual-only attachment meshes")
            added_links.add(name)
        elif element.tag == "joint":
            parent, child = element.find("parent"), element.find("child")
            name = element.attrib["name"]
            if (element.attrib.get("type") != "fixed" or parent is None or child is None
                    or parent.attrib.get("link") not in base_links
                    or name in base_joints or name in added_joint_names
                    or child.attrib.get("link") in added_children):
                raise ValueError("observation attachments require unique fixed joints to original links")
            added_joint_names.add(name)
            added_children.add(child.attrib.get("link"))
        else:
            raise ValueError("observation URDF adds an unsupported physical-model element")
    if added_links != added_children:
        raise ValueError("each observation attachment needs exactly one fixed parent")
    return observation


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
    extended_policy: bool = False,
) -> np.ndarray:
    """Return background pixels explained by one of the rendered robot states."""

    if not isinstance(extended_policy, bool):
        raise TypeError("extended_policy must be bool")
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
        # cylinder. Legacy keeps finite-k conservative; the explicit extended
        # policy also checks unresolved neighborhoods inside that cylinder.
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
        tree = cKDTree(rendered_samples)
        _distances, candidate_indices = tree.query(
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
        if extended_policy:
            # A finite k-nearest set in a sphere can miss a valid point in the
            # pixel/depth cylinder: slightly-too-deep pixels may rank ahead of a
            # depth-consistent silhouette edge. Query the remaining nonempty
            # neighborhoods exactly, without enlarging either configured bound.
            unresolved = np.flatnonzero(~depth_consistent & np.any(valid_candidate, axis=1))
            neighborhoods = tree.query_ball_point(captured_samples[unresolved], math.sqrt(2.0) + 1e-12)
            for sample_index, neighbors in zip(unresolved, neighborhoods):
                indices = np.asarray(neighbors, dtype=int)
                if not len(indices):
                    continue
                pixel_squared = ((rendered_rows[indices] - captured_rows[sample_index]) ** 2
                    + (rendered_columns[indices] - captured_columns[sample_index]) ** 2)
                depth_consistent[sample_index] = np.any((pixel_squared <= radius * radius)
                    & (np.abs(rendered_depth[indices] - depth[captured_rows[sample_index], captured_columns[sample_index]]) <= tolerance))
        remove[captured_rows[depth_consistent], captured_columns[depth_consistent]] = (
            True
        )
    remove &= ~target
    return remove


def _is_robot_visual_mesh(
    name: str, *, finger_only: bool, include_observation_attachments: bool = True,
) -> bool:
    if finger_only:
        # The DMTAC modules are fixed to the moving 4B2 fingertip links.  They
        # therefore have to follow every sampled gripper state just like the
        # 4B2 CAD meshes.  Rendering them only in the first (fully-open) state
        # leaves capture-time tactile-sensor points in the static obstacle
        # cloud whenever the physical gripper is not exactly at that state.
        return ("_eg2_4b2_cad_" in name or "_dm_tac_w2m_" in name
                or (include_observation_attachments and _is_observation_attachment(name)))
    return name.startswith(("left_", "right_"))


def _is_observation_attachment(name: str) -> bool:
    return name.startswith(("left_self_observation_", "right_self_observation_"))


def _camera_mesh_contains_optical_origin(name: str, camera_vertices: np.ndarray) -> bool:
    """A camera cannot observe through its own closed CAD sensor housing.

    The optical center may lie on/in that housing rather than in a modeled
    aperture. Only a named camera mesh enclosing this exact capture origin is
    omitted from self rendering; the other camera and all motion proxies stay.
    """
    return (any(token in name.lower() for token in ("realsense", "camera"))
            and np.all(np.min(camera_vertices, axis=0) <= 1e-5)
            and np.all(np.max(camera_vertices, axis=0) >= -1e-5))


def _tool_surface_side(name: str) -> str | None:
    if any(token in name for token in (
        "eg2_4b2_cad_", "dm_tac_w2m_", "legacy_gripper_housing",
        "realman_to_inspire", "realsense_d405", "arm_link6",
    )):
        for side in ("left", "right"):
            if name.startswith(side + "_"):
                return side
    return None


def _is_arm_body_surface(name: str) -> bool:
    # The flange/tool already has its own calibrated surface rule. Explicit
    # accessories keep their smaller independent ray/depth rule, not this one.
    parts = name.split("_")
    return (len(parts) >= 4 and parts[0] in ("left", "right")
            and parts[1] == "arm" and parts[2] in {f"link{i}" for i in range(6)})


def _is_arm_base_surface(name: str) -> bool:
    return name.startswith(('left_arm_base_link_', 'right_arm_base_link_'))


def _has_primitive_rule(name: str) -> bool:
    return (_is_arm_body_surface(name) or _is_arm_base_surface(name)
            or _tool_surface_side(name) is not None) and not _is_observation_attachment(name)


def _surface_samples(trimesh: Any, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices, _ = trimesh.remesh.subdivide_to_size(vertices, faces, max_edge=0.002)
    # Retain an actual triangle point per 1 mm cell, never the cell center.
    # A continuous surface is within 2 mm + sqrt(3)*1 mm of a reference;
    # this coverage error is NOT added to the removal tolerance.
    _, first = np.unique(np.floor(vertices / 0.001).astype(np.int64), axis=0, return_index=True)
    return vertices[np.sort(first)]


def _native_body_surface_membership(
    points: np.ndarray, geometry: Any, lower: np.ndarray, upper: np.ndarray, tolerance: float,
) -> tuple[np.ndarray, int]:
    """Use the original native CAD BVH, not an inflated AABB as deletion mask."""
    import hppfcl as fcl  # Same native compatibility module as the RM65 oracle.

    matched = np.zeros(len(points), dtype=bool)
    candidates = np.flatnonzero(np.all((points >= lower - tolerance)
                                      & (points <= upper + tolerance), axis=1))
    request = fcl.DistanceRequest()
    point_geometry, identity = fcl.Sphere(0.), fcl.Transform3f()
    for index in candidates:
        distance = float(fcl.distance(geometry, identity, point_geometry,
            fcl.Transform3f(np.eye(3), points[index]), request, fcl.DistanceResult()))
        if not math.isfinite(distance):
            raise RuntimeError("native captured-body surface distance is nonfinite")
        matched[index] = distance <= tolerance
    return matched, len(candidates)


def _asymmetric_surface_membership(
    observed: np.ndarray, surface: np.ndarray, bounds: np.ndarray,
    *, fast_neighbors: int = 8, normalized_squared_limit: float = 1.0,
    tree: cKDTree | None = None,
) -> np.ndarray:
    """Match actual CAD samples, not their enclosing boxes or empty interiors.

    A symmetric maximum-radius sphere encloses the signed split ellipsoid.
    Finite-k lookup is only a fast path; unresolved neighborhoods are checked
    exactly against the samples at the configured squared-norm limit. Both
    search paths expand their enclosing sphere by the same radial factor.
    """
    if not len(observed) or not len(surface):
        return np.zeros(len(observed), dtype=bool)
    extent = np.maximum(-bounds[:, 0], bounds[:, 1])
    if tree is None:
        tree = cKDTree(surface / extent)
    queries = observed / extent
    count = min(fast_neighbors, len(surface))
    search_radius = math.sqrt(normalized_squared_limit) + 1e-12
    distances, indices = tree.query(queries, k=count, distance_upper_bound=search_radius)
    if count == 1:
        distances, indices = distances[:, None], indices[:, None]
    valid = indices < len(surface)
    delta = observed[:, None] - surface[np.minimum(indices, len(surface) - 1)]
    radii = np.where(delta >= 0, bounds[:, 1], -bounds[:, 0])
    matched = np.any(valid & (np.sum((delta / radii) ** 2, axis=2) <= normalized_squared_limit + 1e-12), axis=1)
    unresolved = np.flatnonzero(~matched & np.isfinite(distances[:, 0]))
    for start in range(0, len(unresolved), 128):
        rows = unresolved[start:start + 128]
        neighborhoods = tree.query_ball_point(queries[rows], search_radius)
        for row, neighbors in zip(rows, neighborhoods):
            delta = observed[row] - surface[np.asarray(neighbors, int)]
            radii = np.where(delta >= 0, bounds[:, 1], -bounds[:, 0])
            matched[row] = np.any(np.sum((delta / radii) ** 2, axis=1) <= normalized_squared_limit + 1e-12)
    return matched


def _self_proxy_membership(points_tool: np.ndarray, tool_from_box: np.ndarray,
                           half_size: np.ndarray) -> np.ndarray:
    """Explain the captured self envelope without a second padding layer."""
    box_points = (points_tool - tool_from_box[:3, 3]) @ tool_from_box[:3, :3]
    return np.all(np.abs(box_points) <= half_size + 1e-12, axis=1)


def _projected_box_roi(camera_from_frame, lower, upper, intrinsics, image_shape):
    """Conservative pixel broad phase; a box crossing the camera plane uses the full image."""
    corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                        for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
    corners = corners @ camera_from_frame[:3, :3].T + camera_from_frame[:3, 3]
    height, width = image_shape
    x, y, z = corners.T
    planes = (z, intrinsics['fx'] * x + intrinsics['cx'] * z,
              -intrinsics['fx'] * x + (width - intrinsics['cx']) * z,
              intrinsics['fy'] * y + intrinsics['cy'] * z,
              -intrinsics['fy'] * y + (height - intrinsics['cy']) * z)
    if any(np.max(plane) < 0 for plane in planes):
        return slice(0, 0), slice(0, 0)
    if np.max(corners[:, 2]) <= 0:
        return slice(0, 0), slice(0, 0)
    if np.min(corners[:, 2]) <= 0:
        return slice(0, height), slice(0, width)
    uv = corners[:, :2] / corners[:, 2, None]
    uv = uv * [intrinsics['fx'], intrinsics['fy']] + [intrinsics['cx'], intrinsics['cy']]
    lo = np.clip(np.floor(uv.min(axis=0)) - 1, 0, [width, height]).astype(int)
    hi = np.clip(np.ceil(uv.max(axis=0)) + 2, 0, [width, height]).astype(int)
    return slice(int(lo[1]), max(int(lo[1]), int(hi[1]))), slice(int(lo[0]), max(int(lo[0]), int(hi[0])))


class RM65CapturedRobotObservationFilter:
    """Explain background with capture-time CAD and separate self envelopes."""

    def __init__(
        self,
        *,
        urdf_path: str | Path,
        config: RM65CapturedRobotObservationFilterConfig | None = None,
        tool_tcp_pose_flange_m_rad_by_side: Mapping[str, Sequence[float]] | None = None,
        self_observation_proxy_boxes: Sequence[HandProxyBoxSpec] = (),
    ) -> None:
        logical_path = Path(urdf_path).absolute()
        physical_path = logical_path.resolve(strict=True)
        if not physical_path.is_file():
            raise ValueError("captured-robot filter URDF must be a regular file")
        self.logical_urdf_path = logical_path
        self.physical_urdf_path = physical_path
        self.urdf_sha256 = hashlib.sha256(physical_path.read_bytes()).hexdigest()
        self.config = config or RM65CapturedRobotObservationFilterConfig()
        self._flange_from_tool: dict[str, np.ndarray] | None = None
        if tool_tcp_pose_flange_m_rad_by_side is not None:
            if set(tool_tcp_pose_flange_m_rad_by_side) != {"left", "right"}:
                raise ValueError("tool TCP calibration must contain exactly left and right")
            self._flange_from_tool = {}
            for side, value in tool_tcp_pose_flange_m_rad_by_side.items():
                pose = np.asarray(value, dtype=float)
                if pose.shape != (6,) or not np.all(np.isfinite(pose)):
                    raise ValueError("tool TCP calibration requires six finite meters/radians")
                self._flange_from_tool[side] = pose6_to_matrix(pose)
        if (self.config.geometry_mode != "legacy"
                and self.config.tool_surface_bounds_m is not None
                and self._flange_from_tool is None):
            raise ValueError("extended tool surfaces require explicit left/right TCP calibration")
        self._self_proxy_boxes = []
        if self_observation_proxy_boxes and (
            self.config.geometry_mode == "legacy" or self.config.tool_surface_bounds_m is None
        ):
            raise ValueError("self observation boxes require an extended policy with tool surface bounds")
        for box in self_observation_proxy_boxes:
            if not isinstance(box, HandProxyBoxSpec) or box.side not in ('left', 'right'):
                raise ValueError('Self observation requires typed left/right hand boxes')
            expected_parent = 'arm_link6' if box.side == 'left' else 'right_arm_link6'
            if box.parent_link_name != expected_parent or self._flange_from_tool is None:
                raise ValueError('Self observation boxes require flange attachment and calibrated TCP')
            half = np.asarray(box.box_size_xyz, dtype=float) / 2
            if not np.isfinite(half).all() or np.any(half <= 0):
                raise ValueError('Self observation box dimensions must be finite and positive')
            tool_from_box = np.linalg.inv(self._flange_from_tool[box.side]) @ box.parent_to_box_pose_matrix()
            if not np.isfinite(tool_from_box).all():
                raise ValueError('Self observation box placement must be finite')
            self._self_proxy_boxes.append((box, tool_from_box, half))
        if self.config.geometry_mode == "link_primitives" and (
            not self._self_proxy_boxes or self._flange_from_tool is None
            or self.config.tool_surface_bounds_m is None
        ):
            raise ValueError("link_primitives requires calibrated self boxes and explicit tool bounds")
        if self.config.geometry_mode == "link_primitives":
            for link in ET.parse(physical_path).getroot().findall('link'):
                name = link.attrib['name']
                if (name.startswith(('left_', 'right_'))
                        and link.find('visual/geometry/mesh') is not None
                        and not _has_primitive_rule(name + '_0')):
                    raise ValueError('Unrecognized self accessory requires geometry_mode=cad_surface')
        self._lock = threading.Lock()
        self._runtime: tuple[Any, ...] | None = None
        self._tool_surfaces: dict[str, dict[str, np.ndarray]] | None = None
        self._body_surfaces: dict[str, tuple[Any, np.ndarray, np.ndarray]] | None = None
        self._mesh_buffers: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._tool_queries: dict[tuple[str, tuple[str, ...]], tuple[np.ndarray, cKDTree]] = {}
        self._body_sample_trees: dict[str, tuple[cKDTree, float]] = {}
        self._input_binding_hash = (
            None if self.config.geometry_mode == "legacy" else canonical_content_hash(
                {"urdf_sha256": self.urdf_sha256,
                 "flange_from_tool": None if self._flange_from_tool is None else {
                     side: pose.tolist() for side, pose in sorted(self._flange_from_tool.items())
                 },
                 "self_proxy_boxes": [
                     {"side": box.side, "tool_from_box": pose.tolist(), "half_size_m": half.tolist()}
                     for box, pose, half in self._self_proxy_boxes
                ]}, domain="dexteleop.rm65.captured_filter_inputs.v1"
            )
        )

    @classmethod
    def for_binding(cls, *, urdf_path, config=None, observation_urdf_path=None,
                    self_observation_proxy_urdf_path=None,
                    tool_tcp_pose_flange_m_rad_by_side=None):
        """Bind explicit capture geometry without changing the physical model."""
        if config is not None and not isinstance(config, RM65CapturedRobotObservationFilterConfig):
            raise TypeError("config must be RM65CapturedRobotObservationFilterConfig")
        selected = config or RM65CapturedRobotObservationFilterConfig()
        if selected.geometry_mode == "legacy":
            if observation_urdf_path is not None or self_observation_proxy_urdf_path is not None:
                raise ValueError("observation resources require an explicit extended geometry_mode")
            return cls(urdf_path=urdf_path, config=selected)
        observation = (urdf_path if observation_urdf_path is None else
            validate_observation_urdf_candidate(urdf_path, observation_urdf_path))
        if self_observation_proxy_urdf_path is not None and selected.tool_surface_bounds_m is None:
            raise ValueError("self observation boxes require explicit tool surface bounds")
        boxes = (() if self_observation_proxy_urdf_path is None else
            load_dmtac_self_observation_proxy_boxes(self_observation_proxy_urdf_path))
        return cls(urdf_path=observation, config=selected,
            tool_tcp_pose_flange_m_rad_by_side=tool_tcp_pose_flange_m_rad_by_side,
            self_observation_proxy_boxes=boxes)

    @property
    def input_binding_hash(self) -> str | None:
        return self._input_binding_hash

    def _capture_primitive_geometry(self, arm, source_from_camera):
        """Return capture-only boxes plus immutable local query references.

        Calibrated self boxes are trusted contains volumes. Other boxes only
        select small pixel regions; local surface queries still reject empty
        corners and the open finger gap. No whole-image CAD rays are needed.
        """
        if self._flange_from_tool is None:
            raise ValueError("link primitives require calibrated left/right tool TCPs")
        pin, _, _, model, visual, selected = self._runtime_components()
        if any(not _has_primitive_rule(visual.geometryObjects[i].name) for i in selected):
            raise ValueError('Unrecognized self accessory requires geometry_mode=cad_surface')
        bodies = self._body_surface_references()
        for index in selected:
            item = visual.geometryObjects[index]
            if item.name in bodies and item.name not in self._body_sample_trees:
                vertices, faces = self._geometry_mesh(index, item.geometry)
                # Vertices and triangle centers are actual surface points.
                # Their mesh-derived coverage bound sends uncertain cases to
                # the original BVH; expensive dense body subdivision is unnecessary.
                triangles = vertices[faces]
                samples = np.concatenate((vertices, triangles.mean(axis=1)))
                _, first = np.unique(np.floor(samples / .005).astype(np.int64),
                                     axis=0, return_index=True)
                max_edge = float(np.linalg.norm(
                    triangles - triangles[:, [1, 2, 0]], axis=2).max())
                self._body_sample_trees[item.name] = (
                    cKDTree(samples[first]), max_edge + math.sqrt(3) * .005)
        data, visual_data = model.createData(), pin.GeometryData(visual)
        q = np.r_[arm[0], np.zeros(6), arm[1], np.zeros(6)]
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        pin.updateGeometryPlacements(model, data, visual, visual_data, q)
        tools = {}
        for side in ('left', 'right'):
            frame_id = model.getFrameId(side + '_arm_link6')
            if frame_id >= len(model.frames):
                raise ValueError('link primitives require the reviewed arm flange frame')
            tools[side] = np.asarray(data.oMf[frame_id].homogeneous) @ self._flange_from_tool[side]
        excluded = set()
        for index in selected:
            item = visual.geometryObjects[index]
            if 'camera' not in item.name and 'realsense' not in item.name:
                continue
            vertices = np.asarray(item.geometry.vertices(), float)
            camera_from_geometry = (np.linalg.inv(source_from_camera)
                                    @ np.asarray(visual_data.oMg[index].homogeneous))
            camera_vertices = (vertices @ camera_from_geometry[:3, :3].T
                               + camera_from_geometry[:3, 3])
            if _camera_mesh_contains_optical_origin(item.name, camera_vertices):
                excluded.add(item.name)
        queries = self._tool_query_references(self._tool_surface_references(), excluded)
        boxes = []
        # Trusted stepped envelopes first, so the common hand returns need no
        # surface queries at all. These never enter rigid collision geometry.
        for box, tool_from_box, half in self._self_proxy_boxes:
            boxes.append((box.geometry_name, tools[box.side] @ tool_from_box,
                          -half, half, 'contains', None))
        for side, reference in queries.items():
            _surface, tree = reference
            extent = np.max(np.abs(self.config.tool_surface_bounds_m), axis=1)
            boxes.append((side, tools[side], tree.mins * extent, tree.maxes * extent,
                          'tool', reference))
        for index in selected:
            item = visual.geometryObjects[index]
            if item.name in bodies:
                shape, lower, upper = bodies[item.name]
                boxes.append((item.name, np.asarray(visual_data.oMg[index].homogeneous).copy(),
                              lower, upper, 'body',
                              (shape, *self._body_sample_trees[item.name])))
            elif _is_arm_base_surface(item.name):
                vertices, _ = self._geometry_mesh(index, item.geometry)
                boxes.append((item.name, np.asarray(visual_data.oMg[index].homogeneous).copy(),
                    vertices.min(axis=0), vertices.max(axis=0), 'base', None))
        return boxes, excluded

    def _filter_with_primitives(self, *, depth, target, intrinsics, source_from_camera,
                                capture_qpos_by_side):
        fx, fy, cx, cy = (float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy"))
        if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
            raise ValueError("invalid capture camera intrinsics")
        source_from_camera = self._validated_pose(source_from_camera, field_name="source_from_camera")
        if set(capture_qpos_by_side) != {"left", "right"}:
            raise ValueError("capture qpos must contain exactly left and right")
        arm = [np.asarray(capture_qpos_by_side[s], float) for s in ("left", "right")]
        if any(q.shape != (6,) or not np.isfinite(q).all() for q in arm):
            raise ValueError("capture qpos must contain six finite radians per arm")
        with self._lock:
            placed, excluded = self._capture_primitive_geometry(arm, source_from_camera)
        remove = np.zeros(depth.shape, bool)
        camera_from_source = np.linalg.inv(source_from_camera)
        counts, native_counts, query_counts, ray_counts = {}, {}, {}, {}
        base_rois = []
        for name, source_from_frame, lower, upper, kind, reference in placed:
            tolerance = (self.config.body_surface_tolerance_m if kind == 'body'
                         else self.config.depth_tolerance_m)
            margin = (np.max(np.abs(self.config.tool_surface_bounds_m), axis=1)
                      * math.sqrt(self.config.tool_surface_normalized_squared_limit)
                      if kind == 'tool' else np.full(3, tolerance)
                      if kind == 'body' else np.zeros(3))
            row_slice, column_slice = _projected_box_roi(camera_from_source @ source_from_frame,
                lower - margin, upper + margin, intrinsics, depth.shape)
            if kind == 'base':
                if row_slice.start >= row_slice.stop or column_slice.start >= column_slice.stop:
                    continue
                # Preserve the old 12px/25mm base rule, but ray-test only the
                # base's small projection (also at zero observed depth pixels).
                radius = math.ceil(self.config.silhouette_radius_px)
                row_slice = slice(max(0, row_slice.start - radius), min(depth.shape[0], row_slice.stop + radius))
                column_slice = slice(max(0, column_slice.start - radius), min(depth.shape[1], column_slice.stop + radius))
                corners = np.array([[x, y, z] for x in (lower[0], upper[0])
                    for y in (lower[1], upper[1]) for z in (lower[2], upper[2])])
                camera_from_frame = camera_from_source @ source_from_frame
                z = (corners @ camera_from_frame[:3, :3].T + camera_from_frame[:3, 3])[:, 2]
                observed = depth[row_slice, column_slice]
                possible = ((observed > 0) & ~target[row_slice, column_slice]
                    & ~remove[row_slice, column_slice]
                    & (observed >= z.min() - self.config.depth_tolerance_m)
                    & (observed <= z.max() + self.config.depth_tolerance_m))
                if not possible.any():
                    continue
                if row_slice.start < row_slice.stop and column_slice.start < column_slice.stop:
                    base_rois.append((row_slice, column_slice))
                continue
            rows, columns = np.nonzero((depth[row_slice, column_slice] > 0)
                & ~target[row_slice, column_slice] & ~remove[row_slice, column_slice])
            rows, columns = rows + row_slice.start, columns + column_slice.start
            z = depth[rows, columns]
            points = np.column_stack(((columns - cx) * z / fx, (rows - cy) * z / fy, z))
            frame_from_camera = np.linalg.inv(source_from_frame) @ source_from_camera
            local = points @ frame_from_camera[:3, :3].T + frame_from_camera[:3, 3]
            candidates = np.flatnonzero(np.all((local >= lower - margin)
                                                & (local <= upper + margin), axis=1))
            inside = np.zeros(len(local), bool)
            query_counts[name] = len(candidates)
            if kind == 'contains':
                inside[candidates] = True
            elif kind == 'tool':
                surface, tree = reference
                inside[candidates] = _asymmetric_surface_membership(local[candidates], surface,
                    np.asarray(self.config.tool_surface_bounds_m), tree=tree,
                    normalized_squared_limit=self.config.tool_surface_normalized_squared_limit)
            else:
                shape, tree, coverage_bound = reference
                distances, _ = tree.query(local[candidates], distance_upper_bound=tolerance + coverage_bound)
                inside[candidates[distances <= tolerance]] = True
                band = candidates[(distances > tolerance) & np.isfinite(distances)]
                matched, calls = _native_body_surface_membership(local[band], shape, lower, upper, tolerance)
                inside[band[matched]] = True
                native_counts[name] = calls
            counts[name] = int(inside.sum())
            remove[rows[inside], columns[inside]] = True
        if base_rois:
            rows = slice(min(r.start for r, _ in base_rois), max(r.stop for r, _ in base_rois))
            columns = slice(min(c.start for _, c in base_rois), max(c.stop for _, c in base_rois))
            cropped = depth[rows, columns]
            cropped_intrinsics = dict(intrinsics, cx=cx - columns.start, cy=cy - rows.start)
            with self._lock:
                rendered, _ = self._render_depth_maps(depth_m=cropped, intrinsics=cropped_intrinsics,
                    source_from_camera=source_from_camera, capture_qpos_by_side=capture_qpos_by_side)
            # The original renderer includes every occluder, even when only
            # the small base projection is ray-tested. Separate base meshes
            # can expose hidden silhouettes and erase neighboring floor pixels.
            explained = captured_robot_observation_mask(depth_m=cropped, target_mask=target[rows, columns],
                rendered_depth_maps_m=rendered, silhouette_radius_px=self.config.silhouette_radius_px,
                depth_tolerance_m=self.config.depth_tolerance_m,
                depth_consistent_neighbor_candidates=self.config.depth_consistent_neighbor_candidates,
                extended_policy=True)
            counts['base_union'] = int(np.count_nonzero(explained & ~remove[rows, columns]))
            remove[rows, columns] |= explained
            ray_counts['base_union'] = int(cropped.size)
        filtered = depth.copy()
        filtered[remove] = 0
        evidence = {"schema": "dexteleop.rm65.captured_robot_observation_filter.v1",
            "method": "link_proxy_roi_local_surfaces_v1", "urdf_sha256": self.urdf_sha256,
            "filter_config_hash": self.config.semantic_content_hash,
            "filter_input_binding_hash": self._input_binding_hash,
            "primitive_box_count": len(placed), "rendered_depth_map_count": len(ray_counts),
            "primitive_removed_points_by_geometry": counts,
            "local_surface_query_counts_by_geometry": query_counts,
            "native_boundary_query_counts_by_geometry": native_counts,
            "base_roi_ray_counts_by_geometry": ray_counts,
            "excluded_capture_origin_camera_meshes": sorted(excluded),
            "removed_background_points": int(remove.sum()), "removed_target_points": 0,
            "target_depth_points_before": int(np.count_nonzero(target & (depth > 0))),
            "target_depth_points_after": int(np.count_nonzero(target & (filtered > 0)))}
        evidence["semantic_content_hash"] = canonical_content_hash(evidence,
            domain="dexteleop.rm65.captured_robot_observation_filter_result.v1")
        return RM65CapturedRobotObservationFilterResult(filtered, remove, evidence)

    def _mesh_buffers_for(self, geometry_index: int, shape: Any) -> tuple[np.ndarray, np.ndarray]:
        # The loaded visual model owns fixed mesh topology. Cache only local
        # buffers under the renderer lock; FK, camera transforms and ray hits
        # are recomputed for every view and sampled finger state.
        buffers = self._mesh_buffers.get(geometry_index)
        if buffers is None:
            vertices = np.asarray(shape.vertices(), dtype=float).copy(order="K")
            faces = np.asarray(
                [
                    (int(triangle[0]), int(triangle[1]), int(triangle[2]))
                    for triangle in (
                        shape.tri_indices(index) for index in range(int(shape.num_tris))
                    )
                ],
                dtype=np.int64,
            )
            vertices.setflags(write=False)
            faces.setflags(write=False)
            buffers = vertices, faces
            self._mesh_buffers[geometry_index] = buffers
        return buffers


    def _geometry_mesh(self, index: int, shape: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._mesh_buffers_for(index, shape)

    def _tool_query_references(self, surfaces, excluded):
        """Cache local samples/trees, never camera poses or removal decisions."""
        result = {}
        extent = np.max(np.abs(self.config.tool_surface_bounds_m), axis=1)
        for side, geometries in surfaces.items():
            names = tuple(name for name in geometries if name not in excluded)
            key = (side, names)
            if key not in self._tool_queries:
                surface = np.concatenate([geometries[name] for name in names])
                surface.setflags(write=False)
                self._tool_queries[key] = (surface, cKDTree(surface / extent))
            result[side] = self._tool_queries[key]
        return result

    def _body_surface_references(self) -> dict[str, tuple[Any, np.ndarray, np.ndarray]]:
        """Reuse native arm CAD BVHs in each geometry's local coordinates."""
        if self._body_surfaces is not None:
            return self._body_surfaces
        _, _, _, _, visual, selected = self._runtime_components()
        references = {}
        for index in selected:
            item = visual.geometryObjects[index]
            if not _is_arm_body_surface(item.name):
                continue
            shape = item.geometry
            vertices = np.asarray(shape.vertices(), dtype=float)
            references[item.name] = (shape, vertices.min(axis=0), vertices.max(axis=0))
        self._body_surfaces = references
        return references

    def _tool_surface_references(self) -> dict[str, dict[str, np.ndarray]]:
        """Cache complete local CAD surfaces independent of camera or arm pose.

        Deterministic subdivision stays on the original triangles. Every
        triangle edge is <= 2 mm; no clustered/display mesh or randomized
        samples participate. Sampling is conservative at tolerance boundaries.
        """
        if self._tool_surfaces is not None:
            return self._tool_surfaces
        assert self._flange_from_tool is not None
        pin, trimesh, _, model, visual, selected = self._runtime_components()
        data, visual_data = model.createData(), pin.GeometryData(visual)
        samples: dict[int, np.ndarray] = {}
        for index in selected:
            item = visual.geometryObjects[index]
            if _tool_surface_side(item.name) is None:
                continue
            shape = item.geometry
            vertices, faces = self._geometry_mesh(index, shape)
            samples[index] = _surface_samples(trimesh, vertices, faces)
        references: dict[str, dict[str, list[np.ndarray]]] = {"left": {}, "right": {}}
        for state, finger_q in enumerate(self.config.finger_sweep_q_rad):
            q = np.zeros(24)
            q[6:12] = finger_q
            q[18:24] = finger_q
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            pin.updateGeometryPlacements(model, data, visual, visual_data, q)
            for index, vertices in samples.items():
                name = visual.geometryObjects[index].name
                if state and not _is_robot_visual_mesh(name, finger_only=True):
                    continue
                side = _tool_surface_side(name)
                assert side is not None
                frame_id = model.getFrameId(side + "_arm_link6")
                if frame_id >= len(model.frames):
                    raise ValueError("tool surface filter requires the reviewed arm flange frame")
                source_from_tool = np.asarray(data.oMf[frame_id].homogeneous) @ self._flange_from_tool[side]
                tool_from_geometry = np.linalg.inv(source_from_tool) @ np.asarray(visual_data.oMg[index].homogeneous)
                references[side].setdefault(name, []).append(
                    vertices @ tool_from_geometry[:3, :3].T + tool_from_geometry[:3, 3])
        completed: dict[str, dict[str, np.ndarray]] = {}
        for side, geometries in references.items():
            if not geometries:
                raise ValueError(f"selected URDF lacks {side} tool surface geometry")
            completed[side] = {
                name: np.unique(np.concatenate(parts), axis=0) for name, parts in geometries.items()
            }
        self._tool_surfaces = completed
        return self._tool_surfaces

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

    def _render_depth_maps(self, **kwargs):
        if self.config.geometry_mode == "legacy":
            return self._render_legacy_depth_maps(**kwargs)
        return self._render_extended_depth_maps(**kwargs)

    def _render_extended_depth_maps(
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
        # CAD visibility must not depend on missing/noisy observed depth.
        # In particular a zero-depth pixel can still explain its neighbors.
        rows, columns = np.indices(depth_m.shape).reshape(2, -1)
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
        excluded_origin_housings: set[str] = set()
        tool_from_camera: dict[str, list[list[float]]] = {}
        attachment_indices = tuple(i for i in selected
            if _is_observation_attachment(visual_model.geometryObjects[i].name))
        attachment_layers = 0
        previous_attachment_poses: np.ndarray | None = None
        body_from_camera: dict[str, list[list[float]]] = {}
        # Full local tool references already include every finger state. The
        # Extra nominal finger rays contain no body hits for calibrated
        # filtering. Explicit attachments still follow their native parent
        # throughout the sampled finger states; identical rigid poses dedup.
        tool_surfaces_enabled = (self._flange_from_tool is not None
                                 and self.config.tool_surface_bounds_m is not None)
        render_states = (self.config.finger_sweep_q_rad if self._flange_from_tool is None
                         or self.config.tool_surface_bounds_m is None or attachment_indices
                         else self.config.finger_sweep_q_rad[:1])
        for sample_index, finger_q_rad in enumerate(render_states):
            sample_q = q.copy()
            sample_q[6:12] = finger_q_rad
            sample_q[18:24] = finger_q_rad
            pin.forwardKinematics(model, data, sample_q)
            pin.updateFramePlacements(model, data)
            pin.updateGeometryPlacements(
                model, data, visual_model, visual_data, sample_q
            )
            attachment_poses = np.asarray([
                np.r_[np.asarray(visual_data.oMg[i].rotation).reshape(-1),
                      np.asarray(visual_data.oMg[i].translation)] for i in attachment_indices])
            attachment_changed = (previous_attachment_poses is None or
                not np.allclose(previous_attachment_poses, attachment_poses, rtol=0, atol=1e-12))
            if sample_index and tool_surfaces_enabled and not attachment_changed:
                continue
            if sample_index == 0 and self._flange_from_tool is not None:
                for side in ("left", "right"):
                    frame_id = model.getFrameId(side + "_arm_link6")
                    if frame_id >= len(model.frames):
                        raise ValueError("tool surface filter requires the reviewed arm flange frame")
                    source_from_tool = np.asarray(data.oMf[frame_id].homogeneous) @ self._flange_from_tool[side]
                    tool_from_camera[side] = (np.linalg.inv(source_from_tool) @ source_from_camera).tolist()
            vertices_parts: list[np.ndarray] = []
            faces_parts: list[np.ndarray] = []
            body_face_parts: list[np.ndarray] = []
            attachment_parts: list[tuple[np.ndarray, np.ndarray]] = []
            vertex_offset = 0
            geometry_count = 0
            for geometry_index in selected:
                item = visual_model.geometryObjects[geometry_index]
                if sample_index and tool_surfaces_enabled and not _is_observation_attachment(item.name):
                    continue
                if sample_index and not _is_robot_visual_mesh(
                    item.name, finger_only=True
                ):
                    continue
                shape = item.geometry
                local_vertices, faces = self._geometry_mesh(geometry_index, shape)
                source_from_geometry = visual_data.oMg[geometry_index]
                source_vertices = (
                    local_vertices @ np.asarray(source_from_geometry.rotation).T
                    + np.asarray(source_from_geometry.translation)
                )
                camera_vertices = (
                    source_vertices @ camera_from_source[:3, :3].T
                    + camera_from_source[:3, 3]
                )
                if sample_index == 0 and _is_arm_body_surface(item.name):
                    geometry_from_source = np.eye(4)
                    geometry_from_source[:3, :3] = np.asarray(source_from_geometry.rotation).T
                    geometry_from_source[:3, 3] = -(geometry_from_source[:3, :3]
                        @ np.asarray(source_from_geometry.translation))
                    body_from_camera[item.name] = (geometry_from_source @ source_from_camera).tolist()
                if _camera_mesh_contains_optical_origin(item.name, camera_vertices):
                    excluded_origin_housings.add(item.name)
                    continue
                vertices_parts.append(camera_vertices)
                faces_parts.append(faces + vertex_offset)
                body_face_parts.append(np.full(len(faces), _tool_surface_side(item.name) is None))
                if _is_observation_attachment(item.name):
                    attachment_parts.append((camera_vertices, faces))
                vertex_offset += len(local_vertices)
                geometry_count += 1
            if not vertices_parts:
                raise RuntimeError("captured-robot renderer selected no mesh geometry")
            if not sample_index or not tool_surfaces_enabled:
                vertices = np.concatenate(vertices_parts, axis=0)
                faces = np.concatenate(faces_parts, axis=0)
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                locations, ray_indices, triangle_indices = intersector_type(mesh).intersects_location(
                    origins, directions, multiple_hits=False
                )
                if tool_surfaces_enabled:
                    # Keep first-hit occlusion by tools; their self matches
                    # use complete local CAD, not cropped ray hits.
                    body_hit = np.concatenate(body_face_parts)[triangle_indices]
                    locations, ray_indices = locations[body_hit], ray_indices[body_hit]
                rendered = np.full(depth_m.shape, np.nan, dtype=float)
                rendered[rows[ray_indices], columns[ray_indices]] = locations[:, 2]
                rendered_depth_maps.append(rendered)
                triangle_counts.append(len(faces))
                geometry_counts.append(geometry_count)
            if attachment_parts and attachment_changed:
                attachment_layers += 1
                # Explicit observation-only accessories are a separate depth
                # hypothesis. Nominal CAD may occlude an unmodeled cable or
                # accessory at the very pixel being explained. Do not extend
                # this multi-layer rule to arbitrary robot/environment meshes.
                extra_vertices, extra_faces, offset = [], [], 0
                for points, triangles in attachment_parts:
                    extra_vertices.append(points)
                    extra_faces.append(triangles + offset)
                    offset += len(points)
                extra_mesh = trimesh.Trimesh(vertices=np.concatenate(extra_vertices),
                    faces=np.concatenate(extra_faces), process=False)
                hits, pixels, _ = intersector_type(extra_mesh).intersects_location(
                    origins, directions, multiple_hits=False)
                extra_depth = np.full(depth_m.shape, np.nan, dtype=float)
                extra_depth[rows[pixels], columns[pixels]] = hits[:, 2]
                rendered_depth_maps.append(extra_depth)
                triangle_counts.append(sum(len(faces) for _, faces in attachment_parts))
                geometry_counts.append(len(attachment_parts))
                previous_attachment_poses = attachment_poses.copy()
        return tuple(rendered_depth_maps), {
            "rendered_depth_map_count": len(rendered_depth_maps),
            "rendered_geometry_count_by_state": tuple(geometry_counts),
            "rendered_triangle_count_by_state": tuple(triangle_counts),
            "excluded_capture_origin_camera_meshes": sorted(excluded_origin_housings),
            "tool_from_camera_by_side": tool_from_camera,
            "self_observation_attachment_geometry_count": len(attachment_indices),
            "self_observation_attachment_layer_count": attachment_layers,
            "body_geometry_from_camera": body_from_camera,
        }

    def _render_legacy_depth_maps(
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
                    item.name, finger_only=True, include_observation_attachments=False
                ):
                    continue
                local_vertices, faces = self._mesh_buffers_for(geometry_index, item.geometry)
                source_from_geometry = visual_data.oMg[geometry_index]
                source_vertices = (
                    local_vertices @ np.asarray(source_from_geometry.rotation).T
                    + np.asarray(source_from_geometry.translation)
                )
                camera_vertices = (
                    source_vertices @ camera_from_source[:3, :3].T
                    + camera_from_source[:3, 3]
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


    def _call_legacy(
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


    def __call__(
        self,
        *,
        depth_m: np.ndarray,
        target_mask: np.ndarray,
        intrinsics: Mapping[str, float],
        source_from_camera: np.ndarray,
        capture_qpos_by_side: Mapping[str, np.ndarray],
    ) -> RM65CapturedRobotObservationFilterResult:
        if self.config.geometry_mode == "legacy":
            return self._call_legacy(depth_m=depth_m, target_mask=target_mask,
                intrinsics=intrinsics, source_from_camera=source_from_camera,
                capture_qpos_by_side=capture_qpos_by_side)
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
        if self.config.geometry_mode == "link_primitives":
            return self._filter_with_primitives(depth=depth, target=target, intrinsics=intrinsics,
                source_from_camera=source_from_camera, capture_qpos_by_side=capture_qpos_by_side)
        with self._lock:
            rendered, render_evidence = self._render_depth_maps(
                depth_m=depth,
                intrinsics=intrinsics,
                source_from_camera=source_from_camera,
                capture_qpos_by_side=capture_qpos_by_side,
            )
            tool_surfaces = (None if self._flange_from_tool is None or self.config.tool_surface_bounds_m is None
                             else self._tool_surface_references())
            body_surfaces = (self._body_surface_references()
                if self.config.body_surface_tolerance_m > 0
                and render_evidence.get("body_geometry_from_camera") else {})
            tool_queries = ({} if tool_surfaces is None else self._tool_query_references(
                tool_surfaces, set(render_evidence["excluded_capture_origin_camera_meshes"])))
        remove = captured_robot_observation_mask(
            depth_m=depth,
            target_mask=target,
            rendered_depth_maps_m=rendered,
            silhouette_radius_px=self.config.silhouette_radius_px,
            depth_tolerance_m=self.config.depth_tolerance_m,
            depth_consistent_neighbor_candidates=(
                self.config.depth_consistent_neighbor_candidates
            ),
            extended_policy=True,
        )
        tool_surface_counts: dict[str, int] = {}
        self_proxy_counts: dict[str, int] = {}
        if tool_surfaces is not None:
            rows, columns = np.nonzero((depth > 0) & ~target & ~remove)
            z = depth[rows, columns]
            camera_points = np.column_stack((
                (columns - intrinsics["cx"]) * z / intrinsics["fx"],
                (rows - intrinsics["cy"]) * z / intrinsics["fy"], z))
            for side, (surface, tree) in tool_queries.items():
                tool_surface_counts[side] = len(surface)
                tool_from_camera = np.asarray(render_evidence["tool_from_camera_by_side"][side])
                local_points = camera_points @ tool_from_camera[:3, :3].T + tool_from_camera[:3, 3]
                matched = _asymmetric_surface_membership(
                    local_points, surface, np.asarray(self.config.tool_surface_bounds_m),
                    normalized_squared_limit=self.config.tool_surface_normalized_squared_limit,
                    tree=tree)
                remove[rows[matched], columns[matched]] = True
                # These boxes explain only this camera's captured robot pose.
                # They never become planning obstacles or follow future qpos.
                for box, tool_from_box, half in self._self_proxy_boxes:
                    if box.side != side:
                        continue
                    matched = _self_proxy_membership(local_points, tool_from_box, half)
                    added = matched & ~remove[rows, columns]
                    self_proxy_counts[box.geometry_name] = int(np.count_nonzero(added))
                    remove[rows[matched], columns[matched]] = True
        body_query_counts: dict[str, int] = {}
        body_added = 0
        if body_surfaces:
            rows, columns = np.nonzero((depth > 0) & ~target & ~remove)
            z = depth[rows, columns]
            camera_points = np.column_stack((
                (columns - intrinsics["cx"]) * z / intrinsics["fx"],
                (rows - intrinsics["cy"]) * z / intrinsics["fy"], z))
            body_matched = np.zeros(len(rows), dtype=bool)
            for name, (geometry, lower, upper) in body_surfaces.items():
                geometry_from_camera = np.asarray(render_evidence["body_geometry_from_camera"][name])
                local_points = (camera_points @ geometry_from_camera[:3, :3].T
                                + geometry_from_camera[:3, 3])
                pending = np.flatnonzero(~body_matched)
                matched, count = _native_body_surface_membership(local_points[pending],
                    geometry, lower, upper, self.config.body_surface_tolerance_m)
                body_query_counts[name] = count
                body_matched[pending[matched]] = True
            remove[rows[body_matched], columns[body_matched]] = True
            body_added = int(body_matched.sum())
        if np.any(remove & target):
            raise RuntimeError("captured-robot filter attempted to remove target pixels")
        filtered = depth.copy()
        filtered[remove] = 0.0
        semantic_evidence = {
            "schema": "dexteleop.rm65.captured_robot_observation_filter.v1",
            "method": (
                "urdf_cad_and_capture_self_proxy_v6" if self._self_proxy_boxes else
                "urdf_cad_body_and_tool_surfaces_v5" if body_surfaces else
                "urdf_cad_full_tool_surface_split_ellipsoid_v4"
                if tool_surfaces is not None else
                "urdf_cad_depth_finger_sweep_optical_origin_exclusion_v3"
            ),
            "urdf_sha256": self.urdf_sha256,
            "filter_config_hash": self.config.semantic_content_hash,
            "filter_input_binding_hash": self._input_binding_hash,
            "silhouette_radius_px": self.config.silhouette_radius_px,
            "depth_tolerance_m": self.config.depth_tolerance_m,
            "depth_consistent_neighbor_candidates": (
                self.config.depth_consistent_neighbor_candidates
            ),
            "finger_sweep_q_rad": self.config.finger_sweep_q_rad,
            "render_ray_domain": "whole_image_independent_of_observed_depth",
            "tool_surface_bounds_m": self.config.tool_surface_bounds_m if tool_surfaces is not None else None,
            "tool_surface_normalized_squared_limit": self.config.tool_surface_normalized_squared_limit if tool_surfaces is not None else None,
            "tool_surface_max_triangle_edge_m": 0.002 if tool_surfaces is not None else None,
            "tool_surface_reference_cell_m": 0.001 if tool_surfaces is not None else None,
            "tool_surface_reference_coverage_bound_m": (0.002 + math.sqrt(3) * 0.001) if tool_surfaces is not None else None,
            "tool_surface_sample_counts_by_side": tool_surface_counts,
            "self_proxy_additional_removed_points_by_geometry": self_proxy_counts,
            "self_proxy_boxes": [
                {"name": box.geometry_name, "side": box.side,
                 "tool_from_box": pose.tolist(), "size_m": (half * 2).tolist()}
                for box, pose, half in self._self_proxy_boxes],
            "body_surface_tolerance_m": self.config.body_surface_tolerance_m,
            "body_surface_native_query_counts_by_geometry": body_query_counts,
            "body_surface_additional_removed_points": body_added,
            "body_geometry_from_camera": render_evidence.get("body_geometry_from_camera", {}),
            "tool_from_camera_by_side": render_evidence["tool_from_camera_by_side"],
            "rendered_depth_map_count": render_evidence["rendered_depth_map_count"],
            "self_observation_attachment_geometry_count": render_evidence.get(
                "self_observation_attachment_geometry_count", 0),
            "self_observation_attachment_layer_count": render_evidence.get(
                "self_observation_attachment_layer_count", 0),
            "rendered_geometry_count_by_state": render_evidence[
                "rendered_geometry_count_by_state"
            ],
            "rendered_triangle_count_by_state": render_evidence[
                "rendered_triangle_count_by_state"
            ],
            "excluded_capture_origin_camera_meshes": render_evidence["excluded_capture_origin_camera_meshes"],
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
    "validate_observation_urdf_candidate",
]

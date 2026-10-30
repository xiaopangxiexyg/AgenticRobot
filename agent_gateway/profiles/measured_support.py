"""Sensor-only support top measurement for held-object release planning.

From one bound metric-depth cut and the support's semantic mask:

1. fit the measured top plane (``project_depth_mask_support_surface``, or the
   bounded patch search when the whole mask does not fit one plane);
2. sample that plane on a 2 mm grid over the mask's measured XY extent and keep
   samples whose pixel lies in the mask with planar metric depth;
3. take the largest fully observed axis-aligned rectangle (optionally also the
   next largest distinct ones, e.g. both free strips beside an object resting
   on the top), shrink each by a lateral margin and certify it with
   ``check_observed_support_footprint``.

No CAD pose, native body or fixed-support assumption is used, so a free box or
a mat qualifies the same way from simulated or real depth. Occluded or
unobserved parts of the top are excluded, never assumed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from robot_motion.object_transport import SupportSurfaceObservation

from .back_projection import (
    LiveAlignedDepthCapture,
    ObservedSupportFootprintCoverage,
    ProjectedSupportSurface,
    check_observed_support_footprint,
    project_depth_mask_support_surface,
    search_depth_mask_support_surface,
)

GRID_RESOLUTION_M = 0.002
MAXIMUM_PLANE_RESIDUAL_M = 0.002
DEFAULT_LATERAL_MARGIN_M = 0.002
MINIMUM_REGION_SIDE_M = 0.02
MAXIMUM_GRID_CELLS = 100_000


@dataclass(frozen=True)
class MeasuredSupportTop:
    """A certified, fully observed rectangle on one measured top plane."""

    plane: ProjectedSupportSurface
    coverage: ObservedSupportFootprintCoverage
    region_min_xy_m: tuple[float, float]
    region_max_xy_m: tuple[float, float]
    observed_min_xy_m: tuple[float, float]
    observed_max_xy_m: tuple[float, float]
    top_z_m: float
    tilt_deg: float

    def observation(self, *, support_id: str, reference_frame_id: str, scene_revision: int,
                    measured_at: float, evidence_ref: str) -> SupportSurfaceObservation:
        return SupportSurfaceObservation(
            support_id=support_id, reference_frame_id=reference_frame_id,
            scene_revision=scene_revision, measured_at=measured_at, top_z_m=self.top_z_m,
            normal_reference_xyz=self.plane.normal_reference_xyz,
            region_min_xy_m=self.region_min_xy_m, region_max_xy_m=self.region_max_xy_m,
            evidence_ref=evidence_ref, geometry_complete=True)


def _mask_points(capture: LiveAlignedDepthCapture, mask: np.ndarray, maximum_depth_m: float):
    depth = capture.depth_m
    valid = mask & np.isfinite(depth) & (depth > 0.) & (depth <= maximum_depth_m)
    rows, cols = np.nonzero(valid)
    intrinsics = capture.intrinsics
    d = depth[rows, cols].astype(np.float64)
    camera = np.column_stack(((cols - float(intrinsics["cx"])) * d / float(intrinsics["fx"]),
                              (rows - float(intrinsics["cy"])) * d / float(intrinsics["fy"]), d))
    transform = capture.reference_from_camera
    return camera @ transform[:3, :3].T + transform[:3, 3]


def plane_coverage_grid(capture: LiveAlignedDepthCapture, mask: np.ndarray,
                        plane: ProjectedSupportSurface, *, xs: np.ndarray, ys: np.ndarray,
                        maximum_plane_residual_m: float = MAXIMUM_PLANE_RESIDUAL_M) -> np.ndarray:
    """Per-sample version of the footprint predicate; grid[j, i] is (xs[i], ys[j])."""
    grid_x, grid_y = np.meshgrid(xs, ys)
    anchor = np.asarray(plane.plane_anchor_reference_xyz_m)
    normal = np.asarray(plane.normal_reference_xyz)
    x, y = grid_x.ravel(), grid_y.ravel()
    z = anchor[2] - (normal[0] * (x - anchor[0]) + normal[1] * (y - anchor[1])) / normal[2]
    points = np.stack((x, y, z), axis=1)
    transform = capture.reference_from_camera
    camera = (points - transform[:3, 3]) @ transform[:3, :3]
    intrinsics = capture.intrinsics
    covered = np.zeros(len(points), dtype=bool)
    front = camera[:, 2] > 0
    u = np.full(len(points), -1, dtype=int)
    v = np.full(len(points), -1, dtype=int)
    u[front] = np.rint(float(intrinsics["fx"]) * camera[front, 0] / camera[front, 2]
                       + float(intrinsics["cx"])).astype(int)
    v[front] = np.rint(float(intrinsics["fy"]) * camera[front, 1] / camera[front, 2]
                       + float(intrinsics["cy"])).astype(int)
    height, width = mask.shape
    inside = front & (u >= 0) & (v >= 0) & (u < width) & (v < height)
    index = np.flatnonzero(inside)
    selected = mask[v[index], u[index]]
    index = index[selected]
    depth = capture.depth_m[v[index], u[index]].astype(np.float64)
    positive = depth > 0
    index, depth = index[positive], depth[positive]
    measured_camera = np.column_stack(
        ((u[index] - float(intrinsics["cx"])) * depth / float(intrinsics["fx"]),
         (v[index] - float(intrinsics["cy"])) * depth / float(intrinsics["fy"]), depth))
    measured = measured_camera @ transform[:3, :3].T + transform[:3, 3]
    residual = np.abs((measured - anchor) @ normal)
    covered[index[residual <= maximum_plane_residual_m]] = True
    return covered.reshape(grid_x.shape)


def largest_true_rectangle(grid: np.ndarray) -> tuple[int, int, int, int] | None:
    """Largest all-true axis-aligned block as (row0, col0, row1, col1), inclusive."""
    rows, cols = grid.shape
    heights = np.zeros(cols, dtype=int)
    best, best_area = None, 0
    for row in range(rows):
        heights = np.where(grid[row], heights + 1, 0)
        stack: list[int] = []
        for col in range(cols + 1):
            current = int(heights[col]) if col < cols else 0
            while stack and int(heights[stack[-1]]) >= current:
                height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                area = height * (col - left)
                if height and area > best_area:
                    best_area = area
                    best = (row - height + 1, left, row, col - 1)
            stack.append(col)
    return best


def maximal_true_rectangles(grid: np.ndarray, *, limit: int,
                            maximum_overlap_fraction: float = .75) -> list[tuple[int, int, int, int]]:
    """Up to ``limit`` large all-true blocks, largest first, as (row0, col0, row1, col1).

    Candidates are the row-histogram maximal blocks. A later block is kept only
    if no kept block covers more than ``maximum_overlap_fraction`` of it, so a
    sub-block of a kept block never repeats, while two crossing free strips
    around an obstruction are both returned. The first is ``largest_true_rectangle``.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    rows, cols = grid.shape
    heights = np.zeros(cols, dtype=int)
    found: dict[tuple[int, int, int, int], int] = {}
    for row in range(rows):
        heights = np.where(grid[row], heights + 1, 0)
        stack: list[int] = []
        for col in range(cols + 1):
            current = int(heights[col]) if col < cols else 0
            while stack and int(heights[stack[-1]]) >= current:
                height = int(heights[stack.pop()])
                left = stack[-1] + 1 if stack else 0
                if height:
                    found[(row - height + 1, left, row, col - 1)] = height * (col - left)
            stack.append(col)

    def overlap(a, b):
        rows_ = min(a[2], b[2]) - max(a[0], b[0]) + 1
        cols_ = min(a[3], b[3]) - max(a[1], b[1]) + 1
        return max(rows_, 0) * max(cols_, 0)

    chosen: list[tuple[int, int, int, int]] = []
    # Stable on first discovery, so ties resolve exactly like largest_true_rectangle.
    for block, area in sorted(found.items(), key=lambda item: -item[1]):
        if all(overlap(block, other) <= maximum_overlap_fraction * area for other in chosen):
            chosen.append(block)
            if len(chosen) == limit:
                break
    return chosen


def fit_support_top_plane(capture: LiveAlignedDepthCapture, mask: np.ndarray, *,
                          maximum_depth_m: float) -> ProjectedSupportSurface:
    """Whole-mask top fit first; a bounded local patch search when it is not one plane."""
    try:
        return project_depth_mask_support_surface(capture, mask, mask_clipped=False,
                                                  maximum_depth_m=maximum_depth_m)
    except ValueError:
        searched = search_depth_mask_support_surface(
            capture, mask, mask_clipped=False, maximum_depth_m=maximum_depth_m,
            patch_size_px=24, stride_px=12, maximum_patches=1024)
        return searched.candidate


def measure_support_top(capture: LiveAlignedDepthCapture, mask: np.ndarray, *,
                        maximum_depth_m: float,
                        lateral_margin_m: float = DEFAULT_LATERAL_MARGIN_M,
                        minimum_region_side_m: float = MINIMUM_REGION_SIDE_M) -> MeasuredSupportTop:
    """Certify the largest fully observed rectangle on the support's measured top."""
    return measure_support_tops(capture, mask, maximum_depth_m=maximum_depth_m,
                                lateral_margin_m=lateral_margin_m,
                                minimum_region_side_m=minimum_region_side_m, limit=1)[0]


def measure_support_tops(capture: LiveAlignedDepthCapture, mask: np.ndarray, *,
                         maximum_depth_m: float,
                         lateral_margin_m: float = DEFAULT_LATERAL_MARGIN_M,
                         minimum_region_side_m: float = MINIMUM_REGION_SIDE_M,
                         limit: int = 1) -> tuple[MeasuredSupportTop, ...]:
    """Certify up to ``limit`` distinct fully observed rectangles, largest first.

    All share one measured plane. A rectangle narrower than the minimum release
    region after its margin is dropped; none left raises like a too-small top.
    """
    if not isinstance(capture, LiveAlignedDepthCapture):
        raise TypeError("A bound metric-depth capture is required")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape:
        raise ValueError("Support mask must be a paired full-resolution boolean grid")
    if not selected.any():
        raise ValueError("Support mask is empty")
    border = (selected[0].any() or selected[-1].any()
              or selected[:, 0].any() or selected[:, -1].any())
    if border:
        raise ValueError("Support mask touches the image border; its top may be clipped")
    for name, value in (("lateral_margin_m", lateral_margin_m),
                        ("minimum_region_side_m", minimum_region_side_m)):
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    plane = fit_support_top_plane(capture, selected, maximum_depth_m=maximum_depth_m)
    points = _mask_points(capture, selected, maximum_depth_m)
    if len(points) < 16:
        raise ValueError("Support mask has insufficient valid metric depth")
    low = np.min(points[:, :2], axis=0)
    high = np.max(points[:, :2], axis=0)
    counts = np.floor((high - low) / GRID_RESOLUTION_M).astype(int) + 1
    if int(counts[0]) * int(counts[1]) > MAXIMUM_GRID_CELLS:
        raise ValueError("Support extent exceeds the bounded sample budget")
    xs = low[0] + GRID_RESOLUTION_M * np.arange(int(counts[0]))
    ys = low[1] + GRID_RESOLUTION_M * np.arange(int(counts[1]))
    grid = plane_coverage_grid(capture, selected, plane, xs=xs, ys=ys)
    blocks = maximal_true_rectangles(grid, limit=limit)
    if not blocks:
        raise ValueError("No measured support top sample is observed")
    observed = replace(plane,
                       observed_bounds_min_xy_m=tuple(float(value) for value in low),
                       observed_bounds_max_xy_m=tuple(float(value) for value in high))
    anchor = np.asarray(plane.plane_anchor_reference_xyz_m)
    normal = np.asarray(plane.normal_reference_xyz)
    tilt = math.degrees(math.acos(max(-1., min(1., float(normal[2])))))
    tops = []
    for row0, col0, row1, col1 in blocks:
        region_min = np.array([xs[col0], ys[row0]]) + lateral_margin_m
        region_max = np.array([xs[col1], ys[row1]]) - lateral_margin_m
        if np.any(region_max - region_min < minimum_region_side_m):
            continue
        coverage = check_observed_support_footprint(
            capture, selected, observed,
            region_min_xy_m=tuple(float(value) for value in region_min),
            region_max_xy_m=tuple(float(value) for value in region_max),
            resolution_m=GRID_RESOLUTION_M, maximum_plane_residual_m=MAXIMUM_PLANE_RESIDUAL_M)
        corners = np.array([[region_min[0], region_min[1]], [region_min[0], region_max[1]],
                            [region_max[0], region_min[1]], [region_max[0], region_max[1]]])
        corner_z = anchor[2] - (normal[0] * (corners[:, 0] - anchor[0])
                                + normal[1] * (corners[:, 1] - anchor[1])) / normal[2]
        # The release planner treats top_z_m as the highest support point under the
        # footprint: keep the upper plane margin and the highest certified corner.
        top_z = max(float(plane.upper_top_z_m), float(np.max(corner_z)))
        tops.append(MeasuredSupportTop(
            plane=observed, coverage=coverage,
            region_min_xy_m=tuple(float(value) for value in region_min),
            region_max_xy_m=tuple(float(value) for value in region_max),
            observed_min_xy_m=tuple(float(value) for value in low),
            observed_max_xy_m=tuple(float(value) for value in high),
            top_z_m=top_z, tilt_deg=tilt))
    if not tops:
        raise ValueError("Observed support top is smaller than the minimum release region")
    return tuple(tops)


def _window_any(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square-window dilation by an integral image (numpy only)."""
    padded = np.pad(mask.astype(np.int32), radius + 1)
    integral = padded.cumsum(axis=0).cumsum(axis=1)
    size = 2 * radius + 1
    height, width = mask.shape
    total = (integral[size:size + height, size:size + width]
             - integral[:height, size:size + width]
             - integral[size:size + height, :width]
             + integral[:height, :width])
    return total > 0


def measure_surface_below(capture: LiveAlignedDepthCapture, mask: np.ndarray, *, top_z_m: float,
                          maximum_depth_m: float, ring_px: int = 12, minimum_points: int = 60,
                          below_top_margin_m: float = .005) -> float | None:
    """Height of the surface around an object's footprint, from a measured depth ring.

    Points of a ring around the mask that lie below the object's top face are
    the surface it rests on; taller neighbours, the arm and the camera
    background are excluded by height and depth. Returns None when the ring
    has too few such points (fully occluded surroundings).
    """
    selected = np.asarray(mask, dtype=bool)
    ring = _window_any(selected, ring_px) & ~_window_any(selected, 3)
    points = _mask_points(capture, ring, maximum_depth_m)
    if not len(points):
        return None
    below = points[points[:, 2] < top_z_m - below_top_margin_m]
    if len(below) < minimum_points:
        return None
    return float(np.median(below[:, 2]))

@dataclass(frozen=True)
class MeasuredObjectTop:
    """Measured top face of a box-like object: center, upward height, yaw, extent."""

    center_xy_m: tuple[float, float]
    top_z_m: float
    yaw_rad: float
    extent_xy_m: tuple[float, float]
    top_point_count: int
    plane_residual_m: float
    tilt_deg: float


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull of 2-D points, counter-clockwise, without duplicates."""
    unique = np.unique(np.round(points, 6), axis=0)
    if len(unique) < 3:
        return unique
    order = unique[np.lexsort((unique[:, 1], unique[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for point in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in order[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def minimum_area_rectangle(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """(center_xy, yaw in (-pi/4, pi/4], extent along yaw axes) of the minimum box."""
    hull = _convex_hull(np.asarray(points, dtype=float))
    if len(hull) < 3:
        raise ValueError("Object top has too few distinct measured points")
    best = None
    for index in range(len(hull)):
        edge = hull[(index + 1) % len(hull)] - hull[index]
        if not np.any(edge):
            continue
        angle = math.atan2(edge[1], edge[0])
        rotation = np.array([[math.cos(angle), math.sin(angle)],
                             [-math.sin(angle), math.cos(angle)]])
        local = hull @ rotation.T
        low, high = local.min(axis=0), local.max(axis=0)
        area = float(np.prod(high - low))
        if best is None or area < best[0]:
            best = (area, angle, low, high, rotation)
    _, angle, low, high, rotation = best
    center = ((low + high) / 2.) @ rotation
    extent = high - low
    # A box is symmetric under quarter turns: report the yaw in (-pi/4, pi/4].
    turns = round(angle / (math.pi / 2.))
    yaw = angle - turns * math.pi / 2.
    if yaw <= -math.pi / 4.:
        yaw += math.pi / 2.
        turns -= 1
    if turns % 2:
        extent = extent[::-1]
    return center, float(yaw), np.asarray(extent, dtype=float)


def measure_object_top(capture: LiveAlignedDepthCapture, mask: np.ndarray, *,
                       maximum_depth_m: float,
                       maximum_plane_residual_m: float = MAXIMUM_PLANE_RESIDUAL_M,
                       minimum_top_points: int = 24) -> MeasuredObjectTop:
    """Fit a box-like object's top plane from depth and its footprint orientation."""
    if not isinstance(capture, LiveAlignedDepthCapture):
        raise TypeError("A bound metric-depth capture is required")
    selected = np.asarray(mask)
    if selected.dtype != np.bool_ or selected.shape != capture.depth_m.shape or not selected.any():
        raise ValueError("Object mask must be a non-empty paired boolean grid")
    if (selected[0].any() or selected[-1].any() or selected[:, 0].any() or selected[:, -1].any()):
        raise ValueError("Object mask touches the image border; its top may be clipped")
    plane = fit_support_top_plane(capture, selected, maximum_depth_m=maximum_depth_m)
    points = _mask_points(capture, selected, maximum_depth_m)
    anchor = np.asarray(plane.plane_anchor_reference_xyz_m)
    normal = np.asarray(plane.normal_reference_xyz)
    residual = np.abs((points - anchor) @ normal)
    top = points[residual <= maximum_plane_residual_m]
    if len(top) < minimum_top_points:
        raise ValueError("Object top has insufficient measured coverage")
    center, yaw, extent = minimum_area_rectangle(top[:, :2])
    top_z = float(anchor[2] - (normal[0] * (center[0] - anchor[0])
                               + normal[1] * (center[1] - anchor[1])) / normal[2])
    tilt = math.degrees(math.acos(max(-1., min(1., float(normal[2])))))
    return MeasuredObjectTop(
        center_xy_m=(float(center[0]), float(center[1])), top_z_m=top_z, yaw_rad=yaw,
        extent_xy_m=(float(extent[0]), float(extent[1])), top_point_count=int(len(top)),
        plane_residual_m=float(np.quantile(residual[residual <= maximum_plane_residual_m], .95)),
        tilt_deg=tilt)


__all__ = ["MeasuredObjectTop", "MeasuredSupportTop", "fit_support_top_plane",
           "largest_true_rectangle", "measure_object_top", "measure_support_top",
           "measure_surface_below", "minimum_area_rectangle", "plane_coverage_grid"]

"""Sensor-only support top measurement for held-object release planning.

From one bound metric-depth cut and the support's semantic mask:

1. fit the measured top plane (``project_depth_mask_support_surface``, or the
   bounded patch search when the whole mask does not fit one plane);
2. sample that plane on a 2 mm grid over the mask's measured XY extent and keep
   samples whose pixel lies in the mask with planar metric depth;
3. take the largest fully observed axis-aligned rectangle, shrink it by a
   lateral margin and certify it with ``check_observed_support_footprint``.

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
    block = largest_true_rectangle(grid)
    if block is None:
        raise ValueError("No measured support top sample is observed")
    row0, col0, row1, col1 = block
    region_min = np.array([xs[col0], ys[row0]]) + lateral_margin_m
    region_max = np.array([xs[col1], ys[row1]]) - lateral_margin_m
    if np.any(region_max - region_min < minimum_region_side_m):
        raise ValueError("Observed support top is smaller than the minimum release region")
    observed = replace(plane,
                       observed_bounds_min_xy_m=tuple(float(value) for value in low),
                       observed_bounds_max_xy_m=tuple(float(value) for value in high))
    coverage = check_observed_support_footprint(
        capture, selected, observed,
        region_min_xy_m=tuple(float(value) for value in region_min),
        region_max_xy_m=tuple(float(value) for value in region_max),
        resolution_m=GRID_RESOLUTION_M, maximum_plane_residual_m=MAXIMUM_PLANE_RESIDUAL_M)
    anchor = np.asarray(plane.plane_anchor_reference_xyz_m)
    normal = np.asarray(plane.normal_reference_xyz)
    corners = np.array([[region_min[0], region_min[1]], [region_min[0], region_max[1]],
                        [region_max[0], region_min[1]], [region_max[0], region_max[1]]])
    corner_z = anchor[2] - (normal[0] * (corners[:, 0] - anchor[0])
                            + normal[1] * (corners[:, 1] - anchor[1])) / normal[2]
    # The release planner treats top_z_m as the highest support point under the
    # footprint: keep the upper plane margin and the highest certified corner.
    top_z = max(float(plane.upper_top_z_m), float(np.max(corner_z)))
    tilt = math.degrees(math.acos(max(-1., min(1., float(normal[2])))))
    return MeasuredSupportTop(
        plane=observed, coverage=coverage,
        region_min_xy_m=tuple(float(value) for value in region_min),
        region_max_xy_m=tuple(float(value) for value in region_max),
        observed_min_xy_m=tuple(float(value) for value in low),
        observed_max_xy_m=tuple(float(value) for value in high),
        top_z_m=top_z, tilt_deg=tilt)


__all__ = ["MeasuredSupportTop", "fit_support_top_plane", "largest_true_rectangle",
           "measure_support_top", "plane_coverage_grid"]

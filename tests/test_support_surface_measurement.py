"""Depth-only support top measurement used by support release queries."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from agent_gateway.profiles.back_projection import LiveAlignedDepthCapture
from agent_gateway.profiles.measured_support import (
    largest_true_rectangle,
    measure_support_top,
)

pytestmark = [pytest.mark.unit]

WIDTH, HEIGHT = 320, 240
INTRINSICS = {"width": WIDTH, "height": HEIGHT, "fx": 300., "fy": 300., "cx": 160., "cy": 120.}
TABLE_Z, TOP_Z = 0.01, 0.05
BOX_MIN, BOX_MAX = np.array([0.0, 0.0]), np.array([0.09, 0.09])


def _look_at(eye, target):
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, down, forward))
    pose[:3, 3] = eye
    return pose


def _render(eye=(0.045, -0.25, 0.35), occluder=None):
    """Ray-cast a box (top and front face) on a table; return depth, box mask, camera pose."""
    pose = _look_at(eye, (0.045, 0.045, TOP_Z))
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    rays = np.stack(((u - INTRINSICS["cx"]) / INTRINSICS["fx"],
                     (v - INTRINSICS["cy"]) / INTRINSICS["fy"], np.ones_like(u)), axis=-1)
    world = rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    depth = np.full((HEIGHT, WIDTH), np.inf)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)

    def hit(t, inside, on_box):
        nonlocal depth, mask
        better = inside & (t > 0) & (t < depth)
        depth = np.where(better, t, depth)
        mask = np.where(better, on_box, mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_top = (TOP_Z - origin[2]) / world[..., 2]
        p = origin + t_top[..., None] * world
        hit(t_top, (p[..., 0] >= BOX_MIN[0]) & (p[..., 0] <= BOX_MAX[0])
            & (p[..., 1] >= BOX_MIN[1]) & (p[..., 1] <= BOX_MAX[1]), True)
        t_front = (BOX_MIN[1] - origin[1]) / world[..., 1]
        p = origin + t_front[..., None] * world
        hit(t_front, (p[..., 0] >= BOX_MIN[0]) & (p[..., 0] <= BOX_MAX[0])
            & (p[..., 2] >= TABLE_Z) & (p[..., 2] <= TOP_Z), True)
        t_table = (TABLE_Z - origin[2]) / world[..., 2]
        hit(t_table, np.ones_like(t_table, dtype=bool), False)
    if occluder is not None:
        x0, y0, x1, y1, distance = occluder
        region = np.zeros_like(mask)
        region[y0:y1, x0:x1] = True
        depth = np.where(region, distance, depth)
        mask = mask & ~region
    depth = np.where(np.isfinite(depth), depth, 0.).astype(np.float32)
    return depth, mask, pose


def _capture(depth, pose):
    return LiveAlignedDepthCapture(
        source_id="camera:test", role="main", captured_at=1.0, station_revision=0,
        capture_group_id="group:test",
        depth_content_hash=hashlib.sha256(depth.tobytes()).hexdigest(),
        intrinsics=INTRINSICS, reference_from_camera=pose, depth_m=depth)


def test_largest_true_rectangle_finds_the_maximal_block():
    grid = np.zeros((6, 7), dtype=bool)
    grid[1:5, 2:6] = True
    grid[2, 3] = False
    grid[0:6, 0] = True
    row0, col0, row1, col1 = largest_true_rectangle(grid)
    area = (row1 - row0 + 1) * (col1 - col0 + 1)
    assert area == 8 and grid[row0:row1 + 1, col0:col1 + 1].all()
    assert largest_true_rectangle(np.zeros((3, 3), dtype=bool)) is None


def test_box_top_is_certified_inside_the_true_face_from_depth_only():
    depth, mask, pose = _render()
    top = measure_support_top(_capture(depth, pose), mask, maximum_depth_m=2.)
    low, high = np.asarray(top.region_min_xy_m), np.asarray(top.region_max_xy_m)
    assert np.all(low >= BOX_MIN) and np.all(high <= BOX_MAX)
    # Most of the 9 cm face is observed from this oblique view.
    assert np.all(high - low > 0.07)
    assert TOP_Z <= top.top_z_m <= TOP_Z + 0.004
    assert top.tilt_deg < 0.2
    assert top.coverage.sample_count > 1000
    observation = top.observation(support_id="green_box", reference_frame_id="station",
                                  scene_revision=0, measured_at=1.0, evidence_ref="test:support")
    assert observation.geometry_complete and observation.top_z_m == top.top_z_m


def test_occluded_part_of_the_top_is_excluded_not_assumed():
    clean = measure_support_top(_capture(*_render()[::2]), _render()[1], maximum_depth_m=2.)
    # An arm in front of the camera hides the right part of the face.
    depth, mask, pose = _render(occluder=(175, 0, WIDTH, HEIGHT, 0.2))
    occluded = measure_support_top(_capture(depth, pose), mask, maximum_depth_m=2.)
    clean_area = np.prod(np.subtract(clean.region_max_xy_m, clean.region_min_xy_m))
    area = np.prod(np.subtract(occluded.region_max_xy_m, occluded.region_min_xy_m))
    assert 0 < area < 0.8 * clean_area
    # Every certified point is still on the true face.
    assert np.all(np.asarray(occluded.region_min_xy_m) >= BOX_MIN)
    assert np.all(np.asarray(occluded.region_max_xy_m) <= BOX_MAX)


def test_mask_touching_the_image_border_is_not_measured():
    depth, mask, pose = _render()
    clipped = mask.copy()
    clipped[:, 0] = True
    with pytest.raises(ValueError, match="border"):
        measure_support_top(_capture(depth, pose), clipped, maximum_depth_m=2.)


def test_too_small_visible_top_is_rejected():
    depth, mask, pose = _render(occluder=(0, 0, 150, HEIGHT, 0.2))
    tiny = mask.copy()
    rows, cols = np.nonzero(tiny)
    keep = cols < cols.min() + 6
    tiny[:] = False
    tiny[rows[keep], cols[keep]] = True
    with pytest.raises(ValueError):
        measure_support_top(_capture(depth, pose), tiny, maximum_depth_m=2.)

"""Depth-only support top measurement on a tilted top (handover I6, review b3).

The release planner treats ``top_z_m`` as the highest support point under the
certified footprint: the highest certified corner of the fitted plane plus the
plane residual. On a level box every corner has the same height, so the level
tests cannot tell that rule from "the plane height at the center". These tests
tilt the top inside the 1 degree near-horizontal invariant and check the rule
corner by corner, check that a resting object's rim still does not lift it,
and pin that a clearly inclined top is refused by the default path (inclined
supports need their own measurement path, handover S3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_support_surface_measurement import (
    BOX_MAX,
    BOX_MIN,
    HEIGHT,
    INTRINSICS,
    TABLE_Z,
    TOP_Z,
    WIDTH,
    _capture,
    _dilated,
    _look_at,
)

from agent_gateway.profiles.measured_support import (
    MAXIMUM_PLANE_RESIDUAL_M,
    measure_support_tops,
)

pytestmark = [pytest.mark.unit]

CENTER_X = float(BOX_MIN[0] + BOX_MAX[0]) / 2.


def _plane_z(x, tilt_deg):
    """Top surface height: through (center, TOP_Z), rising along +x."""
    return TOP_Z + math.tan(math.radians(tilt_deg)) * (np.asarray(x) - CENTER_X)


def _render_tilted(tilt_deg, *, eye=(0.045, -0.25, 0.35), resting=None):
    """Ray-cast a tilted box top over a table; return depth, top mask, camera pose.

    ``resting`` = (x0, y0, x1, y1, height) puts a block on the top (cap and the
    three faces the camera can see).
    """
    pose = _look_at(eye, (CENTER_X, 0.045, TOP_Z))
    v, u = np.mgrid[0:HEIGHT, 0:WIDTH].astype(float)
    rays = np.stack(((u - INTRINSICS["cx"]) / INTRINSICS["fx"],
                     (v - INTRINSICS["cy"]) / INTRINSICS["fy"], np.ones_like(u)), axis=-1)
    world = rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    slope = math.tan(math.radians(tilt_deg))
    depth = np.full((HEIGHT, WIDTH), np.inf)
    mask = np.zeros((HEIGHT, WIDTH), dtype=bool)

    def hit(t, inside, on_top):
        nonlocal depth, mask
        better = inside & (t > 0) & (t < depth)
        depth = np.where(better, t, depth)
        mask = np.where(better, on_top, mask)

    with np.errstate(divide="ignore", invalid="ignore"):
        # z = TOP_Z + slope * (x - CENTER_X)  <=>  n . p = c with n = (-slope, 0, 1).
        normal = np.array([-slope, 0., 1.])
        c = TOP_Z - slope * CENTER_X
        t_top = (c - origin @ normal) / (world @ normal)
        p = origin + t_top[..., None] * world
        hit(t_top, (p[..., 0] >= BOX_MIN[0]) & (p[..., 0] <= BOX_MAX[0])
            & (p[..., 1] >= BOX_MIN[1]) & (p[..., 1] <= BOX_MAX[1]), True)
        t_table = (TABLE_Z - origin[2]) / world[..., 2]
        hit(t_table, np.ones_like(t_table, dtype=bool), False)
        if resting is not None:
            x0, y0, x1, y1, height = resting
            upper = float(_plane_z(x1, tilt_deg)) + height
            t_cap = (upper - origin[2]) / world[..., 2]
            p = origin + t_cap[..., None] * world
            hit(t_cap, (p[..., 0] >= x0) & (p[..., 0] <= x1)
                & (p[..., 1] >= y0) & (p[..., 1] <= y1), False)
            # Its front and side faces, from the tilted top up to the cap.
            for plane, axis in ((y0, 1), (x0, 0), (x1, 0)):
                t_side = (plane - origin[axis]) / world[..., axis]
                p = origin + t_side[..., None] * world
                other = 1 - axis
                low, high = ((x0, x1) if axis == 1 else (y0, y1))
                hit(t_side, (p[..., other] >= low) & (p[..., other] <= high)
                    & (p[..., 2] >= _plane_z(p[..., 0], tilt_deg)) & (p[..., 2] <= upper), False)
    depth = np.where(np.isfinite(depth), depth, 0.).astype(np.float32)
    return depth, mask, pose


def _corner_heights(top, tilt_deg):
    xs = (top.region_min_xy_m[0], top.region_max_xy_m[0])
    return [float(_plane_z(x, tilt_deg)) for x in xs for _ in (0, 1)]


@pytest.mark.parametrize("tilt_deg", [0.5, 0.8, -0.8])
def test_top_height_is_the_highest_certified_corner_plus_the_residual(tilt_deg):
    depth, mask, pose = _render_tilted(tilt_deg)
    (top,) = measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2.)
    assert top.tilt_deg == pytest.approx(abs(tilt_deg), abs=0.15)
    corners = _corner_heights(top, tilt_deg)
    expected = max(corners) + MAXIMUM_PLANE_RESIDUAL_M
    assert top.top_z_m == pytest.approx(expected, abs=0.0005)
    # The rule is not the plane height at the region center: on this tilt the
    # two differ by more than the tolerance above.
    center = float(_plane_z((top.region_min_xy_m[0] + top.region_max_xy_m[0]) / 2., tilt_deg))
    assert top.top_z_m - (center + MAXIMUM_PLANE_RESIDUAL_M) > 0.0003
    # Every certified corner lies below the reported top.
    assert all(z <= top.top_z_m for z in corners)


def test_a_resting_block_rim_does_not_lift_a_tilted_top():
    tilt = 0.8
    block = (0.0, 0.05, 0.04, 0.09, 0.04)
    depth, mask, pose = _render_tilted(tilt, resting=block)
    clean_depth, _clean_mask, _ = _render_tilted(tilt)
    rim = (np.abs(depth - clean_depth) > 1e-6) & _dilated(mask, 1)
    assert 0 < rim.sum() < 0.02 * mask.sum()
    tops = measure_support_tops(_capture(depth, pose), mask | rim, maximum_depth_m=2., limit=3)
    assert tops
    for top in tops:
        highest = max(_corner_heights(top, tilt)) + MAXIMUM_PLANE_RESIDUAL_M
        assert top.top_z_m == pytest.approx(highest, abs=0.0005)
        assert top.top_z_m < TOP_Z + 0.004


@pytest.mark.parametrize("tilt_deg", [5., 10., 20.])
def test_an_inclined_top_is_refused_by_the_default_path(tilt_deg):
    depth, mask, pose = _render_tilted(tilt_deg)
    with pytest.raises(ValueError):
        measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2.)

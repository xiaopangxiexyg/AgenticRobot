"""Depth-only measurement of an inclined support top (handover S3).

The default path keeps refusing a top tilted more than 1 degree. With
``allow_inclined`` the near-horizontal fits are still tried first (a level top
measures exactly as before); a top they refuse is fitted as one inclined plane
of at most 30 degrees whose height at the certified region's center, plus the
plane residual, is ``top_z_m``.
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
    _look_at,
    _render,
)

from agent_gateway.profiles.measured_support import (
    INCLINED_SUPPORT_TOP_MEASUREMENT_ID,
    MAXIMUM_PLANE_RESIDUAL_M,
    SUPPORT_TOP_MEASUREMENT_ID,
    measure_support_tops,
)

pytestmark = [pytest.mark.unit]

CENTER_X = float(BOX_MIN[0] + BOX_MAX[0]) / 2.


def _plane_z(x, tilt_deg):
    return TOP_Z + math.tan(math.radians(tilt_deg)) * (np.asarray(x) - CENTER_X)


def _render_inclined(tilt_deg, *, eye=(0.045, -0.25, 0.35), resting=None):
    """Ray-cast an inclined board top over a table (rising along +x)."""
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
        normal = np.array([-slope, 0., 1.])
        t_top = ((TOP_Z - slope * CENTER_X) - origin @ normal) / (world @ normal)
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
            for plane, axis in ((y0, 1), (x0, 0), (x1, 0)):
                t_side = (plane - origin[axis]) / world[..., axis]
                p = origin + t_side[..., None] * world
                other = 1 - axis
                low, high = ((x0, x1) if axis == 1 else (y0, y1))
                hit(t_side, (p[..., other] >= low) & (p[..., other] <= high)
                    & (p[..., 2] >= _plane_z(p[..., 0], tilt_deg)) & (p[..., 2] <= upper), False)
    depth = np.where(np.isfinite(depth), depth, 0.).astype(np.float32)
    return depth, mask, pose


def test_measurement_identities_are_distinct():
    assert INCLINED_SUPPORT_TOP_MEASUREMENT_ID != SUPPORT_TOP_MEASUREMENT_ID


@pytest.mark.parametrize("tilt_deg", [10., 20.])
def test_the_default_path_still_refuses_an_inclined_top(tilt_deg):
    depth, mask, pose = _render_inclined(tilt_deg)
    with pytest.raises(ValueError):
        measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2.)


@pytest.mark.parametrize("tilt_deg", [10., 20.])
def test_an_inclined_top_is_measured_when_allowed(tilt_deg):
    depth, mask, pose = _render_inclined(tilt_deg)
    (top,) = measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2., allow_inclined=True)
    assert top.inclined is True
    assert top.tilt_deg == pytest.approx(tilt_deg, abs=0.3)
    center = (np.asarray(top.region_min_xy_m) + top.region_max_xy_m) / 2.
    assert top.top_z_m == pytest.approx(float(_plane_z(center[0], tilt_deg)) + MAXIMUM_PLANE_RESIDUAL_M,
                                        abs=0.0007)
    low, high = np.asarray(top.region_min_xy_m), np.asarray(top.region_max_xy_m)
    assert np.all(low >= BOX_MIN) and np.all(high <= BOX_MAX) and np.all(high - low > 0.04)
    observation = top.observation(support_id="green_board", reference_frame_id="station",
                                  scene_revision=0, measured_at=1., evidence_ref="test:board")
    assert observation.maximum_tilt_deg == 30.
    normal = np.asarray(observation.normal_reference_xyz)
    assert math.degrees(math.acos(normal[2])) == pytest.approx(tilt_deg, abs=0.3)
    assert normal[0] < 0.                      # the top rises along +x, its normal leans back


def test_a_level_top_measures_exactly_as_before_when_inclines_are_allowed():
    depth, mask, pose = _render()
    default = measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2., limit=3)
    allowed = measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2., limit=3,
                                   allow_inclined=True)
    assert allowed == default
    assert all(top.inclined is False for top in allowed)
    assert allowed[0].observation(support_id="green_box", reference_frame_id="station",
                                  scene_revision=0, measured_at=1.,
                                  evidence_ref="test:box").maximum_tilt_deg == 1.


def test_a_top_steeper_than_the_profile_is_refused():
    depth, mask, pose = _render_inclined(35.)
    with pytest.raises(ValueError):
        measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2., allow_inclined=True)


def test_an_object_on_the_incline_does_not_carry_the_plane():
    tilt = 15.
    block = (0.0, 0.05, 0.04, 0.09, 0.04)
    depth, mask, pose = _render_inclined(tilt, resting=block)
    tops = measure_support_tops(_capture(depth, pose), mask, maximum_depth_m=2., limit=3,
                                allow_inclined=True)
    assert tops
    for top in tops:
        assert top.tilt_deg == pytest.approx(tilt, abs=0.3)
        center = (np.asarray(top.region_min_xy_m) + top.region_max_xy_m) / 2.
        assert top.top_z_m == pytest.approx(float(_plane_z(center[0], tilt)) + MAXIMUM_PLANE_RESIDUAL_M,
                                            abs=0.0007)

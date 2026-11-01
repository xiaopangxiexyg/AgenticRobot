"""The real place's support measurement: the footprint+4 cm window, else the footprint+1 cm window.

Synthetic holding-wrist depth of a downward camera over the table (z = 0) and one box top; no device.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import object_motion
from agent_gateway.profiles.real.object_motion import measure_place_support

W, H, F = 640, 480, 400.0
TOP = 0.05
CUBE = 0.04
CAMERA_HEIGHT = TOP + 0.2


def _view(box_low, box_high, *, camera_xy=(0.0, 0.0), role="wrist_right"):
    """Depth of a downward pinhole camera over the table and one box top, as a capture."""
    v, u = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = (u - W / 2.) / F, (v - H / 2.) / F
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1., -1., -1.])
    pose[:3, 3] = (camera_xy[0], camera_xy[1], CAMERA_HEIGHT)

    def hit(height):
        distance = CAMERA_HEIGHT - height
        return distance, camera_xy[0] + distance * dx, camera_xy[1] - distance * dy

    depth, _, _ = hit(0.)
    distance, x, y = hit(TOP)
    on_box = (x >= box_low[0]) & (x <= box_high[0]) & (y >= box_low[1]) & (y <= box_high[1])
    depth = np.where(on_box, distance, depth)
    depth = depth + np.random.default_rng(0).normal(0., 0.001, depth.shape)
    return SimpleNamespace(role=role, depth_m=depth, intrinsics={"fx": F, "fy": F, "cx": W / 2., "cy": H / 2.},
                           source_from_camera=tuple(pose.reshape(-1)))


def _measure(view, footprint_low, footprint_high):
    return measure_place_support(captures=(view,), side="right", footprint_min_xy=footprint_low,
                                 footprint_max_xy=footprint_high, below_z_m=TOP + 0.06)


def _footprint(centre):
    centre = np.asarray(centre, dtype=float)
    return centre - CUBE / 2., centre + CUBE / 2.


def test_a_support_larger_than_the_wide_window_is_measured_in_it():
    # A 20 cm top under a 4 cm footprint: the footprint + 4 cm window is one plane.
    fitted, stats = _measure(_view((-0.10, -0.10), (0.10, 0.10)), *_footprint((0., 0.)))
    assert fitted is not None and stats["support_window"] == "footprint_plus_4cm"
    anchor, normal, low, high = fitted
    assert anchor[2] == pytest.approx(TOP, abs=0.002) and normal[2] > 0.999
    np.testing.assert_allclose(high - low, CUBE + 2 * object_motion.REAL_OBJECT_MOTION_SUPPORT_MARGIN_M)


def test_a_small_support_is_measured_in_the_footprint_plus_1cm_window():
    # A 7 cm top: the wide window reaches the table 5 cm below, the 1 cm window stays on the top.
    fitted, stats = _measure(_view((-0.035, -0.035), (0.035, 0.035)), *_footprint((0., 0.)))
    assert fitted is not None and stats["support_window"] == "footprint_plus_1cm"
    wide = stats["wide_window"]
    assert ("dropoff" in wide and not wide["dropoff"]["covers_window"]) or (
        wide["plane_residual_p95_m"] > object_motion.REAL_OBJECT_MOTION_SUPPORT_MAX_RESIDUAL_M)
    assert stats["dropoff"]["points_below_plane"] == 0
    anchor, normal, low, high = fitted
    assert anchor[2] == pytest.approx(TOP, abs=0.002) and normal[2] > 0.999
    np.testing.assert_allclose(high - low, CUBE + 2 * object_motion.REAL_OBJECT_MOTION_SMALL_SUPPORT_MARGIN_M)


@pytest.mark.parametrize("centre", [(0.03, 0.), (0., -0.03)])
def test_a_footprint_over_the_edge_of_a_small_support_is_refused(centre):
    # The 7 cm top again, the footprint hanging 1.5 cm over its edge: neither window is one plane.
    fitted, stats = _measure(_view((-0.035, -0.035), (0.035, 0.035)), *_footprint(centre))
    assert fitted is None and stats["support_window"] is None
    small = stats["small_window"]
    assert ("dropoff" in small and not small["dropoff"]["covers_window"]) or (
        small["plane_residual_p95_m"] > object_motion.REAL_OBJECT_MOTION_SUPPORT_MAX_RESIDUAL_M)


def test_the_surface_beyond_an_edge_refuses_a_window_whose_trimmed_fit_passes():
    # The footprint 3 mm over the 7 cm top's edge: the trimmed fit of the 1 cm window still
    # finds the top, but the table that shows beyond the edge lies below it. (A strip right
    # past the edge hides behind the top from a camera over it; the release query certifies
    # the whole footprint before the place.)
    fitted, stats = _measure(_view((-0.035, -0.035), (0.035, 0.035)), *_footprint((0.018, 0.)))
    assert fitted is None and stats["support_window"] is None
    small = stats["small_window"]
    assert small["plane_residual_p95_m"] <= object_motion.REAL_OBJECT_MOTION_SUPPORT_MAX_RESIDUAL_M
    assert small["dropoff"]["covers_window"] is False and small["dropoff"]["points_below_plane"] > 0


def test_a_large_support_with_an_edge_inside_the_wide_window_uses_the_small_window():
    # A 14 x 20 cm top whose edge is 2 cm beyond the footprint: the wide window crosses it.
    fitted, stats = _measure(_view((-0.10, -0.10), (0.04, 0.10)), *_footprint((0., 0.)))
    assert fitted is not None and stats["support_window"] == "footprint_plus_1cm"
    assert stats["wide_window"]["dropoff"]["covers_window"] is False
    assert fitted[0][2] == pytest.approx(TOP, abs=0.002)


def test_a_support_smaller_than_the_footprint_is_not_taken_for_the_support():
    # A 3 cm top under a 4 cm footprint (in reality hidden by the held object): the window's
    # plane is the table below it, so a place 1 cm above the small top is refused for its gap.
    fitted, stats = _measure(_view((-0.015, -0.015), (0.015, 0.015)), *_footprint((0., 0.)))
    assert fitted is not None and stats["support_window"] == "footprint_plus_4cm"
    assert fitted[0][2] == pytest.approx(0., abs=0.003)
    assert TOP + 0.01 - fitted[0][2] > object_motion.REAL_OBJECT_MOTION_PLACE_CONTACT_GAP_M


def test_only_the_holding_wrist_view_is_fitted():
    # Another role's view never measures the support.
    fitted, stats = _measure(_view((-0.035, -0.035), (0.035, 0.035), role="main"), *_footprint((0., 0.)))
    assert fitted is None and stats["reason"] == "wrist_camera_view_missing"

"""Release planning onto an inclined support (handover S3/S4), geometry only.

A support observation keeps the 1 degree near-horizontal invariant unless its
producer declares an inclined bound (at most 30 degrees). On an inclined top
the existing normal-aware math keeps the lowest held-box corner at the
requested clearance along the normal; opt-in leveling turns the held box so
its axis nearest the normal lies on it, so the object lands flat.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from test_object_transport import observation, pose, support

from robot_motion.object_transport import (
    MAXIMUM_SUPPORT_TILT_DEG,
    NoReleaseCandidates,
    SupportSurfaceObservation,
    plan_release_approach_candidates,
)

pytestmark = [pytest.mark.unit]

HALF = .0175
CORNERS = np.asarray([(x, y, z) for x in (-HALF, HALF) for y in (-HALF, HALF) for z in (-HALF, HALF)])
OPTIONS = {"now": 10.02, "clearance_m": .006, "minimum_planning_clearance_m": .004,
           "maximum_clearance_m": .05, "lateral_margin_m": .005}


def _normal(tilt_deg, yaw_deg=0.):
    tilt, yaw = math.radians(tilt_deg), math.radians(yaw_deg)
    return (math.sin(tilt) * math.cos(yaw), math.sin(tilt) * math.sin(yaw), math.cos(tilt))


def _inclined(tilt_deg, **changes):
    values = {"support_id": "green_board", "top_z_m": .03,
              "normal_reference_xyz": _normal(tilt_deg), "region_min_xy_m": (-.41, -.02),
              "region_max_xy_m": (-.33, .06), "maximum_tilt_deg": 30.}
    values.update(changes)
    return support(**values)


def _held():
    return observation(box_size_m=(.035, .035, .035),
                       reference_from_object=pose((-.285, -.03, .0766), (1., -1., 20.)))


def _lowest_gap(goal, surface):
    normal = np.asarray(surface.normal_reference_xyz)
    center = (np.asarray(surface.region_min_xy_m) + surface.region_max_xy_m) / 2.
    plane_point = np.r_[center, surface.top_z_m]
    world = CORNERS @ goal[:3, :3].T + goal[:3, 3]
    return float(np.min((world - plane_point) @ normal))


def _tilt_from_normal_deg(goal, surface):
    normal = np.asarray(surface.normal_reference_xyz)
    return math.degrees(math.acos(min(1., float(np.max(np.abs(goal[:3, :3].T @ normal))))))


def test_the_default_bound_keeps_the_near_horizontal_invariant():
    assert support().maximum_tilt_deg == 1.
    with pytest.raises(ValueError, match="near-horizontal"):
        support(normal_reference_xyz=_normal(5.))
    for bound in (0., -1., MAXIMUM_SUPPORT_TILT_DEG + 1., True, float("nan")):
        with pytest.raises(ValueError):
            _inclined(10., maximum_tilt_deg=bound)
    with pytest.raises(ValueError):
        _inclined(35.)                       # steeper than its declared bound
    assert isinstance(_inclined(20.), SupportSurfaceObservation)


@pytest.mark.parametrize("tilt_deg", [10., 20.])
def test_a_held_orientation_release_keeps_the_lowest_corner_at_the_clearance(tilt_deg):
    surface = _inclined(tilt_deg)
    candidates = plan_release_approach_candidates(_held(), surface, **OPTIONS)
    assert candidates and all(item.leveling_rad == 0. for item in candidates)
    for item in candidates:
        assert _lowest_gap(item.proposal.object_goal, surface) == pytest.approx(.006, abs=1e-9)
        # The held orientation is kept: tilted from the incline by about its angle.
        assert _tilt_from_normal_deg(item.proposal.object_goal, surface) > tilt_deg - 3.


@pytest.mark.parametrize("tilt_deg, yaw_deg", [(10., 0.), (20., 0.), (20., 90.), (15., -35.)])
def test_a_leveled_release_lies_flat_on_the_incline(tilt_deg, yaw_deg):
    surface = _inclined(tilt_deg, normal_reference_xyz=_normal(tilt_deg, yaw_deg))
    held = _held()
    leveled = plan_release_approach_candidates(held, surface, level_to_support=True,
                                               include_nearest_to_held=True, **OPTIONS)
    assert leveled
    region_min, region_max = np.asarray(surface.region_min_xy_m), np.asarray(surface.region_max_xy_m)
    for item in leveled:
        goal = item.proposal.object_goal
        assert _tilt_from_normal_deg(goal, surface) == pytest.approx(0., abs=1e-6)
        assert _lowest_gap(goal, surface) == pytest.approx(.006, abs=1e-9)
        # The turn is the smallest one: about the incline (plus the held tilt).
        assert abs(math.degrees(item.leveling_rad) - tilt_deg) < 3.
        world = CORNERS @ goal[:3, :3].T + goal[:3, 3]
        assert np.all(world[:, :2].min(axis=0) >= region_min + .005 - 1e-12)
        assert np.all(world[:, :2].max(axis=0) <= region_max - .005 + 1e-12)
        np.testing.assert_allclose(item.proposal.tool_goal @ held.tool_from_object, goal, atol=1e-12)


def test_leveling_on_a_level_support_only_removes_the_held_tilt():
    surface = support(region_min_xy_m=(-.41, -.02), region_max_xy_m=(-.33, .06))
    (first, *_) = plan_release_approach_candidates(_held(), surface, level_to_support=True, **OPTIONS)
    assert math.degrees(first.leveling_rad) == pytest.approx(math.hypot(1., 1.), abs=.2)
    assert _tilt_from_normal_deg(first.proposal.object_goal, surface) == pytest.approx(0., abs=1e-6)


def test_leveling_is_explicit_and_excludes_a_lean():
    surface = _inclined(10.)
    with pytest.raises(TypeError, match="explicitly boolean"):
        plan_release_approach_candidates(_held(), surface, level_to_support=1, **OPTIONS)
    with pytest.raises(ValueError, match="not leaned"):
        plan_release_approach_candidates(_held(), surface, level_to_support=True, lean_rad=.1,
                                         lean_toward_xy_m=(0., -.26), **OPTIONS)


def test_a_leveled_footprint_that_does_not_fit_is_typed():
    with pytest.raises(NoReleaseCandidates):
        plan_release_approach_candidates(
            _held(), _inclined(20., region_min_xy_m=(-.38, 0.), region_max_xy_m=(-.35, .03)),
            level_to_support=True, **OPTIONS)

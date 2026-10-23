"""Closing-window evidence is neither an AABB nor proof of contact mechanics."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from deployments.perception.dmtac_grasp_overlay import (
    reviewed_graspgenx_dmtac_solid_geometry,
)
from deployments.perception.grasp_first_path import (
    FirstGraspPathPolicy,
    grasp_capture_support_cost,
)
from robot_motion.grasp_visible_scene import (
    estimate_grasp_surface_normals,
    parallel_jaw_capture_support,
    parallel_jaw_opposed_capture_support,
    parallel_jaw_pad_height_support,
    parallel_jaw_surface_alignment,
)


def geometry():
    return replace(reviewed_graspgenx_dmtac_solid_geometry(), finger_length_m=.100)


def support(points):
    return parallel_jaw_capture_support(np.asarray(points, dtype=float).reshape(-1, 3),
        geometry=geometry(), opening_m=.0705)


def test_separately_overlapping_axes_do_not_invent_joint_support():
    # Each Y/Z interval overlaps separately, but the same points never do.
    points = np.array([[-.02, .04, .02], [.02, .04, .02],
                       [-.02, 0, .12], [.02, 0, .12]])
    low, high = points.min(0), points.max(0)
    assert low[1] <= .015 and high[1] >= -.015
    assert low[2] <= .071 and high[2] >= -.029
    assert support(points) == (0, 0.0)


def test_grazing_tip_has_less_support_than_interior_without_requiring_contact():
    count, grazing = support([[-.02, 0, .069], [.02, 0, .069]])
    _, interior = support([[-.02, 0, .03], [.02, 0, .03]])
    assert count == 2 and grazing == pytest.approx(.1) and interior == 1
    policy = FirstGraspPathPolicy()
    assert grasp_capture_support_cost(SimpleNamespace(capture_support_score=grazing), policy=policy) > (
        grasp_capture_support_cost(SimpleNamespace(capture_support_score=interior), policy=policy))
    assert grasp_capture_support_cost(SimpleNamespace(), policy=policy) == 0


def test_tall_object_can_have_local_support_without_centroid_containment():
    points = [[-.02, 0, .02], [.02, 0, .02], [-.02, .30, .02], [.02, .30, .02]]
    assert np.mean(points, axis=0)[1] > geometry().pad_height_m / 2
    assert support(points) == (2, 1.0)


def height_support(points):
    return parallel_jaw_pad_height_support(np.asarray(points, dtype=float).reshape(-1, 3),
        geometry=geometry(), opening_m=.0705)


def test_top_rim_is_not_good_pad_height_support_despite_full_axial_score():
    ys = np.linspace(.012, .055, 100)
    points = np.column_stack((np.tile((-.025, .025), 50), ys, np.full(100, .025)))
    assert support(points)[1] == 1.
    evidence = height_support(points)
    assert evidence.score < .1
    assert .015 < evidence.center_shift_m < .04
    centered = points.copy()
    centered[:, 1] -= evidence.center_shift_m
    assert height_support(centered).score == pytest.approx(1.)


def test_height_support_is_local_and_does_not_require_the_whole_centroid():
    assert height_support([[-.02, 0, .02], [.02, 0, .02],
                           [-.02, .30, .02], [.02, .30, .02]]).score == 1.
    ys = np.linspace(-.15, .15, 601)
    uniform = np.column_stack((np.zeros(601), ys, np.full(601, .02)))
    evidence = height_support(uniform)
    assert evidence.score == pytest.approx(1.)
    assert abs(evidence.center_shift_m) < 1e-12


def test_height_support_matches_brute_force_and_density_replication():
    rng = np.random.default_rng(42)
    points = rng.uniform((-.03, -.06, -.02), (.03, .09, .07), size=(100,3))
    evidence = height_support(points)
    brute = np.sum(np.clip(1-np.abs(points[:,1,None]-points[None,:,1])/.015,0,1),axis=0)
    assert evidence.best_weighted_points == pytest.approx(max(brute))
    duplicate = height_support(np.repeat(points,3,axis=0))
    assert duplicate.score == pytest.approx(evidence.score)
    assert duplicate.center_shift_m == evidence.center_shift_m
    assert height_support([]).score == 0.


def test_collision_tip_extension_is_not_inference_reach():
    assert support([[-.02, 0, .0715], [.02, 0, .0715]]) == (0, 0.0)
    assert support([]) == (0, 0.0)


def test_opposed_support_reveals_a_shallow_side_hidden_by_central_points():
    x = np.linspace(-.03, .03, 101)
    # A tilted top face extends beyond one fingertip. The old in-window
    # median stays high by excluding those missed points before scoring.
    z = .055 + (x+.03)*.5
    points = np.column_stack((x, np.zeros_like(x), z))
    assert support(points)[1] > .3
    evidence = parallel_jaw_opposed_capture_support(points, geometry=geometry(), opening_m=.0705)
    assert evidence.side_scores[0] > .5
    assert evidence.side_scores[1] == 0.
    assert evidence.score == 0.
    centered = points.copy(); centered[:, 2] = .045
    assert parallel_jaw_opposed_capture_support(centered, geometry=geometry(), opening_m=.0705).score == 1.
    policy = FirstGraspPathPolicy()
    assert grasp_capture_support_cost(SimpleNamespace(opposed_capture_support_score=0.), policy=policy) > (
        grasp_capture_support_cost(SimpleNamespace(opposed_capture_support_score=1.), policy=policy))


def test_opposed_support_is_symmetric_local_and_not_a_missing_view_gate():
    z = np.linspace(-.2, .4, 601)
    points = np.array([(x, 0., depth) for x in (-.025, .025) for depth in z])
    f = lambda p: parallel_jaw_opposed_capture_support(p, geometry=geometry(), opening_m=.0705)
    evidence = f(points)
    assert evidence.score == 1.  # Local pinch, not whole-object coverage.
    exchanged = points.copy(); exchanged[:, :2] *= -1
    assert f(exchanged).score == evidence.score
    assert f(np.repeat(points, 3, axis=0)).score == evidence.score
    assert f(np.empty((0, 3))).score == 0.
    assert f(np.array([[0., 0., .03]])).score == 0.
    # No new hard rejection: an unobserved side is only a search preference.
    from deployments.perception.grasp_first_path import grasp_capture_height_acceptable
    assert grasp_capture_height_acceptable(SimpleNamespace(opposed_capture_support_score=0.),
                                         policy=FirstGraspPathPolicy())


def test_central_density_cannot_hide_observed_missed_closing_edge():
    f = lambda p: parallel_jaw_opposed_capture_support(p, geometry=geometry(), opening_m=.0705)
    edges = np.array([[-.025, 0., .045]]*10 + [[.025, 0., .080]]*10)
    central = np.column_stack((np.linspace(-.003, .003, 80), np.zeros(80), np.full(80, .045)))
    assert f(edges).score == 0.
    assert f(np.vstack((edges, central))).score == 0.
    # The same failure must not be hidden on a continuous tilted surface.
    x = np.linspace(-.025, .025, 101)
    slope = np.column_stack((x, np.zeros_like(x), .055+x))
    dense_x = np.linspace(-.003, .003, 1000)
    dense = np.column_stack((dense_x, np.zeros_like(dense_x), .055+dense_x))
    assert f(np.vstack((slope, dense))).score == f(slope).score


def test_sparse_non_saturated_opposed_support_is_replication_invariant():
    points = np.array([(x, y, .061) for x in (-.025, .025) for y in (-.001, 0., .001)]
                      + [(x, 0., .080) for x in (-.025, .025)])
    f = lambda p: parallel_jaw_opposed_capture_support(p, geometry=geometry(), opening_m=.0705)
    original = f(points)
    assert 0 < original.score < 1.
    for times in (2, 3, 5):
        assert f(np.repeat(points, times, axis=0)).score == original.score


def test_single_extreme_point_cannot_manufacture_an_opposed_surface():
    ys = np.linspace(-.005, .005, 100)
    points = np.vstack((np.column_stack((np.full(100, -.005), ys, np.full(100, .045))),
                        np.column_stack((np.full(100, .005), ys, np.full(100, .080)))))
    f = lambda p: parallel_jaw_opposed_capture_support(p, geometry=geometry(), opening_m=.0705)
    assert f(points).score == 0.
    contaminated = np.vstack((points, [[.034, 0., .045]]))
    assert f(contaminated).score == 0.
    assert f(np.repeat(contaminated, 3, axis=0)).score == 0.
    exchanged = contaminated.copy(); exchanged[:, :2] *= -1
    assert f(exchanged).score == 0.
    # Replication is not independent local support, and unobserved support
    # remains a soft zero rather than a newly introduced rejection.
    sparse = np.array([[-.025, 0., .03], [.025, 0., .03]])
    assert f(np.repeat(sparse, 10, axis=0)).score == 0.


def _surface_patches():
    return np.array([(x, y, z) for x in (-.022, .022)
                     for y in np.linspace(-.008, .008, 9)
                     for z in np.linspace(.015, .045, 13)])


def _alignment(points, geometry_override=None):
    g = geometry() if geometry_override is None else geometry_override
    normals, confidence = estimate_grasp_surface_normals(points, radius_m=g.pad_height_m/2)
    return parallel_jaw_surface_alignment(points, normals, confidence,
                                         geometry=g, opening_m=g.maximum_opening_m)


def test_surface_alignment_prefers_observed_contact_direction_not_world_horizontal():
    from scipy.spatial.transform import Rotation
    points = _surface_patches()
    normals, confidence = estimate_grasp_surface_normals(points, radius_m=.015)
    aligned = parallel_jaw_surface_alignment(points, normals, confidence,
                                            geometry=geometry(), opening_m=.0705)
    assert aligned.observed_sides == 2 and aligned.score > .5
    # Rotate the hand, not the observed object. This generic mismatch is not
    # a fixed-world inclination rule and applies to any locally planar patch.
    for angle in (12., 27., 43.):
        rotation = Rotation.from_euler('z', angle, degrees=True).as_matrix()
        tilted = parallel_jaw_surface_alignment(points @ rotation, normals @ rotation,
            confidence, geometry=geometry(), opening_m=.0705)
        assert tilted.score < aligned.score
    # A tilted object and matching hand retain exactly the same local evidence.
    world_rotation = Rotation.from_rotvec([.7, -.4, .9]).as_matrix()
    rotated_normals, rotated_confidence = estimate_grasp_surface_normals(
        points @ world_rotation.T + [1., -2., .8], radius_m=.015)
    result = parallel_jaw_surface_alignment(
        points, rotated_normals @ world_rotation, rotated_confidence,
        geometry=geometry(), opening_m=.0705)
    assert result.score == pytest.approx(aligned.score, abs=1e-10)


def test_surface_alignment_is_scale_replication_and_finger_exchange_invariant():
    points = _surface_patches()
    baseline = _alignment(points)
    assert _alignment(np.repeat(points, 4, axis=0)).score == pytest.approx(baseline.score)
    exchanged = points * [-1., -1., 1.]
    assert _alignment(exchanged).score == pytest.approx(baseline.score)
    g = geometry()
    for scale in (.4, 2.1):
        scaled = replace(g, palm_z_m=g.palm_z_m*scale, finger_length_m=g.finger_length_m*scale,
            finger_thickness_m=g.finger_thickness_m*scale, pad_height_m=g.pad_height_m*scale,
            maximum_opening_m=g.maximum_opening_m*scale)
        assert _alignment(points*scale, scaled).score == pytest.approx(baseline.score)


def test_surface_alignment_supports_curvature_and_local_pinches_on_long_objects():
    angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
    cylinder = np.array([(radius*np.cos(a), y, .03+radius*np.sin(a))
                         for a in angles for y in np.linspace(-.01, .01, 11)
                         for radius in (.022,)])
    assert _alignment(cylinder).score > .4
    points = _surface_patches()
    remote = points + [0., .35, .3]
    assert _alignment(np.vstack((points, remote))).score == pytest.approx(_alignment(points).score)


def test_missing_or_sparse_surface_is_unknown_not_a_hard_grasp_rejection():
    from deployments.perception.grasp_first_path import grasp_capture_height_acceptable
    for points in (np.empty((0, 3)), np.array([[0., 0., .02]]),
                   np.repeat([[-.02, 0., .02], [.02, 0., .02]], 20, axis=0)):
        evidence = _alignment(points)
        assert evidence.score == 0 and evidence.observed_sides == 0
    points = _surface_patches()
    one_side = _alignment(points[points[:, 0] < 0])
    assert one_side.observed_sides in (0, 1)  # A single spatial plane is not two opposing surfaces.
    assert grasp_capture_height_acceptable(SimpleNamespace(surface_alignment_score=0.),
                                          policy=FirstGraspPathPolicy())
    assert grasp_capture_support_cost(SimpleNamespace(), policy=FirstGraspPathPolicy()) == 0.


@pytest.mark.parametrize('points', (np.zeros((2, 2)), [[0., np.nan, .01]]))
def test_surface_normal_input_validation(points):
    with pytest.raises(ValueError, match='finite Nx3'):
        estimate_grasp_surface_normals(points, radius_m=.015)


@pytest.mark.parametrize('radius', (True, 0., -.01, np.nan))
def test_surface_normal_radius_validation(radius):
    with pytest.raises(ValueError, match='radius'):
        estimate_grasp_surface_normals(np.zeros((0, 3)), radius_m=radius)


def test_surface_alignment_validates_confidence_and_unit_normals():
    points = _surface_patches()
    normals, confidence = estimate_grasp_surface_normals(points, radius_m=.015)
    for n, c in ((normals*2, confidence), (normals, confidence+2),
                 (normals[:-1], confidence), (normals, confidence[:-1])):
        with pytest.raises(ValueError, match='matching unit normals'):
            parallel_jaw_surface_alignment(points, n, c, geometry=geometry(), opening_m=.0705)


def test_unreliable_extreme_cannot_hide_an_observed_misaligned_surface():
    points = np.array([(x, y, .03) for x in (-.005, .005) for y in (-.002, 0., .002)])
    normals = np.array([[1., 0., 0.]]*3 + [[0., 0., 1.]]*3)
    confidence = np.ones(6)
    evidence = parallel_jaw_surface_alignment(points, normals, confidence,
        geometry=geometry(), opening_m=.0705)
    contaminated = parallel_jaw_surface_alignment(
        np.vstack((points, [[.034, 0., .03]])), np.vstack((normals, [[0., 0., 0.]])),
        np.r_[confidence, 0.], geometry=geometry(), opening_m=.0705)
    assert contaminated.score == evidence.score
    assert contaminated.observed_sides == evidence.observed_sides


def test_isotropic_and_linear_neighbourhoods_do_not_invent_normals():
    isotropic = np.vstack((np.eye(3), -np.eye(3))) * .003
    line = np.column_stack((np.linspace(-.005, .005, 20), np.zeros((20, 2))))
    for points in (isotropic, line):
        normals, confidence = estimate_grasp_surface_normals(points, radius_m=.015)
        np.testing.assert_array_equal(confidence, 0.)
        np.testing.assert_array_equal(normals, 0.)


@pytest.mark.parametrize('outlier_x', [.018, .019, .020])
def test_pca_confident_extreme_cannot_improve_score_by_losing_weak_side(outlier_x):
    left = np.array([[-.004, y, z] for y in np.linspace(-.008, .008, 9)
                     for z in np.linspace(.022, .038, 9)])
    right = np.array([[x, y, .03] for x in (.004, .005, .006)
                      for y in np.linspace(-.008, .008, 9)])
    points = np.vstack((left, right))
    baseline = _alignment(points)
    contaminated = np.vstack((points, [[outlier_x, 0., .03]]))
    _, confidence = estimate_grasp_surface_normals(contaminated, radius_m=.015)
    assert confidence[-1] > .1  # Exercise actual PCA, not a manually zeroed normal.
    result = _alignment(contaminated)
    assert baseline.observed_sides == 2 and result.observed_sides == 1
    # PCA and weighted reductions can differ by one ULP after tie reordering;
    # no material increase may be manufactured by losing the weak side.
    assert result.score <= baseline.score + 1e-12


@pytest.mark.parametrize("points", (np.zeros((2, 2)), [[0, 0, float("nan")]], [[0, 0, float("inf")]]))
def test_invalid_support_points_rejected(points):
    with pytest.raises(ValueError, match="finite Nx3"):
        parallel_jaw_capture_support(points, geometry=geometry(), opening_m=.0705)


@pytest.mark.parametrize("opening", (True, -.01, 0., .071, float("nan")))
def test_invalid_opening_rejected(opening):
    with pytest.raises(ValueError, match="opening"):
        parallel_jaw_capture_support(np.empty((0, 3)), geometry=geometry(), opening_m=opening)


@pytest.mark.parametrize("score", (True, -.1, 1.1, float("nan"), "0.5"))
def test_invalid_candidate_support_rejected(score):
    with pytest.raises(ValueError, match="support"):
        grasp_capture_support_cost(SimpleNamespace(capture_support_score=score), policy=FirstGraspPathPolicy())


@pytest.mark.parametrize("weight", (True, -.01, float("inf")))
def test_invalid_support_weight_rejected(weight):
    with pytest.raises(ValueError, match="capture_support_weight"):
        FirstGraspPathPolicy(capture_support_weight=weight)

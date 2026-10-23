"""Shape-independent observed closure evidence, not a force-closure claim."""

from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.grasp_visible_scene import (
    ParallelJawSolidGeometry,
    estimate_grasp_surface_normals,
    parallel_jaw_pad_height_support,
    parallel_jaw_surface_alignment,
)


GEOMETRY = ParallelJawSolidGeometry(
    palm_z_m=-.024, finger_length_m=.0867, finger_thickness_m=.0199,
    pad_height_m=.030, maximum_opening_m=.0705,
)


def alignment(points, normals, confidence=None, geometry=GEOMETRY):
    return parallel_jaw_surface_alignment(
        np.asarray(points), np.asarray(normals),
        np.ones(len(points)) if confidence is None else confidence,
        geometry=geometry, opening_m=geometry.maximum_opening_m,
    )


def faces(width=.05, height=.05, center_y=0., length=.028):
    points = np.array([(x, y, z) for x in (-width/2, width/2)
                       for y in np.linspace(-height/2, height/2, 31) + center_y
                       for z in np.linspace(.03-length/2, .03+length/2, 15)])
    return points, np.tile([1., 0., 0.], (len(points), 1))


@pytest.mark.parametrize("angle", (15., 40., 65.))
def test_two_edges_of_one_plane_do_not_earn_a_second_contact_bonus(angle):
    normal = np.array([np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0.])
    xs, zs = np.meshgrid(np.linspace(-.03, .03, 81), np.linspace(.02, .04, 20))
    ys = -normal[0] * xs / normal[1]
    points = np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))
    points = points[np.abs(points[:, 1]) <= GEOMETRY.pad_height_m/2]
    result = alignment(points, np.tile(normal, (len(points), 1)))
    assert result.score <= .5 * normal[0]**2 + 1e-10
    assert result.opposition_score == pytest.approx(0., abs=1e-12)


@pytest.mark.parametrize("width,height", ((.006, .008), (.018, .025), (.05, .05), (.065, .15)))
def test_opposed_faces_of_different_sizes_remain_good_local_contacts(width, height):
    points, normals = faces(width, height)
    result = alignment(points, normals)
    assert result.score > .99
    assert result.opposition_score == pytest.approx(1.)
    assert result.pad_height_score > .99
    assert abs(result.center_shift_m) < height/30 + 1e-9


def test_one_observed_face_is_partial_evidence_not_two_contacts_or_a_rejection():
    points, normals = faces()
    single = points[:, 0] < 0
    result = alignment(points[single], normals[single])
    assert result.observed_sides == 1
    assert result.score == pytest.approx(.5)
    assert result.opposition_score == 0.
    from deployments.perception.grasp_first_path import FirstGraspPathPolicy, grasp_capture_height_acceptable
    from types import SimpleNamespace
    assert grasp_capture_height_acceptable(
        SimpleNamespace(surface_alignment_score=result.score), policy=FirstGraspPathPolicy())


@pytest.mark.parametrize("directional_count", (1, 2))
def test_unrelated_normals_cannot_qualify_a_directional_outlier_patch(directional_count):
    points = np.array([(x, y, .03) for x in (-.02, .02)
                       for y in np.linspace(-.004, .004, 10)])
    normals = np.tile([0., 1., 0.], (len(points), 1))
    for start in (0, 10):
        normals[start:start+directional_count] = [1., 0., 0.]
    result = alignment(points, normals)
    assert result.score == 0.
    # Tiny but nonzero directional noise in unrelated normals is not enough
    # to turn the isolated strong normal into a confident contact patch.
    weak = normals.copy()
    unrelated = weak[:, 0] == 0.
    weak[unrelated] = [np.sqrt(.02), np.sqrt(.98), 0.]
    assert alignment(points, weak).score <= .02 + 1e-12


@pytest.mark.parametrize("noise", ("uniform", "two_levels"))
def test_normal_direction_depth_noise_does_not_split_one_face_into_two_contacts(noise):
    points, normals = faces()
    single = points[:, 0] < 0
    points, normals = points[single], normals[single]
    rng = np.random.default_rng(921)
    offsets = (rng.uniform(-.00015, .00015, len(points)) if noise == "uniform"
               else np.where(np.arange(len(points)) % 2, -.00015, .00015))
    points[:, 0] += offsets
    result = alignment(points, normals)
    assert .49 <= result.score <= .56
    assert result.opposition_score <= .12


@pytest.mark.parametrize("width", (.0005, .0015, .003))
def test_unresolved_thin_surfaces_keep_partial_credit_instead_of_a_thickness_gate(width):
    points, normals = faces(width=width)
    result = alignment(points, normals)
    assert result.score >= .5
    from deployments.perception.grasp_first_path import FirstGraspPathPolicy, grasp_capture_height_acceptable
    from types import SimpleNamespace
    assert grasp_capture_height_acceptable(SimpleNamespace(capture_height_support_score=1.,
        contact_height_support_score=result.pad_height_score,
        surface_alignment_score=result.score), policy=FirstGraspPathPolicy())


@pytest.mark.parametrize("radius", (.004, .016, .03))
def test_curved_contacts_and_unequal_normal_signs_keep_opposition(radius):
    angles = np.linspace(0., 2*np.pi, 96, endpoint=False)
    points = np.array([(radius*np.cos(a), y, .03 + radius*np.sin(a))
                       for a in angles for y in np.linspace(-.025, .025, 21)])
    normals = np.array([(np.cos(a), 0., np.sin(a)) for a in angles for _ in range(21)])
    normal_sign = np.where(np.arange(len(normals)) % 3, -1., 1.)[:, None]
    result = alignment(points, normals*normal_sign)
    assert result.score > .6
    assert result.opposition_score > .6


def test_contact_band_uses_observed_closing_faces_not_dense_top_surface():
    side, normals = faces(height=.05, center_y=.024)
    # Dense grazing top face masks the missing body contact in a density score.
    top = np.array([(x, -.001, z) for x in np.linspace(-.025, .025, 121)
                    for z in np.linspace(.016, .044, 71)])
    points = np.vstack((side, top))
    normals = np.vstack((normals, np.tile([0., 1., 0.], (len(top), 1))))
    density = parallel_jaw_pad_height_support(points, geometry=GEOMETRY, opening_m=.0705)
    result = alignment(points, normals)
    assert density.score > .9
    # The band reaches just over half the side's best local support, not the
    # near-perfect coverage manufactured by the dense top face.
    assert .5 < result.pad_height_score < .7
    assert .009 < result.center_shift_m < .039
    shifted = points - [0., result.center_shift_m, 0.]
    improved = alignment(shifted, normals)
    assert improved.pad_height_score > .99
    assert improved.score > result.score


def test_grazing_contact_band_has_a_bounded_observation_driven_recovery():
    points, normals = faces(height=.05, center_y=.035)
    result = alignment(points, normals)
    assert result.pad_height_score < .5
    assert .015 < result.center_shift_m < .04
    assert alignment(points - [0., result.center_shift_m, 0.], normals).score > .99


def test_long_object_local_pinch_does_not_require_global_centroid():
    points, normals = faces(height=.2)
    baseline = alignment(points, normals)
    remote = points + [0., .8, .4]
    result = alignment(np.vstack((points, remote)), np.vstack((normals, normals)))
    assert result.score == pytest.approx(baseline.score)
    assert result.center_shift_m == pytest.approx(baseline.center_shift_m)


def test_local_neck_pinch_is_not_penalized_for_a_denser_distant_body():
    neck, neck_normals = faces(width=.018, height=.02, length=.018)
    body = np.array([(x, y, z) for x in (-.029, .029)
                     for y in np.linspace(.15, .25, 101)
                     for z in np.linspace(.005, .055, 51)])
    body_normals = np.tile([1., 0., 0.], (len(body), 1))
    # Extra distant-body samples must not turn local support into a global
    # object-density contest; a bottle neck or tool handle can be the target.
    baseline = alignment(neck, neck_normals)
    result = alignment(np.vstack((neck, body)), np.vstack((neck_normals, body_normals)))
    assert result.score == pytest.approx(baseline.score)
    assert result.pad_height_score == pytest.approx(baseline.pad_height_score)
    assert result.center_shift_m == pytest.approx(baseline.center_shift_m)


def test_contact_score_is_not_a_world_horizontal_or_fixed_object_size_rule():
    points, normals = faces(width=.032, height=.043)
    baseline = alignment(points, normals)
    rotation = Rotation.from_rotvec([.8, -.6, .4]).as_matrix()
    origin = np.array([.13, -.4, .24])
    world = points @ rotation.T + origin
    local = (world-origin) @ rotation
    result = alignment(local, (normals @ rotation.T) @ rotation)
    assert result.score == pytest.approx(baseline.score)
    for scale in (.3, 1.7):
        g = replace(GEOMETRY, **{name: getattr(GEOMETRY, name)*scale for name in (
            "palm_z_m", "finger_length_m", "finger_thickness_m", "pad_height_m", "maximum_opening_m")})
        assert alignment(points*scale, normals, geometry=g).score == pytest.approx(baseline.score)
    exchanged = alignment(points * [-1., -1., 1.], normals * [-1., -1., 1.])
    replicated = alignment(np.repeat(points, 3, axis=0), np.repeat(normals, 3, axis=0))
    assert exchanged.score == pytest.approx(baseline.score)
    assert replicated.score == pytest.approx(baseline.score)


def test_fast_cost_prefers_one_real_closing_face_over_two_bands_on_a_tilted_top():
    from deployments.perception.grasp_first_path import FirstGraspPathPolicy, grasp_search_preference
    from types import SimpleNamespace

    angle = np.deg2rad(40.)
    normal = np.array([np.sin(angle), np.cos(angle), 0.])
    xs, zs = np.meshgrid(np.linspace(-.015, .015, 41), np.linspace(.02, .04, 15))
    top = np.column_stack((xs.ravel(), (-xs*np.tan(angle)).ravel(), zs.ravel()))
    top_evidence = alignment(top, np.tile(normal, (len(top), 1)))
    body, normals = faces(width=.032, height=.042)
    visible = body[:, 0] < 0
    body_evidence = alignment(body[visible], normals[visible])
    policy = FirstGraspPathPolicy()
    candidate = lambda evidence: SimpleNamespace(surface_alignment_score=evidence.score,
        contact_height_support_score=evidence.pad_height_score)
    assert grasp_search_preference(candidate(body_evidence), .06, policy=policy) < (
        grasp_search_preference(candidate(top_evidence), 0., policy=policy))


def test_weighted_band_matches_brute_force_and_ignores_zero_weight_density():
    rng = np.random.default_rng(83)
    points = rng.uniform((-.03, -.06, 0.), (.03, .09, .05), size=(100, 3))
    weights = rng.uniform(0., 1., 100)
    result = parallel_jaw_pad_height_support(points, geometry=GEOMETRY, opening_m=.0705,
                                             point_weights=weights)
    kernels = np.clip(1-np.abs(points[:, 1, None]-points[None, :, 1])/.015, 0., 1.)
    assert result.best_weighted_points == pytest.approx(max(weights @ kernels))
    assert result.weighted_points == pytest.approx(weights @ np.clip(1-np.abs(points[:, 1])/.015, 0., 1.))
    replicated = parallel_jaw_pad_height_support(np.repeat(points, 3, axis=0), geometry=GEOMETRY,
        opening_m=.0705, point_weights=np.repeat(weights, 3))
    assert replicated.score == pytest.approx(result.score)
    assert replicated.center_shift_m == result.center_shift_m


@pytest.mark.parametrize("shape", ("thin_box", "cylinder", "sphere", "tapered_body"))
def test_actual_pca_contacts_generalize_beyond_a_single_box(shape):
    if shape == "thin_box":
        points, _ = faces(width=.012, height=.018, length=.018)
    elif shape == "tapered_body":
        points = np.array([(sign*(.019 + .15*y), y, z) for sign in (-1., 1.)
                           for y in np.linspace(-.025, .025, 25)
                           for z in np.linspace(.016, .044, 15)])
    elif shape == "cylinder":
        points = np.array([(.018*np.cos(a), y, .03 + .018*np.sin(a))
                           for a in np.linspace(0., 2*np.pi, 56, endpoint=False)
                           for y in np.linspace(-.025, .025, 25)])
    else:
        # Equal-area random directions avoid a duplicated pole and a dense
        # latitude ring pretending to be an additional contact patch.
        rng = np.random.default_rng(246)
        directions = rng.normal(size=(2400, 3))
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        points = .022 * directions + [0., 0., .03]
    normals, confidence = estimate_grasp_surface_normals(points, radius_m=.015)
    result = alignment(points, normals, confidence)
    assert result.score > .2
    assert result.observed_sides == 2
    assert result.opposition_score > .4
    rng = np.random.default_rng(915)
    noisy = points + rng.uniform(-.00015, .00015, size=points.shape)
    n, c = estimate_grasp_surface_normals(noisy, radius_m=.015)
    perturbed = alignment(noisy, n, c)
    assert perturbed.score > .15
    assert abs(perturbed.score-result.score) < .2


@pytest.mark.parametrize("weights", ([1.], [1., np.nan], [1., -1.], [1., 2.]))
def test_weighted_band_rejects_invalid_observation_weights(weights):
    with pytest.raises(ValueError, match="weights"):
        parallel_jaw_pad_height_support(np.zeros((2, 3)), geometry=GEOMETRY,
                                       opening_m=.0705, point_weights=weights)

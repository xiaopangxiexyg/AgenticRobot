from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion import spatial_goal_regions as regions
from robot_motion.spatial_goal_regions import (
    generate_continuation_region_goals,
    generate_relation_region_goals,
)
from robot_motion.spatial_goals import RelationGoalCandidate, generate_relation_goals

pytestmark = pytest.mark.unit


def _parameters(**changes):
    return {
        "current_tcp_pose_xyz_quat_xyzw": [0., 0., 0., 0., 0., 0., 1.],
        "anchor_xyz_m": [.2, -.1, .3],
        "offset_bounds_xyz_m": [[-.04, .04], [-.02, .06], [.1, .2]],
        "offset_frame": "reference",
        "orientation_mode": "align_axis",
        "tool_axis": "z",
        "axis_reference_xyz": [0., 0., -1.],
        "tool_axis_rotation_bounds_rad": [-.8, .6],
        "max_candidates": 8,
        **changes,
    }


def test_center_then_halton_pairs_are_bounded_unit_quaternion_proposals():
    parameters = _parameters()
    candidates = generate_relation_region_goals(**parameters)
    assert isinstance(candidates, tuple)
    assert len(candidates) == 8
    # The first Halton bases are 2, 3, 5, 7; the first draw is the lower corner.
    expected = [
        [0., .02, .15, -.1],
        [-.04, -.02, .1, -.8],
        [0., -.02 + .08 / 3, .12, -.6],
        [-.02, -.02 + .16 / 3, .14, -.4],
    ]
    np.testing.assert_allclose(
        [(*item.offset_xyz_m, item.tool_axis_rotation_rad) for item in candidates[:4]],
        expected, atol=1e-15,
    )
    for candidate in candidates:
        assert isinstance(candidate, RelationGoalCandidate)
        offset = np.array(candidate.offset_xyz_m)
        bounds = np.array(parameters["offset_bounds_xyz_m"])
        assert np.all(bounds[:, 0] <= offset)
        assert np.all(offset <= bounds[:, 1])
        assert -.8 <= candidate.tool_axis_rotation_rad <= .6
        pose = candidate.goal.pose_xyz_quat_xyzw
        assert all(math.isfinite(value) for value in pose)
        assert math.hypot(*pose[3:]) == pytest.approx(1., abs=1e-7)
        np.testing.assert_allclose(np.array(pose[:3]) - parameters["anchor_xyz_m"], offset)
        np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix()[:, 2], [0, 0, -1], atol=1e-7)
        assert candidate.goal.axis_alignment_error_rad < 1e-6


@pytest.mark.parametrize("max_candidates", [1, 8])
def test_point_region_returns_one_candidate(max_candidates):
    parameters = _parameters(
        offset_bounds_xyz_m=((.1, .1), (-.2, -.2), (.3, .3)),
        tool_axis_rotation_bounds_rad=(.7, .7), max_candidates=max_candidates,
    )
    candidate, = generate_relation_region_goals(**parameters)
    expected, = generate_relation_goals(
        current_tcp_pose_xyz_quat_xyzw=parameters["current_tcp_pose_xyz_quat_xyzw"],
        anchor_xyz_m=parameters["anchor_xyz_m"], offsets_xyz_m=[[.1, -.2, .3]],
        offset_frame="reference", orientation_mode="align_axis", tool_axis="z",
        axis_reference_xyz=[0., 0., -1.], tool_axis_rotation_offsets_rad=[.7],
    )
    assert candidate == expected


@pytest.mark.parametrize("active_dimension", [0, 1, 2, 3])
def test_line_region_removes_fixed_dimensions_and_omits_duplicate_midpoint(active_dimension):
    bounds = [[0., 0.], [.2, .2], [.1, .1], [.3, .3]]
    bounds[active_dimension] = [1., 3.]
    candidates = generate_relation_region_goals(**_parameters(
        offset_bounds_xyz_m=bounds[:3], tool_axis_rotation_bounds_rad=bounds[3],
        max_candidates=5,
    ))
    values = [(*item.offset_xyz_m, item.tool_axis_rotation_rad) for item in candidates]
    assert len(values) == 4
    np.testing.assert_allclose([value[active_dimension] for value in values], [2., 1., 1.5, 2.5])
    for value in values:
        for index, (lower, upper) in enumerate(bounds):
            if lower == upper:
                assert value[index] == pytest.approx(lower)


def test_periodic_midpoint_duplicate_is_skipped_without_replacement_draws():
    candidates = generate_relation_region_goals(**_parameters(
        offset_bounds_xyz_m=[[0., 0.]] * 3,
        tool_axis_rotation_bounds_rad=[math.pi, 3 * math.pi], max_candidates=5,
    ))
    assert len(candidates) == 4
    np.testing.assert_allclose(
        [item.tool_axis_rotation_rad for item in candidates],
        [0., -math.pi, -math.pi / 2, math.pi / 2], atol=1e-14,
    )
    quaternions = [np.array(item.goal.pose_xyz_quat_xyzw[3:]) for item in candidates]
    for index, quaternion in enumerate(quaternions):
        assert math.hypot(*quaternion) == pytest.approx(1., abs=1e-7)
        for other in quaternions[:index]:
            assert abs(float(quaternion @ other)) < 1 - 1e-7


@pytest.mark.parametrize("bounds", [
    [math.tau, math.nextafter(math.tau, math.inf)], [0., 1e-13],
])
def test_rotation_aliases_use_existing_generator_duplicate_precision(bounds):
    parameters = _parameters(offset_bounds_xyz_m=[[0., 0.]] * 3,
                             tool_axis_rotation_bounds_rad=bounds)
    actual = generate_relation_region_goals(**parameters)
    midpoint = bounds[0] + (bounds[1] - bounds[0]) / 2
    expected = generate_relation_goals(
        current_tcp_pose_xyz_quat_xyzw=parameters["current_tcp_pose_xyz_quat_xyzw"],
        anchor_xyz_m=parameters["anchor_xyz_m"], offsets_xyz_m=[[0., 0., 0.]],
        offset_frame="reference", orientation_mode="align_axis", tool_axis="z",
        axis_reference_xyz=[0., 0., -1.],
        tool_axis_rotation_offsets_rad=[midpoint, *bounds],
    )
    assert len(actual) == len(expected) == 1
    assert actual == expected


@pytest.mark.parametrize("line", [False, True])
def test_counts_are_stable_prefixes_and_reproducible_without_seed_or_environment(line, monkeypatch):
    parameters = _parameters()
    if line:
        parameters.update(offset_bounds_xyz_m=[[0., 0.]] * 3,
                          tool_axis_rotation_bounds_rad=[math.pi, 3 * math.pi])
    expected = generate_relation_region_goals(**parameters)
    state = np.random.get_state()
    try:
        np.random.seed(917)
        before = np.random.get_state()
        monkeypatch.setenv("PYTHONHASHSEED", "23")
        assert generate_relation_region_goals(**parameters) == expected
        after = np.random.get_state()
        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert before[2:] == after[2:]
        for count in range(1, 9):
            actual = generate_relation_region_goals(**{**parameters, "max_candidates": count})
            assert 1 <= len(actual) <= count
            assert actual == expected[:len(actual)]
    finally:
        np.random.set_state(state)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("frame", ["reference", "tool"])
def test_tool_frame_uses_original_orientation_before_alignment_and_rotation(axis, frame):
    rotation = Rotation.from_euler("xyz", [23., 47., -18.], degrees=True)
    current = [.1, -.2, .4, *rotation.as_quat().tolist()]
    anchor = [.2, -.1, .3]
    parameters = _parameters(
        current_tcp_pose_xyz_quat_xyzw=current, anchor_xyz_m=anchor,
        offset_frame=frame, tool_axis=axis, axis_reference_xyz=[1., 2., -3.],
        tool_axis_rotation_bounds_rad=[math.pi, 2 * math.pi],
    )
    candidates = generate_relation_region_goals(**parameters)
    for candidate in candidates:
        pose = candidate.goal.pose_xyz_quat_xyzw
        displacement = np.array(pose[:3]) - anchor
        recovered_offset = rotation.inv().apply(displacement) if frame == "tool" else displacement
        np.testing.assert_allclose(recovered_offset, candidate.offset_xyz_m, atol=1e-15)
        bounds = np.array(parameters["offset_bounds_xyz_m"])
        assert np.all(bounds[:, 0] - 1e-15 <= recovered_offset)
        assert np.all(recovered_offset <= bounds[:, 1] + 1e-15)
        np.testing.assert_allclose(
            Rotation.from_quat(pose[3:]).as_matrix()[:, ("x", "y", "z").index(axis)],
            np.array([1., 2., -3.]) / math.sqrt(14), atol=1e-7,
        )
        assert math.hypot(*pose[3:]) == pytest.approx(1., abs=1e-7)
        assert np.dot(pose[3:], current[3:]) >= 0
        # The returned canonical angle is an alias of an angle in [pi, 2*pi].
        assert math.pi - 1e-14 <= candidate.tool_axis_rotation_rad + math.tau <= math.tau + 1e-14


@pytest.mark.parametrize("turn", [0., math.tau, -math.tau, 2 * math.tau])
def test_preserve_requires_fixed_periodic_zero_and_keeps_input_orientation(turn):
    current = [1., 2., 3., *Rotation.from_euler("xyz", [20., -50., 80.], degrees=True).as_quat().tolist()]
    candidates = generate_relation_region_goals(**_parameters(
        current_tcp_pose_xyz_quat_xyzw=current, orientation_mode="preserve",
        tool_axis=None, axis_reference_xyz=None,
        tool_axis_rotation_bounds_rad=[turn, turn],
    ))
    assert len(candidates) == 8
    for candidate in candidates:
        assert candidate.tool_axis_rotation_rad == pytest.approx(0., abs=1e-12)
        np.testing.assert_allclose(candidate.goal.pose_xyz_quat_xyzw[3:], current[3:], atol=1e-7)
        assert candidate.goal.rotation_distance_rad == pytest.approx(0., abs=1e-12)
        assert candidate.goal.axis_alignment_error_rad is None


@pytest.mark.parametrize("bounds", [[0., 1e-13], [-math.pi, math.pi], [0., math.tau], [.1, .1]])
def test_preserve_rejects_nonfixed_or_nonzero_rotations(bounds):
    with pytest.raises(ValueError):
        generate_relation_region_goals(**_parameters(
            orientation_mode="preserve", tool_axis=None, axis_reference_xyz=None,
            tool_axis_rotation_bounds_rad=bounds,
        ))


@pytest.mark.parametrize("count", [True, False, 0, 9, -1, 1., "2", None, np.int64(2), 10**400])
def test_candidate_limit_is_a_strict_bounded_integer(count):
    with pytest.raises(ValueError, match="max_candidates"):
        generate_relation_region_goals(**_parameters(max_candidates=count))


@pytest.mark.parametrize("bounds", [
    None, [], [[0., 1.]] * 2, [[0., 1.]] * 4,
    [0., [0., 1.], [0., 1.]], [[0.], [0., 1.], [0., 1.]],
    [[0., 1., 2.], [0., 1.], [0., 1.]], [[1., 0.], [0., 1.], [0., 1.]],
    [[True, 1.], [0., 1.], [0., 1.]], [[0., False], [0., 1.], [0., 1.]],
    [["0", 1.], [0., 1.], [0., 1.]], [[0., float("nan")], [0., 1.], [0., 1.]],
    [[-float("inf"), 1.], [0., 1.], [0., 1.]],
    [[0., 10**400], [0., 1.], [0., 1.]],
    [[-1e308, 1e308], [0., 1.], [0., 1.]],
    np.array([[0., 1.]] * 3),
])
def test_offset_bounds_reject_invalid_shapes_values_order_and_overflow(bounds):
    with pytest.raises(ValueError):
        generate_relation_region_goals(**_parameters(offset_bounds_xyz_m=bounds))


@pytest.mark.parametrize("bounds", [
    None, [], [0.], [0., 1., 2.], [1., 0.], [True, 1.], [0., False],
    [0., "1"], [0., float("nan")], [0., float("inf")], [0., 10**400],
    [-1e308, 1e308], [0., math.nextafter(math.tau, math.inf)],
    np.array([0., 1.]),
])
def test_rotation_bounds_reject_invalid_shapes_values_order_width_and_overflow(bounds):
    with pytest.raises(ValueError):
        generate_relation_region_goals(**_parameters(tool_axis_rotation_bounds_rad=bounds))


@pytest.mark.parametrize("changes", [
    {"current_tcp_pose_xyz_quat_xyzw": [0., 0., 0., 0., 0., 0., 0.]},
    {"current_tcp_pose_xyz_quat_xyzw": [0., 0., 0., 0., 0., 0., 2.]},
    {"current_tcp_pose_xyz_quat_xyzw": [True, 0., 0., 0., 0., 0., 1.]},
    {"current_tcp_pose_xyz_quat_xyzw": [10**400, 0., 0., 0., 0., 0., 1.]},
    {"anchor_xyz_m": [0., float("nan"), 0.]},
    {"anchor_xyz_m": [0., 0., True]},
    {"anchor_xyz_m": [10**400, 0., 0.]},
    {"axis_reference_xyz": [0., 0., 0.]},
    {"axis_reference_xyz": [True, 0., 1.]},
    {"axis_reference_xyz": [float("inf"), 0., 1.]},
    {"axis_reference_xyz": [10**400, 0., 1.]},
    {"axis_reference_xyz": [1.7e308, 1.7e308, 1.7e308]},
    {"tool_axis": None}, {"tool_axis": "w"},
    {"offset_frame": "camera"}, {"orientation_mode": "look_at"},
    {"orientation_mode": "preserve", "tool_axis_rotation_bounds_rad": [0., 0.]},
])
def test_existing_pose_and_alignment_validation_is_retained(changes):
    with pytest.raises(ValueError):
        generate_relation_region_goals(**_parameters(**changes))


@pytest.mark.parametrize("changes", [
    {"anchor_xyz_m": [1.7e308, 0., 0.], "offset_bounds_xyz_m": [[1.7e308, 1.7e308], [0., 0.], [0., 0.]]},
    {"current_tcp_pose_xyz_quat_xyzw": [-1.7e308, 0., 0., 0., 0., 0., 1.], "anchor_xyz_m": [1.7e308, 0., 0.]},
    {"anchor_xyz_m": [1e100, 0., 0.]},
])
def test_finite_inputs_that_overflow_goal_computation_are_rejected(changes):
    with pytest.raises(ValueError, match="finite|overflow"):
        generate_relation_region_goals(**_parameters(**changes))


def _continuation_goals(parameters, delta=(1., 0., 0.)):
    return generate_continuation_region_goals(**parameters, continuation_delta_xyz_m=delta)


def _candidate_identity(candidate):
    return (*candidate.offset_xyz_m, round(candidate.tool_axis_rotation_rad, 12))


def _assert_continuation_bounds_and_geometry(parameters, candidates):
    bounds = parameters["offset_bounds_xyz_m"]
    assert 1 <= len(candidates) <= parameters["max_candidates"]
    assert len({_candidate_identity(candidate) for candidate in candidates}) == len(candidates)
    for candidate in candidates:
        assert all(lower <= value <= upper for value, (lower, upper) in zip(
            candidate.offset_xyz_m, bounds, strict=True))
        expected, = generate_relation_goals(
            **{key: parameters[key] for key in (
                "current_tcp_pose_xyz_quat_xyzw", "anchor_xyz_m", "offset_frame",
                "orientation_mode", "tool_axis", "axis_reference_xyz",
            )},
            offsets_xyz_m=[list(candidate.offset_xyz_m)],
            tool_axis_rotation_offsets_rad=[candidate.tool_axis_rotation_rad],
        )
        np.testing.assert_allclose(candidate.goal.pose_xyz_quat_xyzw,
            expected.goal.pose_xyz_quat_xyzw, rtol=0., atol=1e-7)
        assert all(math.isfinite(value) for value in candidate.goal.pose_xyz_quat_xyzw)
        assert math.hypot(*candidate.goal.pose_xyz_quat_xyzw[3:]) == pytest.approx(1., abs=1e-7)


def test_continuation_sampler_requires_an_explicit_delta():
    with pytest.raises(TypeError, match="continuation_delta_xyz_m"):
        generate_continuation_region_goals(**_parameters())


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_continuation_xyz_signed_extrema_are_centre_line_endpoints(axis, sign):
    parameters = _parameters()
    baseline = generate_relation_region_goals(**parameters)
    delta = [0., 0., 0.]
    delta[axis] = sign
    actual = _continuation_goals(parameters, delta)
    assert actual[0] == baseline[0]
    expected_negative = list(baseline[0].offset_xyz_m)
    expected_positive = list(expected_negative)
    lower, upper = parameters["offset_bounds_xyz_m"][axis]
    expected_negative[axis] = lower if sign > 0. else upper
    expected_positive[axis] = upper if sign > 0. else lower
    assert actual[1].offset_xyz_m == pytest.approx(expected_negative)
    assert actual[2].offset_xyz_m == pytest.approx(expected_positive)
    assert actual[1].tool_axis_rotation_rad == baseline[0].tool_axis_rotation_rad
    assert actual[2].tool_axis_rotation_rad == baseline[0].tool_axis_rotation_rad
    _assert_continuation_bounds_and_geometry(parameters, actual)


@pytest.mark.parametrize("delta", [(1., 2., -3.), (-1., -2., 3.), (1e308, 1e308, 0.)])
@pytest.mark.parametrize("bounds", [
    [[-1., 1.], [-2., 2.], [-4., 4.]],
    [[-.04, .04], [-.02, .06], [.1, .2]],
])
def test_continuation_diagonal_intersects_the_box_on_the_centre_line(delta, bounds):
    parameters = _parameters(offset_bounds_xyz_m=bounds)
    actual = _continuation_goals(parameters, delta)
    centre = np.array(actual[0].offset_xyz_m)
    direction = np.array(delta) / max(abs(value) for value in delta)
    for sign, candidate in zip((-1., 1.), actual[1:3], strict=True):
        offset = np.array(candidate.offset_xyz_m)
        displacement = offset - centre
        np.testing.assert_allclose(np.cross(displacement, direction), 0., atol=1e-14)
        assert sign * float(displacement @ direction) > 0.
        assert any(value == lower or value == upper for value, (lower, upper) in zip(
            offset, parameters["offset_bounds_xyz_m"], strict=True))
    _assert_continuation_bounds_and_geometry(parameters, actual)


def test_continuation_fixed_axes_do_not_limit_or_mix_into_free_axes():
    parameters = _parameters(offset_bounds_xyz_m=[[.25, .25], [-2., 2.], [.1, .1]])
    actual = _continuation_goals(parameters, (1e308, .5, -1e308))
    assert actual[1].offset_xyz_m == (.25, -2., .1)
    assert actual[2].offset_xyz_m == (.25, 2., .1)
    _assert_continuation_bounds_and_geometry(parameters, actual)


@pytest.mark.parametrize("tool_axis", ["x", "y", "z"])
@pytest.mark.parametrize("sign", [-1., 1.])
def test_continuation_tool_projection_uses_original_not_aligned_or_spun_axes(tool_axis, sign):
    original = Rotation.from_euler("z", 90., degrees=True)
    parameters = _parameters(
        current_tcp_pose_xyz_quat_xyzw=[.1, -.2, .4, *original.as_quat().tolist()],
        offset_frame="tool", tool_axis=tool_axis,
        axis_reference_xyz=[1., 2., -3.],
        offset_bounds_xyz_m=[[-.04, .04], [-.08, .08], [-.03, .03]],
        tool_axis_rotation_bounds_rad=[.7, .9],
    )
    actual = _continuation_goals(parameters, (sign, 0., 0.))
    assert actual[1].offset_xyz_m == pytest.approx((0., sign * .08, 0.), abs=1e-15)
    assert actual[2].offset_xyz_m == pytest.approx((0., -sign * .08, 0.), abs=1e-15)
    centre_public = np.array(parameters["anchor_xyz_m"])
    for candidate in actual[1:3]:
        displacement = np.array(candidate.goal.pose_xyz_quat_xyzw[:3]) - centre_public
        np.testing.assert_allclose(original.inv().apply(displacement), candidate.offset_xyz_m, atol=1e-15)
    _assert_continuation_bounds_and_geometry(parameters, actual)


@pytest.mark.parametrize("count", [1, 2, 3, 6, 8])
def test_continuation_budget_retains_early_halton_without_extra_draws(count, monkeypatch):
    parameters = _parameters(max_candidates=count)
    baseline = generate_relation_region_goals(**parameters)
    calls = []

    def original_region(**kwargs):
        calls.append(kwargs)
        return generate_relation_region_goals(**kwargs)

    monkeypatch.setattr(regions, "generate_relation_region_goals", original_region)
    actual = _continuation_goals(parameters)
    assert calls == [parameters]
    assert actual[0] == baseline[0]
    assert len(actual) == count
    assert actual == _continuation_goals(_parameters())[:count]
    if count >= 6:
        assert actual[3:] == baseline[1:count - 2]
        assert actual[3] == baseline[1]
    _assert_continuation_bounds_and_geometry(parameters, actual)


@pytest.mark.parametrize("frame", ["reference", "tool"])
@pytest.mark.parametrize("count", [1, 2, 3, 6, 8])
def test_continuation_zero_delta_returns_the_exact_baseline(frame, count, monkeypatch):
    parameters = _parameters(offset_frame=frame, max_candidates=count)
    baseline = generate_relation_region_goals(**parameters)
    monkeypatch.setattr(regions, "generate_relation_region_goals", lambda **kwargs: baseline)
    monkeypatch.setattr(regions, "generate_relation_goals",
        lambda **kwargs: pytest.fail("No extra geometry for zero delta"))
    assert _continuation_goals(parameters, (0., -0., 0.)) is baseline


@pytest.mark.parametrize("frame", ["reference", "tool"])
@pytest.mark.parametrize("rotation_bounds", [[.2, .2], [-math.pi, math.pi]])
def test_continuation_fixed_offsets_keep_original_rotation_sampling(frame, rotation_bounds):
    parameters = _parameters(
        offset_frame=frame, offset_bounds_xyz_m=[[.1, .1], [-.2, -.2], [.3, .3]],
        tool_axis_rotation_bounds_rad=rotation_bounds,
    )
    assert _continuation_goals(parameters, (1., -2., 3.)) == generate_relation_region_goals(**parameters)


@pytest.mark.parametrize("frame", ["reference", "tool"])
def test_continuation_no_free_axis_projection_keeps_baseline(frame):
    parameters = _parameters(
        offset_frame=frame,
        current_tcp_pose_xyz_quat_xyzw=[0., 0., 0., 0., 0., 1., 0.],
        offset_bounds_xyz_m=[[.1, .1], [-.2, -.2], [-.3, .3]],
    )
    assert _continuation_goals(parameters, (1., 2., 0.)) == generate_relation_region_goals(**parameters)


@pytest.mark.parametrize("turn", [0., math.tau, -math.tau, 2 * math.tau])
def test_continuation_preserve_keeps_fixed_periodic_zero_contract(turn):
    parameters = _parameters(
        orientation_mode="preserve", tool_axis=None, axis_reference_xyz=None,
        tool_axis_rotation_bounds_rad=[turn, turn],
    )
    actual = _continuation_goals(parameters)
    assert all(abs(candidate.tool_axis_rotation_rad) <= 1e-12 for candidate in actual)
    _assert_continuation_bounds_and_geometry(parameters, actual)


@pytest.mark.parametrize("rotation_bounds", [
    [math.tau, math.nextafter(math.tau, math.inf)], [0., 1e-13],
])
def test_continuation_period_aliases_deduplicate_extrema_and_halton(rotation_bounds):
    parameters = _parameters(
        offset_bounds_xyz_m=[[-.04, .04], [0., 0.], [0., 0.]],
        tool_axis_rotation_bounds_rad=rotation_bounds,
    )
    actual = _continuation_goals(parameters)
    assert sum(candidate.offset_xyz_m == (-.04, 0., 0.) for candidate in actual) == 1
    assert all(round(candidate.tool_axis_rotation_rad, 12) == 0. for candidate in actual)
    _assert_continuation_bounds_and_geometry(parameters, actual)


@pytest.mark.parametrize("delta", [
    [math.nan, 0., 0.], [0., math.inf, 0.], [0., 0., -math.inf],
    [True, 0., 0.], ["1", 0., 0.], [10**400, 0., 0.],
    [], [0., 0.], [0.] * 4, None, np.zeros(3),
])
@pytest.mark.parametrize("count", [1, 8])
def test_continuation_invalid_delta_is_rejected_even_without_extra_room(delta, count):
    with pytest.raises(ValueError, match="continuation_delta_xyz_m"):
        _continuation_goals(_parameters(max_candidates=count), delta)


@pytest.mark.parametrize("changes", [
    {"max_candidates": True}, {"max_candidates": 0}, {"max_candidates": 9},
    {"offset_frame": "camera"}, {"orientation_mode": "look_at"},
    {"tool_axis": None}, {"axis_reference_xyz": [0., 0., 0.]},
    {"anchor_xyz_m": [math.nan, 0., 0.]},
    {"current_tcp_pose_xyz_quat_xyzw": [0.] * 7},
    {"offset_bounds_xyz_m": [[1., 0.], [0., 0.], [0., 0.]]},
    {"tool_axis_rotation_bounds_rad": [0., math.nextafter(math.tau, math.inf)]},
    {"orientation_mode": "preserve", "tool_axis": None, "axis_reference_xyz": None,
     "tool_axis_rotation_bounds_rad": [0., 1e-13]},
])
def test_continuation_validates_original_inputs_before_delta(changes):
    parameters = _parameters(**changes)
    with pytest.raises(ValueError) as original:
        generate_relation_region_goals(**parameters)
    with pytest.raises(ValueError) as continuation:
        _continuation_goals(parameters, (math.nan, 0., 0.))
    assert str(continuation.value) == str(original.value)


@pytest.mark.parametrize("frame", ["reference", "tool"])
def test_continuation_is_deterministic_without_mutating_inputs_or_old_sampling(frame):
    parameters = _parameters(offset_frame=frame)
    delta = [1., -2., 3.]
    before = deepcopy((parameters, delta))
    baseline = generate_relation_region_goals(**parameters)
    first = _continuation_goals(parameters, delta)
    assert _continuation_goals(parameters, delta) == first
    assert (parameters, delta) == before
    assert generate_relation_region_goals(**parameters) == baseline


def test_continuation_narrow_rounding_intervals_stay_inside_closed_bounds():
    parameters = _parameters(offset_bounds_xyz_m=[
        [.1, math.nextafter(.1, math.inf)], [-.2, math.nextafter(-.2, math.inf)],
        [.3, math.nextafter(.3, math.inf)],
    ])
    _assert_continuation_bounds_and_geometry(parameters,
        _continuation_goals(parameters, (1., -2., 3.)))


@pytest.mark.parametrize("delta", [(1e308, 1e308, -1e308), (5e-324, 0., 0.)])
@pytest.mark.parametrize("frame", ["reference", "tool"])
def test_continuation_finite_direction_scale_does_not_change_geometry(delta, frame):
    parameters = _parameters(offset_frame=frame,
        current_tcp_pose_xyz_quat_xyzw=[0., 0., 0.,
            *Rotation.from_euler("xyz", [23., 47., -18.], degrees=True).as_quat().tolist()])
    unit_scale = max(abs(value) for value in delta)
    normalized = tuple(value / unit_scale for value in delta)
    assert _continuation_goals(parameters, delta) == _continuation_goals(parameters, normalized)
    _assert_continuation_bounds_and_geometry(parameters, _continuation_goals(parameters, delta))


@pytest.mark.parametrize("problem", ["nonfinite_pose", "nonunit_quaternion", "offset_outside", "nonfinite_offset"])
def test_continuation_new_endpoints_keep_legacy_output_guards(monkeypatch, problem):
    parameters = _parameters()
    baseline = generate_relation_region_goals(**parameters)
    candidate = baseline[0]
    if problem in {"nonfinite_pose", "nonunit_quaternion"}:
        pose = list(candidate.goal.pose_xyz_quat_xyzw)
        if problem == "nonfinite_pose":
            pose[0] = math.nan
        else:
            pose[3:] = [0.] * 4
        candidate = replace(candidate, goal=replace(candidate.goal, pose_xyz_quat_xyzw=tuple(pose)))
    else:
        candidate = replace(candidate, offset_xyz_m=(math.nan if problem == "nonfinite_offset" else 1., 0., .15))
    monkeypatch.setattr(regions, "generate_relation_region_goals", lambda **kwargs: baseline)
    monkeypatch.setattr(regions, "generate_relation_goals", lambda **kwargs: (candidate,))
    with pytest.raises(ValueError, match="finite|unit norm|bounds"):
        _continuation_goals(parameters)


@pytest.mark.parametrize("error", [OverflowError, FloatingPointError])
def test_continuation_new_endpoint_overflow_is_rejected(monkeypatch, error):
    parameters = _parameters()
    baseline = generate_relation_region_goals(**parameters)

    def overflow(**kwargs):
        raise error("endpoint overflow")

    monkeypatch.setattr(regions, "generate_relation_region_goals", lambda **kwargs: baseline)
    monkeypatch.setattr(regions, "generate_relation_goals", overflow)
    with pytest.raises(ValueError, match="Extrema.*finite.*overflow"):
        _continuation_goals(parameters)

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.spatial_goals import compute_spatial_goal

pytestmark = pytest.mark.unit


def _parameters(**changes):
    return {"current_tcp_pose_xyz_quat_xyzw": [0., 0., 0., 0., 0., 0., 1.],
        "anchor_xyz_m": [.2, -.1, .3], "offset_xyz_m": [0., 0., .03],
        "offset_frame": "reference", "orientation_mode": "preserve",
        "tool_axis": None, "axis_reference_xyz": None, **changes}


def test_relation_uses_anchor_not_current_translation():
    result = compute_spatial_goal(**_parameters())
    np.testing.assert_allclose(result.pose_xyz_quat_xyzw, [.2, -.1, .33, 0, 0, 0, 1])
    assert result.rotation_distance_rad == 0
    assert result.axis_alignment_error_rad is None


def test_tool_offset_uses_input_orientation_before_alignment():
    pose = [0., 0., 0., *Rotation.from_euler("y", 90, degrees=True).as_quat().tolist()]
    result = compute_spatial_goal(**_parameters(current_tcp_pose_xyz_quat_xyzw=pose,
        offset_frame="tool", orientation_mode="align_axis", tool_axis="z", axis_reference_xyz=[0, 0, -1]))
    np.testing.assert_allclose(result.pose_xyz_quat_xyzw[:3], [.23, -.1, .3])
    np.testing.assert_allclose(xyzquat_to_pose_matrix(result.pose_xyz_quat_xyzw)[:3, 2], [0, 0, -1], atol=1e-7)
    assert result.rotation_distance_rad == pytest.approx(math.pi / 2)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("direction", [[0, 0, 1], [0, 0, -1], [1, 1, -1]])
def test_axis_alignment_takes_geodesically_shortest_turn(axis, direction):
    rng = np.random.default_rng(14)
    for _ in range(20):
        rotation = Rotation.random(random_state=rng)
        current = [0., 0., 0., *rotation.as_quat().tolist()]
        result = compute_spatial_goal(**_parameters(current_tcp_pose_xyz_quat_xyzw=current,
            orientation_mode="align_axis", tool_axis=axis, axis_reference_xyz=direction))
        goal = xyzquat_to_pose_matrix(result.pose_xyz_quat_xyzw)
        index = ("x", "y", "z").index(axis)
        expected = np.array(direction) / np.linalg.norm(direction)
        np.testing.assert_allclose(goal[:3, index], expected, atol=1e-7)
        angle = math.acos(float(np.clip(rotation.as_matrix()[:, index] @ expected, -1., 1.)))
        assert result.rotation_distance_rad == pytest.approx(angle, abs=1e-12)
        assert result.axis_alignment_error_rad < 1e-12


@pytest.mark.parametrize("direction, angle", [([0, 0, 1], 0.), ([0, 0, -1], math.pi), ([1e-13, 0, -1], math.pi)])
def test_parallel_and_antiparallel_alignment_is_deterministic(direction, angle):
    parameters = _parameters(orientation_mode="align_axis", tool_axis="z", axis_reference_xyz=direction)
    first = compute_spatial_goal(**parameters)
    assert first == compute_spatial_goal(**parameters)
    assert first.rotation_distance_rad == pytest.approx(angle)


@pytest.mark.parametrize("changes", [
    {"offset_frame": "camera"}, {"orientation_mode": "look_at"},
    {"offset_xyz_m": [True, 0., 0.]}, {"anchor_xyz_m": [0., float("nan"), 0.]},
    {"current_tcp_pose_xyz_quat_xyzw": [0., 0., 0., 0., 0., 0., 0.]},
    {"current_tcp_pose_xyz_quat_xyzw": [0., 0., 0., 0., 0., 0., 2.]},
    {"tool_axis": "z"}, {"axis_reference_xyz": [0., 0., 1.]},
    {"orientation_mode": "align_axis"},
    {"orientation_mode": "align_axis", "tool_axis": "z", "axis_reference_xyz": [0., 0., 0.]},
    {"orientation_mode": "align_axis", "tool_axis": "z", "axis_reference_xyz": [float("inf"), 0., 0.]},
])
def test_invalid_proposals_fail_before_any_path_work(changes):
    with pytest.raises(ValueError):
        compute_spatial_goal(**_parameters(**changes))

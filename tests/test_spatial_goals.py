from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robot_motion.relative_action import xyzquat_to_pose_matrix
from robot_motion.spatial_goals import (
    compute_spatial_goal,
    generate_object_above_goals,
    generate_relation_goals,
)

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


def test_above_goals_keep_height_region_and_tool_axis_and_deduplicate_yaw():
    current = [.1, -.2, .4, *Rotation.from_euler("xyz", [25, 60, -10], degrees=True).as_quat().tolist()]
    candidates = generate_object_above_goals(current_tcp_pose_xyz_quat_xyzw=current,
        anchor_xyz_m=[-.2, .03, .07], heights_m=[.10, .125, .15, .10],
        tool_yaw_offsets_rad=[0., math.pi/2, -math.pi, math.pi, 2*math.pi])
    assert len(candidates) == 9
    for candidate in candidates:
        pose = xyzquat_to_pose_matrix(candidate.goal.pose_xyz_quat_xyzw)
        np.testing.assert_allclose(pose[:3, 3], [-.2, .03, .07+candidate.height_m], atol=1e-8)
        np.testing.assert_allclose(pose[:3, 2], [0, 0, -1], atol=1e-7)
        assert np.dot(candidate.goal.pose_xyz_quat_xyzw[3:], current[3:]) >= 0
        assert candidate.goal.rotation_distance_rad <= math.pi
        assert candidate.goal.axis_alignment_error_rad < 1e-6


@pytest.mark.parametrize("heights,yaws", [([], [0.]), ([.1], []), ([-.01], [0.]),
    ([True], [0.]), ([.1], [float("nan")]), ([.1]*9, [0.]*8)])
def test_above_goals_reject_invalid_or_unbounded_samples(heights, yaws):
    with pytest.raises(ValueError):
        generate_object_above_goals(current_tcp_pose_xyz_quat_xyzw=[0, 0, 0, 0, 0, 0, 1],
            anchor_xyz_m=[0, 0, 0], heights_m=heights, tool_yaw_offsets_rad=yaws)


def _relation_parameters(**changes):
    return {**_parameters(), "offsets_xyz_m": [[.04, -.03, .14]],
            "tool_axis_rotation_offsets_rad": [0.], **changes}


def _relation_candidates(**changes):
    values = _relation_parameters(**changes)
    values.pop("offset_xyz_m")
    return generate_relation_goals(**values)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("frame", ["reference", "tool"])
def test_relation_samples_support_every_tool_axis_and_original_tool_offsets(axis, frame):
    rotation = Rotation.from_euler("xyz", [23, 47, -18], degrees=True)
    current = [0., 0., 0., *rotation.as_quat().tolist()]
    anchor, offset = np.array([.2, -.1, .3]), np.array([.04, -.03, .14])
    candidates = _relation_candidates(current_tcp_pose_xyz_quat_xyzw=current,
        orientation_mode="align_axis", tool_axis=axis, axis_reference_xyz=[1., 2., -3.],
        offset_frame=frame, tool_axis_rotation_offsets_rad=[0., math.pi / 2, -math.pi / 2])
    assert len(candidates) == 3
    expected = anchor + (offset if frame == "reference" else rotation.apply(offset))
    for candidate in candidates:
        pose = xyzquat_to_pose_matrix(candidate.goal.pose_xyz_quat_xyzw)
        np.testing.assert_allclose(pose[:3, 3], expected, atol=1e-8)
        np.testing.assert_allclose(pose[:3, ("x", "y", "z").index(axis)],
                                   np.array([1, 2, -3]) / math.sqrt(14), atol=1e-7)
        assert np.dot(candidate.goal.pose_xyz_quat_xyzw[3:], current[3:]) >= 0


def test_relation_samples_deduplicate_in_order_and_preserve_orientation():
    candidates = _relation_candidates(offsets_xyz_m=[[.04, -.03, .14], [0, 0, .2], [.04, -.03, .14]],
        tool_axis_rotation_offsets_rad=[0., 2 * math.pi, -2 * math.pi])
    assert [item.offset_xyz_m for item in candidates] == [(.04, -.03, .14), (0., 0., .2)]
    assert all(item.tool_axis_rotation_rad == 0 for item in candidates)
    assert all(item.goal.pose_xyz_quat_xyzw[3:] == (0, 0, 0, 1) for item in candidates)


@pytest.mark.parametrize("changes", [
    {"offsets_xyz_m": []}, {"offsets_xyz_m": [[True, 0, 0]]},
    {"offsets_xyz_m": [[0, float("nan"), 0]]}, {"offsets_xyz_m": [[0, 0]]},
    {"tool_axis_rotation_offsets_rad": []}, {"tool_axis_rotation_offsets_rad": [float("inf")]},
    {"tool_axis_rotation_offsets_rad": [.1]},
    {"offsets_xyz_m": [[0, 0, 0]] * 9, "tool_axis_rotation_offsets_rad": [0.] * 8},
    {"tool_axis": "z"}, {"orientation_mode": "align_axis"},
])
def test_relation_samples_reject_bad_geometry_and_bound_before_deduplication(changes):
    with pytest.raises(ValueError):
        _relation_candidates(**changes)


@pytest.mark.parametrize("changes", [
    {"time_budget_sec": True}, {"time_budget_sec": 0}, {"time_budget_sec": 31},
    {"time_budget_sec": float("nan")}, {"heights_m": [-.1]}, {"tool_yaw_offsets_rad": [float("inf")]},
    {"heights_m": [.1]*8, "tool_yaw_offsets_rad": [0.]*9}, {"extra": 1},
])
def test_path_query_rejects_invalid_budget_or_unbounded_geometry(changes):
    from agent_gateway.skills.perception.object_goal_query import (
        validate_query_parameters,
    )

    with pytest.raises(ValueError):
        validate_query_parameters({"component_id": "right_arm", "reference_frame_id": "test/station",
            "anchor_xyz_m": [0, 0, 0], "heights_m": [.1], "tool_yaw_offsets_rad": [0.],
            "time_budget_sec": 5., **changes})


@pytest.mark.parametrize("mutation", ["missing_evidence", "duplicate_index", "wrong_winner", "missing_hash",
    "false_no_path", "non_object", "pose_size", "pose_nan", "pose_bool", "pose_quaternion", "pose_mismatch"])
def test_path_query_rejects_inconsistent_path_evidence(mutation):
    from agent_gateway.skills.perception.object_goal_query import (
        RANKING,
        validate_query_result,
    )

    diagnostic = {"candidate_index": 0, "reason_code": "path_found", "wall_sec": .1,
                  "ranking_cost": [0., 1., 0.], "trajectory_hash": "c"*64}
    result = {"component_id": "right_arm", "reference_frame_id": "test/station", "anchor_xyz_m": [0, 0, 0],
        "source_tcp_pose_xyz_quat_xyzw": [0, 0, 0, 0, 0, 0, 1], "query_status": "path_found",
        "scene_hash": "a"*64, "source_qpos_hash": "b"*64, "source_sim_time_sec": 0.,
        "pose_targets_by_component": {"right_arm": [0, 0, 0, 0, 0, 0, 1]}, "candidate_count": 1,
        "checked_count": 1, "all_candidates_checked": True, "selected_index": 0,
        "candidate_diagnostics": [diagnostic], "planning_wall_sec": .1, "ranking_criteria": RANKING,
        "motion_executed": False, "execution_authorized": False, "input_source": "caller_anchor_and_frozen_measured_scene",
        "constraint_scope": "frozen_scene_full_path"}
    validate_query_result(result)
    if mutation == "missing_evidence":
        result.update(candidate_diagnostics=[], checked_count=0, all_candidates_checked=False)
    elif mutation == "duplicate_index":
        result.update(candidate_count=2, checked_count=2, candidate_diagnostics=[diagnostic, diagnostic])
    elif mutation == "wrong_winner":
        result.update(candidate_count=2, all_candidates_checked=False, selected_index=1)
    elif mutation == "missing_hash":
        diagnostic.pop("trajectory_hash")
    elif mutation.startswith("pose_"):
        pose = [0, 0, 0, 0, 0, 0, 1]
        if mutation == "pose_size":
            pose.pop()
        elif mutation == "pose_nan":
            pose[0] = float("nan")
        elif mutation == "pose_bool":
            pose[0] = True
        elif mutation == "pose_quaternion":
            pose[-1] = 2.
        else:
            pose[0] = .01
        diagnostic["pose_target_xyz_quat_xyzw"] = pose
    elif mutation == "non_object":
        result["candidate_diagnostics"] = [None]
        with pytest.raises(TypeError, match="must be an object"):
            validate_query_result(result)
        return
    else:
        result.update(query_status="no_path_found", selected_index=None, pose_targets_by_component={})
    with pytest.raises(ValueError):
        validate_query_result(result)

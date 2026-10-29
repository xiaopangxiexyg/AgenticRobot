"""Host-neutral tool-entity relation and hover search shared by the probe and Gateways."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent_gateway.skills.perception import tool_entity as skill
from robot_motion import tool_entity_goals as goals
from robot_motion import tool_entity_scene as scene

pytestmark = [pytest.mark.unit]


def _entity():
    # Minimal two-finger box: 10 cm long gripper, 8 cm wide closing region.
    return goals.ToolEntityBox(
        tcp_from_box_xyz_m=(0., 0., -0.05), tcp_from_box_rotation=tuple(np.eye(3).reshape(-1)),
        half_extents_m=(0.05, 0.02, 0.05), tip_center_tcp_m=(0., 0., 0.),
        parts=(((0., 0.04, -0.05), (0.01, 0.005, 0.05)), ((0., -0.04, -0.05), (0.01, 0.005, 0.05))),
        grasp_region=((0., 0., -0.03), (0.01, 0.035, 0.03)), source={"geometry": "test"})


def _down_pose(x, y, z):
    # Tool +Z points along world -Z.
    return [x, y, z, 1., 0., 0., 0.]


def _table(z=0.0, half=0.3, step=0.01):
    xs = np.arange(-half, half + 1e-9, step)
    grid = np.array([(x, y, z) for x in xs for y in xs])
    return grid


def test_shared_constants_match_the_skill_contract():
    assert skill.SEMANTICS_RELATION == goals.TOOL_OBJECT_RELATION_SEMANTICS_V1
    assert skill.SEMANTICS_HOVER == goals.TOOL_ENTITY_HOVER_SEMANTICS_V1
    assert skill.MAX_HOVER_TIP_OFFSET_M == goals.MAX_HOVER_TIP_OFFSET_M
    assert skill.MAX_HOVER_CLEARANCE_M == goals.MAX_HOVER_CLEARANCE_M
    assert math.isclose(skill.MAX_TOOL_AXIS_TILT_RAD, goals.MAX_TOOL_AXIS_TILT_RAD)
    assert set(skill.RELATION_CLASSES) == set(goals.TOOL_OBJECT_RELATIONS)


@pytest.mark.parametrize("kwargs", [
    {"orientation_policy": "sideways"},
    {"tool_axis_max_tilt_rad": 0.2},
    {"orientation_policy": "tool_axis_down", "tool_axis_max_tilt_rad": 2.0},
    {"max_candidates": 9},
    {"max_candidates": True},
    {"tip_offset_bounds_xy_m": [[0.02, -0.02], [0., 0.]]},
    {"tip_offset_bounds_xy_m": [[-0.3, 0.], [0., 0.]]},
    {"clearance_bounds_m": [-0.01, 0.05]},
    {"clearance_bounds_m": [0.01, float("nan")]},
])
def test_hover_request_is_rejected_before_any_scene_work(kwargs):
    request = {"tip_offset_bounds_xy_m": [[-0.02, 0.02], [-0.02, 0.02]], "clearance_bounds_m": [0.03, 0.08],
               "orientation_policy": "preserve", "tool_axis_max_tilt_rad": 0., "max_candidates": 8, **kwargs}
    with pytest.raises(ValueError):
        scene.validate_hover_request(**request)


def _search(outcomes, *, clock=None, **kwargs):
    calls = []

    def plan(pose, *, time_budget_sec):
        calls.append((pose, time_budget_sec))
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        return outcome(pose) if callable(outcome) else outcome

    request = {"entity": _entity(), "anchor_xyz_m": [0., 0., 0.], "current_tcp_pose_xyz_quat_xyzw": _down_pose(0., 0., 0.3),
               "points_m": _table(), "support_margin_m": 0., "tip_offset_bounds_xy_m": [[-0.02, 0.02], [-0.02, 0.02]],
               "clearance_bounds_m": [0.03, 0.08], "orientation_policy": "preserve", "tool_axis_max_tilt_rad": 0.,
               "max_candidates": 4, "translation_tolerance_m": 0.003, "rotation_tolerance_rad": 0.03,
               "plan_candidate": plan, **kwargs}
    if clock is not None:
        request["clock"] = clock
    return scene.search_entity_hover_paths(**request), calls


def _found(pose):
    return scene.HoverPlanOutcome(status="path_found", reason="planned_not_executed",
                                  endpoint_pose_xyz_quat_xyzw=tuple(pose), trajectory_duration_sec=1.5)


def test_first_certified_proposal_is_selected_and_never_executed():
    result, calls = _search([_found])
    assert result["status"] == "path_found" and result["reason"] == "planned_not_executed"
    assert result["selected_candidate_index"] == 0 and len(calls) == 1
    assert result["target_tcp_pose_xyz_quat_xyzw"] == list(calls[0][0])
    assert result["achieved"]["inside_requested_region"] is True
    assert 0.03 - 1e-9 <= result["achieved"]["achieved_clearance_m"] <= 0.08 + 1e-9
    assert calls[0][1] <= scene.ENTITY_HOVER_CANDIDATE_BUDGET_SEC
    assert result["semantics"] == goals.TOOL_ENTITY_HOVER_SEMANTICS_V1


def test_each_proposal_gets_the_five_second_hover_budget():
    assert scene.ENTITY_HOVER_CANDIDATE_BUDGET_SEC == 5. and scene.ENTITY_HOVER_REGION_BUDGET_SEC == 20.
    retry = scene.HoverPlanOutcome(status="path_rejected", reason="ik_search_exhausted", retryable=True)
    result, calls = _search([retry, _found], clock=lambda: 0.)
    assert [budget for _, budget in calls] == [5., 5.]
    assert result["candidate_budget_sec"] == 5. and result["region_budget_sec"] == 20.


def test_an_endpoint_outside_the_original_bounds_is_not_certified():
    def drifted(pose):
        moved = list(pose)
        moved[2] += 0.2
        return _found(moved)

    result, calls = _search([drifted, _found])
    assert result["status"] == "path_found" and result["selected_candidate_index"] == 1
    assert result["candidate_attempts"][0]["inside_requested_region"] is False
    assert len(calls) == 2


def test_retryable_rejections_continue_and_fatal_rejections_stop():
    retry = scene.HoverPlanOutcome(status="path_rejected", reason="ik_search_exhausted", retryable=True)
    fatal = scene.HoverPlanOutcome(status="path_rejected", reason="path_start_collision_blocked")
    result, _ = _search([retry, _found])
    assert result["status"] == "path_found" and result["selected_candidate_index"] == 1
    result, calls = _search([fatal, _found])
    assert (result["status"], result["reason"], len(calls)) == ("path_rejected", "path_start_collision_blocked", 1)
    result, calls = _search([retry])
    assert (result["status"], result["reason"]) == ("path_rejected", "region_no_candidate_path")
    assert result["attempted_candidate_count"] == result["candidate_count"] == len(calls) == 4


def test_unfinished_candidates_make_the_search_incomplete_not_rejected():
    incomplete = scene.HoverPlanOutcome(status="query_incomplete", reason="path_planning_time_budget_exhausted")
    retry = scene.HoverPlanOutcome(status="path_rejected", reason="ik_search_exhausted", retryable=True)
    result, _ = _search([incomplete, retry])
    assert (result["status"], result["reason"]) == ("query_incomplete", "region_candidate_search_incomplete")
    assert result["unresolved_candidate_count"] == 1


def test_a_late_answer_is_not_a_certified_result():
    now = [0.]

    def clock():
        return now[0]

    def slow(pose):
        now[0] += 5.
        return _found(pose)

    result, _ = _search([slow], clock=clock, region_budget_sec=6., candidate_budget_sec=3.)
    assert result["status"] == "query_incomplete"
    assert result["candidate_attempts"][0]["region_validation_budget_exceeded"] is True
    assert result["target_tcp_pose_xyz_quat_xyzw"] is None


def test_deadline_before_generation_reports_budget_exhaustion():
    result, calls = _search([_found], clock=lambda: 10., deadline=5.)
    assert (result["status"], result["reason"], result["candidate_count"], calls) == (
        "query_incomplete", "query_budget_exhausted", 0, [])


def test_cancellation_is_raised_not_reported():
    with pytest.raises(InterruptedError):
        _search([_found], cancel_requested=lambda: True)


def test_planner_must_return_a_typed_outcome():
    with pytest.raises(TypeError):
        _search([lambda pose: {"status": "path_found"}])


def test_outcome_shape_is_strict():
    with pytest.raises(ValueError):
        scene.HoverPlanOutcome(status="path_found", reason="x")
    with pytest.raises(ValueError):
        scene.HoverPlanOutcome(status="path_rejected", reason="x", endpoint_pose_xyz_quat_xyzw=(0.,) * 7)
    with pytest.raises(ValueError):
        scene.HoverPlanOutcome(status="executed", reason="x")


def test_relations_use_each_frames_capture_pose_and_list_unbound_arms():
    entity = _entity()
    # Points between the fingers of a gripper at the origin pointing down
    # (TCP +Z = world -Z, so the closing region lies above the TCP in world Z).
    between = np.array([(dx, dy, 0.03) for dx in (-0.004, 0., 0.004) for dy in np.linspace(-0.02, 0.02, 5)])
    tcp_here = _down_pose(0., 0., 0.)
    tcp_far = _down_pose(1., 1., 1.)
    surfaces = [
        scene.MeasuredObjectSurface(reference="cube@main", label="cube", role="main", points_m=between,
                                    surface_cluster={"voxel_count": 15, "component_count": 1, "kept_voxel_count": 15},
                                    tcp_pose_by_side={"right": tcp_here}),
        scene.MeasuredObjectSurface(reference="cube@wrist", label="cube", role="wrist_right", points_m=between,
                                    tcp_pose_by_side={"right": tcp_far}),
    ]
    result = scene.describe_tool_object_relations(surfaces=surfaces, tool_entity_by_side={"right": entity})
    assert result["evaluated_sides"] == ["right"] and result["unbound_sides"] == ["left"]
    by_role = {view["role"]: view for view in result["views"]}
    assert by_role["main"]["relation"].startswith("between_fingers")
    assert by_role["wrist_right"]["relation"] == "far"
    assert {view["gripper_pose_source"] for view in result["views"]} == {"frame_capture_qpos"}
    (fused,) = result["objects"]
    assert fused["relation"].startswith("between_fingers") and fused["view_count"] == 2
    assert result["semantics"] == goals.TOOL_OBJECT_RELATION_SEMANTICS_V1


def test_relations_reject_two_surfaces_of_one_label_in_one_view():
    points = np.zeros((5, 3))
    surfaces = [scene.MeasuredObjectSurface(reference=f"cube@{i}", label="cube", role="main", points_m=points,
                                            tcp_pose_by_side={"right": _down_pose(0., 0., 1.)}) for i in range(2)]
    with pytest.raises(ValueError, match="one surface per camera role"):
        scene.describe_tool_object_relations(surfaces=surfaces, tool_entity_by_side={"right": _entity()})


def test_mask_surface_respects_depth_range_and_erosion():
    depth = np.full((40, 40), 0.5, dtype=np.float32)
    depth[:, 30:] = 3.0
    mask = np.zeros((40, 40), dtype=bool)
    mask[5:35, 5:38] = True
    capture = type("Capture", (), {"depth_m": depth, "intrinsics": {"fx": 100., "fy": 100., "cx": 20., "cy": 20.},
                                   "source_from_camera": np.eye(4)})()
    points, evidence = scene.object_surface_from_mask(mask, capture,
                                                      maximum_depth_m=scene.RELATION_MAXIMUM_DEPTH_M)
    assert points is not None and np.all(points[:, 2] <= scene.RELATION_MAXIMUM_DEPTH_M + 1e-9)
    assert evidence["mask_erosion_px"] == scene.OBJECT_MASK_EROSION_PX
    with pytest.raises(ValueError):
        scene.object_surface_from_mask(np.zeros((10, 10), bool), capture, maximum_depth_m=1.5)

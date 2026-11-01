"""perception.query_grasp_approach@1: contract and measured geometry."""

from __future__ import annotations

import math

import numpy as np
import pytest

from agent_gateway.profiles.measured_support import minimum_area_rectangle
from agent_gateway.skills.perception import grasp_approach_query as skill

pytestmark = [pytest.mark.unit]

FRAME = "mujoco_sim_hil/station"
POSE = [0.1, 0.0, 0.2, 0., 0., 0., 1.]


def _found(**changes):
    value = {
        "semantics": skill.SEMANTICS, "status": "grasp_approach_found",
        "reason": "collision_free_approach_found", "object_id": "debug_cube",
        "reference_frame_id": FRAME, "station_revision": 0,
        **{key: list(POSE) for key in skill.POSE_KEYS},
        "approach_family": "top_down",
        "measured_object": {"role": "main", "language_prompt": "blue cube", "score": 0.8,
                            "top_center_xyz_m": [-0.19, 0.02, 0.036], "yaw_rad": 0.01,
                            "extent_xy_m": [0.035, 0.035], "top_point_count": 300,
                            "support_z_m": 0.001},
        "yaw_offset_rad": 0., "candidate_count": 6, "attempted_candidate_count": 2,
        "selected_candidate_index": 1,
        "candidate_rejections": [{"candidate": "top_down",
                                  "reason_code": "approach_safe_tcp_target_blocked"}],
        "path_planning_wall_sec": 1.2,
        "views": [{"role": "main", "outcome": "measured", "score": 0.8}],
        "motion_executed": False, "execution_authorized": False}
    value.update(changes)
    return value


def test_grasp_query_contract_names_the_object_and_the_execution_order():
    definition = skill.definition()
    schema = skill.schema(definition_hash=definition.semantic_content_hash, profile_hash="0" * 64,
                          prompt_by_object_id={"debug_cube": "blue cube"}, reference_frame_id=FRAME)
    properties = schema.parameter_schema["properties"]
    assert list(properties["object_id"]["enum"]) == ["debug_cube"]
    assert "blue cube" in properties["object_id"]["description"]
    assert dict(properties["reference_frame_id"]) == {"const": FRAME}
    for consumer in ("motion.approach_grasp_pose", "motion.approach_grasp_contact"):
        assert consumer in definition.summary
    skill.validate_parameters({"object_id": "debug_cube", "reference_frame_id": FRAME},
                              object_ids=("debug_cube",), reference_frame_id=FRAME)
    with pytest.raises(ValueError):
        skill.validate_parameters({"object_id": "yellow_cube", "reference_frame_id": FRAME},
                                  object_ids=("debug_cube",), reference_frame_id=FRAME)


def test_grasp_query_results_are_consistent_with_their_status():
    objects = ("debug_cube",)
    skill.validate_result(_found(), object_ids=objects, reference_frame_id=FRAME)
    skill.validate_result(_found(approach_family="side", yaw_offset_rad=None), object_ids=objects,
                          reference_frame_id=FRAME)
    rejected = _found(status="no_grasp_approach", reason="gripper_not_open",
                      **{key: None for key in skill.POSE_KEYS}, yaw_offset_rad=None,
                      approach_family=None, selected_candidate_index=None, measured_object=None,
                      candidate_rejections=[], views=[])
    skill.validate_result(rejected, object_ids=objects, reference_frame_id=FRAME)
    for bad in (_found(reason="gripper_not_open"), _found(selected_candidate_index=4),
                _found(approach_family=None), _found(approach_family="diagonal"),
                _found(approach_family="side"),
                _found(candidate_rejections=[{"candidate": "x"}] * 3),
                _found(motion_executed=True), _found(measured_object=None),
                _found(views=[{"role": "main", "outcome": "not_segmented", "score": None}]),
                {**rejected, "approach_tcp_pose_xyz_quat_xyzw": list(POSE)}):
        with pytest.raises(ValueError):
            skill.validate_result(bad, object_ids=objects, reference_frame_id=FRAME)


@pytest.mark.parametrize("yaw_deg", [0., 12., 33., -40., 75.])
def test_minimum_area_rectangle_recovers_box_yaw_modulo_quarter_turns(yaw_deg):
    rng = np.random.default_rng(7)
    yaw = math.radians(yaw_deg)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    points = rng.uniform(-.0175, .0175, size=(300, 2)) @ rotation.T + (.2, -.1)
    center, estimate, extent = minimum_area_rectangle(points)
    error = (math.degrees(estimate) - yaw_deg + 45.) % 90. - 45.
    assert abs(error) < 1. and np.allclose(center, (.2, -.1), atol=1e-3)
    assert np.all(np.abs(extent - .035) < .002)


def test_grasp_candidates_offer_level_face_grasps_only():
    """Pitched face grasps held the 35 mm cube by its corner and faulted every lift."""
    from types import SimpleNamespace

    from agent_gateway.profiles.mujoco import grasp_approach_query as query

    top = SimpleNamespace(yaw_rad=.01, center_xy_m=(-.19, .02), top_z_m=.036)
    candidates = query.candidate_grasps(top, surface_z=.001, motion_side="right")
    assert [item[0] for item in candidates] == [
        "top_down", "top_down_turned_left", "top_down_turned_right", "side", "front",
        "top_down_half_turn"]
    # The level side and front face grasps approach along perpendicular
    # horizontal axes; both hold the cube level (tool z horizontal).
    side = next(item for item in candidates if item[0] == "side")
    front = next(item for item in candidates if item[0] == "front")
    side_axis = side[5][:3, 2]
    front_axis = front[5][:3, 2]
    assert abs(float(side_axis @ front_axis)) < 1e-6
    for label, family, _offset, approach, pregrasp, contact in candidates:
        if family == "top_down":
            continue
        # tool z is the approach axis: horizontal for every face grasp.
        for pose in (approach, pregrasp, contact):
            assert abs(pose[2, 2]) < 1e-9, label
    # Without a measured surface ring only top-down grasps are proposed.
    assert {item[1] for item in query.candidate_grasps(top, surface_z=None, motion_side="right")} == {"top_down"}

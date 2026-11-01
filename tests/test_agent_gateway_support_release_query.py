"""perception.query_support_release_pose@1: contract and default-off CLI wiring."""

from __future__ import annotations

import math

import pytest

from agent_gateway.profiles.mujoco import gateway_cli
from agent_gateway.skills.perception import support_release_query as skill

pytestmark = [pytest.mark.unit]

FRAME = "mujoco_sim_hil/station"


def _found(**changes):
    value = {
        "semantics": skill.SEMANTICS, "status": "release_pose_found",
        "reason": "collision_free_release_path_found", "support_object_id": "green_box",
        "reference_frame_id": FRAME, "station_revision": 0,
        "release_object_pose_xyz_quat_xyzw": [-0.11, -0.03, 0.064, 0., 0., 0.7071068, 0.7071068],
        "release_clearance_m": skill.DEFAULT_CLEARANCE_M,
        "held_object_pose_xyz_quat_xyzw": [-0.19, 0.02, 0.07, 0., 0., 0., 1.],
        "yaw_change_rad": -math.pi / 2, "release_lean_rad": 0., "support_top_z_m": 0.04,
        "support_region_min_xy_m": [-0.15, -0.06], "support_region_max_xy_m": [-0.07, 0.02],
        "segmentation": {"role": "wrist_right", "language_prompt": "green box", "score": 0.8,
                         "mask_pixel_count": 4000, "mask_bounds_xyxy": [300, 60, 500, 265]},
        "measured_support": {"top_point_count": 900, "plane_residual_m": 0.0002,
                             "plane_tilt_deg": 0.1, "footprint_sample_count": 1600,
                             "observed_region_min_xy_m": [-0.16, -0.07],
                             "observed_region_max_xy_m": [-0.06, 0.03]},
        "candidate_count": 40, "attempted_candidate_count": 40, "selected_candidate_index": 6,
        "path_planning_wall_sec": 2.5,
        "candidate_rejections": [{"reason_code": "safe_tcp_target_blocked", "count": 3}],
        "views": [{"role": "wrist_right", "outcome": "measured", "score": 0.8,
                   "certified_area_m2": 0.0064},
                  {"role": "main", "outcome": "top_not_measured", "score": 0.4,
                   "certified_area_m2": None}],
        "motion_executed": False, "execution_authorized": False}
    value.update(changes)
    return value


def test_release_query_contract_hides_nothing_the_agent_must_type():
    definition = skill.definition()
    schema = skill.schema(definition_hash=definition.semantic_content_hash, profile_hash="0" * 64,
                          support_object_ids=("green_box", "red_box"), reference_frame_id=FRAME)
    parameters = schema.parameter_schema
    assert definition.skill_kind == "inspection" and definition.controlled_component_ids == ()
    assert list(parameters["required"]) == ["support_object_id", "reference_frame_id"]
    assert dict(parameters["properties"]["reference_frame_id"]) == {"const": FRAME}
    assert list(parameters["properties"]["support_object_id"]["enum"]) == ["green_box", "red_box"]
    assert "phase=carry" in definition.summary and "release_support_object_id" in definition.summary
    # RPent's Gateway adapter refuses a Skill summary over 2000 characters, which
    # fails the whole discovery (rpent_agent_gateway.contracts).
    assert len(definition.summary) <= 2_000
    supports = ("green_box", "red_box")
    skill.validate_parameters({"support_object_id": "red_box", "reference_frame_id": FRAME},
                              support_object_ids=supports, reference_frame_id=FRAME)
    for bad in ({"support_object_id": "yellow_cube", "reference_frame_id": FRAME},
                {"support_object_id": "red_box", "reference_frame_id": "world"},
                {"support_object_id": "red_box", "reference_frame_id": FRAME, "clearance_m": 0.2},
                {"support_object_id": "red_box", "reference_frame_id": FRAME, "relation": "left"}):
        with pytest.raises(ValueError):
            skill.validate_parameters(bad, support_object_ids=supports, reference_frame_id=FRAME)


def test_release_query_results_are_consistent_with_their_status():
    supports = ("green_box", "red_box")
    skill.validate_result(_found(), support_object_ids=supports, reference_frame_id=FRAME)
    skill.validate_result_for_parameters(
        {"support_object_id": "green_box", "reference_frame_id": FRAME}, _found())
    rejected = _found(status="no_release_pose", reason="no_collision_free_release_path",
                      release_object_pose_xyz_quat_xyzw=None, yaw_change_rad=None,
                      release_lean_rad=None, selected_candidate_index=None)
    skill.validate_result(rejected, support_object_ids=supports, reference_frame_id=FRAME)
    skill.validate_result(_found(release_lean_rad=math.radians(6.)), support_object_ids=supports,
                          reference_frame_id=FRAME)
    assert "release_lean_rad" in skill.definition().summary
    for bad in (
        _found(reason="no_collision_free_release_path"),
        _found(release_object_pose_xyz_quat_xyzw=[-0.11, -0.03, 0.03, 0., 0., 0., 1.]),
        _found(motion_executed=True),
        _found(selected_candidate_index=40),
        _found(release_lean_rad=None),
        _found(release_lean_rad=-0.01),
        _found(release_lean_rad=math.radians(11.)),
        _found(views=[{"role": "main", "outcome": "measured", "score": 0.4,
                       "certified_area_m2": 0.001}]),
        {**rejected, "release_object_pose_xyz_quat_xyzw": [0., 0., .1, 0., 0., 0., 1.]},
        {**rejected, "release_lean_rad": 0.},
    ):
        with pytest.raises(ValueError):
            skill.validate_result(bad, support_object_ids=supports, reference_frame_id=FRAME)
    with pytest.raises(ValueError, match="another support"):
        skill.validate_result_for_parameters(
            {"support_object_id": "red_box", "reference_frame_id": FRAME}, _found())


def test_release_query_cli_is_default_off_and_needs_named_release_and_prompts(tmp_path):
    args = gateway_cli.build_parser().parse_args(["--socket-path", "/unused", "--task-config", "/unused"])
    assert args.enable_support_release_queries is False and args.support_release_prompt == []
    base = ["--socket-path", str(tmp_path / "g.sock"), "--task-config", "/unused",
            "--artifact-root", str(tmp_path), "--no-realtime-worker"]
    with pytest.raises(SystemExit):
        gateway_cli.main([*base, "--enable-support-release-queries"])
    with pytest.raises(SystemExit):
        gateway_cli.main([*base, "--support-release-prompt", "green_box=green box"])

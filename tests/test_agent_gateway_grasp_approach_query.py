"""perception.query_grasp_approach@1: contract, measurement and the native grasp-place chain."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.measured_support import minimum_area_rectangle
from agent_gateway.skills.perception import grasp_approach_query as skill

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "sim/configs/scene_suites/rm65_dmtac_place_v1.json"
QUERY = ("perception.query_grasp_approach", "1")
RELEASE = ("perception.query_support_release_pose", "1")
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


def _color_segment(rgb_rule):
    def segment(image, *, language_prompt, pixel_uv, min_score):
        from deployments.perception.sam2_grounding_dino import GroundedObjectSegment

        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
        rule = rgb_rule.get(language_prompt)
        if rule is None:
            return None
        mask = rule(rgb)
        if mask.sum() < 30:
            return None
        rows, cols = np.nonzero(mask)
        return GroundedObjectSegment(mask=mask, score=0.9, label=language_prompt,
            box_xyxy=(int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)))
    return segment


def _dominant(channel):
    others = [index for index in range(3) if index != channel]
    return lambda rgb: ((rgb[..., channel] > 90) & (rgb[..., channel] > rgb[..., others[0]] + 40)
                        & (rgb[..., channel] > rgb[..., others[1]] + 40))


@pytest.mark.integration
@pytest.mark.optional
# The full physical chain (lift ~65 s, carry ~70 s of guarded 4 kHz physics)
# needs ~190-230 s; the 120 s default stays for every other test.
@pytest.mark.slow_sim
@pytest.mark.timeout(480)
@pytest.mark.parametrize("case", ["box_on_box", "box_near_post"])
def test_native_queried_grasp_then_queried_release_places_the_cube(tmp_path, monkeypatch, case):
    """No scripted pregrasp: query, approach, close, lift, query release, carry, open."""
    pytest.importorskip("mujoco")
    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig,
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from agent_gateway.protocol.contracts import AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1
    from deployments.perception.sam2_grounding_dino import encode_grounded_object_segment_artifacts
    from deployments.perception.sam3_service import Sam3ServiceLocalizer
    from sim.mujoco.deploy import MujocoSimHILTargetConfig
    from sim.mujoco.scene_suite import materialize_case
    from test_agent_gateway_mujoco_tcp_pose_comparison import CLIENT, _request

    task = materialize_case(case, tmp_path / "scene", variant=0, suite_path=SUITE)
    localizer = Sam3ServiceLocalizer(endpoint="http://127.0.0.1:8114", timeout_sec=45., min_score=0.3,
                                     multi_instance_text=False)
    monkeypatch.setattr(Sam3ServiceLocalizer, "start", lambda self: setattr(self, "_test_started", True))
    monkeypatch.setattr(Sam3ServiceLocalizer, "started",
                        property(lambda self: getattr(self, "_test_started", False)))
    monkeypatch.setattr(localizer, "segment", _color_segment(
        {"blue cube": _dominant(2), "green box": _dominant(1)}))
    monkeypatch.setattr(localizer, "encode_segment_artifacts", encode_grounded_object_segment_artifacts)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=task, realtime_worker=False,
            runtime_session_id=CLIENT, control_domain_id=CLIENT,
            execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(CLIENT,), artifact_storage=MujocoAgentArtifactStorageConfig(root=artifacts),
        segmentation_localizer=localizer, enable_grippers=True, enable_grasp_approach=True,
        grasp_contact_target_body="debug_cube", enable_object_state=True,
        enable_object_follow_observation=True, enable_object_motion=True,
        enable_object_support_motion_v2=True, object_support_body_names=("green_box",),
        enable_named_release_v3=True, enable_support_release_queries=True,
        support_release_prompt_by_id={"green_box": "green box"},
        enable_grasp_approach_queries=True, grasp_query_prompt_by_id={"debug_cube": "blue cube"})
    try:
        app = session.application
        manifest = session.contract_bundle.route_manifest
        assert manifest.route(*QUERY).route.control_effect_class == "read_only"
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        sequence = [0]

        def call(skill_id, parameters, version="1"):
            sequence[0] += 1
            request = _request(session, adapter, skill_id=skill_id, version=version,
                               parameters=parameters, sequence=sequence[0])
            outcome = adapter.submit_skill_request(request.to_wire())
            assert outcome["accepted"], outcome
            session.wait(timeout_sec=300.)
            operation_id = outcome["operation"]["operation_id"]
            stop = time.monotonic() + 300.
            operation = session.service.get_operation(operation_id)
            while (operation.status not in AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1
                   and time.monotonic() < stop):
                time.sleep(.05)
                operation = session.service.get_operation(operation_id)
            assert operation.status == "completed", operation
            return operation.execution_result

        frame = session.contract_bundle.grasp_approach_query_registration.reference_frame_id
        grasp = call(QUERY[0], {"object_id": "debug_cube", "reference_frame_id": frame})
        assert grasp["status"] == "grasp_approach_found", grasp
        motion = session.contract_bundle.motion_profile
        lineage = {name: getattr(motion, name) for name in (
            "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
            "tool_config_hash", "calibration_hash")}
        for key, skill_id in (("approach_tcp_pose_xyz_quat_xyzw", "motion.approach_grasp_pose"),
                              ("pregrasp_tcp_pose_xyz_quat_xyzw", "motion.approach_grasp_pose"),
                              ("contact_tcp_pose_xyz_quat_xyzw", "motion.approach_grasp_contact")):
            call(skill_id, {"pose_targets_by_component": {"right_arm": grasp[key]}, **lineage,
                            "controlled_frame_id": "tool_tcp"})
        call("manipulation.set_gripper_state", {"component_id": "right_gripper",
                                                "target_state": "closed"}, "3")
        rig = app.runtime.rig
        for _ in range(round(.65 / rig.config.timing.physics_timestep_sec)):
            rig.step_physics()
        snapshot = session.object_observer.diagnostic_snapshot()
        assert tuple(snapshot["sensor_contacts"]) == (True, True), snapshot
        closed = call(QUERY[0], {"object_id": "debug_cube", "reference_frame_id": frame})
        assert closed["reason"] == "gripper_not_open"

        object_profile = session.contract_bundle.object_motion_contract.motion_profile
        common = {"object_id": "debug_cube", "component_id": "right_arm",
                  "controlled_frame_id": "object_body",
                  **{name: getattr(object_profile, name) for name in lineage}}
        call("motion.move_grasped_object_to_pose",
             {**common, "phase": "lift", "translation_delta_xyz_m": [0., 0., .04]})
        release = call(RELEASE[0], {"support_object_id": "green_box", "reference_frame_id": frame})
        assert release["status"] == "release_pose_found", release
        call("motion.move_grasped_object_to_pose",
             {**common, "phase": "carry",
              "object_pose_xyz_quat_xyzw": release["release_object_pose_xyz_quat_xyzw"]})
        call("manipulation.set_gripper_state", {"component_id": "right_gripper", "target_state": "open",
                                                "release_support_object_id": "green_box"}, "3")
        assert session.object_observer.release_snapshot()["stable_supported_release"] is True

        def native(scene):
            body = int(scene.model.body("green_box").id)
            geom = [g for g in range(scene.model.ngeom) if scene.model.geom_bodyid[g] == body][0]
            return (scene.data.geom_xpos[geom].copy(), scene.model.geom_size[geom].copy(),
                    scene.body_pose("debug_cube").copy())

        center, half, cube = app.capture_passive_application_cut(scene_projector=native).scene_projection
        assert np.all(np.abs(cube[:2, 3] - center[:2]) <= half[:2])
        assert abs(cube[2, 3] - .0175 - (center[2] + half[2])) <= .004
    finally:
        session.close()

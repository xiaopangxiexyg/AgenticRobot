"""perception.query_support_release_pose@1: contract, Gateway wiring and native use."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.mujoco import gateway_cli
from agent_gateway.skills.perception import support_release_query as skill

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "sim/configs/scene_suites/rm65_dmtac_place_v1.json"
QUERY = ("perception.query_support_release_pose", "1")
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


def _green_segment(image, *, language_prompt, pixel_uv, min_score):
    from deployments.perception.sam2_grounding_dino import GroundedObjectSegment

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    mask = (rgb[..., 1] > 90) & (rgb[..., 1] > rgb[..., 0] + 40) & (rgb[..., 1] > rgb[..., 2] + 40)
    if language_prompt != "green box" or mask.sum() < 50:
        return None
    rows, cols = np.nonzero(mask)
    return GroundedObjectSegment(mask=mask, score=0.9, label=language_prompt,
        box_xyxy=(int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)))


@pytest.mark.integration
@pytest.mark.optional
# The full physical chain (lift ~65 s, carry ~70 s of guarded 4 kHz physics)
# needs ~190-230 s; the 120 s default stays for every other test.
@pytest.mark.slow_sim
@pytest.mark.timeout(480)
def test_native_release_pose_is_carried_and_released_on_the_measured_box(tmp_path, monkeypatch):
    """Grasp, lift, query from depth only, carry to the returned pose and release on the box."""
    pytest.importorskip("mujoco")
    from scipy.spatial.transform import Rotation

    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig,
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from deployments.perception.sam2_grounding_dino import encode_grounded_object_segment_artifacts
    from deployments.perception.sam3_service import Sam3ServiceLocalizer
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from sim.mujoco.deploy import MujocoSimHILTargetConfig
    from sim.mujoco.scene_suite import materialize_case
    from sim.mujoco.transforms import pose_to_xyz_quat_xyzw
    from test_agent_gateway_mujoco_grasp_approach import _compact_world_grasp
    from test_agent_gateway_mujoco_tcp_pose_comparison import CLIENT, _request

    task = materialize_case("box_on_box", tmp_path / "scene", variant=0, suite_path=SUITE)
    localizer = Sam3ServiceLocalizer(endpoint="http://127.0.0.1:8114", timeout_sec=45., min_score=0.3,
                                     multi_instance_text=False)
    monkeypatch.setattr(Sam3ServiceLocalizer, "start", lambda self: setattr(self, "_test_started", True))
    monkeypatch.setattr(Sam3ServiceLocalizer, "started",
                        property(lambda self: getattr(self, "_test_started", False)))
    monkeypatch.setattr(localizer, "segment", _green_segment)
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
        support_release_prompt_by_id={"green_box": "green box"})
    try:
        app = session.application
        manifest = session.contract_bundle.route_manifest
        route = manifest.route(*QUERY).route
        assert route.control_effect_class == "read_only" and route.may_move_robot is False
        registration = session.contract_bundle.support_release_query_registration
        assert registration.roles[0] == "wrist_right"
        app.start_observation_ready()
        session.start_perception_resources()
        adapter = session.wire_adapter(authenticated_client_id=CLIENT)
        sequence = [0]

        def call(skill_id, parameters, version="1"):
            from agent_gateway.protocol.contracts import AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1

            sequence[0] += 1
            request = _request(session, adapter, skill_id=skill_id, version=version,
                               parameters=parameters, sequence=sequence[0])
            outcome = adapter.submit_skill_request(request.to_wire())
            assert outcome["accepted"], outcome
            session.wait(timeout_sec=200.)
            operation_id = outcome["operation"]["operation_id"]
            stop = time.monotonic() + 200.
            operation = session.service.get_operation(operation_id)
            # Passive reads run on their own executor, outside the motion wait.
            while (operation.status not in AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1
                   and time.monotonic() < stop):
                time.sleep(.05)
                operation = session.service.get_operation(operation_id)
            assert operation.status == "completed", (
                skill_id, operation.status, operation.reason_code, dict(operation.progress_summary or {}))
            return operation.execution_result

        query = {"support_object_id": "green_box", "reference_frame_id": registration.reference_frame_id}
        before_grasp = call(QUERY[0], query)
        assert before_grasp["status"] == "no_release_pose"
        assert before_grasp["reason"] == "held_object_not_verified_lifted"

        motion = session.contract_bundle.motion_profile

        def move(world, skill_id):
            public = app.transform_tcp_pose_xyzquat(tuple(pose_to_xyz_quat_xyzw(world)),
                source_frame_id="mujoco_sim_hil/world", target_frame_id=motion.reference_frame_id)
            call(skill_id, {"pose_targets_by_component": {"right_arm": public},
                            **{name: getattr(motion, name) for name in (
                                "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
                                "tool_config_hash", "calibration_hash")},
                            "controlled_frame_id": "tool_tcp"})

        grasp = _compact_world_grasp()
        grasp[0, 3] += -.028 + (-.285 + .30)
        grasp[1, 3] += -.02
        grasp[2, 3] += -.033
        grasp[:3, :3] = grasp[:3, :3] @ Rotation.from_euler("x", 6., degrees=True).as_matrix()
        for distance, skill_id in ((-.07, "motion.approach_grasp_pose"),
                                   (-.025, "motion.approach_grasp_pose"),
                                   (.01, "motion.approach_grasp_contact")):
            goal = grasp.copy()
            goal[:3, 3] += grasp[:3, 2] * distance
            move(goal, skill_id)
        call("manipulation.set_gripper_state", {"component_id": "right_gripper",
                                                "target_state": "closed"}, "3")
        rig = app.runtime.rig
        for _ in range(round(.65 / rig.config.timing.physics_timestep_sec)):
            rig.step_physics()
        supported = call(QUERY[0], query)
        assert supported["reason"] == "held_object_not_verified_lifted"

        object_profile = session.contract_bundle.object_motion_contract.motion_profile
        lineage = {name: getattr(object_profile, name) for name in (
            "reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
            "tool_config_hash", "calibration_hash")}
        common = {"object_id": "debug_cube", "component_id": "right_arm",
                  "controlled_frame_id": "object_body", **lineage}
        call("motion.move_grasped_object_to_pose",
             {**common, "phase": "lift", "translation_delta_xyz_m": [0., 0., .04]})
        released = call(QUERY[0], query)
        assert released["status"] == "release_pose_found", released
        measured_views = [view["role"] for view in released["views"] if view["outcome"] == "measured"]
        assert released["segmentation"]["role"] in measured_views

        def native(scene):
            body = int(scene.model.body("green_box").id)
            geom = [g for g in range(scene.model.ngeom) if scene.model.geom_bodyid[g] == body][0]
            return (scene.data.geom_xpos[geom].copy(), scene.model.geom_size[geom].copy(),
                    scene.body_pose("debug_cube").copy())

        center, half, _ = app.capture_passive_application_cut(scene_projector=native).scene_projection
        public_from_world = app.target_frame_profile.frame_graph.resolve(
            "mujoco_sim_hil/world", registration.reference_frame_id).matrix_target_source
        world_release = np.linalg.inv(public_from_world) @ xyzquat_to_pose_matrix(
            released["release_object_pose_xyz_quat_xyzw"])
        # Native truth only grades the depth-only measurement here.
        corners = np.array([[sx, sy, -.0175] for sx in (-.0175, .0175) for sy in (-.0175, .0175)])
        footprint = corners @ world_release[:3, :3].T + world_release[:3, 3]
        assert np.all(np.abs(footprint[:, :2] - center[:2]) <= half[:2])
        drop = world_release[2, 3] - .0175 - (center[2] + half[2])
        assert released["release_clearance_m"] <= drop <= released["release_clearance_m"] + .006

        call("motion.move_grasped_object_to_pose",
             {**common, "phase": "carry",
              "object_pose_xyz_quat_xyzw": released["release_object_pose_xyz_quat_xyzw"]})
        call("manipulation.set_gripper_state", {"component_id": "right_gripper", "target_state": "open",
                                                "release_support_object_id": "green_box"}, "3")
        assert session.object_observer.release_snapshot()["stable_supported_release"] is True
        center, half, cube = app.capture_passive_application_cut(scene_projector=native).scene_projection
        assert np.all(np.abs(cube[:2, 3] - center[:2]) <= half[:2])
        assert abs(cube[2, 3] - .0175 - (center[2] + half[2])) <= .004
    finally:
        session.close()


@pytest.mark.integration
@pytest.mark.optional
# Grasp, lift, two queries, carry and release with guarded 4 kHz physics: ~200 s.
@pytest.mark.slow_sim
@pytest.mark.timeout(480)
def test_native_release_moves_off_an_object_left_where_the_release_was_planned(tmp_path, monkeypatch):
    """A block lands on the mat where the first query would release: the second avoids it."""
    pytest.importorskip("mujoco")
    from scipy.spatial.transform import Rotation

    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig,
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from agent_gateway.protocol.contracts import AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1
    from deployments.perception.sam2_grounding_dino import encode_grounded_object_segment_artifacts
    from deployments.perception.sam3_service import Sam3ServiceLocalizer
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from sim.mujoco.deploy import MujocoSimHILTargetConfig
    from sim.mujoco.scene_suite import materialize_case
    from sim.mujoco.transforms import pose_to_xyz_quat_xyzw
    from test_agent_gateway_mujoco_grasp_approach import _compact_world_grasp
    from test_agent_gateway_mujoco_tcp_pose_comparison import CLIENT, _request

    def green_mat_segment(image, *, language_prompt, pixel_uv, min_score):
        if language_prompt != "green mat":
            return None
        return _green_segment(image, language_prompt="green box", pixel_uv=pixel_uv, min_score=min_score)

    task = materialize_case("mat_disturbance", tmp_path / "scene", variant=0, suite_path=SUITE)
    localizer = Sam3ServiceLocalizer(endpoint="http://127.0.0.1:8114", timeout_sec=45., min_score=0.3,
                                     multi_instance_text=False)
    monkeypatch.setattr(Sam3ServiceLocalizer, "start", lambda self: setattr(self, "_test_started", True))
    monkeypatch.setattr(Sam3ServiceLocalizer, "started",
                        property(lambda self: getattr(self, "_test_started", False)))
    monkeypatch.setattr(localizer, "segment", green_mat_segment)
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
        enable_object_support_motion_v2=True, object_support_body_names=("green_mat",),
        enable_named_release_v3=True, enable_support_release_queries=True,
        support_release_prompt_by_id={"green_mat": "green mat"})
    try:
        app = session.application
        registration = session.contract_bundle.support_release_query_registration
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
            session.wait(timeout_sec=200.)
            operation_id = outcome["operation"]["operation_id"]
            stop = time.monotonic() + 200.
            operation = session.service.get_operation(operation_id)
            while (operation.status not in AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1
                   and time.monotonic() < stop):
                time.sleep(.05)
                operation = session.service.get_operation(operation_id)
            assert operation.status == "completed", (
                skill_id, operation.status, operation.reason_code, dict(operation.progress_summary or {}))
            return operation.execution_result

        motion = session.contract_bundle.motion_profile
        lineage = ("reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash",
                   "tool_config_hash", "calibration_hash")
        grasp = _compact_world_grasp()
        grasp[0, 3] += -.028 + (-.285 + .30)
        grasp[1, 3] += -.02
        grasp[2, 3] += -.033
        grasp[:3, :3] = grasp[:3, :3] @ Rotation.from_euler("x", 6., degrees=True).as_matrix()
        for distance, skill_id in ((-.07, "motion.approach_grasp_pose"),
                                   (-.025, "motion.approach_grasp_pose"),
                                   (.01, "motion.approach_grasp_contact")):
            goal = grasp.copy()
            goal[:3, 3] += grasp[:3, 2] * distance
            public = app.transform_tcp_pose_xyzquat(tuple(pose_to_xyz_quat_xyzw(goal)),
                source_frame_id="mujoco_sim_hil/world", target_frame_id=motion.reference_frame_id)
            call(skill_id, {"pose_targets_by_component": {"right_arm": public},
                            **{name: getattr(motion, name) for name in lineage},
                            "controlled_frame_id": "tool_tcp"})
        call("manipulation.set_gripper_state", {"component_id": "right_gripper",
                                                "target_state": "closed"}, "3")
        rig = app.runtime.rig
        for _ in range(round(.65 / rig.config.timing.physics_timestep_sec)):
            rig.step_physics()
        object_profile = session.contract_bundle.object_motion_contract.motion_profile
        common = {"object_id": "debug_cube", "component_id": "right_arm",
                  "controlled_frame_id": "object_body", **{name: getattr(object_profile, name) for name in lineage}}
        call("motion.move_grasped_object_to_pose",
             {**common, "phase": "lift", "translation_delta_xyz_m": [0., 0., .04]})
        query = {"support_object_id": "green_mat", "reference_frame_id": registration.reference_frame_id}
        first = call("perception.query_support_release_pose", query)
        assert first["status"] == "release_pose_found", first
        public_from_world = app.target_frame_profile.frame_graph.resolve(
            "mujoco_sim_hil/world", registration.reference_frame_id).matrix_target_source
        planned = (np.linalg.inv(public_from_world) @ xyzquat_to_pose_matrix(
            first["release_object_pose_xyz_quat_xyzw"]))[:3, 3]

        def box(scene, name):
            body = int(scene.model.body(name).id)
            geom = [g for g in range(scene.model.ngeom) if scene.model.geom_bodyid[g] == body][0]
            return scene.data.geom_xpos[geom].copy(), scene.model.geom_size[geom].copy()

        def place_block(scene):
            # The environment leaves a block where the first release was planned.
            center, half = box(scene, "green_mat")
            _, own = box(scene, "orange_block")
            model, data, mj = scene.model, scene.data, scene.mujoco
            joint = int(model.body("orange_block").jntadr[0])
            address, dof = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
            xy = np.clip(planned[:2], center[:2] - half[:2] + own[:2] + .001,
                         center[:2] + half[:2] - own[:2] - .001)
            data.qpos[address:address + 3] = (xy[0], xy[1], center[2] + half[2] + own[2] + .001)
            data.qpos[address + 3:address + 7] = (1., 0., 0., 0.)
            data.qvel[dof:dof + 6] = 0.
            mj.mj_forward(model, data)

        rig.capture_scene(place_block)
        for _ in range(round(.5 / rig.config.timing.physics_timestep_sec)):
            rig.step_physics()
        released = call("perception.query_support_release_pose", query)
        assert released["status"] == "release_pose_found", released

        def native(scene):
            return box(scene, "green_mat"), box(scene, "orange_block")

        (center, half), (block, block_half) = app.capture_passive_application_cut(
            scene_projector=native).scene_projection
        world_release = np.linalg.inv(public_from_world) @ xyzquat_to_pose_matrix(
            released["release_object_pose_xyz_quat_xyzw"])
        corners = np.array([[sx, sy, -.0175] for sx in (-.0175, .0175) for sy in (-.0175, .0175)])
        footprint = corners @ world_release[:3, :3].T + world_release[:3, 3]
        # Native truth grades only: on the mat and clear of the block.
        assert np.all(np.abs(footprint[:, :2] - center[:2]) <= half[:2])
        low, high = footprint[:, :2].min(axis=0), footprint[:, :2].max(axis=0)
        assert np.any(low > block[:2] + block_half[:2]) or np.any(high < block[:2] - block_half[:2])

        call("motion.move_grasped_object_to_pose",
             {**common, "phase": "carry",
              "object_pose_xyz_quat_xyzw": released["release_object_pose_xyz_quat_xyzw"]})
        call("manipulation.set_gripper_state", {"component_id": "right_gripper", "target_state": "open",
                                                "release_support_object_id": "green_mat"}, "3")
        assert session.object_observer.release_snapshot()["stable_supported_release"] is True
    finally:
        session.close()

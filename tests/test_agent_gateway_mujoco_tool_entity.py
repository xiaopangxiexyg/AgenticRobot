"""MuJoCo Gateway publication of the shared tool-entity relation and hover Skills."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.mujoco import gateway_cli

pytestmark = [pytest.mark.unit]

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "sim/configs/tasks/rm65_right_dmtacm_blue_cube_70mm_agent_v1.json"
URDF = ROOT / "assets/realman_dual_arm_inspire_gripper_dmtacm/realman_dual_arm_inspire_gripper_dmtacm.urdf"
RELATION = ("perception.describe_tool_object_relations", "1")
HOVER = ("perception.query_tool_hover_paths", "1")


def _common(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(exist_ok=True)
    return ["--socket-path", str(tmp_path / "gateway.sock"), "--task-config", str(TASK),
            "--artifact-root", str(artifacts), "--no-realtime-worker"]


def _observed():
    return ["--enable-safe-tcp-pose", "--safe-tcp-motion-profile", "object_relation",
            "--enable-object-goal-queries", "--enable-surface-anchors", "--enable-surface-anchor-v2",
            "--enable-observed-point-motion", "--safe-motion-observation-urdf", str(URDF),
            "--safe-motion-observation-collision-scene-profile", "dual_arm_hand_proxy_dmtac_grasp",
            "--safe-motion-observation-role", "main"]


def test_tool_entity_cli_flags_are_default_off():
    args = gateway_cli.build_parser().parse_args(["--socket-path", "/unused", "--task-config", "/unused"])
    assert args.enable_tool_object_relations is False and args.enable_tool_hover_queries is False


@pytest.mark.parametrize(("flags", "message"), [
    (["--enable-tool-hover-queries"], "--enable-tool-hover-queries requires"),
    (["--enable-tool-object-relations"], "--enable-tool-object-relations requires --artifact-root"),
    # The frozen multi-view capture belongs to the planned safe TCP owner.
    (["--enable-tool-object-relations", "--artifact-root", "/unused/artifacts"], "--enable-safe-tcp-pose"),
])
def test_tool_entity_prerequisites_fail_before_resources(monkeypatch, capsys, flags, message):
    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session",
                        lambda **kwargs: pytest.fail("prerequisites must fail before assembly"))
    with pytest.raises(SystemExit) as captured:
        gateway_cli.main(["--socket-path", "/unused/socket", "--task-config", "/unused/task.json", *flags])
    assert captured.value.code == 2
    assert message in capsys.readouterr().err


def test_left_motion_side_keeps_the_baseline_only_boundary(monkeypatch, capsys):
    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session",
                        lambda **kwargs: pytest.fail("left side must fail before assembly"))
    with pytest.raises(SystemExit):
        gateway_cli.main(["--socket-path", "/unused/socket", "--task-config", "/unused/task.json",
                          "--motion-side", "left", "--enable-tool-object-relations"])
    assert "baseline TCP/state Routes only" in capsys.readouterr().err


def test_tool_entity_flags_reach_the_factory(monkeypatch, tmp_path):
    from deployments.perception.sam3_service import Sam3ServiceLocalizer

    class AssemblyReached(Exception):
        pass

    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        raise AssemblyReached

    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session", assemble)
    monkeypatch.setattr(Sam3ServiceLocalizer, "start", lambda self: pytest.fail("CLI cannot start SAM3"))
    with pytest.raises(AssemblyReached):
        gateway_cli.main([*_common(tmp_path), *_observed(), "--enable-tool-object-relations",
                          "--enable-tool-hover-queries"])
    assert received["enable_tool_object_relations"] is True
    assert received["enable_tool_hover_queries"] is True
    assert isinstance(received["segmentation_localizer"], Sam3ServiceLocalizer)
    assert received["safe_motion_observation"].start_clearance == "inherited-positive-v1"


def _blue_segment(image, *, language_prompt, pixel_uv, min_score):
    from deployments.perception.sam2_grounding_dino import GroundedObjectSegment

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    mask = (rgb[..., 2] > 90) & (rgb[..., 2] > rgb[..., 0] + 40) & (rgb[..., 2] > rgb[..., 1] + 40)
    if language_prompt != "blue cube" or mask.sum() < 20:
        return None
    rows, cols = np.nonzero(mask)
    return GroundedObjectSegment(mask=mask, score=0.9, label=language_prompt,
        box_xyxy=(int(cols.min()), int(rows.min()), int(cols.max() + 1), int(rows.max() + 1)))


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
def test_native_mujoco_relations_hover_and_safe_move_through_the_gateway(tmp_path, monkeypatch):
    """Relation, anchor and hover queries, then the selected target through motion.move_tcp_to_pose_safe@1."""
    pytest.importorskip("mujoco")
    from agent_gateway.profiles.mujoco.motion_factory import (
        MujocoAgentArtifactStorageConfig,
        build_mujoco_agent_tcp_pose_gateway_session,
        mujoco_agent_tcp_pose_execution_observation_plan_v1,
    )
    from agent_gateway.profiles.mujoco.motion_observed_scene import (
        MujocoObservedSafeMotionConfig,
    )
    from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
    from agent_gateway.protocol.contracts import (
        AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1,
        AgentSkillRequest,
    )
    from deployments.perception.dmtac_tool_entity import default_tool_entity_by_side
    from deployments.perception.sam2_grounding_dino import (
        encode_grounded_object_segment_artifacts,
    )
    from deployments.perception.sam3_service import Sam3ServiceLocalizer
    from robot_motion.observed_scene import SafeMotionObservationConfig
    from robot_motion.relative_action import xyzquat_to_pose_matrix
    from robot_motion.rm65.tool_profiles import load_rm65_tool_profile
    from robot_motion.tool_entity_scene import pose_matrix
    from sim.mujoco.deploy import MujocoSimHILTargetConfig

    localizer = Sam3ServiceLocalizer(endpoint="http://127.0.0.1:8114", timeout_sec=45., min_score=0.3,
                                     multi_instance_text=False)
    monkeypatch.setattr(Sam3ServiceLocalizer, "start", lambda self: setattr(self, "_test_started", True))
    monkeypatch.setattr(Sam3ServiceLocalizer, "started", property(lambda self: getattr(self, "_test_started", False)))
    monkeypatch.setattr(localizer, "segment", _blue_segment)
    monkeypatch.setattr(localizer, "encode_segment_artifacts", encode_grounded_object_segment_artifacts)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    client = "test:tool-entity"
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(task_config_path=TASK, realtime_worker=False,
            runtime_session_id=client, execution_observation_plan=mujoco_agent_tcp_pose_execution_observation_plan_v1(),
            exposed_pose_sides=("right",), allow_scripted_task_reset=False),
        allowed_client_ids=(client,), artifact_storage=MujocoAgentArtifactStorageConfig(root=artifacts),
        max_retained_operations=8, segmentation_localizer=localizer, enable_safe_tcp_pose=True,
        safe_tcp_motion_profile="object_relation", enable_object_goal_queries=True, enable_surface_anchors=True,
        enable_surface_anchor_v2=True, enable_observed_point_motion=True,
        safe_motion_observation=MujocoObservedSafeMotionConfig(robot_urdf_path=URDF,
            collision_scene_profile="dual_arm_hand_proxy_dmtac_grasp",
            observation=SafeMotionObservationConfig(roles=("main",)), start_clearance="inherited-positive-v1"),
        enable_tool_object_relations=True, enable_tool_hover_queries=True)
    try:
        app = session.application
        manifest = session.contract_bundle.route_manifest
        for key in (RELATION, HOVER):
            route = manifest.route(*key).route
            assert route.control_effect_class == "read_only" and route.may_move_robot is False
        app.start_observation_ready()
        session.start_perception_resources()
        published = session.observation_producer.capture_and_publish(session.service)
        assert published.accepted
        snapshot = published.snapshot
        sequence = [0]

        def run(skill, parameters, bindings=(), timeout_ms=40_000):
            entry = manifest.route(*skill)
            route = entry.route
            profile = route.allowed_parameter_profiles[0]
            sequence[0] += 1
            request = AgentSkillRequest(client_id=client, request_id=f"request:{sequence[0]}",
                idempotency_key=f"tool:{sequence[0]}", client_sequence=sequence[0],
                runtime_session_id=manifest.runtime_session_id, skill_route_manifest_hash=manifest.semantic_content_hash,
                expected_station_revision=snapshot.station_revision, skill_id=route.skill_id,
                skill_version=route.skill_version, skill_definition_hash=route.skill_definition_hash,
                route_id=route.route_id, route_version=route.route_version, route_hash=route.semantic_content_hash,
                deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
                parameter_profile_id=profile.parameter_profile_id, parameter_profile_hash=profile.parameter_profile_hash,
                parameters=parameters, observation_snapshot_id=snapshot.snapshot_id, requested_mode="sim_active",
                requested_timeout_ms=timeout_ms, input_artifact_bindings=tuple(bindings))
            admitted = session.service.submit_request(request=request, current_station_revision=snapshot.station_revision)
            assert admitted.accepted, admitted
            operation_id = admitted.operation.operation_id
            assert session.service.launch_operation(operation_id).accepted
            stop = time.monotonic() + timeout_ms / 1000 + 10
            completed = session.service.get_operation(operation_id)
            while completed.status not in AGENT_SKILL_TERMINAL_OPERATION_STATUSES_V1 and time.monotonic() < stop:
                time.sleep(0.1)
                completed = session.service.get_operation(operation_id)
            assert completed.status == "completed", completed
            return completed

        arms = {side: [app.scene.joint_qpos_addr(name) for name in app.scene.arm_specs[side].joint_names]
                for side in ("left", "right")}
        before = app.scene.data.qpos.copy()
        relations = run(RELATION, {"objects": [{"label": "cube", "language_prompt": "blue cube"}],
                                   "roles": None, "min_score": 0.3}, timeout_ms=60_000).execution_result
        assert relations["motion_executed"] is False and list(relations["evaluated_sides"]) == ["left", "right"]
        assert relations["views"], relations["unmeasured"]
        assert {item["relation"] for item in relations["objects"]} <= {"far", "near"}
        anchor = run(("perception.capture_surface_anchor", "2"), {"role": "main", "selection_mode": "mask_interior",
            "target_pixel_uv": None, "language_prompt": "blue cube", "min_score": 0.3}, timeout_ms=60_000)
        binding = AgentArtifactBinding(slot_id="surface_anchor", ordinal=0, artifact=next(
            item.artifact for item in anchor.output_artifact_bindings if item.slot_id == "surface_anchor"))
        registration = session.contract_bundle.tool_entity_registrations[-1]
        common = {"component_id": "right_arm", "reference_frame_id": registration.reference_frame_id,
                  "tip_offset_bounds_xy_m": [[-0.02, 0.02], [-0.02, 0.02]], "clearance_bounds_m": [0.03, 0.10],
                  "tool_axis_max_tilt_rad": 0.0, "max_candidates": 8}
        # A preserved horizontal tool above this cube puts the wrist into the
        # twin's central camera pole (the same fixed geometry as the station
        # URDF); every proposal must be refused, never silently accepted.
        preserved = run(HOVER, {**common, "orientation_policy": "preserve"},
                        bindings=(binding,)).execution_result["hover_query"]
        assert preserved["status"] == "path_rejected", preserved["reason"]
        assert {item["reason"] for item in preserved["candidate_attempts"]} == {"safe_tcp_target_blocked"}
        down = run(HOVER, {**common, "orientation_policy": "tool_axis_down"}, bindings=(binding,))
        registration.payload_validator.validate_execution_result_for_parameters(
            {**common, "orientation_policy": "tool_axis_down"}, down.execution_result)
        query = down.execution_result["hover_query"]
        assert query["motion_executed"] is False and query["execution_authorized"] is False
        assert query["start_clearance"] == "inherited-positive-v1" and query["candidate_budget_sec"] == 5.
        assert query["status"] == "path_found", [(a["status"], a["reason"]) for a in query["candidate_attempts"]]
        assert query["achieved"]["inside_requested_region"] is True
        # Read-only Skills never moved the simulated arms.
        for indices in arms.values():
            np.testing.assert_allclose(app.scene.data.qpos[indices], before[indices], atol=1e-12)

        # Two-step use: the selected target goes unchanged to the reviewed safe motion Skill.
        safe_profile = session.contract_bundle.safe_tcp_pose_contract.motion_profile
        assert safe_profile.reference_frame_id == query["reference_frame_id"]
        adapter = session.wire_adapter(authenticated_client_id=client)
        observation = adapter.get_observation_snapshot()
        entry = manifest.route("motion.move_tcp_to_pose_safe", "1")
        route = entry.route
        parameter = route.allowed_parameter_profiles[0]
        base = session.contract_bundle.motion_profile
        # The pose the hover planner reached and re-measured; preserve_target keeps it.
        target = [float(x) for x in query["planned_endpoint_pose_xyz_quat_xyzw"]]
        sequence[0] += 1
        move = AgentSkillRequest(client_id=client, request_id=f"request:{sequence[0]}",
            idempotency_key=f"tool:{sequence[0]}", client_sequence=sequence[0],
            runtime_session_id=manifest.runtime_session_id, skill_route_manifest_hash=manifest.semantic_content_hash,
            expected_station_revision=int(observation["station_revision"]), skill_id=route.skill_id,
            skill_version=route.skill_version, skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id, route_version=route.route_version, route_hash=route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=parameter.parameter_profile_id, parameter_profile_hash=parameter.parameter_profile_hash,
            parameters={"pose_targets_by_component": {base.component_id: target},
                "reference_frame_id": base.reference_frame_id, "reference_frame_spec_hash": base.reference_frame_spec_hash,
                "controlled_frame_id": "tool_tcp", "frame_graph_hash": base.frame_graph_hash,
                "tool_config_hash": base.tool_config_hash, "calibration_hash": base.calibration_hash,
                # Keep the executed hover inside the certified bounds (see the Skill summary).
                "pose_goal_policy": "preserve_target"},
            observation_snapshot_id=observation["snapshot_id"], requested_mode="sim_active",
            requested_timeout_ms=min(120_000, route.max_operation_duration_ms))
        admitted = adapter.submit_skill_request(move.to_wire())
        assert admitted["accepted"], admitted
        assert session.wait(timeout_sec=90)
        moved = session.service.operation_store.get(admitted["operation"]["operation_id"])
        assert moved.execution_outcome == "completed", moved
        final = adapter.get_observation_snapshot()["measurement_summary"]["pose_xyz_quat_xyzw_by_component"][
            base.component_id]
        assert np.linalg.norm(np.asarray(final[:3]) - np.asarray(target[:3])) <= safe_profile.translation_tolerance_m
        # The executed hover against the native cube: the gripper box clears it
        # inside the requested band (plus arrival tolerance), tip above it, no contact.
        profiles = {side: load_rm65_tool_profile(getattr(app.task_config, side + "_tool_profile_path"), side=side)
                    for side in ("left", "right")}
        entity = default_tool_entity_by_side({side: tuple(item["tcp"]["pose_flange_m_rad"])
                                              for side, item in profiles.items()}, profiles)["right"]
        capture_port = session.back_projection_provider.capture_port
        world_from_public = xyzquat_to_pose_matrix(app.transform_tcp_pose_xyzquat((0., 0., 0., 0., 0., 0., 1.),
            source_frame_id=query["reference_frame_id"], target_frame_id=capture_port.source_frame_id))
        tcp_world = world_from_public @ pose_matrix(final)
        model, mujoco = app.scene.model, app.scene.mujoco
        cube = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "debug_cube_collision")
        cube_top = float(app.scene.data.geom_xpos[cube][2] + model.geom_size[cube][2])
        clearance = float(entity.corners_world(tcp_world)[:, 2].min()) - cube_top
        assert 0.03 - 0.005 <= clearance <= 0.10 + 0.005, clearance
        tip = entity.tip_world(tcp_world)
        assert np.linalg.norm(tip[:2] - app.scene.data.geom_xpos[cube][:2]) <= model.geom_size[cube][0] + 0.02
        touching = [(int(c.geom1), int(c.geom2)) for c in app.scene.data.contact[:app.scene.data.ncon]
                    if cube in (int(c.geom1), int(c.geom2))
                    and "table" not in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM,
                        int(c.geom2 if int(c.geom1) == cube else c.geom1)) or "")]
        assert touching == []
        print("native tool-entity hover", {"clearance_m": clearance, "tip_xy_world": tip[:2].tolist(),
              "selected_candidate_index": query["selected_candidate_index"],
              "hover_wall_sec": query["path_planning_wall_sec"], "relations": [
                  (item["side"], item["relation"]) for item in relations["objects"]]})
    finally:
        session.close()


def test_mujoco_hover_budgets_are_forwarded_and_checked(monkeypatch, tmp_path, capsys):
    class AssemblyReached(Exception):
        pass

    received = {}

    def assemble(**kwargs):
        received.update(kwargs)
        raise AssemblyReached

    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session", assemble)
    with pytest.raises(AssemblyReached):
        gateway_cli.main([*_common(tmp_path), *_observed(), "--enable-tool-hover-queries",
                          "--tool-hover-candidate-budget-sec", "15", "--tool-hover-region-budget-sec", "60"])
    assert (received["tool_hover_candidate_budget_sec"], received["tool_hover_region_budget_sec"]) == (15., 60.)
    monkeypatch.setattr(gateway_cli, "build_mujoco_agent_tcp_pose_gateway_session",
                        lambda **kwargs: pytest.fail("invalid budgets must fail before assembly"))
    for flags, message in ((["--tool-hover-region-budget-sec", "60"], "requires --enable-tool-hover-queries"),
                           (["--enable-tool-hover-queries", "--tool-hover-candidate-budget-sec", "30"],
                            "invalid tool hover budget")):
        with pytest.raises(SystemExit) as captured:
            gateway_cli.main([*_common(tmp_path), *_observed(), *flags])
        assert captured.value.code == 2
        assert message in capsys.readouterr().err

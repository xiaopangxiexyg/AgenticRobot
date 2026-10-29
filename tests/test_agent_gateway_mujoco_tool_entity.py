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
def test_native_mujoco_relation_anchor_and_hover_run_through_the_gateway(tmp_path, monkeypatch):
    """Relation, surface anchor and hover requests through the admitted MuJoCo service."""
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
    from deployments.perception.sam2_grounding_dino import (
        encode_grounded_object_segment_artifacts,
    )
    from deployments.perception.sam3_service import Sam3ServiceLocalizer
    from robot_motion.observed_scene import SafeMotionObservationConfig
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
        manifest = session.contract_bundle.route_manifest
        for key in (RELATION, HOVER):
            route = manifest.route(*key).route
            assert route.control_effect_class == "read_only" and route.may_move_robot is False
        session.application.start_observation_ready()
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

        before = session.application.scene.data.qpos.copy()
        relations = run(RELATION, {"objects": [{"label": "cube", "language_prompt": "blue cube"}],
                                   "roles": None, "min_score": 0.3}, timeout_ms=60_000).execution_result
        assert relations["motion_executed"] is False and list(relations["evaluated_sides"]) == ["left", "right"]
        assert relations["views"], relations["unmeasured"]
        assert {item["relation"] for item in relations["objects"]} <= {"far", "near"}
        anchor = run(("perception.capture_surface_anchor", "2"), {"role": "main", "selection_mode": "mask_interior",
            "target_pixel_uv": None, "language_prompt": "blue cube", "min_score": 0.3}, timeout_ms=60_000)
        artifact = next(item.artifact for item in anchor.output_artifact_bindings if item.slot_id == "surface_anchor")
        reference = session.contract_bundle.tool_entity_registrations[-1].reference_frame_id
        hover = run(HOVER, {"component_id": "right_arm", "reference_frame_id": reference,
                            "tip_offset_bounds_xy_m": [[-0.02, 0.02], [-0.02, 0.02]], "clearance_bounds_m": [0.03, 0.10],
                            "orientation_policy": "preserve", "tool_axis_max_tilt_rad": 0.0, "max_candidates": 8},
                    bindings=(AgentArtifactBinding(slot_id="surface_anchor", ordinal=0, artifact=artifact),))
        query = hover.execution_result["hover_query"]
        assert query["motion_executed"] is False and query["execution_authorized"] is False
        assert query["status"] in {"path_found", "query_incomplete", "path_rejected"}
        assert query["start_clearance"] == "inherited-positive-v1"
        if query["status"] == "path_found":
            assert query["achieved"]["inside_requested_region"] is True
            assert len(query["target_tcp_pose_xyz_quat_xyzw"]) == 7
        # Read-only Skills never moved the simulated arms.
        np.testing.assert_allclose(session.application.scene.data.qpos[:12], before[:12], atol=1e-9)
        print("native tool-entity", [(item["side"], item["label"], item["relation"], item["surface_distance_m"])
              for item in relations["objects"]], {k: query[k] for k in (
            "status", "reason", "selected_candidate_index", "achieved", "attempted_candidate_count",
            "path_planning_wall_sec", "observed_voxel_count", "target_tcp_pose_xyz_quat_xyzw")},
              [(a["candidate_index"], a["status"], a["reason"]) for a in query["candidate_attempts"]])
    finally:
        session.close()

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.motion_factory import (
    MujocoAgentArtifactStorageConfig,
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from robot_motion.deploy.schema import xyzquat_to_pose_matrix
from sim.mujoco.deploy import (
    MujocoSimHILTargetConfig,
    prepare_mujoco_sim_hil_target,
)
from sim.mujoco.pointcloud import pointcloud_from_depth

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PREGRASP_TASK = (
    _REPOSITORY_ROOT
    / "sim/configs/tasks/"
    "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
)


def test_raw_and_encoded_depth_cuts_share_content_and_capture_identity():
    from types import SimpleNamespace
    from agent_gateway.profiles.mujoco.back_projection import (
        live_aligned_depth_capture, live_aligned_depth_capture_from_array,
    )
    from sim.mujoco.depth_snapshot import MujocoRawRgbdView, _encoded_depth_frame

    depth = np.asfortranarray([[1., float("nan")], [-1., 2.]], dtype=np.float64)
    metadata = dict(width=2, height=2, fovy_deg=60.,
        camera_pose_world_xyz_quat_xyzw=(0., 0., 1., 0., 0., 0., 1.))
    encoded = _encoded_depth_frame(view_id="front", raw_depth=depth, metadata=metadata, captured_at=12.5)
    view = MujocoRawRgbdView(view_id="front", rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=depth, optical_camera_pose_world_xyz_quat_xyzw=encoded.camera_pose_world_xyz_quat_xyzw,
        intrinsics=encoded.intrinsics)
    args = dict(source_id="front", role="main", station_revision=3,
        application=SimpleNamespace(transform_tcp_pose_xyzquat=lambda pose, **kw: pose),
        source_frame_id="world", reference_frame_id="station")
    before = live_aligned_depth_capture(encoded, **args)
    after = live_aligned_depth_capture_from_array(view.depth_m, captured_at=encoded.captured_at,
        camera_pose_world_xyz_quat_xyzw=view.optical_camera_pose_world_xyz_quat_xyzw,
        intrinsics=view.intrinsics, **args)
    assert before.depth_content_hash == after.depth_content_hash
    assert before.capture_group_id == after.capture_group_id
    assert before.captured_at == after.captured_at == 12.5
    np.testing.assert_array_equal(before.depth_m, after.depth_m)
    np.testing.assert_array_equal(before.reference_from_camera, after.reference_from_camera)


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
def test_mujoco_back_project_skill_returns_scene_frame_point(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")

    initial = prepare_mujoco_sim_hil_target(
        MujocoSimHILTargetConfig(
            task_config_path=_PREGRASP_TASK,
            realtime_worker=False,
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=True,
        )
    )
    task_path = tmp_path / "back_project_pregrasp_initial_state.json"
    task_path.write_text(
        json.dumps(
            {
                "extends": str(_PREGRASP_TASK),
                "name": "mujoco_back_project_skill_acceptance",
                "right_home_qpos": initial.scene.get_arm_qpos("right").tolist(),
                "scripted_pregrasp_reset": False,
                "scripted_contact_reset": False,
            }
        ),
        encoding="utf-8",
    )
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    client_id = "test:mujoco-back-project"
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=MujocoSimHILTargetConfig(
            task_config_path=task_path,
            realtime_worker=False,
            runtime_session_id="mujoco-back-project-skill-acceptance",
            execution_observation_plan=(
                mujoco_agent_tcp_pose_execution_observation_plan_v1()
            ),
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=False,
        ),
        allowed_client_ids=(client_id,),
        artifact_storage=MujocoAgentArtifactStorageConfig(root=artifact_root),
    )
    try:
        application = session.application
        application.start_observation_ready()
        session.start_perception_resources()
        observation = session.observation_producer.capture_and_publish(
            session.service
        )
        assert observation.accepted and observation.snapshot is not None
        snapshot = observation.snapshot

        frame = application.capture_depth_npy_frame(view_id="front")
        depth = np.load(io.BytesIO(frame.data), allow_pickle=False)
        cloud = pointcloud_from_depth(
            depth,
            fovy_deg=float(frame.intrinsics["fovy_deg"]),
            T_world_camera=xyzquat_to_pose_matrix(
                frame.camera_pose_world_xyz_quat_xyzw
            ),
            depth_min_m=0.05,
            depth_max_m=5.0,
        )
        cube_world = np.asarray(
            application.scene.data.body(
                application.task_config.object_body_name
            ).xpos,
            dtype=np.float64,
        )
        cube_distances = np.linalg.norm(cloud.points_world - cube_world, axis=1)
        pixel_index = int(np.argmin(cube_distances))
        assert cube_distances[pixel_index] < 0.03
        pixel_uv = cloud.uv[pixel_index].tolist()
        expected_world = cloud.points_world[pixel_index].astype(np.float64)

        depth_binding = session.contract_bundle.depth_capture.deployment_binding
        station_from_world = application.target_frame_profile.frame_graph.resolve(
            depth_binding.source_frame_id,
            depth_binding.reference_frame_id,
        ).matrix_target_source
        expected_station = (
            station_from_world @ np.append(expected_world, 1.0)
        )[:3]

        entry = session.contract_bundle.route_manifest.route(
            "perception.back_project", "1"
        )
        route = entry.route
        parameter_profile = route.allowed_parameter_profiles[0]
        assert route.control_effect_class == "read_only"
        assert route.may_move_robot is False
        request = AgentSkillRequest(
            client_id=client_id,
            request_id="request:mujoco-back-project:1",
            idempotency_key="mujoco-back-project:1",
            client_sequence=1,
            runtime_session_id=(
                session.contract_bundle.route_manifest.runtime_session_id
            ),
            skill_route_manifest_hash=(
                session.contract_bundle.route_manifest.semantic_content_hash
            ),
            expected_station_revision=snapshot.station_revision,
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
            parameter_profile_id=parameter_profile.parameter_profile_id,
            parameter_profile_hash=parameter_profile.parameter_profile_hash,
            parameters={
                "role": "main",
                "target_pixel_uv": pixel_uv,
                "pixel_region_xyxy": None,
                "world_z_min_m": None,
                "world_z_max_m": None,
            },
            observation_snapshot_id=snapshot.snapshot_id,
            requested_mode="sim_active",
            requested_timeout_ms=3_000,
        )
        admitted = session.service.submit_request(
            request=request,
            current_station_revision=snapshot.station_revision,
        )
        assert admitted.accepted and admitted.operation is not None
        assert session.service.launch_operation(
            admitted.operation.operation_id
        ).accepted
        assert session.back_projection_executor is not None
        assert session.back_projection_executor.wait(
            admitted.operation.operation_id,
            timeout_sec=10.0,
        )
        completed = session.service.get_operation(admitted.operation.operation_id)
        assert completed is not None and completed.status == "completed"
        actual = np.asarray(completed.execution_result["world_xyz_m"])
        np.testing.assert_allclose(actual, expected_station, rtol=0.0, atol=0.01)
        assert completed.execution_result["depth_m"] == pytest.approx(
            float(depth[pixel_uv[1], pixel_uv[0]]), abs=1e-4
        )
        assert completed.execution_result["reference_frame_id"] == (
            depth_binding.reference_frame_id
        )
    finally:
        session.close()

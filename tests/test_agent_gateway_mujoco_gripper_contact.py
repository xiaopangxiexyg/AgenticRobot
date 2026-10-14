from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.mujoco.gripper_skill import (
    MujocoGripperControlPort,
    assemble_mujoco_gripper_agent_gateway_session,
)
from agent_gateway.profiles.mujoco.motion_factory import (
    MujocoAgentArtifactStorageConfig,
    build_mujoco_agent_tcp_pose_gateway_session,
    mujoco_agent_tcp_pose_execution_observation_plan_v1,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.skills.gripper.runtime import GripperConvergenceProfile
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectSegment,
    encode_grounded_object_segment_artifacts,
)
from robot_motion.deploy.schema import (
    pose_matrix_to_xyzquat,
    xyzquat_to_pose_matrix,
)
from sim.mujoco.deploy import (
    MujocoSimHILTargetConfig,
    capture_mujoco_sim_hil_tool_tcp_pose_by_side,
    prepare_mujoco_sim_hil_target,
)
from sim.mujoco.pointcloud import pointcloud_from_depth

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PREGRASP_TASK = (
    _REPOSITORY_ROOT
    / "sim/configs/tasks/"
    "rm65_right_cube_lift_rl_normal_pregrasp_bridge_v5_cube_pose.json"
)


def _motion_request(session, snapshot, target_pose, *, client_id: str) -> AgentSkillRequest:
    bundle = session.contract_bundle
    entry = bundle.route_manifest.route("motion.move_tcp_to_pose", "1")
    route = entry.route
    binding = bundle.deployment_binding
    profile = bundle.motion_profile
    return AgentSkillRequest(
        client_id=client_id,
        request_id="mujoco-contact:lower-to-grasp",
        idempotency_key="mujoco-contact:lower-to-grasp",
        client_sequence=1,
        runtime_session_id=binding.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=snapshot.station_revision,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.semantic_content_hash,
        parameter_profile_id=profile.profile_id,
        parameter_profile_hash=profile.semantic_content_hash,
        parameters={
            "pose_targets_by_component": {
                "right_arm": [float(value) for value in target_pose],
            },
            "reference_frame_id": profile.reference_frame_id,
            "reference_frame_spec_hash": profile.reference_frame_spec_hash,
            "controlled_frame_id": "tool_tcp",
            "frame_graph_hash": profile.frame_graph_hash,
            "tool_config_hash": profile.tool_config_hash,
            "calibration_hash": profile.calibration_hash,
        },
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="sim_active",
        requested_timeout_ms=10_000,
    )


def _gripper_request(bundle, snapshot, *, client_id: str) -> AgentSkillRequest:
    entry = bundle.route_manifest.skill_routes[0]
    route = entry.route
    parameter_profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=client_id,
        request_id="mujoco-contact:close-right-gripper",
        idempotency_key="mujoco-contact:close-right-gripper",
        client_sequence=1,
        runtime_session_id=bundle.route_manifest.runtime_session_id,
        skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
        expected_station_revision=snapshot.station_revision,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=(
            bundle.deployment_binding.semantic_content_hash
        ),
        parameter_profile_id=parameter_profile.parameter_profile_id,
        parameter_profile_hash=parameter_profile.parameter_profile_hash,
        parameters={"target_state": "closed"},
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode=bundle.route_manifest.current_mode,
        requested_timeout_ms=5_000,
    )


class _MuJoCoGroundTruthSegmenter:
    semantic_content_hash = "test:mujoco-ground-truth-segmenter"
    model_spec = SimpleNamespace(
        model_profile_id="test:mujoco-ground-truth-model",
        semantic_content_hash="test:mujoco-ground-truth-model-hash",
    )
    encode_segment_artifacts = staticmethod(encode_grounded_object_segment_artifacts)

    def __init__(self) -> None:
        self.masks: dict[str, np.ndarray] = {}
        self.started = False

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False

    def segment(self, image, *, language_prompt, pixel_uv, min_score):
        assert self.started
        assert pixel_uv is None
        assert image.size == (640, 480)
        mask = self.masks[language_prompt]
        ys, xs = np.nonzero(mask)
        return GroundedObjectSegment(
            mask=mask,
            score=max(0.95, min_score),
            label=language_prompt,
            box_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max()) + 1,
                int(ys.max()) + 1,
            ),
        )


def _perception_request(
    session,
    snapshot,
    *,
    client_id: str,
    skill_id: str,
    sequence: int,
    parameters: dict[str, object],
    timeout_ms: int,
) -> AgentSkillRequest:
    entry = session.contract_bundle.route_manifest.route(skill_id, "1")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=client_id,
        request_id=f"mujoco-perception:{skill_id}:{sequence}",
        idempotency_key=f"mujoco-perception:{skill_id}:{sequence}",
        client_sequence=sequence,
        runtime_session_id=session.contract_bundle.route_manifest.runtime_session_id,
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
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
        observation_snapshot_id=snapshot.snapshot_id,
        requested_mode="sim_active",
        requested_timeout_ms=timeout_ms,
    )


@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow_sim
def test_agent_skill_target_actuates_gripper_into_real_mujoco_cube_contact(
    tmp_path: Path,
) -> None:
    pytest.importorskip("mujoco")

    # Reuse the task-owned pregrasp joint solution as a declared initial state.
    # The actual Agent target below disables both scripted reset motions.
    pregrasp = prepare_mujoco_sim_hil_target(
        MujocoSimHILTargetConfig(
            task_config_path=_PREGRASP_TASK,
            realtime_worker=False,
            exposed_pose_sides=("right",),
            allow_scripted_task_reset=True,
        )
    )
    task_path = tmp_path / "agent_skill_pregrasp_initial_state.json"
    task_path.write_text(
        json.dumps(
            {
                "extends": str(_PREGRASP_TASK),
                "name": "agent_skill_gripper_contact_acceptance",
                "right_home_qpos": pregrasp.scene.get_arm_qpos("right").tolist(),
                "scripted_pregrasp_reset": False,
                "scripted_contact_reset": False,
            }
        ),
        encoding="utf-8",
    )
    config = MujocoSimHILTargetConfig(
        task_config_path=task_path,
        realtime_worker=False,
        runtime_session_id="mujoco-agent-gripper-contact-test",
        execution_observation_plan=(
            mujoco_agent_tcp_pose_execution_observation_plan_v1()
        ),
        exposed_pose_sides=("right",),
        allow_scripted_task_reset=False,
    )
    client_id = "test:mujoco-gripper-contact"
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    segmenter = _MuJoCoGroundTruthSegmenter()
    session = build_mujoco_agent_tcp_pose_gateway_session(
        target_config=config,
        allowed_client_ids=(client_id,),
        artifact_storage=MujocoAgentArtifactStorageConfig(root=artifact_root),
        segmentation_localizer=segmenter,
    )
    gripper_session = None
    try:
        app = session.application
        scene = app.scene
        task = app.task_config
        assert task.scripted_pregrasp_reset is False
        assert task.scripted_contact_reset is False
        assert tuple(app.runtime.action_schema.component_ids) == ("right_arm",)
        assert app.runtime.rig.config.gripper_actuation_profile is not None

        # Lower the already aligned pregrasp to the block with the normal Agent
        # TCP Route. All object/gripper contact below comes from MuJoCo physics.
        cube_position = np.asarray(
            scene.data.body(task.object_body_name).xpos,
            dtype=np.float64,
        ).copy()
        np.testing.assert_allclose(
            cube_position,
            np.asarray(task.cube_spawn_qpos[:3], dtype=np.float64),
            rtol=0.0,
            atol=1e-9,
        )
        tool_tcp_arm_base = capture_mujoco_sim_hil_tool_tcp_pose_by_side(
            scene,
            task,
        )["right"]
        world_from_arm_base = scene.body_pose("right_arm_base_link")
        world_from_tcp = world_from_arm_base @ tool_tcp_arm_base
        grasp_body = scene.data.body(task.right_grasp_body_name)
        world_from_grasp = np.eye(4, dtype=np.float64)
        world_from_grasp[:3, :3] = np.asarray(grasp_body.xmat).reshape(3, 3)
        world_from_grasp[:3, 3] = grasp_body.xpos
        tcp_from_grasp = np.linalg.inv(world_from_tcp) @ world_from_grasp
        target_grasp = np.array(world_from_grasp, copy=True)
        target_grasp[:3, 3] = cube_position + np.asarray((0.0, 0.0, 0.010))
        world_from_target_tcp = target_grasp @ np.linalg.inv(tcp_from_grasp)
        station_from_world = app.target_frame_profile.frame_graph.resolve(
            "mujoco_sim_hil/world",
            session.contract_bundle.motion_profile.reference_frame_id,
        ).matrix_target_source
        target_pose = pose_matrix_to_xyzquat(
            station_from_world @ world_from_target_tcp
        )
        assert target_pose is not None

        app.start_observation_ready()
        session.start_perception_resources()
        observation = session.observation_producer.capture_and_publish(
            session.service
        )
        assert observation.accepted and observation.snapshot is not None
        motion_request = _motion_request(
            session,
            observation.snapshot,
            target_pose,
            client_id=client_id,
        )
        motion_submission = session.service.submit_request(
            request=motion_request,
            current_station_revision=observation.snapshot.station_revision,
        )
        assert motion_submission.accepted and motion_submission.operation is not None
        assert session.service.launch_operation(
            motion_submission.operation.operation_id
        ).accepted
        assert session.motion_executor.wait(timeout_sec=20.0)
        motion_operation = session.service.get_operation(
            motion_submission.operation.operation_id
        )
        assert motion_operation is not None
        assert motion_operation.execution_outcome == "completed"
        assert motion_operation.task_outcome == "succeeded"

        target_binding = session.contract_bundle.deployment_binding
        port = MujocoGripperControlPort(
            rig=app.runtime.rig,
            gripper_name=task.right_gripper_name,
            clock=app.runtime.clock,
        )
        gripper_session = assemble_mujoco_gripper_agent_gateway_session(
            runtime_session_id=target_binding.runtime_session_id,
            control_domain_id=target_binding.control_domain_id,
            target_binding_id=target_binding.target_binding_id,
            station_manifest_hash=(
                session.contract_bundle.station_manifest.semantic_content_hash
            ),
            execution_manifest_hash=(
                session.contract_bundle.execution_manifest.semantic_content_hash
            ),
            port=port,
            allowed_client_ids=(client_id,),
            current_station_revision=lambda: (
                session.observation_producer.current_station_revision
            ),
            clock_domain_id=config.clock_domain_id,
            # Deterministic physics has one owner; poll each physics tick so
            # contact settle time and MuJoCo time advance together.
            convergence=GripperConvergenceProfile(poll_interval_sec=0.0),
        )
        gripper_observation = gripper_session.publish_observation()
        gripper_request = _gripper_request(
            gripper_session.contract_bundle,
            gripper_observation,
            client_id=client_id,
        )
        gripper_submission = gripper_session.service.submit_request(
            request=gripper_request,
            current_station_revision=gripper_observation.station_revision,
        )
        assert gripper_submission.accepted and gripper_submission.operation is not None
        assert gripper_session.service.launch_operation(
            gripper_submission.operation.operation_id
        ).accepted
        assert gripper_session.wait(timeout_sec=15.0)
        gripper_operation = gripper_session.service.get_operation(
            gripper_submission.operation.operation_id
        )
        assert gripper_operation is not None
        assert gripper_operation.execution_outcome == "completed"
        result = dict(gripper_operation.execution_result or {})
        assert result["commanded_state"] == "closed"
        assert result["completion_kind"] == "settled_before_full_close"
        assert result["terminal_hold_confirmed"] is True

        model = scene.model
        cube_body_id = int(model.body(task.object_body_name).id)
        finger_body_id_by_side = {
            "left": int(model.body(task.grasp_latch_left_finger_body_name).id),
            "right": int(model.body(task.grasp_latch_right_finger_body_name).id),
        }
        cube_geoms = {
            geom_id
            for geom_id, body_id in enumerate(model.geom_bodyid)
            if int(body_id) == cube_body_id
        }
        finger_geoms_by_side = {
            side: {
                geom_id
                for geom_id, body_id in enumerate(model.geom_bodyid)
                if int(body_id) == finger_body_id
            }
            for side, finger_body_id in finger_body_id_by_side.items()
        }
        contacted_sides: set[str] = set()
        for index in range(int(scene.data.ncon)):
            contact = scene.data.contact[index]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            if float(contact.dist) >= 0.0:
                continue
            if geom_a in cube_geoms:
                other_geom = geom_b
            elif geom_b in cube_geoms:
                other_geom = geom_a
            else:
                continue
            for side, finger_geoms in finger_geoms_by_side.items():
                if other_geom in finger_geoms:
                    contacted_sides.add(side)

        assert contacted_sides == {"left", "right"}
        assert scene.get_gripper_close(task.right_gripper_name) < 0.96

        # Exercise zhaoxi's read-only segmentation and placement Routes on the
        # same real scene cut after bilateral MuJoCo contact. This deterministic
        # test segmenter uses the rendered depth plus known body geometry; it
        # validates Gateway/placement wiring, not learned-model quality.
        depth_frame = app.capture_depth_npy_frame(view_id="front")
        depth = np.load(io.BytesIO(depth_frame.data), allow_pickle=False)
        cloud = pointcloud_from_depth(
            depth,
            fovy_deg=float(depth_frame.intrinsics["fovy_deg"]),
            T_world_camera=xyzquat_to_pose_matrix(
                depth_frame.camera_pose_world_xyz_quat_xyzw
            ),
            depth_min_m=0.05,
            depth_max_m=5.0,
        )
        current_cube = np.asarray(
            scene.data.body(task.object_body_name).xpos,
            dtype=np.float64,
        )
        half_size = np.asarray(task.object_size_m, dtype=np.float64) * 0.5
        held_indices = np.flatnonzero(
            np.all(np.abs(cloud.points_world - current_cube) <= half_size + 0.003, axis=1)
        )
        destination_world = current_cube + np.asarray((0.12, 0.0, 0.0))
        table_indices = (
            (np.abs(cloud.points_world[:, 2] - task.table_top_z_m) <= 0.005)
            & (np.abs(cloud.points_world[:, 0] - destination_world[0]) <= 0.05)
            & (np.abs(cloud.points_world[:, 1] - destination_world[1]) <= 0.05)
        )
        destination_indices = np.flatnonzero(table_indices)
        assert held_indices.size >= 9
        assert destination_indices.size >= 9

        held_mask = np.zeros(depth.shape, dtype=bool)
        held_uv = cloud.uv[held_indices]
        held_mask[held_uv[:, 1], held_uv[:, 0]] = True
        destination_mask = np.zeros(depth.shape, dtype=bool)
        destination_uv = cloud.uv[destination_indices]
        destination_mask[destination_uv[:, 1], destination_uv[:, 0]] = True
        segmenter.masks = {
            "held object": held_mask,
            "destination surface": destination_mask,
        }

        placement_observation = session.observation_producer.capture_and_publish(
            session.service
        )
        assert placement_observation.accepted
        assert placement_observation.snapshot is not None
        snapshot = placement_observation.snapshot

        def run_perception_skill(
            *, skill_id: str, sequence: int, parameters: dict[str, object], timeout_ms: int
        ):
            request = _perception_request(
                session,
                snapshot,
                client_id=client_id,
                skill_id=skill_id,
                sequence=sequence,
                parameters=parameters,
                timeout_ms=timeout_ms,
            )
            admitted = session.service.submit_request(
                request=request,
                current_station_revision=snapshot.station_revision,
            )
            assert admitted.accepted and admitted.operation is not None
            assert session.service.launch_operation(
                admitted.operation.operation_id
            ).accepted
            executor = (
                session.segmentation_executor
                if skill_id == "perception.segment"
                else session.placement_executor
            )
            assert executor is not None
            assert executor.wait(
                admitted.operation.operation_id,
                timeout_sec=30.0,
            )
            completed = session.service.get_operation(
                admitted.operation.operation_id
            )
            assert completed is not None and completed.status == "completed"
            return completed

        held_segment = run_perception_skill(
            skill_id="perception.segment",
            sequence=2,
            parameters={
                "role": "main",
                "language_prompt": "held object",
                "target_pixel_uv": None,
                "min_score": 0.5,
            },
            timeout_ms=60_000,
        )
        destination_segment = run_perception_skill(
            skill_id="perception.segment",
            sequence=3,
            parameters={
                "role": "main",
                "language_prompt": "destination surface",
                "target_pixel_uv": None,
                "min_score": 0.5,
            },
            timeout_ms=60_000,
        )
        placement = run_perception_skill(
            skill_id="perception.plan_placement",
            sequence=4,
            parameters={
                "held_segment_operation_id": held_segment.operation_id,
                "destination_segment_operation_id": destination_segment.operation_id,
                "clearance_m": 0.005,
            },
            timeout_ms=120_000,
        )
        placement_result = dict(placement.execution_result or {})
        assert placement_result["planning_status"] == "planned"
        assert placement_result["geometry_eligible_for_motion_planning"] is True
        assert placement_result["held_gripper_state_checked"] is True
        assert placement_result["motion_or_hold_performed"] is False
    finally:
        if gripper_session is not None:
            gripper_session.close()
        session.close()

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real.grasp_observation import (
    build_grasp_observation_executor, read_retained_grasp_observation,
)
from agent_gateway.profiles.real.grasp_approach_plan import build_grasp_visible_scene_reader
from agent_gateway.profiles.real.point_grasp_gateway import (
    StationBoundPointGraspProvider, build_station_bound_point_grasp_executor,
)
from agent_gateway.profiles.real.realsense_perception import RealAlignedRgbdCapture
from agent_gateway.profiles.real.rm65_camera_geometry import RM65RgbCameraGeometry, _DynamicCameraMount
from agent_gateway.profiles.real.station_point_grasp_deployment import build_station_bound_point_grasp_registration
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from dataset_contracts.grasp_multiview_snapshot import decode_grasp_multiview_snapshot
from robot_motion.rm65.captured_robot_observations import RM65CapturedRobotObservationFilterResult
from deployments.perception.sam2_vgn_point_grasp import PointGraspCandidate, PointGraspResult

from test_agent_gateway_real_station_point_grasp import _provider_stack


def frozen_stack():
    source, revision, planner, base, provider = _provider_stack(backend_id="graspgenx")
    p = base.profile
    b = base.binding
    registration = build_station_bound_point_grasp_registration(
        **{name: getattr(p, name) for name in ("source_binding_hash", "frame_binding_hash", "frame_graph_hash",
            "source_frame_id", "source_frame_spec_hash", "reference_frame_id", "reference_frame_spec_hash",
            "backend_id", "model_profile_id", "model_profile_hash", "inference_implementation_hash")},
        **{name: getattr(b, name) for name in ("runtime_session_id", "control_domain_id", "target_binding_id",
            "station_manifest_hash", "execution_manifest_hash")},
        aligned_rgbd_source_catalog=tuple({"source_id": sid, "role": role, "native_width": 32, "native_height": 32}
            for sid, role in (("front", "main"), ("left_wrist", "wrist_left"), ("right_wrist", "wrist_right"))),
        frozen_observation=True,
    )
    return provider, registration


def test_frozen_registration_is_closed_and_read_only():
    _, registration = frozen_stack()
    assert registration.skill.definition.skill_version == "4"
    assert registration.binding.deployment_binding_version == "4"
    for registered in (registration, registration.observation_registration):
        skill = registered.skill
        route = skill.routes[0]
        assert route.control_effect_class == "read_only"
        assert route.skill_definition_hash == skill.definition.semantic_content_hash
        assert route.skill_io_contract_hash == skill.io_contract.semantic_content_hash
        assert route.parameter_schema_hash == skill.schema_entry.parameter_schema_hash
        assert route.execution_result_schema_hash == skill.schema_entry.execution_result_schema_hash
    parameters = {"role": "main", "target_pixel_uv": [16, 16], "max_candidates": 2,
                  "observation_operation_id": "observation:1", "view_policy": "auto"}
    registration.payload_validator.validate_parameters(parameters)
    with pytest.raises(ValueError, match="retained"):
        registration.payload_validator.validate_parameters({k: v for k, v in parameters.items() if k != "observation_operation_id"})


@pytest.mark.parametrize("anchor,missing", [("main", "wrist_right"), ("wrist_left", "wrist_right"), ("wrist_left", "main")])
def test_capture_inference_and_planning_reuse_exact_observation(tmp_path, anchor, missing):
    old_provider, registration = frozen_stack()
    captures = []
    filter_calls = []
    planner = old_provider.planner
    mask = np.zeros((32, 32), bool)
    mask[10:22, 10:22] = True
    planner.last_target_mask = mask
    planner.point_segmenter.segment_point = lambda *a, **kw: mask
    generated_scenes = []

    def capture(*, role, expected_station_revision, require_stationary):
        assert require_stationary
        captures.append(role)
        source_id = {"main": "front", "wrist_left": "left_wrist", "wrist_right": "right_wrist"}[role]
        if role == missing:
            raise RuntimeError("real_perception_frame_unavailable:"+source_id)
        pose = np.eye(4)
        if role == "wrist_left":
            pose[0, 3] = .002
        return RealAlignedRgbdCapture(
            source_id=source_id, role=role, acquired_at=9.95,
            station_revision=7, source_generation=2, frame_index=11, capture_group_id="real-aligned-rgbd:"+"d"*64,
            source_binding_hash=registration.profile.source_binding_hash, width=32, height=32,
            intrinsics={"width": 32, "height": 32, "fx": 100., "fy": 100., "cx": 16., "cy": 16., "fovy_deg": 18.18},
            source_from_camera=tuple(pose.reshape(-1)), rgb=np.full((32, 32, 3), 80, np.uint8),
            depth_m=np.full((32, 32), .5, np.float32), rgb_content_hash="a"*64, depth_content_hash="b"*64,
            capture_qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
            pose_measured_at=9.95, pose_available_at=9.952, pose_evidence_refs=("sample:1",))

    def generate(image, depth_m, *, pixel_uv, intrinsics, base_from_camera, max_candidates,
                 visible_scene_factory, use_visible_target_points):
        assert image.size == (32, 32)
        assert use_visible_target_points
        scene = visible_scene_factory(depth_m=depth_m, target_mask=mask, intrinsics=intrinsics, geometry=planner.solid_geometry)
        generated_scenes.append(scene)
        camera_pose = np.eye(4)
        camera_pose[2, 3] = .5
        ref_pose = base_from_camera @ camera_pose
        return PointGraspResult(
            target_point_camera_m=(0, 0, .5), target_point_base_m=tuple(ref_pose[:3, 3]),
            target_mask_bbox_xyxy=(10, 10, 22, 22),
            candidates=(PointGraspCandidate(score=.9, model_score=.95, click_distance_m=.01, gripper_width_m=.04,
                camera_from_tcp=tuple(camera_pose.reshape(-1)), base_from_tcp=tuple(ref_pose.reshape(-1)), visible_collision_free=True),))

    def capture_filter(**kwargs):
        filter_calls.append(kwargs)
        return RM65CapturedRobotObservationFilterResult(kwargs["depth_m"], np.zeros_like(mask), {})

    planner.plan = generate
    old_provider.perception_provider.capture_aligned_rgbd = capture
    store = RecoverableAgentArtifactStore(root=tmp_path/"artifacts", quota_bytes=64*1024*1024,
        max_artifact_bytes=16*1024*1024, max_chunk_bytes=1024*1024)
    observation_executor = build_grasp_observation_executor(
        registration=registration.observation_registration, perception_provider=old_provider.perception_provider,
        frame_binding=old_provider.frame_binding, artifact_store=store, runtime_session_id=registration.binding.runtime_session_id,
        current_station_revision=lambda: 7, max_retained_operations=4, clock=lambda: 10., non_motion_read_window=nullcontext)
    selected_roles = ["main", "wrist_left", "wrist_right"]
    observed = observation_executor._read(SimpleNamespace(parameters={"selected_roles": selected_roles}, expected_station_revision=7),
        SimpleNamespace(operation_id="observation:1"))
    validator = registration.observation_registration.payload_validator
    validator.validate_execution_result(observed.execution_result)
    validator.validate_execution_result_for_parameters({"selected_roles": selected_roles}, observed.execution_result)
    assert len(observed.execution_result["frames"]) == 2
    assert observed.execution_result["excluded_views"][0]["role"] == missing
    completion = SimpleNamespace(accepted=True, operation=SimpleNamespace(
        runtime_session_id=registration.binding.runtime_session_id, status="completed", execution_outcome="completed",
        execution_result=observed.execution_result, output_artifact_bindings=observed.output_artifact_bindings))
    observation_executor.completion_for = lambda _: completion

    def read_observation(identifier, revision):
        assert identifier == "observation:1"
        return read_retained_grasp_observation(executor=observation_executor, operation_id=identifier,
            artifact_store=store, runtime_session_id=registration.binding.runtime_session_id,
            expected_station_revision=revision, frame_binding_hash=old_provider.frame_binding.semantic_content_hash)

    provider = StationBoundPointGraspProvider(
        planner=planner, registration=registration, perception_provider=old_provider.perception_provider,
        frame_binding=old_provider.frame_binding, current_station_revision=lambda: 7,
        captured_robot_observation_filter=capture_filter, observation_reader=read_observation)
    grasp_executor = build_station_bound_point_grasp_executor(
        registration=registration, provider=provider, artifact_store=store, runtime_session_id=registration.binding.runtime_session_id,
        current_station_revision=lambda: 7, max_retained_operations=4, clock=lambda: 10.)
    parameters = {"role": anchor, "target_pixel_uv": (16, 16), "max_candidates": 2,
        "observation_operation_id": "observation:1", "view_policy": "auto"}
    result = grasp_executor._read(SimpleNamespace(parameters=parameters, expected_station_revision=7,
        input_artifact_bindings=(), semantic_content_hash="c"*64), SimpleNamespace(operation_id="grasp:1"))
    registration.payload_validator.validate_execution_result(result.execution_result)
    registration.payload_validator.validate_execution_result_for_parameters(parameters, result.execution_result)
    assert captures == selected_roles  # No second camera read during inference.
    assert len(filter_calls) == 2
    assert len(generated_scenes) == 1
    reader = build_grasp_visible_scene_reader(artifact_store=store, frame_graph=provider.frame_binding.frame_graph,
        execution_frame_id=provider.frame_binding.reference_frame_id)
    scene = reader(SimpleNamespace(output_artifact_bindings=result.output_artifact_bindings), result.execution_result)
    pose = np.asarray(result.execution_result["T_reference_from_camera"]).reshape(4, 4)
    np.testing.assert_allclose(scene.points, generated_scenes[0].points @ pose[:3, :3].T + pose[:3, 3])
    np.testing.assert_array_equal(scene.target_mask, generated_scenes[0].target_mask)
    with pytest.raises(ValueError, match="revision"):
        read_observation("observation:1", 8)


def test_stationary_camera_history_rejects_movement_and_matches_capture():
    provider, _ = frozen_stack()
    source = provider.perception_provider.source_binding.camera_sources["front"]
    binding = replace(provider.perception_provider.source_binding,
        camera_sources={"front": source, "left_wrist": source})
    geometry = RM65RgbCameraGeometry(source_binding=binding,
        target_builder=SimpleNamespace(solver_world_targets_for_qpos_pair=lambda **_: {"left": np.eye(4), "right": np.eye(4)}),
        static_source_from_camera={"front": np.eye(4)},
        dynamic_mounts=(_DynamicCameraMount("left_wrist", "wrist_left", "left", "link6", "camera", tuple(np.eye(4).reshape(-1))),),
        urdf_sha256="test")
    with pytest.raises(RuntimeError, match="history_unavailable"):
        geometry.stationary_geometry_cut(source_id="left_wrist", frame_acquired_at=10., now=10.01)
    for measured in (9.8, 9.85, 9.9, 9.95, 10.):
        geometry.publish_measured_qpos(qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
            measured_at=measured, available_at=measured+.001, evidence_refs=("sample:test",))
    cut, qpos, measured, _, _ = geometry.stationary_geometry_cut(source_id="left_wrist", frame_acquired_at=9.95, now=10.01)
    assert measured == cut.measured_at == 9.95
    assert not qpos["left"].flags.writeable
    geometry.publish_measured_qpos(qpos_by_side={"left": np.zeros(6), "right": np.full(6, .01)},
        measured_at=10.02, available_at=10.021, evidence_refs=("sample:moved",))
    with pytest.raises(RuntimeError, match="not_stationary"):
        geometry.stationary_geometry_cut(source_id="left_wrist", frame_acquired_at=10., now=10.03)


@pytest.mark.parametrize("failure", [None, "grasp_observation_not_stationary", "grasp_observation_pose_history_unavailable"])
def test_real_rgbd_adapter_requires_the_capture_stationary_cut(failure):
    from test_agent_gateway_real_realsense_perception import _provider, _RecordingRgbCameraGeometry

    geometry = _RecordingRgbCameraGeometry()
    stationary_calls = []
    measured_qpos = {"left": np.full(6, .1), "right": np.full(6, .2)}

    def stationary_cut(*, source_id, frame_acquired_at, now):
        stationary_calls.append((source_id, frame_acquired_at, now))
        if failure:
            raise RuntimeError(failure)
        cut = geometry.pose_cut(source_id=source_id, frame_acquired_at=frame_acquired_at, now=now)
        geometry.pose_cut_calls.clear()
        return cut, measured_qpos, 9.96, 9.961, ("sample:captured",)

    geometry.stationary_geometry_cut = stationary_cut
    observed_times = iter((10., 10.2))
    provider, sources, _, _ = _provider(wrist_depth=True, rgb_camera_geometry=geometry,
                                      clock=lambda: next(observed_times))
    if failure:
        with pytest.raises(RuntimeError, match=failure):
            provider.capture_aligned_rgbd(role="wrist_left", expected_station_revision=7, require_stationary=True)
    else:
        capture = provider.capture_aligned_rgbd(role="wrist_left", expected_station_revision=7, require_stationary=True)
        assert capture.capture_qpos_by_side is measured_qpos
        assert capture.pose_measured_at == 9.96
        assert capture.pose_available_at == 9.961
        assert capture.pose_evidence_refs == ("sample:captured",)
    assert stationary_calls == [("left_wrist", 9.97, 10.2)]
    assert sources["left_wrist"].latest_frame_reads == 1
    assert geometry.pose_cut_calls == []  # Never fall back to a current wrist pose.

from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real import point_grasp_render_gateway, realsense_perception
from agent_gateway.profiles.real.perception_application import (
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.profiles.real.perception_deployment import (
    RealPerceptionCaptureProfile,
)
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.point_grasp_gateway import (
    StationBoundPointGraspProvider,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
    RealSenseAgentPerceptionProvider,
)
from agent_gateway.profiles.real.rm65_grasp_reachability import (
    RM65PointGraspReachabilityAdapter,
)
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
    StationPointGraspArmReachabilityProfile,
    build_station_bound_point_grasp_registration,
)
from agent_gateway.profiles.real.station_point_grasp_render_deployment import (
    REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1,
    build_station_bound_point_grasp_render_registration,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,
    LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,
)
from deployments.perception.sam2_graspnet_point_grasp import (
    _normalize_graspnet_rigid_matrix,
)
from deployments.perception.sam2_vgn_point_grasp import (
    PointGraspCandidate,
    PointGraspResult,
)
from teleop_runtime.video_frame import VideoFrame


class _Source:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            camera_key="front",
            width=4,
            height=3,
            depth_width=4,
            depth_height=3,
            align_depth_to_color=True,
            enable_depth=True,
            color_space="rgb",
            serial="front-serial",
        )
        self.read_count = 0
        self.frame = VideoFrame(
            image=np.full((3, 4, 3), 32, dtype=np.uint8),
            source_timestamp=9.95,
            receive_timestamp=9.951,
            available_timestamp=9.952,
            frame_index=11,
            metadata={
                "camera_key": "front",
                "serial": "front-serial",
                "source_generation": 2,
            },
            depth_image=np.full((3, 4), 500, dtype=np.uint16),
            depth_metadata={
                "aligned_to": "color",
                "depth_scale_m_per_unit": 0.001,
            },
        )

    def get_latest_frame(self):
        self.read_count += 1
        return self.frame

    def device_info(self):
        return {
            "running": True,
            "lifecycle_state": "running",
            "worker_alive": True,
            "serial_number": "front-serial",
            "generation": 2,
            "intrinsics": {
                "width": 4,
                "height": 3,
                "fx": 4.0,
                "fy": 3.0,
                "ppx": 2.0,
                "ppy": 1.5,
            },
        }

    def close(self) -> None:
        return None


class _Planner:
    backend_id = "vgn"

    def __init__(self, point_segmenter: object, revision=None, events=None) -> None:
        self.point_segmenter = point_segmenter
        self.model_spec = SimpleNamespace(
            model_profile_id="vgn:test",
            semantic_content_hash="vgn-model:test",
        )
        self.semantic_content_hash = "station-point-grasp-planner:test"
        self.revision = revision
        self.events = events
        self.starts = 0
        self.closes = 0
        self.base_from_camera = None

    def start(self) -> None:
        self.starts += 1
        if self.events is not None:
            self.events.append("planner.start")

    def close(self) -> None:
        self.closes += 1
        if self.events is not None:
            self.events.append("planner.close")

    def plan(
        self,
        image,
        depth_m,
        *,
        pixel_uv,
        intrinsics,
        base_from_camera,
        max_candidates,
    ) -> PointGraspResult:
        assert image.size == (4, 3)
        assert depth_m.shape == (3, 4)
        assert pixel_uv == (1, 1)
        assert intrinsics["width"] == 4
        assert max_candidates == 2
        self.base_from_camera = np.asarray(base_from_camera).copy()
        camera_from_tcp = np.eye(4)
        camera_from_tcp[:3, 3] = (0.0, 0.0, 0.5)
        reference_from_tcp = self.base_from_camera @ camera_from_tcp
        target_camera = np.asarray((0.0, 0.0, 0.5, 1.0))
        target_reference = self.base_from_camera @ target_camera
        if self.revision is not None:
            self.revision["value"] += 1
        return PointGraspResult(
            target_point_camera_m=tuple(target_camera[:3]),
            target_point_base_m=tuple(target_reference[:3]),
            target_mask_bbox_xyxy=(0, 0, 4, 3),
            candidates=(
                PointGraspCandidate(
                    score=0.9,
                    model_score=0.95,
                    click_distance_m=0.01,
                    gripper_width_m=0.04,
                    camera_from_tcp=tuple(camera_from_tcp.reshape(-1)),
                    base_from_tcp=tuple(reference_from_tcp.reshape(-1)),
                    visible_collision_free=True,
                ),
            ),
        )


class _Localizer:
    def __init__(self, events=None) -> None:
        self.model_spec = SimpleNamespace(
            model_profile_id="sam2:test",
            semantic_content_hash="sam2-model:test",
        )
        self.semantic_content_hash = "sam2-localizer:test"
        self.events = events

    def start(self) -> None:
        if self.events is not None:
            self.events.append("localizer.start")

    def close(self) -> None:
        if self.events is not None:
            self.events.append("localizer.close")

    def localize(self, *_args, **_kwargs):
        return ()

    def segment_point(self, image, *, pixel_uv):
        mask = np.ones((image.height, image.width), dtype=bool)
        assert mask[pixel_uv[1], pixel_uv[0]]
        return mask


def _frame_binding():
    station_from_source = np.asarray(
        [
            [-1.0, 0.0, 0.0, 0.4],
            [0.0, -1.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    return build_real_perception_station_frame_binding(
        collision_scene_profile="station-point-grasp-test",
        scene_mount_hash="station-point-grasp-mounts:test",
        scene_world_from_canonical_station=np.linalg.inv(station_from_source),
    )


def _provider_stack(*, revision=None, arm_reachability_evaluator=None):
    source = _Source()
    binding = RealPerceptionSourceBinding(
        camera_sources={"front": source},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:station-point-grasp-test",
    )
    revision = {"value": 7} if revision is None else revision
    perception = RealSenseAgentPerceptionProvider(
        source_binding=binding,
        current_station_revision=lambda: revision["value"],
        clock=lambda: 10.0,
    )
    perception.bind_profile(
        RealPerceptionCaptureProfile(
            rgb_sources=(("front", "main", 4, 3),),
            depth_sources=(("front", "main", 4, 3),),
            source_binding_hash=binding.semantic_content_hash,
        )
    )
    frame_binding = _frame_binding()
    localizer = _Localizer()
    planner = _Planner(localizer, revision=revision if revision.get("mutate") else None)
    reachability_profile = (
        None
        if arm_reachability_evaluator is None
        else StationPointGraspArmReachabilityProfile(
            arm_side="right",
            evaluator_profile_id="rm65.reachability:test",
            evaluator_profile_hash="rm65-reachability-profile:test",
            solver_reference_frame_id=frame_binding.source_frame_id,
            solver_reference_frame_spec_hash=frame_binding.source_frame_spec_hash,
            frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        )
    )
    registration = build_station_bound_point_grasp_registration(
        aligned_rgbd_source_catalog=tuple(
            dict(item) for item in perception.aligned_rgbd_source_catalog()
        ),
        source_binding_hash=binding.semantic_content_hash,
        frame_binding_hash=frame_binding.semantic_content_hash,
        frame_graph_hash=frame_binding.frame_graph.semantic_content_hash,
        source_frame_id=frame_binding.source_frame_id,
        source_frame_spec_hash=frame_binding.source_frame_spec_hash,
        reference_frame_id=frame_binding.reference_frame_id,
        reference_frame_spec_hash=frame_binding.reference_frame_spec_hash,
        backend_id=planner.backend_id,
        model_profile_id=planner.model_spec.model_profile_id,
        model_profile_hash=planner.model_spec.semantic_content_hash,
        inference_implementation_hash=planner.semantic_content_hash,
        runtime_session_id="station-point-grasp:test",
        control_domain_id="real-perception-only:test",
        target_binding_id="real-realsense-only:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
        arm_reachability_profile=reachability_profile,
    )
    provider = StationBoundPointGraspProvider(
        planner=planner,
        registration=registration,
        perception_provider=perception,
        frame_binding=frame_binding,
        current_station_revision=lambda: revision["value"],
        arm_reachability_evaluator=arm_reachability_evaluator,
    )
    return source, revision, planner, registration, provider


def _request(*, include_arm_reachability=None):
    parameters = {
        "role": "main",
        # Wire decoding freezes JSON arrays before provider validation.
        "target_pixel_uv": (1, 1),
        "max_candidates": 2,
    }
    if include_arm_reachability is not None:
        parameters["include_arm_reachability"] = include_arm_reachability
    return SimpleNamespace(
        parameters=parameters,
        input_artifact_bindings=(),
        expected_station_revision=7,
    )


def test_station_point_grasp_route_is_live_bound_read_only_and_proposal_only() -> None:
    source, _revision, planner, registration, provider = _provider_stack()
    route = registration.skill.routes[0]
    assert route.skill_version == "2"
    assert route.supported_modes == ("real_observe",)
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert route.may_move_robot is False
    assert route.requires_operator_confirmation is False
    assert registration.skill.io_contract.input_artifact_slots == ()
    assert registration.skill.io_contract.output_artifact_slots[0].slot_id == (
        REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2
    )

    prepared = provider.plan_request(_request())

    assert source.read_count == 1
    assert planner.base_from_camera is not None
    assert prepared.execution_result["reference_frame_id"] == "canonical_station"
    assert prepared.execution_result["station_revision"] == 7
    assert prepared.execution_result["source_generation"] == 2
    assert prepared.execution_result["frame_index"] == 11
    assert prepared.execution_result["proposal_semantics"] == (
        "ranked_unprivileged_motion_proposals_only"
    )
    assert prepared.execution_result["downstream_control_effects"] == "forbidden"
    assert "selected_ordinal" not in prepared.execution_result
    registration.payload_validator.validate_parameters(_request().parameters)
    with pytest.raises(ValueError, match="parameter fields changed"):
        registration.payload_validator.validate_parameters(
            _request().parameters
            | {"T_reference_from_camera": list(np.eye(4).reshape(-1))}
        )
    registration.payload_validator.validate_execution_result(prepared.execution_result)
    registration.payload_validator.validate_execution_result_for_parameters(
        _request().parameters,
        prepared.execution_result,
    )


def test_station_point_grasp_profile_accepts_graspgenx_backend() -> None:
    _source, _revision, _planner, registration, _provider = _provider_stack()

    profile = replace(registration.profile, backend_id="graspgenx")

    assert profile.backend_id == "graspgenx"


def test_station_point_grasp_v3_reachability_is_request_gated_and_annotation_only() -> (
    None
):
    calls = []

    def evaluate(candidates):
        calls.append(candidates)
        profile = registration.profile.arm_reachability_profile
        assert profile is not None
        context = {
            "arm_side": profile.arm_side,
            "evaluator_profile_id": profile.evaluator_profile_id,
            "evaluator_profile_hash": profile.evaluator_profile_hash,
            "candidate_reference_frame_id": registration.profile.reference_frame_id,
            "solver_reference_frame_id": profile.solver_reference_frame_id,
            "solver_reference_frame_spec_hash": (
                profile.solver_reference_frame_spec_hash
            ),
            "frame_graph_hash": profile.frame_graph_hash,
            "time_budget_ms": profile.time_budget_ms,
            "max_solve_cycles": profile.max_solve_cycles,
            "max_measured_qpos_age_ms": profile.max_measured_qpos_age_ms,
            "standard_position_tolerance_m": (profile.standard_position_tolerance_m),
            "standard_orientation_tolerance_deg": (
                profile.standard_orientation_tolerance_deg
            ),
            "basic_position_tolerance_m": profile.basic_position_tolerance_m,
            "basic_orientation_tolerance_deg": (
                profile.basic_orientation_tolerance_deg
            ),
            "measured_qpos_revision": 4,
            "measured_qpos_at": 9.95,
            "measured_qpos_age_ms": 50.0,
            "evaluation_scope": ("finite_time_endpoint_ik_not_path_or_grasp_success"),
            "candidate_order_effect": "annotation_only_order_unchanged",
        }
        annotation = {
            "classification": "standard_reachable",
            "position_residual_m": 0.001,
            "orientation_residual_deg": 0.5,
            "solve_cycles": 2,
            "elapsed_ms": 4.0,
            "termination_reason": "standard_tolerance_met",
        }
        return context, tuple(annotation for _candidate in candidates)

    _source, _revision, _planner, registration, provider = _provider_stack(
        arm_reachability_evaluator=evaluate
    )
    assert registration.skill.definition.skill_version == "3"

    disabled_request = _request(include_arm_reachability=False)
    disabled = provider.plan_request(disabled_request).execution_result
    assert calls == []
    assert "arm_reachability_context" not in disabled
    assert "arm_reachability" not in disabled["candidates"][0]
    registration.payload_validator.validate_execution_result(disabled)
    registration.payload_validator.validate_execution_result_for_parameters(
        disabled_request.parameters, disabled
    )

    enabled_request = _request(include_arm_reachability=True)
    enabled = provider.plan_request(enabled_request).execution_result
    assert len(calls) == 1
    np.testing.assert_allclose(
        calls[0][0],
        np.asarray(enabled["candidates"][0]["T_reference_from_tool_tcp"]).reshape(4, 4),
    )
    assert enabled["candidates"][0]["arm_reachability"]["classification"] == (
        "standard_reachable"
    )
    registration.payload_validator.validate_parameters(enabled_request.parameters)
    registration.payload_validator.validate_execution_result(enabled)
    registration.payload_validator.validate_execution_result_for_parameters(
        enabled_request.parameters, enabled
    )

    seed_exhausted = copy.deepcopy(enabled)
    seed_exhausted_annotation = seed_exhausted["candidates"][0][
        "arm_reachability"
    ]
    seed_exhausted_annotation.update(
        {
            "classification": "difficult_to_reach",
            "position_residual_m": 0.06,
            "orientation_residual_deg": 25.0,
            "solve_cycles": (
                registration.profile.arm_reachability_profile.max_solve_cycles
            ),
            "termination_reason": "seed_budget_exhausted",
        }
    )
    registration.payload_validator.validate_execution_result(seed_exhausted)

    with pytest.raises(ValueError, match="does not match the request"):
        registration.payload_validator.validate_execution_result_for_parameters(
            disabled_request.parameters, enabled
        )


def test_rm65_reachability_adapter_accepts_normalized_graspnet_candidate() -> None:
    binding = _frame_binding()
    raw_pose = np.asarray(
        [
            [
                -0.17259738000158176,
                0.18359564713816373,
                0.9677307159760798,
                -0.13079618433586204,
            ],
            [
                0.04199108059770058,
                -0.9802105728615037,
                0.19345253616022093,
                -0.07363994251000638,
            ],
            [
                0.9840969607643857,
                0.07402546072335396,
                0.16147240422405731,
                0.1374787059167376,
            ],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = _normalize_graspnet_rigid_matrix(
        raw_pose,
        field_name="GraspNet camera_from_grasp",
    )

    class _MotionPort:
        @staticmethod
        def latest_measured_qpos():
            return SimpleNamespace(
                qpos_by_side={"left": np.zeros(6), "right": np.zeros(6)},
                revision=4,
                measured_at=9.95,
            )

    class _ReachabilityPlanner:
        @staticmethod
        def assess_endpoint_ik(**kwargs):
            np.testing.assert_allclose(
                kwargs["target_tool_tcp_world"][:3, :3].T
                @ kwargs["target_tool_tcp_world"][:3, :3],
                np.eye(3),
                atol=1e-12,
                rtol=0.0,
            )
            return SimpleNamespace(
                to_dict=lambda: {
                    "classification": "standard_reachable",
                    "position_residual_m": 0.001,
                    "orientation_residual_deg": 0.5,
                    "solve_cycles": 1,
                    "elapsed_ms": 1.0,
                    "termination_reason": "standard_tolerance_met",
                }
            )

    profile = SimpleNamespace(
        solver_reference_frame_id=binding.source_frame_id,
        solver_reference_frame_spec_hash=binding.source_frame_spec_hash,
        frame_graph_hash=binding.frame_graph.semantic_content_hash,
        arm_side="right",
        evaluator_profile_id="rm65.reachability:test",
        evaluator_profile_hash="rm65-reachability-profile:test",
        time_budget_ms=20.0,
        max_solve_cycles=2,
        max_measured_qpos_age_ms=200.0,
        standard_position_tolerance_m=0.005,
        standard_orientation_tolerance_deg=2.0,
        basic_position_tolerance_m=0.010,
        basic_orientation_tolerance_deg=5.0,
    )
    adapter = object.__new__(RM65PointGraspReachabilityAdapter)
    adapter.profile = profile
    adapter.planner = _ReachabilityPlanner()
    adapter.motion_port = _MotionPort()
    adapter.frame_graph = binding.frame_graph
    adapter.candidate_reference_frame_id = binding.reference_frame_id
    adapter.clock = lambda: 10.0
    adapter.monotonic = lambda: 1.0

    with pytest.raises(ValueError, match="rotation must be orthonormal"):
        adapter((raw_pose,))
    context, annotations = adapter((normalized,))

    assert context["candidate_order_effect"] == "annotation_only_order_unchanged"
    assert annotations[0]["classification"] == "standard_reachable"


def test_station_point_grasp_fails_if_revision_changes_during_inference() -> None:
    revision = {"value": 7, "mutate": True}
    _source, _revision, _planner, _registration, provider = _provider_stack(
        revision=revision
    )

    with pytest.raises(RuntimeError, match="during point-grasp model inference"):
        provider.plan_request(_request())


def test_debug_target_coordinate_overlay_projects_depth_target_in_station_frame() -> (
    None
):
    image = Image.new("RGB", (200, 160), (10, 10, 10))
    rendered = point_grasp_render_gateway._render_target_coordinate_debug_overlay(
        image,
        target_point_reference_m=(0.0, 0.0, 1.0),
        reference_from_camera=np.eye(4),
        intrinsics={
            "width": 200,
            "height": 160,
            "fx": 100.0,
            "fy": 100.0,
            "cx": 100.0,
            "cy": 80.0,
        },
    )

    assert rendered is not image
    assert image.getpixel((100, 74)) == (10, 10, 10)
    assert rendered.getpixel((100, 74)) == (0, 255, 255)
    assert rendered.getpixel((103, 80)) == (255, 64, 64)
    assert rendered.getpixel((100, 84)) == (64, 255, 64)


def test_debug_target_coordinate_profile_is_default_off_and_hash_bound() -> None:
    _source, _revision, _planner, point_grasp, _provider = _provider_stack()
    default = build_station_bound_point_grasp_render_registration(
        point_grasp_registration=point_grasp,
    )
    enabled = build_station_bound_point_grasp_render_registration(
        point_grasp_registration=point_grasp,
        debug_target_coordinate_overlay=True,
    )

    assert default.profile.debug_target_coordinate_overlay is False
    assert enabled.profile.debug_target_coordinate_overlay is True
    assert (
        default.profile.semantic_content_hash != enabled.profile.semantic_content_hash
    )


def test_point_grasp_is_default_off_and_lifecycle_order_is_reversible(
    monkeypatch,
    tmp_path,
) -> None:
    events = []
    source = _Source()
    binding = RealPerceptionSourceBinding(
        camera_sources={"front": source},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:composition-test",
    )
    client_bindings = compile_agent_gateway_unix_client_bindings(
        primary_client_id="rpent:test",
        primary_client_uid=1000,
    )
    default = assemble_camera_only_real_perception_composition(
        source_binding=binding,
        socket_path=tmp_path / "default.sock",
        artifact_root=tmp_path,
        client_bindings=client_bindings,
        runtime_session_id="real-perception:default-test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        clock=lambda: 10.0,
    )
    try:
        assert default.station_point_grasp_registration is None
        assert default.station_point_grasp_render_registration is None
        assert [
            entry.route.skill_id
            for entry in default.contract_bundle.route_manifest.skill_routes
        ] == ["perception.capture_depth", "perception.capture_rgb"]
    finally:
        default.application.close()

    monkeypatch.setattr(
        realsense_perception,
        "start_realsense_sources",
        lambda _sources: events.append("cameras.start"),
    )
    monkeypatch.setattr(
        realsense_perception,
        "admit_realsense_sources",
        lambda _sources: events.append("cameras.admit"),
    )
    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        lambda _sources: events.append("cameras.close"),
    )
    localizer = _Localizer(events)
    planner = _Planner(localizer, events=events)
    composition = assemble_camera_only_real_perception_composition(
        source_binding=binding,
        socket_path=tmp_path / "enabled.sock",
        artifact_root=tmp_path,
        client_bindings=client_bindings,
        runtime_session_id="real-perception:enabled-test",
        control_domain_id="real-perception-only:test",
        frame_binding=_frame_binding(),
        object_localizer=localizer,
        point_grasp_planner=planner,
        clock=lambda: 10.0,
    )

    class _Server:
        def serve_forever(self, poll_interval=0.5):
            return None

        def shutdown(self):
            return None

        def server_close(self):
            events.append("gateway.close")
            composition.session.close()

    composition.application.server_factory = lambda: (
        events.append("socket.create") or _Server()
    )
    composition.application.start()
    assert events[:5] == [
        "cameras.start",
        "cameras.admit",
        "localizer.start",
        "planner.start",
        "socket.create",
    ]
    adapter = composition.session.wire_adapter(authenticated_client_id="rpent:test")
    observation = adapter.get_observation_snapshot()
    manifest = composition.contract_bundle.route_manifest
    entry = manifest.route("perception.plan_point_grasp", "2")
    route = entry.route
    profile = route.allowed_parameter_profiles[0]
    submitted = adapter.submit_read_only_skill_request(
        AgentSkillRequest(
            client_id="rpent:test",
            request_id="request:station-point-grasp:test",
            idempotency_key="idempotency:station-point-grasp:test",
            client_sequence=1,
            runtime_session_id=manifest.runtime_session_id,
            skill_route_manifest_hash=manifest.semantic_content_hash,
            expected_station_revision=int(observation["station_revision"]),
            skill_id=route.skill_id,
            skill_version=route.skill_version,
            skill_definition_hash=route.skill_definition_hash,
            route_id=route.route_id,
            route_version=route.route_version,
            route_hash=route.semantic_content_hash,
            deployment_binding_hash=(entry.deployment_binding.deployment_binding_hash),
            parameter_profile_id=profile.parameter_profile_id,
            parameter_profile_hash=profile.parameter_profile_hash,
            parameters={
                "role": "main",
                "target_pixel_uv": [1, 1],
                "max_candidates": 2,
            },
            observation_snapshot_id=str(observation["snapshot_id"]),
            requested_mode="real_observe",
            requested_timeout_ms=100_000,
        ).to_wire()
    )
    assert submitted["accepted"] is True
    operation_id = submitted["operation"]["operation_id"]
    assert composition.session.station_point_grasp_executor is not None
    assert composition.session.station_point_grasp_executor.wait(
        operation_id,
        timeout_sec=2.0,
    )
    operation = composition.session.service.operation_store.get(operation_id)
    assert operation is not None
    assert operation.status == "completed"
    assert operation.execution_result["proposal_semantics"] == (
        "ranked_unprivileged_motion_proposals_only"
    )
    assert operation.output_artifact_bindings[0].slot_id == (
        REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2
    )
    render_entry = manifest.route("perception.render_point_grasp_candidate", "1")
    render_route = render_entry.route
    assert render_route.control_effect_class == "read_only"
    assert render_route.privilege_tier == "t0_read_only"
    assert render_route.cancellation_policy == "cancel_only"
    assert render_route.supported_modes == ("real_observe",)
    assert render_route.may_move_robot is False
    assert render_route.requires_operator_confirmation is False
    assert render_route.requires_deadman is False
    render_slot = composition.station_point_grasp_render_registration
    assert render_slot is not None
    output_slot = render_slot.skill.io_contract.output_artifact_slots[0]
    assert output_slot.slot_id == REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1
    assert output_slot.allowed_schema_ids == (POINT_GRASP_ANNOTATED_SCHEMA_ID_V1,)
    assert output_slot.allowed_media_types == (
        LOCALIZATION_ANNOTATED_IMAGE_MEDIA_TYPE_V1,
    )
    assert output_slot.allowed_encodings == (LOCALIZATION_ANNOTATED_IMAGE_ENCODING_V1,)
    assert output_slot.allowed_compressions == (
        LOCALIZATION_ANNOTATED_IMAGE_COMPRESSION_V1,
    )
    render_profile = render_route.allowed_parameter_profiles[0]

    def submit_render(parameters, *, sequence, suffix, station_observation=None):
        selected_observation = (
            observation if station_observation is None else station_observation
        )
        return adapter.submit_read_only_skill_request(
            AgentSkillRequest(
                client_id="rpent:test",
                request_id=f"request:station-point-grasp-render:{suffix}",
                idempotency_key=f"idempotency:station-point-grasp-render:{suffix}",
                client_sequence=sequence,
                runtime_session_id=manifest.runtime_session_id,
                skill_route_manifest_hash=manifest.semantic_content_hash,
                expected_station_revision=int(selected_observation["station_revision"]),
                skill_id=render_route.skill_id,
                skill_version=render_route.skill_version,
                skill_definition_hash=render_route.skill_definition_hash,
                route_id=render_route.route_id,
                route_version=render_route.route_version,
                route_hash=render_route.semantic_content_hash,
                deployment_binding_hash=(
                    render_entry.deployment_binding.deployment_binding_hash
                ),
                parameter_profile_id=render_profile.parameter_profile_id,
                parameter_profile_hash=render_profile.parameter_profile_hash,
                parameters=parameters,
                observation_snapshot_id=str(selected_observation["snapshot_id"]),
                requested_mode="real_observe",
                requested_timeout_ms=20_000,
            ).to_wire()
        )

    render_parameters = {
        "point_grasp_operation_id": operation_id,
        "candidate_ordinal": 0,
        "view_role": "main",
    }
    rendered = submit_render(render_parameters, sequence=2, suffix="success")
    assert rendered["accepted"] is True
    render_operation_id = rendered["operation"]["operation_id"]
    render_executor = composition.session.station_point_grasp_render_executor
    assert render_executor is not None
    assert render_executor.wait(render_operation_id, timeout_sec=2.0)
    render_operation = composition.session.service.operation_store.get(
        render_operation_id
    )
    assert render_operation is not None
    assert render_operation.status == "completed"
    result = render_operation.execution_result
    assert result is not None
    assert result["point_grasp_operation_id"] == operation_id
    assert result["point_grasp_request_hash"] == operation.request_hash
    assert (
        result["source_capture_group_id"]
        == operation.execution_result["capture_group_id"]
    )
    assert result["candidate_ordinal"] == 0
    assert result["view_role"] == "main"
    assert result["view_source_id"] == "front"
    assert result["view_station_revision"] == result["source_station_revision"]
    assert result["depth_occlusion_tested"] is False
    assert result["downstream_control_effects"] == "forbidden"
    reference_from_tcp = np.asarray(result["T_reference_from_tool_tcp"]).reshape(4, 4)
    reference_from_camera = np.asarray(result["T_reference_from_camera"]).reshape(4, 4)
    camera_from_tcp = np.asarray(result["T_camera_from_tool_tcp"]).reshape(4, 4)
    np.testing.assert_allclose(
        np.linalg.inv(reference_from_camera) @ reference_from_tcp,
        camera_from_tcp,
    )
    assert (
        result["frame_binding_hash"] == composition.frame_binding.semantic_content_hash
    )
    assert result["frame_graph_hash"] == (
        composition.frame_binding.frame_graph.semantic_content_hash
    )
    assert result["source_frame_id"] == composition.frame_binding.source_frame_id
    assert result["reference_frame_id"] == (
        composition.frame_binding.reference_frame_id
    )
    assert len(result["frames"]) == 1
    frame = result["frames"][0]
    assert frame == {
        "ordinal": 0,
        "role": "main",
        "width": 4,
        "height": 3,
        "media_type": "image/jpeg",
        "media_content_hash": render_operation.output_artifact_bindings[
            0
        ].artifact.sha256,
    }
    artifact_binding = render_operation.output_artifact_bindings[0]
    assert artifact_binding.slot_id == REAL_STATION_POINT_GRASP_RENDER_OUTPUT_SLOT_ID_V1
    assert artifact_binding.ordinal == 0
    assert artifact_binding.artifact.schema_id == POINT_GRASP_ANNOTATED_SCHEMA_ID_V1
    assert artifact_binding.artifact.media_type == "image/jpeg"
    assert artifact_binding.artifact.encoding == "jpeg.rgb8.v1"
    assert artifact_binding.artifact.compression == "none"
    assert source.read_count == 2  # proposal generation and rendered view
    inconsistent = dict(result)
    inconsistent_camera_from_tcp = list(result["T_camera_from_tool_tcp"])
    inconsistent_camera_from_tcp[3] += 0.01
    inconsistent["T_camera_from_tool_tcp"] = inconsistent_camera_from_tcp
    with pytest.raises(ValueError, match="transform chain is inconsistent"):
        render_slot.payload_validator.validate_execution_result(inconsistent)

    unknown = submit_render(
        render_parameters | {"point_grasp_operation_id": "operation:unknown"},
        sequence=3,
        suffix="unknown",
    )
    assert unknown["accepted"] is True
    unknown_id = unknown["operation"]["operation_id"]
    assert render_executor.wait(unknown_id, timeout_sec=2.0)
    unknown_operation = composition.session.service.operation_store.get(unknown_id)
    assert unknown_operation is not None
    assert unknown_operation.status == "failed"
    assert unknown_operation.reason_code == "passive_read_executor_exception:ValueError"

    missing_ordinal = submit_render(
        render_parameters | {"candidate_ordinal": 1},
        sequence=4,
        suffix="missing-ordinal",
    )
    assert missing_ordinal["accepted"] is True
    missing_ordinal_id = missing_ordinal["operation"]["operation_id"]
    assert render_executor.wait(missing_ordinal_id, timeout_sec=2.0)
    missing_ordinal_operation = composition.session.service.operation_store.get(
        missing_ordinal_id
    )
    assert missing_ordinal_operation is not None
    assert missing_ordinal_operation.status == "failed"

    injected_matrix = submit_render(
        render_parameters | {"T_reference_from_tool_tcp": list(np.eye(4).reshape(-1))},
        sequence=5,
        suffix="injected-matrix",
    )
    assert injected_matrix["accepted"] is False
    assert injected_matrix["reason_code"] == "parameter_payload_rejected"

    unavailable_role = submit_render(
        render_parameters | {"view_role": "wrist_left"},
        sequence=6,
        suffix="unavailable-role",
    )
    assert unavailable_role["accepted"] is False
    assert unavailable_role["reason_code"] == "parameter_payload_rejected"

    composition.session.observation_producer.revision_tracker.observe(
        {"forced_test_control_fingerprint": "changed"}
    )
    changed_observation = adapter.get_observation_snapshot()
    assert changed_observation["station_revision"] > observation["station_revision"]
    stale_source = submit_render(
        render_parameters,
        sequence=7,
        suffix="stale-source",
        station_observation=changed_observation,
    )
    assert stale_source["accepted"] is True
    stale_source_id = stale_source["operation"]["operation_id"]
    assert render_executor.wait(stale_source_id, timeout_sec=2.0)
    stale_source_operation = composition.session.service.operation_store.get(
        stale_source_id
    )
    assert stale_source_operation is not None
    assert stale_source_operation.status == "failed"
    assert stale_source_operation.output_artifact_bindings == ()
    # Planning and the repeated render requests above share the one application
    # camera resource transaction; Skill calls must never restart it.
    assert events.count("cameras.start") == 1
    assert events.count("cameras.close") == 0
    composition.application.close()
    assert planner.starts == 1
    assert planner.closes == 1
    assert events.count("cameras.start") == 1
    assert events.count("cameras.close") == 1
    assert events[-4:] == [
        "gateway.close",
        "planner.close",
        "localizer.close",
        "cameras.close",
    ]

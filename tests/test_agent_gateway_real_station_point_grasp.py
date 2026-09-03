from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from agent_gateway.profiles.real import realsense_perception
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
from agent_gateway.profiles.real.station_point_grasp_deployment import (
    REAL_STATION_POINT_GRASP_OUTPUT_SLOT_ID_V2,
    build_station_bound_point_grasp_registration,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
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


def _provider_stack(*, revision=None):
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
    )
    provider = StationBoundPointGraspProvider(
        planner=planner,
        registration=registration,
        perception_provider=perception,
        frame_binding=frame_binding,
        current_station_revision=lambda: revision["value"],
    )
    return source, revision, planner, registration, provider


def _request():
    return SimpleNamespace(
        parameters={
            "role": "main",
            # Wire decoding freezes JSON arrays before provider validation.
            "target_pixel_uv": (1, 1),
            "max_candidates": 2,
        },
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


def test_station_point_grasp_fails_if_revision_changes_during_inference() -> None:
    revision = {"value": 7, "mutate": True}
    _source, _revision, _planner, _registration, provider = _provider_stack(
        revision=revision
    )

    with pytest.raises(RuntimeError, match="during point-grasp model inference"):
        provider.plan_request(_request())


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
    composition.application.close()
    assert planner.starts == 1
    assert planner.closes == 1
    assert events[-4:] == [
        "gateway.close",
        "planner.close",
        "localizer.close",
        "cameras.close",
    ]

"""CPU-only authenticated RGB-D/mask reuse tests without cameras, models or simulation."""

import hashlib
import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.back_projection import (
    CorrelatedPlacementCapture,
    LiveAlignedDepthCapture,
    LiveBackProjectionProvider,
    build_back_projection_registration,
)
from agent_gateway.profiles.mujoco.rgb_deployment import (
    build_mujoco_rgb_capture_registration_v2,
)
from agent_gateway.profiles.paired_rgbd import encode_paired_observation
from agent_gateway.profiles.rgbd_capture import build_rgbd_capture_registration
from agent_gateway.profiles.segmentation import (
    LiveRgbSegmentationCapture,
    build_segmentation_registration,
)
from agent_gateway.profiles.surface_anchor import _json_bytes
from agent_gateway.profiles.surface_anchor_v3 import (
    build_surface_anchor_executor_v3,
    build_surface_anchor_registration_v3,
)
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import (
    ArtifactStoreIntegrityError,
    RecoverableAgentArtifactStore,
)
from agent_gateway.skills.perception.surface_anchor import ANCHOR_FIELDS
from agent_gateway.skills.perception.surface_anchor_v2 import (
    EXACT_ALGORITHM,
    MAX_CLEARANCE_MEDOID_ALGORITHM,
    SELECTION_FIELDS,
    capture_schema_v2,
    validate_anchor_v2,
)
from agent_gateway.skills.perception.surface_anchor_v3 import PARAMETER_FIELDS
from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.perception_artifacts import (
    DEPTH_FRAME_COMPRESSION_V1,
    DEPTH_FRAME_ENCODING_V1,
    DEPTH_FRAME_MEDIA_TYPE_V1,
    DEPTH_FRAME_SCHEMA_ID_V1,
    PAIRED_RGBD_SCHEMA_ID_V1,
    RGB_FRAME_COMPRESSION_V1,
    RGB_FRAME_ENCODING_V1,
    RGB_FRAME_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
    SEGMENTATION_MASK_COMPRESSION_V1,
    SEGMENTATION_MASK_ENCODING_V1,
    SEGMENTATION_MASK_MEDIA_TYPE_V1,
    SEGMENTATION_MASK_SCHEMA_ID_V1,
    SEGMENTATION_OVERLAY_COMPRESSION_V1,
    SEGMENTATION_OVERLAY_ENCODING_V1,
    SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,
    SEGMENTATION_OVERLAY_SCHEMA_ID_V1,
    SURFACE_ANCHOR_SCHEMA_ID_V2,
)

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _lookup_retained(operations, key):
    operation, request = operations.get(key, (None, None))
    return None if operation is None or request is None else (request, operation)




def _registrations(policy="prompt_nearest", *, deployment_identity=None):
    source_catalog = (
        {"source_id": "front", "role": "main", "native_width": 8, "native_height": 7},
    )
    identity = {
        "runtime_session_id": "session:test",
        "control_domain_id": "domain:test",
        "target_binding_id": "target:test",
        "station_manifest_hash": "station:test",
        "execution_manifest_hash": "execution:test",
    }
    identity.update(deployment_identity or {})
    projection = build_back_projection_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_catalog=source_catalog,
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame:test",
        frame_graph_hash="graph:test",
        **identity,
    )
    rgb2 = build_mujoco_rgb_capture_registration_v2(
        source_catalog=source_catalog, **identity
    )
    rgb = build_rgbd_capture_registration(
        rgb_registration=rgb2, back_projection=projection
    )
    segmentation = build_segmentation_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_id_by_role={"main": "front"},
        model_profile_id="segment:test",
        model_profile_hash="model:test",
        inference_implementation_hash="implementation:test",
        skill_version="3",
        rgb_capture_route_hash=rgb.entry.route.semantic_content_hash,
        **identity,
    )
    anchor = build_surface_anchor_registration_v3(
        back_projection=projection,
        segmentation=segmentation,
        capture_rgb=rgb,
        selection_policy=policy,
    )
    return SimpleNamespace(
        projection=projection, rgb=rgb, segmentation=segmentation, anchor=anchor
    )


def _request(registration, parameters, bindings=(), *, name="anchor"):
    skill = registration.skill
    route = skill.routes[0]
    binding = (
        registration.entry.deployment_binding
        if hasattr(registration, "entry")
        else registration.binding.to_ref()
    )
    profile = skill.schema_entry.parameter_profiles[0]
    return AgentSkillRequest(
        client_id="client:test",
        request_id="request:" + name,
        idempotency_key="idempotency:" + name,
        client_sequence=0,
        runtime_session_id="session:test",
        skill_route_manifest_hash="manifest:test",
        expected_station_revision=4,
        skill_id=skill.definition.skill_id,
        skill_version=skill.definition.skill_version,
        skill_definition_hash=skill.definition.semantic_content_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters,
        observation_snapshot_id="observation:test",
        requested_mode="sim_active",
        requested_timeout_ms=10000,
        input_artifact_bindings=bindings,
    )


def _npy(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _jpeg(size=(8, 7), color="white"):
    stream = io.BytesIO()
    Image.new("RGB", size, color).save(stream, format="JPEG")
    return stream.getvalue()


def _operation(request, result, bindings, name):
    return SimpleNamespace(
        operation_id="operation:" + name,
        status="completed",
        execution_outcome="completed",
        execution_result=result,
        output_artifact_bindings=bindings,
        request_hash=request.semantic_content_hash,
        runtime_session_id=request.runtime_session_id,
    )


@pytest.fixture
def case(tmp_path):
    regs = _registrations()
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts", quota_bytes=1 << 20, max_artifact_bytes=1 << 20
    )
    sample = SimpleNamespace(
        **vars(regs),
        store=store,
        now=12.0,
        revision=4,
        cancelled=False,
        operations={},
        projection_calls=[],
    )

    def put(data, *, slot, ordinal=0):
        formats = {
            "rgb_frames": (
                RGB_FRAME_MEDIA_TYPE_V1,
                RGB_FRAME_SCHEMA_ID_V1,
                RGB_FRAME_ENCODING_V1,
                RGB_FRAME_COMPRESSION_V1,
            ),
            "depth_frames": (
                DEPTH_FRAME_MEDIA_TYPE_V1,
                DEPTH_FRAME_SCHEMA_ID_V1,
                DEPTH_FRAME_ENCODING_V1,
                DEPTH_FRAME_COMPRESSION_V1,
            ),
            "paired_observations": (
                "application/json",
                PAIRED_RGBD_SCHEMA_ID_V1,
                "utf-8",
                "none",
            ),
            "segmentation_mask": (
                SEGMENTATION_MASK_MEDIA_TYPE_V1,
                SEGMENTATION_MASK_SCHEMA_ID_V1,
                SEGMENTATION_MASK_ENCODING_V1,
                SEGMENTATION_MASK_COMPRESSION_V1,
            ),
            "segmentation_overlay": (
                SEGMENTATION_OVERLAY_MEDIA_TYPE_V1,
                SEGMENTATION_OVERLAY_SCHEMA_ID_V1,
                SEGMENTATION_OVERLAY_ENCODING_V1,
                SEGMENTATION_OVERLAY_COMPRESSION_V1,
            ),
        }
        media, schema, encoding, compression = formats[slot]
        ref = artifact_ref_for_bytes(
            data,
            media_type=media,
            schema_id=schema,
            encoding=encoding,
            compression=compression,
        )
        commit_artifact_bytes(
            store=store,
            data=data,
            artifact=ref,
            producer_id="test:fixture",
            runtime_session_id="session:test",
            idempotency_key=ref.sha256,
            now=10.0,
        )
        return AgentArtifactBinding(
            slot_id=slot,
            ordinal=ordinal,
            artifact=ref,
            production_receipt_ref="test:receipt",
        )

    sample.put = put
    rgb_data = _jpeg()
    sample.rgb_output = put(rgb_data, slot="rgb_frames")
    depth = np.full((7, 8), 0.5, dtype=np.float32)
    depth_npy = _npy(depth)
    sample.depth_output = put(depth_npy, slot="depth_frames")
    depth_hash = canonical_content_hash(
        {
            "dtype": "float32",
            "unit": "meter",
            "invalid_value": 0.0,
            "shape": list(depth.shape),
            "byte_sha256": hashlib.sha256(depth.tobytes()).hexdigest(),
        },
        domain="dexteleop.mujoco_back_projection.depth_content.v1",
    )
    with Image.open(io.BytesIO(rgb_data)) as image:
        rgb = LiveRgbSegmentationCapture(
            source_id="front",
            role="main",
            captured_at=10.0,
            station_revision=4,
            image=image,
            rgb_content_hash=sample.rgb_output.artifact.sha256,
        )
    aligned = LiveAlignedDepthCapture(
        source_id="front",
        role="main",
        captured_at=10.0,
        station_revision=4,
        capture_group_id="capture:test",
        depth_content_hash=depth_hash,
        intrinsics={
            "width": 8,
            "height": 7,
            "fx": 4.0,
            "fy": 4.0,
            "cx": 4.0,
            "cy": 3.0,
        },
        reference_from_camera=np.eye(4),
        depth_m=depth,
    )
    sample.cut = CorrelatedPlacementCapture(rgb, aligned, rgb_data, depth_npy)
    sample.capture_request = _request(
        regs.rgb,
        {
            "selected_roles": ["main"],
            "width": 8,
            "height": 7,
            "max_total_bytes": 65536,
            "include_depth": True,
        },
        name="capture",
    )
    paired = encode_paired_observation(
        sample.cut,
        rgb_artifact=sample.rgb_output.artifact,
        depth_artifact=sample.depth_output.artifact,
        reference_frame_id="canonical_station",
        reference_frame_spec_hash="frame:test",
        frame_graph_hash="graph:test",
        clock_domain_id="clock:test",
        source_operation_id="operation:capture",
        source_request_hash=sample.capture_request.semantic_content_hash,
        runtime_session_id="session:test",
    )
    sample.paired_output = put(paired, slot="paired_observations")
    sample.capture_result = {
        "snapshot_id": "rgb-snapshot:test",
        "station_revision": 4,
        "capture_window_started_at": 10.0,
        "capture_window_ended_at": 10.0,
        "total_bytes": len(rgb_data) + len(depth_npy) + len(paired),
        "include_depth": True,
        "frames": [
            {
                "ordinal": 0,
                "source_id": "front",
                "role": "main",
                "acquired_at": 10.0,
                "media_type": "image/jpeg",
                "width": 8,
                "height": 7,
                "media_content_hash": sample.rgb_output.artifact.sha256,
            }
        ],
        "paired_observations": [
            {
                "role": "main",
                "source_id": "front",
                "rgb_frame_ordinal": 0,
                "depth_artifact_ordinal": 0,
                "observation_artifact_ordinal": 0,
                "observation_sha256": sample.paired_output.artifact.sha256,
            }
        ],
    }
    sample.capture = _operation(
        sample.capture_request,
        sample.capture_result,
        (sample.depth_output, sample.paired_output, sample.rgb_output),
        "capture",
    )
    sample.operations["operation:capture"] = (sample.capture, sample.capture_request)
    sample.mask = np.zeros((7, 8), dtype=bool)
    sample.mask[1:6, 1:7] = True
    sample.mask_output = put(_npy(sample.mask), slot="segmentation_mask")
    sample.overlay_output = put(_jpeg(color="red"), slot="segmentation_overlay")
    targets = [
        {"target_pixel_uv": None, "language_prompt": "absent", "min_score": 0.5},
        {"target_pixel_uv": [2, 2], "language_prompt": None, "min_score": 0.6},
    ]
    sample.segment_request = _request(
        regs.segmentation,
        {
            "role": "main",
            "rgb_capture_operation_id": "operation:capture",
            "targets": targets,
        },
        (
            AgentArtifactBinding(
                slot_id="input_image", ordinal=0, artifact=sample.rgb_output.artifact
            ),
        ),
        name="segment",
    )
    sample.segment_result = {
        "source_id": "front",
        "role": "main",
        "captured_at": 10.0,
        "station_revision": 4,
        "input_rgb_content_hash": sample.rgb_output.artifact.sha256,
        "image_width": 8,
        "image_height": 7,
        "model_profile_id": "segment:test",
        "model_profile_hash": "model:test",
        "overlay_media_content_hash": sample.overlay_output.artifact.sha256,
        "completed_count": 1,
        "no_mask_count": 1,
        "rgb_capture_operation_id": "operation:capture",
        "rgb_capture_request_hash": sample.capture.request_hash,
        "input_rgb_artifact_id": sample.rgb_output.artifact.artifact_id,
        "input_rgb_frame_ordinal": 0,
        "relations": [
            {
                "indices": [0, 1],
                "metric": "euclidean_distance_between_occupied_pixel_centers",
                "status": "unavailable",
                "unavailable_reason": "absent_mask",
                "distance_px": None,
                "closest_pixels_uv": None,
                "semantic_identity_verified": False,
                "physical_clearance_m": None,
            }
        ],
        "items": [
            {
                "index": 0,
                **targets[0],
                "status": "no_mask",
                "mask_artifact_ordinal": None,
                "mask": None,
            },
            {
                "index": 1,
                **targets[1],
                "status": "completed",
                "mask_artifact_ordinal": 0,
                "mask": {
                    "score": 0.9,
                    "label": None,
                    "coordinate_format": "xyxy_pixel_0based_half_open",
                    "mask_bounds_xyxy": [1, 1, 7, 6],
                    "mask_pixel_count": 30,
                    "mask_media_content_hash": sample.mask_output.artifact.sha256,
                },
            },
        ],
    }
    sample.segment = _operation(
        sample.segment_request,
        sample.segment_result,
        (sample.mask_output, sample.overlay_output),
        "segment",
    )
    sample.operations["operation:segment"] = (sample.segment, sample.segment_request)

    class NoCapture:
        def capture_aligned_depth(self, **unused):
            raise AssertionError("Anchor reuse must not recapture")

        def capture_placement_rgbd(self, **unused):
            raise AssertionError("Anchor reuse must not recapture")

    sample.provider = LiveBackProjectionProvider(
        capture_port=NoCapture(), registration=regs.projection
    )
    project = sample.provider.project_capture

    def project_capture(*args, **kwargs):
        sample.projection_calls.append(args[0])
        result = project(*args, **kwargs)
        if hasattr(sample, "after_projection"):
            sample.after_projection()
        return result

    sample.provider.project_capture = project_capture
    sample.executor = build_surface_anchor_executor_v3(
        registration=regs.anchor,
        provider=sample.provider,
        segmentation_registration=regs.segmentation,
        rgb_registration=regs.rgb,
        store=store,
        lookup_operation=lambda key: _lookup_retained(sample.operations, key),
        current_station_revision=lambda: sample.revision,
        clock=lambda: sample.now,
        clock_domain_id="clock:test",
        max_retained_operations=3,
    )
    sample.executor.cancel_requested = lambda unused: sample.cancelled
    sample.operation = SimpleNamespace(
        operation_id="operation:anchor", deadline_at=20.0
    )
    yield sample
    store.close()


def _anchor_request(case, *, mode="mask_interior", pixel=None, index=1):
    bindings = [
        AgentArtifactBinding(
            slot_id="paired_observation",
            ordinal=0,
            artifact=case.paired_output.artifact,
        )
    ]
    if mode == "mask_interior":
        bindings.append(
            AgentArtifactBinding(
                slot_id="segmentation_mask",
                ordinal=0,
                artifact=case.mask_output.artifact,
            )
        )
    return _request(
        case.anchor,
        {
            "role": "main",
            "selection_mode": mode,
            "target_pixel_uv": pixel,
            "rgb_capture_operation_id": "operation:capture",
            "segmentation_operation_id": "operation:segment"
            if mode == "mask_interior"
            else None,
            "target_index": index if mode == "mask_interior" else None,
        },
        tuple(bindings),
    )


def _read(case, request=None):
    return case.executor._read(
        _anchor_request(case) if request is None else request, case.operation
    )


def _replace_producer_request(case, name, **changes):
    source, request = case.operations["operation:" + name]
    request = replace(request, **changes)
    source.request_hash = request.semantic_content_hash
    case.operations["operation:" + name] = (source, request)
    return request


def _replace_pair(case, **changes):
    metadata = json.loads(
        case.store.managed_read_path(
            case.paired_output.artifact.artifact_id
        ).read_bytes()
    )
    metadata.update(changes)
    case.paired_output = case.put(_json_bytes(metadata), slot="paired_observations")
    case.capture.output_artifact_bindings = (
        case.depth_output,
        case.paired_output,
        case.rgb_output,
    )
    case.capture_result["paired_observations"][0]["observation_sha256"] = (
        case.paired_output.artifact.sha256
    )
    case.capture_result["total_bytes"] = sum(
        binding.artifact.byte_length
        for binding in case.capture.output_artifact_bindings
    )


def test_registration_closes_hashes_and_preserves_exact_v2_result_schema(case):
    anchor = case.anchor
    anchor.payload_validator.validate_route(anchor.entry.route)
    assert anchor.skill.definition.skill_version == "3"
    assert (
        set(anchor.skill.schema_entry.parameter_schema["properties"])
        == PARAMETER_FIELDS
    )
    schema = capture_schema_v2(
        definition=anchor.skill.definition,
        back_projection_schema=case.projection.skill.schema_entry,
        segmentation_schema=SimpleNamespace(
            parameter_schema={
                "properties": case.segmentation.skill.schema_entry.parameter_schema[
                    "properties"
                ]["targets"]["items"]["properties"]
            }
        ),
        profile_hash="test",
    )
    assert (
        anchor.skill.schema_entry.execution_result_schema
        == schema.execution_result_schema
    )
    assert [
        (slot.slot_id, slot.min_count, slot.max_count)
        for slot in anchor.skill.io_contract.input_artifact_slots
    ] == [("paired_observation", 1, 1), ("segmentation_mask", 0, 1)]
    assert [
        slot.slot_id for slot in anchor.skill.io_contract.output_artifact_slots
    ] == ["surface_anchor"]
    assert (
        anchor.entry.route.control_effect_class == "read_only"
        and not anchor.entry.route.may_move_robot
    )


@pytest.mark.parametrize(
    "field",
    [
        "runtime_session_id",
        "control_domain_id",
        "target_binding_id",
        "station_manifest_hash",
        "execution_manifest_hash",
    ],
)
def test_session_binding_changes_do_not_change_static_anchor_contract(case, field):
    other = _registrations(deployment_identity={field: "other:" + field})
    for name in ("projection", "rgb", "segmentation", "anchor"):
        before, after = getattr(case, name), getattr(other, name)
        assert before.skill == after.skill
    assert case.anchor.entry.route == other.anchor.entry.route
    assert (
        case.anchor.entry.deployment_binding.deployment_binding_hash
        != other.anchor.entry.deployment_binding.deployment_binding_hash
    )
    provider = LiveBackProjectionProvider(
        capture_port=case.provider.capture_port, registration=other.projection
    )
    with pytest.raises(ValueError, match="providers differ"):
        build_surface_anchor_executor_v3(
            registration=case.anchor,
            provider=provider,
            segmentation_registration=other.segmentation,
            rgb_registration=other.rgb,
            store=case.store,
            lookup_operation=lambda key: _lookup_retained(case.operations, key),
            current_station_revision=lambda: 4,
            clock=lambda: 12.0,
            clock_domain_id="clock:test",
            max_retained_operations=3,
        )


    assert not case.projection_calls


def test_mujoco_service_localized_paired_anchor_chain_at_capacity_four(
    monkeypatch, tmp_path
):
    """Real service/store wiring, with only application cuts and model inference fake."""
    from test_agent_gateway_mujoco_tcp_pose import (
        _build_no_model_contract_bundle_fixture,
    )
    from test_agent_gateway_surface_anchor import _goal_parameters

    from agent_gateway.profiles.mujoco.motion_application import (
        MujocoAgentSkillMotionApplication,
    )
    from agent_gateway.profiles.mujoco.motion_deployment import (
        build_mujoco_agent_tcp_pose_contract_bundle,
    )
    from agent_gateway.profiles.mujoco.motion_factory import (
        _build_standard_tcp_pose_envelope_runtime,
    )
    from agent_gateway.profiles.mujoco.motion_gateway_session import (
        assemble_mujoco_agent_tcp_pose_gateway_session,
    )
    from control_station.sim_hil_profile.application import (
        MujocoSimHILPassiveApplicationCut,
    )
    from deployments.perception.sam2_grounding_dino import (
        GroundedObjectDetection,
        GroundedObjectSegment,
        encode_grounded_object_segment_artifacts,
    )
    from robot_motion.runtime.control_admission import ControlCut
    from sim.mujoco.depth_snapshot import EncodedMujocoDepthFrame
    from sim.mujoco.rgb_snapshot import EncodedMujocoRgbFrame

    identity = ("sam2:test", "sam2-model:test", "sam2-localizer:test")
    application, runtime, _direct, _baseline, _manifest = (
        _build_no_model_contract_bundle_fixture(
            monkeypatch,
            enable_artifact_perception=True,
            segmentation_identity=identity,
        )
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("This supplied-image chain must not render, recapture or plan motion")

    # Supply the existing assembler's lower ports without constructing a Scene/rig.
    application.scene = object()
    runtime.rig = SimpleNamespace(config=SimpleNamespace(), capture_scene=forbidden)
    application.target = SimpleNamespace(
        planned_tcp_motion=SimpleNamespace(
            pose_projector=forbidden, reference_hz=15.0, planning_clearance_padding_m=0.002
        )
    )
    application.depth_source_catalog = lambda: (
        {"view_id": "front", "native_width": 8, "native_height": 7},
    )
    application.visual_source_catalog = lambda: (
        {"source_id": "front", "native_width": 8, "native_height": 7},
    )
    application.tcp_pose_envelope_runtime = _build_standard_tcp_pose_envelope_runtime(
        prepared=application,
        target_config=SimpleNamespace(clock_domain_id=runtime.config.clock_domain_id),
        enable_safe_tcp_pose=True,
    )
    bundle = build_mujoco_agent_tcp_pose_contract_bundle(
        application=application,
        enable_artifact_perception=True,
        segmentation_model_profile_id=identity[0],
        segmentation_model_profile_hash=identity[1],
        segmentation_inference_implementation_hash=identity[2],
        enable_object_localization=True,
        enable_segmentation_geometry_v3=True,
        enable_rgbd_capture_v3=True,
        enable_safe_tcp_pose=True,
        enable_spatial_goals=True,
        enable_object_goal_queries=True,
        enable_surface_anchors=True,
        enable_surface_anchor_v2=True,
        enable_surface_anchor_v3=True,
    )
    motion_application = object.__new__(MujocoAgentSkillMotionApplication)
    motion_application.__dict__.update(application.__dict__)
    motion_application._lock = threading.RLock()
    motion_application._started = motion_application._held = (
        motion_application._closed
    ) = False
    motion_application._active_operation_id = None
    motion_application.capture_passive_application_cut = lambda **_: (
        MujocoSimHILPassiveApplicationCut(
            captured_at=10.0,
            authority_epoch=0,
            execution_epoch=0,
            dispatch_owner_generation=1,
            lifecycle_state="configured",
            execution_gate="not_started",
            execution_gate_epoch=0,
            fault_class=None,
            fault_rearmable=False,
            scene_projection={
                "finite_state": True,
                "pose_xyz_quat_xyzw_by_component": {
                    "right_arm": [0, 0, 0.5, 0, 0, 0, 1]
                },
            },
            control_cut=ControlCut(
                runtime_session_id=runtime.config.runtime_session_id,
                control_domain_id=runtime.config.control_domain_id,
                clock_domain_id=runtime.config.clock_domain_id,
                clock_reset_epoch=0,
                cutoff_at=10.0,
                cutoff_admission_sequence=0,
            ),
        )
    )
    rgb_bytes = _jpeg(color=(48, 96, 144))
    depth_bytes = _npy(np.full((7, 8), 0.5, dtype=np.float32))
    with Image.open(io.BytesIO(rgb_bytes)) as source:
        expected_pixels = np.asarray(source.convert("RGB")).copy()
    capture_calls = []

    def capture_visual_depth_cut(*, view_id):
        capture_calls.append(view_id)
        assert capture_calls == ["front"]
        return (
            EncodedMujocoRgbFrame("front", 8, 7, "image/jpeg", rgb_bytes),
            EncodedMujocoDepthFrame(
                view_id="front", captured_at=10.0, width=8, height=7,
                media_type=DEPTH_FRAME_MEDIA_TYPE_V1,
                schema_id=DEPTH_FRAME_SCHEMA_ID_V1,
                encoding=DEPTH_FRAME_ENCODING_V1,
                compression=DEPTH_FRAME_COMPRESSION_V1,
                data=depth_bytes,
                camera_pose_world_xyz_quat_xyzw=(0, 0, 0, 0, 0, 0, 1),
                intrinsics={"width": 8, "height": 7, "fx": 4., "fy": 4., "cx": 4., "cy": 3.},
            ),
        )

    motion_application.capture_visual_depth_cut = capture_visual_depth_cut
    motion_application.capture_visual_jpeg_snapshot = forbidden
    motion_application.capture_depth_npy_frame = forbidden

    class Localizer:
        semantic_content_hash = identity[2]
        model_spec = SimpleNamespace(
            model_profile_id=identity[0], semantic_content_hash=identity[1]
        )
        encode_segment_artifacts = staticmethod(encode_grounded_object_segment_artifacts)

        def __init__(self):
            self.starts = self.closes = 0
            self.started = False
            self.calls = []

        def start(self):
            self.starts += 1
            self.started = True

        def close(self):
            self.closes += 1
            self.started = False

        def localize(self, image, *, language_prompt, max_detections):
            assert self.started and language_prompt == "proposal" and max_detections == 1
            np.testing.assert_array_equal(np.asarray(image), expected_pixels)
            self.calls.append("localize")
            return (GroundedObjectDetection((1, 1, 7, 6), 0.9, "proposal"),)

        def segment(self, image, *, language_prompt, pixel_uv, min_score):
            assert self.started and language_prompt is None
            assert pixel_uv == (4, 3) and min_score == 0.25
            np.testing.assert_array_equal(np.asarray(image), expected_pixels)
            self.calls.append("segment")
            mask = np.zeros((7, 8), dtype=bool)
            mask[1:6, 1:7] = True
            return GroundedObjectSegment(mask, 0.9, "proposal", (1, 1, 7, 6))

    localizer = Localizer()
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts", quota_bytes=32 << 20, max_artifact_bytes=16 << 20,
        max_chunk_bytes=1 << 20,
    )
    session = assemble_mujoco_agent_tcp_pose_gateway_session(
        application=motion_application, contract_bundle=bundle,
        allowed_client_ids=("test:combo",), max_retained_operations=4,
        artifact_store=store, owns_artifact_store=True,
        segmentation_localizer=localizer,
    )
    try:
        session.start_perception_resources()
        session.start_perception_resources()
        assert localizer.starts == 1
        assert session.localization_provider.localizer is localizer
        assert session.segmentation_provider.localizer is localizer
        published = session.observation_producer.capture_and_publish(session.service)
        assert published.accepted and published.snapshot is not None
        snapshot = published.snapshot
        ports = {(item.skill_id, item.skill_version): item.execution_port
                 for item in session.service.executor_registry.bindings}
        assert ports["perception.capture_rgb", "3"] is session.rgbd_capture_executor
        assert ports["perception.localize_objects", "1"] is session.localization_executor
        assert ports["perception.segment", "3"] is session.segmentation_geometry_executor
        sequence = 0

        def submit(
            skill, version, parameters, inputs=(), *, status="completed", rejection_reason=None
        ):
            nonlocal sequence
            sequence += 1
            entry = bundle.route_manifest.route(skill, version)
            route, profile = entry.route, entry.route.allowed_parameter_profiles[0]
            request = AgentSkillRequest(
                client_id="test:combo", request_id=f"request:combo:{sequence}",
                idempotency_key=f"idempotency:combo:{sequence}", client_sequence=sequence,
                runtime_session_id=bundle.route_manifest.runtime_session_id,
                skill_route_manifest_hash=bundle.route_manifest.semantic_content_hash,
                expected_station_revision=snapshot.station_revision,
                observation_snapshot_id=snapshot.snapshot_id,
                skill_id=skill, skill_version=version,
                skill_definition_hash=route.skill_definition_hash,
                route_id=route.route_id, route_version=route.route_version,
                route_hash=route.semantic_content_hash,
                deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
                parameter_profile_id=profile.parameter_profile_id,
                parameter_profile_hash=profile.parameter_profile_hash,
                parameters=parameters, requested_mode="sim_active",
                requested_timeout_ms=route.max_operation_duration_ms,
                input_artifact_bindings=inputs,
            )
            admitted = session.service.submit_request(
                request=request, current_station_revision=snapshot.station_revision
            )
            if rejection_reason is not None:
                assert not admitted.accepted and admitted.operation is None
                assert admitted.reason_code == rejection_reason
                return admitted
            assert admitted.accepted and admitted.operation is not None, admitted
            op_id = admitted.operation.operation_id
            assert session.service.operation_store.request_for_operation(op_id) == request
            assert session.service.launch_operation(op_id).accepted
            assert ports[skill, version].wait(op_id, timeout_sec=5.0)
            outcome = session.service.get_operation(op_id)
            assert outcome is not None and outcome.status == status, (
                outcome, ports[skill, version].failure_diagnostics
            )
            assert outcome.execution_outcome == status
            return outcome

        def output(operation, source_slot, target_slot):
            matching = [b for b in operation.output_artifact_bindings if b.slot_id == source_slot]
            assert len(matching) == 1
            return replace(matching[0], slot_id=target_slot, ordinal=0, production_receipt_ref=None)

        captured = submit("perception.capture_rgb", "3", {
            "selected_roles": ["main"], "width": 8, "height": 7,
            "max_total_bytes": 65536, "include_depth": True,
        })
        original = output(captured, "rgb_frames", "input_image")
        assert store.managed_read_path(original.artifact.artifact_id).read_bytes() == rgb_bytes
        rgb_hash = hashlib.sha256(rgb_bytes).hexdigest()
        assert original.artifact.sha256 == rgb_hash
        localized = submit("perception.localize_objects", "1", {
            "input_source": "artifact", "local_image_path": None,
            "language_prompt": "proposal", "max_detections": 1,
        }, (original,))
        assert localized.execution_result["input_media_content_hash"] == rgb_hash
        assert (localized.execution_result["image_width"], localized.execution_result["image_height"]) == (8, 7)
        annotated = output(localized, "annotated_image", "input_image")
        assert annotated.artifact.sha256 != rgb_hash
        x0, y0, x1, y1 = localized.execution_result["detections"][0]["box_xyxy"]
        pixel = [(x0 + x1) // 2, (y0 + y1) // 2]
        segment_parameters = {
            "role": "main", "rgb_capture_operation_id": captured.operation_id,
            "targets": [{"target_pixel_uv": pixel, "language_prompt": None, "min_score": 0.25}],
        }
        segmented = submit("perception.segment", "3", segment_parameters, (original,))
        segment = segmented.execution_result
        assert segment["input_rgb_content_hash"] == rgb_hash
        assert segment["input_rgb_artifact_id"] == original.artifact.artifact_id
        assert segment["rgb_capture_request_hash"] == captured.request_hash
        assert segment["rgb_capture_operation_id"] == captured.operation_id
        assert segment["input_rgb_frame_ordinal"] == 0
        assert (segment["image_width"], segment["image_height"]) == (8, 7)
        pair = output(captured, "paired_observations", "paired_observation")
        metadata = json.loads(store.managed_read_path(pair.artifact.artifact_id).read_bytes())
        anchor_parameters = {
            "role": "main", "selection_mode": "mask_interior", "target_pixel_uv": None,
            "rgb_capture_operation_id": captured.operation_id,
            "segmentation_operation_id": segmented.operation_id, "target_index": 0,
        }
        anchor_inputs = (pair, output(segmented, "segmentation_mask", "segmentation_mask"))
        anchored = submit("perception.capture_surface_anchor", "3", anchor_parameters, anchor_inputs)
        answer = anchored.execution_result
        validate_anchor_v2(answer)
        assert answer["selection"]["input_rgb_content_hash"] == rgb_hash == metadata["rgb_artifact"]["sha256"]
        assert answer["selection"]["mask_media_content_hash"] == anchor_inputs[1].artifact.sha256
        assert tuple(answer["selection"]["requested_pixel_uv"]) == tuple(pixel)
        assert answer["measurement"]["capture_group_id"] == metadata["capture_group_id"]
        assert answer["measurement"]["depth_content_hash"] == metadata["depth_content_hash"]
        assert answer["captured_at"] == metadata["captured_at"] == 10.0
        assert metadata["source_operation_id"] == captured.operation_id
        assert metadata["source_request_hash"] == captured.request_hash
        assert answer["source_operation_id"] == anchored.operation_id
        assert answer["source_request_hash"] == anchored.request_hash
        for key in ("reference_frame_id", "reference_frame_spec_hash", "frame_graph_hash"):
            assert answer["measurement"][key] == metadata[key]
        assert session.service.operation_store.retained_operation_count == 4
        assert session.service.operation_store.total_evicted == 0
        assert {op.operation_id for op in session.service.operation_store.operations()} == {
            captured.operation_id, localized.operation_id, segmented.operation_id, anchored.operation_id
        }

        # The preview is not an original RGB Artifact; reject before allocating an operation.
        submit("perception.segment", "3", segment_parameters, (annotated,),
               rejection_reason="input_artifact_bindings_rejected")
        assert session.service.operation_store.total_admitted == 4
        assert session.service.operation_store.total_evicted == 0
        assert localizer.calls == ["localize", "segment"]

        # A real fifth, model-free operation evicts capture, not its CAS bytes.
        submit("perception.compute_spatial_goal_from_anchor", "2", {
            **_goal_parameters(), "reference_frame_id": bundle.motion_profile.reference_frame_id,
        }, (output(anchored, "surface_anchor", "surface_anchor"),))
        assert session.service.get_operation(captured.operation_id) is None
        assert session.service.operation_store.request_for_operation(captured.operation_id) is None
        assert session.service.operation_store.total_evicted == 1
        assert store.managed_read_path(pair.artifact.artifact_id).read_bytes()
        rejected = submit("perception.capture_surface_anchor", "3", anchor_parameters, anchor_inputs, status="failed")
        assert rejected.execution_outcome == "failed"
        assert rejected.execution_result is None and not rejected.output_artifact_bindings
        assert rejected.progress_summary["failure_diagnostic"]["message"] == "Reuse producer is unavailable in this session."
        assert session.service.get_operation(segmented.operation_id) is not None
        assert capture_calls == ["front"] and localizer.calls == ["localize", "segment"]
        assert localizer.starts == 1 and not motion_application.started
        # These synthetic model proposals establish wiring, not object identity or a grasp.
    finally:
        session.close()
    assert localizer.closes == 1 and motion_application.closed


@pytest.mark.parametrize("changed", ["session", "binding", "both"])
def test_matching_static_route_does_not_admit_another_session_request(case, changed):
    other = _registrations(deployment_identity={"runtime_session_id": "session:other"})
    request = _anchor_request(case)
    assert request.route_hash == other.anchor.entry.route.semantic_content_hash
    updates = {}
    if changed in {"session", "both"}:
        updates["runtime_session_id"] = "session:other"
    if changed in {"binding", "both"}:
        updates["deployment_binding_hash"] = (
            other.anchor.entry.deployment_binding.deployment_binding_hash
        )
    with pytest.raises(ValueError):
        _read(case, replace(request, **updates))
    assert not case.projection_calls


def test_exact_reuses_original_paired_bytes_and_preserves_pixel_without_segmentation(
    case,
):
    case.operations.pop("operation:segment")
    result = _read(case, _anchor_request(case, mode="exact", pixel=[2, 2]))
    answer = result.execution_result
    validate_anchor_v2(answer)
    assert set(answer) == ANCHOR_FIELDS | {"selection", "mask_geometry"}
    assert answer["source_operation_id"] == "operation:anchor"
    assert answer["captured_at"] == 10.0 and answer["clock_domain_id"] == "clock:test"
    assert answer["target_pixel_uv"] == [2, 2] and answer["mask_geometry"] is None
    assert (
        answer["selection"]["min_score"] == 0.0
        and answer["selection"]["language_prompt"] is None
    )
    nullable = SELECTION_FIELDS - {
        "mode",
        "requested_pixel_uv",
        "min_score",
        "algorithm_id",
        "input_rgb_content_hash",
    }
    assert all(answer["selection"][key] is None for key in nullable)
    assert answer["selection"]["algorithm_id"] == EXACT_ALGORITHM
    output = result.output_artifact_bindings[0]
    assert output.artifact.schema_id == SURFACE_ANCHOR_SCHEMA_ID_V2
    assert case.store.managed_read_path(
        output.artifact.artifact_id
    ).read_bytes() == _json_bytes(answer)
    assert len(case.projection_calls) == 1


@pytest.mark.parametrize("pixel", [None, [2, 2]])
def test_target_one_uses_mask_ordinal_zero_and_inherits_authenticated_point(
    case, pixel
):
    result = _read(case, _anchor_request(case, pixel=pixel)).execution_result
    validate_anchor_v2(result)
    assert result["selection"]["requested_pixel_uv"] == [2, 2]
    assert result["selection"]["min_score"] == 0.6
    assert (
        result["selection"]["mask_media_content_hash"]
        == case.mask_output.artifact.sha256
    )
    assert result["target_pixel_uv"] != [2, 2]
    assert case.mask[result["target_pixel_uv"][1], result["target_pixel_uv"][0]]
    assert result["mask_geometry"]["valid_point_count"] == 30


def _text_item(case):
    item = case.segment_result["items"][1]
    item.update(target_pixel_uv=None, language_prompt="blue target")
    targets = [dict(target) for target in case.segment_request.parameters["targets"]]
    targets[1].update(target_pixel_uv=None, language_prompt="blue target")
    case.segment_request = _replace_producer_request(
        case,
        "segment",
        parameters={**dict(case.segment_request.parameters), "targets": targets},
    )


def test_text_item_is_inherited_without_new_prompt_or_model_call(case):
    _text_item(case)
    result = _read(case).execution_result
    assert result["selection"]["language_prompt"] == "blue target"
    assert result["selection"]["requested_pixel_uv"] is None


@pytest.mark.parametrize(
    "kind", ["point_mismatch", "text_with_pixel", "no_mask", "missing_index"]
)
def test_mask_selection_never_substitutes_a_different_prompt_or_target(case, kind):
    if kind == "text_with_pixel":
        _text_item(case)
    request = _anchor_request(
        case,
        pixel=[3, 2]
        if kind == "point_mismatch"
        else [2, 2]
        if kind == "text_with_pixel"
        else None,
        index=0 if kind == "no_mask" else 2 if kind == "missing_index" else 1,
    )
    with pytest.raises(ValueError):
        _read(case, request)
    assert not case.projection_calls


@pytest.mark.parametrize("name", ["capture", "segment"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("client_id", "other-client"),
        ("runtime_session_id", "other-session"),
        ("skill_route_manifest_hash", "other-manifest"),
        ("expected_station_revision", 5),
        ("skill_id", "perception.other"),
        ("skill_version", "2"),
        ("skill_definition_hash", "wrong-definition"),
        ("route_id", "other.route"),
        ("route_version", "2"),
        ("route_hash", "wrong-route"),
        ("deployment_binding_hash", "wrong-binding"),
        ("parameter_profile_id", "other-profile"),
        ("parameter_profile_hash", "other-profile-hash"),
        ("requested_mode", "real_observe"),
    ],
)
def test_producer_identity_mismatch_rejected_even_with_consistent_request_hash(
    case, name, field, value
):
    _replace_producer_request(case, name, **{field: value})
    with pytest.raises(ValueError):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize("name", ["capture", "segment"])
@pytest.mark.parametrize(
    "kind",
    [
        "evicted",
        "failed",
        "cancelled",
        "outcome",
        "no_result",
        "request_hash",
        "session",
        "id",
    ],
)
def test_only_retained_completed_producers_are_accepted(case, name, kind):
    source = case.operations["operation:" + name][0]
    if kind == "evicted":
        case.operations.pop("operation:" + name)
    elif kind in {"failed", "cancelled"}:
        source.status = kind
    elif kind == "outcome":
        source.execution_outcome = "failed"
    elif kind == "no_result":
        source.execution_result = None
    elif kind == "request_hash":
        source.request_hash = "0" * 64
    elif kind == "session":
        source.runtime_session_id = "other-session"
    elif kind == "id":
        source.operation_id = "operation:other"
    with pytest.raises(ValueError):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_operation_id", "operation:other"),
        ("source_request_hash", "a" * 64),
        ("runtime_session_id", "other-session"),
        ("station_revision", 5),
        ("clock_domain_id", "other-clock"),
        ("reference_frame_id", "other-frame"),
        ("reference_frame_spec_hash", "other-spec"),
        ("frame_graph_hash", "other-graph"),
        ("captured_at", 11.0),
        ("role", "wrist_left"),
        ("source_id", "other-source"),
        ("width", 7),
        ("height", 6),
    ],
)
def test_actual_paired_metadata_must_match_authenticated_producer_and_bound_frame(
    case, field, value
):
    _replace_pair(case, **{field: value})
    with pytest.raises((ValueError, RuntimeError)):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("rgb_capture_operation_id", "operation:other"),
        ("rgb_capture_request_hash", "a" * 64),
        ("input_rgb_artifact_id", "other-rgb"),
        ("input_rgb_frame_ordinal", 1),
        ("input_rgb_content_hash", "a" * 64),
        ("image_width", 9),
        ("image_height", 8),
        ("captured_at", 11.0),
    ],
)
def test_segmentation_rgb_provenance_and_grid_cannot_be_mixed(case, field, value):
    case.segment_result[field] = value
    with pytest.raises(ValueError):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize(
    "kind",
    [
        "paired_forged",
        "mask_forged",
        "mask_wrong_ref",
        "wrong_input_rgb",
        "wrong_rgb_ref",
        "missing_paired_output",
        "mask_wrong_ordinal",
        "no_mask_input",
        "extra_mask_exact",
        "paired_input_ordinal",
    ],
)
def test_input_descriptors_are_exact_original_producer_outputs(case, kind):
    request = _anchor_request(case)
    if kind == "paired_forged":
        changed = case.put(_json_bytes({"fake": True}), slot="paired_observations")
        request = replace(
            request,
            input_artifact_bindings=(
                replace(request.input_artifact_bindings[0], artifact=changed.artifact),
                request.input_artifact_bindings[1],
            ),
        )
    elif kind == "mask_forged":
        changed = case.put(_npy(~case.mask), slot="segmentation_mask")
        request = replace(
            request,
            input_artifact_bindings=(
                request.input_artifact_bindings[0],
                replace(request.input_artifact_bindings[1], artifact=changed.artifact),
            ),
        )
    elif kind == "mask_wrong_ref":
        binding = request.input_artifact_bindings[1]
        request = replace(
            request,
            input_artifact_bindings=(
                request.input_artifact_bindings[0],
                replace(
                    binding,
                    artifact=replace(
                        binding.artifact, byte_length=binding.artifact.byte_length + 1
                    ),
                ),
            ),
        )
    elif kind == "wrong_input_rgb":
        changed = case.put(_jpeg(color="blue"), slot="rgb_frames")
        _replace_producer_request(
            case,
            "segment",
            input_artifact_bindings=(
                AgentArtifactBinding(
                    slot_id="input_image", ordinal=0, artifact=changed.artifact
                ),
            ),
        )
    elif kind == "wrong_rgb_ref":
        case.capture.output_artifact_bindings = (
            case.depth_output,
            case.paired_output,
            replace(
                case.rgb_output,
                artifact=replace(case.rgb_output.artifact, encoding="other"),
            ),
        )
    elif kind == "missing_paired_output":
        case.capture.output_artifact_bindings = (case.depth_output, case.rgb_output)
    elif kind == "mask_wrong_ordinal":
        case.segment_result["items"][1]["mask_artifact_ordinal"] = 1
    elif kind == "no_mask_input":
        request = replace(
            request, input_artifact_bindings=request.input_artifact_bindings[:1]
        )
    elif kind == "extra_mask_exact":
        exact = _anchor_request(case, mode="exact", pixel=[2, 2])
        request = replace(
            exact, input_artifact_bindings=request.input_artifact_bindings
        )
    elif kind == "paired_input_ordinal":
        request = replace(
            request,
            input_artifact_bindings=(
                replace(request.input_artifact_bindings[0], ordinal=1),
                request.input_artifact_bindings[1],
            ),
        )
    with pytest.raises((ValueError, RuntimeError)):
        _read(case, request)
    assert not case.projection_calls


@pytest.mark.parametrize(
    "which",
    ["paired_output", "rgb_output", "depth_output", "mask_output", "overlay_output"],
)
def test_corrupt_managed_bytes_fail_closed_before_projection(case, which):
    artifact = getattr(case, which).artifact
    path = case.store.managed_read_path(artifact.artifact_id)
    path.chmod(0o600)
    path.write_bytes(b"changed")
    with pytest.raises(ArtifactStoreIntegrityError):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize("kind", ["shape", "dtype", "count", "bounds", "trailing"])
def test_mask_bytes_are_decoded_not_trusted_from_json(case, kind):
    mask = case.mask
    if kind == "shape":
        mask = mask[:-1]
    if kind == "dtype":
        mask = mask.astype(np.uint8)
    data = _npy(mask) + (b"trailing" if kind == "trailing" else b"")
    case.mask_output = case.put(data, slot="segmentation_mask")
    case.segment.output_artifact_bindings = (case.mask_output, case.overlay_output)
    evidence = case.segment_result["items"][1]["mask"]
    evidence["mask_media_content_hash"] = case.mask_output.artifact.sha256
    if kind == "count":
        evidence["mask_pixel_count"] = 29
    if kind == "bounds":
        evidence["mask_bounds_xyxy"] = [0, 0, 7, 6]
    with pytest.raises(RuntimeError):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize("where", ["initial", "after_projection", "after_commit"])
@pytest.mark.parametrize("kind", ["cancel", "deadline", "revision"])
def test_interruption_never_returns_a_successful_anchor(case, monkeypatch, where, kind):
    def interrupt():
        if kind == "cancel":
            case.cancelled = True
        elif kind == "deadline":
            case.now = 20.0
        else:
            case.revision = 5

    if where == "initial":
        interrupt()
    elif where == "after_projection":
        case.after_projection = interrupt
    else:
        from agent_gateway.profiles import surface_anchor_v3 as owner

        original = owner.commit_artifact_bytes

        def commit(**kwargs):
            result = original(**kwargs)
            interrupt()
            return result

        monkeypatch.setattr(owner, "commit_artifact_bytes", commit)
    with pytest.raises(RuntimeError):
        _read(case)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "unknown"),
        ("selection_mode", "auto"),
        ("target_pixel_uv", [8, 1]),
        ("target_pixel_uv", [True, 2]),
        ("target_pixel_uv", [2.0, 2]),
        ("rgb_capture_operation_id", None),
        ("rgb_capture_operation_id", " padded"),
        ("segmentation_operation_id", None),
        ("target_index", True),
        ("target_index", 3),
        ("target_index", -1),
    ],
)
def test_parameters_reject_invalid_native_grid_or_selector(case, field, value):
    request = _anchor_request(case)
    with pytest.raises((TypeError, ValueError)):
        case.anchor.payload_validator.validate_parameters(
            {**dict(request.parameters), field: value}
        )


@pytest.mark.parametrize("extra", ["language_prompt", "min_score", "world_xyz_m"])
def test_new_parameters_never_accept_new_text_score_or_caller_xyz(case, extra):
    with pytest.raises(ValueError):
        case.anchor.payload_validator.validate_parameters(
            {**dict(_anchor_request(case).parameters), extra: None}
        )


@pytest.mark.parametrize("kind", ["no_pixel", "segment_id", "target_index"])
def test_exact_requires_pixel_and_null_segmentation_selectors(case, kind):
    value = dict(_anchor_request(case, mode="exact", pixel=[2, 2]).parameters)
    if kind == "no_pixel":
        value["target_pixel_uv"] = None
    elif kind == "segment_id":
        value["segmentation_operation_id"] = "operation:segment"
    else:
        value["target_index"] = 1
    with pytest.raises(ValueError):
        case.anchor.payload_validator.validate_parameters(value)


def test_explicit_medoid_policy_changes_identity_not_anchor_document_shape():
    one, two = _registrations(), _registrations("max_clearance_medoid")
    assert (
        one.anchor.entry.route.semantic_content_hash
        != two.anchor.entry.route.semantic_content_hash
    )
    assert two.anchor.skill.schema_entry.execution_result_schema["properties"][
        "selection"
    ]["properties"]["algorithm_id"]["enum"] == (
        EXACT_ALGORITHM,
        MAX_CLEARANCE_MEDOID_ALGORITHM,
    )


def test_inputs_and_retained_records_are_not_rewritten(case):
    before = _json_bytes(case.capture_result), _json_bytes(case.segment_result)
    requests = (
        case.capture_request.semantic_content_hash,
        case.segment_request.semantic_content_hash,
    )
    _read(case)
    assert before == (
        _json_bytes(case.capture_result),
        _json_bytes(case.segment_result),
    )
    assert requests == (
        case.capture_request.semantic_content_hash,
        case.segment_request.semantic_content_hash,
    )


@pytest.mark.parametrize("kind", ["different_rgb", "different_depth", "different_grid"])
def test_actual_paired_media_refs_and_grid_cannot_be_substituted(case, kind):
    if kind == "different_rgb":
        other = case.put(_jpeg(color="blue"), slot="rgb_frames")
        _replace_pair(case, rgb_artifact=other.artifact.to_wire())
    elif kind == "different_depth":
        other = case.put(
            _npy(np.full((7, 8), 0.6, dtype=np.float32)), slot="depth_frames"
        )
        _replace_pair(case, depth_artifact=other.artifact.to_wire())
    else:
        # A valid JPEG descriptor still cannot turn a different grid into this cut.
        case.rgb_output = case.put(_jpeg(size=(7, 7)), slot="rgb_frames")
        case.capture.output_artifact_bindings = (
            case.depth_output,
            case.paired_output,
            case.rgb_output,
        )
        case.capture_result["frames"][0]["media_content_hash"] = (
            case.rgb_output.artifact.sha256
        )
        _replace_pair(case, rgb_artifact=case.rgb_output.artifact.to_wire())
    with pytest.raises(ValueError):
        _read(case, _anchor_request(case, mode="exact", pixel=[2, 2]))
    assert not case.projection_calls


@pytest.mark.parametrize(
    "which",
    ["paired_output", "rgb_output", "depth_output", "mask_output", "overlay_output"],
)
def test_managed_path_resolution_is_not_a_substitute_for_hashing_the_bytes_read(
    case, monkeypatch, tmp_path, which
):
    artifact = getattr(case, which).artifact
    original = case.store._blob_path
    changed = tmp_path / "changed-at-cas-read"
    changed.write_bytes(b"!" * artifact.byte_length)
    monkeypatch.setattr(
        case.store,
        "_blob_path",
        lambda key: changed if key == artifact.sha256 else original(key),
    )
    with pytest.raises(ArtifactStoreIntegrityError, match="verification"):
        _read(case)
    assert case.store.artifact(artifact.artifact_id).state == "quarantined"
    assert not case.projection_calls


@pytest.mark.parametrize(
    "boundary", ["lookup", "managed_path", "decode", "select", "mask_geometry"]
)
def test_cancel_is_checked_between_each_bounded_reuse_stage(
    case, monkeypatch, boundary
):
    from agent_gateway.profiles import surface_anchor_v3 as owner

    if boundary == "lookup":
        original = case.operations.get

        def lookup(key, default):
            value = original(key, default)
            case.cancelled = True
            return value

        monkeypatch.setattr(case, "operations", SimpleNamespace(get=lookup))
    elif boundary == "managed_path":
        original = case.store.read_bytes

        def resolve(artifact, *, maximum):
            data = original(artifact, maximum=maximum)
            case.cancelled = True
            return data

        monkeypatch.setattr(case.store, "read_bytes", resolve)
    else:
        name = {
            "decode": "decode_paired_observation",
            "select": "select_mask_interior_pixel",
            "mask_geometry": "project_depth_mask_geometry",
        }[boundary]
        original = getattr(owner, name)

        def interrupt(*args, **kwargs):
            result = original(*args, **kwargs)
            case.cancelled = True
            return result

        monkeypatch.setattr(owner, name, interrupt)
    with pytest.raises(RuntimeError, match="cancelled or expired"):
        _read(case)
    assert not case.projection_calls


@pytest.mark.parametrize(
    "kind",
    ["projection_frame", "segmentation_route", "capture_route", "capture_binding"],
)
def test_executor_rejects_providers_different_from_the_published_registration(
    case, kind
):
    projection, segmentation, rgb = case.projection, case.segmentation, case.rgb
    if kind == "projection_frame":
        # Rebuild a coherent but different projection and capture; the anchor must
        # still reject them instead of retaining the original advertised hashes.
        from agent_gateway.profiles.mujoco.rgb_deployment import (
            build_mujoco_rgb_capture_registration_v2,
        )

        identity = {
            key: getattr(projection.binding, key)
            for key in (
                "runtime_session_id",
                "control_domain_id",
                "target_binding_id",
                "station_manifest_hash",
                "execution_manifest_hash",
            )
        }
        sources = (
            {
                "source_id": "front",
                "role": "main",
                "native_width": 8,
                "native_height": 7,
            },
        )
        projection = build_back_projection_registration(
            target_kind="sim",
            current_mode="sim_active",
            source_catalog=sources,
            reference_frame_id="canonical_station",
            reference_frame_spec_hash="frame:other",
            frame_graph_hash="graph:other",
            **identity,
        )
        rgb2 = build_mujoco_rgb_capture_registration_v2(
            source_catalog=sources, **identity
        )
        rgb = build_rgbd_capture_registration(
            rgb_registration=rgb2, back_projection=projection
        )
    elif kind == "segmentation_route":
        profile = replace(segmentation.profile, rgb_capture_route_hash="a" * 64)
        # An incoherent typed replacement must fail before it reaches execution.
        with pytest.raises(ValueError):
            replace(segmentation, profile=profile)
        return
    elif kind == "capture_route":
        rgb = replace(
            rgb,
            entry=replace(
                rgb.entry, route=replace(rgb.entry.route, route_id="other.rgb")
            ),
        )
    else:
        rgb = replace(
            rgb,
            entry=replace(
                rgb.entry,
                deployment_binding=replace(
                    rgb.entry.deployment_binding, target_binding_id="other-target"
                ),
            ),
        )
    provider = LiveBackProjectionProvider(
        capture_port=case.provider.capture_port, registration=projection
    )
    with pytest.raises(ValueError):
        build_surface_anchor_executor_v3(
            registration=case.anchor,
            provider=provider,
            segmentation_registration=segmentation,
            rgb_registration=rgb,
            store=case.store,
            lookup_operation=lambda key: _lookup_retained(case.operations, key),
            current_station_revision=lambda: 4,
            clock=lambda: 12.0,
            clock_domain_id="clock:test",
            max_retained_operations=3,
        )

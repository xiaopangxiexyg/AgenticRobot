from __future__ import annotations

import base64
import io
import hashlib
import threading
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from agent_gateway.profiles import segmentation as segmentation_profile
from agent_gateway.profiles.real import realsense_perception
from agent_gateway.profiles.real.perception_application import (
    assemble_camera_only_real_perception_composition,
)
from agent_gateway.profiles.real.perception_frames import (
    build_real_perception_station_frame_binding,
)
from agent_gateway.profiles.real.perception_gateway import (
    assemble_real_perception_gateway_session,
)
from agent_gateway.profiles.real.realsense_perception import (
    RealPerceptionSourceBinding,
)
from agent_gateway.profiles.segmentation import (
    SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
    SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
    LiveRgbSegmentationCapture,
    LiveSegmentationProvider,
    build_segmentation_executor,
    build_segmentation_registration,
)
from agent_gateway.profiles.segmentation_batch import (
    _COLORS,
    _indexed_contour_overlay,
    build_segmentation_batch_executor,
    prepare_segmentation_batch,
)
from agent_gateway.protocol.contracts import AgentSkillRequest
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from agent_gateway.skills.perception.definition import segment_skill_definition_v1
from agent_gateway.skills.perception.segmentation_schemas import (
    segment_schema_entry_v1,
    validate_segment_parameters_v1,
)
from agent_gateway.transport.client_bindings import (
    compile_agent_gateway_unix_client_bindings,
)
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectSegment,
    Sam2GroundingDinoLocalizer,
    Sam2PointMaskUnavailable,
    encode_grounded_object_segment_artifacts,
)
from teleop_runtime.video_frame import VideoFrame


class _FakeSegmenter:
    def __init__(self) -> None:
        self.model_spec = SimpleNamespace(
            model_profile_id="sam2:test",
            semantic_content_hash="sam2-model:test",
        )
        self.semantic_content_hash = "sam2-localizer:test"
        self.started = False
        self.closed = False
        self.images: list[tuple[int, int]] = []
        self.on_segment = None

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.started = False
        self.closed = True

    def segment(self, image, *, language_prompt, pixel_uv, min_score):
        assert self.started
        self.images.append(image.size)
        if self.on_segment is not None:
            self.on_segment()
        mask = np.zeros((image.height, image.width), dtype=bool)
        if pixel_uv is None:
            x0 = image.width // 4
            x1 = max(x0 + 1, 3 * image.width // 4)
            y0 = image.height // 4
            y1 = max(y0 + 1, 3 * image.height // 4)
            mask[y0:y1, x0:x1] = True
            label = language_prompt
            bounds = (x0, y0, x1, y1)
        else:
            u, v = pixel_uv
            mask[max(0, v - 5) : v + 6, max(0, u - 5) : u + 6] = True
            label = None
            bounds = (
                max(0, u - 5),
                max(0, v - 5),
                min(image.width, u + 6),
                min(image.height, v + 6),
            )
        return GroundedObjectSegment(
            mask=mask,
            score=max(0.95, float(min_score)),
            label=label,
            box_xyxy=bounds,
        )

    def localize(self, *_args, **_kwargs):
        return ()

    encode_segment_artifacts = staticmethod(encode_grounded_object_segment_artifacts)


class _RealCameraSource:
    def __init__(self) -> None:
        self.generation = 1
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
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        image[:, :2, 0] = 255
        self.frame = VideoFrame(
            image=image,
            source_timestamp=9.95,
            receive_timestamp=9.951,
            available_timestamp=9.952,
            frame_index=1,
            metadata={
                "camera_key": "front",
                "serial": "front-serial",
                "source_generation": self.generation,
            },
            depth_image=np.full((3, 4), 500, dtype=np.uint16),
            depth_metadata={
                "aligned_to": "color",
                "depth_scale_m_per_unit": 0.001,
            },
        )

    def get_latest_frame(self):
        return self.frame

    def device_info(self):
        return {
            "running": True,
            "lifecycle_state": "running",
            "worker_alive": True,
            "serial_number": "front-serial",
            "generation": self.generation,
            "intrinsics": {
                "width": 4,
                "height": 3,
                "fx": 4.0,
                "fy": 3.0,
                "ppx": 2.0,
                "ppy": 1.5,
            },
        }

    def start(self):
        return None

    def close(self):
        return None

    def advance_generation(self) -> None:
        self.generation += 1
        self.frame.metadata["source_generation"] = self.generation


def test_segment_schema_uses_simple_nullable_fields_and_validates_exclusivity() -> None:
    definition = segment_skill_definition_v1()
    schema = segment_schema_entry_v1(
        skill_definition_hash=definition.semantic_content_hash,
        parameter_profile_id="segment:test",
        parameter_profile_hash="segment-profile:test",
        allowed_roles=("main", "wrist_left"),
    )

    assert "oneOf" not in repr(schema.parameter_schema)
    assert "anyOf" not in repr(schema.parameter_schema)
    validate_segment_parameters_v1(
        {
            "role": "main",
            "language_prompt": "red cube",
            "target_pixel_uv": None,
            "min_score": 0.4,
        },
        allowed_roles=("main", "wrist_left"),
    )
    validate_segment_parameters_v1(
        {
            "role": "wrist_left",
            "language_prompt": None,
            "target_pixel_uv": [12, 7],
            "min_score": 0.0,
        },
        allowed_roles=("main", "wrist_left"),
    )
    for prompt, pixel in ((None, None), ("cube", [1, 1])):
        with pytest.raises(ValueError, match="exactly one"):
            validate_segment_parameters_v1(
                {
                    "role": "main",
                    "language_prompt": prompt,
                    "target_pixel_uv": pixel,
                    "min_score": 0.2,
                },
                allowed_roles=("main",),
            )


@pytest.mark.parametrize("mask_dtype", [np.bool_, np.float32, np.float64, np.uint8])
def test_shared_localizer_point_segmentation_preserves_score_and_mask(mask_dtype) -> None:
    class _Predictor:
        def set_image(self, image):
            assert image.shape == (6, 8, 3)

        def predict(self, **kwargs):
            assert kwargs["point_coords"].tolist() == [[3.0, 2.0]]
            assert kwargs.get("return_logits", False) is False
            masks = np.zeros((2, 6, 8), dtype=mask_dtype)
            masks[0, 2, 3] = True
            masks[1, 1:4, 2:5] = True
            return masks, np.asarray([0.4, 0.9]), None

    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started = True
    localizer._closed = False
    localizer._predictor = _Predictor()
    image = Image.new("RGB", (8, 6), "white")

    result = localizer.segment(
        image,
        language_prompt=None,
        pixel_uv=(3, 2),
        min_score=0.5,
    )

    assert result is not None
    assert result.score == pytest.approx(0.9)
    assert result.box_xyxy == (2, 1, 5, 4)
    assert result.mask.dtype == np.bool_
    assert result.mask.flags.writeable is False
    np.testing.assert_array_equal(
        localizer.segment_point(image, pixel_uv=(3, 2)), result.mask
    )


@pytest.mark.parametrize("failure", ["prompt_miss", "invalid_shape"])
def test_shared_localizer_distinguishes_prompt_miss_from_invalid_shape(failure) -> None:
    masks = np.zeros((2, 10, 12), dtype=bool)
    masks[0, 2:4, 2:4] = True
    masks[1] = True
    scores = np.array([.95, .08])
    if failure == "invalid_shape":
        masks, scores = np.ones((1, 2, 2), dtype=bool), np.array([1.])
    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started, localizer._closed = True, False
    localizer._predictor = SimpleNamespace(
        set_image=lambda image: None,
        predict=lambda **kwargs: (masks, scores, None),
    )
    expected = Sam2PointMaskUnavailable if failure == "prompt_miss" else RuntimeError
    message = "requested pixel" if failure == "prompt_miss" else "invalid point mask shape"
    with pytest.raises(expected, match=message) as error:
        localizer.segment_point(Image.new("RGB", (12, 10)), pixel_uv=(6, 6))
    assert type(error.value) is expected


def test_shared_localizer_rejects_non_finite_point_score() -> None:
    class _Predictor:
        def set_image(self, _image):
            return None

        def predict(self, **_kwargs):
            masks = np.zeros((1, 3, 4), dtype=bool)
            masks[0, 1, 1] = True
            return masks, np.asarray([np.nan]), None

    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started = True
    localizer._closed = False
    localizer._predictor = _Predictor()

    with pytest.raises(RuntimeError, match="non-finite point-mask scores"):
        localizer.segment(
            Image.new("RGB", (4, 3), "white"),
            language_prompt=None,
            pixel_uv=(1, 1),
            min_score=0.0,
        )


@pytest.mark.parametrize("invalid_value", [-1.0, 0.5, 2.0, np.nan, np.inf, -np.inf])
def test_shared_localizer_rejects_non_binary_point_mask(invalid_value) -> None:
    class _Predictor:
        def set_image(self, _image):
            return None

        def predict(self, **_kwargs):
            masks = np.zeros((1, 3, 4), dtype=np.float32)
            masks[0, 1, 1] = 1.0
            masks[0, 0, 0] = invalid_value
            return masks, np.asarray([0.9]), None

    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started = True
    localizer._closed = False
    localizer._predictor = _Predictor()

    with pytest.raises(RuntimeError, match="non-binary mask"):
        localizer.segment(
            Image.new("RGB", (4, 3), "white"),
            language_prompt=None,
            pixel_uv=(1, 1),
            min_score=0.0,
        )


@pytest.mark.parametrize("mask_dtype", [np.bool_, np.float32, np.float64, np.uint8])
def test_shared_localizer_text_segmentation_combines_grounding_and_sam_scores(mask_dtype) -> None:
    class _Predictor:
        def set_image(self, image):
            assert image.shape == (6, 8, 3)

        def predict(self, **kwargs):
            np.testing.assert_allclose(kwargs["box"], [2.0, 1.5, 6.0, 4.5])
            assert kwargs.get("return_logits", False) is False
            masks = np.zeros((2, 6, 8), dtype=mask_dtype)
            masks[0, 2:4, 3:6] = True
            masks[1, 1:5, 2:7] = True
            return masks, np.asarray([0.4, 0.7]), None

    def grounding_predict(**kwargs):
        assert kwargs["caption"] == "red cube"
        return (
            np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
            np.asarray([0.8], dtype=np.float32),
            ["red cube"],
        )

    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started = True
    localizer._closed = False
    localizer._predictor = _Predictor()
    localizer._grounding_transform = lambda image, _target: (image, None)
    localizer._grounding_predict = grounding_predict
    localizer.box_threshold = 0.35
    localizer.text_threshold = 0.25
    localizer.device = "cuda"

    result = localizer.segment(
        Image.new("RGB", (8, 6), "white"),
        language_prompt="red cube",
        pixel_uv=None,
        min_score=0.65,
    )

    assert result is not None
    assert result.score == pytest.approx(0.7)
    assert result.label == "red cube"
    assert result.box_xyxy == (2, 1, 7, 5)
    assert result.mask.dtype == np.bool_
    assert result.mask.flags.writeable is False


def test_shared_localizer_rejects_non_finite_text_score() -> None:
    class _Predictor:
        def set_image(self, _image):
            return None

        def predict(self, **_kwargs):
            masks = np.ones((1, 3, 4), dtype=bool)
            return masks, np.asarray([0.9]), None

    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started = True
    localizer._closed = False
    localizer._predictor = _Predictor()
    localizer._grounding_transform = lambda image, _target: (image, None)
    localizer._grounding_predict = lambda **_kwargs: (
        np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
        np.asarray([np.nan], dtype=np.float32),
        ["cube"],
    )
    localizer.box_threshold = 0.35
    localizer.text_threshold = 0.25
    localizer.device = "cpu"

    with pytest.raises(RuntimeError, match="Grounding DINO.*non-finite"):
        localizer.segment(
            Image.new("RGB", (4, 3), "white"),
            language_prompt="cube",
            pixel_uv=None,
            min_score=0.0,
        )


@pytest.mark.parametrize("invalid_value", [-1.0, 0.5, 2.0, np.nan, np.inf, -np.inf])
def test_shared_localizer_rejects_non_binary_text_mask(invalid_value) -> None:
    class _Predictor:
        def set_image(self, _image):
            return None

        def predict(self, **_kwargs):
            masks = np.ones((1, 3, 4), dtype=np.float32)
            masks[0, 0, 0] = invalid_value
            return masks, np.asarray([0.9]), None

    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._lock = threading.RLock()
    localizer._started = True
    localizer._closed = False
    localizer._predictor = _Predictor()
    localizer._grounding_transform = lambda image, _target: (image, None)
    localizer._grounding_predict = lambda **_kwargs: (
        np.asarray([[0.5, 0.5, 0.5, 0.5]], dtype=np.float32),
        np.asarray([0.9], dtype=np.float32),
        ["cube"],
    )
    localizer.box_threshold = 0.35
    localizer.text_threshold = 0.25
    localizer.device = "cpu"

    with pytest.raises(RuntimeError, match="non-binary mask"):
        localizer.segment(
            Image.new("RGB", (4, 3), "white"),
            language_prompt="cube",
            pixel_uv=None,
            min_score=0.0,
        )


def _batch_resources(version="2"):
    class Segmenter(_FakeSegmenter):
        def segment(self, image, **kwargs):
            if kwargs["language_prompt"] == "absent":
                self.images.append(image.size)
                return None
            return super().segment(image, **kwargs)

    options = {"target_kind": "sim", "current_mode": "sim_active", "source_id_by_role": {"main": "front"},
        "model_profile_id": "sam2:test", "model_profile_hash": "sam2-model:test",
        "inference_implementation_hash": "sam2-localizer:test", "runtime_session_id": "segment:test",
        "control_domain_id": "segment:test", "target_binding_id": "target:test",
        "station_manifest_hash": "station:test", "execution_manifest_hash": "execution:test"}
    original = build_segmentation_registration(**options)
    registration = build_segmentation_registration(**options, skill_version=version)
    capture = LiveRgbSegmentationCapture(source_id="front", role="main", captured_at=10.,
        station_revision=4, image=Image.new("RGB", (64, 48), "white"), rgb_content_hash="a" * 64)
    calls = []

    def read(**kwargs):
        calls.append(kwargs)
        return capture

    model = Segmenter()
    model.start()
    provider = LiveSegmentationProvider(localizer=model, registration=original,
        capture_port=SimpleNamespace(capture_segment_rgb=read))
    return SimpleNamespace(registration=registration, provider=provider, capture=capture, calls=calls, model=model, options=options)


def _batch_parameters(*prompts):
    return {"role": "main", "targets": [
        {"language_prompt": prompt, "target_pixel_uv": None, "min_score": .5} for prompt in prompts]}


def _geometry_provenance():
    return {"rgb_capture_operation_id": "rgb:test", "rgb_capture_request_hash": "b" * 64,
        "input_rgb_artifact_id": "artifact:rgb:test", "input_rgb_frame_ordinal": 0}


@pytest.mark.parametrize("changes", [
    {"model_profile_hash": "other"}, {"inference_implementation_hash": "other"},
    {"source_id_by_role": {"main": "other"}}, {"runtime_session_id": "other"},
    {"control_domain_id": "other"}, {"target_binding_id": "other"},
    {"station_manifest_hash": "other"}, {"execution_manifest_hash": "other"},
])
def test_independent_batch_entry_rejects_a_different_shared_provider_before_work(changes):
    case = _batch_resources()
    other = build_segmentation_registration(**{**case.options, **changes}, skill_version="2")
    with pytest.raises(ValueError, match="differ"):
        prepare_segmentation_batch(registration=other, provider=case.provider, capture=case.capture,
            parameters=_batch_parameters("cube"), check_active=lambda: None)
    assert not case.model.images and not case.calls


@pytest.mark.parametrize("prompts", [("cube",), ("cube", "absent", "cup"), ("absent", "absent")])
def test_batch_preserves_order_and_explicit_no_mask_without_relabelling(prompts):
    case = _batch_resources()
    parameters = _batch_parameters(*prompts)
    prepared = prepare_segmentation_batch(registration=case.registration, provider=case.provider,
        capture=case.capture, parameters=parameters, check_active=lambda: None)
    result = prepared.execution_result
    succeeded = sum(prompt != "absent" for prompt in prompts)
    assert result["completed_count"] == succeeded and result["no_mask_count"] == len(prompts) - succeeded
    assert len(case.model.images) == len(prompts) and not case.calls
    assert len(prepared.masks_npy) == succeeded
    assert [item["index"] for item in result["items"]] == list(range(len(prompts)))
    assert [item["language_prompt"] for item in result["items"]] == list(prompts)
    assert [item["mask_artifact_ordinal"] for item in result["items"] if item["mask"] is not None] == list(range(succeeded))
    assert all(item["mask"] is None and item["mask_artifact_ordinal"] is None
        for item in result["items"] if item["status"] == "no_mask")
    image = Image.open(io.BytesIO(prepared.overlay_jpeg))
    assert image.size == (64, 48) and image.format == "JPEG"
    assert result["input_rgb_content_hash"] == case.capture.rgb_content_hash
    assert case.model.started and not case.model.closed


@pytest.mark.parametrize("changes", [
    {"targets": []}, {"targets": _batch_parameters("a", "b", "c", "d")["targets"]},
    {"targets": [{"language_prompt": "cube", "target_pixel_uv": [1, 1], "min_score": .5}]},
    {"targets": [{"language_prompt": None, "target_pixel_uv": None, "min_score": .5}]},
    {"targets": [{"language_prompt": "cube", "target_pixel_uv": None, "min_score": True}]},
    {"role": "unregistered"},
])
def test_batch_invalid_parameters_rejected_before_inference(changes):
    case = _batch_resources()
    parameters = {**_batch_parameters("cube"), **changes}
    with pytest.raises(ValueError):
        prepare_segmentation_batch(registration=case.registration, provider=case.provider,
            capture=case.capture, parameters=parameters, check_active=lambda: None)
    assert not case.model.images


def test_batch_all_pixels_checked_before_any_inference_and_camera_identity_pinned():
    case = _batch_resources()
    parameters = _batch_parameters("cube", "cup")
    parameters["targets"][1].update(language_prompt=None, target_pixel_uv=[64, 1])
    with pytest.raises(ValueError, match="outside"):
        prepare_segmentation_batch(registration=case.registration, provider=case.provider,
            capture=case.capture, parameters=parameters, check_active=lambda: None)
    assert not case.model.images
    with pytest.raises(ValueError, match="camera identity"):
        prepare_segmentation_batch(registration=case.registration, provider=case.provider,
            capture=replace(case.capture, source_id="other"), parameters=_batch_parameters("cube"), check_active=lambda: None)


def test_batch_model_exception_is_not_a_no_mask_and_cannot_continue():
    case = _batch_resources()
    case.model.on_segment = lambda: (_ for _ in ()).throw(RuntimeError("inference failed"))
    with pytest.raises(RuntimeError, match="inference failed"):
        prepare_segmentation_batch(registration=case.registration, provider=case.provider,
            capture=case.capture, parameters=_batch_parameters("cube", "cup"), check_active=lambda: None)
    assert len(case.model.images) == 1


@pytest.mark.parametrize("version", ["2", "3"])
def test_batch_targets_receive_copies_of_one_cut_even_if_model_mutates_its_input(version):
    parameters = _batch_parameters("cube", "cup", "bottle")
    baseline = _batch_resources(version)
    unchanged = _prepare_geometry(baseline, parameters)
    case = _batch_resources(version)
    original = case.model.segment
    original_rgb = case.capture.image.tobytes()
    received_rgb = []

    def segment(image, **kwargs):
        received_rgb.append(image.tobytes())
        result = original(image, **kwargs)
        image.paste((0, 0, 0), (0, 0, image.width, image.height))
        return result

    case.model.segment = segment
    prepared = _prepare_geometry(case, parameters)
    assert received_rgb == [original_rgb] * 3
    assert case.capture.image.tobytes() == original_rgb
    assert prepared.overlay_jpeg == unchanged.overlay_jpeg
    assert prepared.execution_result == unchanged.execution_result


@pytest.mark.parametrize("field,value", [
    ("source_id", "other"), ("input_rgb_content_hash", "bad"), ("image_width", 0),
    ("model_profile_hash", "other"), ("completed_count", True), ("station_revision", -1),
])
def test_batch_no_mask_still_validates_source_and_counts(field, value):
    case = _batch_resources()
    prepared = prepare_segmentation_batch(registration=case.registration, provider=case.provider,
        capture=case.capture, parameters=_batch_parameters("absent"), check_active=lambda: None)
    body = deepcopy(prepared.execution_result)
    body[field] = value
    with pytest.raises(ValueError):
        case.registration.payload_validator.validate_execution_result(body)


@pytest.fixture
def batch_service(tmp_path, request):
    from test_agent_gateway_service import (
        _AuthoritativeStationPort,
        _ManualClock,
        _policy,
        _snapshot,
    )

    from agent_gateway.protocol.session_contracts import AgentGatewaySessionCapabilities
    from agent_gateway.protocol.skill_catalog import (
        SkillRouteManifest,
        SkillRouteManifestEntry,
    )
    from agent_gateway.protocol.skill_registration import (
        build_skill_registration_catalogs,
    )
    from agent_gateway.runtime.artifact_transfer import AgentArtifactTransferBroker
    from agent_gateway.runtime.event_journal import BoundedAgentGatewayEventJournal
    from agent_gateway.runtime.executor_registry import SkillExecutorRegistry
    from agent_gateway.runtime.payload_validation import (
        AgentSkillPayloadValidatorRegistry,
    )
    from agent_gateway.runtime.protocol_runtime import AgentGatewayProtocolRuntime
    from agent_gateway.runtime.service import AgentGatewayService

    options = getattr(request, "param", "2")
    if not isinstance(options, dict):
        options = {"version": options}
    case = _batch_resources(options["version"])
    geometry = options["version"] == "3"
    size = options.get("size", (64, 48))
    case.capture = replace(case.capture, image=Image.new("RGB", size, "white"))
    registration = case.registration
    binding = registration.binding
    rgb = None
    if geometry:
        from agent_gateway.profiles.mujoco.rgb_deployment import (
            build_mujoco_rgb_capture_registration, build_mujoco_rgb_capture_registration_v2,
        )
        builder = build_mujoco_rgb_capture_registration_v2 if options.get("rgb_version") == "2" else build_mujoco_rgb_capture_registration
        rgb = builder(source_catalog=tuple({"source_id": source, "role": role,
            "native_width": size[0], "native_height": size[1]}
            for source, role in (("front", "main"), ("left_wrist", "wrist_left"))),
            **{key: case.options[key] for key in ("runtime_session_id", "control_domain_id",
                "target_binding_id", "station_manifest_hash", "execution_manifest_hash")})
    catalogs = build_skill_registration_catalogs((registration.skill, *((rgb.registration,) if rgb else ())))
    manifest = SkillRouteManifest(runtime_session_id=binding.runtime_session_id,
        control_domain_id=binding.control_domain_id, station_manifest_hash=binding.station_manifest_hash,
        execution_manifest_hash=binding.execution_manifest_hash,
        skill_definition_catalog_hash=catalogs.definition_catalog.semantic_content_hash,
        skill_schema_catalog_hash=catalogs.schema_catalog.semantic_content_hash,
        skill_io_catalog_hash=catalogs.io_catalog.semantic_content_hash,
        target_binding_id=binding.target_binding_id, target_kind="sim", current_mode="sim_active",
        supported_transport_features=("idempotency_v1", "poll_status_v1"),
        skill_routes=(SkillRouteManifestEntry(route=registration.skill.routes[0], deployment_binding=binding.to_ref(),
            evaluation_plan_hash=None, availability_hash=registration.availability_hash),
            *((SkillRouteManifestEntry(route=rgb.registration.routes[0], deployment_binding=rgb.deployment_binding.to_ref(),
                evaluation_plan_hash=None, availability_hash=rgb.availability_hash),) if rgb else ())), explicit_exclusions=("motion",))
    store = RecoverableAgentArtifactStore(root=tmp_path / "batch-cas", quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024)
    clock = _ManualClock()
    capabilities = AgentGatewaySessionCapabilities(runtime_session_id=manifest.runtime_session_id,
        artifact_transfer_enabled=True, artifact_access_mechanisms=("chunked_rpc",),
        max_control_request_bytes=131072, max_control_response_bytes=131072,
        max_artifact_bytes=store.max_artifact_bytes, max_artifact_chunk_bytes=1024,
        max_artifact_refs_per_operation=4, max_concurrent_uploads=1, max_concurrent_downloads=1,
        artifact_store_quota_bytes=store.quota_bytes, event_payload_max_bytes=65536, event_retention_count=64)
    runtime = AgentGatewayProtocolRuntime(manifest=manifest, io_catalog=catalogs.io_catalog,
        session_capabilities=capabilities,
        event_journal=BoundedAgentGatewayEventJournal(runtime_session_id=manifest.runtime_session_id,
            max_events=64, event_payload_max_bytes=65536), artifact_store=store,
        artifact_transfer_broker=AgentArtifactTransferBroker(store=store, session_capabilities=capabilities,
            max_chunk_bytes=1024), clock=clock)
    case.revision = 4
    def lookup(operation_id):
        return service.operation_store.request_for_operation(operation_id), service.get_operation(operation_id)

    executor = build_segmentation_batch_executor(registration=registration, provider=case.provider,
        artifact_store=store, runtime_session_id=manifest.runtime_session_id,
        current_station_revision=lambda: case.revision, clock=clock,
        **({"rgb_capture_registration": rgb.registration, "rgb_capture_validator": rgb.payload_validator,
            "lookup_operation": lookup} if rgb else {}))
    rgb_executor = None
    if rgb:
        from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
        from agent_gateway.runtime.artifact_producer import artifact_ref_for_bytes, commit_artifact_bytes
        from agent_gateway.runtime.passive_read_executor import PassiveReadResult, PassiveReadSkillExecutor
        from dataset_contracts.perception_artifacts import (
            RGB_FRAME_SCHEMA_ID_V1, RGB_FRAME_ENCODING_V1, RGB_FRAME_COMPRESSION_V1,
        )
        stream = io.BytesIO()
        case.capture.image.save(stream, format="JPEG")
        data = stream.getvalue()
        case.capture = replace(case.capture, rgb_content_hash=hashlib.sha256(data).hexdigest())

        def read_rgb(original, operation):
            artifact = artifact_ref_for_bytes(data, media_type="image/jpeg", schema_id=RGB_FRAME_SCHEMA_ID_V1,
                encoding=RGB_FRAME_ENCODING_V1, compression=RGB_FRAME_COMPRESSION_V1)
            commit_artifact_bytes(store=store, data=data, artifact=artifact, producer_id="rgb:test",
                runtime_session_id=manifest.runtime_session_id, idempotency_key="rgb:test", now=clock())
            frames = [{"ordinal": ordinal, "source_id": source, "role": role, "media_type": "image/jpeg",
                "width": size[0], "height": size[1], "media_content_hash": artifact.sha256,
                **({"acquired_at": 10.} if options.get("rgb_version") == "2" else {})}
                for ordinal, (source, role) in enumerate((("left_wrist", "wrist_left"), ("front", "main")))]
            case.rgb_data = data
            return PassiveReadResult(execution_result={"snapshot_id": "rgb:test", "station_revision": 4,
                **({"capture_window_started_at": 10., "capture_window_ended_at": 10.} if options.get("rgb_version") == "2"
                    else {"captured_at": 10.}), "frames": frames, "total_bytes": 2 * len(data)},
                output_artifact_bindings=tuple(AgentArtifactBinding(slot_id="rgb_frames", ordinal=ordinal,
                    artifact=artifact, production_receipt_ref=f"rgb:test:{ordinal}") for ordinal in range(2)),
                evidence_refs=("rgb:test",), reason_code="rgb_test_capture_completed")

        rgb_executor = PassiveReadSkillExecutor(port_id="rgb:test", executor_id=rgb.deployment_binding.executor_id,
            read=read_rgb, current_station_revision=lambda: case.revision, clock=clock)
    domain = _AuthoritativeStationPort(clock)
    from agent_gateway.runtime.executor_registry import SkillExecutorBinding
    registry = SkillExecutorRegistry(manifest=manifest, bindings=tuple(SkillExecutorBinding.from_manifest_entry(
        manifest=manifest, entry=entry, execution_port=executor if entry.route.skill_id == "perception.segment" else rgb_executor)
        for entry in manifest.skill_routes))
    service = AgentGatewayService(manifest=manifest, access_policy=_policy(mode="sim_active"),
        executor_registry=registry,
        domain_control_port=domain, payload_validators=AgentSkillPayloadValidatorRegistry(registry_id="batch:test",
            validators=(registration.payload_validator, *((rgb.payload_validator,) if rgb else ()))), protocol_runtime=runtime, clock=clock)
    assert service.publish_observation(snapshot=_snapshot(manifest)).accepted
    case.service, case.executor, case.manifest, case.clock, case.domain = service, executor, manifest, clock, domain
    if rgb:
        from test_agent_gateway_service import _request
        source_request = replace(_request(manifest, rgb.registration.routes[0], sequence=0, key="rgb:source", request_id="rgb:source"),
            parameters={"selected_roles": ["wrist_left", "main"], "width": size[0], "height": size[1], "max_total_bytes": 16 * 1024 * 1024})
        admitted = service.submit_request(request=source_request, current_station_revision=4)
        assert admitted.accepted
        assert service.launch_operation(admitted.operation.operation_id).accepted
        assert rgb_executor.wait(admitted.operation.operation_id, timeout_sec=2.)
        case.rgb_operation = service.get_operation(admitted.operation.operation_id)
        assert case.rgb_operation.status == "completed", case.rgb_operation.diagnostic
        case.rgb_binding = AgentArtifactBinding(slot_id="input_image", ordinal=0,
            artifact=case.rgb_operation.output_artifact_bindings[1].artifact)
        case.rgb_registration, case.rgb_source_request, case.store = rgb, source_request, store
    try:
        yield case
    finally:
        assert executor.wait_all(timeout_sec=2.)
        if rgb_executor:
            assert rgb_executor.wait_all(timeout_sec=2.)
        store.close()
        case.model.close()


def _launch_batch(case, parameters):
    from test_agent_gateway_service import _request

    request = replace(_request(case.manifest, case.registration.skill.routes[0]), parameters=parameters)
    if case.registration.skill.definition.skill_version == "3":
        request = replace(request, parameters={"rgb_capture_operation_id": case.rgb_operation.operation_id, **parameters},
            input_artifact_bindings=(case.rgb_binding,))
    admitted = case.service.submit_request(request=request, current_station_revision=4)
    assert admitted.accepted, admitted.reason_code
    case.operation_id = admitted.operation.operation_id
    assert case.service.launch_operation(case.operation_id).accepted
    assert case.executor.wait(case.operation_id, timeout_sec=2.)
    return case.service.get_operation(case.operation_id)


@pytest.mark.parametrize("prompts", [("cube", "absent", "cup"), ("absent", "absent", "absent"), ("cube", "cube")])
@pytest.mark.parametrize("batch_service", ["2", "3"], indirect=True)
def test_batch_real_passive_registry_commits_one_cut_and_dense_artifacts(batch_service, prompts):
    case = batch_service
    operation = _launch_batch(case, _batch_parameters(*prompts))
    assert operation.status == "completed", (operation.progress_summary, case.executor.failure_diagnostics)
    assert len(case.calls) == (0 if hasattr(case, "rgb_operation") else 1)
    assert len(case.model.images) == len(prompts)
    assert not case.domain.calls
    masks = [binding for binding in operation.output_artifact_bindings if binding.slot_id == "segmentation_mask"]
    overlays = [binding for binding in operation.output_artifact_bindings if binding.slot_id == "segmentation_overlay"]
    result = operation.execution_result
    assert len(masks) == result["completed_count"] and len(overlays) == 1
    assert [binding.ordinal for binding in masks] == list(range(len(masks)))
    assert overlays[0].artifact.sha256 == result["overlay_media_content_hash"]
    assert [binding.artifact.sha256 for binding in masks] == [
        item["mask"]["mask_media_content_hash"] for item in result["items"] if item["mask"] is not None]
    if case.registration.skill.definition.skill_version == "3":
        assert [tuple(row["indices"]) for row in result["relations"]] == [
            (one, two) for one in range(len(prompts)) for two in range(one + 1, len(prompts))]


@pytest.mark.parametrize("action", ["cancel", "expire", "revision"])
@pytest.mark.parametrize("batch_service", ["2", "3"], indirect=True)
def test_batch_checks_cancel_deadline_and_revision_before_next_model_call(batch_service, action):
    case = batch_service

    def interrupt():
        if action == "cancel":
            outcome = case.service.cancel_operation(operation_id=case.operation_id, reason="test cancel", current_station_revision=4)
            assert outcome.accepted
        elif action == "expire":
            case.clock.set(11.)
        else:
            case.revision = 5

    case.model.on_segment = interrupt
    operation = _launch_batch(case, _batch_parameters("cube", "cup", "bottle"))
    assert operation.status != "completed"
    assert len(case.model.images) == 1 and len(case.calls) == (0 if hasattr(case, "rgb_operation") else 1)
    assert operation.execution_result is None and not operation.output_artifact_bindings
    assert not case.domain.calls


@pytest.mark.parametrize("action", ["cancel", "expire"])
@pytest.mark.parametrize("batch_service", ["2", "3"], indirect=True)
def test_batch_last_artifact_commit_cannot_publish_late_success(batch_service, monkeypatch, action):
    from agent_gateway.profiles import segmentation_batch

    case = batch_service
    original = segmentation_batch.commit_artifact_bytes
    committed = []

    def commit(**kwargs):
        artifact = original(**kwargs)
        committed.append(artifact)
        if artifact.schema_id == "dexteleop.perception.segmentation_overlay.v1":
            if action == "expire":
                case.clock.set(11.)
            else:
                assert case.service.cancel_operation(operation_id=case.operation_id,
                    reason="cancel after commit", current_station_revision=4).accepted
        return artifact

    monkeypatch.setattr(segmentation_batch, "commit_artifact_bytes", commit)
    operation = _launch_batch(case, _batch_parameters("cube", "cup"))
    assert len(committed) == 3 and len(case.calls) == (0 if hasattr(case, "rgb_operation") else 1)
    assert operation.status != "completed" and operation.execution_result is None
    assert not operation.output_artifact_bindings and not case.domain.calls


def _geometry_case(masks):
    case = _batch_resources("3")
    shape = next((mask.shape for mask in masks if mask is not None), (48, 64))
    case.capture = replace(case.capture, image=Image.new("RGB", (shape[1], shape[0]), "white"))

    def segment(image, *, language_prompt, **kwargs):
        case.model.images.append(image.size)
        mask = masks[int(language_prompt)]
        if mask is None:
            return None
        rows, cols = np.nonzero(mask)
        return GroundedObjectSegment(mask=mask, score=.95, label=language_prompt,
            box_xyxy=(int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1))

    case.model.segment = segment
    return case, _batch_parameters(*(str(index) for index in range(len(masks))))


def _prepare_geometry(case, parameters, check_active=lambda: None):
    geometry = case.registration.skill.definition.skill_version == "3"
    return prepare_segmentation_batch(registration=case.registration, provider=case.provider,
        capture=case.capture, parameters={**parameters, "rgb_capture_operation_id": "rgb:test"} if geometry else parameters,
        rgb_provenance=_geometry_provenance() if geometry else None, check_active=check_active)


@pytest.mark.parametrize("shape", ["full", "hole", "border", "single", "line", "overlap"])
def test_contour_preserves_non_edge_pixels_and_source_arrays(monkeypatch, shape):
    rng = np.random.default_rng(24)
    source = rng.integers(0, 256, (23, 31, 3), dtype=np.uint8)
    first = np.zeros(source.shape[:2], dtype=bool)
    if shape == "full":
        first[:] = True
    elif shape == "hole":
        first[2:20, 2:28] = True
        first[7:14, 9:17] = False
    elif shape == "border":
        first[:12, :16] = True
    elif shape == "single":
        first[4, 20] = True
    elif shape == "line":
        first[10, 2:28] = True
    else:
        first[4:16, 6:21] = True
    masks = [first, None, first.copy() if shape == "overlap" else None]
    originals = [None if m is None else m.copy() for m in masks]
    # Test edge colors independently of the separately tested label drawing.
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", lambda *_args, **_kwargs: None)
    expected = source.copy()
    for index, mask in enumerate(masks):
        if mask is None:
            continue
        for row, col in zip(*np.nonzero(mask)):
            occupied = [0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x]
                for y in range(row - 1, row + 2) for x in range(col - 1, col + 2)]
            if not all(occupied):
                expected[row, col] = _COLORS[index]
    image = Image.fromarray(source)
    overlay = _indexed_contour_overlay(image, masks, check_active=lambda: None)
    np.testing.assert_array_equal(np.asarray(overlay), expected)
    np.testing.assert_array_equal(np.asarray(image), source)
    for before, after in zip(originals, masks):
        if before is not None:
            np.testing.assert_array_equal(before, after)


def test_contour_labels_use_original_indices_and_occupied_anchors(monkeypatch):
    masks = [np.zeros((48, 64), dtype=bool) for _ in range(2)]
    masks[0][20:32, 20:32] = True
    masks[1][35:45, 42:61] = True
    calls = []
    original = ImageDraw.ImageDraw.text

    def record(self, xy, text, **kwargs):
        calls.append((xy, text, kwargs["fill"]))
        return original(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    result = _indexed_contour_overlay(Image.new("RGB", (64, 48), "white"),
        [masks[0], None, masks[1]], check_active=lambda: None)
    assert [(text, color) for _, text, color in calls] == [("0", _COLORS[0]), ("2", _COLORS[2])]
    for ((x, y), text, _), mask in zip(calls, masks):
        left, top, right, bottom = ImageDraw.Draw(result).textbbox((x, y), text)
        assert 0 <= left < right <= 64 and 0 <= top < bottom <= 48
        assert mask[top, left]


def test_contour_all_absent_has_no_annotations():
    source = Image.new("RGB", (64, 48), (12, 23, 34))
    result = _indexed_contour_overlay(source, [None, None, None], check_active=lambda: None)
    assert result is not source and result.tobytes() == source.tobytes()


@pytest.mark.parametrize("identical", [True, False])
def test_contour_shared_anchor_keeps_both_target_labels_readable(monkeypatch, identical):
    first = np.zeros((48, 64), dtype=bool)
    first[8:32, 14:40] = True
    second = first.copy()
    if not identical:
        second[16:28, 20:32] = False
    calls = []
    original = ImageDraw.ImageDraw.text

    def record(self, xy, text, **kwargs):
        calls.append((self.textbbox(xy, text), text, kwargs["fill"]))
        return original(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    _indexed_contour_overlay(Image.new("RGB", (64, 48), "white"),
        [first, None, second], check_active=lambda: None)
    assert [(text, color) for _, text, color in calls] == [("0", _COLORS[0]), ("2", _COLORS[2])]
    first_box, second_box = calls[0][0], calls[1][0]
    assert first_box[2] < second_box[0]
    assert 0 <= first_box[0] and second_box[2] <= 64
    assert first_box[1] == second_box[1] == 8


@pytest.mark.parametrize("cancel_at", [1, 3, 7, 12])
def test_contour_rechecks_cancellation_during_drawing(cancel_at):
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        if calls == cancel_at:
            raise RuntimeError("test cancelled")

    with pytest.raises(RuntimeError, match="test cancelled"):
        _indexed_contour_overlay(Image.new("RGB", (32, 24)),
            [np.ones((24, 32), dtype=bool)] * 3, check_active=check)
    assert calls == cancel_at


@pytest.mark.parametrize("absent", [False, True])
def test_geometry_contour_keeps_masks_scores_grid_and_input_unchanged(absent):
    first = np.zeros((48, 64), dtype=bool)
    first[16:40, 16:40] = True
    masks = [None, None, None] if absent else [first, None, first.copy()]
    case, parameters = _geometry_case(masks)
    original = case.capture.image.tobytes()
    prepared = _prepare_geometry(case, parameters)
    result = prepared.execution_result
    assert case.capture.image.tobytes() == original
    assert [item["index"] for item in result["items"]] == [0, 1, 2]
    assert len(prepared.masks_npy) == (0 if absent else 2)
    for data in prepared.masks_npy:
        np.testing.assert_array_equal(np.load(io.BytesIO(data), allow_pickle=False), first)
    assert all(item["mask"]["score"] == .95 for item in result["items"] if item["mask"])
    assert result["relations"][1]["status"] == ("unavailable" if absent else "overlap")
    decoded = Image.open(io.BytesIO(prepared.overlay_jpeg))
    assert decoded.format == "JPEG" and decoded.size == (64, 48) and decoded.mode == "RGB"
    assert result["overlay_media_content_hash"] == hashlib.sha256(prepared.overlay_jpeg).hexdigest()
    expected = io.BytesIO()
    _indexed_contour_overlay(case.capture.image, masks, check_active=lambda: None).save(expected, format="JPEG", quality=90)
    assert expected.getvalue() == prepared.overlay_jpeg


def test_v2_keeps_legacy_overlay_bytes_and_model_identity():
    case = _batch_resources("2")
    prepared = _prepare_geometry(case, _batch_parameters("cube", "absent", "cup"))
    image = case.capture.image
    pixels = np.asarray(image).copy()
    for item, data in zip([i for i in prepared.execution_result["items"] if i["mask"]], prepared.masks_npy):
        mask = np.load(io.BytesIO(data), allow_pickle=False)
        pixels[mask] = (3 * pixels[mask].astype(np.uint16) + 2 * np.asarray(_COLORS[item["index"]], dtype=np.uint16)) // 5
    legacy = Image.fromarray(pixels)
    draw = ImageDraw.Draw(legacy)
    for item in prepared.execution_result["items"]:
        color = _COLORS[item["index"]]
        if item["mask"]:
            x0, y0, x1, y1 = item["mask"]["mask_bounds_xyxy"]
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=color, width=2)
            draw.text((x0, min(legacy.height - 1, y0 + 12 * item["index"])), str(item["index"]), fill=color)
        y = 12 * item["index"]
        draw.rectangle((0, y, min(legacy.width - 1, 150), y + 11), fill="black")
        draw.text((1, y), f"[{item['index']}] {item['status']}", fill=color)
    stream = io.BytesIO()
    legacy.save(stream, format="JPEG", quality=90)
    assert stream.getvalue() == prepared.overlay_jpeg
    v3 = _batch_resources("3").registration
    for field in ("model_profile_id", "model_profile_hash", "inference_implementation_hash"):
        assert getattr(v3.profile, field) == getattr(case.registration.profile, field)
    assert v3.profile.profile_id != case.registration.profile.profile_id


def test_geometry_exact_distances_and_deterministic_unordered_pairs():
    masks = [np.zeros((10, 12), dtype=bool) for _ in range(3)]
    masks[0][2, 2] = masks[1][6, 5] = masks[2][2, 2] = True
    case, parameters = _geometry_case(masks)
    result = _prepare_geometry(case, parameters).execution_result
    relations = result["relations"]
    assert [row["indices"] for row in relations] == [[0, 1], [0, 2], [1, 2]]
    assert [row["distance_px"] for row in relations] == [5., 0., 5.]
    assert [row["status"] for row in relations] == ["separated", "overlap", "separated"]
    assert relations[0]["closest_pixels_uv"] == [[2, 2], [5, 6]]
    assert relations[1]["closest_pixels_uv"] == [[2, 2], [2, 2]]
    assert all(row["metric"] == "euclidean_distance_between_occupied_pixel_centers"
        and row["semantic_identity_verified"] is False and row["physical_clearance_m"] is None
        for row in relations)
    assert result["input_rgb_content_hash"] == case.capture.rgb_content_hash
    assert (result["source_id"], result["image_width"], result["image_height"]) == ("front", 12, 10)
    again = _prepare_geometry(case, parameters)
    assert again.execution_result == result
    reverse, reverse_parameters = _geometry_case(masks[:2][::-1])
    assert _prepare_geometry(reverse, reverse_parameters).execution_result["relations"][0]["distance_px"] == 5.


def test_geometry_resolves_equal_distance_endpoints_in_row_major_order():
    first, second = (np.zeros((6, 7), dtype=bool) for _ in range(2))
    first[2, 2] = first[3, 3] = True
    second[1, 2] = second[2, 1] = second[4, 3] = True
    case, parameters = _geometry_case([first, second])
    relation, = _prepare_geometry(case, parameters).execution_result["relations"]
    assert relation["distance_px"] == 1.
    assert relation["closest_pixels_uv"] == [[2, 2], [2, 1]]


def test_geometry_sparse_and_dense_masks_match_a_brute_force_oracle():
    rng = np.random.default_rng(41)
    for density in (.1, .5, 1.):
        first, second = (rng.random((9, 11)) < density for _ in range(2))
        first[0, 0] = second[-1, -1] = True
        rows_one, cols_one = np.nonzero(first)
        rows_two, cols_two = np.nonzero(second)
        squared = ((rows_one[:, None] - rows_two) ** 2 + (cols_one[:, None] - cols_two) ** 2)
        one, two = np.unravel_index(np.argmin(squared), squared.shape)
        case, parameters = _geometry_case([first, second])
        relation, = _prepare_geometry(case, parameters).execution_result["relations"]
        assert relation["distance_px"] == float(np.sqrt(squared[one, two]))
        assert relation["closest_pixels_uv"] == [
            [int(cols_one[one]), int(rows_one[one])], [int(cols_two[two]), int(rows_two[two])]]


@pytest.mark.parametrize("prompts", [("cube",), ("cube", "absent", "cup"), ("absent", "absent")])
def test_geometry_absence_is_unavailable_not_zero_and_single_target_has_no_pairs(prompts):
    case = _batch_resources("3")
    parameters = _batch_parameters(*prompts)
    result = _prepare_geometry(case, parameters).execution_result
    assert len(result["relations"]) == len(prompts) * (len(prompts) - 1) // 2
    for row in result["relations"]:
        if any(prompts[index] == "absent" for index in row["indices"]):
            assert row["status"] == "unavailable" and row["unavailable_reason"] == "absent_mask"
            assert row["distance_px"] is None and row["closest_pixels_uv"] is None
        else:
            assert row["status"] == "overlap" and row["distance_px"] == 0.


@pytest.mark.parametrize("field,value", [
    ("source_id", "another"), ("role", "other"), ("captured_at", 11.),
    ("station_revision", 5), ("input_rgb_content_hash", "b" * 64),
    ("image_width", 63), ("image_height", 47),
])
def test_geometry_rejects_provider_result_from_another_cut_before_computation(monkeypatch, field, value):
    case = _batch_resources("3")
    original = case.provider.try_segment_capture

    def changed(**kwargs):
        prepared = original(**kwargs)
        return replace(prepared, execution_result={**prepared.execution_result, field: value})

    monkeypatch.setattr(case.provider, "try_segment_capture", changed)
    monkeypatch.setattr("scipy.ndimage.distance_transform_edt", lambda *_: pytest.fail("geometry before identity validation"))
    with pytest.raises((ValueError, RuntimeError)):
        _prepare_geometry(case, _batch_parameters("cube", "cup"))
    assert len(case.model.images) == 1


@pytest.mark.parametrize("corruption", ["hash", "dtype", "shape", "count", "bounds", "non_binary", "trailing"])
def test_geometry_revalidates_actual_provider_npy_before_computation(monkeypatch, corruption):
    case = _batch_resources("3")
    original = case.provider.try_segment_capture

    def changed(**kwargs):
        prepared = original(**kwargs)
        result, data = dict(prepared.execution_result), prepared.mask_npy
        mask = np.load(io.BytesIO(data), allow_pickle=False)
        if corruption in {"dtype", "shape"}:
            stream = io.BytesIO()
            np.save(stream, mask.astype(np.uint8) if corruption == "dtype" else mask[:, :-1], allow_pickle=False)
            data = stream.getvalue()
        elif corruption == "count":
            result["mask_pixel_count"] -= 1
        elif corruption == "bounds":
            result["mask_bounds_xyxy"][0] += 1
        elif corruption == "non_binary":
            data = data[:-1] + b"\x02"
        elif corruption == "trailing":
            data += b"extra"
        if corruption != "hash":
            result["mask_media_content_hash"] = hashlib.sha256(data).hexdigest()
        else:
            result["mask_media_content_hash"] = "0" * 64
        return replace(prepared, execution_result=result, mask_npy=data)

    monkeypatch.setattr(case.provider, "try_segment_capture", changed)
    monkeypatch.setattr("scipy.ndimage.distance_transform_edt", lambda *_: pytest.fail("geometry before NPY validation"))
    with pytest.raises((RuntimeError, ValueError), match="mask|Mask|NPY"):
        _prepare_geometry(case, _batch_parameters("cube", "cup"))
    assert len(case.model.images) == 1


@pytest.mark.parametrize("size", [(1921, 1), (1, 1081)])
def test_geometry_grid_cap_rejects_before_inference(size):
    case = _batch_resources("3")
    case.capture = replace(case.capture, image=Image.new("RGB", size))
    with pytest.raises(ValueError, match="grid exceeds"):
        _prepare_geometry(case, _batch_parameters("cube", "cup"))
    assert not case.model.images


@pytest.mark.parametrize("field,value", [
    ("indices", [1, 0]), ("metric", "bbox_distance"),
    ("status", "separated"), ("unavailable_reason", "absent_mask"),
    ("distance_px", float("nan")), ("distance_px", True),
    ("distance_px", 1.), ("closest_pixels_uv", None),
    ("closest_pixels_uv", [[100, 0], [100, 0]]),
    ("semantic_identity_verified", True), ("semantic_identity_verified", 0),
    ("physical_clearance_m", 0.),
])
def test_geometry_result_validator_rejects_false_claims_and_invalid_endpoints(field, value):
    case = _batch_resources("3")
    result = deepcopy(_prepare_geometry(case, _batch_parameters("cube", "cup")).execution_result)
    result["relations"][0][field] = value
    with pytest.raises(ValueError):
        case.registration.payload_validator.validate_execution_result(result)


@pytest.mark.parametrize("relations", [[], [{"indices": [0, 1]}] * 2])
def test_geometry_result_validator_requires_complete_pairs(relations):
    case = _batch_resources("3")
    result = _prepare_geometry(case, _batch_parameters("cube", "cup")).execution_result
    with pytest.raises(ValueError):
        case.registration.payload_validator.validate_execution_result({**result, "relations": relations})


@pytest.mark.parametrize("version,prompts", [("2", ("cube", "cup")), ("3", ("cube",)), ("3", ("absent", "cube"))])
def test_legacy_or_unavailable_geometry_does_not_import_scipy(monkeypatch, version, prompts):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "scipy" or name.startswith("scipy."):
            pytest.fail("SciPy must remain lazy")
        return original(name, *args, **kwargs)

    case = _batch_resources(version)
    monkeypatch.setattr(builtins, "__import__", guarded)
    _prepare_geometry(case, _batch_parameters(*prompts))


@pytest.mark.parametrize("action", ["cancel", "expire", "revision"])
@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
def test_geometry_rechecks_active_after_edt_before_any_artifact_commit(batch_service, monkeypatch, action):
    import scipy.ndimage

    original = scipy.ndimage.distance_transform_edt
    case = batch_service
    calls = []

    def interrupted(*args, **kwargs):
        distances = original(*args, **kwargs)
        calls.append(True)
        if action == "cancel":
            assert case.service.cancel_operation(operation_id=case.operation_id,
                reason="cancel in geometry", current_station_revision=4).accepted
        elif action == "expire":
            case.clock.set(11.)
        else:
            case.revision = 5
        return distances

    monkeypatch.setattr(scipy.ndimage, "distance_transform_edt", interrupted)
    monkeypatch.setattr("agent_gateway.profiles.segmentation_batch.commit_artifact_bytes",
        lambda **_: pytest.fail("commit after interruption"))
    operation = _launch_batch(case, _batch_parameters("cube", "cup", "bottle"))
    assert calls == [True]
    assert operation.status != "completed" and operation.execution_result is None
    assert not operation.output_artifact_bindings and not case.domain.calls


def test_segmentation_registration_is_passive_and_artifact_bearing(tmp_path) -> None:
    registration = build_segmentation_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_id_by_role={"main": "front"},
        model_profile_id="sam2:test",
        model_profile_hash="sam2-model:test",
        inference_implementation_hash="sam2-localizer:test",
        runtime_session_id="segment:test",
        control_domain_id="segment:test",
        target_binding_id="target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )

    route = registration.skill.routes[0]
    assert route.control_effect_class == "read_only"
    assert route.may_move_robot is False
    assert route.requires_task_evaluation is False
    assert [
        slot.slot_id for slot in registration.skill.io_contract.output_artifact_slots
    ] == [
        SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
        SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
    ]
    with pytest.raises(TypeError):
        registration.profile.source_id_by_role["other"] = "other-camera"
    with pytest.raises(ValueError, match="deployment contracts are inconsistent"):
        replace(
            registration,
            binding=replace(registration.binding, profile_hash="tampered-profile"),
        )
    with pytest.raises(ValueError, match="another Skill contract"):
        replace(
            registration,
            skill=replace(
                registration.skill,
                routes=(
                    replace(
                        registration.skill.routes[0],
                        max_operation_duration_ms=59_999,
                    ),
                ),
            ),
        )

    wrong_model = _FakeSegmenter()
    wrong_model.model_spec = SimpleNamespace(
        model_profile_id="another-model",
        semantic_content_hash="another-model-hash",
    )
    with pytest.raises(ValueError, match="model identity differs"):
        LiveSegmentationProvider(
            localizer=wrong_model,
            capture_port=SimpleNamespace(capture_segment_rgb=lambda **_kwargs: None),
            registration=registration,
        )

    provider = LiveSegmentationProvider(
        localizer=_FakeSegmenter(),
        capture_port=SimpleNamespace(capture_segment_rgb=lambda **_kwargs: None),
        registration=registration,
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "registration-artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    try:
        with pytest.raises(ValueError, match="runtime session differs"):
            build_segmentation_executor(
                registration=registration,
                provider=provider,
                artifact_store=store,
                runtime_session_id="another-session",
                current_station_revision=lambda: 0,
                max_retained_operations=1,
            )
        equivalent_registration = build_segmentation_registration(
            target_kind="sim",
            current_mode="sim_active",
            source_id_by_role={"main": "front"},
            model_profile_id="sam2:test",
            model_profile_hash="sam2-model:test",
            inference_implementation_hash="sam2-localizer:test",
            runtime_session_id="segment:test",
            control_domain_id="segment:test",
            target_binding_id="target:test",
            station_manifest_hash="station:test",
            execution_manifest_hash="execution:test",
        )
        with pytest.raises(ValueError, match="registrations differ"):
            build_segmentation_executor(
                registration=equivalent_registration,
                provider=provider,
                artifact_store=store,
                runtime_session_id="segment:test",
                current_station_revision=lambda: 0,
                max_retained_operations=1,
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    ("capture_role", "capture_revision", "message"),
    (
        ("other", 4, "another camera role"),
        ("main", 5, "another Station revision"),
    ),
)
def test_segmentation_provider_rejects_capture_identity_drift(
    capture_role, capture_revision, message
) -> None:
    registration = build_segmentation_registration(
        target_kind="real",
        current_mode="real_observe",
        source_id_by_role={"main": "front", "other": "side"},
        model_profile_id="sam2:test",
        model_profile_hash="sam2-model:test",
        inference_implementation_hash="sam2-localizer:test",
        runtime_session_id="segment:capture-drift",
        control_domain_id="segment:capture-drift",
        target_binding_id="target:capture-drift",
        station_manifest_hash="station:capture-drift",
        execution_manifest_hash="execution:capture-drift",
    )

    class _CapturePort:
        @staticmethod
        def capture_segment_rgb(*, role, expected_station_revision):
            assert role == "main"
            assert expected_station_revision == 4
            return LiveRgbSegmentationCapture(
                source_id={"main": "front", "other": "side"}[capture_role],
                role=capture_role,
                captured_at=1.0,
                station_revision=capture_revision,
                image=Image.new("RGB", (4, 3), "white"),
                rgb_content_hash="0" * 64,
            )

    localizer = _FakeSegmenter()
    localizer.started = True
    provider = LiveSegmentationProvider(
        localizer=localizer,
        capture_port=_CapturePort(),
        registration=registration,
    )
    request = SimpleNamespace(
        expected_station_revision=4,
        parameters={
            "role": "main",
            "language_prompt": None,
            "target_pixel_uv": [1, 1],
            "min_score": 0.5,
        },
    )

    with pytest.raises(RuntimeError, match=message):
        provider.segment_request(request)


def test_segmentation_provider_recomputes_mask_bounds() -> None:
    registration = build_segmentation_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_id_by_role={"main": "front"},
        model_profile_id="sam2:test",
        model_profile_hash="sam2-model:test",
        inference_implementation_hash="sam2-localizer:test",
        runtime_session_id="segment:bounds",
        control_domain_id="segment:bounds",
        target_binding_id="target:bounds",
        station_manifest_hash="station:bounds",
        execution_manifest_hash="execution:bounds",
    )

    class _CapturePort:
        @staticmethod
        def capture_segment_rgb(*, role, expected_station_revision):
            return LiveRgbSegmentationCapture(
                source_id="front",
                role=role,
                captured_at=1.0,
                station_revision=expected_station_revision,
                image=Image.new("RGB", (4, 3), "white"),
                rgb_content_hash="0" * 64,
            )

    class _InconsistentLocalizer(_FakeSegmenter):
        def segment(self, image, *, language_prompt, pixel_uv, min_score):
            mask = np.zeros((3, 4), dtype=bool)
            mask[1, 1:3] = True
            return SimpleNamespace(
                mask=mask,
                score=0.9,
                label=None,
                box_xyxy=(0, 0, 4, 3),
            )

    localizer = _InconsistentLocalizer()
    provider = LiveSegmentationProvider(
        localizer=localizer,
        capture_port=_CapturePort(),
        registration=registration,
    )

    with pytest.raises(RuntimeError, match="bounds differ from its mask"):
        provider.segment_request(
            SimpleNamespace(
                expected_station_revision=4,
                parameters={
                    "role": "main",
                    "language_prompt": None,
                    "target_pixel_uv": [1, 1],
                    "min_score": 0.5,
                },
            )
        )


@pytest.mark.parametrize(
    ("corruption", "message"),
    (
        ("empty-mask", "not NPY v1"),
        ("mask-float-shape", "dtype/order/shape changed"),
        ("mask-payload-mismatch", "count differs from metadata"),
        ("empty-overlay", "invalid JPEG"),
        ("overlay-size-mismatch", "identity differs from metadata"),
        ("overlay-mode-mismatch", "identity differs from metadata"),
    ),
)
def test_segmentation_provider_rejects_corrupt_encoded_artifacts(
    corruption, message
) -> None:
    registration = build_segmentation_registration(
        target_kind="sim",
        current_mode="sim_active",
        source_id_by_role={"main": "front"},
        model_profile_id="sam2:test",
        model_profile_hash="sam2-model:test",
        inference_implementation_hash="sam2-localizer:test",
        runtime_session_id="segment:corrupt-artifact",
        control_domain_id="segment:corrupt-artifact",
        target_binding_id="target:corrupt-artifact",
        station_manifest_hash="station:corrupt-artifact",
        execution_manifest_hash="execution:corrupt-artifact",
    )

    class _CapturePort:
        @staticmethod
        def capture_segment_rgb(*, role, expected_station_revision):
            return LiveRgbSegmentationCapture(
                source_id="front",
                role=role,
                captured_at=1.0,
                station_revision=expected_station_revision,
                image=Image.new("RGB", (4, 3), "white"),
                rgb_content_hash="0" * 64,
            )

    class _CorruptEncodingLocalizer(_FakeSegmenter):
        def encode_segment_artifacts(self, image, segment):
            encoded = encode_grounded_object_segment_artifacts(image, segment)
            if corruption == "empty-mask":
                return replace(encoded, mask_npy=b"")
            if corruption == "mask-float-shape":
                return replace(
                    encoded,
                    mask_npy=encoded.mask_npy.replace(b"(3, 4)", b"(3.,4)", 1),
                )
            if corruption == "mask-payload-mismatch":
                mask_npy = bytearray(encoded.mask_npy)
                mask_npy[-1] = 0
                return replace(encoded, mask_npy=bytes(mask_npy))
            if corruption == "empty-overlay":
                return replace(encoded, overlay_jpeg=b"")
            overlay = io.BytesIO()
            if corruption == "overlay-size-mismatch":
                Image.new("RGB", (1, 1), "white").save(overlay, format="JPEG")
            else:
                Image.new("L", (4, 3), 255).save(overlay, format="JPEG")
            return replace(encoded, overlay_jpeg=overlay.getvalue())

    localizer = _CorruptEncodingLocalizer()
    localizer.started = True
    provider = LiveSegmentationProvider(
        localizer=localizer,
        capture_port=_CapturePort(),
        registration=registration,
    )

    with pytest.raises(RuntimeError, match=message):
        provider.segment_request(
            SimpleNamespace(
                expected_station_revision=4,
                parameters={
                    "role": "main",
                    "language_prompt": None,
                    "target_pixel_uv": [1, 1],
                    "min_score": 0.5,
                },
            )
        )


@pytest.mark.parametrize("backend", ["fixture", "sam3-service"])
def test_real_gateway_segments_live_realsense_frame_and_commits_artifacts(
    monkeypatch, tmp_path, backend
) -> None:
    monkeypatch.setattr(
        realsense_perception,
        "start_realsense_sources",
        lambda _sources: None,
    )
    monkeypatch.setattr(
        realsense_perception,
        "admit_realsense_sources",
        lambda _sources: None,
    )
    monkeypatch.setattr(
        realsense_perception,
        "_formal_close_camera_sources",
        lambda _sources: None,
    )
    segmenter = _FakeSegmenter()
    if backend == "sam3-service":
        from deployments.perception.sam3_service import Sam3ServiceLocalizer

        segmenter = Sam3ServiceLocalizer()
        segmenter.images, segmenter.on_segment = [], None
        original_segment = segmenter.segment

        def infer(method, kwargs):
            if method == "healthz":
                return {"status": "ok"}
            image = Image.open(io.BytesIO(base64.b64decode(kwargs["image_base64"])))
            stream = io.BytesIO()
            Image.new("L", image.size, 255).save(stream, format="PNG")
            return {"found": True, "score": .95,
                    "mask_shape": [image.height, image.width],
                    "mask_png_base64": base64.b64encode(stream.getvalue()).decode()}

        def segment_and_observe(image, **kwargs):
            result = original_segment(image, **kwargs)
            segmenter.images.append(image.size)
            if segmenter.on_segment is not None:
                segmenter.on_segment()
            return result

        monkeypatch.setattr(segmenter, "_call", infer)
        monkeypatch.setattr(segmenter, "segment", segment_and_observe)
    camera = _RealCameraSource()
    source_binding = RealPerceptionSourceBinding(
        camera_sources={"front": camera},
        world_from_depth_camera_by_source={"front": np.eye(4)},
        semantic_content_hash="source-binding:segmentation-real-test",
    )
    composition = assemble_camera_only_real_perception_composition(
        source_binding=source_binding,
        socket_path=tmp_path / "real-segment.sock",
        artifact_root=tmp_path,
        client_bindings=compile_agent_gateway_unix_client_bindings(
            primary_client_id="test:real-segmentation",
            primary_client_uid=1000,
        ),
        runtime_session_id="test:real-segmentation",
        control_domain_id="test:real-segmentation",
        frame_binding=build_real_perception_station_frame_binding(
            collision_scene_profile="segmentation-real-test",
            scene_mount_hash="scene-mounts:segmentation-real-test",
            scene_world_from_canonical_station=np.eye(4),
        ),
        object_localizer=segmenter,
        clock=lambda: 10.0,
    )
    try:
        with pytest.raises(
            ValueError,
            match="reuse the lifecycle-owned localization model",
        ):
            assemble_real_perception_gateway_session(
                bundle=composition.contract_bundle,
                source_binding=source_binding,
                artifact_store=composition.artifact_store,
                allowed_client_ids=("test:real-segmentation",),
                localization_provider=composition.localization_provider,
                segmentation_localizer=_FakeSegmenter(),
                clock=lambda: 10.0,
            )
        composition.resources.start()
        published = composition.session.observation_producer.capture_and_publish(
            composition.session.service
        )
        assert published.accepted
        adapter = composition.session.wire_adapter(
            authenticated_client_id="test:real-segmentation"
        )
        observation = adapter.get_observation_snapshot()
        manifest = composition.contract_bundle.route_manifest
        entry = manifest.route("perception.segment", "1")
        route = entry.route
        profile = route.allowed_parameter_profiles[0]
        request = AgentSkillRequest(
            client_id="test:real-segmentation",
            request_id="request:real-segmentation:1",
            idempotency_key="idempotency:real-segmentation:1",
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
                "language_prompt": "red region",
                "target_pixel_uv": None,
                "min_score": 0.5,
            },
            observation_snapshot_id=str(observation["snapshot_id"]),
            requested_mode="real_observe",
            requested_timeout_ms=60_000,
        )
        submitted = adapter.submit_read_only_skill_request(request.to_wire())
        assert submitted["accepted"] is True
        operation_id = submitted["operation"]["operation_id"]
        assert composition.session.segmentation_executor is not None
        assert composition.session.segmentation_executor.wait(
            operation_id, timeout_sec=2.0
        )
        operation = composition.session.service.operation_store.get(operation_id)
        assert operation is not None
        assert operation.status == "completed", operation.reason_code
        assert operation.execution_result["mode"] == "text"
        assert operation.execution_result["source_id"] == "front"
        assert operation.execution_result["role"] == "main"
        assert [binding.slot_id for binding in operation.output_artifact_bindings] == [
            SEGMENT_MASK_OUTPUT_SLOT_ID_V1,
            SEGMENT_OVERLAY_OUTPUT_SLOT_ID_V1,
        ]
        assert segmenter.images == [(4, 3)]

        def advance_revision_after_inference() -> None:
            segmenter.on_segment = None
            composition.session.observation_producer.revision_tracker.observe(
                {"forced_test_revision": "after_inference"}
            )

        segmenter.on_segment = advance_revision_after_inference
        stale_after_inference = replace(
            request,
            request_id="request:real-segmentation:2",
            idempotency_key="idempotency:real-segmentation:2",
            client_sequence=2,
        )
        submitted = adapter.submit_read_only_skill_request(
            stale_after_inference.to_wire()
        )
        stale_operation_id = submitted["operation"]["operation_id"]
        assert composition.session.segmentation_executor.wait(
            stale_operation_id, timeout_sec=2.0
        )
        stale_operation = composition.session.service.operation_store.get(
            stale_operation_id
        )
        assert stale_operation is not None
        assert stale_operation.status == "failed"
        assert stale_operation.output_artifact_bindings == ()
        assert any(
            "before segmentation Artifact commit" in item["message"]
            for item in composition.session.segmentation_executor.failure_diagnostics
            if item["operation_id"] == stale_operation_id
        )

        advanced = composition.session.observation_producer.capture_and_publish(
            composition.session.service
        )
        assert advanced.accepted
        current_observation = adapter.get_observation_snapshot()
        original_commit = segmentation_profile.commit_artifact_bytes
        advanced_during_commit = False

        def commit_and_advance_revision(**kwargs):
            nonlocal advanced_during_commit
            committed = original_commit(**kwargs)
            if not advanced_during_commit:
                advanced_during_commit = True
                composition.session.observation_producer.revision_tracker.observe(
                    {"forced_test_revision": "during_commit"}
                )
            return committed

        monkeypatch.setattr(
            segmentation_profile,
            "commit_artifact_bytes",
            commit_and_advance_revision,
        )
        stale_during_commit = replace(
            request,
            request_id="request:real-segmentation:3",
            idempotency_key="idempotency:real-segmentation:3",
            client_sequence=3,
            expected_station_revision=int(current_observation["station_revision"]),
            observation_snapshot_id=str(current_observation["snapshot_id"]),
        )
        submitted = adapter.submit_read_only_skill_request(
            stale_during_commit.to_wire()
        )
        stale_commit_operation_id = submitted["operation"]["operation_id"]
        assert composition.session.segmentation_executor.wait(
            stale_commit_operation_id, timeout_sec=2.0
        )
        stale_commit_operation = composition.session.service.operation_store.get(
            stale_commit_operation_id
        )
        assert stale_commit_operation is not None
        assert stale_commit_operation.status == "failed"
        assert stale_commit_operation.output_artifact_bindings == ()
        assert any(
            "during segmentation Artifact commit" in item["message"]
            for item in composition.session.segmentation_executor.failure_diagnostics
            if item["operation_id"] == stale_commit_operation_id
        )
    finally:
        composition.application.close()
    if backend == "fixture":
        assert segmenter.closed is True
    else:
        assert not segmenter.started
        with pytest.raises(RuntimeError, match="cannot restart"):
            segmenter.start()

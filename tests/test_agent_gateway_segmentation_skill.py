from __future__ import annotations

import base64
import io
import threading
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

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

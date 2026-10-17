from __future__ import annotations

import hashlib
import io
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from agent_gateway.profiles.real.localization_deployment import (
    REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1,
    REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1,
    build_real_object_localization_registration,
)
from agent_gateway.profiles.real.localization_gateway import ObjectLocalizationProvider
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from agent_gateway.runtime.artifact_producer import (
    artifact_ref_for_bytes,
    commit_artifact_bytes,
)
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore
from dataset_contracts.perception_artifacts import (
    LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
    LOCALIZATION_INPUT_IMAGE_SCHEMA_ID_V1,
    LOCALIZATION_INPUT_IMAGE_SCHEMA_IDS_V1,
    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
    RGB_FRAME_SCHEMA_ID_V1,
)
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectDetection,
    Sam2GroundingDinoLocalizer,
    _DeviceBoundGroundingPredictor,
    _mask_box,
    Sam2GroundingDinoModelSpec,
)


class _FakeSpec:
    model_profile_id = "sam2-grounding-dino:test"
    semantic_content_hash = "model-profile:test"


class _FakeLocalizer:
    model_spec = _FakeSpec()
    semantic_content_hash = "localizer:test"

    def __init__(self) -> None:
        self.started = False
        self.closed = False
        self.calls = []

    def start(self) -> None:
        self.started = True

    def close(self) -> None:
        self.closed = True

    def localize(self, image, *, language_prompt, max_detections):
        self.calls.append((image.size, language_prompt, max_detections))
        return (
            GroundedObjectDetection(
                box_xyxy=(1, 2, 8, 9),
                score=0.875,
                label=language_prompt,
            ),
        )


def _registration(*, local_path_enabled: bool):
    return build_real_object_localization_registration(
        model_profile_id=_FakeSpec.model_profile_id,
        model_profile_hash=_FakeSpec.semantic_content_hash,
        inference_implementation_hash=_FakeLocalizer.semantic_content_hash,
        local_path_enabled=local_path_enabled,
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        target_binding_id="target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )


def _jpeg_bytes() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (12, 10), color=(32, 64, 96)).save(output, format="JPEG")
    return output.getvalue()


def _store(root: Path) -> RecoverableAgentArtifactStore:
    return RecoverableAgentArtifactStore(
        root=root,
        quota_bytes=64 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024,
    )


def test_localization_registration_closes_over_input_and_output_artifacts() -> None:
    registration = _registration(local_path_enabled=False)
    route = registration.skill.routes[0]

    assert registration.skill.definition.skill_id == "perception.localize_objects"
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert route.may_move_robot is False
    assert route.supported_modes == ("real_observe",)
    assert registration.skill.io_contract.input_artifact_slots[0].slot_id == (
        REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1
    )
    assert registration.skill.io_contract.input_artifact_slots[0].min_count == 1
    assert (
        registration.skill.io_contract.input_artifact_slots[0].allowed_schema_ids
        == LOCALIZATION_INPUT_IMAGE_SCHEMA_IDS_V1
    )
    assert registration.skill.io_contract.output_artifact_slots[0].slot_id == (
        REAL_LOCALIZE_OBJECTS_OUTPUT_SLOT_ID_V1
    )
    parameter_properties = registration.skill.schema_entry.parameter_schema[
        "properties"
    ]
    result_properties = registration.skill.schema_entry.execution_result_schema[
        "properties"
    ]
    assert parameter_properties["input_source"] == {"const": "artifact"}
    assert parameter_properties["local_image_path"] == {"type": "null"}
    assert result_properties["input_source"] == {"const": "artifact"}

    registration.payload_validator.validate_parameters(
        {
            "input_source": "artifact",
            "local_image_path": None,
            "language_prompt": "apple",
            "max_detections": 20,
        }
    )
    with pytest.raises(ValueError, match="disabled"):
        registration.payload_validator.validate_parameters(
            {
                "input_source": "local_path",
                "local_image_path": "/tmp/frame.png",
                "language_prompt": None,
                "max_detections": 20,
            }
        )


def test_local_path_mode_is_bounded_to_explicit_test_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    image_path = allowed / "frame.jpg"
    image_path.write_bytes(_jpeg_bytes())
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg_bytes())
    registration = _registration(local_path_enabled=True)
    local_source_schema = registration.skill.schema_entry.parameter_schema[
        "properties"
    ]["input_source"]
    assert local_source_schema["type"] == "string"
    assert local_source_schema["enum"] == ("artifact", "local_path")
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(
        localizer=localizer,
        registration=registration,
        allowed_local_image_roots=(allowed,),
    )
    request = SimpleNamespace(
        parameters={
            "input_source": "local_path",
            "local_image_path": str(image_path),
            "language_prompt": None,
            "max_detections": 5,
        },
        input_artifact_bindings=(),
    )

    prepared = provider.localize_request(request, artifact_store=SimpleNamespace())

    assert prepared.execution_result["mode"] == "automatic"
    assert prepared.execution_result["detections"] == [
        {
            "ordinal": 0,
            "box_xyxy": [1, 2, 8, 9],
            "score": 0.875,
            "label": None,
        }
    ]
    assert Image.open(io.BytesIO(prepared.annotated_jpeg)).size == (12, 10)
    registration.payload_validator.validate_execution_result(prepared.execution_result)

    request.parameters["local_image_path"] = str(outside)
    with pytest.raises(PermissionError, match="outside"):
        provider.localize_request(request, artifact_store=SimpleNamespace())


@pytest.mark.parametrize(
    "input_schema_id",
    (LOCALIZATION_INPUT_IMAGE_SCHEMA_ID_V1, RGB_FRAME_SCHEMA_ID_V1),
    ids=("localization-input", "captured-rgb-frame"),
)
def test_artifact_mode_reads_committed_bytes_and_preserves_prompt(
    tmp_path: Path,
    input_schema_id: str,
) -> None:
    data = _jpeg_bytes()
    registration = _registration(local_path_enabled=False)
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(
        localizer=localizer,
        registration=registration,
    )
    with _store(tmp_path / "store") as store:
        artifact = artifact_ref_for_bytes(
            data,
            media_type=LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
            schema_id=input_schema_id,
            encoding=LOCALIZATION_INPUT_JPEG_ENCODING_V1,
            compression=LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
        )
        committed = commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id="test-client",
            runtime_session_id="session:test",
            idempotency_key=f"input:{input_schema_id}",
            now=1.0,
        )
        request = SimpleNamespace(
            parameters={
                "input_source": "artifact",
                "local_image_path": None,
                "language_prompt": "apple",
                "max_detections": 10,
            },
            input_artifact_bindings=(
                AgentArtifactBinding(
                    slot_id=REAL_LOCALIZE_OBJECTS_INPUT_SLOT_ID_V1,
                    ordinal=0,
                    artifact=committed,
                ),
            ),
        )

        prepared = provider.localize_request(request, artifact_store=store)

    assert (
        prepared.execution_result["input_media_content_hash"]
        == hashlib.sha256(data).hexdigest()
    )
    assert prepared.execution_result["mode"] == "text"
    assert prepared.execution_result["language_prompt"] == "apple"
    assert prepared.execution_result["detections"][0]["label"] == "apple"
    registration.payload_validator.validate_execution_result(prepared.execution_result)
    registration.payload_validator.validate_execution_result_for_parameters(
        request.parameters,
        prepared.execution_result,
    )


def test_model_asset_manifest_rejects_checkpoint_hash_drift(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sam = bundle / "sam.pt"
    grounding = bundle / "grounding.pth"
    sam.write_bytes(b"sam")
    grounding.write_bytes(b"grounding")
    payload = {
        "schema_version": "dexteleop.perception_model_assets.v1",
        "model_profile_id": "test-model:v1",
        "sources": {
            "sam2": {
                "git_url": "https://github.com/facebookresearch/sam2.git",
                "revision": "aa9b8722d0585b661ded4b3dff1bd103540554ae",
            },
            "grounding_dino": {
                "git_url": "https://github.com/IDEA-Research/GroundingDINO.git",
                "revision": "856dde20aee659246248e20734ef9ba5214f5e44",
            },
        },
        "checkpoints": {
            "sam2": {
                "relative_path": sam.name,
                "sha256": hashlib.sha256(sam.read_bytes()).hexdigest(),
                "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            },
            "grounding_dino": {
                "relative_path": grounding.name,
                "sha256": hashlib.sha256(grounding.read_bytes()).hexdigest(),
                "model_config": "GroundingDINO_SwinT_OGC.py",
            },
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    spec = Sam2GroundingDinoModelSpec.from_asset_bundle(bundle)
    assert spec.model_profile_id == "test-model:v1"
    assert spec.sam2_checkpoint_sha256 == payload["checkpoints"]["sam2"]["sha256"]

    sam.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        Sam2GroundingDinoModelSpec.from_asset_bundle(bundle)


def test_model_asset_manifest_rejects_unreviewed_source_revision(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    sam = bundle / "sam.pt"
    grounding = bundle / "grounding.pth"
    sam.write_bytes(b"sam")
    grounding.write_bytes(b"grounding")
    payload = {
        "schema_version": "dexteleop.perception_model_assets.v1",
        "model_profile_id": "test-model:v1",
        "sources": {
            "sam2": {
                "git_url": "https://github.com/facebookresearch/sam2.git",
                "revision": "1" * 40,
            },
            "grounding_dino": {
                "git_url": "https://github.com/IDEA-Research/GroundingDINO.git",
                "revision": "856dde20aee659246248e20734ef9ba5214f5e44",
            },
        },
        "checkpoints": {
            "sam2": {
                "relative_path": sam.name,
                "sha256": hashlib.sha256(sam.read_bytes()).hexdigest(),
                "model_config": "configs/sam2.1/sam2.1_hiera_l.yaml",
            },
            "grounding_dino": {
                "relative_path": grounding.name,
                "sha256": hashlib.sha256(grounding.read_bytes()).hexdigest(),
                "model_config": "GroundingDINO_SwinT_OGC.py",
            },
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SAM2 source revision is not the reviewed"):
        Sam2GroundingDinoModelSpec.from_asset_bundle(bundle)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (4, 0), (1, 1), (4, 5), (480, 640)])
@pytest.mark.parametrize("density", [0., .01, .5, 1.])
def test_mask_box_matches_all_foreground_coordinates_without_expanding_them(shape, density, monkeypatch):
    rng = np.random.default_rng(17)
    foreground = rng.random(shape) < density
    rows, columns = np.nonzero(foreground)
    expected = ((int(columns.min()), int(rows.min()), int(columns.max())+1, int(rows.max())+1)
                if len(columns) else None)

    original_nonzero = np.nonzero
    def reject_full_coordinates(array):
        if np.asarray(array).ndim != 1:
            raise AssertionError("Do not expand the full mask into per-pixel coordinates")
        return original_nonzero(array)

    monkeypatch.setattr(np, "nonzero", reject_full_coordinates)
    assert _mask_box(foreground) == expected


@pytest.mark.parametrize("layout", ["C", "F", "transpose", "reverse", "strided"])
def test_mask_box_keeps_half_open_edges_for_numeric_and_noncontiguous_masks(layout):
    mask = np.zeros((9, 11), np.float32)
    mask[0, 0], mask[6, 10], mask[8, 3] = -1., .5, np.nan
    if layout in ("C", "F"):
        mask = np.array(mask, order=layout)
    elif layout == "transpose":
        mask = mask.T
    elif layout == "reverse":
        mask = mask[::-1, ::-1]
    else:
        mask = mask[::2, ::2]
    before = mask.copy()
    rows, columns = np.nonzero(mask.astype(bool))
    assert _mask_box(mask) == (int(columns.min()), int(rows.min()), int(columns.max())+1, int(rows.max())+1)
    np.testing.assert_array_equal(mask, before)


@pytest.mark.parametrize("shape", [(), (4,), (1, 3, 4)])
def test_mask_box_does_not_silently_accept_invalid_model_dimensions(shape):
    with pytest.raises(ValueError):
        _mask_box(np.ones(shape))


def _point_localizer():
    class Predictor:
        def __init__(self):
            self.encodes = 0
            self.prompts = []
            self.fail_next = False

        def set_image(self, image):
            self.encodes += 1
            if self.fail_next:
                self.fail_next = False
                raise RuntimeError("encoder_failed")
            self.shape = image.shape[:2]

        def predict(self, *, point_coords=None, box=None, **_):
            point = point_coords[0] if point_coords is not None else (box[:2] + box[2:]) / 2
            u, v = map(int, point)
            self.prompts.append((u, v))
            mask = np.zeros(self.shape, dtype=bool)
            mask[v, u] = True
            return mask[None], np.array([1.]), None

    # No checkpoint, CUDA, model import or device construction is performed.
    localizer = object.__new__(Sam2GroundingDinoLocalizer)
    localizer._lock = threading.RLock()
    localizer._started, localizer._closed = True, False
    localizer._encoded_image = None
    localizer._grounding_input = None
    localizer._predictor = Predictor()
    return localizer


class _DeviceModel:
    def __init__(self, physical_device="cpu"):
        self.physical_device = physical_device
        self.moves = []
        self.fail_move = False

    def to(self, device):
        self.moves.append(device)
        if self.fail_move:
            raise RuntimeError("device_placement_failed")
        return self

    def parameters(self):
        yield SimpleNamespace(device=self.physical_device)


@pytest.mark.parametrize("configured,physical", [("cpu", "cpu"), ("cuda", "cuda:1"), ("cuda:0", "cuda:0")])
def test_grounding_places_once_on_first_text_call_and_keeps_resolved_device(configured, physical):
    model = _DeviceModel(physical)
    calls = []

    def official_predict(*, model, device, **parameters):
        calls.append((model.to(device), device, parameters))
        return parameters["caption"]

    bound = _DeviceBoundGroundingPredictor(model, official_predict, configured)
    assert model.moves == []
    with pytest.raises(ValueError, match="bound to its session device"):
        bound.to(physical)
    image = object()
    for caption in ("blue cube", "red cube", "absent object"):
        assert bound(image=image, caption=caption, box_threshold=.35, text_threshold=.25) == caption
    assert model.moves == [configured]
    assert [item[1] for item in calls] == [physical]*3
    assert all(item[0] is model and item[2]["image"] is image for item in calls)
    assert all(item[2]["box_threshold"] == .35 and item[2]["text_threshold"] == .25 for item in calls)
    with pytest.raises(ValueError, match="bound to its session device"):
        bound.to("another_device")
    assert model.moves == [configured]


def test_grounding_device_placement_failure_is_not_hidden_or_retried():
    model = _DeviceModel()
    model.fail_move = True
    calls = []
    bound = _DeviceBoundGroundingPredictor(model, lambda **params: calls.append(params), "cpu")
    with pytest.raises(RuntimeError, match="device_placement_failed"):
        bound(image=object(), caption="blue cube", box_threshold=.35, text_threshold=.25)
    assert model.moves == ["cpu"]
    assert calls == []
    with pytest.raises(ValueError, match="bound to its session device"):
        bound.to("cpu")


def test_grounding_inference_failure_is_not_a_reason_to_recreate_or_move_model():
    model = _DeviceModel()
    calls = []
    def failed_predict(*, model, device, **parameters):
        model.to(device)
        calls.append(parameters)
        raise RuntimeError("inference_failed")

    bound = _DeviceBoundGroundingPredictor(model, failed_predict, "cpu")
    for _ in range(2):
        with pytest.raises(RuntimeError, match="inference_failed"):
            bound(image=object(), caption="blue cube", box_threshold=.35, text_threshold=.25)
    assert model.moves == ["cpu"]
    assert len(calls) == 2


def test_grounding_model_has_one_lifecycle_owner_and_is_released_on_close():
    import weakref

    localizer = _point_localizer()
    model = _DeviceModel()
    retained = weakref.ref(model)
    localizer._grounding_predict = _DeviceBoundGroundingPredictor(model, lambda **_: (), "cpu")
    del model
    assert retained() is not None
    localizer.close()
    assert localizer._grounding_predict is None
    assert retained() is None


def test_same_rgb_reuses_encoder_but_every_point_gets_a_fresh_mask():
    localizer = _point_localizer()
    image = Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8))
    first = localizer.segment_point(image, pixel_uv=(1, 1))
    second = localizer.segment_point(image.copy(), pixel_uv=(3, 2))
    assert localizer._predictor.encodes == 1
    assert localizer._predictor.prompts == [(1, 1), (3, 2)]
    assert first[1, 1] and not first[2, 3]
    assert second[2, 3] and not second[1, 1]


def test_grounded_box_then_point_reuses_only_the_same_rgb_encoding():
    localizer = _point_localizer()
    localizer._grounding_transform = lambda image, target: (image, target)
    localizer._grounding_predict = lambda **_: (
        np.array([[.5, .5, .4, .5]]), np.array([.9]), ["blue cube"])
    localizer.box_threshold = localizer.text_threshold = localizer.nms_iou_threshold = .5
    localizer.device = "cpu"
    image = Image.new("RGB", (5, 4))
    detected = localizer.localize(image, language_prompt="blue cube", max_detections=1)
    clicked = localizer.segment_point(image.copy(), pixel_uv=(1, 1))
    assert detected[0].box_xyxy == (2, 2, 3, 3)
    assert clicked[1, 1] and not clicked[2, 2]
    assert localizer._predictor.encodes == 1
    assert localizer._predictor.prompts == [(2, 2), (1, 1)]


def _grounded_transform_localizer():
    localizer = _point_localizer()
    localizer.device = "cpu"
    localizer.box_threshold = localizer.text_threshold = localizer.nms_iou_threshold = .5
    transforms, predictions = [], []

    def transform(image, target):
        pixels = np.array(image)
        transforms.append(pixels)
        return pixels, target

    def predict(*, image, caption, **_):
        predictions.append((image, caption))
        if caption == "absent object":
            return np.empty((0, 4)), np.array([]), []
        return np.array([[.5, .5, .4, .5]]), np.array([.9]), [caption]

    localizer._grounding_transform = transform
    localizer._grounding_predict = predict
    return localizer, transforms, predictions


def test_grounding_transform_reuses_sam_rgb_identity_but_not_text_predictions():
    localizer, transforms, predictions = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    for caption in ("blue cube", "red cube", "absent object", "blue cube"):
        detected = localizer.localize(image.copy(), language_prompt=caption, max_detections=1)
        assert [item.label for item in detected] == ([] if caption == "absent object" else [caption])
    assert len(transforms) == 1
    assert [caption for _, caption in predictions] == ["blue cube", "red cube", "absent object", "blue cube"]
    assert all(pixels is transforms[0] for pixels, _ in predictions)
    assert localizer._predictor.encodes == 1
    assert len(localizer._predictor.prompts) == 3
    localizer.close()
    assert localizer._grounding_input is localizer._encoded_image is None


def test_grounding_cache_hit_checks_full_rgb_only_once_under_the_model_lock(monkeypatch):
    localizer, _, _ = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    localizer.localize(image, language_prompt="blue cube", max_detections=1)
    original = np.array_equal
    checks = []

    def checked(left, right):
        if np.asarray(left).ndim == 3:
            checks.append(True)
        return original(left, right)

    monkeypatch.setattr(np, "array_equal", checked)
    localizer.localize(image.copy(), language_prompt="red cube", max_detections=1)
    assert len(checks) == 1


@pytest.mark.parametrize("change", ["pixel", "shape", "point_on_other_image"])
def test_grounding_transform_is_invalidated_with_the_shared_rgb_key(change):
    localizer, transforms, predictions = _grounded_transform_localizer()
    original = np.zeros((4, 5, 3), np.uint8)
    localizer.localize(Image.fromarray(original), language_prompt="blue cube", max_detections=1)
    first_input = localizer._grounding_input
    altered = original.copy()
    altered[0, 0, 0] = 1
    if change == "shape":
        altered = np.zeros((5, 5, 3), np.uint8)
    if change == "point_on_other_image":
        localizer.segment_point(Image.fromarray(altered), pixel_uv=(1, 1))
        assert localizer._grounding_input is None
        altered = original
    localizer.localize(Image.fromarray(altered), language_prompt="red cube", max_detections=1)
    assert len(transforms) == len(predictions) == 2
    assert localizer._grounding_input is not first_input
    np.testing.assert_array_equal(predictions[-1][0], altered)
    # There is no per-frame cache pool: revisiting an evicted RGB transforms it again.
    other = np.full((4, 5, 3), 2, np.uint8)
    localizer.localize(Image.fromarray(other), language_prompt="blue cube", max_detections=1)
    localizer.localize(Image.fromarray(altered), language_prompt="blue cube", max_detections=1)
    assert len(transforms) == 4


def test_grounding_transform_failures_do_not_bind_partial_inputs_to_sam_pixels():
    localizer, transforms, predictions = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    localizer.localize(image, language_prompt="blue cube", max_detections=1)
    different = Image.new("RGB", (5, 4), color=(1, 0, 0))
    localizer._predictor.fail_next = True
    with pytest.raises(RuntimeError, match="encoder_failed"):
        localizer.localize(different, language_prompt="red cube", max_detections=1)
    assert localizer._grounding_input is localizer._encoded_image is None
    localizer.localize(different, language_prompt="red cube", max_detections=1)
    assert len(transforms) == len(predictions) == 3

    transform = localizer._grounding_transform
    def failed_transform(*_):
        raise RuntimeError("transform_failed")
    localizer._grounding_transform = failed_transform
    with pytest.raises(RuntimeError, match="transform_failed"):
        localizer.localize(image, language_prompt="blue cube", max_detections=1)
    assert localizer._grounding_input is None
    localizer._grounding_transform = transform
    localizer.localize(image, language_prompt="blue cube", max_detections=1)
    assert len(transforms) == len(predictions) == 4


def test_zero_detection_does_not_create_a_second_image_identity_or_unused_sam_encoding():
    localizer, transforms, predictions = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    localizer.localize(image, language_prompt="absent object", max_detections=1)
    assert localizer._grounding_input is localizer._encoded_image is None
    assert localizer._predictor.encodes == 0
    localizer.localize(image, language_prompt="blue cube", max_detections=1)
    assert len(transforms) == len(predictions) == 2


def test_grounding_transform_and_predictions_remain_serialized_on_the_model_lock():
    from concurrent.futures import ThreadPoolExecutor

    localizer, transforms, predictions = _grounded_transform_localizer()
    model = _DeviceModel()
    predict = localizer._grounding_predict
    def official_predict(*, model, device, **parameters):
        model.to(device)
        return predict(**parameters)
    localizer._grounding_predict = _DeviceBoundGroundingPredictor(model, official_predict, "cpu")
    image = Image.new("RGB", (5, 4))
    captions = ["blue cube", "red cube", "bottle", "cup"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda caption: localizer.localize(image, language_prompt=caption, max_detections=1), captions))
    assert [items[0].label for items in results] == captions
    assert len(transforms) == localizer._predictor.encodes == 1
    assert len(predictions) == len(captions)
    assert model.moves == ["cpu"]


def test_no_detection_skips_unused_sam_encoding_and_records_timing():
    localizer = _point_localizer()
    localizer._grounding_transform = lambda image, target: (image, target)
    localizer._grounding_predict = lambda **_: (np.empty((0, 4)), np.array([]), [])
    localizer.box_threshold = localizer.text_threshold = localizer.nms_iou_threshold = .5
    localizer.device = "cpu"
    assert localizer.localize(Image.new("RGB", (5, 4)), language_prompt="absent object", max_detections=1) == ()
    assert localizer._predictor.encodes == 0
    assert localizer._predictor.prompts == []
    assert localizer._encoded_image is None
    timing = localizer.last_call_timing_ms
    assert set(timing) == {"lock_wait", "text_image_preprocessing", "grounding_dino", "total"}
    assert all(value >= 0 for value in timing.values())
    timing.clear()
    assert localizer.last_call_timing_ms
    localizer._grounding_predict = lambda **_: (np.empty((0, 4)), np.array([.9]), ["invalid"])
    with pytest.raises(RuntimeError, match="inconsistent detection arrays"):
        localizer.localize(Image.new("RGB", (5, 4)), language_prompt="absent object", max_detections=1)
    assert localizer._predictor.encodes == 0


def test_changed_pixels_shape_and_failed_encoding_invalidate_the_single_entry():
    localizer = _point_localizer()
    array = np.zeros((4, 5, 3), dtype=np.uint8)
    localizer.segment_point(Image.fromarray(array), pixel_uv=(1, 1))
    array[0, 0, 0] = 1
    localizer.segment_point(Image.fromarray(array), pixel_uv=(1, 1))
    assert localizer._predictor.encodes == 2
    localizer.segment_point(Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8)), pixel_uv=(1, 1))
    assert localizer._predictor.encodes == 3
    localizer._predictor.fail_next = True
    with pytest.raises(RuntimeError, match="encoder_failed"):
        localizer.segment_point(Image.fromarray(array), pixel_uv=(1, 1))
    assert localizer._encoded_image is None
    localizer.segment_point(Image.fromarray(array), pixel_uv=(1, 1))
    assert localizer._predictor.encodes == 5
    localizer.close()
    assert localizer._encoded_image is None
    with pytest.raises(RuntimeError, match="not running"):
        localizer.segment_point(Image.fromarray(array), pixel_uv=(1, 1))


def test_rgb_cache_owns_its_pixels_and_is_serialized_with_prediction():
    from concurrent.futures import ThreadPoolExecutor

    localizer = _point_localizer()
    array = np.zeros((4, 5, 3), dtype=np.uint8)
    with localizer._lock:
        localizer._set_prediction_image(array)
    array[0, 0, 0] = 1
    assert localizer._encoded_image[0, 0, 0] == 0
    image = Image.fromarray(array)
    points = [(1, 1), (2, 2), (3, 1), (4, 3)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        masks = list(pool.map(lambda uv: localizer.segment_point(image, pixel_uv=uv), points))
    assert localizer._predictor.encodes == 2
    assert all(mask[v, u] for mask, (u, v) in zip(masks, points, strict=True))


@pytest.mark.parametrize("next_entry", ["localize", "segment", "point"])
def test_full_mask_api_shares_the_source_rgb_key_with_box_and_point_queries(next_entry):
    localizer, transforms, predictions = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    first = localizer.segment(image, language_prompt="blue cube", pixel_uv=None, min_score=.5)
    assert first.label == "blue cube"
    assert first.score == pytest.approx(.9)
    assert first.box_xyxy == (2, 2, 3, 3)
    assert not first.mask.flags.writeable

    if next_entry == "localize":
        result = localizer.localize(image.copy(), language_prompt="red cube", max_detections=1)
        assert result[0].label == "red cube"
    elif next_entry == "segment":
        result = localizer.segment(image.copy(), language_prompt="red cube", pixel_uv=None, min_score=.5)
        assert result.label == "red cube" and result.score == pytest.approx(.9)
    else:
        result = localizer.segment_point(image.copy(), pixel_uv=(1, 1))
        assert result[1, 1] and not result[2, 2]
    assert localizer._predictor.encodes == len(transforms) == 1
    assert len(predictions) == (1 if next_entry == "point" else 2)
    assert len(localizer._predictor.prompts) == 2
    assert first.label == "blue cube" and first.mask[2, 2]


def test_full_mask_and_box_queries_keep_one_image_owner_across_threads():
    from concurrent.futures import ThreadPoolExecutor

    localizer, transforms, predictions = _grounded_transform_localizer()
    model = _DeviceModel()
    predict = localizer._grounding_predict

    def official_predict(*, model, device, **parameters):
        model.to(device)
        return predict(**parameters)

    localizer._grounding_predict = _DeviceBoundGroundingPredictor(model, official_predict, "cpu")
    image = Image.new("RGB", (5, 4))

    def query(index):
        caption = "object " + str(index)
        if index % 3 == 0:
            return localizer.localize(image.copy(), language_prompt=caption, max_detections=1)[0].label
        if index % 3 == 1:
            return localizer.segment(image.copy(), language_prompt=caption, pixel_uv=None, min_score=.5).label
        assert localizer.segment_point(image.copy(), pixel_uv=(1, 1))[1, 1]
        return "point"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(query, range(12)))
    assert results == ["point" if i % 3 == 2 else "object " + str(i) for i in range(12)]
    assert len(transforms) == localizer._predictor.encodes == 1
    assert len(predictions) == 8
    assert len(localizer._predictor.prompts) == 12
    assert model.moves == ["cpu"]


def test_text_mask_miss_does_not_bind_preprocessing_to_the_previous_rgb():
    localizer, transforms, predictions = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    localizer.segment(image, language_prompt="blue cube", pixel_uv=None, min_score=.5)
    different = Image.new("RGB", (5, 4), color=(1, 0, 0))
    assert localizer.segment(different, language_prompt="absent object", pixel_uv=None, min_score=.5) is None
    assert localizer._grounding_input is None
    assert localizer._predictor.encodes == 1
    result = localizer.segment(different, language_prompt="red cube", pixel_uv=None, min_score=.5)
    assert result.label == "red cube"
    assert len(transforms) == len(predictions) == 3
    assert localizer._predictor.encodes == 2
    np.testing.assert_array_equal(predictions[-1][0], np.array(different))


def test_text_mask_encoder_failure_invalidates_the_shared_rgb_identity():
    localizer, transforms, predictions = _grounded_transform_localizer()
    image = Image.new("RGB", (5, 4))
    localizer.localize(image, language_prompt="blue cube", max_detections=1)
    localizer._predictor.fail_next = True
    with pytest.raises(RuntimeError, match="encoder_failed"):
        localizer.segment(Image.new("RGB", (5, 4), color=(1, 0, 0)),
                          language_prompt="red cube", pixel_uv=None, min_score=.5)
    assert localizer._encoded_image is localizer._grounding_input is None
    assert localizer.localize(image, language_prompt="blue cube", max_detections=1)[0].label == "blue cube"
    assert len(transforms) == len(predictions) == localizer._predictor.encodes == 3

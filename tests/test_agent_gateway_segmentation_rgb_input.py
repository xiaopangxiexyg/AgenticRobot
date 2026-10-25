"""Authenticate finite historical RGB inputs without cameras, models or renderers."""

from dataclasses import replace
import hashlib
import io
from types import SimpleNamespace

import pytest
from PIL import Image

from agent_gateway.profiles.segmentation_batch import resolve_segmentation_rgb_input
from agent_gateway.runtime.artifact_producer import artifact_ref_for_bytes, commit_artifact_bytes
from agent_gateway.protocol.artifact_contracts import AgentArtifactBinding
from dataset_contracts.perception_artifacts import (
    RGB_FRAME_SCHEMA_ID_V1, RGB_FRAME_ENCODING_V1, RGB_FRAME_COMPRESSION_V1,
)
from test_agent_gateway_segmentation_skill import batch_service, _batch_parameters, _launch_batch
from test_agent_gateway_service import _request

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _input(case):
    source = SimpleNamespace(**{key: getattr(case.rgb_operation, key) for key in (
        "operation_id", "status", "execution_outcome", "runtime_session_id", "request_hash",
        "output_artifact_bindings")}, execution_result=dict(case.rgb_operation.execution_result))
    # Retain mutable test copies, not replacements in the authoritative store.
    source.execution_result["frames"] = [dict(frame) for frame in source.execution_result["frames"]]
    value = SimpleNamespace(case=case, source=source, original=case.rgb_source_request,
        request=replace(_request(case.manifest, case.registration.skill.routes[0]),
            parameters={**_batch_parameters("cube"), "rgb_capture_operation_id": source.operation_id},
            input_artifact_bindings=(case.rgb_binding,)))
    producer_id = source.operation_id
    value.lookup = lambda operation_id: (value.original, value.source) if operation_id == producer_id else None
    value.active = lambda: None
    return value


def _resolve(value):
    return resolve_segmentation_rgb_input(request=value.request, registration=value.case.registration,
        rgb_capture_registration=value.case.rgb_registration.registration,
        rgb_capture_validator=value.case.rgb_registration.payload_validator,
        artifact_store=value.case.store, lookup_operation=value.lookup, check_active=value.active)


def _replace_bytes(value, data):
    artifact = artifact_ref_for_bytes(data, media_type="image/jpeg", schema_id=RGB_FRAME_SCHEMA_ID_V1,
        encoding=RGB_FRAME_ENCODING_V1, compression=RGB_FRAME_COMPRESSION_V1)
    commit_artifact_bytes(store=value.case.store, data=data, artifact=artifact, producer_id="test:rgb-bytes",
        runtime_session_id=value.request.runtime_session_id, idempotency_key=artifact.sha256, now=value.case.clock())
    value.source.output_artifact_bindings = tuple(AgentArtifactBinding(slot_id="rgb_frames", ordinal=i,
        artifact=artifact, production_receipt_ref=f"test:rgb:{i}") for i in range(2))
    value.source.execution_result["total_bytes"] = 2 * len(data)
    for frame in value.source.execution_result["frames"]:
        frame["media_content_hash"] = artifact.sha256
    value.request = replace(value.request, input_artifact_bindings=(AgentArtifactBinding(
        slot_id="input_image", ordinal=0, artifact=artifact),))


@pytest.mark.parametrize("batch_service", [
    {"version": "3", "rgb_version": "1", "size": (1280, 720)},
    {"version": "3", "rgb_version": "2", "size": (1280, 720)},
], indirect=True)
def test_v3_consumes_original_1280_grid_with_nonzero_producer_ordinal_and_no_capture(batch_service):
    case = batch_service
    pixels = []
    original = case.model.segment

    def segment(image, **kwargs):
        pixels.append((image.tobytes(), kwargs["pixel_uv"]))
        return original(image, **kwargs)

    case.model.segment = segment
    parameters = {"role": "main", "targets": [{"language_prompt": None,
        "target_pixel_uv": [1000, 600], "min_score": .5}]}
    operation = _launch_batch(case, parameters)
    assert operation.status == "completed", case.executor.failure_diagnostics
    result = operation.execution_result
    with Image.open(io.BytesIO(case.rgb_data)) as image:
        assert pixels == [(image.tobytes(), (1000, 600))]
    assert (result["image_width"], result["image_height"]) == (1280, 720)
    assert result["input_rgb_frame_ordinal"] == 1
    assert result["rgb_capture_operation_id"] == case.rgb_operation.operation_id
    assert result["rgb_capture_request_hash"] == case.rgb_source_request.semantic_content_hash
    assert result["input_rgb_artifact_id"] == case.rgb_binding.artifact.artifact_id
    assert result["input_rgb_content_hash"] == hashlib.sha256(case.rgb_data).hexdigest()
    assert result["captured_at"] == 10. and not case.calls and not case.domain.calls
    assert len(operation.output_artifact_bindings) == 2


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("field,value", [
    ("client_id", "other-client"), ("runtime_session_id", "other-session"),
    ("skill_route_manifest_hash", "other-manifest"), ("route_hash", "other-route"),
    ("skill_version", "2"), ("route_id", "other.rgb.capture"), ("route_version", "2"),
    ("skill_definition_hash", "other-definition"), ("skill_id", "perception.segment"),
    ("expected_station_revision", 5),
])
def test_rgb_input_rejects_wrong_original_identity_even_with_consistent_request_hash(batch_service, field, value):
    sample = _input(batch_service)
    sample.original = replace(sample.original, **{field: value})
    sample.source.request_hash = sample.original.semantic_content_hash
    with pytest.raises(ValueError, match="authenticated"):
        _resolve(sample)
    assert not batch_service.model.images and not batch_service.calls


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("mutation", ["evicted", "producer_id", "status", "outcome", "request_hash",
    "session", "binding_ordinal", "full_ref", "frame_ordinal", "role", "source", "frame_hash", "revision"])
def test_rgb_input_rejects_unavailable_or_forged_producer_evidence(batch_service, mutation):
    sample = _input(batch_service)
    source = sample.source
    if mutation == "evicted":
        sample.lookup = lambda _: None
    elif mutation == "producer_id":
        source.operation_id = "another-operation"
    elif mutation == "status":
        source.status = "failed"
    elif mutation == "outcome":
        source.execution_outcome = "failed"
    elif mutation == "request_hash":
        source.request_hash = "a" * 64
    elif mutation == "session":
        source.runtime_session_id = "other-session"
    elif mutation == "binding_ordinal":
        source.output_artifact_bindings = source.output_artifact_bindings[:1]
    elif mutation == "full_ref":
        sample.request = replace(sample.request, input_artifact_bindings=(replace(batch_service.rgb_binding,
            artifact=replace(batch_service.rgb_binding.artifact,
                byte_length=batch_service.rgb_binding.artifact.byte_length + 1)),))
    else:
        frame = source.execution_result["frames"][1]
        if mutation == "frame_ordinal": frame["ordinal"] = True
        elif mutation == "role": frame["role"] = "wrist_left"
        elif mutation == "source": frame["source_id"] = "left_wrist"
        elif mutation == "frame_hash": frame["media_content_hash"] = "a" * 64
        elif mutation == "revision": source.execution_result["station_revision"] = 5
    with pytest.raises(ValueError):
        _resolve(sample)
    assert not batch_service.model.images and not batch_service.calls


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
def test_rgb_exif_cannot_rotate_or_mirror_the_bound_grid(batch_service, orientation):
    sample = _input(batch_service)
    stream = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", (64, 48), "white").save(stream, format="JPEG", exif=exif)
    _replace_bytes(sample, stream.getvalue())
    if orientation == 1:
        capture, provenance = _resolve(sample)
        assert capture.image.size == (64, 48) and provenance["input_rgb_frame_ordinal"] == 1
    else:
        with pytest.raises(ValueError, match="EXIF"):
            _resolve(sample)


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("kind", ["grid", "png", "gray", "truncated", "oversize"])
def test_rgb_actual_encoding_and_grid_checked_before_model(batch_service, kind):
    sample = _input(batch_service)
    stream = io.BytesIO()
    size = (63, 48) if kind == "grid" else (1921, 48) if kind == "oversize" else (64, 48)
    Image.new("L" if kind == "gray" else "RGB", size, "white").save(
        stream, format="PNG" if kind == "png" else "JPEG")
    data = stream.getvalue()
    _replace_bytes(sample, data[:-20] if kind == "truncated" else data)
    with pytest.raises(ValueError):
        _resolve(sample)
    assert not batch_service.model.images


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("kind", ["cas_corrupt", "read_changed"])
def test_rgb_rechecks_actual_hash_even_after_managed_path_resolution(batch_service, monkeypatch, tmp_path, kind):
    sample = _input(batch_service)
    store = batch_service.store
    path = store.managed_read_path(batch_service.rgb_binding.artifact.artifact_id)
    if kind == "cas_corrupt":
        path.chmod(0o600)
        path.write_bytes(b"changed")
        from agent_gateway.runtime.artifact_store import ArtifactStoreIntegrityError
        error = ArtifactStoreIntegrityError
    else:
        monkeypatch.setattr(store, "read_bytes", lambda artifact, *, maximum: b"changed")
        error = ValueError
    with pytest.raises(error):
        _resolve(sample)
    assert not batch_service.model.images


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("missing", ["input", "operation"])
def test_v3_requires_both_input_artifact_and_operation_before_admission(batch_service, missing):
    sample = _input(batch_service)
    request = sample.request
    if missing == "input":
        request = replace(request, input_artifact_bindings=())
    else:
        request = replace(request, parameters=_batch_parameters("cube"))
    outcome = batch_service.service.submit_request(request=request, current_station_revision=4)
    assert not outcome.accepted and not batch_service.model.images and not batch_service.calls


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("field,value", [
    ("schema_id", "dexteleop.perception.segmentation_overlay.v1"),
    ("media_type", "image/png"), ("encoding", "png.rgb8.v1"),
    ("compression", "gzip"), ("byte_length", 16 * 1024 * 1024 + 1),
])
def test_v3_rejects_non_rgb_or_over_budget_artifact_descriptors(batch_service, field, value):
    sample = _input(batch_service)
    binding = batch_service.rgb_binding
    sample.request = replace(sample.request, input_artifact_bindings=(replace(binding,
        artifact=replace(binding.artifact, **{field: value})),))
    with pytest.raises(ValueError):
        _resolve(sample)
    assert not batch_service.model.images


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("value", [None, "", True, " padded", "x" * 257])
def test_v3_rejects_invalid_producer_id_before_model(batch_service, value):
    sample = _input(batch_service)
    sample.request = replace(sample.request, parameters={**dict(sample.request.parameters),
        "rgb_capture_operation_id": value})
    with pytest.raises(ValueError):
        _resolve(sample)
    assert not batch_service.model.images


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
def test_missing_producer_fails_the_real_operation_without_recapture(batch_service):
    operation = _launch_batch(batch_service, {**_batch_parameters("cube"),
        "rgb_capture_operation_id": "unknown-producer"})
    assert operation.status != "completed" and operation.execution_result is None
    assert not operation.output_artifact_bindings
    assert not batch_service.calls and not batch_service.model.images


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
def test_rgb_historical_capture_does_not_gain_a_new_ttl(batch_service):
    sample = _input(batch_service)
    batch_service.clock.set(10000.)
    capture, _ = _resolve(sample)
    assert capture.captured_at == 10. and capture.station_revision == 4


@pytest.mark.parametrize("batch_service", ["3"], indirect=True)
@pytest.mark.parametrize("boundary", ["lookup", "artifact_record", "read_bytes"])
def test_rgb_resolution_checks_active_between_bounded_stages(batch_service, monkeypatch, boundary):
    sample = _input(batch_service)
    interrupted = False

    def active():
        if interrupted:
            raise RuntimeError("cancelled before inference")

    sample.active = active
    lookup = sample.lookup
    if boundary == "lookup":
        def interrupt_lookup(value):
            nonlocal interrupted
            result = lookup(value)
            interrupted = True
            return result
        sample.lookup = interrupt_lookup
    else:
        method = "artifact" if boundary == "artifact_record" else "read_bytes"
        original = getattr(batch_service.store, method)

        def interrupt_read(*args, **kwargs):
            nonlocal interrupted
            result = original(*args, **kwargs)
            interrupted = True
            return result

        monkeypatch.setattr(batch_service.store, method, interrupt_read)
    with pytest.raises(RuntimeError, match="cancelled"):
        _resolve(sample)
    assert not batch_service.model.images and not batch_service.calls

from __future__ import annotations

import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from deployments.perception.sam2_grounding_dino import Sam2PointMaskUnavailable
from deployments.perception.sam3_service import Sam3ServiceLocalizer


def encoded_mask(mask):
    stream = io.BytesIO()
    Image.fromarray(np.asarray(mask, dtype=np.uint8), mode="L").save(
        stream, format="PNG"
    )
    return base64.b64encode(stream.getvalue()).decode()


@pytest.fixture
def service():
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[2:6, 7:10] = 255
    state = SimpleNamespace(
        calls=[],
        result={
            "found": True,
            "score": 0.9,
            "mask_shape": [8, 12],
            "mask_png_base64": encoded_mask(mask),
        },
        error=None,
        raw=None,
        redirect=False,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            assert self.path == "/call"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.calls.append(request)
            if state.redirect:
                self.send_response(302)
                self.send_header("Location", "http://192.0.2.1/call")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            result = (
                {"status": "ok"} if request["method"] == "healthz" else state.result
            )
            envelope = (
                {"ok": True, "result": result}
                if state.error is None
                else {"ok": False, "error": state.error}
            )
            self.wfile.write(
                state.raw if state.raw is not None else json.dumps(envelope).encode()
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    state.endpoint = f"http://127.0.0.1:{server.server_port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def test_text_and_point_share_original_rpc_without_model_or_server_ownership(
    service, monkeypatch
):
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    image = Image.new("RGB", (12, 8), (20, 80, 160))
    with pytest.raises(RuntimeError, match="not running"):
        client.segment_point(image, pixel_uv=(8, 3))
    client.start()
    client.start()
    detections = client.localize(image, language_prompt=" blue cube ", max_detections=8)
    assert len(detections) == 1
    assert detections[0].box_xyxy == (7, 2, 10, 6)
    assert detections[0].score == 0.9 and detections[0].label == "blue cube"
    mask = client.segment_point(image, pixel_uv=(8, 3))
    assert mask.shape == (8, 12) and mask.dtype == bool and mask[3, 8]
    assert [r["method"] for r in service.calls] == ["healthz", "segment", "segment"]
    assert service.calls[1]["kwargs"]["text_prompt"] == "blue cube"
    assert "point" not in service.calls[1]["kwargs"]
    assert service.calls[2]["kwargs"]["point"] == [3, 8]
    assert "text_prompt" not in service.calls[2]["kwargs"]
    raw = base64.b64decode(service.calls[2]["kwargs"]["image_base64"])
    np.testing.assert_array_equal(
        np.asarray(Image.open(io.BytesIO(raw))), np.asarray(image)
    )
    client.close()
    client.close()
    assert len(service.calls) == 3  # In particular, never send shutdown.
    with pytest.raises(RuntimeError, match="cannot restart"):
        client.start()


@pytest.mark.parametrize("mode", ["text", "point"])
def test_shared_segmentation_port_preserves_mask_score_and_artifacts(service, mode):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    assert not client.started
    client.start()
    image = Image.new("RGB", (12, 8), (20, 80, 160))
    prompt = {"language_prompt": " blue cube ", "pixel_uv": None} if mode == "text" else {
        "language_prompt": None, "pixel_uv": (8, 3),
    }
    try:
        assert client.started
        segment = client.segment(image, **prompt, min_score=.5)
        assert segment.score == .9
        assert segment.box_xyxy == (7, 2, 10, 6)
        assert segment.label == ("blue cube" if mode == "text" else None)
        assert segment.mask.shape == (8, 12) and segment.mask[3, 8]
        encoded = client.encode_segment_artifacts(image, segment)
        np.testing.assert_array_equal(
            np.load(io.BytesIO(encoded.mask_npy), allow_pickle=False), segment.mask,
        )
        assert Image.open(io.BytesIO(encoded.overlay_jpeg)).format == "JPEG"
        assert client.segment(image, **prompt, min_score=.99) is None
        service.result = {"found": False}
        assert client.segment(image, **prompt, min_score=.5) is None
    finally:
        client.close()
    assert not client.started


@pytest.mark.parametrize("text,pixel,score", [
    (None, None, .5), ("cube", (8, 3), .5), ("", None, .5),
    (None, (-1, 3), .5), (None, [8, 3], .5), ("cube", None, True),
    ("cube", None, float("nan")), ("cube", None, 1.1),
])
def test_shared_segmentation_port_rejects_invalid_inputs_before_rpc(text, pixel, score):
    client = Sam3ServiceLocalizer()
    with pytest.raises((TypeError, ValueError)):
        client.segment(Image.new("RGB", (12, 8)),
                       language_prompt=text, pixel_uv=pixel, min_score=score)


def test_no_mask_and_best_mask_prompt_miss_remain_recoverable(service):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    client.start()
    image = Image.new("RGB", (12, 8))
    with pytest.raises(Sam2PointMaskUnavailable, match="requested pixel"):
        client.segment_point(image, pixel_uv=(1, 1))
    service.result = {"found": False, "reason": "no candidate"}
    assert client.localize(image, language_prompt="cube", max_detections=1) == ()
    with pytest.raises(Sam2PointMaskUnavailable):
        client.segment_point(image, pixel_uv=(8, 3))


@pytest.mark.parametrize(
    "replacement",
    [
        {"mask_shape": [12, 8]},
        {"mask_png_base64": "not-base64"},
        {"score": float("nan")},
        {"score": 0.1},
        {"found": 1},
        {"mask_png_base64": encoded_mask(np.zeros((8, 12), dtype=np.uint8))},
        {"mask_png_base64": encoded_mask(np.ones((8, 12), dtype=np.uint8))},
    ],
)
def test_malformed_outputs_are_faults_not_association_misses(service, replacement):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    client.start()
    service.result.update(replacement)
    with pytest.raises(RuntimeError) as error:
        client.segment_point(Image.new("RGB", (12, 8)), pixel_uv=(8, 3))
    assert not isinstance(error.value, Sam2PointMaskUnavailable)


@pytest.mark.parametrize("fault", ["rpc", "json", "redirect"])
def test_transport_faults_surface_without_fallback_or_retry(service, fault):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint, timeout_sec=0.1)
    client.start()
    if fault == "rpc":
        service.error = {"type": "RuntimeError", "message": "model_failed"}
    elif fault == "json":
        service.raw = b"not json"
    else:
        service.redirect = True
    expected = {
        "rpc": "model_failed",
        "json": "invalid JSON",
        "redirect": "must not redirect",
    }
    with pytest.raises(RuntimeError, match=expected[fault]):
        client.segment_point(Image.new("RGB", (12, 8)), pixel_uv=(8, 3))
    assert len(service.calls) == 2


def test_service_timeout_is_forwarded_and_never_retried(service, monkeypatch):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint, timeout_sec=0.05)
    client.start()
    calls = []

    def timed_out(request, *, timeout):
        calls.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(client, "_opener", SimpleNamespace(open=timed_out))
    with pytest.raises(RuntimeError, match="transport failed: timed out"):
        client.segment_point(Image.new("RGB", (12, 8)), pixel_uv=(8, 3))
    assert calls == [0.05]
    assert len(service.calls) == 1


def test_oversized_service_response_is_a_protocol_fault(service, monkeypatch):
    from deployments.perception import sam3_service

    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    client.start()
    monkeypatch.setattr(sam3_service, "MAX_RPC_RESPONSE_BYTES", 8)
    with pytest.raises(RuntimeError, match="exceeds its byte bound"):
        client.segment_point(Image.new("RGB", (12, 8)), pixel_uv=(8, 3))
    assert len(service.calls) == 2


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:8114",
        "http://192.168.1.2:8114",
        "http://example.com",
        "http://user:pass@localhost:8114",
        "http://localhost:8114/call",
        "http://localhost:8114?x=1",
        "http://localhost:0",
        "http://localhost:65536",
    ],
)
def test_non_loopback_or_ambiguous_endpoints_rejected(endpoint):
    with pytest.raises(ValueError):
        Sam3ServiceLocalizer(endpoint=endpoint)


@pytest.mark.parametrize(
    "options",
    [
        {"timeout_sec": 0},
        {"timeout_sec": True},
        {"timeout_sec": np.inf},
        {"min_score": -0.1},
        {"min_score": 1.1},
        {"min_score": np.nan},
    ],
)
def test_invalid_service_bounds_rejected(options):
    with pytest.raises(ValueError):
        Sam3ServiceLocalizer(**options)


def test_unprompted_localization_is_explicitly_unsupported(service):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    with pytest.raises(ValueError, match="automatic proposals are unsupported"):
        client.localize(
            Image.new("RGB", (12, 8)), language_prompt=None, max_detections=10
        )
    assert service.calls == []


def test_new_image_and_new_point_never_reuse_an_old_mask(service):
    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    client.start()
    image = Image.new("RGB", (12, 8))
    assert client.segment_point(image, pixel_uv=(8, 3))[3, 8]
    mask = np.zeros((8, 12), dtype=np.uint8)
    mask[1, 2] = 255
    service.result["mask_png_base64"] = encoded_mask(mask)
    assert client.segment_point(image, pixel_uv=(2, 1))[1, 2]
    image.putpixel((0, 0), (255, 0, 0))
    client.segment_point(image, pixel_uv=(2, 1))
    assert len(service.calls) == 4
    assert (
        service.calls[2]["kwargs"]["image_base64"]
        != service.calls[3]["kwargs"]["image_base64"]
    )


def test_existing_localization_provider_accepts_service_identity_and_unchanged_schema(
    service, tmp_path
):
    from agent_gateway.profiles.real.localization_deployment import (
        build_real_object_localization_registration,
    )
    from agent_gateway.profiles.real.localization_gateway import (
        ObjectLocalizationProvider,
    )

    client = Sam3ServiceLocalizer(endpoint=service.endpoint)
    registration = build_real_object_localization_registration(
        model_profile_id=client.model_spec.model_profile_id,
        model_profile_hash=client.model_spec.semantic_content_hash,
        inference_implementation_hash=client.semantic_content_hash,
        local_path_enabled=True,
        runtime_session_id="session:test",
        control_domain_id="domain:test",
        target_binding_id="target:test",
        station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )
    image_path = tmp_path / "input.png"
    Image.new("RGB", (12, 8)).save(image_path)
    provider = ObjectLocalizationProvider(
        localizer=client,
        registration=registration,
        allowed_local_image_roots=(tmp_path,),
    )
    provider.start()
    try:
        result = provider.localize_request(
            SimpleNamespace(
                parameters={
                    "input_source": "local_path",
                    "local_image_path": str(image_path),
                    "language_prompt": "blue cube",
                    "max_detections": 10,
                },
                input_artifact_bindings=(),
            ),
            artifact_store=SimpleNamespace(),
        )
        registration.payload_validator.validate_execution_result(
            result.execution_result
        )
        assert result.execution_result["detection_count"] == 1
        assert (
            result.execution_result["model_profile_id"] == "sam3-service-text-point-v1"
        )
    finally:
        provider.close()

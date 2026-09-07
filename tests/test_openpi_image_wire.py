from __future__ import annotations

import json
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from deployments.openpi.client.transport import (
    OpenPIWebsocketTransport,
    packb,
    unpackb,
)
from deployments.openpi.image_wire import (
    OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1,
    OPENPI_IMAGE_WIRE_JPEG_V1,
    OPENPI_IMAGE_WIRE_LEGACY_RAW_V1,
    OPENPI_IMAGE_WIRE_RESIZE_RAW_V1,
    OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1,
    OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1,
    OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2,
    OpenPIImageWireClientConfig,
    OpenPIImageWireError,
    OpenPIImageWireServerConfig,
    OpenPIImageWireTrainingTransform,
    build_openpi_image_wire_capability,
    decode_openpi_image_wire_request,
    encode_openpi_image_wire_request,
    install_openpi_image_wire_server_adapter,
    resize_openpi_image_with_pad_v1,
    simulate_openpi_image_wire_roundtrip,
    validate_openpi_image_wire_capability,
)


def _image(height: int = 8, width: int = 12) -> np.ndarray:
    rows = np.arange(height, dtype=np.uint8)[:, None]
    columns = np.arange(width, dtype=np.uint8)[None, :]
    return np.stack(
        (
            np.broadcast_to(rows * 17, (height, width)),
            np.broadcast_to(columns * 13, (height, width)),
            (rows * 7 + columns * 11).astype(np.uint8),
        ),
        axis=-1,
    )


def _capability(profile: str = "jpeg") -> dict:
    return build_openpi_image_wire_capability(
        OpenPIImageWireServerConfig(
            profile=profile,
            target_height=6,
            target_width=6,
            jpeg_quality=90,
        ),
        image_keys=("base_0_rgb",),
    )


def _client(codec: str, *, fallback: bool = False) -> OpenPIImageWireClientConfig:
    return OpenPIImageWireClientConfig(
        preferred_codec=codec,
        allow_legacy_fallback=fallback,
        image_keys=("base_0_rgb",),
    )


def test_legacy_raw_is_an_exact_noop_without_server_capability() -> None:
    request = {
        "base_0_rgb": _image(),
        "state": np.arange(4, dtype=np.float32),
    }

    encoded, receipt = encode_openpi_image_wire_request(
        request,
        OpenPIImageWireClientConfig(),
        {},
    )

    assert encoded is request
    assert receipt is None
    assert packb(encoded) == packb(request)


def test_missing_capability_can_fallback_to_legacy_raw_without_copying() -> None:
    request = {"base_0_rgb": _image()}

    encoded, receipt = encode_openpi_image_wire_request(
        request,
        _client(OPENPI_IMAGE_WIRE_JPEG_V1, fallback=True),
        {},
    )

    assert encoded is request
    assert receipt is not None
    assert receipt.effective_codec == OPENPI_IMAGE_WIRE_LEGACY_RAW_V1
    assert receipt.fallback_reason == "server_capability_absent"


def test_missing_capability_fails_before_send_in_strict_mode() -> None:
    with pytest.raises(OpenPIImageWireError, match="server_capability_absent"):
        encode_openpi_image_wire_request(
            {"base_0_rgb": _image()},
            _client(OPENPI_IMAGE_WIRE_JPEG_V1, fallback=False),
            {},
        )


def test_capability_hash_detects_metadata_drift() -> None:
    capability = _capability()
    assert validate_openpi_image_wire_capability(capability) == capability

    drifted = {**capability, "target_shape": [7, 6, 3]}
    with pytest.raises(OpenPIImageWireError, match="hash mismatch"):
        validate_openpi_image_wire_capability(drifted)


def test_resize_raw_roundtrip_matches_versioned_openpi_resize() -> None:
    source = _image()
    capability = _capability("resize-raw")
    request = {"base_0_rgb": source, "state": np.zeros(2, dtype=np.float32)}

    encoded, encode_receipt = encode_openpi_image_wire_request(
        request,
        _client(OPENPI_IMAGE_WIRE_RESIZE_RAW_V1),
        {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability},
    )
    decoded, decode_receipt = decode_openpi_image_wire_request(
        encoded,
        capability,
    )

    expected = resize_openpi_image_with_pad_v1(source, 6, 6)
    np.testing.assert_array_equal(decoded["base_0_rgb"], expected)
    assert request["base_0_rgb"].shape == (8, 12, 3)
    assert encode_receipt is not None
    assert encode_receipt.source_image_bytes == source.nbytes
    assert encode_receipt.encoded_image_bytes == expected.nbytes
    assert decode_receipt is not None
    assert decode_receipt.decoded_image_bytes == expected.nbytes


def test_jpeg_roundtrip_is_bounded_rgb_and_reuses_training_function() -> None:
    source = np.tile(_image(8, 12), (20, 20, 1))
    expected = simulate_openpi_image_wire_roundtrip(
        source,
        codec=OPENPI_IMAGE_WIRE_JPEG_V1,
        target_height=224,
        target_width=224,
        jpeg_quality=90,
    )

    assert expected.shape == (224, 224, 3)
    assert expected.dtype == np.uint8
    assert expected.flags.c_contiguous


def test_conditioned_request_only_replaces_nested_observation_images() -> None:
    capability = _capability("resize-raw")
    actions = np.arange(32, dtype=np.float32).reshape(2, 16)
    request = {
        "__openpi_request__": {
            "wire_protocol": "openpi.conditioned_rtc.v1",
            "operation": "infer_conditioned_rtc_v1",
        },
        "observation": {"base_0_rgb": _image(), "state": np.zeros(4)},
        "conditioning_actions": actions,
    }

    encoded, _ = encode_openpi_image_wire_request(
        request,
        _client(OPENPI_IMAGE_WIRE_RESIZE_RAW_V1),
        {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability},
    )
    decoded, _ = decode_openpi_image_wire_request(encoded, capability)

    np.testing.assert_array_equal(decoded["conditioning_actions"], actions)
    assert decoded["observation"]["base_0_rgb"].shape == (6, 6, 3)
    assert request["observation"]["base_0_rgb"].shape == (8, 12, 3)


def test_jpeg_preference_rolls_back_to_resized_raw_on_capable_server() -> None:
    capability = _capability("resize-raw")

    encoded, receipt = encode_openpi_image_wire_request(
        {"base_0_rgb": _image()},
        _client(OPENPI_IMAGE_WIRE_JPEG_V1, fallback=False),
        {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability},
    )

    assert receipt is not None
    assert receipt.effective_codec == OPENPI_IMAGE_WIRE_RESIZE_RAW_V1
    assert encoded["base_0_rgb"]["codec"] == OPENPI_IMAGE_WIRE_RESIZE_RAW_V1


def test_prepared_server_adapter_advertises_decodes_and_restores_class() -> None:
    class FakeWebsocketPolicyServer:
        def __init__(self, *, metadata: dict) -> None:
            self._metadata = dict(metadata)
            self.received = None

        def _infer(self, request):
            self.received = request
            return {"actions": np.zeros((1, 2), dtype=np.float32)}

    module = SimpleNamespace(WebsocketPolicyServer=FakeWebsocketPolicyServer)
    config = OpenPIImageWireServerConfig(
        profile="jpeg",
        target_height=6,
        target_width=6,
        jpeg_quality=90,
    )
    metadata = {"dataset_adapter": {"camera_slots": ["base_0_rgb"]}}

    request_key = "observation.images.front"
    with install_openpi_image_wire_server_adapter(
        module,
        config,
        image_keys=(request_key,),
    ):
        server = module.WebsocketPolicyServer(metadata=metadata)
        capability = server._metadata[OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1]
        assert capability["image_keys"] == [request_key]
        encoded, _ = encode_openpi_image_wire_request(
            {request_key: _image()},
            OpenPIImageWireClientConfig(
                preferred_codec=OPENPI_IMAGE_WIRE_JPEG_V1,
                image_keys=(request_key,),
            ),
            {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability},
        )
        result = server._infer(encoded)

        assert result["actions"].shape == (1, 2)
        assert server.received[request_key].shape == (6, 6, 3)
        assert server.last_image_wire_decode_receipt.image_count == 1

    assert module.WebsocketPolicyServer is FakeWebsocketPolicyServer


def test_transport_encodes_after_handshake_and_records_wire_sizes() -> None:
    capability = _capability("resize-raw")

    class Connection:
        sent: bytes | None = None

        def send(self, value: bytes) -> None:
            self.sent = value

        def recv(self, timeout=None) -> bytes:
            del timeout
            return packb({"actions": np.zeros((1, 2), dtype=np.float32)})

    connection = Connection()
    transport = OpenPIWebsocketTransport(
        image_wire_config=_client(OPENPI_IMAGE_WIRE_RESIZE_RAW_V1),
    )
    transport._ws = connection
    transport._metadata = {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability}

    result = transport.infer({"base_0_rgb": _image()})

    assert result["actions"].shape == (1, 2)
    assert connection.sent is not None
    packed_request = unpackb(connection.sent)
    assert packed_request["base_0_rgb"]["codec"] == (
        OPENPI_IMAGE_WIRE_RESIZE_RAW_V1
    )
    receipt = transport.last_image_wire_receipt
    assert receipt is not None
    assert receipt.packed_request_bytes == len(connection.sent)
    assert receipt.response_bytes is not None


def test_policy_binds_optimized_codec_to_actual_openpi_request_keys() -> None:
    try:
        from deployments.openpi.client.policy import OpenPIClientPolicyConfig
    except ModuleNotFoundError as exc:
        if exc.name == "osqp":
            pytest.skip("The isolated OpenPI environment has no robot-control extras.")
        raise

    config = OpenPIClientPolicyConfig(
        required_camera_slots=(
            "base_0_rgb",
            "left_wrist_0_rgb",
            "right_wrist_0_rgb",
        ),
        required_image_keys=("front", "left_wrist", "right_wrist"),
        image_key_map={
            "front": "observation.images.front",
            "left_wrist": "observation.images.left_wrist",
            "right_wrist": "observation.images.right_wrist",
        },
        image_wire_config={
            "preferred_codec": OPENPI_IMAGE_WIRE_JPEG_V1,
            "allow_legacy_fallback": True,
        },
    )

    assert config.image_wire_config.image_keys == (
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    assert config.image_wire_config.preferred_codec == OPENPI_IMAGE_WIRE_JPEG_V1


def test_prepared_runtime_maps_camera_slots_to_actual_request_keys() -> None:
    from deployments.openpi.prepared_server_runtime import (
        _prepared_request_image_keys,
    )

    camera_slots = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    request_keys = (
        "observation.images.front",
        "observation.images.left_wrist",
        "observation.images.right_wrist",
    )
    resolution = SimpleNamespace(
        deployment_artifact=SimpleNamespace(
            camera_input=SimpleNamespace(
                server_camera_slots=camera_slots,
                inputs=tuple(
                    SimpleNamespace(model_input_key=key) for key in request_keys
                ),
            )
        )
    )
    config = SimpleNamespace(model=SimpleNamespace(image_keys=camera_slots))

    assert _prepared_request_image_keys(resolution, config) == request_keys


def test_actual_openpi_server_class_can_be_wrapped_without_model_or_socket() -> None:
    server_module = pytest.importorskip(
        "openpi.serving.websocket_policy_server",
        reason="The local DexTeleop environment does not install OpenPI server code.",
    )

    class Policy:
        def infer(self, request):
            assert request["base_0_rgb"].shape == (224, 224, 3)
            return {"actions": np.zeros((1, 2), dtype=np.float32)}

    config = OpenPIImageWireServerConfig(profile="jpeg")
    metadata = {"dataset_adapter": {"camera_slots": ["base_0_rgb"]}}
    original = server_module.WebsocketPolicyServer

    with install_openpi_image_wire_server_adapter(server_module, config):
        server = server_module.WebsocketPolicyServer(
            policy=Policy(),
            host="127.0.0.1",
            port=None,
            metadata=metadata,
        )
        capability = server._metadata[OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1]
        encoded, _ = encode_openpi_image_wire_request(
            {"base_0_rgb": _image(480, 640)},
            _client(OPENPI_IMAGE_WIRE_JPEG_V1),
            {OPENPI_IMAGE_WIRE_CAPABILITY_KEY_V1: capability},
        )
        result = server._infer(encoded)

        assert result["actions"].shape == (1, 2)
        assert server.last_image_wire_decode_receipt.image_count == 1

    assert server_module.WebsocketPolicyServer is original


def test_training_transform_is_opt_in_and_chains_existing_transform() -> None:
    class Upstream:
        epoch = None

        def __call__(self, sample):
            return {**sample, "upstream_applied": True}

        def set_epoch(self, epoch: int) -> None:
            self.epoch = epoch

    upstream = Upstream()
    transform = OpenPIImageWireTrainingTransform.from_profile_value(
        {
            "schema": OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1,
            "profile": "jpeg",
            "target_shape": [6, 6, 3],
            "jpeg_quality": 90,
        },
        image_keys=("base_0_rgb",),
        upstream=upstream,
    )
    assert transform is not None
    source = _image()
    sample = {"image": {"base_0_rgb": source}, "state": np.zeros(2)}

    output = transform(sample)
    transform.set_epoch(7)

    assert output["upstream_applied"] is True
    assert output["image"]["base_0_rgb"].shape == (6, 6, 3)
    assert sample["image"]["base_0_rgb"] is source
    assert upstream.epoch == 7
    assert transform.contract["schema"] == OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1
    assert transform.contract["server_capability"]["image_keys"] == [
        "base_0_rgb"
    ]
    assert OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1 == (
        "image_wire_training_alignment"
    )


def test_training_transform_without_upstream_is_pickle_safe() -> None:
    transform = OpenPIImageWireTrainingTransform.from_profile_value(
        {
            "schema": OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1,
            "profile": "resize-raw",
            "target_shape": [6, 6, 3],
            "jpeg_quality": 90,
        },
        image_keys=("base_0_rgb",),
    )

    restored = pickle.loads(pickle.dumps(transform))

    assert restored.contract == transform.contract


def test_training_transform_v2_selects_deterministic_sixty_forty_mix() -> None:
    transform = OpenPIImageWireTrainingTransform.from_profile_value(
        {
            "schema": OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2,
            "profile": "jpeg",
            "target_shape": [6, 6, 3],
            "jpeg_quality": 90,
            "application_probability": 0.6,
        },
        image_keys=("base_0_rgb", "left_wrist_0_rgb"),
    )
    assert transform is not None
    identities = [(episode, row) for episode in range(20) for row in range(50)]
    epoch_zero = [transform.applies_to_identity(identity) for identity in identities]
    assert 550 <= sum(epoch_zero) <= 650
    assert epoch_zero == [
        transform.applies_to_identity(identity) for identity in identities
    ]

    restored = pickle.loads(pickle.dumps(transform))
    assert epoch_zero == [
        restored.applies_to_identity(identity) for identity in identities
    ]
    restored.set_epoch(1)
    epoch_one = [restored.applies_to_identity(identity) for identity in identities]
    assert 550 <= sum(epoch_one) <= 650
    assert epoch_one != epoch_zero
    assert restored.contract["application_probability"] == 0.6
    assert restored.contract["bypass_semantics"] == "preserve_upstream_sample_v1"
    assert restored.contract["server_capability"]["jpeg"]["quality"] == 90
    summary = restored.selection_summary(iter(identities), epoch=1)
    assert summary["sample_count"] == 1000
    assert summary["codec_sample_count"] == sum(epoch_one)
    assert summary["bypass_sample_count"] == 1000 - sum(epoch_one)
    assert summary["effective_application_probability"] == sum(epoch_one) / 1000
    assert summary["configured_application_probability"] == 0.6


def test_training_transform_v2_applies_or_preserves_the_whole_sample() -> None:
    transform = OpenPIImageWireTrainingTransform.from_profile_value(
        {
            "schema": OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2,
            "profile": "jpeg",
            "target_shape": [6, 6, 3],
            "jpeg_quality": 90,
            "application_probability": 0.6,
        },
        image_keys=("base_0_rgb", "left_wrist_0_rgb"),
    )
    assert transform is not None
    applied_identity = next(
        (index, 0)
        for index in range(100)
        if transform.applies_to_identity((index, 0))
    )
    bypass_identity = next(
        (index, 0)
        for index in range(100)
        if not transform.applies_to_identity((index, 0))
    )
    source = _image()
    wrist = np.flip(source, axis=1).copy()
    sample = {
        "image": {
            "base_0_rgb": source,
            "left_wrist_0_rgb": wrist,
        },
        "state": np.zeros(2),
    }

    applied = transform.apply_with_identity(sample, identity=applied_identity)
    bypassed = transform.apply_with_identity(sample, identity=bypass_identity)

    assert applied["image"]["base_0_rgb"].shape == (6, 6, 3)
    assert applied["image"]["left_wrist_0_rgb"].shape == (6, 6, 3)
    assert bypassed is sample
    assert bypassed["image"]["base_0_rgb"] is source
    assert bypassed["image"]["left_wrist_0_rgb"] is wrist
    with pytest.raises(OpenPIImageWireError, match="stable sample identity"):
        transform(sample)


@pytest.mark.parametrize(
    "probability",
    [-0.1, 1.1, float("nan"), True, "0.6"],
)
def test_training_transform_v2_rejects_invalid_probability(probability: object) -> None:
    with pytest.raises(OpenPIImageWireError, match="application_probability"):
        OpenPIImageWireTrainingTransform.from_profile_value(
            {
                "schema": OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V2,
                "profile": "jpeg",
                "target_shape": [6, 6, 3],
                "jpeg_quality": 90,
                "application_probability": probability,
            },
            image_keys=("base_0_rgb",),
        )


def test_writing_profiles_add_isolated_q90_sixty_percent_variants() -> None:
    profile_root = Path("deployments/openpi/configs/training")
    allowed_differences = {
        "profile_id",
        "openpi_config_name",
        "default_exp_name",
        "openpi_image_wire_training",
    }
    for stage in range(1, 5):
        base = json.loads(
            (
                profile_root
                / f"writing_stage{stage}_0804_0806_front_h16_absolute_v1.json"
            ).read_text(encoding="utf-8")
        )
        variant = json.loads(
            (
                profile_root
                / (
                    f"writing_stage{stage}_0804_0806_front_h16_absolute_"
                    "jpeg_q90_p60_v1.json"
                )
            ).read_text(encoding="utf-8")
        )
        assert "openpi_image_wire_training" not in base
        changed = {
            key
            for key in set(base) | set(variant)
            if base.get(key) != variant.get(key)
        }
        assert changed == allowed_differences
        assert variant["organized_dataset_root"] == base["organized_dataset_root"]
        assert variant["openpi_asset_id"] == base["openpi_asset_id"]
        assert variant["openpi_config_name"] != base["openpi_config_name"]
        assert variant["default_exp_name"] != base["default_exp_name"]

        transform = OpenPIImageWireTrainingTransform.from_profile_value(
            variant["openpi_image_wire_training"],
            image_keys=tuple(variant["camera_slot_to_dataset_key"]),
        )
        assert transform is not None
        assert transform.application_probability == 0.6
        assert transform.server_config.profile == "jpeg"
        assert transform.server_config.jpeg_quality == 90
        assert transform.server_config.target_height == 224
        assert transform.server_config.target_width == 224


def test_prepared_checkpoint_must_match_opt_in_training_contract() -> None:
    from deployments.openpi.prepared_deployment import (
        _validate_image_wire_training_metadata,
    )

    profile = {
        "camera_slot_to_dataset_key": {"base_0_rgb": "front"},
        "openpi_image_wire_training": {
            "schema": OPENPI_IMAGE_WIRE_TRAINING_SCHEMA_V1,
            "profile": "jpeg",
            "target_shape": [224, 224, 3],
            "jpeg_quality": 90,
        },
    }
    transform = OpenPIImageWireTrainingTransform.from_profile_value(
        profile["openpi_image_wire_training"],
        image_keys=("base_0_rgb",),
    )
    assert transform is not None
    payload = {
        OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1: transform.contract,
    }

    _validate_image_wire_training_metadata(payload, profile)
    drifted = {
        OPENPI_IMAGE_WIRE_TRAINING_METADATA_KEY_V1: {
            **transform.contract,
            "application_probability": 0.5,
        }
    }
    with pytest.raises(ValueError, match="training alignment differs"):
        _validate_image_wire_training_metadata(drifted, profile)

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

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
    LOCALIZATION_INPUT_JPEG_ENCODING_V1,
    LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
)
from deployments.perception.sam2_grounding_dino import (
    GroundedObjectDetection,
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


def test_artifact_mode_reads_committed_bytes_and_preserves_prompt(
    tmp_path: Path,
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
            schema_id=LOCALIZATION_INPUT_IMAGE_SCHEMA_ID_V1,
            encoding=LOCALIZATION_INPUT_JPEG_ENCODING_V1,
            compression=LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
        )
        committed = commit_artifact_bytes(
            store=store,
            data=data,
            artifact=artifact,
            producer_id="test-client",
            runtime_session_id="session:test",
            idempotency_key="input:test",
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

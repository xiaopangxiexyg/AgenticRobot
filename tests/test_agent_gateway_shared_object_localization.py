from __future__ import annotations
import hashlib
import inspect
import io
import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from agent_gateway.profiles import localization, localization_deployment
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
from dataset_contracts.canonical_encoding import canonical_content_hash
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
        self.lifecycle_calls = []

    def start(self) -> None:
        self.lifecycle_calls.append("start")
        self.started = True

    def close(self) -> None:
        self.lifecycle_calls.append("close")
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


def _registration(*, local_path_enabled: bool, current_mode="real_observe"):
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
        current_mode=current_mode,
    )


@pytest.mark.parametrize("current_mode,local_path_enabled,expected", (
    ("real_observe", False, "fe86d0590cb24f60389c4a1b50dd7a5859d835be0324ad96500ccbdc8e0ad771"),
    ("real_observe", True, "bdcf1769c4482e6b80ed312ac4fd05bc95aa2f2fe0e69ffbd77e3ef8d3eecdb0"),
    ("real_active", False, "65b511e375ea8da3d9469d292b73bacab0ad0412a67401bd77d7eb485fc51cd5"),
    ("real_active", True, "33ccbd63fe46467abb8203cb2452805a808ae360c4082cbcaa70fbba3c23db0d"),
))
def test_real_localization_v1_wire_identity_stays_compatible(
    current_mode, local_path_enabled, expected,
):
    # Frozen from accepted development 682419118 before sharing the implementation.
    registration = _registration(local_path_enabled=local_path_enabled, current_mode=current_mode)
    skill = registration.skill
    wire = {
        "definition": skill.definition.to_wire(),
        "schema": skill.schema_entry.to_wire(),
        "io": skill.io_contract.to_wire(),
        "route": skill.routes[0].to_wire(),
        "profile_hash": registration.profile.semantic_content_hash,
        "binding": registration.binding.to_ref().to_wire(),
        "binding_hash": registration.binding.semantic_content_hash,
        "availability_hash": registration.availability_hash,
        "validator": {key: value for key, value in vars(registration.payload_validator).items()
                      if isinstance(value, str)},
    }
    assert canonical_content_hash(
        wire, domain="test.supplied_image_localization_real_compatibility.v1",
    ) == expected


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


def _shared_registration(
    *, target_kind="sim", current_mode="sim_active", local_path_enabled=False,
    model_profile_hash=_FakeSpec.semantic_content_hash,
    inference_implementation_hash=_FakeLocalizer.semantic_content_hash,
):
    return localization_deployment.build_object_localization_registration(
        target_kind=target_kind, current_mode=current_mode, local_path_enabled=local_path_enabled,
        model_profile_id=_FakeSpec.model_profile_id,
        model_profile_hash=model_profile_hash,
        inference_implementation_hash=inference_implementation_hash,
        runtime_session_id="session:test", control_domain_id="domain:test",
        target_binding_id="target:test", station_manifest_hash="station:test",
        execution_manifest_hash="execution:test",
    )


def _image_request(store, data):
    committed = commit_artifact_bytes(
        store=store, data=data,
        artifact=artifact_ref_for_bytes(
            data, media_type=LOCALIZATION_INPUT_JPEG_MEDIA_TYPE_V1,
            schema_id=RGB_FRAME_SCHEMA_ID_V1,
            encoding=LOCALIZATION_INPUT_JPEG_ENCODING_V1,
            compression=LOCALIZATION_INPUT_IMAGE_COMPRESSION_V1,
        ),
        producer_id="test:rgb", runtime_session_id="session:test",
        idempotency_key="input", now=1.,
    )
    return SimpleNamespace(
        parameters={"input_source": "artifact", "local_image_path": None,
                    "language_prompt": "blue object", "max_detections": 3},
        input_artifact_bindings=(AgentArtifactBinding(slot_id="input_image", ordinal=0, artifact=committed),),
        semantic_content_hash="request-hash:test",
    )


def test_real_localization_modules_reexport_the_shared_objects():
    from agent_gateway.profiles import localization, localization_deployment
    from agent_gateway.profiles.real import localization_deployment as legacy_deployment
    from agent_gateway.profiles.real import localization_gateway as legacy_gateway

    for name in legacy_deployment.__all__:
        assert getattr(legacy_deployment, name) is getattr(localization_deployment, name)
    for name in legacy_gateway.__all__:
        assert getattr(legacy_gateway, name) is getattr(localization, name)


@pytest.mark.parametrize("orientation", [None, 1, 2, 6])
@pytest.mark.parametrize("kind", ["real", "sim"])
def test_supplied_image_grid_policy_preserves_real_and_rejects_sim_rotation(
    tmp_path, orientation, kind,
):
    registration = _shared_registration(
        target_kind=kind,
        current_mode="real_observe" if kind == "real" else "sim_active",
    )
    image = Image.new("RGB", (12, 10), color=(32, 64, 96))
    exif = Image.Exif()
    if orientation is not None:
        exif[274] = orientation
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", exif=exif)
    data = encoded.getvalue()
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(localizer=localizer, registration=registration)
    with _store(tmp_path / "store") as store:
        request = _image_request(store, data)
        if kind == "sim" and orientation not in (None, 1):
            with pytest.raises(ValueError, match="(?i)exif|orientation"):
                provider.localize_request(request, artifact_store=store)
            assert localizer.calls == []
        else:
            prepared = provider.localize_request(request, artifact_store=store)
            assert prepared.execution_result["input_media_content_hash"] == hashlib.sha256(data).hexdigest()
            assert localizer.calls[0][0] == ((10, 12) if orientation == 6 else (12, 10))
            registration.payload_validator.validate_execution_result(prepared.execution_result)


@pytest.mark.parametrize("target_kind", ["mujoco", "REAL", " sim", "", None, 1, []])
def test_localization_rejects_unknown_target_kind(target_kind):
    with pytest.raises((TypeError, ValueError)):
        _shared_registration(target_kind=target_kind)


@pytest.mark.parametrize("target_kind,current_mode", [
    ("real", "sim_active"), ("real", "sim_observe"), ("sim", "real_active"),
    ("sim", "real_observe"), ("sim", "sim_observe"), ("sim", None), ("real", []),
])
def test_localization_rejects_mixed_or_unknown_mode(target_kind, current_mode):
    with pytest.raises((TypeError, ValueError)):
        _shared_registration(target_kind=target_kind, current_mode=current_mode)


def test_sim_localization_rejects_local_path_capability():
    with pytest.raises(ValueError, match="local-path"):
        _shared_registration(local_path_enabled=True)
    registration = _shared_registration()
    with pytest.raises(ValueError, match="local-path"):
        replace(registration.profile, local_path_enabled=True)
    with pytest.raises(ValueError):
        registration.payload_validator.validate_parameters({
            "input_source": "local_path", "local_image_path": "forbidden-image.jpg",
            "language_prompt": "blue object", "max_detections": 3,
        })


def test_sim_localization_closes_over_read_only_target_and_unchanged_io():
    sim = _shared_registration()
    real = _registration(local_path_enabled=False)
    assert sim.skill.definition == real.skill.definition
    assert sim.skill.schema_entry.parameter_schema == real.skill.schema_entry.parameter_schema
    assert sim.skill.schema_entry.execution_result_schema == real.skill.schema_entry.execution_result_schema
    assert sim.skill.io_contract == real.skill.io_contract
    assert sim.profile.semantic_content_hash != real.profile.semantic_content_hash
    assert sim.binding.semantic_content_hash != real.binding.semantic_content_hash
    route = sim.skill.routes[0]
    assert route.target_kind == sim.binding.to_ref().target_kind == "sim"
    assert route.supported_modes == ("sim_active",)
    assert route.route_id == "sim.local_model.perception.localize_objects"
    assert sim.binding.executor_id == "sim-passive-object-localization.v1"
    assert route.control_effect_class == "read_only"
    assert route.privilege_tier == "t0_read_only"
    assert not route.may_move_robot and not route.may_open_publication


@pytest.mark.parametrize("field", ["profile", "binding", "skill", "payload_validator"])
def test_localization_rejects_cross_target_registration_parts(field):
    sim = _shared_registration()
    real = _registration(local_path_enabled=False)
    with pytest.raises(ValueError):
        replace(sim, **{field: getattr(real, field)})


def test_only_sim_profile_hash_commits_original_grid_policy(monkeypatch):
    records = []

    def capture(payload, *, domain):
        records.append((payload, domain))
        return canonical_content_hash(payload, domain=domain)

    monkeypatch.setattr(localization_deployment, "canonical_content_hash", capture)
    sim = _shared_registration()
    real = _registration(local_path_enabled=False)
    records.clear()
    _ = sim.profile.semantic_content_hash
    _ = real.profile.semantic_content_hash
    assert records[0][0]["input_pixel_grid_policy"] == "original_artifact_grid_exif_absent_or_integer_1.v1"
    assert records[0][1] == "dexteleop.agent_skill.sim_object_localization_profile.v1"
    assert "input_pixel_grid_policy" not in records[1][0]
    assert records[1][1] == "dexteleop.agent_skill.real_object_localization_profile.v1"


@pytest.mark.parametrize("orientation", [
    0, 3, 4, 5, 7, 8, 9,
    pytest.param(True, id="boolean-one"), pytest.param(1.0, id="float-one"),
    pytest.param("1", id="string-one"),
])
def test_sim_rejects_remaining_exif_values_before_inference(tmp_path, monkeypatch, orientation):
    encoded = io.BytesIO()
    exif = Image.Exif()
    if type(orientation) is int:
        exif[274] = orientation
    Image.new("RGB", (12, 10)).save(encoded, format="JPEG", exif=exif)
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(localizer=localizer, registration=_shared_registration())
    with _store(tmp_path / "store") as store:
        request = _image_request(store, encoded.getvalue())
        if type(orientation) is not int:
            # Inject invalid decoded metadata, not a coerced JPEG SHORT value.
            original_open = Image.open

            def open_with_invalid_exif(*args, **kwargs):
                source = original_open(*args, **kwargs)
                source.getexif = lambda: {274: orientation}
                return source

            monkeypatch.setattr(localization.Image, "open", open_with_invalid_exif)
        with pytest.raises(ValueError, match="EXIF"):
            provider.localize_request(request, artifact_store=store)
    assert localizer.calls == []
    assert localizer.lifecycle_calls == []


@pytest.mark.parametrize("target_kind,current_mode", [
    ("real", "real_observe"), ("real", "real_active"), ("sim", "sim_active"),
])
def test_localization_executor_borrows_model_and_preserves_target_receipts(
    tmp_path, target_kind, current_mode,
):
    registration = _shared_registration(target_kind=target_kind, current_mode=current_mode)
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(localizer=localizer, registration=registration)
    # Real retains its public facade; sim borrows an already-started session owner.
    start = provider.start if target_kind == "real" else localizer.start
    close = provider.close if target_kind == "real" else localizer.close
    start()
    try:
        with _store(tmp_path / "store") as store:
            data = _jpeg_bytes()
            request = _image_request(store, data)
            executor = localization.build_object_localization_executor(
                registration=registration, provider=provider, artifact_store=store,
                runtime_session_id="session:test", current_station_revision=lambda: 1,
                max_retained_operations=4, clock=lambda: 2.,
            )
            for index in range(2):
                result = executor._read(request, SimpleNamespace(operation_id=f"operation:test:{index}"))
                registration.payload_validator.validate_execution_result(result.execution_result)
                registration.payload_validator.validate_execution_result_for_parameters(
                    request.parameters, result.execution_result)
                assert result.execution_result["input_media_content_hash"] == hashlib.sha256(data).hexdigest()
                assert result.evidence_refs[0].startswith(f"{target_kind}-object-localization:")
                assert len(result.output_artifact_bindings) == 1
                assert result.output_artifact_bindings[0].slot_id == "annotated_image"
            assert executor.executor_id == f"{target_kind}-passive-object-localization.v1"
            assert executor.port_id == f"agent-gateway:{target_kind}-passive-object-localization:session:test"
            assert localizer.calls == [((12, 10), "blue object", 3)] * 2
            assert localizer.lifecycle_calls == ["start"]
            assert localizer.started and not localizer.closed
            assert provider.started is (target_kind == "real")
    finally:
        close()
    assert localizer.lifecycle_calls == ["start", "close"]
    assert localizer.closed
    assert not provider.started


@pytest.mark.parametrize("target_kind,current_mode", [
    ("real", "real_observe"), ("real", "real_active"), ("sim", "sim_active"),
])
def test_localization_executor_accepts_equivalent_independent_registrations(
    tmp_path, target_kind, current_mode,
):
    registration_a = _shared_registration(target_kind=target_kind, current_mode=current_mode)
    registration_b = _shared_registration(target_kind=target_kind, current_mode=current_mode)
    assert registration_a is not registration_b
    assert registration_a.payload_validator is not registration_b.payload_validator
    assert (registration_a.profile, registration_a.skill, registration_a.binding,
            registration_a.availability_hash) == (
                registration_b.profile, registration_b.skill, registration_b.binding,
                registration_b.availability_hash)
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(localizer=localizer, registration=registration_a)
    with _store(tmp_path / "store") as store:
        request = _image_request(store, _jpeg_bytes())
        build = (localization.build_real_object_localization_executor if target_kind == "real"
                 else localization.build_object_localization_executor)
        executor = build(
            registration=registration_b, provider=provider, artifact_store=store,
            runtime_session_id="session:test", current_station_revision=lambda: 1,
            max_retained_operations=4, clock=lambda: 2.,
        )
        registration_b.payload_validator.validate_parameters(request.parameters)
        result = executor._read(request, SimpleNamespace(operation_id="operation:equivalent"))
        registration_b.payload_validator.validate_execution_result(result.execution_result)
        registration_b.payload_validator.validate_execution_result_for_parameters(
            request.parameters, result.execution_result)
        assert result.evidence_refs[0].startswith(f"{target_kind}-object-localization:")
    assert localizer.calls == [((12, 10), "blue object", 3)]
    assert localizer.lifecycle_calls == []


def test_localization_provider_rejects_foreign_model_implementation():
    localizer = _FakeLocalizer()
    with pytest.raises(ValueError, match="Localizer identity"):
        ObjectLocalizationProvider(
            localizer=localizer,
            registration=_shared_registration(inference_implementation_hash="foreign:implementation"),
        )
    assert localizer.calls == localizer.lifecycle_calls == []


@pytest.mark.parametrize("mismatch", [
    "provider", "runtime_session", "real_executor", "model_profile", "implementation", "owner", "availability",
])
def test_localization_executor_rejects_mixed_provider_or_session(tmp_path, mismatch):
    sim = _shared_registration()
    localizer = _FakeLocalizer()
    provider = ObjectLocalizationProvider(localizer=localizer, registration=sim)
    with _store(tmp_path / "store") as store:
        kwargs = {
            "registration": sim, "provider": provider, "artifact_store": store,
            "runtime_session_id": "session:test", "current_station_revision": lambda: 1,
            "max_retained_operations": 4,
        }
        build = localization.build_object_localization_executor
        if mismatch == "provider":
            kwargs["registration"] = _registration(local_path_enabled=False)
        elif mismatch == "runtime_session":
            kwargs["runtime_session_id"] = "foreign:test"
        elif mismatch == "real_executor":
            build = localization.build_real_object_localization_executor
        elif mismatch == "model_profile":
            kwargs["registration"] = _shared_registration(model_profile_hash="foreign:model-profile")
        elif mismatch == "implementation":
            kwargs["registration"] = _shared_registration(inference_implementation_hash="foreign:implementation")
        elif mismatch == "owner":
            kwargs["registration"] = replace(sim, binding=replace(sim.binding, target_binding_id="foreign:owner"))
        else:
            kwargs["registration"] = replace(sim, availability_hash="foreign:availability")
        with pytest.raises(ValueError):
            build(**kwargs)
    assert localizer.calls == localizer.lifecycle_calls == []


@pytest.mark.parametrize("build,required,default_name,default_value", [
    (build_real_object_localization_registration,
     ("model_profile_id", "model_profile_hash", "inference_implementation_hash", "local_path_enabled",
      "runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash",
      "execution_manifest_hash"), "current_mode", "real_observe"),
    (localization.build_real_object_localization_executor,
     ("registration", "provider", "artifact_store", "runtime_session_id", "current_station_revision",
      "max_retained_operations"), "clock", time.time),
])
def test_real_localization_builders_keep_keyword_only_signatures(
    build, required, default_name, default_value,
):
    parameters = inspect.signature(build).parameters
    assert tuple(parameters) == (*required, default_name)
    assert all(parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values())
    assert all(parameters[name].default is inspect.Parameter.empty for name in required)
    assert parameters[default_name].default == default_value

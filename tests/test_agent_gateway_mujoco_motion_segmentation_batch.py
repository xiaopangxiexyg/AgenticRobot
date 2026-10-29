"""Pure assembly/CLI boundaries; no Scene, model inference or device startup."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_gateway.profiles.mujoco import gateway_cli, motion_deployment, motion_factory
from agent_gateway.profiles.mujoco.motion_application import (
    MujocoAgentSkillMotionApplication,
)
from agent_gateway.profiles.mujoco.motion_gateway_session import (
    MujocoAgentTcpPoseGatewaySession,
    assemble_mujoco_agent_tcp_pose_gateway_session,
)
from agent_gateway.profiles.segmentation import build_segmentation_registration
from agent_gateway.runtime.artifact_store import RecoverableAgentArtifactStore

pytestmark = [pytest.mark.unit, pytest.mark.contract]
MODEL_IDENTITY = ("sam2:test", "sam2-model:test", "sam2-localizer:test")


@pytest.fixture
def bundle_pair(monkeypatch):
    from test_agent_gateway_mujoco_tcp_pose import (
        _build_no_model_contract_bundle_fixture,
    )

    application, _runtime, _direct, baseline, _manifest = (
        _build_no_model_contract_bundle_fixture(
            monkeypatch,
            enable_artifact_perception=True,
            segmentation_identity=MODEL_IDENTITY,
        )
    )
    kwargs = {
        "application": application,
        "enable_artifact_perception": True,
        "segmentation_model_profile_id": MODEL_IDENTITY[0],
        "segmentation_model_profile_hash": MODEL_IDENTITY[1],
        "segmentation_inference_implementation_hash": MODEL_IDENTITY[2],
    }
    enabled = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs,
        enable_segmentation_batch_v2=True,
    )
    return application, baseline, enabled, kwargs


@pytest.mark.parametrize("flag", ["enable_segmentation_batch_v2", "enable_segmentation_geometry_v3", "enable_object_localization"])
def test_batch_cli_flag_is_default_off_and_explicit(flag):
    parser = gateway_cli.build_parser()
    assert parser.get_default(flag) is False
    args = parser.parse_args(
        [
            "--socket-path",
            "/not-used/g.sock",
            "--task-config",
            "/not-used/task.json",
            "--artifact-root",
            "/not-used/artifacts",
            "--segmentation-asset-bundle",
            "/not-used/model",
            "--" + flag.replace("_", "-"),
        ]
    )
    assert getattr(args, flag) is True


@pytest.mark.parametrize(
    "dependencies",
    [
        [],
        ["--artifact-root", "/not-used/artifacts"],
        ["--segmentation-asset-bundle", "/not-used/model"],
    ],
)
@pytest.mark.parametrize("flag", ["--enable-segmentation-batch-v2", "--enable-segmentation-geometry-v3", "--enable-object-localization"])
def test_batch_cli_rejects_missing_dependency_before_reading_paths(
    monkeypatch, capsys, dependencies, flag
):
    monkeypatch.setattr(
        gateway_cli,
        "_absolute_existing_file",
        lambda *_: pytest.fail("resource path read"),
    )
    with pytest.raises(SystemExit) as error:
        gateway_cli.main(
            [
                "--socket-path",
                "/not-used/g.sock",
                "--task-config",
                "/not-used/task.json",
                flag,
                "--object-localization-backend", "sam2-grounding-dino",
                *dependencies,
            ]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "requires --segmentation-asset-bundle" in message or "requires --artifact-root" in message


@pytest.mark.parametrize("owner", ["factory", "deployment"])
@pytest.mark.parametrize("enabled", [None, 0, 1, "true", []])
@pytest.mark.parametrize("flag", ["enable_segmentation_batch_v2", "enable_segmentation_geometry_v3", "enable_object_localization"])
def test_batch_flag_is_strict_boolean_before_resources(monkeypatch, owner, enabled, flag):
    monkeypatch.setattr(
        motion_factory,
        "prepare_mujoco_sim_hil_target",
        lambda *_: pytest.fail("Scene preparation"),
    )
    monkeypatch.setattr(
        motion_deployment,
        "_exact_application",
        lambda *_: pytest.fail("application inspection"),
    )
    with pytest.raises(TypeError, match=flag + " must be boolean"):
        if owner == "factory":
            motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
                target_config=None,
                allowed_client_ids=("test",),
                **{flag: enabled},
            )
        else:
            motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
                application=None,
                **{flag: enabled},
            )


@pytest.mark.parametrize(
    "artifact,model", [(False, False), (False, True), (True, False)]
)
@pytest.mark.parametrize("owner", ["factory", "deployment"])
@pytest.mark.parametrize("version,flag", [
    ("2", "enable_segmentation_batch_v2"), ("3", "enable_segmentation_geometry_v3")])
def test_batch_dependencies_fail_before_resources(owner, artifact, model, version, flag):
    expected_message = (
        r"MuJoCo segmentation requires Artifact perception\."
        if owner == "deployment" and not artifact and model
        else f"Segmentation batch v{version} requires"
    )
    with pytest.raises(ValueError, match=expected_message):
        if owner == "factory":
            motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
                target_config=None,
                allowed_client_ids=("test",),
                **{flag: True},
                artifact_storage=object() if artifact else None,
                segmentation_localizer=object() if model else None,
            )
        else:
            motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
                application=None,
                **{flag: True},
                enable_artifact_perception=artifact,
                segmentation_model_profile_id=MODEL_IDENTITY[0] if model else None,
                segmentation_model_profile_hash=MODEL_IDENTITY[1] if model else None,
                segmentation_inference_implementation_hash=MODEL_IDENTITY[2]
                if model
                else None,
            )


@pytest.mark.parametrize("owner", ["factory", "deployment"])
@pytest.mark.parametrize("bad", [False, 17, " "])
def test_batch_rejects_untyped_model_identity_before_resources(tmp_path, owner, bad):
    with pytest.raises(ValueError, match="typed model identity strings"):
        if owner == "factory":
            localizer = SimpleNamespace(
                start=lambda: None,
                close=lambda: None,
                segment=lambda *_: None,
                encode_segment_artifacts=lambda *_: None,
                semantic_content_hash=MODEL_IDENTITY[2],
                model_spec=SimpleNamespace(
                    model_profile_id=bad, semantic_content_hash=MODEL_IDENTITY[1]
                ),
            )
            motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
                target_config=None,
                allowed_client_ids=("test",),
                enable_segmentation_batch_v2=True,
                artifact_storage=motion_factory.MujocoAgentArtifactStorageConfig(
                    root=tmp_path
                ),
                segmentation_localizer=localizer,
            )
        else:
            motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
                application=None,
                enable_segmentation_batch_v2=True,
                enable_artifact_perception=True,
                segmentation_model_profile_id=bad,
                segmentation_model_profile_hash=MODEL_IDENTITY[1],
                segmentation_inference_implementation_hash=MODEL_IDENTITY[2],
            )


def test_default_publications_match_explicit_disabled_and_v1_consumers_remain(
    bundle_pair,
):
    _application, baseline, enabled, kwargs = bundle_pair
    disabled = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs,
        enable_segmentation_batch_v2=False,
    )
    assert disabled.segmentation_batch is baseline.segmentation_batch is None
    for name in (
        "definition_catalog",
        "schema_catalog",
        "io_catalog",
        "route_manifest",
    ):
        assert (
            getattr(disabled, name).semantic_content_hash
            == getattr(baseline, name).semantic_content_hash
        )
    for name in ("segmentation", "placement"):
        old_registration = getattr(baseline, name)
        new_registration = getattr(enabled, name)
        assert old_registration is not None
        assert new_registration is not None
        for field in ("profile", "skill", "binding", "availability_hash"):
            assert getattr(new_registration, field) == getattr(old_registration, field)
    for name in ("depth_capture", "rgb_capture"):
        old_registration = getattr(baseline, name)
        new_registration = getattr(enabled, name)
        assert old_registration is not None
        assert new_registration is not None
        for field in (
            "profile",
            "registration",
            "deployment_binding",
            "availability_hash",
        ):
            assert getattr(new_registration, field) == getattr(old_registration, field)
    for definition in baseline.definition_catalog.skill_definitions:
        assert definition in enabled.definition_catalog.skill_definitions
    for contract in baseline.io_catalog.io_contracts:
        assert contract in enabled.io_catalog.io_contracts
    for old_entry in baseline.route_manifest.skill_routes:
        route = old_entry.route
        assert (
            enabled.route_manifest.route(
                route.skill_id,
                route.skill_version,
                route.route_id,
                route.route_version,
            )
            == old_entry
        )
        assert enabled.schema_catalog.entry(
            route.skill_id, route.skill_version
        ) == baseline.schema_catalog.entry(route.skill_id, route.skill_version)
        old_validator = baseline.payload_validators.validator_for(route)
        new_validator = enabled.payload_validators.validator_for(route)
        assert new_validator.validator_id == old_validator.validator_id
        old_validator.validate_route(route)
        new_validator.validate_route(route)
    registration = enabled.segmentation_batch
    assert registration.skill.definition.skill_version == "2"
    assert registration.skill.definition in enabled.definition_catalog.skill_definitions
    assert registration.skill.io_contract in enabled.io_catalog.io_contracts
    entry = enabled.route_manifest.route("perception.segment", "2")
    assert entry.route == registration.skill.routes[0]
    assert entry.deployment_binding == registration.binding.to_ref()
    assert (
        enabled.schema_catalog.entry("perception.segment", "2")
        == registration.skill.schema_entry
    )
    registration.payload_validator.validate_route(entry.route)


def test_batch_factory_rejects_untyped_artifact_storage_before_resources():
    with pytest.raises(TypeError, match="artifact_storage must be typed"):
        motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=None,
            allowed_client_ids=("test",),
            enable_segmentation_batch_v2=True,
            artifact_storage=object(),
            segmentation_localizer=object(),
        )


def test_session_rejects_missing_batch_executor_before_resource_checks(bundle_pair):
    _application, _baseline, bundle, _kwargs = bundle_pair
    session = object.__new__(MujocoAgentTcpPoseGatewaySession)
    session.contract_bundle = bundle
    session.segmentation_batch_executor = None
    with pytest.raises(
        ValueError, match="Segmentation batch registration and executor must coexist"
    ):
        session.__post_init__()


@pytest.mark.parametrize(
    "field",
    [
        "runtime_session_id",
        "control_domain_id",
        "target_binding_id",
        "station_manifest_hash",
        "execution_manifest_hash",
        "model_profile_hash",
    ],
)
def test_bundle_rejects_batch_bound_to_another_owner_or_model(bundle_pair, field):
    _application, baseline, enabled, _kwargs = bundle_pair
    original = baseline.segmentation
    arguments = {
        name: getattr(original.binding, name)
        for name in (
            "runtime_session_id",
            "control_domain_id",
            "target_binding_id",
            "station_manifest_hash",
            "execution_manifest_hash",
        )
    }
    arguments.update(
        {
            name: getattr(original.profile, name)
            for name in (
                "target_kind",
                "current_mode",
                "source_id_by_role",
                "model_profile_id",
                "model_profile_hash",
                "inference_implementation_hash",
            )
        }
    )
    arguments[field] = "another-owner-or-model"
    foreign = build_segmentation_registration(**arguments, skill_version="2")
    with pytest.raises(ValueError, match="Segmentation batch must share"):
        replace(enabled, segmentation_batch=foreign)


def test_bundle_rejects_batch_wrong_version_and_orphan_route(bundle_pair):
    _application, _baseline, enabled, _kwargs = bundle_pair
    with pytest.raises(ValueError, match="Segmentation batch v2 requires"):
        replace(enabled, segmentation_batch=enabled.segmentation)
    with pytest.raises(ValueError, match="batch Route requires its typed registration"):
        replace(enabled, segmentation_batch=None)


@pytest.mark.parametrize("geometry,paired,localization", [
    pytest.param(False, False, False, id="False-False"),
    pytest.param(True, False, False, id="True-False"),
    pytest.param(True, True, False, id="True-True"),
    pytest.param(False, False, True, id="localization"),
    pytest.param(True, True, True, id="paired-localization"),
])
def test_session_registers_batch_with_the_same_provider_and_one_model_owner(
    bundle_pair, tmp_path, monkeypatch, geometry, paired, localization
):
    from agent_gateway.profiles import segmentation_batch

    application, _baseline, bundle, kwargs = bundle_pair
    if geometry or localization:
        bundle = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
            **kwargs, enable_segmentation_batch_v2=True, enable_segmentation_geometry_v3=geometry,
            enable_rgbd_capture_v3=paired, enable_object_localization=localization)
    motion_application = object.__new__(MujocoAgentSkillMotionApplication)
    motion_application.__dict__.update(application.__dict__)
    motion_application._lock = threading.RLock()
    motion_application._started = motion_application._held = (
        motion_application._closed
    ) = False
    motion_application._active_operation_id = None
    motion_application.capture_passive_application_cut = lambda **_: None
    localizer = SimpleNamespace(
        semantic_content_hash=MODEL_IDENTITY[2],
        started=False,
        starts=0,
        closes=0,
        model_spec=SimpleNamespace(
            model_profile_id=MODEL_IDENTITY[0], semantic_content_hash=MODEL_IDENTITY[1]
        ),
        segment=lambda *_args, **_kwargs: pytest.fail("inference during assembly"),
        localize=lambda *_args, **_kwargs: pytest.fail("localization during assembly"),
        encode_segment_artifacts=lambda *_args, **_kwargs: pytest.fail(
            "encoding during assembly"
        ),
    )

    def start():
        localizer.starts += 1
        localizer.started = True

    localizer.start = start
    localizer.close = lambda: setattr(localizer, "closes", localizer.closes + 1)
    captured = {}
    build = segmentation_batch.build_segmentation_batch_executor

    def build_batch(**kwargs):
        captured[kwargs["registration"].skill.definition.skill_version] = kwargs
        return build(**kwargs)

    monkeypatch.setattr(
        segmentation_batch, "build_segmentation_batch_executor", build_batch
    )
    store = RecoverableAgentArtifactStore(
        root=tmp_path / "artifacts",
        quota_bytes=32 * 1024 * 1024,
        max_artifact_bytes=16 * 1024 * 1024,
        max_chunk_bytes=1024 * 1024,
    )
    try:
        session = assemble_mujoco_agent_tcp_pose_gateway_session(
            application=motion_application,
            contract_bundle=bundle,
            allowed_client_ids=("test",),
            artifact_store=store,
            segmentation_localizer=localizer,
        )
        assert captured["2"]["provider"] is session.segmentation_provider
        assert captured["2"]["provider"].registration is bundle.segmentation
        assert captured["2"]["registration"] is bundle.segmentation_batch
        ports = {
            (b.skill_id, b.skill_version): b.execution_port
            for b in session.service.executor_registry.bindings
        }
        assert ports["perception.segment", "1"] is session.segmentation_executor
        assert ports["perception.segment", "2"] is session.segmentation_batch_executor
        assert session.segmentation_executor is not session.segmentation_batch_executor
        if localization:
            assert session.localization_provider.localizer is localizer
            assert session.localization_provider.registration is bundle.localization
            assert ports["perception.localize_objects", "1"] is session.localization_executor
        else:
            assert session.localization_provider is session.localization_executor is None
        if geometry:
            assert captured["3"]["provider"] is session.segmentation_provider
            assert captured["3"]["registration"] is bundle.segmentation_geometry
            assert ports["perception.segment", "3"] is session.segmentation_geometry_executor
            assert session.segmentation_geometry_executor is not session.segmentation_batch_executor
            assert captured["3"]["rgb_capture_registration"] is (
                bundle.rgbd_capture.skill if paired else bundle.rgb_capture.registration)
            assert captured["3"]["rgb_capture_validator"] is (
                bundle.rgbd_capture.payload_validator if paired else bundle.rgb_capture.payload_validator)
            if paired:
                assert ports["perception.capture_rgb", "3"] is session.rgbd_capture_executor
                assert bundle.segmentation_geometry.profile.rgb_capture_route_hash == bundle.rgbd_capture.entry.route.semantic_content_hash
            assert captured["3"]["lookup_operation"]("unknown-operation") is None
            assert not any(e.route.skill_id == "perception.capture_surface_anchor"
                           for e in bundle.route_manifest.skill_routes)
        assert localizer.starts == 0
        session.start_perception_resources()
        session.start_perception_resources()
        assert localizer.starts == 1
    finally:
        if localizer.started:
            localizer.close()
        store.close()


@pytest.mark.parametrize("name", ["segmentation_batch", "segmentation_geometry"])
def test_batch_close_keeps_shared_dependencies_until_the_original_executor_drains(name):
    calls = []
    settled = False

    def wait_all(*, timeout_sec):
        assert timeout_sec == 65.0
        calls.append("wait_batch")
        return settled

    from test_agent_gateway_mujoco_tcp_pose import _gateway_close_outcomes
    _, outcome = _gateway_close_outcomes()
    session = SimpleNamespace(
        _close_lock=threading.Lock(),
        history_recorder=None,
        _close_outcome=None,
        service=SimpleNamespace(
            close=lambda **_: calls.append("service_close") or outcome
        ),
        observation_producer=SimpleNamespace(current_station_revision=0),
        state_executor=None,
        depth_executor=None,
        rgb_executor=None,
        rgbd_capture_executor=None,
        tcp_pose_comparison_executor=None,
        object_state_executor=None,
        spatial_goal_executor=None,
        object_goal_query_executor=None,
        localization_executor=None, tcp_continuation_query_executor=None, safe_tcp_pose_v2_executor=None,
        history_executor=None,
        segmentation_executor=None,
        segmentation_geometry_executor=None,
        grasp_observation_executor=None,
        fast_grasp_executor=None,
        fast_grasp_provider=None,
        fast_grasp_planner=None,
        placement_executor=None,
        surface_anchor_executors=(),
        candidate_query_executors=(),
        tool_entity_executors=(),
        segmentation_batch_executor=SimpleNamespace(wait_all=wait_all),
        contract_bundle=SimpleNamespace(
            surface_anchor_registrations=(), candidate_query_registrations=(),
            tool_entity_registrations=(),
            segmentation_batch=SimpleNamespace(
                skill=SimpleNamespace(
                    routes=(SimpleNamespace(max_operation_duration_ms=60_000),)
                )
            ),
        ),
        _segmentation_model_started=True,
        segmentation_localizer=SimpleNamespace(
            close=lambda: calls.append("model_close")
        ),
        _finish_lower_close=lambda _: calls.append("lower_close"),
        application=SimpleNamespace(closed=True),
        owns_source_observation_history_store=False,
        source_observation_history_store=None,
        _source_observation_history_store_closed=False,
        owns_artifact_store=True,
        artifact_store=SimpleNamespace(close=lambda: calls.append("store_close")),
        _artifact_store_closed=False,
    )
    if name == "segmentation_geometry":
        session.segmentation_geometry_executor = session.segmentation_batch_executor
        session.segmentation_batch_executor = None
        session.contract_bundle.segmentation_geometry = session.contract_bundle.segmentation_batch
        session.contract_bundle.segmentation_batch = None
    with pytest.raises(RuntimeError, match=name + "_executor"):
        MujocoAgentTcpPoseGatewaySession.close(session)
    assert calls == ["service_close", "wait_batch"]
    settled = True
    assert MujocoAgentTcpPoseGatewaySession.close(session) is outcome
    assert calls == [
        "service_close",
        "wait_batch",
        "wait_batch",
        "model_close",
        "lower_close",
        "store_close",
    ]


@pytest.mark.parametrize("include_v2", [False, True])
def test_geometry_publication_is_opt_in_and_preserves_previous_contracts(bundle_pair, include_v2):
    _application, baseline, batch, kwargs = bundle_pair
    disabled = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_segmentation_geometry_v3=False,
        enable_segmentation_batch_v2=include_v2)
    previous = batch if include_v2 else baseline
    assert disabled.segmentation_geometry is None
    for name in ("definition_catalog", "schema_catalog", "io_catalog", "route_manifest"):
        assert getattr(disabled, name).semantic_content_hash == getattr(previous, name).semantic_content_hash
    geometry = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_segmentation_geometry_v3=True,
        enable_segmentation_batch_v2=include_v2)
    registration = geometry.segmentation_geometry
    assert registration.skill.definition.skill_version == "3"
    assert (geometry.segmentation_batch is not None) == include_v2
    for old_entry in previous.route_manifest.skill_routes:
        route = old_entry.route
        assert geometry.route_manifest.route(route.skill_id, route.skill_version) == old_entry
        assert geometry.schema_catalog.entry(route.skill_id, route.skill_version) == previous.schema_catalog.entry(
            route.skill_id, route.skill_version)
    entry = geometry.route_manifest.route("perception.segment", "3")
    assert entry.route == registration.skill.routes[0]
    assert entry.deployment_binding == registration.binding.to_ref()
    assert registration.skill.definition in geometry.definition_catalog.skill_definitions
    assert registration.skill.io_contract in geometry.io_catalog.io_contracts
    assert geometry.schema_catalog.entry("perception.segment", "3") == registration.skill.schema_entry
    geometry.payload_validators.validator_for(entry.route).validate_route(entry.route)
    assert registration.skill.schema_entry.parameter_schema_hash != batch.segmentation_batch.skill.schema_entry.parameter_schema_hash
    slot, = registration.skill.io_contract.input_artifact_slots
    assert (slot.slot_id, slot.min_count, slot.max_count) == ("input_image", 1, 1)
    assert "rgb_capture_operation_id" in registration.skill.schema_entry.parameter_schema["required"]
    assert registration.skill.schema_entry.execution_result_schema_hash != batch.segmentation_batch.skill.schema_entry.execution_result_schema_hash
    with pytest.raises(ValueError, match="batch Route requires its typed registration"):
        replace(geometry, segmentation_geometry=None)
    with pytest.raises(ValueError, match="batch v3 requires"):
        replace(geometry, segmentation_geometry=geometry.segmentation)


@pytest.mark.parametrize("owner", ["factory", "session"])
def test_geometry_requires_producer_and_consumer_retention_before_resources(bundle_pair, tmp_path, owner):
    application, _baseline, _batch, kwargs = bundle_pair
    if owner == "session":
        bundle = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
            **kwargs, enable_segmentation_geometry_v3=True)
        motion_application = object.__new__(MujocoAgentSkillMotionApplication)
        motion_application.__dict__.update(application.__dict__)
        store = RecoverableAgentArtifactStore(root=tmp_path / "cas", quota_bytes=32 * 1024 * 1024,
            max_artifact_bytes=16 * 1024 * 1024)
        try:
            with pytest.raises(ValueError, match="retained producer and consumer"):
                assemble_mujoco_agent_tcp_pose_gateway_session(application=motion_application, contract_bundle=bundle,
                    allowed_client_ids=("test",), artifact_store=store, segmentation_localizer=object(),
                    max_retained_operations=1)
        finally:
            store.close()
    else:
        with pytest.raises(ValueError, match="retention.*producer and consumer"):
            motion_factory.build_mujoco_agent_tcp_pose_gateway_session(target_config=None,
                allowed_client_ids=("test",), segmentation_localizer=object(),
                artifact_storage=motion_factory.MujocoAgentArtifactStorageConfig(root=tmp_path),
                enable_segmentation_geometry_v3=True, max_retained_operations=1)


def test_paired_geometry_rejects_a_registration_bound_to_the_legacy_rgb_route(bundle_pair):
    _, _, _, kwargs = bundle_pair
    legacy = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_segmentation_geometry_v3=True)
    paired = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_segmentation_geometry_v3=True, enable_rgbd_capture_v3=True)
    with pytest.raises(ValueError, match="exact selected RGB producer Route"):
        replace(paired, segmentation_geometry=legacy.segmentation_geometry)


def test_localization_publication_is_opt_in_and_closes_existing_catalogs(bundle_pair):
    _, baseline, _, kwargs = bundle_pair
    disabled = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_object_localization=False,
    )
    assert disabled.localization is baseline.localization is None
    for name in ("definition_catalog", "schema_catalog", "io_catalog", "route_manifest"):
        assert getattr(disabled, name).semantic_content_hash == getattr(baseline, name).semantic_content_hash
    enabled = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_object_localization=True,
    )
    registration = enabled.localization
    entry = enabled.route_manifest.route("perception.localize_objects", "1")
    assert entry.route == registration.skill.routes[0]
    assert entry.deployment_binding == registration.binding.to_ref()
    assert entry.route.target_kind == "sim" and entry.route.supported_modes == ("sim_active",)
    assert entry.route.control_effect_class == "read_only" and not entry.route.may_move_robot
    assert registration.skill.definition in enabled.definition_catalog.skill_definitions
    assert registration.skill.io_contract in enabled.io_catalog.io_contracts
    assert enabled.schema_catalog.entry("perception.localize_objects", "1") == registration.skill.schema_entry
    enabled.payload_validators.validator_for(entry.route).validate_route(entry.route)
    for old_entry in baseline.route_manifest.skill_routes:
        route = old_entry.route
        assert enabled.route_manifest.route(route.skill_id, route.skill_version) == old_entry
    assert not registration.profile.local_path_enabled
    with pytest.raises(ValueError, match="typed registration"):
        replace(enabled, localization=None)


@pytest.mark.parametrize("artifact,model", [(False, False), (False, True), (True, False)])
@pytest.mark.parametrize("owner", ["factory", "deployment"])
def test_localization_dependencies_fail_before_resources(owner, artifact, model):
    with pytest.raises(ValueError, match="Object localization requires"):
        if owner == "factory":
            motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
                target_config=None, allowed_client_ids=("test",), enable_object_localization=True,
                artifact_storage=object() if artifact else None,
                segmentation_localizer=SimpleNamespace(localize=lambda: None) if model else None,
            )
        else:
            motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
                application=None, enable_object_localization=True, enable_artifact_perception=artifact,
                segmentation_model_profile_id=MODEL_IDENTITY[0] if model else None,
                segmentation_model_profile_hash=MODEL_IDENTITY[1] if model else None,
                segmentation_inference_implementation_hash=MODEL_IDENTITY[2] if model else None,
            )


def test_localization_service_binding_keeps_segmentation_and_motion_identities(bundle_pair):
    _application, _baseline, _batch, kwargs = bundle_pair
    legacy = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_object_localization=True)
    service = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(
        **kwargs, enable_object_localization=True, segmentation_inference_execution_mode="external_service")
    assert service.localization.binding.deployment_binding_version == "2"
    assert service.localization.binding.model_execution_owner == "external_model_service"
    assert service.localization.binding.semantic_content_hash != legacy.localization.binding.semantic_content_hash
    assert service.segmentation.profile == legacy.segmentation.profile
    assert service.segmentation.binding == legacy.segmentation.binding
    for entry in legacy.route_manifest.skill_routes:
        route = entry.route
        if route.skill_id != "perception.localize_objects":
            assert service.route_manifest.route(route.skill_id, route.skill_version) == entry


def test_localization_requires_model_method_and_full_chain_retention_before_scene():
    with pytest.raises(ValueError, match="shared segmentation localizer"):
        motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=None, allowed_client_ids=("test",), enable_object_localization=True,
            artifact_storage=object(), segmentation_localizer=object(),
        )
    with pytest.raises(ValueError, match="capture, localization, segmentation and anchor retention"):
        motion_factory.build_mujoco_agent_tcp_pose_gateway_session(
            target_config=None, allowed_client_ids=("test",), enable_object_localization=True,
            artifact_storage=object(), segmentation_localizer=SimpleNamespace(localize=lambda: None),
            enable_surface_anchor_v3=True, max_retained_operations=3,
        )


@pytest.mark.parametrize("field", [
    "runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash",
    "execution_manifest_hash", "model_profile_id", "model_profile_hash", "inference_implementation_hash",
])
def test_localization_bundle_rejects_foreign_model_or_owner(bundle_pair, field):
    from agent_gateway.profiles.localization_deployment import (
        build_object_localization_registration,
    )

    _, _, _, kwargs = bundle_pair
    bundle = motion_deployment.build_mujoco_agent_tcp_pose_contract_bundle(**kwargs, enable_object_localization=True)
    registration = bundle.localization
    values = {name: getattr(registration.binding, name) for name in (
        "runtime_session_id", "control_domain_id", "target_binding_id", "station_manifest_hash", "execution_manifest_hash",
    )}
    values.update({name: getattr(registration.profile, name) for name in (
        "model_profile_id", "model_profile_hash", "inference_implementation_hash",
    )})
    values[field] = "foreign-owner-or-model"
    foreign = build_object_localization_registration(
        **values, target_kind="sim", current_mode="sim_active", local_path_enabled=False,
    )
    with pytest.raises(ValueError, match="localization differs"):
        replace(bundle, localization=foreign)

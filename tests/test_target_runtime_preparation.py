from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.target_runtime import (
    TargetRuntimeHandleRollbackError,
    TargetRuntimeProviderRegistryEntry,
    TargetRuntimeStatusSnapshot,
)
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.target_runtime.integrations.rm65_policy_arm_qpos_candidate import (
    RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
    RM65PolicyArmQposCandidateTargetConfig,
    build_rm65_policy_arm_qpos_candidate_target_profile_binding,
)
from teleop_runtime.target_runtime import (
    TargetRuntimeProfileBinding,
    TargetRuntimeProfileCatalog,
    TargetRuntimeProfilePreflight,
    prepare_target_runtime_handle,
)

# The existing target-runtime contract fixture is deliberately arbitrary:
# one 7-DoF manipulator and one independent 1-DoF tool, with observations,
# operations, assets, and no left/right or RM65 assumptions.
from tests.test_target_runtime_contracts import (
    _FakeHandle,
    _FakeProvider,
    _manifest,
)


def _binding_for_manifest(
    provider_entry: TargetRuntimeProviderRegistryEntry,
) -> TargetRuntimeProfileBinding:
    manifest = _manifest()
    observation = manifest.execution_observation_capability
    assert observation is not None
    return TargetRuntimeProfileBinding(
        profile_id="fixture-single-manipulator-tool-profile-v1",
        profile_version="1",
        requirement_id_prefix="fixture:single-manipulator-tool",
        provider_entry=provider_entry,
        supported_clock_domain_ids=(manifest.clock_domain_id,),
        expected_manifest_version=manifest.manifest_version,
        expected_embodiment_definition_id=manifest.embodiment_definition_id,
        expected_embodiment_definition_hash=manifest.embodiment_definition_hash,
        expected_target_handler_id=manifest.target_handler_id,
        expected_action_schema_hash=manifest.action_schema_hash,
        expected_component_target_capability_hash=(
            manifest.component_target_capability.semantic_content_hash
        ),
        expected_component_ids=manifest.component_ids,
        expected_execution_observation_capability_hash=(
            observation.semantic_content_hash
        ),
        expected_operation_capability_hash_by_id={
            operation_id: capability.semantic_content_hash
            for operation_id, capability in manifest.operation_by_id.items()
        },
        expected_asset_reference_hash_by_id={
            asset_id: reference.semantic_content_hash
            for asset_id, reference in manifest.asset_by_id.items()
        },
    )


def _preflight_and_provider() -> tuple[
    TargetRuntimeProfilePreflight,
    _FakeProvider,
    dict[str, str],
]:
    manifest = _manifest()
    provider = _FakeProvider(manifest.provider, manifest)
    entry = TargetRuntimeProviderRegistryEntry(
        provider_identity=manifest.provider,
        loader=lambda: provider,
    )
    binding = _binding_for_manifest(entry)
    config = {"profile": "fixture"}
    preflight = TargetRuntimeProfileCatalog((binding,)).preflight(
        binding.profile_id,
        config,
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        bound_at=30.0,
    )
    return preflight, provider, config


def test_preparation_creates_only_an_inert_arbitrary_topology_handle() -> None:
    preflight, provider, config = _preflight_and_provider()

    preparation = prepare_target_runtime_handle(preflight, config)

    assert provider.created_handle_count == 1
    assert provider.rolled_back_handle_count == 0
    assert preparation.runtime_handle_created
    assert preparation.runtime_handle_acquisition_state == "pending"
    assert not preparation.target_operation_invoked
    assert not preparation.execution_submitted
    assert not preparation.confers_execution_authorization
    assert preparation.initial_status.lifecycle_state == "created"
    assert not preparation.initial_status.ready
    component_ids = (
        preparation.runtime_handle.resolved_target_runtime_manifest.component_ids
    )
    assert component_ids == (
        "manipulator/joint_position",
        "tool/grip",
    )
    handle = preparation.commit_runtime_handle()
    assert preparation.runtime_handle_acquisition_state == "committed"
    assert handle is preparation.runtime_handle
    with pytest.raises(RuntimeError, match="only be committed"):
        preparation.commit_runtime_handle()


def test_preparation_context_rolls_back_an_uncommitted_handle_once() -> None:
    preflight, provider, config = _preflight_and_provider()

    with prepare_target_runtime_handle(preflight, config) as preparation:
        assert preparation.runtime_handle_acquisition_state == "pending"

    assert preparation.runtime_handle_acquisition_state == "rolled_back"
    assert provider.rolled_back_handle_count == 1
    assert provider.rollback_reasons == [
        "target runtime preparation context exited uncommitted"
    ]
    assert not preparation.rollback_uncommitted_handle(
        reason="fixture repeated rollback"
    )


def test_preparation_context_does_not_rollback_a_committed_handle() -> None:
    preflight, provider, config = _preflight_and_provider()

    with prepare_target_runtime_handle(preflight, config) as preparation:
        handle = preparation.commit_runtime_handle()

    assert preparation.runtime_handle_acquisition_state == "committed"
    assert preparation.runtime_handle is handle
    assert provider.rolled_back_handle_count == 0


def test_preparation_rejects_config_that_resolves_another_manifest() -> None:
    manifest = _manifest()

    class ConfigSensitiveProvider(_FakeProvider):
        def resolve_target_runtime_manifest(
            self,
            config: object,
            *,
            runtime_session_id: str,
            control_domain_id: str,
        ):
            if config == {"profile": "fixture"}:
                return self.manifest
            return replace(
                self.manifest,
                embodiment_instance_hash="fixture-another-instance-hash-v1",
            )

    provider = ConfigSensitiveProvider(manifest.provider, manifest)
    entry = TargetRuntimeProviderRegistryEntry(
        provider_identity=manifest.provider,
        loader=lambda: provider,
    )
    binding = _binding_for_manifest(entry)
    preflight = TargetRuntimeProfileCatalog((binding,)).preflight(
        binding.profile_id,
        {"profile": "fixture"},
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        bound_at=30.0,
    )

    with pytest.raises(ValueError, match="another manifest"):
        prepare_target_runtime_handle(preflight, {"profile": "other"})
    assert provider.created_handle_count == 0


def test_preparation_rejects_noncreated_handle_without_operations() -> None:
    preflight, provider, config = _preflight_and_provider()
    manifest = preflight.resolved_manifest
    handle = _FakeHandle(manifest)

    def capture_ready_status() -> TargetRuntimeStatusSnapshot:
        return TargetRuntimeStatusSnapshot(
            manifest_hash=manifest.semantic_content_hash,
            runtime_session_id=manifest.runtime_session_id,
            control_domain_id=manifest.control_domain_id,
            target_handler_id=manifest.target_handler_id,
            lifecycle_state="ready",
            execution_epoch=0,
            execution_epoch_source_id="fixture-epoch-registry",
            ready=True,
            fault_latched=False,
            status_sequence=0,
            captured_at=30.0,
            clock_domain_id=manifest.clock_domain_id,
            reason="fixture provider activated during construction",
        )

    handle.target_runtime_status_provider.capture_target_runtime_status = (
        capture_ready_status
    )

    def create_ready_handle(*args: object, **kwargs: object) -> _FakeHandle:
        provider.created_handle_count += 1
        return handle

    provider.create_target_runtime_handle = create_ready_handle

    with pytest.raises(ValueError, match="must begin in created state"):
        prepare_target_runtime_handle(preflight, config)
    assert provider.created_handle_count == 1
    assert provider.rolled_back_handle_count == 1
    assert provider.rollback_reasons == [
        "target runtime initial-status validation failed"
    ]


def test_preparation_rolls_back_when_initial_status_capture_fails() -> None:
    preflight, provider, config = _preflight_and_provider()
    manifest = preflight.resolved_manifest
    handle = _FakeHandle(manifest)

    def fail_status_capture() -> TargetRuntimeStatusSnapshot:
        raise RuntimeError("fixture status unavailable")

    handle.target_runtime_status_provider.capture_target_runtime_status = (
        fail_status_capture
    )

    def create_handle(*args: object, **kwargs: object) -> _FakeHandle:
        provider.created_handle_count += 1
        return handle

    provider.create_target_runtime_handle = create_handle

    with pytest.raises(RuntimeError, match="fixture status unavailable"):
        prepare_target_runtime_handle(preflight, config)
    assert provider.created_handle_count == 1
    assert provider.rolled_back_handle_count == 1
    assert provider.rollback_reasons == [
        "target runtime initial-status validation failed"
    ]


def test_preparation_context_rolls_back_when_caller_fails() -> None:
    preflight, provider, config = _preflight_and_provider()

    with pytest.raises(RuntimeError, match="fixture caller failed"):
        with prepare_target_runtime_handle(preflight, config):
            raise RuntimeError("fixture caller failed")

    assert provider.rolled_back_handle_count == 1
    assert provider.rollback_reasons == [
        "target runtime preparation context failed before handle commitment"
    ]


def test_preparation_context_retains_caller_and_rollback_failures() -> None:
    preflight, provider, config = _preflight_and_provider()

    def fail_rollback(*args: object, **kwargs: object) -> None:
        raise OSError("fixture cleanup failed")

    provider.rollback_target_runtime_handle_construction = fail_rollback

    with pytest.raises(TargetRuntimeHandleRollbackError) as raised:
        with prepare_target_runtime_handle(preflight, config):
            raise RuntimeError("fixture caller failed")

    assert isinstance(raised.value.triggering_error, RuntimeError)
    assert str(raised.value.triggering_error) == "fixture caller failed"
    assert isinstance(raised.value.rollback_error, OSError)
    assert raised.value.acquisition.state == "pending"


def test_preflight_only_profile_fails_before_loading_provider() -> None:
    binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()
    real_config = RM65PolicyArmQposCandidateTargetConfig(
        control_execution_manifest=(
            build_rm65_policy_arm_qpos_candidate_manifest()
        ),
        embodiment_instance_id="lab-a-rm65-pair-01",
        embodiment_instance_hash="fixture-rm65-instance-hash-v1",
        clock_domain_id="control.monotonic",
    )
    preflight = TargetRuntimeProfileCatalog((binding,)).preflight(
        RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
        real_config,
        runtime_session_id="session-rm65-preflight-only",
        control_domain_id="rm65-arms",
        bound_at=30.0,
    )
    load_count = 0

    def fail_if_loaded() -> object:
        nonlocal load_count
        load_count += 1
        raise AssertionError("preflight-only preparation must not load provider")

    guarded_binding = replace(
        binding,
        provider_entry=TargetRuntimeProviderRegistryEntry(
            provider_identity=binding.provider_entry.provider_identity,
            loader=fail_if_loaded,
        ),
    )
    guarded_preflight = replace(preflight, profile_binding=guarded_binding)

    with pytest.raises(RuntimeError, match="preflight-only"):
        prepare_target_runtime_handle(guarded_preflight, real_config)
    assert load_count == 0

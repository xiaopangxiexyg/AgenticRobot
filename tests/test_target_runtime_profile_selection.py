from __future__ import annotations

from dataclasses import replace

import pytest

from robot_motion.target_runtime import (
    TargetRuntimeOperationCapability,
    TargetRuntimeProviderRegistryEntry,
)
from robot_motion.action.execution_observation import (
    ExecutionObservationCapabilitySpec,
)
from teleop_runtime.deploy.rm65_candidate_manifest import (
    build_rm65_policy_arm_qpos_candidate_manifest,
)
from teleop_runtime.deploy.rm65_target_runtime_provider import (
    RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
    RM65PolicyArmQposCandidateTargetConfig,
    RM65PolicyArmQposCandidateTargetProvider,
    build_rm65_policy_arm_qpos_candidate_embodiment_definition_hash,
    build_rm65_policy_arm_qpos_candidate_target_profile_binding,
)
from teleop_runtime.target_profiles import (
    TargetRuntimeProfileCatalog,
    TargetRuntimeProfilePreflight,
)


def _config(
    *,
    clock_domain_id: str = "control.monotonic",
) -> RM65PolicyArmQposCandidateTargetConfig:
    return RM65PolicyArmQposCandidateTargetConfig(
        control_execution_manifest=(
            build_rm65_policy_arm_qpos_candidate_manifest()
        ),
        embodiment_instance_id="lab-a-rm65-pair-01",
        embodiment_instance_hash="instance-profile-hash:lab-a-rm65-pair-01",
        clock_domain_id=clock_domain_id,
    )


def test_rm65_profile_binding_pins_complete_preflight_surface() -> None:
    binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()

    assert binding.profile_id == (
        RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID
    )
    assert binding.expected_embodiment_definition_hash == (
        build_rm65_policy_arm_qpos_candidate_embodiment_definition_hash()
    )
    assert binding.supported_clock_domain_ids == ("control.monotonic",)
    assert binding.expected_manifest_version == "1"
    assert binding.expected_component_ids == ("left_arm", "right_arm")
    assert binding.expected_execution_observation_capability_hash is None
    assert dict(binding.expected_operation_capability_hash_by_id) == {}
    assert set(binding.expected_asset_reference_hash_by_id) == {
        "legacy-control-execution-manifest",
        "legacy-dispatch-inventory",
        "legacy-safety-migration-matrix",
        "legacy-worker-drain-plan",
    }


def test_profile_hash_excludes_loader_object_but_binds_provider_identity() -> None:
    binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()
    entry = binding.provider_entry
    alternate_entry = TargetRuntimeProviderRegistryEntry(
        provider_identity=entry.provider_identity,
        loader=lambda: entry.loader(),
    )
    alternate_loader = replace(binding, provider_entry=alternate_entry)
    another_provider = replace(
        entry.provider_identity,
        provider_id="fixture.another-provider",
        implementation_contract_hash="fixture-another-provider-contract",
    )
    another_identity = replace(
        binding,
        provider_entry=TargetRuntimeProviderRegistryEntry(
            provider_identity=another_provider,
            loader=entry.loader,
        ),
    )

    assert alternate_loader.semantic_content_hash == binding.semantic_content_hash
    assert another_identity.semantic_content_hash != binding.semantic_content_hash


def test_catalog_lists_without_loading_and_loads_only_selected_profile() -> None:
    binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()
    selected_load_count = 0
    decoy_load_count = 0

    def load_selected() -> object:
        nonlocal selected_load_count
        selected_load_count += 1
        return binding.provider_entry.loader()

    def load_decoy() -> object:
        nonlocal decoy_load_count
        decoy_load_count += 1
        raise AssertionError("unselected provider must not load")

    selected = replace(
        binding,
        provider_entry=TargetRuntimeProviderRegistryEntry(
            provider_identity=binding.provider_entry.provider_identity,
            loader=load_selected,
        ),
    )
    decoy_identity = replace(
        binding.provider_entry.provider_identity,
        provider_id="fixture.decoy-provider",
        implementation_contract_hash="fixture-decoy-provider-contract",
    )
    decoy = replace(
        binding,
        profile_id="fixture-decoy-profile",
        provider_entry=TargetRuntimeProviderRegistryEntry(
            provider_identity=decoy_identity,
            loader=load_decoy,
        ),
    )
    catalog = TargetRuntimeProfileCatalog((decoy, selected))

    assert catalog.profile_ids() == (
        "fixture-decoy-profile",
        RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
    )
    assert tuple(item.profile_id for item in catalog.profile_bindings()) == (
        "fixture-decoy-profile",
        RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
    )
    assert selected_load_count == decoy_load_count == 0

    with pytest.raises(KeyError, match="Unknown target runtime profile"):
        catalog.preflight(
            "missing-profile",
            _config(),
            runtime_session_id="session-preflight-1",
            control_domain_id="rm65-arms",
            bound_at=100.0,
        )
    assert selected_load_count == decoy_load_count == 0

    result = catalog.preflight(
        RM65_POLICY_ARM_QPOS_CANDIDATE_TARGET_PROFILE_ID,
        _config(),
        runtime_session_id="session-preflight-1",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )

    assert selected_load_count == 1
    assert decoy_load_count == 0
    assert result.profile_binding is selected
    assert result.requirement.required_component_ids == (
        "left_arm",
        "right_arm",
    )
    assert result.binding_receipt.bound_component_ids == (
        "left_arm",
        "right_arm",
    )
    assert not result.runtime_handle_created
    assert not result.confers_execution_authorization


def test_profile_preflight_binds_nonempty_operation_and_observation_surface() -> None:
    base_binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()
    observation = ExecutionObservationCapabilitySpec(
        capability_id="fixture-rm65-observation-v1",
        target_handler_id=base_binding.expected_target_handler_id,
        clock_domain_id="control.monotonic",
        supported_reference_modes=("off",),
        supported_measurement_modes=("off", "full_frames"),
        representation_by_component={
            component_id: "joint_vector"
            for component_id in base_binding.expected_component_ids
        },
        schema_hash_by_component={
            component_id: f"fixture-measurement:{component_id}:v1"
            for component_id in base_binding.expected_component_ids
        },
        measurement_stage_id="fixture-measured-component-state",
    )
    diagnostics = TargetRuntimeOperationCapability(
        operation_id="embodiment.read_diagnostics",
        operation_version="1",
        request_schema_hash="fixture-diagnostics-request-v1",
        receipt_schema_hash="fixture-diagnostics-receipt-v1",
        allowed_lifecycle_states=("created", "holding", "ready"),
        motion_effect="none",
        requires_operator_confirmation=False,
        required_permit_kind="none",
        required_permit_operation=None,
    )
    base_provider = RM65PolicyArmQposCandidateTargetProvider()

    class ExtendedPreflightProvider:
        provider_identity = base_provider.provider_identity

        def resolve_target_runtime_manifest(
            self,
            config: object,
            *,
            runtime_session_id: str,
            control_domain_id: str,
        ):
            manifest = base_provider.resolve_target_runtime_manifest(
                config,
                runtime_session_id=runtime_session_id,
                control_domain_id=control_domain_id,
            )
            return replace(
                manifest,
                execution_observation_capability=observation,
                operation_capabilities=(diagnostics,),
            )

        def create_target_runtime_handle(self, *args: object, **kwargs: object):
            raise AssertionError("profile preflight must not create a handle")

    binding = replace(
        base_binding,
        profile_id="fixture-rm65-observation-operation-profile",
        provider_entry=TargetRuntimeProviderRegistryEntry(
            provider_identity=base_provider.provider_identity,
            loader=ExtendedPreflightProvider,
        ),
        expected_execution_observation_capability_hash=(
            observation.semantic_content_hash
        ),
        expected_operation_capability_hash_by_id={
            diagnostics.operation_id: diagnostics.semantic_content_hash
        },
    )

    result = TargetRuntimeProfileCatalog((binding,)).preflight(
        binding.profile_id,
        _config(),
        runtime_session_id="session-preflight-1",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )

    assert result.requirement.required_operation_ids == (
        "embodiment.read_diagnostics",
    )
    assert result.binding_receipt.bound_operation_ids == (
        "embodiment.read_diagnostics",
    )
    assert result.resolved_manifest.execution_observation_capability == (
        observation
    )
    assert result.binding_receipt.execution_observation_binding is None


@pytest.mark.parametrize(
    ("changes", "mismatch"),
    (
        (
            {"expected_manifest_version": "2"},
            "manifest_version",
        ),
        (
            {"expected_embodiment_definition_hash": "drift-definition"},
            "embodiment_definition_hash",
        ),
        (
            {"expected_target_handler_id": "drift-target-handler"},
            "target_handler_id",
        ),
        (
            {"expected_action_schema_hash": "drift-action-schema"},
            "action_schema_hash",
        ),
        (
            {
                "expected_component_target_capability_hash": (
                    "drift-target-capability"
                )
            },
            "component_target_capability_hash",
        ),
        (
            {"expected_component_ids": ("left_arm",)},
            "component_ids",
        ),
        (
            {
                "expected_execution_observation_capability_hash": (
                    "unexpected-observation"
                )
            },
            "execution_observation_capability_hash",
        ),
        (
            {
                "expected_operation_capability_hash_by_id": {
                    "embodiment.home": "unexpected-operation"
                }
            },
            "operation_capability_hash_by_id",
        ),
        (
            {"expected_asset_reference_hash_by_id": {}},
            "asset_reference_hash_by_id",
        ),
        (
            {"supported_clock_domain_ids": ("another.clock",)},
            "clock_domain_id",
        ),
    ),
)
def test_profile_preflight_fails_closed_on_resolved_surface_drift(
    changes: dict[str, object],
    mismatch: str,
) -> None:
    binding = replace(
        build_rm65_policy_arm_qpos_candidate_target_profile_binding(),
        **changes,
    )
    catalog = TargetRuntimeProfileCatalog((binding,))

    with pytest.raises(ValueError, match=mismatch):
        catalog.preflight(
            binding.profile_id,
            _config(),
            runtime_session_id="session-preflight-1",
            control_domain_id="rm65-arms",
            bound_at=100.0,
        )


def test_profile_catalog_rejects_duplicate_or_invalid_bindings() -> None:
    binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()
    with pytest.raises(KeyError, match="already registered"):
        TargetRuntimeProfileCatalog((binding, binding))
    with pytest.raises(ValueError, match="supported_clock_domain_id"):
        replace(binding, supported_clock_domain_ids=())
    with pytest.raises(ValueError, match="expected_component_id"):
        replace(
            binding,
            expected_component_ids=("left_arm", "left_arm"),
        )
    with pytest.raises(ValueError, match="unique after normalization"):
        replace(
            binding,
            expected_asset_reference_hash_by_id={
                "asset": "hash-a",
                " asset ": "hash-b",
            },
        )


def test_profile_preflight_rejects_tampered_binding_receipt() -> None:
    binding = build_rm65_policy_arm_qpos_candidate_target_profile_binding()
    result = TargetRuntimeProfileCatalog((binding,)).preflight(
        binding.profile_id,
        _config(),
        runtime_session_id="session-preflight-1",
        control_domain_id="rm65-arms",
        bound_at=100.0,
    )
    tampered_receipt = replace(
        result.binding_receipt,
        bound_component_ids=("left_arm",),
    )

    with pytest.raises(ValueError, match="another binding"):
        TargetRuntimeProfilePreflight(
            profile_binding=result.profile_binding,
            resolved_manifest=result.resolved_manifest,
            requirement=result.requirement,
            binding_receipt=tampered_receipt,
        )

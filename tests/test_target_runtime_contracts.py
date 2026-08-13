from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.action_schema import ActionPayloadSchema
from robot_motion.action import (
    ActionComponentCatalogV2,
    ComponentActionSchema,
    ComponentTargetCapabilitySpec,
    ControlComponentSpecV2,
    TargetComponentRouteSpec,
    TargetResourceCombinationSpec,
)
from robot_motion.action.execution_observation import (
    ExecutionObservationCapabilitySpec,
    ExecutionObservationPlan,
)
from robot_motion.runtime.dispatch_owner import (
    DispatchOwnerRegistry,
    LifecycleControlAuthorization,
    LifecycleControlDispatchPermit,
)
from robot_motion.target_runtime import (
    TARGET_RUNTIME_PROVIDER_API_V1,
    ResolvedTargetRuntimeManifest,
    TargetRuntimeAssetReference,
    TargetRuntimeOperationCapability,
    TargetRuntimeOperationReceipt,
    TargetRuntimeOperationRequest,
    TargetRuntimeProviderIdentity,
    TargetRuntimeProviderRegistry,
    TargetRuntimeProviderRegistryEntry,
    TargetRuntimeRequirement,
    TargetRuntimeStatusSnapshot,
    bind_target_runtime_requirement,
    validate_target_runtime_operation_invocation,
    validate_target_runtime_operation_receipt,
)


def _joint_schema() -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id="generic-joint-position-7-v1",
        schema_version="1",
        representation_family="joint_position",
        dtype="float32",
        shape=(7,),
        coordinate_names=tuple(f"joint_{index}" for index in range(7)),
        units=("rad",) * 7,
        value_semantics="absolute_target",
    )


def _tool_schema() -> ActionPayloadSchema:
    return ActionPayloadSchema(
        schema_id="generic-tool-grip-1-v1",
        schema_version="1",
        representation_family="joint_position",
        dtype="float32",
        shape=(1,),
        coordinate_names=("grip",),
        units=("normalized_0_1",),
        value_semantics="absolute_target",
    )


def _action_schema() -> ComponentActionSchema:
    joint = _joint_schema()
    tool = _tool_schema()
    return ComponentActionSchema(
        schema_id="single-manipulator-tool-v1",
        schema_version="1",
        component_catalog=ActionComponentCatalogV2(
            catalog_id="single-manipulator-tool-catalog-v1",
            catalog_version="1",
            components=(
                ControlComponentSpecV2(
                    component_id="manipulator/joint_position",
                    controlled_entity_id="manipulator",
                    command_channel_id="manipulator/joint_position",
                    selection_group_id="manipulator_motion",
                    resource_claims=("manipulator/joints",),
                    accepted_payload_schema_hashes=(
                        joint.semantic_content_hash,
                    ),
                ),
                ControlComponentSpecV2(
                    component_id="tool/grip",
                    controlled_entity_id="tool",
                    command_channel_id="tool/grip",
                    selection_group_id="tool_motion",
                    resource_claims=("tool/actuator",),
                    accepted_payload_schema_hashes=(
                        tool.semantic_content_hash,
                    ),
                ),
            ),
        ),
        payload_schema_by_component={
            "manipulator/joint_position": joint,
            "tool/grip": tool,
        },
    )


def _target_capability() -> ComponentTargetCapabilitySpec:
    schema = _action_schema()
    routes = tuple(
        TargetComponentRouteSpec(
            component_id=component_id,
            input_payload_schema_hash=(
                schema.payload_schema_by_component[
                    component_id
                ].semantic_content_hash
            ),
            handler_contract_id="generic-single-target-handler-v1",
            handler_contract_hash="generic-single-target-handler-hash-v1",
            output_target_schema_hash=(
                f"resolved-target-schema:{component_id}:v1"
            ),
        )
        for component_id in schema.component_ids
    )
    resources = (
        TargetResourceCombinationSpec(
            resource_id="manipulator/joints",
            participant_component_ids=("manipulator/joint_position",),
            allowed_active_component_sets=(("manipulator/joint_position",),),
            handler_contract_id="generic-single-target-handler-v1",
            handler_contract_hash="generic-single-target-handler-hash-v1",
        ),
        TargetResourceCombinationSpec(
            resource_id="tool/actuator",
            participant_component_ids=("tool/grip",),
            allowed_active_component_sets=(("tool/grip",),),
            handler_contract_id="generic-single-target-handler-v1",
            handler_contract_hash="generic-single-target-handler-hash-v1",
        ),
    )
    return ComponentTargetCapabilitySpec(
        capability_id="generic-single-target-v1",
        capability_version="1",
        action_schema=schema,
        component_routes=routes,
        resource_combinations=resources,
    )


def _observation_capability() -> ExecutionObservationCapabilitySpec:
    return ExecutionObservationCapabilitySpec(
        capability_id="generic-single-observation-v1",
        target_handler_id="target-handler:single-manipulator:v1",
        clock_domain_id="clock:target-runtime",
        supported_reference_modes=("off",),
        supported_measurement_modes=("off", "full_frames"),
        representation_by_component={
            "manipulator/joint_position": "joint_vector",
            "tool/grip": "joint_vector",
        },
        schema_hash_by_component={
            "manipulator/joint_position": "measurement-schema:joints7:v1",
            "tool/grip": "measurement-schema:grip1:v1",
        },
        measurement_stage_id="measured-component-state",
    )


def _home_operation() -> TargetRuntimeOperationCapability:
    return TargetRuntimeOperationCapability(
        operation_id="embodiment.home",
        operation_version="1",
        request_schema_hash="operation-request:home:v1",
        receipt_schema_hash="operation-receipt:home:v1",
        allowed_lifecycle_states=("holding",),
        motion_effect="required",
        requires_operator_confirmation=True,
        required_permit_kind="lifecycle_control",
        required_permit_operation="home",
        resource_claims=("manipulator/joints",),
    )


def _manifest() -> ResolvedTargetRuntimeManifest:
    return ResolvedTargetRuntimeManifest(
        manifest_id="resolved-target:single-manipulator:session-1",
        manifest_version="1",
        runtime_session_id="session-1",
        control_domain_id="control-domain-1",
        clock_domain_id="clock:target-runtime",
        embodiment_definition_id="single-manipulator-tool-v1",
        embodiment_definition_hash="embodiment-definition-hash-v1",
        embodiment_instance_id="lab-a-single-manipulator-01",
        embodiment_instance_hash="embodiment-instance-hash-v1",
        target_handler_id="target-handler:single-manipulator:v1",
        provider=TargetRuntimeProviderIdentity(
            provider_id="single-manipulator-provider",
            provider_version="1.2.0",
            provider_api=TARGET_RUNTIME_PROVIDER_API_V1,
            implementation_contract_hash="provider-contract-hash-v1",
            package_name="dexteleop-single-manipulator",
            package_version="1.2.0",
            runtime_handle_mode="inert_handle",
            dependency_lock_hash="provider-lock-hash-v1",
        ),
        component_target_capability=_target_capability(),
        execution_observation_capability=_observation_capability(),
        operation_capabilities=(
            _home_operation(),
            TargetRuntimeOperationCapability(
                operation_id="embodiment.read_diagnostics",
                operation_version="1",
                request_schema_hash="operation-request:diagnostics:v1",
                receipt_schema_hash="operation-receipt:diagnostics:v1",
                allowed_lifecycle_states=("created", "holding", "ready"),
                motion_effect="none",
                requires_operator_confirmation=False,
                required_permit_kind="none",
                required_permit_operation=None,
            ),
        ),
        asset_references=(
            TargetRuntimeAssetReference(
                asset_id="robot-description",
                asset_kind="urdf",
                content_hash="urdf-content-hash-v1",
            ),
            TargetRuntimeAssetReference(
                asset_id="tool-description",
                asset_kind="tool_profile",
                content_hash="tool-content-hash-v1",
            ),
        ),
    )


def _status(
    manifest: ResolvedTargetRuntimeManifest,
    *,
    lifecycle_state: str,
    execution_epoch: int = 4,
) -> TargetRuntimeStatusSnapshot:
    return TargetRuntimeStatusSnapshot(
        manifest_hash=manifest.semantic_content_hash,
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        target_handler_id=manifest.target_handler_id,
        lifecycle_state=lifecycle_state,
        execution_epoch=execution_epoch,
        execution_epoch_source_id="execution-epoch-registry:session-1",
        ready=lifecycle_state == "ready",
        fault_latched=lifecycle_state == "faulted",
        status_sequence=8,
        captured_at=20.0,
        clock_domain_id=manifest.clock_domain_id,
        reason="fixture status",
        source_event_ids=("event:status:8",),
    )


def _home_request(
    manifest: ResolvedTargetRuntimeManifest,
    *,
    execution_epoch: int = 4,
    confirmation: str | None = "operator-confirmation:home:1",
) -> TargetRuntimeOperationRequest:
    return TargetRuntimeOperationRequest(
        request_id="operation-request:home:1",
        manifest_hash=manifest.semantic_content_hash,
        operation_id="embodiment.home",
        request_schema_hash="operation-request:home:v1",
        payload={"target_profile_id": "station-home-v2"},
        target_execution_epoch=execution_epoch,
        requested_at=21.0,
        clock_domain_id=manifest.clock_domain_id,
        source_event_ids=("operator-intent:home:1",),
        operator_confirmation_event_id=confirmation,
    )


def _home_permit(
    manifest: ResolvedTargetRuntimeManifest,
    *,
    execution_epoch: int = 4,
) -> LifecycleControlDispatchPermit:
    registry = DispatchOwnerRegistry(
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        initial_authority_epoch=0,
        initial_execution_epoch=execution_epoch,
        clock=lambda: 20.5,
    )
    return registry.issue_lifecycle_control_permit(
        LifecycleControlAuthorization(
            authorization_event_id="operator-confirmation:home:1",
            runtime_session_id=manifest.runtime_session_id,
            control_domain_id=manifest.control_domain_id,
            lifecycle_state_machine_id="lifecycle:single-manipulator:1",
            requester_kind="lifecycle_state_machine",
            operation="home",
            target_execution_epoch=execution_epoch,
            operator_or_safety_policy_id="operator-policy:lab-a:v1",
        )
    )


def test_manifest_is_component_keyed_and_order_canonical() -> None:
    manifest = _manifest()
    reordered = replace(
        manifest,
        operation_capabilities=tuple(reversed(manifest.operation_capabilities)),
        asset_references=tuple(reversed(manifest.asset_references)),
    )

    assert manifest.component_ids == (
        "manipulator/joint_position",
        "tool/grip",
    )
    assert manifest.semantic_content_hash == reordered.semantic_content_hash
    assert tuple(manifest.operation_by_id) == (
        "embodiment.home",
        "embodiment.read_diagnostics",
    )
    assert tuple(manifest.asset_by_id) == (
        "robot-description",
        "tool-description",
    )


def test_requirement_binds_action_operations_and_measurement_plan() -> None:
    manifest = _manifest()
    plan = ExecutionObservationPlan(
        plan_id="collection-state-plan-v1",
        measurement_mode="full_frames",
        component_ids=manifest.component_ids,
        required_representation_by_component={
            component_id: "joint_vector"
            for component_id in manifest.component_ids
        },
    )
    requirement = TargetRuntimeRequirement(
        requirement_id="formal-collection-target-v1",
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        clock_domain_id=manifest.clock_domain_id,
        embodiment_instance_id=manifest.embodiment_instance_id,
        target_handler_id=manifest.target_handler_id,
        action_schema_hash=manifest.action_schema_hash,
        required_component_ids=manifest.component_ids,
        required_operation_ids=("embodiment.home",),
        observation_plan=plan,
    )

    receipt = bind_target_runtime_requirement(
        requirement,
        manifest,
        bound_at=10.0,
    )

    assert receipt.manifest_hash == manifest.semantic_content_hash
    assert receipt.bound_component_ids == manifest.component_ids
    assert receipt.bound_operation_ids == ("embodiment.home",)
    assert receipt.execution_observation_binding is not None
    assert receipt.execution_observation_binding.measurement_mode == "full_frames"
    assert receipt.execution_observation_binding.component_ids == manifest.component_ids


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("action_schema_hash", "another-schema", "action schema"),
        (
            "embodiment_instance_id",
            "another-instance",
            "identity mismatch",
        ),
        (
            "clock_domain_id",
            "another-clock",
            "identity mismatch",
        ),
        (
            "required_operation_ids",
            ("embodiment.zero_force",),
            "required operations",
        ),
    ),
)
def test_requirement_fails_closed_on_incompatible_target(
    field: str,
    value: object,
    match: str,
) -> None:
    manifest = _manifest()
    requirement = TargetRuntimeRequirement(
        requirement_id="deployment-target-v1",
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        clock_domain_id=manifest.clock_domain_id,
        embodiment_instance_id=manifest.embodiment_instance_id,
        target_handler_id=manifest.target_handler_id,
        action_schema_hash=manifest.action_schema_hash,
        required_component_ids=manifest.component_ids,
    )

    with pytest.raises(ValueError, match=match):
        bind_target_runtime_requirement(
            replace(requirement, **{field: value}),
            manifest,
            bound_at=10.0,
        )


def test_manifest_rejects_observation_from_another_target_handler() -> None:
    manifest = _manifest()
    wrong = replace(
        manifest.execution_observation_capability,
        target_handler_id="another-target-handler",
    )

    with pytest.raises(ValueError, match="another target handler"):
        replace(manifest, execution_observation_capability=wrong)


def test_status_does_not_allow_ready_claim_outside_ready_state() -> None:
    manifest = _manifest()

    with pytest.raises(ValueError, match="ready must be true"):
        replace(_status(manifest, lifecycle_state="holding"), ready=True)


def test_operation_invocation_binds_lifecycle_epoch_confirmation_and_permit() -> None:
    manifest = _manifest()
    status = _status(manifest, lifecycle_state="holding")
    request = _home_request(manifest)
    permit = _home_permit(manifest)

    capability = validate_target_runtime_operation_invocation(
        manifest,
        status,
        request,
        dispatch_permit=permit,
    )

    assert capability.operation_id == "embodiment.home"
    assert capability.motion_effect == "required"


def test_operation_invocation_rejects_stale_or_unconfirmed_home() -> None:
    manifest = _manifest()
    status = _status(manifest, lifecycle_state="holding")

    with pytest.raises(ValueError, match="operator confirmation"):
        validate_target_runtime_operation_invocation(
            manifest,
            status,
            _home_request(manifest, confirmation=None),
            dispatch_permit=_home_permit(manifest),
        )
    with pytest.raises(ValueError, match="stale execution epoch"):
        validate_target_runtime_operation_invocation(
            manifest,
            status,
            _home_request(manifest, execution_epoch=3),
            dispatch_permit=_home_permit(manifest, execution_epoch=3),
        )
    with pytest.raises(ValueError, match="another operator confirmation"):
        validate_target_runtime_operation_invocation(
            manifest,
            status,
            _home_request(manifest),
            dispatch_permit=replace(
                _home_permit(manifest),
                authorization_event_id="operator-confirmation:home:other",
            ),
        )


def test_motion_capability_cannot_bypass_dispatch_ownership() -> None:
    with pytest.raises(ValueError, match="motion effect requires a permit"):
        replace(
            _home_operation(),
            required_permit_kind="none",
            required_permit_operation=None,
        )
    with pytest.raises(ValueError, match="canonical permit operation"):
        replace(
            _home_operation(),
            required_permit_operation="embodiment.home",
        )


def test_operation_receipt_is_bound_to_request_and_declared_schema() -> None:
    manifest = _manifest()
    request = _home_request(manifest)
    receipt = TargetRuntimeOperationReceipt(
        request_id=request.request_id,
        request_hash=request.semantic_content_hash,
        manifest_hash=manifest.semantic_content_hash,
        operation_id=request.operation_id,
        receipt_schema_hash="operation-receipt:home:v1",
        target_execution_epoch=request.target_execution_epoch,
        status="completed",
        completed_at=22.0,
        clock_domain_id=request.clock_domain_id,
        evidence_refs=("backend-home-receipt:1",),
        result_payload={"reached": True},
    )

    validate_target_runtime_operation_receipt(manifest, request, receipt)

    with pytest.raises(ValueError, match="receipt binding"):
        validate_target_runtime_operation_receipt(
            manifest,
            request,
            replace(receipt, receipt_schema_hash="another-receipt-schema"),
        )


class _FakeExecutionPort:
    def __init__(self, manifest: ResolvedTargetRuntimeManifest) -> None:
        self.target_handler_id = manifest.target_handler_id
        self.activation_mode = "required"
        self.component_target_capability = manifest.component_target_capability

    def submit(self, request: object) -> object:
        raise AssertionError("The registry must not submit target work.")


class _FakeExecutionStatusProvider:
    def capture_target_execution_status_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> None:
        return None


class _FakeObservationProvider:
    def __init__(self, manifest: ResolvedTargetRuntimeManifest) -> None:
        self.execution_observation_binding = None
        self.execution_observation_capability = (
            manifest.execution_observation_capability
        )


class _FakeStatusProvider:
    def __init__(self, manifest: ResolvedTargetRuntimeManifest) -> None:
        self.resolved_target_runtime_manifest = manifest

    def capture_target_runtime_status(self) -> TargetRuntimeStatusSnapshot:
        return _status(
            self.resolved_target_runtime_manifest,
            lifecycle_state="created",
            execution_epoch=0,
        )


class _FakeOperationPort:
    def __init__(self, manifest: ResolvedTargetRuntimeManifest) -> None:
        self.resolved_target_runtime_manifest = manifest

    def execute_target_runtime_operation(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("The registry must not execute target operations.")


class _FakeHandle:
    def __init__(self, manifest: ResolvedTargetRuntimeManifest) -> None:
        self.resolved_target_runtime_manifest = manifest
        self.execution_port = _FakeExecutionPort(manifest)
        self.execution_status_provider = _FakeExecutionStatusProvider()
        self.execution_observation_provider = _FakeObservationProvider(manifest)
        self.target_runtime_status_provider = _FakeStatusProvider(manifest)
        self.target_runtime_operation_port = _FakeOperationPort(manifest)


class _FakeProvider:
    def __init__(
        self,
        identity: TargetRuntimeProviderIdentity,
        manifest: ResolvedTargetRuntimeManifest,
    ) -> None:
        self.provider_identity = identity
        self.manifest = manifest
        self.created_handle_count = 0

    def resolve_target_runtime_manifest(
        self,
        config: object,
        *,
        runtime_session_id: str,
        control_domain_id: str,
    ) -> ResolvedTargetRuntimeManifest:
        assert config == {"profile": "fixture"}
        assert runtime_session_id == self.manifest.runtime_session_id
        assert control_domain_id == self.manifest.control_domain_id
        return self.manifest

    def create_target_runtime_handle(
        self,
        config: object,
        *,
        manifest: ResolvedTargetRuntimeManifest,
        binding_receipt: object,
    ) -> _FakeHandle:
        assert config == {"profile": "fixture"}
        assert manifest.semantic_content_hash == self.manifest.semantic_content_hash
        self.created_handle_count += 1
        return _FakeHandle(manifest)


def _runtime_requirement(
    manifest: ResolvedTargetRuntimeManifest,
) -> TargetRuntimeRequirement:
    return TargetRuntimeRequirement(
        requirement_id="provider-registry-fixture-v1",
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
        clock_domain_id=manifest.clock_domain_id,
        embodiment_instance_id=manifest.embodiment_instance_id,
        target_handler_id=manifest.target_handler_id,
        action_schema_hash=manifest.action_schema_hash,
        required_component_ids=manifest.component_ids,
        required_operation_ids=("embodiment.home",),
    )


def test_provider_registry_is_allowlisted_lazy_and_does_not_own_handles() -> None:
    manifest = _manifest()
    provider = _FakeProvider(manifest.provider, manifest)
    load_count = 0

    def load_provider() -> _FakeProvider:
        nonlocal load_count
        load_count += 1
        return provider

    registry = TargetRuntimeProviderRegistry()
    registry.register(
        TargetRuntimeProviderRegistryEntry(
            provider_identity=manifest.provider,
            loader=load_provider,
        )
    )

    assert registry.provider_ids() == (manifest.provider.provider_id,)
    assert registry.provider_identities() == (manifest.provider,)
    assert load_count == 0

    resolved = registry.resolve_manifest(
        manifest.provider.provider_id,
        {"profile": "fixture"},
        runtime_session_id=manifest.runtime_session_id,
        control_domain_id=manifest.control_domain_id,
    )
    assert resolved.semantic_content_hash == manifest.semantic_content_hash
    assert load_count == 1

    binding = bind_target_runtime_requirement(
        _runtime_requirement(manifest),
        manifest,
        bound_at=30.0,
    )
    handle_a = registry.create_handle(
        manifest.provider.provider_id,
        {"profile": "fixture"},
        manifest=manifest,
        binding_receipt=binding,
    )
    handle_b = registry.create_handle(
        manifest.provider.provider_id,
        {"profile": "fixture"},
        manifest=manifest,
        binding_receipt=binding,
    )

    assert load_count == 1
    assert provider.created_handle_count == 2
    assert handle_a is not handle_b
    assert handle_a.resolved_target_runtime_manifest is manifest


def test_provider_identity_hash_binds_runtime_handle_mode() -> None:
    identity = _manifest().provider
    preflight_identity = replace(
        identity,
        runtime_handle_mode="preflight_only",
    )

    assert (
        identity.semantic_content_hash
        != preflight_identity.semantic_content_hash
    )
    assert preflight_identity.semantic_payload()["runtime_handle_mode"] == (
        "preflight_only"
    )

    with pytest.raises(ValueError, match="handle mode"):
        replace(identity, runtime_handle_mode="device_handle")


def test_provider_registry_rejects_loaded_identity_or_handle_drift() -> None:
    manifest = _manifest()
    another_identity = replace(
        manifest.provider,
        provider_version="9.0.0",
    )
    registry = TargetRuntimeProviderRegistry()
    registry.register(
        TargetRuntimeProviderRegistryEntry(
            provider_identity=manifest.provider,
            loader=lambda: _FakeProvider(another_identity, manifest),
        )
    )

    with pytest.raises(ValueError, match="identity differs"):
        registry.resolve_manifest(
            manifest.provider.provider_id,
            {"profile": "fixture"},
            runtime_session_id=manifest.runtime_session_id,
            control_domain_id=manifest.control_domain_id,
        )

    provider = _FakeProvider(manifest.provider, manifest)
    registry = TargetRuntimeProviderRegistry()
    registry.register(
        TargetRuntimeProviderRegistryEntry(
            provider_identity=manifest.provider,
            loader=lambda: provider,
        )
    )
    binding = bind_target_runtime_requirement(
        _runtime_requirement(manifest),
        manifest,
        bound_at=30.0,
    )
    wrong_manifest = replace(manifest, embodiment_instance_hash="another-instance")
    provider.create_target_runtime_handle = lambda *args, **kwargs: _FakeHandle(
        wrong_manifest
    )

    with pytest.raises(ValueError, match="another manifest"):
        registry.create_handle(
            manifest.provider.provider_id,
            {"profile": "fixture"},
            manifest=manifest,
            binding_receipt=binding,
        )

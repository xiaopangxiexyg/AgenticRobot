from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import replace

import pytest

from agent_gateway.protocol.contracts import (
    AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1,
    AgentObservationSnapshot,
    AgentOwnDomainHoldRequest,
    AgentOwnOperationCancelRequest,
    AgentOwnOperationQuery,
    AgentSkillRequest,
)
from agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
)
from agent_gateway.runtime.access_policy import (
    AgentSkillAccessPolicy,
    issue_skill_execution_grant,
)
from agent_gateway.runtime.executor_registry import (
    SkillExecutorBinding,
    SkillExecutorRegistry,
)
from agent_gateway.runtime.operation_port import (
    AgentGatewayAuthoritativePortReceipt,
    DeterministicNoIoAgentSkillOperationPort,
    NullAgentSkillOperationPort,
)
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from agent_gateway.runtime.passive_read_executor import (
    PassiveReadResult,
    PassiveReadSkillExecutor,
)
from agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from agent_gateway.runtime.service import (
    AgentGatewayService,
    NoIoAgentGatewayService,
)


class _ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self._value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value


class _AuthoritativeStationPort:
    no_io = False
    port_id = "agent-gateway:authoritative-station-test"

    def __init__(self, clock: _ManualClock) -> None:
        self.clock = clock
        self.calls: list[str] = []

    def _receipt(
        self,
        *,
        kind: str,
        operation,
        proves_hold: bool = False,
    ) -> AgentGatewayAuthoritativePortReceipt:
        self.calls.append(kind)
        operation_id = None if operation is None else operation.operation_id
        return AgentGatewayAuthoritativePortReceipt(
            port_id=self.port_id,
            operation_kind=kind,
            operation_id=operation_id,
            succeeded=True,
            status=f"station_{kind}_confirmed",
            completed_at=self.clock(),
            call_sequence=len(self.calls) - 1,
            lower_operation_id=(
                None
                if operation is None
                else f"station-operation:{operation.operation_id}"
            ),
            evidence_refs=(f"lower-{kind}-receipt:{len(self.calls)}",),
            proves_motion_or_hold=proves_hold,
        )

    def activate(self, *, request, operation):
        assert request.semantic_content_hash == operation.request_hash
        return self._receipt(kind="activate", operation=operation)

    def cancel(self, *, request, operation, reason):
        assert request.semantic_content_hash == operation.request_hash
        assert reason
        return self._receipt(kind="cancel", operation=operation)

    def request_hold(self, *, operation, reason):
        assert reason
        return self._receipt(
            kind="hold",
            operation=operation,
            proves_hold=True,
        )


def _capability(
    *,
    cancellation_policy: str = "cancel_only",
    requires_task_evaluation: bool = False,
) -> SkillExecutionRoute:
    return SkillExecutionRoute(
        skill_id="inspection.station_state",
        skill_version="1",
        skill_definition_hash="definition:inspection.station_state:v1",
        route_id="null.fixture.station_reader",
        route_version="1",
        parameter_schema_hash="params:inspect:v1",
        execution_result_schema_hash="result:inspect:v1",
        task_evaluation_schema_hash="evaluation:inspect:v1",
        allowed_parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="safe-v1",
                parameter_profile_hash="profile:inspect:v1",
                parameter_schema_hash="params:inspect:v1",
            ),
        ),
        control_effect_class="read_only",
        privilege_tier="t0_read_only",
        supported_modes=("dry_run",),
        target_kind="null",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy=cancellation_policy,
        requires_task_evaluation=requires_task_evaluation,
    )


def _manifest(
    capability: SkillExecutionRoute,
    *,
    authoritative_receipt_feature: bool = False,
) -> SkillRouteManifest:
    return SkillRouteManifest(
        runtime_session_id="runtime:b2a-service",
        control_domain_id="domain:arms",
        station_manifest_hash="station:b2a-service",
        execution_manifest_hash="execution:null-b2a-service",
        skill_definition_catalog_hash="definition-catalog:b2a-service",
        skill_schema_catalog_hash="schema-catalog:b2a-service",
        target_binding_id="target:null-b2a-service",
        target_kind="null",
        current_mode="dry_run",
        supported_transport_features=(
            "idempotency_v1",
            "poll_status_v1",
            *(
                (AGENT_GATEWAY_AUTHORITATIVE_PORT_RECEIPT_FEATURE_V1,)
                if authoritative_receipt_feature
                else ()
            ),
        ),
        skill_routes=(
            SkillRouteManifestEntry(
                route=capability,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:null-b2a-service",
                    deployment_binding_hash="binding:null-b2a-service",
                    target_binding_id="target:null-b2a-service",
                    target_kind="null",
                    execution_manifest_hash="execution:null-b2a-service",
                ),
                evaluation_plan_hash=(
                    "evaluation-plan:inspect:v1"
                    if capability.requires_task_evaluation
                    else None
                ),
                availability_hash="availability:null-ready",
            ),
        ),
        explicit_exclusions=(
            "device_io",
            "motion",
            "operation_queue",
            "production_entrypoint",
        ),
    )


def _policy(
    *,
    mode: str = "dry_run",
    allowed_client_ids: tuple[str, ...] = ("rpent:b2a",),
) -> AgentSkillAccessPolicy:
    return AgentSkillAccessPolicy(
        access_policy_id="agent-access:b2a-service",
        allowed_client_ids=allowed_client_ids,
        allowed_modes=(mode,),
        max_request_timeout_ms=1_000,
        max_grant_lifetime_sec=30.0,
    )


def _payload_validators(
    capability: SkillExecutionRoute,
) -> AgentSkillPayloadValidatorRegistry:
    def validate_empty_parameters(payload) -> None:
        if not isinstance(payload, Mapping) or payload:
            raise ValueError("fixture parameters must be an empty object")

    def validate_object(payload) -> None:
        if not isinstance(payload, Mapping):
            raise TypeError("fixture payload must be an object")

    return AgentSkillPayloadValidatorRegistry(
        registry_id="payload-validators:station-service-test",
        validators=(
            AgentSkillPayloadValidatorSet(
                validator_id="payload-validator:station-reader:v1",
                skill_id=capability.skill_id,
                skill_version=capability.skill_version,
                route_id=capability.route_id,
                route_version=capability.route_version,
                parameter_schema_hash=capability.parameter_schema_hash,
                execution_result_schema_hash=(
                    capability.execution_result_schema_hash
                ),
                task_evaluation_schema_hash=(
                    capability.task_evaluation_schema_hash
                ),
                validate_parameters=validate_empty_parameters,
                validate_execution_result=validate_object,
                validate_task_evaluation=validate_object,
            ),
        ),
    )


def _request(
    manifest: SkillRouteManifest,
    capability: SkillExecutionRoute,
    *,
    sequence: int = 1,
    key: str = "idem:1",
    request_id: str = "request:1",
    timeout_ms: int = 500,
    grant_id: str | None = None,
    snapshot_id: str = "snapshot:b2a:4",
    client_id: str = "rpent:b2a",
) -> AgentSkillRequest:
    entry = manifest.route(
        capability.skill_id,
        capability.skill_version,
        capability.route_id,
        capability.route_version,
    )
    profile = capability.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id=client_id,
        request_id=request_id,
        idempotency_key=key,
        client_sequence=sequence,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=4,
        skill_id=capability.skill_id,
        skill_version=capability.skill_version,
        skill_definition_hash=capability.skill_definition_hash,
        route_id=capability.route_id,
        route_version=capability.route_version,
        route_hash=capability.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={},
        observation_snapshot_id=snapshot_id,
        requested_mode=manifest.current_mode,
        requested_timeout_ms=timeout_ms,
        skill_execution_grant_id=grant_id,
    )


def _snapshot(
    manifest: SkillRouteManifest,
    *,
    snapshot_id: str = "snapshot:b2a:4",
    station_revision: int = 4,
    captured_at: float = 9.8,
    available_at: float = 9.9,
    active_operation_id: str | None = None,
    measurement_valid: bool = True,
) -> AgentObservationSnapshot:
    return AgentObservationSnapshot(
        snapshot_id=snapshot_id,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        station_revision=station_revision,
        captured_at=captured_at,
        available_at=available_at,
        clock_domain_id="agent-gateway-monotonic",
        authority_epoch=0,
        execution_epoch=0,
        dispatch_owner_generation=0,
        lifecycle_state="hold",
        execution_gate="hold",
        active_operation_id=active_operation_id,
        source_binding_id=None,
        producer_epoch=None,
        target_binding_id=manifest.target_binding_id,
        execution_manifest_hash=manifest.execution_manifest_hash,
        frame_graph_hash="frames:null-b2a",
        tool_config_hash="tool:null-b2a",
        calibration_hash="calibration:null-b2a",
        measurement_summary={"fixture": "no_io"},
        measurement_freshness_ms=0,
        measurement_valid=measurement_valid,
    )


def _service(
    *,
    cancellation_policy: str = "cancel_only",
    requires_task_evaluation: bool = False,
    port=None,
    allowed_client_ids: tuple[str, ...] = ("rpent:b2a",),
):
    clock = _ManualClock()
    capability = _capability(
        cancellation_policy=cancellation_policy,
        requires_task_evaluation=requires_task_evaluation,
    )
    manifest = _manifest(capability)
    if port is None:
        port = DeterministicNoIoAgentSkillOperationPort(clock=clock)
    service = NoIoAgentGatewayService(
        manifest=manifest,
        access_policy=_policy(allowed_client_ids=allowed_client_ids),
        operation_port=port,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=4
        ),
        clock=clock,
    )
    published = service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    )
    assert published.accepted
    return service, port, clock, manifest, capability


def _submit(service, manifest, capability, *, now: float = 10.0):
    return service.submit_request(
        request=_request(manifest, capability),
        current_station_revision=4,
        now=now,
    )


def test_default_v2_protocol_runtime_publishes_inline_io_and_replay_events() -> None:
    service, _port, _clock, manifest, capability = _service()
    runtime = service.protocol_runtime
    assert not runtime.session_capabilities.artifact_transfer_enabled
    assert runtime.io_catalog.semantic_content_hash == manifest.skill_io_catalog_hash
    assert not runtime.io_catalog.contract(
        capability.skill_id,
        capability.skill_version,
    ).output_artifact_slots

    admitted = _submit(service, manifest, capability)
    assert admitted.accepted and admitted.operation is not None
    first = runtime.replay_events(
        after_sequence=0,
        max_events=16,
        operations=service.operation_store.operations(),
    )
    assert [event.event_kind for event in first.events] == ["operation_updated"]
    replay = runtime.replay_events(
        after_sequence=first.latest_sequence,
        max_events=16,
        operations=service.operation_store.operations(),
    )
    assert replay.events == ()


def _own_operation_query(
    manifest: SkillRouteManifest,
    operation_id: str,
    *,
    client_id: str = "rpent:b2a",
) -> AgentOwnOperationQuery:
    return AgentOwnOperationQuery(
        client_id=client_id,
        query_id="query:own-operation:1",
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        operation_id=operation_id,
    )


def _cancel_request(
    manifest: SkillRouteManifest,
    operation_id: str,
    *,
    idempotency_key: str = "control-idem:cancel:1",
    reason_code: str = "agent_task_aborted",
) -> AgentOwnOperationCancelRequest:
    return AgentOwnOperationCancelRequest(
        client_id="rpent:b2a",
        control_request_id="control:cancel:1",
        idempotency_key=idempotency_key,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        operation_id=operation_id,
        observation_snapshot_id="snapshot:b2a:4",
        expected_station_revision=4,
        reason_code=reason_code,
    )


def _hold_request(
    manifest: SkillRouteManifest,
    *,
    idempotency_key: str = "control-idem:hold:1",
    reason_code: str = "agent_requested_pause",
) -> AgentOwnDomainHoldRequest:
    return AgentOwnDomainHoldRequest(
        client_id="rpent:b2a",
        control_request_id="control:hold:1",
        idempotency_key=idempotency_key,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        control_domain_id=manifest.control_domain_id,
        target_binding_id=manifest.target_binding_id,
        observation_snapshot_id="snapshot:b2a:4",
        expected_station_revision=4,
        reason_code=reason_code,
    )


def _activate_running(service, operation_id: str, clock: _ManualClock):
    clock.set(10.1)
    outcome = service.activate_operation(
        operation_id=operation_id,
        current_station_revision=4,
        now=10.1,
    )
    assert outcome.accepted
    assert outcome.operation.status == "running"
    return outcome.operation


def test_submit_admits_but_does_not_call_port_or_execute_operation() -> None:
    service, port, _, manifest, capability = _service()
    submission = _submit(service, manifest, capability)
    assert submission.accepted
    assert submission.reason_code == "operation_admitted_not_executed"
    assert submission.operation.status == "admitted"
    assert not submission.operation.terminal
    assert port.receipts == ()
    assert service.get_active_operation() is submission.operation


def test_gateway_admits_a_policy_allowed_non_rpent_upper_client() -> None:
    external_client_id = "other-upper:planner-1"
    service, port, _, manifest, capability = _service(
        allowed_client_ids=(external_client_id,),
    )

    submission = service.submit_request(
        request=_request(
            manifest,
            capability,
            client_id=external_client_id,
        ),
        current_station_revision=4,
        now=10.0,
    )

    assert submission.accepted
    assert submission.operation.status == "admitted"
    assert port.receipts == ()


def test_observation_receiver_binds_session_manifest_revision_and_request() -> None:
    service, _, _, manifest, capability = _service()
    assert service.get_routes() is manifest
    observation = service.get_observation()
    assert observation.snapshot_id == "snapshot:b2a:4"

    idempotent = service.publish_observation(snapshot=observation, now=10.0)
    assert idempotent.accepted
    assert idempotent.reason_code == "observation_idempotent_lookup"
    wrong_target = service.publish_observation(
        snapshot=replace(observation, target_binding_id="target:wrong"),
        now=10.0,
    )
    assert wrong_target.reason_code == "observation_target_binding_mismatch"

    mismatched_request = replace(
        _request(manifest, capability),
        observation_snapshot_id="snapshot:unknown",
    )
    rejected = service.submit_request(
        request=mismatched_request,
        current_station_revision=4,
        now=10.0,
    )
    assert rejected.reason_code == "observation_snapshot_id_mismatch"


def test_idempotent_replay_does_not_reassess_or_create_an_operation() -> None:
    service, port, _, manifest, capability = _service()
    request = _request(manifest, capability)
    first = service.submit_request(
        request=request,
        current_station_revision=4,
        now=10.0,
    )
    replay = service.submit_request(
        request=request,
        current_station_revision=999,
        now=99.0,
    )
    assert replay.accepted
    assert replay.idempotent_replay
    assert replay.operation is first.operation
    assert replay.access_decision is None
    assert port.receipts == ()

    conflict = service.submit_request(
        request=replace(request, parameters={"changed": True}),
        current_station_revision=4,
        now=10.1,
    )
    assert conflict.reason_code == "idempotency_key_conflict"


def test_fake_worker_activation_and_completion_remain_distinct_states() -> None:
    service, port, clock, manifest, capability = _service()
    admitted = _submit(service, manifest, capability).operation
    running = _activate_running(service, admitted.operation_id, clock)
    assert port.call_count("activate") == 1
    assert running.reason_code == "fake_activation_accepted_not_completed"
    assert not running.terminal

    clock.set(10.2)
    completed = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=running.semantic_content_hash,
        succeeded=True,
        reason_code="fake_measured_terminal_fixture",
        current_station_revision=5,
        execution_result={"fixture_result": "complete"},
        execution_receipt_refs=("fake-worker-result:1",),
        now=10.2,
    )
    assert completed.accepted
    assert completed.operation.status == "completed"
    assert completed.operation.execution_outcome == "completed"
    assert completed.operation.task_outcome == "not_assessed"
    assert completed.operation.execution_result["fixture_result"] == "complete"
    assert service.get_active_operation() is None


def test_generic_service_accepts_authoritative_start_and_hold_receipts() -> None:
    clock = _ManualClock()
    capability = _capability(
        cancellation_policy="hold_on_cancel_disconnect_or_timeout"
    )
    manifest = _manifest(capability)
    port = _AuthoritativeStationPort(clock)
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        operation_port=port,
        payload_validators=_payload_validators(capability),
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=4
        ),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    admitted = _submit(service, manifest, capability).operation
    running = service.activate_operation(
        operation_id=admitted.operation_id,
        current_station_revision=4,
        now=10.1,
    )
    assert running.accepted
    assert running.reason_code == "operation_running_station_owned"
    assert running.operation.reason_code == (
        "station_activation_accepted_not_completed"
    )
    assert running.port_receipt.authoritative_lower_receipt
    assert not running.port_receipt.proves_motion_or_hold

    cancelled = service.cancel_operation(
        operation_id=running.operation.operation_id,
        reason="planner_aborted_task",
        current_station_revision=5,
        now=10.2,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert cancelled.port_receipt.authoritative_lower_receipt
    assert cancelled.port_receipt.proves_motion_or_hold
    assert port.calls == ["activate", "hold"]


@pytest.mark.parametrize("advertise_receipt", (False, True))
def test_authoritative_control_receipt_projection_is_feature_gated(
    advertise_receipt: bool,
) -> None:
    clock = _ManualClock()
    capability = _capability(cancellation_policy="hold_on_cancel")
    manifest = _manifest(
        capability,
        authoritative_receipt_feature=advertise_receipt,
    )
    port = _AuthoritativeStationPort(clock)
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        operation_port=port,
        payload_validators=_payload_validators(capability),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    admitted = _submit(service, manifest, capability).operation
    assert admitted is not None
    assert service.activate_operation(
        operation_id=admitted.operation_id,
        current_station_revision=4,
        now=10.1,
    ).accepted

    request = _cancel_request(manifest, admitted.operation_id)
    outcome = service.cancel_own_operation(
        request=request,
        current_station_revision=4,
        now=10.2,
    )
    replay = service.cancel_own_operation(
        request=request,
        current_station_revision=4,
        now=10.3,
    )
    assert outcome.accepted and replay.accepted and replay.idempotent_replay
    wire = outcome.to_wire()
    replay_wire = replay.to_wire()
    if advertise_receipt:
        receipt = wire["port_receipt"]
        assert receipt is not None
        assert receipt["operation_kind"] == "hold"
        assert receipt["operation_id"] == admitted.operation_id
        assert receipt["authoritative_lower_receipt"] is True
        assert receipt["proves_motion_or_hold"] is True
        assert replay_wire["port_receipt"] == receipt
    else:
        assert wire["port_receipt"] is None
        assert replay_wire["port_receipt"] is None


def test_no_io_receipt_never_projects_as_authoritative_proof() -> None:
    clock = _ManualClock()
    capability = _capability()
    manifest = _manifest(
        capability,
        authoritative_receipt_feature=True,
    )
    port = DeterministicNoIoAgentSkillOperationPort(clock=clock)
    service = NoIoAgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        operation_port=port,
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    admitted = _submit(service, manifest, capability).operation
    assert admitted is not None
    outcome = service.cancel_own_operation(
        request=_cancel_request(manifest, admitted.operation_id),
        current_station_revision=4,
        now=10.1,
    )
    assert outcome.accepted
    assert outcome.port_receipt is not None
    assert outcome.port_receipt.authoritative_lower_receipt is False
    assert outcome.to_wire()["port_receipt"] is None


def test_authoritative_service_requires_server_side_payload_validators() -> None:
    capability = _capability()
    with pytest.raises(TypeError, match="requires server-side payload validators"):
        AgentGatewayService(
            manifest=_manifest(capability),
            access_policy=_policy(),
            operation_port=_AuthoritativeStationPort(_ManualClock()),
        )


def test_authoritative_service_rejects_schema_invalid_parameters() -> None:
    clock = _ManualClock()
    capability = _capability()
    manifest = _manifest(capability)
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        operation_port=_AuthoritativeStationPort(clock),
        payload_validators=_payload_validators(capability),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    invalid = replace(
        _request(manifest, capability),
        parameters={"unadvertised": True},
    )
    rejected = service.submit_request(
        request=invalid,
        current_station_revision=4,
        now=10.0,
    )
    assert not rejected.accepted
    assert rejected.reason_code == "parameter_payload_rejected"
    assert service.get_active_operation() is None


def test_result_validator_receives_original_immutable_request_parameters() -> None:
    clock = _ManualClock()
    capability = _capability()
    manifest = _manifest(capability)
    seen_parameters: list[dict[str, object]] = []

    def validate_parameters(payload) -> None:
        if dict(payload) != {"max_age_ms": 25}:
            raise ValueError("unexpected fixture parameters")

    def validate_result(payload) -> None:
        if set(payload) != {"age_ms"}:
            raise ValueError("unexpected fixture result")

    def validate_result_for_parameters(parameters, result) -> None:
        seen_parameters.append(dict(parameters))
        if result["age_ms"] > parameters["max_age_ms"]:
            raise ValueError("result exceeds request freshness bound")

    validators = AgentSkillPayloadValidatorRegistry(
        registry_id="payload-validators:contextual-result-test",
        validators=(
            AgentSkillPayloadValidatorSet(
                validator_id="payload-validator:contextual-result-test:v1",
                skill_id=capability.skill_id,
                skill_version=capability.skill_version,
                route_id=capability.route_id,
                route_version=capability.route_version,
                parameter_schema_hash=capability.parameter_schema_hash,
                execution_result_schema_hash=(
                    capability.execution_result_schema_hash
                ),
                task_evaluation_schema_hash=(
                    capability.task_evaluation_schema_hash
                ),
                validate_parameters=validate_parameters,
                validate_execution_result=validate_result,
                validate_task_evaluation=lambda _payload: None,
                validate_execution_result_for_parameters=(
                    validate_result_for_parameters
                ),
            ),
        ),
    )
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        operation_port=_AuthoritativeStationPort(clock),
        payload_validators=validators,
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    request = replace(
        _request(manifest, capability),
        parameters={"max_age_ms": 25},
    )
    admitted = service.submit_request(
        request=request,
        current_station_revision=4,
        now=10.0,
    ).operation
    running = service.activate_operation(
        operation_id=admitted.operation_id,
        current_station_revision=4,
        now=10.1,
    ).operation

    rejected = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=running.semantic_content_hash,
        succeeded=True,
        reason_code="fixture_result_too_old",
        current_station_revision=4,
        execution_result={"age_ms": 26},
        now=10.2,
    )
    assert not rejected.accepted
    assert rejected.reason_code == "execution_result_payload_rejected"
    assert rejected.operation.status == "running"
    assert seen_parameters == [{"max_age_ms": 25}]

    completed = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=running.semantic_content_hash,
        succeeded=True,
        reason_code="fixture_result_fresh",
        current_station_revision=4,
        execution_result={"age_ms": 25},
        now=10.3,
    )
    assert completed.accepted
    assert completed.operation.execution_result == {"age_ms": 25}
    assert seen_parameters == [
        {"max_age_ms": 25},
        {"max_age_ms": 25},
    ]


def test_no_io_service_rejects_authoritative_port() -> None:
    capability = _capability()
    manifest = _manifest(capability)
    with pytest.raises(TypeError, match="only a no_io"):
        NoIoAgentGatewayService(
            manifest=manifest,
            access_policy=_policy(),
            operation_port=_AuthoritativeStationPort(_ManualClock()),
        )


def test_successful_authoritative_hold_receipt_requires_lower_hold_proof() -> None:
    with pytest.raises(ValueError, match="must prove lower HOLD"):
        AgentGatewayAuthoritativePortReceipt(
            port_id="station:test",
            operation_kind="hold",
            operation_id="operation:test",
            succeeded=True,
            status="hold_confirmed",
            completed_at=10.0,
            call_sequence=0,
            lower_operation_id="station-operation:test",
            evidence_refs=("lower-hold-receipt:test",),
            proves_motion_or_hold=False,
        )


def test_required_task_evaluation_closes_after_execution_and_can_fail_task() -> None:
    service, port, clock, manifest, capability = _service(
        requires_task_evaluation=True
    )
    admitted = _submit(service, manifest, capability).operation
    running = _activate_running(service, admitted.operation_id, clock)
    execution = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=running.semantic_content_hash,
        succeeded=True,
        reason_code="fake_execution_completed",
        current_station_revision=4,
        execution_result={"fixture_result": "complete"},
        execution_receipt_refs=("fake-worker-result:required-eval",),
        now=10.2,
    )
    assert execution.accepted
    assert execution.operation.status == "awaiting_evaluation"
    assert execution.operation.execution_outcome == "completed"
    assert execution.operation.evaluation_status == "pending"
    assert execution.operation.task_outcome == "pending"
    assert not execution.operation.terminal

    evaluated = service.complete_evaluation(
        operation_id=execution.operation.operation_id,
        expected_operation_hash=execution.operation.semantic_content_hash,
        evaluation_status="completed",
        task_outcome="failed",
        reason_code="fake_mixed_evaluation_failed",
        current_station_revision=4,
        evaluator_receipt_refs=(
            "evaluation:environment:1",
            "evaluation:policy-self:1",
        ),
        aggregate_evaluation_receipt_ref="evaluation:aggregate:1",
        task_evaluation={"verdict": "failed"},
        now=10.3,
    )
    assert evaluated.accepted
    assert evaluated.operation.status == "completed"
    assert evaluated.operation.execution_outcome == "completed"
    assert evaluated.operation.task_outcome == "failed"
    assert evaluated.operation.terminal
    assert service.operation_store.active_operation_id is None
    assert port.call_count("activate") == 1


def test_cancel_control_wins_while_task_evaluation_is_pending() -> None:
    service, _, clock, manifest, capability = _service(
        requires_task_evaluation=True
    )
    admitted = _submit(service, manifest, capability).operation
    running = _activate_running(service, admitted.operation_id, clock)
    awaiting = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=running.semantic_content_hash,
        succeeded=True,
        reason_code="fake_execution_completed",
        current_station_revision=4,
        execution_result={"fixture_result": "complete"},
        now=10.2,
    ).operation
    cancelled = service.cancel_operation(
        operation_id=awaiting.operation_id,
        reason="operator_cancelled_pending_evaluation",
        current_station_revision=4,
        now=10.3,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert cancelled.operation.execution_outcome == "completed"
    assert cancelled.operation.evaluation_status == "pending"

    late = service.complete_evaluation(
        operation_id=awaiting.operation_id,
        expected_operation_hash=awaiting.semantic_content_hash,
        evaluation_status="completed",
        task_outcome="succeeded",
        reason_code="late_policy_self_success",
        current_station_revision=4,
        evaluator_receipt_refs=("evaluation:policy-self:late",),
        aggregate_evaluation_receipt_ref="evaluation:aggregate:late",
        task_evaluation={"verdict": "succeeded"},
        now=10.4,
    )
    assert not late.accepted
    assert late.operation.status == "cancelled"


def test_null_port_fails_closed_and_receipt_cannot_claim_lower_evidence() -> None:
    clock = _ManualClock()
    port = NullAgentSkillOperationPort(clock=clock)
    service, _, _, manifest, capability = _service(port=port)
    admitted = _submit(service, manifest, capability).operation
    clock.set(10.1)
    outcome = service.activate_operation(
        operation_id=admitted.operation_id,
        current_station_revision=4,
        now=10.1,
    )
    assert not outcome.accepted
    assert outcome.operation.status == "failed"
    assert outcome.port_receipt.authoritative_lower_receipt is False
    assert outcome.port_receipt.proves_motion_or_hold is False
    assert outcome.port_receipt.to_wire()["evidence_scope"] == (
        "fake_or_null_application_only"
    )


def test_one_active_operation_is_busy_and_never_implicitly_queued() -> None:
    service, _, _, manifest, capability = _service()
    first = _submit(service, manifest, capability)
    second = service.submit_request(
        request=_request(
            manifest,
            capability,
            sequence=2,
            key="idem:2",
            request_id="request:2",
        ),
        current_station_revision=4,
        now=10.1,
    )
    assert first.accepted
    assert second.reason_code == "active_operation_exists_no_queue"
    assert service.operation_store.retained_operation_count == 1


def test_cancel_reservation_rejects_late_success() -> None:
    service, port, clock, manifest, capability = _service()
    admitted = _submit(service, manifest, capability).operation
    running = _activate_running(service, admitted.operation_id, clock)
    stale_running_hash = running.semantic_content_hash

    clock.set(10.2)
    cancelled = service.cancel_operation(
        operation_id=running.operation_id,
        reason="test_cancel",
        current_station_revision=5,
        now=10.2,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert port.call_count("cancel") == 1

    clock.set(10.3)
    late = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=stale_running_hash,
        succeeded=True,
        reason_code="late_success_must_not_win",
        current_station_revision=5,
        execution_result={"must_not_commit": True},
        now=10.3,
    )
    assert not late.accepted
    assert late.operation.status == "cancelled"


class _BlockingActivationPort(DeterministicNoIoAgentSkillOperationPort):
    def __init__(self, *, clock: _ManualClock) -> None:
        super().__init__(clock=clock)
        self.entered = threading.Event()
        self.release = threading.Event()

    def activate(self, *, request, operation):
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("test activation was not released")
        return super().activate(request=request, operation=operation)


@pytest.mark.parametrize("control_kind", ("cancel", "hold"))
def test_status_cancel_and_hold_are_not_blocked_by_stalled_activation(
    control_kind: str,
) -> None:
    clock = _ManualClock()
    port = _BlockingActivationPort(clock=clock)
    service, _, _, manifest, capability = _service(port=port)
    admitted = _submit(service, manifest, capability).operation
    clock.set(10.1)
    activation_outcomes = []
    worker = threading.Thread(
        target=lambda: activation_outcomes.append(
            service.activate_operation(
                operation_id=admitted.operation_id,
                current_station_revision=4,
                now=10.1,
            )
        )
    )
    worker.start()
    assert port.entered.wait(timeout=1.0)
    assert service.get_operation(admitted.operation_id).status == "activated"

    clock.set(10.2)
    if control_kind == "cancel":
        control = service.cancel_operation(
            operation_id=admitted.operation_id,
            reason="cancel_while_activation_stalled",
            current_station_revision=5,
            now=10.2,
        )
    else:
        control = service.request_hold(
            reason="hold_while_activation_stalled",
            current_station_revision=5,
            now=10.2,
        )
    assert control.accepted
    assert control.operation.status == "cancelled"
    assert worker.is_alive()

    port.release.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(activation_outcomes) == 1
    assert not activation_outcomes[0].accepted
    assert activation_outcomes[0].reason_code == (
        "operation_superseded_after_port_call"
    )
    assert service.get_operation(admitted.operation_id).status == "cancelled"


def test_failed_cancel_remains_active_and_blocks_late_worker_result() -> None:
    clock = _ManualClock()
    port = DeterministicNoIoAgentSkillOperationPort(
        succeeds_by_kind={"cancel": False},
        clock=clock,
    )
    service, _, _, manifest, capability = _service(port=port)
    admitted = _submit(service, manifest, capability).operation
    running = _activate_running(service, admitted.operation_id, clock)
    old_hash = running.semantic_content_hash

    clock.set(10.2)
    cancellation = service.cancel_operation(
        operation_id=running.operation_id,
        reason="injected_failure",
        current_station_revision=5,
        now=10.2,
    )
    assert not cancellation.accepted
    assert not cancellation.operation.terminal
    assert service.get_active_operation() is cancellation.operation

    late = service.complete_operation(
        operation_id=running.operation_id,
        expected_operation_hash=old_hash,
        succeeded=True,
        reason_code="must_not_win",
        current_station_revision=5,
        execution_result={"unsafe": True},
        now=10.3,
    )
    assert not late.accepted
    assert service.get_active_operation() is not None


def test_b2c_status_and_cancel_are_scoped_to_the_owning_client() -> None:
    clock = _ManualClock()
    capability = _capability()
    manifest = _manifest(capability)
    port = DeterministicNoIoAgentSkillOperationPort(clock=clock)
    policy = AgentSkillAccessPolicy(
        access_policy_id="agent-access:b2c-own-operation",
        allowed_client_ids=("rpent:b2a", "rpent:other"),
        allowed_modes=("dry_run",),
        max_request_timeout_ms=1_000,
        max_grant_lifetime_sec=30.0,
    )
    service = NoIoAgentGatewayService(
        manifest=manifest,
        access_policy=policy,
        operation_port=port,
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    admitted = _submit(service, manifest, capability).operation

    found = service.get_own_operation(
        query=_own_operation_query(manifest, admitted.operation_id)
    )
    assert found.accepted
    assert found.operation is admitted
    assert found.to_wire()["operation"]["operation_id"] == admitted.operation_id

    other_client = service.get_own_operation(
        query=_own_operation_query(
            manifest,
            admitted.operation_id,
            client_id="rpent:other",
        )
    )
    assert not other_client.accepted
    assert other_client.reason_code == "own_operation_not_found"
    assert other_client.operation is None

    cancelled = service.cancel_own_operation(
        request=_cancel_request(manifest, admitted.operation_id),
        current_station_revision=4,
        now=10.1,
    )
    replay = service.cancel_own_operation(
        request=_cancel_request(manifest, admitted.operation_id),
        current_station_revision=99,
        now=99.0,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert replay.accepted and replay.idempotent_replay
    assert port.call_count("cancel") == 1


def test_b2c_hold_is_one_way_exact_domain_scoped_and_idempotent() -> None:
    service, port, _, manifest, _ = _service()
    wrong_domain = replace(
        _hold_request(manifest),
        control_domain_id="domain:other",
    )
    rejected = service.request_hold_own_domain(
        request=wrong_domain,
        current_station_revision=4,
        now=10.0,
    )
    assert not rejected.accepted
    assert rejected.reason_code == "control_domain_mismatch"
    assert port.receipts == ()

    request = _hold_request(manifest)
    first = service.request_hold_own_domain(
        request=request,
        current_station_revision=4,
        now=10.0,
    )
    replay = service.request_hold_own_domain(
        request=request,
        current_station_revision=5,
        now=10.1,
    )
    conflict = service.request_hold_own_domain(
        request=replace(request, reason_code="agent_uncertain_state"),
        current_station_revision=5,
        now=10.1,
    )
    assert first.accepted
    assert replay.accepted and replay.idempotent_replay
    assert not conflict.accepted
    assert conflict.reason_code == "control_idempotency_key_conflict"
    assert port.call_count("hold") == 1
    for forbidden in ("estop", "resume", "rearm"):
        assert not hasattr(service, forbidden)


def test_b2c_stop_requests_allow_stale_observation_but_not_future_revision() -> None:
    service, port, _, manifest, capability = _service()
    admitted = _submit(service, manifest, capability).operation
    stale_observation = replace(
        _cancel_request(manifest, admitted.operation_id),
        expected_station_revision=3,
    )
    cancelled = service.cancel_own_operation(
        request=stale_observation,
        current_station_revision=4,
        now=10.1,
    )
    assert cancelled.accepted
    assert port.call_count("cancel") == 1

    future = replace(
        _hold_request(manifest, idempotency_key="control-idem:future"),
        expected_station_revision=5,
    )
    rejected = service.request_hold_own_domain(
        request=future,
        current_station_revision=4,
        now=10.2,
    )
    assert not rejected.accepted
    assert rejected.reason_code == "station_revision_precedes_observed_revision"
    assert port.call_count("hold") == 0


def test_timeout_and_disconnect_use_declared_hold_policy() -> None:
    service, port, clock, manifest, capability = _service(
        cancellation_policy="hold_on_cancel_disconnect_or_timeout"
    )
    admitted = _submit(service, manifest, capability).operation
    clock.set(admitted.deadline_at)
    timed_out = service.expire_due_operation(
        current_station_revision=5,
        now=admitted.deadline_at,
    )
    assert timed_out.accepted
    assert timed_out.operation.status == "timed_out"
    assert port.call_count("hold") == 1

    second_request = _request(
        manifest,
        capability,
        sequence=2,
        key="idem:2",
        request_id="request:2",
        snapshot_id="snapshot:b2a:5",
    )
    second_request = replace(second_request, expected_station_revision=5)
    clock.set(11.0)
    refreshed = service.publish_observation(
        snapshot=_snapshot(
            manifest,
            snapshot_id="snapshot:b2a:5",
            station_revision=5,
            captured_at=10.9,
            available_at=10.9,
        ),
        now=11.0,
    )
    assert refreshed.accepted
    second = service.submit_request(
        request=second_request,
        current_station_revision=5,
        now=11.0,
    )
    assert second.accepted
    wrong_client = service.handle_client_disconnect(
        client_id="rpent:other",
        reason="unrelated_disconnect",
        current_station_revision=5,
        now=11.1,
    )
    assert not wrong_client.accepted
    disconnected = service.handle_client_disconnect(
        client_id="rpent:b2a",
        reason="connection_lost",
        current_station_revision=5,
        now=11.1,
    )
    assert disconnected.accepted
    assert disconnected.operation.status == "cancelled"
    assert port.call_count("hold") == 2


def test_hold_on_all_stop_policy_routes_explicit_cancel_through_hold() -> None:
    service, port, clock, manifest, capability = _service(
        cancellation_policy="hold_on_cancel_disconnect_or_timeout"
    )
    admitted = _submit(service, manifest, capability).operation
    running = _activate_running(service, admitted.operation_id, clock)
    cancelled = service.cancel_operation(
        operation_id=running.operation_id,
        reason="planner_aborted_task",
        current_station_revision=5,
        now=10.2,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert port.call_count("hold") == 1
    assert port.call_count("cancel") == 0


def test_close_is_idempotent_closes_submission_and_calls_hold_once() -> None:
    service, port, _, manifest, capability = _service()
    first = service.close(current_station_revision=4, now=10.0)
    second = service.close(current_station_revision=999, now=99.0)
    assert first is second
    assert first.accepted
    assert port.call_count("hold") == 1
    assert service.is_closed

    rejected = _submit(service, manifest, capability)
    assert not rejected.accepted
    assert rejected.reason_code == "agent_gateway_service_closed"


def test_operator_grant_budget_is_consumed_once_for_idempotent_retry() -> None:
    clock = _ManualClock(12.0)
    capability = SkillExecutionRoute(
        skill_id="lifecycle.capture_runtime_home",
        skill_version="1",
        skill_definition_hash="definition:capture-home:v1",
        route_id="real.rm65.lifecycle_controller",
        route_version="1",
        parameter_schema_hash="params:capture-home:v1",
        execution_result_schema_hash="result:capture-home:v1",
        task_evaluation_schema_hash="evaluation:capture-home:v1",
        allowed_parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="safe-v1",
                parameter_profile_hash="profile:capture-home:v1",
                parameter_schema_hash="params:capture-home:v1",
            ),
        ),
        control_effect_class="lifecycle",
        privilege_tier="t3_lifecycle",
        supported_modes=("real_observe",),
        target_kind="real",
        may_move_robot=False,
        may_open_publication=False,
        requires_operator_confirmation=True,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="hold_on_disconnect_or_timeout",
    )
    manifest = SkillRouteManifest(
        runtime_session_id="runtime:real-observe-b2a",
        control_domain_id="domain:arms",
        station_manifest_hash="station:real-observe-b2a",
        execution_manifest_hash="execution:real-observe-b2a",
        skill_definition_catalog_hash="definition-catalog:real-observe-b2a",
        skill_schema_catalog_hash="schema-catalog:real-observe-b2a",
        target_binding_id="target:real-observe-b2a",
        target_kind="real",
        current_mode="real_observe",
        supported_transport_features=("idempotency_v1",),
        skill_routes=(
            SkillRouteManifestEntry(
                route=capability,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:real-observe-b2a",
                    deployment_binding_hash="binding:real-observe-b2a",
                    target_binding_id="target:real-observe-b2a",
                    target_kind="real",
                    execution_manifest_hash="execution:real-observe-b2a",
                ),
                evaluation_plan_hash=None,
                availability_hash="availability:real-observe-ready",
            ),
        ),
        explicit_exclusions=("motion", "publication_open", "device_io"),
    )
    policy = _policy(mode="real_observe")
    pre_grant_request = _request(manifest, capability)
    grant = issue_skill_execution_grant(
        access_policy=policy,
        manifest=manifest,
        route=capability,
        request=pre_grant_request,
        operator_authorization_receipt_id="operator-confirmation:b2a",
        client_id="rpent:b2a",
        parameter_profile_id=capability.allowed_parameter_profile_ids[0],
        parameter_profile_hash=capability.allowed_parameter_profile_hashes[0],
        max_invocations=1,
        issued_at=10.0,
        expires_at=20.0,
    )
    request = replace(
        pre_grant_request,
        skill_execution_grant_id=grant.grant_id,
    )
    port = DeterministicNoIoAgentSkillOperationPort(clock=clock)
    service = NoIoAgentGatewayService(
        manifest=manifest,
        access_policy=policy,
        operation_port=port,
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(
            manifest,
            captured_at=11.8,
            available_at=11.9,
        ),
        now=12.0,
    ).accepted
    first = service.submit_request(
        request=request,
        current_station_revision=4,
        grant=grant,
        now=12.0,
    )
    replay = service.submit_request(
        request=request,
        current_station_revision=4,
        grant=grant,
        now=19.9,
    )
    assert first.accepted
    assert replay.idempotent_replay
    assert service.grant_invocations_used(grant) == 1
    assert port.receipts == ()


def test_b2a_service_rejects_non_no_io_port() -> None:
    capability = _capability()

    class _NotAllowedPort:
        no_io = False

        def activate(self, **_kwargs):
            raise AssertionError

        def cancel(self, **_kwargs):
            raise AssertionError

        def request_hold(self, **_kwargs):
            raise AssertionError

    with pytest.raises(TypeError, match="only a no_io operation port"):
        NoIoAgentGatewayService(
            manifest=_manifest(capability),
            access_policy=_policy(),
            operation_port=_NotAllowedPort(),
        )


def _two_route_manifest() -> tuple[
    SkillRouteManifest,
    SkillExecutionRoute,
    SkillExecutionRoute,
]:
    first = _capability()
    second = replace(
        first,
        skill_id="inspection.tcp_pose",
        skill_definition_hash="definition:inspection.tcp_pose:v1",
        route_id="null.fixture.tcp_pose_reader",
        skill_io_contract_hash=None,
    )
    single = _manifest(first)
    second_entry = SkillRouteManifestEntry(
        route=second,
        deployment_binding=replace(
            single.skill_routes[0].deployment_binding,
            deployment_binding_id="deployment:null-tcp-pose-reader",
            deployment_binding_hash="binding:null-tcp-pose-reader",
        ),
        evaluation_plan_hash=None,
        availability_hash="availability:null-tcp-pose-ready",
    )
    return (
        replace(
            single,
            skill_routes=(single.skill_routes[0], second_entry),
            skill_io_catalog_hash=None,
        ),
        first,
        second,
    )


def test_executor_registry_requires_exactly_one_executor_per_route() -> None:
    manifest, _, _ = _two_route_manifest()
    port = DeterministicNoIoAgentSkillOperationPort(port_id="executor:first")
    first_binding = SkillExecutorBinding.from_manifest_entry(
        manifest=manifest,
        entry=manifest.skill_routes[0],
        execution_port=port,
    )

    with pytest.raises(ValueError, match="Every advertised exact Route"):
        SkillExecutorRegistry(
            manifest=manifest,
            bindings=(first_binding,),
        )
    with pytest.raises(ValueError, match="does not close over"):
        SkillExecutorRegistry(
            manifest=manifest,
            bindings=(
                first_binding,
                replace(first_binding, route_hash="route:wrong"),
            ),
        )


def test_service_activates_and_cancels_through_exact_route_executor() -> None:
    manifest, first, second = _two_route_manifest()
    first_port = DeterministicNoIoAgentSkillOperationPort(
        port_id="executor:first"
    )
    second_port = DeterministicNoIoAgentSkillOperationPort(
        port_id="executor:second"
    )
    registry = SkillExecutorRegistry(
        manifest=manifest,
        bindings=tuple(
            SkillExecutorBinding.from_manifest_entry(
                manifest=manifest,
                entry=entry,
                execution_port=(first_port if entry.route == first else second_port),
            )
            for entry in manifest.skill_routes
        ),
    )
    domain_port = DeterministicNoIoAgentSkillOperationPort(
        port_id="domain-control:test"
    )
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        executor_registry=registry,
        domain_control_port=domain_port,
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=4
        ),
        clock=_ManualClock(),
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted

    request = _request(manifest, second)
    admitted = service.submit_request(
        request=request,
        current_station_revision=4,
        now=10.0,
    )
    assert admitted.accepted and admitted.operation is not None
    activated = service.activate_operation(
        operation_id=admitted.operation.operation_id,
        current_station_revision=4,
        now=10.0,
    )
    assert activated.accepted
    cancelled = service.cancel_operation(
        operation_id=admitted.operation.operation_id,
        reason="fixture_cancel",
        current_station_revision=4,
        now=10.0,
    )
    assert cancelled.accepted
    assert tuple(item.operation_kind for item in first_port.receipts) == ()
    assert tuple(item.operation_kind for item in second_port.receipts) == (
        "activate",
        "cancel",
    )
    assert domain_port.receipts == ()


def test_passive_read_executor_is_reusable_and_never_requests_hold() -> None:
    clock = _ManualClock()
    capability = _capability()
    manifest = _manifest(capability)
    reads: list[str] = []

    def read(request, operation) -> PassiveReadResult:
        reads.append(operation.operation_id)
        assert request.semantic_content_hash == operation.request_hash
        return PassiveReadResult(
            execution_result={"operation_id": operation.operation_id},
            evidence_refs=(f"passive-cut:{operation.operation_id}",),
        )

    executor = PassiveReadSkillExecutor(
        port_id="passive-reader:test",
        executor_id="passive-reader-executor:test",
        read=read,
        current_station_revision=lambda: 4,
        clock=clock,
    )
    registry = SkillExecutorRegistry.bind_all_routes_to_port(
        manifest=manifest,
        execution_port=executor,
    )
    domain_port = _AuthoritativeStationPort(clock)
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        executor_registry=registry,
        domain_control_port=domain_port,
        payload_validators=_payload_validators(capability),
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=4
        ),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted

    operation_ids: list[str] = []
    for sequence in (1, 2):
        request = _request(
            manifest,
            capability,
            sequence=sequence,
            key=f"idem:{sequence}",
            request_id=f"request:{sequence}",
        )
        admitted = service.submit_request(
            request=request,
            current_station_revision=4,
            now=10.0,
        )
        assert admitted.accepted and admitted.operation is not None
        operation_id = admitted.operation.operation_id
        operation_ids.append(operation_id)
        launch = service.launch_operation(operation_id)
        assert launch.accepted
        assert executor.wait(operation_id, timeout_sec=1.0)
        completed = service.operation_store.get(operation_id)
        assert completed is not None and completed.status == "completed"
        assert completed.execution_result == {"operation_id": operation_id}

    assert reads == operation_ids
    assert tuple(item.operation_kind for item in executor.port_receipts) == (
        "activate",
        "activate",
    )
    assert domain_port.calls == []
    assert all(
        not receipt.proves_motion_or_hold
        for receipt in executor.port_receipts
    )


def test_domain_hold_during_passive_read_never_binds_read_to_motion_port() -> None:
    clock = _ManualClock()
    capability = _capability()
    manifest = _manifest(capability)
    read_started = threading.Event()
    release_read = threading.Event()

    def read(_request, operation) -> PassiveReadResult:
        read_started.set()
        assert release_read.wait(timeout=1.0)
        return PassiveReadResult(
            execution_result={"operation_id": operation.operation_id},
            evidence_refs=(f"passive-cut:{operation.operation_id}",),
        )

    executor = PassiveReadSkillExecutor(
        port_id="passive-reader:hold-isolation",
        executor_id="passive-reader-executor:hold-isolation",
        read=read,
        current_station_revision=lambda: 4,
        clock=clock,
    )
    domain_port = _AuthoritativeStationPort(clock)
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=_policy(),
        executor_registry=SkillExecutorRegistry.bind_all_routes_to_port(
            manifest=manifest,
            execution_port=executor,
        ),
        domain_control_port=domain_port,
        payload_validators=_payload_validators(capability),
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=4
        ),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    admitted = service.submit_request(
        request=_request(manifest, capability),
        current_station_revision=4,
        now=10.0,
    )
    assert admitted.accepted and admitted.operation is not None
    operation_id = admitted.operation.operation_id
    assert service.launch_operation(operation_id).accepted
    assert read_started.wait(timeout=1.0)

    held = service.request_hold(
        reason="fixture_domain_hold",
        current_station_revision=4,
        now=10.0,
    )
    assert held.accepted
    assert held.operation is not None and held.operation.status == "cancelled"
    assert held.port_receipt is not None
    assert held.port_receipt.operation_id is None
    assert domain_port.calls == ["hold"]
    assert tuple(item.operation_kind for item in executor.port_receipts) == (
        "activate",
    )

    release_read.set()
    assert executor.wait(operation_id, timeout_sec=1.0)
    terminal = service.operation_store.get(operation_id)
    assert terminal is not None and terminal.status == "cancelled"

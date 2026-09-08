from __future__ import annotations

from dataclasses import replace

from agent_gateway.protocol.contracts import (
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
    assess_skill_request_access,
)
from agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)


def _route() -> SkillExecutionRoute:
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
                parameter_profile_id="inspect-v1",
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
        cancellation_policy="cancel_only",
    )


def _manifest() -> SkillRouteManifest:
    route = _route()
    return SkillRouteManifest(
        runtime_session_id="runtime:b2a",
        control_domain_id="domain:arms",
        station_manifest_hash="station:b2a",
        execution_manifest_hash="execution:null-b2a",
        skill_definition_catalog_hash="definition-catalog:b2a",
        skill_schema_catalog_hash="schema-catalog:b2a",
        target_binding_id="target:null-b2a",
        target_kind="null",
        current_mode="dry_run",
        supported_transport_features=("idempotency_v1", "poll_status_v1"),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:null-b2a",
                    deployment_binding_hash="binding:null-b2a",
                    target_binding_id="target:null-b2a",
                    target_kind="null",
                    execution_manifest_hash="execution:null-b2a",
                ),
                evaluation_plan_hash=None,
                availability_hash="availability:null-ready",
            ),
        ),
        explicit_exclusions=("device_io", "operation_queue", "production"),
    )


def _policy() -> AgentSkillAccessPolicy:
    return AgentSkillAccessPolicy(
        access_policy_id="agent-access:b2a",
        allowed_client_ids=("rpent:b2a",),
        allowed_modes=("dry_run",),
        max_request_timeout_ms=1_000,
        max_grant_lifetime_sec=30.0,
    )


def _request(
    *,
    sequence: int = 1,
    key: str = "idem:1",
    request_id: str = "request:1",
    parameters: dict[str, object] | None = None,
    manifest: SkillRouteManifest | None = None,
    route: SkillExecutionRoute | None = None,
) -> AgentSkillRequest:
    manifest = _manifest() if manifest is None else manifest
    route = _route() if route is None else route
    entry = manifest.route(
        route.skill_id,
        route.skill_version,
        route.route_id,
        route.route_version,
    )
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="rpent:b2a",
        request_id=request_id,
        idempotency_key=key,
        client_sequence=sequence,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=4,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=entry.deployment_binding.deployment_binding_hash,
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters=parameters or {},
        observation_snapshot_id="snapshot:b2a:4",
        requested_mode=manifest.current_mode,
        requested_timeout_ms=500,
    )


def _admit(
    store: BoundedInMemoryAgentSkillOperationStore,
    request: AgentSkillRequest,
    *,
    now: float = 10.0,
    manifest: SkillRouteManifest | None = None,
):
    manifest = _manifest() if manifest is None else manifest
    decision = assess_skill_request_access(
        policy=_policy(),
        manifest=manifest,
        request=request,
        current_station_revision=4,
        now=now,
    )
    assert decision.accepted
    return store.admit(
        request=request,
        access_decision=decision,
        current_station_revision=4,
        admitted_at=now,
    )


def _fail(
    store: BoundedInMemoryAgentSkillOperationStore,
    operation,
    *,
    now: float,
):
    return store.transition_worker_projection(
        operation_id=operation.operation_id,
        expected_operation_hash=operation.semantic_content_hash,
        next_status="failed",
        reason_code="fake_worker_failed",
        current_station_revision=4,
        updated_at=now,
        progress_summary={"phase": "failed"},
    )


def test_same_idempotency_body_returns_original_operation_and_conflict_rejects() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    request = _request()
    first = _admit(store, request)
    replay = _admit(store, request, now=10.1)
    assert first.accepted
    assert replay.accepted
    assert replay.idempotent_replay
    assert replay.operation is first.operation
    assert store.retained_operation_count == 1

    changed = replace(request, parameters={"different": True})
    conflict = store.lookup_idempotency(changed)
    assert conflict is not None
    assert not conflict.accepted
    assert conflict.reason_code == "idempotency_key_conflict"
    assert conflict.operation is first.operation


def test_one_active_operation_rejects_busy_without_creating_a_queue() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    first = _admit(store, _request())
    second_request = _request(sequence=2, key="idem:2", request_id="request:2")
    busy = _admit(store, second_request, now=10.1)
    assert not busy.accepted
    assert busy.reason_code == "active_operation_exists_no_queue"
    assert store.retained_operation_count == 1

    failed = _fail(store, first.operation, now=10.2)
    assert failed.accepted
    admitted = _admit(store, second_request, now=10.3)
    assert admitted.accepted
    assert store.retained_operation_count == 2


def test_total_admission_limit_is_atomic_but_preserves_idempotent_replay() -> None:
    store = BoundedInMemoryAgentSkillOperationStore(max_total_admissions=1)
    first_request = _request()
    first = _admit(store, first_request)
    assert first.accepted
    assert _fail(store, first.operation, now=10.1).accepted

    replay = _admit(store, first_request, now=10.2)
    assert replay.accepted
    assert replay.idempotent_replay
    second = _admit(
        store,
        _request(sequence=2, key="idem:2", request_id="request:2"),
        now=10.3,
    )
    assert not second.accepted
    assert second.reason_code == "operation_admission_limit_reached"
    assert store.total_admitted == 1


def test_route_admission_limit_does_not_broaden_or_block_other_routes() -> None:
    first_route = _route()
    second_route = replace(
        first_route,
        skill_id="inspection.tcp_pose",
        skill_definition_hash="definition:inspection.tcp_pose:v1",
        route_id="null.fixture.tcp_pose_reader",
    )
    single_manifest = _manifest()
    second_entry = SkillRouteManifestEntry(
        route=second_route,
        deployment_binding=replace(
            single_manifest.skill_routes[0].deployment_binding,
            deployment_binding_id="deployment:null-tcp-pose-reader",
            deployment_binding_hash="binding:null-tcp-pose-reader",
        ),
        evaluation_plan_hash=None,
        availability_hash="availability:null-tcp-pose-ready",
    )
    manifest = replace(
        single_manifest,
        skill_routes=(single_manifest.skill_routes[0], second_entry),
    )
    store = BoundedInMemoryAgentSkillOperationStore(
        max_admissions_by_route_hash={
            first_route.semantic_content_hash: 1,
            second_route.semantic_content_hash: 2,
        }
    )

    first = _admit(
        store,
        _request(manifest=manifest, route=first_route),
        manifest=manifest,
    )
    assert first.accepted
    assert _fail(store, first.operation, now=10.1).accepted
    same_route = _admit(
        store,
        _request(
            sequence=2,
            key="idem:first:2",
            request_id="request:first:2",
            manifest=manifest,
            route=first_route,
        ),
        now=10.2,
        manifest=manifest,
    )
    assert not same_route.accepted
    assert same_route.reason_code == "route_admission_limit_reached"

    other_route = _admit(
        store,
        _request(
            sequence=3,
            key="idem:second:1",
            request_id="request:second:1",
            manifest=manifest,
            route=second_route,
        ),
        now=10.3,
        manifest=manifest,
    )
    assert other_route.accepted
    assert store.total_admitted == 2
    assert store.admissions_for_route_hash(first_route.semantic_content_hash) == 1
    assert store.admissions_for_route_hash(second_route.semantic_content_hash) == 1


def test_duplicate_or_regressed_sequence_is_rejected_after_terminal_operation() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    first = _admit(store, _request(sequence=3))
    assert _fail(store, first.operation, now=10.1).accepted

    duplicate = _admit(
        store,
        _request(sequence=3, key="idem:duplicate", request_id="request:duplicate"),
        now=10.2,
    )
    regressed = _admit(
        store,
        _request(sequence=2, key="idem:old", request_id="request:old"),
        now=10.2,
    )
    assert duplicate.reason_code == "duplicate_or_out_of_order_client_sequence"
    assert regressed.reason_code == "duplicate_or_out_of_order_client_sequence"
    assert store.active_operation_id is None


def test_terminal_history_eviction_is_bounded_and_keeps_sequence_guard() -> None:
    store = BoundedInMemoryAgentSkillOperationStore(max_retained_operations=1)
    first_request = _request()
    first = _admit(store, first_request)
    assert _fail(store, first.operation, now=10.1).accepted

    second = _admit(
        store,
        _request(sequence=2, key="idem:2", request_id="request:2"),
        now=10.2,
    )
    assert second.accepted
    assert store.retained_operation_count == 1
    assert store.total_evicted == 1
    assert store.get(first.operation.operation_id) is None
    assert store.lookup_idempotency(first_request) is None

    assert _fail(store, second.operation, now=10.3).accepted
    stale_retry = _admit(store, first_request, now=10.4)
    assert stale_retry.reason_code == "duplicate_or_out_of_order_client_sequence"


def test_worker_projection_uses_exact_operation_hash_cas() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    admitted = _admit(store, _request()).operation
    activated = store.transition_worker_projection(
        operation_id=admitted.operation_id,
        expected_operation_hash=admitted.semantic_content_hash,
        next_status="activated",
        reason_code="fake_activation_started",
        current_station_revision=4,
        updated_at=10.1,
        progress_summary={"phase": "activated"},
    )
    assert activated.accepted
    stale = store.transition_worker_projection(
        operation_id=admitted.operation_id,
        expected_operation_hash=admitted.semantic_content_hash,
        next_status="failed",
        reason_code="late_failure",
        current_station_revision=4,
        updated_at=10.2,
    )
    assert not stale.accepted
    assert stale.reason_code == "operation_hash_mismatch"
    assert stale.operation is activated.operation


def test_control_reservation_invalidates_late_result_before_port_confirmation() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    admitted = _admit(store, _request()).operation
    activated = store.transition_worker_projection(
        operation_id=admitted.operation_id,
        expected_operation_hash=admitted.semantic_content_hash,
        next_status="activated",
        reason_code="activated",
        current_station_revision=4,
        updated_at=10.1,
    ).operation
    running = store.transition_worker_projection(
        operation_id=activated.operation_id,
        expected_operation_hash=activated.semantic_content_hash,
        next_status="running",
        reason_code="running",
        current_station_revision=4,
        updated_at=10.2,
        lower_operation_id="fake-lower:1",
    ).operation
    reservation = store.reserve_control(
        operation_id=running.operation_id,
        control_kind="hold",
        terminal_status="cancelled",
        reason_code="hold_reserved",
        current_station_revision=5,
        updated_at=10.3,
    )
    assert reservation.accepted
    assert reservation.operation.status == "running"
    assert reservation.operation.semantic_content_hash != running.semantic_content_hash

    late_success = store.transition_worker_projection(
        operation_id=running.operation_id,
        expected_operation_hash=running.semantic_content_hash,
        next_status="completed",
        reason_code="late_success",
        current_station_revision=5,
        updated_at=10.4,
        execution_result={"success": True},
    )
    assert late_success.reason_code == "operation_hash_mismatch"

    finalized = store.finalize_control(
        operation_id=running.operation_id,
        expected_operation_hash=reservation.operation.semantic_content_hash,
        reason_code="fake_hold_confirmed",
        current_station_revision=5,
        updated_at=10.4,
        execution_receipt_refs=("fake-hold-receipt:1",),
    )
    assert finalized.accepted
    assert finalized.operation.status == "cancelled"
    assert store.active_operation_id is None


def test_emergency_stop_supersedes_an_existing_control_reservation() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    admitted = _admit(store, _request()).operation
    reserved = store.reserve_control(
        operation_id=admitted.operation_id,
        control_kind="cancel",
        terminal_status="cancelled",
        reason_code="cancel_reserved",
        current_station_revision=4,
        updated_at=10.1,
    )
    assert reserved.accepted

    stopped = store.terminate_after_emergency_stop(
        operation_id=admitted.operation_id,
        emergency_stop_confirmed=True,
        reason_code="operation_cancelled_after_emergency_stop",
        current_station_revision=4,
        updated_at=10.2,
        execution_receipt_refs=("emergency-stop:dual-arm",),
    )

    assert stopped.accepted
    assert stopped.operation.status == "cancelled"
    assert "emergency-stop:dual-arm" in (
        stopped.operation.execution_receipt_refs
    )
    assert store.active_operation_id is None
    late_cancel = store.finalize_control(
        operation_id=admitted.operation_id,
        expected_operation_hash=reserved.operation.semantic_content_hash,
        reason_code="late_cancel",
        current_station_revision=4,
        updated_at=10.3,
    )
    assert not late_cancel.accepted
    assert late_cancel.operation is stopped.operation


def test_failed_control_remains_active_and_blocks_worker_progress() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    admitted = _admit(store, _request()).operation
    reserved = store.reserve_control(
        operation_id=admitted.operation_id,
        control_kind="cancel",
        terminal_status="cancelled",
        reason_code="cancel_reserved",
        current_station_revision=4,
        updated_at=10.1,
    ).operation
    failure = store.record_control_failure(
        operation_id=admitted.operation_id,
        expected_operation_hash=reserved.semantic_content_hash,
        reason_code="cancel_not_confirmed",
        current_station_revision=4,
        updated_at=10.2,
    )
    assert failure.accepted
    assert not failure.operation.terminal
    assert store.active_operation_id == admitted.operation_id

    worker = store.transition_worker_projection(
        operation_id=admitted.operation_id,
        expected_operation_hash=failure.operation.semantic_content_hash,
        next_status="activated",
        reason_code="must_not_activate",
        current_station_revision=4,
        updated_at=10.3,
    )
    assert worker.reason_code == "operation_control_reserved"


def test_deadline_is_half_open_and_requires_explicit_timeout_control() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    admitted = _admit(store, _request(), now=10.0).operation
    assert admitted.deadline_at == 10.5
    assert store.due_active_operation(now=10.499) is None
    assert store.due_active_operation(now=10.5) is admitted

    activation = store.transition_worker_projection(
        operation_id=admitted.operation_id,
        expected_operation_hash=admitted.semantic_content_hash,
        next_status="activated",
        reason_code="too_late",
        current_station_revision=4,
        updated_at=10.5,
    )
    assert activation.reason_code == "operation_deadline_expired"


def test_station_revision_cannot_regress_between_operations_in_one_session() -> None:
    store = BoundedInMemoryAgentSkillOperationStore()
    first = _admit(store, _request()).operation
    terminal = store.transition_worker_projection(
        operation_id=first.operation_id,
        expected_operation_hash=first.semantic_content_hash,
        next_status="failed",
        reason_code="station_advanced",
        current_station_revision=5,
        updated_at=10.1,
    )
    assert terminal.accepted
    assert store.latest_station_revision == 5

    stale_revision = _admit(
        store,
        _request(sequence=2, key="idem:2", request_id="request:2"),
        now=10.2,
    )
    assert stale_revision.reason_code == "station_revision_regressed"
    assert store.active_operation_id is None

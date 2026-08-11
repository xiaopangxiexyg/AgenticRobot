from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import replace

from control_station.agent_gateway.protocol.contracts import (
    AgentObservationSnapshot,
    AgentSkillRequest,
)
from control_station.agent_gateway.protocol.skill_catalog import (
    SkillDeploymentBindingRef,
    SkillExecutionRoute,
    SkillParameterProfileRef,
    SkillRouteManifest,
    SkillRouteManifestEntry,
)
from control_station.agent_gateway.runtime.access_policy import AgentSkillAccessPolicy
from control_station.agent_gateway.runtime.operation_store import (
    BoundedInMemoryAgentSkillOperationStore,
)
from control_station.agent_gateway.runtime.payload_validation import (
    AgentSkillPayloadValidatorRegistry,
    AgentSkillPayloadValidatorSet,
)
from control_station.agent_gateway.runtime.reusable_station_executor import (
    ReusableStationAgentSkillOperationExecutor,
)
from control_station.agent_gateway.runtime.service import AgentGatewayService
from control_station.agent_gateway.runtime.station_operation_worker import (
    BoundedStationAgentSkillOperationWorker,
    StationLifecycleActivationEvidence,
    StationLifecycleControlEvidence,
    StationLifecycleEvaluationCompletion,
    StationLifecycleExecutionCompletion,
    StationOperationWorkerBinding,
)


class _ThreadSafeClock:
    def __init__(self, value: float = 10.0) -> None:
        self._value = value
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._value

    def advance(self, delta: float) -> None:
        with self._lock:
            self._value += float(delta)


class _FakeStationLifecycle:
    executor_id = "control-station.fake-lifecycle:v1"

    def __init__(
        self,
        *,
        block_activation: bool = False,
        block_run: bool = False,
        activation_error: bool = False,
        invalid_result: bool = False,
        evaluation: StationLifecycleEvaluationCompletion | None = None,
    ) -> None:
        self.activation_entered = threading.Event()
        self.activation_release = threading.Event()
        self.run_entered = threading.Event()
        self.run_release = threading.Event()
        if not block_activation:
            self.activation_release.set()
        if not block_run:
            self.run_release.set()
        self.activation_error = activation_error
        self.invalid_result = invalid_result
        self.evaluation = evaluation
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def _call(self, value: str) -> None:
        with self._lock:
            self.calls.append(value)

    def activate(self, *, request, operation, cancel_requested):
        assert request.semantic_content_hash == operation.request_hash
        self._call("activate")
        self.activation_entered.set()
        while not self.activation_release.wait(timeout=0.01):
            if cancel_requested():
                break
        if self.activation_error:
            raise RuntimeError("injected activation failure")
        if cancel_requested():
            return StationLifecycleActivationEvidence(
                succeeded=False,
                status="fake_activation_cancelled_under_hold",
                lower_operation_id=f"lower:{operation.operation_id}",
                evidence_refs=("lower-hold:activation-cancel",),
                lower_hold_confirmed=True,
            )
        return StationLifecycleActivationEvidence(
            succeeded=True,
            status="fake_station_application_started",
            lower_operation_id=f"lower:{operation.operation_id}",
            evidence_refs=("station-start:1",),
        )

    def run(self, *, request, operation, cancel_requested):
        assert request.semantic_content_hash == operation.request_hash
        self._call("run")
        self.run_entered.set()
        while not self.run_release.wait(timeout=0.01):
            if cancel_requested():
                break
        if cancel_requested():
            return StationLifecycleExecutionCompletion(
                succeeded=False,
                reason_code="fake_execution_preempted",
                execution_receipt_refs=("lower-hold:run-preempted",),
                terminal_hold_confirmed=True,
            )
        result = (
            {"unexpected": "payload"}
            if self.invalid_result
            else {"fixture_result": "complete"}
        )
        return StationLifecycleExecutionCompletion(
            succeeded=True,
            reason_code="fake_execution_completed_under_hold",
            execution_receipt_refs=("execution-chain:1", "lower-hold:terminal"),
            terminal_hold_confirmed=True,
            execution_result=result,
            evaluation=self.evaluation,
        )

    def cancel(self, *, request, operation, reason):
        assert request.semantic_content_hash == operation.request_hash
        assert reason
        self._call("cancel")
        self.activation_release.set()
        self.run_release.set()
        return StationLifecycleControlEvidence(
            succeeded=True,
            status="fake_station_cancel_confirmed",
            lower_operation_id=f"lower:{operation.operation_id}",
            evidence_refs=("lower-cancel:1",),
        )

    def request_hold(self, *, operation, reason):
        assert reason
        self._call("hold")
        self.activation_release.set()
        self.run_release.set()
        return StationLifecycleControlEvidence(
            succeeded=True,
            status="fake_lower_hold_confirmed",
            lower_operation_id=(
                None if operation is None else f"lower:{operation.operation_id}"
            ),
            evidence_refs=("lower-hold:1",),
            lower_hold_confirmed=True,
        )

    def close(self):
        self._call("close")


def _route(*, requires_task_evaluation: bool = False) -> SkillExecutionRoute:
    return SkillExecutionRoute(
        skill_id="task.fake_station_operation",
        skill_version="1",
        skill_definition_hash="definition:fake-station-operation:v1",
        route_id="station.fake.single_use",
        route_version="1",
        parameter_schema_hash="params:fake-empty:v1",
        execution_result_schema_hash="result:fake:v1",
        task_evaluation_schema_hash="evaluation:fake:v1",
        allowed_parameter_profiles=(
            SkillParameterProfileRef(
                parameter_profile_id="server-bound",
                parameter_profile_hash="profile:server-bound:v1",
                parameter_schema_hash="params:fake-empty:v1",
            ),
        ),
        control_effect_class="motion",
        privilege_tier="t1_sim_or_no_motion",
        supported_modes=("sim_active",),
        target_kind="sim",
        may_move_robot=True,
        may_open_publication=True,
        requires_operator_confirmation=False,
        requires_deadman=False,
        max_operation_duration_ms=1_000,
        cancellation_policy="hold_on_cancel_disconnect_or_timeout",
        requires_task_evaluation=requires_task_evaluation,
    )


def _manifest(route: SkillExecutionRoute) -> SkillRouteManifest:
    return SkillRouteManifest(
        runtime_session_id="runtime:station-worker-test",
        control_domain_id="domain:station-worker-test",
        station_manifest_hash="station-manifest:worker-test",
        execution_manifest_hash="execution-manifest:worker-test",
        skill_definition_catalog_hash="definition-catalog:worker-test",
        skill_schema_catalog_hash="schema-catalog:worker-test",
        target_binding_id="target:station-worker-test",
        target_kind="sim",
        current_mode="sim_active",
        supported_transport_features=("idempotency_v1", "poll_status_v1"),
        skill_routes=(
            SkillRouteManifestEntry(
                route=route,
                deployment_binding=SkillDeploymentBindingRef(
                    deployment_binding_id="deployment:station-worker-test",
                    deployment_binding_hash="binding:station-worker-test",
                    target_binding_id="target:station-worker-test",
                    target_kind="sim",
                    execution_manifest_hash="execution-manifest:worker-test",
                ),
                evaluation_plan_hash=(
                    "evaluation-plan:station-worker-test"
                    if route.requires_task_evaluation
                    else None
                ),
                availability_hash="availability:station-worker-test",
            ),
        ),
        explicit_exclusions=("device_io", "operation_queue", "second_owner"),
    )


def _validators(route: SkillExecutionRoute) -> AgentSkillPayloadValidatorRegistry:
    def validate_parameters(payload: Mapping[str, object]) -> None:
        if payload:
            raise ValueError("parameters must be empty")

    def validate_result(payload: Mapping[str, object]) -> None:
        if set(payload) != {"fixture_result"}:
            raise ValueError("invalid fixture result")

    def validate_evaluation(payload: Mapping[str, object]) -> None:
        if payload != {"verdict": "succeeded"}:
            raise ValueError("invalid fixture evaluation")

    return AgentSkillPayloadValidatorRegistry(
        registry_id="validators:station-worker-test",
        validators=(
            AgentSkillPayloadValidatorSet(
                validator_id="validator:station-worker-test",
                skill_id=route.skill_id,
                skill_version=route.skill_version,
                route_id=route.route_id,
                route_version=route.route_version,
                parameter_schema_hash=route.parameter_schema_hash,
                execution_result_schema_hash=route.execution_result_schema_hash,
                task_evaluation_schema_hash=route.task_evaluation_schema_hash,
                validate_parameters=validate_parameters,
                validate_execution_result=validate_result,
                validate_task_evaluation=validate_evaluation,
            ),
        ),
    )


def _snapshot(manifest: SkillRouteManifest) -> AgentObservationSnapshot:
    return AgentObservationSnapshot(
        snapshot_id="snapshot:station-worker-test:4",
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        station_revision=4,
        captured_at=9.8,
        available_at=9.9,
        clock_domain_id="agent-gateway-monotonic",
        authority_epoch=1,
        execution_epoch=1,
        dispatch_owner_generation=1,
        lifecycle_state="hold",
        execution_gate="hold",
        active_operation_id=None,
        source_binding_id=None,
        producer_epoch=None,
        target_binding_id=manifest.target_binding_id,
        execution_manifest_hash=manifest.execution_manifest_hash,
        frame_graph_hash="frame-graph:worker-test",
        tool_config_hash="tool:worker-test",
        calibration_hash="calibration:worker-test",
        measurement_summary={},
        measurement_freshness_ms=1,
        measurement_valid=True,
    )


def _request(manifest: SkillRouteManifest, route: SkillExecutionRoute) -> AgentSkillRequest:
    entry = manifest.skill_routes[0]
    profile = route.allowed_parameter_profiles[0]
    return AgentSkillRequest(
        client_id="rpent:station-worker-test",
        request_id="request:station-worker-test:1",
        idempotency_key="idempotency:station-worker-test:1",
        client_sequence=1,
        runtime_session_id=manifest.runtime_session_id,
        skill_route_manifest_hash=manifest.semantic_content_hash,
        expected_station_revision=4,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        skill_definition_hash=route.skill_definition_hash,
        route_id=route.route_id,
        route_version=route.route_version,
        route_hash=route.semantic_content_hash,
        deployment_binding_hash=(
            entry.deployment_binding.deployment_binding_hash
        ),
        parameter_profile_id=profile.parameter_profile_id,
        parameter_profile_hash=profile.parameter_profile_hash,
        parameters={},
        observation_snapshot_id="snapshot:station-worker-test:4",
        requested_mode="sim_active",
        requested_timeout_ms=900,
    )


def _setup(
    lifecycle: _FakeStationLifecycle,
    *,
    requires_task_evaluation: bool = False,
    clock: _ThreadSafeClock | None = None,
):
    clock = _ThreadSafeClock() if clock is None else clock
    route = _route(requires_task_evaluation=requires_task_evaluation)
    manifest = _manifest(route)
    binding = StationOperationWorkerBinding(
        worker_id="worker:station-worker-test:1",
        executor_id=lifecycle.executor_id,
        runtime_session_id=manifest.runtime_session_id,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        deployment_binding_hash=(
            manifest.skill_routes[0].deployment_binding.deployment_binding_hash
        ),
    )
    worker = BoundedStationAgentSkillOperationWorker(
        port_id="agent-gateway:station-worker-test",
        binding=binding,
        lifecycle=lifecycle,
        current_station_revision=lambda: 4,
        clock=clock,
    )
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="access:station-worker-test",
            allowed_client_ids=("rpent:station-worker-test",),
            allowed_modes=("sim_active",),
            max_request_timeout_ms=1_000,
            max_grant_lifetime_sec=30.0,
        ),
        operation_port=worker,
        payload_validators=_validators(route),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted
    admitted = service.submit_request(
        request=_request(manifest, route),
        current_station_revision=4,
        now=10.0,
    )
    assert admitted.accepted
    return service, worker, admitted.operation


def test_single_use_worker_completes_through_authoritative_service_cas() -> None:
    lifecycle = _FakeStationLifecycle()
    service, worker, admitted = _setup(lifecycle)
    launched = worker.launch(service=service, operation_id=admitted.operation_id)
    assert launched.accepted
    assert worker.wait(timeout_sec=1.0)

    completed = service.get_operation(admitted.operation_id)
    assert completed.status == "completed"
    assert completed.execution_result == {"fixture_result": "complete"}
    assert completed.lower_operation_id == f"lower:{admitted.operation_id}"
    assert worker.state == "finished"
    assert lifecycle.calls == ["activate", "run", "close"]
    replay = worker.launch(service=service, operation_id=admitted.operation_id)
    assert not replay.accepted
    assert replay.status == "operation_not_admitted"


def test_cancel_during_running_uses_out_of_band_hold_and_late_result_loses() -> None:
    lifecycle = _FakeStationLifecycle(block_run=True)
    service, worker, admitted = _setup(lifecycle)
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert lifecycle.run_entered.wait(timeout=1.0)

    cancelled = service.cancel_operation(
        operation_id=admitted.operation_id,
        reason="planner_cancelled",
        current_station_revision=4,
        now=10.0,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert cancelled.port_receipt.proves_motion_or_hold
    assert worker.wait(timeout_sec=1.0)
    assert service.get_operation(admitted.operation_id).status == "cancelled"
    assert worker.last_completion_outcome is not None
    assert not worker.last_completion_outcome.accepted
    assert lifecycle.calls.count("run") == 1
    assert lifecycle.calls.count("hold") >= 1


def test_cancel_during_activation_prevents_run_and_cannot_be_revived() -> None:
    lifecycle = _FakeStationLifecycle(block_activation=True)
    service, worker, admitted = _setup(lifecycle)
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert lifecycle.activation_entered.wait(timeout=1.0)

    cancelled = service.cancel_operation(
        operation_id=admitted.operation_id,
        reason="cancel_during_activation",
        current_station_revision=4,
        now=10.0,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    assert worker.wait(timeout_sec=1.0)
    assert service.get_operation(admitted.operation_id).status == "cancelled"
    assert "run" not in lifecycle.calls


def test_invalid_execution_payload_fails_closed_after_fresh_hold() -> None:
    lifecycle = _FakeStationLifecycle(invalid_result=True)
    service, worker, admitted = _setup(lifecycle)
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert worker.wait(timeout_sec=1.0)

    failed = service.get_operation(admitted.operation_id)
    assert failed.status == "failed"
    assert failed.execution_result is None
    assert failed.reason_code == "station_worker_execution_result_rejected"
    assert lifecycle.calls.count("hold") == 1


def test_activation_exception_records_lower_hold_before_terminal_failure() -> None:
    lifecycle = _FakeStationLifecycle(activation_error=True)
    service, worker, admitted = _setup(lifecycle)
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert worker.wait(timeout_sec=1.0)

    failed = service.get_operation(admitted.operation_id)
    assert failed.status == "failed"
    activation_receipt = worker.receipts[0]
    assert activation_receipt.operation_kind == "activate"
    assert not activation_receipt.succeeded
    assert activation_receipt.proves_motion_or_hold
    assert lifecycle.calls == ["activate", "hold", "close"]


def test_worker_can_commit_embedded_required_evaluation_after_execution() -> None:
    evaluation = StationLifecycleEvaluationCompletion(
        evaluation_status="completed",
        task_outcome="succeeded",
        reason_code="fake_required_environment_succeeded",
        evaluator_receipt_refs=("evaluator:environment:1",),
        aggregate_evaluation_receipt_ref="aggregate:environment:1",
        task_evaluation={"verdict": "succeeded"},
    )
    lifecycle = _FakeStationLifecycle(evaluation=evaluation)
    service, worker, admitted = _setup(
        lifecycle,
        requires_task_evaluation=True,
    )
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert worker.wait(timeout_sec=1.0)

    completed = service.get_operation(admitted.operation_id)
    assert completed.status == "completed"
    assert completed.execution_outcome == "completed"
    assert completed.evaluation_status == "completed"
    assert completed.task_outcome == "succeeded"
    assert worker.last_evaluation_outcome.accepted


def test_hold_before_worker_launch_closes_operation_without_starting_thread() -> None:
    lifecycle = _FakeStationLifecycle()
    service, worker, admitted = _setup(lifecycle)
    cancelled = service.cancel_operation(
        operation_id=admitted.operation_id,
        reason="cancel_before_launch",
        current_station_revision=4,
        now=10.0,
    )
    assert cancelled.accepted
    assert cancelled.operation.status == "cancelled"
    launch = worker.launch(service=service, operation_id=admitted.operation_id)
    assert not launch.accepted
    assert launch.status == "operation_not_admitted"
    assert lifecycle.calls == ["hold"]


def test_worker_deadline_preempts_running_result_through_timeout_hold() -> None:
    clock = _ThreadSafeClock()
    lifecycle = _FakeStationLifecycle(block_run=True)
    service, worker, admitted = _setup(lifecycle, clock=clock)
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert lifecycle.run_entered.wait(timeout=1.0)

    clock.advance(1.0)
    assert worker.wait(timeout_sec=1.0)
    timed_out = service.get_operation(admitted.operation_id)
    assert timed_out.status == "timed_out"
    assert worker.last_completion_outcome.accepted
    assert worker.last_completion_outcome.reason_code == (
        "operation_timed_out_after_station_control"
    )
    assert lifecycle.calls.count("hold") >= 1


def test_worker_deadline_preempts_activation_before_run() -> None:
    clock = _ThreadSafeClock()
    lifecycle = _FakeStationLifecycle(block_activation=True)
    service, worker, admitted = _setup(lifecycle, clock=clock)
    assert worker.launch(service=service, operation_id=admitted.operation_id).accepted
    assert lifecycle.activation_entered.wait(timeout=1.0)

    clock.advance(1.0)
    assert worker.wait(timeout_sec=1.0)
    timed_out = service.get_operation(admitted.operation_id)
    assert timed_out.status == "timed_out"
    assert "run" not in lifecycle.calls
    assert lifecycle.calls.count("hold") >= 1


def test_reusable_executor_creates_fresh_worker_for_sequential_operations() -> None:
    clock = _ThreadSafeClock()
    route = _route()
    manifest = _manifest(route)
    worker_binding = StationOperationWorkerBinding(
        worker_id="worker:reusable-station-test",
        executor_id=_FakeStationLifecycle.executor_id,
        runtime_session_id=manifest.runtime_session_id,
        skill_id=route.skill_id,
        skill_version=route.skill_version,
        route_id=route.route_id,
        route_version=route.route_version,
        deployment_binding_hash=(
            manifest.skill_routes[0].deployment_binding.deployment_binding_hash
        ),
    )
    lifecycles: list[_FakeStationLifecycle] = []

    def lifecycle_factory(*, request, operation):
        assert request.semantic_content_hash == operation.request_hash
        lifecycle = _FakeStationLifecycle()
        lifecycles.append(lifecycle)
        return lifecycle

    executor = ReusableStationAgentSkillOperationExecutor(
        port_id="agent-gateway:reusable-station-test",
        binding=worker_binding,
        lifecycle_factory=lifecycle_factory,
        domain_hold=lambda **_kwargs: StationLifecycleControlEvidence(
            succeeded=True,
            status="fake_idle_lower_hold_confirmed",
            lower_operation_id=None,
            evidence_refs=("lower-hold:idle",),
            lower_hold_confirmed=True,
        ),
        current_station_revision=lambda: 4,
        clock=clock,
    )
    service = AgentGatewayService(
        manifest=manifest,
        access_policy=AgentSkillAccessPolicy(
            access_policy_id="access:reusable-station-test",
            allowed_client_ids=("rpent:station-worker-test",),
            allowed_modes=("sim_active",),
            max_request_timeout_ms=1_000,
            max_grant_lifetime_sec=30.0,
        ),
        operation_port=executor,
        payload_validators=_validators(route),
        operation_store=BoundedInMemoryAgentSkillOperationStore(
            max_retained_operations=4,
            max_admissions_by_route_hash={route.semantic_content_hash: 4},
        ),
        clock=clock,
    )
    assert service.publish_observation(
        snapshot=_snapshot(manifest),
        now=10.0,
    ).accepted

    first_request = _request(manifest, route)
    first = service.submit_request(
        request=first_request,
        current_station_revision=4,
        now=10.0,
    )
    assert first.accepted and first.operation is not None
    assert service.launch_operation(first.operation.operation_id).accepted
    assert executor.wait(timeout_sec=1.0)
    assert service.get_operation(first.operation.operation_id).status == "completed"

    second_request = replace(
        first_request,
        request_id="request:station-worker-test:2",
        idempotency_key="idempotency:station-worker-test:2",
        client_sequence=2,
    )
    second = service.submit_request(
        request=second_request,
        current_station_revision=4,
        now=10.0,
    )
    assert second.accepted and second.operation is not None
    assert service.launch_operation(second.operation.operation_id).accepted
    assert executor.wait(timeout_sec=1.0)
    assert service.get_operation(second.operation.operation_id).status == "completed"

    assert len(lifecycles) == 2
    assert lifecycles[0] is not lifecycles[1]
    assert len(executor.workers) == 2
    assert all(worker.port_id == executor.port_id for worker in executor.workers)

    closed = service.close(current_station_revision=4, now=10.0)
    assert closed.accepted
    assert closed.port_receipt is not None
    assert closed.port_receipt.proves_motion_or_hold
    executor.close_after_hold(hold_receipt=closed.port_receipt)

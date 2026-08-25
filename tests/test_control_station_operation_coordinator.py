from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from control_station import (
    BoundedStationEventLog,
    ControlStationRunManifest,
    ControlStationRuntime,
    ExecutionDomainBinding,
    FakeSourceSlotAdapter,
    NoIoStationOperationCoordinator,
    NullExecutionTargetAdapter,
    SourceSlotSpec,
    StationIntentDraft,
    StationOperationCoordinatorUnavailable,
)
from control_station.web import (
    CS1B_SAFE_INTENT_KINDS,
    LocalIntentAccessPolicy,
    build_coordinated_control_station_server,
)


_TOKEN = "cs1c-test-token-0123456789-abcdef"


def _manifest() -> ControlStationRunManifest:
    source = SourceSlotSpec(
        source_slot_id="policy-primary",
        source_kind="policy",
        producer_id="policy:primary",
        source_action_contract_hash="policy-action-contract:v1",
        producer_epoch_policy_hash="producer-epoch-policy:v1",
        supported_uses=("shadow",),
        expected_emission_seam="policy-action-emission:v1",
        required_for_run=True,
    )
    target = ExecutionDomainBinding(
        binding_id="target:null",
        target_kind="null",
        control_domain_id="domain:arms",
        embodiment_instance_id="embodiment:null",
        target_runtime_action_contract_hash="target-action-contract:v1",
        execution_capability_profile_hash="execution-capability:null-v1",
        component_catalog_hash="component-catalog:arms-v1",
        selection_group_ids=("arms",),
        owner_registry_id="owner-registry:null",
        authority_epoch_source_id="authority-epoch:null",
        execution_epoch_source_id="execution-epoch:null",
        trace_requirement_profile_id="trace:null-v1",
    )
    return ControlStationRunManifest(
        manifest_id="control-station:cs1c-null-v1",
        manifest_version="1",
        execution_binding=target,
        source_slots=(source,),
        authority_policy_hash="authority-policy:no-motion-v1",
        decision_schedule_hash="decision-schedule:none-v1",
        trace_requirement_profile_id="trace:station-cs1c-v1",
        capture_policy_id="capture:none-v1",
        allowed_intent_kinds=CS1B_SAFE_INTENT_KINDS,
        operator_control_lease_sec=10.0,
        explicit_exclusions=("capture", "motion", "production_entrypoint"),
    )


def _runtime() -> ControlStationRuntime:
    return ControlStationRuntime(
        manifest=_manifest(),
        runtime_session_id="session:cs1c",
    )


def _adapters(runtime: ControlStationRuntime):
    return (
        NullExecutionTargetAdapter(
            adapter_id="target:null-adapter",
            target_binding_id=runtime.manifest.execution_binding.binding_id,
            clock=time.monotonic,
        ),
        FakeSourceSlotAdapter(
            adapter_id="source:policy-adapter",
            source_slot_id="policy-primary",
            clock=time.monotonic,
        ),
    )


def _draft(
    runtime: ControlStationRuntime,
    *,
    sequence: int,
    intent_kind: str,
    subject_id: str | None = None,
) -> StationIntentDraft:
    return StationIntentDraft(
        client_id="browser:test",
        client_sequence=sequence,
        intent_kind=intent_kind,
        produced_at=time.monotonic(),
        produced_clock_domain_id="browser-performance",
        expected_state_revision=runtime.state.state_revision,
        manifest_hash=runtime.manifest.semantic_content_hash,
        subject_id=subject_id,
        reason="cs1c-test",
    )


def _coordinator(
    runtime: ControlStationRuntime,
    *,
    adapters=None,
    event_log: BoundedStationEventLog | None = None,
    queue_capacity: int = 4,
) -> NoIoStationOperationCoordinator:
    return NoIoStationOperationCoordinator(
        coordinator_id="coordinator:cs1c",
        runtime=runtime,
        adapters=tuple(adapters or _adapters(runtime)),
        event_log=event_log,
        queue_capacity=queue_capacity,
    )


def _acquire_lease(
    coordinator: NoIoStationOperationCoordinator,
    runtime: ControlStationRuntime,
) -> None:
    submission = coordinator.submit_intent(
        _draft(runtime, sequence=0, intent_kind="acquire_control_lease")
    )
    assert submission.transition.receipt.accepted
    assert submission.enqueue_receipt.status == "operation_not_requested"


def test_bounded_event_log_reports_eviction_gap_without_backpressure() -> None:
    event_log = BoundedStationEventLog(capacity=2, clock=iter([1.0, 2.0, 3.0]).__next__)
    for index in range(3):
        event_log.publish(
            event_kind="fixture",
            source_fact_id=f"fact:{index}",
            source_fact_hash=f"hash:{index}",
            state_revision=index,
            status="observed",
        )
    snapshot = event_log.snapshot(after_sequence=-1)
    assert [event.event_sequence for event in snapshot.events] == [1, 2]
    assert snapshot.gap_detected is True
    assert snapshot.total_evicted == 1
    assert event_log.snapshot(after_sequence=1).gap_detected is False


def test_coordinator_requires_exact_no_io_adapter_coverage() -> None:
    runtime = _runtime()
    with pytest.raises(ValueError, match="exact manifest subjects"):
        _coordinator(runtime, adapters=_adapters(runtime)[:1])
    with pytest.raises(TypeError, match="DeterministicNoIo"):
        NoIoStationOperationCoordinator(
            coordinator_id="bad",
            runtime=runtime,
            adapters=(object(),),  # type: ignore[arg-type]
        )


def test_coordinator_executes_adapter_off_caller_and_admits_completion() -> None:
    runtime = _runtime()
    event_log = BoundedStationEventLog(capacity=16)
    adapters = _adapters(runtime)
    coordinator = _coordinator(runtime, adapters=adapters, event_log=event_log)
    coordinator.start()
    try:
        _acquire_lease(coordinator, runtime)
        submission = coordinator.submit_intent(
            _draft(
                runtime,
                sequence=1,
                intent_kind="connect_target",
                subject_id=runtime.manifest.execution_binding.binding_id,
            )
        )
        assert submission.enqueue_receipt.accepted is True
        assert submission.enqueue_receipt.status == "operation_enqueued"
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        assert runtime.state.target_state == "connected"
        assert not runtime.state.pending_operation_by_subject
        assert len(coordinator.completions) == 1
        assert adapters[0].perform_count == 1
        kinds = [event.event_kind for event in event_log.snapshot().events]
        assert kinds == [
            "station_intent_transition",
            "station_intent_transition",
            "station_operation_enqueued",
            "station_operation_completion",
        ]
    finally:
        assert coordinator.stop(timeout_sec=2.0)


class _BlockingTargetAdapter(NullExecutionTargetAdapter):
    def __init__(self, runtime: ControlStationRuntime) -> None:
        super().__init__(
            adapter_id="target:blocking",
            target_binding_id=runtime.manifest.execution_binding.binding_id,
            clock=time.monotonic,
        )
        self.entered = threading.Event()
        self.release = threading.Event()

    def perform(self, intent):
        self.entered.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("blocking test adapter timed out")
        return super().perform(intent)


class _RaisingTargetAdapter(NullExecutionTargetAdapter):
    def perform(self, intent):
        raise RuntimeError("test-only adapter failure details")


def test_capacity_is_reserved_before_admission_and_cannot_strand_pending_work() -> None:
    runtime = _runtime()
    target = _BlockingTargetAdapter(runtime)
    source = _adapters(runtime)[1]
    coordinator = _coordinator(
        runtime,
        adapters=(target, source),
        queue_capacity=1,
    )
    coordinator.start()
    try:
        _acquire_lease(coordinator, runtime)
        coordinator.submit_intent(
            _draft(
                runtime,
                sequence=1,
                intent_kind="connect_target",
                subject_id=runtime.manifest.execution_binding.binding_id,
            )
        )
        assert target.entered.wait(timeout=1.0)
        sequence_before = runtime.next_admission_sequence
        revision_before = runtime.state.state_revision
        with pytest.raises(
            StationOperationCoordinatorUnavailable,
            match="operation_coordinator_queue_full",
        ):
            coordinator.submit_intent(
                _draft(
                    runtime,
                    sequence=2,
                    intent_kind="connect_source",
                    subject_id="policy-primary",
                )
            )
        assert runtime.next_admission_sequence == sequence_before
        assert runtime.state.state_revision == revision_before
        assert runtime.state.source_state_by_slot["policy-primary"] == "disconnected"
        target.release.set()
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        assert runtime.state.target_state == "connected"
    finally:
        target.release.set()
        assert coordinator.stop(timeout_sec=2.0)


def test_adapter_exception_becomes_typed_failure_and_worker_survives() -> None:
    runtime = _runtime()
    target = _RaisingTargetAdapter(
        adapter_id="target:raising",
        target_binding_id=runtime.manifest.execution_binding.binding_id,
        clock=time.monotonic,
    )
    coordinator = _coordinator(runtime, adapters=(target, _adapters(runtime)[1]))
    coordinator.start()
    try:
        _acquire_lease(coordinator, runtime)
        coordinator.submit_intent(
            _draft(
                runtime,
                sequence=1,
                intent_kind="connect_target",
                subject_id=runtime.manifest.execution_binding.binding_id,
            )
        )
        assert coordinator.wait_until_idle(timeout_sec=2.0)
        assert coordinator.is_running
        assert runtime.state.target_state == "configured"
        completion = coordinator.completions[0].completion_transition
        assert completion.accepted is True
        assert completion.result.succeeded is False
        assert completion.result.status == "adapter_exception:RuntimeError"
        assert "failure details" not in completion.result.status
    finally:
        assert coordinator.stop(timeout_sec=2.0)


def _policy() -> LocalIntentAccessPolicy:
    return LocalIntentAccessPolicy(
        access_policy_id="local-intent-policy:cs1c-test",
        bearer_token=_TOKEN,
    )


@contextmanager
def _running_server(
    runtime: ControlStationRuntime,
    coordinator: NoIoStationOperationCoordinator,
    event_log: BoundedStationEventLog,
):
    server = build_coordinated_control_station_server(
        runtime,
        access_policy=_policy(),
        event_log=event_log,
        operation_coordinator=coordinator,
        port=0,
    )
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.05},
        daemon=True,
    )
    thread.start()
    try:
        host, port = server.server_address[:2]
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)


def _post_json(root: str, payload: dict[str, object]):
    request = Request(
        f"{root}/api/intents",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {_TOKEN}",
            "Content-Type": "application/json",
        },
    )
    try:
        response = urlopen(request, timeout=2.0)  # noqa: S310 - loopback fixture
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))
    with response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _http_payload(
    runtime: ControlStationRuntime,
    *,
    sequence: int,
    intent_kind: str,
    subject_id: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "client_id": "browser:test",
        "client_sequence": sequence,
        "intent_kind": intent_kind,
        "produced_at": time.monotonic(),
        "produced_clock_domain_id": "browser-performance",
        "expected_state_revision": runtime.state.state_revision,
        "manifest_hash": runtime.manifest.semantic_content_hash,
        "reason": "cs1c-http-test",
    }
    if subject_id is not None:
        result["subject_id"] = subject_id
    return result


def test_http_only_enqueues_operation_and_events_are_rebuildable_cache() -> None:
    runtime = _runtime()
    event_log = BoundedStationEventLog(capacity=16)
    target = _BlockingTargetAdapter(runtime)
    coordinator = _coordinator(
        runtime,
        adapters=(target, _adapters(runtime)[1]),
        event_log=event_log,
    )
    coordinator.start()
    try:
        with _running_server(runtime, coordinator, event_log) as root:
            status, _ = _post_json(
                root,
                _http_payload(
                    runtime,
                    sequence=0,
                    intent_kind="acquire_control_lease",
                ),
            )
            assert status == 200
            status, body = _post_json(
                root,
                _http_payload(
                    runtime,
                    sequence=1,
                    intent_kind="connect_target",
                    subject_id=runtime.manifest.execution_binding.binding_id,
                ),
            )
            assert status == 200
            assert target.entered.wait(timeout=1.0)
            assert body["adapter_executed_inline"] is False
            assert body["operation_enqueue"]["status"] == "operation_enqueued"
            assert runtime.state.target_state == "connect_requested"
            with urlopen(f"{root}/api/events?after_sequence=-1", timeout=2.0) as response:  # noqa: S310
                events = json.loads(response.read().decode("utf-8"))
            assert events["authoritative"] is False
            assert events["motion_authorized"] is False
            assert any(
                event["event_kind"] == "station_operation_enqueued"
                for event in events["events"]
            )
            target.release.set()
            assert coordinator.wait_until_idle(timeout_sec=2.0)
            with urlopen(f"{root}/api/state", timeout=2.0) as response:  # noqa: S310
                state = json.loads(response.read().decode("utf-8"))
            assert state["maturity"] == "cs1c_no_io_operation_coordination"
            assert state["capabilities"]["operation_coordinator"] is True
            assert state["capabilities"]["motion"] is False
    finally:
        target.release.set()
        assert coordinator.stop(timeout_sec=2.0)


def test_http_busy_rejection_occurs_before_runtime_admission() -> None:
    runtime = _runtime()
    event_log = BoundedStationEventLog(capacity=8)
    coordinator = _coordinator(runtime, event_log=event_log, queue_capacity=1)
    with _running_server(runtime, coordinator, event_log) as root:
        status, _ = _post_json(
            root,
            _http_payload(
                runtime,
                sequence=0,
                intent_kind="acquire_control_lease",
            ),
        )
        assert status == 200
        sequence_before = runtime.next_admission_sequence
        status, body = _post_json(
            root,
            _http_payload(
                runtime,
                sequence=1,
                intent_kind="connect_target",
                subject_id=runtime.manifest.execution_binding.binding_id,
            ),
        )
    assert status == 503
    assert body["error"] == "operation_coordinator_not_running"
    assert body["intent_admitted"] is False
    assert runtime.next_admission_sequence == sequence_before
    assert runtime.state.target_state == "configured"

from __future__ import annotations

from dataclasses import dataclass

import pytest

from control_station import (
    ControlStationRunManifest,
    ControlStationRuntime,
    DeterministicNoIoStationAdapter,
    ExecutionDomainBinding,
    NullExecutionTargetAdapter,
    SourceSlotSpec,
    StationCompositionSessionOperationPort,
    StationIntentDraft,
    StationPendingOperationSuperseded,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


class _Clock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


@dataclass(frozen=True)
class _Evidence:
    operation_kind: str
    required_evidence_refs: tuple[str, ...]
    causal_ready_at: float
    causal_clock_domain_id: str = "control.monotonic"


def _manifest() -> ControlStationRunManifest:
    return ControlStationRunManifest(
        manifest_id="test:generic-composition-session-port",
        manifest_version="development",
        execution_binding=ExecutionDomainBinding(
            binding_id="target:sim",
            target_kind="sim",
            control_domain_id="domain:arms",
            embodiment_instance_id="embodiment:test",
            target_runtime_action_contract_hash="target-action:test",
            execution_capability_profile_hash="execution-capability:test",
            component_catalog_hash="component-catalog:test",
            selection_group_ids=("arms",),
            owner_registry_id="owner-registry:test",
            authority_epoch_source_id="authority-epoch:test",
            execution_epoch_source_id="execution-epoch:test",
            trace_requirement_profile_id="trace:target:test",
        ),
        source_slots=(
            SourceSlotSpec(
                source_slot_id="composer",
                source_kind="composite",
                producer_id="composer:test",
                source_action_contract_hash="source-action:test",
                producer_epoch_policy_hash="producer-epoch:test",
                supported_uses=("candidate", "authority"),
                expected_emission_seam="emission:composer:test",
            ),
        ),
        authority_policy_hash="authority-policy:test",
        decision_schedule_hash="decision-schedule:test",
        trace_requirement_profile_id="trace:station:test",
        capture_policy_id="capture:none",
        allowed_intent_kinds=(
            "acquire_control_lease",
            "connect_source",
            "connect_target",
            "run_preflight",
            "request_candidate",
            "start_composition_session",
            "rearm_composition_session",
            "hold",
        ),
    )


def _pending_start():
    clock = _Clock()
    manifest = _manifest()
    runtime = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id="station-session:test",
        clock_domain_id="control.monotonic",
        clock=clock,
    )
    target = NullExecutionTargetAdapter(
        adapter_id="target-adapter:test",
        target_binding_id=manifest.execution_binding.binding_id,
        clock=clock,
        clock_domain_id=runtime.clock_domain_id,
    )
    source = DeterministicNoIoStationAdapter(
        adapter_id="source-adapter:test",
        subject_id="composer",
        supported_operations=(
            "connect_source",
            "request_candidate",
            "run_preflight",
        ),
        clock=clock,
        clock_domain_id=runtime.clock_domain_id,
    )
    sequence = 0

    def submit(kind: str, *, subject: str | None = None, spec: str | None = None):
        nonlocal sequence
        transition = runtime.submit_intent(
            StationIntentDraft(
                client_id="operator:test",
                client_sequence=sequence,
                intent_kind=kind,
                produced_at=clock(),
                produced_clock_domain_id=runtime.clock_domain_id,
                expected_state_revision=runtime.state.state_revision,
                manifest_hash=manifest.semantic_content_hash,
                subject_id=subject,
                resource_binding_id=(
                    manifest.execution_binding.binding_id
                    if kind
                    in {"start_composition_session", "rearm_composition_session"}
                    else None
                ),
                operation_spec_id=spec,
                reason="test",
            )
        )
        sequence += 1
        if transition.receipt.status == "operation_requested" and kind not in {
            "start_composition_session",
            "rearm_composition_session",
        }:
            adapter = target if subject == target.subject_id else source
            completion = runtime.execute_pending_operation(transition, adapter)
            assert completion.completion_transition.accepted
        return transition

    submit("acquire_control_lease")
    submit("connect_target", subject=target.subject_id)
    submit("connect_source", subject=source.subject_id)
    submit("run_preflight", subject=target.subject_id)
    submit("run_preflight", subject=source.subject_id)
    submit("request_candidate", subject=source.subject_id)
    request = submit(
        "start_composition_session",
        subject=source.subject_id,
        spec="composition-operation:test",
    )
    assert request.receipt.status == "operation_requested"
    return runtime, clock, request, submit


def test_generic_port_completes_exact_pending_operation_once() -> None:
    runtime, clock, request, _submit = _pending_start()
    port = StationCompositionSessionOperationPort(
        runtime=runtime,
        request_transition=request,
    )
    clock.value = 10.1

    result = port(
        _Evidence(
            operation_kind="start_composition_session",
            required_evidence_refs=("activation:1", "preflight:1", "activation:1"),
            causal_ready_at=10.05,
        )
    )

    assert port.completed
    assert port.result is result
    assert runtime.state.lifecycle_state == "running"
    assert runtime.state.source_state_by_slot["composer"] == "active"
    assert runtime.state.pending_operation_by_subject == {}
    assert set(result.evidence_refs) == {
        "activation:1",
        "preflight:1",
        "composition-operation:test",
    }
    with pytest.raises(RuntimeError, match="already completed"):
        port(
            _Evidence(
                operation_kind="start_composition_session",
                required_evidence_refs=("activation:1",),
                causal_ready_at=10.05,
            )
        )


def test_generic_port_rejects_wrong_kind_clock_and_future_evidence() -> None:
    runtime, clock, request, _submit = _pending_start()
    port = StationCompositionSessionOperationPort(
        runtime=runtime,
        request_transition=request,
    )

    with pytest.raises(ValueError, match="another operation kind"):
        port(
            _Evidence(
                operation_kind="rearm_composition_session",
                required_evidence_refs=("activation:1",),
                causal_ready_at=10.0,
            )
        )
    with pytest.raises(ValueError, match="Cross-clock"):
        port(
            _Evidence(
                operation_kind="start_composition_session",
                required_evidence_refs=("activation:1",),
                causal_ready_at=10.0,
                causal_clock_domain_id="device.monotonic",
            )
        )
    with pytest.raises(ValueError, match="predates lower readiness"):
        port(
            _Evidence(
                operation_kind="start_composition_session",
                required_evidence_refs=("activation:1",),
                causal_ready_at=10.1,
            )
        )

    assert port.completion_transition is None
    assert runtime.state.pending_operation_by_subject
    clock.value = 10.1
    assert port(
        _Evidence(
            operation_kind="start_composition_session",
            required_evidence_refs=("activation:1",),
            causal_ready_at=10.1,
        )
    ).succeeded


def test_generic_port_can_resolve_lower_preparation_failure() -> None:
    runtime, clock, request, _submit = _pending_start()
    port = StationCompositionSessionOperationPort(
        runtime=runtime,
        request_transition=request,
    )
    clock.value = 10.1

    result = port.complete_failure(
        status="allocation_activation_failed:ReadinessError",
        evidence_refs=("preflight:failed",),
    )

    assert not result.succeeded
    assert port.completed
    assert runtime.state.lifecycle_state == "fault"
    assert runtime.state.pending_operation_by_subject == {}
    assert set(result.evidence_refs) == {
        "preflight:failed",
        "composition-operation:test",
    }


def test_generic_port_never_revives_a_station_superseded_request() -> None:
    runtime, _clock, request, submit = _pending_start()
    port = StationCompositionSessionOperationPort(
        runtime=runtime,
        request_transition=request,
    )
    hold = submit("hold")
    assert hold.receipt.accepted

    with pytest.raises(
        StationPendingOperationSuperseded,
        match="station_pending_operation_superseded",
    ):
        port(
            _Evidence(
                operation_kind="start_composition_session",
                required_evidence_refs=("activation:1",),
                causal_ready_at=10.0,
            )
        )

    assert runtime.state.lifecycle_state == "hold"
    assert port.completion_transition is None

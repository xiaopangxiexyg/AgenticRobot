from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from control_composition.lifecycle import (
    CompositionSessionOperationSpec,
    CompositionSessionRearmReceipt,
    CompositionSessionRearmSpec,
    advance_composition_session,
    initial_composition_session_state,
)
from control_station import (
    COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1,
    COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1,
    CompositionSessionController,
    ControlStationRunManifest,
    ControlStationRuntime,
    ExecutionDomainBinding,
    FakeSourceSlotAdapter,
    HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1,
    HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1,
    HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1,
    NullExecutionTargetAdapter,
    SourceSlotSpec,
    StationIntentDraft,
    StationOperationResultDraft,
    admit_station_operation_result,
    admit_station_intent,
    apply_station_operation_result,
    apply_station_intent,
    initial_control_station_state,
)
from dataset_interface.composition_control_trace import (
    project_composition_session_rearm_receipt,
    project_composition_session_rearm_spec,
)
from control_station.targets.authoritative import (
    authoritative_authority_epoch_source_id,
)


def _source_slot(
    source_slot_id: str = "policy-primary",
    *,
    source_kind: str = "policy",
    producer_id: str = "policy:primary",
    required_for_run: bool = True,
    supported_uses: tuple[str, ...] = ("shadow", "candidate", "authority"),
) -> SourceSlotSpec:
    return SourceSlotSpec(
        source_slot_id=source_slot_id,
        source_kind=source_kind,
        producer_id=producer_id,
        source_action_contract_hash=f"action-contract:{source_slot_id}",
        producer_epoch_policy_hash=f"epoch-policy:{source_slot_id}",
        supported_uses=supported_uses,
        expected_emission_seam=f"emission:{source_slot_id}",
        required_binding_refs=("binding:z", "binding:a"),
        trace_requirement_refs=("trace:proposal",),
        required_for_run=required_for_run,
    )


def _target(*, target_kind: str = "null") -> ExecutionDomainBinding:
    return ExecutionDomainBinding(
        binding_id=f"target:{target_kind}",
        target_kind=target_kind,
        control_domain_id="domain:arms",
        embodiment_instance_id=f"embodiment:{target_kind}",
        target_runtime_action_contract_hash="target-action-contract:v1",
        execution_capability_profile_hash="execution-capability:v1",
        component_catalog_hash="component-catalog:v1",
        selection_group_ids=("arms",),
        owner_registry_id="owner-registry:shared",
        authority_epoch_source_id="authority-epoch:shared",
        execution_epoch_source_id="execution-epoch:shared",
        trace_requirement_profile_id=f"trace:{target_kind}",
        backend_inventory_hash=("inventory:real" if target_kind == "real" else None),
    )


def _manifest(
    *,
    target_kind: str = "null",
    slots: tuple[SourceSlotSpec, ...] | None = None,
) -> ControlStationRunManifest:
    allowed = (
        "acquire_control_lease",
        "release_control_lease",
        "connect_source",
        "disconnect_source",
        "connect_target",
        "disconnect_target",
        "run_preflight",
        "start_shadow",
        "stop_shadow",
        "request_motion_authority",
        "hold",
        "estop",
        "stop_runtime",
    )
    return ControlStationRunManifest(
        manifest_id="control-station:null-policy-v1",
        manifest_version="1",
        execution_binding=_target(target_kind=target_kind),
        source_slots=slots or (_source_slot(),),
        authority_policy_hash="authority-policy:v1",
        decision_schedule_hash="decision-schedule:v1",
        trace_requirement_profile_id="trace:station-null-v1",
        capture_policy_id="capture:none-v1",
        allowed_intent_kinds=allowed,
        operator_control_lease_sec=10.0,
        explicit_exclusions=("real_motion", "episode_owner"),
    )


def _intent(
    manifest: ControlStationRunManifest,
    state,
    *,
    sequence: int,
    kind: str,
    subject_id: str | None = None,
    resource_binding_id: str | None = None,
    operation_spec_id: str | None = None,
    client_id: str = "operator:a",
    at: float | None = None,
):
    timestamp = 1.0 + sequence if at is None else float(at)
    draft = StationIntentDraft(
        client_id=client_id,
        client_sequence=sequence,
        intent_kind=kind,
        produced_at=timestamp,
        produced_clock_domain_id="browser-monotonic",
        expected_state_revision=state.state_revision,
        manifest_hash=manifest.semantic_content_hash,
        subject_id=subject_id,
        resource_binding_id=resource_binding_id,
        operation_spec_id=operation_spec_id,
        reason="unit-test",
    )
    return admit_station_intent(
        draft,
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        available_at=timestamp,
        available_clock_domain_id="control-monotonic",
        admission_sequence=sequence,
    )


def _apply(manifest, state, *, sequence: int, kind: str, subject_id=None, **kwargs):
    transition = apply_station_intent(
        manifest,
        state,
        _intent(
            manifest,
            state,
            sequence=sequence,
            kind=kind,
            subject_id=subject_id,
            **kwargs,
        ),
    )
    return transition.state_after, transition


def _complete(manifest, state, request_transition, *, succeeded: bool = True):
    intent = request_transition.intent
    draft = StationOperationResultDraft(
        source_intent_id=intent.intent_id,
        operation_kind=intent.intent_kind,
        subject_id=intent.subject_id,
        succeeded=succeeded,
        completed_at=intent.available_at + 0.01,
        completed_clock_domain_id="adapter-monotonic",
        status="completed" if succeeded else "adapter_failed",
        evidence_refs=(f"evidence:{intent.intent_id}",),
    )
    result = admit_station_operation_result(
        draft,
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        available_at=intent.available_at + 0.02,
        available_clock_domain_id="control-monotonic",
        admission_sequence=1000 + intent.admission_sequence,
    )
    transition = apply_station_operation_result(manifest, state, result)
    return transition.state_after, transition


def _human_takeover_manifest() -> ControlStationRunManifest:
    replay = _source_slot(
        "replay-primary",
        source_kind="replay",
        producer_id="replay:primary",
        supported_uses=("candidate", "authority"),
    )
    human = _source_slot(
        "human-primary",
        source_kind="human",
        producer_id="human:primary",
        supported_uses=("candidate", "authority"),
    )
    base = _manifest(target_kind="sim", slots=(replay, human))
    return replace(
        base,
        manifest_id="control-station:sim-replay-human-takeover-v1",
        allowed_intent_kinds=tuple(
            sorted(
                set(base.allowed_intent_kinds)
                | {"release_candidate", "request_human_takeover"}
            )
        ),
    )


def _human_takeover_ready_state(manifest: ControlStationRunManifest):
    state = initial_control_station_state(manifest, runtime_session_id="session:takeover")
    state, lease_transition = _apply(
        manifest,
        state,
        sequence=0,
        kind="acquire_control_lease",
    )
    assert lease_transition.receipt.accepted
    assert lease_transition.receipt.intent_hash is None
    return replace(
        state,
        lifecycle_state="running",
        target_state="preflighted",
        source_state_by_slot={
            "replay-primary": "active",
            "human-primary": "candidate",
        },
    )


def _human_takeover_completion(
    manifest: ControlStationRunManifest,
    state,
    request_transition,
    *,
    succeeded: bool,
    status: str,
    evidence_refs: tuple[str, ...] = ("evidence:human-takeover-aggregate",),
):
    intent = request_transition.intent
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id=intent.intent_id,
            operation_kind=intent.intent_kind,
            subject_id=intent.subject_id,
            succeeded=succeeded,
            completed_at=intent.available_at + 0.01,
            completed_clock_domain_id="adapter-monotonic",
            status=status,
            evidence_refs=evidence_refs,
        ),
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        available_at=intent.available_at + 0.02,
        available_clock_domain_id="control-monotonic",
        admission_sequence=1000 + intent.admission_sequence,
    )
    return apply_station_operation_result(manifest, state, result)


def test_manifest_is_order_independent_and_binds_exact_target() -> None:
    policy = _source_slot()
    human = _source_slot(
        "human-optional",
        source_kind="human",
        producer_id="human:optional",
        required_for_run=False,
        supported_uses=("record_only", "shadow"),
    )
    first = _manifest(slots=(policy, human))
    second = _manifest(slots=(human, policy))
    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.source_slots[0].source_slot_id == "human-optional"

    sim = _manifest(target_kind="sim", slots=(human, policy))
    assert sim.semantic_content_hash != first.semantic_content_hash


def test_real_target_requires_explicit_backend_inventory() -> None:
    with pytest.raises(ValueError, match="backend_inventory_hash"):
        ExecutionDomainBinding(
            binding_id="target:real",
            target_kind="real",
            control_domain_id="domain:arms",
            embodiment_instance_id="embodiment:real",
            target_runtime_action_contract_hash="target-action-contract:v1",
            execution_capability_profile_hash="execution-capability:v1",
            component_catalog_hash="component-catalog:v1",
            selection_group_ids=("arms",),
            owner_registry_id="owner-registry:shared",
            authority_epoch_source_id="authority-epoch:shared",
            execution_epoch_source_id="execution-epoch:shared",
            trace_requirement_profile_id="trace:real",
        )


def test_station_intent_has_no_raw_action_payload_escape_hatch() -> None:
    manifest = _manifest()
    with pytest.raises(TypeError, match="unexpected keyword argument 'qpos'"):
        StationIntentDraft(
            client_id="operator:a",
            client_sequence=0,
            intent_kind="start_typed_motion_test",
            produced_at=1.0,
            produced_clock_domain_id="browser-monotonic",
            expected_state_revision=0,
            manifest_hash=manifest.semantic_content_hash,
            operation_spec_id="bounded-test:v1",
            qpos=[0.0] * 12,
        )


def test_cs0_connect_preflight_and_shadow_lifecycle() -> None:
    manifest = _manifest()
    state = initial_control_station_state(manifest, runtime_session_id="session:station")
    assert state.lifecycle_state == "configured"

    state, transition = _apply(
        manifest,
        state,
        sequence=0,
        kind="acquire_control_lease",
    )
    assert transition.receipt.accepted
    assert state.operator_control_lease is not None

    state, request = _apply(
        manifest,
        state,
        sequence=1,
        kind="connect_target",
        subject_id=manifest.execution_binding.binding_id,
    )
    assert state.target_state == "connect_requested"
    assert state.lifecycle_state == "configured"
    state, transition = _complete(manifest, state, request)
    assert transition.accepted
    assert state.target_state == "connected"

    state, request = _apply(
        manifest,
        state,
        sequence=2,
        kind="connect_source",
        subject_id="policy-primary",
    )
    assert state.source_state_by_slot["policy-primary"] == "connect_requested"
    state, _ = _complete(manifest, state, request)
    assert state.lifecycle_state == "connected"

    state, request = _apply(
        manifest,
        state,
        sequence=3,
        kind="run_preflight",
        subject_id=manifest.execution_binding.binding_id,
    )
    state, _ = _complete(manifest, state, request)
    state, request = _apply(
        manifest,
        state,
        sequence=4,
        kind="run_preflight",
        subject_id="policy-primary",
    )
    state, _ = _complete(manifest, state, request)
    assert state.lifecycle_state == "preflighted"

    state, request = _apply(
        manifest,
        state,
        sequence=5,
        kind="start_shadow",
        subject_id="policy-primary",
    )
    assert request.receipt.status == "operation_requested"
    assert state.source_state_by_slot["policy-primary"] == "shadow_start_requested"
    assert state.lifecycle_state == "preflighted"
    state, transition = _complete(manifest, state, request)
    assert transition.accepted
    assert state.lifecycle_state == "shadow_ready"
    assert state.source_state_by_slot["policy-primary"] == "shadow"

    _, rejected = _apply(
        manifest,
        state,
        sequence=6,
        kind="disconnect_target",
        subject_id=manifest.execution_binding.binding_id,
    )
    assert not rejected.receipt.accepted
    assert rejected.receipt.status == "active_source_prevents_target_disconnect"


def test_state_revision_and_operator_lease_fail_closed() -> None:
    manifest = _manifest()
    state = initial_control_station_state(manifest, runtime_session_id="session:station")
    state, _ = _apply(
        manifest,
        state,
        sequence=0,
        kind="acquire_control_lease",
    )

    stale_draft = StationIntentDraft(
        client_id="operator:a",
        client_sequence=1,
        intent_kind="connect_target",
        produced_at=2.0,
        produced_clock_domain_id="browser-monotonic",
        expected_state_revision=0,
        manifest_hash=manifest.semantic_content_hash,
        subject_id=manifest.execution_binding.binding_id,
    )
    stale = admit_station_intent(
        stale_draft,
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        available_at=2.0,
        available_clock_domain_id="control-monotonic",
        admission_sequence=1,
    )
    rejected = apply_station_intent(manifest, state, stale)
    assert not rejected.receipt.accepted
    assert rejected.receipt.status == "stale_state_revision"
    assert rejected.state_after is state

    _, other_client = _apply(
        manifest,
        state,
        sequence=2,
        kind="connect_target",
        subject_id=manifest.execution_binding.binding_id,
        client_id="operator:b",
    )
    assert not other_client.receipt.accepted
    assert other_client.receipt.status == "operator_control_lease_required"

    duplicate_draft = StationIntentDraft(
        client_id="operator:a",
        client_sequence=0,
        intent_kind="connect_target",
        produced_at=3.0,
        produced_clock_domain_id="browser-monotonic",
        expected_state_revision=state.state_revision,
        manifest_hash=manifest.semantic_content_hash,
        subject_id=manifest.execution_binding.binding_id,
    )
    duplicate = admit_station_intent(
        duplicate_draft,
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        available_at=3.0,
        available_clock_domain_id="control-monotonic",
        admission_sequence=3,
    )
    duplicate_transition = apply_station_intent(manifest, state, duplicate)
    assert not duplicate_transition.receipt.accepted
    assert (
        duplicate_transition.receipt.status
        == "duplicate_or_out_of_order_client_sequence"
    )


def test_human_takeover_is_one_bound_multi_source_operation() -> None:
    manifest = _human_takeover_manifest()
    state = _human_takeover_ready_state(manifest)

    pending_state, request = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_human_takeover",
        subject_id="human-primary",
        resource_binding_id="replay-primary",
        operation_spec_id="human-takeover-spec:sha256:one",
    )

    assert request.receipt.accepted
    assert request.receipt.status == "operation_requested"
    assert pending_state.source_state_by_slot == state.source_state_by_slot
    pending = pending_state.pending_operation_by_subject[
        "handoff:replay-primary->human-primary"
    ]
    assert pending.subject_id == "human-primary"
    assert pending.prior_runtime_state == "candidate"
    assert pending.resource_binding_id == "replay-primary"
    assert pending.resource_prior_runtime_state == "active"
    assert pending.operation_spec_id == "human-takeover-spec:sha256:one"

    completion = _human_takeover_completion(
        manifest,
        pending_state,
        request,
        succeeded=True,
        status=HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1,
    )
    assert completion.accepted
    assert completion.state_after.lifecycle_state == "running"
    assert completion.state_after.source_state_by_slot == {
        "replay-primary": "preflighted",
        "human-primary": "active",
    }
    assert completion.state_after.pending_operation_by_subject == {}


def test_second_active_source_cannot_bypass_explicit_handoff() -> None:
    manifest = _human_takeover_manifest()
    state = _human_takeover_ready_state(manifest)
    _, transition = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_motion_authority",
        subject_id="human-primary",
    )
    assert not transition.receipt.accepted
    assert transition.receipt.status == "active_source_requires_explicit_handoff"


@pytest.mark.parametrize(
    ("subject_id", "resource_binding_id", "operation_spec_id"),
    (
        (None, "replay-primary", "human-takeover-spec:sha256:one"),
        ("human-primary", None, "human-takeover-spec:sha256:one"),
        ("human-primary", "replay-primary", None),
    ),
)
def test_human_takeover_rejects_incomplete_operation_binding(
    subject_id: str | None,
    resource_binding_id: str | None,
    operation_spec_id: str | None,
) -> None:
    manifest = _human_takeover_manifest()
    state = _human_takeover_ready_state(manifest)
    _, transition = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_human_takeover",
        subject_id=subject_id,
        resource_binding_id=resource_binding_id,
        operation_spec_id=operation_spec_id,
    )
    assert not transition.receipt.accepted
    assert transition.receipt.status == "human_takeover_binding_incomplete"


def test_human_takeover_ready_requires_terminal_status_and_aggregate_evidence() -> None:
    manifest = _human_takeover_manifest()
    state = _human_takeover_ready_state(manifest)
    pending_state, request = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_human_takeover",
        subject_id="human-primary",
        resource_binding_id="replay-primary",
        operation_spec_id="human-takeover-spec:sha256:one",
    )

    missing_evidence = _human_takeover_completion(
        manifest,
        pending_state,
        request,
        succeeded=True,
        status=HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1,
        evidence_refs=(),
    )
    assert not missing_evidence.accepted
    assert missing_evidence.status == "operation_result_missing_control_evidence"

    wrong_terminal_status = _human_takeover_completion(
        manifest,
        pending_state,
        request,
        succeeded=True,
        status="completed",
    )
    assert not wrong_terminal_status.accepted
    assert wrong_terminal_status.status == "human_takeover_result_not_ready"


@pytest.mark.parametrize(
    ("status", "lifecycle_state"),
    (
        (HUMAN_TAKEOVER_OPERATION_HELD_STATUS_V1, "hold"),
        (HUMAN_TAKEOVER_OPERATION_FAULTED_STATUS_V1, "fault"),
    ),
)
def test_failed_human_takeover_is_fail_closed(
    status: str,
    lifecycle_state: str,
) -> None:
    manifest = _human_takeover_manifest()
    state = _human_takeover_ready_state(manifest)
    pending_state, request = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_human_takeover",
        subject_id="human-primary",
        resource_binding_id="replay-primary",
        operation_spec_id="human-takeover-spec:sha256:one",
    )
    completion = _human_takeover_completion(
        manifest,
        pending_state,
        request,
        succeeded=False,
        status=status,
    )
    assert completion.accepted
    assert completion.state_after.lifecycle_state == lifecycle_state
    assert completion.state_after.source_state_by_slot == {
        "replay-primary": "preflighted",
        "human-primary": "candidate",
    }


def test_hold_cancels_pending_human_takeover_and_late_completion_cannot_revive() -> None:
    manifest = _human_takeover_manifest()
    state = _human_takeover_ready_state(manifest)
    pending_state, request = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_human_takeover",
        subject_id="human-primary",
        resource_binding_id="replay-primary",
        operation_spec_id="human-takeover-spec:sha256:one",
    )

    _, conflicting = _apply(
        manifest,
        pending_state,
        sequence=2,
        kind="release_candidate",
        subject_id="human-primary",
    )
    assert not conflicting.receipt.accepted
    assert conflicting.receipt.status == "human_takeover_operation_pending"

    held_state, hold = _apply(
        manifest,
        pending_state,
        sequence=2,
        kind="hold",
    )
    assert hold.receipt.accepted
    assert held_state.lifecycle_state == "hold"
    assert held_state.pending_operation_by_subject == {}

    late = _human_takeover_completion(
        manifest,
        held_state,
        request,
        succeeded=True,
        status=HUMAN_TAKEOVER_OPERATION_READY_STATUS_V1,
    )
    assert not late.accepted
    assert late.status == "human_takeover_result_after_latched_safety_state"
    assert late.state_after == held_state


def test_safety_intent_bypasses_normal_control_lease_but_grants_no_motion() -> None:
    manifest = _manifest()
    state = initial_control_station_state(manifest, runtime_session_id="session:station")
    state, transition = _apply(
        manifest,
        state,
        sequence=0,
        kind="hold",
        client_id="safety-console",
    )
    assert transition.receipt.accepted
    assert state.lifecycle_state == "hold"

    request = _intent(
        manifest,
        state,
        sequence=1,
        kind="request_motion_authority",
        client_id="safety-console",
    )
    rejected = apply_station_intent(manifest, state, request)
    assert not rejected.receipt.accepted
    assert rejected.receipt.status == "runtime_hold_or_fault_latched"


def test_composition_session_contract_is_single_direction_and_non_resumable() -> None:
    spec = CompositionSessionOperationSpec(
        session_key="episode:composition:0",
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("replay", "human"),
        composition_spec_hash="composition:weighted-se3",
        authority_gate_spec_hash="composition-gate:formal",
        execution_binding_hash="target-binding:sim",
        capture_boundary_id="capture-boundary:0",
    )
    state = initial_composition_session_state(spec, session_epoch=0)
    active, started = advance_composition_session(
        state,
        transition="start",
        reason="station_ready",
        source_fact_id="station-result:start",
        source_fact_hash="hash:start",
        start_intent_id="station-intent:start",
    )
    assert active.phase == "active"
    held, paused = advance_composition_session(
        active,
        transition="hold",
        reason="operator_pause",
        source_fact_id="station-intent:hold",
        source_fact_hash="hash:hold",
    )
    assert held.phase == "held"
    assert paused.operation_spec_id == spec.operation_spec_id
    with pytest.raises(ValueError, match="Unsupported composition session"):
        advance_composition_session(
            held,
            transition="start",
            reason="fresh_input_is_not_rearm",
            source_fact_id="input:fresh",
            source_fact_hash="hash:fresh",
            start_intent_id="station-intent:start-2",
        )
    ended, stopped = advance_composition_session(
        held,
        transition="stop",
        reason="episode_end",
        source_fact_id="station-result:stop",
        source_fact_hash="hash:stop",
    )
    assert ended.phase == "ended"
    assert stopped.transition == "stop"


def test_composition_session_rearm_spec_and_receipt_have_typed_trace_schema() -> None:
    logical = CompositionSessionOperationSpec(
        session_key="episode:composition:trace-rearm",
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("replay", "human"),
        composition_spec_hash="composition:weighted-se3",
        authority_gate_spec_hash="composition-gate:formal",
        execution_binding_hash="target-binding:sim",
        capture_boundary_id="capture-boundary:trace-rearm",
    )
    held = initial_composition_session_state(logical, session_epoch=3)
    active, _ = advance_composition_session(
        held,
        transition="start",
        reason="station_ready",
        source_fact_id="station-operation:start",
        source_fact_hash="hash:start",
        start_intent_id="station-intent:start",
    )
    held, _ = advance_composition_session(
        active,
        transition="hold",
        reason="operator_pause",
        source_fact_id="station-intent:hold",
        source_fact_hash="hash:hold",
    )
    rearm = CompositionSessionRearmSpec(
        rearm_key="trace-rearm:4",
        logical_operation_spec_id=logical.operation_spec_id,
        logical_operation_spec_hash=logical.semantic_content_hash,
        prior_session_epoch=held.session_epoch,
        prior_state_hash=held.semantic_content_hash,
        execution_binding_hash=logical.execution_binding_hash,
        readiness_observer_binding_id="sim-readiness:v1",
        readiness_timeout_sec=0.1,
    )
    receipt = CompositionSessionRearmReceipt(
        rearm_spec_id=rearm.rearm_spec_id,
        rearm_spec_hash=rearm.semantic_content_hash,
        logical_operation_spec_id=logical.operation_spec_id,
        logical_operation_spec_hash=logical.semantic_content_hash,
        prior_session_epoch=3,
        next_session_epoch=4,
        prior_composer_producer_epoch=7,
        composer_producer_epoch=8,
        prior_authority_epoch=10,
        authority_epoch=11,
        prior_execution_epoch=20,
        execution_epoch=21,
        cancellation_receipt_id="cancel:21",
        measurement_id="measurement:4",
        measurement_hash="measurement-hash:4",
        producer_activation_id="producer-activation:8",
        authority_event_ids=("authority-event:11",),
        ready_gate_event_id="gate-ready:4",
        activation_input_barrier_time=8.0,
        completed_at=8.01,
        clock_domain_id="sim-monotonic",
        satisfied_readiness_checks=rearm.required_readiness_checks,
        status=COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1,
    )
    spec_fact = project_composition_session_rearm_spec(
        rearm,
        runtime_session_id="runtime:trace-rearm",
        control_domain_id="domain:arms",
        embodiment_instance_id="embodiment:sim",
        occurred_at=8.0,
        clock_domain_id="sim-monotonic",
    )
    receipt_fact = project_composition_session_rearm_receipt(
        receipt,
        runtime_session_id="runtime:trace-rearm",
        control_domain_id="domain:arms",
        embodiment_instance_id="embodiment:sim",
    )
    assert spec_fact.event_type == "composition_session_rearm_spec"
    assert spec_fact.payload["prior_state_hash"] == held.semantic_content_hash
    assert receipt_fact.event_type == "composition_session_rearm_receipt"
    assert receipt_fact.payload["next_session_epoch"] == 4
    assert receipt_fact.payload["status"] == (
        COMPOSITION_SESSION_OPERATION_REARMED_STATUS_V1
    )


def test_composition_session_controller_owns_generic_identity_barrier_and_hold() -> None:
    clock = lambda: 7.0
    controller = CompositionSessionController(
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("replay", "human"),
        composition_spec_hash="composition:weighted-se3",
        authority_gate_spec_hash="composition-gate:formal",
        execution_binding_hash="target-binding:sim-or-real",
        clock=clock,
        clock_domain_id="controller-monotonic",
    )
    spec = CompositionSessionOperationSpec(
        session_key="episode:composition:controller",
        composer_source_slot_id="composer",
        contributor_source_slot_ids=("human", "replay"),
        composition_spec_hash="composition:weighted-se3",
        authority_gate_spec_hash="composition-gate:formal",
        execution_binding_hash="target-binding:sim-or-real",
        capture_boundary_id="capture-boundary:controller",
    )
    observed_specs = []
    observed_transitions = []
    observed_barriers = []
    controller.bind_session_trace_observers(
        spec_observer=observed_specs.append,
        transition_observer=lambda event, **metadata: (
            observed_transitions.append((event, metadata))
        ),
    )
    controller.bind_publication_barrier_observer(
        lambda snapshot, **metadata: (
            observed_barriers.append((snapshot, metadata))
        )
    )
    assert controller.register(spec) == spec.operation_spec_id
    assert controller.register(spec) == spec.operation_spec_id
    assert observed_specs == [spec]
    assert controller.current_state is not None
    assert controller.current_state.session_epoch == 0

    start_result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id="intent:start",
            operation_kind="start_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=6.9,
            completed_clock_domain_id="adapter-monotonic",
            status=COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
            evidence_refs=(spec.operation_spec_id,),
        ),
        runtime_session_id="runtime:composition",
        control_domain_id="control:composition",
        available_at=7.0,
        available_clock_domain_id="controller-monotonic",
        admission_sequence=0,
    )
    controller.complete_start_operation(
        operation_spec_id=spec.operation_spec_id,
        result=start_result,
        completion_accepted=True,
        source_fact_id="station-operation:start",
        source_fact_hash="station-operation-hash:start",
        start_intent_id="intent:start",
    )
    assert controller.current_state is not None
    assert controller.current_state.phase == "active"
    assert controller.publication_barrier.snapshot().enabled
    with pytest.raises(RuntimeError, match="single-use"):
        controller.require_registered_spec(spec.operation_spec_id)

    lower_transitions = []

    def lower_hold() -> None:
        assert not controller.publication_barrier.snapshot().enabled
        lower_transitions.append("hold")

    controller.prepare_safety_transition(
        intent_kind="hold",
        causal_station_fact_id="intent:hold",
        source_fact_id="station-intent:hold",
        source_fact_hash="station-intent-hash:hold",
        occurred_at=7.1,
        clock_domain_id="controller-monotonic",
        lower_hold_or_fault=lower_hold,
    )
    assert lower_transitions == ["hold"]
    assert controller.current_state.phase == "held"
    assert not controller.publication_barrier.snapshot().enabled
    assert [item[0].transition for item in observed_transitions] == [
        "start",
        "hold",
    ]
    assert [item[0].enabled for item in observed_barriers] == [True, False]


def test_station_composition_session_has_one_composer_authority_owner() -> None:
    slots = (
        _source_slot(
            "human",
            source_kind="human",
            producer_id="human:one",
            supported_uses=("contributor",),
        ),
        _source_slot(
            "replay",
            source_kind="replay",
            producer_id="replay:one",
            supported_uses=("contributor",),
        ),
        _source_slot(
            "composer",
            source_kind="composite",
            producer_id="composer:one",
            supported_uses=("candidate", "authority"),
        ),
    )
    base = _manifest(target_kind="sim", slots=slots)
    manifest = replace(
        base,
        allowed_intent_kinds=tuple(
            sorted(
                {
                    *base.allowed_intent_kinds,
                    "request_candidate",
                    "release_candidate",
                    "start_composition_session",
                    "stop_composition_session",
                }
            )
        ),
    )
    initial = initial_control_station_state(
        manifest,
        runtime_session_id="session:composition",
    )
    leased, lease_transition = _apply(
        manifest,
        initial,
        sequence=0,
        kind="acquire_control_lease",
    )
    assert lease_transition.receipt.accepted
    armed = replace(
        leased,
        lifecycle_state="armed",
        target_state="preflighted",
        source_state_by_slot={
            "human": "preflighted",
            "replay": "preflighted",
            "composer": "candidate",
        },
    )
    pending, request = _apply(
        manifest,
        armed,
        sequence=1,
        kind="start_composition_session",
        subject_id="composer",
        resource_binding_id=manifest.execution_binding.binding_id,
        operation_spec_id="composition-session-spec:one",
    )
    assert request.receipt.status == "operation_requested"
    assert pending.source_state_by_slot["composer"] == "candidate"
    result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id=request.intent.intent_id,
            operation_kind="start_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=3.0,
            completed_clock_domain_id="adapter-monotonic",
            status=COMPOSITION_SESSION_OPERATION_READY_STATUS_V1,
            evidence_refs=("composition-session-spec:one",),
        ),
        runtime_session_id=pending.runtime_session_id,
        control_domain_id=pending.control_domain_id,
        available_at=3.1,
        available_clock_domain_id="control-monotonic",
        admission_sequence=20,
    )
    started = apply_station_operation_result(manifest, pending, result)
    assert started.accepted
    assert started.state_after.lifecycle_state == "running"
    assert started.state_after.source_state_by_slot == {
        "human": "preflighted",
        "replay": "preflighted",
        "composer": "active",
    }
    stop_pending, stop_request = _apply(
        manifest,
        started.state_after,
        sequence=2,
        kind="stop_composition_session",
        subject_id="composer",
        resource_binding_id=manifest.execution_binding.binding_id,
        operation_spec_id="composition-session-spec:one",
    )
    stop_result = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id=stop_request.intent.intent_id,
            operation_kind="stop_composition_session",
            subject_id="composer",
            succeeded=True,
            completed_at=4.0,
            completed_clock_domain_id="adapter-monotonic",
            status=COMPOSITION_SESSION_OPERATION_STOPPED_STATUS_V1,
            evidence_refs=("execution-hold:one",),
        ),
        runtime_session_id=stop_pending.runtime_session_id,
        control_domain_id=stop_pending.control_domain_id,
        available_at=4.1,
        available_clock_domain_id="control-monotonic",
        admission_sequence=21,
    )
    stopped = apply_station_operation_result(
        manifest,
        stop_pending,
        stop_result,
    )
    assert stopped.accepted
    assert stopped.state_after.lifecycle_state == "hold"
    assert stopped.state_after.source_state_by_slot["composer"] == "candidate"


def test_sim_candidate_and_authority_require_typed_operation_completion() -> None:
    human = _source_slot(
        "human-primary",
        source_kind="human",
        producer_id="human:primary",
        supported_uses=("candidate", "authority"),
    )
    base = _manifest(target_kind="sim", slots=(human,))
    manifest = ControlStationRunManifest(
        manifest_id="control-station:sim-human-authority-v1",
        manifest_version="1",
        execution_binding=base.execution_binding,
        source_slots=base.source_slots,
        authority_policy_hash=base.authority_policy_hash,
        decision_schedule_hash=base.decision_schedule_hash,
        trace_requirement_profile_id=base.trace_requirement_profile_id,
        capture_policy_id=base.capture_policy_id,
        allowed_intent_kinds=tuple(
            sorted(
                set(base.allowed_intent_kinds)
                | {
                    "request_candidate",
                    "release_candidate",
                    "release_motion_authority",
                }
            )
        ),
        operator_control_lease_sec=base.operator_control_lease_sec,
        explicit_exclusions=base.explicit_exclusions,
    )
    state = initial_control_station_state(manifest, runtime_session_id="session:sim")
    operations = (
        ("acquire_control_lease", None),
        ("connect_target", manifest.execution_binding.binding_id),
        ("connect_source", human.source_slot_id),
        ("run_preflight", manifest.execution_binding.binding_id),
        ("run_preflight", human.source_slot_id),
    )
    for sequence, (kind, subject) in enumerate(operations):
        state, transition = _apply(
            manifest,
            state,
            sequence=sequence,
            kind=kind,
            subject_id=subject,
        )
        if transition.receipt.status == "operation_requested":
            state, _ = _complete(manifest, state, transition)
    assert state.lifecycle_state == "preflighted"

    state, candidate_request = _apply(
        manifest,
        state,
        sequence=5,
        kind="request_candidate",
        subject_id=human.source_slot_id,
    )
    assert candidate_request.receipt.status == "operation_requested"
    assert state.source_state_by_slot[human.source_slot_id] == "preflighted"
    missing_evidence = admit_station_operation_result(
        StationOperationResultDraft(
            source_intent_id=candidate_request.intent.intent_id,
            operation_kind="request_candidate",
            subject_id=human.source_slot_id,
            succeeded=True,
            completed_at=6.1,
            completed_clock_domain_id="adapter-monotonic",
            status="candidate_bound",
        ),
        runtime_session_id=state.runtime_session_id,
        control_domain_id=state.control_domain_id,
        available_at=6.2,
        available_clock_domain_id="control-monotonic",
        admission_sequence=100,
    )
    rejected = apply_station_operation_result(manifest, state, missing_evidence)
    assert not rejected.accepted
    assert rejected.status == "operation_result_missing_control_evidence"
    assert rejected.state_after is state

    state, _ = _complete(manifest, state, candidate_request)
    assert state.source_state_by_slot[human.source_slot_id] == "candidate"
    assert state.lifecycle_state == "armed"

    state, authority_request = _apply(
        manifest,
        state,
        sequence=6,
        kind="request_motion_authority",
        subject_id=human.source_slot_id,
    )
    assert authority_request.receipt.status == "operation_requested"
    assert state.source_state_by_slot[human.source_slot_id] == "candidate"
    state, _ = _complete(manifest, state, authority_request)
    assert state.source_state_by_slot[human.source_slot_id] == "active"
    assert state.lifecycle_state == "running"

    state, release_authority = _apply(
        manifest,
        state,
        sequence=7,
        kind="release_motion_authority",
        subject_id=human.source_slot_id,
    )
    failed_release_state, failed_release = _complete(
        manifest,
        state,
        release_authority,
        succeeded=False,
    )
    assert failed_release.accepted
    assert failed_release_state.source_state_by_slot[human.source_slot_id] == "active"
    assert failed_release_state.lifecycle_state == "fault"

    state, _ = _complete(manifest, state, release_authority)
    assert state.source_state_by_slot[human.source_slot_id] == "candidate"
    assert state.lifecycle_state == "hold"

    state, release_candidate = _apply(
        manifest,
        state,
        sequence=8,
        kind="release_candidate",
        subject_id=human.source_slot_id,
    )
    state, _ = _complete(manifest, state, release_candidate)
    assert state.source_state_by_slot[human.source_slot_id] == "preflighted"
    assert state.lifecycle_state == "hold"


def test_same_manifest_and_intent_sequence_is_deterministic() -> None:
    manifest = _manifest()

    def run_once():
        state = initial_control_station_state(manifest, runtime_session_id="session:station")
        hashes = [state.semantic_content_hash]
        operations = (
            ("acquire_control_lease", None),
            ("connect_target", manifest.execution_binding.binding_id),
            ("connect_source", "policy-primary"),
            ("run_preflight", manifest.execution_binding.binding_id),
            ("run_preflight", "policy-primary"),
            ("start_shadow", "policy-primary"),
        )
        for sequence, (kind, subject_id) in enumerate(operations):
            state, transition = _apply(
                manifest,
                state,
                sequence=sequence,
                kind=kind,
                subject_id=subject_id,
            )
            hashes.append(transition.semantic_content_hash)
            if transition.receipt.status == "operation_requested":
                state, completion = _complete(manifest, state, transition)
                hashes.append(completion.semantic_content_hash)
        return tuple(hashes)

    assert run_once() == run_once()


def test_failed_adapter_operation_restores_prior_state_without_false_readiness() -> None:
    manifest = _manifest()
    state = initial_control_station_state(manifest, runtime_session_id="session:station")
    state, _ = _apply(
        manifest,
        state,
        sequence=0,
        kind="acquire_control_lease",
    )
    state, request = _apply(
        manifest,
        state,
        sequence=1,
        kind="connect_target",
        subject_id=manifest.execution_binding.binding_id,
    )
    assert state.target_state == "connect_requested"
    state, completion = _complete(manifest, state, request, succeeded=False)
    assert completion.accepted
    assert completion.status == "operation_failed:adapter_failed"
    assert state.target_state == "configured"
    assert state.lifecycle_state == "configured"
    assert not state.pending_operation_by_subject


def test_hold_is_latched_and_cannot_be_cleared_by_connect_or_preflight() -> None:
    manifest = _manifest()
    state = initial_control_station_state(manifest, runtime_session_id="session:station")
    state, _ = _apply(
        manifest,
        state,
        sequence=0,
        kind="acquire_control_lease",
    )
    state, _ = _apply(manifest, state, sequence=1, kind="hold")
    _, rejected = _apply(
        manifest,
        state,
        sequence=2,
        kind="connect_target",
        subject_id=manifest.execution_binding.binding_id,
    )
    assert not rejected.receipt.accepted
    assert rejected.receipt.status == "runtime_hold_or_fault_latched"
    assert rejected.state_after.lifecycle_state == "hold"


def test_control_station_core_modules_do_not_import_backend_or_sim_packages() -> None:
    package_root = Path(__file__).parents[1] / "control_station"
    # Concrete application profiles are leaves: they bind one selected lower
    # runtime and are intentionally allowed to import that runtime lazily.
    # The stable Station core remains backend-neutral; migration debt in
    # ``composition/`` is checked separately and must not spread there.
    concrete_profile_modules = {
        "real_teleop_entrypoint.py",
        "rm65_calibration_debug_application.py",
        "sim_hil.py",
    }
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(package_root.glob("*.py"))
        if path.name not in concrete_profile_modules
    )
    assert "from hardware" not in source
    assert "import hardware" not in source
    assert "from sim" not in source
    assert "import sim" not in source
    assert "teleop_runtime.deploy.run_policy" not in source


def test_reusable_source_and_target_contracts_do_not_bind_a_concrete_backend() -> None:
    package_root = Path(__file__).parents[1] / "control_station"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for directory in ("sources", "targets")
        for path in sorted((package_root / directory).rglob("*.py"))
    )
    assert "from sim" not in source
    assert "import sim" not in source
    assert "from hardware" not in source
    assert "import hardware" not in source


def test_composition_modules_do_not_import_private_sibling_symbols() -> None:
    composition_root = Path(__file__).parents[1] / "control_station" / "composition"
    violations: list[str] = []
    for path in sorted(composition_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 1 or node.module is None:
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    violations.append(f"{path.name}:{node.lineno}:{alias.name}")
    assert violations == []


def test_authority_store_is_the_canonical_manifest_epoch_source() -> None:
    runtime = SimpleNamespace(
        config=SimpleNamespace(
            runtime_session_id="session-a",
            control_domain_id="arms-a",
        )
    )
    assert authoritative_authority_epoch_source_id(runtime) == (
        "authority-store:session-a:arms-a"
    )


def test_root_control_station_is_explicitly_packaged_without_legacy_alias() -> None:
    repository_root = Path(__file__).parents[1]
    pyproject = (repository_root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"control_station*"' in pyproject
    assert '"web/static/*.html"' in pyproject
    assert '"refinement_annotation/static/*.html"' in pyproject
    assert '"refinement_annotation/static/*.js"' in pyproject
    assert '"refinement_annotation/static/*.css"' in pyproject
    assert not (repository_root / "teleop_runtime" / "control_station" / "__init__.py").exists()


class _ManualClock:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def test_cs0b_runtime_keeps_request_pending_until_adapter_result() -> None:
    manifest = _manifest()
    receiver_clock = _ManualClock(10.0)
    adapter_clock = _ManualClock(20.0)
    runtime = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id="session:station-runtime",
        clock=receiver_clock,
    )

    def submit(sequence: int, kind: str, subject_id: str | None = None):
        return runtime.submit_intent(
            StationIntentDraft(
                client_id="operator:a",
                client_sequence=sequence,
                intent_kind=kind,
                # Numeric values from another clock domain are intentionally
                # not compared to the receiver monotonic clock.
                produced_at=1000.0 + sequence,
                produced_clock_domain_id="browser-monotonic",
                expected_state_revision=runtime.state.state_revision,
                manifest_hash=manifest.semantic_content_hash,
                subject_id=subject_id,
            )
        )

    lease = submit(0, "acquire_control_lease")
    assert lease.receipt.accepted
    request = submit(1, "connect_target", manifest.execution_binding.binding_id)
    assert request.receipt.status == "operation_requested"
    assert runtime.state.target_state == "connect_requested"
    assert runtime.state.lifecycle_state == "configured"

    adapter = NullExecutionTargetAdapter(
        adapter_id="null-target-adapter",
        target_binding_id=manifest.execution_binding.binding_id,
        clock=adapter_clock,
    )
    execution = runtime.execute_pending_operation(request, adapter)
    assert execution.completion_transition.accepted
    assert runtime.state.target_state == "connected"
    assert adapter.perform_count == 1
    assert runtime.next_admission_sequence == 3


def test_cs0b_fake_source_failure_cannot_create_false_shadow_state() -> None:
    manifest = _manifest()
    receiver_clock = _ManualClock(10.0)
    adapter_clock = _ManualClock(10.0)
    runtime = ControlStationRuntime(
        manifest=manifest,
        runtime_session_id="session:station-runtime",
        clock=receiver_clock,
    )

    runtime.submit_intent(
        StationIntentDraft(
            client_id="operator:a",
            client_sequence=0,
            intent_kind="acquire_control_lease",
            produced_at=10.0,
            produced_clock_domain_id="control-monotonic",
            expected_state_revision=0,
            manifest_hash=manifest.semantic_content_hash,
        )
    )
    request = runtime.submit_intent(
        StationIntentDraft(
            client_id="operator:a",
            client_sequence=1,
            intent_kind="connect_source",
            produced_at=10.0,
            produced_clock_domain_id="control-monotonic",
            expected_state_revision=runtime.state.state_revision,
            manifest_hash=manifest.semantic_content_hash,
            subject_id="policy-primary",
        )
    )
    adapter = FakeSourceSlotAdapter(
        adapter_id="fake-policy-source",
        source_slot_id="policy-primary",
        clock=adapter_clock,
        fail_operations=("connect_source",),
    )
    execution = runtime.execute_pending_operation(request, adapter)
    assert execution.completion_transition.accepted
    assert execution.completion_transition.status == "operation_failed:configured_failure"
    assert runtime.state.source_state_by_slot["policy-primary"] == "disconnected"
    assert runtime.state.lifecycle_state == "configured"


def test_cs0b_null_and_fake_adapters_expose_no_action_or_dispatch_api() -> None:
    clock = _ManualClock(0.0)
    target = NullExecutionTargetAdapter(
        adapter_id="null-target",
        target_binding_id="target:null",
        clock=clock,
    )
    source = FakeSourceSlotAdapter(
        adapter_id="fake-source",
        source_slot_id="policy-primary",
        clock=clock,
    )
    forbidden = {
        "send",
        "send_command",
        "dispatch",
        "predict",
        "prepare_next_action",
        "pop_action",
        "begin_episode",
        "finalize_episode",
    }
    assert not any(hasattr(target, name) for name in forbidden)
    assert not any(hasattr(source, name) for name in forbidden)


def test_residual_rebase_intent_requires_one_active_composer() -> None:
    composer = _source_slot(
        "composer-primary",
        source_kind="composite",
        producer_id="composer:primary",
        supported_uses=("candidate", "authority"),
    )
    base = _manifest(target_kind="sim", slots=(composer,))
    manifest = replace(
        base,
        manifest_id="control-station:residual-rebase-development",
        allowed_intent_kinds=tuple(
            sorted(
                set(base.allowed_intent_kinds)
                | {"request_residual_rebase_and_clear"}
            )
        ),
    )
    state = initial_control_station_state(
        manifest,
        runtime_session_id="session:residual-rebase",
    )
    state, lease_transition = _apply(
        manifest,
        state,
        sequence=0,
        kind="acquire_control_lease",
    )
    assert lease_transition.receipt.accepted
    state = replace(
        state,
        lifecycle_state="running",
        target_state="preflighted",
        source_state_by_slot={"composer-primary": "active"},
    )

    next_state, transition = _apply(
        manifest,
        state,
        sequence=1,
        kind="request_residual_rebase_and_clear",
        subject_id="composer-primary",
        resource_binding_id="active-residual-anchor:0",
        operation_spec_id="residual-rebase-profile:1",
    )

    assert transition.receipt.accepted
    assert transition.receipt.status == "residual_rebase_intent_accepted"
    assert transition.receipt.intent_hash == transition.intent.semantic_content_hash
    assert next_state.pending_operation_by_subject == {}
    inactive = replace(
        next_state,
        lifecycle_state="armed",
        source_state_by_slot={"composer-primary": "candidate"},
    )
    _unchanged, rejected = _apply(
        manifest,
        inactive,
        sequence=2,
        kind="request_residual_rebase_and_clear",
        subject_id="composer-primary",
        resource_binding_id="active-residual-anchor:0",
        operation_spec_id="residual-rebase-profile:1",
    )
    assert not rejected.receipt.accepted
    assert rejected.receipt.status == "residual_rebase_requires_active_composer"
    assert rejected.receipt.intent_hash == rejected.intent.semantic_content_hash

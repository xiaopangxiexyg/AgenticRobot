from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from control_station.agent_gateway.task_schemas import (
    CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1,
)
from control_station.sim_hil_acceptance import (
    StationControlTraceEpisodeSessionCapture,
    StationControlTraceFinalizationReceipt,
)
from control_station.sim_hil_profile.execution_evidence import (
    CubeLiftExecutionEvidenceCollector,
)
from control_station.sim_hil_profile.terminal_evaluator import (
    CubeLiftLowerHoldProof,
)


_AUTHORITY_EVENTS = (
    "proposal_snapshot",
    "decision_schedule_tick",
    "authority_state_snapshot",
    "execution_gate_snapshot",
    "arbitration_input_snapshot",
    "arbitration_computation",
    "arbitration_publication_attempt",
    "arbitration_decision",
    "execution_directive",
)
_TARGET_EVENTS = (
    "action_adaptation_request",
    "target_capability_assessment",
    "adapted_action_candidate",
    "action_adaptation_receipt",
    "target_execution_request",
    "target_execution_submission",
    "target_execution_completion",
    "component_execution_target",
    "component_execution_target_admission",
)
_DISPATCH_EVENTS = ("command", "dispatch_owner", "dispatch_receipt")


def _fact(event_type: str, suffix: str) -> SimpleNamespace:
    return SimpleNamespace(
        event_type=event_type,
        fact_id=(
            f"dispatch-receipt:{suffix}"
            if event_type == "dispatch_receipt"
            else f"{event_type}:{suffix}"
        ),
        semantic_content_hash=f"hash:{event_type}:{suffix}",
    )


def _cycle(index: int = 0):
    suffix = str(index)
    request = SimpleNamespace(
        request_id=f"target-request:{suffix}",
        semantic_content_hash=f"hash:target-request:{suffix}",
        control_domain_id="mujoco-sim-hil",
    )
    submission = SimpleNamespace(
        accepted_for_processing=True,
        target_execution_request_id=request.request_id,
        target_execution_request_hash=request.semantic_content_hash,
        receipt_id=f"target-submission:{suffix}",
        semantic_content_hash=f"hash:target-submission:{suffix}",
    )
    target = SimpleNamespace(
        target_id=f"target:{suffix}",
        semantic_content_hash=f"hash:target:{suffix}",
        execution_fence_sequence=index + 3,
    )
    completion = SimpleNamespace(
        status="resolved",
        target_id=target.target_id,
        target_hash=target.semantic_content_hash,
        receipt_id=f"target-completion:{suffix}",
        semantic_content_hash=f"hash:target-completion:{suffix}",
    )
    admission = SimpleNamespace(
        accepted=True,
        target_id=target.target_id,
        target_hash=target.semantic_content_hash,
        receipt_id=f"target-admission:{suffix}",
        semantic_content_hash=f"hash:target-admission:{suffix}",
    )
    backend_receipts = {
        "right_arm": SimpleNamespace(accepted=True),
        "right_gripper": SimpleNamespace(accepted=True),
    }
    dispatch = SimpleNamespace(
        completion_receipt=completion,
        target=target,
        target_admission=admission,
        dispatch_receipts=backend_receipts,
    )
    proposal = SimpleNamespace(
        proposal_id=f"proposal:{suffix}",
        semantic_content_hash=f"hash:proposal:{suffix}",
    )
    proposal_admission = SimpleNamespace(
        accepted=True,
        proposal_id=proposal.proposal_id,
        semantic_content_hash=f"hash:proposal-admission:{suffix}",
        proposal=proposal,
    )
    disposition = SimpleNamespace(
        action="publish_proposal",
        disposition_id=f"disposition:{suffix}",
        semantic_content_hash=f"hash:disposition:{suffix}",
    )
    input_receipt = SimpleNamespace(
        accepted=True,
        composition_input_id=f"composition-input:{suffix}",
        semantic_content_hash=f"hash:composition-input:{suffix}",
    )
    policy = SimpleNamespace(
        role="policy",
        accepted=True,
        source_emission_id=f"policy-emission:{suffix}",
        semantic_content_hash=f"hash:policy-submission:{suffix}",
        input_receipt=input_receipt,
    )
    composition = SimpleNamespace(
        decision_key=f"decision-key:{suffix}",
        semantic_content_hash=f"hash:composition-step:{suffix}",
        control_cut=SimpleNamespace(
            runtime_session_id="runtime:cube-lift:test",
            control_domain_id="mujoco-sim-hil",
        ),
        disposition=disposition,
        proposal_admission=proposal_admission,
        blocked_reason=None,
        error=None,
        hold_transition=None,
    )
    facts = tuple(
        _fact(event_type, suffix)
        for event_type in (
            *_AUTHORITY_EVENTS,
            "shared_safety_result",
            *_TARGET_EVENTS,
            *_DISPATCH_EVENTS,
        )
    )
    execution = SimpleNamespace(
        status="dispatched",
        cancellation=None,
        cancellation_receipt=None,
        safety_result=SimpleNamespace(status="safe", safe_action=object()),
        target_execution_request=request,
        target_submission_receipt=submission,
        target_completion_receipt=completion,
        target_dispatch_result=dispatch,
        dispatch_validation=SimpleNamespace(accepted=True),
        dispatch_receipts=backend_receipts,
        dispatch_receipt_id=f"dispatch-receipt:{suffix}",
        trace_facts=facts,
    )
    cycle = SimpleNamespace(
        cycle_index=index,
        semantic_content_hash=f"hash:cycle:{suffix}",
        policy_submission=policy,
        composition_step=composition,
        execution=execution,
    )
    association = SimpleNamespace(
        status="available",
        control_domain_id="mujoco-sim-hil",
        execution_fence_sequence=target.execution_fence_sequence,
        association_id=f"measurement-reference-association:{suffix}",
        semantic_content_hash=f"hash:measurement-association:{suffix}",
        receipt_references=(
            SimpleNamespace(
                receipt_id=admission.receipt_id,
                receipt_hash=admission.semantic_content_hash,
            ),
        ),
    )
    snapshot = SimpleNamespace(
        latest_association=association,
        measurement_count=index + 1,
        unmatched_measurement_count=0,
    )
    return cycle, snapshot


def _hold() -> CubeLiftLowerHoldProof:
    return CubeLiftLowerHoldProof(
        runtime_session_id="runtime:cube-lift:test",
        control_domain_id="mujoco-sim-hil",
        hold_receipt_ref="lower-hold:terminal",
        gate_snapshot_hash="gate-snapshot:terminal",
        gate_epoch=4,
        cancellation_receipt_ref="cancellation-receipt:terminal",
        confirmed_at=20.0,
        evidence_refs=("gate-event:terminal", "cancellation:terminal"),
    )


def test_execution_collector_projects_exact_bounded_result() -> None:
    collector = CubeLiftExecutionEvidenceCollector(
        runtime_session_id="runtime:cube-lift:test",
        control_domain_id="mujoco-sim-hil",
    )
    first, first_measurement = _cycle(0)
    second, second_measurement = _cycle(1)
    collector.observe_cycle(first, measurement_snapshot=first_measurement)
    collector.observe_cycle(second, measurement_snapshot=second_measurement)
    cut = collector.seal_for_hold(_hold())
    trace = SimpleNamespace(
        runtime_session_id="runtime:cube-lift:test",
        control_domain_id="mujoco-sim-hil",
        trace_terminal_receipt_ref="trace-terminal:durable",
        trace_commit_receipt_ref="trace-commit:durable",
    )

    result = cut.to_execution_result(
        lower_hold_proof=_hold(),
        trace_finalization=trace,
    )

    assert result["schema_version"] == CUBE_LIFT_EXECUTION_RESULT_SCHEMA_V1
    assert result["cycle_count"] == 2
    assert result["dispatched_cycle_count"] == 2
    assert result["measurement_count"] == 2
    assert result["first_cycle_receipt_ref"] == "dispatch-receipt:0"
    assert result["last_cycle_receipt_ref"] == "dispatch-receipt:1"
    assert result["terminal_hold_receipt_ref"] == "lower-hold:terminal"


def test_execution_collector_rejects_missing_safety_fact() -> None:
    collector = CubeLiftExecutionEvidenceCollector(
        runtime_session_id="runtime:cube-lift:test",
        control_domain_id="mujoco-sim-hil",
    )
    cycle, snapshot = _cycle(0)
    cycle.execution.trace_facts = tuple(
        fact
        for fact in cycle.execution.trace_facts
        if fact.event_type != "shared_safety_result"
    )
    with pytest.raises(ValueError, match="safety trace evidence is incomplete"):
        collector.observe_cycle(cycle, measurement_snapshot=snapshot)


def test_execution_collector_rejects_reused_measurement() -> None:
    collector = CubeLiftExecutionEvidenceCollector(
        runtime_session_id="runtime:cube-lift:test",
        control_domain_id="mujoco-sim-hil",
    )
    first, first_measurement = _cycle(0)
    second, second_measurement = _cycle(1)
    second_measurement.latest_association.semantic_content_hash = (
        first_measurement.latest_association.semantic_content_hash
    )
    second_measurement.latest_association.association_id = (
        first_measurement.latest_association.association_id
    )
    collector.observe_cycle(first, measurement_snapshot=first_measurement)
    with pytest.raises(ValueError, match="association was reused"):
        collector.observe_cycle(second, measurement_snapshot=second_measurement)


def test_trace_finalization_receipt_is_content_addressed() -> None:
    receipt = StationControlTraceFinalizationReceipt(
        trace_session_id="trace-session:test",
        runtime_session_id="runtime:cube-lift:test",
        control_domain_id="mujoco-sim-hil",
        episode_uuid="episode:test",
        terminal_fact_id="runtime-terminal:test",
        terminal_fact_hash="terminal-fact-hash:test",
        terminal_journal_sequence=12,
        terminal_record_integrity_hash="terminal-record-hash:test",
        trace_seal_hash="trace-seal-hash:test",
        trace_commit_hash="trace-commit-hash:test",
        trace_commit_relative_path="control_trace/episode:test/capture_commit.cjson",
    )
    assert receipt.trace_terminal_receipt_ref.startswith(
        "control-trace-terminal-durable:"
    )
    assert receipt.trace_commit_receipt_ref == (
        "control-trace-capture-commit:trace-commit-hash:test"
    )
    assert receipt.to_wire()["semantic_content_hash"] == (
        receipt.semantic_content_hash
    )


def test_trace_session_complete_runtime_joins_only_durable_lower_facts() -> None:
    capture = object.__new__(StationControlTraceEpisodeSessionCapture)
    capture._lock = threading.RLock()
    capture._runtime_finalization = None
    capture._prepared = None
    commit = SimpleNamespace(
        trace_seal_hash="trace-seal-hash:test",
        semantic_content_hash="trace-commit-hash:test",
    )
    publish = SimpleNamespace(
        durable=True,
        semantic_content_hash=commit.semantic_content_hash,
        episode_uuid="episode:test",
        relative_path="control_trace/episode:test/capture_commit.cjson",
    )
    capture._final_report = SimpleNamespace(
        capture_commit_status="committed",
        capture_commit_candidate=commit,
        capture_commit_publish_receipt=publish,
        episode_uuid="episode:test",
    )
    terminal_fact = SimpleNamespace(
        event_type="runtime_trace_terminal",
        fact_id="runtime-terminal:test",
        semantic_content_hash="terminal-fact-hash:test",
    )
    terminal_entry = SimpleNamespace(
        journal_sequence=12,
        fact=terminal_fact,
        episode_membership=None,
    )
    capture.trace_runtime = SimpleNamespace(
        session=SimpleNamespace(
            trace_session_id="trace-session:test",
            runtime_session_id="runtime:cube-lift:test",
            control_domain_id="mujoco-sim-hil",
        ),
        journal_entries=lambda: (terminal_entry,),
        snapshot=lambda: SimpleNamespace(durable_sequences=(12,)),
        records=lambda: (
            SimpleNamespace(
                journal_sequence=12,
                record_integrity_hash="terminal-record-hash:test",
            ),
        ),
    )
    capture.spool_worker = SimpleNamespace(
        flush_once=lambda: SimpleNamespace(failure_code=None)
    )

    receipt = capture.complete_runtime()

    assert receipt.trace_session_id == "trace-session:test"
    assert capture.complete_runtime() is receipt


def test_trace_session_rejects_terminal_inside_episode() -> None:
    capture = object.__new__(StationControlTraceEpisodeSessionCapture)
    capture._lock = threading.RLock()
    capture._runtime_finalization = None
    capture._prepared = None
    commit = SimpleNamespace(
        trace_seal_hash="trace-seal-hash:test",
        semantic_content_hash="trace-commit-hash:test",
    )
    capture._final_report = SimpleNamespace(
        capture_commit_status="committed",
        capture_commit_candidate=commit,
        capture_commit_publish_receipt=SimpleNamespace(
            durable=True,
            semantic_content_hash=commit.semantic_content_hash,
            episode_uuid="episode:test",
            relative_path="control_trace/episode:test/capture_commit.cjson",
        ),
        episode_uuid="episode:test",
    )
    terminal_entry = SimpleNamespace(
        journal_sequence=12,
        fact=SimpleNamespace(
            event_type="runtime_trace_terminal",
            fact_id="runtime-terminal:test",
            semantic_content_hash="terminal-fact-hash:test",
        ),
        episode_membership=SimpleNamespace(episode_uuid="episode:test"),
    )
    capture.trace_runtime = SimpleNamespace(
        journal_entries=lambda: (terminal_entry,),
    )
    capture.spool_worker = SimpleNamespace(
        flush_once=lambda: SimpleNamespace(failure_code=None)
    )

    with pytest.raises(RuntimeError, match="outside the formal episode"):
        capture.complete_runtime()

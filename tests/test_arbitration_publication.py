from __future__ import annotations

from types import SimpleNamespace

from robot_motion.action.arbitration_runtime import (
    ArbitrationPublicationAttempt,
    ArbitrationRuntime,
    AuthoritativeArbitrationDecision,
    DecisionScheduleSpec,
)


def test_only_successful_publication_creates_authoritative_decision() -> None:
    spec = DecisionScheduleSpec(
        spec_id="30hz", decision_hz=30, max_target_lateness_sec=10
    )
    runtime = ArbitrationRuntime(spec)
    computation = SimpleNamespace(
        result_id="result-0",
        output_status="selected",
        gate_epoch=0,
        publication_deadline_at=10.0,
        target_effective_at=1.0,
        arbitration_input_snapshot_id="input-0",
        selected_action_hash="action-0",
        selected_by_group={"arms": "proposal-0"},
        composed_action="immutable-action",
        authority_epoch=0,
        decision_schedule_spec_hash=spec.semantic_content_hash,
    )
    gate = SimpleNamespace(status="ready", gate_epoch=0)

    attempt, decision = runtime.publish(
        computation, attempted_at=1.0, live_gate_snapshot=gate
    )

    assert isinstance(attempt, ArbitrationPublicationAttempt)
    assert isinstance(decision, AuthoritativeArbitrationDecision)
    assert decision.selected_action == "immutable-action"


def test_schedule_mismatch_is_a_publication_rejection_not_a_decision() -> None:
    spec = DecisionScheduleSpec(spec_id="30hz", decision_hz=30)
    runtime = ArbitrationRuntime(spec)
    computation = SimpleNamespace(
        result_id="result-0",
        output_status="selected",
        gate_epoch=0,
        publication_deadline_at=10.0,
        target_effective_at=1.0,
        arbitration_input_snapshot_id="input-0",
        selected_action_hash="action-0",
        selected_by_group={},
        composed_action="action",
        authority_epoch=0,
        decision_schedule_spec_hash="different-schedule",
    )

    attempt, decision = runtime.publish(
        computation,
        attempted_at=1.0,
        live_gate_snapshot=SimpleNamespace(status="ready", gate_epoch=0),
    )

    assert not attempt.accepted
    assert attempt.rejection_reason == "decision_schedule_mismatch"
    assert decision is None

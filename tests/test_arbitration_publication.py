from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_motion.action import (
    BIMANUAL_QPOS_V1,
    RobotAction,
    project_robot_action_v1_to_component_action,
)
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
    action = project_robot_action_v1_to_component_action(
        RobotAction(
            action_space=BIMANUAL_QPOS_V1,
            arm_qpos_by_side={"left": np.zeros(6, dtype=np.float32)},
            source="test",
        )
    )
    computation = SimpleNamespace(
        result_id="result-0",
        output_status="selected",
        gate_epoch=0,
        publication_deadline_at=10.0,
        target_effective_at=1.0,
        supported_until=2.0,
        arbitration_input_snapshot_id="input-0",
        selected_action_hash=action.semantic_content_hash,
        selected_by_group={"arms": "proposal-0"},
        composed_candidate=action,
        authority_epoch=0,
        decision_schedule_spec_hash=spec.semantic_content_hash,
        semantic_content_hash="result-hash-0",
        target_clock_domain_id="control.monotonic",
    )
    gate = SimpleNamespace(status="ready", gate_epoch=0)
    proposal = SimpleNamespace(
        proposal_id="proposal-0",
        producer_kind="model",
        effective_clock_domain_id="control.monotonic",
    )
    input_snapshot = SimpleNamespace(
        candidate_authorization_snapshot=SimpleNamespace(
            required_producer_kinds=(),
            authorization_by_proposal_id={},
            dispositions_by_proposal_id={},
        ),
        proposal_snapshot=SimpleNamespace(
            candidates=(SimpleNamespace(proposal=proposal),),
        ),
    )

    attempt, decision = runtime.publish(
        computation,
        attempted_at=1.0,
        live_gate_snapshot=gate,
        input_snapshot=input_snapshot,
    )

    assert isinstance(attempt, ArbitrationPublicationAttempt)
    assert isinstance(decision, AuthoritativeArbitrationDecision)
    assert decision.selected_action == action
    assert decision.target_clock_domain_id == "control.monotonic"


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
        composed_candidate="action",
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


def test_publication_at_support_boundary_is_rejected() -> None:
    spec = DecisionScheduleSpec(
        spec_id="support-boundary",
        decision_hz=30,
        max_target_lateness_sec=10,
    )
    runtime = ArbitrationRuntime(spec)
    computation = SimpleNamespace(
        result_id="result-support-boundary",
        output_status="selected",
        gate_epoch=0,
        publication_deadline_at=10.0,
        target_effective_at=1.0,
        supported_until=1.2,
        arbitration_input_snapshot_id="input-support-boundary",
        selected_action_hash="action-support-boundary",
        selected_by_group={"arms": "proposal-support-boundary"},
        composed_candidate="unused-because-rejected",
        authority_epoch=0,
        decision_schedule_spec_hash=spec.semantic_content_hash,
        semantic_content_hash="result-support-boundary-hash",
        target_clock_domain_id="control.monotonic",
    )

    attempt, decision = runtime.publish(
        computation,
        attempted_at=1.2,
        live_gate_snapshot=SimpleNamespace(status="ready", gate_epoch=0),
    )

    assert decision is None
    assert attempt.rejection_reason == "support_expired_before_publication"

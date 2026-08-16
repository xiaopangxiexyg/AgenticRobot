from __future__ import annotations

from dataclasses import replace

import pytest

from agent_gateway.protocol.evaluation_contracts import (
    SkillAggregateEvaluationReceipt,
    SkillEvaluationPlan,
    SkillEvaluationReceipt,
    SkillEvaluatorSpec,
    assess_evaluation_receipts,
)


def _spec(
    evaluator_id: str,
    *,
    kind: str,
    role: str,
    evidence_domain_id: str,
    location: str,
) -> SkillEvaluatorSpec:
    return SkillEvaluatorSpec(
        evaluator_id=evaluator_id,
        evaluator_version="1",
        evaluator_kind=kind,
        input_evidence_schema_hash=f"input:{evaluator_id}:v1",
        output_schema_hash=f"output:{evaluator_id}:v1",
        role=role,
        execution_location=location,
        deadline_ms=500,
        failure_policy="mark_error",
        evidence_domain_id=evidence_domain_id,
        configuration_hash=f"config:{evaluator_id}:v1",
    )


def _plan() -> SkillEvaluationPlan:
    return SkillEvaluationPlan(
        evaluation_plan_id="evaluation.cube_lift.mixed",
        evaluation_plan_version="1",
        skill_definition_hash="definition:cube-lift:v1",
        route_hash="route:mujoco:cube-lift:v1",
        deployment_binding_hash="binding:mujoco:cube-lift:v1",
        evaluator_specs=(
            _spec(
                "environment.cube_lift_kernel",
                kind="environment",
                role="required",
                evidence_domain_id="domain:mujoco-task-kernel",
                location="lower_runtime",
            ),
            _spec(
                "external.scene_vlm",
                kind="external",
                role="contributing",
                evidence_domain_id="domain:wrist-rgb-model-a",
                location="external_service",
            ),
            _spec(
                "policy.self_assessment",
                kind="policy_self",
                role="contributing",
                evidence_domain_id="domain:wrist-rgb-model-a",
                location="policy_runtime",
            ),
        ),
        aggregation_policy_id="aggregation.cube_lift.mixed",
        aggregation_policy_hash="aggregation-policy:cube-lift:v1",
        required_evaluation_deadline_ms=1_000,
        missing_evidence_policy="inconclusive",
        disagreement_policy="aggregation_policy",
        task_outcome_schema_hash="task-outcome:cube-lift:v1",
    )


def _receipt(
    plan: SkillEvaluationPlan,
    evaluator_id: str,
    *,
    verdict: str,
    evaluated_at: float = 10.2,
) -> SkillEvaluationReceipt:
    spec = plan.evaluator(evaluator_id, "1")
    return SkillEvaluationReceipt(
        operation_id="operation:cube-lift:1",
        request_hash="request:cube-lift:1",
        skill_id="policy.cube_lift",
        skill_version="1",
        skill_definition_hash=plan.skill_definition_hash,
        route_id="sim.mujoco.rm65_inspire.openpi",
        route_version="1",
        route_hash=plan.route_hash,
        deployment_binding_hash=plan.deployment_binding_hash,
        evaluation_plan_hash=plan.semantic_content_hash,
        evaluator_id=spec.evaluator_id,
        evaluator_version=spec.evaluator_version,
        evaluator_hash=spec.semantic_content_hash,
        evaluator_kind=spec.evaluator_kind,
        evidence_domain_id=spec.evidence_domain_id,
        input_evidence_hash=f"evidence:{evaluator_id}:cut-42",
        evaluated_cut_id="cut:episode:42",
        evaluated_at=evaluated_at,
        verdict=verdict,
        reason_codes=(f"{evaluator_id}:{verdict}",),
        evidence_refs=(f"artifact:{evaluator_id}:42",),
    )


def test_plan_is_bound_to_exact_route_deployment_and_evaluator_configs() -> None:
    plan = _plan()
    assert len(plan.semantic_content_hash) == 64
    assert plan.required_evaluator_ids == ("environment.cube_lift_kernel",)
    assert plan.to_wire()["evaluation_plan_hash"] == plan.semantic_content_hash
    assert (
        replace(plan, deployment_binding_hash="binding:other").semantic_content_hash
        != plan.semantic_content_hash
    )
    with pytest.raises(ValueError, match="deadline cannot exceed"):
        replace(plan, required_evaluation_deadline_ms=100)


def test_external_policy_self_and_environment_receipts_can_be_mixed() -> None:
    plan = _plan()
    receipts = (
        _receipt(plan, "environment.cube_lift_kernel", verdict="succeeded"),
        _receipt(plan, "external.scene_vlm", verdict="failed"),
        _receipt(plan, "policy.self_assessment", verdict="succeeded"),
    )
    assessment = assess_evaluation_receipts(
        plan=plan,
        receipts=receipts,
        operation_id="operation:cube-lift:1",
        request_hash="request:cube-lift:1",
        evaluation_deadline_at=11.0,
    )
    assert assessment.complete_for_required
    assert assessment.has_verdict_disagreement
    assert len(assessment.accepted_receipt_hashes) == 3
    assert assessment.independent_evidence_domain_count == 2
    assert assessment.evidence_domain_evaluator_ids[
        "domain:wrist-rgb-model-a"
    ] == ("external.scene_vlm", "policy.self_assessment")


def test_same_evidence_domain_is_not_counted_as_independent_votes() -> None:
    plan = _plan()
    assessment = assess_evaluation_receipts(
        plan=plan,
        receipts=(
            _receipt(plan, "external.scene_vlm", verdict="succeeded"),
            _receipt(plan, "policy.self_assessment", verdict="succeeded"),
        ),
        operation_id="operation:cube-lift:1",
        request_hash="request:cube-lift:1",
        evaluation_deadline_at=11.0,
    )
    assert assessment.independent_evidence_domain_count == 1
    assert assessment.missing_required_evaluator_ids == (
        "environment.cube_lift_kernel",
    )
    assert not assessment.complete_for_required


def test_late_or_wrong_binding_receipts_are_ignored_fail_closed() -> None:
    plan = _plan()
    late = _receipt(
        plan,
        "environment.cube_lift_kernel",
        verdict="succeeded",
        evaluated_at=11.0,
    )
    wrong_binding = replace(
        _receipt(plan, "external.scene_vlm", verdict="succeeded"),
        deployment_binding_hash="binding:other",
    )
    assessment = assess_evaluation_receipts(
        plan=plan,
        receipts=(late, wrong_binding),
        operation_id="operation:cube-lift:1",
        request_hash="request:cube-lift:1",
        evaluation_deadline_at=11.0,
    )
    assert set(assessment.ignored_receipt_reasons.values()) == {
        "deployment_binding_hash_mismatch",
        "evaluation_receipt_late",
    }
    assert not assessment.complete_for_required


def test_policy_self_receipt_has_no_execution_or_safety_authority_fields() -> None:
    receipt = _receipt(_plan(), "policy.self_assessment", verdict="succeeded")
    assert receipt.evaluator_kind == "policy_self"
    for forbidden in (
        "dispatch_permit",
        "motion_permit",
        "operator_control_lease",
        "safety_receipt",
        "lifecycle_rearm_receipt",
    ):
        assert not hasattr(receipt, forbidden)


def test_score_and_confidence_require_explicit_semantics() -> None:
    receipt = _receipt(_plan(), "external.scene_vlm", verdict="succeeded")
    with pytest.raises(ValueError, match="score requires"):
        replace(receipt, score=0.8)
    with pytest.raises(ValueError, match="confidence requires"):
        replace(receipt, confidence=0.9)
    typed = replace(
        receipt,
        score=0.8,
        score_schema_hash="score:binary-success:v1",
        confidence=0.9,
        confidence_semantics_id="confidence:calibrated-probability:v1",
    )
    assert typed.score == 0.8


def test_aggregate_receipt_retains_consumed_ignored_and_missing_lineage() -> None:
    plan = _plan()
    environment = _receipt(
        plan,
        "environment.cube_lift_kernel",
        verdict="succeeded",
    )
    aggregate = SkillAggregateEvaluationReceipt(
        operation_id=environment.operation_id,
        request_hash=environment.request_hash,
        evaluation_plan_hash=plan.semantic_content_hash,
        aggregation_policy_id=plan.aggregation_policy_id,
        aggregation_policy_hash=plan.aggregation_policy_hash,
        consumed_evaluation_receipt_hashes=(
            environment.semantic_content_hash,
        ),
        ignored_evaluation_receipt_hashes=("receipt:late",),
        missing_evaluator_ids=("external.scene_vlm",),
        verdict="inconclusive",
        reason_codes=("contributing_evaluator_missing",),
        generated_at=10.8,
    )
    assert aggregate.to_wire()["verdict"] == "inconclusive"
    with pytest.raises(ValueError, match="must be disjoint"):
        replace(
            aggregate,
            ignored_evaluation_receipt_hashes=(
                environment.semantic_content_hash,
            ),
        )

from __future__ import annotations

import pytest

from robot_motion.action import (
    SafetyMigrationGateAssessment,
    SafetyMigrationProfileMatrix,
    SafetyMigrationRuleAssessment,
    assess_safety_migration_gate,
)


def _gate(stage: str, status: str) -> SafetyMigrationGateAssessment:
    if status == "ready":
        return SafetyMigrationGateAssessment(
            gate_stage=stage,
            status=status,
            evidence_ids=(f"evidence:{stage}",),
        )
    if status == "not_applicable":
        return SafetyMigrationGateAssessment(
            gate_stage=stage,
            status=status,
            evidence_ids=(f"profile-exclusion:{stage}",),
        )
    return SafetyMigrationGateAssessment(
        gate_stage=stage,
        status=status,
        reasons=(f"missing:{stage}",),
    )


def _rule(
    rule_id: str,
    *,
    m5_8a: str = "ready",
    m5_8b: str = "blocked",
    m6: str = "blocked",
    disposition: str | None = None,
) -> SafetyMigrationRuleAssessment:
    return SafetyMigrationRuleAssessment(
        rule_id=rule_id,
        current_owner="legacy:test",
        target_disposition=disposition,
        disposition_evidence_ids=("evidence:disposition",) if disposition else (),
        gate_assessments=(
            _gate("m6_shared_authority", m6),
            _gate("m5_8a_inventory", m5_8a),
            _gate("m5_8b_legacy_mediation", m5_8b),
        ),
    )


def test_inventory_readiness_does_not_imply_real_mediation_or_m6() -> None:
    matrix = SafetyMigrationProfileMatrix(
        profile_id="policy_direct_pose_v1",
        matrix_version="test-v1",
        rules=(_rule("workspace-bound"),),
    )

    assert assess_safety_migration_gate(
        matrix, gate_stage="m5_8a_inventory"
    ).ready
    mediation = assess_safety_migration_gate(
        matrix, gate_stage="m5_8b_legacy_mediation"
    )
    authority = assess_safety_migration_gate(
        matrix, gate_stage="m6_shared_authority"
    )
    assert not mediation.ready
    assert mediation.blocking_rule_ids == ("workspace-bound",)
    assert not authority.ready


def test_m6_ready_requires_final_disposition_and_evidence() -> None:
    with pytest.raises(ValueError, match="final target disposition"):
        _rule("workspace-bound", m5_8b="ready", m6="ready")

    rule = _rule(
        "workspace-bound",
        m5_8b="ready",
        m6="ready",
        disposition="moved_shared_safety",
    )
    matrix = SafetyMigrationProfileMatrix(
        profile_id="policy_direct_pose_v1",
        matrix_version="test-v1",
        rules=(rule,),
    )
    assert assess_safety_migration_gate(
        matrix, gate_stage="m6_shared_authority"
    ).ready


def test_not_applicable_is_explicit_and_does_not_block_profile_gate() -> None:
    rule = _rule(
        "dmtac",
        m5_8a="not_applicable",
        m5_8b="not_applicable",
        m6="not_applicable",
        disposition="not_applicable",
    )
    matrix = SafetyMigrationProfileMatrix(
        profile_id="arm_only_v1",
        matrix_version="test-v1",
        rules=(rule,),
    )
    decision = assess_safety_migration_gate(
        matrix, gate_stage="m6_shared_authority"
    )
    assert decision.ready
    assert decision.not_applicable_rule_ids == ("dmtac",)


def test_ready_and_not_applicable_require_evidence() -> None:
    with pytest.raises(ValueError, match="requires evidence"):
        SafetyMigrationGateAssessment(
            gate_stage="m5_8a_inventory",
            status="ready",
        )
    with pytest.raises(ValueError, match="exclusion evidence"):
        SafetyMigrationGateAssessment(
            gate_stage="m5_8a_inventory",
            status="not_applicable",
        )


def test_matrix_hash_is_independent_of_rule_and_gate_input_order() -> None:
    rule_a = _rule("a")
    rule_b = _rule("b", m5_8a="not_applicable")
    first = SafetyMigrationProfileMatrix(
        profile_id="test-profile",
        matrix_version="test-v1",
        rules=(rule_b, rule_a),
    )
    second = SafetyMigrationProfileMatrix(
        profile_id="test-profile",
        matrix_version="test-v1",
        rules=(rule_a, rule_b),
    )
    assert first.semantic_content_hash == second.semantic_content_hash


def test_rule_requires_exactly_one_assessment_per_gate() -> None:
    with pytest.raises(ValueError, match="duplicate gate"):
        SafetyMigrationRuleAssessment(
            rule_id="duplicate",
            current_owner="legacy:test",
            target_disposition=None,
            gate_assessments=(
                _gate("m5_8a_inventory", "ready"),
                _gate("m5_8a_inventory", "ready"),
                _gate("m6_shared_authority", "blocked"),
            ),
        )

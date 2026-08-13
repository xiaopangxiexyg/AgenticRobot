from __future__ import annotations

from tests.test_rm65_dispatch_inventory import _execution_manifest
from teleop_runtime.backends.arms.rm65_execution_evidence import (
    build_rm65_execution_evidence_bundle,
)


def test_candidate_evidence_bundle_is_complete_but_unbound() -> None:
    bundle = build_rm65_execution_evidence_bundle(_execution_manifest())

    assert bundle.inventory_gate.ready
    assert not bundle.legacy_mediation_gate.ready
    assert not bundle.shared_authority_gate.ready
    assert bundle.integration_status == "evidence_only_unbound"
    assert not bundle.production_binding_enabled
    assert not bundle.confers_execution_authorization
    assert not bundle.inventory_assessment.legacy_mediation_ready
    assert len(bundle.dispatch_inventory.entries) == 16
    assert len(bundle.worker_drain_plan.participants) == 3


def test_candidate_evidence_bundle_is_deterministic_and_path_independent() -> None:
    first = build_rm65_execution_evidence_bundle(_execution_manifest())
    second = build_rm65_execution_evidence_bundle(_execution_manifest())

    assert first.semantic_content_hash == second.semantic_content_hash
    assert first.safety_migration_matrix.semantic_content_hash == (
        second.safety_migration_matrix.semantic_content_hash
    )
    assert "/tmp/" not in first.semantic_content_hash

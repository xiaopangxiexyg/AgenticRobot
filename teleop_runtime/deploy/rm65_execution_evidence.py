"""Evidence-only composition for the first RM65 policy-qpos candidate.

The bundle makes the M5.8-a review inputs inseparable without creating a
runtime owner, worker, permit, backend, or hardware authorization.  A future
production composition root may consume this bundle, but must still complete
M5.8-b and M6 gates independently.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ControlExecutionManifest,
    SafetyMigrationGateDecision,
    SafetyMigrationProfileMatrix,
    assess_safety_migration_gate,
)
from robot_motion.runtime import (
    DispatchSeamInventoryAssessment,
    DispatchSeamInventoryManifest,
)
from teleop_runtime.action.rm65_candidate_safety_matrix import (
    build_m6_candidate_policy_arm_qpos_scoped_safety_matrix,
)
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    assess_rm65_arms_only_dispatch_inventory,
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainPlan,
    build_rm65_arms_only_worker_drain_plan,
)


RM65_EXECUTION_EVIDENCE_BUNDLE_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_execution_evidence_bundle.v1"
)


@dataclass(frozen=True)
class RM65ExecutionEvidenceBundle:
    execution_manifest: ControlExecutionManifest
    dispatch_inventory: DispatchSeamInventoryManifest
    inventory_assessment: DispatchSeamInventoryAssessment
    worker_drain_plan: RM65ArmsOnlyWorkerDrainPlan
    safety_migration_matrix: SafetyMigrationProfileMatrix
    inventory_gate: SafetyMigrationGateDecision
    legacy_mediation_gate: SafetyMigrationGateDecision
    shared_authority_gate: SafetyMigrationGateDecision
    integration_status: str
    semantic_content_hash: str

    @property
    def production_binding_enabled(self) -> bool:
        return False

    @property
    def confers_execution_authorization(self) -> bool:
        return False


def build_rm65_execution_evidence_bundle(
    manifest: ControlExecutionManifest,
) -> RM65ExecutionEvidenceBundle:
    """Build one deterministic, fail-closed M5.8-a evidence bundle."""

    if not isinstance(manifest, ControlExecutionManifest):
        raise TypeError("manifest must be a ControlExecutionManifest.")
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    inventory_assessment = assess_rm65_arms_only_dispatch_inventory(inventory)
    worker_drain_plan = build_rm65_arms_only_worker_drain_plan(
        manifest=manifest,
        inventory=inventory,
    )
    matrix = build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
        manifest=manifest,
        inventory=inventory,
        inventory_assessment=inventory_assessment,
        worker_drain_plan=worker_drain_plan,
    )
    inventory_gate = assess_safety_migration_gate(
        matrix,
        gate_stage="m5_8a_inventory",
    )
    legacy_gate = assess_safety_migration_gate(
        matrix,
        gate_stage="m5_8b_legacy_mediation",
    )
    authority_gate = assess_safety_migration_gate(
        matrix,
        gate_stage="m6_shared_authority",
    )
    if not inventory_gate.ready:
        raise ValueError("RM65 candidate does not pass its M5.8-a inventory gate.")
    if legacy_gate.ready or authority_gate.ready:
        raise ValueError(
            "Evidence-only composition must not claim real mediation or authority."
        )

    semantic_payload = {
        "execution_manifest_id": manifest.manifest_id,
        "execution_manifest_hash": manifest.semantic_content_hash,
        "dispatch_inventory_id": inventory.inventory_id,
        "dispatch_inventory_hash": inventory.semantic_content_hash,
        "inventory_assessment_hash": inventory_assessment.semantic_content_hash,
        "worker_drain_plan_id": worker_drain_plan.plan_id,
        "worker_drain_plan_hash": worker_drain_plan.semantic_content_hash,
        "safety_migration_matrix_hash": matrix.semantic_content_hash,
        "inventory_gate_ready": inventory_gate.ready,
        "legacy_mediation_gate_ready": legacy_gate.ready,
        "shared_authority_gate_ready": authority_gate.ready,
        "integration_status": "evidence_only_unbound",
    }
    return RM65ExecutionEvidenceBundle(
        execution_manifest=manifest,
        dispatch_inventory=inventory,
        inventory_assessment=inventory_assessment,
        worker_drain_plan=worker_drain_plan,
        safety_migration_matrix=matrix,
        inventory_gate=inventory_gate,
        legacy_mediation_gate=legacy_gate,
        shared_authority_gate=authority_gate,
        integration_status="evidence_only_unbound",
        semantic_content_hash=canonical_content_hash(
            semantic_payload,
            domain=RM65_EXECUTION_EVIDENCE_BUNDLE_HASH_DOMAIN_V1,
        ),
    )


__all__ = [
    "RM65ExecutionEvidenceBundle",
    "build_rm65_execution_evidence_bundle",
]

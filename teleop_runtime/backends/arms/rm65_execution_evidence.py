"""Evidence-only composition for the first RM65 policy-qpos candidate.

The bundle makes the M5.8-a review inputs inseparable without creating a
runtime owner, worker, permit, backend, or hardware authorization.  A future
production composition root may consume this bundle, but must still complete
M5.8-b and M6 gates independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from dataset_contracts.canonical_encoding import canonical_content_hash
from robot_motion.action import (
    ControlExecutionManifest,
    ControlExecutionManifestIdentity,
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
from .rm65_dispatch_inventory import (
    assess_rm65_arms_only_dispatch_inventory,
    build_rm65_arms_only_dispatch_inventory,
)
from .rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainPlan,
    build_rm65_arms_only_worker_drain_plan,
)


RM65_EXECUTION_EVIDENCE_BUNDLE_HASH_DOMAIN_V1 = (
    "dexteleop.rm65_execution_evidence_bundle.v1"
)
RM65_M5_8B0_STOP_ONLY_FIELD_EVIDENCE_V1 = (
    "rm65_m5_8b0_stop_only_field_evidence_v1"
)
RM65_M5_8B1_IDLE_WORKER_FIELD_EVIDENCE_V1 = (
    "rm65_m5_8b1_real_a_idle_worker_drain_field_evidence_v1"
)


@dataclass(frozen=True)
class RM65HistoricalExecutionEvidenceContract:
    """Hash-only identity retained by one immutable field-evidence schema.

    The 2026-07-21 evidence did not archive the full execution-manifest
    payload, so this contract cannot and must not reconstruct a historical
    ``ControlExecutionManifest``.  It only resolves the exact identities that
    the field artifact declared at capture time.
    """

    evidence_schema: str
    manifest_identity: ControlExecutionManifestIdentity
    dispatch_inventory_hash: str
    worker_drain_plan_hash: str | None
    required_worker_participant_ids: tuple[str, ...] = ()
    manifest_payload_status: str = "hash_only_not_archived"

    def __post_init__(self) -> None:
        if not str(self.evidence_schema).strip():
            raise ValueError("evidence_schema must be non-empty.")
        if not isinstance(
            self.manifest_identity,
            ControlExecutionManifestIdentity,
        ):
            raise TypeError(
                "manifest_identity must be a ControlExecutionManifestIdentity."
            )
        if not str(self.dispatch_inventory_hash).strip():
            raise ValueError("dispatch_inventory_hash must be non-empty.")
        if self.worker_drain_plan_hash is not None and not str(
            self.worker_drain_plan_hash
        ).strip():
            raise ValueError("worker_drain_plan_hash must be non-empty when set.")
        if self.manifest_payload_status != "hash_only_not_archived":
            raise ValueError("Historical RM65 evidence is hash-only.")
        participants = tuple(
            str(value).strip() for value in self.required_worker_participant_ids
        )
        if any(not value for value in participants) or len(set(participants)) != len(
            participants
        ):
            raise ValueError("required_worker_participant_ids must be unique.")
        object.__setattr__(self, "required_worker_participant_ids", participants)


_RM65_2026_07_21_MANIFEST_IDENTITY = ControlExecutionManifestIdentity(
    manifest_id="m6_candidate_policy_arm_qpos_v1",
    manifest_version="v1",
    manifest_hash=(
        "964ea0f52b877eeead80145f4b50a2c7c22051f5615773faef1d504805b46bc4"
    ),
)
_RM65_2026_07_21_DISPATCH_INVENTORY_HASH = (
    "62c603cfabe0552648c7075108a7171baeecd4f1daaf13f37e461decd5c1de53"
)
_RM65_2026_07_21_WORKER_DRAIN_PLAN_HASH = (
    "547e54b18928fe2890edaab34a88052b0c323502a89f8ac3066b105c2dab4e90"
)
_RM65_2026_07_21_WORKER_PARTICIPANTS = (
    "position-deploy-reference-worker",
    "rm65-left-background-stream-worker",
    "rm65-right-background-stream-worker",
)
_RM65_HISTORICAL_EXECUTION_EVIDENCE_CONTRACTS = MappingProxyType(
    {
        RM65_M5_8B0_STOP_ONLY_FIELD_EVIDENCE_V1: (
            RM65HistoricalExecutionEvidenceContract(
                evidence_schema=RM65_M5_8B0_STOP_ONLY_FIELD_EVIDENCE_V1,
                manifest_identity=_RM65_2026_07_21_MANIFEST_IDENTITY,
                dispatch_inventory_hash=(
                    _RM65_2026_07_21_DISPATCH_INVENTORY_HASH
                ),
                worker_drain_plan_hash=None,
            )
        ),
        RM65_M5_8B1_IDLE_WORKER_FIELD_EVIDENCE_V1: (
            RM65HistoricalExecutionEvidenceContract(
                evidence_schema=RM65_M5_8B1_IDLE_WORKER_FIELD_EVIDENCE_V1,
                manifest_identity=_RM65_2026_07_21_MANIFEST_IDENTITY,
                dispatch_inventory_hash=(
                    _RM65_2026_07_21_DISPATCH_INVENTORY_HASH
                ),
                worker_drain_plan_hash=(
                    _RM65_2026_07_21_WORKER_DRAIN_PLAN_HASH
                ),
                required_worker_participant_ids=(
                    _RM65_2026_07_21_WORKER_PARTICIPANTS
                ),
            )
        ),
    }
)


def resolve_rm65_historical_execution_evidence_contract(
    evidence_schema: str,
) -> RM65HistoricalExecutionEvidenceContract:
    """Resolve one archived field-evidence schema without using current builders."""

    normalized = str(evidence_schema).strip()
    try:
        return _RM65_HISTORICAL_EXECUTION_EVIDENCE_CONTRACTS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown RM65 historical execution evidence schema: {normalized!r}."
        ) from exc


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
    "RM65_M5_8B0_STOP_ONLY_FIELD_EVIDENCE_V1",
    "RM65_M5_8B1_IDLE_WORKER_FIELD_EVIDENCE_V1",
    "RM65ExecutionEvidenceBundle",
    "RM65HistoricalExecutionEvidenceContract",
    "build_rm65_execution_evidence_bundle",
    "resolve_rm65_historical_execution_evidence_contract",
]

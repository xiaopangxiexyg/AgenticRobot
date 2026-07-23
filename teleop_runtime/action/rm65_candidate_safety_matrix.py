"""Scoped safety-migration evidence for the first arms-only M6 candidate.

This builder signs only the read-only M5.8-a inventory boundary.  Every real
legacy-mediation and shared-authority rule remains blocked until its explicit
runtime/hardware evidence exists.
"""

from __future__ import annotations

from robot_motion.action.execution_manifest import (
    M6_CANDIDATE_POLICY_ARM_QPOS_V1,
    ControlExecutionManifest,
)
from robot_motion.action.safety_migration import (
    SafetyMigrationGateAssessment,
    SafetyMigrationProfileMatrix,
    SafetyMigrationRuleAssessment,
)
from robot_motion.runtime.dispatch_inventory import (
    DispatchSeamInventoryAssessment,
    DispatchSeamInventoryManifest,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    RM65ArmsOnlyWorkerDrainPlan,
)


M6_CANDIDATE_POLICY_ARM_QPOS_SAFETY_MATRIX_V1 = (
    "m6_candidate_policy_arm_qpos_v1.scoped_safety_matrix_v1"
)

_REQUIRED_EXCLUSIONS = frozenset(
    {
        "hand_inspire_v1",
        "hand_dmtac_v1",
        "replay_authority",
        "pose_impedance",
        "fix_arm_human_motion",
        "human_motion_takeover",
    }
)


def _ready_inventory(*evidence_ids: str) -> SafetyMigrationGateAssessment:
    return SafetyMigrationGateAssessment(
        gate_stage="m5_8a_inventory",
        status="ready",
        evidence_ids=tuple(evidence_ids),
    )


def _blocked(stage: str, *reasons: str) -> SafetyMigrationGateAssessment:
    return SafetyMigrationGateAssessment(
        gate_stage=stage,
        status="blocked",
        reasons=tuple(reasons),
    )


def _included_rule(
    *,
    rule_id: str,
    current_owner: str,
    target_disposition: str,
    inventory_evidence: tuple[str, ...],
    disposition_evidence: tuple[str, ...],
    mediation_blockers: tuple[str, ...],
    authority_blockers: tuple[str, ...],
) -> SafetyMigrationRuleAssessment:
    return SafetyMigrationRuleAssessment(
        rule_id=rule_id,
        current_owner=current_owner,
        target_disposition=target_disposition,
        disposition_evidence_ids=disposition_evidence,
        gate_assessments=(
            _ready_inventory(*inventory_evidence),
            _blocked("m5_8b_legacy_mediation", *mediation_blockers),
            _blocked("m6_shared_authority", *authority_blockers),
        ),
    )


def _excluded_rule(
    *,
    rule_id: str,
    capability_id: str,
    manifest: ControlExecutionManifest,
) -> SafetyMigrationRuleAssessment:
    evidence = (
        f"execution-manifest:{manifest.semantic_content_hash}",
        f"manifest-exclusion:{capability_id}",
    )
    return SafetyMigrationRuleAssessment(
        rule_id=rule_id,
        current_owner="not_enabled_by_execution_manifest",
        target_disposition="not_applicable",
        disposition_evidence_ids=evidence,
        gate_assessments=tuple(
            SafetyMigrationGateAssessment(
                gate_stage=stage,
                status="not_applicable",
                evidence_ids=evidence,
            )
            for stage in (
                "m5_8a_inventory",
                "m5_8b_legacy_mediation",
                "m6_shared_authority",
            )
        ),
    )


def build_m6_candidate_policy_arm_qpos_scoped_safety_matrix(
    *,
    manifest: ControlExecutionManifest,
    inventory: DispatchSeamInventoryManifest,
    inventory_assessment: DispatchSeamInventoryAssessment,
    worker_drain_plan: RM65ArmsOnlyWorkerDrainPlan,
) -> SafetyMigrationProfileMatrix:
    """Build an M5.8-a-ready, M5.8-b/M6-blocked evidence matrix."""

    if not isinstance(manifest, ControlExecutionManifest):
        raise TypeError("manifest must be a ControlExecutionManifest.")
    if not isinstance(inventory, DispatchSeamInventoryManifest):
        raise TypeError("inventory must be a DispatchSeamInventoryManifest.")
    if not isinstance(inventory_assessment, DispatchSeamInventoryAssessment):
        raise TypeError(
            "inventory_assessment must be a DispatchSeamInventoryAssessment."
        )
    if not isinstance(worker_drain_plan, RM65ArmsOnlyWorkerDrainPlan):
        raise TypeError("worker_drain_plan must be an RM65ArmsOnlyWorkerDrainPlan.")
    if manifest.manifest_id != M6_CANDIDATE_POLICY_ARM_QPOS_V1:
        raise ValueError("Scoped matrix requires the first M6 candidate manifest.")
    if (
        inventory.execution_manifest_id != manifest.manifest_id
        or inventory.execution_manifest_hash != manifest.semantic_content_hash
    ):
        raise ValueError("Dispatch inventory is bound to another execution manifest.")
    if (
        inventory_assessment.inventory_id != inventory.inventory_id
        or inventory_assessment.inventory_hash != inventory.semantic_content_hash
    ):
        raise ValueError("Inventory assessment does not describe this inventory.")
    if not inventory_assessment.inventory_structurally_complete:
        raise ValueError("Scoped matrix requires a structurally complete M5.8-a inventory.")
    if inventory_assessment.confers_execution_authorization:
        raise ValueError("Inventory assessment must not confer execution authorization.")
    if (
        worker_drain_plan.execution_manifest_id != manifest.manifest_id
        or worker_drain_plan.execution_manifest_hash != manifest.semantic_content_hash
        or worker_drain_plan.dispatch_inventory_id != inventory.inventory_id
        or worker_drain_plan.dispatch_inventory_hash != inventory.semantic_content_hash
    ):
        raise ValueError(
            "Worker-drain plan is not bound to this manifest and inventory."
        )
    if worker_drain_plan.integration_status != "scoped_unbound":
        raise ValueError("Scoped matrix requires an unbound worker-drain plan.")
    exclusions = {item.capability_id for item in manifest.exclusions}
    if not _REQUIRED_EXCLUSIONS.issubset(exclusions):
        raise ValueError("First-candidate manifest lacks required exclusion evidence.")

    common_inventory_evidence = (
        f"execution-manifest:{manifest.semantic_content_hash}",
        f"dispatch-inventory:{inventory.semantic_content_hash}",
        f"inventory-assessment:{inventory_assessment.semantic_content_hash}",
        f"worker-drain-plan:{worker_drain_plan.semantic_content_hash}",
    )
    rules = (
        _included_rule(
            rule_id="policy_qpos_source_validation",
            current_owner="legacy_policy_action_adapter_and_rm65_send_command",
            target_disposition="preserved_source_validation",
            inventory_evidence=common_inventory_evidence
            + ("contract:absolute-bimanual-qpos-proposal-v1",),
            disposition_evidence=("contract:source-validation-firewall-v1",),
            mediation_blockers=(
                "production policy caller has not activated strict typed source validation",
            ),
            authority_blockers=(
                "real source-validation parity and target binding are not accepted",
            ),
        ),
        _included_rule(
            rule_id="policy_static_safety_override",
            current_owner="legacy_action_safety_filter_metadata_override",
            target_disposition="moved_shared_safety",
            inventory_evidence=common_inventory_evidence
            + ("contract:safety-constraint-override-admission-v1",),
            disposition_evidence=("contract:static-tightening-only-safety-override-v1",),
            mediation_blockers=(
                "production ActionSafetyFilter still consumes per-action metadata overrides",
            ),
            authority_blockers=(
                "artifact/session override receipt is not the sole production safety input",
            ),
        ),
        _included_rule(
            rule_id="qpos_keyframe_limits_and_anchor",
            current_owner="legacy_action_safety_filter_and_position_executor",
            target_disposition="moved_shared_safety",
            inventory_evidence=common_inventory_evidence
            + ("contract:previous-safe-committed-target-anchor-v1",),
            disposition_evidence=("contract:canonical-qpos-keyframe-safety-v1",),
            mediation_blockers=(
                "real shared anchor commit/admission is not wired to the legacy worker",
            ),
            authority_blockers=(
                "qpos limit and rebase parity lack real-hardware acceptance evidence",
            ),
        ),
        _included_rule(
            rule_id="scheduled_segment_generation_and_activation",
            current_owner="legacy_position_and_rm65_background_workers",
            target_disposition="moved_reference_planner",
            inventory_evidence=common_inventory_evidence
            + ("contract:scheduled-reference-segment-v1",),
            disposition_evidence=("contract:target-time-anchor-and-generation-timeline-v1",),
            mediation_blockers=(
                "real worker does not enforce shared execution epoch/group generation",
            ),
            authority_blockers=(
                "canonical scheduled-segment owner exists only in scoped sim",
            ),
        ),
        _included_rule(
            rule_id="trajectory_degraded_and_quick_stop",
            current_owner="legacy_rm65_trajectory_planner_and_stop_paths",
            target_disposition="moved_reference_planner",
            inventory_evidence=common_inventory_evidence
            + ("contract:rm65-quick-stop-operation-and-tick-v1",),
            disposition_evidence=("contract:reference-trajectory-safety-decision-v1",),
            mediation_blockers=(
                "quick-stop operation/tick permit is pure-only and degraded reasons lack production lineage",
            ),
            authority_blockers=(
                "real stop acknowledgement and trajectory safety parity are not accepted",
            ),
        ),
        _included_rule(
            rule_id="backend_readiness_receipt_and_final_dispatch",
            current_owner="legacy_rm65_backend_guards_and_sdk_receipts",
            target_disposition="moved_backend_guard",
            inventory_evidence=common_inventory_evidence,
            disposition_evidence=("contract:typed-final-dispatch-seam-v1",),
            mediation_blockers=(
                "typed RM65 joint-target wrapper is not production-bound and other required seams remain unwrapped",
            ),
            authority_blockers=(
                "no real one-shot permit, SDK receipt, or latency acceptance evidence",
            ),
        ),
        _included_rule(
            rule_id="execution_cancellation_and_worker_drain",
            current_owner="legacy_worker_stop_clear_and_join_logic",
            target_disposition="moved_backend_guard",
            inventory_evidence=common_inventory_evidence
            + ("contract:all-required-worker-epoch-drain-v1",),
            disposition_evidence=("contract:m1c-cancellation-and-drain-v1",),
            mediation_blockers=(
                "real position/RM65 workers do not emit typed drain receipts",
            ),
            authority_blockers=(
                "software, SDK, backend-state, and measured-stop acknowledgement are incomplete",
            ),
        ),
        _included_rule(
            rule_id="measured_tracking_divergence",
            current_owner="legacy_tracking_and_fault_observation",
            target_disposition="moved_shared_safety",
            inventory_evidence=common_inventory_evidence
            + ("contract:measured-divergence-guard-v1",),
            disposition_evidence=("contract:measured-divergence-shared-safety-v1",),
            mediation_blockers=(
                "legacy-owner inventory may keep shared divergence guard disabled",
            ),
            authority_blockers=(
                "real qpos thresholds, freshness, and violation policy lack hardware evidence",
            ),
        ),
        _included_rule(
            rule_id="home_recovery_and_rearm",
            current_owner="legacy_policy_qpos_init_and_position_manual_rearm",
            target_disposition="reimplemented",
            inventory_evidence=common_inventory_evidence
            + ("contract:lifecycle-home-recovery-rearm-v1",),
            disposition_evidence=("contract:typed-lifecycle-state-machine-v1",),
            mediation_blockers=(
                "production lifecycle state machine does not yet own exact home/rearm permits",
            ),
            authority_blockers=(
                "fresh measured rebase and real recovery acceptance are incomplete",
            ),
        ),
        _excluded_rule(
            rule_id="excluded_hand_inspire",
            capability_id="hand_inspire_v1",
            manifest=manifest,
        ),
        _excluded_rule(
            rule_id="excluded_hand_dmtac",
            capability_id="hand_dmtac_v1",
            manifest=manifest,
        ),
        _excluded_rule(
            rule_id="excluded_human_motion_takeover",
            capability_id="human_motion_takeover",
            manifest=manifest,
        ),
        _excluded_rule(
            rule_id="excluded_replay_authority",
            capability_id="replay_authority",
            manifest=manifest,
        ),
        _excluded_rule(
            rule_id="excluded_pose_impedance",
            capability_id="pose_impedance",
            manifest=manifest,
        ),
        _excluded_rule(
            rule_id="excluded_fix_arm_human_motion",
            capability_id="fix_arm_human_motion",
            manifest=manifest,
        ),
    )
    return SafetyMigrationProfileMatrix(
        profile_id=manifest.manifest_id,
        matrix_version=M6_CANDIDATE_POLICY_ARM_QPOS_SAFETY_MATRIX_V1,
        rules=rules,
    )

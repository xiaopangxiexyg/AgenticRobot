"""Pure policy-emission to execution-target compatibility and proposal preparation.

This module compares immutable source and target facts only.  It does not
record ControlTrace, call IK/safety, admit a proposal, schedule a reference,
request authority, or dispatch a command.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.replay import TargetRuntimeActionContract
from robot_motion.action import (
    ActionComponentCatalog,
    ActionProposalDraft,
    ReferenceActivationSpec,
)

from .runtime import PolicyActionEmission


POLICY_EMISSION_TARGET_ASSESSMENT_HASH_DOMAIN_V1 = (
    "dexteleop.policy_emission_target_assessment.v1"
)
POLICY_EMISSION_TARGET_PROPOSAL_PREPARATION_HASH_DOMAIN_V1 = (
    "dexteleop.policy_emission_target_proposal_preparation.v1"
)

REPRESENTATION_PROOF_LEGACY_V1 = "legacy_representation_receipt_v1"
REPRESENTATION_PROOF_TYPED_CANONICAL_V1 = (
    "typed_canonicalization_receipt_v1"
)


def _required_text(value: object, *, name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty.")
    return text


@dataclass(frozen=True)
class PolicyEmissionTargetCompatibilityAssessment:
    """Typed, non-authoritative result of one source/target comparison."""

    accepted: bool
    reasons: tuple[str, ...]
    emission_id: str
    emission_hash: str
    target_contract_hash: str
    component_catalog_hash: str
    reference_activation_spec_hash: str
    target_clock_domain_id: str
    authorized_component_ids: tuple[str, ...]
    component_coverage: tuple[str, ...]
    selection_groups: tuple[str, ...]
    representation_proof_kind: str | None
    representation_trace_id: str | None
    source_pose_frame_contract_hash: str | None
    source_pose_frame_binding_hash: str | None
    target_transform_receipt_hash: str | None
    available_target_lead_sec: float
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        reasons = tuple(dict.fromkeys(str(value) for value in self.reasons))
        object.__setattr__(self, "reasons", reasons)
        if bool(self.accepted) != (not reasons):
            raise ValueError("accepted must be true exactly when reasons is empty.")
        payload = {
            "accepted": bool(self.accepted),
            "reasons": reasons,
            "emission_id": self.emission_id,
            "emission_hash": self.emission_hash,
            "target_contract_hash": self.target_contract_hash,
            "component_catalog_hash": self.component_catalog_hash,
            "reference_activation_spec_hash": (
                self.reference_activation_spec_hash
            ),
            "target_clock_domain_id": self.target_clock_domain_id,
            "authorized_component_ids": self.authorized_component_ids,
            "component_coverage": self.component_coverage,
            "selection_groups": self.selection_groups,
            "representation_proof_kind": self.representation_proof_kind,
            "representation_trace_id": self.representation_trace_id,
            "source_pose_frame_contract_hash": (
                self.source_pose_frame_contract_hash
            ),
            "source_pose_frame_binding_hash": (
                self.source_pose_frame_binding_hash
            ),
            "target_transform_receipt_hash": (
                self.target_transform_receipt_hash
            ),
            "available_target_lead_sec": self.available_target_lead_sec,
        }
        object.__setattr__(
            self,
            "semantic_content_hash",
            canonical_content_hash(
                payload,
                domain=POLICY_EMISSION_TARGET_ASSESSMENT_HASH_DOMAIN_V1,
            ),
        )


def assess_policy_emission_target_compatibility(
    emission: PolicyActionEmission,
    *,
    target_contract: TargetRuntimeActionContract,
    component_catalog: ActionComponentCatalog,
    reference_activation_spec: ReferenceActivationSpec,
    target_clock_domain_id: str,
    authorized_component_ids: tuple[str, ...] | None = None,
    legacy_pose_frame_contract_hash: str | None = None,
    legacy_pose_frame_binding_hash: str | None = None,
) -> PolicyEmissionTargetCompatibilityAssessment:
    """Assess representation, component, frame, clock and lead compatibility.

    ``authorized_component_ids`` is an explicit authority scope, not an
    implicit subset rule.  It defaults to the complete target catalog and may
    not split a coupled selection group.
    """

    if not isinstance(emission, PolicyActionEmission):
        raise TypeError("emission must be PolicyActionEmission.")
    if not isinstance(target_contract, TargetRuntimeActionContract):
        raise TypeError("target_contract must be TargetRuntimeActionContract.")
    if not isinstance(component_catalog, ActionComponentCatalog):
        raise TypeError("component_catalog must be ActionComponentCatalog.")
    if not isinstance(reference_activation_spec, ReferenceActivationSpec):
        raise TypeError(
            "reference_activation_spec must be ReferenceActivationSpec."
        )
    target_clock = _required_text(
        target_clock_domain_id,
        name="target_clock_domain_id",
    )
    if (
        legacy_pose_frame_contract_hash is None
    ) != (legacy_pose_frame_binding_hash is None):
        raise ValueError(
            "legacy pose frame contract and binding hashes must be provided together."
        )
    if legacy_pose_frame_contract_hash is not None:
        legacy_pose_frame_contract_hash = _required_text(
            legacy_pose_frame_contract_hash,
            name="legacy_pose_frame_contract_hash",
        )
        legacy_pose_frame_binding_hash = _required_text(
            legacy_pose_frame_binding_hash,
            name="legacy_pose_frame_binding_hash",
        )

    catalog_ids = tuple(item.component_id for item in component_catalog.components)
    authorized = (
        catalog_ids
        if authorized_component_ids is None
        else tuple(
            _required_text(value, name="authorized_component_id")
            for value in authorized_component_ids
        )
    )
    if not authorized or len(set(authorized)) != len(authorized):
        raise ValueError(
            "authorized_component_ids must be non-empty and unique."
        )
    unknown_authorized = set(authorized).difference(catalog_ids)
    if unknown_authorized:
        raise ValueError(
            "authorized_component_ids contain unknown target components: "
            f"{sorted(unknown_authorized)!r}."
        )
    for group in component_catalog.selection_group_ids:
        members = component_catalog.components_for_group(group)
        if not members or not members[0].coupled:
            continue
        selected = {
            item.component_id for item in members if item.component_id in authorized
        }
        if selected and selected != {item.component_id for item in members}:
            raise ValueError(
                f"authorized_component_ids split coupled group {group!r}."
            )

    reasons: list[str] = []
    if target_contract.component_schema_hash != (
        component_catalog.semantic_content_hash
    ):
        reasons.append("target_component_schema_mismatch")
    if tuple(target_contract.component_ids) != catalog_ids:
        reasons.append("target_component_catalog_mismatch")
    if target_contract.unresolved_dimensions:
        reasons.append("target_contract_has_unresolved_dimensions")
    capability_hash = target_contract.execution_capability_profile_hash
    if capability_hash is not None and capability_hash != (
        reference_activation_spec.execution_capability_profile_hash
    ):
        reasons.append("target_activation_capability_mismatch")
    if (
        emission.produced_clock_domain_id != target_clock
        or emission.effective_clock_domain_id != target_clock
    ):
        reasons.append("emission_clock_domain_mismatch")

    available_lead = (
        float(emission.target_effective_at) - float(emission.emitted_at)
    )
    if not math.isfinite(available_lead):
        reasons.append("emission_target_lead_non_finite")
    elif available_lead + 1e-12 < (
        reference_activation_spec.target_lead_time_sec
    ):
        reasons.append("target_activation_lead_insufficient")

    action = emission.pre_safety_robot_action
    validation = emission.source_validation
    if action is None:
        reasons.append("emission_action_missing")
    if validation is None:
        reasons.append("source_validation_missing")
    elif not validation.accepted:
        reasons.append("source_validation_rejected")
    if emission.pre_safety_faults:
        reasons.append("emission_pre_safety_fault")

    proof_kind: str | None = None
    proof_id = emission.representation_trace_id
    source_frame_hash: str | None = None
    source_frame_binding_hash: str | None = None
    transform_receipt_hash: str | None = None
    if emission.canonical_action is not None:
        proof_kind = REPRESENTATION_PROOF_TYPED_CANONICAL_V1
        source_frame_hash = (
            emission.canonical_action.canonicalization_receipt.frame_contract_hash
        )
    elif emission.representation_receipt is not None:
        proof_kind = REPRESENTATION_PROOF_LEGACY_V1
        source_frame_hash = legacy_pose_frame_contract_hash
        source_frame_binding_hash = legacy_pose_frame_binding_hash
    if proof_id is None:
        reasons.append("representation_proof_missing")

    coverage: tuple[str, ...] = ()
    selection_groups: tuple[str, ...] = ()
    if action is not None:
        by_effector = {
            item.effector_name: item for item in component_catalog.components
        }
        unknown_effectors = set(action.active_effectors).difference(by_effector)
        if unknown_effectors:
            reasons.append("action_effector_not_in_target_catalog")
        coverage = tuple(
            item.component_id
            for item in component_catalog.components
            if item.effector_name in action.active_effectors
        )
        if set(coverage) != set(authorized):
            reasons.append("authorized_component_coverage_mismatch")
        selection_groups = tuple(
            group
            for group in component_catalog.selection_group_ids
            if any(
                item.component_id in coverage
                for item in component_catalog.components_for_group(group)
            )
        )
        if any(
            action.action_space
            not in target_contract.accepted_action_spaces_by_component[
                component_id
            ]
            for component_id in coverage
            if component_id in target_contract.accepted_action_spaces_by_component
        ):
            reasons.append("target_action_space_mismatch")

        if action.arm_pose_by_side:
            target_frame_hash = target_contract.pose_frame_contract_hash
            if target_frame_hash is None:
                reasons.append("target_pose_frame_contract_missing")
            elif source_frame_hash is None:
                reasons.append("source_pose_frame_contract_missing")
            elif source_frame_hash != target_frame_hash:
                transform_receipt_hash = (
                    target_contract.supported_transform_hash_by_source_frame.get(
                        source_frame_hash
                    )
                )
                if transform_receipt_hash is None:
                    reasons.append("target_pose_frame_contract_mismatch")

    normalized_reasons = tuple(dict.fromkeys(reasons))
    return PolicyEmissionTargetCompatibilityAssessment(
        accepted=not normalized_reasons,
        reasons=normalized_reasons,
        emission_id=emission.emission_id,
        emission_hash=emission.semantic_content_hash,
        target_contract_hash=target_contract.semantic_content_hash,
        component_catalog_hash=component_catalog.semantic_content_hash,
        reference_activation_spec_hash=(
            reference_activation_spec.semantic_content_hash
        ),
        target_clock_domain_id=target_clock,
        authorized_component_ids=authorized,
        component_coverage=coverage,
        selection_groups=selection_groups,
        representation_proof_kind=proof_kind,
        representation_trace_id=proof_id,
        source_pose_frame_contract_hash=source_frame_hash,
        source_pose_frame_binding_hash=source_frame_binding_hash,
        target_transform_receipt_hash=transform_receipt_hash,
        available_target_lead_sec=available_lead,
    )


@dataclass(frozen=True)
class PolicyEmissionTargetProposalPreparation:
    """Pure source-to-proposal preparation; never a receiver admission receipt."""

    assessment: PolicyEmissionTargetCompatibilityAssessment
    source_event_id: str
    proposal_draft: ActionProposalDraft | None
    preparation_id: str = field(init=False)
    semantic_content_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(
            self.assessment,
            PolicyEmissionTargetCompatibilityAssessment,
        ):
            raise TypeError(
                "assessment must be PolicyEmissionTargetCompatibilityAssessment."
            )
        source_event_id = _required_text(
            self.source_event_id,
            name="source_event_id",
        )
        if source_event_id != self.assessment.emission_id:
            raise ValueError("source_event_id must equal the assessed emission ID.")
        if self.assessment.accepted != (self.proposal_draft is not None):
            raise ValueError(
                "proposal_draft must exist exactly when target compatibility is accepted."
            )
        draft = self.proposal_draft
        draft_payload = None
        if draft is not None:
            if not isinstance(draft, ActionProposalDraft):
                raise TypeError("proposal_draft must be ActionProposalDraft.")
            if draft.representation_trace_id != (
                self.assessment.representation_trace_id
            ):
                raise ValueError(
                    "Proposal representation proof must match the assessment."
                )
            if tuple(draft.component_coverage) != (
                self.assessment.component_coverage
            ):
                raise ValueError(
                    "Proposal component coverage must match the assessment."
                )
            if tuple(draft.selection_groups) != self.assessment.selection_groups:
                raise ValueError(
                    "Proposal selection groups must match the assessment."
                )
            draft_payload = {
                "runtime_session_id": draft.runtime_session_id,
                "control_domain_id": draft.control_domain_id,
                "embodiment_instance_id": draft.embodiment_instance_id,
                "producer_kind": draft.producer_kind,
                "producer_id": draft.producer_id,
                "producer_epoch": draft.producer_epoch,
                "sequence": draft.sequence,
                "produced_at": draft.produced_at,
                "produced_clock_domain_id": draft.produced_clock_domain_id,
                "effective_from": draft.effective_from,
                "effective_until": draft.effective_until,
                "target_effective_at": draft.target_effective_at,
                "effective_clock_domain_id": draft.effective_clock_domain_id,
                "component_schema_hash": draft.component_schema_hash,
                "component_coverage": draft.component_coverage,
                "selection_groups": draft.selection_groups,
                "authority_priority": draft.authority_priority,
                "authority_request": draft.authority_request,
                "validity": draft.validity,
                "status": draft.status,
                "fallback_policy": draft.fallback_policy,
                "policy_node_id": draft.policy_node_id,
                "inference_id": draft.inference_id,
                "policy_output_id": draft.policy_output_id,
                "chunk_id": draft.chunk_id,
                "chunk_keyframe_index": draft.chunk_keyframe_index,
                "representation_trace_id": draft.representation_trace_id,
            }
        payload = {
            "assessment_hash": self.assessment.semantic_content_hash,
            "source_event_id": source_event_id,
            "emission_hash": self.assessment.emission_hash,
            "target_contract_hash": self.assessment.target_contract_hash,
            "component_catalog_hash": self.assessment.component_catalog_hash,
            "reference_activation_spec_hash": (
                self.assessment.reference_activation_spec_hash
            ),
            "proposal_draft": draft_payload,
        }
        semantic_hash = canonical_content_hash(
            payload,
            domain=(
                POLICY_EMISSION_TARGET_PROPOSAL_PREPARATION_HASH_DOMAIN_V1
            ),
        )
        object.__setattr__(self, "source_event_id", source_event_id)
        object.__setattr__(self, "semantic_content_hash", semantic_hash)
        object.__setattr__(
            self,
            "preparation_id",
            f"policy-emission-target-proposal-preparation:{semantic_hash}",
        )


def prepare_policy_emission_target_proposal(
    emission: PolicyActionEmission,
    *,
    target_contract: TargetRuntimeActionContract,
    component_catalog: ActionComponentCatalog,
    reference_activation_spec: ReferenceActivationSpec,
    target_clock_domain_id: str,
    runtime_session_id: str,
    producer_id: str,
    producer_epoch: int,
    sequence: int,
    authorized_component_ids: tuple[str, ...] | None = None,
    legacy_pose_frame_contract_hash: str | None = None,
    legacy_pose_frame_binding_hash: str | None = None,
) -> PolicyEmissionTargetProposalPreparation:
    """Prepare a non-authoritative policy proposal from one assessed emission.

    The target contract supplies the control-domain and embodiment identities.
    Rejected compatibility produces no proposal draft.  Accepted preparation
    still requires receiver-side proposal admission, authority, canonical
    safety, target resolution/admission, reference activation, and dispatch.
    """

    session = _required_text(runtime_session_id, name="runtime_session_id")
    producer = _required_text(producer_id, name="producer_id")
    epoch = int(producer_epoch)
    proposal_sequence = int(sequence)
    if isinstance(producer_epoch, bool) or epoch != producer_epoch or epoch < 0:
        raise ValueError("producer_epoch must be a non-negative integer.")
    if isinstance(sequence, bool) or proposal_sequence != sequence or proposal_sequence < 0:
        raise ValueError("sequence must be a non-negative integer.")

    assessment = assess_policy_emission_target_compatibility(
        emission,
        target_contract=target_contract,
        component_catalog=component_catalog,
        reference_activation_spec=reference_activation_spec,
        target_clock_domain_id=target_clock_domain_id,
        authorized_component_ids=authorized_component_ids,
        legacy_pose_frame_contract_hash=legacy_pose_frame_contract_hash,
        legacy_pose_frame_binding_hash=legacy_pose_frame_binding_hash,
    )
    draft: ActionProposalDraft | None = None
    if assessment.accepted:
        action = emission.pre_safety_robot_action
        if action is None:
            raise AssertionError(
                "Accepted target compatibility must include a policy action."
            )
        draft = ActionProposalDraft(
            runtime_session_id=session,
            control_domain_id=target_contract.control_domain_id,
            embodiment_instance_id=target_contract.embodiment_instance_id,
            producer_kind="policy",
            producer_id=producer,
            producer_epoch=epoch,
            sequence=proposal_sequence,
            action=action,
            produced_at=emission.produced_at,
            produced_clock_domain_id=emission.produced_clock_domain_id,
            effective_from=emission.effective_from,
            effective_until=emission.effective_until,
            target_effective_at=emission.target_effective_at,
            effective_clock_domain_id=emission.effective_clock_domain_id,
            component_schema_hash=component_catalog.semantic_content_hash,
            component_coverage=assessment.component_coverage,
            selection_groups=assessment.selection_groups,
            authority_priority=0,
            authority_request="none",
            validity="valid",
            status="ready",
            fallback_policy="no_action",
            policy_node_id=emission.policy_node_id,
            inference_id=emission.inference_id,
            policy_output_id=emission.policy_output_id,
            chunk_id=str(emission.chunk_id),
            chunk_keyframe_index=emission.chunk_keyframe_index,
            representation_trace_id=assessment.representation_trace_id,
            audit_metadata={
                "policy_action_emission_id": emission.emission_id,
                "policy_action_emission_hash": emission.semantic_content_hash,
                "policy_emission_target_assessment_hash": (
                    assessment.semantic_content_hash
                ),
                "target_runtime_action_contract_hash": (
                    target_contract.semantic_content_hash
                ),
                "reference_activation_spec_hash": (
                    reference_activation_spec.semantic_content_hash
                ),
                "representation_proof_kind": assessment.representation_proof_kind,
                "target_transform_receipt_hash": (
                    assessment.target_transform_receipt_hash
                ),
            },
        )
    return PolicyEmissionTargetProposalPreparation(
        assessment=assessment,
        source_event_id=emission.emission_id,
        proposal_draft=draft,
    )


__all__ = [
    "POLICY_EMISSION_TARGET_ASSESSMENT_HASH_DOMAIN_V1",
    "POLICY_EMISSION_TARGET_PROPOSAL_PREPARATION_HASH_DOMAIN_V1",
    "REPRESENTATION_PROOF_LEGACY_V1",
    "REPRESENTATION_PROOF_TYPED_CANONICAL_V1",
    "PolicyEmissionTargetCompatibilityAssessment",
    "PolicyEmissionTargetProposalPreparation",
    "assess_policy_emission_target_compatibility",
    "prepare_policy_emission_target_proposal",
]

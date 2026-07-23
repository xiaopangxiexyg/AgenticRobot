"""Pure replay source-to-target compatibility assessment.

This module compares immutable semantic contracts.  It does not read a
dataset, inspect live hardware, authorize motion, or dispatch commands.
"""

from __future__ import annotations

from typing import Mapping

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.replay import (
    REPLAY_BINDING_ASSESSMENT_HASH_DOMAIN_V1,
    REPLAY_BINDING_ELIGIBILITY_HASH_DOMAIN_V1,
    REPLAY_BINDING_INDETERMINATE,
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_USE_PASSIVE_SHADOW,
    REPLAY_BINDING_USE_RECORD_ONLY,
    REPLAY_BINDING_USES_V1,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_BINDING_VERIFIED_INCOMPATIBLE,
    REPLAY_EXECUTION_AUTHORIZATION_HASH_DOMAIN_V1,
    REPLAY_COMPAT_COMPATIBLE_SUBSET,
    REPLAY_COMPAT_DERIVED_MATCH,
    REPLAY_COMPAT_EXACT_MATCH,
    REPLAY_COMPAT_MISMATCH,
    REPLAY_COMPAT_NOT_APPLICABLE,
    REPLAY_COMPAT_UNKNOWN,
    REPLAY_COMPAT_VERIFIED_TRANSFORM,
    REPLAY_SOURCE_INTERPRETATION_INVALID,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    CandidateBindingEligibility,
    ReplayBindingAssessment,
    ReplayCompatibilityCheck,
    ReplayExecutionAuthorization,
    ReplaySourceActionContract,
    ReplayTargetBindingSpec,
    TargetRuntimeActionContract,
)
from robot_motion.action.components import ActionComponentCatalog


REPLAY_BINDING_VERIFIER_CONTRACT_ID_V1 = "dexteleop.replay_binding_verifier.v1"
REPLAY_BINDING_VERIFIER_CONTRACT_HASH_V1 = canonical_content_hash(
    {
        "contract_id": REPLAY_BINDING_VERIFIER_CONTRACT_ID_V1,
        "dimensions": (
            "source_interpretation",
            "action_stage",
            "action_space",
            "component_schema",
            "pose_frame",
            "units_convention",
            "component_coverage",
            "selection_group_atomicity",
            "action_specs",
            "execution_capability",
            "target_contract_completeness",
        ),
        "pose_frame_policy": (
            "explicit_qpos_spaces_without_pose_frame_are_not_applicable"
        ),
    },
    domain="dexteleop.replay_binding_verifier_contract.v1",
)

FRAMELESS_REPLAY_ACTION_SPACES_V1 = frozenset(
    {"bimanual_qpos_v1", "bimanual_qpos_hand16_v1"}
)


def build_target_runtime_action_contract(
    *,
    target_contract_id: str,
    control_domain_id: str,
    embodiment_instance_id: str,
    component_catalog: ActionComponentCatalog,
    pose_frame_contract_hash: str | None,
    action_spec_hash_by_component: Mapping[str, str] | None = None,
    units_convention_id: str = "si_m_rad_xyzw_v1",
    supported_transform_hash_by_source_frame: Mapping[str, str] | None = None,
    execution_capability_profile_hash: str | None = None,
    unresolved_dimensions: tuple[str, ...] = (),
) -> TargetRuntimeActionContract:
    """Build one backend-neutral immutable target action contract."""

    component_ids = tuple(item.component_id for item in component_catalog.components)
    return TargetRuntimeActionContract(
        target_contract_id=target_contract_id,
        control_domain_id=control_domain_id,
        embodiment_instance_id=embodiment_instance_id,
        component_schema_hash=component_catalog.semantic_content_hash,
        component_ids=component_ids,
        selection_group_by_component={
            item.component_id: item.selection_group_id
            for item in component_catalog.components
        },
        accepted_action_spaces_by_component={
            item.component_id: (component_catalog.action_space,)
            for item in component_catalog.components
        },
        coupled_selection_groups=tuple(
            dict.fromkeys(
                item.selection_group_id
                for item in component_catalog.components
                if item.coupled
            )
        ),
        pose_frame_contract_hash=pose_frame_contract_hash,
        action_spec_hash_by_component=dict(action_spec_hash_by_component or {}),
        units_convention_id=units_convention_id,
        supported_transform_hash_by_source_frame=dict(
            supported_transform_hash_by_source_frame or {}
        ),
        execution_capability_profile_hash=execution_capability_profile_hash,
        unresolved_dimensions=unresolved_dimensions,
    )


def build_replay_binding_assessment(
    source: ReplaySourceActionContract,
    target: TargetRuntimeActionContract,
    *,
    binding_id: str,
    transform_receipt_id: str | None = None,
    transform_receipt_hash: str | None = None,
) -> tuple[ReplayTargetBindingSpec, ReplayBindingAssessment]:
    """Build and assess one exact source-to-target relation."""

    binding = ReplayTargetBindingSpec(
        binding_id=binding_id,
        source_action_contract_id=source.contract_id,
        source_action_contract_hash=source.semantic_content_hash,
        target_runtime_contract_id=target.target_contract_id,
        target_runtime_contract_hash=target.semantic_content_hash,
        transform_receipt_id=transform_receipt_id,
        transform_receipt_hash=transform_receipt_hash,
    )
    return binding, assess_replay_binding(source, target, binding)


def build_replay_execution_authorization(
    assessment: ReplayBindingAssessment,
    *,
    requested_mode: str,
    safety_preflight_id: str,
    execution_capability_evidence_id: str,
    dispatch_owner_state_id: str,
    operator_authorization_id: str,
    authorized: bool,
    reasons: tuple[str, ...] = (),
) -> ReplayExecutionAuthorization:
    payload = {
        "requested_mode": str(requested_mode),
        "assessment_id": assessment.assessment_id,
        "assessment_hash": assessment.semantic_content_hash,
        "safety_preflight_id": str(safety_preflight_id),
        "execution_capability_evidence_id": str(
            execution_capability_evidence_id
        ),
        "dispatch_owner_state_id": str(dispatch_owner_state_id),
        "operator_authorization_id": str(operator_authorization_id),
        "authorized": bool(authorized),
        "reasons": tuple(str(value) for value in reasons),
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REPLAY_EXECUTION_AUTHORIZATION_HASH_DOMAIN_V1,
    )
    return ReplayExecutionAuthorization(
        authorization_id=f"replay-execution-authorization:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


def _check(
    dimension: str,
    status: str,
    reason: str,
    *evidence_refs: str,
) -> ReplayCompatibilityCheck:
    return ReplayCompatibilityCheck(
        dimension=dimension,
        status=status,
        reason=reason,
        evidence_refs=tuple(value for value in evidence_refs if value),
    )


def assess_replay_binding(
    source: ReplaySourceActionContract,
    target: TargetRuntimeActionContract,
    binding: ReplayTargetBindingSpec,
) -> ReplayBindingAssessment:
    """Return a deterministic three-state assessment for one immutable target."""

    if binding.source_action_contract_id != source.contract_id or (
        binding.source_action_contract_hash != source.semantic_content_hash
    ):
        raise ValueError("Replay binding spec does not reference the supplied source contract.")
    if binding.target_runtime_contract_id != target.target_contract_id or (
        binding.target_runtime_contract_hash != target.semantic_content_hash
    ):
        raise ValueError("Replay binding spec does not reference the supplied target contract.")

    checks: list[ReplayCompatibilityCheck] = []
    if source.interpretation_status == REPLAY_SOURCE_INTERPRETATION_VERIFIED:
        checks.append(
            _check(
                "source_interpretation",
                (
                    REPLAY_COMPAT_DERIVED_MATCH
                    if source.derivation_receipt_id is not None
                    else REPLAY_COMPAT_EXACT_MATCH
                ),
                (
                    "source action semantics are deterministically derived"
                    if source.derivation_receipt_id is not None
                    else "source action semantics are explicitly resolved"
                ),
                source.contract_id,
                source.derivation_receipt_id or "",
            )
        )
    elif source.interpretation_status == REPLAY_SOURCE_INTERPRETATION_INVALID:
        checks.append(
            _check(
                "source_interpretation",
                REPLAY_COMPAT_MISMATCH,
                "source action contract is invalid",
                source.contract_id,
            )
        )
    else:
        checks.append(
            _check(
                "source_interpretation",
                REPLAY_COMPAT_UNKNOWN,
                "source action contract is not fully interpretable",
                source.contract_id,
            )
        )

    checks.append(
        _check(
            "action_stage",
            REPLAY_COMPAT_EXACT_MATCH,
            f"canonical replay stage {source.action_stage} is explicitly selected",
        )
    )

    missing_components = tuple(
        component
        for component in source.component_ids
        if component not in target.component_ids
    )
    checks.append(
        _check(
            "component_coverage",
            REPLAY_COMPAT_MISMATCH if missing_components else (
                REPLAY_COMPAT_EXACT_MATCH
                if set(source.component_ids) == set(target.component_ids)
                else REPLAY_COMPAT_COMPATIBLE_SUBSET
            ),
            (
                "target lacks source components: " + ",".join(missing_components)
                if missing_components
                else "source components are a supported target subset"
            ),
        )
    )

    unsupported_spaces = tuple(
        component
        for component in source.component_ids
        if component in target.accepted_action_spaces_by_component
        and source.action_space
        not in target.accepted_action_spaces_by_component[component]
    )
    checks.append(
        _check(
            "action_space",
            REPLAY_COMPAT_MISMATCH if unsupported_spaces else REPLAY_COMPAT_EXACT_MATCH,
            (
                "target rejects source action space for: " + ",".join(unsupported_spaces)
                if unsupported_spaces
                else "target accepts the source action space for all source components"
            ),
        )
    )

    source_component_set = set(source.component_ids)
    target_component_set = set(target.component_ids)
    specs_match_for_source = all(
        source.action_spec_hash_by_component.get(component) is not None
        and source.action_spec_hash_by_component.get(component)
        == target.action_spec_hash_by_component.get(component)
        for component in source.component_ids
    )
    groups_match_for_source = all(
        source.selection_group_by_component.get(component)
        == target.selection_group_by_component.get(component)
        for component in source.component_ids
        if component in target.selection_group_by_component
    )
    if source.component_schema_hash is None:
        component_schema_status = REPLAY_COMPAT_UNKNOWN
        component_schema_reason = "source component schema is unresolved"
    elif source.component_schema_hash == target.component_schema_hash:
        component_schema_status = REPLAY_COMPAT_EXACT_MATCH
        component_schema_reason = "source and target component schemas match"
    elif (
        source_component_set < target_component_set
        and specs_match_for_source
        and groups_match_for_source
    ):
        component_schema_status = REPLAY_COMPAT_COMPATIBLE_SUBSET
        component_schema_reason = (
            "source schema is an explicitly verified component subset"
        )
    else:
        component_schema_status = REPLAY_COMPAT_UNKNOWN
        component_schema_reason = (
            "component schema hashes differ without sufficient subset evidence"
        )
    checks.append(
        _check(
            "component_schema",
            component_schema_status,
            component_schema_reason,
        )
    )

    if (
        source.action_space in FRAMELESS_REPLAY_ACTION_SPACES_V1
        and source.pose_frame_contract_hash is None
        and target.pose_frame_contract_hash is None
    ):
        frame_status = REPLAY_COMPAT_NOT_APPLICABLE
        frame_reason = "joint-space replay does not use a pose-frame contract"
    elif source.pose_frame_contract_hash is None or target.pose_frame_contract_hash is None:
        frame_status = REPLAY_COMPAT_UNKNOWN
        frame_reason = "source or target pose-frame contract is unresolved"
    elif source.pose_frame_contract_hash == target.pose_frame_contract_hash:
        frame_status = REPLAY_COMPAT_EXACT_MATCH
        frame_reason = "source and target pose-frame contracts match"
    elif (
        target.supported_transform_hash_by_source_frame.get(
            source.pose_frame_contract_hash
        )
        == binding.transform_receipt_hash
        and binding.transform_receipt_hash is not None
    ):
        frame_status = REPLAY_COMPAT_VERIFIED_TRANSFORM
        frame_reason = "a versioned target-supported pose transform is bound"
    else:
        frame_status = REPLAY_COMPAT_MISMATCH
        frame_reason = "pose-frame contracts differ without a verified transform"
    checks.append(_check("pose_frame", frame_status, frame_reason))

    if source.units_convention_id is None or target.units_convention_id is None:
        units_status = REPLAY_COMPAT_UNKNOWN
        units_reason = "source or target units/convention contract is unresolved"
    elif source.units_convention_id == target.units_convention_id:
        units_status = REPLAY_COMPAT_EXACT_MATCH
        units_reason = "units and conventions match"
    else:
        units_status = REPLAY_COMPAT_MISMATCH
        units_reason = "units or conventions differ"
    checks.append(_check("units_convention", units_status, units_reason))

    atomicity_mismatch: list[str] = []
    selection_group_mismatch = [
        component
        for component in source.component_ids
        if component in target.selection_group_by_component
        and source.selection_group_by_component[component]
        != target.selection_group_by_component[component]
    ]
    source_components = set(source.component_ids)
    for group in target.coupled_selection_groups:
        target_members = {
            component
            for component, group_id in target.selection_group_by_component.items()
            if group_id == group
        }
        if source_components.intersection(target_members) and not target_members.issubset(
            source_components
        ):
            atomicity_mismatch.append(group)
    checks.append(
        _check(
            "selection_group_atomicity",
            (
                REPLAY_COMPAT_MISMATCH
                if atomicity_mismatch or selection_group_mismatch
                else REPLAY_COMPAT_EXACT_MATCH
            ),
            (
                "source splits coupled target groups: " + ",".join(atomicity_mismatch)
                if atomicity_mismatch
                else (
                    "source and target selection groups differ for: "
                    + ",".join(selection_group_mismatch)
                    if selection_group_mismatch
                    else "source component coverage preserves target group atomicity"
                )
            ),
        )
    )

    spec_mismatch: list[str] = []
    spec_unknown: list[str] = []
    for component in source.component_ids:
        source_spec = source.action_spec_hash_by_component.get(component)
        target_spec = target.action_spec_hash_by_component.get(component)
        if source_spec is None or target_spec is None:
            spec_unknown.append(component)
        elif source_spec != target_spec:
            spec_mismatch.append(component)
    checks.append(
        _check(
            "action_specs",
            REPLAY_COMPAT_MISMATCH if spec_mismatch else (
                REPLAY_COMPAT_UNKNOWN if spec_unknown else REPLAY_COMPAT_EXACT_MATCH
            ),
            (
                "action specs differ for: " + ",".join(spec_mismatch)
                if spec_mismatch
                else (
                    "action specs unresolved for: " + ",".join(spec_unknown)
                    if spec_unknown
                    else "action specs match"
                )
            ),
        )
    )

    checks.append(
        _check(
            "execution_capability",
            REPLAY_COMPAT_UNKNOWN
            if target.execution_capability_profile_hash is None
            else REPLAY_COMPAT_EXACT_MATCH,
            (
                "target execution capability is unresolved"
                if target.execution_capability_profile_hash is None
                else "target execution capability profile is explicitly bound"
            ),
        )
    )
    checks.append(
        _check(
            "target_contract_completeness",
            (
                REPLAY_COMPAT_UNKNOWN
                if target.unresolved_dimensions
                else REPLAY_COMPAT_EXACT_MATCH
            ),
            (
                "target contract has unresolved dimensions: "
                + ",".join(target.unresolved_dimensions)
                if target.unresolved_dimensions
                else "target runtime action contract is complete"
            ),
        )
    )

    mismatches = tuple(
        check for check in checks if check.status == REPLAY_COMPAT_MISMATCH
    )
    unknowns = tuple(check for check in checks if check.status == REPLAY_COMPAT_UNKNOWN)
    if mismatches:
        verdict = REPLAY_BINDING_VERIFIED_INCOMPATIBLE
        permitted = (REPLAY_BINDING_USE_RECORD_ONLY,)
        blocking = tuple(f"{check.dimension}:{check.reason}" for check in mismatches)
    elif unknowns:
        verdict = REPLAY_BINDING_INDETERMINATE
        permitted = (
            REPLAY_BINDING_USE_RECORD_ONLY,
            REPLAY_BINDING_USE_PASSIVE_SHADOW,
        )
        blocking = tuple(f"{check.dimension}:{check.reason}" for check in unknowns)
    else:
        verdict = REPLAY_BINDING_VERIFIED_COMPATIBLE
        permitted = REPLAY_BINDING_USES_V1
        blocking = ()

    payload = {
        "binding_spec_id": binding.binding_id,
        "binding_spec_hash": binding.semantic_content_hash,
        "source_action_contract_id": source.contract_id,
        "source_action_contract_hash": source.semantic_content_hash,
        "target_runtime_contract_id": target.target_contract_id,
        "target_runtime_contract_hash": target.semantic_content_hash,
        "verdict": verdict,
        "checks": tuple(
            {
                "dimension": check.dimension,
                "status": check.status,
                "reason": check.reason,
                "evidence_refs": check.evidence_refs,
            }
            for check in checks
        ),
        "permitted_uses": permitted,
        "blocking_reasons": blocking,
        "verifier_contract_id": REPLAY_BINDING_VERIFIER_CONTRACT_ID_V1,
        "verifier_contract_hash": REPLAY_BINDING_VERIFIER_CONTRACT_HASH_V1,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REPLAY_BINDING_ASSESSMENT_HASH_DOMAIN_V1,
    )
    return ReplayBindingAssessment(
        assessment_id=f"replay-binding-assessment:{semantic_hash}",
        binding_spec_id=binding.binding_id,
        binding_spec_hash=binding.semantic_content_hash,
        source_action_contract_id=source.contract_id,
        source_action_contract_hash=source.semantic_content_hash,
        target_runtime_contract_id=target.target_contract_id,
        target_runtime_contract_hash=target.semantic_content_hash,
        verdict=verdict,
        checks=tuple(checks),
        permitted_uses=permitted,
        blocking_reasons=blocking,
        verifier_contract_id=REPLAY_BINDING_VERIFIER_CONTRACT_ID_V1,
        verifier_contract_hash=REPLAY_BINDING_VERIFIER_CONTRACT_HASH_V1,
        semantic_content_hash=semantic_hash,
    )


def build_candidate_binding_eligibility(
    assessment: ReplayBindingAssessment,
    *,
    source_emission_id: str,
    requested_use: str,
    same_legacy_emission: bool,
    source_interpretation_verified: bool,
) -> CandidateBindingEligibility:
    """Gate replay use without converting compatibility into motion authority."""

    use = str(requested_use)
    if use not in REPLAY_BINDING_USES_V1:
        raise ValueError(f"Unsupported replay binding use {use!r}.")
    reasons: list[str] = []
    allowed_by_assessment = assessment.allows(use)
    if not allowed_by_assessment:
        reasons.append(f"assessment_disallows:{use}")
    if not source_interpretation_verified:
        reasons.append("source_interpretation_not_verified")
    conditional = bool(
        use == REPLAY_BINDING_USE_PASSIVE_SHADOW
        and assessment.verdict == REPLAY_BINDING_INDETERMINATE
    )
    if conditional and not same_legacy_emission:
        reasons.append("indeterminate_shadow_requires_same_legacy_emission")
    evaluation_allowed = bool(
        use == REPLAY_BINDING_USE_PASSIVE_SHADOW and not reasons
    )
    authoritative_allowed = bool(
        use == REPLAY_BINDING_USE_EXECUTION_CANDIDATE
        and assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
        and not reasons
    )
    if use == REPLAY_BINDING_USE_RECORD_ONLY:
        reasons.append("record_only")
    payload = {
        "assessment_id": assessment.assessment_id,
        "assessment_hash": assessment.semantic_content_hash,
        "source_emission_id": str(source_emission_id),
        "requested_use": use,
        "evaluation_allowed": evaluation_allowed,
        "authoritative_allowed": authoritative_allowed,
        "conditional_evaluation": conditional and evaluation_allowed,
        "same_legacy_emission": bool(same_legacy_emission),
        "reasons": tuple(dict.fromkeys(reasons)),
        "target_runtime_contract_hash": assessment.target_runtime_contract_hash,
    }
    semantic_hash = canonical_content_hash(
        payload,
        domain=REPLAY_BINDING_ELIGIBILITY_HASH_DOMAIN_V1,
    )
    return CandidateBindingEligibility(
        eligibility_id=f"replay-binding-eligibility:{semantic_hash}",
        **payload,
        semantic_content_hash=semantic_hash,
    )


__all__ = [
    "REPLAY_BINDING_VERIFIER_CONTRACT_HASH_V1",
    "REPLAY_BINDING_VERIFIER_CONTRACT_ID_V1",
    "assess_replay_binding",
    "build_replay_binding_assessment",
    "build_candidate_binding_eligibility",
    "build_replay_execution_authorization",
    "build_target_runtime_action_contract",
]

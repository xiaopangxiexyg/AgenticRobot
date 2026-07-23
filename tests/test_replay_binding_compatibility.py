from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.replay import (
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_BINDING_INDETERMINATE,
    REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
    REPLAY_BINDING_USE_PASSIVE_SHADOW,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_BINDING_VERIFIED_INCOMPATIBLE,
    REPLAY_EXECUTION_MODE_SIM_AUTHORITY,
    REPLAY_COMPAT_COMPATIBLE_SUBSET,
    REPLAY_COMPAT_DERIVED_MATCH,
    REPLAY_COMPAT_VERIFIED_TRANSFORM,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    ReplaySourceActionContract,
    ReplaySourceContract,
    TargetRuntimeActionContract,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_interface.replay import derive_replay_source_action_contract
from robot_motion.replay.compatibility import (
    build_candidate_binding_eligibility,
    build_replay_execution_authorization,
)
from teleop_runtime.action.replay_binding import build_replay_binding_assessment


ACTION_SPACE = "bimanual_ee_pose_v1"


def _source(
    *,
    components: tuple[str, ...] = ("right_arm",),
    groups: dict[str, str] | None = None,
    frame: str | None = "frame-a",
    component_schema: str | None = "schema-a",
    specs: dict[str, str] | None = None,
) -> ReplaySourceActionContract:
    return ReplaySourceActionContract(
        contract_id="source-contract",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_space=ACTION_SPACE,
        component_ids=components,
        selection_group_by_component=(
            groups or {component: component for component in components}
        ),
        pose_frame_contract_hash=frame,
        component_schema_hash=component_schema,
        action_spec_hash_by_component=(
            specs or {component: f"{component}-spec" for component in components}
        ),
        units_convention_id="si_m_rad_xyzw_v1",
    )


def _target(
    *,
    target_id: str = "target-a",
    components: tuple[str, ...] = ("right_arm",),
    groups: dict[str, str] | None = None,
    coupled_groups: tuple[str, ...] = (),
    frame: str | None = "frame-a",
    component_schema: str = "schema-a",
    specs: dict[str, str] | None = None,
    transforms: dict[str, str] | None = None,
    capability: str | None = "capability-a",
    unresolved: tuple[str, ...] = (),
) -> TargetRuntimeActionContract:
    return TargetRuntimeActionContract(
        target_contract_id=target_id,
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        component_schema_hash=component_schema,
        component_ids=components,
        selection_group_by_component=(
            groups or {component: component for component in components}
        ),
        accepted_action_spaces_by_component={
            component: (ACTION_SPACE,) for component in components
        },
        coupled_selection_groups=coupled_groups,
        pose_frame_contract_hash=frame,
        action_spec_hash_by_component=(
            specs or {component: f"{component}-spec" for component in components}
        ),
        units_convention_id="si_m_rad_xyzw_v1",
        supported_transform_hash_by_source_frame=transforms or {},
        execution_capability_profile_hash=capability,
        unresolved_dimensions=unresolved,
    )


def _assessment(
    source: ReplaySourceActionContract,
    target: TargetRuntimeActionContract,
    *,
    transform_hash: str | None = None,
):
    return build_replay_binding_assessment(
        source,
        target,
        binding_id=f"{source.contract_id}-to-{target.target_contract_id}",
        transform_receipt_id=(
            None if transform_hash is None else "transform-receipt"
        ),
        transform_receipt_hash=transform_hash,
    )


def _status_by_dimension(assessment) -> dict[str, str]:
    return {check.dimension: check.status for check in assessment.checks}


def test_exact_contract_is_verified_compatible() -> None:
    _, assessment = _assessment(_source(), _target())
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    assert assessment.allows(REPLAY_BINDING_USE_EXECUTION_CANDIDATE)


def test_versioned_legacy_derivation_is_typed_evidence() -> None:
    dataset_contract = ReplaySourceContract(
        contract_id="legacy-source",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        source_dataset_id="dataset-a",
        episode_id="episode-000001",
        action_stage=REPLAY_ACTION_STAGE_SELECTED,
        action_field="action",
        source_action_space=ACTION_SPACE,
        source_frame_contract_hash="frame-a",
        source_component_schema_hash="schema-a",
    )
    source, receipt = derive_replay_source_action_contract(
        dataset_contract,
        component_ids=("right_arm",),
        selection_groups=("right_arm",),
        action_spec_hash_by_component={"right_arm": "right_arm-spec"},
    )
    _, assessment = _assessment(source, _target())
    assert receipt.derivation_status == REPLAY_SOURCE_INTERPRETATION_VERIFIED
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    assert _status_by_dimension(assessment)["source_interpretation"] == (
        REPLAY_COMPAT_DERIVED_MATCH
    )


def test_verified_frame_transform_is_compatible() -> None:
    target = _target(
        frame="frame-b",
        transforms={"frame-a": "transform-hash"},
    )
    _, assessment = _assessment(
        _source(),
        target,
        transform_hash="transform-hash",
    )
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    assert _status_by_dimension(assessment)["pose_frame"] == (
        REPLAY_COMPAT_VERIFIED_TRANSFORM
    )


def test_explicit_component_subset_is_compatible_but_coupled_split_is_not() -> None:
    target = _target(
        components=("left_arm", "right_arm"),
        component_schema="schema-full",
    )
    _, subset = _assessment(_source(), target)
    assert subset.verdict == REPLAY_BINDING_VERIFIED_COMPATIBLE
    assert _status_by_dimension(subset)["component_schema"] == (
        REPLAY_COMPAT_COMPATIBLE_SUBSET
    )

    coupled_groups = {"left_arm": "bimanual", "right_arm": "bimanual"}
    coupled_target = _target(
        target_id="target-coupled",
        components=("left_arm", "right_arm"),
        groups=coupled_groups,
        coupled_groups=("bimanual",),
        component_schema="schema-full",
    )
    coupled_source = _source(groups={"right_arm": "bimanual"})
    _, split = _assessment(coupled_source, coupled_target)
    assert split.verdict == REPLAY_BINDING_VERIFIED_INCOMPATIBLE


def test_unknown_target_is_not_incompatible_or_compatible() -> None:
    target = _target(
        target_id="target-unknown-frame",
        frame=None,
        unresolved=("pose_frame_contract",),
    )
    _, assessment = _assessment(_source(), target)
    assert assessment.verdict == REPLAY_BINDING_INDETERMINATE
    assert not assessment.allows(REPLAY_BINDING_USE_EXECUTION_CANDIDATE)

    detached = build_candidate_binding_eligibility(
        assessment,
        source_emission_id="emission-a",
        requested_use=REPLAY_BINDING_USE_PASSIVE_SHADOW,
        same_legacy_emission=False,
        source_interpretation_verified=True,
    )
    assert not detached.evaluation_allowed
    assert not detached.authoritative_allowed

    conditional = build_candidate_binding_eligibility(
        assessment,
        source_emission_id="emission-a",
        requested_use=REPLAY_BINDING_USE_PASSIVE_SHADOW,
        same_legacy_emission=True,
        source_interpretation_verified=True,
    )
    assert conditional.evaluation_allowed
    assert conditional.conditional_evaluation
    assert not conditional.authoritative_allowed

    execution = build_candidate_binding_eligibility(
        assessment,
        source_emission_id="emission-a",
        requested_use=REPLAY_BINDING_USE_EXECUTION_CANDIDATE,
        same_legacy_emission=True,
        source_interpretation_verified=True,
    )
    assert not execution.authoritative_allowed


def test_assessment_is_bound_to_exact_target_contract() -> None:
    source = _source()
    _, first = _assessment(source, _target(target_id="target-a"))
    _, second = _assessment(source, _target(target_id="target-b"))
    assert first.target_runtime_contract_hash != second.target_runtime_contract_hash
    assert first.semantic_content_hash != second.semantic_content_hash


def test_known_action_spec_mismatch_fails_closed() -> None:
    target = _target(specs={"right_arm": "different-spec"})
    _, assessment = _assessment(_source(), target)
    assert assessment.verdict == REPLAY_BINDING_VERIFIED_INCOMPATIBLE
    eligibility = build_candidate_binding_eligibility(
        assessment,
        source_emission_id="emission-a",
        requested_use=REPLAY_BINDING_USE_PASSIVE_SHADOW,
        same_legacy_emission=True,
        source_interpretation_verified=True,
    )
    assert not eligibility.evaluation_allowed
    assert not eligibility.authoritative_allowed


def test_binding_evidence_rejects_tampered_semantic_hashes() -> None:
    _, assessment = _assessment(_source(), _target())
    with pytest.raises(ValueError, match="assessment hash mismatch"):
        replace(assessment, semantic_content_hash="tampered")
    eligibility = build_candidate_binding_eligibility(
        assessment,
        source_emission_id="emission-a",
        requested_use=REPLAY_BINDING_USE_PASSIVE_SHADOW,
        same_legacy_emission=True,
        source_interpretation_verified=True,
    )
    with pytest.raises(ValueError, match="eligibility hash mismatch"):
        replace(eligibility, semantic_content_hash="tampered")


def test_execution_authorization_is_separate_hashed_evidence() -> None:
    _, assessment = _assessment(_source(), _target())
    authorization = build_replay_execution_authorization(
        assessment,
        requested_mode=REPLAY_EXECUTION_MODE_SIM_AUTHORITY,
        safety_preflight_id="sim-safety-v1",
        execution_capability_evidence_id="sim-capability-v1",
        dispatch_owner_state_id="dispatch-owner-token-1",
        operator_authorization_id="sim-config-opt-in",
        authorized=True,
    )
    assert authorization.authorized
    assert authorization.assessment_hash == assessment.semantic_content_hash
    with pytest.raises(ValueError, match="authorization hash mismatch"):
        replace(authorization, semantic_content_hash="tampered")

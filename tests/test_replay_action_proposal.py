from __future__ import annotations

import numpy as np
import pytest

from dataset_contracts.replay import (
    REPLAY_ACTION_STAGE_COMMANDED,
    REPLAY_ACTION_STAGE_SAFE,
    REPLAY_ACTION_STAGE_SELECTED,
    REPLAY_BINDING_INDETERMINATE,
    REPLAY_BINDING_VERIFIED_COMPATIBLE,
    REPLAY_BINDING_VERIFIED_INCOMPATIBLE,
    REPLAY_MODE_CANONICAL_ACTION,
    REPLAY_SCHEDULE_FIXED_RATE,
    REPLAY_SOURCE_INTERPRETATION_VERIFIED,
    ReplayScheduleSpec,
    ReplaySourceActionContract,
    ReplaySourceContract,
    TargetRuntimeActionContract,
)
from dataset_contracts.v23 import DEXTELEOP_DATASET_SCHEMA_V23
from dataset_interface.replay import replay_action_stage_from_contract
from robot_motion.action import ArmActionSpec, RobotAction
from robot_motion.replay.proposal_source import (
    ReplayProposalSource,
    ReplayProposalSourceConfig,
)
from teleop_runtime.action.replay_binding import build_replay_binding_assessment


def _source_contract(*, stage: str = REPLAY_ACTION_STAGE_SELECTED) -> ReplaySourceContract:
    return ReplaySourceContract(
        contract_id="dataset-a:episode-000001:action",
        replay_mode=REPLAY_MODE_CANONICAL_ACTION,
        dataset_schema=DEXTELEOP_DATASET_SCHEMA_V23,
        source_dataset_id="dataset-a",
        episode_id="episode-000001",
        action_stage=stage,
        action_field="action",
        source_action_space="bimanual_ee_pose_v1",
        source_frame_contract_hash="source-frame-hash",
        source_component_schema_hash="target-components",
        old_safety_profile_hash=("old-safety" if stage == REPLAY_ACTION_STAGE_SAFE else None),
    )


def _schedule() -> ReplayScheduleSpec:
    return ReplayScheduleSpec(
        schedule_id="dataset-a:episode-000001:15hz",
        schedule_mode=REPLAY_SCHEDULE_FIXED_RATE,
        source_hz=15.0,
        target_hz=None,
        resampling_policy="none",
        source_start_index=10,
    )


def _binding_evidence(
    *,
    verdict: str = REPLAY_BINDING_VERIFIED_COMPATIBLE,
    stage: str = REPLAY_ACTION_STAGE_SELECTED,
):
    source_action = ReplaySourceActionContract(
        contract_id="source-action-contract",
        interpretation_status=REPLAY_SOURCE_INTERPRETATION_VERIFIED,
        action_stage=stage,
        action_space="bimanual_ee_pose_v1",
        component_ids=("right_arm",),
        selection_group_by_component={"right_arm": "right_arm"},
        pose_frame_contract_hash="source-frame-hash",
        component_schema_hash="target-components",
        action_spec_hash_by_component={"right_arm": "right-arm-spec"},
        units_convention_id="si_m_rad_xyzw_v1",
    )
    target_frame = (
        "different-frame-hash"
        if verdict == REPLAY_BINDING_VERIFIED_INCOMPATIBLE
        else (
            None if verdict == REPLAY_BINDING_INDETERMINATE else "source-frame-hash"
        )
    )
    target = TargetRuntimeActionContract(
        target_contract_id="target-action-contract",
        control_domain_id="robot-0",
        embodiment_instance_id="rm65-pair-0",
        component_schema_hash="target-components",
        component_ids=("right_arm",),
        selection_group_by_component={"right_arm": "right_arm"},
        accepted_action_spaces_by_component={
            "right_arm": ("bimanual_ee_pose_v1",)
        },
        pose_frame_contract_hash=target_frame,
        action_spec_hash_by_component={"right_arm": "right-arm-spec"},
        units_convention_id="si_m_rad_xyzw_v1",
        execution_capability_profile_hash="position-capability-v1",
        unresolved_dimensions=(
            ("pose_frame_contract",) if target_frame is None else ()
        ),
    )
    binding, assessment = build_replay_binding_assessment(
        source_action,
        target,
        binding_id="dataset-a-to-rm65",
    )
    assert assessment.verdict == verdict
    return source_action, target, binding, assessment


def _action() -> RobotAction:
    arm_spec = ArmActionSpec.tool_tcp("right", reference_frame="right_arm_base")
    return RobotAction.bimanual_ee_pose(
        right_arm=np.asarray(
            [0.2, 0.1, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        timestamp=50.0,
        source="legacy_replay",
        metadata={
            "arm_action_specs_by_side": {"right": arm_spec.to_metadata()},
            "replay_frame_index": 11,
            "replay_source": "legacy_replay",
        },
    )


def _adapter(
    *,
    stage: str = REPLAY_ACTION_STAGE_SELECTED,
    verdict: str = REPLAY_BINDING_VERIFIED_COMPATIBLE,
):
    source_action, target, binding, assessment = _binding_evidence(
        verdict=verdict,
        stage=stage,
    )
    return ReplayProposalSource(
        ReplayProposalSourceConfig(
            runtime_session_id="session-0",
            control_domain_id="robot-0",
            embodiment_instance_id="rm65-pair-0",
            producer_id="replay-main",
            producer_epoch=1,
            component_schema_hash="target-components",
            component_coverage=("right_arm",),
            selection_groups=("right_arm",),
            source_contract=_source_contract(stage=stage),
            source_action_contract=source_action,
            target_runtime_contract=target,
            derivation_receipt=None,
            schedule=_schedule(),
            target_binding_spec=binding,
            binding_assessment=assessment,
        )
    )


def test_v23_root_action_has_explicit_selected_pre_safety_semantics() -> None:
    metadata = {"dexteleop_dataset_schema": DEXTELEOP_DATASET_SCHEMA_V23}
    assert replay_action_stage_from_contract(metadata, action_field="action") == (
        REPLAY_ACTION_STAGE_SELECTED
    )
    with pytest.raises(ValueError, match="only assigns"):
        replay_action_stage_from_contract(metadata, action_field="custom_action")


def test_backend_command_cannot_become_canonical_replay_proposal() -> None:
    with pytest.raises(ValueError, match="proposal, selected, or safe"):
        _source_contract(stage=REPLAY_ACTION_STAGE_COMMANDED)


def test_replay_adapter_moves_debug_metadata_to_typed_lineage() -> None:
    result = _adapter().adapt_action(
        _action(),
        source_event_id="episode-000001:frame:11",
        source_emission_id="legacy-replay-emission:11",
        source_action_index=11,
        source_timestamp=1.0 / 15.0,
        source_timestamp_clock_domain_id="capture_relative",
        sequence=0,
        produced_at=10.0,
        produced_clock_domain_id="local_monotonic",
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        effective_clock_domain_id="local_monotonic",
    )

    assert result.source_adapter_receipt.accepted
    assert result.replay_action_record is not None
    assert result.proposal_draft is not None
    assert result.proposal_draft.producer_kind == "replay"
    assert set(result.proposal_draft.action.metadata) == {"arm_action_specs_by_side"}
    assert result.proposal_draft.representation_trace_id == (
        result.source_adapter_receipt.representation_trace_id
    )
    assert result.proposal_draft.audit_metadata["replay_action_record_id"] == (
        result.replay_action_record.replay_action_record_id
    )


def test_recorded_safe_stage_never_grants_current_safety_bypass() -> None:
    result = _adapter(stage=REPLAY_ACTION_STAGE_SAFE).adapt_action(
        _action(),
        source_event_id="episode-000001:frame:11",
        source_emission_id="legacy-replay-emission:11",
        source_action_index=11,
        source_timestamp=1.0 / 15.0,
        source_timestamp_clock_domain_id="capture_relative",
        sequence=0,
        produced_at=10.0,
        produced_clock_domain_id="local_monotonic",
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        effective_clock_domain_id="local_monotonic",
    )
    assert result.source_adapter_receipt.old_safe_does_not_bypass_current_safety
    assert result.proposal_draft is not None
    assert not hasattr(result.proposal_draft, "safety_bypass")


def test_incompatible_target_binding_fails_before_proposal_admission() -> None:
    result = _adapter(verdict=REPLAY_BINDING_VERIFIED_INCOMPATIBLE).adapt_action(
        _action(),
        source_event_id="episode-000001:frame:11",
        source_emission_id="legacy-replay-emission:11",
        source_action_index=11,
        source_timestamp=1.0 / 15.0,
        source_timestamp_clock_domain_id="capture_relative",
        sequence=0,
        produced_at=10.0,
        produced_clock_domain_id="local_monotonic",
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        effective_clock_domain_id="local_monotonic",
    )
    assert not result.source_adapter_receipt.accepted
    assert result.binding_assessment.verdict == REPLAY_BINDING_VERIFIED_INCOMPATIBLE
    assert result.source_adapter_receipt.rejection_reasons == (
        "assessment_disallows:passive_shadow",
    )
    assert result.replay_action_record is not None
    assert result.proposal_draft is None


def test_indeterminate_binding_is_conditional_only_for_same_passive_emission() -> None:
    adapter = _adapter(verdict=REPLAY_BINDING_INDETERMINATE)
    common = dict(
        source_event_id="episode-000001:frame:11",
        source_action_index=11,
        source_timestamp=1.0 / 15.0,
        source_timestamp_clock_domain_id="capture_relative",
        sequence=0,
        produced_at=10.0,
        produced_clock_domain_id="local_monotonic",
        effective_from=10.0,
        effective_until=10.1,
        target_effective_at=10.0,
        effective_clock_domain_id="local_monotonic",
    )
    detached = adapter.adapt_action(
        _action(),
        source_emission_id="detached-emission",
        same_legacy_emission=False,
        **common,
    )
    assert detached.replay_action_record is not None
    assert detached.proposal_draft is None
    assert not detached.binding_eligibility.evaluation_allowed

    conditional = adapter.adapt_action(
        _action(),
        source_emission_id="same-legacy-emission",
        same_legacy_emission=True,
        **common,
    )
    assert conditional.proposal_draft is not None
    assert conditional.binding_eligibility.evaluation_allowed
    assert conditional.binding_eligibility.conditional_evaluation
    assert not conditional.binding_eligibility.authoritative_allowed

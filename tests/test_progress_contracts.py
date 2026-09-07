from __future__ import annotations

from dataclasses import replace

import pytest

from collection_runtime.progress.requests import (
    CompleteEpisodeProgressRequest,
    ProgressRequestContext,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_facts import (
    AttemptFailureEventV1,
    EpisodeProgressStartedV1,
    ProgressFactContractError,
    ProgressFactEnvelopeV1,
    ProgressOperationReceiptV1,
    progress_payload_schema_hash,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressProfileContractError,
    ProgressStepDefinitionV1,
    ProgressStepArmControlV1,
    ProgressStepArmControlV2,
    TaskProgressProfileV1,
    validate_progress_episode_binding,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _linear_profile() -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="cup_task",
        profile_version="1",
        task_id="cup",
        steps=(
            ProgressStepDefinitionV1("pick", "PICK_OBJECT", "Pick", 1),
            ProgressStepDefinitionV1("place", "PLACE_OBJECT", "Place", 2),
        ),
        initial_step_id="pick",
        terminal_step_ids=("place",),
    )


def test_linear_profile_normalizes_implicit_next_step() -> None:
    implicit = _linear_profile()
    explicit = replace(
        implicit,
        steps=(
            ProgressStepDefinitionV1(
                "pick",
                "PICK_OBJECT",
                "Pick",
                1,
                ("place",),
            ),
            ProgressStepDefinitionV1("place", "PLACE_OBJECT", "Place", 2),
        ),
    )

    assert implicit.steps[0].allowed_next_step_ids == ("place",)
    assert explicit.semantic_content_hash == implicit.semantic_content_hash
    assert (
        TaskProgressProfileV1.from_payload(implicit.to_payload())
        == implicit
    )


def test_progress_step_arm_control_is_optional_and_hash_bound() -> None:
    legacy = _linear_profile()
    assert "arm_control" not in legacy.steps[0].to_payload()
    assert TaskProgressProfileV1.from_payload(legacy.to_payload()) == legacy

    profiled = replace(
        legacy,
        steps=(
            replace(
                legacy.steps[0],
                arm_control=ProgressStepArmControlV1(
                    left="fixed_hold",
                    right="teleop",
                ),
            ),
            replace(
                legacy.steps[1],
                arm_control=ProgressStepArmControlV1(
                    left="teleop",
                    right="fixed_hold",
                ),
            ),
        ),
    )

    assert profiled.semantic_content_hash != legacy.semantic_content_hash
    assert profiled.steps[0].arm_control is not None
    assert profiled.steps[0].arm_control.fixed_sides == ("left",)
    assert profiled.steps[1].arm_control.teleop_sides == ("left",)
    assert TaskProgressProfileV1.from_payload(profiled.to_payload()) == profiled

    with pytest.raises(ProgressProfileContractError, match="arm_and_end_effector"):
        ProgressStepArmControlV1(
            left="fixed_hold",
            right="teleop",
            fixed_scope="arm_only",
        )


def test_progress_step_arm_control_v2_preserves_honest_arm_only_scope() -> None:
    control = ProgressStepArmControlV2(
        left="fixed_hold",
        right="teleop",
        fixed_scope="arm_only",
    )

    assert control.fixed_sides == ("left",)
    assert control.teleop_sides == ("right",)
    assert ProgressStepArmControlV2.from_payload(control.to_payload()) == control

    profile = replace(
        _linear_profile(),
        steps=(
            replace(_linear_profile().steps[0], arm_control=control),
            _linear_profile().steps[1],
        ),
    )
    assert TaskProgressProfileV1.from_payload(profile.to_payload()) == profile


def test_repeated_semantic_type_has_distinct_progress_identity() -> None:
    profile = TaskProgressProfileV1(
        profile_id="two_picks",
        profile_version="1",
        task_id="two_picks",
        steps=(
            ProgressStepDefinitionV1("pick_1", "PICK_OBJECT", "Pick 1", 1),
            ProgressStepDefinitionV1("pick_2", "PICK_OBJECT", "Pick 2", 2),
        ),
        initial_step_id="pick_1",
        terminal_step_ids=("pick_2",),
    )

    assert profile.steps[0].semantic_type == profile.steps[1].semantic_type
    assert profile.steps[0].progress_step_id != profile.steps[1].progress_step_id


def test_rooted_tree_requires_unique_parent_and_leaf_terminals() -> None:
    profile = TaskProgressProfileV1(
        profile_id="branch",
        profile_version="1",
        task_id="branch",
        topology_kind="ROOTED_TREE",
        steps=(
            ProgressStepDefinitionV1(
                "root",
                "START",
                "Start",
                1,
                ("right", "left"),
            ),
            ProgressStepDefinitionV1("left", "PLACE", "Left", 2),
            ProgressStepDefinitionV1("right", "PLACE", "Right", 3),
        ),
        initial_step_id="root",
        terminal_step_ids=("right", "left"),
    )

    assert profile.steps[0].allowed_next_step_ids == ("left", "right")
    assert profile.terminal_step_ids == ("left", "right")
    assert profile.ancestor_path_to("right") == ("root",)

    with pytest.raises(ProgressProfileContractError, match="branch merges"):
        TaskProgressProfileV1(
            profile_id="merge",
            profile_version="1",
            task_id="merge",
            topology_kind="ROOTED_TREE",
            steps=(
                ProgressStepDefinitionV1(
                    "root",
                    "START",
                    "Start",
                    1,
                    ("left", "right"),
                ),
                ProgressStepDefinitionV1(
                    "left",
                    "MOVE",
                    "Left",
                    2,
                    ("end",),
                ),
                ProgressStepDefinitionV1(
                    "right",
                    "MOVE",
                    "Right",
                    3,
                    ("end",),
                ),
                ProgressStepDefinitionV1("end", "END", "End", 4),
            ),
            initial_step_id="root",
            terminal_step_ids=("end",),
        )


def test_episode_binding_requires_exact_ancestor_path() -> None:
    profile = _linear_profile()
    partial = ProgressEpisodeBindingV1(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=profile.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope="PROGRESS_STEP_WINDOW",
        entry_step_id="place",
        entry_mode="ACTUAL_TRUNCATED_HISTORY",
        history_complete=True,
        assumed_completed_path_step_ids=("pick",),
        truncated_at_start=True,
    )
    validate_progress_episode_binding(profile=profile, binding=partial)

    invalid = replace(partial, assumed_completed_path_step_ids=())
    with pytest.raises(ProgressProfileContractError, match="history_complete"):
        validate_progress_episode_binding(profile=profile, binding=invalid)

    with pytest.raises(ProgressProfileContractError, match="FULL_TASK"):
        validate_progress_episode_binding(
            profile=profile,
            binding=replace(
                partial,
                recording_scope="FULL_TASK",
            ),
        )

    assert ProgressEpisodeBindingV1.from_payload(partial.to_payload()) == partial


def test_capture_failure_contract_only_accepts_operator_report_time() -> None:
    with pytest.raises(
        ProgressFactContractError,
        match="Unsupported failure timing semantics",
    ):
        AttemptFailureEventV1(
            progress_step_id="pick",
            attempt_id="attempt:pick",
            reported_at=1.0,
            timing_semantics="ESTIMATED_ONSET",
        )


def test_non_applied_operation_cannot_claim_an_effective_time() -> None:
    with pytest.raises(
        ProgressFactContractError,
        match="non-APPLIED operation cannot claim",
    ):
        ProgressOperationReceiptV1(
            operation="FORMAL_CAPTURE_PAUSE",
            status="REJECTED",
            requested_at=1.0,
            effective_at=1.1,
            clock_domain_id="episode_clock",
        )


def test_progress_fact_envelope_roundtrip_and_schema_hash_fail_closed() -> None:
    fact = ProgressFactEnvelopeV1(
        fact_id="fact:start",
        episode_capture_identity_hash="2" * 64,
        progress_sequence_index=1,
        effective_at=1.0,
        effective_clock_domain_id="episode_clock",
        available_at=1.1,
        available_clock_domain_id="receiver_clock",
        origin="collection_runtime",
        actor_id="fixture",
        payload=EpisodeProgressStartedV1(
            progress_step_id="pick",
            attempt_id="attempt:1",
        ),
    )
    assert ProgressFactEnvelopeV1.from_payload(fact.to_payload()) == fact
    assert fact.payload_schema_hash == progress_payload_schema_hash(
        fact.payload_schema_id
    )

    changed = fact.to_payload()
    changed["payload_schema_hash"] = "0" * 64
    with pytest.raises(ProgressFactContractError, match="schema hash"):
        ProgressFactEnvelopeV1.from_payload(changed)


def test_progress_contracts_reject_lossy_scalar_coercion() -> None:
    with pytest.raises(ProgressProfileContractError, match="non-empty string"):
        ProgressStepDefinitionV1(None, "PICK", "Pick", 1)  # type: ignore[arg-type]
    with pytest.raises(ProgressProfileContractError, match="positive integer"):
        ProgressStepDefinitionV1(
            "pick",
            "PICK",
            "Pick",
            1.9,  # type: ignore[arg-type]
        )
    with pytest.raises(ProgressProfileContractError, match="must be boolean"):
        replace(_linear_profile(), allow_retry_after_failure="false")  # type: ignore[arg-type]
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=_linear_profile().semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope="FULL_TASK",
        entry_step_id="pick",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    with pytest.raises(ProgressProfileContractError, match="must be boolean"):
        replace(binding, history_complete="false")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-negative integer"):
        EpisodeCaptureIdentity(
            "episode",
            1.5,  # type: ignore[arg-type]
            "fixture",
            0.0,
            "episode_clock",
        )
    with pytest.raises(ValueError, match="must be numeric"):
        EpisodeCaptureIdentity(
            "episode",
            1,
            "fixture",
            "0.0",  # type: ignore[arg-type]
            "episode_clock",
        )
    with pytest.raises(ValueError, match="must be numeric"):
        ProgressRequestContext(
            request_id="request",
            effective_at="1.0",  # type: ignore[arg-type]
            effective_clock_domain_id="episode_clock",
            available_at=1.0,
            available_clock_domain_id="episode_clock",
            actor_id="fixture",
        )
    with pytest.raises(ValueError, match="must be boolean"):
        CompleteEpisodeProgressRequest(
            completed_progress_step="false",  # type: ignore[arg-type]
        )


def test_progress_fact_envelope_rejects_lossy_numeric_coercion() -> None:
    with pytest.raises(ProgressFactContractError, match="positive integer"):
        ProgressFactEnvelopeV1(
            fact_id="fact:start",
            episode_capture_identity_hash="2" * 64,
            progress_sequence_index=1.5,  # type: ignore[arg-type]
            effective_at=1.0,
            effective_clock_domain_id="episode_clock",
            available_at=1.0,
            available_clock_domain_id="episode_clock",
            origin="collection_runtime",
            actor_id="fixture",
            payload=EpisodeProgressStartedV1(
                progress_step_id="pick",
                attempt_id="attempt:1",
            ),
        )
    with pytest.raises(ProgressFactContractError, match="must be numeric"):
        ProgressFactEnvelopeV1(
            fact_id="fact:start",
            episode_capture_identity_hash="2" * 64,
            progress_sequence_index=1,
            effective_at="1.0",  # type: ignore[arg-type]
            effective_clock_domain_id="episode_clock",
            available_at=1.0,
            available_clock_domain_id="episode_clock",
            origin="collection_runtime",
            actor_id="fixture",
            payload=EpisodeProgressStartedV1(
                progress_step_id="pick",
                attempt_id="attempt:1",
            ),
        )

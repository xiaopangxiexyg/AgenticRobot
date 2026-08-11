from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import pytest

from dataset_interface.arm_hold_cohort import (
    ArmHoldCohortDryRun,
    ArmHoldCohortError,
    ArmHoldEpisodeExclusion,
    ArmHoldEpisodeRecord,
    ArmHoldSourceReceipt,
    ArmHoldSplitResult,
    split_arm_hold_episodes,
)
from dataset_interface.arm_hold_profile import load_arm_hold_dataset_profile


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "dataset_interface"
    / "arm_hold_profiles"
    / "writing_0803_0804_0805_front_h16_v1.json"
)


def _episode(source: str, index: int, *, left_exit: bool = False) -> ArmHoldEpisodeRecord:
    common = {
        ("label", "left:teleop"): 20,
        ("label", "left:fixed_hold"): 20,
        ("label", "right:teleop"): 20,
        ("label", "right:fixed_hold"): 20,
        ("pair_state", "00"): 10,
        ("pair_state", "01"): 10,
        ("pair_state", "10"): 10,
        ("transition", "left:teleop->fixed_hold"): 1,
        ("transition", "right:teleop->fixed_hold"): 1,
        ("transition", "right:fixed_hold->teleop"): 1,
    }
    transitions = [
        "left:teleop->fixed_hold",
        "right:teleop->fixed_hold",
        "right:fixed_hold->teleop",
    ]
    if left_exit:
        common[("transition", "left:fixed_hold->teleop")] = 1
        transitions.append("left:fixed_hold->teleop")
    return ArmHoldEpisodeRecord(
        source_dataset_id=source,
        source_episode_index=index,
        episode_uuid=f"{source}-episode-{index}",
        episode_capture_identity_hash=f"{source}-identity-{index:03d}",
        frame_count=40,
        annotation_set_id=f"set-{source}-{index}",
        event_counts=common,
        observed_pair_states=("00", "01", "10"),
        transition_directions=tuple(transitions),
    )


def _cohort() -> tuple[ArmHoldEpisodeRecord, ...]:
    result = []
    for source_index, source in enumerate(("s0803", "s0804", "s0805")):
        for index in range(30):
            result.append(
                _episode(
                    source,
                    index,
                    left_exit=source_index == 2 and index < 9,
                )
            )
    return tuple(result)


def test_split_is_deterministic_order_invariant_and_has_no_leakage() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    episodes = _cohort()

    first = split_arm_hold_episodes(episodes, profile.split)
    second = split_arm_hold_episodes(tuple(reversed(episodes)), profile.split)

    assert first.to_payload() == second.to_payload()
    assert len(first.train_episode_identities) == 72
    assert len(first.validation_episode_identities) == 18
    assert not (
        set(first.train_episode_identities)
        & set(first.validation_episode_identities)
    )
    for source in ("s0803", "s0804", "s0805"):
        source_validation_count = sum(
            identity.startswith(f"{source}-")
            for identity in first.validation_episode_identities
        )
        assert source_validation_count == 6


def test_split_protects_rare_left_exit_without_copying_episodes() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    result = split_arm_hold_episodes(_cohort(), profile.split)
    event = ("transition", "left:fixed_hold->teleop")

    assert result.train_coverage[event] >= 7
    assert result.validation_coverage[event] >= 2
    assert result.train_coverage[event] + result.validation_coverage[event] == 9


def test_split_fails_when_rare_coverage_gate_is_impossible() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    split_spec = copy.deepcopy(profile.to_dict()["split"])
    for constraint in split_spec["coverage_constraints"]:
        if constraint["event_key"] == "left:fixed_hold->teleop":
            constraint["minimum_train_events"] = 8
            constraint["minimum_validation_events"] = 2

    with pytest.raises(ArmHoldCohortError, match="impossible"):
        split_arm_hold_episodes(_cohort(), split_spec)


def test_split_result_rejects_duplicate_membership() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    split = split_arm_hold_episodes(_cohort(), profile.split)

    with pytest.raises(ArmHoldCohortError, match="duplicated"):
        ArmHoldSplitResult(
            train_episode_identities=(
                *split.train_episode_identities,
                split.train_episode_identities[0],
            ),
            validation_episode_identities=split.validation_episode_identities,
            train_coverage=split.train_coverage,
            validation_coverage=split.validation_coverage,
        )


def test_dry_run_rejects_coverage_not_bound_to_membership() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    episodes = _cohort()
    split = split_arm_hold_episodes(episodes, profile.split)
    wrong_train_coverage = Counter(split.train_coverage)
    wrong_train_coverage[("label", "left:teleop")] += 1

    with pytest.raises(ArmHoldCohortError, match="coverage"):
        ArmHoldCohortDryRun(
            profile_hash=profile.semantic_content_hash,
            source_receipts=(
                ArmHoldSourceReceipt("s0803", "d0803", "hash-3", 30, 1200, 30),
                ArmHoldSourceReceipt("s0804", "d0804", "hash-4", 30, 1200, 30),
                ArmHoldSourceReceipt("s0805", "d0805", "hash-5", 30, 1200, 30),
            ),
            included_episodes=episodes,
            exclusions=(),
            split=ArmHoldSplitResult(
                train_episode_identities=split.train_episode_identities,
                validation_episode_identities=split.validation_episode_identities,
                train_coverage=wrong_train_coverage,
                validation_coverage=split.validation_coverage,
            ),
        )


def test_dry_run_rejects_unaccounted_catalog_episode() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    episodes = _cohort()
    split = split_arm_hold_episodes(episodes, profile.split)

    with pytest.raises(ArmHoldCohortError, match="every catalog Episode"):
        ArmHoldCohortDryRun(
            profile_hash=profile.semantic_content_hash,
            source_receipts=(
                ArmHoldSourceReceipt("s0803", "d0803", "hash-3", 31, 1200, 31),
                ArmHoldSourceReceipt("s0804", "d0804", "hash-4", 30, 1200, 30),
                ArmHoldSourceReceipt("s0805", "d0805", "hash-5", 30, 1200, 30),
            ),
            included_episodes=episodes,
            exclusions=(),
            split=split,
        )


def test_dry_run_debug_payload_is_observational_only() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    episodes = _cohort()
    split = split_arm_hold_episodes(episodes, profile.split)
    dry_run = ArmHoldCohortDryRun(
        profile_hash=profile.semantic_content_hash,
        source_receipts=(
            ArmHoldSourceReceipt("s0803", "d0803", "hash-3", 31, 1200, 31),
            ArmHoldSourceReceipt("s0804", "d0804", "hash-4", 30, 1200, 30),
            ArmHoldSourceReceipt("s0805", "d0805", "hash-5", 30, 1200, 30),
        ),
        included_episodes=episodes,
        exclusions=(
            ArmHoldEpisodeExclusion(
                "s0803",
                99,
                "excluded-uuid",
                "excluded-identity",
                "TERMINAL_OUTCOME_NOT_SUCCESS",
            ),
        ),
        split=split,
    )

    compact = dry_run.summary_payload(debug=False)
    debug = dry_run.summary_payload(debug=True)

    assert compact["dry_run_hash"] == debug["dry_run_hash"]
    assert "included_episodes" not in compact
    assert "exclusions" not in compact
    assert "split" not in compact
    assert len(debug["included_episodes"]) == 90
    assert debug["exclusions"][0]["reason"] == (
        "TERMINAL_OUTCOME_NOT_SUCCESS"
    )

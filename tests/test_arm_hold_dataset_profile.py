from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dataset_interface.arm_hold_profile import (
    ARM_HOLD_DATASET_PROFILE_PURPOSE_V1,
    ARM_HOLD_DATASET_PROFILE_SCHEMA_V1,
    ArmHoldDatasetProfile,
    ArmHoldDatasetProfileError,
    load_arm_hold_dataset_profile,
    resolve_arm_hold_source_roots,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "dataset_interface"
    / "arm_hold_profiles"
    / "writing_0803_0804_0805_front_h16_v1.json"
)


def _profile_payload() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def test_writing_profile_loads_as_model_neutral_source_contract() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)

    assert profile.schema == ARM_HOLD_DATASET_PROFILE_SCHEMA_V1
    assert profile.purpose == ARM_HOLD_DATASET_PROFILE_PURPOSE_V1
    assert profile.profile_id == "writing_0803_0804_0805_front_h16_v1"
    assert [source.source_dataset_id for source in profile.sources] == [
        "writing4_2026_08_03",
        "writing4_2026_08_04",
        "writing4_2026_08_05",
    ]
    assert sum(source.expected_episode_count for source in profile.sources) == 818
    assert sum(source.expected_frame_count for source in profile.sources) == 332130
    assert profile.sample_view["camera_key"] == "front"
    assert profile.sample_view["horizon"] == 16
    assert profile.sample_view["tail_policy"] == "drop"
    assert profile.sample_view["invalid_label_policy"] == (
        "drop_window_if_any_side_invalid_v1"
    )
    assert profile.episode_admission["review_policy"] == (
        "exclude_episode_review_pending_v1"
    )
    assert profile.episode_admission["annotation_selection"] == {
        "a7290543-3b08-488b-909c-cfbce1123006": (
            "stage-1785840272085-e43e8daf"
        ),
        "e9d1ed12-a5d0-43ba-a63c-c95804348e08": (
            "stage-1785898817316-c18d20dc"
        ),
        "28842dd1-bd4e-4d86-9625-c6b6682c4e2e": (
            "stage-1785907794626-8e3134b8"
        ),
    }

    assert ArmHoldDatasetProfile.from_dict(
        profile.to_dict()
    ).semantic_content_hash == profile.semantic_content_hash


def test_profile_rejects_partial_side_invalid_label_training() -> None:
    payload = _profile_payload()
    payload["sample_view"]["invalid_label_policy"] = "mask_invalid_side_v1"

    with pytest.raises(ArmHoldDatasetProfileError, match="invalid_label_policy"):
        ArmHoldDatasetProfile.from_dict(payload)


def test_writing_profile_records_unsupported_11_without_requiring_it() -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    constraints = {
        (constraint["event_kind"], constraint["event_key"]): (
            constraint["minimum_train_events"],
            constraint["minimum_validation_events"],
        )
        for constraint in profile.split["coverage_constraints"]
    }

    assert ("pair_state", "11") not in constraints
    assert constraints[("transition", "left:fixed_hold->teleop")] == (7, 2)


def test_rare_transition_is_asset_policy_not_loader_requirement() -> None:
    payload = _profile_payload()
    split = payload["split"]
    assert isinstance(split, dict)
    constraints = split["coverage_constraints"]
    assert isinstance(constraints, list)
    split["coverage_constraints"] = [
        item
        for item in constraints
        if not (
            item["event_kind"] == "transition"
            and item["event_key"] == "left:fixed_hold->teleop"
        )
    ]

    profile = ArmHoldDatasetProfile.from_dict(payload)

    assert all(
        constraint["event_key"] != "left:fixed_hold->teleop"
        for constraint in profile.split["coverage_constraints"]
    )


@pytest.mark.parametrize(
    "invalid_locator",
    [
        "/absolute/dataset",
        "../dataset",
        "nested/../dataset",
        "nested//dataset",
        "nested\\dataset",
    ],
)
def test_profile_rejects_nonportable_source_locator(invalid_locator: str) -> None:
    payload = _profile_payload()
    sources = payload["sources"]
    assert isinstance(sources, list)
    sources[0]["relative_dataset_root"] = invalid_locator

    with pytest.raises(ArmHoldDatasetProfileError, match="relative_dataset_root"):
        ArmHoldDatasetProfile.from_dict(payload)


def test_profile_rejects_unknown_fields_and_duplicate_sources() -> None:
    payload = _profile_payload()
    payload["model_config"] = "must_not_live_in_dataset_profile"
    with pytest.raises(ArmHoldDatasetProfileError, match="unknown fields"):
        ArmHoldDatasetProfile.from_dict(payload)

    payload = _profile_payload()
    sources = payload["sources"]
    assert isinstance(sources, list)
    sources.append(copy.deepcopy(sources[0]))
    with pytest.raises(ArmHoldDatasetProfileError, match="source_dataset_id"):
        ArmHoldDatasetProfile.from_dict(payload)


def test_profile_rejects_projection_contract_mismatch() -> None:
    payload = _profile_payload()
    contracts = payload["required_contracts"]
    assert isinstance(contracts, dict)
    contracts["arm_hold_projection_hash"] = "different"

    with pytest.raises(ArmHoldDatasetProfileError, match="canonical"):
        ArmHoldDatasetProfile.from_dict(payload)


def test_profile_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    profile_path = tmp_path / "duplicate.json"
    profile_path.write_text(
        '{"schema":"first","schema":"second"}', encoding="utf-8"
    )

    with pytest.raises(ArmHoldDatasetProfileError, match="duplicate key"):
        load_arm_hold_dataset_profile(profile_path)


def test_workspace_location_is_not_part_of_profile_identity(tmp_path: Path) -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    profile_hash = profile.semantic_content_hash

    first = resolve_arm_hold_source_roots(profile, tmp_path / "first")
    second = resolve_arm_hold_source_roots(profile, tmp_path / "second")

    assert profile.semantic_content_hash == profile_hash
    assert first != second
    assert set(first) == set(second)


def test_set_like_profile_fields_have_order_invariant_identity() -> None:
    payload = _profile_payload()
    original = ArmHoldDatasetProfile.from_dict(payload)
    sources = payload["sources"]
    admission = payload["episode_admission"]
    split = payload["split"]
    assert isinstance(sources, list)
    assert isinstance(admission, dict)
    assert isinstance(split, dict)
    sources.reverse()
    admission["required_modalities"].reverse()
    admission["required_columns"].reverse()
    split["stratification_keys"].reverse()
    split["coverage_constraints"].reverse()

    reordered = ArmHoldDatasetProfile.from_dict(payload)

    assert reordered.semantic_content_hash == original.semantic_content_hash


def test_compatibility_symlinks_resolve_to_canonical_sources(tmp_path: Path) -> None:
    profile = load_arm_hold_dataset_profile(PROFILE_PATH)
    canonical_root = tmp_path / "canonical"
    compatibility_root = tmp_path / "compatibility"
    canonical_root.mkdir()
    compatibility_root.mkdir()
    for source in profile.sources:
        canonical_source = canonical_root / source.relative_dataset_root
        canonical_source.mkdir()
        (compatibility_root / source.relative_dataset_root).symlink_to(
            canonical_source,
            target_is_directory=True,
        )

    canonical = resolve_arm_hold_source_roots(profile, canonical_root)
    compatibility = resolve_arm_hold_source_roots(profile, compatibility_root)

    assert canonical == compatibility
    assert profile.semantic_content_hash == load_arm_hold_dataset_profile(
        PROFILE_PATH
    ).semantic_content_hash

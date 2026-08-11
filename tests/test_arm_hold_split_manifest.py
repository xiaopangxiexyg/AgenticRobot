from __future__ import annotations

from collections import Counter
import copy
from pathlib import Path

import pytest

from dataset_contracts.canonical_encoding import canonical_bytes, canonical_loads
from dataset_interface.arm_hold_cohort import (
    ArmHoldCohortDryRun,
    ArmHoldEpisodeRecord,
    ArmHoldSourceReceipt,
    ArmHoldSplitResult,
)
from dataset_interface.arm_hold_profile import (
    ArmHoldDatasetProfile,
    load_arm_hold_dataset_profile,
)
from dataset_interface.arm_hold_split_manifest import (
    ARM_HOLD_SPLIT_MANIFEST_FILE,
    ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1,
    ArmHoldSplitManifestError,
    build_arm_hold_split_manifest,
    load_arm_hold_split_manifest,
    materialize_arm_hold_split_manifest,
)


PROFILE_PATH = (
    Path(__file__).parents[1]
    / "dataset_interface"
    / "arm_hold_profiles"
    / "writing_0803_0804_0805_front_h16_v1.json"
)


def _profile() -> ArmHoldDatasetProfile:
    payload = copy.deepcopy(load_arm_hold_dataset_profile(PROFILE_PATH).to_dict())
    for source in payload["sources"]:
        source["expected_episode_count"] = 2
        source["expected_frame_count"] = 20
    for constraint in payload["split"]["coverage_constraints"]:
        constraint["minimum_train_events"] = 1
        constraint["minimum_validation_events"] = 1
    return ArmHoldDatasetProfile.from_dict(payload)


def _events() -> dict[tuple[str, str], int]:
    return {
        ("label", "left:teleop"): 2,
        ("label", "left:fixed_hold"): 2,
        ("label", "right:teleop"): 2,
        ("label", "right:fixed_hold"): 2,
        ("pair_state", "00"): 2,
        ("pair_state", "01"): 2,
        ("pair_state", "10"): 2,
        ("transition", "left:teleop->fixed_hold"): 1,
        ("transition", "left:fixed_hold->teleop"): 1,
        ("transition", "right:teleop->fixed_hold"): 1,
        ("transition", "right:fixed_hold->teleop"): 1,
    }


def _coverage(
    episodes: tuple[ArmHoldEpisodeRecord, ...], identities: tuple[str, ...]
) -> Counter[tuple[str, str]]:
    selected = set(identities)
    result: Counter[tuple[str, str]] = Counter()
    for episode in episodes:
        if episode.episode_capture_identity_hash in selected:
            result.update(episode.event_counts)
    return result


def _cohort(profile: ArmHoldDatasetProfile) -> ArmHoldCohortDryRun:
    episodes = tuple(
        ArmHoldEpisodeRecord(
            source_dataset_id=source.source_dataset_id,
            source_episode_index=index,
            episode_uuid=f"{source.source_dataset_id}-episode-{index}",
            episode_capture_identity_hash=(
                f"{source.source_dataset_id}-identity-{index}"
            ),
            frame_count=10,
            annotation_set_id=f"annotation-{source.source_dataset_id}-{index}",
            event_counts=_events(),
            observed_pair_states=("00", "01", "10"),
            transition_directions=(
                "left:teleop->fixed_hold",
                "left:fixed_hold->teleop",
                "right:teleop->fixed_hold",
                "right:fixed_hold->teleop",
            ),
        )
        for source in profile.sources
        for index in range(2)
    )
    train = tuple(
        episode.episode_capture_identity_hash
        for episode in episodes
        if episode.source_episode_index == 0
    )
    validation = tuple(
        episode.episode_capture_identity_hash
        for episode in episodes
        if episode.source_episode_index == 1
    )
    receipts = tuple(
        ArmHoldSourceReceipt(
            source_dataset_id=source.source_dataset_id,
            dataset_name=source.expected_dataset_name,
            source_dataset_fingerprint=f"fingerprint-{source.source_dataset_id}",
            catalog_episode_count=2,
            catalog_frame_count=20,
            finalized_episode_count=2,
        )
        for source in profile.sources
    )
    return ArmHoldCohortDryRun(
        profile_hash=profile.semantic_content_hash,
        source_receipts=receipts,
        included_episodes=episodes,
        exclusions=(),
        split=ArmHoldSplitResult(
            train_episode_identities=train,
            validation_episode_identities=validation,
            train_coverage=_coverage(episodes, train),
            validation_coverage=_coverage(episodes, validation),
        ),
    )


def test_manifest_roundtrip_is_content_addressed_idempotent_and_portable(
    tmp_path: Path,
) -> None:
    profile = _profile()
    manifest = build_arm_hold_split_manifest(profile, _cohort(profile))

    first = materialize_arm_hold_split_manifest(manifest, tmp_path / "first")
    repeated = materialize_arm_hold_split_manifest(manifest, tmp_path / "first")
    second_root = materialize_arm_hold_split_manifest(manifest, tmp_path / "second")
    loaded = load_arm_hold_split_manifest(first.manifest_root)

    assert not first.idempotent
    assert repeated.idempotent
    assert not second_root.idempotent
    assert first.manifest_hash == second_root.manifest_hash
    assert first.manifest_root.name == manifest.semantic_content_hash
    assert loaded.to_payload() == manifest.to_payload()
    assert not list(first.manifest_root.parent.glob(".building-*"))
    assert first.manifest_root.stat().st_mode & 0o777 == 0o775
    assert (
        first.manifest_root / ARM_HOLD_SPLIT_MANIFEST_FILE
    ).stat().st_mode & 0o777 == 0o664


def test_manifest_rejects_embedded_profile_mismatch() -> None:
    profile = _profile()
    payload = copy.deepcopy(profile.to_dict())
    payload["profile_id"] = "different-profile"
    other = ArmHoldDatasetProfile.from_dict(payload)

    with pytest.raises(ArmHoldSplitManifestError, match="profile hash"):
        build_arm_hold_split_manifest(other, _cohort(profile))


def test_manifest_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    profile = _profile()
    manifest = build_arm_hold_split_manifest(profile, _cohort(profile))
    result = materialize_arm_hold_split_manifest(manifest, tmp_path)
    path = result.manifest_root / ARM_HOLD_SPLIT_MANIFEST_FILE
    payload = canonical_loads(
        path.read_bytes(),
        expected_domain=ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1,
    )
    payload["unknown"] = True
    path.write_bytes(
        canonical_bytes(
            payload, domain=ARM_HOLD_SPLIT_MANIFEST_STORAGE_DOMAIN_V1
        )
    )

    with pytest.raises(ArmHoldSplitManifestError, match="fields mismatch"):
        load_arm_hold_split_manifest(result.manifest_root)


def test_existing_torn_manifest_is_not_overwritten(tmp_path: Path) -> None:
    profile = _profile()
    manifest = build_arm_hold_split_manifest(profile, _cohort(profile))
    result = materialize_arm_hold_split_manifest(manifest, tmp_path)
    path = result.manifest_root / ARM_HOLD_SPLIT_MANIFEST_FILE
    path.write_bytes(b"torn")

    with pytest.raises(ArmHoldSplitManifestError, match="Cannot load"):
        materialize_arm_hold_split_manifest(manifest, tmp_path)
    assert path.read_bytes() == b"torn"


def test_directory_name_must_match_manifest_hash(tmp_path: Path) -> None:
    profile = _profile()
    manifest = build_arm_hold_split_manifest(profile, _cohort(profile))
    result = materialize_arm_hold_split_manifest(manifest, tmp_path)
    renamed = result.manifest_root.with_name("0" * 64)
    result.manifest_root.rename(renamed)

    with pytest.raises(ArmHoldSplitManifestError, match="directory name"):
        load_arm_hold_split_manifest(renamed)

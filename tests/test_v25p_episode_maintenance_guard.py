from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_contracts.canonical_encoding import canonical_bytes
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p import V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1
from collection_runtime.dataset_episode_maintenance import (
    delete_episode_and_compact,
)


def test_v25p_identity_blocks_in_place_episode_reindexing(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True)
    episodes_path.write_text(
        json.dumps({"episode_index": 0, "frame_count": 0}) + "\n",
        encoding="utf-8",
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="episode-u",
        legacy_episode_index=0,
        lifecycle_owner_id="test.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    identity_path = (
        root / "meta/v25p/episodes/episode-u/identity.cjson"
    )
    identity_path.parent.mkdir(parents=True)
    identity_path.write_bytes(
        canonical_bytes(
            identity.to_payload(),
            domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
    )
    before = episodes_path.read_bytes()

    with pytest.raises(RuntimeError, match="v2.5-p capture"):
        delete_episode_and_compact(root, 0)

    assert episodes_path.read_bytes() == before
    assert identity_path.is_file()
    assert not list(tmp_path.glob("dataset.before_delete_episode_*"))


def test_native_v25p_identity_blocks_in_place_episode_reindexing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    episodes_path = root / "meta" / "episodes.jsonl"
    episodes_path.parent.mkdir(parents=True)
    episodes_path.write_text(
        json.dumps({"episode_index": 0, "frame_count": 0}) + "\n",
        encoding="utf-8",
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="native-episode-u",
        legacy_episode_index=0,
        lifecycle_owner_id="test.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    identity_path = (
        root / "meta/v25p/native/episodes/native-episode-u/identity.cjson"
    )
    identity_path.parent.mkdir(parents=True)
    identity_path.write_bytes(
        canonical_bytes(
            identity.to_payload(),
            domain=V25P_EPISODE_IDENTITY_FILE_DOMAIN_V1,
        )
    )
    before = episodes_path.read_bytes()

    with pytest.raises(RuntimeError, match="v2.5-p capture"):
        delete_episode_and_compact(root, 0)

    assert episodes_path.read_bytes() == before
    assert identity_path.is_file()
    assert not list(tmp_path.glob("dataset.before_delete_episode_*"))

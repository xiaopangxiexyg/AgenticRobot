from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_interface.episode_exclusion import (
    DECISIONS_RELATIVE_PATH,
    SCHEMA,
    read_episode_exclusions,
    set_episode_excluded,
)


@pytest.mark.parametrize("source_present", [False, True])
def test_episode_exclusion_rejects_null_binding_for_absent_source(
    tmp_path: Path, *, source_present: bool,
) -> None:
    if source_present:
        (tmp_path / "meta").mkdir()
        (tmp_path / "meta/episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "frame_count": 4}) + chr(10), encoding="utf-8",
        )
    path = tmp_path / DECISIONS_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": SCHEMA, "episode_index": 999, "excluded": True,
        "episode_manifest_hash": None,
    }) + chr(10), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid episode exclusion"):
        read_episode_exclusions(tmp_path)


def test_episode_exclusion_restoration_and_source_removal_keep_identity_contract(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    source = tmp_path / "meta/episodes.jsonl"
    source.write_text(json.dumps({"episode_index": 0, "frame_count": 4}) + chr(10), encoding="utf-8")
    set_episode_excluded(tmp_path, 0, excluded=True)
    assert read_episode_exclusions(tmp_path) == {0: True}
    set_episode_excluded(tmp_path, 0, excluded=False)
    assert read_episode_exclusions(tmp_path) == {0: False}
    source.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid episode exclusion"):
        read_episode_exclusions(tmp_path)
    with pytest.raises(ValueError, match="not finalized"):
        set_episode_excluded(tmp_path, 999, excluded=True)

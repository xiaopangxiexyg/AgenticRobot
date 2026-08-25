from __future__ import annotations

import json

from experiments.sim_learning.state_bc.contracts import (
    STATE_BC_LAYOUT_HASH_V1,
    StateBCSelector,
)
from experiments.sim_learning.state_bc.dataset import select_formal_collection


def test_formal_collection_selection_is_explicit_and_retains_exclusions(tmp_path):
    source = tmp_path / "2026-08-07_part01"
    (source / "meta").mkdir(parents=True)
    (source / "episode_000.parquet").touch()
    (source / "meta" / "info.json").write_text(
        json.dumps({"policy_io_layout_hash": STATE_BC_LAYOUT_HASH_V1}), encoding="utf-8"
    )
    (source / "meta" / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "episode_id": "episode-000",
                        "episode_index": 0,
                        "frame_count": 5,
                        "parquet_path": "episode_000.parquet",
                        "summary": {"collection_integrity_valid": True},
                    }
                ),
                json.dumps(
                    {
                        "episode_id": "episode-001",
                        "episode_index": 1,
                        "frame_count": 1,
                        "parquet_path": "missing.parquet",
                        "summary": {"collection_integrity_valid": True},
                    }
                ),
                json.dumps(
                    {
                        "episode_id": "episode-002",
                        "episode_index": 2,
                        "frame_count": 5,
                        "parquet_path": "episode_000.parquet",
                        "summary": {"collection_integrity_valid": False},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    selection = select_formal_collection(
        StateBCSelector(selector_id="fixture-day", source_dirs=(str(source),), min_episode_frames=2)
    )

    assert [episode.episode_id for episode in selection.selected_episodes] == ["episode-000"]
    assert selection.selected_episodes[0].qualified_episode_id.endswith("::episode-000")
    assert selection.excluded_episodes
    assert any(reason == "episode_too_short" for reason in selection.excluded_episodes.values())
    assert any(reason == "collection_integrity_invalid" for reason in selection.excluded_episodes.values())

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_interface.logical_dataset import (
    LOGICAL_EPISODE_REF_SCHEMA_V1,
    load_logical_episode_selection,
)


def _record(root: Path, index: int, identity: str) -> dict[str, object]:
    return {
        "schema": LOGICAL_EPISODE_REF_SCHEMA_V1,
        "source_dataset_root": str(root),
        "source_episode_index": index,
        "episode_identity": identity,
        "camera_slots": ["base_0_rgb"],
        "view_profile": "front_only",
        "quality_class": "success",
    }


def test_logical_manifest_filters_one_source_without_renumbering(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "a"
    source_b = tmp_path / "b"
    source_a.mkdir()
    source_b.mkdir()
    manifest = tmp_path / "episodes.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                _record(source_a, 7, "sha256:a"),
                _record(source_b, 3, "sha256:b"),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    selected = load_logical_episode_selection(
        manifest,
        source_dataset_root=source_a,
        camera_slots=("base_0_rgb",),
    )

    assert selected.episode_indices == frozenset((7,))
    assert selected.episode_identity_by_index == {7: "sha256:a"}
    assert selected.manifest_sha256.startswith("sha256:")


def test_logical_manifest_rejects_camera_contract_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "episodes.jsonl"
    manifest.write_text(
        json.dumps(_record(source, 0, "sha256:a")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="camera_slots"):
        load_logical_episode_selection(
            manifest,
            source_dataset_root=source,
            camera_slots=("left_wrist_0_rgb", "right_wrist_0_rgb"),
        )

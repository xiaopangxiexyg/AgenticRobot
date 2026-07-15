from __future__ import annotations

import json
from pathlib import Path

from utils.dataset_postprocess.relocate_mask_metadata import rewrite_metadata_tree


def test_rewrite_metadata_tree_updates_json_and_jsonl_without_touching_other_text(tmp_path: Path) -> None:
    root = tmp_path / "masks"
    root.mkdir()
    old = "/old/masks"
    new = "/new/masks"
    json_path = root / "manifest.json"
    json_path.write_text(
        json.dumps({"source": {"root": f"{old}/table"}, "unchanged": "/oldish/masks"}),
        encoding="utf-8",
    )
    jsonl_path = root / "summary.jsonl"
    jsonl_path.write_text(json.dumps({"path": f"{old}/final/episode_000000"}) + "\n", encoding="utf-8")

    dry_run = rewrite_metadata_tree(root, [(old, new)], apply=False)
    assert dry_run == {"metadata_file_count": 2, "changed_file_count": 2, "replacement_count": 2}
    assert old in json_path.read_text(encoding="utf-8")

    applied = rewrite_metadata_tree(root, [(old, new)], apply=True)
    assert applied == dry_run
    assert json.loads(json_path.read_text(encoding="utf-8"))["source"]["root"] == f"{new}/table"
    assert json.loads(jsonl_path.read_text(encoding="utf-8"))["path"] == f"{new}/final/episode_000000"


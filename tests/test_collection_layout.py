from __future__ import annotations

import time
from pathlib import Path

from teleop_runtime.collection import collection_layout, run_collect


def test_run_collect_preserves_layout_compatibility_aliases() -> None:
    assert (
        run_collect._sanitize_task_name_for_path
        is collection_layout.sanitize_task_name_for_path
    )
    assert run_collect._default_task_id_from_name is collection_layout.default_task_id_from_name
    assert run_collect._resolve_collection_layout is collection_layout.resolve_collection_layout
    assert run_collect._apply_collection_layout is collection_layout.apply_collection_layout


def test_task_name_sanitization_preserves_legacy_path_shape() -> None:
    assert collection_layout.sanitize_task_name_for_path("") == "task"
    assert collection_layout.sanitize_task_name_for_path(" Pick / Place ") == "Pick_-_Place"
    assert collection_layout.sanitize_task_name_for_path("...___") == "task"
    assert collection_layout.default_task_id_from_name("Pick / Place") == "pick_-_place"
    assert collection_layout.default_task_id_from_name("***") == "task"


def test_resolve_layout_uses_explicit_values_and_date_prefix(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "dataset_root": str(tmp_path),
            "task": "",
            "task_name": "Pick Blocks",
            "task_id": "pick-blocks-v2",
            "task_description": "Pick blocks with the right hand",
            "dataset_name": "pick_blocks_v2",
        },
    )()

    dataset_root, task_name, task_id, description, dataset_name = (
        collection_layout.resolve_collection_layout(args)
    )

    expected_dir = f"{time.strftime('%Y-%m-%d')}_Pick_Blocks"
    assert dataset_root == tmp_path / expected_dir
    assert task_name == "Pick Blocks"
    assert task_id == "pick-blocks-v2"
    assert description == "Pick blocks with the right hand"
    assert dataset_name == "pick_blocks_v2"


def test_legacy_task_fills_missing_metadata_and_apply_mutates_namespace(tmp_path: Path) -> None:
    args = type(
        "Args",
        (),
        {
            "dataset_root": str(tmp_path),
            "task": "Legacy Push",
            "task_name": "",
            "task_id": "",
            "task_description": "",
            "dataset_name": "",
        },
    )()

    collection_layout.apply_collection_layout(args)

    expected_dir = f"{time.strftime('%Y-%m-%d')}_Legacy_Push"
    assert args.dataset_base_root == str(tmp_path)
    assert args.dataset_root == str(tmp_path / expected_dir)
    assert args.dataset_dir_name == expected_dir
    assert args.task_name == "Legacy Push"
    assert args.task_id == "Legacy Push"
    assert args.task == "Legacy Push"
    assert args.task_description == "Legacy Push"
    assert args.dataset_name == expected_dir

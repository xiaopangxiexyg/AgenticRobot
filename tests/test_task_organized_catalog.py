from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.dataset_postprocess.task_organized_catalog import (
    MoveSpec,
    _inventory,
)


@pytest.mark.integration
def test_inventory_is_stable_across_same_filesystem_directory_rename(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "payload.bin").write_bytes(b"unchanged")
    before = _inventory(source)
    target = tmp_path / "organized/source"
    target.parent.mkdir()

    os.rename(source, target)
    os.symlink(str(target), str(source), target_is_directory=True)

    assert source.resolve() == target.resolve()
    assert _inventory(target) == before


def test_move_spec_keeps_dataset_basename() -> None:
    spec = MoveSpec(
        key="example",
        source_relative="old/dataset",
        target_relative="organized/category/dataset",
        classification="category",
    )

    assert Path(spec.source_relative).name == Path(spec.target_relative).name

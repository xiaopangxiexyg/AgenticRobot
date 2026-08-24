from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.probes import recompress_verified_tactile_batch as batch


def _stub_plan(*_args: object, **_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(stream_count=4, raw_bytes=128, manifest_count=1)


def test_build_plan_can_freeze_one_eligible_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    roots = [Path("/datasets/a"), Path("/datasets/b"), Path("/datasets/c")]
    monkeypatch.setattr(batch, "_dataset_roots", lambda _root: roots)
    monkeypatch.setattr(batch, "plan_tactile_zstd_recompression", _stub_plan)
    monkeypatch.setattr(batch, "_safety_counts", lambda _root: (0, 0))

    plan = batch._build_plan(Path("/datasets"), entry_indices=(1,))

    assert plan["available_dataset_count"] == 3
    assert plan["selected_entry_indices"] == [1]
    assert plan["dataset_count"] == 1
    assert plan["datasets"][0]["dataset_root"] == "/datasets/b"


@pytest.mark.parametrize("entry_indices", [(-1,), (3,), (1, 1)])
def test_build_plan_rejects_invalid_entry_selection(
    monkeypatch: pytest.MonkeyPatch,
    entry_indices: tuple[int, ...],
) -> None:
    roots = [Path("/datasets/a"), Path("/datasets/b"), Path("/datasets/c")]
    monkeypatch.setattr(batch, "_dataset_roots", lambda _root: roots)
    monkeypatch.setattr(batch, "plan_tactile_zstd_recompression", _stub_plan)
    monkeypatch.setattr(batch, "_safety_counts", lambda _root: (0, 0))

    with pytest.raises(ValueError):
        batch._build_plan(Path("/datasets"), entry_indices=entry_indices)


def test_unfiltered_plan_keeps_all_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    roots = [Path("/datasets/a"), Path("/datasets/b")]
    monkeypatch.setattr(batch, "_dataset_roots", lambda _root: roots)
    monkeypatch.setattr(batch, "plan_tactile_zstd_recompression", _stub_plan)
    monkeypatch.setattr(batch, "_safety_counts", lambda _root: (0, 0))

    plan = batch._build_plan(Path("/datasets"))

    assert plan["available_dataset_count"] == 2
    assert plan["selected_entry_indices"] == []
    assert plan["dataset_count"] == 2

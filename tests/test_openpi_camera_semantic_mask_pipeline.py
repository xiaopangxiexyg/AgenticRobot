from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dataset_interface import pipeline


def _recording_index_factory(calls: list[dict[str, Any]]):
    def factory(dataset_root: Path, **kwargs: Any) -> object:
        calls.append(
            {
                "dataset_root": dataset_root,
                "kwargs": kwargs,
            }
        )
        return object()

    return factory


def test_mask_pipeline_rejects_camera_prepared_view_without_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_calls: list[dict[str, Any]] = []
    validation_calls: list[tuple[Path, bool]] = []
    monkeypatch.delenv(
        pipeline.CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV,
        raising=False,
    )
    monkeypatch.setattr(
        pipeline,
        "is_camera_semantic_prepared_view",
        lambda _root: True,
    )
    monkeypatch.setattr(
        pipeline,
        "DexTeleopDatasetIndex",
        _recording_index_factory(index_calls),
    )
    monkeypatch.setattr(
        pipeline,
        "load_camera_semantic_prepared_view",
        lambda root, *, verify_payload_hashes: validation_calls.append(
            (root, verify_payload_hashes)
        ),
    )

    with pytest.raises(ValueError, match="require explicit MaskedFramePipeline"):
        pipeline._masked_pipeline_dataset_index(
            tmp_path,
            allow_camera_semantic_prepared_view=None,
        )

    assert index_calls == []
    assert validation_calls == []


@pytest.mark.parametrize("opt_in_source", ["argument", "environment"])
def test_mask_pipeline_validates_opted_in_camera_prepared_view_before_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    opt_in_source: str,
) -> None:
    index_calls: list[dict[str, Any]] = []
    validation_calls: list[tuple[Path, bool]] = []
    monkeypatch.setattr(
        pipeline,
        "is_camera_semantic_prepared_view",
        lambda _root: True,
    )
    monkeypatch.setattr(
        pipeline,
        "DexTeleopDatasetIndex",
        _recording_index_factory(index_calls),
    )
    monkeypatch.setattr(
        pipeline,
        "load_camera_semantic_prepared_view",
        lambda root, *, verify_payload_hashes: validation_calls.append(
            (root, verify_payload_hashes)
        ),
    )
    if opt_in_source == "environment":
        monkeypatch.setenv(
            pipeline.CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV,
            "1",
        )
        argument = None
    else:
        monkeypatch.delenv(
            pipeline.CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV,
            raising=False,
        )
        argument = True

    result = pipeline._masked_pipeline_dataset_index(
        tmp_path,
        allow_camera_semantic_prepared_view=argument,
    )

    resolved = tmp_path.resolve()
    assert result is not None
    assert validation_calls == [(resolved, False)]
    assert index_calls == [
        {
            "dataset_root": resolved,
            "kwargs": {"parse_capture_contract": False},
        }
    ]


def test_mask_pipeline_keeps_full_capture_contract_for_regular_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index_calls: list[dict[str, Any]] = []
    monkeypatch.setenv(
        pipeline.CAMERA_SEMANTIC_PREPARED_VIEW_OPT_IN_ENV,
        "1",
    )
    monkeypatch.setattr(
        pipeline,
        "is_camera_semantic_prepared_view",
        lambda _root: False,
    )
    monkeypatch.setattr(
        pipeline,
        "DexTeleopDatasetIndex",
        _recording_index_factory(index_calls),
    )
    monkeypatch.setattr(
        pipeline,
        "load_camera_semantic_prepared_view",
        lambda *_args, **_kwargs: pytest.fail(
            "regular datasets must not use prepared-view validation"
        ),
    )

    pipeline._masked_pipeline_dataset_index(
        tmp_path,
        allow_camera_semantic_prepared_view=None,
    )

    assert index_calls == [
        {
            "dataset_root": tmp_path.resolve(),
            "kwargs": {},
        }
    ]

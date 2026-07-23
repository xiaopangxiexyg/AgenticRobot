from __future__ import annotations

import json

import pytest

from dataset_contracts.capture_modalities import CaptureModalityContract
from dataset_interface.capture_modalities import (
    CaptureModalityBindingError,
    bind_capture_modality_contract,
    dataset_is_blank,
    read_capture_modality_contract,
)
from dataset_interface.metadata import DexTeleopDatasetIndex


def _populate(root) -> None:
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "frame_count": 1}) + "\n",
        encoding="utf-8",
    )
    (meta / "info.json").write_text(
        json.dumps({"episode_count": 1, "frame_count": 1}),
        encoding="utf-8",
    )


def test_blank_dataset_can_bind_and_replace_capture_contract(tmp_path) -> None:
    root = tmp_path / "dataset"
    first = CaptureModalityContract(ignored_camera_keys=("right_wrist",))
    second = CaptureModalityContract(ignored_tactile_keys=("left",))

    bind_capture_modality_contract(root, first)
    assert dataset_is_blank(root)
    bind_capture_modality_contract(root, second)

    assert read_capture_modality_contract(root) == second


def test_populated_dataset_requires_exact_capture_contract(tmp_path) -> None:
    root = tmp_path / "dataset"
    expected = CaptureModalityContract(
        ignored_camera_keys=("right_wrist",),
        ignored_tactile_keys=("left",),
    )
    bind_capture_modality_contract(root, expected)
    _populate(root)

    bind_capture_modality_contract(root, expected)
    with pytest.raises(CaptureModalityBindingError, match="different modality ignore"):
        bind_capture_modality_contract(root, CaptureModalityContract())


def test_populated_legacy_dataset_means_no_modalities_ignored(tmp_path) -> None:
    root = tmp_path / "dataset"
    _populate(root)

    bind_capture_modality_contract(root, CaptureModalityContract())
    with pytest.raises(CaptureModalityBindingError):
        bind_capture_modality_contract(
            root,
            CaptureModalityContract(ignored_camera_keys=("left_wrist",)),
        )


def test_dataset_index_exposes_dataset_level_ignore_declaration(tmp_path) -> None:
    root = tmp_path / "dataset"
    contract = CaptureModalityContract(
        ignored_camera_keys=("right_wrist", "left_wrist"),
        ignored_tactile_keys=("right",),
    )
    bind_capture_modality_contract(root, contract)
    meta = root / "meta"
    (meta / "episodes.jsonl").touch()
    (meta / "info.json").write_text("{}", encoding="utf-8")

    index = DexTeleopDatasetIndex(root)
    assert index.ignored_camera_keys == ("left_wrist", "right_wrist")
    assert index.ignored_tactile_keys == ("right",)


def test_capture_contract_is_only_dataset_metadata_not_episode_metadata(tmp_path) -> None:
    root = tmp_path / "dataset"
    contract = CaptureModalityContract(ignored_camera_keys=("right_wrist",))
    bind_capture_modality_contract(root, contract)
    _populate(root)

    episode = json.loads(
        (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").strip()
    )
    assert "capture_modality_contract" not in episode
    assert read_capture_modality_contract(root) == contract


def test_capture_contract_rejects_unknown_missing_value_semantics() -> None:
    payload = CaptureModalityContract().to_dict()
    payload["missing_value_policy"]["camera"] = "black_frame"

    with pytest.raises(ValueError, match="missing_value_policy"):
        CaptureModalityContract.from_dict(payload)

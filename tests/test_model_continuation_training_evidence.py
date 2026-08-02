from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_contracts import (
    ModelContinuationCheckpointReceipt,
    ModelContinuationTrainingRunManifest,
    ModelContinuationTrainingSourceBinding,
)
from dataset_interface import model_continuation_training_source_binding


def _digest(character: str) -> str:
    return f"sha256:{character * 64}"


def _source() -> ModelContinuationTrainingSourceBinding:
    return ModelContinuationTrainingSourceBinding(
        source_id="fixture",
        dataset_schema="dexteleop_dataset_v2.3",
        camera_keys=("front", "left_wrist"),
        dataset_identity_hash=_digest("1"),
        episode_manifest_hash=_digest("2"),
        media_index_hash=_digest("3"),
    )


def test_training_run_and_checkpoint_receipts_round_trip_strictly() -> None:
    run = ModelContinuationTrainingRunManifest(
        output_model_id="fixture-trained-rtc",
        policy_artifact_template_hash=_digest("4"),
        base_checkpoint_identity_hash=_digest("5"),
        sampling_spec_hash=_digest("6"),
        trainer_implementation_id="fixture-trainer-v1",
        trainer_parameters_hash=_digest("7"),
        train_config_hash=_digest("8"),
        source_bindings=(_source(),),
    )
    restored = ModelContinuationTrainingRunManifest.from_dict(run.to_dict())
    assert restored == run
    assert restored.semantic_content_hash == run.semantic_content_hash

    receipt = ModelContinuationCheckpointReceipt(
        training_run_manifest_hash=run.semantic_content_hash,
        policy_artifact_hash=_digest("9"),
        model_id=run.output_model_id,
        checkpoint_step=120,
    )
    assert (
        ModelContinuationCheckpointReceipt.from_dict(receipt.to_dict())
        == receipt
    )
    tampered = receipt.to_dict()
    tampered["checkpoint_step"] = 121
    with pytest.raises(ValueError, match="semantic_content_hash"):
        ModelContinuationCheckpointReceipt.from_dict(tampered)


def test_training_run_does_not_silently_drop_invalid_source_entries() -> None:
    run = ModelContinuationTrainingRunManifest(
        output_model_id="fixture-trained-rtc",
        policy_artifact_template_hash=_digest("4"),
        base_checkpoint_identity_hash=_digest("5"),
        sampling_spec_hash=_digest("6"),
        trainer_implementation_id="fixture-trainer-v1",
        trainer_parameters_hash=_digest("7"),
        train_config_hash=_digest("8"),
        source_bindings=(_source(),),
    )
    payload = run.to_dict()
    payload["source_bindings"] = [run.source_bindings[0].to_dict(), "bad"]
    with pytest.raises(ValueError, match="Every source_bindings entry"):
        ModelContinuationTrainingRunManifest.from_dict(payload)


def _write_dataset(root: Path) -> None:
    metadata = root / "meta"
    metadata.mkdir(parents=True)
    (metadata / "info.json").write_text(
        json.dumps(
            {
                "dexteleop_dataset_schema": "dexteleop_dataset_v2.3",
                "dataset_name": "fixture-source",
            }
        ),
        encoding="utf-8",
    )
    records = (
        {
            "episode_index": 0,
            "frame_count": 10,
            "video_paths": {
                "front": "videos/front/episode_000000.mp4",
                "left_wrist": "videos/left/episode_000000.mp4",
            },
        },
        {
            "episode_index": 1,
            "frame_count": 12,
            "video_paths": {
                "front": "videos/front/episode_000001.mp4",
                "left_wrist": "videos/left/episode_000001.mp4",
            },
        },
    )
    (metadata / "episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_source_binding_is_stable_and_sensitive_to_selected_media_index(
    tmp_path: Path,
) -> None:
    _write_dataset(tmp_path)
    first = model_continuation_training_source_binding(
        tmp_path,
        camera_keys=("front",),
    )
    second = model_continuation_training_source_binding(
        tmp_path,
        camera_keys=("front",),
    )
    wrist = model_continuation_training_source_binding(
        tmp_path,
        camera_keys=("left_wrist",),
    )
    assert first == second
    assert first.episode_manifest_hash == wrist.episode_manifest_hash
    assert first.media_index_hash != wrist.media_index_hash

    episodes = tmp_path / "meta" / "episodes.jsonl"
    text = episodes.read_text(encoding="utf-8")
    episodes.write_text(
        text.replace("episode_000001.mp4", "episode_000099.mp4", 1),
        encoding="utf-8",
    )
    changed = model_continuation_training_source_binding(
        tmp_path,
        camera_keys=("front",),
    )
    assert changed.episode_manifest_hash != first.episode_manifest_hash
    assert changed.media_index_hash != first.media_index_hash


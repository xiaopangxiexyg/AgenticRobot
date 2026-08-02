"""Read-only source-lineage projection for trained model-RTC runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.model_continuation_training import (
    ModelContinuationTrainingSourceBinding,
)

from .episode_curation import source_dataset_base_fingerprint


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset metadata is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Dataset metadata must be an object: {path}")
    return payload


def _read_episode_records(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Episode manifest is missing: {path}")
    records: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = line.strip()
        if not text:
            continue
        payload = json.loads(text)
        if not isinstance(payload, Mapping):
            raise ValueError(
                f"Episode manifest row must be an object: {path}:{line_number}"
            )
        try:
            episode_index = int(payload["episode_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Episode manifest row has no integer episode_index: "
                f"{path}:{line_number}"
            ) from exc
        if episode_index in seen:
            raise ValueError(
                f"Duplicate episode_index={episode_index} in {path}."
            )
        seen.add(episode_index)
        records.append(payload)
    if not records:
        raise ValueError(f"Episode manifest is empty: {path}")
    return tuple(records)


def _media_index_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    camera_keys: tuple[str, ...],
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    for record in records:
        episode_index = int(record["episode_index"])
        video_paths = record.get("video_paths")
        if not isinstance(video_paths, Mapping):
            raise ValueError(
                f"episode_index={episode_index} has no video_paths mapping."
            )
        selected: dict[str, str] = {}
        for camera_key in camera_keys:
            value = video_paths.get(camera_key)
            path = str(value or "").strip()
            if not path:
                raise ValueError(
                    f"episode_index={episode_index} has no video path for "
                    f"{camera_key!r}."
                )
            selected[camera_key] = path
        episodes.append(
            {
                "episode_index": episode_index,
                "frame_count": int(record.get("frame_count", 0) or 0),
                "video_paths": selected,
            }
        )
    return {
        "schema": "dexteleop_model_continuation_media_index_v1",
        "camera_keys": list(camera_keys),
        "episodes": episodes,
    }


def model_continuation_training_source_binding(
    dataset_root: str | Path,
    *,
    camera_keys: Sequence[str],
    source_id: str | None = None,
) -> ModelContinuationTrainingSourceBinding:
    """Bind immutable dataset metadata and the selected camera media index.

    The media hash binds the episode-to-video mapping, frame counts and logical
    camera selection.  It deliberately does not hash multi-gigabyte video
    contents; content-addressed media can later be supplied by a stronger
    dataset manifest without changing this training-run receipt.
    """

    root = Path(dataset_root).expanduser().resolve()
    info_path = root / "meta" / "info.json"
    episodes_path = root / "meta" / "episodes.jsonl"
    info = _read_json(info_path)
    records = _read_episode_records(episodes_path)
    selected_cameras = tuple(
        dict.fromkeys(str(value).strip() for value in camera_keys)
    )
    if not selected_cameras or any(not value for value in selected_cameras):
        raise ValueError("camera_keys must contain non-empty logical views.")
    dataset_schema = str(info.get("dexteleop_dataset_schema") or "").strip()
    if not dataset_schema:
        raise ValueError("Dataset metadata has no dexteleop_dataset_schema.")
    resolved_source_id = str(
        source_id
        or info.get("dataset_name")
        or info.get("dataset_dir_name")
        or root.name
    ).strip()
    if not resolved_source_id:
        raise ValueError("source_id cannot be resolved.")
    media_hash = canonical_content_hash(
        _media_index_payload(records, camera_keys=selected_cameras),
        domain="dexteleop.model_continuation_media_index.v1",
    )
    return ModelContinuationTrainingSourceBinding(
        source_id=resolved_source_id,
        dataset_schema=dataset_schema,
        camera_keys=selected_cameras,
        dataset_identity_hash=(
            f"sha256:{source_dataset_base_fingerprint(root)}"
        ),
        episode_manifest_hash=_sha256_bytes(episodes_path.read_bytes()),
        media_index_hash=media_hash,
    )


__all__ = ["model_continuation_training_source_binding"]

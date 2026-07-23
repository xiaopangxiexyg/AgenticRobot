"""Read-only access to the optional Plus episode curation sidecar.

The reader is intentionally tolerant of a missing sidecar: all existing v2.3
and v2.4 datasets remain valid and simply have no Plus curation record.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dataset_contracts.episode_curation import (
    EpisodeCurationContractError,
    EpisodeCurationRecord,
    PLUS_ANNOTATION_SET_SCHEMA_V1,
    PLUS_ANNOTATION_SET_STATUSES,
    annotation_episode_id,
)


SIDECAR_FILENAME = "episode_curation.jsonl"
MANIFEST_FILENAME = "manifest.json"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def source_dataset_base_fingerprint(dataset_root: str | Path) -> str:
    """Hash immutable source manifests while excluding all Plus directories.

    The function only reads ``meta`` files.  It never traverses the dataset,
    and specifically does not include ``meta/annotations`` or any external
    annotation root in the fingerprint.
    """

    root = Path(dataset_root).expanduser().resolve()
    relative_files = (
        "meta/info.json",
        "meta/tasks.jsonl",
        "meta/modality_validity.json",
    )
    entries: list[dict[str, str]] = []
    for relative in relative_files:
        path = root / relative
        if path.is_file():
            entries.append({"path": relative, "sha256": _sha256_bytes(path.read_bytes())})
        else:
            entries.append({"path": relative, "sha256": "missing"})
    return _sha256_bytes(_json_bytes({"schema": "dexteleop_plus_source_fingerprint_v1", "files": entries}))


def source_episode_manifest_hash(
    dataset_root: str | Path,
    episode_index: int,
) -> str | None:
    """Hash one legacy ``episodes.jsonl`` record, or return ``None`` if absent."""

    path = Path(dataset_root).expanduser().resolve() / "meta" / "episodes.jsonl"
    if not path.is_file():
        return None
    selected: dict[str, Any] | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON in {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"episode manifest row must be an object in {path}:{line_number}")
        try:
            row_index = int(value.get("episode_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"episode_index missing in {path}:{line_number}") from exc
        if row_index == int(episode_index):
            if selected is not None:
                raise ValueError(f"duplicate episode_index={episode_index} in {path}")
            selected = value
    if selected is None:
        return None
    return _sha256_bytes(_json_bytes(selected))


class EpisodeCurationReader:
    """Load an append-only overlay without requiring it to exist."""

    def __init__(self, annotation_root: str | Path):
        self.root = Path(annotation_root).expanduser().resolve()
        self.manifest_path = self.root / MANIFEST_FILENAME
        self.sidecar_path = self.root / SIDECAR_FILENAME
        self.manifest: dict[str, Any] = self._read_manifest()
        self._records_by_id = self._read_records()

    def _read_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.is_file():
            return {}
        value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"annotation manifest must be an object: {self.manifest_path}")
        if value.get("schema_version") != PLUS_ANNOTATION_SET_SCHEMA_V1:
            raise ValueError(f"unsupported Plus annotation manifest: {self.manifest_path}")
        status = str(value.get("status", "draft"))
        if status not in PLUS_ANNOTATION_SET_STATUSES:
            raise ValueError(f"invalid Plus annotation manifest status: {status!r}")
        return value

    def _read_records(self) -> dict[str, EpisodeCurationRecord]:
        if not self.sidecar_path.is_file():
            return {}
        records: dict[str, EpisodeCurationRecord] = {}
        for line_number, line in enumerate(self.sidecar_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = EpisodeCurationRecord.from_dict(json.loads(text))
            except (TypeError, ValueError, json.JSONDecodeError, EpisodeCurationContractError) as exc:
                raise ValueError(f"invalid Plus curation record {self.sidecar_path}:{line_number}") from exc
            previous = records.get(record.annotation_episode_id)
            if previous is not None and previous.to_dict() != record.to_dict():
                raise ValueError(
                    f"conflicting Plus curation revisions for {record.annotation_episode_id}"
                )
            if self.manifest:
                manifest_set_id = str(self.manifest.get("annotation_set_id", "") or "")
                if manifest_set_id and record.annotation_set_id != manifest_set_id:
                    raise ValueError("Plus record annotation_set_id does not match its manifest")
                manifest_fingerprint = str(
                    self.manifest.get("source_dataset_base_fingerprint", "") or ""
                )
                if manifest_fingerprint and record.source_dataset_base_fingerprint != manifest_fingerprint:
                    raise ValueError("Plus record source fingerprint does not match its manifest")
                manifest_binding = str(self.manifest.get("source_binding", "") or "")
                if manifest_binding and record.source_binding != manifest_binding:
                    raise ValueError("Plus record source binding does not match its manifest")
                task_profile = self.manifest.get("task_profile")
                manifest_profile_id = (
                    str(task_profile.get("task_profile_id", "") or "")
                    if isinstance(task_profile, dict)
                    else ""
                )
                if manifest_profile_id and record.task_profile_id != manifest_profile_id:
                    raise ValueError("Plus record task profile does not match its manifest")
            records[record.annotation_episode_id] = record
        return records

    @property
    def available(self) -> bool:
        return bool(self._records_by_id)

    @property
    def status(self) -> str:
        return str(self.manifest.get("status", "missing")) if self.manifest else "missing"

    @property
    def usable_for_training(self) -> bool:
        return self.status in {"complete", "partial"}

    def by_annotation_episode_id(self, value: str) -> EpisodeCurationRecord | None:
        return self._records_by_id.get(str(value).strip())

    def for_episode(
        self,
        dataset_root: str | Path,
        episode_index: int,
    ) -> EpisodeCurationRecord | None:
        base_fingerprint = source_dataset_base_fingerprint(dataset_root)
        manifest_hash = source_episode_manifest_hash(dataset_root, int(episode_index))
        identity = annotation_episode_id(
            source_dataset_base_fingerprint=base_fingerprint,
            source_episode_manifest_hash=manifest_hash,
            legacy_episode_index=int(episode_index),
        )
        return self.by_annotation_episode_id(identity)


__all__ = [
    "EpisodeCurationReader",
    "MANIFEST_FILENAME",
    "SIDECAR_FILENAME",
    "source_dataset_base_fingerprint",
    "source_episode_manifest_hash",
]

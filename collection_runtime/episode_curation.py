"""Device-, control-, and UI-neutral Plus episode-curation persistence.

The application frontend decides when an already finalized source episode is
ready for review.  This module only owns the append-only external sidecar,
manifest publication, and immutable source binding used to persist that
review.  It never owns the raw Episode lifecycle or operator/UI state.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from dataset_contracts.episode_curation import (
    EpisodeCurationRecord,
    PLUS_ANNOTATION_SET_SCHEMA_V1,
    PLUS_ANNOTATION_SET_STATUSES,
    PLUS_SOURCE_BINDINGS,
    TaskVariantProfile,
    annotation_episode_id,
    annotation_set_id,
)
from dataset_interface.episode_curation import (
    MANIFEST_FILENAME,
    SIDECAR_FILENAME,
    SOURCE_DATASET_BASE_FINGERPRINT_V2,
    source_dataset_base_fingerprint,
    source_episode_manifest_hash,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_json_dump(payload))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
        _fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def load_task_variant_profile(path: str | Path | None) -> TaskVariantProfile | None:
    """Load a finite profile; ``None`` means the application shows no variants."""

    if path is None or not str(path).strip():
        return None
    profile_path = Path(path).expanduser().resolve()
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    return TaskVariantProfile.from_dict(payload)


@dataclass
class EpisodeCurationStore:
    """Append-only sidecar bound to one immutable source dataset."""

    annotation_root: Path
    source_dataset_root: Path
    source_binding: str = "v23_compat_content_locator_v1"
    task_profile: TaskVariantProfile | None = None
    source_schema: str = "dexteleop_dataset_unknown"

    def __post_init__(self) -> None:
        self.annotation_root = Path(self.annotation_root).expanduser().resolve()
        self.source_dataset_root = Path(self.source_dataset_root).expanduser().resolve()
        if self.source_binding not in PLUS_SOURCE_BINDINGS:
            raise ValueError(f"unsupported Plus source binding: {self.source_binding!r}")
        self.base_fingerprint = source_dataset_base_fingerprint(
            self.source_dataset_root
        )
        self.base_fingerprint_schema = SOURCE_DATASET_BASE_FINGERPRINT_V2
        self.episode_sidecar_path = self.annotation_root / SIDECAR_FILENAME
        self.manifest_path = self.annotation_root / MANIFEST_FILENAME
        if self.manifest_path.is_file():
            try:
                existing_manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"invalid Plus annotation manifest: {self.manifest_path}"
                ) from exc
            if isinstance(existing_manifest, Mapping):
                existing_schema = str(
                    existing_manifest.get(
                        "source_dataset_base_fingerprint_schema", ""
                    )
                    or ""
                )
                existing_fingerprint = str(
                    existing_manifest.get("source_dataset_base_fingerprint", "")
                    or ""
                )
                if not existing_schema and existing_fingerprint:
                    # Read/write compatibility for an already-created
                    # provisional v1 set. Its manifest owns the original
                    # lineage token; do not silently rewrite it after the
                    # dataset's mutable counters advance.
                    self.base_fingerprint = existing_fingerprint
                    self.base_fingerprint_schema = (
                        "dexteleop_plus_source_fingerprint_v1"
                    )
        profile_id = (
            None if self.task_profile is None else self.task_profile.task_profile_id
        )
        self.annotation_set_id = annotation_set_id(
            source_dataset_base_fingerprint=self.base_fingerprint,
            source_binding=self.source_binding,
            task_profile_id=profile_id,
        )

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLUS_ANNOTATION_SET_SCHEMA_V1,
            "annotation_set_id": self.annotation_set_id,
            "status": "draft",
            "source_dataset_base_fingerprint": self.base_fingerprint,
            "source_dataset_base_fingerprint_schema": self.base_fingerprint_schema,
            "source_dataset_schema": self.source_schema,
            "source_binding": self.source_binding,
            "task_profile": (
                None if self.task_profile is None else self.task_profile.to_dict()
            ),
            "created_at_utc": _utc_now(),
        }

    def _ensure_manifest(self) -> dict[str, Any]:
        self.annotation_root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.is_file():
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("Plus annotation manifest must be an object")
            if value.get("schema_version") != PLUS_ANNOTATION_SET_SCHEMA_V1:
                raise ValueError("Plus annotation manifest schema mismatch")
            if value.get("annotation_set_id") != self.annotation_set_id:
                raise ValueError(
                    "Plus annotation manifest is bound to another source/profile"
                )
            if value.get("source_dataset_base_fingerprint") != self.base_fingerprint:
                raise ValueError("Plus annotation manifest source fingerprint mismatch")
            manifest_fingerprint_schema = str(
                value.get("source_dataset_base_fingerprint_schema", "") or ""
            )
            if manifest_fingerprint_schema and (
                manifest_fingerprint_schema != self.base_fingerprint_schema
            ):
                raise ValueError("Plus annotation manifest fingerprint schema mismatch")
            if value.get("source_binding") != self.source_binding:
                raise ValueError("Plus annotation manifest source binding mismatch")
            if value.get("source_dataset_schema") != self.source_schema:
                raise ValueError("Plus annotation manifest source schema mismatch")
            profile_payload = value.get("task_profile")
            profile_id = (
                None
                if self.task_profile is None
                else self.task_profile.task_profile_id
            )
            manifest_profile_id = (
                profile_payload.get("task_profile_id")
                if isinstance(profile_payload, dict)
                else None
            )
            if manifest_profile_id != profile_id:
                raise ValueError("Plus annotation manifest task profile mismatch")
            status = str(value.get("status", "draft"))
            if status not in PLUS_ANNOTATION_SET_STATUSES:
                raise ValueError(
                    f"invalid Plus annotation manifest status: {status!r}"
                )
            if status in {"invalid", "superseded"}:
                raise ValueError(
                    f"cannot append to Plus annotation set with status={status!r}"
                )
            return value
        payload = self._manifest_payload()
        _atomic_write_json(self.manifest_path, payload)
        return payload

    def _existing_records(self) -> dict[str, EpisodeCurationRecord]:
        if not self.episode_sidecar_path.is_file():
            return {}
        records: dict[str, EpisodeCurationRecord] = {}
        for line_number, line in enumerate(
            self.episode_sidecar_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                record = EpisodeCurationRecord.from_dict(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - reframe malformed sidecar.
                raise ValueError(
                    f"invalid Plus sidecar record at line {line_number}"
                ) from exc
            previous = records.get(record.annotation_episode_id)
            if previous is not None and previous.to_dict() != record.to_dict():
                raise ValueError(
                    "conflicting Plus sidecar records for "
                    f"{record.annotation_episode_id}"
                )
            records[record.annotation_episode_id] = record
        return records

    def append(self, record: EpisodeCurationRecord) -> EpisodeCurationRecord:
        if record.annotation_set_id and record.annotation_set_id != self.annotation_set_id:
            raise ValueError("record annotation_set_id does not match this sidecar")
        if record.source_dataset_base_fingerprint != self.base_fingerprint:
            raise ValueError("record source fingerprint does not match this sidecar")
        manifest = self._ensure_manifest()
        if str(manifest.get("status", "draft")) not in {"draft", "partial"}:
            raise ValueError(
                "only draft or partial Plus annotation sets may receive new records"
            )
        records = self._existing_records()
        previous = records.get(record.annotation_episode_id)
        if previous is not None:
            if previous.to_dict() == record.to_dict():
                return previous
            raise ValueError(
                "episode already has a conflicting Plus curation record: "
                f"{record.annotation_episode_id}"
            )
        if not record.annotation_set_id:
            record = EpisodeCurationRecord.from_dict(
                {**record.to_dict(), "annotation_set_id": self.annotation_set_id}
            )
        self.episode_sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        line = _json_dump(record.to_dict()) + "\n"
        with self.episode_sidecar_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.episode_sidecar_path.parent)
        del manifest
        return record

    def set_status(self, status: str) -> dict[str, Any]:
        """Publish a draft/partial set without rewriting episode records."""

        value = str(status or "").strip()
        if value not in PLUS_ANNOTATION_SET_STATUSES:
            raise ValueError(f"invalid Plus annotation set status: {value!r}")
        manifest = self._ensure_manifest()
        current = str(manifest.get("status", "draft"))
        if current in {"invalid", "superseded"} and value != current:
            raise ValueError(
                f"cannot transition Plus annotation set from {current!r}"
            )
        manifest = dict(manifest)
        manifest["status"] = value
        manifest["updated_at_utc"] = _utc_now()
        _atomic_write_json(self.manifest_path, manifest)
        return manifest

    def append_for_episode(
        self,
        *,
        episode_index: int,
        disposition: str,
        reason_code: str = "unknown",
        variant_id: str | None = None,
        review_status: str = "confirmed",
    ) -> EpisodeCurationRecord:
        manifest_hash = source_episode_manifest_hash(
            self.source_dataset_root, int(episode_index)
        )
        if manifest_hash is None:
            raise ValueError(
                "cannot curate missing source episode manifest record: "
                f"episode_index={int(episode_index)}"
            )
        identity = annotation_episode_id(
            source_dataset_base_fingerprint=self.base_fingerprint,
            source_episode_manifest_hash=manifest_hash,
            legacy_episode_index=int(episode_index),
        )
        profile_id = (
            None if self.task_profile is None else self.task_profile.task_profile_id
        )
        if self.task_profile is not None and variant_id:
            variant_id = self.task_profile.require_variant(variant_id)
        elif variant_id:
            raise ValueError("variant_id cannot be written without a task profile")
        record = EpisodeCurationRecord(
            annotation_episode_id=identity,
            annotation_set_id=self.annotation_set_id,
            legacy_episode_index=int(episode_index),
            source_dataset_base_fingerprint=self.base_fingerprint,
            source_episode_manifest_hash=manifest_hash,
            source_schema=self.source_schema,
            source_binding=self.source_binding,
            disposition=disposition,
            reason_code=reason_code,
            task_profile_id=profile_id,
            variant_id=variant_id,
            review_status=review_status,
        )
        return self.append(record)


__all__ = ["EpisodeCurationStore", "load_task_variant_profile"]

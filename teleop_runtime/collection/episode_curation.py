"""Default-off writer for the optional Plus episode curation sidecar.

This module is intentionally outside the raw collection writer.  It performs
small, post-finalize JSON writes only after an operator explicitly confirms a
curation value in the GUI.  A failure here is reported to the GUI and never
changes whether the original episode is saved or discarded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping
import uuid

from dataset_contracts.episode_curation import (
    EpisodeCurationRecord,
    PLUS_ANNOTATION_SET_SCHEMA_V1,
    PLUS_ANNOTATION_SET_STATUSES,
    PLUS_DISPOSITIONS,
    PLUS_REASON_CODES,
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
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))


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
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
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
    """Load a finite profile; ``None`` means the GUI does not show variants."""

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
        self.base_fingerprint = source_dataset_base_fingerprint(self.source_dataset_root)
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
                    # provisional v1 set.  Its manifest owns the original
                    # lineage token; do not silently rewrite it after the
                    # dataset's mutable counters advance.
                    self.base_fingerprint = existing_fingerprint
                    self.base_fingerprint_schema = "dexteleop_plus_source_fingerprint_v1"
        profile_id = None if self.task_profile is None else self.task_profile.task_profile_id
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
            "task_profile": None if self.task_profile is None else self.task_profile.to_dict(),
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
                raise ValueError("Plus annotation manifest is bound to another source/profile")
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
                raise ValueError(f"invalid Plus annotation manifest status: {status!r}")
            if status in {"invalid", "superseded"}:
                raise ValueError(f"cannot append to Plus annotation set with status={status!r}")
            return value
        payload = self._manifest_payload()
        _atomic_write_json(self.manifest_path, payload)
        return payload

    def _existing_records(self) -> dict[str, EpisodeCurationRecord]:
        if not self.episode_sidecar_path.is_file():
            return {}
        records: dict[str, EpisodeCurationRecord] = {}
        for line_number, line in enumerate(self.episode_sidecar_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = EpisodeCurationRecord.from_dict(json.loads(line))
            except Exception as exc:  # noqa: BLE001 - reframe malformed sidecar context.
                raise ValueError(f"invalid Plus sidecar record at line {line_number}") from exc
            previous = records.get(record.annotation_episode_id)
            if previous is not None and previous.to_dict() != record.to_dict():
                raise ValueError(f"conflicting Plus sidecar records for {record.annotation_episode_id}")
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
            raise ValueError(f"episode already has a conflicting Plus curation record: {record.annotation_episode_id}")
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
            raise ValueError(f"cannot transition Plus annotation set from {current!r}")
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
        manifest_hash = source_episode_manifest_hash(self.source_dataset_root, int(episode_index))
        if manifest_hash is None:
            raise ValueError(
                f"cannot curate missing source episode manifest record: episode_index={int(episode_index)}"
            )
        identity = annotation_episode_id(
            source_dataset_base_fingerprint=self.base_fingerprint,
            source_episode_manifest_hash=manifest_hash,
            legacy_episode_index=int(episode_index),
        )
        profile_id = None if self.task_profile is None else self.task_profile.task_profile_id
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


@dataclass
class EpisodeCurationSession:
    """Small runtime state shared by RM65 and UMI GUI adapters."""

    enabled: bool = False
    annotation_root: Path | None = None
    task_profile: TaskVariantProfile | None = None
    source_binding: str = "v23_compat_content_locator_v1"
    source_schema: str = "dexteleop_dataset_unknown"
    variant_id: str = ""
    disposition: str = "unknown"
    reason_code: str = "unknown"
    pending_episode_index: int | None = None
    pending_dataset_root: Path | None = None
    pending_token: str = ""
    status: str = "off"
    message: str = ""
    _persisted_binding_lock_message: str = field(
        default="",
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.enabled = bool(self.enabled)
        if self.annotation_root is not None:
            self.annotation_root = Path(self.annotation_root).expanduser().resolve()
        if not self.enabled:
            self.status = "off"
            return
        if self.annotation_root is None:
            return
        sidecar_path = self.annotation_root / SIDECAR_FILENAME
        try:
            has_records = bool(
                sidecar_path.is_file()
                and any(
                    line.strip()
                    for line in sidecar_path.read_text(encoding="utf-8").splitlines()
                )
            )
        except OSError as exc:
            self._persisted_binding_lock_message = (
                "无法确认 Plus 外部整理绑定，已禁止原地切换/重编号："
                f"{type(exc).__name__}: {exc}"
            )
        else:
            if has_records:
                self._persisted_binding_lock_message = (
                    "Plus 外部整理记录已绑定当前数据集的 episode 身份；"
                    "本次运行禁止切换数据集或原地删除/重编号。"
                )

    @classmethod
    def from_args(cls, args: Any, *, source_binding: str, source_schema: str = "") -> "EpisodeCurationSession":
        enabled = bool(getattr(args, "plus_episode_curation", False))
        annotation_root = getattr(args, "plus_annotation_root", "")
        profile_path = getattr(args, "plus_task_profile", "")
        profile = load_task_variant_profile(profile_path) if enabled and profile_path else None
        if enabled and not str(annotation_root or "").strip():
            raise ValueError("--plus-annotation-root is required when Plus episode curation is enabled")
        return cls(
            enabled=enabled,
            annotation_root=None if not annotation_root else Path(annotation_root),
            task_profile=profile,
            source_binding=source_binding,
            source_schema=source_schema or "dexteleop_dataset_unknown",
            status="ready" if enabled else "off",
        )

    def ui_state(self) -> dict[str, Any]:
        return {
            "plus_curation_enabled": bool(self.enabled),
            "plus_curation_task_profile_id": (
                "" if self.task_profile is None else self.task_profile.task_profile_id
            ),
            "plus_curation_variants": (
                [] if self.task_profile is None else [item.to_dict() for item in self.task_profile.variants]
            ),
            "plus_curation_variant_id": self.variant_id,
            "plus_curation_disposition": self.disposition,
            "plus_curation_reason_code": self.reason_code,
            "plus_curation_pending": self.pending_episode_index is not None,
            "plus_curation_episode_index": self.pending_episode_index,
            "plus_curation_pending_token": self.pending_token,
            "plus_curation_pending_dataset_root": (
                "" if self.pending_dataset_root is None else str(self.pending_dataset_root)
            ),
            "plus_curation_status": self.status,
            "plus_curation_message": self.message,
        }

    def persisted_binding_lock_reason(self) -> str:
        """Return a conservative blocker for in-place source reconfiguration.

        A persisted Plus record contains the source episode index and manifest
        hash.  Deleting/reindexing the source dataset, or rebinding this live
        session to another source root, would make that immutable external
        locator ambiguous.  The raw dataset remains independently valid, but
        the GUI must not offer those mutations once this annotation set has
        received at least one durable record.
        """

        return str(self._persisted_binding_lock_message or "")

    def set_variant(self, variant_id: str | None) -> None:
        value = str(variant_id or "").strip()
        if not value:
            self.variant_id = ""
            return
        if self.task_profile is None:
            raise ValueError("Plus variant selection requires a task profile")
        self.variant_id = self.task_profile.require_variant(value)

    def set_review(self, *, disposition: str, reason_code: str = "unknown") -> None:
        """Validate and retain the current GUI review draft for this episode."""

        disposition_value = str(disposition or "unknown").strip()
        reason_value = str(reason_code or "unknown").strip()
        if disposition_value not in PLUS_DISPOSITIONS:
            raise ValueError(f"unsupported Plus disposition: {disposition_value!r}")
        if reason_value not in PLUS_REASON_CODES:
            raise ValueError(f"unsupported Plus reason_code: {reason_value!r}")
        if disposition_value in {"keep_nominal", "unknown"} and reason_value != "unknown":
            raise ValueError(
                "Plus reason_code must be unknown for keep_nominal or unknown disposition"
            )
        self.disposition = disposition_value
        self.reason_code = reason_value

    def begin_episode(self, *, episode_index: int, variant_id: str | None = None) -> None:
        if not self.enabled:
            return
        had_pending = self.pending_episode_index is not None
        save_in_flight = self.status == "saving"
        if variant_id is not None:
            self.set_variant(variant_id)
        self.disposition = "unknown"
        self.reason_code = "unknown"
        self.pending_episode_index = None
        self.pending_dataset_root = None
        self.pending_token = ""
        self.status = "recording"
        self.message = (
            "上一个 Plus episode 正在后台写入；本 episode 已开始。"
            if had_pending and save_in_flight
            else "上一个 Plus episode 未提交，已跳过本次整理；本 episode 已开始。"
            if had_pending
            else ""
        )

    def mark_pending(self, *, dataset_root: str | Path, episode_index: int) -> None:
        if not self.enabled:
            return
        self.pending_dataset_root = Path(dataset_root).expanduser().resolve()
        self.pending_episode_index = int(episode_index)
        self.pending_token = uuid.uuid4().hex
        self.disposition = "unknown"
        self.reason_code = "unknown"
        self.status = "pending"
        self.message = "episode 已保存；请选择 Plus 整理标签并提交。"

    def clear_pending(self, *, message: str = "") -> None:
        self.pending_episode_index = None
        self.pending_dataset_root = None
        self.pending_token = ""
        self.disposition = "unknown"
        self.reason_code = "unknown"
        if self.enabled:
            self.status = "ready"
            self.message = str(message or "")

    def matches_pending(self, *, episode_index: int, pending_token: str) -> bool:
        return bool(
            self.enabled
            and self.pending_episode_index == int(episode_index)
            and self.pending_token
            and str(pending_token or "") == self.pending_token
        )

    def save(
        self,
        *,
        disposition: str,
        reason_code: str = "unknown",
        pending_token: str | None = None,
        variant_id: str | None = None,
        source_episode_index: int | None = None,
        source_dataset_root: str | Path | None = None,
    ) -> EpisodeCurationRecord:
        if not self.enabled:
            raise RuntimeError("Plus episode curation is disabled")
        has_source_snapshot = source_episode_index is not None or source_dataset_root is not None
        if has_source_snapshot and (source_episode_index is None or source_dataset_root is None):
            raise RuntimeError("Plus source snapshot requires episode index and dataset root")
        if not has_source_snapshot:
            if self.pending_episode_index is None or self.pending_dataset_root is None:
                raise RuntimeError("no finalized episode is waiting for Plus curation")
            if pending_token is not None and str(pending_token) != self.pending_token:
                raise RuntimeError("Plus pending episode token is stale")
        assert self.annotation_root is not None
        # Snapshot the finalized source before the external write. The
        # operator may start the next raw episode while this small sidecar
        # write is running; that must not make the annotation attach to the
        # new episode or overwrite its GUI state.
        pending_episode_index = (
            int(self.pending_episode_index)
            if source_episode_index is None
            else int(source_episode_index)
        )
        pending_dataset_root = (
            Path(self.pending_dataset_root)
            if source_dataset_root is None
            else Path(source_dataset_root).expanduser().resolve()
        )
        pending_token_value = str(pending_token or self.pending_token)
        pending_variant_id = (
            self.variant_id if variant_id is None else str(variant_id).strip()
        ) or None
        source_schema = self.source_schema
        info_path = pending_dataset_root / "meta" / "info.json"
        if info_path.is_file():
            try:
                info_payload = json.loads(info_path.read_text(encoding="utf-8"))
                if isinstance(info_payload, Mapping):
                    source_schema = str(
                        info_payload.get("dexteleop_dataset_schema")
                        or info_payload.get("dexteleop.dataset_schema")
                        or info_payload.get("schema")
                        or source_schema
                    )
            except (OSError, ValueError, TypeError):
                # The sidecar is optional; preserve the configured fallback
                # rather than turning a source metadata quirk into a control
                # or episode-save failure.
                pass
        store = EpisodeCurationStore(
            annotation_root=self.annotation_root,
            source_dataset_root=pending_dataset_root,
            source_binding=self.source_binding,
            task_profile=self.task_profile,
            source_schema=source_schema,
        )
        record = store.append_for_episode(
            episode_index=pending_episode_index,
            disposition=disposition,
            reason_code=reason_code,
            variant_id=pending_variant_id,
        )
        self._persisted_binding_lock_message = (
            "Plus 外部整理记录已绑定当前数据集的 episode 身份；"
            "本次运行禁止切换数据集或原地删除/重编号。"
        )
        # Do not clobber a newer episode's state if the operator started it
        # before the sidecar thread completed.
        if (
            self.pending_episode_index == pending_episode_index
            and self.pending_dataset_root == pending_dataset_root
            and self.pending_token == pending_token_value
        ):
            self.status = "saved"
            self.message = f"Plus episode {record.legacy_episode_index} 已保存。"
            self.pending_episode_index = None
            self.pending_dataset_root = None
            self.pending_token = ""
            self.disposition = "unknown"
            self.reason_code = "unknown"
        return record


__all__ = [
    "EpisodeCurationSession",
    "EpisodeCurationStore",
    "load_task_variant_profile",
]

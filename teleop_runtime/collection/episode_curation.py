"""Application session for optional Plus episode curation.

The device-, control-, and UI-neutral sidecar store is owned by
``collection_runtime.episode_curation``.  This module retains the RM65/UMI
application session and forwards the historical store imports for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping
import uuid

from collection_runtime.episode_curation import (
    EpisodeCurationStore,
    load_task_variant_profile,
)
from dataset_contracts.episode_curation import (
    EpisodeCurationRecord,
    PLUS_DISPOSITIONS,
    PLUS_REASON_CODES,
    TaskVariantProfile,
)
from dataset_interface.episode_curation import SIDECAR_FILENAME


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

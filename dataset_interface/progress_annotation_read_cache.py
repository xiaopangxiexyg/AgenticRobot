"""Local read-through cache for immutable progress-annotation inputs.

The cache never writes into a dataset.  Native finalized catalogs and sample
axes are keyed by their sealed identities so a slow SSHFS/NFS mount is not
re-read on every annotation-server restart.  Cache corruption is treated as a
miss and the authoritative dataset remains the only fallback source.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

from dataset_contracts.progress_facts import (
    ProgressEpisodeManifestV1,
    ProgressFactEnvelopeV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_progress_stage_entry_binding,
)
from dataset_interface.progress_annotation_jobs import (
    EpisodeSampleContinuity,
    ProgressAnnotationCatalogBatch,
    ProgressAnnotationCatalogCursor,
    ProgressAnnotationWorkItem,
    RecordingPauseSegment,
)
from dataset_interface.progress_artifacts import SealedProgressEpisode
from dataset_interface.progress_timeline import resolve_progress_timeline
from dataset_interface.sample_axis import EpisodeSampleAxis


_CATALOG_CACHE_SCHEMA = "dexteleop.refinement_annotation.catalog_cache.v2"
_SAMPLE_AXIS_CACHE_SCHEMA = (
    "dexteleop.refinement_annotation.sample_continuity_cache.v2"
)


def _default_cache_root() -> Path:
    configured = os.environ.get("DEXTELEOP_ANNOTATION_READ_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    cache_base = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = (
        Path(cache_base).expanduser().resolve()
        if cache_base
        else Path.home() / ".cache"
    )
    return base / "dexteleop" / "refinement-annotation-read"


def _json_fingerprint(value: tuple[object, ...]) -> list[object]:
    return [
        _json_fingerprint(item) if isinstance(item, tuple) else item
        for item in value
    ]


def _relative_path(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root).as_posix()


def _cached_dataset_path(root: Path, value: object) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Cached dataset path is empty.")
    relative = Path(text)
    if relative.is_absolute() or any(part == ".." for part in relative.parts):
        raise ValueError("Cached dataset path is not a safe relative path.")
    # Do not resolve every cached Episode path here. On SSHFS/NFS that would
    # turn a local warm-cache load back into hundreds of remote metadata
    # round trips. Exact source paths are resolved and root-confined when the
    # operator opens that Episode.
    path = Path(os.path.abspath(root / relative))
    path.relative_to(root)
    return path


def _cursor_payload(cursor: ProgressAnnotationCatalogCursor) -> dict[str, Any]:
    return {
        "device": cursor.device,
        "inode": cursor.inode,
        "offset": cursor.offset,
        "line_number": cursor.line_number,
        "mtime_ns": cursor.mtime_ns,
        "prefix_probe_offset": cursor.prefix_probe_offset,
        "prefix_probe_hex": cursor.prefix_probe.hex(),
    }


def _cursor_from_payload(payload: Mapping[str, Any]) -> ProgressAnnotationCatalogCursor:
    return ProgressAnnotationCatalogCursor(
        device=int(payload["device"]),
        inode=int(payload["inode"]),
        offset=int(payload["offset"]),
        line_number=int(payload["line_number"]),
        mtime_ns=int(payload["mtime_ns"]),
        prefix_probe_offset=int(payload["prefix_probe_offset"]),
        prefix_probe=bytes.fromhex(str(payload["prefix_probe_hex"])),
    )


def _work_item_payload(root: Path, item: ProgressAnnotationWorkItem) -> dict[str, Any]:
    progress = item.progress
    if not isinstance(progress, SealedProgressEpisode):
        raise TypeError("Only native sealed progress Episodes are cacheable.")
    return {
        "episode_index": item.episode_index,
        "episode_uuid": item.episode_uuid,
        "episode_identity_hash": item.episode_identity_hash,
        "progress_manifest_hash": item.progress_manifest_hash,
        "progress_episode_dir": _relative_path(root, item.progress_episode_dir),
        "parquet_path": _relative_path(root, item.parquet_path),
        "video_paths": {
            key: _relative_path(root, value)
            for key, value in item.video_paths.items()
        },
        "frame_count": item.frame_count,
        "start_timestamp": item.start_timestamp,
        "end_timestamp": item.end_timestamp,
        "recording_pause_segments": [
            value.to_payload() for value in item.recording_pause_segments
        ],
        "profile": progress.profile.to_payload(),
        "binding": progress.binding.to_payload(),
        "facts": [value.to_payload() for value in progress.facts],
        "manifest": progress.manifest.to_payload(),
        "stage_profile": (
            None
            if progress.stage_profile is None
            else progress.stage_profile.to_payload()
        ),
        "stage_entry_binding": (
            None
            if progress.stage_entry_binding is None
            else progress.stage_entry_binding.to_payload()
        ),
    }


def _work_item_from_payload(
    root: Path,
    payload: Mapping[str, Any],
) -> ProgressAnnotationWorkItem:
    profile = TaskProgressProfileV1.from_payload(payload["profile"])
    binding = ProgressEpisodeBindingV1.from_payload(payload["binding"])
    facts = tuple(
        ProgressFactEnvelopeV1.from_payload(value)
        for value in payload["facts"]
    )
    manifest = ProgressEpisodeManifestV1.from_payload(payload["manifest"])
    timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=facts,
        manifest=manifest,
        require_sealed=True,
    )
    stage_profile_payload = payload.get("stage_profile")
    stage_entry_payload = payload.get("stage_entry_binding")
    stage_profile = (
        None
        if stage_profile_payload is None
        else TaskStageProfileV1.from_payload(stage_profile_payload)
    )
    stage_entry = (
        None
        if stage_entry_payload is None
        else ProgressStageEntryBindingV1.from_payload(stage_entry_payload)
    )
    if (stage_profile is None) != (stage_entry is None):
        raise ValueError("Cached fine-stage artifacts are incomplete.")
    if stage_profile is not None and stage_entry is not None:
        validate_progress_stage_entry_binding(
            progress_profile=profile,
            progress_entry_step_id=binding.entry_step_id,
            stage_profile=stage_profile,
            stage_binding=stage_entry,
        )
    episode_uuid = str(payload["episode_uuid"])
    episode_identity_hash = str(payload["episode_identity_hash"])
    progress_manifest_hash = str(payload["progress_manifest_hash"])
    if (
        episode_identity_hash != binding.episode_capture_identity_hash
        or progress_manifest_hash != manifest.semantic_content_hash
        or stage_profile is None
        or stage_entry is None
    ):
        raise ValueError("Cached work item does not match its sealed identities.")
    progress_dir = _cached_dataset_path(root, payload["progress_episode_dir"])
    return ProgressAnnotationWorkItem(
        dataset_root=root,
        episode_index=int(payload["episode_index"]),
        episode_uuid=episode_uuid,
        episode_identity_hash=episode_identity_hash,
        progress_manifest_hash=progress_manifest_hash,
        progress_episode_dir=progress_dir,
        parquet_path=_cached_dataset_path(root, payload["parquet_path"]),
        video_paths={
            str(key): _cached_dataset_path(root, value)
            for key, value in payload["video_paths"].items()
        },
        frame_count=int(payload["frame_count"]),
        start_timestamp=float(payload["start_timestamp"]),
        end_timestamp=float(payload["end_timestamp"]),
        progress=SealedProgressEpisode(
            episode_dir=progress_dir,
            profile=profile,
            binding=binding,
            facts=facts,
            manifest=manifest,
            timeline=timeline,
            stage_profile=stage_profile,
            stage_entry_binding=stage_entry,
        ),
        # Annotation state remains live and is intentionally rediscovered by
        # visible-row summaries or exact Episode opening.
        annotation_set_count=0,
        has_draft=False,
        recording_pause_segments=tuple(
            RecordingPauseSegment(
                start_timestamp=float(value["start_timestamp"]),
                end_timestamp=float(value["end_timestamp"]),
                reason=str(value.get("reason", "")),
                reasons=tuple(str(reason) for reason in value.get("reasons", ())),
            )
            for value in payload.get("recording_pause_segments", ())
            if isinstance(value, Mapping)
        ),
    )


class ProgressAnnotationReadCache:
    """Best-effort local cache; every failure safely degrades to a miss."""

    def __init__(self, cache_root: str | Path | None = None) -> None:
        self.cache_root = (
            _default_cache_root()
            if cache_root is None
            else Path(cache_root).expanduser().resolve()
        )
        self._available = True
        try:
            self.cache_root.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._available = False

    @staticmethod
    def _dataset_key(dataset_root: Path) -> str:
        return hashlib.sha256(str(dataset_root).encode("utf-8")).hexdigest()

    def _catalog_path(self, dataset_root: Path) -> Path:
        return self.cache_root / f"catalog-{self._dataset_key(dataset_root)}.json"

    @staticmethod
    def _sample_axis_key(item: ProgressAnnotationWorkItem) -> str:
        identity = "\0".join(
            (
                _SAMPLE_AXIS_CACHE_SCHEMA,
                item.episode_identity_hash,
                item.progress_manifest_hash,
                str(item.frame_count),
                str(item.parquet_path),
            )
        )
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _sample_axis_path(self, item: ProgressAnnotationWorkItem) -> Path:
        return self.cache_root / f"sample-axis-{self._sample_axis_key(item)}.json"

    @staticmethod
    def _read_json(path: Path) -> Mapping[str, Any]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError("Cache payload is not an object.")
        return value

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        if not self._available:
            return
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_root,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        except OSError:
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def load_catalog(
        self,
        dataset_root: Path,
        catalog_fingerprint: tuple[object, ...],
    ) -> ProgressAnnotationCatalogBatch | None:
        batch = self.load_catalog_prefix(dataset_root)
        if (
            batch is None
            or batch.catalog_fingerprint != catalog_fingerprint
        ):
            return None
        return batch

    @staticmethod
    def _cursor_matches_catalog(
        dataset_root: Path,
        cursor: ProgressAnnotationCatalogCursor,
    ) -> bool:
        path = dataset_root / "meta" / "episodes.jsonl"
        try:
            with path.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                if (
                    stat.st_dev != cursor.device
                    or stat.st_ino != cursor.inode
                    or stat.st_size < cursor.offset
                ):
                    return False
                if (
                    stat.st_size == cursor.offset
                    and stat.st_mtime_ns != cursor.mtime_ns
                ):
                    return False
                handle.seek(cursor.prefix_probe_offset)
                return handle.read(len(cursor.prefix_probe)) == cursor.prefix_probe
        except OSError:
            return False

    def load_catalog_prefix(
        self,
        dataset_root: Path,
    ) -> ProgressAnnotationCatalogBatch | None:
        """Load a validated append-only prefix, even after new Episodes arrive."""

        if not self._available:
            return None
        root = dataset_root.expanduser().resolve()
        try:
            payload = self._read_json(self._catalog_path(root))
            if (
                payload.get("schema") != _CATALOG_CACHE_SCHEMA
                or payload.get("dataset_root") != str(root)
            ):
                return None
            cursor_payload = payload.get("cursor")
            if not isinstance(cursor_payload, Mapping):
                return None
            cursor = _cursor_from_payload(cursor_payload)
            if not self._cursor_matches_catalog(root, cursor):
                return None
            items_payload = payload.get("items")
            if not isinstance(items_payload, list):
                return None
            items = tuple(
                _work_item_from_payload(root, value)
                for value in items_payload
                if isinstance(value, Mapping)
            )
            if len(items) != len(items_payload):
                return None
            return ProgressAnnotationCatalogBatch(
                items=tuple(sorted(items, key=lambda item: item.episode_index)),
                cursor=cursor,
                catalog_fingerprint=cursor.fingerprint,
                full_scan=True,
            )
        except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
            return None

    def save_catalog(self, batch: ProgressAnnotationCatalogBatch) -> None:
        if not self._available or batch.cursor is None or not batch.full_scan:
            return
        if not batch.items:
            root = None
        else:
            roots = {item.dataset_root for item in batch.items}
            root = next(iter(roots)) if len(roots) == 1 else None
        if root is None:
            return
        try:
            items = [_work_item_payload(root, item) for item in batch.items]
        except (TypeError, ValueError, OSError):
            return
        self._atomic_write(
            self._catalog_path(root),
            {
                "schema": _CATALOG_CACHE_SCHEMA,
                "dataset_root": str(root),
                "catalog_fingerprint": _json_fingerprint(
                    batch.catalog_fingerprint
                ),
                "cursor": _cursor_payload(batch.cursor),
                "items": items,
            },
        )

    def load_sample_continuity(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> EpisodeSampleContinuity | None:
        if not self._available:
            return None
        try:
            payload = self._read_json(self._sample_axis_path(item))
            values = payload.get("cut_times")
            segment_values = payload.get("segment_indices")
            break_values = payload.get("break_indices")
            source_values = payload.get("sources")
            if (
                payload.get("schema") != _SAMPLE_AXIS_CACHE_SCHEMA
                or payload.get("cache_key") != self._sample_axis_key(item)
                or payload.get("clock_domain_id")
                != item.progress.timeline.clock_domain_id
                or not isinstance(values, list)
                or len(values) != item.frame_count
                or not isinstance(segment_values, list)
                or len(segment_values) != item.frame_count
                or not isinstance(break_values, list)
                or not isinstance(source_values, list)
            ):
                return None
            cut_times = tuple(float(value) for value in values)
            if not all(math.isfinite(value) for value in cut_times):
                return None
            sample_axis = EpisodeSampleAxis(
                clock_domain_id=item.progress.timeline.clock_domain_id,
                cut_times=cut_times,
            )
            segment_indices = tuple(
                None if value is None else int(value)
                for value in segment_values
            )
            break_indices = tuple(int(value) for value in break_values)
            if any(
                value <= 0 or value >= item.frame_count
                for value in break_indices
            ) or tuple(sorted(set(break_indices))) != break_indices:
                return None
            return EpisodeSampleContinuity(
                sample_axis=sample_axis,
                segment_indices=segment_indices,
                break_indices=break_indices,
                sources=tuple(str(value) for value in source_values),
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None

    def save_sample_continuity(
        self,
        item: ProgressAnnotationWorkItem,
        continuity: EpisodeSampleContinuity,
    ) -> None:
        sample_axis = continuity.sample_axis
        if len(sample_axis.cut_times) != item.frame_count:
            return
        self._atomic_write(
            self._sample_axis_path(item),
            {
                "schema": _SAMPLE_AXIS_CACHE_SCHEMA,
                "cache_key": self._sample_axis_key(item),
                "clock_domain_id": sample_axis.clock_domain_id,
                "cut_times": list(sample_axis.cut_times),
                "segment_indices": list(continuity.segment_indices),
                "break_indices": list(continuity.break_indices),
                "sources": list(continuity.sources),
            },
        )

    def load_sample_axis(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> EpisodeSampleAxis | None:
        """Compatibility projection for callers that only need timestamps."""

        continuity = self.load_sample_continuity(item)
        return None if continuity is None else continuity.sample_axis

    def save_sample_axis(
        self,
        item: ProgressAnnotationWorkItem,
        sample_axis: EpisodeSampleAxis,
    ) -> None:
        """Compatibility writer for timestamp-only callers."""

        self.save_sample_continuity(
            item,
            EpisodeSampleContinuity(
                sample_axis=sample_axis,
                segment_indices=tuple(None for _ in sample_axis.cut_times),
                break_indices=(),
                sources=(),
            ),
        )


__all__ = ["ProgressAnnotationReadCache"]

"""Read-only work-item discovery for post-capture stage annotation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .progress_artifacts import (
    ProgressArtifactReadCache,
    ProgressArtifactReadError,
    SealedProgressEpisode,
    load_sealed_progress_episode,
)
from .sample_axis import EpisodeSampleAxis
from .retrospective_progress_artifacts import (
    LoadedRetrospectiveProgressEpisode,
    RetrospectiveProgressArtifactReadError,
    load_retrospective_progress_episode,
    load_retrospective_progress_workspace_profiles,
)
from .retrospective_campaign_artifacts import (
    RetrospectiveCampaignArtifactReadError,
    load_retrospective_campaign_index,
)


class ProgressAnnotationJobError(ValueError):
    """Raised when finalized Episode metadata cannot form a safe work item."""


_FULL_SCAN_MAX_WORKERS = 24
_FULL_SCAN_PARALLEL_THRESHOLD = 8


@dataclass(frozen=True)
class ProgressAnnotationCatalogCursor:
    """Byte cursor for one append-only finalized Episode catalog."""

    device: int
    inode: int
    offset: int
    line_number: int
    mtime_ns: int
    prefix_probe_offset: int
    prefix_probe: bytes

    @property
    def fingerprint(self) -> tuple[object, ...]:
        return (
            "file",
            self.device,
            self.inode,
            self.offset,
            self.mtime_ns,
        )


@dataclass(frozen=True)
class ProgressAnnotationCatalogBatch:
    """Eligible work items read since a catalog cursor."""

    items: tuple["ProgressAnnotationWorkItem", ...]
    cursor: ProgressAnnotationCatalogCursor | None
    catalog_fingerprint: tuple[object, ...]
    full_scan: bool


@dataclass(frozen=True)
class ProgressAnnotationWorkItem:
    dataset_root: Path
    episode_index: int
    episode_uuid: str
    episode_identity_hash: str
    progress_manifest_hash: str
    progress_episode_dir: Path
    parquet_path: Path
    video_paths: Mapping[str, Path]
    frame_count: int
    start_timestamp: float
    end_timestamp: float
    progress: SealedProgressEpisode | LoadedRetrospectiveProgressEpisode
    annotation_set_count: int
    has_draft: bool

    @property
    def default_camera_key(self) -> str:
        if "front" in self.video_paths:
            return "front"
        return next(iter(sorted(self.video_paths)))

    def summary_payload(self) -> dict[str, Any]:
        source_binding = getattr(self.progress, "source_binding", None)
        stage_count = (
            len(self.progress.stage_profile.stages)
            if self.progress.stage_profile is not None
            else 0
        )
        return {
            "episode_index": self.episode_index,
            "episode_uuid": self.episode_uuid,
            "episode_identity_hash": self.episode_identity_hash,
            "progress_manifest_hash": self.progress_manifest_hash,
            "frame_count": self.frame_count,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "duration_sec": max(
                0.0,
                self.end_timestamp - self.start_timestamp,
            ),
            "camera_keys": list(sorted(self.video_paths)),
            "default_camera_key": self.default_camera_key,
            "annotation_set_count": self.annotation_set_count,
            "has_draft": self.has_draft,
            "progress_profile_id": self.progress.profile.profile_id,
            "task_stage_profile_id": (
                None
                if self.progress.stage_profile is None
                else self.progress.stage_profile.profile_id
            ),
            "resolution_kind": self.progress.timeline.resolution_kind,
            "recording_scope": self.progress.binding.recording_scope,
            "termination_cause": self.progress.timeline.termination_cause,
            "full_task_outcome": self.progress.timeline.full_task_outcome,
            "source_batch_id": getattr(
                source_binding,
                "source_batch_id",
                None,
            ),
            # UI-only scheduling hint: a single-stage Episode can wait for an
            # explicit video request, while multi-stage Episodes benefit from
            # eager playback and next-Episode warmup.
            "stage_count": stage_count,
        }


def progress_annotation_catalog_fingerprint(
    dataset_root: str | Path,
) -> tuple[object, ...]:
    """Fingerprint either a native finalized catalog or migration workspace."""

    root = Path(dataset_root).expanduser().resolve()
    native = root / "meta" / "episodes.jsonl"
    try:
        stat = native.stat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        return ("error", type(exc).__name__, getattr(exc, "errno", None))
    else:
        return (
            "file",
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )

    episodes_root = root / "episodes"
    profile = root / "profile.cjson"
    stages = root / "stages.cjson"
    if not (episodes_root.is_dir() and profile.is_file() and stages.is_file()):
        return ("missing",)
    try:
        profile_stat = profile.stat()
        stages_stat = stages.stat()
        episode_seals = tuple(
            (
                child.name,
                (child / "migration_manifest.cjson").stat().st_size,
                (child / "migration_manifest.cjson").stat().st_mtime_ns,
            )
            for child in sorted(episodes_root.iterdir(), key=lambda item: item.name)
            if child.is_dir() and (child / "migration_manifest.cjson").is_file()
        )
        campaigns_root = root / "campaigns"
        campaign_artifacts = (
            tuple(
                (
                    str(path.relative_to(root)),
                    path.stat().st_size,
                    path.stat().st_mtime_ns,
                )
                for path in sorted(campaigns_root.rglob("*.cjson"))
                if path.is_file()
            )
            if campaigns_root.is_dir()
            else ()
        )
    except OSError as exc:
        return ("error", type(exc).__name__, getattr(exc, "errno", None))
    return (
        "retrospective",
        profile_stat.st_size,
        profile_stat.st_mtime_ns,
        stages_stat.st_size,
        stages_stat.st_mtime_ns,
        campaign_artifacts,
        episode_seals,
    )


def _retrospective_camera_key(modality_id: str) -> str:
    aliases = {
        "front_rgb": "front",
        "left_wrist_rgb": "left_wrist",
        "right_wrist_rgb": "right_wrist",
    }
    return aliases.get(modality_id, modality_id)


def _retrospective_work_items(
    root: Path,
    *,
    allowed_source_roots: Sequence[str | Path],
) -> tuple[ProgressAnnotationWorkItem, ...]:
    if not allowed_source_roots:
        raise ProgressAnnotationJobError(
            "A retrospective workspace requires at least one explicit "
            "allowed source root."
        )
    try:
        progress_profile, stage_profile = (
            load_retrospective_progress_workspace_profiles(root)
        )
    except RetrospectiveProgressArtifactReadError as exc:
        raise ProgressAnnotationJobError(str(exc)) from exc

    episodes_root = root / "episodes"
    admitted_identities: frozenset[str] | None = None
    if (root / "campaigns").is_dir():
        try:
            campaign_index = load_retrospective_campaign_index(root)
        except RetrospectiveCampaignArtifactReadError as exc:
            raise ProgressAnnotationJobError(str(exc)) from exc
        admitted_identities = (
            campaign_index.admitted_episode_identity_hashes
        )
    results: list[ProgressAnnotationWorkItem] = []
    for episode_dir in sorted(episodes_root.iterdir(), key=lambda item: item.name):
        if not episode_dir.is_dir():
            continue
        if not (episode_dir / "migration_manifest.cjson").is_file():
            continue
        if (
            admitted_identities is not None
            and episode_dir.name not in admitted_identities
        ):
            continue
        try:
            progress = load_retrospective_progress_episode(
                episode_dir,
                progress_profile=progress_profile,
                stage_profile=stage_profile,
                allowed_source_roots=allowed_source_roots,
            )
        except RetrospectiveProgressArtifactReadError as exc:
            raise ProgressAnnotationJobError(
                f"Cannot admit retrospective Episode {episode_dir.name}: {exc}"
            ) from exc
        source_root = progress.resolved_source_dataset_root
        video_paths = {
            _retrospective_camera_key(video.modality_id): (
                source_root / video.source_relative_path
            ).resolve()
            for video in progress.source_binding.admitted_videos
        }
        parquet_path = (
            source_root
            / progress.source_binding.source_parquet_relative_path
        ).resolve()
        episode_uuid = progress.source_binding.episode_capture_identity_hash
        annotation_root = (
            root / "meta" / "progress" / "stage_annotations" / episode_uuid
        )
        set_count = (
            sum(
                1
                for child in annotation_root.iterdir()
                if child.is_dir() and (child / "manifest.cjson").is_file()
            )
            if annotation_root.is_dir()
            else 0
        )
        results.append(
            ProgressAnnotationWorkItem(
                dataset_root=root,
                episode_index=progress.source_binding.source_episode_index,
                episode_uuid=episode_uuid,
                episode_identity_hash=episode_uuid,
                progress_manifest_hash=(
                    progress.migration_manifest.semantic_content_hash
                ),
                progress_episode_dir=episode_dir,
                parquet_path=parquet_path,
                video_paths=video_paths,
                frame_count=progress.sample_axis.sample_count,
                start_timestamp=progress.window.start_at,
                end_timestamp=progress.window.end_at,
                progress=progress,
                annotation_set_count=set_count,
                has_draft=(
                    root
                    / "meta"
                    / "progress"
                    / "stage_annotation_drafts"
                    / episode_uuid
                    / "draft.json"
                ).is_file(),
            )
        )
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.episode_index,
                item.episode_uuid,
            ),
        )
    )


def _safe_dataset_path(root: Path, value: object, *, name: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ProgressAnnotationJobError(f"{name} is missing.")
    path = (root / text).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ProgressAnnotationJobError(
            f"{name} escapes the dataset root."
        ) from exc
    if not path.is_file():
        raise ProgressAnnotationJobError(
            f"{name} does not exist: {text}."
        )
    return path


def _work_item_from_record(
    root: Path,
    record: Mapping[str, Any],
    *,
    read_cache: ProgressArtifactReadCache | None = None,
    load_annotation_state: bool = True,
) -> ProgressAnnotationWorkItem | None:
    metadata = record.get("metadata")
    progress_ref = (
        metadata.get("progress_episode")
        if isinstance(metadata, Mapping)
        else None
    )
    if not isinstance(progress_ref, Mapping):
        return None
    episode_uuid = str(progress_ref.get("episode_uuid", "")).strip()
    manifest_ref = str(progress_ref.get("manifest_ref", "")).strip()
    if not episode_uuid or not manifest_ref:
        return None
    progress_dir = _safe_dataset_path(
        root,
        manifest_ref,
        name="progress manifest_ref",
    ).parent
    try:
        # The sealed Episode owns the exact profile/binding snapshot used
        # during capture.  The corresponding metadata fields are legacy
        # references and must not make an Episode disappear when an editable
        # collection configuration has since changed.
        progress = load_sealed_progress_episode(
            progress_dir,
            expected_episode_identity_hash=str(
                progress_ref.get("episode_identity_hash", "")
            ),
            read_cache=read_cache,
        )
    except ProgressArtifactReadError:
        # An active or partially finalized Episode is intentionally invisible
        # to the annotation application.
        return None
    if progress.stage_profile is None or progress.stage_entry_binding is None:
        return None
    video_values = record.get("video_paths", {})
    if not isinstance(video_values, Mapping) or not video_values:
        return None
    videos = {
        str(camera_key): _safe_dataset_path(
            root,
            relative_path,
            name=f"video_paths[{camera_key!r}]",
        )
        for camera_key, relative_path in video_values.items()
    }
    parquet_path = _safe_dataset_path(
        root,
        record.get("parquet_path"),
        name="parquet_path",
    )
    try:
        episode_index = int(record["episode_index"])
        frame_count = int(record["frame_count"])
        start_timestamp = float(record["start_timestamp"])
        end_timestamp = float(record["end_timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProgressAnnotationJobError(
            f"Episode {episode_uuid!r} has invalid summary values."
        ) from exc
    annotation_root = (
        root / "meta" / "progress" / "stage_annotations" / episode_uuid
    )
    set_count = 0
    has_draft = False
    if load_annotation_state:
        set_count = (
            sum(
                1
                for child in annotation_root.iterdir()
                if child.is_dir() and (child / "manifest.cjson").is_file()
            )
            if annotation_root.is_dir()
            else 0
        )
        has_draft = (
            root
            / "meta"
            / "progress"
            / "stage_annotation_drafts"
            / episode_uuid
            / "draft.json"
        ).is_file()
    return ProgressAnnotationWorkItem(
        dataset_root=root,
        episode_index=episode_index,
        episode_uuid=episode_uuid,
        episode_identity_hash=progress.binding.episode_capture_identity_hash,
        progress_manifest_hash=progress.manifest.semantic_content_hash,
        progress_episode_dir=progress_dir,
        parquet_path=parquet_path,
        video_paths=videos,
        frame_count=frame_count,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        progress=progress,
        annotation_set_count=set_count,
        has_draft=has_draft,
    )


def _cursor_matches_file(
    handle: Any,
    cursor: ProgressAnnotationCatalogCursor,
) -> bool:
    stat = os.fstat(handle.fileno())
    if (
        stat.st_dev != cursor.device
        or stat.st_ino != cursor.inode
        or stat.st_size < cursor.offset
    ):
        return False
    if stat.st_size == cursor.offset and stat.st_mtime_ns != cursor.mtime_ns:
        return False
    handle.seek(cursor.prefix_probe_offset)
    observed = handle.read(len(cursor.prefix_probe))
    return observed == cursor.prefix_probe


def read_progress_annotation_work_item_batch(
    dataset_root: str | Path,
    *,
    cursor: ProgressAnnotationCatalogCursor | None = None,
    allowed_source_roots: Sequence[str | Path] = (),
    progress_callback: Callable[[int, int], None] | None = None,
    item_callback: Callable[["ProgressAnnotationWorkItem"], None] | None = None,
    load_annotation_state: bool = True,
) -> ProgressAnnotationCatalogBatch:
    """Read only newly appended catalog records when the cursor is valid.

    The cursor is validated using file identity, size, and a small exact byte
    probe at the prior boundary.  Replacement, truncation, or in-place rewrite
    falls back to a full streaming scan; no configuration hash participates in
    work-item eligibility.
    """

    root = Path(dataset_root).expanduser().resolve()
    path = root / "meta" / "episodes.jsonl"
    retrospective = (
        (root / "episodes").is_dir()
        and (root / "profile.cjson").is_file()
        and (root / "stages.cjson").is_file()
    )
    if path.is_file() and retrospective:
        raise ProgressAnnotationJobError(
            "Annotation root ambiguously contains both a native Episode "
            "catalog and a retrospective workspace."
        )
    if retrospective:
        return ProgressAnnotationCatalogBatch(
            items=_retrospective_work_items(
                root,
                allowed_source_roots=allowed_source_roots,
            ),
            cursor=None,
            catalog_fingerprint=progress_annotation_catalog_fingerprint(root),
            full_scan=True,
        )
    if not path.is_file():
        return ProgressAnnotationCatalogBatch(
            items=(),
            cursor=None,
            catalog_fingerprint=progress_annotation_catalog_fingerprint(root),
            full_scan=True,
        )

    results: list[ProgressAnnotationWorkItem] = []
    try:
        with path.open("rb") as handle:
            incremental = cursor is not None and _cursor_matches_file(
                handle,
                cursor,
            )
            if incremental:
                assert cursor is not None
                handle.seek(cursor.offset)
                line_number = cursor.line_number
            else:
                handle.seek(0)
                line_number = 0

            records: list[Mapping[str, Any]] = []
            while True:
                raw_line = handle.readline()
                if not raw_line:
                    break
                line_number += 1
                try:
                    line = raw_line.decode("utf-8")
                except UnicodeError as exc:
                    raise ProgressAnnotationJobError(
                        f"Cannot decode episodes.jsonl line {line_number}: {exc}"
                    ) from exc
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ProgressAnnotationJobError(
                        f"Invalid episodes.jsonl line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, Mapping):
                    raise ProgressAnnotationJobError(
                        f"episodes.jsonl line {line_number} is not an object."
                    )
                records.append(value)

            offset = handle.tell()
            stat = os.fstat(handle.fileno())
            probe_offset = max(0, offset - 256)
            handle.seek(probe_offset)
            probe = handle.read(offset - probe_offset)

        total = len(records)
        if progress_callback is not None:
            progress_callback(0, total)

        # A native catalog is append-only, but admitting one Episode involves
        # several independent metadata reads.  On SSHFS/NFS, doing those reads
        # serially multiplies network round-trip latency by the Episode count.
        # Full scans therefore use a bounded read-only pool.  Incremental scans
        # remain sequential because they normally contain only one new record
        # and must not compete with active annotation I/O for no useful gain.
        read_cache = ProgressArtifactReadCache()
        if (
            not incremental
            and total >= _FULL_SCAN_PARALLEL_THRESHOLD
        ):
            workers = min(_FULL_SCAN_MAX_WORKERS, total)
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="annotation-catalog",
            ) as executor:
                hydrated = executor.map(
                    lambda record: _work_item_from_record(
                        root,
                        record,
                        read_cache=read_cache,
                        load_annotation_state=load_annotation_state,
                    ),
                    records,
                )
                for completed, item in enumerate(hydrated, start=1):
                    if item is not None:
                        results.append(item)
                        if item_callback is not None:
                            item_callback(item)
                    if progress_callback is not None:
                        progress_callback(completed, total)
        else:
            for completed, record in enumerate(records, start=1):
                item = _work_item_from_record(
                    root,
                    record,
                    read_cache=read_cache,
                    load_annotation_state=load_annotation_state,
                )
                if item is not None:
                    results.append(item)
                    if item_callback is not None:
                        item_callback(item)
                if progress_callback is not None:
                    progress_callback(completed, total)
    except ProgressAnnotationJobError:
        raise
    except OSError as exc:
        raise ProgressAnnotationJobError(
            f"Cannot read finalized Episode catalog: {exc}"
        ) from exc

    next_cursor = ProgressAnnotationCatalogCursor(
        device=stat.st_dev,
        inode=stat.st_ino,
        offset=offset,
        line_number=line_number,
        mtime_ns=stat.st_mtime_ns,
        prefix_probe_offset=probe_offset,
        prefix_probe=probe,
    )
    return ProgressAnnotationCatalogBatch(
        items=tuple(sorted(results, key=lambda item: item.episode_index)),
        cursor=next_cursor,
        catalog_fingerprint=next_cursor.fingerprint,
        full_scan=not incremental,
    )


def list_progress_annotation_work_items(
    dataset_root: str | Path,
    *,
    allowed_source_roots: Sequence[str | Path] = (),
) -> tuple[ProgressAnnotationWorkItem, ...]:
    """List only finalized Episodes with complete sealed stage semantics."""

    return read_progress_annotation_work_item_batch(
        dataset_root,
        allowed_source_roots=allowed_source_roots,
    ).items


def load_episode_sample_axis(
    work_item: ProgressAnnotationWorkItem,
) -> EpisodeSampleAxis:
    """Load only the timestamp column needed by timeline indexing."""

    if isinstance(work_item.progress, LoadedRetrospectiveProgressEpisode):
        return work_item.progress.sample_axis

    try:
        import pyarrow.parquet as pq  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ProgressAnnotationJobError(
            "pyarrow is required to read finalized Episode timestamps."
        ) from exc
    try:
        table = pq.read_table(work_item.parquet_path, columns=["timestamp"])
        values = tuple(float(value.as_py()) for value in table["timestamp"])
    except Exception as exc:
        raise ProgressAnnotationJobError(
            f"Cannot read Episode sample timestamps: {exc}"
        ) from exc
    if len(values) != work_item.frame_count:
        raise ProgressAnnotationJobError(
            "Parquet timestamp count differs from finalized frame_count."
        )
    return EpisodeSampleAxis(
        clock_domain_id=work_item.progress.timeline.clock_domain_id,
        cut_times=values,
    )


__all__ = [
    "ProgressAnnotationCatalogBatch",
    "ProgressAnnotationCatalogCursor",
    "ProgressAnnotationJobError",
    "ProgressAnnotationWorkItem",
    "list_progress_annotation_work_items",
    "load_episode_sample_axis",
    "progress_annotation_catalog_fingerprint",
    "read_progress_annotation_work_item_batch",
]

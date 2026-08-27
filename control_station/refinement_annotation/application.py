"""Application coordination for post-capture fine-stage annotation."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import errno
import json
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from collection_runtime.progress.stage_annotation_store import (
    STAGE_ANNOTATION_RELATIVE_ROOT,
    DatasetStageAnnotationWorkspace,
    LoadedStageAnnotationSet,
    StageAnnotationCommitReport,
    StageAnnotationStoreError,
    commit_stage_annotation_set,
    load_stage_annotation_set,
)
from collection_runtime.progress.detail_stage_annotation_store import (
    DetailStageAnnotationStoreError,
    commit_detail_stage_annotation_set,
)
from collection_runtime.progress.failure_trigger_store import (
    FailureTriggerStoreError,
    commit_failure_trigger_annotation_set,
    failure_trigger_path,
    load_failure_trigger_annotation_set,
)
from collection_runtime.progress.recoverable_failure_store import (
    RecoverableFailureStoreError,
    commit_recoverable_failure_annotation_set,
    load_recoverable_failure_annotation_set,
    recoverable_failure_path,
)
from dataset_contracts.causal_snapshot import LOCAL_WALL_CLOCK
from dataset_contracts.progress_failure_triggers import (
    FailureTriggerAnnotationSetV1,
)
from dataset_contracts.progress_recoverable_failures import (
    RecoverableFailureAnnotationSetV1,
)
from dataset_contracts.progress_detail_annotations import (
    DetailStageAnnotationBundleV1,
    DetailStageAnnotationSetBindingV1,
    validate_detail_stage_annotation_set,
)
from dataset_contracts.progress_stage_annotations import (
    AbnormalIntervalAnnotationV1,
    CaptureEndStepAnnotationV1,
    EpisodeAbnormalAnnotationV1,
    EpisodeReviewRequestAnnotationV1,
    ProgressBoundaryOverrideAnnotationV1,
    StageAnnotationBundleV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
    capture_end_projected_attempt_id,
)
from dataset_interface.progress_annotation_jobs import (
    EpisodeSampleContinuity,
    ProgressAnnotationCatalogCursor,
    ProgressAnnotationWorkItem,
    load_episode_sample_continuity,
    progress_annotation_catalog_fingerprint,
    progress_annotation_catalog_probe_fingerprint,
    read_progress_annotation_work_item_batch,
)
from dataset_interface.progress_failure_triggers import (
    FailureTriggerResolutionError,
    build_failure_trigger_annotations,
    failure_trigger_sample_axis_hash,
    normalize_failure_trigger_drafts,
)
from dataset_interface.progress_recoverable_failures import (
    RecoverableFailureResolutionError,
    build_recoverable_failure_annotations,
    normalize_recoverable_failure_drafts,
    recoverable_failure_sample_axis_hash,
)
from dataset_interface.progress_stages import resolve_stage_training_index
from dataset_interface.progress_detail_annotation_artifacts import (
    LoadedDetailStageAnnotationSet,
)
from dataset_interface.retrospective_progress_artifacts import (
    LoadedRetrospectiveProgressEpisode,
)
from dataset_interface.sample_axis import EpisodeSampleAxis
from dataset_interface.progress_annotation_read_cache import (
    ProgressAnnotationReadCache,
)

from .detail_annotation import (
    DetailAnnotationModule,
    DetailProfileSelection,
)
from .episode_status import (
    EpisodeAnnotationSummary,
    build_episode_status_payload,
    summarize_annotation_sets,
)


_DRAFT_SCHEMA = "dexteleop.stage_annotation_draft.local.v1"
_DETAIL_DRAFT_SCHEMA = "dexteleop.detail_stage_annotation_draft.local.v1"
_SAMPLE_AXIS_CACHE_LIMIT = 16
_ANNOTATION_SET_CACHE_LIMIT = 16
_ANNOTATION_SUMMARY_CACHE_LIMIT = 256
_CATALOG_EXACT_RECHECK_INTERVAL_SEC = 60.0
# Keep the HTTP/browser page deliberately bounded.  Direct Python callers
# still receive the historical complete list unless they opt into pagination.
_EPISODE_PAGE_MAX = 1000
_RETRYABLE_STAGE_STORE_ERRNOS = frozenset(
    value
    for value in (
        errno.EAGAIN,
        errno.EINTR,
        errno.EIO,
        getattr(errno, "ESTALE", None),
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        getattr(errno, "ENOTCONN", None),
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
    )
    if value is not None
)


def _active_wall_intervals(
    item: ProgressAnnotationWorkItem,
    start_at: float,
    end_at: float,
) -> tuple[tuple[float, float], ...]:
    """Return wall-clock pieces that contain collectable Episode time."""

    clipped = sorted(
        (
            max(start_at, pause.start_timestamp),
            min(end_at, pause.end_timestamp),
        )
        for pause in item.recording_pause_segments
        if pause.end_timestamp > start_at and pause.start_timestamp < end_at
    )
    merged: list[tuple[float, float]] = []
    for begin, end in clipped:
        if end <= begin:
            continue
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    active: list[tuple[float, float]] = []
    cursor = start_at
    for begin, end in merged:
        if cursor < begin:
            active.append((cursor, begin))
        cursor = max(cursor, end)
    if cursor < end_at:
        active.append((cursor, end_at))
    return tuple(active)


def _equal_active_time_splits(
    item: ProgressAnnotationWorkItem,
    start_at: float,
    end_at: float,
    count: int,
) -> tuple[tuple[float, float], ...]:
    """Split collectable time evenly without assigning a leading/tail pause."""

    if count <= 0 or end_at <= start_at:
        return ()
    active = _active_wall_intervals(item, start_at, end_at)
    total = sum(end - begin for begin, end in active)
    if total <= 0.0:
        duration = end_at - start_at
        return tuple(
            (
                start_at + duration * index / count,
                start_at + duration * (index + 1) / count,
            )
            for index in range(count)
        )

    def wall_at_active_offset(offset: float) -> float:
        if offset <= 0.0:
            return active[0][0]
        remaining = min(offset, total)
        for begin, end in active:
            duration = end - begin
            if remaining <= duration:
                return begin + remaining
            remaining -= duration
        return active[-1][1]

    boundaries = tuple(
        wall_at_active_offset(total * index / count)
        for index in range(count + 1)
    )
    return tuple(zip(boundaries, boundaries[1:]))


class RefinementAnnotationApplicationError(ValueError):
    """Raised when a UI request cannot preserve annotation semantics."""


class RefinementAnnotationApplication:
    """Own one dataset view and at most one Episode editing lock."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        annotator_id: str,
        second_review: bool = False,
        allowed_source_roots: Sequence[str | Path] = (),
        defer_initial_refresh: bool = False,
        read_cache: ProgressAnnotationReadCache | None = None,
        detail_profile_root: str | Path | None = None,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.annotator_id = str(annotator_id).strip()
        if not self.annotator_id:
            raise ValueError("annotator_id must be non-empty.")
        self.annotation_mode = (
            "SECOND_REVIEW" if bool(second_review) else "STANDARD"
        )
        self.allowed_source_roots = tuple(
            Path(value).expanduser().resolve()
            for value in allowed_source_roots
        )
        self.detail_profile_root = (
            Path(detail_profile_root).expanduser().resolve()
            if detail_profile_root is not None
            else Path(__file__).resolve().parents[2]
            / "collection_runtime"
            / "progress"
            / "profiles"
        )
        self.workspace = DatasetStageAnnotationWorkspace(
            self.dataset_root,
            annotator_id=self.annotator_id,
        )
        self.workspace.acquire()
        self._read_cache = read_cache or ProgressAnnotationReadCache()
        self._lock = threading.RLock()
        self._episode_operation_lock = threading.RLock()
        self._summary_load_locks = tuple(
            threading.Lock() for _ in range(16)
        )
        self._refresh_condition = threading.Condition(self._lock)
        self._closed = False
        self._active_episode_uuid: str | None = None
        self._items: dict[str, ProgressAnnotationWorkItem] = {}
        self._catalog_cursor: ProgressAnnotationCatalogCursor | None = None
        self._known_catalog_fingerprint: tuple[object, ...] | None = None
        self._known_catalog_probe: tuple[object, ...] | None = None
        self._last_exact_catalog_check_at = 0.0
        self._refresh_in_flight = False
        self._refresh_thread: threading.Thread | None = None
        self._refresh_error: Exception | None = None
        self._refresh_completed = 0
        self._refresh_total: int | None = None
        self._annotation_sets_cache: OrderedDict[
            str,
            tuple[LoadedStageAnnotationSet, ...],
        ] = OrderedDict()
        self._annotation_summary_cache: OrderedDict[
            str,
            EpisodeAnnotationSummary,
        ] = OrderedDict()
        self._annotation_root_fingerprints: dict[
            str,
            tuple[object, ...],
        ] = {}
        self._detail_annotations = DetailAnnotationModule(
            self.dataset_root,
            profile_root=self.detail_profile_root,
        )
        self._episode_list_revision = 0
        self._sample_axis_cache: OrderedDict[
            str,
            tuple[tuple[object, ...], EpisodeSampleContinuity],
        ] = OrderedDict()
        self._ordered_items_cache: tuple[ProgressAnnotationWorkItem, ...] = ()
        self._ordered_items_cache_revision = -1
        # The GUI can return immediately and expose a loading state while the
        # first remote catalog scan runs. Direct application callers retain
        # the historical synchronous initialization by default.
        self.refresh(wait=not defer_initial_refresh, force=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            refresh_thread = self._refresh_thread
            self._refresh_condition.notify_all()
        if (
            refresh_thread is not None
            and refresh_thread is not threading.current_thread()
        ):
            refresh_thread.join(timeout=5.0)
        with self._episode_operation_lock:
            with self._lock:
                self._active_episode_uuid = None
                self.workspace.close()
                self._items.clear()
                self._ordered_items_cache = ()
                self._ordered_items_cache_revision = -1
                self._annotation_sets_cache.clear()
                self._annotation_summary_cache.clear()
                self._annotation_root_fingerprints.clear()
                self._sample_axis_cache.clear()
                self._detail_annotations.close()

    def _catalog_fingerprint(self) -> tuple[object, ...]:
        return progress_annotation_catalog_fingerprint(self.dataset_root)

    def _catalog_probe(self) -> tuple[object, ...]:
        return progress_annotation_catalog_probe_fingerprint(self.dataset_root)

    def _finish_refresh(
        self,
        items: dict[str, ProgressAnnotationWorkItem] | None,
        cursor: ProgressAnnotationCatalogCursor | None,
        catalog_fingerprint: tuple[object, ...] | None,
        catalog_probe: tuple[object, ...] | None,
        error: Exception | None,
    ) -> None:
        with self._lock:
            if not self._closed and error is None and items is not None:
                # Drafts and annotation sets may be written while a remote
                # catalog scan is in flight.  Preserve those monotonic local
                # observations instead of replacing them with an older scan.
                for episode_uuid, current in self._items.items():
                    discovered = items.get(episode_uuid)
                    if discovered is None:
                        continue
                    has_draft = discovered.has_draft or current.has_draft
                    annotation_set_count = max(
                        discovered.annotation_set_count,
                        current.annotation_set_count,
                    )
                    if (
                        has_draft != discovered.has_draft
                        or annotation_set_count
                        != discovered.annotation_set_count
                    ):
                        items[episode_uuid] = replace(
                            discovered,
                            has_draft=has_draft,
                            annotation_set_count=annotation_set_count,
                        )
                self._items = items
                self._catalog_cursor = cursor
                self._known_catalog_fingerprint = catalog_fingerprint
                self._known_catalog_probe = catalog_probe
                self._last_exact_catalog_check_at = time.monotonic()
                self._refresh_error = None
                self._episode_list_revision += 1
                eligible = set(items)
                for cache in (
                    self._annotation_sets_cache,
                    self._annotation_summary_cache,
                    self._annotation_root_fingerprints,
                    self._sample_axis_cache,
                ):
                    for episode_uuid in tuple(cache):
                        if episode_uuid not in eligible:
                            cache.pop(episode_uuid, None)
                self._detail_annotations.prune(eligible)
            elif not self._closed and error is not None:
                self._refresh_error = error
            self._refresh_in_flight = False
            self._refresh_thread = None
            self._refresh_condition.notify_all()

    def _run_refresh(self, *, force_full: bool, raise_error: bool) -> None:
        with self._lock:
            cursor = None if force_full else self._catalog_cursor
            current_items = dict(self._items)
        try:
            batch = None
            if force_full:
                fingerprint = self._catalog_fingerprint()
                cached_prefix = self._read_cache.load_catalog_prefix(
                    self.dataset_root
                )
                if cached_prefix is not None:
                    if cached_prefix.catalog_fingerprint == fingerprint:
                        batch = cached_prefix
                    else:
                        cursor = cached_prefix.cursor
                        current_items = {
                            item.episode_uuid: item
                            for item in cached_prefix.items
                        }
                        for item in cached_prefix.items:
                            self._publish_refresh_item(item)
            if batch is None:
                batch = read_progress_annotation_work_item_batch(
                    self.dataset_root,
                    cursor=cursor,
                    allowed_source_roots=self.allowed_source_roots,
                    progress_callback=self._update_refresh_progress,
                    item_callback=(
                        self._publish_refresh_item if force_full else None
                    ),
                    # Exact badges are loaded lazily for visible rows. Avoid an
                    # O(Episodes x annotation versions) SSHFS walk at startup.
                    load_annotation_state=False,
                )
                if batch.full_scan:
                    self._read_cache.save_catalog(batch)
        except Exception as exc:
            self._finish_refresh(None, None, None, None, exc)
            if raise_error:
                raise
            return

        if batch.full_scan:
            items = {item.episode_uuid: item for item in batch.items}
        else:
            items = current_items
            items.update(
                (item.episode_uuid, item)
                for item in batch.items
            )
            self._read_cache.save_catalog(
                replace(
                    batch,
                    items=tuple(
                        sorted(
                            items.values(),
                            key=lambda item: item.episode_index,
                        )
                    ),
                    full_scan=True,
                )
            )
        self._finish_refresh(
            items,
            batch.cursor,
            batch.catalog_fingerprint,
            self._catalog_probe(),
            None,
        )

    def _run_background_refresh(self, *, force_full: bool) -> None:
        self._run_refresh(force_full=force_full, raise_error=False)

    def _update_refresh_progress(self, completed: int, total: int) -> None:
        with self._lock:
            if self._closed:
                return
            self._refresh_completed = max(0, int(completed))
            self._refresh_total = max(0, int(total))

    def _publish_refresh_item(self, item: ProgressAnnotationWorkItem) -> None:
        """Expose verified full-scan results as soon as each is available."""

        with self._lock:
            if self._closed:
                return
            existing = self._items.get(item.episode_uuid)
            if existing is not None:
                item = replace(
                    item,
                    has_draft=item.has_draft or existing.has_draft,
                    annotation_set_count=max(
                        item.annotation_set_count,
                        existing.annotation_set_count,
                    ),
                )
            self._items[item.episode_uuid] = item
            self._episode_list_revision += 1

    def refresh(
        self,
        *,
        wait: bool = True,
        force: bool = True,
    ) -> None:
        """Refresh the work-item snapshot without holding the app lock.

        Synchronous callers retain the historical behavior.  Browser polling
        can pass ``wait=False`` so a slow remote catalog scan is coalesced in a
        daemon worker and never blocks Episode/detail/commit requests.
        """

        with self._lock:
            self._require_open()
            if self._refresh_in_flight:
                if not wait:
                    return
                while self._refresh_in_flight:
                    self._refresh_condition.wait()
                if self._refresh_error is not None:
                    raise self._refresh_error
                return

        # Polling first uses a bounded metadata probe.  Retrospective exact
        # fingerprints enumerate every sealed Episode, so repeat that scan
        # only when the probe changes or at a conservative fail-safe interval.
        catalog_probe: tuple[object, ...] | None = None
        catalog_fingerprint: tuple[object, ...] | None = None
        if not force:
            catalog_probe = self._catalog_probe()
            now = time.monotonic()
            with self._lock:
                probe_unchanged = catalog_probe == self._known_catalog_probe
                exact_recent = (
                    now - self._last_exact_catalog_check_at
                    < _CATALOG_EXACT_RECHECK_INTERVAL_SEC
                )
                refresh_error = self._refresh_error
            if probe_unchanged and exact_recent and refresh_error is None:
                return
            catalog_fingerprint = self._catalog_fingerprint()
        with self._lock:
            self._require_open()
            needs_refresh = (
                force
                or self._refresh_error is not None
                or catalog_fingerprint != self._known_catalog_fingerprint
            )
            if not needs_refresh:
                self._known_catalog_probe = catalog_probe
                self._last_exact_catalog_check_at = time.monotonic()
                return
            if self._refresh_in_flight:
                if not wait:
                    return
                while self._refresh_in_flight:
                    self._refresh_condition.wait()
                if self._refresh_error is not None:
                    raise self._refresh_error
                return
            self._refresh_in_flight = True
            self._refresh_error = None
            self._refresh_completed = 0
            self._refresh_total = None
            if not wait:
                refresh_thread = threading.Thread(
                    target=self._run_background_refresh,
                    kwargs={"force_full": force},
                    name="refinement-annotation-refresh",
                    daemon=True,
                )
                self._refresh_thread = refresh_thread
                refresh_thread.start()
                return
        self._run_refresh(force_full=force, raise_error=True)

    @staticmethod
    def _sample_axis_cache_key(
        item: ProgressAnnotationWorkItem,
    ) -> tuple[object, ...]:
        # A work item is admitted only from an immutable sealed Episode.  Its
        # semantic identities are therefore a stronger cache key than a remote
        # stat call on every open, and avoid an SSHFS round trip on cache hits.
        return (
            item.episode_identity_hash,
            item.progress_manifest_hash,
            item.frame_count,
            str(item.parquet_path),
        )

    def _sample_continuity(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> EpisodeSampleContinuity:
        """Reuse immutable Episode timestamps and continuity while open."""

        cache_key = self._sample_axis_cache_key(item)
        with self._lock:
            cached = self._sample_axis_cache.get(item.episode_uuid)
            if cached is not None and cached[0] == cache_key:
                self._sample_axis_cache.move_to_end(item.episode_uuid)
                return cached[1]

        persistent = self._read_cache.load_sample_continuity(item)
        if persistent is not None:
            continuity = persistent
        else:
            continuity = load_episode_sample_continuity(item)
            self._read_cache.save_sample_continuity(item, continuity)
        with self._lock:
            self._sample_axis_cache[item.episode_uuid] = (
                cache_key,
                continuity,
            )
            self._sample_axis_cache.move_to_end(item.episode_uuid)
            while len(self._sample_axis_cache) > _SAMPLE_AXIS_CACHE_LIMIT:
                self._sample_axis_cache.popitem(last=False)
        return continuity

    def _sample_axis(self, item: ProgressAnnotationWorkItem) -> EpisodeSampleAxis:
        return self._sample_continuity(item).sample_axis

    def list_episodes(
        self,
        *,
        wait_for_refresh: bool = True,
        force_refresh: bool = False,
        include_annotation_summaries: bool = True,
        offset: int | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Return the current Episode snapshot, optionally as one page.

        The annotation editor historically returned every Episode to direct
        callers.  That remains the default for compatibility, while the HTTP
        client can request a bounded page so a large dataset does not require
        serializing and retaining the complete catalog in one browser update.
        """

        if offset is not None:
            try:
                offset = int(offset)
            except (TypeError, ValueError) as exc:
                raise RefinementAnnotationApplicationError(
                    "Episode page offset must be an integer."
                ) from exc
            if offset < 0:
                raise RefinementAnnotationApplicationError(
                    "Episode page offset must be non-negative."
                )
        if limit is not None:
            try:
                limit = int(limit)
            except (TypeError, ValueError) as exc:
                raise RefinementAnnotationApplicationError(
                    "Episode page limit must be an integer."
                ) from exc
            if limit <= 0 or limit > _EPISODE_PAGE_MAX:
                raise RefinementAnnotationApplicationError(
                    f"Episode page limit must be between 1 and {_EPISODE_PAGE_MAX}."
                )
        if (offset is None) != (limit is None):
            raise RefinementAnnotationApplicationError(
                "Episode page offset and limit must be supplied together."
            )
        self.refresh(wait=wait_for_refresh, force=force_refresh)
        with self._lock:
            self._require_open()
            revision = self._episode_list_revision
            refreshing = self._refresh_in_flight
            refresh_error = self._refresh_error
            refresh_completed = self._refresh_completed
            refresh_total = self._refresh_total
            if self._ordered_items_cache_revision != revision:
                self._ordered_items_cache = tuple(
                    sorted(
                        self._items.values(),
                        key=lambda value: (
                            value.episode_index,
                            value.episode_uuid,
                        ),
                    )
                )
                self._ordered_items_cache_revision = revision
            ordered_items = self._ordered_items_cache
        total_count = len(ordered_items)
        page_offset = 0 if offset is None else offset
        page_items = (
            ordered_items
            if limit is None
            else ordered_items[page_offset : page_offset + limit]
        )
        return {
            "dataset_root": str(self.dataset_root),
            "annotator_id": self.annotator_id,
            "annotation_mode": self.annotation_mode,
            "revision": revision,
            "refreshing": refreshing,
            "refresh_error": (
                None if refresh_error is None else str(refresh_error)
            ),
            "refresh_progress": {
                "completed": refresh_completed,
                "total": refresh_total,
            },
            "total_count": total_count,
            "offset": page_offset,
            "limit": limit,
            "has_more": page_offset + len(page_items) < total_count,
            "episodes": [
                self._episode_summary(
                    item,
                    load_annotations=include_annotation_summaries,
                )
                for item in page_items
            ],
        }

    def episode_annotation_summaries(
        self,
        episode_uuids: Sequence[str],
    ) -> dict[str, Any]:
        """Load exact annotation badges only for a small visible UI batch."""

        normalized = tuple(
            dict.fromkeys(str(value).strip() for value in episode_uuids)
        )
        if not normalized or any(not value for value in normalized):
            raise RefinementAnnotationApplicationError(
                "episode_uuids must contain at least one non-empty identity."
            )
        if len(normalized) > 32:
            raise RefinementAnnotationApplicationError(
                "At most 32 Episode annotation summaries may be loaded at once."
            )

        def load_one(
            episode_uuid: str,
        ) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
            try:
                item = self._item(episode_uuid)
                with self._summary_load_lock(episode_uuid):
                    with self._lock:
                        active = self._active_episode_uuid
                        cached = episode_uuid in self._annotation_sets_cache
                    if (
                        active != episode_uuid
                        and cached
                        and not self._annotation_cache_matches_storage(item)
                    ):
                        with self._lock:
                            # A different machine published a new immutable
                            # manifest-last set. Reload only after the Episode
                            # annotation directory actually changed; visible
                            # rows must not reread every sealed version every
                            # refresh interval on a remote mount.
                            self._annotation_sets_cache.pop(episode_uuid, None)
                            self._annotation_summary_cache.pop(episode_uuid, None)
                            self._annotation_root_fingerprints.pop(
                                episode_uuid,
                                None,
                            )
                        self._detail_annotations.invalidate(episode_uuid)
                    summary = self._episode_summary(
                        item,
                        load_annotations=True,
                    )
            except (
                OSError,
                RefinementAnnotationApplicationError,
                StageAnnotationStoreError,
                ValueError,
            ) as exc:
                return None, {
                    "episode_uuid": episode_uuid,
                    "error": str(exc),
                }
            return (
                {
                    "episode_uuid": episode_uuid,
                    "annotation_set_count": summary["annotation_set_count"],
                    "needs_second_review": summary["needs_second_review"],
                    "review_state": summary["review_state"],
                    "latest_annotation_mode": summary[
                        "latest_annotation_mode"
                    ],
                    "whole_episode_abnormal": summary[
                        "whole_episode_abnormal"
                    ],
                    "whole_episode_abnormal_reason": summary[
                        "whole_episode_abnormal_reason"
                    ],
                    "detail_annotation_complete": summary[
                        "detail_annotation_complete"
                    ],
                    "detail_annotation_state": summary[
                        "detail_annotation_state"
                    ],
                    "annotation_status": summary["annotation_status"],
                    "annotation_summary_loaded": True,
                },
                None,
            )

        # Visible rows are independent immutable reads.  A small pool avoids
        # multiplying SSHFS/NFS latency by the batch size without allowing a
        # sidebar refresh to monopolize the application.
        if len(normalized) == 1:
            loaded = (load_one(normalized[0]),)
        else:
            with ThreadPoolExecutor(
                max_workers=min(8, len(normalized)),
                thread_name_prefix="annotation-summary",
            ) as executor:
                loaded = tuple(executor.map(load_one, normalized))
        summaries: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for summary, error in loaded:
            if error is not None:
                errors.append(error)
            elif summary is not None:
                summaries.append(summary)
        return {"summaries": summaries, "errors": errors}

    def _annotation_root_fingerprint(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> tuple[object, ...]:
        root = (
            self.dataset_root
            / STAGE_ANNOTATION_RELATIVE_ROOT
            / item.episode_uuid
        )
        try:
            value = root.stat()
        except FileNotFoundError:
            return ("missing",)
        except OSError as exc:
            return ("error", type(exc).__name__, getattr(exc, "errno", None))
        return (
            "directory",
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    def _annotation_cache_matches_storage(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> bool:
        with self._lock:
            cached = self._annotation_root_fingerprints.get(item.episode_uuid)
        return cached is not None and cached == self._annotation_root_fingerprint(
            item
        )

    def _detail_profile_selection(
        self,
        item: ProgressAnnotationWorkItem,
        *,
        selection_id: str | None = None,
    ) -> tuple[DetailProfileSelection | None, str | None]:
        return self._detail_annotations.profile_selection(
            item,
            selection_id=selection_id,
        )

    def _detail_profile_selections(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> tuple[tuple[DetailProfileSelection, ...], str | None]:
        return self._detail_annotations.profile_selections(item)

    def _persisted_detail_profile_selection_id(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> str | None:
        """Read the optional selector for sidebar status without an Episode lock.

        The visible Episode summary is deliberately a read-only batch path and
        cannot acquire an editing lock for every row.  Drafts are atomically
        replaced by :class:`DatasetStageAnnotationWorkspace`, so reading this
        small selector directly is safe and keeps the summary consistent with
        an explicit coarse-only choice across process restarts.  A missing or
        legacy/malformed draft preserves the historical ``None`` resolution.
        """

        try:
            payload = json.loads(
                self.workspace.draft_path(
                    episode_uuid=item.episode_uuid,
                ).read_text(encoding="utf-8")
            )
        except (
            FileNotFoundError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ):
            return None
        if not isinstance(payload, Mapping):
            return None
        value = payload.get("detail_profile_selection_id")
        if value is None:
            return None
        return value.strip() if isinstance(value, str) else None

    def _current_detail_annotation_set(
        self,
        item: ProgressAnnotationWorkItem,
        base_annotation_set: LoadedStageAnnotationSet,
        selection: DetailProfileSelection,
    ) -> tuple[LoadedDetailStageAnnotationSet | None, str | None]:
        return self._detail_annotations.current_annotation_set(
            item,
            base_annotation_set,
            selection,
        )

    @staticmethod
    def _detail_occurrence_payload(value: Any) -> dict[str, Any]:
        return {
            "annotation_id": value.annotation_id,
            "coarse_attempt_id": value.coarse_attempt_id,
            "progress_step_id": value.progress_step_id,
            "stage_id": value.stage_id,
            "occurrence_index_in_attempt": value.occurrence_index_in_attempt,
            "start_at": value.start_at,
            "end_at": value.end_at,
            "outcome": value.outcome,
            "confidence": value.confidence,
            "source_failure_fact_id": value.source_failure_fact_id,
        }

    def _detail_draft_from_annotation_set(
        self,
        item: ProgressAnnotationWorkItem,
        value: LoadedDetailStageAnnotationSet,
        *,
        draft_revision: int,
    ) -> dict[str, Any]:
        failure_stage_by_attempt = {
            occurrence.coarse_attempt_id: occurrence.stage_id
            for occurrence in value.bundle.occurrences
            if occurrence.outcome == "FAILURE"
            and occurrence.source_failure_fact_id is not None
        }
        return {
            "schema": _DETAIL_DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            "base_annotation_set_id": value.binding.base_annotation_set_id,
            "base_annotation_set_manifest_hash": (
                value.binding.base_annotation_set_manifest_hash
            ),
            "base_task_stage_profile_hash": (
                value.binding.base_task_stage_profile_hash
            ),
            "detail_task_stage_profile_hash": (
                value.binding.detail_task_stage_profile_hash
            ),
            "target_progress_step_id": value.binding.target_progress_step_id,
            "draft_revision": int(draft_revision),
            "occurrences": [
                self._detail_occurrence_payload(occurrence)
                for occurrence in value.bundle.occurrences
            ],
            "failure_stage_by_attempt": failure_stage_by_attempt,
            "confirmed": True,
            "last_committed_detail_annotation_set_id": (
                value.binding.annotation_set_id
            ),
            "last_committed_detail_manifest_hash": (
                value.manifest.semantic_content_hash
            ),
        }

    @staticmethod
    def _detail_stage_payload(stage: Any) -> dict[str, Any]:
        return {
            "stage_id": stage.stage_id,
            "progress_step_id": stage.progress_step_id,
            "display_name": stage.display_name,
            "description": stage.description,
            "ordinal": stage.ordinal,
        }

    def _detail_profile_options_payload(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> tuple[list[dict[str, Any]], str | None]:
        selections, reason = self._detail_profile_selections(item)
        options = [
            {
                "selection_id": selection.selection_id,
                "profile_id": selection.stage_profile.profile_id,
                "profile_version": selection.stage_profile.profile_version,
                "target_progress_step_id": selection.target_progress_step_id,
                "stages": [
                    self._detail_stage_payload(stage)
                    for stage in selection.stage_profile.stages_for_step(
                        selection.target_progress_step_id
                    )
                ],
            }
            for selection in selections
        ]
        return options, reason

    def _bootstrap_detail_draft(
        self,
        item: ProgressAnnotationWorkItem,
        base_annotation_set: LoadedStageAnnotationSet,
        selection: DetailProfileSelection,
    ) -> dict[str, Any]:
        return self._bootstrap_detail_draft_from_occurrences(
            item,
            base_annotation_set.bundle.occurrences,
            selection,
            base_annotation_set_id=base_annotation_set.binding.annotation_set_id,
            base_annotation_set_manifest_hash=(
                base_annotation_set.manifest.semantic_content_hash
            ),
        )

    def _bootstrap_detail_draft_from_coarse_draft(
        self,
        item: ProgressAnnotationWorkItem,
        coarse_draft: Mapping[str, Any],
        selection: DetailProfileSelection,
    ) -> dict[str, Any]:
        coarse_bundle = self._bundle_from_draft(item, coarse_draft)
        return self._bootstrap_detail_draft_from_occurrences(
            item,
            coarse_bundle.occurrences,
            selection,
            base_annotation_set_id=None,
            base_annotation_set_manifest_hash=None,
        )

    def _bootstrap_detail_draft_from_occurrences(
        self,
        item: ProgressAnnotationWorkItem,
        base_occurrences: Sequence[Any],
        selection: DetailProfileSelection,
        *,
        base_annotation_set_id: str | None,
        base_annotation_set_manifest_hash: str | None,
    ) -> dict[str, Any]:
        """Create an unsealed equal-split detail working view.

        This is only a UI starting point.  It is never treated as detail data
        by readers until an explicit detail seal passes the immutable
        contract, and failure Attempts remain unresolved until the annotator
        selects their fine failure stage.
        """

        assert item.progress.stage_profile is not None
        target = selection.target_progress_step_id
        stages = selection.stage_profile.stages_for_step(target)
        target_occurrences = sorted(
            (
                value
                for value in base_occurrences
                if value.progress_step_id == target
            ),
            key=lambda value: (
                value.coarse_attempt_id,
                value.start_at,
                value.end_at,
                value.annotation_id,
            ),
        )
        if not stages or not target_occurrences:
            raise RefinementAnnotationApplicationError(
                "The selected detail profile has no target stages or the "
                "coarse annotation draft has no target occurrence."
            )
        grouped: dict[str, list[Any]] = {}
        for value in target_occurrences:
            grouped.setdefault(value.coarse_attempt_id, []).append(value)
        occurrences: list[dict[str, Any]] = []
        failure_stage_by_attempt: dict[str, str | None] = {}
        for attempt_id, values in grouped.items():
            ordered_values = sorted(
                values,
                key=lambda value: (value.start_at, value.end_at),
            )
            source_outcome = ordered_values[-1].outcome
            source_failure_fact_id = next(
                (
                    value.source_failure_fact_id
                    for value in ordered_values
                    if value.source_failure_fact_id is not None
                ),
                None,
            )
            if source_failure_fact_id is not None:
                failure_stage_by_attempt[attempt_id] = None
            total_duration = sum(
                max(0.0, value.end_at - value.start_at)
                for value in ordered_values
            )
            stage_cursor = 0
            remaining_stages = len(stages)
            for segment_index, value in enumerate(ordered_values):
                segment_duration = max(0.0, value.end_at - value.start_at)
                segments_left = len(ordered_values) - segment_index
                if segment_index == len(ordered_values) - 1:
                    count = remaining_stages
                elif total_duration <= 0.0:
                    count = max(1, remaining_stages // segments_left)
                else:
                    count = round(
                        len(stages) * segment_duration / total_duration
                    )
                    count = max(1, count)
                    count = min(count, remaining_stages - (segments_left - 1))
                selected_stages = tuple(
                    stages[stage_cursor : stage_cursor + count]
                )
                if not selected_stages:
                    selected_stages = (stages[-1],)
                stage_cursor += len(selected_stages)
                remaining_stages = max(0, len(stages) - stage_cursor)
                split_bounds = _equal_active_time_splits(
                    item,
                    value.start_at,
                    value.end_at,
                    len(selected_stages),
                )
                for local_index, (stage, bounds) in enumerate(
                    zip(selected_stages, split_bounds),
                    start=1,
                ):
                    start_at, end_at = bounds
                    truncated = (
                        source_outcome == "TRUNCATED_BY_ABNORMAL"
                        and segment_index == len(ordered_values) - 1
                        and local_index == len(selected_stages)
                    )
                    occurrences.append(
                        {
                            "annotation_id": (
                                f"detail-draft:{attempt_id}:{stage.stage_id}:"
                                f"{segment_index + 1}:{local_index}"
                            ),
                            "coarse_attempt_id": attempt_id,
                            "progress_step_id": target,
                            "stage_id": stage.stage_id,
                            "occurrence_index_in_attempt": len(
                                [
                                    existing
                                    for existing in occurrences
                                    if existing["coarse_attempt_id"] == attempt_id
                                ]
                            )
                            + 1,
                            "start_at": start_at,
                            "end_at": end_at,
                            "outcome": (
                                "TRUNCATED_BY_ABNORMAL"
                                if truncated
                                else "SUCCESS"
                            ),
                            "confidence": 1.0,
                            "source_failure_fact_id": None,
                        }
                    )
        return {
            "schema": _DETAIL_DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            "base_annotation_set_id": base_annotation_set_id,
            "base_annotation_set_manifest_hash": (
                base_annotation_set_manifest_hash
            ),
            "base_task_stage_profile_hash": (
                item.progress.stage_profile.semantic_content_hash
            ),
            "detail_task_stage_profile_hash": (
                selection.stage_profile.semantic_content_hash
            ),
            "target_progress_step_id": target,
            "draft_revision": 0,
            "occurrences": occurrences,
            "failure_stage_by_attempt": failure_stage_by_attempt,
            "confirmed": False,
        }

    def _normalize_detail_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
        *,
        base_annotation_set: LoadedStageAnnotationSet | None,
        selection: DetailProfileSelection,
    ) -> dict[str, Any]:
        if not isinstance(draft, Mapping):
            raise RefinementAnnotationApplicationError(
                "Detail draft must be an object."
            )
        if draft.get("schema") != _DETAIL_DRAFT_SCHEMA:
            raise RefinementAnnotationApplicationError(
                "Unsupported detail annotation draft schema."
            )
        if str(draft.get("episode_uuid", "")).strip() != item.episode_uuid:
            raise RefinementAnnotationApplicationError(
                "Detail draft Episode identity changed."
            )
        expected = {
            "base_annotation_set_id": (
                None
                if base_annotation_set is None
                else base_annotation_set.binding.annotation_set_id
            ),
            "base_annotation_set_manifest_hash": (
                None
                if base_annotation_set is None
                else base_annotation_set.manifest.semantic_content_hash
            ),
            "base_task_stage_profile_hash": (
                item.progress.stage_profile.semantic_content_hash
            ),
            "detail_task_stage_profile_hash": (
                selection.stage_profile.semantic_content_hash
            ),
            "target_progress_step_id": selection.target_progress_step_id,
        }
        for key, value in expected.items():
            if draft.get(key) != value:
                raise RefinementAnnotationApplicationError(
                    f"Detail draft is bound to a different {key}."
                )
        raw_occurrences = draft.get("occurrences", ())
        if not isinstance(raw_occurrences, (list, tuple)) or any(
            not isinstance(value, Mapping) for value in raw_occurrences
        ):
            raise RefinementAnnotationApplicationError(
                "Detail occurrences must be a sequence of objects."
            )
        clock = item.progress.timeline.clock_domain_id
        try:
            contracts = tuple(
                StageOccurrenceAnnotationV1(
                    annotation_id=value.get("annotation_id", ""),
                    coarse_attempt_id=value.get("coarse_attempt_id", ""),
                    progress_step_id=value.get("progress_step_id", ""),
                    stage_id=value.get("stage_id", ""),
                    occurrence_index_in_attempt=value.get(
                        "occurrence_index_in_attempt",
                        0,
                    ),
                    start_at=value.get("start_at"),
                    end_at=value.get("end_at"),
                    clock_domain_id=clock,
                    outcome=value.get("outcome", ""),
                    confidence=value.get("confidence", 1.0),
                    source_failure_fact_id=value.get(
                        "source_failure_fact_id"
                    ),
                )
                for value in raw_occurrences
            )
            bundle = DetailStageAnnotationBundleV1(occurrences=contracts)
        except (TypeError, ValueError) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid detail stage occurrence: {exc}"
            ) from exc
        if any(
            value.progress_step_id != selection.target_progress_step_id
            for value in bundle.occurrences
        ):
            raise RefinementAnnotationApplicationError(
                "Detail occurrences must remain inside the selected coarse "
                "progress step."
            )
        raw_failure_map = draft.get("failure_stage_by_attempt", {})
        if not isinstance(raw_failure_map, Mapping):
            raise RefinementAnnotationApplicationError(
                "failure_stage_by_attempt must be an object."
            )
        failure_map: dict[str, str | None] = {}
        for attempt_id, stage_id in raw_failure_map.items():
            normalized_attempt_id = str(attempt_id).strip()
            if not normalized_attempt_id:
                raise RefinementAnnotationApplicationError(
                    "Detail failure Attempt identity must be non-empty."
                )
            if stage_id is not None and not str(stage_id).strip():
                raise RefinementAnnotationApplicationError(
                    "Detail failure stage identity must be non-empty or null."
                )
            failure_map[normalized_attempt_id] = (
                None if stage_id is None else str(stage_id).strip()
            )
        result = {
            "schema": _DETAIL_DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            **expected,
            "draft_revision": int(draft.get("draft_revision", 0)),
            "occurrences": [
                self._detail_occurrence_payload(value)
                for value in bundle.occurrences
            ],
            "failure_stage_by_attempt": failure_map,
            "confirmed": bool(draft.get("confirmed", False)),
        }
        for key in (
            "last_committed_detail_annotation_set_id",
            "last_committed_detail_manifest_hash",
        ):
            if key in draft:
                result[key] = draft[key]
        return result

    def _detail_bundle_from_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
    ) -> DetailStageAnnotationBundleV1:
        clock = item.progress.timeline.clock_domain_id
        try:
            return DetailStageAnnotationBundleV1(
                occurrences=tuple(
                    StageOccurrenceAnnotationV1(
                        annotation_id=value.get("annotation_id", ""),
                        coarse_attempt_id=value.get("coarse_attempt_id", ""),
                        progress_step_id=value.get("progress_step_id", ""),
                        stage_id=value.get("stage_id", ""),
                        occurrence_index_in_attempt=value.get(
                            "occurrence_index_in_attempt",
                            0,
                        ),
                        start_at=value.get("start_at"),
                        end_at=value.get("end_at"),
                        clock_domain_id=clock,
                        outcome=value.get("outcome", ""),
                        confidence=value.get("confidence", 1.0),
                        source_failure_fact_id=value.get(
                            "source_failure_fact_id"
                        ),
                    )
                    for value in draft.get("occurrences", ())
                )
            )
        except (TypeError, ValueError) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid detail stage annotation: {exc}"
            ) from exc

    @staticmethod
    def _detail_occurrence_field(value: Any, name: str) -> Any:
        return (
            value.get(name)
            if isinstance(value, Mapping)
            else getattr(value, name)
        )

    def _detail_failure_attempt_payload(
        self,
        item: ProgressAnnotationWorkItem,
        base_occurrences: Sequence[Any],
        selection: DetailProfileSelection,
        detail_draft: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        target = selection.target_progress_step_id
        by_attempt: dict[str, list[Any]] = {}
        for value in base_occurrences:
            progress_step_id = self._detail_occurrence_field(
                value,
                "progress_step_id",
            )
            source_failure_fact_id = self._detail_occurrence_field(
                value,
                "source_failure_fact_id",
            )
            if progress_step_id == target and source_failure_fact_id:
                by_attempt.setdefault(
                    self._detail_occurrence_field(
                        value,
                        "coarse_attempt_id",
                    ),
                    [],
                ).append(value)
        timeline_attempts = {
            value.attempt_id: value for value in item.progress.timeline.attempts
        }
        selected = detail_draft.get("failure_stage_by_attempt", {})
        stages = selection.stage_profile.stages_for_step(target)
        result: list[dict[str, Any]] = []
        for attempt_id, values in sorted(by_attempt.items()):
            source = timeline_attempts.get(attempt_id)
            failure_fact_id = next(
                self._detail_occurrence_field(
                    value,
                    "source_failure_fact_id",
                )
                for value in values
                if self._detail_occurrence_field(
                    value,
                    "source_failure_fact_id",
                )
                is not None
            )
            result.append(
                {
                    "attempt_id": attempt_id,
                    "progress_step_id": target,
                    "start_at": min(
                        self._detail_occurrence_field(value, "start_at")
                        for value in values
                    ),
                    "end_at": max(
                        self._detail_occurrence_field(value, "end_at")
                        for value in values
                    ),
                    "failure_fact_id": failure_fact_id,
                    "failure_reported_at": (
                        None if source is None else source.failure_reported_at
                    ),
                    "selected_stage_id": selected.get(attempt_id),
                    "options": [
                        self._detail_stage_payload(stage) for stage in stages
                    ],
                }
            )
        return result

    def _detail_payload(
        self,
        item: ProgressAnnotationWorkItem,
        *,
        base_annotation_set: LoadedStageAnnotationSet | None,
        base_occurrences: Sequence[Any] = (),
        selection: DetailProfileSelection | None,
        detail_annotation_set: LoadedDetailStageAnnotationSet | None,
        detail_draft: Mapping[str, Any] | None,
        reason: str | None,
    ) -> dict[str, Any]:
        profile_options, profile_options_reason = (
            self._detail_profile_options_payload(item)
        )
        selected_profile_selection_id = (
            None if selection is None else selection.selection_id
        )
        if selection is None:
            return {
                "available": False,
                "resolution_level": "COARSE_FALLBACK",
                "draft_state": "UNAVAILABLE",
                "reason": reason
                or profile_options_reason
                or "没有匹配的详细 profile。",
                "profile": None,
                "profile_options": profile_options,
                "selected_profile_selection_id": (
                    selected_profile_selection_id
                ),
                "base": None,
                "failure_attempts": [],
                "sealed": None,
            }
        profile = selection.stage_profile
        profile_payload = {
            "selection_id": selection.selection_id,
            "profile_id": profile.profile_id,
            "profile_version": profile.profile_version,
            "target_progress_step_id": selection.target_progress_step_id,
            "stages": [
                self._detail_stage_payload(stage)
                for stage in profile.stages_for_step(
                    selection.target_progress_step_id
                )
            ],
        }
        if base_annotation_set is None:
            return {
                "available": detail_draft is not None,
                "resolution_level": "COARSE_FALLBACK",
                "draft_state": (
                    "EDITING_UNSEALED"
                    if detail_draft is not None
                    else "PROFILE_PREVIEW"
                ),
                "reason": reason
                or (
                    "详细 profile 已匹配；粗、细标注可以独立编辑。"
                    "详细 overlay 会在粗标注版本确定后绑定。"
                    if detail_draft is not None
                    else "详细 profile 已匹配；正在生成可编辑的详细时间轴。"
                ),
                "profile": profile_payload,
                "profile_options": profile_options,
                "selected_profile_selection_id": (
                    selected_profile_selection_id
                ),
                "base": None,
                "failure_attempts": self._detail_failure_attempt_payload(
                    item,
                    base_occurrences,
                    selection,
                    detail_draft or {},
                ),
                "sealed": None,
            }
        sealed = (
            None
            if detail_annotation_set is None
            else {
                "annotation_set_id": (
                    detail_annotation_set.binding.annotation_set_id
                ),
                "manifest_hash": (
                    detail_annotation_set.manifest.semantic_content_hash
                ),
                "created_at": detail_annotation_set.binding.created_at,
                "annotator_id": detail_annotation_set.binding.annotator_id,
            }
        )
        draft_state = "EDITING"
        if detail_annotation_set is not None and detail_draft is not None:
            if (
                detail_draft.get("confirmed")
                and detail_draft.get(
                    "last_committed_detail_annotation_set_id"
                )
                == detail_annotation_set.binding.annotation_set_id
            ):
                draft_state = "SEALED"
        return {
            "available": True,
            "resolution_level": (
                "DETAIL_COMPLETE"
                if detail_annotation_set is not None
                else "COARSE_FALLBACK"
            ),
            "draft_state": draft_state,
            "reason": reason,
            "profile": profile_payload,
            "profile_options": profile_options,
            "selected_profile_selection_id": selected_profile_selection_id,
            "base": {
                "annotation_set_id": (
                    base_annotation_set.binding.annotation_set_id
                ),
                "manifest_hash": base_annotation_set.manifest.semantic_content_hash,
                "profile_hash": (
                    base_annotation_set.binding.task_stage_profile_hash
                ),
            },
            "failure_attempts": self._detail_failure_attempt_payload(
                item,
                base_annotation_set.bundle.occurrences,
                selection,
                detail_draft or {},
            ),
            "sealed": sealed,
        }

    @staticmethod
    def _report_operation_progress(
        callback: Callable[[int, str], None] | None,
        percent: int,
        message: str,
    ) -> None:
        if callback is not None:
            callback(max(0, min(100, int(percent))), str(message))

    def episode_detail(
        self,
        episode_uuid: str,
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        with self._episode_operation_lock:
            self._report_operation_progress(
                progress_callback,
                5,
                "正在确认 Episode 与编辑状态",
            )
            normalized = str(episode_uuid).strip()
            self._item(normalized)
            with self._lock:
                current = self._active_episode_uuid
            if current == normalized:
                self._report_operation_progress(
                    progress_callback,
                    35,
                    "正在读取阶段标注与样本时间轴",
                )
                detail = self._episode_detail_owned(normalized)
                self._report_operation_progress(
                    progress_callback,
                    95,
                    "Episode 数据已就绪",
                )
                return detail

            # Keep the previous Episode lock until the target detail has been
            # read successfully.  A remote read/draft failure must leave the
            # browser's still-visible Episode active and editable instead of
            # silently switching ownership behind the failed HTTP response.
            self.workspace.acquire_episode(
                episode_uuid=normalized,
                annotator_id=self.annotator_id,
            )
            self._report_operation_progress(
                progress_callback,
                30,
                "已取得 Episode 编辑锁，正在读取标注",
            )
            try:
                with self._summary_load_lock(normalized):
                    with self._lock:
                        # Do not trust a directory timestamp after taking a
                        # cross-machine lock: NFS attribute caches may briefly
                        # retain an old value. Preserve the original exact
                        # discovery behavior for every newly opened Episode.
                        self._annotation_sets_cache.pop(normalized, None)
                        self._annotation_summary_cache.pop(normalized, None)
                        self._annotation_root_fingerprints.pop(normalized, None)
                    self._detail_annotations.invalidate(normalized)
                    try:
                        detail = self._episode_detail_owned(normalized)
                    except Exception:
                        # Do not retain partially refreshed target state after a
                        # failed open; a later retry must start from storage.
                        with self._lock:
                            self._annotation_sets_cache.pop(normalized, None)
                            self._annotation_summary_cache.pop(normalized, None)
                            self._annotation_root_fingerprints.pop(
                                normalized,
                                None,
                            )
                            self._sample_axis_cache.pop(normalized, None)
                        self._detail_annotations.invalidate(normalized)
                        raise

                    if current is not None:
                        self.workspace.release_episode(episode_uuid=current)
                    with self._lock:
                        self._active_episode_uuid = normalized
                self._report_operation_progress(
                    progress_callback,
                    95,
                    "Episode 数据与编辑锁已就绪",
                )
                return detail
            except Exception as exc:
                with self._summary_load_lock(normalized):
                    with self._lock:
                        self._annotation_sets_cache.pop(normalized, None)
                        self._annotation_summary_cache.pop(normalized, None)
                        self._annotation_root_fingerprints.pop(normalized, None)
                        self._sample_axis_cache.pop(normalized, None)
                    self._detail_annotations.invalidate(normalized)
                try:
                    self.workspace.release_episode(episode_uuid=normalized)
                except StageAnnotationStoreError as release_exc:
                    raise RefinementAnnotationApplicationError(
                        f"Cannot open Episode {normalized!r}, and its temporary "
                        "editing lock could not be released. Verify the lock "
                        f"owner before retrying: {release_exc}"
                    ) from exc
                raise

    def _episode_detail_owned(self, episode_uuid: str) -> dict[str, Any]:
        item = self._item(episode_uuid)
        progress_manifest_name = (
            "migration_manifest.cjson"
            if isinstance(item.progress, LoadedRetrospectiveProgressEpisode)
            else "manifest.cjson"
        )
        required_sources = (
            (
                item.progress_episode_dir / progress_manifest_name,
                (self.dataset_root,),
            ),
            (
                item.parquet_path,
                (self.dataset_root, *self.allowed_source_roots),
            ),
            *(
                (
                    path,
                    (self.dataset_root, *self.allowed_source_roots),
                )
                for path in item.video_paths.values()
            ),
        )
        missing: list[str] = []
        for path, allowed_roots in required_sources:
            try:
                resolved = path.resolve(strict=True)
                if not any(
                    resolved.is_relative_to(root)
                    for root in allowed_roots
                ):
                    raise ValueError("source path escapes its admitted roots")
            except (FileNotFoundError, OSError, ValueError):
                missing.append(str(path))
                continue
            if not resolved.is_file():
                missing.append(str(path))
        if missing:
            raise RefinementAnnotationApplicationError(
                "Cannot open Episode because sealed source files are no "
                f"longer available: {', '.join(missing)}"
            )
        stage_profile = item.progress.stage_profile
        stage_entry = item.progress.stage_entry_binding
        assert stage_profile is not None
        assert stage_entry is not None
        # These three reads are independent and each can be dominated by
        # remote filesystem latency.  Overlap them without weakening the
        # Episode lock already held by the caller.
        with ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="annotation-episode-open",
        ) as executor:
            annotation_sets_future = executor.submit(
                self._sealed_annotation_sets,
                item,
            )
            sample_axis_future = executor.submit(self._sample_continuity, item)
            stored_draft_future = executor.submit(
                self.workspace.load_draft,
                episode_uuid=episode_uuid,
            )
            annotation_sets = annotation_sets_future.result()
            sample_continuity = sample_axis_future.result()
            sample_axis = sample_continuity.sample_axis
            try:
                stored_draft = stored_draft_future.result()
            except StageAnnotationStoreError as exc:
                raise RefinementAnnotationApplicationError(
                    "Cannot load the local annotation draft. The dataset "
                    f"storage may be disconnected or unreadable: {exc}"
                ) from exc
        draft = stored_draft
        if stored_draft is not None:
            self._mark_item_has_draft(episode_uuid)
        draft_persistence_state = "BOOTSTRAP_UNEDITED"
        if (
            self.annotation_mode == "SECOND_REVIEW"
            and (
                draft is None
                or draft.get("draft_annotation_mode", "STANDARD")
                != "SECOND_REVIEW"
            )
        ):
            draft = (
                self._draft_from_annotation_set(
                    item,
                    annotation_sets[-1],
                    draft_revision=(
                        0
                        if draft is None
                        else int(draft.get("draft_revision", 0)) + 1
                    ),
                )
                if annotation_sets
                else self._bootstrap_draft(item)
            )
        elif draft is None:
            draft = (
                self._draft_from_annotation_set(
                    item,
                    annotation_sets[-1],
                    draft_revision=0,
                )
                if annotation_sets
                else self._bootstrap_draft(item)
            )
        else:
            draft = self._normalize_draft(
                item,
                draft,
                sample_axis=sample_axis,
                allow_legacy_adjacent_recoverable_failures=True,
            )
        review_label_auto_cleared = bool(
            self.annotation_mode == "SECOND_REVIEW"
            and draft.get("episode_review_request") is not None
        )
        if self.annotation_mode == "SECOND_REVIEW":
            adjusted_draft = {
                **draft,
                "draft_annotation_mode": "SECOND_REVIEW",
                "episode_review_request": None,
            }
            # The prior immutable set remains excluded until this review draft
            # is explicitly sealed; only the mutable working label is cleared
            # automatically on open.
            if adjusted_draft != draft:
                try:
                    self.workspace.save_draft(
                        episode_uuid=episode_uuid,
                        draft=adjusted_draft,
                    )
                except StageAnnotationStoreError as exc:
                    raise RefinementAnnotationApplicationError(
                        "Cannot persist the second-review working draft. The "
                        "sealed annotation set was not changed; check that "
                        "the dataset storage is writable and connected: "
                        f"{exc}"
                    ) from exc
            draft = adjusted_draft
        latest_annotation_set = annotation_sets[-1] if annotation_sets else None
        annotation_summary = self._summarize_annotation_sets(annotation_sets)
        # Profile files are editable configuration.  Observe changes when an
        # operator explicitly opens an Episode, while sidebar summary batches
        # reuse the module inventory instead of rescanning once per row.
        self._detail_annotations.refresh_profiles()
        requested_detail_selection = draft.get(
            "detail_profile_selection_id"
        )
        if requested_detail_selection == "":
            # An explicit empty selection is different from a legacy draft
            # without this field: the former means the annotator deliberately
            # chose coarse-only annotation, while the latter keeps the old
            # single-candidate auto-resolution behavior.
            detail_selection = None
            detail_profile_reason = "当前仅使用粗粒度标注（已关闭详细补充）。"
        else:
            detail_selection, detail_profile_reason = (
                self._detail_profile_selection(
                    item,
                    selection_id=(
                        None
                        if requested_detail_selection is None
                        else str(requested_detail_selection)
                    ),
                )
            )
        detail_annotation_set: LoadedDetailStageAnnotationSet | None = None
        detail_reason = detail_profile_reason
        detail_draft: dict[str, Any] | None = None
        detail_payload_selection = detail_selection
        if detail_selection is not None:
            target_progress_step_id = detail_selection.target_progress_step_id
            target_has_stages = bool(
                detail_selection.stage_profile.stages_for_step(
                    target_progress_step_id
                )
            )
            if latest_annotation_set is not None:
                base_occurrences = latest_annotation_set.bundle.occurrences
                target_has_occurrences = any(
                    occurrence.progress_step_id == target_progress_step_id
                    for occurrence in base_occurrences
                )
            else:
                # Before the first coarse seal, the editable coarse draft is
                # the source of the target Attempt bounds.  It is not yet an
                # immutable base, but it is sufficient to start detail work.
                base_occurrences = self._bundle_from_draft(
                    item,
                    draft,
                ).occurrences
                target_has_occurrences = any(
                    occurrence.progress_step_id == target_progress_step_id
                    for occurrence in base_occurrences
                )
            if not target_has_stages or not target_has_occurrences:
                # A detail profile may match the dataset while this Episode
                # ended before its target Step was recorded.  Detail is an
                # additive overlay, so it must not make the valid coarse
                # annotation impossible to reopen.
                detail_reason = (
                    "当前粗粒度草稿没有覆盖详细 profile 的目标子任务；"
                    "继续使用粗粒度标注。"
                )
                detail_payload_selection = None
                draft = {**draft, "detail_overlay": None}
            elif latest_annotation_set is not None:
                detail_annotation_set, detail_load_reason = (
                    self._current_detail_annotation_set(
                        item,
                        base_annotation_set=latest_annotation_set,
                        selection=detail_selection,
                    )
                )
                if detail_load_reason is not None:
                    detail_reason = detail_load_reason
                stored_detail = draft.get("detail_overlay")
                if isinstance(stored_detail, Mapping):
                    try:
                        detail_draft = self._normalize_detail_draft(
                            item,
                            stored_detail,
                            base_annotation_set=latest_annotation_set,
                            selection=detail_selection,
                        )
                    except RefinementAnnotationApplicationError:
                        # A draft bound to an older coarse version must not be
                        # carried forward.  The base annotation remains intact.
                        detail_draft = None
                if detail_draft is None:
                    if detail_annotation_set is not None:
                        detail_draft = self._detail_draft_from_annotation_set(
                            item,
                            detail_annotation_set,
                            draft_revision=int(draft.get("draft_revision", 0)),
                        )
                    else:
                        detail_draft = self._bootstrap_detail_draft(
                            item,
                            latest_annotation_set,
                            detail_selection,
                        )
                draft = {
                    **draft,
                    "detail_overlay": detail_draft,
                }
            else:
                # No immutable coarse set is required to start editing.  Keep
                # this draft explicitly unbound; a later coarse seal will
                # attach the same detail draft to its new immutable base.
                stored_detail = draft.get("detail_overlay")
                if isinstance(stored_detail, Mapping):
                    try:
                        detail_draft = self._normalize_detail_draft(
                            item,
                            stored_detail,
                            base_annotation_set=None,
                            selection=detail_selection,
                        )
                    except RefinementAnnotationApplicationError:
                        detail_draft = None
                if detail_draft is None:
                    detail_draft = self._bootstrap_detail_draft_from_coarse_draft(
                        item,
                        draft,
                        detail_selection,
                    )
                draft = {
                    **draft,
                    "detail_overlay": detail_draft,
                }
        else:
            draft = {**draft, "detail_overlay": None}
        if (
            latest_annotation_set is not None
            and self._draft_matches_annotation_set(
                item,
                draft,
                latest_annotation_set,
            )
        ):
            draft_persistence_state = "SEALED_SNAPSHOT"
        elif stored_draft is not None or annotation_sets:
            draft_persistence_state = (
                "SECOND_REVIEW_UNSEALED"
                if self.annotation_mode == "SECOND_REVIEW"
                else "UNSEALED_DRAFT"
            )
        draft_requires_seal = draft_persistence_state in {
            "UNSEALED_DRAFT",
            "SECOND_REVIEW_UNSEALED",
        }
        detail_status = self._detail_annotations.status(
            item,
            annotation_sets,
            selection_id=(
                None
                if draft.get("detail_profile_selection_id") is None
                else str(draft.get("detail_profile_selection_id"))
            ),
        )
        annotation_status = build_episode_status_payload(
            annotation_summary,
            has_draft=item.has_draft,
            detail_state=detail_status.state,
            recording_pause_count=len(item.recording_pause_segments),
        )
        step_by_id = item.progress.profile.step_by_id
        return {
            **item.summary_payload(),
            "annotation_set_count": len(annotation_sets),
            "needs_second_review": annotation_summary.needs_second_review,
            "review_state": annotation_summary.review_state,
            "latest_annotation_mode": annotation_summary.latest_annotation_mode,
            "whole_episode_abnormal": (
                annotation_summary.whole_episode_abnormal
            ),
            "whole_episode_abnormal_reason": (
                annotation_summary.whole_episode_abnormal_reason
            ),
            "annotation_status": annotation_status,
            "annotation_summary_loaded": True,
            "annotation_mode": self.annotation_mode,
            "detail_profile_selection_id": (
                draft.get("detail_profile_selection_id")
            ),
            "review_label_auto_cleared": review_label_auto_cleared,
            "draft_persistence_state": draft_persistence_state,
            "draft_requires_seal": draft_requires_seal,
            "annotation_sets": [
                self._annotation_set_summary(value)
                for value in reversed(annotation_sets)
            ],
            "clock_domain_id": item.progress.timeline.clock_domain_id,
            # UI playback follows the formally accepted sample axis rather
            # than wall-time duration.  This keeps video/frame alignment exact
            # across intentional Hold/Pause gaps without changing any stored
            # timestamp or annotation contract.
            "sample_timestamps": list(sample_axis.cut_times),
            "sample_continuity_break_indices": list(
                sample_continuity.break_indices
            ),
            "sample_continuity_sources": list(sample_continuity.sources),
            "recording_pause_segments": [
                value.to_payload()
                for value in item.recording_pause_segments
            ],
            "entry_stage_id": stage_entry.entry_stage_id,
            "entry_condition": stage_entry.entry_condition,
            "capture_end_step_editor": (
                self._capture_end_step_editor_payload(item)
            ),
            "progress_steps": [
                {
                    **step.to_payload(),
                    "stages": [
                        stage.to_payload()
                        for stage in stage_profile.stages_for_step(
                            step.progress_step_id
                        )
                    ],
                }
                for step in item.progress.profile.steps
            ],
            "detail_annotation": self._detail_payload(
                item,
                base_annotation_set=latest_annotation_set,
                base_occurrences=base_occurrences if detail_selection is not None else (),
                selection=detail_payload_selection,
                detail_annotation_set=detail_annotation_set,
                detail_draft=detail_draft,
                reason=detail_reason,
            ),
            "detail_annotation_complete": detail_status.complete,
            "detail_annotation_state": detail_status.state,
            "attempts": [
                {
                    "attempt_id": attempt.attempt_id,
                    "progress_step_id": attempt.progress_step_id,
                    "progress_step_display_name": (
                        step_by_id[attempt.progress_step_id].display_name
                    ),
                    "start_at": attempt.start_at,
                    "end_at": attempt.end_at,
                    "outcome": attempt.outcome,
                    "failure_fact_id": attempt.failure_fact_id,
                    "failure_reported_at": attempt.failure_reported_at,
                }
                for attempt in item.progress.timeline.attempts
            ],
            "transitions": [
                {
                    "advance_fact_id": value.advance_fact_id,
                    "from_progress_step_id": value.from_progress_step_id,
                    "from_attempt_id": value.from_attempt_id,
                    "to_progress_step_id": value.to_progress_step_id,
                    "to_attempt_id": value.to_attempt_id,
                    "effective_at": value.effective_at,
                }
                for value in item.progress.timeline.transitions
            ],
            "media_urls": {
                camera: f"/media/{episode_uuid}/{camera}"
                for camera in item.video_paths
            },
            "draft": draft,
        }

    @staticmethod
    def _summarize_annotation_sets(
        values: Sequence[LoadedStageAnnotationSet],
    ) -> EpisodeAnnotationSummary:
        return summarize_annotation_sets(values)

    def _episode_summary(
        self,
        item: ProgressAnnotationWorkItem,
        *,
        load_annotations: bool,
    ) -> dict[str, Any]:
        with self._lock:
            annotation_summary = self._annotation_summary_cache.get(
                item.episode_uuid
            )
            if annotation_summary is not None:
                self._annotation_summary_cache.move_to_end(item.episode_uuid)
        annotation_sets: tuple[LoadedStageAnnotationSet, ...] | None = None
        if annotation_summary is None and load_annotations:
            annotation_sets = self._sealed_annotation_sets(item)
            annotation_summary = self._summarize_annotation_sets(
                annotation_sets
            )
        detail_annotation_complete: bool | None = None
        detail_annotation_state: str | None = None
        if load_annotations and annotation_summary is not None:
            if annotation_sets is None:
                annotation_sets = self._sealed_annotation_sets(item)
            selection_id = self._persisted_detail_profile_selection_id(item)
            detail_status = self._detail_annotations.status(
                item,
                annotation_sets,
                selection_id=selection_id,
            )
            detail_annotation_complete = detail_status.complete
            detail_annotation_state = detail_status.state
        if annotation_summary is None:
            set_count = item.annotation_set_count
            needs_second_review = None
            review_state = None
            latest_annotation_mode = None
            whole_episode_abnormal = None
            whole_episode_abnormal_reason = None
        else:
            set_count = annotation_summary.annotation_set_count
            needs_second_review = annotation_summary.needs_second_review
            review_state = annotation_summary.review_state
            latest_annotation_mode = (
                annotation_summary.latest_annotation_mode
            )
            whole_episode_abnormal = (
                annotation_summary.whole_episode_abnormal
            )
            whole_episode_abnormal_reason = (
                annotation_summary.whole_episode_abnormal_reason
            )
        return {
            **item.summary_payload(),
            "annotation_set_count": set_count,
            "needs_second_review": needs_second_review,
            "review_state": review_state,
            "latest_annotation_mode": latest_annotation_mode,
            "whole_episode_abnormal": whole_episode_abnormal,
            "whole_episode_abnormal_reason": (
                whole_episode_abnormal_reason
            ),
            "detail_annotation_complete": detail_annotation_complete,
            "detail_annotation_state": detail_annotation_state,
            "annotation_summary_loaded": annotation_summary is not None,
            "annotation_status": build_episode_status_payload(
                annotation_summary,
                has_draft=item.has_draft,
                detail_state=detail_annotation_state,
                recording_pause_count=len(item.recording_pause_segments),
            ),
        }

    def reopen_sealed_annotation(
        self,
        episode_uuid: str,
        *,
        annotation_set_id: str | None = None,
    ) -> dict[str, Any]:
        with self._episode_operation_lock:
            self._require_active_episode(episode_uuid)
            return self._reopen_sealed_annotation_owned(
                episode_uuid,
                annotation_set_id=annotation_set_id,
            )

    def _reopen_sealed_annotation_owned(
        self,
        episode_uuid: str,
        *,
        annotation_set_id: str | None = None,
    ) -> dict[str, Any]:
        """Materialize one immutable set as an unsaved editable draft.

        This is deliberately read-only with respect to persistence.  The
        existing local draft remains on disk until the annotator explicitly
        saves or publishes the returned replacement.
        """

        item = self._item(episode_uuid)
        available = self._sealed_annotation_sets(item)
        requested = str(annotation_set_id or "").strip()
        selected = next(
            (
                value
                for value in available
                if value.binding.annotation_set_id == requested
            ),
            None,
        )
        if not requested and available:
            selected = available[-1]
        if selected is None:
            raise RefinementAnnotationApplicationError(
                "Requested sealed annotation set is unavailable or invalid."
            )
        current_draft = self.workspace.load_draft(
            episode_uuid=item.episode_uuid
        )
        draft_revision = (
            int(current_draft.get("draft_revision", 0)) + 1
            if current_draft is not None
            else 0
        )
        draft = self._draft_from_annotation_set(
            item,
            selected,
            draft_revision=draft_revision,
        )
        review_label_auto_cleared = bool(
            self.annotation_mode == "SECOND_REVIEW"
            and draft.get("episode_review_request") is not None
        )
        if self.annotation_mode == "SECOND_REVIEW":
            draft["draft_annotation_mode"] = "SECOND_REVIEW"
            draft["episode_review_request"] = None
        return {
            "source_annotation_set": self._annotation_set_summary(selected),
            "review_label_auto_cleared": review_label_auto_cleared,
            "draft": draft,
        }

    def media_path(self, episode_uuid: str, camera_key: str) -> Path:
        item = self._item(episode_uuid)
        try:
            return item.video_paths[str(camera_key)]
        except KeyError as exc:
            raise RefinementAnnotationApplicationError(
                f"Unknown camera key {camera_key!r}."
            ) from exc

    def save_draft(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        with self._episode_operation_lock:
            self._require_active_episode(episode_uuid)
            return self._save_draft_owned(episode_uuid, draft)

    def _save_draft_owned(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        item = self._item(episode_uuid)
        normalized = self._normalize_draft(item, draft)
        try:
            self.workspace.save_draft(
                episode_uuid=episode_uuid,
                draft=normalized,
            )
        except StageAnnotationStoreError as exc:
            raise RefinementAnnotationApplicationError(
                "Cannot persist the annotation draft. The in-browser edits "
                "remain available; check that the dataset storage is "
                f"writable and connected, then retry: {exc}"
            ) from exc
        self._mark_item_has_draft(episode_uuid)
        return normalized

    def add_recoverable_failure(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        onset_at: object,
        recovered_at: object,
        note: object = None,
    ) -> dict[str, Any]:
        """Add one sample-aligned analysis observation without persisting it."""

        with self._episode_operation_lock:
            self._require_active_episode(episode_uuid)
            item = self._item(episode_uuid)
            sample_axis = self._sample_axis(item)
            normalized = self._normalize_draft(
                item,
                draft,
                sample_axis=sample_axis,
            )
            if note is not None and not isinstance(note, str):
                raise RefinementAnnotationApplicationError(
                    "Recoverable-failure note must be a string or null."
                )
            annotation_id = f"recoverable-failure:{uuid.uuid4().hex}"
            requested = [
                *normalized.get("recoverable_failures", ()),
                {
                    "annotation_id": annotation_id,
                    "onset_at": onset_at,
                    "recovered_at": recovered_at,
                    "note": None if note is None else note.strip() or None,
                },
            ]
            normalized_values = self._normalize_recoverable_failures(
                item,
                normalized.get("occurrences", ()),
                requested,
                sample_axis=sample_axis,
            )
            normalized["recoverable_failures"] = normalized_values
            added = next(
                value
                for value in normalized_values
                if value["annotation_id"] == annotation_id
            )
            return {"draft": normalized, "recoverable_failure": added}

    def reflow_failed_attempt_stage(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        attempt_id: str,
        failed_stage_id: str,
    ) -> dict[str, Any]:
        with self._episode_operation_lock:
            self._require_active_episode(episode_uuid)
            return self._reflow_failed_attempt_stage_owned(
                episode_uuid,
                draft,
                attempt_id=attempt_id,
                failed_stage_id=failed_stage_id,
            )

    def reflow_capture_end_step(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        end_progress_step_id: str,
    ) -> dict[str, Any]:
        """Rebuild only the annotation projection of an ambiguous final stop."""

        with self._episode_operation_lock:
            self._require_active_episode(episode_uuid)
            item = self._item(episode_uuid)
            sample_axis = self._sample_axis(item)
            normalized = self._normalize_draft(
                item,
                draft,
                sample_axis=sample_axis,
                allow_legacy_adjacent_recoverable_failures=True,
            )
            occurrences, capture_end_step = (
                self._rebuild_capture_end_step_projection(
                    item,
                    occurrences=tuple(normalized.get("occurrences", ())),
                    abnormal_intervals=tuple(
                        normalized.get("abnormal_intervals", ())
                    ),
                    recoverable_failures=tuple(
                        normalized.get("recoverable_failures", ())
                    ),
                    end_progress_step_id=end_progress_step_id,
                )
            )
            rebuilt = self._normalize_draft(
                item,
                {
                    **normalized,
                    "occurrences": occurrences,
                    "capture_end_step": capture_end_step,
                    "deletion_history": [],
                },
                sample_axis=sample_axis,
            )
            bundle = self._bundle_from_draft(item, rebuilt)
            assert item.progress.stage_profile is not None
            resolve_stage_training_index(
                progress_profile=item.progress.profile,
                progress_binding=item.progress.binding,
                stage_profile=item.progress.stage_profile,
                stage_entry_binding=item.progress.stage_entry_binding,
                timeline=item.progress.timeline,
                annotations=bundle,
                sample_axis=sample_axis,
            )
            return {"draft": rebuilt}

    def _reflow_failed_attempt_stage_owned(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        attempt_id: str,
        failed_stage_id: str,
    ) -> dict[str, Any]:
        """Rebuild one failed Attempt and its retry suffix in profile order.

        This is an editor operation, not a new dataset fact.  Capture-time
        Failure identity and timing remain immutable; only the mutable stage
        occurrences are rematerialized.  Reflow continues through consecutive
        retry Attempts for the same coarse progress step so a changed terminal
        failed stage cannot leave a stale, backwards retry label behind.
        """

        item = self._item(episode_uuid)
        sample_axis = self._sample_axis(item)
        normalized = self._normalize_draft(
            item,
            draft,
            sample_axis=sample_axis,
        )
        stage_profile = item.progress.stage_profile
        stage_entry = item.progress.stage_entry_binding
        assert stage_profile is not None
        assert stage_entry is not None

        requested_attempt_id = str(attempt_id).strip()
        requested_stage_id = str(failed_stage_id).strip()
        attempts = tuple(item.progress.timeline.attempts)
        target_index = next(
            (
                index
                for index, attempt in enumerate(attempts)
                if attempt.attempt_id == requested_attempt_id
            ),
            None,
        )
        if target_index is None:
            raise RefinementAnnotationApplicationError(
                "Failed-stage edit references an unknown Attempt."
            )
        target_attempt = attempts[target_index]
        if (
            target_attempt.outcome != "FAILURE"
            or target_attempt.failure_fact_id is None
            or target_attempt.failure_reported_at is None
        ):
            raise RefinementAnnotationApplicationError(
                "Only a capture-time failed Attempt can select a failed stage."
            )

        stages = stage_profile.stages_for_step(
            target_attempt.progress_step_id
        )
        stage_index = {stage.stage_id: index for index, stage in enumerate(stages)}
        if requested_stage_id not in stage_index:
            raise RefinementAnnotationApplicationError(
                "Selected failed stage does not belong to the Attempt's "
                "progress step."
            )

        raw_occurrences = [
            dict(value) for value in normalized.get("occurrences", ())
        ]
        by_attempt: dict[str, list[dict[str, Any]]] = {}
        for occurrence in raw_occurrences:
            by_attempt.setdefault(
                str(occurrence.get("coarse_attempt_id", "")),
                [],
            ).append(occurrence)
        for values in by_attempt.values():
            values.sort(
                key=lambda value: (
                    float(value.get("start_at", 0.0)),
                    float(value.get("end_at", 0.0)),
                    str(value.get("annotation_id", "")),
                )
            )

        expected_by_step = {
            step.progress_step_id: 0
            for step in item.progress.profile.steps
        }
        entry_definition = stage_profile.stage_by_id[
            stage_entry.entry_stage_id
        ]
        expected_by_step[entry_definition.progress_step_id] = (
            entry_definition.ordinal - 1
        )

        for attempt in attempts[:target_index]:
            attempt_stages = stage_profile.stages_for_step(
                attempt.progress_step_id
            )
            attempt_index = {
                stage.stage_id: index
                for index, stage in enumerate(attempt_stages)
            }
            expected = expected_by_step[attempt.progress_step_id]
            values = by_attempt.get(attempt.attempt_id, ())
            if (
                not values
                and attempt.progress_step_id
                == target_attempt.progress_step_id
            ):
                raise RefinementAnnotationApplicationError(
                    "Cannot infer the failed-stage entry because an earlier "
                    "Attempt in the same progress step is unannotated."
                )
            for occurrence in values:
                outcome = str(occurrence.get("outcome", "")).upper()
                if outcome == "TRUNCATED_BY_ABNORMAL":
                    raise RefinementAnnotationApplicationError(
                        "Resolve abnormal-truncated stage edits before changing "
                        "a later failed stage."
                    )
                current = attempt_index.get(
                    str(occurrence.get("stage_id", ""))
                )
                if current is None or current != expected:
                    raise RefinementAnnotationApplicationError(
                        "Existing stage draft is already out of profile order; "
                        "it must be corrected before reflowing a Failure."
                    )
                if outcome == "SUCCESS":
                    expected += 1
                elif outcome == "FAILURE":
                    break
            expected_by_step[attempt.progress_step_id] = expected

        target_entry_index = expected_by_step[target_attempt.progress_step_id]
        requested_failure_index = stage_index[requested_stage_id]
        if requested_failure_index < target_entry_index:
            raise RefinementAnnotationApplicationError(
                "A failed stage cannot move before the Attempt entry stage."
            )

        affected_attempts = []
        for attempt in attempts[target_index:]:
            if attempt.progress_step_id != target_attempt.progress_step_id:
                break
            affected_attempts.append(attempt)
        if not affected_attempts:
            raise RefinementAnnotationApplicationError(
                "No failed Attempt suffix is available for reflow."
            )

        abnormal_intervals = tuple(
            normalized.get("abnormal_intervals", ())
        )

        def attempt_bounds(
            attempt: Any,
            values: Sequence[Mapping[str, Any]],
        ) -> tuple[float, float]:
            start = (
                min(float(value["start_at"]) for value in values)
                if values
                else float(attempt.start_at)
            )
            end_value = (
                attempt.failure_reported_at
                if attempt.outcome == "FAILURE"
                else attempt.end_at
            )
            if end_value is None:
                raise RefinementAnnotationApplicationError(
                    "A stage suffix cannot be rebuilt for an open Attempt."
                )
            end = (
                max(float(value["end_at"]) for value in values)
                if values and attempt.outcome != "FAILURE"
                else float(end_value)
            )
            if end <= start:
                raise RefinementAnnotationApplicationError(
                    "A stage suffix Attempt has no positive time span."
                )
            if any(
                start < float(interval.get("end_at", start))
                and float(interval.get("start_at", end)) < end
                for interval in abnormal_intervals
            ):
                raise RefinementAnnotationApplicationError(
                    "Temporarily remove or finish the overlapping abnormal "
                    "interval before changing the failed stage."
                )
            return start, end

        def partition_bounds(
            *,
            start: float,
            end: float,
            count: int,
        ) -> tuple[float, ...]:
            samples = tuple(
                float(value)
                for value in sample_axis.cut_times
                if start <= float(value) <= end
            )
            if len(samples) < count:
                raise RefinementAnnotationApplicationError(
                    "The Attempt has too few accepted samples to assign at "
                    "least one sample to every selected stage."
                )
            boundaries = [start]
            for boundary_index in range(1, count):
                split_index = math.ceil(
                    boundary_index * len(samples) / count
                )
                split_index = min(
                    max(1, split_index),
                    len(samples) - 1,
                )
                boundary = (
                    samples[split_index - 1] + samples[split_index]
                ) / 2.0
                if not boundaries[-1] < boundary < end:
                    raise RefinementAnnotationApplicationError(
                        "Accepted sample times cannot form non-empty ordered "
                        "stage intervals."
                    )
                boundaries.append(boundary)
            boundaries.append(end)
            return tuple(boundaries)

        replacement_by_attempt: dict[str, list[dict[str, Any]]] = {}
        expected = target_entry_index
        changed_attempt_ids: list[str] = []
        for suffix_index, attempt in enumerate(affected_attempts):
            existing = by_attempt.get(attempt.attempt_id, ())
            start, end = attempt_bounds(attempt, existing)
            tail_outcome = "SUCCESS"
            failure_fact_id: str | None = None
            if suffix_index == 0:
                terminal_index = requested_failure_index
                tail_outcome = "FAILURE"
                failure_fact_id = target_attempt.failure_fact_id
            elif attempt.outcome == "FAILURE":
                captured_failure = next(
                    (
                        value
                        for value in existing
                        if str(value.get("source_failure_fact_id", ""))
                        == str(attempt.failure_fact_id)
                    ),
                    None,
                )
                existing_terminal = (
                    stage_index.get(str(captured_failure.get("stage_id", "")))
                    if captured_failure is not None
                    else None
                )
                terminal_index = max(
                    expected,
                    expected
                    if existing_terminal is None
                    else existing_terminal,
                )
                tail_outcome = "FAILURE"
                failure_fact_id = attempt.failure_fact_id
            elif attempt.outcome == "SUCCESS":
                terminal_index = len(stages) - 1
            else:
                existing_terminal = (
                    stage_index.get(str(existing[-1].get("stage_id", "")))
                    if existing
                    else None
                )
                terminal_index = max(
                    expected,
                    expected
                    if existing_terminal is None
                    else existing_terminal,
                )
                if (
                    existing
                    and str(existing[-1].get("outcome", "")).upper()
                    == "FAILURE"
                ):
                    tail_outcome = "FAILURE"

            if expected >= len(stages) or terminal_index >= len(stages):
                raise RefinementAnnotationApplicationError(
                    "Retry suffix would continue after the final stage."
                )
            selected = stages[expected : terminal_index + 1]
            bounds = partition_bounds(
                start=start,
                end=end,
                count=len(selected),
            )
            rebuilt: list[dict[str, Any]] = []
            for occurrence_index, stage in enumerate(selected, start=1):
                is_tail = occurrence_index == len(selected)
                outcome = tail_outcome if is_tail else "SUCCESS"
                rebuilt.append(
                    {
                        "annotation_id": (
                            f"draft:{attempt.attempt_id}:reflow:"
                            f"{occurrence_index}:{uuid.uuid4().hex[:8]}"
                        ),
                        "coarse_attempt_id": attempt.attempt_id,
                        "progress_step_id": attempt.progress_step_id,
                        "stage_id": stage.stage_id,
                        "occurrence_index_in_attempt": occurrence_index,
                        "start_at": bounds[occurrence_index - 1],
                        "end_at": bounds[occurrence_index],
                        "outcome": outcome,
                        "confidence": 1.0,
                        "source_failure_fact_id": (
                            failure_fact_id
                            if is_tail and outcome == "FAILURE"
                            else None
                        ),
                    }
                )
            replacement_by_attempt[attempt.attempt_id] = rebuilt
            changed_attempt_ids.append(attempt.attempt_id)
            expected = (
                terminal_index
                if tail_outcome == "FAILURE"
                else terminal_index + 1
            )

        retained = [
            value
            for value in raw_occurrences
            if str(value.get("coarse_attempt_id", ""))
            not in replacement_by_attempt
        ]
        rebuilt_occurrences = retained + [
            occurrence
            for attempt in affected_attempts
            for occurrence in replacement_by_attempt.get(
                attempt.attempt_id,
                (),
            )
        ]
        rebuilt_occurrences.sort(
            key=lambda value: (
                float(value.get("start_at", 0.0)),
                float(value.get("end_at", 0.0)),
                str(value.get("annotation_id", "")),
            )
        )
        result = {
            **normalized,
            "occurrences": rebuilt_occurrences,
            "deletion_history": [],
        }
        result["failure_triggers"] = self._normalize_failure_triggers(
            item,
            rebuilt_occurrences,
            tuple(normalized.get("failure_triggers", ())),
            sample_axis=sample_axis,
        )
        return {
            "draft": self._normalize_draft(
                item,
                result,
                sample_axis=sample_axis,
            ),
            "changed_attempt_ids": tuple(changed_attempt_ids),
        }

    def commit(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        with self._episode_operation_lock:
            self._report_operation_progress(
                progress_callback,
                5,
                "正在确认 Episode 编辑锁",
            )
            self._require_active_episode(episode_uuid)
            return self._commit_owned(
                episode_uuid,
                draft,
                progress_callback=progress_callback,
            )

    def commit_detail(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        with self._episode_operation_lock:
            self._report_operation_progress(
                progress_callback,
                5,
                "正在确认 Episode 编辑锁",
            )
            self._require_active_episode(episode_uuid)
            return self._commit_detail_owned(
                episode_uuid,
                draft,
                progress_callback=progress_callback,
            )

    def _commit_detail_owned(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        item = self._item(episode_uuid)
        self._report_operation_progress(
            progress_callback,
            15,
            "正在校验粗标注版本与详细时间轴",
        )
        normalized = self._normalize_draft(item, draft)
        annotation_sets = self._sealed_annotation_sets(item)
        selection, profile_reason = self._detail_profile_selection(
            item,
            selection_id=(
                None
                if normalized.get("detail_profile_selection_id") is None
                else str(normalized.get("detail_profile_selection_id"))
            ),
        )
        if selection is None:
            raise RefinementAnnotationApplicationError(
                profile_reason
                or "No detail stage profile is selected for this Episode."
            )
        raw_detail = normalized.get("detail_overlay")
        if not isinstance(raw_detail, Mapping):
            raise RefinementAnnotationApplicationError(
                "No detail annotation draft is available for this Episode."
            )
        if not annotation_sets:
            detail_draft = self._normalize_detail_draft(
                item,
                raw_detail,
                base_annotation_set=None,
                selection=selection,
            )
            if not detail_draft.get("confirmed", False):
                raise RefinementAnnotationApplicationError(
                    "Confirm the detail annotation after resolving every "
                    "target Failure Attempt."
                )
            pending_draft = {
                **normalized,
                "detail_overlay": detail_draft,
            }
            try:
                self.workspace.save_draft(
                    episode_uuid=episode_uuid,
                    draft=pending_draft,
                )
            except StageAnnotationStoreError as exc:
                raise RefinementAnnotationApplicationError(
                    "The detailed draft could not be saved before the coarse "
                    f"annotation is sealed: {exc}"
                ) from exc
            self._mark_item_has_draft(episode_uuid)
            self._report_operation_progress(
                progress_callback,
                97,
                "详细草稿已保存，等待粗标注版本",
            )
            return {
                "detail_annotation_set_id": None,
                "detail_manifest_hash": None,
                "detail_annotation": self._detail_payload(
                    item,
                    base_annotation_set=None,
                    base_occurrences=self._bundle_from_draft(
                        item,
                        normalized,
                    ).occurrences,
                    selection=selection,
                    detail_annotation_set=None,
                    detail_draft=detail_draft,
                    reason=(
                        "详细草稿已保存；粗标注可稍后独立封存，"
                        "之后再发布 detail overlay。"
                    ),
                ),
                "draft": pending_draft,
                "draft_cache_warning": None,
                "pending_coarse": True,
            }
        base_annotation_set = annotation_sets[-1]
        if not self._draft_matches_annotation_set(
            item,
            normalized,
            base_annotation_set,
        ):
            raise RefinementAnnotationApplicationError(
                "The coarse annotation changed after this detail draft was "
                "opened. Reopen the Episode and reapply the detail edits."
            )
        detail_draft = self._normalize_detail_draft(
            item,
            raw_detail,
            base_annotation_set=base_annotation_set,
            selection=selection,
        )
        if not detail_draft.get("confirmed", False):
            raise RefinementAnnotationApplicationError(
                "Confirm the detail annotation after resolving every target "
                "Failure Attempt."
            )
        detail_bundle = self._detail_bundle_from_draft(item, detail_draft)
        assert item.progress.stage_profile is not None
        annotation_set_id = (
            f"detail-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        )
        binding = DetailStageAnnotationSetBindingV1(
            annotation_set_id=annotation_set_id,
            episode_capture_identity_hash=item.episode_identity_hash,
            source_progress_manifest_hash=item.progress_manifest_hash,
            base_annotation_set_id=base_annotation_set.binding.annotation_set_id,
            base_annotation_set_manifest_hash=(
                base_annotation_set.manifest.semantic_content_hash
            ),
            base_task_stage_profile_hash=(
                item.progress.stage_profile.semantic_content_hash
            ),
            detail_task_stage_profile_hash=(
                selection.stage_profile.semantic_content_hash
            ),
            target_progress_step_id=selection.target_progress_step_id,
            bound_clock_domain_id=item.progress.timeline.clock_domain_id,
            annotator_id=self.annotator_id,
            created_at=time.time(),
            created_clock_domain_id=LOCAL_WALL_CLOCK,
        )
        self._report_operation_progress(
            progress_callback,
            45,
            "正在校验详细 Attempt 覆盖与 Failure 归属",
        )
        try:
            validate_detail_stage_annotation_set(
                source_progress_manifest=item.progress.manifest,
                base_binding=base_annotation_set.binding,
                base_bundle=base_annotation_set.bundle,
                base_manifest=base_annotation_set.manifest,
                base_stage_profile=item.progress.stage_profile,
                detail_stage_profile=selection.stage_profile,
                binding=binding,
                bundle=detail_bundle,
            )
            self.workspace.require_episode(episode_uuid=episode_uuid)
            report = commit_detail_stage_annotation_set(
                dataset_root=self.dataset_root,
                episode_uuid=episode_uuid,
                source_progress_manifest=item.progress.manifest,
                base_annotation_set=base_annotation_set,
                base_stage_profile=item.progress.stage_profile,
                detail_stage_profile=selection.stage_profile,
                binding=binding,
                bundle=detail_bundle,
            )
        except (DetailStageAnnotationStoreError, TypeError, ValueError) as exc:
            raise RefinementAnnotationApplicationError(
                "The detail annotation was not published; the coarse "
                f"annotation remains unchanged: {exc}"
            ) from exc
        committed_detail_draft = {
            **detail_draft,
            "confirmed": True,
            "last_committed_detail_annotation_set_id": annotation_set_id,
            "last_committed_detail_manifest_hash": (
                report.manifest.semantic_content_hash
            ),
        }
        committed_draft = {
            **normalized,
            "detail_overlay": committed_detail_draft,
        }
        draft_cache_warning = None
        try:
            self.workspace.save_draft(
                episode_uuid=episode_uuid,
                draft=committed_draft,
                durable=False,
            )
        except StageAnnotationStoreError as exc:
            draft_cache_warning = (
                "The detail annotation was committed, but the local draft "
                "cache could not be updated. Reopening will recover it from "
                f"the immutable detail set: {exc}"
            )
        loaded = LoadedDetailStageAnnotationSet(
            annotation_dir=report.annotation_dir,
            binding=binding,
            bundle=detail_bundle,
            manifest=report.manifest,
        )
        self._detail_annotations.invalidate(episode_uuid)
        self._mark_item_has_draft(episode_uuid)
        self._report_operation_progress(
            progress_callback,
            97,
            "详细标注已封存",
        )
        return {
            "detail_annotation_set_id": annotation_set_id,
            "detail_manifest_hash": report.manifest.semantic_content_hash,
            "detail_annotation": self._detail_payload(
                item,
                base_annotation_set=base_annotation_set,
                selection=selection,
                detail_annotation_set=loaded,
                detail_draft=committed_detail_draft,
                reason=None,
            ),
            "draft": committed_draft,
            "draft_cache_warning": draft_cache_warning,
        }

    def _commit_owned(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
        *,
        progress_callback: Callable[[int, str], None] | None = None,
    ) -> dict[str, Any]:
        item = self._item(episode_uuid)
        self._report_operation_progress(
            progress_callback,
            15,
            "正在规范化并校验时间轴",
        )
        normalized = self._normalize_draft(item, draft)
        self._validate_review_request_transition(item, normalized)
        normalized = self._project_occurrences_around_abnormal_intervals(
            item,
            normalized,
        )
        bundle = self._bundle_from_draft(item, normalized)
        sample_axis = self._sample_axis(item)
        self._report_operation_progress(
            progress_callback,
            32,
            "正在计算训练覆盖与排除区间",
        )
        assert item.progress.stage_profile is not None
        index = resolve_stage_training_index(
            progress_profile=item.progress.profile,
            progress_binding=item.progress.binding,
            stage_profile=item.progress.stage_profile,
            stage_entry_binding=item.progress.stage_entry_binding,
            timeline=item.progress.timeline,
            annotations=bundle,
            sample_axis=sample_axis,
        )
        annotation_set_id = (
            f"stage-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        )
        binding = StageAnnotationSetBindingV1(
            annotation_set_id=annotation_set_id,
            episode_capture_identity_hash=item.episode_identity_hash,
            source_progress_manifest_hash=item.progress_manifest_hash,
            task_stage_profile_hash=(
                item.progress.stage_profile.semantic_content_hash
            ),
            bound_clock_domain_id=item.progress.timeline.clock_domain_id,
            annotator_id=self.annotator_id,
            created_at=time.time(),
            created_clock_domain_id=LOCAL_WALL_CLOCK,
            annotation_mode=self.annotation_mode,
        )
        trigger_set = self._failure_trigger_set_from_draft(
            item,
            normalized,
            bundle=bundle,
            binding=binding,
            sample_axis=sample_axis,
        )
        recoverable_failure_set = self._recoverable_failure_set_from_draft(
            item,
            normalized,
            bundle=bundle,
            binding=binding,
            sample_axis=sample_axis,
        )
        annotation_dir = (
            self.dataset_root
            / STAGE_ANNOTATION_RELATIVE_ROOT
            / item.episode_uuid
            / annotation_set_id
        )
        self._report_operation_progress(
            progress_callback,
            48,
            "正在写入不可变附加标注",
        )
        try:
            self.workspace.require_episode(episode_uuid=episode_uuid)
            trigger_report = commit_failure_trigger_annotation_set(
                annotation_dir=annotation_dir,
                annotation_set=trigger_set,
            )
            recoverable_failure_report = None
            if recoverable_failure_set.annotations:
                recoverable_failure_report = (
                    commit_recoverable_failure_annotation_set(
                        annotation_dir=annotation_dir,
                        annotation_set=recoverable_failure_set,
                    )
                )
        except (FailureTriggerStoreError, RecoverableFailureStoreError) as exc:
            raise RefinementAnnotationApplicationError(
                "Cannot publish the immutable annotation sidecars. The stage "
                "manifest was not published, so no partial annotation set is "
                f"visible. The in-browser edits remain available: {exc}"
            ) from exc
        self._report_operation_progress(
            progress_callback,
            66,
            "附加标注已写入，正在原子发布阶段标注",
        )
        report = self._commit_stage_annotation_set_with_retry(
            item=item,
            binding=binding,
            bundle=bundle,
        )
        self._report_operation_progress(
            progress_callback,
            88,
            "正式标注已封存，正在更新工作缓存",
        )
        new_base_annotation_set = LoadedStageAnnotationSet(
            annotation_dir=report.annotation_dir,
            binding=binding,
            bundle=bundle,
            manifest=report.manifest,
        )
        rebased_detail_draft: dict[str, Any] | None = None
        rebased_detail_reason: str | None = None
        rebased_selection_id = normalized.get("detail_profile_selection_id")
        if rebased_selection_id == "":
            rebased_detail_selection = None
            rebased_profile_reason = (
                "当前仅使用粗粒度标注（已关闭详细补充）。"
            )
        else:
            rebased_detail_selection, rebased_profile_reason = (
                self._detail_profile_selection(
                    item,
                    selection_id=(
                        None
                        if rebased_selection_id is None
                        else str(rebased_selection_id)
                    ),
                )
            )
        if rebased_detail_selection is not None:
            pending_detail = normalized.get("detail_overlay")
            if isinstance(pending_detail, Mapping):
                try:
                    was_unbound = (
                        pending_detail.get("base_annotation_set_id") is None
                    )
                    rebased_input = {
                        **pending_detail,
                        "base_annotation_set_id": (
                            new_base_annotation_set.binding.annotation_set_id
                        ),
                        "base_annotation_set_manifest_hash": (
                            new_base_annotation_set.manifest.semantic_content_hash
                        ),
                    }
                    if not was_unbound:
                        # A coarse edit creates a new immutable base.  An
                        # already-bound detail edit must be re-confirmed
                        # against that new base instead of being silently
                        # treated as the old immutable overlay.
                        rebased_input["confirmed"] = False
                        rebased_input.pop(
                            "last_committed_detail_annotation_set_id",
                            None,
                        )
                        rebased_input.pop(
                            "last_committed_detail_manifest_hash",
                            None,
                        )
                    rebased_detail_draft = self._normalize_detail_draft(
                        item,
                        rebased_input,
                        base_annotation_set=new_base_annotation_set,
                        selection=rebased_detail_selection,
                    )
                except RefinementAnnotationApplicationError:
                    rebased_detail_draft = None
            if rebased_detail_draft is None:
                try:
                    rebased_detail_draft = self._bootstrap_detail_draft(
                        item,
                        new_base_annotation_set,
                        rebased_detail_selection,
                    )
                except RefinementAnnotationApplicationError as exc:
                    rebased_detail_reason = str(exc)
        else:
            rebased_detail_reason = rebased_profile_reason
        committed_draft = {
            **normalized,
            # Rebind a pending detail draft to this new coarse base.  The
            # immutable detail artifact remains optional and is still sealed
            # independently after the annotator confirms it.
            "detail_overlay": rebased_detail_draft,
            "last_committed_annotation_set_id": annotation_set_id,
            "last_committed_manifest_hash": (
                report.manifest.semantic_content_hash
            ),
        }
        draft_cache_warning = None
        try:
            self.workspace.save_draft(
                episode_uuid=episode_uuid,
                draft=committed_draft,
                durable=False,
            )
        except StageAnnotationStoreError as exc:
            # The immutable set and its optional sidecars are already complete.
            # Treat the mutable draft only as a rebuildable cache; asking the
            # operator to retry would publish a duplicate set.
            draft_cache_warning = (
                "The immutable annotation set was committed, but the local "
                "draft cache could not be updated. Reopening the Episode will "
                "recover from the sealed set. Check the dataset mount before "
                f"continuing: {exc}"
            )
        with self._lock:
            self._annotation_sets_cache.pop(episode_uuid, None)
            self._annotation_summary_cache.pop(episode_uuid, None)
            self._annotation_root_fingerprints.pop(episode_uuid, None)
        self._detail_annotations.invalidate(episode_uuid)
        self._mark_item_annotation_committed(episode_uuid)
        self._report_operation_progress(
            progress_callback,
            97,
            "封存结果已更新",
        )
        return {
            "annotation_set_id": annotation_set_id,
            "manifest_hash": report.manifest.semantic_content_hash,
            "failure_trigger_manifest_hash": (
                trigger_report.semantic_content_hash
            ),
            "failure_trigger_count": len(trigger_set.annotations),
            "recoverable_failure_manifest_hash": (
                None
                if recoverable_failure_report is None
                else recoverable_failure_report.semantic_content_hash
            ),
            "recoverable_failure_count": len(
                recoverable_failure_set.annotations
            ),
            "annotator_id": self.annotator_id,
            "annotation_mode": self.annotation_mode,
            "created_at": binding.created_at,
            "coverage_status": index.coverage_status,
            "annotated_sample_count": index.annotated_sample_count,
            "unannotated_sample_count": index.unannotated_sample_count,
            "exclusion_count": len(index.exclusions),
            "draft": committed_draft,
            "detail_annotation": self._detail_payload(
                item,
                base_annotation_set=new_base_annotation_set,
                selection=rebased_detail_selection,
                detail_annotation_set=None,
                detail_draft=rebased_detail_draft,
                reason=rebased_detail_reason,
            ),
            "draft_cache_warning": draft_cache_warning,
        }

    def _validate_review_request_transition(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
    ) -> None:
        """Keep a sealed review request pending until a review-mode seal.

        A STANDARD draft may freely add or cancel a request before its first
        seal.  Once the latest immutable set carries the request, however,
        publishing another STANDARD set without it would make latest-version
        consumers treat the Episode as reviewed even though no second review
        occurred.
        """

        if self.annotation_mode == "SECOND_REVIEW":
            return
        annotation_sets = self._sealed_annotation_sets(item)
        if not annotation_sets:
            return
        if (
            annotation_sets[-1].bundle.episode_review_request is not None
            and draft.get("episode_review_request") is None
        ):
            raise RefinementAnnotationApplicationError(
                "A sealed second-review request can only be cleared by "
                "sealing the Episode in SECOND_REVIEW mode."
            )

    def _commit_stage_annotation_set_with_retry(
        self,
        *,
        item: ProgressAnnotationWorkItem,
        binding: StageAnnotationSetBindingV1,
        bundle: StageAnnotationBundleV1,
    ) -> StageAnnotationCommitReport:
        """Retry one idempotent publish after a likely transient mount fault."""

        assert item.progress.stage_profile is not None
        arguments = {
            "dataset_root": self.dataset_root,
            "episode_uuid": item.episode_uuid,
            "source_progress_manifest": item.progress.manifest,
            "stage_profile": item.progress.stage_profile,
            "binding": binding,
            "bundle": bundle,
        }
        try:
            self.workspace.require_episode(episode_uuid=item.episode_uuid)
            return commit_stage_annotation_set(**arguments)
        except StageAnnotationStoreError as first_error:
            if not self._is_retryable_stage_store_error(
                first_error,
                episode_uuid=item.episode_uuid,
            ):
                raise self._stage_publish_application_error(
                    first_error
                ) from first_error
        time.sleep(0.05)
        try:
            self.workspace.require_episode(episode_uuid=item.episode_uuid)
            return commit_stage_annotation_set(**arguments)
        except StageAnnotationStoreError as retry_error:
            raise self._stage_publish_application_error(retry_error) from retry_error

    def _is_retryable_stage_store_error(
        self,
        error: StageAnnotationStoreError,
        *,
        episode_uuid: str,
    ) -> bool:
        cause: BaseException | None = error
        observed_errno = None
        while cause is not None:
            if isinstance(cause, OSError) and cause.errno is not None:
                observed_errno = cause.errno
                break
            cause = cause.__cause__
        if observed_errno in _RETRYABLE_STAGE_STORE_ERRNOS:
            return True
        if observed_errno != errno.EPERM:
            return False

        # SSHFS/SFTP can surface a temporary connection-side mkdir failure as
        # EPERM.  Retry it only when the nearest existing owned directory is
        # still writable, so an ordinary permanent permission denial is not
        # hidden behind repeated attempts.
        candidate = (
            self.dataset_root
            / STAGE_ANNOTATION_RELATIVE_ROOT
            / episode_uuid
        )
        try:
            while candidate != self.dataset_root and not candidate.exists():
                candidate = candidate.parent
            return (
                candidate.is_dir()
                and os.access(candidate, os.W_OK | os.X_OK)
            )
        except OSError:
            return False

    @staticmethod
    def _stage_publish_application_error(
        error: StageAnnotationStoreError,
    ) -> RefinementAnnotationApplicationError:
        return RefinementAnnotationApplicationError(
            "Cannot publish the immutable annotation set. The current "
            "in-browser edits were not discarded. Check that the dataset "
            f"storage is writable and connected, then retry: {error}"
        )

    def reconcile_abnormal_intervals(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Restore stage fragments whose originating abnormal cut was removed.

        This operation is intentionally non-persistent.  The browser remains
        the owner of the editable draft, while the application performs the
        semantic reconstruction so old sealed sets and new edits use exactly
        the same fail-closed rules as commit.
        """

        with self._episode_operation_lock:
            self._require_active_episode(episode_uuid)
            item = self._item(episode_uuid)
            sample_axis = self._sample_axis(item)
            normalized = self._normalize_draft(
                item,
                draft,
                sample_axis=sample_axis,
            )
            abnormal_intervals = self._abnormal_intervals_from_draft(
                item,
                normalized,
            )
            restored = self._coalesce_orphaned_abnormal_fragments(
                normalized.get("occurrences", ()),
                abnormal_intervals,
            )
            restored = self._renumber_occurrences(restored)
            result = {**normalized, "occurrences": restored}
            result["failure_triggers"] = self._normalize_failure_triggers(
                item,
                restored,
                tuple(normalized.get("failure_triggers", ())),
                sample_axis=sample_axis,
            )
            return {
                "draft": self._normalize_draft(
                    item,
                    result,
                    sample_axis=sample_axis,
                )
            }

    @staticmethod
    def _abnormal_intervals_from_draft(
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
    ) -> tuple[AbnormalIntervalAnnotationV1, ...]:
        clock = item.progress.timeline.clock_domain_id
        try:
            return tuple(
                sorted(
                    (
                        AbnormalIntervalAnnotationV1(
                            annotation_id=value.get("annotation_id", ""),
                            start_at=value.get("start_at"),
                            end_at=value.get("end_at"),
                            clock_domain_id=clock,
                            reason_code=value.get("reason_code", ""),
                            note=value.get("note"),
                        )
                        for value in draft.get("abnormal_intervals", ())
                    ),
                    key=lambda value: (
                        value.start_at,
                        value.end_at,
                        value.annotation_id,
                    ),
                )
            )
        except (TypeError, ValueError) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid abnormal interval: {exc}"
            ) from exc

    @staticmethod
    def _coalesce_orphaned_abnormal_fragments(
        occurrences: Sequence[Mapping[str, Any]],
        abnormal_intervals: Sequence[AbnormalIntervalAnnotationV1],
    ) -> list[dict[str, Any]]:
        projected = sorted(
            (dict(value) for value in occurrences),
            key=lambda value: (
                float(value["start_at"]),
                float(value["end_at"]),
                str(value.get("annotation_id", "")),
            ),
        )
        coalesced: list[dict[str, Any]] = []
        index = 0
        while index < len(projected):
            occurrence = projected[index]
            # TRUNCATED_BY_ABNORMAL denotes a prefix whose *end* was cut by
            # an abnormal interval.  Its start may legitimately touch an
            # earlier, still-present abnormal interval when several cuts split
            # one stage.  Treating that start boundary as justification for the
            # truncation leaves a gap when a later cut is removed.
            touches_abnormal = any(
                math.isclose(
                    float(occurrence["end_at"]),
                    abnormal.start_at,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                for abnormal in abnormal_intervals
            )
            if (
                str(occurrence.get("outcome", "")).upper()
                == "TRUNCATED_BY_ABNORMAL"
                and not touches_abnormal
            ):
                following = (
                    projected[index + 1]
                    if index + 1 < len(projected)
                    else None
                )
                if not (
                    following
                    and following.get("coarse_attempt_id")
                    == occurrence.get("coarse_attempt_id")
                    and following.get("progress_step_id")
                    == occurrence.get("progress_step_id")
                    and following.get("stage_id")
                    == occurrence.get("stage_id")
                ):
                    raise RefinementAnnotationApplicationError(
                        "A TRUNCATED_BY_ABNORMAL stage no longer touches an "
                        "abnormal interval and has no same-stage suffix to "
                        "restore without guessing."
                    )
                restored = dict(following)
                restored["start_at"] = occurrence["start_at"]
                projected[index + 1] = restored
                index += 1
                continue
            coalesced.append(occurrence)
            index += 1
        return coalesced

    @staticmethod
    def _renumber_occurrences(
        occurrences: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        result = [dict(value) for value in occurrences]
        seen_ids: set[str] = set()
        occurrence_index_by_attempt: dict[str, int] = {}
        for occurrence in result:
            annotation_id = str(occurrence.get("annotation_id", ""))
            if annotation_id in seen_ids:
                raise RefinementAnnotationApplicationError(
                    "Abnormal-boundary projection produced a duplicate stage "
                    f"annotation identity {annotation_id!r}."
                )
            seen_ids.add(annotation_id)
            attempt_id = str(occurrence.get("coarse_attempt_id", ""))
            next_index = occurrence_index_by_attempt.get(attempt_id, 0) + 1
            occurrence_index_by_attempt[attempt_id] = next_index
            occurrence["occurrence_index_in_attempt"] = next_index
        return result

    def _project_occurrences_around_abnormal_intervals(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Split partially covered stages at abnormal-interval boundaries.

        Abnormal intervals behave like local Episode boundaries for stage
        semantics.  The editable draft may keep the annotator's unsplit stage
        block, while the immutable annotation set must contain non-overlapping
        contiguous occurrences.  A prefix cut by an abnormal interval becomes
        ``TRUNCATED_BY_ABNORMAL``; the suffix retains the original outcome and
        immutable Failure binding.

        A successful occurrence fully hidden by an abnormal interval is not
        deleted implicitly.  That is an operator decision and remains an
        explicit red-button deletion in the UI.  A captured Failure occurrence
        may be omitted only when the abnormal interval also covers its immutable
        Failure cut; the coarse Failure fact itself remains untouched.  When a
        previously committed abnormal interval is removed during re-editing,
        orphaned derived prefixes are joined only to the immediately following
        occurrence of the same stage and Attempt.
        """

        abnormal_intervals = self._abnormal_intervals_from_draft(item, draft)

        projected: list[dict[str, Any]] = []
        for raw_occurrence in draft.get("occurrences", ()):
            occurrence = dict(raw_occurrence)
            try:
                original_start = float(occurrence["start_at"])
                original_end = float(occurrence["end_at"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RefinementAnnotationApplicationError(
                    "Stage occurrence has invalid bounds."
                ) from exc
            if (
                not math.isfinite(original_start)
                or not math.isfinite(original_end)
                or original_end <= original_start
            ):
                raise RefinementAnnotationApplicationError(
                    "Stage occurrence has invalid bounds."
                )

            segments = [(original_start, original_end)]
            for abnormal in abnormal_intervals:
                next_segments: list[tuple[float, float]] = []
                for segment_start, segment_end in segments:
                    if not (
                        segment_start < abnormal.end_at
                        and abnormal.start_at < segment_end
                    ):
                        next_segments.append((segment_start, segment_end))
                        continue
                    if segment_start < abnormal.start_at:
                        next_segments.append(
                            (segment_start, abnormal.start_at)
                        )
                    if abnormal.end_at < segment_end:
                        next_segments.append((abnormal.end_at, segment_end))
                segments = next_segments

            if not segments:
                outcome = str(occurrence.get("outcome", "")).upper()
                failure_cut_is_abnormal = bool(
                    outcome == "FAILURE"
                    and occurrence.get("source_failure_fact_id")
                    and any(
                        abnormal.start_at <= original_end <= abnormal.end_at
                        for abnormal in abnormal_intervals
                    )
                )
                if failure_cut_is_abnormal:
                    continue
                raise RefinementAnnotationApplicationError(
                    "An abnormal interval fully covers a successful stage "
                    f"occurrence {occurrence.get('stage_id', '<unknown>')!r}. "
                    "Select that stage block and use the red delete button; "
                    "the application will not delete it implicitly."
                )

            original_id = str(occurrence.get("annotation_id", "")).strip()
            for fragment_index, (segment_start, segment_end) in enumerate(
                segments,
                start=1,
            ):
                fragment = dict(occurrence)
                keeps_original_end = math.isclose(
                    segment_end,
                    original_end,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                fragment["start_at"] = segment_start
                fragment["end_at"] = segment_end
                if keeps_original_end:
                    fragment["annotation_id"] = original_id
                else:
                    fragment["annotation_id"] = (
                        f"{original_id}:abnormal-prefix:{fragment_index}"
                    )
                    fragment["outcome"] = "TRUNCATED_BY_ABNORMAL"
                    fragment["source_failure_fact_id"] = None
                projected.append(fragment)

        projected = self._coalesce_orphaned_abnormal_fragments(
            projected,
            abnormal_intervals,
        )

        projected = self._renumber_occurrences(projected)

        result = {**dict(draft), "occurrences": projected}
        result["failure_triggers"] = self._normalize_failure_triggers(
            item,
            projected,
            tuple(draft.get("failure_triggers", ())),
        )
        return result

    def _sealed_annotation_sets(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> tuple[LoadedStageAnnotationSet, ...]:
        with self._lock:
            cached = self._annotation_sets_cache.get(item.episode_uuid)
            if cached is not None:
                self._annotation_sets_cache.move_to_end(item.episode_uuid)
                return cached
        root = (
            self.dataset_root
            / STAGE_ANNOTATION_RELATIVE_ROOT
            / item.episode_uuid
        )
        assert item.progress.stage_profile is not None
        for attempt in range(2):
            before = self._annotation_root_fingerprint(item)
            if before == ("missing",):
                self._cache_annotation_sets(
                    item.episode_uuid,
                    (),
                    root_fingerprint=before,
                )
                return ()
            if before and before[0] == "error":
                raise RefinementAnnotationApplicationError(
                    f"Cannot inspect sealed annotation directory: {root}"
                )
            loaded: list[LoadedStageAnnotationSet] = []
            for child in sorted(root.iterdir(), key=lambda value: value.name):
                if (
                    not child.is_dir()
                    or not (child / "manifest.cjson").is_file()
                ):
                    continue
                try:
                    value = load_stage_annotation_set(child)
                except StageAnnotationStoreError as exc:
                    raise RefinementAnnotationApplicationError(
                        f"Cannot reopen invalid sealed annotation set "
                        f"{child.name!r}: {exc}"
                    ) from exc
                binding = value.binding
                if (
                    binding.episode_capture_identity_hash
                    != item.episode_identity_hash
                    or binding.source_progress_manifest_hash
                    != item.progress_manifest_hash
                    or binding.task_stage_profile_hash
                    != item.progress.stage_profile.semantic_content_hash
                ):
                    raise RefinementAnnotationApplicationError(
                        f"Sealed annotation set {child.name!r} is bound to a "
                        "different Episode, Progress seal, or stage profile."
                    )
                loaded.append(value)
            after = self._annotation_root_fingerprint(item)
            if before != after:
                if attempt == 0:
                    continue
                raise RefinementAnnotationApplicationError(
                    "Sealed annotation versions changed repeatedly while they "
                    "were being read; retry after the other publisher finishes."
                )
            result = tuple(
                sorted(
                    loaded,
                    key=lambda value: (
                        value.binding.created_at,
                        value.binding.annotation_set_id,
                    ),
                )
            )
            self._cache_annotation_sets(
                item.episode_uuid,
                result,
                root_fingerprint=after,
            )
            return result
        raise AssertionError("annotation set read retry loop did not return")

    def _cache_annotation_sets(
        self,
        episode_uuid: str,
        values: tuple[LoadedStageAnnotationSet, ...],
        *,
        root_fingerprint: tuple[object, ...] | None = None,
    ) -> None:
        summary = self._summarize_annotation_sets(values)
        with self._lock:
            previous_summary = self._annotation_summary_cache.get(episode_uuid)
            self._annotation_sets_cache[episode_uuid] = values
            self._annotation_sets_cache.move_to_end(episode_uuid)
            self._annotation_summary_cache[episode_uuid] = summary
            self._annotation_summary_cache.move_to_end(episode_uuid)
            if root_fingerprint is not None:
                self._annotation_root_fingerprints[episode_uuid] = (
                    root_fingerprint
                )
            item = self._items.get(episode_uuid)
            item_count_changed = (
                item is not None
                and item.annotation_set_count
                != summary.annotation_set_count
            )
            if item_count_changed:
                assert item is not None
                self._items[episode_uuid] = replace(
                    item,
                    annotation_set_count=summary.annotation_set_count,
                )
            if previous_summary != summary or item_count_changed:
                self._episode_list_revision += 1
            while len(self._annotation_sets_cache) > _ANNOTATION_SET_CACHE_LIMIT:
                evicted_episode_uuid, _ = self._annotation_sets_cache.popitem(
                    last=False
                )
                self._annotation_root_fingerprints.pop(
                    evicted_episode_uuid,
                    None,
                )
            while (
                len(self._annotation_summary_cache)
                > _ANNOTATION_SUMMARY_CACHE_LIMIT
            ):
                self._annotation_summary_cache.popitem(last=False)

    @staticmethod
    def _annotation_set_summary(
        value: LoadedStageAnnotationSet,
    ) -> dict[str, Any]:
        return {
            "annotation_set_id": value.binding.annotation_set_id,
            "annotator_id": value.binding.annotator_id,
            "annotation_mode": value.binding.annotation_mode,
            "created_at": value.binding.created_at,
            "manifest_hash": value.manifest.semantic_content_hash,
            "annotation_count": len(value.bundle.occurrences),
            "needs_second_review": (
                value.bundle.episode_review_request is not None
            ),
            "review_state": (
                "PENDING"
                if value.bundle.episode_review_request is not None
                else (
                    "COMPLETED"
                    if value.binding.annotation_mode == "SECOND_REVIEW"
                    else "NONE"
                )
            ),
            "whole_episode_abnormal": (
                value.bundle.episode_abnormal is not None
            ),
            "whole_episode_abnormal_reason": (
                None
                if value.bundle.episode_abnormal is None
                else value.bundle.episode_abnormal.reason_code
            ),
        }

    def _draft_matches_annotation_set(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
        value: LoadedStageAnnotationSet,
    ) -> bool:
        """Return whether a mutable cache exactly represents one sealed set."""

        # The mutable cache predates the current GUI and old versions may not
        # contain commit bookkeeping fields.  Compare editable semantics, not
        # profile/config hashes or cache metadata.  A byte-for-byte equivalent
        # editable view is safe to treat as sealed; any actual edit remains an
        # unsealed draft and is protected during navigation.
        if draft.get("draft_annotation_mode") != value.binding.annotation_mode:
            return False
        try:
            if (
                self._bundle_from_draft(item, draft).semantic_content_hash
                != value.bundle.semantic_content_hash
            ):
                return False
            sealed_draft = self._draft_from_annotation_set(
                item,
                value,
                draft_revision=int(draft.get("draft_revision", 0)),
            )
        except (RefinementAnnotationApplicationError, TypeError, ValueError):
            return False
        return (
            list(draft.get("failure_triggers", ()))
            == list(sealed_draft.get("failure_triggers", ()))
            and list(draft.get("recoverable_failures", ()))
            == list(sealed_draft.get("recoverable_failures", ()))
        )

    def _draft_from_annotation_set(
        self,
        item: ProgressAnnotationWorkItem,
        value: LoadedStageAnnotationSet,
        *,
        draft_revision: int,
    ) -> dict[str, Any]:
        bundle = value.bundle
        episode_abnormal = (
            None
            if bundle.episode_abnormal is None
            else {
                "annotation_id": bundle.episode_abnormal.annotation_id,
                "reason_code": bundle.episode_abnormal.reason_code,
                "note": bundle.episode_abnormal.note,
            }
        )
        episode_review_request = (
            None
            if bundle.episode_review_request is None
            else {
                "annotation_id": (
                    bundle.episode_review_request.annotation_id
                ),
                "note": bundle.episode_review_request.note,
            }
        )
        capture_end_step = (
            None
            if bundle.capture_end_step is None
            else {
                "annotation_id": bundle.capture_end_step.annotation_id,
                "source_terminal_fact_id": (
                    bundle.capture_end_step.source_terminal_fact_id
                ),
                "source_attempt_id": bundle.capture_end_step.source_attempt_id,
                "end_progress_step_id": (
                    bundle.capture_end_step.end_progress_step_id
                ),
            }
        )
        occurrences = [
            {
                "annotation_id": annotation.annotation_id,
                "coarse_attempt_id": annotation.coarse_attempt_id,
                "progress_step_id": annotation.progress_step_id,
                "stage_id": annotation.stage_id,
                "occurrence_index_in_attempt": (
                    annotation.occurrence_index_in_attempt
                ),
                "start_at": annotation.start_at,
                "end_at": annotation.end_at,
                "outcome": annotation.outcome,
                "confidence": annotation.confidence,
                "source_failure_fact_id": (
                    annotation.source_failure_fact_id
                ),
            }
            for annotation in bundle.occurrences
        ]
        trigger_drafts: list[Mapping[str, Any]] = []
        trigger_file = failure_trigger_path(value.annotation_dir)
        sample_axis = self._sample_axis(item)
        if trigger_file.is_file():
            try:
                trigger_set = load_failure_trigger_annotation_set(
                    value.annotation_dir
                )
            except FailureTriggerStoreError as exc:
                raise RefinementAnnotationApplicationError(
                    f"Cannot reopen invalid failure-trigger sidecar: {exc}"
                ) from exc
            if (
                trigger_set.stage_annotation_set_id
                != value.binding.annotation_set_id
                or trigger_set.episode_capture_identity_hash
                != item.episode_identity_hash
                or trigger_set.source_progress_manifest_hash
                != item.progress_manifest_hash
                or trigger_set.task_stage_profile_hash
                != item.progress.stage_profile.semantic_content_hash
                or trigger_set.stage_annotation_bundle_hash
                != bundle.semantic_content_hash
                or trigger_set.sample_axis_hash
                != failure_trigger_sample_axis_hash(sample_axis)
                or trigger_set.bound_clock_domain_id
                != item.progress.timeline.clock_domain_id
            ):
                raise RefinementAnnotationApplicationError(
                    "Failure-trigger sidecar does not bind the Episode, "
                    "Progress seal, stage set, sample axis, or clock."
                )
            trigger_drafts = [
                annotation.to_payload(include_hash=False)
                for annotation in trigger_set.annotations
            ]
        recoverable_failure_drafts: list[Mapping[str, Any]] = []
        recoverable_file = recoverable_failure_path(value.annotation_dir)
        if recoverable_file.is_file():
            try:
                recoverable_set = load_recoverable_failure_annotation_set(
                    value.annotation_dir
                )
            except RecoverableFailureStoreError as exc:
                raise RefinementAnnotationApplicationError(
                    "Cannot reopen invalid recoverable-failure sidecar: "
                    f"{exc}"
                ) from exc
            if (
                recoverable_set.stage_annotation_set_id
                != value.binding.annotation_set_id
                or recoverable_set.episode_capture_identity_hash
                != item.episode_identity_hash
                or recoverable_set.source_progress_manifest_hash
                != item.progress_manifest_hash
                or recoverable_set.task_stage_profile_hash
                != item.progress.stage_profile.semantic_content_hash
                or recoverable_set.stage_annotation_bundle_hash
                != bundle.semantic_content_hash
                or recoverable_set.sample_axis_hash
                != recoverable_failure_sample_axis_hash(sample_axis)
                or recoverable_set.bound_clock_domain_id
                != item.progress.timeline.clock_domain_id
            ):
                raise RefinementAnnotationApplicationError(
                    "Recoverable-failure sidecar does not bind the Episode, "
                    "Progress seal, stage set, sample axis, or clock."
                )
            recoverable_failure_drafts = [
                annotation.to_payload(include_hash=False)
                for annotation in recoverable_set.annotations
            ]
        draft = {
            "schema": _DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            "draft_annotation_mode": value.binding.annotation_mode,
            "draft_revision": int(draft_revision),
            "occurrences": occurrences,
            "failure_triggers": self._normalize_failure_triggers(
                item,
                occurrences,
                trigger_drafts,
                sample_axis=sample_axis,
            ),
            "recoverable_failures": self._normalize_recoverable_failures(
                item,
                occurrences,
                recoverable_failure_drafts,
                sample_axis=sample_axis,
                allow_adjacent_legacy=True,
            ),
            "deletion_history": [],
            "abnormal_intervals": [
                {
                    "annotation_id": annotation.annotation_id,
                    "start_at": annotation.start_at,
                    "end_at": annotation.end_at,
                    "reason_code": annotation.reason_code,
                    "note": annotation.note,
                }
                for annotation in bundle.abnormal_intervals
            ],
            "progress_boundary_overrides": [
                {
                    "annotation_id": annotation.annotation_id,
                    "source_advance_fact_id": (
                        annotation.source_advance_fact_id
                    ),
                    "corrected_effective_at": (
                        annotation.corrected_effective_at
                    ),
                }
                for annotation in bundle.progress_boundary_overrides
            ],
            "capture_end_step": capture_end_step,
            "episode_abnormal": episode_abnormal,
            "episode_review_request": episode_review_request,
            "privileged_tools_enabled": bool(
                bundle.abnormal_intervals
                or bundle.progress_boundary_overrides
                or bundle.episode_abnormal is not None
            ),
            # None preserves the legacy single-candidate auto-resolution.
            # An empty string is an explicit coarse-only choice made from the
            # STANDARD annotation profile selector.
            "detail_profile_selection_id": None,
            "last_committed_annotation_set_id": (
                value.binding.annotation_set_id
            ),
            "last_committed_manifest_hash": (
                value.manifest.semantic_content_hash
            ),
            "detail_overlay": None,
        }
        return draft

    def _summary_load_lock(self, episode_uuid: str) -> threading.Lock:
        return self._summary_load_locks[
            hash(str(episode_uuid)) % len(self._summary_load_locks)
        ]

    def _require_active_episode(self, episode_uuid: str) -> str:
        normalized = str(episode_uuid).strip()
        with self._lock:
            current = self._active_episode_uuid
        if current != normalized:
            raise RefinementAnnotationApplicationError(
                f"Episode {normalized!r} is not the active locked editing "
                "workspace. Open it before saving or publishing annotations."
            )
        try:
            self.workspace.require_episode(episode_uuid=normalized)
        except StageAnnotationStoreError as exc:
            raise RefinementAnnotationApplicationError(str(exc)) from exc
        return normalized

    def _item(self, episode_uuid: str) -> ProgressAnnotationWorkItem:
        normalized = str(episode_uuid).strip()
        with self._lock:
            self._require_open()
            item = self._items.get(normalized)
        if item is None:
            # A direct request for an Episode that arrived during background
            # discovery still gets the historical fail-closed lookup behavior.
            self.refresh(force=True)
            with self._lock:
                self._require_open()
                item = self._items.get(normalized)
            if item is None:
                raise RefinementAnnotationApplicationError(
                    "Episode is not finalized, sealed, stage-profiled, or "
                    "otherwise eligible for annotation."
                )
        assert item is not None
        return item

    def _mark_item_has_draft(self, episode_uuid: str) -> None:
        normalized = str(episode_uuid).strip()
        with self._lock:
            item = self._items.get(normalized)
            if item is not None and not item.has_draft:
                self._items[normalized] = replace(item, has_draft=True)
                self._episode_list_revision += 1

    def _mark_item_annotation_committed(self, episode_uuid: str) -> None:
        normalized = str(episode_uuid).strip()
        with self._lock:
            item = self._items.get(normalized)
            if item is None:
                return
            self._items[normalized] = replace(
                item,
                has_draft=True,
                annotation_set_count=item.annotation_set_count + 1,
            )
            self._episode_list_revision += 1

    @staticmethod
    def _capture_end_step_context(
        item: ProgressAnnotationWorkItem,
    ) -> tuple[Any, tuple[Any, ...]] | None:
        timeline = item.progress.timeline
        profile = item.progress.profile
        attempts = tuple(timeline.attempts)
        if (
            profile.topology_kind != "LINEAR"
            or not attempts
            or getattr(timeline, "terminal_fact_id", None) is None
            or getattr(timeline, "termination_cause", None)
            not in {"MANUAL_STOP", "ABORTED", "UNKNOWN", "WINDOW_COMPLETED"}
        ):
            return None
        source = attempts[-1]
        if (
            source.end_at is None
            or source.end_at <= source.start_at
            or source.failure_fact_id is not None
            or source.outcome
            not in {"ABORTED", "TRUNCATED_AT_CAPTURE_END", "UNKNOWN"}
        ):
            return None
        ordered_steps = tuple(profile.steps)
        source_index = next(
            (
                index
                for index, step in enumerate(ordered_steps)
                if step.progress_step_id == source.progress_step_id
            ),
            None,
        )
        if source_index is None:
            return None
        entry_index = next(
            (
                index
                for index, step in enumerate(ordered_steps)
                if step.progress_step_id == item.progress.binding.entry_step_id
            ),
            None,
        )
        if entry_index is None or entry_index > source_index:
            return None
        candidates = ordered_steps[entry_index:]
        stage_profile = item.progress.stage_profile
        if stage_profile is None or any(
            not stage_profile.stages_for_step(step.progress_step_id)
            for step in candidates
        ):
            return None
        return source, candidates

    def _capture_end_step_editor_payload(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> dict[str, Any] | None:
        context = self._capture_end_step_context(item)
        if context is None:
            return None
        source, candidates = context
        attempts = tuple(item.progress.timeline.attempts)
        source_step_index = next(
            index
            for index, step in enumerate(candidates)
            if step.progress_step_id == source.progress_step_id
        )

        def option_payload(index: int, step: Any) -> dict[str, Any]:
            anchor_index = len(attempts) - 1
            if index <= source_step_index:
                anchor_index = next(
                    (
                        attempt_index
                        for attempt_index in range(len(attempts) - 1, -1, -1)
                        if attempts[attempt_index].progress_step_id
                        == step.progress_step_id
                    ),
                    -1,
                )
            selectable = bool(
                anchor_index >= 0
                and not any(
                    attempt.failure_fact_id is not None
                    for attempt in attempts[anchor_index:]
                )
            )
            return {
                "progress_step_id": step.progress_step_id,
                "display_name": step.display_name,
                "ordinal": step.ordinal,
                "selectable": selectable,
                "disabled_reason": (
                    None
                    if selectable
                    else "该选择会覆盖真实 Failure Attempt，不能使用"
                ),
            }

        return {
            "source_attempt_id": source.attempt_id,
            "source_progress_step_id": source.progress_step_id,
            "source_attempt_outcome": source.outcome,
            "source_terminal_fact_id": item.progress.timeline.terminal_fact_id,
            "termination_cause": item.progress.timeline.termination_cause,
            "default_end_progress_step_id": source.progress_step_id,
            "options": [
                option_payload(index, step)
                for index, step in enumerate(candidates)
            ],
        }

    def _rebuild_capture_end_step_projection(
        self,
        item: ProgressAnnotationWorkItem,
        *,
        occurrences: Sequence[Mapping[str, Any]],
        abnormal_intervals: Sequence[Mapping[str, Any]],
        recoverable_failures: Sequence[Mapping[str, Any]],
        end_progress_step_id: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        context = self._capture_end_step_context(item)
        if context is None:
            raise RefinementAnnotationApplicationError(
                "This Episode has no eligible ambiguous capture-end Attempt."
            )
        source, candidates = context
        selected_id = str(end_progress_step_id).strip()
        selected_index = next(
            (
                index
                for index, step in enumerate(candidates)
                if step.progress_step_id == selected_id
            ),
            None,
        )
        if selected_index is None:
            raise RefinementAnnotationApplicationError(
                "Selected capture-end Step is outside the Episode entry-to-end "
                "linear profile range."
            )
        source_step_index = next(
            index
            for index, step in enumerate(candidates)
            if step.progress_step_id == source.progress_step_id
        )
        timeline_attempts = tuple(item.progress.timeline.attempts)
        source_attempt_index = len(timeline_attempts) - 1
        if selected_index <= source_step_index:
            anchor_attempt_index = next(
                (
                    index
                    for index in range(source_attempt_index, -1, -1)
                    if timeline_attempts[index].progress_step_id == selected_id
                ),
                None,
            )
            if anchor_attempt_index is None:
                raise RefinementAnnotationApplicationError(
                    "The selected earlier ending Step has no capture-time "
                    "Attempt to anchor its annotation projection."
                )
            anchor = timeline_attempts[anchor_attempt_index]
            selected_steps = (candidates[selected_index],)
        else:
            anchor_attempt_index = source_attempt_index
            anchor = source
            selected_steps = candidates[
                source_step_index : selected_index + 1
            ]
        if any(
            attempt.failure_fact_id is not None
            for attempt in timeline_attempts[anchor_attempt_index:]
        ):
            raise RefinementAnnotationApplicationError(
                "Changing the ending Step would discard a captured Failure "
                "Attempt, so the projection is rejected."
            )
        if anchor.end_at is None or anchor.end_at <= anchor.start_at:
            raise RefinementAnnotationApplicationError(
                "The capture-end projection anchor has no positive duration."
            )
        source_end = float(source.end_at)
        if any(
            float(value.get("start_at", source_end)) < source_end
            and float(value.get("end_at", anchor.start_at))
            > float(anchor.start_at)
            for value in abnormal_intervals
        ):
            raise RefinementAnnotationApplicationError(
                "Remove or finish abnormal intervals overlapping the final "
                "capture Attempt before changing its ending Step."
            )
        suffix_attempt_ids = {
            attempt.attempt_id
            for attempt in timeline_attempts[anchor_attempt_index:]
        }
        affected = tuple(
            value
            for value in occurrences
            if str(value.get("coarse_attempt_id", "")) in suffix_attempt_ids
            or ":capture-end-step:" in str(
                value.get("coarse_attempt_id", "")
            )
        )
        affected_ids = {
            str(value.get("annotation_id", "")) for value in affected
        }
        if any(
            str(value.get("stage_occurrence_annotation_id", ""))
            in affected_ids
            for value in recoverable_failures
        ):
            raise RefinementAnnotationApplicationError(
                "The final capture projection has recoverable-failure events. "
                "Remove or reannotate those events before changing the ending Step."
            )
        retained = [
            dict(value)
            for value in occurrences
            if str(value.get("annotation_id", "")) not in affected_ids
        ]
        stage_profile = item.progress.stage_profile
        stage_entry = item.progress.stage_entry_binding
        assert stage_profile is not None
        assert stage_entry is not None
        source_stages = stage_profile.stages_for_step(
            anchor.progress_step_id
        )
        source_stage_index = {
            stage.stage_id: index
            for index, stage in enumerate(source_stages)
        }
        cursor = 0
        if stage_entry.entry_stage_id in source_stage_index:
            cursor = source_stage_index[stage_entry.entry_stage_id]
        for value in sorted(
            (
                value
                for value in retained
                if str(value.get("progress_step_id", ""))
                == anchor.progress_step_id
            ),
            key=lambda value: (
                float(value.get("start_at", 0.0)),
                float(value.get("end_at", 0.0)),
            ),
        ):
            index = source_stage_index.get(str(value.get("stage_id", "")))
            if index is None:
                continue
            cursor = index if value.get("outcome") == "FAILURE" else index + 1
        if cursor >= len(source_stages):
            raise RefinementAnnotationApplicationError(
                "The final captured Progress Step has no remaining fine stage "
                "to project."
            )
        stage_specs: list[tuple[Any, Any, str]] = []
        for step_index, step in enumerate(selected_steps):
            stages = stage_profile.stages_for_step(step.progress_step_id)
            if step_index == 0:
                stages = stages[cursor:]
            attempt_id = capture_end_projected_attempt_id(
                source_attempt_id=anchor.attempt_id,
                source_progress_step_id=anchor.progress_step_id,
                progress_step_id=step.progress_step_id,
            )
            stage_specs.extend(
                (step, stage, attempt_id) for stage in stages
            )
        if not stage_specs:
            raise RefinementAnnotationApplicationError(
                "Selected capture-end Step produced no annotatable fine stages."
            )
        duration = source_end - float(anchor.start_at)
        occurrence_index_by_attempt: dict[str, int] = {}
        rebuilt: list[dict[str, Any]] = []
        for global_index, (step, stage, attempt_id) in enumerate(
            stage_specs,
            start=1,
        ):
            occurrence_index = occurrence_index_by_attempt.get(attempt_id, 0) + 1
            occurrence_index_by_attempt[attempt_id] = occurrence_index
            rebuilt.append(
                {
                    "annotation_id": (
                        f"draft:{source.attempt_id}:capture-end:"
                        f"{step.progress_step_id}:{occurrence_index}"
                    ),
                    "coarse_attempt_id": attempt_id,
                    "progress_step_id": step.progress_step_id,
                    "stage_id": stage.stage_id,
                    "occurrence_index_in_attempt": occurrence_index,
                    "start_at": float(anchor.start_at)
                    + duration * (global_index - 1) / len(stage_specs),
                    "end_at": float(anchor.start_at)
                    + duration * global_index / len(stage_specs),
                    "outcome": "SUCCESS",
                    "confidence": 1.0,
                    "source_failure_fact_id": None,
                }
            )
        result = retained + rebuilt
        result.sort(
            key=lambda value: (
                float(value.get("start_at", 0.0)),
                float(value.get("end_at", 0.0)),
                str(value.get("annotation_id", "")),
            )
        )
        return result, {
            "annotation_id": (
                f"capture-end:{item.progress.timeline.terminal_fact_id}"
            ),
            "source_terminal_fact_id": item.progress.timeline.terminal_fact_id,
            "source_attempt_id": source.attempt_id,
            "end_progress_step_id": selected_id,
        }

    def _bootstrap_draft(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> dict[str, Any]:
        stage_profile = item.progress.stage_profile
        stage_entry = item.progress.stage_entry_binding
        assert stage_profile is not None
        assert stage_entry is not None
        stage_cursor_by_step = {
            step.progress_step_id: 0 for step in item.progress.profile.steps
        }
        entry_stage = stage_profile.stage_by_id[stage_entry.entry_stage_id]
        stage_cursor_by_step[entry_stage.progress_step_id] = (
            entry_stage.ordinal - 1
        )
        occurrences: list[dict[str, Any]] = []
        capture_end_context = self._capture_end_step_context(item)
        # A MANUAL_STOP proves where capture ended, but it does not prove that
        # unrecorded later steps completed.  Bootstrap from the last recorded
        # Attempt and let the annotator explicitly extend the projection.
        auto_capture_end_context = (
            capture_end_context
            if item.progress.timeline.termination_cause == "MANUAL_STOP"
            else None
        )
        capture_end_source_id = (
            None
            if auto_capture_end_context is None
            else auto_capture_end_context[0].attempt_id
        )
        for attempt in item.progress.timeline.attempts:
            if attempt.end_at is None or attempt.end_at <= attempt.start_at:
                continue
            if attempt.attempt_id == capture_end_source_id:
                continue
            stages = stage_profile.stages_for_step(
                attempt.progress_step_id
            )
            cursor = stage_cursor_by_step[attempt.progress_step_id]
            if cursor >= len(stages):
                continue
            if attempt.outcome == "FAILURE":
                selected = (stages[cursor],)
                outcomes = ("FAILURE",)
            elif attempt.outcome == "SUCCESS":
                selected = stages[cursor:]
                outcomes = tuple("SUCCESS" for _ in selected)
            else:
                selected = (stages[cursor],)
                outcomes = ("SUCCESS",)
            split_bounds = _equal_active_time_splits(
                item,
                attempt.start_at,
                attempt.end_at,
                len(selected),
            )
            for index, (stage, outcome, bounds) in enumerate(
                zip(selected, outcomes, split_bounds),
                start=1,
            ):
                start_at, end_at = bounds
                occurrences.append(
                    {
                        "annotation_id": (
                            f"draft:{attempt.attempt_id}:{index}"
                        ),
                        "coarse_attempt_id": attempt.attempt_id,
                        "progress_step_id": attempt.progress_step_id,
                        "stage_id": stage.stage_id,
                        "occurrence_index_in_attempt": index,
                        "start_at": start_at,
                        "end_at": end_at,
                        "outcome": outcome,
                        "confidence": 1.0,
                        "source_failure_fact_id": (
                            attempt.failure_fact_id
                            if outcome == "FAILURE"
                            else None
                        ),
                    }
                )
            if attempt.outcome == "SUCCESS":
                stage_cursor_by_step[attempt.progress_step_id] = len(stages)
            elif attempt.outcome != "FAILURE":
                stage_cursor_by_step[attempt.progress_step_id] = min(
                    len(stages),
                    cursor + 1,
                )
        capture_end_step = None
        if auto_capture_end_context is not None:
            occurrences, capture_end_step = (
                self._rebuild_capture_end_step_projection(
                    item,
                    occurrences=occurrences,
                    abnormal_intervals=(),
                    recoverable_failures=(),
                    end_progress_step_id=(
                        auto_capture_end_context[0].progress_step_id
                    ),
                )
            )
        return {
            "schema": _DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            "draft_annotation_mode": self.annotation_mode,
            "draft_revision": 0,
            "occurrences": occurrences,
            "failure_triggers": self._normalize_failure_triggers(
                item,
                occurrences,
                (),
            ),
            "recoverable_failures": [],
            "deletion_history": [],
            "abnormal_intervals": [],
            "progress_boundary_overrides": [],
            "capture_end_step": capture_end_step,
            "episode_abnormal": None,
            "episode_review_request": None,
            "privileged_tools_enabled": False,
            "detail_profile_selection_id": None,
            "detail_overlay": None,
        }

    def _normalize_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
        *,
        sample_axis: Any | None = None,
        allow_legacy_adjacent_recoverable_failures: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(draft, Mapping):
            raise RefinementAnnotationApplicationError(
                "Draft payload must be an object."
            )
        if draft.get("schema") != _DRAFT_SCHEMA:
            raise RefinementAnnotationApplicationError(
                "Unsupported stage annotation draft schema."
            )
        if str(draft.get("episode_uuid", "")).strip() != item.episode_uuid:
            raise RefinementAnnotationApplicationError(
                "Draft Episode identity changed."
            )
        allowed = {
            "schema",
            "episode_uuid",
            "draft_annotation_mode",
            "draft_revision",
            "occurrences",
            "failure_triggers",
            "recoverable_failures",
            "deletion_history",
            "abnormal_intervals",
            "progress_boundary_overrides",
            "capture_end_step",
            "episode_abnormal",
            "episode_review_request",
            "privileged_tools_enabled",
            "detail_profile_selection_id",
            "last_committed_annotation_set_id",
            "last_committed_manifest_hash",
            "detail_overlay",
        }
        if unknown := set(draft) - allowed:
            raise RefinementAnnotationApplicationError(
                f"Unsupported draft fields: {sorted(unknown)}."
            )
        privileged = bool(draft.get("privileged_tools_enabled", False))
        abnormal = list(draft.get("abnormal_intervals", ()))
        overrides = list(draft.get("progress_boundary_overrides", ()))
        occurrences = list(draft.get("occurrences", ()))
        failure_triggers = list(draft.get("failure_triggers", ()))
        recoverable_failures = list(draft.get("recoverable_failures", ()))
        deletion_history = list(draft.get("deletion_history", ()))
        capture_end_step = draft.get("capture_end_step")
        episode_abnormal = draft.get("episode_abnormal")
        episode_review_request = draft.get("episode_review_request")
        detail_profile_selection_id = draft.get("detail_profile_selection_id")
        if detail_profile_selection_id is not None:
            if not isinstance(detail_profile_selection_id, str):
                raise RefinementAnnotationApplicationError(
                    "detail_profile_selection_id must be a string or null."
                )
            detail_profile_selection_id = detail_profile_selection_id.strip()
        detail_overlay = draft.get("detail_overlay")
        if detail_overlay is not None and not isinstance(
            detail_overlay,
            Mapping,
        ):
            raise RefinementAnnotationApplicationError(
                "detail_overlay must be an object or null."
            )
        for name, values in (
            ("occurrences", occurrences),
            ("failure_triggers", failure_triggers),
            ("recoverable_failures", recoverable_failures),
            ("deletion_history", deletion_history),
            ("abnormal_intervals", abnormal),
            ("progress_boundary_overrides", overrides),
        ):
            if any(not isinstance(value, Mapping) for value in values):
                raise RefinementAnnotationApplicationError(
                    f"Every {name} item must be an object."
                )
        if (
            episode_abnormal is not None
            and not isinstance(episode_abnormal, Mapping)
        ):
            raise RefinementAnnotationApplicationError(
                "episode_abnormal must be an object or null."
            )
        if (
            episode_review_request is not None
            and not isinstance(episode_review_request, Mapping)
        ):
            raise RefinementAnnotationApplicationError(
                "episode_review_request must be an object or null."
            )
        if capture_end_step is not None:
            if not isinstance(capture_end_step, Mapping):
                raise RefinementAnnotationApplicationError(
                    "capture_end_step must be an object or null."
                )
            try:
                normalized_capture_end = CaptureEndStepAnnotationV1(
                    annotation_id=capture_end_step.get(
                        "annotation_id",
                        "",
                    ),
                    source_terminal_fact_id=capture_end_step.get(
                        "source_terminal_fact_id",
                        "",
                    ),
                    source_attempt_id=capture_end_step.get(
                        "source_attempt_id",
                        "",
                    ),
                    end_progress_step_id=capture_end_step.get(
                        "end_progress_step_id",
                        "",
                    ),
                    clock_domain_id=item.progress.timeline.clock_domain_id,
                )
            except (TypeError, ValueError) as exc:
                raise RefinementAnnotationApplicationError(
                    f"Invalid capture-end Step annotation: {exc}"
                ) from exc
            capture_end_step = {
                "annotation_id": normalized_capture_end.annotation_id,
                "source_terminal_fact_id": (
                    normalized_capture_end.source_terminal_fact_id
                ),
                "source_attempt_id": normalized_capture_end.source_attempt_id,
                "end_progress_step_id": (
                    normalized_capture_end.end_progress_step_id
                ),
            }
        if not privileged and (abnormal or overrides or episode_abnormal):
            raise RefinementAnnotationApplicationError(
                "Abnormal/whole-Episode/coarse-boundary tools require the "
                "explicit privileged-tools switch."
            )
        failure_triggers = self._normalize_failure_triggers(
            item,
            occurrences,
            failure_triggers,
            sample_axis=sample_axis,
        )
        recoverable_failures = self._normalize_recoverable_failures(
            item,
            occurrences,
            recoverable_failures,
            sample_axis=sample_axis,
            allow_adjacent_legacy=(
                allow_legacy_adjacent_recoverable_failures
            ),
        )
        return {
            "schema": _DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            "draft_annotation_mode": self.annotation_mode,
            "draft_revision": int(draft.get("draft_revision", 0)),
            "occurrences": occurrences,
            "failure_triggers": failure_triggers,
            "recoverable_failures": recoverable_failures,
            "deletion_history": deletion_history,
            "abnormal_intervals": abnormal,
            "progress_boundary_overrides": overrides,
            "capture_end_step": capture_end_step,
            "episode_abnormal": episode_abnormal,
            "episode_review_request": episode_review_request,
            "privileged_tools_enabled": privileged,
            "detail_profile_selection_id": detail_profile_selection_id,
            "detail_overlay": detail_overlay,
            **(
                {
                    key: draft[key]
                    for key in (
                        "last_committed_annotation_set_id",
                        "last_committed_manifest_hash",
                    )
                    if key in draft
                }
            ),
        }

    def _normalize_failure_triggers(
        self,
        item: ProgressAnnotationWorkItem,
        occurrences: Sequence[Mapping[str, Any]],
        existing: Sequence[Mapping[str, Any]],
        *,
        sample_axis: Any | None = None,
    ) -> list[dict[str, Any]]:
        """Normalize trigger drafts against the current failed-stage bounds."""

        if sample_axis is None:
            sample_axis = self._sample_axis(item)
        clock = item.progress.timeline.clock_domain_id
        try:
            occurrence_contracts = tuple(
                StageOccurrenceAnnotationV1(
                    annotation_id=value.get("annotation_id", ""),
                    coarse_attempt_id=value.get("coarse_attempt_id", ""),
                    progress_step_id=value.get("progress_step_id", ""),
                    stage_id=value.get("stage_id", ""),
                    occurrence_index_in_attempt=value.get(
                        "occurrence_index_in_attempt",
                        0,
                    ),
                    start_at=value.get("start_at"),
                    end_at=value.get("end_at"),
                    clock_domain_id=clock,
                    outcome=value.get("outcome", ""),
                    confidence=value.get("confidence", 1.0),
                    source_failure_fact_id=value.get(
                        "source_failure_fact_id"
                    ),
                )
                for value in occurrences
            )
            return normalize_failure_trigger_drafts(
                timeline=item.progress.timeline,
                occurrences=occurrence_contracts,
                sample_axis=sample_axis,
                existing=existing,
            )
        except (TypeError, ValueError, FailureTriggerResolutionError) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid failure_trigger annotation: {exc}"
            ) from exc

    def _failure_trigger_set_from_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
        *,
        bundle: StageAnnotationBundleV1,
        binding: StageAnnotationSetBindingV1,
        sample_axis: Any,
    ) -> FailureTriggerAnnotationSetV1:
        try:
            annotations = build_failure_trigger_annotations(
                timeline=item.progress.timeline,
                occurrences=bundle.occurrences,
                sample_axis=sample_axis,
                drafts=draft.get("failure_triggers", ()),
            )
        except (TypeError, ValueError, FailureTriggerResolutionError) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid failure_trigger annotation: {exc}"
            ) from exc
        return FailureTriggerAnnotationSetV1(
            annotation_set_id=binding.annotation_set_id,
            stage_annotation_set_id=binding.annotation_set_id,
            episode_capture_identity_hash=(
                binding.episode_capture_identity_hash
            ),
            source_progress_manifest_hash=(
                binding.source_progress_manifest_hash
            ),
            task_stage_profile_hash=binding.task_stage_profile_hash,
            stage_annotation_bundle_hash=bundle.semantic_content_hash,
            sample_axis_hash=failure_trigger_sample_axis_hash(sample_axis),
            bound_clock_domain_id=binding.bound_clock_domain_id,
            annotator_id=binding.annotator_id,
            created_at=binding.created_at,
            created_clock_domain_id=binding.created_clock_domain_id,
            annotations=annotations,
        )

    def _normalize_recoverable_failures(
        self,
        item: ProgressAnnotationWorkItem,
        occurrences: Sequence[Mapping[str, Any]],
        existing: Sequence[Mapping[str, Any]],
        *,
        sample_axis: Any | None = None,
        allow_adjacent_legacy: bool = False,
    ) -> list[dict[str, Any]]:
        """Bind temporary failures to one successful occurrence and samples."""

        if sample_axis is None:
            sample_axis = self._sample_axis(item)
        clock = item.progress.timeline.clock_domain_id
        try:
            occurrence_contracts = tuple(
                StageOccurrenceAnnotationV1(
                    annotation_id=value.get("annotation_id", ""),
                    coarse_attempt_id=value.get("coarse_attempt_id", ""),
                    progress_step_id=value.get("progress_step_id", ""),
                    stage_id=value.get("stage_id", ""),
                    occurrence_index_in_attempt=value.get(
                        "occurrence_index_in_attempt",
                        0,
                    ),
                    start_at=value.get("start_at"),
                    end_at=value.get("end_at"),
                    clock_domain_id=clock,
                    outcome=value.get("outcome", ""),
                    confidence=value.get("confidence", 1.0),
                    source_failure_fact_id=value.get(
                        "source_failure_fact_id"
                    ),
                )
                for value in occurrences
            )
            return normalize_recoverable_failure_drafts(
                occurrences=occurrence_contracts,
                sample_axis=sample_axis,
                existing=existing,
                allow_adjacent_legacy=allow_adjacent_legacy,
            )
        except (
            TypeError,
            ValueError,
            RecoverableFailureResolutionError,
        ) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid recoverable-failure annotation: {exc}"
            ) from exc

    def _recoverable_failure_set_from_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
        *,
        bundle: StageAnnotationBundleV1,
        binding: StageAnnotationSetBindingV1,
        sample_axis: EpisodeSampleAxis,
    ) -> RecoverableFailureAnnotationSetV1:
        try:
            annotations = build_recoverable_failure_annotations(
                occurrences=bundle.occurrences,
                sample_axis=sample_axis,
                drafts=draft.get("recoverable_failures", ()),
            )
        except (
            TypeError,
            ValueError,
            RecoverableFailureResolutionError,
        ) as exc:
            raise RefinementAnnotationApplicationError(
                f"Invalid recoverable-failure annotation: {exc}"
            ) from exc
        return RecoverableFailureAnnotationSetV1(
            annotation_set_id=binding.annotation_set_id,
            stage_annotation_set_id=binding.annotation_set_id,
            episode_capture_identity_hash=(
                binding.episode_capture_identity_hash
            ),
            source_progress_manifest_hash=(
                binding.source_progress_manifest_hash
            ),
            task_stage_profile_hash=binding.task_stage_profile_hash,
            stage_annotation_bundle_hash=bundle.semantic_content_hash,
            sample_axis_hash=recoverable_failure_sample_axis_hash(sample_axis),
            bound_clock_domain_id=binding.bound_clock_domain_id,
            annotator_id=binding.annotator_id,
            created_at=binding.created_at,
            created_clock_domain_id=binding.created_clock_domain_id,
            annotations=annotations,
        )

    def _bundle_from_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
    ) -> StageAnnotationBundleV1:
        clock = item.progress.timeline.clock_domain_id
        return StageAnnotationBundleV1(
            occurrences=tuple(
                StageOccurrenceAnnotationV1(
                    annotation_id=value.get("annotation_id", ""),
                    coarse_attempt_id=value.get("coarse_attempt_id", ""),
                    progress_step_id=value.get("progress_step_id", ""),
                    stage_id=value.get("stage_id", ""),
                    occurrence_index_in_attempt=value.get(
                        "occurrence_index_in_attempt",
                        0,
                    ),
                    start_at=value.get("start_at"),
                    end_at=value.get("end_at"),
                    clock_domain_id=clock,
                    outcome=value.get("outcome", ""),
                    confidence=value.get("confidence", 1.0),
                    source_failure_fact_id=value.get(
                        "source_failure_fact_id"
                    ),
                )
                for value in draft.get("occurrences", ())
            ),
            abnormal_intervals=tuple(
                AbnormalIntervalAnnotationV1(
                    annotation_id=value.get("annotation_id", ""),
                    start_at=value.get("start_at"),
                    end_at=value.get("end_at"),
                    clock_domain_id=clock,
                    reason_code=value.get("reason_code", ""),
                    note=value.get("note"),
                )
                for value in draft.get("abnormal_intervals", ())
            ),
            progress_boundary_overrides=tuple(
                ProgressBoundaryOverrideAnnotationV1(
                    annotation_id=value.get("annotation_id", ""),
                    source_advance_fact_id=value.get(
                        "source_advance_fact_id",
                        "",
                    ),
                    corrected_effective_at=value.get(
                        "corrected_effective_at"
                    ),
                    clock_domain_id=clock,
                )
                for value in draft.get(
                    "progress_boundary_overrides",
                    (),
                )
            ),
            capture_end_step=(
                None
                if draft.get("capture_end_step") is None
                else CaptureEndStepAnnotationV1(
                    annotation_id=draft["capture_end_step"].get(
                        "annotation_id",
                        "",
                    ),
                    source_terminal_fact_id=draft["capture_end_step"].get(
                        "source_terminal_fact_id",
                        "",
                    ),
                    source_attempt_id=draft["capture_end_step"].get(
                        "source_attempt_id",
                        "",
                    ),
                    end_progress_step_id=draft["capture_end_step"].get(
                        "end_progress_step_id",
                        "",
                    ),
                    clock_domain_id=clock,
                )
            ),
            episode_abnormal=(
                None
                if draft.get("episode_abnormal") is None
                else EpisodeAbnormalAnnotationV1(
                    annotation_id=draft["episode_abnormal"].get(
                        "annotation_id",
                        "",
                    ),
                    reason_code=draft["episode_abnormal"].get(
                        "reason_code",
                        "",
                    ),
                    note=draft["episode_abnormal"].get("note"),
                )
            ),
            episode_review_request=(
                None
                if draft.get("episode_review_request") is None
                else EpisodeReviewRequestAnnotationV1(
                    annotation_id=draft["episode_review_request"].get(
                        "annotation_id",
                        "",
                    ),
                    note=draft["episode_review_request"].get("note"),
                )
            ),
        )

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("Refinement annotation application is closed.")


__all__ = [
    "RefinementAnnotationApplication",
    "RefinementAnnotationApplicationError",
]

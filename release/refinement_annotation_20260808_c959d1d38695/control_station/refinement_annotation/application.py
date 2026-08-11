"""Application coordination for post-capture fine-stage annotation."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import errno
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
from dataset_contracts.progress_stage_annotations import (
    AbnormalIntervalAnnotationV1,
    EpisodeAbnormalAnnotationV1,
    EpisodeReviewRequestAnnotationV1,
    ProgressBoundaryOverrideAnnotationV1,
    StageAnnotationBundleV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
)
from dataset_interface.progress_annotation_jobs import (
    ProgressAnnotationCatalogCursor,
    ProgressAnnotationWorkItem,
    load_episode_sample_axis,
    progress_annotation_catalog_fingerprint,
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
from dataset_interface.sample_axis import EpisodeSampleAxis


_DRAFT_SCHEMA = "dexteleop.stage_annotation_draft.local.v1"
_SAMPLE_AXIS_CACHE_LIMIT = 16
_ANNOTATION_SET_CACHE_LIMIT = 16
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


class RefinementAnnotationApplicationError(ValueError):
    """Raised when a UI request cannot preserve annotation semantics."""


@dataclass(frozen=True)
class _EpisodeAnnotationSummary:
    annotation_set_count: int
    review_state: str
    latest_annotation_mode: str | None
    whole_episode_abnormal: bool
    whole_episode_abnormal_reason: str | None

    @property
    def needs_second_review(self) -> bool:
        return self.review_state == "PENDING"


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
        self.workspace = DatasetStageAnnotationWorkspace(
            self.dataset_root,
            annotator_id=self.annotator_id,
        )
        self.workspace.acquire()
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
        self._refresh_in_flight = False
        self._refresh_thread: threading.Thread | None = None
        self._refresh_error: Exception | None = None
        self._refresh_completed = 0
        self._refresh_total: int | None = None
        self._annotation_sets_cache: OrderedDict[
            str,
            tuple[LoadedStageAnnotationSet, ...],
        ] = OrderedDict()
        self._annotation_summary_cache: dict[
            str,
            _EpisodeAnnotationSummary,
        ] = {}
        self._episode_list_revision = 0
        self._sample_axis_cache: OrderedDict[
            str,
            tuple[tuple[object, ...], EpisodeSampleAxis],
        ] = OrderedDict()
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
                self._annotation_sets_cache.clear()
                self._annotation_summary_cache.clear()
                self._sample_axis_cache.clear()

    def _catalog_fingerprint(self) -> tuple[object, ...]:
        return progress_annotation_catalog_fingerprint(self.dataset_root)

    def _finish_refresh(
        self,
        items: dict[str, ProgressAnnotationWorkItem] | None,
        cursor: ProgressAnnotationCatalogCursor | None,
        catalog_fingerprint: tuple[object, ...] | None,
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
                self._refresh_error = None
                self._episode_list_revision += 1
                eligible = set(items)
                for cache in (
                    self._annotation_sets_cache,
                    self._annotation_summary_cache,
                    self._sample_axis_cache,
                ):
                    for episode_uuid in tuple(cache):
                        if episode_uuid not in eligible:
                            cache.pop(episode_uuid, None)
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
            batch = read_progress_annotation_work_item_batch(
                self.dataset_root,
                cursor=cursor,
                allowed_source_roots=self.allowed_source_roots,
                progress_callback=self._update_refresh_progress,
                # Exact badges are loaded lazily for visible rows. Avoid an
                # O(Episodes x annotation versions) SSHFS walk at startup.
                load_annotation_state=False,
            )
        except Exception as exc:
            self._finish_refresh(None, None, None, exc)
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
        self._finish_refresh(
            items,
            batch.cursor,
            batch.catalog_fingerprint,
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

        # A forced scan does not need a pre-scan fingerprint. This matters for
        # retrospective workspaces whose fingerprint itself enumerates every
        # Episode on a remote mount.
        catalog_fingerprint = None if force else self._catalog_fingerprint()
        with self._lock:
            self._require_open()
            needs_refresh = (
                force
                or self._refresh_error is not None
                or catalog_fingerprint != self._known_catalog_fingerprint
            )
            if not needs_refresh:
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

    def _sample_axis(self, item: ProgressAnnotationWorkItem) -> EpisodeSampleAxis:
        """Reuse immutable Episode timestamps while the annotation app is open."""

        cache_key = self._sample_axis_cache_key(item)
        with self._lock:
            cached = self._sample_axis_cache.get(item.episode_uuid)
            if cached is not None and cached[0] == cache_key:
                self._sample_axis_cache.move_to_end(item.episode_uuid)
                return cached[1]

        sample_axis = load_episode_sample_axis(item)
        with self._lock:
            self._sample_axis_cache[item.episode_uuid] = (
                cache_key,
                sample_axis,
            )
            self._sample_axis_cache.move_to_end(item.episode_uuid)
            while len(self._sample_axis_cache) > _SAMPLE_AXIS_CACHE_LIMIT:
                self._sample_axis_cache.popitem(last=False)
        return sample_axis

    def list_episodes(
        self,
        *,
        wait_for_refresh: bool = True,
        force_refresh: bool = False,
        include_annotation_summaries: bool = True,
    ) -> dict[str, Any]:
        self.refresh(wait=wait_for_refresh, force=force_refresh)
        with self._lock:
            self._require_open()
            items = tuple(self._items.values())
            revision = self._episode_list_revision
            refreshing = self._refresh_in_flight
            refresh_error = self._refresh_error
            refresh_completed = self._refresh_completed
            refresh_total = self._refresh_total
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
            "episodes": [
                self._episode_summary(
                    item,
                    load_annotations=include_annotation_summaries,
                )
                for item in sorted(
                    items,
                    key=lambda value: value.episode_index,
                )
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
        summaries: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for episode_uuid in normalized:
            try:
                item = self._item(episode_uuid)
                with self._summary_load_lock(episode_uuid):
                    with self._lock:
                        active = self._active_episode_uuid
                        if active != episode_uuid:
                            # A different machine may have published since
                            # this row was last visible. Immutable manifest-last
                            # sets are safe to rediscover without its edit lock.
                            self._annotation_sets_cache.pop(episode_uuid, None)
                            self._annotation_summary_cache.pop(episode_uuid, None)
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
                errors.append(
                    {"episode_uuid": episode_uuid, "error": str(exc)}
                )
                continue
            summaries.append(
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
                    "annotation_summary_loaded": True,
                }
            )
        return {"summaries": summaries, "errors": errors}

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
                        # Another machine may have committed while this Episode
                        # was not locally owned.  Reload after acquiring its
                        # cross-machine lock and after any summary read finishes.
                        self._annotation_sets_cache.pop(normalized, None)
                        self._annotation_summary_cache.pop(normalized, None)
                    try:
                        detail = self._episode_detail_owned(normalized)
                    except Exception:
                        # Do not retain partially refreshed target state after a
                        # failed open; a later retry must start from storage.
                        with self._lock:
                            self._annotation_sets_cache.pop(normalized, None)
                            self._annotation_summary_cache.pop(normalized, None)
                            self._sample_axis_cache.pop(normalized, None)
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
                        self._sample_axis_cache.pop(normalized, None)
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
        stage_profile = item.progress.stage_profile
        stage_entry = item.progress.stage_entry_binding
        assert stage_profile is not None
        assert stage_entry is not None
        annotation_sets = self._sealed_annotation_sets(item)
        sample_axis = self._sample_axis(item)
        try:
            stored_draft = self.workspace.load_draft(
                episode_uuid=episode_uuid
            )
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
            "annotation_summary_loaded": True,
            "annotation_mode": self.annotation_mode,
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
            "entry_stage_id": stage_entry.entry_stage_id,
            "entry_condition": stage_entry.entry_condition,
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
    ) -> _EpisodeAnnotationSummary:
        latest = values[-1] if values else None
        latest_mode = None if latest is None else latest.binding.annotation_mode
        needs_review = bool(
            latest is not None
            and latest.bundle.episode_review_request is not None
        )
        review_state = (
            "PENDING"
            if needs_review
            else (
                "COMPLETED"
                if latest_mode == "SECOND_REVIEW"
                else "NONE"
            )
        )
        episode_abnormal = (
            None if latest is None else latest.bundle.episode_abnormal
        )
        return _EpisodeAnnotationSummary(
            annotation_set_count=len(values),
            review_state=review_state,
            latest_annotation_mode=latest_mode,
            whole_episode_abnormal=episode_abnormal is not None,
            whole_episode_abnormal_reason=(
                None
                if episode_abnormal is None
                else episode_abnormal.reason_code
            ),
        )

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
        if annotation_summary is None and load_annotations:
            annotation_sets = self._sealed_annotation_sets(item)
            annotation_summary = self._summarize_annotation_sets(
                annotation_sets
            )
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
            "annotation_summary_loaded": annotation_summary is not None,
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
        committed_draft = {
            **normalized,
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
            "draft_cache_warning": draft_cache_warning,
        }

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
            touches_abnormal = any(
                math.isclose(
                    float(occurrence["start_at"]),
                    abnormal.end_at,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
                or math.isclose(
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
        if not root.is_dir():
            self._cache_annotation_sets(item.episode_uuid, ())
            return ()
        assert item.progress.stage_profile is not None
        loaded: list[LoadedStageAnnotationSet] = []
        for child in sorted(root.iterdir(), key=lambda value: value.name):
            if not child.is_dir() or not (child / "manifest.cjson").is_file():
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
        result = tuple(
            sorted(
                loaded,
                key=lambda value: (
                    value.binding.created_at,
                    value.binding.annotation_set_id,
                ),
            )
        )
        self._cache_annotation_sets(item.episode_uuid, result)
        return result

    def _cache_annotation_sets(
        self,
        episode_uuid: str,
        values: tuple[LoadedStageAnnotationSet, ...],
    ) -> None:
        summary = self._summarize_annotation_sets(values)
        with self._lock:
            previous_summary = self._annotation_summary_cache.get(episode_uuid)
            self._annotation_sets_cache[episode_uuid] = values
            self._annotation_sets_cache.move_to_end(episode_uuid)
            self._annotation_summary_cache[episode_uuid] = summary
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
                self._annotation_sets_cache.popitem(last=False)

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
            "episode_abnormal": episode_abnormal,
            "episode_review_request": episode_review_request,
            "privileged_tools_enabled": bool(
                bundle.abnormal_intervals
                or bundle.progress_boundary_overrides
                or bundle.episode_abnormal is not None
            ),
            "last_committed_annotation_set_id": (
                value.binding.annotation_set_id
            ),
            "last_committed_manifest_hash": (
                value.manifest.semantic_content_hash
            ),
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
        for attempt in item.progress.timeline.attempts:
            if attempt.end_at is None or attempt.end_at <= attempt.start_at:
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
            duration = attempt.end_at - attempt.start_at
            for index, (stage, outcome) in enumerate(
                zip(selected, outcomes),
                start=1,
            ):
                start_at = attempt.start_at + (
                    duration * (index - 1) / len(selected)
                )
                end_at = attempt.start_at + (
                    duration * index / len(selected)
                )
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
            "episode_abnormal": None,
            "episode_review_request": None,
            "privileged_tools_enabled": False,
        }

    def _normalize_draft(
        self,
        item: ProgressAnnotationWorkItem,
        draft: Mapping[str, Any],
        *,
        sample_axis: Any | None = None,
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
            "episode_abnormal",
            "episode_review_request",
            "privileged_tools_enabled",
            "last_committed_annotation_set_id",
            "last_committed_manifest_hash",
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
        episode_abnormal = draft.get("episode_abnormal")
        episode_review_request = draft.get("episode_review_request")
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
            "episode_abnormal": episode_abnormal,
            "episode_review_request": episode_review_request,
            "privileged_tools_enabled": privileged,
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

"""Application coordination for post-capture fine-stage annotation."""

from __future__ import annotations

from collections import OrderedDict
import math
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from collection_runtime.progress import (
    STAGE_ANNOTATION_RELATIVE_ROOT,
    DatasetStageAnnotationWorkspace,
    LoadedStageAnnotationSet,
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
from dataset_contracts.causal_snapshot import LOCAL_WALL_CLOCK
from dataset_contracts.progress_failure_triggers import (
    FailureTriggerAnnotationSetV1,
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
    ProgressAnnotationWorkItem,
    list_progress_annotation_work_items,
    load_episode_sample_axis,
)
from dataset_interface.progress_failure_triggers import (
    FailureTriggerResolutionError,
    build_failure_trigger_annotations,
    failure_trigger_sample_axis_hash,
    normalize_failure_trigger_drafts,
)
from dataset_interface.progress_stages import resolve_stage_training_index
from dataset_interface.sample_axis import EpisodeSampleAxis


_DRAFT_SCHEMA = "dexteleop.stage_annotation_draft.local.v1"
_SAMPLE_AXIS_CACHE_LIMIT = 16


class RefinementAnnotationApplicationError(ValueError):
    """Raised when a UI request cannot preserve annotation semantics."""


class RefinementAnnotationApplication:
    """Own one exclusive annotation workspace, never dataset capture."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        annotator_id: str,
        second_review: bool = False,
    ) -> None:
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.annotator_id = str(annotator_id).strip()
        if not self.annotator_id:
            raise ValueError("annotator_id must be non-empty.")
        self.annotation_mode = (
            "SECOND_REVIEW" if bool(second_review) else "STANDARD"
        )
        self.workspace = DatasetStageAnnotationWorkspace(self.dataset_root)
        self.workspace.acquire()
        self._lock = threading.RLock()
        self._closed = False
        self._items: dict[str, ProgressAnnotationWorkItem] = {}
        self._annotation_sets_cache: dict[
            str,
            tuple[LoadedStageAnnotationSet, ...],
        ] = {}
        self._sample_axis_cache: OrderedDict[
            str,
            tuple[tuple[int, int, int, int], EpisodeSampleAxis],
        ] = OrderedDict()
        self.refresh()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.workspace.close()

    def refresh(self) -> None:
        with self._lock:
            self._require_open()
            self._items = {
                item.episode_uuid: item
                for item in list_progress_annotation_work_items(
                    self.dataset_root
                )
            }

    @staticmethod
    def _sample_axis_fingerprint(
        item: ProgressAnnotationWorkItem,
    ) -> tuple[int, int, int, int] | None:
        try:
            stat = item.parquet_path.stat()
        except OSError:
            return None
        return (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
        )

    def _sample_axis(self, item: ProgressAnnotationWorkItem) -> EpisodeSampleAxis:
        """Reuse immutable Episode timestamps while the annotation app is open."""

        fingerprint = self._sample_axis_fingerprint(item)
        if fingerprint is not None:
            with self._lock:
                cached = self._sample_axis_cache.get(item.episode_uuid)
                if cached is not None and cached[0] == fingerprint:
                    self._sample_axis_cache.move_to_end(item.episode_uuid)
                    return cached[1]

        sample_axis = load_episode_sample_axis(item)
        if fingerprint is not None:
            with self._lock:
                self._sample_axis_cache[item.episode_uuid] = (
                    fingerprint,
                    sample_axis,
                )
                self._sample_axis_cache.move_to_end(item.episode_uuid)
                while len(self._sample_axis_cache) > _SAMPLE_AXIS_CACHE_LIMIT:
                    self._sample_axis_cache.popitem(last=False)
        return sample_axis

    def list_episodes(self) -> dict[str, Any]:
        self.refresh()
        return {
            "dataset_root": str(self.dataset_root),
            "annotator_id": self.annotator_id,
            "annotation_mode": self.annotation_mode,
            "episodes": [
                self._episode_summary(item)
                for item in sorted(
                    self._items.values(),
                    key=lambda value: value.episode_index,
                )
            ],
        }

    def episode_detail(self, episode_uuid: str) -> dict[str, Any]:
        item = self._item(episode_uuid)
        stage_profile = item.progress.stage_profile
        stage_entry = item.progress.stage_entry_binding
        assert stage_profile is not None
        assert stage_entry is not None
        annotation_sets = self._sealed_annotation_sets(item)
        sample_axis = self._sample_axis(item)
        draft = self.workspace.load_draft(episode_uuid=episode_uuid)
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
            draft = self._bootstrap_draft(item)
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
            draft = {
                **draft,
                "draft_annotation_mode": "SECOND_REVIEW",
                "episode_review_request": None,
            }
            # The prior immutable set remains excluded until this review draft
            # is explicitly sealed; only the mutable working label is cleared
            # automatically on open.
            self.workspace.save_draft(
                episode_uuid=episode_uuid,
                draft=draft,
            )
        step_by_id = item.progress.profile.step_by_id
        return {
            **item.summary_payload(),
            "annotation_set_count": len(annotation_sets),
            "annotation_mode": self.annotation_mode,
            "review_label_auto_cleared": review_label_auto_cleared,
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

    def _episode_summary(
        self,
        item: ProgressAnnotationWorkItem,
    ) -> dict[str, Any]:
        annotation_sets = self._sealed_annotation_sets(item)
        latest = annotation_sets[-1] if annotation_sets else None
        return {
            **item.summary_payload(),
            "annotation_set_count": len(annotation_sets),
            "needs_second_review": bool(
                latest is not None
                and latest.bundle.episode_review_request is not None
            ),
            "latest_annotation_mode": (
                None if latest is None else latest.binding.annotation_mode
            ),
        }

    def reopen_sealed_annotation(
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
        item = self._item(episode_uuid)
        normalized = self._normalize_draft(item, draft)
        self.workspace.save_draft(
            episode_uuid=episode_uuid,
            draft=normalized,
        )
        return normalized

    def commit(
        self,
        episode_uuid: str,
        draft: Mapping[str, Any],
    ) -> dict[str, Any]:
        item = self._item(episode_uuid)
        normalized = self._normalize_draft(item, draft)
        normalized = self._project_occurrences_around_abnormal_intervals(
            item,
            normalized,
        )
        bundle = self._bundle_from_draft(item, normalized)
        sample_axis = self._sample_axis(item)
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
        report = commit_stage_annotation_set(
            dataset_root=self.dataset_root,
            episode_uuid=episode_uuid,
            source_progress_manifest=item.progress.manifest,
            stage_profile=item.progress.stage_profile,
            binding=binding,
            bundle=bundle,
        )
        try:
            trigger_report = commit_failure_trigger_annotation_set(
                annotation_dir=report.annotation_dir,
                annotation_set=trigger_set,
            )
        except FailureTriggerStoreError as exc:
            raise RefinementAnnotationApplicationError(
                "Stage annotation set was committed, but its optional "
                f"failure-trigger sidecar could not be committed: {exc}"
            ) from exc
        committed_draft = {
            **normalized,
            "last_committed_annotation_set_id": annotation_set_id,
            "last_committed_manifest_hash": (
                report.manifest.semantic_content_hash
            ),
        }
        self.workspace.save_draft(
            episode_uuid=episode_uuid,
            draft=committed_draft,
        )
        self._annotation_sets_cache.pop(episode_uuid, None)
        self.refresh()
        return {
            "annotation_set_id": annotation_set_id,
            "manifest_hash": report.manifest.semantic_content_hash,
            "failure_trigger_manifest_hash": (
                trigger_report.semantic_content_hash
            ),
            "failure_trigger_count": len(trigger_set.annotations),
            "annotator_id": self.annotator_id,
            "annotation_mode": self.annotation_mode,
            "created_at": binding.created_at,
            "coverage_status": index.coverage_status,
            "annotated_sample_count": index.annotated_sample_count,
            "unannotated_sample_count": index.unannotated_sample_count,
            "exclusion_count": len(index.exclusions),
            "draft": committed_draft,
        }

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

        raw_abnormal = tuple(draft.get("abnormal_intervals", ()))
        clock = item.progress.timeline.clock_domain_id
        try:
            abnormal_intervals = tuple(
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
                        for value in raw_abnormal
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

        projected.sort(
            key=lambda value: (
                float(value["start_at"]),
                float(value["end_at"]),
                str(value.get("annotation_id", "")),
            )
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
        projected = coalesced

        seen_ids: set[str] = set()
        occurrence_index_by_attempt: dict[str, int] = {}
        for occurrence in projected:
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
        cached = self._annotation_sets_cache.get(item.episode_uuid)
        if cached is not None:
            return cached
        root = (
            self.dataset_root
            / STAGE_ANNOTATION_RELATIVE_ROOT
            / item.episode_uuid
        )
        if not root.is_dir():
            self._annotation_sets_cache[item.episode_uuid] = ()
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
        self._annotation_sets_cache[item.episode_uuid] = result
        return result

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
        }

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

    def _item(self, episode_uuid: str) -> ProgressAnnotationWorkItem:
        with self._lock:
            self._require_open()
            normalized = str(episode_uuid).strip()
            item = self._items.get(normalized)
            if item is None:
                self.refresh()
                item = self._items.get(normalized)
            if item is None:
                raise RefinementAnnotationApplicationError(
                    "Episode is not finalized, sealed, stage-profiled, or "
                    "otherwise eligible for annotation."
                )
            return item

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
        deletion_history = list(draft.get("deletion_history", ()))
        episode_abnormal = draft.get("episode_abnormal")
        episode_review_request = draft.get("episode_review_request")
        for name, values in (
            ("occurrences", occurrences),
            ("failure_triggers", failure_triggers),
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
        return {
            "schema": _DRAFT_SCHEMA,
            "episode_uuid": item.episode_uuid,
            "draft_annotation_mode": self.annotation_mode,
            "draft_revision": int(draft.get("draft_revision", 0)),
            "occurrences": occurrences,
            "failure_triggers": failure_triggers,
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

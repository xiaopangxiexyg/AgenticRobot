from __future__ import annotations

from pathlib import Path

import pytest

from collection_runtime.progress import (
    DatasetStageAnnotationWorkspace,
    StageAnnotationStoreError,
    commit_stage_annotation_set,
    load_stage_annotation_set,
)
from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stage_annotations import (
    AbnormalIntervalAnnotationV1,
    EpisodeAbnormalAnnotationV1,
    EpisodeReviewRequestAnnotationV1,
    StageAnnotationBundleV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageContractError,
    ProgressStageDefinitionV1,
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
    validate_progress_stage_entry_binding,
)
from dataset_interface.progress_stages import resolve_stage_training_index
from dataset_interface.progress_timeline import (
    ResolvedProgressAttempt,
    ResolvedProgressTimeline,
)
from dataset_interface.sample_axis import EpisodeSampleAxis


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _profiles() -> tuple[TaskProgressProfileV1, TaskStageProfileV1]:
    progress = TaskProgressProfileV1(
        profile_id="fixture-progress",
        profile_version="1",
        task_id="fixture",
        steps=(
            ProgressStepDefinitionV1("a", "TASK_A", "A", 1),
        ),
        initial_step_id="a",
        terminal_step_ids=("a",),
    )
    stages = TaskStageProfileV1(
        profile_id="fixture-stages",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        stages=(
            ProgressStageDefinitionV1("a", "a1", "A1", "first", 1),
            ProgressStageDefinitionV1("a", "a2", "A2", "second", 2),
            ProgressStageDefinitionV1("a", "a3", "A3", "third", 3),
        ),
    )
    return progress, stages


def _binding(
    progress: TaskProgressProfileV1,
) -> ProgressEpisodeBindingV1:
    return ProgressEpisodeBindingV1(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=progress.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope="FULL_TASK",
        entry_step_id="a",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )


def _timeline(
    progress: TaskProgressProfileV1,
    binding: ProgressEpisodeBindingV1,
) -> ResolvedProgressTimeline:
    return ResolvedProgressTimeline(
        episode_capture_identity_hash=binding.episode_capture_identity_hash,
        task_progress_profile_hash=progress.semantic_content_hash,
        progress_episode_binding_hash=binding.semantic_content_hash,
        clock_domain_id="episode_clock",
        fact_bundle_hash="2" * 64,
        fact_count=8,
        sealed=True,
        phase="TERMINATED",
        attempts=(
            ResolvedProgressAttempt(
                attempt_id="attempt:a:1",
                progress_step_id="a",
                attempt_ordinal_in_capture=1,
                start_fact_id="start",
                start_at=0.0,
                clock_domain_id="episode_clock",
                end_fact_id="failure:1",
                end_at=3.0,
                outcome="FAILURE",
                failure_fact_id="failure:1",
                failure_reported_at=3.0,
                failure_timing_semantics="OPERATOR_REPORT_TIME",
            ),
            ResolvedProgressAttempt(
                attempt_id="attempt:a:2",
                progress_step_id="a",
                attempt_ordinal_in_capture=2,
                start_fact_id="retry:1",
                start_at=4.0,
                clock_domain_id="episode_clock",
                end_fact_id="termination",
                end_at=8.0,
                outcome="SUCCESS",
            ),
        ),
        transitions=(),
        capture_pause_intervals=(),
        undone_advance_fact_ids=(),
        invalidated_attempt_ids=(),
        current_progress_step_id="a",
        active_attempt_id=None,
        terminal_fact_id="termination",
        termination_cause="FULL_TASK_COMPLETED",
        full_task_outcome="SUCCESS",
        source_progress_manifest_hash="3" * 64,
    )


def test_noninitial_stage_entry_requires_explicit_recovery() -> None:
    progress, stages = _profiles()
    binding = ProgressStageEntryBindingV1(
        episode_capture_identity_hash="1" * 64,
        task_stage_profile_hash=stages.semantic_content_hash,
        entry_stage_id="a2",
        entry_condition="NOMINAL",
    )

    with pytest.raises(
        ProgressStageContractError,
        match="RECOVERY_FROM_FAILURE",
    ):
        validate_progress_stage_entry_binding(
            progress_profile=progress,
            progress_entry_step_id="a",
            stage_profile=stages,
            stage_binding=binding,
        )

    recovery = ProgressStageEntryBindingV1(
        episode_capture_identity_hash="1" * 64,
        task_stage_profile_hash=stages.semantic_content_hash,
        entry_stage_id="a2",
        entry_condition="RECOVERY_FROM_FAILURE",
    )
    validate_progress_stage_entry_binding(
        progress_profile=progress,
        progress_entry_step_id="a",
        stage_profile=stages,
        stage_binding=recovery,
    )


def test_failure_retry_preserves_stage_and_excludes_only_failed_occurrence() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)
    timeline = _timeline(progress, binding)
    annotations = StageAnnotationBundleV1(
        occurrences=(
            StageOccurrenceAnnotationV1(
                "occ:a1",
                "attempt:a:1",
                "a",
                "a1",
                1,
                0.0,
                1.0,
                "episode_clock",
                "SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                "occ:a2:failed",
                "attempt:a:1",
                "a",
                "a2",
                2,
                1.0,
                3.0,
                "episode_clock",
                "FAILURE",
                source_failure_fact_id="failure:1",
            ),
            StageOccurrenceAnnotationV1(
                "occ:a2:retry",
                "attempt:a:2",
                "a",
                "a2",
                1,
                4.0,
                6.0,
                "episode_clock",
                "SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                "occ:a3",
                "attempt:a:2",
                "a",
                "a3",
                2,
                6.0,
                8.0,
                "episode_clock",
                "SUCCESS",
            ),
        )
    )

    index = resolve_stage_training_index(
        progress_profile=progress,
        progress_binding=binding,
        stage_profile=stages,
        stage_entry_binding=ProgressStageEntryBindingV1(
            episode_capture_identity_hash="1" * 64,
            task_stage_profile_hash=stages.semantic_content_hash,
            entry_stage_id="a1",
        ),
        timeline=timeline,
        annotations=annotations,
        sample_axis=EpisodeSampleAxis(
            "episode_clock",
            tuple(float(index) for index in range(9)),
        ),
    )

    assert index.coverage_status == "COMPLETE"
    assert tuple(item.stage_id for item in index.occurrences) == (
        "a1",
        "a2",
        "a2",
        "a3",
    )
    assert len(index.exclusions) == 1
    exclusion = index.exclusions[0]
    assert exclusion.exclusion_kind == "FAILED_STAGE_OCCURRENCE"
    assert exclusion.sample_interval.sample_index_begin == 1
    assert exclusion.sample_interval.sample_index_end_exclusive == 4


def test_whole_episode_abnormal_is_complete_exclusion_without_stage_guessing() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)

    index = resolve_stage_training_index(
        progress_profile=progress,
        progress_binding=binding,
        stage_profile=stages,
        stage_entry_binding=None,
        timeline=_timeline(progress, binding),
        annotations=StageAnnotationBundleV1(
            episode_abnormal=EpisodeAbnormalAnnotationV1(
                annotation_id="episode:bad",
                reason_code="BAD_CAPTURE",
            )
        ),
        sample_axis=EpisodeSampleAxis(
            "episode_clock",
            tuple(float(index) for index in range(9)),
        ),
    )

    assert index.coverage_status == "EXCLUDED_EPISODE"
    assert index.exclusions[0].sample_interval.sample_count == 9


def test_pending_second_review_is_whole_episode_exclusion_not_abnormal() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)

    index = resolve_stage_training_index(
        progress_profile=progress,
        progress_binding=binding,
        stage_profile=stages,
        stage_entry_binding=None,
        timeline=_timeline(progress, binding),
        annotations=StageAnnotationBundleV1(
            episode_review_request=EpisodeReviewRequestAnnotationV1(
                annotation_id="episode:review",
                note="complex occasional behavior",
            )
        ),
        sample_axis=EpisodeSampleAxis(
            "episode_clock",
            tuple(float(index) for index in range(9)),
        ),
    )

    assert index.coverage_status == "EXCLUDED_EPISODE"
    assert index.exclusions[0].exclusion_kind == (
        "WHOLE_EPISODE_REVIEW_PENDING"
    )
    assert index.exclusions[0].sample_interval.sample_count == 9


def test_abnormal_interval_may_cover_fixed_capture_failure() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)
    timeline = _timeline(progress, binding)

    index = resolve_stage_training_index(
        progress_profile=progress,
        progress_binding=binding,
        stage_profile=stages,
        stage_entry_binding=None,
        timeline=timeline,
        annotations=StageAnnotationBundleV1(
            occurrences=(
                StageOccurrenceAnnotationV1(
                    "occ:a1",
                    "attempt:a:1",
                    "a",
                    "a1",
                    1,
                    0.0,
                    1.0,
                    "episode_clock",
                    "SUCCESS",
                ),
                StageOccurrenceAnnotationV1(
                    "occ:a2:retry",
                    "attempt:a:2",
                    "a",
                    "a2",
                    1,
                    4.0,
                    6.0,
                    "episode_clock",
                    "SUCCESS",
                ),
                StageOccurrenceAnnotationV1(
                    "occ:a3",
                    "attempt:a:2",
                    "a",
                    "a3",
                    2,
                    6.0,
                    8.0,
                    "episode_clock",
                    "SUCCESS",
                ),
            ),
            abnormal_intervals=(
                AbnormalIntervalAnnotationV1(
                    annotation_id="abnormal:failure",
                    start_at=1.0,
                    end_at=3.5,
                    clock_domain_id="episode_clock",
                    reason_code="BAD_CAPTURE",
                ),
            ),
        ),
        sample_axis=EpisodeSampleAxis(
            "episode_clock",
            tuple(float(sample_index) for sample_index in range(9)),
        ),
    )

    assert tuple(item.stage_id for item in index.occurrences) == (
        "a1",
        "a2",
        "a3",
    )
    assert tuple(item.exclusion_kind for item in index.exclusions) == (
        "ABNORMAL_INTERVAL",
    )


def test_abnormal_truncation_must_touch_the_abnormal_boundary() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)

    with pytest.raises(
        ValueError,
        match="must touch an abnormal interval boundary",
    ):
        resolve_stage_training_index(
            progress_profile=progress,
            progress_binding=binding,
            stage_profile=stages,
            stage_entry_binding=None,
            timeline=_timeline(progress, binding),
            annotations=StageAnnotationBundleV1(
                occurrences=(
                    StageOccurrenceAnnotationV1(
                        "occ:a1:truncated",
                        "attempt:a:1",
                        "a",
                        "a1",
                        1,
                        0.0,
                        0.5,
                        "episode_clock",
                        "TRUNCATED_BY_ABNORMAL",
                    ),
                ),
                abnormal_intervals=(
                    AbnormalIntervalAnnotationV1(
                        annotation_id="abnormal:later",
                        start_at=1.0,
                        end_at=2.0,
                        clock_domain_id="episode_clock",
                        reason_code="BAD_CAPTURE",
                    ),
                ),
            ),
            sample_axis=EpisodeSampleAxis(
                "episode_clock",
                tuple(float(sample_index) for sample_index in range(9)),
            ),
        )


def test_deliberately_omitted_leading_stage_keeps_later_identity() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)

    index = resolve_stage_training_index(
        progress_profile=progress,
        progress_binding=binding,
        stage_profile=stages,
        stage_entry_binding=ProgressStageEntryBindingV1(
            episode_capture_identity_hash="1" * 64,
            task_stage_profile_hash=stages.semantic_content_hash,
            entry_stage_id="a1",
        ),
        timeline=_timeline(progress, binding),
        annotations=StageAnnotationBundleV1(
            occurrences=(
                StageOccurrenceAnnotationV1(
                    "occ:a2:failed",
                    "attempt:a:1",
                    "a",
                    "a2",
                    1,
                    1.0,
                    3.0,
                    "episode_clock",
                    "FAILURE",
                    source_failure_fact_id="failure:1",
                ),
                StageOccurrenceAnnotationV1(
                    "occ:a2:retry",
                    "attempt:a:2",
                    "a",
                    "a2",
                    1,
                    4.0,
                    6.0,
                    "episode_clock",
                    "SUCCESS",
                ),
                StageOccurrenceAnnotationV1(
                    "occ:a3",
                    "attempt:a:2",
                    "a",
                    "a3",
                    2,
                    6.0,
                    8.0,
                    "episode_clock",
                    "SUCCESS",
                ),
            )
        ),
        sample_axis=EpisodeSampleAxis(
            "episode_clock",
            tuple(float(index) for index in range(9)),
        ),
    )

    assert tuple(item.stage_id for item in index.occurrences) == (
        "a2",
        "a2",
        "a3",
    )
    assert index.coverage_status == "PARTIAL"
    assert index.unannotated_sample_count == 1


def test_posthoc_final_stage_failure_preserves_capture_success_fact() -> None:
    progress, stages = _profiles()
    binding = _binding(progress)
    annotations = StageAnnotationBundleV1(
        occurrences=(
            StageOccurrenceAnnotationV1(
                "occ:a1",
                "attempt:a:1",
                "a",
                "a1",
                1,
                0.0,
                1.0,
                "episode_clock",
                "SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                "occ:a2:failed",
                "attempt:a:1",
                "a",
                "a2",
                2,
                1.0,
                3.0,
                "episode_clock",
                "FAILURE",
                source_failure_fact_id="failure:1",
            ),
            StageOccurrenceAnnotationV1(
                "occ:a2:retry",
                "attempt:a:2",
                "a",
                "a2",
                1,
                4.0,
                6.0,
                "episode_clock",
                "SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                "occ:a3:posthoc-failed",
                "attempt:a:2",
                "a",
                "a3",
                2,
                6.0,
                8.0,
                "episode_clock",
                "FAILURE",
            ),
        )
    )

    index = resolve_stage_training_index(
        progress_profile=progress,
        progress_binding=binding,
        stage_profile=stages,
        stage_entry_binding=None,
        timeline=_timeline(progress, binding),
        annotations=annotations,
        sample_axis=EpisodeSampleAxis(
            "episode_clock",
            tuple(float(index) for index in range(9)),
        ),
    )

    assert tuple(item.outcome for item in index.occurrences[-2:]) == (
        "SUCCESS",
        "FAILURE",
    )
    assert tuple(item.exclusion_kind for item in index.exclusions) == (
        "FAILED_STAGE_OCCURRENCE",
        "FAILED_STAGE_OCCURRENCE",
    )


def test_annotation_workspace_lock_and_immutable_commit(tmp_path: Path) -> None:
    progress, stages = _profiles()
    manifest = ProgressEpisodeManifestV1(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=progress.semantic_content_hash,
        progress_episode_binding_hash="2" * 64,
        fact_count=1,
        fact_bundle_hash="3" * 64,
        terminal_fact_id="termination",
        sealed_at=9.0,
        sealed_clock_domain_id="episode_clock",
    )
    binding = StageAnnotationSetBindingV1(
        annotation_set_id="set-1",
        episode_capture_identity_hash="1" * 64,
        source_progress_manifest_hash=manifest.semantic_content_hash,
        task_stage_profile_hash=stages.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        annotator_id="fixture",
        created_at=10.0,
        created_clock_domain_id="wall_clock",
    )
    bundle = StageAnnotationBundleV1(
        episode_abnormal=EpisodeAbnormalAnnotationV1(
            annotation_id="episode:bad",
            reason_code="BAD_CAPTURE",
        )
    )

    with DatasetStageAnnotationWorkspace(tmp_path) as workspace:
        workspace.save_draft(
            episode_uuid="episode-1",
            draft={"cursor": 1.5},
        )
        assert workspace.load_draft(
            episode_uuid="episode-1"
        ) == {"cursor": 1.5}
        with pytest.raises(StageAnnotationStoreError, match="already owns"):
            with DatasetStageAnnotationWorkspace(tmp_path):
                pass

        report = commit_stage_annotation_set(
            dataset_root=tmp_path,
            episode_uuid="episode-1",
            source_progress_manifest=manifest,
            stage_profile=stages,
            binding=binding,
            bundle=bundle,
        )

    loaded = load_stage_annotation_set(report.annotation_dir)
    assert loaded.binding == binding
    assert loaded.bundle == bundle
    assert loaded.manifest == report.manifest

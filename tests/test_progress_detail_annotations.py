from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from collection_runtime.progress.detail_stage_annotation_store import (
    commit_detail_stage_annotation_set,
    load_detail_stage_annotation_set,
)
from dataset_contracts.progress_detail_annotations import (
    DetailStageAnnotationBundleV1,
    DetailStageAnnotationContractError,
    DetailStageAnnotationManifestV1,
    DetailStageAnnotationSetBindingV1,
    validate_detail_stage_profile_compatibility,
)
from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.progress_profile import (
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stage_annotations import (
    StageAnnotationBundleV1,
    StageAnnotationManifestV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageDefinitionV1,
    TaskStageProfileV1,
)
from dataset_interface.progress_detail_annotation_artifacts import (
    LoadedDetailStageAnnotationSet,
)
from dataset_interface.progress_detail_annotations import (
    resolve_detail_stage_annotation_view,
)
from dataset_interface.progress_stage_annotation_artifacts import (
    LoadedStageAnnotationSet,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _profiles() -> tuple[TaskStageProfileV1, TaskStageProfileV1]:
    progress = TaskProgressProfileV1(
        profile_id="detail-fixture-progress",
        profile_version="1",
        task_id="detail-fixture",
        steps=(
            ProgressStepDefinitionV1(
                "prepare",
                "PREPARE",
                "Prepare",
                1,
                allowed_next_step_ids=("write",),
            ),
            ProgressStepDefinitionV1(
                "write",
                "WRITE",
                "Write",
                2,
            ),
        ),
        initial_step_id="prepare",
        terminal_step_ids=("write",),
    )
    base = TaskStageProfileV1(
        profile_id="detail-fixture-stages",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        stages=(
            ProgressStageDefinitionV1(
                "prepare",
                "prepare.done",
                "Prepare",
                "Prepare the object.",
                1,
            ),
            ProgressStageDefinitionV1(
                "write",
                "write.coarse",
                "Write",
                "Complete the writing step.",
                1,
            ),
        ),
    )
    detail = TaskStageProfileV1(
        profile_id="detail-fixture-stages",
        profile_version="2",
        task_progress_profile_hash=progress.semantic_content_hash,
        stages=(
            ProgressStageDefinitionV1(
                "prepare",
                "prepare.done",
                "Prepare",
                "Prepare the object.",
                1,
            ),
            ProgressStageDefinitionV1(
                "write",
                "write.start",
                "Start",
                "Start writing.",
                1,
            ),
            ProgressStageDefinitionV1(
                "write",
                "write.body",
                "Write",
                "Write the content.",
                2,
            ),
            ProgressStageDefinitionV1(
                "write",
                "write.finish",
                "Finish",
                "Finish writing.",
                3,
            ),
        ),
    )
    return base, detail


def _source_manifest(
    *,
    task_progress_profile_hash: str = "2" * 64,
) -> ProgressEpisodeManifestV1:
    return ProgressEpisodeManifestV1(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=task_progress_profile_hash,
        progress_episode_binding_hash="3" * 64,
        fact_count=1,
        fact_bundle_hash="4" * 64,
        terminal_fact_id="terminal",
        sealed_at=20.0,
        sealed_clock_domain_id="episode_clock",
    )


def _base_set(
    *,
    base_profile: TaskStageProfileV1,
) -> LoadedStageAnnotationSet:
    source = _source_manifest(
        task_progress_profile_hash=base_profile.task_progress_profile_hash,
    )
    binding = StageAnnotationSetBindingV1(
        annotation_set_id="coarse-1",
        episode_capture_identity_hash=source.episode_capture_identity_hash,
        source_progress_manifest_hash=source.semantic_content_hash,
        task_stage_profile_hash=base_profile.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        annotator_id="coarse-annotator",
        created_at=21.0,
        created_clock_domain_id="wall_clock",
    )
    bundle = StageAnnotationBundleV1(
        occurrences=(
            StageOccurrenceAnnotationV1(
                annotation_id="base:prepare",
                coarse_attempt_id="attempt:1",
                progress_step_id="prepare",
                stage_id="prepare.done",
                occurrence_index_in_attempt=1,
                start_at=0.0,
                end_at=1.0,
                clock_domain_id="episode_clock",
                outcome="SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                annotation_id="base:write",
                coarse_attempt_id="attempt:1",
                progress_step_id="write",
                stage_id="write.coarse",
                occurrence_index_in_attempt=1,
                start_at=1.0,
                end_at=10.0,
                clock_domain_id="episode_clock",
                outcome="SUCCESS",
            ),
        )
    )
    manifest = StageAnnotationManifestV1(
        annotation_set_binding_hash=binding.semantic_content_hash,
        annotation_bundle_hash=bundle.semantic_content_hash,
    )
    return LoadedStageAnnotationSet(
        annotation_dir=Path("coarse"),
        binding=binding,
        bundle=bundle,
        manifest=manifest,
    )


def _detail_set(
    *,
    base: LoadedStageAnnotationSet,
    base_profile: TaskStageProfileV1,
    detail_profile: TaskStageProfileV1,
) -> LoadedDetailStageAnnotationSet:
    source = _source_manifest(
        task_progress_profile_hash=base_profile.task_progress_profile_hash,
    )
    binding = DetailStageAnnotationSetBindingV1(
        annotation_set_id="detail-1",
        episode_capture_identity_hash=source.episode_capture_identity_hash,
        source_progress_manifest_hash=source.semantic_content_hash,
        base_annotation_set_id=base.binding.annotation_set_id,
        base_annotation_set_manifest_hash=base.manifest.semantic_content_hash,
        base_task_stage_profile_hash=base_profile.semantic_content_hash,
        detail_task_stage_profile_hash=detail_profile.semantic_content_hash,
        target_progress_step_id="write",
        bound_clock_domain_id="episode_clock",
        annotator_id="detail-annotator",
        created_at=22.0,
        created_clock_domain_id="wall_clock",
    )
    bundle = DetailStageAnnotationBundleV1(
        occurrences=(
            StageOccurrenceAnnotationV1(
                annotation_id="detail:start",
                coarse_attempt_id="attempt:1",
                progress_step_id="write",
                stage_id="write.start",
                occurrence_index_in_attempt=1,
                start_at=1.0,
                end_at=2.0,
                clock_domain_id="episode_clock",
                outcome="SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                annotation_id="detail:body",
                coarse_attempt_id="attempt:1",
                progress_step_id="write",
                stage_id="write.body",
                occurrence_index_in_attempt=2,
                start_at=2.0,
                end_at=8.0,
                clock_domain_id="episode_clock",
                outcome="SUCCESS",
            ),
            StageOccurrenceAnnotationV1(
                annotation_id="detail:finish",
                coarse_attempt_id="attempt:1",
                progress_step_id="write",
                stage_id="write.finish",
                occurrence_index_in_attempt=3,
                start_at=8.0,
                end_at=10.0,
                clock_domain_id="episode_clock",
                outcome="SUCCESS",
            ),
        )
    )
    manifest = DetailStageAnnotationManifestV1(
        annotation_set_binding_hash=binding.semantic_content_hash,
        annotation_bundle_hash=bundle.semantic_content_hash,
    )
    return LoadedDetailStageAnnotationSet(
        annotation_dir=Path("detail"),
        binding=binding,
        bundle=bundle,
        manifest=manifest,
    )


def test_missing_detail_overlay_is_explicit_coarse_fallback() -> None:
    base_profile, detail_profile = _profiles()
    base = _base_set(base_profile=base_profile)

    resolved = resolve_detail_stage_annotation_view(
        source_progress_manifest=_source_manifest(
            task_progress_profile_hash=base_profile.task_progress_profile_hash,
        ),
        base_annotation_set=base,
        base_stage_profile=base_profile,
        detail_stage_profile=detail_profile,
    )

    assert resolved.resolution_level == "COARSE_FALLBACK"
    assert resolved.detail_annotation_set is None
    assert [item.stage_id for item in resolved.bundle.occurrences] == [
        "prepare.done",
        "write.coarse",
    ]


def test_detail_overlay_replaces_only_target_coarse_occurrences() -> None:
    base_profile, detail_profile = _profiles()
    base = _base_set(base_profile=base_profile)
    detail = _detail_set(
        base=base,
        base_profile=base_profile,
        detail_profile=detail_profile,
    )

    resolved = resolve_detail_stage_annotation_view(
        source_progress_manifest=_source_manifest(
            task_progress_profile_hash=base_profile.task_progress_profile_hash,
        ),
        base_annotation_set=base,
        base_stage_profile=base_profile,
        detail_stage_profile=detail_profile,
        detail_annotation_set=detail,
    )

    assert resolved.resolution_level == "DETAIL_COMPLETE"
    assert [item.stage_id for item in resolved.bundle.occurrences] == [
        "prepare.done",
        "write.start",
        "write.body",
        "write.finish",
    ]
    assert resolved.bundle.occurrences[0] == base.bundle.occurrences[0]
    assert all(
        item.annotation_id != "base:write"
        for item in resolved.bundle.occurrences
    )


def test_detail_profile_must_not_change_non_target_steps() -> None:
    base_profile, detail_profile = _profiles()
    changed = replace(
        detail_profile,
        stages=(
            ProgressStageDefinitionV1(
                "prepare",
                "prepare.changed",
                "Changed",
                "Changed.",
                1,
            ),
            *detail_profile.stages_for_step("write"),
        ),
    )

    with pytest.raises(
        DetailStageAnnotationContractError,
        match="non-target",
    ):
        validate_detail_stage_profile_compatibility(
            base_stage_profile=base_profile,
            detail_stage_profile=changed,
            target_progress_step_id="write",
        )


def test_detail_store_roundtrip_preserves_immutable_overlay(tmp_path: Path) -> None:
    base_profile, detail_profile = _profiles()
    base = _base_set(base_profile=base_profile)
    detail = _detail_set(
        base=base,
        base_profile=base_profile,
        detail_profile=detail_profile,
    )
    source = _source_manifest(
        task_progress_profile_hash=base_profile.task_progress_profile_hash,
    )

    report = commit_detail_stage_annotation_set(
        dataset_root=tmp_path,
        episode_uuid="episode-1",
        source_progress_manifest=source,
        base_annotation_set=base,
        base_stage_profile=base_profile,
        detail_stage_profile=detail_profile,
        binding=detail.binding,
        bundle=detail.bundle,
    )
    loaded = load_detail_stage_annotation_set(report.annotation_dir)

    assert loaded.binding == detail.binding
    assert loaded.bundle == detail.bundle
    assert loaded.manifest == report.manifest
    assert report.annotation_dir == (
        tmp_path / "meta/progress/detail_stage_annotations/episode-1/detail-1"
    )

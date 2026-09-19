from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

import collection_runtime.progress.retrospective_store as retrospective_store
from collection_runtime.progress.profiles import load_managed_task_profile
from collection_runtime.progress.retrospective_store import (
    RetrospectiveProgressStoreError,
    commit_retrospective_progress_episode,
    publish_retrospective_campaign_definition,
    resolve_retrospective_migration_identity,
    seal_retrospective_campaign_publication,
)
from collection_runtime.progress.stage_annotation_store import (
    commit_stage_annotation_set,
)
from control_station.refinement_annotation.application import (
    RefinementAnnotationApplication,
)
from control_station.refinement_annotation.cli import (
    main as refinement_annotation_main,
)
from dataset_contracts.progress_profile import ProgressEpisodeBindingV1
from dataset_contracts.progress_stage_annotations import (
    StageAnnotationBundleV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
)
from dataset_contracts.retrospective_progress import (
    RetrospectiveProgressContractError,
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveProgressWindowV1,
    RetrospectiveSourceBindingV1,
    RetrospectiveVideoBindingV1,
)
from dataset_contracts.retrospective_campaign import (
    RetrospectiveAnnotationCampaignV1,
    RetrospectiveCampaignEpisodeSealV1,
    RetrospectiveCampaignPublicationReceiptV1,
)
from dataset_interface.progress_skill_projection import (
    load_builtin_writing_stage1_skill_projection_catalog,
    validate_progress_skill_projection,
)
from dataset_interface.progress_annotation_jobs import (
    ProgressAnnotationJobError,
    list_progress_annotation_work_items,
    progress_annotation_catalog_fingerprint,
    progress_annotation_catalog_probe_fingerprint,
)
from dataset_interface.retrospective_progress import (
    RetrospectiveProgressResolutionError,
    resolve_retrospective_progress_window,
    retrospective_sample_axis_hash,
)
from dataset_interface.progress_stages import resolve_stage_training_index
from dataset_interface.episode_curation import (
    source_dataset_base_fingerprint,
    source_episode_manifest_hash,
)
from dataset_interface.retrospective_progress_artifacts import (
    RetrospectiveProgressArtifactReadError,
    load_retrospective_progress_episode,
)
from dataset_interface.sample_axis import EpisodeSampleAxis
from dataset_interface.writing_stage1_v23 import (
    WritingStage1V23LayerError,
    writing_stage1_v23_semantic_layer_root,
)


_IDENTITY_HASH = "a" * 64
_DATASET_HASH = "b" * 64
_INFO_HASH = "c" * 64
_RECORD_HASH = "d" * 64
_PARQUET_HASH = "e" * 64
_VIDEO_HASH = "f" * 64


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifacts():
    bundle = load_managed_task_profile(
        "dmtac_bimanual_writing_acquire_present_window_v1"
    )
    assert bundle.stage_profile is not None
    axis = EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(0.0, 0.1, 0.2, 0.3, 0.4),
    )
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        task_progress_profile_hash=(
            bundle.progress_profile.semantic_content_hash
        ),
        bound_clock_domain_id=axis.clock_domain_id,
        recording_scope="PROGRESS_STEP_WINDOW",
        entry_step_id="right_acquire_brush",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
        truncated_at_start=False,
    )
    source = RetrospectiveSourceBindingV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_dataset_base_fingerprint=_DATASET_HASH,
        source_dataset_locator="dataset://writing-2026-07-23",
        source_schema="dexteleop_dataset_v2.3",
        source_batch_id="writing-2026-07-23",
        source_episode_index=17,
        provenance_source_episode_index=18,
        source_info_hash=_INFO_HASH,
        source_episode_record_hash=_RECORD_HASH,
        source_parquet_relative_path="data/chunk-000/episode_000017.parquet",
        source_parquet_hash=_PARQUET_HASH,
        source_parquet_byte_count=1024,
        admitted_videos=(
            RetrospectiveVideoBindingV1(
                modality_id="front_rgb",
                source_relative_path=(
                    "videos/chunk-000/observation.images.front/"
                    "episode_000017.mp4"
                ),
                content_hash=_VIDEO_HASH,
                byte_count=2048,
                declared_frame_count=axis.sample_count,
                decoded_frame_count=axis.sample_count,
            ),
        ),
    )
    window = RetrospectiveProgressWindowV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_binding_hash=source.semantic_content_hash,
        task_progress_profile_hash=(
            bundle.progress_profile.semantic_content_hash
        ),
        progress_episode_binding_hash=binding.semantic_content_hash,
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        sample_count=axis.sample_count,
        clock_domain_id=axis.clock_domain_id,
        progress_step_id="right_acquire_brush",
        attempt_id="retrospective:stage1:attempt:1",
        start_at=axis.cut_times[0],
        end_at=axis.cut_times[-1],
    )
    manifest = RetrospectiveProgressMigrationManifestV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_binding_hash=source.semantic_content_hash,
        retrospective_window_hash=window.semantic_content_hash,
        task_progress_profile_hash=(
            bundle.progress_profile.semantic_content_hash
        ),
        progress_episode_binding_hash=binding.semantic_content_hash,
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        migration_actor_id="writing-stage1-migration",
        migration_created_at=123.0,
        migration_clock_domain_id="annotation_wall_clock",
    )
    return bundle, axis, binding, source, window, manifest


def test_retrospective_contracts_round_trip_and_resolve_without_facts() -> None:
    bundle, axis, binding, source, window, manifest = _artifacts()

    source_round_trip = RetrospectiveSourceBindingV1.from_payload(
        source.to_payload()
    )
    window_round_trip = RetrospectiveProgressWindowV1.from_payload(
        window.to_payload()
    )
    manifest_round_trip = (
        RetrospectiveProgressMigrationManifestV1.from_payload(
            manifest.to_payload()
        )
    )
    assert source_round_trip == source
    assert window_round_trip == window
    assert manifest_round_trip == manifest

    resolved = resolve_retrospective_progress_window(
        progress_profile=bundle.progress_profile,
        progress_binding=binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source,
        window=window,
        manifest=manifest,
    )

    assert resolved.resolution_kind == "RETROSPECTIVE_MIGRATED"
    assert resolved.termination_cause == "WINDOW_COMPLETED"
    assert resolved.full_task_outcome == "NOT_APPLICABLE"
    assert resolved.transitions == ()
    assert resolved.capture_pause_intervals == ()
    assert len(resolved.attempts) == 1
    assert resolved.attempts[0].outcome == "SUCCESS"
    assert resolved.attempts[0].sample_interval.sample_index_begin == 0
    assert (
        resolved.attempts[0].sample_interval.sample_index_end_exclusive
        == axis.sample_count
    )


def test_retrospective_resolution_fails_closed_on_sample_axis_drift() -> None:
    bundle, axis, binding, source, window, manifest = _artifacts()
    drifted_axis = EpisodeSampleAxis(
        clock_domain_id=axis.clock_domain_id,
        cut_times=(0.0, 0.1, 0.21),
    )

    with pytest.raises(
        RetrospectiveProgressResolutionError,
        match="sample axis mismatch",
    ):
        resolve_retrospective_progress_window(
            progress_profile=bundle.progress_profile,
            progress_binding=binding,
            stage_profile=bundle.stage_profile,
            sample_axis=drifted_axis,
            source_binding=source,
            window=window,
            manifest=manifest,
        )


def test_retrospective_video_binding_rejects_frame_count_mismatch() -> None:
    with pytest.raises(
        RetrospectiveProgressContractError,
        match="equal declared and decoded",
    ):
        RetrospectiveVideoBindingV1(
            modality_id="front_rgb",
            source_relative_path="videos/front/episode.mp4",
            content_hash=_VIDEO_HASH,
            byte_count=2048,
            declared_frame_count=3,
            decoded_frame_count=2,
        )


def test_retrospective_window_cannot_claim_capture_causal_resolution() -> None:
    _, _, _, _, window, _ = _artifacts()
    payload = window.to_payload()
    payload["resolution_kind"] = "CAPTURE_CAUSAL"

    with pytest.raises(
        RetrospectiveProgressContractError,
        match="visibly migrated",
    ):
        RetrospectiveProgressWindowV1.from_payload(payload)


def test_writing_stage1_skill_is_dual_arm_and_shared_by_window_4_5() -> None:
    catalog = load_builtin_writing_stage1_skill_projection_catalog()
    skill = catalog.skill_by_id[
        "writing.bimanual_acquire_brush_and_present_for_cap_removal"
    ]
    assert skill.skill_version == "2"
    assert skill.controlled_component_ids == (
        "left_arm",
        "right_arm",
        "right_hand",
    )
    assert "left_hand" not in skill.controlled_component_ids

    profile_ids = (
        "dmtac_bimanual_writing_acquire_present_window_v1",
        "dmtac_rm65_bimanual_position_writing_four_stage_progress",
        "dmtac_rm65_bimanual_position_writing_five_stage_progress",
        "dmtac_rm65_bimanual_position_writing_five_stage_progress_detailed",
    )
    bundles = tuple(load_managed_task_profile(value) for value in profile_ids)
    profiles_by_hash = {
        (
            bundle.progress_profile.semantic_content_hash,
            bundle.stage_profile.semantic_content_hash
            if bundle.stage_profile is not None
            else None,
        ): bundle
        for bundle in bundles
    }
    assert len(catalog.projections) == 4
    for projection in catalog.projections:
        bundle = profiles_by_hash[
            (
                projection.source_progress_profile_hash,
                projection.source_stage_profile_hash,
            )
        ]
        assert bundle.stage_profile is not None
        validate_progress_skill_projection(
            progress_profile=bundle.progress_profile,
            stage_profile=bundle.stage_profile,
            skill_contract=skill,
            projection=projection,
        )


def test_window_profile_declares_both_arms_teleoperated() -> None:
    bundle, *_ = _artifacts()
    step = bundle.progress_profile.step_by_id["right_acquire_brush"]
    assert step.arm_control is not None
    assert step.arm_control.left == "teleop"
    assert step.arm_control.right == "teleop"
    assert tuple(
        item.stage_id
        for item in bundle.stage_profile.stages_for_step(
            "right_acquire_brush"
        )
    ) == (
        "right_acquire_brush.approach",
        "right_acquire_brush.grasp",
        "right_acquire_brush.lift_and_present",
    )


def test_three_stage_annotations_cover_migrated_window_without_capture_facts(
) -> None:
    bundle, axis, binding, source, window, manifest = _artifacts()
    resolved = resolve_retrospective_progress_window(
        progress_profile=bundle.progress_profile,
        progress_binding=binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source,
        window=window,
        manifest=manifest,
    )
    annotations = StageAnnotationBundleV1(
        occurrences=(
            StageOccurrenceAnnotationV1(
                annotation_id="stage:approach",
                coarse_attempt_id=window.attempt_id,
                progress_step_id="right_acquire_brush",
                stage_id="right_acquire_brush.approach",
                occurrence_index_in_attempt=1,
                start_at=0.0,
                end_at=0.1,
                clock_domain_id=axis.clock_domain_id,
                outcome="SUCCESS",
                confidence=1.0,
            ),
            StageOccurrenceAnnotationV1(
                annotation_id="stage:grasp",
                coarse_attempt_id=window.attempt_id,
                progress_step_id="right_acquire_brush",
                stage_id="right_acquire_brush.grasp",
                occurrence_index_in_attempt=2,
                start_at=0.1,
                end_at=0.2,
                clock_domain_id=axis.clock_domain_id,
                outcome="SUCCESS",
                confidence=1.0,
            ),
            StageOccurrenceAnnotationV1(
                annotation_id="stage:lift-and-present",
                coarse_attempt_id=window.attempt_id,
                progress_step_id="right_acquire_brush",
                stage_id="right_acquire_brush.lift_and_present",
                occurrence_index_in_attempt=3,
                start_at=0.2,
                end_at=0.4,
                clock_domain_id=axis.clock_domain_id,
                outcome="SUCCESS",
                confidence=1.0,
            ),
        )
    )

    stage_index = resolve_stage_training_index(
        progress_profile=bundle.progress_profile,
        progress_binding=binding,
        stage_profile=bundle.stage_profile,
        stage_entry_binding=None,
        timeline=resolved,
        annotations=annotations,
        sample_axis=axis,
    )

    assert stage_index.coverage_status == "COMPLETE"
    assert stage_index.annotated_sample_count == axis.sample_count
    assert stage_index.unannotated_sample_count == 0
    assert tuple(
        (
            item.stage_id,
            item.sample_interval.sample_index_begin,
            item.sample_interval.sample_index_end_exclusive,
        )
        for item in stage_index.occurrences
    ) == (
        ("right_acquire_brush.approach", 0, 1),
        ("right_acquire_brush.grasp", 1, 2),
        ("right_acquire_brush.lift_and_present", 2, 5),
    )


def _workspace_artifacts(tmp_path: Path):
    source_root = tmp_path / "source"
    (source_root / "meta").mkdir(parents=True)
    (source_root / "data/chunk-000").mkdir(parents=True)
    (source_root / "videos/front/chunk-000").mkdir(parents=True)
    info_path = source_root / "meta/info.json"
    info_path.write_text(
        json.dumps(
            {
                "codebase_version": "v2.3",
                "dexteleop_dataset_schema": "dexteleop_dataset_v2.3",
                "episode_count": 1,
                "frame_count": 5,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    parquet_relative = "data/chunk-000/episode_000000.parquet"
    parquet_path = source_root / parquet_relative
    pq.write_table(
        pa.table({"timestamp": pa.array((0.0, 0.1, 0.2, 0.3, 0.4))}),
        parquet_path,
    )
    video_relative = "videos/front/chunk-000/episode_000000.mp4"
    video_path = source_root / video_relative
    video_path.write_bytes(b"prevalidated-video-fixture")
    record = {
        "episode_id": "episode_000000",
        "episode_index": 0,
        "frame_count": 5,
        "parquet_path": parquet_relative,
        "video_paths": {"front": video_relative},
    }
    (source_root / "meta/episodes.jsonl").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    bundle = load_managed_task_profile(
        "dmtac_bimanual_writing_acquire_present_window_v1"
    )
    assert bundle.stage_profile is not None
    axis = EpisodeSampleAxis(
        clock_domain_id="episode_clock",
        cut_times=(0.0, 0.1, 0.2, 0.3, 0.4),
    )
    progress_binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        task_progress_profile_hash=bundle.progress_profile.semantic_content_hash,
        bound_clock_domain_id=axis.clock_domain_id,
        recording_scope="PROGRESS_STEP_WINDOW",
        entry_step_id="right_acquire_brush",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    source_binding = RetrospectiveSourceBindingV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_dataset_base_fingerprint=source_dataset_base_fingerprint(
            source_root
        ),
        source_dataset_locator=str(source_root),
        source_schema="dexteleop_dataset_v2.3",
        source_batch_id="2026-07-23",
        source_episode_index=0,
        provenance_source_episode_index=1,
        source_info_hash=_file_sha256(info_path),
        source_episode_record_hash=source_episode_manifest_hash(
            source_root, 0
        ),
        source_parquet_relative_path=parquet_relative,
        source_parquet_hash=_file_sha256(parquet_path),
        source_parquet_byte_count=parquet_path.stat().st_size,
        admitted_videos=(
            RetrospectiveVideoBindingV1(
                modality_id="front_rgb",
                source_relative_path=video_relative,
                content_hash=_file_sha256(video_path),
                byte_count=video_path.stat().st_size,
                declared_frame_count=5,
                decoded_frame_count=5,
            ),
        ),
    )
    window = RetrospectiveProgressWindowV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_binding_hash=source_binding.semantic_content_hash,
        task_progress_profile_hash=bundle.progress_profile.semantic_content_hash,
        progress_episode_binding_hash=progress_binding.semantic_content_hash,
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        sample_count=5,
        clock_domain_id=axis.clock_domain_id,
        progress_step_id="right_acquire_brush",
        attempt_id="retrospective:stage1:attempt:1",
        start_at=0.0,
        end_at=0.4,
    )
    manifest = RetrospectiveProgressMigrationManifestV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_binding_hash=source_binding.semantic_content_hash,
        retrospective_window_hash=window.semantic_content_hash,
        task_progress_profile_hash=bundle.progress_profile.semantic_content_hash,
        progress_episode_binding_hash=progress_binding.semantic_content_hash,
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        sample_axis_hash=retrospective_sample_axis_hash(axis),
        migration_actor_id="fixture",
        migration_created_at=1.0,
        migration_clock_domain_id="annotation_clock",
    )
    return (
        source_root,
        video_path,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    )


def test_external_workspace_commit_load_and_stage_annotation_are_compatible(
    tmp_path: Path,
) -> None:
    (
        source_root,
        _,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    workspace = tmp_path / "workspace"
    first = commit_retrospective_progress_episode(
        workspace_root=workspace,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    assert not first.idempotent
    assert (first.episode_dir / "migration_manifest.cjson").is_file()
    second = commit_retrospective_progress_episode(
        workspace_root=workspace,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    assert second.idempotent

    loaded = load_retrospective_progress_episode(
        first.episode_dir,
        progress_profile=bundle.progress_profile,
        stage_profile=bundle.stage_profile,
        allowed_source_roots=(source_root,),
    )
    assert loaded.timeline.resolution_kind == "RETROSPECTIVE_MIGRATED"
    assert loaded.timeline.source_progress_manifest_hash == (
        manifest.semantic_content_hash
    )

    annotation_binding = StageAnnotationSetBindingV1(
        annotation_set_id="first-review",
        episode_capture_identity_hash=_IDENTITY_HASH,
        source_progress_manifest_hash=manifest.semantic_content_hash,
        task_stage_profile_hash=bundle.stage_profile.semantic_content_hash,
        bound_clock_domain_id=axis.clock_domain_id,
        annotator_id="fixture",
        created_at=2.0,
        created_clock_domain_id="annotation_clock",
    )
    report = commit_stage_annotation_set(
        dataset_root=workspace,
        episode_uuid=_IDENTITY_HASH,
        source_progress_manifest=manifest,
        stage_profile=bundle.stage_profile,
        binding=annotation_binding,
        bundle=StageAnnotationBundleV1(),
    )
    assert report.manifest.annotation_set_binding_hash == (
        annotation_binding.semantic_content_hash
    )


def test_external_workspace_reader_rejects_allowlist_and_source_hash_drift(
    tmp_path: Path,
) -> None:
    (
        source_root,
        video_path,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    report = commit_retrospective_progress_episode(
        workspace_root=tmp_path / "workspace",
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )

    with pytest.raises(
        RetrospectiveProgressArtifactReadError,
        match="allowlist",
    ):
        load_retrospective_progress_episode(
            report.episode_dir,
            progress_profile=bundle.progress_profile,
            stage_profile=bundle.stage_profile,
            allowed_source_roots=(tmp_path / "foreign",),
        )

    original = video_path.read_bytes()
    video_path.write_bytes(b"x" * len(original))
    with pytest.raises(
        RetrospectiveProgressArtifactReadError,
        match="hash drift",
    ):
        load_retrospective_progress_episode(
            report.episode_dir,
            progress_profile=bundle.progress_profile,
            stage_profile=bundle.stage_profile,
            allowed_source_roots=(source_root,),
        )


def test_refinement_annotation_gui_adapter_opens_and_seals_retrospective_episode(
    tmp_path: Path,
) -> None:
    (
        source_root,
        _,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    workspace = tmp_path / "workspace"
    commit_retrospective_progress_episode(
        workspace_root=workspace,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )

    with pytest.raises(ProgressAnnotationJobError, match="source root"):
        list_progress_annotation_work_items(workspace)

    items = list_progress_annotation_work_items(
        workspace,
        allowed_source_roots=(source_root,),
    )
    assert len(items) == 1
    assert items[0].summary_payload()["resolution_kind"] == (
        "RETROSPECTIVE_MIGRATED"
    )
    assert items[0].summary_payload()["source_batch_id"] == "2026-07-23"
    assert tuple(items[0].video_paths) == ("front",)

    application = RefinementAnnotationApplication(
        workspace,
        annotator_id="fixture-annotator",
        allowed_source_roots=(source_root,),
    )
    try:
        detail = application.episode_detail(_IDENTITY_HASH)
        assert detail["resolution_kind"] == "RETROSPECTIVE_MIGRATED"
        assert detail["recording_scope"] == "PROGRESS_STEP_WINDOW"
        assert detail["full_task_outcome"] == "NOT_APPLICABLE"
        assert detail["annotation_status"]["loaded"] is True
        assert detail["annotation_status"]["schema"] == (
            "dexteleop.refinement_annotation.episode_status.v1"
        )
        assert tuple(
            stage["stage_id"]
            for stage in detail["progress_steps"][0]["stages"]
        ) == (
            "right_acquire_brush.approach",
            "right_acquire_brush.grasp",
            "right_acquire_brush.lift_and_present",
        )

        draft = detail["draft"]
        draft["occurrences"][0]["end_at"] = 0.1
        draft["occurrences"][1]["start_at"] = 0.1
        draft["occurrences"][1]["end_at"] = 0.2
        draft["occurrences"][2]["start_at"] = 0.2
        report = application.commit(_IDENTITY_HASH, draft)
        assert report["coverage_status"] == "COMPLETE"
        assert report["annotated_sample_count"] == 5
        assert report["unannotated_sample_count"] == 0
        assert (
            workspace
            / "meta"
            / "progress"
            / "stage_annotations"
            / _IDENTITY_HASH
            / report["annotation_set_id"]
            / "manifest.cjson"
        ).is_file()
    finally:
        application.close()


def test_retrospective_catalog_probe_does_not_enumerate_episode_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    episodes = workspace / "episodes"
    campaigns = workspace / "campaigns"
    episodes.mkdir(parents=True)
    campaigns.mkdir()
    (workspace / "profile.cjson").write_text("{}", encoding="utf-8")
    (workspace / "stages.cjson").write_text("{}", encoding="utf-8")
    (episodes / "episode-a").mkdir()
    original_iterdir = Path.iterdir

    def rejected_enumeration(path: Path):
        if path in {episodes, campaigns}:
            raise AssertionError("cheap catalog probe enumerated children")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", rejected_enumeration)

    probe = progress_annotation_catalog_probe_fingerprint(workspace)

    assert probe[0] == "retrospective_probe"


def test_pilot_publisher_automatically_creates_and_recovers_migration_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty_workspace = tmp_path / "new-workspace"
    monkeypatch.setattr(retrospective_store.time, "time", lambda: 42.5)
    created = resolve_retrospective_migration_identity(
        empty_workspace,
        requested_actor_id="migration-operator",
        requested_created_at=None,
    )
    assert (created.actor_id, created.created_at, created.recovered) == (
        "migration-operator",
        42.5,
        False,
    )

    (
        _,
        _,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    workspace = tmp_path / "sealed-workspace"
    commit_retrospective_progress_episode(
        workspace_root=workspace,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    recovered = resolve_retrospective_migration_identity(
        workspace,
        requested_actor_id="",
        requested_created_at=None,
    )
    assert (recovered.actor_id, recovered.created_at, recovered.recovered) == (
        "fixture",
        1.0,
        True,
    )

    with pytest.raises(RetrospectiveProgressStoreError, match="actor conflicts"):
        resolve_retrospective_migration_identity(
            workspace,
            requested_actor_id="different-operator",
            requested_created_at=None,
        )
    with pytest.raises(
        RetrospectiveProgressStoreError,
        match="timestamp conflicts",
    ):
        resolve_retrospective_migration_identity(
            workspace,
            requested_actor_id="fixture",
            requested_created_at=2.0,
        )


def test_writing_v23_dataset_local_layer_is_additive_and_source_bound(
    tmp_path: Path,
) -> None:
    (
        source_root,
        _,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    source_artifacts_before = {
        path.relative_to(source_root): (
            path.stat().st_ino,
            path.stat().st_size,
            _file_sha256(path),
        )
        for path in source_root.rglob("*")
        if path.is_file()
    }
    layer_root = writing_stage1_v23_semantic_layer_root(
        source_root,
        require_exists=False,
    )
    source_binding = replace(source_binding, source_dataset_locator=".")
    window = replace(
        window,
        source_binding_hash=source_binding.semantic_content_hash,
    )
    manifest = replace(
        manifest,
        source_binding_hash=source_binding.semantic_content_hash,
        retrospective_window_hash=window.semantic_content_hash,
    )
    commit_retrospective_progress_episode(
        workspace_root=layer_root,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    assert writing_stage1_v23_semantic_layer_root(source_root) == layer_root
    assert len(
        list_progress_annotation_work_items(
            layer_root,
            allowed_source_roots=(source_root,),
        )
    ) == 1
    assert refinement_annotation_main(
        [
            str(source_root),
            "--writing-v23-semantic-layer",
            "writing_stage1_retrospective_v1",
            "--dry-run",
        ]
    ) == 0
    source_artifacts_after = {
        path.relative_to(source_root): (
            path.stat().st_ino,
            path.stat().st_size,
            _file_sha256(path),
        )
        for path in source_root.rglob("*")
        if path.is_file() and "semantic_layers" not in path.parts
    }
    assert source_artifacts_after == source_artifacts_before

    info_path = source_root / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["dexteleop_dataset_schema"] = "dexteleop_dataset_v2.5p2"
    info_path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(WritingStage1V23LayerError, match="only accepts"):
        writing_stage1_v23_semantic_layer_root(
            source_root,
            semantic_layer_id="another-layer",
            require_exists=False,
        )


def test_dataset_local_campaign_definition_and_receipt_are_idempotent(
    tmp_path: Path,
) -> None:
    (
        source_root,
        _,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    layer_root = writing_stage1_v23_semantic_layer_root(
        source_root,
        require_exists=False,
    )
    source_binding = replace(source_binding, source_dataset_locator=".")
    window = replace(
        window,
        source_binding_hash=source_binding.semantic_content_hash,
    )
    manifest = replace(
        manifest,
        source_binding_hash=source_binding.semantic_content_hash,
        retrospective_window_hash=window.semantic_content_hash,
    )
    definition = publish_retrospective_campaign_definition(
        semantic_layer_root=layer_root,
        semantic_layer_id="writing_stage1_retrospective_v1",
        campaign_id="pilot_v1",
        campaign_kind="PILOT_QA",
        source_dataset_base_fingerprint=(
            source_binding.source_dataset_base_fingerprint
        ),
        cohort_plan_hash="9" * 64,
        eligible_episode_identity_hashes=(_IDENTITY_HASH, "8" * 64),
        selected_episode_identity_hashes=(_IDENTITY_HASH,),
        requested_actor_id="fixture",
        requested_created_at=1.0,
    )
    assert not definition.idempotent
    assert not definition.identity_recovered
    assert RetrospectiveAnnotationCampaignV1.from_payload(
        definition.campaign.to_payload()
    ) == definition.campaign

    commit_retrospective_progress_episode(
        workspace_root=layer_root,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    assert list_progress_annotation_work_items(
        layer_root,
        allowed_source_roots=(source_root,),
    ) == ()
    fingerprint_before_receipt = progress_annotation_catalog_fingerprint(
        layer_root
    )
    seal = RetrospectiveCampaignEpisodeSealV1(
        episode_capture_identity_hash=_IDENTITY_HASH,
        migration_manifest_hash=manifest.semantic_content_hash,
    )
    receipt = seal_retrospective_campaign_publication(
        semantic_layer_root=layer_root,
        campaign=definition.campaign,
        episode_seals=(seal,),
    )
    assert not receipt.idempotent
    assert RetrospectiveCampaignPublicationReceiptV1.from_payload(
        receipt.receipt.to_payload()
    ) == receipt.receipt
    assert progress_annotation_catalog_fingerprint(
        layer_root
    ) != fingerprint_before_receipt
    assert len(
        list_progress_annotation_work_items(
            layer_root,
            allowed_source_roots=(source_root,),
        )
    ) == 1

    recovered = publish_retrospective_campaign_definition(
        semantic_layer_root=layer_root,
        semantic_layer_id="writing_stage1_retrospective_v1",
        campaign_id="pilot_v1",
        campaign_kind="PILOT_QA",
        source_dataset_base_fingerprint=(
            source_binding.source_dataset_base_fingerprint
        ),
        cohort_plan_hash="9" * 64,
        eligible_episode_identity_hashes=(_IDENTITY_HASH, "8" * 64),
        selected_episode_identity_hashes=(_IDENTITY_HASH,),
    )
    assert recovered.idempotent
    assert recovered.identity_recovered
    assert seal_retrospective_campaign_publication(
        semantic_layer_root=layer_root,
        campaign=recovered.campaign,
        episode_seals=(seal,),
    ).idempotent

    with pytest.raises(
        RetrospectiveProgressStoreError,
        match="campaign content conflicts",
    ):
        publish_retrospective_campaign_definition(
            semantic_layer_root=layer_root,
            semantic_layer_id="writing_stage1_retrospective_v1",
            campaign_id="pilot_v1",
            campaign_kind="PILOT_QA",
            source_dataset_base_fingerprint=(
                source_binding.source_dataset_base_fingerprint
            ),
            cohort_plan_hash="7" * 64,
            eligible_episode_identity_hashes=(_IDENTITY_HASH, "8" * 64),
            selected_episode_identity_hashes=(_IDENTITY_HASH,),
        )


def test_dataset_local_source_binding_survives_dataset_directory_move(
    tmp_path: Path,
) -> None:
    (
        source_root,
        _,
        bundle,
        axis,
        progress_binding,
        source_binding,
        window,
        manifest,
    ) = _workspace_artifacts(tmp_path)
    source_binding = replace(source_binding, source_dataset_locator=".")
    window = replace(
        window,
        source_binding_hash=source_binding.semantic_content_hash,
    )
    manifest = replace(
        manifest,
        source_binding_hash=source_binding.semantic_content_hash,
        retrospective_window_hash=window.semantic_content_hash,
    )
    layer_root = writing_stage1_v23_semantic_layer_root(
        source_root,
        require_exists=False,
    )
    commit_retrospective_progress_episode(
        workspace_root=layer_root,
        progress_profile=bundle.progress_profile,
        progress_binding=progress_binding,
        stage_profile=bundle.stage_profile,
        sample_axis=axis,
        source_binding=source_binding,
        window=window,
        migration_manifest=manifest,
    )
    definition = publish_retrospective_campaign_definition(
        semantic_layer_root=layer_root,
        semantic_layer_id="writing_stage1_retrospective_v1",
        campaign_id="pilot_v1",
        campaign_kind="PILOT_QA",
        source_dataset_base_fingerprint=(
            source_binding.source_dataset_base_fingerprint
        ),
        cohort_plan_hash="9" * 64,
        eligible_episode_identity_hashes=(_IDENTITY_HASH,),
        selected_episode_identity_hashes=(_IDENTITY_HASH,),
        requested_actor_id="fixture",
        requested_created_at=1.0,
    )
    seal_retrospective_campaign_publication(
        semantic_layer_root=layer_root,
        campaign=definition.campaign,
        episode_seals=(
            RetrospectiveCampaignEpisodeSealV1(
                episode_capture_identity_hash=_IDENTITY_HASH,
                migration_manifest_hash=manifest.semantic_content_hash,
            ),
        ),
    )

    moved_root = tmp_path / "moved-source"
    source_root.rename(moved_root)
    moved_layer = writing_stage1_v23_semantic_layer_root(moved_root)
    items = list_progress_annotation_work_items(
        moved_layer,
        allowed_source_roots=(moved_root,),
    )
    assert len(items) == 1
    assert items[0].parquet_path.is_relative_to(moved_root)
    assert all(path.is_relative_to(moved_root) for path in items[0].video_paths.values())


def test_writing_v23_layer_rejects_symlink_escape(tmp_path: Path) -> None:
    source_root, *_ = _workspace_artifacts(tmp_path)
    outside = tmp_path / "outside-semantic-layers"
    outside.mkdir()
    (source_root / "semantic_layers").symlink_to(
        outside,
        target_is_directory=True,
    )
    with pytest.raises(WritingStage1V23LayerError, match="symbolic link"):
        writing_stage1_v23_semantic_layer_root(
            source_root,
            require_exists=False,
        )

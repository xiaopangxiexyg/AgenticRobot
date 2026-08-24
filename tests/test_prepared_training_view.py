from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from collection_runtime.progress import (
    CompleteEpisodeProgressRequest,
    ProgressController,
    ProgressFactStore,
    ProgressRequestContext,
    StartEpisodeProgressRequest,
    commit_stage_annotation_set,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.progress_stage_annotations import (
    StageAnnotationBundleV1,
    StageAnnotationSetBindingV1,
    StageOccurrenceAnnotationV1,
)
from dataset_contracts.progress_stages import (
    ProgressStageDefinitionV1,
    ProgressStageEntryBindingV1,
    TaskStageProfileV1,
)
from dataset_contracts.progress_skill_projection import (
    ProgressSkillProjectionV1,
    SkillTrainingContractV1,
)
from dataset_interface.progress_skill_projection import SkillProjectionCatalog
from dataset_interface.modality_artifacts import build_modality_artifact_inventory
from dataset_interface.prepared_training_view import (
    PreparedSourceDataset,
    PreparedTrainingViewError,
    build_prepared_training_view,
    load_prepared_training_view,
)
from dataset_interface.prepared_training import (
    select_strict_successful_skill_episodes,
)
from dataset_interface.prepared_training.sources import prepare_episode
from dataset_interface.progress_annotation_jobs import (
    list_progress_annotation_work_items,
)
from dataset_interface.sample_axis import SampleIndexInterval


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _context(request_id: str, at: float) -> ProgressRequestContext:
    return ProgressRequestContext(
        request_id=request_id,
        effective_at=at,
        effective_clock_domain_id="episode-clock",
        available_at=at,
        available_clock_domain_id="episode-clock",
        actor_id="fixture",
    )


def _profiles(*, variant: str = "base") -> tuple[
    TaskProgressProfileV1,
    TaskStageProfileV1,
]:
    progress = TaskProgressProfileV1(
        profile_id=f"prepared-progress-{variant}",
        profile_version="1",
        task_id="prepared-task",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="write",
                semantic_type="WRITE",
                display_name="Write",
                ordinal=1,
            ),
        ),
        initial_step_id="write",
        terminal_step_ids=("write",),
    )
    stages = TaskStageProfileV1(
        profile_id=f"prepared-stages-{variant}",
        profile_version="1",
        task_progress_profile_hash=progress.semantic_content_hash,
        stages=(
            ProgressStageDefinitionV1(
                "write",
                "write-main",
                "Write",
                "Execute the writing motion.",
                1,
            ),
        ),
    )
    return progress, stages


def _dataset(
    root: Path,
    *,
    episode_uuid: str,
    episode_index: int,
    profiles: tuple[TaskProgressProfileV1, TaskStageProfileV1],
    annotate: bool,
    annotation_set_id: str = "set-1",
    continuity_segments: list[int] | None = None,
) -> str:
    progress, stages = profiles
    identity = EpisodeCaptureIdentity(
        episode_uuid=episode_uuid,
        legacy_episode_index=episode_index,
        lifecycle_owner_id="fixture",
        prepared_at=float(episode_index),
        prepared_clock_domain_id="fixture-clock",
    )
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=identity.semantic_content_hash,
        task_progress_profile_hash=progress.semantic_content_hash,
        bound_clock_domain_id="episode-clock",
        recording_scope="FULL_TASK",
        entry_step_id="write",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    stage_binding = ProgressStageEntryBindingV1(
        episode_capture_identity_hash=identity.semantic_content_hash,
        task_stage_profile_hash=stages.semantic_content_hash,
        entry_stage_id="write-main",
    )
    progress_dir = root / "meta" / "progress" / "episodes" / episode_uuid
    store = ProgressFactStore(
        episode_dir=progress_dir,
        identity=identity,
        profile=progress,
        binding=binding,
        stage_profile=stages,
        stage_entry_binding=stage_binding,
    )
    controller = ProgressController(store=store)
    controller.start(
        StartEpisodeProgressRequest("attempt-1"),
        context=_context("start", 1.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context("complete", 4.0),
    )
    report = controller.seal(
        sealed_at=5.0,
        sealed_clock_domain_id="episode-clock",
    )

    parquet_path = root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    columns: dict[str, object] = {
        "timestamp": [1.0, 2.0, 3.0, 4.0],
        "observation.state": [[0.0], [1.0], [2.0], [3.0]],
    }
    if continuity_segments is not None:
        columns["collection.continuity_segment_index"] = continuity_segments
    pq.write_table(pa.table(columns), parquet_path)
    video_path = root / "videos" / "front" / f"episode_{episode_index:06d}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    video_path.write_bytes(f"video:{episode_uuid}".encode())
    catalog_path = root / "meta" / "episodes.jsonl"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(
        json.dumps(
            {
                "episode_index": episode_index,
                "frame_count": 4,
                "start_timestamp": 1.0,
                "end_timestamp": 4.0,
                "parquet_path": parquet_path.relative_to(root).as_posix(),
                "video_paths": {
                    "front": video_path.relative_to(root).as_posix(),
                },
                "metadata": {
                    "progress_episode": {
                        "episode_uuid": episode_uuid,
                        "episode_identity_hash": identity.semantic_content_hash,
                        "manifest_ref": (
                            progress_dir / "manifest.cjson"
                        ).relative_to(root).as_posix(),
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if annotate:
        annotation_binding = StageAnnotationSetBindingV1(
            annotation_set_id=annotation_set_id,
            episode_capture_identity_hash=identity.semantic_content_hash,
            source_progress_manifest_hash=report.manifest.semantic_content_hash,
            task_stage_profile_hash=stages.semantic_content_hash,
            bound_clock_domain_id="episode-clock",
            annotator_id="fixture",
            created_at=6.0,
            created_clock_domain_id="fixture-clock",
        )
        bundle = StageAnnotationBundleV1(
            occurrences=(
                StageOccurrenceAnnotationV1(
                    annotation_id=f"{annotation_set_id}:write-main",
                    coarse_attempt_id="attempt-1",
                    progress_step_id="write",
                    stage_id="write-main",
                    occurrence_index_in_attempt=1,
                    start_at=1.0,
                    end_at=4.0,
                    clock_domain_id="episode-clock",
                    outcome="SUCCESS",
                ),
            )
        )
        commit_stage_annotation_set(
            dataset_root=root,
            episode_uuid=episode_uuid,
            source_progress_manifest=report.manifest,
            stage_profile=stages,
            binding=annotation_binding,
            bundle=bundle,
        )
    return identity.semantic_content_hash


def _snapshot(root: Path) -> dict[str, tuple[int, int]]:
    return {
        path.relative_to(root).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in root.rglob("*")
        if path.is_file()
    }


def test_prepared_episode_training_semantics_split_at_collection_continuity(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    _dataset(
        source_root,
        episode_uuid="episode-continuity",
        episode_index=0,
        profiles=_profiles(),
        annotate=False,
        continuity_segments=[0, 0, 1, 1],
    )
    source = PreparedSourceDataset("continuity-source", source_root)
    item = list_progress_annotation_work_items(source_root)[0]

    prepared = prepare_episode(
        source=source,
        item=item,
        dataset_fingerprint="fixture-fingerprint",
        selector=None,
        column_projection=("timestamp", "observation.state"),
        modality_selection=("numeric",),
    )

    assert prepared.semantic_index.nominal_intervals == (
        SampleIndexInterval(0, 2),
        SampleIndexInterval(2, 4),
    )


def _add_then_detach_tactile(root: Path) -> None:
    manifest_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/manifest.json"
    )
    payload_relative = (
        "tactile/observation.hand.left.tactile/chunk-000/"
        "episode_000000/L_RAW.uint8.bin"
    )
    payload_path = root / payload_relative
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes(b"1234")
    (root / manifest_relative).write_text(
        json.dumps(
            {
                "format": "binary_stream_v1",
                "side": "left",
                "frame_count": 4,
                "sensor_keys": ["L_RAW"],
                "streams": {
                    "L_RAW": {
                        "path": payload_relative,
                        "dtype": "uint8",
                        "shape": [1],
                        "frame_count": 4,
                        "encoding": "raw_c_order",
                        "compression": "none",
                    }
                },
                "frames": [
                    {
                        "sample_frame_index": index,
                        "tactile_frame_index": index,
                    }
                    for index in range(4)
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    catalog_path = root / "meta" / "episodes.jsonl"
    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    record["tactile_paths"] = {"left": manifest_relative}
    catalog_path.write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    build_modality_artifact_inventory(root)
    payload_path.unlink()


def test_multi_source_build_is_read_only_relative_and_idempotent(
    tmp_path: Path,
) -> None:
    profiles = _profiles()
    first = tmp_path / "2026-08-01"
    second = tmp_path / "2026-08-02"
    first_identity = _dataset(
        first,
        episode_uuid="episode-a",
        episode_index=0,
        profiles=profiles,
        annotate=True,
    )
    second_identity = _dataset(
        second,
        episode_uuid="episode-b",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    before = {"first": _snapshot(first), "second": _snapshot(second)}
    organized = tmp_path / "organized"

    result = build_prepared_training_view(
        organized,
        (
            PreparedSourceDataset("date-0801", first),
            PreparedSourceDataset("date-0802", second),
        ),
        column_projection=("timestamp", "observation.state"),
    )
    loaded = load_prepared_training_view(
        result.view_root,
        verify_source_files=True,
    )

    assert result.episode_count == 2
    assert loaded.episodes.num_rows == 2
    assert set(
        loaded.episodes["episode_capture_identity_hash"].to_pylist()
    ) == {first_identity, second_identity}
    assert set(loaded.episodes["annotation_status"].to_pylist()) == {
        "SELECTED",
        "UNANNOTATED",
    }
    assert all(path.is_absolute() for path in loaded.source_datasets.values())
    source_payload = canonical_source_payload(result.view_root)
    assert all(
        not Path(row["source_root_locator"]).is_absolute()
        for row in source_payload["sources"]
    )
    assert before == {"first": _snapshot(first), "second": _snapshot(second)}
    with pytest.raises(PreparedTrainingViewError, match="not self-contained"):
        load_prepared_training_view(
            result.view_root,
            require_self_contained=True,
        )

    repeated = build_prepared_training_view(
        organized,
        (
            PreparedSourceDataset("date-0801", first),
            PreparedSourceDataset("date-0802", second),
        ),
        column_projection=("timestamp", "observation.state"),
    )
    assert repeated.training_view_hash == result.training_view_hash
    assert repeated.idempotent is True


def test_visual_prepared_view_never_requires_detached_tactile_payload(
    tmp_path: Path,
) -> None:
    profiles = _profiles()
    source = tmp_path / "source"
    _dataset(
        source,
        episode_uuid="episode-visual-detached",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    _add_then_detach_tactile(source)

    result = build_prepared_training_view(
        tmp_path / "organized",
        (PreparedSourceDataset("detached-source", source),),
        modality_selection=("numeric", "video"),
        column_projection=("timestamp", "observation.state"),
    )
    loaded = load_prepared_training_view(
        result.view_root,
        verify_source_files=True,
    )

    assert loaded.episodes.num_rows == 1
    assert tuple(loaded.manifest["modality_selection"]) == ("numeric", "video")
    assert all(
        not str(record.get("relative_path", "")).startswith("tactile/")
        for record in loaded.source_inventory
    )


def test_prepared_training_view_excludes_source_training_invalid_episode(
    tmp_path: Path,
) -> None:
    profiles = _profiles()
    invalid_source = tmp_path / "invalid-source"
    valid_source = tmp_path / "valid-source"
    invalid_identity = _dataset(
        invalid_source,
        episode_uuid="episode-invalid-video",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    valid_identity = _dataset(
        valid_source,
        episode_uuid="episode-valid-video",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    (invalid_source / "meta/modality_validity.json").write_text(
        json.dumps(
            {
                "schema_version": "dexteleop.modality_validity.v1",
                "training": {
                    "episodes": {
                        "0": {
                            "valid": False,
                            "reason": "required camera video failed integrity validation",
                        }
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_prepared_training_view(
        tmp_path / "organized",
        (
            PreparedSourceDataset("invalid", invalid_source),
            PreparedSourceDataset("valid", valid_source),
        ),
    )
    loaded = load_prepared_training_view(result.view_root)

    assert result.episode_count == 1
    assert loaded.episodes["episode_capture_identity_hash"].to_pylist() == [
        valid_identity
    ]
    source_payload = canonical_source_payload(result.view_root)
    excluded_by_source = {
        row["source_dataset_id"]: row[
            "training_validity_excluded_episode_indices"
        ]
        for row in source_payload["sources"]
    }
    assert excluded_by_source == {"invalid": [0], "valid": []}

    with pytest.raises(
        PreparedTrainingViewError,
        match="not found or are excluded",
    ):
        build_prepared_training_view(
            tmp_path / "explicit-invalid",
            (PreparedSourceDataset("invalid", invalid_source),),
            include_episode_identities=(invalid_identity,),
        )


def test_prepared_view_camera_selection_is_hash_bound_and_inventory_scoped(
    tmp_path: Path,
) -> None:
    profiles = _profiles()
    source = tmp_path / "source"
    _dataset(
        source,
        episode_uuid="episode-camera-selection",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    wrist = source / "videos/left_wrist/episode_000000.mp4"
    wrist.parent.mkdir(parents=True, exist_ok=True)
    wrist.write_bytes(b"left-wrist")
    catalog_path = source / "meta/episodes.jsonl"
    record = json.loads(catalog_path.read_text(encoding="utf-8"))
    record["video_paths"]["left_wrist"] = wrist.relative_to(source).as_posix()
    catalog_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    front = build_prepared_training_view(
        tmp_path / "front-organized",
        (PreparedSourceDataset("source", source),),
        camera_selection=("front",),
    )
    all_views = build_prepared_training_view(
        tmp_path / "all-organized",
        (PreparedSourceDataset("source", source),),
    )
    loaded = load_prepared_training_view(front.view_root)

    assert loaded.manifest["camera_selection"] == ["front"]
    assert loaded.episodes["camera_keys"].to_pylist() == [["front"]]
    assert {
        record["role"]
        for record in loaded.source_inventory
        if str(record["role"]).startswith("EPISODE_VIDEO:")
    } == {"EPISODE_VIDEO:front"}
    assert front.training_view_hash != all_views.training_view_hash


def test_strict_skill_selection_accepts_episode_uuid_annotation_selector(
    tmp_path: Path,
) -> None:
    profiles = _profiles()
    source = tmp_path / "source"
    identity = _dataset(
        source,
        episode_uuid="episode-selector",
        episode_index=0,
        profiles=profiles,
        annotate=True,
    )
    _dataset_annotation_copy(source, profiles, annotation_set_id="set-2")
    skill = SkillTrainingContractV1(
        skill_id="fixture.write",
        skill_version="1",
        semantic_type="WRITE",
        controlled_component_ids=("right_arm",),
        entry_predicate_ids=("writing.ready",),
        success_predicate_ids=("writing.complete",),
    )
    progress, stages = profiles
    catalog = SkillProjectionCatalog(
        skills=(skill,),
        projections=(
            ProgressSkillProjectionV1(
                projection_id="fixture.write",
                projection_version="1",
                source_progress_profile_hash=progress.semantic_content_hash,
                source_stage_profile_hash=stages.semantic_content_hash,
                source_progress_step_id="write",
                source_stage_ids=("write-main",),
                projection_kind="EXACT_STEP",
                skill_training_contract_hash=skill.semantic_content_hash,
                projected_component_ids=("right_arm",),
                asserted_entry_predicate_ids=("writing.ready",),
                asserted_success_predicate_ids=("writing.complete",),
            ),
        ),
    )

    report = select_strict_successful_skill_episodes(
        (PreparedSourceDataset("source", source),),
        skill_projection_catalog=catalog,
        target_skill_id=skill.skill_id,
        annotation_selection={"episode-selector": "set-2"},
    )

    assert report.included_episode_identities == (identity,)
    assert report.annotation_selection == {identity: "set-2"}
    assert report.exclusions == ()


def test_cross_profile_skill_projection_is_explicit_and_non_destructive(
    tmp_path: Path,
) -> None:
    first_profiles = _profiles(variant="first")
    second_profiles = _profiles(variant="second")
    first = tmp_path / "2026-08-01"
    second = tmp_path / "2026-08-02"
    _dataset(
        first,
        episode_uuid="episode-a",
        episode_index=0,
        profiles=first_profiles,
        annotate=True,
    )
    _dataset(
        second,
        episode_uuid="episode-b",
        episode_index=0,
        profiles=second_profiles,
        annotate=True,
    )
    before = {"first": _snapshot(first), "second": _snapshot(second)}
    skill = SkillTrainingContractV1(
        skill_id="fixture.write",
        skill_version="1",
        semantic_type="WRITE",
        controlled_component_ids=("right_arm",),
        entry_predicate_ids=("writing.ready",),
        success_predicate_ids=("writing.complete",),
    )
    projections = tuple(
        ProgressSkillProjectionV1(
            projection_id=f"fixture.{label}.write",
            projection_version="1",
            source_progress_profile_hash=progress.semantic_content_hash,
            source_stage_profile_hash=stages.semantic_content_hash,
            source_progress_step_id="write",
            source_stage_ids=("write-main",),
            projection_kind="EXACT_STEP",
            skill_training_contract_hash=skill.semantic_content_hash,
            projected_component_ids=("right_arm",),
            asserted_entry_predicate_ids=("writing.ready",),
            asserted_success_predicate_ids=("writing.complete",),
        )
        for label, (progress, stages) in (
            ("first", first_profiles),
            ("second", second_profiles),
        )
    )
    catalog = SkillProjectionCatalog(
        skills=(skill,),
        projections=projections,
    )
    result = build_prepared_training_view(
        tmp_path / "organized",
        (
            PreparedSourceDataset("date-0801", first),
            PreparedSourceDataset("date-0802", second),
        ),
        skill_projection_catalog=catalog,
        target_skill_id=skill.skill_id,
    )
    loaded = load_prepared_training_view(
        result.view_root,
        verify_source_files=True,
    )
    assert result.skill_interval_count == 2
    assert loaded.skill_intervals is not None
    assert loaded.skill_intervals.num_rows == 2
    assert set(loaded.skill_intervals["skill_id"].to_pylist()) == {
        "fixture.write"
    }
    assert len(loaded.skill_projection_receipts) == 2
    assert loaded.manifest["progress_profile_hash"] is None
    assert len(loaded.manifest["progress_profile_hashes"]) == 2
    assert before == {"first": _snapshot(first), "second": _snapshot(second)}


def canonical_source_payload(view_root: Path) -> dict[str, object]:
    from dataset_contracts.canonical_encoding import canonical_loads
    from dataset_interface.prepared_training_view import (
        PREPARED_SOURCE_DATASETS_DOMAIN_V1,
    )

    value = canonical_loads(
        (view_root / "source_datasets.cjson").read_bytes(),
        expected_domain=PREPARED_SOURCE_DATASETS_DOMAIN_V1,
    )
    assert isinstance(value, dict)
    return value


def test_multiple_annotation_sets_require_explicit_selection(tmp_path: Path) -> None:
    profiles = _profiles()
    source = tmp_path / "source"
    identity = _dataset(
        source,
        episode_uuid="episode-a",
        episode_index=0,
        profiles=profiles,
        annotate=True,
    )
    # Publish a second, valid immutable interpretation of the same Episode.
    _dataset_annotation_copy(source, profiles, annotation_set_id="set-2")

    with pytest.raises(PreparedTrainingViewError, match="multiple annotation"):
        build_prepared_training_view(
            tmp_path / "ambiguous",
            (PreparedSourceDataset("source", source),),
        )

    result = build_prepared_training_view(
        tmp_path / "selected",
        (PreparedSourceDataset("source", source),),
        annotation_selection={identity: "set-2"},
    )
    loaded = load_prepared_training_view(result.view_root)
    assert loaded.episodes["annotation_set_id"].to_pylist() == ["set-2"]


def _dataset_annotation_copy(
    root: Path,
    profiles: tuple[TaskProgressProfileV1, TaskStageProfileV1],
    *,
    annotation_set_id: str,
) -> None:
    from dataset_interface.progress_annotation_jobs import (
        list_progress_annotation_work_items,
    )

    item = list_progress_annotation_work_items(root)[0]
    stages = profiles[1]
    binding = StageAnnotationSetBindingV1(
        annotation_set_id=annotation_set_id,
        episode_capture_identity_hash=item.episode_identity_hash,
        source_progress_manifest_hash=item.progress_manifest_hash,
        task_stage_profile_hash=stages.semantic_content_hash,
        bound_clock_domain_id="episode-clock",
        annotator_id="fixture-2",
        created_at=7.0,
        created_clock_domain_id="fixture-clock",
    )
    bundle = StageAnnotationBundleV1(
        occurrences=(
            StageOccurrenceAnnotationV1(
                annotation_id=f"{annotation_set_id}:write-main",
                coarse_attempt_id="attempt-1",
                progress_step_id="write",
                stage_id="write-main",
                occurrence_index_in_attempt=1,
                start_at=1.0,
                end_at=4.0,
                clock_domain_id="episode-clock",
                outcome="SUCCESS",
            ),
        )
    )
    commit_stage_annotation_set(
        dataset_root=root,
        episode_uuid=item.episode_uuid,
        source_progress_manifest=item.progress.manifest,
        stage_profile=stages,
        binding=binding,
        bundle=bundle,
    )


def test_profile_conflict_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _dataset(
        first,
        episode_uuid="episode-a",
        episode_index=0,
        profiles=_profiles(variant="first"),
        annotate=False,
    )
    _dataset(
        second,
        episode_uuid="episode-b",
        episode_index=0,
        profiles=_profiles(variant="second"),
        annotate=False,
    )

    with pytest.raises(PreparedTrainingViewError, match="exactly one progress"):
        build_prepared_training_view(
            tmp_path / "organized",
            (
                PreparedSourceDataset("first", first),
                PreparedSourceDataset("second", second),
            ),
        )


def test_duplicate_episode_identity_deduplicates_only_exact_content(
    tmp_path: Path,
) -> None:
    profiles = _profiles()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_identity = _dataset(
        first,
        episode_uuid="same-episode",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    second_identity = _dataset(
        second,
        episode_uuid="same-episode",
        episode_index=0,
        profiles=profiles,
        annotate=False,
    )
    assert first_identity == second_identity

    result = build_prepared_training_view(
        tmp_path / "deduplicated",
        (
            PreparedSourceDataset("first", first),
            PreparedSourceDataset("second", second),
        ),
    )
    assert result.episode_count == 1

    (second / "videos" / "front" / "episode_000000.mp4").write_bytes(
        b"conflicting-video"
    )
    with pytest.raises(PreparedTrainingViewError, match="conflicting source"):
        build_prepared_training_view(
            tmp_path / "conflict",
            (
                PreparedSourceDataset("first", first),
                PreparedSourceDataset("second", second),
            ),
        )


def test_reader_rejects_tampered_derived_table(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _dataset(
        source,
        episode_uuid="episode-a",
        episode_index=0,
        profiles=_profiles(),
        annotate=False,
    )
    result = build_prepared_training_view(
        tmp_path / "organized",
        (PreparedSourceDataset("source", source),),
    )
    intervals = result.view_root / "semantic_intervals.parquet"
    intervals.write_bytes(intervals.read_bytes() + b"tamper")

    with pytest.raises(PreparedTrainingViewError, match="hash/size mismatch"):
        load_prepared_training_view(result.view_root)

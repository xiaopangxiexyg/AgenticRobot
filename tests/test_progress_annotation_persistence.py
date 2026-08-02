from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from collection_runtime.progress import (
    AbortEpisodeProgressRequest,
    ProgressController,
    ProgressFactStore,
    ProgressRequestContext,
    RefinementAnnotationStore,
    RefinementAnnotationStoreError,
    RefinementAnnotationStoreIncompleteError,
    StartEpisodeProgressRequest,
)
from dataset_contracts.canonical_encoding import canonical_bytes
from dataset_contracts.progress_annotations import (
    REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1,
    RefinementAnnotationSetBindingV1,
    RefinementBoundaryAnnotationV1,
)
from dataset_contracts.progress_facts import ProgressEpisodeManifestV1
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_refinement import (
    RefinementBoundaryDefinitionV1,
    TaskRefinementProfileV1,
)
from dataset_interface.progress_annotations import (
    ProgressAnnotationReadError,
    load_selected_refinement_annotation_set,
)


pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _progress_profile() -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="annotation-progress",
        profile_version="1",
        task_id="annotation",
        steps=(
            ProgressStepDefinitionV1("a", "MANIPULATE", "A", 1),
        ),
        initial_step_id="a",
        terminal_step_ids=("a",),
    )


def _refinement_profile() -> TaskRefinementProfileV1:
    return TaskRefinementProfileV1(
        profile_id="annotation-refinement",
        profile_version="1",
        task_progress_profile_hash=_progress_profile().semantic_content_hash,
        boundaries=(
            RefinementBoundaryDefinitionV1(
                progress_step_id="a",
                refinement_boundary_id="enter_a2",
                ordinal=1,
                semantics="ENTER_STAGE",
            ),
            RefinementBoundaryDefinitionV1(
                progress_step_id="a",
                refinement_boundary_id="enter_a3",
                ordinal=2,
                semantics="ENTER_STAGE",
            ),
        ),
    )


def _source_manifest() -> ProgressEpisodeManifestV1:
    return ProgressEpisodeManifestV1(
        episode_capture_identity_hash="1" * 64,
        task_progress_profile_hash=_progress_profile().semantic_content_hash,
        progress_episode_binding_hash="2" * 64,
        fact_count=1,
        fact_bundle_hash="3" * 64,
        terminal_fact_id="termination",
        sealed_at=10.0,
        sealed_clock_domain_id="episode_clock",
    )


def _binding() -> RefinementAnnotationSetBindingV1:
    return RefinementAnnotationSetBindingV1(
        annotation_set_id="annotations:1",
        source_progress_manifest_hash=(
            _source_manifest().semantic_content_hash
        ),
        task_refinement_profile_hash=(
            _refinement_profile().semantic_content_hash
        ),
        bound_clock_domain_id="episode_clock",
    )


def _annotation(
    annotation_id: str,
    boundary_id: str,
    at: float,
) -> RefinementBoundaryAnnotationV1:
    return RefinementBoundaryAnnotationV1(
        annotation_id=annotation_id,
        coarse_attempt_id="attempt:a",
        refinement_boundary_id=boundary_id,
        effective_at=at,
        clock_domain_id="episode_clock",
        confidence=0.9,
    )


def _store(
    root: Path,
    *,
    max_annotations_per_chunk: int = 128,
) -> RefinementAnnotationStore:
    return RefinementAnnotationStore(
        annotation_dir=root,
        source_progress_manifest=_source_manifest(),
        refinement_profile=_refinement_profile(),
        binding=_binding(),
        max_annotations_per_chunk=max_annotations_per_chunk,
    )


def test_annotation_store_chunks_seals_and_rebuilds_selected_view(
    tmp_path: Path,
) -> None:
    root = tmp_path / "annotations"
    store = _store(root, max_annotations_per_chunk=2)
    annotations = (
        _annotation("annotation:1", "enter_a2", 2.0),
        _annotation("annotation:2", "enter_a3", 3.0),
        _annotation("annotation:3", "enter_a3", 3.1),
        _annotation("annotation:4", "enter_a2", 4.0),
        _annotation("annotation:5", "enter_a3", 5.0),
    )
    report = store.append_annotations(annotations)
    assert report.accepted_count == 5
    assert report.idempotent_count == 0
    assert report.published_chunk_count == 3
    assert tuple(
        sorted(path.name for path in (root / "chunks").iterdir())
    ) == (
        "chunk-000000.cjson",
        "chunk-000001.cjson",
        "chunk-000002.cjson",
    )
    seal = store.seal()
    assert seal.manifest.annotation_count == 5
    store.close()

    selected = load_selected_refinement_annotation_set(
        annotation_dir=root,
        source_progress_manifest=_source_manifest(),
        refinement_profile=_refinement_profile(),
    )
    assert selected.annotation_set_id == "annotations:1"
    assert (
        selected.source_annotation_manifest_hash
        == seal.manifest.semantic_content_hash
    )
    assert selected.sealed
    assert tuple(
        item.annotation_id for item in selected.observations
    ) == tuple(item.annotation_id for item in annotations)


def test_unsealed_restart_recovers_chunks_and_idempotent_retry(
    tmp_path: Path,
) -> None:
    root = tmp_path / "annotations"
    first = _annotation("annotation:1", "enter_a2", 2.0)
    second = _annotation("annotation:2", "enter_a3", 3.0)
    store = _store(root)
    store.append_annotations((first,))
    store.close()

    reopened = _store(root)
    assert reopened.annotations == (first,)
    report = reopened.append_annotations((first, second))
    assert report.accepted_count == 1
    assert report.idempotent_count == 1
    assert report.published_chunk_count == 1
    reopened.seal()
    reopened.close()

    sealed = _store(root)
    assert sealed.sealed
    assert sealed.annotations == (first, second)
    sealed.close()


def test_annotation_id_conflict_and_foreign_clock_fail_closed(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "annotations")
    original = _annotation("annotation:1", "enter_a2", 2.0)
    store.append_annotations((original,))
    with pytest.raises(RefinementAnnotationStoreError, match="ID conflict"):
        store.append_annotations(
            (replace(original, effective_at=2.1),)
        )
    with pytest.raises(RefinementAnnotationStoreError, match="Episode clock"):
        store.append_annotations(
            (
                replace(
                    _annotation("annotation:2", "enter_a3", 3.0),
                    clock_domain_id="foreign_clock",
                ),
            )
        )
    store.close()


def test_empty_annotation_set_can_seal_but_cannot_append_afterward(
    tmp_path: Path,
) -> None:
    root = tmp_path / "annotations"
    store = _store(root)
    report = store.seal()
    assert report.manifest.annotation_count == 0
    assert report.manifest.chunk_hashes == ()
    with pytest.raises(RefinementAnnotationStoreError, match="sealed"):
        store.append_annotations(
            (_annotation("annotation:1", "enter_a2", 2.0),)
        )
    store.close()

    selected = load_selected_refinement_annotation_set(
        annotation_dir=root,
        source_progress_manifest=_source_manifest(),
        refinement_profile=_refinement_profile(),
    )
    assert selected.observations == ()


def test_reader_rejects_unsealed_and_extra_chunk_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "annotations"
    store = _store(root)
    store.append_annotations(
        (_annotation("annotation:1", "enter_a2", 2.0),)
    )
    store.close()
    with pytest.raises(ProgressAnnotationReadError, match="not sealed"):
        load_selected_refinement_annotation_set(
            annotation_dir=root,
            source_progress_manifest=_source_manifest(),
            refinement_profile=_refinement_profile(),
        )

    reopened = _store(root)
    reopened.seal()
    reopened.close()
    extra = root / "chunks" / "chunk-000001.cjson"
    extra.write_bytes(
        canonical_bytes(
            {
                "schema": "dexteleop.refinement_annotation_chunk.v1",
                "chunk_index": 1,
                "first_annotation_index": 1,
                "annotations": (
                    _annotation(
                        "annotation:extra",
                        "enter_a3",
                        3.0,
                    ).to_payload(),
                ),
            },
            domain=REFINEMENT_ANNOTATION_CHUNK_FILE_DOMAIN_V1,
        )
    )
    with pytest.raises(
        ProgressAnnotationReadError,
        match="does not bind",
    ):
        load_selected_refinement_annotation_set(
            annotation_dir=root,
            source_progress_manifest=_source_manifest(),
            refinement_profile=_refinement_profile(),
        )


def test_store_rejects_corrupt_or_gapped_chunks_on_restart(
    tmp_path: Path,
) -> None:
    corrupt_root = tmp_path / "corrupt"
    store = _store(corrupt_root)
    store.append_annotations(
        (_annotation("annotation:1", "enter_a2", 2.0),)
    )
    store.close()
    (corrupt_root / "chunks" / "chunk-000000.cjson").write_bytes(b"{bad")
    with pytest.raises(
        RefinementAnnotationStoreIncompleteError,
        match="Invalid annotation chunk",
    ):
        _store(corrupt_root)

    gapped_root = tmp_path / "gapped"
    store = _store(gapped_root)
    store.close()
    (gapped_root / "chunks" / "chunk-000001.cjson").write_bytes(b"unused")
    with pytest.raises(
        RefinementAnnotationStoreIncompleteError,
        match="not a contiguous",
    ):
        _store(gapped_root)


def test_binding_rejects_wrong_source_profile_or_clock(tmp_path: Path) -> None:
    source = _source_manifest()
    profile = _refinement_profile()
    binding = _binding()
    cases = (
        replace(binding, source_progress_manifest_hash="9" * 64),
        replace(binding, task_refinement_profile_hash="8" * 64),
        replace(binding, bound_clock_domain_id="foreign_clock"),
    )
    for index, invalid in enumerate(cases):
        with pytest.raises(ValueError):
            RefinementAnnotationStore(
                annotation_dir=tmp_path / f"invalid-{index}",
                source_progress_manifest=source,
                refinement_profile=profile,
                binding=invalid,
            )


def test_old_episode_annotation_and_new_episode_progress_can_commit_in_parallel(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old-episode"
    old_root.mkdir()
    raw_path = old_root / "raw.bin"
    raw_path.write_bytes(b"immutable raw payload")
    annotation_store = _store(old_root / "annotations")

    live_identity = EpisodeCaptureIdentity(
        episode_uuid="live-episode",
        legacy_episode_index=1,
        lifecycle_owner_id="fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="episode_clock",
    )
    live_profile = _progress_profile()
    live_binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=live_identity.semantic_content_hash,
        task_progress_profile_hash=live_profile.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope="FULL_TASK",
        entry_step_id="a",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    live_store = ProgressFactStore(
        episode_dir=tmp_path / "live-episode" / "progress",
        identity=live_identity,
        profile=live_profile,
        binding=live_binding,
    )
    controller = ProgressController(store=live_store)

    def annotate_old_episode() -> None:
        annotation_store.append_annotations(
            (_annotation("annotation:parallel", "enter_a2", 2.0),)
        )
        annotation_store.seal()

    def collect_new_episode_progress() -> None:
        controller.start(
            StartEpisodeProgressRequest("attempt:live"),
            context=ProgressRequestContext(
                request_id="live:start",
                effective_at=1.0,
                effective_clock_domain_id="episode_clock",
                available_at=1.0,
                available_clock_domain_id="episode_clock",
                actor_id="fixture",
            ),
        )
        controller.abort(
            AbortEpisodeProgressRequest("MANUAL_STOP"),
            context=ProgressRequestContext(
                request_id="live:stop",
                effective_at=2.0,
                effective_clock_domain_id="episode_clock",
                available_at=2.0,
                available_clock_domain_id="episode_clock",
                actor_id="fixture",
            ),
        )
        controller.seal(
            sealed_at=2.0,
            sealed_clock_domain_id="episode_clock",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(annotate_old_episode),
            executor.submit(collect_new_episode_progress),
        )
        for future in futures:
            future.result()

    assert raw_path.read_bytes() == b"immutable raw payload"
    assert annotation_store.sealed
    assert live_store.sealed
    annotation_store.close()
    live_store.close()

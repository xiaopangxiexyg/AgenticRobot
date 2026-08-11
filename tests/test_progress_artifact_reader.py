from __future__ import annotations

from pathlib import Path

import pytest

from collection_runtime.progress import (
    CompleteEpisodeProgressRequest,
    ProgressController,
    ProgressFactStore,
    ProgressRequestContext,
    StartEpisodeProgressRequest,
)
from collection_runtime.immutable_canonical_store import (
    publish_immutable_canonical,
)
from dataset_contracts.progress_facts import PROGRESS_PROFILE_FILE_DOMAIN_V1
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_interface.progress_artifacts import (
    DATASET_PROGRESS_PROFILE_RELATIVE_PATH,
    ProgressArtifactReadCache,
    ProgressArtifactReadError,
    load_dataset_progress_profile,
    load_sealed_progress_episode,
)


def _context(request_id: str, at: float) -> ProgressRequestContext:
    return ProgressRequestContext(
        request_id=request_id,
        effective_at=at,
        effective_clock_domain_id="episode_clock",
        available_at=at,
        available_clock_domain_id="episode_clock",
        actor_id="fixture",
    )


def _progress_episode(
    root: Path,
    *,
    seal: bool,
) -> tuple[
    EpisodeCaptureIdentity,
    TaskProgressProfileV1,
    ProgressEpisodeBindingV1,
    ProgressFactStore,
]:
    identity = EpisodeCaptureIdentity(
        episode_uuid="reader-fixture",
        legacy_episode_index=0,
        lifecycle_owner_id="fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="preparation_clock",
    )
    profile = TaskProgressProfileV1(
        profile_id="reader-profile",
        profile_version="1",
        task_id="reader-task",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="execute",
                semantic_type="EXECUTE_TASK",
                display_name="Execute",
                ordinal=1,
            ),
        ),
        initial_step_id="execute",
        terminal_step_ids=("execute",),
    )
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=identity.semantic_content_hash,
        task_progress_profile_hash=profile.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope="FULL_TASK",
        entry_step_id="execute",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    store = ProgressFactStore(
        episode_dir=root,
        identity=identity,
        profile=profile,
        binding=binding,
    )
    controller = ProgressController(store=store)
    controller.start(
        StartEpisodeProgressRequest("attempt:1"),
        context=_context("start", 1.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context("complete", 2.0),
    )
    if seal:
        controller.seal(
            sealed_at=3.0,
            sealed_clock_domain_id="episode_clock",
        )
    return identity, profile, binding, store


def test_load_sealed_progress_episode_is_read_only_and_resolves_timeline(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "progress"
    identity, profile, binding, store = _progress_episode(
        episode_dir,
        seal=True,
    )
    before = {
        path.relative_to(episode_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in episode_dir.iterdir()
    }
    assert store.manifest is not None

    loaded = load_sealed_progress_episode(
        episode_dir,
        expected_episode_identity_hash=identity.semantic_content_hash,
        expected_profile_hash=profile.semantic_content_hash,
        expected_binding_hash=binding.semantic_content_hash,
        expected_manifest_hash=store.manifest.semantic_content_hash,
    )

    after = {
        path.relative_to(episode_dir): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in episode_dir.iterdir()
    }
    assert before == after
    assert loaded.manifest.fact_count == 2
    assert loaded.timeline.phase == "TERMINATED"
    assert loaded.timeline.full_task_outcome == "SUCCESS"
    assert loaded.timeline.attempts[0].outcome == "SUCCESS"


def test_sealed_episode_read_cache_reuses_manifest_bound_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dataset_interface.progress_artifacts as artifacts_module

    episode_dir = tmp_path / "progress"
    identity, _, _, _ = _progress_episode(episode_dir, seal=True)
    original = artifacts_module._read_profile
    calls = 0

    def counted(path: Path) -> TaskProgressProfileV1:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(artifacts_module, "_read_profile", counted)
    cache = ProgressArtifactReadCache()
    first = load_sealed_progress_episode(
        episode_dir,
        expected_episode_identity_hash=identity.semantic_content_hash,
        read_cache=cache,
    )
    second = load_sealed_progress_episode(
        episode_dir,
        expected_episode_identity_hash=identity.semantic_content_hash,
        read_cache=cache,
    )

    assert first.profile is second.profile
    assert calls == 1


def test_progress_artifact_reader_rejects_unsealed_episode(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "progress"
    *_, store = _progress_episode(episode_dir, seal=False)
    store.close()

    with pytest.raises(ProgressArtifactReadError, match="manifest.cjson"):
        load_sealed_progress_episode(episode_dir)


def test_progress_artifact_reader_rejects_torn_or_ambiguous_fact_log(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "progress"
    _progress_episode(episode_dir, seal=True)
    fact_path = episode_dir / "facts.cjsonl"
    fact_path.write_bytes(fact_path.read_bytes() + b"\n")

    with pytest.raises(ProgressArtifactReadError, match="empty line"):
        load_sealed_progress_episode(episode_dir)


def test_progress_artifact_reader_checks_owner_metadata_hashes(
    tmp_path: Path,
) -> None:
    episode_dir = tmp_path / "progress"
    _progress_episode(episode_dir, seal=True)

    with pytest.raises(
        ProgressArtifactReadError,
        match="Episode identity hash mismatch",
    ):
        load_sealed_progress_episode(
            episode_dir,
            expected_episode_identity_hash="0" * 64,
        )


def test_dataset_progress_profile_reader_verifies_all_episode_profiles(
    tmp_path: Path,
) -> None:
    episode_dir = (
        tmp_path / "meta" / "progress" / "episodes" / "reader-fixture"
    )
    _, profile, _, _ = _progress_episode(episode_dir, seal=True)
    publish_immutable_canonical(
        tmp_path / DATASET_PROGRESS_PROFILE_RELATIVE_PATH,
        profile.to_payload(),
        domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
    )

    assert load_dataset_progress_profile(tmp_path) == profile
    assert (
        load_dataset_progress_profile(
            tmp_path,
            require_explicit_binding=True,
        )
        == profile
    )


def test_dataset_progress_profile_reader_rejects_mixed_episode_profiles(
    tmp_path: Path,
) -> None:
    first_dir = (
        tmp_path / "meta" / "progress" / "episodes" / "reader-fixture"
    )
    _, profile, _, _ = _progress_episode(first_dir, seal=True)
    changed = TaskProgressProfileV1(
        profile_id="other-profile",
        profile_version="1",
        task_id=profile.task_id,
        steps=profile.steps,
        initial_step_id=profile.initial_step_id,
        terminal_step_ids=profile.terminal_step_ids,
    )
    publish_immutable_canonical(
        tmp_path
        / "meta"
        / "progress"
        / "episodes"
        / "other-episode"
        / "profile.cjson",
        changed.to_payload(),
        domain=PROGRESS_PROFILE_FILE_DOMAIN_V1,
    )

    with pytest.raises(
        ProgressArtifactReadError,
        match="different profile hashes",
    ):
        load_dataset_progress_profile(tmp_path)

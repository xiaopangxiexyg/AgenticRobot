from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset_contracts.canonical_encoding import canonical_loads
from dataset_contracts.embodiment import (
    CaptureProjectionComponent,
    CaptureProjectionProfile,
    EmbodimentCaptureProfile,
    EmbodimentComponentCaptureSpec,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonSubtask,
    LongHorizonTaskProfile,
    TaskSemanticsContractError,
)
from dataset_contracts.v25p import (
    V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    V25P_EPISODE_RELATIVE_ROOT_V1,
)
from dataset_interface.v25p import V25pReadError, load_v25p_episode
from collection_runtime.v25p_sidecar import V25pEpisodeSidecarStore


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _store(root: Path, *, clock=lambda: 10.0) -> V25pEpisodeSidecarStore:
    embodiment = EmbodimentCaptureProfile(
        profile_id="generic_arm",
        profile_version="1",
        topology_id="one_arm",
        component_catalog_contract_id="robot_motion.component_catalog.v1",
        component_catalog_contract_hash=HASH_A,
        target_capability_contract_id="robot_motion.target_capability.v1",
        target_capability_contract_hash=HASH_B,
        components=(
            EmbodimentComponentCaptureSpec(
                component_id="arm.primary",
                component_kind="manipulator",
                controlled_entity_id="arm",
                command_payload_schema_hash=HASH_C,
                measurement_payload_schema_hash=HASH_D,
            ),
        ),
    )
    projection = CaptureProjectionProfile(
        projection_id="generic_v23",
        projection_version="1",
        temporal_alignment_contract_id="causal.v1",
        temporal_alignment_contract_hash=HASH_A,
        row_encoder_contract_id="row.v23",
        row_encoder_contract_hash=HASH_B,
        policy_io_layout_hash=HASH_C,
        pose_frame_contract_hash=HASH_D,
        causal_snapshot_contract_hash=HASH_E,
        components=(
            CaptureProjectionComponent(
                "arm.primary", "action", "actions.arm", HASH_C
            ),
            CaptureProjectionComponent(
                "arm.primary", "state", "observation.state.arm", HASH_D
            ),
        ),
    )
    task = LongHorizonTaskProfile(
        profile_id="pick",
        profile_version="1",
        task_id="pick",
        subtasks=(LongHorizonSubtask("approach", "Approach"),),
    )
    return V25pEpisodeSidecarStore(
        dataset_root=root,
        dataset_id="dataset",
        embodiment_profile=embodiment,
        capture_projection_profile=projection,
        task_profile=task,
        clock=clock,
        clock_domain_id="wall",
    )


def _identity(index: int = 0, episode_uuid: str = "episode-u") -> EpisodeCaptureIdentity:
    return EpisodeCaptureIdentity(
        episode_uuid=episode_uuid,
        legacy_episode_index=index,
        lifecycle_owner_id="test-owner",
        prepared_at=1.0,
        prepared_clock_domain_id="control",
    )


def _write_base_episode(root: Path, index: int) -> None:
    path = root / "meta" / "episodes.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"episode_index": index, "frame_count": 3}) + "\n",
        encoding="utf-8",
    )


def test_sidecar_seals_only_after_legacy_record_and_auto_closes_attempt(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    identity = _identity()
    store.prepare_episode(identity)
    binding = store.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="control",
    )
    store.begin_subtask(
        identity,
        track_id="primary_subtask",
        attempt_id="attempt-1",
        subtask_id="approach",
        effective_at=3.0,
        available_at=3.0,
        effective_clock_domain_id="control",
        available_clock_domain_id="control",
        actor_id="operator",
    )
    _write_base_episode(tmp_path, 0)
    report = store.finalize_after_legacy_episode(
        identity,
        sealed_at=20.0,
        sealed_clock_domain_id="wall",
        task_effective_at=4.0,
        task_effective_clock_domain_id="control",
    )

    assert report.embodiment_binding_hash == binding.semantic_content_hash
    assert report.auto_closed_attempt_count == 1
    episode_dir = tmp_path / V25P_EPISODE_RELATIVE_ROOT_V1 / "episode-u"
    assert len(tuple((episode_dir / "task_events").glob("*.cjson"))) == 2
    payload = canonical_loads(
        (episode_dir / "episode_manifest.cjson").read_bytes(),
        expected_domain=V25P_EPISODE_MANIFEST_FILE_DOMAIN_V1,
    )
    assert payload["semantic_content_hash"] == report.episode_manifest_hash


def test_duplicate_success_is_rejected_before_event_publication(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="control",
    )
    for sequence in (1, 2):
        attempt = f"attempt-{sequence}"
        store.begin_subtask(
            identity,
            track_id="primary_subtask",
            attempt_id=attempt,
            subtask_id="approach",
            effective_at=float(sequence * 2),
            available_at=float(sequence * 2),
            effective_clock_domain_id="control",
            available_clock_domain_id="control",
            actor_id="operator",
        )
        if sequence == 1:
            store.finish_subtask(
                identity,
                track_id="primary_subtask",
                attempt_id=attempt,
                subtask_id="approach",
                outcome="success",
                effective_at=3.0,
                available_at=3.0,
                effective_clock_domain_id="control",
                available_clock_domain_id="control",
                actor_id="operator",
            )
    with pytest.raises(TaskSemanticsContractError, match="consecutive successful"):
        store.finish_subtask(
            identity,
            track_id="primary_subtask",
            attempt_id="attempt-2",
            subtask_id="approach",
            outcome="success",
            effective_at=5.0,
            available_at=5.0,
            effective_clock_domain_id="control",
            available_clock_domain_id="control",
            actor_id="operator",
        )
    event_dir = (
        tmp_path / V25P_EPISODE_RELATIVE_ROOT_V1 / "episode-u" / "task_events"
    )
    assert len(tuple(event_dir.glob("*.cjson"))) == 3


def test_discard_removes_only_unsealed_episode_sidecars(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(
        identity,
        bound_at=2.0,
        bound_clock_domain_id="control",
    )
    episode_dir = tmp_path / V25P_EPISODE_RELATIVE_ROOT_V1 / "episode-u"
    assert episode_dir.is_dir()
    store.discard_episode(identity)
    assert not episode_dir.exists()
    assert (tmp_path / "meta" / "v25p" / "dataset_manifest.cjson").is_file()


def test_dataset_binding_is_idempotent_across_store_restart(tmp_path: Path) -> None:
    first = _store(tmp_path, clock=lambda: 10.0)
    second = _store(tmp_path, clock=lambda: 99.0)
    assert second.dataset_manifest == first.dataset_manifest


def test_sealed_sidecar_round_trips_through_read_only_interface(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(identity, bound_at=2.0, bound_clock_domain_id="control")
    store.begin_subtask(
        identity,
        track_id="primary_subtask",
        attempt_id="attempt-1",
        subtask_id="approach",
        effective_at=3.0,
        available_at=3.0,
        effective_clock_domain_id="control",
        available_clock_domain_id="control",
        actor_id="operator",
    )
    store.finish_subtask(
        identity,
        track_id="primary_subtask",
        attempt_id="attempt-1",
        subtask_id="approach",
        outcome="success",
        effective_at=4.0,
        available_at=4.0,
        effective_clock_domain_id="control",
        available_clock_domain_id="control",
        actor_id="operator",
    )
    _write_base_episode(tmp_path, 0)
    store.finalize_after_legacy_episode(
        identity,
        sealed_at=20.0,
        sealed_clock_domain_id="wall",
    )

    loaded = load_v25p_episode(tmp_path, identity.episode_uuid)

    assert loaded.identity == identity
    assert len(loaded.task_events) == 2
    assert loaded.readiness.valid is True
    assert loaded.readiness.native_v25 is False


def test_reader_rejects_rewritten_legacy_episode_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    identity = _identity()
    store.prepare_episode(identity)
    store.bind_prepared_episode(identity, bound_at=2.0, bound_clock_domain_id="control")
    _write_base_episode(tmp_path, 0)
    store.finalize_after_legacy_episode(
        identity,
        sealed_at=20.0,
        sealed_clock_domain_id="wall",
    )
    _write_base_episode(tmp_path, 0)
    path = tmp_path / "meta" / "episodes.jsonl"
    path.write_text(
        json.dumps({"episode_index": 0, "frame_count": 4}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(V25pReadError, match="no longer matches"):
        load_v25p_episode(tmp_path, identity.episode_uuid)

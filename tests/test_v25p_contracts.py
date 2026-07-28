from __future__ import annotations

from dataclasses import replace

import pytest

from dataset_contracts.embodiment import (
    CaptureProjectionComponent,
    CaptureProjectionProfile,
    EmbodimentCaptureProfile,
    EmbodimentComponentCaptureSpec,
    EpisodeEmbodimentBinding,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.task_semantics import (
    LongHorizonSubtask,
    LongHorizonTaskProfile,
    SubtaskAttemptEvent,
    TaskSemanticsContractError,
    TaskSemanticsEpisodeManifest,
    validate_subtask_attempt_events,
)
from dataset_contracts.v25p import V25pDatasetManifest, V25pEpisodeManifest
from dataset_interface.task_semantics import project_task_semantics_to_frames
from dataset_interface.v25p import (
    V25P_ASSESSMENT_INVALID,
    V25P_ASSESSMENT_V25_COMPATIBILITY,
    V25P_ASSESSMENT_V25_NATIVE_MERGE,
    assess_v25p_merge_readiness,
)


HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _profiles():
    embodiment = EmbodimentCaptureProfile(
        profile_id="rm65_bimanual",
        profile_version="1",
        topology_id="dual_arm_two_gripper",
        component_catalog_contract_id="robot_motion.component_catalog.v1",
        component_catalog_contract_hash=HASH_A,
        target_capability_contract_id="robot_motion.target_capability.v1",
        target_capability_contract_hash=HASH_B,
        components=(
            EmbodimentComponentCaptureSpec(
                component_id="arm.primary",
                component_kind="manipulator",
                controlled_entity_id="arm_a",
                command_payload_schema_hash=HASH_C,
                measurement_payload_schema_hash=HASH_D,
                coordinate_names=("j1", "j2"),
                units="rad",
            ),
        ),
    )
    projection = CaptureProjectionProfile(
        projection_id="rm65_v23",
        projection_version="1",
        temporal_alignment_contract_id="dexteleop.causal_snapshot.v1",
        temporal_alignment_contract_hash=HASH_A,
        row_encoder_contract_id="dexteleop.row_encoder.v23",
        row_encoder_contract_hash=HASH_B,
        policy_io_layout_hash=HASH_C,
        pose_frame_contract_hash=HASH_D,
        causal_snapshot_contract_hash=HASH_E,
        components=(
            CaptureProjectionComponent(
                component_id="arm.primary",
                dataset_role="action",
                dataset_field="actions.arm_primary",
                value_schema_hash=HASH_C,
            ),
            CaptureProjectionComponent(
                component_id="arm.primary",
                dataset_role="state",
                dataset_field="observation.state.arm_primary",
                value_schema_hash=HASH_D,
            ),
        ),
    )
    task = LongHorizonTaskProfile(
        profile_id="long_pick",
        profile_version="1",
        task_id="pick_and_place",
        subtasks=(
            LongHorizonSubtask("pick", "Pick"),
            LongHorizonSubtask("place", "Place"),
        ),
    )
    return embodiment, projection, task


def _event(
    sequence: int,
    *,
    attempt: str,
    subtask: str,
    kind: str,
    outcome: str | None = None,
) -> SubtaskAttemptEvent:
    timestamp = float(sequence)
    return SubtaskAttemptEvent(
        event_id=f"event-{sequence}",
        episode_uuid="episode-u",
        legacy_episode_index=4,
        event_sequence=sequence,
        track_id="primary_subtask",
        attempt_id=attempt,
        subtask_id=subtask,
        event_kind=kind,
        outcome=outcome,
        effective_at=timestamp,
        available_at=timestamp + 0.01,
        effective_clock_domain_id="session",
        available_clock_domain_id="session",
        actor_id="operator",
    )


def test_contract_round_trips_and_preserves_v23_base() -> None:
    embodiment, projection, task = _profiles()
    identity = EpisodeCaptureIdentity("episode-u", 4, "dataset_writer", 1.0, "session")

    assert EmbodimentCaptureProfile.from_payload(
        embodiment.to_payload()
    ) == embodiment
    assert CaptureProjectionProfile.from_payload(projection.to_payload()) == projection
    assert LongHorizonTaskProfile.from_payload(task.to_payload()) == task
    assert EpisodeCaptureIdentity.from_payload(identity.to_payload()) == identity
    assert projection.base_dataset_schema == "dexteleop_dataset_v2.3"
    with pytest.raises(ValueError, match="v2.3"):
        replace(projection, base_dataset_schema="dexteleop_dataset_v2.5")


def test_failed_then_success_is_allowed_but_duplicate_success_is_rejected() -> None:
    _, _, profile = _profiles()
    recovery = (
        _event(1, attempt="pick-1", subtask="pick", kind="begin"),
        _event(2, attempt="pick-1", subtask="pick", kind="finish", outcome="failure"),
        _event(3, attempt="pick-2", subtask="pick", kind="begin"),
        _event(4, attempt="pick-2", subtask="pick", kind="finish", outcome="success"),
    )
    report = validate_subtask_attempt_events(profile, recovery, require_closed=True)
    assert report.attempt_count == 2
    duplicate_success = recovery + (
        _event(5, attempt="pick-3", subtask="pick", kind="begin"),
        _event(6, attempt="pick-3", subtask="pick", kind="finish", outcome="success"),
    )
    with pytest.raises(TaskSemanticsContractError, match="consecutive successful"):
        validate_subtask_attempt_events(profile, duplicate_success, require_closed=True)


def test_open_attempt_cannot_be_sealed_and_frame_projection_is_rebuildable() -> None:
    _, _, profile = _profiles()
    events = (_event(1, attempt="pick-1", subtask="pick", kind="begin"),)
    with pytest.raises(TaskSemanticsContractError, match="finish event"):
        validate_subtask_attempt_events(profile, events, require_closed=True)
    snapshots = project_task_semantics_to_frames(
        profile=profile,
        events=events,
        frame_times=(0.5, 1.0, 2.0),
        frame_clock_domain_id="session",
        track_id="primary_subtask",
    )
    assert snapshots[0].active_subtask_id is None
    assert snapshots[1].active_subtask_id == "pick"
    assert snapshots[2].active_attempt_id == "pick-1"


def test_backdated_post_annotation_projects_by_effective_time_not_append_order() -> None:
    _, _, profile = _profiles()
    events = (
        SubtaskAttemptEvent(
            event_id="post-begin",
            episode_uuid="episode-u",
            legacy_episode_index=4,
            event_sequence=1,
            track_id="primary_subtask",
            attempt_id="pick-post",
            subtask_id="pick",
            event_kind="begin",
            effective_at=10.0,
            available_at=100.0,
            effective_clock_domain_id="capture",
            available_clock_domain_id="annotation",
            actor_id="reviewer",
            origin="post_annotation",
        ),
        SubtaskAttemptEvent(
            event_id="post-finish",
            episode_uuid="episode-u",
            legacy_episode_index=4,
            event_sequence=2,
            track_id="primary_subtask",
            attempt_id="pick-post",
            subtask_id="pick",
            event_kind="finish",
            outcome="success",
            effective_at=20.0,
            available_at=101.0,
            effective_clock_domain_id="capture",
            available_clock_domain_id="annotation",
            actor_id="reviewer",
            origin="post_annotation",
            source_event_ids=("post-begin",),
        ),
    )
    validate_subtask_attempt_events(profile, events, require_closed=True)
    snapshots = project_task_semantics_to_frames(
        profile=profile,
        events=events,
        frame_times=(9.0, 15.0, 21.0),
        frame_clock_domain_id="capture",
        track_id="primary_subtask",
    )
    assert snapshots[0].active_subtask_id is None
    assert snapshots[1].active_subtask_id == "pick"
    assert snapshots[2].previous_outcome == "success"


def test_merge_assessment_never_claims_native_v25() -> None:
    embodiment, projection, task = _profiles()
    identity = EpisodeCaptureIdentity("episode-u", 4, "dataset_writer", 1.0, "session")
    binding = EpisodeEmbodimentBinding(
        episode_uuid="episode-u",
        legacy_episode_index=4,
        episode_identity_hash=identity.semantic_content_hash,
        embodiment_profile_hash=embodiment.semantic_content_hash,
        capture_projection_profile_hash=projection.semantic_content_hash,
        bound_at=1.1,
        bound_clock_domain_id="session",
    )
    events = (
        _event(1, attempt="pick-1", subtask="pick", kind="begin"),
        _event(2, attempt="pick-1", subtask="pick", kind="finish", outcome="success"),
    )
    task_manifest = TaskSemanticsEpisodeManifest.create(
        profile=task,
        episode_uuid="episode-u",
        legacy_episode_index=4,
        episode_identity_hash=identity.semantic_content_hash,
        events=events,
        sealed_at=3.0,
        sealed_clock_domain_id="session",
    )
    dataset_manifest = V25pDatasetManifest(
        dataset_id="dataset-u",
        policy_io_layout_hash=projection.policy_io_layout_hash,
        pose_frame_contract_hash=projection.pose_frame_contract_hash,
        embodiment_profile_hashes=(embodiment.semantic_content_hash,),
        capture_projection_profile_hashes=(projection.semantic_content_hash,),
        long_horizon_task_profile_hashes=(task.semantic_content_hash,),
        created_at=0.0,
        created_clock_domain_id="session",
    )
    episode_manifest = V25pEpisodeManifest(
        episode_uuid="episode-u",
        legacy_episode_index=4,
        episode_identity_hash=identity.semantic_content_hash,
        embodiment_binding_hash=binding.semantic_content_hash,
        base_episode_manifest_hash=HASH_A,
        task_semantics_manifest_hash=task_manifest.semantic_content_hash,
        sealed_at=3.1,
        sealed_clock_domain_id="session",
    )
    readiness = assess_v25p_merge_readiness(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
        identity=identity,
        embodiment_profile=embodiment,
        projection_profile=projection,
        embodiment_binding=binding,
        task_semantics_manifest=task_manifest,
    )
    assert readiness.classification == V25P_ASSESSMENT_V25_COMPATIBILITY
    assert readiness.valid
    assert not readiness.native_v25

    native_candidate = assess_v25p_merge_readiness(
        dataset_manifest=dataset_manifest,
        episode_manifest=replace(episode_manifest, v24_capture_commit_hash=HASH_B),
        identity=identity,
        embodiment_profile=embodiment,
        projection_profile=projection,
        embodiment_binding=binding,
        task_semantics_manifest=task_manifest,
    )
    assert native_candidate.classification == V25P_ASSESSMENT_V25_NATIVE_MERGE
    assert not native_candidate.native_v25

    invalid = assess_v25p_merge_readiness(
        dataset_manifest=dataset_manifest,
        episode_manifest=episode_manifest,
        identity=identity,
        embodiment_profile=embodiment,
        projection_profile=projection,
        embodiment_binding=replace(binding, embodiment_profile_hash=HASH_A),
        task_semantics_manifest=task_manifest,
    )
    assert invalid.classification == V25P_ASSESSMENT_INVALID
    assert not invalid.valid

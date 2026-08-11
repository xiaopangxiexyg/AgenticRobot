"""Resolve migrated legacy Progress windows without inventing capture facts."""

from __future__ import annotations

from dataclasses import dataclass

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    TaskProgressProfileV1,
    validate_progress_episode_binding,
)
from dataset_contracts.progress_stages import (
    TaskStageProfileV1,
    validate_task_stage_profile,
)
from dataset_contracts.retrospective_progress import (
    RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1,
    RetrospectiveProgressMigrationManifestV1,
    RetrospectiveProgressWindowV1,
    RetrospectiveSourceBindingV1,
)

from .sample_axis import EpisodeSampleAxis, SampleIndexInterval


RETROSPECTIVE_SAMPLE_AXIS_HASH_DOMAIN_V1 = (
    "dataset_interface.retrospective_sample_axis.semantic.v1"
)


class RetrospectiveProgressResolutionError(ValueError):
    """Raised when migrated semantics cannot be proven from their bindings."""


@dataclass(frozen=True)
class ResolvedRetrospectiveAttempt:
    """One migrated Attempt; source anchors are sample indices, not facts."""

    attempt_id: str
    progress_step_id: str
    attempt_ordinal_in_window: int
    sample_interval: SampleIndexInterval
    start_at: float
    end_at: float
    clock_domain_id: str
    outcome: str
    failure_fact_id: str | None = None
    failure_reported_at: float | None = None


@dataclass(frozen=True)
class ResolvedRetrospectiveProgressWindow:
    """Verified read view over one sealed retrospective migration."""

    episode_capture_identity_hash: str
    task_progress_profile_hash: str
    progress_episode_binding_hash: str
    task_stage_profile_hash: str
    source_binding_hash: str
    source_dataset_base_fingerprint: str
    source_episode_record_hash: str
    source_parquet_hash: str
    sample_axis_hash: str
    sample_count: int
    clock_domain_id: str
    attempts: tuple[ResolvedRetrospectiveAttempt, ...]
    transitions: tuple[()] = ()
    capture_pause_intervals: tuple[()] = ()
    sealed: bool = True
    termination_cause: str = "WINDOW_COMPLETED"
    full_task_outcome: str = "NOT_APPLICABLE"
    resolution_kind: str = RETROSPECTIVE_PROGRESS_RESOLUTION_KIND_V1
    source_progress_manifest_hash: str = ""
    migration_manifest_hash: str = ""


def retrospective_sample_axis_hash(axis: EpisodeSampleAxis) -> str:
    """Hash the exact accepted-sample axis used by migrated annotations."""

    if not isinstance(axis, EpisodeSampleAxis):
        raise TypeError("axis must be EpisodeSampleAxis.")
    return canonical_content_hash(
        {
            "clock_domain_id": axis.clock_domain_id,
            "cut_times": axis.cut_times,
            "sample_count": axis.sample_count,
        },
        domain=RETROSPECTIVE_SAMPLE_AXIS_HASH_DOMAIN_V1,
    )


def _require_equal(*, name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise RetrospectiveProgressResolutionError(
            f"{name} mismatch: expected={expected!r} actual={actual!r}."
        )


def resolve_retrospective_progress_window(
    *,
    progress_profile: TaskProgressProfileV1,
    progress_binding: ProgressEpisodeBindingV1,
    stage_profile: TaskStageProfileV1,
    sample_axis: EpisodeSampleAxis,
    source_binding: RetrospectiveSourceBindingV1,
    window: RetrospectiveProgressWindowV1,
    manifest: RetrospectiveProgressMigrationManifestV1,
) -> ResolvedRetrospectiveProgressWindow:
    """Validate and resolve a full Episode Step window.

    No :class:`ProgressFactEnvelopeV1` values are accepted or emitted here.
    The returned Attempt is anchored directly to the immutable sample axis.
    """

    try:
        validate_progress_episode_binding(
            profile=progress_profile,
            binding=progress_binding,
        )
        validate_task_stage_profile(
            progress_profile=progress_profile,
            stage_profile=stage_profile,
        )
    except (TypeError, ValueError) as exc:
        raise RetrospectiveProgressResolutionError(
            f"Invalid retrospective profile binding: {exc}"
        ) from exc
    for value, expected_type, name in (
        (sample_axis, EpisodeSampleAxis, "sample_axis"),
        (
            source_binding,
            RetrospectiveSourceBindingV1,
            "source_binding",
        ),
        (window, RetrospectiveProgressWindowV1, "window"),
        (
            manifest,
            RetrospectiveProgressMigrationManifestV1,
            "manifest",
        ),
    ):
        if not isinstance(value, expected_type):
            raise TypeError(f"{name} must be {expected_type.__name__}.")

    if progress_binding.recording_scope != "PROGRESS_STEP_WINDOW":
        raise RetrospectiveProgressResolutionError(
            "A migrated first-subtask Episode must use "
            "PROGRESS_STEP_WINDOW."
        )
    if progress_binding.entry_mode != "PROFILE_INITIAL":
        raise RetrospectiveProgressResolutionError(
            "A complete migrated Step window must use PROFILE_INITIAL."
        )
    if not progress_binding.history_complete or progress_binding.truncated_at_start:
        raise RetrospectiveProgressResolutionError(
            "A complete migrated Step window must declare complete history "
            "and no start truncation."
        )
    if sample_axis.sample_count < 1:
        raise RetrospectiveProgressResolutionError(
            "A migrated Step window requires at least one accepted sample."
        )

    axis_hash = retrospective_sample_axis_hash(sample_axis)
    expected_hashes = {
        "Episode identity": source_binding.episode_capture_identity_hash,
        "source binding": source_binding.semantic_content_hash,
        "Progress profile": progress_profile.semantic_content_hash,
        "Progress Episode binding": progress_binding.semantic_content_hash,
        "task Stage profile": stage_profile.semantic_content_hash,
        "sample axis": axis_hash,
    }
    window_values = {
        "Episode identity": window.episode_capture_identity_hash,
        "source binding": window.source_binding_hash,
        "Progress profile": window.task_progress_profile_hash,
        "Progress Episode binding": window.progress_episode_binding_hash,
        "task Stage profile": window.task_stage_profile_hash,
        "sample axis": window.sample_axis_hash,
    }
    manifest_values = {
        "Episode identity": manifest.episode_capture_identity_hash,
        "source binding": manifest.source_binding_hash,
        "Progress profile": manifest.task_progress_profile_hash,
        "Progress Episode binding": manifest.progress_episode_binding_hash,
        "task Stage profile": manifest.task_stage_profile_hash,
        "sample axis": manifest.sample_axis_hash,
    }
    for name, expected in expected_hashes.items():
        _require_equal(name=name, actual=window_values[name], expected=expected)
        _require_equal(name=name, actual=manifest_values[name], expected=expected)
    _require_equal(
        name="retrospective window",
        actual=manifest.retrospective_window_hash,
        expected=window.semantic_content_hash,
    )
    _require_equal(
        name="sample count",
        actual=window.sample_count,
        expected=sample_axis.sample_count,
    )
    _require_equal(
        name="window clock",
        actual=window.clock_domain_id,
        expected=sample_axis.clock_domain_id,
    )
    _require_equal(
        name="Progress binding clock",
        actual=progress_binding.bound_clock_domain_id,
        expected=sample_axis.clock_domain_id,
    )
    _require_equal(
        name="Progress Step",
        actual=window.progress_step_id,
        expected=progress_binding.entry_step_id,
    )
    _require_equal(
        name="window start",
        actual=window.start_at,
        expected=sample_axis.cut_times[0],
    )
    _require_equal(
        name="window end",
        actual=window.end_at,
        expected=sample_axis.cut_times[-1],
    )
    for video in source_binding.admitted_videos:
        _require_equal(
            name=f"{video.modality_id} video frame count",
            actual=video.decoded_frame_count,
            expected=sample_axis.sample_count,
        )

    attempt = ResolvedRetrospectiveAttempt(
        attempt_id=window.attempt_id,
        progress_step_id=window.progress_step_id,
        attempt_ordinal_in_window=1,
        sample_interval=SampleIndexInterval(0, sample_axis.sample_count),
        start_at=window.start_at,
        end_at=window.end_at,
        clock_domain_id=window.clock_domain_id,
        outcome=window.active_attempt_disposition,
    )
    return ResolvedRetrospectiveProgressWindow(
        episode_capture_identity_hash=window.episode_capture_identity_hash,
        task_progress_profile_hash=window.task_progress_profile_hash,
        progress_episode_binding_hash=window.progress_episode_binding_hash,
        task_stage_profile_hash=window.task_stage_profile_hash,
        source_binding_hash=window.source_binding_hash,
        source_dataset_base_fingerprint=(
            source_binding.source_dataset_base_fingerprint
        ),
        source_episode_record_hash=source_binding.source_episode_record_hash,
        source_parquet_hash=source_binding.source_parquet_hash,
        sample_axis_hash=window.sample_axis_hash,
        sample_count=window.sample_count,
        clock_domain_id=window.clock_domain_id,
        attempts=(attempt,),
        termination_cause=window.termination_cause,
        full_task_outcome=window.full_task_outcome,
        source_progress_manifest_hash=manifest.semantic_content_hash,
        migration_manifest_hash=manifest.semantic_content_hash,
    )


__all__ = [
    "RETROSPECTIVE_SAMPLE_AXIS_HASH_DOMAIN_V1",
    "ResolvedRetrospectiveAttempt",
    "ResolvedRetrospectiveProgressWindow",
    "RetrospectiveProgressResolutionError",
    "resolve_retrospective_progress_window",
    "retrospective_sample_axis_hash",
]

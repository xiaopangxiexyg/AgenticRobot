from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from collection_runtime.progress import (
    AbortEpisodeProgressRequest,
    AdvanceProgressStepRequest,
    CompleteEpisodeProgressRequest,
    ProgressController,
    ProgressFactStore,
    ProgressRequestContext,
    ReportAttemptFailureRequest,
    ResumeFailedAttemptRequest,
    StartEpisodeProgressRequest,
    UndoLastProgressAdvanceRequest,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_facts import (
    AttemptRetryEventV1,
    HoldPauseEffectReceiptV1,
    ProgressFactEnvelopeV1,
    ProgressOperationReceiptV1,
    ResumeAfterFailureReceiptV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from dataset_interface.progress_timeline import (
    ProgressTimelineResolutionError,
    resolve_progress_timeline,
)
from dataset_interface.progress_corrections import (
    FailureAttemptReassignment,
    ProgressCorrectionResolutionError,
    reassign_failure_to_attempt,
)


pytestmark = [pytest.mark.integration, pytest.mark.contract]


def _context(
    index: int,
    at: float,
    *,
    available_at: float | None = None,
) -> ProgressRequestContext:
    return ProgressRequestContext(
        request_id=f"timeline:fact:{index}",
        effective_at=at,
        effective_clock_domain_id="episode_clock",
        available_at=at if available_at is None else available_at,
        available_clock_domain_id="episode_clock",
        actor_id="timeline-fixture",
    )


def _operation(
    operation: str,
    *,
    requested_at: float,
    status: str = "APPLIED",
    effective_at: float | None = None,
) -> ProgressOperationReceiptV1:
    if effective_at is None and status == "APPLIED":
        effective_at = requested_at
    return ProgressOperationReceiptV1(
        operation=operation,
        status=status,
        requested_at=requested_at,
        effective_at=effective_at,
        clock_domain_id="episode_clock",
    )


def _runtime(
    root: Path,
    *,
    scope: str = "FULL_TASK",
) -> tuple[
    EpisodeCaptureIdentity,
    TaskProgressProfileV1,
    ProgressEpisodeBindingV1,
    ProgressFactStore,
    ProgressController,
]:
    identity = EpisodeCaptureIdentity(
        episode_uuid="timeline-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="timeline-fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="episode_clock",
    )
    profile = TaskProgressProfileV1(
        profile_id="timeline-linear",
        profile_version="1",
        task_id="timeline-task",
        steps=(
            ProgressStepDefinitionV1("a", "PICK", "A", 1),
            ProgressStepDefinitionV1("b", "PLACE", "B", 2),
        ),
        initial_step_id="a",
        terminal_step_ids=("b",),
    )
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=identity.semantic_content_hash,
        task_progress_profile_hash=profile.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope=scope,
        entry_step_id="a",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    store = ProgressFactStore(
        episode_dir=root,
        identity=identity,
        profile=profile,
        binding=binding,
    )
    return identity, profile, binding, store, ProgressController(store=store)


def test_resolves_sealed_success_attempts_and_transition(tmp_path: Path) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    start = controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    advance = controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 1.0),
    )
    terminal = controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(3, 2.0),
    )
    seal = controller.seal(
        sealed_at=2.0,
        sealed_clock_domain_id="episode_clock",
    )

    timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=store.facts,
        manifest=seal.manifest,
        require_sealed=True,
    )

    assert timeline.sealed
    assert (
        timeline.source_progress_manifest_hash
        == seal.manifest.semantic_content_hash
    )
    assert timeline.phase == "TERMINATED"
    assert timeline.terminal_fact_id == terminal.fact.fact_id
    assert timeline.full_task_outcome == "SUCCESS"
    assert [
        (
            attempt.attempt_id,
            attempt.outcome,
            attempt.start_fact_id,
            attempt.end_fact_id,
        )
        for attempt in timeline.attempts
    ] == [
        ("attempt:a", "SUCCESS", start.fact.fact_id, advance.fact.fact_id),
        ("attempt:b", "SUCCESS", advance.fact.fact_id, terminal.fact.fact_id),
    ]
    assert len(timeline.transitions) == 1
    assert timeline.transitions[0].advance_fact_id == advance.fact.fact_id
    store.close()


def test_manifest_seal_time_cannot_precede_termination(tmp_path: Path) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 1.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(3, 2.0),
    )
    seal = controller.seal(
        sealed_at=2.0,
        sealed_clock_domain_id="episode_clock",
    )
    invalid_manifest = replace(seal.manifest, sealed_at=1.9)

    with pytest.raises(
        ProgressTimelineResolutionError,
        match="does not bind",
    ):
        resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=store.facts,
            manifest=invalid_manifest,
            require_sealed=True,
        )
    store.close()


def test_undo_removes_transient_attempt_and_implicit_success(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    mistaken = controller.advance(
        AdvanceProgressStepRequest("attempt:b:mistaken"),
        context=_context(2, 1.0),
    )
    controller.undo_last_advance(
        UndoLastProgressAdvanceRequest(mistaken.fact.fact_id),
        context=_context(3, 1.1),
    )
    accepted = controller.advance(
        AdvanceProgressStepRequest("attempt:b:accepted"),
        context=_context(4, 2.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(5, 3.0),
    )

    timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=store.facts,
    )

    assert [attempt.attempt_id for attempt in timeline.attempts] == [
        "attempt:a",
        "attempt:b:accepted",
    ]
    assert timeline.attempts[0].end_fact_id == accepted.fact.fact_id
    assert timeline.attempts[1].attempt_ordinal_in_capture == 2
    assert timeline.undone_advance_fact_ids == (mistaken.fact.fact_id,)
    assert timeline.invalidated_attempt_ids == ("attempt:b:mistaken",)
    assert [item.advance_fact_id for item in timeline.transitions] == [
        accepted.fact.fact_id
    ]
    store.close()


def test_failure_retry_resolves_attempts_and_intentional_pause(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a:1"),
        context=_context(1, 0.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2, 1.0),
    )
    partial = HoldPauseEffectReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        application_hold=_operation(
            "APPLICATION_HOLD",
            requested_at=1.0,
            status="REJECTED",
        ),
        formal_capture_pause=_operation(
            "FORMAL_CAPTURE_PAUSE",
            requested_at=1.0,
            effective_at=1.05,
        ),
    )
    controller.record_hold_pause_receipt(
        partial,
        context=_context(3, 1.1),
    )
    applied = HoldPauseEffectReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        application_hold=_operation(
            "APPLICATION_HOLD",
            requested_at=1.0,
            effective_at=1.08,
        ),
        formal_capture_pause=_operation(
            "FORMAL_CAPTURE_PAUSE",
            requested_at=1.0,
            effective_at=1.05,
        ),
    )
    hold_result = controller.record_hold_pause_receipt(
        applied,
        context=_context(4, 1.2),
    )
    rejected_resume = ResumeAfterFailureReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        hold_pause_receipt_fact_id=hold_result.fact.fact_id,
        resume_operation_id="resume:rejected",
        application_resume=_operation(
            "APPLICATION_RESUME",
            requested_at=2.0,
            status="REJECTED",
        ),
        formal_capture_resume=_operation(
            "FORMAL_CAPTURE_RESUME",
            requested_at=2.0,
            status="REJECTED",
        ),
    )
    controller.record_resume_receipt(
        rejected_resume,
        context=_context(5, 2.0),
    )
    applied_resume = ResumeAfterFailureReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        hold_pause_receipt_fact_id=hold_result.fact.fact_id,
        resume_operation_id="resume:applied",
        application_resume=_operation(
            "APPLICATION_RESUME",
            requested_at=2.0,
            effective_at=2.04,
        ),
        formal_capture_resume=_operation(
            "FORMAL_CAPTURE_RESUME",
            requested_at=2.0,
            effective_at=2.05,
        ),
    )
    resume_result = controller.record_resume_receipt(
        applied_resume,
        context=_context(6, 2.1),
    )
    retry = controller.resume_failed_attempt(
        ResumeFailedAttemptRequest("attempt:a:2"),
        context=_context(7, 2.2),
    )
    controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(8, 3.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(9, 4.0),
    )

    timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=store.facts,
    )

    assert [
        (item.attempt_id, item.outcome)
        for item in timeline.attempts
    ] == [
        ("attempt:a:1", "FAILURE"),
        ("attempt:a:2", "SUCCESS"),
        ("attempt:b", "SUCCESS"),
    ]
    assert timeline.attempts[0].failure_fact_id == failure.fact.fact_id
    assert timeline.attempts[1].start_fact_id == retry.fact.fact_id
    assert len(timeline.capture_pause_intervals) == 1
    pause = timeline.capture_pause_intervals[0]
    assert pause.source_failure_fact_id == failure.fact.fact_id
    assert pause.start_at == 1.05
    assert pause.end_at == 2.05
    assert pause.resume_receipt_fact_id == resume_result.fact.fact_id
    assert pause.closed
    store.close()


def test_applied_capture_pause_remains_visible_when_hold_failed_and_aborted(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2, 1.0),
    )
    controller.record_hold_pause_receipt(
        HoldPauseEffectReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            application_hold=_operation(
                "APPLICATION_HOLD",
                requested_at=1.0,
                status="TIMEOUT",
            ),
            formal_capture_pause=_operation(
                "FORMAL_CAPTURE_PAUSE",
                requested_at=1.0,
                effective_at=1.02,
            ),
        ),
        context=_context(3, 1.1),
    )
    controller.abort(
        AbortEpisodeProgressRequest("SAFETY_STOP"),
        context=_context(4, 1.2),
    )

    timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=store.facts,
    )

    assert timeline.attempts[0].outcome == "FAILURE"
    assert len(timeline.capture_pause_intervals) == 1
    assert timeline.capture_pause_intervals[0].start_at == 1.02
    assert not timeline.capture_pause_intervals[0].closed
    store.close()


def test_open_timeline_requires_manifest_only_when_requested(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    open_timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=store.facts,
    )
    assert open_timeline.phase == "ACTIVE_ATTEMPT"
    assert open_timeline.active_attempt_id == "attempt:a"
    with pytest.raises(
        ProgressTimelineResolutionError,
        match="sealed progress manifest",
    ):
        resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=store.facts,
            require_sealed=True,
        )
    store.close()


def test_rejects_advance_effective_before_attempt_start(tmp_path: Path) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 2.0),
    )
    advance = controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 3.0),
    )
    malformed = ProgressFactEnvelopeV1(
        fact_id=advance.fact.fact_id,
        episode_capture_identity_hash=(
            advance.fact.episode_capture_identity_hash
        ),
        progress_sequence_index=2,
        effective_at=1.0,
        effective_clock_domain_id="episode_clock",
        available_at=3.0,
        available_clock_domain_id="episode_clock",
        origin=advance.fact.origin,
        actor_id=advance.fact.actor_id,
        payload=advance.fact.payload,
        source_fact_ids=advance.fact.source_fact_ids,
    )
    with pytest.raises(
        ProgressTimelineResolutionError,
        match="effective before source|cannot precede",
    ):
        resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=(store.facts[0], malformed),
        )
    store.close()


def test_rejects_retry_before_formal_capture_resume(tmp_path: Path) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2, 1.0),
    )
    hold_result = controller.record_hold_pause_receipt(
        HoldPauseEffectReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            application_hold=_operation(
                "APPLICATION_HOLD",
                requested_at=1.0,
            ),
            formal_capture_pause=_operation(
                "FORMAL_CAPTURE_PAUSE",
                requested_at=1.0,
            ),
        ),
        context=_context(3, 1.1),
    )
    resume_result = controller.record_resume_receipt(
        ResumeAfterFailureReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            hold_pause_receipt_fact_id=hold_result.fact.fact_id,
            resume_operation_id="resume:late-effective",
            application_resume=_operation(
                "APPLICATION_RESUME",
                requested_at=2.0,
                effective_at=3.0,
            ),
            formal_capture_resume=_operation(
                "FORMAL_CAPTURE_RESUME",
                requested_at=2.0,
                effective_at=3.0,
            ),
        ),
        context=_context(4, 2.0, available_at=3.0),
    )
    retry_payload = AttemptRetryEventV1(
        source_failure_fact_id=failure.fact.fact_id,
        hold_pause_receipt_fact_id=hold_result.fact.fact_id,
        resume_receipt_fact_id=resume_result.fact.fact_id,
        progress_step_id="a",
        new_attempt_id="attempt:a:retry",
        attempt_ordinal_in_capture=2,
    )
    retry_fact = ProgressFactEnvelopeV1(
        fact_id="timeline:fact:5",
        episode_capture_identity_hash=binding.episode_capture_identity_hash,
        progress_sequence_index=5,
        effective_at=2.5,
        effective_clock_domain_id="episode_clock",
        available_at=3.0,
        available_clock_domain_id="episode_clock",
        origin="online_operator",
        actor_id="timeline-fixture",
        payload=retry_payload,
        source_fact_ids=(
            failure.fact.fact_id,
            hold_result.fact.fact_id,
            resume_result.fact.fact_id,
        ),
    )
    with pytest.raises(
        ProgressTimelineResolutionError,
        match="before formal capture Resume",
    ):
        resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=store.facts + (retry_fact,),
        )
    store.close()


def test_rejects_manifest_that_does_not_bind_fact_bundle(tmp_path: Path) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 1.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(3, 2.0),
    )
    seal = controller.seal(
        sealed_at=2.0,
        sealed_clock_domain_id="episode_clock",
    )
    mismatched = replace(
        seal.manifest,
        terminal_fact_id="not-the-terminal-fact",
    )
    with pytest.raises(
        ProgressTimelineResolutionError,
        match="does not bind",
    ):
        resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=store.facts,
            manifest=mismatched,
        )
    store.close()


def test_resolver_matches_online_state_at_each_failure_retry_prefix(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")

    def assert_same_state() -> None:
        timeline = resolve_progress_timeline(
            profile=profile,
            binding=binding,
            facts=store.facts,
        )
        assert timeline.phase == controller.state.phase
        assert (
            timeline.current_progress_step_id
            == controller.state.current_progress_step_id
        )
        assert timeline.active_attempt_id == controller.state.active_attempt_id
        assert timeline.fact_count == controller.state.fact_count

    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    assert_same_state()
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2, 1.0),
    )
    assert_same_state()
    hold = controller.record_hold_pause_receipt(
        HoldPauseEffectReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            application_hold=_operation(
                "APPLICATION_HOLD",
                requested_at=1.0,
            ),
            formal_capture_pause=_operation(
                "FORMAL_CAPTURE_PAUSE",
                requested_at=1.0,
            ),
        ),
        context=_context(3, 1.1),
    )
    assert_same_state()
    resume = controller.record_resume_receipt(
        ResumeAfterFailureReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            hold_pause_receipt_fact_id=hold.fact.fact_id,
            resume_operation_id="resume:prefix",
            application_resume=_operation(
                "APPLICATION_RESUME",
                requested_at=2.0,
            ),
            formal_capture_resume=_operation(
                "FORMAL_CAPTURE_RESUME",
                requested_at=2.0,
            ),
        ),
        context=_context(4, 2.0),
    )
    assert_same_state()
    controller.resume_failed_attempt(
        ResumeFailedAttemptRequest("attempt:a:retry"),
        context=_context(5, 2.1),
    )
    assert_same_state()
    controller.abort(
        AbortEpisodeProgressRequest("MANUAL_STOP"),
        context=_context(6, 3.0),
    )
    assert_same_state()
    assert resume.fact.fact_id in store.facts[-2].source_fact_ids
    store.close()


def test_resolver_supports_explicit_rooted_tree_branch_offline(
    tmp_path: Path,
) -> None:
    identity = EpisodeCaptureIdentity(
        episode_uuid="timeline-tree",
        legacy_episode_index=0,
        lifecycle_owner_id="timeline-fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="episode_clock",
    )
    profile = TaskProgressProfileV1(
        profile_id="timeline-tree",
        profile_version="1",
        task_id="timeline-tree",
        topology_kind="ROOTED_TREE",
        steps=(
            ProgressStepDefinitionV1(
                "root",
                "START",
                "Root",
                1,
                ("left", "right"),
            ),
            ProgressStepDefinitionV1("left", "END", "Left", 2),
            ProgressStepDefinitionV1("right", "END", "Right", 3),
        ),
        initial_step_id="root",
        terminal_step_ids=("left", "right"),
    )
    binding = ProgressEpisodeBindingV1(
        episode_capture_identity_hash=identity.semantic_content_hash,
        task_progress_profile_hash=profile.semantic_content_hash,
        bound_clock_domain_id="episode_clock",
        recording_scope="FULL_TASK",
        entry_step_id="root",
        entry_mode="PROFILE_INITIAL",
        history_complete=True,
    )
    store = ProgressFactStore(
        episode_dir=tmp_path / "tree",
        identity=identity,
        profile=profile,
        binding=binding,
    )
    controller = ProgressController(store=store, allow_rooted_tree=True)
    controller.start(
        StartEpisodeProgressRequest("attempt:root"),
        context=_context(1, 0.0),
    )
    controller.advance(
        AdvanceProgressStepRequest(
            "attempt:right",
            target_progress_step_id="right",
        ),
        context=_context(2, 1.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(3, 2.0),
    )

    timeline = resolve_progress_timeline(
        profile=profile,
        binding=binding,
        facts=store.facts,
    )
    assert [item.progress_step_id for item in timeline.attempts] == [
        "root",
        "right",
    ]
    assert timeline.full_task_outcome == "SUCCESS"
    store.close()


def test_reassign_failure_atomically_removes_mistaken_linear_advance(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    advance = controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 1.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(3, 2.0),
    )
    controller.abort(
        AbortEpisodeProgressRequest("MANUAL_STOP"),
        context=_context(4, 3.0),
    )
    manifest = controller.seal(
        sealed_at=4.0,
        sealed_clock_domain_id="episode_clock",
    ).manifest

    resolved = reassign_failure_to_attempt(
        profile=profile,
        binding=binding,
        facts=store.facts,
        manifest=manifest,
        correction=FailureAttemptReassignment(
            correction_id="correction:failure-owner",
            source_progress_manifest_hash=manifest.semantic_content_hash,
            source_failure_fact_id=failure.fact.fact_id,
            target_attempt_id="attempt:a",
        ),
    )

    assert [attempt.attempt_id for attempt in resolved.timeline.attempts] == [
        "attempt:a"
    ]
    corrected = resolved.timeline.attempts[0]
    assert corrected.outcome == "FAILURE"
    assert corrected.failure_fact_id == failure.fact.fact_id
    assert corrected.end_at == 2.0
    assert resolved.timeline.transitions == ()
    assert resolved.invalidated_advance_fact_ids == (advance.fact.fact_id,)
    assert resolved.invalidated_attempt_ids == ("attempt:b",)
    # Capture truth and its sealed bundle are unchanged.
    assert resolved.timeline.fact_bundle_hash == manifest.fact_bundle_hash
    assert resolved.timeline.resolution_kind == "RETROSPECTIVE_CORRECTION"
    assert len(store.facts) == 4
    store.close()


def test_reassign_failure_rejects_unbound_manifest_hash(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 1.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(3, 2.0),
    )
    controller.abort(
        AbortEpisodeProgressRequest("MANUAL_STOP"),
        context=_context(4, 3.0),
    )
    manifest = controller.seal(
        sealed_at=4.0,
        sealed_clock_domain_id="episode_clock",
    ).manifest

    with pytest.raises(
        ProgressCorrectionResolutionError,
        match="not bound",
    ):
        reassign_failure_to_attempt(
            profile=profile,
            binding=binding,
            facts=store.facts,
            manifest=manifest,
            correction=FailureAttemptReassignment(
                correction_id="correction:bad-binding",
                source_progress_manifest_hash="0" * 64,
                source_failure_fact_id=failure.fact.fact_id,
                target_attempt_id="attempt:a",
            ),
        )
    store.close()


def test_reassign_failure_with_retry_suffix_fails_closed(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2, 1.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(3, 2.0),
    )
    hold = controller.record_hold_pause_receipt(
        HoldPauseEffectReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            application_hold=_operation(
                "APPLICATION_HOLD",
                requested_at=2.0,
            ),
            formal_capture_pause=_operation(
                "FORMAL_CAPTURE_PAUSE",
                requested_at=2.0,
            ),
        ),
        context=_context(4, 2.1),
    )
    resume = controller.record_resume_receipt(
        ResumeAfterFailureReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            hold_pause_receipt_fact_id=hold.fact.fact_id,
            resume_operation_id="resume:correction-suffix",
            application_resume=_operation(
                "APPLICATION_RESUME",
                requested_at=2.2,
            ),
            formal_capture_resume=_operation(
                "FORMAL_CAPTURE_RESUME",
                requested_at=2.2,
            ),
        ),
        context=_context(5, 2.3),
    )
    controller.resume_failed_attempt(
        ResumeFailedAttemptRequest("attempt:b:retry"),
        context=_context(6, 2.4),
    )
    controller.abort(
        AbortEpisodeProgressRequest("MANUAL_STOP"),
        context=_context(7, 3.0),
    )
    manifest = controller.seal(
        sealed_at=4.0,
        sealed_clock_domain_id="episode_clock",
    ).manifest

    with pytest.raises(
        ProgressCorrectionResolutionError,
        match="later Retry or Attempt",
    ):
        reassign_failure_to_attempt(
            profile=profile,
            binding=binding,
            facts=store.facts,
            manifest=manifest,
            correction=FailureAttemptReassignment(
                correction_id="correction:needs-suffix",
                source_progress_manifest_hash=manifest.semantic_content_hash,
                source_failure_fact_id=failure.fact.fact_id,
                target_attempt_id="attempt:a",
            ),
        )
    assert resume.state.phase == "RESUME_RECEIVED"
    store.close()

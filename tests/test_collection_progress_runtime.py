from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path

import pytest

from collection_runtime.progress import (
    AbortEpisodeProgressRequest,
    AdvanceProgressStepRequest,
    CompleteEpisodeProgressRequest,
    ProgressController,
    ProgressFactStore,
    ProgressReductionError,
    ProgressRequestContext,
    ProgressStoreError,
    ProgressStoreIncompleteError,
    ReportAttemptFailureRequest,
    ResumeFailedAttemptRequest,
    StartEpisodeProgressRequest,
    UndoLastProgressAdvanceRequest,
    replay_progress_facts,
)
from collection_runtime.progress.failure_trigger_capture_store import (
    load_failure_trigger_capture_records,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_facts import (
    HoldPauseEffectReceiptV1,
    ProgressOperationReceiptV1,
    ResumeAfterFailureReceiptV1,
)
from dataset_contracts.progress_profile import (
    ProgressEpisodeBindingV1,
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)


pytestmark = [pytest.mark.integration, pytest.mark.contract]


def _context(index: int, at: float | None = None) -> ProgressRequestContext:
    timestamp = float(index if at is None else at)
    return ProgressRequestContext(
        request_id=f"fact:{index}",
        effective_at=timestamp,
        effective_clock_domain_id="episode_clock",
        available_at=timestamp,
        available_clock_domain_id="episode_clock",
        actor_id="fixture",
    )


def _operation(
    operation: str,
    *,
    requested_at: float,
    status: str = "APPLIED",
) -> ProgressOperationReceiptV1:
    return ProgressOperationReceiptV1(
        operation=operation,
        status=status,
        requested_at=requested_at,
        effective_at=requested_at if status == "APPLIED" else None,
        clock_domain_id="episode_clock",
    )


def _linear_runtime(
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
        episode_uuid="episode-progress",
        legacy_episode_index=0,
        lifecycle_owner_id="fixture",
        prepared_at=0.0,
        prepared_clock_domain_id="episode_clock",
    )
    profile = TaskProgressProfileV1(
        profile_id="linear",
        profile_version="1",
        task_id="fixture",
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
    return (
        identity,
        profile,
        binding,
        store,
        ProgressController(store=store),
    )


def test_advance_then_immediate_undo_restores_active_attempt(tmp_path: Path) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    before = controller.state
    advance = controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2),
    )
    after = controller.undo_last_advance(
        UndoLastProgressAdvanceRequest(advance.fact.fact_id),
        context=_context(3),
    ).state

    assert after.current_progress_step_id == before.current_progress_step_id
    assert after.active_attempt_id == before.active_attempt_id
    assert (
        after.latest_attempt_ordinal_in_capture
        == before.latest_attempt_ordinal_in_capture
    )
    assert after.last_effective_advance_fact_id is None
    store.close()


def test_online_undo_is_only_available_before_any_later_progress_fact(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    advance = controller.advance(
        AdvanceProgressStepRequest("attempt:b"),
        context=_context(2),
    )
    controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(3),
    )

    with pytest.raises(ProgressReductionError, match="immediately reference"):
        controller.undo_last_advance(
            UndoLastProgressAdvanceRequest(advance.fact.fact_id),
            context=_context(4),
        )
    store.close()


def test_controller_rejects_effective_time_outside_bound_episode_clock(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    context = _context(1)
    with pytest.raises(ProgressReductionError, match="bound Episode clock"):
        controller.start(
            StartEpisodeProgressRequest("attempt:a"),
            context=ProgressRequestContext(
                request_id=context.request_id,
                effective_at=context.effective_at,
                effective_clock_domain_id="foreign_clock",
                available_at=context.available_at,
                available_clock_domain_id=context.available_clock_domain_id,
                actor_id=context.actor_id,
            ),
        )
    store.close()


def test_controller_rejects_progress_effective_time_regression(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 2.0),
    )
    with pytest.raises(ProgressReductionError, match="cannot move backwards"):
        controller.advance(
            AdvanceProgressStepRequest("attempt:b"),
            context=_context(2, 1.0),
        )
    assert len(store.facts) == 1
    assert controller.state.current_progress_step_id == "a"
    store.close()


def test_store_accepts_sample_clock_distinct_from_preparation_clock(
    tmp_path: Path,
) -> None:
    identity, profile, binding, store, _ = _linear_runtime(
        tmp_path / "valid"
    )
    store.close()
    distinct_clock_store = ProgressFactStore(
        episode_dir=tmp_path / "distinct-sample-clock",
        identity=identity,
        profile=profile,
        binding=ProgressEpisodeBindingV1(
            episode_capture_identity_hash=(
                binding.episode_capture_identity_hash
            ),
            task_progress_profile_hash=(
                binding.task_progress_profile_hash
            ),
            bound_clock_domain_id="sample_wall_clock",
            recording_scope=binding.recording_scope,
            entry_step_id=binding.entry_step_id,
            entry_mode=binding.entry_mode,
            history_complete=binding.history_complete,
        ),
    )
    distinct_clock_store.close()
    with pytest.raises(TypeError, match="fsync_on_append must be boolean"):
        ProgressFactStore(
            episode_dir=tmp_path / "invalid-fsync-setting",
            identity=identity,
            profile=profile,
            binding=binding,
            fsync_on_append="false",  # type: ignore[arg-type]
        )


def test_failure_is_durable_outbox_and_replays_after_restart(tmp_path: Path) -> None:
    identity, profile, binding, store, controller = _linear_runtime(
        tmp_path / "progress"
    )
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    assert controller.state.active_attempt_id is None
    assert failure.effect_requests[0].idempotency_key == failure.fact.fact_id
    assert failure.effect_requests[0].requested_operations == (
        "APPLICATION_HOLD",
        "FORMAL_CAPTURE_PAUSE",
    )
    store.close()

    reopened = ProgressFactStore(
        episode_dir=tmp_path / "progress",
        identity=identity,
        profile=profile,
        binding=binding,
    )
    recovered = ProgressController(store=reopened)
    assert recovered.state.phase == "AWAITING_HOLD_RECEIPT"
    assert recovered.pending_effect_requests() == failure.effect_requests

    duplicate = recovered.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    assert duplicate.idempotent
    assert duplicate.effect_requests == failure.effect_requests
    reopened.close()


def test_failure_trigger_capture_defaults_on_and_can_be_disabled(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "default")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    records = load_failure_trigger_capture_records(store.episode_dir)
    assert len(records) == 1
    assert records[0].source_failure_fact_id == failure.fact.fact_id
    assert records[0].recorded_at == failure.fact.payload.reported_at
    assert records[0].clock_domain_id == failure.fact.effective_clock_domain_id
    store.close()

    *_, disabled_store, disabled_controller = _linear_runtime(
        tmp_path / "disabled"
    )
    disabled_controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    disabled_controller.report_failure(
        ReportAttemptFailureRequest(capture_failure_trigger=False),
        context=_context(2),
    )
    assert load_failure_trigger_capture_records(disabled_store.episode_dir) == ()
    disabled_store.close()


def test_partial_hold_receipt_cannot_retry(tmp_path: Path) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    partial = HoldPauseEffectReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        application_hold=_operation("APPLICATION_HOLD", requested_at=2.0),
        formal_capture_pause=_operation(
            "FORMAL_CAPTURE_PAUSE",
            requested_at=2.0,
            status="REJECTED",
        ),
    )
    receipt = controller.record_hold_pause_receipt(
        partial,
        context=_context(3),
    )

    assert receipt.state.phase == "AWAITING_HOLD_RECEIPT"
    assert receipt.effect_requests == failure.effect_requests
    with pytest.raises(ProgressReductionError, match="applied Resume"):
        controller.resume_failed_attempt(
            ResumeFailedAttemptRequest("attempt:a:retry"),
            context=_context(4),
        )
    store.close()


def test_rejects_effect_receipt_available_before_its_operation(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    receipt = HoldPauseEffectReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        application_hold=_operation(
            "APPLICATION_HOLD",
            requested_at=2.0,
        ),
        formal_capture_pause=ProgressOperationReceiptV1(
            operation="FORMAL_CAPTURE_PAUSE",
            status="APPLIED",
            requested_at=2.0,
            effective_at=2.5,
            clock_domain_id="episode_clock",
        ),
    )

    with pytest.raises(
        ProgressReductionError,
        match="after its receipt fact",
    ):
        controller.record_hold_pause_receipt(
            receipt,
            context=_context(3, 2.25),
        )

    assert controller.state.phase == "AWAITING_HOLD_RECEIPT"
    assert store.next_sequence_index == 3
    store.close()


def test_controller_rejects_formal_pause_outside_bound_episode_clock(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    receipt = HoldPauseEffectReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        application_hold=_operation(
            "APPLICATION_HOLD",
            requested_at=2.0,
        ),
        formal_capture_pause=ProgressOperationReceiptV1(
            operation="FORMAL_CAPTURE_PAUSE",
            status="REJECTED",
            requested_at=2.0,
            effective_at=None,
            clock_domain_id="foreign_clock",
        ),
    )
    with pytest.raises(ProgressReductionError, match="bound Episode clock"):
        controller.record_hold_pause_receipt(receipt, context=_context(3))
    store.close()


def test_failure_hold_resume_retry_and_full_task_seal(tmp_path: Path) -> None:
    identity, profile, binding, store, controller = _linear_runtime(
        tmp_path / "progress"
    )
    controller.start(
        StartEpisodeProgressRequest("attempt:a:1"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    hold = HoldPauseEffectReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        application_hold=_operation("APPLICATION_HOLD", requested_at=2.0),
        formal_capture_pause=_operation(
            "FORMAL_CAPTURE_PAUSE",
            requested_at=2.0,
        ),
    )
    hold_result = controller.record_hold_pause_receipt(
        hold,
        context=_context(3, 2.1),
    )
    resume = ResumeAfterFailureReceiptV1(
        source_failure_fact_id=failure.fact.fact_id,
        hold_pause_receipt_fact_id=hold_result.fact.fact_id,
        resume_operation_id="resume:1",
        application_resume=_operation(
            "APPLICATION_RESUME",
            requested_at=3.0,
        ),
        formal_capture_resume=_operation(
            "FORMAL_CAPTURE_RESUME",
            requested_at=3.0,
        ),
    )
    resume_result = controller.record_resume_receipt(
        resume,
        context=_context(4, 3.0),
    )
    assert resume_result.state.phase == "RESUME_RECEIVED"
    with pytest.raises(ProgressReductionError, match="before formal capture"):
        controller.resume_failed_attempt(
            ResumeFailedAttemptRequest("attempt:a:2"),
            context=_context(5, 2.9),
        )
    retry = controller.resume_failed_attempt(
        ResumeFailedAttemptRequest("attempt:a:2"),
        context=_context(5, 3.1),
    )
    assert retry.state.active_attempt_id == "attempt:a:2"
    controller.advance(
        AdvanceProgressStepRequest("attempt:b:1"),
        context=_context(6, 4.0),
    )
    controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(7, 5.0),
    )
    seal = controller.seal(
        sealed_at=5.0,
        sealed_clock_domain_id="episode_clock",
    )
    assert seal.manifest.fact_count == 7
    store.close()

    sealed = ProgressFactStore(
        episode_dir=tmp_path / "progress",
        identity=identity,
        profile=profile,
        binding=binding,
    )
    assert sealed.sealed
    assert sealed.replay_state().phase == "TERMINATED"
    sealed.close()


def test_failure_resume_cannot_enter_next_step_without_retry_then_advance(
    tmp_path: Path,
) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a:1"),
        context=_context(1, 0.0),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2, 1.0),
    )
    with pytest.raises(ProgressReductionError, match="active Attempt"):
        controller.advance(
            AdvanceProgressStepRequest("attempt:b:illegal"),
            context=_context(3, 1.1),
        )
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
        context=_context(4, 1.2),
    )
    controller.record_resume_receipt(
        ResumeAfterFailureReceiptV1(
            source_failure_fact_id=failure.fact.fact_id,
            hold_pause_receipt_fact_id=hold.fact.fact_id,
            resume_operation_id="resume:same-step",
            application_resume=_operation(
                "APPLICATION_RESUME",
                requested_at=2.0,
            ),
            formal_capture_resume=_operation(
                "FORMAL_CAPTURE_RESUME",
                requested_at=2.0,
            ),
        ),
        context=_context(5, 2.0),
    )
    with pytest.raises(ProgressReductionError, match="active Attempt"):
        controller.advance(
            AdvanceProgressStepRequest("attempt:b:still-illegal"),
            context=_context(6, 2.1),
        )

    retry = controller.resume_failed_attempt(
        ResumeFailedAttemptRequest("attempt:a:2"),
        context=_context(7, 2.2),
    )
    assert retry.state.current_progress_step_id == "a"
    assert retry.state.active_attempt_id == "attempt:a:2"
    advanced = controller.advance(
        AdvanceProgressStepRequest("attempt:b:1"),
        context=_context(8, 2.3),
    )
    assert advanced.state.current_progress_step_id == "b"
    store.close()


def test_abort_closes_pending_failure_without_fake_receipt(tmp_path: Path) -> None:
    *_, store, controller = _linear_runtime(tmp_path / "progress")
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    failure = controller.report_failure(
        ReportAttemptFailureRequest(),
        context=_context(2),
    )
    terminal = controller.abort(
        AbortEpisodeProgressRequest("SAFETY_STOP"),
        context=_context(3),
    )
    assert terminal.state.phase == "TERMINATED"
    assert terminal.fact.payload.unresolved_source_failure_fact_ids == (
        failure.fact.fact_id,
    )
    controller.seal(
        sealed_at=3.0,
        sealed_clock_domain_id="episode_clock",
    )
    store.close()


def test_abnormal_full_task_termination_cannot_claim_success(
    tmp_path: Path,
) -> None:
    _, profile, binding, store, controller = _linear_runtime(
        tmp_path / "progress"
    )
    start = controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1, 0.0),
    )
    terminal = controller.abort(
        AbortEpisodeProgressRequest("MANUAL_STOP"),
        context=_context(2, 1.0),
    )
    invalid_terminal = replace(
        terminal.fact,
        payload=replace(
            terminal.fact.payload,
            full_task_outcome="SUCCESS",
        ),
    )
    with pytest.raises(
        ProgressReductionError,
        match="must use FAILURE or UNKNOWN",
    ):
        replay_progress_facts(
            profile=profile,
            binding=binding,
            facts=(start.fact, invalid_terminal),
        )
    store.close()


def test_window_completion_does_not_claim_full_task_success(tmp_path: Path) -> None:
    *_, store, controller = _linear_runtime(
        tmp_path / "progress",
        scope="PROGRESS_STEP_WINDOW",
    )
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    result = controller.complete(
        CompleteEpisodeProgressRequest(),
        context=_context(2),
    )
    assert result.fact.payload.termination_cause == "WINDOW_COMPLETED"
    assert (
        result.fact.payload.active_attempt_disposition
        == "TRUNCATED_AT_CAPTURE_END"
    )
    assert result.fact.payload.full_task_outcome == "NOT_APPLICABLE"
    store.close()


def test_torn_progress_log_fails_closed(tmp_path: Path) -> None:
    identity, profile, binding, store, controller = _linear_runtime(
        tmp_path / "progress"
    )
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )
    store.close()
    fact_path = tmp_path / "progress" / "facts.cjsonl"
    with fact_path.open("ab") as handle:
        handle.write(b"{torn")

    with pytest.raises(ProgressStoreIncompleteError, match="torn final"):
        ProgressFactStore(
            episode_dir=tmp_path / "progress",
            identity=identity,
            profile=profile,
            binding=binding,
        )


def test_uncertain_append_poisons_store_until_reopened(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, profile, binding, store, controller = _linear_runtime(
        tmp_path / "progress"
    )
    controller.start(
        StartEpisodeProgressRequest("attempt:a"),
        context=_context(1),
    )

    def fail_fsync(fd: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(
        "collection_runtime.progress.store.os.fsync",
        fail_fsync,
    )
    with pytest.raises(ProgressStoreError, match="durability is uncertain"):
        controller.advance(
            AdvanceProgressStepRequest("attempt:b"),
            context=_context(2),
        )
    assert store.poisoned
    assert store.poison_reason is not None
    with pytest.raises(ProgressStoreError, match="poisoned"):
        controller.advance(
            AdvanceProgressStepRequest("attempt:b"),
            context=_context(2),
        )
    store.close()
    monkeypatch.undo()

    reopened = ProgressFactStore(
        episode_dir=tmp_path / "progress",
        identity=identity,
        profile=profile,
        binding=binding,
    )
    recovered = ProgressController(store=reopened)
    assert recovered.state.current_progress_step_id == "b"
    assert recovered.state.active_attempt_id == "attempt:b"
    assert len(reopened.facts) == 2
    reopened.close()


def test_seal_closes_fact_log_before_publishing_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, profile, binding, store, controller = _linear_runtime(
        tmp_path / "progress"
    )
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
    fact_fd = store._fd
    assert fact_fd is not None
    real_close = os.close
    failed_once = False

    def fail_fact_log_close_once(fd: int) -> None:
        nonlocal failed_once
        if fd == fact_fd and not failed_once:
            failed_once = True
            raise OSError("injected fact-log close failure")
        real_close(fd)

    monkeypatch.setattr(
        "collection_runtime.progress.store.os.close",
        fail_fact_log_close_once,
    )
    with pytest.raises(ProgressStoreError, match="durability is uncertain"):
        controller.seal(
            sealed_at=2.0,
            sealed_clock_domain_id="episode_clock",
        )
    assert store.poisoned
    assert not store.manifest_path.exists()
    store.close()
    monkeypatch.undo()

    reopened = ProgressFactStore(
        episode_dir=tmp_path / "progress",
        identity=identity,
        profile=profile,
        binding=binding,
    )
    assert reopened.replay_state().phase == "TERMINATED"
    recovered = ProgressController(store=reopened)
    recovered.seal(
        sealed_at=2.0,
        sealed_clock_domain_id="episode_clock",
    )
    assert reopened.manifest_path.is_file()
    reopened.close()


def test_rooted_tree_runtime_is_explicitly_gated(tmp_path: Path) -> None:
    identity = EpisodeCaptureIdentity(
        "tree-episode",
        0,
        "fixture",
        0.0,
        "episode_clock",
    )
    profile = TaskProgressProfileV1(
        profile_id="tree",
        profile_version="1",
        task_id="tree",
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
        identity.semantic_content_hash,
        profile.semantic_content_hash,
        "episode_clock",
        "FULL_TASK",
        "root",
        "PROFILE_INITIAL",
        True,
    )
    store = ProgressFactStore(
        episode_dir=tmp_path / "tree",
        identity=identity,
        profile=profile,
        binding=binding,
    )
    with pytest.raises(ValueError, match="ROOTED_TREE runtime is gated"):
        ProgressController(store=store)
    controller = ProgressController(store=store, allow_rooted_tree=True)
    controller.start(
        StartEpisodeProgressRequest("attempt:root"),
        context=_context(1),
    )
    with pytest.raises(ProgressReductionError, match="branch point"):
        controller.advance(
            AdvanceProgressStepRequest("attempt:left"),
            context=_context(2),
        )
    advance = controller.advance(
        AdvanceProgressStepRequest(
            "attempt:right",
            target_progress_step_id="right",
        ),
        context=_context(2),
    )
    assert controller.state.current_progress_step_id == "right"
    replay = controller.advance(
        AdvanceProgressStepRequest(
            "attempt:right",
            target_progress_step_id="right",
        ),
        context=_context(2),
    )
    assert replay.idempotent
    with pytest.raises(
        ProgressReductionError,
        match="Repeated Advance request content changed",
    ):
        controller.advance(
            AdvanceProgressStepRequest("attempt:right"),
            context=_context(2),
        )
    store.close()

from __future__ import annotations

import inspect
from collections import deque
from pathlib import Path

import pytest

from collection_runtime.sample_acceptance import AcceptedSampleReceipt
from control_station.real_teleop.progress_workflow import (
    RealTeleopProgressWorkflow,
)
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.progress_facts import (
    AttemptFailureEventV1,
    HoldPauseEffectReceiptV1,
)
from dataset_contracts.progress_profile import (
    ProgressStepDefinitionV1,
    TaskProgressProfileV1,
)
from teleop_runtime.real_teleop.control_loop import (
    _build_promotable_progress_hold_pause,
    _consume_manual_episode_stop_request,
    run_real_teleop_control_loop,
)
from teleop_runtime.real_teleop.progress_adapter import RealTeleopProgressEpisode
from teleop_runtime.real_teleop.progress_contracts import (
    ProgressMutationCommand,
    PromotableHoldPause,
)


def _profile() -> TaskProgressProfileV1:
    return TaskProgressProfileV1(
        profile_id="workflow-linear",
        profile_version="1",
        task_id="workflow-task",
        steps=(
            ProgressStepDefinitionV1(
                progress_step_id="pick",
                semantic_type="PICK",
                display_name="抓取",
                ordinal=1,
            ),
            ProgressStepDefinitionV1(
                progress_step_id="place",
                semantic_type="PLACE",
                display_name="放置",
                ordinal=2,
            ),
        ),
        initial_step_id="pick",
        terminal_step_ids=("place",),
    )


class _Recorder:
    def __init__(self, episode: RealTeleopProgressEpisode) -> None:
        self.active_progress_episode = episode
        self.is_recording = True
        self.is_paused = False
        self.receipts: deque[AcceptedSampleReceipt] = deque()

    def drain_accepted_sample_receipts(
        self,
    ) -> tuple[AcceptedSampleReceipt, ...]:
        values = tuple(self.receipts)
        self.receipts.clear()
        return values


class _DetachedRecorder:
    active_progress_episode = None

    def drain_accepted_sample_receipts(
        self,
    ) -> tuple[AcceptedSampleReceipt, ...]:
        raise AssertionError(
            "detached progress workflow must not access recorder queues"
        )


def _episode(tmp_path: Path) -> RealTeleopProgressEpisode:
    identity = EpisodeCaptureIdentity(
        episode_uuid="workflow-episode",
        legacy_episode_index=4,
        lifecycle_owner_id="fixture",
        prepared_at=1.0,
        prepared_clock_domain_id="fixture_clock",
    )
    episode = RealTeleopProgressEpisode(
        dataset_root=tmp_path,
        identity=identity,
        profile=_profile(),
    )
    episode.bind(started_at=10.0)
    return episode


def test_sample_acceptance_does_not_access_detached_recorder_queue() -> None:
    workflow = RealTeleopProgressWorkflow(recorder=_DetachedRecorder())

    result = workflow.consume_sample_acceptance(available_at=1.0)

    assert not result.handled
    assert result.phase == "ABSENT"


def test_multi_step_advance_and_fresh_online_undo(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    callbacks = {
        "request_application_hold": lambda _reason: None,
        "pause_formal_capture": lambda _now, _reason: False,
    }

    result = workflow.handle_command(
        ProgressMutationCommand("advance-1", "ADVANCE_PROGRESS"),
        now=11.0,
        **callbacks,
    )
    assert result.phase == "ACTIVE_ATTEMPT"
    assert episode.state.current_progress_step_id == "place"
    assert workflow.availability().can_undo_last_advance

    workflow.handle_command(
        ProgressMutationCommand("undo-1", "UNDO_LAST_ADVANCE"),
        now=12.0,
        **callbacks,
    )
    assert episode.state.current_progress_step_id == "pick"
    assert not workflow.availability().can_undo_last_advance


def test_failure_pause_resume_retry_uses_first_accepted_sample(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    recorder.receipts.append(
        AcceptedSampleReceipt(4, 8, 8, 12.9)
    )
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    hold = {"active": False}

    def request_hold(_reason: str) -> None:
        hold["active"] = True

    def pause_capture(_now: float, _reason: str) -> bool:
        recorder.is_paused = True
        return True

    failed = workflow.handle_command(
        ProgressMutationCommand("failure-1", "REPORT_FAILURE"),
        now=13.0,
        request_application_hold=request_hold,
        pause_formal_capture=pause_capture,
    )
    assert failed.phase == "AWAITING_HOLD_RECEIPT"
    failed = workflow.record_pending_failure_effect_receipt(
        application_hold_effective_at=13.01,
        formal_capture_pause_effective_at=13.0,
        available_at=13.02,
    )
    assert failed.phase == "PAUSED_NO_ACTIVE_ATTEMPT"
    assert recorder.receipts == deque()

    recorder.is_paused = False
    waiting_for_exact_resume = workflow.observe_applied_resume(
        now=13.9,
        application_running=True,
        formal_capture_paused=recorder.is_paused,
        formal_capture_resume_effective_at=None,
    )
    assert not waiting_for_exact_resume.handled
    assert waiting_for_exact_resume.phase == "PAUSED_NO_ACTIVE_ATTEMPT"
    resumed = workflow.observe_applied_resume(
        now=14.0,
        application_running=True,
        formal_capture_paused=recorder.is_paused,
        formal_capture_resume_effective_at=13.8,
    )
    assert resumed.phase == "RESUME_RECEIVED"

    recorder.receipts.extend(
        (
            AcceptedSampleReceipt(4, 9, 9, 13.9),
            AcceptedSampleReceipt(4, 10, 10, 14.1),
        )
    )
    # The control loop drains terminal events before handing accepted receipts
    # to the workflow.  The workflow must not try to drain the recorder again.
    terminal_receipts = tuple(recorder.drain_accepted_sample_receipts())
    retried = workflow.consume_sample_acceptance(
        available_at=14.2,
        receipts=terminal_receipts,
    )
    assert retried.handled
    assert retried.phase == "ACTIVE_ATTEMPT"
    assert episode.state.current_progress_step_id == "pick"
    assert episode.state.active_attempt_id.endswith(":retry:10")


def test_failure_while_formal_capture_is_already_paused_is_idempotent(
    tmp_path: Path,
) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    recorder.is_paused = True
    workflow = RealTeleopProgressWorkflow(recorder=recorder)

    failed = workflow.handle_command(
        ProgressMutationCommand(
            "failure-during-modality-pause",
            "REPORT_FAILURE",
        ),
        now=15.0,
        request_application_hold=lambda _reason: None,
        pause_formal_capture=lambda _now, _reason: False,
    )
    assert failed.phase == "AWAITING_HOLD_RECEIPT"

    applied = workflow.record_pending_failure_effect_receipt(
        application_hold_effective_at=15.01,
        formal_capture_pause_effective_at=14.9,
        available_at=15.02,
    )

    assert applied.phase == "PAUSED_NO_ACTIVE_ATTEMPT"
    receipt = next(
        fact
        for fact in episode.store.facts
        if isinstance(fact.payload, HoldPauseEffectReceiptV1)
    )
    assert receipt.payload.formal_capture_pause.requested_at == 15.0
    assert receipt.payload.formal_capture_pause.effective_at == 15.0
    assert receipt.payload.application_hold.effective_at == 15.01


def test_manual_episode_stop_is_consumed_while_progress_receipt_is_pending() -> None:
    class _Input:
        def __init__(self) -> None:
            self.pending = True

        def consume_episode_stop_request(self) -> bool:
            result = self.pending
            self.pending = False
            return result

    calls: list[tuple[object, ...]] = []
    consumed = _consume_manual_episode_stop_request(
        clutch_input=_Input(),
        recording_active=lambda: True,
        stop_recording=lambda now, **kwargs: calls.append(
            (now, kwargs["end_reason"], kwargs["summary"])
        ),
        cancel_recording_countdown=lambda *_args, **_kwargs: calls.append(
            ("cancel",)
        ),
        now=16.0,
        step_index=23,
        episode_last_state="RUNNING",
    )

    assert consumed
    assert calls == [
        (
            16.0,
            "manual_key_stop",
            {"last_step_index": 23, "last_state": "RUNNING"},
        )
    ]


def test_rule_completion_stop_reason_reaches_episode_finalize_owner() -> None:
    class _Input:
        def __init__(self) -> None:
            self.pending = True

        def consume_episode_stop_request_with_reason(self) -> tuple[bool, str]:
            result = self.pending
            self.pending = False
            return result, "rule_based_complete" if result else ""

    calls: list[tuple[object, ...]] = []
    consumed = _consume_manual_episode_stop_request(
        clutch_input=_Input(),
        recording_active=lambda: True,
        stop_recording=lambda now, **kwargs: calls.append(
            (now, kwargs["end_reason"], kwargs["summary"])
        ),
        cancel_recording_countdown=lambda *_args, **_kwargs: calls.append(
            ("cancel",)
        ),
        now=17.0,
        step_index=57,
        episode_last_state="RUNNING",
    )

    assert consumed
    assert calls == [
        (
            17.0,
            "rule_based_complete",
            {"last_step_index": 57, "last_state": "RUNNING"},
        )
    ]


def test_manual_episode_stop_precedes_replayed_hold_in_control_loop() -> None:
    source = inspect.getsource(run_real_teleop_control_loop)
    stop_index = source.index("if _consume_manual_episode_stop_request(")
    hold_index = source.index("if clutch_input.consume_hold_request():")

    assert stop_index < hold_index


def test_existing_manual_hold_is_promoted_without_moving_pause_cut(
    tmp_path: Path,
) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    recorder.is_paused = True
    recorder.receipts.append(AcceptedSampleReceipt(4, 8, 8, 39.9))
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    candidate = PromotableHoldPause(
        episode_capture_identity_hash=episode.identity.semantic_content_hash,
        hold_requested_at=40.0,
        formal_capture_pause_effective_at=40.0,
        application_hold_effective_at=40.01,
        hold_reason="manual",
    )
    calls = {"hold": 0, "pause_reasons": []}

    result = workflow.handle_command(
        ProgressMutationCommand("promote-hold", "REPORT_FAILURE"),
        now=40.5,
        application_held=True,
        promotable_hold_pause=candidate,
        request_application_hold=lambda _reason: calls.__setitem__(
            "hold",
            calls["hold"] + 1,
        ),
        pause_formal_capture=lambda _at, reason: (
            calls["pause_reasons"].append(reason) or True
        ),
    )

    assert result.phase == "PAUSED_NO_ACTIVE_ATTEMPT"
    assert calls["hold"] == 0
    assert calls["pause_reasons"] == [
        "progress_failure_promoted_from_hold"
    ]
    assert recorder.receipts == deque()
    failure = next(
        fact
        for fact in episode.store.facts
        if isinstance(fact.payload, AttemptFailureEventV1)
    )
    assert failure.effective_at == 40.0
    assert failure.available_at == 40.5
    assert failure.payload.reported_at == 40.0
    assert failure.payload.timing_semantics == "PROMOTED_HOLD_TIME"
    receipt = next(
        fact
        for fact in episode.store.facts
        if isinstance(fact.payload, HoldPauseEffectReceiptV1)
    )
    assert receipt.payload.application_hold.requested_at == 40.0
    assert receipt.payload.application_hold.effective_at == 40.01
    assert receipt.payload.formal_capture_pause.effective_at == 40.0
    assert workflow.availability(
        application_held=True,
        promotable_hold_pause=candidate,
    ).phase == "PAUSED_NO_ACTIVE_ATTEMPT"


def test_hold_progress_controls_fail_closed_without_same_episode_cut(
    tmp_path: Path,
) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    wrong_episode = PromotableHoldPause(
        episode_capture_identity_hash="wrong-episode",
        hold_requested_at=50.0,
        formal_capture_pause_effective_at=50.0,
        application_hold_effective_at=50.01,
        hold_reason="manual",
    )

    held = workflow.availability(
        application_held=True,
        promotable_hold_pause=wrong_episode,
    )
    assert held.application_held
    assert not held.failure_promotes_existing_hold
    with pytest.raises(RuntimeError, match="same uninterrupted Hold"):
        workflow.handle_command(
            ProgressMutationCommand("bad-promotion", "REPORT_FAILURE"),
            now=50.2,
            application_held=True,
            promotable_hold_pause=wrong_episode,
            request_application_hold=lambda _reason: None,
            pause_formal_capture=lambda _at, _reason: True,
        )
    assert episode.state.phase == "ACTIVE_ATTEMPT"


@pytest.mark.parametrize(
    ("hold_reason", "already_paused", "expected"),
    (
        ("manual", False, True),
        ("two_pedal_b", False, True),
        ("two_pedal_c", False, True),
        ("tracker_loss_timeout", False, False),
        ("position_force_guard_collision", False, False),
        ("manual", True, False),
    ),
)
def test_only_operator_hold_that_created_pause_is_promotable(
    hold_reason: str,
    already_paused: bool,
    expected: bool,
) -> None:
    candidate = _build_promotable_progress_hold_pause(
        hold_reason=hold_reason,
        application_hold_was_active=False,
        formal_capture_was_paused=already_paused,
        hold_requested_at=55.0,
        application_hold_effective_at=55.01,
        formal_capture_pause_effective_at=55.0,
        episode_capture_identity_hash="episode-hash",
        progress_phase="ACTIVE_ATTEMPT",
    )
    assert (candidate is not None) is expected


def test_promoted_failure_outbox_reuses_original_receipt_only(
    tmp_path: Path,
) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    candidate = PromotableHoldPause(
        episode_capture_identity_hash=episode.identity.semantic_content_hash,
        hold_requested_at=60.0,
        formal_capture_pause_effective_at=60.0,
        application_hold_effective_at=60.02,
        hold_reason="two_pedal_c",
    )
    episode.report_failure(
        intent_id="durable-promoted",
        effective_at=60.0,
        available_at=60.5,
        timing_semantics="PROMOTED_HOLD_TIME",
    )
    calls = {"hold": 0}

    result = workflow.request_pending_failure_effect(
        now=60.6,
        promotable_hold_pause=candidate,
        request_application_hold=lambda _reason: calls.__setitem__(
            "hold",
            calls["hold"] + 1,
        ),
        pause_formal_capture=lambda _at, _reason: True,
    )

    assert result.phase == "PAUSED_NO_ACTIVE_ATTEMPT"
    assert calls["hold"] == 0


def test_failure_outbox_replays_until_both_effects_are_applied(
    tmp_path: Path,
) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    effect_state = {"hold": False, "pause": False, "hold_requests": 0}

    def request_hold(_reason: str) -> None:
        effect_state["hold_requests"] += 1

    failed = workflow.handle_command(
        ProgressMutationCommand("failure-partial", "REPORT_FAILURE"),
        now=20.0,
        request_application_hold=request_hold,
        pause_formal_capture=lambda _now, _reason: True,
    )
    assert failed.phase == "AWAITING_HOLD_RECEIPT"
    assert len(episode.controller.pending_effect_requests()) == 1

    effect_state["hold"] = True
    effect_state["pause"] = True
    requested_again = workflow.request_pending_failure_effect(
        now=20.1,
        request_application_hold=request_hold,
        pause_formal_capture=lambda _now, _reason: True,
    )
    assert requested_again.phase == "AWAITING_HOLD_RECEIPT"
    applied = workflow.record_pending_failure_effect_receipt(
        application_hold_effective_at=20.11,
        formal_capture_pause_effective_at=20.0,
        available_at=20.12,
    )
    assert applied.phase == "PAUSED_NO_ACTIVE_ATTEMPT"
    assert effect_state["hold_requests"] == 2
    assert episode.controller.pending_effect_requests() == ()

    replay_after_receipt = workflow.request_pending_failure_effect(
        now=20.2,
        request_application_hold=request_hold,
        pause_formal_capture=lambda _now, _reason: True,
    )
    assert not replay_after_receipt.handled
    assert replay_after_receipt.phase == "PAUSED_NO_ACTIVE_ATTEMPT"


def test_retry_ignores_stale_receipt_and_rejects_wrong_episode(
    tmp_path: Path,
) -> None:
    episode = _episode(tmp_path)
    recorder = _Recorder(episode)
    workflow = RealTeleopProgressWorkflow(recorder=recorder)
    hold = {"active": False}

    def request_hold(_reason: str) -> None:
        hold["active"] = True

    workflow.handle_command(
        ProgressMutationCommand("failure-stale", "REPORT_FAILURE"),
        now=30.0,
        request_application_hold=request_hold,
        pause_formal_capture=lambda _now, _reason: True,
    )
    workflow.record_pending_failure_effect_receipt(
        application_hold_effective_at=30.01,
        formal_capture_pause_effective_at=30.0,
        available_at=30.02,
    )
    workflow.observe_applied_resume(
        now=31.0,
        application_running=True,
        formal_capture_paused=False,
        formal_capture_resume_effective_at=30.8,
    )
    recorder.receipts.append(AcceptedSampleReceipt(4, 20, 20, 30.7))

    stale = workflow.consume_sample_acceptance(available_at=31.1)
    assert not stale.handled
    assert stale.phase == "RESUME_RECEIVED"

    recorder.receipts.append(AcceptedSampleReceipt(99, 21, 21, 31.2))
    with pytest.raises(RuntimeError, match="does not belong"):
        workflow.consume_sample_acceptance(available_at=31.3)

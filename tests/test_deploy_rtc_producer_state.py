from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.deploy import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    POLICY_ASYNC_PRODUCER_ELIGIBLE,
    POLICY_ASYNC_PRODUCER_HOLD,
    POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION,
    MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1,
    RAW_RTC_FINAL_CHUNK_PROFILE_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1,
    RTC_3C_VIRTUAL_SHADOW_PROFILE_V1,
    PolicyAsyncExecutionProfile,
    PolicyChunkResult,
    PolicyInferenceRequest,
    admit_policy_chunk_result,
    admit_policy_inference_request,
    commit_emitted_policy_async_producer_action,
    create_policy_async_producer,
    empty_policy_action_plan,
    invalidate_policy_async_producer,
    pop_due_policy_async_producer_action,
    rearm_policy_async_producer,
)
from robot_motion.deploy.actions import NormalizedAction
from robot_motion.deploy.schema import ACTION_DIM
from robot_motion.position_control.config import PositionControlRates


def _action(value: float) -> NormalizedAction:
    return NormalizedAction(
        vector=np.full((ACTION_DIM,), value, dtype=np.float32),
        metadata={"source": "rtc-producer-unit"},
    )


def _plan(*, epoch: int = 1, cursor_at: float = 0.0):
    return empty_policy_action_plan(
        plan_id="plan:rtc-producer-unit",
        session_id="session:rtc-producer-unit",
        control_domain_id="domain:rtc-producer-unit",
        producer_id="producer:rtc-producer-unit",
        producer_epoch=epoch,
        policy_node_id="node:rtc-producer-unit",
        clock_domain_id="clock:rtc-producer-unit",
        cursor_at=cursor_at,
        created_at=cursor_at,
    )


def _state(*, worker_session_id: str = "worker-session:0"):
    return create_policy_async_producer(
        state_id="async-producer:unit",
        plan=_plan(),
        worker_session_id=worker_session_id,
        created_at=0.0,
    )


def _request(
    plan,
    *,
    index: int,
    admitted_at: float,
    first_target_at: float,
    horizon: int = 4,
    period: float = 0.1,
):
    return PolicyInferenceRequest(
        request_id=f"request:{index}",
        inference_id=f"inference:{index}",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id=plan.session_id,
        control_domain_id=plan.control_domain_id,
        producer_id=plan.producer_id,
        producer_epoch=plan.producer_epoch,
        policy_node_id=plan.policy_node_id,
        artifact_hash="artifact:rtc-producer-unit",
        observation_cut_id=f"observation:{index}",
        observation_timestamp=admitted_at,
        observation_step_index=index,
        triggered_at=admitted_at,
        admitted_at=admitted_at,
        first_target_effective_at=first_target_at,
        clock_domain_id=plan.clock_domain_id,
        action_period_sec=period,
        prediction_horizon_steps=horizon,
        minimum_execution_horizon_steps=1,
        source_plan_id=plan.plan_id,
        source_plan_generation=plan.generation,
        source_plan_hash=plan.semantic_content_hash,
        source_plan_cursor_at=plan.cursor_at,
        producer_plan_committed_through=plan.producer_committed_through,
        forecast_delay_steps=0,
        delay_estimator_spec_hash="delay:rtc-producer-unit",
        action_codec_hash="codec:rtc-producer-unit",
        frame_contract_hash="frame:rtc-producer-unit",
        layout_hash="layout:rtc-producer-unit",
        normalization_hash="normalization:rtc-producer-unit",
        valid_until=first_target_at + period * horizon,
    )


def _result(
    request: PolicyInferenceRequest,
    *,
    received_at: float,
    values: tuple[float, ...] = (1.0, 1.1, 1.2, 1.3),
    status: str = "ready",
) -> PolicyChunkResult:
    ready = status == "ready"
    return PolicyChunkResult(
        request_id=request.request_id,
        request_hash=request.semantic_content_hash,
        inference_id=request.inference_id,
        rtc_mode=request.rtc_mode,
        session_id=request.session_id,
        control_domain_id=request.control_domain_id,
        producer_id=request.producer_id,
        producer_epoch=request.producer_epoch,
        policy_node_id=request.policy_node_id,
        artifact_hash=request.artifact_hash,
        action_codec_hash=request.action_codec_hash,
        frame_contract_hash=request.frame_contract_hash,
        layout_hash=request.layout_hash,
        normalization_hash=request.normalization_hash,
        observation_cut_id=request.observation_cut_id,
        observation_timestamp=request.observation_timestamp,
        observation_step_index=request.observation_step_index,
        policy_output_id=f"output:{request.request_id}" if ready else None,
        chunk_id=int(request.request_id.rsplit(":", 1)[-1]) if ready else None,
        inference_started_at=request.admitted_at,
        inference_completed_at=received_at,
        result_received_at=received_at,
        clock_domain_id=request.clock_domain_id,
        status=status,
        source_chunk_size=len(values) if ready else 0,
        decoded_absolute_actions=(
            tuple(_action(value) for value in values) if ready else ()
        ),
        model_output_hash=f"model:{request.request_id}" if ready else None,
        fault_code=None if ready else "synthetic_worker_fault",
        fault_detail=None if ready else "unit",
        execution_horizon_steps=len(values) if ready else None,
    )


def _admit_ready_chunk(state, *, index: int = 1):
    request = _request(
        state.plan,
        index=index,
        admitted_at=0.0,
        first_target_at=0.2,
    )
    submitted = admit_policy_inference_request(state, request)
    assert submitted.receipt.accepted
    completed = admit_policy_chunk_result(
        submitted.state,
        _result(request, received_at=0.1),
        worker_session_id=submitted.state.worker_session_id,
        admitted_at=0.1,
        actual_cursor_at=0.1,
    )
    assert completed.receipt.accepted
    return completed.state, request


def test_rtc_3c_profile_freezes_default_off_lookahead_and_cadence() -> None:
    profile = RTC_3C_VIRTUAL_SHADOW_PROFILE_V1
    assert profile.profile_id == RTC_3C_VIRTUAL_SHADOW_PROFILE_ID_V1
    assert not profile.position_lookahead_enabled
    assert not profile.send_actions
    assert (profile.observation_hz, profile.decision_hz, profile.action_hz) == (
        15.0,
        30.0,
        30.0,
    )
    assert profile.position_rates.to_metadata()["independent_clocks"]
    assert profile.position_rates.reference_hz == 60.0
    assert profile.position_rates.command_hz == 120.0
    assert profile.semantic_content_hash == replace(profile).semantic_content_hash

    with pytest.raises(ValueError, match="lookahead disabled"):
        replace(profile, position_lookahead_enabled=True)
    with pytest.raises(ValueError, match="RTC-3c virtual profile"):
        replace(profile, send_actions=True)
    with pytest.raises(ValueError, match="cadence is fixed"):
        replace(profile, decision_hz=29.0)

    noninteger = PolicyAsyncExecutionProfile(
        profile_id="rtc-noninteger-cadence-unit-v1",
        observation_hz=17.0,
        decision_hz=29.0,
        action_hz=29.0,
        position_rates=PositionControlRates(
            sample_hz=17.0,
            reference_hz=60.0,
            ik_hz=60.0,
            command_hz=120.0,
        ),
    )
    assert noninteger.semantic_content_hash == replace(
        noninteger
    ).semantic_content_hash


@pytest.mark.parametrize(
    "profile",
    [RAW_RTC_FINAL_CHUNK_PROFILE_V1, MODEL_RTC_FLOW_INPAINT_SHADOW_PROFILE_V1],
)
def test_model_and_raw_profiles_leave_active_dispatch_to_embedding_app(
    profile: PolicyAsyncExecutionProfile,
) -> None:
    active = replace(profile, send_actions=True)

    assert active.send_actions
    assert not active.position_lookahead_enabled
    assert active.producer_profile.semantic_content_hash == (
        profile.producer_profile.semantic_content_hash
    )


def test_request_and_matching_result_share_one_atomic_state_truth() -> None:
    state = _state()
    request = _request(
        state.plan,
        index=1,
        admitted_at=0.0,
        first_target_at=0.2,
    )
    submitted = admit_policy_inference_request(state, request)
    assert submitted.receipt.accepted
    assert submitted.state.active_request is request
    assert submitted.state.plan.in_flight_request_id == request.request_id
    assert submitted.state.plan.in_flight_inference_id == request.inference_id

    duplicate_submit = admit_policy_inference_request(submitted.state, request)
    assert not duplicate_submit.receipt.accepted
    assert duplicate_submit.receipt.reasons == ("active_request_exists",)
    assert duplicate_submit.state.semantic_content_hash == (
        submitted.state.semantic_content_hash
    )

    completed = admit_policy_chunk_result(
        submitted.state,
        _result(request, received_at=0.1),
        worker_session_id="worker-session:0",
        admitted_at=0.1,
        actual_cursor_at=0.1,
    )
    assert completed.receipt.accepted
    assert completed.merge_receipt is not None
    assert completed.merge_receipt.accepted
    assert completed.state.active_request is None
    assert completed.state.plan.in_flight_request_id is None
    assert len(completed.state.plan.keyframes) == 4


def test_old_result_never_clears_a_new_active_request() -> None:
    state, old_request = _admit_ready_chunk(_state())
    new_request = _request(
        state.plan,
        index=2,
        admitted_at=0.15,
        first_target_at=0.3,
    )
    submitted = admit_policy_inference_request(state, new_request)
    assert submitted.receipt.accepted
    before = submitted.state.semantic_content_hash

    stale = admit_policy_chunk_result(
        submitted.state,
        _result(old_request, received_at=0.16),
        worker_session_id="worker-session:0",
        admitted_at=0.16,
    )
    assert not stale.receipt.accepted
    assert stale.receipt.reasons == ("result_not_active_request",)
    assert stale.state.semantic_content_hash == before
    assert stale.state.active_request is new_request
    assert stale.state.plan.in_flight_request_id == new_request.request_id


def test_matching_fault_settles_only_its_active_request() -> None:
    state = _state()
    request = _request(
        state.plan,
        index=1,
        admitted_at=0.0,
        first_target_at=0.2,
    )
    submitted = admit_policy_inference_request(state, request)
    faulted = admit_policy_chunk_result(
        submitted.state,
        _result(request, received_at=0.1, status="fault"),
        worker_session_id="worker-session:0",
        admitted_at=0.1,
        actual_cursor_at=0.1,
    )
    assert not faulted.receipt.accepted
    assert faulted.receipt.reasons == ("worker_result_fault",)
    assert faulted.merge_receipt is not None
    assert faulted.state.active_request is None
    assert faulted.state.plan.in_flight_request_id is None


def test_transition_first_rejects_old_session_without_disturbing_rearm() -> None:
    state = _state()
    old_request = _request(
        state.plan,
        index=1,
        admitted_at=0.0,
        first_target_at=0.2,
    )
    submitted = admit_policy_inference_request(state, old_request)
    invalidated = invalidate_policy_async_producer(
        submitted.state,
        lifecycle_state=POLICY_ASYNC_PRODUCER_HOLD,
        invalidated_at=0.05,
        reason="hold_before_completion",
        worker_session_state=(
            POLICY_WORKER_SESSION_RETIRED_WAITING_FOR_COMPLETION
        ),
    )
    assert invalidated.receipt.accepted
    assert invalidated.state.plan.producer_epoch == 2
    assert invalidated.state.active_request is None
    assert invalidated.state.plan.keyframes == ()

    rearmed = rearm_policy_async_producer(
        invalidated.state,
        worker_session_id="worker-session:1",
        rearmed_at=0.06,
        reason="fresh_worker_session",
    )
    assert rearmed.state.lifecycle_state == POLICY_ASYNC_PRODUCER_ELIGIBLE
    new_request = _request(
        rearmed.state.plan,
        index=2,
        admitted_at=0.07,
        first_target_at=0.2,
    )
    new_submitted = admit_policy_inference_request(rearmed.state, new_request)
    before = new_submitted.state.semantic_content_hash

    late = admit_policy_chunk_result(
        new_submitted.state,
        _result(old_request, received_at=0.08),
        worker_session_id="worker-session:0",
        admitted_at=0.08,
    )
    assert not late.receipt.accepted
    assert late.receipt.reasons == (
        "worker_session_mismatch",
        "result_not_active_request",
    )
    assert late.state.semantic_content_hash == before
    assert late.state.active_request is new_request


def test_due_pop_reports_selected_skipped_and_lateness_without_burst() -> None:
    state, _request_value = _admit_ready_chunk(_state())
    popped = pop_due_policy_async_producer_action(state, decision_at=0.45)
    assert popped.receipt.accepted
    assert popped.pop_outcome is not None
    outcome = popped.pop_outcome
    assert outcome.keyframe is not None
    assert outcome.keyframe.chunk_keyframe_index == 2
    assert outcome.discarded_expired_count == 2
    assert outcome.selected_target_effective_at == pytest.approx(0.4)
    assert outcome.oldest_skipped_target_effective_at == pytest.approx(0.2)
    assert outcome.selected_lateness_sec == pytest.approx(0.05)
    assert [item.chunk_keyframe_index for item in outcome.plan.keyframes] == [3]

    same_cut = pop_due_policy_async_producer_action(
        popped.state,
        decision_at=0.45,
    )
    assert same_cut.pop_outcome is not None
    assert same_cut.pop_outcome.keyframe is None
    assert same_cut.pop_outcome.discarded_expired_count == 0


def test_emission_commit_advances_only_through_the_popped_current_keyframe() -> None:
    state, _request_value = _admit_ready_chunk(_state())
    popped = pop_due_policy_async_producer_action(state, decision_at=0.45)
    assert popped.pop_outcome is not None
    selected = popped.pop_outcome.keyframe
    assert selected is not None

    with pytest.raises(ValueError, match="cannot be early"):
        replace(
            popped.pop_outcome,
            keyframe=popped.pop_outcome.plan.keyframes[0],
            selected_target_effective_at=0.5,
            selected_lateness_sec=0.45 - 0.5,
        )
    assert popped.state.plan.producer_committed_through is None

    committed = commit_emitted_policy_async_producer_action(
        popped.state,
        pop_outcome=popped.pop_outcome,
        emission_id="emission:selected-current",
        committed_at=0.45,
    )
    assert committed.receipt.accepted
    assert committed.receipt.emission_id == "emission:selected-current"
    assert committed.state.plan.producer_committed_through == pytest.approx(0.4)
    assert [item.chunk_keyframe_index for item in committed.state.plan.keyframes] == [3]
    assert all(
        item.target_effective_at
        > committed.state.plan.producer_committed_through
        for item in committed.state.plan.keyframes
    )

    duplicate = commit_emitted_policy_async_producer_action(
        committed.state,
        pop_outcome=popped.pop_outcome,
        emission_id="emission:duplicate",
        committed_at=0.46,
    )
    assert not duplicate.receipt.accepted
    assert duplicate.receipt.reasons == (
        "pop_state_mismatch",
        "commit_through_not_advancing",
    )


def test_cursor_can_advance_while_request_is_in_flight_before_matching_merge() -> None:
    state, _first_request = _admit_ready_chunk(_state())
    request = _request(
        state.plan,
        index=2,
        admitted_at=0.15,
        first_target_at=0.4,
    )
    submitted = admit_policy_inference_request(state, request)
    advanced = pop_due_policy_async_producer_action(
        submitted.state,
        decision_at=0.25,
    )
    assert advanced.pop_outcome is not None
    assert advanced.pop_outcome.keyframe is not None
    assert advanced.state.active_request is request
    assert advanced.state.plan.cursor_at == 0.25

    completed = admit_policy_chunk_result(
        advanced.state,
        _result(request, received_at=0.3),
        worker_session_id="worker-session:0",
        admitted_at=0.3,
        actual_cursor_at=0.3,
    )
    assert completed.receipt.accepted
    assert completed.state.active_request is None
    assert all(
        keyframe.target_effective_at >= 0.3
        for keyframe in completed.state.plan.keyframes
    )

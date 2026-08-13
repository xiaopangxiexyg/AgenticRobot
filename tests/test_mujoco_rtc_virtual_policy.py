from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_motion.action import (
    IMMEDIATE_QREF_SIM_V1,
    SCHEDULED_REFERENCE_SEGMENT_SIM_V1,
)
from sim.contracts import TimingSpec
from sim.mujoco.adapters import (
    MujocoPositionExecutionConfig,
    MujocoPositionExecutionRig,
)
from sim.mujoco.deploy import (
    DeterministicVirtualPolicyRtcRunner,
    MujocoDeployConfig,
    MujocoDeployRuntime,
    VirtualBimanualQposPolicy,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


class FakeScene:
    arm_specs = {"left": object(), "right": object()}
    gripper_specs = {}

    def __init__(self) -> None:
        self.qpos = {
            "left": np.zeros(6, dtype=float),
            "right": np.zeros(6, dtype=float),
        }
        self.physics_commands: list[dict[str, np.ndarray]] = []

    def get_arm_qpos(self, side: str) -> np.ndarray:
        return self.qpos[side].copy()

    def step_kinematic_direct_qpos_debug(
        self,
        qpos_by_side,
        *,
        gripper_close_by_name,
        zero_controlled_qvel,
    ) -> None:
        del gripper_close_by_name, zero_controlled_qvel
        command = {
            side: np.asarray(qpos, dtype=float).reshape(6).copy()
            for side, qpos in qpos_by_side.items()
        }
        self.physics_commands.append(command)
        self.qpos = {side: value.copy() for side, value in command.items()}


def _build_runner(
    *,
    scheduled_reference: bool = False,
    decision_hz: float = 30.0,
):
    clock = ManualClock()
    scene = FakeScene()
    rig = MujocoPositionExecutionRig(
        scene,
        MujocoPositionExecutionConfig(
            timing=TimingSpec(
                sample_hz=15.0,
                policy_hz=15.0,
                reference_hz=60.0,
                ik_hz=60.0,
                command_hz=120.0,
                physics_hz=1000.0,
            ),
            actuation_profile=None,
            gripper_actuation_profile=None,
            realtime_worker=False,
            follow_qpos_alpha=1.0,
        ),
        monotonic_clock=clock,
    )
    runtime = MujocoDeployRuntime(
        rig,
        MujocoDeployConfig(
            runtime_session_id="virtual-rtc-session",
            control_domain_id="virtual-rtc-domain",
            embodiment_instance_id="virtual-rtc-embodiment",
            decision_hz=decision_hz,
            reference_activation_profile_id=(
                SCHEDULED_REFERENCE_SEGMENT_SIM_V1
                if scheduled_reference
                else IMMEDIATE_QREF_SIM_V1
            ),
        ),
        clock=clock,
    )
    runtime.start()
    runner = DeterministicVirtualPolicyRtcRunner(
        runtime,
        policy=VirtualBimanualQposPolicy(),
        set_control_time=lambda value: setattr(clock, "value", value),
        start_at=clock.value,
    )
    return runner, runtime, scene, clock


def _selected_left_qpos(decision) -> np.ndarray:
    selected = decision.authoritative_step.decision
    assert selected is not None
    return np.asarray(
        selected.selected_action.value("left_arm").values,
        dtype=float,
    )


def test_zero_latency_virtual_policy_reaches_existing_authoritative_sim_sink() -> None:
    runner, runtime, scene, clock = _build_runner()
    period = runner.action_period_sec
    submission = runner.schedule_prediction(
        requested_at=clock.value,
        inference_completed_at=clock.value,
        prediction_horizon_steps=3,
        forecast_delay_steps=0,
    )
    assert submission.accepted

    first = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert [item.status for item in first.completion_releases] == ["admitted"]
    assert first.emitted_keyframe is not None
    assert first.emitted_keyframe.chunk_keyframe_index == 0
    assert first.proposal_receipt is not None and first.proposal_receipt.accepted
    assert first.proposal_receipt.proposal is not None
    assert first.proposal_receipt.proposal.chunk_keyframe_index == 0
    assert first.proposal_receipt.proposal.inference_id == (
        first.emitted_keyframe.inference_id
    )
    assert first.authoritative_step.status == "dispatched"
    np.testing.assert_allclose(_selected_left_qpos(first), np.full(6, 0.01), atol=1e-7)

    second = runner.step(
        decision_at=10.0 + period,
        physics_ticks_after_dispatch=1,
    )
    assert second.emitted_keyframe is not None
    assert second.emitted_keyframe.chunk_keyframe_index == 1
    assert second.proposal_receipt is not None
    assert second.proposal_receipt.proposal is not None
    assert second.proposal_receipt.proposal.chunk_keyframe_index == 1
    assert second.authoritative_step.status == "dispatched"
    np.testing.assert_allclose(_selected_left_qpos(second), np.full(6, 0.02), atol=1e-7)
    assert runner.emission_count == 2
    assert len(scene.physics_commands) == 2
    assert sum(fact.event_type == "action_proposal" for fact in runtime.trace_facts) == 2
    event_types = {fact.event_type for fact in runtime.trace_facts}
    assert {"command", "dispatch_receipt", "measurement_alignment"}.issubset(
        event_types
    )
    runtime.close()


@pytest.mark.parametrize("delay_steps", (1, 2, 3, 4))
def test_fixed_latency_has_no_synchronous_fallback_before_forecast_target(
    delay_steps: int,
) -> None:
    runner, runtime, scene, _clock = _build_runner()
    period = runner.action_period_sec
    received_at = 10.0 + delay_steps * period
    submission = runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=received_at,
        prediction_horizon_steps=delay_steps + 1,
        forecast_delay_steps=delay_steps,
    )
    assert submission.accepted
    assert submission.request is not None
    assert submission.request.first_target_effective_at == pytest.approx(received_at)

    for index in range(delay_steps):
        waiting = runner.step(
            decision_at=10.0 + index * period,
            physics_ticks_after_dispatch=1,
        )
        assert waiting.emitted_keyframe is None and waiting.underrun
    assert runner.emission_count == 0
    assert not scene.physics_commands

    ready = runner.step(decision_at=received_at, physics_ticks_after_dispatch=1)
    assert ready.completion_releases[0].status == "admitted"
    assert ready.emitted_keyframe is not None
    assert ready.emitted_keyframe.target_effective_at == pytest.approx(received_at)
    assert ready.authoritative_step.status == "dispatched"
    assert runner.emission_count == 1
    runtime.close()


def test_overforecast_result_does_not_pull_execution_before_target_time() -> None:
    runner, runtime, scene, _clock = _build_runner()
    period = runner.action_period_sec
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0 + period,
        result_received_at=10.0 + period,
        prediction_horizon_steps=4,
        forecast_delay_steps=3,
    )
    runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    early = runner.step(
        decision_at=10.0 + period,
        physics_ticks_after_dispatch=1,
    )
    assert early.completion_releases[0].status == "admitted"
    assert early.emitted_keyframe is None
    assert runner.pending_result_count == 0
    assert runner.emission_count == 0
    before_target = runner.step(
        decision_at=10.0 + 2 * period,
        physics_ticks_after_dispatch=1,
    )
    assert before_target.emitted_keyframe is None

    target = runner.step(
        decision_at=10.0 + 3 * period,
        physics_ticks_after_dispatch=1,
    )
    assert target.emitted_keyframe is not None
    assert target.emitted_keyframe.chunk_keyframe_index == 0
    assert runner.emission_count == 1
    assert len(scene.physics_commands) == 1
    runtime.close()


def test_underforecast_late_result_crops_obsolete_prefix_then_emits_once() -> None:
    runner, runtime, scene, _clock = _build_runner()
    period = runner.action_period_sec
    received_at = 10.0 + 3 * period
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=received_at,
        prediction_horizon_steps=4,
        forecast_delay_steps=1,
    )
    for index in range(3):
        decision = runner.step(
            decision_at=10.0 + index * period,
            physics_ticks_after_dispatch=1,
        )
        assert decision.emitted_keyframe is None

    late = runner.step(decision_at=received_at, physics_ticks_after_dispatch=1)
    release = late.completion_releases[0]
    assert release.status == "admitted"
    assert release.merge_receipt.discarded_obsolete_count == 2
    assert release.merge_receipt.admitted_keyframe_count == 2
    assert late.emitted_keyframe is not None
    assert late.emitted_keyframe.chunk_keyframe_index == 2
    assert runner.emission_count == 1
    assert len(scene.physics_commands) == 1
    assert sum(fact.event_type == "action_proposal" for fact in runtime.trace_facts) == 1
    np.testing.assert_allclose(_selected_left_qpos(late), np.full(6, 0.03), atol=1e-7)
    runtime.close()


def test_result_at_decision_cutoff_is_admitted_but_just_after_waits_for_next_tick() -> None:
    at_cutoff_runner, at_cutoff_runtime, _scene, _clock = _build_runner()
    period = at_cutoff_runner.action_period_sec
    cutoff = 10.0 + period
    at_cutoff_runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=cutoff,
        result_received_at=cutoff,
        prediction_horizon_steps=2,
        forecast_delay_steps=1,
    )
    admitted = at_cutoff_runner.step(
        decision_at=cutoff,
        physics_ticks_after_dispatch=1,
    )
    assert [item.status for item in admitted.completion_releases] == ["admitted"]
    assert admitted.emitted_keyframe is not None
    assert admitted.emitted_keyframe.chunk_keyframe_index == 0
    at_cutoff_runtime.close()

    after_runner, after_runtime, _scene, _clock = _build_runner()
    epsilon = period * 1e-6
    after_runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=cutoff + epsilon,
        result_received_at=cutoff + epsilon,
        prediction_horizon_steps=3,
        forecast_delay_steps=1,
    )
    before_release = after_runner.step(
        decision_at=cutoff,
        physics_ticks_after_dispatch=1,
    )
    assert before_release.completion_releases == ()
    assert before_release.emitted_keyframe is None
    assert after_runner.pending_result_count == 1

    next_tick = after_runner.step(
        decision_at=10.0 + 2 * period,
        physics_ticks_after_dispatch=1,
    )
    assert [item.status for item in next_tick.completion_releases] == ["admitted"]
    assert next_tick.completion_releases[0].merge_receipt.discarded_obsolete_count == 1
    assert next_tick.emitted_keyframe is not None
    assert next_tick.emitted_keyframe.chunk_keyframe_index == 1
    after_runtime.close()


def test_zero_execution_suffix_is_rejected_without_proposal_or_dispatch() -> None:
    runner, runtime, scene, _clock = _build_runner()
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=2,
        forecast_delay_steps=0,
        execution_horizon_steps=0,
    )
    decision = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert decision.completion_releases[0].status == "rejected"
    assert decision.completion_releases[0].merge_receipt.reasons == (
        "no_eligible_future",
    )
    assert decision.emitted_keyframe is None
    assert decision.proposal_receipt is None
    assert decision.underrun
    assert runner.emission_count == 0
    assert not scene.physics_commands
    runtime.close()


def test_non_integer_cadence_skips_missed_keyframes_without_catchup_burst() -> None:
    runner, runtime, scene, _clock = _build_runner(decision_hz=29.0)
    period = runner.action_period_sec
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=5,
        forecast_delay_steps=0,
    )
    first = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert first.emitted_keyframe is not None
    assert first.emitted_keyframe.chunk_keyframe_index == 0

    late_time = 10.0 + 3.2 * period
    late = runner.step(decision_at=late_time, physics_ticks_after_dispatch=1)
    assert late.emitted_keyframe is not None
    assert late.emitted_keyframe.chunk_keyframe_index == 3
    assert late.plan_pop.discarded_expired_count == 2
    assert late.plan_pop.selected_target_effective_at == pytest.approx(
        10.0 + 3.0 * period
    )
    assert late.plan_pop.oldest_skipped_target_effective_at == pytest.approx(
        10.0 + period
    )
    assert late.plan_pop.selected_lateness_sec == pytest.approx(0.2 * period)
    np.testing.assert_allclose(_selected_left_qpos(late), np.full(6, 0.04), atol=1e-7)

    repeated_cut = runner.step(
        decision_at=late_time,
        physics_ticks_after_dispatch=1,
    )
    assert repeated_cut.emitted_keyframe is None
    assert repeated_cut.authoritative_step.status == "not_due"
    assert runner.emission_count == 2
    assert len(scene.physics_commands) == 2
    assert sum(fact.event_type == "action_proposal" for fact in runtime.trace_facts) == 2
    runtime.close()


def test_duplicate_virtual_result_is_rejected_without_a_second_emission() -> None:
    runner, runtime, scene, _clock = _build_runner()
    submission = runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=2,
        forecast_delay_steps=0,
    )
    assert submission.request is not None and submission.result is not None
    assert runner.schedule_result_delivery(
        request=submission.request,
        result=submission.result,
        deliver_at=10.0,
        delivery_id="duplicate-delivery",
    )

    decision = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert [release.status for release in decision.completion_releases] == [
        "admitted",
        "rejected",
    ]
    assert decision.completion_releases[1].merge_receipt.reasons == (
        "request_not_in_flight",
    )
    assert runner.emission_count == 1
    assert len(scene.physics_commands) == 1
    runtime.close()


def test_virtual_policy_fault_stays_an_underrun_and_never_reaches_proposal_sink() -> None:
    runner, runtime, scene, _clock = _build_runner()
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=2,
        forecast_delay_steps=0,
        fault_code="synthetic_model_fault",
    )
    decision = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert decision.completion_releases[0].status == "rejected"
    assert decision.completion_releases[0].merge_receipt.reasons == (
        "worker_result_fault",
    )
    assert decision.emitted_keyframe is None
    assert decision.proposal_receipt is None
    assert decision.underrun
    assert not scene.physics_commands
    runtime.close()


def test_hold_active_at_completion_blocks_virtual_policy_dispatch() -> None:
    runner, runtime, scene, _clock = _build_runner()
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=2,
        forecast_delay_steps=0,
    )
    runtime.set_execution_gate(
        "hold",
        reason="virtual_rtc_completion_hold",
        source_event_ids=("virtual-hold",),
    )
    decision = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert decision.completion_releases[0].status == "admitted"
    assert decision.emitted_keyframe is not None
    assert decision.proposal_receipt is not None and decision.proposal_receipt.accepted
    assert decision.authoritative_step.status == "execution_hold_active"
    assert decision.authoritative_step.decision is None
    assert not scene.physics_commands
    runtime.close()


def test_wrong_epoch_delivery_cannot_emit_and_obsoletes_the_later_valid_result() -> None:
    runner, runtime, scene, _clock = _build_runner()
    period = runner.action_period_sec
    submission = runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0 + period,
        result_received_at=10.0 + period,
        prediction_horizon_steps=2,
        forecast_delay_steps=0,
    )
    assert submission.request is not None and submission.result is not None
    wrong_epoch = replace(
        submission.result,
        producer_epoch=submission.result.producer_epoch + 1,
        inference_completed_at=10.0,
        result_received_at=10.0,
    )
    assert runner.schedule_result_delivery(
        request=submission.request,
        result=wrong_epoch,
        deliver_at=10.0,
        delivery_id="wrong-epoch-delivery",
    )

    rejected = runner.step(decision_at=10.0, physics_ticks_after_dispatch=1)
    assert rejected.completion_releases[0].status == "rejected"
    assert "producer_epoch_mismatch" in (
        rejected.completion_releases[0].merge_receipt.reasons
    )
    assert rejected.emitted_keyframe is None

    obsolete = runner.step(
        decision_at=10.0 + period,
        physics_ticks_after_dispatch=1,
    )
    assert obsolete.completion_releases[0].status == "rejected"
    assert obsolete.completion_releases[0].merge_receipt.reasons == (
        "request_not_in_flight",
    )
    assert runner.emission_count == 0
    assert not scene.physics_commands
    runtime.close()


def test_virtual_policy_reaches_scheduled_reference_generation_and_measurement_trace() -> None:
    runner, runtime, scene, _clock = _build_runner(scheduled_reference=True)
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=1,
        forecast_delay_steps=0,
    )
    decision = runner.step(decision_at=10.0)
    assert decision.authoritative_step.status == "dispatched"
    assert runtime.reference_receiver is not None
    assert len(runtime.reference_receiver.segments) == 1
    segment = runtime.reference_receiver.segments[0]
    assert segment.group_segment_generation == 1
    assert runtime.core.pending_execution_segment_by_group["arms"].group_segment_generation == 1
    assert runtime.core.active_execution_segment_by_group == {}

    for _ in range(18):
        runtime.rig.step_physics()
    assert runtime.core.active_execution_segment_by_group["arms"].group_segment_generation == 1
    assert runtime.core.pending_execution_segment_by_group == {}
    assert scene.physics_commands
    runtime.close()
    event_types = {fact.event_type for fact in runtime.trace_facts}
    assert {
        "reference_segment",
        "reference_segment_admission",
        "reference_target_activation",
        "planner_command_frame",
    }.issubset(event_types)


def test_virtual_rtc_successor_commits_endpoint_and_traces_qcmd_measurement() -> None:
    runner, runtime, scene, _clock = _build_runner(scheduled_reference=True)
    period = runner.action_period_sec
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=2,
        forecast_delay_steps=0,
    )

    first_decision = runner.step(
        decision_at=10.0,
        physics_ticks_after_dispatch=18,
    )
    assert first_decision.authoritative_step.status == "dispatched"
    assert runtime.reference_receiver is not None
    assert runtime.core.active_execution_segment_by_group[
        "arms"
    ].group_segment_generation == 1

    second_decision = runner.step(
        decision_at=10.0 + period,
        physics_ticks_after_dispatch=34,
    )
    assert second_decision.authoritative_step.status == "dispatched"
    assert second_decision.emitted_keyframe is not None
    assert second_decision.emitted_keyframe.chunk_keyframe_index == 1
    first_segment, second_segment = runtime.reference_receiver.segments
    assert first_segment.group_segment_generation == 1
    assert second_segment.group_segment_generation == 2
    assert second_segment.start_parent.source_kind == "committed_safe_endpoint"
    assert second_segment.start_parent.source_fact_id == (
        first_segment.reference_segment_id
    )
    assert second_segment.scheduled_from == pytest.approx(first_segment.scheduled_until)
    assert runtime.reference_receiver.generation_activations[-1].active_generation == 2
    assert runtime.core.active_execution_segment_by_group[
        "arms"
    ].group_segment_generation == 2
    assert runtime.core.pending_execution_segment_by_group == {}
    first_terminal = next(
        item
        for item in runtime.reference_receiver.terminals
        if item.reference_segment_id == first_segment.reference_segment_id
    )
    assert first_terminal.status == "retired_by_successor_activation"

    planner_frames = runtime.rig.planner_command_frame_observations
    assert {item.group_segment_generation for item in planner_frames} == {1, 2}
    assert all(
        set(item.planner_qcmd_by_side) == {"left", "right"}
        and set(item.accepted_qcmd_by_side) == {"left", "right"}
        for item in planner_frames
    )
    measurement_facts = tuple(
        fact for fact in runtime.trace_facts if fact.event_type == "measurement_alignment"
    )
    assert len(measurement_facts) == 2
    final_measurement = measurement_facts[-1].payload["measurement"]
    np.testing.assert_allclose(
        final_measurement["measured_qpos_by_side"]["left"],
        scene.qpos["left"],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        final_measurement["accepted_qcmd_by_side"]["left"],
        scene.qpos["left"],
        atol=1e-12,
    )
    assert final_measurement["physics_ticks_after_dispatch"] == 34
    assert runner.emission_count == 2
    runtime.close()


def test_hold_invalidates_plan_and_rejects_retained_old_epoch_completion() -> None:
    runner, runtime, scene, _clock = _build_runner(scheduled_reference=True)
    period = runner.action_period_sec
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0,
        prediction_horizon_steps=3,
        forecast_delay_steps=0,
    )
    first = runner.step(decision_at=10.0, physics_ticks_after_dispatch=18)
    assert first.authoritative_step.status == "dispatched"
    assert len(runner.plan.keyframes) == 2

    pending = runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0 + 2 * period,
        result_received_at=10.0 + 2 * period,
        prediction_horizon_steps=2,
        forecast_delay_steps=2,
    )
    assert pending.accepted
    assert runner.pending_result_count == 1
    old_epoch = runner.plan.producer_epoch
    old_emission_count = runner.emission_count
    old_segment_count = len(runtime.reference_receiver.segments)

    invalidation = runner.invalidate_for_execution_gate(
        "hold",
        occurred_at=10.0 + 0.5 * period,
        reason="virtual RTC lifecycle stop",
        source_event_ids=("virtual-rtc-stop",),
    )
    assert invalidation.old_producer_epoch == old_epoch
    assert invalidation.new_producer_epoch == old_epoch + 1
    assert invalidation.discarded_keyframe_count == 2
    assert invalidation.discarded_in_flight_request
    assert invalidation.retained_pending_result_count == 1
    assert invalidation.old_plan_hash != invalidation.new_plan_hash
    assert runner.lifecycle_invalidated
    assert runner.plan.producer_epoch == old_epoch + 1
    assert runner.plan.keyframes == ()
    assert runner.plan.in_flight_request_id is None

    rejected_restart = runner.schedule_prediction(
        requested_at=10.0 + 0.5 * period,
        inference_completed_at=10.0 + 0.5 * period,
    )
    assert not rejected_restart.accepted
    assert rejected_restart.rejection_code == "lifecycle_invalidated"

    late = runner.step(
        decision_at=10.0 + 2 * period,
        physics_ticks_after_dispatch=1,
    )
    assert len(late.completion_releases) == 1
    release = late.completion_releases[0]
    assert release.status == "rejected"
    assert release.merge_receipt.reasons == (
        "producer_epoch_mismatch",
        "request_not_in_flight",
    )
    assert late.emitted_keyframe is None
    assert late.proposal_receipt is None
    assert late.authoritative_step.status == "execution_hold_active"
    assert runner.emission_count == old_emission_count
    assert len(runtime.reference_receiver.segments) == old_segment_count
    assert len(scene.physics_commands) == 18
    first_segment = runtime.reference_receiver.segments[0]
    terminal = next(
        item
        for item in runtime.reference_receiver.terminals
        if item.reference_segment_id == first_segment.reference_segment_id
    )
    assert terminal.status == "canceled"
    runtime.close()
    event_types = tuple(fact.event_type for fact in runtime.trace_facts)
    assert "execution_gate_event" in event_types
    assert "producer_epoch_activation" in event_types
    assert "execution_cancellation" in event_types
    assert "reference_segment_terminal" in event_types


def _run_jitter_fixture() -> tuple[
    tuple[str, ...],
    tuple[float, ...],
    tuple[str, ...],
]:
    runner, runtime, scene, _clock = _build_runner()
    period = runner.action_period_sec
    runner.schedule_prediction(
        requested_at=10.0,
        inference_completed_at=10.0 + 1.5 * period,
        result_received_at=10.0 + 1.5 * period,
        prediction_horizon_steps=3,
        forecast_delay_steps=2,
    )
    decisions = [
        runner.step(decision_at=10.0, physics_ticks_after_dispatch=1),
        runner.step(decision_at=10.0 + period, physics_ticks_after_dispatch=1),
        runner.step(decision_at=10.0 + 2 * period, physics_ticks_after_dispatch=1),
    ]
    runner.schedule_prediction(
        requested_at=10.0 + 2 * period,
        inference_completed_at=10.0 + 3.2 * period,
        result_received_at=10.0 + 3.2 * period,
        prediction_horizon_steps=3,
        forecast_delay_steps=1,
    )
    decisions.extend(
        (
            runner.step(
                decision_at=10.0 + 3 * period,
                physics_ticks_after_dispatch=1,
            ),
            runner.step(
                decision_at=10.0 + 4 * period,
                physics_ticks_after_dispatch=1,
            ),
        )
    )
    hashes = tuple(item.semantic_content_hash for item in decisions)
    qpos = tuple(
        float(_selected_left_qpos(item)[0])
        for item in decisions
        if item.emitted_keyframe is not None
    )
    assert len(scene.physics_commands) == len(qpos)
    runtime.close()
    trace_hashes = tuple(fact.semantic_content_hash for fact in runtime.trace_facts)
    return hashes, qpos, trace_hashes


def test_jitter_fixture_replays_identical_merge_decision_and_sim_results() -> None:
    first_hashes, first_qpos, first_trace_hashes = _run_jitter_fixture()
    second_hashes, second_qpos, second_trace_hashes = _run_jitter_fixture()
    assert second_hashes == first_hashes
    assert second_qpos == first_qpos
    assert second_trace_hashes == first_trace_hashes
    assert first_qpos == pytest.approx((0.01, 0.02, 0.02))

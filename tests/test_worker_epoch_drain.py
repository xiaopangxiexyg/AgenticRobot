from __future__ import annotations

import pytest

from robot_motion.position_control import ExecutionCapabilityProfile
from robot_motion.runtime import (
    CancellationAcknowledgementTracker,
    ExecutionCancellationParticipantSpec,
    ExecutionCancellationRequest,
    ExecutionEpochRegistry,
    WorkerEpochDrainReceipt,
    acknowledge_worker_epoch_drain,
    aggregate_worker_epoch_drains,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value


def _capability() -> ExecutionCapabilityProfile:
    return ExecutionCapabilityProfile(
        profile_id="worker-drain-fixture-v1",
        validated_max_selected_keyframe_hz=60.0,
        configured_selected_keyframe_hz=30.0,
        validated_max_coupled_ik_hz=None,
        configured_coupled_ik_hz=None,
        reference_resample_hz=60.0,
        command_hz=120.0,
        backend_servo_hz=1000.0,
        validated_action_spaces=("bimanual_qpos_v1",),
        validated_selection_groups=("arms",),
        supports_local_group_stop=False,
        acceptance_evidence_id="worker-drain-fixture-evidence-v1",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        cancellation_ack_deadline_sec=0.1,
        required_cancellation_ack_levels=("software_queue",),
    )


def _cancellation(clock: ManualClock):
    registry = ExecutionEpochRegistry(
        runtime_session_id="session-worker-drain",
        control_domain_id="rm65-arms",
        clock_domain_id="control-monotonic",
        initial_execution_epoch=3,
        initial_segment_generation_by_group={"arms": 7},
        clock=clock,
    )
    return registry.cancel(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=2,
            requested_scope="control_domain",
            invalidated_reference_segment_ids=("segment-old",),
            invalidated_queued_command_ids=("qcmd-old",),
            source_event_ids=("operator-hold",),
        ),
        capability=_capability(),
    )


def _participants():
    return (
        ExecutionCancellationParticipantSpec.create(
            participant_id="position-command-worker",
            worker_kind="position_command_worker",
            affected_selection_group_ids=("arms",),
        ),
        ExecutionCancellationParticipantSpec.create(
            participant_id="rm65-background-stream",
            worker_kind="backend_stream_worker",
            affected_selection_group_ids=("arms",),
        ),
    )


def _drained(cancellation, participant, *, acknowledged_at: float):
    return WorkerEpochDrainReceipt.evaluate(
        cancellation=cancellation,
        participant=participant,
        observed_execution_epoch=cancellation.execution_epoch,
        discarded_work_ids=(f"old-work:{participant.participant_id}",),
        last_old_epoch_invocation_started_at=acknowledged_at - 0.002,
        last_old_epoch_invocation_completed_at=acknowledged_at - 0.001,
        in_flight_old_epoch_count=0,
        no_future_old_epoch_calls=True,
        acknowledged_at=acknowledged_at,
    )


def test_all_required_workers_must_drain_before_software_queue_ack() -> None:
    clock = ManualClock()
    cancellation = _cancellation(clock)
    participants = _participants()
    clock.value = cancellation.effective_at + 0.01
    receipts = tuple(
        _drained(cancellation, participant, acknowledged_at=clock.value)
        for participant in participants
    )
    aggregate = aggregate_worker_epoch_drains(
        cancellation=cancellation,
        participants=participants,
        receipts=receipts,
        aggregated_at=clock.value,
    )
    tracker = CancellationAcknowledgementTracker(clock=clock)
    tracker.register(cancellation)
    acknowledgement = acknowledge_worker_epoch_drain(
        aggregate,
        tracker=tracker,
        clock_domain_id="control-monotonic",
    )

    assert aggregate.complete
    assert aggregate.accepted_participant_ids == (
        "position-command-worker",
        "rm65-background-stream",
    )
    assert acknowledgement.acknowledgement_level == "software_queue"
    assert acknowledgement.status == "all_required_workers_drained"
    assert tracker.receipt(cancellation.cancellation_id).complete


def test_queue_clear_without_in_flight_drain_is_rejected() -> None:
    clock = ManualClock()
    cancellation = _cancellation(clock)
    participant = _participants()[0]
    clock.value = cancellation.effective_at + 0.01
    receipt = WorkerEpochDrainReceipt.evaluate(
        cancellation=cancellation,
        participant=participant,
        observed_execution_epoch=cancellation.execution_epoch,
        discarded_work_ids=("queued-old",),
        last_old_epoch_invocation_started_at=clock.value - 0.001,
        last_old_epoch_invocation_completed_at=None,
        in_flight_old_epoch_count=1,
        no_future_old_epoch_calls=False,
        acknowledged_at=clock.value,
    )

    assert not receipt.accepted
    assert receipt.status == "old_epoch_invocations_in_flight"


def test_missing_participant_cannot_project_software_queue_ack() -> None:
    clock = ManualClock()
    cancellation = _cancellation(clock)
    participants = _participants()
    clock.value = cancellation.effective_at + 0.01
    aggregate = aggregate_worker_epoch_drains(
        cancellation=cancellation,
        participants=participants,
        receipts=(
            _drained(cancellation, participants[0], acknowledged_at=clock.value),
        ),
        aggregated_at=clock.value,
    )
    tracker = CancellationAcknowledgementTracker(clock=clock)
    tracker.register(cancellation)

    assert not aggregate.complete
    assert aggregate.missing_participant_ids == ("rm65-background-stream",)
    with pytest.raises(RuntimeError, match="Incomplete worker drain"):
        acknowledge_worker_epoch_drain(
            aggregate,
            tracker=tracker,
            clock_domain_id="control-monotonic",
        )
    assert tracker.receipt(cancellation.cancellation_id).pending_levels == (
        "software_queue",
    )


def test_worker_must_observe_the_new_execution_epoch() -> None:
    clock = ManualClock()
    cancellation = _cancellation(clock)
    participant = _participants()[0]
    clock.value = cancellation.effective_at + 0.01
    receipt = WorkerEpochDrainReceipt.evaluate(
        cancellation=cancellation,
        participant=participant,
        observed_execution_epoch=cancellation.previous_execution_epoch,
        discarded_work_ids=(),
        last_old_epoch_invocation_started_at=None,
        last_old_epoch_invocation_completed_at=None,
        in_flight_old_epoch_count=0,
        no_future_old_epoch_calls=True,
        acknowledged_at=clock.value,
    )

    assert not receipt.accepted
    assert receipt.status == "execution_epoch_not_observed"

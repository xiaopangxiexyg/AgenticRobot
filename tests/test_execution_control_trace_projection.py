from __future__ import annotations

from robot_motion.position_control import ExecutionCapabilityProfile
from robot_motion.runtime import (
    CancellationAcknowledgementTracker,
    ExecutionCancellationParticipantSpec,
    ExecutionCancellationRequest,
    ExecutionEpochRegistry,
    SafetyControlOperation,
    SafetyControlOperationRegistry,
    WorkerEpochDrainReceipt,
    aggregate_worker_epoch_drains,
)
from teleop_runtime.control_trace import (
    project_execution_fault_escalation,
    project_safety_control_operation,
    project_safety_control_tick,
    project_worker_epoch_drain,
    project_worker_epoch_drain_aggregate,
)


class ManualClock:
    def __init__(self, value: float = 10.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _capability() -> ExecutionCapabilityProfile:
    return ExecutionCapabilityProfile(
        profile_id="trace-worker-drain-v1",
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
        acceptance_evidence_id="trace-worker-drain-fixture",
        decision_runtime_budget_sec=0.005,
        ik_runtime_budget_sec=1.0 / 60.0,
        end_to_end_control_budget_sec=1.0 / 30.0,
        cancellation_ack_deadline_sec=0.1,
        required_cancellation_ack_levels=("software_queue",),
    )


def _cancel(clock: ManualClock):
    registry = ExecutionEpochRegistry(
        runtime_session_id="trace-session",
        control_domain_id="rm65-arms",
        clock_domain_id="control-monotonic",
        initial_execution_epoch=0,
        initial_segment_generation_by_group={"arms": 0},
        clock=clock,
    )
    event = registry.cancel(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=2,
            requested_scope="control_domain",
            source_event_ids=("operator-hold",),
        ),
        capability=_capability(),
    )
    return registry, event


def test_safety_operation_and_tick_projection_preserve_new_epoch_lineage() -> None:
    clock = ManualClock()
    registry, cancellation = _cancel(clock)
    operation = SafetyControlOperation.create_rm65_quick_stop(
        operation_id="quick-stop-1",
        runtime_session_id="trace-session",
        control_domain_id="rm65-arms",
        safety_source="execution_cancellation_runtime",
        source_event_id=cancellation.cancellation_id,
        target_execution_epoch=cancellation.execution_epoch,
        affected_selection_groups=("arms",),
        frozen_stop_target_hash="frozen-target",
        maximum_duration_sec=0.1,
        maximum_tick_count=2,
        authorized_at=clock(),
    )
    operations = SafetyControlOperationRegistry(
        runtime_session_id="trace-session",
        control_domain_id="rm65-arms",
        clock=clock,
    )
    operations.admit(operation)
    tick = operations.admit_tick(
        operation_id=operation.operation_id,
        tick_sequence=1,
        current_execution_epoch=registry.snapshot().execution_epoch,
        command_lineage_id="quick-stop-qcmd-1",
    )
    work = tick.as_execution_work_item(
        operation,
        authority_epoch=2,
        segment_generation_by_group={"arms": 0},
    )

    operation_fact = project_safety_control_operation(
        operation,
        embodiment_instance_id="rm65-bimanual-0",
        clock_domain_id="control-monotonic",
    )
    tick_fact = project_safety_control_tick(
        tick,
        work,
        operation=operation,
        embodiment_instance_id="rm65-bimanual-0",
        clock_domain_id="control-monotonic",
    )

    assert operation_fact.event_type == "safety_control_operation"
    assert tick_fact.event_type == "safety_control_tick"
    assert tick_fact.payload["work_kind"] == "safety_backend_command"
    assert tick_fact.payload["target_execution_epoch"] == cancellation.execution_epoch
    assert tick_fact.payload["source_safety_operation_id"] == operation.operation_id


def test_worker_drain_and_deadline_fault_have_explicit_trace_records() -> None:
    clock = ManualClock()
    _registry, cancellation = _cancel(clock)
    participant = ExecutionCancellationParticipantSpec.create(
        participant_id="position-deploy-reference-worker",
        worker_kind="reference_worker",
        affected_selection_group_ids=("arms",),
    )
    clock.value = cancellation.effective_at + 0.01
    receipt = WorkerEpochDrainReceipt.evaluate(
        cancellation=cancellation,
        participant=participant,
        observed_execution_epoch=cancellation.execution_epoch,
        discarded_work_ids=("old-reference",),
        last_old_epoch_invocation_started_at=None,
        last_old_epoch_invocation_completed_at=None,
        in_flight_old_epoch_count=0,
        no_future_old_epoch_calls=True,
        acknowledged_at=clock(),
    )
    aggregate = aggregate_worker_epoch_drains(
        cancellation=cancellation,
        participants=(participant,),
        receipts=(receipt,),
        aggregated_at=clock(),
    )
    receipt_fact = project_worker_epoch_drain(
        receipt,
        cancellation=cancellation,
        embodiment_instance_id="rm65-bimanual-0",
    )
    aggregate_fact = project_worker_epoch_drain_aggregate(
        aggregate,
        cancellation=cancellation,
        embodiment_instance_id="rm65-bimanual-0",
    )

    assert receipt_fact.payload["no_future_old_epoch_calls"] is True
    assert aggregate_fact.payload["complete"] is True

    late_clock = ManualClock(cancellation.acknowledgement_deadline_at + 0.001)
    tracker = CancellationAcknowledgementTracker(clock=late_clock)
    tracker.register(cancellation)
    escalation = tracker.fault_escalation_request(cancellation.cancellation_id)
    assert escalation is not None
    escalation_fact = project_execution_fault_escalation(
        escalation,
        embodiment_instance_id="rm65-bimanual-0",
        clock_domain_id="control-monotonic",
    )
    assert escalation_fact.event_type == "execution_fault_escalation"
    assert escalation_fact.payload["requested_gate_status"] == "fault"

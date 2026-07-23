from __future__ import annotations

from tests.test_rm65_dispatch_inventory import _execution_manifest
from teleop_runtime.backends.arms.rm65_dispatch_inventory import (
    build_rm65_arms_only_dispatch_inventory,
)
from teleop_runtime.backends.arms.rm65_worker_drain import (
    build_rm65_arms_only_worker_drain_plan,
    drain_rm65_arms_only_workers,
)
from robot_motion.runtime import WorkerEpochDrainReceipt
from tests.test_worker_epoch_drain import ManualClock, _cancellation


def _plan():
    manifest = _execution_manifest()
    inventory = build_rm65_arms_only_dispatch_inventory(manifest)
    return build_rm65_arms_only_worker_drain_plan(
        manifest=manifest,
        inventory=inventory,
    )


def test_arms_only_plan_names_every_real_async_motion_participant() -> None:
    plan = _plan()

    assert plan.integration_status == "scoped_unbound"
    assert tuple(item.participant.participant_id for item in plan.participant_bindings) == (
        "position-deploy-reference-worker",
        "rm65-left-background-stream-worker",
        "rm65-right-background-stream-worker",
    )
    assert {item.participant.worker_kind for item in plan.participant_bindings} == {
        "reference_worker",
        "backend_stream_worker",
    }
    assert all(
        item.participant.affected_selection_group_ids == ("arms",)
        for item in plan.participant_bindings
    )


def test_plan_is_evidence_only_and_does_not_claim_live_worker_drain() -> None:
    plan = _plan()

    assert plan.semantic_content_hash
    assert "unbound" in plan.integration_status
    assert all(
        "_background_stream_loop_impl" in item.implementation_path
        for item in plan.participant_bindings
        if item.participant.worker_kind == "backend_stream_worker"
    )
    assert {
        "rm65.motion.target_admission.send_command",
        "rm65.motion.background.worker_dequeue",
        "rm65.motion.pre_sdk.send_joint_target",
        "rm65.safety.quick_stop.tick.pre_sdk",
        "rm65.safety.position_stream_disengage.stop",
    } == {
        seam_id
        for item in plan.participant_bindings
        for seam_id in item.related_dispatch_seam_ids
    }


def test_each_arm_backend_is_a_distinct_drain_participant() -> None:
    plan = _plan()
    backend_bindings = tuple(
        item
        for item in plan.participant_bindings
        if item.participant.worker_kind == "backend_stream_worker"
    )

    assert len(backend_bindings) == 2
    assert backend_bindings[0].participant.semantic_content_hash != (
        backend_bindings[1].participant.semantic_content_hash
    )


def test_coordinator_drains_upstream_before_both_backend_workers() -> None:
    plan = _plan()
    clock = ManualClock()
    cancellation = _cancellation(clock)
    clock.value = cancellation.effective_at + 0.01
    participant_by_id = {
        item.participant.participant_id: item.participant
        for item in plan.participant_bindings
    }
    order: list[str] = []

    class Worker:
        def __init__(self, participant_id: str) -> None:
            self.participant_id = participant_id

        def drain_execution_cancellation(self, event):
            order.append(self.participant_id)
            participant = participant_by_id[self.participant_id]
            return WorkerEpochDrainReceipt.evaluate(
                cancellation=event,
                participant=participant,
                observed_execution_epoch=event.execution_epoch,
                discarded_work_ids=(),
                last_old_epoch_invocation_started_at=None,
                last_old_epoch_invocation_completed_at=None,
                in_flight_old_epoch_count=0,
                no_future_old_epoch_calls=True,
                acknowledged_at=clock.value,
            )

    result = drain_rm65_arms_only_workers(
        plan=plan,
        cancellation=cancellation,
        position_reference_worker=Worker("position-deploy-reference-worker"),
        left_background_worker=Worker("rm65-left-background-stream-worker"),
        right_background_worker=Worker("rm65-right-background-stream-worker"),
        clock=clock,
    )

    assert order == [
        "position-deploy-reference-worker",
        "rm65-left-background-stream-worker",
        "rm65-right-background-stream-worker",
    ]
    assert result.aggregate.complete

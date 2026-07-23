from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robot_motion.runtime import (
    EffectorCommand,
    ExecutionCancellationParticipantSpec,
    ExecutionCancellationRequest,
    ExecutionEpochGuard,
    ExecutionEpochRegistry,
    ExecutionWorkItem,
)
from teleop_runtime.backends.arms.rm65 import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor,
    PositionDeployExecutorConfig,
)
from tests.test_position_deploy_executor import _filtered_pose, _observation, _pipeline
from tests.test_worker_epoch_drain import _capability


class _BlockingArm:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[tuple[float, ...]] = []

    def Movej_CANFD(self, joint_deg, **_kwargs):
        self.calls.append(tuple(float(value) for value in joint_deg))
        self.entered.set()
        assert self.release.wait(timeout=1.0)
        return 0


def _registry() -> ExecutionEpochRegistry:
    return ExecutionEpochRegistry(
        runtime_session_id="session-rm65-worker-drain",
        control_domain_id="rm65-arms",
        clock_domain_id="control-monotonic",
        initial_execution_epoch=3,
        initial_segment_generation_by_group={"arms": 7},
    )


def _old_reference_work() -> ExecutionWorkItem:
    return ExecutionWorkItem(
        work_item_id="reference-segment-old",
        work_kind="reference_segment",
        control_domain_id="rm65-arms",
        authority_epoch=2,
        execution_epoch=3,
        segment_generation_by_group={"arms": 7},
        source_decision_id="decision-old",
        source_reference_segment_id="segment-old",
    )


def _cancel(registry: ExecutionEpochRegistry):
    return registry.cancel(
        ExecutionCancellationRequest(
            reason="hold",
            authority_epoch=2,
            requested_scope="control_domain",
            invalidated_reference_segment_ids=("segment-old",),
            source_event_ids=("operator-hold",),
        ),
        capability=_capability(),
    )


def test_position_reference_worker_drain_discards_old_queued_lineage() -> None:
    registry = _registry()
    participant = ExecutionCancellationParticipantSpec.create(
        participant_id="position-deploy-reference-worker",
        worker_kind="reference_worker",
        affected_selection_group_ids=("arms",),
    )
    executor = PositionDeployExecutor(
        config=PositionDeployExecutorConfig(
            sample_hz=15.0,
            reference_hz=60.0,
            command_hz=120.0,
        ),
        canonical_pipeline=_pipeline(("left",)),
        execution_epoch_guard=ExecutionEpochGuard(registry),
        worker_drain_participant=participant,
    )
    executor.submit(
        _filtered_pose("left", np.asarray([0.1, 0.0, 0.3, 0, 0, 0, 1], dtype=np.float32)),
        _observation(sides=("left",)),
        "policy",
        execution_work_item=_old_reference_work(),
    )

    cancellation = _cancel(registry)
    receipt = executor.drain_execution_cancellation(cancellation)

    assert receipt.accepted
    assert receipt.status == "drained"
    assert receipt.discarded_work_ids == ("reference-segment-old",)
    assert executor.snapshot()["queue_depth"] == 0


def test_partial_epoch_binding_cannot_be_attached_to_real_hardware_config() -> None:
    registry = _registry()
    participant = ExecutionCancellationParticipantSpec.create(
        participant_id="rm65-left-background-stream-worker",
        worker_kind="backend_stream_worker",
        affected_selection_group_ids=("arms",),
    )

    with pytest.raises(RuntimeError, match="fake-only"):
        RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="192.0.2.1",
                side="left",
                connect_hardware=True,
            ),
            execution_epoch_guard=ExecutionEpochGuard(registry),
            worker_drain_participant=participant,
        )


def test_backend_drain_waits_for_inflight_old_sdk_call_then_blocks_tail_send() -> None:
    registry = _registry()
    participant = ExecutionCancellationParticipantSpec.create(
        participant_id="rm65-left-background-stream-worker",
        worker_kind="backend_stream_worker",
        affected_selection_group_ids=("arms",),
    )
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            background_streaming_hz=200.0,
        ),
        execution_epoch_guard=ExecutionEpochGuard(registry),
        worker_drain_participant=participant,
    )
    arm = _BlockingArm()
    backend.arm = arm
    upstream = ExecutionWorkItem(
        work_item_id="position-command-old",
        work_kind="position_command",
        control_domain_id="rm65-arms",
        authority_epoch=2,
        execution_epoch=3,
        segment_generation_by_group={"arms": 7},
        source_decision_id="decision-old",
        source_reference_segment_id="segment-old",
    )
    backend.send_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, 0.01),
            metadata={"source": "policy"},
            execution_work_item=upstream,
        )
    )

    backend._stream_stop_event.clear()
    worker = threading.Thread(target=backend._background_stream_loop_impl, daemon=True)
    worker.start()
    assert arm.entered.wait(timeout=1.0)

    cancellation = _cancel(registry)
    result: dict[str, object] = {}

    def drain() -> None:
        result["receipt"] = backend.drain_execution_cancellation(cancellation)

    drain_thread = threading.Thread(target=drain, daemon=True)
    drain_thread.start()
    time.sleep(0.02)
    assert drain_thread.is_alive(), "drain must wait for the in-flight SDK call"

    arm.release.set()
    drain_thread.join(timeout=1.0)
    assert not drain_thread.is_alive()
    receipt = result["receipt"]
    assert receipt.accepted
    assert receipt.in_flight_old_epoch_count == 0
    assert receipt.no_future_old_epoch_calls
    assert receipt.discarded_work_ids == (
        "position-command-old:rm65:left:target:1",
    )

    time.sleep(0.03)
    backend._stream_stop_event.set()
    worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert len(arm.calls) == 1

    # A stale producer cannot repopulate the mailbox after drain.
    try:
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.02),
                metadata={"source": "policy"},
                execution_work_item=upstream,
            )
        )
    except RuntimeError as exc:
        assert "stale execution work" in str(exc)
    else:
        raise AssertionError("old execution work was accepted after cancellation")


def test_epoch_mediated_quick_stop_cannot_inherit_motion_work() -> None:
    registry = _registry()
    participant = ExecutionCancellationParticipantSpec.create(
        participant_id="rm65-left-background-stream-worker",
        worker_kind="backend_stream_worker",
        affected_selection_group_ids=("arms",),
    )
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            position_command_planner="acceleration_limited",
        ),
        execution_epoch_guard=ExecutionEpochGuard(registry),
        worker_drain_participant=participant,
    )
    backend.arm = _BlockingArm()
    motion_work = ExecutionWorkItem(
        work_item_id="position-command-before-quick-stop",
        work_kind="position_command",
        control_domain_id="rm65-arms",
        authority_epoch=2,
        execution_epoch=3,
        segment_generation_by_group={"arms": 7},
        source_decision_id="decision-old",
        source_reference_segment_id="segment-old",
    )

    with pytest.raises(RuntimeError, match="cannot inherit ordinary motion work"):
        backend.send_command(
            EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.full(6, 0.01),
                metadata={
                    "source": "policy",
                    "collision_avoidance_status": "hold_last_safe",
                    "ik_jump_gate_status": "pass",
                    "action_pose_source": "limited_solver_output_qpos",
                },
                execution_work_item=motion_work,
            )
        )
    assert backend._stream_execution_work_item is None
    assert backend.arm.calls == []

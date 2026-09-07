from __future__ import annotations

from dataclasses import replace
import threading
import time

import numpy as np
import pytest

from robot_motion.deploy import (
    ASYNC_ALIGNED_CHUNK_MODE_V1,
    POLICY_WORKER_SESSION_ABANDONED_UNDRAINED,
    POLICY_WORKER_SESSION_ACTIVE,
    POLICY_WORKER_SESSION_DRAINED,
    PolicyInferenceRequest,
    RTC_FLOW_INPAINT_MODE_V1,
    admit_policy_chunk_result,
    admit_policy_inference_request,
    create_policy_async_producer,
    empty_policy_action_plan,
    merge_policy_chunk_result,
)
from robot_motion.deploy.policy_worker import (
    POLICY_EXECUTOR_CANCEL_FAILED,
    POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED,
    POLICY_EXECUTOR_CANCEL_PENDING,
    POLICY_EXECUTOR_CANCEL_REQUESTED,
    POLICY_WORKER_FAULT_DETAIL_MAX_CHARS,
    _fault_result,
)
from robot_motion.deploy.observation_view import PolicyObservationView
from robot_motion.runtime.observation_cut import (
    ObservationBundle,
    ObservationCutRecord,
    ObservationSampleRef,
)
from sim.mujoco.deploy.observation import (
    MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1,
    MUJOCO_QPOS_STREAM_ID_V1,
)
from sim.mujoco.deploy.rtc_policy import (
    VIRTUAL_QPOS_ACTION_CODEC_HASH_V1,
    VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1,
    VIRTUAL_QPOS_NORMALIZATION_HASH_V1,
    VirtualBimanualQposPolicy,
)
from teleop_runtime.deploy.policy_worker import (
    BoundedPolicyInferenceWorker,
    PolicyInferenceWorkItem,
)


def _virtual_work_item(
    *,
    worker_id: str = "rtc-worker:unit",
    worker_session_id: str = "rtc-worker-session:unit:0",
):
    sample = ObservationSampleRef(
        stream_id=MUJOCO_QPOS_STREAM_ID_V1,
        sample_id="sample:qpos:0",
        sequence=7,
        measured_at=10.0,
        measured_clock_domain_id="clock:unit",
        available_at=10.0,
        available_clock_domain_id="clock:unit",
        payload_object_id="payload:qpos:0",
    )
    record = ObservationCutRecord(
        observation_cut_id="cut:qpos:0",
        runtime_session_id="session:unit",
        control_domain_id="domain:unit",
        policy_node_id="node:unit",
        policy_node_epoch=1,
        sequence=7,
        cutoff_at=10.0,
        cutoff_clock_domain_id="clock:unit",
        cutoff_clock_reset_epoch=0,
        cutoff_admission_sequence=0,
        built_at=10.0,
        built_clock_domain_id="clock:unit",
        selected_sample_by_stream={MUJOCO_QPOS_STREAM_ID_V1: sample},
        context_dependencies=(),
        observation_contract_hash="observation-contract:unit",
        policy_io_layout_hash=MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1,
        trigger_reason="worker-unit",
        semantic_content_hash="observation-cut-hash:unit",
    )
    plan = empty_policy_action_plan(
        plan_id="plan:unit",
        session_id=record.runtime_session_id,
        control_domain_id=record.control_domain_id,
        producer_id="policy:unit",
        producer_epoch=1,
        policy_node_id=record.policy_node_id,
        clock_domain_id=record.cutoff_clock_domain_id,
        cursor_at=10.0,
        created_at=10.0,
    )
    policy = VirtualBimanualQposPolicy()
    request = PolicyInferenceRequest(
        request_id="request:unit:0",
        inference_id="inference:unit:0",
        rtc_mode=ASYNC_ALIGNED_CHUNK_MODE_V1,
        session_id=plan.session_id,
        control_domain_id=plan.control_domain_id,
        producer_id=plan.producer_id,
        producer_epoch=plan.producer_epoch,
        policy_node_id=plan.policy_node_id,
        artifact_hash=policy.spec.artifact_hash,
        observation_cut_id=record.observation_cut_id,
        observation_timestamp=record.built_at,
        observation_step_index=record.sequence,
        triggered_at=10.0,
        admitted_at=10.0,
        first_target_effective_at=10.0,
        clock_domain_id=plan.clock_domain_id,
        action_period_sec=0.05,
        prediction_horizon_steps=2,
        minimum_execution_horizon_steps=1,
        source_plan_id=plan.plan_id,
        source_plan_generation=plan.generation,
        source_plan_hash=plan.semantic_content_hash,
        source_plan_cursor_at=plan.cursor_at,
        producer_plan_committed_through=None,
        forecast_delay_steps=0,
        delay_estimator_spec_hash="delay-spec:unit",
        action_codec_hash=VIRTUAL_QPOS_ACTION_CODEC_HASH_V1,
        frame_contract_hash=VIRTUAL_QPOS_FRAME_CONTRACT_HASH_V1,
        layout_hash=MUJOCO_QPOS_POLICY_IO_LAYOUT_HASH_V1,
        normalization_hash=VIRTUAL_QPOS_NORMALIZATION_HASH_V1,
        valid_until=10.1,
    )
    mutable_left = np.zeros(6, dtype=np.float32)
    mutable_right = np.zeros(6, dtype=np.float32)
    actor_observation = PolicyObservationView(
        contract_id="virtual-qpos-worker-view-v1",
        contract_hash="virtual-qpos-worker-view-hash-v1",
        payload={
            "timestamp": record.built_at,
            "step_index": record.sequence,
            "arm_qpos_by_side": {
                "left": mutable_left,
                "right": mutable_right,
            },
        },
    )
    work_item = PolicyInferenceWorkItem(
        worker_id=worker_id,
        worker_session_id=worker_session_id,
        submit_sequence=0,
        request=request,
        actor_observation=actor_observation,
        submitted_at=10.0,
    )
    return work_item, plan, policy, record, mutable_left, mutable_right


class BlockingVirtualPolicyExecutor:
    def __init__(self, policy, record, *, blocked: bool = True) -> None:
        self.policy = policy
        self.record = record
        self.started = threading.Event()
        self.release = threading.Event()
        if not blocked:
            self.release.set()

    def __call__(self, request, actor_observation):
        self.started.set()
        if not self.release.wait(timeout=2.0):
            raise TimeoutError("virtual policy release timed out")
        qpos = actor_observation.payload["arm_qpos_by_side"]
        bundle = ObservationBundle(
            record=self.record,
            payload_handles_by_stream={
                MUJOCO_QPOS_STREAM_ID_V1: {
                    "arm_qpos_by_side": {
                        "left": qpos["left"],
                        "right": qpos["right"],
                    }
                }
            },
        )
        return self.policy.predict(
            request,
            bundle,
            inference_started_at=request.admitted_at,
            inference_completed_at=request.admitted_at,
            result_received_at=request.admitted_at,
        )


class CancellableBlockingVirtualPolicyExecutor(BlockingVirtualPolicyExecutor):
    def __init__(self, policy, record) -> None:
        super().__init__(policy, record)
        self.cancel_count = 0

    def request_cancel(self) -> bool:
        self.cancel_count += 1
        self.release.set()
        return True


def _poll_completion(worker, *, timeout_sec: float = 2.0):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        completion = worker.poll_completion()
        if completion is not None:
            return completion
        time.sleep(0.001)
    raise AssertionError("policy worker completion timed out")


def test_worker_has_one_in_flight_request_and_detaches_actor_observation() -> None:
    work_item, _plan, policy, record, mutable_left, mutable_right = (
        _virtual_work_item()
    )
    executor = BlockingVirtualPolicyExecutor(policy, record)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    worker.start()
    try:
        accepted = worker.submit(work_item)
        assert accepted.accepted and accepted.status == "accepted"
        assert accepted.worker_session_id == work_item.worker_session_id
        assert worker.session_snapshot().session_state == POLICY_WORKER_SESSION_ACTIVE
        assert executor.started.wait(timeout=1.0)
        second = worker.submit(work_item)
        assert not second.accepted
        assert second.status == "in_flight_request_exists"

        mutable_left[:] = 9.0
        mutable_right[:] = -9.0
        executor.release.set()
        completion = _poll_completion(worker)
        assert completion.status == "ready"
        assert completion.worker_session_id == work_item.worker_session_id
        first_action = completion.result.decoded_absolute_actions[0]
        np.testing.assert_allclose(
            first_action.arm_joint_targets["left"],
            np.full(6, 0.01),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            first_action.arm_joint_targets["right"],
            np.full(6, -0.01),
            atol=1e-7,
        )
        duplicate = worker.submit(work_item)
        assert not duplicate.accepted
        assert duplicate.status == "duplicate_request_id"
    finally:
        stop = worker.stop(timeout_sec=1.0)
    assert stop.thread_exited
    assert stop.status == "stopped"
    assert stop.session_state == POLICY_WORKER_SESSION_DRAINED
    assert not stop.execution_drain_evidence
    assert stop.executor_cancellation is not None
    assert (
        stop.executor_cancellation.status
        == POLICY_EXECUTOR_CANCEL_NOT_SUPPORTED
    )


def test_worker_exception_becomes_typed_fault_completion() -> None:
    work_item, _plan, _policy, _record, _left, _right = _virtual_work_item()

    def fail(_request, _observation):
        raise RuntimeError("synthetic virtual policy failure")

    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=fail,
    )
    worker.start()
    try:
        assert worker.submit(work_item).accepted
        completion = _poll_completion(worker)
        assert completion.status == "fault"
        assert completion.result.source_chunk_size == 0
        assert completion.result.decoded_absolute_actions == ()
        assert completion.result.fault_code == (
            "policy_worker_exception:RuntimeError"
        )
    finally:
        worker.stop(timeout_sec=1.0)


def test_conditioned_worker_fault_preserves_binding_without_fake_receipts() -> None:
    work_item, plan, _policy, _record, _left, _right = _virtual_work_item()
    request = replace(
        work_item.request,
        rtc_mode=RTC_FLOW_INPAINT_MODE_V1,
        conditioning_binding_hash="sha256:conditioned-worker-fault",
    )
    producer = create_policy_async_producer(
        state_id="async-producer:conditioned-worker-fault",
        plan=plan,
        worker_session_id=work_item.worker_session_id,
        created_at=10.0,
    )
    requested = admit_policy_inference_request(producer, request)
    assert requested.receipt.accepted

    result = _fault_result(
        request,
        inference_started_at=10.0,
        inference_completed_at=10.0,
        result_received_at=10.0,
        error=RuntimeError("transport deadline\n" + "x" * 1000),
    )

    assert result.conditioning_binding_hash == request.conditioning_binding_hash
    assert result.conditioning_encoding_receipt_hash is None
    assert result.conditioning_consumption_receipt_hash is None
    assert result.fault_code == "policy_worker_exception:RuntimeError"
    assert result.fault_detail is not None
    assert len(result.fault_detail) == POLICY_WORKER_FAULT_DETAIL_MAX_CHARS
    assert "\n" not in result.fault_detail
    admitted = admit_policy_chunk_result(
        requested.state,
        result,
        worker_session_id=work_item.worker_session_id,
        admitted_at=10.0,
        actual_cursor_at=10.0,
    )
    assert not admitted.receipt.accepted
    assert admitted.receipt.reasons == ("worker_result_fault",)


def test_stop_requests_executor_cancel_and_only_thread_exit_proves_drain() -> None:
    work_item, _plan, policy, record, _left, _right = _virtual_work_item()
    executor = CancellableBlockingVirtualPolicyExecutor(policy, record)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    worker.start()
    assert worker.submit(work_item).accepted
    assert executor.started.wait(timeout=1.0)

    stop = worker.stop(timeout_sec=1.0)

    assert executor.cancel_count == 1
    assert stop.thread_exited
    assert stop.status == "stopped"
    assert stop.session_state == POLICY_WORKER_SESSION_DRAINED
    assert not stop.execution_drain_evidence
    assert stop.executor_cancellation is not None
    assert stop.executor_cancellation.executing_request_id == (
        work_item.request.request_id
    )
    assert stop.executor_cancellation.hook_available
    assert stop.executor_cancellation.request_accepted
    assert (
        stop.executor_cancellation.status == POLICY_EXECUTOR_CANCEL_REQUESTED
    )
    assert stop.pending_completion_count == 1


def test_invalid_executor_cancel_hook_is_typed_and_never_claims_drain() -> None:
    work_item, _plan, policy, record, _left, _right = _virtual_work_item()

    class InvalidCancellationExecutor(BlockingVirtualPolicyExecutor):
        def request_cancel(self):
            return None

    executor = InvalidCancellationExecutor(policy, record)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    worker.start()
    try:
        assert worker.submit(work_item).accepted
        assert executor.started.wait(timeout=1.0)
        stop = worker.stop(timeout_sec=0.001)
        assert not stop.thread_exited
        assert stop.status == "executing_call_not_drained"
        assert stop.session_state == POLICY_WORKER_SESSION_ABANDONED_UNDRAINED
        assert not stop.execution_drain_evidence
        assert stop.executor_cancellation is not None
        assert stop.executor_cancellation.status == POLICY_EXECUTOR_CANCEL_FAILED
        assert stop.executor_cancellation.failure_type == "TypeError"
    finally:
        executor.release.set()
        worker.stop(timeout_sec=1.0)


def test_blocking_cancel_hook_cannot_block_zero_timeout_stop() -> None:
    work_item, _plan, policy, record, _left, _right = _virtual_work_item()

    class BlockingCancellationExecutor(BlockingVirtualPolicyExecutor):
        def __init__(self, policy, record) -> None:
            super().__init__(policy, record)
            self.cancel_release = threading.Event()

        def request_cancel(self) -> bool:
            self.cancel_release.wait(timeout=2.0)
            return True

    executor = BlockingCancellationExecutor(policy, record)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    worker.start()
    try:
        assert worker.submit(work_item).accepted
        assert executor.started.wait(timeout=1.0)
        started = time.perf_counter()
        stop = worker.stop(timeout_sec=0.0)
        assert time.perf_counter() - started < 0.05
        assert not stop.thread_exited
        assert stop.status == "executing_call_not_drained"
        assert stop.executor_cancellation is not None
        assert stop.executor_cancellation.status == POLICY_EXECUTOR_CANCEL_PENDING
        assert not stop.executor_cancellation.request_accepted
    finally:
        executor.cancel_release.set()
        executor.release.set()
        worker.stop(timeout_sec=1.0)


def test_stop_timeout_never_claims_drain_and_late_result_fails_new_epoch_plan() -> None:
    work_item, plan, policy, record, _left, _right = _virtual_work_item()
    executor = BlockingVirtualPolicyExecutor(policy, record)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    worker.start()
    assert worker.submit(work_item).accepted
    assert executor.started.wait(timeout=1.0)

    incomplete = worker.stop(timeout_sec=0.001)
    assert not incomplete.thread_exited
    assert incomplete.status == "executing_call_not_drained"
    assert incomplete.executing_request_id == work_item.request.request_id
    assert incomplete.session_state == POLICY_WORKER_SESSION_ABANDONED_UNDRAINED
    assert worker.session_state == POLICY_WORKER_SESSION_ABANDONED_UNDRAINED
    assert not incomplete.execution_drain_evidence

    executor.release.set()
    deadline = time.monotonic() + 2.0
    while worker.state != "stopped" and time.monotonic() < deadline:
        time.sleep(0.001)
    assert worker.state == "stopped"
    assert worker.session_state == POLICY_WORKER_SESSION_DRAINED
    stopped = worker.stop(timeout_sec=1.0)
    assert stopped.thread_exited
    assert stopped.pending_completion_count == 1
    completion = _poll_completion(worker)
    assert completion.result.producer_epoch == 1

    invalidated_plan = empty_policy_action_plan(
        plan_id=plan.plan_id,
        session_id=plan.session_id,
        control_domain_id=plan.control_domain_id,
        producer_id=plan.producer_id,
        producer_epoch=2,
        policy_node_id=plan.policy_node_id,
        clock_domain_id=plan.clock_domain_id,
        cursor_at=10.0,
        created_at=10.0,
    )
    late = merge_policy_chunk_result(
        invalidated_plan,
        work_item.request,
        completion.result,
        admitted_at=10.05,
        actual_cursor_at=10.05,
    )
    assert not late.receipt.accepted
    assert late.receipt.reasons == (
        "producer_epoch_mismatch",
        "request_not_in_flight",
    )
    assert late.plan.semantic_content_hash == invalidated_plan.semantic_content_hash


def test_submit_before_start_is_nonblocking_rejection() -> None:
    work_item, _plan, policy, record, _left, _right = _virtual_work_item()
    executor = BlockingVirtualPolicyExecutor(policy, record, blocked=False)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    rejected = worker.submit(work_item)
    assert not rejected.accepted
    assert rejected.status == "worker_not_running"
    stop = worker.stop(timeout_sec=0.0)
    assert stop.thread_exited
    assert stop.status == "stopped"
    assert stop.session_state == POLICY_WORKER_SESSION_DRAINED


def test_work_item_cannot_cross_worker_session_identity() -> None:
    work_item, _plan, policy, record, _left, _right = _virtual_work_item(
        worker_session_id="rtc-worker-session:old",
    )
    executor = BlockingVirtualPolicyExecutor(policy, record, blocked=False)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id="rtc-worker-session:new",
        executor=executor,
    )
    worker.start()
    try:
        with pytest.raises(ValueError, match="another policy worker session"):
            worker.submit(work_item)
        snapshot = worker.session_snapshot()
        assert snapshot.worker_session_id == "rtc-worker-session:new"
        assert snapshot.session_state == POLICY_WORKER_SESSION_ACTIVE
        assert snapshot.active_request_id is None
    finally:
        worker.stop(timeout_sec=1.0)


def test_worker_completion_enters_only_the_bound_timed_plan_state() -> None:
    work_item, plan, policy, record, _left, _right = _virtual_work_item()
    producer = create_policy_async_producer(
        state_id="async-producer:worker-integration",
        plan=plan,
        worker_session_id=work_item.worker_session_id,
        created_at=10.0,
    )
    requested = admit_policy_inference_request(producer, work_item.request)
    assert requested.receipt.accepted

    executor = BlockingVirtualPolicyExecutor(policy, record, blocked=False)
    worker = BoundedPolicyInferenceWorker(
        worker_id=work_item.worker_id,
        worker_session_id=work_item.worker_session_id,
        executor=executor,
    )
    worker.start()
    try:
        assert worker.submit(work_item).accepted
        completion = _poll_completion(worker)
        admitted = admit_policy_chunk_result(
            requested.state,
            completion.result,
            worker_session_id=completion.worker_session_id,
            admitted_at=10.0,
            actual_cursor_at=10.0,
        )
        assert admitted.receipt.accepted
        assert admitted.merge_receipt is not None
        assert admitted.merge_receipt.accepted
        assert admitted.state.active_request is None
        assert admitted.state.plan.in_flight_request_id is None
        assert [
            keyframe.chunk_keyframe_index
            for keyframe in admitted.state.plan.keyframes
        ] == [0, 1]
    finally:
        worker.stop(timeout_sec=1.0)

from __future__ import annotations

import argparse
import threading
import time

import numpy as np
import pytest

from robot_motion.action import ArmActionSpec
from robot_motion.deploy import (
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    ThreadedPolicyPredictionWorker,
    build_deployment_observation,
)
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES
from teleop_runtime.deploy import run_policy

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def _deploy_args(argv: list[str]):
    parser = argparse.ArgumentParser()
    run_policy.add_args(parser)
    return parser.parse_args(argv)


def _observation(step_index: int):
    pose = np.eye(4, dtype=float)
    return build_deployment_observation(
        active_sides=("left", "right"),
        arms_enabled=True,
        hands_enabled=False,
        step_index=step_index,
        arm_states={
            side: {
                "current_qpos": np.zeros(6),
                "actual_qpos": np.zeros(6),
                "current_tcp_pose": pose,
                "actual_tcp_pose": pose,
            }
            for side in ("left", "right")
        },
        hand_states={},
        timestamp=time.monotonic(),
        metadata={
            "arm_action_specs_by_side": {
                side: ArmActionSpec.tool_tcp(side).to_metadata()
                for side in ("left", "right")
            }
        },
    )


def _chunk(size: int = 16) -> np.ndarray:
    result = np.full((size, 26), np.nan, dtype=np.float32)
    for index in range(size):
        result[index, ACTION_ARM_POSE_SLICES["left"]] = (
            0.1 + index * 0.001,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        result[index, ACTION_ARM_POSE_SLICES["right"]] = (
            0.2 + index * 0.001,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    return result


def _runtime() -> PolicyDeployRuntime:
    return PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=False,
            policy_node_id="async-prediction-test",
            policy_clock_domain_id="local_monotonic",
            action_keyframe_period_sec=1.0 / 15.0,
        )
    )


class _BlockingPolicy:
    name = "blocking-openpi-fixture"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.reset_calls = 0

    def predict(self, observation):
        assert tuple(observation.active_sides) == ("left", "right")
        self.calls += 1
        self.entered.set()
        assert self.release.wait(timeout=2.0)
        self.release.clear()
        return _chunk()

    def reset(self) -> None:
        self.reset_calls += 1


def _wait_for_candidate(worker: ThreadedPolicyPredictionWorker, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        candidate = worker.poll_completion()
        if candidate is not None:
            return candidate
        time.sleep(0.001)
    raise AssertionError(worker.status())


def test_prediction_call_is_non_blocking_and_admission_stays_with_caller() -> None:
    runtime = _runtime()
    policy = _BlockingPolicy()
    worker = ThreadedPolicyPredictionWorker(
        source_id="unit-openpi-prefetch",
        runtime=runtime,
        policy=policy,
    )
    worker.start()
    try:
        started_at = time.perf_counter()
        assert worker.submit(
            _observation(0),
            inference_id=runtime.reserve_prediction_inference_id(),
        )
        assert time.perf_counter() - started_at < 0.05
        assert policy.entered.wait(timeout=1.0)
        assert runtime.action_queue_length() == 0

        policy.release.set()
        candidate = _wait_for_candidate(worker)
        assert runtime.action_queue_length() == 0
        result = runtime.admit_prediction_result(
            candidate,
            now=time.monotonic(),
            queue_policy="replace",
        )
        assert result.accepted
        assert runtime.action_queue_length() == 16
    finally:
        policy.release.set()
        worker.close()


def test_prefetch_append_keeps_chunk_boundary_non_empty() -> None:
    runtime = _runtime()
    policy = _BlockingPolicy()
    worker = ThreadedPolicyPredictionWorker(
        source_id="unit-openpi-continuity",
        runtime=runtime,
        policy=policy,
    )
    worker.start()
    try:
        assert worker.submit(
            _observation(0),
            inference_id=runtime.reserve_prediction_inference_id(),
        )
        assert policy.entered.wait(timeout=1.0)
        policy.release.set()
        first = _wait_for_candidate(worker)
        assert runtime.admit_prediction_result(
            first,
            now=time.monotonic(),
            queue_policy="replace",
        ).accepted

        for _ in range(12):
            assert runtime.pop_action() is not None
        assert runtime.action_queue_length() == 4

        policy.entered.clear()
        assert worker.submit(
            _observation(12),
            inference_id=runtime.reserve_prediction_inference_id(),
        )
        assert policy.entered.wait(timeout=1.0)
        policy.release.set()
        second = _wait_for_candidate(worker)
        append = runtime.admit_prediction_result(
            second,
            now=time.monotonic(),
            queue_policy="append",
        )
        assert append.accepted
        assert append.queue_len_before == 4
        assert append.queue_len_after == 20

        observed_chunk_ids = []
        while runtime.action_queue_length() > 0:
            action = runtime.pop_action()
            assert action is not None
            observed_chunk_ids.append(action.metadata["deploy_action_chunk_id"])
        assert observed_chunk_ids[:4] == [0] * 4
        assert observed_chunk_ids[4:] == [1] * 16
    finally:
        policy.release.set()
        worker.close()


def test_activation_reset_discards_late_completion() -> None:
    runtime = _runtime()
    policy = _BlockingPolicy()
    worker = ThreadedPolicyPredictionWorker(
        source_id="unit-openpi-generation",
        runtime=runtime,
        policy=policy,
    )
    worker.start()
    try:
        assert worker.submit(
            _observation(0),
            inference_id=runtime.reserve_prediction_inference_id(),
        )
        assert policy.entered.wait(timeout=1.0)
        worker.reset_for_activation()
        policy.release.set()

        deadline = time.monotonic() + 1.0
        while worker.reset_pending:
            assert time.monotonic() < deadline
            assert worker.poll_completion() is None
            time.sleep(0.001)
        assert worker.discarded_late_completion_count == 1
        assert runtime.action_queue_length() == 0
        assert policy.reset_calls >= 2
    finally:
        policy.release.set()
        worker.close()


def test_prediction_transaction_policy_is_rejected_before_worker_start() -> None:
    class _TransactionalPolicy(_BlockingPolicy):
        def begin_prediction_transaction(self):
            return object()

    with pytest.raises(TypeError, match="use the synchronous queue"):
        ThreadedPolicyPredictionWorker(
            source_id="unit-transaction-rejection",
            runtime=_runtime(),
            policy=_TransactionalPolicy(),
        )


def test_loader_transaction_facade_uses_underlying_policy_capabilities() -> None:
    runtime = _runtime()
    inner = _BlockingPolicy()

    class _LoaderFacade:
        policy = inner

        def predict(self, observation):
            return inner.predict(observation)

        def begin_prediction_transaction(self):
            raise AssertionError("facade transaction must remain hidden")

        def commit_prediction_transaction(self, _checkpoint):
            raise AssertionError("facade transaction must remain hidden")

        def rollback_prediction_transaction(self, _checkpoint):
            raise AssertionError("facade transaction must remain hidden")

    worker = ThreadedPolicyPredictionWorker(
        source_id="unit-loader-facade",
        runtime=runtime,
        policy=_LoaderFacade(),
        lifecycle_policy=inner,
        transaction_policy=inner,
    )
    worker.start()
    try:
        assert worker.submit(
            _observation(0),
            inference_id=runtime.reserve_prediction_inference_id(),
        )
        assert inner.entered.wait(timeout=1.0)
        inner.release.set()
        assert _wait_for_candidate(worker) is not None
    finally:
        inner.release.set()
        worker.close()


def test_rejected_close_cancellation_still_drains_worker_and_closes_policy() -> None:
    class _RejectingCancelPolicy(_BlockingPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.closed = False

        def request_cancel(self) -> bool:
            self.release.set()
            return False

        def close(self) -> None:
            self.closed = True

    runtime = _runtime()
    policy = _RejectingCancelPolicy()
    worker = ThreadedPolicyPredictionWorker(
        source_id="unit-close-drain",
        runtime=runtime,
        policy=policy,
    )
    worker.start()
    assert worker.submit(
        _observation(0),
        inference_id=runtime.reserve_prediction_inference_id(),
    )
    assert policy.entered.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="cancellation failed"):
        worker.close()
    assert worker.status()["worker_alive"] is False
    assert policy.closed is True


def test_threaded_prediction_cli_rejects_rtc_and_short_chunk_threshold() -> None:
    rtc_args = _deploy_args(
        [
            "--policy-prediction-mode",
            "threaded-prefetch",
            "--rtc-profile",
            "rtc_3c_virtual_shadow_15_30_60_120_v1",
        ]
    )
    with pytest.raises(ValueError, match="only valid for the legacy non-RTC"):
        run_policy.validate_args(rtc_args)

    short_chunk_args = _deploy_args(
        [
            "--policy-prediction-mode",
            "threaded-prefetch",
            "--action-chunk-execution-steps",
            "4",
        ]
    )
    with pytest.raises(ValueError, match="must be smaller"):
        run_policy.validate_args(short_chunk_args)

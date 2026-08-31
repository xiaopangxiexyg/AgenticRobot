from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from robot_motion.action import ArmActionSpec
from robot_motion.deploy import (
    DeployActionPreparer,
    DeployRobotActionAdapter,
    PolicyDeployRuntime,
    PolicyDeployRuntimeConfig,
    ThreadedPolicyEmissionSource,
    ThreadedPolicyEmissionSourceConfig,
    build_deployment_observation,
)
from robot_motion.deploy.schema import ACTION_ARM_POSE_SLICES


def _observation(step_index: int) -> object:
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


def _chunk() -> np.ndarray:
    result = np.full((2, 26), np.nan, dtype=np.float32)
    for index in range(2):
        result[index, ACTION_ARM_POSE_SLICES["left"]] = (
            0.1 + index * 0.01,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        result[index, ACTION_ARM_POSE_SLICES["right"]] = (
            0.2 + index * 0.01,
            0.0,
            0.3,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    return result


class _BlockingPolicy:
    name = "blocking-fixture"

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.reset_calls = 0

    def predict(self, observation):
        assert tuple(observation.active_sides) == ("left", "right")
        self.calls += 1
        if self.calls == 1:
            self.entered.set()
            assert self.release.wait(timeout=2.0)
        return _chunk()

    def reset(self) -> None:
        self.reset_calls += 1


def _source(
    policy: _BlockingPolicy,
    *,
    prediction_request_queue_threshold: int = 0,
    policy_interval_sec: float = 0.02,
    action_period_sec: float = 0.02,
    worker_join_timeout_sec: float = 2.0,
) -> ThreadedPolicyEmissionSource:
    runtime = PolicyDeployRuntime(
        PolicyDeployRuntimeConfig(
            active_sides=("right",),
            model_observation_sides=("left", "right"),
            arms_enabled=True,
            hands_enabled=False,
            policy_node_id="unit-policy",
            policy_clock_domain_id="local_monotonic",
            action_keyframe_period_sec=action_period_sec,
            action_maximum_hold_sec=0.1,
            extend_final_keyframe_to_maximum_hold=True,
        )
    )
    next_step = 0

    def capture(*, step_index=None):
        nonlocal next_step
        selected = next_step if step_index is None else int(step_index)
        next_step = max(next_step + 1, selected + 1)
        return _observation(selected)

    return ThreadedPolicyEmissionSource(
        ThreadedPolicyEmissionSourceConfig(
            source_id="unit-threaded-policy",
            policy_name=policy.name,
            policy_interval_sec=policy_interval_sec,
            action_period_sec=action_period_sec,
            maximum_hold_sec=0.1,
            prediction_request_queue_threshold=(
                prediction_request_queue_threshold
            ),
            worker_join_timeout_sec=worker_join_timeout_sec,
        ),
        runtime=runtime,
        policy=policy,
        observation_capture=capture,
        action_preparer=DeployActionPreparer(
            DeployRobotActionAdapter(
                active_sides=("right",),
                arms_enabled=True,
                hands_enabled=False,
            )
        ),
    )


def _wait_for_emission(source: ThreadedPolicyEmissionSource, timeout: float = 1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        emission = source.step(now=time.monotonic())
        if emission is not None:
            return emission
        time.sleep(0.002)
    raise AssertionError(source.status())


def test_threaded_source_keeps_model_call_off_control_thread_and_paces_chunk() -> None:
    policy = _BlockingPolicy()
    source = _source(policy)
    source.start()
    try:
        started_at = time.perf_counter()
        assert source.step(now=time.monotonic()) is None
        assert time.perf_counter() - started_at < 0.05
        assert policy.entered.wait(timeout=1.0)
        policy.release.set()

        first = _wait_for_emission(source)
        assert first.pre_safety_robot_action is not None
        assert first.pre_safety_robot_action.active_effectors == ("right_arm",)
        assert source.step(now=time.monotonic()) is None
        time.sleep(0.025)
        second = _wait_for_emission(source)
        assert second.chunk_id == first.chunk_id
        assert second.chunk_keyframe_index == first.chunk_keyframe_index + 1
    finally:
        source.close()


def test_activation_generation_discards_late_prediction_before_new_emission() -> None:
    policy = _BlockingPolicy()
    source = _source(policy)
    source.start()
    try:
        assert source.step(now=time.monotonic()) is None
        assert policy.entered.wait(timeout=1.0)
        source.reset_for_activation()
        policy.release.set()

        emission = _wait_for_emission(source)
        status = source.status()
        assert emission.inference_id.endswith(":inference:1")
        assert status["discarded_late_completion_count"] == 1
        assert policy.reset_calls >= 2
    finally:
        source.close()


class _FailingPrefetchPolicy(_BlockingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.fail_entered = threading.Event()
        self.fail_release = threading.Event()

    def predict(self, observation):
        assert tuple(observation.active_sides) == ("left", "right")
        self.calls += 1
        if self.calls == 1:
            return _chunk()
        self.fail_entered.set()
        assert self.fail_release.wait(timeout=2.0)
        raise RuntimeError("fixture prefetch failure")


def test_failed_prefetch_preserves_current_admitted_chunk_tail() -> None:
    policy = _FailingPrefetchPolicy()
    source = _source(
        policy,
        prediction_request_queue_threshold=1,
        policy_interval_sec=0.005,
        action_period_sec=0.05,
    )
    source.start()
    try:
        first = _wait_for_emission(source)
        time.sleep(0.006)
        assert source.step(now=time.monotonic()) is None
        assert policy.fail_entered.wait(timeout=1.0)
        policy.fail_release.set()

        deadline = time.monotonic() + 1.0
        while source.status()["prediction_failure_count"] == 0:
            assert time.monotonic() < deadline
            source.step(now=time.monotonic())
            time.sleep(0.001)

        second = _wait_for_emission(source)
        assert second.chunk_id == first.chunk_id
        assert second.chunk_keyframe_index == 1
        assert source.status()["prediction_failure_count"] >= 1
    finally:
        source.close()


class _DelayedRearmPolicy(_BlockingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.begin_calls = 0

    def request_cancel(self) -> bool:
        return True

    def begin_session(self) -> bool:
        self.begin_calls += 1
        return self.begin_calls >= 3


def test_cancelled_transport_rearm_is_retried_without_blocking_step() -> None:
    policy = _DelayedRearmPolicy()
    source = _source(policy)
    source.start()
    try:
        assert source.step(now=time.monotonic()) is None
        assert policy.entered.wait(timeout=1.0)
        source.reset_for_activation()
        policy.release.set()

        emission = _wait_for_emission(source)
        assert emission is not None
        assert policy.begin_calls >= 3
        assert source.status()["last_error"] is None
    finally:
        source.close()


class _RejectingCloseCancellationPolicy(_BlockingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.allow_cancel = False
        self.close_calls = 0

    def request_cancel(self) -> bool:
        # Model a transport that reports a rejected cancellation after the
        # in-flight request has nevertheless been released.  close() must
        # still fence the mailbox, drain the worker, and close the policy.
        self.release.set()
        return self.allow_cancel

    def close(self) -> None:
        self.close_calls += 1


def test_rejected_close_cancellation_still_drains_worker_and_closes_policy() -> None:
    policy = _RejectingCloseCancellationPolicy()
    source = _source(policy)
    source.start()
    assert source.step(now=time.monotonic()) is None
    assert policy.entered.wait(timeout=1.0)

    try:
        with pytest.raises(RuntimeError, match="cancellation failed"):
            source.close()

        assert source.status()["worker_alive"] is False
        assert source.started is False
        assert policy.close_calls == 1

        # The first call reported the cancellation contract failure, but all
        # owned resources are already closed; retry is idempotent.
        source.close()
        assert policy.close_calls == 1
    finally:
        # Keep the failure-first version of the implementation from leaking a
        # daemon worker while this regression test demonstrates the defect.
        policy.allow_cancel = True
        policy.release.set()
        try:
            source.close()
        except RuntimeError:
            pass


class _SlowClosePolicy(_BlockingPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.close_calls = 0

    def request_cancel(self) -> bool:
        return True

    def close(self) -> None:
        self.close_calls += 1


def test_close_timeout_retains_policy_until_worker_drains_on_retry() -> None:
    policy = _SlowClosePolicy()
    source = _source(policy, worker_join_timeout_sec=0.01)
    source.start()
    assert source.step(now=time.monotonic()) is None
    assert policy.entered.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="did not drain"):
        source.close()

    assert source.status()["worker_alive"] is True
    assert policy.close_calls == 0

    policy.release.set()
    source.close()
    assert source.status()["worker_alive"] is False
    assert policy.close_calls == 1

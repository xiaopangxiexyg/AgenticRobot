"""Offline owner acquisition, interrupted drain and backend release contracts."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from teleop_runtime.collection.hand_state_polling import HandStateHistoryPoller
from teleop_runtime.deploy.impedance_force_guard import ImpedanceForceGuardMonitor
from teleop_runtime.position_control.deploy_executor import (
    PositionDeployExecutor, PositionDeployExecutorConfig,
)
from teleop_runtime.position_control.official_all_ik_governor import RM65OfficialAllIKGovernorPlanner
from teleop_runtime.position_control import official_all_ik_governor as governor_module
from teleop_runtime.position_control import deploy_executor as executor_module
from tests.test_deploy_resource_lifecycle import _lifecycle, _PositionExecutor
from tests.test_official_all_ik_governor import _CollisionPlanner, _CandidateProvider
from tests.test_position_deploy_executor import _pipeline, _filtered_pose, _pose, _observation
from utils.owned_thread import OwnedThread


def _executor(**kwargs):
    return PositionDeployExecutor(config=PositionDeployExecutorConfig(
        stale_after_sec=.5, shutdown_timeout_sec=.03),
        canonical_pipeline=_pipeline(("left",)), **kwargs)


def _planner():
    return RM65OfficialAllIKGovernorPlanner(_CollisionPlanner(),
        continuous_waypoint_precompute_collision_planner=_CollisionPlanner(),
        endpoint_to_tool_by_side={"left": np.eye(4), "right": np.eye(4)},
        candidate_provider=_CandidateProvider(),
        continuous_waypoint_admission_enabled=True,
        continuous_waypoint_async_precompute_enabled=True)


def test_c2_construction_does_not_acquire_a_worker():
    planner = _planner()
    try:
        assert planner._continuous_precompute_thread is None
        planner.start_background_workers()
        assert isinstance(planner._continuous_precompute_thread, OwnedThread)
        planner.start_background_workers()
    finally:
        planner.close()


def test_never_attempted_thread_has_positive_no_acquisition_receipt():
    worker = OwnedThread(target=lambda: None)
    assert worker.join_and_confirm(timeout=0.) is True
    with pytest.raises(RuntimeError, match="abandoned"):
        worker.start()


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_hold_is_pending_until_a_later_close_retries(failure):
    calls = []
    def hold():
        calls.append(1)
        if len(calls) == 1:
            raise failure("first hold failed")
    executor = _executor(hold_callback=hold)
    executor.submit(_filtered_pose("left", _pose(.16)), _observation(), "unit")
    with pytest.raises(RuntimeError, match="first hold failed"):
        executor.close()
    assert executor.snapshot()["hold_retry_required"] is True
    assert executor.snapshot().get("dispatch_workers_stopped", False) is False
    executor.close()
    assert calls == [1, 1]
    assert executor.snapshot()["dispatch_workers_stopped"] is True
    with pytest.raises(RuntimeError, match="closed"):
        executor.start()
    executor.close()
    assert calls == [1, 1]


def test_partial_start_retains_both_owners_and_close_handles_never_started_peer(monkeypatch):
    executor = _executor(dispatch_callback=lambda _command: None,
        discrete_dispatch_callback=lambda _command: None)
    start = threading.Thread.start
    def interrupted_start(thread):
        start(thread)
        if thread.name.startswith("position-deploy-"):
            raise KeyboardInterrupt("after native launch")
    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", interrupted_start)
        with pytest.raises(KeyboardInterrupt, match="after native launch"):
            executor.start()
    try:
        with pytest.raises(KeyboardInterrupt, match="after native launch"):
            executor.start()
    finally:
        executor.close()
    assert executor.snapshot()["dispatch_workers_stopped"] is True


@pytest.mark.parametrize("kind", ["executor", "c2", "impedance_guard", "hand_poller"])
def test_false_is_alive_after_interrupted_join_never_releases_owner(kind, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    def target(*_args):
        entered.set()
        assert release.wait(3.)
    if kind == "executor":
        owner = _executor()
        owner._run = target
        owner.start()
        worker = owner._thread
    elif kind == "c2":
        owner = _planner()
        owner._continuous_waypoint_precompute_loop = target
        owner._continuous_precompute_affinity_ready.set()
        owner.start_background_workers()
        worker = owner._continuous_precompute_thread
    else:
        if kind == "hand_poller":
            owner = HandStateHistoryPoller(backends_by_side={
                "left": SimpleNamespace(get_state=lambda: {})})
            owner._poll_loop = target
        else:
            owner = ImpedanceForceGuardMonitor(readers_by_side={"left": lambda: {}},
                emergency_stop=lambda _fault: None)
            owner._poll_loop = target
            owner._ready_by_side["left"].set()
        owner.start()
        worker = owner._threads[0]
    assert entered.wait(.5)
    real_join = worker.join
    try:
        with monkeypatch.context() as patch:
            def interrupted_join(*_args, **_kwargs):
                raise KeyboardInterrupt("join interrupted")
            patch.setattr(worker, "join", interrupted_join)
            with pytest.raises(BaseException, match="join interrupted"):
                owner.close()
            patch.setattr(worker, "join", lambda *a, **k: None)
            patch.setattr(worker, "is_alive", lambda: False)
            with pytest.raises(BaseException, match="did not stop|shutdown unconfirmed"):
                owner.close()
    finally:
        release.set()
        real_join(timeout=1.)
        owner.close()


def test_planner_start_failure_keeps_owner_and_blocks_readmission():
    calls = []
    failure = KeyboardInterrupt("planner worker acquisition interrupted")
    def start():
        calls.append("start")
        raise failure
    planner = SimpleNamespace(reset=lambda *_: None, plan_pose_reference=lambda *_: None,
        start_background_workers=start,
        shutdown_background_workers=lambda **_: calls.append("stop"))
    executor = _executor(collision_planner_factory=lambda: planner)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt) as raised:
            executor.prepare_collision_planner()
        assert raised.value is failure
    with pytest.raises(KeyboardInterrupt):
        executor.start()
    executor.close()
    assert calls == ["start", "stop"]
    assert executor.snapshot()["dispatch_workers_stopped"] is True


def test_close_racing_factory_retains_returned_owner_before_refusing_start():
    entered, release = threading.Event(), threading.Event()
    calls, errors = [], []
    planner = SimpleNamespace(reset=lambda *_: None, plan_pose_reference=lambda *_: None,
        start_background_workers=lambda: calls.append("start"),
        shutdown_background_workers=lambda **_: calls.append("stop"))
    def factory():
        entered.set()
        assert release.wait(2.)
        return planner
    executor = _executor(collision_planner_factory=factory)
    def prepare():
        try:
            executor.prepare_collision_planner()
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=prepare)
    worker.start()
    assert entered.wait(.5)
    try:
        with pytest.raises(RuntimeError, match="acquisition is still in progress"):
            executor.close()
        assert executor.snapshot()["dispatch_workers_stopped"] is False
    finally:
        release.set()
        worker.join(timeout=1.)
        executor.close()
    assert len(errors) == 1 and "closed while preparing" in str(errors[0])
    assert calls == ["stop"]
    assert executor._collision_planner is planner


def test_waiter_observes_its_failed_attempt_even_after_a_successful_retry(monkeypatch):
    entered, release, waiter_waiting, retry_done = (threading.Event() for _ in range(4))
    calls, errors = [], []
    def hold():
        calls.append(1)
        if len(calls) == 1:
            entered.set()
            assert release.wait(1.)
            raise RuntimeError("first attempt")
    executor = _executor(hold_callback=hold)
    executor.submit(_filtered_pose("left", _pose(.16)), _observation(), "unit")
    event = executor._closed_event
    wait = event.wait
    def delayed_wait(timeout=None):
        waiter_waiting.set()
        assert wait(timeout=1.)
        assert retry_done.wait(1.)
        return True
    monkeypatch.setattr(event, "wait", delayed_wait)
    def capture():
        try:
            executor.close()
        except BaseException as exc:
            errors.append(exc)
    first = threading.Thread(target=capture)
    waiter = threading.Thread(target=capture)
    first.start()
    assert entered.wait(.5)
    waiter.start()
    assert waiter_waiting.wait(.5)
    release.set()
    first.join(timeout=1.)
    try:
        executor.close()
    finally:
        retry_done.set()
        waiter.join(timeout=1.)
    assert len(errors) == 2 and errors[0] is errors[1]
    assert len(calls) == 2


@pytest.mark.parametrize("snapshot", [{}, {"dispatch_workers_stopped": False},
    {"dispatch_workers_stopped": 1}, {"dispatch_workers_stopped": "true"},
    {"reference_worker_alive": False, "discrete_worker_alive": False}, None])
@pytest.mark.parametrize("close_raises", [False, True])
def test_missing_or_invalid_stop_receipt_retains_backends(snapshot, close_raises):
    events = []
    executor = _PositionExecutor(events,
        close_error=RuntimeError("close failed") if close_raises else None)
    executor.snapshot = lambda: snapshot
    lifecycle, _ = _lifecycle(events, position_executor=executor)
    errors = lifecycle.close(keyboard_interrupt_stop_error=None)
    assert "left_arm" not in events and "left_hand" not in events
    assert "fixed_init_restore" not in events
    assert "camera_front" in events and events[-1] == "debug_logger"
    assert any("shutdown is unconfirmed" in error for error in errors)


@pytest.mark.parametrize("failure_site", ["executor_snapshot", "debug_flush", "debug_event", "guard_snapshot"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_diagnostic_failures_cannot_abort_independent_cleanup(failure_site, failure):
    events = []
    lifecycle, logger = _lifecycle(events)
    def fail(*_args):
        raise failure("diagnostic failure")
    if failure_site == "executor_snapshot":
        lifecycle.position_deploy_executor.snapshot = fail
    elif failure_site == "debug_flush":
        lifecycle.flush_position_executor_debug_events = fail
    elif failure_site == "debug_event":
        logger.log_event = fail
    else:
        lifecycle.position_force_guard_monitor.snapshot = fail
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt) as caught:
            lifecycle.close(keyboard_interrupt_stop_error=None)
        errors = caught.value.cleanup_errors
    else:
        errors = lifecycle.close(keyboard_interrupt_stop_error=None)
    assert errors
    assert "camera_front" in events and events[-1] == "debug_logger"
    if failure_site == "executor_snapshot":
        assert "left_arm" not in events and "left_hand" not in events
        assert "fixed_init_restore" not in events
    else:
        assert "left_arm" in events and "left_hand" in events


@pytest.mark.parametrize("resource,blocked,allowed", [
    ("position_force_guard_monitor", "left_arm", "left_hand"),
    ("impedance_force_guard_monitor", "left_arm", "left_hand"),
    ("hand_state_poller", "left_hand", "left_arm"),
])
def test_dependent_owner_close_failure_retains_only_affected_backend(resource, blocked, allowed):
    events = []
    lifecycle, _ = _lifecycle(events)
    def fail():
        raise KeyboardInterrupt("owner drain interrupted")
    getattr(lifecycle, resource).close = fail
    with pytest.raises(KeyboardInterrupt) as caught:
        lifecycle.close(keyboard_interrupt_stop_error=None)
    errors = caught.value.cleanup_errors
    assert blocked not in events and allowed in events
    if blocked == "left_arm":
        assert "fixed_init_restore" not in events
    assert "camera_front" in events and events[-1] == "debug_logger"
    assert any("shutdown is unconfirmed" in error for error in errors)


@pytest.mark.parametrize("fail_at", ["second_queue", "process_constructor"])
def test_c2_partial_queue_acquisition_is_registered_before_next_constructor(fail_at, monkeypatch):
    planner = _planner()
    planner.continuous_waypoint_worker_isolation = "process"
    planner._continuous_waypoint_process_spec = object()
    acquired, closed = [], []
    def make_queue(**_kwargs):
        if fail_at == "second_queue" and acquired:
            raise RuntimeError("queue constructor failed")
        value = SimpleNamespace(cancel_join_thread=lambda: None,
            close=lambda: closed.append(value))
        acquired.append(value)
        return value
    def make_process(**_kwargs):
        raise RuntimeError("process constructor failed")
    monkeypatch.setattr(governor_module.mp, "get_context", lambda *_: SimpleNamespace(
        Queue=make_queue, Process=make_process))
    try:
        with pytest.raises(RuntimeError, match="constructor failed"):
            planner.start_background_workers()
    finally:
        planner.close()
    assert len(acquired) == (1 if fail_at == "second_queue" else 2)
    assert closed == acquired


@pytest.mark.parametrize("fail_on", [1, 2])
def test_executor_constructor_failure_is_sticky_and_never_started_handles_are_closed(fail_on, monkeypatch):
    executor = _executor(dispatch_callback=lambda _: None,
        discrete_dispatch_callback=lambda _: None)
    failure = KeyboardInterrupt("worker constructor interrupted")
    created = []
    original = executor_module.OwnedThread
    def make(*args, **kwargs):
        if len(created) + 1 == fail_on:
            raise failure
        worker = original(*args, **kwargs)
        created.append(worker)
        return worker
    monkeypatch.setattr(executor_module, "OwnedThread", make)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt) as caught:
            executor.start()
        assert caught.value is failure
    executor.close()
    assert executor.shutdown_snapshot()["dispatch_workers_stopped"] is True


def test_planner_start_port_requires_a_drain_port_before_acquisition():
    calls = []
    planner = SimpleNamespace(reset=lambda *_: None, plan_pose_reference=lambda *_: None,
        start_background_workers=lambda: calls.append("started"))
    executor = _executor(collision_planner_factory=lambda: planner)
    with pytest.raises(TypeError, match="requires shutdown_background_workers"):
        executor.prepare_collision_planner()
    executor.close()
    assert calls == []


def test_c2_close_fences_future_start_even_if_acquisition_gate_times_out():
    planner = _planner()
    held, release = threading.Event(), threading.Event()
    def hold_gate():
        with planner._background_lifecycle_lock:
            held.set()
            assert release.wait(1.)
    worker = threading.Thread(target=hold_gate)
    worker.start()
    assert held.wait(.5)
    try:
        with pytest.raises(RuntimeError, match="acquisition is still in progress"):
            planner.shutdown_background_workers(timeout_sec=.01)
    finally:
        release.set()
        worker.join(timeout=1.)
    with pytest.raises(RuntimeError, match="closed"):
        planner.start_background_workers()
    planner.close()


def test_stop_receipt_never_reads_full_diagnostics_or_projector(monkeypatch):
    executor = _executor()
    def fail():
        raise AssertionError("full diagnostics must not determine stop proof")
    monkeypatch.setattr(executor, "snapshot", fail)
    executor.close()
    assert executor.shutdown_snapshot()["dispatch_workers_stopped"] is True


def test_cleanup_preserves_exit_object_and_continues_when_stderr_is_broken(monkeypatch):
    events = []
    lifecycle, _ = _lifecycle(events)
    failure = SystemExit(7)
    def fail():
        raise failure
    class BrokenStderr:
        def write(self, _text):
            raise OSError("stderr unavailable")
        def flush(self):
            pass
    lifecycle.impedance_force_guard_monitor.close = fail
    monkeypatch.setattr("teleop_runtime.deploy.resource_lifecycle.sys.stderr", BrokenStderr())
    with pytest.raises(SystemExit) as caught:
        lifecycle.close(keyboard_interrupt_stop_error=None)
    assert caught.value is failure
    assert "left_arm" not in events and "left_hand" in events
    assert "camera_front" in events and events[-1] == "debug_logger"
    assert any("cleanup_stderr" in error for error in caught.value.cleanup_errors)


@pytest.mark.parametrize("consumer", ["prepare_first_submission", "submit"])
def test_concurrent_consumer_cannot_use_owned_but_unready_planner(consumer):
    entered, release, consumer_done = threading.Event(), threading.Event(), threading.Event()
    failure = RuntimeError("planner startup failed")
    calls, errors = [], []
    def start():
        entered.set()
        assert release.wait(1.)
        raise failure
    planner = SimpleNamespace(reset=lambda *_: calls.append("reset"),
        plan_pose_reference=lambda *_: None, start_background_workers=start,
        shutdown_background_workers=lambda **_: calls.append("stop"))
    executor = PositionDeployExecutor(config=PositionDeployExecutorConfig(
        stale_after_sec=.5, shutdown_timeout_sec=.03, collision_required=True),
        canonical_pipeline=_pipeline(("left", "right")),
        collision_planner_factory=lambda: planner)
    def prepare():
        try:
            executor.prepare_collision_planner()
        except BaseException as exc:
            errors.append(exc)
    def consume():
        try:
            if consumer == "submit":
                executor.submit(_filtered_pose("left", _pose(.16)), _observation(), "unit")
            else:
                executor.prepare_first_submission()
        except BaseException as exc:
            errors.append(exc)
        finally:
            consumer_done.set()
    first, second = threading.Thread(target=prepare), threading.Thread(target=consume)
    first.start()
    assert entered.wait(.5)
    second.start()
    try:
        assert not consumer_done.wait(.03)
    finally:
        release.set()
        first.join(timeout=1.)
        second.join(timeout=1.)
        executor.close()
    assert len(errors) == 2 and all(exc is failure for exc in errors)
    assert calls == ["stop"]


def test_close_fences_admission_without_waiting_for_a_blocked_state_owner():
    executor = _executor()
    held, release = threading.Event(), threading.Event()
    def hold_state():
        with executor._lock:
            held.set()
            assert release.wait(1.)
    worker = threading.Thread(target=hold_state)
    worker.start()
    assert held.wait(.5)
    started = time.monotonic()
    try:
        with pytest.raises(RuntimeError, match="state barrier"):
            executor.close()
        assert time.monotonic() - started < .4
        assert executor.shutdown_snapshot()["dispatch_workers_stopped"] is False
        assert executor._closed is True
    finally:
        release.set()
        worker.join(timeout=1.)
        executor.close()
    assert executor.shutdown_snapshot()["dispatch_workers_stopped"] is True


def test_c2_ipc_close_failure_after_child_exit_is_retryable_without_notifying_closed_queue():
    planner = _planner()
    puts, joins = [], []
    class Queue:
        closed = False
        failed = False
        def put_nowait(self, value):
            assert not self.closed, "closed queue must not be notified again"
            puts.append(value)
        def cancel_join_thread(self):
            pass
        def close(self):
            self.closed = True
            if not self.failed:
                self.failed = True
                raise KeyboardInterrupt("after queue closed")
    first, second = Queue(), Queue()
    process = SimpleNamespace(_popen=object(), is_alive=lambda: False, exitcode=0,
        join=lambda **_: joins.append(1))
    planner._continuous_precompute_process = process
    planner._continuous_precompute_process_launch_attempted = True
    planner._continuous_precompute_process_request_queue = first
    planner._continuous_precompute_process_result_queue = second
    try:
        for _ in range(2):
            with pytest.raises(KeyboardInterrupt, match="after queue closed"):
                planner.close()
        planner.close()
    finally:
        first.failed = second.failed = True
        planner.close()
    assert first.closed and second.closed and len(puts) == len(joins) == 1

from __future__ import annotations

import threading

import pytest

from robot_motion.runtime.lifecycle import (
    LifecycleCleanupError,
    LifecycleStartupError,
    run_best_effort_cleanup,
)
from robot_motion.runtime.rig import BimanualRig
from collection_runtime.async_writer import (
    AsyncWriterConfig,
    AsyncWriterPool,
    WriteJob,
)
from teleop_runtime.collection.arm_state_polling import ArmHardwareStatePoller
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.observers import CollectionWriterObserver
from teleop_runtime.collection.runtime import CollectionRuntime
from teleop_runtime.session import TeleopSession, TeleopSessionConfig
from teleop_runtime.recording import RecordingConfig, SessionRecorder
from teleop_runtime.sources.camera.realsense_rgb import (
    RealSenseRGBCameraConfig,
    RealSenseRGBSource,
)


class _LifecycleResource:
    def __init__(
        self,
        name: str,
        calls: list[str],
        *,
        fail_start: bool = False,
        fail_close: bool = False,
    ) -> None:
        self._name = str(name)
        self.calls = calls
        self.fail_start = bool(fail_start)
        self.fail_close = bool(fail_close)

    @property
    def name(self) -> str:
        return self._name

    def start(self) -> None:
        self.calls.append(f"start:{self.name}")
        if self.fail_start:
            raise RuntimeError(f"start boom {self.name}")

    def close(self) -> None:
        self.calls.append(f"close:{self.name}")
        if self.fail_close:
            raise RuntimeError(f"close boom {self.name}")

    def get_latest_channels(self):
        return {}

    def is_ready(self) -> bool:
        return True

    def send_command(self, _command):
        return None

    def send_commands(self, _command, phase_recorder=None):
        del phase_recorder
        return {}


def test_lifecycle_cleanup_attempts_every_step_and_preserves_labels() -> None:
    calls: list[str] = []

    def step(name: str, *, fail: bool = False):
        def run() -> None:
            calls.append(name)
            if fail:
                raise RuntimeError(f"{name} failed")

        return run

    with pytest.raises(LifecycleCleanupError) as caught:
        run_best_effort_cleanup(
            "unit.shutdown",
            (
                ("camera", step("camera", fail=True)),
                ("writer", step("writer")),
                ("arm", step("arm", fail=True)),
            ),
        )

    assert calls == ["camera", "writer", "arm"]
    assert [failure.label for failure in caught.value.failures] == ["camera", "arm"]
    assert "camera failed" in str(caught.value)
    assert "arm failed" in str(caught.value)
    assert caught.value.primary_error is None
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert str(caught.value.__cause__) == "camera failed"


def test_cleanup_failure_in_finally_explicitly_preserves_primary_exception() -> None:
    primary = ValueError("primary operation boom")
    calls: list[str] = []

    def fail_cleanup(name: str):
        def run() -> None:
            calls.append(name)
            raise RuntimeError(f"cleanup boom {name}")

        return run

    def run_with_finally() -> None:
        try:
            raise primary
        finally:
            run_best_effort_cleanup(
                "unit.finally",
                (
                    ("writer", fail_cleanup("writer")),
                    ("camera", fail_cleanup("camera")),
                ),
            )

    with pytest.raises(LifecycleCleanupError) as caught:
        run_with_finally()

    assert calls == ["writer", "camera"]
    assert caught.value.primary_error is primary
    assert caught.value.__cause__ is primary
    assert [failure.label for failure in caught.value.failures] == [
        "writer",
        "camera",
    ]
    assert "primary ValueError: primary operation boom" in str(caught.value)
    assert "cleanup boom writer" in str(caught.value)
    assert "cleanup boom camera" in str(caught.value)


def test_cleanup_failure_in_exit_explicitly_preserves_primary_exception() -> None:
    primary = LookupError("context body boom")

    def close_resource() -> None:
        raise OSError("close boom")

    class ClosingContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            del exc_type, traceback
            assert exc is primary
            run_best_effort_cleanup(
                "unit.__exit__",
                (("resource", close_resource),),
            )

    def run_with_context() -> None:
        with ClosingContext():
            raise primary

    with pytest.raises(LifecycleCleanupError) as caught:
        run_with_context()

    assert caught.value.primary_error is primary
    assert caught.value.__cause__ is primary
    assert [failure.label for failure in caught.value.failures] == ["resource"]
    assert isinstance(caught.value.failures[0].error, OSError)


def test_bimanual_start_rollback_is_reverse_order_and_keeps_primary() -> None:
    calls: list[str] = []
    left = _LifecycleResource("left", calls, fail_close=True)
    right = _LifecycleResource("right", calls, fail_start=True)
    rig = BimanualRig({"left_arm": left, "right_arm": right})

    with pytest.raises(LifecycleStartupError) as caught:
        rig.start()

    assert calls == [
        "start:left",
        "start:right",
        "close:right",
        "close:left",
    ]
    assert str(caught.value.start_error) == "start boom right"
    assert [failure.label for failure in caught.value.rollback_failures] == [
        "backend[left_arm]"
    ]


def test_bimanual_close_attempts_all_backends_in_reverse_order() -> None:
    calls: list[str] = []
    left = _LifecycleResource("left", calls, fail_close=True)
    right = _LifecycleResource("right", calls, fail_close=True)
    rig = BimanualRig({"left_arm": left, "right_arm": right})

    with pytest.raises(LifecycleCleanupError) as caught:
        rig.close()

    assert calls == ["close:right", "close:left"]
    assert [failure.label for failure in caught.value.failures] == [
        "backend[right_arm]",
        "backend[left_arm]",
    ]


def test_teleop_session_start_rolls_back_attempted_sources() -> None:
    calls: list[str] = []
    first = _LifecycleResource("first", calls, fail_close=True)
    second = _LifecycleResource("second", calls, fail_start=True)
    session = TeleopSession(
        TeleopSessionConfig(record_enabled=False),
        sources=(first, second),
    )

    with pytest.raises(LifecycleStartupError) as caught:
        session.start()

    assert calls == [
        "start:first",
        "start:second",
        "close:second",
        "close:first",
    ]
    assert str(caught.value.start_error) == "start boom second"
    assert [failure.label for failure in caught.value.rollback_failures] == [
        "source[first]"
    ]
    assert not session._started


def test_teleop_session_close_attempts_recorder_rig_and_sources_in_reverse_order() -> None:
    calls: list[str] = []
    first = _LifecycleResource("first", calls, fail_close=True)
    second = _LifecycleResource("second", calls, fail_close=True)
    rig = _LifecycleResource("rig", calls, fail_close=True)
    recorder = _LifecycleResource("recorder", calls, fail_close=True)
    session = TeleopSession(
        TeleopSessionConfig(record_enabled=False),
        sources=(first, second),
        rig=rig,
        recorder=recorder,
    )
    session._started = True

    with pytest.raises(LifecycleCleanupError) as caught:
        session.close()

    assert calls == [
        "close:recorder",
        "close:rig",
        "close:second",
        "close:first",
    ]
    assert [failure.label for failure in caught.value.failures] == [
        "recorder",
        "rig[rig]",
        "source[second]",
        "source[first]",
    ]
    assert not session._started


def test_async_writer_failure_still_stops_every_worker() -> None:
    handler_called = threading.Event()

    def failing_handler(_job: WriteJob) -> None:
        handler_called.set()
        raise RuntimeError("handler boom")

    pool = AsyncWriterPool(
        failing_handler,
        config=AsyncWriterConfig(worker_count=3, max_queue_size=8),
        name="fault_injected_writer",
    )
    pool.submit(WriteJob(kind="row", payload={}))
    assert handler_called.wait(timeout=1.0)

    with pytest.raises(LifecycleCleanupError) as caught:
        pool.close(wait=True)

    assert "handler_failures" in str(caught.value)
    assert "handler boom" in str(caught.value)
    assert pool._threads == []
    assert not pool._started
    assert not any(
        thread.name.startswith("fault_injected_writer_") and thread.is_alive()
        for thread in threading.enumerate()
    )


class _FailingSession:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def close(self) -> None:
        self.calls.append("session")
        raise RuntimeError("session close boom")


class _FailingWriterPool:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def close(self, *, wait=True) -> None:
        self.calls.append(f"writer:{wait}")
        raise RuntimeError("writer close boom")


def test_collection_runtime_closes_writer_even_when_session_close_fails() -> None:
    calls: list[str] = []
    session = _FailingSession(calls)
    writer_pool = _FailingWriterPool(calls)
    runtime = CollectionRuntime(
        components=CollectorComponents(session=session),
        sample_builder=lambda _result, _components: None,
        write_planner=lambda _sample, _components: (),
        writer_pool=writer_pool,
    )
    runtime._started = True

    with pytest.raises(LifecycleCleanupError) as caught:
        runtime.close()

    assert calls == ["session", "writer:True"]
    assert [failure.label for failure in caught.value.failures] == [
        "session",
        "writer_observer",
    ]
    assert "session close boom" in str(caught.value)
    assert "writer close boom" in str(caught.value)
    assert not runtime._started


class _FailingRoute:
    def __init__(self, name: str, calls: list[str]) -> None:
        self.name = name
        self.calls = calls

    def close(self, *, wait=True) -> None:
        self.calls.append(f"route:{self.name}:{wait}")
        raise RuntimeError(f"route {self.name} boom")


def test_writer_observer_closes_all_routes_then_pool_despite_failures() -> None:
    calls: list[str] = []
    writer_pool = _FailingWriterPool(calls)
    observer = CollectionWriterObserver(
        components=CollectorComponents(session=object()),
        sample_builder=lambda _result, _components: None,
        write_planner=lambda _sample, _components: (),
        writer_pool=writer_pool,
    )
    observer._storage_routes = {
        "front": _FailingRoute("front", calls),
        "wrist": _FailingRoute("wrist", calls),
    }

    with pytest.raises(LifecycleCleanupError) as caught:
        observer.close(flush=True)

    assert calls == [
        "route:wrist:True",
        "route:front:True",
        "writer:True",
    ]
    assert [failure.label for failure in caught.value.failures] == [
        "storage_route[wrist]",
        "storage_route[front]",
        "writer_pool",
    ]


def test_realsense_close_joins_worker_even_when_pipeline_stop_fails() -> None:
    calls: list[str] = []

    class Pipeline:
        def stop(self) -> None:
            calls.append("pipeline.stop")
            raise RuntimeError("pipeline stop boom")

    class Worker:
        name = "camera-worker"

        def join(self, timeout=None) -> None:
            calls.append(f"worker.join:{timeout}")

        def is_alive(self) -> bool:
            return False

    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(serial="unit", camera_key="front")
    )
    source._pipeline = Pipeline()
    source._thread = Worker()

    with pytest.raises(LifecycleCleanupError) as caught:
        source.close()

    assert calls[0] == "pipeline.stop"
    assert len(calls) == 2
    join_timeout_sec = float(calls[1].removeprefix("worker.join:"))
    assert join_timeout_sec >= source.config.wait_timeout_ms / 1000.0
    assert join_timeout_sec <= source.config.wait_timeout_ms / 1000.0 + 0.5
    assert [failure.label for failure in caught.value.failures] == ["pipeline.stop"]
    assert source._thread is None


def test_realsense_close_timeout_faults_and_blocks_new_worker_generation() -> None:
    class Worker:
        name = "stuck-camera-worker"

        def __init__(self) -> None:
            self.alive = True

        def join(self, timeout=None) -> None:
            del timeout

        def is_alive(self) -> bool:
            return self.alive

    source = RealSenseRGBSource(
        RealSenseRGBCameraConfig(serial="unit", camera_key="front")
    )
    worker = Worker()
    source._thread = worker
    source._running = True
    source._lifecycle_state = "running"

    with pytest.raises(LifecycleCleanupError, match="join_timeout"):
        source.close()

    assert source.device_info()["lifecycle_state"] == "faulted"
    assert source.device_info()["worker_alive"] is True
    with pytest.raises(RuntimeError, match="prior worker.*still alive"):
        source.start()

    worker.alive = False
    source.close()
    assert source.device_info()["lifecycle_state"] == "stopped"
    assert source._thread is None


def test_session_recorder_closes_every_handle_after_flush_failures() -> None:
    calls: list[str] = []

    class Handle:
        def __init__(self, name: str, *, fail_flush: bool = False) -> None:
            self.name = name
            self.fail_flush = fail_flush

        def flush(self) -> None:
            calls.append(f"flush:{self.name}")
            if self.fail_flush:
                raise OSError(f"flush boom {self.name}")

        def close(self) -> None:
            calls.append(f"close:{self.name}")

    recorder = SessionRecorder(config=RecordingConfig(enabled=False))
    recorder._handles = {
        "first": Handle("first"),
        "second": Handle("second", fail_flush=True),
    }

    with pytest.raises(LifecycleCleanupError) as caught:
        recorder.close()

    assert calls == [
        "flush:second",
        "close:second",
        "flush:first",
        "close:first",
    ]
    assert [failure.label for failure in caught.value.failures] == [
        "stream[second].flush"
    ]
    assert recorder._handles == {}


def test_arm_state_poller_close_attempts_threads_in_reverse_and_keeps_live_worker() -> None:
    calls: list[str] = []

    class Worker:
        def __init__(self, name: str, *, alive: bool) -> None:
            self.name = name
            self.alive = alive

        def join(self, timeout=None) -> None:
            calls.append(f"join:{self.name}:{timeout}")

        def is_alive(self) -> bool:
            return self.alive

    first = Worker("first", alive=False)
    second = Worker("second", alive=True)
    poller = ArmHardwareStatePoller(backends_by_side={})
    poller._threads = [first, second]

    with pytest.raises(LifecycleCleanupError) as caught:
        poller.close()

    assert calls == ["join:second:1.0", "join:first:1.0"]
    assert [failure.label for failure in caught.value.failures] == [
        "worker[second]"
    ]
    assert poller._threads == [second]


def test_realsense_native_stop_timeout_cannot_be_forgotten_by_second_close(monkeypatch) -> None:
    release, finished = threading.Event(), threading.Event()
    calls = []
    class Pipeline:
        def stop(self):
            calls.append("stop")
            try:
                release.wait(3)
            finally:
                finished.set()
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="unit", camera_key="front"))
    source._pipeline = Pipeline()
    original = source._stop_pipeline_bounded
    monkeypatch.setattr(source, "_stop_pipeline_bounded", lambda **kw: original(timeout_sec=.02))
    try:
        with pytest.raises(LifecycleCleanupError, match="pipeline.stop"):
            source.close()
        # The capture thread may be gone while native SDK cleanup is still active.
        with pytest.raises(LifecycleCleanupError, match="pipeline.stop"):
            source.close()
        assert calls == ["stop"]
        assert source.device_info()["lifecycle_state"] == "faulted"
        assert source.device_info()["pipeline_stop"]["worker_alive"] is True
        with pytest.raises(RuntimeError, match="native pipeline stop"):
            source.start()
    finally:
        release.set()
        assert finished.wait(1)
    assert source.wait_for_pending_pipeline_stop(timeout_sec=1) is None
    source.close()
    assert calls == ["stop"]
    assert source.device_info()["lifecycle_state"] == "stopped"


def test_realsense_native_handle_release_is_part_of_owned_stop(monkeypatch) -> None:
    release, entered = threading.Event(), threading.Event()
    class Pipeline:
        def stop(self):
            pass
        def __del__(self):
            entered.set()
            release.wait(3)
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="unit", camera_key="front"))
    source._pipeline = Pipeline()
    original = source._stop_pipeline_bounded
    monkeypatch.setattr(source, "_stop_pipeline_bounded", lambda **kw: original(timeout_sec=.02))
    try:
        with pytest.raises(LifecycleCleanupError, match="pipeline.stop"):
            source.close()
        assert entered.is_set()
        assert source.device_info()["pipeline_stop"]["phase"] == "releasing_native_handle"
    finally:
        release.set()
    assert source.wait_for_pending_pipeline_stop(timeout_sec=1) is None
    source.close()


def test_realsense_native_stop_error_remains_faulted_after_worker_returns() -> None:
    calls = []
    class Pipeline:
        def stop(self):
            calls.append("stop")
            raise RuntimeError("native failure")
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="unit", camera_key="front"))
    source._pipeline = Pipeline()
    for _ in range(2):
        with pytest.raises(LifecycleCleanupError, match="native failure"):
            source.close()
    assert calls == ["stop"]
    assert source.device_info()["lifecycle_state"] == "faulted"
    assert source.device_info()["pipeline_stop"]["worker_alive"] is False


def test_realsense_explicit_native_stop_budget_does_not_change_capture_join_budget() -> None:
    budgets = []
    class Pipeline:
        def stop(self):
            pass
    class Worker:
        name = "capture"
        def join(self, timeout=None):
            budgets.append(timeout)
        def is_alive(self):
            return False
    source = RealSenseRGBSource(RealSenseRGBCameraConfig(serial="unit", pipeline_stop_timeout_sec=5))
    source._pipeline = Pipeline()
    source._thread = Worker()
    original = source._stop_pipeline_bounded
    def stop(**kwargs):
        budgets.append(kwargs["timeout_sec"])
        return original(**kwargs)
    source._stop_pipeline_bounded = stop
    source.close()
    assert budgets == [5, 1.5]


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_realsense_native_stop_budget_rejects_invalid_config(budget) -> None:
    with pytest.raises(ValueError, match="pipeline_stop_timeout_sec"):
        RealSenseRGBCameraConfig(serial="unit", pipeline_stop_timeout_sec=budget)



@pytest.mark.parametrize("watchdog_fails", (False, True))
def test_collection_pending_shutdown_retains_dependencies_until_owner_drains(monkeypatch, watchdog_fails):
    from types import SimpleNamespace

    from teleop_runtime.real_teleop import shutdown_runtime as shutdown

    calls, active = [], [True]
    def stop_watchdog():
        calls.append("watchdog")
        if watchdog_fails:
            raise RuntimeError("watchdog unavailable")
    state = {name: None for name in (
        "startup_progress", "position_force_guard_monitor", "dataset_recorder", "camera_previewer",
        "tactile_visualizer", "actuator_trace_recorder", "position_debug_session_recorder",
        "position_episode_debug_runtime", "original_stdout", "silenced_stdout",
    )}
    state.update(args=SimpleNamespace(),
        episode_recording_runtime=SimpleNamespace(collection_transition_active=lambda: active[0]),
        main_loop_watchdog=SimpleNamespace(stop=stop_watchdog),
        episode_recorder=_LifecycleResource("episode", calls),
        clutch_input=_LifecycleResource("clutch", calls), camera_sources={"main": object()},
        components=SimpleNamespace(session=_LifecycleResource("session", calls),
            human_shadow_frontend=None, arm_state_read_executor=None,
            left_hand_backend=None, right_hand_backend=None, left_state_backend=None, right_state_backend=None),
        position_audit_fanout=[], position_debug_close_threads=[],
        update_control_debug_runtime_metadata=lambda **_: None)
    monkeypatch.setattr(shutdown, "_arm_control_mode_arg", lambda _: "position")
    monkeypatch.setattr(shutdown, "_terminal_force_collision_summary", lambda *_, **__: {})
    monkeypatch.setattr(shutdown, "_close_camera_sources", lambda _: calls.append("cameras"))
    for name in ("_attach_position_audit_sink", "_attach_ik_shadow_trace_sink", "_attach_singularity_transition_trace_sink"):
        monkeypatch.setattr(shutdown, name, lambda *_, **__: None)
    with pytest.raises(shutdown.CollectionShutdownPendingError) as error:
        shutdown.shutdown_real_teleop_runtime(state, terminal_error=None)
    assert calls == ["watchdog"]
    assert error.value.runtime_state["components"] is state["components"]
    assert error.value.runtime_state["camera_sources"] is state["camera_sources"]
    if watchdog_fails:
        assert isinstance(error.value.watchdog_error, RuntimeError)
    active[0] = False
    if watchdog_fails:
        with pytest.raises(LifecycleCleanupError):
            shutdown.shutdown_real_teleop_runtime(error.value.runtime_state, terminal_error=None)
    else:
        shutdown.shutdown_real_teleop_runtime(error.value.runtime_state, terminal_error=None)
    for name in ("close:episode", "close:clutch", "cameras", "close:session"):
        assert calls.count(name) == 1


@pytest.mark.parametrize("primary_kind", ("none", "runtime", "interrupt", "emergency"))
def test_pending_collection_shutdown_preserves_primary_exit_and_resource_owner(monkeypatch, primary_kind):
    from teleop_runtime.real_teleop import runtime
    from teleop_runtime.real_teleop.shutdown_runtime import (
        CollectionShutdownPendingError,
    )

    resource = object()
    pending = CollectionShutdownPendingError({"resource": resource})
    def fail(*_, **__):
        raise pending
    monkeypatch.setattr(runtime, "shutdown_real_teleop_runtime", fail)
    primary = KeyboardInterrupt() if primary_kind == "interrupt" else RuntimeError("original")
    terminal = primary if primary_kind in ("runtime", "interrupt") else None
    emergency = primary if primary_kind == "emergency" else None
    if primary_kind == "none":
        with pytest.raises(CollectionShutdownPendingError) as error:
            runtime._shutdown_resources_preserving_exit({}, terminal_error=terminal, emergency_stop_error=emergency)
        assert error.value is pending
    else:
        runtime._shutdown_resources_preserving_exit({}, terminal_error=terminal, emergency_stop_error=emergency)
        assert primary.collection_shutdown_pending is pending
        assert primary.collection_shutdown_pending.runtime_state["resource"] is resource

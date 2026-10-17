from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop import collection_control_plane as control_plane_module
from teleop_runtime.real_teleop import dataset_recorder as dataset_recorder_module
from teleop_runtime.real_teleop import (
    episode_recording_runtime as episode_recording_module,
)
from teleop_runtime.real_teleop.collection_control_plane import (
    CollectionControlOperationBusy,
    CollectionControlPlane,
)
from teleop_runtime.real_teleop.dataset_finalize import RealDatasetFinalizeTask
from teleop_runtime.real_teleop.dataset_recorder import RealTeleopDatasetRecorder
from tests.support.real_teleop_recording import (
    _Clutch,
    _episode_runtime,
    poll_recording_transition,
)


def _wait_for_result(
    plane: CollectionControlPlane,
    generation: int,
    *,
    timeout_sec: float = 1.0,
):
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        result = plane.poll(generation)
        if result is not None:
            return result
        time.sleep(0.001)
    raise AssertionError(f"operation {generation} did not complete")


class _PauseWorkerAfterUnlock:
    """Use a real lock, with one legal scheduling cut after worker release."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.worker: threading.Thread | None = None
        self.reached = threading.Event()
        self.release = threading.Event()

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *_exc) -> None:
        self.lock.release()
        if threading.current_thread() is self.worker and not self.reached.is_set():
            self.reached.set()
            assert self.release.wait(timeout=2.0)


def test_submit_never_waits_for_blocked_operation() -> None:
    entered = threading.Event()
    release = threading.Event()
    plane = CollectionControlPlane(name="unit-test")

    def blocked() -> str:
        entered.set()
        assert release.wait(timeout=1.0)
        return "ready"

    started_at = time.perf_counter()
    generation = plane.submit("prepare", blocked)
    submit_elapsed = time.perf_counter() - started_at

    assert submit_elapsed < 0.05
    assert entered.wait(timeout=1.0)
    assert plane.busy
    assert plane.active_operation == (generation, "prepare")
    release.set()
    result = _wait_for_result(plane, generation)
    assert result.succeeded
    assert result.value == "ready"
    assert result.kind == "prepare"
    assert result.elapsed_sec >= 0.0
    assert plane.active_operation is None


def test_single_flight_rejects_overlapping_operation() -> None:
    entered = threading.Event()
    release = threading.Event()
    plane = CollectionControlPlane(name="unit-test")

    generation = plane.submit(
        "prepare",
        lambda: (entered.set(), release.wait(timeout=1.0)),
    )
    assert entered.wait(timeout=1.0)

    with pytest.raises(CollectionControlOperationBusy, match="prepare"):
        plane.submit("stop", lambda: None)

    release.set()
    assert _wait_for_result(plane, generation).succeeded


def test_submit_thread_start_failure_does_not_wedge_single_flight(
    monkeypatch,
) -> None:
    plane = CollectionControlPlane(name="unit-test")
    original_start = control_plane_module.threading.Thread.start

    def fail_start(_thread) -> None:
        raise RuntimeError("simulated control-plane thread exhaustion")

    monkeypatch.setattr(control_plane_module.threading.Thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="simulated control-plane thread exhaustion"):
        plane.submit("prepare", lambda: "unreachable")

    assert not plane.busy
    assert plane.active_kind is None
    assert plane.poll() is None

    monkeypatch.setattr(
        control_plane_module.threading.Thread,
        "start",
        original_start,
    )
    generation = plane.submit("retry", lambda: "ready")
    result = _wait_for_result(plane, generation)
    assert result.succeeded
    assert result.value == "ready"


def test_worker_failure_is_polled_without_escaping_worker_thread() -> None:
    plane = CollectionControlPlane(name="unit-test")

    def fail() -> None:
        raise RuntimeError("writer ready timeout")

    generation = plane.submit("prepare", fail)
    result = _wait_for_result(plane, generation)

    assert not result.succeeded
    assert isinstance(result.error, RuntimeError)
    assert str(result.error) == "writer ready timeout"
    assert plane.poll(generation) is None


def test_finalize_thread_start_failure_publishes_one_joinable_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    task = RealDatasetFinalizeTask(
        episode_index=4,
        frame_count=17,
        reason="unit",
        queued_at=time.time(),
    )
    recorder.finalize_tasks = [task]

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("simulated finalize thread exhaustion")

    monkeypatch.setattr(
        dataset_recorder_module.threading.Thread,
        "start",
        fail_start,
    )

    assert not recorder._start_finalize_task(task)
    assert task.done
    assert not task.success
    assert task.thread is None
    assert "simulated finalize thread exhaustion" in task.error
    assert recorder.poll_finalize_tasks() == [task]
    assert recorder.finalize_tasks == []


def test_finalize_thread_construction_failure_is_a_consumable_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    task = RealDatasetFinalizeTask(
        episode_index=5,
        frame_count=9,
        reason="unit",
        queued_at=time.time(),
    )
    recorder.finalize_tasks = [task]

    def fail_thread_construction(**_kwargs: object) -> threading.Thread:
        raise RuntimeError("simulated finalize thread construction failure")

    monkeypatch.setattr(
        dataset_recorder_module.threading,
        "Thread",
        fail_thread_construction,
    )

    assert not recorder._start_finalize_task(task)
    assert task.done
    assert task.thread is None
    assert "thread construction failure" in task.error
    assert recorder.poll_finalize_tasks() == [task]


def test_thread_writer_pause_never_waits_for_inflight_sample_and_resume_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PendingSample:
        settled = False
        result_calls = 0

        def done(self) -> bool:
            return bool(self.settled)

        def result(self) -> bool:
            self.result_calls += 1
            if not self.settled:
                raise AssertionError("pause must not wait for the sample")
            return True

    pending = PendingSample()
    temporal_resets: list[str] = []
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.collection_worker_mode = "thread"
    recorder.active_episode_index = 3
    recorder.sample_builder = SimpleNamespace(
        frame_index=11,
        reset_temporal_history=lambda: temporal_resets.append("reset"),
    )
    recorder._sample_future = pending
    recorder._sample_overload_reason = None
    recorder._sample_modality_reason = None
    recorder._collection_integrity_reasons = []
    recorder._pause_started_at = None
    recorder._pause_started_monotonic = None
    recorder._pause_reason = None
    recorder._pause_reasons = []
    recorder._pause_started_frame_index = None
    recorder._pause_started_step_index = None
    recorder._pause_started_last_state = None
    recorder._pause_segments = []
    recorder._paused_accumulated_sec = 0.0
    recorder._record_timestamps = []
    recorder._continuity_segment_index = 0

    monotonic_calls = iter((10.0, 10.1, 10.2))
    monkeypatch.setattr(dataset_recorder_module.time, "monotonic", lambda: next(monotonic_calls))

    assert recorder.pause_episode(
        reason="hold_requested",
        now=100.0,
        step_index=21,
        last_state="RUNNING",
    )
    assert recorder.is_paused
    assert recorder._sample_future is pending
    assert pending.result_calls == 0

    assert not recorder.resume_episode(
        reason="runtime_running_warmup_ready",
        now=101.0,
        step_index=22,
    )
    assert recorder.is_paused
    assert temporal_resets == []
    assert pending.result_calls == 0

    pending.settled = True
    recorder.sample_builder.frame_index = 12
    assert recorder.resume_episode(
        reason="runtime_running_warmup_ready",
        now=102.0,
        step_index=23,
    )
    assert not recorder.is_paused
    assert recorder._sample_future is None
    assert pending.result_calls == 1
    assert temporal_resets == ["reset"]
    assert recorder.continuity_segment_index == 1
    assert recorder._pause_segments[-1]["start_frame_index"] == 12


def test_thread_writer_resume_stays_paused_after_inflight_sample_failure() -> None:
    class FailedSample:
        def done(self) -> bool:
            return True

        def result(self) -> None:
            raise RuntimeError("writer queue failed")

    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.collection_worker_mode = "thread"
    recorder.active_episode_index = 3
    recorder.sample_builder = SimpleNamespace(
        frame_index=11,
        reset_temporal_history=lambda: None,
    )
    recorder._sample_future = FailedSample()
    recorder._sample_overload_reason = None
    recorder._sample_modality_reason = None
    recorder._collection_integrity_reasons = []
    recorder._pause_started_at = 100.0
    recorder._pause_started_monotonic = time.monotonic()
    recorder._pause_reason = "hold_requested"
    recorder._pause_reasons = ["hold_requested"]
    recorder._pause_started_frame_index = 11
    recorder._pause_started_step_index = 21
    recorder._pause_started_last_state = "RUNNING"
    recorder._pause_segments = []
    recorder._paused_accumulated_sec = 0.0
    recorder._record_timestamps = []
    recorder._continuity_segment_index = 0

    with pytest.raises(RuntimeError, match="uncommitted sample worker failure"):
        recorder.resume_episode(
            reason="runtime_running_warmup_ready",
            now=101.0,
            step_index=22,
        )

    assert recorder.is_paused
    assert recorder._sample_future is None
    assert recorder.collection_overload_reason.startswith(
        "collection_sample_drain_error:RuntimeError:writer queue failed"
    )
    assert recorder.continuity_segment_index == 0


def test_process_writer_resume_orders_snapshot_before_temporal_reset() -> None:
    class PendingSnapshot:
        settled = False
        result_calls = 0

        def done(self) -> bool:
            return bool(self.settled)

        def result(self) -> bool:
            self.result_calls += 1
            if not self.settled:
                raise AssertionError("resume must not wait for snapshot submission")
            return True

    pending = PendingSnapshot()
    ordered_events: list[str] = []
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.collection_worker_mode = "process"
    recorder.active_episode_index = 3
    recorder._process_episode_submitted_count = 11
    recorder.process_worker = SimpleNamespace(
        reset_temporal_history=lambda: ordered_events.append("reset")
    )
    recorder._sample_future = pending
    recorder._sample_overload_reason = None
    recorder._sample_modality_reason = None
    recorder._collection_integrity_reasons = []
    recorder._pause_started_at = 100.0
    recorder._pause_started_monotonic = time.monotonic()
    recorder._pause_reason = "hold_requested"
    recorder._pause_reasons = ["hold_requested"]
    recorder._pause_started_frame_index = 11
    recorder._pause_started_step_index = 21
    recorder._pause_started_last_state = "RUNNING"
    recorder._pause_segments = []
    recorder._paused_accumulated_sec = 0.0
    recorder._record_timestamps = []
    recorder._continuity_segment_index = 0

    assert not recorder.resume_episode(
        reason="runtime_running_warmup_ready",
        now=101.0,
        step_index=22,
    )
    assert ordered_events == []
    assert pending.result_calls == 0

    pending.settled = True
    recorder._process_episode_submitted_count = 12
    assert recorder.resume_episode(
        reason="runtime_running_warmup_ready",
        now=102.0,
        step_index=23,
    )
    assert pending.result_calls == 1
    assert ordered_events == ["reset"]
    assert not recorder.is_paused
    assert recorder._pause_segments[-1]["start_frame_index"] == 12


def test_sample_executor_submission_failure_is_contained_as_overload() -> None:
    class FailedExecutor:
        def submit(self, *_args: object, **_kwargs: object) -> object:
            raise RuntimeError("executor unavailable")

    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.collection_worker_mode = "thread"
    recorder.active_episode_index = 2
    recorder.args = SimpleNamespace(collection_progress_arm_control_mode="profile")
    recorder._async_sample_enabled = True
    recorder._sample_future = None
    recorder._sample_executor = FailedExecutor()
    recorder._sample_overload_reason = None
    recorder._sample_modality_reason = None
    recorder._collection_integrity_reasons = []
    recorder._pause_started_at = None

    assert not recorder.record_step_async(SimpleNamespace())
    assert recorder._sample_future is None
    assert recorder.collection_overload_reason == (
        "collection_sample_submit_error:RuntimeError:executor unavailable"
    )
    assert recorder._collection_integrity_reasons == [
        "collection_sample_submit_error:RuntimeError:executor unavailable"
    ]

    recorder._sample_executor = None
    recorder._sample_overload_reason = None
    assert not recorder.record_step_async(SimpleNamespace())
    assert recorder.collection_overload_reason == (
        "collection_sample_submit_error:"
        "RuntimeError:teleop sample executor is not available"
    )


def test_thread_writer_live_frequency_status_never_reads_writer_io_lock() -> None:
    writer_calls: list[int] = []

    def tactile_frame_counts(episode_index: int) -> dict[str, int]:
        writer_calls.append(int(episode_index))
        return {"left": 9}

    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.collection_worker_mode = "thread"
    recorder.active_episode_index = 4
    recorder.sample_builder = SimpleNamespace(
        frame_index=12,
        last_camera_rejection=None,
        last_camera_frame_age_sec_by_key={},
    )
    recorder.writer = SimpleNamespace(tactile_frame_counts=tactile_frame_counts)
    recorder.writer_observer = None
    recorder.process_worker = None
    recorder._tactile_frame_counts_cache = {4: {"left": 7}}
    recorder._record_timestamps = []
    recorder._episode_start_monotonic = time.monotonic() - 1.0
    recorder._pause_started_at = None
    recorder._pause_started_monotonic = None
    recorder._paused_accumulated_sec = 0.0
    recorder.frequency_warning_hz = 15.0
    recorder.frequency_window_sec = 3.0
    recorder.frequency_min_elapsed_sec = 2.0
    recorder._async_sample_enabled = True
    recorder._sample_async_skipped_busy = 0
    recorder._sample_process_rejected_busy = 0
    recorder._sample_overload_reason = None
    recorder._collection_integrity_reasons = []

    live_status = recorder.collection_frequency_status()

    assert writer_calls == []
    assert live_status["tactile_frame_counts"] == {"left": 7}

    assert recorder._refresh_tactile_frame_counts() == {"left": 9}
    finalize_status = recorder.collection_frequency_status()

    assert writer_calls == [4]
    assert finalize_status["tactile_frame_counts"] == {"left": 9}
    assert recorder._tactile_frame_counts_cache == {4: {"left": 9}}


def test_completed_finalize_remains_owned_until_result_is_polled() -> None:
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    completed = SimpleNamespace(done=True, thread=None, success=True)
    recorder.finalize_tasks = [completed]
    recorder.saved_episodes = 0

    # Worker completion alone is not the application receipt.  Shutdown and
    # episode admission must continue to see this lifecycle as outstanding
    # until the owner consumes the task exactly once.
    assert recorder.finalizing
    assert recorder.poll_finalize_tasks() == [completed]
    assert recorder.saved_episodes == 1
    assert not recorder.finalizing
    assert recorder.poll_finalize_tasks() == []


def test_shutdown_publishes_finalize_that_completed_before_close_poll(
    tmp_path: Path,
) -> None:
    task = SimpleNamespace(
        done=True,
        thread=None,
        success=True,
        episode_index=8,
        frame_count=21,
        queued_at=10.0,
        started_at=10.1,
        completed_at=10.2,
        stage_timings_ms={},
        error="",
        v25p_error="",
        v25p_native_error="",
        progress_error="",
    )
    recorder = RealTeleopDatasetRecorder.__new__(RealTeleopDatasetRecorder)
    recorder.active_episode_index = None
    recorder._pause_started_at = None
    recorder.finalize_tasks = [task]
    recorder.saved_episodes = 0
    recorder.dataset_root = tmp_path
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=lambda **_kwargs: None,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert runtime.close(timeout_sec=0.2)
    assert recorder.saved_episodes == 1
    assert recorder.finalize_tasks == []
    assert clutch.finalize_states[-1] == {
        "status": "saved",
        "message": clutch.finalize_states[-1]["message"],
        "episode_index": 8,
        "frame_count": 21,
    }
    assert "正式数据集 episode 8 已保存" in str(
        clutch.finalize_states[-1]["message"]
    )
    assert clutch.round_trip_advances == 1
    assert clutch.plus_episode_indices == [8]


def test_cancelled_result_is_cleaned_before_completion_is_published() -> None:
    entered = threading.Event()
    release = threading.Event()
    cleaned: list[object] = []
    candidate = object()
    clock_value = [10.0]
    plane = CollectionControlPlane(name="unit-test", clock=lambda: clock_value[0])

    def prepare() -> object:
        entered.set()
        assert release.wait(timeout=1.0)
        clock_value[0] = 20.0
        return candidate

    def discard(value: object) -> None:
        cleaned.append(value)
        clock_value[0] = 30.0

    generation = plane.submit(
        "prepare",
        prepare,
        discard_value=discard,
    )
    assert entered.wait(timeout=1.0)
    assert plane.cancel(generation)
    assert plane.busy
    release.set()

    result = _wait_for_result(plane, generation)
    assert result.cancelled
    assert not result.succeeded
    assert result.value is None
    assert cleaned == [candidate]
    assert result.started_at == 10.0
    assert result.completed_at == 20.0
    assert result.elapsed_sec == 10.0
    assert not plane.busy


@pytest.mark.parametrize("action", ("cancel", "close"))
def test_request_at_worker_publication_cut_discards_once_before_drain(
    action: str,
) -> None:
    plane = CollectionControlPlane(name="publication-cut-test")
    cut = _PauseWorkerAfterUnlock()
    plane._lock = cut
    candidate = object()
    cleaned: list[object] = []
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()

    def prepare() -> object:
        cut.worker = threading.current_thread()
        return candidate

    def discard(value: object) -> None:
        cleanup_entered.set()
        assert cleanup_release.wait(timeout=2.0)
        cleaned.append(value)

    generation = plane.submit("prepare", prepare, discard_value=discard)
    try:
        assert cut.reached.wait(timeout=1.0)
        if action == "cancel":
            assert plane.cancel(generation)
        else:
            assert not plane.close(timeout_sec=0.0)
        cut.release.set()
        assert cleanup_entered.wait(timeout=1.0)
        assert plane.busy
        assert plane.active_operation == (generation, "prepare")
        assert plane.poll(generation) is None
        if action == "close":
            assert not plane.close(timeout_sec=0.0)
        if action == "cancel":
            with pytest.raises(CollectionControlOperationBusy, match="prepare"):
                plane.submit("next", object)
            assert plane.wait(generation, timeout_sec=0.0) is None
        cleanup_release.set()
        if action == "cancel":
            result = _wait_for_result(plane, generation)
            assert result.cancelled
            assert not result.succeeded
            assert result.value is None
            assert result.error is None
            assert result.discard_error is None
        else:
            assert plane.close(timeout_sec=1.0)
            assert plane.poll(generation) is None
        assert cleaned == [candidate]
        assert not plane.busy
    finally:
        cut.release.set()
        cleanup_release.set()
        plane.close(timeout_sec=1.0)
        if cut.worker is not None:
            cut.worker.join(timeout=1.0)


def test_cancel_of_completed_value_keeps_cleanup_off_the_caller_and_single_flight() -> None:
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()
    candidate = object()
    plane = CollectionControlPlane(name="unit-test")

    def discard(value: object) -> None:
        assert value is candidate
        cleanup_entered.set()
        assert cleanup_release.wait(timeout=1.0)

    generation = plane.submit(
        "prepare",
        lambda: candidate,
        discard_value=discard,
    )
    deadline = time.monotonic() + 1.0
    while plane.busy and time.monotonic() < deadline:
        time.sleep(0.001)
    assert not plane.busy
    assert plane.active_kind == "prepare"
    assert plane.active_operation == (generation, "prepare")

    started_at = time.perf_counter()
    assert plane.cancel(generation)
    cancel_elapsed = time.perf_counter() - started_at

    assert cancel_elapsed < 0.05
    assert cleanup_entered.wait(timeout=1.0)
    assert plane.busy
    assert plane.poll(generation) is None
    with pytest.raises(CollectionControlOperationBusy, match="prepare"):
        plane.submit("next", lambda: None)

    cleanup_release.set()
    result = _wait_for_result(plane, generation)
    assert result.cancelled
    assert not result.succeeded
    assert result.value is None
    assert result.discard_error is None
    assert not plane.busy


@pytest.mark.parametrize("cancel_after_completion", (False, True))
def test_close_unconsumed_cancelled_completion_does_not_discard_again(
    cancel_after_completion: bool,
) -> None:
    plane = CollectionControlPlane(name="cancelled-completion-test")
    candidate = object()
    cleaned: list[object] = []
    entered = threading.Event()
    release = threading.Event()

    def prepare() -> object:
        entered.set()
        assert release.wait(timeout=1.0)
        return candidate

    generation = plane.submit("prepare", prepare, discard_value=cleaned.append)
    assert entered.wait(timeout=1.0)
    try:
        if cancel_after_completion:
            release.set()
            deadline = time.monotonic() + 1.0
            while plane.busy and time.monotonic() < deadline:
                time.sleep(0.001)
            assert not plane.busy
        assert plane.cancel(generation)
        release.set()
        deadline = time.monotonic() + 1.0
        while plane.busy and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not plane.busy
        assert plane.close(timeout_sec=1.0)
        assert cleaned == [candidate]
        assert plane.poll(generation) is None
    finally:
        release.set()
        plane.close(timeout_sec=1.0)


@pytest.mark.parametrize("cancel_after_completion", (False, True))
def test_cancelled_cleanup_failure_remains_one_consumable_result(
    cancel_after_completion: bool,
) -> None:
    plane = CollectionControlPlane(name="discard-failure-test")
    candidate = object()
    failure = RuntimeError("candidate retirement failed")
    cleaned: list[object] = []
    fail_cleanup = True
    entered = threading.Event()
    release = threading.Event()

    def prepare() -> object:
        entered.set()
        assert release.wait(timeout=1.0)
        return candidate

    def discard(value: object) -> None:
        cleaned.append(value)
        if fail_cleanup:
            raise failure

    generation = plane.submit("prepare", prepare, discard_value=discard)
    assert entered.wait(timeout=1.0)
    try:
        if cancel_after_completion:
            release.set()
            deadline = time.monotonic() + 1.0
            while plane.busy and time.monotonic() < deadline:
                time.sleep(0.001)
            assert not plane.busy
        assert plane.cancel(generation)
        release.set()
        result = _wait_for_result(plane, generation)
        assert result.cancelled
        assert not result.succeeded
        assert result.value is None
        assert result.error is None
        assert result.discard_error is failure
        assert cleaned == [candidate]
        assert plane.poll(generation) is None
        assert plane.active_operation == (generation, "prepare")
        assert plane.busy
        assert plane.cleanup_error is failure
        with pytest.raises(CollectionControlOperationBusy):
            plane.submit("next", object)
        fail_cleanup = False
        deadline = time.monotonic() + 1.0
        while not plane.close(timeout_sec=0.05) and time.monotonic() < deadline:
            pass
        assert plane.close(timeout_sec=0.05)
        assert plane.active_operation is None
        assert plane.cleanup_error is None
        assert all(value is candidate for value in cleaned)
    finally:
        fail_cleanup = False
        release.set()
        plane.close(timeout_sec=1.0)


def test_close_discards_unconsumed_success_and_rejects_new_work() -> None:
    cleaned: list[object] = []
    candidate = object()
    plane = CollectionControlPlane(name="unit-test")
    generation = plane.submit(
        "prepare",
        lambda: candidate,
        discard_value=cleaned.append,
    )
    deadline = time.monotonic() + 1.0
    while plane.busy and time.monotonic() < deadline:
        time.sleep(0.001)

    assert plane.close()

    assert cleaned == [candidate]
    assert plane.poll(generation) is None
    with pytest.raises(RuntimeError, match="closed"):
        plane.submit("prepare", lambda: None)


def test_close_completed_value_does_not_wait_beyond_cleanup_budget() -> None:
    plane = CollectionControlPlane(name="completed-close-budget-test")
    candidate = object()
    cleaned: list[object] = []
    cleanup_entered = threading.Event()
    cleanup_release = threading.Event()
    close_returned = threading.Event()
    close_results: list[bool] = []

    def discard(value: object) -> None:
        cleanup_entered.set()
        assert cleanup_release.wait(timeout=2.0)
        cleaned.append(value)

    plane.submit("prepare", lambda: candidate, discard_value=discard)
    deadline = time.monotonic() + 1.0
    while plane.busy and time.monotonic() < deadline:
        time.sleep(0.001)
    assert not plane.busy

    def close() -> None:
        close_results.append(plane.close(timeout_sec=0.0))
        close_returned.set()

    caller = threading.Thread(target=close, daemon=True)
    caller.start()
    try:
        assert cleanup_entered.wait(timeout=1.0)
        assert close_returned.wait(timeout=0.25)
        assert close_results == [False]
        assert plane.busy
        assert plane.poll() is None
        assert not plane.close(timeout_sec=0.0)
        with pytest.raises(RuntimeError, match="closed"):
            plane.submit("next", object)
        cleanup_release.set()
        assert plane.close(timeout_sec=1.0)
        assert cleaned == [candidate]
    finally:
        cleanup_release.set()
        caller.join(timeout=1.0)
        plane.close(timeout_sec=1.0)


@pytest.mark.parametrize("action", ("cancel", "close"))
@pytest.mark.parametrize("failure_point", ("construction", "start"))
def test_completed_cleanup_worker_failure_retains_candidate_for_retry(
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    failure_point: str,
) -> None:
    plane = CollectionControlPlane(name="cleanup-submission-test")
    candidate = object()
    cleaned: list[object] = []
    generation = plane.submit("prepare", lambda: candidate, discard_value=cleaned.append)
    deadline = time.monotonic() + 1.0
    while plane.busy and time.monotonic() < deadline:
        time.sleep(0.001)
    assert not plane.busy

    def fail(*_args, **_kwargs) -> None:
        raise RuntimeError("cleanup worker unavailable")

    with monkeypatch.context() as failure_patch:
        if failure_point == "construction":
            failure_patch.setattr(control_plane_module.threading, "Thread", fail)
        else:
            failure_patch.setattr(control_plane_module.threading.Thread, "start", fail)
        if action == "cancel":
            assert not plane.cancel(generation)
        else:
            assert not plane.close(timeout_sec=0.0)
            assert plane.poll(generation) is None
        assert plane.poll(generation) is None
        assert plane.busy
        assert isinstance(plane.cleanup_error, RuntimeError)
        with pytest.raises(RuntimeError, match="closed" if action == "close" else "pending"):
            plane.submit("next", object)
        assert cleaned == []
        assert plane.active_operation == (generation, "prepare")

    if action == "cancel":
        assert plane.cancel(generation)
        result = _wait_for_result(plane, generation)
        assert result.cancelled
        assert result.value is None
    else:
        assert plane.close(timeout_sec=1.0)
        assert plane.poll(generation) is None
    assert cleaned == [candidate]
    assert plane.active_operation is None
    assert plane.close(timeout_sec=1.0)


def test_close_waits_for_running_operation_and_its_discard_cleanup() -> None:
    entered = threading.Event()
    release = threading.Event()
    cleaned: list[object] = []
    candidate = object()
    plane = CollectionControlPlane(name="unit-test")

    def prepare() -> object:
        entered.set()
        assert release.wait(timeout=1.0)
        return candidate

    plane.submit("prepare", prepare, discard_value=cleaned.append)
    assert entered.wait(timeout=1.0)
    releaser = threading.Timer(0.05, release.set)
    releaser.start()
    started_at = time.monotonic()
    try:
        assert plane.close(timeout_sec=1.0)
    finally:
        releaser.join(timeout=1.0)

    assert time.monotonic() - started_at >= 0.04
    assert cleaned == [candidate]
    assert not plane.busy


def test_close_timeout_is_bounded_and_a_later_close_can_finish_drain() -> None:
    entered = threading.Event()
    release = threading.Event()
    cleaned: list[object] = []
    plane = CollectionControlPlane(name="unit-test")

    plane.submit(
        "prepare",
        lambda: (entered.set(), release.wait(timeout=1.0)),
        discard_value=cleaned.append,
    )
    assert entered.wait(timeout=1.0)

    started_at = time.monotonic()
    assert not plane.close(timeout_sec=0.01)
    assert time.monotonic() - started_at < 0.1
    release.set()
    assert plane.close(timeout_sec=1.0)
    assert cleaned == [(None, True)]


def test_wait_consumes_only_the_requested_completed_generation() -> None:
    release = threading.Event()
    plane = CollectionControlPlane(name="unit-test")
    generation = plane.submit(
        "prepare",
        lambda: (release.wait(timeout=1.0), "ready")[1],
    )

    assert plane.wait(generation + 1, timeout_sec=0.0) is None
    assert plane.wait(generation, timeout_sec=0.01) is None
    release.set()
    result = plane.wait(generation, timeout_sec=1.0)
    assert result is not None
    assert result.succeeded
    assert result.value == "ready"
    assert plane.poll(generation) is None


class _FailingUi:
    def publish(self, **_state: object) -> None:
        raise RuntimeError("ui publish unavailable")

    def append_message(self, _title: str, _message: str) -> None:
        raise RuntimeError("ui message unavailable")


def test_episode_open_stop_and_finalize_ignore_ui_transport_failure(
    tmp_path: Path,
) -> None:
    finalize_ready = False
    finalize_consumed = False
    finalize_task = SimpleNamespace(
        episode_index=4,
        frame_count=12,
        queued_at=10.0,
        started_at=10.1,
        completed_at=10.2,
        stage_timings_ms={},
        success=True,
        error="",
        v25p_error="",
        v25p_native_error="",
        progress_error="",
    )

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        active_episode_index=None,
        camera_sources={},
        require_configured_for_current_args=lambda: None,
        collection_frequency_status=lambda: {"sample_hz": 15.0},
        close=lambda: None,
    )

    def start_episode(_metadata: object, *, now: float) -> int:
        del now
        recorder.is_recording = True
        recorder.active_episode_index = 4
        return 4

    def stop_episode(**_kwargs: object) -> object:
        recorder.is_recording = False
        recorder.finalizing = True
        return finalize_task

    def poll_finalize_tasks() -> list[object]:
        nonlocal finalize_consumed
        if not finalize_ready or finalize_consumed:
            return []
        finalize_consumed = True
        recorder.finalizing = False
        return [finalize_task]

    recorder.start_episode = start_episode
    recorder.stop_episode = stop_episode
    recorder.poll_finalize_tasks = poll_finalize_tasks
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        control_ui=_FailingUi(),
    )

    assert runtime.start_recording(time.time(), start_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert recorder.is_recording
    assert clutch.recording_states[-1]["event_status"] == "recording"
    assert runtime.stop_recording(time.time(), end_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert clutch.recording_states[-1]["event_status"] == "finalizing"
    finalize_ready = True
    runtime.poll_dataset_finalize(time.time())

    assert clutch.finalize_states[-1]["status"] == "saved"
    assert clutch.round_trip_advances == 1
    assert clutch.plus_episode_indices == [4, 4]
    assert any(
        tag == "WebGUI" and "collection lifecycle is retained" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_episode_open_rpc_is_nonblocking_and_recorder_is_detached_until_publish(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.dataset_root = tmp_path
            self.writer = SimpleNamespace(next_episode_index=4)
            self.writer_started = True
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.active_episode_index = None
            self.camera_sources = {}

        def require_configured_for_current_args(self) -> None:
            return None

        def start_episode(self, _metadata: object, *, now: float) -> int:
            del now
            entered.set()
            assert release.wait(timeout=1.0)
            self.active_episode_index = 4
            self.is_recording = True
            return 4

        def collection_frequency_status(self) -> dict[str, object]:
            return {}

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            return None

    recorder = Recorder()
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        publish_runtime_hardware_status=lambda: (_ for _ in ()).throw(
            RuntimeError("status transport unavailable")
        ),
    )

    started_at = time.perf_counter()
    assert runtime.start_recording(time.time(), start_reason="unit")
    assert time.perf_counter() - started_at < 0.05
    assert entered.wait(timeout=1.0)
    assert slot[0] is None
    assert runtime.collection_transition_active()
    assert not runtime.recording_active()

    release.set()
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert recorder.is_recording
    assert runtime.recording_active()
    assert clutch.plus_episode_indices == [4]
    assert clutch.recording_states[-1]["event_status"] == "recording"
    assert any(
        tag == "Dataset" and "hardware-status refresh failed" in message
        for tag, message in logs
    )


@pytest.mark.parametrize("cleanup_retry", (None, "poll", "cancel"))
def test_pending_episode_open_is_cancellable_and_never_becomes_sample_visible(
    tmp_path: Path,
    cleanup_retry: str | None,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    stop_reasons: list[str] = []
    fail_cleanup = cleanup_retry is not None

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.dataset_root = tmp_path
            self.writer = SimpleNamespace(next_episode_index=4)
            self.writer_started = True
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.active_episode_index = None
            self.camera_sources = {}
            self.closed = False

        def require_configured_for_current_args(self) -> None:
            return None

        def start_episode(self, _metadata: object, *, now: float) -> int:
            del now
            entered.set()
            assert release.wait(timeout=1.0)
            self.active_episode_index = 4
            self.is_recording = True
            return 4

        def stop_episode(self, *, reason: str, **_kwargs: object) -> None:
            stop_reasons.append(str(reason))
            self.active_episode_index = None
            self.is_recording = False
            return None

        def close(self) -> None:
            if fail_cleanup:
                raise RuntimeError("cancelled open cleanup unavailable")
            self.closed = True

        def collection_frequency_status(self) -> dict[str, object]:
            return {}

        def poll_finalize_tasks(self) -> list[object]:
            return []

    recorder = Recorder()
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert runtime.start_recording(time.time(), start_reason="unit")
    assert entered.wait(timeout=1.0)
    assert slot[0] is None
    assert runtime.countdown_deadline() == float("inf")
    assert clutch.countdown_states[-1]["remaining_sec"] == 0.0
    assert "正在打开" in str(clutch.countdown_states[-1]["message"])

    assert runtime.cancel_recording_countdown(
        time.time(),
        reason="unit_hold",
    )
    assert "正在取消" in str(clutch.countdown_states[-1]["message"])
    release.set()
    if cleanup_retry is not None:
        deadline = time.monotonic() + 1.0
        while not clutch.recording_states and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(0.001)
        assert runtime.collection_transition_active()
        assert slot[0] is None and not recorder.closed
        assert not runtime.recording_active()
        assert clutch.plus_episode_indices == []
        assert clutch.recording_states[-1]["event_status"] == "error"
        assert "unit_hold" in str(clutch.recording_states[-1]["event_message"])
        assert not runtime.start_recording(time.time(), start_reason="premature_retry")
        fail_cleanup = False
        if cleanup_retry == "cancel":
            assert runtime.cancel_recording_countdown(time.time(), reason="cleanup_retry")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert not runtime.collection_transition_active()
    assert slot[0] is None
    assert recorder.closed
    assert stop_reasons == ["episode_open_cancelled"] * (2 if cleanup_retry else 1)
    assert not runtime.recording_active()
    assert clutch.plus_episode_indices == []
    assert clutch.recording_states[-1]["event_status"] == "idle"
    assert "未向控制循环发布可采样 Episode" in str(
        clutch.recording_states[-1]["event_message"]
    )
    expected_reason = {
        None: "unit_hold",
        "poll": "operator_cancelled",
        "cancel": "cleanup_retry",
    }[cleanup_retry]
    assert expected_reason in str(clutch.recording_states[-1]["event_message"])
    assert args.dataset_startup_error == ""
    assert runtime.close()


def test_debug_trajectory_start_failure_does_not_close_formal_episode(
    tmp_path: Path,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        active_episode_index=None,
        camera_sources={},
        require_configured_for_current_args=lambda: None,
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def start_episode(_metadata: object, *, now: float) -> int:
        del now
        recorder.is_recording = True
        recorder.active_episode_index = 4
        return 4

    def stop_episode(**_kwargs: object) -> None:
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.start_episode = start_episode
    recorder.stop_episode = stop_episode
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        debug_episode_enabled=True,
        start_episode_recording=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("debug trajectory disk unavailable")
        ),
    )

    assert runtime.start_recording(time.time(), start_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert recorder.is_recording
    assert args.dataset_startup_error == ""
    assert clutch.recording_states[-1]["event_status"] == "recording"
    assert any(
        tag == "Episode"
        and "formal dataset episode remains active" in message
        and "debug trajectory disk unavailable" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_position_audit_start_and_metadata_failures_are_observational(
    tmp_path: Path,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        active_episode_index=None,
        camera_sources={},
        require_configured_for_current_args=lambda: None,
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def start_episode(_metadata: object, *, now: float) -> int:
        del now
        recorder.is_recording = True
        recorder.active_episode_index = 5
        return 5

    def stop_episode(**_kwargs: object) -> None:
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.start_episode = start_episode
    recorder.stop_episode = stop_episode
    position_debug_runtime = SimpleNamespace(
        start=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("audit attach metadata unavailable")
        ),
        stop=lambda *_args, **_kwargs: None,
        start_close=lambda _action: None,
        current_recorder=lambda: None,
    )

    class FailingPlus(_Clutch):
        def begin_plus_curation_episode(self, episode_index: int) -> None:
            del episode_index
            raise LookupError("Plus projection unavailable")

    slot: list[object | None] = [recorder]
    clutch = FailingPlus()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        position_episode_debug_runtime_override=position_debug_runtime,
        update_control_debug_runtime_metadata=lambda **_kwargs: (
            _ for _ in ()
        ).throw(RuntimeError("debug metadata store unavailable")),
    )
    args.control_debug_config.enabled = True

    assert runtime.start_recording(time.time(), start_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert recorder.is_recording
    assert recorder.active_episode_index == 5
    assert args.dataset_startup_error == ""
    assert clutch.recording_states[-1]["event_status"] == "recording"
    assert any(
        tag == "ControlDebug"
        and "formal dataset start continues" in message
        and "audit attach metadata unavailable" in message
        for tag, message in logs
    )
    assert any(
        tag == "ControlDebug"
        and "episode index mismatch metadata publication failed" in message
        and "debug metadata store unavailable" in message
        for tag, message in logs
    )
    assert any(
        tag == "Plus"
        and "正式原始数采继续" in message
        and "Plus projection unavailable" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_position_audit_detach_failure_cannot_veto_formal_stop(
    tmp_path: Path,
) -> None:
    formal_stop_calls: list[dict[str, object]] = []
    audit_close_timeouts: list[float] = []

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=True,
        is_paused=False,
        frame_count=12,
        active_episode_index=4,
        camera_sources={},
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def stop_episode(**kwargs: object) -> None:
        formal_stop_calls.append(dict(kwargs))
        before_finalize = kwargs.get("before_finalize")
        assert callable(before_finalize)
        audit_result = before_finalize()
        assert audit_result is not None
        assert audit_result["recovered_after_stop_error"] is True
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.stop_episode = stop_episode
    audit_recorder = SimpleNamespace(
        close=lambda *, timeout_sec: audit_close_timeouts.append(float(timeout_sec)),
        snapshot=lambda: {"closed": True, "enqueue_drop_count": 0},
    )
    audit_slot: list[object | None] = [audit_recorder]

    def stop_position_debug(*, reason: str) -> None:
        assert reason == "unit"
        audit_slot[0] = None
        raise RuntimeError("audit detach metadata unavailable")

    position_debug_runtime = SimpleNamespace(
        start=lambda *_args, **_kwargs: False,
        stop=stop_position_debug,
        start_close=lambda _action: (_ for _ in ()).throw(
            RuntimeError("audit close thread unavailable")
        ),
        current_recorder=lambda: audit_slot[0],
    )
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        position_episode_debug_runtime_override=position_debug_runtime,
    )

    assert runtime.stop_recording(time.time(), end_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert len(formal_stop_calls) == 1
    assert formal_stop_calls[0]["defer_finalize_start"] is False
    assert audit_close_timeouts == [5.0]
    assert recorder.is_recording is False
    assert slot[0] is recorder
    assert args.dataset_startup_error == ""
    assert clutch.recording_states[-1]["event_status"] == "discarded"
    assert any(
        tag == "ControlDebug"
        and "episode audit stop failed" in message
        and "audit detach metadata unavailable" in message
        for tag, message in logs
    )
    assert any(
        tag == "ControlDebug"
        and "will be delegated to formal finalize" in message
        and "audit close thread unavailable" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_episode_open_configuration_failure_republishes_unmutated_recorder(
    tmp_path: Path,
) -> None:
    reject_configuration = True

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.dataset_root = tmp_path
            self.writer = SimpleNamespace(next_episode_index=4)
            self.writer_started = True
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.active_episode_index = None
            self.camera_sources = {}
            self.closed = False

        def require_configured_for_current_args(self) -> None:
            if reject_configuration:
                raise RuntimeError("profile changed")

        def start_episode(self, _metadata: object, *, now: float) -> int:
            self.is_recording = True
            self.active_episode_index = self.writer.next_episode_index
            return self.active_episode_index

        def stop_episode(self, **_kwargs: object) -> None:
            self.is_recording = False
            self.active_episode_index = None

        def collection_frequency_status(self) -> dict[str, object]:
            return {}

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    recorder = Recorder()
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert runtime.start_recording(time.time(), start_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert not recorder.closed
    assert clutch.recording_states[-1]["event_status"] == "error"

    # A later open has its own mutation boundary and reason, even when the
    # same ready writer and runtime survive a complete open/stop cycle.
    reject_configuration = False
    assert runtime.start_recording(time.time(), start_reason="configured_retry")
    poll_recording_transition(runtime)
    assert slot[0] is recorder and recorder.is_recording
    assert clutch.recording_states[-1]["event_status"] == "recording"
    assert any(tag == "DatasetTiming" and "reason=configured_retry" in message
               for tag, message in logs)
    assert runtime.stop_recording(time.time(), end_reason="retry_finished")
    poll_recording_transition(runtime)
    assert slot[0] is recorder and not recorder.is_recording
    assert not recorder.closed

    reject_configuration = True
    assert runtime.start_recording(time.time(), start_reason="profile_changed_again")
    poll_recording_transition(runtime)
    assert slot[0] is recorder and not recorder.closed
    assert clutch.recording_states[-1]["event_status"] == "error"
    assert args.dataset_startup_error == ""
    assert runtime.close()


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_episode_open_rpc_failure_isolates_attempted_writer_for_fresh_retry(
    tmp_path: Path, cleanup_fails: bool,
) -> None:
    fail_cleanup = cleanup_fails
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.dataset_root = tmp_path
            self.writer = SimpleNamespace(next_episode_index=4)
            self.writer_started = True
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.active_episode_index = None
            self.camera_sources = {}
            self.closed = False

        def require_configured_for_current_args(self) -> None:
            return None

        def start_episode(self, _metadata: object, *, now: float) -> int:
            del now
            raise RuntimeError("process reply timeout")

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            if fail_cleanup:
                raise RuntimeError("open cleanup unavailable")
            self.closed = True

    recorder = Recorder()
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert runtime.start_recording(time.time(), start_reason="unit")
    if cleanup_fails:
        deadline = time.monotonic() + 1.0
        while not args.dataset_startup_error and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(.001)
        assert runtime.collection_transition_active()
        assert slot[0] is None and not recorder.closed
        assert "process reply timeout" in args.dataset_startup_error
        fail_cleanup = False
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is None
    assert recorder.closed
    assert "process reply timeout" in args.dataset_startup_error
    assert clutch.recording_states[-1]["event_status"] == "error"


def test_episode_writer_prepare_is_nonblocking_and_published_only_after_ready(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    instances: list[object] = []

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.dataset_root = tmp_path
            self.writer_started = False
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            entered.set()
            assert release.wait(timeout=1.0)
            self.writer_started = True

        def configured_for_current_args(self) -> bool:
            return True

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    slot: list[object | None] = [None]
    clutch = _Clutch()
    transition_calls: list[str] = []
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=transition_calls,
    )

    started_at = time.perf_counter()
    assert runtime.schedule_recording_countdown(time.time(), start_reason="unit")
    assert time.perf_counter() - started_at < 0.05
    assert entered.wait(timeout=1.0)
    assert slot[0] is None
    assert runtime.countdown_deadline() == float("inf")
    assert clutch.countdown_states[-1]["remaining_sec"] == 0.0

    release.set()
    deadline = time.monotonic() + 1.0
    while runtime.countdown_deadline() == float("inf") and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is instances[0]
    assert runtime.countdown_deadline() != float("inf")
    assert runtime.countdown_deadline() is not None
    assert transition_calls == ["formal_episode_start"]
    assert clutch.restore_calls == 1
    runtime.close()


def test_debug_trajectory_stop_failure_does_not_invalidate_formal_stop(
    tmp_path: Path,
) -> None:
    stop_attempts = 0

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=True,
        is_paused=False,
        pause_reason="",
        frame_count=12,
        active_episode_index=4,
        camera_sources={},
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: pytest.fail("formal recorder must remain reusable"),
    )

    def stop_episode(**_kwargs: object) -> None:
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.stop_episode = stop_episode
    debug_recorder = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        is_recording=True,
        frame_count=3,
        episode_dir=tmp_path / "debug",
        record_event=lambda *_args, **_kwargs: None,
    )

    def stop_debug(recorder_arg: object, **_kwargs: object) -> None:
        nonlocal stop_attempts
        assert recorder_arg is debug_recorder
        stop_attempts += 1
        if stop_attempts == 1:
            raise RuntimeError("debug trailer unavailable")
        debug_recorder.is_recording = False

    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        debug_episode_enabled=True,
        stop_episode_recording=stop_debug,
        episode_recorder_override=debug_recorder,
    )

    assert runtime.stop_recording(time.time(), end_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert recorder.is_recording is False
    assert args.dataset_startup_error == ""
    assert clutch.recording_states[-1]["event_status"] == "discarded"
    assert any(
        tag == "Episode" and "formal dataset stop remains committed" in message
        for tag, message in logs
    )
    assert runtime.recording_active() is False
    assert runtime.close()
    assert stop_attempts == 1


def test_debug_pause_resume_event_failure_cannot_veto_formal_continuity(
    tmp_path: Path,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=True,
        is_paused=False,
        frame_count=12,
        active_episode_index=4,
        camera_sources={},
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def pause_episode(**_kwargs: object) -> bool:
        recorder.is_paused = True
        recorder.pause_reason = "unit_hold"
        return True

    def resume_episode(**_kwargs: object) -> bool:
        recorder.is_paused = False
        recorder.pause_reason = ""
        return True

    def stop_episode(**_kwargs: object) -> None:
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.pause_episode = pause_episode
    recorder.resume_episode = resume_episode
    recorder.stop_episode = stop_episode
    debug_recorder = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        is_recording=True,
        frame_count=3,
        episode_dir=tmp_path / "debug",
        record_event=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("debug event unavailable")
        ),
    )

    def stop_debug(recorder_arg: object, **_kwargs: object) -> None:
        assert recorder_arg is debug_recorder
        debug_recorder.is_recording = False

    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        debug_episode_enabled=True,
        stop_episode_recording=stop_debug,
        episode_recorder_override=debug_recorder,
    )

    assert runtime.pause_recording(time.time(), reason="unit_hold")
    assert recorder.is_paused
    assert runtime.recording_paused()

    running_step = SimpleNamespace(state=SimpleNamespace(value="RUNNING"))
    assert not runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert recorder.is_paused is False
    assert runtime.recording_paused() is False
    quarantine_messages = [
        message
        for tag, message in logs
        if tag == "Episode" and "已隔离" in message
    ]
    assert len(quarantine_messages) == 1
    assert "operation=event:recording_paused" in quarantine_messages[0]
    assert runtime.close()


def test_debug_only_resume_boundary_failure_stops_episode_without_escaping(
    tmp_path: Path,
) -> None:
    debug_events: list[str] = []
    stop_reasons: list[str] = []

    def record_event(event_type: str, **_kwargs: object) -> None:
        debug_events.append(str(event_type))
        if event_type == "recording_resumed":
            raise RuntimeError("debug resume marker unavailable")

    debug_recorder = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        is_recording=True,
        frame_count=3,
        episode_dir=tmp_path / "debug",
        record_event=record_event,
    )

    def stop_debug(recorder_arg: object, *, end_reason: str, **_kwargs: object) -> None:
        assert recorder_arg is debug_recorder
        stop_reasons.append(str(end_reason))
        debug_recorder.is_recording = False

    slot: list[object | None] = [None]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=lambda **_kwargs: None,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        debug_episode_enabled=True,
        stop_episode_recording=stop_debug,
        episode_recorder_override=debug_recorder,
    )

    assert runtime.pause_recording(time.time(), reason="unit_hold")
    running_step = SimpleNamespace(state=SimpleNamespace(value="RUNNING"))
    assert not runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert not runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert runtime.collection_transition_active()

    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert debug_events == ["recording_paused", "recording_resumed"]
    assert stop_reasons == ["debug_resume_boundary_failure"]
    assert not debug_recorder.is_recording
    assert not runtime.recording_active()
    assert not runtime.recording_paused()
    assert args.dataset_startup_error == ""
    assert clutch.hold_reasons == []
    assert any(
        tag == "Episode" and "无法写入恢复边界" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_debug_only_stop_failure_does_not_poison_formal_writer_health(
    tmp_path: Path,
) -> None:
    debug_recorder = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        is_recording=True,
        frame_count=3,
        episode_dir=tmp_path / "debug",
        record_event=lambda *_args, **_kwargs: None,
    )
    stop_attempts = 0

    def fail_stop_debug(*_args: object, **_kwargs: object) -> None:
        nonlocal stop_attempts
        stop_attempts += 1
        raise RuntimeError("debug writer close unavailable")

    slot: list[object | None] = [None]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=lambda **_kwargs: None,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        debug_episode_enabled=True,
        stop_episode_recording=fail_stop_debug,
        episode_recorder_override=debug_recorder,
    )

    assert runtime.stop_recording(time.time(), end_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert stop_attempts == 2
    assert not runtime.recording_active()
    assert args.dataset_startup_error == ""
    assert clutch.hold_reasons == []
    assert clutch.recording_states[-1]["event_status"] == "error"
    assert "正式 dataset writer 状态不受影响" in str(
        clutch.recording_states[-1]["event_message"]
    )
    assert any(
        tag == "Episode" and "已隔离该记录器" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_formal_resume_requires_the_writer_pause_gate_to_clear(
    tmp_path: Path,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    allow_resume = False
    resume_calls = 0
    debug_events: list[str] = []
    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=True,
        is_paused=False,
        pause_reason="",
        frame_count=12,
        active_episode_index=4,
        camera_sources={},
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def pause_episode(**_kwargs: object) -> bool:
        recorder.is_paused = True
        recorder.pause_reason = "unit_hold"
        return True

    def resume_episode(**_kwargs: object) -> bool:
        nonlocal resume_calls
        resume_calls += 1
        if allow_resume:
            recorder.is_paused = False
            recorder.pause_reason = ""
            return True
        return False

    recorder.pause_episode = pause_episode
    recorder.resume_episode = resume_episode
    # Model the successful stop boundary too: shutdown now retains an active
    # recorder until its actual stop and debug-stop mutations are published.
    def stop_formal(**_kwargs):
        recorder.is_recording = False

    def stop_debug(target, **_kwargs):
        target.is_recording = False

    recorder.stop_episode = stop_formal
    debug_recorder = SimpleNamespace(
        config=SimpleNamespace(enabled=True),
        is_recording=True,
        frame_count=3,
        episode_dir=tmp_path / "debug",
        record_event=lambda event, **_kwargs: debug_events.append(str(event)),
    )
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        debug_episode_enabled=True,
        episode_recorder_override=debug_recorder,
        stop_episode_recording=stop_debug,
    )

    assert runtime.pause_recording(time.time(), reason="unit_hold")
    running_step = SimpleNamespace(state=SimpleNamespace(value="RUNNING"))
    assert not runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert not runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )

    assert resume_calls == 1
    assert recorder.is_paused
    assert runtime.recording_paused()
    assert debug_events == ["recording_paused"]
    assert clutch.recording_states[-1]["event_status"] == "resume_pending"
    assert "未确认解除暂停" in str(
        clutch.recording_states[-1]["event_message"]
    )
    assert sum(
        1 for tag, message in logs
        if tag == "Dataset" and "未确认解除暂停" in message
    ) == 1

    allow_resume = True
    assert not runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert runtime.resume_recording_if_ready(
        time.time(),
        step_result=running_step,
    )
    assert resume_calls == 2
    assert recorder.is_paused is False
    assert runtime.recording_paused() is False
    assert debug_events == ["recording_paused", "recording_resumed"]
    assert runtime.close()


def test_formal_pause_failure_stops_the_episode_without_escaping_control(
    tmp_path: Path,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    stop_calls: list[dict[str, object]] = []
    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=True,
        is_paused=False,
        pause_reason="",
        frame_count=12,
        active_episode_index=4,
        camera_sources={},
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def pause_episode(**_kwargs: object) -> bool:
        raise RuntimeError("simulated sample drain failure")

    def stop_episode(**kwargs: object) -> None:
        stop_calls.append(dict(kwargs))
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.pause_episode = pause_episode
    recorder.stop_episode = stop_episode
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert not runtime.pause_recording(time.time(), reason="unit_hold")
    assert runtime.collection_transition_active()
    assert slot[0] is None

    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert not runtime.collection_transition_active()
    assert slot[0] is recorder
    assert recorder.is_recording is False
    assert len(stop_calls) == 1
    assert stop_calls[0]["reason"].startswith("pause_failure:formal_writer:unit_hold")
    assert stop_calls[0]["summary"] == {
        "pause_failure_operation": "formal_writer",
        "pause_failure": "RuntimeError: simulated sample drain failure",
        "pause_reason": "unit_hold",
    }
    assert clutch.recording_states[-1]["event_status"] == "discarded"
    assert any(
        tag == "Dataset"
        and "暂停边界建立失败" in message
        and "simulated sample drain failure" in message
        for tag, message in logs
    )
    assert runtime.close()


def test_stop_submission_failure_keeps_recorder_owned_paused_and_retryable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("test must reuse the published recorder")

    stop_calls = 0
    stop_reasons: list[str] = []
    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer=SimpleNamespace(next_episode_index=4),
        writer_started=True,
        finalizing=False,
        is_recording=True,
        is_paused=False,
        pause_reason="",
        frame_count=12,
        active_episode_index=4,
        camera_sources={},
        collection_frequency_status=lambda: {},
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def stop_episode(**kwargs: object) -> None:
        nonlocal stop_calls
        stop_calls += 1
        stop_reasons.append(str(kwargs["reason"]))
        recorder.is_recording = False
        recorder.active_episode_index = None
        return None

    recorder.stop_episode = stop_episode
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )
    original_start = control_plane_module.threading.Thread.start

    def fail_start(_thread) -> None:
        raise RuntimeError("simulated stop thread exhaustion")

    monkeypatch.setattr(control_plane_module.threading.Thread, "start", fail_start)
    assert not runtime.stop_recording(time.time(), end_reason="manual_key_stop")

    assert slot[0] is recorder
    assert recorder.is_recording
    assert not runtime.collection_transition_active()
    assert runtime.recording_paused()
    assert stop_calls == 0
    assert clutch.recording_states[-1]["event_status"] == "paused"
    assert "保留重试能力" in str(
        clutch.recording_states[-1]["event_message"]
    )
    assert any(
        tag == "Dataset" and "simulated stop thread exhaustion" in message
        for tag, message in logs
    )

    monkeypatch.setattr(
        control_plane_module.threading.Thread,
        "start",
        original_start,
    )
    recorder.frame_count = 17
    assert runtime.stop_recording(time.time(), end_reason="manual_key_stop_retry")
    assert slot[0] is None
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert recorder.is_recording is False
    assert stop_calls == 1
    assert stop_reasons == ["manual_key_stop_retry"]
    assert clutch.recording_states[-1]["frame_count"] == 17
    assert not runtime.recording_paused()
    assert runtime.close()


def test_writer_prepare_submission_failure_keeps_existing_recorder_retryable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("ready recorder must be reused")

    config_writes = 0
    recorder = SimpleNamespace(
        dataset_root=tmp_path,
        writer_started=True,
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        active_episode_index=None,
        configured_for_current_args=lambda: True,
        poll_finalize_tasks=lambda: [],
        close=lambda: None,
    )

    def write_camera_device_config() -> None:
        nonlocal config_writes
        config_writes += 1

    recorder.write_camera_device_config = write_camera_device_config
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    transition_calls: list[str] = []
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=transition_calls,
    )
    original_start = control_plane_module.threading.Thread.start

    def fail_start(_thread) -> None:
        raise RuntimeError("simulated prepare thread exhaustion")

    monkeypatch.setattr(control_plane_module.threading.Thread, "start", fail_start)
    assert not runtime.schedule_recording_countdown(
        time.time(),
        start_reason="unit",
    )

    assert slot[0] is recorder
    assert not runtime.collection_transition_active()
    assert runtime.countdown_deadline() is None
    assert config_writes == 0
    assert clutch.recording_states[-1]["event_status"] == "error"
    assert any(
        tag == "Dataset" and "simulated prepare thread exhaustion" in message
        for tag, message in logs
    )

    monkeypatch.setattr(
        control_plane_module.threading.Thread,
        "start",
        original_start,
    )
    assert runtime.schedule_recording_countdown(
        time.time(),
        start_reason="unit_retry",
    )
    assert slot[0] is None
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert config_writes == 1
    assert runtime.countdown_deadline() is not None
    assert transition_calls == ["formal_episode_start"]
    assert runtime.close()


def test_failed_replacement_keeps_published_recorder_and_discards_candidate(
    tmp_path: Path,
) -> None:
    candidate_instances: list[object] = []

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.dataset_root = tmp_path
            self.writer_started = False
            self.closed = False
            candidate_instances.append(self)

        def ensure_started(self) -> None:
            raise RuntimeError("writer ready timeout")

        def close(self) -> None:
            self.closed = True

    existing = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path,
        writer_started=True,
        closed=False,
        configured_for_current_args=lambda: False,
        poll_finalize_tasks=lambda: [],
    )
    existing.close = lambda: setattr(existing, "closed", True)
    slot: list[object | None] = [existing]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        publish_runtime_hardware_status=lambda: (_ for _ in ()).throw(
            RuntimeError("status transport unavailable")
        ),
    )

    assert runtime.schedule_recording_countdown(time.time(), start_reason="unit")
    deadline = time.monotonic() + 1.0
    while runtime.countdown_deadline() is not None and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is existing
    assert existing.closed is False
    assert len(candidate_instances) == 1
    assert candidate_instances[0].closed is True
    assert args.dataset_startup_error == "RuntimeError: writer ready timeout"
    assert clutch.recording_states[-1]["event_status"] == "error"
    assert any(
        tag == "Dataset" and "hardware-status refresh failed" in message
        for tag, message in logs
    )
    runtime.close()


def test_dataset_selection_replacement_is_transactional_and_nonblocking(
    tmp_path: Path,
) -> None:
    candidate_entered = threading.Event()
    release_candidate = threading.Event()
    retirement_entered = threading.Event()
    release_retirement = threading.Event()
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.dataset_root = Path(args.dataset_root)
            self.writer_started = False
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            candidate_entered.set()
            assert release_candidate.wait(timeout=1.0)
            self.writer_started = True

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    old_root = tmp_path / "old"
    selected_root = tmp_path / "selected"
    old_recorder = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=old_root,
        writer_started=True,
        control_trace_episode_sink=None,
        v25p_sidecar_store=None,
        poll_finalize_tasks=list,
    )

    def retire_old() -> None:
        retirement_entered.set()
        assert release_retirement.wait(timeout=1.0)
        old_recorder.closed = True

    old_recorder.closed = False
    old_recorder.close = retire_old
    slot: list[object | None] = [old_recorder]
    clutch = _Clutch()
    transition_calls: list[str] = []
    runtime, args, _logs = _episode_runtime(
        tmp_path=old_root,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=transition_calls,
    )
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = selected_root
    callbacks: list[str] = []

    def commit_selection(snapshot: dict[str, object]) -> None:
        callbacks.append("commit")
        vars(args).clear()
        vars(args).update(snapshot)

    started_at = time.perf_counter()
    outcome = runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=commit_selection,
        on_complete=lambda: callbacks.append("complete"),
        on_failure=lambda exc: callbacks.append(f"failure:{exc}"),
    )

    assert outcome == "pending"
    assert time.perf_counter() - started_at < 0.05
    assert candidate_entered.wait(timeout=1.0)
    assert slot[0] is None
    assert args.dataset_root == old_root
    assert callbacks == []
    assert runtime.collection_transition_active()

    release_candidate.set()
    deadline = time.monotonic() + 1.0
    while not retirement_entered.is_set() and time.monotonic() < deadline:
        poll_started_at = time.perf_counter()
        runtime.poll_dataset_finalize(time.time())
        assert time.perf_counter() - poll_started_at < 0.05
        time.sleep(0.001)

    assert retirement_entered.is_set()
    assert slot[0] is instances[0]
    assert instances[0].args is args
    assert args.dataset_root == selected_root
    assert callbacks == ["commit"]
    assert runtime.collection_transition_active()
    assert old_recorder.closed is False

    release_retirement.set()
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert callbacks == ["commit", "complete"]
    assert old_recorder.closed is True
    assert transition_calls == ["dataset_selection_reconfigure"]
    runtime.close()


def test_dataset_selection_retirement_start_failure_is_retained_and_retried(
    tmp_path: Path,
    monkeypatch,
) -> None:
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.writer_started = False
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            self.writer_started = True

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    old_recorder = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path / "old",
        writer_started=True,
        control_trace_episode_sink=None,
        v25p_sidecar_store=None,
        closed=False,
        poll_finalize_tasks=list,
    )
    old_recorder.close = lambda: setattr(old_recorder, "closed", True)
    slot: list[object | None] = [old_recorder]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=old_recorder.dataset_root,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = tmp_path / "selected"
    callbacks: list[str] = []
    original_start = control_plane_module.threading.Thread.start
    start_attempts = 0

    def fail_retirement_once(thread) -> None:
        nonlocal start_attempts
        start_attempts += 1
        if start_attempts == 2:
            raise RuntimeError("simulated retirement thread exhaustion")
        original_start(thread)

    monkeypatch.setattr(
        control_plane_module.threading.Thread,
        "start",
        fail_retirement_once,
    )

    def commit_selected_state(snapshot) -> None:
        callbacks.append("commit")
        vars(args).clear()
        vars(args).update(snapshot)

    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=commit_selected_state,
        on_complete=lambda: callbacks.append("complete"),
        on_failure=lambda exc: callbacks.append(f"failure:{exc}"),
    ) == "pending"

    deadline = time.monotonic() + 1.0
    while callbacks != ["commit"] and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert callbacks == ["commit"]
    assert slot[0] is instances[0]
    assert not old_recorder.closed
    assert runtime.collection_transition_active()
    assert any(
        tag == "Dataset"
        and "retirement is retained for retry" in message
        and "simulated retirement thread exhaustion" in message
        for tag, message in logs
    )

    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.005)

    assert callbacks == ["commit", "complete"]
    assert old_recorder.closed
    assert not runtime.collection_transition_active()
    assert start_attempts == 3
    assert runtime.close()


def test_dataset_selection_retirement_close_failure_keeps_owned_retry(
    tmp_path: Path,
) -> None:
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.writer_started = False
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            self.writer_started = True

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    old_recorder = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path / "old",
        writer_started=True,
        control_trace_episode_sink=None,
        v25p_sidecar_store=None,
        closed=False,
        close_calls=0,
        poll_finalize_tasks=list,
    )

    def close_old_recorder() -> None:
        old_recorder.close_calls += 1
        if old_recorder.close_calls == 1:
            raise RuntimeError("simulated transient Queue teardown failure")
        old_recorder.closed = True

    old_recorder.close = close_old_recorder
    slot: list[object | None] = [old_recorder]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=old_recorder.dataset_root,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = tmp_path / "selected"
    callbacks: list[str] = []

    def commit_selected_state(snapshot) -> None:
        callbacks.append("commit")
        vars(args).clear()
        vars(args).update(snapshot)

    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=commit_selected_state,
        on_complete=lambda: callbacks.append("complete"),
        on_failure=lambda exc: callbacks.append(f"failure:{exc}"),
    ) == "pending"

    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        if any(
            tag == "Dataset"
            and "old recorder remains owned" in message
            and "simulated transient Queue teardown failure" in message
            for tag, message in logs
        ):
            break
        time.sleep(0.001)

    assert callbacks == ["commit"]
    assert slot[0] is instances[0]
    assert old_recorder.close_calls == 1
    assert not old_recorder.closed
    assert runtime.collection_transition_active()

    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.005)

    assert callbacks == ["commit", "complete"]
    assert old_recorder.close_calls == 2
    assert old_recorder.closed
    assert not runtime.collection_transition_active()
    assert runtime.close()


def test_failed_dataset_selection_republishes_old_recorder_and_keeps_args(
    tmp_path: Path,
) -> None:
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            raise RuntimeError("selection writer ready timeout")

        def close(self) -> None:
            self.closed = True

    old_recorder = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path / "old",
        writer_started=True,
        control_trace_episode_sink=None,
        v25p_sidecar_store=None,
        closed=False,
        poll_finalize_tasks=list,
    )
    old_recorder.close = lambda: setattr(old_recorder, "closed", True)
    slot: list[object | None] = [old_recorder]
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path / "old",
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=[],
    )
    old_args = dict(vars(args))
    selected_args = dict(old_args)
    selected_args["dataset_root"] = tmp_path / "selected"
    failures: list[BaseException] = []

    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=lambda _snapshot: pytest.fail("failed candidate committed"),
        on_complete=lambda: pytest.fail("failed selection completed"),
        on_failure=failures.append,
    ) == "pending"
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is old_recorder
    assert vars(args) == old_args
    assert old_recorder.closed is False
    assert len(instances) == 1
    assert instances[0].closed is True
    assert len(failures) == 1
    assert str(failures[0]) == "selection writer ready timeout"
    runtime.close()


def test_dataset_selection_status_publish_failure_cannot_skip_old_retirement(
    tmp_path: Path,
) -> None:
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            return None

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    old_recorder = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path / "old",
        writer_started=True,
        control_trace_episode_sink=None,
        v25p_sidecar_store=None,
        closed=False,
        poll_finalize_tasks=list,
    )
    old_recorder.close = lambda: setattr(old_recorder, "closed", True)
    slot: list[object | None] = [old_recorder]
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path / "old",
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=[],
        publish_runtime_hardware_status=lambda: (_ for _ in ()).throw(
            RuntimeError("status transport unavailable")
        ),
    )
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = tmp_path / "selected"
    callbacks: list[str] = []

    def commit(snapshot: dict[str, object]) -> None:
        vars(args).clear()
        vars(args).update(snapshot)
        callbacks.append("commit")

    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=commit,
        on_complete=lambda: callbacks.append("complete"),
        on_failure=lambda exc: callbacks.append(f"failure:{exc}"),
    ) == "pending"
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is instances[0]
    assert old_recorder.closed is True
    assert callbacks == ["commit", "complete"]
    assert not runtime.collection_transition_active()
    assert any(
        tag == "Dataset" and "hardware-status refresh failed" in message
        for tag, message in logs
    )
    runtime.close()


def test_first_dataset_selection_completion_callback_failure_is_observational(
    tmp_path: Path,
) -> None:
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.closed = False
            instances.append(self)

        def ensure_started(self) -> None:
            return None

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    slot: list[object | None] = [None]
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path / "old",
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=[],
    )
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = tmp_path / "selected"
    callbacks: list[str] = []

    def commit(snapshot: dict[str, object]) -> None:
        vars(args).clear()
        vars(args).update(snapshot)
        callbacks.append("commit")

    def fail_after_commit() -> None:
        callbacks.append("complete")
        raise RuntimeError("ui publication unavailable")

    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=commit,
        on_complete=fail_after_commit,
        on_failure=lambda exc: callbacks.append(f"failure:{exc}"),
    ) == "pending"
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert callbacks == ["commit", "complete"]
    assert slot[0] is instances[0]
    assert args.dataset_root == tmp_path / "selected"
    assert not runtime.collection_transition_active()
    assert any(
        tag == "Dataset" and "completion callback failed" in message
        for tag, message in logs
    )
    runtime.close()


def test_dataset_selection_is_single_flight_and_no_recorder_prepares_writer_async(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    instances: list[object] = []

    class Recorder:
        def __init__(self, *, args: object, **_kwargs: object) -> None:
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.closed = False
            self.is_recording = False
            self.finalizing = False
            self.frame_count = 0
            self.writer_started = True
            self.device_config_writes = 0
            instances.append(self)

        def ensure_started(self) -> None:
            entered.set()
            assert release.wait(timeout=1.0)

        def close(self) -> None:
            self.closed = True

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def configured_for_current_args(self) -> bool:
            return True

        def write_camera_device_config(self) -> None:
            self.device_config_writes += 1

    old_recorder = SimpleNamespace(
        finalizing=False,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path / "old",
        writer_started=True,
        control_trace_episode_sink=None,
        v25p_sidecar_store=None,
        closed=False,
        poll_finalize_tasks=list,
    )
    old_recorder.close = lambda: setattr(old_recorder, "closed", True)
    slot: list[object | None] = [old_recorder]
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path / "old",
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=[],
    )
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = tmp_path / "selected"
    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=lambda _snapshot: None,
        on_complete=lambda: None,
        on_failure=lambda _exc: None,
    ) == "pending"
    assert entered.wait(timeout=1.0)
    with pytest.raises(CollectionControlOperationBusy, match="prepare_selection"):
        runtime.schedule_dataset_selection(
            selected_args,
            bind_dataset_contract=False,
            on_commit=lambda _snapshot: None,
            on_complete=lambda: None,
            on_failure=lambda _exc: None,
        )
    runtime.close()
    release.set()
    deadline = time.monotonic() + 1.0
    while (
        (not old_recorder.closed or not instances[0].closed)
        and time.monotonic() < deadline
    ):
        time.sleep(0.001)
    assert old_recorder.closed is True
    assert instances[0].closed is True

    empty_slot: list[object | None] = [None]
    empty_transition_calls: list[str] = []
    inline_callbacks: list[str] = []
    contract_entered = threading.Event()
    contract_release = threading.Event()

    def bind_contract(*_args: object, **_kwargs: object) -> None:
        contract_entered.set()
        assert contract_release.wait(timeout=1.0)

    empty_runtime, empty_args, _logs = _episode_runtime(
        tmp_path=tmp_path / "empty",
        recorder_type=Recorder,
        dataset_slot=empty_slot,
        clutch=_Clutch(),
        transition_calls=empty_transition_calls,
        bind_capture_modality_contract=bind_contract,
    )
    inline_snapshot = dict(vars(empty_args))
    inline_snapshot["dataset_root"] = tmp_path / "inline"

    def inline_commit(snapshot: dict[str, object]) -> None:
        vars(empty_args).clear()
        vars(empty_args).update(snapshot)
        inline_callbacks.append("commit")

    assert empty_runtime.schedule_dataset_selection(
        inline_snapshot,
        bind_dataset_contract=True,
        on_commit=inline_commit,
        on_complete=lambda: inline_callbacks.append("complete"),
        on_failure=lambda exc: inline_callbacks.append(f"failure:{exc}"),
    ) == "pending"
    assert contract_entered.wait(timeout=1.0)
    assert inline_callbacks == []
    assert empty_args.dataset_root == tmp_path / "empty"
    assert empty_slot[0] is None
    assert len(instances) == 1

    contract_release.set()
    deadline = time.monotonic() + 1.0
    while (
        empty_runtime.collection_transition_active()
        and time.monotonic() < deadline
    ):
        empty_runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert inline_callbacks == ["commit", "complete"]
    assert empty_args.dataset_root == tmp_path / "inline"
    assert empty_slot[0] is instances[1]
    assert instances[1].args is empty_args
    assert instances[1].dataset_root == tmp_path / "inline"
    assert empty_transition_calls == ["dataset_selection_reconfigure"]
    assert len(instances) == 2

    assert empty_runtime.schedule_recording_countdown(
        time.time(),
        start_reason="unit_after_selection",
    )
    deadline = time.monotonic() + 1.0
    while empty_runtime.collection_transition_active() and time.monotonic() < deadline:
        empty_runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)
    assert empty_slot[0] is instances[1]
    assert len(instances) == 2
    assert instances[1].device_config_writes == 1

    empty_runtime.close()
    # EpisodeRecordingRuntime closes only detached/in-flight candidates.  The
    # published idle recorder remains owned by the outer real-teleop runtime,
    # whose ordered shutdown closes it after the collection plane drains.
    assert instances[1].closed is False
    instances[1].close()
    assert instances[1].closed is True


def test_dataset_selection_without_formal_collection_remains_contract_only(
    tmp_path: Path,
) -> None:
    instances: list[object] = []

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            instances.append(self)

    slot: list[object | None] = [None]
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path / "initial",
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=[],
    )
    args.no_collection = True
    selected_args = dict(vars(args))
    selected_args["dataset_root"] = tmp_path / "selected"
    callbacks: list[str] = []

    def commit(snapshot: dict[str, object]) -> None:
        vars(args).clear()
        vars(args).update(snapshot)
        callbacks.append("commit")

    assert runtime.schedule_dataset_selection(
        selected_args,
        bind_dataset_contract=False,
        on_commit=commit,
        on_complete=lambda: callbacks.append("complete"),
        on_failure=lambda exc: callbacks.append(f"failure:{exc}"),
    ) == "pending"
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert callbacks == ["commit", "complete"]
    assert args.dataset_root == tmp_path / "selected"
    assert slot[0] is None
    assert instances == []
    runtime.close()


@pytest.mark.parametrize("cancel_cut", ("ready_handshake", "publication"))
def test_cancelled_prepare_never_publishes_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_cut: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    instances: list[object] = []
    plane = CollectionControlPlane(name="prepare-publication-test")
    cut = _PauseWorkerAfterUnlock()
    if cancel_cut == "publication":
        plane._lock = cut
    monkeypatch.setattr(episode_recording_module, "CollectionControlPlane", lambda **_kwargs: plane)

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.dataset_root = tmp_path
            self.writer_started = False
            self.closed = False
            self.close_calls = 0
            instances.append(self)

        def ensure_started(self) -> None:
            if cancel_cut == "publication":
                cut.worker = threading.current_thread()
            else:
                entered.set()
                assert release.wait(timeout=1.0)
            self.writer_started = True

        def poll_finalize_tasks(self):
            return []

        def close(self) -> None:
            self.closed = True
            self.close_calls += 1

    slot: list[object | None] = [None]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert runtime.schedule_recording_countdown(time.time(), start_reason="unit")
    try:
        ready = cut.reached if cancel_cut == "publication" else entered
        assert ready.wait(timeout=1.0)
        assert runtime.cancel_recording_countdown(time.time(), reason="unit_cancel")
        assert runtime.countdown_deadline() == float("inf")
        release.set()
        cut.release.set()
        deadline = time.monotonic() + 1.0
        while runtime.countdown_deadline() is not None and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(0.001)

        assert slot[0] is None
        assert instances[0].closed is True
        assert instances[0].close_calls == 1
        assert runtime.countdown_deadline() is None
        assert clutch.recording_states[-1]["event_status"] == "idle"
    finally:
        release.set()
        cut.release.set()
        runtime.close()


@pytest.mark.parametrize(
    ("origin", "cleanup_failure", "cancel_reason"),
    [
        ("new", None, "operator_hold"),
        ("replacement", None, "operator_hold"),
        ("reused", None, "operator_hold"),
        ("new", "submit", "operator_hold"),
        ("replacement", "submit", "operator_hold"),
        ("new", "close", "operator_hold"),
        ("replacement", "close", "operator_hold"),
        pytest.param("new", None, "", id="empty_reason"),
    ],
)
def test_prepare_cancel_launch_failure_keeps_candidate_out_of_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    origin: str,
    cleanup_failure: str | None,
    cancel_reason: str,
) -> None:
    release = threading.Event()
    instances: list[object] = []
    candidate = None
    plane = CollectionControlPlane(name="prepare-cancel-failure")
    monkeypatch.setattr(
        episode_recording_module, "CollectionControlPlane", lambda **_kwargs: plane,
    )

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            self.dataset_root = tmp_path
            self.finalizing = False
            self.is_recording = False
            self.is_paused = False
            self.frame_count = 0
            self.writer_started = False
            self.closed = False
            self.close_calls = 0
            instances.append(self)

        def configured_for_current_args(self) -> bool:
            return origin == "reused"

        def ensure_started(self) -> None:
            self.writer_started = True

        def write_camera_device_config(self) -> None:
            pass

        def poll_finalize_tasks(self):
            return []

        def close(self) -> None:
            self.close_calls += 1
            if self is candidate:
                if cleanup_failure == "close" and self.close_calls == 1:
                    raise RuntimeError("simulated candidate close failure")
                assert release.wait(timeout=1.0)
            self.closed = True

    previous = None if origin == "new" else Recorder()
    if previous is not None:
        previous.writer_started = True
    slot: list[object | None] = [previous]
    clutch = _Clutch()
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    def fail_start(_thread: threading.Thread) -> None:
        raise RuntimeError("simulated candidate worker unavailable")

    try:
        assert runtime.schedule_recording_countdown(time.time(), start_reason="unit")
        deadline = time.monotonic() + 1.0
        while plane.busy and time.monotonic() < deadline:
            time.sleep(0.001)
        assert not plane.busy and plane.active_kind == "prepare_writer"
        candidate = instances[-1]
        assert slot[0] is None
        with monkeypatch.context() as failure_patch:
            failure_patch.setattr(control_plane_module.threading.Thread, "start", fail_start)
            assert runtime.cancel_recording_countdown(time.time(), reason=cancel_reason)
            if cleanup_failure == "submit":
                runtime.poll_dataset_finalize(time.time())
        if cleanup_failure != "submit":
            started_at = time.perf_counter()
            runtime.poll_dataset_finalize(time.time())
            assert time.perf_counter() - started_at < 0.05

        if origin == "reused":
            # Cancellation now retains a failed launch in the same plane;
            # even a no-op discard is consumed asynchronously before reuse.
            deadline = time.monotonic() + 1.0
            while slot[0] is None and time.monotonic() < deadline:
                runtime.poll_dataset_finalize(time.time())
                time.sleep(0.001)
            assert slot[0] is previous
            assert candidate.close_calls == 0
        else:
            assert slot[0] is None
            assert not candidate.closed
            assert runtime.collection_transition_active()
            assert not runtime.schedule_recording_countdown(time.time(), start_reason="retry")
            if previous is not None:
                assert previous.closed
                assert previous.close_calls == 1
            if cleanup_failure is not None:
                deadline = time.monotonic() + 1.0
                while not any("simulated candidate" in message for _, message in logs):
                    assert time.monotonic() < deadline
                    runtime.poll_dataset_finalize(time.time())
                    time.sleep(0.001)
            release.set()
            poll_recording_transition(runtime, timeout_sec=2.0)
            assert candidate.closed
            assert candidate.close_calls == (2 if cleanup_failure == "close" else 1)
            assert slot[0] is None

        assert not runtime.collection_transition_active()
        assert runtime.countdown_deadline() is None
        assert clutch.recording_states[-1]["event_status"] == "idle"
        # A new admission must not inherit the cancelled transaction's reason.
        assert runtime.schedule_recording_countdown(time.time(), start_reason="new_request")
        poll_recording_transition(runtime)
        assert slot[0] is not None
        assert not slot[0].closed
        assert runtime.countdown_deadline() is not None
        assert runtime.cancel_recording_countdown(time.time(), reason="test_cleanup")
    finally:
        release.set()
        runtime.close(timeout_sec=1.0)


def test_episode_stop_drain_is_nonblocking_and_committed_on_poll(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
    stop_calls: list[dict[str, object]] = []
    finalize_results: list[object] = []
    finalize_task = SimpleNamespace(
        episode_index=4,
        frame_count=17,
        queued_at=10.0,
        started_at=10.1,
        completed_at=10.2,
        stage_timings_ms={},
        success=True,
        error="",
        v25p_error="",
        v25p_native_error="",
        progress_error="",
    )

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("stop test must reuse the published recorder")

    recorder = SimpleNamespace(
        finalizing=False,
        is_recording=True,
        is_paused=False,
        frame_count=17,
        dataset_root=tmp_path,
        collection_frequency_status=lambda: {"sample_hz": 15.0},
        poll_finalize_tasks=(
            lambda: [finalize_results.pop(0)] if finalize_results else []
        ),
    )

    def stop_episode(**kwargs: object):
        stop_calls.append(dict(kwargs))
        entered.set()
        assert release.wait(timeout=1.0)
        recorder.is_recording = False
        return finalize_task

    recorder.stop_episode = stop_episode
    recorder.close = lambda: None
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        discard_episode_on_stop=lambda _reason: False,
    )

    started_at = time.perf_counter()
    assert runtime.stop_recording(
        time.time(),
        end_reason="unit",
        summary={"reason": "test"},
    )
    assert time.perf_counter() - started_at < 0.05
    assert entered.wait(timeout=1.0)
    assert "discard" not in stop_calls[0]
    assert not runtime.recording_active()
    assert clutch.recording_states[-1]["event_status"] == "stopping"

    release.set()
    deadline = time.monotonic() + 1.0
    while clutch.recording_states[-1]["event_status"] == "stopping" and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert clutch.recording_states[-1]["event_status"] == "finalizing"
    assert clutch.recording_states[-1]["episode_index"] == 4
    assert clutch.round_trip_advances == 0

    finalize_results.append(finalize_task)
    runtime.poll_dataset_finalize(time.time())
    assert clutch.finalize_states[-1]["status"] == "saved"
    assert clutch.round_trip_advances == 1
    assert clutch.plus_episode_indices == [4]
    assert runtime.close()


def test_rule_abort_discards_nonempty_episode_at_writer_owner_boundary(
    tmp_path: Path,
) -> None:
    stop_calls: list[dict[str, object]] = []
    policy_thread_ids: list[int] = []
    control_thread_id = threading.get_ident()

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("stop test must reuse the published recorder")

    recorder = SimpleNamespace(
        finalizing=False,
        is_recording=True,
        is_paused=False,
        frame_count=17,
        active_episode_index=4,
        dataset_root=tmp_path,
        collection_frequency_status=lambda: {"sample_hz": 15.0},
        poll_finalize_tasks=lambda: [],
    )

    def stop_episode(**kwargs: object) -> None:
        stop_calls.append(dict(kwargs))
        recorder.is_recording = False

    recorder.stop_episode = stop_episode
    recorder.close = lambda: None
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        discard_episode_on_stop=lambda reason: (
            policy_thread_ids.append(threading.get_ident())
            or reason == "rule_based_abort"
        ),
    )

    assert runtime.stop_recording(time.time(), end_reason="rule_based_abort")
    assert policy_thread_ids == [control_thread_id]
    deadline = time.monotonic() + 1.0
    while (
        clutch.recording_states[-1]["event_status"] == "stopping"
        and time.monotonic() < deadline
    ):
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert len(stop_calls) == 1
    assert stop_calls[0]["reason"] == "rule_based_abort"
    assert stop_calls[0]["discard"] is True
    assert clutch.recording_states[-1]["event_status"] == "discarded"
    assert "17 帧已全部丢弃" in clutch.recording_states[-1]["event_message"]
    assert runtime.close()


def test_failed_finalize_does_not_advance_round_trip_task(tmp_path: Path) -> None:
    failed_task = SimpleNamespace(
        episode_index=5,
        frame_count=11,
        queued_at=10.0,
        started_at=10.1,
        completed_at=10.2,
        stage_timings_ms={},
        success=False,
        error="parquet flush failed",
        v25p_error="",
        v25p_native_error="",
        progress_error="",
    )
    finalize_results: list[object] = [failed_task]

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("finalize test must reuse the published recorder")

    recorder = SimpleNamespace(
        finalizing=True,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path,
        close=lambda: None,
    )

    def poll_failed_finalize_tasks() -> list[object]:
        if not finalize_results:
            recorder.finalizing = False
            return []
        recorder.finalizing = False
        return [finalize_results.pop(0)]

    recorder.poll_finalize_tasks = poll_failed_finalize_tasks
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    runtime.poll_dataset_finalize(time.time())

    assert clutch.finalize_states[-1]["status"] == "error"
    assert clutch.round_trip_advances == 0
    assert clutch.plus_episode_indices == []
    assert runtime.close()


def test_saved_finalize_consumers_fail_independently_without_rewriting_writer_result(
    tmp_path: Path,
) -> None:
    finalize_task = SimpleNamespace(
        episode_index=6,
        frame_count=19,
        queued_at=10.0,
        started_at=10.1,
        completed_at=10.2,
        stage_timings_ms={},
        success=True,
        error="",
        v25p_error="",
        v25p_native_error="",
        progress_error="",
    )
    finalize_results: list[object] = [finalize_task]
    recorder = SimpleNamespace(
        finalizing=True,
        is_recording=False,
        is_paused=False,
        frame_count=0,
        dataset_root=tmp_path,
        close=lambda: None,
    )

    def poll_finalize_tasks() -> list[object]:
        recorder.finalizing = False
        return [finalize_results.pop(0)] if finalize_results else []

    recorder.poll_finalize_tasks = poll_finalize_tasks

    class FailingFinalizeConsumers(_Clutch):
        def __init__(self) -> None:
            super().__init__()
            self.consumer_calls: list[str] = []

        def set_episode_finalize_state(self, **state: object) -> None:
            self.consumer_calls.append(f"ui:{state['status']}")
            raise RuntimeError("finalize projection unavailable")

        def advance_round_trip_task_after_saved_episode(self) -> None:
            self.consumer_calls.append("round_trip")
            raise RuntimeError("round-trip state unavailable")

        def set_plus_curation_pending(
            self,
            *,
            dataset_root: Path,
            episode_index: int,
        ) -> None:
            assert dataset_root == tmp_path
            assert episode_index == 6
            self.consumer_calls.append("plus")
            raise RuntimeError("Plus state unavailable")

    clutch = FailingFinalizeConsumers()
    slot: list[object | None] = [recorder]
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=lambda **_kwargs: None,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    runtime.poll_dataset_finalize(time.time())

    assert clutch.consumer_calls == ["ui:saved", "round_trip", "plus"]
    assert sum("Writer 结果保持不变" in message for _tag, message in logs) == 3
    assert any("正式数据集 episode 6 已保存" in message for _tag, message in logs)
    # No finalize failure was fabricated from an application-side consumer,
    # so orderly shutdown still reports the writer lifecycle as successful.
    assert runtime.close()


def test_close_stops_and_publishes_an_active_episode_before_plane_shutdown(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    stop_calls: list[dict[str, object]] = []
    finalize_task = SimpleNamespace(
        episode_index=7,
        frame_count=23,
        queued_at=10.0,
        started_at=10.1,
        completed_at=10.2,
        stage_timings_ms={},
        success=True,
        error="",
        v25p_error="",
        v25p_native_error="",
        progress_error="",
    )

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("shutdown test must reuse the published recorder")

    finalize_results: list[object] = []
    recorder = SimpleNamespace(
        finalizing=False,
        is_recording=True,
        is_paused=False,
        frame_count=23,
        active_episode_index=7,
        dataset_root=tmp_path,
        collection_frequency_status=lambda: {"sample_hz": 15.0},
    )

    def poll_finalize_tasks() -> list[object]:
        if not finalize_results:
            return []
        recorder.finalizing = False
        return [finalize_results.pop(0)]

    recorder.poll_finalize_tasks = poll_finalize_tasks

    def stop_episode(**kwargs: object):
        stop_calls.append(dict(kwargs))
        entered.set()
        assert release.wait(timeout=1.0)
        recorder.is_recording = False
        recorder.finalizing = True
        finalize_results.append(finalize_task)
        return finalize_task

    recorder.stop_episode = stop_episode
    recorder.close = lambda: None
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )
    releaser = threading.Timer(0.05, release.set)
    releaser.start()
    try:
        assert runtime.close(
            now=123.0,
            end_reason="shutdown",
            summary={"reason": "finally"},
            timeout_sec=1.0,
        )
    finally:
        releaser.join(timeout=1.0)

    assert entered.is_set()
    assert stop_calls == [
        {
            "reason": "shutdown",
            "step_index": 0,
            "last_state": None,
            "summary": {"reason": "finally"},
            "defer_finalize_start": False,
            "before_finalize": None,
        }
    ]
    assert slot[0] is recorder
    assert clutch.recording_states[-1]["event_status"] == "finalizing"
    assert clutch.recording_states[-1]["episode_index"] == 7
    assert clutch.finalize_states[-1]["status"] == "saved"
    assert clutch.round_trip_advances == 1


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_failed_stop_isolated_recorder_and_allows_fresh_retry(tmp_path: Path, cleanup_fails: bool) -> None:
    fail_cleanup = cleanup_fails
    entered = threading.Event()
    release = threading.Event()
    recorder = SimpleNamespace(
        finalizing=False,
        is_recording=True,
        is_paused=False,
        frame_count=5,
        dataset_root=tmp_path,
        collection_frequency_status=lambda: {"sample_hz": 15.0},
        poll_finalize_tasks=lambda: [],
        closed=False,
    )

    def stop_episode(**_kwargs: object):
        entered.set()
        assert release.wait(timeout=1.0)
        raise RuntimeError("sample drain failed")

    recorder.stop_episode = stop_episode
    def close_recorder():
        if fail_cleanup:
            raise RuntimeError("stop cleanup unavailable")
        recorder.closed = True
    recorder.close = close_recorder

    class Recorder:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("fresh retry is outside this stop test")

    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
    )

    assert runtime.stop_recording(time.time(), end_reason="unit")
    assert entered.wait(timeout=1.0)
    release.set()
    if cleanup_fails:
        deadline = time.monotonic() + 1.0
        while not args.dataset_startup_error and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(.001)
        assert runtime.collection_transition_active()
        assert slot[0] is None and not recorder.closed
        fail_cleanup = False
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert recorder.closed is True
    assert slot[0] is None
    assert clutch.recording_states[-1]["event_status"] == "error"
    assert clutch.hold_reasons == ["dataset_stop_failure"]
    assert args.dataset_startup_error == "RuntimeError: sample drain failed"
    runtime.close()



@pytest.mark.parametrize("failure_point", ("start", "close"))
def test_cancelled_ready_prepare_retains_cleanup_and_never_admits_candidate(tmp_path, monkeypatch, failure_point):
    instances = []
    fail_cleanup = True
    plane = CollectionControlPlane(name="retained-prepare-test")
    monkeypatch.setattr(episode_recording_module, "CollectionControlPlane", lambda **_: plane)

    class Recorder:
        def __init__(self, **_):
            self.finalizing = self.is_recording = self.is_paused = False
            self.frame_count = self.close_calls = 0
            self.dataset_root = tmp_path
            self.writer_started = self.closed = False
            instances.append(self)
        def ensure_started(self):
            self.writer_started = True
        def poll_finalize_tasks(self):
            return []
        def close(self):
            self.close_calls += 1
            if failure_point == "close" and fail_cleanup:
                raise RuntimeError("candidate close unavailable")
            self.closed = True

    slot, clutch = [None], _Clutch()
    runtime, _args, logs = _episode_runtime(tmp_path=tmp_path, recorder_type=Recorder,
        dataset_slot=slot, clutch=clutch, transition_calls=[])
    assert runtime.schedule_recording_countdown(time.time(), start_reason="unit")
    deadline = time.monotonic() + 1.0
    while plane.busy and time.monotonic() < deadline:
        time.sleep(.001)
    assert not plane.busy
    try:
        with monkeypatch.context() as failure_patch:
            if failure_point == "start":
                def fail_start(_thread):
                    raise RuntimeError("cleanup thread unavailable")
                failure_patch.setattr(control_plane_module.threading.Thread, "start", fail_start)
            assert runtime.cancel_recording_countdown(time.time(), reason="operator_cancel")
            deadline = time.monotonic() + 1.0
            while plane.cleanup_error is None and time.monotonic() < deadline:
                runtime.poll_dataset_finalize(time.time())
                time.sleep(.001)
            runtime.poll_dataset_finalize(time.time())
            assert plane.cleanup_error is not None
            assert slot[0] is None
            assert runtime.collection_transition_active()
            assert runtime.countdown_deadline() == float("inf")
            assert not instances[0].closed
            if failure_point == "close":
                assert clutch.recording_states[-1]["event_status"] == "error"
            assert any("回收" in str(message) for message in logs)
        fail_cleanup = False
        deadline = time.monotonic() + 1.0
        while runtime.collection_transition_active() and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(.001)
        assert slot[0] is None
        assert instances[0].closed
        assert runtime.countdown_deadline() is None
        assert plane.cleanup_error is None
        assert clutch.recording_states[-1]["event_status"] == "idle"
    finally:
        fail_cleanup = False
        runtime.close()


def test_retained_cleanup_poll_retries_are_rate_limited(monkeypatch):
    plane = CollectionControlPlane(name="retry-budget-test")
    generation = plane.submit("prepare", object, discard_value=lambda _: None)
    deadline = time.monotonic() + 1.0
    while plane.busy and time.monotonic() < deadline:
        time.sleep(.001)
    clock, attempts = [0.0], []
    def fail_start(_):
        attempts.append(clock[0])
        raise RuntimeError("unavailable")
    with monkeypatch.context() as patch:
        patch.setattr(control_plane_module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
        patch.setattr(control_plane_module.threading.Thread, "start", fail_start)
        assert not plane.cancel(generation)
        for _ in range(100):
            assert plane.poll(generation) is None
        assert attempts == [0.0]
        clock[0] = .051
        assert plane.poll(generation) is None
        assert attempts == [0.0, .051]
    assert plane.close(timeout_sec=1.0)



def test_failed_ready_handshake_retains_candidate_until_failed_close_recovers(tmp_path, monkeypatch):
    instances, fail_cleanup = [], [True]
    plane = CollectionControlPlane(name="failed-ready-cleanup")
    monkeypatch.setattr(episode_recording_module, "CollectionControlPlane", lambda **_: plane)
    class Recorder:
        def __init__(self, **_):
            self.finalizing = self.is_recording = self.is_paused = False
            self.frame_count = 0
            self.dataset_root = tmp_path
            self.writer_started = self.closed = False
            instances.append(self)
        def ensure_started(self):
            raise RuntimeError("ready handshake failed")
        def close(self):
            if fail_cleanup[0]:
                raise RuntimeError("failed candidate close unavailable")
            self.closed = True
        def poll_finalize_tasks(self):
            return []
    slot, clutch = [None], _Clutch()
    runtime, args, _ = _episode_runtime(tmp_path=tmp_path, recorder_type=Recorder,
        dataset_slot=slot, clutch=clutch, transition_calls=[])
    try:
        assert runtime.schedule_recording_countdown(time.time(), start_reason="unit")
        deadline = time.monotonic() + 1.0
        while not args.dataset_startup_error and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(.001)
        assert args.dataset_startup_error == "RuntimeError: ready handshake failed"
        assert plane.cleanup_error is not None
        assert runtime.collection_transition_active()
        assert slot[0] is None and not instances[0].closed
        fail_cleanup[0] = False
        deadline = time.monotonic() + 1.0
        while runtime.collection_transition_active() and time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            time.sleep(.001)
        assert not runtime.collection_transition_active()
        assert instances[0].closed
        assert slot[0] is None and runtime.countdown_deadline() is None
        assert args.dataset_startup_error == "RuntimeError: ready handshake failed"
    finally:
        fail_cleanup[0] = False
        runtime.close()


@pytest.mark.parametrize("kind", ["prepare_writer", "prepare_selection"])
@pytest.mark.parametrize("cut", ["completed", "running"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_prepare_shutdown_retires_detached_previous_without_losing_owner(
    tmp_path: Path, monkeypatch, kind: str, cut: str, close_fails: bool,
) -> None:
    entered, release = threading.Event(), threading.Event()
    caller = threading.get_ident()
    plane = CollectionControlPlane(name="failed-prepare-shutdown")
    monkeypatch.setattr(episode_recording_module, "CollectionControlPlane", lambda **_kwargs: plane)
    close_threads = []
    candidates = []

    def fail_prepare():
        entered.set()
        assert release.wait(timeout=2.0)
        raise OSError("original writer preparation failure")

    class Candidate:
        def __init__(self, **_kwargs):
            self.dataset_root = tmp_path
            self.close_calls = 0
            candidates.append(self)

        def ensure_started(self):
            fail_prepare()

        def close(self):
            self.close_calls += 1

    previous = SimpleNamespace(
        dataset_root=tmp_path, is_recording=False, is_paused=False,
        frame_count=0, finalizing=False, writer_started=True,
        control_trace_episode_sink=None, v25p_sidecar_store=None,
        configured_for_current_args=lambda: True,
        write_camera_device_config=fail_prepare,
        poll_finalize_tasks=list, closed=False, close_calls=0,
    )

    def close_previous():
        close_threads.append(threading.get_ident())
        previous.close_calls += 1
        if close_fails and previous.close_calls == 1:
            raise RuntimeError("old recorder close remains pending")
        previous.closed = True

    previous.close = close_previous
    slot = [previous]
    clutch = _Clutch()
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path, recorder_type=Candidate, dataset_slot=slot,
        clutch=clutch, transition_calls=[],
    )
    failures = []
    callbacks = []
    try:
        if kind == "prepare_writer":
            assert runtime.schedule_recording_countdown(time.time(), start_reason="shutdown-test")
        else:
            assert runtime.schedule_dataset_selection(
                dict(vars(args)), bind_dataset_contract=False,
                on_commit=lambda _snapshot: callbacks.append("commit"),
                on_complete=lambda: callbacks.append("complete"), on_failure=failures.append,
            ) == "pending"
        assert entered.wait(timeout=1.0) and slot[0] is None
        if cut == "completed":
            release.set()
            deadline = time.monotonic() + 1.0
            while plane.busy and time.monotonic() < deadline:
                time.sleep(0.001)
            assert not plane.busy
        else:
            started = time.monotonic()
            assert runtime.close(timeout_sec=0.02) is False
            assert time.monotonic() - started < 0.25
            assert runtime.collection_transition_active()
            assert previous.close_calls == 0
            release.set()
        deadline = time.monotonic() + 1.0
        drained = False
        while time.monotonic() < deadline:
            drained = runtime.close(timeout_sec=0.05)
            if drained:
                break
            time.sleep(0.001)
        assert drained
        assert previous.closed
        assert previous.close_calls == (2 if close_fails else 1)
        assert caller not in close_threads
        assert slot[0] is None
        assert not runtime.collection_transition_active()
        assert not any(state.get("remaining_sec", 0) > 0 for state in clutch.countdown_states)
        if kind == "prepare_selection":
            assert callbacks == []
            assert len(failures) == 1
            assert isinstance(failures[0], OSError)
            assert "original writer preparation failure" in str(failures[0])
            assert len(candidates) == 1 and candidates[0].close_calls == 1
        else:
            assert "original writer preparation failure" in args.dataset_startup_error
    finally:
        release.set()
        runtime.close(timeout_sec=1.0)


@pytest.mark.parametrize("failure_mode", ["callback", "worker_start", "blocked"])
def test_dataset_selection_shutdown_retains_retirement_until_bounded_recovery(
    tmp_path: Path, monkeypatch, failure_mode: str,
) -> None:
    instances = []
    allow_close = False
    block_close = False
    release = threading.Event()
    entered = threading.Event()
    close_threads = []
    caller_thread = threading.get_ident()

    class Recorder:
        def __init__(self, *, args, **_kwargs):
            self.args = args
            self.dataset_root = Path(args.dataset_root)
            self.finalizing = self.is_recording = self.is_paused = False
            self.frame_count = 0
            self.writer_started = self.closed = False
            instances.append(self)

        def ensure_started(self):
            self.writer_started = True

        def poll_finalize_tasks(self):
            return []

        def close(self):
            self.closed = True

    old = SimpleNamespace(
        finalizing=False, is_recording=False, is_paused=False, frame_count=0,
        dataset_root=tmp_path / "old", writer_started=True,
        control_trace_episode_sink=None, v25p_sidecar_store=None,
        closed=False, poll_finalize_tasks=list,
    )

    def close_old():
        close_threads.append(threading.get_ident())
        entered.set()
        if block_close:
            assert release.wait(timeout=2.0)
        if not allow_close:
            raise RuntimeError("retirement remains unavailable")
        old.closed = True

    old.close = close_old
    slot = [old]
    runtime, args, logs = _episode_runtime(
        tmp_path=old.dataset_root, recorder_type=Recorder, dataset_slot=slot,
        clutch=_Clutch(), transition_calls=[],
    )
    callbacks = []
    selected = dict(vars(args), dataset_root=tmp_path / "selected")

    def commit(snapshot):
        vars(args).clear()
        vars(args).update(snapshot)
        callbacks.append("commit")

    original_start = control_plane_module.threading.Thread.start
    try:
        assert runtime.schedule_dataset_selection(
            selected, bind_dataset_contract=False, on_commit=commit,
            on_complete=lambda: callbacks.append("complete"),
            on_failure=lambda error: pytest.fail(str(error)),
        ) == "pending"
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            runtime.poll_dataset_finalize(time.time())
            if any("old recorder remains owned" in message and
                   "retirement remains unavailable" in message for _, message in logs):
                break
            time.sleep(0.001)
        assert callbacks == ["commit"]
        assert not old.closed and runtime.collection_transition_active()
        assert slot[0] is instances[0]

        if failure_mode == "worker_start":
            def fail_start(thread):
                if "retire_selection_previous" in thread.name:
                    raise RuntimeError("retirement worker cannot start")
                original_start(thread)
            monkeypatch.setattr(control_plane_module.threading.Thread, "start", fail_start)
        elif failure_mode == "blocked":
            block_close = allow_close = True
            entered.clear()

        started = time.monotonic()
        assert runtime.close(timeout_sec=0.02) is False
        assert time.monotonic() - started < 0.25
        assert runtime.collection_transition_active()
        assert slot[0] is instances[0] and not instances[0].closed
        assert caller_thread not in close_threads
        if failure_mode == "blocked":
            assert entered.is_set()
        monkeypatch.setattr(control_plane_module.threading.Thread, "start", original_start)
        allow_close = True
        release.set()
        deadline = time.monotonic() + 1.0
        drained = False
        while time.monotonic() < deadline:
            drained = runtime.close(timeout_sec=0.05)
            if drained:
                break
            time.sleep(0.001)
        assert drained and old.closed
        assert not runtime.collection_transition_active()
        assert callbacks == ["commit", "complete"]
        assert caller_thread not in close_threads
    finally:
        monkeypatch.setattr(control_plane_module.threading.Thread, "start", original_start)
        allow_close = True
        release.set()
        runtime.close(timeout_sec=1.0)


@pytest.mark.parametrize("cut", ["completed", "publication_gap"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_open_rejection_shutdown_retains_writer_across_result_publication(
    tmp_path: Path, monkeypatch, cut: str, close_fails: bool,
) -> None:
    entered, release = threading.Event(), threading.Event()
    clock_calls = 0
    close_threads = []
    caller = threading.get_ident()

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        if clock_calls == 2:
            entered.set()
            assert release.wait(timeout=2.0)
        return time.monotonic()

    plane = CollectionControlPlane(name="open-publication-shutdown", clock=clock)
    monkeypatch.setattr(episode_recording_module, "CollectionControlPlane", lambda **_kwargs: plane)
    recorder = SimpleNamespace(
        dataset_root=tmp_path, writer=SimpleNamespace(next_episode_index=4),
        writer_started=True, finalizing=False, is_recording=False,
        is_paused=False, frame_count=0, active_episode_index=None,
        camera_sources={}, closed=False, close_calls=0, poll_finalize_tasks=list,
    )

    def reject_configuration():
        raise OSError("original pre-mutation profile rejection")

    def close_recorder():
        close_threads.append(threading.get_ident())
        recorder.close_calls += 1
        if close_fails and recorder.close_calls == 1:
            raise RuntimeError("writer close remains pending")
        recorder.closed = True

    recorder.require_configured_for_current_args = reject_configuration
    recorder.close = close_recorder
    slot = [recorder]
    runtime, _args, logs = _episode_runtime(
        tmp_path=tmp_path, recorder_type=lambda **_kwargs: None,
        dataset_slot=slot, clutch=_Clutch(), transition_calls=[],
    )
    try:
        assert runtime.start_recording(time.time(), start_reason="shutdown-test")
        assert entered.wait(timeout=1.0) and slot[0] is None
        if cut == "completed":
            release.set()
            deadline = time.monotonic() + 1.0
            while plane.busy and time.monotonic() < deadline:
                time.sleep(0.001)
            assert not plane.busy
        else:
            started = time.monotonic()
            assert runtime.close(timeout_sec=0.02) is False
            assert time.monotonic() - started < 0.25
            assert runtime.collection_transition_active()
            assert recorder.close_calls == 0
            release.set()
        deadline = time.monotonic() + 1.0
        drained = False
        while time.monotonic() < deadline:
            drained = runtime.close(timeout_sec=0.05)
            if drained:
                break
            time.sleep(0.001)
        assert drained
        assert recorder.closed
        assert recorder.close_calls == (2 if close_fails else 1)
        assert caller not in close_threads
        assert slot[0] is None
        assert not runtime.collection_transition_active()
        assert sum(tag == "Dataset" and "original pre-mutation profile rejection" in message
                   for tag, message in logs) == 1
    finally:
        release.set()
        runtime.close(timeout_sec=1.0)


@pytest.mark.parametrize("phase", ["open", "stop", "finalize"])
def test_bounded_shutdown_retries_publish_normal_episode_seal(
    tmp_path: Path, monkeypatch, phase: str,
) -> None:
    entered, release = threading.Event(), threading.Event()
    clock_calls = 0
    task = SimpleNamespace(
        episode_index=4, frame_count=12, queued_at=10.0, started_at=10.1,
        completed_at=10.2, stage_timings_ms={}, success=True, error="",
        v25p_error="", v25p_native_error="", progress_error="",
    )
    calls = []
    results = []

    def clock():
        nonlocal clock_calls
        clock_calls += 1
        if (phase == "open" and clock_calls == 2) or (phase == "stop" and clock_calls == 4):
            entered.set()
            assert release.wait(timeout=2.0)
        return time.monotonic()

    plane = CollectionControlPlane(name="normal-seal-retry", clock=clock)
    monkeypatch.setattr(episode_recording_module, "CollectionControlPlane", lambda **_kwargs: plane)
    recorder = SimpleNamespace(
        dataset_root=tmp_path, writer=SimpleNamespace(next_episode_index=4),
        writer_started=True, finalizing=False, is_recording=False,
        is_paused=False, frame_count=12, active_episode_index=None,
        camera_sources={}, require_configured_for_current_args=lambda: None,
        collection_frequency_status=lambda: {"sample_hz": 15.0},
        close=lambda: calls.append("close"),
    )

    def start_episode(_metadata, *, now):
        calls.append("start")
        recorder.is_recording = True
        recorder.active_episode_index = 4
        return 4

    def stop_episode(**_kwargs):
        calls.append("stop")
        recorder.is_recording = False
        recorder.finalizing = True
        results.append(task)
        return task

    def poll_finalize_tasks():
        if phase == "finalize" and not release.is_set():
            entered.set()
            return []
        if not results:
            return []
        recorder.finalizing = False
        return [results.pop(0)]

    recorder.start_episode = start_episode
    recorder.stop_episode = stop_episode
    recorder.poll_finalize_tasks = poll_finalize_tasks
    slot = [recorder]
    clutch = _Clutch()
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path, recorder_type=lambda **_kwargs: None,
        dataset_slot=slot, clutch=clutch, transition_calls=[],
    )
    try:
        assert runtime.start_recording(time.time(), start_reason="shutdown-test")
        if phase == "open":
            assert entered.wait(timeout=1.0)
        started = time.monotonic()
        assert runtime.close(timeout_sec=0.03) is False
        assert time.monotonic() - started < 0.25
        assert entered.is_set()
        assert runtime.collection_transition_active()
        assert not runtime.start_recording(time.time(), start_reason="late")
        assert not runtime.schedule_recording_countdown(time.time(), start_reason="late")
        with pytest.raises(RuntimeError, match="shutdown"):
            runtime.schedule_dataset_selection(
                dict(vars(args)), bind_dataset_contract=False,
                on_commit=lambda _snapshot: None, on_complete=lambda: None,
                on_failure=lambda _error: None,
            )
        assert clutch.finalize_states == []
        assert "close" not in calls
        release.set()
        assert runtime.close(timeout_sec=1.0)
        assert not runtime.collection_transition_active()
        assert slot[0] is recorder
        assert calls == ["start", "stop"]
        assert len(clutch.finalize_states) == 1
        assert clutch.finalize_states[0]["status"] == "saved"
        assert clutch.round_trip_advances == 1
        assert clutch.plus_episode_indices == [4, 4]
    finally:
        release.set()
        runtime.close(timeout_sec=1.0)

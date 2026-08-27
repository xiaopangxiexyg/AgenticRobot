from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop import collection_control_plane as control_plane_module
from teleop_runtime.real_teleop import dataset_recorder as dataset_recorder_module
from teleop_runtime.real_teleop.collection_control_plane import (
    CollectionControlOperationBusy,
    CollectionControlPlane,
)
from teleop_runtime.real_teleop.dataset_recorder import RealTeleopDatasetRecorder
from teleop_runtime.real_teleop.dataset_finalize import RealDatasetFinalizeTask
from teleop_runtime.real_teleop.episode_recording_runtime import (
    build_episode_recording_runtime,
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
    release.set()
    result = _wait_for_result(plane, generation)
    assert result.succeeded
    assert result.value == "ready"
    assert result.kind == "prepare"
    assert result.elapsed_sec >= 0.0


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
    plane = CollectionControlPlane(name="unit-test")

    def prepare() -> object:
        entered.set()
        assert release.wait(timeout=1.0)
        return candidate

    generation = plane.submit(
        "prepare",
        prepare,
        discard_value=cleaned.append,
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
    assert not plane.busy


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


class _CaptureModalityContract:
    def __init__(
        self,
        *,
        ignored_camera_keys: tuple[str, ...],
        ignored_tactile_keys: tuple[str, ...],
    ) -> None:
        self.ignored_camera_keys = ignored_camera_keys
        self.ignored_tactile_keys = ignored_tactile_keys


class _Clutch:
    def __init__(self) -> None:
        self.countdown_states: list[dict[str, object]] = []
        self.recording_states: list[dict[str, object]] = []
        self.finalize_states: list[dict[str, object]] = []
        self.restore_calls = 0
        self.round_trip_advances = 0
        self.hold_reasons: list[str] = []
        self.plus_episode_indices: list[int] = []

    def episode_start_admission(self) -> tuple[bool, str]:
        return True, ""

    def set_episode_start_countdown_state(self, **state: object) -> None:
        self.countdown_states.append(dict(state))

    def set_episode_recording_state(self, **state: object) -> None:
        self.recording_states.append(dict(state))

    def set_episode_finalize_state(self, **state: object) -> None:
        self.finalize_states.append(dict(state))

    def _restore_gripper_task_policy_for_collection(self) -> None:
        self.restore_calls += 1

    def advance_round_trip_task_after_saved_episode(self) -> None:
        self.round_trip_advances += 1

    def request_hold(self, *, reason: str) -> None:
        self.hold_reasons.append(reason)

    def begin_plus_curation_episode(self, episode_index: int) -> None:
        self.plus_episode_indices.append(int(episode_index))

    def set_plus_curation_pending(
        self,
        *,
        dataset_root: Path,
        episode_index: int,
    ) -> None:
        del dataset_root
        self.plus_episode_indices.append(int(episode_index))

    def _current_active_controls(self) -> list[str]:
        return []


class _FailingUi:
    def publish(self, **_state: object) -> None:
        raise RuntimeError("ui publish unavailable")

    def append_message(self, _title: str, _message: str) -> None:
        raise RuntimeError("ui message unavailable")


def _episode_runtime(
    *,
    tmp_path: Path,
    recorder_type: type,
    dataset_slot: list[object | None],
    clutch: _Clutch,
    transition_calls: list[str],
    control_ui: object | None = None,
    debug_episode_enabled: bool = False,
    start_episode_recording=lambda *_args, **_kwargs: None,
    stop_episode_recording=lambda *_args, **_kwargs: None,
    episode_recorder_override: object | None = None,
    publish_runtime_hardware_status=lambda: None,
    bind_capture_modality_contract=lambda *_args, **_kwargs: None,
    position_episode_debug_runtime_override: object | None = None,
    update_control_debug_runtime_metadata=lambda **_kwargs: None,
):
    logs: list[tuple[str, str]] = []
    episode_state = [None]
    episode_recorder = (
        SimpleNamespace(
            config=SimpleNamespace(enabled=bool(debug_episode_enabled)),
            is_recording=False,
            frame_count=0,
            episode_dir=None,
            record_event=lambda *_args, **_kwargs: None,
        )
        if episode_recorder_override is None
        else episode_recorder_override
    )
    support = {
        "CaptureModalityContract": _CaptureModalityContract,
        "DEFAULT_EPISODE_START_COUNTDOWN_SEC": 3.0,
        "RealTeleopDatasetRecorder": recorder_type,
        "_arm_control_mode_arg": lambda _args: "position",
        "_build_episode_metadata": lambda *_args, **_kwargs: {},
        "_dataset_recording_debug_text": lambda **_kwargs: "",
        "_episode_root_dir": lambda *_args: tmp_path,
        "_episode_start_countdown_ready_to_record": (
            lambda *, now, deadline, warmup_ready: warmup_ready and now >= deadline
        ),
        "_episode_start_countdown_remaining": (
            lambda now, deadline: max(0.0, deadline - now)
        ),
        "_log": lambda tag, message, **_kwargs: logs.append((tag, message)),
        "_start_episode_recording": start_episode_recording,
        "_stop_episode_recording": stop_episode_recording,
        "bind_capture_modality_contract": bind_capture_modality_contract,
    }
    args = SimpleNamespace(
        no_collection=False,
        dataset_root=tmp_path,
        dataset_base_root=tmp_path,
        dataset_dir_name=tmp_path.name,
        dataset_name="unit-dataset",
        collection_task_id="unit-task",
        collection_task_name="Unit task",
        collection_task_name_zh="单元任务",
        collection_task_description="Unit test task",
        ignore_camera=(),
        ignore_tactile=(),
        no_camera=True,
        camera_config=tmp_path / "camera.json",
        write_video=False,
        tactile_storage="inline",
        storage_queue_size=4,
        storage_queue_backpressure_policy="block",
        collection_warning_hz=15.0,
        dataset_startup_error="",
        arm_control_mode="position",
        control_debug_config=SimpleNamespace(enabled=False),
    )
    runtime = build_episode_recording_runtime(
        args=args,
        components=SimpleNamespace(),
        clutch_input=clutch,
        episode_recorder=episode_recorder,
        camera_sources={},
        control_ui=control_ui,
        startup_device_errors={},
        fix_hold_coordinator=SimpleNamespace(active=False),
        position_episode_debug_runtime=(
            SimpleNamespace(
                start=lambda *_args, **_kwargs: False,
                stop=lambda *_args, **_kwargs: None,
                start_close=lambda _action: None,
                current_recorder=lambda: None,
            )
            if position_episode_debug_runtime_override is None
            else position_episode_debug_runtime_override
        ),
        update_control_debug_runtime_metadata=update_control_debug_runtime_metadata,
        publish_runtime_hardware_status=publish_runtime_hardware_status,
        collection_startup_blockers=lambda: {},
        recording_warmup_status=lambda *_args, **_kwargs: (True, "ready"),
        reset_gripper_warmup_probe=lambda _reason: None,
        prepare_dataset_transition=transition_calls.append,
        dataset_recorder_provider=lambda: dataset_slot[0],
        set_dataset_recorder=lambda value: dataset_slot.__setitem__(0, value),
        camera_previewer_provider=lambda: None,
        episode_last_state_provider=lambda: episode_state[0],
        set_episode_last_state=lambda value: episode_state.__setitem__(0, value),
        step_index_provider=lambda: 0,
        support=support,
    )
    return runtime, args, logs


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


def test_pending_episode_open_is_cancellable_and_never_becomes_sample_visible(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    stop_reasons: list[str] = []

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
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert not runtime.collection_transition_active()
    assert slot[0] is None
    assert recorder.closed
    assert stop_reasons == ["episode_open_cancelled"]
    assert not runtime.recording_active()
    assert clutch.plus_episode_indices == []
    assert clutch.recording_states[-1]["event_status"] == "idle"
    assert "未向控制循环发布可采样 Episode" in str(
        clutch.recording_states[-1]["event_message"]
    )
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
            raise RuntimeError("profile changed")

        def poll_finalize_tasks(self) -> list[object]:
            return []

        def close(self) -> None:
            self.closed = True

    recorder = Recorder()
    slot: list[object | None] = [recorder]
    clutch = _Clutch()
    runtime, _args, _logs = _episode_runtime(
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


def test_episode_open_rpc_failure_isolates_attempted_writer_for_fresh_retry(
    tmp_path: Path,
) -> None:
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

    def stop_episode(**_kwargs: object) -> None:
        nonlocal stop_calls
        stop_calls += 1
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
    assert runtime.stop_recording(time.time(), end_reason="manual_key_stop_retry")
    assert slot[0] is None
    deadline = time.monotonic() + 1.0
    while runtime.collection_transition_active() and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is recorder
    assert recorder.is_recording is False
    assert stop_calls == 1
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


def test_cancelled_prepare_never_publishes_candidate(tmp_path: Path) -> None:
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

        def close(self) -> None:
            self.closed = True

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
    assert entered.wait(timeout=1.0)
    assert runtime.cancel_recording_countdown(time.time(), reason="unit_cancel")
    assert runtime.countdown_deadline() == float("inf")
    release.set()
    deadline = time.monotonic() + 1.0
    while runtime.countdown_deadline() is not None and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert slot[0] is None
    assert instances[0].closed is True
    assert clutch.recording_states[-1]["event_status"] == "idle"
    runtime.close()


def test_episode_stop_drain_is_nonblocking_and_committed_on_poll(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()
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

    def stop_episode(**_kwargs: object):
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
    )

    started_at = time.perf_counter()
    assert runtime.stop_recording(
        time.time(),
        end_reason="unit",
        summary={"reason": "test"},
    )
    assert time.perf_counter() - started_at < 0.05
    assert entered.wait(timeout=1.0)
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


def test_failed_stop_isolated_recorder_and_allows_fresh_retry(tmp_path: Path) -> None:
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
    recorder.close = lambda: setattr(recorder, "closed", True)

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
    deadline = time.monotonic() + 1.0
    while clutch.recording_states[-1]["event_status"] == "stopping" and time.monotonic() < deadline:
        runtime.poll_dataset_finalize(time.time())
        time.sleep(0.001)

    assert recorder.closed is True
    assert slot[0] is None
    assert clutch.recording_states[-1]["event_status"] == "error"
    assert clutch.hold_reasons == ["dataset_stop_failure"]
    assert args.dataset_startup_error == "RuntimeError: sample drain failed"
    runtime.close()

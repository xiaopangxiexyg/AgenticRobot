from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from teleop_runtime.real_teleop.collection_control_plane import (
    CollectionControlOperationBusy,
    CollectionControlPlane,
)
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

    plane.close()

    assert cleaned == [candidate]
    assert plane.poll(generation) is None
    with pytest.raises(RuntimeError, match="closed"):
        plane.submit("prepare", lambda: None)


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

    def _restore_gripper_task_policy_for_collection(self) -> None:
        self.restore_calls += 1

    def advance_round_trip_task_after_saved_episode(self) -> None:
        self.round_trip_advances += 1

    def request_hold(self, *, reason: str) -> None:
        self.hold_reasons.append(reason)

    def begin_plus_curation_episode(self, episode_index: int) -> None:
        self.plus_episode_indices.append(int(episode_index))


def _episode_runtime(
    *,
    tmp_path: Path,
    recorder_type: type,
    dataset_slot: list[object | None],
    clutch: _Clutch,
    transition_calls: list[str],
    publish_runtime_hardware_status=lambda: None,
):
    logs: list[tuple[str, str]] = []
    episode_state = [None]
    episode_recorder = SimpleNamespace(
        config=SimpleNamespace(enabled=False),
        is_recording=False,
        frame_count=0,
        episode_dir=None,
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
        "_start_episode_recording": lambda *_args, **_kwargs: None,
        "_stop_episode_recording": lambda *_args, **_kwargs: None,
        "bind_capture_modality_contract": lambda *_args, **_kwargs: None,
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
        control_ui=None,
        startup_device_errors={},
        fix_hold_coordinator=SimpleNamespace(),
        position_episode_debug_runtime=SimpleNamespace(
            start=lambda *_args, **_kwargs: False,
            stop=lambda *_args, **_kwargs: None,
            start_close=lambda _action: None,
            current_recorder=lambda: None,
        ),
        update_control_debug_runtime_metadata=lambda **_kwargs: None,
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
    runtime, _args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
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
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
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


def test_dataset_selection_is_single_flight_and_no_recorder_commits_inline(
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
            instances.append(self)

        def ensure_started(self) -> None:
            entered.set()
            assert release.wait(timeout=1.0)

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
    empty_runtime, empty_args, _logs = _episode_runtime(
        tmp_path=tmp_path / "empty",
        recorder_type=Recorder,
        dataset_slot=empty_slot,
        clutch=_Clutch(),
        transition_calls=[],
    )
    inline_snapshot = dict(vars(empty_args))
    inline_snapshot["dataset_root"] = tmp_path / "inline"
    inline_callbacks: list[str] = []

    def inline_commit(snapshot: dict[str, object]) -> None:
        vars(empty_args).clear()
        vars(empty_args).update(snapshot)
        inline_callbacks.append("commit")

    assert empty_runtime.schedule_dataset_selection(
        inline_snapshot,
        bind_dataset_contract=False,
        on_commit=inline_commit,
        on_complete=lambda: inline_callbacks.append("complete"),
        on_failure=lambda exc: inline_callbacks.append(f"failure:{exc}"),
    ) == "committed"
    assert inline_callbacks == ["commit", "complete"]
    assert empty_args.dataset_root == tmp_path / "inline"
    assert empty_slot[0] is None
    empty_runtime.close()


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
    finalize_task = SimpleNamespace(episode_index=4, frame_count=17)

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
        poll_finalize_tasks=lambda: [],
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
    assert clutch.round_trip_advances == 1
    runtime.close()


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

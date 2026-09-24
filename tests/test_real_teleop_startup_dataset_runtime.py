"""Writer preparation contracts at the current asynchronous recording boundary."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.support.real_teleop_recording import (
    _Clutch,
    _episode_runtime,
    poll_recording_transition,
)


class _Recorder:
    def __init__(self, *, args: object, components: object, camera_sources: object) -> None:
        self.args = args
        self.components = components
        self.camera_sources = camera_sources
        self.dataset_root = Path(args.dataset_root)
        self.is_recording = False
        self.finalizing = False
        self.frame_count = 0
        self.writer_started = False
        self.closed = False

    def ensure_started(self) -> None:
        self.writer_started = True

    def poll_finalize_tasks(self) -> list[object]:
        return []

    def close(self) -> None:
        self.closed = True


def test_dataset_writer_construction_is_lazy_and_ready_before_publication(
    tmp_path: Path,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    instances: list[_Recorder] = []

    class Recorder(_Recorder):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            instances.append(self)

        def ensure_started(self) -> None:
            entered.set()
            assert release.wait(timeout=1.0)
            super().ensure_started()

    slot: list[object | None] = [None]
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=[],
    )
    try:
        assert instances == []
        assert runtime.schedule_recording_countdown(10.0, start_reason="unit")
        assert entered.wait(timeout=1.0)
        recorder = instances[0]
        assert recorder.args is args
        assert recorder.camera_sources == {}
        assert not recorder.writer_started
        assert slot[0] is None
        assert runtime.countdown_deadline() == float("inf")

        release.set()
        poll_recording_transition(runtime)
        assert slot[0] is recorder
        assert recorder.writer_started
        assert not recorder.is_recording
        assert not recorder.closed
    finally:
        release.set()
        assert runtime.close()
    # Publication transfers the idle writer back to the outer runtime.
    assert slot[0] is recorder
    assert not recorder.closed
    recorder.close()
    assert recorder.closed


@pytest.mark.parametrize("debug_enabled", [False, True])
def test_disabled_formal_collection_never_constructs_writer(
    tmp_path: Path,
    debug_enabled: bool,
) -> None:
    class UnexpectedRecorder:
        def __init__(self, **_kwargs: object) -> None:
            pytest.fail("--no-collection must not construct a formal writer")

    slot: list[object | None] = [None]
    transitions: list[str] = []
    runtime, args, _logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=UnexpectedRecorder,
        dataset_slot=slot,
        clutch=_Clutch(),
        transition_calls=transitions,
        debug_episode_enabled=debug_enabled,
    )
    args.no_collection = True
    try:
        assert runtime.schedule_recording_countdown(
            10.0, start_reason="unit"
        ) is debug_enabled
        assert slot == [None]
        assert transitions == []
        assert not runtime.collection_transition_active()
        assert args.dataset_startup_error == ""
        assert (runtime.countdown_deadline() is not None) is debug_enabled
    finally:
        assert runtime.close()


@pytest.mark.parametrize("failure_stage", ["construction", "ready"])
@pytest.mark.parametrize("failure_type", [ValueError, OSError])
def test_failed_writer_preparation_isolated_and_retry_uses_fresh_candidate(
    tmp_path: Path,
    failure_stage: str,
    failure_type: type[Exception],
) -> None:
    should_fail = True
    instances: list[_Recorder] = []
    publications: list[tuple[str, tuple[bool, ...]]] = []
    messages: list[tuple[str, str]] = []

    class Recorder(_Recorder):
        def __init__(self, **kwargs: object) -> None:
            if should_fail and failure_stage == "construction":
                raise failure_type("invalid writer")
            super().__init__(**kwargs)
            instances.append(self)

        def ensure_started(self) -> None:
            if should_fail:
                raise failure_type("invalid writer")
            super().ensure_started()

    slot: list[object | None] = [None]
    clutch = _Clutch()
    runtime, args, logs = _episode_runtime(
        tmp_path=tmp_path,
        recorder_type=Recorder,
        dataset_slot=slot,
        clutch=clutch,
        transition_calls=[],
        control_ui=SimpleNamespace(
            append_message=lambda tag, message: messages.append((tag, message)),
        ),
        publish_runtime_hardware_status=lambda: publications.append(
            (args.dataset_startup_error, tuple(item.closed for item in instances))
        ),
    )
    detail = f"{failure_type.__name__}: invalid writer"
    try:
        assert runtime.schedule_recording_countdown(10.0, start_reason="unit")
        poll_recording_transition(runtime)
        assert slot[0] is None
        assert runtime.countdown_deadline() is None
        assert not runtime.recording_active()
        assert args.dataset_startup_error == detail
        expected_closed = () if failure_stage == "construction" else (True,)
        assert publications == [(detail, expected_closed)]
        assert any(detail in message for _tag, message in messages)
        assert any(detail in message for _tag, message in logs)
        assert clutch.recording_states[-1]["event_status"] == "error"
        assert clutch.recording_states[-1]["active"] is False
        assert clutch.hold_reasons == []

        should_fail = False
        failed_candidates = tuple(instances)
        assert runtime.schedule_recording_countdown(11.0, start_reason="retry")
        poll_recording_transition(runtime)
        assert slot[0] is instances[-1]
        assert all(slot[0] is not previous for previous in failed_candidates)
        assert instances[-1].writer_started
        assert not instances[-1].closed
        assert args.dataset_startup_error == ""
        assert publications[-1][0] == ""
    finally:
        assert runtime.close()
        if slot[0] is not None:
            slot[0].close()

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

from teleop_runtime.collection.async_writer import AsyncWriterStats
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop


class _FakeWriterPool:
    def stats(self) -> AsyncWriterStats:
        return AsyncWriterStats(
            queued=0,
            submitted=3,
            completed=3,
            dropped=0,
            failures=0,
            worker_count=1,
            max_queue_size=8,
            queue_high_water=1,
            submit_wait_total_sec=0.0,
            submit_wait_max_sec=0.0,
        )


class _BlockingWriterObserver:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.flush_started = threading.Event()
        self.release_flush = threading.Event()

    def flush(self) -> None:
        self.events.append("writer_flush")
        self.flush_started.set()
        self.release_flush.wait(timeout=2.0)

    def storage_stats(self) -> dict[str, object]:
        return {"enabled": True, "storage_queue_size": 2, "routes": {}}


class _FakeWriter:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.finalized: list[tuple[int, dict[str, object]]] = []

    def finalize_episode(self, episode_index: int, *, summary=None) -> None:
        self.events.append("writer_finalize_episode")
        self.finalized.append((int(episode_index), dict(summary or {})))


def _new_recorder(tmp_path: Path, events: list[str]) -> real_teleop.RealTeleopDatasetRecorder:
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.active_episode_index = 12
    recorder.active_started_at = time.time()
    recorder.sample_builder = SimpleNamespace(frame_index=7)
    recorder._record_timestamps = deque()
    recorder._episode_start_monotonic = time.monotonic()
    recorder._pause_started_at = None
    recorder._pause_started_monotonic = None
    recorder._pause_reason = None
    recorder._pause_started_frame_index = None
    recorder._pause_started_step_index = None
    recorder._pause_started_last_state = None
    recorder._pause_segments = []
    recorder._paused_accumulated_sec = 0.0
    recorder.dataset_root = tmp_path
    recorder.camera_sources = {}
    recorder.writer_pool = _FakeWriterPool()
    recorder.writer_observer = _BlockingWriterObserver(events)
    recorder.writer = _FakeWriter(events)
    recorder.finalize_tasks = []
    recorder.saved_episodes = 0
    recorder.discarded_episodes = 0
    recorder.collection_frequency_status = lambda: {"sample_hz": 15.0}
    recorder.pause_summary = lambda: {}
    return recorder


def test_deferred_finalize_starts_only_after_explicit_post_control_launch(tmp_path: Path) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)

    def close_audit() -> dict[str, object]:
        events.append("position_audit_close")
        return {"closed": True, "elapsed_ms": 1.0}

    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=42,
        last_state="RUNNING",
        defer_finalize_start=True,
        before_finalize=close_audit,
    )

    assert task is not None
    assert task.pending
    assert events == []

    started = recorder.start_pending_finalize_tasks()

    assert started == [task]
    assert recorder.writer_observer.flush_started.wait(timeout=1.0)
    assert task.running
    assert events == ["position_audit_close", "writer_flush"]

    recorder.writer_observer.release_flush.set()
    recorder.wait_finalize_tasks()

    assert task.done
    assert task.success
    assert task.started_at is not None
    assert task.started_at >= task.queued_at
    assert events == [
        "position_audit_close",
        "writer_flush",
        "writer_finalize_episode",
    ]
    assert set(task.stage_timings_ms) == {
        "position_audit_close",
        "writer_flush",
        "writer_finalize_episode",
    }
    episode_index, summary = recorder.writer.finalized[0]
    assert episode_index == 12
    assert summary["finalize_queued_at"] == task.queued_at
    assert summary["finalize_started_at"] == task.started_at
    assert summary["position_audit_finalize"] == {"closed": True, "elapsed_ms": 1.0}
    assert set(summary["finalize_prewrite_timing_ms"]) == {
        "position_audit_close",
        "writer_flush",
    }

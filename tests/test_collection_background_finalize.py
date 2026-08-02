from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from teleop_runtime.collection import run_collect
from collection_runtime.async_writer import AsyncWriterStats


class _FakeWriterPool:
    def stats(self) -> AsyncWriterStats:
        return AsyncWriterStats(
            queued=0,
            submitted=0,
            completed=0,
            dropped=0,
            failures=0,
            worker_count=1,
            max_queue_size=8,
            queue_high_water=0,
            submit_wait_total_sec=0.0,
            submit_wait_max_sec=0.0,
        )


class _FakeWriterObserver:
    def storage_stats(self):
        return {"enabled": True, "storage_queue_size": 2, "routes": {}}


class _FakeRuntime:
    def __init__(self):
        self.submitted_samples = 5
        self.writer_pool = _FakeWriterPool()
        self.writer_observer = _FakeWriterObserver()
        self.flush_started = threading.Event()
        self.release_flush = threading.Event()

    def flush(self) -> None:
        self.flush_started.set()
        self.release_flush.wait(timeout=2.0)


class _FakeWriter:
    def __init__(self):
        self.finalized = []

    def finalize_episode(self, episode_index: int, *, summary=None) -> None:
        self.finalized.append((episode_index, dict(summary or {})))


class CollectionBackgroundFinalizeTest(unittest.TestCase):
    def test_save_active_episode_returns_before_writer_flush_finishes(self) -> None:
        runtime = _FakeRuntime()
        writer = _FakeWriter()
        state = run_collect.CollectControlState(
            recording_active=True,
            active_episode_index=3,
            active_episode_started_at=time.time(),
        )
        app = SimpleNamespace(
            control_state=state,
            sample_builder=SimpleNamespace(frame_index=7),
            runtime=runtime,
            writer=writer,
            jitter_arm_retargeters={},
        )

        saved = run_collect._save_active_episode(
            app,
            step_count=11,
            reason="unit/save",
            show_standby=False,
        )

        self.assertTrue(saved)
        self.assertFalse(state.recording_active)
        self.assertTrue(runtime.flush_started.wait(timeout=1.0))
        self.assertEqual(writer.finalized, [])

        runtime.release_flush.set()
        run_collect._wait_finalize_tasks(app)

        self.assertEqual(len(writer.finalized), 1)
        episode_index, summary = writer.finalized[0]
        self.assertEqual(episode_index, 3)
        self.assertTrue(summary["background_finalize"])
        self.assertEqual(summary["episode_frame_count"], 7)
        self.assertEqual(state.saved_episodes, 1)


if __name__ == "__main__":
    unittest.main()

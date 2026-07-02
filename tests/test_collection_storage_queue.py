from __future__ import annotations

import threading
import unittest

from teleop_runtime.collection.async_writer import WriteJob
from teleop_runtime.collection.build_collect_components import CollectorComponents
from teleop_runtime.collection.observers import (
    CollectionWriterObserver,
    CollectionWriterObserverConfig,
)
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.schemas import TeleopObservation
from teleop_runtime.session import SessionStepResult


class _FakeWriterPool:
    def __init__(self):
        self.started = False
        self.submitted: list[WriteJob] = []
        self.video_submit_started = threading.Event()
        self.release_video_submit = threading.Event()

    def start(self) -> None:
        self.started = True

    def submit(self, job: WriteJob) -> bool:
        if job.kind == "lerobot_v21.video_frame":
            self.video_submit_started.set()
            self.release_video_submit.wait(timeout=2.0)
        self.submitted.append(job)
        return True

    def join(self) -> None:
        pass

    def close(self, *, wait=True) -> None:
        self.release_video_submit.set()


class CollectionStorageQueueTest(unittest.TestCase):
    def test_per_camera_storage_queue_drops_oldest_video_frame(self) -> None:
        writer_pool = _FakeWriterPool()
        frame_index = {"value": 0}

        def sample_builder(_result, _components):
            value = frame_index["value"]
            frame_index["value"] += 1
            return {"frame_index": value}

        def write_planner(sample, _components):
            frame = int(sample["frame_index"])
            return (
                WriteJob(kind="lerobot_v21.row", payload={"frame_index": frame}),
                WriteJob(
                    kind="lerobot_v21.video_frame",
                    payload={"frame_index": frame, "camera_key": "front"},
                    critical=False,
                    metadata={"storage_queue": "video", "camera_key": "front"},
                ),
            )

        observer = CollectionWriterObserver(
            components=CollectorComponents(session=object()),
            sample_builder=sample_builder,
            write_planner=write_planner,
            writer_pool=writer_pool,
            config=CollectionWriterObserverConfig(storage_queue_size=2),
        )
        result = SessionStepResult(
            timestamp=1.0,
            state=TeleopRunState.RUNNING,
            control_hz=30.0,
            observation=TeleopObservation(timestamp=1.0),
            command=None,
        )

        observer.handle_step(result)
        self.assertTrue(writer_pool.video_submit_started.wait(timeout=1.0))
        observer.handle_step(result)
        observer.handle_step(result)
        observer.handle_step(result)

        stats_before_flush = observer.storage_stats()["routes"]["front"]
        self.assertEqual(stats_before_flush["dropped"], 1)
        writer_pool.release_video_submit.set()
        observer.flush()

        video_frames = [
            int(job.payload["frame_index"])
            for job in writer_pool.submitted
            if job.kind == "lerobot_v21.video_frame"
        ]
        self.assertEqual(video_frames, [0, 2, 3])
        stats_after_flush = observer.storage_stats()["routes"]["front"]
        self.assertEqual(stats_after_flush["submitted"], 3)
        self.assertEqual(stats_after_flush["dropped"], 1)
        self.assertEqual(stats_after_flush["high_water"], 2)


if __name__ == "__main__":
    unittest.main()

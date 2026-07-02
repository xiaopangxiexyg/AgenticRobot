from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from teleop_runtime.collection.sample_assembler import CameraFrameMatch, CollectorSample
from teleop_runtime.collection.video_frame import VideoFrame
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
    ROW_JOB_KIND,
    VIDEO_FRAME_JOB_KIND,
)


def _camera_sample() -> CollectorSample:
    frame = VideoFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        source_timestamp=1.0,
        receive_timestamp=1.01,
        frame_index=9,
        metadata={"color_space": "rgb"},
    )
    return CollectorSample(
        index=0,
        episode_index=0,
        frame_index=0,
        timestamp=1.0,
        session_state="running",
        control_hz=30.0,
        action=np.zeros(26, dtype=np.float32),
        observation_state=np.zeros(38, dtype=np.float32),
        observation_last_action=np.zeros(26, dtype=np.float32),
        arm_state_source="unit",
        active_sides=("left",),
        camera_frames={
            "front": CameraFrameMatch(
                camera_key="front",
                frame=frame,
                selected_timestamp=1.0,
                age_sec=0.0,
            )
        },
    )


class WriterVideoStatsTest(unittest.TestCase):
    def test_finalize_summary_records_video_dropped_and_missing_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_parquet=False,
                    write_video=True,
                    allow_missing_video_frames=True,
                )
            )
            writer.start_episode(episode_index=0)
            jobs = writer.plan_sample_jobs(_camera_sample())
            self.assertTrue(any(job.kind == VIDEO_FRAME_JOB_KIND for job in jobs))
            row_job = next(job for job in jobs if job.kind == ROW_JOB_KIND)
            writer.handle_job(row_job)

            writer.finalize_episode(
                0,
                summary={
                    "storage_queue_stats": {
                        "enabled": True,
                        "storage_queue_size": 1,
                        "routes": {"front": {"dropped": 1}},
                    }
                },
            )

            episodes_path = Path(tmpdir) / "meta" / "episodes.jsonl"
            record = json.loads(episodes_path.read_text(encoding="utf-8").splitlines()[0])
            stats = record["summary"]["video_frame_stats"]["front"]
            self.assertEqual(stats["expected"], 1)
            self.assertEqual(stats["written"], 0)
            self.assertEqual(stats["dropped"], 1)
            self.assertEqual(stats["missing"], 1)


if __name__ == "__main__":
    unittest.main()

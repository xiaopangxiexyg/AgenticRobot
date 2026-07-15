from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from teleop_runtime.collection.sample_assembler import CameraFrameMatch, CollectorSample
from teleop_runtime.collection.video_frame import VideoFrame
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
    ROW_JOB_KIND,
    VIDEO_FRAME_JOB_KIND,
)


def _camera_sample(
    *,
    index: int = 0,
    episode_index: int = 0,
    frame_index: int = 0,
) -> CollectorSample:
    frame = VideoFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        source_timestamp=1.0,
        receive_timestamp=1.01,
        frame_index=9,
        metadata={"color_space": "rgb"},
    )
    return CollectorSample(
        index=int(index),
        episode_index=int(episode_index),
        frame_index=int(frame_index),
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
    def test_finalize_rewrites_dataset_global_index(self) -> None:
        pytest.importorskip("pyarrow.parquet")
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_parquet=True,
                    write_video=False,
                )
            )
            writer.start_episode(episode_index=0)
            sample0 = _camera_sample(index=100, episode_index=0, frame_index=0)
            row_job0 = next(job for job in writer.plan_sample_jobs(sample0) if job.kind == ROW_JOB_KIND)
            writer.handle_job(row_job0)
            writer.finalize_episode(0)

            writer.start_episode(episode_index=1)
            sample1 = _camera_sample(index=0, episode_index=1, frame_index=0)
            row_job1 = next(job for job in writer.plan_sample_jobs(sample1) if job.kind == ROW_JOB_KIND)
            writer.handle_job(row_job1)
            writer.finalize_episode(1)

            pq = __import__("pyarrow.parquet", fromlist=["read_table"])
            row0 = pq.read_table(Path(tmpdir) / "data/chunk-000/episode_000000.parquet").to_pylist()[0]
            row1 = pq.read_table(Path(tmpdir) / "data/chunk-000/episode_000001.parquet").to_pylist()[0]
            self.assertEqual(row0["index"], 0)
            self.assertEqual(row1["index"], 1)
            self.assertEqual(writer.next_sample_index, 2)

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
            quality = record["summary"]["trajectory_quality"]
            self.assertFalse(quality["force_collision_detected"])
            self.assertEqual(quality["force_collision_label"], "clean")

    def test_finalize_records_trajectory_force_collision_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_parquet=False,
                    write_video=False,
                )
            )
            sample = _camera_sample()
            sample.metadata.update(
                {
                    "pose_quality": {
                        "available": True,
                        "status": "warning",
                        "score": 0.7,
                        "bad": False,
                    },
                }
            )
            sample.arm_states["left"] = {
                "hardware_result_text": (
                    "RuntimeError: left: force 55.0000N > hard_stop_force_n"
                )
            }
            writer.start_episode(episode_index=0)
            row_job = next(job for job in writer.plan_sample_jobs(sample) if job.kind == ROW_JOB_KIND)
            writer.handle_job(row_job)

            writer.finalize_episode(0)

            episodes_path = Path(tmpdir) / "meta" / "episodes.jsonl"
            record = json.loads(episodes_path.read_text(encoding="utf-8").splitlines()[0])
            quality = record["summary"]["trajectory_quality"]
            self.assertTrue(quality["force_collision_detected"])
            self.assertEqual(quality["force_collision_label"], "detected_collision")
            self.assertNotIn("force_limit_exceeded_count", quality)
            self.assertNotIn("max_force_norm_n", quality)
            self.assertTrue(record["metadata"]["force_collision_detected"])
            self.assertEqual(record["metadata"]["force_collision_label"], "detected_collision")
            self.assertEqual(
                record["metadata"]["trajectory_quality"]["force_collision_label"],
                "detected_collision",
            )

            stats_path = Path(tmpdir) / "meta" / "episodes_stats.jsonl"
            stats = json.loads(stats_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertTrue(stats["force_collision_detected"])
            self.assertEqual(stats["force_collision_label"], "detected_collision")
            self.assertEqual(stats["pose_quality_warning_count"], 1)

    def test_emergency_stop_state_is_recorded_without_force_collision_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_parquet=False,
                    write_video=False,
                )
            )
            sample = _camera_sample()
            sample.arm_states["left"] = {
                "send_result_code": 0,
                "send_result_text": "emergency_stop:Arm_Stop:0",
                "emergency_stopped": True,
            }
            writer.start_episode(episode_index=0)
            row_job = next(job for job in writer.plan_sample_jobs(sample) if job.kind == ROW_JOB_KIND)
            row = row_job.payload["row"]
            self.assertEqual(row["observation.arm.left.send_result_text"], "emergency_stop:Arm_Stop:0")
            self.assertEqual(row["observation.arm.left.emergency_stopped"], 1)
            writer.handle_job(row_job)

            writer.finalize_episode(0)

            episodes_path = Path(tmpdir) / "meta" / "episodes.jsonl"
            record = json.loads(episodes_path.read_text(encoding="utf-8").splitlines()[0])
            self.assertFalse(record["summary"]["force_collision_detected"])
            self.assertEqual(record["summary"]["force_collision_label"], "clean")
            self.assertFalse(record["metadata"]["force_collision_detected"])


if __name__ == "__main__":
    unittest.main()

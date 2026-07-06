from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset_postprocess.wrist_cam_adjust.depth import (
    DepthRangeMaskConfig,
    depth_range_mask,
    unpack_binary_mask,
)
from teleop_runtime.collection.sample_assembler import CameraFrameMatch, CollectorSample
from teleop_runtime.collection.video_frame import VideoFrame
from teleop_runtime.collection.writer_v21 import (
    LeRobotV21Writer,
    LeRobotV21WriterConfig,
    ROW_JOB_KIND,
    WRIST_DEPTH_FRAME_JOB_KIND,
)


def _sample_with_depth(camera_key: str = "left_wrist") -> CollectorSample:
    depth = np.asarray(
        [
            [0, 50, 95, 120],
            [250, 400, 401, 800],
        ],
        dtype=np.uint16,
    )
    frame = VideoFrame(
        image=np.zeros((2, 4, 3), dtype=np.uint8),
        source_timestamp=1.0,
        receive_timestamp=1.01,
        frame_index=7,
        metadata={"color_space": "rgb"},
        depth_image=depth,
        depth_metadata={
            "depth_scale_m_per_unit": 0.001,
            "camera_timestamp_sec": 12.5,
            "sensor_timestamp_ms": 12500.0,
            "frame_number": 42,
        },
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
            camera_key: CameraFrameMatch(
                camera_key=camera_key,
                frame=frame,
                selected_timestamp=1.0,
                age_sec=0.0,
            )
        },
    )


class WristDepthStorageTest(unittest.TestCase):
    def test_depth_range_mask_uses_inclusive_metric_bounds(self) -> None:
        result = depth_range_mask(
            np.asarray([[0, 94, 95, 400, 401]], dtype=np.uint16),
            depth_scale_m_per_unit=0.001,
            config=DepthRangeMaskConfig(depth_min_m=0.095, depth_max_m=0.4),
        )
        np.testing.assert_array_equal(result.mask, np.asarray([[0, 0, 1, 1, 0]], dtype=np.uint8))
        self.assertAlmostEqual(result.foreground_fraction, 2.0 / 5.0)
        self.assertAlmostEqual(result.valid_fraction, 4.0 / 5.0)

    def test_writer_saves_wrist_depth_mask_and_raw_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_parquet=False,
                    write_video=False,
                    write_wrist_depth_mask=True,
                    write_wrist_depth_raw=True,
                    wrist_depth_camera_keys=("left_wrist", "right_wrist"),
                    wrist_depth_min_m=0.095,
                    wrist_depth_max_m=0.4,
                )
            )
            writer.start_episode(episode_index=0)
            jobs = writer.plan_sample_jobs(_sample_with_depth())
            self.assertTrue(any(job.kind == WRIST_DEPTH_FRAME_JOB_KIND for job in jobs))
            for job in jobs:
                writer.handle_job(job)

            row = writer._episodes[0].rows_by_frame[0]  # noqa: SLF001 - verifies pending row patching.
            self.assertEqual(row["observation.images.left_wrist.wrist_depth_mask_frame_index"], 0)
            self.assertEqual(row["observation.images.left_wrist.wrist_depth_raw_frame_index"], 0)
            self.assertAlmostEqual(
                row["observation.images.left_wrist.wrist_depth_mask_foreground_fraction"],
                4.0 / 8.0,
            )

            writer.finalize_episode(0)
            episodes_path = Path(tmpdir) / "meta" / "episodes.jsonl"
            record = json.loads(episodes_path.read_text(encoding="utf-8").splitlines()[0])
            mask_manifest_rel = record["wrist_depth_mask_paths"]["left_wrist"]
            raw_manifest_rel = record["wrist_depth_raw_paths"]["left_wrist"]
            mask_manifest = json.loads((Path(tmpdir) / mask_manifest_rel).read_text(encoding="utf-8"))
            raw_manifest = json.loads((Path(tmpdir) / raw_manifest_rel).read_text(encoding="utf-8"))

            mask_stream = Path(tmpdir) / mask_manifest["stream"]["path"]
            raw_stream = Path(tmpdir) / raw_manifest["stream"]["path"]
            unpacked = unpack_binary_mask(mask_stream.read_bytes(), height=2, width=4)
            expected_mask = np.asarray([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
            np.testing.assert_array_equal(unpacked, expected_mask)
            np.testing.assert_array_equal(
                np.frombuffer(raw_stream.read_bytes(), dtype=np.uint16).reshape(2, 4),
                _sample_with_depth().camera_frames["left_wrist"].frame.depth_image,
            )

    def test_writer_ignores_front_depth_for_wrist_streams(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = LeRobotV21Writer(
                LeRobotV21WriterConfig(
                    root_dir=tmpdir,
                    write_parquet=False,
                    write_video=False,
                    write_wrist_depth_mask=True,
                    wrist_depth_camera_keys=("left_wrist", "right_wrist"),
                )
            )
            writer.start_episode(episode_index=0)
            jobs = writer.plan_sample_jobs(_sample_with_depth(camera_key="front"))
            self.assertFalse(any(job.kind == WRIST_DEPTH_FRAME_JOB_KIND for job in jobs))
            self.assertTrue(any(job.kind == ROW_JOB_KIND for job in jobs))


if __name__ == "__main__":
    unittest.main()

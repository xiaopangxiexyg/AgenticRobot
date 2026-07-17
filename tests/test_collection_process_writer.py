from __future__ import annotations

import time

import numpy as np

from teleop_runtime.collection.process_writer import (
    ProcessCollectionConfig,
    ProcessCollectionWorker,
)
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilderConfig,
    CollectorSampleSnapshot,
)
from teleop_runtime.collection.video_frame import NearestFrameBuffer, VideoFrame
from teleop_runtime.collection.writer_v21 import LeRobotV21WriterConfig
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.session import SessionStepResult


def _snapshot() -> CollectorSampleSnapshot:
    timestamp = time.time()
    return CollectorSampleSnapshot(
        step_result=SessionStepResult(
            timestamp=timestamp,
            state=TeleopRunState.RUNNING,
            control_hz=60.0,
            command=None,
            command_dispatched=False,
        ),
        arm_states={"left": {}, "right": {}},
        hand_states={
            "left": {
                "tactile_dict": {
                    "sensor_a": np.arange(8 * 12, dtype=np.uint8).reshape(8, 12),
                    "sensor_b": np.arange(8 * 12, dtype=np.uint8).reshape(8, 12),
                },
                "tactile_timestamp": timestamp,
                "tactile_devices": ["sensor_a", "sensor_b"],
                "tactile_frame_ids": {"sensor_a": 1, "sensor_b": 1},
                "tactile_fields": ["raw"],
            },
            "right": {},
        },
        camera_frames={
            "front": VideoFrame(
                image=np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3),
                source_timestamp=timestamp,
                receive_timestamp=timestamp,
                frame_index=7,
                metadata={"color_space": "rgb"},
            )
        },
        position_reference=None,
        active_sides=("left", "right"),
        component_metadata={"test": True},
        snapshot_timings_s={
            "arm_snapshot_s": 0.0,
            "hand_snapshot_s": 0.0,
            "camera_snapshot_s": 0.0,
        },
    )


def test_process_collection_worker_writes_and_finalizes_without_hardware(tmp_path) -> None:
    # Real camera sources expose a NearestFrameBuffer containing a thread lock.
    # The process worker must detach it before ``spawn`` while the parent keeps
    # using the original config for causal frame selection.
    camera_buffer = NearestFrameBuffer()
    sample_builder_config = CollectorSampleBuilderConfig(
        active_sides=("left", "right"),
        camera_buffers={"front": camera_buffer},
        require_camera_frames=True,
    )
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=sample_builder_config,
        config=ProcessCollectionConfig(
            queue_size=2,
            shared_slot_count=2,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
    )
    assert sample_builder_config.camera_buffers == {"front": camera_buffer}
    assert worker.sample_builder_config.camera_buffers == {}
    assert worker.sample_builder_config.require_camera_frames is False
    try:
        worker.start()
        episode_index = worker.start_episode({"formal_dataset": True})
        assert worker.submit_snapshot(_snapshot()) is True
        worker.flush()
        stats = worker.stats()
        assert stats.submitted_samples == 1
        assert stats.completed_samples == 1
        assert stats.active_episode_frame_count == 1
        assert stats.worker_error is None
        assert stats.shared_payload_bytes["p50"] > 8 * 12 * 3
        assert stats.timing_ms["parent.pack_shared"]["count"] == 1
        assert stats.timing_ms["worker.handle.lerobot_v21.tactile_frame"]["count"] == 1
        worker.finalize_episode(episode_index, summary={"test": True})
    finally:
        worker.close()

    assert (tmp_path / "meta" / "info.json").is_file()
    assert (tmp_path / "meta" / "episodes.jsonl").is_file()
    assert (
        tmp_path
        / "tactile"
        / "observation.hand.left.tactile"
        / "chunk-000"
        / "episode_000000"
        / "sensor_a.uint8.bin"
    ).is_file()

from __future__ import annotations

import json
import shutil
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.stream_catalog import NativeStreamContract
from dataset_interface.streams import (
    NativeStreamExternalPayloadReference,
    load_native_stream_episode,
)
from teleop_runtime.collection.process_writer import (
    ProcessCollectionConfig,
    ProcessCollectionWorker,
    ProcessExternalNativeStreamBinding,
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


def _native_q6_contract() -> NativeStreamContract:
    return NativeStreamContract(
        stream_id="human.left_hand_q6",
        modality="human_joint_input",
        producer_id="virtual-q6",
        value_schema="float32[6]",
        shape=(6,),
        dtype="<f4",
        units="normalized_joint_position",
        frame=None,
        measured_clock_domain_id="local_monotonic",
        available_clock_domain_id="local_monotonic",
        nominal_rate_hz=30.0,
        min_expected_rate_hz=10.0,
        max_expected_rate_hz=60.0,
        access_role="actor_input",
    )


def _selected_rgb_contract() -> NativeStreamContract:
    return NativeStreamContract(
        stream_id="camera.front.selected_rgb",
        modality="selected_rgb",
        producer_id="virtual-front-camera",
        value_schema="uint8[8,12,3]",
        shape=(8, 12, 3),
        dtype="|u1",
        units="rgb_u8",
        frame=None,
        measured_clock_domain_id="camera.source",
        available_clock_domain_id="local_wall_clock",
        nominal_rate_hz=15.0,
        min_expected_rate_hz=1.0,
        max_expected_rate_hz=30.0,
        encoding="mp4/mp4v",
        compression="none",
        access_role="actor_input",
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
        # Pause/resume uses this ordered control message before accepting the
        # next continuity segment in the same episode.
        worker.reset_temporal_history()
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


def test_process_worker_native_stream_capture_is_bounded_and_episode_bound(
    tmp_path,
) -> None:
    contract = _native_q6_contract()
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-native-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=4,
            shared_slot_count=2,
            shared_slot_bytes=4096,
            worker_nice=0,
            native_inline_message_max_bytes=4096,
        ),
        native_stream_contracts={contract.stream_id: contract},
    )
    prepared = SimpleNamespace(
        episode_uuid="process-native-episode",
        legacy_episode_index=0,
    )
    capture = worker.create_native_stream_capture(prepared)
    try:
        worker.start()
        assert worker.start_episode(
            {"formal_dataset": True},
            episode_uuid=prepared.episode_uuid,
        ) == prepared.legacy_episode_index
        assert worker.submit_snapshot(_snapshot())

        accepted = capture.append(
            stream_id=contract.stream_id,
            sample_id="q6-0",
            sequence=0,
            measured_at=1.0,
            measured_clock_domain_id="local_monotonic",
            available_at=1.1,
            available_clock_domain_id="local_monotonic",
            source_payload_object_id="virtual-q6-0",
            payload=np.arange(6, dtype=np.float32),
            written_at=1.2,
            written_clock_domain_id="local_monotonic",
        )
        assert accepted.accepted
        rejected = capture.append(
            stream_id=contract.stream_id,
            sample_id="q6-too-large",
            sequence=1,
            measured_at=1.3,
            measured_clock_domain_id="local_monotonic",
            available_at=1.4,
            available_clock_domain_id="local_monotonic",
            source_payload_object_id="virtual-q6-too-large",
            payload=np.arange(6, dtype=np.float32),
            written_at=1.5,
            written_clock_domain_id="local_monotonic",
            metadata={"oversized": "x" * 8192},
        )
        assert not rejected.accepted
        assert rejected.rejection_reason == "inline_message_too_large"

        manifest = capture.seal(
            sealed_at=2.0,
            sealed_clock_domain_id="local_monotonic",
        )
        assert manifest.episode_uuid == prepared.episode_uuid
        assert manifest.legacy_episode_index == prepared.legacy_episode_index
        worker.finalize_episode(0, summary={"test": True})
        stats = worker.stats()
        assert stats.submitted_native_stream_samples == 1
        assert stats.completed_native_stream_samples == 1
        assert stats.rejected_native_stream_samples == 1
    finally:
        worker.close()

    captured = load_native_stream_episode(tmp_path, prepared.episode_uuid)
    stream_range = captured.manifest.stream_ranges[0]
    assert stream_range.sample_count == 1
    assert stream_range.dropped_count == 1
    assert stream_range.gap_reasons == ("inline_message_too_large",)


def test_process_worker_selected_rgb_external_receipt_round_trip(tmp_path) -> None:
    pytest.importorskip("cv2")
    contract = _selected_rgb_contract()
    binding = ProcessExternalNativeStreamBinding(
        stream_id=contract.stream_id,
        source_kind="selected_rgb",
        source_key="front",
    )
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-selected-rgb-test",
            write_parquet=False,
            write_video=True,
            video_codec="mp4v",
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=4,
            shared_slot_count=2,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
        native_stream_contracts={contract.stream_id: contract},
        native_external_stream_bindings={contract.stream_id: binding},
    )
    prepared = SimpleNamespace(
        episode_uuid="process-selected-rgb-episode",
        legacy_episode_index=0,
    )
    capture = worker.create_native_stream_capture(prepared)
    snapshot = _snapshot()
    try:
        worker.start()
        assert worker.start_episode(
            {"formal_dataset": True},
            episode_uuid=prepared.episode_uuid,
        ) == prepared.legacy_episode_index
        assert worker.submit_snapshot(snapshot)
        worker.flush()

        with pytest.raises(RuntimeError, match="freeze"):
            worker.finalize_episode(0, summary={"test": True})

        capture.freeze_ingestion()
        result = worker.finalize_episode(0, summary={"test": True})
        assert result.legacy_finalized
        assert result.artifact_receipt_error == ""
        assert len(result.native_stream_artifact_receipts) == 1
        receipt = result.native_stream_artifact_receipts[0]
        assert receipt.stream_id == contract.stream_id
        assert receipt.stored_payload_object_id.endswith("#video_frame=0")

        manifest = capture.seal_after_legacy_finalize(
            result.native_stream_artifact_receipts,
            sealed_at=2.0,
            sealed_clock_domain_id="local_monotonic",
        )
        assert manifest.stream_ranges[0].sample_count == 1
    finally:
        worker.close()

    captured = load_native_stream_episode(tmp_path, prepared.episode_uuid)
    records = captured.records_by_stream[contract.stream_id]
    assert len(records) == 1
    reference = captured.payload_by_stream_and_sample[
        contract.stream_id
    ][records[0].sample_id]
    assert isinstance(reference, NativeStreamExternalPayloadReference)
    assert reference.stored_payload_object_id.endswith("#video_frame=0")
    assert (tmp_path / reference.artifact_relative_path).is_file()


def test_process_worker_selected_rgb_deduplicates_reused_source_frame(
    tmp_path,
) -> None:
    pytest.importorskip("cv2")
    contract = _selected_rgb_contract()
    binding = ProcessExternalNativeStreamBinding(
        stream_id=contract.stream_id,
        source_kind="selected_rgb",
        source_key="front",
    )
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-selected-rgb-dedupe-test",
            write_parquet=False,
            write_video=True,
            video_codec="mp4v",
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=4,
            shared_slot_count=2,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
        native_stream_contracts={contract.stream_id: contract},
        native_external_stream_bindings={contract.stream_id: binding},
    )
    prepared = SimpleNamespace(
        episode_uuid="process-selected-rgb-dedupe",
        legacy_episode_index=0,
    )
    capture = worker.create_native_stream_capture(prepared)
    snapshot = _snapshot()
    try:
        worker.start()
        worker.start_episode(
            {"formal_dataset": True},
            episode_uuid=prepared.episode_uuid,
        )
        assert worker.submit_snapshot(snapshot)
        worker.flush()
        assert worker.submit_snapshot(snapshot)
        worker.flush()
        capture.freeze_ingestion()
        result = worker.finalize_episode(0, summary={"test": True})
        assert len(result.native_stream_artifact_receipts) == 1
        manifest = capture.seal_after_legacy_finalize(
            result.native_stream_artifact_receipts,
            sealed_at=2.0,
            sealed_clock_domain_id="local_monotonic",
        )
        assert manifest.stream_ranges[0].sample_count == 1
    finally:
        worker.close()


def test_process_collection_worker_reloads_zero_after_dataset_is_removed(tmp_path) -> None:
    dataset_root = tmp_path / "dataset"
    meta_root = dataset_root / "meta"
    meta_root.mkdir(parents=True)
    (meta_root / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "frame_count": 2}) + "\n"
    )
    (meta_root / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "frame_count": 2}) + "\n"
    )
    (meta_root / "tasks.jsonl").write_text("")
    sample_builder_config = CollectorSampleBuilderConfig(
        active_sides=("left", "right"),
        camera_buffers={},
        require_camera_frames=False,
    )
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=dataset_root,
            nominal_fps=15.0,
            task_id="reload-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=sample_builder_config,
        config=ProcessCollectionConfig(
            queue_size=1,
            shared_slot_count=1,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
    )
    try:
        worker.start()
        assert worker.next_episode_index == 1
        assert worker.next_sample_index == 2
        shutil.rmtree(dataset_root)
        worker.reload_finalized_dataset_state()
        assert worker.next_episode_index == 0
        assert worker.next_sample_index == 0
        assert worker.start_episode({"after_external_delete": True}) == 0
        assert worker.discard_episode(0) is True
    finally:
        worker.close()

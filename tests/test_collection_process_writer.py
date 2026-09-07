from __future__ import annotations

import json
import queue
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dataset_contracts.canonical_encoding import canonical_content_hash
from dataset_contracts.episode_identity import EpisodeCaptureIdentity
from dataset_contracts.v25p_native_storage import V25P_PARQUET_CHUNK_LAYOUT_V1
from dataset_contracts.v25p_native import V25pComponentStreamProfileV2
from dataset_contracts.v22 import PolicyIOLayout
from dataset_contracts.stream_catalog import NativeStreamContract
from dataset_interface.v25p_native import load_v25p_native_episode
from dataset_interface.streams import (
    NativeStreamExternalPayloadReference,
    load_native_stream_episode,
)
from teleop_runtime.collection.process_writer import (
    ProcessCollectionConfig,
    ProcessCollectionWorker,
    ProcessExternalNativeStreamBinding,
    ProcessV25pShadowConfig,
)
from teleop_runtime.collection.policy_io import load_policy_io_layout
from teleop_runtime.collection.sample_assembler import (
    CollectorSampleBuilderConfig,
    CollectorSampleSnapshot,
)
from teleop_runtime.collection.video_frame import NearestFrameBuffer, VideoFrame
from teleop_runtime.collection.writer_v21 import LeRobotV21WriterConfig
from teleop_runtime.diagnostics import TeleopRunState
from teleop_runtime.session import SessionStepResult
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


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


def _strict_snapshot(
    *, timestamp: float, joint_timestamp: float
) -> CollectorSampleSnapshot:
    def pose(x: float) -> np.ndarray:
        value = np.eye(4, dtype=np.float32)
        value[0, 3] = x
        return value

    command = TeleopCommand(timestamp=timestamp)
    for side, x in (("left", 0.1), ("right", 0.2)):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="cartesian_pose",
                value=pose(x),
            )
        )
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_hand",
                side=side,
                command_type="joint_vector",
                value=np.zeros(6, dtype=np.float32),
            )
        )
    return CollectorSampleSnapshot(
        step_result=SessionStepResult(
            timestamp=timestamp,
            state=TeleopRunState.RUNNING,
            control_hz=60.0,
            command=command,
            command_dispatched=False,
        ),
        arm_states={
            side: {
                "current_qpos": np.zeros(6, dtype=np.float32),
                "current_tcp_pose": pose(x),
                "runtime_arm_state_source": "causal_measured_hardware",
            }
            for side, x in (("left", 0.1), ("right", 0.2))
        },
        hand_states={
            side: {
                "gripper_measured_scalar": 0.2 if side == "left" else 0.8,
                "joint_timestamp": joint_timestamp,
            }
            for side in ("left", "right")
        },
        camera_frames={},
        position_reference=None,
        active_sides=("left", "right"),
        component_metadata={"test": "strict_v22"},
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
            sample_acceptance_receipt_capacity=4,
        ),
    )
    assert sample_builder_config.camera_buffers == {"front": camera_buffer}
    assert worker.sample_builder_config.camera_buffers == {}
    assert worker.sample_builder_config.require_camera_frames is False
    try:
        worker.start()
        episode_index = worker.start_episode({"formal_dataset": True})
        snapshot = _snapshot()
        assert worker.submit_snapshot(snapshot) is True
        worker.flush()
        stats = worker.stats()
        assert stats.submitted_samples == 1
        assert stats.completed_samples == 1
        assert stats.active_episode_frame_count == 1
        assert stats.worker_error is None
        assert stats.shared_payload_bytes["p50"] > 8 * 12 * 3
        assert stats.timing_ms["parent.pack_shared"]["count"] == 1
        assert stats.timing_ms["worker.handle.lerobot_v21.tactile_frame"]["count"] == 1
        accepted = worker.drain_sample_acceptance_receipts()
        assert len(accepted) == 1
        assert accepted[0].episode_index == episode_index
        assert accepted[0].sample_index == 0
        assert accepted[0].episode_frame_index == 0
        assert accepted[0].sample_timestamp == snapshot.step_result.timestamp
        assert worker.drain_sample_acceptance_receipts() == ()
        terminal = worker.drain_sample_terminal_events()
        assert len(terminal) == 1
        assert terminal[0].accepted is True
        assert terminal[0].receipt == accepted[0]
        assert worker.drain_sample_terminal_events() == ()
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


def test_process_collection_worker_rejects_submission_outside_lifecycle(tmp_path) -> None:
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-lifecycle-admission-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=1,
            shared_slot_count=1,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
    )
    with pytest.raises(RuntimeError, match="not started"):
        worker.submit_snapshot(_snapshot())
    worker.close(force=True)
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit_snapshot(_snapshot())


def test_process_worker_keeps_fatal_terminal_event_without_receipt_opt_in(tmp_path) -> None:
    """Fatal writer outcomes remain visible in headless/rule-based mode."""

    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-terminal-only-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=1,
            shared_slot_count=1,
            shared_slot_bytes=1024,
            worker_nice=0,
            sample_acceptance_receipt_capacity=0,
        ),
    )

    class _OneMessageQueue:
        def __init__(self, message: dict[str, object]) -> None:
            self._message = message

        def get_nowait(self) -> dict[str, object]:
            if self._message is None:
                raise queue.Empty
            message = self._message
            self._message = None
            return message

    worker._ack_queue = _OneMessageQueue(
        {
            "kind": "worker_error",
            "error": "worker.handle.test: RuntimeError: disk failure",
            "terminal_metadata": {"sample_id": "s-1"},
        }
    )
    try:
        assert worker._sample_acceptance_receipt_buffer.capacity == 0
        assert worker._sample_terminal_event_buffer.capacity > 0
        events = worker.drain_sample_terminal_events()
        assert len(events) == 1
        assert events[0].terminal_kind == "fatal"
        assert events[0].metadata == {"sample_id": "s-1"}
    finally:
        worker.close(force=True)


def test_process_collection_worker_writes_native_tactile_bundle(tmp_path) -> None:
    core_root = tmp_path / "core"
    tactile_root = tmp_path / "tactile-device"
    core_root.mkdir()
    tactile_root.mkdir()
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=core_root,
            dataset_name="process-native-tactile",
            nominal_fps=15.0,
            task_id="process-native-tactile",
            write_parquet=False,
            write_video=False,
            tactile_artifact_integrity=True,
            tactile_bundle_root=tactile_root,
            tactile_bundle_require_distinct_device=False,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=2,
            shared_slot_count=2,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="process-native-tactile-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="tests.process_writer",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    try:
        worker.start()
        episode_index = worker.start_episode(
            {"formal_dataset": True},
            episode_identity=identity,
        )
        assert worker.submit_snapshot(_snapshot()) is True
        worker.flush()
        worker.finalize_episode(episode_index, summary={"test": True})
    finally:
        worker.close()

    assert not (core_root / "tactile").exists()
    episode_record = json.loads(
        (core_root / "meta/episodes.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    dataset_uuid = episode_record["tactile_bundle"]["dataset_uuid"]
    bundle_root = tactile_root / "datasets" / dataset_uuid
    assert (bundle_root / episode_record["tactile_paths"]["left"]).is_file()
    assert (
        core_root / episode_record["tactile_bundle"]["core_commit_ref"]
    ).is_file()


def test_process_worker_rejects_stale_v22_sample_without_poisoning_worker(tmp_path) -> None:
    layout = load_policy_io_layout(
        Path(__file__).parents[1]
        / (
            "teleop_runtime/configs/policy_io_layouts/"
            "dual_arm_bimanual_dmtac_gripper_v22.json"
        )
    )
    assert layout is not None
    timestamp = time.time()
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-strict-v22-test",
            write_parquet=False,
            write_video=False,
            policy_io_layout=layout,
            strict_v22_policy_io=True,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=2,
            shared_slot_count=2,
            shared_slot_bytes=65536,
            worker_nice=0,
            sample_acceptance_receipt_capacity=4,
        ),
    )
    try:
        worker.start()
        episode_index = worker.start_episode({"formal_dataset": True})
        assert worker.submit_snapshot(
            _strict_snapshot(timestamp=timestamp, joint_timestamp=timestamp - 0.503)
        )
        worker.flush()
        stale_stats = worker.stats()
        assert stale_stats.worker_error is None
        assert stale_stats.sample_rejections == 1
        assert stale_stats.completed_samples == 0
        assert worker.drain_sample_acceptance_receipts() == ()
        stale_terminal = worker.drain_sample_terminal_events()
        assert len(stale_terminal) == 1
        assert stale_terminal[0].accepted is False
        assert stale_terminal[0].terminal_kind == "rejected"

        assert worker.submit_snapshot(
            _strict_snapshot(timestamp=timestamp + 0.1, joint_timestamp=timestamp + 0.0)
        )
        worker.flush()
        valid_stats = worker.stats()
        assert valid_stats.worker_error is None
        assert valid_stats.sample_rejections == 1
        assert valid_stats.completed_samples == 1
        assert valid_stats.active_episode_frame_count == 1
        accepted = worker.drain_sample_acceptance_receipts()
        assert len(accepted) == 1
        assert accepted[0].sample_timestamp == pytest.approx(timestamp + 0.1)
        worker.finalize_episode(episode_index, summary={"test": True})
    finally:
        worker.close()


def test_process_worker_v25p_shadow_uses_owner_identity_and_same_legacy_row(
    tmp_path,
) -> None:
    layout = PolicyIOLayout(name="process_v25p_no_end_effector")
    pose_hash = canonical_content_hash(
        {"name": "process_v25p_pose_frames"},
        domain="test.process_writer.v25p_pose_frames.v1",
    )
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-v25p-shadow-test",
            write_parquet=False,
            write_video=False,
            policy_io_layout=layout,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=2,
            shared_slot_count=2,
            shared_slot_bytes=65536,
            worker_nice=0,
        ),
        v25p_shadow_config=ProcessV25pShadowConfig(
            dataset_id="process-v25p-shadow-dataset",
            pose_frame_contract_hash=pose_hash,
            physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        ),
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="process-v25p-owner-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="test.process_writer.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    timestamp = time.time()
    sample_count = 48
    try:
        worker.start()
        episode_index = worker.start_episode(
            {"formal_dataset": True},
            episode_identity=identity,
        )
        assert episode_index == identity.legacy_episode_index
        for frame_index in range(sample_count):
            sample_timestamp = timestamp + frame_index / 15.0
            deadline = time.monotonic() + 5.0
            while not worker.submit_snapshot(
                _strict_snapshot(
                    timestamp=sample_timestamp,
                    joint_timestamp=sample_timestamp,
                )
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("process shadow queue did not accept sample")
                time.sleep(0.001)
        worker.flush()
        active_stats = worker.stats()
        shadow_timing = active_stats.timing_ms["worker.v25p_native_shadow_append"]
        assert shadow_timing["count"] == sample_count
        assert shadow_timing["p95"] is not None
        assert float(shadow_timing["p95"]) < 10.0
        result = worker.finalize_episode(episode_index, summary={"test": True})
    finally:
        worker.close()

    assert result.legacy_finalized
    assert result.v25p_native_shadow_error == ""
    assert result.v25p_native_shadow_report is not None
    assert result.v25p_native_shadow_report.receipt.status == "complete"
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.identity == identity
    assert loaded.native_capture_manifest.sample_count == sample_count
    assert isinstance(loaded.component_stream_profile, V25pComponentStreamProfileV2)
    assert {
        stream.representation_family
        for stream in loaded.component_stream_profile.streams
        if stream.dataset_role == "action"
    } == {"cartesian_pose"}
    episode_dir = tmp_path / "meta/v25p/native/episodes" / identity.episode_uuid
    assert (episode_dir / "native_storage_manifest.cjson").is_file()


@pytest.mark.slow_sim
def test_process_worker_v25p_parquet_long_episode_soak(tmp_path) -> None:
    layout = PolicyIOLayout(name="process_v25p_parquet_soak_no_end_effector")
    pose_hash = canonical_content_hash(
        {"name": "process_v25p_parquet_soak_pose_frames"},
        domain="test.process_writer.v25p_parquet_soak_pose_frames.v1",
    )
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="process-v25p-parquet-soak",
            write_parquet=False,
            write_video=False,
            policy_io_layout=layout,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=8,
            shared_slot_count=4,
            shared_slot_bytes=65536,
            worker_nice=0,
        ),
        v25p_shadow_config=ProcessV25pShadowConfig(
            dataset_id="process-v25p-parquet-soak-dataset",
            pose_frame_contract_hash=pose_hash,
            physical_storage_layout=V25P_PARQUET_CHUNK_LAYOUT_V1,
        ),
    )
    identity = EpisodeCaptureIdentity(
        episode_uuid="process-v25p-parquet-soak-episode",
        legacy_episode_index=0,
        lifecycle_owner_id="test.process_writer.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    sample_count = 1024
    timestamp = time.time()
    try:
        worker.start()
        episode_index = worker.start_episode(
            {"formal_dataset": True},
            episode_identity=identity,
        )
        for frame_index in range(sample_count):
            sample_timestamp = timestamp + frame_index / 15.0
            deadline = time.monotonic() + 5.0
            while not worker.submit_snapshot(
                _strict_snapshot(
                    timestamp=sample_timestamp,
                    joint_timestamp=sample_timestamp,
                )
            ):
                if time.monotonic() >= deadline:
                    raise TimeoutError("process Parquet soak queue did not accept sample")
                time.sleep(0.001)
        worker.flush()
        timing = worker.stats().timing_ms["worker.v25p_native_shadow_append"]
        assert timing["count"] == sample_count
        assert timing["p95"] is not None
        assert float(timing["p95"]) < 10.0
        result = worker.finalize_episode(episode_index, summary={"soak": True})
    finally:
        worker.close()

    assert result.legacy_finalized
    assert result.v25p_native_shadow_error == ""
    assert result.v25p_native_shadow_report.receipt.status == "complete"
    loaded = load_v25p_native_episode(tmp_path, identity.episode_uuid)
    assert loaded.native_capture_manifest.sample_count == sample_count
    episode_dir = tmp_path / "meta/v25p/native/episodes" / identity.episode_uuid
    assert len(tuple((episode_dir / "chunks").glob("*.cjson"))) == 8


def test_process_v25p_shadow_interruption_cannot_publish_commit_and_can_restart(
    tmp_path,
) -> None:
    layout = PolicyIOLayout(name="process_v25p_restart_no_end_effector")
    pose_hash = canonical_content_hash(
        {"name": "process_v25p_restart_pose_frames"},
        domain="test.process_writer.v25p_restart_pose_frames.v1",
    )

    def new_worker() -> ProcessCollectionWorker:
        return ProcessCollectionWorker(
            writer_config=LeRobotV21WriterConfig(
                root_dir=tmp_path,
                nominal_fps=15.0,
                task_id="process-v25p-restart-test",
                write_parquet=False,
                write_video=False,
                policy_io_layout=layout,
            ),
            sample_builder_config=CollectorSampleBuilderConfig(
                active_sides=("left", "right"),
                camera_buffers={},
                require_camera_frames=False,
            ),
            config=ProcessCollectionConfig(
                queue_size=2,
                shared_slot_count=2,
                shared_slot_bytes=65536,
                worker_nice=0,
            ),
            v25p_shadow_config=ProcessV25pShadowConfig(
                dataset_id="process-v25p-restart-dataset",
                pose_frame_contract_hash=pose_hash,
            ),
        )

    interrupted_identity = EpisodeCaptureIdentity(
        episode_uuid="process-v25p-interrupted",
        legacy_episode_index=0,
        lifecycle_owner_id="test.process_writer.owner",
        prepared_at=1.0,
        prepared_clock_domain_id="test_clock",
    )
    timestamp = time.time()
    interrupted_worker = new_worker()
    interrupted_worker.start()
    assert interrupted_worker.start_episode(
        {"formal_dataset": True},
        episode_identity=interrupted_identity,
    ) == 0
    assert interrupted_worker.submit_snapshot(
        _strict_snapshot(timestamp=timestamp, joint_timestamp=timestamp)
    )
    interrupted_worker.flush()
    interrupted_worker.close()

    interrupted_root = (
        tmp_path
        / "meta/v25p/native/episodes"
        / interrupted_identity.episode_uuid
    )
    assert (interrupted_root / "identity.cjson").is_file()
    assert not (interrupted_root / "native_capture_manifest.cjson").exists()
    assert not (interrupted_root / "episode_manifest.cjson").exists()

    recovered_identity = EpisodeCaptureIdentity(
        episode_uuid="process-v25p-recovered",
        legacy_episode_index=0,
        lifecycle_owner_id="test.process_writer.owner",
        prepared_at=2.0,
        prepared_clock_domain_id="test_clock",
    )
    recovered_worker = new_worker()
    try:
        recovered_worker.start()
        episode_index = recovered_worker.start_episode(
            {"formal_dataset": True},
            episode_identity=recovered_identity,
        )
        assert recovered_worker.submit_snapshot(
            _strict_snapshot(
                timestamp=timestamp + 1.0,
                joint_timestamp=timestamp + 1.0,
            )
        )
        recovered_worker.flush()
        result = recovered_worker.finalize_episode(
            episode_index,
            summary={"test": "restart"},
        )
    finally:
        recovered_worker.close()

    assert result.v25p_native_shadow_error == ""
    assert result.v25p_native_shadow_report.receipt.status == "complete"
    loaded = load_v25p_native_episode(tmp_path, recovered_identity.episode_uuid)
    assert loaded.identity == recovered_identity


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
        assert worker.start_episode({"after_existing_episode": True}) == 1
        assert worker.next_episode_index == 2
        assert worker.discard_episode(1) is True
        assert worker.next_episode_index == 1
        assert worker.next_sample_index == 2
        assert worker.start_episode({"after_existing_discard_retry": True}) == 1
        assert worker.discard_episode(1) is True
        shutil.rmtree(dataset_root)
        worker.reload_finalized_dataset_state()
        assert worker.next_episode_index == 0
        assert worker.next_sample_index == 0
        assert worker.start_episode({"after_external_delete": True}) == 0
        assert worker.next_episode_index == 1
        assert worker.discard_episode(0) is True
        assert worker.next_episode_index == 0
        assert worker.next_sample_index == 0
        assert worker.start_episode({"after_discard_retry": True}) == 0
        assert worker.discard_episode(0) is True
    finally:
        worker.close()


def test_process_collection_discard_clears_reused_episode_caches(tmp_path) -> None:
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="discard-cache-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=2,
            shared_slot_count=2,
            shared_slot_bytes=65536,
            worker_nice=0,
            sample_acceptance_receipt_capacity=4,
        ),
    )
    try:
        worker.start()
        assert worker.start_episode({"attempt": 1}) == 0
        assert worker.submit_snapshot(_snapshot()) is True
        worker.flush()
        assert worker.tactile_frame_counts(0) == {"left": 1}

        assert worker.discard_episode(0) is True
        assert worker.next_episode_index == 0
        assert worker.next_sample_index == 0
        assert worker.active_episode_frame_count == 0
        assert worker.tactile_frame_counts(0) == {}
        assert worker.drain_sample_acceptance_receipts() == ()

        assert worker.start_episode({"attempt": 2}) == 0
        assert worker.tactile_frame_counts(0) == {}
        assert worker.discard_episode(0) is True
    finally:
        worker.close()


def test_process_collection_rejects_duplicate_or_mismatched_legacy_lifecycle(
    tmp_path,
) -> None:
    worker = ProcessCollectionWorker(
        writer_config=LeRobotV21WriterConfig(
            root_dir=tmp_path,
            nominal_fps=15.0,
            task_id="lifecycle-guard-test",
            write_parquet=False,
            write_video=False,
        ),
        sample_builder_config=CollectorSampleBuilderConfig(
            active_sides=("left", "right"),
            camera_buffers={},
            require_camera_frames=False,
        ),
        config=ProcessCollectionConfig(
            queue_size=2,
            shared_slot_count=1,
            shared_slot_bytes=4096,
            worker_nice=0,
        ),
    )
    try:
        worker.start()
        assert worker.start_episode({"attempt": 1}) == 0
        with pytest.raises(RuntimeError, match="still active"):
            worker.start_episode({"attempt": 2})
        with pytest.raises(ValueError, match="does not match"):
            worker.finalize_episode(99, summary={})
        with pytest.raises(ValueError, match="does not match"):
            worker.discard_episode(99)
        assert worker.discard_episode(0) is True
        assert worker.discard_episode(0) is False
        with pytest.raises(RuntimeError, match="No legacy episode"):
            worker.finalize_episode(0, summary={})
        worker.reload_finalized_dataset_state()
    finally:
        worker.close()

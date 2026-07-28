from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset_contracts.capture_modalities import CaptureModalityContract
from teleop_runtime.collection.async_writer import AsyncWriterStats
from teleop_runtime.collection.schema_v22 import StrictV22SampleRejection
from teleop_runtime.control_trace import (
    ControlTraceRuntime,
    EpisodeTraceRecorderSink,
    PendingEpisodeTraceRelease,
    RuntimeTraceSession,
)
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
        self.next_episode_index = 12
        self.started: list[dict[str, object]] = []
        self.discarded: list[int] = []

    def start_episode(self, *, metadata=None) -> int:
        self.events.append("writer_start_episode")
        self.started.append(dict(metadata or {}))
        return int(self.next_episode_index)

    def discard_episode(self, episode_index: int) -> bool:
        self.events.append("writer_discard_episode")
        self.discarded.append(int(episode_index))
        return True

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


def test_recorder_configuration_guard_detects_post_warmup_task_change(tmp_path: Path) -> None:
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.args = SimpleNamespace(
        dataset_root=str(tmp_path / "dataset"),
        ignore_camera=["right_wrist"],
        ignore_tactile=["left"],
        collection_task_id="task-a",
        collection_task_description="Task A",
        dataset_name="dataset-a",
    )
    recorder.dataset_root = tmp_path / "dataset"
    recorder.capture_modality_contract = CaptureModalityContract(
        ignored_camera_keys=("right_wrist",),
        ignored_tactile_keys=("left",),
    )
    recorder.writer_storage_config = SimpleNamespace(
        task_id="task-a",
        task_description="Task A",
        dataset_name="dataset-a",
    )

    assert recorder.configured_for_current_args() is True
    recorder.require_configured_for_current_args()

    recorder.args.collection_task_id = "task-b"
    assert recorder.configured_for_current_args() is False
    with pytest.raises(RuntimeError, match="changed after warmup"):
        recorder.require_configured_for_current_args()


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
    assert summary["collection_integrity_valid"] is True
    assert summary["collection_integrity_reasons"] == []
    assert set(summary["finalize_prewrite_timing_ms"]) == {
        "position_audit_close",
        "writer_flush",
    }


def test_stop_episode_carries_terminal_collision_summary_to_formal_finalize(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)

    task = recorder.stop_episode(
        reason="force_collision",
        step_index=43,
        last_state="RUNNING",
        summary={
            "force_collision_detected": True,
            "force_collision_event": {
                "detected": True,
                "source": "controller_hard_stop",
            },
        },
        defer_finalize_start=True,
    )

    assert task is not None
    assert task.pending
    assert task.summary["stop_reason"] == "force_collision"
    assert task.summary["force_collision_detected"] is True
    assert task.summary["force_collision_event"] == {
        "detected": True,
        "source": "controller_hard_stop",
    }


def test_finalize_summary_persists_collection_integrity_failure(tmp_path: Path) -> None:
    recorder = _new_recorder(tmp_path, [])
    recorder._mark_collection_integrity_issue("collection_snapshot_thread_busy")
    recorder._mark_collection_integrity_issue("collection_snapshot_thread_busy")

    summary = recorder._finalize_summary(
        reason="manual_key_stop",
        step_index=42,
        last_state="RUNNING",
        frame_count=7,
        queued_at=time.time(),
        frequency_status={"sample_hz": 15.0},
        pause_summary={},
    )

    assert summary["collection_integrity_valid"] is False
    assert summary["collection_integrity_reasons"] == [
        "collection_snapshot_thread_busy"
    ]


def test_finalize_marks_storage_queue_drops_as_integrity_failure(tmp_path: Path) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)
    recorder.writer_observer.storage_stats = lambda: {
        "enabled": True,
        "storage_queue_size": 4,
        "routes": {
            "left_wrist": {"dropped": 3},
            "wrist_depth:right_wrist": {"dropped": 1},
        },
    }

    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=42,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert task is not None
    recorder.start_pending_finalize_tasks()
    assert recorder.writer_observer.flush_started.wait(timeout=1.0)
    recorder.writer_observer.release_flush.set()
    recorder.wait_finalize_tasks()

    summary = recorder.writer.finalized[0][1]
    assert summary["collection_integrity_valid"] is False
    assert summary["collection_integrity_reasons"] == [
        "storage_queue_dropped:left_wrist=3,wrist_depth:right_wrist=1"
    ]


def test_thread_sample_contract_rejection_does_not_escape_record_step() -> None:
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.active_episode_index = 12
    recorder._pause_started_at = None
    recorder.collection_worker_mode = "thread"
    recorder._collection_integrity_reasons = []
    recorder._sample_overload_reason = None
    recorder._sample_modality_reason = None
    recorder.sample_builder = SimpleNamespace(
        include_debug_metadata=False,
        include_action_debug_metadata=False,
        last_camera_rejection=None,
        frame_index=0,
    )
    recorder.writer_observer = SimpleNamespace(
        handle_step=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            StrictV22SampleRejection(
                "left measured state stale",
                side="left",
                reason="stale_measured_state",
                age_sec=0.503,
            )
        )
    )
    recorder.writer_runtime = None

    result = recorder.record_step(SimpleNamespace(timestamp=10.0))

    assert result is None
    assert recorder.collection_overload_reason.startswith(
        "collection_sample_rejected:stale_measured_state:left:"
    )
    assert recorder._collection_integrity_reasons == [recorder.collection_overload_reason]


def test_async_sample_drain_failure_is_recorded_without_raising(tmp_path: Path) -> None:
    recorder = _new_recorder(tmp_path, [])
    future = Future()
    future.set_exception(RuntimeError("worker.plan_jobs failed"))
    recorder._sample_future = future

    assert recorder._drain_async_sample() is None
    assert recorder.collection_overload_reason == (
        "collection_sample_drain_error:RuntimeError:worker.plan_jobs failed"
    )
    assert recorder._collection_integrity_reasons == [
        "collection_sample_drain_error:RuntimeError:worker.plan_jobs failed"
    ]


def test_control_trace_episode_sink_is_subordinate_to_legacy_finalize(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-1",
            runtime_session_id="runtime-session-1",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=32,
    )
    sink = EpisodeTraceRecorderSink(
        trace,
        lifecycle_owner_id="canonical-gui-recorder",
        uuid_factory=lambda: "episode-uuid-12",
    )
    prepared = sink.prepare_episode(legacy_episode_index=12, prepared_at=1.0)
    sink.bind_prepared_episode(prepared, bound_at=1.0)
    recorder.control_trace_episode_sink = sink
    recorder._active_control_trace_preparation = prepared

    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=42,
        last_state="RUNNING",
        defer_finalize_start=True,
    )

    assert task is not None
    assert task.control_trace_release is not None
    assert not task.control_trace_release.seal.trace_sealed
    recorder.start_pending_finalize_tasks()
    assert recorder.writer_observer.flush_started.wait(timeout=1.0)
    recorder.writer_observer.release_flush.set()
    recorder.wait_finalize_tasks()

    assert task.success
    assert task.control_trace_finalize_report is not None
    assert task.control_trace_finalize_report.legacy_finalize_status == "finalized"
    assert task.control_trace_finalize_report.capture_commit_candidate is None
    assert events[-1] == "writer_finalize_episode"


def test_async_trace_release_never_blocks_stop_or_preempts_legacy_finalize(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-async",
            runtime_session_id="runtime-session-async",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=32,
    )
    scheduled: list[tuple[object, PendingEpisodeTraceRelease]] = []

    def schedule_release(callback):
        pending = PendingEpisodeTraceRelease()
        scheduled.append((callback, pending))
        return pending

    sink = EpisodeTraceRecorderSink(
        trace,
        lifecycle_owner_id="canonical-gui-recorder",
        release_scheduler=schedule_release,
        uuid_factory=lambda: "episode-uuid-async",
    )
    prepared = sink.prepare_episode(legacy_episode_index=12, prepared_at=1.0)
    sink.bind_prepared_episode(prepared, bound_at=1.0)
    recorder.control_trace_episode_sink = sink
    recorder._active_control_trace_preparation = prepared

    started_at = time.perf_counter()
    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=42,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert time.perf_counter() - started_at < 0.1
    assert task is not None
    assert isinstance(task.control_trace_release, PendingEpisodeTraceRelease)
    assert len(scheduled) == 1

    recorder.start_pending_finalize_tasks()
    assert recorder.writer_observer.flush_started.wait(timeout=1.0)
    recorder.writer_observer.release_flush.set()
    deadline = time.monotonic() + 1.0
    while "writer_finalize_episode" not in events and time.monotonic() < deadline:
        time.sleep(0.001)
    assert "writer_finalize_episode" in events
    assert not task.done

    callback, pending = scheduled.pop()
    pending.set_result(callback())
    recorder.wait_finalize_tasks()

    assert task.success
    assert task.control_trace_finalize_report is not None
    assert task.control_trace_finalize_report.legacy_finalize_status == "finalized"


def test_control_trace_binding_metadata_is_additive_at_legacy_episode_start(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.writer = _FakeWriter(events)
    recorder.active_episode_index = None
    recorder.active_started_at = None
    recorder.finalize_tasks = []
    recorder.collection_worker_mode = "thread"
    selected_episode: list[int] = []
    recorder.sample_builder = SimpleNamespace(
        set_episode=lambda episode_index: selected_episode.append(int(episode_index))
    )
    recorder.actual_feedback_warmup_status = lambda _now: (True, "ready")
    recorder._record_timestamps = deque()
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-start",
            runtime_session_id="runtime-session-start",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=32,
    )
    sink = EpisodeTraceRecorderSink(
        trace,
        lifecycle_owner_id="canonical-gui-recorder",
        uuid_factory=lambda: "episode-uuid-12",
    )
    recorder.control_trace_episode_sink = None
    recorder._active_control_trace_preparation = None
    recorder.attach_control_trace_episode_sink(sink)

    episode_index = recorder.start_episode({"existing": "metadata"}, now=1.0)

    assert episode_index == 12
    assert selected_episode == [12]
    assert recorder.writer.started[0]["existing"] == "metadata"
    trace_metadata = recorder.writer.started[0]["control_trace_episode"]
    assert trace_metadata["episode_uuid"] == "episode-uuid-12"
    assert "capture_commit_status" not in trace_metadata
    assert trace_metadata["seal_assessment_ref"].endswith(
        "/episode-uuid-12/seal_assessment.cjson"
    )
    assert trace_metadata["capture_commit_ref"].endswith(
        "/episode-uuid-12/capture_commit.cjson"
    )
    assert trace.episode_membership is not None
    assert trace.episode_membership.episode_uuid == "episode-uuid-12"


def test_process_native_stream_start_uses_trace_episode_uuid(
    tmp_path: Path,
) -> None:
    class _Capture:
        def __init__(self, prepared) -> None:
            self.episode_uuid = prepared.episode_uuid
            self.legacy_episode_index = prepared.legacy_episode_index

        def seal(self, *, sealed_at: float, sealed_clock_domain_id: str):
            raise AssertionError("seal is not part of episode start")

        def close_unsealed(self) -> None:
            return None

    class _ProcessWriter:
        next_episode_index = 12
        native_stream_contracts = {"camera.front.selected_rgb": object()}

        def __init__(self) -> None:
            self.started: list[tuple[dict[str, object], str]] = []

        def start_episode(self, *, metadata, episode_uuid: str) -> int:
            self.started.append((dict(metadata), str(episode_uuid)))
            return 12

        def discard_episode(self, _episode_index: int) -> bool:
            return True

    writer = _ProcessWriter()
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.writer = writer
    recorder.process_worker = writer
    recorder.collection_worker_mode = "process"
    recorder.active_episode_index = None
    recorder.active_started_at = None
    recorder.finalize_tasks = []
    recorder.gripper_force_contract = None
    recorder.actual_feedback_warmup_status = lambda _now: (True, "ready")
    recorder._record_timestamps = deque()
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-process-native",
            runtime_session_id="runtime-session-process-native",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=32,
    )
    sink = EpisodeTraceRecorderSink(
        trace,
        lifecycle_owner_id="canonical-gui-recorder",
        uuid_factory=lambda: "episode-uuid-process-native",
        committed_capabilities=("control_trace_core", "native_streams"),
        native_stream_writer_factory=_Capture,
    )
    recorder.control_trace_episode_sink = sink
    recorder._active_control_trace_preparation = None

    episode_index = recorder.start_episode({"existing": "metadata"}, now=1.0)

    assert episode_index == 12
    assert writer.started[0][1] == "episode-uuid-process-native"
    assert (
        writer.started[0][0]["control_trace_episode"]["episode_uuid"]
        == "episode-uuid-process-native"
    )


def test_episode_owner_passes_one_identity_to_v25p_and_control_trace(
    tmp_path: Path,
) -> None:
    class _Sidecar:
        dataset_root = tmp_path

        def __init__(self) -> None:
            self.prepared = None
            self.bound = None

        def prepare_episode(self, identity) -> None:
            self.prepared = identity

        def bind_prepared_episode(self, identity, **_kwargs) -> None:
            assert identity is self.prepared
            self.bound = identity

        def discard_episode(self, identity) -> None:
            assert identity is self.prepared

    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.writer = _FakeWriter([])
    recorder.collection_worker_mode = "thread"
    recorder.sample_builder = SimpleNamespace(set_episode=lambda _index: None)
    recorder.process_worker = SimpleNamespace(native_stream_contracts={})
    recorder.active_episode_index = None
    recorder.active_started_at = None
    recorder.finalize_tasks = []
    recorder.gripper_force_contract = None
    recorder.actual_feedback_warmup_status = lambda _now: (True, "ready")
    recorder._record_timestamps = deque()
    recorder._episode_uuid_factory = lambda: "episode-owner-uuid"
    sidecar = _Sidecar()
    recorder.v25p_sidecar_store = sidecar
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-shared-identity",
            runtime_session_id="runtime-session-shared-identity",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=32,
    )
    sink = EpisodeTraceRecorderSink(
        trace,
        lifecycle_owner_id="subordinate-control-trace",
        uuid_factory=lambda: "must-not-be-used",
    )
    recorder.control_trace_episode_sink = sink
    recorder._active_control_trace_preparation = None

    episode_index = recorder.start_episode({}, now=1.0)

    assert episode_index == 12
    assert sidecar.prepared is sidecar.bound
    assert recorder.active_episode_identity is sidecar.prepared
    assert sidecar.prepared.episode_uuid == "episode-owner-uuid"
    assert trace.episode_membership is not None
    assert trace.episode_membership.episode_uuid == "episode-owner-uuid"
    metadata = recorder.writer.started[0]
    assert metadata["v25p_episode"]["episode_uuid"] == "episode-owner-uuid"
    assert metadata["control_trace_episode"]["episode_uuid"] == "episode-owner-uuid"


def test_v25p_bind_failure_rolls_back_legacy_and_trace_preparation(
    tmp_path: Path,
) -> None:
    class _FailingSidecar:
        dataset_root = tmp_path

        def __init__(self) -> None:
            self.prepared = None
            self.discarded = None

        def prepare_episode(self, identity) -> None:
            self.prepared = identity

        def bind_prepared_episode(self, identity, **_kwargs) -> None:
            assert identity is self.prepared
            raise RuntimeError("sidecar bind failed")

        def discard_episode(self, identity) -> None:
            self.discarded = identity

    writer = _FakeWriter([])
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.writer = writer
    recorder.collection_worker_mode = "thread"
    recorder.sample_builder = SimpleNamespace(set_episode=lambda _index: None)
    recorder.process_worker = SimpleNamespace(native_stream_contracts={})
    recorder.active_episode_index = None
    recorder.active_started_at = None
    recorder.finalize_tasks = []
    recorder.gripper_force_contract = None
    recorder.actual_feedback_warmup_status = lambda _now: (True, "ready")
    recorder._record_timestamps = deque()
    recorder._episode_uuid_factory = lambda: "episode-bind-failure"
    sidecar = _FailingSidecar()
    recorder.v25p_sidecar_store = sidecar
    trace = ControlTraceRuntime(
        RuntimeTraceSession(
            trace_session_id="trace-session-bind-failure",
            runtime_session_id="runtime-session-bind-failure",
            control_domain_id="robot-0",
            profile_id="plain_dry_run_v1",
            started_at=0.0,
            clock_domain_id="local_monotonic",
        ),
        capacity=32,
    )
    sink = EpisodeTraceRecorderSink(
        trace,
        lifecycle_owner_id="subordinate-control-trace",
    )
    recorder.control_trace_episode_sink = sink
    recorder._active_control_trace_preparation = None

    with pytest.raises(RuntimeError, match="sidecar bind failed"):
        recorder.start_episode({}, now=1.0)

    assert writer.discarded == [12]
    assert sidecar.discarded is sidecar.prepared
    assert trace.episode_membership is None
    assert recorder.active_episode_index is None


def test_control_trace_post_finalize_failure_does_not_rewrite_legacy_success(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)

    class _FailingTraceSink:
        def commit_now(self) -> float:
            return time.time()

        def legacy_finalize_completed(self, _released, *, completed_at: float):
            raise RuntimeError(f"trace publisher unavailable at {completed_at}")

        def legacy_finalize_failed(self, _released, *, completed_at: float, error: str):
            return None

    recorder.control_trace_episode_sink = _FailingTraceSink()
    task = recorder.stop_episode(
        reason="manual_key_stop",
        step_index=42,
        last_state="RUNNING",
        defer_finalize_start=True,
    )
    assert task is not None
    recorder.start_pending_finalize_tasks()
    assert recorder.writer_observer.flush_started.wait(timeout=1.0)
    recorder.writer_observer.release_flush.set()
    recorder.wait_finalize_tasks()

    assert task.success
    assert task.control_trace_error.startswith("post_finalize:RuntimeError:")
    assert recorder.writer.finalized[0][0] == 12


def test_process_finalize_receipts_are_forwarded_only_on_opt_in_result(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    recorder = _new_recorder(tmp_path, events)
    receipt = object()

    class _ExternalProcessWriter:
        native_external_stream_bindings = {
            "camera.front.selected_rgb": object(),
        }
        process_status: dict[str, object] = {}

        def flush(self) -> None:
            events.append("writer_flush")

        def finalize_episode(self, episode_index: int, *, summary=None):
            assert episode_index == 12
            events.append("writer_finalize_episode")
            return SimpleNamespace(
                native_stream_artifact_receipts=(receipt,),
                artifact_receipt_error="",
            )

    writer = _ExternalProcessWriter()
    recorder.collection_worker_mode = "process"
    recorder.process_worker = writer
    recorder.writer = writer
    forwarded: list[tuple[object, ...]] = []

    class _ReceiptTraceSink:
        def resolve_release(self, released):
            events.append("native_stream_release_freeze")
            return ("resolved", released)

        def commit_now(self) -> float:
            return time.time()

        def legacy_finalize_completed(
            self,
            _released,
            *,
            completed_at: float,
            native_stream_artifact_receipts,
        ):
            assert completed_at > 0.0
            forwarded.append(tuple(native_stream_artifact_receipts))
            return "trace-finalized"

    recorder.control_trace_episode_sink = _ReceiptTraceSink()
    task = real_teleop.RealDatasetFinalizeTask(
        episode_index=12,
        frame_count=1,
        reason="test",
        queued_at=time.time(),
        summary={"collection_integrity_valid": True},
        control_trace_release=object(),
    )
    task.mark_started()

    recorder._run_finalize_task(task)

    assert task.success
    assert task.control_trace_finalize_report == "trace-finalized"
    assert forwarded == [(receipt,)]
    assert events == [
        "writer_flush",
        "native_stream_release_freeze",
        "writer_finalize_episode",
    ]
    assert task.control_trace_release[0] == "resolved"


def test_thread_writer_contract_failure_becomes_pauseable_integrity_reason() -> None:
    recorder = real_teleop.RealTeleopDatasetRecorder.__new__(
        real_teleop.RealTeleopDatasetRecorder
    )
    recorder.collection_worker_mode = "thread"
    recorder._sample_overload_reason = None
    recorder._collection_integrity_reasons = []
    future: Future[object] = Future()
    future.set_exception(RuntimeError("invalid v2.3 measured TCP"))
    recorder._sample_future = future

    reason = recorder.poll_collection_overload()

    assert reason.startswith("collection_writer_error:RuntimeError:")
    assert recorder._sample_future is None
    assert recorder._collection_integrity_reasons == [reason]

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from robot_motion.position_control import (
    POSITION_TRANSPORT_TARGET_SEQUENCE_KEY,
    PercentileStats,
    PositionAuditFanout,
    PositionCommandAuditEvent,
    PositionReferenceAuditEvent,
    PositionReferenceSideAudit,
    PositionTimingAuditEvent,
    resolve_periodic_tick,
)
from teleop_runtime.backends.arms import RM65ArmBackend, RM65ArmBackendConfig
from teleop_runtime.collection.diagnostics import PhaseTimer
from teleop_runtime.debug.run_vive_rm65_real_teleop import _record_position_loop_timings
from teleop_runtime.position_control import (
    POSITION_COMMAND_SIDECAR_NAME,
    POSITION_REFERENCE_SIDECAR_NAME,
    POSITION_SUMMARY_NAME,
    POSITION_TIMING_SIDECAR_NAME,
    PositionDebugRecorder,
    PositionDebugRecorderConfig,
)
from teleop_runtime.schemas import EffectorCommand


def _command_event(
    *,
    target_sequence: int = 7,
    tick_sequence: int = 1,
    side: str = "left",
    wakeup_monotonic_time: float = 1.001,
    wakeup_gap_sec: float | None = None,
    sdk_send_gap_sec: float | None = None,
    sdk_accepted: bool = True,
    skipped_tick_count: int = 0,
    deadline_missed: bool = False,
    planner_lock_delay_sec: float = 0.0002,
) -> PositionCommandAuditEvent:
    scheduled_monotonic_time = wakeup_monotonic_time - 0.001
    return PositionCommandAuditEvent(
        side=side,
        tick_sequence=tick_sequence,
        timestamp_wall=100.0,
        timestamp_monotonic=wakeup_monotonic_time + 0.001,
        scheduled_monotonic_time=scheduled_monotonic_time,
        wakeup_monotonic_time=wakeup_monotonic_time,
        deadline_monotonic_time=scheduled_monotonic_time + 0.01,
        wakeup_lag_sec=0.001,
        raw_wakeup_lag_sec=0.001,
        skipped_tick_count=skipped_tick_count,
        deadline_missed=deadline_missed,
        planner_sample_monotonic_time=(
            wakeup_monotonic_time + planner_lock_delay_sec
        ),
        planner_lock_delay_sec=planner_lock_delay_sec,
        wakeup_gap_sec=wakeup_gap_sec,
        target_active=True,
        target_sequence=target_sequence,
        segment_start_qpos=np.zeros(6),
        segment_target_qpos=np.ones(6),
        segment_start_monotonic_time=0.99,
        segment_duration_sec=0.02,
        interpolation_alpha=0.5,
        q_cmd=np.full(6, 0.5),
        sdk_attempted=True,
        sdk_attempt_sequence=tick_sequence,
        sdk_send_gap_sec=sdk_send_gap_sec,
        sdk_call_ms=0.2,
        sdk_result_code=0 if sdk_accepted else 1,
        sdk_result_text="0" if sdk_accepted else "1",
        sdk_accepted=sdk_accepted,
        sdk_accepted_sequence=tick_sequence if sdk_accepted else None,
    )


def _reference_event(*, target_sequence: int = 7) -> PositionReferenceAuditEvent:
    return PositionReferenceAuditEvent(
        epoch=11,
        reference_sequence=3,
        source="test",
        source_wall_time=99.9,
        generated_wall_time=100.0,
        published_wall_time=100.1,
        generated_monotonic_time=1.0,
        published_monotonic_time=1.1,
        references={
            "left": PositionReferenceSideAudit(
                side="left",
                q_ik_raw=np.ones(6),
                q_ref=np.ones(6),
                pose_ref=np.eye(4),
                valid=True,
                hold=False,
            )
        },
        transport_target_sequences={"left": target_sequence},
    )


def test_percentiles_are_exact_for_retained_window_and_report_truncation() -> None:
    stats = PercentileStats(max_samples=3)
    for value in (1.0, 2.0, 3.0, 4.0):
        stats.record(value)

    snapshot = stats.snapshot()

    assert snapshot.count == 4
    assert snapshot.sample_count == 3
    assert snapshot.dropped_sample_count == 1
    assert snapshot.p50 == 3.0
    assert np.isclose(snapshot.p95, 3.9)
    assert np.isclose(snapshot.p99, 3.98)
    assert snapshot.max == 4.0


def test_periodic_tick_accounts_skips_without_catchup_burst() -> None:
    decision = resolve_periodic_tick(
        scheduled_monotonic_time=10.0,
        wakeup_monotonic_time=10.031,
        period_sec=0.01,
    )

    assert decision.skipped_tick_count == 3
    assert np.isclose(decision.scheduled_monotonic_time, 10.03)
    assert np.isclose(decision.wakeup_lag_sec, 0.001)
    assert np.isclose(decision.deadline_monotonic_time, 10.04)
    assert decision.deadline_missed(10.041)


def test_real_main_loop_timing_bridge_emits_stage_and_deadline_summary() -> None:
    clock_values = iter((10.0, 10.005, 10.030))
    timer = PhaseTimer(clock=lambda: next(clock_values))
    timer.mark("loop.session_run_step")
    events: list[PositionTimingAuditEvent] = []
    fanout = PositionAuditFanout((events.append,))

    _record_position_loop_timings(
        fanout,
        timer=timer,
        step_index=7,
        state="running",
        outer_control_hz=45.0,
    )

    assert [event.stage for event in events] == [
        "loop.session_run_step",
        "main_loop_total",
    ]
    assert np.isclose(events[0].elapsed_ms, 5.0)
    assert np.isclose(events[1].elapsed_ms, 30.0)
    assert events[1].deadline_ms == 1000.0 / 45.0
    assert events[1].deadline_missed
    assert events[1].metadata["step_index"] == 7


def test_audit_fanout_is_dynamic_and_isolates_sink_failures() -> None:
    observed: list[PositionCommandAuditEvent] = []

    def failing_sink(_event) -> None:
        raise RuntimeError("recorder failed")

    fanout = PositionAuditFanout((observed.append, failing_sink))
    event = _command_event()

    fanout(event)

    assert observed == [event]
    assert fanout.error_count == 1
    assert "recorder failed" in str(fanout.last_error)
    assert fanout.add_sink(observed.append)
    extra_sink = fanout.sinks()[-1]
    assert not fanout.add_sink(extra_sink)
    assert fanout.remove_sink(extra_sink)
    assert not fanout.remove_sink(extra_sink)


def test_full_recorder_late_binds_command_that_raced_ahead_of_reference(
    tmp_path: Path,
) -> None:
    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(output_dir=tmp_path, mode="full")
    )

    assert recorder.record(_command_event(target_sequence=17))
    assert recorder.record(_reference_event(target_sequence=17))
    assert recorder.record(
        PositionTimingAuditEvent(
            stage="collision_and_ik",
            elapsed_ms=4.25,
            timestamp_wall=100.2,
            timestamp_monotonic=1.2,
            epoch=11,
            reference_sequence=3,
            deadline_ms=22.22,
        )
    )
    recorder.close()

    command_rows = [
        json.loads(line)
        for line in (tmp_path / POSITION_COMMAND_SIDECAR_NAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    reference_rows = [
        json.loads(line)
        for line in (tmp_path / POSITION_REFERENCE_SIDECAR_NAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    summary = json.loads((tmp_path / POSITION_SUMMARY_NAME).read_text(encoding="utf-8"))
    timing_rows = [
        json.loads(line)
        for line in (tmp_path / POSITION_TIMING_SIDECAR_NAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert len(command_rows) == 1
    assert command_rows[0]["reference_epoch"] == 11
    assert command_rows[0]["reference_sequence"] == 3
    assert command_rows[0]["reference_binding_status"] == "late_bound"
    assert reference_rows[0]["transport_target_sequences"] == {"left": 17}
    assert timing_rows[0]["stage"] == "collision_and_ik"
    assert timing_rows[0]["elapsed_ms"] == 4.25
    assert summary["enqueue_drop_count"] == 0
    assert summary["statistics"]["counters"]["command_reference_late_bound_count"] == 1
    assert (
        summary["statistics"]["percentiles"]["timing.collision_and_ik.elapsed_ms"][
            "p99"
        ]
        == 4.25
    )


def test_off_recorder_has_no_thread_queue_or_files() -> None:
    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(output_dir=None, mode="off")
    )

    assert not recorder.record(_command_event())
    snapshot = recorder.snapshot()
    assert snapshot["mode"] == "off"
    assert not snapshot["enabled"]
    assert snapshot["closed"]
    assert snapshot["files"]["summary"] is None


def test_summary_mode_writes_stats_without_high_rate_sidecars(tmp_path: Path) -> None:
    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(
            output_dir=tmp_path,
            mode="summary",
            context={
                "active_sides": ["left", "right"],
                "arm_execution": {
                    "resolved_profile": "position_45_150hz",
                    "rates": {"reference_hz": 45.0, "arm_control_hz": 150.0},
                },
            },
        )
    )
    recorder.record(_command_event())
    recorder.record(_reference_event())
    recorder.close()

    summary = json.loads((tmp_path / POSITION_SUMMARY_NAME).read_text(encoding="utf-8"))

    assert summary["mode"] == "summary"
    assert summary["context"]["active_sides"] == ["left", "right"]
    assert summary["context"]["arm_execution"]["rates"]["reference_hz"] == 45.0
    assert summary["statistics"]["counters"]["command_tick_count"] == 1
    assert summary["statistics"]["percentiles"]["sdk_call_ms"]["p99"] == 0.2
    assert not (tmp_path / POSITION_COMMAND_SIDECAR_NAME).exists()
    assert not (tmp_path / POSITION_REFERENCE_SIDECAR_NAME).exists()


def test_runtime_summary_checkpoint_is_disabled_until_recorder_close(
    tmp_path: Path,
) -> None:
    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(
            output_dir=tmp_path,
            mode="summary",
            summary_checkpoint_every_events=1,
        )
    )
    assert recorder.record(_command_event())
    recorder._queue.join()

    assert not (tmp_path / POSITION_SUMMARY_NAME).exists()
    assert recorder.snapshot()["runtime_summary_checkpoints"] is False

    recorder.close()
    summary = json.loads((tmp_path / POSITION_SUMMARY_NAME).read_text(encoding="utf-8"))
    assert summary["processed_count"] == 1
    assert summary["runtime_summary_checkpoints"] is False


def test_summary_reports_aggregate_and_per_side_actual_rates_and_counters(
    tmp_path: Path,
) -> None:
    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(output_dir=tmp_path, mode="summary")
    )
    recorder.record(
        _command_event(
            side="left",
            tick_sequence=1,
            wakeup_gap_sec=0.005,
            sdk_send_gap_sec=0.008,
        )
    )
    recorder.record(
        _command_event(
            side="right",
            tick_sequence=1,
            wakeup_gap_sec=0.010,
            sdk_send_gap_sec=0.020,
            sdk_accepted=False,
            skipped_tick_count=2,
            deadline_missed=True,
        )
    )
    recorder.close()

    summary = json.loads((tmp_path / POSITION_SUMMARY_NAME).read_text(encoding="utf-8"))
    counters = summary["statistics"]["counters"]
    metrics = summary["statistics"]["percentiles"]

    assert counters["command_tick_count"] == 2
    assert counters["command_skipped_tick_count"] == 2
    assert counters["command_deadline_miss_count"] == 1
    assert counters["sdk_attempt_count"] == 2
    assert counters["sdk_accepted_count"] == 1
    assert counters["sdk_failed_count"] == 1
    assert counters["command.left.tick_count"] == 1
    assert counters["command.right.tick_count"] == 1
    assert counters["command.right.skipped_tick_count"] == 2
    assert counters["command.right.deadline_miss_count"] == 1
    assert counters["sdk.left.attempt_count"] == 1
    assert counters["sdk.left.accepted_count"] == 1
    assert counters["sdk.right.attempt_count"] == 1
    assert counters["sdk.right.failed_count"] == 1
    assert metrics["command_wakeup_actual_hz"]["p50"] == 150.0
    assert metrics["sdk_send_actual_hz"]["p50"] == 87.5
    assert metrics["command.left.wakeup_actual_hz"]["p50"] == 200.0
    assert metrics["command.right.wakeup_actual_hz"]["p50"] == 100.0
    assert metrics["sdk.left.send_actual_hz"]["p50"] == 125.0
    assert metrics["sdk.right.send_actual_hz"]["p50"] == 50.0
    assert metrics["command.left.wakeup_gap_ms"]["p50"] == 5.0
    assert metrics["sdk.right.send_gap_ms"]["p50"] == 20.0
    assert metrics["command.left.wakeup_lag_ms"]["p50"] == 1.0
    assert np.isclose(metrics["command_planner_lock_delay_ms"]["p50"], 0.2)
    assert np.isclose(
        metrics["command.left.planner_lock_delay_ms"]["p50"],
        0.2,
    )
    assert metrics["sdk.right.call_ms"]["p50"] == 0.2


def test_summary_derives_wakeup_hz_per_side_without_infinite_zero_gap(
    tmp_path: Path,
) -> None:
    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(output_dir=tmp_path, mode="summary")
    )
    recorder.record(
        _command_event(
            side="left",
            tick_sequence=1,
            wakeup_monotonic_time=1.001,
        )
    )
    recorder.record(
        _command_event(
            side="left",
            tick_sequence=2,
            wakeup_monotonic_time=1.011,
        )
    )
    recorder.record(
        _command_event(
            side="left",
            tick_sequence=3,
            wakeup_monotonic_time=1.011,
            wakeup_gap_sec=0.0,
        )
    )
    recorder.close()

    summary = json.loads((tmp_path / POSITION_SUMMARY_NAME).read_text(encoding="utf-8"))
    metrics = summary["statistics"]["percentiles"]

    assert np.isclose(metrics["command_derived_wakeup_gap_ms"]["p50"], 10.0)
    assert metrics["command_wakeup_actual_hz"]["sample_count"] == 1
    assert np.isclose(metrics["command_wakeup_actual_hz"]["p50"], 100.0)
    assert metrics["command.left.wakeup_actual_hz"]["sample_count"] == 1
    assert np.isclose(metrics["command.left.wakeup_actual_hz"]["p50"], 100.0)


def test_recorder_queue_is_bounded_and_producer_never_waits(tmp_path: Path) -> None:
    entered = threading.Event()
    release = threading.Event()

    class _SlowRecorder(PositionDebugRecorder):
        def _process_event(self, event) -> None:
            entered.set()
            release.wait(timeout=1.0)
            super()._process_event(event)

    recorder = _SlowRecorder(
        PositionDebugRecorderConfig(
            output_dir=tmp_path,
            mode="summary",
            queue_capacity=1,
        )
    )
    assert recorder.record(_command_event(tick_sequence=1))
    assert entered.wait(timeout=1.0)
    assert recorder.record(_command_event(tick_sequence=2))
    assert not recorder.record(_command_event(tick_sequence=3))
    release.set()
    recorder.close()

    assert recorder.snapshot()["enqueue_drop_count"] == 1


class _AuditArm:
    def __init__(self) -> None:
        self.sent_joint_deg: list[np.ndarray] = []

    def Movej_CANFD(self, joint_deg, follow=False, trajectory_mode=0, radio=0):
        del follow, trajectory_mode, radio
        self.sent_joint_deg.append(np.asarray(joint_deg, dtype=float).copy())
        return 0


def test_rm65_stream_stamps_target_and_emits_complete_tick_audit() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=150.0,
            background_streaming_target_max_age_sec=1.0,
        )
    )
    arm = _AuditArm()
    backend.arm = arm
    events: list[PositionCommandAuditEvent] = []

    def sink(event) -> None:
        assert isinstance(event, PositionCommandAuditEvent)
        events.append(event)
        backend._stream_stop_event.set()

    backend.set_position_audit_sink(sink)
    command = EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="joint_vector",
        value=np.full(6, 0.1),
    )
    backend.send_command(command)

    assert command.metadata[POSITION_TRANSPORT_TARGET_SEQUENCE_KEY] == 1
    backend._stream_stop_event.clear()
    backend._background_stream_loop_impl()

    assert len(events) == 1
    event = events[0]
    assert event.target_sequence == 1
    assert event.sdk_attempted
    assert event.sdk_result_code == 0
    assert event.sdk_accepted
    assert event.sdk_attempt_sequence == 1
    assert event.sdk_accepted_sequence == 1
    assert event.segment_start_qpos == (0.0,) * 6
    assert event.segment_target_qpos == (0.1,) * 6
    assert event.q_cmd is not None
    assert event.interpolation_alpha is not None
    assert event.planner_sample_monotonic_time is not None
    assert event.planner_sample_monotonic_time >= event.wakeup_monotonic_time
    assert event.planner_lock_delay_sec is not None
    assert event.planner_lock_delay_sec >= 0.0
    assert event.metadata["segment_planner"] == (
        "robot_motion.position_control.JointCommandInterpolator"
    )
    stream = backend.get_state()["background_streaming"]
    assert stream["scheduled_tick_count"] == 1
    assert stream["executed_tick_count"] == 1
    assert stream["skipped_tick_count"] == 0
    assert stream["timing_percentiles"]["wakeup_lag_ms"]["count"] == 1
    assert stream["timing_percentiles"]["sdk_call_ms"]["count"] == 1


def test_rm65_stream_samples_planner_time_after_target_publication_race(
    monkeypatch,
) -> None:
    """A target may be published after wakeup but before the worker gets the lock."""

    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=150.0,
            background_streaming_target_max_age_sec=1.0,
        )
    )
    backend.arm = _AuditArm()
    target = np.full(6, 0.1, dtype=float)
    with backend._state_lock:
        backend._stream_planner.reset(
            {"left": np.zeros(6)},
            epoch=0,
            monotonic_time=9.0,
        )
        backend._stream_planner.accept_qpos_target(
            {"left": target},
            epoch=0,
            sequence=1,
            published_monotonic_time=10.05,
            monotonic_time=10.05,
            source="race_test",
            anchor_qpos_by_side={"left": np.zeros(6)},
        )
        backend._stream_target_qpos = target.copy()
        backend._stream_target_source = "race_test"
        backend._stream_target_timestamp = 100.0
        backend._stream_target_monotonic = 10.05
        backend._stream_target_sequence = 1
        backend._stream_segment_start_qpos = np.zeros(6)
        backend._stream_segment_start_time = 10.05

    class _Clock:
        def __init__(self) -> None:
            self._values = iter((10.0, 10.0, 10.0, 10.06))
            self._fallback = 10.06

        def __call__(self) -> float:
            try:
                return next(self._values)
            except StopIteration:
                self._fallback += 0.001
                return self._fallback

    monkeypatch.setattr(
        "teleop_runtime.backends.arms.rm65.time.monotonic",
        _Clock(),
    )
    events: list[PositionCommandAuditEvent] = []

    def sink(event) -> None:
        if isinstance(event, PositionCommandAuditEvent):
            events.append(event)
            backend._stream_stop_event.set()

    backend.set_position_audit_sink(sink)
    backend._background_stream_loop_impl()

    assert len(events) == 1
    event = events[0]
    assert event.wakeup_monotonic_time == 10.0
    assert event.segment_start_monotonic_time == 10.05
    assert event.planner_sample_monotonic_time == 10.06
    assert np.isclose(event.planner_lock_delay_sec, 0.06)
    assert event.sdk_accepted is True
    assert backend._stream_fault is None


def test_rm65_stream_snapshot_accounts_skipped_and_executed_misses_separately() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        )
    )

    tick_sequence = backend._record_background_stream_tick(
        wakeup_lag_sec=0.001,
        raw_wakeup_lag_sec=0.021,
        wakeup_gap_sec=0.025,
        skipped_tick_count=2,
        deadline_missed=True,
    )
    stream = backend.get_state()["background_streaming"]

    assert tick_sequence == 1
    assert stream["scheduled_tick_count"] == 3
    assert stream["executed_tick_count"] == 1
    assert stream["skipped_tick_count"] == 2
    assert stream["deadline_miss_count"] == 3
    assert stream["executed_deadline_miss_count"] == 1
    assert stream["timing_percentiles"]["wakeup_lag_ms"]["p99"] == 1.0
    assert stream["timing_percentiles"]["raw_wakeup_lag_ms"]["p99"] == 21.0


def test_rm65_worker_exception_emits_terminal_fault_audit_and_isolates_observer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
        )
    )
    events: list[PositionTimingAuditEvent] = []

    def crashing_worker() -> None:
        raise RuntimeError("worker exploded before first tick")

    def failing_observer(event) -> None:
        events.append(event)
        raise RuntimeError("audit observer failed")

    monkeypatch.setattr(backend, "_background_stream_loop_impl", crashing_worker)
    backend.set_position_audit_sink(failing_observer)

    # The observer failure must not escape the worker wrapper or hide its fault.
    backend._background_stream_loop()

    assert len(events) == 1
    terminal = events[0]
    assert isinstance(terminal, PositionTimingAuditEvent)
    assert terminal.stage == "rm65_background_stream_worker_terminal"
    assert terminal.side == "left"
    assert terminal.metadata["terminal"] is True
    assert terminal.metadata["terminal_status"] == "fault"
    assert terminal.metadata["fault_latched"] is True
    assert terminal.metadata["worker_exception_type"] == "RuntimeError"
    assert "worker exploded before first tick" in terminal.metadata["fault_message"]
    assert terminal.metadata["shutdown_complete"] is True

    stream = backend.get_state()["background_streaming"]
    assert stream["shutdown_complete"] is True
    assert stream["fault_count"] == 1
    assert "worker exploded before first tick" in stream["fault"]
    assert stream["audit_sink_error_count"] == 1
    assert "audit observer failed" in stream["last_audit_sink_error"]

    recorder = PositionDebugRecorder(
        PositionDebugRecorderConfig(output_dir=tmp_path, mode="full")
    )
    assert recorder.record(terminal)
    recorder.close()
    timing_rows = [
        json.loads(line)
        for line in (tmp_path / POSITION_TIMING_SIDECAR_NAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    summary = json.loads((tmp_path / POSITION_SUMMARY_NAME).read_text(encoding="utf-8"))
    counters = summary["statistics"]["counters"]

    assert timing_rows[0]["metadata"]["terminal_status"] == "fault"
    assert counters["timing.rm65_background_stream_worker_terminal.count"] == 1
    assert counters["timing.rm65_background_stream_worker_terminal.fault_count"] == 1
    assert counters["timing.left.rm65_background_stream_worker_terminal.count"] == 1
    assert counters["timing.left.rm65_background_stream_worker_terminal.fault_count"] == 1

from __future__ import annotations

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
    POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED,
    SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP,
    resolve_periodic_tick,
)
from teleop_runtime.backends.arms.rm65 import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK,
    _final_follow_qpos_target,
    _next_background_stream_schedule_after_skipped_tick,
    _position_trajectory_policy,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.schemas import EffectorCommand


class _FakeArm:
    def __init__(self, *, result: int = 0) -> None:
        self.result = int(result)
        self.sent_joint_deg: list[np.ndarray] = []
        self.slow_stop_calls = 0
        self.uninit_calls = 0
        self.close_calls = 0

    def Movej_CANFD(self, joint_deg, follow=False, trajectory_mode=0, radio=0):
        del follow, trajectory_mode, radio
        self.sent_joint_deg.append(np.asarray(joint_deg, dtype=float).reshape(6).copy())
        return self.result

    def Arm_Slow_Stop(self):
        self.slow_stop_calls += 1
        return 0

    def RM_API_UnInit(self):
        self.uninit_calls += 1

    def Arm_Socket_Close(self):
        self.close_calls += 1


class _RecordingDualArmRetargeter:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def set_current_qpos_pair(self, **kwargs) -> None:
        self.calls.append(dict(kwargs))


def test_sample_aligned_backend_builds_configured_source_latch_mode() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            position_command_planner=(
                POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
            ),
            position_sample_aligned_source_hz=60.0,
            position_sample_aligned_source_latch_mode=(
                SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
            ),
            position_sample_aligned_horizon_source_steps=2,
            position_velocity_ramp_sec=0.05,
            position_max_command_step_deg=0.675,
            arm_follow_qpos_alpha=1.0,
            singularity_protection_enabled=False,
        )
    )

    assert backend._stream_planner.snapshot()["source_latch_mode"] == (
        SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
    )

    backend.arm = _FakeArm()
    first = _command(0.05)
    first.metadata["sample_aligned_source_timestamp"] = 100.0
    backend.send_command(first)
    assert first.metadata[
        "position_transport_sample_aligned_bucket_held"
    ] is False

    # This is intentionally inside the historical 1/60 s time bucket.  The
    # strict-timestamp successor must still admit it as the next source cut.
    second = _command(0.10)
    second.metadata["sample_aligned_source_timestamp"] = 100.012
    backend.send_command(second)
    assert second.metadata[
        "position_transport_sample_aligned_bucket_held"
    ] is False
    assert second.metadata["sample_aligned_admission"]["source_bucket"] == 1
    assert second.metadata["sample_aligned_admission"]["source_latch_mode"] == (
        SAMPLE_ALIGNED_SOURCE_LATCH_STRICT_TIMESTAMP
    )


def test_rm65_skipped_tick_rebases_to_one_fresh_period_without_catchup() -> None:
    assert np.isclose(
        _next_background_stream_schedule_after_skipped_tick(
            next_scheduled=10.04,
            wakeup_monotonic=10.031,
            skipped_tick_count=3,
            period_sec=0.01,
            no_catch_up_after_send=True,
        ),
        10.041,
    )
    assert _next_background_stream_schedule_after_skipped_tick(
        next_scheduled=10.04,
        wakeup_monotonic=10.031,
        skipped_tick_count=3,
        period_sec=0.01,
        no_catch_up_after_send=False,
    ) == 10.04


def test_rm65_stream_skips_late_tick_without_sending_catchup_command(monkeypatch) -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=100.0,
            background_streaming_target_max_age_sec=1.0,
            background_streaming_no_catch_up_after_send=True,
        )
    )
    arm = _FakeArm()
    backend.arm = arm
    backend.send_command(
        EffectorCommand(
            effector_name="left_arm",
            side="left",
            command_type="joint_vector",
            value=np.full(6, 0.1),
        )
    )
    events: list[object] = []

    def sink(event) -> None:
        events.append(event)
        if len(events) >= 2:
            backend._stream_stop_event.set()

    backend.set_position_audit_sink(sink)
    real_resolve = resolve_periodic_tick
    resolve_calls = 0

    def delayed_first_tick(**kwargs):
        nonlocal resolve_calls
        decision = real_resolve(**kwargs)
        if resolve_calls == 0:
            decision = replace(decision, skipped_tick_count=1)
        resolve_calls += 1
        return decision

    monkeypatch.setattr(
        "teleop_runtime.backends.arms.rm65.resolve_periodic_tick",
        delayed_first_tick,
    )
    backend._stream_stop_event.clear()
    backend._background_stream_loop_impl()

    assert len(events) == 2
    assert len(arm.sent_joint_deg) == 1
    assert events[0].sdk_attempted is False
    assert events[0].metadata["late_tick_skipped"] is True
    assert events[0].metadata["transport_block_reason"] == (
        "late_tick_skipped_no_catch_up"
    )
    assert events[1].sdk_attempted is True
    assert backend.get_state()["background_streaming"][
        "no_catch_up_after_send"
    ] is True


def test_rm65_stream_rechecks_target_age_before_sdk_send(monkeypatch) -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=100.0,
            background_streaming_target_max_age_sec=0.1,
        )
    )
    arm = _FakeArm()
    backend.arm = arm
    target = np.full(6, 0.1, dtype=float)
    with backend._state_lock:
        backend._stream_planner.reset(
            {"left": np.zeros(6)},
            epoch=0,
            monotonic_time=0.0,
        )
        backend._stream_planner.accept_qpos_target(
            {"left": target},
            epoch=0,
            sequence=1,
            published_monotonic_time=0.0,
            monotonic_time=0.0,
            source="stale_before_send_test",
            anchor_qpos_by_side={"left": np.zeros(6)},
        )
        backend._stream_target_qpos = target.copy()
        backend._stream_target_source = "stale_before_send_test"
        backend._stream_target_timestamp = 100.0
        backend._stream_target_monotonic = 0.0
        backend._stream_target_sequence = 1
        backend._stream_segment_start_qpos = np.zeros(6)
        backend._stream_segment_start_time = 0.0

    # The target is fresh at wakeup/planner sampling (0.01 s), then crosses
    # its 100 ms deadline while waiting for the command lane (0.2 s).
    clock_values = iter((0.0, 0.0, 0.0, 0.01, 0.2))

    def monotonic() -> float:
        try:
            return next(clock_values)
        except StopIteration:
            return 0.2

    monkeypatch.setattr("teleop_runtime.backends.arms.rm65.time.monotonic", monotonic)
    events: list[object] = []

    def sink(event) -> None:
        events.append(event)
        backend._stream_stop_event.set()

    backend.set_position_audit_sink(sink)
    backend._stream_stop_event.clear()
    backend._background_stream_loop_impl()

    assert len(events) == 1
    event = events[0]
    assert event.sdk_attempted is False
    assert event.stale_target_expired is True
    assert event.metadata["target_stale_checked_before_send"] is True
    assert event.metadata["transport_block_reason"] == "target_stale_before_send"
    assert arm.sent_joint_deg == []


def _backend(
    *,
    result: int = 0,
    target_max_age_sec: float = 1.0,
    target_admission_mode: str = RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK,
) -> tuple[RM65ArmBackend, _FakeArm]:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            control_dt=1.0 / 75.0,
            max_angvel=(0.7, 0.7, 0.9, 1.5, 1.5, 2.0),
            singularity_protection_enabled=False,
            background_streaming_enabled=True,
            background_streaming_hz=150.0,
            background_streaming_target_max_age_sec=target_max_age_sec,
            background_streaming_target_admission_mode=target_admission_mode,
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            position_velocity_ramp_sec=0.10,
            position_degraded_speed_scale=0.10,
        )
    )
    arm = _FakeArm(result=result)
    backend.arm = arm
    return backend, arm


def _command(
    target: float,
    *,
    collision_status: str = "tracking",
    step_limited: bool = False,
    source: str | None = None,
) -> EffectorCommand:
    metadata = {
        "action_pose_source": "limited_solver_output_qpos",
        "collision_avoidance_status": collision_status,
        "ik_jump_gate_status": "pass",
        "execution_step_metadata": {
            "enabled": True,
            "step_limited": bool(step_limited),
        },
        "ik_solve_pose_error": {
            "translation_m": 0.0,
            "rotation_deg": 0.0,
        },
    }
    if source is not None:
        metadata["source"] = source
    return EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="joint_vector",
        value=np.full(6, target),
        metadata=metadata,
    )


def test_rm65_position_planner_default_remains_linear() -> None:
    config = RM65ArmBackendConfig(
        arm_ip="0.0.0.0",
        side="left",
        connect_hardware=False,
    )
    assert config.position_command_planner == POSITION_COMMAND_PLANNER_LINEAR


def test_rm65_singularity_analysis_exposes_lock_and_sdk_timing() -> None:
    class AnalyzingArm(_FakeArm):
        def Algo_Kin_Singularity_Analyse(self, _joint_deg):
            return 0, 0.2

    backend, _arm = _backend()
    backend.arm = AnalyzingArm()

    result = backend._analyze_singularity(np.zeros(6))

    timing = result["analysis_timing"]
    assert result["status_code"] == 0
    assert timing["lock_wait_ms"] >= 0.0
    assert timing["sdk_call_ms"] >= 0.0
    assert timing["total_ms"] >= timing["sdk_call_ms"]


def test_latest_target_admission_does_not_wait_for_inflight_sdk_send_lock() -> None:
    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    )
    completed = threading.Event()
    receipts = []
    errors: list[BaseException] = []

    def admit_target() -> None:
        try:
            receipts.append(backend.send_command(_command(0.05)))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            completed.set()

    with backend._send_lock:
        thread = threading.Thread(target=admit_target, daemon=True)
        thread.start()
        assert completed.wait(0.25), "latest target admission waited for _send_lock"
    thread.join(timeout=0.25)

    assert not errors
    assert len(receipts) == 1
    timing = receipts[0].metadata["target_admission_timing"]
    assert timing["mode"] == RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
    assert timing["send_lock_wait_ms"] == 0.0
    phase_timing = receipts[0].metadata["command_phase_timing"]
    assert phase_timing["entry_state_lock_wait_ms"] >= 0.0
    assert phase_timing["singularity_protection_ms"] >= 0.0
    assert phase_timing["singularity_analysis_lock_wait_ms"] == 0.0
    assert phase_timing["singularity_analysis_sdk_call_ms"] == 0.0
    assert phase_timing["target_admission_ms"] == pytest.approx(
        timing["total_ms"]
    )
    assert phase_timing["total_through_target_admission_ms"] >= timing["total_ms"]
    assert phase_timing["post_admission_metadata_ms"] >= 0.0
    assert phase_timing["final_state_lock_wait_ms"] >= 0.0
    assert phase_timing["final_state_snapshot_ms"] >= 0.0
    assert phase_timing["quick_stop_metadata_ms"] >= 0.0
    assert phase_timing["singularity_transition_trace_ms"] >= 0.0
    assert phase_timing["total_send_command_ms"] >= phase_timing[
        "total_through_target_admission_ms"
    ]
    assert backend.get_state()["background_streaming"][
        "last_target_admission_timing"
    ]["mode"] == RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING


@pytest.mark.parametrize(
    "source",
    ("replay_qpos_init", "replay_qpos_init_final_hold"),
)
def test_qpos_init_serializes_target_admission_for_velocity_continuity(
    source: str,
) -> None:
    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    )
    completed = threading.Event()
    receipts = []
    errors: list[BaseException] = []

    def admit_target() -> None:
        try:
            receipts.append(backend.send_command(_command(0.05, source=source)))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            completed.set()

    with backend._send_lock:
        thread = threading.Thread(target=admit_target, daemon=True)
        thread.start()
        assert not completed.wait(0.03)
    assert completed.wait(0.25)
    thread.join(timeout=0.25)

    assert not errors
    assert len(receipts) == 1
    timing = receipts[0].metadata["target_admission_timing"]
    assert timing["requested_mode"] == (
        RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
    )
    assert timing["mode"] == RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
    assert timing["fallback_reason"] == (
        "qpos_init_requires_serialized_velocity_continuity"
    )
    assert timing["send_lock_wait_ms"] >= 20.0


def test_openpi_policy_keeps_latest_target_admission_nonblocking() -> None:
    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    )
    completed = threading.Event()
    receipts = []
    errors: list[BaseException] = []

    def admit_target() -> None:
        try:
            receipts.append(
                backend.send_command(
                    _command(0.05, source="openpi_v22_policy")
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            completed.set()

    with backend._send_lock:
        thread = threading.Thread(target=admit_target, daemon=True)
        thread.start()
        assert completed.wait(0.25), "OpenPI reference admission waited for _send_lock"
    thread.join(timeout=0.25)

    assert not errors
    assert len(receipts) == 1
    timing = receipts[0].metadata["target_admission_timing"]
    assert timing["requested_mode"] == (
        RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
    )
    assert timing["mode"] == (
        RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
    )
    assert timing["fallback_reason"] is None
    assert timing["send_lock_wait_ms"] == 0.0


def test_synchronous_target_admission_remains_explicit_rollback() -> None:
    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK,
    )
    completed = threading.Event()
    receipts = []
    errors: list[BaseException] = []

    def admit_target() -> None:
        try:
            receipts.append(backend.send_command(_command(0.05)))
        except BaseException as exc:  # pragma: no cover - assertion reports it
            errors.append(exc)
        finally:
            completed.set()

    with backend._send_lock:
        thread = threading.Thread(target=admit_target, daemon=True)
        thread.start()
        assert not completed.wait(0.03)
    assert completed.wait(0.25)
    thread.join(timeout=0.25)

    assert not errors
    assert len(receipts) == 1
    timing = receipts[0].metadata["target_admission_timing"]
    assert timing["mode"] == RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
    assert timing["send_lock_wait_ms"] >= 20.0


def test_inflight_sdk_completion_activates_latest_target_without_velocity_reset() -> None:
    class BlockingFirstSendArm(_FakeArm):
        def __init__(self) -> None:
            super().__init__()
            self.first_send_entered = threading.Event()
            self.release_first_send = threading.Event()

        def Movej_CANFD(self, joint_deg, follow=False, trajectory_mode=0, radio=0):
            if not self.first_send_entered.is_set():
                self.first_send_entered.set()
                if not self.release_first_send.wait(0.5):
                    raise TimeoutError("unit test did not release first SDK send")
            return super().Movej_CANFD(
                joint_deg,
                follow=follow,
                trajectory_mode=trajectory_mode,
                radio=radio,
            )

    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    )
    blocking_arm = BlockingFirstSendArm()
    backend.arm = blocking_arm
    first_receipt = backend.send_command(_command(0.04))
    backend._stream_thread = threading.Thread(
        target=backend._background_stream_loop,
        daemon=True,
    )
    backend._stream_thread.start()
    try:
        assert blocking_arm.first_send_entered.wait(0.25)
        admitted_started = time.perf_counter()
        deferred_receipt = backend.send_command(_command(0.08))
        admission_ms = (time.perf_counter() - admitted_started) * 1000.0
        assert admission_ms < 20.0
        assert deferred_receipt.metadata["target_admission_timing"][
            "target_activation_deferred"
        ] is True
        latest_receipt = backend.send_command(_command(0.12))
        assert latest_receipt.metadata["target_admission_timing"][
            "target_activation_deferred"
        ] is True
        assert deferred_receipt.metadata["transport_target_sequence"] < (
            latest_receipt.metadata["transport_target_sequence"]
        )
        inflight = backend.get_state()["background_streaming"]
        assert inflight["target_sequence"] == first_receipt.metadata[
            "transport_target_sequence"
        ]
        assert inflight["deferred_target_pending"] is True
        assert inflight["deferred_target_sequence"] == latest_receipt.metadata[
            "transport_target_sequence"
        ]
        assert inflight["deferred_target_replacement_count"] == 1
        blocking_arm.release_first_send.set()

        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            stream = backend.get_state()["background_streaming"]
            if (
                stream["deferred_target_activation_count"] >= 1
                and len(blocking_arm.sent_joint_deg) >= 2
            ):
                break
            time.sleep(0.001)
        stream = backend.get_state()["background_streaming"]
        assert stream["fault"] is None
        assert stream["inflight_target_preemption_count"] == 0
        assert stream["deferred_target_activation_count"] == 1
        assert stream["deferred_target_replacement_count"] == 1
        assert stream["deferred_target_pending"] is False
        assert stream["last_deferred_target_activation"][
            "completed_target_sequence"
        ] < stream["last_deferred_target_activation"][
            "activated_target_sequence"
        ]
        assert stream["last_deferred_target_activation"][
            "planner_velocity_preserved"
        ] is True
        assert stream["last_deferred_target_activation"][
            "planner_rebased_from_zero_velocity"
        ] is False
        assert stream["target_sequence"] == latest_receipt.metadata[
            "transport_target_sequence"
        ]
        assert len(blocking_arm.sent_joint_deg) >= 2
    finally:
        blocking_arm.release_first_send.set()
        backend.close()


def test_deferred_target_published_after_sdk_completion_uses_publication_as_activation_cut() -> None:
    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    )
    first_receipt = backend.send_command(_command(0.04))
    with backend._state_lock:
        backend._stream_sdk_inflight_target_sequence = first_receipt.metadata[
            "transport_target_sequence"
        ]

    deferred_receipt = backend.send_command(_command(0.08))
    assert deferred_receipt.metadata["target_admission_timing"][
        "target_activation_deferred"
    ] is True
    with backend._state_lock:
        assert backend._stream_deferred_target is not None
        published = backend._stream_deferred_target.timestamp_monotonic
        activated = backend._activate_deferred_acceleration_target_locked(
            accepted_qpos=np.zeros(6),
            activated_monotonic_time=published - 0.001,
            completed_target_sequence=first_receipt.metadata[
                "transport_target_sequence"
            ],
        )

    assert activated is True
    stream = backend.get_state()["background_streaming"]
    activation = stream["last_deferred_target_activation"]
    assert activation["published_monotonic_time"] == pytest.approx(published)
    assert activation["activated_monotonic_time"] == pytest.approx(published)
    assert activation["publication_after_sdk_completion_sec"] == pytest.approx(
        0.001
    )
    assert activation["planner_velocity_preserved"] is True
    assert stream["fault"] is None


def test_inflight_quick_stop_discards_deferred_tracking_target() -> None:
    backend, _arm = _backend(
        target_admission_mode=RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING,
    )
    first_receipt = backend.send_command(_command(0.04))
    with backend._state_lock:
        backend._stream_sdk_inflight_target_sequence = first_receipt.metadata[
            "transport_target_sequence"
        ]

    deferred_receipt = backend.send_command(_command(0.08))
    assert deferred_receipt.metadata["target_admission_timing"][
        "target_activation_deferred"
    ] is True

    hold_receipt = backend.send_command(
        _command(0.20, collision_status="hold_last_safe")
    )
    stream = backend.get_state()["background_streaming"]

    assert stream["quick_stop_active"] is True
    assert stream["deferred_target_pending"] is False
    assert stream["target_sequence"] == hold_receipt.metadata[
        "transport_target_sequence"
    ]
    assert stream["target_sequence"] > deferred_receipt.metadata[
        "transport_target_sequence"
    ]


def test_acceleration_planner_defaults_to_30ms_target_velocity_filter() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )

    assert backend._stream_planner.target_velocity_filter_tau_sec == pytest.approx(0.03)


def test_acceleration_planner_target_velocity_filter_can_be_bypassed() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            position_target_velocity_filter_tau_sec=0.0,
        )
    )

    assert backend._stream_planner.target_velocity_filter_tau_sec == pytest.approx(0.0)


def test_final_follow_qpos_alpha_filters_from_last_accepted_command() -> None:
    output, applied = _final_follow_qpos_target(
        np.ones(6),
        np.zeros(6),
        alpha=0.1,
        bypass=False,
    )

    assert applied is True
    assert np.allclose(output, 0.1)


def test_final_follow_qpos_alpha_is_bypassed_for_quick_stop() -> None:
    planned = np.full(6, 0.25)
    output, applied = _final_follow_qpos_target(
        planned,
        np.zeros(6),
        alpha=0.1,
        bypass=True,
    )

    assert applied is False
    assert np.allclose(output, planned)


def test_acceleration_mode_uses_explicit_acceleration_limit_without_speed_scaling() -> None:
    acceleration = (8.0, 8.0, 10.0, 16.0, 16.0, 21.0)
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            control_dt=1.0 / 75.0,
            max_angvel=(0.7, 0.7, 0.9, 1.5, 1.5, 2.0),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            position_velocity_ramp_sec=0.5,
            position_max_acceleration_rad_s2=acceleration,
            position_degraded_speed_scale=0.5,
        )
    )

    assert np.allclose(
        backend._stream_planner.limits.max_acceleration_rad_s2,
        acceleration,
    )


def test_acceleration_mode_uses_independent_uniform_command_step_envelope() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            control_dt=1.0 / 60.0,
            max_angvel=(5.6, 5.6, 7.2, 12.0, 12.0, 16.0),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
            position_max_command_step_deg=0.675,
        )
    )

    expected = np.full(6, np.deg2rad(0.675))
    np.testing.assert_allclose(
        backend._stream_planner.limits.max_command_step_rad,
        expected,
    )

    # Changing the reference/control dt must not silently rescale an explicit
    # high-rate planner envelope.
    backend.set_command_control_dt(1.0 / 120.0)
    np.testing.assert_allclose(
        backend._stream_planner.limits.max_command_step_rad,
        expected,
    )


def test_sample_aligned_backend_stamps_admitted_endpoint_and_holds_inner_frames() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            control_dt=1.0 / 60.0,
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            background_streaming_target_admission_mode=(
                RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
            ),
            position_command_planner=(
                POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
            ),
            position_sample_aligned_source_hz=15.0,
            position_velocity_ramp_sec=0.075,
            position_max_acceleration_rad_s2=(
                56.0,
                56.0,
                72.0,
                120.0,
                120.0,
                160.0,
            ),
            position_max_command_step_deg=0.675,
            arm_follow_qpos_alpha=1.0,
            singularity_protection_enabled=False,
        )
    )
    backend.arm = _FakeArm()

    first = _command(1.0)
    first.metadata["sample_aligned_source_timestamp"] = 100.0
    first_receipt = backend.send_command(first)
    first_admission_timing = first_receipt.metadata["target_admission_timing"]
    assert first_admission_timing["requested_mode"] == (
        RM65_BACKGROUND_TARGET_ADMISSION_LATEST_NONBLOCKING
    )
    assert first_admission_timing["mode"] == (
        RM65_BACKGROUND_TARGET_ADMISSION_SYNCHRONOUS_LOCK
    )
    assert first_admission_timing["fallback_reason"] == (
        "sample_aligned_jerk_limited_planner_requires_synchronous_activation"
    )
    admitted = np.asarray(first.metadata["backend_final_target_qpos"], dtype=float)
    assert np.max(admitted) < 0.1
    assert first.metadata["sample_aligned_admission"]["admission_limited"] is True
    assert first.metadata["position_transport_sample_aligned_bucket_held"] is False

    intermediate = _command(-1.0)
    intermediate.metadata["sample_aligned_source_timestamp"] = 100.0 + 1.0 / 60.0
    backend.send_command(intermediate)
    np.testing.assert_allclose(
        intermediate.metadata["backend_final_target_qpos"],
        admitted,
    )
    assert intermediate.metadata["position_transport_sample_aligned_bucket_held"] is True

    next_cut = _command(0.5)
    next_cut.metadata["sample_aligned_source_timestamp"] = 100.0 + 4.0 / 60.0
    backend.send_command(next_cut)
    assert next_cut.metadata["position_transport_sample_aligned_bucket_held"] is False


def test_sample_aligned_backend_builds_causal_observer_planner() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            position_command_planner=(
                POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
            ),
            position_sample_aligned_source_hz=15.0,
            position_sample_aligned_initial_jerk_delta_regularization=1e-5,
            position_sample_aligned_reference_mode="causal_hermite",
            position_sample_aligned_observer_window=4,
            position_sample_aligned_observer_efficiency_power=2.0,
            position_sample_aligned_observer_velocity_ema_alpha=0.5,
            position_sample_aligned_observer_velocity_cap_scale=0.5,
            position_sample_aligned_observer_coherence_min=0.5,
            position_sample_aligned_observer_speed_guard_scale=0.4,
            position_sample_aligned_terminal_velocity_displacement_scale=0.75,
            position_max_command_step_deg=0.675,
            arm_follow_qpos_alpha=1.0,
            singularity_protection_enabled=False,
        )
    )

    snapshot = backend._stream_planner.snapshot()
    assert snapshot["reference_mode"] == "causal_hermite"
    assert snapshot["observer_window"] == 4
    assert snapshot["terminal_velocity_displacement_scale"] == pytest.approx(0.75)
    assert snapshot["governor_regularization"]["initial_jerk_delta"] == pytest.approx(
        1e-5
    )

def test_command_step_envelope_default_remains_velocity_times_control_dt() -> None:
    max_angvel = np.asarray((0.7, 0.7, 0.9, 1.5, 1.5, 2.0), dtype=float)
    control_dt = 1.0 / 75.0
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            control_dt=control_dt,
            max_angvel=tuple(max_angvel.tolist()),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )

    np.testing.assert_allclose(
        backend._stream_planner.limits.max_command_step_rad,
        max_angvel * control_dt,
    )


@pytest.mark.parametrize("value", [0.0, -0.1, float("nan"), float("inf")])
def test_command_step_envelope_rejects_non_positive_or_non_finite(value: float) -> None:
    with pytest.raises(ValueError, match="position_max_command_step_deg"):
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            position_max_command_step_deg=value,
        )


def test_command_step_envelope_rejects_linear_planner() -> None:
    with pytest.raises(ValueError, match="acceleration/jerk-limited"):
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            position_command_planner=POSITION_COMMAND_PLANNER_LINEAR,
            position_max_command_step_deg=0.675,
        )


def test_acceleration_mode_coupled_ik_sync_uses_accepted_qcmd_not_preview_qref() -> None:
    left_backend, _ = _backend()
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )
    left_qcmd = np.full(6, 0.1)
    right_qcmd = np.full(6, -0.2)
    left_backend.reset_motion_state(left_qcmd)
    right_backend.reset_motion_state(right_qcmd)
    left_preview = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.full(6, 0.7),
        )
    )
    right_preview = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.full(6, -0.7),
        )
    )
    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left_backend,
        right_command_backend=right_backend,
        left_preview_backend=left_preview,
        right_preview_backend=right_preview,
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
    )

    real_teleop._sync_dual_arm_retargeter_for_real_control(
        components,
        reset_tracking=False,
    )

    assert len(retargeter.calls) == 1
    call = retargeter.calls[0]
    np.testing.assert_allclose(call["left_qpos"], left_qcmd)
    np.testing.assert_allclose(call["right_qpos"], right_qcmd)
    assert "left_qvel" not in call
    assert "right_qvel" not in call
    assert call["reset_tracking"] is False
    sync = components.arm_loop_step_dt_tracker["retargeter_execution_state_sync"]
    assert sync["execution_qvel_feedback_enabled"] is False
    assert sync["source_by_side"] == {
        "left": "last_accepted_qpos",
        "right": "last_accepted_qpos",
    }
    assert sync["left_qvel_rad_s"] is None
    assert sync["right_qvel_rad_s"] is None


def test_acceleration_mode_coupled_ik_sync_can_use_coherent_planner_velocity() -> None:
    left_backend, _ = _backend()
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )
    expected_qpos = {}
    expected_qvel = {}
    for side, backend, sign in (
        ("left", left_backend, 1.0),
        ("right", right_backend, -1.0),
    ):
        planner = backend._stream_planner
        planner.reset(np.zeros(6), monotonic_time=10.0)
        planner.accept_target(
            np.full(6, sign * 0.2),
            sequence=1,
            published_monotonic_time=10.0,
            monotonic_time=10.0,
        )
        proposal = planner.propose(monotonic_time=10.0 + 1.0 / 120.0)
        planner.commit(proposal)
        expected_qpos[side] = proposal.q_command.copy()
        expected_qvel[side] = proposal.qd_command_rad_s.copy()
        with backend._state_lock:
            backend._record_command_accepted_locked(
                proposal.q_command,
                timestamp=20.0,
                monotonic_time=20.0,
            )

    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left_backend,
        right_command_backend=right_backend,
        left_preview_backend=left_backend,
        right_preview_backend=right_backend,
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
        position_execution_qvel_feedback_enabled=True,
    )

    real_teleop._sync_dual_arm_retargeter_for_real_control(
        components,
        reset_tracking=False,
    )

    call = retargeter.calls[0]
    for side in ("left", "right"):
        np.testing.assert_allclose(call[f"{side}_qpos"], expected_qpos[side])
        np.testing.assert_allclose(call[f"{side}_qvel"], expected_qvel[side])
    sync = components.arm_loop_step_dt_tracker["retargeter_execution_state_sync"]
    assert sync["execution_qvel_feedback_enabled"] is True
    assert sync["source_by_side"] == {
        "left": "last_accepted_qpos+trajectory_planner_motion",
        "right": "last_accepted_qpos+trajectory_planner_motion",
    }


def test_execution_qvel_feedback_fails_closed_without_planner_snapshot() -> None:
    left_backend, _ = _backend()
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=False,
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )
    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left_backend,
        right_command_backend=right_backend,
        left_preview_backend=left_backend,
        right_preview_backend=right_backend,
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
        position_execution_qvel_feedback_enabled=True,
    )

    with pytest.raises(
        RuntimeError,
        match="right_arm coupled IK qvel feedback requires background streaming",
    ):
        real_teleop._sync_dual_arm_retargeter_for_real_control(
            components,
            reset_tracking=False,
        )
    assert retargeter.calls == []


def test_execution_qvel_feedback_fails_closed_for_legacy_planner() -> None:
    backends = {
        side: RM65ArmBackend(
            RM65ArmBackendConfig(
                arm_ip="0.0.0.0",
                side=side,
                connect_hardware=False,
                initial_qpos=np.zeros(6),
                position_command_planner=POSITION_COMMAND_PLANNER_LINEAR,
            )
        )
        for side in ("left", "right")
    }
    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=backends["left"],
        right_command_backend=backends["right"],
        left_preview_backend=backends["left"],
        right_preview_backend=backends["right"],
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
        position_execution_qvel_feedback_enabled=True,
    )

    with pytest.raises(
        RuntimeError,
        match="qvel feedback requires RM65 acceleration-limited",
    ):
        real_teleop._sync_dual_arm_retargeter_for_real_control(
            components,
            reset_tracking=False,
        )
    assert retargeter.calls == []


def test_execution_qvel_feedback_rejects_mixed_execution_ticks(monkeypatch) -> None:
    left_backend, _ = _backend()
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )
    left_state = left_backend.get_state(
        include_derived_pose=False,
        include_timing_percentiles=False,
    )
    left_state["background_streaming"]["trajectory_planner"]["q_command"][0] = 0.1
    monkeypatch.setattr(left_backend, "get_state", lambda **_kwargs: left_state)
    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left_backend,
        right_command_backend=right_backend,
        left_preview_backend=left_backend,
        right_preview_backend=right_backend,
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
        position_execution_qvel_feedback_enabled=True,
    )

    with pytest.raises(RuntimeError, match="states from different execution ticks"):
        real_teleop._sync_dual_arm_retargeter_for_real_control(
            components,
            reset_tracking=False,
        )
    assert retargeter.calls == []


def test_sample_aligned_backend_exposes_coherent_execution_motion_snapshot() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            position_command_planner=(
                POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
            ),
            position_sample_aligned_source_hz=60.0,
            singularity_protection_enabled=False,
        )
    )

    state = backend.get_state(
        include_derived_pose=False,
        include_timing_percentiles=False,
    )
    planner = state["background_streaming"]["trajectory_planner"]
    np.testing.assert_allclose(planner["q_command"], state["last_accepted_qpos"])
    np.testing.assert_allclose(planner["qd_command_rad_s"], 0.0)


def test_acceleration_mode_coupled_ik_sync_skips_percentile_snapshot_hot_path(
    monkeypatch,
) -> None:
    left_backend, _ = _backend()
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            position_command_planner=POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
        )
    )
    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left_backend,
        right_command_backend=right_backend,
        left_preview_backend=left_backend,
        right_preview_backend=right_backend,
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
    )

    def fail_if_percentiles_are_built():
        raise AssertionError("coupled IK sync entered diagnostic percentile hot path")

    monkeypatch.setattr(
        left_backend,
        "_background_stream_timing_percentiles_snapshot",
        fail_if_percentiles_are_built,
    )
    monkeypatch.setattr(
        right_backend,
        "_background_stream_timing_percentiles_snapshot",
        fail_if_percentiles_are_built,
    )

    real_teleop._sync_dual_arm_retargeter_for_real_control(
        components,
        reset_tracking=False,
    )

    assert len(retargeter.calls) == 1


def test_linear_mode_coupled_ik_sync_retains_preview_behavior() -> None:
    left_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.full(6, 0.1),
        )
    )
    right_backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.full(6, -0.1),
        )
    )
    left_preview = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.full(6, 0.6),
        )
    )
    right_preview = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="right",
            connect_hardware=False,
            initial_qpos=np.full(6, -0.6),
        )
    )
    retargeter = _RecordingDualArmRetargeter()
    components = SimpleNamespace(
        active_sides=("left", "right"),
        left_command_backend=left_backend,
        right_command_backend=right_backend,
        left_preview_backend=left_preview,
        right_preview_backend=right_preview,
        dual_arm_retargeter=retargeter,
        arm_loop_step_dt_tracker={},
    )

    real_teleop._sync_dual_arm_retargeter_for_real_control(components)

    np.testing.assert_allclose(retargeter.calls[0]["left_qpos"], 0.6)
    np.testing.assert_allclose(retargeter.calls[0]["right_qpos"], -0.6)


def test_position_trajectory_policy_distinguishes_stop_degrade_and_normal() -> None:
    stop = _position_trajectory_policy(
        {"collision_avoidance_status": "hold_last_safe"},
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert stop.quick_stop is True
    assert stop.speed_scale == 0.0

    degraded = _position_trajectory_policy(
        {
            "collision_avoidance_status": "tracking",
            "execution_step_metadata": {"step_limited": True},
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert degraded.quick_stop is False
    assert degraded.mode == "degraded"
    assert degraded.speed_scale == pytest.approx(0.1)

    singularity = _position_trajectory_policy(
        {"collision_avoidance_status": "tracking"},
        {"singularity_gate_status": "hold_entry"},
        degraded_speed_scale=0.1,
    )
    assert singularity.quick_stop is False
    assert singularity.mode == "degraded"
    assert singularity.speed_scale == pytest.approx(0.1)

    accepted_wrist_fallback = _position_trajectory_policy(
        {"collision_avoidance_status": "tracking"},
        {"singularity_gate_status": "allow_position_wrist_singular"},
        degraded_speed_scale=0.1,
    )
    assert accepted_wrist_fallback.mode == "tracking"
    assert accepted_wrist_fallback.speed_scale == 1.0

    accepted_elbow_fallback = _position_trajectory_policy(
        {"collision_avoidance_status": "tracking"},
        {"singularity_gate_status": "allow_position_elbow_singular"},
        degraded_speed_scale=0.1,
    )
    assert accepted_elbow_fallback.mode == "tracking"
    assert accepted_elbow_fallback.speed_scale == 1.0

    accepted_shoulder_fallback = _position_trajectory_policy(
        {"collision_avoidance_status": "tracking"},
        {"singularity_gate_status": "allow_position_shoulder_singular"},
        degraded_speed_scale=0.1,
    )
    assert accepted_shoulder_fallback.mode == "tracking"
    assert accepted_shoulder_fallback.speed_scale == 1.0

    accepted_current_singular = _position_trajectory_policy(
        {"collision_avoidance_status": "tracking"},
        {"singularity_gate_status": "allow_current_singular"},
        degraded_speed_scale=0.1,
    )
    assert accepted_current_singular.mode == "tracking"
    assert accepted_current_singular.speed_scale == 1.0

    normal = _position_trajectory_policy(
        {"collision_avoidance_status": "tracking"},
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert normal.mode == "tracking"
    assert normal.speed_scale == 1.0


def test_real_agent_speed_scale_caps_final_trajectory_without_expanding_safety() -> None:
    requested = _position_trajectory_policy(
        {
            "collision_avoidance_status": "tracking",
            "real_agent_motion_operation": {"speed_scale": 0.35},
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert requested.quick_stop is False
    assert requested.mode == "tracking"
    assert requested.speed_scale == pytest.approx(0.35)
    assert requested.reasons == ("agent_speed_scale",)

    safety_degraded = _position_trajectory_policy(
        {
            "collision_avoidance_status": "tracking",
            "execution_step_metadata": {"step_limited": True},
            "real_agent_motion_operation": {"speed_scale": 0.8},
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert safety_degraded.speed_scale == pytest.approx(0.1)
    assert safety_degraded.reasons == (
        "execution_step_limited",
        "agent_speed_scale",
    )

    tighter_request = _position_trajectory_policy(
        {
            "collision_avoidance_status": "boundary_active",
            "real_agent_motion_operation": {"speed_scale": 0.05},
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert tighter_request.speed_scale == pytest.approx(0.05)


@pytest.mark.parametrize("speed_scale", [True, 0.0, 1.01, float("nan"), "bad"])
def test_invalid_real_agent_speed_scale_fails_closed(speed_scale) -> None:
    policy = _position_trajectory_policy(
        {
            "collision_avoidance_status": "tracking",
            "real_agent_motion_operation": {"speed_scale": speed_scale},
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )
    assert policy.quick_stop is True
    assert policy.speed_scale == 0.0
    assert policy.reason == "agent_speed_scale_invalid"


def test_real_agent_speed_scale_reaches_acceleration_limited_planner() -> None:
    backend, _arm = _backend()
    command = _command(0.2)
    command.metadata["real_agent_motion_operation"] = {"speed_scale": 0.25}

    receipt = backend.send_command(command)
    stream = backend.get_state()["background_streaming"]

    assert receipt.accepted is True
    assert stream["trajectory_policy_mode"] == "tracking"
    assert stream["trajectory_policy_speed_scale"] == pytest.approx(0.25)
    assert stream["trajectory_planner"]["speed_scale"] == pytest.approx(0.25)
    assert command.metadata["position_trajectory_policy_speed_scale"] == pytest.approx(
        0.25
    )


def test_finite_ik_pose_residual_keeps_full_position_tracking_speed() -> None:
    policy = _position_trajectory_policy(
        {
            "collision_avoidance_status": "tracking",
            "ik_solve_pose_error": {
                "translation_m": 0.03,
                "rotation_deg": 8.0,
            },
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )

    assert policy.mode == "tracking"
    assert policy.speed_scale == 1.0
    assert policy.quick_stop is False
    assert policy.reason == "normal"


def test_invalid_ik_pose_residual_remains_fail_closed() -> None:
    policy = _position_trajectory_policy(
        {
            "collision_avoidance_status": "tracking",
            "ik_solve_pose_error": {
                "translation_m": float("nan"),
                "rotation_deg": 0.0,
            },
        },
        {"singularity_gate_status": "pass"},
        degraded_speed_scale=0.1,
    )

    assert policy.mode == "quick_stop"
    assert policy.quick_stop is True
    assert policy.speed_scale == 0.0
    assert policy.reason == "ik_pose_error_nonfinite"


def test_collision_hold_queues_exact_current_qpos_quick_stop() -> None:
    backend, arm = _backend()
    command = _command(0.2, collision_status="hold_last_safe")

    receipt = backend.send_command(command)
    stream = backend.get_state()["background_streaming"]

    assert receipt.accepted is True
    assert stream["quick_stop_active"] is True
    assert stream["continuous_hold"] is True
    assert stream["trajectory_policy_mode"] == "quick_stop"
    np.testing.assert_allclose(backend._stream_target_qpos, 0.0)
    np.testing.assert_allclose(command.value, 0.0)
    np.testing.assert_allclose(command.metadata["backend_final_target_qpos"], 0.0)
    assert arm.sent_joint_deg == []


def test_execution_step_warning_queues_ten_percent_speed_target() -> None:
    backend, _arm = _backend()
    command = _command(0.05, step_limited=True)

    backend.send_command(command)
    stream = backend.get_state()["background_streaming"]

    assert stream["quick_stop_active"] is False
    assert stream["trajectory_policy_mode"] == "degraded"
    assert stream["trajectory_policy_speed_scale"] == pytest.approx(0.1)
    assert stream["trajectory_planner"]["speed_scale"] == pytest.approx(0.1)


def test_manual_hold_uses_software_quick_stop_without_sdk_slow_stop() -> None:
    backend, arm = _backend()
    release = threading.Event()
    backend._stream_thread = threading.Thread(target=release.wait, daemon=True)
    backend._stream_thread.start()

    backend.hold_motion("tracker timeout")
    stream = backend.get_state()["background_streaming"]

    assert stream["quick_stop_active"] is True
    assert "manual_hold:tracker timeout" in stream["quick_stop_reason"]
    assert stream["target_active"] is True
    assert arm.slow_stop_calls == 0

    release.set()
    backend._stream_thread.join(timeout=0.2)
    backend._stream_thread = None
    backend.close()


def test_stale_target_becomes_persistent_quick_stop_in_acceleration_mode() -> None:
    backend, _arm = _backend(target_max_age_sec=0.1)
    backend.send_command(_command(0.05))
    now = time.monotonic()
    with backend._state_lock:
        backend._stream_target_monotonic = now - 0.2

    assert backend._expire_stale_stream_target(now_monotonic=now) is True
    stream = backend.get_state()["background_streaming"]

    assert stream["target_active"] is True
    assert stream["quick_stop_active"] is True
    assert stream["target_stale"] is False
    assert stream["stale_count"] == 1


def test_stale_sample_aligned_target_reanchors_command_space_hold() -> None:
    backend = RM65ArmBackend(
        RM65ArmBackendConfig(
            arm_ip="0.0.0.0",
            side="left",
            connect_hardware=False,
            initial_qpos=np.zeros(6),
            background_streaming_enabled=True,
            background_streaming_hz=120.0,
            background_streaming_target_max_age_sec=0.1,
            position_command_planner=(
                POSITION_COMMAND_PLANNER_SAMPLE_ALIGNED_JERK_LIMITED
            ),
            position_sample_aligned_source_hz=60.0,
            position_sample_aligned_horizon_source_steps=2,
            position_sample_aligned_safety_horizon_source_steps=20,
            position_velocity_ramp_sec=0.05,
            position_max_acceleration_rad_s2=(
                56.0,
                56.0,
                72.0,
                120.0,
                120.0,
                160.0,
            ),
            position_max_command_step_deg=0.675,
            singularity_protection_enabled=False,
        )
    )
    backend.arm = _FakeArm()
    command = _command(0.2)
    command.metadata["sample_aligned_source_timestamp"] = 100.0
    backend.send_command(command)

    with backend._state_lock:
        proposal = backend._stream_planner.propose(monotonic_time=100.0)
        backend._stream_planner.commit(proposal)
        accepted = np.asarray(proposal.q_command, dtype=float)
        backend._record_command_accepted_locked(
            accepted,
            timestamp=time.time(),
            monotonic_time=100.0,
        )
        backend._stream_target_monotonic = 99.8
    before = backend._stream_planner.snapshot()
    assert np.max(np.abs(before["qd_command_rad_s"])) > 0.0

    assert backend._expire_stale_stream_target(now_monotonic=100.0) is True
    after = backend._stream_planner.snapshot()
    np.testing.assert_allclose(after["q_command"], accepted)
    np.testing.assert_allclose(after["qd_command_rad_s"], 0.0)
    np.testing.assert_allclose(after["qdd_command_rad_s2"], 0.0)
    assert backend.get_state()["background_streaming"]["target_active"] is False


def test_failed_sdk_send_does_not_commit_online_trajectory_state() -> None:
    backend, arm = _backend(result=7)
    audit_events = []
    backend.set_position_audit_sink(audit_events.append)
    backend.send_command(_command(0.05))

    backend._background_stream_loop()
    state = backend.get_state()

    assert state["background_streaming"]["fault"] is not None
    assert state["background_streaming"]["fault_stop_attempted"] is True
    assert state["background_streaming"]["fault_stop_success"] is True
    assert arm.slow_stop_calls == 1
    np.testing.assert_allclose(
        state["background_streaming"]["trajectory_planner"]["q_command"],
        0.0,
    )
    np.testing.assert_allclose(state["last_accepted_qpos"], 0.0)
    command_events = [
        event for event in audit_events if getattr(event, "event_type", "") == "position_command_150"
    ]
    assert len(command_events) == 1
    metadata = command_events[0].metadata
    assert metadata["position_command_planner"] == POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED
    assert len(metadata["trajectory_qd_reference_rad_s"]) == 6
    assert len(metadata["trajectory_qd_command_rad_s"]) == 6
    assert len(metadata["trajectory_qdd_command_rad_s2"]) == 6
    assert "trajectory_position_error_rad" in metadata
    assert metadata["trajectory_target_preempted"] is True
    assert metadata["trajectory_proposal_token"] >= 0


def test_successful_stream_continues_fixed_qpos_ticks_after_manual_hold() -> None:
    backend, arm = _backend()
    backend._stream_thread = threading.Thread(
        target=backend._background_stream_loop,
        daemon=True,
    )
    backend._stream_thread.start()
    backend.send_command(_command(0.05))
    deadline = time.monotonic() + 0.3
    while len(arm.sent_joint_deg) < 4 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(arm.sent_joint_deg) >= 4

    backend.hold_motion("operator hold")
    hold_count = len(arm.sent_joint_deg)
    deadline = time.monotonic() + 0.2
    while len(arm.sent_joint_deg) < hold_count + 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(arm.sent_joint_deg) >= hold_count + 3
    held = np.asarray(arm.sent_joint_deg[-3:])
    np.testing.assert_allclose(
        held,
        np.repeat(held[-1][None, :], held.shape[0], axis=0),
        atol=1e-10,
    )
    assert arm.slow_stop_calls == 0

    backend.close()


def test_disengage_position_stream_stops_once_and_withdraws_hold_target() -> None:
    backend, arm = _backend()
    backend._stream_thread = threading.Thread(
        target=backend._background_stream_loop,
        daemon=True,
    )
    backend._stream_thread.start()
    backend.send_command(_command(0.05))
    deadline = time.monotonic() + 0.3
    while len(arm.sent_joint_deg) < 4 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(arm.sent_joint_deg) >= 4

    backend.hold_motion("operator hold")
    backend.disengage_position_command_stream("operator manual interaction")
    send_count_after_disengage = len(arm.sent_joint_deg)
    time.sleep(0.03)

    stream = backend.get_state()["background_streaming"]
    assert stream["target_active"] is False
    assert stream["quick_stop_active"] is False
    assert stream["continuous_hold"] is False
    assert len(arm.sent_joint_deg) == send_count_after_disengage
    assert arm.slow_stop_calls == 1

    backend.close()

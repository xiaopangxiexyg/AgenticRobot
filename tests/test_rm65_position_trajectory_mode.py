from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from robot_motion.position_control import (
    POSITION_COMMAND_PLANNER_ACCELERATION_LIMITED,
    POSITION_COMMAND_PLANNER_LINEAR,
)
from teleop_runtime.backends.arms.rm65 import (
    RM65ArmBackend,
    RM65ArmBackendConfig,
    _final_follow_qpos_target,
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


def _backend(*, result: int = 0, target_max_age_sec: float = 1.0) -> tuple[RM65ArmBackend, _FakeArm]:
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
) -> EffectorCommand:
    return EffectorCommand(
        effector_name="left_arm",
        side="left",
        command_type="joint_vector",
        value=np.full(6, target),
        metadata={
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
        },
    )


def test_rm65_position_planner_default_remains_linear() -> None:
    config = RM65ArmBackendConfig(
        arm_ip="0.0.0.0",
        side="left",
        connect_hardware=False,
    )
    assert config.position_command_planner == POSITION_COMMAND_PLANNER_LINEAR


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
    with pytest.raises(ValueError, match="acceleration_limited"):
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
    assert call["reset_tracking"] is False
    sync = components.arm_loop_step_dt_tracker["retargeter_execution_state_sync"]
    assert sync["source_by_side"] == {
        "left": "last_accepted_qpos",
        "right": "last_accepted_qpos",
    }


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

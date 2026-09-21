from __future__ import annotations

import pytest

from robot_motion.rebot_b601_rs import (
    B601MitControlProfile,
    B601PolicyAssistCommand,
    B601SharedControlConfig,
    B601SharedControlMode,
    B601SharedControlRuntime,
    compute_b601_mit_torque,
)

ZERO = (0.0,) * 6


def _config() -> B601SharedControlConfig:
    return B601SharedControlConfig(
        position_kp_nm_per_rad=(2.0,) * 6,
        velocity_kd_nm_s_per_rad=(0.0,) * 6,
        max_target_error_rad=(0.1,) * 6,
        max_abs_assist_torque_nm=(0.2,) * 6,
        max_assist_torque_rate_nm_s=(1.0,) * 6,
        authority_rise_per_sec=10.0,
        policy_stale_timeout_sec=0.1,
    )


def _command(sequence: int, timestamp: float) -> B601PolicyAssistCommand:
    return B601PolicyAssistCommand(
        sequence=sequence,
        source_timestamp_sec=timestamp,
        receive_timestamp_sec=timestamp,
        qpos_target_rad=(0.2,) * 6,
    )


def test_runtime_emits_current_position_reference_with_bounded_overlay() -> None:
    runtime = B601SharedControlRuntime(_config())
    assert runtime.publish_policy(_command(1, 1.0)).accepted is True

    frame = runtime.step(
        control_sequence=5,
        control_timestamp_sec=1.01,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.02,
    )

    assert frame.reference.sequence == 5
    assert frame.reference.qpos_rad == ZERO
    assert frame.reference.qvel_rad_s == ZERO
    assert frame.reference.joint_torque_overlay_nm == pytest.approx((0.02,) * 6)
    assert frame.decision.state.mode is B601SharedControlMode.POLICY_ASSIST


def test_runtime_clutch_removes_overlay_without_changing_measured_reference() -> None:
    runtime = B601SharedControlRuntime(_config())
    runtime.publish_policy(_command(1, 1.0))
    runtime.step(
        control_sequence=1,
        control_timestamp_sec=1.01,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.02,
    )
    runtime.set_policy_clutch(True)

    qpos = (0.3,) * 6
    frame = runtime.step(
        control_sequence=2,
        control_timestamp_sec=1.02,
        measured_qpos_rad=qpos,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.01,
    )

    assert frame.reference.qpos_rad == qpos
    assert frame.reference.joint_torque_overlay_nm == ZERO
    assert frame.decision.state.mode is B601SharedControlMode.HUMAN_OVERRIDE


def test_runtime_stale_policy_emits_zero_overlay() -> None:
    runtime = B601SharedControlRuntime(_config())
    runtime.publish_policy(_command(1, 1.0))

    frame = runtime.step(
        control_sequence=1,
        control_timestamp_sec=1.2,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.01,
    )

    assert frame.reference.joint_torque_overlay_nm == ZERO
    assert frame.decision.state.mode is B601SharedControlMode.PASSIVE
    assert frame.decision.reason == "policy_command_stale"


def test_runtime_clear_policy_withdraws_overlay_and_preserves_replay_guard() -> None:
    runtime = B601SharedControlRuntime(_config())
    assert runtime.publish_policy(_command(1, 1.0)).accepted is True
    assert runtime.clear_policy() == 1

    frame = runtime.step(
        control_sequence=1,
        control_timestamp_sec=1.01,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.01,
    )
    assert frame.reference.joint_torque_overlay_nm == ZERO
    assert frame.decision.state.mode is B601SharedControlMode.PASSIVE
    assert frame.decision.reason == "no_policy_command"
    assert runtime.publish_policy(_command(1, 1.02)).reason == (
        "non_advancing_sequence"
    )


def test_runtime_reference_reuses_existing_gravity_and_total_torque_path() -> None:
    runtime = B601SharedControlRuntime(_config())
    runtime.publish_policy(_command(1, 1.0))
    frame = runtime.step(
        control_sequence=1,
        control_timestamp_sec=1.01,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.1,
    )
    profile = B601MitControlProfile(
        name="shared-control-test",
        position_kp_nm_per_rad=ZERO,
        damping_kd_nm_s_per_rad=ZERO,
        max_abs_torque_nm=(2.0,) * 6,
        max_torque_rate_nm_s=(10.0,) * 6,
        max_abs_overlay_torque_nm=(0.2,) * 6,
    )
    breakdown = compute_b601_mit_torque(
        frame.reference,
        profile,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        gravity_torque_nm=(0.5,) * 6,
    )

    assert breakdown.position_torque_nm == ZERO
    assert breakdown.damping_torque_nm == ZERO
    assert breakdown.overlay_effective_nm == pytest.approx((0.1,) * 6)
    assert breakdown.requested_torque_nm == pytest.approx((0.6,) * 6)

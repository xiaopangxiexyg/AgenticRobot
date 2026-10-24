from __future__ import annotations

from dataclasses import replace

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


@pytest.mark.parametrize("now,accepted,reason", [
    (.9, False, "policy_command_from_future"),
    (1.01, True, "accepted"),
    (1.11, False, "policy_command_stale"),
])
def test_read_only_policy_sampling_reuses_validity_without_advancing_state(now, accepted, reason):
    runtime = B601SharedControlRuntime(_config())
    assert runtime.publish_policy(_command(1, 1.0)).accepted
    state = runtime.state
    for _ in range(3):
        sample = runtime.sample_policy(now_sec=now)
        assert sample.accepted is accepted
        assert sample.reason == reason
        assert runtime.state is state
    assert not runtime.publish_policy(_command(1, 1.0)).accepted


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


def test_damping_velocity_uses_joint_receive_time_and_not_snapshot_jitter():
    config = replace(_config(), velocity_kd_nm_s_per_rad=(0.5,) * 6, damping_velocity_window_sec=0.015)
    runtime = B601SharedControlRuntime(config)
    baseline = B601SharedControlRuntime(replace(config, damping_velocity_window_sec=0.0))
    for index in range(8):
        # Six independent clocks/ages, linearly increasing positions. The raw
        # snapshot difference may be twice or half the actual receive slope.
        joint_times = tuple(1 + index * 0.01 - j * 0.0003 for j in range(6))
        now = 1 + index * 0.01 + (0.001 if index % 2 else 0.006)
        runtime.publish_policy(_command(index, now))
        baseline.publish_policy(_command(index, now))
        q = tuple(0.2 * (t - 1) for t in joint_times)
        frame = runtime.step(
            control_sequence=index, control_timestamp_sec=now,
            measured_qpos_rad=q, measured_qvel_rad_s=(0.4 if index % 2 else 0.1,) * 6,
            dt_sec=0.01, joint_observation_timestamps_sec=joint_times,
        )
        expected = baseline.step(
            control_sequence=index, control_timestamp_sec=now,
            measured_qpos_rad=q, measured_qvel_rad_s=(0.2 if index else 0.1,) * 6,
            dt_sec=0.01,
        )
        assert frame.decision.assist_torque_nm == pytest.approx(expected.decision.assist_torque_nm)
        assert frame.decision.reason == expected.decision.reason
        if index > 0:
            assert frame.control_qvel_rad_s == pytest.approx((0.2,) * 6)
        assert frame.reference.receive_timestamp_sec == now
        assert frame.reference.qpos_rad == q



def test_joint_velocity_reuse_gap_and_bad_metadata_never_invent_a_sample():
    runtime = B601SharedControlRuntime(replace(_config(), damping_velocity_window_sec=0.015))

    def tick(sequence, now, observed, q=ZERO, raw=ZERO):
        return runtime.step(
            control_sequence=sequence, control_timestamp_sec=now,
            measured_qpos_rad=q, measured_qvel_rad_s=raw, dt_sec=0.01,
            joint_observation_timestamps_sec=observed,
        )

    tick(0, 1.001, (1,) * 6)
    frame = tick(1, 1.011, (1.01,) * 6, (0.002,) * 6)
    assert frame.decision.assist_torque_nm == ZERO  # No policy still fails passive.
    # Reuse holds the prior estimate, without turning the unchanged position
    # into a fabricated zero-velocity observation at a later poll time.
    tick(2, 1.012, (1.01,) * 6, (0.002,) * 6)
    before = runtime.state
    for observed, q in [((1.009,) * 6, (0.002,) * 6), ((1.01,) * 6, ZERO),
                        ((1.02,) * 5 + (float("nan"),), ZERO), ((0.5,) * 6, ZERO),
                        ((1.04,) * 5 + (2,), ZERO), (None, ZERO)]:
        with pytest.raises((ValueError, TypeError)):
            tick(3, 1.021, observed, q)
        assert runtime.state is before
    runtime.publish_policy(_command(1, 1.021))
    recovered = tick(3, 1.021, (1.02,) * 6, q=(0.004,) * 6)
    assert recovered.control_qvel_rad_s == pytest.approx((0.2,) * 6)
    # A long genuine gap resets the window. The first sample uses explicit
    # raw feedback as fallback, not the old slope or invented stationary data.
    runtime.publish_policy(_command(2, 1.2))
    frame = tick(4, 1.2, (1.199,) * 6, raw=(0.03,) * 6)
    assert frame.control_qvel_rad_s == (0.03,) * 6



def test_report_interval_beyond_short_window_uses_joint_interval_not_snapshot_fallback():
    runtime = B601SharedControlRuntime(replace(_config(), damping_velocity_window_sec=0.015))
    runtime.step(control_sequence=0, control_timestamp_sec=1,
                 measured_qpos_rad=ZERO, measured_qvel_rad_s=ZERO, dt_sec=0.01,
                 joint_observation_timestamps_sec=(1,) * 6)
    runtime.publish_policy(_command(1, 1.021))
    frame = runtime.step(control_sequence=1, control_timestamp_sec=1.021,
                         measured_qpos_rad=(0.004,) * 6,
                         measured_qvel_rad_s=(0.4,) * 6, dt_sec=0.01,
                         joint_observation_timestamps_sec=(1.02,) * 6)
    assert frame.control_qvel_rad_s == pytest.approx((0.2,) * 6)



@pytest.mark.parametrize("stop", ["clutch", "override", "stale"])
def test_receive_window_keeps_immediate_original_policy_withdrawal(stop):
    runtime = B601SharedControlRuntime(replace(_config(), damping_velocity_window_sec=.015))
    runtime.publish_policy(_command(1, 1.0))
    first = runtime.step(control_sequence=0, control_timestamp_sec=1.0,
        measured_qpos_rad=ZERO, measured_qvel_rad_s=ZERO, dt_sec=.01,
        joint_observation_timestamps_sec=(1.0,) * 6)
    assert any(first.reference.joint_torque_overlay_nm)
    if stop == "clutch":
        runtime.set_policy_clutch(True)
    now = 1.2 if stop == "stale" else 1.01
    withdrawn = runtime.step(control_sequence=1, control_timestamp_sec=now,
        measured_qpos_rad=ZERO, measured_qvel_rad_s=ZERO, dt_sec=.01,
        human_override_detected=stop == "override", joint_observation_timestamps_sec=(now,) * 6)
    assert withdrawn.reference.joint_torque_overlay_nm == ZERO
    assert withdrawn.decision.state.authority == 0.0

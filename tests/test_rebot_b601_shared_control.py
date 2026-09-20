from __future__ import annotations

import pytest

from robot_motion.rebot_b601_rs import (
    B601_SHARED_POLICY_FRAME,
    B601PolicyAssistCommand,
    B601PolicyCommandBuffer,
    B601PolicyCommandSample,
    B601SharedControlConfig,
    B601SharedControlMode,
    B601SharedControlState,
    compute_b601_shared_control_step,
)

ZERO = (0.0,) * 6


def _command(
    sequence: int,
    *,
    source_timestamp_sec: float = 1.0,
    qpos_target_rad: tuple[float, ...] = (1.0,) * 6,
) -> B601PolicyAssistCommand:
    return B601PolicyAssistCommand(
        sequence=sequence,
        source_timestamp_sec=source_timestamp_sec,
        receive_timestamp_sec=source_timestamp_sec + 0.01,
        qpos_target_rad=qpos_target_rad,
    )


def _config() -> B601SharedControlConfig:
    return B601SharedControlConfig(
        position_kp_nm_per_rad=(10.0,) * 6,
        velocity_kd_nm_s_per_rad=(1.0,) * 6,
        max_target_error_rad=(0.1,) * 6,
        max_abs_assist_torque_nm=(0.5,) * 6,
        max_assist_torque_rate_nm_s=(2.0,) * 6,
        authority_rise_per_sec=2.0,
        policy_stale_timeout_sec=0.2,
    )


def _accepted(command: B601PolicyAssistCommand) -> B601PolicyCommandSample:
    return B601PolicyCommandSample(command, True, "accepted", 0.01)


def test_policy_command_rejects_wrong_frame_and_clock_order() -> None:
    with pytest.raises(ValueError, match="frame_id"):
        B601PolicyAssistCommand(
            sequence=1,
            source_timestamp_sec=1.0,
            receive_timestamp_sec=1.1,
            qpos_target_rad=ZERO,
            frame_id="wrong",
        )
    with pytest.raises(ValueError, match="must not precede"):
        B601PolicyAssistCommand(
            sequence=1,
            source_timestamp_sec=2.0,
            receive_timestamp_sec=1.0,
            qpos_target_rad=ZERO,
            frame_id=B601_SHARED_POLICY_FRAME,
        )


def test_latest_value_buffer_rejects_replay_and_revokes_previous_command() -> None:
    buffer = B601PolicyCommandBuffer()
    assert buffer.publish(_command(2)).accepted is True
    replay = buffer.publish(_command(2, source_timestamp_sec=1.1))
    assert replay.accepted is False
    assert replay.reason == "non_advancing_sequence"

    revoked = buffer.sample(
        now_sec=1.15,
        maximum_age_sec=0.2,
        maximum_future_skew_sec=0.01,
    )
    assert revoked.accepted is False
    assert revoked.command is None
    assert revoked.reason == "no_policy_command"

    assert buffer.publish(_command(3, source_timestamp_sec=1.0)).accepted is True

    stale = buffer.sample(
        now_sec=1.21,
        maximum_age_sec=0.2,
        maximum_future_skew_sec=0.01,
    )
    assert stale.accepted is False
    assert stale.command is None
    assert stale.reason == "policy_command_stale"


def test_buffer_clear_withdraws_command_without_reopening_replay() -> None:
    buffer = B601PolicyCommandBuffer()
    assert buffer.publish(_command(2)).accepted is True

    assert buffer.clear() == 2
    sample = buffer.sample(
        now_sec=1.05,
        maximum_age_sec=0.2,
        maximum_future_skew_sec=0.01,
    )
    assert sample.accepted is False
    assert sample.reason == "no_policy_command"

    replay = buffer.publish(_command(2, source_timestamp_sec=1.1))
    assert replay.accepted is False
    assert replay.reason == "non_advancing_sequence"
    assert buffer.publish(_command(3, source_timestamp_sec=1.1)).accepted is True


def test_buffer_can_sample_with_owner_selected_watchdog_config() -> None:
    buffer = B601PolicyCommandBuffer()
    assert buffer.publish(_command(1)).accepted is True
    assert buffer.sample_for_config(now_sec=1.19, config=_config()).accepted is True
    stale = buffer.sample_for_config(now_sec=1.21, config=_config())
    assert stale.accepted is False
    assert stale.reason == "policy_command_stale"


def test_buffer_rejects_future_and_invalid_samples() -> None:
    buffer = B601PolicyCommandBuffer()
    invalid = _command(1)
    invalid = B601PolicyAssistCommand(
        sequence=invalid.sequence,
        source_timestamp_sec=invalid.source_timestamp_sec,
        receive_timestamp_sec=invalid.receive_timestamp_sec,
        qpos_target_rad=invalid.qpos_target_rad,
        valid=False,
    )
    receipt = buffer.publish(invalid)
    assert receipt.accepted is False
    assert receipt.reason == "command_marked_invalid"
    assert buffer.publish(_command(2, source_timestamp_sec=2.0)).accepted is True
    sample = buffer.sample(
        now_sec=1.0,
        maximum_age_sec=0.2,
        maximum_future_skew_sec=0.05,
    )
    assert sample.accepted is False
    assert sample.reason == "policy_command_from_future"


def test_invalid_new_command_revokes_previous_valid_command() -> None:
    buffer = B601PolicyCommandBuffer()
    assert buffer.publish(_command(1)).accepted is True
    invalid = B601PolicyAssistCommand(
        sequence=2,
        source_timestamp_sec=1.0,
        receive_timestamp_sec=1.01,
        qpos_target_rad=ZERO,
        valid=False,
    )

    assert buffer.publish(invalid).reason == "command_marked_invalid"
    assert buffer.sample(
        now_sec=1.02,
        maximum_age_sec=0.2,
        maximum_future_skew_sec=0.01,
    ).reason == "no_policy_command"


def test_assist_applies_error_torque_authority_and_slew_bounds() -> None:
    output = compute_b601_shared_control_step(
        _config(),
        B601SharedControlState(),
        command_sample=_accepted(_command(1)),
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.1,
        policy_clutch=False,
    )
    assert output.state.mode is B601SharedControlMode.POLICY_ASSIST
    assert output.state.authority == pytest.approx(0.2)
    assert output.raw_target_error_rad == (1.0,) * 6
    assert output.bounded_target_error_rad == (0.1,) * 6
    assert output.raw_assist_torque_nm == (1.0,) * 6
    assert output.assist_torque_nm == (0.1,) * 6
    assert output.target_error_limited is True
    assert output.torque_limited is True
    assert output.slew_limited is False


def test_slew_limit_is_independent_of_assist_torque_limit() -> None:
    state = B601SharedControlState(authority=1.0)
    output = compute_b601_shared_control_step(
        _config(),
        state,
        command_sample=_accepted(_command(1)),
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.05,
        policy_clutch=False,
    )
    assert output.assist_torque_nm == (0.1,) * 6
    assert output.slew_limited is True


def test_clutch_drops_assist_immediately_and_requires_new_sequence() -> None:
    active = B601SharedControlState(
        mode=B601SharedControlMode.POLICY_ASSIST,
        authority=1.0,
        previous_assist_torque_nm=(0.4,) * 6,
        last_policy_sequence=7,
    )
    old = _accepted(_command(7))
    override = compute_b601_shared_control_step(
        _config(),
        active,
        command_sample=old,
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.01,
        policy_clutch=True,
    )
    assert override.state.mode is B601SharedControlMode.HUMAN_OVERRIDE
    assert override.assist_torque_nm == ZERO
    assert override.state.resume_after_sequence == 7

    still_old = compute_b601_shared_control_step(
        _config(),
        override.state,
        command_sample=old,
        measured_qpos_rad=(0.5,) * 6,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.1,
        policy_clutch=False,
    )
    assert still_old.state.mode is B601SharedControlMode.RECOVERY
    assert still_old.reason == "awaiting_rebased_policy"
    assert still_old.assist_torque_nm == ZERO

    rebased = compute_b601_shared_control_step(
        _config(),
        still_old.state,
        command_sample=_accepted(
            _command(8, qpos_target_rad=(0.51,) * 6)
        ),
        measured_qpos_rad=(0.5,) * 6,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.1,
        policy_clutch=False,
    )
    assert rebased.state.mode is B601SharedControlMode.RECOVERY
    assert rebased.state.authority == pytest.approx(0.2)
    assert rebased.state.resume_after_sequence is None
    assert rebased.assist_torque_nm == pytest.approx((0.02,) * 6)


def test_stale_or_missing_policy_fails_passive_without_old_assist() -> None:
    active = B601SharedControlState(
        mode=B601SharedControlMode.POLICY_ASSIST,
        authority=1.0,
        previous_assist_torque_nm=(0.4,) * 6,
        last_policy_sequence=4,
    )
    output = compute_b601_shared_control_step(
        _config(),
        active,
        command_sample=B601PolicyCommandSample(
            command=None,
            accepted=False,
            reason="policy_command_stale",
            age_sec=0.4,
        ),
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.01,
        policy_clutch=False,
    )
    assert output.state.mode is B601SharedControlMode.PASSIVE
    assert output.reason == "policy_command_stale"
    assert output.assist_torque_nm == ZERO


def test_human_override_signal_uses_same_fail_passive_contract() -> None:
    output = compute_b601_shared_control_step(
        _config(),
        B601SharedControlState(
            mode=B601SharedControlMode.POLICY_ASSIST,
            authority=1.0,
            previous_assist_torque_nm=(0.2,) * 6,
            last_policy_sequence=3,
        ),
        command_sample=_accepted(_command(3)),
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.01,
        policy_clutch=False,
        human_override_detected=True,
    )
    assert output.state.mode is B601SharedControlMode.HUMAN_OVERRIDE
    assert output.reason == "human_override_detected"
    assert output.assist_torque_nm == ZERO

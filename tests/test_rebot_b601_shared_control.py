from __future__ import annotations

import math

import pytest

from robot_motion.rebot_b601_rs import (
    B601_SHARED_POLICY_FRAME,
    B601AssistCommissioningState,
    B601AssistResponseCriteria,
    B601CartesianAssistConfig,
    B601CartesianAssistResponseCriteria,
    B601CartesianAssistSession,
    B601PolicyAssistCommand,
    B601PolicyCommandBuffer,
    B601PolicyCommandSample,
    B601SharedControlConfig,
    B601SharedControlMode,
    B601SharedControlRuntime,
    B601SharedControlState,
    B601SingleJointAssistConfig,
    B601SingleJointAssistSession,
    compute_b601_shared_control_step,
    evaluate_cartesian_assist_response,
    evaluate_single_joint_assist_response,
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


def _assist_session(
    **overrides: object,
) -> tuple[B601SingleJointAssistConfig, B601SharedControlRuntime, B601SingleJointAssistSession]:
    config = B601SingleJointAssistConfig(
        countdown_sec=1.0,
        assist_duration_sec=2.0,
        **overrides,
    )
    runtime = B601SharedControlRuntime(config.shared_control_config())
    return config, runtime, B601SingleJointAssistSession(config, runtime)


def test_single_joint_commissioning_uses_only_target_joint_gains() -> None:
    config = B601SingleJointAssistConfig(joint_name="joint6")
    shared = config.shared_control_config()

    assert shared.position_kp_nm_per_rad == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    assert shared.velocity_kd_nm_s_per_rad == ZERO


def test_single_joint_commissioning_reports_effective_static_torque() -> None:
    gain_limited = B601SingleJointAssistConfig(
        position_kp_nm_per_rad=1.0,
        max_target_error_rad=0.05,
        max_abs_assist_torque_nm=0.20,
    )
    cap_limited = B601SingleJointAssistConfig(
        position_kp_nm_per_rad=10.0,
        max_target_error_rad=0.05,
        max_abs_assist_torque_nm=0.20,
    )

    assert gain_limited.maximum_static_assist_torque_nm == pytest.approx(0.05)
    assert cap_limited.maximum_static_assist_torque_nm == pytest.approx(0.20)


def test_single_joint_commissioning_rejects_motion_and_out_of_limit_target() -> None:
    _config_value, _runtime, session = _assist_session()
    limits = ((-1.0, 1.0),) * 6

    with pytest.raises(ValueError, match="not stationary"):
        session.arm(
            measured_qpos_rad=ZERO,
            measured_qvel_rad_s=(0.0, 0.0, 0.0, 0.0, 0.0, 0.03),
            position_limits_rad=limits,
            timestamp_sec=1.0,
        )

    with pytest.raises(ValueError, match="outside"):
        session.arm(
            measured_qpos_rad=(0.0, 0.0, 0.0, 0.0, 0.0, 0.95),
            measured_qvel_rad_s=ZERO,
            position_limits_rad=limits,
            timestamp_sec=1.0,
        )


def test_single_joint_commissioning_is_one_shot_and_has_no_return_target() -> None:
    config, runtime, session = _assist_session()
    limits = ((-1.0, 1.0),) * 6
    anchor = (0.0, 0.0, 0.0, 0.0, 0.0, 0.20)

    armed = session.arm(
        measured_qpos_rad=anchor,
        measured_qvel_rad_s=ZERO,
        position_limits_rad=limits,
        timestamp_sec=10.0,
    )
    assert armed.event == "ARMED"
    assert session.target_qpos_rad == pytest.approx((*anchor[:5], 0.30))
    assert session.step(
        measured_qpos_rad=anchor,
        measured_qvel_rad_s=ZERO,
        timestamp_sec=10.5,
    ) == ()

    events = session.step(
        measured_qpos_rad=anchor,
        measured_qvel_rad_s=ZERO,
        timestamp_sec=11.0,
    )
    assert [event.event for event in events] == ["ACTIVE"]
    active = runtime.step(
        control_sequence=1,
        control_timestamp_sec=11.0,
        measured_qpos_rad=anchor,
        measured_qvel_rad_s=ZERO,
        dt_sec=0.05,
    )
    assert active.decision.assist_torque_nm[-1] > 0.0
    assert active.reference.qpos_rad == anchor

    completed = session.step(
        measured_qpos_rad=(*anchor[:5], 0.25),
        measured_qvel_rad_s=ZERO,
        timestamp_sec=13.01,
    )
    assert [event.event for event in completed] == ["COMPLETE"]
    assert session.state is B601AssistCommissioningState.COMPLETE
    passive = runtime.step(
        control_sequence=2,
        control_timestamp_sec=13.01,
        measured_qpos_rad=(*anchor[:5], 0.25),
        measured_qvel_rad_s=ZERO,
        dt_sec=0.05,
    )
    assert passive.decision.assist_torque_nm == ZERO
    assert passive.reference.qpos_rad == (*anchor[:5], 0.25)
    assert config.offset_rad > 0.0


def test_assist_response_requires_delivered_torque_and_mechanical_motion() -> None:
    config = B601SingleJointAssistConfig(
        joint_name="joint6",
        offset_rad=0.1,
        max_abs_assist_torque_nm=0.05,
    )
    criteria = B601AssistResponseCriteria(minimum_response_rad=0.001)
    anchor = ZERO
    no_motion = evaluate_single_joint_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_qpos_rad=anchor,
        measured_qpos_samples_rad=((0.0, 0.0, 0.0, 0.0, 0.0, 0.00038),) * 6,
        assist_torque_samples_nm=((0.0, 0.0, 0.0, 0.0, 0.0, 0.05),) * 6,
        criteria=criteria,
    )
    not_delivered = evaluate_single_joint_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_qpos_rad=anchor,
        measured_qpos_samples_rad=((0.0, 0.0, 0.0, 0.0, 0.0, 0.002),) * 6,
        assist_torque_samples_nm=((0.0, 0.0, 0.0, 0.0, 0.0, 0.01),) * 6,
        criteria=criteria,
    )
    passed = evaluate_single_joint_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_qpos_rad=anchor,
        measured_qpos_samples_rad=((0.0, 0.0, 0.0, 0.0, 0.0, 0.002),) * 6,
        assist_torque_samples_nm=((0.0, 0.0, 0.0, 0.0, 0.0, 0.05),) * 6,
        criteria=criteria,
    )

    assert no_motion["status"] == "INSUFFICIENT_RESPONSE"
    assert no_motion["passed"] is False
    assert not_delivered["status"] == "ASSIST_NOT_DELIVERED"
    assert passed["status"] == "PASS"
    assert passed["passed"] is True


def test_assist_response_rejects_wrong_direction_and_support_drift() -> None:
    config = B601SingleJointAssistConfig(joint_name="joint6", offset_rad=0.1)
    criteria = B601AssistResponseCriteria(minimum_response_rad=0.001)
    assist = ((0.0, 0.0, 0.0, 0.0, 0.0, 0.05),) * 6
    wrong_direction = evaluate_single_joint_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_qpos_rad=ZERO,
        measured_qpos_samples_rad=((0.0, 0.0, 0.0, 0.0, 0.0, -0.002),) * 6,
        assist_torque_samples_nm=assist,
        criteria=criteria,
    )
    support_drift = evaluate_single_joint_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_qpos_rad=ZERO,
        measured_qpos_samples_rad=((0.01, 0.0, 0.0, 0.0, 0.0, 0.002),) * 6,
        assist_torque_samples_nm=assist,
        criteria=criteria,
    )

    assert wrong_direction["status"] == "WRONG_DIRECTION"
    assert support_drift["status"] == "SUPPORT_DRIFT_EXCEEDED"


@pytest.mark.parametrize(
    ("measured", "reason"),
    [
        ((0.0, 0.0, 0.0, 0.0, 0.0, -0.002), "wrong direction"),
        ((0.01, 0.0, 0.0, 0.0, 0.0, 0.002), "support-joint drift"),
        ((0.0, 0.0, 0.0, 0.0, 0.0, 0.16), "excursion"),
    ],
)
def test_active_assist_motion_guard_fails_before_another_command(
    measured: tuple[float, ...],
    reason: str,
) -> None:
    _config_value, _runtime, session = _assist_session()
    session.anchor_qpos_rad = ZERO
    session.state = B601AssistCommissioningState.ACTIVE

    with pytest.raises(ValueError, match=reason):
        session.validate_active_motion(
            measured,
            criteria=B601AssistResponseCriteria(minimum_response_rad=0.001),
        )


def test_single_joint_countdown_motion_requires_a_new_anchor() -> None:
    _config_value, runtime, session = _assist_session()
    session.arm(
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=ZERO,
        position_limits_rad=((-1.0, 1.0),) * 6,
        timestamp_sec=1.0,
    )

    events = session.step(
        measured_qpos_rad=ZERO,
        measured_qvel_rad_s=(0.0, 0.0, 0.0, 0.0, 0.0, 0.03),
        timestamp_sec=1.1,
    )

    assert [event.event for event in events] == ["COUNTDOWN_REJECTED"]
    assert session.state is B601AssistCommissioningState.WAITING
    assert runtime.clear_policy() is None


def _pose(x: float = 0.0, y: float = 0.0, z: float = 0.0):
    return (
        (1.0, 0.0, 0.0, x),
        (0.0, 1.0, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def test_cartesian_assist_captures_world_axis_target_and_withdraws() -> None:
    config = B601CartesianAssistConfig(
        axis="y",
        offset_m=-0.01,
        countdown_sec=1.0,
        assist_duration_sec=2.0,
    )
    session = B601CartesianAssistSession(config)

    armed = session.arm(
        measured_pose=_pose(0.2, -0.1, 0.3),
        measured_qvel_rad_s=ZERO,
        timestamp_sec=10.0,
    )
    assert armed.event == "ARMED"
    assert session.anchor_position_m == pytest.approx((0.2, -0.1, 0.3))
    assert session.target_position_m == pytest.approx((0.2, -0.11, 0.3))
    assert session.step(
        measured_qvel_rad_s=ZERO,
        timestamp_sec=11.0,
    )[0].event == "ACTIVE"
    assert session.authority(timestamp_sec=11.5) == pytest.approx(0.25)
    completed = session.step(
        measured_qvel_rad_s=ZERO,
        timestamp_sec=13.01,
    )
    assert completed[0].event == "COMPLETE"
    assert session.authority(timestamp_sec=13.01) == 0.0


def test_cartesian_assist_guard_rejects_wrong_axis_and_lateral_motion() -> None:
    criteria = B601CartesianAssistResponseCriteria(
        minimum_response_m=0.001,
        maximum_lateral_drift_m=0.003,
    )
    session = B601CartesianAssistSession(
        B601CartesianAssistConfig(axis="x", countdown_sec=1.0)
    )
    session.anchor_position_m = (0.0, 0.0, 0.0)
    session.state = B601AssistCommissioningState.ACTIVE

    with pytest.raises(ValueError, match="wrong direction"):
        session.validate_active_position((-0.002, 0.0, 0.0), criteria=criteria)
    with pytest.raises(ValueError, match="lateral drift"):
        session.validate_active_position((0.002, 0.004, 0.0), criteria=criteria)
    with pytest.raises(ValueError, match="excursion"):
        session.validate_active_position((0.016, 0.0, 0.0), criteria=criteria)

    session.anchor_rotation = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    angle = 0.10
    rotated = (
        (math.cos(angle), -math.sin(angle), 0.0, 0.002),
        (math.sin(angle), math.cos(angle), 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    with pytest.raises(ValueError, match="orientation drift"):
        session.validate_active_pose(rotated, criteria=criteria)


def test_cartesian_assist_response_requires_force_and_task_space_motion() -> None:
    config = B601CartesianAssistConfig(
        axis="z",
        offset_m=0.01,
        stiffness_n_per_m=50.0,
        max_abs_force_n=0.5,
    )
    criteria = B601CartesianAssistResponseCriteria(minimum_response_m=0.001)
    response = evaluate_cartesian_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_position_m=(0.0, 0.0, 0.0),
        measured_position_samples_m=((0.0, 0.0, 0.002),) * 6,
        force_samples_n=((0.0, 0.0, 0.5),) * 6,
        rotation_drift_samples_rad=(0.0,) * 6,
        criteria=criteria,
    )
    missing_force = evaluate_cartesian_assist_response(
        config=config,
        state=B601AssistCommissioningState.COMPLETE,
        anchor_position_m=(0.0, 0.0, 0.0),
        measured_position_samples_m=((0.0, 0.0, 0.002),) * 6,
        force_samples_n=((0.0, 0.0, 0.1),) * 6,
        rotation_drift_samples_rad=(0.0,) * 6,
        criteria=criteria,
    )

    assert response["status"] == "PASS"
    assert response["passed"] is True
    assert missing_force["status"] == "ASSIST_NOT_DELIVERED"


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

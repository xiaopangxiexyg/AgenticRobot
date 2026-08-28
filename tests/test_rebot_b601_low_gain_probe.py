from __future__ import annotations

import numpy as np
import pytest
from robot_motion.rebot_b601_rs import B601_RS_ARM_JOINT_NAMES
from scripts.hardware import rebot_b601_rs_low_gain_probe as probe

pytestmark = [pytest.mark.unit, pytest.mark.contract]


def test_condition_torque_applies_absolute_then_slew_limits() -> None:
    conditioned, torque_limited, slew_limited = probe.condition_torque(
        (10.0, -10.0, 0.1, 0.0, 0.0, 0.0),
        (0.0,) * 6,
        absolute_limits_nm=(2.0,) * 6,
        max_rate_nm_s=(1.0,) * 6,
        dt_sec=0.5,
    )

    assert conditioned == (0.5, -0.5, 0.1, 0.0, 0.0, 0.0)
    assert torque_limited is True
    assert slew_limited is True


def test_condition_torque_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="dt_sec must be positive"):
        probe.condition_torque(
            (0.0,) * 6,
            (0.0,) * 6,
            absolute_limits_nm=(1.0,) * 6,
            max_rate_nm_s=(1.0,) * 6,
            dt_sec=0.0,
        )


def test_software_profile_has_single_torque_owner() -> None:
    profile = probe.build_software_profile(
        kp=0.5,
        kd=0.5,
        torque_limits_nm=(2.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=0.1,
    )

    assert profile.position_kp_nm_per_rad == (0.5,) * 6
    assert profile.damping_kd_nm_s_per_rad == (0.5,) * 6
    assert profile.max_torque_rate_nm_s == (2.0,) * 6
    assert profile.metadata["torque_owner"] == "software_composed_tau; vendor_kp_kd_zero"


def test_compute_software_torque_composes_gravity_pd_and_damping() -> None:
    class FakeModel:
        def arm_generalized_gravity(self, qpos_rad: object) -> tuple[float, ...]:
            assert tuple(qpos_rad) == (0.0,) * 6
            return (1.0, -2.0, 3.0, -4.0, 5.0, -6.0)

    reference = probe.build_reference(
        sequence=1,
        timestamp_sec=1.0,
        qpos_rad=(0.1, -0.1, 0.2, -0.2, 0.3, -0.3),
        qvel_rad_s=(0.0,) * 6,
    )
    profile = probe.build_software_profile(
        kp=1.0,
        kd=0.5,
        torque_limits_nm=(20.0,) * 6,
        torque_rise_time_sec=1.0,
        velocity_limit_rad_s=1.0,
    )

    torque = probe.compute_software_torque(
        FakeModel(),
        reference=reference,
        measured_qpos_rad=(0.0,) * 6,
        measured_qvel_rad_s=(0.1,) * 6,
        profile=profile,
        gravity_scale=0.5,
    )

    expected = tuple(
        0.5 * gravity + position_error - 0.05
        for gravity, position_error in zip(
            (1.0, -2.0, 3.0, -4.0, 5.0, -6.0),
            (0.1, -0.1, 0.2, -0.2, 0.3, -0.3),
            strict=True,
        )
    )
    assert np.allclose(torque, expected)


def test_hold_reference_never_moves_from_initial_pose() -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    qref, qvel, done = probe.plan_reference(
        q0,
        mode="hold",
        joint="joint3",
        elapsed_sec=10.0,
        step_rad=0.05,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14)),
    )

    assert qref == q0
    assert qvel == (0.0,) * 6
    assert done is True


def test_active_joint_resolution_keeps_joint2_joint3_support() -> None:
    assert probe.resolve_active_joint_names("joint1") == (
        "joint1",
        "joint2",
        "joint3",
    )
    assert probe.resolve_active_joint_names("joint3") == (
        "joint3",
        "joint2",
    )


def test_active_joint_resolution_rejects_missing_fault_support() -> None:
    with pytest.raises(ValueError, match="mandatory support"):
        probe.resolve_active_joint_names("joint1", support_joints=("joint4",))


def _fault_hold_limits() -> tuple[tuple[float, float], ...]:
    return (
        (-2.8, 2.8),
        (0.0, 3.14),
        (0.0, 3.14),
        (-1.57, 1.57),
        (-1.57, 1.57),
        (-3.14, 3.14),
    )


def test_fault_hold_snapshot_accepts_only_fresh_stationary_in_limit_state() -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)

    accepted = probe._validate_fault_hold_snapshot(
        q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.05,
        position_limits_rad=_fault_hold_limits(),
        fault_qpos_rad=(0.1, 0.72, 1.08, 0.0, -0.1, 0.2),
        fault_qvel_rad_s=(0.0,) * 6,
    )

    assert accepted == q


def test_fault_hold_snapshot_rejects_stale_state() -> None:
    with pytest.raises(probe.LowGainProbeError, match="stale"):
        probe._validate_fault_hold_snapshot(
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.16,
            position_limits_rad=_fault_hold_limits(),
        )


def test_fault_hold_snapshot_rejects_motion_and_large_rebase() -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    with pytest.raises(probe.LowGainProbeError, match="moving"):
        probe._validate_fault_hold_snapshot(
            q,
            qvel_rad_s=(0.0, 0.021, 0.0, 0.0, 0.0, 0.0),
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
        )
    with pytest.raises(probe.LowGainProbeError, match="rebase"):
        probe._validate_fault_hold_snapshot(
            q,
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
            fault_qpos_rad=(0.1, 0.76, 1.1, 0.0, -0.1, 0.2),
        )


def test_fault_hold_snapshot_rejects_fault_position_without_velocity() -> None:
    with pytest.raises(probe.LowGainProbeError, match="velocity is unavailable"):
        probe._validate_fault_hold_snapshot(
            (0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
            fault_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
        )


def test_protective_hold_feedback_rejects_drift_from_fixed_reference() -> None:
    with pytest.raises(probe.LowGainProbeError, match="drift"):
        probe._validate_protective_hold_feedback(
            (0.1, 0.76, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            hold_qpos_rad=(0.1, 0.7, 1.1, 0.0, -0.1, 0.2),
            observed_age_sec=0.0,
            position_limits_rad=_fault_hold_limits(),
        )


def test_latched_fault_hold_falls_back_to_full_disable_on_invalid_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    command_qrefs: list[tuple[float, ...]] = []

    def fake_send_hold(
        _backend: object,
        _model: object,
        *,
        qpos_rad: tuple[float, ...],
        measured_qpos_rad: tuple[float, ...],
        measured_qvel_rad_s: tuple[float, ...],
        **_kwargs: object,
    ) -> tuple[probe.B601CommandReceipt, tuple[float, ...]]:
        assert measured_qpos_rad == q
        assert measured_qvel_rad_s == (0.0,) * 6
        command_qrefs.append(qpos_rad)
        receipt = probe.B601CommandReceipt(
            sequence=len(command_qrefs),
            mode="mit",
            frame_id="b601_rs/joint_mit_reference",
            requested_at_sec=1.0,
            completed_at_sec=1.0,
            requested_joint_names=("joint2", "joint3"),
            sent_joint_names=("joint2", "joint3"),
            failed_joint_names=(),
            errors=(),
            accepted=True,
        )
        return receipt, (0.0,) * 6

    class FakeBackend:
        def poll_feedback(self, *, request_feedback: bool) -> object:
            assert request_feedback is True
            return type(
                "State",
                (),
                {
                    "valid": False,
                    "stale": True,
                    "positions_rad": None,
                    "fault_reason": "simulated feedback loss",
                },
            )()

        def safe_stop(self, *, joint_names: tuple[str, ...]) -> probe.B601ModeReceipt:
            if joint_names == probe._NON_FAULT_HOLD_JOINTS:
                operation = "safe_stop"
            else:
                operation = "full_safe_stop"
            return probe.B601ModeReceipt(
                operation=operation,
                requested_joint_names=tuple(joint_names),
                completed_joint_names=tuple(joint_names),
                failed_joint_names=(),
                errors=(),
                completed=True,
            )

    monkeypatch.setattr(probe, "_send_fault_hold_command", fake_send_hold)
    result = probe._run_latched_fault_hold(
        FakeBackend(),
        object(),
        qpos_rad=q,
        qvel_rad_s=(0.0,) * 6,
        observed_age_sec=0.01,
        fault_qpos_rad=q,
        fault_qvel_rad_s=(0.0,) * 6,
        position_limits_rad=_fault_hold_limits(),
        max_age_sec=0.15,
        max_velocity_rad_s=0.02,
        max_rebase_rad=0.05,
        profile=object(),
        gravity_scale=1.0,
        previous_tau=(0.0,) * 6,
        torque_limits_nm=(1.0,) * 6,
        rate_hz=20.0,
        stop={"requested": True, "full_disable_requested": False},
        max_iterations=2,
    )

    hold_count, _latest_tau, first_receipt, non_hold_receipt, full_receipt, errors = result
    assert hold_count == 1
    assert command_qrefs == [q]
    assert first_receipt is not None and first_receipt.accepted
    assert non_hold_receipt is not None and non_hold_receipt.completed
    assert full_receipt is not None and full_receipt.completed
    assert any("feedback is not complete and valid" in error for error in errors)


def test_fault_hold_snapshot_rejects_out_of_limit_q() -> None:
    with pytest.raises(probe.LowGainProbeError, match="outside URDF limits"):
        probe._validate_fault_hold_snapshot(
            (0.1, 3.2, 1.1, 0.0, -0.1, 0.2),
            qvel_rad_s=(0.0,) * 6,
            observed_age_sec=0.01,
            position_limits_rad=_fault_hold_limits(),
        )


def test_sigusr1_requests_full_disable() -> None:
    stop = {"requested": False, "full_disable_requested": False}

    probe._signal_handler(stop, probe.signal.SIGUSR1, None)

    assert stop == {"requested": True, "full_disable_requested": True}


def test_slow_step_is_relative_and_bounded() -> None:
    q0 = (0.1, 0.7, 1.1, 0.0, -0.1, 0.2)
    limits = ((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14))
    qref, qvel, done = probe.plan_reference(
        q0,
        mode="slow-step",
        joint="joint3",
        elapsed_sec=1.0,
        step_rad=0.05,
        step_speed_rad_s=0.02,
        step_direction=1,
        position_limits_rad=limits,
    )

    assert np.allclose(qref, (0.1, 0.7, 1.12, 0.0, -0.1, 0.2))
    assert qvel == (0.0, 0.0, 0.02, 0.0, 0.0, 0.0)
    assert done is False


def test_slow_step_rejects_target_outside_urdf_limit() -> None:
    with pytest.raises(ValueError, match="outside URDF limits"):
        probe.plan_reference(
            (0.1, 0.7, 3.13, 0.0, -0.1, 0.2),
            mode="slow-step",
            joint="joint3",
            elapsed_sec=0.0,
            step_rad=0.05,
            step_speed_rad_s=0.02,
            step_direction=1,
            position_limits_rad=((-2.8, 2.8), (0.0, 3.14), (0.0, 3.14), (-1.57, 1.57), (-1.57, 1.57), (-3.14, 3.14)),
        )


def test_effort_limits_are_read_from_urdf_and_scaled(tmp_path) -> None:
    effort_values = (36.0, 36.0, 36.0, 14.0, 14.0, 14.0)
    joints = "".join(
        f'<joint name="{name}" type="revolute">'
        f'<limit effort="{effort}" lower="-1" upper="1" velocity="1"/>'
        "</joint>"
        for name, effort in zip(probe._ALL_JOINTS, effort_values, strict=True)
    )
    urdf = tmp_path / "b601_test.urdf"
    urdf.write_text(f'<robot name="b601">{joints}</robot>', encoding="utf-8")

    limits = probe._resolve_effort_limits(urdf, scale=0.25)

    assert limits == (9.0, 9.0, 9.0, 3.5, 3.5, 3.5)


def test_parser_help_is_read_only() -> None:
    parser = probe.build_parser()
    assert "--allow-motion" in parser.format_help()
    assert tuple(B601_RS_ARM_JOINT_NAMES) == tuple(probe._ALL_JOINTS)
    args = parser.parse_args(["--config", "config.yaml", "--rebot-source-root", "source"])
    assert args.joint == "joint1"
    assert tuple(args.support_joints) == ("joint2", "joint3")

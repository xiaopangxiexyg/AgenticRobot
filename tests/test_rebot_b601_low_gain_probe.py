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

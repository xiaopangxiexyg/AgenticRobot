from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from hardware.rm65.calibration_profiles import load_profile
from hardware.rm65.impedance import RM65ImpedanceConfig
from teleop_runtime.debug import run_vive_rm65_real_teleop as real_teleop
from teleop_runtime.real_teleop.control_profile import effective_dual_arm_temporary_safety


@pytest.mark.parametrize(
    "profile_name",
    (
        "hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json",
        "hardware/rm65/utils/configs/calibrations/rm65_right_DMTAC_M.json",
    ),
)
@pytest.mark.parametrize("mode", ("pose", "joint"))
def test_dmtac_impedance_profiles_use_hardened_admittance_and_fast_return(
    profile_name: str,
    mode: str,
) -> None:
    """Keep the bimanual force-control tuning symmetric and bounded."""

    config = RM65ImpedanceConfig.from_profile(load_profile(profile_name), mode=mode)

    assert config.linear_compliance_m_per_n == pytest.approx(0.0012)
    assert config.angular_compliance_rad_per_nm == pytest.approx(0.02)
    assert config.max_linear_offset_m == pytest.approx(0.03)
    assert config.max_angular_offset_rad == pytest.approx(0.12)
    assert config.max_linear_return_velocity_m_s == pytest.approx(0.06)
    assert config.max_angular_return_velocity_rad_s == pytest.approx(0.3)
    assert config.return_rate_s == pytest.approx(4.0)

    thresholds = config.saturation_thresholds()
    assert thresholds["linear_filtered_force_n_for_max_offset"] == pytest.approx(25.45)
    assert thresholds["angular_filtered_torque_nm_for_max_offset"] == pytest.approx(6.04)


def test_hardened_profile_return_rate_and_caps_take_effect_in_admittance() -> None:
    config = RM65ImpedanceConfig.from_profile(
        load_profile("hardware/rm65/utils/configs/calibrations/rm65_left_DMTAC_M.json"),
        mode="pose",
    )
    admittance = config.make_admittance()
    admittance.offset = [0.03, 0.0, 0.0, 0.12, 0.0, 0.0]

    update = admittance.update(
        [0.0] * 6,
        1.0 / 40.0,
        return_activity_wrench=[0.0] * 6,
        return_force_deadband_n=config.effective_return_force_deadband_n,
        return_torque_deadband_nm=config.effective_return_torque_deadband_nm,
    )

    # The faster return is still bounded independently per translation/rotation.
    assert update["offset"][0] == pytest.approx(0.0285)
    assert update["offset"][3] == pytest.approx(0.1125)
    assert update["linear_return_velocity_limit_applied"] is True
    assert update["angular_return_velocity_limit_applied"] is True


def _parse_real_teleop_args(*extra: str):
    argv = [
        "prog",
        "--no-gui",
        "--active-sides",
        "both",
        "--left-arm-ip",
        "192.168.10.21",
        "--right-arm-ip",
        "192.168.10.22",
        "--no-arm-ip-save-latest",
        *extra,
    ]
    with patch.object(sys, "argv", argv):
        return real_teleop.parse_args()


def test_bimanual_impedance_does_not_turn_temporary_output_limits_into_target_chase() -> None:
    args = _parse_real_teleop_args("--arm-execution-profile", "impedance_40hz")

    assert args.arm_control_mode == "impedance"
    # Parser defaults remain available for the position path, but impedance
    # does not apply this post-IK stateful Cartesian limiter.
    assert args.dual_arm_temp_max_xyz_step_m == pytest.approx(0.03)
    assert args.dual_arm_temp_max_rot_step_deg == pytest.approx(8.0)
    assert effective_dual_arm_temporary_safety(
        args,
        position_canfd_passthrough=False,
    ) == {
        "requested": True,
        "enabled": False,
        "disabled_reason": "impedance_uses_pre_ik_coupled_safety_bounds",
    }

    overridden = _parse_real_teleop_args(
        "--arm-execution-profile",
        "impedance_40hz",
        "--dual-arm-temp-max-xyz-step-m",
        "0.01",
        "--dual-arm-temp-max-rot-step-deg=1.5",
    )
    assert overridden.dual_arm_temp_max_xyz_step_m == pytest.approx(0.01)
    assert overridden.dual_arm_temp_max_rot_step_deg == pytest.approx(1.5)


def test_bimanual_impedance_uses_fixed_pre_ik_local_candidate_bounds() -> None:
    profile = real_teleop._impedance_bimanual_ik_safety_profile()

    assert profile.name == "dual_arm_teleop_impedance_local_bounds"
    assert profile.local_step_max_position_worsen_m == pytest.approx(0.003)
    assert profile.local_step_max_orientation_worsen_deg == pytest.approx(0.5)
    assert profile.local_step_dynamic_worsen_enabled is False
    assert profile.metadata["impedance_bimanual_local_step_bounds"] == {
        "max_position_worsen_m": 0.003,
        "max_orientation_worsen_deg": 0.5,
        "dynamic_worsen_enabled": False,
        "scope": "impedance_bimanual_degraded_ik_candidate_only",
    }


def test_bimanual_position_keeps_legacy_temporary_limiter_values() -> None:
    args = _parse_real_teleop_args("--arm-execution-profile", "position")

    assert args.arm_control_mode == "position"
    assert args.dual_arm_temp_max_xyz_step_m == pytest.approx(0.03)
    assert args.dual_arm_temp_max_rot_step_deg == pytest.approx(8.0)

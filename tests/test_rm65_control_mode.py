from __future__ import annotations

import pytest

from robot_motion.rm65.control_mode import (
    RM65_ARM_CONTROL_MODE_IMPEDANCE,
    RM65_ARM_CONTROL_MODE_POSITION,
    normalize_rm65_arm_control_mode,
)
from teleop_runtime.deploy import run_policy
from teleop_runtime.replay import hardware as replay_hardware


@pytest.mark.parametrize("value", (None, "position", "pos", "movej", "canfd"))
def test_shared_rm65_position_mode_aliases(value: object) -> None:
    assert normalize_rm65_arm_control_mode(value) == RM65_ARM_CONTROL_MODE_POSITION


@pytest.mark.parametrize(
    "value",
    ("impedance", "force", "force-control", "admittance", "force_admittance"),
)
def test_shared_rm65_impedance_mode_aliases(value: object) -> None:
    assert normalize_rm65_arm_control_mode(value) == RM65_ARM_CONTROL_MODE_IMPEDANCE


def test_deploy_reexports_shared_rm65_control_mode_normalizer() -> None:
    assert run_policy.normalize_rm65_arm_control_mode is normalize_rm65_arm_control_mode


def test_replay_wrapper_preserves_existing_validation_context() -> None:
    assert replay_hardware.normalize_rm65_arm_control_mode("force") == "impedance"
    with pytest.raises(ValueError, match="Unsupported replay RM65 arm control mode"):
        replay_hardware.normalize_rm65_arm_control_mode("unsupported")

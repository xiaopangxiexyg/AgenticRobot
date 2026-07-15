from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from teleop_runtime.configs import DEFAULT_ARM_WORKSPACE_CONFIG_PATH, build_annotated_dual_rm65_calibration
from teleop_runtime.real_teleop.workspace import (
    apply_workspace_config,
    apply_workspace_overrides,
    arm_workspace_config_path,
    parse_vector3_arg,
    resolve_workspace_override_args,
    workspace_override_for_side,
)
from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint


def test_legacy_workspace_helpers_are_direct_aliases() -> None:
    assert legacy_entrypoint._parse_vector3_arg is parse_vector3_arg
    assert legacy_entrypoint._resolve_workspace_override_args is resolve_workspace_override_args
    assert legacy_entrypoint._workspace_override_for_side is workspace_override_for_side
    assert legacy_entrypoint._arm_workspace_config_path is arm_workspace_config_path
    assert legacy_entrypoint._apply_workspace_config is apply_workspace_config
    assert legacy_entrypoint._apply_workspace_overrides is apply_workspace_overrides


def test_extracted_workspace_policy_applies_config_then_side_override() -> None:
    calibration = build_annotated_dual_rm65_calibration()
    args = SimpleNamespace(
        arm_workspace_config=str(DEFAULT_ARM_WORKSPACE_CONFIG_PATH),
        arm_workspace_min="-0.65,-0.80,0.00",
        arm_workspace_max="0.80,0.80,0.95",
        left_arm_workspace_min=None,
        left_arm_workspace_max="0.75,0.70,0.90",
        right_arm_workspace_min=None,
        right_arm_workspace_max=None,
    )

    resolve_workspace_override_args(args)
    configured = apply_workspace_config(args, calibration)
    overridden = apply_workspace_overrides(args, calibration)

    assert set(configured) == {"left_arm", "right_arm"}
    assert set(overridden) == {"left_arm", "right_arm"}
    np.testing.assert_allclose(
        calibration.arms["left_arm"].workspace_min,
        np.array([-0.65, -0.80, 0.00]),
    )
    np.testing.assert_allclose(
        calibration.arms["left_arm"].workspace_max,
        np.array([0.75, 0.70, 0.90]),
    )
    assert calibration.metadata["workspace_config_source"] == str(DEFAULT_ARM_WORKSPACE_CONFIG_PATH)
    assert calibration.arms["left_arm"].metadata["workspace_override_runtime"] is True

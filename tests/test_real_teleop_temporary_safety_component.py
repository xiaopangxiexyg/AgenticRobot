from __future__ import annotations

import numpy as np

from teleop_runtime.debug import run_vive_rm65_real_teleop as legacy_entrypoint
from teleop_runtime.real_teleop.temporary_dual_arm_safety import (
    TemporaryDualArmSafetyConfig,
    TemporaryDualArmSafetyLimiter,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _pose(x: float) -> np.ndarray:
    value = np.eye(4, dtype=float)
    value[0, 3] = float(x)
    return value


def _cartesian_command(x: float) -> TeleopCommand:
    return TeleopCommand(
        commands={
            "left_arm": EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="cartesian_pose",
                value=_pose(x),
                metadata={
                    "target_pose_base": _pose(x),
                    "solver_output_qpos": np.arange(6, dtype=float),
                },
            )
        }
    )


def _joint_command(qpos: np.ndarray) -> TeleopCommand:
    return TeleopCommand(
        commands={
            "left_arm": EffectorCommand(
                effector_name="left_arm",
                side="left",
                command_type="joint_vector",
                value=np.asarray(qpos, dtype=float),
                metadata={"target_pose_base": _pose(0.2)},
            )
        }
    )


def _assert_same_command(actual: TeleopCommand, expected: TeleopCommand) -> None:
    assert actual.metadata == expected.metadata
    assert set(actual.commands) == set(expected.commands)
    for name in actual.commands:
        actual_command = actual.commands[name]
        expected_command = expected.commands[name]
        assert actual_command.command_type == expected_command.command_type
        np.testing.assert_allclose(actual_command.value, expected_command.value, atol=1e-12)
        assert set(actual_command.metadata) == set(expected_command.metadata)
        for key, actual_value in actual_command.metadata.items():
            expected_value = expected_command.metadata[key]
            if isinstance(actual_value, np.ndarray):
                np.testing.assert_allclose(actual_value, expected_value, atol=1e-12)
            else:
                assert actual_value == expected_value


def test_extracted_temporary_safety_matches_legacy_cartesian_and_joint_contracts() -> None:
    assert legacy_entrypoint.TemporaryDualArmSafetyConfig is TemporaryDualArmSafetyConfig
    assert legacy_entrypoint.TemporaryDualArmSafetyLimiter is TemporaryDualArmSafetyLimiter
    legacy = legacy_entrypoint.TemporaryDualArmSafetyLimiter(
        legacy_entrypoint.TemporaryDualArmSafetyConfig(
            max_xyz_step_m=0.1,
            max_joint_step_deg=10.0,
        )
    )
    extracted = TemporaryDualArmSafetyLimiter(
        TemporaryDualArmSafetyConfig(max_xyz_step_m=0.1, max_joint_step_deg=10.0)
    )

    _assert_same_command(
        extracted.filter_command(_cartesian_command(0.0), None, 0.0),
        legacy.filter_command(_cartesian_command(0.0), None, 0.0),
    )
    _assert_same_command(
        extracted.filter_command(_cartesian_command(0.3), None, 0.1),
        legacy.filter_command(_cartesian_command(0.3), None, 0.1),
    )

    legacy_joint = legacy_entrypoint.TemporaryDualArmSafetyLimiter(
        legacy_entrypoint.TemporaryDualArmSafetyConfig(max_joint_step_deg=10.0)
    )
    extracted_joint = TemporaryDualArmSafetyLimiter(
        TemporaryDualArmSafetyConfig(max_joint_step_deg=10.0)
    )
    _assert_same_command(
        extracted_joint.filter_command(_joint_command(np.zeros(6)), None, 0.0),
        legacy_joint.filter_command(_joint_command(np.zeros(6)), None, 0.0),
    )
    _assert_same_command(
        extracted_joint.filter_command(
            _joint_command(np.deg2rad(np.full(6, 30.0))),
            None,
            0.1,
        ),
        legacy_joint.filter_command(
            _joint_command(np.deg2rad(np.full(6, 30.0))),
            None,
            0.1,
        ),
    )

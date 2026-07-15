from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from teleop_runtime.real_teleop.fix_arm_collision_guard import (
    fixed_arm_final_collision_fault,
)
from teleop_runtime.schemas import EffectorCommand, TeleopCommand


def _command() -> TeleopCommand:
    command = TeleopCommand(timestamp=1.0)
    for side, value in (("left", 0.1), ("right", -0.2)):
        command.add_command(
            EffectorCommand(
                effector_name=f"{side}_arm",
                side=side,
                command_type="joint_vector",
                value=np.full(6, value),
                metadata={"solver_output_qpos": np.full(6, value)},
            )
        )
    return command


def test_fixed_arm_guard_allows_safe_final_pair() -> None:
    checker_calls = []

    def check_qpos_pair_collision(**kwargs):
        checker_calls.append(kwargs)
        return SimpleNamespace(
            safe=True,
            collision_pair_count=12,
            min_distance_m=0.12,
            required_min_distance_m=0.03,
        )

    fault = fixed_arm_final_collision_fault(
        _command(),
        fix_arm_controller=SimpleNamespace(fixed=True, fixed_side="left"),
        dual_arm_retargeter=SimpleNamespace(check_qpos_pair_collision=check_qpos_pair_collision),
        now=2.0,
    )

    assert fault is None
    assert len(checker_calls) == 1
    np.testing.assert_allclose(checker_calls[0]["left_qpos"], np.full(6, 0.1))
    np.testing.assert_allclose(checker_calls[0]["right_qpos"], np.full(6, -0.2))


def test_fixed_arm_guard_fails_closed_when_final_pair_is_unsafe() -> None:
    fault = fixed_arm_final_collision_fault(
        _command(),
        fix_arm_controller=SimpleNamespace(fixed=True, fixed_side="right"),
        dual_arm_retargeter=SimpleNamespace(
            check_qpos_pair_collision=lambda **_kwargs: SimpleNamespace(
                safe=False,
                reason="collision_margin_low",
                collision_pair_count=12,
                min_distance_m=0.01,
                required_min_distance_m=0.03,
            )
        ),
        now=2.0,
    )

    assert fault is not None
    assert fault.code == "fix_arm_final_collision_check_failed"
    assert fault.metadata["fixed_side"] == "right"
    assert fault.metadata["collision_pair_count"] == 12
    assert fault.metadata["reason"] == "collision_margin_low"

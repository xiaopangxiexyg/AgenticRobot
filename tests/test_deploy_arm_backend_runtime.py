from __future__ import annotations

import ast
import inspect

from teleop_runtime.deploy import arm_backend_runtime, run_policy


COMPATIBILITY_EXPORTS = (
    "annotate_deploy_arm_command_limits",
    "apply_arm_step_dt_defaults",
    "build_arm_backends",
    "deploy_arm_backend_control_dt_sec",
    "deploy_arm_command_max_joint_step",
    "deploy_arm_command_step_dt_sec",
    "effective_arm_canfd_follow",
    "effective_arm_control_hz",
    "effective_position_reference_hz",
    "resolve_arm_step_dt_max_scale",
)


def test_run_policy_reexports_deployment_arm_backend_owner() -> None:
    for name in COMPATIBILITY_EXPORTS:
        assert getattr(run_policy, name) is getattr(arm_backend_runtime, name)
    assert (
        run_policy._resolve_position_reference_and_command_hz
        is arm_backend_runtime.resolve_position_reference_and_command_hz
    )
    assert (
        run_policy._position_deploy_ik_tuning
        is arm_backend_runtime._position_deploy_ik_tuning
    )


def test_deployment_arm_backend_owner_does_not_control_resource_lifecycle() -> None:
    tree = ast.parse(inspect.getsource(arm_backend_runtime))
    forbidden_operations = {"close", "hold_motion", "send_command", "start"}
    called_operations = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called_operations.isdisjoint(forbidden_operations)

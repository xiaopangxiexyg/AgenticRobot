from __future__ import annotations

import ast
import inspect

from teleop_runtime.deploy import hand_backend_runtime, run_policy


def test_run_policy_reexports_hand_startup_diagnostics() -> None:
    assert (
        run_policy._disconnected_gripper_selection_label
        is hand_backend_runtime.disconnected_gripper_selection_label
    )
    assert (
        run_policy._hand_backend_startup_diagnostic
        is hand_backend_runtime.hand_backend_startup_diagnostic
    )
    assert (
        run_policy._start_hand_backends_checked
        is hand_backend_runtime.start_hand_backends_checked
    )


def test_run_policy_no_longer_constructs_concrete_hand_backends() -> None:
    tree = ast.parse(inspect.getsource(run_policy))
    constructor_names = {
        "DMTacGripperBackend",
        "InspireTacHandBackend",
        "InspireTacHandBackendConfig",
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(constructor_names)

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import teleop_runtime
import teleop_runtime.deploy as deploy_namespace
from teleop_runtime.registry import (
    backend_registry,
    retargeter_registry,
    rig_registry,
    source_registry,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_package_and_pure_rm65_contract_import_do_not_load_vendor_sdk() -> None:
    script = """
import json
import sys
import teleop_runtime
import teleop_runtime.deploy.rm65_candidate_manifest
import teleop_runtime.target_runtime_preparation
import teleop_runtime.backends.arms.rm65_dispatch_inventory

forbidden_prefixes = (
    "hardware.rm65.Robotic_Arm",
    "hardware.rm65.rm_api2_adapter",
)
print(json.dumps(sorted(
    name
    for name in sys.modules
    if name.startswith(forbidden_prefixes)
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == []


def test_lazy_top_level_and_deploy_exports_preserve_object_identity() -> None:
    from teleop_runtime.backends.arms.rm65 import RM65ArmBackend
    from teleop_runtime.deploy.actions import ActionChunkBuffer

    assert "RM65ArmBackend" in teleop_runtime.__all__
    assert "RM65ArmBackend" in dir(teleop_runtime)
    assert teleop_runtime.RM65ArmBackend is RM65ArmBackend
    assert deploy_namespace.ActionChunkBuffer is ActionChunkBuffer


def test_explicit_registry_use_preserves_builtin_component_discovery() -> None:
    assert backend_registry.names() == (
        "dmtac_gripper",
        "inspire_tac_hand",
        "rm65_arm",
        "rm65_pose_impedance_arm",
    )
    assert source_registry.names() == ("realsense_rgb",)
    assert retargeter_registry.names() == (
        "dual_arm_collision_aware",
        "joints_to_dexhand",
        "pose_to_arm",
        "scalar_to_gripper",
    )
    assert rig_registry.names() == ()

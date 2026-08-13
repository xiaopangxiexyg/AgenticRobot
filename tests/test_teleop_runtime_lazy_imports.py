from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import teleop_runtime
import teleop_runtime.deploy as deploy_namespace
import teleop_runtime.deploy.rm65_candidate_manifest as legacy_rm65_candidate
import teleop_runtime.deploy.rm65_execution_evidence as legacy_rm65_evidence
import teleop_runtime.deploy.rm65_target_runtime_provider as legacy_rm65_target
import teleop_runtime.target_profiles as legacy_target_profiles
import teleop_runtime.target_runtime as target_runtime_namespace
import teleop_runtime.target_runtime_preparation as legacy_target_preparation
from teleop_runtime.backends.arms import (
    rm65_candidate_manifest as canonical_rm65_candidate,
)
from teleop_runtime.backends.arms import (
    rm65_execution_evidence as canonical_rm65_evidence,
)
from teleop_runtime.target_runtime.integrations import (
    rm65_policy_arm_qpos_candidate as canonical_rm65_target,
)
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
import teleop_runtime.target_runtime
import teleop_runtime.target_runtime_preparation
import teleop_runtime.backends.arms.rm65_dispatch_inventory

forbidden_prefixes = (
    "hardware.rm65.Robotic_Arm",
    "hardware.rm65.rm_api2_adapter",
)
print(json.dumps({
    "forbidden": sorted(
        name
        for name in sys.modules
        if name.startswith(forbidden_prefixes)
    ),
    "integration_loaded": (
        "teleop_runtime.target_runtime.integrations."
        "rm65_policy_arm_qpos_candidate" in sys.modules
    ),
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == {
        "forbidden": [],
        "integration_loaded": False,
    }


def test_target_runtime_compatibility_exports_preserve_object_identity() -> None:
    assert legacy_target_profiles.TargetRuntimeProfileCatalog is (
        target_runtime_namespace.TargetRuntimeProfileCatalog
    )
    assert legacy_target_preparation.prepare_target_runtime_handle is (
        target_runtime_namespace.prepare_target_runtime_handle
    )
    assert legacy_rm65_target.RM65PolicyArmQposCandidateTargetProvider is (
        canonical_rm65_target.RM65PolicyArmQposCandidateTargetProvider
    )
    assert legacy_rm65_candidate.build_rm65_policy_arm_qpos_candidate_manifest is (
        canonical_rm65_candidate.build_rm65_policy_arm_qpos_candidate_manifest
    )
    assert legacy_rm65_evidence.RM65ExecutionEvidenceBundle is (
        canonical_rm65_evidence.RM65ExecutionEvidenceBundle
    )
    assert deploy_namespace.RM65ExecutionEvidenceBundle is (
        canonical_rm65_evidence.RM65ExecutionEvidenceBundle
    )


def test_rm65_backend_contract_import_does_not_reverse_import_deploy() -> None:
    script = """
import json
import sys
import teleop_runtime.backends.arms.rm65_idle_worker_preflight_binding

print(json.dumps(sorted(
    name
    for name in sys.modules
    if name == "teleop_runtime.deploy"
    or name.startswith("teleop_runtime.deploy.")
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

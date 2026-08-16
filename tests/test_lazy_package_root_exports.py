from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_LAZY_COMPATIBILITY_PACKAGES = (
    "control_station",
    "dataset_contracts",
    "dataset_interface",
    "collection_runtime.progress",
)


def _run_isolated_python(source: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=_REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_lazy_package_root_exports_resolve_to_their_original_owners() -> None:
    package_names = json.dumps(_LAZY_COMPATIBILITY_PACKAGES)
    source = f"""
from importlib import import_module

for package_name in {package_names}:
    package = import_module(package_name)
    exports = package._LAZY_EXPORTS
    assert set(package.__all__).issubset(exports)
    for public_name, (module_name, attribute_name) in exports.items():
        owner = import_module(module_name, package.__name__)
        assert getattr(package, public_name) is getattr(owner, attribute_name)
"""
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr


def test_general_mujoco_deploy_import_does_not_load_composition_runtime() -> None:
    source = """
import sys

import sim.mujoco.deploy as deploy

assert "control_composition" not in sys.modules
assert "MujocoCompositionRuntimeConfig" not in deploy.__dict__
assert "MujocoExecutionDifferenceObserver" not in deploy.__dict__
assert deploy.MujocoDeployRuntime.__module__ == "sim.mujoco.deploy.runtime"

config_type = deploy.MujocoCompositionRuntimeConfig
assert config_type.__module__ == "sim.mujoco.deploy.composition_binding"
assert "control_composition" in sys.modules
"""
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr


def test_agent_skill_target_legacy_path_is_identity_preserving() -> None:
    source = """
from sim.mujoco.deploy import sim_hil_target
from sim.mujoco.deploy.agent_skill_target import (
    MujocoAgentSkillMotionTarget,
    build_mujoco_agent_skill_motion_target,
)

assert "MujocoAgentSkillMotionTarget" not in sim_hil_target.__dict__
assert "build_mujoco_agent_skill_motion_target" not in sim_hil_target.__dict__
assert sim_hil_target.MujocoAgentSkillMotionTarget is MujocoAgentSkillMotionTarget
assert (
    sim_hil_target.build_mujoco_agent_skill_motion_target
    is build_mujoco_agent_skill_motion_target
)
"""
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr

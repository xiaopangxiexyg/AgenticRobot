from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


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


@pytest.mark.parametrize(
    "package_name,named_only",
    [
        ("dataset_contracts", [
            "EPISODE_REVIEW_REQUEST_ANNOTATION_SCHEMA_V1",
            "EpisodeReviewRequestAnnotationV1",
            "PROGRESS_ARM_CONTROL_MODES_V1",
            "PROGRESS_FIXED_SCOPES_V2",
            "PROGRESS_FIXED_SCOPE_ARM_AND_END_EFFECTOR_V1",
            "PROGRESS_FIXED_SCOPE_ARM_ONLY_V2",
            "PROGRESS_STEP_ARM_CONTROL_SCHEMA_V1",
            "PROGRESS_STEP_ARM_CONTROL_SCHEMA_V2",
            "STAGE_ANNOTATION_MODES_V1",
        ]),
        ("dataset_interface", [
            "assert_dataset_capture_integrity_compatible",
            "load_latest_detail_stage_annotation_set",
            "resolve_latest_detail_stage_annotation_view",
        ]),
        ("collection_runtime.progress", []),
    ],
)
def test_package_root_keeps_lazy_cache_and_wildcard_boundary(package_name, named_only):
    source = f"""
from importlib import import_module
from importlib.util import resolve_name
from types import SimpleNamespace
import sys

package_name = {package_name!r}
named_only = {named_only!r}
package = import_module(package_name)
exports = package._LAZY_EXPORTS
public = list(package.__all__)
assert isinstance(package.__all__, list)
assert len(public) == len(set(public))
assert set(exports) - set(public) == set(named_only)
owner_modules = {{resolve_name(module, package_name) for module, _ in exports.values()}}
assert owner_modules.isdisjoint(sys.modules)
before_dir = dir(package)
assert set(exports).issubset(before_dir)
assert owner_modules.isdisjoint(sys.modules)

# Exercise the facade without loading every optional owner a second time.
# The original-owner test above checks real imports and all object identities.
owners = {{}}
for module_name, attribute_name in exports.values():
    owner = owners.setdefault(module_name, SimpleNamespace())
    setattr(owner, attribute_name, object())
calls = []
def import_owner(module_name, package_arg):
    assert package_arg == package_name
    calls.append(module_name)
    return owners[module_name]
package.import_module = import_owner

first = public[0]
value = getattr(package, first)
module_name, attribute_name = exports[first]
assert value is getattr(owners[module_name], attribute_name)
assert package.__dict__[first] is value
assert calls == [module_name]
assert getattr(package, first) is value
assert calls == [module_name]
try:
    getattr(package, "_missing_export_for_test")
except AttributeError as exc:
    assert str(exc) == f"module {{package_name!r}} has no attribute '_missing_export_for_test'"
    assert exc.__cause__ is None
else:
    raise AssertionError("unknown export was accepted")
assert calls == [module_name]

wildcard = {{}}
exec(f"from {{package_name}} import *", wildcard)
assert [name for name in wildcard if name != "__builtins__"] == public
assert set(named_only).isdisjoint(wildcard)
for name in named_only:
    module_name, attribute_name = exports[name]
    assert getattr(package, name) is getattr(owners[module_name], attribute_name)
    assert name in package.__dict__
cached_calls = list(calls)
after_cache = {{}}
exec(f"from {{package_name}} import *", after_cache)
assert [name for name in after_cache if name != "__builtins__"] == public
assert set(named_only).isdisjoint(after_cache)
assert calls == cached_calls
assert dir(package) == before_dir
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


def test_target_measurement_legacy_exports_are_identity_preserving() -> None:
    source = """
import sim.mujoco.deploy as deploy
from sim.mujoco.deploy import sim_hil_target
from sim.mujoco.deploy import target_measurement

names = (
    "build_mujoco_sim_hil_coordinate_profiles",
    "build_mujoco_sim_hil_pose_observation_projector",
    "capture_mujoco_sim_hil_safety_anchor",
    "capture_mujoco_sim_hil_tool_tcp_pose_by_side",
)
for name in names:
    owner_value = getattr(target_measurement, name)
    assert getattr(sim_hil_target, name) is owner_value
    assert getattr(deploy, name) is owner_value
"""
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr


def test_target_preparation_legacy_exports_are_identity_preserving() -> None:
    source = """
import sim.mujoco.deploy as deploy
from sim.mujoco.deploy import sim_hil_target
from sim.mujoco.deploy import target_config
from sim.mujoco.deploy import target_preparation

assert (
    target_preparation.MujocoSimHILTargetConfig
    is target_config.MujocoSimHILTargetConfig
)
names = (
    "MujocoSimHILPreparedTarget",
    "MujocoSimHILTargetConfig",
    "prepare_mujoco_sim_hil_target",
)
for name in names:
    owner_value = getattr(target_preparation, name)
    assert getattr(sim_hil_target, name) is owner_value
    assert getattr(deploy, name) is owner_value
"""
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr

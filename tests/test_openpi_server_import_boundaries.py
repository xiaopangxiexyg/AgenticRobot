from __future__ import annotations

import ast
import importlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]


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


def _declared_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else (node.target,)
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return names


def test_lazy_export_maps_reference_symbols_declared_by_their_owner() -> None:
    for package_name in (
        "deployments.client",
        "deployments.openpi.client",
        "robot_motion",
    ):
        package = importlib.import_module(package_name)
        for public_name, relative_module in package._EXPORTS.items():
            module_name = importlib.util.resolve_name(
                relative_module,
                package_name,
            )
            spec = importlib.util.find_spec(module_name)
            assert spec is not None and spec.origin is not None, module_name
            assert public_name in _declared_module_names(Path(spec.origin)), (
                f"{package_name}.{public_name} is not declared by {module_name}"
            )


def test_openpi_server_and_checkpoint_tools_do_not_import_control_stacks() -> None:
    source = r'''
import importlib
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile

from dataset_contracts.normalization import NormalizationContract
from dataset_contracts.policy_artifact import PolicyArtifactContract
from dataset_contracts.v22 import PolicyIOLayout

blocked = (
    "control_station",
    "hardware",
    "osqp",
    "sim",
    "teleop_runtime",
    "robot_motion.position_control",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(
            tuple(prefix + "." for prefix in blocked)
        ):
            raise ImportError("forbidden lightweight-tool import: " + fullname)
        return None

sys.meta_path.insert(0, Blocker())

module_names = (
    "deployments.policy_artifact",
    "deployments.openpi.artifact_identity",
    "deployments.openpi.prepared_deployment",
    "deployments.openpi.prepared_server_runtime",
    "deployments.openpi.server",
    "deployments.openpi.training.evidence",
    "deployments.openpi.training.prepare",
    "deployments.openpi.prepare_camera_compat",
    "deployments.openpi.training.launch",
)
for module_name in module_names:
    importlib.import_module(module_name)

from deployments.openpi import server

with tempfile.TemporaryDirectory() as temp_dir:
    root = Path(temp_dir)
    checkpoint = root / "checkpoints" / "model" / "10"
    checkpoint.mkdir(parents=True)
    artifact = PolicyArtifactContract(
        policy_io_layout=PolicyIOLayout.legacy_v21_q6(),
        model_id="model",
        checkpoint_step=10,
        metadata={
            "normalization_contract": NormalizationContract.identity().to_dict(),
        },
    )
    (checkpoint / "policy_artifact.json").write_text(
        json.dumps(artifact.to_dict()),
        encoding="utf-8",
    )

    openpi_root = root / "openpi"
    (openpi_root / "scripts").mkdir(parents=True)
    (openpi_root / "configs").mkdir()
    (openpi_root / "scripts" / "serve_dexteleop_policy.py").write_text(
        "raise SystemExit('dry-run must not execute this file')\n",
        encoding="utf-8",
    )
    (openpi_root / "configs" / "dexteleop_policy_registry.json").write_text(
        '{"models": {}}',
        encoding="utf-8",
    )

    assert server.main(
        [
            "--checkpoint-dir",
            str(checkpoint),
            "--openpi-root",
            str(openpi_root),
            "--dry-run",
        ]
    ) == 0

assert "osqp" not in sys.modules
assert "robot_motion.position_control" not in sys.modules
'''
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr


def test_package_root_compatibility_exports_are_lazy() -> None:
    source = r'''
import importlib.abc
import sys

blocked = (
    "hardware",
    "osqp",
    "teleop_runtime",
    "robot_motion.position_control",
)

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in blocked or fullname.startswith(
            tuple(prefix + "." for prefix in blocked)
        ):
            raise ImportError("forbidden package-root import: " + fullname)
        return None

sys.meta_path.insert(0, Blocker())

import deployments.client as client
import deployments.openpi.client as openpi_client
import robot_motion

assert client.load_policy_artifact.__module__ == "deployments.policy_artifact"
assert (
    openpi_client.openpi_rtc_artifact_hash.__module__
    == "deployments.openpi.artifact_identity"
)
assert robot_motion.ACTION_POSE_MODE_ABSOLUTE == "absolute"
assert "osqp" not in sys.modules
assert "robot_motion.position_control" not in sys.modules
'''
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr

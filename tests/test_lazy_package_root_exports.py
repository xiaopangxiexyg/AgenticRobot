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

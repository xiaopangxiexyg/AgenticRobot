from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
_ANNOTATION_FORBIDDEN_IMPORTS = {
    "control_composition",
    "deployments",
    "hardware",
    "model",
    "robot_motion",
    "sim",
    "teleop_runtime",
}


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


def test_annotation_cli_import_excludes_non_annotation_runtime_packages() -> None:
    source = """
import json
import pathlib
import sys

root = pathlib.Path.cwd().resolve()
import control_station.refinement_annotation.cli

loaded = sorted(
    name
    for name, module in sys.modules.items()
    if getattr(module, "__file__", None)
    and (
        pathlib.Path(module.__file__).resolve() == root
        or root in pathlib.Path(module.__file__).resolve().parents
    )
)
print(json.dumps(loaded))
"""
    result = _run_isolated_python(source)

    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout)
    loaded_roots = {name.split(".", 1)[0] for name in loaded}
    assert not loaded_roots.intersection(_ANNOTATION_FORBIDDEN_IMPORTS)
    assert {
        "collection_runtime",
        "control_station",
        "dataset_contracts",
        "dataset_interface",
    }.issubset(loaded_roots)

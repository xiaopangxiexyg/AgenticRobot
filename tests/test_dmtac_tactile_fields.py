from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from teleop_runtime import dmtac_tactile_fields
from teleop_runtime.backends.grippers import dmtac as dmtac_backend


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_backend_preserves_dmtac_tactile_field_compatibility_exports() -> None:
    for name in dmtac_tactile_fields.__all__:
        assert getattr(dmtac_backend, name) is getattr(dmtac_tactile_fields, name)


def test_cli_field_contract_import_does_not_load_dmtac_backend() -> None:
    script = """
import importlib
import json
import sys

importlib.import_module("teleop_runtime.deploy.cli_value_parsers")
importlib.import_module("teleop_runtime.real_teleop.cli_value_parsers")
print(json.dumps({
    "backend_loaded": "teleop_runtime.backends.grippers.dmtac" in sys.modules,
    "registry_loaded": "teleop_runtime.registry" in sys.modules,
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
        "backend_loaded": False,
        "registry_loaded": False,
    }


def test_normalization_and_calibration_contracts_are_preserved() -> None:
    normalize = dmtac_tactile_fields.normalize_dmtac_tactile_fields

    assert normalize(None) == ("raw",)
    assert normalize("raw, distributed-force, raw") == (
        "raw",
        "distributed_force",
    )
    assert normalize("all") == dmtac_tactile_fields.DMTAC_TACTILE_FIELDS
    assert dmtac_tactile_fields.dmtac_tactile_requires_calibration("raw") is False
    assert (
        dmtac_tactile_fields.dmtac_tactile_requires_calibration("distributed-force")
        is True
    )

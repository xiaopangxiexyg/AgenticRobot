from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCTOR_PATH = REPO_ROOT / "scripts" / "dev_environment_doctor.py"


def _load_doctor_module():
    spec = importlib.util.spec_from_file_location("dev_environment_doctor", DOCTOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_environment_matches_existing_source_roots() -> None:
    doctor = _load_doctor_module()
    config = doctor.load_project_environment()

    assert config.name == "dexteleop310"
    assert config.python == "3.10"
    assert "collection_runtime" in config.package_roots
    assert "control_composition" in config.package_roots
    assert "rm65_inspire_tac" not in config.package_roots
    assert all((REPO_ROOT / name / "__init__.py").is_file() for name in config.package_roots)


def test_full_environment_install_does_not_pin_one_worktree() -> None:
    installer = (REPO_ROOT / "scripts" / "install_conda_env.sh").read_text(
        encoding="utf-8"
    )

    assert 'pip uninstall --yes dexteleop' in installer
    assert '--editable "$REPO_ROOT"' not in installer


def test_personal_todos_are_ignored_and_not_tracked() -> None:
    ignore_lines = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    tracked = subprocess.run(
        ["git", "ls-files", "--", "todos"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "/todos/" in ignore_lines
    assert tracked.stdout == ""


def test_doctor_cli_is_machine_readable_and_does_not_require_hardware() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(DOCTOR_PATH),
            "--json",
            "--skip-lock-check",
            "--skip-dependency-check",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    findings = {item["key"]: item for item in payload["findings"]}
    assert findings["repo.cwd"]["level"] == "PASS"
    assert findings["source.root_resolution"]["level"] == "PASS"
    assert findings["vendor_sdk"]["level"] in {"PASS", "WARN"}


def test_doctor_rejects_execution_outside_the_worktree_root(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(DOCTOR_PATH),
            "--json",
            "--skip-lock-check",
            "--skip-dependency-check",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    payload = json.loads(completed.stdout)
    findings = {item["key"]: item for item in payload["findings"]}
    assert findings["repo.cwd"]["level"] == "FAIL"

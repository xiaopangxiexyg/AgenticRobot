from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_control.framework_boundary import (
    enforce_framework_boundary,
    inspect_framework_boundary,
    load_framework_boundary_policy,
)
from lab_control.workspace import LabError


def _write_policy(
    root: Path,
    *,
    maximum: int = 1024,
    forbidden: list[str] | None = None,
) -> Path:
    policy = root / "integration" / "policy.json"
    policy.parent.mkdir(parents=True, exist_ok=True)
    policy.write_text(
        json.dumps(
            {
                "schema": "lab.framework-boundary.v1",
                "core_globs": ["core/**/*.py"],
                "controlled_globs": ["integration/*.json", "tests/**/*.py"],
                "forbidden_core_literals": forbidden or ["ExampleProduct"],
                "max_controlled_file_bytes": maximum,
            }
        ),
        encoding="utf-8",
    )
    return policy


def test_repository_framework_boundary_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    policy = load_framework_boundary_policy(
        root / "integration" / "policies" / "lab-framework-boundaries.json"
    )
    report = enforce_framework_boundary(root, policy=policy)

    assert report["status"] == "PASSED"
    assert int(report["core_file_count"]) >= 1
    assert int(report["controlled_file_count"]) >= int(report["core_file_count"])


def test_project_literal_is_loaded_from_policy_and_rejected(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "core" / "engine.py").write_text(
        "BACKEND = 'exampleproduct'\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "test_engine.py").write_text("pass\n", encoding="utf-8")
    policy = load_framework_boundary_policy(_write_policy(tmp_path))

    report = inspect_framework_boundary(tmp_path, policy=policy)

    assert report["status"] == "FAILED"
    assert any("project-specific literal" in value for value in report["violations"])
    with pytest.raises(LabError, match="framework boundary failed"):
        enforce_framework_boundary(tmp_path, policy=policy)


def test_large_file_and_symlink_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "core").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "core" / "engine.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests" / "test_engine.py").write_bytes(b"x" * 65)
    (tmp_path / "tests" / "link.py").symlink_to(tmp_path / "core" / "engine.py")
    policy = load_framework_boundary_policy(_write_policy(tmp_path, maximum=64))

    report = inspect_framework_boundary(tmp_path, policy=policy)

    assert report["status"] == "FAILED"
    assert any("exceeds 64 bytes" in value for value in report["violations"])
    assert any("must not be a symlink" in value for value in report["violations"])


def test_unsafe_policy_glob_is_rejected(tmp_path: Path) -> None:
    policy = _write_policy(tmp_path)
    payload = json.loads(policy.read_text(encoding="utf-8"))
    payload["core_globs"] = ["../outside/**/*.py"]
    policy.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="unsafe framework boundary glob"):
        load_framework_boundary_policy(policy)

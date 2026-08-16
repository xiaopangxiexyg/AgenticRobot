from __future__ import annotations

import json
import stat
import subprocess
from pathlib import Path

import pytest

from lab_control import git
from lab_control.control_install import (
    inspect_control_plane_install,
    install_control_plane,
    plan_control_plane_install,
)
from lab_control.runs import control_plane_identity
from lab_control.workspace import LabError


def _run(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = (tmp_path / "managed").resolve()
    repository.mkdir()
    _run("git", "init", "-b", "main", str(repository))
    _run("git", "config", "user.name", "Control Install Test", cwd=repository)
    _run(
        "git",
        "config",
        "user.email",
        "control-install@example.invalid",
        cwd=repository,
    )
    lab = repository / "lab"
    lab.write_text(
        """#!/usr/bin/env python3
import json
import os
import sys
print(json.dumps({"argv": sys.argv[1:], "control": os.environ.get("LAB_CONTROL_REPOSITORY_ROOT"), "manifest": os.environ.get("LAB_CONTROL_PLANE_INSTALL_MANIFEST")}))
""",
        encoding="utf-8",
    )
    lab.chmod(0o755)
    dev = repository / "dev"
    dev.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    dev.chmod(0o755)
    (repository / "lab_control").mkdir()
    (repository / "lab_control" / "__init__.py").write_text("VALUE = 1\n")
    (repository / "integration").mkdir()
    (repository / "integration" / "project.json").write_text("{}\n")
    (repository / "stack").mkdir()
    (repository / "stack" / "baseline.json").write_text("{}\n")
    _run(
        "git",
        "add",
        "lab",
        "dev",
        "lab_control",
        "integration",
        "stack",
        cwd=repository,
    )
    _run("git", "commit", "-m", "protected control payload", cwd=repository)
    protected = git.current_commit(repository)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "git@example.invalid:Org/Control.git",
        cwd=repository,
    )
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        protected,
        cwd=repository,
    )
    install_root = (tmp_path / "state" / "control-plane").resolve()
    return repository, install_root, protected


def _plan(repository: Path, install_root: Path) -> dict[str, object]:
    return plan_control_plane_install(
        repository,
        managed_control_repository=repository,
        expected_identity="Org/Control",
        protected_branch="main",
        install_root=install_root,
    )


def test_installs_only_protected_ref_outside_checkout_and_runs_wrapper(
    tmp_path: Path,
) -> None:
    repository, install_root, protected = _fixture(tmp_path)
    (repository / "task-only.txt").write_text("not protected\n", encoding="utf-8")
    _run("git", "add", "task-only.txt", cwd=repository)
    _run("git", "commit", "-m", "untrusted task head", cwd=repository)
    assert git.current_commit(repository) != protected
    plan = _plan(repository, install_root)

    assert plan["commit"] == protected
    assert plan["action"] == "INSTALL_READY"
    receipt = install_control_plane(
        plan,
        confirmed_plan_sha256=str(plan["plan_sha256"]),
    )

    assert receipt["installed_new"] is True
    assert receipt["network_used"] is False
    assert receipt["task_payload_authorized"] is False
    version = install_root / "versions" / protected
    assert (version / "payload" / "lab").is_file()
    assert not (version / "payload" / "task-only.txt").exists()
    assert (install_root / "current").resolve() == version
    assert not bool((version / "payload" / "lab").stat().st_mode & stat.S_IWUSR)
    completed = subprocess.run(
        [str(install_root / "bin" / "lab"), "probe"],
        check=True,
        capture_output=True,
        text=True,
    )
    output = json.loads(completed.stdout)
    assert output["argv"] == ["probe"]
    assert output["control"] == str(repository)
    assert output["manifest"] == str(install_root / "current" / "install.json")
    wrapper = (install_root / "bin" / "lab").read_text(encoding="utf-8")
    assert 'LAB_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/profiles"' in wrapper
    assert (
        'LAB_QUEUE_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/queue-profiles"'
        in wrapper
    )
    assert (
        'LAB_RUNNER_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/runners"'
        in wrapper
    )
    assert (
        'LAB_ENVIRONMENT_PROFILE_ROOT="$CONTROL_ROOT/current/payload/integration/environments"'
        in wrapper
    )
    assert 'LAB_IMPACT_RULES="$CONTROL_ROOT/current/payload/stack/impact-rules.json"' in wrapper
    status = inspect_control_plane_install(install_root)
    assert status["status"] == "READY"
    assert status["current_commit"] == protected
    (repository / "lab_control" / "__init__.py").write_text(
        "VALUE = 999\n", encoding="utf-8"
    )
    with pytest.raises(LabError, match="outside its active payload"):
        control_plane_identity(
            repository,
            install_manifest_path=install_root / "current" / "install.json",
        )
    installed_identity = control_plane_identity(
        repository,
        install_manifest_path=install_root / "current" / "install.json",
        _executing_payload_root=install_root / "versions" / protected / "payload",
    )
    direct_identity = control_plane_identity(repository)
    assert installed_identity["source_commit"] == protected
    assert installed_identity["source_dirty"] is False
    assert installed_identity["install_manifest_sha256"] == status["manifest"][
        "manifest_sha256"
    ]
    assert installed_identity["executor_sha256"] != direct_identity["executor_sha256"]
    assert direct_identity["source_dirty"] is True

    noop_plan = _plan(repository, install_root)
    assert noop_plan["action"] == "NOOP"
    noop = install_control_plane(
        noop_plan,
        confirmed_plan_sha256=str(noop_plan["plan_sha256"]),
    )
    assert noop["installed_new"] is False
    assert noop["activated"] is True

    installed_lab = version / "payload" / "lab"
    installed_lab.chmod(0o755)
    installed_lab.write_text("tampered\n", encoding="utf-8")
    installed_lab.chmod(0o555)
    with pytest.raises(LabError, match="payload hash mismatch"):
        inspect_control_plane_install(install_root)


def test_refuses_tampered_existing_version_and_install_inside_checkout(
    tmp_path: Path,
) -> None:
    repository, install_root, protected = _fixture(tmp_path)
    plan = _plan(repository, install_root)
    install_control_plane(plan, confirmed_plan_sha256=str(plan["plan_sha256"]))
    manifest = install_root / "versions" / protected / "install.json"
    manifest.chmod(0o600)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["tree_oid"] = "f" * 40
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    manifest.chmod(0o444)
    tampered_plan = _plan(repository, install_root)
    with pytest.raises(LabError, match="manifest hash mismatch"):
        install_control_plane(
            tampered_plan,
            confirmed_plan_sha256=str(tampered_plan["plan_sha256"]),
        )

    with pytest.raises(LabError, match="outside the managed checkout"):
        plan_control_plane_install(
            repository,
            managed_control_repository=repository,
            expected_identity="Org/Control",
            protected_branch="main",
            install_root=repository / ".control-plane",
        )


def test_archive_refuses_symlink_payload(tmp_path: Path) -> None:
    repository, install_root, _ = _fixture(tmp_path)
    (repository / "lab_control" / "escape.py").symlink_to("/etc/passwd")
    _run("git", "add", "lab_control/escape.py", cwd=repository)
    _run("git", "commit", "-m", "protected symlink", cwd=repository)
    protected = git.current_commit(repository)
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        protected,
        cwd=repository,
    )
    plan = _plan(repository, install_root)

    with pytest.raises(LabError, match="only files/directories"):
        install_control_plane(
            plan,
            confirmed_plan_sha256=str(plan["plan_sha256"]),
        )
    assert not (install_root / "current").exists()

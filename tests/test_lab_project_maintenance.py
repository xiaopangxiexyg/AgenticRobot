from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lab_control import git
from lab_control.project_maintenance import apply_guarded_project_checkout
from lab_control.project_sync import plan_project_sync
from lab_control.runtime_inventory import collect_checkout_runtime_inventory
from lab_control.workspace import LabError, WorkspaceManager


def _run(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, identity: str) -> Path:
    path.mkdir()
    _run("git", "init", "-b", "main", str(path))
    _run("git", "config", "user.name", "Project Maintenance Test", cwd=path)
    _run(
        "git",
        "config",
        "user.email",
        "project-maintenance@example.invalid",
        cwd=path,
    )
    (path / "state.txt").write_text("base\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=path)
    _run("git", "commit", "-m", "base", cwd=path)
    _run("git", "remote", "add", "origin", f"git@example.invalid:{identity}.git", cwd=path)
    _run("git", "update-ref", "refs/remotes/origin/main", "HEAD", cwd=path)
    return path.resolve()


def _manager(tmp_path: Path) -> tuple[WorkspaceManager, Path, Path]:
    control = _repository(tmp_path / "control", "Org/Control")
    component = _repository(tmp_path / "component", "Org/Component")
    base = git.current_commit(component)
    _run("git", "switch", "-c", "incoming", cwd=component)
    (component / "state.txt").write_text("incoming\n", encoding="utf-8")
    _run("git", "commit", "-am", "incoming", cwd=component)
    incoming = git.current_commit(component)
    _run("git", "switch", "main", cwd=component)
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        incoming,
        cwd=component,
    )
    project_path = tmp_path / "project.json"
    project_path.write_text(
        json.dumps(
            {
                "schema": "lab.project.v3",
                "project_id": "generic-stack",
                "control_repository": "control",
                "default_development_host": "test-host",
                "repositories": {
                    "control": {
                        "aliases": [],
                        "checkout_mode": "stable_branch",
                        "default_path": ".",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_CONTROL",
                        "stable_branch": "main",
                    },
                    "component": {
                        "aliases": [],
                        "checkout_mode": "stable_branch",
                        "default_path": "component",
                        "path_base": "control_repository_parent",
                        "path_environment": "LAB_REPOSITORY_COMPONENT",
                        "stable_branch": "main",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v2",
                "baseline_id": "baseline-test",
                "repositories": {
                    "control": {
                        "canonical_identity": "Org/Control",
                        "commit": git.current_commit(control),
                        "tree_oid": git.tree_oid(control),
                        "submodules": {},
                    },
                    "component": {
                        "canonical_identity": "Org/Component",
                        "commit": base,
                        "tree_oid": git.tree_oid(component, base),
                        "submodules": {},
                    },
                },
                "repository_links": [],
            }
        ),
        encoding="utf-8",
    )
    manager = WorkspaceManager(
        control_repository=control,
        environ={
            "LAB_HOME": str(tmp_path / "lab-home"),
            "LAB_PROJECT_MANIFEST": str(project_path),
            "LAB_REPOSITORY_CONTROL": str(control),
            "LAB_REPOSITORY_COMPONENT": str(component),
        },
    )
    return manager, baseline_path, component


def test_project_apply_requires_confirmed_plan_and_persists_receipt(
    tmp_path: Path,
) -> None:
    manager, baseline_path, component = _manager(tmp_path)
    plan = plan_project_sync(
        manager,
        baseline_path=baseline_path,
        repository_names=("component",),
    )
    inspection = plan["repositories"]["component"]
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()

    def inventory(checkout: Path) -> dict[str, object]:
        return collect_checkout_runtime_inventory(
            checkout,
            proc_root=proc,
            scanner_euid=os.geteuid(),
        )

    with pytest.raises(LabError, match="confirmed.*does not match"):
        apply_guarded_project_checkout(
            manager,
            plan,
            repository="component",
            confirmed_plan_sha256="0" * 64,
            reservation_id="maint-project-001",
            inventory_collector=inventory,
        )
    receipt = apply_guarded_project_checkout(
        manager,
        plan,
        repository="component",
        confirmed_plan_sha256=str(inspection["checkout_sync_plan_sha256"]),
        reservation_id="maint-project-002",
        inventory_collector=inventory,
    )

    assert receipt["status"] == "COMPLETE"
    assert receipt["network_used"] is False
    assert receipt["submodules_updated"] is False
    assert receipt["task_worktrees_updated"] is False
    assert Path(str(receipt["receipt_path"])).is_file()
    assert git.current_commit(component) == inspection["remote_commit"]
    assert git.status_short(component) == ()

    current_plan = plan_project_sync(
        manager,
        baseline_path=baseline_path,
        repository_names=("component",),
    )
    current_inspection = current_plan["repositories"]["component"]
    noop = apply_guarded_project_checkout(
        manager,
        current_plan,
        repository="component",
        confirmed_plan_sha256=str(
            current_inspection["checkout_sync_plan_sha256"]
        ),
        reservation_id="maint-project-003",
        inventory_collector=inventory,
    )
    assert noop["status"] == "UP_TO_DATE"
    assert noop["checkout_updated"] is False
    assert noop["guarded_sync"] is None


def test_project_apply_refuses_any_repository_in_a_gitlink_relationship(
    tmp_path: Path,
) -> None:
    manager, baseline_path, _ = _manager(tmp_path)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline["repository_links"] = [
        {
            "parent_repository": "control",
            "submodule_path": "components/component",
            "child_repository": "component",
        }
    ]
    baseline["repositories"]["control"]["submodules"] = {
        "components/component": baseline["repositories"]["component"]["commit"]
    }
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    plan = plan_project_sync(
        manager,
        baseline_path=baseline_path,
        repository_names=("component",),
    )
    inspection = plan["repositories"]["component"]

    with pytest.raises(LabError, match="refuses linked repositories"):
        apply_guarded_project_checkout(
            manager,
            plan,
            repository="component",
            confirmed_plan_sha256=str(inspection["checkout_sync_plan_sha256"]),
            reservation_id="maint-project-004",
        )

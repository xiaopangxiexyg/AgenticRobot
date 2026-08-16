from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lab_control import git
from lab_control.linked_sync import (
    apply_linked_project_checkout,
    plan_linked_project_checkout,
)
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


def _commit(repository: Path, path: str, content: str, message: str) -> str:
    target = repository / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    _run("git", "add", path, cwd=repository)
    _run("git", "commit", "-m", message, cwd=repository)
    return _run("git", "rev-parse", "HEAD", cwd=repository)


def _fixture(tmp_path: Path) -> dict[str, object]:
    remotes = (tmp_path / "remotes").resolve()
    child_remote = remotes / "Org" / "Child.git"
    parent_remote = remotes / "Org" / "Parent.git"
    child_remote.parent.mkdir(parents=True)
    _run("git", "init", "--bare", str(child_remote))
    _run("git", "init", "--bare", str(parent_remote))

    child_seed = (tmp_path / "child-seed").resolve()
    _run("git", "clone", str(child_remote), str(child_seed))
    _run("git", "switch", "-c", "main", cwd=child_seed)
    _run("git", "config", "user.name", "Linked Sync Test", cwd=child_seed)
    _run("git", "config", "user.email", "linked@example.invalid", cwd=child_seed)
    child_base = _commit(child_seed, "child.py", "VALUE = 1\n", "child base")
    _run("git", "push", "-u", "origin", "main", cwd=child_seed)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=child_remote)

    parent_seed = (tmp_path / "parent-seed").resolve()
    _run("git", "clone", str(parent_remote), str(parent_seed))
    _run("git", "switch", "-c", "main", cwd=parent_seed)
    _run("git", "config", "user.name", "Linked Sync Test", cwd=parent_seed)
    _run("git", "config", "user.email", "linked@example.invalid", cwd=parent_seed)
    _commit(parent_seed, "README.md", "# parent\n", "parent base")
    _run(
        "git",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_remote),
        "components/child",
        cwd=parent_seed,
    )
    _run("git", "-C", "components/child", "checkout", child_base, cwd=parent_seed)
    _run("git", "add", ".gitmodules", "components/child", cwd=parent_seed)
    _run("git", "commit", "-m", "pin child base", cwd=parent_seed)
    parent_base = git.current_commit(parent_seed)
    _run("git", "push", "-u", "origin", "main", cwd=parent_seed)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=parent_remote)

    checkout = (tmp_path / "checkout" / "parent").resolve()
    checkout.parent.mkdir()
    _run("git", "clone", "--branch", "main", str(parent_remote), str(checkout))
    _run(
        "git",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--no-fetch",
        cwd=checkout,
    )
    child_checkout = (checkout / "components" / "child").resolve()

    child_next = _commit(child_seed, "child.py", "VALUE = 2\n", "child next")
    _run("git", "push", "origin", "main", cwd=child_seed)
    nested_seed = parent_seed / "components" / "child"
    _run("git", "fetch", str(child_remote), "main", cwd=nested_seed)
    _run("git", "checkout", "--detach", child_next, cwd=nested_seed)
    _run("git", "add", "components/child", cwd=parent_seed)
    _run("git", "commit", "-m", "advance child pin", cwd=parent_seed)
    parent_next = git.current_commit(parent_seed)
    _run("git", "push", "origin", "main", cwd=parent_seed)

    _run("git", "fetch", str(parent_remote), "main", cwd=checkout)
    _run("git", "update-ref", "refs/remotes/origin/main", "FETCH_HEAD", cwd=checkout)
    _run("git", "fetch", str(child_remote), "main", cwd=child_checkout)
    _run(
        "git",
        "remote",
        "set-url",
        "origin",
        "git@example.invalid:Org/Parent.git",
        cwd=checkout,
    )
    _run(
        "git",
        "remote",
        "set-url",
        "origin",
        "git@example.invalid:Org/Child.git",
        cwd=child_checkout,
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v2",
                "baseline_id": "linked-baseline-v1",
                "repositories": {
                    "parent": {
                        "canonical_identity": "Org/Parent",
                        "commit": parent_base,
                        "tree_oid": git.tree_oid(checkout, parent_base),
                        "submodules": {"components/child": child_base},
                    },
                    "child": {
                        "canonical_identity": "Org/Child",
                        "commit": child_base,
                        "tree_oid": git.tree_oid(child_checkout, child_base),
                        "submodules": {},
                    },
                },
                "repository_links": [
                    {
                        "parent_repository": "parent",
                        "submodule_path": "components/child",
                        "child_repository": "child",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    project_path = tmp_path / "project.json"
    project_path.write_text(
        json.dumps(
            {
                "schema": "lab.project.v3",
                "project_id": "linked-project",
                "control_repository": "parent",
                "default_development_host": "test-host",
                "repositories": {
                    "parent": {
                        "aliases": [],
                        "checkout_mode": "stable_branch",
                        "default_path": ".",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_PARENT",
                        "stable_branch": "main",
                    },
                    "child": {
                        "aliases": [],
                        "checkout_mode": "pinned_gitlink",
                        "default_path": "components/child",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_CHILD",
                        "stable_branch": "main",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manager = WorkspaceManager(
        control_repository=checkout,
        environ={
            "LAB_HOME": str(tmp_path / "lab-home"),
            "LAB_PROJECT_MANIFEST": str(project_path),
            "LAB_REPOSITORY_PARENT": str(checkout),
            "LAB_REPOSITORY_CHILD": str(child_checkout),
        },
    )
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()

    def inventory(path: Path) -> dict[str, object]:
        return collect_checkout_runtime_inventory(
            path,
            proc_root=proc,
            scanner_euid=os.geteuid(),
        )

    return {
        "manager": manager,
        "baseline": baseline_path,
        "parent": checkout,
        "child": child_checkout,
        "parent_base": parent_base,
        "parent_next": parent_next,
        "child_base": child_base,
        "child_next": child_next,
        "inventory": inventory,
        "proc": proc,
    }


def _plan(fixture: dict[str, object]) -> dict[str, object]:
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    project = plan_project_sync(
        manager,
        baseline_path=Path(str(fixture["baseline"])),
        repository_names=("parent",),
    )
    return plan_linked_project_checkout(
        manager,
        project,
        root_repository="parent",
    )


def test_linked_plan_and_apply_update_parent_then_exact_child(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    plan = _plan(fixture)

    assert plan["action"] == "APPLY_READY"
    assert plan["root"]["head_action"] == "FAST_FORWARD_READY"
    assert plan["children"]["child"]["target_commit"] == fixture["child_next"]
    assert plan["network_used"] is False

    receipt = apply_linked_project_checkout(
        manager,
        plan,
        confirmed_plan_sha256=str(plan["plan_sha256"]),
        reservation_id="linked-001",
        inventory_collector=fixture["inventory"],
    )

    parent = Path(str(fixture["parent"]))
    child = Path(str(fixture["child"]))
    assert receipt["status"] == "COMPLETE"
    assert receipt["recovery_required"] is False
    assert receipt["network_used"] is False
    assert receipt["submodule_fetch_used"] is False
    assert receipt["lfs_smudge_enabled"] is False
    assert git.current_commit(parent) == fixture["parent_next"]
    assert git.current_commit(child) == fixture["child_next"]
    assert git.current_branch(child) is None
    assert git.status_short(parent) == ()
    assert git.status_short(child) == ()
    assert all(
        raw["status"] == "RELEASED"
        for raw in receipt["released_reservations"].values()
    )

    noop_plan = _plan(fixture)
    assert noop_plan["action"] == "NOOP"
    noop = apply_linked_project_checkout(
        manager,
        noop_plan,
        confirmed_plan_sha256=str(noop_plan["plan_sha256"]),
        reservation_id="linked-002",
        inventory_collector=fixture["inventory"],
    )
    assert noop["status"] == "UP_TO_DATE"
    assert noop["reservations"] == {}


def test_partial_child_failure_is_quarantined_and_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    plan = _plan(fixture)
    original_run_git = git.run_git

    def fail_child_checkout(
        repository: Path,
        arguments,
        *,
        check: bool = True,
        environment=None,
    ):
        argv = list(arguments)
        if argv and argv[0] == "checkout":
            raise git.GitError("injected child checkout failure")
        return original_run_git(
            repository,
            argv,
            check=check,
            environment=environment,
        )

    monkeypatch.setattr("lab_control.linked_sync.git.run_git", fail_child_checkout)
    receipt = apply_linked_project_checkout(
        manager,
        plan,
        confirmed_plan_sha256=str(plan["plan_sha256"]),
        reservation_id="linked-003",
        inventory_collector=fixture["inventory"],
    )

    parent = Path(str(fixture["parent"]))
    child = Path(str(fixture["child"]))
    assert receipt["status"] == "QUARANTINED"
    assert receipt["recovery_required"] is True
    assert "injected child checkout failure" in str(receipt["error"])
    assert git.current_commit(parent) == fixture["parent_next"]
    assert git.current_commit(child) == fixture["child_base"]
    assert git.status_short(parent) != ()
    assert all(
        raw["status"] == "RELEASED"
        for raw in receipt["released_reservations"].values()
    )

    monkeypatch.setattr("lab_control.linked_sync.git.run_git", original_run_git)
    recovery_plan = _plan(fixture)
    assert recovery_plan["root"]["head_action"] == (
        "HEAD_AT_TARGET_WITH_GITLINK_DRIFT"
    )
    recovered = apply_linked_project_checkout(
        manager,
        recovery_plan,
        confirmed_plan_sha256=str(recovery_plan["plan_sha256"]),
        reservation_id="linked-004",
        inventory_collector=fixture["inventory"],
    )
    assert recovered["status"] == "COMPLETE"
    assert git.current_commit(child) == fixture["child_next"]
    assert git.status_short(parent) == ()


def test_linked_preflight_process_blocker_moves_no_checkout_and_releases(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    parent = Path(str(fixture["parent"]))
    child = Path(str(fixture["child"]))
    proc = Path(str(fixture["proc"]))
    process = proc / "101"
    process.mkdir()
    uid = os.geteuid()
    (process / "status").write_text(
        f"Name:\ttest\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="utf-8",
    )
    (process / "cwd").symlink_to(parent)
    (process / "exe").symlink_to("/usr/bin/python3")
    (process / "cmdline").write_bytes(b"python\0worker.py\0")
    (process / "environ").write_bytes(b"PATH=/usr/bin\0")
    (process / "maps").write_bytes(b"")
    plan = _plan(fixture)

    with pytest.raises(LabError, match="active processes"):
        apply_linked_project_checkout(
            manager,
            plan,
            confirmed_plan_sha256=str(plan["plan_sha256"]),
            reservation_id="linked-005",
            inventory_collector=fixture["inventory"],
        )

    assert git.current_commit(parent) == fixture["parent_base"]
    assert git.current_commit(child) == fixture["child_base"]
    reservation_root = (
        manager.state_root
        / "maintenance-reservations"
        / manager.project.project_id
    )
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in reservation_root.glob("*.json")
    ]
    assert len(states) == 2
    assert all(state["status"] == "RELEASED" for state in states)

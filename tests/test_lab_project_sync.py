from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from lab_control import git
from lab_control.cli import main
from lab_control.project_sync import fetch_project_sync, plan_project_sync
from lab_control.workspace import LabError, WorkspaceManager
from lab_control.writer_lease import WriterLeaseManager


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
    child_remote = remotes / "TestOrg" / "Component.git"
    parent_remote = remotes / "TestOrg" / "Main.git"
    child_remote.parent.mkdir(parents=True)
    _run("git", "init", "--bare", str(child_remote))
    _run("git", "init", "--bare", str(parent_remote))

    child_seed = (tmp_path / "child-seed").resolve()
    _run("git", "clone", str(child_remote), str(child_seed))
    _run("git", "switch", "-c", "main", cwd=child_seed)
    _run("git", "config", "user.name", "Project Sync Test", cwd=child_seed)
    _run(
        "git",
        "config",
        "user.email",
        "project-sync@example.invalid",
        cwd=child_seed,
    )
    child_base = _commit(child_seed, "component.py", "VALUE = 1\n", "child base")
    _run("git", "push", "-u", "origin", "main", cwd=child_seed)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=child_remote)

    parent_seed = (tmp_path / "parent-seed").resolve()
    _run("git", "clone", str(parent_remote), str(parent_seed))
    _run("git", "switch", "-c", "main", cwd=parent_seed)
    _run("git", "config", "user.name", "Project Sync Test", cwd=parent_seed)
    _run(
        "git",
        "config",
        "user.email",
        "project-sync@example.invalid",
        cwd=parent_seed,
    )
    _commit(parent_seed, "README.md", "# main\n", "parent base")
    _run(
        "git",
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child_remote),
        "components/component",
        cwd=parent_seed,
    )
    _run("git", "-C", "components/component", "checkout", child_base, cwd=parent_seed)
    _run("git", "add", ".gitmodules", "components/component", cwd=parent_seed)
    _run("git", "commit", "-m", "pin child base", cwd=parent_seed)
    parent_base = _run("git", "rev-parse", "HEAD", cwd=parent_seed)
    _run("git", "push", "-u", "origin", "main", cwd=parent_seed)
    _run("git", "symbolic-ref", "HEAD", "refs/heads/main", cwd=parent_remote)

    repositories = (tmp_path / "repositories").resolve()
    repositories.mkdir()
    child = repositories / "component"
    parent = repositories / "main"
    _run("git", "clone", "--branch", "main", str(child_remote), str(child))
    _run("git", "clone", "--branch", "main", str(parent_remote), str(parent))
    _run("git", "checkout", "--detach", child_base, cwd=child)
    _run(
        "git",
        "remote",
        "set-url",
        "origin",
        "git@example.invalid:TestOrg/Component.git",
        cwd=child,
    )
    _run(
        "git",
        "remote",
        "set-url",
        "origin",
        "git@example.invalid:TestOrg/Main.git",
        cwd=parent,
    )

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v2",
                "baseline_id": "sync-baseline-v1",
                "repositories": {
                    "main": {
                        "canonical_identity": "TestOrg/Main",
                        "commit": parent_base,
                        "tree_oid": _run(
                            "git", "rev-parse", f"{parent_base}^{{tree}}", cwd=parent
                        ),
                        "submodules": {"components/component": child_base},
                    },
                    "component": {
                        "canonical_identity": "TestOrg/Component",
                        "commit": child_base,
                        "tree_oid": _run(
                            "git", "rev-parse", f"{child_base}^{{tree}}", cwd=child
                        ),
                        "submodules": {},
                    },
                },
                "repository_links": [
                    {
                        "parent_repository": "main",
                        "submodule_path": "components/component",
                        "child_repository": "component",
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
                "project_id": "sync-project",
                "control_repository": "main",
                "default_development_host": "local-dev-control",
                "repositories": {
                    "main": {
                        "aliases": ["control"],
                        "checkout_mode": "stable_branch",
                        "default_path": "main",
                        "path_base": "control_repository_parent",
                        "path_environment": "LAB_REPOSITORY_MAIN",
                        "stable_branch": "main",
                    },
                    "component": {
                        "aliases": [],
                        "checkout_mode": "pinned_gitlink",
                        "default_path": "component",
                        "path_base": "control_repository_parent",
                        "path_environment": "LAB_REPOSITORY_COMPONENT",
                        "stable_branch": "main",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manager = WorkspaceManager(
        control_repository=parent,
        environ={
            "LAB_HOME": str(tmp_path / "lab-home"),
            "LAB_PROJECT_MANIFEST": str(project_path),
            "LAB_REPOSITORY_MAIN": str(parent),
            "LAB_REPOSITORY_COMPONENT": str(child),
        },
    )

    child_next = _commit(child_seed, "component.py", "VALUE = 2\n", "child next")
    _run("git", "push", "origin", "main", cwd=child_seed)
    nested_child = parent_seed / "components" / "component"
    _run("git", "fetch", str(child_remote), "main", cwd=nested_child)
    _run("git", "checkout", child_next, cwd=nested_child)
    _run("git", "add", "components/component", cwd=parent_seed)
    _run("git", "commit", "-m", "advance child pin", cwd=parent_seed)
    parent_next = _run("git", "rev-parse", "HEAD", cwd=parent_seed)
    _run("git", "push", "origin", "main", cwd=parent_seed)

    remotes_by_repository = {
        parent.resolve(): parent_remote,
        child.resolve(): child_remote,
    }

    def fetch_runner(
        argv: list[str],
        environment: Mapping[str, str],
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        repository = Path(argv[argv.index("-C") + 1]).resolve()
        local_argv = list(argv)
        origin_index = local_argv.index("origin")
        local_argv[origin_index] = str(remotes_by_repository[repository])
        assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert "--no-recurse-submodules" in local_argv
        assert "--no-tags" in local_argv
        return subprocess.run(
            local_argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(environment),
        )

    return {
        "manager": manager,
        "baseline": baseline_path,
        "parent": parent,
        "child": child,
        "parent_base": parent_base,
        "parent_next": parent_next,
        "child_base": child_base,
        "child_next": child_next,
        "fetch_runner": fetch_runner,
    }


def test_project_sync_fetches_dependencies_first_without_moving_worktrees(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    manifest = manager.create_task(
        manager.plan_task(task_id="SYNC-0001", repository_names=["component"])
    )
    task_component = Path(
        str(manifest["repositories"]["component"]["worktree_path"])
    )
    task_before = git.current_commit(task_component)
    component_lease = WriterLeaseManager(manager).claim(
        "SYNC-0001",
        "component",
        actor="agent-component-01",
    )
    parent = Path(str(fixture["parent"]))
    child = Path(str(fixture["child"]))
    parent_before = git.current_commit(parent)
    child_before = git.current_commit(child)

    plan = plan_project_sync(
        manager,
        baseline_path=Path(str(fixture["baseline"])),
        repository_names=("control",),
    )

    assert plan["fetch_order"] == ["component", "main"]
    assert plan["dependency_repositories_added"] == ["component"]
    assert plan["fetch_status"] == "READY"
    assert plan["repositories"]["component"]["checkout_sync_action"] == (
        "PINNED_GITLINK"
    )
    assert plan["repositories"]["component"]["active_writers"] == [
        {
            "task_id": "SYNC-0001",
            "branch": "task/SYNC-0001",
            "host": "local-dev-control",
            "epoch": 1,
            "actor": "agent-component-01",
            "lease_id": component_lease["lease_id"],
            "expires_at": component_lease["expires_at"],
            "worktree_path": str(task_component),
        }
    ]

    receipt = fetch_project_sync(
        manager,
        plan,
        fetch_runner=fixture["fetch_runner"],
    )

    assert receipt["status"] == "COMPLETE"
    assert list(receipt["receipts"]) == ["component", "main"]
    assert receipt["checkout_updated"] is False
    assert receipt["task_worktrees_updated"] is False
    assert receipt["submodules_fetched"] is False
    assert receipt["lfs_smudge_enabled"] is False
    receipt_path = Path(str(receipt["receipt_path"]))
    assert receipt_path.is_file()
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt
    assert git.current_commit(parent) == parent_before == fixture["parent_base"]
    assert git.current_commit(child) == child_before == fixture["child_base"]
    assert git.current_commit(task_component) == task_before
    post = receipt["post_fetch_plan"]
    assert post["checkout_sync_status"] == "FAST_FORWARD_AVAILABLE"
    assert post["repository_links"] == [
        {
            "parent_repository": "main",
            "submodule_path": "components/component",
            "child_repository": "component",
            "remote_parent_commit": fixture["parent_next"],
            "gitlink_commit": fixture["child_next"],
            "child_object_available": True,
            "state": "READY",
        }
    ]


def test_project_sync_fetch_allows_dirty_management_but_never_applies_it(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    parent = Path(str(fixture["parent"]))
    dirty = parent / "local-note.txt"
    dirty.write_text("preserve me\n", encoding="utf-8")

    plan = plan_project_sync(
        manager,
        baseline_path=Path(str(fixture["baseline"])),
    )

    assert plan["repositories"]["main"]["fetch_action"] == "FETCH_READY"
    assert plan["repositories"]["main"]["checkout_sync_action"] == "BLOCKED_DIRTY"
    receipt = fetch_project_sync(
        manager,
        plan,
        fetch_runner=fixture["fetch_runner"],
    )
    assert receipt["status"] == "COMPLETE"
    assert dirty.read_text(encoding="utf-8") == "preserve me\n"
    assert git.current_commit(parent) == fixture["parent_base"]
    assert receipt["post_fetch_plan"]["checkout_sync_status"] == "BLOCKED"


def test_project_v3_rejects_an_unsafe_stable_branch(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    project_path = manager.project.path
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    payload["repositories"]["main"]["stable_branch"] = "../escape"
    project_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="unsafe stable branch"):
        WorkspaceManager(
            control_repository=Path(str(fixture["parent"])),
            environ=manager.environ,
        )


def test_lab_sync_inspect_is_a_read_only_user_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    manager = fixture["manager"]
    assert isinstance(manager, WorkspaceManager)
    for name, value in manager.environ.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("LAB_BASELINE_LOCK", str(fixture["baseline"]))
    monkeypatch.setattr(
        "lab_control.cli._control_repository",
        lambda: Path(str(fixture["parent"])),
    )
    parent_before = git.current_commit(Path(str(fixture["parent"])))
    child_before = git.current_commit(Path(str(fixture["child"])))

    assert main(["sync", "inspect", "--repos", "control", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "lab.project-sync-plan.v1"
    assert payload["fetch_order"] == ["component", "main"]
    assert payload["network_used"] is False
    assert payload["checkout_updated"] is False
    assert git.current_commit(Path(str(fixture["parent"]))) == parent_before
    assert git.current_commit(Path(str(fixture["child"]))) == child_before

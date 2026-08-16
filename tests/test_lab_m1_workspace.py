from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.workspace import LabError, WorkspaceManager
from lab_control.writer_lease import WriterLeaseManager


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Lab Test")
    _git(path, "config", "user.email", "lab-test@example.invalid")
    (path / "README.md").write_text(f"# {path.name}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _write_project_manifest(
    path: Path,
    *,
    repositories: dict[str, dict[str, object]],
) -> Path:
    payload = {
        "schema": "lab.project.v3",
        "project_id": "example-stack",
        "control_repository": "Controller",
        "default_development_host": "developer-a",
        "repositories": repositories,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_framework_core_has_no_current_project_or_runtime_identifiers() -> None:
    repository = Path(__file__).resolve().parents[1]
    forbidden = (
        "agenticrobot",
        "rpent",
        "openpi",
        "robot_motion",
        "dual_rm65",
        "model_id",
        "/dev/video",
        "/dev/nvidia",
    )
    offenders: dict[str, list[str]] = {}
    for path in sorted((repository / "lab_control").glob("*.py")):
        lowered = path.read_text(encoding="utf-8").casefold()
        matches = [token for token in forbidden if token.casefold() in lowered]
        if matches:
            offenders[path.name] = matches
    assert offenders == {}


@pytest.fixture
def workspace_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceManager, dict[str, Path]]:
    repositories_root = tmp_path / "repositories"
    roots = {
        "Controller": _repository(repositories_root / "Controller"),
        "ComponentX": _repository(repositories_root / "ComponentX"),
    }
    manifest = _write_project_manifest(
        tmp_path / "project.json",
        repositories={
            "Controller": {
                "aliases": ["main"],
                "checkout_mode": "stable_branch",
                "default_path": ".",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_CONTROLLER",
                "stable_branch": "main",
            },
            "ComponentX": {
                "aliases": ["component-x"],
                "checkout_mode": "stable_branch",
                "default_path": "ComponentX",
                "path_base": "control_repository_parent",
                "path_environment": "LAB_REPOSITORY_COMPONENT_X",
                "stable_branch": "main",
            },
        },
    )
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(roots["Controller"]))
    monkeypatch.setenv("LAB_REPOSITORY_COMPONENT_X", str(roots["ComponentX"]))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    manager = WorkspaceManager(control_repository=roots["Controller"])
    return manager, roots


def test_create_builds_one_multi_repository_workspace_without_moving_anchors(
    workspace_environment: tuple[WorkspaceManager, dict[str, Path]],
) -> None:
    manager, roots = workspace_environment
    original_heads = {name: _git(path, "rev-parse", "HEAD") for name, path in roots.items()}
    plan = manager.plan_task(
        task_id="T101",
        repository_names=["main", "component-x"],
        scope_assignments=["Controller=src/**", "ComponentX=lib/**"],
    )

    task = manager.create_task(plan)

    assert task["environment"]["status"] == "DEFERRED"
    for name, anchor in roots.items():
        assert _git(anchor, "branch", "--show-current") == "main"
        assert _git(anchor, "rev-parse", "HEAD") == original_heads[name]
        worktree = Path(task["repositories"][name]["worktree_path"])
        assert _git(worktree, "branch", "--show-current") == "task/T101"
    workspace = json.loads((plan.task_root / "T101.code-workspace").read_text())
    assert [folder["name"] for folder in workspace["folders"]] == [
        "Controller",
        "ComponentX",
    ]
    assert manager.inspect_task("T101")["status"] == "ACTIVE"


def test_dry_run_and_open_print_have_no_hidden_workspace_effects(
    workspace_environment: tuple[WorkspaceManager, dict[str, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, _ = workspace_environment
    assert main(["task", "create", "T102", "--repos", "main", "--dry-run"]) == 0
    assert not manager.task_manifest_path("T102").exists()

    task = manager.create_task(
        manager.plan_task(task_id="T103", repository_names=["Controller"])
    )
    assert main(["open", "T103", "--print"]) == 0
    output = capsys.readouterr().out
    assert str(Path(task["workspace_root"]) / "T103.code-workspace") in output


def test_inspect_reports_branch_and_scope_drift(
    workspace_environment: tuple[WorkspaceManager, dict[str, Path]],
) -> None:
    manager, _ = workspace_environment
    task = manager.create_task(
        manager.plan_task(
            task_id="T104",
            repository_names=["Controller"],
            scope_assignments=["Controller=src/**"],
        )
    )
    worktree = Path(task["repositories"]["Controller"]["worktree_path"])
    (worktree / "outside.txt").write_text("drift\n", encoding="utf-8")

    inspection = manager.inspect_task("T104")

    assert inspection["status"] == "DRIFTED"
    assert inspection["repositories"]["Controller"]["dirty"] is True
    assert inspection["repositories"]["Controller"]["scope_violations"] == [
        "outside.txt"
    ]


def test_change_groups_tasks_without_sharing_writer_authority(
    workspace_environment: tuple[WorkspaceManager, dict[str, Path]],
) -> None:
    manager, _ = workspace_environment
    manager.create_change("CHG-7", title="Cross repository change")
    for task_id, repository in (("T105", "Controller"), ("T106", "ComponentX")):
        manager.create_task(
            manager.plan_task(
                task_id=task_id,
                repository_names=[repository],
                change_id="CHG-7",
            )
        )

    change = manager.inspect_change("CHG-7")

    assert change["task_count"] == 2
    assert set(change["tasks"]) == {"T105", "T106"}


def test_writer_lease_is_exclusive_per_task_repository(
    workspace_environment: tuple[WorkspaceManager, dict[str, Path]],
) -> None:
    manager, _ = workspace_environment
    manager.create_task(
        manager.plan_task(
            task_id="T107",
            repository_names=["Controller", "ComponentX"],
        )
    )
    writer = WriterLeaseManager(manager)
    controller = writer.claim("T107", "Controller", actor="agent-a")
    component = writer.claim("T107", "ComponentX", actor="agent-b")

    assert controller["actor"] == "agent-a"
    assert component["actor"] == "agent-b"
    with pytest.raises(LabError, match="writer lease is held by agent-a"):
        writer.claim("T107", "Controller", actor="agent-c")


def test_selected_gitlink_child_is_nested_and_must_match_parent_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = _repository(tmp_path / "Controller")
    child = _repository(parent / "components" / "ComponentY")
    child_commit = _git(child, "rev-parse", "HEAD")
    _git(
        parent,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{child_commit},components/ComponentY",
    )
    _git(parent, "commit", "-m", "pin component")
    manifest = _write_project_manifest(
        tmp_path / "project.json",
        repositories={
            "Controller": {
                "aliases": [],
                "checkout_mode": "stable_branch",
                "default_path": ".",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_CONTROLLER",
                "stable_branch": "main",
            },
            "ComponentY": {
                "aliases": [],
                "checkout_mode": "pinned_gitlink",
                "default_path": "components/ComponentY",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_COMPONENT_Y",
                "stable_branch": "main",
            },
        },
    )
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(parent))
    monkeypatch.setenv("LAB_REPOSITORY_COMPONENT_Y", str(child))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    manager = WorkspaceManager(control_repository=parent)

    plan = manager.plan_task(
        task_id="T108",
        repository_names=["Controller", "ComponentY"],
    )
    task = manager.create_task(plan)

    parent_worktree = Path(task["repositories"]["Controller"]["worktree_path"])
    child_worktree = Path(task["repositories"]["ComponentY"]["worktree_path"])
    assert child_worktree == parent_worktree / "components" / "ComponentY"
    assert _git(child_worktree, "rev-parse", "HEAD") == child_commit

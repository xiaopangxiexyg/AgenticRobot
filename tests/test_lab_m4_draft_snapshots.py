from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_control import git as lab_git
from lab_control.cli import main
from lab_control.draft_snapshots import capture_repository_draft
from lab_control.queue_profiles import load_queue_profile
from lab_control.run_requests import RunRequestManager
from lab_control.runner_protocol import RunnerCoordinator
from lab_control.runner_staging import ExactSnapshotStager
from lab_control.runs import control_plane_identity
from lab_control.workspace import LabError, WorkspaceManager


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, *, title: str) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Draft Snapshot Test")
    _git(path, "config", "user.email", "draft-test@example.invalid")
    (path / "README.md").write_text(f"# {title}\n", encoding="utf-8")
    (path / "source.txt").write_text("base\n", encoding="utf-8")
    (path / "deleted.txt").write_text("delete me\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _write_queue_profile(
    root: Path,
    *,
    repositories: list[str],
    snapshot_modes: list[str] | None = None,
) -> None:
    (root / "draft-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v3",
                "profile_id": "draft-check",
                "description": "Project-neutral synthetic snapshot fixture",
                "target_id": "test-target",
                "risk_level": "R0",
                "snapshot_modes": snapshot_modes or ["clean", "draft"],
                "repositories": repositories,
                "command_ids": ["contract-suite"],
                "capabilities": ["python"],
            }
        ),
        encoding="utf-8",
    )


def _write_runner_profile(root: Path) -> None:
    (root / "runner-a.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-profile.v1",
                "runner_id": "runner-a",
                "description": "Project-neutral runner fixture",
                "target_ids": ["test-target"],
                "slots": ["slot-a1"],
                "capabilities": ["python"],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )


def _configure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repositories: dict[str, Path],
    repository_config: dict[str, dict[str, object]],
) -> tuple[WorkspaceManager, Path, Path]:
    manifest = tmp_path / "project.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "lab.project.v3",
                "project_id": "draft-stack",
                "control_repository": "Controller",
                "default_development_host": "developer-a",
                "repositories": repository_config,
            }
        ),
        encoding="utf-8",
    )
    queue_root = tmp_path / "queue-profiles"
    runner_root = tmp_path / "runner-profiles"
    queue_root.mkdir()
    runner_root.mkdir()
    _write_queue_profile(queue_root, repositories=list(repositories))
    _write_runner_profile(runner_root)
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(manifest))
    for name, path in repositories.items():
        monkeypatch.setenv(f"LAB_REPOSITORY_{name.upper()}", str(path))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_QUEUE_PROFILE_ROOT", str(queue_root))
    monkeypatch.setenv("LAB_RUNNER_PROFILE_ROOT", str(runner_root))
    monkeypatch.setenv("LAB_REQUEST_STATE_ROOT", str(tmp_path / "request-state"))
    control_plane_repository = Path(__file__).resolve().parents[1]
    return (
        WorkspaceManager(control_repository=control_plane_repository),
        queue_root,
        runner_root,
    )


def _index_bytes(repository: Path) -> bytes:
    path = Path(_git(repository, "rev-parse", "--git-path", "index"))
    if not path.is_absolute():
        path = repository / path
    return path.read_bytes()


def test_draft_snapshot_preserves_workspace_and_stages_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    controller = _repository(tmp_path / "repositories" / "Controller", title="One")
    manager, queue_root, runner_root = _configure(
        tmp_path,
        monkeypatch,
        repositories={"Controller": controller},
        repository_config={
            "Controller": {
                "aliases": ["main"],
                "checkout_mode": "stable_branch",
                "default_path": ".",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_CONTROLLER",
                "stable_branch": "main",
            }
        },
    )
    task = manager.create_task(
        manager.plan_task(
            task_id="T601",
            repository_names=["Controller"],
            scope_assignments=["Controller=**"],
        )
    )
    worktree = Path(task["repositories"]["Controller"]["worktree_path"])
    (worktree / "source.txt").write_text("staged\n", encoding="utf-8")
    _git(worktree, "add", "source.txt")
    (worktree / "source.txt").write_text("working tree wins\n", encoding="utf-8")
    (worktree / "deleted.txt").unlink()
    (worktree / "included.py").write_text("VALUE = 7\n", encoding="utf-8")
    (worktree / "excluded.txt").write_text("local only\n", encoding="utf-8")
    before = {
        "branch": _git(worktree, "branch", "--show-current"),
        "head": _git(worktree, "rev-parse", "HEAD"),
        "status": _git(worktree, "status", "--short"),
        "index": _index_bytes(worktree),
    }

    assert (
        main(
            [
                "test",
                "T601",
                "draft-check",
                "--detach",
                "--draft",
                "--include-untracked",
                "Controller=included.py",
                "--json",
            ]
        )
        == 0
    )
    submission = json.loads(capsys.readouterr().out)
    request = submission["request"]
    raw_repository = request["snapshot"]["repositories"]["Controller"]
    assert request["schema"] == "lab.run-request.v4"
    assert request["snapshot"]["snapshot_mode"] == "draft"
    assert request["evidence_tier"] == "Development"
    assert raw_repository["synthetic"] is True
    assert raw_repository["base_commit"] == before["head"]
    assert raw_repository["included_untracked"] == ["included.py"]
    assert raw_repository["excluded_untracked"] == ["excluded.txt"]
    assert _git(worktree, "branch", "--show-current") == before["branch"]
    assert _git(worktree, "rev-parse", "HEAD") == before["head"]
    assert _git(worktree, "status", "--short") == before["status"]
    assert _index_bytes(worktree) == before["index"]

    requests = RunRequestManager(manager)
    coordinator = RunnerCoordinator(
        requests.store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(manager.control_repository),
    )
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    receipt = ExactSnapshotStager(
        coordinator,
        repository_catalog=manager.catalog,
        slot_root=tmp_path / "slots",
    ).stage(str(request["request_id"]), "runner-a")
    staged = (
        tmp_path
        / "slots"
        / "slot-a1"
        / str(request["request_id"])
        / receipt["repositories"]["Controller"]["relative_path"]
    )
    assert lab_git.current_branch(staged) is None
    assert lab_git.current_commit(staged) == raw_repository["commit"]
    assert lab_git.status_short(staged) == ()
    assert (staged / "source.txt").read_text(encoding="utf-8") == "working tree wins\n"
    assert not (staged / "deleted.txt").exists()
    assert (staged / "included.py").read_text(encoding="utf-8") == "VALUE = 7\n"
    assert not (staged / "excluded.txt").exists()

    repeated = requests.submit_draft_profile(
        "T601",
        "draft-check",
        evidence_tier="Development",
        include_untracked={"Controller": ["included.py"]},
    )
    assert repeated["created"] is False
    assert repeated["request"] == request


def test_draft_profile_authorization_and_untracked_secret_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _repository(tmp_path / "repositories" / "Controller", title="One")
    manager, queue_root, _ = _configure(
        tmp_path,
        monkeypatch,
        repositories={"Controller": controller},
        repository_config={
            "Controller": {
                "aliases": [],
                "checkout_mode": "stable_branch",
                "default_path": ".",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_CONTROLLER",
                "stable_branch": "main",
            }
        },
    )
    task = manager.create_task(
        manager.plan_task(task_id="T602", repository_names=["Controller"])
    )
    worktree = Path(task["repositories"]["Controller"]["worktree_path"])
    (worktree / ".env").write_text("TOKEN=secret\n", encoding="utf-8")

    with pytest.raises(LabError, match="denies sensitive untracked path"):
        RunRequestManager(manager).submit_draft_profile(
            "T602",
            "draft-check",
            evidence_tier="Development",
            include_untracked={"Controller": [".env"]},
        )

    profile = load_queue_profile(queue_root, "draft-check")
    assert profile.snapshot_modes == ("clean", "draft")

    profile_path = queue_root / "draft-check.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["execution_class"] = "project-specific"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabError, match="unexpected shape"):
        load_queue_profile(queue_root, "draft-check")


def test_linked_draft_updates_parent_gitlink_and_stages_one_consistent_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = _repository(tmp_path / "repositories" / "Component", title="Child")
    parent = _repository(tmp_path / "repositories" / "Controller", title="Parent")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "components/child",
    )
    _git(parent, "commit", "-am", "add component")
    component = parent / "components" / "child"
    manager, queue_root, runner_root = _configure(
        tmp_path,
        monkeypatch,
        repositories={"Controller": parent, "Component": component},
        repository_config={
            "Controller": {
                "aliases": [],
                "checkout_mode": "stable_branch",
                "default_path": ".",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_CONTROLLER",
                "stable_branch": "main",
            },
            "Component": {
                "aliases": [],
                "checkout_mode": "pinned_gitlink",
                "default_path": "components/child",
                "path_base": "control_repository",
                "path_environment": "LAB_REPOSITORY_COMPONENT",
                "stable_branch": "main",
            },
        },
    )
    task = manager.create_task(
        manager.plan_task(
            task_id="T603",
            repository_names=["Controller", "Component"],
            scope_assignments=["Controller=**", "Component=**"],
        )
    )
    parent_worktree = Path(task["repositories"]["Controller"]["worktree_path"])
    child_worktree = Path(task["repositories"]["Component"]["worktree_path"])
    child_head = _git(child_worktree, "rev-parse", "HEAD")
    parent_head = _git(parent_worktree, "rev-parse", "HEAD")
    child_index = _index_bytes(child_worktree)
    parent_index = _index_bytes(parent_worktree)
    (child_worktree / "source.txt").write_text("draft child\n", encoding="utf-8")

    requests = RunRequestManager(manager)
    submission = requests.submit_draft_profile(
        "T603",
        "draft-check",
        evidence_tier="Development",
        include_untracked={},
    )
    request = submission["request"]
    repositories = request["snapshot"]["repositories"]
    child_commit = repositories["Component"]["commit"]
    parent_commit = repositories["Controller"]["commit"]
    assert child_commit != child_head
    assert parent_commit != parent_head
    assert repositories["Controller"]["submodules"]["components/child"] == child_commit
    assert request["snapshot"]["repository_links"] == [
        {
            "parent_repository": "Controller",
            "submodule_path": "components/child",
            "child_repository": "Component",
            "commit": child_commit,
        }
    ]
    assert _git(child_worktree, "rev-parse", "HEAD") == child_head
    assert _git(parent_worktree, "rev-parse", "HEAD") == parent_head
    assert _index_bytes(child_worktree) == child_index
    assert _index_bytes(parent_worktree) == parent_index

    coordinator = RunnerCoordinator(
        requests.store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(manager.control_repository),
    )
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    receipt = ExactSnapshotStager(
        coordinator,
        repository_catalog=manager.catalog,
        slot_root=tmp_path / "slots",
    ).stage(str(request["request_id"]), "runner-a")
    staged_parent = (
        tmp_path
        / "slots"
        / "slot-a1"
        / str(request["request_id"])
        / receipt["repositories"]["Controller"]["relative_path"]
    )
    staged_child = staged_parent / "components" / "child"
    assert lab_git.current_commit(staged_parent) == parent_commit
    assert lab_git.current_commit(staged_child) == child_commit
    assert lab_git.status_short(staged_parent) == ()
    assert lab_git.status_short(staged_child) == ()
    assert (staged_child / "source.txt").read_text(encoding="utf-8") == "draft child\n"


def test_draft_rejects_a_dirty_linked_repository_omitted_from_the_task(
    tmp_path: Path,
) -> None:
    child = _repository(tmp_path / "Component", title="Child")
    parent = _repository(tmp_path / "Controller", title="Parent")
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "components/child",
    )
    _git(parent, "commit", "-am", "add component")
    (parent / "components" / "child" / "source.txt").write_text(
        "uncaptured child change\n", encoding="utf-8"
    )

    with pytest.raises(LabError, match="cannot omit dirty linked repository"):
        capture_repository_draft("Controller", parent, scopes=("**",))

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_control.baseline import load_baseline_lock
from lab_control.snapshot_resolution import resolve_clean_snapshot_identity
from lab_control.workspace import LabError, WorkspaceManager


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, *, identity: str) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Repository Link Test")
    _git(path, "config", "user.email", "repository-link@example.invalid")
    _git(path, "remote", "add", "origin", f"git@example.invalid:{identity}.git")
    (path / "README.md").write_text(f"# {path.name}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _linked_repositories(tmp_path: Path) -> dict[str, object]:
    repositories = (tmp_path / "repositories").resolve()
    child = _repository(
        repositories / "component-y",
        identity="TestOrg/ComponentY",
    )
    child_baseline = _git(child, "rev-parse", "HEAD")
    parent = _repository(
        repositories / "main-project",
        identity="TestOrg/MainProject",
    )
    _git(
        parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(child),
        "components/component-y",
    )
    _git(parent, "commit", "-m", "pin child baseline")
    parent_baseline = _git(parent, "rev-parse", "HEAD")

    (child / "source.py").write_text("REVISION = 2\n", encoding="utf-8")
    _git(child, "add", "source.py")
    _git(child, "commit", "-m", "child revision two")
    child_next = _git(child, "rev-parse", "HEAD")
    _git(parent / "components" / "component-y", "fetch", "origin")

    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v2",
                "baseline_id": "linked-baseline-v2",
                "repositories": {
                    "main-project": {
                        "canonical_identity": "TestOrg/MainProject",
                        "commit": parent_baseline,
                        "tree_oid": _git(
                            parent,
                            "rev-parse",
                            f"{parent_baseline}^{{tree}}",
                        ),
                        "submodules": {"components/component-y": child_baseline},
                    },
                    "component-y": {
                        "canonical_identity": "TestOrg/ComponentY",
                        "commit": child_baseline,
                        "tree_oid": _git(
                            child,
                            "rev-parse",
                            f"{child_baseline}^{{tree}}",
                        ),
                        "submodules": {},
                    },
                },
                "repository_links": [
                    {
                        "parent_repository": "main-project",
                        "submodule_path": "components/component-y",
                        "child_repository": "component-y",
                    }
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    project_manifest = tmp_path / "project.json"
    project_manifest.write_text(
        json.dumps(
            {
                "schema": "lab.project.v1",
                "project_id": "linked-project",
                "control_repository": "main-project",
                "default_development_host": "local-dev-control",
                "repositories": {
                    "main-project": {
                        "aliases": ["main"],
                        "default_path": "main-project",
                        "path_environment": "LAB_REPOSITORY_MAIN_PROJECT",
                    },
                    "component-y": {
                        "aliases": [],
                        "default_path": "component-y",
                        "path_environment": "LAB_REPOSITORY_COMPONENT_Y",
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
            "LAB_PROJECT_MANIFEST": str(project_manifest),
            "LAB_REPOSITORY_MAIN_PROJECT": str(parent),
            "LAB_REPOSITORY_COMPONENT_Y": str(child),
        },
    )
    return {
        "parent": parent,
        "child": child,
        "child_baseline": child_baseline,
        "child_next": child_next,
        "baseline": baseline_path,
        "manager": manager,
    }


def _initialize_task_submodule(task_parent: Path) -> Path:
    _git(
        task_parent,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "update",
        "--init",
        "--checkout",
        "--no-fetch",
    )
    return task_parent / "components" / "component-y"


def _nested_manager(linked: dict[str, object], tmp_path: Path) -> WorkspaceManager:
    parent = Path(str(linked["parent"]))
    project_manifest = tmp_path / "nested-project.json"
    project_manifest.write_text(
        json.dumps(
            {
                "schema": "lab.project.v2",
                "project_id": "nested-project",
                "control_repository": "main-project",
                "default_development_host": "local-dev-control",
                "repositories": {
                    "main-project": {
                        "aliases": ["main"],
                        "default_path": ".",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_MAIN_PROJECT",
                    },
                    "component-y": {
                        "aliases": [],
                        "default_path": "components/component-y",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_COMPONENT_Y",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return WorkspaceManager(
        control_repository=parent,
        environ={
            "LAB_HOME": str(tmp_path / "nested-lab-home"),
            "LAB_PROJECT_MANIFEST": str(project_manifest),
        },
    )


def test_baseline_v1_remains_compatible_without_repository_links(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "component", identity="TestOrg/Component")
    baseline_path = tmp_path / "legacy-baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v1",
                "baseline_id": "legacy-baseline-v1",
                "repositories": {
                    "component": {
                        "canonical_identity": "TestOrg/Component",
                        "commit": _git(repository, "rev-parse", "HEAD"),
                        "tree_oid": _git(repository, "rev-parse", "HEAD^{tree}"),
                        "submodules": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    baseline = load_baseline_lock(baseline_path)

    assert baseline.schema == "lab.baseline.v1"
    assert baseline.repository_links == ()


def test_baseline_v2_rejects_a_gitlink_child_pin_mismatch(tmp_path: Path) -> None:
    linked = _linked_repositories(tmp_path)
    baseline_path = Path(str(linked["baseline"]))
    payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    payload["repositories"]["component-y"]["commit"] = "f" * 40
    baseline_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="repository link gitlink mismatch"):
        load_baseline_lock(baseline_path)


def test_parent_gitlink_derives_an_unmodified_child_revision(tmp_path: Path) -> None:
    linked = _linked_repositories(tmp_path)
    manager = linked["manager"]
    assert isinstance(manager, WorkspaceManager)
    manifest = manager.create_task(
        manager.plan_task(
            task_id="LINK-0001",
            repository_names=["main-project"],
        )
    )
    task_parent = Path(
        str(manifest["repositories"]["main-project"]["worktree_path"])
    )
    task_child = _initialize_task_submodule(task_parent)
    _git(task_child, "checkout", "--detach", str(linked["child_next"]))
    _git(task_parent, "add", "components/component-y")
    _git(task_parent, "commit", "-m", "bump linked child")

    _, snapshot = resolve_clean_snapshot_identity(
        manager,
        task_id="LINK-0001",
        profile_id="linked-clean",
        profile_sha256="a" * 64,
        repositories=("main-project", "component-y"),
        baseline_path=Path(str(linked["baseline"])),
    )

    repositories = snapshot["repositories"]
    assert isinstance(repositories, dict)
    child = repositories["component-y"]
    assert child["commit"] == linked["child_next"]
    assert child["source"] == "parent_gitlink"
    assert child["parent_repository"] == "main-project"
    assert child["submodule_path"] == "components/component-y"
    assert snapshot["repository_links"] == [
        {
            "parent_repository": "main-project",
            "submodule_path": "components/component-y",
            "child_repository": "component-y",
            "commit": linked["child_next"],
        }
    ]


def test_clean_snapshot_reads_gitlinks_without_initializing_task_submodules(
    tmp_path: Path,
) -> None:
    linked = _linked_repositories(tmp_path)
    manager = linked["manager"]
    assert isinstance(manager, WorkspaceManager)
    manifest = manager.create_task(
        manager.plan_task(
            task_id="LINK-0003",
            repository_names=["main-project"],
        )
    )
    task_parent = Path(
        str(manifest["repositories"]["main-project"]["worktree_path"])
    )
    assert _git(task_parent, "submodule", "status").startswith("-")

    _, snapshot = resolve_clean_snapshot_identity(
        manager,
        task_id="LINK-0003",
        profile_id="linked-clean",
        profile_sha256="c" * 64,
        repositories=("main-project", "component-y"),
        baseline_path=Path(str(linked["baseline"])),
    )

    repositories = snapshot["repositories"]
    assert isinstance(repositories, dict)
    assert repositories["main-project"]["submodules"] == {
        "components/component-y": linked["child_baseline"]
    }
    assert repositories["component-y"]["source"] == "baseline"


def test_task_places_a_selected_child_at_its_parent_gitlink_path(
    tmp_path: Path,
) -> None:
    linked = _linked_repositories(tmp_path)
    manager = _nested_manager(linked, tmp_path)

    manifest = manager.create_task(
        manager.plan_task(
            task_id="LINK-0004",
            repository_names=["component-y", "main-project"],
        )
    )

    task_root = Path(str(manifest["workspace_root"]))
    parent = task_root / "main-project"
    child = parent / "components" / "component-y"
    assert Path(
        str(manifest["repositories"]["component-y"]["worktree_path"])
    ) == child
    assert manifest["repositories"]["component-y"]["workspace_placement"] == {
        "kind": "nested_gitlink",
        "parent_repository": "main-project",
        "submodule_path": "components/component-y",
    }
    assert _git(parent, "status", "--short") == ""
    assert _git(child, "branch", "--show-current") == "task/LINK-0004"
    workspace = json.loads(
        (task_root / "LINK-0004.code-workspace").read_text(encoding="utf-8")
    )
    assert workspace["folders"] == [
        {"name": "main-project", "path": "main-project"},
        {
            "name": "component-y",
            "path": "main-project/components/component-y",
        },
    ]


def test_nested_task_rejects_a_child_base_that_differs_from_the_parent_gitlink(
    tmp_path: Path,
) -> None:
    linked = _linked_repositories(tmp_path)
    manager = _nested_manager(linked, tmp_path)

    with pytest.raises(LabError, match="base differs from parent gitlink"):
        manager.plan_task(
            task_id="LINK-0005",
            repository_names=["main-project", "component-y"],
            base_assignments=[f"component-y={linked['child_next']}"],
        )


def test_task_child_override_must_equal_parent_gitlink(tmp_path: Path) -> None:
    linked = _linked_repositories(tmp_path)
    manager = linked["manager"]
    assert isinstance(manager, WorkspaceManager)
    manifest = manager.create_task(
        manager.plan_task(
            task_id="LINK-0002",
            repository_names=["main-project", "component-y"],
        )
    )
    task_parent = Path(
        str(manifest["repositories"]["main-project"]["worktree_path"])
    )
    _initialize_task_submodule(task_parent)

    with pytest.raises(LabError, match="override differs from parent gitlink"):
        resolve_clean_snapshot_identity(
            manager,
            task_id="LINK-0002",
            profile_id="linked-clean",
            profile_sha256="b" * 64,
            repositories=("main-project", "component-y"),
            baseline_path=Path(str(linked["baseline"])),
        )

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lab_control import git as lab_git
from lab_control.cli import main
from lab_control.runs import RunManager
from lab_control.workspace import LabError, WorkspaceManager


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
    (path / "README.md").write_text("# Controller\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _write_profile(
    root: Path,
    *,
    profile_id: str,
    source: str,
    timeout_seconds: int = 10,
    environment_id: str | None = None,
) -> None:
    payload = {
        "schema": "lab.profile.v1",
        "profile_id": profile_id,
        "description": f"Temporary {profile_id} Profile",
        "execution": "local",
        "risk_level": "R0",
        "requires_clean": True,
        "commands": [
            {
                "command_id": profile_id,
                "repository": "Controller",
                "argv": ["{python}", "-c", source],
                "timeout_seconds": timeout_seconds,
            }
        ],
    }
    if environment_id is not None:
        payload["commands"][0]["environment_id"] = environment_id
    (root / f"{profile_id}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def run_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceManager, Path, Path]:
    controller = _repository(tmp_path / "repositories" / "Controller")
    project_manifest = tmp_path / "project.json"
    project_manifest.write_text(
        json.dumps(
            {
                "schema": "lab.project.v3",
                "project_id": "example-stack",
                "control_repository": "Controller",
                "default_development_host": "developer-a",
                "repositories": {
                    "Controller": {
                        "aliases": ["main"],
                        "checkout_mode": "stable_branch",
                        "default_path": ".",
                        "path_base": "control_repository",
                        "path_environment": "LAB_REPOSITORY_CONTROLLER",
                        "stable_branch": "main",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    profile_root = tmp_path / "profiles"
    profile_root.mkdir()
    environment_root = tmp_path / "environments"
    environment_root.mkdir()
    _write_profile(profile_root, profile_id="pass", source="print('passed')")
    _write_profile(
        profile_root,
        profile_id="fail",
        source="raise SystemExit(7)",
    )
    _write_profile(
        profile_root,
        profile_id="mutate",
        source="from pathlib import Path; Path('generated.txt').write_text('dirty')",
    )
    (profile_root / "impact-rules.json").write_text(
        json.dumps(
            {
                "schema": "lab.impact-rules.v1",
                "fallback_profiles": {},
                "rules": [
                    {
                        "rule_id": "controller-source",
                        "paths": {"Controller": ["src/**"]},
                        "profiles": ["pass"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(project_manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(controller))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_PROFILE_ROOT", str(profile_root))
    monkeypatch.setenv("LAB_ENVIRONMENT_PROFILE_ROOT", str(environment_root))
    repository = Path(__file__).resolve().parents[1]
    manager = WorkspaceManager(control_repository=repository)
    return manager, controller, profile_root


def _create_task(manager: WorkspaceManager, task_id: str) -> tuple[dict[str, object], Path]:
    task = manager.create_task(
        manager.plan_task(
            task_id=task_id,
            repository_names=["Controller"],
            scope_assignments=["Controller=src/**"],
        )
    )
    repositories = task["repositories"]
    assert isinstance(repositories, dict)
    controller = repositories["Controller"]
    assert isinstance(controller, dict)
    return task, Path(str(controller["worktree_path"]))


def test_clean_run_uses_detached_snapshot_and_records_exact_evidence(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, _ = run_environment
    _, worktree = _create_task(manager, "T201")

    run = RunManager(manager).run_clean_profile("T201", "pass")

    outcome = run["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["classification"] == "PASSED"
    assert run["evidence_tier"] == "Development"
    repositories = run["repositories"]
    assert isinstance(repositories, dict)
    tested = repositories["Controller"]
    assert isinstance(tested, dict)
    assert tested["commit"] == _git(worktree, "rev-parse", "HEAD")
    snapshot = Path(str(run["snapshot_root"])) / "Controller"
    assert lab_git.current_branch(snapshot) is None
    assert lab_git.status_short(snapshot) == ()
    artifacts = run["artifacts"]
    assert isinstance(artifacts, list) and len(artifacts) == 1
    artifact = artifacts[0]
    assert isinstance(artifact, dict)
    log = Path(str(run["run_root"])) / str(artifact["path"])
    assert hashlib.sha256(log.read_bytes()).hexdigest() == artifact["sha256"]


def test_dirty_task_is_rejected_before_a_run_directory_is_created(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, _ = run_environment
    _, worktree = _create_task(manager, "T202")
    dirty = worktree / "src" / "dirty.txt"
    dirty.parent.mkdir()
    dirty.write_text("not checkpointed\n", encoding="utf-8")

    with pytest.raises(LabError, match="rejects dirty repository"):
        RunManager(manager).run_clean_profile("T202", "pass")

    assert RunManager(manager).list_runs("T202") == []


def test_failed_assertion_is_recorded_instead_of_collapsed_into_pass(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, _ = run_environment
    _create_task(manager, "T203")

    run = RunManager(manager).run_clean_profile("T203", "fail")

    outcome = run["outcome"]
    assert isinstance(outcome, dict)
    assert outcome == {
        "assertions_passed": False,
        "classification": "FAILED_ASSERTION",
        "environment_valid": True,
        "evidence_complete": True,
        "evidence_errors": [],
        "execution_completed": True,
        "source_errors": [],
        "source_stable": True,
    }
    commands = run["commands"]
    assert isinstance(commands, list)
    assert commands[0]["exit_code"] == 7


def test_payload_that_mutates_the_snapshot_is_invalid_source(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, _ = run_environment
    _create_task(manager, "T204")

    run = RunManager(manager).run_clean_profile("T204", "mutate")

    outcome = run["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["classification"] == "INVALID_SOURCE"
    assert outcome["assertions_passed"] is False
    assert outcome["source_stable"] is False


def test_repeated_runs_share_source_identity_but_not_runtime_directories(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, _ = run_environment
    _create_task(manager, "T205")
    runner = RunManager(manager)

    first = runner.run_clean_profile("T205", "pass")
    second = runner.run_clean_profile("T205", "pass")

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["run_id"] != second["run_id"]
    assert first["snapshot_root"] != second["snapshot_root"]
    assert first["run_root"] != second["run_root"]


def test_open_run_and_against_run_use_recorded_snapshot_identity(
    run_environment: tuple[WorkspaceManager, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, _, _ = run_environment
    _create_task(manager, "T206")
    run = RunManager(manager).run_clean_profile("T206", "pass")
    run_id = str(run["run_id"])

    assert main(["open", "--run", run_id, "--print"]) == 0
    assert str(run["snapshot_workspace"]) in capsys.readouterr().out
    assert main(["inspect", "T206", "--against-run", run_id, "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["against_run"]["run_id"] == run_id
    assert inspection["against_run"]["repositories"]["Controller"][
        "changed_paths"
    ] == []


def test_recommended_profile_is_derived_from_committed_task_diff(
    run_environment: tuple[WorkspaceManager, Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, _, _ = run_environment
    _, worktree = _create_task(manager, "T207")
    source = worktree / "src" / "feature.py"
    source.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    _git(worktree, "add", "src/feature.py")
    _git(worktree, "commit", "-m", "feature")

    assert main(["test", "T207", "--recommended", "--json"]) == 0
    batch = json.loads(capsys.readouterr().out)
    assert batch["profiles"] == ["pass"]
    assert batch["all_passed"] is True


def test_candidate_evidence_refuses_an_unprotected_control_plane(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, _ = run_environment
    _create_task(manager, "T208")

    with pytest.raises(
        LabError,
        match="clean protected control-plane main|HEAD to equal origin/main",
    ):
        RunManager(manager).run_clean_profile(
            "T208",
            "pass",
            evidence_tier="Candidate",
        )


def test_environment_profile_binds_imports_to_the_snapshot_checkout(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, profile_root = run_environment
    _, worktree = _create_task(manager, "T210")
    package = worktree / "src" / "controller_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 42\n", encoding="utf-8")
    _git(worktree, "add", "src/controller_pkg/__init__.py")
    _git(worktree, "commit", "-m", "add package")
    environment_root = Path(manager.environ["LAB_ENVIRONMENT_PROFILE_ROOT"])
    environment_root.joinpath("controller-python.json").write_text(
        json.dumps(
            {
                "schema": "lab.environment.v1",
                "environment_id": "controller-python",
                "repository": "Controller",
                "resolver": {"kind": "python", "executable": sys.executable},
                "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
                "import_modules": ["controller_pkg"],
                "source_roots": ["src"],
            }
        ),
        encoding="utf-8",
    )
    _write_profile(
        profile_root,
        profile_id="environment-pass",
        source="import controller_pkg; assert controller_pkg.VALUE == 42",
        environment_id="controller-python",
    )

    run = RunManager(manager).run_clean_profile("T210", "environment-pass")

    outcome = run["outcome"]
    assert isinstance(outcome, dict)
    assert outcome["classification"] == "PASSED"
    environment = run["environment"]
    assert isinstance(environment, dict)
    resolved = environment["resolved"]
    assert isinstance(resolved, dict)
    origin = resolved["controller-python"]["import_origins"]["controller_pkg"]
    assert str(run["snapshot_root"]) in origin


def test_profile_cannot_raise_the_local_risk_or_dirty_boundary(
    run_environment: tuple[WorkspaceManager, Path, Path],
) -> None:
    manager, _, profile_root = run_environment
    _create_task(manager, "T209")
    profile = json.loads((profile_root / "pass.json").read_text(encoding="utf-8"))
    profile["risk_level"] = "R3"
    (profile_root / "unsafe.json").write_text(
        json.dumps({**profile, "profile_id": "unsafe"}),
        encoding="utf-8",
    )

    with pytest.raises(LabError, match="only permits R0/R1"):
        RunManager(manager).run_clean_profile("T209", "unsafe")


def test_checked_in_profile_is_small_local_and_download_free() -> None:
    repository = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (repository / "integration" / "profiles" / "lab-control-unit.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["execution"] == "local"
    assert payload["risk_level"] == "R0"
    assert payload["requires_clean"] is True
    serialized = json.dumps(payload).casefold()
    assert "openpi" not in serialized
    assert "gpu" not in serialized
    assert "device" not in serialized
    assert "download" not in serialized
    assert payload["commands"][0]["argv"] == [
        "{python}",
        "dev",
        "test",
        "lab-control",
    ]
    assert payload["commands"][0]["repository"] == "AgenticRobot"
    assert payload["commands"][0]["environment_id"] == "agenticrobot-py310"

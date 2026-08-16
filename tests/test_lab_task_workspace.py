from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lab_control import git as lab_git
from lab_control import runs as lab_runs
from lab_control.baseline import load_baseline_lock
from lab_control.cli import main
from lab_control.dev_cli import inspect_environment
from lab_control.dev_config import load_dev_config
from lab_control.environments import load_environment_profile
from lab_control.profiles import load_profile
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
    (path / "README.md").write_text(f"# {path.name}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


@pytest.fixture
def lab_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    repositories = tmp_path / "repositories"
    roots = {
        "AgenticRobot": _repository(repositories / "AgenticRobot"),
        "RPent": _repository(repositories / "RPent"),
        "openpi": _repository(repositories / "openpi"),
    }
    monkeypatch.setenv("LAB_REPOSITORY_AGENTICROBOT", str(roots["AgenticRobot"]))
    monkeypatch.setenv("LAB_REPOSITORY_RPENT", str(roots["RPent"]))
    monkeypatch.setenv("LAB_REPOSITORY_OPENPI", str(roots["openpi"]))
    monkeypatch.setenv(
        "LAB_PROJECT_MANIFEST",
        str(Path(__file__).resolve().parents[1] / "integration" / "project.json"),
    )
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    return roots


@pytest.fixture
def profile_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "profiles"
    root.mkdir()
    monkeypatch.setenv("LAB_PROFILE_ROOT", str(root))
    return root


@pytest.fixture
def environment_profile_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = tmp_path / "environments"
    root.mkdir()
    monkeypatch.setenv("LAB_ENVIRONMENT_PROFILE_ROOT", str(root))
    return root


def _write_profile(
    root: Path,
    *,
    profile_id: str,
    argv: list[str],
    risk_level: str = "R0",
    timeout_seconds: int = 30,
) -> Path:
    payload = {
        "schema": "lab.profile.v1",
        "profile_id": profile_id,
        "description": f"Test profile {profile_id}",
        "execution": "local",
        "risk_level": risk_level,
        "requires_clean": True,
        "commands": [
            {
                "command_id": "probe",
                "repository": "AgenticRobot",
                "argv": argv,
                "timeout_seconds": timeout_seconds,
            }
        ],
    }
    path = root / f"{profile_id}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_create_builds_isolated_multi_repo_workspace(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots = lab_environment
    original_heads = {
        name: _git(path, "rev-parse", "HEAD") for name, path in roots.items()
    }
    assert (
        main(
            [
                "change",
                "create",
                "CHG-0042",
                "--title",
                "Workspace change",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    result = main(
        [
            "task",
            "create",
            "AR-0101",
            "--repos",
            "main,RPent",
            "--title",
            "Workspace contract",
            "--change-id",
            "CHG-0042",
            "--scope",
            "AgenticRobot=lab_control/**",
            "--scope",
            "RPent=integrations/**",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    task_root = Path(payload["workspace_root"])
    manifest = json.loads((task_root / "task.yaml").read_text(encoding="utf-8"))
    workspace = json.loads(
        (task_root / "AR-0101.code-workspace").read_text(encoding="utf-8")
    )

    assert manifest["schema"] == "lab.task.v1"
    assert manifest["change_id"] == "CHG-0042"
    assert manifest["environment"]["status"] == "DEFERRED"
    assert set(manifest["repositories"]) == {"AgenticRobot", "RPent"}
    assert workspace["folders"] == [
        {"name": "AgenticRobot", "path": "AgenticRobot"},
        {"name": "RPent", "path": "RPent"},
    ]
    for name in ("AgenticRobot", "RPent"):
        checkout = task_root / name
        assert _git(checkout, "branch", "--show-current") == "task/AR-0101"
        assert _git(checkout, "rev-parse", "HEAD") == original_heads[name]
        assert _git(roots[name], "rev-parse", "HEAD") == original_heads[name]
        assert _git(roots[name], "status", "--short") == ""


def test_change_groups_independent_tasks_and_reports_child_drift(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["change", "create", "CHG-0100", "--title", "Cross repo"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "task",
                "create",
                "AR-0100",
                "--repos",
                "AgenticRobot",
                "--change-id",
                "CHG-0100",
                "--json",
            ]
        )
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "task",
                "create",
                "OP-0100",
                "--repos",
                "openpi",
                "--change-id",
                "CHG-0100",
                "--json",
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)

    assert main(["inspect", "CHG-0100", "--json"]) == 0
    change = json.loads(capsys.readouterr().out)
    assert change["schema"] == "lab.change-inspection.v1"
    assert change["status"] == "ACTIVE"
    assert set(change["tasks"]) == {"AR-0100", "OP-0100"}
    assert change["task_count"] == 2
    assert Path(first["workspace_root"]) != Path(second["workspace_root"])

    _git(Path(first["workspace_root"]) / "AgenticRobot", "switch", "--detach")
    assert main(["change", "inspect", "CHG-0100", "--json"]) == 0
    drifted = json.loads(capsys.readouterr().out)
    assert drifted["status"] == "ATTENTION"
    assert drifted["tasks"]["AR-0100"]["status"] == "DRIFTED"


def test_change_inspection_aggregates_child_profile_recommendations(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="change-probe",
        argv=["{python}", "-c", "print('change')"],
    )
    assert main(["change", "create", "CHG-0101"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "task",
                "create",
                "AR-0113",
                "--repos",
                "AgenticRobot",
                "--change-id",
                "CHG-0101",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manifest = manager.load_task("AR-0113")
    checkout = Path(manifest["repositories"]["AgenticRobot"]["worktree_path"])
    (checkout / "README.md").write_text("# change task\n", encoding="utf-8")

    assert main(["change", "inspect", "CHG-0101", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["readiness"] == {
        "status": "READY",
        "recommended": [
            {"task_id": "AR-0113", "profile_id": "change-probe"}
        ],
    }
    child = inspection["tasks"]["AR-0113"]
    assert child["readiness"]["recommended_profiles"] == ["change-probe"]


def test_task_rejects_unknown_change_and_unimplemented_remote_host(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "task",
                "create",
                "AR-0103",
                "--repos",
                "AgenticRobot",
                "--change-id",
                "CHG-MISSING",
            ]
        )
        == 2
    )
    assert "Change does not exist" in capsys.readouterr().err
    assert (
        main(
            [
                "task",
                "create",
                "AR-0104",
                "--repos",
                "AgenticRobot",
                "--dev-host",
                "remote-gpu-hardware",
            ]
        )
        == 2
    )
    assert "remote-hosted Task creation is not implemented" in capsys.readouterr().err


def test_create_preflights_all_repositories_before_writing(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots = lab_environment
    _git(roots["RPent"], "branch", "task/AR-0102")

    result = main(["task", "create", "AR-0102", "--repos", "AgenticRobot,RPent"])

    assert result == 2
    assert "local branch already exists in RPent" in capsys.readouterr().err
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(roots["AgenticRobot"]),
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/task/AR-0102",
        ],
        check=False,
    )
    assert completed.returncode == 1


def test_create_revalidates_branch_after_plan_before_writing(
    lab_environment: dict[str, Path],
) -> None:
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    plan = manager.plan_task(
        task_id="AR-0108",
        repository_names=["AgenticRobot", "RPent"],
    )
    _git(lab_environment["RPent"], "branch", "task/AR-0108")

    with pytest.raises(LabError, match="branch appeared before create in RPent"):
        manager.create_task(plan)

    assert not plan.task_root.exists()
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(lab_environment["AgenticRobot"]),
            "show-ref",
            "--verify",
            "--quiet",
            "refs/heads/task/AR-0108",
        ],
        check=False,
    )
    assert completed.returncode == 1


def test_create_fails_closed_when_repository_lock_is_held(
    lab_environment: dict[str, Path],
) -> None:
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    plan = manager.plan_task(
        task_id="AR-0109",
        repository_names=["AgenticRobot"],
    )
    lock_key = f"AgenticRobot:{lab_environment['AgenticRobot']}"

    with (
        manager.advisory_locks("repositories", [lock_key]),
        pytest.raises(LabError, match="holds repositories lock"),
    ):
        manager.create_task(plan)

    assert not plan.task_root.exists()


def test_create_dry_run_has_no_git_or_filesystem_side_effects(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    result = main(
        [
            "task",
            "create",
            "AR-0103",
            "--repos",
            "AgenticRobot,openpi",
            "--dry-run",
            "--json",
        ]
    )

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert not Path(payload["workspace_root"]).exists()
    assert not (tmp_path / "lab-home" / "Worktrees").exists()
    for repository in lab_environment.values():
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "show-ref",
                "--verify",
                "--quiet",
                "refs/heads/task/AR-0103",
            ],
            check=False,
        )
        assert completed.returncode == 1


def test_inspect_reports_dirty_paths_and_scope_drift(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        main(
            [
                "task",
                "create",
                "AR-0104",
                "--repos",
                "AgenticRobot",
                "--scope",
                "AgenticRobot=lab_control/**",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manifest = manager.load_task("AR-0104")
    checkout = Path(manifest["repositories"]["AgenticRobot"]["worktree_path"])
    (checkout / "outside.txt").write_text("scope violation\n", encoding="utf-8")

    result = main(["inspect", "AR-0104", "--json"])

    assert result == 0
    inspection = json.loads(capsys.readouterr().out)
    repository = inspection["repositories"]["AgenticRobot"]
    assert inspection["status"] == "DRIFTED"
    assert repository["dirty"] is True
    assert repository["changed_paths"] == ["outside.txt"]
    assert repository["scope_violations"] == ["outside.txt"]


def test_inspect_recommends_impacted_profile_and_tracks_run_freshness(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="clean-probe",
        argv=["{python}", "-c", "print('fresh-evidence')"],
    )
    assert main(["task", "create", "AR-0108", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manifest = manager.load_task("AR-0108")
    checkout = Path(manifest["repositories"]["AgenticRobot"]["worktree_path"])
    readme = checkout / "README.md"
    readme.write_text("# changed\n", encoding="utf-8")

    assert main(["inspect", "AR-0108", "--json"]) == 0
    dirty = json.loads(capsys.readouterr().out)
    readiness = dirty["readiness"]
    assert readiness["recommended_profiles"] == ["clean-probe"]
    assert readiness["profiles"]["clean-probe"]["evidence_freshness"] == "NOT_RUN"
    assert "Checkpoint/commit" in dirty["next_action"]

    _git(checkout, "add", "README.md")
    _git(checkout, "commit", "-m", "task change")
    assert main(["test", "AR-0108", "clean-probe", "--json"]) == 0
    run = json.loads(capsys.readouterr().out)
    assert run["writer_leases"]["AgenticRobot"]["state"] == "AVAILABLE"
    assert main(["inspect", "AR-0108", "--json"]) == 0
    current = json.loads(capsys.readouterr().out)
    evidence = current["readiness"]["profiles"]["clean-probe"]
    assert evidence["evidence_freshness"] == "CURRENT"
    assert evidence["latest_run"]["classification"] == "PASSED"

    readme.write_text("# changed again\n", encoding="utf-8")
    assert main(["inspect", "AR-0108", "--json"]) == 0
    stale = json.loads(capsys.readouterr().out)
    evidence = stale["readiness"]["profiles"]["clean-probe"]
    assert evidence["evidence_freshness"] == "STALE"
    assert any(
        "uncommitted changes" in reason for reason in evidence["evidence_reasons"]
    )

    assert (
        main(
            [
                "inspect",
                "AR-0108",
                "--against-run",
                run["run_id"],
                "--json",
            ]
        )
        == 0
    )
    comparison = json.loads(capsys.readouterr().out)["against_run"]
    assert comparison["run_id"] == run["run_id"]
    assert comparison["repositories"]["AgenticRobot"]["changed_paths"] == [
        "README.md"
    ]


def test_recommended_batch_preflights_dirty_task_then_runs_all_profiles(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for profile_id in ("probe-a", "probe-b"):
        _write_profile(
            profile_root,
            profile_id=profile_id,
            argv=["{python}", "-c", f"print({profile_id!r})"],
        )
    assert main(["task", "create", "AR-0109", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manifest = manager.load_task("AR-0109")
    checkout = Path(manifest["repositories"]["AgenticRobot"]["worktree_path"])
    (checkout / "README.md").write_text("# recommended\n", encoding="utf-8")

    assert main(["test", "AR-0109", "--recommended", "--json"]) == 2
    assert "require checkpointed/committed" in capsys.readouterr().err
    assert RunManager(manager).list_runs("AR-0109") == []

    _git(checkout, "add", "README.md")
    _git(checkout, "commit", "-m", "recommended change")
    assert main(["test", "AR-0109", "--recommended", "--json"]) == 0
    batch = json.loads(capsys.readouterr().out)
    assert batch["schema"] == "lab.recommended-test-batch.v1"
    assert batch["profiles"] == ["probe-a", "probe-b"]
    assert batch["all_passed"] is True
    assert batch["no_tests_required"] is False
    assert [run["outcome"]["classification"] for run in batch["runs"]] == [
        "PASSED",
        "PASSED",
    ]


def test_recommended_batch_is_a_noop_when_task_has_no_changes(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="clean-probe",
        argv=["{python}", "-c", "print('not-required')"],
    )
    assert main(["task", "create", "AR-0110", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    assert main(["test", "AR-0110", "--recommended", "--json"]) == 0
    batch = json.loads(capsys.readouterr().out)
    assert batch["profiles"] == []
    assert batch["runs"] == []
    assert batch["all_passed"] is True
    assert batch["no_tests_required"] is True


def test_verify_requires_exact_clean_protected_control_plane(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_repository = lab_environment["AgenticRobot"]
    monkeypatch.setattr(
        "lab_control.cli._control_repository",
        lambda: control_repository,
    )
    monkeypatch.setattr("lab_control.runs._control_plane_hash", lambda _: "a" * 64)
    protected_commit = _git(control_repository, "rev-parse", "HEAD")
    _git(
        control_repository,
        "update-ref",
        "refs/remotes/origin/main",
        protected_commit,
    )
    _write_profile(
        profile_root,
        profile_id="candidate-probe",
        argv=["{python}", "-c", "print('candidate')"],
    )
    assert main(["task", "create", "AR-0111", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    assert main(["verify", "AR-0111", "candidate-probe", "--json"]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["evidence_tier"] == "Candidate"
    assert verified["outcome"]["classification"] == "PASSED"

    (control_repository / "README.md").write_text(
        "# dirty control plane\n",
        encoding="utf-8",
    )
    assert main(["verify", "AR-0111", "candidate-probe", "--json"]) == 2
    assert "clean protected control-plane main" in capsys.readouterr().err
    manager = WorkspaceManager(control_repository=control_repository)
    assert len(RunManager(manager).list_runs("AR-0111")) == 1


def test_verify_rejects_clean_control_plane_ahead_of_protected_ref(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control_repository = lab_environment["AgenticRobot"]
    monkeypatch.setattr(
        "lab_control.cli._control_repository",
        lambda: control_repository,
    )
    monkeypatch.setattr("lab_control.runs._control_plane_hash", lambda _: "a" * 64)
    protected_commit = _git(control_repository, "rev-parse", "HEAD")
    _git(
        control_repository,
        "update-ref",
        "refs/remotes/origin/main",
        protected_commit,
    )
    _write_profile(
        profile_root,
        profile_id="candidate-probe",
        argv=["{python}", "-c", "print('candidate')"],
    )
    assert main(["task", "create", "AR-0112", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()
    (control_repository / "CONTROL.md").write_text("candidate\n", encoding="utf-8")
    _git(control_repository, "add", "CONTROL.md")
    _git(control_repository, "commit", "-m", "unpublished control plane")

    assert main(["verify", "AR-0112", "candidate-probe", "--json"]) == 2
    assert "HEAD to equal origin/main" in capsys.readouterr().err
    manager = WorkspaceManager(control_repository=control_repository)
    assert RunManager(manager).list_runs("AR-0112") == []


def test_open_prints_workspace_without_launching_editor(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-0105", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["open", "AR-0105", "--print"])

    assert result == 0
    workspace = Path(capsys.readouterr().out.strip())
    assert workspace.name == "AR-0105.code-workspace"
    assert workspace.is_file()


def test_inspect_rejects_manifest_path_tampering(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-0106", "--repos", "main"]) == 0
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manifest_path = manager.task_manifest_path("AR-0106")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repositories"]["AgenticRobot"]["worktree_path"] = str(
        lab_environment["RPent"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = main(["inspect", "AR-0106"])

    assert result == 2
    assert "worktree path mismatch" in capsys.readouterr().err


def test_inspect_rejects_manifest_management_path_tampering(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["task", "create", "AR-0107", "--repos", "main"]) == 0
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manifest_path = manager.task_manifest_path("AR-0107")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["repositories"]["AgenticRobot"]["management_path"] = str(
        lab_environment["RPent"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = main(["inspect", "AR-0107"])

    assert result == 2
    assert "management path mismatch" in capsys.readouterr().err


def test_clean_profile_runs_in_detached_snapshot_and_records_evidence(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="clean-probe",
        argv=["{python}", "-c", "print('snapshot-ok')"],
    )
    assert main(["task", "create", "AR-0201", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0201", "clean-probe", "--json"])

    assert result == 0
    run = json.loads(capsys.readouterr().out)
    assert run["outcome"]["classification"] == "PASSED"
    assert run["outcome"]["source_stable"] is True
    assert run["evidence_tier"] == "Development"
    assert len(run["repositories"]["AgenticRobot"]["commit"]) == 40
    assert len(run["repositories"]["AgenticRobot"]["tree_oid"]) == 40
    snapshot = Path(run["snapshot_root"]) / "AgenticRobot"
    assert _git(snapshot, "branch", "--show-current") == ""
    assert _git(snapshot, "status", "--short") == ""
    run_root = Path(run["commands"][0]["log"]).name
    assert run_root == "01-probe.log"
    manifest = (
        Path(lab_environment["AgenticRobot"]).parents[1]
        / "lab-home"
        / "Runtime"
        / "tasks"
        / "AR-0201"
        / run["run_id"]
        / "run.yaml"
    )
    assert manifest.is_file()
    log = manifest.parent / "01-probe.log"
    assert "snapshot-ok" in log.read_text(encoding="utf-8")

    assert main(["open", "--run", run["run_id"], "--print"]) == 0
    workspace = Path(capsys.readouterr().out.strip())
    assert workspace == Path(run["snapshot_workspace"])
    assert workspace.is_file()

    assert main(["runs", "AR-0201", "--json"]) == 0
    runs = json.loads(capsys.readouterr().out)
    assert [item["run_id"] for item in runs] == [run["run_id"]]


def test_clean_profile_payload_gets_an_isolated_minimal_environment(
    lab_environment: dict[str, Path],
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("LAB_TEST_SECRET", "must-not-reach-payload")
    _write_profile(
        profile_root,
        profile_id="isolated-environment",
        argv=[
            "{python}",
            "-c",
            (
                "import os; "
                "assert 'LAB_TEST_SECRET' not in os.environ; "
                "assert os.environ['HOME'].startswith(os.environ['RUN_ROOT']); "
                "assert os.environ['TMPDIR'].startswith(os.environ['RUN_ROOT'])"
            ),
        ],
    )
    assert main(["task", "create", "AR-0208", "--repos", "main"]) == 0
    capsys.readouterr()

    assert main(["test", "AR-0208", "isolated-environment", "--json"]) == 0
    run = json.loads(capsys.readouterr().out)

    assert run["outcome"]["classification"] == "PASSED"
    assert "LAB_TEST_SECRET" not in run["environment"]["payload_environment_keys"]
    assert Path(run["environment"]["isolated_home"]).is_relative_to(
        Path(run["run_root"])
    )


def test_clean_profile_refuses_a_log_path_planted_by_earlier_payload(
    lab_environment: dict[str, Path],
    profile_root: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    victim = tmp_path / "victim.txt"
    victim.write_text("unchanged\n", encoding="utf-8")
    payload = {
        "schema": "lab.profile.v1",
        "profile_id": "log-link-defense",
        "description": "refuse a payload-planted log symlink",
        "execution": "local",
        "risk_level": "R0",
        "requires_clean": True,
        "commands": [
            {
                "command_id": "plant",
                "repository": "AgenticRobot",
                "argv": [
                    "{python}",
                    "-c",
                    (
                        "import os, pathlib; "
                        f"pathlib.Path(os.environ['RUN_ROOT'], '02-second.log')"
                        f".symlink_to({str(victim)!r})"
                    ),
                ],
                "timeout_seconds": 30,
            },
            {
                "command_id": "second",
                "repository": "AgenticRobot",
                "argv": ["{python}", "-c", "print('must-not-run')"],
                "timeout_seconds": 30,
            },
        ],
    }
    (profile_root / "log-link-defense.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    assert main(["task", "create", "AR-0209", "--repos", "main"]) == 0
    capsys.readouterr()

    assert main(["test", "AR-0209", "log-link-defense", "--json"]) == 1
    run = json.loads(capsys.readouterr().out)

    assert run["outcome"]["classification"] == "INFRASTRUCTURE_FAILURE"
    assert run["outcome"]["evidence_complete"] is False
    assert "occupied log path" in run["commands"][1]["start_error"]
    assert victim.read_text(encoding="utf-8") == "unchanged\n"


def test_clean_profile_caps_payload_output_files(
    lab_environment: dict[str, Path],
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(lab_runs, "MAX_COMMAND_LOG_BYTES", 4096)
    _write_profile(
        profile_root,
        profile_id="bounded-output",
        argv=["{python}", "-c", "print('x' * 10000)"],
    )
    assert main(["task", "create", "AR-0211", "--repos", "main"]) == 0
    capsys.readouterr()

    assert main(["test", "AR-0211", "bounded-output", "--json"]) == 1
    run = json.loads(capsys.readouterr().out)

    assert run["outcome"]["classification"] == "FAILED_ASSERTION"
    assert run["commands"][0]["exit_code"] != 0
    assert run["artifacts"][0]["size_bytes"] <= 4096


def test_repeated_clean_runs_share_identity_but_not_snapshot_directory(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="repeatable-probe",
        argv=["{python}", "-c", "print('ok')"],
    )
    assert main(["task", "create", "AR-0210", "--repos", "main"]) == 0
    capsys.readouterr()

    assert main(["test", "AR-0210", "repeatable-probe", "--json"]) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(["test", "AR-0210", "repeatable-probe", "--json"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["snapshot_id"] == second["snapshot_id"]
    assert first["run_id"] != second["run_id"]
    assert first["snapshot_root"] != second["snapshot_root"]
    assert Path(first["snapshot_root"]).is_dir()
    assert Path(second["snapshot_root"]).is_dir()


def test_environment_bound_run_uses_declared_python_and_snapshot_overlay(
    lab_environment: dict[str, Path],
    profile_root: Path,
    environment_profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository = lab_environment["AgenticRobot"]
    package = repository / "sample_package"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 'snapshot'\n", encoding="utf-8")
    _git(repository, "add", "sample_package/__init__.py")
    _git(repository, "commit", "-m", "add sample package")
    (environment_profile_root / "test-python.json").write_text(
        json.dumps(
            {
                "schema": "lab.environment.v1",
                "environment_id": "test-python",
                "repository": "AgenticRobot",
                "resolver": {"kind": "python", "executable": sys.executable},
                "python_major_minor": (
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                ),
                "import_modules": ["sample_package"],
                "source_roots": ["."],
            }
        ),
        encoding="utf-8",
    )
    (profile_root / "environment-probe.json").write_text(
        json.dumps(
            {
                "schema": "lab.profile.v1",
                "profile_id": "environment-probe",
                "description": "environment binding test",
                "execution": "local",
                "risk_level": "R0",
                "requires_clean": True,
                "commands": [
                    {
                        "command_id": "environment-probe",
                        "repository": "AgenticRobot",
                        "environment_id": "test-python",
                        "argv": [
                            "{python}",
                            "-c",
                            "import sample_package; assert sample_package.VALUE == 'snapshot'",
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert main(["doctor", "--environment", "test-python", "--json"]) == 0
    doctor = json.loads(capsys.readouterr().out)
    assert doctor["status"] == "READY"
    assert doctor["environments"]["test-python"]["python_major_minor"] == (
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )
    assert main(["task", "create", "AR-0213", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0213", "environment-probe", "--json"])

    assert result == 0
    run = json.loads(capsys.readouterr().out)
    receipt = run["environment"]["resolved"]["test-python"]
    assert run["outcome"]["classification"] == "PASSED"
    assert run["commands"][0]["environment_id"] == "test-python"
    assert Path(receipt["import_origins"]["sample_package"]).is_relative_to(
        Path(run["snapshot_root"]) / "AgenticRobot"
    )
    assert run["environment"]["source_overlay"] == [
        str(Path(run["snapshot_root"]) / "AgenticRobot")
    ]


def test_environment_probe_rejects_import_from_another_source_tree(
    lab_environment: dict[str, Path],
    profile_root: Path,
    environment_profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (environment_profile_root / "wrong-source.json").write_text(
        json.dumps(
            {
                "schema": "lab.environment.v1",
                "environment_id": "wrong-source",
                "repository": "AgenticRobot",
                "resolver": {"kind": "python", "executable": sys.executable},
                "python_major_minor": (
                    f"{sys.version_info.major}.{sys.version_info.minor}"
                ),
                "import_modules": ["json"],
                "source_roots": ["."],
            }
        ),
        encoding="utf-8",
    )
    (profile_root / "wrong-source.json").write_text(
        json.dumps(
            {
                "schema": "lab.profile.v1",
                "profile_id": "wrong-source",
                "description": "reject stdlib provenance",
                "execution": "local",
                "risk_level": "R0",
                "requires_clean": True,
                "commands": [
                    {
                        "command_id": "wrong-source",
                        "repository": "AgenticRobot",
                        "environment_id": "wrong-source",
                        "argv": ["{python}", "-c", "print('must-not-run')"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["task", "create", "AR-0214", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0214", "wrong-source", "--json"])

    assert result == 1
    run = json.loads(capsys.readouterr().out)
    assert run["outcome"]["classification"] == "INVALID_ENVIRONMENT"
    assert run["outcome"]["execution_completed"] is False
    assert "imports json from the wrong checkout" in run["environment"]["errors"][0]
    assert run["commands"] == []


def test_clean_run_resolves_unmodified_repository_from_baseline(
    lab_environment: dict[str, Path],
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots = lab_environment
    _git(
        roots["RPent"],
        "remote",
        "add",
        "origin",
        "git@example.invalid:TestOrg/RPent.git",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v1",
                "baseline_id": "test-baseline-v1",
                "repositories": {
                    "RPent": {
                        "canonical_identity": "TestOrg/RPent",
                        "commit": _git(roots["RPent"], "rev-parse", "HEAD"),
                        "tree_oid": _git(
                            roots["RPent"],
                            "rev-parse",
                            "HEAD^{tree}",
                        ),
                        "submodules": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAB_BASELINE_LOCK", str(baseline_path))
    profile_path = profile_root / "baseline-probe.json"
    profile_path.write_text(
        json.dumps(
            {
                "schema": "lab.profile.v1",
                "profile_id": "baseline-probe",
                "description": "Task override plus baseline",
                "execution": "local",
                "risk_level": "R0",
                "requires_clean": True,
                "commands": [
                    {
                        "command_id": "main-probe",
                        "repository": "AgenticRobot",
                        "argv": ["{python}", "-c", "print('main')"],
                    },
                    {
                        "command_id": "rpent-probe",
                        "repository": "RPent",
                        "argv": ["{python}", "-c", "print('rpent')"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["task", "create", "AR-0211", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0211", "baseline-probe", "--json"])

    assert result == 0
    run = json.loads(capsys.readouterr().out)
    assert run["outcome"]["classification"] == "PASSED"
    assert run["baseline"]["baseline_id"] == "test-baseline-v1"
    assert run["repositories"]["AgenticRobot"]["source"] == "task_override"
    assert run["repositories"]["RPent"]["source"] == "baseline"
    assert run["repositories"]["RPent"]["baseline_id"] == "test-baseline-v1"
    assert set(Path(run["snapshot_root"]).iterdir()) >= {
        Path(run["snapshot_root"]) / "AgenticRobot",
        Path(run["snapshot_root"]) / "RPent",
    }
    assert main(["inspect", "AR-0211", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    profile = inspection["readiness"]["profiles"]["baseline-probe"]
    assert profile["available"] is True
    assert profile["evidence_freshness"] == "CURRENT"


def test_baseline_tree_mismatch_fails_before_snapshot(
    lab_environment: dict[str, Path],
    profile_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    roots = lab_environment
    _git(
        roots["RPent"],
        "remote",
        "add",
        "origin",
        "git@example.invalid:TestOrg/RPent.git",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "schema": "lab.baseline.v1",
                "baseline_id": "bad-baseline-v1",
                "repositories": {
                    "RPent": {
                        "canonical_identity": "TestOrg/RPent",
                        "commit": _git(roots["RPent"], "rev-parse", "HEAD"),
                        "tree_oid": "f" * 40,
                        "submodules": {},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAB_BASELINE_LOCK", str(baseline_path))
    _write_profile(
        profile_root,
        profile_id="baseline-mismatch",
        argv=["{python}", "-c", "print('not-run')"],
    )
    profile = json.loads(
        (profile_root / "baseline-mismatch.json").read_text(encoding="utf-8")
    )
    profile["commands"][0]["repository"] = "RPent"
    (profile_root / "baseline-mismatch.json").write_text(
        json.dumps(profile),
        encoding="utf-8",
    )
    assert main(["task", "create", "AR-0212", "--repos", "AgenticRobot"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0212", "baseline-mismatch"])

    assert result == 2
    assert "baseline tree mismatch" in capsys.readouterr().err
    assert not (tmp_path / "lab-home" / "Runtime").exists()


def test_clean_profile_rejects_dirty_task_before_creating_run(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _write_profile(
        profile_root,
        profile_id="clean-only",
        argv=["{python}", "-c", "print('not-run')"],
    )
    assert main(["task", "create", "AR-0202", "--repos", "main"]) == 0
    capsys.readouterr()
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    task = manager.load_task("AR-0202")
    checkout = Path(task["repositories"]["AgenticRobot"]["worktree_path"])
    (checkout / "dirty.py").write_text("dirty = True\n", encoding="utf-8")

    result = main(["test", "AR-0202", "clean-only"])

    assert result == 2
    assert "rejects dirty repository" in capsys.readouterr().err
    assert not (tmp_path / "lab-home" / "Runtime").exists()


def test_profile_failure_is_a_recorded_failed_run(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="failing-probe",
        argv=["{python}", "-c", "raise SystemExit(3)"],
    )
    assert main(["task", "create", "AR-0203", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0203", "failing-probe", "--json"])

    assert result == 1
    run = json.loads(capsys.readouterr().out)
    assert run["status"] == "FAILED"
    assert run["outcome"]["classification"] == "FAILED_ASSERTION"
    assert run["commands"][0]["exit_code"] == 3


def test_profile_start_failure_is_recorded_as_infrastructure_failure(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="missing-command",
        argv=["definitely-not-a-real-lab-command"],
    )
    assert main(["task", "create", "AR-0208", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0208", "missing-command", "--json"])

    assert result == 1
    run = json.loads(capsys.readouterr().out)
    assert run["outcome"]["classification"] == "INFRASTRUCTURE_FAILURE"
    assert run["outcome"]["execution_completed"] is False
    assert run["commands"][0]["start_error"]


def test_profile_timeout_is_recorded_and_process_is_stopped(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="timeout-probe",
        argv=["{python}", "-c", "import time; time.sleep(10)"],
        timeout_seconds=1,
    )
    assert main(["task", "create", "AR-0209", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0209", "timeout-probe", "--json"])

    assert result == 1
    run = json.loads(capsys.readouterr().out)
    assert run["outcome"]["classification"] == "INFRASTRUCTURE_FAILURE"
    assert run["commands"][0]["timed_out"] is True


def test_profile_that_changes_snapshot_is_invalid_source(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="source-mutator",
        argv=[
            "{python}",
            "-c",
            "from pathlib import Path; Path('generated.py').write_text('changed')",
        ],
    )
    assert main(["task", "create", "AR-0204", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0204", "source-mutator", "--json"])

    assert result == 1
    run = json.loads(capsys.readouterr().out)
    assert run["outcome"]["classification"] == "INVALID_SOURCE"
    assert run["outcome"]["source_stable"] is False


def test_m2_rejects_profiles_above_local_risk_boundary(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="device-profile",
        argv=["{python}", "-c", "print('must-not-run')"],
        risk_level="H1",
    )
    assert main(["task", "create", "AR-0205", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0205", "device-profile"])

    assert result == 2
    assert "only permits R0/R1" in capsys.readouterr().err


def test_open_run_rejects_manifest_snapshot_path_tampering(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_profile(
        profile_root,
        profile_id="path-receipt-probe",
        argv=["{python}", "-c", "print('ok')"],
    )
    assert main(["task", "create", "AR-0207", "--repos", "main"]) == 0
    capsys.readouterr()
    assert main(["test", "AR-0207", "path-receipt-probe", "--json"]) == 0
    run = json.loads(capsys.readouterr().out)
    manifest_path = Path(run["run_root"]) / "run.yaml"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_workspace"] = str(
        lab_environment["RPent"] / "wrong.code-workspace"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = main(["open", "--run", run["run_id"], "--print"])

    assert result == 2
    assert "snapshot workspace mismatch" in capsys.readouterr().err


def test_profile_python_requirement_fails_before_snapshot(
    lab_environment: dict[str, Path],
    profile_root: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_path = _write_profile(
        profile_root,
        profile_id="wrong-python",
        argv=["{python}", "-c", "print('must-not-run')"],
    )
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["python_major_minor"] = "0.0"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    assert main(["task", "create", "AR-0206", "--repos", "main"]) == 0
    capsys.readouterr()

    result = main(["test", "AR-0206", "wrong-python"])

    assert result == 2
    assert "requires Python 0.0" in capsys.readouterr().err


def test_dev_doctor_reports_python_and_checkout_provenance() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = load_dev_config(repository / "integration" / "dev.json")

    payload = inspect_environment(repository)

    assert payload["repository"]["root"] == str(repository)
    assert payload["python"]["required_major_minor"] == "3.10"
    assert payload["python"]["compatible"] is (
        (sys.version_info.major, sys.version_info.minor)
        == config.python_major_minor
    )
    assert all(
        raw_import["within_checkout"] for raw_import in payload["imports"].values()
    )
    assert tuple(payload["imports"]) == config.import_modules
    assert payload["dev_config"]["sha256"] == config.sha256


def test_checked_in_lab_control_profile_is_local_clean_r0() -> None:
    repository = Path(__file__).resolve().parents[1]

    profile = load_profile(
        repository / "integration" / "profiles",
        "lab-control-unit",
    )

    assert profile.execution == "local"
    assert profile.risk_level == "R0"
    assert profile.requires_clean is True
    assert profile.python_major_minor is None
    assert profile.repositories == ("AgenticRobot",)
    assert profile.commands[0].environment_id == "agenticrobot-py310"


def test_checked_in_repository_environments_bind_expected_conda_bases() -> None:
    repository = Path(__file__).resolve().parents[1]
    root = repository / "integration" / "environments"

    agenticrobot = load_environment_profile(root, "agenticrobot-py310")
    rpent = load_environment_profile(root, "rpent-py311")
    openpi = load_environment_profile(root, "openpi-source-py311")

    assert agenticrobot.repository == "AgenticRobot"
    assert agenticrobot.resolver_kind == "conda"
    assert agenticrobot.resolver_value == "dexteleop310"
    assert agenticrobot.python_major_minor == "3.10"
    assert rpent.repository == "RPent"
    assert rpent.resolver_value == "rpent"
    assert rpent.python_major_minor == "3.11"
    assert openpi.repository == "openpi"
    assert openpi.resolver_value == "rpent"
    assert openpi.python_major_minor == "3.11"
    assert openpi.import_modules == ("openpi", "openpi_client")
    assert openpi.source_roots == ("src", "packages/openpi-client/src")

    cross_profile = load_profile(
        repository / "integration" / "profiles",
        "agenticrobot-rpent-contract",
    )
    assert cross_profile.repositories == ("AgenticRobot", "RPent")
    assert [command.environment_id for command in cross_profile.commands] == [
        "agenticrobot-py310",
        "rpent-py311",
    ]

    openpi_profile = load_profile(
        repository / "integration" / "profiles",
        "openpi-source-anchor",
    )
    assert openpi_profile.repositories == ("openpi",)
    assert openpi_profile.risk_level == "R0"
    assert openpi_profile.commands[0].environment_id == "openpi-source-py311"


def test_checked_in_baseline_pins_all_three_repositories() -> None:
    repository = Path(__file__).resolve().parents[1]

    baseline = load_baseline_lock(repository / "stack" / "baseline.json")

    assert (
        baseline.baseline_id
        == "baseline-20260815-openpi-submodule-candidate"
    )
    assert set(baseline.repositories) == {"AgenticRobot", "RPent", "openpi"}
    assert all(len(pin.commit) == 40 for pin in baseline.repositories.values())
    assert all(len(pin.tree_oid) == 40 for pin in baseline.repositories.values())
    assert baseline.repositories["openpi"].submodules == {
        "third_party/aloha": "d1dc83afd89ded4379851257fe5d85632d31d5ec",
        "third_party/libero": "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c",
    }
    assert baseline.repositories["AgenticRobot"].submodules == {
        "model/openpi": "0d2d7f4e5526cca5f65e17b34f17ca8adad17407"
    }
    assert [
        (
            link.parent_repository,
            link.submodule_path,
            link.child_repository,
        )
        for link in baseline.repository_links
    ] == [("AgenticRobot", "model/openpi", "openpi")]


@pytest.mark.parametrize(
    "remote_url",
    [
        "git@github.com:CYBER-QAQ/AgenticRobot.git",
        "git@github-cyber-qaq:CYBER-QAQ/AgenticRobot.git",
        "ssh://git@github.com/CYBER-QAQ/AgenticRobot.git",
        "https://github.com/CYBER-QAQ/AgenticRobot.git",
    ],
)
def test_repository_identity_ignores_transport_and_ssh_alias(remote_url: str) -> None:
    assert lab_git.canonical_repository_identity(remote_url) == "CYBER-QAQ/AgenticRobot"


@pytest.mark.parametrize("task_id", ["../escape", "bad/id", ".", "", "a" * 65])
def test_create_rejects_unsafe_task_ids_before_writing(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
    task_id: str,
) -> None:
    result = main(["task", "create", task_id, "--repos", "main"])

    assert result == 2
    assert "task ID" in capsys.readouterr().err

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.command_execution import (
    TrustedCommandExecutor,
    load_runner_command_map,
)
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


def _repository(path: Path) -> Path:
    path.mkdir(parents=True)
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Command Execution Test")
    _git(path, "config", "user.email", "command-test@example.invalid")
    (path / "source.txt").write_text("stable\n", encoding="utf-8")
    _git(path, "add", "source.txt")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _write_queue_profile(root: Path) -> None:
    (root / "contract-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v1",
                "profile_id": "contract-check",
                "description": "Generic protected command fixture",
                "target_id": "test-target",
                "risk_level": "R0",
                "requires_clean": True,
                "repositories": ["Controller"],
                "command_ids": ["contract-suite"],
                "capabilities": ["system-command"],
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
                "description": "Generic command runner fixture",
                "target_ids": ["test-target"],
                "slots": ["slot-a1"],
                "capabilities": ["system-command"],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )


def _write_command_map(root: Path, argv: list[str]) -> None:
    (root / "runner-a.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-command-map.v1",
                "runner_id": "runner-a",
                "commands": [
                    {
                        "command_id": "contract-suite",
                        "repository": "Controller",
                        "argv": argv,
                        "timeout_seconds": 30,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def command_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    WorkspaceManager,
    RunRequestManager,
    RunnerCoordinator,
    ExactSnapshotStager,
    Path,
    Path,
    str,
]:
    controller = _repository(tmp_path / "repositories" / "Controller")
    project_manifest = tmp_path / "project.json"
    project_manifest.write_text(
        json.dumps(
            {
                "schema": "lab.project.v3",
                "project_id": "command-stack",
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
    queue_root = tmp_path / "queue-profiles"
    runner_root = tmp_path / "runner-profiles"
    command_root = tmp_path / "runner-commands"
    slot_root = tmp_path / "runner-slots"
    queue_root.mkdir()
    runner_root.mkdir()
    command_root.mkdir()
    _write_queue_profile(queue_root)
    _write_runner_profile(runner_root)
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(project_manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(controller))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_QUEUE_PROFILE_ROOT", str(queue_root))
    monkeypatch.setenv("LAB_RUNNER_PROFILE_ROOT", str(runner_root))
    monkeypatch.setenv("LAB_RUNNER_COMMAND_ROOT", str(command_root))
    monkeypatch.setenv("LAB_RUNNER_SLOT_ROOT", str(slot_root))
    monkeypatch.setenv("LAB_REQUEST_STATE_ROOT", str(tmp_path / "request-state"))
    control_repository = Path(__file__).resolve().parents[1]
    manager = WorkspaceManager(control_repository=control_repository)
    requests = RunRequestManager(manager)
    coordinator = RunnerCoordinator(
        requests.store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(control_repository),
    )
    task = manager.create_task(
        manager.plan_task(
            task_id="T701",
            repository_names=["Controller"],
            scope_assignments=["Controller=**"],
        )
    )
    assert task["task_id"] == "T701"
    submission = requests.submit_clean_profile(
        "T701", "contract-check", evidence_tier="Development"
    )
    request_id = str(submission["request"]["request_id"])
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    stager = ExactSnapshotStager(
        coordinator,
        repository_catalog=manager.catalog,
        slot_root=slot_root,
    )
    stager.stage(request_id, "runner-a")
    return (
        manager,
        requests,
        coordinator,
        stager,
        command_root,
        slot_root,
        request_id,
    )


def test_protected_system_command_executes_once_and_records_logs(
    command_environment: tuple[
        WorkspaceManager,
        RunRequestManager,
        RunnerCoordinator,
        ExactSnapshotStager,
        Path,
        Path,
        str,
    ],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, requests, coordinator, stager, command_root, slot_root, request_id = (
        command_environment
    )
    _write_command_map(
        command_root,
        [
            sys.executable,
            "-c",
            "import os; from pathlib import Path; print(Path(os.environ['LAB_EXECUTION_CONTEXT']).is_file()); print(Path.cwd().joinpath('source.txt').read_text().strip())",
        ],
    )

    assert (
        main(
            [
                "runner",
                "execute",
                "runner-a",
                request_id,
                "--slot-root",
                str(slot_root),
                "--command-root",
                str(command_root),
                "--json",
            ]
        )
        == 0
    )
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["classification"] == "PASSED"
    assert receipt["commands"][0]["exit_code"] == 0
    assert requests.store.load(request_id)["state"] == "COMPLETED"
    log_path = slot_root / "slot-a1" / request_id / receipt["commands"][0]["log"]
    assert "True\nstable" in log_path.read_text(encoding="utf-8")

    repeated = TrustedCommandExecutor(
        coordinator, stager, command_root=command_root
    ).execute(request_id, "runner-a")
    assert repeated == receipt


def test_nonzero_command_is_failed_assertion_with_log_evidence(
    command_environment: tuple[
        WorkspaceManager,
        RunRequestManager,
        RunnerCoordinator,
        ExactSnapshotStager,
        Path,
        Path,
        str,
    ],
) -> None:
    _, requests, coordinator, stager, command_root, _, request_id = command_environment
    _write_command_map(
        command_root,
        [sys.executable, "-c", "print('failed check'); raise SystemExit(7)"],
    )

    receipt = TrustedCommandExecutor(
        coordinator, stager, command_root=command_root
    ).execute(request_id, "runner-a")

    assert receipt["classification"] == "FAILED_ASSERTION"
    assert receipt["commands"][0]["exit_code"] == 7
    state = requests.store.load(request_id)
    assert state["state"] == "FAILED"
    assert state["outcome"]["classification"] == "FAILED_ASSERTION"
    artifacts = requests.store.load_attachment(request_id, "artifact-receipt")
    assert artifacts["evidence_complete"] is True
    assert len(artifacts["artifacts"]) == 1


def test_command_map_is_allowlist_and_rejects_runtime_specific_fields(
    command_environment: tuple[
        WorkspaceManager,
        RunRequestManager,
        RunnerCoordinator,
        ExactSnapshotStager,
        Path,
        Path,
        str,
    ],
) -> None:
    _, requests, coordinator, stager, command_root, _, request_id = command_environment
    _write_command_map(command_root, [sys.executable, "-c", "print('not run')"])
    path = command_root / "runner-a.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["resources"] = ["project-specific"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="unexpected shape"):
        load_runner_command_map(command_root, "runner-a")
    with pytest.raises(LabError, match="unexpected shape"):
        TrustedCommandExecutor(
            coordinator, stager, command_root=command_root
        ).execute(request_id, "runner-a")
    assert requests.store.load(request_id)["state"] == "PREPARING"


def test_command_that_changes_staged_source_is_invalid_source(
    command_environment: tuple[
        WorkspaceManager,
        RunRequestManager,
        RunnerCoordinator,
        ExactSnapshotStager,
        Path,
        Path,
        str,
    ],
) -> None:
    _, requests, coordinator, stager, command_root, _, request_id = command_environment
    _write_command_map(
        command_root,
        [
            sys.executable,
            "-c",
            "from pathlib import Path; Path('source.txt').write_text('changed')",
        ],
    )

    receipt = TrustedCommandExecutor(
        coordinator, stager, command_root=command_root
    ).execute(request_id, "runner-a")

    assert receipt["classification"] == "INVALID_SOURCE"
    assert receipt["source_stable"] is False
    assert requests.store.load(request_id)["outcome"]["classification"] == "INVALID_SOURCE"


def test_cancel_request_terminates_the_registered_command(
    command_environment: tuple[
        WorkspaceManager,
        RunRequestManager,
        RunnerCoordinator,
        ExactSnapshotStager,
        Path,
        Path,
        str,
    ],
) -> None:
    _, requests, coordinator, stager, command_root, _, request_id = command_environment
    _write_command_map(command_root, [sys.executable, "-c", "import time; time.sleep(30)"])
    executor = TrustedCommandExecutor(coordinator, stager, command_root=command_root)
    result: list[dict[str, object]] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            result.append(executor.execute(request_id, "runner-a"))
        except (LabError, OSError) as exc:  # pragma: no cover - diagnostic preservation
            errors.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 5
    while requests.store.load(request_id)["state"] != "RUNNING":
        if time.monotonic() >= deadline:
            pytest.fail("command did not enter RUNNING state")
        time.sleep(0.02)
    requests.store.cancel(request_id, actor="developer", reason="test cancellation")
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert errors == []
    assert result[0]["classification"] == "CANCELLED"
    assert result[0]["commands"][0]["cancelled"] is True
    assert requests.store.load(request_id)["state"] == "CANCELLED"

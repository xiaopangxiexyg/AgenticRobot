from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.run_requests import RunRequestManager
from lab_control.runner_protocol import RunnerCoordinator, load_runner_profile
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
    _git(path, "config", "user.name", "Runner Test")
    _git(path, "config", "user.email", "runner-test@example.invalid")
    (path / "README.md").write_text("# Controller\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


@pytest.fixture
def runner_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path]:
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
    queue_root = tmp_path / "queue-profiles"
    queue_root.mkdir()
    (queue_root / "contract-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v1",
                "profile_id": "contract-check",
                "description": "Generic contract fixture",
                "target_id": "test-target",
                "risk_level": "R0",
                "requires_clean": True,
                "repositories": ["Controller"],
                "command_ids": ["contract-suite"],
                "capabilities": ["python"],
            }
        ),
        encoding="utf-8",
    )
    runner_root = tmp_path / "runner-profiles"
    runner_root.mkdir()
    (runner_root / "runner-a.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-profile.v1",
                "runner_id": "runner-a",
                "description": "Generic runner fixture",
                "target_ids": ["test-target"],
                "slots": ["slot-a1"],
                "capabilities": ["python"],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(project_manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(controller))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_QUEUE_PROFILE_ROOT", str(queue_root))
    monkeypatch.setenv("LAB_RUNNER_PROFILE_ROOT", str(runner_root))
    monkeypatch.setenv("LAB_REQUEST_STATE_ROOT", str(tmp_path / "request-state"))
    control_repository = Path(__file__).resolve().parents[1]
    workspace = WorkspaceManager(control_repository=control_repository)
    requests = RunRequestManager(workspace)
    coordinator = RunnerCoordinator(
        requests.store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(control_repository),
    )
    return workspace, requests, coordinator, queue_root, runner_root


def _create_task(manager: WorkspaceManager, task_id: str) -> None:
    manager.create_task(
        manager.plan_task(
            task_id=task_id,
            repository_names=["Controller"],
            scope_assignments=["Controller=**"],
        )
    )


def _submit(
    manager: WorkspaceManager,
    requests: RunRequestManager,
    task_id: str,
) -> str:
    _create_task(manager, task_id)
    submission = requests.submit_clean_profile(
        task_id, "contract-check", evidence_tier="Development"
    )
    request = submission["request"]
    assert isinstance(request, dict)
    return str(request["request_id"])


def _artifact(*, suffix: str = "") -> dict[str, object]:
    return {
        "artifact_id": f"stdout{suffix}",
        "relative_path": f"logs/stdout{suffix}.log",
        "media_type": "text/plain",
        "byte_size": 17,
        "sha256": "a" * 64,
        "storage_id": "run-store",
    }


def test_runner_inspects_claims_and_reuses_its_active_claim(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    manager, requests, coordinator, _, _ = runner_environment
    request_id = _submit(manager, requests, "T401")

    inspection = coordinator.inspect("runner-a")
    assert inspection["eligible_count"] == 1
    assert inspection["pending"][0]["request_id"] == request_id
    assert inspection["pending"][0]["reasons"] == []
    with pytest.raises(LabError, match="does not belong"):
        coordinator.claim_next("runner-a", "slot-unknown")

    claim = coordinator.claim_next("runner-a", "slot-a1")
    assert claim is not None
    assert claim["request_id"] == request_id
    assert claim["runner_id"] == "runner-a"
    assert requests.store.load(request_id)["state"] == "RESERVED"
    assert coordinator.claim_next("runner-a", "slot-a1") == claim


def test_runner_recovers_a_reservation_if_claim_persistence_was_interrupted(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    manager, requests, coordinator, _, _ = runner_environment
    request_id = _submit(manager, requests, "T402")
    requests.store.reserve(request_id, slot_id="slot-a1", actor="runner-a")
    assert not requests.store.has_attachment(request_id, "runner-claim")

    recovered = coordinator.claim_next("runner-a", "slot-a1")

    assert recovered is not None
    assert recovered["request_id"] == request_id
    assert requests.store.has_attachment(request_id, "runner-claim")


def test_profile_drift_makes_a_pending_request_ineligible(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    manager, requests, coordinator, queue_root, _ = runner_environment
    _submit(manager, requests, "T403")
    profile_path = queue_root / "contract-check.json"
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    payload["description"] = "changed after submission"
    profile_path.write_text(json.dumps(payload), encoding="utf-8")

    inspection = coordinator.inspect("runner-a")

    assert inspection["eligible_count"] == 0
    reasons = inspection["pending"][0]["reasons"]
    assert "queue Profile hash differs" in reasons[0]
    assert coordinator.claim_next("runner-a", "slot-a1") is None


def test_runner_lifecycle_requires_a_valid_artifact_receipt(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    manager, requests, coordinator, _, _ = runner_environment
    request_id = _submit(manager, requests, "T404")
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    coordinator.begin_preparing(request_id, "runner-a")
    coordinator.begin_running(request_id, "runner-a")
    coordinator.begin_collecting(request_id, "runner-a")

    with pytest.raises(LabError, match="invalid artifact relative path"):
        coordinator.record_artifact_receipt(
            request_id,
            "runner-a",
            artifacts=[{**_artifact(), "relative_path": "../escape.log"}],
            evidence_complete=True,
        )
    receipt = coordinator.record_artifact_receipt(
        request_id,
        "runner-a",
        artifacts=[_artifact()],
        evidence_complete=True,
    )
    assert receipt["artifacts"][0]["storage_id"] == "run-store"
    assert (
        coordinator.record_artifact_receipt(
            request_id,
            "runner-a",
            artifacts=[_artifact()],
            evidence_complete=True,
        )
        == receipt
    )
    with pytest.raises(LabError, match="changed its content"):
        coordinator.record_artifact_receipt(
            request_id,
            "runner-a",
            artifacts=[_artifact(suffix="-other")],
            evidence_complete=True,
        )

    completed = coordinator.complete(
        request_id, "runner-a", detail="registered assertions passed"
    )
    assert completed["state"] == "COMPLETED"
    assert completed["outcome"]["classification"] == "PASSED"


def test_incomplete_evidence_cannot_be_promoted_to_completed(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    manager, requests, coordinator, _, _ = runner_environment
    request_id = _submit(manager, requests, "T405")
    coordinator.claim_next("runner-a", "slot-a1")
    coordinator.begin_preparing(request_id, "runner-a")
    coordinator.begin_running(request_id, "runner-a")
    coordinator.begin_collecting(request_id, "runner-a")
    coordinator.record_artifact_receipt(
        request_id,
        "runner-a",
        artifacts=[],
        evidence_complete=False,
    )

    with pytest.raises(LabError, match="incomplete artifact evidence"):
        coordinator.complete(request_id, "runner-a", detail="not sufficient")


def test_runner_acknowledges_cancellation_only_after_claim_validation(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    manager, requests, coordinator, _, _ = runner_environment
    request_id = _submit(manager, requests, "T406")
    coordinator.claim_next("runner-a", "slot-a1")
    requests.store.cancel(
        request_id, actor="developer", reason="stop after registered teardown"
    )

    state = coordinator.acknowledge_cancel(
        request_id, "runner-a", detail="registered processes stopped"
    )

    assert state["state"] == "CANCELLED"


def test_runner_profile_symlinks_are_rejected(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
) -> None:
    _, _, _, _, runner_root = runner_environment
    (runner_root / "runner-link.json").symlink_to(runner_root / "runner-a.json")

    with pytest.raises(LabError, match="unknown or unsafe runner Profile"):
        load_runner_profile(runner_root, "runner-link")


def test_cli_inspects_and_claims_without_executing_payload(
    runner_environment: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, Path
    ],
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, requests, _, _, _ = runner_environment
    request_id = _submit(manager, requests, "T407")

    assert main(["runner", "inspect", "runner-a", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    assert inspection["eligible_count"] == 1

    assert main(["runner", "claim", "runner-a", "slot-a1", "--json"]) == 0
    claim = json.loads(capsys.readouterr().out)
    assert claim["request_id"] == request_id
    assert requests.store.load(request_id)["state"] == "RESERVED"

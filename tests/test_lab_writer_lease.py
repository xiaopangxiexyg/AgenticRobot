from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.workspace import LabError, WorkspaceManager
from lab_control.writer_lease import WriterLeaseManager, writer_lease_fingerprint


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
def lab_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Path]:
    repositories = tmp_path / "repositories"
    roots = {
        "AgenticRobot": _repository(repositories / "AgenticRobot"),
        "RPent": _repository(repositories / "RPent"),
        "openpi": _repository(repositories / "openpi"),
    }
    for name, root in roots.items():
        monkeypatch.setenv(f"LAB_REPOSITORY_{name.upper()}", str(root))
    monkeypatch.setenv(
        "LAB_PROJECT_MANIFEST",
        str(Path(__file__).resolve().parents[1] / "integration" / "project.json"),
    )
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    return roots


def _manager_with_task(
    lab_environment: dict[str, Path],
    *,
    task_id: str = "LEASE-0001",
) -> WorkspaceManager:
    manager = WorkspaceManager(control_repository=lab_environment["AgenticRobot"])
    manager.create_task(
        manager.plan_task(
            task_id=task_id,
            repository_names=["AgenticRobot", "RPent"],
        )
    )
    return manager


def test_different_agents_can_claim_different_repositories_but_not_the_same_one(
    lab_environment: dict[str, Path],
) -> None:
    manager = _manager_with_task(lab_environment)
    writer = WriterLeaseManager(manager)
    assert not (manager.state_root / "writer-leases").exists()

    initial = writer.inspect_task("LEASE-0001")
    assert initial["status"] == "READY"
    assert {
        name: status["state"]
        for name, status in initial["repositories"].items()
    } == {"AgenticRobot": "AVAILABLE", "RPent": "AVAILABLE"}
    assert not (manager.state_root / "writer-leases").exists()

    main_lease = writer.claim(
        "LEASE-0001",
        "main",
        actor="agent-main-01",
    )
    component_lease = writer.claim(
        "LEASE-0001",
        "RPent",
        actor="agent-rpent-01",
    )

    assert main_lease["repository"] == "AgenticRobot"
    assert component_lease["repository"] == "RPent"
    assert writer.claim(
        "LEASE-0001",
        "AgenticRobot",
        actor="agent-main-01",
    ) == main_lease
    with pytest.raises(LabError, match="held by agent-main-01"):
        writer.claim(
            "LEASE-0001",
            "AgenticRobot",
            actor="agent-main-02",
        )

    current = writer.inspect_task("LEASE-0001")
    assert current["repositories"]["AgenticRobot"]["state"] == "ACTIVE"
    assert current["repositories"]["RPent"]["state"] == "ACTIVE"
    fingerprint = writer_lease_fingerprint(current)
    assert fingerprint["AgenticRobot"]["lease_id"] == main_lease["lease_id"]
    assert fingerprint["RPent"]["lease_id"] == component_lease["lease_id"]


def test_writer_cli_and_task_inspection_share_one_authoritative_state(
    lab_environment: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    _manager_with_task(lab_environment, task_id="LEASE-CLI-01")

    assert (
        main(
            [
                "task",
                "writer",
                "claim",
                "LEASE-CLI-01",
                "--repo",
                "main",
                "--actor",
                "agent-main-01",
                "--ttl-seconds",
                "600",
                "--json",
            ]
        )
        == 0
    )
    lease = json.loads(capsys.readouterr().out)
    assert lease["repository"] == "AgenticRobot"

    assert main(["inspect", "LEASE-CLI-01", "--json"]) == 0
    inspection = json.loads(capsys.readouterr().out)
    writer_status = inspection["repositories"]["AgenticRobot"]["writer_lease"]
    assert writer_status["state"] == "ACTIVE"
    assert writer_status["lease"]["actor"] == "agent-main-01"
    assert inspection["writer_coordination"]["active_repositories"] == [
        "AgenticRobot"
    ]

    assert (
        main(
            [
                "task",
                "writer",
                "release",
                "LEASE-CLI-01",
                "--repo",
                "AgenticRobot",
                "--actor",
                "agent-main-01",
                "--lease-id",
                lease["lease_id"],
                "--json",
            ]
        )
        == 0
    )
    released = json.loads(capsys.readouterr().out)
    assert released["status"] == "RELEASED"


def test_writer_lease_renew_and_release_are_exact_and_auditable(
    lab_environment: dict[str, Path],
) -> None:
    manager = _manager_with_task(lab_environment, task_id="LEASE-0002")
    now = [datetime(2026, 8, 16, 1, 0, tzinfo=timezone.utc)]
    writer = WriterLeaseManager(manager, clock=lambda: now[0])
    lease = writer.claim(
        "LEASE-0002",
        "AgenticRobot",
        actor="agent-main-01",
        ttl_seconds=600,
    )
    now[0] += timedelta(seconds=120)

    renewed = writer.renew(
        "LEASE-0002",
        "AgenticRobot",
        actor="agent-main-01",
        lease_id=str(lease["lease_id"]),
        ttl_seconds=900,
    )

    assert renewed["revision"] == 2
    assert renewed["claimed_at"] == lease["claimed_at"]
    assert renewed["receipt_sha256"] != lease["receipt_sha256"]
    with pytest.raises(LabError, match="identity mismatch"):
        writer.release(
            "LEASE-0002",
            "AgenticRobot",
            actor="agent-other",
            lease_id=str(lease["lease_id"]),
        )

    released = writer.release(
        "LEASE-0002",
        "AgenticRobot",
        actor="agent-main-01",
        lease_id=str(lease["lease_id"]),
    )
    assert released["status"] == "RELEASED"
    assert released["released_revision"] == 2
    assert writer.release(
        "LEASE-0002",
        "AgenticRobot",
        actor="agent-main-01",
        lease_id=str(lease["lease_id"]),
    ) == released
    assert writer.inspect_repository(
        "LEASE-0002", "AgenticRobot"
    )["state"] == "AVAILABLE"
    events = (
        manager.state_root
        / "writer-leases"
        / "LEASE-0002"
        / "AgenticRobot"
        / "events"
    )
    assert sorted(path.name for path in events.iterdir()) == [
        f"{lease['lease_id']}-claim.json",
        f"{lease['lease_id']}-release.json",
        f"{lease['lease_id']}-renew-2.json",
    ]


def test_expired_writer_lease_requires_exact_explicit_replacement(
    lab_environment: dict[str, Path],
) -> None:
    manager = _manager_with_task(lab_environment, task_id="LEASE-0003")
    now = [datetime(2026, 8, 16, 2, 0, tzinfo=timezone.utc)]
    writer = WriterLeaseManager(manager, clock=lambda: now[0])
    old = writer.claim(
        "LEASE-0003",
        "AgenticRobot",
        actor="agent-old",
        ttl_seconds=60,
    )
    now[0] += timedelta(seconds=61)

    status = writer.inspect_repository("LEASE-0003", "AgenticRobot")
    assert status["state"] == "EXPIRED"
    assert status["claim_allowed"] is False
    assert status["replace_expired_required"] is True
    with pytest.raises(LabError, match="exact --replace-expired"):
        writer.claim(
            "LEASE-0003",
            "AgenticRobot",
            actor="agent-new",
        )
    with pytest.raises(LabError, match="exact --replace-expired"):
        writer.claim(
            "LEASE-0003",
            "AgenticRobot",
            actor="agent-new",
            replace_expired="WL-wrong",
        )

    replacement = writer.claim(
        "LEASE-0003",
        "AgenticRobot",
        actor="agent-new",
        replace_expired=str(old["lease_id"]),
    )
    assert replacement["actor"] == "agent-new"
    assert replacement["replaces_lease_id"] == old["lease_id"]
    assert writer.inspect_repository(
        "LEASE-0003", "AgenticRobot"
    )["state"] == "ACTIVE"


def test_writer_lease_refuses_tampering_and_concurrent_claims(
    lab_environment: dict[str, Path],
) -> None:
    manager = _manager_with_task(lab_environment, task_id="LEASE-0004")
    writer = WriterLeaseManager(manager)
    with manager.advisory_locks(
        "writer-leases", ["LEASE-0004:AgenticRobot"]
    ), pytest.raises(LabError, match="another lab operation"):
        writer.claim(
            "LEASE-0004",
            "AgenticRobot",
            actor="agent-main-01",
        )

    lease = writer.claim(
        "LEASE-0004",
        "AgenticRobot",
        actor="agent-main-01",
    )
    active = (
        manager.state_root
        / "writer-leases"
        / "LEASE-0004"
        / "AgenticRobot"
        / "active.json"
    )
    payload = json.loads(active.read_text(encoding="utf-8"))
    payload["actor"] = "agent-tampered"
    active.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabError, match="receipt hash mismatch"):
        writer.inspect_repository("LEASE-0004", "AgenticRobot")
    assert lease["actor"] == "agent-main-01"

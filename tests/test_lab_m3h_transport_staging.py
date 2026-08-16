from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control import git as lab_git
from lab_control.cli import main
from lab_control.queue_profiles import load_queue_profile
from lab_control.request_transport import (
    RequestEnvelopeIngress,
    build_request_envelope,
    validate_request_envelope,
)
from lab_control.run_requests import (
    RunRequestManager,
    RunRequestStore,
    build_run_request,
)
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
    _git(path, "config", "user.name", "Transport Test")
    _git(path, "config", "user.email", "transport-test@example.invalid")
    (path / "README.md").write_text(f"# {title}\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _write_queue_profile(root: Path, *, repositories: list[str]) -> None:
    (root / "contract-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v1",
                "profile_id": "contract-check",
                "description": "Generic contract fixture",
                "target_id": "test-target",
                "risk_level": "R0",
                "requires_clean": True,
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


@pytest.fixture
def queued_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, str]:
    controller = _repository(tmp_path / "repositories" / "Controller", title="One")
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
    runner_root = tmp_path / "runner-profiles"
    queue_root.mkdir()
    runner_root.mkdir()
    _write_queue_profile(queue_root, repositories=["Controller"])
    _write_runner_profile(runner_root)
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(project_manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(controller))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_QUEUE_PROFILE_ROOT", str(queue_root))
    monkeypatch.setenv("LAB_RUNNER_PROFILE_ROOT", str(runner_root))
    monkeypatch.setenv("LAB_REQUEST_STATE_ROOT", str(tmp_path / "request-state"))
    control_repository = Path(__file__).resolve().parents[1]
    manager = WorkspaceManager(control_repository=control_repository)
    manager.create_task(
        manager.plan_task(
            task_id="T501",
            repository_names=["Controller"],
            scope_assignments=["Controller=**"],
        )
    )
    requests = RunRequestManager(manager)
    submission = requests.submit_clean_profile(
        "T501", "contract-check", evidence_tier="Development"
    )
    request = submission["request"]
    assert isinstance(request, dict)
    request_id = str(request["request_id"])
    coordinator = RunnerCoordinator(
        requests.store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(control_repository),
    )
    return manager, requests, coordinator, controller, request_id


def test_authenticated_request_envelope_is_replay_safe(
    queued_request: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, str
    ],
    tmp_path: Path,
) -> None:
    _, requests, _, _, request_id = queued_request
    request = requests.store.load_request(request_id)
    issued = datetime(2026, 8, 16, 8, 0, tzinfo=timezone.utc)
    secret = b"s" * 32
    envelope = build_request_envelope(
        request,
        sender_id="controller-a",
        recipient_id="runner-node-b",
        key_id="transport-key-1",
        secret=secret,
        issued_at=issued,
    )
    remote_store = RunRequestStore(tmp_path / "remote-state")
    ingress = RequestEnvelopeIngress(
        remote_store,
        recipient_id="runner-node-b",
        trusted_sender_keys={"controller-a": {"transport-key-1": secret}},
    )

    first = ingress.ingest(envelope, observed_at=issued + timedelta(seconds=1))
    repeated = ingress.ingest(envelope, observed_at=issued + timedelta(seconds=2))

    assert first["created"] is True
    assert repeated["created"] is False
    assert repeated["receipt"] == first["receipt"]
    assert remote_store.load(request_id)["state"] == "SUBMITTED"


def test_request_envelope_rejects_tampering_wrong_peer_and_expiry(
    queued_request: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, str
    ],
) -> None:
    _, requests, _, _, request_id = queued_request
    issued = datetime(2026, 8, 16, 8, 0, tzinfo=timezone.utc)
    secret = b"k" * 32
    envelope = build_request_envelope(
        requests.store.load_request(request_id),
        sender_id="controller-a",
        recipient_id="runner-node-b",
        key_id="key-1",
        secret=secret,
        issued_at=issued,
        ttl_seconds=60,
    )
    trusted = {"controller-a": {"key-1": secret}}
    tampered = json.loads(json.dumps(envelope))
    tampered["target_id"] = "different"
    with pytest.raises(LabError, match="unexpected shape"):
        validate_request_envelope(
            tampered,
            recipient_id="runner-node-b",
            trusted_sender_keys=trusted,
            observed_at=issued,
        )
    with pytest.raises(LabError, match="identity mismatch"):
        validate_request_envelope(
            envelope,
            recipient_id="runner-node-c",
            trusted_sender_keys=trusted,
            observed_at=issued,
        )
    with pytest.raises(LabError, match="expired"):
        validate_request_envelope(
            envelope,
            recipient_id="runner-node-b",
            trusted_sender_keys=trusted,
            observed_at=issued + timedelta(seconds=61),
        )
    with pytest.raises(LabError, match="authentication failed"):
        validate_request_envelope(
            envelope,
            recipient_id="runner-node-b",
            trusted_sender_keys={"controller-a": {"key-1": b"x" * 32}},
            observed_at=issued,
        )


def test_runner_stages_an_exact_clean_snapshot_without_lfs_smudge(
    queued_request: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, str
    ],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, requests, coordinator, controller, request_id = queued_request
    marker = tmp_path / "smudge-was-run"
    (controller / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    (controller / "payload.bin").write_bytes(b"Git blob, not an external asset\n")
    _git(controller, "config", "filter.lfs.smudge", f"touch {marker}")
    _git(controller, "config", "filter.lfs.required", "true")
    _git(controller, "add", ".gitattributes", "payload.bin")
    _git(controller, "commit", "-m", "add filtered source fixture")

    requests.store.cancel(
        request_id,
        actor="controller-a",
        reason="replace fixture with the filtered exact source",
    )
    manager.create_task(
        manager.plan_task(
            task_id="T503",
            repository_names=["Controller"],
            scope_assignments=["Controller=**"],
        )
    )
    submission = requests.submit_clean_profile(
        "T503", "contract-check", evidence_tier="Development"
    )
    replacement = submission["request"]
    assert isinstance(replacement, dict)
    request_id = str(replacement["request_id"])
    assert main(["runner", "claim", "runner-a", "slot-a1", "--json"]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "runner",
                "stage",
                "runner-a",
                request_id,
                "--slot-root",
                str(tmp_path / "slots"),
                "--json",
            ]
        )
        == 0
    )
    staged_output = json.loads(capsys.readouterr().out)
    stager = ExactSnapshotStager(
        coordinator,
        repository_catalog=manager.catalog,
        slot_root=tmp_path / "slots",
    )
    receipt = stager.stage(request_id, "runner-a")
    assert staged_output == receipt
    assert stager.stage(request_id, "runner-a") == receipt

    raw_repository = receipt["repositories"]["Controller"]
    staged = (
        tmp_path
        / "slots"
        / "slot-a1"
        / request_id
        / str(raw_repository["relative_path"])
    )
    request_repository = requests.store.load_request(request_id)["snapshot"][
        "repositories"
    ]["Controller"]
    assert lab_git.current_branch(staged) is None
    assert lab_git.current_commit(staged) == request_repository["commit"]
    assert lab_git.status_short(staged) == ()
    assert not marker.exists()
    assert receipt["network_access"] is False
    assert receipt["lfs_smudge"] is False


def test_stager_fails_closed_when_exact_objects_are_missing(
    queued_request: tuple[
        WorkspaceManager, RunRequestManager, RunnerCoordinator, Path, str
    ],
    tmp_path: Path,
) -> None:
    _, _, coordinator, _, request_id = queued_request
    unrelated = _repository(tmp_path / "unrelated", title="Unrelated")
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    stager = ExactSnapshotStager(
        coordinator,
        repository_catalog={"Controller": unrelated},
        slot_root=tmp_path / "slots",
    )

    with pytest.raises(LabError, match="lacks exact local Git objects"):
        stager.stage(request_id, "runner-a")
    with pytest.raises(LabError, match="incomplete/quarantined"):
        stager.stage(request_id, "runner-a")


def test_unfiltered_staging_rejects_oversized_git_blobs(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "bounded", title="Bounded source")
    target = tmp_path / "oversized-slot"

    with pytest.raises(lab_git.GitError, match="exceeds staging limit"):
        lab_git.create_unfiltered_detached_worktree(
            repository,
            target=target,
            commit=_git(repository, "rev-parse", "HEAD"),
            maximum_blob_bytes=4,
        )

    assert not target.exists()


def test_stager_materializes_a_linked_child_at_the_gitlink_path(
    tmp_path: Path,
) -> None:
    child = _repository(tmp_path / "repositories" / "Component", title="Child")
    child_commit = _git(child, "rev-parse", "HEAD")
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
    _git(parent, "commit", "-am", "add linked component")
    parent_commit = _git(parent, "rev-parse", "HEAD")
    queue_root = tmp_path / "queue-profiles"
    runner_root = tmp_path / "runner-profiles"
    queue_root.mkdir()
    runner_root.mkdir()
    _write_queue_profile(queue_root, repositories=["Controller", "Component"])
    _write_runner_profile(runner_root)
    profile = load_queue_profile(queue_root, "contract-check")
    control_repository = Path(__file__).resolve().parents[1]
    request = build_run_request(
        task_id="T502",
        snapshot_id="snapshot-linked-1",
        snapshot_identity={
            "repositories": {
                "Controller": {
                    "commit": parent_commit,
                    "tree_oid": _git(parent, "rev-parse", f"{parent_commit}^{{tree}}"),
                    "submodules": {"components/child": child_commit},
                },
                "Component": {
                    "commit": child_commit,
                    "tree_oid": _git(child, "rev-parse", f"{child_commit}^{{tree}}"),
                    "submodules": {},
                },
            },
            "repository_links": [
                {
                    "parent_repository": "Controller",
                    "submodule_path": "components/child",
                    "child_repository": "Component",
                    "commit": child_commit,
                }
            ],
        },
        profile=profile,
        evidence_tier="Development",
        control_plane=control_plane_identity(control_repository),
    )
    store = RunRequestStore(tmp_path / "request-state")
    state, created = store.submit(request)
    assert created is True and state["state"] == "SUBMITTED"
    coordinator = RunnerCoordinator(
        store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(control_repository),
    )
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    receipt = ExactSnapshotStager(
        coordinator,
        repository_catalog={"Controller": parent, "Component": child},
        slot_root=tmp_path / "slots",
    ).stage(str(request["request_id"]), "runner-a")

    controller_path = (
        tmp_path
        / "slots"
        / "slot-a1"
        / str(request["request_id"])
        / receipt["repositories"]["Controller"]["relative_path"]
    )
    component_path = controller_path / "components" / "child"
    assert lab_git.current_commit(component_path) == child_commit
    assert lab_git.status_short(component_path) == ()
    assert lab_git.status_short(controller_path) == ()

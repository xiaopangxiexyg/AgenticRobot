from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from lab_control.cli import main
from lab_control.queue_profiles import load_queue_profile
from lab_control.run_requests import (
    RunRequestManager,
    build_run_request,
    validate_run_request,
)
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
    _git(path, "config", "user.name", "Queue Test")
    _git(path, "config", "user.email", "queue-test@example.invalid")
    (path / "README.md").write_text("# Controller\n", encoding="utf-8")
    _git(path, "add", "README.md")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


@pytest.fixture
def request_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[WorkspaceManager, Path]:
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
    profile_root = tmp_path / "queue-profiles"
    profile_root.mkdir()
    (profile_root / "queued-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v1",
                "profile_id": "queued-check",
                "description": "Hermetic queued request fixture",
                "target_id": "fixture-runner",
                "risk_level": "R0",
                "requires_clean": True,
                "repositories": ["Controller"],
                "command_ids": ["controller-check"],
                "capabilities": ["python"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("LAB_PROJECT_MANIFEST", str(project_manifest))
    monkeypatch.setenv("LAB_REPOSITORY_CONTROLLER", str(controller))
    monkeypatch.setenv("LAB_HOME", str(tmp_path / "lab-home"))
    monkeypatch.setenv("LAB_QUEUE_PROFILE_ROOT", str(profile_root))
    monkeypatch.setenv(
        "LAB_REQUEST_STATE_ROOT", str(tmp_path / "request-state")
    )
    control_repository = Path(__file__).resolve().parents[1]
    return WorkspaceManager(control_repository=control_repository), profile_root


def _create_task(manager: WorkspaceManager, task_id: str) -> Path:
    task = manager.create_task(
        manager.plan_task(
            task_id=task_id,
            repository_names=["Controller"],
            scope_assignments=["Controller=**"],
        )
    )
    repositories = task["repositories"]
    assert isinstance(repositories, dict)
    controller = repositories["Controller"]
    assert isinstance(controller, dict)
    return Path(str(controller["worktree_path"]))


def test_detached_submission_is_portable_and_idempotent(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T301")
    requests = RunRequestManager(manager)

    first = requests.submit_clean_profile(
        "T301", "queued-check", evidence_tier="Development"
    )
    second = requests.submit_clean_profile(
        "T301", "queued-check", evidence_tier="Development"
    )

    assert first["created"] is True
    assert second["created"] is False
    assert first["request"] == second["request"]
    encoded = json.dumps(first["request"])
    assert "management_path" not in encoded
    assert "worktree_path" not in encoded
    assert "argv" not in encoded
    control_plane = first["request"]["control_plane"]
    assert set(control_plane) == {
        "version",
        "source_commit",
        "source_tree_oid",
        "source_dirty",
        "executor_sha256",
        "install_manifest_sha256",
    }
    state = first["state"]
    assert isinstance(state, dict)
    assert state["state"] == "SUBMITTED"
    assert state["history"][0]["state"] == "SUBMITTED"
    assert len(str(state["idempotency_key"])) == 64


def test_dirty_task_is_rejected_before_request_creation(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    worktree = _create_task(manager, "T302")
    (worktree / "dirty.txt").write_text("not checkpointed\n", encoding="utf-8")
    requests = RunRequestManager(manager)

    with pytest.raises(LabError, match="rejects dirty repository"):
        requests.submit_clean_profile(
            "T302", "queued-check", evidence_tier="Development"
        )

    assert requests.store.list("T302") == []


def test_slot_reservation_and_full_state_lifecycle(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T303")
    _create_task(manager, "T304")
    requests = RunRequestManager(manager)
    first = requests.submit_clean_profile(
        "T303", "queued-check", evidence_tier="Development"
    )
    second = requests.submit_clean_profile(
        "T304", "queued-check", evidence_tier="Development"
    )
    first_id = str(first["request"]["request_id"])
    second_id = str(second["request"]["request_id"])

    requests.store.reserve(first_id, slot_id="slot-01", actor="runner-01")
    with pytest.raises(LabError, match="changed its slot"):
        requests.store.reserve(first_id, slot_id="slot-other", actor="runner-01")
    with pytest.raises(LabError, match="is held by"):
        requests.store.reserve(second_id, slot_id="slot-01", actor="runner-02")
    requests.store.begin_preparing(first_id, actor="runner-01")
    requests.store.begin_running(first_id, actor="runner-01")
    requests.store.begin_collecting(first_id, actor="runner-01")
    completed = requests.store.complete(
        first_id, actor="runner-01", detail="all assertions passed"
    )

    assert completed["state"] == "COMPLETED"
    assert completed["outcome"] == {
        "classification": "PASSED",
        "detail": "all assertions passed",
    }
    assert len(completed["history"]) == 6
    reserved = requests.store.reserve(
        second_id, slot_id="slot-01", actor="runner-02"
    )
    assert reserved["state"] == "RESERVED"


def test_cancel_is_immediate_before_reservation_and_acknowledged_after(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T305")
    _create_task(manager, "T306")
    requests = RunRequestManager(manager)
    first = requests.submit_clean_profile(
        "T305", "queued-check", evidence_tier="Development"
    )
    first_id = str(first["request"]["request_id"])

    cancelled = requests.store.cancel(
        first_id, actor="developer", reason="no longer needed"
    )
    assert cancelled["state"] == "CANCELLED"
    assert requests.store.cancel(
        first_id, actor="developer", reason="duplicate"
    ) == cancelled

    second = requests.submit_clean_profile(
        "T306", "queued-check", evidence_tier="Development"
    )
    second_id = str(second["request"]["request_id"])
    requests.store.reserve(second_id, slot_id="slot-02", actor="runner")
    requested = requests.store.cancel(
        second_id, actor="developer", reason="stop after teardown"
    )
    assert requested["state"] == "CANCEL_REQUESTED"
    acknowledged = requests.store.acknowledge_cancel(
        second_id, actor="runner", detail="registered processes stopped"
    )
    assert acknowledged["state"] == "CANCELLED"
    assert acknowledged["outcome"]["classification"] == "CANCELLED"


def test_watch_and_event_tampering_are_explicit(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T307")
    requests = RunRequestManager(manager)
    submission = requests.submit_clean_profile(
        "T307", "queued-check", evidence_tier="Development"
    )
    request_id = str(submission["request"]["request_id"])

    changed = requests.store.watch(request_id, after_sequence=-1, timeout_seconds=0)
    unchanged = requests.store.watch(request_id, after_sequence=0, timeout_seconds=0)
    assert changed["changed"] is True
    assert unchanged["changed"] is False

    event = (
        requests.store.requests_root
        / request_id
        / "events"
        / "000000-SUBMITTED.json"
    )
    payload = json.loads(event.read_text(encoding="utf-8"))
    payload["actor"] = "tampered"
    event.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(LabError, match="event hash mismatch"):
        requests.store.load(request_id)


def test_state_summary_must_match_the_immutable_event_chain(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T307B")
    requests = RunRequestManager(manager)
    submission = requests.submit_clean_profile(
        "T307B", "queued-check", evidence_tier="Development"
    )
    request_id = str(submission["request"]["request_id"])

    state_path = requests.store.requests_root / request_id / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["history"][0]["sequence"] = 9
    state_path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(LabError, match="sequence is not contiguous"):
        requests.store.load(request_id)


def test_queue_profile_symlinks_are_rejected(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    _, profile_root = request_environment
    source = profile_root / "queued-check.json"
    alias = profile_root / "linked-check.json"
    alias.symlink_to(source)

    with pytest.raises(LabError, match="unknown or unsafe queue Profile"):
        load_queue_profile(profile_root, "linked-check")


def test_repository_links_are_bound_without_project_specific_logic(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    _, profile_root = request_environment
    profile = load_queue_profile(profile_root, "queued-check")
    child_commit = "3" * 40
    identity = {
        "repositories": {
            "Parent": {
                "commit": "1" * 40,
                "tree_oid": "2" * 40,
                "submodules": {"components/child": child_commit},
                "management_path": "/not/portable",
            },
            "Child": {
                "commit": child_commit,
                "tree_oid": "4" * 40,
                "submodules": {},
            },
        },
        "repository_links": [
            {
                "parent_repository": "Parent",
                "submodule_path": "components/child",
                "child_repository": "Child",
                "commit": child_commit,
            }
        ],
    }
    request = build_run_request(
        task_id="T308",
        snapshot_id="snapshot-308",
        snapshot_identity=identity,
        profile=profile,
        evidence_tier="Development",
        control_plane={
            "version": "test-control-v1",
            "source_commit": "6" * 40,
            "source_tree_oid": "7" * 40,
            "source_dirty": False,
            "executor_sha256": "8" * 64,
            "install_manifest_sha256": None,
        },
    )
    assert request["snapshot"]["repository_links"][0]["commit"] == child_commit

    tampered = copy.deepcopy(request)
    tampered["snapshot"]["repositories"]["Child"]["commit"] = "5" * 40
    with pytest.raises(LabError, match="differs from snapshot identities"):
        validate_run_request(tampered)


def test_v2_reader_keeps_m3f_v1_requests_readable(
    request_environment: tuple[WorkspaceManager, Path],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T308B")
    submission = RunRequestManager(manager).submit_clean_profile(
        "T308B", "queued-check", evidence_tier="Development"
    )
    request = copy.deepcopy(submission["request"])
    request["schema"] = "lab.run-request.v1"
    del request["control_plane"]["install_manifest_sha256"]
    content = {
        key: value
        for key, value in request.items()
        if key not in {"request_id", "idempotency_key"}
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    idempotency_key = hashlib.sha256(encoded).hexdigest()
    request["idempotency_key"] = idempotency_key
    request["request_id"] = f"Q-{idempotency_key[:24]}"

    assert validate_run_request(request) == request


def test_cli_submits_statuses_and_cancels_without_execution(
    request_environment: tuple[WorkspaceManager, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    manager, _ = request_environment
    _create_task(manager, "T309")

    assert main(["test", "T309", "queued-check", "--detach", "--json"]) == 0
    submission = json.loads(capsys.readouterr().out)
    request_id = submission["request"]["request_id"]
    assert submission["state"]["state"] == "SUBMITTED"

    assert main(["requests", "status", request_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "SUBMITTED"
    assert (
        main(
            [
                "requests",
                "cancel",
                request_id,
                "--actor",
                "developer",
                "--reason",
                "test complete",
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["state"] == "CANCELLED"

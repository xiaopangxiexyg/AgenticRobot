from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lab_control.command_execution import TrustedCommandExecutor
from lab_control.queue_profiles import load_queue_profile
from lab_control.result_transfer import ResultPackageBuilder, ResultPackageIngress
from lab_control.run_requests import RunRequestStore, build_run_request
from lab_control.runner_protocol import RunnerCoordinator
from lab_control.runner_staging import ExactSnapshotStager
from lab_control.runs import control_plane_identity
from lab_control.workspace import LabError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Result Transfer Test")
    _git(path, "config", "user.email", "result-transfer@example.invalid")
    _git(
        path,
        "remote",
        "add",
        "origin",
        "https://example.invalid/example/Controller.git",
    )
    (path / "source.txt").write_text("stable\n", encoding="utf-8")
    _git(path, "add", "source.txt")
    _git(path, "commit", "-m", "initial")
    return path.resolve()


def _profiles(tmp_path: Path, *, command: list[str]) -> tuple[Path, Path, Path]:
    queue_root = tmp_path / "queue-profiles"
    runner_root = tmp_path / "runner-profiles"
    command_root = tmp_path / "runner-commands"
    queue_root.mkdir()
    runner_root.mkdir()
    command_root.mkdir()
    (queue_root / "contract-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v1",
                "profile_id": "contract-check",
                "description": "Generic result transfer fixture",
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
    (runner_root / "runner-a.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-profile.v1",
                "runner_id": "runner-a",
                "description": "Generic result transfer runner",
                "target_ids": ["test-target"],
                "slots": ["slot-a1"],
                "capabilities": ["system-command"],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )
    (command_root / "runner-a.json").write_text(
        json.dumps(
            {
                "schema": "lab.runner-command-map.v1",
                "runner_id": "runner-a",
                "commands": [
                    {
                        "command_id": "contract-suite",
                        "repository": "Controller",
                        "argv": command,
                        "timeout_seconds": 30,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return queue_root, runner_root, command_root


def _execute(
    tmp_path: Path,
    *,
    command: list[str],
) -> tuple[RunRequestStore, RunRequestStore, Path, str]:
    repository = _repository(tmp_path / "repository")
    queue_root, runner_root, command_root = _profiles(tmp_path, command=command)
    commit = _git(repository, "rev-parse", "HEAD")
    control_root = Path(__file__).resolve().parents[1]
    control_identity = control_plane_identity(control_root)
    request = build_run_request(
        task_id="T602",
        snapshot_id="snapshot-result-1",
        snapshot_identity={
            "repositories": {
                "Controller": {
                    "commit": commit,
                    "tree_oid": _git(repository, "rev-parse", f"{commit}^{{tree}}"),
                    "submodules": {},
                }
            },
            "repository_links": [],
        },
        profile=load_queue_profile(queue_root, "contract-check"),
        evidence_tier="Development",
        control_plane=control_identity,
    )
    origin_store = RunRequestStore(tmp_path / "origin-state")
    remote_store = RunRequestStore(tmp_path / "remote-state")
    origin_store.submit(request)
    remote_store.submit(request)
    coordinator = RunnerCoordinator(
        remote_store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_identity,
    )
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    slot_root = tmp_path / "slots"
    stager = ExactSnapshotStager(
        coordinator,
        repository_catalog={"Controller": repository},
        slot_root=slot_root,
    )
    stager.stage(str(request["request_id"]), "runner-a")
    TrustedCommandExecutor(
        coordinator, stager, command_root=command_root
    ).execute(str(request["request_id"]), "runner-a")
    return origin_store, remote_store, slot_root, str(request["request_id"])


def test_result_package_returns_terminal_evidence_without_moving_origin_state(
    tmp_path: Path,
) -> None:
    origin_store, remote_store, slot_root, request_id = _execute(
        tmp_path,
        command=[sys.executable, "-c", "print('generic result')"],
    )
    secret = b"result-package-test-key-material" * 2
    package_root = tmp_path / "result-package"
    manifest = ResultPackageBuilder(remote_store, slot_root=slot_root).build(
        request_id,
        package_root,
        sender_id="runner-node-b",
        recipient_id="controller-a",
        key_id="transport-key-1",
        secret=secret,
    )
    ingress = ResultPackageIngress(
        origin_store,
        result_root=tmp_path / "imported-results",
        recipient_id="controller-a",
        trusted_sender_keys={"runner-node-b": {"transport-key-1": secret}},
    )

    result = ingress.ingest(package_root)
    repeated = ingress.ingest(package_root)

    assert result["created"] is True
    assert repeated["created"] is False
    assert repeated["receipt"] == result["receipt"]
    assert origin_store.load(request_id)["state"] == "SUBMITTED"
    receipt = result["receipt"]
    assert receipt["remote_state"] == "COMPLETED"
    assert receipt["classification"] == "PASSED"
    imported = Path(str(receipt["import_path"]))
    assert imported.is_dir()
    artifact = manifest["artifacts"][0]
    imported_log = imported / str(artifact["package_path"])
    assert "generic result" in imported_log.read_text(encoding="utf-8")

    with imported_log.open("ab") as stream:
        stream.write(b"changed after import")
    with pytest.raises(LabError, match="imported result artifact changed"):
        ingress.ingest(package_root)


def test_result_package_rejects_tampered_artifact(tmp_path: Path) -> None:
    origin_store, remote_store, slot_root, request_id = _execute(
        tmp_path,
        command=[sys.executable, "-c", "print('original')"],
    )
    secret = b"result-package-test-key-material" * 2
    package_root = tmp_path / "result-package"
    manifest = ResultPackageBuilder(remote_store, slot_root=slot_root).build(
        request_id,
        package_root,
        sender_id="runner-node-b",
        recipient_id="controller-a",
        key_id="transport-key-1",
        secret=secret,
    )
    artifact = package_root / str(manifest["artifacts"][0]["package_path"])
    with artifact.open("ab") as stream:
        stream.write(b"tampered")

    with pytest.raises(LabError, match="artifact hash mismatch"):
        ResultPackageIngress(
            origin_store,
            result_root=tmp_path / "imported-results",
            recipient_id="controller-a",
            trusted_sender_keys={"runner-node-b": {"transport-key-1": secret}},
        ).ingest(package_root)

    assert not (tmp_path / "imported-results").exists()


def test_result_package_enforces_small_artifact_boundary(tmp_path: Path) -> None:
    _, remote_store, slot_root, request_id = _execute(
        tmp_path,
        command=[sys.executable, "-c", "print('more than four bytes')"],
    )

    with pytest.raises(LabError, match="artifact exceeds 4 bytes"):
        ResultPackageBuilder(
            remote_store,
            slot_root=slot_root,
            maximum_artifact_bytes=4,
        ).build(
            request_id,
            tmp_path / "result-package",
            sender_id="runner-node-b",
            recipient_id="controller-a",
            key_id="transport-key-1",
            secret=b"result-package-test-key-material" * 2,
        )


def test_result_package_rejects_wrong_peer_and_request_content(tmp_path: Path) -> None:
    origin_store, remote_store, slot_root, request_id = _execute(
        tmp_path,
        command=[sys.executable, "-c", "print('generic result')"],
    )
    secret = b"result-package-test-key-material" * 2
    package_root = tmp_path / "result-package"
    ResultPackageBuilder(remote_store, slot_root=slot_root).build(
        request_id,
        package_root,
        sender_id="runner-node-b",
        recipient_id="controller-a",
        key_id="transport-key-1",
        secret=secret,
    )

    with pytest.raises(LabError, match="identity mismatch"):
        ResultPackageIngress(
            origin_store,
            result_root=tmp_path / "wrong-peer-results",
            recipient_id="controller-c",
            trusted_sender_keys={"runner-node-b": {"transport-key-1": secret}},
        ).ingest(package_root)
    with pytest.raises(LabError, match="authentication failed"):
        ResultPackageIngress(
            origin_store,
            result_root=tmp_path / "wrong-key-results",
            recipient_id="controller-a",
            trusted_sender_keys={"runner-node-b": {"transport-key-1": b"x" * 32}},
        ).ingest(package_root)

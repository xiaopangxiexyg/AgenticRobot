from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from lab_control import git as lab_git
from lab_control.queue_profiles import load_queue_profile
from lab_control.run_requests import RunRequestStore, build_run_request
from lab_control.runner_protocol import RunnerCoordinator
from lab_control.runner_staging import ExactSnapshotStager
from lab_control.runs import control_plane_identity
from lab_control.source_transfer import (
    SourcePackageBuilder,
    SourcePackageIngress,
)
from lab_control.workspace import LabError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _source_pair(tmp_path: Path) -> tuple[Path, Path, str]:
    seed = tmp_path / "seed"
    seed.mkdir()
    _git(seed, "init", "-b", "main")
    _git(seed, "config", "user.name", "Source Transfer Test")
    _git(seed, "config", "user.email", "source-transfer@example.invalid")
    (seed / "source.txt").write_text("base\n", encoding="utf-8")
    _git(seed, "add", "source.txt")
    _git(seed, "commit", "-m", "base")
    sender = tmp_path / "sender"
    receiver = tmp_path / "receiver"
    subprocess.run(["git", "clone", str(seed), str(sender)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(seed), str(receiver)], check=True, capture_output=True)
    for repository in (sender, receiver):
        _git(repository, "config", "user.name", "Source Transfer Test")
        _git(repository, "config", "user.email", "source-transfer@example.invalid")
        _git(
            repository,
            "remote",
            "set-url",
            "origin",
            "https://example.invalid/example/Controller.git",
        )
    return sender.resolve(), receiver.resolve(), _git(receiver, "rev-parse", "HEAD")


def _profiles(tmp_path: Path) -> tuple[Path, Path]:
    queue_root = tmp_path / "queue-profiles"
    runner_root = tmp_path / "runner-profiles"
    queue_root.mkdir()
    runner_root.mkdir()
    (queue_root / "contract-check.json").write_text(
        json.dumps(
            {
                "schema": "lab.queue-profile.v1",
                "profile_id": "contract-check",
                "description": "Generic source transfer fixture",
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
                "description": "Generic source transfer runner",
                "target_ids": ["test-target"],
                "slots": ["slot-a1"],
                "capabilities": ["system-command"],
                "accepted_risk_levels": ["R0"],
                "accepted_evidence_tiers": ["Development"],
            }
        ),
        encoding="utf-8",
    )
    return queue_root, runner_root


def _request(sender: Path, queue_root: Path) -> dict[str, object]:
    commit = _git(sender, "rev-parse", "HEAD")
    return build_run_request(
        task_id="T601",
        snapshot_id="snapshot-source-1",
        snapshot_identity={
            "repositories": {
                "Controller": {
                    "commit": commit,
                    "tree_oid": _git(sender, "rev-parse", f"{commit}^{{tree}}"),
                    "submodules": {},
                }
            },
            "repository_links": [],
        },
        profile=load_queue_profile(queue_root, "contract-check"),
        evidence_tier="Development",
        control_plane=control_plane_identity(Path(__file__).resolve().parents[1]),
    )


def test_source_package_transfers_exact_objects_and_enables_staging(
    tmp_path: Path,
) -> None:
    sender, receiver, receiver_base = _source_pair(tmp_path)
    queue_root, runner_root = _profiles(tmp_path)
    (sender / "source.txt").write_text("candidate\n", encoding="utf-8")
    _git(sender, "commit", "-am", "candidate")
    request = _request(sender, queue_root)
    target = request["snapshot"]["repositories"]["Controller"]["commit"]
    assert lab_git.run_git(receiver, ["cat-file", "-e", target], check=False).returncode != 0
    secret = b"source-package-test-key-material" * 2
    package_root = tmp_path / "source-package"

    manifest = SourcePackageBuilder(
        repository_catalog={"Controller": sender}
    ).build(
        request,
        package_root,
        base_commits={"Controller": receiver_base},
        sender_id="controller-a",
        recipient_id="runner-node-b",
        key_id="transport-key-1",
        secret=secret,
    )

    store = RunRequestStore(tmp_path / "remote-state")
    ingress = SourcePackageIngress(
        store,
        repository_catalog={"Controller": receiver},
        recipient_id="runner-node-b",
        trusted_sender_keys={"controller-a": {"transport-key-1": secret}},
    )
    result = ingress.ingest(package_root)
    repeated = ingress.ingest(package_root)

    assert manifest["repositories"]["Controller"]["transfer_mode"] == "pack"
    assert result["created"] is True
    assert repeated["created"] is False
    assert repeated["receipt"] == result["receipt"]
    assert lab_git.resolve_commit(receiver, target) == target
    assert lab_git.tree_oid(receiver, target) == request["snapshot"]["repositories"][
        "Controller"
    ]["tree_oid"]

    coordinator = RunnerCoordinator(
        store,
        runner_profile_root=runner_root,
        queue_profile_root=queue_root,
        control_plane=control_plane_identity(Path(__file__).resolve().parents[1]),
    )
    assert coordinator.claim_next("runner-a", "slot-a1") is not None
    receipt = ExactSnapshotStager(
        coordinator,
        repository_catalog={"Controller": receiver},
        slot_root=tmp_path / "slots",
    ).stage(str(request["request_id"]), "runner-a")
    staged = (
        tmp_path
        / "slots"
        / "slot-a1"
        / str(request["request_id"])
        / str(receipt["repositories"]["Controller"]["relative_path"])
    )
    assert (staged / "source.txt").read_text(encoding="utf-8") == "candidate\n"
    assert lab_git.current_commit(staged) == target


def test_source_package_rejects_tampered_pack_before_request_ingress(
    tmp_path: Path,
) -> None:
    sender, receiver, receiver_base = _source_pair(tmp_path)
    queue_root, _ = _profiles(tmp_path)
    (sender / "source.txt").write_text("candidate\n", encoding="utf-8")
    _git(sender, "commit", "-am", "candidate")
    request = _request(sender, queue_root)
    secret = b"source-package-test-key-material" * 2
    package_root = tmp_path / "source-package"
    manifest = SourcePackageBuilder(
        repository_catalog={"Controller": sender}
    ).build(
        request,
        package_root,
        base_commits={"Controller": receiver_base},
        sender_id="controller-a",
        recipient_id="runner-node-b",
        key_id="transport-key-1",
        secret=secret,
    )
    pack = package_root / str(manifest["repositories"]["Controller"]["pack_path"])
    with pack.open("ab") as stream:
        stream.write(b"tampered")
    store = RunRequestStore(tmp_path / "remote-state")

    with pytest.raises(LabError, match="pack hash mismatch"):
        SourcePackageIngress(
            store,
            repository_catalog={"Controller": receiver},
            recipient_id="runner-node-b",
            trusted_sender_keys={"controller-a": {"transport-key-1": secret}},
        ).ingest(package_root)

    assert not (store.requests_root / str(request["request_id"])).exists()


def test_source_package_rejects_large_blob_and_wrong_receiver_base(
    tmp_path: Path,
) -> None:
    sender, _, receiver_base = _source_pair(tmp_path)
    queue_root, _ = _profiles(tmp_path)
    (sender / "large.bin").write_bytes(b"x" * 32)
    _git(sender, "add", "large.bin")
    _git(sender, "commit", "-m", "large source")
    request = _request(sender, queue_root)
    builder = SourcePackageBuilder(
        repository_catalog={"Controller": sender}, maximum_blob_bytes=16
    )

    with pytest.raises(LabError, match="blob exceeds 16 bytes"):
        builder.build(
            request,
            tmp_path / "large-package",
            base_commits={"Controller": receiver_base},
            sender_id="controller-a",
            recipient_id="runner-node-b",
            key_id="transport-key-1",
            secret=b"source-package-test-key-material" * 2,
        )
    with pytest.raises(LabError, match="receiver base"):
        builder.build(
            request,
            tmp_path / "missing-base-package",
            base_commits={},
            sender_id="controller-a",
            recipient_id="runner-node-b",
            key_id="transport-key-1",
            secret=b"source-package-test-key-material" * 2,
        )


def test_source_package_present_mode_transfers_no_pack(tmp_path: Path) -> None:
    sender, receiver, receiver_base = _source_pair(tmp_path)
    queue_root, _ = _profiles(tmp_path)
    request = _request(sender, queue_root)
    secret = b"source-package-test-key-material" * 2
    package_root = tmp_path / "present-package"

    manifest = SourcePackageBuilder(
        repository_catalog={"Controller": sender}
    ).build(
        request,
        package_root,
        base_commits={"Controller": receiver_base},
        sender_id="controller-a",
        recipient_id="runner-node-b",
        key_id="transport-key-1",
        secret=secret,
    )
    result = SourcePackageIngress(
        RunRequestStore(tmp_path / "remote-state"),
        repository_catalog={"Controller": receiver},
        recipient_id="runner-node-b",
        trusted_sender_keys={"controller-a": {"transport-key-1": secret}},
    ).ingest(package_root)

    record = manifest["repositories"]["Controller"]
    assert record["transfer_mode"] == "present"
    assert record["pack_path"] is None
    assert result["created"] is True

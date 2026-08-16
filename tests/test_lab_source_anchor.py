from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lab_control import git
from lab_control.preparation_profile import (
    ControlPlanePlacement,
    PreparationProfile,
    SourceAnchorPlacement,
)
from lab_control.source_anchor import materialize_local_source_anchor
from lab_control.trusted_preparation import prepare_source_anchor
from lab_control.workspace import LabError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, *, payload: bytes = b"source\n") -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Source Anchor Test")
    _git(path, "config", "user.email", "source-anchor@example.invalid")
    (path / "source.bin").write_bytes(payload)
    _git(path, "add", "source.bin")
    _git(path, "commit", "-m", "initial")
    _git(path, "remote", "add", "origin", "git@example.invalid:TestOrg/Anchor.git")
    return path.resolve()


def test_materializes_clean_detached_anchor_without_network_or_lfs(
    tmp_path: Path,
) -> None:
    source = _repository(tmp_path / "source")
    target = (tmp_path / "anchor").resolve()
    commit = git.current_commit(source)

    receipt = materialize_local_source_anchor(
        source_repository=source,
        target=target,
        expected_identity="TestOrg/Anchor",
        commit=commit,
    )

    assert receipt["commit"] == commit
    assert receipt["tree_oid"] == git.tree_oid(source, commit)
    assert receipt["network_used"] is False
    assert receipt["lfs_smudge_enabled"] is False
    assert receipt["large_file_downloaded"] is False
    assert receipt["detached"] is True
    assert git.origin_identity(target) == "TestOrg/Anchor"
    assert git.status_short(target) == ()
    assert len(receipt["receipt_sha256"]) == 64


def test_rejects_dirty_source_and_large_tracked_blob(tmp_path: Path) -> None:
    dirty = _repository(tmp_path / "dirty")
    (dirty / "untracked.py").write_text("unsafe\n", encoding="utf-8")
    with pytest.raises(LabError, match="input is dirty"):
        materialize_local_source_anchor(
            source_repository=dirty,
            target=(tmp_path / "dirty-anchor").resolve(),
            expected_identity="TestOrg/Anchor",
            commit=git.current_commit(dirty),
        )

    large = _repository(tmp_path / "large", payload=b"x" * 33)
    with pytest.raises(LabError, match="large tracked blob"):
        materialize_local_source_anchor(
            source_repository=large,
            target=(tmp_path / "large-anchor").resolve(),
            expected_identity="TestOrg/Anchor",
            commit=git.current_commit(large),
            max_checkout_blob_bytes=32,
        )
    assert not (tmp_path / "large-anchor").exists()


def test_materializes_exact_submodule_from_existing_local_objects(
    tmp_path: Path,
) -> None:
    submodule = _repository(tmp_path / "component")
    source = _repository(tmp_path / "source")
    subprocess.run(
        [
            "git",
            "-c",
            "protocol.file.allow=always",
            "-C",
            str(source),
            "submodule",
            "add",
            str(submodule),
            "third_party/component",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(source, "commit", "-am", "add component")
    target = (tmp_path / "anchor").resolve()

    receipt = materialize_local_source_anchor(
        source_repository=source,
        target=target,
        expected_identity="TestOrg/Anchor",
        commit=git.current_commit(source),
        submodule_sources={"third_party/component": submodule},
    )

    assert receipt["submodules"] == {
        "third_party/component": git.current_commit(submodule)
    }
    observed, initialized = git.submodule_status(target)
    assert observed == receipt["submodules"]
    assert initialized is True
    assert git.status_short(target) == ()


def test_protected_profile_resolves_source_and_target_paths(tmp_path: Path) -> None:
    source = _repository(tmp_path / "source")
    target = (tmp_path / "protected-anchor").resolve()
    profile = PreparationProfile(
        profile_id="test-preparation-v1",
        host_id="test-host",
        stable_sync={},
        source_anchors={
            "openpi": SourceAnchorPlacement(
                repository="openpi",
                canonical_identity="TestOrg/Anchor",
                source_path=source,
                target_path=target,
                submodule_sources={},
                max_checkout_blob_bytes=16 * 1024 * 1024,
            )
        },
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=tmp_path / "control-plane",
            current_link=tmp_path / "control-plane" / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(tmp_path,),
        path=tmp_path / "profile.json",
        sha256="b" * 64,
    )

    protected = prepare_source_anchor(
        profile,
        repository="openpi",
        commit=git.current_commit(source),
    )

    assert protected["source_anchor_receipt"]["target"] == str(target)
    assert protected["network_used"] is False
    assert protected["task_payload_authorized"] is False
    assert protected["device_access"] is False
    assert len(protected["receipt_sha256"]) == 64

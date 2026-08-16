from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from lab_control import git
from lab_control.trusted_fetch import (
    fetch_stable_remote_ref,
    validate_trusted_fetch_receipt,
)
from lab_control.workspace import LabError


def _run(*arguments: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    remote = (tmp_path / "remotes" / "TestOrg" / "Stable.git").resolve()
    remote.parent.mkdir(parents=True)
    _run("git", "init", "--bare", str(remote))
    seed = (tmp_path / "seed").resolve()
    _run("git", "clone", str(remote), str(seed))
    _run("git", "switch", "-c", "main", cwd=seed)
    _run("git", "config", "user.name", "Trusted Fetch Test", cwd=seed)
    _run("git", "config", "user.email", "fetch@example.invalid", cwd=seed)
    (seed / "state.txt").write_text("base\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=seed)
    _run("git", "commit", "-m", "base", cwd=seed)
    _run("git", "push", "-u", "origin", "main", cwd=seed)

    stable = (tmp_path / "stable").resolve()
    _run("git", "clone", "--branch", "main", str(remote), str(stable))
    _run(
        "git",
        "remote",
        "set-url",
        "origin",
        "git@example.invalid:TestOrg/Stable.git",
        cwd=stable,
    )
    return seed, stable, remote


def _local_fetch_runner(remote: Path):
    def run(
        argv: list[str],
        environment: Mapping[str, str],
        timeout_seconds: int,
    ) -> subprocess.CompletedProcess[str]:
        assert "--no-tags" in argv
        assert "--no-recurse-submodules" in argv
        assert "--no-write-fetch-head" in argv
        assert environment["GIT_LFS_SKIP_SMUDGE"] == "1"
        assert environment["GIT_TERMINAL_PROMPT"] == "0"
        local_argv = list(argv)
        local_argv[-2] = str(remote)
        return subprocess.run(
            local_argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(environment),
        )

    return run


def test_trusted_fetch_moves_only_the_remote_ref_and_records_receipt(
    tmp_path: Path,
) -> None:
    seed, stable, remote = _fixture(tmp_path)
    local_before = git.current_commit(stable)
    tree_before = git.tree_oid(stable)
    (seed / "state.txt").write_text("next\n", encoding="utf-8")
    _run("git", "commit", "-am", "next", cwd=seed)
    _run("git", "push", "origin", "main", cwd=seed)
    incoming = git.current_commit(seed)

    receipt = fetch_stable_remote_ref(
        stable,
        expected_identity="TestOrg/Stable",
        fetch_runner=_local_fetch_runner(remote),
    )

    assert validate_trusted_fetch_receipt(receipt) == receipt
    assert git.current_commit(stable) == local_before
    assert git.tree_oid(stable) == tree_before
    assert git.status_short(stable) == ()
    assert receipt["remote_commit_after"] == incoming
    assert receipt["checkout_updated"] is False
    assert receipt["submodules_fetched"] is False
    assert receipt["lfs_smudge_enabled"] is False
    assert receipt["large_assets_downloaded"] is False
    assert receipt["task_payload_authorized"] is False


def test_trusted_fetch_refuses_dirty_wrong_branch_and_wrong_identity(
    tmp_path: Path,
) -> None:
    _, stable, remote = _fixture(tmp_path)
    runner = _local_fetch_runner(remote)
    (stable / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(LabError, match="dirty stable checkout"):
        fetch_stable_remote_ref(
            stable,
            expected_identity="TestOrg/Stable",
            fetch_runner=runner,
        )
    (stable / "dirty.txt").unlink()

    with pytest.raises(LabError, match="canonical repository identity mismatch"):
        fetch_stable_remote_ref(
            stable,
            expected_identity="OtherOrg/Stable",
            fetch_runner=runner,
        )
    _run("git", "switch", "-c", "task/T301", cwd=stable)
    with pytest.raises(LabError, match="wrong branch"):
        fetch_stable_remote_ref(
            stable,
            expected_identity="TestOrg/Stable",
            fetch_runner=runner,
        )


def test_trusted_fetch_receipt_is_content_addressed_and_fail_closed(
    tmp_path: Path,
) -> None:
    _, stable, remote = _fixture(tmp_path)
    receipt = fetch_stable_remote_ref(
        stable,
        expected_identity="TestOrg/Stable",
        fetch_runner=_local_fetch_runner(remote),
    )
    tampered = dict(receipt)
    tampered["checkout_updated"] = True

    with pytest.raises(LabError, match="unsafe authority envelope"):
        validate_trusted_fetch_receipt(tampered)


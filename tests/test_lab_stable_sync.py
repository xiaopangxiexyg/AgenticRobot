from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lab_control import git
from lab_control.stable_sync import (
    apply_stable_checkout_fast_forward,
    plan_stable_checkout_sync,
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


def _repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "-b", "main")
    _git(path, "config", "user.name", "Stable Sync Test")
    _git(path, "config", "user.email", "stable-sync@example.invalid")
    (path / "state.txt").write_text("base\n", encoding="utf-8")
    _git(path, "add", "state.txt")
    _git(path, "commit", "-m", "base")
    _git(path, "remote", "add", "origin", "git@example.invalid:TestOrg/Stable.git")
    return path.resolve()


def _stage_fetched_fast_forward(repository: Path) -> tuple[str, str]:
    before = git.current_commit(repository)
    _git(repository, "switch", "-c", "incoming")
    (repository / "state.txt").write_text("incoming\n", encoding="utf-8")
    _git(repository, "commit", "-am", "incoming")
    incoming = git.current_commit(repository)
    _git(repository, "switch", "main")
    _git(repository, "update-ref", "refs/remotes/origin/main", incoming)
    return before, incoming


def test_plans_and_applies_only_unchanged_fast_forward(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "stable")
    before, incoming = _stage_fetched_fast_forward(repository)

    plan = plan_stable_checkout_sync(
        repository,
        expected_identity="TestOrg/Stable",
    )
    assert plan["action"] == "FAST_FORWARD_READY"
    assert plan["local_commit"] == before
    assert plan["remote_commit"] == incoming
    assert plan["network_used"] is False
    assert plan["worktree_files_copied"] is False

    receipt = apply_stable_checkout_fast_forward(plan)

    assert receipt["before_commit"] == before
    assert receipt["after_commit"] == incoming
    assert receipt["fast_forwarded"] is True
    assert len(receipt["receipt_sha256"]) == 64
    assert git.current_branch(repository) == "main"
    assert git.status_short(repository) == ()
    assert (
        plan_stable_checkout_sync(
            repository,
            expected_identity="TestOrg/Stable",
        )["action"]
        == "NOOP"
    )


def test_refuses_dirty_ahead_and_changed_plans(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "stable")
    _stage_fetched_fast_forward(repository)
    plan = plan_stable_checkout_sync(
        repository,
        expected_identity="TestOrg/Stable",
    )
    (repository / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(LabError, match="state changed"):
        apply_stable_checkout_fast_forward(plan)
    assert git.current_branch(repository) == "main"

    _git(repository, "add", "dirty.txt")
    _git(repository, "commit", "-m", "local-only")
    ahead = plan_stable_checkout_sync(
        repository,
        expected_identity="TestOrg/Stable",
    )
    assert ahead["action"] == "BLOCKED_DIVERGED"
    with pytest.raises(LabError, match="not applicable"):
        apply_stable_checkout_fast_forward(ahead)

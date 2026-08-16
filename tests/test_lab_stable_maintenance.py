from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from lab_control import git
from lab_control.maintenance_gate import (
    StableCheckoutTarget,
    acquire_maintenance_reservation,
    assert_managed_runtime_start_allowed,
    release_maintenance_reservation,
)
from lab_control.runtime_inventory import collect_checkout_runtime_inventory
from lab_control.stable_maintenance import guarded_stable_checkout_fast_forward
from lab_control.stable_sync import plan_stable_checkout_sync
from lab_control.workspace import LabError


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _fixture(tmp_path: Path) -> tuple[StableCheckoutTarget, dict[str, object], Path]:
    repository = (tmp_path / "stable").resolve()
    repository.mkdir()
    _git(repository, "init", "-b", "main")
    _git(repository, "config", "user.name", "Maintenance Test")
    _git(repository, "config", "user.email", "maintenance@example.invalid")
    (repository / "state.txt").write_text("base\n", encoding="utf-8")
    _git(repository, "add", "state.txt")
    _git(repository, "commit", "-m", "base")
    _git(repository, "remote", "add", "origin", "git@example.invalid:Org/Stable.git")
    _git(repository, "switch", "-c", "incoming")
    (repository / "state.txt").write_text("incoming\n", encoding="utf-8")
    _git(repository, "commit", "-am", "incoming")
    incoming = git.current_commit(repository)
    _git(repository, "switch", "main")
    _git(repository, "update-ref", "refs/remotes/origin/main", incoming)
    target = StableCheckoutTarget(
        project_id="generic-project",
        host_id="test-host",
        repository="component",
        checkout_path=repository,
        canonical_identity="Org/Stable",
        branch="main",
        remote_ref="refs/remotes/origin/main",
        state_root=(tmp_path / "state").resolve(),
    )
    plan = plan_stable_checkout_sync(
        repository,
        expected_identity="Org/Stable",
    )
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()
    return target, plan, proc


def _collector(proc: Path):
    def collect(checkout: Path) -> dict[str, object]:
        return collect_checkout_runtime_inventory(
            checkout,
            proc_root=proc,
            scanner_euid=os.geteuid(),
        )

    return collect


def _dependent_process(proc: Path, checkout: Path) -> None:
    process = proc / "101"
    process.mkdir()
    uid = os.geteuid()
    (process / "status").write_text(
        f"Name:\ttest\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="utf-8",
    )
    (process / "cwd").symlink_to(checkout)
    (process / "exe").symlink_to("/usr/bin/python3")
    (process / "cmdline").write_bytes(b"python\0worker.py\0")
    (process / "environ").write_bytes(b"PATH=/usr/bin\0")
    (process / "maps").write_bytes(b"")


def test_guarded_fast_forward_records_idle_evidence_and_releases_gate(
    tmp_path: Path,
) -> None:
    target, plan, proc = _fixture(tmp_path)
    before = plan["local_commit"]
    incoming = plan["remote_commit"]

    receipt = guarded_stable_checkout_fast_forward(
        target,
        plan=plan,
        reservation_id="maint-001",
        inventory_collector=_collector(proc),
    )

    assert receipt["before_commit"] == before
    assert receipt["after_commit"] == incoming
    assert receipt["network_used"] is False
    assert receipt["task_worktrees_updated"] is False
    assert receipt["submodules_updated"] is False
    assert git.current_commit(target.checkout_path) == incoming
    assert git.status_short(target.checkout_path) == ()
    assert receipt["released_reservation"]["status"] == "RELEASED"
    assert assert_managed_runtime_start_allowed(target)["allowed"] is True


def test_runtime_gate_blocks_during_reservation_and_reopens_after_release(
    tmp_path: Path,
) -> None:
    target, plan, _ = _fixture(tmp_path)
    reservation = acquire_maintenance_reservation(
        target,
        plan=plan,
        reservation_id="maint-002",
    )
    with pytest.raises(LabError, match="blocked by maintenance reservation"):
        assert_managed_runtime_start_allowed(target)
    released = release_maintenance_reservation(
        target,
        active_receipt=reservation,
    )
    assert released["status"] == "RELEASED"
    assert assert_managed_runtime_start_allowed(target)["allowed"] is True


def test_guarded_fast_forward_refuses_active_process_and_still_releases_gate(
    tmp_path: Path,
) -> None:
    target, plan, proc = _fixture(tmp_path)
    _dependent_process(proc, target.checkout_path)

    with pytest.raises(LabError, match="active dependent processes"):
        guarded_stable_checkout_fast_forward(
            target,
            plan=plan,
            reservation_id="maint-003",
            inventory_collector=_collector(proc),
        )

    assert git.current_commit(target.checkout_path) == plan["local_commit"]
    assert assert_managed_runtime_start_allowed(target)["allowed"] is True
    state_path = (
        target.state_root
        / "maintenance-reservations"
        / target.project_id
        / f"{target.repository}.json"
    )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["status"] == "RELEASED"


def test_reservation_refuses_tampered_or_changed_plan(tmp_path: Path) -> None:
    target, plan, _ = _fixture(tmp_path)
    tampered = dict(plan)
    tampered["remote_commit"] = "f" * 40
    with pytest.raises(LabError, match="hash mismatch"):
        acquire_maintenance_reservation(
            target,
            plan=tampered,
            reservation_id="maint-004",
        )

    (target.checkout_path / "dirty.txt").write_text("preserve\n", encoding="utf-8")
    with pytest.raises(LabError, match="state changed"):
        acquire_maintenance_reservation(
            target,
            plan=plan,
            reservation_id="maint-005",
        )
    assert (target.checkout_path / "dirty.txt").read_text(encoding="utf-8") == (
        "preserve\n"
    )

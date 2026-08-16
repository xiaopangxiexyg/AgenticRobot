from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

from lab_control import git
from lab_control.maintenance import build_maintenance_receipt
from lab_control.maintenance_reservation import (
    acquire_maintenance_reservation,
    assert_managed_runtime_start_allowed,
)
from lab_control.preparation_profile import (
    ControlPlanePlacement,
    PreparationProfile,
    StableSyncPlacement,
)
from lab_control.runtime_inventory import collect_checkout_runtime_inventory
from lab_control.stable_sync import (
    apply_stable_checkout_fast_forward,
    plan_stable_checkout_sync,
)
from lab_control.trusted_fetch import fetch_stable_remote_ref
from lab_control.trusted_preparation import (
    perform_guarded_stable_sync,
    synchronize_stable_checkout,
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
    _run(
        "git",
        "config",
        "user.email",
        "trusted-fetch@example.invalid",
        cwd=seed,
    )
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
        local_argv = list(argv)
        assert local_argv[-2] == "origin"
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


def test_explicit_fetch_then_guarded_fast_forward(tmp_path: Path) -> None:
    seed, stable, remote = _fixture(tmp_path)
    local_before = git.current_commit(stable)
    (seed / "state.txt").write_text("next\n", encoding="utf-8")
    _run("git", "commit", "-am", "next", cwd=seed)
    _run("git", "push", "origin", "main", cwd=seed)
    remote_commit = git.current_commit(seed)

    fetch = fetch_stable_remote_ref(
        stable,
        expected_identity="TestOrg/Stable",
        fetch_runner=_local_fetch_runner(remote),
    )

    assert fetch["local_commit_unchanged"] == local_before
    assert fetch["remote_commit_after"] == remote_commit
    assert fetch["network_used"] is True
    assert fetch["checkout_updated"] is False
    assert fetch["submodules_fetched"] is False
    assert fetch["lfs_smudge_enabled"] is False
    assert fetch["large_assets_downloaded"] is False
    assert fetch["task_payload_authorized"] is False
    plan = plan_stable_checkout_sync(
        stable,
        expected_identity="TestOrg/Stable",
    )
    assert plan["action"] == "FAST_FORWARD_READY"
    applied = apply_stable_checkout_fast_forward(plan)
    assert applied["after_commit"] == remote_commit


def test_fetch_refuses_dirty_or_wrong_identity(tmp_path: Path) -> None:
    _, stable, remote = _fixture(tmp_path)
    (stable / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(LabError, match="dirty stable checkout"):
        fetch_stable_remote_ref(
            stable,
            expected_identity="TestOrg/Stable",
            fetch_runner=_local_fetch_runner(remote),
        )
    assert not git.ref_exists(stable, "refs/remotes/origin/not-main")

    (stable / "dirty.txt").unlink()
    with pytest.raises(LabError, match="identity mismatch"):
        fetch_stable_remote_ref(
            stable,
            expected_identity="SomeOtherOrg/Stable",
            fetch_runner=_local_fetch_runner(remote),
        )


def test_protected_profile_runs_complete_stable_sync_cycle(tmp_path: Path) -> None:
    seed, stable, remote = _fixture(tmp_path)
    (seed / "state.txt").write_text("cycle\n", encoding="utf-8")
    _run("git", "commit", "-am", "cycle", cwd=seed)
    _run("git", "push", "origin", "main", cwd=seed)
    profile = PreparationProfile(
        profile_id="test-preparation-v1",
        host_id="test-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/Stable",
                path=stable,
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=tmp_path / "control-plane",
            current_link=tmp_path / "control-plane" / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(tmp_path,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )
    profile.control_plane.install_root.mkdir()

    waiting = synchronize_stable_checkout(
        profile,
        repository="AgenticRobot",
        fetch_runner=_local_fetch_runner(remote),
    )

    assert waiting["status"] == "WAITING_FOR_MAINTENANCE"
    assert waiting["sync_plan"]["action"] == "FAST_FORWARD_READY"
    assert waiting["apply_receipt"] is None
    assert git.current_commit(stable) != git.current_commit(seed)

    proc = (tmp_path / "proc").resolve()
    proc.mkdir()
    inventory = collect_checkout_runtime_inventory(
        stable,
        proc_root=proc,
        scanner_euid=os.geteuid(),
    )
    reservation = acquire_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        reservation_id="maint-sync-001",
    )
    maintenance = build_maintenance_receipt(
        profile,
        repository="AgenticRobot",
        runtime_inventory=inventory,
        maintenance_reservation=reservation,
    )
    receipt = synchronize_stable_checkout(
        profile,
        repository="AgenticRobot",
        fetch_runner=_local_fetch_runner(remote),
        maintenance_receipt=maintenance,
    )

    assert receipt["status"] == "SYNCED"
    assert receipt["fetch_receipt"]["task_payload_authorized"] is False
    assert receipt["sync_plan"]["action"] == "FAST_FORWARD_READY"
    assert receipt["maintenance_receipt"] == maintenance
    assert receipt["maintenance_release_receipt"]["status"] == "RELEASED"
    assert receipt["apply_receipt"]["after_commit"] == git.current_commit(seed)
    assert receipt["task_payload_authorized"] is False
    assert receipt["device_access"] is False
    assert len(receipt["receipt_sha256"]) == 64

    (seed / "state.txt").write_text("guarded auto sync\n", encoding="utf-8")
    _run("git", "commit", "-am", "guarded auto sync", cwd=seed)
    _run("git", "push", "origin", "main", cwd=seed)
    auto_proc = (tmp_path / "auto-proc").resolve()
    auto_proc.mkdir()
    guarded = perform_guarded_stable_sync(
        profile,
        repository="AgenticRobot",
        reservation_id="maint-sync-002",
        proc_root=auto_proc,
        scanner_euid=os.geteuid(),
        fetch_runner=_local_fetch_runner(remote),
    )
    assert guarded["status"] == "SYNCED"
    assert guarded["maintenance_released"] is True
    assert guarded["task_payload_authorized"] is False
    assert guarded["device_access"] is False
    assert git.current_commit(stable) == git.current_commit(seed)

    blocked_proc = (tmp_path / "blocked-proc").resolve()
    process = blocked_proc / "123"
    process.mkdir(parents=True)
    uid = os.geteuid()
    (process / "status").write_text(
        f"Name:\ttest\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n",
        encoding="utf-8",
    )
    (process / "cwd").symlink_to(stable)
    (process / "exe").symlink_to("/usr/bin/python3")
    for name in ("cmdline", "environ", "maps"):
        (process / name).write_bytes(b"")
    with pytest.raises(LabError, match="active dependent processes"):
        perform_guarded_stable_sync(
            profile,
            repository="AgenticRobot",
            reservation_id="maint-sync-003",
            proc_root=blocked_proc,
            scanner_euid=uid,
            fetch_runner=_local_fetch_runner(remote),
        )
    allowed = assert_managed_runtime_start_allowed(
        profile,
        repository="AgenticRobot",
    )
    assert allowed["allowed"] is True

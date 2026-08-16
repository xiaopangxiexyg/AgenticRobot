from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control.maintenance import (
    build_maintenance_receipt,
    validate_maintenance_receipt,
)
from lab_control.maintenance_reservation import (
    acquire_maintenance_reservation,
    assert_managed_runtime_start_allowed,
    release_maintenance_reservation,
)
from lab_control.preparation_profile import (
    ControlPlanePlacement,
    PreparationProfile,
    StableSyncPlacement,
)
from lab_control.project_cli import main as project_main
from lab_control.runtime_inventory import collect_checkout_runtime_inventory
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


def _profile(tmp_path: Path) -> PreparationProfile:
    repository = (tmp_path / "stable").resolve()
    _run("git", "init", "-b", "main", str(repository))
    _run("git", "config", "user.name", "Maintenance Test", cwd=repository)
    _run(
        "git",
        "config",
        "user.email",
        "maintenance@example.invalid",
        cwd=repository,
    )
    (repository / "state.txt").write_text("stable\n", encoding="utf-8")
    _run("git", "add", "state.txt", cwd=repository)
    _run("git", "commit", "-m", "stable", cwd=repository)
    _run(
        "git",
        "remote",
        "add",
        "origin",
        "git@example.invalid:TestOrg/Stable.git",
        cwd=repository,
    )
    _run(
        "git",
        "update-ref",
        "refs/remotes/origin/main",
        "HEAD",
        cwd=repository,
    )
    control_plane_root = (tmp_path / "control-plane").resolve()
    control_plane_root.mkdir()
    return PreparationProfile(
        profile_id="test-preparation-v1",
        host_id="test-host",
        stable_sync={
            "AgenticRobot": StableSyncPlacement(
                repository="AgenticRobot",
                canonical_identity="TestOrg/Stable",
                path=repository,
                branch="main",
                remote_ref="refs/remotes/origin/main",
            )
        },
        source_anchors={},
        control_plane=ControlPlanePlacement(
            repository="AgenticRobot",
            install_root=control_plane_root,
            current_link=control_plane_root / "current",
            environment_id="test-control-env",
            max_versions=3,
        ),
        allowed_target_roots=(tmp_path,),
        path=tmp_path / "profile.json",
        sha256="a" * 64,
    )


def _rehash(receipt: dict[str, object]) -> None:
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()


def _inventory(
    tmp_path: Path,
    profile: PreparationProfile,
) -> dict[str, object]:
    proc = (tmp_path / "proc").resolve()
    proc.mkdir(exist_ok=True)
    checkout = profile.stable_sync["AgenticRobot"].path
    return collect_checkout_runtime_inventory(
        checkout,
        proc_root=proc,
        scanner_euid=os.geteuid(),
    )


def test_maintenance_receipt_binds_clean_idle_checkout(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    reservation = acquire_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        reservation_id="maint-test-001",
    )

    receipt = build_maintenance_receipt(
        profile,
        repository="AgenticRobot",
        runtime_inventory=_inventory(tmp_path, profile),
        maintenance_reservation=reservation,
    )

    validated = validate_maintenance_receipt(
        receipt,
        profile=profile,
        repository="AgenticRobot",
    )
    assert validated["stable_checkout_idle"] is True
    assert validated["device_access"] is False
    assert validated["motion_safety_claimed"] is False
    assert len(str(validated["receipt_sha256"])) == 64


def test_managed_runtime_gate_blocks_active_maintenance_and_reopens(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    reservation = acquire_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        reservation_id="maint-gate-001",
    )

    with pytest.raises(LabError, match="blocked by maintenance reservation"):
        assert_managed_runtime_start_allowed(
            profile,
            repository="AgenticRobot",
        )
    released = release_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        active_receipt=reservation,
    )
    assert released["status"] == "RELEASED"
    allowed = assert_managed_runtime_start_allowed(
        profile,
        repository="AgenticRobot",
    )
    assert allowed["allowed"] is True
    assert allowed["active_maintenance_reservation"] is None


def test_maintenance_receipt_refuses_incomplete_or_active_inventory(
    tmp_path: Path,
) -> None:
    profile = _profile(tmp_path)
    reservation = acquire_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        reservation_id="maint-test-002",
    )
    incomplete = _inventory(tmp_path, profile)
    incomplete["inventory_complete"] = False
    incomplete["inaccessible_owner_pids"] = [123]
    _rehash(incomplete)
    with pytest.raises(LabError, match="incomplete or unsafe"):
        build_maintenance_receipt(
            profile,
            repository="AgenticRobot",
            runtime_inventory=incomplete,
            maintenance_reservation=reservation,
        )
    active = _inventory(tmp_path, profile)
    active["dependencies"] = [{"pid": 123, "evidence": ["cwd"]}]
    _rehash(active)
    with pytest.raises(LabError, match="active dependent processes"):
        build_maintenance_receipt(
            profile,
            repository="AgenticRobot",
            runtime_inventory=active,
            maintenance_reservation=reservation,
        )
    released = release_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        active_receipt=reservation,
    )
    with pytest.raises(LabError, match="not active"):
        build_maintenance_receipt(
            profile,
            repository="AgenticRobot",
            runtime_inventory=_inventory(tmp_path, profile),
            maintenance_reservation=released,
        )


def test_maintenance_receipt_rejects_tampering_and_expiration(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    reservation = acquire_maintenance_reservation(
        profile,
        repository="AgenticRobot",
        reservation_id="maint-test-005",
    )
    receipt = build_maintenance_receipt(
        profile,
        repository="AgenticRobot",
        runtime_inventory=_inventory(tmp_path, profile),
        maintenance_reservation=reservation,
    )
    tampered = dict(receipt)
    tampered["local_commit"] = "f" * 40
    with pytest.raises(LabError, match="hash mismatch"):
        validate_maintenance_receipt(
            tampered,
            profile=profile,
            repository="AgenticRobot",
        )

    expired = dict(receipt)
    now = datetime.now(timezone.utc)
    expired["issued_at"] = (now - timedelta(minutes=2)).isoformat()
    expired["expires_at"] = (now - timedelta(minutes=1)).isoformat()
    _rehash(expired)
    with pytest.raises(LabError, match="expired"):
        validate_maintenance_receipt(
            expired,
            profile=profile,
            repository="AgenticRobot",
        )


def test_projectctl_trusted_prepare_status_uses_protected_placements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile = _profile(tmp_path)
    loaded: dict[str, object] = {}

    def load_host(root: Path, profile_id: str) -> object:
        loaded["host_root"] = root
        loaded["host_profile_id"] = profile_id
        return object()

    def load_preparation(
        root: Path,
        profile_id: str,
        *,
        host_profile: object,
    ) -> PreparationProfile:
        loaded["preparation_root"] = root
        loaded["preparation_profile_id"] = profile_id
        loaded["host_profile"] = host_profile
        return profile

    monkeypatch.setattr("lab_control.project_cli.load_host_profile", load_host)
    monkeypatch.setattr(
        "lab_control.project_cli.load_preparation_profile",
        load_preparation,
    )

    assert (
        project_main(
            [
                "trusted-prepare",
                "status",
                "--preparation-profile",
                "remote-bootstrap-v1",
                "--host-profile",
                "remote-host-v1",
                "--repository",
                "AgenticRobot",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["action"] == "NOOP"
    assert output["network_used"] is False
    assert output["repository"] == str(profile.stable_sync["AgenticRobot"].path)
    assert str(loaded["host_root"]).endswith("integration/host_profiles")
    assert str(loaded["preparation_root"]).endswith(
        "integration/preparation_profiles"
    )

    observed: dict[str, object] = {}

    def collect(path: Path) -> dict[str, object]:
        observed["path"] = path
        return {
            "schema": "test.runtime-inventory",
            "inventory_complete": True,
            "device_access": False,
        }

    monkeypatch.setattr(
        "lab_control.project_cli.collect_checkout_runtime_inventory",
        collect,
    )
    assert (
        project_main(
            [
                "trusted-prepare",
                "runtime-inventory",
                "--preparation-profile",
                "remote-bootstrap-v1",
                "--host-profile",
                "remote-host-v1",
                "--repository",
                "AgenticRobot",
            ]
        )
        == 0
    )
    inventory = json.loads(capsys.readouterr().out)
    assert inventory["inventory_complete"] is True
    assert observed["path"] == profile.stable_sync["AgenticRobot"].path

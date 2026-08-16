from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lab_control.runtime_inventory import (
    collect_checkout_runtime_inventory,
    validate_checkout_runtime_inventory,
)
from lab_control.workspace import LabError


def _process(proc: Path, pid: int, *, uid: int, checkout: Path | None = None) -> None:
    root = proc / str(pid)
    root.mkdir()
    (root / "status").write_text(f"Name:\ttest\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    outside = proc.parent / "outside"
    outside.mkdir(exist_ok=True)
    (root / "cwd").symlink_to(checkout if checkout is not None else outside)
    (root / "exe").symlink_to("/usr/bin/python3")
    (root / "cmdline").write_bytes(b"python\0worker.py\0")
    (root / "environ").write_bytes(b"PATH=/usr/bin\0")
    (root / "maps").write_bytes(b"")


def _rehash(receipt: dict[str, object]) -> None:
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode()
    receipt["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()


def test_inventory_detects_owner_process_checkout_dependency(tmp_path: Path) -> None:
    checkout = (tmp_path / "stable").resolve()
    checkout.mkdir()
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()
    uid = os.geteuid()
    _process(proc, 101, uid=uid, checkout=checkout)
    _process(proc, 102, uid=uid)
    _process(proc, 201, uid=uid + 1, checkout=checkout)

    receipt = collect_checkout_runtime_inventory(
        checkout,
        proc_root=proc,
        scanner_euid=uid,
    )

    assert receipt["inventory_complete"] is True
    assert receipt["dependencies"] == [{"pid": 101, "evidence": ["cwd"]}]
    assert receipt["owner_pids_scanned"] == [101, 102]
    assert receipt["foreign_pids_skipped"] == 1
    assert receipt["processes_modified"] is False
    assert receipt["device_access"] is False
    assert (
        validate_checkout_runtime_inventory(receipt, checkout_path=checkout)
        == receipt
    )


def test_inventory_refuses_wrong_scanner_and_incomplete_owner_metadata(
    tmp_path: Path,
) -> None:
    checkout = (tmp_path / "stable").resolve()
    checkout.mkdir()
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()
    uid = os.geteuid()
    with pytest.raises(LabError, match="checkout owner"):
        collect_checkout_runtime_inventory(
            checkout,
            proc_root=proc,
            scanner_euid=uid + 1,
        )

    process = proc / "101"
    process.mkdir()
    (process / "status").write_text("Name:\ttest\nUid:\tinvalid\n")
    receipt = collect_checkout_runtime_inventory(
        checkout,
        proc_root=proc,
        scanner_euid=uid,
    )
    assert receipt["inventory_complete"] is False
    assert receipt["inaccessible_owner_pids"] == [101]
    with pytest.raises(LabError, match="incomplete or unsafe"):
        validate_checkout_runtime_inventory(receipt, checkout_path=checkout)


def test_inventory_receipt_rejects_staleness_and_tampering(tmp_path: Path) -> None:
    checkout = (tmp_path / "stable").resolve()
    checkout.mkdir()
    proc = (tmp_path / "proc").resolve()
    proc.mkdir()
    receipt = collect_checkout_runtime_inventory(
        checkout,
        proc_root=proc,
        scanner_euid=os.geteuid(),
    )
    tampered = dict(receipt)
    tampered["foreign_pids_skipped"] = 999
    with pytest.raises(LabError, match="hash mismatch"):
        validate_checkout_runtime_inventory(tampered, checkout_path=checkout)

    stale = dict(receipt)
    stale["observed_at"] = (
        datetime.now(timezone.utc) - timedelta(minutes=2)
    ).isoformat()
    _rehash(stale)
    with pytest.raises(LabError, match="stale"):
        validate_checkout_runtime_inventory(stale, checkout_path=checkout)

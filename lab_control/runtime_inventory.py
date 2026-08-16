from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from .workspace import LabError

RUNTIME_INVENTORY_SCHEMA_ID = "lab.checkout-runtime-inventory.v1"
MAX_PROCESS_FILE_BYTES = 8 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _owned_uid(status: bytes) -> int | None:
    for line in status.splitlines():
        if line.startswith(b"Uid:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1])
                except ValueError:
                    return None
    return None


def _read_limited(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(MAX_PROCESS_FILE_BYTES + 1)
    if len(content) > MAX_PROCESS_FILE_BYTES:
        raise LabError(f"runtime inventory process metadata is oversized: {path}")
    return content


def _path_within(raw_path: str, checkout: Path) -> bool:
    try:
        path = Path(raw_path)
        return path.is_absolute() and path.resolve().is_relative_to(checkout)
    except OSError:
        return False


def _safe_existing_directory(path: Path, *, label: str) -> Path:
    requested = path.expanduser()
    resolved = requested.resolve()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != resolved
        or not resolved.is_dir()
    ):
        raise LabError(f"{label} is unsafe: {resolved}")
    return resolved


def collect_checkout_runtime_inventory(
    checkout_path: Path,
    *,
    proc_root: Path = Path("/proc"),
    scanner_euid: int | None = None,
) -> dict[str, object]:
    """Observe same-owner checkout dependencies without signalling any process."""

    checkout = _safe_existing_directory(
        checkout_path,
        label="runtime inventory checkout",
    )
    owner_uid = checkout.stat().st_uid
    effective_uid = os.geteuid() if scanner_euid is None else scanner_euid
    if effective_uid != owner_uid:
        raise LabError("runtime inventory scanner must run as the checkout owner")
    proc = _safe_existing_directory(proc_root, label="runtime inventory proc root")

    dependencies: list[dict[str, object]] = []
    inaccessible_owner_pids: list[int] = []
    owner_pids_scanned: list[int] = []
    foreign_pids_skipped = 0
    vanished_pids = 0
    checkout_bytes = str(checkout).encode()
    process_paths = sorted(
        (path for path in proc.iterdir() if path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for process_path in process_paths:
        pid = int(process_path.name)
        try:
            status = _read_limited(process_path / "status")
        except FileNotFoundError:
            vanished_pids += 1
            continue
        except (OSError, LabError):
            inaccessible_owner_pids.append(pid)
            continue
        uid = _owned_uid(status)
        if uid is None:
            inaccessible_owner_pids.append(pid)
            continue
        if uid != owner_uid:
            foreign_pids_skipped += 1
            continue
        owner_pids_scanned.append(pid)
        evidence: set[str] = set()
        for name in ("cwd", "exe"):
            try:
                target = os.readlink(process_path / name)
            except FileNotFoundError:
                continue
            except OSError:
                inaccessible_owner_pids.append(pid)
                evidence.add(f"{name}:unreadable")
                continue
            if _path_within(target, checkout):
                evidence.add(name)
        for name in ("cmdline", "environ", "maps"):
            try:
                metadata = _read_limited(process_path / name)
            except FileNotFoundError:
                continue
            except (OSError, LabError):
                inaccessible_owner_pids.append(pid)
                evidence.add(f"{name}:unreadable")
                continue
            if checkout_bytes in metadata:
                evidence.add(name)
        concrete = sorted(item for item in evidence if not item.endswith(":unreadable"))
        if concrete:
            dependencies.append({"pid": pid, "evidence": concrete})
    inaccessible = sorted(set(inaccessible_owner_pids))
    content: dict[str, object] = {
        "schema": RUNTIME_INVENTORY_SCHEMA_ID,
        "checkout_path": str(checkout),
        "checkout_owner_uid": owner_uid,
        "scanner_euid": effective_uid,
        "scope": "checkout-owner-active-processes",
        "owner_pids_scanned": sorted(set(owner_pids_scanned)),
        "foreign_pids_skipped": foreign_pids_skipped,
        "vanished_pids": vanished_pids,
        "inaccessible_owner_pids": inaccessible,
        "dependencies": dependencies,
        "inventory_complete": not inaccessible,
        "processes_modified": False,
        "device_access": False,
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def validate_checkout_runtime_inventory(
    receipt: object,
    *,
    checkout_path: Path,
    max_age_seconds: int = 30,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "checkout_path",
        "checkout_owner_uid",
        "scanner_euid",
        "scope",
        "owner_pids_scanned",
        "foreign_pids_skipped",
        "vanished_pids",
        "inaccessible_owner_pids",
        "dependencies",
        "inventory_complete",
        "processes_modified",
        "device_access",
        "observed_at",
        "receipt_sha256",
    }:
        raise LabError("runtime inventory receipt has an unexpected shape")
    checkout = _safe_existing_directory(
        checkout_path,
        label="runtime inventory checkout",
    )
    owner_uid = checkout.stat().st_uid
    if (
        receipt.get("schema") != RUNTIME_INVENTORY_SCHEMA_ID
        or receipt.get("checkout_path") != str(checkout)
        or receipt.get("checkout_owner_uid") != owner_uid
        or receipt.get("scanner_euid") != owner_uid
        or receipt.get("scope") != "checkout-owner-active-processes"
        or receipt.get("inventory_complete") is not True
        or receipt.get("inaccessible_owner_pids") != []
        or receipt.get("processes_modified") is not False
        or receipt.get("device_access") is not False
    ):
        raise LabError("runtime inventory receipt is incomplete or unsafe")
    if not 1 <= max_age_seconds <= 60:
        raise LabError("runtime inventory max age must be between 1 and 60 seconds")
    observed_at = receipt.get("observed_at")
    if not isinstance(observed_at, str):
        raise LabError("runtime inventory receipt has invalid timestamp")
    try:
        observed = datetime.fromisoformat(observed_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("runtime inventory receipt has invalid timestamp") from exc
    now = datetime.now(timezone.utc)
    if observed.tzinfo is None or not 0 <= (now - observed).total_seconds() <= max_age_seconds:
        raise LabError("runtime inventory receipt is stale or not yet valid")
    for field in ("owner_pids_scanned", "inaccessible_owner_pids", "dependencies"):
        if not isinstance(receipt.get(field), list):
            raise LabError(f"runtime inventory receipt has invalid {field}")
    dependencies = receipt.get("dependencies")
    if not isinstance(dependencies, list):
        raise LabError("runtime inventory receipt has invalid dependencies")
    for dependency in dependencies:
        if (
            not isinstance(dependency, dict)
            or set(dependency) != {"pid", "evidence"}
            or not isinstance(dependency.get("pid"), int)
            or not isinstance(dependency.get("evidence"), list)
            or not dependency["evidence"]
            or not all(isinstance(item, str) for item in dependency["evidence"])
        ):
            raise LabError("runtime inventory receipt has invalid dependency")
    if not isinstance(receipt.get("foreign_pids_skipped"), int) or not isinstance(
        receipt.get("vanished_pids"), int
    ):
        raise LabError("runtime inventory receipt has invalid process counts")
    digest = receipt.get("receipt_sha256")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        not _SHA256_PATTERN.fullmatch(str(digest))
        or digest != hashlib.sha256(_canonical_json(content)).hexdigest()
    ):
        raise LabError("runtime inventory receipt hash mismatch")
    return receipt

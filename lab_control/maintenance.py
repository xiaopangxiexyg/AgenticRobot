from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

from . import git
from .maintenance_reservation import validate_maintenance_reservation
from .preparation_profile import PreparationProfile
from .remote_request import OID_PATTERN, SHA256_PATTERN
from .runtime_inventory import validate_checkout_runtime_inventory
from .workspace import LabError, validate_repository_name, validate_task_id

MAINTENANCE_RECEIPT_SCHEMA_ID = "lab.maintenance-receipt.v1"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _timestamp(value: object, *, label: str) -> datetime:
    if not isinstance(value, str):
        raise LabError(f"maintenance receipt has invalid {label}")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError(f"maintenance receipt has invalid {label}") from exc
    if parsed.tzinfo is None:
        raise LabError(f"maintenance receipt {label} requires a timezone")
    return parsed


def build_maintenance_receipt(
    profile: PreparationProfile,
    *,
    repository: str,
    runtime_inventory: object,
    maintenance_reservation: object,
    validity_seconds: int = 30,
) -> dict[str, object]:
    """Issue a short-lived idle-checkout receipt from trusted observation facts."""

    validate_repository_name(repository)
    placement = profile.stable_sync.get(repository)
    if placement is None:
        raise LabError(f"repository has no stable maintenance policy: {repository}")
    inventory = validate_checkout_runtime_inventory(
        runtime_inventory,
        checkout_path=placement.path,
    )
    dependencies = inventory.get("dependencies")
    if not isinstance(dependencies, list) or dependencies:
        raise LabError("maintenance checkout still has active dependent processes")
    reservation = validate_maintenance_reservation(
        maintenance_reservation,
        profile=profile,
        repository=repository,
        require_active=True,
    )
    if reservation.get("local_commit") != git.current_commit(placement.path):
        raise LabError("maintenance reservation does not bind current checkout HEAD")
    if not 1 <= validity_seconds <= 60:
        raise LabError("maintenance receipt validity must be between 1 and 60 seconds")
    if git.current_branch(placement.path) != placement.branch or git.status_short(
        placement.path
    ):
        raise LabError("maintenance checkout is not clean on the protected branch")
    issued = datetime.now(timezone.utc)
    content: dict[str, object] = {
        "schema": MAINTENANCE_RECEIPT_SCHEMA_ID,
        "preparation_profile_id": profile.profile_id,
        "preparation_profile_sha256": profile.sha256,
        "host_id": profile.host_id,
        "repository": repository,
        "checkout_path": str(placement.path),
        "branch": placement.branch,
        "local_commit": git.current_commit(placement.path),
        "runtime_inventory": inventory,
        "maintenance_reservation": reservation,
        "maintenance_reservation_id": reservation["reservation_id"],
        "runtime_start_blocked": True,
        "runtime_dependency_inventory_complete": True,
        "stable_checkout_idle": True,
        "active_checkout_processes": [],
        "device_access": False,
        "motion_safety_claimed": False,
        "issued_at": issued.isoformat(),
        "expires_at": (issued + timedelta(seconds=validity_seconds)).isoformat(),
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def validate_maintenance_receipt(
    receipt: object,
    *,
    profile: PreparationProfile,
    repository: str,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "preparation_profile_id",
        "preparation_profile_sha256",
        "host_id",
        "repository",
        "checkout_path",
        "branch",
        "local_commit",
        "runtime_inventory",
        "maintenance_reservation",
        "maintenance_reservation_id",
        "runtime_start_blocked",
        "runtime_dependency_inventory_complete",
        "stable_checkout_idle",
        "active_checkout_processes",
        "device_access",
        "motion_safety_claimed",
        "issued_at",
        "expires_at",
        "receipt_sha256",
    }:
        raise LabError("maintenance receipt has an unexpected shape")
    if not SHA256_PATTERN.fullmatch(str(receipt.get("receipt_sha256", ""))):
        raise LabError("maintenance receipt has invalid hash")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != hashlib.sha256(_canonical_json(content)).hexdigest():
        raise LabError("maintenance receipt hash mismatch")
    validate_repository_name(repository)
    placement = profile.stable_sync.get(repository)
    if placement is None:
        raise LabError(f"repository has no stable maintenance policy: {repository}")
    if (
        receipt.get("schema") != MAINTENANCE_RECEIPT_SCHEMA_ID
        or receipt.get("preparation_profile_id") != profile.profile_id
        or receipt.get("preparation_profile_sha256") != profile.sha256
        or receipt.get("host_id") != profile.host_id
        or receipt.get("repository") != repository
        or receipt.get("checkout_path") != str(placement.path)
        or receipt.get("branch") != placement.branch
    ):
        raise LabError("maintenance receipt identity mismatch")
    validate_task_id(str(receipt["host_id"]))
    reservation_id = receipt.get("maintenance_reservation_id")
    if not isinstance(reservation_id, str):
        raise LabError("maintenance receipt has invalid reservation ID")
    validate_task_id(reservation_id)
    if not OID_PATTERN.fullmatch(str(receipt.get("local_commit", ""))):
        raise LabError("maintenance receipt has invalid local commit")
    inventory = validate_checkout_runtime_inventory(
        receipt.get("runtime_inventory"),
        checkout_path=placement.path,
    )
    reservation = validate_maintenance_reservation(
        receipt.get("maintenance_reservation"),
        profile=profile,
        repository=repository,
        require_active=True,
    )
    if (
        inventory.get("dependencies") != []
        or reservation.get("reservation_id") != reservation_id
        or reservation.get("local_commit") != receipt.get("local_commit")
        or receipt.get("runtime_start_blocked") is not True
        or receipt.get("runtime_dependency_inventory_complete") is not True
        or receipt.get("stable_checkout_idle") is not True
        or receipt.get("active_checkout_processes") != []
        or receipt.get("device_access") is not False
        or receipt.get("motion_safety_claimed") is not False
    ):
        raise LabError("maintenance receipt has an unsafe idle envelope")
    issued = _timestamp(receipt.get("issued_at"), label="issued_at")
    expires = _timestamp(receipt.get("expires_at"), label="expires_at")
    now = datetime.now(timezone.utc)
    if expires <= issued or not issued <= now <= expires:
        raise LabError("maintenance receipt is expired or not yet valid")
    return receipt

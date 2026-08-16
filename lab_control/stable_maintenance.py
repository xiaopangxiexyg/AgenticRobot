from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .maintenance_gate import (
    StableCheckoutTarget,
    acquire_maintenance_reservation,
    release_maintenance_reservation,
    validate_maintenance_reservation,
)
from .runtime_inventory import (
    collect_checkout_runtime_inventory,
    validate_checkout_runtime_inventory,
)
from .stable_sync import (
    apply_stable_checkout_fast_forward,
    plan_stable_checkout_sync,
    validate_stable_sync_plan,
)
from .workspace import LabError

MAINTENANCE_RECEIPT_SCHEMA_ID = "lab.maintenance-receipt.v2"
GUARDED_SYNC_RECEIPT_SCHEMA_ID = "lab.guarded-stable-sync-receipt.v2"
InventoryCollector = Callable[[Path], dict[str, object]]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


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
    target: StableCheckoutTarget,
    *,
    plan: object,
    runtime_inventory: object,
    maintenance_reservation: object,
    validity_seconds: int = 30,
) -> dict[str, object]:
    """Bind a short-lived idle observation to a protected checkout plan."""

    validated_plan = validate_stable_sync_plan(plan)
    inventory = validate_checkout_runtime_inventory(
        runtime_inventory,
        checkout_path=target.checkout_path,
    )
    dependencies = inventory.get("dependencies")
    if not isinstance(dependencies, list) or dependencies:
        raise LabError("maintenance checkout still has active dependent processes")
    reservation = validate_maintenance_reservation(
        maintenance_reservation,
        target=target,
        require_unexpired=True,
    )
    if (
        reservation.get("plan_sha256") != validated_plan.get("plan_sha256")
        or reservation.get("local_commit") != validated_plan.get("local_commit")
        or reservation.get("local_tree_oid") != validated_plan.get("local_tree_oid")
        or reservation.get("checkout_status") != []
    ):
        raise LabError("maintenance reservation does not bind the stable sync plan")
    if not 1 <= validity_seconds <= 60:
        raise LabError("maintenance receipt validity must be between 1 and 60 seconds")
    current = plan_stable_checkout_sync(
        target.checkout_path,
        expected_identity=target.canonical_identity,
        branch=target.branch,
        remote_ref=target.remote_ref,
    )
    if current != validated_plan:
        raise LabError("stable sync state changed before maintenance receipt")
    issued = datetime.now(timezone.utc)
    content: dict[str, object] = {
        "schema": MAINTENANCE_RECEIPT_SCHEMA_ID,
        "target": target.identity(),
        "policy_sha256": target.policy_sha256,
        "plan_sha256": validated_plan["plan_sha256"],
        "local_commit": validated_plan["local_commit"],
        "local_tree_oid": validated_plan["local_tree_oid"],
        "remote_commit": validated_plan["remote_commit"],
        "remote_tree_oid": validated_plan["remote_tree_oid"],
        "runtime_inventory": inventory,
        "maintenance_reservation": reservation,
        "maintenance_reservation_id": reservation["reservation_id"],
        "runtime_start_blocked": True,
        "runtime_dependency_inventory_complete": True,
        "stable_checkout_idle": True,
        "active_checkout_processes": [],
        "network_used": False,
        "issued_at": issued.isoformat(),
        "expires_at": (issued + timedelta(seconds=validity_seconds)).isoformat(),
    }
    return {**content, "receipt_sha256": _hash(content)}


def validate_maintenance_receipt(
    receipt: object,
    *,
    target: StableCheckoutTarget,
) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "target",
        "policy_sha256",
        "plan_sha256",
        "local_commit",
        "local_tree_oid",
        "remote_commit",
        "remote_tree_oid",
        "runtime_inventory",
        "maintenance_reservation",
        "maintenance_reservation_id",
        "runtime_start_blocked",
        "runtime_dependency_inventory_complete",
        "stable_checkout_idle",
        "active_checkout_processes",
        "network_used",
        "issued_at",
        "expires_at",
        "receipt_sha256",
    }:
        raise LabError("maintenance receipt has an unexpected shape")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt.get("receipt_sha256") != _hash(content):
        raise LabError("maintenance receipt hash mismatch")
    if (
        receipt.get("schema") != MAINTENANCE_RECEIPT_SCHEMA_ID
        or receipt.get("target") != target.identity()
        or receipt.get("policy_sha256") != target.policy_sha256
        or receipt.get("runtime_start_blocked") is not True
        or receipt.get("runtime_dependency_inventory_complete") is not True
        or receipt.get("stable_checkout_idle") is not True
        or receipt.get("active_checkout_processes") != []
        or receipt.get("network_used") is not False
    ):
        raise LabError("maintenance receipt identity or idle envelope is unsafe")
    inventory = validate_checkout_runtime_inventory(
        receipt.get("runtime_inventory"),
        checkout_path=target.checkout_path,
    )
    reservation = validate_maintenance_reservation(
        receipt.get("maintenance_reservation"),
        target=target,
        require_unexpired=True,
    )
    if (
        inventory.get("dependencies") != []
        or reservation.get("reservation_id")
        != receipt.get("maintenance_reservation_id")
        or reservation.get("plan_sha256") != receipt.get("plan_sha256")
        or reservation.get("local_commit") != receipt.get("local_commit")
        or reservation.get("local_tree_oid") != receipt.get("local_tree_oid")
        or reservation.get("checkout_status") != []
    ):
        raise LabError("maintenance receipt does not bind its evidence")
    issued = _timestamp(receipt.get("issued_at"), label="issued_at")
    expires = _timestamp(receipt.get("expires_at"), label="expires_at")
    now = datetime.now(timezone.utc)
    if expires <= issued or not issued <= now <= expires:
        raise LabError("maintenance receipt is expired or not yet valid")
    return receipt


def guarded_stable_checkout_fast_forward(
    target: StableCheckoutTarget,
    *,
    plan: object,
    reservation_id: str,
    inventory_collector: InventoryCollector = collect_checkout_runtime_inventory,
) -> dict[str, object]:
    """Apply one exact fast-forward while a cooperating runtime start gate is held."""

    validated_plan = validate_stable_sync_plan(plan)
    if validated_plan.get("action") != "FAST_FORWARD_READY":
        raise LabError("guarded stable sync requires a fast-forward-ready plan")
    reservation = acquire_maintenance_reservation(
        target,
        plan=validated_plan,
        reservation_id=reservation_id,
    )
    maintenance: dict[str, object] | None = None
    sync_receipt: dict[str, object] | None = None
    released: dict[str, object] | None = None
    try:
        inventory = inventory_collector(target.checkout_path)
        maintenance = build_maintenance_receipt(
            target,
            plan=validated_plan,
            runtime_inventory=inventory,
            maintenance_reservation=reservation,
        )
        validate_maintenance_receipt(maintenance, target=target)
        current = plan_stable_checkout_sync(
            target.checkout_path,
            expected_identity=target.canonical_identity,
            branch=target.branch,
            remote_ref=target.remote_ref,
        )
        if current != validated_plan:
            raise LabError("stable sync state changed after maintenance observation")
        sync_receipt = apply_stable_checkout_fast_forward(validated_plan)
    finally:
        released = release_maintenance_reservation(
            target,
            active_receipt=reservation,
        )
    if maintenance is None or sync_receipt is None or released is None:
        raise LabError("guarded stable sync did not produce complete evidence")
    content: dict[str, object] = {
        "schema": GUARDED_SYNC_RECEIPT_SCHEMA_ID,
        "target": target.identity(),
        "policy_sha256": target.policy_sha256,
        "plan_sha256": validated_plan["plan_sha256"],
        "reservation_id": reservation_id,
        "maintenance_receipt": maintenance,
        "stable_sync_receipt": sync_receipt,
        "released_reservation": released,
        "before_commit": validated_plan["local_commit"],
        "after_commit": sync_receipt["after_commit"],
        "network_used": False,
        "worktree_files_copied": False,
        "task_worktrees_updated": False,
        "submodules_updated": False,
        "lfs_smudge_enabled": False,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**content, "receipt_sha256": _hash(content)}

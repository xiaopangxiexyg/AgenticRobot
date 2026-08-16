from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .baseline import load_baseline_lock
from .maintenance_gate import StableCheckoutTarget
from .project_sync import validate_project_sync_plan
from .runtime_inventory import collect_checkout_runtime_inventory
from .stable_maintenance import InventoryCollector, guarded_stable_checkout_fast_forward
from .stable_sync import plan_stable_checkout_sync
from .workspace import (
    LabError,
    WorkspaceManager,
    _atomic_write_text,
    validate_repository_name,
    validate_task_id,
)

PROJECT_GUARDED_SYNC_SCHEMA_ID = "lab.project-guarded-sync-receipt.v2"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def persist_project_sync_receipt(
    manager: WorkspaceManager,
    receipt: dict[str, object],
) -> None:
    sync_id = validate_task_id(str(receipt.get("sync_id", "")))
    expected = manager.state_root / "project-sync" / f"{sync_id}.json"
    configured = Path(str(receipt.get("receipt_path", ""))).resolve()
    if configured != expected.resolve():
        raise LabError("guarded sync receipt path identity mismatch")
    root = manager.state_root
    if root.is_symlink():
        raise LabError(f"project sync state root is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    receipt_root = expected.parent
    if receipt_root.is_symlink():
        raise LabError(f"project sync receipt root is a symlink: {receipt_root}")
    receipt_root.mkdir(parents=True, exist_ok=True)
    if not receipt_root.resolve().is_relative_to(root.resolve()):
        raise LabError("guarded sync receipt root escapes configured state")
    if expected.is_symlink():
        raise LabError(f"guarded sync receipt path is a symlink: {expected}")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with manager.advisory_locks("project-sync", [sync_id]):
        if expected.is_file():
            if expected.read_text(encoding="utf-8") != encoded:
                raise LabError(f"guarded sync receipt identity collision: {sync_id}")
            return
        _atomic_write_text(expected, encoded)


def apply_guarded_project_checkout(
    manager: WorkspaceManager,
    plan: object,
    *,
    repository: str,
    confirmed_plan_sha256: str,
    reservation_id: str,
    inventory_collector: InventoryCollector = collect_checkout_runtime_inventory,
) -> dict[str, object]:
    """Apply one unlinked stable checkout; linked repositories require a later transaction."""

    name = validate_repository_name(repository)
    validate_task_id(reservation_id)
    validated = validate_project_sync_plan(plan)
    if validated.get("project_manifest_sha256") != manager.project.sha256:
        raise LabError("project manifest changed after sync planning")
    requested = validated.get("requested_repositories")
    effective = validated.get("effective_repositories")
    if requested != [name] or effective != [name]:
        raise LabError("M3b guarded apply requires exactly one unlinked repository")
    baseline_path = Path(str(validated.get("baseline_path", "")))
    baseline = load_baseline_lock(baseline_path)
    if baseline.sha256 != validated.get("baseline_sha256"):
        raise LabError("baseline lock changed after sync planning")
    linked_names = {
        linked
        for link in baseline.repository_links
        for linked in (link.parent_repository, link.child_repository)
    }
    if name in linked_names:
        raise LabError(
            "M3b refuses linked repositories; use the future multi-repository "
            "gitlink transaction"
        )
    if manager.project.checkout_modes.get(name) != "stable_branch":
        raise LabError("M3b guarded apply only supports stable-branch checkouts")
    raw_repositories = validated.get("repositories")
    if not isinstance(raw_repositories, dict):
        raise LabError("project sync plan lacks repository inspections")
    inspection = raw_repositories.get(name)
    if not isinstance(inspection, dict):
        raise LabError(f"project sync plan lacks repository: {name}")
    checkout_action = inspection.get("checkout_sync_action")
    if checkout_action not in {"FAST_FORWARD_READY", "NOOP"}:
        raise LabError(
            "guarded project checkout is neither current nor fast-forward ready: "
            f"{checkout_action}"
        )
    stable_plan = plan_stable_checkout_sync(
        Path(str(inspection.get("management_path", ""))),
        expected_identity=str(inspection.get("canonical_identity_expected", "")),
        branch=str(inspection.get("stable_branch", "")),
        remote_ref=str(inspection.get("remote_ref", "")),
    )
    if stable_plan.get("plan_sha256") != inspection.get(
        "checkout_sync_plan_sha256"
    ):
        raise LabError("stable checkout plan changed after project inspection")
    if stable_plan.get("plan_sha256") != confirmed_plan_sha256:
        raise LabError("confirmed stable checkout plan hash does not match")
    if stable_plan.get("action") == "NOOP":
        completed_at = datetime.now(timezone.utc)
        sync_id = (
            f"APPLY-{completed_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
            f"{str(stable_plan['plan_sha256'])[:8]}"
        )
        receipt_path = manager.state_root / "project-sync" / f"{sync_id}.json"
        noop_content: dict[str, object] = {
            "schema": PROJECT_GUARDED_SYNC_SCHEMA_ID,
            "sync_id": sync_id,
            "completed_at": completed_at.isoformat(),
            "receipt_path": str(receipt_path),
            "project_id": manager.project.project_id,
            "project_manifest_sha256": manager.project.sha256,
            "baseline_id": baseline.baseline_id,
            "baseline_sha256": baseline.sha256,
            "repository": name,
            "project_plan_sha256": validated["plan_sha256"],
            "confirmed_stable_plan_sha256": confirmed_plan_sha256,
            "reservation_id": reservation_id,
            "guarded_sync": None,
            "status": "UP_TO_DATE",
            "network_used": False,
            "checkout_updated": False,
            "task_worktrees_updated": False,
            "worktree_files_copied": False,
            "submodules_updated": False,
            "lfs_smudge_enabled": False,
            "large_assets_downloaded": False,
            "processes_signalled": False,
            "executor_pid": os.getpid(),
        }
        receipt = {
            **noop_content,
            "receipt_sha256": _hash(noop_content),
        }
        persist_project_sync_receipt(manager, receipt)
        return receipt
    host_id = manager.environ.get(
        "LAB_HOST_ID",
        manager.project.default_development_host,
    )
    validate_task_id(host_id)
    target = StableCheckoutTarget(
        project_id=manager.project.project_id,
        host_id=host_id,
        repository=name,
        checkout_path=Path(str(stable_plan["repository"])),
        canonical_identity=str(stable_plan["canonical_identity"]),
        branch=str(stable_plan["branch"]),
        remote_ref=str(stable_plan["remote_ref"]),
        state_root=manager.state_root,
    )
    guarded = guarded_stable_checkout_fast_forward(
        target,
        plan=stable_plan,
        reservation_id=reservation_id,
        inventory_collector=inventory_collector,
    )
    completed_at = datetime.now(timezone.utc)
    sync_id = (
        f"APPLY-{completed_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{str(stable_plan['plan_sha256'])[:8]}"
    )
    receipt_path = manager.state_root / "project-sync" / f"{sync_id}.json"
    content: dict[str, object] = {
        "schema": PROJECT_GUARDED_SYNC_SCHEMA_ID,
        "sync_id": sync_id,
        "completed_at": completed_at.isoformat(),
        "receipt_path": str(receipt_path),
        "project_id": manager.project.project_id,
        "project_manifest_sha256": manager.project.sha256,
        "baseline_id": baseline.baseline_id,
        "baseline_sha256": baseline.sha256,
        "repository": name,
        "project_plan_sha256": validated["plan_sha256"],
        "confirmed_stable_plan_sha256": confirmed_plan_sha256,
        "guarded_sync": guarded,
        "status": "COMPLETE",
        "network_used": False,
        "checkout_updated": True,
        "task_worktrees_updated": False,
        "worktree_files_copied": False,
        "submodules_updated": False,
        "lfs_smudge_enabled": False,
        "large_assets_downloaded": False,
        "processes_signalled": False,
        "executor_pid": os.getpid(),
    }
    receipt = {**content, "receipt_sha256": _hash(content)}
    persist_project_sync_receipt(manager, receipt)
    return receipt

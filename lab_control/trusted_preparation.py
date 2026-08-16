from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import git
from .maintenance import build_maintenance_receipt, validate_maintenance_receipt
from .maintenance_reservation import (
    acquire_maintenance_reservation,
    release_maintenance_reservation,
)
from .preparation_profile import PreparationProfile
from .runtime_inventory import collect_checkout_runtime_inventory
from .source_anchor import materialize_local_source_anchor
from .stable_sync import (
    apply_stable_checkout_fast_forward,
    plan_stable_checkout_sync,
)
from .trusted_fetch import (
    FetchRunner,
    fetch_stable_remote_ref,
    validate_trusted_fetch_receipt,
)
from .workspace import LabError, validate_repository_name


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def synchronize_stable_checkout(
    profile: PreparationProfile,
    *,
    repository: str,
    fetch_runner: FetchRunner | None = None,
    maintenance_receipt: object | None = None,
) -> dict[str, object]:
    """Execute explicit fetch + guarded ff-only for one protected placement."""

    validate_repository_name(repository)
    placement = profile.stable_sync.get(repository)
    if placement is None:
        raise LabError(f"repository has no stable sync policy: {repository}")
    if fetch_runner is None:
        raw_fetch = fetch_stable_remote_ref(
            placement.path,
            expected_identity=placement.canonical_identity,
            branch=placement.branch,
        )
    else:
        raw_fetch = fetch_stable_remote_ref(
            placement.path,
            expected_identity=placement.canonical_identity,
            branch=placement.branch,
            fetch_runner=fetch_runner,
        )
    fetched = validate_trusted_fetch_receipt(raw_fetch)
    plan = plan_stable_checkout_sync(
        placement.path,
        expected_identity=placement.canonical_identity,
        branch=placement.branch,
        remote_ref=placement.remote_ref,
    )
    action = plan.get("action")
    maintenance: dict[str, object] | None = None
    if maintenance_receipt is not None:
        maintenance = validate_maintenance_receipt(
            maintenance_receipt,
            profile=profile,
            repository=repository,
        )
        if maintenance.get("local_commit") != plan.get("local_commit"):
            raise LabError("maintenance receipt does not bind current sync HEAD")
    release: dict[str, object] | None = None
    try:
        if action == "NOOP":
            status = "UP_TO_DATE"
            applied = None
        elif action == "FAST_FORWARD_READY":
            if maintenance is None:
                applied = None
                status = "WAITING_FOR_MAINTENANCE"
            else:
                applied = apply_stable_checkout_fast_forward(plan)
                status = "SYNCED"
        else:
            applied = None
            status = "BLOCKED"
    finally:
        if maintenance is not None:
            reservation = maintenance.get("maintenance_reservation")
            release = release_maintenance_reservation(
                profile,
                repository=repository,
                active_receipt=reservation,
            )
    content: dict[str, object] = {
        "schema": "lab.stable-sync-cycle.v1",
        "preparation_profile_id": profile.profile_id,
        "preparation_profile_sha256": profile.sha256,
        "host_id": profile.host_id,
        "repository": repository,
        "fetch_receipt": fetched,
        "sync_plan": plan,
        "maintenance_receipt": maintenance_receipt,
        "maintenance_release_receipt": release,
        "apply_receipt": applied,
        "status": status,
        "task_payload_authorized": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def prepare_source_anchor(
    profile: PreparationProfile,
    *,
    repository: str,
    commit: str,
) -> dict[str, object]:
    """Materialize one exact protected source anchor without network access."""

    validate_repository_name(repository)
    placement = profile.source_anchors.get(repository)
    if placement is None:
        raise LabError(f"repository has no source anchor policy: {repository}")
    source_submodules, initialized = git.submodule_status(placement.source_path)
    if not initialized or set(source_submodules) != set(placement.submodule_sources):
        raise LabError("source anchor local submodule set differs from protected Profile")
    receipt = materialize_local_source_anchor(
        source_repository=placement.source_path,
        target=placement.target_path,
        expected_identity=placement.canonical_identity,
        commit=commit,
        submodule_sources=placement.submodule_sources,
        max_checkout_blob_bytes=placement.max_checkout_blob_bytes,
    )
    content: dict[str, object] = {
        "schema": "lab.protected-source-anchor.v1",
        "preparation_profile_id": profile.profile_id,
        "preparation_profile_sha256": profile.sha256,
        "host_id": profile.host_id,
        "repository": repository,
        "source_anchor_receipt": receipt,
        "network_used": False,
        "task_payload_authorized": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }


def perform_guarded_stable_sync(
    profile: PreparationProfile,
    *,
    repository: str,
    reservation_id: str,
    proc_root: Path = Path("/proc"),
    scanner_euid: int | None = None,
    fetch_runner: FetchRunner | None = None,
) -> dict[str, object]:
    """Reserve starts, inventory, fetch, ff-only if eligible, and release."""

    validate_repository_name(repository)
    placement = profile.stable_sync.get(repository)
    if placement is None:
        raise LabError(f"repository has no stable sync policy: {repository}")
    reservation = acquire_maintenance_reservation(
        profile,
        repository=repository,
        reservation_id=reservation_id,
    )
    try:
        inventory = collect_checkout_runtime_inventory(
            placement.path,
            proc_root=proc_root,
            scanner_euid=scanner_euid,
        )
        maintenance = build_maintenance_receipt(
            profile,
            repository=repository,
            runtime_inventory=inventory,
            maintenance_reservation=reservation,
        )
        cycle = synchronize_stable_checkout(
            profile,
            repository=repository,
            fetch_runner=fetch_runner,
            maintenance_receipt=maintenance,
        )
    except BaseException:
        try:
            release_maintenance_reservation(
                profile,
                repository=repository,
                active_receipt=reservation,
            )
        except LabError:
            pass
        raise
    content: dict[str, object] = {
        "schema": "lab.guarded-stable-sync.v1",
        "preparation_profile_id": profile.profile_id,
        "preparation_profile_sha256": profile.sha256,
        "host_id": profile.host_id,
        "repository": repository,
        "reservation_id": reservation_id,
        "runtime_inventory_receipt_sha256": inventory["receipt_sha256"],
        "maintenance_receipt_sha256": maintenance["receipt_sha256"],
        "stable_sync_cycle_receipt_sha256": cycle["receipt_sha256"],
        "status": cycle["status"],
        "maintenance_released": cycle["maintenance_release_receipt"] is not None,
        "task_payload_authorized": False,
        "device_access": False,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .baseline import load_baseline_lock
from .maintenance_gate import (
    StableCheckoutTarget,
    acquire_checkout_maintenance_reservation,
    release_maintenance_reservation,
)
from .project_maintenance import persist_project_sync_receipt
from .project_sync import plan_project_sync, validate_project_sync_plan
from .runtime_inventory import (
    collect_checkout_runtime_inventory,
    validate_checkout_runtime_inventory,
)
from .stable_sync import (
    apply_stable_checkout_head_fast_forward,
    plan_stable_checkout_sync,
)
from .workspace import (
    LabError,
    WorkspaceManager,
    validate_repository_name,
    validate_task_id,
)

LINKED_SYNC_PLAN_SCHEMA_ID = "lab.linked-checkout-sync-plan.v1"
LINKED_SYNC_RECEIPT_SCHEMA_ID = "lab.linked-checkout-sync-receipt.v1"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _status_paths(status: tuple[str, ...]) -> set[str]:
    paths: set[str] = set()
    for entry in status:
        if len(entry) < 4 or " -> " in entry:
            raise LabError("linked checkout has an unsupported status entry")
        paths.add(entry[3:])
    return paths


def _target(
    manager: WorkspaceManager,
    *,
    repository: str,
    path: Path,
    canonical_identity: str,
    branch: str,
) -> StableCheckoutTarget:
    host_id = manager.environ.get(
        "LAB_HOST_ID",
        manager.project.default_development_host,
    )
    validate_task_id(host_id)
    return StableCheckoutTarget(
        project_id=manager.project.project_id,
        host_id=host_id,
        repository=repository,
        checkout_path=path,
        canonical_identity=canonical_identity,
        branch=branch,
        remote_ref=f"refs/remotes/origin/{branch}",
        state_root=manager.state_root,
    )


def validate_linked_sync_plan(plan: object) -> dict[str, object]:
    if not isinstance(plan, dict) or set(plan) != {
        "schema",
        "project_id",
        "project_manifest_sha256",
        "baseline_id",
        "baseline_path",
        "baseline_sha256",
        "project_plan_sha256",
        "root_repository",
        "root",
        "children",
        "action",
        "network_used",
        "worktree_files_copied",
        "rebase_reset_clean_permitted",
        "submodule_fetch_permitted",
        "lfs_smudge_enabled",
        "plan_sha256",
    }:
        raise LabError("linked checkout sync plan has an unexpected shape")
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        plan.get("schema") != LINKED_SYNC_PLAN_SCHEMA_ID
        or plan.get("network_used") is not False
        or plan.get("worktree_files_copied") is not False
        or plan.get("rebase_reset_clean_permitted") is not False
        or plan.get("submodule_fetch_permitted") is not False
        or plan.get("lfs_smudge_enabled") is not False
        or plan.get("plan_sha256") != _hash(content)
    ):
        raise LabError("linked checkout sync plan is invalid or unsafe")
    return plan


def plan_linked_project_checkout(
    manager: WorkspaceManager,
    project_plan: object,
    *,
    root_repository: str,
) -> dict[str, object]:
    """Resolve one stable parent and its direct pinned children without mutation."""

    root_name = validate_repository_name(root_repository)
    validated = validate_project_sync_plan(project_plan)
    if validated.get("project_manifest_sha256") != manager.project.sha256:
        raise LabError("project manifest changed after linked sync planning")
    if validated.get("requested_repositories") != [root_name]:
        raise LabError("linked sync plan requires exactly one requested root repository")
    baseline_path = Path(str(validated.get("baseline_path", "")))
    baseline = load_baseline_lock(baseline_path)
    if baseline.sha256 != validated.get("baseline_sha256"):
        raise LabError("baseline lock changed after linked sync planning")
    direct_links = [
        link
        for link in baseline.repository_links
        if link.parent_repository == root_name
    ]
    if not direct_links:
        raise LabError("linked sync root has no pinned child repositories")
    children = {link.child_repository for link in direct_links}
    if len(children) != len(direct_links):
        raise LabError("linked sync root repeats a child repository")
    if any(
        link.parent_repository in children or link.child_repository == root_name
        for link in baseline.repository_links
        if link.parent_repository != root_name
    ):
        raise LabError("M3c supports direct gitlinks only; nested/overlapping links refused")
    effective = validated.get("effective_repositories")
    if not isinstance(effective, list) or not all(
        isinstance(item, str) for item in effective
    ):
        raise LabError("linked sync effective repository set is invalid")
    expected_effective = {root_name, *children}
    if set(effective) != expected_effective:
        raise LabError("linked sync effective repository set is incomplete")
    if manager.project.checkout_modes.get(root_name) != "stable_branch" or any(
        manager.project.checkout_modes.get(child) != "pinned_gitlink"
        for child in children
    ):
        raise LabError("linked sync requires one stable parent and pinned children")
    raw_repositories = validated.get("repositories")
    raw_links = validated.get("repository_links")
    if not isinstance(raw_repositories, dict) or not isinstance(raw_links, list):
        raise LabError("project sync plan lacks linked repository facts")
    root_inspection = raw_repositories.get(root_name)
    if not isinstance(root_inspection, dict):
        raise LabError("project sync plan lacks linked root inspection")
    root_path = Path(str(root_inspection.get("management_path", "")))
    root_status = git.status_short(root_path)
    allowed_link_paths = {link.submodule_path for link in direct_links}
    stable_plan = plan_stable_checkout_sync(
        root_path,
        expected_identity=str(root_inspection.get("canonical_identity_expected", "")),
        branch=str(root_inspection.get("stable_branch", "")),
        remote_ref=str(root_inspection.get("remote_ref", "")),
    )
    root_action = stable_plan.get("action")
    if root_action == "BLOCKED_DIRTY":
        if (
            stable_plan.get("local_commit") != stable_plan.get("remote_commit")
            or _status_paths(root_status) - allowed_link_paths
        ):
            raise LabError("linked root has unrelated dirty state")
        root_action = "HEAD_AT_TARGET_WITH_GITLINK_DRIFT"
    elif root_action not in {"FAST_FORWARD_READY", "NOOP"}:
        raise LabError(f"linked root is not safely updateable: {root_action}")

    link_facts = {
        (item.get("parent_repository"), item.get("child_repository")): item
        for item in raw_links
        if isinstance(item, dict)
    }
    child_plans: dict[str, object] = {}
    child_updates_needed = False
    for link in sorted(direct_links, key=lambda item: item.child_repository):
        child_name = link.child_repository
        inspection = raw_repositories.get(child_name)
        readiness = link_facts.get((root_name, child_name))
        if not isinstance(inspection, dict) or not isinstance(readiness, dict):
            raise LabError(f"linked sync lacks child facts: {child_name}")
        if readiness.get("state") != "READY":
            raise LabError(f"linked child target object is unavailable: {child_name}")
        child_path = Path(str(inspection.get("management_path", "")))
        expected_path = (root_path / link.submodule_path).resolve()
        if child_path.resolve() != expected_path:
            raise LabError(f"linked child checkout path mismatch: {child_name}")
        if git.repository_root(child_path) != child_path.resolve():
            raise LabError(f"linked child is not an exact repository root: {child_name}")
        expected_identity = str(inspection.get("canonical_identity_expected", ""))
        if git.origin_identity(child_path) != expected_identity:
            raise LabError(f"linked child canonical identity mismatch: {child_name}")
        child_status = git.status_short(child_path)
        if child_status:
            raise LabError(f"linked child checkout is dirty: {child_name}")
        current_commit = git.current_commit(child_path)
        current_tree = git.tree_oid(child_path)
        target_commit = str(readiness.get("gitlink_commit", ""))
        if git.resolve_commit(child_path, target_commit) != target_commit:
            raise LabError(f"linked child target commit cannot be resolved: {child_name}")
        target_tree = git.tree_oid(child_path, target_commit)
        if current_commit != target_commit:
            child_updates_needed = True
        child_plans[child_name] = {
            "repository": child_name,
            "canonical_identity": expected_identity,
            "checkout_path": str(child_path),
            "stable_branch": manager.project.stable_branches[child_name],
            "submodule_path": link.submodule_path,
            "current_commit": current_commit,
            "current_tree_oid": current_tree,
            "checkout_status": list(child_status),
            "target_commit": target_commit,
            "target_tree_oid": target_tree,
            "object_available": True,
        }
    action = (
        "NOOP"
        if root_action == "NOOP" and not child_updates_needed and not root_status
        else "APPLY_READY"
    )
    root_plan: dict[str, object] = {
        "repository": root_name,
        "canonical_identity": root_inspection.get("canonical_identity_expected"),
        "checkout_path": str(root_path),
        "stable_branch": root_inspection.get("stable_branch"),
        "remote_ref": root_inspection.get("remote_ref"),
        "current_commit": git.current_commit(root_path),
        "current_tree_oid": git.tree_oid(root_path),
        "checkout_status": list(root_status),
        "target_commit": stable_plan.get("remote_commit"),
        "target_tree_oid": stable_plan.get("remote_tree_oid"),
        "head_action": root_action,
        "stable_plan": stable_plan,
    }
    content: dict[str, object] = {
        "schema": LINKED_SYNC_PLAN_SCHEMA_ID,
        "project_id": manager.project.project_id,
        "project_manifest_sha256": manager.project.sha256,
        "baseline_id": baseline.baseline_id,
        "baseline_path": str(baseline.path),
        "baseline_sha256": baseline.sha256,
        "project_plan_sha256": validated["plan_sha256"],
        "root_repository": root_name,
        "root": root_plan,
        "children": child_plans,
        "action": action,
        "network_used": False,
        "worktree_files_copied": False,
        "rebase_reset_clean_permitted": False,
        "submodule_fetch_permitted": False,
        "lfs_smudge_enabled": False,
    }
    return {**content, "plan_sha256": _hash(content)}


def _checkout_facts(path: Path) -> dict[str, object]:
    return {
        "commit": git.current_commit(path),
        "tree_oid": git.tree_oid(path),
        "branch": git.current_branch(path),
        "status": list(git.status_short(path)),
    }


def _safe_checkout_facts(path: Path) -> dict[str, object]:
    try:
        return _checkout_facts(path)
    except (git.GitError, OSError) as exc:
        return {
            "inspection_error": str(exc),
            "checkout_path": str(path),
        }


def _persist_outcome(
    manager: WorkspaceManager,
    *,
    plan: dict[str, object],
    reservation_id: str,
    status: str,
    reservations: dict[str, object],
    inventories: dict[str, object],
    releases: dict[str, object],
    root_update: dict[str, object] | None,
    child_updates: dict[str, object],
    final_facts: dict[str, object],
    error: str | None,
) -> dict[str, object]:
    completed_at = datetime.now(timezone.utc)
    sync_id = (
        f"LINK-{completed_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{str(plan['plan_sha256'])[:8]}"
    )
    receipt_path = manager.state_root / "project-sync" / f"{sync_id}.json"
    content: dict[str, object] = {
        "schema": LINKED_SYNC_RECEIPT_SCHEMA_ID,
        "sync_id": sync_id,
        "completed_at": completed_at.isoformat(),
        "receipt_path": str(receipt_path),
        "project_id": manager.project.project_id,
        "linked_plan_sha256": plan["plan_sha256"],
        "project_plan_sha256": plan["project_plan_sha256"],
        "root_repository": plan["root_repository"],
        "reservation_id": reservation_id,
        "reservations": reservations,
        "runtime_inventories": inventories,
        "released_reservations": releases,
        "root_update": root_update,
        "child_updates": child_updates,
        "final_facts": final_facts,
        "status": status,
        "error": error,
        "recovery_required": status == "QUARANTINED",
        "network_used": False,
        "worktree_files_copied": False,
        "task_worktrees_updated": False,
        "submodule_fetch_used": False,
        "lfs_smudge_enabled": False,
        "rebase_reset_clean_used": False,
        "processes_signalled": False,
        "device_access": False,
        "executor_pid": os.getpid(),
    }
    receipt = {**content, "receipt_sha256": _hash(content)}
    persist_project_sync_receipt(manager, receipt)
    return receipt


def apply_linked_project_checkout(
    manager: WorkspaceManager,
    plan: object,
    *,
    confirmed_plan_sha256: str,
    reservation_id: str,
    inventory_collector=collect_checkout_runtime_inventory,
) -> dict[str, object]:
    """Apply a direct parent/gitlink-child set with quarantine on partial mutation."""

    validated = validate_linked_sync_plan(plan)
    validate_task_id(reservation_id)
    if validated.get("plan_sha256") != confirmed_plan_sha256:
        raise LabError("confirmed linked sync plan hash does not match")
    if (
        validated.get("project_id") != manager.project.project_id
        or validated.get("project_manifest_sha256") != manager.project.sha256
    ):
        raise LabError("linked sync plan does not match the active project manifest")
    baseline = load_baseline_lock(Path(str(validated.get("baseline_path", ""))))
    if (
        baseline.baseline_id != validated.get("baseline_id")
        or baseline.sha256 != validated.get("baseline_sha256")
    ):
        raise LabError("linked sync plan does not match the active baseline lock")
    root_raw = validated.get("root")
    children_raw = validated.get("children")
    if not isinstance(root_raw, dict) or not isinstance(children_raw, dict):
        raise LabError("linked sync plan lacks checkout identities")
    if validated.get("action") == "NOOP":
        fresh_project = plan_project_sync(
            manager,
            baseline_path=Path(str(validated["baseline_path"])),
            repository_names=(str(validated["root_repository"]),),
        )
        fresh_linked = plan_linked_project_checkout(
            manager,
            fresh_project,
            root_repository=str(validated["root_repository"]),
        )
        if fresh_linked != validated:
            raise LabError("linked sync state changed before no-op receipt")
        return _persist_outcome(
            manager,
            plan=validated,
            reservation_id=reservation_id,
            status="UP_TO_DATE",
            reservations={},
            inventories={},
            releases={},
            root_update=None,
            child_updates={},
            final_facts={
                str(validated["root_repository"]): _checkout_facts(
                    Path(str(root_raw["checkout_path"]))
                ),
                **{
                    name: _checkout_facts(Path(str(raw["checkout_path"])))
                    for name, raw in children_raw.items()
                    if isinstance(name, str) and isinstance(raw, dict)
                },
            },
            error=None,
        )
    if validated.get("action") != "APPLY_READY":
        raise LabError(f"linked sync plan is not applicable: {validated.get('action')}")

    targets: dict[str, StableCheckoutTarget] = {}
    checkout_plans = {
        str(validated["root_repository"]): root_raw,
        **children_raw,
    }
    for name, raw in checkout_plans.items():
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise LabError("linked sync plan has an invalid checkout")
        targets[name] = _target(
            manager,
            repository=name,
            path=Path(str(raw["checkout_path"])),
            canonical_identity=str(raw["canonical_identity"]),
            branch=str(raw["stable_branch"]),
        )

    reservations: dict[str, object] = {}
    releases: dict[str, object] = {}
    inventories: dict[str, object] = {}
    acquired_names: list[str] = []
    try:
        for index, name in enumerate(sorted(targets)):
            raw = checkout_plans[name]
            if not isinstance(raw, dict):
                raise LabError("linked sync plan has an invalid checkout binding")
            receipt = acquire_checkout_maintenance_reservation(
                targets[name],
                operation_plan_sha256=str(validated["plan_sha256"]),
                local_commit=str(raw["current_commit"]),
                local_tree_oid=str(raw["current_tree_oid"]),
                expected_status=tuple(str(item) for item in raw["checkout_status"]),
                reservation_id=(
                    f"MAINT-{_hash([reservation_id, name])[:16]}-{index + 1}"
                ),
            )
            reservations[name] = receipt
            acquired_names.append(name)
        for name in sorted(targets):
            inventory = inventory_collector(targets[name].checkout_path)
            validated_inventory = validate_checkout_runtime_inventory(
                inventory,
                checkout_path=targets[name].checkout_path,
            )
            if validated_inventory.get("dependencies") != []:
                raise LabError(f"linked checkout has active processes: {name}")
            inventories[name] = validated_inventory

        baseline_path = Path(str(validated["baseline_path"]))
        fresh_project = plan_project_sync(
            manager,
            baseline_path=baseline_path,
            repository_names=(str(validated["root_repository"]),),
        )
        fresh_linked = plan_linked_project_checkout(
            manager,
            fresh_project,
            root_repository=str(validated["root_repository"]),
        )
        if fresh_linked != validated:
            raise LabError("linked sync state changed after maintenance observation")
    except Exception:
        for name in reversed(acquired_names):
            releases[name] = release_maintenance_reservation(
                targets[name],
                active_receipt=reservations[name],
            )
        raise

    root_update: dict[str, object] | None = None
    child_updates: dict[str, object] = {}
    mutation_error: str | None = None
    try:
        root_action = root_raw.get("head_action")
        if root_action == "FAST_FORWARD_READY":
            root_update = apply_stable_checkout_head_fast_forward(
                root_raw.get("stable_plan")
            )
        elif root_action not in {"NOOP", "HEAD_AT_TARGET_WITH_GITLINK_DRIFT"}:
            raise LabError(f"linked root action changed: {root_action}")
        for name in sorted(children_raw):
            raw = children_raw[name]
            if not isinstance(raw, dict):
                raise LabError("linked child plan is invalid")
            path = Path(str(raw["checkout_path"]))
            before = _checkout_facts(path)
            target_commit = str(raw["target_commit"])
            if before["commit"] != target_commit:
                git.run_git(
                    path,
                    ["checkout", "--detach", "--no-recurse-submodules", target_commit],
                    environment={
                        "GIT_LFS_SKIP_SMUDGE": "1",
                        "GIT_TERMINAL_PROMPT": "0",
                    },
                )
            after = _checkout_facts(path)
            if (
                after["commit"] != target_commit
                or after["tree_oid"] != raw.get("target_tree_oid")
                or after["status"] != []
            ):
                raise LabError(f"linked child final identity mismatch: {name}")
            child_updates[name] = {
                "before": before,
                "after": after,
                "target_commit": target_commit,
                "checkout_updated": before["commit"] != target_commit,
                "network_used": False,
                "submodule_fetch_used": False,
                "lfs_smudge_enabled": False,
            }
        root_path = Path(str(root_raw["checkout_path"]))
        root_final = _checkout_facts(root_path)
        if (
            root_final["commit"] != root_raw.get("target_commit")
            or root_final["tree_oid"] != root_raw.get("target_tree_oid")
            or root_final["branch"] != root_raw.get("stable_branch")
            or root_final["status"] != []
        ):
            raise LabError("linked root final identity or clean status mismatch")
        final_gitlinks = git.submodule_gitlinks(root_path, str(root_final["commit"]))
        for name, raw in children_raw.items():
            if not isinstance(raw, dict) or final_gitlinks.get(
                str(raw.get("submodule_path"))
            ) != raw.get("target_commit"):
                raise LabError(f"linked root final gitlink mismatch: {name}")
    except (
        LabError,
        git.GitError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        KeyboardInterrupt,
    ) as exc:
        mutation_error = str(exc)
    finally:
        for name in reversed(acquired_names):
            try:
                releases[name] = release_maintenance_reservation(
                    targets[name],
                    active_receipt=reservations[name],
                )
            except (LabError, OSError) as exc:
                mutation_error = (
                    f"{mutation_error}; release failed for {name}: {exc}"
                    if mutation_error
                    else f"release failed for {name}: {exc}"
                )
    final_facts: dict[str, object] = {
        name: _safe_checkout_facts(target.checkout_path)
        for name, target in sorted(targets.items())
    }
    status = "COMPLETE" if mutation_error is None else "QUARANTINED"
    return _persist_outcome(
        manager,
        plan=validated,
        reservation_id=reservation_id,
        status=status,
        reservations=reservations,
        inventories=inventories,
        releases=releases,
        root_update=root_update,
        child_updates=child_updates,
        final_facts=final_facts,
        error=mutation_error,
    )

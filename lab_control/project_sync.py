from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .baseline import BaselineLock, load_baseline_lock
from .stable_sync import plan_stable_checkout_sync
from .workspace import (
    LabError,
    WorkspaceManager,
    validate_repository_name,
    validate_task_id,
)
from .writer_lease import WriterLeaseManager

PROJECT_SYNC_PLAN_SCHEMA_ID = "lab.project-sync-plan.v1"
PROJECT_SYNC_FETCH_SCHEMA_ID = "lab.project-sync-fetch.v1"
METADATA_FETCH_RECEIPT_SCHEMA_ID = "lab.metadata-fetch-receipt.v1"

FetchRunner = Callable[
    [list[str], Mapping[str, str], int],
    subprocess.CompletedProcess[str],
]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _hash_document(content: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(content)).hexdigest()


def _default_fetch_runner(
    argv: list[str],
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=dict(environment),
    )


def _canonical_repository_names(
    manager: WorkspaceManager,
    repository_names: Sequence[str],
) -> tuple[str, ...]:
    if not repository_names:
        return tuple(sorted(manager.catalog))
    selected: list[str] = []
    for raw_name in repository_names:
        validate_repository_name(raw_name)
        name = manager.repository_aliases.get(raw_name.casefold(), raw_name)
        if name not in manager.catalog:
            available = ", ".join(sorted(manager.catalog))
            raise LabError(
                f"unknown sync repository {raw_name!r}; available: {available}"
            )
        if name in selected:
            raise LabError(f"duplicate sync repository: {name}")
        selected.append(name)
    return tuple(selected)


def _effective_repositories(
    baseline: BaselineLock,
    requested: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    effective = set(requested)
    changed = True
    while changed:
        changed = False
        for link in baseline.repository_links:
            if (
                link.parent_repository in effective
                and link.child_repository not in effective
            ):
                effective.add(link.child_repository)
                changed = True

    children: dict[str, set[str]] = {}
    for link in baseline.repository_links:
        if (
            link.parent_repository in effective
            and link.child_repository in effective
        ):
            children.setdefault(link.parent_repository, set()).add(
                link.child_repository
            )
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visiting:
            raise LabError(f"sync repository links contain a cycle at: {name}")
        if name in visited:
            return
        visiting.add(name)
        for child in sorted(children.get(name, set())):
            visit(child)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in sorted(effective):
        visit(name)
    added_dependencies = tuple(sorted(effective - set(requested)))
    return tuple(ordered), added_dependencies


def _active_writers(
    manager: WorkspaceManager,
    repository: str,
) -> list[dict[str, object]]:
    writers: list[dict[str, object]] = []
    writer_manager = WriterLeaseManager(manager)
    if not manager.worktrees_root.is_dir():
        return writers
    for manifest_path in sorted(manager.worktrees_root.glob("*/task.yaml")):
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise LabError(f"unsafe Task manifest during sync inspection: {manifest_path}")
        task_id = manifest_path.parent.name
        manifest = manager.load_task(task_id)
        raw_repositories = manifest.get("repositories")
        if not isinstance(raw_repositories, dict):
            raise LabError(f"Task manifest has no repositories: {task_id}")
        raw_repository = raw_repositories.get(repository)
        if raw_repository is None:
            continue
        if not isinstance(raw_repository, dict):
            raise LabError(f"Task has invalid repository writer: {task_id}/{repository}")
        status = writer_manager.inspect_repository(
            task_id,
            repository,
            manifest=manifest,
        )
        if status.get("state") != "ACTIVE":
            continue
        lease = status.get("lease")
        if not isinstance(lease, dict):
            raise LabError(f"active writer lease is missing: {task_id}/{repository}")
        writers.append(
            {
                "task_id": task_id,
                "branch": raw_repository.get("branch"),
                "host": lease.get("host"),
                "epoch": lease.get("writer_epoch"),
                "actor": lease.get("actor"),
                "lease_id": lease.get("lease_id"),
                "expires_at": lease.get("expires_at"),
                "worktree_path": raw_repository.get("worktree_path"),
            }
        )
    return writers


def _inspect_repository(
    manager: WorkspaceManager,
    baseline: BaselineLock,
    name: str,
) -> dict[str, object]:
    pin = baseline.repositories.get(name)
    if pin is None:
        raise LabError(f"sync baseline has no repository pin: {name}")
    configured = manager.catalog.get(name)
    if configured is None:
        raise LabError(f"sync project has no repository path: {name}")
    stable_branch = manager.project.stable_branches[name]
    checkout_mode = manager.project.checkout_modes[name]
    remote_ref = f"refs/remotes/origin/{stable_branch}"
    inspection: dict[str, object] = {
        "repository": name,
        "configured_path": str(configured),
        "canonical_identity_expected": pin.canonical_identity,
        "stable_branch": stable_branch,
        "checkout_mode": checkout_mode,
        "remote_ref": remote_ref,
        "active_writers": _active_writers(manager, name),
        "fetch_action": "BLOCKED",
        "checkout_sync_action": "BLOCKED_REPOSITORY_UNAVAILABLE",
    }
    try:
        root = git.repository_root(configured)
        if root != configured.resolve():
            raise LabError(
                f"configured path is not an exact repository root: {configured} -> {root}"
            )
        identity = git.origin_identity(root)
        inspection["management_path"] = str(root)
        inspection["canonical_identity_observed"] = identity
        if identity != pin.canonical_identity:
            inspection["blocker"] = "canonical repository identity mismatch"
            inspection["checkout_sync_action"] = "BLOCKED_IDENTITY_MISMATCH"
            return inspection
        local_commit = git.current_commit(root)
        local_tree = git.tree_oid(root, local_commit)
        baseline_available = False
        baseline_tree_matches = False
        try:
            baseline_commit = git.resolve_commit(root, pin.commit)
            baseline_available = baseline_commit == pin.commit
            baseline_tree_matches = git.tree_oid(root, baseline_commit) == pin.tree_oid
        except git.GitError:
            pass
        inspection.update(
            {
                "branch": git.current_branch(root),
                "local_commit": local_commit,
                "local_tree_oid": local_tree,
                "dirty": bool(git.status_short(root)),
                "baseline_commit": pin.commit,
                "baseline_tree_oid": pin.tree_oid,
                "baseline_object_available": baseline_available,
                "baseline_tree_matches": baseline_tree_matches,
                "remote_commit": (
                    git.resolve_commit(root, remote_ref)
                    if git.ref_exists(root, remote_ref)
                    else None
                ),
                "fetch_action": "FETCH_READY",
            }
        )
        if baseline_available and not baseline_tree_matches:
            inspection["fetch_action"] = "BLOCKED_BASELINE_TREE_MISMATCH"
            inspection["blocker"] = "baseline commit resolves to the wrong tree"
        if checkout_mode == "pinned_gitlink":
            if inspection["dirty"]:
                inspection["checkout_sync_action"] = "BLOCKED_DIRTY_GITLINK"
            elif local_commit != pin.commit or local_tree != pin.tree_oid:
                inspection["checkout_sync_action"] = "BLOCKED_GITLINK_DRIFT"
            else:
                inspection["checkout_sync_action"] = "PINNED_GITLINK"
        else:
            stable_plan = plan_stable_checkout_sync(
                root,
                expected_identity=pin.canonical_identity,
                branch=stable_branch,
                remote_ref=remote_ref,
            )
            inspection["checkout_sync_action"] = stable_plan["action"]
            inspection["checkout_sync_plan_sha256"] = stable_plan["plan_sha256"]
    except (git.GitError, LabError) as exc:
        inspection["blocker"] = str(exc)
    return inspection


def _link_readiness(
    manager: WorkspaceManager,
    baseline: BaselineLock,
    selected: set[str],
    repositories: Mapping[str, object],
) -> list[dict[str, object]]:
    readiness: list[dict[str, object]] = []
    for link in baseline.repository_links:
        if link.parent_repository not in selected:
            continue
        parent = repositories.get(link.parent_repository)
        child = repositories.get(link.child_repository)
        if not isinstance(parent, dict) or not isinstance(child, dict):
            raise LabError("sync link references an unresolved repository")
        parent_path_value = parent.get("management_path")
        child_path_value = child.get("management_path")
        remote_parent = parent.get("remote_commit")
        target_commit: str | None = None
        state = "BLOCKED_PARENT_REMOTE_REF_MISSING"
        if (
            isinstance(parent_path_value, str)
            and isinstance(remote_parent, str)
        ):
            try:
                target_commit = git.submodule_gitlinks(
                    Path(parent_path_value), remote_parent
                ).get(link.submodule_path)
            except git.GitError:
                target_commit = None
            state = (
                "BLOCKED_PARENT_GITLINK_MISSING"
                if target_commit is None
                else "BLOCKED_CHILD_OBJECT_MISSING"
            )
        child_object_available = False
        if isinstance(child_path_value, str) and target_commit is not None:
            try:
                child_object_available = (
                    git.resolve_commit(Path(child_path_value), target_commit)
                    == target_commit
                )
            except git.GitError:
                pass
            if child_object_available:
                state = "READY"
        readiness.append(
            {
                "parent_repository": link.parent_repository,
                "submodule_path": link.submodule_path,
                "child_repository": link.child_repository,
                "remote_parent_commit": remote_parent,
                "gitlink_commit": target_commit,
                "child_object_available": child_object_available,
                "state": state,
            }
        )
    return readiness


def plan_project_sync(
    manager: WorkspaceManager,
    *,
    baseline_path: Path,
    repository_names: Sequence[str] = (),
) -> dict[str, object]:
    """Inspect multi-repository sync facts without network or repository writes."""

    baseline = load_baseline_lock(baseline_path)
    requested = _canonical_repository_names(manager, repository_names)
    missing = sorted(set(requested) - set(baseline.repositories))
    if missing:
        raise LabError("sync baseline lacks repositories: " + ", ".join(missing))
    ordered, added_dependencies = _effective_repositories(baseline, requested)
    repositories = {
        name: _inspect_repository(manager, baseline, name) for name in ordered
    }
    links = _link_readiness(manager, baseline, set(ordered), repositories)
    fetch_ready = all(
        isinstance(repository, dict)
        and repository.get("fetch_action") == "FETCH_READY"
        for repository in repositories.values()
    )
    checkout_actions = [
        str(repository.get("checkout_sync_action"))
        for repository in repositories.values()
        if isinstance(repository, dict)
    ]
    settled_actions = {"NOOP", "PINNED_GITLINK"}
    if all(action in settled_actions for action in checkout_actions):
        checkout_status = "UP_TO_DATE"
    elif all(
        action in settled_actions | {"FAST_FORWARD_READY"}
        for action in checkout_actions
    ):
        checkout_status = "FAST_FORWARD_AVAILABLE"
    else:
        checkout_status = "BLOCKED"
    content: dict[str, object] = {
        "schema": PROJECT_SYNC_PLAN_SCHEMA_ID,
        "project_id": manager.project.project_id,
        "project_manifest_sha256": manager.project.sha256,
        "baseline_id": baseline.baseline_id,
        "baseline_path": str(baseline.path),
        "baseline_sha256": baseline.sha256,
        "requested_repositories": list(requested),
        "effective_repositories": list(ordered),
        "dependency_repositories_added": list(added_dependencies),
        "fetch_order": list(ordered),
        "repositories": repositories,
        "repository_links": links,
        "fetch_status": "READY" if fetch_ready else "BLOCKED",
        "checkout_sync_status": checkout_status,
        "network_used": False,
        "checkout_updated": False,
        "task_worktrees_updated": False,
        "worktree_files_copied": False,
        "lfs_smudge_enabled": False,
        "submodules_fetched": False,
        "rebase_reset_clean_permitted": False,
    }
    return {**content, "plan_sha256": _hash_document(content)}


def validate_project_sync_plan(plan: object) -> dict[str, object]:
    if not isinstance(plan, dict) or plan.get("schema") != PROJECT_SYNC_PLAN_SCHEMA_ID:
        raise LabError("project sync plan is invalid")
    supplied_hash = plan.get("plan_sha256")
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if not isinstance(supplied_hash, str) or supplied_hash != _hash_document(content):
        raise LabError("project sync plan hash mismatch")
    if plan.get("fetch_status") != "READY":
        raise LabError("project sync fetch is blocked by the inspected repository state")
    return plan


def _persist_fetch_receipt(
    manager: WorkspaceManager,
    receipt: Mapping[str, object],
) -> None:
    sync_id = validate_task_id(str(receipt.get("sync_id", "")))
    expected_path = manager.state_root / "project-sync" / f"{sync_id}.json"
    configured_path = Path(str(receipt.get("receipt_path", ""))).resolve()
    if configured_path != expected_path.resolve():
        raise LabError("project sync receipt path identity mismatch")
    state_root = manager.state_root
    if state_root.is_symlink():
        raise LabError(f"project sync state root is a symlink: {state_root}")
    state_root.mkdir(parents=True, exist_ok=True)
    receipt_root = expected_path.parent
    if receipt_root.is_symlink():
        raise LabError(f"project sync receipt root is a symlink: {receipt_root}")
    receipt_root.mkdir(parents=True, exist_ok=True)
    if not receipt_root.resolve().is_relative_to(state_root.resolve()):
        raise LabError("project sync receipt root escapes configured state")
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with manager.advisory_locks("project-sync", [sync_id]):
        if expected_path.is_symlink():
            raise LabError(f"project sync receipt path is a symlink: {expected_path}")
        if expected_path.is_file():
            if expected_path.read_text(encoding="utf-8") != encoded:
                raise LabError(f"project sync receipt identity collision: {sync_id}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            dir=receipt_root,
            prefix=f".{sync_id}.",
            suffix=".tmp",
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, expected_path)
        finally:
            temporary_path.unlink(missing_ok=True)


def _fetch_metadata(
    repository: Path,
    *,
    canonical_identity: str,
    branch: str,
    timeout_seconds: int,
    fetch_runner: FetchRunner,
) -> dict[str, object]:
    root = git.repository_root(repository)
    if root != repository.resolve():
        raise LabError(f"metadata fetch requires an exact repository root: {repository}")
    if git.origin_identity(root) != canonical_identity:
        raise LabError("metadata fetch canonical repository identity mismatch")
    before = {
        "branch": git.current_branch(root),
        "commit": git.current_commit(root),
        "tree_oid": git.tree_oid(root),
        "status": list(git.status_short(root)),
    }
    remote_ref = f"refs/remotes/origin/{branch}"
    remote_before = (
        git.resolve_commit(root, remote_ref) if git.ref_exists(root, remote_ref) else None
    )
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    argv = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "gc.auto=0",
        "-C",
        str(root),
        "fetch",
        "--no-tags",
        "--no-recurse-submodules",
        "--no-write-fetch-head",
        "origin",
        f"refs/heads/{branch}:{remote_ref}",
    ]
    try:
        completed = fetch_runner(argv, environment, timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise LabError(f"metadata fetch failed: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())[:1000]
        raise LabError(f"metadata fetch failed: {detail or completed.returncode}")
    after = {
        "branch": git.current_branch(root),
        "commit": git.current_commit(root),
        "tree_oid": git.tree_oid(root),
        "status": list(git.status_short(root)),
    }
    if after != before:
        raise LabError("metadata fetch unexpectedly changed the management checkout")
    if not git.ref_exists(root, remote_ref):
        raise LabError("metadata fetch did not create the protected remote ref")
    remote_after = git.resolve_commit(root, remote_ref)
    content: dict[str, object] = {
        "schema": METADATA_FETCH_RECEIPT_SCHEMA_ID,
        "canonical_identity": canonical_identity,
        "repository": str(root),
        "branch": branch,
        "remote_ref": remote_ref,
        "remote_commit_before": remote_before,
        "remote_commit_after": remote_after,
        "checkout_before": before,
        "checkout_after": after,
        "network_used": True,
        "checkout_updated": False,
        "task_worktrees_updated": False,
        "worktree_files_copied": False,
        "submodules_fetched": False,
        "lfs_smudge_enabled": False,
        "large_assets_downloaded": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return {**content, "receipt_sha256": _hash_document(content)}


def fetch_project_sync(
    manager: WorkspaceManager,
    plan: object,
    *,
    timeout_seconds: int = 120,
    fetch_runner: FetchRunner = _default_fetch_runner,
) -> dict[str, object]:
    """Fetch exact stable refs for a plan; never update any checkout."""

    validated = validate_project_sync_plan(plan)
    if not 1 <= timeout_seconds <= 300:
        raise LabError("project sync fetch timeout must be between 1 and 300 seconds")
    if validated.get("project_manifest_sha256") != manager.project.sha256:
        raise LabError("project sync manifest changed after planning")
    baseline_path = Path(str(validated.get("baseline_path", "")))
    baseline = load_baseline_lock(baseline_path)
    if baseline.sha256 != validated.get("baseline_sha256"):
        raise LabError("project sync baseline changed after planning")
    raw_repositories = validated.get("repositories")
    raw_order = validated.get("fetch_order")
    if not isinstance(raw_repositories, dict) or not isinstance(raw_order, list):
        raise LabError("project sync plan lacks repositories")
    receipts: dict[str, object] = {}
    failures: dict[str, str] = {}
    for name_value in raw_order:
        if not isinstance(name_value, str):
            raise LabError("project sync plan has an invalid fetch order")
        name = validate_repository_name(name_value)
        raw_repository = raw_repositories.get(name)
        if not isinstance(raw_repository, dict):
            raise LabError(f"project sync plan lacks repository: {name}")
        try:
            receipts[name] = _fetch_metadata(
                Path(str(raw_repository.get("management_path", ""))),
                canonical_identity=str(
                    raw_repository.get("canonical_identity_expected", "")
                ),
                branch=str(raw_repository.get("stable_branch", "")),
                timeout_seconds=timeout_seconds,
                fetch_runner=fetch_runner,
            )
        except LabError as exc:
            failures[name] = str(exc)
    post_plan = plan_project_sync(
        manager,
        baseline_path=baseline.path,
        repository_names=tuple(str(name) for name in raw_order),
    )
    status = "COMPLETE" if not failures else "PARTIAL"
    completed_at = datetime.now(timezone.utc)
    sync_id = (
        f"SYNC-{completed_at.strftime('%Y%m%dT%H%M%S%fZ')}-"
        f"{str(validated['plan_sha256'])[:8]}"
    )
    receipt_path = manager.state_root / "project-sync" / f"{sync_id}.json"
    content: dict[str, object] = {
        "schema": PROJECT_SYNC_FETCH_SCHEMA_ID,
        "sync_id": sync_id,
        "completed_at": completed_at.isoformat(),
        "receipt_path": str(receipt_path),
        "project_id": manager.project.project_id,
        "plan_sha256": validated["plan_sha256"],
        "fetch_order": list(raw_order),
        "receipts": receipts,
        "failures": failures,
        "post_fetch_plan": post_plan,
        "status": status,
        "network_used": True,
        "checkout_updated": False,
        "task_worktrees_updated": False,
        "worktree_files_copied": False,
        "submodules_fetched": False,
        "lfs_smudge_enabled": False,
        "large_assets_downloaded": False,
    }
    receipt = {**content, "receipt_sha256": _hash_document(content)}
    _persist_fetch_receipt(manager, receipt)
    return receipt

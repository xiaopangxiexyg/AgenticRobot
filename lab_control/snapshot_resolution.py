from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import git
from .baseline import load_baseline_lock
from .workspace import (
    LabError,
    WorkspaceManager,
    validate_repository_name,
    validate_task_id,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_clean_snapshot_identity(
    workspace_manager: WorkspaceManager,
    *,
    task_id: str,
    profile_id: str,
    profile_sha256: str,
    repositories: tuple[str, ...],
    baseline_path: Path,
) -> tuple[str, dict[str, object]]:
    """Resolve Task overrides plus baseline pins into one exact clean identity."""

    validate_task_id(task_id)
    validate_task_id(profile_id)
    if len(profile_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in profile_sha256
    ):
        raise LabError(f"profile {profile_id} requires a full SHA-256 identity")
    if not repositories or len(repositories) != len(set(repositories)):
        raise LabError(f"profile {profile_id} has invalid repositories")
    for name in repositories:
        validate_repository_name(name)

    inspection = workspace_manager.inspect_task(task_id)
    if inspection.get("status") != "ACTIVE":
        raise LabError(f"task {task_id} has branch or scope drift")
    inspected_repositories = inspection.get("repositories")
    if not isinstance(inspected_repositories, dict):
        raise LabError(f"task {task_id} inspection has no repositories")
    task_manifest = workspace_manager.load_task(task_id)
    manifest_repositories = task_manifest.get("repositories")
    if not isinstance(manifest_repositories, dict):
        raise LabError(f"task manifest has no repositories: {task_id}")

    baseline = load_baseline_lock(baseline_path) if baseline_path.is_file() else None
    baseline_used = False
    snapshot_repositories: dict[str, dict[str, object]] = {}
    for name in repositories:
        raw_repository = inspected_repositories.get(name)
        if isinstance(raw_repository, dict):
            if raw_repository.get("dirty"):
                raise LabError(
                    f"clean profile {profile_id} rejects dirty repository: {name}"
                )
            commit = str(raw_repository.get("commit", ""))
            task_path = Path(str(raw_repository.get("path", "")))
            raw_manifest_repository = manifest_repositories.get(name)
            if not isinstance(raw_manifest_repository, dict):
                raise LabError(f"task manifest repository missing: {name}")
            try:
                tree = git.tree_oid(task_path, commit)
                submodules = git.submodule_gitlinks(task_path, commit)
            except git.GitError as exc:
                raise LabError(str(exc)) from exc
            snapshot_repositories[name] = {
                "commit": commit,
                "tree_oid": tree,
                "submodules": submodules,
                "management_path": raw_manifest_repository["management_path"],
                "source": "task_override",
            }
            continue

        if baseline is None:
            baseline = load_baseline_lock(baseline_path)
        pin = baseline.repositories.get(name)
        if pin is None:
            raise LabError(
                f"profile {profile_id} requires {name}, but Task {task_id} does "
                f"not override it and baseline {baseline.baseline_id} has no pin"
            )
        configured_management = workspace_manager.catalog.get(name)
        if configured_management is None:
            raise LabError(f"baseline references unknown repository: {name}")
        try:
            management_path = git.repository_root(configured_management)
            identity = git.origin_identity(management_path)
            commit = git.resolve_commit(management_path, pin.commit)
            tree = git.tree_oid(management_path, commit)
        except git.GitError as exc:
            raise LabError(
                f"baseline repository {name} is unavailable locally: {exc}"
            ) from exc
        if identity != pin.canonical_identity:
            raise LabError(
                f"baseline repository identity mismatch for {name}: "
                f"{identity} != {pin.canonical_identity}"
            )
        if tree != pin.tree_oid:
            raise LabError(
                f"baseline tree mismatch for {name}: {tree} != {pin.tree_oid}"
            )
        snapshot_repositories[name] = {
            "commit": commit,
            "tree_oid": tree,
            "submodules": dict(pin.submodules),
            "management_path": str(management_path),
            "source": "baseline",
            "baseline_id": baseline.baseline_id,
        }
        baseline_used = True

    resolved_links: list[dict[str, str]] = []
    repository_links = baseline.repository_links if baseline is not None else ()
    for link in repository_links:
        if baseline is None:
            raise LabError("repository links require a baseline identity")
        parent = snapshot_repositories.get(link.parent_repository)
        if parent is None or link.child_repository not in repositories:
            continue
        raw_parent_submodules = parent.get("submodules")
        if not isinstance(raw_parent_submodules, dict):
            raise LabError(
                f"resolved parent repository lacks submodules: "
                f"{link.parent_repository}"
            )
        gitlink_commit = raw_parent_submodules.get(link.submodule_path)
        if not isinstance(gitlink_commit, str):
            raise LabError(
                f"resolved parent repository lacks declared gitlink: "
                f"{link.parent_repository}/{link.submodule_path}"
            )
        child = snapshot_repositories.get(link.child_repository)
        if child is not None and child.get("commit") != gitlink_commit:
            if child.get("source") == "task_override":
                raise LabError(
                    f"task repository override differs from parent gitlink: "
                    f"{link.parent_repository}/{link.submodule_path}="
                    f"{gitlink_commit}, {link.child_repository}="
                    f"{child.get('commit')}"
                )
            pin = baseline.repositories[link.child_repository]
            configured_management = workspace_manager.catalog.get(
                link.child_repository
            )
            if configured_management is None:
                raise LabError(
                    f"repository link references unknown local repository: "
                    f"{link.child_repository}"
                )
            try:
                management_path = git.repository_root(configured_management)
                identity = git.origin_identity(management_path)
                derived_commit = git.resolve_commit(
                    management_path,
                    gitlink_commit,
                )
                derived_tree = git.tree_oid(management_path, derived_commit)
                derived_submodules = git.submodule_gitlinks(
                    management_path,
                    derived_commit,
                )
            except git.GitError as exc:
                raise LabError(
                    f"gitlink-derived repository {link.child_repository} is "
                    f"unavailable locally: {exc}"
                ) from exc
            if identity != pin.canonical_identity:
                raise LabError(
                    f"gitlink-derived repository identity mismatch for "
                    f"{link.child_repository}: {identity} != "
                    f"{pin.canonical_identity}"
                )
            child = {
                "commit": derived_commit,
                "tree_oid": derived_tree,
                "submodules": derived_submodules,
                "management_path": str(management_path),
                "source": "parent_gitlink",
                "baseline_id": baseline.baseline_id,
                "parent_repository": link.parent_repository,
                "submodule_path": link.submodule_path,
            }
            snapshot_repositories[link.child_repository] = child
        if child is not None and child.get("commit") != gitlink_commit:
            raise LabError(
                f"resolved repository differs from parent gitlink: "
                f"{link.child_repository}"
            )
        resolved_links.append(
            {
                "parent_repository": link.parent_repository,
                "submodule_path": link.submodule_path,
                "child_repository": link.child_repository,
                "commit": gitlink_commit,
            }
        )
        baseline_used = True

    snapshot_identity: dict[str, object] = {
        "task_id": task_id,
        "task_manifest_sha256": _sha256_file(
            workspace_manager.task_manifest_path(task_id)
        ),
        "profile_id": profile_id,
        "profile_sha256": profile_sha256,
        "repositories": snapshot_repositories,
        "repository_links": resolved_links,
    }
    if baseline_used:
        if baseline is None:
            raise LabError("resolved baseline identity is unavailable")
        snapshot_identity["baseline"] = {
            "baseline_id": baseline.baseline_id,
            "sha256": baseline.sha256,
        }
    return _sha256_json(snapshot_identity)[:24], snapshot_identity

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from . import git
from .baseline import load_baseline_lock
from .workspace import (
    LabError,
    WorkspaceManager,
    validate_repository_name,
    validate_submodule_path,
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
    task_repository_overrides: Mapping[str, Mapping[str, object]] | None = None,
    repository_link_overrides: Sequence[Mapping[str, str]] | None = None,
    snapshot_mode: str = "clean",
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
    if snapshot_mode not in {"clean", "draft"}:
        raise LabError(f"unsupported snapshot mode: {snapshot_mode}")
    overrides = dict(task_repository_overrides or {})
    unknown_overrides = set(overrides) - set(repositories)
    if unknown_overrides:
        raise LabError(
            "snapshot overrides contain unrequested repositories: "
            + ", ".join(sorted(unknown_overrides))
        )

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
            override = overrides.get(name)
            if raw_repository.get("dirty") and override is None:
                raise LabError(
                    f"clean profile {profile_id} rejects dirty repository: {name}"
                )
            task_path = Path(str(raw_repository.get("path", "")))
            raw_manifest_repository = manifest_repositories.get(name)
            if not isinstance(raw_manifest_repository, dict):
                raise LabError(f"task manifest repository missing: {name}")
            if override is None:
                commit = str(raw_repository.get("commit", ""))
                try:
                    tree = git.tree_oid(task_path, commit)
                    submodules = git.submodule_gitlinks(task_path, commit)
                except git.GitError as exc:
                    raise LabError(str(exc)) from exc
                override_metadata: dict[str, object] = {}
            else:
                expected_override = {
                    "commit",
                    "tree_oid",
                    "submodules",
                    "base_commit",
                    "synthetic",
                    "included_untracked",
                    "excluded_untracked",
                }
                if set(override) != expected_override:
                    raise LabError(f"invalid snapshot repository override: {name}")
                commit = str(override.get("commit", ""))
                tree = str(override.get("tree_oid", ""))
                raw_override_submodules = override.get("submodules")
                if not isinstance(raw_override_submodules, dict) or not all(
                    isinstance(path, str) and isinstance(oid, str)
                    for path, oid in raw_override_submodules.items()
                ):
                    raise LabError(f"invalid snapshot override submodules: {name}")
                submodules = dict(raw_override_submodules)
                try:
                    if (
                        git.resolve_commit(task_path, commit) != commit
                        or git.tree_oid(task_path, commit) != tree
                        or git.submodule_gitlinks(task_path, commit) != submodules
                    ):
                        raise LabError(f"snapshot override identity mismatch: {name}")
                except git.GitError as exc:
                    raise LabError(str(exc)) from exc
                override_metadata = {
                    "base_commit": override["base_commit"],
                    "synthetic": override["synthetic"],
                    "included_untracked": override["included_untracked"],
                    "excluded_untracked": override["excluded_untracked"],
                }
            snapshot_repositories[name] = {
                "commit": commit,
                "tree_oid": tree,
                "submodules": submodules,
                "management_path": raw_manifest_repository["management_path"],
                "source": "task_override",
                **override_metadata,
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
    links_from_baseline = repository_link_overrides is None
    if links_from_baseline:
        repository_links = tuple(
            (
                link.parent_repository,
                link.submodule_path,
                link.child_repository,
            )
            for link in (baseline.repository_links if baseline is not None else ())
        )
    else:
        assert repository_link_overrides is not None
        normalized_links: list[tuple[str, str, str]] = []
        for raw_link in repository_link_overrides:
            if set(raw_link) != {
                "parent_repository",
                "submodule_path",
                "child_repository",
            }:
                raise LabError(
                    "snapshot repository link override has an unexpected shape"
                )
            normalized_links.append(
                (
                    validate_repository_name(raw_link["parent_repository"]),
                    validate_submodule_path(raw_link["submodule_path"]),
                    validate_repository_name(raw_link["child_repository"]),
                )
            )
        if len(normalized_links) != len(set(normalized_links)):
            raise LabError("snapshot repository link overrides contain duplicates")
        repository_links = tuple(sorted(normalized_links))
    for parent_name, submodule_path, child_name in repository_links:
        parent = snapshot_repositories.get(parent_name)
        if parent is None or child_name not in repositories:
            continue
        raw_parent_submodules = parent.get("submodules")
        if not isinstance(raw_parent_submodules, dict):
            raise LabError(
                f"resolved parent repository lacks submodules: {parent_name}"
            )
        gitlink_commit = raw_parent_submodules.get(submodule_path)
        if not isinstance(gitlink_commit, str):
            raise LabError(
                f"resolved parent repository lacks declared gitlink: "
                f"{parent_name}/{submodule_path}"
            )
        child = snapshot_repositories.get(child_name)
        if child is not None and child.get("commit") != gitlink_commit:
            if child.get("source") == "task_override":
                raise LabError(
                    f"task repository override differs from parent gitlink: "
                    f"{parent_name}/{submodule_path}="
                    f"{gitlink_commit}, {child_name}="
                    f"{child.get('commit')}"
                )
            if baseline is None:
                raise LabError(
                    f"repository link requires an exact Task override for {child_name}"
                )
            pin = baseline.repositories[child_name]
            configured_management = workspace_manager.catalog.get(child_name)
            if configured_management is None:
                raise LabError(
                    f"repository link references unknown local repository: {child_name}"
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
                    f"gitlink-derived repository {child_name} is "
                    f"unavailable locally: {exc}"
                ) from exc
            if identity != pin.canonical_identity:
                raise LabError(
                    f"gitlink-derived repository identity mismatch for "
                    f"{child_name}: {identity} != "
                    f"{pin.canonical_identity}"
                )
            child = {
                "commit": derived_commit,
                "tree_oid": derived_tree,
                "submodules": derived_submodules,
                "management_path": str(management_path),
                "source": "parent_gitlink",
                "baseline_id": baseline.baseline_id,
                "parent_repository": parent_name,
                "submodule_path": submodule_path,
            }
            snapshot_repositories[child_name] = child
        if child is not None and child.get("commit") != gitlink_commit:
            raise LabError(
                f"resolved repository differs from parent gitlink: {child_name}"
            )
        resolved_links.append(
            {
                "parent_repository": parent_name,
                "submodule_path": submodule_path,
                "child_repository": child_name,
                "commit": gitlink_commit,
            }
        )
        if links_from_baseline:
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
    if snapshot_mode == "draft":
        snapshot_identity["snapshot_mode"] = "draft"
        for raw_repository in snapshot_repositories.values():
            raw_repository.setdefault("base_commit", raw_repository["commit"])
            raw_repository.setdefault("synthetic", False)
            raw_repository.setdefault("included_untracked", [])
            raw_repository.setdefault("excluded_untracked", [])
    if baseline_used:
        if baseline is None:
            raise LabError("resolved baseline identity is unavailable")
        snapshot_identity["baseline"] = {
            "baseline_id": baseline.baseline_id,
            "sha256": baseline.sha256,
        }
    return _sha256_json(snapshot_identity)[:24], snapshot_identity

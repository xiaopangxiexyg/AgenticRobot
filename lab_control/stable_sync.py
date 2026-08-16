from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import git
from .workspace import LabError

STABLE_SYNC_PLAN_SCHEMA_ID = "lab.stable-sync-plan.v1"


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _is_ancestor(repository: Path, older: str, newer: str) -> bool:
    result = git.run_git(
        repository,
        ["merge-base", "--is-ancestor", older, newer],
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    detail = result.stderr.strip() or result.stdout.strip()
    raise LabError(f"cannot compare stable sync commits: {detail}")


def validate_stable_sync_plan(plan: object) -> dict[str, object]:
    """Validate the complete, content-addressed stable-sync plan envelope."""

    if not isinstance(plan, dict) or set(plan) != {
        "schema",
        "canonical_identity",
        "repository",
        "branch",
        "remote_ref",
        "current_branch",
        "local_commit",
        "local_tree_oid",
        "remote_commit",
        "remote_tree_oid",
        "dirty",
        "action",
        "network_used",
        "worktree_files_copied",
        "rebase_reset_clean_permitted",
        "plan_sha256",
    }:
        raise LabError("stable sync plan has an unexpected shape")
    if plan.get("schema") != STABLE_SYNC_PLAN_SCHEMA_ID:
        raise LabError("stable sync plan is invalid")
    supplied_hash = plan.get("plan_sha256")
    content = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if (
        not isinstance(supplied_hash, str)
        or hashlib.sha256(_canonical_json(content)).hexdigest() != supplied_hash
    ):
        raise LabError("stable sync plan hash mismatch")
    if (
        plan.get("network_used") is not False
        or plan.get("worktree_files_copied") is not False
        or plan.get("rebase_reset_clean_permitted") is not False
    ):
        raise LabError("stable sync plan permits an unsafe operation")
    return plan


def plan_stable_checkout_sync(
    repository: Path,
    *,
    expected_identity: str,
    branch: str = "main",
    remote_ref: str = "refs/remotes/origin/main",
) -> dict[str, object]:
    """Plan against already-fetched refs without network or repository writes."""

    requested = repository.expanduser()
    root = requested.resolve()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != root
        or git.repository_root(root) != root
    ):
        raise LabError(f"stable sync requires a real repository root: {root}")
    if git.origin_identity(root) != expected_identity:
        raise LabError("stable sync canonical repository identity mismatch")
    current_branch = git.current_branch(root)
    local_commit = git.current_commit(root)
    local_tree = git.tree_oid(root, local_commit)
    dirty = bool(git.status_short(root))
    remote_exists = git.ref_exists(root, remote_ref)
    remote_commit = git.resolve_commit(root, remote_ref) if remote_exists else None
    remote_tree = git.tree_oid(root, remote_commit) if remote_commit is not None else None

    if current_branch != branch:
        action = "BLOCKED_WRONG_BRANCH"
    elif dirty:
        action = "BLOCKED_DIRTY"
    elif remote_commit is None:
        action = "BLOCKED_REMOTE_REF_MISSING"
    elif local_commit == remote_commit:
        action = "NOOP"
    elif _is_ancestor(root, local_commit, remote_commit):
        action = "FAST_FORWARD_READY"
    elif _is_ancestor(root, remote_commit, local_commit):
        action = "BLOCKED_LOCAL_AHEAD"
    else:
        action = "BLOCKED_DIVERGED"
    content: dict[str, object] = {
        "schema": STABLE_SYNC_PLAN_SCHEMA_ID,
        "canonical_identity": expected_identity,
        "repository": str(root),
        "branch": branch,
        "remote_ref": remote_ref,
        "current_branch": current_branch,
        "local_commit": local_commit,
        "local_tree_oid": local_tree,
        "remote_commit": remote_commit,
        "remote_tree_oid": remote_tree,
        "dirty": dirty,
        "action": action,
        "network_used": False,
        "worktree_files_copied": False,
        "rebase_reset_clean_permitted": False,
    }
    return {**content, "plan_sha256": hashlib.sha256(_canonical_json(content)).hexdigest()}


def apply_stable_checkout_fast_forward(
    plan: object,
) -> dict[str, object]:
    """Apply one unchanged FAST_FORWARD_READY plan; never fetch/rebase/reset/clean."""

    validated = validate_stable_sync_plan(plan)
    supplied_hash = validated["plan_sha256"]
    if validated.get("action") != "FAST_FORWARD_READY":
        raise LabError(
            f"stable sync plan is not applicable: {validated.get('action')}"
        )
    repository_value = validated.get("repository")
    identity = validated.get("canonical_identity")
    branch = validated.get("branch")
    remote_ref = validated.get("remote_ref")
    if not all(isinstance(value, str) and value for value in (
        repository_value,
        identity,
        branch,
        remote_ref,
    )):
        raise LabError("stable sync plan lacks protected identity fields")
    current = plan_stable_checkout_sync(
        Path(str(repository_value)),
        expected_identity=str(identity),
        branch=str(branch),
        remote_ref=str(remote_ref),
    )
    for field in (
        "plan_sha256",
        "local_commit",
        "local_tree_oid",
        "remote_commit",
        "remote_tree_oid",
        "action",
    ):
        if current.get(field) != validated.get(field):
            raise LabError(f"stable sync state changed after planning: {field}")
    repository = Path(str(repository_value))
    git.run_git(repository, ["merge", "--ff-only", str(remote_ref)])
    verified = plan_stable_checkout_sync(
        repository,
        expected_identity=str(identity),
        branch=str(branch),
        remote_ref=str(remote_ref),
    )
    if (
        verified.get("action") != "NOOP"
        or verified.get("local_commit") != validated.get("remote_commit")
        or verified.get("local_tree_oid") != validated.get("remote_tree_oid")
    ):
        raise LabError("stable sync fast-forward failed final identity verification")
    receipt_content: dict[str, object] = {
        "schema": "lab.stable-sync-receipt.v1",
        "plan_sha256": supplied_hash,
        "canonical_identity": identity,
        "repository": repository_value,
        "branch": branch,
        "remote_ref": remote_ref,
        "before_commit": validated["local_commit"],
        "after_commit": verified["local_commit"],
        "after_tree_oid": verified["local_tree_oid"],
        "fast_forwarded": True,
        "network_used": False,
        "worktree_files_copied": False,
        "rebase_reset_clean_used": False,
    }
    return {
        **receipt_content,
        "receipt_sha256": hashlib.sha256(
            _canonical_json(receipt_content)
        ).hexdigest(),
    }

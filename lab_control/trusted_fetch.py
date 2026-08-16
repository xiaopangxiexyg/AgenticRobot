from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .remote_request import OID_PATTERN, SHA256_PATTERN
from .workspace import LabError

TRUSTED_FETCH_RECEIPT_SCHEMA_ID = "lab.trusted-fetch-receipt.v1"
_BRANCH_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}")
FetchRunner = Callable[
    [list[str], Mapping[str, str], int],
    subprocess.CompletedProcess[str],
]


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _safe_branch(branch: str) -> None:
    if (
        not _BRANCH_PATTERN.fullmatch(branch)
        or branch.startswith(("/", "-"))
        or branch.endswith("/")
        or ".." in branch
        or "@{" in branch
        or "//" in branch
    ):
        raise LabError(f"trusted fetch has unsafe branch: {branch}")


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


def fetch_stable_remote_ref(
    repository: Path,
    *,
    expected_identity: str,
    branch: str = "main",
    timeout_seconds: int = 120,
    fetch_runner: FetchRunner = _default_fetch_runner,
) -> dict[str, object]:
    """Fetch one stable branch without checkout, LFS, submodules, or assets."""

    _safe_branch(branch)
    if not 1 <= timeout_seconds <= 300:
        raise LabError("trusted fetch timeout must be between 1 and 300 seconds")
    requested = repository.expanduser()
    root = requested.resolve()
    if (
        not requested.is_absolute()
        or requested.is_symlink()
        or requested.absolute() != root
        or git.repository_root(root) != root
    ):
        raise LabError(f"trusted fetch requires a real repository root: {root}")
    if git.origin_identity(root) != expected_identity:
        raise LabError("trusted fetch canonical repository identity mismatch")
    if git.current_branch(root) != branch:
        raise LabError("trusted fetch stable checkout is on the wrong branch")
    if git.status_short(root):
        raise LabError("trusted fetch refuses a dirty stable checkout")

    local_before = git.current_commit(root)
    tree_before = git.tree_oid(root, local_before)
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
        raise LabError(f"trusted fetch failed: {exc}") from exc
    if completed.returncode != 0:
        detail = " ".join((completed.stderr or completed.stdout).split())[:1000]
        raise LabError(
            f"trusted fetch failed without updating the checkout: "
            f"{detail or f'exit {completed.returncode}'}"
        )

    if (
        git.current_branch(root) != branch
        or git.current_commit(root) != local_before
        or git.tree_oid(root, local_before) != tree_before
        or git.status_short(root)
    ):
        raise LabError("trusted fetch unexpectedly changed the stable checkout")
    if not git.ref_exists(root, remote_ref):
        raise LabError("trusted fetch did not create the protected remote ref")
    remote_after = git.resolve_commit(root, remote_ref)
    receipt_content: dict[str, object] = {
        "schema": TRUSTED_FETCH_RECEIPT_SCHEMA_ID,
        "canonical_identity": expected_identity,
        "repository": str(root),
        "branch": branch,
        "remote_ref": remote_ref,
        "local_commit_unchanged": local_before,
        "local_tree_oid_unchanged": tree_before,
        "remote_commit_before": remote_before,
        "remote_commit_after": remote_after,
        "network_used": True,
        "source_objects_fetched": True,
        "checkout_updated": False,
        "submodules_fetched": False,
        "lfs_smudge_enabled": False,
        "large_assets_downloaded": False,
        "task_payload_authorized": False,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return {
        **receipt_content,
        "receipt_sha256": hashlib.sha256(
            _canonical_json(receipt_content)
        ).hexdigest(),
    }


def validate_trusted_fetch_receipt(receipt: object) -> dict[str, object]:
    if not isinstance(receipt, dict) or set(receipt) != {
        "schema",
        "canonical_identity",
        "repository",
        "branch",
        "remote_ref",
        "local_commit_unchanged",
        "local_tree_oid_unchanged",
        "remote_commit_before",
        "remote_commit_after",
        "network_used",
        "source_objects_fetched",
        "checkout_updated",
        "submodules_fetched",
        "lfs_smudge_enabled",
        "large_assets_downloaded",
        "task_payload_authorized",
        "fetched_at",
        "receipt_sha256",
    }:
        raise LabError("trusted fetch receipt has an unexpected shape")
    if receipt.get("schema") != TRUSTED_FETCH_RECEIPT_SCHEMA_ID:
        raise LabError("unsupported trusted fetch receipt schema")
    identity = receipt.get("canonical_identity")
    repository = receipt.get("repository")
    branch = receipt.get("branch")
    remote_ref = receipt.get("remote_ref")
    if (
        not isinstance(identity, str)
        or identity.count("/") != 1
        or not isinstance(repository, str)
        or not Path(repository).is_absolute()
        or not isinstance(branch, str)
        or not isinstance(remote_ref, str)
    ):
        raise LabError("trusted fetch receipt has invalid source identity")
    _safe_branch(branch)
    if remote_ref != f"refs/remotes/origin/{branch}":
        raise LabError("trusted fetch receipt has invalid remote ref")
    for field in (
        "local_commit_unchanged",
        "local_tree_oid_unchanged",
        "remote_commit_after",
    ):
        if not OID_PATTERN.fullmatch(str(receipt.get(field, ""))):
            raise LabError(f"trusted fetch receipt has invalid {field}")
    before = receipt.get("remote_commit_before")
    if before is not None and not OID_PATTERN.fullmatch(str(before)):
        raise LabError("trusted fetch receipt has invalid remote_commit_before")
    if (
        receipt.get("network_used") is not True
        or receipt.get("source_objects_fetched") is not True
        or receipt.get("checkout_updated") is not False
        or receipt.get("submodules_fetched") is not False
        or receipt.get("lfs_smudge_enabled") is not False
        or receipt.get("large_assets_downloaded") is not False
        or receipt.get("task_payload_authorized") is not False
    ):
        raise LabError("trusted fetch receipt has an unsafe authority envelope")
    fetched_at = receipt.get("fetched_at")
    if not isinstance(fetched_at, str):
        raise LabError("trusted fetch receipt has invalid timestamp")
    try:
        timestamp = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise LabError("trusted fetch receipt has invalid timestamp") from exc
    if timestamp.tzinfo is None:
        raise LabError("trusted fetch receipt timestamp requires a timezone")
    if not SHA256_PATTERN.fullmatch(str(receipt.get("receipt_sha256", ""))):
        raise LabError("trusted fetch receipt has invalid hash")
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if receipt["receipt_sha256"] != hashlib.sha256(_canonical_json(content)).hexdigest():
        raise LabError("trusted fetch receipt hash mismatch")
    return receipt

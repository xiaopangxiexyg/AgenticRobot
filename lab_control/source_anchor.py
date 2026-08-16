from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from . import git
from .remote_request import OID_PATTERN
from .workspace import LabError

SOURCE_ANCHOR_RECEIPT_SCHEMA_ID = "lab.source-anchor-receipt.v1"
DEFAULT_MAX_CHECKOUT_BLOB_BYTES = 16 * 1024 * 1024


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _run(arguments: list[str], *, environment: Mapping[str, str]) -> str:
    completed = subprocess.run(
        arguments,
        check=False,
        capture_output=True,
        text=True,
        env=dict(environment),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise LabError(f"source anchor Git operation failed: {detail}")
    return completed.stdout


def _trusted_git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "GIT_LFS_SKIP_SMUDGE": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _validate_checkout_blob_sizes(
    repository: Path,
    *,
    commit: str,
    max_blob_bytes: int,
) -> None:
    if not 1 <= max_blob_bytes <= DEFAULT_MAX_CHECKOUT_BLOB_BYTES:
        raise LabError("source anchor blob limit exceeds protected maximum")
    listing = git.run_git(
        repository,
        ["ls-tree", "-r", "-l", "--full-tree", commit],
    ).stdout
    for line in listing.splitlines():
        metadata, separator, path = line.partition("\t")
        fields = metadata.split()
        if not separator or len(fields) != 4:
            raise LabError("cannot parse source anchor Git tree")
        object_type = fields[1]
        size = fields[3]
        if object_type != "blob":
            continue
        if not size.isdigit():
            raise LabError(f"source anchor blob lacks size: {path}")
        if int(size) > max_blob_bytes:
            raise LabError(
                f"source anchor refuses large tracked blob: {path} "
                f"({size} > {max_blob_bytes} bytes)"
            )


def materialize_local_source_anchor(
    *,
    source_repository: Path,
    target: Path,
    expected_identity: str,
    commit: str,
    submodule_sources: Mapping[str, Path] | None = None,
    max_checkout_blob_bytes: int = DEFAULT_MAX_CHECKOUT_BLOB_BYTES,
) -> dict[str, object]:
    """Create an independent clean clone from already-present local Git objects.

    The operation never fetches and never runs repository hooks or LFS smudge.
    It is intentionally a library-only trusted preparation primitive; callers
    must separately authorize the target host write.
    """

    source_requested = source_repository.expanduser()
    source = source_requested.resolve()
    if (
        not source_requested.is_absolute()
        or source_requested.is_symlink()
        or source_requested.absolute() != source
        or git.repository_root(source) != source
    ):
        raise LabError(f"source anchor input is not a real repository root: {source}")
    if git.status_short(source):
        raise LabError(f"source anchor input is dirty: {source}")
    if git.origin_identity(source) != expected_identity:
        raise LabError("source anchor canonical repository identity mismatch")
    if not OID_PATTERN.fullmatch(commit) or git.resolve_commit(source, commit) != commit:
        raise LabError("source anchor requires an exact local commit")
    _validate_checkout_blob_sizes(
        source,
        commit=commit,
        max_blob_bytes=max_checkout_blob_bytes,
    )

    expected_submodules, initialized = git.submodule_status(source)
    if not initialized:
        raise LabError("source anchor input has uninitialized or drifting submodules")
    sources = dict(submodule_sources or {})
    if set(sources) != set(expected_submodules):
        raise LabError("source anchor requires exact local sources for all submodules")
    for relative_path, expected_commit in expected_submodules.items():
        candidate = sources[relative_path].expanduser()
        resolved = candidate.resolve()
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or candidate.absolute() != resolved
            or git.repository_root(resolved) != resolved
            or git.current_commit(resolved) != expected_commit
            or git.status_short(resolved)
        ):
            raise LabError(f"source anchor submodule source is unsafe: {relative_path}")

    target_requested = target.expanduser()
    if not target_requested.is_absolute() or ".." in target_requested.parts:
        raise LabError("source anchor target must be an absolute non-traversing path")
    if target_requested.exists() or target_requested.is_symlink():
        raise LabError(f"source anchor target already exists: {target_requested}")
    parent = target_requested.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent.absolute():
        raise LabError(f"source anchor target parent is missing or unsafe: {parent}")

    environment = _trusted_git_environment()
    _run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "clone",
            "--local",
            "--no-checkout",
            "--no-tags",
            "--",
            str(source),
            str(target_requested),
        ],
        environment=environment,
    )
    origin_url = git.run_git(source, ["remote", "get-url", "origin"]).stdout.strip()
    git.run_git(target_requested, ["remote", "set-url", "origin", origin_url])
    _run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "filter.lfs.smudge=",
            "-c",
            "filter.lfs.required=false",
            "-C",
            str(target_requested),
            "checkout",
            "--detach",
            commit,
        ],
        environment=environment,
    )
    if expected_submodules:
        path_to_name: dict[str, str] = {}
        mapping = git.run_git(
            target_requested,
            ["config", "-f", ".gitmodules", "--get-regexp", "^submodule\\..*\\.path$"],
        ).stdout
        for line in mapping.splitlines():
            key, separator, relative_path = line.partition(" ")
            if not separator or not key.startswith("submodule.") or not key.endswith(
                ".path"
            ):
                raise LabError("cannot parse source anchor submodule mapping")
            path_to_name[relative_path] = key[len("submodule.") : -len(".path")]
        if set(path_to_name) != set(expected_submodules):
            raise LabError("source anchor .gitmodules differs from expected submodules")
        for relative_path, source_path in sorted(sources.items()):
            git.run_git(
                target_requested,
                [
                    "config",
                    "--local",
                    f"submodule.{path_to_name[relative_path]}.url",
                    str(source_path.resolve()),
                ],
            )
        _run(
            [
                "git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "protocol.file.allow=always",
                "-c",
                "filter.lfs.smudge=",
                "-c",
                "filter.lfs.required=false",
                "-C",
                str(target_requested),
                "submodule",
                "update",
                "--init",
                "--checkout",
                "--no-fetch",
            ],
            environment=environment,
        )

    target_commit = git.current_commit(target_requested)
    target_tree = git.tree_oid(target_requested, target_commit)
    target_submodules, target_initialized = git.submodule_status(target_requested)
    if (
        git.origin_identity(target_requested) != expected_identity
        or target_commit != commit
        or target_submodules != expected_submodules
        or not target_initialized
        or git.status_short(target_requested)
    ):
        raise LabError("materialized source anchor failed final identity verification")
    content: dict[str, object] = {
        "schema": SOURCE_ANCHOR_RECEIPT_SCHEMA_ID,
        "canonical_identity": expected_identity,
        "source_repository": str(source),
        "target": str(target_requested),
        "commit": target_commit,
        "tree_oid": target_tree,
        "submodules": target_submodules,
        "clean": True,
        "detached": git.current_branch(target_requested) is None,
        "network_used": False,
        "lfs_smudge_enabled": False,
        "large_file_downloaded": False,
        "max_checkout_blob_bytes": max_checkout_blob_bytes,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "source_anchor_ready": True,
    }
    return {
        **content,
        "receipt_sha256": hashlib.sha256(_canonical_json(content)).hexdigest(),
    }

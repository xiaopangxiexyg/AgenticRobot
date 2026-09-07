#!/usr/bin/env python3
"""Claim a DexTeleop worktree and optionally attach development assets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from dev_asset_store import (
    AssetStoreError,
    _git_config_set,
    ensure_worktree_asset_link,
    inspect_asset_store,
    record_asset_root,
    require_available_store,
    resolve_asset_root,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _require_git_success(
    completed: subprocess.CompletedProcess[str],
    action: str,
) -> str:
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or action
        raise AssetStoreError(detail)
    return completed.stdout.strip()


def _branch_exists(repo_root: Path, branch: str) -> bool:
    completed = _run_git(
        repo_root, "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"
    )
    return completed.returncode == 0


def build_branch_description(
    *,
    owner: str,
    base_commit: str,
    write_scope: str,
    semantic_scope: str,
    excludes: str,
    dependencies: str,
    validation: str,
) -> str:
    values = {
        "owner": owner,
        "state": "active",
        "base": base_commit,
        "write_scope": write_scope,
        "semantic_scope": semantic_scope,
        "excludes": excludes,
        "dependencies": dependencies,
        "validation": validation,
    }
    for key, value in values.items():
        if not value.strip():
            raise AssetStoreError(f"branch description field {key!r} cannot be empty")
        if any(separator in value for separator in (";", "\r", "\n")):
            raise AssetStoreError(
                f"branch description field {key!r} cannot contain record separators"
            )
    return "; ".join(f"{key}={value.strip()}" for key, value in values.items())


def claim_worktree(args: argparse.Namespace) -> dict[str, object]:
    repo_root = args.repo_root.expanduser().resolve()
    worktree = args.path.expanduser()
    if not worktree.is_absolute():
        raise AssetStoreError("--path must be an absolute worktree path")
    worktree = worktree.resolve()
    if os.path.lexists(worktree):
        raise AssetStoreError(f"worktree path already exists: {worktree}")
    if args.branch in {"main", "master"}:
        raise AssetStoreError("refusing to claim a shared main branch")
    if not args.branch.startswith("task/"):
        raise AssetStoreError(
            "--branch must use the task/<owner>/<slug>-<date> namespace"
        )
    if _branch_exists(repo_root, args.branch):
        raise AssetStoreError(f"branch already exists: {args.branch}")

    base_commit = _require_git_success(
        _run_git(repo_root, "rev-parse", "--verify", f"{args.base}^{{commit}}"),
        f"cannot resolve base {args.base!r}",
    )
    asset_root, source = resolve_asset_root(
        repo_root,
        explicit_root=args.asset_root,
    )
    asset_status = inspect_asset_store(asset_root, source=source)
    canonical_asset_root = None
    if asset_status.available or args.asset_root is not None or args.require_assets:
        canonical_asset_root = require_available_store(asset_status)
    description = build_branch_description(
        owner=args.owner,
        base_commit=base_commit,
        write_scope=args.write_scope,
        semantic_scope=args.semantic_scope,
        excludes=args.excludes,
        dependencies=args.dependencies,
        validation=args.validation,
    )

    _require_git_success(
        _run_git(
            repo_root,
            "worktree",
            "add",
            "-b",
            args.branch,
            str(worktree),
            base_commit,
        ),
        "cannot create worktree",
    )
    try:
        _git_config_set(
            repo_root,
            f"branch.{args.branch}.description",
            description,
        )
        link_result = "not-requested-or-unavailable"
        if canonical_asset_root is not None:
            record_asset_root(repo_root, canonical_asset_root)
            link_result = ensure_worktree_asset_link(
                repo_root,
                worktree,
                canonical_asset_root,
            )
    except AssetStoreError as exc:
        raise AssetStoreError(
            f"worktree was created at {worktree}, but initialization was incomplete: {exc}; "
            "the tool did not remove or reset the claimed worktree"
        ) from exc

    return {
        "branch": args.branch,
        "worktree": str(worktree),
        "base_commit": base_commit,
        "description": description,
        "asset_root": str(canonical_asset_root) if canonical_asset_root else None,
        "asset_source": source,
        "asset_link_result": link_result,
        "asset_warning": None
        if canonical_asset_root
        else (
            asset_status.error
            or "Development assets unavailable; asset-dependent checks require setup."
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an isolated DexTeleop worktree and declare its ownership. "
            "Available development assets are linked automatically."
        )
    )
    parser.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS
    )
    parser.add_argument("--branch", required=True, help="new task branch")
    parser.add_argument(
        "--path", type=Path, required=True, help="absolute new worktree path"
    )
    parser.add_argument("--base", required=True, help="baseline branch, tag, or commit")
    parser.add_argument("--owner", required=True)
    parser.add_argument("--write-scope", required=True)
    parser.add_argument("--semantic-scope", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument(
        "--require-assets",
        action="store_true",
        help="fail before creation unless the selected asset store is complete",
    )
    parser.add_argument("--excludes", default="unrelated code and runtime artifacts")
    parser.add_argument("--dependencies", default="none")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload = claim_worktree(args)
    except AssetStoreError as exc:
        if args.json:
            print(
                json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True)
            )
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"ok": True, **payload}, indent=2, sort_keys=True))
    else:
        print(f"Claimed {payload['branch']} at {payload['worktree']}")
        print(f"Base: {payload['base_commit']}")
        print(f"Assets: {payload['asset_root']} ({payload['asset_link_result']})")
        if payload["asset_warning"]:
            print(f"WARNING: {payload['asset_warning']}")
        print("State: active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

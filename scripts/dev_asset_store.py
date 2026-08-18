#!/usr/bin/env python3
"""Configure and inspect DexTeleop's machine-local development asset store.

The store is intentionally outside Git.  A shared Git config value records its
location once per clone, while each linked worktree receives an ignored
``assets`` compatibility symlink.  This module never downloads, copies, edits,
or deletes asset content.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS_PATH = (
    REPO_ROOT / "robot_motion" / "configs" / "development_asset_requirements_v1.json"
)
ASSET_ROOT_ENV = "DEXTELEOP_ASSET_ROOT"
ASSET_ROOT_GIT_CONFIG = "dexteleop.assetRoot"


class AssetStoreError(RuntimeError):
    """Raised when an asset-store operation cannot be completed safely."""


@dataclass(frozen=True)
class BundleRequirement:
    bundle_id: str
    relative_path: str
    required_by_default: bool
    required_paths: tuple[str, ...]


@dataclass(frozen=True)
class BundleStatus:
    bundle_id: str
    relative_path: str
    required_by_default: bool
    available: bool
    missing_paths: tuple[str, ...]


@dataclass(frozen=True)
class StoreStatus:
    root: str | None
    source: str | None
    available: bool
    bundles: tuple[BundleStatus, ...]
    error: str | None = None


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )


def _git_config_get(repo_root: Path, key: str) -> str | None:
    completed = _run_git(repo_root, "config", "--local", "--get", key)
    if completed.returncode == 1:
        return None
    if completed.returncode != 0:
        raise AssetStoreError(
            completed.stderr.strip() or f"cannot read Git config {key!r}"
        )
    value = completed.stdout.strip()
    return value or None


def _git_config_set(repo_root: Path, key: str, value: str) -> None:
    completed = _run_git(repo_root, "config", "--local", key, value)
    if completed.returncode != 0:
        raise AssetStoreError(
            completed.stderr.strip() or f"cannot write Git config {key!r}"
        )


def record_asset_root(repo_root: Path, asset_root: Path) -> None:
    """Persist a validated asset-root binding in clone-shared local Git config."""

    _git_config_set(repo_root, ASSET_ROOT_GIT_CONFIG, str(asset_root.resolve()))


def load_requirements(path: Path = REQUIREMENTS_PATH) -> tuple[BundleRequirement, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetStoreError(f"cannot load development asset requirements: {exc}") from exc
    if payload.get("schema_version") != "dexteleop.development_asset_requirements.v1":
        raise AssetStoreError("unsupported development asset requirements schema")
    raw_bundles = payload.get("bundles")
    if not isinstance(raw_bundles, list) or not raw_bundles:
        raise AssetStoreError("development asset requirements contain no bundles")

    result: list[BundleRequirement] = []
    seen_ids: set[str] = set()
    for raw in raw_bundles:
        if not isinstance(raw, dict):
            raise AssetStoreError("development asset bundle must be an object")
        bundle_id = raw.get("bundle_id")
        relative_path = raw.get("relative_path")
        required_paths = raw.get("required_paths")
        required_by_default = raw.get("required_by_default")
        if not isinstance(bundle_id, str) or not bundle_id.strip():
            raise AssetStoreError("development asset bundle_id must be non-empty")
        if bundle_id in seen_ids:
            raise AssetStoreError(f"duplicate development asset bundle_id: {bundle_id}")
        if not isinstance(relative_path, str) or not relative_path.strip():
            raise AssetStoreError(f"bundle {bundle_id!r} has invalid relative_path")
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AssetStoreError(f"bundle {bundle_id!r} escapes the asset root")
        if not isinstance(required_by_default, bool):
            raise AssetStoreError(
                f"bundle {bundle_id!r} required_by_default must be boolean"
            )
        if (
            not isinstance(required_paths, list)
            or not required_paths
            or any(not isinstance(item, str) or not item for item in required_paths)
        ):
            raise AssetStoreError(f"bundle {bundle_id!r} has invalid required_paths")
        for item in required_paths:
            path_item = Path(item)
            if path_item.is_absolute() or ".." in path_item.parts:
                raise AssetStoreError(
                    f"bundle {bundle_id!r} required path escapes its bundle"
                )
        seen_ids.add(bundle_id)
        result.append(
            BundleRequirement(
                bundle_id=bundle_id,
                relative_path=relative_path,
                required_by_default=required_by_default,
                required_paths=tuple(required_paths),
            )
        )
    return tuple(result)


def resolve_asset_root(
    repo_root: Path,
    *,
    explicit_root: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path | None, str | None]:
    """Resolve a store with explicit > environment > Git config > local precedence."""

    environment = os.environ if environ is None else environ
    if explicit_root is not None:
        return Path(explicit_root).expanduser().resolve(), "explicit"
    env_root = environment.get(ASSET_ROOT_ENV, "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve(), f"env:{ASSET_ROOT_ENV}"
    configured_root = _git_config_get(repo_root, ASSET_ROOT_GIT_CONFIG)
    if configured_root:
        return Path(configured_root).expanduser().resolve(), f"git:{ASSET_ROOT_GIT_CONFIG}"
    compatibility_root = repo_root / "assets"
    if os.path.lexists(compatibility_root):
        return compatibility_root.resolve(), "worktree:assets"
    return None, None


def inspect_asset_store(
    root: Path | None,
    *,
    source: str | None,
    requirements: Sequence[BundleRequirement] | None = None,
) -> StoreStatus:
    selected = tuple(requirements) if requirements is not None else load_requirements()
    if root is None:
        return StoreStatus(
            root=None,
            source=source,
            available=False,
            bundles=(),
            error=(
                f"no development asset store is configured; set {ASSET_ROOT_ENV} or run "
                "'python scripts/dev_asset_store.py configure --asset-root /path/to/assets'"
            ),
        )
    if not root.is_dir():
        return StoreStatus(
            root=str(root),
            source=source,
            available=False,
            bundles=(),
            error=f"development asset root is not a directory: {root}",
        )

    bundles: list[BundleStatus] = []
    for requirement in selected:
        bundle_root = root / requirement.relative_path
        missing = tuple(
            relative
            for relative in requirement.required_paths
            if not (bundle_root / relative).exists()
        )
        bundles.append(
            BundleStatus(
                bundle_id=requirement.bundle_id,
                relative_path=requirement.relative_path,
                required_by_default=requirement.required_by_default,
                available=not missing,
                missing_paths=missing,
            )
        )
    required = tuple(item for item in bundles if item.required_by_default)
    return StoreStatus(
        root=str(root),
        source=source,
        available=bool(required) and all(item.available for item in required),
        bundles=tuple(bundles),
    )


def require_available_store(status: StoreStatus) -> Path:
    if status.error:
        raise AssetStoreError(status.error)
    unavailable = [
        item
        for item in status.bundles
        if item.required_by_default and not item.available
    ]
    if unavailable:
        details = "; ".join(
            f"{item.bundle_id}: {', '.join(item.missing_paths)}" for item in unavailable
        )
        raise AssetStoreError(f"development asset store is incomplete: {details}")
    if not status.available or status.root is None:
        raise AssetStoreError("development asset store is unavailable")
    return Path(status.root)


def _resolved_git_common_dir(repo_root: Path) -> Path:
    completed = _run_git(repo_root, "rev-parse", "--git-common-dir")
    if completed.returncode != 0:
        raise AssetStoreError(completed.stderr.strip() or f"not a Git worktree: {repo_root}")
    raw = Path(completed.stdout.strip())
    return (repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()


def require_same_repository_worktree(repo_root: Path, worktree: Path) -> Path:
    candidate = worktree.expanduser().resolve()
    completed = _run_git(candidate, "rev-parse", "--show-toplevel")
    if completed.returncode != 0:
        raise AssetStoreError(
            completed.stderr.strip() or f"not a registered Git worktree: {candidate}"
        )
    top_level = Path(completed.stdout.strip()).resolve()
    if top_level != candidate:
        raise AssetStoreError(
            f"target must be the worktree root: expected {top_level}, got {candidate}"
        )
    if _resolved_git_common_dir(repo_root) != _resolved_git_common_dir(candidate):
        raise AssetStoreError(f"target belongs to a different Git repository: {candidate}")
    return candidate


def ensure_worktree_asset_link(
    repo_root: Path,
    worktree: Path,
    asset_root: Path,
) -> str:
    """Create an ignored compatibility link without replacing any existing path."""

    target_worktree = require_same_repository_worktree(repo_root, worktree)
    canonical_root = asset_root.expanduser().resolve()
    target = target_worktree / "assets"
    if os.path.lexists(target):
        try:
            existing_root = target.resolve(strict=True)
        except OSError as exc:
            raise AssetStoreError(f"existing worktree asset path is unusable: {target}: {exc}") from exc
        if existing_root != canonical_root:
            raise AssetStoreError(
                f"refusing to replace existing worktree asset path {target}; it resolves to "
                f"{existing_root}, expected {canonical_root}"
            )
        return "already-linked" if target.is_symlink() else "canonical-directory"
    target.symlink_to(canonical_root, target_is_directory=True)
    return "linked"


def status_payload(status: StoreStatus) -> dict[str, object]:
    payload = asdict(status)
    payload["config_key"] = ASSET_ROOT_GIT_CONFIG
    payload["environment_variable"] = ASSET_ROOT_ENV
    payload["requirements_path"] = str(REQUIREMENTS_PATH)
    return payload


def configure_store(repo_root: Path, asset_root: Path, *, link_current: bool) -> dict[str, object]:
    status = inspect_asset_store(asset_root.resolve(), source="explicit")
    canonical_root = require_available_store(status)
    record_asset_root(repo_root, canonical_root)
    link_result = None
    if link_current:
        link_result = ensure_worktree_asset_link(repo_root, repo_root, canonical_root)
    return {
        "configured": True,
        "root": str(canonical_root),
        "config_key": ASSET_ROOT_GIT_CONFIG,
        "worktree": str(repo_root.resolve()),
        "link_result": link_result,
    }


def _print_payload(payload: Mapping[str, object], *, emit_json: bool) -> None:
    if emit_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key, value in payload.items():
        if key == "bundles" and isinstance(value, list):
            print("bundles:")
            for item in value:
                marker = "ready" if item["available"] else "missing"
                required = "required" if item["required_by_default"] else "optional"
                print(f"  [{marker}] {item['bundle_id']} ({required})")
            continue
        print(f"{key}: {value}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manage the clone-shared, machine-local DexTeleop development asset store."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=argparse.SUPPRESS,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="inspect asset discovery and bundles")
    status_parser.add_argument("--asset-root", type=Path)
    status_parser.add_argument("--json", action="store_true")

    configure_parser = subparsers.add_parser(
        "configure", help="validate and record the canonical asset root for this clone"
    )
    configure_parser.add_argument("--asset-root", type=Path, required=True)
    configure_parser.add_argument(
        "--no-link-current",
        action="store_true",
        help="record the store without creating/checking this worktree's assets path",
    )
    configure_parser.add_argument("--json", action="store_true")

    link_parser = subparsers.add_parser(
        "link-worktree", help="create the ignored assets compatibility link in one worktree"
    )
    link_parser.add_argument("--worktree", type=Path, required=True)
    link_parser.add_argument("--asset-root", type=Path)
    link_parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        if args.command == "status":
            root, source = resolve_asset_root(repo_root, explicit_root=args.asset_root)
            status = inspect_asset_store(root, source=source)
            _print_payload(status_payload(status), emit_json=args.json)
            return 0 if status.available else 1
        if args.command == "configure":
            payload = configure_store(
                repo_root,
                args.asset_root,
                link_current=not args.no_link_current,
            )
            _print_payload(payload, emit_json=args.json)
            return 0
        if args.command == "link-worktree":
            root, source = resolve_asset_root(repo_root, explicit_root=args.asset_root)
            status = inspect_asset_store(root, source=source)
            asset_root = require_available_store(status)
            result = ensure_worktree_asset_link(repo_root, args.worktree, asset_root)
            _print_payload(
                {
                    "worktree": str(args.worktree.expanduser().resolve()),
                    "root": str(asset_root),
                    "source": source,
                    "link_result": result,
                },
                emit_json=args.json,
            )
            return 0
    except AssetStoreError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

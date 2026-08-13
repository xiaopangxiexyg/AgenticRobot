#!/usr/bin/env python3
"""Safely remove ``__pycache__`` directories contained in this repository."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import termios
import tty
from pathlib import Path
from typing import Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_REPOSITORY_MARKERS = (
    "AGENTS.md",
    "pyproject.toml",
    "robot_motion",
    "teleop_runtime",
)
_CACHE_DIRECTORY_NAME = "__pycache__"


class CleanupSafetyError(RuntimeError):
    """Raised when a cleanup target does not satisfy the safety contract."""


def _validate_repository_root(repository_root: Path) -> Path:
    root = repository_root.resolve(strict=True)
    if not root.is_dir():
        raise CleanupSafetyError(f"repository root is not a directory: {root}")

    missing = [name for name in _REPOSITORY_MARKERS if not (root / name).exists()]
    if missing:
        raise CleanupSafetyError(
            f"refusing cleanup because repository markers are missing: {missing}"
        )
    return root


def find_pycache_directories(repository_root: Path) -> tuple[Path, ...]:
    """Return safe, real ``__pycache__`` directories below *repository_root*.

    Directory symlinks, nested repositories, and nested mount points are
    pruned before traversal. A symlink named ``__pycache__`` is therefore
    reported by neither discovery nor deletion.
    """

    root = _validate_repository_root(repository_root)
    root_device = root.stat().st_dev
    candidates: list[Path] = []

    def fail_on_walk_error(error: OSError) -> None:
        raise CleanupSafetyError(f"repository traversal failed: {error}") from error

    for current, directory_names, _file_names in os.walk(
        root,
        topdown=True,
        onerror=fail_on_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        if current_path != root and (current_path / ".git").exists():
            directory_names[:] = []
            continue
        retained_directories: list[str] = []
        for name in directory_names:
            child = current_path / name
            if name == ".git" or child.is_symlink():
                continue
            if child.stat(follow_symlinks=False).st_dev != root_device:
                continue
            if os.path.ismount(child):
                continue
            if name == _CACHE_DIRECTORY_NAME:
                resolved = child.resolve(strict=True)
                try:
                    resolved.relative_to(root)
                except ValueError as error:
                    raise CleanupSafetyError(
                        f"refusing out-of-repository cache target: {resolved}"
                    ) from error
                if resolved.name != _CACHE_DIRECTORY_NAME or not resolved.is_dir():
                    raise CleanupSafetyError(
                        f"refusing unexpected cache target: {resolved}"
                    )
                candidates.append(resolved)
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories

    return tuple(sorted(set(candidates)))


def remove_pycache_directories(
    repository_root: Path,
    *,
    dry_run: bool,
    expected_candidates: Sequence[Path] | None = None,
) -> tuple[Path, ...]:
    """Discover and optionally remove safe cache directories."""

    root = _validate_repository_root(repository_root)
    root_device = root.stat().st_dev
    candidates = find_pycache_directories(root)
    if expected_candidates is not None and candidates != tuple(expected_candidates):
        raise CleanupSafetyError(
            "cache candidates changed after preview; rerun the cleanup command"
        )
    if dry_run:
        return candidates

    removed: list[Path] = []
    for candidate in candidates:
        # Revalidate immediately before deletion so a changed target fails closed.
        if candidate.is_symlink() or candidate.name != _CACHE_DIRECTORY_NAME:
            raise CleanupSafetyError(f"refusing changed cache target: {candidate}")
        resolved = candidate.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise CleanupSafetyError(
                f"refusing out-of-repository cache target: {resolved}"
            ) from error
        if not resolved.is_dir():
            raise CleanupSafetyError(f"cache target is not a directory: {resolved}")
        if resolved.stat().st_dev != root_device:
            raise CleanupSafetyError(
                f"refusing cache target on another filesystem: {resolved}"
            )
        for parent in (resolved, *resolved.parents):
            if parent == root:
                break
            if os.path.ismount(parent):
                raise CleanupSafetyError(
                    f"refusing cache target below a nested mount: {resolved}"
                )
        shutil.rmtree(resolved)
        removed.append(resolved)
    return tuple(removed)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="list matching directories and exit without prompting",
    )
    mode.add_argument(
        "--confirm-clean",
        action="store_true",
        help="list and remove matching directories without an interactive prompt",
    )
    return parser


def _interactive_cleanup_confirmed() -> bool:
    print(
        "Press uppercase P to remove the listed directories; "
        "press any other key to cancel: ",
        end="",
        flush=True,
    )
    descriptor = sys.stdin.fileno()
    previous_attributes = termios.tcgetattr(descriptor)
    try:
        tty.setcbreak(descriptor)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(descriptor, termios.TCSADRAIN, previous_attributes)
        print()
    return key == "P"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        candidates = remove_pycache_directories(
            REPOSITORY_ROOT,
            dry_run=True,
        )
    except (CleanupSafetyError, FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    for path in candidates:
        print(f"would remove: {path.relative_to(REPOSITORY_ROOT)}")
    suffix = "directory" if len(candidates) == 1 else "directories"
    print(f"found {len(candidates)} removable __pycache__ {suffix}")

    if not candidates:
        return 0
    if args.dry_run:
        print("dry run complete; nothing was removed")
        return 0
    if not args.confirm_clean:
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            print(
                "cleanup canceled: interactive confirmation requires a terminal; "
                "use --confirm-clean to clean non-interactively"
            )
            return 0
        try:
            confirmed = _interactive_cleanup_confirmed()
        except KeyboardInterrupt:
            print("\ncleanup canceled; nothing was removed")
            return 130
        if not confirmed:
            print("cleanup canceled; nothing was removed")
            return 0

    try:
        removed = remove_pycache_directories(
            REPOSITORY_ROOT,
            dry_run=False,
            expected_candidates=candidates,
        )
    except (CleanupSafetyError, FileNotFoundError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    removed_suffix = "directory" if len(removed) == 1 else "directories"
    print(f"removed {len(removed)} __pycache__ {removed_suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

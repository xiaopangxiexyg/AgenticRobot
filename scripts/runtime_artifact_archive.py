#!/usr/bin/env python3
"""Freeze complete legacy roots as a retained phase-A rollback archive.

This entrypoint is deliberately copy-only.  It cannot delete a source, switch
reader defaults, or replace an existing archive.  ``copy`` additionally
requires the explicit ``--apply-copy`` acknowledgement.

This is not the canonical-layout migration command.  After classifying one
closed semantic family, use ``scripts/runtime_artifact_migration.py`` for its
copy/verify publication into ``runtime_artifacts/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from runtime_artifact_migration import (  # noqa: E402
    MigrationSafetyError,
    MigrationSource,
    copy_sources_to_archive,
    inventory_sources,
    plan_archive_copy,
    verify_archive,
)


def _absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"path must be absolute: {value}")
    return path


def _source(value: str) -> MigrationSource:
    label, separator, root = value.partition("=")
    if not separator or not label or not root:
        raise argparse.ArgumentTypeError(
            "source must use LABEL=/absolute/path syntax"
        )
    try:
        return MigrationSource(label=label, root=_absolute_path(root))
    except (ValueError, OSError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        action="append",
        type=_source,
        required=True,
        metavar="LABEL=/ABSOLUTE/PATH",
        help="repeat once for each disjoint legacy artifact root",
    )


def _add_archive_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--migration-id", required=True)
    parser.add_argument("--archive-base", required=True, type=_absolute_path)
    parser.add_argument("--quiescence-sec", type=float, default=300.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory = subparsers.add_parser(
        "inventory",
        help="inspect source metadata without changing source or destination",
    )
    _add_sources(inventory)
    inventory.add_argument("--include-hashes", action="store_true")

    plan = subparsers.add_parser(
        "plan",
        help="run copy preflight without creating the archive",
    )
    _add_sources(plan)
    _add_archive_target(plan)

    copy = subparsers.add_parser(
        "copy",
        help="copy, hash-verify, and atomically publish a retained archive",
    )
    _add_sources(copy)
    _add_archive_target(copy)
    copy.add_argument(
        "--apply-copy",
        action="store_true",
        required=True,
        help="required acknowledgement; sources are still retained",
    )

    verify = subparsers.add_parser(
        "verify",
        help="read-only verification of a published archive",
    )
    verify.add_argument("--manifest", required=True, type=_absolute_path)
    verify.add_argument(
        "--skip-source-check",
        action="store_true",
        help="verify only the archive when original sources are unavailable",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    passed = True
    try:
        if args.command == "inventory":
            result = inventory_sources(
                args.source,
                include_hashes=bool(args.include_hashes),
            )
        elif args.command == "plan":
            result = plan_archive_copy(
                args.source,
                archive_base=args.archive_base,
                migration_id=args.migration_id,
                quiescence_sec=args.quiescence_sec,
            )
        elif args.command == "copy":
            manifest = copy_sources_to_archive(
                args.source,
                archive_base=args.archive_base,
                migration_id=args.migration_id,
                quiescence_sec=args.quiescence_sec,
            )
            verification = verify_archive(manifest, verify_sources=True)
            passed = bool(verification.get("passed"))
            result = {
                "copy_manifest": str(manifest),
                "source_deletion_permitted": False,
                "verification": verification,
            }
        elif args.command == "verify":
            result = verify_archive(
                args.manifest,
                verify_sources=not args.skip_source_check,
            )
            passed = bool(result.get("passed"))
        else:  # pragma: no cover - argparse owns the command choices.
            raise MigrationSafetyError(f"unsupported command: {args.command}")
    except (MigrationSafetyError, FileNotFoundError, FileExistsError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())

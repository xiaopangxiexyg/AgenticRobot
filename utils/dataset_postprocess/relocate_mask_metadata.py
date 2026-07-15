"""Safely rewrite absolute artifact roots recorded in mask JSON metadata.

Mask bitpack readers resolve their binary stream relative to each manifest, but
batch summaries and provenance fields keep absolute source/output paths.  This
tool updates those audit paths after an intentional directory relocation while
leaving all binary files untouched.  It is dry-run by default.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _parse_replacement(value: str) -> tuple[str, str]:
    try:
        source, destination = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--replace must be OLD=NEW") from exc
    source = source.rstrip("/")
    destination = destination.rstrip("/")
    if not source or not destination:
        raise argparse.ArgumentTypeError("--replace requires non-empty OLD and NEW paths")
    return source, destination


def _replace_path(value: str, replacements: tuple[tuple[str, str], ...]) -> tuple[str, bool]:
    for source, destination in replacements:
        if value == source:
            return destination, True
        prefix = f"{source}/"
        if value.startswith(prefix):
            return f"{destination}{value[len(source):]}", True
    return value, False


def _rewrite_value(value: Any, replacements: tuple[tuple[str, str], ...]) -> tuple[Any, int]:
    if isinstance(value, str):
        rewritten, changed = _replace_path(value, replacements)
        return rewritten, int(changed)
    if isinstance(value, list):
        rewritten: list[Any] = []
        count = 0
        for item in value:
            next_item, item_count = _rewrite_value(item, replacements)
            rewritten.append(next_item)
            count += item_count
        return rewritten, count
    if isinstance(value, dict):
        rewritten_dict: dict[str, Any] = {}
        count = 0
        for key, item in value.items():
            next_item, item_count = _rewrite_value(item, replacements)
            rewritten_dict[str(key)] = next_item
            count += item_count
        return rewritten_dict, count
    return value, 0


def _atomic_write(path: Path, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def rewrite_metadata_file(path: Path, replacements: tuple[tuple[str, str], ...], *, apply: bool) -> int:
    """Return replacement count for one JSON or JSONL metadata file."""

    original = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in original.splitlines() if line.strip()]
        rewritten, count = _rewrite_value(rows, replacements)
        if count and apply:
            _atomic_write(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rewritten))
        return count
    payload = json.loads(original)
    rewritten, count = _rewrite_value(payload, replacements)
    if count and apply:
        _atomic_write(path, json.dumps(rewritten, ensure_ascii=False, indent=2) + "\n")
    return count


def rewrite_metadata_tree(root: str | Path, replacements: list[tuple[str, str]], *, apply: bool) -> dict[str, int]:
    target = Path(root).expanduser().resolve()
    ordered = tuple(sorted(replacements, key=lambda item: len(item[0]), reverse=True))
    files = sorted([*target.rglob("*.json"), *target.rglob("*.jsonl")])
    changed_files = 0
    replacements_count = 0
    for path in files:
        count = rewrite_metadata_file(path, ordered, apply=apply)
        if count:
            changed_files += 1
            replacements_count += count
    return {
        "metadata_file_count": len(files),
        "changed_file_count": changed_files,
        "replacement_count": replacements_count,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Relocated mask root to scan recursively.")
    parser.add_argument("--replace", action="append", type=_parse_replacement, required=True, help="OLD=NEW path mapping. Repeatable.")
    parser.add_argument("--apply", action="store_true", help="Write changes atomically. Default is read-only dry-run.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = rewrite_metadata_tree(args.root, list(args.replace), apply=bool(args.apply))
    result.update({"root": str(args.root.expanduser().resolve()), "applied": bool(args.apply)})
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

